# Инструкция: обучение и использование AllergoProbe

Короткое руководство: как обучить модель на ваших размеченных снимках и как пользоваться API.

## Что нужно

- **Python 3.10+** (рекомендуется 3.12). Проверка: `python --version`
- **Git**
- **Полноразмерные PNG-снимки** (~5440×3648 px) — для обучения
- Все команды ниже — для **PowerShell** (Windows). Для macOS/Linux замените `.venv\Scripts\` на `.venv/bin/`.

## Установка (один раз)

```powershell
git clone https://github.com/zhubek/allergo-probe-analyzer
cd allergo-probe-analyzer

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Если PowerShell ругается на запуск скрипта активации, выполните один раз:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

---

## Часть 1 — Обучение модели

### Шаг 1. Подготовить данные

1. Файл с разметкой `data/labels.db` уже в репозитории.
2. Положить полноразмерные PNG-снимки в папку `data/images/`. Имена должны совпадать с записями в БД.

Быстрая проверка покрытия:
```powershell
.venv\Scripts\python.exe -c "import sqlite3; c=sqlite3.connect('data/labels.db'); print('размеченных:', c.execute('SELECT COUNT(DISTINCT filename) FROM annotations').fetchone()[0])"
```

### Шаг 2. Запустить обучение

```powershell
.venv\Scripts\python.exe finetune\classify_train.py --workers 4
```

Что делает скрипт:
- из каждого снимка извлекает кандидаты-блобы (~700 на снимок)
- считает 23 признака на каждый блоб (размер, форма, цвет, текстура, позиция)
- случайно делит снимки на **80% обучение / 20% тест** (seed=42, воспроизводимо)
- обучает LogisticRegression
- проверяет качество через 5-fold кросс-валидацию + на отложенной тестовой выборке
- сохраняет модель в `models/classifier.joblib`

⏱️ **~10–15 минут** на 4 ядрах для 170 снимков.

### Шаг 3. Проверить, что модель обучилась

В конце вывода смотреть на **ROC-AUC**:

- `cv_roc_auc ≈ 0.90` — хорошо
- `test_roc_auc ≈ cv_roc_auc` — модель не переобучилась
- большой разрыв (test сильно хуже cv) — модель запомнила обучающие снимки, нужно проверить разметку

Проверить какая модель активна:
```powershell
.venv\Scripts\python.exe -c "import allergo_core as c, json; d=c.active_thresholds(); print('режим:', d['mode']); print('порог:', d.get('score_threshold'))"
```

Должно быть `режим: classifier`.

---

## Часть 2 — Использование API

### Шаг 1. Запустить сервер

```powershell
.venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Сервер слушает на `http://localhost:8000`. Остановить — `Ctrl+C`.

### Шаг 2. Открыть документацию в браузере

| URL | Что показывает |
|---|---|
| http://localhost:8000/docs | Интерактивные тесты эндпоинтов (Swagger) |
| http://localhost:8000/health | Жив ли сервер |
| http://localhost:8000/model | Какая модель активна и порог |

### Шаг 3. Два основных эндпоинта

**A) `POST /analyze` — JSON со списком найденных точек:**
```powershell
curl.exe -F "file=@C:\путь\к\снимку.png" http://localhost:8000/analyze
```

Ответ:
```json
{
  "width": 5440, "height": 3648,
  "count": 12, "has_points": true,
  "points": [
    {"x": 3409.0, "y": 627.9, "area": 432, "w": 21, "h": 24, "score": 0.867},
    ...
  ]
}
```

- `x, y` — координаты центра блоба
- `area`, `w`, `h` — площадь и размер рамки в пикселях
- `score` — уверенность модели (0–1)

**B) `POST /analyze/image` — картинка с разметкой:**
```powershell
curl.exe -F "file=@C:\путь\к\снимку.png" http://localhost:8000/analyze/image -o labeled.jpg
start labeled.jpg
```

- 🟡 жёлтые кружочки — все обнаруженные точки
- 🟥 красные рамки — положительные блобы (score ≥ порог)

### Шаг 4. Менять чувствительность без переобучения

Порог по умолчанию **0.70**. Можно изменить через переменную окружения:

```powershell
$env:ALLERGO_SCORE_THRESHOLD = "0.85"   # строже — меньше срабатываний
$env:ALLERGO_SCORE_THRESHOLD = "0.50"   # мягче — больше срабатываний
```

Потом перезапустить сервер. Ориентировочные значения:

| Порог | Recall (полнота) | Предсказаний на снимок |
|---|---|---|
| 0.50 | 93% | ~70 |
| 0.65 | 88% | ~41 |
| **0.70** ⭐ | **84%** | **~34** |
| 0.75 | 82% | ~28 |
| 0.85 | 72% | ~18 |

---

## Шпаргалка

```powershell
# Обучить модель
.venv\Scripts\python.exe finetune\classify_train.py --workers 4

# Оценить на тестовой выборке
.venv\Scripts\python.exe finetune\evaluate.py --split test --workers 4

# Подобрать оптимальный порог
.venv\Scripts\python.exe finetune\sweep_threshold.py --workers 4

# Запустить API
.venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000

# Использовать API
curl.exe -F "file=@image.png" http://localhost:8000/analyze
curl.exe -F "file=@image.png" http://localhost:8000/analyze/image -o labeled.jpg
```

## Откат к старой логике (порогам)

Если что-то пошло не так и нужна старая логика на порогах:

```powershell
del finetune\classifier.joblib
```

Перезапустить сервер. Система автоматически вернётся к режиму с порогами из `thresholds.json` или хардкод-настроек.
