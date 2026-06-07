# Allergo Probe Analyzer — API

Detects "positive" dark-blue blobs in microscopy images using a trained
classifier (sklearn LogisticRegression on 19 hand-engineered per-blob
features). Includes a fine-tuning pipeline so the model adapts to your own
labeled data.

## What gets detected

- **Positives** — significant dark-blue blobs that match the kind a human
  expert would mark. The classifier learns "what counts as positive" from
  `data/labels.db` (bbox annotations).
- **Score per prediction** — the classifier's confidence in 0–1. Tune via
  `?threshold=` query param or the `ALLERGO_SCORE_THRESHOLD` env var.

The detection logic lives in `allergo_core.py`; the classifier is
`models/classifier.joblib` (trained by `finetune/classify_train.py`).

## Documentation

- **`docs/DEPLOYMENT.md`** — cross-platform guide to running locally
  (Windows/macOS/Linux) and deploying to a server or container.
- **`docs/USAGE_RU.md`** — Краткая инструкция на русском: обучение модели и
  использование API.
- **`finetune/README.md`** — fine-tuning workflow: train your own classifier
  or threshold set from `data/labels.db`.
- **`CLAUDE.md`** — orientation for a Claude Code session (decisions, gotchas).

## Project layout

```
allergo_core.py        canonical detection (loads classifier or thresholds)
api.py                 FastAPI service
requirements.txt       pinned dependencies
Dockerfile             container build

data/                  inputs
  labels.db            ground truth (bbox annotations)
  images/              YOU put full-res PNGs here (gitignored)

models/                trained artifacts
  classifier.joblib    the trained model (committed)
  thresholds.json      fallback threshold config (committed)

finetune/              fine-tuning scripts (see finetune/README.md)
  features.py
  matching.py
  train.py             threshold grid search
  classify_train.py    classifier trainer
  evaluate.py          P/R/F1 evaluator
  sweep_threshold.py   score-threshold sweep

samples/               two 1200px sample images for quick wiring tests
docs/                  DEPLOYMENT.md, USAGE_RU.md
```

Source full-resolution images are not committed (medical data) — only the
2 downscaled samples are.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
```

Requires Python 3.10+ (3.12 recommended). On Windows use
`.venv\Scripts\python.exe` instead of `.venv/bin/...`.

Interactive docs at `http://localhost:8000/docs`.

### With Docker

```bash
docker build -t allergo-probe-analyzer .
docker run --rm -p 8000:8000 allergo-probe-analyzer
```

> **Feed full-resolution images** (~5440 px wide) for best results — the
> training data is full-res. The bundled `samples/` are downscaled to 1200 px
> for wiring tests, so their counts won't match what you'd see on real data.

## Endpoints

### `GET /health`
Liveness check → `{"status": "ok"}`

### `GET /model`
Reports active mode (`classifier` / `tuned` / `default`), source path, and
the metrics the classifier was validated against. Use this to verify the
trained model is loaded.

### `POST /analyze` — JSON of positive blobs

```bash
curl -F "file=@your-image.png" "http://localhost:8000/analyze?threshold=0.70"
```

Response:
```json
{
  "width": 5440, "height": 3648,
  "count": 5, "has_points": true,
  "points": [
    {"x": 4067.4, "y": 539.6, "area": 1037, "w": 52, "h": 41, "score": 0.934}
  ]
}
```
`x, y` = blob centre (full-resolution pixels); `area` = blob area in px;
`w, h` = bounding-box size; `score` = classifier confidence (0–1).

### `POST /analyze/image` — labeled JPEG

```bash
curl -F "file=@your-image.png" "http://localhost:8000/analyze/image?threshold=0.70" -o labeled.jpg
```

Returns a JPEG at the source resolution (quality 95) with red boxes around
each positive blob. The `X-Positive-Count` header gives the count.

## Image input modes (both POST endpoints)

| Mode | How |
|---|---|
| File upload | `multipart/form-data`, field `file` |
| Raw bytes | body = image bytes, `Content-Type: image/png` (or jpeg) |
| URL | JSON `{"url": "https://..."}` — server downloads it |
| Base64 | JSON `{"image_b64": "..."}` |

Max image size 60 MB.

## Tuning at inference (no retraining)

These env vars / query params change behavior without touching the model:

| Setting | Default | What it does |
|---|---|---|
| `?threshold=` / `ALLERGO_SCORE_THRESHOLD` | 0.70 | Classifier decision cutoff (0–1) |
| `ALLERGO_MAX_ASPECT_RATIO` | 3.5 | Reject blobs more elongated than this |
| `ALLERGO_MAX_AREA_ZSCORE` | 9.0 | Reject extreme-size blobs *if* aspect>1.8 (catches streaks, spares round positives) |
| `ALLERGO_CLASSIFIER` | `models/classifier.joblib` | Path to classifier; set to `NUL` (Windows) / `/dev/null` (Unix) to fall back to thresholds |
| `ALLERGO_THRESHOLDS` | `models/thresholds.json` | Same idea for threshold fallback |

## Examples

```bash
# JSON analysis on a sample (downscaled — counts won't be realistic)
curl -F "file=@samples/sample_sparse.jpg" http://localhost:8000/analyze

# JSON, full-res image, custom threshold
curl -F "file=@your-image.png" "http://localhost:8000/analyze?threshold=0.85"

# Labeled image from URL
curl -H "Content-Type: application/json" \
     -d '{"url":"https://example.com/img.png"}' \
     http://localhost:8000/analyze/image -o labeled.jpg
```

## Fine-tuning to your data

See [`finetune/README.md`](finetune/README.md) for the workflow. In short:

```bash
# put your full-res PNGs in data/images/  (filenames must match data/labels.db)
python finetune/classify_train.py --workers 4   # ~10-15 min
python finetune/evaluate.py                      # check P/R/F1
```

The trained classifier lands at `models/classifier.joblib`; `allergo_core`
auto-loads it.
