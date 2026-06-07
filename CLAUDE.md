# CLAUDE.md — context for Claude Code

Orientation for a fresh Claude Code session on this repo. Read first.
Records non-obvious decisions so you don't re-derive them.

## What this project is

Image-analysis tool for **allergo-probe microscopy images**: pale blue-green
fields stained with scattered dark-blue dots, photographed through a microscope
(circular vignette around the field). Goal: detect "positive" dark-blue blobs —
the ones a human expert would mark.

Ships as a **FastAPI service** (`api.py`). Stateless — image in, JSON or
labeled JPEG out. No database. The trained model lives at
`models/classifier.joblib`.

## Repo map

| Path | Purpose |
|---|---|
| `allergo_core.py` | Canonical detection. `detect_clusters(img)` routes through the classifier when present, falls back to thresholds. `detect_dots(img)` is the original dot finder (still threshold-based). |
| `api.py` | FastAPI app — `GET /health`, `GET /model`, `POST /analyze`, `POST /analyze/image`. |
| `data/labels.db` | SQLite ground truth (357 images, 310 bbox annotations, single class `cluster`). |
| `data/images/` | Full-res PNGs (gitignored, medical data). |
| `models/classifier.joblib` | **The trained model.** LogisticRegression on 19 features. Auto-loaded by `allergo_core` at import. |
| `models/thresholds.json` | Tuned threshold fallback (used if classifier missing). |
| `finetune/` | Fine-tuning scripts (features, training, evaluation). See `finetune/README.md`. |
| `samples/` | Two 1200px sample images for quick wiring tests. |
| `Dockerfile`, `.dockerignore` | Container build (copies `models/`, `finetune/`, app code — NOT `data/`). |
| `docs/DEPLOYMENT.md` | Cross-platform deploy guide. |
| `docs/USAGE_RU.md` | Russian usage instructions. |
| `detect_dots.py`, `detect_clusters.py`, `validate_clusters.py` | **Legacy** CLI scripts predating the classifier. Carry their own copy of constants. Still work for batch processing but bypass the classifier. |

## Critical findings (don't re-derive)

1. **Three-layer config precedence** in `allergo_core.py`:
   classifier.joblib → thresholds.json → hardcoded defaults. The classifier
   wins when present.

2. **Detection in classifier mode**: `_detect_clusters_classifier` runs a
   **relaxed** dark-blue mask (much more permissive than the original) to
   propose candidates, then scores each with the classifier. Plus post-hoc
   hard caps: reject if `aspect_ratio > 3.5` OR (`area_zscore > 9` AND
   `aspect_ratio > 1.8`). The conjunction catches streaks without losing
   round positives in dense fields.

3. **19 features** in `finetune/features.py`. Specifically excluded:
   - **darkness features** (`mean_darkness`, `std_darkness`, etc.) — they
     correlated with the signal but rewarded "extra-dark" artefacts. Color
     identity (mean_R/G/B + mean_blueness) does the same job without that
     side effect.
   - **multiple correlated size/ratio features** — kept one scale-invariant
     `area_zscore` instead of `area_ratio`, `area_zscore`, `abs_area_zscore`.
   The clean 19-feature set has interpretable coefficients (no
   multicollinearity).

4. **Decision threshold default = 0.70**. Sweet spot from the precision/recall
   curve: ~83% recall, ~34 predictions/image. Tunable per-request via
   `?threshold=` query param.

5. **Test ROC-AUC ~0.90, no overfitting** (CV ROC-AUC ≈ test ROC-AUC; honest
   80/20 train/test split with `seed=42`). Test recall ~77%.

6. **Expert labels are sparse.** Humans average ~1.8 marked positives per
   image but plenty more real positives exist unmarked. "False positives" by
   our matcher are *partially* model predictions on unlabeled-but-real blobs.
   Treat **recall** as the meaningful metric, not precision.

7. **Brown/orange objects are artefacts**, excluded by the blue-channel test
   in the relaxed mask + color features in the classifier.

## Environment & gotchas

- **Python venv required.** System pip is PEP-668 managed/blocked on most
  systems. Use a venv:
  `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
  Dependencies in `requirements.txt`: fastapi, uvicorn, numpy, scipy, pillow,
  scikit-learn (1.5.2 pinned — 1.9.x fails on Windows AppLocker), joblib.
- **Windows path quirk**: `.venv\Scripts\python.exe`, not `.venv/bin/python`.
- **Source images NOT in the repo** (medical data). Only the two 1200px
  `samples/` are committed.
- **Feed full-resolution images** (~5440 px wide) — the classifier's
  scale-invariant features rely on the per-image candidate distribution
  being representative; very downscaled images change the distribution.
- **Full-res detection is CPU-heavy** (σ=60 Gaussian + sobel + per-blob
  feature extraction). ~5-10s per image on 1 core, ~3-5s with 4 workers.

## How to run

```bash
# local (Unix)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000     # docs at /docs

# local (Windows / PowerShell)
python -m venv .venv && .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000

# docker
docker build -t allergo-probe-analyzer .
docker run --rm -p 8000:8000 allergo-probe-analyzer

# fine-tuning (after putting images in data/images/)
python finetune/classify_train.py --workers 4    # ~10-15 min
python finetune/evaluate.py                       # check P/R/F1
```

## If you want to extend it

- **Tune at inference** (no retraining): `ALLERGO_SCORE_THRESHOLD`,
  `ALLERGO_MAX_ASPECT_RATIO`, `ALLERGO_MAX_AREA_ZSCORE` env vars or
  `?threshold=` query param.
- **Add features**: edit `finetune/features.py`, retrain via
  `finetune/classify_train.py`. Watch for multicollinearity with the existing
  set.
- **Add a class**: the trainer is single-class (treats `class_name = "cluster"`
  uniformly). For multi-class, you'd need per-class models or a multinomial
  classifier.
- **More annotation data** is the highest-leverage improvement. The biggest
  remaining false-positive failure modes (specific streak shapes, edge
  artefacts) need explicit negative examples to learn.

## Refactor notes (do this if you touch detection logic)

- The legacy `detect_dots.py`, `detect_clusters.py`, `validate_clusters.py`
  carry their own copies of detection constants — change them too if you
  change the algorithm, or refactor them to import from `allergo_core`.
- `allergo_core.detect_dots` still uses the threshold + watershed approach;
  only `detect_clusters` was migrated to the classifier. Dot counts (the
  yellow circles in the original `/analyze/image`) are no longer drawn by
  the API at all.
