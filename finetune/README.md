# `finetune/` — fine-tuning scripts

Python scripts that learn from `data/labels.db` and produce trained artifacts
in `models/`. Two layered ways to adapt the detector to your data:

1. **Threshold tuning** (`train.py`) — grid-searches the 5 dark-blue-mask
   constants. No learning, just brute force. Output: `models/thresholds.json`.
2. **Feature classifier** (`classify_train.py`) — trains an sklearn
   `LogisticRegression` (default) or `RandomForestClassifier` on ~19 per-blob
   features (size, shape, color, position) using `data/labels.db` as ground
   truth. Output: `models/classifier.joblib`. **Takes precedence over
   `models/thresholds.json`** when present.

`allergo_core` auto-loads whichever artifact is on disk and falls back to
hardcoded defaults if neither is present. `GET /model` reports which is active.

## Repo layout

```
data/                  ← inputs
  labels.db            ground truth (357 images, 310 bbox annotations)
  images/              YOU put the full-res PNGs here (gitignored)

models/                ← outputs
  classifier.joblib    trained classifier (committed)
  thresholds.json      tuned thresholds fallback (committed)
  test_split.json      train/test image split (gitignored)
  grid_results.csv     all combos from train.py (gitignored)

finetune/              ← scripts (this folder)
  features.py          per-blob feature extractor (19 features)
  matching.py          TP/FP/FN matching against GT bboxes
  train.py             threshold grid search
  classify_train.py    classifier trainer
  evaluate.py          P/R/F1 of whichever mode is active
  sweep_threshold.py   classifier score-threshold sweep
```

Run all scripts from the **project root**, not from inside `finetune/`.

## Workflow — feature classifier (recommended)

```bash
# 0. Put full-res PNGs in data/images/  (filenames must match data/labels.db)

# 1. Baseline — score whatever's currently active
python finetune/evaluate.py

# 2. Train the classifier (~10-15 min on 4 workers)
python finetune/classify_train.py

# 3. Re-score with the new classifier
python finetune/evaluate.py

# 4. Optional: see precision/recall curve over decision thresholds
python finetune/sweep_threshold.py
```

What `classify_train.py` does internally:

1. Splits images **80/20 train/test** (deterministic, seed=42) so you get an
   honest held-out evaluation, not just CV.
2. Runs a **relaxed** dark-blue mask over each image to propose candidate blobs.
3. Labels each candidate as positive if its center is inside any GT bbox.
4. Extracts 19 features per blob (see `features.py:FEATURE_NAMES`).
5. Trains `LogisticRegression` with `class_weight='balanced'` (or
   `RandomForestClassifier` with `--model rf`).
6. Runs 5-fold stratified CV on train + measures the **held-out test set**.
7. Saves the fitted pipeline + metadata to `models/classifier.joblib`.

## Workflow — threshold tuning (light, no learning)

```bash
# Baseline
python finetune/evaluate.py

# Quick smoke test
python finetune/train.py --quick --limit 20

# Full grid (~30-60 min)
python finetune/train.py
```

## Tunable knobs at inference

These work without retraining — set env var or use the `?threshold=` query
param on the API:

| Env var | Default | Effect |
|---|---|---|
| `ALLERGO_SCORE_THRESHOLD` | 0.70 | Classifier decision cutoff (0–1). Higher = fewer, more confident |
| `ALLERGO_MAX_ASPECT_RATIO` | 3.5 | Reject blobs more elongated than this |
| `ALLERGO_MAX_AREA_ZSCORE` | 9.0 | Reject blobs with extreme size *if* also elongated (aspect > 1.8). Catches streaks without losing legitimate large round positives. |

## Checking what's active

```bash
curl http://localhost:8000/model
```

Returns mode (`classifier` / `tuned` / `default`), source path, and metrics.

## Reverting (and precedence)

Precedence: `models/classifier.joblib` > `models/thresholds.json` > hardcoded
defaults. To revert one layer:

```bash
# Disable the classifier
rm models/classifier.joblib
# Or temporarily:
export ALLERGO_CLASSIFIER=/dev/null    # macOS/Linux
$env:ALLERGO_CLASSIFIER = "NUL"        # PowerShell

# Drop tuned thresholds too:
rm models/thresholds.json
```

`allergo_core` re-checks both at process start. Confirm with `GET /model`.

## When to retrain

- You add a noticeable batch of new annotations to `data/labels.db`.
- You change input characteristics (different camera, different staining).
- You change features in `features.py`.

## Caveats

- **Single class right now.** `data/labels.db` has one class (`cluster`); the
  trainer ignores `class_name`. Add per-class models when more classes appear.
- **Recall vs precision.** Expert labels are sparse — many predictions the
  matcher calls "false positives" are actually real but un-annotated. Recall
  is the meaningful metric; precision is artificially low.
- **Center-in-box matching.** A prediction "hits" a GT box if its center is
  inside it. For tighter matching, swap in IoU > τ in `matching.py`.
