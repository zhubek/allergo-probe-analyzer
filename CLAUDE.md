# CLAUDE.md — context for Claude Code

This file orients a fresh Claude Code session working on this repo. Read it
first. It records the non-obvious decisions and findings so you don't have to
re-derive them.

## What this project is

An image-analysis tool for **allergo-probe microscopy images**: pale blue-green
fields stained with scattered **dark-blue dots**, photographed through a
microscope (circular black vignette around the field). Two jobs:

1. **Count all the dots** — every clearly-dark dark-blue dot in the field.
2. **Flag the "positive" objects** — the prominent **large dark-blue blobs**,
   which is what a human annotator circles in red on the source images.

It ships as a **FastAPI service** (`api.py`) plus standalone **CLI batch
scripts**. There is intentionally **no database** — the service is stateless
(image in → analysis out). See the README for the reasoning.

## Repo map

| File | Purpose |
|---|---|
| `allergo_core.py` | **Canonical** detection logic. `detect_dots(img)` and `detect_clusters(img)` take a PIL image, return dicts. The API uses this. |
| `api.py` | FastAPI app. `POST /analyze` (JSON of positive blobs), `POST /analyze/image` (labeled JPEG), `GET /health`. |
| `detect_dots.py` | CLI: detect every dot across a folder → per-image `*_dots.csv` + `*_overlay.jpg`. |
| `detect_clusters.py` | CLI: detect positive large blobs → `*_clusters.csv` + `*_clusters.jpg` + `clusters_summary.csv`. |
| `validate_clusters.py` | Measures detector recall against the human red-circle annotations. |
| `circled_dots_analysis.csv` | Per-annotation feature dump from the investigation. |
| `samples/` | Two **downscaled** (1200px) sample images for quick wiring tests. |
| `Dockerfile`, `.dockerignore` | Container build. |
| `docs/ALGORITHM.md` | How detection works + how parameters were calibrated. |
| `docs/ANALYSIS.md` | The investigation: what the red circles actually mark. |

> The CLI scripts (`detect_dots.py`, `detect_clusters.py`) currently carry their
> own copy of the detection constants/logic, predating `allergo_core.py`. They
> are verified and produce identical numbers, but if you change the algorithm,
> change `allergo_core.py` and consider refactoring the CLIs to import from it.

## Critical findings (don't re-derive these)

1. **Detection rule for a dot:** a pixel is a dark-blue dot if it is *darker than
   its local smoothed bright field* (`field - luma > 70`, field = gaussian σ=60),
   *inside the bright area* (`field > 120`, which excludes the vignette ring),
   and *blue* (`B - R > 25`, which also rejects brown/orange artefacts). An
   earlier absolute-luma cutoff failed because the vignette edge is also dark.

2. **What the red circles mark — IMPORTANT:** the human annotations circle the
   **largest dark-blue blobs** in each field (circled blobs sit at ~the 92nd
   size-percentile; 86% are in the top 10% largest of their image).
   It is **NOT** a "cluster of 3+ dots" rule — that hypothesis was tested and
   gave only ~3% recall; local dot density around circles is identical to
   background. **Size is the signal, not dot-count.** Full story in
   `docs/ANALYSIS.md`.

3. **Brown/orange objects are artefacts**, not targets, and are excluded by the
   blueness requirement.

## Environment & gotchas

- **Python venv required.** System pip is PEP-668 managed/blocked. Use a venv:
  `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
  Dependencies: fastapi, uvicorn, requests, python-multipart, numpy, scipy, pillow.
- **The source images are NOT in the repo** (full-res ~24 MB each, medical data,
  public repo). Only the 1200px `samples/` are committed.
- **Feed full-resolution images to the detectors.** Thresholds (esp.
  `MIN_BLOB_AREA=250`) are calibrated for the original ~5440px-wide images. On
  the downscaled samples blob counts run low / zero — that's expected, not a bug.
- **Full-res detection is slow** (σ=60 gaussian + watershed per image); batch
  runs take minutes. For parameter sweeps, extract dot centroids once and cluster
  on points (scipy cKDTree) instead of redoing image morphology per combo.
- The original images came from `снимки с разметками/` ("images with annotations"
  in Russian) and a `разметка.zip` / `снимки с разметками.zip`. These are
  gitignored.

## How to run

```bash
# local
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000   # docs at /docs

# docker
docker build -t allergo-probe-analyzer . && docker run --rm -p 8000:8000 allergo-probe-analyzer

# CLI over a folder of full-res PNGs
.venv/bin/python detect_dots.py --dir path/to/images
.venv/bin/python detect_clusters.py --dir path/to/images
```

## If you want to extend it

- **Tune sensitivity:** `MIN_BLOB_AREA` (positive cutoff), `DARK_DROP`,
  `BLUE_EXCESS` in `allergo_core.py`.
- **Add result storage:** that's the point to add Postgres + a `docker-compose.yml`
  (the user decided against a DB for now — confirm before adding).
- **More annotation data** would let you replace the size heuristic with a
  trained classifier; 35 circles was enough to find the rule but is a small set.
