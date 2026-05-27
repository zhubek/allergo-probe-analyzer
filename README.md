# Allergo Probe Analyzer — API

Detects dark-blue dots in microscopy images and flags the "positive" objects
(the large dark-blue blobs that match the human red-circle annotations).

## What gets detected

- **Dots** — every clearly-dark dark-blue dot (dark relative to the local bright
  field, blue, not warm/brown). Touching dots are split via watershed.
- **Positives ("points like this")** — the large dark-blue blobs. Inferred from
  the human annotations: the circles mark the *largest* dark-blue blobs in each
  field (validated at 86% recall vs the 35 hand-drawn circles). It is **not** a
  "3+ separate dots" rule — see the analysis notes below.

The detection logic lives in `allergo_core.py` (shared with the CLI scripts
`detect_dots.py` and `detect_clusters.py`).

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

## Endpoints

### `GET /health`
Liveness check → `{"status": "ok"}`

### `POST /analyze` — JSON of positive blobs
Returns the positive large dark-blue blobs (count + locations).

```json
{
  "width": 5440,
  "height": 3648,
  "count": 5,
  "has_points": true,
  "points": [
    {"x": 4067.4, "y": 539.6, "area": 1037, "w": 52, "h": 41}
  ]
}
```
`x,y` = blob centre (full-resolution pixels); `area` = blob area in px;
`w,h` = bounding-box size.

### `POST /analyze/image` — labeled image
Returns a JPEG (`image/jpeg`, downscaled to 1800px longest side):
- **all dots** → small yellow circles
- **positive blobs** → red boxes

Counts are also returned in response headers `X-Dot-Count` and `X-Positive-Count`.

## Image input (both POST endpoints)

Any one of:

| Mode | How |
|---|---|
| File upload | `multipart/form-data`, field `file` |
| Raw bytes | body = image bytes, `Content-Type: image/png` (or jpeg) |
| URL | JSON `{"url": "https://..."}` — server downloads it |
| Base64 | JSON `{"image_b64": "..."}` |

Max image size 60 MB.

## Examples

Two downsized sample microscopy images are in [`samples/`](samples/) so you can
try the API immediately (`sample_sparse.jpg`, `sample_dense.jpg`).

```bash
# JSON analysis, sample file upload
curl -F "file=@samples/sample_sparse.jpg" http://localhost:8000/analyze

# JSON analysis, file upload (any image)
curl -F "file=@image.png" http://localhost:8000/analyze

# JSON analysis, image URL
curl -H "Content-Type: application/json" \
     -d '{"url":"https://example.com/image.png"}' \
     http://localhost:8000/analyze

# Labeled image, raw bytes -> save JPEG
curl --data-binary @image.png -H "Content-Type: image/png" \
     http://localhost:8000/analyze/image -o labeled.jpg
```

## Tuning

In `allergo_core.py`:
- `MIN_BLOB_AREA` (default 250) — size cutoff for a "positive" blob. Raise to
  mark only the very largest; lower to include medium blobs.
- `DARK_DROP` / `BLUE_EXCESS` — how dark / how blue a pixel must be to count.
