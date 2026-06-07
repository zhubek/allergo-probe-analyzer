"""Allergo probe image-analysis API.

Endpoints
---------
GET  /health                  -> liveness check
POST /analyze                 -> JSON: positive large dark-blue blobs (count + locations)
POST /analyze/image           -> JPEG: all dots marked + positive blobs highlighted

Image input (any one of these) on both POST endpoints:
  * multipart file upload   : field name "file"
  * raw image bytes         : Content-Type image/* with the bytes as the body
  * JSON {"url": "..."}     : server downloads the image
  * JSON {"image_b64": "..."}: base64-encoded image

Run:
  .venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
"""
import base64
import io

import requests
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageDraw, UnidentifiedImageError

import allergo_core as core

app = FastAPI(title="Allergo Probe Analyzer", version="1.0")

MAX_BYTES = 60 * 1024 * 1024     # 60 MB cap (source images are ~24 MB)
DOWNLOAD_TIMEOUT = 15            # seconds
OVERLAY_MAXDIM = 0               # 0 = no downscale (return labeled image at source resolution)
OVERLAY_JPEG_QUALITY = 95        # higher quality, larger file
THRESHOLD_QUERY = Query(None, ge=0.0, le=1.0,
                        description="Classifier decision threshold (0..1). When set, "
                                    "overrides ALLERGO_SCORE_THRESHOLD and the "
                                    "classifier.joblib default. Has no effect in "
                                    "threshold mode (no classifier loaded).")


async def load_image(request: Request, file: UploadFile | None) -> Image.Image:
    """Resolve an image from multipart file, raw body, JSON url, or JSON base64."""
    data: bytes | None = None

    if file is not None:                                  # multipart upload
        data = await file.read()
    else:
        ctype = request.headers.get("content-type", "")
        body = await request.body()
        if ctype.startswith("image/"):                    # raw image bytes
            data = body
        elif body:                                        # JSON: url or base64
            import json
            try:
                payload = json.loads(body)
            except ValueError:
                raise HTTPException(400, "Body is not valid JSON, raw image, or upload.")
            if "url" in payload:
                data = _download(payload["url"])
            elif "image_b64" in payload:
                try:
                    data = base64.b64decode(payload["image_b64"], validate=True)
                except Exception:
                    raise HTTPException(400, "image_b64 is not valid base64.")
            else:
                raise HTTPException(400, "JSON must contain 'url' or 'image_b64'.")

    if not data:
        raise HTTPException(400, "No image provided.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"Image exceeds {MAX_BYTES // (1024*1024)} MB limit.")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "Could not decode image data.")


def _download(url: str) -> bytes:
    if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
        raise HTTPException(400, "url must be an http(s) URL.")
    try:
        r = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
        r.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(502, f"Failed to download image: {e}")
    return r.content


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model():
    """Report which thresholds are active — defaults or a tuned thresholds.json."""
    return core.active_thresholds()


@app.post("/analyze")
async def analyze(request: Request,
                  file: UploadFile | None = File(default=None),
                  threshold: float | None = THRESHOLD_QUERY):
    """Return the positive (large dark-blue blob) detections as JSON."""
    img = await load_image(request, file)
    clusters = core.detect_clusters(img, score_threshold=threshold)
    return JSONResponse({
        "width": img.width,
        "height": img.height,
        "count": len(clusters),
        "has_points": len(clusters) > 0,
        "points": clusters,            # each: {x, y, area, w, h, score?}
    })


@app.post("/analyze/image")
async def analyze_image(request: Request,
                        file: UploadFile | None = File(default=None),
                        threshold: float | None = THRESHOLD_QUERY):
    """Return a JPEG with positive blobs marked as red boxes.

    Returned at source resolution (no downscale by default; see OVERLAY_MAXDIM).
    JPEG quality 95 to preserve detail.
    """
    img = await load_image(request, file)
    clusters = core.detect_clusters(img, score_threshold=threshold)

    if OVERLAY_MAXDIM and max(img.width, img.height) > OVERLAY_MAXDIM:
        s = OVERLAY_MAXDIM / max(img.width, img.height)
        ov = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    else:
        s = 1.0
        ov = img.copy()

    # Scale box stroke + padding so they're visible regardless of image size.
    dim = max(ov.width, ov.height)
    pad = max(8, dim // 300)            # ~18px at 5440, 4px at 1200
    stroke = max(2, dim // 700)         # ~7px at 5440, 2px at 1200

    dr = ImageDraw.Draw(ov)
    for c in clusters:
        x, y = c["x"] * s, c["y"] * s
        hw, hh = c["w"] * s / 2 + pad, c["h"] * s / 2 + pad
        dr.rectangle([x - hw, y - hh, x + hw, y + hh], outline=(255, 0, 0), width=stroke)

    buf = io.BytesIO()
    ov.save(buf, "JPEG", quality=OVERLAY_JPEG_QUALITY)
    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={"X-Positive-Count": str(len(clusters))},
    )
