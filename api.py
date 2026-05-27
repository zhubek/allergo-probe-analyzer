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
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageDraw, UnidentifiedImageError

import allergo_core as core

app = FastAPI(title="Allergo Probe Analyzer", version="1.0")

MAX_BYTES = 60 * 1024 * 1024     # 60 MB cap (source images are ~24 MB)
DOWNLOAD_TIMEOUT = 15            # seconds
OVERLAY_MAXDIM = 1800            # downscale the returned labeled image


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


@app.post("/analyze")
async def analyze(request: Request, file: UploadFile | None = File(default=None)):
    """Return the positive (large dark-blue blob) detections as JSON."""
    img = await load_image(request, file)
    clusters = core.detect_clusters(img)
    return JSONResponse({
        "width": img.width,
        "height": img.height,
        "count": len(clusters),
        "has_points": len(clusters) > 0,
        "points": clusters,            # each: {x, y, area, w, h}
    })


@app.post("/analyze/image")
async def analyze_image(request: Request, file: UploadFile | None = File(default=None)):
    """Return a JPEG: all dots circled (yellow) + positive blobs boxed (red)."""
    img = await load_image(request, file)
    dots = core.detect_dots(img)
    clusters = core.detect_clusters(img)

    s = min(1.0, OVERLAY_MAXDIM / max(img.width, img.height))
    ov = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    dr = ImageDraw.Draw(ov)
    for d in dots:                      # all dots: small yellow circles
        x, y = d["x"] * s, d["y"] * s
        r = max(2, d["radius"] * s)
        dr.ellipse([x - r, y - r, x + r, y + r], outline=(255, 215, 0), width=1)
    for c in clusters:                  # positives: red boxes
        x, y = c["x"] * s, c["y"] * s
        hw, hh = c["w"] * s / 2 + 4, c["h"] * s / 2 + 4
        dr.rectangle([x - hw, y - hh, x + hw, y + hh], outline=(255, 0, 0), width=2)

    buf = io.BytesIO()
    ov.save(buf, "JPEG", quality=85)
    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={
            "X-Dot-Count": str(len(dots)),
            "X-Positive-Count": str(len(clusters)),
        },
    )
