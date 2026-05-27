#!/usr/bin/env python3
"""Detect dark-blue dots/clusters in microscopy images.

A dot is a region that is both clearly DARKER than its local background and
clearly BLUE (B >> R). Touching/overlapping dots are separated with a
distance-transform watershed so a cluster of N dots counts as N, not 1.

Outputs, per input image:
  <name>_dots.csv      -> id,x,y,area,radius,mean_blueness
  <name>_overlay.jpg   -> downscaled image with each detection circled

Usage:
  python detect_dots.py IMAGE [IMAGE ...]
  python detect_dots.py --dir FOLDER
"""
import sys, os, csv, glob, warnings
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

# watershed leaves a few empty label ids; center_of_mass on those divides by
# zero harmlessly (filtered out by area). Silence that specific warning.
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

# ---- tunable parameters -------------------------------------------------
# Target = the CLEARLY DARK navy dots only. A dot must drop well below the
# LOCAL bright field (DARK_DROP), the local field must itself be bright
# (FIELD_MIN, which excludes the dim vignette ring), and the dot must be
# bluish (BLUE_EXCESS). This ignores faint/light dots, big pale blobs, and
# artefacts. Calibrated on 20260429095155329 and 20260502161559484.
MIN_AREA      = 10     # px: ignore specks smaller than this (noise)
MAX_AREA      = 4000   # px: ignore large regions (pale blobs / artefacts)
DARK_DROP     = 70     # a dot must be this much darker than its local field
FIELD_MIN     = 120    # local field must be at least this bright (excl. vignette)
FIELD_SIGMA   = 60     # px: scale of the smooth bright-field estimate
BLUE_EXCESS   = 25     # and bluish: how much B must exceed R
MIN_PEAK_DIST = 4      # px: min separation between two dot centers when splitting
                       #     touching dots (smaller -> splits more aggressively)
OVERLAY_MAXDIM= 1800   # px: longest side of the saved overlay image
# -------------------------------------------------------------------------

def detect(path):
    img = Image.open(path).convert("RGB")
    a = np.asarray(img).astype(np.float32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]

    luma = 0.299 * R + 0.587 * G + 0.114 * B
    blueness = B - R  # blue channel relative to red (navy dots: B >> R)

    # smooth model of the bright field; the vignette makes this dim near the edge
    field = ndimage.gaussian_filter(luma, FIELD_SIGMA)

    # a dot = clearly darker than its LOCAL field, inside the bright area, blue.
    #   (field > FIELD_MIN) drops the dim vignette ring
    #   (field - luma > DARK_DROP) keeps only clearly-dark dots, not faint ones
    mask = (field > FIELD_MIN) & ((field - luma) > DARK_DROP) & (blueness > BLUE_EXCESS)

    # 5) split touching/overlapping dots via a distance-transform watershed.
    #    Each dot is a local maximum of the distance-to-background; we seed one
    #    marker per peak and let the watershed cut merged blobs along the valleys.
    lbl = split_touching(mask)
    n = int(lbl.max())
    if n == 0:
        return img, []
    areas = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    centroids = ndimage.center_of_mass(np.ones_like(lbl), lbl, range(1, n + 1))
    mean_blue = ndimage.mean(blueness, lbl, range(1, n + 1))

    dots = []
    for i in range(n):
        area = areas[i]
        if area < MIN_AREA or area > MAX_AREA:
            continue
        cy, cx = centroids[i]
        dots.append({
            "x": round(cx, 1), "y": round(cy, 1),
            "area": int(area),
            "radius": round((area / np.pi) ** 0.5, 1),
            "mean_blueness": round(float(mean_blue[i]), 1),
        })
    return img, dots


def split_touching(mask):
    """Label `mask` so each dot gets its own id, splitting merged blobs.

    Distance transform -> local-maxima markers (>= MIN_PEAK_DIST apart) ->
    watershed_ift over the inverted distance surface. Returns an int label
    image (0 = background).
    """
    dist = ndimage.distance_transform_edt(mask)
    # light smoothing turns each dot in a cluster into a distinct rounded peak,
    # so two touching dots yield two maxima instead of one merged plateau.
    dist = ndimage.gaussian_filter(dist, 1.0)

    # local maxima of the distance map = dot centers. A pixel is a peak if it
    # equals the max within a (2*MIN_PEAK_DIST+1) neighborhood.
    fp = np.ones((2 * MIN_PEAK_DIST + 1,) * 2, dtype=bool)
    localmax = (ndimage.maximum_filter(dist, footprint=fp) == dist) & (dist > 0)

    # merge adjacent peak pixels into single markers
    markers, _ = ndimage.label(localmax)
    if markers.max() == 0:
        out, _ = ndimage.label(mask)
        return out

    # watershed_ift needs uint8/uint16 input; flood from markers across the
    # inverted (so centers are basins) distance surface, confined to the mask.
    surface = (dist.max() - dist)
    surface = (surface / (surface.max() + 1e-9) * 255).astype(np.uint8)
    ws = ndimage.watershed_ift(surface, markers.astype(np.int32))
    ws[~mask] = 0
    return ws

def save_outputs(path, img, dots):
    base = os.path.splitext(path)[0]
    # CSV
    with open(base + "_dots.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "area", "radius", "mean_blueness"])
        for i, d in enumerate(dots, 1):
            w.writerow([i, d["x"], d["y"], d["area"], d["radius"], d["mean_blueness"]])
    # overlay (downscaled for size)
    W, H = img.size
    scale = min(1.0, OVERLAY_MAXDIM / max(W, H))
    ov = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS).convert("RGB")
    dr = ImageDraw.Draw(ov)
    for d in dots:
        x, y = d["x"] * scale, d["y"] * scale
        r = max(3, d["radius"] * scale + 2)
        dr.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=2)
    ov.save(base + "_overlay.jpg", "JPEG", quality=85)
    return base

def main(argv):
    if "--dir" in argv:
        d = argv[argv.index("--dir") + 1]
        paths = sorted(glob.glob(os.path.join(d, "*.png")) +
                       glob.glob(os.path.join(d, "*.jpg")))
        # skip our own generated overlays so re-running a folder is idempotent
        paths = [p for p in paths if not p.endswith("_overlay.jpg")]
    else:
        paths = [p for p in argv[1:] if not p.startswith("--")]
    for p in paths:
        img, dots = detect(p)
        base = save_outputs(p, img, dots)
        print(f"{os.path.basename(p)}: {len(dots)} dots -> {base}_dots.csv, {base}_overlay.jpg")

if __name__ == "__main__":
    main(sys.argv)
