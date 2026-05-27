#!/usr/bin/env python3
"""Detect POSITIVE objects = the large, dark-blue blobs (prominent clumps).

Inferred from the human red-circle annotations and validated against them:
  * the annotator marks the LARGEST dark-blue blobs in each field
    (86% of circles fall on a top-10%-largest blob; median = 99th percentile)
  * it is NOT a "3+ separate dots" rule -- requiring >=2 split dots inside a
    cluster recovered <25% of the annotations; size is the real signal
  * brown / orange objects (warm hue) are excluded

Method
------
1. dark-blue blob mask (same dot criteria as detect_dots.py), warm hue removed
2. label blobs; keep those with area >= MIN_BLOB_AREA  (the "large" ones)
3. report one detection per kept blob: centroid, area, bbox

Outputs per image:
  <name>_clusters.csv   -> id,x,y,area,w,h
  <name>_clusters.jpg   -> overlay with each blob boxed (red)
plus a combined clusters_summary.csv with --dir.

Usage:
  python detect_clusters.py --dir FOLDER
  python detect_clusters.py IMAGE [IMAGE ...]
"""
import sys, os, csv, glob, warnings
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

warnings.filterwarnings("ignore", message="invalid value encountered in divide")

# ---- dark-blue blob criteria (from detect_dots.py) ----------------------
DARK_DROP   = 70    # a pixel must be this much darker than its local field
FIELD_MIN   = 120   # local field must be at least this bright (excl. vignette)
FIELD_SIGMA = 60    # px: scale of the smooth bright-field estimate
BLUE_EXCESS = 25    # bluishness: how much B must exceed R (also excludes warm/brown)
# ---- "large blob" rule --------------------------------------------------
# Median ordinary dot ~ 90 px; circled blobs ~ 350 px (p92 of size). A cutoff
# around 250 px keeps the prominent clumps and drops ordinary single dots.
MIN_BLOB_AREA = 250
MAX_BLOB_AREA = 8000   # ignore huge artefacts
OVERLAY_MAXDIM = 1800
# -------------------------------------------------------------------------


def detect_clusters(path):
    img = Image.open(path).convert("RGB")
    a = np.asarray(img).astype(np.float32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    luma = 0.299 * R + 0.587 * G + 0.114 * B
    field = ndimage.gaussian_filter(luma, FIELD_SIGMA)

    # dark-blue mask; (B - R) > BLUE_EXCESS also rejects warm/brown objects
    mask = (field > FIELD_MIN) & ((field - luma) > DARK_DROP) & ((B - R) > BLUE_EXCESS)

    lbl, n = ndimage.label(mask)
    clusters = []
    if n:
        areas = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        slices = ndimage.find_objects(lbl)
        cents = ndimage.center_of_mass(np.ones_like(lbl), lbl, range(1, n + 1))
        for i in range(n):
            if not (MIN_BLOB_AREA <= areas[i] <= MAX_BLOB_AREA):
                continue
            cy, cx = cents[i]; sl = slices[i]
            clusters.append({
                "x": round(float(cx), 1), "y": round(float(cy), 1),
                "area": int(areas[i]),
                "w": int(sl[1].stop - sl[1].start),
                "h": int(sl[0].stop - sl[0].start),
            })
    return img, clusters


def save_outputs(path, img, clusters):
    base = os.path.splitext(path)[0]
    with open(base + "_clusters.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "area", "w", "h"])
        for i, c in enumerate(clusters, 1):
            w.writerow([i, c["x"], c["y"], c["area"], c["w"], c["h"]])
    W, H = img.size
    s = min(1.0, OVERLAY_MAXDIM / max(W, H))
    ov = img.resize((int(W * s), int(H * s)), Image.LANCZOS).convert("RGB")
    dr = ImageDraw.Draw(ov)
    for c in clusters:
        x, y = c["x"] * s, c["y"] * s
        hw, hh = c["w"] * s / 2 + 4, c["h"] * s / 2 + 4
        dr.rectangle([x - hw, y - hh, x + hw, y + hh], outline=(255, 0, 0), width=2)
    ov.save(base + "_clusters.jpg", "JPEG", quality=85)
    return base


def main(argv):
    if "--dir" in argv:
        d = argv[argv.index("--dir") + 1]
        paths = sorted(glob.glob(os.path.join(d, "*.png")))
    else:
        paths = [p for p in argv[1:] if not p.startswith("--")]
    summary = []
    for p in paths:
        img, clusters = detect_clusters(p)
        base = save_outputs(p, img, clusters)
        summary.append((os.path.basename(p), len(clusters)))
        print(f"{os.path.basename(p)}: {len(clusters)} large blobs -> {base}_clusters.csv")
    if "--dir" in argv:
        d = argv[argv.index("--dir") + 1]
        with open(os.path.join(d, "clusters_summary.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["image", "n_clusters"])
            for name, n in summary:
                w.writerow([name, n])
        print("wrote clusters_summary.csv")


if __name__ == "__main__":
    main(sys.argv)
