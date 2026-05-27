"""Core detection for allergo_probe microscopy images.

Two detectors over the same dark-blue mask:
  * detect_dots()     -> every clearly-dark dark-blue dot (watershed-split)
  * detect_clusters()  -> "positive" large dark-blue blobs (the annotated objects)

Both take a PIL.Image (RGB) and return plain dicts, so the CLI scripts and the
API can share one implementation.
"""
import warnings
import numpy as np
from PIL import Image
from scipy import ndimage

warnings.filterwarnings("ignore", message="invalid value encountered in divide")

# ---- dark-blue criteria -------------------------------------------------
DARK_DROP    = 70    # a pixel must be this much darker than its local field
FIELD_MIN    = 120   # local field must be at least this bright (excl. vignette)
FIELD_SIGMA  = 60    # px: scale of the smooth bright-field estimate
BLUE_EXCESS  = 25    # bluishness B-R (also excludes warm/brown objects)
# ---- single-dot detection ----------------------------------------------
DOT_MIN_AREA = 10
DOT_MAX_AREA = 4000
MIN_PEAK_DIST = 4    # watershed: min separation between two dot centers
# ---- "positive" large-blob rule -----------------------------------------
MIN_BLOB_AREA = 250  # circled blobs sit at ~p92 of size; this keeps the large ones
MAX_BLOB_AREA = 8000
# -------------------------------------------------------------------------


def _channels(img):
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    luma = 0.299 * R + 0.587 * G + 0.114 * B
    return a, R, G, B, luma


def dark_blue_mask(luma, R, B):
    """Boolean mask of clearly-dark, blue (non-warm) pixels inside the field."""
    field = ndimage.gaussian_filter(luma, FIELD_SIGMA)
    return (field > FIELD_MIN) & ((field - luma) > DARK_DROP) & ((B - R) > BLUE_EXCESS)


def _split_touching(mask):
    """Distance-transform watershed so touching dots count separately."""
    dist = ndimage.gaussian_filter(ndimage.distance_transform_edt(mask), 1.0)
    fp = np.ones((2 * MIN_PEAK_DIST + 1,) * 2, dtype=bool)
    localmax = (ndimage.maximum_filter(dist, footprint=fp) == dist) & (dist > 0)
    markers, k = ndimage.label(localmax)
    if k == 0:
        out, _ = ndimage.label(mask)
        return out
    surface = (dist.max() - dist)
    surface = (surface / (surface.max() + 1e-9) * 255).astype(np.uint8)
    ws = ndimage.watershed_ift(surface, markers.astype(np.int32))
    ws[~mask] = 0
    return ws


def detect_dots(img):
    """Return list of {x,y,area,radius} for every clearly-dark dark-blue dot."""
    a, R, G, B, luma = _channels(img)
    mask = dark_blue_mask(luma, R, B)
    lbl = _split_touching(mask)
    n = int(lbl.max())
    if n == 0:
        return []
    areas = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    cents = ndimage.center_of_mass(np.ones_like(lbl), lbl, range(1, n + 1))
    dots = []
    for i in range(n):
        if not (DOT_MIN_AREA <= areas[i] <= DOT_MAX_AREA):
            continue
        cy, cx = cents[i]
        dots.append({"x": round(float(cx), 1), "y": round(float(cy), 1),
                     "area": int(areas[i]),
                     "radius": round(float((areas[i] / np.pi) ** 0.5), 1)})
    return dots


def detect_clusters(img):
    """Return list of {x,y,area,w,h} for each 'positive' large dark-blue blob."""
    a, R, G, B, luma = _channels(img)
    mask = dark_blue_mask(luma, R, B)
    lbl, n = ndimage.label(mask)
    if n == 0:
        return []
    areas = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    cents = ndimage.center_of_mass(np.ones_like(lbl), lbl, range(1, n + 1))
    slices = ndimage.find_objects(lbl)
    out = []
    for i in range(n):
        if not (MIN_BLOB_AREA <= areas[i] <= MAX_BLOB_AREA):
            continue
        cy, cx = cents[i]; sl = slices[i]
        out.append({"x": round(float(cx), 1), "y": round(float(cy), 1),
                    "area": int(areas[i]),
                    "w": int(sl[1].stop - sl[1].start),
                    "h": int(sl[0].stop - sl[0].start)})
    return out
