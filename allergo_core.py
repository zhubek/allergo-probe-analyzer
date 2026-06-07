"""Core detection for allergo_probe microscopy images.

Two detectors over the same dark-blue mask:
  * detect_dots()     -> every clearly-dark dark-blue dot (watershed-split)
  * detect_clusters()  -> "positive" large dark-blue blobs (the annotated objects)

Both take a PIL.Image (RGB) and return plain dicts, so the CLI scripts and the
API can share one implementation.

Three layered ways the cluster detector can be configured:

  1. Hardcoded defaults below — used when nothing else is present.
  2. ``models/thresholds.json`` — written by ``finetune/train.py``; overrides
     the tunable thresholds at import time. Path can be overridden via the
     ``ALLERGO_THRESHOLDS`` env var.
  3. ``models/classifier.joblib`` — written by ``finetune/classify_train.py``;
     when present takes precedence over both 1 and 2. ``detect_clusters`` then
     uses a relaxed candidate mask + per-blob feature classifier instead of the
     5-threshold AND-rule. Path can be overridden via ``ALLERGO_CLASSIFIER``.
     Decision threshold tunable via ``ALLERGO_SCORE_THRESHOLD`` (default 0.5).
     Extreme outliers further filtered via ``ALLERGO_MAX_ASPECT_RATIO`` and
     ``ALLERGO_MAX_AREA_ZSCORE`` (see ``_detect_clusters_classifier``).

``active_thresholds()`` reports which mode is in use.
"""
import json
import os
import warnings
from pathlib import Path

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

# Names that can be overridden by finetune/thresholds.json.
_TUNABLE = ("FIELD_MIN", "DARK_DROP", "BLUE_EXCESS", "MIN_BLOB_AREA", "MAX_BLOB_AREA")
_thresholds_source = "default"   # "default" | "<path to thresholds.json>"
_thresholds_meta = None          # the JSON file contents (when loaded)

_classifier_pipeline = None      # sklearn Pipeline (scaler + clf), or None
_classifier_meta = None          # dict from classify_train.py (cv metrics, feature names, ...)
_classifier_source = None        # path of the loaded classifier file


def _maybe_load_thresholds():
    """Override module-level constants from models/thresholds.json if present."""
    global _thresholds_source, _thresholds_meta
    path = Path(os.environ.get("ALLERGO_THRESHOLDS") or
                Path(__file__).resolve().parent / "models" / "thresholds.json")
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        warnings.warn(f"allergo_core: ignoring {path} ({e})")
        return
    for name in _TUNABLE:
        if name in data:
            globals()[name] = type(globals()[name])(data[name])
    _thresholds_source = str(path)
    _thresholds_meta = data


def _maybe_load_classifier():
    """Load models/classifier.joblib if present. Takes precedence over thresholds."""
    global _classifier_pipeline, _classifier_meta, _classifier_source
    path = Path(os.environ.get("ALLERGO_CLASSIFIER") or
                Path(__file__).resolve().parent / "models" / "classifier.joblib")
    if not path.exists():
        return
    try:
        import joblib                                            # lazy: dep only needed when used
        data = joblib.load(path)
    except Exception as e:                                       # noqa: BLE001
        warnings.warn(f"allergo_core: ignoring classifier {path} ({e})")
        return
    _classifier_pipeline = data["pipeline"]
    _classifier_meta = data.get("meta", {})
    _classifier_source = str(path)


def active_thresholds():
    """Report which detection mode is in use and its key settings."""
    if _classifier_pipeline is not None:
        return {
            "mode": "classifier",
            "source": _classifier_source,
            "score_threshold": float(os.environ.get("ALLERGO_SCORE_THRESHOLD",
                                                    _classifier_meta.get("score_threshold", 0.5))),
            "classifier_meta": {
                k: v for k, v in _classifier_meta.items()
                # don't dump the giant per-feature normalization vectors in the API response
                if k not in ("feature_scale_mean", "feature_scale_std")
            },
        }
    return {
        "mode": "tuned" if _thresholds_source != "default" else "default",
        "source": _thresholds_source,
        "metrics": (_thresholds_meta or {}).get("metrics"),
        "values": {name: globals()[name] for name in _TUNABLE},
    }


_maybe_load_thresholds()
_maybe_load_classifier()


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


def detect_clusters(img, score_threshold=None):
    """Return list of {x,y,area,w,h} for each 'positive' dark-blue blob.

    When a classifier is loaded (finetune/classifier.joblib) it overrides the
    threshold rule: each candidate from a relaxed mask is scored by the
    classifier, and only those above the score threshold are returned. Each
    returned dict also carries the classifier ``score`` in that mode.

    ``score_threshold`` (only meaningful in classifier mode): override the
    decision cutoff per call. When None, falls back to ``ALLERGO_SCORE_THRESHOLD``
    env var, then the value baked into classifier.joblib's metadata.
    """
    if _classifier_pipeline is not None:
        return _detect_clusters_classifier(img, score_threshold)

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


def _detect_clusters_classifier(img, score_threshold=None):
    """detect_clusters path used when finetune/classifier.joblib is loaded.

    After classifier scoring, applies hard rejection rules to filter the
    extreme outliers a linear classifier rewards but shouldn't: blobs that are
    *suspiciously* large or elongated. Caps tunable via env vars
    ``ALLERGO_MAX_AREA_ZSCORE`` (default 9.0) and ``ALLERGO_MAX_ASPECT_RATIO``
    (default 3.5). Set them to a large number (e.g. 1e9) to disable.
    """
    from finetune.features import compute_image_stack, candidate_features, FEATURE_NAMES
    stack = compute_image_stack(img)
    feats, centers, bboxes = candidate_features(stack)
    if len(feats) == 0:
        return []
    proba = _classifier_pipeline.predict_proba(feats)[:, 1]
    if score_threshold is None:
        score_threshold = float(os.environ.get(
            "ALLERGO_SCORE_THRESHOLD",
            _classifier_meta.get("score_threshold", 0.5)))
    threshold = float(score_threshold)

    keep_mask = proba >= threshold

    # Hard rejection rules for extreme outliers that LR can't filter.
    # Tunable via env vars; set to a large number (1e9) to disable.
    max_area_zscore = float(os.environ.get("ALLERGO_MAX_AREA_ZSCORE", 9.0))
    max_aspect_ratio = float(os.environ.get("ALLERGO_MAX_ASPECT_RATIO", 3.5))
    # "needs some elongation" gate — area_zscore cap only kicks in if the blob
    # is also at least mildly elongated. This spares round positives in dense
    # fields (which legitimately have very high area_zscore vs the tiny median).
    AREA_CAP_ASPECT_GATE = 1.8

    try:
        j_ar = FEATURE_NAMES.index("aspect_ratio")
        # rule 1: extreme aspect alone is enough (very long thin shapes)
        keep_mask &= (feats[:, j_ar] <= max_aspect_ratio)
    except ValueError:
        j_ar = None

    try:
        j_az = FEATURE_NAMES.index("area_zscore")
        # rule 2: extreme size + at least mildly elongated (likely streak)
        extreme_size = feats[:, j_az] > max_area_zscore
        if j_ar is not None:
            extreme_size &= (feats[:, j_ar] > AREA_CAP_ASPECT_GATE)
        keep_mask &= ~extreme_size
    except ValueError:
        pass

    keep = np.where(keep_mask)[0]
    out = []
    for i in keep:
        cx, cy = float(centers[i, 0]), float(centers[i, 1])
        x0, y0, w, h = bboxes[i]
        out.append({"x": round(cx, 1), "y": round(cy, 1),
                    "area": int(feats[i, 0]),       # area is feature 0
                    "w": int(w), "h": int(h),
                    "score": round(float(proba[i]), 3)})
    return out
