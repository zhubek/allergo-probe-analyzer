"""Per-blob feature extraction.

The classifier uses these features to decide whether a candidate blob looks
like an expert-labeled positive. Shared between training (classify_train.py)
and inference (allergo_core.py classifier mode).

Pipeline:
    1. compute_image_stack(img)        ->  per-pixel arrays (luma, field, B-R, gradient, ...)
    2. candidate_features(stack)       ->  (features, centers, labels) per candidate blob

The "relaxed mask" used by candidate_features() is intentionally more permissive
than allergo_core's defaults — the goal at this stage is high recall (don't
miss any real positive). The classifier handles selectivity downstream.
"""
import numpy as np
from scipy import ndimage

# Order is part of the saved model — DO NOT reorder without retraining.
# Base features = one-blob measurements. Derived features = ratios/z-scores against
# this image's own population of candidates (so the model is scale-invariant) plus
# image-context features so the model knows what kind of scene it's looking at.
BASE_FEATURE_NAMES = (
    "area", "width", "height",
    "aspect_ratio", "fill_ratio", "compactness",
    "mean_R", "mean_G", "mean_B",
    "mean_blueness", "std_blueness",
    "mean_gradient",
    "dist_from_center",
)
DERIVED_FEATURE_NAMES = (
    # scale-invariant size — one feature instead of three correlated ones.
    "area_zscore",           # (area - mean_area) / std_area
    # color identity vs typical dot color in this image.
    "blueness_ratio",        # mean_blueness / median(mean_blueness)
    "gradient_ratio",        # mean_gradient / median(mean_gradient)
    # color-outlier — main artefact-rejection feature (warm stains, off-color specks).
    "color_distance_rgb",    # Euclidean distance in (R,G,B) from image median color
    # image-context (constant across all candidates in one image)
    "n_candidates_in_image",
    "image_mean_blueness",
)
FEATURE_NAMES = BASE_FEATURE_NAMES + DERIVED_FEATURE_NAMES
N_FEATURES = len(FEATURE_NAMES)
_BASE_COL = {name: i for i, name in enumerate(BASE_FEATURE_NAMES)}    # index lookup

# Relaxed candidate-mask defaults: high recall, low precision (classifier filters).
RELAXED_FIELD_MIN   = 100
RELAXED_DARK_DROP   = 50
RELAXED_BLUE_EXCESS = 15
RELAXED_MIN_AREA    = 50
RELAXED_MAX_AREA    = 20000
FIELD_SIGMA = 60     # match allergo_core


def compute_image_stack(img):
    """Compute per-pixel arrays needed for the mask and per-blob features.

    Heavy work (the σ=60 Gaussian, the Sobel gradient) happens here and is
    reused across every candidate blob in the image.
    """
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    luma = 0.299 * R + 0.587 * G + 0.114 * B
    field = ndimage.gaussian_filter(luma, FIELD_SIGMA)
    diff = field - luma          # "darkness drop" — positive where pixel < local average
    blueness = B - R             # >0 = bluish, <0 = warm (brown/orange)
    gx = ndimage.sobel(luma, axis=1)
    gy = ndimage.sobel(luma, axis=0)
    grad = np.hypot(gx, gy).astype(np.float32)
    H, W = luma.shape
    return {
        "R": R, "G": G, "B": B,
        "luma": luma, "field": field, "diff": diff,
        "blueness": blueness, "grad": grad,
        "H": H, "W": W,
    }


def relaxed_mask(stack,
                 field_min=RELAXED_FIELD_MIN,
                 dark_drop=RELAXED_DARK_DROP,
                 blue_excess=RELAXED_BLUE_EXCESS):
    """High-recall candidate mask (more permissive than allergo_core defaults)."""
    return ((stack["field"] > field_min)
            & (stack["diff"] > dark_drop)
            & (stack["blueness"] > blue_excess))


def candidate_features(stack,
                       min_area=RELAXED_MIN_AREA,
                       max_area=RELAXED_MAX_AREA):
    """Detect candidate blobs and return their features.

    Returns:
        feats   : (N, N_FEATURES) float32 — feature matrix, one row per candidate.
        centers : (N, 2) float32 — (cx, cy) center of each candidate.
        bboxes  : (N, 4) int32 — (x0, y0, w, h) bbox per candidate.
    """
    mask = relaxed_mask(stack)
    lbl, n = ndimage.label(mask)
    empty = (np.empty((0, N_FEATURES), dtype=np.float32),
             np.empty((0, 2), dtype=np.float32),
             np.empty((0, 4), dtype=np.int32))
    if n == 0:
        return empty

    all_labels = np.arange(1, n + 1)
    all_areas = ndimage.sum(np.ones_like(lbl, dtype=np.float32), lbl, all_labels)
    keep = (all_areas >= min_area) & (all_areas <= max_area)
    if not keep.any():
        return empty

    labels = all_labels[keep]
    areas = all_areas[keep]

    slices = ndimage.find_objects(lbl)
    centroids = ndimage.center_of_mass(np.ones_like(lbl, dtype=np.float32), lbl, labels)
    mean_R = ndimage.mean(stack["R"], lbl, labels)
    mean_G = ndimage.mean(stack["G"], lbl, labels)
    mean_B = ndimage.mean(stack["B"], lbl, labels)
    mean_blueness = ndimage.mean(stack["blueness"], lbl, labels)
    std_blueness = ndimage.standard_deviation(stack["blueness"], lbl, labels)
    mean_gradient = ndimage.mean(stack["grad"], lbl, labels)

    H, W = stack["H"], stack["W"]
    cx_arr = np.asarray([c[1] for c in centroids], dtype=np.float32)
    cy_arr = np.asarray([c[0] for c in centroids], dtype=np.float32)

    K = len(labels)
    base = np.zeros((K, len(BASE_FEATURE_NAMES)), dtype=np.float32)
    bboxes = np.zeros((K, 4), dtype=np.int32)
    for i, lab_id in enumerate(labels):
        sl = slices[lab_id - 1]                  # find_objects is 0-indexed by label-1
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        h = y1 - y0
        w = x1 - x0
        min_dim = max(min(w, h), 1)
        max_dim = max(max(w, h), 1)
        base[i] = (
            areas[i],
            w, h,
            max_dim / min_dim,                   # aspect_ratio
            float(areas[i]) / max(w * h, 1),     # fill_ratio
            float(areas[i]) / (max_dim * max_dim),  # compactness
            mean_R[i], mean_G[i], mean_B[i],
            mean_blueness[i], std_blueness[i],
            mean_gradient[i],
            float(np.hypot(cx_arr[i] - W / 2.0, cy_arr[i] - H / 2.0)),
        )
        bboxes[i] = (x0, y0, w, h)

    # ---- derived (per-image scale-invariant + context) features ----
    EPS = 1e-6
    col_area = base[:, _BASE_COL["area"]]
    col_blue = base[:, _BASE_COL["mean_blueness"]]
    col_grad = base[:, _BASE_COL["mean_gradient"]]
    col_R    = base[:, _BASE_COL["mean_R"]]
    col_G    = base[:, _BASE_COL["mean_G"]]
    col_B    = base[:, _BASE_COL["mean_B"]]

    img_mean_area   = float(col_area.mean())
    img_std_area    = max(float(col_area.std()), EPS)
    img_median_blue = float(np.median(col_blue))
    img_median_grad = float(np.median(col_grad))
    img_mean_blue   = float(col_blue.mean())
    img_median_R    = float(np.median(col_R))
    img_median_G    = float(np.median(col_G))
    img_median_B    = float(np.median(col_B))

    area_zscore = (col_area - img_mean_area) / img_std_area
    blue_ratio  = col_blue / max(img_median_blue, EPS)
    grad_ratio  = col_grad / max(img_median_grad, EPS)
    color_distance_rgb = np.sqrt(
        (col_R - img_median_R) ** 2
        + (col_G - img_median_G) ** 2
        + (col_B - img_median_B) ** 2
    )
    ones = np.ones(K, dtype=np.float32)
    derived = np.column_stack([
        area_zscore,
        blue_ratio, grad_ratio,
        color_distance_rgb,
        ones * float(K),
        ones * img_mean_blue,
    ]).astype(np.float32)

    feats = np.column_stack([base, derived]).astype(np.float32)
    centers = np.column_stack([cx_arr, cy_arr]).astype(np.float32)
    return feats, centers, bboxes
