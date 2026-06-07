"""Matching predicted blobs against ground-truth bboxes from labels.db.

A predicted blob "hits" a GT box if the blob's center (cx, cy) falls inside
the box [x, x+w] x [y, y+h] (top-left origin, as stored in labels.db).

  TP = GT boxes recalled (at least one predicted center inside)
  FP = predictions whose center is inside no GT box
  FN = GT boxes not recalled
"""
import numpy as np


def match(pred_centers_xy, gt_boxes_xywh):
    """Return (tp, fp, fn) for one image.

    pred_centers_xy : iterable of (cx, cy) in full-resolution pixels.
    gt_boxes_xywh   : iterable of (x, y, w, h) where (x, y) is top-left.
    """
    K = len(pred_centers_xy)
    M = len(gt_boxes_xywh)
    if M == 0:
        return 0, K, 0
    if K == 0:
        return 0, 0, M

    p = np.asarray(pred_centers_xy, dtype=np.float64)
    g = np.asarray(gt_boxes_xywh, dtype=np.float64)
    px = p[:, 0:1]                      # (K, 1)
    py = p[:, 1:2]
    gxmin = g[:, 0]                     # (M,)
    gymin = g[:, 1]
    gxmax = g[:, 0] + g[:, 2]
    gymax = g[:, 1] + g[:, 3]
    inside = (px >= gxmin) & (px <= gxmax) & (py >= gymin) & (py <= gymax)  # (K, M)

    tp = int(inside.any(axis=0).sum())  # GT boxes hit by >=1 prediction
    fp = int((~inside.any(axis=1)).sum())  # predictions inside no GT
    fn = M - tp
    return tp, fp, fn


def prf1(tp, fp, fn):
    """Standard precision / recall / F1 (returns 0.0 for zero denominators)."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1
