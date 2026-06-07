"""Grid-search the dark-blue mask + area thresholds against labels.db.

Reads ground-truth bboxes from labels.db, evaluates every (FIELD_MIN, DARK_DROP,
BLUE_EXCESS, MIN_BLOB_AREA, MAX_BLOB_AREA) combo over the annotated images, and
writes the best combo to finetune/thresholds.json along with its
precision/recall/F1. allergo_core.py picks the file up automatically.

Usage:
    python finetune/train.py                        # full grid, all annotated images
    python finetune/train.py --quick                # small grid (smoke test)
    python finetune/train.py --limit 20             # use first 20 images only
    python finetune/train.py --workers 4            # cap memory: 4 parallel images
    python finetune/train.py --images D:/raw_pngs   # custom image folder
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# allow `from finetune.matching import ...` when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finetune.matching import match, prf1     # noqa: E402

FIELD_SIGMA = 60   # not tuned (expensive Gaussian) — kept at allergo_core default


def grids(quick=False):
    """Mask and area parameter grids. Quick mode is for fast iteration."""
    if quick:
        FM = [110, 130]
        DD = [60, 70, 80]
        BE = [20, 25, 30]
        MN = [150, 250, 400]
        MX = [8000]
    else:
        FM = [100, 120, 140]
        DD = [50, 60, 70, 80, 90]
        BE = [15, 20, 25, 30, 35]
        MN = [100, 150, 200, 250, 350, 500, 750, 1000]
        MX = [4000, 8000, 15000]
    mask_grid = [(fm, dd, be) for fm in FM for dd in DD for be in BE]
    area_grid = [(mn, mx) for mn in MN for mx in MX]
    return mask_grid, area_grid


def evaluate_image(image_path, gt_xywh, mask_grid, area_grid):
    """Run every combo against one image. Returns {combo: (tp, fp, gt_total)}."""
    a = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    luma = 0.299 * R + 0.587 * G + 0.114 * B
    field = ndimage.gaussian_filter(luma, FIELD_SIGMA)
    diff = field - luma
    blueness = B - R

    M_total = len(gt_xywh)
    out = {}
    for (fm, dd, be) in mask_grid:
        mask = (field > fm) & (diff > dd) & (blueness > be)
        lbl, n = ndimage.label(mask)
        if n == 0:
            for (mn, mx) in area_grid:
                out[(fm, dd, be, mn, mx)] = (0, 0, M_total)
            continue
        areas = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        cents = ndimage.center_of_mass(np.ones_like(lbl), lbl, range(1, n + 1))
        cents_xy = np.array([(c[1], c[0]) for c in cents], dtype=np.float64)
        for (mn, mx) in area_grid:
            keep = (areas >= mn) & (areas <= mx)
            tp, fp, _ = match(cents_xy[keep], gt_xywh)
            out[(fm, dd, be, mn, mx)] = (tp, fp, M_total)
    return out


def _worker(payload):
    image_path, gt_xywh, mask_grid, area_grid = payload
    try:
        return image_path, evaluate_image(image_path, gt_xywh, mask_grid, area_grid), None
    except Exception as e:                                       # noqa: BLE001
        return image_path, None, repr(e)


def load_annotations(db_path):
    """Return {filename: [(x, y, w, h), ...]} from labels.db."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT filename, x, y, width, height FROM annotations"
    ).fetchall()
    conn.close()
    gt = {}
    for fn, x, y, w, h in rows:
        gt.setdefault(fn, []).append((float(x), float(y), float(w), float(h)))
    return gt


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default="data/labels.db")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--out", default="models/thresholds.json")
    ap.add_argument("--grid-out", default="models/grid_results.csv")
    ap.add_argument("--quick", action="store_true", help="small grid for fast iteration")
    ap.add_argument("--limit", type=int, default=0, help="use only first N annotated images (0=all)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    db_path = Path(args.db)
    img_dir = Path(args.images)
    if not db_path.exists():
        sys.exit(f"labels DB not found: {db_path}")
    if not img_dir.exists():
        sys.exit(f"images folder not found: {img_dir}\n"
                 f"Create it and drop the full-resolution PNGs in there "
                 f"(filenames must match the DB).")

    gt_by_file = load_annotations(db_path)
    total_ann = sum(len(v) for v in gt_by_file.values())
    print(f"Loaded {total_ann} annotations across {len(gt_by_file)} images.")

    workitems, missing = [], 0
    for fn, gts in gt_by_file.items():
        p = img_dir / fn
        if p.exists():
            workitems.append((str(p), gts))
        else:
            missing += 1
    if not workitems:
        sys.exit(f"No annotated images found in {img_dir}.\n"
                 f"Drop the full-resolution PNGs there — filenames must match the DB.")
    if missing:
        print(f"Warning: {missing} annotated images missing from {img_dir} — skipped.")
    if args.limit:
        workitems = workitems[: args.limit]
    print(f"Training on {len(workitems)} images.")

    mask_grid, area_grid = grids(quick=args.quick)
    n_combos = len(mask_grid) * len(area_grid)
    print(f"Grid: {len(mask_grid)} mask × {len(area_grid)} area = {n_combos} combos. "
          f"Workers: {args.workers}.")

    accum = {(fm, dd, be, mn, mx): [0, 0, 0]
             for (fm, dd, be) in mask_grid for (mn, mx) in area_grid}
    pool_args = [(p, gts, mask_grid, area_grid) for (p, gts) in workitems]

    t0 = time.time()
    done = 0
    log_every = max(1, len(pool_args) // 20)
    with mp.Pool(args.workers) as pool:
        for path, result, err in pool.imap_unordered(_worker, pool_args):
            done += 1
            if err:
                print(f"[{done}/{len(pool_args)}] FAILED {Path(path).name}: {err}")
                continue
            for combo, (tp, fp, mt) in result.items():
                a = accum[combo]
                a[0] += tp; a[1] += fp; a[2] += mt
            if done % log_every == 0 or done == len(pool_args):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0.0
                eta = (len(pool_args) - done) / rate if rate else 0.0
                print(f"  [{done}/{len(pool_args)}]  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    # rank combos
    rows = []
    for combo, (tp, fp, mt) in accum.items():
        fn = mt - tp
        p, r, f1 = prf1(tp, fp, fn)
        rows.append((*combo, tp, fp, fn, p, r, f1))
    # F1 desc, then recall desc (favor recall in medical context), then min_area asc
    rows.sort(key=lambda r: (-r[10], -r[9], r[3]))
    best = rows[0]
    fm, dd, be, mn, mx, tp, fp, fn, p, r, f1 = best

    print()
    print("=" * 78)
    print("Top 10 combos by F1 (tie-break: recall, then lower MIN_BLOB_AREA):")
    print(f"  {'FM':>4} {'DD':>3} {'BE':>3} {'MIN':>6} {'MAX':>6}   "
          f"{'TP':>4} {'FP':>5} {'FN':>4}   {'P':>5}  {'R':>5}  {'F1':>5}")
    for row in rows[:10]:
        f_, d_, b_, mn_, mx_, tp_, fp_, fn_, p_, r_, f1_ = row
        print(f"  {f_:>4} {d_:>3} {b_:>3} {mn_:>6} {mx_:>6}   "
              f"{tp_:>4} {fp_:>5} {fn_:>4}   {p_:.3f}  {r_:.3f}  {f1_:.3f}")
    print("=" * 78)
    print(f"Best: FIELD_MIN={fm} DARK_DROP={dd} BLUE_EXCESS={be} "
          f"MIN_BLOB_AREA={mn} MAX_BLOB_AREA={mx}")
    print(f"      precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}  "
          f"({tp} TP, {fp} FP, {fn} FN)")

    Path(args.out).write_text(json.dumps({
        "FIELD_MIN": fm, "DARK_DROP": dd, "BLUE_EXCESS": be,
        "MIN_BLOB_AREA": mn, "MAX_BLOB_AREA": mx,
        "metrics": {"precision": round(p, 4), "recall": round(r, 4),
                    "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn},
        "trained_on": {"n_images": len(workitems), "n_annotations": total_ann,
                       "grid_size": n_combos, "quick": args.quick},
    }, indent=2))
    print(f"\nWrote {args.out}  (allergo_core will pick this up automatically)")

    with open(args.grid_out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["FIELD_MIN", "DARK_DROP", "BLUE_EXCESS", "MIN_BLOB_AREA", "MAX_BLOB_AREA",
                    "TP", "FP", "FN", "precision", "recall", "F1"])
        for row in rows:
            f_, d_, b_, mn_, mx_, tp_, fp_, fn_, p_, r_, f1_ = row
            w.writerow([f_, d_, b_, mn_, mx_, tp_, fp_, fn_,
                        f"{p_:.4f}", f"{r_:.4f}", f"{f1_:.4f}"])
    print(f"Wrote {args.grid_out}")


if __name__ == "__main__":
    main()
