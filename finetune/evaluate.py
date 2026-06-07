"""Measure precision/recall/F1 of the current detector against labels.db.

Uses allergo_core.detect_clusters with whichever thresholds are active — the
hardcoded defaults if no thresholds.json exists, the tuned values if one does.
Useful as a baseline before training, and to confirm gains after.

Usage:
    python finetune/evaluate.py                        # all annotated images
    python finetune/evaluate.py --limit 20             # subsample for speed
    python finetune/evaluate.py --images D:/raw_pngs   # custom image folder
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

from PIL import Image

# allow imports from project root when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import allergo_core as core                                  # noqa: E402
from finetune.matching import match, prf1                    # noqa: E402
from finetune.train import load_annotations                  # noqa: E402


def evaluate_one(payload):
    image_path, gt_xywh = payload
    try:
        img = Image.open(image_path).convert("RGB")
        preds = core.detect_clusters(img)
        pred_xy = [(p["x"], p["y"]) for p in preds]
        tp, fp, fn = match(pred_xy, gt_xywh)
        return image_path, tp, fp, fn, len(preds), None
    except Exception as e:                                       # noqa: BLE001
        return image_path, 0, 0, len(gt_xywh), 0, repr(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default="data/labels.db")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--split", choices=("all", "train", "test"), default="all",
                    help="Evaluate on all images, only train images, or only "
                         "held-out test images (read from finetune/test_split.json)")
    args = ap.parse_args()

    db_path = Path(args.db)
    img_dir = Path(args.images)
    if not db_path.exists():
        sys.exit(f"labels DB not found: {db_path}")
    if not img_dir.exists():
        sys.exit(f"images folder not found: {img_dir}")

    gt_by_file = load_annotations(db_path)
    workitems = [(str(img_dir / fn), gts) for fn, gts in gt_by_file.items()
                 if (img_dir / fn).exists()]
    if not workitems:
        sys.exit(f"No annotated images found in {img_dir}.")

    if args.split != "all":
        split_path = Path("models/test_split.json")
        if not split_path.exists():
            sys.exit(f"--split={args.split} requested but {split_path} not found. "
                     f"Run finetune/classify_train.py first.")
        import json
        split = json.loads(split_path.read_text())
        keep = set(split[args.split])
        before = len(workitems)
        workitems = [(p, g) for p, g in workitems if Path(p).name in keep]
        print(f"Filtered to --split={args.split}: {len(workitems)}/{before} images.")

    if args.limit:
        workitems = workitems[: args.limit]

    print(f"Active thresholds: FIELD_MIN={core.FIELD_MIN}  DARK_DROP={core.DARK_DROP}  "
          f"BLUE_EXCESS={core.BLUE_EXCESS}  "
          f"MIN_BLOB_AREA={core.MIN_BLOB_AREA}  MAX_BLOB_AREA={core.MAX_BLOB_AREA}")
    print(f"Evaluating on {len(workitems)} annotated images "
          f"({sum(len(g) for _, g in workitems)} GT boxes). Workers: {args.workers}.")

    tp_total = fp_total = fn_total = preds_total = 0
    t0 = time.time()
    done = 0
    with mp.Pool(args.workers) as pool:
        for path, tp, fp, fn, n_preds, err in pool.imap_unordered(evaluate_one, workitems):
            done += 1
            if err:
                print(f"  [{done}/{len(workitems)}] FAILED {Path(path).name}: {err}")
                continue
            tp_total += tp; fp_total += fp; fn_total += fn; preds_total += n_preds
            if done % max(1, len(workitems) // 10) == 0 or done == len(workitems):
                print(f"  [{done}/{len(workitems)}]  elapsed={time.time()-t0:.0f}s")

    p, r, f1 = prf1(tp_total, fp_total, fn_total)
    gt_total = tp_total + fn_total
    print()
    print("=" * 60)
    print(f"  Ground-truth boxes : {gt_total}")
    print(f"  Predictions made   : {preds_total}")
    print(f"  TP (GT recalled)   : {tp_total}")
    print(f"  FP (no GT hit)     : {fp_total}")
    print(f"  FN (GT missed)     : {fn_total}")
    print(f"  Precision          : {p:.3f}")
    print(f"  Recall             : {r:.3f}")
    print(f"  F1                 : {f1:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
