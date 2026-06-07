"""Sweep the classifier's score threshold over the held-out test images.

For each image: extract candidate features once, get classifier scores once.
For each candidate score threshold: filter predictions and compute TP/FP/FN.

This is essentially evaluate.py for the classifier, but parameterized over
``score_threshold`` so we can see the full precision/recall trade-off in one
pass instead of re-running detection per threshold.

Usage:
    python finetune/sweep_threshold.py                     # default 0.30 -> 0.95 step 0.05
    python finetune/sweep_threshold.py --split all         # sweep over every image (slower)
    python finetune/sweep_threshold.py --start 0.5 --stop 0.99 --step 0.02
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finetune.features import candidate_features, compute_image_stack         # noqa: E402
from finetune.matching import match                                           # noqa: E402
from finetune.train import load_annotations                                   # noqa: E402


def score_one(payload):
    """Worker: load image, extract features, get classifier scores + centers."""
    path, gt, pipeline = payload
    try:
        img = Image.open(path).convert("RGB")
        stack = compute_image_stack(img)
        feats, centers, _ = candidate_features(stack)
        if len(feats) == 0:
            return path, np.empty((0,), dtype=np.float32), np.empty((0, 2), dtype=np.float32), gt, None
        scores = pipeline.predict_proba(feats)[:, 1].astype(np.float32)
        return path, scores, centers, gt, None
    except Exception as e:                                                    # noqa: BLE001
        return path, None, None, gt, repr(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default="data/labels.db")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--classifier", default="models/classifier.joblib")
    ap.add_argument("--split", choices=("all", "train", "test"), default="test")
    ap.add_argument("--start", type=float, default=0.30)
    ap.add_argument("--stop", type=float, default=0.95)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if not Path(args.classifier).exists():
        sys.exit(f"classifier not found: {args.classifier}")
    data = joblib.load(args.classifier)
    pipeline = data["pipeline"]
    meta = data.get("meta", {})

    gt_by_file = load_annotations(args.db)
    workitems = [(str(Path(args.images) / f), gts) for f, gts in gt_by_file.items()
                 if (Path(args.images) / f).exists()]
    if args.split != "all":
        split_path = Path("models/test_split.json")
        if not split_path.exists():
            sys.exit("models/test_split.json missing — run classify_train.py first")
        split = json.loads(split_path.read_text())
        keep = set(split[args.split])
        workitems = [(p, g) for p, g in workitems if Path(p).name in keep]
    print(f"Sweeping classifier threshold on {len(workitems)} images "
          f"(--split={args.split}), workers={args.workers}.")

    payloads = [(p, g, pipeline) for p, g in workitems]
    per_image_scored = []     # list of (scores, centers, gt_xywh)
    t0 = time.time()
    done = 0
    with mp.Pool(args.workers) as pool:
        for path, scores, centers, gt, err in pool.imap_unordered(score_one, payloads):
            done += 1
            if err:
                print(f"  FAILED {Path(path).name}: {err}")
                continue
            per_image_scored.append((scores, centers, gt))
            if done % max(1, len(payloads) // 10) == 0 or done == len(payloads):
                print(f"  [{done}/{len(payloads)}]  elapsed={time.time()-t0:.0f}s")

    thresholds = np.arange(args.start, args.stop + args.step / 2, args.step)
    print()
    print(f"{'threshold':>9}  {'TP':>5} {'FP':>6} {'FN':>4}  "
          f"{'P':>6} {'R':>6} {'F1':>6} {'F2':>6}  {'preds/img':>9}")
    rows = []
    n_imgs = len(per_image_scored)
    for thr in thresholds:
        tp_total = fp_total = fn_total = preds_total = 0
        for scores, centers, gt in per_image_scored:
            keep = scores >= thr
            pred_xy = [(float(c[0]), float(c[1])) for c in centers[keep]]
            tp, fp, fn = match(pred_xy, gt)
            tp_total += tp; fp_total += fp; fn_total += fn
            preds_total += len(pred_xy)
        p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
        r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        f2 = 5 * p * r / (4 * p + r) if (4 * p + r) else 0.0
        preds_per_img = preds_total / max(n_imgs, 1)
        rows.append({"threshold": float(thr), "tp": tp_total, "fp": fp_total,
                     "fn": fn_total, "p": p, "r": r, "f1": f1, "f2": f2,
                     "preds_per_img": preds_per_img})
        print(f"{thr:>9.2f}  {tp_total:>5} {fp_total:>6} {fn_total:>4}  "
              f"{p:>.4f} {r:>.4f} {f1:>.4f} {f2:>.4f}  {preds_per_img:>9.1f}")

    out_path = Path(args.classifier).parent / f"threshold_sweep_{args.split}.json"
    out_path.write_text(json.dumps({
        "split": args.split,
        "n_images": n_imgs,
        "classifier": str(args.classifier),
        "classifier_model": meta.get("model"),
        "rows": rows,
    }, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
