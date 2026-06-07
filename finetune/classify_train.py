"""Train the per-blob classifier from labels.db + full-res images.

Pipeline:
    1. For each annotated image: extract candidates (relaxed mask) + features.
    2. Label each candidate: positive if its center falls inside any GT bbox.
    3. Standardize features + train LogisticRegression with class_weight='balanced'
       (recall-favoring, since experts under-labeled).
    4. 5-fold stratified cross-validation for an honest generalization estimate.
    5. Save pipeline + metadata + scaler stats to finetune/classifier.joblib.

allergo_core auto-loads the classifier at import once the file exists.

Usage:
    python finetune/classify_train.py
    python finetune/classify_train.py --limit 20 --workers 2
    python finetune/classify_train.py --model rf      # random forest instead of LR
"""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finetune.features import (                                      # noqa: E402
    FEATURE_NAMES, N_FEATURES, candidate_features, compute_image_stack,
)
from finetune.train import load_annotations                          # noqa: E402


def extract_one(payload):
    """Worker: load image, extract candidates, label by GT bbox containment."""
    image_path, gt_xywh = payload
    try:
        img = Image.open(image_path).convert("RGB")
        stack = compute_image_stack(img)
        feats, centers, _ = candidate_features(stack)
        if len(feats) == 0:
            return image_path, np.empty((0, N_FEATURES), dtype=np.float32), \
                   np.empty((0,), dtype=np.int8), 0, None

        labels = np.zeros(len(feats), dtype=np.int8)
        gt_hits = 0
        if gt_xywh:
            cx, cy = centers[:, 0], centers[:, 1]
            for gx, gy, gw, gh in gt_xywh:
                inside = (cx >= gx) & (cx <= gx + gw) & (cy >= gy) & (cy <= gy + gh)
                if inside.any():
                    gt_hits += 1
                    labels[inside] = 1
        return image_path, feats, labels, gt_hits, None
    except Exception as e:                                           # noqa: BLE001
        return image_path, None, None, 0, repr(e)


def build_pipeline(kind):
    """Build the sklearn pipeline. Both options handle imbalance via class_weight."""
    if kind == "logreg":
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs")
    elif kind == "rf":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     max_depth=None, min_samples_leaf=5,
                                     n_jobs=-1, random_state=42)
    else:
        raise ValueError(f"unknown model kind: {kind}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default="data/labels.db")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--out", default="models/classifier.joblib")
    ap.add_argument("--model", choices=("logreg", "rf"), default="logreg",
                    help="logreg (interpretable) | rf (nonlinear)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--test-fraction", type=float, default=0.2,
                    help="Fraction of IMAGES held out for the final test set (0 = no holdout)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for the train/test image split (reproducible)")
    args = ap.parse_args()

    db_path = Path(args.db)
    img_dir = Path(args.images)
    if not db_path.exists():
        sys.exit(f"labels DB not found: {db_path}")
    if not img_dir.exists():
        sys.exit(f"images folder not found: {img_dir}")

    gt_by_file = load_annotations(db_path)
    workitems = [(str(img_dir / f), gts) for f, gts in gt_by_file.items()
                 if (img_dir / f).exists()]
    if args.limit:
        workitems = workitems[: args.limit]
    if not workitems:
        sys.exit(f"no annotated images present in {img_dir}")

    total_gt = sum(len(g) for _, g in workitems)
    print(f"Extracting features from {len(workitems)} images "
          f"({total_gt} GT bboxes). Workers: {args.workers}.")

    # Track features PER IMAGE so we can split by image (no leakage).
    per_image = []   # list of (filename, feats, labels, gt_recalled_in_this_image)
    gt_recalled_total = 0
    t0 = time.time()
    done = 0
    log_every = max(1, len(workitems) // 10)
    with mp.Pool(args.workers) as pool:
        for path, feats, labels, hits, err in pool.imap_unordered(extract_one, workitems):
            done += 1
            if err:
                print(f"  [{done}/{len(workitems)}] FAILED {Path(path).name}: {err}")
                continue
            per_image.append((Path(path).name, feats, labels, hits))
            gt_recalled_total += hits
            if done % log_every == 0 or done == len(workitems):
                rate = done / (time.time() - t0)
                eta = (len(workitems) - done) / rate if rate else 0
                print(f"  [{done}/{len(workitems)}]  elapsed={time.time()-t0:.0f}s  ETA={eta:.0f}s")

    if not per_image:
        sys.exit("No images processed successfully.")

    # ---- image-level split (deterministic via --seed) -----------------------
    import random
    rng = random.Random(args.seed)
    shuffled = sorted(per_image, key=lambda r: r[0])     # sort first so order is reproducible
    rng.shuffle(shuffled)
    n_test_imgs = int(round(len(shuffled) * args.test_fraction))
    test_imgs = shuffled[:n_test_imgs]
    train_imgs = shuffled[n_test_imgs:]

    def stack(rows):
        if not rows:
            return (np.empty((0, N_FEATURES), dtype=np.float32),
                    np.empty((0,), dtype=np.int8))
        return (np.concatenate([r[1] for r in rows], axis=0),
                np.concatenate([r[2] for r in rows], axis=0))

    X_train, y_train = stack(train_imgs)
    X_test, y_test = stack(test_imgs)
    n_pos_train = int(y_train.sum()); n_neg_train = int(len(y_train) - n_pos_train)
    n_pos_test = int(y_test.sum());   n_neg_test  = int(len(y_test) - n_pos_test)

    print()
    print(f"Images: {len(per_image)} total -> {len(train_imgs)} train + {len(test_imgs)} test "
          f"(seed={args.seed}, test_fraction={args.test_fraction})")
    print(f"Train: {len(X_train):>5} candidates  ({n_pos_train} pos / {n_neg_train} neg)")
    print(f"Test : {len(X_test):>5} candidates  ({n_pos_test} pos / {n_neg_test} neg)")
    print(f"Proposer recall ceiling (all images): {gt_recalled_total}/{total_gt} "
          f"({gt_recalled_total/total_gt*100:.1f}%)")
    if n_pos_train == 0:
        sys.exit("No positives in TRAIN split. Decrease --test-fraction or check labels.")

    pipe = build_pipeline(args.model)

    print()
    print(f"Cross-validating on TRAIN ({args.model}, 5-fold stratified)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_validate(pipe, X_train, y_train, cv=skf,
                        scoring=("precision", "recall", "f1", "roc_auc"),
                        n_jobs=1)
    for k in ("test_precision", "test_recall", "test_f1", "test_roc_auc"):
        m = cv[k]
        print(f"  {k[5:]:>10s}: mean={m.mean():.3f}  std={m.std():.3f}  "
              f"folds={['%.3f' % x for x in m]}")

    print()
    print("Fitting final model on TRAIN split...")
    pipe.fit(X_train, y_train)

    # ---- honest held-out test metrics ---------------------------------------
    test_metrics = None
    if len(X_test) and n_pos_test:
        from sklearn.metrics import (precision_score, recall_score, f1_score,
                                     roc_auc_score)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        test_metrics = {
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
            "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc":   float(roc_auc_score(y_test, y_proba)),
            "n_test_images": len(test_imgs),
            "n_test_candidates": int(len(X_test)),
            "n_test_positives": n_pos_test,
        }
        print()
        print("Held-out TEST metrics (model never saw these images):")
        for k in ("precision", "recall", "f1", "roc_auc"):
            print(f"  {k:>10s}: {test_metrics[k]:.3f}")
    else:
        print("\n(no test set — set --test-fraction > 0 to get held-out metrics)")

    train_filenames = [r[0] for r in train_imgs]
    test_filenames = [r[0] for r in test_imgs]
    meta = {
        "model": args.model,
        "feature_names": list(FEATURE_NAMES),
        "n_total_images": len(workitems),
        "n_train_images": len(train_imgs),
        "n_test_images": len(test_imgs),
        "n_train_candidates": int(len(X_train)),
        "n_train_positives": n_pos_train,
        "n_train_negatives": n_neg_train,
        "proposer_recall_ceiling": gt_recalled_total / total_gt,
        "split_seed": args.seed,
        "test_fraction": args.test_fraction,
        "train_filenames": train_filenames,
        "test_filenames": test_filenames,
        "cv_metrics": {k[5:]: {"mean": float(v.mean()), "std": float(v.std())}
                       for k, v in cv.items() if k.startswith("test_")},
        "test_metrics": test_metrics,
        "feature_scale_mean": pipe["scaler"].mean_.tolist(),
        "feature_scale_std": pipe["scaler"].scale_.tolist(),
        # default decision threshold; allergo_core will read it but it's tunable.
        "score_threshold": 0.5,
    }
    joblib.dump({"pipeline": pipe, "meta": meta}, args.out)
    print(f"Saved {args.out}")

    # Sidecar JSON for evaluate.py / other tools to look up the split.
    # Placed next to the classifier (same dir as args.out).
    split_path = Path(args.out).parent / "test_split.json"
    import json
    split_path.write_text(json.dumps({
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "train": train_filenames,
        "test": test_filenames,
    }, indent=2))
    print(f"Saved {split_path}")

    # Show feature importances (interpretable for LR, also useful for RF).
    print()
    if args.model == "logreg":
        coefs = pipe["clf"].coef_[0]
        order = np.argsort(-np.abs(coefs))
        print("Feature coefficients (sorted by |abs| — positive coef pushes toward 'positive'):")
        for j in order:
            print(f"  {FEATURE_NAMES[j]:>18s}  {coefs[j]:+.3f}")
    else:
        imp = pipe["clf"].feature_importances_
        order = np.argsort(-imp)
        print("Feature importances (RF, sorted desc):")
        for j in order:
            print(f"  {FEATURE_NAMES[j]:>18s}  {imp[j]:.3f}")


if __name__ == "__main__":
    main()
