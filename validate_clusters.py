#!/usr/bin/env python3
"""Validate detect_clusters against the human red-circle annotations.

For each image: find red circles, run the cluster detector, and check whether
each circle has a detected cluster centered inside it (recall). Also report how
many detected clusters fall outside any circle (the model finds more than were
annotated, which is expected since annotators mark only examples)."""
import glob, os
import numpy as np
from PIL import Image
from scipy import ndimage
import detect_clusters as DC


def find_circles(a):
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    red = (R > 140) & (G < 90) & (B < 90)
    red = ndimage.binary_dilation(red, iterations=6)
    lbl, n = ndimage.label(red)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        if len(xs) < 150:
            continue
        w = xs.max() - xs.min(); h = ys.max() - ys.min()
        if w < 25 or h < 25:
            continue
        out.append((int(xs.mean()), int(ys.mean()), max(w, h) // 2))
    return out


tot_circ = hit = 0
tot_clusters = 0
for p in sorted(glob.glob("снимки с разметками/снимки с разметками/*.png")):
    a = np.asarray(Image.open(p).convert("RGB")).astype(int)
    circles = find_circles(a)
    _, clusters = DC.detect_clusters(p)
    tot_clusters += len(clusters)
    cx = np.array([c["x"] for c in clusters]); cy = np.array([c["y"] for c in clusters])
    matched = 0
    for (ox, oy, orad) in circles:
        if len(clusters):
            d = np.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
            if d.min() <= orad + 8:   # a cluster centre lands inside the circle
                matched += 1
    tot_circ += len(circles); hit += matched
    print(f"{os.path.basename(p)}: circles={len(circles)} matched={matched} "
          f"clusters_found={len(clusters)}")

print(f"\nRECALL on annotations: {hit}/{tot_circ} = {hit/tot_circ:.0%}")
print(f"Total clusters detected across 20 images: {tot_clusters}")
