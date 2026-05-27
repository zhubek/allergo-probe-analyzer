# What the red-circle annotations mark

The source images carry human annotations: red circles drawn around certain
objects (1–4 per image, **35 circles across 20 images**). This note records how
we figured out *what* the annotator was marking, so the rule isn't re-guessed.

## The question
"What is special about the dots inside the circles?" — i.e. what rule separates
an annotated object from the thousands of ordinary dots?

## Hypotheses tested

### ❌ "A cluster of 3+ dots grouped together"
Intuitive from the montage (circled objects *look* like little clumps), but it
**failed** when tested:

- Requiring a cluster of ≥3 distinct detected dots → **0–31% recall** across all
  gap settings. Requiring ≥2 → 9–66%. Only `≥1` (i.e. "any dark dot") reached
  100%, which is no discriminator at all (10k–14k objects).
- **Local dot density around circles equals background:** ~1.3 dot-neighbours
  within 40 px both inside circles and at random dots. Annotated spots are *not*
  in denser regions.
- Inside a typical circle the detector sees a median of **1** blob, not 3.

Conclusion: the clumpy *appearance* is real to the eye, but at the pixel level
those clumps are one or two merged blobs — not three countable dots. Dot-count
is the wrong rule.

### ✅ "The largest, darkest dark-blue blobs"
This matched strongly:

| Metric | Circled objects | Ordinary dots |
|---|---|---|
| blob area | ~358 px (median 347) | ~106 px (median 93) |
| luma (lower = darker) | ~95 | ~109 |
| red channel | ~37 | ~55 |

- Circled blobs sit at the **92nd size-percentile on average (median 99th)**
  within their image.
- **86%** of circles land on a **top-10%-largest** blob; **89%** top-25%; only
  **6%** on a below-median blob.

So the annotator consistently circled the **biggest, most strongly-stained
dark-blue blobs** — whether that's one large cell or several fused. Size
captures this; dot-counting does not.

## Brown/orange objects
A few circles in the montage bordered brown/orange streaks rather than blue
clumps. These are **artefacts**, treated as non-targets. The detection mask's
blueness requirement (`B − R > 25`) excludes them automatically, and they
account for most of the ~14% of annotations the size rule misses.

## Validation
`detect_clusters` (large-blob rule, `MIN_BLOB_AREA=250`) achieves **86% recall**
(30/35) against the human circles — vs **3%** for the dot-count rule. On at least
one image the detector independently boxed the exact object the human circled.
Reproduce with `validate_clusters.py` (needs the full-res images present).

## Caveats
- 35 annotations is a **small sample**. "Size" is a strong statistical signal,
  not necessarily a rule the annotator consciously applied.
- With more annotation data, the size heuristic could be replaced by a trained
  classifier using the features in `circled_dots_analysis.csv`.
