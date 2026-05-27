# Detection algorithm

All detection lives in `allergo_core.py`. Both detectors share one **dark-blue
mask**; they differ only in how they group and filter the result.

## Step 0 — the dark-blue mask

For each pixel compute:
- `luma = 0.299R + 0.587G + 0.114B` — brightness
- `blueness = B - R` — how blue (navy dots have B ≫ R)
- `field = gaussian(luma, σ=60)` — a smooth model of the local bright field

A pixel is **dark-blue** when **all** hold:

| Condition | Why |
|---|---|
| `field > 120` (`FIELD_MIN`) | We're in the bright field, **not** the dim vignette ring around the circular edge. |
| `field - luma > 70` (`DARK_DROP`) | The pixel is *clearly darker than its local surroundings* — a real dot, not faint texture. |
| `B - R > 25` (`BLUE_EXCESS`) | The pixel is blue. Also rejects **brown/orange** artefacts (warm hue → low B−R). |

### Why field-relative, not absolute darkness
An earlier version used an absolute cutoff (`luma < 90`). It failed: the
vignette darkens the field smoothly toward the edge (bright centre ~207 luma,
edge ~64), so an absolute threshold circled the **entire dim ring** instead of
dots. Measuring darkness *relative to the local smoothed field* fixes this — the
ring's field value is itself low, so `field > 120` excludes it while
`field - luma > 70` still fires on genuine dots anywhere in the bright area.

## `detect_dots` — every dot

1. Build the dark-blue mask.
2. **Split touching dots** with a distance-transform watershed: distance
   transform → light gaussian (σ=1) so each dot in a clump becomes a distinct
   peak → local maxima as markers (≥ `MIN_PEAK_DIST=4` px apart) →
   `watershed_ift`. Without this, a clump of 3 touching dots counts as 1.
3. Label, drop blobs outside `[DOT_MIN_AREA=10, DOT_MAX_AREA=4000]` px.
4. Return `{x, y, area, radius}` per dot (full-resolution pixel coordinates).

Typical full-res image: ~700–1100 dots.

## `detect_clusters` — positive large blobs

1. Build the same dark-blue mask (**no** watershed split — we want whole blobs).
2. Label connected components.
3. Keep blobs with area in `[MIN_BLOB_AREA=250, MAX_BLOB_AREA=8000]` px.
4. Return `{x, y, area, w, h}` per blob.

This implements the annotation rule (see `docs/ANALYSIS.md`): positives are the
**largest dark-blue blobs**, which register as single large connected components.

## Parameters (all in `allergo_core.py`)

| Name | Default | Effect |
|---|---|---|
| `FIELD_SIGMA` | 60 | Scale of the bright-field model. |
| `FIELD_MIN` | 120 | Higher → crops more of the vignette edge. |
| `DARK_DROP` | 70 | Higher → only darker dots count (fewer). |
| `BLUE_EXCESS` | 25 | Blueness floor; also the brown/orange reject. |
| `MIN_PEAK_DIST` | 4 | Lower → splits touching dots more aggressively. |
| `DOT_MIN/MAX_AREA` | 10 / 4000 | Single-dot size bounds. |
| `MIN_BLOB_AREA` | 250 | **Positive** cutoff: lower → mark medium blobs too. |
| `MAX_BLOB_AREA` | 8000 | Drop huge dark artefacts. |

> Calibrated on the original ~5440×3648 px images. Scale `*_AREA` and the
> `σ`/distance values roughly with resolution if you analyze resized images.
