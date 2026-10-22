# waterdet results

## Label mode `gsw` (training period `dry`)

- Tree: depth 2, 4 leaves, params {'max_depth': 2, 'min_samples_leaf': 50}
- Points: 6954 train / 3046 test (116 / 52 spatial blocks)
- Train overall accuracy: 0.9810 (shown only to expose over-fitting)

| Method (spatial hold-out) | Overall acc. [95% CI] | Kappa [95% CI] | Water F1 | Water IoU |
|---|---|---|---|---|
| **Decision Tree** | 0.9852 [0.969, 0.995] | 0.9704 [0.936, 0.990] | 0.9855 | 0.9714 |
| NDWI>0 (thr 0.000) | 0.9734 | 0.9468 | 0.9742 | 0.9496 |
| MNDWI>0 (thr 0.000) | 0.9839 | 0.9678 | 0.9842 | 0.9689 |
| NDWI Otsu (thr 0.012) | 0.9770 | 0.9540 | 0.9776 | 0.9561 |
| MNDWI Otsu (thr 0.066) | 0.9879 | 0.9757 | 0.9880 | 0.9763 |

_These scores measure agreement with JRC-derived labels on held-out spatial blocks, using an approximately equal per-class sample (the block split can shift the ratio). They are NOT independent truth, and water precision / F1 / IoU change with the class ratio. Independent, area-weighted reference validation is still needed for ground-truth accuracy._

## Label mode `gsw_dw` (training period `dry`)

- Tree: depth 2, 3 leaves, params {'max_depth': 2, 'min_samples_leaf': 1}
- Points: 6974 train / 3026 test (121 / 46 spatial blocks)
- Train overall accuracy: 0.9996 (shown only to expose over-fitting)

| Method (spatial hold-out) | Overall acc. [95% CI] | Kappa [95% CI] | Water F1 | Water IoU |
|---|---|---|---|---|
| **Decision Tree** | 0.9960 [0.992, 0.999] | 0.9921 [0.984, 0.998] | 0.9960 | 0.9921 |
| NDWI>0 (thr 0.000) | 0.9815 | 0.9630 | 0.9817 | 0.9640 |
| MNDWI>0 (thr 0.000) | 0.9937 | 0.9874 | 0.9937 | 0.9875 |
| NDWI Otsu (thr 0.012) | 0.9874 | 0.9749 | 0.9875 | 0.9753 |
| MNDWI Otsu (thr 0.066) | 0.9967 | 0.9934 | 0.9967 | 0.9934 |

_These scores measure agreement with JRC-derived labels on held-out spatial blocks, using an approximately equal per-class sample (the block split can shift the ratio). They are NOT independent truth, and water precision / F1 / IoU change with the class ratio. Independent, area-weighted reference validation is still needed for ground-truth accuracy._

## Water extent (primary model)

| Period | Water km² | Land km² | No data % |
|---|---|---|---|
| dry | 230.10 | 381.56 | 0.2 |
| wet | 236.88 | 375.34 | 0.1 |

### Change dry_to_wet

- stable land: 370.64 km²
- stable water: 226.27 km²
- water gain (land -> water): 10.56 km²
- water loss (water -> land): 3.80 km²
- compared_km2: 611.27 km²

## Notes and warnings

- No independent reference points found at data/reference_points.csv. Accuracy above is agreement with JRC-derived labels only. Optional: docs/OPTIONAL_ACCURACY_CHECK.md explains how to get a true accuracy figure.

## Provenance

- Python 3.13.16; packages: waterdet 0.1.0, numpy 2.1.3, scipy 1.16.3, pandas 2.2.3, scikit-learn 1.6.1, rasterio 1.5.2, earthengine-api 1.7.39
- Input `s2_dry.tif` sha256 `1779b694527fe1f9…`
- Input `s2_wet.tif` sha256 `eeef3e335953aa7d…`
- Full config snapshot (GEE project id redacted) is in `summary.json`.
