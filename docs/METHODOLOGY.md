# Methodology

Everything here describes what the code in `src/waterdet/` does. Numbers in brackets are defaults from `config.yaml`.

## 1. Imagery

- **Source:** Sentinel-2 L2A surface reflectance, `COPERNICUS/S2_SR_HARMONIZED` on Earth Engine. The harmonized
  collection removes the 1000-count offset that processing baseline 04.00 introduced after 2022-01-25.
- **Bands:** B2 blue, B3 green, B4 red, B8 NIR (all 10 m), B11 SWIR-1 (20 m, resampled to 10 m on export), scaled to 0-1.
- **Cloud mask:** Cloud Score+ (`GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`), band `cs_cdf` >= 0.60 [`cloud_score_threshold`].
  Google suggests 0.50-0.65; higher removes thin cloud, haze and cirrus shadow. Scenes with `CLOUDY_PIXEL_PERCENTAGE`
  of 60 or more are dropped first [`max_scene_cloud_pct`], which is only a cheap pre-filter.
- **Composite:** per-pixel median of the clear observations in the season window. A pixel needs at least 3 clear
  observations [`min_valid_obs`], otherwise it is written as no-data and never guessed.
- **Export:** one GeoTIFF per season on a 10 m UTM grid (zone from the AOI centre), bands
  `B2 B3 B4 B8 B11 n_obs gsw_occurrence gsw_max_extent dw_water`, masked pixels = -9999. Both seasons share one grid,
  which the pipeline checks.

## 2. Features

```
NDWI  = (B3 - B8)  / (B3 + B8)      McFeeters 1996   green vs NIR
MNDWI = (B3 - B11) / (B3 + B11)     Xu 2006          green vs SWIR-1
feature vector = [B2, B3, B4, B8, B11, NDWI, MNDWI]
```

NDWI separates water from vegetation well but confuses water with built-up surfaces. MNDWI suppresses built-up
areas better, but it uses the 20 m SWIR band and is fooled by dark shadow, which has low SWIR as well. The tree
sees both and can learn which to trust where.

## 3. Training labels

The labels come from JRC Global Surface Water v1.4 (GSW), not from hand digitising.

| Class | Rule | Then |
|---|---|---|
| water | GSW occurrence >= 75% [`min_gsw_occurrence`] | eroded by 3 px [`edge_buffer_px`] |
| land | GSW has never seen water here (`max_extent == 0` and `occurrence == 0`) | eroded by 3 px |
| ignore | everything else: seasonal water, shorelines, invalid pixels | not used |

- **Why erode:** GSW is 30 m (Landsat); the imagery is 10 m. Near a shoreline one GSW pixel spans three
  Sentinel-2 pixels, so labels there are wrong by up to that much. Only pixels at least 3 px inside a confident region are kept.
- **Why 75% and not 50%:** 50% still includes water that is only present part of the year. A tree trained on
  it learns "sometimes water" as "water".
- **Dynamic World mode (`gsw_dw`):** GSW ends in 2021, so land reclaimed since then is still "water" in GSW.
  In this mode water also needs Dynamic World's median water probability >= 0.5 [`dw_water_threshold`], and land needs
  < 0.5. Pixels where they disagree are ignored. Both modes are trained and compared.

## 4. Sampling and the train/test split

- Up to 5,000 points per class are drawn at random from the labelled pixels [`n_per_class`]. If a class has fewer
  labelled pixels, fewer are drawn and `class_weight="balanced"` in the tree compensates. Fewer than 60 per class is an error.
- Points are grouped into 2 km square blocks [`block_size_m`]. **Whole blocks** go to train or test (30% of points
  [`test_size`]), and both sides must hold at least 30 points of each class. A random seed sequence is retried
  up to 200 times to satisfy this.
- **Why:** neighbouring 10 m pixels are nearly identical. A random pixel split puts near-duplicates on both sides and
  reports accuracy that will not hold on new ground. The block split asks the harder, honest question: does the model work in
  places it has not seen?

## 5. Model and tuning

- `sklearn.tree.DecisionTreeClassifier`, Gini criterion, `class_weight="balanced"`. A tree was chosen because its
  rules can be printed (`tree_rules.txt`) and argued about, more compactly than a full ensemble. Random Forests also support interpretation methods.
- Grid: `max_depth` in {2,3,4,5,6,8,10} x `min_samples_leaf` in {1,10,50}, scored by F1 of the water class, with
  5-fold `GroupKFold` over the spatial blocks.
- **One-standard-error rule:** among all settings whose CV score is within one standard error of the best, take the
  simplest (smallest depth, then largest leaf size). This guards against picking a deep tree because of one lucky fold.
- The final tree is refit on all training points. Train accuracy is printed only to expose over-fitting; the
  pipeline warns when it exceeds hold-out accuracy by more than 0.03.

## 6. Baselines

Fixed `NDWI > 0` and `MNDWI > 0`, plus Otsu thresholds on each index (computed on all valid pixels, no labels
needed). They are scored on the same test points. If the tree does not clearly beat them, say so; a tuned threshold is
a legitimate answer to this problem.

## 7. Accuracy assessment: three different things

Confusion-matrix convention: rows = reference, columns = predicted, order [land, water]. Water precision is the
"user's accuracy"; water recall is the "producer's accuracy"; IoU = TP / (TP + FP + FN).

1. **Spatial hold-out against JRC-derived labels.** Reported with 95% CIs from resampling whole blocks (500 draws).
   This is *agreement with the labels* on unseen blocks. It cannot exceed the quality of the labels, it starts with up to equal numbers per class (the spatial split may change the ratio) (so precision, F1 and IoU depend on that ratio), and it never tests the ambiguous pixels that were
   deliberately left out of the labels.
2. **Independent reference points (your labels).** ~220 pixels drawn at random in three strata of the primary map:
   water core, land core and shoreline (within 2 px of the mapped edge). The strata have very different areas, so
   unweighted accuracy would be badly biased. The estimator is `p[i,j] = sum_h W_h * n_h(ref=i, pred=j) / n_h`, with
   `W_h` the stratum's share of mapped pixels (Olofsson et al. 2014). CIs come from resampling points within each
   stratum (500 draws). The strata are *frozen* in `data/reference_points.meta.json` so later runs score the same frame. **Report this only after independent labels are supplied.** Check imagery dates: a different date can introduce label error.
3. **Temporal transfer.** The same test points are scored on the other season's composite. The labels are static, so
   this only tests whether the model still works on permanent water and land under different conditions.

## 8. Change map

Dry and wet maps are compared pixel by pixel: stable land, stable water, gain (land to water), loss (water to
land), no data wherever either date is invalid. This is a product of two classifications, so errors in each add up. Treat small
patches as noise and check large changes against the imagery.

## 9. Differences from the proposal

| Proposal | This repository | Reason |
|---|---|---|
| QA60 cloud bit mask | Cloud Score+ `cs_cdf` >= 0.60 | QA60 is a coarse 60 m mask and is masked out in Earth Engine for 2022-01-25 to 2024-02-28, so it cannot be reused for earlier dates. Cloud Score+ also flags haze and shadow. |
| One cloud-free date | Seasonal median composite | A usable clear date over a humid coastal area cannot be guaranteed; a median of several clear looks also reduces noise. Cost: the map describes a season, not a day. |
| Labels: GSW occurrence > 50% | >= 75%, eroded 3 px, plus optional Dynamic World agreement | 30 m vs 10 m mismatch, partial-year water, and GSW ending in 2021. |
| Random 70/30 split of points | Spatial block split and block-grouped CV | Spatial autocorrelation. |
| `smileCart(maxNodes=50)` on Earth Engine | scikit-learn tree tuned by depth and leaf size, 1-SE rule | The proposal's own approach text says to tune depth with scikit-learn. |
| Accuracy from held-out GSW points | Same, plus independent reference points with area weighting | Scoring against the training source is circular. |
| NDWI / MNDWI as inputs | Also fixed-threshold and Otsu baselines | Shows what the learned tree adds. |
| Optional second date | Built in, with a change map | Part of the proposal's stated goal of a monitoring product. |

## 10. Limitations to state when presenting

- Labels are historical (1984-2021) and coarse (30 m); the imagery is 2025-2026 at 10 m. Hold-out accuracy is
  agreement with those labels, not truth.
- Seasonal and tidal water is excluded from training. Its accuracy is known only through the reference points.
- Dark building shadow, wet bare soil, mangrove and turbid water are the expected error sources. The shoreline
  stratum exists to measure edge error.
- B11 is 20 m; MNDWI is blockier than NDWI.
- Dropping `unsure` reference points makes accuracy look better. The report counts them.
- Reference points are judged from imagery of a different date than the composite. Dynamic features may be
  mislabelled.
- Median compositing hides short events (a flood lasting days). This is a seasonal product.
- Earth Engine collections change. The summary records package versions and input-file hashes, but not the version of
  Earth Engine's catalogue; re-exporting later may give slightly different composites.
- The Earth Engine export step cannot be run offline. Its parameters are tested against a mock, and its calls were
  checked against Google's documentation, but the first live run is the real test.

## 11. Reproducibility

Seeds are fixed [`seed: 42`]. `summary.json` stores the full config (project id redacted), package versions and
SHA-256 hashes of the input rasters. The test-suite builds a synthetic study area with known truth, including a shadow
trap for MNDWI and a reclaimed-land trap for GSW, and checks that the pipeline recovers it.
