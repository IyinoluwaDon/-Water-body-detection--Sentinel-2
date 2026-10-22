# Project 1: Sentinel-2 Water Body Detection around Lagos Lagoon

This document is the full project brief for a reader, reviewer, or chatbot that needs to explain the project in a report or presentation. It covers the purpose, proposal alignment, data, method, implementation, results, reproduction steps, and limitations. It describes the code and the recorded run; it does not claim results that were not measured.

For current measured results, use [the recorded result summary](../results/presentation/summary.md) and [the reproducibility record](../results/review/latest_real_run.md). For an accessible slide order and individual figures, use the [presentation guide](PRESENTATION_GUIDE.md) and [presentation pack](../results/presentation/README_presentation.md). The [project review](PROJECT_REVIEW.md) lists implementation fixes and open scientific issues.

## Project at a glance

| Item | Description |
|---|---|
| Project | SIWES Project 1: Water Body Detection with NDWI and a Decision Tree |
| Study area | Lagos Lagoon and surrounding creeks, shoreline, built-up areas, and ponds; bounding box `[3.30, 6.40, 3.55, 6.60]` in longitude/latitude |
| Main question | Can Sentinel-2 bands and water indices support a clear, repeatable water/land map, and how does the mapped extent differ between two seasonal composites? |
| Main method | A tuned, interpretable scikit-learn Decision Tree trained against confident historical JRC Global Surface Water labels |
| Other comparisons | Dynamic World agreement label mode; NDWI and MNDWI fixed-zero and Otsu thresholds |
| Main products | Dry- and wet-period water maps, seasonal change map, model rules, validation figures, metrics, CSVs, and slide-ready PNGs |
| Current scientific status | The pipeline and saved real-data run are reproducible. Independent manual reference labels have not been supplied, so measured scores are agreement with historical labels, not verified ground-truth accuracy. |

## Why the project matters

Water is difficult to map consistently around a coastal city. The Lagos Lagoon area combines open water, narrow creeks, tidal edges, wet ground, vegetation, buildings, and dark shadows. Clouds and haze can obscure optical satellite observations, and a single image may not provide enough clear pixels over the full study area.

The project applies satellite remote sensing and a small machine-learning model to classify water and land. It also makes the processing steps inspectable: the model is a Decision Tree whose rules can be printed and reviewed. Comparing two seasonal composites gives a simple monitoring product and demonstrates how mapped water extent changes over time.

The project is a seasonal mapping and methods demonstration. It does not identify the date of a flood, establish why water changed, measure groundwater, or prove that every mapped change is real. Those claims would need different data and independent field or image interpretation.

## Aim and objectives

The aim is to build a reproducible water-body detection workflow for the Lagos Lagoon area using Sentinel-2 imagery, NDWI/MNDWI features, and a Decision Tree, then show the dry-to-wet seasonal mapping result.

The practical objectives are to:

1. Select a mixed water-and-land study area around Lagos Lagoon.
2. Obtain cloud-screened Sentinel-2 surface-reflectance imagery for two seasonal windows.
3. Calculate NDWI and MNDWI alongside selected raw spectral bands.
4. Create training labels from historical JRC Global Surface Water data, with an optional stricter Dynamic World agreement mode.
5. Train and tune a Decision Tree and assess it on spatially held-out areas.
6. Compare the tree with simple fixed and Otsu water-index thresholds.
7. Produce water maps, seasonal change estimates, interpretable model rules, diagnostics, and materials that can be used to explain the work.

## Alignment with the proposal

The implementation covers the core Project 1 workflow: a Lagos Lagoon AOI, Sentinel-2 imagery, water indices, JRC-derived labels, a tuned Decision Tree, accuracy metrics, maps, and an optional second-period comparison. The code adds grouped spatial validation, threshold baselines, uncertainty intervals, provenance, and a synthetic offline demonstration.

Some choices differ from the proposal's illustrative method. They are deliberate methodological changes and should be mentioned in a report or presentation:

| Proposal approach | Project implementation | Reason and implication |
|---|---|---|
| QA60 cloud bit mask | Cloud Score+ `cs_cdf` threshold, with a scene-cloud prefilter | Supports cloud and haze screening for the selected observations. |
| One cloud-free scene | Seasonal per-pixel median composite | More likely to provide clear coverage, but represents a season rather than a particular day. |
| JRC occurrence above 50% | JRC occurrence at least 75%, with a 3-pixel eroded core; optional Dynamic World agreement | Favors persistent, confident labels and reduces shoreline label noise at the cost of excluding ambiguous or seasonal pixels. |
| Random 70/30 pixel split | Whole 2 km spatial blocks held out; grouped cross-validation | Reduces leakage from nearby, correlated pixels. Nearby blocks may still share spatial structure. |
| Earth Engine `smileCart` illustration | scikit-learn Decision Tree, tuned with grouped folds and a one-standard-error rule | Provides a reproducible grid search and a compact tree with inspectable rules. |
| Report held-out agreement with JRC labels | Report that agreement with uncertainty, threshold baselines, and an optional independent-reference workflow | Makes clear that evaluation against labels from the same historical source is not ground truth. |
| One date | Dry and wet seasonal maps plus a common-valid-area change map | Extends the project to seasonal monitoring; the difference is not proof of a flood or its cause. |

The proposal's example dry window is 1 November 2025 to 28 February 2026 (exclusive end). The recorded run uses 1 December 2025 to 1 March 2026 (exclusive end). The default wet window is 1 June to 1 October 2026 (exclusive end), while the reproduced run uses 1 May to 17 October 2026 (exclusive end). Default and recorded configurations must not be confused.

## Study area, imagery, and labels

### Study area

The AOI is the rectangle `[3.30, 6.40, 3.55, 6.60]` in longitude/latitude. It is intended to include the lagoon, connected creeks, Atlantic shoreline, city surfaces, and small water bodies so that the model encounters both water and challenging land examples. Area calculations use the raster's projected grid and pixel size.

### Satellite inputs

Earth Engine exports Sentinel-2 Level-2A harmonized surface reflectance from `COPERNICUS/S2_SR_HARMONIZED`. Cloud Score+ (`GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`, band `cs_cdf`) screens observations. Whole scenes are first filtered by scene-cloud percentage; per-pixel Cloud Score+ then masks cloud, haze, and related low-quality observations. Each period is reduced to a per-pixel median, and pixels with too few valid looks become no-data.

The exported GeoTIFF stores these bands in order: blue B2, green B3, red B4, near-infrared B8, short-wave infrared B11, valid-observation count, JRC water occurrence, JRC maximum extent, and Dynamic World water probability. B2/B3/B4/B8 are native 10 m bands; B11 is native 20 m and is resampled onto the 10 m export grid. The pipeline checks that the two seasonal rasters have compatible projected grids and honors internal raster masks.

The local inputs used for the reproduced run are `data/s2_dry.tif` and `data/s2_wet.tif`. Their full SHA-256 values are in `results/review/latest_real_run_summary.json`. They are over 250 MB together and are intentionally excluded from Git, so a fresh clone needs its own copies to reproduce the real-data run. The app and slide assets are smaller outputs already included in the repository.

### Features and indices

The model input vector contains five reflectance bands and two normalized water indices:

```text
NDWI  = (B3 - B8)  / (B3 + B8)       green versus near-infrared
MNDWI = (B3 - B11) / (B3 + B11)      green versus short-wave infrared
Features = [B2, B3, B4, B8, B11, NDWI, MNDWI]
```

NDWI often distinguishes water from vegetation, while MNDWI can suppress built-up surfaces. Neither index is universally reliable: dark shadows, wet soil, turbid water, vegetation, and the coarser native B11 band can affect results. Zero denominators and invalid pixels are excluded rather than treated as land.

### Training labels

JRC Global Surface Water v1.4 supplies the main historical labels. It summarizes Landsat observations through 2021 at about 30 m resolution. The primary `gsw` mode labels pixels with occurrence at least 75% as confident water and pixels with zero occurrence and zero maximum extent as confident land. A 3-pixel erosion removes uncertain edges; seasonal, shoreline, missing, and otherwise ambiguous pixels are ignored.

The optional `gsw_dw` mode additionally requires agreement with Dynamic World's median water probability threshold. This provides a second labeling choice, not independent ground truth: Dynamic World also uses Sentinel-2, and the modes use different training and test samples.

## Processing and model workflow

1. **Export or load seasonal stacks.** Earth Engine can check access, preview the AOI, start exports, and report task status. The local pipeline reads completed GeoTIFFs without needing a live Earth Engine session.
2. **Validate the inputs.** Check expected bands, valid label values, internal masks, projected CRS, grid alignment, pixel dimensions, and area units.
3. **Build features and labels.** Calculate NDWI/MNDWI, remove invalid pixels, make confident GSW labels, and optionally make Dynamic World agreement labels.
4. **Sample and split spatially.** Draw up to 5,000 samples from each class using a fixed seed. Assign complete 2 km blocks to train or test (nominally 70/30), retrying split seeds when needed to keep both classes represented.
5. **Tune and fit the tree.** Try depths `{2, 3, 4, 5, 6, 8, 10}` and minimum leaf sizes `{1, 10, 50}` with five-fold `GroupKFold`, scoring water-class F1. Among settings within one standard error of the best score, select the simplest tree. Refit on the training points and save its text rules and plot.
6. **Evaluate.** Score on spatially held-out points and compare with `NDWI > 0`, `MNDWI > 0`, and Otsu thresholds for each index on the same test points. Whole-block bootstrap intervals quantify sampling uncertainty in the spatial holdout.
7. **Map both periods.** Apply the primary fitted model to valid pixels in each season, calculate water/land area, and write GeoTIFF and image outputs.
8. **Compare seasons.** Compare only pixels valid in both periods. Change codes are stable land, stable water, gain, loss, and no-data. Save change areas, diagnostic plots, spatial sample assignments, per-point predictions, provenance, and optional app/presentation assets.
9. **Optionally score independent references.** A stratified frame of water-core, land-core, and shoreline locations can be labelled by a person. With labels supplied, the pipeline reports area-weighted reference metrics and stratified uncertainty. No independent labels were supplied for the current recorded run.

The spatial split is more realistic than randomly splitting neighboring pixels, but it does not remove every source of spatial dependence. The labels also deliberately omit the ambiguous shoreline and seasonal pixels, so held-out scores do not measure performance on all AOI conditions.

## Recorded real-data result

The run was reproduced on 22 October 2026 using the local dry and wet rasters. The files matched the recorded hashes byte for byte. The full settings are in `results/review/recorded_run_config.json`; the current defaults in `config.yaml` differ.

| Measure | Recorded value |
|---|---:|
| Primary label mode | GSW (`gsw`) |
| Primary tree | depth 2, 4 leaves, minimum leaf size 50 |
| Primary spatial holdout | 6,954 training and 3,046 test points, assigned by spatial blocks |
| Overall agreement with held-out JRC-derived labels | 98.52% (block-bootstrap 95% interval: 96.87–99.52%) |
| Kappa | 0.9704 (block-bootstrap 95% interval: 0.9362–0.9904) |
| Water F1 | 0.9855 (block-bootstrap 95% interval: 0.9652–0.9955) |
| Best primary-mode threshold baseline | MNDWI Otsu, 98.79% agreement on the same held-out points |
| Dry mapped water | 230.10 km² |
| Wet mapped water | 236.88 km² |
| Water gain on common valid coverage | 10.56 km² |
| Water loss on common valid coverage | 3.80 km² |
| Net mapped change (gain minus loss) | about +6.76 km² |
| Independently labelled reference points | 0 |

The tree's 98.52% is agreement with historical labels, not verified real-world accuracy. The best threshold baseline scores slightly higher on these test points, so the result does not show that the tree is more accurate than a simple index threshold. The tree's value is that it gives a compact, inspectable set of rules and a reproducible full mapping workflow.

The second `gsw_dw` mode's held-out tree agreement is 99.60%, but it uses a different labeling rule and different samples. Do not compare that number directly with the primary GSW score as though it were a controlled model comparison. The primary tree's feature importance assigns the split to B11 in this run; this is a property of this fitted tree and these labels, not a universal claim that B11 is always the most useful water feature.

The change map compares seasonal classifications on common valid coverage: 370.64 km² stable land, 226.27 km² stable water, 10.56 km² water gain, and 3.80 km² water loss across 611.27 km². These are mapped differences between seasonal composites, not confirmed flood extent or causal findings.

## Validation, evidence, and project status

Validation currently has three parts:

- **Spatial holdout:** agreement with JRC-derived labels on blocks not used to fit the tree, including block-bootstrap intervals.
- **Threshold baselines:** fixed and data-derived index thresholds scored on the same held-out points within a label mode.
- **Synthetic end-to-end tests:** artificial imagery with known truth exercises the pipeline; the demo achieved 99.87% dry and 99.85% wet agreement with its generated truth. These values validate the synthetic test, not the Lagos map.

The optional independent validation process is described in [OPTIONAL_ACCURACY_CHECK.md](OPTIONAL_ACCURACY_CHECK.md). It asks a person to label a stratified set of points using appropriately dated imagery. Until that is done, the project has no empirical ground-truth accuracy estimate. Further limitations include historical 30 m labels, 20 m native B11, tidal or seasonal water exclusion, shadows and wet soil, errors around coasts, and possible spatial dependence across neighboring blocks.

The complete latest test result is **61 passed** with 20 Rasterio pending-deprecation warnings. See `results/review/final_pytest.log` and `final_pytest.xml`. Tests cover index and metric calculations, reference data checks, label generation, spatial splitting and model tuning, raster integrity, pipeline end-to-end behavior, offline Earth Engine graph construction, and app/presentation interactions. Live Earth Engine authentication and export still require the user's registered project and account.

## How to run and reproduce

### Install and run the synthetic demo

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[gee,dev,app]"
.\.venv\Scripts\python -m waterdet.demo --out-dir outputs/demo
.\.venv\Scripts\python -m pytest -ra --junitxml=outputs/tests.xml
```

The demo uses artificial data and does not contact Earth Engine. It writes outputs under `outputs/demo/`.

### Reproduce the recorded Lagos run

Put the two GeoTIFF inputs in `data/`, then run:

```powershell
.\.venv\Scripts\python -m waterdet.pipeline --config results/review/recorded_run_config.json --data-dir data --out-dir outputs/real_run --presentation
```

The recorded configuration has the Earth Engine project ID redacted. A local pipeline rerun uses the downloaded GeoTIFFs and does not need a live Earth Engine export. To create new composites, set `gee.project` in `config.yaml`, authenticate, and follow [HOW_TO_RUN.md](HOW_TO_RUN.md).

### View the app and find deliverables

The Streamlit viewer can show the committed recorded assets without connecting to Earth Engine:

```powershell
.\.venv\Scripts\python -m streamlit run app/streamlit_app.py
```

The main folders are:

| Folder | Contents |
|---|---|
| `src/waterdet/` | Pipeline, export integration, feature calculation, labels, model, evaluation, mapping, diagnostics, and demo. |
| `tests/` | Automated unit, integration, and app checks. |
| `notebooks/waterdet_colab.ipynb` | Single maintained Colab entry point, with an embedded package copy. |
| `results/presentation/` | Recorded-run slide images, summary, key numbers, and recommended slide sequence. |
| `results/review/` | Recorded-run settings, hashes, metrics, real-run log, synthetic demo record, and final test reports. |
| `outputs/real_run/` | Full local run products: rasters, plots, tables, models, logs, and app assets. Ignored by Git. |
| `app/data/` | Smaller committed app assets that can be viewed in the Streamlit app. |

The Colab notebook embeds package code for convenience. After changing maintained source, refresh it with `python scripts/sync_notebook.py` before relying on the notebook copy.

## Important files

- `README.md`: short repository start page.
- `config.yaml`: defaults for future exports and runs.
- `docs/METHODOLOGY.md`: detailed formulas and scientific choices.
- `docs/HOW_TO_RUN.md`: Earth Engine and Colab instructions.
- `docs/PROJECT_REVIEW.md`: proposal alignment, corrections, and outstanding gaps.
- `docs/PRESENTATION_GUIDE.md`: suggested slide story and figure choices.
- `results/presentation/summary.md` and `summary.json`: recorded metrics in human- and machine-readable formats.
- `results/review/latest_real_run.md` and `latest_real_run_summary.json`: real-data reproduction record and detailed result.
- `results/review/final_pytest.log` and `final_pytest.xml`: latest complete test run.
- `LICENSE`: MIT license for project code; source imagery and reference data retain provider terms.

## Data attribution and references

- Sentinel-2 L2A harmonized: [Earth Engine data catalogue](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED), Copernicus/ESA.
- Cloud Score+: [Earth Engine data catalogue](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED), Google.
- JRC Global Surface Water v1.4: [Earth Engine data catalogue](https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_GlobalSurfaceWater), EC JRC/Google; see Pekel et al. (2016).
- Dynamic World: [Earth Engine data catalogue](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1), Google/WRI; see Brown et al. (2022).
- NDWI: McFeeters (1996). MNDWI: Xu (2006). Area-adjusted reference accuracy methodology: Olofsson et al. (2014), when independent labels are collected.

## Guidance for future report and slide generation

Use this overview to explain the project, then verify quoted values against the recorded result files. Keep these distinctions explicit:

1. Say **agreement with historical JRC-derived labels**, not “98.52% accurate,” unless independent reference validation has been completed.
2. The seasonal area differences are map comparisons; do not call them confirmed floods or attribute them to a cause.
3. State that MNDWI Otsu scored 98.79% on the primary mode's held-out points, slightly above the tree's 98.52%.
4. Keep the recorded run's dates and settings separate from the defaults in `config.yaml`.
5. Identify the `gsw_dw` results as a different label mode with a different sample set, not a direct apples-to-apples accuracy comparison.
6. Mark synthetic demo imagery and its scores as artificial test evidence, never as Lagos observations.

An accurate one-sentence description is: **“This project builds an interpretable, reproducible Sentinel-2 workflow to map seasonal water around Lagos Lagoon, training a Decision Tree on confident historical water labels and comparing the maps and model with spatial holdouts and simple water-index thresholds.”**
