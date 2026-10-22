# Water body detection with Sentinel-2 and a Decision Tree

SIWES Project 1 maps water around Lagos Lagoon using five Sentinel-2 bands, NDWI and MNDWI. It learns interpretable rules from historical JRC Global Surface Water labels and compares dry and wet seasonal composites.

The implementation covers the proposal's core workflow. Independent manual spot-checks are still outstanding. Recorded scores measure agreement with historical labels, not verified present-day accuracy. See [proposal alignment and review](docs/PROJECT_REVIEW.md).

## Start here

| What you want | Where to go |
|---|---|
| Understand the method | [Methodology](docs/METHODOLOGY.md) |
| See the recorded Lagos results | [Result summary](results/presentation/summary.md) and [slide images](results/presentation/) |
| Run without a Google account | Offline demonstration below |
| Run on real imagery | [Run guide](docs/HOW_TO_RUN.md) |
| Prepare your presentation | [Presentation guide](docs/PRESENTATION_GUIDE.md) |
| Label independent validation points | [Reference guide](docs/OPTIONAL_ACCURACY_CHECK.md) |
| Inspect fixes and remaining gaps | [Project review](docs/PROJECT_REVIEW.md) |

## What the program does

1. Earth Engine masks cloudy Sentinel-2 observations and exports a seasonal median composite.
2. Python computes NDWI = (green - NIR) / (green + NIR) and MNDWI = (green - SWIR) / (green + SWIR).
3. JRC occurrence supplies confident water and land labels. Shoreline and ambiguous pixels are excluded. A second mode also requires Dynamic World agreement.
4. Whole spatial blocks are separated for training and testing. Grouped cross-validation tunes the Decision Tree.
5. The tree is compared with fixed-zero and Otsu thresholds on the same held-out samples.
6. Water maps, common-coverage seasonal changes, model rules, metrics, error samples and figures are saved.

Class codes: water maps use **0 land, 1 water, 255 no data**. Change maps use **0 stable land, 1 stable water, 2 gain, 3 loss, 255 no data**.

## Install and try it offline

From this repository in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[gee,dev,app]"
.\.venv\Scripts\python -m waterdet.demo --out-dir outputs/demo
.\.venv\Scripts\python -m pytest -ra --junitxml=outputs/tests.xml
.\.venv\Scripts\python -m streamlit run app/streamlit_app.py
```

The demo creates **artificial imagery**, trains both label modes and writes maps, figures, a log, and checks against known synthetic truth. It does not replace the saved Lagos results or app assets. The app displays the recorded Lagos run by default. To view demo outputs instead:

```powershell
$env:WATERDET_APP_DATA = (Resolve-Path outputs/demo/streamlit_data).Path
.\.venv\Scripts\python -m streamlit run app/streamlit_app.py
```

## Run with real Sentinel-2 imagery

Set `gee.project` in `config.yaml` to your registered Earth Engine project ID. Authentication is interactive.

```powershell
.\.venv\Scripts\python -m waterdet.gee_export preflight --config config.yaml
.\.venv\Scripts\python -m waterdet.gee_export export --config config.yaml
.\.venv\Scripts\python -m waterdet.gee_export status --config config.yaml
# Download the completed s2_dry.tif and s2_wet.tif exports from Drive into data/.
.\.venv\Scripts\python -m waterdet.pipeline --config config.yaml --data-dir data --out-dir outputs/real_run --presentation
```

Alternatively upload [the maintained Colab notebook](notebooks/waterdet_colab.ipynb) to Colab. It embeds a generated copy of the package for convenience. Maintainers must run `python scripts/sync_notebook.py` after source changes. Historical executed notebooks are kept under `notebooks/archive/`, not used as entry points.

The default configuration and the recorded run differ. To reconstruct the recorded settings, use [recorded_run_config.json](results/review/recorded_run_config.json), which the config loader also accepts, and replace its redacted project ID before exporting. Matching settings alone cannot guarantee identical imagery from a changing remote catalogue; input hashes identify the original files.

## Recorded Lagos results

These values were reproduced from the supplied Lagos GeoTIFFs during this review. The full run summary, input hashes, configuration, logs, and test records are in `results/review/`.

| Measure | Recorded result |
|---|---:|
| Primary model | GSW labels; depth 2; 4 leaves |
| Spatial held-out agreement | 98.52% |
| Kappa | 0.9704 |
| Water F1 | 0.9855 |
| Best threshold agreement, MNDWI Otsu | 98.79% |
| Dry mapped water | 230.10 km² |
| Wet mapped water | 236.88 km² |
| Gain on common valid coverage | 10.56 km² |
| Loss on common valid coverage | 3.80 km² |
| Independent labelled validation | Not supplied |

Dry dates: 1 December 2025 to 1 March 2026. Wet dates: 1 May to 17 October 2026. End dates are exclusive. The recorded run used at least two clear observations and a 90% scene-cloud prefilter; defaults use three and 60%.

The tree does not beat the best threshold in this run. Its contribution is a reproducible, interpretable classification workflow. Different label modes use different samples, so their scores are not a controlled comparison of accuracy. Seasonal changes are mapped differences, not proof of flooding or causation.

![Recorded dry-season detection](results/presentation/03_before_after_dry.png)

## Outputs and repository layout

| Path | Purpose |
|---|---|
| `src/waterdet/` | Maintained pipeline, export, diagnostics and demo code |
| `tests/` | Unit, end-to-end, offline Earth Engine graph and Streamlit interaction tests |
| `config.yaml` | Defaults for new runs |
| `data/` | Local input GeoTIFFs and optional reference points |
| `outputs/<run>/summary.json` | Metrics, warnings, exact configuration and input hashes |
| `outputs/<run>/run.log` | CLI progress and failure tracebacks |
| `outputs/<run>/maps/` | GIS rasters and map images |
| `outputs/<run>/mode_*/` | Model, CV table, rules, confusion matrix, spatial validation image and sample CSVs |
| `outputs/<run>/presentation/` | Slide-ready figures and key numbers |
| `outputs/<run>/streamlit_data/` | App assets generated from that run |
| `results/presentation/` | Preserved real-run presentation evidence |
| `results/review/` | Review logs, reproducibility records and additional charts |
| `app/data/` | Deployment copy of the recorded app assets |

The supplied `data/s2_dry.tif` and `data/s2_wet.tif` files are present locally, but are excluded from Git because together they exceed 250 MB. A fresh clone needs its own copies in `data/` before it can reproduce the real run. Large generated working outputs are also ignored by Git. Deployment assets are intentional copies; regenerate them together from one run to avoid mismatches. Use a new output folder for each run. Do not load untrusted `model.joblib` files.

## Limits and next steps

The main outstanding task is independently labelling the generated reference points using appropriately dated imagery. Historical 30 m labels and a 10 m output grid do not make a verified 10 m truth dataset. B11 is natively 20 m. Shadows, shorelines, tidal water, mangrove and newly reclaimed land need particular attention. Spatial block splitting reduces leakage but does not guarantee independence across adjacent block boundaries. The test suite and offline demo run without the large source rasters. Live Earth Engine exports require access to your account.

## Data attribution and licence

Code: MIT ([LICENSE](LICENSE)). Imagery and reference data retain their providers' terms.

- [Sentinel-2 L2A harmonized](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED), Copernicus/ESA.
- [Cloud Score+](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED), Google.
- [JRC Global Surface Water v1.4](https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_GlobalSurfaceWater), EC JRC/Google; Pekel et al. (2016).
- [Dynamic World](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1), Google/WRI; Brown et al. (2022).
