# Water body detection from Sentinel-2 (NDWI / MNDWI + Decision Tree)

Maps surface water (lagoon, creeks, rivers, reservoirs, ponds) from Sentinel-2 imagery with a Decision Tree,
and measures how good the map is. The accuracy check does not stop at the training labels: it also scores the
model against points you label yourself from high-resolution imagery.

SIWES Project 1 of 3 (Interspatial Technologies). Author: Iyinoluwa Don-Taiwo.

## SIWES portfolio

| # | Project | Status |
|---|---|---|
| 1 | Water body detection: Sentinel-2, NDWI/MNDWI, Decision Tree (this repo) | Built, ready to run |
| 2 | Building footprint extraction: U-Net on SpaceNet | Planned |
| 3 | Wildfire risk prediction: fused vegetation, thermal and weather data | Planned |

## Results

<!-- RESULTS-START -->
_Not run yet._ After the pipeline finishes, paste the tables from `outputs/summary.md` here and add the two
figures from `outputs/maps/` (`water_dry.png`, `change_dry_to_wet.png`). Do not paste numbers from anywhere else.
<!-- RESULTS-END -->

## How it works

```
Google Earth Engine (server side)                     This repo (your session / laptop)
─────────────────────────────────                     ──────────────────────────────────────────────
Sentinel-2 L2A  ──┐                                    labels.py    GSW (+ Dynamic World) -> water / land / ignore
Cloud Score+ mask ├─> seasonal median composite ──┐    sampling.py  balanced points, spatial blocks, block split
JRC GSW labels   ─┤                               ├──> model.py     Decision Tree, spatial CV, 1-SE rule
Dynamic World    ─┘   export GeoTIFF to Drive ────┘    evaluate.py  hold-out, baselines, bootstrap CIs
                                                       reference.py your independent points, area-weighted accuracy
                                                       predict.py / change.py   water map per season + change map
```

Choices that differ from a basic NDWI tutorial (reasons in [docs/METHODOLOGY.md](docs/METHODOLOGY.md)):

- **Cloud Score+** masking instead of QA60, with a seasonal median composite.
- **Labels from JRC Global Surface Water**, keeping only confident pixels away from shorelines. A second label mode
  also requires Dynamic World to agree, to catch land reclaimed after GSW's 2021 end date.
- **Spatial block split** for train/test and for cross-validation. A random pixel split leaks near-identical
  neighbours into the test set and inflates accuracy.
- **Tree depth tuned** with spatial CV and the one-standard-error rule, then compared with NDWI>0, MNDWI>0 and Otsu
  thresholds so you can see what the learned model adds.
- **Optional independent validation**: ~220 reference points drawn from the map, labelled blind by you, scored with
  area weights (Olofsson et al. 2014) and bootstrap confidence intervals.

## Repository layout

| Path | What it is |
|---|---|
| `config.yaml` | Every setting. Already chosen; the notebook fills in your project ID. |
| `notebooks/waterdet_colab.ipynb` | Guided run: Earth Engine export, pipeline, figures. Start here. |
| `src/waterdet/` | The package (see diagram above, plus `gee_export.py`, `io.py`, `indices.py`, `plots.py`). |
| `tests/` | 50 tests on a synthetic study area with known truth. No Earth Engine needed. |
| `app/` | Streamlit app (map explorer, seasonal change, model rules). See `app/README.md`. |
| `docs/HOW_TO_RUN.md` | Plain steps to run it. `OPTIONAL_ACCURACY_CHECK.md` covers the optional true-accuracy step. |
| `docs/METHODOLOGY.md` | Method, deviations from the proposal, limitations. |
| `data/` | Exported GeoTIFFs and your reference points. Rasters are git-ignored. |
| `outputs/` | Maps, figures, metrics. Git-ignored except `.gitkeep`. |

## Quick start

**Run it in Colab (recommended, nothing to install).** Follow [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md): upload
`notebooks/waterdet_colab.ipynb` to <https://colab.research.google.com>, type your Earth Engine project ID in the one
box, click **Runtime > Run all**, and approve the two Google pop-ups. The notebook carries its own copy of the code.

**Run it locally (Windows / VS Code, optional).**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[gee,dev]"
# set gee.project in config.yaml, then:
python -m waterdet.gee_export export --config config.yaml   # starts Drive exports
# when they finish, download s2_dry.tif and s2_wet.tif from the Drive folder into data\
python -m waterdet.pipeline --config config.yaml --data-dir data --out-dir outputs
python -m pytest                                              # optional: run the test-suite
```

## Presentation and app

Step 8 of the notebook writes `outputs/presentation/` (slide-ready images, `key_numbers.csv`, suggested slide order) and
`outputs/streamlit_data/` (copy into `app/data/`, then `streamlit run app/streamlit_app.py`).

## Outputs

| File | Meaning |
|---|---|
| `outputs/summary.md`, `summary.json` | All metrics, notes and warnings, provenance (package versions, input hashes). |
| `outputs/maps/water_<season>.tif` | 0 = land, 1 = water, 255 = no data (too few clear observations). 10 m, UTM. |
| `outputs/maps/change_dry_to_wet.tif` | 0 stable land, 1 stable water, 2 gain, 3 loss, 255 no data. |
| `outputs/mode_<label mode>/` | Model, readable tree rules, CV table, confusion matrix, tree and importance figures. |
| `outputs/reference_frame/` | CSV and KML of the points to label, plus the strata weights. |

## Known limits (short list, full version in the methodology)

- Training labels are 1984-2021 water history at 30 m; the imagery is 2025-2026 at 10 m. Held-out agreement with
  those labels is **not** accuracy against truth. Only the reference-point section is.
- Seasonal water cannot be scored against permanent-water labels. The seasonal change map is checked visually and by
  the reference points, not by the hold-out.
- SWIR (B11) is 20 m resampled to 10 m, so MNDWI is blockier than NDWI.

## Data and references

Sentinel-2 L2A (Copernicus / ESA); Cloud Score+ (Google); JRC Global Surface Water v1.4 (Pekel et al. 2016,
*Nature* 540:418-422); Dynamic World (Brown et al. 2022, *Scientific Data* 9:251); NDWI (McFeeters 1996,
*Int. J. Remote Sens.* 17:1425-1432); MNDWI (Xu 2006, *Int. J. Remote Sens.* 27:3025-3033); accuracy assessment
(Olofsson et al. 2014, *Remote Sens. Environ.* 148:42-57).

## Licence

MIT. See `LICENSE`.
