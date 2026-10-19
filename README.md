# Water Body Detection from Sentinel-2 (NDWI / MNDWI + Decision Tree)

Maps surface water across the Lagos lagoon area from Sentinel-2 satellite imagery, using a Decision Tree whose rules
can be read and explained. It then measures how good the map is, compares the learned model against simple
threshold rules, and maps how water changes between the dry and wet seasons.

SIWES Project 1 of 3, Interspatial Technologies. Author: Iyinoluwa Don-Taiwo.

**Live app:** `<paste your Streamlit link here>`  |  **Executed notebook:** `notebooks/waterdet_colab_run_oct2026.ipynb`

---

## Contents

1. [The project in one minute](#1-the-project-in-one-minute)
2. [Results](#2-results)
3. [Key ideas in plain language](#3-key-ideas-in-plain-language)
4. [How the pipeline works](#4-how-the-pipeline-works)
5. [Design choices and why](#5-design-choices-and-why)
6. [Repository layout](#6-repository-layout)
7. [How to run it](#7-how-to-run-it)
8. [The Streamlit app](#8-the-streamlit-app)
9. [Outputs: what each file means](#9-outputs-what-each-file-means)
10. [Presentation guide](#10-presentation-guide-slides-and-talking-points)
11. [Report guide](#11-report-guide)
12. [Questions you may be asked](#12-questions-you-may-be-asked)
13. [Limitations](#13-limitations)
14. [Troubleshooting log](#14-troubleshooting-log-problems-met-and-fixed)
15. [Testing and reproducibility](#15-testing-and-reproducibility)
16. [Data, references, licence](#16-data-references-licence)

---

## 1. The project in one minute

**Problem.** Rivers, lagoons and ponds change with the seasons, and surveying them on the ground is slow. Satellites
photograph the same place every few days for free.

**Approach.**
1. Get Sentinel-2 images of the Lagos lagoon area for a dry season (Dec 2025 to Feb 2026) and a wet season (2026).
2. Remove clouds and combine many images into one clean "seasonal median" picture per season.
3. Compute two water indices (NDWI and MNDWI) that make water stand out.
4. Train a Decision Tree to label every 10 m pixel as water or land. Training labels come from the JRC Global Surface
   Water map, not hand drawing.
5. Test the tree on parts of the map it never saw, and compare it with plain threshold rules.
6. Compare dry and wet maps to show where water was gained or lost.

**Deliverables.** Water maps and a change map (GeoTIFF), accuracy tables, a printable set of tree rules, a slide-ready
image pack, and a Streamlit web app.

**Study area.** Lagos lagoon, creeks and Atlantic shoreline. Bounding box `[3.30, 6.40, 3.55, 6.60]` (lon/lat),
about 25 km by 22 km.

---

## 2. Results

> **Before you present:** the numbers below are from the run on the dry season and the label-agreement tests. Open
> `results/summary.md` (copied from your Colab run) and check every figure against it. Fill in the wet-season row from
> that file. If the wet season is mostly no-data because of cloud, say so (see the note under the table). Never quote a
> number that is not in `summary.md`.

![Dry season: image vs detected water](results/presentation/03_before_after_dry.png)
![Dry to wet change](results/presentation/06_dry_wet_change.png)

### Water extent

| Season | Water area (km²) | Land area (km²) | No data |
|---|---|---|---|
| Dry (Dec 2025 to Feb 2026) | 231.28 | 374.77 | 1.1% |
| Wet (2026) | _fill from `results/summary.md`_ | _fill_ | _fill_ |

Note on the wet season: Lagos is cloudy from May to October. The first wet export had no pixels with enough clear
observations (100% no data). The window was widened (May to mid-Oct 2026), the scene cloud filter relaxed to 90%, and the
minimum clear observations lowered from 3 to 2. A pixel that still has too few clear looks is left as no-data, never guessed.

### Model accuracy (spatial hold-out)

Held-out 2 km blocks the model never trained on. Two label modes were trained and compared.

| Method | Label mode `gsw` | Label mode `gsw_dw` |
|---|---|---|
| **Decision Tree** (depth 2, chosen by spatial CV and the one-standard-error rule) | 0.9878 | 0.9980 |
| MNDWI with Otsu threshold (no training) | 0.9888 | 0.9970 |

**How to read this honestly.**
- The tree **ties** the best single threshold (MNDWI + Otsu). It does not clearly beat it. That is a valid finding:
  water is easy to separate in this scene, so a tuned threshold is nearly as good as a learned model.
- The tree's value is that it is **interpretable**: its rules are only a few lines (it uses B11, MNDWI and B4) that can be
  written on a slide, and it can learn from labels rather than needing a threshold chosen by hand.
- These scores are **agreement with JRC labels on unseen blocks. They are not accuracy against ground truth.** The
  labels are coarse (30 m) and historical (1984 to 2021). Say this out loud when presenting.
- `gsw_dw` scores higher because it keeps only pixels where two independent products agree, which are the easiest
  pixels. Higher is not better evidence here.

### Optional: true accuracy

If you label the reference points (see `docs/OPTIONAL_ACCURACY_CHECK.md`), `summary.md` gains an area-weighted accuracy
section with confidence intervals. That is the one figure that can honestly be called accuracy. It has not been done
unless that section exists in `results/summary.md`.

---

## 3. Key ideas in plain language

| Term | Meaning |
|---|---|
| **Sentinel-2** | Two European satellites photographing the Earth every ~5 days. Free. 10 m pixels in visible and near-infrared bands. |
| **Band** | One colour channel the satellite records. B2 blue, B3 green, B4 red, B8 near-infrared (NIR), B11 shortwave-infrared (SWIR). |
| **Surface reflectance** | The image corrected for the atmosphere, so values mean what the ground reflects (0 to 1). |
| **Cloud Score+** | Google's per-pixel "how clear is this?" score. A pixel is kept only if it scores 0.60 or more. |
| **Median composite** | For each pixel, the middle value across all clear images in a season. Removes clouds and noise. |
| **NDWI** | `(Green - NIR) / (Green + NIR)`. Water is bright in green and dark in NIR, so water scores high. |
| **MNDWI** | `(Green - SWIR) / (Green + SWIR)`. Same idea with SWIR; it keeps buildings from looking like water. |
| **Decision Tree** | A flowchart of yes/no questions on pixel values ending in "water" or "land". |
| **JRC Global Surface Water (GSW)** | A map of how often each 30 m pixel held water from 1984 to 2021. Used here as the answer key for training. |
| **Dynamic World** | A Google land-cover product updated continuously. Used to check that GSW's old labels still hold. |
| **Spatial block split** | Whole 2 km squares go to training or testing, never mixed. Prevents cheating through neighbouring pixels. |
| **Cross-validation (CV)** | Repeating the train/test split several times to pick settings reliably. |
| **One-standard-error rule** | Among settings that score about equally well, pick the simplest one. |
| **Otsu threshold** | An automatic way to pick the best cut-off value on one index, with no labels needed. |
| **Confusion matrix** | A 2x2 table of right and wrong calls: land called land, water called water, and the two kinds of mistake. |
| **Precision / recall / IoU** | Precision: of what I called water, how much is. Recall: of the real water, how much I found. IoU: overlap of mapped and true water. |
| **No-data** | A pixel with too few clear observations. Left empty on purpose. |

---

## 4. How the pipeline works

```
Google Earth Engine (cloud, free)                       This repo (Colab or your laptop)
────────────────────────────────                        ──────────────────────────────────────────────
Sentinel-2 L2A  ─┐                                       labels.py     GSW (+ Dynamic World) -> water / land / ignore
Cloud Score+ mask├─> seasonal median composite ─┐        sampling.py   balanced points, 2 km blocks, block split
JRC GSW labels  ─┤                              ├──>     model.py      Decision Tree, spatial CV, 1-SE rule
Dynamic World   ─┘   export GeoTIFF to Drive  ──┘        evaluate.py   hold-out, baselines, bootstrap CIs
                                                         reference.py  optional independent points, area-weighted accuracy
                                                         predict.py    water map per season
                                                         change.py     dry -> wet change map
                                                         presentation.py  slide images + app data
```

Step by step, as the notebook runs it:

| Step | What happens | Where |
|---|---|---|
| 0 | Install packages, sign in to Earth Engine and Google Drive | Colab |
| 1 | Unpack the project code embedded in the notebook | Colab |
| 2 | Earth Engine builds the dry and wet composites plus the label bands and exports `s2_dry.tif`, `s2_wet.tif` to Drive | Earth Engine |
| 3 | Wait for the exports; copy them from Drive into `data/` | Colab |
| 4 | Integrity check of the rasters (grid match, label ranges) | `io.py` |
| 5 | Pipeline: labels, sampling, tuning, training, baselines, hold-out scores, water maps, change map | `pipeline.py` |
| 6 | Optional: score your own labelled reference points | `reference.py` |
| 7 | Show maps and figures | `plots.py` |
| 8 | Build the presentation pack and app data | `presentation.py` |
| 9 | Download the zips | Colab |

---

## 5. Design choices and why

Where this differs from a basic NDWI tutorial and from the original proposal (full table in
[docs/METHODOLOGY.md](docs/METHODOLOGY.md)):

| Choice | Reason |
|---|---|
| **Cloud Score+** instead of the QA60 cloud bit | QA60 is coarse and unavailable for part of 2022 to 2024. Cloud Score+ also catches haze and shadow. |
| **Seasonal median** instead of one date | A clear single date over humid coastal Lagos cannot be guaranteed. Cost: the map describes a season, not a day. |
| **GSW occurrence 75% or more**, edges removed by 3 px | GSW is 30 m; the imagery is 10 m. At shorelines the labels are wrong, so those pixels are not trained on. |
| **Two label modes** (`gsw`, `gsw_dw`) | GSW ends in 2021. Requiring Dynamic World to agree catches land reclaimed since. `gsw` is the primary (the proposal's method). |
| **Spatial block split** | A random pixel split puts near-identical neighbours in train and test and inflates accuracy. |
| **Tuned depth with the 1-SE rule** | Prevents picking a deep tree because of one lucky fold; keeps the tree readable. |
| **Baselines** (NDWI>0, MNDWI>0, Otsu) | Shows what the learned model actually adds. Found: very little here. |
| **Decision Tree, not Random Forest** | Its rules can be printed and argued about. |
| **Area-weighted reference accuracy** (optional) | Scoring against the training source is circular. Independent points with proper weights give a defensible number (Olofsson et al. 2014). |

---

## 6. Repository layout

| Path | What it is |
|---|---|
| `config.yaml` | Every setting (study area, seasons, thresholds, model grid). |
| `notebooks/waterdet_colab.ipynb` | Clean, self-contained notebook. Anyone can run it from scratch. |
| `notebooks/waterdet_colab_run_oct2026.ipynb` | Your executed run, outputs included. Proof of the work. |
| `src/waterdet/` | The Python package (modules in section 4, plus `gee_export.py`, `io.py`, `indices.py`, `config.py`, `geo.py`). |
| `tests/` | 51 tests on a synthetic study area with known truth. No Earth Engine needed. |
| `app/` | Streamlit app. `app/data/` holds the small result files the app displays. |
| `results/` | Copy of your real run: `summary.md`, `summary.json`, `presentation/` images. Used by this README. |
| `docs/HOW_TO_RUN.md` | Plain steps to run in Colab. |
| `docs/METHODOLOGY.md` | Full method, deviations from the proposal, limitations. |
| `docs/OPTIONAL_ACCURACY_CHECK.md` | How to label reference points for a true accuracy figure. |
| `data/`, `outputs/` | Working folders for rasters and fresh outputs. Rasters are git-ignored (too big). |

---

## 7. How to run it

### 7.1 In Colab (recommended)

1. Find your Earth Engine **project ID** (console.cloud.google.com; the **ID** column, like `my-project-123456`).
2. Open <https://colab.research.google.com>, upload `notebooks/waterdet_colab.ipynb`.
3. Type the project ID in the box, then **Runtime > Run all**. Approve the two Google pop-ups.
4. Wait 20 to 30 minutes (the Earth Engine exports are the slow part). Read the figures. Zips download at the end.

Details and the list of defaults: [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md).

### 7.2 Locally (optional)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[gee,dev]"
# set gee.project in config.yaml, then:
python -m waterdet.gee_export export --config config.yaml
# when the exports finish, download s2_dry.tif and s2_wet.tif from Drive into data\
python -m waterdet.pipeline --config config.yaml --data-dir data --out-dir outputs
python -m pytest
```

### 7.3 Settings you might change

Everything is in `config.yaml`. The useful ones:

| Setting | Default | Effect |
|---|---|---|
| `periods.dry / wet` | Dec 2025 to Feb 2026 / Jun to Sep 2026 (wet widened to May to mid-Oct in your run) | Seasons compared. End date is exclusive. |
| `imagery.cloud_score_threshold` | 0.60 | Higher means stricter cloud removal and fewer usable pixels. |
| `imagery.min_valid_obs` | 3 | Clear looks a pixel needs. Lower (2) if the wet season is mostly no-data. |
| `labels.min_gsw_occurrence` | 75 | Stricter or looser definition of "permanent water". |
| `labels.primary_mode` | `gsw` | Which label mode draws the final maps. |
| `model.max_depth` | 2 to 10 | Tree sizes tried. |

---

## 8. The Streamlit app

Six tabs: map explorer (satellite image with water overlay, dry and wet), seasonal change, how the model decides (tree
and rules), accuracy (depth curve, confusion matrix, feature importance), numbers, and data downloads.

Run locally:

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

Deploy free: <https://share.streamlit.io>, sign in with GitHub, New app, choose this repo, branch `main`, main file
`app/streamlit_app.py`, Deploy.

The app reads only small files from `app/data/` (written by notebook Step 8). It never touches Earth Engine or the big
rasters. If it shows "Run the notebook to Step 8", `app/data/summary.json` is missing.

---

## 9. Outputs: what each file means

| File | Meaning |
|---|---|
| `summary.md`, `summary.json` | All metrics, warnings, provenance (package versions, input hashes). The source of truth for every number. |
| `maps/water_<season>.tif` | 0 land, 1 water, 255 no data. 10 m, UTM. |
| `maps/change_dry_to_wet.tif` | 0 stable land, 1 stable water, 2 gained, 3 lost, 255 no data. |
| `mode_<label mode>/` | Trained model, `tree_rules.txt`, CV table, confusion matrix, tree and importance figures. |
| `presentation/` | `01_satellite`, `02_detection`, `03_before_after`, `04_model_inputs`, `05_change_map`, `06_dry_wet_change`, `07_key_numbers`, `08_decision_tree`, `09_depth_choice`, `10_confusion_matrix`, `11_feature_importance`, `key_numbers.csv`. |
| `streamlit_data/` | Everything the app needs. Copy into `app/data/`. |

---

## 10. Presentation guide (slides and talking points)

Suggested 10 slides. Image names refer to `results/presentation/`.

| # | Slide | Content | Image |
|---|---|---|---|
| 1 | Title | Project name, your name, Interspatial Technologies, SIWES | none |
| 2 | Problem | Water changes with season; ground surveys are slow; satellites are free | none |
| 3 | Data | Sentinel-2, 10 m, 5-day revisit; Lagos study area; two seasons | `01_satellite_dry.png` |
| 4 | Cleaning the data | Cloud Score+ mask, seasonal median, why Lagos needs it | `04_model_inputs_dry.png` |
| 5 | Water indices | NDWI and MNDWI formulas and what each does well and badly | `04_model_inputs_dry.png` |
| 6 | Training labels | JRC GSW, 75% rule, edge removal, Dynamic World check | none (diagram) |
| 7 | The model | Decision Tree; show the actual rules | `08_decision_tree.png` |
| 8 | Result: detection | Image next to detected water; water area in km² | `03_before_after_dry.png` |
| 9 | Result: change | Dry to wet gain and loss | `06_dry_wet_change.png` |
| 10 | Accuracy and honesty | Scores, baseline tie, limits, next steps | `10_confusion_matrix.png`, `07_key_numbers.png` |

Optional extra slide: live demo of the Streamlit app.

**Say these sentences:**
- "The model is trained on a historical water map, so my scores measure agreement with that map on areas the model never
  saw. They are not proof against ground truth."
- "A simple MNDWI threshold performs almost as well as the tree. The tree's advantage is that its rules are readable
  and it learns from labels."
- "Test squares were kept apart from training squares, because neighbouring pixels are nearly identical and a random
  split would overstate accuracy."
- "Where the wet season has gaps, that is cloud. I left those pixels empty instead of guessing."

**Do not say:** "98.8% accurate" without the qualifier above; "the tree beats thresholds"; any number not in `summary.md`.

**Live demo order (3 minutes):** Map explorer (dry), toggle water overlay, Seasonal change tab, Model rules tab, then stop.

---

## 11. Report guide

Suggested structure (about 8 to 12 pages):

1. **Abstract** (150 words): problem, method, headline results, key limitation.
2. **Introduction**: why water mapping matters in Lagos; objectives.
3. **Study area and data**: map of the area; Sentinel-2, Cloud Score+, GSW, Dynamic World.
4. **Methods**: preprocessing, indices (with formulas), labels, sampling and spatial split, model and tuning, baselines,
   accuracy measures, change detection. Copy the substance from [docs/METHODOLOGY.md](docs/METHODOLOGY.md).
5. **Results**: water area per season (table), accuracy table with baselines, tree rules, confusion matrix, change map.
6. **Discussion**: why the tree ties the threshold; which pixels are hard (shadows, mangrove, turbid water);
   cloud limits on the wet season; what the change map is and is not.
7. **Limitations**: section 13.
8. **Conclusion and future work**: independent reference points for true accuracy; Sentinel-1 radar (sees through cloud);
   more seasons; other cities; Projects 2 and 3.
9. **References**: section 16.
10. **Appendix**: tree rules (`tree_rules.txt`), `key_numbers.csv`, repo and app links.

How to document a run: save `results/summary.md` and `summary.json`, the executed notebook, and the dates you ran it.
Earth Engine's catalogue can change over time, so a later re-run may differ slightly.

---

## 12. Questions you may be asked

| Question | Short answer |
|---|---|
| Why a Decision Tree? | Its rules can be printed and checked. A Random Forest would be a black box. |
| Why not just threshold NDWI? | We did compare: the tuned tree and a threshold are nearly equal here. The tree adds interpretability and label-driven learning. |
| Is 98.8% your accuracy? | No. It is agreement with JRC labels on unseen areas. True accuracy needs independent reference points. |
| Why spatial split? | Neighbouring pixels are near-duplicates; random splits leak and inflate scores. |
| Why median composites? | Cloud cover. One date may not exist. A median of several clear looks is cleaner. |
| Why are labels from 1984 to 2021 usable for 2026 imagery? | Only permanent water and never-water pixels are used. Those rarely change. The Dynamic World mode checks reclaimed land. |
| Why is the wet season patchy? | Persistent cloud in the rainy season. Pixels with too few clear looks are no-data. |
| What are the error sources? | Building shadow, wet bare soil, mangrove, turbid water, shoreline mixed pixels. |
| What would you do next? | Label independent reference points; add Sentinel-1 radar for cloudy dates; test other cities. |
| Is the change map reliable? | It is a product of two classifications, so errors add. Large patches are meaningful; small ones may be noise. |

---

## 13. Limitations

- Training labels are 30 m and cover 1984 to 2021; the imagery is 10 m and from 2025 to 2026. Hold-out agreement with
  those labels is not accuracy against truth.
- Seasonal and tidal water is excluded from training, so its quality is known only through reference points.
- Wet-season coverage is limited by cloud; the result describes a season, not a day.
- B11 (SWIR) is 20 m resampled to 10 m, so MNDWI is blockier than NDWI.
- Median compositing hides short events such as a flood lasting days.
- The tree ties the best single threshold, so the learned model adds interpretability more than accuracy.
- Earth Engine collections are updated over time, so re-exports may differ slightly.

---

## 14. Troubleshooting log (problems met and fixed)

| Symptom | Cause | Fix |
|---|---|---|
| `Band pattern 'dw_water' did not match any bands` | Dynamic World had no images for the window | Code now falls back to an empty band; covered by a test. |
| `FileNotFoundError: No s2_wet*.tif` | Earlier export failed, or file not yet in Drive | Wait for the export task to show COMPLETED; re-run the copy cell. |
| Wet season 100% no data | Too few clear observations in cloudy months | Widen the window, relax scene cloud filter, set `min_valid_obs: 2`, re-export wet only. |
| `TypeError: NoneType doesn't define __round__` in Step 8 | Season with blank numbers | Blank values now show as "n/a"; covered by a test. |
| `data/reference_points.csv has no labelled rows` | Uploaded CSV with no labels filled | Delete the CSV and its meta file, or fill the label column. The step is optional. |
| Streamlit says "Run the notebook to Step 8" | `app/data/summary.json` missing | Copy `streamlit_data/` contents into `app/data/`. |
| GeoTIFF download buttons missing online | `.gitignore` blocks `*.tif` | `git add -f app/data/*.tif` (the files are small). |
| GitHub rejects a push | Large `s2_*.tif` added | Never commit `data/`; they stay in Drive. |

---

## 15. Testing and reproducibility

```bash
python -m pytest        # 51 tests, about one minute
```

The tests build a synthetic study area with known truth, including a shadow trap for MNDWI and a reclaimed-land trap for
GSW, and check that the pipeline recovers it. The Earth Engine graph is validated offline against a mock. Streamlit's
headless test runner checks the app, including the missing-data case.

Seeds are fixed (`seed: 42`). `summary.json` stores the config (project ID redacted), package versions and SHA-256
hashes of the input rasters.

---

## 16. Data, references, licence

- Sentinel-2 L2A (Copernicus / ESA), via Google Earth Engine `COPERNICUS/S2_SR_HARMONIZED`
- Cloud Score+ (Google), `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`
- JRC Global Surface Water v1.4: Pekel et al. 2016, *Nature* 540:418-422
- Dynamic World: Brown et al. 2022, *Scientific Data* 9:251
- NDWI: McFeeters 1996, *Int. J. Remote Sens.* 17:1425-1432
- MNDWI: Xu 2006, *Int. J. Remote Sens.* 27:3025-3033
- Accuracy assessment: Olofsson et al. 2014, *Remote Sens. Environ.* 148:42-57

Licence: MIT. See `LICENSE`.

### SIWES portfolio

| # | Project | Status |
|---|---|---|
| 1 | Water body detection: Sentinel-2, NDWI/MNDWI, Decision Tree (this repo) | Built and run |
| 2 | Building footprint extraction: U-Net on SpaceNet | Planned |
| 3 | Wildfire risk prediction: fused vegetation, thermal and weather data | Planned |
