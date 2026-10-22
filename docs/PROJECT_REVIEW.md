# Project 1 review and proposal alignment

Review date: 22 October 2026. Source: SIWES_Geospatial_ML_Project_Proposal.docx, Project 1, Water Body Detection with NDWI + Decision Tree. The document was read as project requirements and background; embedded example commands were not treated as instructions to execute. Projects 2 and 3 are outside this repository's scope.

## Conclusion

The code implements the core Project 1 workflow and the optional seasonal comparison. It improves on the illustrative proposal with spatial hold-out evaluation, grouped cross-validation, threshold baselines and provenance. A fresh run of the original input rasters reproduces the saved results. Full scientific validation remains incomplete because no independently labelled reference points have been supplied.

| Proposal requirement | Implementation and evidence | Status |
|---|---|---|
| Mixed water and land AOI | Lagos Lagoon bounding box in config | Implemented |
| Sentinel-2 L2A and cloud masking | gee_export.py; seasonal medians and Cloud Score+ | Implemented; live access needed for new export |
| NDWI and optional MNDWI | indices.py; seven features with five raw bands | Implemented |
| JRC training labels and manual spot-checks | labels.py; reference frame CSV/KML workflow | Automatic labels implemented; manual labels outstanding |
| Decision Tree with depth tuning | model.py; grouped CV and one-standard-error choice | Implemented |
| Confusion matrix, overall accuracy, Kappa | evaluate.py and per-mode figures | Implemented; scores are agreement with label source |
| Visual comparison with GSW | New spatial_validation.png compares held-out labels and predictions | Implemented for new runs |
| Water-extent map | GeoTIFF and PNG per period | Implemented |
| Optional second date and seasonal change | Common-valid-area change map | Implemented |
| Short methodology and presentation outputs | Methodology, slide guide, image packs and review evidence | Implemented |

The proposal's example dates are 1 November 2025 to 28 February 2026 (exclusive). The repository's dry window is 1 December 2025 to 1 March 2026. This is an intentional seasonal adaptation, not an exact date reproduction. The proposal suggests occurrence >50%; this code uses >=75% and an eroded confident core. Both deviations must be stated when presenting.

## Correctness fixes

- Missing GSW values no longer become land training labels. Explicit zero occurrence remains valid land evidence.
- Zero-denominator NDWI/MNDWI pixels are excluded consistently from training and prediction.
- Raster internal masks are honoured on read. Area uses the affine determinant and CRS units; pipeline rejects angular, rotated or mismatched-resolution grids before training.
- Water F1 uses 2TP/(2TP+FP+FN), returning zero when water prediction fails and NaN only when the class is absent from both truth and predictions.
- Reference CSVs reject duplicate/missing IDs and invalid coordinates. A zero shoreline width means no shoreline buffer rather than unlimited morphological expansion.
- Presentation building follows chronological periods and the current summary's change products instead of arbitrary stale files. It no longer recursively deletes presentation folders before input checks.

## Maintainability and evidence improvements

The runnable Colab notebook is the single entry point; its embedded package is refreshed from maintained source by scripts/sync_notebook.py.

Runtime dependency declarations are centralised in pyproject.toml; requirements.txt installs that package. An offline demo reuses the test world's generator instead of maintaining a separate fake-data implementation. CLI runs save run.log, and --presentation builds figures and app assets. New sample assignment and held-out prediction CSVs expose spatial splits and individual disagreements.

README claims were checked against saved JSON and a fresh rerun of the same hashed imagery. Water areas, chosen tree settings and held-out metrics reproduce exactly. Primary GSW agreement is 98.52%, not the earlier 98.8% claim; MNDWI Otsu is 98.79%. Thresholds now include block-bootstrap intervals. The saved wet window and cloud settings are recorded separately from defaults. Generated presentation and app assets have been refreshed from the reproducible run.

## Remaining scientific limits

No independent reference labels were supplied. They cannot be fabricated or inferred from the same GSW layer and then called independent. Use the generated frame and the reference guide. Interpret matching-date imagery and retain unsure labels and notes.

The labels exclude difficult shoreline and seasonal pixels. Balanced sampling changes precision/F1 relative to real class prevalence; held-out blocks need not be exactly 50:50. Adjacent train/test blocks can still share spatial structure. Confidence intervals quantify sampling variation, not systematic label errors. Dynamic World uses Sentinel-2 and is not independent ground truth. Comparing its label mode with GSW mode also changes sample membership.

Seasonal median composites cannot identify a particular flood date. Differences in mapped area alone include coverage effects; net change must use gain minus loss on common valid coverage. No causal flood claim or verified groundwater interpretation is justified.

## Verification record

The latest full test run is recorded as `results/review/final_pytest.log` and `final_pytest.xml`, including app and presentation tests. The exact input hashes match the previous run. Real-data maps, logs, per-point held-out decisions and synthetic end-to-end demonstrations are available under `outputs/real_run/` and `outputs/demo/`; the demo outputs are artificial and never stand in for Lagos observations. Synthetic maps agree with their known generated truth at 99.87% (dry) and 99.85% (wet).
