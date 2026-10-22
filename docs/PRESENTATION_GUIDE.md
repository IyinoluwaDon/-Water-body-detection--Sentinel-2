# Presentation guide

Use the saved Lagos images in results/presentation/ for measured project outputs. Use outputs/demo/presentation/ only to explain testing on artificial data. Do not combine their metrics.

| Slide | Point to explain | Asset |
|---|---|---|
| 1 | Lagos surface-water mapping and SIWES objective | Project title and AOI |
| 2 | Sentinel-2, seasonal cloud masking and spatial resolution | 01_satellite_dry.png |
| 3 | Five raw bands plus NDWI and MNDWI | 04_model_inputs_dry.png |
| 4 | Historical labels, buffered cores and spatial splitting | New-run 13_spatial_validation.png |
| 5 | Decision Tree depth tuning and interpretable rules | 08_decision_tree.png; 09_depth_choice.png |
| 6 | Detected water against satellite appearance | 03_before_after_dry.png |
| 7 | Dry/wet change on common valid coverage | 06_dry_wet_change.png |
| 8 | Threshold comparison and uncertainty | 12_threshold_comparison.png |
| 9 | Testing and reproducibility | results/review/final_pytest.log; outputs/demo/known_truth_checks.json |
| 10 | Limits and independent labelling still needed | docs/PROJECT_REVIEW.md |

The primary recorded tree has 98.52% agreement with JRC-derived held-out labels, Kappa 0.9704 and F1 0.9855. MNDWI Otsu has 98.79% agreement. These figures do not demonstrate that the tree beats thresholds or that it is 98.52% accurate against reality.

Explain that 230.10 km² and 236.88 km² are mapped water in the dry and wet composites. Gain and loss on common coverage are 10.56 and 3.80 km²; net gain is about 6.76 km². This is seasonal mapped change, not confirmed flooding.

Prepare to answer why a tree was chosen (compact explicit rules), why neighbouring pixels are kept in whole blocks, why historical labels can be outdated, and how uncertainty at tidal shores and reclaimed land could be checked. Random Forests can be interpreted too; the single tree is simply easier to display in full.

Run the app before your presentation and keep PNGs available as a fallback. The app uses packaged assets and needs no Earth Engine connection for viewing.
