# Presentation pack

Suggested slide order:

1. `01_satellite_*.png`  The raw input: what the satellite sees.
2. `04_model_inputs_*.png`  The data: 5 bands and 2 water indices the model learns from.
3. `03_before_after_*.png`  The detection: image next to the water the model found.
4. `06_dry_wet_change.png` and `05_change_map.png`  The monitoring result: seasonal change.
5. `08_decision_tree.png` with `08_decision_tree_rules.txt`  How the model decides, in plain rules.
6. `09_depth_choice.png`, `10_confusion_matrix.png`, `13_spatial_validation.png`  How it was tested and where it disagrees.
7. `12_threshold_comparison.png`  Compare the tree with simple index thresholds and show block-bootstrap intervals.
8. `07_key_numbers.png`  The main quantities on one slide.

## One sentence to say out loud about accuracy
"These scores measure agreement with the JRC Global Surface Water map on areas the model never saw; they are not proof against ground truth, which needs independent reference points."
