# Reproduced Lagos run

Run date: 22 October 2026. The locally supplied dry and wet GeoTIFFs match the recorded SHA-256 hashes byte for byte. Rerun command:

```powershell
.\.venv\Scripts\python -m waterdet.pipeline --config results/review/recorded_run_config.json --data-dir data --out-dir outputs/real_run --presentation
```

The fresh run reproduced both seasonal areas, tree choices, held-out confusion metrics and change totals exactly. It generated maps, tree rules, CV tables, spatial-validation plots, sample assignment CSVs, per-point held-out predictions, slide assets and Streamlit assets. Complete run output and `run.log` are under `outputs/real_run/`.

| Result | Fresh run |
|---|---:|
| Primary training labels | GSW (`gsw`) |
| Chosen tree | depth 2, 4 leaves; `min_samples_leaf=50` |
| Spatial held-out agreement with GSW | 98.52% |
| Kappa | 0.9704 |
| Water F1 | 0.9855 |
| Best threshold agreement, MNDWI Otsu | 98.79% |
| Dry water area | 230.10 km² |
| Wet water area | 236.88 km² |
| Water gained on comparable coverage | 10.56 km² |
| Water lost on comparable coverage | 3.80 km² |
| Independent reference labels | 0 |

The other label mode (`gsw_dw`) and the thresholds are evaluated on their own held-out samples. Those point sets differ, so compare methods within the same label mode. Accuracy is agreement with historical map labels; no 98.52% ground-truth accuracy claim is supported. A hand-labelled, independently checked reference set remains the main open validation task.

Dates are dry 1 December 2025 to 1 March 2026 (exclusive end) and wet 1 May to 17 October 2026 (exclusive end). The run uses two minimum clear observations and a 90% scene cloud prefilter. New defaults differ. The precise configuration and hashes are preserved in `outputs/real_run/summary.json`.
