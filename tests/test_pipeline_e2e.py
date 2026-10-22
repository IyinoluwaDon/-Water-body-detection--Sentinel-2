"""End-to-end run on the synthetic world, plus an honest-validation check against known truth."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest
import rasterio

from waterdet.io import read_stack
from waterdet.pipeline import run


@pytest.fixture()
def first_run(tmp_path, data_dir, small_cfg):
    cfg = dict(small_cfg)
    cfg["reference"] = dict(cfg["reference"], csv=str(tmp_path / "missing.csv"))
    out = tmp_path / "out"
    summary = run(cfg, data_dir, out)
    return cfg, out, summary


def test_pipeline_outputs_and_scores(first_run, dry_bands):
    cfg, out, s = first_run
    for f in ["summary.json", "summary.md", "maps/water_dry.tif", "maps/water_wet.tif", "maps/change_dry_to_wet.tif",
              "maps/water_dry.png", "mode_gsw/model.joblib", "mode_gsw_dw/tree_rules.txt",
              "mode_gsw_dw/cv_depth_curve.png", "reference_frame/reference_frame.csv",
              "reference_frame/reference_frame.meta.json"]:
        assert (out / f).exists(), f
    json.loads((out / "summary.json").read_text())                        # valid JSON (no bare NaN)
    clean, noisy = s["modes"]["gsw_dw"], s["modes"]["gsw"]
    assert clean["test_spatial_holdout"]["overall_accuracy"] > 0.97
    assert clean["tree_depth"] <= 4
    assert any("No independent reference" in n for n in s["notes"])
    # DT must beat or match the best fixed threshold; MNDWI>0 is fooled by the shadow patch
    base = clean["baselines_on_same_test_points"]
    assert all(len(x["ci95_block_bootstrap"]["overall_accuracy"]) == 2 for x in base.values())
    assert clean["test_spatial_holdout"]["kappa"] >= max(b["kappa"] for b in base.values()) - 0.01
    assert noisy["test_spatial_holdout"]["overall_accuracy"] < clean["test_spatial_holdout"]["overall_accuracy"]
    assert "wet" in clean["temporal_transfer"] and clean["temporal_transfer"]["wet"]["overall_accuracy"] > 0.95
    prov = s["provenance"]                                                # reproducibility record
    assert set(prov["inputs_sha256"]) == {"dry", "wet"} and all(len(v) == 64 for v in prov["inputs_sha256"].values())
    assert prov["config"]["gee"]["project"] == "<redacted>"               # project id must not leak into outputs
    assert "## Provenance" in (out / "summary.md").read_text(encoding="utf-8")


def test_summary_writer_survives_nan_metrics(tmp_path):
    """NaN metrics become None in the JSON-safe summary; the markdown writer must not crash on them."""
    from waterdet.pipeline import _clean, _write_summary
    nan = float("nan")
    m = {"overall_accuracy": 0.9, "kappa": nan, "water_f1": nan, "water_iou": nan, "water_precision": nan,
         "water_recall": nan, "ci95_block_bootstrap": {"overall_accuracy": [nan, nan], "kappa": [nan, nan]}}
    mode = {"tree_depth": 2, "tree_leaves": 3, "chosen_params": {}, "n_train_points": 10, "n_test_points": 5,
            "n_train_blocks": 3, "n_test_blocks": 2, "train": m, "test_spatial_holdout": m,
            "baselines_on_same_test_points": {"NDWI>0": m}, "baseline_thresholds": {"NDWI>0": 0.0}}
    summary = {"training_period": "dry", "modes": {"gsw": mode}, "areas": {"dry": {"water_km2": 1.0, "land_km2": 2.0,
               "nodata_pct_of_aoi": nan}}, "change": {}, "notes": [],
               "reference_validation": {"n_used": 4, "area_weighted": True, "n_outside": 0, "n_invalid_pixel": 0,
                                         "n_unsure": 0, "methods": {"NDWI>0": m}}}
    _write_summary(tmp_path / "s.md", _clean(summary))
    assert "n/a" in (tmp_path / "s.md").read_text(encoding="utf-8")


def test_map_matches_truth_and_change_map(first_run, dry_bands, wet_bands, world):
    cfg, out, s = first_run
    dry = rasterio.open(out / "maps/water_dry.tif").read(1)
    wet = rasterio.open(out / "maps/water_wet.tif").read(1)
    ok = dry != 255
    assert (dry[ok] == dry_bands["_truth"][ok]).mean() > 0.97             # primary mode = gsw_dw
    assert (wet[:50, 450:] == 255).all()                                   # cloud hole is no-data, not guessed
    # the pond appears only in the wet map -> counted as gain
    chg = rasterio.open(out / "maps/change_dry_to_wet.tif").read(1)
    pond = world["pond"]
    assert (chg[pond] == 2).mean() > 0.8
    assert s["change"]["dry_to_wet"]["water gain (land -> water)"] > 0.05      # km2 (pond ~ 0.06 km2)
    assert s["areas"]["wet"]["nodata_pct_of_aoi"] == pytest.approx(100 * 2500 / 200000, abs=0.1)


def test_independent_reference_validation_is_close_to_truth(tmp_path, first_run, data_dir, dry_bands):
    cfg, out, s = first_run
    frame = pd.read_csv(out / "reference_frame/reference_frame.csv", dtype={"id": str})
    stack = read_stack(data_dir / "s2_dry.tif")
    inv = ~stack.transform
    import rasterio.warp
    xs, ys = rasterio.warp.transform("EPSG:4326", stack.crs, frame["lon"].tolist(), frame["lat"].tolist())
    xs, ys = np.asarray(xs), np.asarray(ys)
    cols = np.floor(inv.a * xs + inv.b * ys + inv.c).astype(int)
    rows = np.floor(inv.d * xs + inv.e * ys + inv.f).astype(int)
    frame["label"] = np.where(dry_bands["_truth"][rows, cols], "water", "land")      # perfect 'human' labeller
    frame.loc[0, "label"] = "unsure"
    csv = tmp_path / "ref" / "reference_points.csv"
    csv.parent.mkdir()
    frame.to_csv(csv, index=False)
    shutil.copy(out / "reference_frame/reference_frame.meta.json", csv.with_name("reference_points.meta.json"))
    cfg2 = dict(cfg); cfg2["reference"] = dict(cfg["reference"], csv=str(csv))
    s2 = run(cfg2, data_dir, tmp_path / "out2")
    ref = s2["reference_validation"]
    assert ref["area_weighted"] and ref["n_unsure"] == 1
    m = ref["methods"]
    dry_map = rasterio.open(tmp_path / "out2/maps/water_dry.tif").read(1)
    ok = dry_map != 255
    true_oa = (dry_map[ok] == dry_bands["_truth"][ok]).mean()
    assert abs(m["model[gsw_dw]"]["overall_accuracy"] - true_oa) < 0.03
    # In this world the shadow patch fools MNDWI>0: the independent check must expose it
    assert m["MNDWI>0"]["overall_accuracy"] < m["model[gsw_dw]"]["overall_accuracy"]
    assert not any("UNWEIGHTED" in n for n in s2["notes"])


def test_grid_mismatch_and_missing_training_raster(tmp_path, data_dir, small_cfg):
    (data_dir / "s2_dry.tif").rename(data_dir / "tmp.tif")
    with pytest.raises(FileNotFoundError, match="training period"):
        run(small_cfg, data_dir, tmp_path / "o")
