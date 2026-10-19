import json

import numpy as np
import pandas as pd
import pytest
import rasterio.transform
import rasterio.warp

from waterdet.evaluate import (binary_metrics, block_bootstrap_ci, metrics_from_matrix, otsu_threshold,
                               stratified_bootstrap_ci, weighted_confusion, weighted_metrics)
from waterdet.io import read_stack
from waterdet.predict import MAP_NODATA
from waterdet.reference import build_sampling_frame, load_reference, meta_path, sample_at_points, strata_from_map


def test_metrics_known_matrix():
    m = binary_metrics([0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 0])
    assert m["confusion_matrix"] == [[3, 1], [1, 3]]
    assert m["overall_accuracy"] == pytest.approx(0.75)
    assert m["kappa"] == pytest.approx(0.5)
    assert m["water_precision"] == pytest.approx(0.75) and m["water_recall"] == pytest.approx(0.75)
    assert m["water_iou"] == pytest.approx(3 / 5)
    perfect = binary_metrics([0, 1, 0, 1], [0, 1, 0, 1])
    assert perfect["kappa"] == pytest.approx(1.0)


def test_metrics_degenerate_cases_do_not_crash():
    m = binary_metrics([0, 0, 0], [0, 0, 0])
    assert m["overall_accuracy"] == 1.0 and np.isnan(m["water_precision"])


def test_otsu_on_bimodal():
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(-0.5, 0.1, 5000), rng.normal(0.4, 0.1, 5000)])
    t = otsu_threshold(v)
    assert -0.15 < t < 0.15
    with pytest.raises(ValueError):
        otsu_threshold(np.array([np.nan] * 5))


def test_block_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    pred = np.where(rng.random(2000) < 0.9, y, 1 - y)
    groups = rng.integers(0, 20, 2000)
    ci = block_bootstrap_ci(y, pred, groups, n_boot=200, seed=0)["overall_accuracy"]
    oa = binary_metrics(y, pred)["overall_accuracy"]
    assert ci[0] < oa < ci[1] and ci[1] - ci[0] < 0.1


def test_weighted_estimator_recovers_population_accuracy():
    """Population: 90% land (error 2%), 8% water core (error 5%), 2% shoreline (error 40%).
    A sample that is 1/3 each stratum must NOT report the naive average; the weighted one must be close to truth."""
    rng = np.random.default_rng(0)
    weights = {"land_core": 0.90, "water_core": 0.08, "shoreline": 0.02}
    err = {"land_core": 0.02, "water_core": 0.05, "shoreline": 0.40}
    true_oa = sum(weights[h] * (1 - err[h]) for h in weights)
    ref, pred, stratum = [], [], []
    for h in weights:
        n = 4000
        r = rng.integers(0, 2, n)
        wrong = rng.random(n) < err[h]
        ref += list(r); pred += list(np.where(wrong, 1 - r, r)); stratum += [h] * n
    ref, pred, stratum = map(np.array, (ref, pred, stratum))
    naive = binary_metrics(ref, pred)["overall_accuracy"]
    w = weighted_metrics(ref, pred, stratum, weights)["overall_accuracy"]
    assert abs(w - true_oa) < 0.01
    assert abs(naive - true_oa) > 0.05                      # naive average is clearly biased here
    assert weighted_confusion(ref, pred, stratum, weights).sum() == pytest.approx(1.0)
    ci = stratified_bootstrap_ci(ref, pred, stratum, weights, n_boot=100, seed=0)["overall_accuracy"]
    assert ci[0] <= w <= ci[1]
    with pytest.raises(ValueError, match="no usable"):
        weighted_confusion(ref[stratum != "shoreline"], pred[stratum != "shoreline"], stratum[stratum != "shoreline"], weights)


def _map_with_lake():
    m = np.zeros((200, 200), np.uint8)
    m[50:150, 60:140] = 1
    m[:20, :20] = MAP_NODATA
    return m


def test_strata_partition():
    m = _map_with_lake()
    s = strata_from_map(m, 2)
    assert (s[:20, :20] == -1).all()
    assert (s[m == MAP_NODATA] == -1).all()
    assert (s[(m != MAP_NODATA)] >= 0).all()
    assert s[100, 100] == 0 and s[5, 150] == 1 and s[50, 100] == 2 and s[149, 100] == 2


def test_sampling_frame_roundtrip_and_blind_csv(tmp_path, data_dir, small_cfg):
    stack = read_stack(data_dir / "s2_dry.tif")
    m = (stack.bands["B8"] < 0.12).astype(np.uint8)          # crude water map
    out = tmp_path / "frame"
    meta = build_sampling_frame(m, stack, n_by_stratum={"water_core": 30, "land_core": 30, "shoreline": 20},
                                shore_px=2, seed=1, period="dry", out_dir=out)
    df = pd.read_csv(out / "reference_frame.csv", dtype={"id": str})
    assert list(df.columns) == ["id", "lon", "lat", "label", "notes"]          # no map class leaks into the CSV
    assert len(df) == 80 and df["id"].is_unique
    assert sum(meta["weights"].values()) == pytest.approx(1.0)
    assert set(meta["strata"].values()) == {"water_core", "land_core", "shoreline"}
    assert "<Placemark>" in (out / "reference_frame.kml").read_text()
    # round trip: lon/lat -> pixel must land on the pixel the point was drawn from
    kept, info = sample_at_points(df, stack, min_obs=3)
    strata = strata_from_map(m, 2)
    for (_, r), name in zip(kept.iterrows(), kept["id"].map(meta["strata"])):
        assert ("water_core", "land_core", "shoreline")[strata[r["row"], r["col"]]] == name
    assert info["n_outside"] == 0


def test_sample_at_points_drops_outside_and_invalid(data_dir, wet_bands):
    stack = read_stack(data_dir / "s2_wet.tif")
    # pixel centres: one inside valid, one in the cloud hole (rows 0-49, cols 450-499), one outside
    pts = [(10, 10), (10, 470), (5000, 5000)]
    xs, ys = rasterio.transform.xy(stack.transform, [p[0] for p in pts], [p[1] for p in pts], offset="center")
    lon, lat = rasterio.warp.transform(stack.crs, "EPSG:4326", xs, ys)
    df = pd.DataFrame({"id": ["a", "b", "c"], "lon": lon, "lat": lat, "y": [0, 0, 0]})
    kept, info = sample_at_points(df, stack, min_obs=3)
    assert list(kept["id"]) == ["a"] and info["n_invalid_pixel"] == 1 and info["n_outside"] == 1


def test_load_reference_label_parsing(tmp_path):
    csv = tmp_path / "r.csv"
    csv.write_text("id,lon,lat,label,notes\nP1,3.4,6.5,Water,\nP2,3.4,6.5, land ,\nP3,3.4,6.5,unsure,\nP4,3.4,6.5,,\n")
    df, info = load_reference(csv)
    assert list(df["y"]) == [1, 0] and info["n_unsure"] == 1 and info["n_blank"] == 1 and info["meta"] is None
    csv.write_text("id,lon,lat,label,notes\nP1,3.4,6.5,lagoon,\n")
    with pytest.raises(ValueError, match="Unrecognised"):
        load_reference(csv)
    csv.write_text("id,lon,lat,label,notes\nP1,3.4,6.5,,\n")
    with pytest.raises(ValueError, match="no labelled rows"):
        load_reference(csv)
    csv.write_text("id,lon,lat\nP1,3.4,6.5\n")
    with pytest.raises(ValueError, match="label"):
        load_reference(csv)


def test_load_reference_rejects_edited_ids(tmp_path):
    csv = tmp_path / "r.csv"
    csv.write_text("id,lon,lat,label,notes\nPX,3.4,6.5,water,\n")
    meta_path(csv).write_text(json.dumps({"strata": {"P1": "water_core"}, "weights": {"water_core": 1.0}}))
    with pytest.raises(ValueError, match="ids"):
        load_reference(csv)


def test_load_reference_rejects_csv_from_a_different_frame(tmp_path):
    """Same ids, different places: ids alone would silently assign the wrong strata and weights."""
    csv = tmp_path / "r.csv"
    csv.write_text("id,lon,lat,label,notes\nP1,3.4,6.5,water,\n")
    meta = {"strata": {"P1": "water_core"}, "weights": {"water_core": 1.0}, "coords": {"P1": [3.4, 6.5]}}
    meta_path(csv).write_text(json.dumps(meta))
    df, _ = load_reference(csv)                                     # matching coordinates load fine
    assert list(df["stratum"]) == ["water_core"]
    csv.write_text("id,lon,lat,label,notes\nP1,3.41,6.5,water,\n")  # ~1 km away
    with pytest.raises(ValueError, match="different runs"):
        load_reference(csv)
