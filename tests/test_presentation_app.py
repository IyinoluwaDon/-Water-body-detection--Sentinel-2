"""The presentation pack is built from a finished run, and the Streamlit app renders from it without errors."""

import json
from pathlib import Path

import pytest
from PIL import Image

from waterdet.pipeline import run
from waterdet.presentation import build_presentation_pack

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture()
def pack(tmp_path, data_dir, small_cfg):
    cfg = dict(small_cfg)
    cfg["reference"] = dict(cfg["reference"], csv=str(tmp_path / "missing.csv"))
    out = tmp_path / "out"
    run(cfg, data_dir, out)
    info = build_presentation_pack(cfg, data_dir, out, max_px=300)
    return cfg, out, info


def test_pack_has_every_slide_asset(pack):
    _, out, info = pack
    pres = out / "presentation"
    for name in ["01_satellite_dry.png", "01_satellite_wet.png", "02_detection_dry.png", "03_before_after_dry.png",
                 "04_model_inputs_dry.png", "05_change_map.png", "06_dry_wet_change.png", "07_key_numbers.png",
                 "08_decision_tree.png", "08_decision_tree_rules.txt", "09_depth_choice.png", "key_numbers.csv",
                 "README_presentation.md"]:
        assert (pres / name).exists(), name
        assert (pres / name).stat().st_size > 0
    assert Image.open(pres / "01_satellite_dry.png").size[0] > 400


def test_streamlit_data_is_small_and_consistent(pack):
    _, out, _ = pack
    app = out / "streamlit_data"
    meta = json.loads((app / "meta.json").read_text())
    for p in meta["periods"]:
        rgb, water = Image.open(app / f"rgb_{p}.png"), Image.open(app / f"water_{p}.png")
        assert rgb.size == water.size, "overlay must line up with the satellite image"
    assert sum(f.stat().st_size for f in app.iterdir()) < 25_000_000, "app data must stay small enough for GitHub"
    assert "water" in (app / "key_numbers.csv").read_text().lower()


def test_streamlit_app_renders_all_tabs(pack, monkeypatch):
    from streamlit.testing.v1 import AppTest

    _, out, _ = pack
    monkeypatch.setenv("WATERDET_APP_DATA", str(out / "streamlit_data"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 6 and len(at.metric) >= 2
    at.radio[0].set_value("wet").run()                      # switch season in the map explorer
    assert not at.exception, [e.value for e in at.exception]


def test_app_explains_missing_data(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("WATERDET_APP_DATA", str(tmp_path / "nothing"))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert at.error and "Run the notebook" in at.error[0].value


def test_blank_numbers_do_not_crash_pack_or_app(tmp_path, data_dir, small_cfg, monkeypatch):
    """Regression: a season with no clear images leaves blank (None) numbers in summary.json."""
    from streamlit.testing.v1 import AppTest

    cfg = dict(small_cfg)
    cfg["reference"] = dict(cfg["reference"], csv=str(tmp_path / "missing.csv"))
    out = tmp_path / "out"
    run(cfg, data_dir, out)
    sj = json.loads((out / "summary.json").read_text())
    sj["areas"]["wet"] = {k: None for k in sj["areas"]["wet"]}
    sj["change"]["dry_to_wet"] = {k: None for k in sj["change"]["dry_to_wet"]}
    (out / "summary.json").write_text(json.dumps(sj))
    build_presentation_pack(cfg, data_dir, out, max_px=300)
    assert "n/a" in (out / "presentation" / "key_numbers.csv").read_text()
    monkeypatch.setenv("WATERDET_APP_DATA", str(out / "streamlit_data"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
