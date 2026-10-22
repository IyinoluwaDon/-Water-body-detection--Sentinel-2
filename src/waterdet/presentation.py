"""Presentation pack and Streamlit app data, built from a finished pipeline run.

Reads  data/s2_<season>.tif  (the satellite bands)  and  outputs/  (maps, summary.json, model figures).
Writes outputs/presentation/   slide-ready PNGs, key_numbers.csv, README_presentation.md
       outputs/streamlit_data/ everything the Streamlit app in app/ needs (small files, safe to commit)

Nothing here recomputes the model. It only draws what the pipeline already produced.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

from .indices import mndwi, ndwi  # noqa: E402
from .io import read_stack  # noqa: E402

WATER_RGB = (0, 94, 184)
LAND_RGB = (245, 245, 240)
NODATA_RGB = (150, 150, 150)
CHANGE_RGB = {0: (245, 245, 240), 1: (158, 202, 225), 2: (27, 158, 119), 3: (217, 95, 2)}  # land, water, gain, loss
CHANGE_LABEL = {0: "stable land", 1: "stable water", 2: "water gained", 3: "water lost"}


def _num(v, nd: int = 2):
    """Round a number, or return 'n/a' when the pipeline left it blank (for example a season with no clear images)."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return round(v, nd)


def _step(shape, max_px: int) -> int:
    return max(1, math.ceil(max(shape) / max_px))


def true_colour(bands: dict, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (uint8 RGB image, valid mask) from B4, B3, B2, stretched to look natural."""
    rgb = np.stack([bands["B4"][::step, ::step], bands["B3"][::step, ::step], bands["B2"][::step, ::step]], axis=-1)
    valid = np.isfinite(rgb).all(axis=-1)
    out = np.zeros(rgb.shape, dtype=np.float32)
    if valid.any():
        lo, hi = np.percentile(rgb[valid], [2, 98])
        out = np.clip((np.nan_to_num(rgb) - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.8
    img = (out * 255).astype(np.uint8)
    img[~valid] = NODATA_RGB
    return img, valid


def _read_class_map(path: Path, step: int) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)[::step, ::step]


def _colourise(codes: np.ndarray, palette: dict[int, tuple], nodata: int = 255) -> np.ndarray:
    img = np.zeros(codes.shape + (3,), dtype=np.uint8)
    img[:] = NODATA_RGB
    for code, rgb in palette.items():
        img[codes == code] = rgb
    return img


def _overlay_rgba(codes: np.ndarray, palette: dict[int, tuple]) -> np.ndarray:
    rgba = np.zeros(codes.shape + (4,), dtype=np.uint8)
    for code, rgb in palette.items():
        rgba[codes == code] = (*rgb, 255)
    return rgba


def _save_fig(fig, path: Path) -> None:
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _title_for(period: str, cfg: dict) -> str:
    p = cfg["periods"][period]
    prefix = "SYNTHETIC DEMO | " if cfg['aoi'].get('name') == 'synthetic' else ""
    return f"{prefix}{period.capitalize()} ({p['start']} to {p['end']}, end exclusive)"


def build_presentation_pack(cfg: dict, data_dir="data", out_dir="outputs", max_px: int = 1400) -> dict:
    """Create the presentation pack and the Streamlit data folder. Returns the folders and file lists."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    primary = summary["primary_mode"]
    tp = summary["training_period"]
    pres, app = out_dir / "presentation", out_dir / "streamlit_data"
    periods = [p for p in sorted(cfg['periods'], key=lambda p: str(cfg['periods'][p]['start']))
               if p in summary['areas'] and (data_dir / f"s2_{p}.tif").exists()
               and (out_dir / "maps" / f"water_{p}.tif").exists()]
    if not periods:
        raise FileNotFoundError("No season has both data/s2_<season>.tif and outputs/maps/water_<season>.tif. Run Step 6 first.")
    for d in (pres, app):
        d.mkdir(parents=True, exist_ok=True)

    shape = read_stack(data_dir / f"s2_{tp}.tif").shape
    step = _step(shape, max_px)
    pixel_m = cfg["imagery"]["scale_m"] * step
    water_legend = [Patch(color=np.array(WATER_RGB) / 255, label="water"), Patch(color=np.array(LAND_RGB) / 255, label="land", ec="#999"),
                    Patch(color=np.array(NODATA_RGB) / 255, label="no clear image")]
    files: dict[str, list] = {"presentation": [], "streamlit_data": []}

    rgb_by_period = {}
    for p in periods:
        stack = read_stack(data_dir / f"s2_{p}.tif")
        rgb, valid = true_colour(stack.bands, step)
        rgb_by_period[p] = rgb
        codes = _read_class_map(out_dir / "maps" / f"water_{p}.tif", step)
        det = _colourise(codes, {0: LAND_RGB, 1: WATER_RGB})
        blend = rgb.copy()
        w = codes == 1
        blend[w] = (0.35 * rgb[w] + 0.65 * np.array(WATER_RGB)).astype(np.uint8)
        title = _title_for(p, cfg)

        fig, ax = plt.subplots(figsize=(9, 7.2)); ax.imshow(rgb); ax.axis("off")
        ax.set_title(f"Satellite image (Sentinel-2, true colour)\n{title}", fontsize=12)
        _save_fig(fig, pres / f"01_satellite_{p}.png")

        fig, ax = plt.subplots(figsize=(9, 7.2)); ax.imshow(det); ax.axis("off")
        ax.set_title(f"Detected water\n{title}", fontsize=12); ax.legend(handles=water_legend, loc="lower left", framealpha=0.95)
        _save_fig(fig, pres / f"02_detection_{p}.png")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
        axes[0].imshow(rgb); axes[0].set_title("What the satellite sees", fontsize=12)
        axes[1].imshow(blend); axes[1].set_title("Water found by the model (blue)", fontsize=12)
        for a in axes:
            a.axis("off")
        fig.suptitle(title, fontsize=13)
        _save_fig(fig, pres / f"03_before_after_{p}.png")

        Image.fromarray(rgb).save(app / f"rgb_{p}.png", optimize=True)
        Image.fromarray(_overlay_rgba(codes, {1: WATER_RGB})).save(app / f"water_{p}.png", optimize=True)
        shutil.copy(out_dir / "maps" / f"water_{p}.tif", app / f"water_{p}.tif")

        if p == tp:
            b = stack.bands
            layers = [("B2 blue", b["B2"], "Blues"), ("B3 green", b["B3"], "Greens"), ("B4 red", b["B4"], "Reds"),
                      ("B8 near-infrared", b["B8"], "magma"), ("B11 shortwave-infrared", b["B11"], "magma"),
                      ("NDWI (water index)", ndwi(b["B3"], b["B8"]), "RdBu"), ("MNDWI (water index)", mndwi(b["B3"], b["B11"]), "RdBu")]
            s2 = _step(shape, 600)
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            for ax, (name, arr, cmap) in zip(axes.ravel(), layers):
                arr = arr[::s2, ::s2]
                vmin, vmax = (-1, 1) if "NDWI" in name else np.nanpercentile(arr, [2, 98])
                im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax); ax.set_title(name, fontsize=11); ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            axes.ravel()[-1].axis("off")
            fig.suptitle("The data the model learns from: 5 satellite bands and 2 water indices", fontsize=14)
            _save_fig(fig, pres / f"04_model_inputs_{p}.png")

    # change map
    change_files = [out_dir / 'maps' / f'change_{key}.tif' for key in summary.get('change', {})
                    if (out_dir / 'maps' / f'change_{key}.tif').exists()]
    if change_files:
        ch = _read_class_map(change_files[0], step)
        pal = {k: v for k, v in CHANGE_RGB.items()}
        img = _colourise(ch, pal)
        fig, ax = plt.subplots(figsize=(9, 7.2)); ax.imshow(img); ax.axis("off")
        ax.set_title(f"Water change: {next(iter(summary['change']))}", fontsize=12)
        ax.legend(handles=[Patch(color=np.array(CHANGE_RGB[k]) / 255, label=CHANGE_LABEL[k], ec="#999") for k in (0, 1, 2, 3)],
                  loc="lower left", framealpha=0.95)
        _save_fig(fig, pres / "05_change_map.png")
        Image.fromarray(_overlay_rgba(ch, {k: v for k, v in CHANGE_RGB.items() if k >= 2})).save(app / "change.png", optimize=True)
        shutil.copy(change_files[0], app / "change.tif")
        if len(periods) >= 2:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
            for a, p in zip(axes[:2], periods[:2]):
                a.imshow(_colourise(_read_class_map(out_dir / "maps" / f"water_{p}.tif", step), {0: LAND_RGB, 1: WATER_RGB}))
                a.set_title(_title_for(p, cfg), fontsize=11)
            axes[2].imshow(img); axes[2].set_title("Change", fontsize=11)
            for a in axes:
                a.axis("off")
            _save_fig(fig, pres / "06_dry_wet_change.png")

    # key numbers
    rows = []
    for p, a in summary.get("areas", {}).items():
        rows += [(f"Water area, {p} season (km2)", _num(a.get("water_km2"))),
                 (f"Water share of mapped area, {p} (%)", _num(a.get("water_pct_of_valid"))),
                 (f"Area with no clear image, {p} (% of study area)", _num(a.get("nodata_pct_of_aoi")))]
    for k, ch_ in summary.get("change", {}).items():
        for name, v in ch_.items():
            rows.append((f"Change {k}: {name} (km2)", _num(v)))
    m = summary["modes"][primary]
    t = m["test_spatial_holdout"]
    for key, label in (("overall_accuracy", "Agreement with JRC labels, unseen areas (overall)"), ("kappa", "Kappa, unseen areas"),
                       ("water_f1", "Water F1, unseen areas"), ("water_iou", "Water IoU, unseen areas")):
        if key in t:
            rows.append((label, _num(t[key], 4)))
    rows.append(("Tree depth chosen", m["tree_depth"])); rows.append(("Tree leaves", m["tree_leaves"]))
    for name, bm in m.get("baselines_on_same_test_points", {}).items():
        rows.append((f"Baseline {name}: overall agreement", _num(bm.get("overall_accuracy"), 4)))
    ref = summary.get("reference_validation")
    if ref and ref.get("area_weighted"):
        rows.append(("Independent reference points used", ref.get("n_used")))
    df = pd.DataFrame(rows, columns=["metric", "value"])
    df["metric"] = df["metric"].str.replace("compared_km2 (km2)", "Area compared (km2)", regex=False)
    df["value"] = df["value"].map(lambda v: f"{v:g}" if isinstance(v, (int, float)) and v is not None else str(v))
    df.to_csv(pres / "key_numbers.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(df) + 0.9)); ax.axis("off")
    tb = ax.table(cellText=df.values.tolist(), colLabels=["Metric", "Value"], loc="center", cellLoc="left", colWidths=[0.8, 0.2])
    tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1, 1.35)
    for (r, c), cell in tb.get_celld().items():
        if r == 0:
            cell.set_facecolor("#005eb8"); cell.set_text_props(color="white", weight="bold")
    ax.set_title("Key numbers", fontsize=13, pad=0)
    _save_fig(fig, pres / "07_key_numbers.png")

    # model figures
    mdir = out_dir / f"mode_{primary}"
    # Put the model and simple alternatives on one plot with uncertainty resampled by whole held-out blocks.
    comparison = [("Decision Tree", m["test_spatial_holdout"])] + list(m["baselines_on_same_test_points"].items())
    comparison.sort(key=lambda item: item[1]["overall_accuracy"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (name, metrics) in enumerate(comparison):
        score = metrics["overall_accuracy"] * 100
        ci = metrics.get("ci95_block_bootstrap", {}).get("overall_accuracy", [None, None])
        error = None if any(x is None for x in ci) else [[max(0, score-ci[0]*100)], [max(0, ci[1]*100-score)]]
        tree_colour = tuple(channel / 255 for channel in WATER_RGB)
        ax.barh(i, score, color=tree_colour if name == "Decision Tree" else "#79a9d1")
        if error:
            ax.errorbar(score, i, xerr=error, fmt="none", ecolor="#222", capsize=3)
        ax.text(score + 0.04, i + 0.2, f"{score:.2f}%", va="bottom", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
    ax.set_yticks(range(len(comparison)), [x[0] for x in comparison])
    ax.set_xlim(min(95, comparison[0][1]["overall_accuracy"]*100 - .8), 100)
    ax.set_xlabel("Agreement with historical labels on held-out spatial blocks (%)")
    ax.set_title("Decision Tree and index thresholds, with block-bootstrap 95% intervals")
    ax.grid(axis="x", alpha=.25); fig.tight_layout()
    _save_fig(fig, pres / "12_threshold_comparison.png")
    shutil.copy(pres / "12_threshold_comparison.png", app / "threshold_comparison.png")

    for src, dst in (("tree.png", "08_decision_tree.png"), ("cv_depth_curve.png", "09_depth_choice.png"),
                     ("confusion_test.png", "10_confusion_matrix.png"), ("feature_importance.png", "11_feature_importance.png"),
                     ("spatial_validation.png", "13_spatial_validation.png"),
                     ("tree_rules.txt", "08_decision_tree_rules.txt")):
        if (mdir / src).exists():
            shutil.copy(mdir / src, pres / dst)
            shutil.copy(mdir / src, app / src)
    shutil.copy(out_dir / "summary.json", app / "summary.json")
    shutil.copy(pres / "key_numbers.csv", app / "key_numbers.csv")
    meta = {"periods": periods, "period_labels": {p: _title_for(p, cfg) for p in periods}, "bbox": cfg["aoi"]["bbox"],
            "aoi_name": cfg["aoi"].get("name", "study area"), "pixel_size_m": pixel_m, "primary_mode": primary, "training_period": tp,
            "has_change": bool(change_files)}
    (app / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    (pres / "README_presentation.md").write_text(
        "# Presentation pack\n\nSuggested slide order:\n\n"
        "1. `01_satellite_*.png`  The raw input: what the satellite sees.\n"
        "2. `04_model_inputs_*.png`  The data: 5 bands and 2 water indices the model learns from.\n"
        "3. `03_before_after_*.png`  The detection: image next to the water the model found.\n"
        "4. `06_dry_wet_change.png` and `05_change_map.png`  The monitoring result: seasonal change.\n"
        "5. `08_decision_tree.png` with `08_decision_tree_rules.txt`  How the model decides, in plain rules.\n"
        "6. `09_depth_choice.png`, `10_confusion_matrix.png`, `13_spatial_validation.png`  How it was tested and where it disagrees.\n"
        "7. `12_threshold_comparison.png`  Compare the tree with simple index thresholds and show block-bootstrap intervals.\n"
        "8. `07_key_numbers.png`  The main quantities on one slide.\n\n"
        "## One sentence to say out loud about accuracy\n"
        "\"These scores measure agreement with the JRC Global Surface Water map on areas the model never saw; "
        "they are not proof against ground truth, which needs independent reference points.\"\n", encoding="utf-8")

    files["presentation"] = sorted(p.name for p in pres.iterdir())
    files["streamlit_data"] = sorted(p.name for p in app.iterdir())
    return {"presentation_dir": str(pres), "streamlit_data_dir": str(app), "files": files, "downsample_step": step}
