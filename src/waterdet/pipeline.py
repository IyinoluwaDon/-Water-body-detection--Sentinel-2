"""End-to-end local pipeline: labels -> spatial split -> tuned Decision Tree -> evaluation -> maps.

Run:  python -m waterdet.pipeline --config config.yaml --data-dir data --out-dir outputs
Expects data/s2_<period>.tif for each period in config (written by waterdet.gee_export).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from datetime import date
from pathlib import Path

import joblib
import numpy as np

from . import plots
from .change import change_map, change_stats
from .config import load_config
from .evaluate import (binary_metrics, block_bootstrap_ci, otsu_threshold, stratified_bootstrap_ci,
                       weighted_metrics)
from .indices import FEATURE_NAMES, feature_stack, normalized_difference
from .io import SPECTRAL_BANDS, Stack, read_stack, write_raster
from .labels import make_labels
from .model import tree_rules, tune_decision_tree
from .predict import MAP_NODATA, predict_map, water_area_km2
from .reference import build_sampling_frame, load_reference, sample_at_points
from .sampling import draw_points, spatial_split

log = logging.getLogger("waterdet")
NDWI_COL, MNDWI_COL = FEATURE_NAMES.index("NDWI"), FEATURE_NAMES.index("MNDWI")


def _clean(obj):
    """Make nested results JSON-safe (numpy -> python, NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _features_at(stack: Stack, rows, cols) -> np.ndarray:
    """(N, 7) feature matrix gathered at pixel positions (indices computed on the gathered values)."""
    gathered = {b: stack.bands[b][rows, cols] for b in SPECTRAL_BANDS}
    return feature_stack(gathered).T


def _baseline_thresholds(stack: Stack, min_obs: int) -> dict:
    """Otsu thresholds from the whole valid image (unsupervised), plus the textbook fixed 0."""
    valid = stack.valid_mask(min_obs)
    ndwi = normalized_difference(stack.bands["B3"][valid], stack.bands["B8"][valid])
    mndwi = normalized_difference(stack.bands["B3"][valid], stack.bands["B11"][valid])
    return {"NDWI>0": (NDWI_COL, 0.0), "MNDWI>0": (MNDWI_COL, 0.0),
            "NDWI Otsu": (NDWI_COL, otsu_threshold(ndwi)), "MNDWI Otsu": (MNDWI_COL, otsu_threshold(mndwi))}


def _baseline_predictions(X: np.ndarray, thresholds: dict) -> dict:
    return {name: (X[:, col] > thr).astype(int) for name, (col, thr) in thresholds.items()}


def _train_one_mode(cfg: dict, stacks: dict, mode: str, out_dir: Path, notes: list) -> tuple:
    tp, min_obs, L, S, M = cfg["training_period"], cfg["imagery"]["min_valid_obs"], cfg["labels"], cfg["sampling"], cfg["model"]
    st = stacks[tp]
    if not st.has_labels:
        raise ValueError(f"s2_{tp}.tif has no label bands (gsw_occurrence, gsw_max_extent, dw_water); re-export it")
    mode_dir = out_dir / f"mode_{mode}"
    mode_dir.mkdir(parents=True, exist_ok=True)

    finite = np.ones(st.shape, bool)
    for b in SPECTRAL_BANDS:
        finite &= np.isfinite(st.bands[b])
    valid = st.valid_mask(min_obs) & finite
    labels = make_labels(st.bands["gsw_occurrence"], st.bands["gsw_max_extent"], valid,
                         min_occurrence=L["min_gsw_occurrence"], edge_px=L["edge_buffer_px"], mode=mode,
                         dw_water=st.bands.get("dw_water"), dw_threshold=L["dw_water_threshold"])
    label_counts = {"water_px": int((labels == 1).sum()), "land_px": int((labels == 0).sum()),
                    "ignored_px": int((labels == -1).sum())}
    block_px = max(1, int(round(S["block_size_m"] / cfg["imagery"]["scale_m"])))
    ps = draw_points(labels, n_per_class=S["n_per_class"], block_px=block_px, seed=cfg["seed"])
    X = _features_at(st, ps.rows, ps.cols)
    tr, te = spatial_split(ps, test_size=S["test_size"], seed=cfg["seed"])

    model, cv, chosen = tune_decision_tree(X[tr], ps.y[tr], ps.block[tr], depths=M["max_depth"],
                                           leaves=M["min_samples_leaf"], cv_folds=M["cv_folds"],
                                           scoring=M.get("scoring", "f1"), seed=cfg["seed"])
    pred_te = model.predict(X[te])
    from .diagnostics import save_holdout
    save_holdout(ps, tr, te, pred_te, st.shape, mode_dir)
    test = binary_metrics(ps.y[te], pred_te)
    test["ci95_block_bootstrap"] = block_bootstrap_ci(ps.y[te], pred_te, ps.block[te], seed=cfg["seed"])
    train = binary_metrics(ps.y[tr], model.predict(X[tr]))

    thresholds = _baseline_thresholds(st, min_obs)
    baselines = {name: binary_metrics(ps.y[te], p) for name, p in _baseline_predictions(X[te], thresholds).items()}
    for name, metrics in baselines.items():
        metrics["ci95_block_bootstrap"] = block_bootstrap_ci(
            ps.y[te], _baseline_predictions(X[te], {name: thresholds[name]})[name], ps.block[te], seed=cfg["seed"]
        )
    baselines_thr = {name: thr for name, (_, thr) in thresholds.items()}

    transfer = {}
    for period, other in stacks.items():
        if period == tp:
            continue
        rows, cols = ps.rows[te], ps.cols[te]
        Xo = _features_at(other, rows, cols)
        ok = other.valid_mask(min_obs)[rows, cols] & np.isfinite(Xo).all(axis=1)
        if ok.sum() < 50 or len(np.unique(ps.y[te][ok])) < 2:
            transfer[period] = {"skipped": f"only {int(ok.sum())} usable test points in {period}"}
            continue
        m = binary_metrics(ps.y[te][ok], model.predict(Xo[ok]))
        m["n_dropped_invalid"] = int((~ok).sum())
        transfer[period] = m
        # Static GSW labels cannot describe seasonal water, so this tests transfer on permanent water/land only.

    joblib.dump(model, mode_dir / "model.joblib")
    (mode_dir / "tree_rules.txt").write_text(tree_rules(model), encoding="utf-8")
    cv.drop(columns=["params"], errors="ignore").to_csv(mode_dir / "cv_results.csv", index=False)
    plots.plot_confusion(test["confusion_matrix"], f"Held-out spatial blocks ({mode})", mode_dir / "confusion_test.png")
    plots.plot_cv_curve(cv, mode_dir / "cv_depth_curve.png")
    plots.plot_tree_figure(model, mode_dir / "tree.png")
    plots.plot_importance(model, mode_dir / "feature_importance.png")

    result = {
        "mode": mode, "label_pixels": label_counts, "n_points": len(ps),
        "n_train_points": len(tr), "n_test_points": len(te),
        "n_train_blocks": int(len(np.unique(ps.block[tr]))), "n_test_blocks": int(len(np.unique(ps.block[te]))),
        "chosen_params": chosen, "tree_depth": int(model.get_depth()), "tree_leaves": int(model.get_n_leaves()),
        "train": train, "test_spatial_holdout": test, "baselines_on_same_test_points": baselines,
        "baseline_thresholds": baselines_thr,
        "feature_importance": dict(zip(FEATURE_NAMES, model.feature_importances_.tolist())),
        "temporal_transfer": transfer,
    }
    gap = train["overall_accuracy"] - test["overall_accuracy"]
    if gap > 0.03:
        notes.append(f"[{mode}] train accuracy exceeds spatial hold-out accuracy by {gap:.3f}: the tree may be over-fitting; "
                     f"consider fewer depths in model.max_depth or a larger min_samples_leaf.")
    return model, result


def _evaluate_reference(cfg: dict, stacks: dict, models: dict, csv_path: Path, notes: list) -> dict | None:
    if not csv_path.exists():
        return None
    df, info = load_reference(csv_path)
    meta = info["meta"]
    period = (meta or {}).get("period", cfg["training_period"])
    if period not in stacks:
        raise ValueError(f"Reference points belong to period {period!r}, which has no raster in the data folder")
    st, min_obs = stacks[period], cfg["imagery"]["min_valid_obs"]
    kept, sinfo = sample_at_points(df, st, min_obs=min_obs)
    X = _features_at(st, kept["row"].to_numpy(), kept["col"].to_numpy())
    ok = np.isfinite(X).all(axis=1)
    kept, X = kept.loc[ok].reset_index(drop=True), X[ok]
    y = kept["y"].to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("Reference points contain only one class; label more points")

    preds = {f"model[{m}]": mdl.predict(X) for m, mdl in models.items()}
    preds.update(_baseline_predictions(X, _baseline_thresholds(st, min_obs)))

    weighted, why = False, None
    if meta is not None and "stratum" in kept:
        weights = meta["weights"]
        present = set(kept["stratum"])
        missing = [h for h in weights if weights[h] > 0 and h not in present]
        if missing:
            why = f"no labelled points left in strata {missing}"
        else:
            weighted = True
    else:
        why = "no reference_points.meta.json next to the CSV"
    out = {"period": period, "n_labelled": int(len(df)), **sinfo,
           "n_blank_labels": info["n_blank"], "n_unsure": info["n_unsure"], "area_weighted": weighted}
    if not weighted:
        notes.append(f"Reference accuracy is UNWEIGHTED ({why}); treat it as indicative only.")
    results = {}
    for name, p in preds.items():
        if weighted:
            w = {h: v for h, v in meta["weights"].items() if h in set(kept["stratum"])}
            m = weighted_metrics(y, p, kept["stratum"].to_numpy(), w)
            m["ci95_stratified_bootstrap"] = stratified_bootstrap_ci(y, p, kept["stratum"].to_numpy(), w, seed=cfg["seed"])
        else:
            m = binary_metrics(y, p)
            m["ci95_bootstrap"] = block_bootstrap_ci(y, p, np.arange(len(y)), seed=cfg["seed"])
        results[name] = m
    out["methods"] = results
    out["n_used"] = int(len(kept))
    n_attempted = len(df) + info["n_unsure"]
    if n_attempted and info["n_unsure"] / n_attempted > 0.10:
        notes.append(f"{info['n_unsure']} reference points were marked 'unsure' (>10%). Dropping hard points makes "
                     f"accuracy look better than it is; say so when you present the numbers.")
    if out["n_used"] < 100:
        notes.append(f"Only {out['n_used']} reference points were usable; aim for >= 200 for stable estimates.")
    return out


def _period_order(cfg: dict) -> list[str]:
    return sorted(cfg["periods"], key=lambda p: date.fromisoformat(str(cfg["periods"][p]["start"])))


def _f(x, nd: int = 4) -> str:
    """Format a number that may be None (NaN becomes None in the JSON-safe summary)."""
    return "n/a" if x is None else f"{x:.{nd}f}"


def _ci(pair) -> str:
    return "n/a" if not pair or pair[0] is None or pair[1] is None else f"[{pair[0]:.3f}, {pair[1]:.3f}]"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _provenance(cfg: dict, stacks: dict) -> dict:
    """What produced these numbers: package versions, input file hashes, config (project id redacted)."""
    from importlib import metadata

    def ver(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None

    snap = json.loads(json.dumps(cfg, default=str))
    snap.setdefault("gee", {})["project"] = "<redacted>"
    return {
        "python": platform.python_version(),
        "packages": {p: ver(p) for p in ("waterdet", "numpy", "scipy", "pandas", "scikit-learn", "rasterio",
                                          "earthengine-api")},
        "inputs_sha256": {p: _sha256(s.path) for p, s in stacks.items() if s.path is not None},
        "config": snap,
    }


def _write_summary(path: Path, summary: dict) -> None:
    L = ["# waterdet results", ""]
    for mode, r in summary["modes"].items():
        t, tr = r["test_spatial_holdout"], r["train"]
        ci = t["ci95_block_bootstrap"]
        L += [f"## Label mode `{mode}` (training period `{summary['training_period']}`)", "",
              f"- Tree: depth {r['tree_depth']}, {r['tree_leaves']} leaves, params {r['chosen_params']}",
              f"- Points: {r['n_train_points']} train / {r['n_test_points']} test "
              f"({r['n_train_blocks']} / {r['n_test_blocks']} spatial blocks)",
              f"- Train overall accuracy: {_f(tr['overall_accuracy'])} (shown only to expose over-fitting)", "",
              "| Method (spatial hold-out) | Overall acc. [95% CI] | Kappa [95% CI] | Water F1 | Water IoU |",
              "|---|---|---|---|---|",
              f"| **Decision Tree** | {_f(t['overall_accuracy'])} {_ci(ci['overall_accuracy'])} | "
              f"{_f(t['kappa'])} {_ci(ci['kappa'])} | {_f(t['water_f1'])} | {_f(t['water_iou'])} |"]
        for name, b in r["baselines_on_same_test_points"].items():
            L.append(f"| {name} (thr {_f(r['baseline_thresholds'][name], 3)}) | {_f(b['overall_accuracy'])} | "
                     f"{_f(b['kappa'])} | {_f(b['water_f1'])} | {_f(b['water_iou'])} |")
        L += ["", "_These scores measure agreement with JRC-derived labels on held-out spatial blocks, using a "
                  "approximately equal per-class sample (the block split can shift the ratio). They are NOT independent truth, "
                  "and water precision / F1 / IoU change "
                  "with the class ratio. Independent, area-weighted reference validation is still needed for ground-truth accuracy._", ""]
    ref = summary.get("reference_validation")
    if ref:
        L += ["## Independent reference validation", "",
              f"{ref['n_used']} usable points ({'area-weighted' if ref['area_weighted'] else 'UNWEIGHTED'}); "
              f"{ref['n_outside']} outside raster, {ref['n_invalid_pixel']} on invalid pixels, {ref['n_unsure']} marked unsure.", "",
              "| Method | Overall acc. | Kappa | Water precision | Water recall | Water F1 | Water IoU |",
              "|---|---|---|---|---|---|---|"]
        for name, m in ref["methods"].items():
            L.append(f"| {name} | {_f(m['overall_accuracy'])} | {_f(m['kappa'])} | {_f(m['water_precision'])} | "
                     f"{_f(m['water_recall'])} | {_f(m['water_f1'])} | {_f(m['water_iou'])} |")
        L.append("")
    L += ["## Water extent (primary model)", "", "| Period | Water km² | Land km² | No data % |", "|---|---|---|---|"]
    for p, a in summary["areas"].items():
        L.append(f"| {p} | {_f(a['water_km2'], 2)} | {_f(a['land_km2'], 2)} | {_f(a['nodata_pct_of_aoi'], 1)} |")
    for k, c in summary.get("change", {}).items():
        L += ["", f"### Change {k}", ""] + [f"- {n}: {_f(v, 2)} km²" for n, v in c.items()]
    if summary["notes"]:
        L += ["", "## Notes and warnings", ""] + [f"- {n}" for n in summary["notes"]]
    prov = summary.get("provenance")
    if prov:
        L += ["", "## Provenance", "", f"- Python {prov['python']}; packages: " +
              ", ".join(f"{k} {v}" for k, v in prov["packages"].items() if v)]
        L += [f"- Input `s2_{k}.tif` sha256 `{v[:16]}…`" for k, v in prov["inputs_sha256"].items()]
        L += ["- Full config snapshot (GEE project id redacted) is in `summary.json`."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def run(cfg: dict, data_dir: Path, out_dir: Path) -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    tp = cfg["training_period"]

    stacks: dict[str, Stack] = {}
    for p in _period_order(cfg):
        f = data_dir / f"s2_{p}.tif"
        if f.exists():
            stacks[p] = read_stack(f)
        elif p == tp:
            raise FileNotFoundError(f"{f} not found. It is the training period; export it from Earth Engine first.")
        else:
            notes.append(f"Period {p!r} skipped: {f.name} not found.")
    ref_stack = stacks[tp]
    _ = ref_stack.pixel_area_km2  # reject angular grids before training or writing area statistics
    resolution = (abs(ref_stack.transform.a), abs(ref_stack.transform.e))
    if (ref_stack.transform.b != 0 or ref_stack.transform.d != 0
            or not np.allclose(resolution, cfg['imagery']['scale_m'])
            or ref_stack.crs.linear_units_factor[1] != 1):
        raise ValueError("Input grid must be north-up with metre pixels matching imagery.scale_m")
    for p, s in stacks.items():
        if s.shape != ref_stack.shape or s.transform != ref_stack.transform or s.crs != ref_stack.crs:
            raise ValueError(f"s2_{p}.tif is not on the same grid as s2_{tp}.tif. Re-export both with the same AOI/scale/CRS.")

    models, results = {}, {}
    for mode in cfg["labels"]["modes"]:
        log.info("Training label mode %s", mode)
        models[mode], results[mode] = _train_one_mode(cfg, stacks, mode, out_dir, notes)

    primary = cfg["labels"]["primary_mode"]
    maps, areas = {}, {}
    for p, s in stacks.items():
        m = predict_map(models[primary], s, min_obs=cfg["imagery"]["min_valid_obs"])
        maps[p] = m
        write_raster(out_dir / "maps" / f"water_{p}.tif", m, s, dtype="uint8", nodata=MAP_NODATA)
        plots.plot_map(m, s, f"Water map ({p}, mode {primary}); blue = water, grey = no data", out_dir / "maps" / f"water_{p}.png")
        areas[p] = water_area_km2(m, s.pixel_area_km2)
        if areas[p]["nodata_pct_of_aoi"] > 15:
            notes.append(f"{p}: {areas[p]['nodata_pct_of_aoi']:.1f}% of the AOI has too few clear observations "
                         f"(min_valid_obs={cfg['imagery']['min_valid_obs']}). Widen the date window for that period.")

    change = {}
    order = [p for p in _period_order(cfg) if p in maps]
    for a, b in zip(order, order[1:]):
        c = change_map(maps[a], maps[b])
        write_raster(out_dir / "maps" / f"change_{a}_to_{b}.tif", c, ref_stack, dtype="uint8", nodata=MAP_NODATA)
        plots.plot_change(c, f"Water change {a} -> {b}", out_dir / "maps" / f"change_{a}_to_{b}.png")
        change[f"{a}_to_{b}"] = change_stats(c, ref_stack.pixel_area_km2)

    fr = cfg.get("reference", {})
    frame_dir = out_dir / "reference_frame"
    build_sampling_frame(maps[tp], ref_stack, n_by_stratum=fr.get("frame_n", {"water_core": 80, "land_core": 80, "shoreline": 60}),
                         shore_px=fr.get("shore_px", 2), seed=cfg["seed"], period=tp, out_dir=frame_dir)
    ref_csv = Path(fr.get("csv", data_dir / "reference_points.csv"))
    reference = _evaluate_reference(cfg, stacks, models, ref_csv, notes)
    if reference is None:
        notes.append(f"No independent reference points found at {ref_csv}. Accuracy above is agreement with JRC-derived "
                     f"labels only. Optional: docs/OPTIONAL_ACCURACY_CHECK.md explains how to get a true accuracy figure.")

    summary = {"training_period": tp, "primary_mode": primary, "modes": results, "areas": areas, "change": change,
               "reference_validation": reference, "notes": notes, "provenance": _provenance(cfg, stacks)}
    (out_dir / "summary.json").write_text(json.dumps(_clean(summary), indent=2), encoding="utf-8")
    _write_summary(out_dir / "summary.md", _clean(summary))
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--presentation", action="store_true", help="Also build slide figures and app data")
    args = ap.parse_args(argv)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(Path(args.out_dir) / 'run.log', encoding='utf-8')])
    cfg = load_config(args.config)
    try:
        summary = run(cfg, Path(args.data_dir), Path(args.out_dir))
        if args.presentation:
            from .presentation import build_presentation_pack
            build_presentation_pack(cfg, args.data_dir, args.out_dir)
    except Exception:
        log.exception("Pipeline failed")
        raise
    log.info("Run complete: %s", Path(args.out_dir) / 'summary.json')
    print(f"Done. See {Path(args.out_dir) / 'summary.md'}")
    for n in summary["notes"]:
        print("NOTE:", n)


if __name__ == "__main__":
    main()
