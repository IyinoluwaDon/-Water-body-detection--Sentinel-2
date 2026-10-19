"""Independent reference (validation) points.

Why this exists: the training labels come from JRC GSW, so scoring the classifier against
GSW only measures agreement with GSW. Points that YOU label from high-resolution imagery
are the only check that is independent of the training labels.

Workflow
  1. `build_sampling_frame` writes a CSV + KML of random points, stratified by the map
     (water core / land core / shoreline). The CSV has NO map class in it, so labelling is blind.
  2. You open the KML in Google Earth, look at each point, and fill `label` with water / land / unsure.
  3. `load_reference` + `sample_at_points` read it back; `evaluate.weighted_metrics` gives
     area-weighted accuracy (the strata are not equal in area, so a plain average would mislead).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio.transform
import rasterio.warp
from scipy import ndimage

from .io import Stack
from .predict import MAP_NODATA

STRATA = ("water_core", "land_core", "shoreline")
_LABEL_MAP = {"water": 1, "w": 1, "1": 1, "land": 0, "l": 0, "0": 0, "non-water": 0, "nonwater": 0}


def meta_path(csv_path: str | Path) -> Path:
    p = Path(csv_path)
    return p.with_name(p.stem + ".meta.json")


def strata_from_map(map_arr: np.ndarray, shore_px: int) -> np.ndarray:
    """Integer raster: 0 water_core, 1 land_core, 2 shoreline, -1 no data."""
    valid = map_arr != MAP_NODATA
    water = map_arr == 1
    struct = np.ones((3, 3), bool)
    grown = ndimage.binary_dilation(water, structure=struct, iterations=shore_px)
    shrunk = ndimage.binary_erosion(water, structure=struct, iterations=shore_px, border_value=1)
    shore = grown & ~shrunk
    out = np.full(map_arr.shape, -1, dtype=np.int8)
    out[valid & water & ~shore] = 0
    out[valid & ~water & ~shore] = 1
    out[valid & shore] = 2
    return out


def build_sampling_frame(map_arr: np.ndarray, stack: Stack, *, n_by_stratum: dict, shore_px: int, seed: int,
                         period: str, out_dir: str | Path) -> dict:
    """Write reference_frame.csv / .kml / .meta.json into out_dir. Returns the meta dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    strata = strata_from_map(map_arr, shore_px)
    rng = np.random.default_rng(seed)
    rows_l, cols_l, names = [], [], []
    counts = {}
    for code, name in enumerate(STRATA):
        flat = np.flatnonzero(strata.ravel() == code)
        counts[name] = int(flat.size)
        want = int(n_by_stratum.get(name, 0))
        if want and flat.size == 0:
            raise ValueError(f"Stratum {name!r} has no pixels; cannot draw reference points from it")
        pick = rng.choice(flat, size=min(want, flat.size), replace=False) if want else np.array([], dtype=int)
        r, c = np.unravel_index(pick, strata.shape)
        rows_l.append(r); cols_l.append(c); names += [name] * len(pick)
    rows, cols = np.concatenate(rows_l), np.concatenate(cols_l)
    order = rng.permutation(len(rows))                  # shuffle so strata are not labelled in blocks
    rows, cols = rows[order], cols[order]
    names = [names[i] for i in order]
    xs, ys = rasterio.transform.xy(stack.transform, rows, cols, offset="center")
    lons, lats = rasterio.warp.transform(stack.crs, "EPSG:4326", list(xs), list(ys))
    ids = [f"P{i + 1:03d}" for i in range(len(rows))]
    df = pd.DataFrame({"id": ids, "lon": np.round(lons, 6), "lat": np.round(lats, 6), "label": "", "notes": ""})
    df.to_csv(out_dir / "reference_frame.csv", index=False)

    placemarks = "\n".join(
        f"<Placemark><name>{i}</name><Point><coordinates>{lo},{la},0</coordinates></Point></Placemark>"
        for i, lo, la in zip(df["id"], df["lon"], df["lat"])
    )
    (out_dir / "reference_frame.kml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n'
        f"<name>waterdet reference points</name>\n{placemarks}\n</Document></kml>\n", encoding="utf-8")

    total_valid = sum(counts.values())
    meta = {
        "period": period,
        "shore_px": shore_px,
        "seed": seed,
        "pixel_counts": counts,
        "weights": {k: v / total_valid for k, v in counts.items()},
        "strata": dict(zip(ids, names)),
        # lets load_reference() prove the labelled CSV belongs to this frame (ids alone would not)
        "coords": {i: [float(lo), float(la)] for i, lo, la in zip(df["id"], df["lon"], df["lat"])},
    }
    (out_dir / "reference_frame.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_reference(csv_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Read the filled-in CSV. Returns (dataframe with numeric `y`, info dict incl. dropped counts)."""
    df = pd.read_csv(csv_path, dtype={"id": str, "label": str})
    for col in ("id", "lon", "lat", "label"):
        if col not in df.columns:
            raise ValueError(f"{csv_path} is missing required column {col!r}")
    lab = df["label"].fillna("").str.strip().str.lower()
    n_blank = int((lab == "").sum())
    n_unsure = int(lab.isin(["unsure", "u", "unknown", "?"]).sum())
    keep = lab.isin(_LABEL_MAP)
    unknown = df.loc[~keep & ~lab.isin(["", "unsure", "u", "unknown", "?"]), ["id", "label"]]
    if len(unknown):
        raise ValueError(f"Unrecognised labels (use water / land / unsure): {unknown.to_dict('records')[:10]}")
    out = df.loc[keep].copy()
    out["y"] = lab[keep].map(_LABEL_MAP).astype(int)
    if not len(out):
        raise ValueError(f"{csv_path} has no labelled rows yet (fill the `label` column with water / land / unsure)")
    meta_file = meta_path(csv_path)
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else None
    if meta is not None:
        out["stratum"] = out["id"].map(meta["strata"])
        n_missing = int(out["stratum"].isna().sum())
        if n_missing:
            raise ValueError(f"{n_missing} reference ids are not in {meta_file.name}; do not edit the ids")
        coords = meta.get("coords")
        if coords:
            bad = [i for i, lo, la in zip(out["id"], out["lon"], out["lat"])
                   if not (abs(coords[i][0] - lo) < 1e-5 and abs(coords[i][1] - la) < 1e-5)]
            if bad:
                raise ValueError(
                    f"{len(bad)} reference points (e.g. {bad[:3]}) have coordinates that differ from {meta_file.name}. "
                    f"The CSV and the meta file come from different runs; use the pair written by the same run."
                )
    return out.reset_index(drop=True), {"n_blank": n_blank, "n_unsure": n_unsure, "meta": meta}


def sample_at_points(df: pd.DataFrame, stack: Stack, *, min_obs: int):
    """Row/col of each point in the stack. Drops points outside the raster or on invalid pixels.

    Returns (df_kept with `row`,`col`, info dict with counts dropped).
    """
    xs, ys = rasterio.warp.transform("EPSG:4326", stack.crs, df["lon"].tolist(), df["lat"].tolist())
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    inv = ~stack.transform  # plain arithmetic: the `Affine * tuple` operator is deprecated in newer affine releases
    cols = np.floor(inv.a * xs + inv.b * ys + inv.c).astype(int)
    rows = np.floor(inv.d * xs + inv.e * ys + inv.f).astype(int)
    h, w = stack.shape
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    valid = np.zeros(len(df), dtype=bool)
    valid[inside] = stack.valid_mask(min_obs)[rows[inside], cols[inside]]
    kept = df.loc[valid].copy()
    kept["row"], kept["col"] = rows[valid], cols[valid]
    info = {"n_outside": int((~inside).sum()), "n_invalid_pixel": int((inside & ~valid).sum()), "n_used": int(valid.sum())}
    return kept.reset_index(drop=True), info
