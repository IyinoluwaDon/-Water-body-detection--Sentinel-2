"""Water change between two dates (maps from predict_map)."""

from __future__ import annotations

import numpy as np

from .predict import MAP_NODATA

CODES = {0: "stable land", 1: "stable water", 2: "water gain (land -> water)", 3: "water loss (water -> land)"}


def change_map(map_a: np.ndarray, map_b: np.ndarray) -> np.ndarray:
    """uint8: 0 stable land, 1 stable water, 2 gain (a land, b water), 3 loss (a water, b land), 255 no data."""
    if map_a.shape != map_b.shape:
        raise ValueError(f"Map shapes differ: {map_a.shape} vs {map_b.shape}")
    out = np.full(map_a.shape, MAP_NODATA, dtype=np.uint8)
    ok = (map_a != MAP_NODATA) & (map_b != MAP_NODATA)
    a, b = map_a[ok], map_b[ok]
    code = np.zeros(a.shape, dtype=np.uint8)
    code[(a == 1) & (b == 1)] = 1
    code[(a == 0) & (b == 1)] = 2
    code[(a == 1) & (b == 0)] = 3
    out[ok] = code
    return out


def change_stats(change: np.ndarray, pixel_area_km2: float) -> dict:
    out = {name: float((change == code).sum() * pixel_area_km2) for code, name in CODES.items()}
    out["compared_km2"] = float((change != MAP_NODATA).sum() * pixel_area_km2)
    return out
