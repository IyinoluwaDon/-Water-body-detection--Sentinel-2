"""Apply a trained model to a whole raster, in row chunks to bound memory."""

from __future__ import annotations

import numpy as np

from .indices import feature_stack
from .io import SPECTRAL_BANDS, Stack

MAP_NODATA = 255


def predict_map(model, stack: Stack, *, min_obs: int, chunk_rows: int = 256) -> np.ndarray:
    """uint8 raster: 0 = land, 1 = water, 255 = no data (cloud-masked / too few observations)."""
    h, w = stack.shape
    valid = stack.valid_mask(min_obs)
    out = np.full((h, w), MAP_NODATA, dtype=np.uint8)
    for r0 in range(0, h, chunk_rows):
        r1 = min(r0 + chunk_rows, h)
        sl = {b: stack.bands[b][r0:r1] for b in SPECTRAL_BANDS}
        feats = feature_stack(sl)                       # (7, rows, W)
        ok = valid[r0:r1] & np.isfinite(feats).all(axis=0)
        if not ok.any():
            continue
        X = feats[:, ok].T                              # (n_valid, 7)
        out_chunk = out[r0:r1]
        out_chunk[ok] = model.predict(X).astype(np.uint8)
    return out


def water_area_km2(map_arr: np.ndarray, pixel_area_km2: float) -> dict:
    """Area summary of a predicted map."""
    water = int((map_arr == 1).sum())
    land = int((map_arr == 0).sum())
    nodata = int((map_arr == MAP_NODATA).sum())
    total = water + land + nodata
    return {
        "water_km2": water * pixel_area_km2,
        "land_km2": land * pixel_area_km2,
        "nodata_km2": nodata * pixel_area_km2,
        "water_pct_of_valid": 100.0 * water / (water + land) if (water + land) else float("nan"),
        "nodata_pct_of_aoi": 100.0 * nodata / total if total else float("nan"),
    }
