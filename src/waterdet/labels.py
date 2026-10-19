"""Training labels derived from JRC Global Surface Water (GSW), optionally cross-checked
against Dynamic World (DW).

Why not simply threshold GSW occurrence?
  * GSW is 30 m (Landsat), Sentinel-2 is 10 m: pixels near a shoreline are mislabelled
    by up to ~1 Landsat pixel (3 Sentinel-2 pixels). We therefore only keep pixels that
    lie `edge_px` inside a confident region and drop the uncertain band in between.
  * GSW v1.4 ends in 2021. Land reclaimed since (e.g. around Lagos) is still 'water' in it.
    Mode 'gsw_dw' requires that Dynamic World (10 m, current) agrees.

Label codes: 1 = water, 0 = land, -1 = ignore (ambiguous, invalid or unlabelled).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

IGNORE = -1


def _erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    # border_value=1: the raster edge is not a class boundary, do not eat the border.
    return ndimage.binary_erosion(mask, structure=np.ones((3, 3), bool), iterations=px, border_value=1)


def make_labels(
    occurrence: np.ndarray,
    max_extent: np.ndarray,
    valid: np.ndarray,
    *,
    min_occurrence: float = 75.0,
    edge_px: int = 3,
    mode: str = "gsw",
    dw_water: np.ndarray | None = None,
    dw_threshold: float = 0.5,
) -> np.ndarray:
    """Return an int8 label raster (1 water, 0 land, -1 ignore).

    water candidate : GSW occurrence >= min_occurrence
    land candidate  : GSW has never seen water here (max_extent == 0 and occurrence == 0)
    Both are eroded by edge_px so shoreline and mixed pixels are never used for training.
    In mode 'gsw_dw', DW must agree: water needs dw_water >= dw_threshold, land needs
    dw_water < dw_threshold (pixels with no DW value are ignored).
    """
    if mode not in ("gsw", "gsw_dw"):
        raise ValueError(f"mode must be 'gsw' or 'gsw_dw', got {mode!r}")
    if not (occurrence.shape == max_extent.shape == valid.shape):
        raise ValueError("occurrence, max_extent and valid must have the same shape")

    occ = np.nan_to_num(occurrence, nan=0.0)
    mx = np.nan_to_num(max_extent, nan=0.0)
    water = occ >= min_occurrence
    land = (mx == 0) & (occ == 0)

    if mode == "gsw_dw":
        if dw_water is None:
            raise ValueError("mode 'gsw_dw' needs the dw_water band")
        dw_ok = np.isfinite(dw_water)
        dw = np.where(dw_ok, dw_water, 0.0)
        water &= dw_ok & (dw >= dw_threshold)
        land &= dw_ok & (dw < dw_threshold)

    water = _erode(water, edge_px)
    land = _erode(land, edge_px)

    labels = np.full(occ.shape, IGNORE, dtype=np.int8)
    labels[land & valid] = 0
    labels[water & valid] = 1  # water and land cannot overlap (occ>=min vs occ==0), order is irrelevant
    return labels
