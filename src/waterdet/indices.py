"""Spectral indices and the feature stack used by the classifier.

NDWI  (McFeeters 1996) = (Green - NIR)   / (Green + NIR)    -> Sentinel-2 (B3 - B8)  / (B3 + B8)
MNDWI (Xu 2006)        = (Green - SWIR1) / (Green + SWIR1)  -> Sentinel-2 (B3 - B11) / (B3 + B11)

Both lie in [-1, 1]; water is usually > 0 but the best threshold varies by scene,
which is exactly why a trained classifier is compared against fixed thresholds.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = ["B2", "B3", "B4", "B8", "B11", "NDWI", "MNDWI"]


def normalized_difference(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """(a - b) / (a + b) as float32. NaN where the denominator is ~0 or an input is NaN."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = a + b
    out = np.full(denom.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(denom) & (np.abs(denom) > eps)
    np.divide(a - b, denom, out=out, where=ok)
    return out


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return normalized_difference(green, nir)


def mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    return normalized_difference(green, swir1)


def feature_stack(bands: dict[str, np.ndarray]) -> np.ndarray:
    """Return a (7, H, W) float32 array ordered as FEATURE_NAMES."""
    for name in ("B2", "B3", "B4", "B8", "B11"):
        if name not in bands:
            raise KeyError(f"Missing band {name!r}; have {sorted(bands)}")
    b2, b3, b4, b8, b11 = (np.asarray(bands[k], dtype=np.float32) for k in ("B2", "B3", "B4", "B8", "B11"))
    return np.stack([b2, b3, b4, b8, b11, ndwi(b3, b8), mndwi(b3, b11)], axis=0)
