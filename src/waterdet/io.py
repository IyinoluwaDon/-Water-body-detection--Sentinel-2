"""Raster input/output for the Earth Engine export and the model outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

NODATA = -9999.0  # sentinel written by gee_export for masked pixels

SPECTRAL_BANDS = ["B2", "B3", "B4", "B8", "B11"]
RAW_BANDS = SPECTRAL_BANDS + ["n_obs"]
LABEL_BANDS = ["gsw_occurrence", "gsw_max_extent", "dw_water"]
EXPORT_BAND_ORDER = RAW_BANDS + LABEL_BANDS  # order used by gee_export


@dataclass
class Stack:
    """A multi-band export held in memory. Invalid pixels are NaN."""

    bands: dict[str, np.ndarray]
    transform: Affine
    crs: CRS
    path: Path | None = None
    _valid_cache: dict = field(default_factory=dict, repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        return next(iter(self.bands.values())).shape

    @property
    def has_labels(self) -> bool:
        return all(b in self.bands for b in LABEL_BANDS)

    @property
    def pixel_area_km2(self) -> float:
        if not self.crs.is_projected:
            raise ValueError("Area requires a projected CRS; reproject the raster to UTM first")
        _, factor = self.crs.linear_units_factor
        return abs(self.transform.a * self.transform.e - self.transform.b * self.transform.d) * factor**2 / 1e6

    def valid_mask(self, min_obs: int) -> np.ndarray:
        """True where all spectral bands are finite and at least min_obs clear observations exist."""
        if min_obs not in self._valid_cache:
            ok = np.isfinite(self.bands["n_obs"]) & (self.bands["n_obs"] >= min_obs)
            for name in SPECTRAL_BANDS:
                ok &= np.isfinite(self.bands[name])
            for other in ("B8", "B11"):
                ok &= np.abs(self.bands["B3"] + self.bands[other]) > 1e-9
            self._valid_cache[min_obs] = ok
        return self._valid_cache[min_obs]


def _band_names(src: rasterio.DatasetReader) -> list[str]:
    desc = list(src.descriptions)
    if desc and all(desc) and set(RAW_BANDS) <= set(desc):
        return desc
    if src.count == len(EXPORT_BAND_ORDER):
        return list(EXPORT_BAND_ORDER)
    if src.count == len(RAW_BANDS):
        return list(RAW_BANDS)
    raise ValueError(
        f"Cannot identify bands in {src.name}: {src.count} bands, descriptions={desc}. "
        f"Expected {len(EXPORT_BAND_ORDER)} bands in order {EXPORT_BAND_ORDER}."
    )


def check_label_bands(bands: dict[str, np.ndarray], *, where: str = "export", max_bad_fraction: float = 0.005) -> None:
    """Fail loudly if the label bands were altered on the way out of Earth Engine.

    The JRC GSW `occurrence` band is masked with its own values, so an `unmask` or resampling step could blend
    values (for example 75 becoming 56). Genuine GSW occurrence is a whole number from 0 to 100, `max_extent` is
    0 or 1, and Dynamic World `water` is a probability from 0 to 1. Anything else means the training labels
    would be silently wrong, so stop here instead of training on them.
    """
    problems = []
    if "gsw_occurrence" in bands:
        v = bands["gsw_occurrence"]
        v = v[np.isfinite(v)]
        if v.size:
            bad = (v < -1e-3) | (v > 100 + 1e-3) | (np.abs(v - np.round(v)) > 1e-3)
            if bad.mean() > max_bad_fraction:
                problems.append(f"gsw_occurrence: {bad.mean():.1%} of values are not whole numbers in 0-100")
    if "gsw_max_extent" in bands:
        v = bands["gsw_max_extent"]
        v = v[np.isfinite(v)]
        if v.size:
            bad = ~np.isin(np.round(v, 3), (0.0, 1.0))
            if bad.mean() > max_bad_fraction:
                problems.append(f"gsw_max_extent: {bad.mean():.1%} of values are not 0 or 1")
    if "dw_water" in bands:
        v = bands["dw_water"]
        v = v[np.isfinite(v)]
        if v.size:
            bad = (v < -1e-3) | (v > 1 + 1e-3)
            if bad.mean() > max_bad_fraction:
                problems.append(f"dw_water: {bad.mean():.1%} of values are outside 0-1")
    if problems:
        raise ValueError(
            f"Label bands in {where} look altered, so training labels would be wrong: " + "; ".join(problems)
            + ". Re-export from Earth Engine; if this repeats, open an issue with this message."
        )


def read_stack(path: str | Path) -> Stack:
    """Read a GeoTIFF exported by gee_export into a Stack (sentinel -9999 -> NaN)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Export it from Earth Engine first (see README).")
    with rasterio.open(path) as src:
        names = _band_names(src)
        data = src.read(masked=True).astype(np.float32).filled(np.nan)
        nodata = src.nodata
        transform, crs = src.transform, src.crs
    if crs is None:
        raise ValueError(f"{path} has no CRS")
    bad = data == NODATA
    if nodata is not None and np.isfinite(nodata):
        bad |= data == np.float32(nodata)
    data[bad] = np.nan
    bands = dict(zip(names, data))
    check_label_bands(bands, where=path.name)
    return Stack(bands=bands, transform=transform, crs=crs, path=path)


def write_raster(path: str | Path, array: np.ndarray, like: Stack, *, dtype: str, nodata) -> Path:
    """Write a single-band raster aligned to `like`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = array.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1, dtype=dtype,
        crs=like.crs, transform=like.transform, nodata=nodata, compress="lzw",
    ) as dst:
        dst.write(array.astype(dtype), 1)
    return path
