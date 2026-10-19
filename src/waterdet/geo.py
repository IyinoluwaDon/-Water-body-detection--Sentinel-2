"""Small geographic helpers (no heavy dependencies)."""

from __future__ import annotations

import math

_M_PER_DEG_LAT = 110_574.0  # approximate near the equator; adequate for size checks
_M_PER_DEG_LON_EQ = 111_320.0


def validate_bbox(bbox) -> tuple[float, float, float, float]:
    """Check a [lon_min, lat_min, lon_max, lat_max] box and return it as a tuple."""
    if bbox is None or len(bbox) != 4:
        raise ValueError("aoi.bbox must have 4 numbers: [lon_min, lat_min, lon_max, lat_max]")
    lon_min, lat_min, lon_max, lat_max = (float(v) for v in bbox)
    if not (-180 <= lon_min < lon_max <= 180):
        raise ValueError(f"Invalid longitudes in aoi.bbox: {lon_min}, {lon_max} (need -180 <= min < max <= 180)")
    if not (-90 <= lat_min < lat_max <= 90):
        raise ValueError(f"Invalid latitudes in aoi.bbox: {lat_min}, {lat_max} (need -90 <= min < max <= 90)")
    return lon_min, lat_min, lon_max, lat_max


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat). 326xx north, 327xx south."""
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def bbox_size_km(bbox) -> tuple[float, float]:
    """Approximate (width_km, height_km) of a bbox."""
    lon_min, lat_min, lon_max, lat_max = validate_bbox(bbox)
    mid_lat = math.radians((lat_min + lat_max) / 2.0)
    width = (lon_max - lon_min) * _M_PER_DEG_LON_EQ * math.cos(mid_lat) / 1000.0
    height = (lat_max - lat_min) * _M_PER_DEG_LAT / 1000.0
    return width, height


def estimate_pixels(bbox, scale_m: float) -> int:
    """Approximate pixel count of an export of bbox at scale_m metres per pixel."""
    w_km, h_km = bbox_size_km(bbox)
    return int(round((w_km * 1000.0 / scale_m) * (h_km * 1000.0 / scale_m)))
