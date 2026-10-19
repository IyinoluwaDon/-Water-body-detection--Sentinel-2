"""Load and validate config.yaml. Fails early with a message that names the bad key."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from .geo import validate_bbox

PLACEHOLDER_PROJECT = "YOUR-GCP-PROJECT-ID"
LABEL_MODES = ("gsw", "gsw_dw")


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    validate_config(cfg)
    return cfg


def _need(cfg: dict, dotted: str):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ValueError(f"config.yaml is missing required key: {dotted}")
        cur = cur[part]
    return cur


def _parse_date(value, key: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{key} must be an ISO date (YYYY-MM-DD), got {value!r}") from exc


def validate_config(cfg: dict) -> None:
    """Raise ValueError on any invalid or missing setting."""
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must contain a mapping at the top level")

    validate_bbox(_need(cfg, "aoi.bbox"))

    periods = _need(cfg, "periods")
    if not isinstance(periods, dict) or not periods:
        raise ValueError("periods must be a non-empty mapping of name -> {start, end}")
    for name, p in periods.items():
        start = _parse_date(_need(p, "start"), f"periods.{name}.start")
        end = _parse_date(_need(p, "end"), f"periods.{name}.end")
        if end <= start:
            raise ValueError(f"periods.{name}: end ({end}) must be after start ({start}); end is exclusive")

    if _need(cfg, "training_period") not in periods:
        raise ValueError(f"training_period {cfg['training_period']!r} is not one of periods {sorted(periods)}")

    scale = _need(cfg, "imagery.scale_m")
    if not (scale and scale >= 10):
        raise ValueError("imagery.scale_m must be >= 10 (Sentinel-2 native resolution is 10 m)")
    if not (0.0 < _need(cfg, "imagery.cloud_score_threshold") < 1.0):
        raise ValueError("imagery.cloud_score_threshold must be between 0 and 1 (Google suggests 0.50-0.65)")
    if _need(cfg, "imagery.min_valid_obs") < 1:
        raise ValueError("imagery.min_valid_obs must be >= 1")

    if not (0 < _need(cfg, "labels.min_gsw_occurrence") <= 100):
        raise ValueError("labels.min_gsw_occurrence must be in (0, 100]")
    if _need(cfg, "labels.edge_buffer_px") < 0:
        raise ValueError("labels.edge_buffer_px must be >= 0")
    if not (0.0 < _need(cfg, "labels.dw_water_threshold") < 1.0):
        raise ValueError("labels.dw_water_threshold must be between 0 and 1 (a probability)")
    modes = _need(cfg, "labels.modes")
    bad = [m for m in modes if m not in LABEL_MODES]
    if bad or not modes:
        raise ValueError(f"labels.modes must be a non-empty subset of {LABEL_MODES}; got {modes}")
    if _need(cfg, "labels.primary_mode") not in modes:
        raise ValueError("labels.primary_mode must be listed in labels.modes")

    if _need(cfg, "sampling.n_per_class") < 100:
        raise ValueError("sampling.n_per_class must be >= 100")
    if not (0.1 <= _need(cfg, "sampling.test_size") <= 0.5):
        raise ValueError("sampling.test_size must be between 0.1 and 0.5")
    if _need(cfg, "sampling.block_size_m") < 10 * scale:
        raise ValueError("sampling.block_size_m is too small; use at least 10 pixels worth of metres")

    for key in ("model.max_depth", "model.min_samples_leaf"):
        vals = _need(cfg, key)
        if not isinstance(vals, list) or not vals or any((not isinstance(v, int)) or v < 1 for v in vals):
            raise ValueError(f"{key} must be a non-empty list of positive integers")
    if _need(cfg, "model.cv_folds") < 2:
        raise ValueError("model.cv_folds must be >= 2")


def require_gee_project(cfg: dict) -> str:
    """Return the GCP project id, refusing the placeholder."""
    project = _need(cfg, "gee.project")
    if not project or project == PLACEHOLDER_PROJECT:
        raise ValueError(
            "gee.project in config.yaml is still the placeholder. Put your Google Cloud project ID "
            "(the one registered for Earth Engine) there."
        )
    return project
