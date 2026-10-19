"""The label-band integrity check must pass clean GSW/DW values and reject blended ones."""

import numpy as np
import pytest

from waterdet.io import check_label_bands


def _clean(n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "gsw_occurrence": rng.integers(0, 101, n).astype(np.float32),
        "gsw_max_extent": rng.integers(0, 2, n).astype(np.float32),
        "dw_water": rng.random(n).astype(np.float32),
    }


def test_clean_bands_pass_and_nan_is_ignored():
    b = _clean()
    b["gsw_occurrence"][:50] = np.nan  # masked pixels are NaN after read_stack
    check_label_bands(b)


def test_blended_occurrence_is_rejected():
    b = _clean()
    b["gsw_occurrence"] = b["gsw_occurrence"] * b["gsw_occurrence"] / 100.0  # the mask-blend failure mode
    with pytest.raises(ValueError, match="gsw_occurrence"):
        check_label_bands(b)


def test_non_binary_max_extent_and_out_of_range_dw_are_rejected():
    b = _clean()
    b["gsw_max_extent"] = b["gsw_max_extent"] * 0.6
    with pytest.raises(ValueError, match="gsw_max_extent"):
        check_label_bands(b)
    b = _clean()
    b["dw_water"] = b["dw_water"] * 3
    with pytest.raises(ValueError, match="dw_water"):
        check_label_bands(b)
