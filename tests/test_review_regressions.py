"""Scientific edge cases found during the proposal alignment review."""
import numpy as np
import pandas as pd
import pytest
from rasterio.crs import CRS
from rasterio.transform import Affine

from waterdet.evaluate import binary_metrics
from waterdet.io import Stack
from waterdet.labels import make_labels
from waterdet.reference import load_reference, strata_from_map


def test_missing_gsw_is_not_land():
    occ = np.array([[0., np.nan, 90., 0.]])
    extent = np.array([[0., 0., 1., np.nan]])
    assert make_labels(occ, extent, np.ones_like(occ, bool), edge_px=0).tolist() == [[0, -1, 1, -1]]


@pytest.mark.parametrize('pred', [[0, 0], [1, 0]])
def test_failed_water_predictions_have_zero_f1(pred):
    assert binary_metrics([0, 1], pred)['water_f1'] == 0


def test_no_water_in_either_classification_has_undefined_f1():
    assert np.isnan(binary_metrics([0, 0], [0, 0])['water_f1'])


def test_area_uses_affine_determinant_and_rejects_degrees():
    s = Stack({}, Affine(10, 2, 0, 3, -10, 0), CRS.from_epsg(32631))
    assert s.pixel_area_km2 == pytest.approx(106 / 1e6)
    s.crs = CRS.from_epsg(4326)
    with pytest.raises(ValueError, match='projected'):
        _ = s.pixel_area_km2


def test_undefined_indices_excluded_from_validity():
    bands = {b: np.ones((2, 2)) for b in ['B2', 'B3', 'B4', 'B8', 'B11', 'n_obs']}
    bands['B8'][0, 0] = -1
    s = Stack(bands, Affine.identity(), CRS.from_epsg(32631))
    assert not s.valid_mask(1)[0, 0]
    assert s.valid_mask(1).sum() == 3


def test_zero_shore_buffer_does_not_expand_until_convergence():
    a = np.array([[0, 1, 255]], dtype=np.uint8)
    assert strata_from_map(a, 0).tolist() == [[1, 0, -1]]


@pytest.mark.parametrize('field,values,message', [('id', ['a', 'a'], 'unique'),
                                                ('lon', [np.nan, 2], 'finite'),
                                                ('lat', [0, 91], 'finite')])
def test_invalid_reference_rejected(tmp_path, field, values, message):
    df = pd.DataFrame(dict(id=['a', 'b'], lon=[3, 3], lat=[6, 6], label=['land', 'water']))
    df[field] = values
    p = tmp_path / 'reference.csv'
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match=message):
        load_reference(p)
