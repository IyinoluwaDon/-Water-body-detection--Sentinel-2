import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from waterdet.labels import make_labels
from waterdet.model import one_standard_error_choice, tree_rules, tune_decision_tree
from waterdet.sampling import draw_points, spatial_split


def _inputs(dry_bands):
    valid = np.ones(dry_bands["gsw_occurrence"].shape, bool)
    return dry_bands["gsw_occurrence"], dry_bands["gsw_max_extent"], valid, dry_bands["dw_water"]


def test_labels_edge_buffer_separates_classes(dry_bands):
    occ, mx, valid, _ = _inputs(dry_bands)
    lab = make_labels(occ, mx, valid, min_occurrence=75, edge_px=3, mode="gsw")
    assert set(np.unique(lab)) <= {-1, 0, 1} and (lab == 1).any() and (lab == 0).any()
    # every labelled water pixel is >= 4 px (Chebyshev) from every labelled land pixel
    dist_to_land = ndimage.distance_transform_cdt(lab != 0, metric="chessboard")
    assert dist_to_land[lab == 1].min() >= 4


def test_labels_dw_mode_drops_reclaimed_land(world, dry_bands):
    occ, mx, valid, dw = _inputs(dry_bands)
    core = np.zeros_like(valid)
    core[320:350, 380:440] = True                                     # well inside the reclaimed patch
    assert world["reclaimed"][core].all()
    gsw = make_labels(occ, mx, valid, mode="gsw", edge_px=3)
    both = make_labels(occ, mx, valid, mode="gsw_dw", dw_water=dw, edge_px=3)
    assert (gsw[core] == 1).mean() > 0.9           # old GSW calls it water (wrong)
    assert (both[core] == 1).sum() == 0            # DW disagrees -> never labelled water
    assert (both[core] == 0).sum() == 0            # ...and not used as land either: ignored


def test_labels_invalid_pixels_and_bad_args(dry_bands):
    occ, mx, valid, dw = _inputs(dry_bands)
    valid = valid.copy()
    valid[:, :100] = False
    lab = make_labels(occ, mx, valid, mode="gsw", edge_px=0)
    assert (lab[:, :100] == -1).all()
    with pytest.raises(ValueError):
        make_labels(occ, mx, valid, mode="gsw_dw")          # dw missing
    with pytest.raises(ValueError):
        make_labels(occ, mx, valid, mode="bogus")


def test_draw_points_balanced_and_errors():
    lab = -np.ones((100, 100), np.int8)
    lab[:50] = 0
    lab[50:60] = 1
    ps = draw_points(lab, n_per_class=300, block_px=20, seed=1)
    assert (ps.y == 1).sum() == 300 and (ps.y == 0).sum() == 300      # 1000 water px available
    assert all(lab[r, c] == y for r, c, y in zip(ps.rows, ps.cols, ps.y))
    ps2 = draw_points(lab, n_per_class=300, block_px=20, seed=1)
    assert (ps.rows == ps2.rows).all()                                  # deterministic
    lab[50:60] = -1
    with pytest.raises(ValueError, match="water"):
        draw_points(lab, n_per_class=300, block_px=20, seed=1)


def test_spatial_split_has_no_block_overlap(dry_bands):
    occ, mx, valid, _ = _inputs(dry_bands)
    lab = make_labels(occ, mx, valid, mode="gsw", edge_px=3)
    ps = draw_points(lab, n_per_class=2000, block_px=50, seed=3)
    tr, te = spatial_split(ps, test_size=0.3, seed=3)
    assert set(ps.block[tr]).isdisjoint(set(ps.block[te]))
    assert len(tr) + len(te) == len(ps) and abs(len(te) / len(ps) - 0.3) < 0.15
    for side in (tr, te):
        assert set(ps.y[side]) == {0, 1}


def test_spatial_split_failure_message():
    lab = np.zeros((100, 100), np.int8)
    lab[:20, :20] = 1                        # every water pixel sits inside ONE 50 px block
    ps = draw_points(lab, n_per_class=200, block_px=50, seed=0, min_per_class=60)
    with pytest.raises(ValueError):
        spatial_split(ps, test_size=0.3, seed=0, max_tries=5)


def test_one_se_rule_prefers_simpler_tree():
    cv = pd.DataFrame({
        "param_max_depth": [2, 4, 8], "param_min_samples_leaf": [1, 1, 1],
        "mean_test_score": [0.965, 0.968, 0.970], "std_test_score": [0.02, 0.02, 0.02],
    })
    # best = 0.970, SE = 0.02 / sqrt(5) = 0.0089 -> threshold 0.9611: depth 2 (0.965) qualifies and is simplest
    assert one_standard_error_choice(cv, 5)["max_depth"] == 2
    cv.loc[0, "mean_test_score"] = 0.940          # now depth 2 is outside one SE -> next simplest
    assert one_standard_error_choice(cv, 5)["max_depth"] == 4


def test_tune_decision_tree_end_to_end(dry_bands):
    from waterdet.indices import feature_stack
    from waterdet.io import SPECTRAL_BANDS
    occ, mx, valid, dw = _inputs(dry_bands)
    lab = make_labels(occ, mx, valid, mode="gsw_dw", dw_water=dw, edge_px=3)   # clean labels
    ps = draw_points(lab, n_per_class=2000, block_px=50, seed=0)
    X = feature_stack({b: dry_bands[b][ps.rows, ps.cols] for b in SPECTRAL_BANDS}).T
    model, cv, chosen = tune_decision_tree(X, ps.y, ps.block, depths=[1, 2, 3, 5], leaves=[1, 20], cv_folds=3,
                                           scoring="f1", seed=0)
    assert chosen["max_depth"] <= 3
    assert model.score(X, ps.y) > 0.97
    assert "NDWI" in tree_rules(model) or "B8" in tree_rules(model)
    with pytest.raises(ValueError, match="spatial blocks"):
        tune_decision_tree(X, ps.y, np.zeros(len(ps.y), int), depths=[2], leaves=[1], cv_folds=3, scoring="f1", seed=0)


def test_noisy_gsw_labels_hurt_the_tree(dry_bands):
    """The reclaimed-land trap: ~20% of 'water' labels are land in plain gsw mode, so fit quality drops."""
    from waterdet.indices import feature_stack
    from waterdet.io import SPECTRAL_BANDS
    occ, mx, valid, dw = _inputs(dry_bands)
    scores = {}
    for mode in ("gsw", "gsw_dw"):
        lab = make_labels(occ, mx, valid, mode=mode, dw_water=dw, edge_px=3)
        ps = draw_points(lab, n_per_class=2000, block_px=50, seed=0)
        X = feature_stack({b: dry_bands[b][ps.rows, ps.cols] for b in SPECTRAL_BANDS}).T
        model, _, _ = tune_decision_tree(X, ps.y, ps.block, depths=[1, 2, 3, 5], leaves=[1, 20], cv_folds=3, scoring="f1", seed=0)
        scores[mode] = model.score(X, ps.y)
    assert scores["gsw_dw"] > scores["gsw"] + 0.05
