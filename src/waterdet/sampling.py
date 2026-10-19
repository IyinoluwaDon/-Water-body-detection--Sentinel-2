"""Class-balanced point sampling and *spatial* train/test splitting.

Neighbouring 10 m pixels are almost identical, so a random pixel split puts near-duplicates
in both train and test and inflates accuracy. Here points are grouped into square blocks
and whole blocks are assigned to either train or test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PointSet:
    rows: np.ndarray
    cols: np.ndarray
    y: np.ndarray
    block: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> "PointSet":
        return PointSet(self.rows[idx], self.cols[idx], self.y[idx], self.block[idx])


def draw_points(labels: np.ndarray, *, n_per_class: int, block_px: int, seed: int, min_per_class: int = 60) -> PointSet:
    """Randomly draw up to n_per_class pixels from each class (0 and 1) of a label raster."""
    rng = np.random.default_rng(seed)
    h, w = labels.shape
    n_block_cols = int(np.ceil(w / block_px))
    rows_all, cols_all, y_all = [], [], []
    for cls in (0, 1):
        flat = np.flatnonzero(labels.ravel() == cls)
        if flat.size < min_per_class:
            raise ValueError(
                f"Class {cls} ({'water' if cls else 'land'}) has only {flat.size} labelled pixels "
                f"(need >= {min_per_class}). The AOI may contain too little of this class, labels may be "
                f"over-filtered (try a smaller labels.edge_buffer_px), or the composite has many invalid pixels."
            )
        pick = rng.choice(flat, size=min(n_per_class, flat.size), replace=False)
        r, c = np.unravel_index(pick, labels.shape)
        rows_all.append(r)
        cols_all.append(c)
        y_all.append(np.full(len(pick), cls, dtype=np.int8))
    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    y = np.concatenate(y_all)
    block = (rows // block_px) * n_block_cols + (cols // block_px)
    return PointSet(rows=rows, cols=cols, y=y, block=block.astype(np.int64))


def spatial_split(ps: PointSet, *, test_size: float, seed: int, min_per_class: int = 30, max_tries: int = 200):
    """Split by block. Returns (train_idx, test_idx). Both sides must contain both classes."""
    blocks = np.unique(ps.block)
    if len(blocks) < 4:
        raise ValueError(f"Only {len(blocks)} spatial blocks contain points; reduce sampling.block_size_m")
    target = test_size * len(ps)
    for attempt in range(max_tries):
        rng = np.random.default_rng(seed + attempt)
        order = rng.permutation(blocks)
        counts = {b: int((ps.block == b).sum()) for b in order}
        chosen, total = [], 0
        for b in order:
            if total >= target:
                break
            chosen.append(b)
            total += counts[b]
        is_test = np.isin(ps.block, chosen)
        ok = True
        for side in (is_test, ~is_test):
            for cls in (0, 1):
                if int(((ps.y == cls) & side).sum()) < min_per_class:
                    ok = False
        if ok:
            return np.flatnonzero(~is_test), np.flatnonzero(is_test)
    raise ValueError(
        f"Could not find a block split with >= {min_per_class} points of each class on both sides "
        f"after {max_tries} tries. Use smaller sampling.block_size_m or a larger AOI."
    )
