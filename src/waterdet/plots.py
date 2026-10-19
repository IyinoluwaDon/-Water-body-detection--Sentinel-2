"""Figures for the report. Uses the non-interactive Agg backend so it works headless."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from sklearn.tree import plot_tree  # noqa: E402

from .indices import FEATURE_NAMES  # noqa: E402
from .io import Stack  # noqa: E402
from .predict import MAP_NODATA  # noqa: E402


def plot_confusion(cm, title: str, path: Path) -> None:
    cm = np.asarray(cm, dtype=float)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:.0f}" if cm.max() > 1.5 else f"{cm[i, j]:.3f}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["land", "water"]); ax.set_yticks([0, 1], ["land", "water"])
    ax.set_xlabel("predicted"); ax.set_ylabel("reference"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_cv_curve(cv, path: Path) -> None:
    """Train vs spatial-CV score by depth (best min_samples_leaf per depth): shows over-fitting."""
    d = cv.copy()
    d["depth"] = d["param_max_depth"].astype(int)
    g = d.groupby("depth")[["mean_train_score", "mean_test_score"]].max()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(g.index, g["mean_train_score"], "o-", label="train")
    ax.plot(g.index, g["mean_test_score"], "s-", label="spatial CV")
    ax.set_xlabel("max_depth"); ax.set_ylabel("score"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Tree depth vs score", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_tree_figure(model, path: Path) -> None:
    depth = model.get_depth()
    fig, ax = plt.subplots(figsize=(max(6, 2.2 * 2 ** min(depth, 4)), 2 + 1.6 * min(depth, 4)))
    plot_tree(model, feature_names=FEATURE_NAMES, class_names=["land", "water"], filled=True, max_depth=4,
              impurity=False, ax=ax, fontsize=7)
    fig.savefig(path, dpi=150); plt.close(fig)


def plot_importance(model, path: Path) -> None:
    imp = model.feature_importances_
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.barh(FEATURE_NAMES[::-1], imp[::-1])
    ax.set_xlabel("importance (Gini)"); ax.set_title("Feature importance", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _rgb(stack: Stack, stride: int) -> np.ndarray:
    rgb = np.stack([stack.bands[b][::stride, ::stride] for b in ("B4", "B3", "B2")], axis=-1)
    lo, hi = np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98)
    return np.clip(np.nan_to_num((rgb - lo) / max(hi - lo, 1e-6)), 0, 1)


def plot_map(map_arr: np.ndarray, stack: Stack, title: str, path: Path, max_px: int = 1800) -> None:
    """Side-by-side true colour and water map (nearest-neighbour thinning for big rasters)."""
    stride = max(1, int(np.ceil(max(map_arr.shape) / max_px)))
    m = map_arr[::stride, ::stride]
    cmap = ListedColormap(["#f2efe9", "#1f78b4", "#cccccc"])
    shown = np.where(m == MAP_NODATA, 2, m)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(_rgb(stack, stride)); axes[0].set_title("Sentinel-2 true colour composite")
    axes[1].imshow(shown, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    axes[1].set_title(title)
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_change(change: np.ndarray, title: str, path: Path, max_px: int = 1800) -> None:
    stride = max(1, int(np.ceil(max(change.shape) / max_px)))
    c = change[::stride, ::stride]
    shown = np.where(c == MAP_NODATA, 4, c)
    cmap = ListedColormap(["#f2efe9", "#1f78b4", "#2ca25f", "#e31a1c", "#cccccc"])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(shown, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    ax.set_title(title); ax.set_axis_off()
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in ("#f2efe9", "#1f78b4", "#2ca25f", "#e31a1c", "#cccccc")]
    ax.legend(handles, ["stable land", "stable water", "gain", "loss", "no data"], loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
