"""Accuracy assessment: metrics, confidence intervals, baselines.

Confusion-matrix convention everywhere: rows = reference (truth), columns = predicted,
class order [land, water]. 'User's accuracy' = precision, 'producer's accuracy' = recall.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def _div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else float("nan")


def metrics_from_matrix(m: np.ndarray) -> dict:
    """Metrics from a 2x2 matrix of counts OR proportions (rows=reference, cols=predicted)."""
    m = np.asarray(m, dtype=float)
    total = m.sum()
    tn, fp, fn, tp = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
    po = (tn + tp) / total if total > 0 else float("nan")
    pe = float((m.sum(1) * m.sum(0)).sum() / total**2) if total > 0 else float("nan")
    kappa = float((po - pe) / (1 - pe)) if total > 0 and pe < 1 else float("nan")
    prec, rec = _div(tp, tp + fp), _div(tp, tp + fn)
    return {
        "overall_accuracy": float(po),
        "kappa": kappa,
        "water_precision": prec,          # user's accuracy of water
        "water_recall": rec,              # producer's accuracy of water
        "water_f1": _div(2 * prec * rec, prec + rec) if np.isfinite(prec) and np.isfinite(rec) else float("nan"),
        "water_iou": _div(tp, tp + fp + fn),
        "land_precision": _div(tn, tn + fn),
        "land_recall": _div(tn, tn + fp),
    }


def binary_metrics(y_true, y_pred) -> dict:
    """Unweighted metrics (valid when points are a simple random sample, or a spatial test set)."""
    m = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out = metrics_from_matrix(m)
    out["confusion_matrix"] = m.tolist()
    out["n"] = int(m.sum())
    return out


def weighted_confusion(ref, pred, stratum, weights: dict) -> np.ndarray:
    """Area-weighted confusion proportions for a stratified sample (Olofsson et al. 2014 logic).

    p[i, j] = sum_h W_h * n_h(ref=i, pred=j) / n_h, with W_h the area proportion of stratum h.
    """
    ref, pred, stratum = np.asarray(ref), np.asarray(pred), np.asarray(stratum)
    w_total = float(sum(weights.values()))
    p = np.zeros((2, 2))
    for h, w in weights.items():
        idx = stratum == h
        n_h = int(idx.sum())
        if n_h == 0:
            raise ValueError(f"Stratum {h!r} has no usable labelled points; cannot compute area-weighted accuracy")
        for i in (0, 1):
            for j in (0, 1):
                p[i, j] += (w / w_total) * float(((ref == i) & (pred == j) & idx).sum()) / n_h
    return p


def weighted_metrics(ref, pred, stratum, weights: dict) -> dict:
    p = weighted_confusion(ref, pred, stratum, weights)
    out = metrics_from_matrix(p)
    out["confusion_proportions"] = p.tolist()
    out["n"] = int(len(ref))
    return out


_CI_KEYS = ("overall_accuracy", "kappa", "water_f1", "water_iou")


def _percentiles(samples: list[dict]) -> dict:
    ci = {}
    for k in _CI_KEYS:
        vals = np.array([s[k] for s in samples], dtype=float)
        vals = vals[np.isfinite(vals)]
        ci[k] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else [float("nan")] * 2
    return ci


def block_bootstrap_ci(y_true, y_pred, groups, *, n_boot: int = 500, seed: int = 0) -> dict:
    """95% CI by resampling whole spatial blocks (respects spatial dependence)."""
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    samples = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        samples.append(metrics_from_matrix(confusion_matrix(y_true[idx], y_pred[idx], labels=[0, 1])))
    return _percentiles(samples)


def stratified_bootstrap_ci(ref, pred, stratum, weights: dict, *, n_boot: int = 500, seed: int = 0) -> dict:
    """95% CI by resampling points within each stratum."""
    ref, pred, stratum = np.asarray(ref), np.asarray(pred), np.asarray(stratum)
    rng = np.random.default_rng(seed)
    members = {h: np.flatnonzero(stratum == h) for h in weights}
    samples = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(m, size=len(m), replace=True) for m in members.values() if len(m)])
        samples.append(metrics_from_matrix(weighted_confusion(ref[idx], pred[idx], stratum[idx], weights)))
    return _percentiles(samples)


def otsu_threshold(values: np.ndarray, nbins: int = 256, lo: float = -1.0, hi: float = 1.0) -> float:
    """Otsu's threshold on an index in [lo, hi] (the classic unsupervised baseline)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 10:
        raise ValueError("Too few finite values for Otsu thresholding")
    hist, edges = np.histogram(np.clip(v, lo, hi), bins=nbins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    p = hist / hist.sum()
    w0 = np.cumsum(p)
    w1 = 1 - w0
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_t * w0 - mu) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = -1
    # When the two modes are well separated the between-class variance is exactly flat across the empty gap,
    # and argmax would return the gap's left edge. Take the middle of that plateau instead (any value in the
    # gap separates the modes, the middle is the least arbitrary choice).
    best = between.max()
    plateau = np.flatnonzero(between >= best - 1e-9 * max(abs(best), 1.0))
    return float(centers[plateau[len(plateau) // 2]])
