"""Decision Tree training with spatially grouped cross-validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text

from .indices import FEATURE_NAMES


def _new_tree(seed: int, **params) -> DecisionTreeClassifier:
    # class_weight='balanced' guards against residual imbalance when one class has few pixels.
    return DecisionTreeClassifier(class_weight="balanced", random_state=seed, **params)


def one_standard_error_choice(cv: pd.DataFrame, n_splits: int) -> dict:
    """Pick the simplest tree whose CV score is within one standard error of the best.

    Simplest = smallest max_depth, then largest min_samples_leaf. This is the standard
    guard against choosing an over-fitted tree because of a lucky fold.
    """
    best = cv.loc[cv["mean_test_score"].idxmax()]
    se = best["std_test_score"] / np.sqrt(n_splits)
    ok = cv[cv["mean_test_score"] >= best["mean_test_score"] - se].copy()
    ok["d"] = ok["param_max_depth"].astype(int)
    ok["l"] = ok["param_min_samples_leaf"].astype(int)
    row = ok.sort_values(["d", "l"], ascending=[True, False]).iloc[0]
    return {"max_depth": int(row["d"]), "min_samples_leaf": int(row["l"])}


def tune_decision_tree(X, y, groups, *, depths, leaves, cv_folds: int, scoring: str, seed: int):
    """Grid search with GroupKFold (whole spatial blocks held out per fold).

    Returns (fitted_model, cv_results_dataframe, chosen_params).
    """
    n_groups = len(np.unique(groups))
    if n_groups < cv_folds:
        raise ValueError(f"cv_folds={cv_folds} but the training set has only {n_groups} spatial blocks")
    search = GridSearchCV(
        _new_tree(seed),
        {"max_depth": list(depths), "min_samples_leaf": list(leaves)},
        cv=GroupKFold(n_splits=cv_folds),
        scoring=scoring,
        refit=False,
        return_train_score=True,
        n_jobs=1,
    )
    search.fit(X, y, groups=groups)
    cv = pd.DataFrame(search.cv_results_)
    chosen = one_standard_error_choice(cv, cv_folds)
    model = _new_tree(seed, **chosen).fit(X, y)
    return model, cv, chosen


def tree_rules(model: DecisionTreeClassifier) -> str:
    """Human-readable if/else rules (class 0 = land, class 1 = water)."""
    return export_text(model, feature_names=FEATURE_NAMES, show_weights=True)
