"""Multi-output Random-Forest regressor wrapper.

Per-tree std is recovered manually so the active-learning loop can use it
as an uncertainty signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


@dataclass
class RegressorEval:
    mae_per_target: dict[str, float]
    r2_per_target: dict[str, float]


def train_regressor(
    X: np.ndarray, Y: np.ndarray,
    n_estimators: int = 300,
    max_depth: int | None = None,
    test_size: float = 0.2,
    seed: int = 7,
    n_jobs: int = -1,
):
    # Replace any NaN target rows with the column mean so RF can fit; the
    # caller usually filters numerical_invalid out before this.
    Y_clean = Y.copy()
    for j in range(Y.shape[1]):
        col = Y[:, j]
        mask = np.isnan(col)
        if mask.any() and (~mask).any():
            Y_clean[mask, j] = float(np.nanmean(col))
        elif mask.all():
            Y_clean[mask, j] = 0.0

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y_clean, test_size=test_size, random_state=seed
    )
    reg = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=n_jobs,
    )
    reg.fit(X_tr, Y_tr)
    return reg, (X_tr, X_te, Y_tr, Y_te)


def evaluate_regressor(reg, X_te: np.ndarray, Y_te: np.ndarray,
                       targets: list[str]) -> RegressorEval:
    Y_pred = reg.predict(X_te)
    mae = {}
    r2 = {}
    for j, name in enumerate(targets):
        mae[name] = float(mean_absolute_error(Y_te[:, j], Y_pred[:, j]))
        # r2_score returns -∞ for trivial targets; mask those.
        try:
            r2[name] = float(r2_score(Y_te[:, j], Y_pred[:, j]))
        except Exception:  # noqa: BLE001
            r2[name] = float("nan")
    return RegressorEval(mae_per_target=mae, r2_per_target=r2)


def regressor_per_tree_std(reg, X: np.ndarray) -> np.ndarray:
    """Return the per-target std across the forest's trees: shape (N, T)."""
    # Each estimator produces shape (N, T). Stack to (n_trees, N, T) → std along axis=0.
    preds = np.stack([tree.predict(X) for tree in reg.estimators_], axis=0)
    return preds.std(axis=0)
