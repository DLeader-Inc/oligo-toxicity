"""
XGBoost baseline trainer with multi-seed 5-fold cross-validation.

This is the CPU-only baseline used in our PoC. The neural net trainer in
``src/models/neural_net.py`` should beat this on a GPU. Both follow the
same interface so they can be swapped via the CLI.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


@dataclass
class CVResult:
    """Holds out-of-fold predictions and metrics from a single seed."""

    seed: int
    spearman: float
    pearson: float
    r2: float
    rmse: float
    preds: np.ndarray   # (n_samples,) out-of-fold predictions
    folds: np.ndarray   # (n_samples,) which fold each sample was held out in


def train_xgboost_cv(X: pd.DataFrame, y: np.ndarray,
                      seeds: Tuple[int, ...] = (42, 123, 456, 789, 2024),
                      n_estimators: int = 200,
                      max_depth: int = 6,
                      learning_rate: float = 0.1) -> List[CVResult]:
    """Train XGBoost with 5-fold CV across multiple random seeds.

    Each seed produces an independent (CV split, model init). Results from
    different seeds quantify the model's stability — std should be small
    (≪ 0.05) for a reliable model.
    """
    results: List[CVResult] = []
    for seed in seeds:
        model = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=seed,
            verbosity=0, n_jobs=-1,
        )
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        preds = np.zeros(len(y))
        folds = np.zeros(len(y), dtype=int)
        for fold_idx, (tri, tei) in enumerate(kf.split(X)):
            model.fit(X.iloc[tri], y[tri])
            preds[tei] = model.predict(X.iloc[tei])
            folds[tei] = fold_idx + 1
        sp, _ = stats.spearmanr(y, preds)
        pe, _ = stats.pearsonr(y, preds)
        r2 = r2_score(y, preds)
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        results.append(CVResult(seed, float(sp), float(pe), float(r2),
                                 rmse, preds, folds))
    return results


def feature_importance_multi_seed(X: pd.DataFrame, y: np.ndarray,
                                    seeds: Tuple[int, ...] = (42, 123, 456, 789, 2024)
                                    ) -> pd.DataFrame:
    """Refit on full data per seed; return per-feature mean+std importance.

    Output columns: ``feature``, ``importance_avg``, ``importance_std``,
    ``importance_pct``, plus one ``importance_seed{S}`` column per seed.
    """
    importances = {}
    for seed in seeds:
        model = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=seed, verbosity=0, n_jobs=-1,
        )
        model.fit(X, y)
        importances[seed] = pd.Series(model.feature_importances_, index=X.columns)

    df = pd.DataFrame(importances)
    avg = df.mean(axis=1).sort_values(ascending=False)
    std = df.std(axis=1)
    out = pd.DataFrame({
        "feature": avg.index,
        "importance_avg": avg.values.round(6),
        "importance_std": std[avg.index].values.round(6),
        "importance_pct": (avg.values / max(avg.sum(), 1e-9) * 100).round(2),
    })
    for seed in seeds:
        out[f"importance_seed{seed}"] = importances[seed][avg.index].values.round(6)
    out["rank"] = range(1, len(out) + 1)
    return out


def aggregate_metrics(results: List[CVResult]) -> dict:
    """Compute mean and std across seeds for each metric."""
    sps = [r.spearman for r in results]
    pes = [r.pearson for r in results]
    r2s = [r.r2 for r in results]
    rmses = [r.rmse for r in results]
    return {
        "spearman_mean": round(float(np.mean(sps)), 4),
        "spearman_std": round(float(np.std(sps)), 4),
        "pearson_mean": round(float(np.mean(pes)), 4),
        "r2_mean": round(float(np.mean(r2s)), 4),
        "rmse_mean": round(float(np.mean(rmses)), 4),
        "n_seeds": len(results),
    }
