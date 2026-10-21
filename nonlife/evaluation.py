"""Exposure-weighted loss metrics, calibration and ranking diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_gamma_deviance, mean_poisson_deviance, mean_tweedie_deviance

MODELS = ["Baseline", "FrequencySeverity", "Tweedie"]
LABELS = {"Baseline": "Portfolio mean", "FrequencySeverity": "Poisson x Gamma", "Tweedie": "Tweedie GLM"}


def ordered_lorenz(loss: np.ndarray, exposure: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Aggregate tied scores before integrating so a constant model has Gini zero."""
    grouped = pd.DataFrame({"score": score, "loss": loss, "exposure": exposure}).groupby("score", sort=True)[["loss", "exposure"]].sum()
    x = np.r_[0.0, grouped.exposure.cumsum().to_numpy() / grouped.exposure.sum()]
    if grouped.loss.sum() <= 0:
        return x, np.full_like(x, np.nan), float("nan")
    y = np.r_[0.0, grouped.loss.cumsum().to_numpy() / grouped.loss.sum()]
    return x, y, float(1.0 - 2.0 * np.trapezoid(y, x))


def metrics(frame: pd.DataFrame, predictions: pd.DataFrame, split: str, powers: list[float],
            model_names: list[str] | None = None) -> pd.DataFrame:
    records = []
    for name in MODELS if model_names is None else model_names:
        pred = predictions[name].to_numpy()
        if not np.isfinite(pred).all() or (pred <= 0).any():
            raise ValueError(f"Invalid predictions from {name}")
        actual = float(frame.ClaimAmount.sum())
        expected = float(np.dot(frame.Exposure, pred))
        row = {"split": split, "model": name, "policies": len(frame),
               "exposure": float(frame.Exposure.sum()), "actual_loss": actual,
               "predicted_loss": expected, "predicted_actual": expected / actual,
               "weighted_mae": mean_absolute_error(frame.PurePremium, pred, sample_weight=frame.Exposure),
               "gini": ordered_lorenz(frame.ClaimAmount.to_numpy(), frame.Exposure.to_numpy(), pred)[2]}
        for power in powers:
            row[f"deviance_p{power}"] = mean_tweedie_deviance(frame.PurePremium, pred, power=power, sample_weight=frame.Exposure)
        records.append(row)
    return pd.DataFrame(records)


def component_metrics(frame: pd.DataFrame, predictions: pd.DataFrame, split: str) -> dict:
    positive = frame.ClaimAmount > 0
    return {"split": split,
            "poisson_deviance": mean_poisson_deviance(frame.Frequency, predictions.Frequency, sample_weight=frame.Exposure),
            "gamma_deviance": mean_gamma_deviance(frame.loc[positive, "AvgClaimAmount"], predictions.loc[positive, "Severity"], sample_weight=frame.loc[positive, "ClaimNb"])}


def calibration(frame: pd.DataFrame, predictions: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    work = pd.DataFrame({"group": groups, "exposure": frame.Exposure, "actual_loss": frame.ClaimAmount, "policies": 1})
    for name in MODELS:
        work[name] = frame.Exposure * predictions[name]
    table = work.groupby("group", observed=True, sort=True).sum()
    table["actual_rate"] = table.actual_loss / table.exposure
    for name in MODELS:
        table[f"{name}_rate"] = table[name] / table.exposure
    table["tweedie_actual_expected"] = table.actual_loss / table.Tweedie
    return table.reset_index()


def risk_groups(prediction: pd.Series, exposure: pd.Series, bins: int = 10) -> pd.Series:
    """Exposure quantiles; identical scores stay together (groups may be uneven)."""
    weights = pd.DataFrame({"score": prediction, "exposure": exposure}).groupby("score").exposure.sum()
    midpoints = (weights.cumsum() - 0.5 * weights) / weights.sum()
    labels = np.minimum(np.floor(midpoints * bins).astype(int) + 1, bins)
    return prediction.map(labels).rename("Risk decile")
