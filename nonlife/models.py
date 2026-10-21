"""Train-only preprocessing and the reference GLM specifications."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder, StandardScaler


@dataclass
class PricingModels:
    transformer: ColumnTransformer
    frequency: PoissonRegressor
    severity: GammaRegressor
    tweedie: TweedieRegressor
    baseline: float

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = self.transformer.transform(frame)
        frequency = self.frequency.predict(x)
        severity = self.severity.predict(x)
        return pd.DataFrame({
            "Baseline": self.baseline,
            "FrequencySeverity": frequency * severity,
            "Tweedie": self.tweedie.predict(x),
            "Frequency": frequency, "Severity": severity,
        }, index=frame.index)


def fit_models(train: pd.DataFrame, config: dict) -> PricingModels:
    transformer = ColumnTransformer([
        ("binned", KBinsDiscretizer(n_bins=10, strategy="quantile", encode="onehot",
                                   quantile_method="averaged_inverted_cdf", random_state=config["seed"]),
         ["VehAge", "DrivAge"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         ["VehBrand", "VehPower", "VehGas", "Region", "Area"]),
        ("bonus_malus", "passthrough", ["BonusMalus"]),
        ("density", make_pipeline(FunctionTransformer(np.log), StandardScaler()), ["Density"]),
    ])
    # Do not include test policies when estimating bins, scales or category levels.
    x = transformer.fit_transform(train)
    positive = train.ClaimAmount.to_numpy() > 0
    if positive.sum() < 2:
        raise ValueError("Need at least two positive-loss training policies.")
    options = {"solver": "newton-cholesky", "max_iter": 100, "tol": 1e-4}
    frequency = PoissonRegressor(alpha=config["frequency_alpha"], **options)
    severity = GammaRegressor(alpha=config["severity_alpha"], **options)
    tweedie = TweedieRegressor(alpha=config["tweedie_alpha"], power=config["tweedie_power"], link="log", **options)
    print(f"Fitting GLMs: {len(train):,} training policies, {x.shape[1]} features ...", flush=True)
    frequency.fit(x, train.Frequency, sample_weight=train.Exposure)
    severity.fit(x[positive], train.loc[positive, "AvgClaimAmount"], sample_weight=train.loc[positive, "ClaimNb"])
    tweedie.fit(x, train.PurePremium, sample_weight=train.Exposure)
    return PricingModels(transformer, frequency, severity, tweedie,
                         float(train.ClaimAmount.sum() / train.Exposure.sum()))
