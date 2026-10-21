import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from nonlife.data import join_tables, prepare_data
from nonlife.evaluation import calibration, metrics, ordered_lorenz, risk_groups
from nonlife.models import fit_models

CONFIG = json.loads((Path(__file__).parents[1] / "config.json").read_text())


def policy_frame(n=4):
    return pd.DataFrame({"Exposure": [.5] * n, "ClaimNb": [0] * n, "ClaimAmount": [0.] * n,
                         "Density": [100.] * n, "VehAge": [5] * n, "DrivAge": [40] * n,
                         "BonusMalus": [50] * n, "VehPower": [5] * n, "VehBrand": ["B1"] * n,
                         "VehGas": ["Regular"] * n, "Region": ["R1"] * n, "Area": ["A"] * n},
                        index=pd.Index(range(1, n + 1), name="IDpol"))


def test_join_sums_multiple_claims_without_losing_zero_claim_policies():
    freq = policy_frame(3).drop(columns="ClaimAmount").reset_index()
    sev = pd.DataFrame({"IDpol": [1, 1, 3, 999], "ClaimAmount": [100., 150., 80., 9999.]})
    result, audit = join_tables(freq, sev)
    assert list(result.ClaimAmount) == [250., 0., 80.]
    assert audit["orphan_severity_rows"] == 1
    assert audit["orphan_claim_amount"] == 9999
    assert len(result) == len(freq)


def test_duplicate_policy_ids_are_rejected():
    freq = policy_frame(3).drop(columns="ClaimAmount").reset_index()
    freq.loc[1, "IDpol"] = 1
    with pytest.raises(ValueError, match="one row per policy"):
        join_tables(freq, pd.DataFrame({"IDpol": [1], "ClaimAmount": [10.]}))


def test_cleaning_and_frequency_severity_identity():
    raw = policy_frame()
    raw["ClaimNb"] = [1, 2, 5, 0]
    raw["ClaimAmount"] = [0., 300., 500000., 0.]
    raw.loc[3, "Exposure"] = 2
    cleaned, audit = prepare_data(raw, CONFIG)
    assert cleaned.loc[1, "ClaimNb"] == 0
    assert cleaned.loc[3, "ClaimNb"] == 4
    assert cleaned.loc[3, "Exposure"] == 1
    assert cleaned.loc[3, "ClaimAmount"] == 200000
    assert audit["claim_amount_removed"] == 300000
    np.testing.assert_allclose(cleaned.Frequency * cleaned.AvgClaimAmount, cleaned.PurePremium)
    assert raw.loc[3, "ClaimAmount"] == 500000  # no mutation of the source


@pytest.mark.parametrize("column,value", [("Exposure", 0), ("Density", 0), ("ClaimNb", -1), ("ClaimNb", .5)])
def test_invalid_actuarial_inputs_fail_fast(column, value):
    raw = policy_frame()
    raw[column] = raw[column].astype(float)
    raw.loc[1, column] = value
    with pytest.raises(ValueError):
        prepare_data(raw, CONFIG)


def test_weighted_intercepts_recover_portfolio_totals_and_handle_new_region():
    raw = policy_frame(100)
    raw["Exposure"] = np.tile([.25, .5, 1., .75], 25)
    raw.loc[[1, 5, 8, 20, 40], "ClaimNb"] = [1, 2, 1, 1, 2]
    raw.loc[[1, 5, 8, 20, 40], "ClaimAmount"] = [100, 2000, 700, 600, 1600]
    train, _ = prepare_data(raw, CONFIG)
    with pytest.warns(UserWarning), threadpool_limits(limits=2):
        fitted = fit_models(train, CONFIG)
    predicted = fitted.predict(train)
    np.testing.assert_allclose(predicted.Frequency, train.ClaimNb.sum() / train.Exposure.sum(), rtol=1e-4)
    np.testing.assert_allclose(predicted.Severity, train.ClaimAmount.sum() / train.ClaimNb.sum(), rtol=1e-4)
    np.testing.assert_allclose(predicted.Tweedie, train.ClaimAmount.sum() / train.Exposure.sum(), rtol=1e-4)
    unseen = train.iloc[:1].copy()
    unseen["Region"] = "UNSEEN_TEST_REGION"
    assert np.isfinite(fitted.predict(unseen).to_numpy()).all()
    assert "UNSEEN_TEST_REGION" not in fitted.transformer.named_transformers_["categorical"].categories_[3]


def test_calibration_and_metrics_use_exposure_not_policy_averages():
    frame = pd.DataFrame({"Exposure": [.1, .9], "ClaimAmount": [10., 90.], "PurePremium": [100., 100.]})
    pred = pd.DataFrame({name: [100., 100.] for name in ["Baseline", "FrequencySeverity", "Tweedie"]})
    result = metrics(frame, pred, "test", [1.9])
    np.testing.assert_allclose(result.predicted_actual, 1)
    np.testing.assert_allclose(result["deviance_p1.9"], 0, atol=1e-12)
    table = calibration(frame, pred, pd.Series(["A", "A"]))
    assert table.loc[0, "actual_rate"] == 100
    assert table.loc[0, "Tweedie_rate"] == 100


def test_ties_have_no_artificial_ranking_and_remain_in_one_risk_group():
    loss, exposure = np.array([0., 500., 10.]), np.array([.1, 1., .4])
    assert ordered_lorenz(loss, exposure, np.ones(3))[2] == pytest.approx(0)
    groups = risk_groups(pd.Series([1., 1., 2.]), pd.Series(exposure))
    assert groups.iloc[0] == groups.iloc[1]
    assert ordered_lorenz(loss, exposure, loss / exposure)[2] > 0
