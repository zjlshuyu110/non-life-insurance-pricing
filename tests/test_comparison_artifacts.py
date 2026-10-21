"""Protect the published experiment against mismatched splits or stale summaries."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "reports/model_comparison"


def test_reference_scores_match_original_study_and_all_models_share_targets():
    original = pd.read_csv(ROOT / "reports/metrics.csv").query("split == 'test'").set_index("model")
    result = pd.read_csv(REPORT / "metrics.csv").query("split == 'test'").set_index("model")
    cols = ["policies", "exposure", "actual_loss", "predicted_loss", "deviance_p1.9", "gini"]
    np.testing.assert_allclose(result.loc[original.index, cols], original[cols], rtol=1e-9)
    assert result.policies.nunique() == result.exposure.nunique() == result.actual_loss.nunique() == 1
    for name in ["MLP_seed0", "MLP_seed1", "MLP_seed2"]:
        assert np.isfinite(result.loc[name, cols].to_numpy(dtype=float)).all()


def test_selection_uses_validation_and_all_final_seeds_reuse_selected_budget():
    selection = pd.read_csv(REPORT / "validation_search.csv")
    choice = json.loads((REPORT / "selection.json").read_text())
    mlps = selection.query("family == 'MLP'")
    best = mlps.loc[mlps.validation_deviance.idxmin()]
    assert choice["mlp_widths"] == json.loads(best.widths)
    assert choice["mlp_epochs"] == best.best_epoch
    runs = pd.read_csv(REPORT / "training_runs.csv")
    assert set(runs.seed) == {0, 1, 2}
    assert (runs.epochs_run == choice["mlp_epochs"]).all()
    result = pd.read_csv(REPORT / "metrics.csv").query("split == 'test'").set_index("model")
    for row in pd.read_csv(REPORT / "paired_bootstrap.csv").itertuples():
        expected = result.loc["MLP_seed0", "deviance_p1.9"] - result.loc[row.comparator, "deviance_p1.9"]
        np.testing.assert_allclose(row.deviance_difference_mlp_minus_glm, expected, atol=1e-10)
