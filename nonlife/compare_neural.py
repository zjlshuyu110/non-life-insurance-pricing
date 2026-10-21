"""Run the full GLM vs ordinary MLP experiment: python -m nonlife.compare_neural."""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_tweedie_deviance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import load_data, prepare_data, sha256
from .evaluation import metrics, ordered_lorenz, risk_groups
from .models import fit_models, make_transformer
from .neural import fit_mlp, predict

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/model_comparison"
LABELS = {"Baseline": "Portfolio mean", "FrequencySeverity": "Poisson x Gamma",
          "Tweedie": "Tweedie GLM (reference)", "TunedTweedie": "Tweedie GLM (validation tuned)",
          **{f"MLP_seed{s}": f"MLP (seed {s})" for s in [0, 1, 2]}}


def dense(x):
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)


def unit_deviance(y, mu, power=1.9):
    return 2 * (y ** (2 - power) / ((1 - power) * (2 - power))
                - y * mu ** (1 - power) / (1 - power) + mu ** (2 - power) / (2 - power))


def paired_bootstrap(y, exposure, predictions, config):
    """IID policy resampling with fixed fitted models; not training uncertainty."""
    comparisons = ["FrequencySeverity", "Tweedie", "TunedTweedie"]
    primary = unit_deviance(y, predictions.MLP_seed0.to_numpy())
    differences = np.column_stack([primary - unit_deviance(y, predictions[m].to_numpy()) for m in comparisons])
    numerator = exposure[:, None] * differences
    observed = numerator.sum(axis=0) / exposure.sum()
    rng = np.random.default_rng(config["bootstrap_seed"])
    samples = np.empty((config["bootstrap_replicates"], len(comparisons)))
    for i in range(len(samples)):
        ids = rng.integers(0, len(y), size=len(y))
        samples[i] = numerator[ids].sum(axis=0) / exposure[ids].sum()
    intervals = np.quantile(samples, [.025, .975], axis=0)
    return pd.DataFrame({"comparator": comparisons, "mlp": "MLP_seed0",
                         "deviance_difference_mlp_minus_glm": observed,
                         "ci95_low": intervals[0], "ci95_high": intervals[1],
                         "replicates": len(samples)})


def run():
    started = time.perf_counter()
    config = json.loads((ROOT / "config.json").read_text())
    experiment = json.loads((ROOT / "comparison_config.json").read_text())
    source = json.loads((ROOT / "reports/run_metadata.json").read_text())
    if source["source_manifest_sha256"] != sha256(ROOT / "data/manifest.json") or source["config"] != config:
        raise ValueError("The comparison requires the original data manifest and target configuration.")
    if not source["full_data_run"]:
        raise ValueError("Run the full GLM study first.")
    if experiment["tweedie_power"] != config["tweedie_power"]:
        raise ValueError("Use the same Tweedie power for both model families.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw, _ = load_data(ROOT)
    frame, _ = prepare_data(raw, config)
    train_ids, test_ids = train_test_split(frame.index, test_size=config["test_size"], random_state=config["seed"])
    train, test = frame.loc[train_ids], frame.loc[test_ids]
    stored = pd.read_csv(ROOT / "artifacts/predictions/full.csv.gz", index_col="IDpol")
    pd.testing.assert_index_equal(test.index, stored.index)
    np.testing.assert_allclose(test.ClaimAmount, stored.ClaimAmount)
    inner_ids, validation_ids = train_test_split(train.index,
        test_size=experiment["validation_fraction_of_training"], random_state=experiment["validation_split_seed"])
    assert set(inner_ids).isdisjoint(validation_ids) and set(train_ids).isdisjoint(test_ids)
    inner, validation = train.loc[inner_ids], train.loc[validation_ids]
    transformer = make_transformer(config["seed"])
    ix, vx = transformer.fit_transform(inner), None
    vx = transformer.transform(validation)
    selection = []
    print(f"Full comparison: fit={len(inner):,}, validation={len(validation):,}, untouched test={len(test):,}", flush=True)
    with threadpool_limits(limits=experiment["cpu_threads"]):
        for alpha in experiment["glm_alphas"]:
            glm = TweedieRegressor(power=config["tweedie_power"], alpha=alpha, link="log",
                                  solver="newton-cholesky", max_iter=100, tol=1e-4)
            glm.fit(ix, inner.PurePremium, sample_weight=inner.Exposure)
            score = mean_tweedie_deviance(validation.PurePremium, glm.predict(vx),
                                         sample_weight=validation.Exposure, power=1.9)
            selection.append({"family": "TweedieGLM", "alpha": alpha, "validation_deviance": score})
            print(f"GLM alpha={alpha}: validation deviance={score:.6f}", flush=True)
    best_glm = min(selection, key=lambda r: r["validation_deviance"])
    scaler = StandardScaler(with_mean=False)
    nix = dense(scaler.fit_transform(ix))
    nvx = dense(scaler.transform(vx))
    histories = []
    for widths in experiment["mlp_candidates"]:
        tag = "-".join(map(str, widths))
        _, _, history, info = fit_mlp(nix, inner.PurePremium.to_numpy(), inner.Exposure.to_numpy(), widths,
            experiment["selection_seed"], experiment,
            validation=(nvx, validation.PurePremium.to_numpy(), validation.Exposure.to_numpy()))
        selection.append({"family": "MLP", "architecture": tag, **info,
                          "validation_deviance": info["best_validation_deviance"]})
        histories.extend({"stage": "selection", "run": tag, **row} for row in history)
        pd.DataFrame(histories).to_csv(OUTPUT / "learning_curves.csv", index=False)
        pd.DataFrame(selection).to_csv(OUTPUT / "validation_search.csv", index=False)
    chosen = min((r for r in selection if r["family"] == "MLP"), key=lambda r: r["validation_deviance"])
    # Selection is complete before producing any final-test neural predictions.
    decision = {"glm_alpha": best_glm["alpha"], "mlp_widths": chosen["widths"],
                "mlp_epochs": chosen["best_epoch"], "primary_mlp_seed": 0,
                "chosen_at_utc": datetime.now(timezone.utc).isoformat()}
    (OUTPUT / "selection.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print("Validation-only selection:", decision, flush=True)
    refit_start = time.perf_counter()
    with threadpool_limits(limits=experiment["cpu_threads"]):
        fitted = fit_models(train, config)
        train_predictions, test_predictions = fitted.predict(train), fitted.predict(test)
        for name in ["Baseline", "FrequencySeverity", "Tweedie"]:
            np.testing.assert_allclose(test_predictions[name], stored[name], rtol=1e-9)
        x, xtest = fitted.transformer.transform(train), fitted.transformer.transform(test)
        tuned = TweedieRegressor(power=1.9, alpha=best_glm["alpha"], link="log",
                                solver="newton-cholesky", max_iter=100, tol=1e-4)
        tuned.fit(x, train.PurePremium, sample_weight=train.Exposure)
        train_predictions["TunedTweedie"], test_predictions["TunedTweedie"] = tuned.predict(x), tuned.predict(xtest)
    glm_refit_seconds = time.perf_counter() - refit_start
    final_scaler = StandardScaler(with_mean=False)
    nx, ntx = dense(final_scaler.fit_transform(x)), dense(final_scaler.transform(xtest))
    model_dir = ROOT / "artifacts/models/neural"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"transformer": fitted.transformer, "scaler": final_scaler, "tuned_glm": tuned}, model_dir / "preprocessing.joblib")
    final_info = []
    for seed in experiment["final_seeds"]:
        model, scale, history, info = fit_mlp(nx, train.PurePremium.to_numpy(), train.Exposure.to_numpy(),
            chosen["widths"], seed, experiment, epochs=chosen["best_epoch"])
        name = f"MLP_seed{seed}"
        train_predictions[name], test_predictions[name] = predict(model, nx, scale), predict(model, ntx, scale)
        final_info.append(info)
        histories.extend({"stage": "final_refit", "run": name, **row} for row in history)
        torch.save({"state_dict": model.state_dict(), "widths": chosen["widths"], "inputs": nx.shape[1],
                    "scale": scale, "seed": seed, "epochs": chosen["best_epoch"]}, model_dir / f"{name}.pt")
    names = list(LABELS)
    scored = pd.concat([metrics(train, train_predictions, "train", [1.5, 1.9], names),
                        metrics(test, test_predictions, "test", [1.5, 1.9], names)], ignore_index=True)
    scored.to_csv(OUTPUT / "metrics.csv", index=False)
    pd.DataFrame(histories).to_csv(OUTPUT / "learning_curves.csv", index=False)
    pd.DataFrame(final_info).to_csv(OUTPUT / "training_runs.csv", index=False)
    test[["Exposure", "ClaimAmount"]].join(test_predictions[names]).to_csv(
        ROOT / "artifacts/predictions/neural_comparison.csv.gz", compression="gzip")
    tables = {}
    groupings = {"age": pd.cut(test.DrivAge, [0, 25, 35, 45, 55, 65, 75, 120], right=False),
                "risk": risk_groups(test_predictions.Tweedie, test.Exposure)}
    for key, groups in groupings.items():
        work = pd.DataFrame({"group": groups, "exposure": test.Exposure, "loss": test.ClaimAmount, "policies": 1})
        for name in names:
            work[name] = test.Exposure * test_predictions[name]
        table = work.groupby("group", observed=True).sum()
        table["observed_rate"] = table.loss / table.exposure
        for name in names:
            table[name + "_rate"] = table[name] / table.exposure
        tables[key] = table.reset_index()
        tables[key].to_csv(OUTPUT / f"calibration_{key}.csv", index=False)
    lorenz = []
    for name in names:
        lx, ly, _ = ordered_lorenz(test.ClaimAmount.to_numpy(), test.Exposure.to_numpy(), test_predictions[name].to_numpy())
        for fraction, loss in zip(np.linspace(0, 1, 201), np.interp(np.linspace(0, 1, 201), lx, ly)):
            lorenz.append({"model": name, "exposure_share": fraction, "loss_share": loss})
    pd.DataFrame(lorenz).to_csv(OUTPUT / "lorenz.csv", index=False)
    print("Computing paired test-policy bootstrap intervals ...", flush=True)
    intervals = paired_bootstrap(test.PurePremium.to_numpy(), test.Exposure.to_numpy(), test_predictions, experiment)
    intervals.to_csv(OUTPUT / "paired_bootstrap.csv", index=False)
    metadata = {"run_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "torch"]},
        "experiment": experiment, "target_config": config, "selection": decision,
        "train_policies": len(train), "inner_fit_policies": len(inner), "validation_policies": len(validation),
        "test_policies": len(test), "feature_count": nx.shape[1], "full_data_run": True,
        "same_test_ids_and_targets_as_original": True, "original_glm_predictions_reproduced": True,
        "split_overlap": 0, "preprocessing_fit_on": "inner fit during selection, full training during refit",
        "post_hoc_calibration": False, "test_used_for_selection": False, "device": "CPU",
        "glm_reference_and_tuned_refit_seconds": round(glm_refit_seconds, 3),
        "final_mlp_runs": final_info, "source_manifest_sha256": sha256(ROOT / "data/manifest.json"),
        "comparison_config_sha256": sha256(ROOT / "comparison_config.json"),
        "bootstrap_scope": "paired IID policy resampling; fitted models fixed; excludes training uncertainty and entity clustering",
        "elapsed_seconds": round(time.perf_counter() - started, 2)}
    (OUTPUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(scored.query("split == 'test'")[["model", "deviance_p1.9", "predicted_actual", "gini"]].to_string(index=False), flush=True)
    from .comparison_reporting import build_report
    build_report(OUTPUT)
    print(f"Finished in {metadata['elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        from .comparison_reporting import build_report
        build_report(OUTPUT)
    else:
        run()
