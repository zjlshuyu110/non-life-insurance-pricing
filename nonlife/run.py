"""CLI: python -m nonlife.run --sensitivity"""
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
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from .data import download_data, load_data, prepare_data, sha256
from .evaluation import calibration, component_metrics, metrics, risk_groups
from .models import fit_models

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--source", choices=["casdatasets", "openml"], default="casdatasets", help="Data download source; pinned authors' snapshot by default.")
    parser.add_argument("--sample", type=int, help="Optional random policy sample for a smoke run; omit for the full data.")
    parser.add_argument("--sensitivity", action="store_true", help="Refit at EUR 50k and without a loss cap, keeping policy splits fixed.")
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    if args.sample is not None and args.sample < 1000:
        parser.error("--sample must be at least 1000; severity models need positive claims.")
    if args.download_only:
        download_data(ROOT, source=args.source)
        return
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    raw, merge_audit = load_data(ROOT)
    if args.sample:
        raw = raw.sample(n=min(args.sample, len(raw)), random_state=config["seed"])
    frame, cleaning_audit = prepare_data(raw, config)
    train_ids, test_ids = train_test_split(frame.index, test_size=config["test_size"], random_state=config["seed"])
    train, test = frame.loc[train_ids], frame.loc[test_ids]
    with threadpool_limits(limits=4):
        fitted = fit_models(train, config)
        train_pred, test_pred = fitted.predict(train), fitted.predict(test)
    results = pd.concat([metrics(train, train_pred, "train", config["evaluation_powers"]),
                         metrics(test, test_pred, "test", config["evaluation_powers"])], ignore_index=True)
    results.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame([component_metrics(train, train_pred, "train"), component_metrics(test, test_pred, "test")]).to_csv(output / "component_metrics.csv", index=False)
    tables = {
        "driver_age": calibration(test, test_pred, pd.cut(test.DrivAge, [0, 25, 35, 45, 55, 65, 75, 120], right=False)),
        "vehicle_age": calibration(test, test_pred, pd.cut(test.VehAge, [0, 2, 5, 10, 15, 25, 200], right=False)),
        "region": calibration(test, test_pred, test.Region),
        "risk_decile": calibration(test, test_pred, risk_groups(test_pred.Tweedie, test.Exposure)),
    }
    for name, table in tables.items():
        table.to_csv(output / f"calibration_{name}.csv", index=False)
    sensitivity = []
    if args.sensitivity:
        # Compare models WITHIN each target definition; deviance across caps is not comparable.
        base = results.query("split == 'test'").copy()
        base["loss_cap"] = config["claim_amount_cap"]
        sensitivity.append(base)
        for cap in config["sensitivity_caps"]:
            print(f"Sensitivity: loss cap = {cap}", flush=True)
            scenario_config = {**config, "claim_amount_cap": cap}
            scenario, _ = prepare_data(raw, scenario_config)
            with threadpool_limits(limits=4):
                scenario_models = fit_models(scenario.loc[train_ids], scenario_config)
                scenario_predictions = scenario_models.predict(scenario.loc[test_ids])
            scored = metrics(scenario.loc[test_ids], scenario_predictions, "test", config["evaluation_powers"])
            scored["loss_cap"] = "Uncapped" if cap is None else cap
            sensitivity.append(scored)
    sensitivity_table = pd.concat(sensitivity, ignore_index=True) if sensitivity else pd.DataFrame(columns=[*results.columns, "loss_cap"])
    sensitivity_table.to_csv(output / "sensitivity.csv", index=False)
    metadata = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "packages": {name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "plotly"]},
        "sample": args.sample, "full_data_run": args.sample is None,
        "config": config, "merge_audit": merge_audit, "cleaning_audit": cleaning_audit,
        "train_policies": len(train), "test_policies": len(test),
        "test_positive_loss_policies": int((test.ClaimAmount > 0).sum()),
        "preprocessing_fit_on": "training policies only", "split_overlap": len(set(train_ids) & set(test_ids)),
        "source_manifest_sha256": sha256(ROOT / "data" / "manifest.json"),
        "data_sources": {name: item["source"] for name, item in json.loads((ROOT / "data" / "manifest.json").read_text())["datasets"].items()},
        "iterations": {"poisson": int(fitted.frequency.n_iter_), "gamma": int(fitted.severity.n_iter_), "tweedie": int(fitted.tweedie.n_iter_)},
    }
    tag = "full" if args.sample is None else f"sample-{args.sample}"
    for folder in ["models", "predictions"]:
        (ROOT / "artifacts" / folder).mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, ROOT / "artifacts" / "models" / f"{tag}.joblib")
    test[["Exposure", "ClaimAmount", "ClaimNb"]].join(test_pred.rename(columns={"Frequency": "PredictedFrequency", "Severity": "PredictedSeverity"})).to_csv(ROOT / "artifacts" / "predictions" / f"{tag}.csv.gz", compression="gzip")
    from .reporting import generate_reports
    generate_reports(output, frame, test, test_pred, results, tables, sensitivity_table, metadata)
    metadata["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(results.query("split == 'test'")[["model", "predicted_actual", "deviance_p1.9", "gini"]].to_string(index=False), flush=True)
    print(f"Finished in {metadata['elapsed_seconds']}s. Open {output / 'dashboard.html'}", flush=True)


if __name__ == "__main__":
    main()
