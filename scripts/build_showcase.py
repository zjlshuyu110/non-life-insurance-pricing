"""Build the offline/GitHub Pages visual lab from verified full-run artifacts."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nonlife.data import load_data, prepare_data, sha256
from nonlife.evaluation import risk_groups


def build() -> None:
    meta = json.loads((ROOT / "reports/run_metadata.json").read_text())
    if not meta["full_data_run"]:
        raise ValueError("The showcase requires a full-data run.")
    if sha256(ROOT / "data/manifest.json") != meta["source_manifest_sha256"]:
        raise ValueError("Data manifest changed since the model run.")
    raw, _ = load_data(ROOT)
    frame, _ = prepare_data(raw, meta["config"])
    saved = pd.read_csv(ROOT / "artifacts/predictions/full.csv.gz", index_col="IDpol")
    test = frame.loc[saved.index].copy()
    age_edges, age_labels = [0, 25, 35, 45, 55, 65, 75, 120], ["Under 25", "25–34", "35–44", "45–54", "55–64", "65–74", "75+"]
    vehicle_edges, vehicle_labels = [0, 2, 5, 10, 15, 25, 200], ["0–1 years", "2–4 years", "5–9 years", "10–14 years", "15–24 years", "25+ years"]
    regions = sorted(test.Region.unique())
    grouped = pd.DataFrame(index=test.index)
    grouped["age"] = pd.cut(test.DrivAge, age_edges, labels=False, right=False)
    grouped["vehicle"] = pd.cut(test.VehAge, vehicle_edges, labels=False, right=False)
    grouped["region"] = test.Region.map({v: i for i, v in enumerate(regions)})
    grouped["decile"] = risk_groups(saved.Tweedie, test.Exposure)
    grouped["policies"], grouped["exposure"], grouped["loss"] = 1, test.Exposure, test.ClaimAmount
    grouped["claims"] = test.ClaimNb
    power = 1.9
    for name, short in [("FrequencySeverity", "fs"), ("Tweedie", "tw"), ("Baseline", "base")]:
        mu = saved[name]
        y = test.PurePremium
        grouped[short] = test.Exposure * mu
        grouped[f"dev_{short}"] = 2 * test.Exposure * (y ** (2 - power) / ((1 - power) * (2 - power)) - y * mu ** (1 - power) / (1 - power) + mu ** (2 - power) / (2 - power))
    cube = grouped.groupby(["age", "vehicle", "region", "decile"], observed=True, sort=True).sum().reset_index()
    if cube.policies.sum() != len(test):
        raise ValueError("A policy fell outside the visualization groups.")
    metrics = pd.read_csv(ROOT / "reports/metrics.csv").query("split == 'test'").set_index("model")
    for name, short in [("FrequencySeverity", "fs"), ("Tweedie", "tw"), ("Baseline", "base")]:
        np.testing.assert_allclose(cube[short].sum(), metrics.loc[name, "predicted_loss"])
        np.testing.assert_allclose(cube[f"dev_{short}"].sum() / cube.exposure.sum(), metrics.loc[name, "deviance_p1.9"])
    model = joblib.load(ROOT / "artifacts/models/full.joblib")
    train = frame.drop(index=test.index)
    profile = {key: str(train[key].mode().iloc[0]) for key in ["VehBrand", "VehGas", "Region", "Area"]}
    profile.update({"VehAge": int(train.VehAge.median()), "VehPower": int(train.VehPower.mode().iloc[0]),
                    "Density": float(train.Density.median()), "DrivAge": 40, "BonusMalus": 50})
    ages, bonuses = list(range(18, 91, 2)), list(range(50, 151, 5))
    grid = pd.DataFrame([{**profile, "DrivAge": age, "BonusMalus": bonus} for age in ages for bonus in bonuses])
    with threadpool_limits(limits=4):
        predicted = model.predict(grid)
    surfaces = {name: predicted[col].to_numpy().reshape(len(ages), len(bonuses)).round(4).tolist()
                for name, col in [("fs", "FrequencySeverity"), ("tw", "Tweedie"), ("frequency", "Frequency"), ("severity", "Severity")]}
    sensitivity = pd.read_csv(ROOT / "reports/sensitivity.csv").to_dict(orient="records")
    for scenario in sensitivity:
        scenario["loss_cap"] = "Uncapped" if str(scenario["loss_cap"]) == "Uncapped" else str(int(float(scenario["loss_cap"])))
    payload = {"metadata": meta, "columns": list(cube.columns), "cube": cube.round(6).values.tolist(),
               "ageLabels": age_labels, "vehicleLabels": vehicle_labels, "regions": regions,
               "surface": {"ages": ages, "bonuses": bonuses, "profile": profile, "values": surfaces},
               "sensitivity": sensitivity, "globalMetrics": metrics.reset_index().to_dict(orient="records")}
    target = ROOT / "docs"
    assets = target / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "study-data.js").write_text("window.STUDY_DATA=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + ";\n", encoding="utf-8")
    (assets / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    for name in ["app.js", "engine.js", "style.css"]:
        shutil.copyfile(ROOT / "visualization" / name, assets / name)
    shutil.copyfile(ROOT / "visualization/index.html", target / "index.html")
    (target / ".nojekyll").write_text("")
    (ROOT / "reports/showcase_validation.json").write_text(json.dumps({
        "policies": int(cube.policies.sum()), "aggregate_cells": len(cube), "surface_profiles": len(grid),
        "weighted_deviance_matches_saved_metrics": True, "predicted_totals_match_saved_metrics": True,
        "individual_policy_records_exported": False, "source_manifest_sha256": meta["source_manifest_sha256"],
    }, indent=2), encoding="utf-8")
    print(f"Built {target / 'index.html'}: {len(cube):,} aggregate cells; {len(grid)} model profiles. All totals verified.")


if __name__ == "__main__":
    build()
