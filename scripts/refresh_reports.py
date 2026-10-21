"""Regenerate reports from saved full-run predictions without refitting models."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nonlife.data import load_data, prepare_data, sha256
from nonlife.reporting import generate_reports

output = ROOT / "reports"
metadata = json.loads((output / "run_metadata.json").read_text())
if not metadata["full_data_run"]:
    raise ValueError("This helper is for the saved full run only.")
if sha256(ROOT / "data" / "manifest.json") != metadata["source_manifest_sha256"]:
    raise ValueError("Source manifest has changed since the model run.")
raw, _ = load_data(ROOT)
frame, _ = prepare_data(raw, metadata["config"])
saved = pd.read_csv(ROOT / "artifacts" / "predictions" / "full.csv.gz", index_col="IDpol")
test = frame.loc[saved.index]
predictions = saved.rename(columns={"PredictedFrequency": "Frequency", "PredictedSeverity": "Severity"})
results = pd.read_csv(output / "metrics.csv")
tables = {name: pd.read_csv(output / f"calibration_{name}.csv") for name in ["driver_age", "vehicle_age", "region", "risk_decile"]}
sensitivity = pd.read_csv(output / "sensitivity.csv")
metadata["data_sources"] = {name: item["source"] for name, item in json.loads((ROOT / "data" / "manifest.json").read_text())["datasets"].items()}
generate_reports(output, frame, test, predictions, results, tables, sensitivity, metadata)
(output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print("Refreshed reports from the saved full-data predictions.")
