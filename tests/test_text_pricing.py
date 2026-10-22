"""Leakage, evidence and model-input boundaries; no model download needed in CI."""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from nonlife.data import sha256
from nonlife.text_pricing import input_gate, normalize, price_fields, process_document, validate_answer

ROOT = Path(__file__).parents[1]
CONFIG = json.loads((ROOT / "text_config.json").read_text())
CASES = json.loads((ROOT / "data/text_demo/applications.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case_id", ["claim", "late"])
def test_ineligible_document_never_reaches_extractor_or_model(case_id):
    class Forbidden:
        def extract(self, text):
            raise AssertionError("Extraction must not run for ineligible input")
    doc = next(c for c in CASES if c["id"] == case_id)
    result = process_document(doc, Forbidden(), None, CONFIG)
    assert result["status"] == "blocked"
    assert result["prediction"] is None and not result["fields"]


def test_timestamps_and_empty_or_oversized_text_fail_closed():
    doc = copy.deepcopy(CASES[0])
    for value in [None, "2026-01-01", "not-a-date"]:
        doc["available_at"] = value
        assert input_gate(doc)
    doc = copy.deepcopy(CASES[0])
    for text in ["", "x" * 6001, None]:
        doc["text"] = text
        assert input_gate(doc)


def test_unsupported_or_ambiguous_spans_are_not_silently_coerced():
    for bad in ["-42", "42.5", "42 or 24", "2026-01-01"]:
        assert normalize("DrivAge", bad) is None
    assert normalize("AnnualKm", "8,000") == 8000
    assert normalize("VehGas", "electricity") is None
    assert normalize("Usage", "diesel") is None
    assert normalize("BonusMalus", "15") is None


def test_evidence_is_verified_before_accepting_high_scoring_answer():
    raw = {"answer": "42", "score": .99, "start": 0, "end": 2}
    assert validate_answer("DrivAge", raw, "24 years", .25)["value"] is None
    assert validate_answer("DrivAge", raw, "42 years", .25)["value"] == 42
    raw["score"] = .1
    assert validate_answer("DrivAge", raw, "42 years", .25)["value"] is None


def test_missing_pricing_fields_never_receive_defaults():
    result = price_fields({"DrivAge": {"value": 42}}, CONFIG["reference_profile"], None)
    assert result["status"] == "review" and result["prediction"] is None


def test_only_the_model_allowlist_enters_pricing():
    class Spy:
        def predict(self, frame):
            assert set(frame) == {"DrivAge", "VehAge", "VehGas", "BonusMalus", "VehBrand", "VehPower", "Area", "Region", "Density"}
            assert frame.DrivAge.iloc[0] == 42
            return pd.DataFrame({"FrequencySeverity": [100.]})
    fields = {k: {"value": v} for k, v in CASES[0]["gold"].items()}
    fields["ClaimAmount"] = {"value": 100000}
    one = price_fields(fields, CONFIG["reference_profile"], Spy())
    fields["AnnualKm"]["value"] = 40000
    two = price_fields(fields, CONFIG["reference_profile"], Spy())
    assert one["prediction"] == two["prediction"]


def test_published_run_is_grounded_and_its_limits_are_retained():
    r = json.loads((ROOT / "reports/text_pricing/results.json").read_text(encoding="utf-8"))
    assert not r["metadata"]["claims_predictive_uplift_tested"]
    for path, key in [("data/text_demo/applications.json", "fixtures_sha256"), ("data/text_demo/manifest.json", "manifest_sha256"),
                      ("text_config.json", "config_sha256"), ("nonlife/text_pricing.py", "extractor_code_sha256")]:
        assert sha256(ROOT / path) == r["metadata"][key]
    for c in r["cases"]:
        if c["prediction"]:
            assert c["status"] == "priced"
            assert all(c["fields"][k]["status"] == "accepted" for k in ["DrivAge", "VehAge", "BonusMalus", "VehGas"])
        for f in c["fields"].values():
            if f["answer"]:
                assert c["text"][f["start"]:f["end"]] == f["answer"]
    missing = next(c for c in r["cases"] if c["id"] == "missing_extra")
    assert missing["fields"]["Usage"]["value"] is None
    assert missing["fields"]["Usage"]["answer"] == "diesel"  # Preserve the observed error.
    for doc in r["public"]:
        assert sha256(ROOT / doc["pdf_path"]) == doc["pdf_sha256"]
        assert sha256(ROOT / doc["text_path"]) == doc["text_sha256"]
        assert doc["status"] == "extraction_only"
        assert "prediction" not in doc
    assert sum(a["exact_match"] for c in r["public"] for a in c["answers"]) == r["metrics"]["public_exact_matches"]
