"""Evidence-grounded neural extraction -> existing GLM, with explicit review gates.

Text examples are synthetic workflow fixtures. No loss labels train the extractor,
and no new pricing performance or information uplift is claimed by this module.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
from datetime import datetime
import json
from pathlib import Path
import re
import time

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from nonlife.data import sha256

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = {
    "DrivAge": "How old is the main driver?",
    "VehAge": "How old is the car?",
    "BonusMalus": "What is the bonus-malus coefficient?",
    "VehGas": "What fuel does the car use?",
    "AnnualKm": "How many kilometres does the driver expect to drive per year?",
    "Usage": "What is the car used for?",
}
REQUIRED = ("DrivAge", "VehAge", "BonusMalus", "VehGas")
FIXED = ("VehBrand", "VehPower", "Region", "Area", "Density")
RANGES = {"DrivAge": (18, 100), "VehAge": (0, 40), "BonusMalus": (50, 150), "AnnualKm": (1, 200000)}
USAGES = {"commuting", "food delivery", "shopping", "business", "private use", "ridesharing", "taxi", "courier", "leisure"}


def input_gate(document: dict, max_chars: int = 6000) -> list[str]:
    """Document metadata comes from ingestion, never from the extraction model."""
    reasons = []
    if document.get("document_type") != "application":
        reasons.append("Only pre-quote applications enter the pricing path.")
    if document.get("language") != "en":
        reasons.append("This extraction model supports English only.")
    try:
        available = datetime.fromisoformat(document["available_at"])
        quote = datetime.fromisoformat(document["quote_at"])
        if available.tzinfo is None or quote.tzinfo is None or available > quote:
            reasons.append("Document must have a timezone-aware timestamp no later than the quote.")
    except (KeyError, ValueError, TypeError):
        reasons.append("Missing or invalid document/quote timestamp.")
    text = document.get("text", "")
    if not isinstance(text, str) or not text.strip() or len(text) > max_chars:
        reasons.append("Empty or oversized text; manual review is required.")
    return reasons


def normalize(field: str, answer: str):
    """Convert a neural-selected span to a typed value, never search for a new answer."""
    answer = answer.strip()
    if field in RANGES:
        # Do not turn a range, negative number, date or decimal into an integer.
        pattern = r"(\d{1,3}(?:,\d{3})+|\d+)"
        suffix = r"(?:\s*(?:years?(?:\s+old)?|kilometres?(?:\s+per\s+year)?|km(?:\s*/\s*year)?))?"
        match = re.fullmatch(pattern + suffix, answer, flags=re.I)
        if not match:
            return None
        value = int(match[1].replace(",", ""))
        lo, hi = RANGES[field]
        return value if lo <= value <= hi else None
    if field == "VehGas":
        return {"diesel": "Diesel", "petrol": "Regular", "gasoline": "Regular", "regular": "Regular"}.get(answer.lower())
    if field == "Usage":
        # Controlled demo taxonomy. Unknown phrases require review.
        return answer.lower() if answer.lower() in USAGES else None
    return None


def validate_answer(field: str, raw: dict, text: str, threshold: float) -> dict:
    answer, start, end = raw.get("answer", ""), raw.get("start", 0), raw.get("end", 0)
    score = float(raw.get("score", 0))
    grounded = isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text) and text[start:end] == answer
    value = normalize(field, answer) if grounded else None
    status = "accepted"
    if not answer:
        status = "no_answer"
    elif not grounded:
        status = "invalid_evidence"
    elif score < threshold:
        status = "low_score"
    elif value is None:
        status = "invalid_value"
    return {"value": value if status == "accepted" else None, "status": status,
            "answer": answer, "start": start, "end": end, "score": score,
            "used_by_glm": field in REQUIRED}


class LocalExtractor:
    def __init__(self, config: dict, model_dir: Path):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
        torch.set_num_threads(config["threads"])
        # Public downloads are safetensors only. No remote Python or pickle execution.
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        model = AutoModelForQuestionAnswering.from_pretrained(str(model_dir), local_files_only=True,
                                                             use_safetensors=True, trust_remote_code=False)
        self.qa = pipeline("question-answering", model=model, tokenizer=tokenizer, device=-1)
        self.threshold = config["score_threshold"]

    def ask(self, text: str, question: str) -> dict:
        raw = self.qa(question=question, context=text, handle_impossible_answer=True,
                      max_answer_len=24, max_seq_len=384, doc_stride=128)
        return {k: raw[k] for k in ("answer", "score", "start", "end")}

    def extract(self, text: str) -> dict:
        return {key: validate_answer(key, self.ask(text, question), text, self.threshold)
                for key, question in QUESTIONS.items()}


def price_fields(fields: dict, fixed_profile: dict, models) -> dict:
    missing = [key for key in REQUIRED if fields.get(key, {}).get("value") is None]
    if missing:
        return {"status": "review", "reasons": ["Missing/rejected pricing fields: " + ", ".join(missing)], "prediction": None}
    if set(fixed_profile) != set(FIXED):
        raise ValueError("The supplied structured profile must contain exactly the five non-text model fields.")
    # Only allowlisted fields reach the GLM. Mileage, usage, text and claims cannot enter it.
    profile = {**fixed_profile, **{key: fields[key]["value"] for key in REQUIRED}}
    with threadpool_limits(limits=4):
        predictions = models.predict(pd.DataFrame([profile])).iloc[0].to_dict()
    return {"status": "priced", "reasons": [], "profile": profile,
            "prediction": {k: float(v) for k, v in predictions.items()}}


def process_document(document: dict, extractor, models, config: dict) -> dict:
    reasons = input_gate(document, config["max_chars"])
    if reasons:
        return {"status": "blocked", "reasons": reasons, "fields": {}, "prediction": None}
    fields = extractor.extract(document["text"])
    # This catches explicit conflicting age statements in these examples. It is
    # not a general contradiction detector and cannot replace document review.
    age_mentions = re.findall(r"(?:main driver (?:is|aged)|I am)\s+(\d+)\s*(?:years? old)?", document["text"], re.I)
    if len(set(age_mentions)) > 1:
        return {"status": "review", "reasons": ["Conflicting driver ages; confirm the application."],
                "fields": fields, "prediction": None}
    result = price_fields(fields, config["reference_profile"], models)
    return {**result, "fields": fields}


def benchmark(root: Path = ROOT) -> dict:
    config = json.loads((root / "text_config.json").read_text())
    fixtures_path = root / "data/text_demo/applications.json"
    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    manifest_path = root / "data/text_demo/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_dir = root / ".cache/text-model"
    for entry in manifest["model_files"]:
        if sha256(model_dir / entry["file"]) != entry["sha256"]:
            raise ValueError("Extraction model cache checksum mismatch: " + entry["file"])
    # This is the project's own trusted model artifact, never an arbitrary user upload.
    model_path = root / "artifacts/models/full.joblib"
    if not model_path.exists():
        raise FileNotFoundError("First run python -m nonlife.run to fit the base GLMs.")
    models = joblib.load(model_path)
    extractor = LocalExtractor(config, model_dir)
    started = time.perf_counter()
    cases = []
    counts = {"fields": 0, "correct": 0, "accepted": 0, "accepted_correct": 0,
              "present": 0, "present_correct": 0, "absent": 0, "absent_abstained": 0}
    for doc in fixtures:
        result = process_document(doc, extractor, models, config)
        comparison = {}
        if doc["evaluate_fields"]:
            for key, gold in doc["gold"].items():
                prediction = result["fields"][key]["value"]
                correct = prediction == gold
                comparison[key] = correct
                counts["fields"] += 1
                counts["correct"] += correct
                counts["accepted"] += prediction is not None
                counts["accepted_correct"] += prediction is not None and correct
                counts["present"] += gold is not None
                counts["present_correct"] += gold is not None and correct
                counts["absent"] += gold is None
                counts["absent_abstained"] += gold is None and prediction is None
        oracle = None
        if doc["evaluate_fields"] and all(doc["gold"].get(k) is not None for k in REQUIRED):
            oracle = price_fields({k: {"value": v} for k, v in doc["gold"].items()}, config["reference_profile"], models)["prediction"]
        cases.append({**doc, **result, "field_correct": comparison, "oracle_prediction": oracle})
        print(doc["id"], result["status"], flush=True)
    public = []
    for entry in manifest["public_documents"]:
        text_path = root / entry["text_path"]
        if sha256(text_path) != entry["text_sha256"] or sha256(root / entry["pdf_path"]) != entry["pdf_sha256"]:
            raise ValueError("Public document checksum mismatch.")
        context = text_path.read_text(encoding="utf-8")
        answers = []
        for q in entry["questions"]:
            raw = extractor.ask(context, q["question"])
            answers.append({**q, **raw, "exact_match": raw["answer"].strip().lower() == q["expected"].lower()})
        public.append({**entry, "text": context, "answers": answers,
                       "status": "extraction_only", "pricing_reason": "Synthetic issued contract; incompatible/missing French GLM fields. No pricing join."})
    differences = [abs(c["prediction"]["FrequencySeverity"] / c["oracle_prediction"]["FrequencySeverity"] - 1)
                   for c in cases if c["prediction"] and c["oracle_prediction"]]
    report = {
        "kind": "synthetic_workflow_demonstration", "config": config,
        "metadata": {"model_sha256": sha256(model_path), "fixtures_sha256": sha256(fixtures_path),
                     "config_sha256": sha256(root / "text_config.json"),
                     "extractor_code_sha256": sha256(root / "nonlife/text_pricing.py"),
                     "versions": {p: version(p) for p in ["torch", "transformers", "tokenizers", "safetensors", "pypdf", "scikit-learn"]},
                     "manifest_sha256": sha256(manifest_path), "runtime_seconds": time.perf_counter() - started,
                     "pricing_retrained": False, "claims_predictive_uplift_tested": False,
                     "evaluation": "Small authored development fixtures, inspected during implementation; not an independent NLP test set."},
        "metrics": {**counts, "applications": len(cases), "priced": sum(c["status"] == "priced" for c in cases),
                    "review": sum(c["status"] == "review" for c in cases), "blocked": sum(c["status"] == "blocked" for c in cases),
                    "maximum_priced_relative_error": max(differences, default=None),
                    "public_questions": sum(len(c["answers"]) for c in public),
                    "public_exact_matches": sum(a["exact_match"] for c in public for a in c["answers"])},
        "cases": cases, "public": public,
    }
    out = root / "reports/text_pricing"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["metrics"], indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="An application JSON with text, document_type, language, available_at and quote_at")
    parser.add_argument("--output", type=Path, help="Write the result of --input to this JSON file")
    args = parser.parse_args()
    if args.input:
        config = json.loads((ROOT / "text_config.json").read_text())
        doc = json.loads(args.input.read_text(encoding="utf-8"))
        reasons = input_gate(doc, config["max_chars"])
        if reasons:
            result = {"status": "blocked", "reasons": reasons, "prediction": None}
        else:
            result = process_document(doc, LocalExtractor(config, ROOT / ".cache/text-model"),
                                      joblib.load(ROOT / "artifacts/models/full.joblib"), config)
        serialized = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(serialized + "\n", encoding="utf-8", newline="\n")
        print(serialized)
    else:
        benchmark()


if __name__ == "__main__":
    main()
