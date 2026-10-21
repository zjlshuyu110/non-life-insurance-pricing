"""Download and audit the original OpenML frequency and severity tables."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

DATASETS = {"freMTPL2freq": 41214, "freMTPL2sev": 41215}
CAS_COMMIT = "c49cbbb37235fc49616cac8ccac32e1491cdc619"
CAS_SHA256 = {
    "freMTPL2freq": "6be2ecdf87fba43a968adab6a1392f0ce9391a560cbb795036219c95d48da7df",
    "freMTPL2sev": "98276cab72280f3368eeef1f367e044147e559c96ec1cadb0c6bf006362121f1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_data(root: Path, source: str = "casdatasets") -> dict:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "data" / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    entries = dict(previous.get("datasets", {}))
    for name, data_id in DATASETS.items():
        destination = raw / f"{name}.csv.gz"
        if destination.exists():
            old = previous.get("datasets", {}).get(name)
            if not old or old["sha256"] != sha256(destination):
                raise ValueError(f"Unverified cache: {destination}. Restore its manifest or remove the file and download again.")
            if old.get("download_source") != source:
                raise ValueError(f"Existing cache comes from {old.get('download_source')}, not {source}. Use a fresh data directory to change source.")
            entries[name] = old
            print(f"Verified cache: {name}", flush=True)
            continue
        print(f"Downloading {source}: {name} ...", flush=True)
        if source == "openml":
            result = fetch_openml(
                data_id=data_id, as_frame=True, target_column=None,
                data_home=str(root / ".cache" / "openml"), parser="auto",
                n_retries=3, delay=3,
            )
            frame = result.frame
            provenance = {"source": f"https://www.openml.org/d/{data_id}", "openml_details": result.details}
        else:
            import rdata
            url = f"https://raw.githubusercontent.com/dutangc/CASdatasets/{CAS_COMMIT}/data/{name}.rda"
            rda = raw / f"{name}.rda"
            with urllib.request.urlopen(url, timeout=60) as response:
                rda.write_bytes(response.read())
            if sha256(rda) != CAS_SHA256[name]:
                raise ValueError(f"Source checksum differs from the pinned dataset: {name}")
            frame = rdata.read_rda(rda)[name]
            provenance = {"source": url, "git_commit": CAS_COMMIT,
                          "rda_sha256": sha256(rda), "source_package_license": "GPL (>= 2)",
                          "note": "Original authors' historical CASdatasets snapshot; downloaded because OpenML was unreachable. Not independently byte-matched to OpenML."}
        frame.to_csv(destination, index=False, compression={"method": "gzip", "mtime": 0})
        entries[name] = {
            "openml_id": data_id,
            "download_source": source, **provenance,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows": len(frame), "columns": list(frame.columns),
            "sha256": sha256(destination),
        }
        # Save after each download so a retry can reuse a completed first table.
        manifest_path.write_text(json.dumps({"datasets": entries}, indent=2), encoding="utf-8")
        print(f"Saved {len(frame):,} rows to {destination.name}", flush=True)
    manifest = {"datasets": entries}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def join_tables(frequency: pd.DataFrame, severity: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frequency, severity = frequency.copy(), severity.copy()
    for table in (frequency, severity):
        ids = pd.to_numeric(table["IDpol"], errors="raise")
        if ids.isna().any() or not np.isfinite(ids).all() or (ids % 1 != 0).any():
            raise ValueError("Policy IDs must be finite integers.")
        table["IDpol"] = ids.astype("int64")
    if frequency["IDpol"].duplicated().any():
        raise ValueError("Frequency table must have one row per policy.")
    if severity["ClaimAmount"].isna().any() or (severity["ClaimAmount"] < 0).any():
        raise ValueError("Severity amounts must be nonnegative and nonmissing.")
    orphan_mask = ~severity["IDpol"].isin(frequency["IDpol"])
    totals = severity.groupby("IDpol")["ClaimAmount"].sum()
    merged = frequency.set_index("IDpol").join(totals, how="left", validate="one_to_one")
    merged["ClaimAmount"] = merged["ClaimAmount"].fillna(0.0)
    for column in merged.select_dtypes(include=["object", "category"]).columns:
        merged[column] = merged[column].astype(str).str.strip("'")
    audit = {
        "frequency_rows": len(frequency), "severity_rows": len(severity),
        "joined_policies": len(merged), "orphan_severity_rows": int(orphan_mask.sum()),
        "orphan_claim_amount": float(severity.loc[orphan_mask, "ClaimAmount"].sum()),
        "raw_total_claim_amount": float(merged["ClaimAmount"].sum()),
        "raw_exposure_years": float(merged["Exposure"].sum()),
    }
    return merged, audit


def load_data(root: Path) -> tuple[pd.DataFrame, dict]:
    manifest_path = root / "data" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Download data first: python -m nonlife.run --download-only")
    manifest = json.loads(manifest_path.read_text())
    frames = {}
    for name in DATASETS:
        path = root / "data" / "raw" / f"{name}.csv.gz"
        if not path.exists() or sha256(path) != manifest["datasets"][name]["sha256"]:
            raise ValueError(f"Missing or changed source file: {path}")
        frames[name] = pd.read_csv(path)
    return join_tables(frames["freMTPL2freq"], frames["freMTPL2sev"])


def prepare_data(raw: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Use the benchmark's fixed caps, with an explicit audit of changed targets."""
    df = raw.copy()
    numeric = ["Exposure", "ClaimNb", "ClaimAmount", "Density", "VehAge", "DrivAge", "BonusMalus", "VehPower"]
    if df.isna().any().any() or not np.isfinite(df[numeric].to_numpy(float)).all():
        raise ValueError("Input has missing or non-finite values; investigate before modelling.")
    if (df["Exposure"] <= 0).any() or (df["Density"] <= 0).any():
        raise ValueError("Exposure and population density must be strictly positive.")
    if (df[["ClaimNb", "ClaimAmount"]] < 0).any().any() or (df["ClaimNb"] % 1 != 0).any():
        raise ValueError("Claim counts must be nonnegative integers; losses must be nonnegative.")
    if ((df["ClaimAmount"] > 0) & (df["ClaimNb"] == 0)).any():
        raise ValueError("Positive losses with zero claim count require reconciliation.")
    cap = config["claim_amount_cap"]
    audit = {
        "count_capped_policies": int((df.ClaimNb > config["claim_count_cap"]).sum()),
        "exposure_capped_policies": int((df.Exposure > config["exposure_cap"]).sum()),
        "amount_capped_policies": int((df.ClaimAmount > cap).sum()) if cap is not None else 0,
        "zero_amount_count_corrections": int(((df.ClaimAmount == 0) & (df.ClaimNb > 0)).sum()),
        "claim_amount_cap": cap,
    }
    df["ClaimNb"] = df.ClaimNb.clip(upper=config["claim_count_cap"])
    df["Exposure"] = df.Exposure.clip(upper=config["exposure_cap"])
    if cap is not None:
        df["ClaimAmount"] = df.ClaimAmount.clip(upper=cap)
    df.loc[df.ClaimAmount == 0, "ClaimNb"] = 0
    df["Frequency"] = df.ClaimNb / df.Exposure
    df["AvgClaimAmount"] = df.ClaimAmount / df.ClaimNb.clip(lower=1)
    df["PurePremium"] = df.ClaimAmount / df.Exposure
    audit.update({
        "claim_amount_removed": float(raw.ClaimAmount.sum() - df.ClaimAmount.sum()),
        "modelled_total_claim_amount": float(df.ClaimAmount.sum()),
        "modelled_exposure_years": float(df.Exposure.sum()),
        "positive_loss_policies": int((df.ClaimAmount > 0).sum()),
        "policy_count": len(df),
    })
    return df, audit
