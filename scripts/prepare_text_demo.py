"""Fetch pinned, checksummed model files; validate and extract the bundled public PDFs.

Only data is downloaded. curl avoids a Windows urllib proxy discovery delay here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nonlife.data import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-model", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/text_demo/manifest.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "text_config.json").read_text())
    cache = ROOT / ".cache/text-model"
    cache.mkdir(parents=True, exist_ok=True)
    if args.download_model:
        for entry in manifest["model_files"]:
            target = cache / entry["file"]
            if target.exists() and sha256(target) == entry["sha256"]:
                continue
            url = f"https://huggingface.co/{config['model_id']}/resolve/{config['revision']}/{entry['file']}"
            temp = target.with_suffix(target.suffix + ".partial")
            subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location", "--connect-timeout", "15",
                            "--max-time", "180", "--output", str(temp), url], check=True)
            if sha256(temp) != entry["sha256"]:
                raise ValueError("Downloaded model checksum mismatch: " + entry["file"])
            temp.replace(target)
            print("Verified", entry["file"])
    from pypdf import PdfReader
    for entry in manifest["public_documents"]:
        pdf = ROOT / entry["pdf_path"]
        if sha256(pdf) != entry["pdf_sha256"]:
            raise ValueError("Public PDF differs from upstream manifest.")
        text = "\n".join(page.extract_text() for page in PdfReader(pdf).pages)
        target = ROOT / entry["text_path"]
        # Hash before replacing the committed text to catch parser-version differences.
        from hashlib import sha256 as hash_bytes
        if hash_bytes(text.encode("utf-8")).hexdigest() != entry["text_sha256"]:
            raise ValueError("Text differs; use the pinned pypdf version.")
        target.write_text(text, encoding="utf-8", newline="\n")
        print("Verified", pdf.name)


if __name__ == "__main__":
    main()
