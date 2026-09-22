"""Build the XSTest benign (over-refusal) benchmark.

Downloads the official XSTest prompt file (Rottger et al., NAACL 2024,
arXiv:2308.01263) from the authors' repository ``paul-rottger/xstest`` and emits
``<root>/benign_xstest/test.csv`` in the repo schema ``id, content, label, group``
(see ``docs/data_format.md``).

Only the 250 prompts labelled ``safe`` are used for the over-refusal-rate (ORR)
benchmark (``label=0``); the 200 ``contrast_*`` unsafe prompts are intentionally
excluded so the set stays benign-only. All 250 are never to be refused, so the
ORR is the fraction of these that the defence wrongly blocks.

Usage
-----
    python scripts/data/build_xstest.py --root D:/Data/benchmarks
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SOURCE_URL = (
    "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
)


def hashlib256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def merge_provenance(root: Path, name: str, entry: dict[str, Any]) -> None:
    pfile = root / "PROVENANCE.json"
    if pfile.exists():
        prov = json.loads(pfile.read_text(encoding="utf-8"))
    else:
        prov = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "schema": "id,content,label,group (docs/data_format.md)",
            "note": "Public research datasets fetched from canonical sources.",
            "benchmarks": {},
        }
    prov["benchmarks"][name] = entry
    pfile.write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  provenance -> {pfile}")


def build(args: argparse.Namespace) -> int:
    root = Path(args.root)
    raw = urllib.request.urlopen(SOURCE_URL, timeout=60).read()
    sha_download = hashlib.sha256(raw).hexdigest()
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    safe = [r for r in rows if r.get("label") == "safe"]
    if len(safe) != 250:
        raise ValueError(f"expected 250 safe prompts, found {len(safe)}")

    records = [
        {
            "id": f"xstest_{int(r['id']):04d}",
            "content": r["prompt"].strip(),
            "label": 0,
            "group": "xstest",
        }
        for r in safe
    ]

    per_type: dict[str, int] = {}
    for r in safe:
        per_type[r["type"]] = per_type.get(r["type"], 0) + 1

    out_csv = root / "benign_xstest" / "test.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "content", "label", "group"])
        writer.writeheader()
        writer.writerows(records)

    provenance = {
        "source": "paul-rottger/xstest (Rottger et al., NAACL 2024, arXiv:2308.01263)",
        "url": SOURCE_URL,
        "file": "xstest_prompts.csv",
        "download_sha256": sha_download,
        "subset": "only label=='safe' (250 prompts; 10 types x 25); unsafe contrast set excluded",
        "prompts_by_type": per_type,
        "rows": len(records),
        "positives": 0,
        "negatives": len(records),
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "test_csv_sha256": hashlib256(out_csv),
        "license": "No explicit data license in the repository; cite Rottger et al. (NAACL 2024).",
    }
    merge_provenance(root, "benign_xstest", provenance)
    print(f"  wrote   {out_csv}  rows={len(records)} (all label 0)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="D:/Data/benchmarks", help="benchmark root")
    return build(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())