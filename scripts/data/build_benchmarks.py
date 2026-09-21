"""Fetch and normalise the paper's benchmarks into the repo's CSV schema.

Never runs automatically as part of the library: this is an explicit, opt-in data
preparation script. It downloads public research datasets from their canonical
GitHub sources, normalises them to the ``CsvBenchmarkAdapter`` schema
(``id, content, label, group``, see ``docs/data_format.md``), and writes one
``<root>/<benchmark>/test.csv`` per benchmark.

Usage
-----
    python scripts/data/build_benchmarks.py --root data/benchmarks
    python scripts/data/build_benchmarks.py --root data/benchmarks --only harmbench dan
    python scripts/data/build_benchmarks.py --root data/benchmarks --raw-only
    python scripts/data/build_benchmarks.py --inspect injecagent

Provenance is written to ``<root>/PROVENANCE.json`` (source URL, licence note,
download timestamp, row counts). Datasets are never committed: ``/data/`` is
gitignored.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

RAW = "https://raw.githubusercontent.com"
USER_AGENT = "aegis-agency-research/1.0 (+data preparation for academic study)"

# --------------------------------------------------------------------------- sources
# Each entry: benchmark name -> (url, licence note)
SOURCES: dict[str, dict[str, Any]] = {
    "harmbench": {
        "url": f"{RAW}/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_test.csv",
        "licence": "HarmBench (Mazeika et al., ICML 2024) — MIT (see repo). Research use.",
        "family": "jailbreak",
    },
    "advbench": {
        "url": f"{RAW}/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
        "licence": "AdvBench / GCG (Zou et al., 2023) — MIT (see repo). Research use.",
        "family": "jailbreak",
    },
    "dan": {
        "url": f"{RAW}/verazuo/jailbreak_llms/main/data/prompts/jailbreak_prompts_2023_12_25.csv",
        "licence": "In-the-wild jailbreak prompts (Shen et al., ACM CCS 2024) — research use.",
        "family": "jailbreak",
    },
    "benign_discord": {
        "url": f"{RAW}/verazuo/jailbreak_llms/main/data/prompts/regular_prompts_2023_12_25.csv",
        "licence": "Regular (non-jailbreak) prompts, same corpus — research use.",
        "family": "benign",
    },
    "injecagent": {
        "url": f"{RAW}/uiuc-kang-lab/InjecAgent/main/data/test_cases_dh_base.json",
        "licence": "InjecAgent (Zhan et al., ACL Findings 2024) — MIT (see repo). Research use.",
        "family": "injection",
    },
}


# --------------------------------------------------------------------------- helpers
def _fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached  {dest.name} ({dest.stat().st_size} bytes)")
        return dest
    print(f"  GET     {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as resp:
        payload = resp.read()
    dest.write_bytes(payload)
    print(f"  saved   {dest.name} ({len(payload)} bytes)")
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rows_from_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def _slug(text: str, n: int = 24) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in text[:n]]
    return "".join(keep).strip("_") or "item"


def _write_csv(path: Path, records: Iterable[dict[str, Any]]) -> int:
    rows = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "content", "label", "group"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return len(rows)


# --------------------------------------------------------------------------- converters
def convert_harmbench(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """HarmBench: `Behavior` (+ optional `ContextString`), all labelled must-block."""
    out: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        behaviour = (r.get("Behavior") or "").strip()
        if not behaviour:
            continue
        context = (r.get("ContextString") or "").strip()
        content = f"{context}\n\n{behaviour}" if context else behaviour
        bid = (r.get("BehaviorID") or "").strip() or f"harmbench_{i:04d}"
        out.append(
            {
                "id": f"harmbench_{bid}",
                "content": content,
                "label": 1,
                "group": (r.get("FunctionalCategory") or "harmbench").strip() or "harmbench",
            }
        )
    return out


def convert_advbench(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """AdvBench: `goal` is the harmful behaviour."""
    out: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        goal = (r.get("goal") or "").strip()
        if not goal:
            continue
        out.append(
            {
                "id": f"advbench_{i:04d}",
                "content": goal,
                "label": 1,
                "group": "advbench",
            }
        )
    return out


def convert_dan(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """In-the-wild jailbreak prompts: keep rows flagged `jailbreak == True`."""
    out: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        prompt = (r.get("prompt") or "").strip()
        if not prompt:
            continue
        if str(r.get("jailbreak", "")).strip().lower() not in {"true", "1", "yes"}:
            continue
        out.append(
            {
                "id": f"dan_{i:05d}",
                "content": prompt,
                "label": 1,
                "group": (r.get("source") or "dan").strip() or "dan",
            }
        )
    return out


def convert_benign(rows: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, Any]]:
    """Regular (non-jailbreak) prompts from the same corpus, sampled deterministically."""
    pool: list[dict[str, str]] = []
    for r in rows:
        prompt = (r.get("prompt") or "").strip()
        if not prompt:
            continue
        if str(r.get("jailbreak", "")).strip().lower() in {"true", "1", "yes"}:
            continue
        pool.append(r)
    rng = random.Random(seed)
    rng.shuffle(pool)
    out: list[dict[str, Any]] = []
    for i, r in enumerate(pool[:limit]):
        out.append(
            {
                "id": f"benign_{i:05d}",
                "content": (r.get("prompt") or "").strip(),
                "label": 0,
                "group": "benign_discord",
            }
        )
    return out


def convert_injecagent(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """InjecAgent direct-harm cases.

    The defended pipeline inspects *untrusted text that reached an agent*. In InjecAgent
    that text is the poisoned tool response; we therefore emit the tool response with the
    attacker instruction embedded, which is exactly what a judge agent would have to read.
    """
    out: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        # `Tool Response` is the poisoned response with the attacker instruction already
        # substituted in; fall back to substituting the template ourselves.
        tool_response = str(rec.get("Tool Response") or "").strip()
        if not tool_response:
            template = str(rec.get("Tool Response Template") or "").strip()
            attacker = str(rec.get("Attacker Instruction") or "").strip()
            tool_response = template.replace("<Attacker Instruction>", attacker).strip()
        user_instr = str(rec.get("User Instruction") or "").strip()
        if not tool_response:
            continue
        content = f"User instruction: {user_instr}\nTool response: {tool_response}".strip()
        attacker = str(rec.get("Attacker Instruction") or "").strip()
        out.append(
            {
                "id": f"injecagent_{i:04d}_{_slug(attacker)}",
                "content": content,
                "label": 1,
                "group": str(rec.get("Attack Type") or "injecagent").strip() or "injecagent",
            }
        )
    return out


CONVERTERS: dict[str, Callable[[list[Any]], list[dict[str, Any]]]] = {
    "harmbench": convert_harmbench,
    "advbench": convert_advbench,
    "dan": convert_dan,
    "injecagent": convert_injecagent,
}


# --------------------------------------------------------------------------- driver
def _load(name: str, raw_dir: Path) -> tuple[Any, Path]:
    src = SOURCES[name]
    suffix = ".json" if src["url"].endswith(".json") else ".csv"
    path = _fetch(src["url"], raw_dir / f"{name}{suffix}")
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8")), path
    return _rows_from_csv(path), path


def build(args: argparse.Namespace) -> int:
    root = Path(args.root)
    raw_dir = root / "_raw"
    provenance: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema": "id,content,label,group (docs/data_format.md)",
        "note": "Public research datasets fetched from canonical GitHub sources.",
        "benchmarks": {},
    }
    names = args.only or ["harmbench", "advbench", "dan", "injecagent", "benign_discord"]

    for name in names:
        if name not in SOURCES:
            print(f"skip unknown benchmark {name!r}")
            continue
        print(f"[{name}] {SOURCES[name]['family']}")
        data, path = _load(name, raw_dir)
        if name == "benign_discord":
            records = convert_benign(data, args.benign_limit, args.seed)
            out_name = "benign"
        else:
            records = CONVERTERS[name](data)
            out_name = name
        n = _write_csv(root / out_name / "test.csv", records)
        provenance["benchmarks"][out_name] = {
            "source_url": SOURCES[name]["url"],
            "licence_note": SOURCES[name]["licence"],
            "family": SOURCES[name]["family"],
            "raw_file": path.name,
            "raw_sha256": _sha256(path),
            "rows": n,
            "positives": sum(1 for r in records if r["label"] == 1),
            "groups": dict(Counter(str(r["group"]) for r in records).most_common(12)),
        }
        print(f"  wrote   {root / out_name / 'test.csv'}  rows={n}")

    _enforce_disjoint_benign(root, provenance)

    (root / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nprovenance -> {root / 'PROVENANCE.json'}")
    return 0


def _enforce_disjoint_benign(root: Path, provenance: dict[str, Any]) -> None:
    """Remove benign rows whose exact text also appears in an attack benchmark.

    Over-refusal (Eq. 5) is only meaningful if the benign set does not contain
    attack payloads. The in-the-wild corpora share a Discord source, so a small
    exact-text overlap is possible; we drop it rather than argue about it.
    """
    benign_csv = root / "benign" / "test.csv"
    if not benign_csv.exists():
        return
    attack: set[str] = set()
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in {"benign", "_raw"}):
        f = d / "test.csv"
        if f.exists():
            attack |= {r["content"].strip() for r in _rows_from_csv(f)}

    rows = _rows_from_csv(benign_csv)
    kept = [r for r in rows if r["content"].strip() not in attack]
    removed = len(rows) - len(kept)
    if removed:
        _write_csv(benign_csv, kept)
        provenance["benchmarks"].setdefault("benign", {})["rows"] = len(kept)
        provenance["benchmarks"]["benign"]["dropped_overlapping_attack_text"] = removed
        print(f"  dedup   benign: dropped {removed} row(s) also present in an attack set")


def inspect(name: str) -> int:
    """Dump the structure of a raw source (for schema verification)."""
    import tempfile

    tmp = Path(tempfile.gettempdir()) / "aegis_raw"
    data, path = _load(name, tmp)
    print(f"{name}: {path} ({path.stat().st_size} bytes)")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        print("keys:", list(data[0].keys()))
        print(json.dumps(data[0], ensure_ascii=False, indent=2)[:2000])
    elif isinstance(data, list):
        print(f"list of {len(data)} csv rows; header: {list(data[0].keys())}")
        print(json.dumps(data[0], ensure_ascii=False, indent=2)[:1200])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/benchmarks", help="output root (data.root)")
    parser.add_argument("--only", nargs="*", default=None, help="subset of benchmark names")
    parser.add_argument("--benign-limit", type=int, default=300, help="benign sample size")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed (reproducible)")
    parser.add_argument("--inspect", default=None, help="dump a raw source structure and exit")
    args = parser.parse_args(argv)

    if args.inspect:
        return inspect(args.inspect)
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
