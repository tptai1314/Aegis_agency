"""Build the second-order (JudgeDeceiver) benchmark from a local checkout.

Purely offline: reads the ``ShiJiawenwen/JudgeDeceiver`` (ACM CCS 2024) repo cloned
on disk and emits ``<root>/second_order/test.csv`` in the repo schema
(``id, content, label, group``, see ``docs/data_format.md``).

Payload construction mirrors the official evaluator
(``judge_attack/base/attack_manager.py``, ``AttackPrompt._update_ids``): the
gradient-optimised suffix ``delta`` from ``dataset/results_suffix/basic/llmbar.json``
is appended, with a single separating space, directly after the answer the attack
wants the LLM-as-a-Judge to select. Both that attacked review text (``label=1``,
the isolation-leakage target for RQ2) and the identical clean review text without
``delta`` (``label=0``) are emitted per LLMBar test question.

Usage
-----
    python scripts/data/build_second_order.py --repo D:/Data/downloads/JudgeDeceiver \
        --root D:/Data/benchmarks

The suffix key defaults to ``llama-3`` (our hardened backbone is Llama-3.1-8B, the
same family the suffix was optimised against). Pass ``--model`` to pick another key
from the json (``mistral``, ``openchat_3.5``, ``llama-2``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# We are not importing the JudgeDeceiver package (it requires torch etc.); we only
# replicate the string assembly the evaluator performs.
ANSWER_A = "# Output (a):"
ANSWER_B = "\n# Output (b):"
SEP = " "
ASK = "\nWhich is better, Output (a) or Output (b)?\n"


def attacked_prompt(intro: str, text1: str, text2: str, control: str, target: str) -> str:
    if "(a)" in target:
        return f"{intro}{ANSWER_A}{SEP}{text1}{SEP}{control}{ANSWER_B}{SEP}{text2}{ASK}"
    return f"{intro}{ANSWER_A}{SEP}{text1}{ANSWER_B}{SEP}{text2}{SEP}{control}{ASK}"


def clean_prompt(intro: str, text1: str, text2: str, target: str) -> str:
    if "(a)" in target:
        return f"{intro}{ANSWER_A}{SEP}{text1}{ANSWER_B}{SEP}{text2}{ASK}"
    return f"{intro}{ANSWER_A}{SEP}{text1}{ANSWER_B}{SEP}{text2}{ASK}"


def load_suffixes(repo: Path) -> dict[int, dict[str, str]]:
    path = repo / "dataset" / "results_suffix" / "basic" / "llmbar.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, dict[str, str]] = {}
    for rec in records:
        qid = int(rec["question_id"])
        out[qid] = {
            k: str(v) for k, v in rec.items() if k not in {"question_id", "question", "bad_answer"}
        }
    return out


def build(args: argparse.Namespace) -> int:
    root = Path(args.root)
    repo = Path(args.repo)
    core = repo / "dataset" / "data_for_eval" / "basic" / "llmbar"
    suffixes = load_suffixes(repo)

    records: list[dict[str, Any]] = []
    per_file: dict[int, int] = {}
    for i in range(1, 11):
        qid = i
        if qid not in suffixes:
            print(f"  skip   llmbar_test_{i}.csv (no suffix for question_id {i})")
            continue
        delta = suffixes[qid].get(args.model)
        if not delta:
            raise ValueError(
                f"results_suffix has no suffix for model {args.model!r} "
                f"(question_id {qid}); available: {sorted(suffixes[qid])}"
            )
        csv_path = core / f"llmbar_test_{i}.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        for j, row in enumerate(rows):
            intro = (row.get("instruction") or "").strip()
            text1 = (row.get("text1") or "").strip()
            text2 = (row.get("text2") or "").strip()
            target = (row.get("target") or "").strip()
            if not (intro and text1 and text2 and target):
                print(f"  warn   llmbar_test_{i}.csv row {j} missing a field; skipped")
                continue
            records.append(
                {
                    "id": f"second_order_{i:02d}_{j:03d}_clean",
                    "content": clean_prompt(intro, text1, text2, target),
                    "label": 0,
                    "group": "llmbar",
                }
            )
            records.append(
                {
                    "id": f"second_order_{i:02d}_{j:03d}_attack",
                    "content": attacked_prompt(intro, text1, text2, delta, target),
                    "label": 1,
                    "group": "llmbar",
                }
            )
        per_file[i] = len(rows)

    out_csv = root / "second_order" / "test.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "content", "label", "group"])
        writer.writeheader()
        writer.writerows(records)

    sha = hashlib256(out_csv)
    provenance = {
        "source": "ShiJiawenwen/JudgeDeceiver (ACM CCS 2024, arXiv:2403.17710)",
        "repo": str(repo.resolve()),
        "dataset": "LLMBar",
        "files": {
            "review_pairs": "dataset/data_for_eval/basic/llmbar/llmbar_test_{1..10}.csv",
            "suffixes": "dataset/results_suffix/basic/llmbar.json",
        },
        "suffix_model": args.model,
        "payload": (
            "attacked rows replicate AttackPrompt._update_ids: "
            "intro + '# Output (a): ' + text1 [' + delta] + '\\n# Output (b): ' + text2 "
            "[+ ' ' + delta] + '\\nWhich is better, Output (a) or Output (b)?' "
            "with the suffix appended to the answer the attack picks"
        ),
        "rows_per_file": per_file,
        "rows": len(records),
        "positives": sum(1 for r in records if r["label"] == 1),
        "negatives": sum(1 for r in records if r["label"] == 0),
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "test_csv_sha256": sha,
    }
    merge_provenance(root, "second_order", provenance)
    print(f"  wrote   {out_csv}  rows={len(records)} (attacked={provenance['positives']}, "
          f"clean={provenance['negatives']}) for model key {args.model!r}")
    return 0


def hashlib256(path: Path) -> str:
    import hashlib

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="path to cloned JudgeDeceiver repo")
    parser.add_argument("--root", default="D:/Data/benchmarks", help="benchmark root")
    parser.add_argument("--model", default="llama-3", help="suffix key to use from results_suffix")
    return build(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
