#!/usr/bin/env python3
"""Run the real-data evaluation on EC2 with real LLM judges (adapter-driven).

    python scripts/run_real_experiment.py --config configs/ec2_real_evaluation.yaml --output outputs/real

Options:
    --config   YAML with sections experiment:/data:/judges: (see configs/ec2_real_evaluation.yaml).
    --output   Output directory for result CSV + provenance JSON.
    --backend  Override the judge backend: openai_compat | anthropic | hf | dummy.
               "dummy" is for offline plumbing tests only; never report its outputs.
    --limit    Cap the number of payloads (0 = all). Use a small value to smoke-test first.

Models, endpoints, and datasets are never downloaded; point the config at what you have
already placed on the machine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aegis_agency.experiments.run_real import main as real_main


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Aegis-Agency real-data evaluation on EC2 (adapters; never auto-download)."
    )
    sub = ap.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate", help="Run the real benchmark evaluation.")
    ev.add_argument("--config", required=True)
    ev.add_argument("--output", default="outputs/real")
    ev.add_argument(
        "--backend",
        default=None,
        choices=["openai_compat", "anthropic", "hf", "dummy"],
        help="dummy = offline plumbing tests only.",
    )
    ev.add_argument("--limit", type=int, default=0, help="Cap payloads (0 = all).")
    args = ap.parse_args()

    path = Path(args.config)
    if not path.exists():
        ap.error(f"Config not found: {path}")
    cli_args = ["--config", str(path), "--output", args.output, "--limit", str(args.limit)]
    if args.backend:
        cli_args += ["--backend", args.backend]
    return real_main(cli_args)


if __name__ == "__main__":
    raise SystemExit(main())
