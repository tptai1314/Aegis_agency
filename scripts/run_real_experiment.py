#!/usr/bin/env python3
"""Run the real-data evaluation on EC2 with real LLM judges (adapter-driven).

    python scripts/run_real_experiment.py evaluate \
        --config configs/ec2_real_evaluation.yaml --output outputs/real
    python scripts/run_real_experiment.py ablate \
        --config configs/ec2_real_evaluation.yaml --output outputs/real
    # offline plumbing check (never report):
    python scripts/run_real_experiment.py evaluate \
        --config configs/ec2_real_evaluation.yaml --output outputs/real --backend dummy --limit 40

Commands:
    evaluate  f-sweep + summary + significance + theory + epsilon (Tables 5/6).
    ablate    component ablations on real verdicts (Table 6): robust agg on/off,
              committee size, isolation on/off (epsilon), measured diversity.

Options:
    --config   YAML with sections experiment:/data:/judges: (see configs/ec2_real_evaluation.yaml).
    --output   Output directory for result CSV + provenance JSON.
    --backend  Override the judge backend: openai_compat | anthropic | hf | dummy.
               "dummy" is for offline plumbing tests only; never report its outputs.
    --limit    Cap the number of payloads (0 = all). Use a small value to smoke-test first.

Models, endpoints, and datasets are never downloaded; point the config at what you have
already placed on the machine. See docs/ec2_experiment_guide.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aegis_agency.cli import add_common_run_args, add_real_run_args
from aegis_agency.experiments.run_real import (
    parse_real_config,
    run_real_ablation,
    run_real_evaluation,
)
from aegis_agency.utils.io import load_yaml
from aegis_agency.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

_RUNNERS = {
    "evaluate": run_real_evaluation,
    "ablate": run_real_ablation,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Aegis-Agency real-data evaluation on EC2 (adapters; never auto-download)."
    )
    sub = ap.add_subparsers(dest="command", required=True)
    commands = [
        ("evaluate", "Run the real benchmark evaluation (sweep + summary + theory + epsilon)."),
        ("ablate", "Run the component ablations on real verdicts (Table 6)."),
    ]
    for name, help_text in commands:
        p = sub.add_parser(name, help=help_text)
        add_common_run_args(p, output_default="outputs/real")
        add_real_run_args(p)
    args = ap.parse_args()

    path = Path(args.config) if args.config else None
    if path is None:
        ap.error("--config is required")
    if not path.exists():
        ap.error(f"Config not found: {path}")

    configure_logging()
    cfg = parse_real_config(load_yaml(path))
    if args.backend:
        cfg.backend = args.backend
    if args.limit:
        cfg.limit = args.limit
    _RUNNERS[args.command](cfg, args.output)
    logger.info(
        "Real '%s' complete; outputs in %s (is_paper_result=False until the paper's "
        "verification checklist is applied).",
        args.command,
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
