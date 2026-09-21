#!/usr/bin/env python3
"""Run a synthetic experiment stage (calibrate / evaluate / ablate / demo).

    python scripts/run_experiment.py calibrate --config configs/experiment_template.yaml --output outputs/exp
    python scripts/run_experiment.py evaluate --config configs/experiment_template.yaml --output outputs/exp
    python scripts/run_experiment.py ablate   --config configs/experiment_template.yaml --output outputs/exp
    python scripts/run_experiment.py demo     --config configs/synthetic_demo.yaml --output outputs/exp

This runs on synthetic verdict data and produces synthetic smoke-test outputs. The real-data
runner is `scripts/run_real_experiment.py` (see docs/ec2_experiment_guide.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aegis_agency.cli import _load_cfg, add_common_run_args, run_demo
from aegis_agency.experiments.run_ablation import run_ablation
from aegis_agency.experiments.run_calibration import run_calibration
from aegis_agency.experiments.run_evaluation import run_evaluation
from aegis_agency.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

_STAGE_HELP = {
    "calibrate": "Calibrate thresholds on synthetic honest data.",
    "evaluate": "Sweep the Byzantine fraction and record metrics.",
    "ablate": "Run the component ablations.",
    "demo": "Calibrate + evaluate + plot in one go.",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Aegis-Agency experiment runner (synthetic verdict data).")
    sub = ap.add_subparsers(dest="command", required=True)
    for name, help_text in _STAGE_HELP.items():
        add_common_run_args(sub.add_parser(name, help=help_text), output_default="outputs/experiment")
    args = ap.parse_args()
    if not args.config:
        ap.error("--config is required")

    configure_logging()
    cfg = _load_cfg(args.config)
    out = Path(args.output)

    if args.command == "calibrate":
        run_calibration(cfg, out)
    elif args.command == "evaluate":
        run_evaluation(cfg, out)
    elif args.command == "ablate":
        run_ablation(cfg, out)
    elif args.command == "demo":
        run_demo(cfg, out)
    logger.info("Stage '%s' complete; outputs in %s (synthetic smoke-test).", args.command, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
