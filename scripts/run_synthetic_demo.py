#!/usr/bin/env python3
"""Run the synthetic smoke-test demo end to end.

    python scripts/run_synthetic_demo.py --config configs/synthetic_demo.yaml --output outputs/synthetic_demo

Produces calibration.json, evaluation_sweep.csv, and asr_vs_f.png -- all clearly labelled as
synthetic smoke-test output, NOT paper results. Equivalent to
`scripts/run_experiment.py demo`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aegis_agency.cli import _load_cfg, add_common_run_args, run_demo
from aegis_agency.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description="Aegis-Agency synthetic smoke-test demo.")
    add_common_run_args(
        ap, output_default="outputs/synthetic_demo", config_default="configs/synthetic_demo.yaml"
    )
    args = ap.parse_args()

    configure_logging()
    run_demo(_load_cfg(args.config), Path(args.output))
    logger.info("Done. Synthetic smoke-test outputs in %s (NOT paper results).", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
