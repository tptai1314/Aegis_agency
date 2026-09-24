#!/usr/bin/env python3
"""Preflight validator for a real-data run: check the config against the world before GPU hours.

    python scripts/check_ready.py --config configs/ec2_real_evaluation.yaml
    python scripts/check_ready.py --config configs/ec2_real_evaluation.yaml --offline
    python scripts/check_ready.py --config my.yaml --output outputs/real_harmbench

Why this exists
---------------
Every failure mode found in the pre-server audit was silent or discovered hours into a run:
a benchmark with no benign slice calibrating to tau = 1.0, payloads longer than the served
context window, an endpoint that is not serving the model the config names, a cache whose
verdicts belong to a different isolation setting, an attack too weak to move a decision. This
script turns each of those into a line of output you read *before* starting the run.

Exit code is 0 when there is no FAIL (WARN is allowed), 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

from aegis_agency.data.adapters import CsvBenchmarkAdapter, resolve_benchmark_dir
from aegis_agency.data.real_judges import (
    JUDGE_PROMPT_VERSION,
    VerdictCache,
    build_real_judges,
)
from aegis_agency.experiments.run_real import (
    RealRunConfig,
    _effective_cache_dir,
    _judge_cache_context,
    parse_real_config,
)
from aegis_agency.utils.io import load_yaml
from aegis_agency.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"
_RESULTS: list[tuple[str, str, str]] = []


def record(status: str, area: str, message: str) -> None:
    _RESULTS.append((status, area, message))


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10.0)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def check_repo() -> None:
    commit = _git("rev-parse", "--short", "HEAD")
    if not commit:
        record(WARN, "repo", "not a git checkout: provenance git_commit will be 'not-a-git-repo'")
        return
    dirty = _git("status", "--porcelain")
    record(INFO, "repo", f"git commit {commit}")
    if dirty:
        record(
            WARN,
            "repo",
            f"{len(dirty.splitlines())} uncommitted path(s); the recorded commit will not match "
            f"the code that actually runs. Commit or stash before a reported run.",
        )
    else:
        record(PASS, "repo", "working tree clean")


def check_dataset(cfg: RealRunConfig) -> list:
    benchmark_dir = resolve_benchmark_dir(cfg.benchmark)
    root = Path(cfg.data_root) / benchmark_dir
    bench_path = root / f"{cfg.split}.csv"
    try:
        payloads = list(CsvBenchmarkAdapter(root=root, split=cfg.split).iter_payloads())
    except Exception as exc:
        record(FAIL, "dataset", f"{bench_path} unreadable: {type(exc).__name__}: {exc}")
        return []
    if not payloads:
        record(FAIL, "dataset", f"{bench_path} yielded no payloads")
        return []
    used = payloads[: cfg.limit] if cfg.limit and cfg.limit > 0 else payloads
    lengths = np.array([len(p.content) for p in used])
    n_block = sum(1 for p in used if int(p.true_label) == 1)
    record(
        INFO,
        "dataset",
        f"{cfg.benchmark} -> {bench_path} ({len(payloads)} rows in file, {len(used)} used)"
        f"{' [limit=' + str(cfg.limit) + ']' if cfg.limit else ''}",
    )
    record(
        INFO,
        "dataset",
        f"labels: {n_block} must-block / {len(used) - n_block} benign; payload length "
        f"min={lengths.min()} median={int(np.median(lengths))} p90={int(np.quantile(lengths, 0.9))} "
        f"max={lengths.max()} chars",
    )
    if cfg.max_prompt_chars:
        over = int((lengths > cfg.max_prompt_chars).sum())
        status = WARN if over else PASS
        record(
            status,
            "dataset",
            f"{over} payload(s) exceed max_prompt_chars={cfg.max_prompt_chars} and will be "
            f"middle-truncated (counted in the run provenance)",
        )
    else:
        longest = int(lengths.max())
        if longest > 4000:
            record(
                WARN,
                "dataset",
                f"max_prompt_chars=0 (no truncation) but the longest payload is {longest} chars. "
                f"With the default VLLM_MAX_LEN=8192 a payload this long will make vLLM return a "
                f"context-length error and abort the run. Set judges.max_prompt_chars (16000 is a "
                f"reasonable default) or raise VLLM_MAX_LEN.",
            )
        else:
            record(PASS, "dataset", "no truncation needed at the default context length")
    return used


def check_calibration(cfg: RealRunConfig, used: list) -> None:
    if not cfg.calibrate:
        record(INFO, "calibration", "disabled (fixed threshold tau=%.3f)" % cfg.threshold)
        return
    n_cal = max(1, len(used) // 4)
    cal = used[:n_cal]
    n_benign = sum(1 for p in cal if int(p.true_label) == 0)
    n_unsafe = n_cal - n_benign
    total_benign = sum(1 for p in used if int(p.true_label) == 0)
    record(
        INFO,
        "calibration",
        f"slice = first {n_cal} payload(s): {n_benign} benign / {n_unsafe} must-block; "
        f"minimum required = {cfg.calibration_min_samples}",
    )
    if total_benign == 0:
        record(
            FAIL,
            "calibration",
            f"benchmark {cfg.benchmark!r} has NO benign payloads, so target_orr calibration is "
            f"undefined. Set calibrate: false, or calibrate on a benchmark with benign items "
            f"(formal, second_order, benign, benign_xstest).",
        )
        return
    if n_benign < cfg.calibration_min_samples and cfg.calibration_on_insufficient == "raise":
        record(
            FAIL,
            "calibration",
            f"only {n_benign} benign calibration item(s) < calibration_min_samples="
            f"{cfg.calibration_min_samples}: the run will abort. Raise data.limit, lower "
            f"calibration_min_samples deliberately, or set calibrate: false.",
        )
    elif n_benign < cfg.calibration_min_samples:
        record(
            WARN,
            "calibration",
            f"only {n_benign} benign calibration item(s): with calibration_on_insufficient="
            f"conservative the run silently keeps tau=1.0 (block nothing), which maximises ASR-UC. "
            f"Prefer more calibration data.",
        )
    else:
        record(PASS, "calibration", f"{n_benign} benign calibration items available")


def check_attack(cfg: RealRunConfig) -> None:
    if cfg.attack == "none":
        record(INFO, "attack", "none: measuring the clean ASR/ORR baseline")
        return
    kwargs = dict(cfg.attack_kwargs)
    if cfg.attack == "collusion":
        radius = float(kwargs.get("radius", 0.1))
        budget = float(kwargs.get("budget", 1.0))
        shift = radius * budget
        record(INFO, "attack", f"collusion: coordinated score shift = {shift:.3f}")
        if shift < 0.2:
            record(
                WARN,
                "attack",
                f"collusion shift {shift:.3f} is small: real judge scores often sit near 0.9-1.0, "
                f"so the attack may never change a decision and every method will look identical. "
                f"Check `attack_effect` in real_evaluation_summary.csv after the first run and "
                f"raise attack_kwargs.radius/budget if it is 0.",
            )
        else:
            record(PASS, "attack", "shift is large enough to plausibly move a decision")
    else:
        record(INFO, "attack", f"{cfg.attack} with kwargs {kwargs or '(defaults)'}")

    if cfg.attack == "injection":
        record(
            WARN,
            "attack",
            "the `injection` attack is a verdict-space simulation (it perturbs honest verdicts "
            "with probability epsilon via RNG). It does NOT read the payload and is unaffected by "
            "judges.isolation; the *measured* isolation leak is real_isolation_epsilon.csv "
            "(measure_epsilon: true).",
        )


def _plan_judges(cfg: RealRunConfig):
    """Build the committee plan without touching the network (clients connect lazily)."""
    if cfg.backend == "dummy":
        return []
    return build_real_judges(
        cfg.backbones,
        backend=cfg.backend,
        n_judges=cfg.n_judges,
        model=cfg.model,
        endpoint=cfg.endpoint,
        embedding_model=cfg.embedding_model,
        isolation=cfg.isolation,
        models=cfg.models,
        endpoints=cfg.endpoints,
        max_prompt_chars=cfg.max_prompt_chars,
    )


def _probe_endpoint(base_url: str, expected_model: str, timeout: float = 6.0) -> tuple[str, str]:
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return FAIL, f"{url} unreachable ({type(exc).__name__}: {exc})"
    ids = [m.get("id") for m in payload.get("data", []) if isinstance(m, dict)]
    if expected_model in ids:
        return PASS, f"{url} serving {expected_model}"
    return WARN, f"{url} reachable but /v1/models lists {ids}, not {expected_model!r}"


def check_judges(cfg: RealRunConfig, offline: bool) -> list:
    if cfg.backend == "dummy":
        record(
            WARN,
            "judges",
            "backend=dummy: ground-truth verdicts, plumbing only. NEVER report these numbers. "
            "The run is diverted to a *_dummy cache directory automatically.",
        )
        return []
    judges = _plan_judges(cfg)
    plan: dict[str, list[str]] = {}
    for judge in judges:
        model = getattr(judge, "model", None) or getattr(judge, "model_path", None) or judge.backbone
        endpoint = getattr(judge, "endpoint_or_path", "")
        plan.setdefault(str(endpoint), [])
        if str(model) not in plan[str(endpoint)]:
            plan[str(endpoint)].append(str(model))
    distinct_models = {m for models in plan.values() for m in models}
    mixed = len(distinct_models) > 1
    record(
        INFO,
        "judges",
        f"backend={cfg.backend}, n_judges={cfg.n_judges}, backbones={list(cfg.backbones)}",
    )
    record(
        PASS if mixed else WARN,
        "judges",
        (
            f"committee uses {len(distinct_models)} distinct model(s) -> "
            + ("DIVERSE (RQ4 arm)" if mixed else "HOMOGENEOUS (all judges share one model)")
        ),
    )
    for endpoint, models in sorted(plan.items()):
        record(INFO, "judges", f"endpoint {endpoint or '(local path)'} -> {models}")
        if cfg.backend != "openai_compat" or offline:
            continue
        status, message = _probe_endpoint(endpoint, models[0])
        record(status, "judges", message)
    if cfg.embedding_model:
        record(
            INFO,
            "judges",
            f"rationale embeddings: {cfg.embedding_model} (downloaded on first use, ~90 MB; one "
            f"encoder instance per judge). Set judges.embedding_model: \"\" for score-only verdicts.",
        )
    else:
        record(INFO, "judges", "rationale embeddings disabled (m = 0, score-only verdicts)")
    return judges


def check_cache(cfg: RealRunConfig, judges: list) -> None:
    cache_dir = _effective_cache_dir(cfg)
    tag = "iso" if cfg.isolation else "noiso"
    expected = cache_dir / f"{cfg.benchmark}_{tag}_honest.jsonl"
    if not cache_dir.exists():
        record(INFO, "cache", f"{cache_dir} does not exist yet: this is a COLD cache run")
        return
    files = sorted(p.name for p in cache_dir.glob("*.jsonl"))
    record(INFO, "cache", f"{cache_dir}: {len(files)} file(s) {files}")
    if not expected.exists():
        record(INFO, "cache", f"{expected.name} absent: the honest committee will be computed fresh")
        return
    if judges:
        context = _judge_cache_context(cfg, cfg.isolation, judges[0])
        cache = VerdictCache(expected, context)
        record(INFO, "cache", f"{expected.name}: {len(cache)} record(s); context={context}")
        record(
            PASS if len(cache) else WARN,
            "cache",
            "rows matching this context are reused; mismatching rows are stale and re-queried "
            "(reported as n_stale_lookups in the run provenance)",
        )
    else:
        record(INFO, "cache", f"{expected.name} present; dummy runs use a separate *_dummy directory")


def check_disk(cfg: RealRunConfig) -> None:
    free_gb = shutil.disk_usage(".").free / (1024 ** 3)
    status = WARN if free_gb < 20 else PASS
    record(status, "disk", f"{free_gb:.1f} GB free on this filesystem (a full run wants >= 60 GB "
                          f"if model weights land here too)")


def check_runtime() -> None:
    record(INFO, "runtime", f"python {sys.version.split()[0]}, prompt version {JUDGE_PROMPT_VERSION}")
    try:
        import vllm  # noqa: F401

        record(PASS, "runtime", "vllm importable (self-hosted backbone possible)")
    except Exception as exc:
        record(WARN, "runtime", f"vllm not importable ({type(exc).__name__}): use an API backend")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a real-run config before spending GPU time.")
    ap.add_argument("--config", required=True, help="YAML config (experiment:/data:/judges:).")
    ap.add_argument("--offline", action="store_true", help="Skip endpoint/network probes.")
    ap.add_argument("--quiet", action="store_true", help="Only print WARN/FAIL lines.")
    args = ap.parse_args()

    configure_logging()
    path = Path(args.config)
    if not path.exists():
        print(f"Config not found: {path}", file=sys.stderr)
        return 1
    cfg = parse_real_config(load_yaml(path))

    check_repo()
    used = check_dataset(cfg)
    if used:
        check_calibration(cfg, used)
    check_attack(cfg)
    judges = check_judges(cfg, args.offline)
    check_cache(cfg, judges)
    check_disk(cfg)
    check_runtime()

    order = {FAIL: 0, WARN: 1, PASS: 2, INFO: 3}
    print(f"\nPreflight for {path} (benchmark={cfg.benchmark}, backend={cfg.backend})\n")
    for status, area, message in sorted(_RESULTS, key=lambda r: (order[r[0]], r[1])):
        if args.quiet and status in (PASS, INFO):
            continue
        print(f"  [{status:4s}] {area:12s} {message}")

    n_fail = sum(1 for s, _, _ in _RESULTS if s == FAIL)
    n_warn = sum(1 for s, _, _ in _RESULTS if s == WARN)
    print(f"\n  {n_fail} FAIL, {n_warn} WARN\n")
    if n_fail:
        print("  Fix the FAIL lines before starting the run.", file=sys.stderr)
        return 1
    print("  No blockers found. A WARN is not a blocker but should be a conscious choice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
