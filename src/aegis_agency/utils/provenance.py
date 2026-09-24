"""Run provenance: capture config, seed, versions, and a synthetic/real flag.

Every result file written by an experiment runner is accompanied by a provenance record so
that no output can be mistaken for a paper result without a traceable origin.

Beyond the run's own metadata the record captures what an *auditor* needs to reproduce the
run exactly: the git commit the code was at, the exact CLI command that started it, the
installed dependency versions, and the GPU/CUDA it ran on. Secrets (API keys, tokens) are
never recorded.
"""

from __future__ import annotations

import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata as _metadata
from typing import Any

#: Dependency versions that matter for numeric reproducibility (missing ones are skipped).
_DEP_NAMES = (
    "numpy",
    "matplotlib",
    "pyyaml",
    "torch",
    "openai",
    "anthropic",
    "transformers",
    "sentence-transformers",
    "vllm",
)


def _utc_now() -> str:
    """ISO-8601 UTC timestamp of the run (when the provenance record was created)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hostname() -> str:
    """Machine the run executed on; needed to tell two runs with equal configs apart."""
    try:
        return socket.gethostname()
    except Exception:  # pragma: no cover - platform-dependent
        return ""


def _aegis_version() -> str:
    try:
        return _metadata.version("aegis-agency")
    except _metadata.PackageNotFoundError:
        return "0.0.0-dev"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "not-a-git-repo"


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _DEP_NAMES:
        try:
            versions[name] = _metadata.version(name)
        except _metadata.PackageNotFoundError:
            continue
    return versions


def _gpu_info() -> str:
    try:
        import torch  # type: ignore[import-not-found]

        gpu_name = torch.cuda.get_device_name(0)
        if not gpu_name:
            return ""
        cuda = torch.version.cuda or "unknown"
        return f"{gpu_name} (CUDA {cuda})"
    except Exception:
        return ""


@dataclass
class RunProvenance:
    """Metadata attached to every result artefact.

    ``config`` holds the run's own knobs and ``extra`` holds side-channel evidence that is not a
    configuration value (dataset fingerprint, truncation counts, cache state, calibration
    outcome). Neither may ever contain a credential: this file is committed alongside results.
    """

    run_id: str
    stage: str
    seed: int | None
    config: dict[str, Any]
    data_source: str  # "synthetic" | "real"
    extra: dict[str, Any] = field(default_factory=dict)
    is_paper_result: bool = False
    aegis_version: str = field(default_factory=_aegis_version)
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    hostname: str = field(default_factory=_hostname)
    created_utc: str = field(default_factory=_utc_now)
    git_commit: str = field(default_factory=_git_commit)
    command: str = field(default_factory=lambda: " ".join(sys.argv))
    deps: dict[str, str] = field(default_factory=_package_versions)
    gpu: str = field(default_factory=_gpu_info)
    note: str = (
        "Synthetic smoke-test output, not a paper result, unless data_source == 'real' "
        "and is_paper_result is explicitly set True with documented provenance."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
