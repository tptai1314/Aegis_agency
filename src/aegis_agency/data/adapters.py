"""Real-data and real-LLM-judge adapters (EC2-only; never auto-download).

These adapters define the expected on-disk formats for the benchmarks the paper uses and the
interface for real LLM judges. They read data the user has *already placed* under
``data.root`` on EC2; they never download anything. Until real files/weights are present the
adapters raise a clear error pointing to the setup docs.

Supported benchmark families (see docs/data_format.md):
* jailbreak : AdvBench/GCG, PAIR, TAP, GPTFuzzer, in-the-wild DAN, HarmBench harness.
* injection : formal injection benchmark, InjecAgent, universal injection.
* benign    : instruction-following / QA held-out benign traffic.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from aegis_agency.data.schemas import Decision, Payload, Verdict
from aegis_agency.utils.logging import get_logger

logger = get_logger(__name__)

#: Mapping from a config-level benchmark name to the on-disk directory under data.root.
BENCHMARK_DIRS: dict[str, str] = {
    "harmbench": "harmbench",
    "advbench": "advbench",
    "injecagent": "injecagent",
    "formal_injection": "formal",
    "dan": "dan",
    "second_order": "second_order",
    "benign": "benign",
}


def resolve_benchmark_dir(name: str) -> str:
    """Return the on-disk directory for a config-level benchmark name."""
    if name not in BENCHMARK_DIRS:
        raise ValueError(
            f"Unknown benchmark {name!r}. Choose one of: {sorted(BENCHMARK_DIRS)}."
        )
    return BENCHMARK_DIRS[name]


class BenchmarkAdapter(abc.ABC):
    """Reads payloads from a benchmark already present on disk (no downloading)."""

    #: Expected columns/keys for the benchmark's item records.
    REQUIRED_FIELDS: tuple[str, ...] = ("id", "content", "label")

    def __init__(self, root: str | Path, split: str = "test"):
        self.root = Path(root)
        self.split = split

    def _require_root(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(
                f"Benchmark root {self.root} does not exist. Place the dataset manually on "
                f"EC2 (see docs/data_format.md and docs/ec2_experiment_guide.md). This "
                f"repository never downloads data automatically."
            )

    @abc.abstractmethod
    def iter_payloads(self) -> Iterator[Payload]:
        """Yield payloads from the benchmark. Must validate REQUIRED_FIELDS per record."""
        raise NotImplementedError


class CsvBenchmarkAdapter(BenchmarkAdapter):
    """Generic CSV adapter: expects columns id, content, label[, group] under root/split.csv.

    label semantics: 1 = must block (unsafe / injection success target), 0 = benign.
    """

    def iter_payloads(self) -> Iterator[Payload]:
        import csv

        self._require_root()
        path = self.root / f"{self.split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Expected {path}. See docs/data_format.md for the required CSV schema."
            )
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = set(self.REQUIRED_FIELDS) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{path} missing required columns {sorted(missing)}; "
                    f"found {reader.fieldnames}. See docs/data_format.md."
                )
            for row in reader:
                yield Payload(
                    payload_id=str(row["id"]),
                    content=str(row["content"]),
                    true_label=int(Decision.BLOCK if int(row["label"]) == 1 else Decision.ALLOW),
                    group=str(row.get("group", "default")),
                    metadata={k: v for k, v in row.items() if k not in {"id", "content", "label", "group"}},
                )


class LLMJudgeAdapter(abc.ABC):
    """Interface for a real hardened LLM judge running on EC2.

    Implementations load a backbone (Llama-3 / Qwen2.5 / Mistral / GPT-4o / Claude-3.5),
    apply SecAlign/StruQ hardening + payload isolation, and return a Verdict. The payload
    MUST be placed in the data channel with the operator's isolation delimiters; the judge
    MUST treat data-channel instructions as inert (Def. 1).
    """

    def __init__(self, backbone: str, endpoint_or_path: str = "", isolation: bool = True):
        self.backbone = backbone
        self.endpoint_or_path = endpoint_or_path
        self.isolation = isolation
        self._ledger: Any = None

    @abc.abstractmethod
    def judge(self, payload: Payload, judge_id: int, rng: np.random.Generator) -> Verdict:
        raise NotImplementedError  # pragma: no cover

    def set_ledger(self, ledger: Any) -> None:
        """Attach a UsageLedger so ``judge()`` records cost/latency (RQ5). Optional."""
        self._ledger = ledger

    def _record_cost(
        self,
        payload_id: str,
        judge_id: int,
        elapsed_s: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        if self._ledger is not None:
            self._ledger.record(
                backbone=self.backbone,
                judge_id=int(judge_id),
                payload_id=payload_id,
                elapsed_s=float(elapsed_s),
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
            )
