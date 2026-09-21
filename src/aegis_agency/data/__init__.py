"""Data layer: schemas, synthetic generators, and real-data / LLM adapters."""

from aegis_agency.data.adapters import (
    BENCHMARK_DIRS,
    BenchmarkAdapter,
    CsvBenchmarkAdapter,
    LLMJudgeAdapter,
    resolve_benchmark_dir,
)
from aegis_agency.data.schemas import (
    CommitteeConfig,
    Decision,
    DecisionResult,
    Payload,
    Verdict,
    stack_verdicts,
    verdict_vector,
)

__all__ = [
    "Verdict",
    "Payload",
    "CommitteeConfig",
    "DecisionResult",
    "Decision",
    "verdict_vector",
    "stack_verdicts",
    "BenchmarkAdapter",
    "CsvBenchmarkAdapter",
    "LLMJudgeAdapter",
    "BENCHMARK_DIRS",
    "resolve_benchmark_dir",
]
