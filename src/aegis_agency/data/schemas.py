"""Structured representations of the paper's objects (Preliminaries, Eq. 1-2).

Objects
-------
Verdict            : v_k = (d_k, s_k, e_k)  -- Eq. (1)
verdict_vector     : u_k = (s_k, e_k) in R^d, d = 1 + m -- Eq. (2)
Payload            : an inspected candidate output with ground-truth label y*
CommitteeConfig    : n judges, aggregation rule, assumed f, backbones, diversity
DecisionResult     : allow/block/escalate decision + metadata (score, threshold, mode, group)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from aegis_agency.utils.validation import check_unit_interval


class Decision(enum.IntEnum):
    """Binary decision label. 0 = allow (release), 1 = block. (Eq. 2)"""

    ALLOW = 0
    BLOCK = 1


class GateMode(str, enum.Enum):
    """Final gate outcome, including the escalation branch of Algorithm 1."""

    ALLOW = "allow"
    BLOCK = "block"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class Verdict:
    """A single judge's verdict v_k = (d_k, s_k, e_k) -- Eq. (1).

    Parameters
    ----------
    decision : int in {0, 1}
        The judge's allow/block decision d_k.
    score : float in [0, 1]
        Calibrated block-score s_k (probability the payload violates the policy).
    embedding : np.ndarray, shape (m,)
        Rationale embedding e_k of the judge's free-text justification. May be empty
        (m = 0), in which case the verdict vector is the score alone (d = 1).
    judge_id : int
        Index of the emitting judge in the committee.
    is_byzantine : bool
        Ground-truth flag (known only in simulation/evaluation) used to score detection
        metrics; never consumed by the aggregation rules.
    """

    decision: int
    score: float
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    judge_id: int = -1
    is_byzantine: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", int(Decision(int(self.decision))))
        object.__setattr__(self, "score", check_unit_interval(float(self.score), "score"))
        emb = np.asarray(self.embedding, dtype=float).ravel()
        if emb.size and not np.all(np.isfinite(emb)):
            raise ValueError("Verdict embedding contains non-finite values.")
        object.__setattr__(self, "embedding", emb)

    def vector(self) -> np.ndarray:
        """Return the verdict vector u_k = (s_k, e_k) in R^d, d = 1 + m -- Eq. (2)."""
        return np.concatenate([[self.score], self.embedding])


def verdict_vector(v: Verdict) -> np.ndarray:
    """Free-function form of :meth:`Verdict.vector` (Eq. 2)."""
    return v.vector()


def stack_verdicts(verdicts: Sequence[Verdict]) -> np.ndarray:
    """Stack verdicts into an (n, d) matrix; validates equal dimension."""
    if not verdicts:
        raise ValueError("Cannot stack an empty verdict list.")
    vecs = [v.vector() for v in verdicts]
    dims = {vec.shape[0] for vec in vecs}
    if len(dims) != 1:
        raise ValueError(f"Verdict vectors have inconsistent dimensions: {sorted(dims)}.")
    return np.vstack(vecs)


@dataclass
class Payload:
    """An inspected candidate output (the object under adjudication).

    Attributes
    ----------
    payload_id : str
    content : str
        The candidate LLM output / inter-agent message text (opaque to the mechanism;
        judges consume it, aggregation does not).
    true_label : int in {0, 1}
        Ground-truth y* (1 = must block: unsafe or task-integrity violation). Known in
        evaluation only.
    group : str
        Mondrian-style group tag for group-conditional metrics (e.g. attack family).
    metadata : dict
        Free-form fields (benchmark, backbone, difficulty, ...).
    """

    payload_id: str
    content: str = ""
    true_label: int = Decision.ALLOW
    group: str = "default"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.true_label = int(Decision(int(self.true_label)))


@dataclass
class CommitteeConfig:
    """Configuration of the judge committee and aggregation rule."""

    n_judges: int
    rule: str = "gmed"  # one of {"majority", "cmed", "gmed", "krum"}
    assumed_f: int = 0  # f used by Krum's neighbour count
    threshold: float = 0.5  # tau
    escalate_band: float = 0.0  # delta (Algorithm 1, line 11); 0 disables escalation
    homogeneous: bool = True  # backbone diversity flag (RQ4)

    def __post_init__(self) -> None:
        if self.n_judges < 1:
            raise ValueError("n_judges must be >= 1.")
        if self.rule not in {"majority", "cmed", "gmed", "krum"}:
            raise ValueError(f"Unknown aggregation rule: {self.rule!r}.")
        check_unit_interval(self.threshold, "threshold")
        if self.escalate_band < 0:
            raise ValueError("escalate_band (delta) must be >= 0.")


@dataclass
class DecisionResult:
    """Output of the Aegis gate (Algorithm 1) with full decision metadata."""

    mode: GateMode
    decision: int  # allow/block collapsed from mode (escalate -> policy-defined)
    aggregate_score: float
    threshold: float
    rule: str
    group: str = "default"
    per_judge_outlier: Optional[np.ndarray] = None  # distance of each verdict to aggregate
    payload_id: str = ""

    def is_certified_safe(self) -> bool:
        """True iff the gate allowed the payload (released it)."""
        return self.mode is GateMode.ALLOW
