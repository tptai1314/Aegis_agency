"""Synthetic payload generator for smoke tests and mechanism validation.

Generates a balanced set of benign (y*=0) and unsafe (y*=1) payloads with group tags that
stand in for attack families (e.g. 'jailbreak', 'injection'). Payload *content* is a short
synthetic string; the verdict values come from :mod:`aegis_agency.judges.synthetic_judges`.
Nothing here is a real benchmark; see :mod:`aegis_agency.data.adapters` for real data.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aegis_agency.data.schemas import Decision, Payload


def generate_payloads(
    n_payloads: int,
    rng: np.random.Generator,
    unsafe_fraction: float = 0.5,
    groups: Sequence[str] = ("jailbreak", "injection", "benign_traffic"),
) -> list[Payload]:
    """Generate ``n_payloads`` synthetic payloads with balanced labels and group tags."""
    if n_payloads < 1:
        raise ValueError("n_payloads must be >= 1.")
    if not (0.0 <= unsafe_fraction <= 1.0):
        raise ValueError("unsafe_fraction must be in [0, 1].")
    payloads: list[Payload] = []
    for i in range(n_payloads):
        label = int(Decision.BLOCK if rng.random() < unsafe_fraction else Decision.ALLOW)
        if label == Decision.BLOCK:
            group = str(rng.choice([g for g in groups if g != "benign_traffic"] or list(groups)))
            content = f"[synthetic unsafe payload #{i} | group={group}]"
        else:
            group = "benign_traffic"
            content = f"[synthetic benign payload #{i}]"
        payloads.append(
            Payload(
                payload_id=f"synthetic-{i:05d}",
                content=content,
                true_label=label,
                group=group,
                metadata={"synthetic": True},
            )
        )
    return payloads
