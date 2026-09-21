"""Empirical estimators for the quantities the guarantees depend on (Section 8, Table 5).

Maps honest committee verdicts (real or synthetic) onto the measurable constants of the
paper so a run can report whether the conditions in Thm 1 / Prop 2 / Prop 3 actually bind:

* ``r``      honest radius        -- per-payload max distance of honest verdict vectors to
                                     their robust aggregate (Lemma 1 / Thm 1).
* ``gamma``  decision margin      -- distance of the honest aggregate from the decision
                                     boundary ``{u : u[0] = tau}``: ``|u_hat[0] - tau|``
                                     (Thm 1). The verdict vector is ordered score-first
                                     (Eq. 2), so the boundary is the affine plane x_0 = tau.
* ``rho``    correlated failure   -- average pairwise correlation of honest judges'
                                     correctness (Prop 3), accompanied by the mean error
                                     rate ``mu``; a score-coordinate correlation is also
                                     reported as an RQ4 homogeneity proxy.
* ``epsilon`` isolation leakage   -- fraction of honest verdicts (decision) that flip when
                                     the payload carries a second-order injection (Def 1).
* paired    significance          -- bootstrap mean-difference CI between two methods on
                                     per-payload outcome arrays (Section 10).

Every function is pure (verdicts/labels in, numbers out) so it is testable offline; the
real-data runner (:mod:`aegis_agency.experiments.run_real`) is the only sanctioned source of
verdicts that may feed these estimators.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aegis_agency.data.schemas import Verdict
from aegis_agency.methods.aggregators import aggregate
from aegis_agency.utils.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- honest radius
def honest_radius(
    committees: Sequence[Sequence[Verdict]],
    rule: str = "gmed",
    f: int = 0,
) -> np.ndarray:
    """Per-payload empirical honest radius ``r_p = max_k || u_k - u_hat ||``.

    The aggregate ``u_hat`` is the rule's center over the *honest* committee (no Byzantine
    rows); ``r`` is then the tightest radius around that center that covers the committee.
    """
    radii = np.empty(len(committees), dtype=float)
    for i, committee in enumerate(committees):
        u = _stack(committee)
        u_hat = aggregate(u, rule, f=f)
        radii[i] = float(np.linalg.norm(u - u_hat, axis=1).max())
    return radii


# ------------------------------------------------------------------------ decision margin
def decision_margin(
    committees: Sequence[Sequence[Verdict]],
    threshold: float,
    rule: str = "gmed",
    f: int = 0,
) -> np.ndarray:
    """Per-payload decision margin ``gamma_p = | u_hat[0] - tau |`` (Thm 1).

    ``u_hat[0]`` is the aggregate block-score ``s``; the decision boundary is the plane
    ``s = tau``, whose distance to ``u_hat`` is exactly ``|s_hat - tau|``. A positive margin
    means the honest committee's aggregate sits on the decided side of the boundary.
    """
    margins = np.empty(len(committees), dtype=float)
    for i, committee in enumerate(committees):
        u = _stack(committee)
        u_hat = aggregate(u, rule, f=f)
        margins[i] = float(abs(u_hat[0] - threshold))
    return margins


# --------------------------------------------------------------------- correlated failure
def failure_matrix(
    committees: Sequence[Sequence[Verdict]],
    labels: Sequence[int],
) -> np.ndarray:
    """``(K, P)`` matrix of honest-error indicators ``1[ d_k != y* ]`` (Prop 3)."""
    rows: list[np.ndarray] = []
    n_judges = len(committees[0]) if committees else 0
    for committee, label in zip(committees, labels):
        fails = np.array([1 if int(v.decision) != int(label) else 0 for v in committee], dtype=float)
        rows.append(fails)
    out = np.vstack(rows).T if rows else np.zeros((n_judges, 0), dtype=float)
    return out


def _pairwise_binomial_phi(fails: np.ndarray) -> float:
    """Mean pairwise phi coefficient over judges with non-degenerate columns.

    For binomial X, Y: phi = (p11 - p1 p2) / sqrt(p1(1-p1) p2(1-p2)). Judges whose observed
    error probability is 0 or 1 (zero variance) are skipped so the ratio stays defined.
    """
    n_judges = fails.shape[0]
    vals: list[float] = []
    for i in range(n_judges):
        pi = fails[i]
        p1 = pi.mean()
        denom_i = p1 * (1.0 - p1)
        if denom_i <= 0.0:
            continue
        for j in range(i + 1, n_judges):
            pj = fails[j]
            p2 = pj.mean()
            denom_j = p2 * (1.0 - p2)
            if denom_j <= 0.0:
                continue
            p11 = float((pi * pj).mean())
            vals.append((p11 - p1 * p2) / np.sqrt(denom_i * denom_j))
    if not vals:
        logger.warning("failure correlation: no non-degenerate judge pair; returning nan.")
        return float("nan")
    return float(np.mean(vals))


def _mean_pairwise_pearson(series: np.ndarray) -> float:
    """Mean pairwise Pearson correlation of judge series across payloads (RQ4 proxy).

    Uses the direct Pearson formula so the result is invariant to the ddof convention.
    """
    n_judges, n_payloads = series.shape
    if n_judges < 2 or n_payloads < 2:
        return float("nan")
    rows = [(x - x.mean()) for x in series]
    vals: list[float] = []
    for i in range(n_judges):
        dit = rows[i]
        v_i = float((dit * dit).sum())
        if v_i <= 0.0:
            continue
        for j in range(i + 1, n_judges):
            djt = rows[j]
            v_j = float((djt * djt).sum())
            if v_j <= 0.0:
                continue
            vals.append(float((dit * djt).sum() / np.sqrt(v_i * v_j)))
    clean = [v for v in vals if np.isfinite(v)]
    if not clean:
        return float("nan")
    return float(np.mean(clean))


def honest_correlation(
    committees: Sequence[Sequence[Verdict]],
    labels: Sequence[int],
) -> dict[str, float]:
    """Correlated honest failures (Prop 3): ``mu`` = mean error, ``rho`` = mean pairwise phi.

    Also returns ``score_rho`` (mean pairwise Pearson of the score coordinate across
    payloads) as a backbone-diversity proxy for RQ4. Degenerate series are skipped.
    """
    fails = failure_matrix(committees, labels)
    if fails.shape[1] == 0:
        return {"mu": float("nan"), "rho": float("nan"), "score_rho": float("nan")}
    mu = float(fails.mean())
    rho = _pairwise_binomial_phi(fails) if fails.shape[0] >= 2 else float("nan")
    scores = np.vstack([[v.score for v in committee] for committee in committees]).T
    score_rho = _mean_pairwise_pearson(scores) if scores.shape[0] >= 2 else float("nan")
    return {"mu": mu, "rho": rho, "score_rho": score_rho}


# ------------------------------------------------------------------------- isolation leak
def epsilon_estimates(
    clean: Sequence[Verdict],
    injected: Sequence[Verdict],
) -> dict[str, float]:
    """Def 1 empirical isolation leak from paired clean/injected verdicts.

    ``epsilon_decision`` = fraction of verdicts whose decision flips under an injected
    payload (0 = perfect isolation on the measured set). ``score_shift`` = mean |score
    difference|. ``tau``-free by construction (compares verdicts, not gate decisions).
    """
    if len(clean) != len(injected) or not clean:
        raise ValueError("clean and injected must be non-empty and equally sized.")
    flips = [1 if int(a.decision) != int(b.decision) else 0 for a, b in zip(clean, injected)]
    shifts = [abs(float(a.score) - float(b.score)) for a, b in zip(clean, injected)]
    return {
        "epsilon_decision": float(np.mean(flips)),
        "score_shift": float(np.mean(shifts)),
        "n_pairs": len(clean),
    }


def per_judge_flip_rates(
    clean: Sequence[Sequence[Verdict]],
    injected: Sequence[Sequence[Verdict]],
) -> dict[int, float]:
    """Per-judge decision flip rate over paired payloads (honest judge isolation, Def 1)."""
    flips: dict[int, list[int]] = {}
    for clean_committee, injected_committee in zip(clean, injected):
        for a, b in zip(clean_committee, injected_committee):
            flips.setdefault(int(a.judge_id), []).append(int(a.decision) != int(b.decision))
    return {j: float(np.mean(v)) for j, v in sorted(flips.items())}


# -------------------------------------------------------------------------- significance
def paired_bootstrap_mean_diff(
    a: np.ndarray | Sequence[float],
    b: np.ndarray | Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Paired-bootstrap mean difference ``mean(b) - mean(a)`` with a two-sided p-value.

    NaN pairs (e.g. a metric undefined for some payloads) are dropped so the arrays stay
    paired. Returns ``mean_diff``, ``ci_lo``, ``ci_hi``, ``p_value``, and ``significant``
    (True when the 1-alpha CI excludes 0 and p < alpha).
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError("a and b must be equally shaped paired sequences.")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size == 0:
        nan = float("nan")
        return {
            "mean_diff": nan, "ci_lo": nan, "ci_hi": nan,
            "p_value": nan, "significant": False, "n_pairs": 0,
        }
    if rng is None:
        rng = np.random.default_rng(0)
    diffs = y - x
    n = diffs.size
    boots = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[k] = diffs[idx].mean()
    mean_diff = float(diffs.mean())
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    if mean_diff > 0.0:
        p_value = 2.0 * float((boots <= 0.0).mean())
    elif mean_diff < 0.0:
        p_value = 2.0 * float((boots >= 0.0).mean())
    else:
        p_value = 1.0
    p_value = float(min(1.0, p_value))
    significant = bool(mean_diff != 0.0 and (lo > 0.0 or hi < 0.0) and p_value < alpha)
    return {
        "mean_diff": mean_diff,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_value": p_value,
        "significant": significant,
        "n_pairs": int(n),
    }


# --------------------------------------------------------------------------------- helper
def _stack(committee: Sequence[Verdict]) -> np.ndarray:
    from aegis_agency.data.schemas import stack_verdicts

    return stack_verdicts(committee)
