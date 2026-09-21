# Pre-registration Protocol (fill in BEFORE the first real run)

Filling this file before seeing real numbers is what turns "we ran something" into a
defensible scientific record. Once a number is on screen, the protocol is frozen: any change
to a field below requires a new line in *Protocol amendments* with the reason, before the
amended run is re-executed.

- Date: __________
- Operator(s): __________
- Machine (hostname / instance type): __________

## 1. Research questions and which artefact answers them

| RQ | Answered by | Artefact |
|---|---|---|
| RQ1: robustness under compromise f | f-sweep mean±std | `real_evaluation_summary.csv` |
| RQ2: second-order injection ε | isolation on/off | `real_isolation_epsilon.csv` |
| RQ3: aggregation rule | rules sweep | `real_evaluation_summary.csv` |
| RQ4: backbone diversity | ρ / score ρ + diverse vs homogeneous | ablation `diverse_backbones` |
| RQ5: security-vs-cost | committee n + cost | `real_ablation.csv` + `real_cost_summary.csv` |

## 2. Frozen configuration

Copy the final config verbatim (commit the config file; pin the run to that file).

- Config file: ``configs/________________.yaml`` (commit hash: ________)
- Commit hash to run on: ________
- `n_seeds`: ______   `seed0`: ______
- Benchmark(s): ______________   split: ______   payload count(s): ________
- `backend`: ________   `backbones`: ________   `models`/`endpoints` overrides: ________
- `n_judges`: ______   `f`: ______   `threshold`: ______   `calibrate`: ______
- `measure_theory`: ______   `measure_epsilon`: ______
- Cache state at start: **cold** (delete `outputs/real_verdict_cache`) / warm (note why)

## 3. Sanity checks that must pass before numbers are accepted

| Check | Expected | Artefact |
|---|---|---|
| `git_commit` recorded and matches the commit above | non-empty | `*provenance.json` |
| `backend` != dummy | real backend name | `*provenance.json` |
| `is_paper_result` == false | false | `*provenance.json` |
| cost ledger shows >0 LLM calls (cold cache) | calls = n_judges × payloads | `real_cost_summary.csv` |
| ε ≈ 0 under isolation on | << 1 | `real_isolation_epsilon.csv` |
| significance table populated vs baseline | rows present | `real_evaluation_significance.csv` |
| theory quantities finite | r, γ, μ, ρ not nan | `real_theory_analysis.csv` |

## 4. Exact commands

Record the exact commands (with flags), which `run_real.sh` mode, and the log file:

```
```

## 5. Protocol amendments (chronological, with reasons)

| Date | Changed from → to | Reason | Run id re-executed |
|---|---|---|---|
|  |  |  |  |

## 6. What counts as a paper result

A number enters Table 5/6 / Figure 6 only when ALL of the above pass AND the run has been
reproduced ≥ 1 more time independently (different execution, same frozen config) with agreeing
summary CSVs. Then set `is_paper_result: true` in the provenance of the canonical run and
update `audits/result_integrity_audit.md`.