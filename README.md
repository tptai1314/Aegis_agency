# Aegis-Agency

**Byzantine-robust, injection-hardened multi-agent LLM defense pipelines — reference implementation.**

This repository implements the mechanism of the manuscript *Aegis-Agency: Byzantine-Robust,
Injection-Hardened Multi-Agent Defense Pipelines for Large Language Models*
(`AegisAgency_main.pdf`, in this project folder): a committee of hardened LLM "judge" agents
inspects a candidate output, and their verdicts are combined by a **Byzantine-robust
aggregation rule** (coordinate-wise median, geometric median, or Krum) instead of a single
trusted coordinator, while the untrusted payload is **isolated** from the judges' instruction
channel.

> ⚠️ **The benchmark CSVs ship in this repository** under `data/benchmarks/` (with sources and
> sha256 in `data/benchmarks/PROVENANCE.json`); only `universal_injection` still needs a real GPU
> optimiser run (`docs/universal_injection_runbook.md`). Model weights and credentials are never
> bundled: provision them on the GPU server you run on (EC2 or any rented/borrowed Ubuntu box —
> nothing in the code is AWS-specific).
>
> ⚠️ **No paper-level experimental results are claimed by this repository unless real
> benchmark outputs are generated and their provenance is recorded.** Everything produced by
> the synthetic demo is a *smoke-test artefact*, not a paper result. Before a real run, validate
> the config with `scripts/check_ready.py`; after it, record the run with `scripts/finalize_run.py`.

## Why a verdict-space implementation
The paper's theory is stated over **verdict vectors** `u_k = (s_k, e_k)` (Eq. 2), so the core
mechanism — aggregation rules, attacks, metrics, and the theorems — is implemented and tested
directly in that space. This lets the algorithm and math be validated offline with no LLM,
while real hardened LLM judges and jailbreak/injection benchmarks are reached through
documented adapters for EC2. See `docs/implementation_notes.md`.

## Paper-to-code map (summary)
| Paper | Code |
|---|---|
| Verdict / verdict vector (Eq. 1-2) | `src/aegis_agency/data/schemas.py` |
| Aggregation: cmed / gmed / Krum / majority (Alg. 2) | `src/aegis_agency/methods/aggregators.py` |
| Aegis adjudication (Alg. 1) | `src/aegis_agency/methods/gate.py` |
| Attacks: compromise / collusion / injection / adaptive (§5.3) | `src/aegis_agency/attacks/` |
| Metrics: ASR-UC, ORR, tolerance, detection F1/AUROC (Eq. 4-5, §10) | `src/aegis_agency/metrics/metrics.py` |
| Theory: C_alpha, integrity condition, correlated variance (L1, T1, P3) | `src/aegis_agency/metrics/theory.py` |
| Baselines: no-defense / single-model / majority / AutoDefense (§10) | `src/aegis_agency/baselines/` |
Full mapping: `audits/paper_to_code_traceability.csv`.

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # runtime + pytest/ruff/mypy
```
Requires Python ≥ 3.11. No GPU, no network needed for tests or the demo.

## Quick start (synthetic smoke-test)
```bash
make demo
# or:
python scripts/run_synthetic_demo.py --config configs/synthetic_demo.yaml --output outputs/synthetic_demo
```
Produces `outputs/synthetic_demo/{calibration.json, evaluation_sweep.csv, asr_vs_f.png}`,
all labelled synthetic. The demo sweeps the Byzantine count `f` and shows the robust rules
holding ASR-under-compromise near zero while the single-model baseline degrades.

CLI:
```bash
python -m aegis_agency.cli --help
python -m aegis_agency.cli info
python -m aegis_agency.cli evaluate --config configs/experiment_template.yaml --output outputs/eval
python -m aegis_agency.cli plot --input outputs/eval/evaluation_sweep.csv --output outputs/eval/asr.png
```

## Real-data usage (GPU server)
The benchmarks are already in the repo, so a fresh clone is enough to start; only the judge
backbone has to be provisioned (gated Llama weights or an API key). Full runbook:
`docs/ec2_experiment_guide.md` (the "ec2" folder/file names are historical, not AWS-specific).

```bash
# on the server, from the repository root
PYTHON=python3.12 bash scripts/ec2/setup.sh          # detects/installs Python >= 3.11 first
python scripts/check_ready.py --config configs/ec2_real_evaluation.yaml   # preflight: FAILs block
HF_TOKEN=... bash scripts/ec2/serve_vllm.sh          # judge backbone on 127.0.0.1:8001

AEGIS_OUT=outputs/real.smoke bash scripts/ec2/run_real.sh smoke   # dummy judge - plumbing only
AEGIS_OUT=outputs/real_harmbench bash scripts/ec2/run_real.sh full    # the real run (Table 5)
AEGIS_OUT=outputs/real_harmbench bash scripts/ec2/run_real.sh ablate  # Table 6 (reuses the cache)

bash scripts/ec2/backup_results.sh --outputs outputs/real_harmbench   # pack results + cache
python scripts/finalize_run.py --run outputs/real_harmbench           # verify + write run record
```

Use a per-benchmark `AEGIS_OUT`: every run writes the same file names, so a shared output
directory overwrites the previous benchmark's results. `judges.isolation` and the judge prompt are
part of the verdict-cache key, so changing either re-measures the committee (see
`docs/reproducibility.md`).

## Folder structure
```
AegisAgency/
  README.md  pyproject.toml  requirements.txt  Makefile  .gitignore  NOTES_AUDIT_FIXES.md
  configs/        default.yaml  synthetic_demo.yaml  experiment_template.yaml
                  local_smoke.yaml  local_ollama_pilot.yaml
                  ec2_real_evaluation.yaml  ec2_rq4_diverse.yaml
  src/aegis_agency/
    cli.py
    data/         schemas.py  synthetic.py  adapters.py  real_judges.py
    judges/       base.py  synthetic_judges.py  isolation.py
    methods/      aggregators.py  calibration.py  gate.py
    attacks/      compromise.py  collusion.py  injection.py  adaptive.py
    metrics/      metrics.py  theory.py  confidence_intervals.py  cost.py  estimators.py
    baselines/    no_defense.py  single_model.py  majority_vote.py  autodefense.py  external_wrappers.py
    experiments/  harness.py  run_calibration.py  run_evaluation.py  run_ablation.py  run_real.py  plot_results.py
    utils/        logging.py  seeding.py  io.py  validation.py  provenance.py
  scripts/        run_synthetic_demo.py  run_experiment.py  run_real_experiment.py  make_plots.py
                  check_ready.py  finalize_run.py  backup_results.ps1
                  data/  build_benchmarks.py  build_formal.py  build_second_order.py  build_xstest.py
                  prepare_universal_injection.py
                  ec2/  setup.sh  serve_vllm.sh  serve_backbones.sh  run_real.sh
                        backup_results.sh  run_universal_suffix.sh  upload_data.ps1
  tests/          (104 tests)
  examples/       example_config.yaml  synthetic_data/  README.md
  data/benchmarks/  8 benchmark dirs + PROVENANCE.json + _raw/
  docs/           implementation_notes.md  data_format.md  baseline_adapters.md
                  reproducibility.md  ec2_experiment_guide.md  universal_injection_runbook.md
  outputs/        (regenerable smoke-test artefacts; git-ignored)
  audits/         paper_implementation_spec.md  paper_to_code_traceability.csv  math_to_code_audit.md
                  implementation_gaps.md  result_integrity_audit.md  preregistration_protocol.md
                  run_records/   (generated by scripts/finalize_run.py)
  TODO_IMPLEMENTATION.md
```

## Implemented algorithms
Coordinate-wise median, geometric median (smoothed Weiszfeld), Krum selection, plain majority
vote; the Aegis adjudication gate (analyze → isolate → judge → robust-aggregate → allow/block/
escalate); threshold + temperature calibration.

## Metrics
ASR-under-compromise (Eq. 4), over-refusal rate (Eq. 5), defense success rate, Byzantine
tolerance fraction, malicious-verdict detection F1/AUROC, utility retention, group-conditional
ASR; token/latency cost models; Wilson / bootstrap confidence intervals; theory checks
(C_alpha, integrity condition, correlated-failure variance, injection flip bound).

Real runs additionally report **`attack_effect`**: the fraction of payloads whose gate decision the
attack actually changed (per `f` and method, in `real_evaluation_sweep.csv` /
`real_evaluation_summary.csv`). If it is 0 the attack never moved a decision, so every method
scores identically and the comparison is uninformative — the runner logs a warning in that case
rather than letting the table be misread as "all defences are equally robust".

## Baselines
Implemented: always-submit, single hardened model, plain majority vote, AutoDefense
(single-Coordinator mean). External (adapter stubs, EC2): real AutoDefense, SecAlign, StruQ,
JudgeDeceiver. A dummy baseline exists for tests only and is never presented as a real system.

## Configuration guide
YAML configs under `configs/` drive the synthetic simulation: committee size `n_judges`,
aggregation `rules`, honest margin/radius/correlation (`margin`/`radius`/`correlation`),
`attack`, Byzantine count `f`, and calibration target. The `data:` and `judges:` sections in
`experiment_template.yaml` are consumed only by the EC2 real-data adapters.

## Reproducibility
Fixed seeds via `utils/seeding.get_rng`; provenance JSON with every result; plots generated
from files with synthetic banners. See `docs/reproducibility.md`.

## Limitations and known gaps
- The synthetic layer validates the **mechanism and math**, not the paper's empirical claims.
- Real LLM judges and benchmarks are wired; the **external baselines** (real AutoDefense,
  SecAlign, StruQ) and the real JudgeDeceiver optimiser are **adapter stubs**.
- Krum's constant is not re-derived (the paper restates it); the aggregator is implemented.
- `epsilon`-isolation and honest concentration are modelled parametrically in synthetic runs;
  real values are measured by the runner (`real_isolation_epsilon.csv`, `real_theory_analysis.csv`).
- `temperature_scale` exists in `methods/calibration.py` (and is unit-tested) but is **not** applied
  to judge scores by the real runner: the paper's "temperature-calibrated scores" is not yet
  implemented end to end.
- The `injection` attack is a verdict-space RNG simulation of leakage; it does not read the payload
  and is independent of `judges.isolation`. The measured leak comes from replaying the injected
  suffix (`measure_epsilon: true`).
- `n_seeds` repeats the attack/aggregation draw only (honest verdicts are cached), so reported
  dispersion is not judge stochasticity.
- `universal_injection` is not shipped (needs a genuine GPU optimiser run).
Full lists: `audits/implementation_gaps.md`, `TODO_IMPLEMENTATION.md`, `NOTES_AUDIT_FIXES.md`.

## Tests
```bash
python -m pytest -q    # 104 tests, synthetic only, no network
python -m ruff check src tests
python -m mypy src tests
```

## License
MIT (placeholder — confirm before release).
