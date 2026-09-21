# EC2 Experiment Guide

This repository is **EC2-ready** but does not download data or run paper-level experiments
during generation. Real experiments are run here, after you provision data, models, and
credentials manually.

> **Nothing in this repo auto-downloads datasets, weights, or benchmarks.** All of that is a
> manual, deliberate step you perform on the server.

## 1. Provision the environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q            # confirm the mechanism works on synthetic data
```
For real LLM judges you will additionally need the backbone runtimes (e.g. `transformers`,
`vllm`, or API SDKs) and any GPUs those require — these are **not** listed as dependencies
because the synthetic mechanism does not need them.

## 2. Place datasets (manual)
Prepare each benchmark as a CSV in the format of `docs/data_format.md`:
```
/data/harmbench/test.csv
/data/injecagent/test.csv
/data/benign/test.csv
```
Sources requiring manual download / license / registration (see `audits/implementation_gaps.md`):
- jailbreak: AdvBench/GCG, PAIR, TAP, GPTFuzzer, in-the-wild DAN, HarmBench.
- injection: formal injection benchmark, InjecAgent, universal injection.
- second-order: JudgeDeceiver optimiser (external repo) to generate injected payloads.

## 3. Wire real judges and baselines
Implement the adapter `predict()` / `judge()` methods:
- `data.adapters.LLMJudgeAdapter` — real hardened judge (backbone + SecAlign/StruQ + isolation).
- `baselines.external_wrappers.{AutoDefenseAdapter, SecAlignAdapter, StruQAdapter}` — real baselines.
- `baselines.external_wrappers.JudgeDeceiverAdapter` — real second-order injection.

Each returns/consumes the schema documented in `docs/baseline_adapters.md`. Provide checkpoint
paths / API endpoints via `ExternalBaselineConfig.model_path_or_endpoint`; **never commit
secrets** — read them from environment variables or a mounted secrets file.

## 4. Run the protocol (RQ1-RQ6)
Replace the synthetic verdict generator with real judge verdicts (via the adapters), then:
```bash
python scripts/run_real_experiment.py evaluate --config configs/ec2_real_evaluation.yaml --output outputs/real
```
This runs the multi-seed f-sweep (mean +/- std over `experiment.n_seeds`), measures the
theory constants (`measure_theory`) and Def 1 epsilon (`measure_epsilon`). Outputs:
```
outputs/real/real_evaluation_sweep.csv          # first-seed per-(f, method) metrics
outputs/real/real_evaluation_summary.csv        # mean +/- std over seeds (Table 5/6 format)
outputs/real/real_evaluation_significance.csv   # paired-bootstrap vs coordinator baseline
outputs/real/real_theory_analysis.csv           # measured r / gamma / mu / rho, Thm 1 check
outputs/real/real_isolation_epsilon.csv         # Def 1 epsilon (isolation on/off)
```
Plot the summary (mean ASR line):
```bash
python scripts/make_plots.py --input outputs/real/real_evaluation_summary.csv --output outputs/real/asr_vs_f.png --real
```
Only pass `--real` to plots when the inputs are genuinely real, verified results; set
`is_paper_result` in provenance only with documented provenance.

> Cost note: `measure_epsilon` replays an injected payload against every judge under both
> isolation settings (~2 extra judge calls per payload per judge). Set it `false` for a
> first cheap smoke pass on EC2.

## 5. Record provenance
Keep the `*provenance.json` files with every result, and update
`audits/result_integrity_audit.md` to state exactly which outputs are real, where the data
came from, and which numbers (if any) enter the paper.

## What EC2 does not change
- The mechanism, math, and metrics are identical to the synthetic runs; only the *source of
  verdicts* changes (synthetic generator → real LLM judges).
- The paper's tables/plots remain placeholders until these real runs are executed and their
  provenance recorded.
