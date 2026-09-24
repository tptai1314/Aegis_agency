# Data Format

The prepared benchmarks **ship in this repository** under `data/benchmarks/` (tracked by git, with
sources, licences and sha256 in `data/benchmarks/PROVENANCE.json`). The builders in
`scripts/data/` are kept so the CSVs can be regenerated from upstream; they are not needed to run
an experiment. Only `universal_injection` still requires a GPU optimiser run
(`docs/universal_injection_runbook.md`).

## Verdict-space objects (always)
- **Verdict** `v_k = (d_k, s_k, e_k)`: decision `d_k ∈ {0,1}`, score `s_k ∈ [0,1]`,
  embedding `e_k ∈ R^m`.
- **Verdict vector** `u_k = (s_k, e_k) ∈ R^d`, `d = 1 + m`.
- **Payload**: `payload_id`, `content`, `true_label ∈ {0,1}` (1 = must block), `group`,
  `metadata`.

## CSV benchmark format (`CsvBenchmarkAdapter`)
Directory layout:
```
<data.root>/
  test.csv          # or train.csv / calibration.csv, chosen by data.split
```
`test.csv` columns (header required):

| column    | type | meaning |
|-----------|------|---------|
| `id`      | str  | unique payload id |
| `content` | str  | candidate LLM output / inter-agent message |
| `label`   | int  | `1` = must block (unsafe / injection success target), `0` = benign |
| `group`   | str  | (optional) attack family / group tag |

Extra columns are preserved in `Payload.metadata`. See
`examples/synthetic_data/example_benchmark.csv`.

## Benchmarks the paper uses (shipped in `data/benchmarks/`)

| Family | Sources (paper Section 10) | Notes |
|--------|----------------------------|-------|
| jailbreak | AdvBench/GCG, PAIR, TAP, GPTFuzzer, in-the-wild DAN, HarmBench harness | label all as `1` (must block); benign controls as `0` |
| injection | formal injection benchmark, InjecAgent, universal injection | label task-integrity violations as `1` |
| benign | held-out instruction-following / QA | label `0`; used for over-refusal / utility |
| second-order | JudgeDeceiver-generated payloads | used to estimate isolation leakage epsilon (RQ2) |

Config-level names (`data.benchmark`) and their directories (`data/adapters.py::BENCHMARK_DIRS`):

| Config key | Dir | Source | Rows | Builder |
|---|---|---|---|---|
| `harmbench` | `harmbench/` | `centerforaisafety/HarmBench` (ICML'24) | 320 (320/0) | `scripts/data/build_benchmarks.py` |
| `advbench` | `advbench/` | `llm-attacks/llm-attacks` (GCG) | 520 (520/0) | `scripts/data/build_benchmarks.py` |
| `dan` | `dan/` | `verazuo/jailbreak_llms` (CCS'24), mixed DAN/PAIR/TAP/GPTFuzzer corpus | 1405 (1405/0) | `scripts/data/build_benchmarks.py` |
| `injecagent` | `injecagent/` | `uiuc-kang-lab/InjecAgent` (ACL'24) | 510 (510/0) | `scripts/data/build_benchmarks.py` |
| `benign` | `benign/` | regular (non-jailbreak) prompts, same corpus | 299 (0/299) | `scripts/data/build_benchmarks.py` |
| `formal_injection` | `formal/` | `liu00222/Open-Prompt-Injection` (USENIX'24), 7 tasks, official pipeline re-created | 1400 (700 clean / 700 attack) | `scripts/data/build_formal.py` |
| `second_order` | `second_order/` | `ShiJiawenwen/JudgeDeceiver` (CCS'24), LLMBar + optimized suffix (key `llama-3`) | 2000 (1000 clean / 1000 attack) | `scripts/data/build_second_order.py` |
| `benign_xstest` | `benign_xstest/` | `paul-rottger/xstest` (NAACL'24), 250 safe prompts | 250 (all `0`) | `scripts/data/build_xstest.py` |
| `universal_injection` | — | `SheltonLiu-N/Universal-Prompt-Injection` (liu2024universal) | **not built** | `scripts/ec2/run_universal_suffix.sh` + `scripts/prepare_universal_injection.py` |

> **The config key is `formal_injection`, not `formal`** (the directory is `formal/`). Passing
> `formal` raises `ValueError: Unknown benchmark 'formal'`.

> **`dan` contains payloads up to ~55k characters.** With the default served context
> (`VLLM_MAX_LEN=8192`) those must be shortened via `judges.max_prompt_chars` or vLLM returns a
> context-length error and the run aborts. `scripts/check_ready.py` reports the count.

> **In-the-wild DAN is a mixed corpus:** in this project's `dan/` benchmark, PAIR / TAP /
> GPTFuzzer / DAN-style prompts all appear in the same `test.csv` (no family column). To
> report a family separately, split it out into its own CSV and place it at one of the
> optional dirs, which `BENCHMARK_DIRS` in `data/adapters.py` already accepts: `pair/`,
> `tap/`, `gptfuzzer/` (each needs its own `test.csv` in the schema above). Record in
> provenance that a heuristic split is not the official family set.

> **Label split matters for which metrics exist.** `harmbench`, `advbench`, `dan` and `injecagent`
> are all must-block: ORR is undefined (`NaN`) and `defense_success_rate` is just
> `1 − ASR-UC`. Measure ORR/utility on `benign`, `benign_xstest`, or the clean half of
> `formal`/`second_order`. Threshold calibration (`experiment.calibrate: true`) also needs benign
> rows in the calibration slice; on an all-unsafe benchmark the run now aborts instead of silently
> falling back to `tau = 1.0`.

> **`universal_injection/` is currently ABSENT** (the earlier simulated stand-in was deleted
> because it was not the real mechanism). It should be built only with the genuine
> gradient-optimized suffix: run `scripts/ec2/run_universal_suffix.sh` on a GPU (local
> RTX 3050 4 GB is too small) against the gated `meta-llama/Llama-2-7b-chat-hf`, `scp` the
> suffix results back, then rebuild so `gradient_optimized_suffix: true` is recorded in
> `PROVENANCE.json` (tracked as T1.5b). Until then the config `universal_injection` is a
> no-op (dir missing -> the adapter raises a clear error, same message as any unplaced
> benchmark).

## Real LLM-judge outputs (`LLMJudgeAdapter`)
A real judge must return a `Verdict`:
- run the hardened backbone (Llama-3 / Qwen2.5 / Mistral / GPT-4o / Claude-3.5) with the
  payload placed in the **data channel** behind the operator's isolation delimiters;
- map the model's decision to `d_k`, its block-probability to `s_k`, and an
  embedding of its rationale to `e_k` (a judge that returns no rationale gets a zero vector of
  the same dimension, so one terse answer cannot break the committee).

Note: `temperature_scale` (`methods/calibration.py`) is implemented and unit-tested but is **not**
applied to judge scores by the runner, so `s_k` is the raw model score rather than the paper's
temperature-calibrated score.

## What is NOT provided
Model weights, API keys, credentials, and the `universal_injection` benchmark are not included.
See `docs/ec2_experiment_guide.md`, `docs/universal_injection_runbook.md`,
`audits/implementation_gaps.md`, and `TODO_IMPLEMENTATION.md`.
