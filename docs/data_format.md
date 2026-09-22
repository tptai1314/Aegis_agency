# Data Format

This repository **never downloads data**. On EC2 you place datasets on disk and point the
config at them. This document defines the expected formats.

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

## Benchmarks the paper uses (place manually on EC2)
These require manual download / license acceptance / repository setup — **not automated
here**. Prepare each as a CSV in the format above (or write a bespoke adapter subclassing
`BenchmarkAdapter`).

| Family | Sources (paper Section 10) | Notes |
|--------|----------------------------|-------|
| jailbreak | AdvBench/GCG, PAIR, TAP, GPTFuzzer, in-the-wild DAN, HarmBench harness | label all as `1` (must block); benign controls as `0` |
| injection | formal injection benchmark, InjecAgent, universal injection | label task-integrity violations as `1` |
| benign | held-out instruction-following / QA | label `0`; used for over-refusal / utility |
| second-order | JudgeDeceiver-generated payloads | used to estimate isolation leakage epsilon (RQ2) |

Prepared benchmarks (already built on the local machine at `D:\Data\benchmarks`, tracked in
`PROVENANCE.json`):

| Dir | Source | Rows | Builder |
|-----|--------|------|---------|
| `formal/` | `liu00222/Open-Prompt-Injection` (USENIX'24), 7 tasks, official pipeline re-created | 1400 (700 clean / 700 attack) | `scripts/data/build_formal.py` |
| `second_order/` | `ShiJiawenwen/JudgeDeceiver` (CCS'24), LLMBar + optimized suffix (key `llama-3`) | 2000 (1000 clean / 1000 attack) | `scripts/data/build_second_order.py` |
| `benign_xstest/` | `paul-rottger/xstest` (NAACL'24), 250 safe prompts | 250 (all `0`) | `scripts/data/build_xstest.py` |

> **In-the-wild DAN is a mixed corpus:** in this project's `dan/` benchmark, PAIR / TAP /
> GPTFuzzer / DAN-style prompts all appear in the same `test.csv` (no family column). To
> report a family separately, split it out into its own CSV and place it at one of the
> optional dirs, which `BENCHMARK_DIRS` in `data/adapters.py` already accepts: `pair/`,
> `tap/`, `gptfuzzer/`, `universal_injection/` (each needs its own `test.csv` in the schema
> above). Record in provenance that a heuristic split is not the official family set.

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
- map the model's decision to `d_k`, its calibrated block-probability to `s_k`, and an
  embedding of its rationale to `e_k`.

## What is NOT provided
Real datasets, model weights, API keys, and the private/under-construction benchmarks are not
included. See `docs/ec2_experiment_guide.md`, `audits/implementation_gaps.md`, and
`TODO_IMPLEMENTATION.md`.
