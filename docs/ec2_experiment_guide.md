# EC2 Experiment Guide — real-data runbook

This repository is **EC2-ready**: it never downloads datasets, weights, or credentials. The
real LLM-judge adapters are implemented (`data/real_judges.py`: OpenAI-compatible / Anthropic
/ local `transformers` judges, isolation prompt, verdict cache, embeddings); only the
**external baselines** (`baselines.external_wrappers.*`) remain stubs. This page is the
concrete runbook to produce the Table 5/6 numbers.

Prerequisites you must provide yourself: an EC2 GPU instance, SSH access, the benchmark CSVs
(prepared locally under `D:\Data\benchmarks`), and the judge backbone (self-hosted vLLM, or
API keys).

---

## 1. Provision the server (once)

Recommended: a GPU instance (e.g. `g5.xlarge` / `g4dn.2xlarge`+, NVIDIA driver preinstalled,
CUDA 12.x, ≥ 16 GB VRAM, ≥ 50 GB disk), Ubuntu 22.04/24.04, SSH key inbound.

Clone the repo on the server and run:

```bash
bash scripts/ec2/setup.sh
```

This installs the system Python, creates `.venv`, installs the package with `.[ec2]` (openai,
anthropic, transformers, sentence-transformers, vLLM), creates `/data/benchmarks`, and runs
the synthetic test suite as a sanity check.

> Gated repos (Llama-3): export `HF_TOKEN=...` (Hugging Face read token) before any step that
> downloads weights. Setup itself needs no token.

## 2. Upload the benchmarks (once)

From your **Windows dev machine** (where the CSVs already live at `D:\Data\benchmarks`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ec2/upload_data.ps1 `
    -HostTarget ubuntu@<ec2-host> -Key "$env:USERPROFILE\.ssh\aegis.pem"
```

This packs `D:\Data\benchmarks` into one tar, `scp`s it, and extracts to `/data/benchmarks`
on the server. Expected tree (matches `data.root` + `resolve_benchmark_dir`):

```
/data/benchmarks/harmbench/test.csv      advbench/  dan/  formal/  injecagent/
/data/benchmarks/benign/test.csv         second_order/
```

Verify the adapter can read them (on the server, repo root, `.venv` active):

```bash
python - <<'PY'
from pathlib import Path
from aegis_agency.data.adapters import CsvBenchmarkAdapter, resolve_benchmark_dir
for name in ("harmbench", "benign", "injecagent", "dan", "formal", "advbench", "second_order"):
    ps = list(CsvBenchmarkAdapter(root=Path("/data/benchmarks") / resolve_benchmark_dir(name)).iter_payloads())
    print(f"{name:14s} n={len(ps):6d} pos={sum(1 for p in ps if p.true_label==1):6d}")
PY
```

## 3. Serve the judge backbone (once per boot)

```bash
HF_TOKEN=<read-token> bash scripts/ec2/serve_vllm.sh      # if using gated Llama-3
```

Starts an OpenAI-compatible vLLM server on `127.0.0.1:8001`, waits for `/health`, logs to
`outputs/vllm.log` (pid in `outputs/vllm.pid`). It is idempotent (skips if already running).
Alternatives:

- **API judges** instead of vLLM: set `judges.backend: openai`-style config accordingly, or
  `anthropic`, and export the API key that `judges.api_key_env` names.
- **HuggingFaceJudge** (no server): set `judges.backend: hf` and
  `judges.model` to a local checkpoint directory. Homogeneous committee of one model.

### Realistic committee on one GPU
`configs/ec2_real_evaluation.yaml` serves **one** model (`judges.model`). All `n_judges`
judges therefore share that backbone — the committee is *homogeneous by default*, which is
fine for a first Table 5/6 run. For a genuinely diverse RQ4 committee, serve one checkpoint
per port and map them in the config:

```yaml
judges:
  models:
    llama-3: meta-llama/Llama-3.1-8B-Instruct
    qwen2.5: Qwen/Qwen2.5-7B-Instruct
  endpoints:
    llama-3: http://127.0.0.1:8001/v1
    qwen2.5: http://127.0.0.1:8002/v1
```

`judges.backbones` lists the intended RQ4 diversity; any backbone not in the maps falls back
to the shared `model`/`endpoint`. The honest verdict cache is keyed by `(payload_id,
judge_id, backbone)`, so changing the mapping invalidates only that backbone's cached rows.

## 4. Configure the run

`configs/ec2_real_evaluation.yaml` — the knobs that matter:

| Key | Default | Meaning |
|---|---|---|
| `experiment.n_seeds` | 3 | sweep repeats; summary reports mean ± std |
| `experiment.measure_theory` | true | estimate r / γ / μ / ρ on honest verdicts |
| `experiment.measure_epsilon` | true | Def 1 ε by replaying an injected payload — costly |
| `data.limit` | 0 | cap payloads (0 = all). Start small |
| `judges.embedding_model` | `all-MiniLM-L6-v2` | `""` disables embeddings (m = 0) |
| `judges.isolation` | true | operator delimiters in the judge prompt (Def 1) |

First run downloads the embedding model (~90 MB) — internet or a pre-warmed
`~/.cache/huggingface` is required.

### Cost of a full harmbench run (400 payloads)
Judge calls are **serial** (one stream); ~2 s per call on an 8B model on a mid GPU:

| Phase | New judge calls | Notes |
|---|---|---|
| honest committee (incl. calibration) | 400 × 7 ≈ 2800 | cached in `outputs/real_verdict_cache` |
| `measure_epsilon` (isolation on/off) | ≈ 2 × 400 × 7 ≈ 5600 | injected verdicts, cached separately |
| total | ≈ 8400 | ≈ 4–5 h serial; re-runs reuse caches |

For the first pass on EC2 set `measure_epsilon: false` (or `--limit 200`) to validate end-to-
end cheaply, then enable it for the final run.

## 5. Run the protocol (RQ1–RQ6)

```bash
bash scripts/ec2/run_real.sh smoke     # dummy judge, 40 payloads — plumbing only
bash scripts/ec2/run_real.sh full      # the real run + asr_vs_f.png
bash scripts/ec2/run_real.sh ablate    # Table 6 cells (reuses the honest cache)
```

`smoke` uses the ground-truth `dummy` judge — **never report those outputs**. `full` runs the
multi-seed f-sweep (mean ± std), theory measurement, and ε, then plots `asr_vs_f.png`.
`ablate` computes the component ablations on the cached honest verdicts — it makes **no** new
honest judge calls (and no attack calls); only the isolation cells replay the injected payload,
which the ε pass already cached. Run it after `full`.

Outputs in `outputs/real/`:

```
real_evaluation_sweep.csv          # first-seed per-(f, method) metrics
real_evaluation_summary.csv        # mean ± std over seeds (Table 5/6 format)
real_evaluation_significance.csv   # paired-bootstrap vs coordinator baseline
real_theory_analysis.csv           # r / gamma / mu / rho + Thm 1 check
real_isolation_epsilon.csv         # Def 1 epsilon (isolation on/off)
real_ablation.csv                  # Table 6: agg on/off, committee n, isolation, ρ(measured)
real_cost_summary.csv              # RQ5: per-build calls / tokens / wall time (iso/noiso)
real_cost_ledger.csv               # RQ5: same cost split per backbone
real_evaluation_provenance.json    # data_source=real; is_paper_result=False
real_ablation_provenance.json      # same provenance contract for the ablation runs
```

`real_cost_summary.csv` / `real_cost_ledger.csv` reflect **calls actually made during that
run**: a re-run over a warm `outputs/real_verdict_cache` reports ~0 cost (the whole point of
the cache). For honest RQ5 cost/latency numbers, measure on a cold cache — clear
`outputs/real_verdict_cache` (or use a fresh `benchmark` name) for the run you report.

Manual alternative (equivalent):

```bash
python scripts/run_real_experiment.py evaluate \
    --config configs/ec2_real_evaluation.yaml --output outputs/real
python scripts/run_real_experiment.py ablate \
    --config configs/ec2_real_evaluation.yaml --output outputs/real
python scripts/make_plots.py \
    --input outputs/real/real_evaluation_summary.csv \
    --output outputs/real/asr_vs_f.png --real
```

## 6. Records (paper honesty)

1. Keep every `*provenance.json` with its CSVs.
2. Edit `audits/result_integrity_audit.md` with: which runs are real, data provenance, exact
   commands + hashes, and which numbers enter the paper.
3. Set `is_paper_result: true` in provenance **only** after the verification checklist is
   applied and the run is reproduced — see `docs/reproducibility.md`.
4. Fill the `--` cells of Table 5/6 and replace the conceptual curves in Figure 6
   (`pgfplots_placeholder_results.tex`), then remove the "Placeholder" wording.

## What EC2 does not change
- The mechanism, math, and metrics are identical to the synthetic runs; only the *source of
  verdicts* changes (synthetic generator → real LLM judges).
- External baselines for a faithful head-to-head (real AutoDefense, SecAlign, StruQ,
  JudgeDeceiver) remain adapter stubs; the structural AutoDefense coordinator is already in
  the comparison set.