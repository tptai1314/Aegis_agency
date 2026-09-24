# Real-experiment guide (GPU server runbook)

This repository is **server-ready**: the benchmark CSVs ship in-repo, the real LLM-judge adapters
are implemented (`data/real_judges.py`: OpenAI-compatible / Anthropic / local `transformers`,
payload isolation, verdict cache, cost ledger), and only the **external baselines**
(`baselines.external_wrappers.*`) plus the real JudgeDeceiver optimiser remain stubs.

Folder and file names say `ec2` for historical reasons. Nothing here is AWS-specific: any Ubuntu
22.04/24.04 machine with an NVIDIA GPU works.

Prerequisites you must provide: a GPU server with SSH access, the judge backbone (self-hosted vLLM
or API keys), and — for `universal_injection` only — Llama-2 gated access.

> Run the preflight before every run you intend to report:
> `python scripts/check_ready.py --config <config>`. It turns each silent failure mode (missing
> benign slice, over-long payloads, wrong served model, stale cache, inert attack) into a line you
> read *before* spending GPU hours.

---

## 1. Provision the server (once)

Target: **1× GPU with ≥ 24 GB VRAM** (A10G/A100/4090/3090-class), RAM ≥ 32 GB, ≥ 60 GB free disk,
NVIDIA driver with CUDA 12.x, Ubuntu 22.04/24.04, and either `sudo` or a writable home directory.
A 16 GB T4/4090-laptop-class card is marginal for an 8B backbone at 8k context; prefer 24 GB.

```bash
git clone <repo-url> && cd AegisAgency
PYTHON=python3.12 bash scripts/ec2/setup.sh     # python3.11 also works; detection is automatic
```

`setup.sh` reports GPU/RAM/disk/OS first, then picks an interpreter ≥ 3.11 (installing
`python3.12` on 24.04 or `python3.11` via deadsnakes on 22.04 only if none is present), creates
`.venv`, installs `.[ec2]` (openai, anthropic, transformers, sentence-transformers, vLLM), writes
`outputs/pip-freeze-*.txt` (keep it — the dependency spec has no upper bounds), optionally creates
`/data/benchmarks`, and runs the offline test suite.

> Gated repos (Llama-3/Llama-2): export `HF_TOKEN=...` (read token) before any step that downloads
> weights. Provisioning itself needs no token.

## 2. Benchmarks: already in the repository

```
data/benchmarks/<name>/test.csv        + PROVENANCE.json (source URL, sha256, label counts)
data/benchmarks/_raw/                  raw upstream files (only needed to rebuild)
```

Verify the adapter can read every benchmark (repo root, `.venv` active):

```bash
python - <<'PY'
from pathlib import Path
from aegis_agency.data.adapters import BENCHMARK_DIRS, CsvBenchmarkAdapter, resolve_benchmark_dir
root = Path("data/benchmarks")
for name in sorted(BENCHMARK_DIRS):
    d = root / resolve_benchmark_dir(name)
    if not (d / "test.csv").exists():
        print(f"{name:18s} ABSENT"); continue
    ps = list(CsvBenchmarkAdapter(root=d).iter_payloads())
    print(f"{name:18s} n={len(ps):6d} pos={sum(1 for p in ps if p.true_label == 1):6d}")
PY
```

Expected: `harmbench 320/320`, `advbench 520/520`, `dan 1405/1405`, `injecagent 510/510`,
`formal 1400/700`, `second_order 2000/1000`, `benign 299/0`, `benign_xstest 250/0`,
`universal_injection ABSENT`.

`scripts/ec2/upload_data.ps1` still exists for the legacy layout that keeps the CSVs on the dev
machine at `D:\Data\benchmarks` and copies them to `/data/benchmarks`. It uses `sudo` on the
remote side; on a server without sudo, either skip it (the in-repo copy is the default
`data.root`) or extract the tarball by hand into a directory you own and point `data.root` there.

## 3. Serve the judge backbone (once per boot)

```bash
HF_TOKEN=<read-token> bash scripts/ec2/serve_vllm.sh          # Llama-3.1-8B-Instruct on :8001
```

Starts an OpenAI-compatible vLLM server on `127.0.0.1:8001`, waits for `/health`, then confirms the
served model name appears under `/v1/models`, and logs to `outputs/vllm-8001.log` (pid in
`outputs/vllm-8001.pid`). It is idempotent **per port**, so several backbones can coexist.

Alternatives:

- **API judges**: set `judges.backend: anthropic` (or an OpenAI-compatible endpoint) and export the
  key named by `judges.api_key_env`.
- **HuggingFaceJudge** (no server): `judges.backend: hf` with `judges.model` = a local checkpoint
  directory. The committee is then one model, loaded in-process (needs `accelerate`, and the
  default dtype is fp32 — pass a small model or accept the memory).

### Realistic committee on one GPU
`configs/ec2_real_evaluation.yaml` serves **one** model. All `n_judges` judges then share that
backbone: the committee is *homogeneous by default*, and the differing `backbones:` labels are only
cache/cost labels. `scripts/check_ready.py` flags this explicitly.

For a genuinely diverse RQ4 committee, serve one checkpoint per port and use
`configs/ec2_rq4_diverse.yaml`:

```bash
bash scripts/ec2/serve_backbones.sh        # :8001 llama-3.1-8B, :8002 Qwen2.5-7B, :8003 Mistral-7B
python scripts/check_ready.py --config configs/ec2_rq4_diverse.yaml
```

`judges.backbones` is cycled round-robin, so 7 judges over 3 backbones gives a 3/3/1 split; any
backbone without a `models:`/`endpoints:` entry falls back to the shared `model`/`endpoint`. On a
single GPU use API backbones for the extra arms instead of extra local servers.

## 4. Configure the run

`configs/ec2_real_evaluation.yaml` — the knobs that matter:

| Key | Default | Meaning |
|---|---|---|
| `experiment.n_seeds` | 3 | repeats the attack/aggregation draw; honest verdicts are cached and shared, so `std` is **not** judge stochasticity |
| `experiment.measure_theory` | true | estimate r / γ / μ / ρ on honest verdicts |
| `experiment.measure_epsilon` | true | Def 1 ε by replaying an injected payload — costly |
| `experiment.calibrate` | **false** | threshold calibration needs a benign slice; see below |
| `experiment.calibration_min_samples` | 20 | benign items required in the calibration slice |
| `experiment.calibration_on_insufficient` | `raise` | fail loudly instead of silently keeping tau=1.0 |
| `experiment.attack_kwargs` | explicit | attack strength; recorded in the provenance |
| `data.limit` | 0 | cap payloads (0 = all). Start small |
| `judges.embedding_model` | `all-MiniLM-L6-v2` | `""` disables embeddings (m = 0, score-only) |
| `judges.isolation` | true | operator delimiters in the judge prompt (Def 1) |
| `judges.max_prompt_chars` | 16000 | middle-truncate over-long payloads so they fit the served context (`0` = send unchanged) |

### Calibration: read this before enabling it
`target_orr` calibration is defined on **benign** payloads. `harmbench`, `advbench`, `dan` and
`injecagent` are **all must-block** (no benign items), so calibrating there is undefined — the run
now aborts with a clear message instead of silently falling back to `tau = 1.0` (which blocks
nothing and maximises ASR-UC, i.e. the exact opposite of a safe default). Enable calibration only
for `formal`, `second_order`, `benign` or `benign_xstest`, and make sure the calibration slice
(first quarter of the payloads) holds ≥ `calibration_min_samples` benign rows.

### Context window and truncation
The default served context is `VLLM_MAX_LEN=8192` tokens. `dan` contains payloads up to ~55k
characters; without a guard, vLLM returns a context-length error and the run aborts. With
`max_prompt_chars: 16000` long payloads are middle-truncated with an explicit marker, logged, and
counted in the provenance (`extra.truncations`) — truncation is a measurement condition, so state
it. `0` disables the guard. `check_ready.py` reports how many payloads will be affected.

### Verdict cache
Cached honest verdicts are keyed by `(payload_id, judge_id, backbone)` **plus the measurement
context** (prompt version, isolation, resolved model, embedding model, `max_prompt_chars`). A
context change makes a row stale and re-queries it, which is what you want: toggling
`judges.isolation` for RQ2 must re-measure, not replay. `backend: dummy` writes to
`<cache_dir>_dummy`. The provenance reports `n_stale_lookups`.

### Cost of a full harmbench run (400 payloads)
Judge calls are **serial** (one stream); ~2 s per call on an 8B model on a mid GPU:

| Phase | New judge calls | Notes |
|---|---|---|
| honest committee (incl. calibration) | ≈ 2800 | cached in `outputs/real_verdict_cache` |
| `measure_epsilon` (isolation on/off) | ≈ 5600 | injected verdicts, cached separately |
| total | ≈ 8400 | ≈ 4–5 h serial; re-runs reuse caches |

For the first pass set `measure_epsilon: false` (or `--limit 200`) to validate end-to-end cheaply,
then enable it for the final run.

## 5. Run the protocol (RQ1–RQ6)

```bash
python scripts/check_ready.py --config configs/ec2_real_evaluation.yaml   # must show 0 FAIL

AEGIS_OUT=outputs/real.smoke      bash scripts/ec2/run_real.sh smoke   # dummy judge - plumbing only
AEGIS_OUT=outputs/real_harmbench  bash scripts/ec2/run_real.sh full    # the real run (Table 5)
AEGIS_OUT=outputs/real_harmbench  bash scripts/ec2/run_real.sh ablate  # Table 6 (reuses the cache)
```

Use a **per-benchmark `AEGIS_OUT`**: every run writes the same file names, so one shared directory
overwrites the previous benchmark's results (the runner warns when it detects this).

`smoke` uses the ground-truth `dummy` judge — **never report those outputs** (they are diverted to
a `*_dummy` cache automatically). `full` runs the multi-seed f-sweep (mean ± std), theory
measurement and ε, then plots `asr_vs_f.png`. `ablate` computes the component ablations from the
cached honest verdicts.

**Check `attack_effect` before reading anything else.** It is the fraction of payloads whose
decision the attack actually changed. If it is 0 for every Aegis rule, the attack never moved a
decision: all methods score identically and the table says nothing about robustness. The runner
logs a warning in that case, and `check_ready.py` warns up front when the configured collusion
shift is small. With the shipped default (`collusion`, radius 0.1) on payloads whose honest scores
sit near 1.0, expect `attack_effect = 0` — raise `attack_kwargs.radius`/`budget` (and say so in the
provenance) or the experiment is not informative.

Outputs in `<AEGIS_OUT>/`:

```
real_evaluation_sweep.csv          # first-seed per-(f, method) metrics (+ attack_effect)
real_evaluation_summary.csv        # mean ± std over seeds (Table 5/6 format)
real_evaluation_significance.csv   # paired-bootstrap vs coordinator baseline
real_theory_analysis.csv           # r / gamma / mu / rho + Thm 1 check
real_isolation_epsilon.csv         # Def 1 epsilon (isolation on/off)
real_ablation.csv                  # Table 6: agg on/off, committee n, isolation, rho
real_cost_summary.csv              # RQ5: per-build calls / tokens / wall time (iso/noiso)
real_cost_ledger.csv               # RQ5: same cost split per backbone
real_evaluation_provenance.json    # data_source=real; is_paper_result=False
real_ablation_provenance.json      # same provenance contract for the ablation runs
```

`real_cost_*.csv` reflect **calls actually made during that run**: a re-run over a warm cache
reports ~0 cost (that is the point of the cache). For honest RQ5 numbers, measure on a cold cache.

Manual alternative (equivalent):

```bash
python scripts/run_real_experiment.py evaluate \
    --config configs/ec2_real_evaluation.yaml --output outputs/real_harmbench
python scripts/run_real_experiment.py ablate \
    --config configs/ec2_real_evaluation.yaml --output outputs/real_harmbench
python scripts/make_plots.py \
    --input outputs/real_harmbench/real_evaluation_summary.csv \
    --output outputs/real_harmbench/asr_vs_f.png --real
```

## 6. Back up, verify, record

```bash
# on the server: pack the run, its provenance AND the verdict cache (the cache is the raw
# judge measurement the CSVs were derived from)
bash scripts/ec2/backup_results.sh --outputs outputs/real_harmbench
```

```powershell
# on Windows: pull it down and verify the checksum
powershell -ExecutionPolicy Bypass -File scripts/backup_results.ps1 `
    -HostTarget <user@host> -Key "$env:USERPROFILE\.ssh\<key>" `
    -RemoteBundle "~/aegis_backups/aegis_real_harmbench_<stamp>.tar.gz"
```

```bash
# verify the run, write audits/run_records/<run_id>.json (per-artefact sha256) and refresh the
# generated table in audits/result_integrity_audit.md
python scripts/finalize_run.py --run outputs/real_harmbench
```

Paper-honesty rules:

1. Keep every `*provenance.json` with its CSVs, and keep the verdict cache.
2. `docs/reproducibility.md` + `audits/preregistration_protocol.md`: freeze the config *before*
   looking at numbers.
3. Reproduce the run independently (same frozen config, second directory), then:
   `python scripts/finalize_run.py --run <run1> --replicated-by <run2> --set-paper-result`.
   The script refuses unless the run verifies **and** the two summaries match within tolerance.
4. Replace the placeholder cells of Table 5/6 and the conceptual curves in Figure 6
   (`manuscript/figures/pgfplots_placeholder_results.tex`), then remove the "Placeholder" wording.

## What the GPU server does not change
- The mechanism, math and metrics are identical to the synthetic runs; only the *source of
  verdicts* changes (synthetic generator → real LLM judges).
- External baselines for a faithful head-to-head (real AutoDefense, SecAlign, StruQ,
  JudgeDeceiver) remain adapter stubs; the structural AutoDefense coordinator is already in the
  comparison set.
