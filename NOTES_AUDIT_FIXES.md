# Notes: pre-server audit fixes

This file documents the changes made in response to the read-only pre-server audit, so a reviewer
can see exactly what changed, why, and how each change is guarded. It is the reference for the
`TestEmbeddingDimensionGuard` / `TestConfigParsingGuards` / `TestCalibrationGuard` /
`TestAttackEffectDiagnostic` / `TestContextWindowGuard` / `TestCacheContextGuard` tests in
`tests/test_real_run.py`.

Nothing here changes the paper's mechanism, aggregation, attacks or metrics. The science-affecting
choices are marked **[policy]** and are now explicit + recorded instead of implicit.

## Blockers fixed

### A — a terse/refusing judge aborted the whole run
**Symptom.** `EmbeddingExtractor.embed("")` returned `zeros(0)`. A judge that refused to answer, or
returned JSON without a `rationale`, produced a 0-dim verdict vector while its peers produced
384-dim ones. `stack_verdicts` rejects mixed dimensions, so one terse response raised
`ValueError: Verdict vectors have inconsistent dimensions: [1, 385]` and killed the run — despite
the refusal path being an intentional `BLOCK` fallback.

**Fix.** `EmbeddingExtractor.embed` now resolves the encoder dimension and returns `zeros(m)` for an
empty rationale (and still `zeros(0)` when embeddings are disabled). Guards:
`TestEmbeddingDimensionGuard`.

### B — `judges.embedding_model: ""` was silently ignored
**Symptom.** `parse_real_config` dropped every empty string, so the documented way to disable
rationale embeddings (`m = 0`, score-only verdicts) silently reverted to `all-MiniLM-L6-v2` — the
opposite of the documented behaviour, with an extra model download.

**Fix.** Only `None` is dropped; `""` survives for the keys where it is a real setting
(`embedding_model`, `injected_suffix`), and every other empty string still falls back to the
default. Guard: `TestConfigParsingGuards`.

### C — calibration silently produced `tau = 1.0` (block nothing)
**Symptom.** `harmbench`, `advbench`, `dan`, `injecagent` are all must-block. `target_orr`
calibration has no benign items, so `calibrate_threshold` returned `tau = 1.0`, i.e. the gate
blocked nothing and ASR-UC was ~1.0 for every method. The run "succeeded" and produced a table that
said nothing. **[policy]**

**Fix.** `calibrate_threshold(..., on_insufficient="raise"|"conservative")` now **raises by default**
with an actionable message naming the class, the counts and the options;
`conservative` preserves the legacy fallback as an explicit opt-in. `RealRunConfig` exposes
`calibration_min_samples` (default 20) and `calibration_on_insufficient` (default `raise`), both
recorded in the provenance; a failed calibration is re-raised with the calibration-slice label
counts attached. The synthetic harness keeps `conservative` by default (its scores are
class-separated by construction and it is not a reported measurement) — that asymmetry is
deliberate and documented in `TrialConfig`. `configs/ec2_real_evaluation.yaml` ships
`calibrate: false` with the reason inline. Guards: `TestCalibrationGuard`,
`tests/test_calibration_gate.py`.

### D — an inert attack made every defence look identical
**Symptom.** Collusion moves the Byzantine judges to `honest_mean − budget·radius` = 0.1 below the
mean. Real honest scores sit near 1.0, so nothing crossed the threshold; the local pilot reported
`asr_uc = 0.0` for every method at every `f`, which reads like "all defences are equally robust".

**Fix.** Every sweep/summary row now carries **`attack_effect`**: the fraction of payloads whose
gate decision the attack actually changed versus the untampered honest committee (measured with a
separate RNG so the reported random stream is unchanged). The runner logs a warning when it is 0
for every Aegis rule at `f ≥ 1`. `attack_kwargs` is explicit in the config, the *effective* kwargs
(including filled-in defaults) are recorded in the provenance, and `check_ready.py` warns up front
when the collusion shift looks too small. Guard: `TestAttackEffectDiagnostic`.

### E — provisioning failed on a fresh Ubuntu box
**Symptom.** `setup.sh` ran `apt-get install python3.11`, which does not exist in the default
Ubuntu 22.04 (3.10) or 24.04 (3.12) archives, so the very first step failed.

**Fix.** `setup.sh` detects an interpreter ≥ 3.11 (`PYTHON` override, then `python3.12`,
`python3.13`, `python3.11`, `python3`), installs `python3.12` on 24.04 or `python3.11` via
deadsnakes on 22.04 only when none is present, installs the matching `-venv` package, reports the
host (GPU/RAM/disk/OS) first, makes `/data/benchmarks` optional (passwordless sudo only), and writes
`outputs/pip-freeze-*.txt` because the dependency spec has no upper bounds.

### F — one over-long payload aborted the run
**Symptom.** `dan` contains payloads up to 55,089 characters; the served context is
`VLLM_MAX_LEN=8192`. The judge passed the payload through unchanged, vLLM returned a
context-length error and the run died (no truncation, no retry, no skip).

**Fix.** `judges.max_prompt_chars` (0 = off) middle-truncates a payload with an explicit marker,
keeping both ends; truncation is logged once, counted in a `TruncationCounter`, and recorded in the
provenance (`extra.truncations`). Prompts are byte-identical when nothing is truncated. Judge-call
errors now name the payload, its length and the model, and point at `max_prompt_chars`. The shipped
config sets `16000`. Guards: `TestContextWindowGuard`.

### G — the verdict cache ignored the measurement conditions
**Symptom.** Cached honest verdicts were keyed only by `(payload_id, judge_id, backbone)`, and
`run_real.py` forced one cache file name for both isolation settings. Toggling `judges.isolation`
for the RQ2 comparison therefore replayed isolation-on verdicts as if they had been measured with
isolation off. Changing the judge prompt, the resolved model or the embedding model had the same
problem.

**Fix.** Every cache record stores an `embedding_dim` and a `context` fingerprint
(`prompt_version`, `isolation`, `backend`, resolved `model`, `embedding_model`,
`max_prompt_chars`); a lookup whose context differs is a miss and is counted as
`n_stale_lookups` in the provenance. Records without a context (older layout) are stale by default,
so the worst case is extra judge calls, never a silently wrong number. The cache file name is
tagged by isolation again (`<benchmark>_{iso,noiso}_honest.jsonl`), and `JUDGE_PROMPT_VERSION`
must be bumped when the prompt changes. Guard: `TestCacheContextGuard`.

## Additional hardening (found while fixing the above)

- **Provenance completeness** (reproducibility): `config` now carries `attack_kwargs` (effective),
  `calibrate`/`target_orr`/`calibration_min_samples`/`calibration_on_insufficient`,
  `escalate_band`, `limit`, `isolation`, `embedding_model`, `max_prompt_chars`, `cache_dir` and the
  `f_values`; a new `extra` block carries the dataset fingerprint (**sha256**, rows, label split),
  truncation counts, cache state/context, calibration outcome and `seed_scope`. `RunProvenance`
  gained `hostname` and `created_utc`. Credential variable *names* stay out of the file, enforced by
  test.
- **`n_seeds` semantics**: honest verdicts are cached and shared, so `n_seeds` repeats only the
  attack/aggregation draw. The runner warns at start and records `seed_scope`; docs no longer imply
  the std is judge stochasticity.
- **`backend: dummy` cache isolation**: dummy (ground-truth) verdicts now go to `<cache_dir>_dummy`,
  so a smoke run cannot pollute the cache a real run reuses.
- **Output-directory collision**: the runner warns when the output directory already holds a
  different benchmark's provenance (all runs write the same file names).
- **`api_key_env` is wired** (it was documented but silently ignored as an unknown key), and
  unknown config keys are now reported instead of dropped in silence.
- **`serve_vllm.sh`** guards per *port* (not globally), uses per-port log/pid files
  (`outputs/vllm-<port>.log|pid`), and confirms the served model name appears under `/v1/models`
  so a `judges.model` mismatch is caught at startup.
- **Logging robustness**: the stderr handler resolves `sys.stderr` at emit time, so log lines are
  not lost (with "I/O operation on closed file") under test capture, and the NullHandler is added
  once instead of on every `get_logger` call.

## New tooling

| Script | Purpose |
|---|---|
| `scripts/check_ready.py` | Preflight: dataset/label/length facts, calibration feasibility, attack strength, served-model probe, cache state, disk, git state. Exit 1 on FAIL. |
| `scripts/finalize_run.py` | Verify a finished run, hash every artefact into `audits/run_records/<run_id>.json`, refresh the generated table in `audits/result_integrity_audit.md`, and refuse `--set-paper-result` unless the run verifies **and** matches an independent replication. |
| `scripts/ec2/backup_results.sh` | Pack results + provenance + verdict cache + manifest (sha256) on the server. |
| `scripts/backup_results.ps1` | Pull that bundle to the dev machine and verify the checksum. |
| `scripts/ec2/serve_backbones.sh` | Bring up several backbones (one per GPU/port) for the RQ4 diverse committee. |
| `configs/ec2_rq4_diverse.yaml` | Diverse-committee config with `models`/`endpoints` maps. |
| `docs/universal_injection_runbook.md` | How the genuine universal suffix is produced, fetched and verified before the benchmark may be used. |

## Explicitly NOT changed (still open, and now documented as such)

- `temperature_scale` is still not applied to judge scores (the paper's "temperature-calibrated
  scores" is not implemented end to end).
- The `injection` attack remains a verdict-space RNG simulation; it does not read the payload and is
  independent of `judges.isolation`. The measured leak is `real_isolation_epsilon.csv`.
- Real AutoDefense / SecAlign / StruQ / JudgeDeceiver remain stubs.
- `universal_injection` remains unbuilt.
- Krum's constant is still not re-derived; the Weiszfeld tolerance is still a tolerance.
- Attack strength (`attack_kwargs`) and the calibration policy are now *visible and recorded*, not
  tuned: choosing values that produce a non-inert attack is an experimental-design decision for the
  authors, and `attack_effect` exists so that decision is made on evidence.
