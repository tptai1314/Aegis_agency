# Reproducibility

## Seeds
All stochastic code draws from an explicit `numpy.random.Generator` created via
`aegis_agency.utils.seeding.get_rng(seed)`. Every config carries a `seed`; the same seed and
config reproduce identical trials (see `tests/test_reproducibility.py`). For whole-process
determinism (third-party libraries), call `set_global_seed(seed)`.

## Config capture
Each runner writes a `*provenance.json` (see `utils/provenance.py`) recording:
`run_id`, `stage`, `seed`, `config`, `data_source` (`synthetic`/`real`), `is_paper_result`
(always `False` for synthetic runs), package version, Python version, and platform — plus
the reproduction metadata an auditor needs:
`git_commit` (code at run time), `command` (exact CLI invocation), `deps` (numpy/torch/
openai/… versions), `gpu` (device + CUDA, when torch is present), `hostname`, and
`created_utc`. Secrets (API keys, tokens) are never captured — including the *names* of the
credential variables (`test_real_run.py::TestConfigParsingGuards` enforces this).

For a real run the `config` snapshot must be complete enough to re-run it, so it carries
`attack` **and** `attack_kwargs`, `calibrate`/`target_orr`/`calibration_min_samples`/
`calibration_on_insufficient`, `escalate_band`, `threshold`, `limit`, `seed`/`n_seeds`,
`isolation`, `embedding_model`, `max_prompt_chars` and `cache_dir`. `extra` adds the evidence
that is not a config value:

- `dataset`: path, **sha256**, split, rows used and the label split — identifies the exact bytes;
- `truncations`: how many payloads the context-window guard shortened, and by how much;
- `cache`: directory, record count, stale-lookup count and the measurement context;
- `calibration`: calibration slice sizes and the resulting thresholds;
- `seed_scope`: `"attack_and_aggregation_only; honest verdicts are cached"`.

> **`n_seeds` does not resample the judges.** Honest verdicts are cached and shared across seeds
> (real judges decode greedily at temperature 0), so the reported `std` measures attack
> randomness only. Say so when reporting dispersion.

Before the first real run, fill `audits/preregistration_protocol.md` (frozen config, sanity
checks, exact commands, amendment log); a run becomes a paper result only after its numbers
reproduce independently and `is_paper_result` is set true with the audit updated.

## Result provenance
- Tables/plots are always generated **from result files**, never from hard-coded numbers.
- Synthetic plots carry a visible "synthetic smoke-test — NOT a paper result" banner.
- `tests/test_reproducibility.py::test_no_committed_result_files_claim_paper_results` fails
  the suite if any shipped provenance file claims `is_paper_result: true`.

## Environment
- Python >= 3.11.
- Runtime deps: `numpy`, `pyyaml`, `matplotlib` (see `requirements.txt`).
- Dev deps: `pytest`, `ruff`, `mypy`, `pandas` (`pip install -e ".[dev]"`).
- No GPU, no network access required for tests or the synthetic demo.

## Commands
```bash
pip install -e ".[dev]"
pytest -q                 # 104 tests, synthetic only
ruff check .              # lint
mypy src                  # types
make demo                 # synthetic smoke-test demo -> outputs/synthetic_demo/
```

## Real-run tooling
```bash
python scripts/check_ready.py --config configs/ec2_real_evaluation.yaml   # preflight (was manual)
bash   scripts/ec2/backup_results.sh --outputs outputs/real_harmbench     # pack run + cache (server)
powershell -ExecutionPolicy Bypass -File scripts/backup_results.ps1 ...   # pull + verify sha256
python scripts/finalize_run.py --run outputs/real_harmbench               # verify + run record
python scripts/finalize_run.py --run outputs/real_harmbench \
       --replicated-by outputs/real_harmbench_rep2 --set-paper-result      # after replication
```
`finalize_run.py` refuses `--set-paper-result` unless the run verifies *and* its summary matches
an independent replication, and it writes the audit table between the `BEGIN AUTO: RUN RECORDS`
markers in `audits/result_integrity_audit.md` (edit the prose outside the markers by hand).

## Cache semantics (why a re-run may cost more than you expect)
Honest verdicts are cached per `(payload_id, judge_id, backbone)` **plus a measurement context**:
prompt-template version, isolation setting, resolved model, embedding model and
`max_prompt_chars`. A row whose context differs is treated as stale and re-queried, so
- toggling `judges.isolation` re-measures the committee (it must: RQ2 compares the two prompts);
- editing the judge prompt requires bumping `JUDGE_PROMPT_VERSION` in `data/real_judges.py`;
- `backend: dummy` writes to `<cache_dir>_dummy` and can never pollute a real run's cache.
The run provenance reports `n_stale_lookups`, which is the honest way to state how much of a
"cheap re-run" actually reused measurements.

## Determinism caveats
- Floating-point reductions (e.g. geometric-median Weiszfeld) are deterministic for a fixed
  input but may differ at the last ULP across BLAS builds; comparisons in tests use
  tolerances.
- `matplotlib` is used in `Agg` (headless) mode; the first run builds a font cache.
