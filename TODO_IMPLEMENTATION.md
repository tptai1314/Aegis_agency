# TODO — Implementation (GPU server / real experiments)

The repository is complete and runnable on synthetic data (104 tests pass; synthetic demo
runs). The items below are what remains to run the **paper-level** experiments; none block the
synthetic mechanism.

> The audit-fix pass (see `NOTES_AUDIT_FIXES.md`) closed the pre-server blockers: silent
> calibration fallback, empty-rationale verdict crashes, context-window aborts, cache reuse
> across changed measurement conditions, and the missing preflight/backup/finalize tooling.

## Data (ships in-repo; only `universal_injection` still needs GPU work)
- [x] Jailbreak benchmarks present under `data/benchmarks/` (AdvBench/GCG, in-the-wild DAN,
      HarmBench) with `PROVENANCE.json` recording source URL + sha256.
- [x] Injection benchmark `formal` (Open-Prompt-Injection, 1400 rows) + `injecagent` (510).
- [x] Held-out benign traffic: `benign` (299) and `benign_xstest` (250) for ORR / utility.
- [x] Second-order payloads `second_order` (JudgeDeceiver LLMBar suffix, 2000 rows).
- [ ] `universal_injection`: needs the GENUINE gradient-optimised suffix
      (`scripts/ec2/run_universal_suffix.sh` on a GPU with Llama-2 access); the simulated
      stand-in was deleted and must not be reported. See `docs/universal_injection_runbook.md`.
- [ ] Optional: split `dan` into PAIR/TAP/GPTFuzzer family CSVs by heuristic (record that the
      split is heuristic, not the official family set).

## Real models / baselines (weights, APIs, GPUs)
- [x] Real judge adapters implemented (`data/real_judges.py`: OpenAI-compatible / Anthropic /
      HF transformers) with payload isolation, verdict cache and cost ledger.
- [ ] Implement `AutoDefenseAdapter.predict()` (real AutoDefense system).
- [ ] Implement `SecAlignAdapter.predict()` and `StruQAdapter.predict()` (checkpoints) — the
      Table 6 row "- hardened judges" and the SecAlign baseline both need these.
- [ ] Implement `JudgeDeceiverAdapter.inject()` (real second-order injection; today `epsilon` is
      measured by replaying a fixed suffix, not by an optimiser).
- [ ] Provide checkpoint paths / endpoints via `ExternalBaselineConfig`; read secrets from env,
      never commit them.

## Empirical quantities to measure (replace synthetic parameters)
- [ ] Honest concentration radius `r` and margin `gamma` from real judges (Assumptions 1-2).
- [ ] Error correlation `rho` for homogeneous vs diverse committees (RQ4).
- [ ] Isolation leakage `epsilon` under JudgeDeceiver, isolation on/off (RQ2).

## Runs and reporting
- [ ] Execute calibrate → evaluate → ablate on real verdicts (`docs/ec2_experiment_guide.md`).
- [ ] Fill the manuscript's placeholder Tables 5-6 and Figure 6 with measured numbers +
      dispersion + significance tests.
- [ ] Update `audits/result_integrity_audit.md` with the provenance of every real result.
- [ ] Use `make_plots.py --real` only for verified real results.

## Optional theory strengthening
- [ ] Re-derive Krum's constant in the verdict space, or keep the cited restatement (currently
      `intentionally_out_of_scope`).
- [ ] Add a Weiszfeld approximation-error term to the Lemma 1 displacement check.

## Author / release
- [ ] Confirm the MIT license placeholder and add author/citation metadata.
- [ ] Confirm the majority-vote tie policy with the paper's authors (documented ambiguity).
