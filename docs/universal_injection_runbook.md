# Universal-injection benchmark runbook (`universal_injection`)

Status: **not built**. The `data/benchmarks/universal_injection/` directory does not exist and
`BENCHMARK_DIRS["universal_injection"]` therefore resolves to a missing directory; selecting it
in a config raises a clear `FileNotFoundError`.

This is deliberate. An earlier stand-in was deleted because it was **simulated** (a pasted
target string, `gradient_optimized_suffix: false`), not the mechanism of
liu2024universal (arXiv:2403.04957). Reporting it as universal injection would
misrepresent the experiment. Until the genuine gradient-optimised suffix `S` is produced, the
benchmark is unavailable and the paper's universal-injection row must be reported as not measured.

## What has to be true before this benchmark may be used

1. The suffix `S` was produced by the official optimiser
   (`SheltonLiu-N/Universal-Prompt-Injection`) against a white-box `meta-llama/Llama-2-7b-chat-hf`,
   with the step count and token length stated in the provenance.
2. `data/benchmarks/universal_injection/provenance.json` reads
   `"gradient_optimized_suffix": true`, and cites the optimiser commit and the results JSON.
3. Payload rows follow the standard schema (`id,content,label,group`) with `label=1` for the
   injected rows (task-integrity violation) and `label=0` for the clean rows.

## Procedure

### 1. Optimise on the GPU server (hours; GPU dedicated to this job)

`HF_TOKEN` must belong to an account with approved Llama-2 access. The script refuses to start if
a vLLM judge server is holding the GPU, and refuses if free VRAM is below ~14 GB.

```bash
cd ~/AegisAgency
kill $(cat outputs/vllm-*.pid) 2>/dev/null || true     # free the GPU first
HF_TOKEN=hf_... bash scripts/ec2/run_universal_suffix.sh \
    --injection semi-dynamic --tokens 150 --num-steps 500
```

The script prints the exact path of the newest results JSON **and** an
`aegis_suffix_run.json` run record, plus the ready-to-paste `scp` commands.

* `--num-steps 500` is the default here; the paper uses 1000. Whatever you use, it enters the
  provenance — a shorter run is a weaker attack, not an equivalent one.
* The token is only ever passed through the environment. The script does **not** run
  `huggingface-cli login --add-to-git-credential`, which would persist a credential on what may
  be a borrowed machine.

### 2. Fetch the results to the dev machine

Run on Windows, from the repository root (paths are printed by step 1):

```powershell
mkdir -Force .\upi_results | Out-Null
scp -i "$env:USERPROFILE\.ssh\<key>" <user@host>:"<results json>" .\upi_results\
scp -i "$env:USERPROFILE\.ssh\<key>" <user@host>:"<aegis_suffix_run.json>" .\upi_results\
```

### 3. Rebuild the benchmark CSV

```powershell
python scripts/prepare_universal_injection.py `
    --source D:\Data\downloads\Universal-Prompt-Injection `
    --output data/benchmarks/universal_injection `
    --suffix-file .\upi_results\<results json>
```

`--output` defaults to `D:/Data/benchmarks/universal_injection`; pass the in-repo path explicitly
on this repository layout so the benchmark rides along with `git`/`rsync`.

### 4. Verify the flag before anything else

```powershell
python -c "import json;p=json.load(open('data/benchmarks/universal_injection/provenance.json'));print(p['gradient_optimized_suffix'])"
# must print True
python scripts/check_ready.py --config configs/ec2_real_evaluation_universal.yaml --offline
```

If `gradient_optimized_suffix` is `False`, the builder fell back to the **simulated** stand-in:
delete the directory and do not run the benchmark.

### 5. Run it like any other benchmark

```bash
# on the server
AEGIS_OUT=outputs/real_universal bash scripts/ec2/run_real.sh full
```

Use a per-benchmark `AEGIS_OUT`: every run writes the same file names
(`real_evaluation_summary.csv`, …), so a shared output directory silently overwrites the previous
benchmark's results.

## Reporting requirements

* State the optimiser commit, step count, token length and injection mode in the paper's
  measurement conditions (the run provenance records them under `extra.dataset.sha256` and the
  benchmark's own `provenance.json`).
* `universal_injection` rows are all `label=1`, so this benchmark cannot contribute ORR; measure
  utility on `benign`/`benign_xstest` instead.
* If the suffix could not be produced, report the benchmark as **not measured** rather than
  substituting the simulated stand-in.
