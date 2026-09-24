# Checklist EC2 — chạy real-data experiment (Aegis-Agency)

Checklist thực thi, tiếng Việt. Đi từng bước từ trái qua phải, đánh dấu `[x]` khi hoàn thành.

## 0. Trạng thái hiện tại (đã biết)

- [x] Benchmarks đã có ở máy Windows: `D:\Data\benchmarks` (8 thư mục: advbench, benign, benign_xstest, dan, formal, harmbench, injecagent, second_order — formal 1400 / second_order 2000 / benign_xstest 250, `PROVENANCE.json` ghi nguồn; `universal_injection` **chưa có** — chờ sinh S thật trên GPU, mục 6)
- [x] Pilot local đã chạy bằng Ollama (llama3.2:3b) — chưa đủ cho paper, không dùng làm kết quả
- [ ] SSH key `~/.ssh/aegis.pem` — **chưa có** (chỉ có `known_hosts`)

## 1. Chuẩn bị trước khi thuê (máy Windows)

- [ ] Tạo key pair trên AWS Console → tải `aegis.pem` → đặt vào `~/.ssh/aegis.pem`
- [ ] (Windows) Phân quyền: `icacls "$env:USERPROFILE\.ssh\aegis.pem" /inheritance:r /grant:r "$env:USERNAME:R"`
- [ ] Chuẩn bị `HF_TOKEN` (Hugging Face read token) nếu dùng `meta-llama/Llama-3.1-8B-Instruct` (repo gated)

## 2. Thuê instance EC2

- [ ] Instance: **`g5.xlarge`** (1×A10G 24 GB VRAM, 4 vCPU, 16 GB RAM) — đủ cho full run Table 5/6
    - `g4dn.2xlarge` (T4 **16 GB**) chỉ ở mức **chớm đủ** cho 1 model 8B: KV cache + embedding model
      dễ OOM, và payload dài dễ vượt context → khuyến nghị **≥ 24 GB VRAM**
    - RAM khuyến nghị **≥ 32 GB** (nhiều process vLLM + 7 sentence-encoder instance)
    - Muốn RQ4 đa backbone ngay: `g5.12xlarge` (4×A10G 96 GB VRAM)
- [ ] AMI: Ubuntu 22.04/24.04 (driver NVIDIA + CUDA 12.x sẵn)
- [ ] Disk: root ≥ **60 GB** (weights + vLLM cache + benchmarks + verdict cache)
- [ ] Security group: mở **SSH inbound (TCP 22)** từ IP của bạn. Các port vLLM (8001+) chỉ bound localhost nên không cần mở public

## 3. Provision server (một lần)

- [ ] SSH vào: `ssh -i ~/.ssh/aegis.pem ubuntu@<ec2-host>`
- [ ] Clone repo: `git clone <repo-url>` rồi `cd AegisAgency` (dataset **đã có sẵn trong repo**)
- [ ] Chạy: `PYTHON=python3.12 bash scripts/ec2/setup.sh`
      (script tự dò Python ≥ 3.11; chỉ cài `python3.12`/deadsnakes khi máy chưa có; tạo `.venv`,
      `pip install -e ".[ec2]"`, ghi `outputs/pip-freeze-*.txt`, chạy sanity test **104 tests**)

## 4. Upload benchmarks — **KHÔNG cần** (dataset nằm trong repo)

- [ ] Bỏ qua bước này nếu dùng `data.root: data/benchmarks` (mặc định của config).
- [ ] Chỉ khi bạn muốn dùng `/data/benchmarks`: chạy `scripts/ec2/upload_data.ps1` từ máy Windows
      (script dùng `sudo` phía server; không có sudo thì giải nén tarball vào thư mục bạn sở hữu
      rồi sửa `data.root`).
- [ ] Verify adapter đọc được mọi benchmark: chạy snippet trong `docs/ec2_experiment_guide.md` §2
      (kỳ vọng: harmbench 320/320, dan 1405/1405, formal 1400/700, second_order 2000/1000,
      benign 299/0, benign_xstest 250/0, `universal_injection ABSENT`).

## 6. Sinh S gốc cho universal_injection (GPU, một lần)

CSV `universal_injection` hiện là **stand-in simulated** (target-string dán đại). Muốn payload đúng cơ chế liu2024universal (gradient-optimized suffix **S** thật) — chạy optimizer trên GPU rồi rebuild ở máy Windows:

- [ ] Cần HF account **đã được duyệt Llama-2** (`meta-llama/Llama-2-7b-chat-hf` là repo gated) + `HF_TOKEN`
- [ ] Trên server (repo root):
    ```bash
    HF_TOKEN=hf_... bash scripts/ec2/run_universal_suffix.sh \
        --injection semi-dynamic --tokens 150 --num-steps 500
    ```
    (default 500 steps ≈ vài giờ trên A10G; paper dùng 1000 — tăng nếu muốn sát ASR paper)
- [ ] Copy kết quả JSON về máy Windows:
    ```powershell
    scp -i ~\.ssh\aegis.pem ubuntu@<host>:"~/Universal-Prompt-Injection/results/eval/llama2/semi-dynamic/momentum_1.0/token_length_150/target_0/0_5_20_normal.json" .\
    ```
- [ ] Rebuild CSV đúng cơ chế (S thật, payload = `input + S`):
    ```powershell
    python scripts/prepare_universal_injection.py --source C:\...\Universal-Prompt-Injection `
        --output D:\Data\benchmarks\universal_injection --suffix-file .\0_5_20_normal.json
    ```
- [ ] Upload lại: `scripts/ec2/upload_data.ps1` (chỉ riêng universal_injection đã đổi)
- [ ] Lưu `results/*.json` + `provenance.json` làm bằng chứng (nằm trong nhóm "run nào thật" ở mục 11)

## 5. Serve judge backbone

- [ ] Chạy (trên server, repo root):
    ```bash
    HF_TOKEN=<read-token> bash scripts/ec2/serve_vllm.sh   # Llama-3.1-8B-Instruct @ :8001
    ```
- [ ] Kiểm tra health: `curl http://127.0.0.1:8001/health`
- [ ] Lần đầu sẽ tải thêm embedding model `all-MiniLM-L6-v2` (~90 MB) — cần internet
- [ ] *(RQ4)* Serve thêm backbone khác mỗi cái một port (xem mục 10)

## 8. Validate end-to-end giá rẻ (bắt buộc trước full run)

- [ ] `python scripts/check_ready.py --config configs/ec2_real_evaluation.yaml` → **0 FAIL**
      (script báo trước: payload quá dài, calibration không khả thi, endpoint sai model, cache cũ,
      attack quá yếu)
- [ ] Sửa `configs/ec2_real_evaluation.yaml`: `limit: 200`, `n_seeds: 1`, `measure_epsilon: false`
- [ ] **`calibrate`**: giữ `false` cho harmbench/advbench/dan/injecagent (toàn label 1 → không có
      benign để calibrate `target_orr`; bật lên sẽ **abort** chứ không còn âm thầm trả `tau=1.0`)
- [ ] `AEGIS_OUT=outputs/real.smoke bash scripts/ec2/run_real.sh smoke` — dummy judge, 40 payloads,
      chỉ kiểm tra plumbing (KHÔNG dùng số liệu này; verdict dummy được ghi vào `*_dummy` riêng)
- [ ] `AEGIS_OUT=outputs/real_trial bash scripts/ec2/run_real.sh full` — chạy thử rẻ hơn để xác nhận
      pipe, cache, outputs
- [ ] Kiểm tra outputs thực sự xuất hiện: `real_evaluation_summary.csv`, `*_provenance.json`

## 9. Full run (Table 5/6)

- [ ] Sửa lại config: `limit: 0`, `n_seeds: 3`, `measure_epsilon: true`, `calibrate: false`
      (chỉ bật `calibrate: true` khi benchmark có benign slice và slice đủ
      `calibration_min_samples=20` benign)
    - Đảm bảo cache verdict rỗng với run sẽ báo cáo (cost/thời gian đo trên cold cache).
      Lưu ý: cache được key theo *measurement context* (prompt version, isolation, model,
      embedding model, `max_prompt_chars`) — đổi bất kỳ mục nào sẽ **đo lại**, không reuse.
- [ ] `AEGIS_OUT=outputs/real_harmbench bash scripts/ec2/run_real.sh full` (≈8400 judge calls ≈ 4–5 h)
- [ ] **Đọc `attack_effect` trong `real_evaluation_summary.csv` trước tiên.** Nếu = 0 ở mọi rule
      Aegis → attack không đổi quyết định nào, mọi method giống nhau, bảng vô nghĩa → tăng
      `attack_kwargs.radius/budget` rồi chạy lại (và ghi lại trong provenance).
- [ ] `AEGIS_OUT=outputs/real_harmbench bash scripts/ec2/run_real.sh ablate` — Table 6 (reuse cache)
- [ ] Chạy nhiều benchmark nếu cần: `advbench`, `injecagent`, `dan`, `formal_injection`,
      `second_order` (đổi `data.benchmark`; **mỗi benchmark một `AEGIS_OUT` riêng**, nếu không sẽ
      ghi đè kết quả của nhau). Lưu ý key đúng là `formal_injection`, không phải `formal`;
      `dan` có payload tới ~55k ký tự nên cần `judges.max_prompt_chars` (mặc định 16000).

## 10. RQ4 — diverse committee (sau khi có Table 5/6)

- [ ] Chọn cách: (a) server có ≥ 3 GPU, hoặc (b) dùng model nhỏ hơn/API cho các arm phụ
- [ ] `bash scripts/ec2/serve_backbones.sh` (mặc định :8001 llama-3.1-8B, :8002 Qwen2.5-7B,
      :8003 Mistral-7B — mỗi backbone một GPU; `serve_vllm.sh` giờ chỉ no-op theo **đúng port**)
- [ ] Dùng `configs/ec2_rq4_diverse.yaml` (đã map sẵn `backbones`/`models`/`endpoints`) rồi
      `python scripts/check_ready.py --config configs/ec2_rq4_diverse.yaml`
- [ ] `AEGIS_OUT=outputs/real_rq4 bash scripts/ec2/run_real.sh full`
- [ ] So sánh ρ (`real_theory_analysis.csv`) và ASR-UC giữa homogeneous vs diverse
- [ ] Nếu không đủ GPU/API: báo RQ4 là **not measured**, KHÔNG suy diễn từ homogeneous arm

## 11. Ghi nhận kết quả (paper honesty — bắt buộc)

- [ ] Giữ nguyên mọi `*provenance.json` cùng với các CSV output **và verdict cache**
      (cache là "raw data" mà CSV được tính ra từ đó)
- [ ] Đóng gói + tải về: `bash scripts/ec2/backup_results.sh --outputs outputs/real_harmbench`
      rồi `scripts/backup_results.ps1` (verify sha256 khi tải)
- [ ] `python scripts/finalize_run.py --run <run dir>` → verify + ghi
      `audits/run_records/<run_id>.json` + cập nhật bảng trong `audits/result_integrity_audit.md`
- [ ] Chỉ set `is_paper_result: true` SAU khi chạy lại độc lập khớp số:
      `python scripts/finalize_run.py --run <run1> --replicated-by <run2> --set-paper-result`
      (script tự từ chối nếu verify fail hoặc hai summary không khớp)
- [ ] Điền các ô `--` trong Table 5/6 và thay figure placeholder (`pgfplots_placeholder_results.tex`)

## 12. Dọn dẹp

- [ ] Lưu outputs + provenance + cache xuống máy Windows (mục 11)
- [ ] Dừng backbone: `kill $(cat outputs/vllm-*.pid)`
- [ ] **Stop instance** khi không dùng (vẫn tính phí disk nhỏ), hoặc **Terminate** nếu xong hẳn
- [ ] Thu hồi key pair / xóa security group không dùng

## Ước tính chi phí tham khảo

- `g5.xlarge` ≈ $1.0/h; `g5.12xlarge` ≈ $4.5/h (giá có thể thay đổi theo region — kiểm tra trên AWS Pricing)
- Full run: ~4–5h trên g5.xlarge → budget ~5–10 USD cho 1 benchmark
- Spot instance có thể giảm ~60–90% phí (rủi ro bị interrupt — dùng cho validate, không dùng cho run cuối)