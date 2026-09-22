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
    - Tiết kiệm: `g4dn.2xlarge` (T4 16 GB VRAM) — vừa đủ 1 model 8B, ít dư địa
    - Muốn RQ4 đa backbone ngay: `g5.12xlarge` (4×A10G 96 GB VRAM)
- [ ] AMI: Ubuntu 22.04/24.04 (driver NVIDIA + CUDA 12.x sẵn)
- [ ] Disk: root ≥ **50 GB** (weights + vLLM cache + benchmarks)
- [ ] Security group: mở **SSH inbound (TCP 22)** từ IP của bạn. Các port vLLM (8001+) chỉ bound localhost nên không cần mở public

## 3. Provision server (một lần)

- [ ] SSH vào: `ssh -i ~/.ssh/aegis.pem ubuntu@<ec2-host>`
- [ ] Clone repo: `git clone <repo-url>` rồi `cd AegisAgency`
- [ ] Chạy: `bash scripts/ec2/setup.sh` (cài python3.11, `.venv`, `pip install -e ".[ec2]"`, tạo `/data/benchmarks`, chạy sanity test ~67 tests)

## 4. Upload benchmarks (một lần, từ máy Windows)

- [ ] Chạy từ thư mục repo:
    ```powershell
    powershell -ExecutionPolicy Bypass -File scripts/ec2/upload_data.ps1 `
        -HostTarget ubuntu@<ec2-host> -Key "$env:USERPROFILE\.ssh\aegis.pem"
    ```
- [ ] Verify trên server: 8 thư mục nằm tại `/data/benchmarks/`

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

- [ ] `bash scripts/ec2/run_real.sh smoke` — dummy judge, 40 payloads, chỉ kiểm tra plumbing (KHÔNG dùng số liệu này)
- [ ] Sửa `configs/ec2_real_evaluation.yaml`: `limit: 200`, `n_seeds: 1`, `measure_epsilon: false`
- [ ] `bash scripts/ec2/run_real.sh full` — chạy thử rẻ hơn để xác nhận pipe, cache, outputs
- [ ] Kiểm tra outputs thực sự xuất hiện: `outputs/real/real_evaluation_summary.csv`, provenance `.json`

## 9. Full run (Table 5/6)

- [ ] Sửa lại config: `limit: 0` (toàn bộ payload), `n_seeds: 3`, `measure_epsilon: true`, `calibrate: true`
    - Đảm bảo cache verdict rỗng với run sẽ báo cáo (cost/thời gian đo trên cold cache)
- [ ] `bash scripts/ec2/run_real.sh full` (ước tính ~8400 judge calls ≈ 4–5h serial trên GPU mid)
- [ ] `bash scripts/ec2/run_real.sh ablate` — Table 6 (reuse honest cache, thêm ít chi phí)
- [ ] Chạy nhiều benchmark nếu cần: `advbench`, `injecagent`, `dan`, `formal`, `second_order` (đổi `data.benchmark`, dùng chung cache)

## 10. RQ4 — diverse committee (sau khi có Table 5/6)

- [ ] Chọn cách: (a) nâng lên `g5.12xlarge` hoặc (b) dùng model nhỏ hơn trên cùng GPU
- [ ] Serve mỗi backbone một port (lưu ý: `serve_vllm.sh` tự thoát nếu đã có vLLM đang chạy → start tay hoặc sửa script)
- [ ] Map trong config:
    ```yaml
    judges:
      backbones: [llama-3, qwen2.5, mistral]
      models:
        llama-3: meta-llama/Llama-3.1-8B-Instruct
        qwen2.5: Qwen/Qwen2.5-7B-Instruct
        mistral: mistralai/Mistral-7B-Instruct-v0.3
      endpoints:
        llama-3: http://127.0.0.1:8001/v1
        qwen2.5: http://127.0.0.1:8002/v1
        mistral: http://127.0.0.1:8003/v1
    ```
- [ ] Chạy lại `run_real.sh full` (cache cũ của backbone khác vẫn được tái dùng)
- [ ] So sánh ρ (đo bằng `estimators.py`) và ASR-UC giữa homogeneous vs diverse

## 11. Ghi nhận kết quả (paper honesty — bắt buộc)

- [ ] Giữ nguyên mọi `*provenance.json` cùng với các CSV output
- [ ] Cập nhật `audits/result_integrity_audit.md`: run nào thật, nguồn data, lệnh + hash
- [ ] Chỉ set `is_paper_result: true` SAU khi áp checklist verify và chạy lại độc lập (`docs/reproducibility.md`)
- [ ] Điền các ô `--` trong Table 5/6 và thay figure placeholder (`pgfplots_placeholder_results.tex`)

## 12. Dọn dẹp (tránh phí)

- [ ] Lưu outputs + provenance xuống máy Windows (scp/git)
- [ ] **Stop instance** khi không dùng (vẫn tính phí disk nhỏ), hoặc **Terminate** nếu xong hẳn
- [ ] Thu hồi key pair / xóa security group không dùng

## Ước tính chi phí tham khảo

- `g5.xlarge` ≈ $1.0/h; `g5.12xlarge` ≈ $4.5/h (giá có thể thay đổi theo region — kiểm tra trên AWS Pricing)
- Full run: ~4–5h trên g5.xlarge → budget ~5–10 USD cho 1 benchmark
- Spot instance có thể giảm ~60–90% phí (rủi ro bị interrupt — dùng cho validate, không dùng cho run cuối)