# Borrowed-GPU-Server Runbook (dùng khi MƯỢN server GPU, không thuê EC2)

Thay thế mục "Launch EC2" bằng checklist này. Phần pipeline (`setup.sh`, `serve_vllm.sh`,
`run_real.sh`) **không đổi** — chỉ khác ở khâu tiếp cận server và quyền sudo.

## 0. Tiền đề ở máy Windows (đã xong)

- HF gated **đã duyệt** cho `meta-llama/Llama-3.1-8B-Instruct` + `meta-llama/Llama-2-7b-chat-hf` (đã test: OK).
- `HF_TOKEN` (read) của **đúng account đã duyệt**, test bằng local trước khi lên server.
- Data sẵn ở `D:\Data\benchmarks` (8 thư mục + `PROVENANCE.json`).

## 1. Hỏi chủ server (hoặc tự kiểm tra trên server)

```bash
nvidia-smi                                   # VRAM >= 24 GB (A10G/A100/4090/3090), driver cu118+
nvidia-smi | grep -i "CUDA Version"
free -g                                      # RAM >= 32 GB (khuyên 64 GB)
df -h /                                      # ổ trống >= 60 GB (Llama-3.1 ~16 GB + HF cache + vLLM)
cat /etc/os-release                          # Ubuntu 20.04/22.04 (cần apt-get + python3.11)
python3.11 --version                         # setup.sh dùng python3.11
ss -tlnp | grep -E ':800[1-3]'               # port 8001-8003 phải trống
curl -sI https://huggingface.co | head -n 1  # có mạng ra HF (tải Llama ~16 GB)
sudo -n true && echo "sudo OK"               # setup.sh cần sudo (mkdir /data + apt install)
```

Tiêu chuẩn tối thiểu: **GPU >= 24 GB, Ubuntu + sudo, RAM >= 32 GB, internet tải HF được**.
Hỏi rõ:
- SSH vào bằng gì (user@host, key hay password, port mấy)?
- Server có **dùng chung** không? Nếu có người khác chạy job → không nên dùng cho run paper (Ảnh GPU bị giành).

## 2. Đưa code + data lên server

Repo cục bộ **không phải git repo**, nên dùng scp/rsync (ko dùng git clone):

```powershell
# (a) code (bỏ thư mục nặng/không cần: .venv, outputs, __pycache__, *_synthetic)
rsync -avz --exclude .venv --exclude outputs --exclude "__pycache__" --exclude "*.egg-info" \
      D:\Documents\NCKH\aegis_agency\AegisAgency user@host:~/AegisAgency

# (b) data benchmarks
powershell -ExecutionPolicy Bypass -File scripts/ec2/upload_data.ps1 `
    -HostTarget user@host -RemoteRoot /data/benchmarks
```

> Không có sudo / `-RemoteRoot /data` lỗi → dùng thư mục user có quyền ghi:
> `-RemoteRoot /home/user/data/benchmarks` **và** sửa `data.root` trong
> `configs/ec2_real_evaluation.yaml` cho khớp.

## 3. Chạy theo đúng thứ tự (SSH vào server)

```bash
# 3.1 provision (ubuntu + sudo) — chạy trong ~/AegisAgency
cd ~/AegisAgency && bash scripts/ec2/setup.sh

# 3.2 job GPU ĐẦU TIÊN: sinh suffix universal injection (vài tiếng, độc chiếm GPU)
HF_TOKEN=hf_... bash scripts/ec2/run_universal_suffix.sh \
    --injection semi-dynamic --tokens 150 --num-steps 500

#    trong lúc chờ: đảm bảo data đã lên /data/benchmarks (bước 2b)
#    sau khi xong: scp results JSON về Windows, rebuild benchmark, upload lại

# 3.3 serve judge Llama-3.1-8B @ :8001 (GPU rảnh sau 3.2)
HF_TOKEN=hf_... bash scripts/ec2/serve_vllm.sh

# 3.4 pipe thật
bash scripts/ec2/run_real.sh smoke    # dummy judge 40 payload — KHÔNG report
bash scripts/ec2/run_real.sh full     # Table 5 (harmbench collusion ~4-5h)
bash scripts/ec2/run_real.sh full     # thêm mỗi benchmark khác 1 lệnh (dan, formal, ...)
bash scripts/ec2/run_real.sh ablate   # Table 6 (reuse honest cache)
```

## 4. Universal injection (khâu nối tay)

```powershell
# trên Windows, sau khi có JSON trên server:
scp user@host:~/Universal-Prompt-Injection/results/eval/llama2/semi-dynamic/.../*.json .
python scripts/prepare_universal_injection.py \
    --source D:\Data\downloads\Universal-Prompt-Injection \
    --output D:\Data\benchmarks\universal_injection \
    --suffix-file <results.json>
# upload lại (bước 2b) rồi chạy thêm 1 run_real.sh full cho universal_injection
```

## 5. RQ4 — diverse committee (làm SAU Table 5/6)

- Serve thêm backbone mỗi cái 1 port: `VLLM_MODEL=... VLLM_PORT=8002 bash scripts/ec2/serve_vllm.sh`
  (script tự thoát nếu có vLLM đang chạy → start tay từng port).
- Sửa map `judges.backbones/models/endpoints` trong `configs/ec2_real_evaluation.yaml`
  (mẫu: `docs/ec2_checklist_vi.md:94-105`).
- Chạy lại `run_real.sh full` — cache honest reuse, tốn ít.
- Chi phí GPU: 1 GPU 24 GB chỉ đủ serve 1×8B → arm diverse bằng **gpt-4o/claude API**
  (không tốn GPU) hoặc server mượn có 2+ GPU.

## 6. Kiểm tra an toàn/giới hạn khi mượn

- Không để token/HF trong file commit — chỉ truyền qua `export HF_TOKEN=...` trong phiên.
- Xác nhận port 8001-8003 không bị firewall chặn ngoài (nếu chạy judge qua mạng khác).
- Nhớ dừng vLLM + xoá `outputs/real` tạm khi trả server (Section 12 `docs/ec2_checklist_vi.md`).
- Giữ mọi `*provenance.json` + CSV output để quay về `audits/result_integrity_audit.md`.