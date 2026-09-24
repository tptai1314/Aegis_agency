# Borrowed-GPU-Server Runbook (dùng khi MƯỢN/thuê server GPU, KHÔNG dùng EC2)

File này thay thế phần "provision an instance" của `docs/ec2_experiment_guide.md` khi bạn không
dùng AWS. **Toàn bộ pipeline không đổi** (`setup.sh`, `serve_vllm.sh`, `run_real.sh`,
`check_ready.py`, `finalize_run.py`); chỉ khác ở khâu tiếp cận server, quyền `sudo`, và cách
truyền dữ liệu. Tên thư mục `scripts/ec2/` chỉ là tên lịch sử — không có gì phụ thuộc AWS.

> **Dataset 8 benchmark đã nằm sẵn trong repo** (`data/benchmarks/`, đã được git track). `git clone`
> là đủ; **không cần upload dữ liệu** trừ khi bạn muốn dùng `/data/benchmarks`.

## 0. Tiền đề ở máy Windows

- HF gated **đã duyệt** cho `meta-llama/Llama-3.1-8B-Instruct` + `meta-llama/Llama-2-7b-chat-hf`.
- `HF_TOKEN` (read) của **đúng account đã duyệt**; thử `huggingface-cli whoami` trước.
- Code **đã commit + push** (`git status` sạch) — vì `provenance.json` ghi `git_commit`, và
  `check_ready.py` sẽ cảnh báo nếu working tree còn file chưa commit.

## 1. Hỏi chủ server (hoặc tự kiểm tra trên server)

```bash
nvidia-smi                                   # VRAM >= 24 GB (A10G/A100/4090/3090), driver >= 525
nvidia-smi | grep -i "CUDA Version"
free -g                                      # RAM >= 32 GB (khuyên 64 GB)
df -h /                                      # trống >= 60 GB (weights ~16 GB + HF cache + vLLM)
cat /etc/os-release                          # Ubuntu 22.04 / 24.04
python3 -V                                   # >= 3.11 (24.04 có 3.12 sẵn; 22.04 cần deadsnakes)
ss -tlnp | grep -E ':800[1-3]'               # port 8001-8003 phải trống
curl -sI https://huggingface.co | head -n 1  # có mạng ra HF
sudo -n true && echo "sudo OK" || echo "KHÔNG có passwordless sudo"
```

`setup.sh` **không bắt buộc sudo**: nếu không có sudo, nó bỏ qua bước tạo `/data/benchmarks`
(config mặc định dùng `data/benchmarks` trong repo) và chỉ cần quyền ghi trong `$HOME`.

Tiêu chuẩn tối thiểu: **GPU ≥ 24 GB, Ubuntu + Python ≥ 3.11, RAM ≥ 32 GB, internet ra HF**.
Hỏi rõ: SSH bằng gì (user@host, key hay password, port nào), server có **dùng chung** không
(nếu có người khác chạy job → không dùng cho run báo cáo, GPU sẽ bị giành).

## 2. Đưa code lên server

Cách 1 (khuyến nghị) — **git clone** (repo này LÀ git repo, `data/benchmarks` đi kèm):

```bash
ssh user@host
git clone <repo-url> && cd AegisAgency
```

Cách 2 — nếu không push được repo, dùng `rsync`/`scp` (vẫn nhớ `data/benchmarks`):

```powershell
rsync -avz --exclude .venv --exclude outputs --exclude "__pycache__" --exclude "*.egg-info" `
      D:\Documents\NCKH\aegis_agency\AegisAgency user@host:~/AegisAgency
```

> Lưu ý: `.gitignore` loại `README.vi.md`, `docs/vi/`, `manuscript_vi/`. Nếu bạn cần tài liệu
> tiếng Việt trên server thì phải copy tay — `git clone` sẽ **không** có chúng.
>
> `scripts/ec2/upload_data.ps1` (chỉ dùng khi muốn `/data/benchmarks`) chạy `sudo` ở phía server.
> Không có sudo → giải nén tarball vào thư mục bạn sở hữu rồi sửa `data.root` cho khớp.

## 3. Chạy theo đúng thứ tự (SSH vào server)

```bash
# 3.1 provision — tự dò Python >= 3.11, không hardcode python3.11
cd ~/AegisAgency
PYTHON=python3.12 bash scripts/ec2/setup.sh          # hoặc để trống, script tự dò
```

```bash
# 3.2 job GPU ĐẦU TIÊN (chỉ khi cần universal_injection): sinh suffix thật, độc chiếm GPU,
#     chạy vài giờ. Script tự từ chối chạy nếu còn vLLM đang giữ GPU.
HF_TOKEN=hf_... bash scripts/ec2/run_universal_suffix.sh \
    --injection semi-dynamic --tokens 150 --num-steps 500
#     → script in ra đường dẫn JSON + lệnh scp; làm theo docs/universal_injection_runbook.md
#     → rebuild CSV trên Windows rồi đưa lại vào repo TRƯỚC khi chạy benchmark này
```

```bash
# 3.3 serve judge Llama-3.1-8B @ :8001 (GPU phải rảnh sau 3.2)
HF_TOKEN=hf_... bash scripts/ec2/serve_vllm.sh
#     RQ4 diverse: bash scripts/ec2/serve_backbones.sh   (cần 3 GPU, hoặc dùng API)
```

```bash
# 3.4 preflight — BẮT BUỘC trước mọi run báo cáo (0 FAIL mới chạy)
python scripts/check_ready.py --config configs/ec2_real_evaluation.yaml

# 3.5 pipe thật (dummy judge, 40 payload — KHÔNG report)
AEGIS_OUT=outputs/real.smoke bash scripts/ec2/run_real.sh smoke

# 3.6 Table 5 (harmbench). Mỗi benchmark PHẢI có AEGIS_OUT riêng, nếu không sẽ ghi đè nhau.
AEGIS_OUT=outputs/real_harmbench bash scripts/ec2/run_real.sh full
AEGIS_OUT=outputs/real_harmbench bash scripts/ec2/run_real.sh ablate     # Table 6, reuse cache

# 3.7 benchmark khác (đổi data.benchmark trong config, giữ AEGIS_OUT riêng)
AEGIS_OUT=outputs/real_injecagent bash scripts/ec2/run_real.sh full
```

**Đọc `attack_effect` trong `real_evaluation_summary.csv` TRƯỚC khi đọc ASR.** Nếu
`attack_effect = 0` ở mọi rule Aegis thì attack không hề đổi một quyết định nào → mọi method
giống nhau và bảng không nói lên điều gì. Với `collusion radius 0.1` (mặc định) trên payload có
honest score ~1.0, khả năng cao `attack_effect = 0`: phải tăng `attack_kwargs.radius/budget`
(và ghi lại trong provenance), không thì đừng báo cáo bảng so sánh.

## 4. Universal injection (khâu nối tay)

Toàn bộ quy trình (optimise → scp về → rebuild → verify `gradient_optimized_suffix: true`) nằm ở
`docs/universal_injection_runbook.md`. Tóm tắt:

```powershell
# trên Windows, sau khi có JSON trên server:
scp -i "$env:USERPROFILE\.ssh\<key>" user@host:"<results json>" .\upi_results\
python scripts/prepare_universal_injection.py `
    --source D:\Data\downloads\Universal-Prompt-Injection `
    --output data/benchmarks/universal_injection `
    --suffix-file .\upi_results\<results json>
python -c "import json;print(json.load(open('data/benchmarks/universal_injection/provenance.json'))['gradient_optimized_suffix'])"
# phải in ra True; False = bản SIMULATED, KHÔNG được dùng
```

## 5. RQ4 — diverse committee (làm SAU Table 5/6)

- Serve thêm backbone mỗi cái một port: `bash scripts/ec2/serve_backbones.sh`
  (mặc định :8001 llama-3.1-8B, :8002 Qwen2.5-7B, :8003 Mistral-7B — mỗi cái một GPU).
- Dùng `configs/ec2_rq4_diverse.yaml` (đã map sẵn `backbones`/`models`/`endpoints`) rồi
  `python scripts/check_ready.py --config configs/ec2_rq4_diverse.yaml`.
- Chạy `AEGIS_OUT=outputs/real_rq4 bash scripts/ec2/run_real.sh full`.
- So `rho`/`score_rho` (`real_theory_analysis.csv`) và ASR-UC giữa homogeneous vs diverse.
- 1 GPU 24 GB **không đủ** serve 3×8B: dùng API (gpt-4o/claude) cho các arm phụ, hoặc server có
  2+ GPU, hoặc báo RQ4 là "not measured".

## 6. Sao lưu + ghi nhận (bắt buộc, tránh mất công)

```bash
# trên server: đóng gói kết quả + provenance + verdict cache + manifest sha256
bash scripts/ec2/backup_results.sh --outputs outputs/real_harmbench
```

```powershell
# trên Windows: tải về + verify sha256 + giải nén
powershell -ExecutionPolicy Bypass -File scripts/backup_results.ps1 `
    -HostTarget user@host -Key "$env:USERPROFILE\.ssh\<key>" `
    -RemoteBundle "~/aegis_backups/aegis_real_harmbench_<stamp>.tar.gz"
python scripts/finalize_run.py --run .\backups\<unpacked>\outputs\real_harmbench
```

`finalize_run.py` verify run, ghi `audits/run_records/<run_id>.json` (sha256 từng artefact) và cập
nhật bảng trong `audits/result_integrity_audit.md`. Chỉ khi chạy lại độc lập khớp số:

```bash
python scripts/finalize_run.py --run <run1> --replicated-by <run2> --set-paper-result
```

## 7. Kiểm tra an toàn khi mượn server

- Không để token/HF trong file commit — chỉ `export HF_TOKEN=...` trong phiên. Script
  universal-suffix cố ý **không** chạy `huggingface-cli login --add-to-git-credential` (tránh lưu
  credential lên máy mượn).
- Port 8001-8003 chỉ bind `127.0.0.1`; không cần mở firewall.
- Trước khi trả server: `kill $(cat outputs/vllm-*.pid)`, giữ lại `outputs/` + `*provenance.json`
  + verdict cache, xoá dữ liệu tạm nếu chủ server yêu cầu.
- Ghi `hostname` + `git_commit` + `pip freeze` vào `audits/result_integrity_audit.md`
  (provenance đã tự ghi `hostname`/`created_utc`/`git_commit`/`deps`).
