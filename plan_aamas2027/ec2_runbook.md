# EC2 RUNBOOK — TỰ HOST UỶ BAN JUDGE

Bổ trợ cho `TODO_AAMAS2027.md`. Giả định: **không giới hạn chi phí**, thời gian là ràng buộc duy nhất.

> ## ⚠️ KẾT LUẬN QUAN TRỌNG NHẤT
> **Compute KHÔNG phải nút thắt.** Toàn bộ suy luận của một sweep đầy đủ chạy trong **15–35 phút** trên instance đề xuất.
> **Nút thắt của 17 ngày còn lại là hai thủ tục hành chính, cả hai đều mặc định là "0/không có":**
> 1. **AWS GPU quota mặc định = 0** cho họ G/VT và P → phải xin tăng, có thể mất **vài giờ đến 72 giờ**.
> 2. **HuggingFace gating** cho Llama-3.1-8B, Llama-3.3-70B, Gemma-2-9B, Meta-SecAlign-8B → duyệt **thủ công**.
>
> **Hai việc này phải bắt đầu NGAY HÔM NAY (21/09), song song với mọi việc khác.** Nếu chờ đến khi code xong mới xin, bạn sẽ mất 1–3 ngày chết.

---

## 1. Chọn instance


| Instance | GPU | VRAM/GPU | Số GPU | Tổng VRAM | vCPU / RAM | $/giờ | FP8 W8A8? |
|---|---|---|---|---|---|---|---|
| **g6e.12xlarge** ✅ **CHỌN** | L40S | 44 GiB | 4 | **179 GiB** | 48 / 384 GiB | **~$10,49** | ✅ sm_89 |
| g6e.48xlarge | L40S | 44 GiB | 8 | 357 GiB | 192 / 1536 GiB | ~$30,13 | ✅ sm_89 |
| g5.12xlarge | A10G | 22 GiB | 4 | 89 GiB | 48 / 192 GiB | ~$5,67 | ❌ sm_86 |
| g7e.48xlarge | RTX PRO 6000 | 96 GiB | 8 | 768 GiB | 192 / 2048 GiB | ~$33.14 | ⚠️ sm_120, nhiều lỗi vLLM đã biết |
| g6e.24xlarge | L40S | 44 GiB | 4 | 178 GiB | 96 / 768 GiB | ~$15.07 | ✅ |
| g5.48xlarge | A10G | 22 GiB | 8 | 178 GiB | 192 / 768 GiB | ~$16.29 | ❌ sm_86 |
| p4d.24xlarge | A100 | 40 GiB | 8 | 320 GiB | 96 / 1152 GiB | ~$21.96 | ❌ **sm_80** |
| p5.48xlarge | H100 | 80 GiB | 8 | 640 GiB | 192 / 2048 GiB | ~$55.04 | ✅ sm_90 |
| g4dn.12xlarge | T4 | 16 GiB | 4 | 64 GiB | 48 / 192 GiB | ~$3.91 | ❌ sm_75 |

**✅ Chốt `g6e.12xlarge`:** 4× L40S là **đủ và dư** — trọng số 4 backbone + adapter SecAlign ≈ **60 GiB**, KV cache ≈ **20 GiB**, overhead ≈ **6 GiB** → **~86 GiB / 179 GiB (48%)**. Rẻ hơn `g6e.48xlarge` **3 lần** và chỉ cần xin **48 vCPU** quota thay vì 192. Đầy đủ lý do ở `ec2_spec_comparison.md` §5.

**Vì sao không dùng p4d/A100:** vLLM chỉ hỗ trợ **FP8 W8A8** trên GPU compute capability ≥ 8.9 (Ada/Hopper/Blackwell). A100 là **sm_80** → không có W8A8, chỉ có W8A16 weight-only. L40S (sm_89) hỗ trợ đầy đủ.

**Vì sao không dùng `g5.12xlarge` làm phương án chính:** A10G chỉ 22 GiB/card → **Gemma-2-9B không lọt** (17,22 GiB trọng số + 10,5 GiB KV = 27,7 GiB). Chỉ chạy được **3 backbone** thay vì 4 → chỉ dùng làm tầng ngân sách.

**Vì sao không dùng `g5.xlarge` / `g4dn.2xlarge`:** `g4dn.2xlarge` (T4 16 GiB) **không chạy được** model 8B ở bf16 (trọng số đã chiếm 16,06 GB); `g5.xlarge` chỉ có **16 GiB RAM hệ thống** — quá chật khi nạp model.

**Lưu ý giá:** snapshot **21/09/2026 12:51 UTC** (Holori) — **kiểm tra lại lúc mua**.

---

## 2. Bố trí serving

**Nguyên tắc: TP=1, một model một GPU.** Mỗi model 7–9B nằm gọn trong một L40S 44 GiB ở bf16 → tensor parallelism không tiết kiệm VRAM mà chỉ thêm all-reduce qua PCIe.

```
GPU0   :8000  meta-llama/Llama-3.1-8B-Instruct      bf16   (+ LoRA: Meta-SecAlign-8B)
GPU1   :8001  Qwen/Qwen2.5-7B-Instruct              bf16
GPU2   :8002  mistralai/Mistral-7B-Instruct-v0.3    bf16
GPU3   :8003  google/gemma-2-9b-it                  bf16
:4000         LiteLLM proxy  <-- endpoint DUY NHẤT mà experiment nói chuyện
```

- vLLM **không** hỗ trợ nhiều model trên một server (issue #13633 đã đóng với `not_planned`). Cần nhiều tiến trình server + một router.
- **LiteLLM, không dùng Ray Serve.** vLLM đã có continuous batching; Ray Serve thêm scheduler và một failure mode mới mà không tăng throughput ở quy mô này.
- **Multi-LoRA chỉ dùng cho đúng một cặp:** `Meta-SecAlign-8B` trên server Llama-3.1-8B (adapter 281 MB). Qwen/Mistral/Gemma khác kiến trúc → không dùng chung base.
- **Không bật `VLLM_SERVER_DEV_MODE=1`** trên cổng ra internet (mở `/sleep`, `/wake_up`, `/collective_rpc`). Bind vLLM vào `127.0.0.1`, chỉ mở LiteLLM (hoặc tunnel SSH).

**Uỷ ban `n = 1..7` trên 4 GPU — không cần sửa code.** Đã kiểm chứng `src/aegis_agency/data/real_judges.py:576`: `backbone = backbones[k % len(backbones)]`. Với 4 backbone và `n_judges=7`, thẩm phán 5–6 lặp lại backbone 0–1 với `judge_id` khác nhau, nên verdict cache `(payload_id, judge_id, backbone)` vẫn tách bạch đúng. Đây cũng là cách một triển khai thật sẽ làm (nhiều thẩm phán trên vài backbone).

**Cấu hình then chốt cho MỌI server:**
```
--max-model-len 2048          # workload ~660 token; 8192 hay 32768 không giúp gì mà tăng KV 4-16x
--max-num-seqs 16
--gpu-memory-utilization 0.92
--enable-prefix-caching
```

---

## 3. Đội hình model + revision (SHA) + trạng thái gating

| Vai trò | Model | Revision (SHA) | Gated? | Ghi chú |
|---|---|---|---|---|
| Judge nền | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | 🔒 **manual** | |
| **Judge đã hardening** | `facebook/Meta-SecAlign-8B` | `fb9b039b45ab4fe5e94517efaaf19f80f4fedda1` | 🔒 **manual + form** (họ tên, ngày sinh, quốc gia, affiliation) | **Là LoRA adapter 281 MB** trên `meta-llama/Meta-Llama-3.1-8B-Instruct` — **không phải model độc lập** |
| Judge đa dạng | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | ✅ mở | |
| Judge đa dạng | `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | ✅ mở | |
| Judge đa dạng | `google/gemma-2-9b-it` | `11c9b309abf73637e4b6f9a3fa1e92e615547819` | 🔒 **manual** | ⚠️ xem landmine §7 |
| Judge mạnh *(tuỳ chọn, cần 8 GPU)* | `meta-llama/Llama-3.3-70B-Instruct` | `6f6073b423013f6a7d4d9f39144961bfbfbc386b` | 🔒 **manual** | FP8, TP=2. **Không cần cho `g6e.12xlarge`** |

**Về SecAlign gốc (Chen et al., CCS 2025):** checkpoint gốc **KHÔNG có trên HuggingFace Hub** — chúng là file zip trên CDN của Meta (`dl.fbaipublicfiles.com/SecAlign/...`), tải bằng `python setup.py`. Có cả bản LoRA (~0.2 GB) và bản full weight (26–30 GB), base: Llama-7B / Mistral-7B-v0.1 / Meta-Llama-3-8B. **Repo SecAlign là slurm-oriented** → phải tự tích hợp, không có `from_pretrained` một dòng.
**StruQ:** cũng chỉ trên CDN, base `huggyllama/llama-7b` và `mistralai/Mistral-7B-v0.1`, full weight. README của StruQ **tự nói repo đã bị thay thế** và trỏ sang SecAlign.

> **Chiến lược thực dụng:** dùng `facebook/Meta-SecAlign-8B` (LoRA, dễ triển khai nhất qua vLLM multi-LoRA) làm judge hardened chính; chỉ dùng SecAlign/StruQ gốc qua CDN nếu còn thời gian sau khi đã có kết quả chính.

---

## 4. VRAM — tính toán

Trọng số: bf16 = params × 2 B; FP8 = × 1 B; INT4 = × 0.5 B.

| Model | Params | bf16 | FP8 | INT4 |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 8.03 B | 14.96 GiB | 7.48 | 3.74 |
| Qwen2.5-7B-Instruct | 7.62 B | 14.19 | 7.09 | 3.55 |
| Mistral-7B-Instruct-v0.3 | 7.25 B | 13.50 | 6.75 | 3.38 |
| Gemma-2-9B-it | 9.24 B | 17.22 | 8.61 | 4.30 |
| Meta-SecAlign-8B | LoRA | ~0.1 trên base | — | — |
| Llama-3.3-70B-Instruct | 70.55 B | 131.4 | **65.7** | 32.9 |

KV cache/token (bf16, tại `--max-model-len 2048`, 16 seq đồng thời): Llama-8B 4.0 GiB · Qwen 1.75 · Mistral 4.0 · **Gemma 10.5** (head_dim 256) · 70B 10.0. Giảm một nửa nếu dùng `--kv-cache-dtype fp8_e4m3`.

→ **g6e.48xlarge (357 GiB) chứa đủ cả 7 slot ở bf16, 1 model/GPU, cộng 70B FP8 trên 2 GPU, và vẫn thừa 2 GPU.**

---

## 5. Thông lượng & wall-clock

Neo đo được: L40S bf16 8B ≈ **50–84 tok/s mỗi stream**; decode có batch bị chi phối bởi đọc trọng số → ở B=16 ước tính **~520 tok/s tổng hợp** mỗi GPU; prefill ~**5.000 tok/s/GPU**.

Với 8.400 call chia cho 6 judge (~1.400 call/judge, 600 in + 60 out):
- prefill 840.000 / 5.000 ≈ 168 s; decode 84.000 / 520 ≈ 162 s
- → **~5–6 phút mỗi judge, chạy song song → ~10–15 phút wall-clock**, cộng 5–20 phút load model.

**Kể cả trường hợp xấu nhất (mọi call dồn vào 1 GPU): ~30 phút.** Workload này không bao giờ bị chặn bởi compute trên instance này.

> ⚠️ Đây là **mô hình ước lượng, không phải số đo**. Hãy chạy `vllm bench serve` ở concurrency 16 trong ~30 phút (ngày 4–5) để hiệu chuẩn **trước** khi sweep chính thức.

---

## 6. Checklist provisioning (theo thứ tự)

**Ngày 1 — HÀNH CHÍNH, LÀM NGAY:**
1. [ ] Xin tăng quota **"Running On-Demand G and VT instances" ≥ 256 vCPU** ở `us-east-1` (g6e.48xlarge cần 192 vCPU trong bucket G/VT). **Mặc định hiện tại = 0.**
2. [ ] Xin quyền truy cập HuggingFace cho: `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct`, `meta-llama/Meta-Llama-3.1-8B-Instruct` (base của SecAlign LoRA), `google/gemma-2-9b-it`, `facebook/Meta-SecAlign-8B`.
3. [ ] Tạo HF token (read) và lưu an toàn.

**Ngày 1–2 (chờ quota):**
4. [ ] Thuê một instance rẻ, tạo **1 TB gp3 (1000 MiB/s, 16.000 IOPS)** làm cache bền.
5. [ ] `pip install hf_transfer`, export `HF_HUB_ENABLE_HF_TRANSFER=1`, tải trọng số với **`--exclude "original/*"`** (repo HF chứa bản trùng, tổng có thể gấp đôi).
6. [ ] Pull sẵn image `vllm/vllm-openai:v0.29.0-cu129` và **ghi lại digest** (`docker inspect`).
   *Dung lượng: 206 GB nếu có 70B, ~65 GB nếu không. Thời gian tải: 1–3 h (có thể 12–35 phút nếu mạng tốt).*

**Ngày 3–5 (quota đã về):**
7. [ ] Khởi tạo **g6e.48xlarge on-demand** ở AZ đã test. **Test-launch sớm** — `InsufficientInstanceCapacity` rất phổ biến với họ GPU; thử rải us-east-1a/b/c/d.
8. [ ] `rsync` trọng số sang **NVMe cục bộ 7,6 TB** (4×1900 GB) để load nhanh hơn nhiều.
9. [ ] Khởi động 5 vLLM server; **ghi lại dòng log `GPU KV cache size:`** của mỗi server (đây là sự thật về cấu hình bạn thực sự có).
10. [ ] `curl :800X/v1/models` + 1 completion smoke cho mỗi cổng.
11. [ ] Dựng LiteLLM ở `:4000`, liệt kê đủ 7 model id.
12. [ ] `vllm bench serve` mỗi judge ở concurrency 16 → hiệu chuẩn tok/s thật.
13. [ ] **Chạy determinism probe:** 100 call giống hệt nhau, hai lần, cùng concurrency, diff kết quả, **công bố tỷ lệ khớp tuyệt đối**.

**Ngày 6+:** sweep chính thức.

---

## 7. Bẫy kỹ thuật (sẽ làm hỏng thí nghiệm nếu bỏ qua)

1. **Gemma-2-9B-it ném lỗi với system message.** Chat template của nó chứa `{{ raise_exception('System role not supported') }}` → **HTTP 400 với mọi system message**. Phải gộp system prompt vào lượt user đầu tiên, hoặc cấp `--chat-template` riêng.
2. **Mistral-v0.3 yêu cầu luân phiên user/assistant nghiêm ngặt** — cũng có thể raise.
3. **"600 token" không giống nhau giữa các judge.** Vocabulary: Llama 128.256 / Qwen 152.064 / Mistral 32.768 / Gemma 256.000. Đếm prompt theo **ký tự** khi thiết kế, và **log số token thật** mỗi call.
4. **vLLM greedy decoding KHÔNG bất biến theo batch.** Thứ tự reduction trong attention/GEMM có thể lật một token khi thành phần batch thay đổi. → **Cố định và báo cáo `--max-num-seqs`**, và công bố kết quả determinism probe ở mục 6.13. Nếu tỷ lệ khớp < 100%, **nói thẳng** — reviewer sẽ tin phần còn lại hơn.
5. **Không dùng Spot cho run chính thức.** g6e.48xlarge spot rẻ (~$9/giờ) nhưng **tỉ lệ ngắt 15–20%**; mỗi lần khởi động lại mất 5–20 phút. Spot chỉ dùng cho pha tải trọng số.
6. **RAM:** cần ≥ 1,5 × tổng dung lượng trọng số + 64 GiB (vLLM staging safetensors vào RAM host trước khi copy sang GPU). Có 70B → cần ≥ 373 GiB. g6e.48xlarge (1536 GiB) thoải mái.
7. **Pin CUDA/driver.** vLLM **v0.29.0** (09/09/2026) mặc định CUDA 13.0; bản `-cu129` cho driver cũ hơn. **Ghim tag VÀ ghi digest.** v0.29.0 đặt Model Runner V2 làm mặc định và deprecate MRV1 → đừng xây trên tính năng chỉ có ở MRV1.
8. **Nếu một số judge chạy FP8 còn số khác bf16 → đó là confound**, phải khai báo rõ trong bài.

---

## 8. Manifest tái lập (bắt buộc cho mỗi run)

Xuất `run_manifest.json` gồm:
- **vLLM version + image digest** (`vllm/vllm-openai:v0.29.0@sha256:…`, `vllm.__version__`, `torch.__version__`, `torch.version.cuda`). *Digest cụ thể: **CHƯA XÁC MINH** — lấy bằng `docker inspect` lúc pull.*
- **Revision SHA của từng model** (bảng §3), truyền bằng `--revision` và cho `snapshot_download`.
- **Tham số decoding nguyên văn:** `temperature=0`, `top_p=1`, `seed=0`, `max_tokens=64`, `n=1`, `repetition_penalty=1.0`, không stop string. Kèm `--max-num-seqs`.
- **`--max-model-len` và `--gpu-memory-utilization`** mỗi server + dòng `GPU KV cache size:` thực tế.
- **Quantization như một *treatment*:** checkpoint nào, scheme nào (`FP8_DYNAMIC`), `--kv-cache-dtype`, phiên bản `llmcompressor`.
- **Môi trường:** instance type, AZ, AMI ID, `nvidia-smi` (driver + CUDA), vCPU/RAM, layout EBS vs instance-store, ngày snapshot giá.
- **Topology + dòng lệnh đầy đủ:** GPU index, port, `--served-model-name`, TP size, `--enable-lora`/`--max-lora-rank`/adapter revision, `--enable-prefix-caching`.
- **Prompt rendering:** chat template chính xác từng model (và override nếu có), tokenizer revision, prompt đã render của một mẫu call — **đặc biệt ghi rõ cách gộp system message cho Gemma/Mistral**.
- **JSONL thô mỗi judge:** `request_id, model, prompt_hash, prompt_token_count, output_token_ids, finish_reason, usage, latency, timestamp`.

**Các mục CHƯA XÁC MINH — phải tự kiểm tra và ghi rõ trong bài:**
- digest image v0.29.0;
- **rank `r` của adapter SecAlign** (repo gated → đọc `adapter_config.json` rồi đặt `--max-lora-rank` cho khớp, **không đoán**);
- adapter PEFT này có load trực tiếp được trong vLLM 0.29.0 không;
- phiên bản driver của DLAMI cho g6e;
- FP8 W8A8 dense trên sm_120 cho 4 model này (nếu dùng g7e).

---

## 9. Nối vào code hiện có

Repo đã có sẵn `OpenAICompatJudge` (`src/aegis_agency/data/real_judges.py:341`) nhận `endpoint` + `model`. LiteLLM cho một `base_url` duy nhất, và `build_real_judges()` đã hỗ trợ map `models`/`endpoints` theo từng backbone (`real_judges.py:553-615`).

```yaml
judges:
  backend: openai_compat
  endpoints:
    default: http://127.0.0.1:4000/v1     # LiteLLM
  api_key_env: LITELLM_MASTER_KEY
  backbones: [llama-3.1-8b, metasecalign-8b, qwen2.5-7b, mistral-7b, gemma-2-9b, llama-3.3-70b]
  models:
    llama-3.1-8b:    judge-llama-3.1-8b-instruct
    metasecalign-8b: judge-metasecalign-8b
    qwen2.5-7b:      judge-qwen2.5-7b-instruct
    mistral-7b:      judge-mistral-7b-instruct
    gemma-2-9b:      judge-gemma-2-9b-it
    llama-3.3-70b:   judge-llama-3.3-70b-instruct
```

> **Giữ chuỗi `judge-*` GIỐNG HỆT ở cả ba nơi:** vLLM `--served-model-name`, LiteLLM `model_name`, và `models:` của experiment. Khi đó mọi phán quyết được log lại đều truy được về đúng artifact đã sinh ra nó — đây là điều reviewer AAMAS sẽ kiểm tra.

**Hai điểm cần sửa trong code trước khi chạy (đã nêu ở `code_tasks.md`):**
- **C-1:** hiện tại **không có song song hoá** — nhưng trên EC2 điều này *không còn quan trọng về wall-clock* (vLLM tự batch). Nó vẫn quan trọng để **giữ `--max-num-seqs` cố định và tái lập được**; nếu client gửi tuần tự thì batch luôn = 1 và kết quả **deterministic hơn**. → **Khuyến nghị: giữ tuần tự cho run chính thức** (chậm hơn nhưng tái lập tốt hơn), và chỉ dùng song song để pilot.
- **C-5:** trục `hardening: off|prompt|checkpoint` — nay chạy được `checkpoint` thật qua `judge-metasecalign-8b`.
