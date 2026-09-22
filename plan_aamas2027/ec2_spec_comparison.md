# SO SÁNH HAI CẤU HÌNH EC2 — cấu hình nào hợp lý hơn?

Ngày 21/09/2026. Mọi thông số dưới đây lấy từ [tài liệu AWS chính thức về instance tăng tốc](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html) (đã fetch và đọc trực tiếp).

---

## 0. TRƯỚC TIÊN: cấu hình thầy/cô đưa ra đến từ đâu?

Cấu hình đó **không phải do thầy/cô nghĩ ra** — nó **trích nguyên văn từ chính repo của bạn**:

> `docs/ec2_experiment_guide.md:17-18`
> *"Recommended: a GPU instance (e.g. `g5.xlarge` / `g4dn.2xlarge`+, NVIDIA driver preinstalled, CUDA 12.x, ≥ 16 GB VRAM, ≥ 50 GB disk), Ubuntu 22.04/24.04, SSH key inbound."*

**Điều này quan trọng vì hai lý do:**

1. Thầy/cô **không sai** — thầy/cô đang đọc đúng tài liệu của bạn. Đừng đối đầu; hãy giải thích rằng tài liệu đó được viết cho một phạm vi nhỏ hơn.
2. **Tài liệu trong repo đã lỗi thời so với thiết kế bài báo.** Chính file đó, ở dòng 77–81, tự nói rằng cấu hình một GPU chỉ cho **uỷ ban đồng nhất** (mọi thẩm phán dùng chung một backbone): *"All `n_judges` judges therefore share that backbone — the committee is homogeneous by default, which is fine for a first Table 5/6 run."*

Tức là: repo viết hướng dẫn đó cho **lần chạy đầu tiên để kiểm tra đường ống**, không phải cho thực nghiệm của bài báo. Khoảng cách này cần được nói rõ.

---

## 1. THÔNG SỐ ĐÃ XÁC MINH

| Instance | RAM hệ thống | vCPU | GPU | **VRAM** |
|---|---|---|---|---|
| `g4dn.2xlarge` (cấu hình thầy/cô) | 32 GiB | 8 | 1× NVIDIA T4 | **16 GiB** |
| `g5.xlarge` (cấu hình thầy/cô) | **16 GiB** | 4 | 1× NVIDIA A10G | **22 GiB** |
| `g5.2xlarge` | 32 GiB | 8 | 1× A10G | 22 GiB |
| `g5.12xlarge` | 192 GiB | 48 | **4× A10G** | **89 GiB** |
| `g5.24xlarge` | 384 GiB | 96 | 4× A10G | 89 GiB |
| `g6e.12xlarge` | 384 GiB | 48 | 4× L40S | 178 GiB |
| `g6e.48xlarge` (đề nghị của tôi) | **1.536 GiB** | 192 | **8× L40S** | **357 GiB** |

> ⚠️ **Chú ý `g5.xlarge`: chỉ có 16 GiB RAM hệ thống.** vLLM phải staging trọng số qua RAM host. Một model 8B bf16 là ~16 GB trọng số → **16 GiB RAM là cực kỳ chật**, dễ OOM hoặc swap. Đây là điểm yếu ít ai để ý của cấu hình thầy/cô đưa ra.

---

## 2. KẾT LUẬN NGẮN

| | Cấu hình thầy/cô (`g5.xlarge` / `g4dn.2xlarge`) | Cấu hình tôi đề nghị (`g6e.48xlarge`) |
|---|---|---|
| Chạy smoke test đường ống | ✅ **Đủ** | ✅ Thừa |
| Chạy uỷ ban **đồng nhất** (n thẩm phán cùng 1 backbone) | ✅ Đủ (nhưng `g5.xlarge` chật RAM) | ✅ |
| Chạy uỷ ban **đa dạng** (5 backbone khác nhau, song song) | ❌ **Không** | ✅ |
| **RQ4 — tương quan lỗi: đồng nhất vs đa dạng** | ❌ **Không làm được** | ✅ |
| Ablation "− hardened judges" với SecAlign thật | ⚠️ Chỉ khi chạy luân phiên | ✅ |
| JudgeDeceiver (tấn công white-box) | ⚠️ Rất chật | ✅ |
| Dung lượng đĩa | ❌ **50 GB không đủ** | ✅ |
| **Kết luận** | **Đủ cho pilot, KHÔNG đủ cho bài báo** | **Đủ cho bài báo** |

**→ Cấu hình tôi gửi hợp lý hơn cho mục tiêu nộp bài. Nhưng cấu hình thầy/cô không phải rác — nó đúng cho giai đoạn pilot.**

---

## 3. BA LÝ DO CỤ THỂ, XẾP THEO MỨC NGHIÊM TRỌNG

### ❌ Lý do 1 — Đĩa 50 GB chắc chắn không đủ (dễ sửa nhất)

| Hạng mục | Dung lượng |
|---|---|
| 5 model 7–9B (bf16, đã `--exclude original/*`) | **~65 GB** |
| Docker image vLLM (`vllm/vllm-openai:v0.29.0`) | ~10 GB |
| Bộ benchmark (đã build xong) | ~35 MB |
| Kết quả + cache verdict | ~1–5 GB |
| **Tổng tối thiểu** | **~80 GB** |
| **Đề nghị (có dư địa)** | **250 GB** |
| Nếu thêm Llama-3.3-70B | **+141 GB → cần 400 GB** |

`g4dn.2xlarge` mặc định chỉ có **225 GB NVMe** cục bộ nên có thể vẫn chạy được, nhưng nếu ai đó gắn EBS 50 GB thì **không thể**. **Đây là chỗ cần sửa chắc chắn.**

### ❌ Lý do 2 — Một GPU 16–22 GB không chứa nổi uỷ ban đa dạng

- `g4dn.2xlarge` có T4 **16 GiB**. Llama-3.1-8B ở bf16 cần **16,06 GB trọng số** → **không còn chỗ cho KV cache**. Thực tế **không chạy được** ở bf16; buộc phải lượng tử hoá (AWQ/GPTQ) hoặc dùng model ≤3B.
- `g5.xlarge` có A10G **22 GiB** → chứa được **một** model 8B bf16 (14,96 GiB) + KV, nhưng **không hai**.

Bài báo cần **5–7 thẩm phán khác backbone chạy đồng thời**. Đây không phải yêu cầu cho vui:

> **RQ4 là một trong bốn đóng góp của bài** — và sau khi phát hiện bị scoop (xem `positioning_and_related_work.md`), nó trở thành **điểm khác biệt quan trọng nhất** so với RoPoLL và "Many Minds, One Verdict". Không chạy được RQ4 ⇒ mất luôn lý do tồn tại của bài.

### ⚠️ Lý do 3 — RAM 16 GiB của `g5.xlarge` (dễ bỏ sót)

vLLM staging trọng số qua RAM host trước khi copy sang GPU. Model 8B ~16 GB mà RAM chỉ 16 GiB → rủi ro OOM khi nạp. `g5.2xlarge` (32 GiB) hoặc `g4dn.2xlarge` (32 GiB) an toàn hơn.

---

## 4. ĐIỀU CẦN NÓI CÔNG BẰNG: MỘT GPU VẪN CÓ THỂ LÀM ĐƯỢC — nếu đổi cách chạy

Tôi không muốn nói quá rằng "phải có 8 GPU". Về mặt kỹ thuật, **một GPU vẫn tạo được uỷ ban đa dạng** bằng cách đổi thứ tự chạy:

**Cách hiện tại (payload-major):** với mỗi payload, hỏi cả 5 thẩm phán. → cần 5 model nằm đồng thời trong VRAM.

**Cách thay thế (model-major):** chạy **cả bộ payload với model A** → đổi model → chạy cả bộ với model B → … → ghép lại. Verdict cache trong repo vốn đã được thiết kế theo khoá `(payload_id, judge_id, backbone)`, nên **về nguyên tắc** ghép được.

**Cái giá phải trả:**
1. **Phải sửa code** (việc **C-15** mới): khoá cache hiện gắn `judge_id`, mà khi chạy từng backbone một thì `judge_id` luôn = 0 → cache không ghép đúng thành uỷ ban 5 người. Cần đổi khoá cache sang `(payload_id, backbone)` rồi suy ra `judge_id` lúc lắp uỷ ban. **~4–6 giờ công.**
2. **Wall-clock dài hơn nhiều:** 5 lượt chạy × (thời gian suy luận + 2–5 phút đổi model). Trên A10G ước tính **6–12 giờ** cho một lượt đầy đủ, so với **~15 phút** trên `g6e.48xlarge`. Với hạn nộp 17 ngày và cần **chạy lại để tái lập**, đây là rủi ro thật.
3. **Không thể chạy uỷ ban 7 người cùng lúc** → RQ5 (biên security-vs-cost theo n) bị hạn chế.

**→ Kết luận:** một GPU là **phương án dự phòng khả thi**, không phải phương án tốt. Nếu chỉ được cấp một GPU thì vẫn nên nhận, nhưng phải xin **thêm đĩa** và biết trước là mất thêm ~1 ngày công.

---

## 5. ✅ KẾT LUẬN CUỐI CÙNG — CHỐT `g6e.12xlarge`

> Tôi **hạ khuyến nghị của chính mình** từ 8 GPU xuống 4 GPU. Lý do ở §5.1.

### 5.1 Phương án chốt

| | |
|---|---|
| **Instance** | **`g6e.12xlarge`** |
| GPU | **4× NVIDIA L40S**, 44 GiB/card → **179 GB VRAM** |
| vCPU / RAM | 48 vCPU / **384 GiB** |
| Đĩa | 3.800 GB NVMe cục bộ + **EBS gp3 250 GB** (provisioned 1.000 MiB/s, 16.000 IOPS) |
| Giá | **~$10,49/giờ** (us-east-1, on-demand, snapshot 21/09/2026 12:51 UTC) |
| Chi phí dự án | 60–80 giờ máy → **~$650–850** |

### 5.2 Vì sao 4 GPU là đủ (đã kiểm tra bằng số)

| Hạng mục | Dung lượng |
|---|---|
| Trọng số: Llama-3.1-8B (14,96) + Qwen2.5-7B (14,19) + Mistral-7B (13,50) + Gemma-2-9B (17,22) + SecAlign-8B LoRA (0,28) | **60,15 GiB** |
| KV cache @ `--max-model-len 2048` × 16 seq | **20,25 GiB** |
| Overhead vLLM (CUDA context, activation, ~1,5 GiB × 5 tiến trình) | **~6 GiB** |
| **Tổng** | **~86 GiB / 179 GiB = 48%** |

→ **Còn dư hơn một nửa.** Bố trí: GPU0 = Llama + SecAlign LoRA, GPU1 = Qwen, GPU2 = Mistral, GPU3 = Gemma.

**Uỷ ban `n = 1..7` chạy được mà không cần sửa code** — đã kiểm chứng `src/aegis_agency/data/real_judges.py:576`: `backbone = backbones[k % len(backbones)]`. Với 4 backbone và `n_judges=7`, thẩm phán 5–6 lặp lại backbone 0–1 với `judge_id` khác nhau, nên verdict cache `(payload_id, judge_id, backbone)` vẫn tách bạch đúng.

### 5.3 Vì sao tôi hạ từ 8 GPU xuống 4 GPU

| Tiêu chí | `g6e.48xlarge` (8 GPU) | `g6e.12xlarge` (4 GPU) |
|---|---|---|
| Đáp ứng RQ1–RQ6 | ✅ | ✅ **như nhau** |
| Thời gian một sweep | ~15 phút | ~20 phút — **không đáng kể** |
| Chi phí | $30,13/giờ | **$10,49/giờ (rẻ 1/3)** |
| vCPU cần xin quota | 192 | **48 (dễ được duyệt hơn nhiều)** |
| Rủi ro `InsufficientInstanceCapacity` | Cao hơn | Thấp hơn |
| Thứ 8 GPU mang lại gì | Llama-3.3-70B + 2 GPU dự phòng | — |

**Llama-70B và GPU dự phòng là "nice-to-have", không phải điều kiện để bài đủ mạnh.** Với hạn nộp 17 ngày và việc thầy/cô kiểm soát ngân sách, một yêu cầu **rẻ hơn 3 lần và dễ duyệt hơn** có giá trị thực tế lớn hơn phần dư kia.

### 5.4 Thứ tự ưu tiên khi thương lượng

| # | Phương án | Giá/giờ | Đánh giá |
|---|---|---|---|
| **1** | **`g6e.12xlarge`** (4× L40S) | ~$10,49 | ✅ **CHỐT** — đủ cho cả 4 đóng góp |
| 2 | `g6e.48xlarge` (8× L40S) | ~$30,13 | Nhận nếu thầy/cô đề nghị; **không cần xin** |
| 3 | `g5.12xlarge` (4× A10G) | ~$5,67 | Tầng ngân sách — chỉ **3 backbone** (Gemma-2-9B cần 27,7 GiB > 22 GiB/card nên không lọt) |
| 4 | `g5.2xlarge` (1× A10G) + đĩa ≥250 GB | ~$1,01 | Dự phòng — cần sửa code (C-15), chậm hơn 6–12 giờ/lượt |
| ❌ | `g5.xlarge` / `g4dn.2xlarge` | — | **Không dùng được** — xem §3 lý do 1–3 |

### 5.5 Ba yêu cầu tối thiểu không thể bỏ (bất kể phương án)

1. **Đĩa ≥ 250 GB** — 50 GB không đủ (cần ~80 GB tối thiểu)
2. **RAM hệ thống ≥ 32 GiB** — `g5.xlarge` chỉ có 16 GiB, quá chật
3. **GPU ≥ 4** (hoặc chấp nhận chậm hơn nhiều với 1 GPU + sửa code)

### 5.6 Về các yêu cầu khác trong đề xuất của thầy/cô

| Yêu cầu | Đánh giá |
|---|---|
| Ubuntu 22.04/24.04 | ✅ Giữ nguyên, đúng |
| **CUDA 12.x** | ✅ **Hoàn toàn ổn** — dùng image `vllm/vllm-openai:v0.29.0-cu129` |
| NVIDIA driver sẵn có | ✅ Đúng |
| Mở SSH inbound key | ✅ Đúng — và **không mở cổng nào khác** |
| `g5.xlarge` / `g4dn.2xlarge` | ❌ Cần nâng lên 4 GPU |
| VRAM ≥ 16 GB | ⚠️ Đúng nhưng **không đủ** — cần **≥ 40 GB/card** để chứa model 9B |
| Disk ≥ 50 GB | ❌ **Không đủ** — cần **≥ 250 GB** |

---

## 6. EMAIL TRẢ LỜI THẦY/CÔ (copy-paste)

> Kính gửi thầy/cô,
>
> Em cảm ơn thầy/cô đã xem qua. Cấu hình thầy/cô trích dẫn đúng là tài liệu hướng dẫn trong repo của em (`docs/ec2_experiment_guide.md`) — nhưng tài liệu đó em viết cho **lần chạy thử đường ống ban đầu**, với một uỷ ban chỉ dùng **một** backbone. Sau khi hoàn thiện thiết kế thực nghiệm, yêu cầu đã cao hơn, và em xin phép trình bày lý do.
>
> **Vấn đề thứ nhất — đĩa 50 GB không đủ.** Cần nạp 5 model 7–9B, mỗi model ~15 GB ở bf16 (tổng ~65 GB), cộng Docker image vLLM (~10 GB). Em xin **≥ 250 GB**.
>
> **Vấn đề thứ hai — một GPU 16–22 GB không đủ.** Bài cần một uỷ ban gồm **4–5 thẩm phán khác backbone** (Llama / Qwen / Mistral / Gemma / SecAlign) để trả lời câu hỏi nghiên cứu về **tương quan lỗi giữa các model**. Đây là một trong bốn đóng góp của bài. Xin lưu ý thêm: `g4dn.2xlarge` (T4 16 GB) thực tế **không chạy được** model 8B ở bf16 vì trọng số đã chiếm 16,06 GB; và `g5.xlarge` chỉ có **16 GiB RAM hệ thống**, rất chật khi nạp model.
>
> **Đề nghị của em: `g6e.12xlarge`** (4× NVIDIA L40S, 179 GB VRAM, 48 vCPU, 384 GiB RAM) + **đĩa 250 GB** — khoảng **$10,49/giờ**. Em đã tính toán và 4 GPU là **đủ và dư**: tổng trọng số 4 model + adapter chỉ chiếm ~60 GB, cộng KV cache và overhead là ~86 GB trên 179 GB (khoảng 48% sử dụng). Thời gian một lượt thực nghiệm đầy đủ khoảng **20 phút**.
>
> **Phương án thay thế:**
> - Nếu **không có L40S**: `p4de.24xlarge` (8× A100 80 GB) dùng được, tương đương về mặt khoa học.
> - Nếu **hạn chế ngân sách**: `g5.12xlarge` (4× A10G, 89 GB VRAM) rẻ hơn (~$5,67/giờ), nhưng em chỉ chạy được **3 backbone** thay vì 4 (model Gemma-2-9B cần 27,7 GB nên không vừa card 22 GB). Em sẽ nêu rõ hạn chế này trong bài.
> - Em **không cần** `g6e.48xlarge` (8 GPU, ~$30/giờ) — cấu hình đó chỉ thêm một model lớn 70B và GPU dự phòng, không cần thiết cho bài.
>
> **Ba yêu cầu tối thiểu, không phụ thuộc phương án:** đĩa **≥ 250 GB** · RAM ≥ 32 GiB · **GPU ≥ 4** (hoặc chấp nhận chậm hơn nhiều với 1 GPU).
>
> Phần còn lại trong đề xuất ban đầu của em vẫn giữ nguyên: Ubuntu 22.04/24.04, NVIDIA driver sẵn có, Docker + NVIDIA Container Toolkit, và **chỉ mở cổng SSH** (em sẽ dùng SSH tunnel, không cần mở cổng GPU ra internet). Về CUDA: **CUDA 12.x là hoàn toàn ổn** — em sẽ dùng image `vllm/vllm-openai:v0.29.0-cu129`.
>
> Em xin cảm ơn thầy/cô.

---

## 7. MỘT VIỆC CẦN SỬA TRONG REPO

Tài liệu `docs/ec2_experiment_guide.md:17-18` đang gây hiểu nhầm và **đã gây hiểu nhầm thật**. Cần sửa thành hai mục rõ ràng:

```markdown
## 1. Provision the server
### Pilot / plumbing-only (homogeneous committee, one backbone)
e.g. `g5.2xlarge` (1x A10G 22 GB, 32 GiB RAM, >= 250 GB disk)
### Paper-level experiments (diverse committee, 4-5 backbones served concurrently)
Recommended: `g6e.12xlarge` (4x L40S, 179 GiB VRAM, 384 GiB RAM, >= 250 GB disk)
Alternative: `p4de.24xlarge` (8x A100 80GB) | Budget: `g5.12xlarge` (4x A10G, 89 GiB, 3 backbones only)
See plan_aamas2027/ec2_runbook.md and plan_aamas2027/ec2_spec_comparison.md.
```

Đây cũng là **task T0.11** cần thêm vào `task_board.csv`.
