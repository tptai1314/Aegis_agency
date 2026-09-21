# YÊU CẦU CẤU HÌNH SERVER (GPU) — bài báo AAMAS 2027

> Tài liệu này để **chuyển cho giảng viên / đơn vị cấp hạ tầng**. Phần §1 là bản ngắn để copy-paste vào email. Phần §2 trở đi là chi tiết kỹ thuật và biện luận.
> Ngày soạn: 21/09/2026. Giá và thông số cần **kiểm tra lại tại thời điểm cấp phát**.
>
> 📌 **Nếu thầy/cô đã đề xuất `g5.xlarge` / `g4dn.2xlarge` + 50 GB đĩa:** đọc **`ec2_spec_comparison.md`** trước — cấu hình đó trích từ chính `docs/ec2_experiment_guide.md` của repo (viết cho pilot), và có 3 điểm cần sửa (đĩa, số GPU, RAM). File đó có sẵn **email trả lời copy-paste được** ở §6.
>
> **Ba yêu cầu tối thiểu không thể bỏ, bất kể phương án nào:** đĩa **≥ 250 GB** · RAM hệ thống **≥ 32 GiB** · **≥ 4 GPU** (hoặc chấp nhận chậm hơn nhiều với 1 GPU).

---

## §1. BẢN NGẮN ĐỂ GỬI EMAIL (copy-paste)

> Kính gửi thầy/cô,
>
> Em cần một server GPU để chạy thực nghiệm cho bài báo nộp hội nghị AAMAS 2027 (hạn nộp **08/10/2026**).
>
> **Cấu hình đề nghị:**
> - **GPU:** 4× NVIDIA L40S (44 GB VRAM mỗi card, tổng **~179 GB**) — hoặc bất kỳ node nào có **≥ 4 GPU, mỗi GPU ≥ 40 GB VRAM**
> - **CPU / RAM:** ≥ 48 vCPU, ≥ 384 GB RAM
> - **Ổ cứng:** ≥ 250 GB SSD (ưu tiên NVMe)
> - **Hệ điều hành:** Ubuntu 22.04 hoặc 24.04 + NVIDIA driver + **Docker có NVIDIA Container Toolkit**
> - **Truy cập:** SSH từ IP của em; **không cần mở cổng nào khác ra internet** (em sẽ dùng SSH tunnel)
>
> **Tương đương trên AWS:** `g6e.12xlarge` (4× L40S), on-demand, khu vực `us-east-1`, **~$10,49/giờ**.
> **Nếu không có L40S:** `p4de.24xlarge` (8× A100 80 GB) hoặc `g5.12xlarge` (4× A10G) — đều dùng được; `g5.12xlarge` rẻ hơn (~$5,67/giờ) nhưng em chỉ chạy được **3 backbone** thay vì 4 và phải nêu rõ hạn chế này trong bài.
> **Nếu ngân sách cho phép và thầy/cô muốn dư dả hơn:** `g6e.48xlarge` (8× L40S, ~$30,13/giờ). Em **không cần** cấu hình này — `g6e.12xlarge` đã đủ cho toàn bộ thực nghiệm.
>
> **Thời lượng:** khoảng **60–80 giờ máy**, rải rác trong 17 ngày. Em sẽ **stop instance khi không dùng** để tiết kiệm (ước tính **~$650–850** tổng ở mức $10,49/giờ của `g6e.12xlarge`).
>
> **Em tự lo:** tài khoản HuggingFace và quyền truy cập các model, toàn bộ code, dữ liệu benchmark, và cấu hình phần mềm.
>
> **Em cần từ thầy/cô:** instance đã được cấp, khả năng SSH, và xác nhận **quota GPU** của tài khoản AWS đã được tăng (mặc định của AWS là **0 vCPU** cho họ G/VT — nếu chưa tăng thì hồ sơ có thể mất tới 72 giờ).
>
> Em xin cảm ơn thầy/cô.

---

## §2. YÊU CẦU KỸ THUẬT (không phụ thuộc nhà cung cấp)

Nếu trường có cluster GPU riêng (không phải AWS), đây là các yêu cầu thực sự — con số ở §3 chỉ là cách ánh xạ.

| Hạng mục | Yêu cầu tối thiểu | Đề nghị | Vì sao |
|---|---|---|---|
| **GPU** | 4 GPU, mỗi GPU **≥ 40 GB VRAM** | **4 GPU × 44 GB** | Cần phục vụ **đồng thời** 4–5 model 7–9B cho uỷ ban thẩm phán. Xem §4 |
| **VRAM tổng** | ≥ 178 GB | **≥ 320 GB** | Trọng số + KV cache + overhead |
| **Kiến trúc GPU** | Không bắt buộc, nhưng **Ada Lovelace (L40S/L4) / Hopper (H100) / Blackwell** cho phép FP8 | L40S (sm_89) | A100 (sm_80) **không** hỗ trợ FP8 W8A8 — chỉ ảnh hưởng model 70B, không chặn |
| **CPU** | ≥ 48 vCPU | **48 vCPU** | vLLM staging + HTTP server, 4–5 tiến trình song song |
| **RAM** | ≥ 384 GB | **384 GB** | Quy tắc: ≥ 1,5 × tổng dung lượng trọng số + 64 GB |
| **Ổ cứng** | 250 GB SSD | **250 GB NVMe** | Trọng số 4 backbone + SecAlign ≈ **65 GB**, cộng Docker image ~10 GB |
| **Mạng** | Ra internet để tải model | — | Tải ~65–206 GB từ HuggingFace |
| **HĐH** | Ubuntu 22.04 / 24.04 | — | Tương thích vLLM tốt nhất |
| **Container** | Docker + NVIDIA Container Toolkit | — | Cách triển khai chuẩn của vLLM |

> **Không cần:** NVLink, InfiniBand, GPU interconnect tốc độ cao. Mỗi model nằm gọn trên **một** GPU (TP=1), nên băng thông liên GPU không phải yếu tố quyết định. PCIe là đủ.

---

## §3. ÁNH XẠ SANG INSTANCE AWS CỤ THỂ

Giá là snapshot **21/09/2026, khu vực us-east-1, on-demand** — cần kiểm tra lại.

### Phương án đề nghị — ✅ CHỐT

| | Thông số |
|---|---|
| **Instance** | **`g6e.12xlarge`** |
| GPU | 4× NVIDIA L40S, 44 GiB/card → **179 GB tổng** |
| vCPU / RAM | 48 vCPU / **384 GiB** |
| Ổ cứng cục bộ | 3.800 GB NVMe (ephemeral) + EBS gp3 gắn thêm |
| Giá | **~$10,49/giờ** (us-east-1, on-demand, snapshot 21/09/2026) |
| FP8 | ✅ có (sm_89) |

**Vì sao 4 GPU là đủ:** tổng trọng số 4 backbone (Llama-3.1-8B, Qwen2.5-7B, Mistral-7B, Gemma-2-9B) + adapter SecAlign-8B = **~60 GiB**; KV cache ~20 GiB; overhead ~6 GiB → **~86 GiB / 179 GiB = 48% sử dụng**. Còn dư nhiều. Uỷ ban `n = 1..7` chạy được vì code đã hỗ trợ **lặp backbone** (`backbones[k % len(backbones)]`, `real_judges.py:576`) — 7 thẩm phán trên 4 backbone, **không cần sửa code**.

### Các phương án thay thế

| Instance | GPU | VRAM tổng | Giá/giờ | Đánh giá |
|---|---|---|---|---|
| `g6e.48xlarge` | 8× L40S | 357 GB | ~$30,13 | ✅ **Dư dả** — thêm được Llama-3.3-70B và 2 GPU dự phòng. **Không cần thiết**; chỉ nhận nếu thầy/cô đề nghị |
| `g5.12xlarge` | 4× A10G | 89 GB | ~$5,67 | ⚠️ **Tầng ngân sách** — chỉ chạy được **3 backbone** (Gemma-2-9B cần 27,7 GiB > 22 GiB/card nên không lọt); phải khai báo hạn chế |
| `g6e.24xlarge` | 4× L40S | 178 GB | ~$15,07 | Cùng GPU với `g6e.12xlarge`, chỉ hơn CPU/RAM — **không cần**, 384 GiB RAM đã đủ |
| `p4de.24xlarge` | 8× A100 80 GB | 640 GB | ~$27–32 *(ước tính)* | ✅ Thay thế tốt về mặt khoa học nếu không có L40S |
| `p5.48xlarge` | 8× H100 | 640 GB | ~$55,04 | ✅ Dư sức, nhưng **đắt gấp 5 lần mà không nhanh hơn** — workload không bị chặn bởi compute |
| `g7e.48xlarge` | 8× RTX PRO 6000 | 768 GB | ~$33,14 | ⚠️ sm_120 còn lỗi vLLM đã biết |
| ❌ `p4d.24xlarge` | 8× A100 40 GB | 320 GB | ~$21,96 | Không FP8, 40 GB/card hơi chật |
| ❌ `g5.48xlarge` | 8× A10G | 178 GB, 22 GB/card | ~$16,29 | 22 GB/card không đủ cho Gemma-2-9B |
| ❌ `g4dn.2xlarge` | 1× T4 | 16 GB | ~$0,53 | **Không chạy được** model 8B ở bf16: trọng số đã chiếm 16,06 GB |
| ❌ `g5.xlarge` | 1× A10G | 22 GB | ~$1,01 | Chỉ 1 model, và **RAM hệ thống chỉ 16 GiB** — chật khi nạp model |

### Nếu không dùng AWS

Bất kỳ node nào có **8× L40S** hoặc **8× A100-80GB** hoặc **4–8× H100** đều tương đương. Các nhà cung cấp L40S rẻ hơn AWS đáng kể (~$0,85–1,90/GPU-giờ ở Lambda / RunPod / Vast.ai), nhưng **ưu tiên AWS** vì quota và capacity dự đoán được, phù hợp với hạn nộp cứng.

---

## §4. TẠI SAO CẦN NGẦN NÀY GPU (biện luận — câu hỏi thầy/cô sẽ hỏi)

Bài báo nghiên cứu **an toàn của một uỷ ban gồm nhiều LLM agent** ("thẩm phán") cùng kiểm tra một đầu ra. Thiết kế thực nghiệm đòi hỏi:

1. **5–7 thẩm phán chạy ĐỒNG THỜI.** Mỗi payload phải được cả uỷ ban đánh giá. Nếu phải nạp/rút model luân phiên, mỗi lần đổi mất **1–5 phút** × hàng nghìn payload → bất khả thi.
2. **Các thẩm phán phải KHÁC backbone** (Llama / Qwen / Mistral / Gemma…) — đây là một **câu hỏi nghiên cứu riêng** (RQ4): uỷ ban đồng nhất vs đa dạng ảnh hưởng thế nào tới tương quan lỗi. Nếu tất cả chạy cùng một model thì không đo được.
3. **Cần một thẩm phán đã được "hardening"** (Meta-SecAlign-8B) để so sánh — model này dùng chung base với Llama nên phục vụ qua multi-LoRA, tiết kiệm 1 slot.
4. **Cần chạy lại độc lập** để xác nhận kết quả (yêu cầu về tính tái lập của hội nghị).
5. **Mỗi model 7–9B ở bf16 chiếm ~15–18 GB trọng số**, cộng KV cache → **không thể nhồi 2 model 9B vào một card 44 GB**.
6. **8 GPU cho phép chạy mọi model ở bf16 không lượng tử hoá** — quan trọng vì nếu một số thẩm phán bị lượng tử hoá còn số khác thì không, đó là **confound** làm mất tính công bằng của so sánh.

**Uỷ ban dự kiến:** Llama-3.1-8B-Instruct · Meta-SecAlign-8B · Qwen2.5-7B-Instruct · Mistral-7B-Instruct-v0.3 · Gemma-2-9B-it · (tuỳ chọn) Llama-3.3-70B-Instruct.

---

## §5. PHẦN MỀM CẦN CÀI

| Thành phần | Phiên bản | Ghi chú |
|---|---|---|
| NVIDIA driver | ≥ 550 (khuyến nghị ≥ 580 nếu dùng CUDA 13) | Kiểm tra bằng `nvidia-smi` |
| Docker + NVIDIA Container Toolkit | Bản mới nhất | Cách triển khai chính |
| **vLLM** | **v0.29.0** (image `vllm/vllm-openai:v0.29.0-cu129`) | Ghim tag **và ghi digest** để tái lập |
| Python | 3.11 hoặc 3.12 | ⚠️ **Không dùng 3.14** — thiếu wheel cho `torch`/`vLLM` |
| LiteLLM | Bản mới nhất | Proxy OpenAI-compatible, gom 6–7 model về 1 endpoint |
| `hf_transfer` | — | Tăng tốc tải trọng số 3–5× |

**Em sẽ tự cài các phần mềm này** — chỉ cần server có sẵn driver NVIDIA, Docker và quyền `sudo`.

---

## §6. LƯU TRỮ & MẠNG

**Lưu trữ:**
- Trọng số model: **~65 GB** (uỷ ban 5 model) hoặc **~206 GB** (nếu thêm Llama-70B).
- Đề nghị **250 GB EBS gp3** với **1.000 MiB/s provisioned throughput** và **16.000 IOPS** (gp3 mặc định chỉ 125 MB/s và sẽ làm chậm việc nạp model).
- `g6e.48xlarge` có sẵn **7,6 TB NVMe cục bộ** — nhanh hơn EBS nhiều, dùng làm bản làm việc; EBS là bản gốc (NVMe cục bộ **mất dữ liệu khi stop**).

**Mạng / bảo mật:**
- **Chỉ mở cổng SSH (22)**, giới hạn theo IP của em.
- **Không mở** các cổng 8000–8004 (vLLM) hay 4000 (LiteLLM) ra internet — em sẽ dùng **SSH tunnel** khi cần.
- Không cần Elastic IP.
- API key và token lấy từ biến môi trường, **không** lưu vào file cấu hình hay commit lên git.

---

## §7. THỜI LƯỢNG & CHI PHÍ DỰ KIẾN

| Hạng mục | Thời gian | Ghi chú |
|---|---|---|
| Cài đặt + tải trọng số | 3–6 giờ | Tải không cần GPU nhưng vẫn tính giờ máy |
| Dựng 4 vLLM server + LiteLLM + kiểm tra | 4–8 giờ | |
| Gỡ lỗi, chạy thử, hiệu chuẩn | 15–25 giờ | Phần khó dự đoán nhất |
| Chạy thực nghiệm chính + ablation | 2–4 giờ | **Bản thân suy luận chỉ ~20 phút mỗi sweep** |
| Chạy lại để tái lập | 2–4 giờ | |
| Dự phòng | 15–25 giờ | |
| **Tổng** | **~60–80 giờ máy** | rải trong 17 ngày |

**Chi phí `g6e.12xlarge`:** 60–80 giờ × ~$10,49 ≈ **$650–850**.

> ⚠️ **Quan trọng:** để instance chạy 24/7 suốt 17 ngày sẽ tốn **~$4.280**. Em sẽ **stop instance** ngay sau mỗi phiên làm việc và chỉ start khi cần. Stop/start giữ nguyên ổ đĩa; chỉ mất thời gian nạp lại model (~10–20 phút).
>
> Nếu ngân sách hạn chế, phương án `g6e.24xlarge` (4× L40S) giảm chi phí còn **~$750–1.100**, đổi lại em phải thu hẹp uỷ ban và khai báo hạn chế đó trong bài.

---

## §8. PHÂN CHIA TRÁCH NHIỆM

**Em tự lo:**
- Tài khoản HuggingFace + xin quyền truy cập 5 model (đang chờ duyệt thủ công)
- Toàn bộ mã nguồn, pipeline dữ liệu (đã xong: 3.054 payload thật), cấu hình thực nghiệm
- Cài đặt phần mềm trong container, chạy và giám sát thực nghiệm
- Soạn thảo bài báo và nộp

**Cần từ thầy/cô / đơn vị hạ tầng:**
- [ ] Instance đã được cấp theo §2–§3
- [ ] **Xác nhận quota GPU của tài khoản AWS đã được tăng** — mặc định của AWS là **0 vCPU** cho họ "G and VT"; `g6e.48xlarge` cần **192 vCPU** trong bucket này. Hồ sơ tăng quota có thể mất **vài giờ đến 72 giờ**. Nếu tài khoản đã có quota thì bỏ qua bước này
- [ ] Thông tin SSH (host, user, key) và quyền `sudo`
- [ ] Xác nhận ngân sách cho ~$650–850 (hoặc phương án nhỏ hơn ở §7)
- [ ] Ưu tiên chọn AZ đã test có sẵn capacity — `InsufficientInstanceCapacity` rất phổ biến với họ GPU; nếu có thể, tạo **On-Demand Capacity Reservation** để giữ chỗ

---

## §9. PHƯƠNG ÁN NẾU KHÔNG ĐƯỢC CẤP GPU NHƯ YÊU CẦU

Theo thứ tự ưu tiên:

1. **4× L40S (g6e.24xlarge)** — chạy được uỷ ban 4–5 thẩm phán; phải dùng FP8 cho một số model và **khai báo confound** trong bài. Vẫn là bài 8 trang hợp lệ.
2. **8× A100-80GB (p4de)** — hoàn toàn tương đương về mặt khoa học, không mất gì (chỉ không có FP8, mà ta không cần).
3. **Thuê GPU ngoài** (Lambda / RunPod / Vast.ai): 8× L40S khoảng **$7–15/giờ**, rẻ hơn AWS 2,5–4 lần. Cần thầy/cô duyệt chi phí qua thẻ hoặc tài khoản của trường.
4. **Thu hẹp bài**: dùng API thay vì self-host (OpenAI + Anthropic + một model mở), chi phí chỉ vài chục USD, nhưng **mất khả năng chạy checkpoint hardening thật (SecAlign)** và **mất tấn công white-box (JudgeDeceiver)** — phải hạ tuyên bố của bài.
5. **Lùi hạn**: giữ bản dài 47 trang vốn đã viết cho **IEEE TDSC / Computers & Security**, bổ sung số liệu rồi nộp tạp chí thay vì hội nghị.

---

## §10. BA VIỆC PHẢI LÀM NGAY, KHÔNG CHỜ SERVER

Ba việc này không phụ thuộc vào server và **có thể mất nhiều ngày**, nên phải bắt đầu hôm nay:

1. **Xin tăng quota GPU** (nếu tài khoản chưa có) — tới 72 giờ.
2. **Xin quyền truy cập HuggingFace** cho 5 model gated — duyệt **thủ công**, Meta là chậm nhất.
3. **Email ban tổ chức AAMAS** (`aamas2027pcs@gmail.com`) — hạn đăng ký tác giả trên OpenReview (17/09/2026) **đã trôi qua**; cần xác nhận có được đăng ký muộn không.

Trong lúc chờ, công việc **không cần GPU** vẫn tiến được: tải trọng số, viết lại bản thảo 8 trang, chuẩn bị pipeline. Dữ liệu benchmark **đã xong** (3.054 payload).
