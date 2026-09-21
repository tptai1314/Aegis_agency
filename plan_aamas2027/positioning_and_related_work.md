# ĐỊNH VỊ LẠI & TIỀN CÔNG TRÌNH — AAMAS 2027

> ⚠️ **Đây là tài liệu quan trọng nhất trong bộ kế hoạch.** Kết quả rà soát ngày 21/09/2026 cho thấy **bài đã bị "scoop" một phần**. Nếu nộp với cách định vị hiện tại trong `manuscript/`, reviewer AAMAS sẽ từ chối vì novelty. Phần dưới đây nói rõ **cái gì đã bị lấy**, **cái gì còn lại**, và **phải viết lại như thế nào**.

---

## 1. Ba bài phải đọc trước khi viết lại (đã xác minh trực tiếp)

### 1.1 🔴 RoPoLL — nguy hiểm nhất về mặt cơ chế
**RoPoLL: Robust Panel of LLM Judges** — Acharya, Pan, Verkhovsky.
[arXiv:2606.30931](https://arxiv.org/abs/2606.30931), cs.AI/cs.MA/cs.LG, nộp **29/06/2026**, có mặt tại [ICML 2026](https://icml.cc/virtual/2026/74508).

**Nội dung (xác minh từ abstract):** hình thức hoá LLM Jury theo **mô hình nhiễm bẩn Huber**; chứng minh PoLL có **bias không bị chặn** dưới bất kỳ nhiễm bẩn dương nào, bất kể cỡ jury; **thay hàm tổng hợp bằng một robust mean estimator, cụ thể hoá là geometric median** — breakdown point tối ưu 1/2; có finite-sample bound + minimax lower bound; thực nghiệm trên **13 judge open-weight (4B–675B)**, 3 benchmark, 4 chế độ nhiễm bẩn tới 50%.

**Tác động trực tiếp lên bài của bạn:**
| Thành phần trong bài bạn | Trạng thái sau RoPoLL |
|---|---|
| `gmed` (geometric median) làm aggregator chống judge Byzantine | ❌ **ĐÃ LÀ TIỀN CÔNG TRÌNH** |
| Ý tưởng "thay coordinator tin cậy bằng tổng hợp robust" | ❌ **ĐÃ LÀ TIỀN CÔNG TRÌNH** |
| breakdown point / ngưỡng `f < n/2` | ❌ Đã được RoPoLL chứng minh (1/2) |
| **Pha loãng: RoPoLL nhiễm bẩn *điểm số* theo thống kê — không có kênh prompt injection, không có payload isolation, không có quyết định allow/block** | ✅ **CÒN NGUYÊN** |

### 1.2 🔴 "Many Minds, One Verdict" — nguy hiểm nhất về mặt động cơ
**Many Minds, One Verdict: The Limits of LLM Consensus as a Security Gate** — Adrian Asher.
Artifact tái lập: [Zenodo 10.5281/zenodo.21235385](https://zenodo.org/records/21235385), công bố **07/07/2026**, CC-BY-4.0. Code: `github.com/cloudygeek/p20_manyminds`.
Trạng thái: **bản thân bài báo chưa tìm thấy venue/DOI/arXiv** — mới xác minh được artifact. (Vẫn phải trích dẫn và phải tính đến.)

**Nội dung (xác minh từ mô tả artifact):** đặt LLM judge làm **cổng an ninh** phê duyệt/chặn tool call của agent. Kết quả: phán quyết **không tất định** (12–55% input y hệt byte bị lật), **nhạy cảm đầu vào** (7/12 hijack bị lật dưới phép biến đổi bảo toàn nghĩa, kể cả chỉ sửa khoảng trắng), **63% sai một cách ổn định nhưng tự tin**. Và: **panel đồng thuận đa tác tử bị chặn trên bởi tương quan lỗi — chỉ còn ~2 phiếu hiệu dụng**; thất bại là do **bias, không phải variance**, nên **một prompt được thiết kế tốt ngang bằng panel 8 vendor**; **không cấu hình nào đạt điểm vận hành dùng được (>90% tấn công bị chặn ở <10% benign bị chặn)**.

**Tác động trực tiếp:**
| Thành phần trong bài bạn | Trạng thái |
|---|---|
| Mệnh đề 3 (`Var(Z̄) = μ(1−μ)[(1−ρ)/(n−f) + ρ]`, thêm judge vô ích khi `ρ>0`) | ⚠️ **Đã được quan sát thực nghiệm** (họ đo được "~2 phiếu hiệu dụng"). Bạn còn **phiên bản hình thức** — nhưng phải nói rõ họ đã quan sát trước |
| Động cơ "cần panel đa dạng để chống lỗi tương quan" | ❌ Đã có người nói |
| **Họ KHÔNG có: judge Byzantine/thông đồng, tổng hợp robust, isolation, bảo đảm quyết định** | ✅ **CÒN NGUYÊN** |

> ✅ **Cách biến mối đe doạ thành động cơ:** "Many Minds, One Verdict" chứng minh **panel đồng thuận thường (honest, chỉ đa dạng) là bị chặn trên**. Bài của bạn trả lời câu hỏi tiếp theo mà họ để mở: **tổng hợp robust + cô lập payload sửa được phần nào của chặn trên đó, và phần nào thì không.** Đây là một khung "kế thừa kết quả âm" rất mạnh và rất AAMAS. **Hãy mở bài bằng đúng câu này.**

### 1.3 🟠 Các bài phải trích dẫn và phân biệt

| Bài | ID | Mức độ gần | Phải phân biệt điều gì |
|---|---|---|---|
| **RobustJudge** — LLMs Cannot Reliably Judge (Yet?) | [arXiv:2506.09443](https://arxiv.org/abs/2506.09443) v3 06/08/2026 | Trung bình–cao | 15 attack × 8 defense × 13 model. Đây là **bản đồ không gian tấn công judge** — bạn phải định vị mình *ở trên* nó: họ đánh giá **một judge**, bạn đánh giá **cơ chế tổng hợp nhiều judge** |
| **Certifiable Semantic Agreement Among LLM Agents** | arXiv:2606.07316 — ⚠️ **chưa tự xác minh** | Cao–trung bình | Chứng chỉ BFT `2f+1` trên uỷ ban LLM agent; có **bổ đề containment tiêu cực** (đa số bọc chứng chỉ không có lợi thế coverage). Phải đọc và trả lời trực tiếp |
| **FLOWSEAL** — Confuse the Model, Control the Flow | arXiv:2609.14003 — ⚠️ **chưa tự xác minh** | Thấp–TB nhưng chiến lược | Luận điểm *"khi cưỡng chế là một phán đoán LLM đưa ra trên chính ngữ cảnh mà kẻ tấn công kiểm soát, thì cơ chế cưỡng chế và bề mặt tấn công trùng nhau"* — **gần như chính xác là lập luận payload isolation của bạn**. Trích dẫn như phát biểu gần nhất của nguyên lý, rồi **mở rộng từ privacy sang verdict integrity** |
| **SentinelAgent** | [arXiv:2505.24201](https://arxiv.org/abs/2505.24201) | Trung bình | Phát hiện bất thường đồ thị thực thi, có nêu "multi-agent collusion" — baseline phát hiện, không phải tổng hợp robust |
| **Audit the Whisper** | [arXiv:2510.04303](https://arxiv.org/abs/2510.04303) | Trung bình | Thông đồng steganographic; có **ColludeBench-v0** (gồm cả peer review). Dùng cho mô hình mối đe doạ thông đồng |
| **Evaluating Multi-Agent Defences Against Jailbreaking** | [arXiv:2506.23576](https://arxiv.org/abs/2506.23576) | Trung bình | Tái lập cấu hình 1/2/3 agent của AutoDefense — baseline tái lập của bạn |
| **PIArena** | [arXiv:2604.08499](https://arxiv.org/abs/2604.08499) + [HF `sleeepeer/PIArena`](https://huggingface.co/datasets/sleeepeer/PIArena) | Công cụ | Harness tấn công injection **thích ứng** — dùng cho RQ3 |
| **EvalHack** | *Information* (MDPI) 17(3):297 — ⚠️ **HTTP 403, chưa đọc được abstract** | ⚠️ **CHƯA RÕ** | Tên gọi "LLM exam-grading panel" **rất gần** setting của bạn. **Phải đọc trước khi nộp** |

---

## 2. ĐÓNG GÓP CÒN LẠI — viết lại cho an toàn

### ❌ Những câu KHÔNG được viết nữa (đã bị lấy)
- "We are the first to apply Byzantine-robust aggregation to LLM judge verdicts."
- "We replace the trusted coordinator with a Byzantine-robust aggregator." (như **đóng góp**, chỉ được nêu như **thành phần kế thừa**)
- Bất kỳ tuyên bố novelty nào thuần tuý về `cmed`/`gmed`/Krum trên phán quyết judge.
- "Diverse committees are needed because errors correlate" (như **phát hiện mới**).

### ✅ Bốn đóng góp còn nguyên và có thể bảo vệ

| # | Đóng góp | Vì sao vẫn mới | Câu viết an toàn |
|---|---|---|---|
| **C1** | **Mô hình mối đe doạ kết hợp**: judge Byzantine/thông đồng **VÀ** injection nằm trong chính payload mà judge phải đọc, **và tương tác giữa hai kênh** | RoPoLL chỉ nhiễu điểm số (không có kênh văn bản); Many Minds chỉ panel honest (không có judge đối kháng) | *"Prior work studies either statistical corruption of judge scores or honest-but-diverse panels. We study the joint threat: an adversary who both owns judges and writes the text they read."* |
| **C2** | **Bảo đảm ở mức QUYẾT ĐỊNH** (`γ > C_α·r ⇒ D(û) = D(u*)`), không phải ở mức ước lượng điểm | RoPoLL chặn **bias của điểm**; bạn chặn **quyết định allow/block** — đối tượng khác, và là đối tượng mà triển khai thực sự dùng | *"RoPoLL bounds score bias; we bound the decision, which is what a gate actually acts on. The two guarantees are not comparable and both are needed."* |
| **C3** | **Payload isolation như cơ chế *định giá* injection vào ngân sách Byzantine** — biến tấn công "miễn phí" thành tấn công bị chặn bởi `f` | FLOWSEAL phát biểu nguyên lý cho **tính bí mật**; bạn **hình thức hoá và đo** nó cho **tính toàn vẹn phán quyết** (`ε`, Mệnh đề 2) | *"FLOWSEAL articulates the principle for confidentiality. We formalise it for verdict integrity and measure the residual leakage ε on real hardened judges."* |
| **C4** | **Đo `r, γ, ρ, ε` trên judge LLM thật** — nghiên cứu "bảo đảm có hiệu lực hay không" | Chưa ai làm. RoPoLL đo bias trên benchmark chấm điểm; bạn đo **các đại lượng quyết định tính đúng của một định lý** trên pipeline phòng thủ thật | *"The theorem is conditional; we convert its conditions into measured quantities and report where they hold and where they fail."* |

**Câu chốt một dòng cho phần Introduction:**
> *"Consensus panels and robust score aggregation have each been studied. We ask what happens when the adversary attacks both at once — corrupting committee members and writing the text the committee must read — and we report precisely how far a robust aggregation guarantee survives, and where it stops."*

---

## 3. VIỆC PHẢI LÀM NGAY (bổ sung vào kế hoạch chính)

- [ ] **POS-1 — Đọc toàn văn RoPoLL** ([arXiv:2606.30931](https://arxiv.org/abs/2606.30931)) và ghi ra: họ dùng aggregator gì, benchmark nào, corrupt model nào, breakdown point. → Viết 1 đoạn so sánh trực tiếp.
- [ ] **POS-2 — Tải artifact Many Minds** ([Zenodo 21235385](https://zenodo.org/records/21235385), 465 KB, offline, CC-BY) và đọc bảng "effective votes". → Đối chiếu với Mệnh đề 3 của mình.
- [ ] **POS-3 — Tìm và đọc bài đầy đủ của "Many Minds, One Verdict"** (chưa có venue). Nếu không tồn tại bản có venue, trích dẫn artifact — nhưng **phải** nêu.
- [ ] **POS-4 — Đọc RobustJudge** ([arXiv:2506.09443](https://arxiv.org/abs/2506.09443) v3) → chọn 2–3 attack từ ma trận 15 attack để chạy như **tấn công ngoài** lên uỷ ban (tăng độ tin cậy).
- [ ] **POS-5 — Xác minh 2 bài còn nghi ngờ:** arXiv:2606.07316 (BFT certificate) và arXiv:2609.14003 (FLOWSEAL). Nếu có thật → trích dẫn + trả lời bổ đề tiêu cực của 2606.07316.
- [ ] **POS-6 — 🔴 Tìm bản đầy đủ của EvalHack** (*Information* MDPI 17(3):297). **Rủi ro scoop chưa đánh giá được.**
- [ ] **POS-7 — Cập nhật `manuscript/references_verified.bib`** với 6–8 mục mới ở trên (kèm BibTeX đúng).
- [ ] **POS-8 — Cập nhật `manuscript/citation_claim_map.csv`** — mỗi tuyên bố novelty phải trỏ tới câu phân biệt với tiền công trình.
- [ ] **POS-9 — Đổi tên/định vị lại tiêu đề nếu cần.** Tiêu đề hiện tại ("Byzantine-Robust, Injection-Hardened...") vẫn ổn vì nó nêu **cả hai** kênh. **Không** đổi thành gì chỉ nói về robust aggregation.

---

## 4. BASELINE PHẢI THÊM (reviewer sẽ hỏi)

| Baseline | Vì sao bắt buộc | Chi phí |
|---|---|---|
| **RoPoLL-style GM panel** | Đối thủ trực tiếp; `gmed` của bạn chính là nó. Phải cho thấy bạn **ngang bằng** ở kịch bản của họ và **tốt hơn** ở kịch bản injection | ~0 (dùng lại cache verdict) |
| **"Single well-designed prompt"** | Many Minds cho thấy nó **ngang panel 8 vendor**. Nếu bạn không so, reviewer sẽ tự so và kết luận bài vô nghĩa | ~0 (1 judge) |
| **Panel 8 vendor** (nhiều API + open model) | Để tái lập đúng kết quả Many Minds trên pipeline của mình | Trung bình — **không còn là vấn đề vì không giới hạn chi phí** |
| **AutoDefense thật** | Thay cho baseline "cấu trúc" hiện tại — nay chạy được trên EC2 | Trung bình |
| **SecAlign / StruQ thật** | Biến ablation "− hardened judges" từ giả định thành thật | Trung bình |

---

## 5. THAY ĐỔI THÍ NGHIỆM DO ĐỊNH VỊ MỚI

1. **Thêm chỉ số "số phiếu hiệu dụng"** `n_eff = 1 / [1 + (n−f−1)ρ]` (hoặc tương đương) và **so với Many Minds**. Biến Mệnh đề 3 từ "kết quả lý thuyết" thành "giải thích hình thức cho một hiện tượng đã được đo".
2. **Báo cáo điểm vận hành** (>90% chặn tấn công ở <10% benign bị chặn — có đạt không?). Nếu **đạt**, đây là kết quả mạnh nhất của bài (Many Minds nói không cấu hình nào đạt). Nếu **không đạt**, phải nói thẳng — và chỉ ra cơ chế nào thiếu.
3. **Tách rõ hai kênh tấn công trong mọi bảng**: cột "Byzantine only", cột "injection only", cột "both". Đây chính là khoảng trống chưa ai lấp.
4. **So với RoPoLL tại matched compute** — dùng đúng 13 judge nếu kham được, hoặc 5–7 judge và nói rõ.

---

## 6. CẢNH BÁO VỀ TÍNH TRUNG THỰC

- Chỉ trích dẫn những gì **đã fetch và đọc**. Bốn mục dưới đây **chưa tự xác minh**: `arXiv:2606.07316`, `arXiv:2609.14003`, `EvalHack` (MDPI), và bản bài báo đầy đủ của Many Minds. **Không** đưa vào BibTeX trước khi đọc.
- Chính sách AAMAS 2027: **citation bịa ⇒ desk reject**. Đây không phải rủi ro lý thuyết.
- Khi mô tả RoPoLL/Many Minds, **không được** hạ thấp hoặc mô tả sai công trình của họ để làm nổi bật đóng góp của mình — reviewer rất có thể là chính những người đó.
