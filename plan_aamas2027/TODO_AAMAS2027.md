# KẾ HOẠCH HOÀN THÀNH & NỘP BÀI — AAMAS 2027

**Bài báo:** *Aegis-Agency: Byzantine-Robust, Injection-Hardened Multi-Agent Defense Pipelines for Large Language Models*
**Hội nghị mục tiêu:** AAMAS 2027 (26th Int. Conf. on Autonomous Agents and Multiagent Systems), 3–7/05/2027, Hà Nội, Việt Nam
**Ngày lập kế hoạch:** 21/09/2026 (Thứ Hai)
**Trạng thái repo tại thời điểm lập:** `git 490d369`, working tree sạch, 0 kết quả thực nghiệm

---

## 0. KẾT LUẬN NHANH (đọc trước nếu chỉ đọc 1 mục)

| Câu hỏi | Trả lời |
|---|---|
| Bài đã viết được chưa? | **Bản thảo dài 47 trang đã hoàn chỉnh về mặt lập luận** (định vị, mô hình mối đe doạ, kiến trúc, định lý, giao thức đánh giá). |
| Có kết quả thực nghiệm chưa? | **CHƯA CÓ MỘT SỐ NÀO.** Mọi ô trong Bảng 5/6 là `--`; Hình 6 là đường cong "conceptual". Repo tự ghi rõ: *"No real LLM judge, no real benchmark, and no external baseline was executed"* (`audits/result_integrity_audit.md:3-6`). |
| Còn bao nhiêu ngày? | **10 ngày** tới hạn nộp abstract (01/10/2026) và **17 ngày** tới hạn nộp bài (08/10/2026). |
| Việc lớn nhất phải làm? | (1) Sinh **số liệu thực** từ LLM judge thật; (2) **cắt từ 47 trang xuống 8 trang** theo template AAMAS; (3) **định vị lại vì đã bị scoop một phần** (xem dưới); (4) **xin AWS quota + HF access — làm hôm nay**. |
| Rủi ro lớn nhất? | Ba việc hành chính **mặc định là "0/không có"** và đều có thể mất nhiều ngày: **đăng ký OpenReview (hạn 17/09 ĐÃ QUA)**, **AWS GPU quota (mặc định 0 vCPU)**, **HuggingFace gating (duyệt thủ công)**. |
| ⚠️ Tin xấu mới | **Bài đã bị scoop một phần.** RoPoLL (arXiv 2606.30931, ICML 2026) đã làm **tổng hợp geometric-median trên panel LLM judge** — tức là chính aggregator `gmed` của bạn. "Many Minds, One Verdict" (07/2026) đã đo **panel đồng thuận bị chặn bởi tương quan lỗi** — tức là Mệnh đề 3. **Không được nộp với cách định vị hiện tại.** → **`positioning_and_related_work.md`** |
| ✅ Tin tốt mới | **Compute không phải nút thắt.** EC2 `g6e.12xlarge` (4× L40S, ~$10,49/giờ) chạy hết một sweep trong **~20 phút**. Và **dữ liệu đã xong**: 3.054 payload thật đã build và kiểm tra. |
| Có khả thi không? | **Có** — nếu bắt đầu 3 việc hành chính hôm nay và chạy song song dữ liệu – thực nghiệm – viết lại. Xem §8 (rủi ro & dự phòng). |

### 📁 Tài liệu kèm theo (đọc cùng file này)
| File | Nội dung |
|---|---|
| **`positioning_and_related_work.md`** | 🔴 **BẮT BUỘC ĐỌC.** Scoop analysis, 4 đóng góp còn lại, câu viết an toàn, 8 bài phải trích dẫn, baseline phải thêm |
| **`ec2_runbook.md`** | Instance, bố trí vLLM/LiteLLM, revision SHA từng model, checklist provisioning theo ngày, bẫy kỹ thuật, manifest tái lập |
| `code_tasks.md` | Việc code (đã kiểm chứng từ mã nguồn) |
| `task_board.csv` | 69 công việc để tick dần |
| `scripts/data/build_benchmarks.py` | ✅ **Đã chạy xong** — sinh 5 benchmark CSV thật |

---

## 1. HIỆN TRẠNG BÀI NGHIÊN CỨU (đọc trực tiếp từ repo)

### 1.1 Bài này làm gì
Một **uỷ ban gồm `n` LLM agent "thẩm phán" (judge)** kiểm tra một đầu ra ứng viên (payload) và đưa ra phán quyết `v_k = (d_k, s_k, e_k)`; các phán quyết được tổng hợp thành quyết định **allow / block / escalate**. Bài nghiên cứu *chính bản thân pipeline phòng thủ*, chống hai mối đe doạ không có tương đương ở mô hình đơn:

1. **Judge nội bộ bị xâm phạm hoặc thông đồng** (Byzantine minority, `f` judge).
2. **Prompt injection bậc hai (second-order)** — chỉ thị độc hại nằm ngay trong nội dung mà judge phải đọc.

Bốn thành phần của Aegis-Agency:
| # | Thành phần | Cơ chế |
|---|---|---|
| C1 | **Byzantine-robust verdict aggregation** | thay "coordinator tin cậy" bằng `cmed` (trung vị theo toạ độ), `gmed` (trung vị hình học / Weiszfeld), `Krum` |
| C2 | **Payload isolation** | tách kênh dữ liệu khỏi kênh chỉ thị của judge (định nghĩa `ε`-isolation) |
| C3 | **Hardened judges** | SecAlign (preference optimisation) + StruQ (structured input) + temperature calibration |
| C4 | **Tổng quát hoá từ jailbreak → task-integrity** | phán quyết không chỉ "nội dung độc hại" mà cả "vi phạm thứ bậc chỉ thị" |

### 1.2 Kết quả lý thuyết đã có (đây là tài sản thật của bài)

| Ký hiệu | Kết quả | Giả thiết | Trạng thái chứng minh |
|---|---|---|---|
| `lem:gmedrobust` (Bổ đề 1) | `‖û − u*‖ ≤ C_α·r`, `C_α = 2(n−f)/(n−2f) = 2(1−α)/(1−2α)` | A1 (honest concentration), `f < n/2` | **Đầy đủ** (Phụ lục A.1) |
| `thm:integrity` (Định lý 1) | Nếu `γ > C_α·r` thì `D(û) = D(u*) = y*(x)` với **mọi** lựa chọn của `f` judge Byzantine | A1, A2 (margin), A3 (`f<n/2`) | **Đầy đủ** (Phụ lục A.2) |
| `cor:availability` (Hệ quả 1) | Bảo đảm đối xứng: không thể ép block oan (over-refusal) | như Định lý 1 | Đầy đủ |
| `prop:cmed` (Mệnh đề 1) | Trung vị toạ độ nằm trong miền honest ⇒ `C_α = 1`, chỉ cần `γ > r` | A1, `f<n/2` | Sketch ở thân bài + chứng minh đầy đủ ở Phụ lục A.3 |
| `lem:krum` (Bổ đề 2) | Krum với `2f+2 < n` không bị đẩy đi tuỳ ý | A3 (Krum) | ⚠️ **Chỉ trích dẫn lại** Blanchard et al. 2017, **không tự chứng minh** |
| `prop:isolation` (Mệnh đề 2) | Injection đổi quyết định 1 judge với xác suất `≤ ε`; đổi `>t` judge với xác suất `≤ C(n−f, t+1)·ε^(t+1)` | Định nghĩa 1 | Sketch + Phụ lục A.4 |
| `prop:correlated` (Mệnh đề 3) | `Var(Z̄) = μ(1−μ)[(1−ρ)/(n−f) + ρ] → μ(1−μ)ρ`; với `μ<1/2` và `ρ>0`, thêm judge **không** cứu được | mô hình exchangeable Bernoulli | **Đầy đủ** (Phụ lục A.5) |

**Đánh giá:** chuỗi lý thuyết là **mạch lạc, không có mâu thuẫn cứng** (`manuscript/math_audit.md` kết luận "No hard contradictions"), nhưng **hoàn toàn có điều kiện** — và chính các điều kiện (`r`, `γ`, `ρ`, `ε`) **chưa từng được đo**. Đây là điểm yếu chí mạng với reviewer.

### 1.3 Khoảng cách thực tế (gap analysis) — xếp theo mức chặn

| ID | Khoảng cách | Bằng chứng | Mức độ |
|---|---|---|---|
| **B1** | **Không có bất kỳ kết quả thực nghiệm nào.** Bảng 5 (`tab:main`), Bảng 6 (`tab:ablation`) toàn `--`; Hình 6 là `pgfplots_placeholder_results.tex` nhãn *"(conceptual)"* | `manuscript/sections/experiments.tex:4-8, 108-153`; `audits/result_integrity_audit.md:3-6, 21-24` | 🔴 CHẶN CỨNG |
| **B2** | ~~Không có dữ liệu benchmark~~ → ✅ **ĐÃ GIẢI QUYẾT 21/09** | `data/benchmarks/` có 5 bộ CSV thật: harmbench 320, advbench 520, dan 1405, injecagent 510, benign 300 = **3.054 payload**; đã kiểm tra schema, id duy nhất, **rời rạc hoàn toàn** với benign | ✅ XONG |
| **B3** | **Chưa từng chạy pipeline thật.** External baselines (AutoDefense / SecAlign / StruQ / JudgeDeceiver) là **stub `raise NotImplementedError`** | `docs/baseline_adapters.md:17-27`; `audits/implementation_gaps.md:18-30` | 🔴 CHẶN CỨNG |
| **B4** | **Sai định dạng.** Bản thảo là `elsarticle` 1 cột 12pt, **47 trang**; AAMAS yêu cầu `aamas.cls` 2 cột, **tối đa 8 trang** + tài liệu tham khảo, ẩn danh kép | `manuscript/main.tex:7`; `manuscript/reviewer_audit.md:74-83` | 🔴 CHẶN CỨNG |
| **B5** | **Rò rỉ tài liệu nội bộ.** 10 chỗ tham chiếu tới một file riêng `idea.tex` ("the idea flags", "to be proven, not assumed") + "within the verified literature" | `problem_formulation.tex:21,29`; `complexity.tex:4,18,56`; `theory.tex:93,198,204`; `proofs.tex:150`; `introduction.tex:101`; `related_work.tex:90` | 🟠 NGHIÊM TRỌNG |
| **B6** | **Các đại lượng quyết định tính đúng của định lý chưa được đo:** `r`, `γ`, `ρ`, `ε` | `TODO_before_submission.md:12-13`; `audits/preregistration_protocol.md:45` | 🟠 NGHIÊM TRỌNG |
| **B7** | **Định vị lệch cộng đồng.** Bản thảo viết cho CCS/USENIX/TDSC (`elsarticle`, `\journal{Computers & Security / IEEE TDSC}`) — AAMAS là hội nghị **hệ đa tác tử**, cần khung "MAS robustness / judgment aggregation / coordination" | `main.tex:77`; `manuscript/topic_alignment_check.md` | 🟠 NGHIÊM TRỌNG |
| **B8** | **Thiếu ẩn danh & metadata thật.** Tác giả "Author One/Two/Three", email `@example.edu`, affiliation ghi chữ "Placeholder", không ORCID, keyword kiểu Elsevier, có khối "Highlights" | `main.tex:86-94, 98-102`; `sections/abstract.tex:36-48` | 🟡 BẮT BUỘC |
| **B9** | **Chưa khai báo sử dụng AI** theo chính sách AAMAS 2027 (bắt buộc nếu AI tham gia tạo giả thuyết/phương pháp) | chính sách AAMAS 2027 | 🟡 BẮT BUỘC |
| **B10** | **Môi trường chưa sẵn sàng.** Chưa có `.venv`; Python mặc định là **3.14.4** (repo yêu cầu ≥3.11 nhưng `torch`/`vLLM`/`sentence-transformers` gần như chắc chắn **chưa có wheel cho 3.14**) | kiểm tra trực tiếp; `pyproject.toml` | 🟡 KỸ THUẬT |
| **B11** | **Bất nhất nội bộ nhỏ:** tài liệu ghi "47 tests" vs "67 tests"; RQ3/RQ6 khác nhau giữa các file; `thm:integrity` nói "median-type rule" nhưng chứng minh chỉ phủ `gmed` | `TODO_IMPLEMENTATION.md:3` vs `README.md:98`; `math_to_code_audit.md` | 🟢 DỌN DẸP |
| **B12** | 🔴 **ĐÃ BỊ SCOOP MỘT PHẦN.** (a) **RoPoLL** (arXiv 2606.30931, ICML 2026) đã làm robust aggregation bằng **geometric median** trên panel LLM judge → chính là `gmed` của bạn; (b) **"Many Minds, One Verdict"** (Zenodo, 07/2026) đã **đo** panel đồng thuận bị chặn bởi tương quan lỗi (~2 phiếu hiệu dụng) → chính là Mệnh đề 3; (c) **FLOWSEAL** (arXiv 2609.14003) đã phát biểu nguyên lý "cơ chế cưỡng chế trùng bề mặt tấn công" → chính là lập luận payload isolation | `positioning_and_related_work.md` (đã tự fetch và xác minh arXiv abs + Zenodo) | 🔴 **CHẶN VỀ NOVELTY** |
| **B13** | **Ba thủ tục hành chính mặc định "0"** và đều có thể mất nhiều ngày: OpenReview author registration (hạn 17/09 **đã qua**), **AWS GPU quota = 0 vCPU**, **HF gating thủ công** | `ec2_runbook.md` §6 | 🔴 CHẶN CỨNG |

---

## 2. RÀNG BUỘC CỨNG CỦA AAMAS 2027

Nguồn: [Call for Main Track](https://warwick.ac.uk/fac/sci/dcs/aamas2027/calls/call-for-main-track/), [Submission Instructions](https://warwick.ac.uk/fac/sci/dcs/aamas2027/guidelines-and-policies/instructions/), [Q&A](https://warwick.ac.uk/fac/sci/dcs/aamas2027/guidelines-and-policies/qa/).

### 2.1 Mốc thời gian (mọi hạn là cuối ngày, AoE = UTC−12)

| Mốc | Ngày | Thứ | Còn lại từ 21/09 | Trạng thái |
|---|---|---|---|---|
| Đăng ký tác giả trên OpenReview | 17/09/2026 | Th5 | **−4 ngày** | ⚠️ **ĐÃ QUA** |
| **Nộp abstract** (100–300 từ + keywords + area) | **01/10/2026** | Th5 | **10 ngày** | ❗ Chưa làm |
| **Nộp bài (PDF)** | **08/10/2026** | Th5 | **17 ngày** | ❗ Chưa có số liệu |
| Rebuttal | 20–24/11/2026 | | | |
| Thông báo kết quả | 21/12/2026 | Th2 | | |
| Camera-ready | 25/01/2027 | Th2 | | |
| Hội nghị | 03–07/05/2027 | | | Hà Nội |

> **HÀNH ĐỘNG KHẨN (hôm nay):** gửi email tới `aamas2027pcs@gmail.com` xin xác nhận rằng việc đăng ký OpenReview muộn 4 ngày có được chấp nhận không. Nếu không được → kích hoạt **Phương án C** (§8.3).

### 2.2 Ràng buộc định dạng (bắt buộc)

| Ràng buộc | Chi tiết |
|---|---|
| Độ dài | **Tối đa 8 trang** nội dung + **không giới hạn** trang tài liệu tham khảo |
| **Không có appendix** | "An appendix included in the main paper is considered part of the paper and therefore counts towards the 8-page limit" (Q&A #2) |
| Định dạng | **Bắt buộc LaTeX**, class `aamas.cls`, 2 cột (`sigconf`), font Libertine, bib style `ACM-Reference-Format.bst` |
| Ẩn danh | **Double-blind**: dùng `\documentclass[sigconf,anonymous]{aamas}` + `\acmSubmissionID{...}` |
| Cấm | Sửa style file, sửa lề/cỡ chữ/giãn dòng, lạm dụng `\vspace` để nhồi trang |
| Supplementary | **1 file zip ≤ 25 MB**; reviewer **không bắt buộc** xem; **không** được đẩy phần thiết yếu (ví dụ chứng minh chính) vào đây; phải ẩn danh |
| Figure | Mỗi hình cần `\Description{...}` (plain text ≤ 2000 ký tự) |

### 2.3 Chính sách (dễ bị desk-reject nếu vi phạm)

- **AI-assisted technologies:** không được ghi AI là tác giả. Nếu AI tham gia **tạo giả thuyết, phương pháp, hoặc thiết kế thực nghiệm** → phải khai báo công cụ, phiên bản, prompt (trong bài hoặc supplementary). Citation bịa ⇒ desk reject.
- **Dual submission:** không nộp trùng nơi khác trong thời gian review. **arXiv/preprint không tính** là archival ⇒ được phép.
- **Findings:** bài không vào Proceedings sẽ **tự động** được xét vào Findings of AAMAS 2027 (cùng độ dài, cùng format), trừ khi opt-out.
- **Reciprocal reviewer policy:** cần kiểm tra nghĩa vụ phản biện của tác giả.

### 2.4 Chọn Area of Interest — **khuyến nghị: `GAAI` (Generative and Agentic AI)**

Lý do khớp nhất với bài:
- *"Failure handling, recovery, and resilience in agentic AI"* ← **chính xác là bài này**
- *"Assurance, verification, and safety in generative and agentic AI systems"*
- *"Coordination, cooperation, and negotiation in generative AI agents"*
- *"Modeling and analysis of generative AI agents"*
- *"Benchmarks, evaluation, and metrics for generative and agentic AI systems"*

Phương án dự phòng: **EMAS** ("Engineering MAS with LLM methods", "Scalability, fault tolerance"), và **GTEP** (mục *"Judgment aggregation and forecasting"*, *"Voting and preference aggregation"* — về bản chất uỷ ban judge chính là judgment aggregation). **Không** chọn area bảo mật (không có).

---

## 3. CHIẾN LƯỢC: THU HẸP PHẠM VI ĐỂ VỪA 8 TRANG

### 3.1 Ngân sách trang / từ

8 trang 2 cột ≈ 800–900 từ/trang ≈ **~5.500 từ nội dung** (sau khi trừ ~1,5 trang cho hình/bảng).

| Phần | Hiện tại | Mục tiêu 8 trang | Ghi chú |
|---|---|---|---|
| Abstract | 437 từ + Highlights | **180 từ**, bỏ Highlights | |
| Introduction | 901 từ | **900 từ** | giữ, viết lại khung AAMAS |
| Related work | 752 từ | **600 từ** | gộp 5 mục thành 1, giữ bảng so sánh |
| Preliminaries + System model + Problem formulation | 533+689+606 = 1.828 từ | **900 từ** | gộp làm 1 mục; bảng ký hiệu → supplementary |
| Methodology + Algorithm | 721+414 = 1.135 từ | **1.000 từ** | giữ Algorithm 1; Algorithm 2 → 4 dòng |
| Theory | 1.537 từ | **1.100 từ** | xem §3.2 |
| Complexity | 509 từ | **200 từ** | chỉ giữ công thức token/latency + Bảng cost |
| Experiments | 927 từ | **1.100 từ** | giữ + điền số thật |
| Discussion + Limitations | 524+340 = 864 từ | **400 từ** | gộp 1 mục |
| Conclusion | 299 từ | **120 từ** | |
| **Tổng** | **~10.400 từ** | **~5.500 từ** | **cần cắt ~47%** |

### 3.2 Bảng cắt gọt (giữ / hạ cấp / bỏ)

| Hạng mục | Quyết định | Lý do |
|---|---|---|
| `lem:gmedrobust`, `thm:integrity`, `cor:availability` | **GIỮ trong thân bài**, in gọn chứng minh hoặc 3 dòng phác thảo | Đây là đóng góp lý thuyết cốt lõi, đã chứng minh đầy đủ |
| `prop:correlated` (correlated failure) | **GIỮ, ưu tiên cao** | Kết quả "AAMAS nhất": đa dạng backbone là **yêu cầu**, không phải tham số tinh chỉnh |
| `prop:cmed` | **NÉN thành 2 câu** | Hệ quả đơn giản của Định lý 1 với `C_α=1` |
| `prop:isolation` | **NÉN thành 2 câu + số đo `ε`** | Tránh phải trình bày chứng minh union bound |
| `lem:krum` | **HẠ XUỐNG 1 CÂU + trích dẫn** | Tránh phơi ra "kết quả duy nhất không có chứng minh" |
| `fig:architecture` | **GIỮ** | Hình duy nhất mô tả hệ thống |
| `fig:results` (dữ liệu thật) | **GIỮ** (thay placeholder bằng số thật) | Bắt buộc |
| `fig:pipeline` | **BỎ** | Trùng `fig:architecture` |
| `fig:theory` | **BỎ** | Sơ đồ phụ thuộc định lý → 1 câu văn |
| `fig:expsetup` | **BỎ** | Mô tả bằng văn + 1 bảng |
| `tab:notation` | **→ supplementary** | Không đủ chỗ |
| `tab:main` | **GIỮ** (điền số thật) | Bảng chính |
| `tab:ablation` | **GIỮ** (điền số thật) | Chứng minh từng thành phần cần thiết |
| `tab:cost` | **GIỮ** (thay số ước lượng bằng số đo thật) | RQ5 security-vs-cost |
| `tab:guarantees` | **GIỮ dạng rút gọn 4 dòng** hoặc bỏ, thay bằng 1 câu |
| `tab:threatmap` | **BỎ** | Thay bằng 1 đoạn văn |
| `appendix/proofs.tex` | **→ supplementary** | Chỉ giữ 2 chứng minh đầy đủ trong bài nếu còn chỗ |
| `appendix/additional_experiments.tex` | **BỎ HOÀN TOÀN** | Trùng lặp; lại còn tuyên bố có hình/bảng mà file không có |
| 7 câu "disclaimer" lặp lại ("reduced ASR ≠ secure") | **GIỮ ĐÚNG 1** trong bài + 1 dòng ở Limitations | Đang lặp ~7 lần |
| 10 tham chiếu "the idea" | **XOÁ SẠCH** | Rò rỉ tài liệu nội bộ; không thể có trong bài ẩn danh |
| Toàn bộ chữ "placeholder" (8 chỗ) | **XOÁ SẠCH** | Sau khi đã có số thật |

### 3.3 Ba đóng góp chốt lại cho bản 8 trang (viết lại theo khung AAMAS)

1. **Định vị (AAMAS framing):** uỷ ban LLM judge là một **hệ đa tác tử**, và tổng hợp phán quyết là một bài toán **judgment aggregation** dưới tấn công **Byzantine nội bộ**. Bài phân tích **an toàn của chính cơ chế phối hợp**, không chỉ của mô hình.
2. **Cơ chế:** Aegis-Agency = robust aggregation (`cmed`/`gmed`/`Krum`) + payload isolation + hardened judges + task-integrity verdict.
3. **Kết quả:** (a) định lý decision-integrity có điều kiện `γ > C_α·r`; (b) **đo được** `r, γ, ρ, ε` trên LLM thật và chỉ ra khi nào bảo đảm **có hiệu lực / vô hiệu**; (c) **đa dạng backbone là yêu cầu** (Mệnh đề 3); (d) biên security-vs-cost theo `n`.

---

## 4. KẾ HOẠCH HÀNH ĐỘNG THEO GIAI ĐOẠN

> Quy ước: `owner` = người thực hiện (điền tên thật). `⛔` = chặn công việc phía sau.

### GIAI ĐOẠN P0 — Quyết định & hành chính (21–22/09) ⛔ chặn tất cả

- [ ] **T0.1 — Xác nhận với ban PC về hạn OpenReview đã qua.**
  Email `aamas2027pcs@gmail.com`: nêu tên bài, area dự kiến (GAAI), hỏi rõ có được cấp tài khoản/đăng ký muộn không. Ghi lại ngày gửi + trả lời vào `plan_aamas2027/decision_log.md`.
  *Effort: 0,5 h. Owner: ___.*
- [ ] **T0.2 — Tạo tài khoản OpenReview cho TOÀN BỘ tác giả** và điền profile đầy đủ (hạn gốc 17/09).
- [ ] **T0.3 — Chốt danh sách tác giả thật + thứ tự + affiliation + email + ORCID.**
  Thay thế "Author One/Two/Three", `author.one@example.edu`, "University Placeholder" trong `manuscript/main.tex:86-94`.
  *Lưu ý: sau khi accepted **không được** đổi danh sách/thứ tự tác giả.*
- [ ] **T0.4 — Chốt phạm vi bản 8 trang** theo §3.2, và chốt **câu chuyện 3 đóng góp** (§3.3). Ghi vào `plan_aamas2027/scope_locked.md`.
- [ ] **T0.5 — Chốt area of interest: `GAAI`** (dự phòng EMAS/GTEP) — dùng khi nộp abstract 01/10.
- [x] **T0.6 — Quyết định nguồn lực — ✅ ĐÃ CHỐT (sau khi kiểm tra lại):** self-host trên **EC2 `g6e.12xlarge` on-demand** — **4× L40S, 179 GB VRAM, 48 vCPU, 384 GiB RAM, ~$10,49/giờ**. Một vLLM server mỗi GPU, **LiteLLM** làm endpoint duy nhất. Uỷ ban 4 backbone (Llama-3.1-8B + SecAlign LoRA, Qwen2.5-7B, Mistral-7B, Gemma-2-9B); `n = 1..7` chạy được nhờ lặp backbone. **Tôi đã hạ từ 8 GPU xuống 4 GPU** vì 4 GPU đã đủ cho cả 4 đóng góp mà rẻ hơn 3 lần và dễ được duyệt hơn (48 vCPU thay vì 192). Lý do đầy đủ: **`ec2_spec_comparison.md` §5**. Runbook: **`ec2_runbook.md`**.
- [ ] **T0.8 — 🔴 XIN AWS GPU QUOTA — LÀM NGAY HÔM NAY.** Quota "Running On-Demand **G and VT** instances" **mặc định = 0 vCPU**; `g6e.12xlarge` cần **48 vCPU** trong bucket G/VT (xin **≥ 96** để có dự phòng). Hồ sơ có thể mất **vài giờ đến 72 giờ** và thường bị hỏi mô tả mục đích sử dụng. Xin luôn quota **G/VT Spot**. *Đây là rủi ro lịch lớn nhất của toàn dự án.*
- [ ] **T0.9 — 🔴 XIN QUYỀN TRUY CẬP HuggingFace — LÀM NGAY HÔM NAY.** 4 repo gated cho cấu hình 4-GPU: `meta-llama/Llama-3.1-8B-Instruct`, **`meta-llama/Meta-Llama-3.1-8B-Instruct`** (base của SecAlign LoRA — repo **khác** với repo trên), `google/gemma-2-9b-it`, `facebook/Meta-SecAlign-8B` (form có ngày sinh/affiliation/geo-IP). Còn `Qwen/Qwen2.5-7B-Instruct` và `mistralai/Mistral-7B-Instruct-v0.3` **không cần xin**. *(Tuỳ chọn thêm `meta-llama/Llama-3.3-70B-Instruct` nếu sau này có 8 GPU.)* Duyệt của Meta là chậm nhất. Tạo HF token trên **đúng tài khoản** đã được duyệt.
- [ ] **T0.10 — Tải trọng số trên instance CPU rẻ TRONG LÚC CHỜ QUOTA.** Tải không cần GPU. Dùng `HF_HUB_ENABLE_HF_TRANSFER=1` và **`--exclude "original/*"`** (mỗi repo Meta chứa bản trùng `original/*.pth` làm tăng gấp đôi dung lượng). Cấu hình 4-GPU cần **~65 GB** trọng số (không có 70B). Tạo trước **250 GB gp3 @1000 MiB/s, 16.000 IOPS**.
- [ ] **T0.7 — Điền `audits/preregistration_protocol.md` TRƯỚC khi chạy số thật** (đóng băng config, sanity checks, lệnh chạy). Sau khi thấy số, file này bị đóng băng; mọi thay đổi phải ghi vào mục "Protocol amendments".

### GIAI ĐOẠN P1 — Chuẩn bị dữ liệu (21–25/09) ⛔ chặn P2

Repo **không bao giờ tự tải dữ liệu**. Cần tạo CSV theo schema `id, content, label, group` (`docs/data_format.md:13-30`).

> ✅ **TOÀN BỘ P1 ĐÃ HOÀN THÀNH ngày 21/09/2026** bằng `scripts/data/build_benchmarks.py`.
> Kết quả: **3.054 payload thật** trong `data/benchmarks/`, provenance đầy đủ ở `data/benchmarks/PROVENANCE.json`.
>
> | Benchmark | Nguồn | Số dòng | Nhãn |
> |---|---|---|---|
> | `harmbench` | HarmBench (ICML 2024), split test | 320 | 1 |
> | `advbench` | AdvBench/GCG (Zou et al. 2023) | 520 | 1 |
> | `dan` | In-the-wild jailbreak prompts (Shen et al., CCS 2024) | 1.405 | 1 |
> | `injecagent` | InjecAgent (Zhan et al. 2024), direct-harm | 510 | 1 |
> | `benign` | Regular prompts (cùng corpus, `jailbreak=False`), mẫu seed=0 | 299 | 0 |
>
> Đã kiểm tra: header đúng schema, `id` duy nhất, `label ∈ {0,1}`, không ô rỗng, và **benign rời rạc hoàn toàn** với cả 4 tập tấn công (1 dòng trùng đã tự động loại).
>
> ⚠️ **Còn lại:** (a) chạy lại verify bằng `CsvBenchmarkAdapter` thật sau khi có venv (T1.6); (b) ✅ **đã bổ sung** chuẩn over-refusal **XSTest** (T1.9) → `benign_xstest/` 250 safe prompts, dùng cho chỉ số ORR; `benign` Discord giữ làm set phụ; (c) ✅ **đã lấy** dữ liệu JudgeDeceiver/second-order từ repo `ShiJiawenwen/JudgeDeceiver` (chỉ cần file có sẵn trong repo, **không cần FastChat/GPU**) và formal-injection từ `liu00222/Open-Prompt-Injection` (T1.3b/T1.5).

- [x] **T1.1 — HarmBench** → `data/benchmarks/harmbench/test.csv` ✅ 320 dòng.
- [x] **T1.2 — AdvBench + DAN in-the-wild** → `advbench/`, `dan/` ✅ 1.925 dòng. *(PAIR/TAP/GPTFuzzer chưa lấy — không bắt buộc cho bản 8 trang; nếu lấy thêm thì `group` giữ tên tấn công.)*
- [x] **T1.3 — Prompt injection / task-integrity**: InjecAgent → `injecagent/` ✅ 510 dòng (dùng trường `Tool Response` đã chèn chỉ thị của kẻ tấn công). *(Formal injection benchmark đã lấy ở T1.3b.)*
- [x] **T1.3b — Formal injection benchmark** → `formal/test.csv` ✅ 1400 dòng (700 clean / 700 attack, 7 task) dựng bằng `scripts/data/build_formal.py` từ repo `liu00222/Open-Prompt-Injection` (USENIX'24): tái hiện `process_*` + `Task.__split_dataset_and_save` + `CombineAttacker.inject`; record `tasks`/`raw_urls`/`sha256` trong `PROVENANCE.json`.
- [x] **T1.4 — Benign held-out** → `benign/test.csv` ✅ 299 dòng, `label=0`.
- [x] **T1.9 — ORR chuẩn (XSTest)** → `benign_xstest/test.csv` ✅ 250 dòng (10 type × 25), `label=0`, dựng bằng `scripts/data/build_xstest.py` từ `paul-rottger/xstest` (NAACL'24); verify rời rạc 0 trùng với cả 6 tập attack; đã merge `benign_xstest` vào `BENCHMARK_DIRS` (`data/adapters.py`).
- [x] **T1.5 — Second-order injection payloads:** tải repo `ShiJiawenwen/JudgeDeceiver` (⚠️ **repo không có file LICENSE** — cần làm rõ trước khi phát hành bản phái sinh), `dataset/results_suffix/basic/llmbar.json` → `second_order/test.csv` ✅ 2000 dòng (1000 clean / 1000 attack, LLMBar, suffix key `llama-3`) dựng bằng `scripts/data/build_second_order.py` (tái hiện prompt-assembly `AttackPrompt._update_ids`; không cần GPU/FastChat).
- [ ] **T1.5b — Universal injection (liu2024universal) — BẮT BUỘC dùng S gốc:** sinh `S` bằng `scripts/ec2/run_universal_suffix.sh` trên GPU (server EC2 T2.6; local RTX 3050 4GB không đủ) → `scp results/*.json` về → rebuild → `D:\Data\benchmarks\universal_injection\test.csv` + ghi `gradient_optimized_suffix: true` trong provenance. Tạm thời **thư mục chưa có** (7 thư mục hiện tại là chuẩn để chạy). Cần HF access `meta-llama/Llama-2-7b-chat-hf` (gated).
- [x] **T1.6 — Script chuyển đổi**: `scripts/data/build_benchmarks.py` ✅ đã chạy. Còn lại: verify lại qua `CsvBenchmarkAdapter` sau khi dựng venv.
- [x] **T1.7 — Provenance dữ liệu** → `data/benchmarks/PROVENANCE.json` ✅ (URL nguồn, SHA-256 file thô, số dòng, phân bố group, ghi chú giấy phép).
- [x] **T1.8 — Rời rạc (disjointness)** ✅ đã kiểm tra và **cưỡng chế tự động** trong script (`_enforce_disjoint_benign`).

### GIAI ĐOẠN P2 — Môi trường, sửa code, chạy pilot (22–27/09)

- [ ] **T2.1 — Dựng môi trường Python 3.11/3.12** (KHÔNG dùng 3.14 — rủi ro thiếu wheel cho `torch`/`vLLM`/`sentence-transformers`):
  ```powershell
  py -3.11 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -e ".[dev]"
  python -m pytest -q          # kỳ vọng toàn bộ test pass
  python -m ruff check src tests
  ```
- [ ] **T2.2 — Chạy demo tổng hợp để xác nhận cơ chế** (`python scripts/run_synthetic_demo.py --config configs/synthetic_demo.yaml --output outputs/synthetic_demo`). Đây **không phải** kết quả bài báo — chỉ kiểm tra plumbing.
- [ ] **T2.3 — Sửa các điểm cản trở chạy thật** (xem `plan_aamas2027/code_tasks.md` để biết chi tiết sau khi rà soát `experiments/run_real.py`):
  - [ ] Song song hoá lời gọi judge (hiện **tuần tự**, ~2 s/call ⇒ 2.800 call ≈ 1,5–2 h cho 1 bộ; song song hoá cắt xuống 10–20 phút).
  - [ ] Retry/backoff cho rate limit API + ghi nhận lỗi.
  - [ ] Kiểm tra tỷ lệ `refusal/unparseable → BLOCK` (`real_judges.py:95-115`) và **báo cáo tỷ lệ này như một chỉ số** (nếu quá cao, kết quả bị nhiễu).
  - [ ] Xác nhận `embedding_model` (all-MiniLM-L6-v2) chạy được, hoặc tắt (`m=0`) nếu hết thời gian — nhưng phải nói rõ trong bài.
- [ ] **T2.4 — Định nghĩa vận hành cho "hardened judges" (C3) trên đường API.**
  ⚠️ Đây là **câu hỏi khoa học bắt buộc**, vì Bảng ablation có dòng "− hardened judges". SecAlign/StruQ checkpoint chưa có. Chọn một trong hai và ghi rõ:
  (a) *prompt-level hardening*: system prompt có chỉ thị thứ bậc + isolation delimiter; điều kiện "không hardened" = bỏ chỉ thị an toàn/thứ bậc (vẫn giữ isolation để cô lập biến);
  (b) *checkpoint thật*: tải SecAlign/StruQ từ HuggingFace, phục vụ trên GPU thuê.
  Nếu chọn (a) ⇒ **bắt buộc** ghi vào Limitations: "hardening ở đây là prompt-level, không phải fine-tuning".
- [ ] **T2.5 — Chạy smoke 40 payload** (`run_real.sh smoke` **chỉ dùng dummy judge — không bao giờ báo cáo**) rồi **pilot 100–200 payload với judge thật**.
  **Acceptance:** `*provenance.json` có `git_commit` khớp, `backend != dummy`; `real_cost_ledger.csv` có `calls > 0`; không crash; tỷ lệ parse lỗi được ghi lại.

### GIAI ĐOẠN P3 — Thực nghiệm chính thức (26/09 – 04/10) ⛔ chặn P4 (phần Bảng/Hình)

- [ ] **T3.1 — Đóng băng config** (`configs/aamas2027_final.yaml`): `n_judges ∈ {5,7}`, `rules: [cmed,gmed,krum]` + majority vote + single model + AutoDefense-structural; `f` quét `0..⌊(n−1)/2⌋`; `n_seeds ≥ 3`; `threshold=0.5`; `calibrate: true`; ghi commit hash.
- [ ] **T3.2 — Run chính (cold cache)** → `outputs/real/real_evaluation_summary.csv` (mean ± std), `real_evaluation_significance.csv` (paired bootstrap vs baseline). **Đây là Bảng 5.**
- [ ] **T3.3 — Run ablation** (`run_real.sh ablate`) → `real_ablation.csv`. **Đây là Bảng 6.** Bốn dòng: full Aegis / −robust agg / −isolation / −hardened judges.
- [ ] **T3.4 — Đo `r`, `γ`, `μ`, `ρ`** (`measure_theory: true`) → `real_theory_analysis.csv`.
  **Kiểm tra sống còn:** điều kiện `γ > C_α·r` có thoả không? Ghi rõ `C_α` tại `α = f/n` tương ứng.
  - Nếu **thoả** ⇒ định lý có hiệu lực thực tế, đây là kết quả mạnh.
  - Nếu **không thoả** ⇒ **vẫn báo cáo trung thực** (đây chính là "chuyển security thành điều kiện đo được"), và thảo luận khoảng cách còn lại.
- [ ] **T3.5 — Đo `ε`** (`measure_epsilon: true`), isolation ON vs OFF → `real_isolation_epsilon.csv` (RQ2).
- [ ] **T3.6 — RQ4: committee đồng nhất vs đa dạng** → so sánh `ρ` và ASR-UC. Đây là kết quả "AAMAS" nhất; nhớ lưu config mapping `models:`/`endpoints:` theo từng backbone.
- [ ] **T3.7 — RQ5: biên security-vs-cost** theo `n = 1..7`, đo **latency và token thật** trên cold cache → `real_cost_summary.csv`, `real_cost_ledger.csv`.
- [ ] **T3.8 — Reproduction gate:** chạy lại **độc lập ≥ 1 lần** với cùng config đóng băng; so khớp CSV summary. Chỉ khi khớp mới được đặt `is_paper_result: true`.
- [ ] **T3.9 — Cập nhật `audits/result_integrity_audit.md`** với provenance của từng kết quả thật (lệnh, hash, ngày, ý nghĩa từng con số vào bảng nào).
- [ ] **T3.10 — Chạy `make_plots.py --real`** cho các hình kết quả (cờ `--real` chỉ dùng khi số đã được xác minh).

**Acceptance của P3:** mỗi ô trong Bảng 5/6 đều truy được về một CSV có provenance; có mean±std qua seed; có kiểm định ý nghĩa; `r,γ,ρ,ε` là số đo, không phải tham số mô phỏng.

### GIAI ĐOẠN P4 — Viết bản 8 trang (25/09 – 06/10) — chạy song song P2/P3

- [ ] **T4.1 — Chuyển sang template AAMAS** (đã tải sẵn ở `manuscript/aamas2027_template/`):
  `\documentclass[sigconf,anonymous]{aamas}`, `\acmSubmissionID{...}`, `\submissionType{Research Paper Track}`, `\bibliographystyle{ACM-Reference-Format}`.
  Port các macro ký hiệu từ `main.tex:54-75`; chuyển `\newtheorem` sang cấu hình tương thích `aamas.cls/acmart` (dùng `\newtheorem*` cho remark, hoặc bỏ đánh số remark).
  Tạo file mới `manuscript/aamas2027/main.tex` — **không sửa đè** bản elsarticle 47 trang (giữ làm bản mở rộng cho tạp chí).
- [ ] **T4.2 — Cắt theo bảng §3.2** tới mục tiêu §3.1. Kiểm tra độ dài sau mỗi lần cắt.
- [ ] **T4.3 — Điền số liệu thật** vào `tab:main`, `tab:ablation`, `tab:cost`, và `fig:results`; **xoá toàn bộ** chữ "placeholder/conceptual" (8 chỗ: `abstract.tex:30-32`, `introduction.tex:95-96`, `experiments.tex:4-8,108-110,117-118,137-139`, `limitations.tex:45-49`, `conclusion.tex:18-19`, `additional_experiments.tex:4-6`, `tikz_experimental_setup.tex:49-50`).
- [ ] **T4.4 — Xoá 10 tham chiếu "the idea"** và cụm "within the verified literature" (`problem_formulation.tex:21,29`; `complexity.tex:4,18,56`; `theory.tex:93,198,204`; `proofs.tex:150`; `introduction.tex:101`; `related_work.tex:90`).
- [ ] **T4.5 — Thêm `\Description{}` cho mọi hình** và kiểm tra hình đọc được khi in đen trắng.
- [ ] **T4.6 — Viết lại định vị cho AAMAS**: mở bài bằng "uỷ ban LLM agent là một hệ đa tác tử; robustness dưới tác tử Byzantine là bài toán MAS"; liên hệ judgment aggregation / fault tolerance của MAS; giữ phần bảo mật như *ứng dụng*, không phải khung chính.
- [ ] **T4.7 — Chuẩn hoá BibTeX:** mở rộng `and others` (`xi2023agentsurvey`, `bai2022constitutional`, `touvron2023llama2`), thêm DOI/trang cho mục arXiv-only, xác nhận venue chuẩn (HarmBench→ICML 2024, TAP→NeurIPS 2024, MetaGPT→ICLR 2024, PsySafe→ACL 2024, ASB→ICLR 2025).
- [ ] **T4.8 — Ẩn danh hoá:** không tên tác giả/trường/ORCID; không link GitHub của nhóm (thay bằng "code will be released"); kiểm tra metadata PDF (Author field!) và tên file.
- [ ] **T4.9 — Khai báo sử dụng AI** theo chính sách AAMAS 2027 (mục riêng trong bài hoặc supplementary): công cụ, phiên bản, mục đích, phần nào của công trình bị ảnh hưởng.
- [ ] **T4.10 — Sửa lỗi LaTeX:** 85 overfull hbox (Bảng rộng → `\small`/`\resizebox`); font `OT1/cmr/m/scit`; làm rõ `d` (chiều verdict vector) vs `d_k` (quyết định từng judge).

**Acceptance của P4:** biên dịch `pdflatex → bibtex → pdflatex ×2` không lỗi, **≤ 8 trang nội dung**, 0 undefined reference/citation, 0 chữ "placeholder", 0 tham chiếu nội bộ.

### GIAI ĐOẠN P5 — Kiểm tra & nộp (05–08/10)

- [ ] **T5.1 — Red-team số liệu:** một người **không** chạy thực nghiệm đối chiếu từng ô trong bảng/hình với CSV gốc. Ghi biên bản vào `plan_aamas2027/number_audit.md`.
- [ ] **T5.2 — Checklist ẩn danh:** không tên/trường/email/ORCID; không "our previous work"; không link repo cá nhân; kiểm tra cả supplementary.
- [ ] **T5.3 — Checklist format:** không sửa `aamas.cls`; không đổi lề/cỡ chữ; không lạm dụng `\vspace`; PDF đúng ≤ 8 trang.
- [ ] **T5.4 — Đóng gói supplementary** (1 zip ≤ 25 MB): chứng minh đầy đủ, `tab:notation`, mã nguồn, CSV kết quả, manifest payload. **Không** đưa phần thiết yếu vào đây.
- [ ] **T5.5 — Đăng ký abstract trên OpenReview (hạn 01/10):** 100–300 từ plain text + keywords + chọn area `GAAI`. *Bắt buộc, không có abstract thì không nộp được bài.*
- [ ] **T5.6 — Nộp PDF trước 08/10 (AoE).** Nộp bản đầu **05/10**, chỉnh sửa tới 08/10 (OpenReview cho phép nộp lại nhiều lần).
- [ ] **T5.7 — (Tuỳ chọn, sau khi nộp) đăng arXiv** để lấy timestamp — không vi phạm dual submission.
- [ ] **T5.8 — Kiểm tra nghĩa vụ theo Reciprocal Reviewer Policy.**

### GIAI ĐOẠN P6 — Sau khi nộp (11/2026 – 05/2027)

- [ ] **T6.1 — Rebuttal 20–24/11/2026:** chuẩn bị sẵn (a) bảng số bổ sung, (b) câu trả lời cho 3 phản biện dự kiến: *"định lý chỉ có điều kiện"*, *"hardening không phải SecAlign thật"*, *"cỡ uỷ ban nhỏ chỉ chịu được f=1"*. Xem `manuscript/reviewer_audit.md`.
- [ ] **T6.2 — Thông báo 21/12/2026.** Nếu vào Findings: vẫn được công bố CC-BY, cùng format.
- [ ] **T6.3 — Camera-ready 25/01/2027:** bỏ `anonymous`, thêm `\begin{acks}` (funding), mục Artifact Availability (Zenodo/arXiv), cập nhật địa chỉ tác giả, `\balance`.
- [ ] **T6.4 — Nếu bị từ chối:** xem Phương án C (§8.3) — chuyển sang tạp chí/hội nghị bảo mật, giữ nguyên bản dài 47 trang vốn đã viết cho IEEE TDSC / Computers & Security.

---

## 5. LỊCH THEO NGÀY (21/09 → 08/10/2026)

| Ngày | Thứ | Việc chính | Mốc kiểm tra |
|---|---|---|---|
| 21/09 | Th2 | **T0.8 xin AWS quota + T0.9 xin HF access + T0.1 email PC** (cả ba đều chờ được, gửi trong 1 giờ đầu), T0.2–T0.5, T0.6 (đã chốt EC2), ✅ T1.1–T1.4 + T1.6–T1.8 dữ liệu | 3 hồ sơ hành chính đã gửi; dữ liệu xong |
| 22/09 | Th3 | T1.1–T1.3 (HarmBench, AdvBench/DAN, injection), T2.1 venv + pytest | Benchmark đọc được bằng adapter |
| 23/09 | Th4 | T1.4–T1.8 (benign, second-order, verify schema, provenance dữ liệu), T2.3 sửa code | `data_provenance.md` xong |
| 24/09 | Th5 | T2.4 (định nghĩa hardening), T0.7 (pre-registration), T2.5 smoke 40 payload | Pre-registration điền xong **trước** khi chạy |
| 25/09 | Th6 | T2.5 pilot 100–200 payload judge thật; T4.1–T4.2 bắt đầu chuyển template | Pilot chạy hết không crash |
| 26/09 | Th7 | T3.1 đóng băng config; **T3.2 run chính bắt đầu** (cold cache); T4.2 tiếp tục cắt trang | Config commit hash đã ghi |
| 27/09 | CN | T3.2 tiếp; T3.3 ablation; T4.2 | Bảng 5 có số thô |
| 28/09 | Th2 | T3.4 đo r/γ/μ/ρ; **quyết định narrative theo kết quả**; T4.2 | `real_theory_analysis.csv` |
| 29/09 | Th3 | T3.5 đo ε; T3.6 RQ4 đồng nhất vs đa dạng | `real_isolation_epsilon.csv` |
| 30/09 | Th4 | T3.7 RQ5 cost/latency; T3.8 reproduction run; T4.3 điền số vào bảng | Hai CSV khớp nhau |
| 01/10 | Th5 | **HẠN NỘP ABSTRACT (T5.5)**; T3.9 audit; T4.3–T4.5 | ✅ Abstract đã nộp |
| 02/10 | Th6 | T4.4 xoá "the idea"; T4.6 định vị AAMAS; T4.7 bib | 0 tham chiếu nội bộ |
| 03/10 | Th7 | T4.8 ẩn danh; T4.9 AI disclosure; T4.10 lỗi LaTeX | Bản nháp 8 trang lần 1 |
| 04/10 | CN | Đọc lại toàn bài; cắt tiếp cho vừa 8 trang; T5.1 red-team số liệu | ≤ 8 trang, số khớp CSV |
| 05/10 | Th2 | T5.2–T5.4 checklists; **nộp bản đầu (T5.6)**; T5.8 | ✅ Đã nộp bản 1 |
| 06/10 | Th3 | Sửa theo phản hồi nội bộ; hoàn thiện supplementary | |
| 07/10 | Th4 | Buffer — dự phòng chạy bổ sung nếu thiếu số | |
| 08/10 | Th5 | **HẠN NỘP BÀI CUỐI (AoE)** — nộp trước 12:00 để tránh quá tải OpenReview | ✅ HOÀN THÀNH |

> **Quy tắc vàng:** mọi mốc đặt **sớm hơn hạn 24 h**. OpenReview quá tải vào giờ cuối là rủi ro thật.

---

## 6. CHI TIẾT KỸ THUẬT

### 6.1 Dữ liệu → file → nhãn

| Benchmark | Nguồn | Đường dẫn | `label` | `group` | Dùng cho |
|---|---|---|---|---|---|
| HarmBench | `mazeika2024harmbench` (ICML 2024) | `harmbench/test.csv` | 1 | `harmbench` | ASR-UC chính (RQ1/RQ3) |
| AdvBench/GCG | `zou2023universal` | `advbench/test.csv` | 1 | `gcg` | RQ3 adaptivity |
| DAN in-the-wild | `shen2023dan` | `dan/test.csv` | 1 | `dan` | tấn công thực tế |
| PAIR / TAP / GPTFuzzer | `chao2023pair`, `mehrotra2023tap`, `yu2023gptfuzzer` | `pair/`, `tap/`, `gptfuzzer/` | 1 | tên tương ứng | RQ3 |
| InjecAgent | `zhan2024injecagent` | `injecagent/test.csv` | 1 | `injecagent` | task-integrity |
| Formal injection | `liu2024formalizing` | `formal/test.csv` | 1 | `formal` | task-integrity |
| Benign (held-out) | tự thu thập, **rời rạc** | `benign/test.csv` | 0 | `benign` | ORR (Eq. 5), utility (RQ6) |
| Second-order injection | JudgeDeceiver-style | `second_order/test.csv` | 1 | `second_order` | đo `ε` (RQ2) |

### 6.2 Config chạy chính (mẫu — copy từ `configs/ec2_real_evaluation.yaml`)

```yaml
experiment:
  n_judges: 7
  rules: [cmed, gmed, krum]
  threshold: 0.5
  escalate_band: 0.05
  attack: collusion
  f: 2
  calibrate: true
  target_orr: 0.05
  seed: 0
  n_seeds: 3              # BẮT BUỘC >=3 để có mean ± std
  measure_theory: true    # r, gamma, mu, rho  -> thoả mãn yêu cầu "crux"
  measure_epsilon: true   # chỉ bật ở run cuối (tốn gấp đôi)
data:
  root: data/benchmarks
  benchmark: harmbench
  split: test
  limit: 0                # pilot: 100-200
judges:
  backend: openai_compat  # | anthropic | hf
  backbones: [gpt-4o, claude-3.5, llama-3, qwen2.5, mistral]
  model: <model-id>
  endpoint: <base-url>
  api_key_env: OPENAI_API_KEY
  embedding_model: all-MiniLM-L6-v2
  isolation: true
  cache_dir: outputs/real_verdict_cache
```

### 6.3 Lệnh chạy

```bash
# kiểm tra plumbing (DỮ LIỆU DUMMY — KHÔNG BAO GIỜ BÁO CÁO)
bash scripts/ec2/run_real.sh smoke

# run chính (Bảng 5 + đo r,γ,ρ,ε)
bash scripts/ec2/run_real.sh full

# ablation (Bảng 6, tái dùng cache)
bash scripts/ec2/run_real.sh ablate

# vẽ hình (chỉ sau khi số đã xác minh)
python scripts/make_plots.py --input outputs/real/real_evaluation_summary.csv \
       --output outputs/real/asr_vs_f.png --real
```

### 6.4 Artefact đầu ra & ánh xạ vào bài báo

| File | Nội dung | Vào đâu |
|---|---|---|
| `real_evaluation_summary.csv` | mean ± std theo seed | **Bảng 5** |
| `real_evaluation_significance.csv` | paired bootstrap vs baseline | chú thích Bảng 5 |
| `real_ablation.csv` | 4 cấu hình thành phần | **Bảng 6** |
| `real_theory_analysis.csv` | `r, γ, μ, ρ` + kiểm tra Định lý 1 | mục Theory / Experiments |
| `real_isolation_epsilon.csv` | `ε` isolation ON/OFF | RQ2 |
| `real_cost_summary.csv`, `real_cost_ledger.csv` | calls/tokens/wall-time | **Bảng cost**, RQ5 |
| `asr_vs_f.png` | ASR-UC vs `f` | **Hình kết quả** |
| `*provenance.json` | git commit, lệnh, deps, GPU, `is_paper_result` | supplementary + audit |

### 6.5 Ước lượng thời gian & chi phí trên EC2 (đã cập nhật)

Instance: **`g6e.12xlarge`** on-demand — 4× L40S (179 GB VRAM, 48 vCPU, 384 GiB RAM), **~$10,49/giờ**. 4 backbone / 5 slot song song, `--max-num-seqs 16`, `--max-model-len 2048`.

| Pha | Số call mới | Thời gian (6 judge song song) | Chi phí |
|---|---|---|---|
| Uỷ ban honest + calibration (400 payload) | ~2.800 | ~3–5 phút | ~$2 |
| Đo `ε` (isolation on/off) | ~5.600 | ~7–10 phút | ~$5 |
| **Một sweep đầy đủ** | **~8.400** | **~10–15 phút** (+5–20 phút load model) | **~$10–20** |
| Toàn bộ dự án (ước tính 40–80 giờ máy, gồm cả thời gian nhàn rỗi có load model) | — | — | **~$1.200–2.500** |

⇒ **Compute hoàn toàn không phải nút thắt.** Kể cả trường hợp xấu nhất (mọi call dồn vào 1 GPU) cũng chỉ ~33 phút.
⇒ **Hệ quả quan trọng cho code (C-1):** giá trị của việc song song hoá client **giảm mạnh** — vLLM đã tự continuous-batching. Thay vào đó, **giữ client tuần tự** để `--max-num-seqs` ổn định và kết quả **tái lập tốt hơn** (vLLM greedy **không** bất biến theo batch). Xem `ec2_runbook.md` §7.4.
⇒ **Dừng instance giữa các sweep** — để chạy không tải tốn ~$725/ngày.

> **Nút thắt thật sự của 17 ngày còn lại là hành chính:** AWS quota (mặc định 0) + HF gating (duyệt thủ công) + OpenReview registration (đã quá hạn). Cả ba phải bắt đầu hôm nay.

---

## 7. ĐỊNH NGHĨA "HOÀN THÀNH" & KỶ LUẬT KHOA HỌC

Bài chỉ được coi là **sẵn sàng nộp** khi **tất cả** điều sau đúng:

- [ ] Mọi ô số trong bài truy được về một file CSV có `*provenance.json` tương ứng.
- [ ] `backend != dummy` trong mọi provenance được trích dẫn.
- [ ] Có **mean ± std qua ≥ 3 seed** cho mọi so sánh.
- [ ] Có **kiểm định ý nghĩa** cho mọi tuyên bố so sánh.
- [ ] `r, γ, ρ, ε` là **số đo**, không phải tham số mô phỏng; và bài **nói rõ** điều kiện `γ > C_α·r` có thoả hay không.
- [ ] Run chính đã **tái lập độc lập ≥ 1 lần**, CSV khớp.
- [ ] `audits/preregistration_protocol.md` đã điền **trước** run đầu và đóng băng.
- [ ] `audits/result_integrity_audit.md` đã cập nhật provenance thật.
- [ ] **0 chữ "placeholder"**, **0 tham chiếu "the idea"**, **0 kết quả dummy/synthetic** được trình bày như kết quả thật.
- [ ] Bản PDF **≤ 8 trang** nội dung, biên dịch sạch, ẩn danh hoàn toàn.
- [ ] Đã khai báo AI theo chính sách AAMAS 2027.
- [ ] Abstract đã đăng ký trên OpenReview trước 01/10.

### Những việc **KHÔNG** được làm
- ❌ Trình bày đầu ra `outputs/synthetic_demo/` hoặc `run_real.sh smoke` (dummy judge) như kết quả thật.
- ❌ Đặt `is_paper_result: true` khi chưa tái lập.
- ❌ Nới lỏng/giữ nguyên tuyên bố khi `γ > C_α·r` không thoả — phải báo cáo trung thực.
- ❌ Giữ bất kỳ câu nào ngụ ý "ASR giảm ⇒ hệ thống an toàn" (`claim_strength_audit.md` đã cấm từ "secure" cho khẳng định).
- ❌ Trích dẫn bài chưa kiểm chứng (chính sách AAMAS: citation bịa ⇒ desk reject).
- ❌ Nộp trùng một bài khác đang review (dual submission).
- ❌ Đưa tên/trường/link repo cá nhân vào bản ẩn danh.

---

## 8. RỦI RO & PHƯƠNG ÁN DỰ PHÒNG

| ID | Rủi ro | Xác suất | Tác động | Giảm thiểu / Phương án |
|---|---|---|---|---|
| R1 | **Không kịp sinh số liệu thực trước 08/10** | Cao | Chí mạng | **Plan B:** thu hẹp còn 1 benchmark (HarmBench 150–200 payload) × 5 judge × API; chỉ cần **1 Bảng chính + 1 ablation + `r,γ,ρ,ε`**. Cắt RQ3/RQ5. |
| R2 | Đăng ký OpenReview muộn bị từ chối | Trung bình | Chí mạng | Liên hệ PC **hôm nay**; nếu bị từ chối ⇒ **Plan C** |
| R3 | `γ > C_α·r` không thoả trên LLM thật | Trung bình–Cao | Trung bình | Đây là **kết quả hợp lệ và đáng báo cáo** (bài vốn định vị "biến security thành điều kiện đo được"). Chuẩn bị sẵn narrative + thảo luận khoảng cách |
| R4 | Đa dạng backbone **không** giảm `ρ` | Trung bình | Trung bình | Vẫn báo cáo; hạ tuyên bố từ "diversity is a requirement" xuống "diversity là điều kiện cần được kiểm chứng" |
| R5 | Tỷ lệ `refusal → BLOCK` cao làm nhiễu kết quả | Trung bình | Trung bình | Đo và báo cáo tỷ lệ; thử 2–3 mẫu prompt judge khác nhau; ghi vào Limitations |
| R6 | Rate limit / chi phí API vượt dự kiến | Trung bình | Thấp–TB | Pilot bằng model "mini"; cache verdict theo `(payload_id, judge_id, backbone)`; song song hoá có giới hạn |
| R7 | Hardening không phải SecAlign/StruQ thật | **Cao** | Trung bình | Khai báo rõ trong Limitations như "prompt-level hardening"; hoặc tải checkpoint SecAlign/StruQ lên GPU thuê nếu còn thời gian |
| R8 | Baseline ngoài (AutoDefense/SecAlign/StruQ thật) vẫn là stub | **Cao** | Trung bình | **Hạ tuyên bố**: chỉ so với AutoDefense **cấu trúc** (mean coordinator) và ghi rõ "head-to-head với hệ thống AutoDefense thật để lại cho bản mở rộng". Reviewer AAMAS thường chấp nhận nếu minh bạch |
| R9 | Rò rỉ danh tính (repo nhóm, "Placeholder" còn sót) | Trung bình | Chí mạng | Checklist T5.2 + kiểm tra metadata PDF + grep "Placeholder\|example.edu" trên bản nộp |
| R10 | Vi phạm chính sách AI (không khai báo) | Trung bình | Chí mạng | T4.9 — khai báo tường minh |
| R11 | Python 3.14 thiếu wheel cho `torch`/`vLLM` | Cao | TB | T2.1 — dùng Python 3.11/3.12 |
| R12 | Hết thời gian viết lại → nộp bản chưa cắt hết 8 trang | Trung bình | Chí mạng | Bắt đầu T4.1 **ngay 25/09**, song song với chạy thực nghiệm; **không** để việc viết lại vào tuần cuối |
| **R13** | 🔴 **AWS GPU quota không về kịp** (mặc định **0 vCPU**, hồ sơ có thể mất **72 h** và bị hỏi mục đích) | Trung bình–Cao | **Chí mạng** | Xin **hôm nay**, xin ≥256 vCPU. Phương án chèn: thuê L40S ở Lambda/RunPod (~$0.85–1.90/giờ, không cần quota) — chấp nhận được vì **không giới hạn chi phí** |
| **R14** | 🔴 **HF gating (Meta) không duyệt kịp** cho Llama-3.1/3.3/SecAlign | Trung bình | Cao | Xin **hôm nay**, cả 5 repo + repo `Meta-Llama-3.1-8B-Instruct`. Dự phòng: thay Llama bằng model mở khác; bỏ slot 70B; dùng SecAlign bản CDN thay vì LoRA HF |
| **R15** | 🔴 **Bị đánh giá là trùng lặp với RoPoLL / Many Minds** | **Cao nếu không sửa định vị** | **Chí mạng** | Làm **P1.1–P1.9** trong `positioning_and_related_work.md`; dẫn đầu bằng **mô hình mối đe doạ kết hợp** (judge Byzantine **+** injection trong payload), không dẫn bằng robust aggregation |
| **R16** | Gemma-2 ném HTTP 400 với system message; Mistral-v0.3 yêu cầu luân phiên nghiêm ngặt → **âm thầm mất 1 judge khỏi uỷ ban** | **Cao** | Cao | Gộp system prompt vào lượt user đầu, hoặc `--chat-template` riêng; **log số token thật mỗi call** (`ec2_runbook.md` §7.1–7.3) |
| **R17** | vLLM greedy **không bất biến theo batch** → kết quả không tái lập được | Trung bình | Cao | Cố định + báo cáo `--max-num-seqs`; chạy **determinism probe** 100 call × 2 và **công bố tỷ lệ khớp** (`ec2_runbook.md` §6.4) |

### 8.3 Phương án dự phòng

- **Plan A (mục tiêu):** nộp AAMAS 2027 main track đúng 08/10 với số liệu thực. Findings là lưới an toàn (tự động xét).
- **Plan B (thu hẹp):** 1 benchmark, 5 judge, 3 seed, 1 bảng chính + 1 ablation + đo `r,γ,ρ,ε`. Vẫn là bài 8 trang trung thực.
- **Plan C (lùi hạn):** nếu OpenReview không cho đăng ký muộn **hoặc** dữ liệu không kịp → giữ bản dài 47 trang vốn đã được viết cho **IEEE TDSC / Computers & Security** (`main.tex:77`), bổ sung số liệu thật rồi nộp tạp chí; AAMAS 2028 là mục tiêu kế tiếp. **Kiểm tra hạn thực tế của các venue dự phòng trước khi cam kết.**

---

## 9. PHỤ LỤC A — Checklist chuyển LaTeX sang AAMAS

- [ ] Dùng `manuscript/aamas2027_template/` (đã tải sẵn: `aamas.cls`, `AAMAS_2027_sample.tex`, `ACM-Reference-Format.bst`, `by.pdf`, `aamas27-logo-small.jpg`).
- [ ] `\documentclass[sigconf,anonymous]{aamas}` cho bản nộp.
- [ ] Giữ nguyên khối copyright, `\setcopyright{ifaamas}`, `\acmConference[...]`, `\copyrightyear`.
- [ ] Điền `\acmSubmissionID{<số OpenReview>}` sau khi đăng ký abstract.
- [ ] `\submissionType{Research Paper Track}`.
- [ ] `\usepackage{balance}` + `\balance` ở cột đầu trang cuối; `\usepackage{booktabs}` (đã có sẵn trong class).
- [ ] **Không** dùng `\journal{}`, khối `Highlights`, `orcidlink`, `elsarticle-num`.
- [ ] `\bibliographystyle{ACM-Reference-Format}` — **không** dùng `elsarticle-num`.
- [ ] Mọi hình có `\Description{}`; caption bảng đặt **trên**, caption hình đặt **dưới**.
- [ ] `\begin{acks}` cho funding (tự động bị bỏ ở chế độ anonymous).
- [ ] **Không** appendix trong thân bài (đếm vào 8 trang) → chuyển sang supplementary zip.
- [ ] Kiểm tra nhãn font: `aamas.cls` cần **Libertine**. Trên máy này MiKTeX báo thiếu metric `txsyc.tfm` và bị chặn ghi log khi biên dịch trong môi trường sandbox — **hãy chạy `latexmk -pdf` một lần trên máy bạn (ngoài sandbox) và để MiKTeX tự cài package `libertine` trước khi bắt đầu viết**, đừng để phát hiện vào ngày nộp.

## 10. PHỤ LỤC B — Checklist 15 phút trước khi bấm nộp

- [ ] PDF ≤ 8 trang nội dung (tài liệu tham khảo không tính).
- [ ] Chế độ `anonymous`; không còn tên/trường/email/ORCID/link repo.
- [ ] Metadata PDF: Author/Ttile/Subject không lộ danh tính.
- [ ] 0 undefined reference, 0 undefined citation, 0 `[?]`.
- [ ] 0 chữ "placeholder", "conceptual", "TODO", "the idea".
- [ ] Mọi số trong bảng khớp CSV gốc.
- [ ] Hình đọc được khi in đen trắng; mọi hình có `\Description`.
- [ ] Abstract đã đăng ký trên OpenReview (nếu chưa ⇒ **không nộp được**).
- [ ] Đã chọn area = `GAAI` (hoặc EMAS/GTEP).
- [ ] Đã khai báo AI theo chính sách.
- [ ] Supplementary zip ≤ 25 MB, ẩn danh, không chứa phần thiết yếu.
- [ ] Đã kiểm tra nghĩa vụ Reciprocal Reviewer.
- [ ] Nộp xong lưu lại: PDF, source zip, số submission, ảnh chụp xác nhận → `plan_aamas2027/submission_record.md`.

---

## 11. TÓM TẮT MỘT TRANG — 10 VIỆC QUAN TRỌNG NHẤT

> **Ba việc đầu tiên chỉ mất ~1 giờ và đều là hồ sơ "chờ được". Hãy gửi chúng TRƯỚC KHI làm bất cứ việc gì khác.**

| # | Việc | Hạn | Vì sao |
|---|---|---|---|
| 1 | 🔴 **Xin AWS GPU quota ≥256 vCPU (bucket G/VT) ở us-east-1** | **21/09, ngay** | Mặc định là **0**; hồ sơ có thể mất **tới 72 h**. Chặn toàn bộ thực nghiệm nếu chậm |
| 2 | 🔴 **Xin HF access 5 repo gated** (Llama-3.1-8B, Llama-3.3-70B, Meta-Llama-3.1-8B, Gemma-2-9B, Meta-SecAlign-8B) | **21/09, ngay** | Duyệt **thủ công**; Meta là chậm nhất |
| 3 | 🔴 **Email ban PC** về hạn OpenReview đã qua (17/09) + tạo tài khoản cho mọi tác giả | **21/09, ngay** | Không có tư cách ⇒ mất luôn cơ hội |
| 4 | 🔴 **Đọc `positioning_and_related_work.md` và chốt định vị mới** | **22/09** | Đã bị scoop một phần (RoPoLL / Many Minds); nộp với định vị cũ ⇒ **chắc chắn bị từ chối vì novelty** |
| 5 | ✅ ~~Tải & chuẩn hoá dữ liệu~~ — **đã xong 21/09**, 3.054 payload | — | Xong |
| 6 | Tải trọng số trên instance CPU rẻ trong lúc chờ quota (`--exclude "original/*"`) | **21–22/09** | Biến thời gian chờ thành việc đã xong |
| 7 | Dựng EC2 + vLLM + LiteLLM theo `ec2_runbook.md`; chạy **determinism probe** | **25/09** | Nền tảng của mọi số liệu |
| 8 | **Chuyển template AAMAS + bắt đầu cắt 47→8 trang** | **25/09** | Việc tốn thời gian nhất, dễ bị dồn vào tuần cuối |
| 9 | Chạy run chính + ablation + đo `r,γ,ρ,ε` + tái lập | **30/09** | Trái tim của bài |
| 10 | **Nộp abstract (01/10)** rồi **nộp PDF (08/10)** | **01/10 · 08/10** | Cứng |

**Nguồn tham chiếu:** [AAMAS 2027 Call for Main Track](https://warwick.ac.uk/fac/sci/dcs/aamas2027/calls/call-for-main-track/) · [Submission Instructions](https://warwick.ac.uk/fac/sci/dcs/aamas2027/guidelines-and-policies/instructions/) · [Q&A](https://warwick.ac.uk/fac/sci/dcs/aamas2027/guidelines-and-policies/qa/) · [OpenReview group](https://ifaamas.org/AAMAS/2027/Conference)
