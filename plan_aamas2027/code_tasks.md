# VIỆC CẦN LÀM VỀ CODE — AAMAS 2027

Bổ trợ cho `TODO_AAMAS2027.md` (mục **T2.3**).
Mọi kết luận dưới đây được **kiểm chứng trực tiếp từ mã nguồn** ngày 21/09/2026 (commit `490d369`), không suy đoán.

---

## 1. Trạng thái đường chạy thật (đã kiểm chứng)

### ✅ Đã có sẵn — KHÔNG cần viết lại

| Hạng mục | Bằng chứng |
|---|---|
| Adapter judge thật, 3 backend | `src/aegis_agency/data/real_judges.py`: `OpenAICompatJudge` (API OpenAI **hoặc** vLLM local), `AnthropicJudge`, `HuggingFaceJudge` |
| Prompt cô lập dữ liệu (StruQ-style) + bật/tắt | `JudgePromptBuilder` với `isolation: bool` (`real_judges.py:145-178`) |
| Cache phán quyết theo `(payload_id, judge_id, backbone)` | `VerdictCache` (`real_judges.py:285-338`) — **đây là thứ làm cho run lặp lại rẻ** |
| Sổ chi phí token/thời gian từng lời gọi | `UsageLedger` (`real_judges.py:214-282`) |
| Fallback `refusal/không parse được → BLOCK` | `_judge_verdict_or_block` (`real_judges.py:95-115`) |
| Chạy thật đầu-cuối: calibrate → evaluate → ablate | `experiments/run_real.py`; CLI `python -m aegis_agency.cli {calibrate,evaluate,ablate}` (`cli.py:128-134`); **không có `NotImplementedError`** trong `run_real.py` |
| Ghi đủ artefact CSV/JSON | `run_real.py:735-737, 876-882, 916` |
| **Đo `r, γ, μ, ρ`** | `metrics/estimators.py`: `honest_radius`, `decision_margin`, `honest_correlation` (trả `mu`, `rho`, `score_rho`) |
| **Đo `ε` bằng cách GỌI LẠI LLM thật** (không mô phỏng) | `run_real.py:427-480` — `_epsilon_rows()` chèn `cfg.injected_suffix` vào payload rồi hỏi lại judge, isolation ON/OFF; tính tỷ lệ flip bằng `epsilon_estimates` |
| **Kiểm định ý nghĩa thống kê** | `metrics/estimators.py::paired_bootstrap_mean_diff` → `real_evaluation_significance.csv` |
| Tỷ lệ flip từng judge | `metrics/estimators.py::per_judge_flip_rates` |
| Provenance (git commit, lệnh, deps, GPU) | `utils/provenance.py` |

> **Kết luận quan trọng:** đường chạy thật **đã hoàn chỉnh về mặt chức năng**. Nút thắt là **dữ liệu + thời gian chạy + viết lại bài**, không phải thiếu code lõi.

### 🔴 Stub chưa có thật (phải quyết định: làm hoặc hạ tuyên bố)

| Hệ thống | Vị trí | Trạng thái |
|---|---|---|
| AutoDefense thật | `baselines/external_wrappers.py:55-63` `AutoDefenseAdapter.predict()` | `raise NotImplementedError` |
| SecAlign (checkpoint) | `external_wrappers.py:66-73` `SecAlignAdapter.predict()` | `raise NotImplementedError` |
| StruQ (checkpoint) | `external_wrappers.py:76-83` `StruQAdapter.predict()` | `raise NotImplementedError` |
| JudgeDeceiver (optimiser thật) | `external_wrappers.py:86-98` `JudgeDeceiverAdapter.inject()` | `raise NotImplementedError` |
| Baseline AutoDefense **cấu trúc** (mean coordinator) | `baselines/autodefense.py` | ✅ **Đã có** — dùng được ngay cho so sánh |
| No-defense / single model / majority vote | `baselines/{no_defense,single_model,majority_vote}.py` | ✅ Đã có |
| Dummy judge (chỉ để test) | `external_wrappers.py:101-110` | ✅ Có — **tuyệt đối không báo cáo** |

---

## 2. VIỆC CODE BẮT BUỘC (theo thứ tự ưu tiên)

### C-1 🔴 Song song hoá lời gọi judge — **ưu tiên cao nhất**
**Hiện trạng:** toàn repo **không có** `ThreadPoolExecutor`, `concurrent.futures`, `asyncio`, hay `multiprocessing` (đã grep toàn `src/`). Mọi lời gọi đi tuần tự; `docs/ec2_experiment_guide.md:114` xác nhận "Judge calls are **serial**".
**Hệ quả:** ~8.400 call/bộ 400 payload ⇒ **~4,5 giờ** cho 1 bộ. Reproduction gate cần ≥ 2 lần chạy ⇒ ~13 h chỉ riêng thời gian chờ.
**Việc:** thêm gọi song song có giới hạn (`max_workers` cấu hình được, mặc định 8 cho API / 4 cho vLLM local) trong đường build committee — `run_real.py:206-245` (`_build_...`, nơi tạo `ledgers` và gọi `judge()` cho từng judge/payload).
**Ràng buộc:** giữ nguyên ngữ nghĩa cache và ledger; `UsageLedger` phải thread-safe (thêm `threading.Lock` vào `record()` — `real_judges.py:225-244`); `VerdictCache.store()` cũng cần lock khi ghi file (`real_judges.py:321-335`).
**Acceptance:** một bộ 400 payload × 7 judge hoàn tất **< 1 giờ** với API; CSV kết quả **giống hệt** bản tuần tự với cùng seed (kiểm tra bằng test đối chiếu).
*Effort: 6–8 h.*

### C-2 🔴 Retry / backoff / xử lý lỗi mạng
**Hiện trạng:** `OpenAICompatJudge.judge()` (`real_judges.py:408-409`) và `AnthropicJudge.judge()` (`:478-479`) ném `RuntimeError` ngay khi lời gọi lỗi. Có `max_retries=3` ở tầng client SDK, nhưng **không** có backoff riêng cho rate limit (HTTP 429) và **không** có ghi nhận lỗi ra file.
**Việc:** bọc vòng retry có exponential backoff + jitter; đếm và ghi số lỗi/retry vào `real_cost_summary.csv`; nếu một judge lỗi quá `k` lần thì **dừng run** thay vì âm thầm bỏ judge (bỏ judge sẽ phá tính toàn vẹn của uỷ ban).
**Acceptance:** chạy pilot qua đêm không chết giữa đường; số lỗi được báo cáo.
*Effort: 3–4 h.*

### C-3 🟠 Đo và báo cáo tỷ lệ `refusal / unparseable → BLOCK`
**Hiện trạng:** fallback tồn tại (`real_judges.py:95-115`) và ghi `logger.warning`, nhưng **không** có bộ đếm tổng hợp, không vào CSV.
**Vì sao quan trọng:** nếu 20% phán quyết là "không parse được ⇒ BLOCK", thì ASR-UC thấp có thể chỉ là hệ quả của việc uỷ ban luôn BLOCK — một **artifact**, không phải robustness. Reviewer sẽ hỏi đúng câu này.
**Việc:** thêm bộ đếm `parse_failures` / `refusals` vào `UsageLedger` (hoặc ledger riêng), xuất cột vào `real_cost_summary.csv`, và **báo cáo trong bài** như một chỉ số chất lượng.
**Acceptance:** có con số tỷ lệ parse lỗi theo từng backbone, xuất hiện trong artefact và trong bài.
*Effort: 2–3 h.*

### C-4 🟠 Làm rõ "second-order injection" là một **tấn công cố định**, không phải JudgeDeceiver tối ưu
**Hiện trạng:** `run_real.py:85` định nghĩa một `DEFAULT_INJECTED_SUFFIX` duy nhất, chèn vào payload (`:457`) rồi hỏi lại judge. Đây là **đo `ε` thật** (tốt!), nhưng **không** phải JudgeDeceiver (không có tối ưu hoá lặp để đánh lừa judge).
**Việc (chọn 1 trong 2, và ghi rõ trong bài):**
- (a) *Nhanh, đủ cho deadline:* giữ suffix cố định, **đổi cách gọi trong bài**: "a fixed second-order instruction-suffix injection" thay vì "JudgeDeceiver-style"; thêm 2–3 biến thể suffix để cho thấy kết quả không phụ thuộc một chuỗi duy nhất (tăng độ tin cậy, chi phí thấp).
- (b) *Đầy đủ:* implement `JudgeDeceiverAdapter.inject()` (tối ưu hoá lặp) — tốn thời gian, **không khuyến nghị** trong 17 ngày.
**Acceptance:** bài không còn ngụ ý đã dùng JudgeDeceiver thật; có ≥ 2 biến thể injection với kết quả nhất quán.
*Effort: 2 h (a) / 16 h+ (b).*

### C-5 🟠 Định nghĩa vận hành "hardened judges" để ablation Bảng 6 có nghĩa
**Hiện trạng:** Bảng 6 (`manuscript/sections/experiments.tex:135-153`) có dòng "− hardened judges", nhưng trong đường chạy thật **không tồn tại công tắc "hardening"** — chỉ có công tắc `isolation` (`real_judges.py:171-178`, `VerdictCache` tag `iso/noiso`). SecAlign/StruQ là stub ⇒ **không thể** chạy ablation này như bài mô tả.
**Việc:** thêm một trục cấu hình độc lập, ví dụ `judges.hardening: off | prompt | checkpoint`:
- `prompt`: system prompt có chỉ thị thứ bậc + cảnh báo nội dung độc hại (mức "hardened" dùng được ngay);
- `off`: bỏ các chỉ thị đó, **giữ nguyên isolation** (để cô lập đúng một biến);
- `checkpoint`: dành cho SecAlign/StruQ thật (chưa có).
Truyền vào `JudgePromptBuilder` một tham số thứ hai (hiện chỉ có `isolation`), và lưu vào **cache key** — nếu không đưa vào key, verdict cũ sẽ bị tái sử dụng sai.
**Acceptance:** chạy được cả 3 cấu hình ablation của Bảng 6 trên judge thật; cache không lẫn giữa các mức hardening.
*Effort: 4–6 h.*

### C-6 🟡 Ghi `is_paper_result` và điều kiện môi trường vào provenance (đã có, cần kiểm tra)
**Việc:** xác nhận `utils/provenance.py` ghi đủ `git_commit`, `command`, `deps`, `gpu` **trong môi trường chạy thật**, và rằng `tests/test_reproducibility.py::test_no_committed_result_files_claim_paper_results` vẫn xanh sau khi sinh artefact thật (test này cấm file provenance nào khai `is_paper_result: true` được commit).
**Acceptance:** `pytest -q` xanh sau khi có `outputs/real/`.
*Effort: 1 h.*

### C-7 🟡 Cảnh báo "cache nóng" khi báo cáo chi phí (RQ5)
**Hiện trạng:** `real_cost_summary.csv` phản ánh **số lời gọi thực tế của lần chạy đó**; chạy lại trên cache nóng ⇒ báo ~0 chi phí (`docs/ec2_experiment_guide.md:154-157`).
**Việc:** thêm cờ/ghi chú `cache_state: cold|warm` vào provenance và vào đầu file CSV cost; fail-fast (hoặc cảnh báo to) nếu người dùng định báo cáo số cost từ một run cache nóng.
**Acceptance:** không thể vô tình công bố "0 token" như số đo thật.
*Effort: 1–2 h.*

### C-8 🟡 Song song hoá/đo latency đúng cách cho RQ5
**Hiện trạng:** `UsageLedger` cộng dồn `elapsed_s` **tuần tự** (`totals()` trả tổng thời gian). Sau khi song song hoá (C-1), **tổng thời gian không còn là latency của pipeline**.
**Việc:** ghi riêng **wall-clock của cả run** và **latency song song = max(latency judge) + latency agg + latency analyze** (đúng công thức `Lat_∥(n)` trong bài, `docs/../paper_implementation_spec.md`) — nếu không, Bảng cost sẽ mâu thuẫn với công thức lý thuyết trong bài.
**Acceptance:** số đo latency khớp định nghĩa `Lat_∥(n)` dùng trong Complexity.
*Effort: 3 h.*

---

## 3. VIỆC CODE NÊN LÀM NẾU CÒN THỜI GIAN

| ID | Việc | Giá trị | Effort |
|---|---|---|---|
| C-9 | Cờ `--limit`/`--benchmark` để chạy nhanh nhiều tổ hợp mà không sửa YAML | giảm sai sót thao tác | 1 h |
| C-10 | Sinh `tab:main`/`tab:ablation` ra LaTeX **trực tiếp từ CSV** | loại bỏ lỗi copy tay — đây là nguồn lỗi số liệu phổ biến nhất | 3 h |
| C-11 | Kiểm tra tự động: mọi ô số trong `.tex` phải khớp CSV (script `check_numbers.py`) | phục vụ T5.1 red-team | 3 h |
| C-12 | Nén/khử trùng lặp cache verdict, giới hạn dung lượng | cache có thể phình vài trăm MB | 1 h |
| C-13 | Chạy `ruff` + `mypy` sạch trước khi nộp (artifact quality) | reviewer AAMAS đánh giá reproducibility | 2 h |
| C-14 | Script sinh supplementary zip ≤ 25 MB, tự kiểm tra ẩn danh | phục vụ T5.4 | 2 h |

---

## 4. THỨ TỰ THỰC HIỆN ĐỀ XUẤT (2 ngày: 22–24/09)

| Thứ tự | Việc | Lý do |
|---|---|---|
| 1 | **C-1** song song hoá + lock ledger/cache | Mọi thứ phía sau phụ thuộc tốc độ này |
| 2 | **C-2** retry/backoff | Chạy dài không được chết |
| 3 | **C-3** đếm parse-failure | Cần trước khi tin bất kỳ con số nào |
| 4 | **C-5** trục hardening | Bảng 6 không chạy được nếu thiếu |
| 5 | **C-4** làm rõ/đa dạng hoá injection | RQ2 cần ít nhất 2 biến thể |
| 6 | **C-7** + **C-8** | Để Bảng cost trung thực và khớp lý thuyết |
| 7 | Chạy pilot (T2.5) rồi mới sang P3 | |

## 5. NGUYÊN TẮC KHÔNG ĐƯỢC VI PHẠM KHI SỬA CODE

- ❌ Không sửa bất kỳ thứ gì làm thay đổi **ngữ nghĩa cơ chế** (aggregator, metric, attack) — code hiện tại đã khớp bài và đã có test; thay đổi sẽ làm số liệu không còn tương ứng với định lý.
- ❌ Không đưa `dummy` judge vào bất kỳ đường báo cáo nào.
- ❌ Không hạ thấp `n_seeds` xuống dưới 3.
- ❌ Không xoá `VerdictCache` giữa các lần chạy **trừ khi** đang đo chi phí (RQ5) — cache là tài sản.
- ✅ Mọi thay đổi code sau khi điền `audits/preregistration_protocol.md` phải ghi vào mục **Protocol amendments** kèm commit hash.
