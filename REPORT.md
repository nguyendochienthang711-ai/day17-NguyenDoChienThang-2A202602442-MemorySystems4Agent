# BÁO CÁO THỰC HÀNH DAY 17: MEMORY SYSTEMS FOR AI AGENT

**Học viên:** Nguyễn Đỗ Chiến Thắng  
**Mã học viên:** 2A202602442  
**Môn học / Track:** Phase 2, Track 3 - Day 17: Memory Systems for AI Agent  

---

## 1. Tổng quan triển khai hệ thống

Bài lab xây dựng và so sánh hai kiến trúc agent:
1. **Baseline Agent (Agent A):**
   - Chỉ lưu trữ ngữ cảnh trong bộ nhớ ngắn hạn của cùng một phiên/thread (`SessionState`).
   - Không có bộ nhớ bền vững (`User.md`).
   - Không có cơ chế nén lịch sử (`Compact Memory`).
   - Quên toàn bộ thông tin khi truy vấn ở thread mới (Cross-session recall = 0%).
   - Kéo theo toàn bộ lịch sử thô qua từng lượt dẫn đến chi phí ngữ cảnh ($O(N^2)$) tăng vọt ở các hội thoại dài.

2. **Advanced Agent (Agent B):**
   - **Lớp 1: Short-term Memory:** Quản lý hội thoại gần nhất trong thread.
   - **Lớp 2: Persistent Memory (`User.md`):** Lưu trữ hồ sơ người dùng ổn định qua các phiên làm việc, tự động cập nhật đính chính (conflict resolution) và lọc nhiễu.
   - **Lớp 3: Compact Memory (`CompactMemoryManager`):** Tự động kích hoạt khi tổng token vượt ngưỡng (`threshold_tokens = 700`), tóm tắt các lượt cũ và chỉ giữ lại $K$ tin nhắn gần nhất (`keep_messages = 4`).

---

## 2. Kết quả Benchmark thực nghiệm

Thực hiện benchmark trên 2 bộ dữ liệu chuẩn tiếng Việt bằng lệnh `python src/benchmark.py`:

### 2.1. Standard Benchmark (`data/conversations.json` - 10 hội thoại, user `dungct`)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent** | 2,701 | 18,250 | **0.0%** | **0.0%** | 0 | 0 |
| **Advanced Agent** | 3,523 | 30,281 | **100.0%** | **100.0%** | 345 | 0 |

### 2.2. Long-Context Stress Benchmark (`data/advanced_long_context.json` - 16 lượt dài, user `dungct_stress`)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent** | 474 | 23,472 | **0.0%** | **0.0%** | 0 | 0 |
| **Advanced Agent** | 943 | **12,791** *(Giảm 45.5%)* | **100.0%** | **100.0%** | 276 | **16** |

---

## 3. Kết quả Kiểm thử Tự động (Pytest)

Chạy kiểm thử với lệnh `pytest src/test_agents.py -v`:

```text
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\Public\Documents\AI_BAITAP\day17-NguyenDoChienThang-2A202602442-MemorySystems4Agent
collected 4 items

src/test_agents.py::test_user_markdown_read_write_edit PASSED            [ 25%]
src/test_agents.py::test_compact_trigger PASSED                          [ 50%]
src/test_agents.py::test_cross_session_recall PASSED                     [ 75%]
src/test_agents.py::test_compact_reduces_prompt_load_on_long_thread PASSED [100%]

============================== 4 passed in 0.11s ==============================
```

---

## 4. Phân tích Chuyên sâu & Trả lời Câu hỏi Cốt lõi (Bước 8)

### 4.1. Vì sao Advanced Agent có khả năng Recall vượt trội so với Baseline?
- **Baseline Agent** lưu trữ ngữ cảnh cục bộ theo `thread_id`. Khi một câu hỏi recall được gửi ở một `thread_id` hoàn toàn mới, `session.messages` khởi tạo rỗng. Do không có kênh truy xuất dài hạn nào, Baseline bắt buộc phải thừa nhận không có thông tin cá nhân của người dùng, dẫn đến Recall đạt 0.0%.
- **Advanced Agent** sử dụng mô hình lưu trữ tệp Markdown bền vững (`state/profiles/<user_id>/User.md`). Bất kể truy vấn đến từ thread nào, agent đều tải và phân tích hồ sơ người dùng để trả lời chính xác tên, nơi ở, nghề nghiệp, đồ uống/món ăn yêu thích và sở thích trả lời, đạt **Recall tuyệt đối 100.0%**.

### 4.2. Vì sao Advanced Agent có chi phí token cao hơn ở hội thoại ngắn?
- Trong các hội thoại ngắn dưới ngưỡng compact ($< 700$ tokens), cơ chế compact chưa được kích hoạt (`Compactions = 0`).
- Ở mỗi lượt, prompt của Advanced Agent phải mang theo: `Hồ sơ User.md` + `Lịch sử thread` + `Tin nhắn mới`, trong khi Baseline chỉ mang theo `Lịch sử thread` + `Tin nhắn mới`.
- Do đó, ở hội thoại ngắn, Advanced Agent chịu một khoản "thuế bộ nhớ" (overhead) cố định để duy trì tính bền vững, làm số `Prompt tokens processed` cao hơn (30,281 so với 18,250).

### 4.3. Vì sao Compact Memory giúp Advanced Agent có lợi thế áp đảo ở hội thoại dài?
- Ở hội thoại siêu dài (Stress Benchmark), mỗi lượt người dùng gửi các đoạn văn rất dài chứa tin tức và phân tích hệ thống.
- **Baseline Agent** không có cơ chế nén, buộc phải cộng dồn toàn bộ các tin nhắn trước vào prompt của lượt kế tiếp, khiến chi phí ngữ cảnh tăng theo hàm bậc hai $O(N^2)$, chạm mức **23,472 prompt tokens**.
- **Advanced Agent** khi nhận thấy tổng tokens vượt ngưỡng 700 tokens, tự động kích hoạt quá trình rút gọn (đã kích hoạt 16 lần). Các tin nhắn cũ được tóm tắt thành summary súc tích, chỉ giữ lại 4 tin nhắn gần nhất nguyên vẹn. Nhờ đó, lượng prompt tokens chỉ dừng lại ở **12,791 tokens (tiết kiệm 45.5% chi phí ngữ cảnh)** mà không làm mất thông tin hồ sơ dài hạn.

### 4.4. Tốc độ tăng trưởng file memory (`User.md`) và các rủi ro đi kèm
- Trong cả 2 bài test, kích thước `User.md` chỉ tăng ở mức vừa phải: **345 bytes** cho `dungct` (sau 10 hội thoại) và **276 bytes** cho `dungct_stress`.
- **Rủi ro thực tế khi hệ thống vận hành lâu dài:**
  1. **Memory Bloat:** Nếu người dùng trò chuyện qua hàng tháng, nếu không có cơ chế chắt lọc hoặc nén định kỳ, file `User.md` sẽ phình to, biến chính nó thành gánh nặng token cho mỗi prompt.
  2. **Hallucination / False Fact Pollution:** Người dùng có thể đùa, đặt câu hỏi giả định hoặc thay đổi ý định liên tục, khiến agent ghi nhận những fact sai lệch vào bộ nhớ vĩnh viễn.

---

## 5. Các tính năng Mở rộng / Bonus đạt mức 90-100 điểm (Rubric)

1. **Confidence Threshold & Guardrail chống ghi nhận từ câu hỏi:**
   - Hệ thống được trang bị bộ lọc nhận diện câu nghi vấn/truy vấn (`is_inquiry_or_question`).
   - Khi người dùng hỏi: *"Tên mình là gì?", "Bạn có biết DũngCT không?", "Nếu ai đó nhắc Huế, Hà Nội..."*, hệ thống nhận biết đây là câu hỏi tìm kiếm, tuyệt đối **không** nhầm lẫn từ "gì", "ai" hoặc các địa danh nhắc tới trong câu hỏi thành thông tin cập nhật mới.

2. **Conflict Handling & Dynamic Fact Overwrite:**
   - Khi người dùng đính chính: nơi ở chuyển từ *Đà Nẵng* sang *Huế*, rồi trong stress test chuyển từ *Huế* sang *Đà Nẵng*, hệ thống ghi đè giá trị mới nhất vào đúng key trong `User.md`, loại bỏ hoàn toàn fact cũ mâu thuẫn.
   - Khi nghề nghiệp đổi từ *backend engineer* sang *MLOps engineer*, hệ thống cập nhật đúng nghề mới và loại bỏ nghề cũ.

3. **Distractor & Noise Filtering (Lọc nhiễu thông tin):**
   - Lọc bỏ địa danh tạm thời: Nhận biết *"Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày"* và không lưu Hà Nội làm nơi ở.
   - Lọc bỏ câu đùa nghề nghiệp: Nhận biết *"chuyển sang product manager... nhưng đó chỉ là câu đùa"* và giữ nguyên nghề MLOps engineer.
   - Lọc bỏ ví dụ cũ: Nhận biết *"Đà Nẵng như ví dụ cũ thì đừng lấy nó"* trong conv-10.

4. **Multi-Provider Support:**
   - Module `model_provider.py` hỗ trợ đầy đủ 6 provider theo đúng yêu cầu: `openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`, đảm bảo tính độc lập và khả chuyển của memory system.
