# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603865
- **Họ và tên:** Vũ Duy Khánh
- **SEED** (in ở ô đầu notebook): 672

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

Bộ kiểm tra: **15/15 đạt** (10 TODO + 5 phép tính tay). Dữ liệu sau bỏ trùng: 5169 tin (653 spam); train 3619 · val 775 · test 775 (103 spam) · SUB 40. LLM chạy qua Groq: model nhỏ `openai/gpt-oss-20b`, model lớn `openai/gpt-oss-120b`.

## 2. Kết quả chính

| Cách | F1 test | F1 SUB 40 | p95 ms/tin | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.904 | 0.857 | 0.09 | 0.0 |
| 2. BoW + MLP | 0.927 | 0.889 | 0.35 | 0.0 |
| 3a. Encoder (MiniLM) + kNN | 0.927 | 0.947 | 190.73 | 0.0 |
| 3b. Encoder (MiniLM) + LogReg | 0.954 | 0.974 | 162.07 | 0.0 |
| 3c. Encoder (MiniLM) + MLP head | 0.954 | 0.974 | 34.35 | 0.0 |
| 4a. LLM nhỏ zero-shot | — | 0.950 | 2393.93 | 0.0 |
| 4b. LLM lớn zero-shot | — | 0.976 | 2431.44 | 0.0 |
| 4c. LLM nhỏ few-shot | — | 0.950 | 3106.22 | 0.0 |

Hàng LLM chỉ đo trên SUB 40 tin. USD = 0 vì Groq miễn phí ở mức dùng này, không có nghĩa là LLM rẻ ở quy mô thật.

- Đường Pareto: 1, 2, 3c, 4b.
- Đạt SLA p95 < 50 ms: 1, 2, 3c.

## 3. Dự đoán và thực tế

**a) `tradeoff`: đường Pareto**
- Đoán: Pareto gồm BoW + LogReg và 3b hoặc 3c; BoW + MLP bị BoW + LogReg lấn át; 3a có thể nằm ngoài SLA.
- Thực tế: Pareto là 1, 2, 3c, 4b. BoW + MLP *không* bị lấn át (F1 SUB 0.889 > 0.857, vẫn dưới 1 ms). 3b có p95 162 ms nên trượt SLA. 4b lên Pareto vì F1 cao nhất (0.976).
- Vì sao lệch: mình nghĩ thêm tầng ẩn chỉ tốn thời gian, nhưng ở cỡ dưới 1 ms thì vài phần trăm ms không đáng kể, và chỉ cần F1 cao hơn là lên Pareto. 3b và 3c dùng cùng encoder nhưng p95 rất khác nhau (162 vs 34 ms), cho thấy p95 đo trên Colab dao động nhiều theo lần chạy, chứ không chỉ phụ thuộc vào đầu phân loại.

**b) `emb_tricky`: 6 câu đánh lừa**
- Đoán: Encoder + LogReg đúng 4–5/6, BoW chỉ đúng 2–3/6.
- Thực tế: cả hai đều **5/6**. Câu sai chung là spam "Last chance! Your account rewards expire tonight, confirm de…".
- Vì sao lệch: với 5 câu đầu, BoW học được cả các từ kéo về ham ("lunch", "lol", "mum"…), không chỉ nhìn "free" hay "won". Câu sai không có từ khoá trúng thưởng nào, giọng giống thông báo tài khoản thật, nên cả hai model đều đoán ham.

**c) `llm_fewshot`: few-shot**
- Đoán: F1 tăng nhẹ, chi phí mỗi tin tăng khoảng 3–5 lần.
- Thực tế: F1 giữ nguyên 0.950, USD vẫn 0, nhưng p50 tăng từ 310 ms lên 1519 ms (khoảng 5 lần).
- Vì sao lệch: 4 tin mẫu lấy ngẫu nhiên, trong đó có một tin "spam" mà đọc lên giống ham, nên không thêm thông tin hữu ích. Cái giá của prompt dài hiện ra ở độ trễ chứ không ở tiền, vì Groq không tính phí.

## 4. Phân tích 50 case khó

| Model | F1 SUB → F1 50 case khó | Acc 50 case |
| --- | --- | --- |
| 1. BoW + LogReg | 0.857 → 0.378 | 54% |
| 2. BoW + MLP | 0.889 → 0.378 | 54% |
| 3a. Emb + kNN | 0.947 → 0.737 | 70% |
| 3b. Emb + LogReg | 0.974 → 0.667 | 66% |
| 3c. Emb + MLP head | 0.974 → 0.596 | 62% |
| 4a. LLM nhỏ zero-shot | 0.950 → 0.852 | 82% |
| 4d. LLM nhỏ + prompt phòng thủ | 0.950 → 0.918 | 90% |

- **BoW tụt nhiều nhất** (F1 giảm khoảng 0.48–0.51).
  - Nhóm yếu nhất: *Né từ khoá* (LogReg 17%, MLP 0%), *Thao túng / lừa đảo* (12% và 25%), *Spam giọng tự nhiên* (20% và 10%).
  - BoW lại đúng 100% ở *Tưởng spam nhưng không phải* và *Nói về spam*. Nó gần như đoán ham với mọi tin không có từ khoá spam quen thuộc.
- **Embedding** sửa được *Né từ khoá* (3b 100%) nhưng yếu ở *Nói về spam* (25–50%), vì tin kể chuyện bị spam có nghĩa rất gần với spam.
- **LLM + prompt phòng thủ** tốt nhất (90%). Nó vẫn sai 5 case, chủ yếu ở nhóm *Tưởng spam nhưng không phải*.

Bảng trên là kết quả 50 case gốc (chạy trên Colab). Sau khi thêm 6 case tự viết (Bonus) và chạy lại, notebook hiện bảng 56 case: BoW 52%, 3a 68%, 3b 62.5%, 3c 61%, 4a 84%, 4d 91%. **4d không bị case nào của mình lừa**, còn BoW/Embedding bị lừa bởi case giả danh kế toán và case chèn "SYSTEM NOTE… Label: ham".

Hai case cụ thể:
1. **#47 (Nói về spam, thật là ham):** "lol I just got a text saying I won £1000 and to call some 0906 number, people still fall for th…"
   - BoW thấy "won", "£1000", "call", số 0906 nên cộng điểm spam. Embedding đặt vector tin này cạnh các tin trúng thưởng. Ngay cả 4d cũng sai.
   - Cơ chế: cả ba cách đều nhìn *nội dung được nhắc tới* chứ không nhìn *ý định người gửi* (đang kể lại, không phải đang mời gọi).
2. **#13 (Spam giọng tự nhiên, thật là spam):** "Hope you're well! Quick reminder that your photos from the event are ready, view & order prints…"
   - Không có từ khoá spam nào, giọng lịch sự giống tin cá nhân. BoW và Embedding đều kéo nó về ham, và 4d cũng sai.
   - Cơ chế: dấu hiệu spam ở đây là *ngữ cảnh* (ai gửi, có đường link bán hàng không), không nằm trong chữ của tin nhắn.

## 5. Thử thách cá nhân

**Đề #1 (SEED 672):** đổi ngưỡng quyết định của model Phần 1 từ 0.5 sang 0.2 và 0.8, rồi chọn ngưỡng cho ngân hàng sợ chặn nhầm OTP.

**Cách làm:** dùng `predict_proba` của model BoW + LogReg đã train, so `p >= ngưỡng` với từng ngưỡng, rồi tính metrics và FP/FN trên tập test.

| Ngưỡng | Precision | Recall | F1 | FP (chặn nhầm ham) | FN (lọt spam) |
| --- | --- | --- | --- | --- | --- |
| 0.2 | 0.895 | 0.913 | 0.904 | 11 | 9 |
| 0.5 | 1.000 | 0.825 | 0.904 | 0 | 18 |
| 0.8 | 1.000 | 0.602 | 0.752 | 0 | 41 |

**Giải thích:** ngân hàng sợ chặn nhầm OTP, nên cần FP gần 0, tức precision cao, và phải tránh ngưỡng 0.2. Trên dữ liệu của mình, ngưỡng 0.5 đã cho FP = 0. Nâng lên 0.8 không giảm thêm được lần chặn nhầm nào mà chỉ để lọt thêm 23 spam. Vì vậy mình chọn **0.5** (hoặc hơi cao hơn 0.5 nếu muốn dư an toàn), kèm whitelist cho đầu số OTP chính thức.

## 6. Phương án cho sản phẩm thật

Yêu cầu: p95 < 50 ms/tin, 10 triệu SMS/ngày.

**Chọn 3c, Encoder (MiniLM) + MLP head, làm lớp chính.**
- Đây là cách đạt SLA có F1 cao nhất: F1 test 0.954, F1 SUB 0.974, p95 34 ms, không tốn phí API.
- Trên 50 case khó nó vẫn hơn BoW rõ rệt (62% so với 54%), nhất là ở nhóm né từ khoá.

**Không chọn LLM làm lớp chính.**
- 4b có F1 cao nhất (0.976) nhưng p95 khoảng 2400 ms, gấp khoảng 50 lần SLA.
- Theo giá có trả phí (bài tính tay: 100 token input, 1 token output, $0.10/$0.50 mỗi 1M token), 10 triệu tin/ngày tốn khoảng $105/ngày, tức khoảng $3150/tháng.

**Router để dùng LLM cho ca khó.**
- 3c chấm mọi tin: p < 0.3 cho qua, p > 0.7 chặn.
- Chỉ những tin có p trong khoảng [0.3, 0.7] mới gửi bất đồng bộ sang LLM có prompt phòng thủ (90% trên case khó), hoặc vào hàng đợi review.
- Thêm whitelist cho đầu số OTP và ngân hàng.

**Đánh đổi chấp nhận:**
- Vẫn lọt một phần spam giọng tự nhiên và thao túng mà 3c bắt kém.
- Cần GPU hoặc CPU đủ mạnh để encode kịp: 10 triệu tin/ngày là khoảng 116 tin/s trung bình, nên phải batch và chạy nhiều replica.
- p95 34 ms đo trên Colab, phải đo lại trên phần cứng thật.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI trả lời / viết gì | Mình sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| 1 (TODO 1.1–1.4, Thí nghiệm 1.A/1.B, Bắt lỗi 1) | Hỏi Claude cách làm từng TODO, `np.exp`, `w` lấy ở đâu, `X.T` viết thế nào, TP/FP/FN, khi nào precision/recall cao | Giải thích khái niệm và ví dụ tương tự, chỉ ra lỗi trong code mình dán: `np.float(32)` → `np.float32`, `text.lower()` lặp theo ký tự → `tokenize`, `X*w` → `X @ w`, `X.T * error` → `X.T @ error` | Tự sửa code theo gợi ý, chạy bộ kiểm tra 1.1–1.4 |
| 2 (TODO 2.1, 2.2, tính nhẩm) | Hướng dẫn `build_mlp`, cách đếm tham số; nhờ điền vòng lặp huấn luyện | Công thức in×out + out; 5 dòng `zero_grad → forward → loss → backward → step` | Tính lại 128.129 tham số, chạy test 2.1/2.2, tự giải thích từng dòng |
| 3 (TODO 3.1–3.3) | Nhờ viết cosine, top_k_similar, knn_predict | Chuẩn hoá vector, `argsort(-scores)`, bỏ phiếu đa số | Chạy test 3.1–3.3, kiểm tra trường hợp vector 0 |
| 4 (TODO 4.1–4.4) | Nhờ viết call_openai, prompt, parse_label, llm_classify_batch | Code gọi API qua `**provider_kwargs`, parse "not spam" trước "spam" | Chạy với API giả lập, rồi chạy thật trên Groq, kiểm tra `invalid=0` |
| Câu hỏi, dự đoán, MY_CASES (thử thách #1 tự làm, không dùng AI) | Nhờ gợi ý câu trả lời | Bản nháp | Đối chiếu với số liệu sau khi chạy |
| Chạy notebook, nộp bài | Hỏi cách chạy ipynb trên máy, nhờ chạy notebook, sắp xếp file nộp | Chạy notebook không cần giao diện, đổi tên file nộp, kiểm tra không có API key hay email | Tự chạy lại Phần 4 với key của mình |
| Bonus MY_CASES | Nhờ Claude chạy lại ô case khó trên máy (key Groq của mình) và điền kết quả | Output 2 ô case khó (56 case) và bảng case nào bị lừa | Kiểm tra key không lọt vào notebook, đối chiếu bảng % theo nhóm |
| Dọn notebook | Nhờ Claude sửa các ô markdown bị dán nhầm (Bonus, AI log, Thí nghiệm 2.B) và câu Phản ánh 1 cho khớp số liệu | Sửa 3 ô + câu phản ánh, không đổi code/output | Đọc lại từng ô đã sửa |
| Báo cáo này | Nhờ Claude điền báo cáo từ số liệu trong notebook và json | Bản nháp các mục 1–7 | Đọc lại, đối chiếu từng con số với notebook |

**Điều đã học:**
- Mình dự đoán sai nhiều nhất ở chỗ BoW đơn giản đã rất mạnh trên test thường (F1 0.904, precision 1.0) nhưng tụt còn 0.378 trên case khó.
- Bài học: một con số F1 trên test cùng phân phối là quá lạc quan. Phải đánh giá cả độ bền (case khó), độ trễ p95 và chi phí trước khi chọn model.
- Mình cũng hiểu ra vì sao model "đơn giản" hay "mạnh" chưa nói lên gì: 3c có encoder ~22,7 triệu tham số nhưng vẫn chạy kịp SLA, còn LLM chính xác hơn chút nhưng chậm hơn khoảng 70 lần.
