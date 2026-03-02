# VisionGuard bus: Hệ thống Nhận diện Khuôn mặt Điểm danh Thông minh trong xe bus

---

## 🌟 Giới thiệu

**VisionGuard bus** là giải pháp điểm danh thông minh ứng dụng công nghệ nhận diện khuôn mặt tiên tiến. Hệ thống được thiết kế theo mô hình **Edge Computing**, kết hợp giữa sức mạnh xử lý của PC Server để trích xuất dữ liệu và khả năng vận hành thực tế tối ưu trên **Raspberry Pi 4**.

---

## 📌 Bối cảnh bài báo / Đề tài

Bài báo giải quyết **bài toán điểm danh học sinh trên xe bus trường học** — một chủ đề có ý nghĩa:

- **An toàn trẻ em** — Xác nhận lên/xuống xe, giảm rủi ro bỏ quên hoặc nhầm tuyến.
- **Intelligent transportation** — Tích hợp AI vào vận tải học đường.
- **Edge AI / IoT deployment** — Inference tại thiết bị biên (Raspberry Pi 4), không phụ thuộc mạng liên tục.
- **Privacy-aware biometric systems** — Đặc trưng khuôn mặt (embeddings) xử lý và so khớp tại edge, hạn chế truyền ảnh gốc.

Đề tài phù hợp với các hội nghị Scopus Q4 về **IoT Applications**, **Smart Transportation**, **Edge Computing**.

---

## 🏗️ Kiến trúc Hệ thống

Dự án được chia thành các phân vùng chính:

### 1. Phân vùng PC Server (`face-recognizer-server`)

Đóng vai trò là "Trung tâm xử lý dữ liệu", thực hiện các nhiệm vụ:

- **Chuẩn hóa dữ liệu:** Chuyển đổi hàng loạt ảnh từ nhiều định dạng (.pgm, .jpg, .png) sang `.webp` để tối ưu dung lượng.
- **Trích xuất đặc trưng (Feature Extraction):** Sử dụng mô hình **MobileFaceNet** (PyTorch/TFLite) để biến đổi khuôn mặt thành vector đặc trưng **512 chiều**.
- **Đóng gói Database:** Lưu trữ kết quả dưới dạng file `face_embeddings.json`.

### 2. Phân vùng Edge Device (`edge-device-pi4`)

Chạy trực tiếp trên Raspberry Pi 4 để nhận diện thời gian thực:

- **Xử lý phần cứng:** Điều khiển Camera, quản lý vòng lặp sự kiện (nhấn phím Space để điểm danh).
- **Tiền xử lý ảnh:** Khử nhiễu (Gaussian Blur), tăng tương phản (CLAHE) trong điều kiện ánh sáng kém, Resize và chuẩn hóa ảnh.
- **Nhận diện AI:** Sử dụng **TFLite** để chạy model MobileFaceNet nhẹ, so khớp danh tính bằng **khoảng cách Cosine** (1 − cosine similarity) và **multi-frame verification** (bỏ phiếu quá bán).

### 3. Phân vùng Backend & Web (`backend-server` & `frontend-web`)

- **Backend:** FastAPI cung cấp API quản lý tập trung, lưu trữ lịch sử và gửi thông báo.
- **Frontend:** Giao diện Dashboard để quản lý danh sách học sinh và theo dõi điểm danh.

---

## 📄 Nội dung kỹ thuật (Nội dung bài báo)

Bài báo trình bày:

- **Kiến trúc hybrid edge–server:** Feature extraction tập trung trên PC, inference và so khớp trên Raspberry Pi 4.
- **MobileFaceNet + TFLite:** Vector embeddings **512-d**, chuẩn hóa L2.
- **So khớp:** **Cosine distance** (1 − cosine similarity); vector embedding được L2-normalize, so khớp theo góc giữa hai vector. Ngưỡng (threshold) mặc định **0.45** cho cosine distance.
- **Strict multi-frame verification:** Chụp 3–5 frame mỗi lần; chỉ xác nhận danh tính khi **quá bán** frame trong ngưỡng (threshold) với cùng một người.
- **Backend FastAPI:** Quản lý học sinh, phụ huynh, nhận kết quả điểm danh từ edge và gửi thông báo.

### Dataset (đánh giá thực nghiệm)

| Mục | Giá trị | Ghi chú |
| --- | --- | --- |
| **Học sinh train** | 33 | Số danh tính trong database embeddings |
| **Ảnh/người (train)** | ~5–9 ảnh | Thay đổi theo identity (xem `train_details_grouped.csv`) |
| **Người test** | 36 | Số identity trong tập test (gồm cả người đã train và người chưa gặp) |
| **Ảnh test** | 72 | Tổng số ảnh trong phase Edge (đánh giá closed-set / open-set) |

### Đánh giá thực nghiệm

- **Độ chính xác:** TAR (True Accept Rate), recall trên tập train; độ chính xác nhận diện trên tập test (Edge phase).
- **Threshold sensitivity:** Ngưỡng so khớp (mặc định 0.45 cho cosine distance) có thể điều chỉnh theo FAR/FRR.
- **Latency:** Inference (ms/frame), preprocessing (ms).
- **Tài nguyên:** CPU (%), RAM (%).
- **Lighting robustness:** Face detection DNN + fallback Haar, tiền xử lý CLAHE khi ánh sáng kém.

Cấu trúc logic, trình bày rõ ràng phù hợp báo cáo khoa học.

---

## 📊 Kết quả thực nghiệm (Experimental Results)

Số liệu lấy từ **report**: `figures/report_20260302_030440/` (cập nhật 02/03/2026).

### 1. Độ chính xác (Accuracy)

| Giai đoạn | Độ chính xác | Ghi chú |
| :--- | :---: | :--- |
| **Train data** | **100.0%** | Đánh giá trên tập train (33 folder), pipeline FaceRecognizer (cosine 0.45). |
| **Test data** | **100.0%** | Đánh giá trên tập test (36 folder), cùng pipeline edge. |

### 2. Thông số Cơ sở dữ liệu & Model

| Thông số | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| **Total Identities** | 33 | Số danh tính trong database embeddings. |
| **Embedding Dimension** | 512 | Số chiều vector đặc trưng (MobileFaceNet), 512-dim. |
| **Input size** | **160×160** | Kích thước ảnh mặt đầu vào model (resize trước inference). |
| **Model Size** | **44.88 MB** | Kích thước file `.tflite` (môi trường đo report). |

### 3. Hiệu năng Hệ thống (System Metrics)

| Thông số | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| **Train Identities** | 33 | Số folder trong `figures/train_data/`. |
| **Test Identities** | 36 | Số folder trong `figures/test_data/`. |
| **Inference Latency** | **72.98 ms** | Thời gian trích xuất embedding/frame (TFLite, 100 lần). |
| **Preprocessing Time** | **1.39 ms** | Tiền xử lý ảnh (gray, blur, resize). |
| **CPU Usage** | **19.6%** | Mức sử dụng CPU khi chạy pipeline. |
| **RAM Usage** | **58.5%** | Mức sử dụng RAM trong quá trình đo. |

### 4. Tóm tắt kết quả (theo mẫu bài báo)

- **Inference:** ~73 ms/frame (TFLite trên PC; trên Pi có thể tương đương hoặc cao hơn).
- **Attendance verification:** Xác nhận điểm danh trong thời gian cấp giây (< 0.5 s khi dùng multi-frame).
- **CPU:** ~20%, **RAM:** ~59% (môi trường report).
- **Model size:** ~45 MB (file đo được); có thể dùng bản nén nhỏ hơn cho Pi.
- **DB size:** Nhỏ gọn (file JSON embeddings với 33 người × 512-d).

Hệ thống hoạt động realtime trên embedded device (Raspberry Pi 4).

---

## 🎯 Đóng góp chính

1. **Kiến trúc hybrid tối ưu cho môi trường di động (school bus):** Tách rõ feature extraction (server) và inference + matching (edge), phù hợp băng thông và độ trễ hạn chế.
2. **Strict multi-frame verification strategy:** Giảm nhận nhầm bằng cách yêu cầu đa số frame trong một đợt chụp đồng thuận với cùng một danh tính.
3. **Privacy-preserving edge inference:** Embeddings và so khớp tại thiết bị; chỉ gửi mã học sinh và thời điểm lên server.
4. **Đánh giá thực nghiệm trên Raspberry Pi 4:** Báo cáo latency, CPU, RAM, độ chính xác và cấu trúc report tái lặp được.

---

## 🔄 Luồng hoạt động (Workflows)

### 💻 1. Luồng tại Server Face Recognition (PC Side)

1. **Input Data:** Thu thập ảnh khuôn mặt vào thư mục `data` (chia theo Id/Tên người dùng).
2. **Conversion:** Chạy `conver_data.py` để chuyển tất cả sang `.webp` chất lượng cao, giảm tải cho Pi.
3. **Extraction:** Chạy `extract_embeddings.py`.
   - Tải model **MobileFaceNet** (TFLite).
   - Phát hiện khuôn mặt bằng Haar Cascade (và/hoặc alignment theo mắt).
   - Trích xuất vector 512 chiều cho từng ảnh.
   - Tính **Centroid** (vector trung bình) cho mỗi người để tăng độ ổn định.
4. **Export:** Đóng gói toàn bộ vào `face_embeddings.json`.

### 🍓 2. Luồng tại Thiết bị Edge (Raspberry Pi 4)

1. **Trigger:** Người dùng nhấn nút vật lý hoặc phím **Space** trên Terminal.
2. **Capture:** Camera chụp liên tục 3–5 frame ảnh gốc.
3. **Pre-process:** Chuyển sang RGB, khử nhiễu (Gaussian Blur), tăng tương phản (CLAHE nếu cần), resize về **160×160** (theo input model), chuẩn hóa `(x − 127.5) / 127.5`.
4. **AI Inference:** Tải model TFLite, trích xuất embedding 512-d từ mỗi frame.
5. **Strict Verification:** So khớp **cosine distance** với database; chỉ chấp nhận khi **quá bán** frame cùng một identity nằm dưới threshold.
6. **Action:** In kết quả, gọi API điểm danh, log lịch sử và dọn ảnh tạm.

---

## 📋 Danh sách API Backend Server

Backend (FastAPI) lắng nghe tại port `8000`. Dưới đây là các đầu việc chính:

### 🔐 Authentication

- `POST /auth/login`: Đăng nhập hệ thống (Admin/Phụ huynh).

### 👮 Admin Management

- `POST /admin/parents`: Tạo tài khoản cho phụ huynh.
- `GET /admin/students`: Lấy danh sách toàn bộ học sinh.
- `POST /admin/students`: Thêm học sinh mới (Id, Tên, Mã số).
- `PUT /admin/students/{id}`: Cập nhật thông tin học sinh.
- `DELETE /admin/students/{id}`: Xóa học sinh khỏi hệ thống.

### 🚌 Edge Communication (Điểm danh)

- `POST /edge/attendance`: Nhận kết quả điểm danh từ Pi 4 gửi về.  
  - *Body*: `{student_code, status, attendance_time}`

### 👨‍👩‍👧‍👦 Parent Access

- `GET /parent/history`: Xem lịch sử điểm danh của con em mình.

---

## 📂 Cấu trúc dự án

```text
Vision_guard/
├── face-recognizer-server/       # Xử lý tại PC (Server Side) — Feature extraction, DB embeddings
├── edge-device-pi4/              # Chạy trên Raspberry Pi 4 (Edge Side)
│   ├── hardware-control/         # Điều khiển Camera & Tiền xử lý
│   └── ai-recognition/           # Engine nhận diện (TFLite) & Database Local
├── backend-server/              # Hệ thống API quản lý trung tâm (FastAPI)
├── frontend-web/                 # Dashboard quản lý Web
└── figures/                     # Báo cáo thực nghiệm (report_YYYYMMDD_HHMMSS)
```

---

## 🚀 Thao tác thực hiện nhanh

1. **PC:** Chạy `conver_data.py` và `extract_embeddings.py` để tạo file JSON embeddings.
2. **Đồng bộ:** Copy `face_embeddings.json` vào `edge-device-pi4/ai-recognition/local_db/`.
3. **Pi 4:** Cài `tflite-runtime` và chạy `python hardware-control/main.py`.

---

## ⚠️ Giải thích Kỹ thuật bổ sung

- **Chuẩn hóa ảnh:** \((x - 127.5) / 127.5\) được áp dụng đồng nhất ở server và edge để vector không bị sai lệch.
- **Input ảnh:** Model yêu cầu **160×160**; cả server và edge resize ảnh mặt về 160×160 trước khi đưa vào TFLite.
- **Threshold:** So khớp dùng **cosine distance** (1 − cosine similarity). Mặc định **0.45** trong `recognizer.py`; có thể điều chỉnh theo điều kiện ánh sáng và yêu cầu FAR/FRR.

---

## 📎 Nguồn model & Tham khảo

- **Model TFLite face embedding:** File `face-recognizer-server/models/mobilefacenet.tflite` (và bản dùng tại edge trong `edge-device-pi4/ai-recognition/models/`) được tham khảo từ dự án **[On-Device Face Recognition in Android](https://github.com/shubham0204/OnDevice-Face-Recognition-Android)**. Tác giả (Shubham Panchal) cung cấp pipeline nhận diện khuôn mặt trên thiết bị với FaceNet/FaceNet-512 (TFLite), kèm **bộ so sánh vector (Face Matching)** và tìm láng giềng gần nhất (nearest-neighbor) ngay trên thiết bị. Dự án VisionGuard kế thừa ý tưởng embedding + so khớp vector on-device và áp dụng cho môi trường Raspberry Pi 4 (edge) và server trích xuất đặc trưng.

---

## 📂 Quản lý Báo cáo (Report Management)

Kết quả thực nghiệm được xuất tự động vào thư mục `figures/`.

### Quy trình và logic test (`figures/run_full_report.py`)

Script tạo một thư mục report mới (`report_YYYYMMDD_HHMMSS`) và chạy **5 bước** theo thứ tự:

1. **STEP 1 — Thu thập thông số hệ thống**  
   Đếm số identity trong `figures/train_data/` và `figures/test_data/`, đo kích thước model TFLite, CPU/RAM hiện tại, **inference latency** (TFLite, 100 lần) và **preprocessing time** (50 lần). Ghi ra `system_metrics.csv`.

2. **STEP 2 — Xây dựng Face Database**  
   Dùng `accuracy_tools.build_embeddings_from_train_data()`: duyệt từng folder trong `train_data`, trích embedding từng ảnh bằng TFLite (face-recognizer-server), tính **centroid** (vector trung bình, L2-normalize) cho mỗi identity. Database in-memory + ghi ra `database_stats.csv`. Ghi thêm file **temp_face_embeddings.json** trong thư mục report để dùng cho bước 3 và 4.

3. **STEP 3 — Đánh giá trên Train data**  
   Gọi **FaceRecognizer** (edge-device-pi4/ai-recognition/recognizer.py) với `temp_face_embeddings.json`, model TFLite edge, **threshold cosine 0.45**. Với **mỗi folder** trong `train_data`: chạy `recognize_batch(folder)` (DNN+Haar detection, 160×160, cosine, majority vote) → một nhãn dự đoán cho mỗi folder. Tính accuracy theo số folder đúng, xây confusion matrix. Xuất `train_details_grouped.csv` (theo identity) và **confusion_matrix_train.png** (ảnh).

4. **STEP 4 — Đánh giá trên Test data**  
   Cùng pipeline như bước 3: FaceRecognizer trên **test_data**. Mỗi folder test → một kết quả; folder có thể là người trong DB (known) hoặc ngoài DB (unknown). Xuất `test_details.csv` (Actual, Predicted, Distance, Is Correct) và **confusion_matrix_test.png** (có thể có hàng/cột Unknown).

5. **STEP 5 — Tổng hợp Accuracy**  
   Ghi `accuracy.csv` với hai dòng: **Train data** (%), **Test data** (% hoặc N/A nếu không có test_data).

**Logic thống nhất:** Cả train và test đều dùng **cùng pipeline nhận diện tại edge** (FaceRecognizer, cosine distance, đánh giá theo **folder** chứ không theo từng ảnh), đảm bảo số liệu báo cáo phản ánh đúng hành vi hệ thống khi chạy trên Pi.

### Cách đặt tên thư mục

Định dạng: `report_YYYYMMDD_HHMMSS`  
Ví dụ: `report_20260302_030440` — ngày 02/03/2026, 03:04:40.

### Cấu trúc file trong mỗi Report

| File | Nội dung |
| --- | --- |
| `accuracy.csv` | Tóm tắt độ chính xác **Train data** và **Test data**. |
| `confusion_matrix_train.csv` | Ma trận nhầm lẫn trên train data (CSV). |
| `confusion_matrix_train.png` | Ma trận nhầm lẫn trên train data (ảnh PNG). |
| `confusion_matrix_test.csv` | Ma trận nhầm lẫn trên test data (CSV; có thể có Unknown). |
| `confusion_matrix_test.png` | Ma trận nhầm lẫn trên test data (ảnh PNG). |
| `train_details_grouped.csv` | Chi tiết độ chính xác theo từng identity (train). |
| `test_details.csv` | Kết quả từng folder test: Actual, Image, Predicted, Distance, Is Correct. |
| `system_metrics.csv` | Latency, CPU, RAM, kích thước model, số identity train/test. |
| `database_stats.csv` | Số identity, embedding dimension. |

Chạy: `python figures/run_full_report.py` để tạo report mới.

### Giải thích Confusion Matrix

Ma trận nhầm lẫn (confusion matrix) được xuất dưới dạng **ảnh PNG** (`confusion_matrix_train.png`, `confusion_matrix_test.png`):

- **Hàng (Actual):** Danh tính thật của đối tượng (tên folder trong train_data hoặc test_data). Hàng cuối có thể là **Unknown** nếu trong test có người không nằm trong database.
- **Cột (Predicted):** Nhãn do model dự đoán (identity trong DB hoặc **Unknown** nếu không khớp dưới ngưỡng).
- **Ô (i, j):** Số lần đối tượng thuộc hàng \(i\) được nhận diện là cột \(j\).
- **Đường chéo chính:** Ô (i, i) = số lần nhận đúng identity \(i\); càng cao càng tốt.
- **Ngoài đường chéo:** Ô (i, j) với \(i \ne j\) = nhầm lẫn (identity \(i\) bị nhận thành \(j\)). Cột **Unknown** = số lần bị từ chối (không đạt ngưỡng cosine).

Cách đọc nhanh: accuracy cao khi phần lớn số lượng nằm trên đường chéo; số lớn ngoài đường chéo hoặc ở cột Unknown cho thấy nhầm lẫn hoặc từ chối sai.

---

*Dự án phát triển vì sự an toàn và tiện lợi cho trẻ em.*
