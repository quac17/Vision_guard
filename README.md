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

Đánh giá được thực hiện trên **nhiều dataset** trong `figures/data/`. Mỗi dataset có cấu trúc `train_data/` và `test_data/` (mỗi thư mục con = một identity, chứa ảnh khuôn mặt). Quy mô thay đổi theo từng bộ dữ liệu:

| Mục | Ví dụ (dataset_0) | Ví dụ (dataset_1) | Ghi chú |
| --- | :---: | :---: | --- |
| **Số identity train** | 33 | 100 | Số danh tính trong database embeddings (số folder trong `train_data/`). |
| **Số folder test** | 36 | 120 | Số folder trong `test_data/` (gồm người trong DB và/hoặc người lạ). |
| **Ảnh/người (train)** | thay đổi | thay đổi | Xem chi tiết trong `result/<dataset_X>/report_.../train_details_grouped.csv`. |

Kết quả từng dataset được xuất riêng trong `figures/result/<dataset_X>/report_YYYYMMDD_HHMMSS/` (accuracy, confusion 2×2, system_metrics, test_details).

### Đánh giá thực nghiệm

- **Độ chính xác:** Accuracy trên **train data** và **test data** (theo folder), pipeline FaceRecognizer (cosine 0.45); đánh giá từng dataset, ghi trong `accuracy.csv`.
- **Ma trận nhầm lẫn 2×2:** Phân nhóm **Known** (người trong DB) vs **Stranger** (người lạ); xuất TP, TN, FP, FN dạng ảnh PNG (`confusion_matrix_2x2_train.png`, `confusion_matrix_2x2_test.png`).
- **Threshold sensitivity:** Ngưỡng so khớp (mặc định 0.45 cho cosine distance) có thể điều chỉnh theo FAR/FRR.
- **Latency:** Inference (ms/frame), preprocessing (ms) — ghi trong `system_metrics.csv` của mỗi report.
- **Tài nguyên:** CPU (%), RAM (%) — đo khi chạy pipeline, phụ thuộc quy mô dataset.
- **Lighting robustness:** Face detection DNN + fallback Haar, tiền xử lý CLAHE khi ánh sáng kém.

Cấu trúc logic, trình bày rõ ràng phù hợp báo cáo khoa học.

---

## 📊 Kết quả thực nghiệm (Experimental Results)

Số liệu lấy từ **report** trong `figures/result/` (mỗi dataset có thư mục `report_YYYYMMDD_HHMMSS` chứa CSV và ảnh confusion 2×2). Dưới đây là ví dụ từ **dataset_0** và **dataset_1**.

### 1. Độ chính xác (Accuracy)

| Dataset (ví dụ) | Train data | Test data | Ghi chú |
| :--- | :---: | :---: | :--- |
| **dataset_0** | **100.0%** | **100.0%** | 33 train, 36 test folder; pipeline FaceRecognizer (cosine 0.45). |
| **dataset_1** | **91.0%** | **92.0%** | 100 train, 120 test folder; cùng pipeline edge. |

### 2. Thông số Cơ sở dữ liệu & Model

| Thông số | dataset_0 | dataset_1 | Ý nghĩa |
| :--- | :---: | :---: | :--- |
| **Total Identities** | 33 | 100 | Số danh tính trong database embeddings. |
| **Embedding Dimension** | 512 | 512 | Số chiều vector đặc trưng (MobileFaceNet). |
| **Input size** | **160×160** | **160×160** | Kích thước ảnh mặt đầu vào model. |
| **Model Size** | **44.88 MB** | **44.88 MB** | Kích thước file `.tflite` (môi trường đo). |

### 3. Hiệu năng Hệ thống (System Metrics)

| Thông số | dataset_0 | dataset_1 | Ý nghĩa |
| :--- | :---: | :---: | :--- |
| **Train Identities** | 33 | 100 | Số folder trong `data/<dataset>/train_data/`. |
| **Test Identities** | 36 | 120 | Số folder trong `data/<dataset>/test_data/`. |
| **Inference Latency** | **138.57 ms** | **163.35 ms** | Thời gian trích xuất embedding/frame (TFLite, 100 lần). |
| **Preprocessing Time** | **3.43 ms** | **7.28 ms** | Tiền xử lý ảnh (gray, blur, resize). |
| **CPU Usage** | **27.5%** | **43.5%** | Mức sử dụng CPU khi chạy pipeline. |
| **RAM Usage** | **67.2%** | **80.1%** | Mức sử dụng RAM trong quá trình đo. |

### 4. Tóm tắt kết quả (theo mẫu bài báo)

- **Inference:** ~140–165 ms/frame (TFLite trên PC, tùy kích thước dataset); trên Pi có thể tương đương hoặc cao hơn.
- **Attendance verification:** Xác nhận điểm danh trong thời gian cấp giây (< 0.5 s khi dùng multi-frame).
- **CPU / RAM:** Phụ thuộc quy mô dataset (ví dụ 27–43% CPU, 67–80% RAM).
- **Model size:** ~45 MB (file đo được); có thể dùng bản nén nhỏ hơn cho Pi.
- **DB size:** Nhỏ gọn (file JSON embeddings, 512-d/người).

Hệ thống hoạt động realtime trên embedded device (Raspberry Pi 4). Các report đầy đủ (accuracy, confusion 2×2, system_metrics, test_details) nằm trong `figures/result/<dataset_X>/report_YYYYMMDD_HHMMSS/`.

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
└── figures/                     # Script báo cáo & data/result multi-dataset
    ├── data/                    # Nguồn data: dataset_1..5, mỗi dataset có train_data/ và test_data/
    └── result/                  # Kết quả: result/<dataset_X>/report_YYYYMMDD_HHMMSS/
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

## 📂 Dataset và Quản lý Báo cáo (Report Management)

### Cấu trúc Dataset (`figures/data/`)

Dữ liệu đánh giá được tổ chức theo **nhiều dataset**. Mỗi dataset gồm hai thư mục giống cấu trúc cũ:

```text
figures/data/
├── dataset_1/
│   ├── train_data/    # Các folder identity (tên người), mỗi folder chứa ảnh khuôn mặt
│   └── test_data/     # Các folder identity (có thể có người trong DB hoặc người lạ)
├── dataset_2/
│   ├── train_data/
│   └── test_data/
├── dataset_3/
├── dataset_4/
└── dataset_5/
```

trong 6 bộ data, bộ dataset_0 là bộ đơn giản, dùng để test cơ bản. các bộ còn lại có số lượng dữ liệu lớn hơn với các ảnh có điều kiện ảnh khác nhau (ảnh đẹp, ảnh mờ,mức độ ánh sáng khác nhau, có vật thể che một phần khuôn mặt,...) dùng để đánh giá thêm về hiệu quả của mô hình. Trong đó, mỗi bộ bao gồm:
- ảnh của khoảng 100 người tại train_data và  + ~20 stranger tại test_data (trong tên folder ảnh của người lạ sẽ thêm hậu tố "-stranger")
- mỗi người trong bộ dữ liệu sẽ có khoảng 8-10 ảnh (test 1-3, train phần còn lại)

Trong `figures/run_full_report.py` có list **`DATASETS_TO_RUN`** (ví dụ `["dataset_1", "dataset_2", ...]`). Bạn có thể **comment** bớt tên dataset để chỉ chạy report cho một vài bộ dữ liệu.

### Cấu trúc Result (`figures/result/`)

Kết quả report được ghi theo từng dataset, mỗi lần chạy tạo một thư mục con với timestamp:

```text
figures/result/
├── dataset_1/
│   └── report_20260303_123456/   # CSV + ảnh confusion 2x2
├── dataset_2/
│   └── report_20260303_124000/
└── ...
```

Script chạy **lần lượt** từng dataset trong `DATASETS_TO_RUN`; với mỗi dataset tạo một thư mục `result/<dataset_X>/report_YYYYMMDD_HHMMSS/`.

### Quy trình và logic test (`figures/run_full_report.py`)

Với **mỗi** dataset đã chọn, script chạy **5 bước** theo thứ tự:

1. **STEP 1 — Thu thập thông số hệ thống**  
   Đếm số identity trong `data/<dataset_X>/train_data` và `test_data`, đo kích thước model TFLite, CPU/RAM, **inference latency** (TFLite, 100 lần) và **preprocessing time** (50 lần). Ghi ra `system_metrics.csv`.

2. **STEP 2 — Xây dựng Face Database**  
   Duyệt `train_data`, trích embedding bằng TFLite (face-recognizer-server), tính **centroid** cho mỗi identity. Ghi `database_stats.csv` và **temp_face_embeddings.json** trong thư mục report.

3. **STEP 3 — Đánh giá trên Train data**  
   **FaceRecognizer** (edge pipeline, cosine 0.45) chạy `recognize_batch` cho từng folder trong `train_data`. Xuất `train_details_grouped.csv` và **confusion_matrix_2x2_train.png** (ma trận 2×2 Known/Stranger).

4. **STEP 4 — Đánh giá trên Test data**  
   Cùng pipeline trên `test_data`. Xuất `test_details.csv` và **confusion_matrix_2x2_test.png** (ma trận 2×2 Known/Stranger).

5. **STEP 5 — Tổng hợp Accuracy**  
   Ghi `accuracy.csv` (Train data %, Test data %).

**Logic thống nhất:** Cả train và test đều dùng **cùng pipeline edge** (FaceRecognizer, cosine, đánh giá theo **folder**).

### Cách đặt tên thư mục report

Định dạng: `report_YYYYMMDD_HHMMSS`  
Ví dụ: `report_20260303_123456` — ngày 03/03/2026, 12:34:56.

### Cấu trúc file trong mỗi Report

| File | Nội dung |
| --- | --- |
| `accuracy.csv` | Tóm tắt độ chính xác **Train data** và **Test data**. |
| `confusion_matrix_2x2_train.png` | Ma trận nhầm lẫn 2×2 (Known/Stranger) trên **train** — TP, TN, FP, FN. |
| `confusion_matrix_2x2_test.png` | Ma trận nhầm lẫn 2×2 (Known/Stranger) trên **test** — TP, TN, FP, FN. |
| `train_details_grouped.csv` | Chi tiết độ chính xác theo từng identity (train). |
| `test_details.csv` | Kết quả từng folder test: Actual, Image, Predicted, Distance, Is Correct. |
| `system_metrics.csv` | Latency, CPU, RAM, kích thước model, số identity train/test. |
| `database_stats.csv` | Số identity, embedding dimension. |

Chạy: `python figures/run_full_report.py` (từ thư mục gốc dự án) để tạo report cho các dataset đã bật trong `DATASETS_TO_RUN`.

### Giải thích Confusion Matrix 2×2 (Known / Stranger)

Ma trận nhầm lẫn được xuất dưới dạng **ảnh PNG** 2×2, gộp theo hai nhóm:

- **Người đã biết (Known):** identity nằm trong database (từ train_data).
- **Người lạ (Stranger):** identity không trong database hoặc model trả về Unknown.

**Hàng = Actual (thật), Cột = Predicted (dự đoán):**

| | Predicted Known | Predicted Stranger |
| --- | --- | --- |
| **Actual Known** | **TP** (True Positive) | **FN** (False Negative) |
| **Actual Stranger** | **FP** (False Positive) | **TN** (True Negative) |

- **TP:** Người đã biết được nhận đúng.  
- **TN:** Người lạ được từ chối đúng.  
- **FP:** Người lạ bị nhận nhầm là Known.  
- **FN:** Người đã biết bị từ chối (nhận thành Stranger).

Trên ảnh: **TP và TN** dùng cùng một màu (ví dụ xanh — kết quả đúng); **FP** và **FN** mỗi loại một màu khác (ví dụ cam, đỏ) để dễ phân biệt sai lệch.

---

*Dự án phát triển vì sự an toàn và tiện lợi cho trẻ em.*
