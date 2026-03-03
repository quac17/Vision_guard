# Edge Device (Raspberry Pi 4)

Phân vùng chạy trên **Raspberry Pi 4**: điều khiển camera, chụp ảnh, tiền xử lý, chạy **TFLite** MobileFaceNet và so khớp với database embeddings tại chỗ. Kết quả điểm danh được gửi lên backend qua API (không gửi ảnh — privacy-preserving).

## Vai trò trong kiến trúc

- **Inference tại edge:** Không cần gửi ảnh lên server; so khớp danh tính trên Pi.
- **Multi-frame verification:** Chụp 3–5 frame mỗi lần, chỉ xác nhận khi đa số frame đồng thuận cùng một identity (strict verification).
- **Backend:** Gửi `student_code`, `status`, `attendance_time` qua `POST /edge/attendance`.

## Cấu trúc thư mục

```text
edge-device-pi4/
├── hardware-control/           # Camera, capture, tiền xử lý, trigger (Space/GPIO)
│   ├── camera_utils.py         # Capture ảnh, listener phím Space, gọi recognizer
│   ├── image_processor.py      # Preprocess (blur, CLAHE, resize, normalize)
│   ├── main.py                 # Entry: capture_images(RAW_DIR, PROCESSED_DIR)
│   ├── dataset/                # Tạo khi chạy (raw/, processed/)
│   │   ├── raw/                # Ảnh gốc sau khi chụp
│   │   └── processed/          # Ảnh đã tiền xử lý → recognizer đọc từ đây
│   └── sound/                  # Tùy chọn: on_bus.mp3, off_bus.mp3, reconize_fail.mp3
├── ai-recognition/              # Nhận diện & database local
│   ├── recognizer.py           # FaceRecognizer: DNN/Haar detection, TFLite, matching, API
│   ├── local_db/
│   │   └── face_embeddings.json # Copy từ face-recognizer-server
│   ├── models/
│   │   ├── mobilefacenet.tflite
│   │   ├── deploy.prototxt
│   │   └── res10_300x300_ssd_iter_140000.caffemodel  # OpenCV DNN face detector
│   ├── run_test.py             # Đánh giá độ chính xác trên test_data (batch)
│   └── architechture.txt       # Mô tả luồng dữ liệu
└── test_reports/               # Output của run_test.py (accuracy, metrics CSV)
```

## Giải thuật chi tiết

### 1. Phần cứng & Trigger (`hardware-control`)

- **Camera:** Chụp ảnh qua OpenCV (hoặc Pi Camera).
- **Trigger:** Nhấn phím **Space** trên terminal (hoặc GPIO `BUTTON_PIN = 17`) để bắt đầu một đợt điểm danh.
- **Capture:** Chụp liên tục **3–5 frame** trong một đợt để dùng cho multi-frame verification.
- **Âm thanh (tùy chọn):** Pygame phát `on_bus.mp3` / `off_bus.mp3` / `reconize_fail.mp3` theo kết quả và `edge_mode`.

### 2. Phát hiện khuôn mặt (`recognizer.py`)

- **Ưu tiên:** OpenCV **DNN** face detector (Caffe: `deploy.prototxt` + `res10_300x300_ssd_iter_140000.caffemodel`) với ngưỡng confidence **0.5** (fallback **0.35** khi không có face ≥ 0.5).
- **Fallback:** **Haar Cascade** mặt chính diện.
- **Ánh sáng kém:** Retry với ảnh đã tăng tương phản **CLAHE** (kênh L trong LAB) để cải thiện detection.

### 3. Tiền xử lý ảnh

- **Alignment:** Căn chỉnh theo hai mắt (rotation) giống server.
- **Resize:** 160×160 (theo input model), RGB.
- **Chuẩn hóa:** `(pixel - 127.5) / 127.5` (đồng bộ với face-recognizer-server).
- **Khử nhiễu:** Gaussian Blur; CLAHE khi cần (trong pipeline tiền xử lý).

### 4. Trích xuất embedding (TFLite)

- **Model:** `mobilefacenet.tflite` (MobileFaceNet, 512-d).
- **Input:** Ảnh 160×160 đã chuẩn hóa.
- **Output:** Vector 512 chiều, **L2-normalize** trước khi so khớp.

### 5. So khớp danh tính (Matching)

- **Độ đo:** **Cosine distance** = `1 - cosine_similarity` (vector đã L2-normalize).
- **Threshold:** Mặc định **0.45** (trong `recognizer.py`; `hardware-control` hiện không truyền threshold — dùng mặc định).
- **Multi-frame verification (strict):**
  - Với mỗi identity trong DB: đếm số frame có `distance < threshold`.
  - Chỉ chấp nhận identity khi **quá bán** frame (ví dụ 2/3 hoặc 3/5) nằm dưới ngưỡng với **cùng một** người.
  - Nếu không có ai thỏa → trả về "Unknown".

### 6. Gửi điểm danh lên backend

- Sau khi xác nhận danh tính: gọi `POST /edge/attendance` với `student_code` (trích từ tên trong DB, ví dụ `s1_SV001` → `SV001`), `status` (= `edge_mode`: `on_bus` / `off_bus`), `attendance_time`.
- **API Base URL:** Biến môi trường `API_BASE_URL` (mặc định trong code nếu không set).

## Thông số kỹ thuật (đồng bộ với report trong `figures/result/`)

| Thông số | Giá trị |
| --- | --- |
| **Độ chính xác (test)** | 85.6–100% (tùy dataset; pipeline FaceRecognizer, cosine 0.45) |
| **Inference latency (PC)** | ~62–94 ms/frame (TFLite, 100 lần); trên Pi có thể cao hơn nhưng vẫn realtime |
| **Preprocessing** | ~1.35–2.21 ms |
| **Model size** | ~44.88 MB (mobilefacenet.tflite) |
| **Threshold (cosine distance)** | 0.45 (trong `recognizer.py`; có thể chỉnh theo FAR/FRR) |

## Cách chạy

1. **Chuẩn bị database:** Copy `face_embeddings.json` từ `face-recognizer-server` vào `ai-recognition/local_db/`.
2. **Model & DNN:** Đặt `mobilefacenet.tflite` trong `ai-recognition/models/`. File `deploy.prototxt` và `res10_300x300_ssd_iter_140000.caffemodel` có thể được tải tự động lần đầu (hoặc copy sẵn).
3. **Cài đặt:**  
   `pip install tflite-runtime opencv-python numpy requests`  
   Tùy chọn âm thanh: `pip install pygame`. Trên Pi có thể dùng `tflite-runtime` thay cho `tensorflow`.
4. **Chạy (từ thư mục `edge-device-pi4`):**  
   `python hardware-control/main.py`  
   Đường dẫn ảnh trong code: `hardware-control/dataset/raw`, `hardware-control/dataset/processed` (tạo tự động khi cần).
5. **Nhấn Space** khi cần điểm danh; kết quả in ra và gửi lên backend nếu `API_BASE_URL` được cấu hình (env hoặc mặc định trong `recognizer.py`).

## Đánh giá độ chính xác

- **Tại edge:** Script `ai-recognition/run_test.py` đọc ảnh từ thư mục test, chạy FaceRecognizer và xuất kết quả (Actual, Predicted, Distance, Is Correct); có thể lưu vào `test_reports/`.
- **Báo cáo đầy đủ (PC):** Dùng `figures/run_full_report.py` để đánh giá theo nhiều dataset (accuracy, confusion 2×2, TAR/FAR/FRR, ROC, EER). Kết quả trong `figures/result/<dataset_X>/report_YYYYMMDD_HHMMSS/`.
