# Edge Device (Raspberry Pi 4)

Phân vùng chạy trên **Raspberry Pi 4**: điều khiển camera, chụp ảnh, tiền xử lý, chạy **TFLite** MobileFaceNet và so khớp với database embeddings tại chỗ. Kết quả điểm danh được gửi lên backend qua API (không gửi ảnh — privacy-preserving).

## Vai trò trong kiến trúc

- **Inference tại edge:** Không cần gửi ảnh lên server; so khớp danh tính trên Pi.
- **Multi-frame verification:** Chụp 3–5 frame mỗi lần, chỉ xác nhận khi đa số frame đồng thuận cùng một identity (strict verification).
- **Backend:** Gửi `student_code`, `status`, `attendance_time` qua `POST /edge/attendance`.

## Cấu trúc thư mục

```text
edge-device-pi4/
├── hardware-control/        # Camera, capture, tiền xử lý, trigger (Space/GPIO)
│   ├── camera_utils.py      # Capture ảnh, listener phím Space
│   ├── image_processor.py   # Preprocess (blur, resize, normalize)
│   └── main.py
└── ai-recognition/          # Nhận diện & database local
    ├── recognizer.py        # FaceRecognizer: TFLite + matching + API
    ├── local_db/
    │   └── face_embeddings.json
    ├── models/
    │   ├── mobilefacenet.tflite
    │   ├── deploy.prototxt
    │   └── res10_300x300_ssd_iter_140000.caffemodel  # OpenCV DNN face detector
    └── run_test.py          # Script đánh giá độ chính xác (test_data)
```

## Giải thuật chi tiết

### 1. Phần cứng & Trigger (`hardware-control`)

- **Camera:** Chụp ảnh qua OpenCV (hoặc Pi Camera).
- **Trigger:** Nhấn phím **Space** trên terminal (hoặc GPIO nút bấm) để bắt đầu một đợt điểm danh.
- **Capture:** Chụp liên tục **3–5 frame** trong một đợt để dùng cho multi-frame verification.

### 2. Phát hiện khuôn mặt (`recognizer.py`)

- **Ưu tiên:** OpenCV **DNN** face detector (Caffe: `deploy.prototxt` + `res10_300x300_ssd_iter_140000.caffemodel`) với ngưỡng confidence linh hoạt (0.35–0.5).
- **Fallback:** **Haar Cascade** mặt chính diện.
- **Ánh sáng kém:** Retry với ảnh đã tăng tương phản **CLAHE** (kênh L trong LAB) để cải thiện detection.

### 3. Tiền xử lý ảnh

- **Alignment:** Căn chỉnh theo hai mắt (rotation) giống server.
- **Resize:** 160×160 (theo input model), RGB.
- **Chuẩn hóa:** `(pixel - 127.5) / 127.5` (đồng bộ với face-recognizer-server).
- **Khử nhiễu:** Gaussian Blur (3×3) có thể áp dụng trước resize (tùy pipeline).

### 4. Trích xuất embedding (TFLite)

- **Model:** `mobilefacenet.tflite` (MobileFaceNet, 512-d).
- **Input:** Ảnh 160×160 đã chuẩn hóa (NHWC hoặc NCHW tùy model).
- **Output:** Vector 512 chiều, **L2-normalize** trước khi so khớp.

### 5. So khớp danh tính (Matching)

- **Độ đo:** **Cosine distance** = `1 - cosine_similarity` (với vector đã L2-normalize, tương đương về mặt góc với Euclidean trên sphere).
- **Threshold:** Mặc định **0.45** (có thể chỉnh theo FAR/FRR và điều kiện ánh sáng).
- **Multi-frame verification (strict):**
  - Với mỗi identity trong DB: đếm số frame có `distance < threshold`.
  - Chỉ chấp nhận identity khi **quá bán** frame (ví dụ 2/3 hoặc 3/5) nằm dưới ngưỡng với **cùng một** người.
  - Nếu không có ai thỏa → trả về "Unknown".

### 6. Gửi điểm danh lên backend

- Sau khi xác nhận danh tính: gọi `POST /edge/attendance` với `student_code` (trích từ tên trong DB, ví dụ `s1_SV001` → `SV001`), `status`, `attendance_time`.

## Thông số kỹ thuật (từ báo cáo)

| Thông số | Giá trị |
| --- | --- |
| Độ chính xác Edge Phase (test) | 96.92% |
| Inference latency | < 80 ms/frame (PC ~40 ms; Pi có thể cao hơn nhưng realtime) |
| Preprocessing | ~1.39 ms |
| Model size | ~0.82 MB |
| Threshold (cosine distance) | 0.45 (điều chỉnh được) |

## Cách chạy

1. Copy `face_embeddings.json` từ `face-recognizer-server` vào `ai-recognition/local_db/`.
2. Cài đặt: `tflite-runtime` (hoặc `tensorflow`), `opencv-python`, `requests`.
3. Chạy: `python hardware-control/main.py` (từ thư mục gốc `edge-device-pi4` hoặc theo cấu hình trong code).
4. Nhấn **Space** khi cần điểm danh; kết quả in ra và gửi lên backend (nếu cấu hình `API_BASE_URL`).

## Đánh giá độ chính xác (test)

- Script `ai-recognition/run_test.py` đọc ảnh từ thư mục test, chạy recognizer và xuất kết quả (Actual, Predicted, Distance, Is Correct) — tương tự dữ liệu trong `figures/report_*/test_details.csv`.
