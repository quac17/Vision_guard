# Face Recognizer Server (PC Side)

Phân vùng **trung tâm xử lý dữ liệu** của VisionGuard: chuẩn hóa ảnh, trích xuất đặc trưng khuôn mặt (feature extraction) bằng **MobileFaceNet**, và đóng gói database embeddings để đưa xuống thiết bị edge.

## Vai trò trong kiến trúc

- **Input:** Ảnh khuôn mặt theo thư mục từng người (định dạng .pgm, .jpg, .png).
- **Output:** File `face_embeddings.json` chứa vector 512-d (hoặc centroid) cho mỗi identity, dùng cho edge inference.

## Giải thuật / Pipeline

### 1. Chuẩn hóa dữ liệu (`conver_data.py`)

- Đọc ảnh từ `data/` (cấu trúc thư mục theo từng người).
- Chuyển đổi hàng loạt sang **.webp** để giảm dung lượng và thống nhất định dạng cho edge.

### 2. Trích xuất đặc trưng (`extract_embeddings.py`)

- **Face detection:** OpenCV **Haar Cascade** (mặt chính diện) + phát hiện mắt để căn chỉnh (alignment).
- **Alignment:** Căn chỉnh khuôn mặt theo hai mắt (rotation) để tăng độ ổn định embedding.
- **Preprocessing:** Resize ảnh mặt về **160×160** (theo yêu cầu model), chuyển RGB, chuẩn hóa: `(pixel - 127.5) / 127.5` (dải [-1, 1]).
- **Model:** **MobileFaceNet** dạng **TFLite** (`models/mobilefacenet.tflite`).
- **Embedding:** Vector **512 chiều**, L2-normalize (nếu cần) để đồng bộ với edge.
- **Centroid (tùy chọn):** Có thể lưu trung bình các embedding theo từng người để giảm nhiễu.

### 3. Định dạng database

- **File:** `face_embeddings.json`.
- **Cấu trúc:** Mapping identity → embedding (list/vector 512-d) hoặc centroid.
- **Kích thước:** Nhỏ gọn (< 500 KB cho ~33 người), phù hợp copy sang Pi.

## Thông số kỹ thuật (từ báo cáo)

| Thông số | Giá trị |
| --- | --- |
| Embedding dimension | 512 |
| Input size | 160×160, RGB |
| Normalization | (x - 127.5) / 127.5 |
| Model size | ~0.82 MB (.tflite) |
| Độ chính xác PC Phase (train) | 99.59% |

## Cách chạy

1. Đặt ảnh vào `data/<tên_người>/`.
2. Chuyển đổi: `python conver_data.py` (ra thư mục `data-convert` hoặc tương đương).
3. Trích xuất: `python extract_embeddings.py` → tạo `face_embeddings.json`.
4. Copy `face_embeddings.json` vào `edge-device-pi4/ai-recognition/local_db/`.

## Cấu trúc thư mục gợi ý

```text
face-recognizer-server/
├── data/                    # Ảnh gốc theo từng identity
├── data-convert/            # Ảnh đã chuyển .webp (sau conver_data.py)
├── models/
│   └── mobilefacenet.tflite
├── conver_data.py
├── extract_embeddings.py
└── face_embeddings.json     # Output cho edge
```
