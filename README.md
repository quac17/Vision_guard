# VisionGuard: Hệ thống Nhận diện Khuôn mặt Điểm danh Thông minh
---

## 🌟 Giới thiệu
**VisionGuard** là giải pháp điểm danh thông minh ứng dụng công nghệ nhận diện khuôn mặt tiên tiến. Hệ thống được thiết kế theo mô hình **Edge Computing**, kết hợp giữa sức mạnh xử lý của PC Server để trích xuất dữ liệu và khả năng vận hành thực tế tối ưu trên **Raspberry Pi 4**.

---

## 🏗️ Kiến trúc Hệ thống

Dự án được chia thành các phân vùng chính:

### 1. Phân vùng PC Server (`face-recognizer-server`)
Đóng vai trò là "Trung tâm xử lý dữ liệu", thực hiện các nhiệm vụ:
- **Chuẩn hóa dữ liệu:** Chuyển đổi hàng loạt ảnh từ nhiều định dạng (.pgm, .jpg, .png) sang `.webp` để tối ưu dung lượng.
- **Trích xuất đặc trưng (Feature Extraction):** Sử dụng mô hình **MobileFaceNet** (PyTorch) để biến đổi khuôn mặt thành vector đặc trưng **512 chiều**.
- **Đóng gói Database:** Lưu trữ kết quả dưới dạng file `face_embeddings.json`.

### 2. Phân vùng Edge Device (`edge-device-pi4`)
Chạy trực tiếp trên Raspberry Pi 4 để nhận diện thời gian thực:
- **Xử lý phần cứng:** Điều khiển Camera, quản lý vòng lặp sự kiện (nhấn phím Space để điểm danh).
- **Tiền xử lý ảnh:** Khử nhiễu (Gaussian Blur), Resize, và chuẩn hóa ảnh.
- **Nhận diện AI:** Sử dụng **TFLite** để chạy model MobileFaceNet nhẹ, so khớp danh tính bằng khoảng cách **Euclidean**.

### 3. Phân vùng Backend & Web (`backend-server` & `frontend-web`)
- **Backend:** FastAPI cung cấp API quản lý tập trung, lưu trữ lịch sử và gửi thông báo.
- **Frontend:** Giao diện Dashboard để quản lý danh sách học sinh và theo dõi điểm danh.

---

## 🔄 Luồng hoạt động (Workflows)

### 💻 1. Luồng tại Server Face Recognition (PC Side)
Quy trình chuẩn bị "Bộ não" cho hệ thống:
1.  **Input Data:** Thu thập ảnh khuôn mặt vào thư mục `data` (chia theo Id/Tên người dùng).
2.  **Conversion:** Chạy `conver_data.py` để chuyển tất cả sang `.webp` chất lượng cao, giảm tải cho Pi.
3.  **Extraction:** Chạy `extract_embeddings.py`.
    - Tải model **MobileFaceNet**.
    - Phát hiện khuôn mặt bằng Haar Cascade.
    - Trích xuất vector 512 chiều cho từng ảnh.
    - Tính toán **Centroid** (vector trung bình) cho mỗi người để tăng độ ổn định.
4.  **Export:** Đóng gói toàn bộ vào `face_embeddings.json`.

### 🍓 2. Luồng tại Thiết bị Edge (Raspberry Pi 4)
Quy trình nhận diện tại hiện trường:
1.  **Trigger:** Người dùng nhấn nút vật lý hoặc phím **Space** trên Terminal.
2.  **Capture:** Camera chụp liên tục 3-5 frame ảnh gốc.
3.  **Pre-process:** 
    - Chuyển sang ảnh màu RGB.
    - Khử nhiễu bằng Gaussian Blur (3x3).
    - Resize về chuẩn **112x112**.
4.  **AI Inference:** 
    - Tải model TFLite.
    - Trích xuất vector đặc trưng 512 chiều từ các frame đã chụp.
5.  **Strict Verification:** 
    - So khớp khoảng cách **Euclidean** với Database.
    - **Điều kiện:** TẤT CẢ các frame trong đợt chụp phải đều nằm trong ngưỡng (threshold) mới xác nhận danh tính.
6.  **Action:** In kết quả lên màn hình, log lịch sử và dọn dẹp bộ nhớ ảnh tạm.

---

## � Danh sách API Backend Server

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
├── face-recognizer-server/       # Xử lý tại PC (Server Side)
├── edge-device-pi4/              # Chạy trên Raspberry Pi 4 (Edge Side)
│   ├── hardware-control/         # Điều khiển Camera & Tiền xử lý
│   └── ai-recognition/           # Engine nhận diện & Database Local
├── backend-server/               # Hệ thống API quản lý trung tâm
└── frontend-web/                 # Dashboard quản lý Web
```

---

## 🚀 Thao tác thực hiện nhanh

1.  **PC**: Chạy `conver_data.py` và `extract_embeddings.py` để lấy file JSON.
2.  **Đồng bộ**: Copy `face_embeddings.json` vào `edge-device-pi4/ai-recognition/local_db/`.
3.  **Pi 4**: Cài `tflite-runtime` và chạy `python hardware-control/main.py`.

---

## ⚠️ Giải thích Kỹ thuật bổ sung
- **Công thức chuẩn hóa:** $(x - 127.5) / 127.5$ được áp dụng đồng nhất ở cả hai phía để đảm bảo vector không bị sai lệch.
- **Euclidean Threshold:** Mặc định là **1.0**. Có thể điều chỉnh trong `recognizer.py` tùy theo điều kiện ánh sáng thực tế.

---
*Dự án phát triển bởi sự an toàn và tiện lợi.*