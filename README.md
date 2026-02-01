# VisionGuard Bus: Smart AI Attendance System
---

## Giới thiệu
**VisionGuard Bus** là hệ thống điểm danh học sinh thông minh trên xe bus trường học sử dụng công nghệ nhận diện khuôn mặt. Hệ thống tích hợp thiết bị đầu cuối (**Edge Computing**) trên Raspberry Pi 4 để đảm bảo tính thời gian thực, giảm độ trễ và tăng cường bảo mật thông tin.

---

## Tính năng chính
-  **Xác thực sinh trắc học:** Nhận diện khuôn mặt học sinh cực nhanh (1-2 giây).
-  **Xử lý tại biên (Edge Processing):** Toàn bộ khâu chụp ảnh, khử nhiễu và nhận diện chạy trực tiếp trên Pi 4.
-  **Thông báo thời gian thực:** Tự động gửi Email/Thông báo cho phụ huynh ngay khi trẻ lên hoặc xuống xe.
-  **Quản lý tập trung:** Giao diện Web hiện đại để quản lý danh sách học sinh và theo dõi hành trình.
-  **Đồng bộ hóa:** Tự động đồng bộ dữ liệu khuôn mặt và cấu hình từ Server xuống thiết bị đầu cuối.

---

## Cấu trúc dự án

```text
smart-bus-system/
├── edge-device-pi4/      # Code chạy trên Raspberry Pi 4
│   ├── hardware-control/ # Điều khiển Camera & Nút bấm, tiền xử lý ảnh
│   ├── ai-recognition/   # Engine nhận diện khuôn mặt & Logic AI
│   └── config.yaml       # Cấu hình thiết bị
├── backend-server/       # Hệ thống API quản lý trung tâm (Node.js/Python)
├── frontend-web/        # Dashboard quản lý (React/Vue)
└── docs/                # Tài liệu kỹ thuật & Sơ đồ hệ thống
```

---

## Công nghệ sử dụng
- **Phần cứng:** Raspberry Pi 4 (8GB RAM), Pi Camera Module V2, Nút bấm vật lý.
- **Ngôn ngữ:** Python (Edge & AI), JavaScript (Frontend & Backend).
- **AI/ML:** OpenCV, Dlib hoặc TensorFlow Lite.
- **Backend:** Node.js (Express) hoặc Python (FastAPI/Flask).
- **Database:** MongoDB hoặc PostgreSQL.
- **Giao tiếp:** REST API hoặc MQTT.

---

## Hướng dẫn cài đặt nhanh

### 1. Cấu hình Raspberry Pi 4
```bash
cd edge-device-pi4
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python hardware-control/main.py
```

### 2. Khởi chạy Backend Server
```bash
cd backend-server
npm install
npm start
```

### 3. Khởi chạy Web UI
```bash
cd frontend-web
npm install
npm run dev
```

---

## Luồng hoạt động (Workflow)
1. **Học sinh lên xe:** Nhấn nút vật lý trên thiết bị.
2. **Chụp ảnh:** Camera chụp và xử lý ảnh (khử nhiễu, chuẩn hóa $640 \times 480$ px).
3. **Nhận diện:** Engine AI so khớp với cơ sở dữ liệu khuôn mặt cục bộ trên Pi.
4. **Gửi dữ liệu:** Kết quả nhận diện được gửi về Backend qua API.
5. **Thông báo:** Server ghi nhận lịch sử và gửi mail thông báo tức thì cho phụ huynh.

---

## Liên hệ
- **Dự án:** VisionGuard Bus
- **Người phát triển:** [Tên của bạn]
- **Email:** [Email của bạn]

---

Dự án này được phát triển vì sự an toàn của trẻ em.
Và sự an tâm của phụ huynh & nhà trường.