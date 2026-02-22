# Vision Guard Backend Server

Hệ thống quản lý điểm danh học sinh trên xe buýt trường học sử dụng FastAPI, PostgreSQL và Docker.

## Tính năng
- **Admin**: Đăng nhập, quản lý tài khoản phụ huynh, quản lý thông tin học sinh (CRUD).
- **Phụ huynh**: Đăng nhập, đăng ký thông tin học sinh kèm 3-5 ảnh nhận dạng.
- **Thiết bị Edge**: API gửi dữ liệu điểm danh (lên xe/xuống xe).
- **Thông báo**: Batch process tự động gửi mail cho phụ huynh khi có dữ liệu điểm danh mới.
- **Docker**: Setup môi trường dễ dàng với Docker Compose.

## Cài đặt và Chạy

### 1. Chuẩn bị
Tạo file `.env` từ mẫu trong thư mục (đã được tạo sẵn với các giá trị mặc định). Lưu ý cấu hình SMTP nếu muốn gửi email thật.

### 2. Chạy với Docker (Khuyên dùng)

**Build hệ thống:**
```bash
docker-compose build
```

**Khởi chạy toàn bộ hệ thống:**
```bash
docker-compose up -d
```

**Khởi chạy riêng lẻ:**
```bash
# Chỉ chạy Database
docker-compose up -d db

# Chỉ chạy API Server
docker-compose up -d api

# Chỉ chạy Batch Notify
docker-compose up -d batch
```

**Dừng riêng lẻ:**
```bash
# Dừng Batch Notify
docker-compose stop batch

# Dừng API Server
docker-compose stop api
```

**Dừng hệ thống:**
```bash
docker-compose down
```

**Reset Database (Xóa toàn bộ dữ liệu và chạy lại init.sql):**
```bash
docker-compose down -v
docker-compose up -d db
```

**Xem Log:**
```bash
docker-compose logs -f [api|batch|db]
```

**Dump Database hiện tại về file init.sql:**
```bash
docker exec -t vision_guard_db pg_dump -U postgres vision_guard > db/init.sql
```

### 3. Chạy Local (Không dùng Docker)

Nếu bạn muốn chạy trực tiếp trên máy:

**Chạy API:**
```bash
uvicorn src.main:app --reload
```

**Chạy Batch:**
```bash
python -m src.batch_check
```

- API docs: `http://localhost:8000/docs`
- Database: PostgreSQL trên cổng `5432`

### 3. Database
File `db/init.sql` đã bao gồm cấu trúc bảng và một tài khoản admin mẫu:
- **Username**: admin
- **Password**: admin123 (Hash placeholder trong init.sql)

## API Endpoints chính

### Auth
- `POST /auth/login`: Đăng nhập lấy JWT Token.

### Admin
- `POST /admin/parents`: Tạo tài khoản cho phụ huynh.
- `GET /admin/students`: Xem danh sách học sinh.
- `POST /admin/students`: Thêm học sinh.

### Phụ huynh
- `POST /parent/register-student`: Đăng ký học sinh và upload ảnh (Sử dụng Multipart Form).

### Edge Device
- `POST /edge/attendance`: Thiết bị gửi dữ liệu điểm danh.

## Cấu trúc thư mục
- `src/main.py`: Entry point của ứng dụng.
- `src/models/`: Định nghĩa các bảng Database (SQLAlchemy).
- `src/schemas/`: Định nghĩa kiểu dữ liệu Input/Output (Pydantic).
- `src/routes/`: Các route API chia theo chức năng.
- `src/services/`: Logic xử lý (Auth, Email).
- `src/batch_check.py`: Script chạy ngầm kiểm tra và gửi email.
