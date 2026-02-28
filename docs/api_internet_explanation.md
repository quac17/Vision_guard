# Giải thích: Call API qua Internet dành cho Backend đang chạy Local

## 1. Vấn đề
Khi bạn có 2 thiết bị (Edge Device - Raspberry Pi và Backend Server) nằm trên 2 mạng khác nhau (ví dụ: Pi dùng 4G, Laptop chạy Backend dùng Wifi nhà), Pi không thể gọi tới `localhost` hay IP nội bộ của Laptop.

## 2. Giải pháp không cần Deploy (Tunneling)
Bạn **có thể** gọi API qua internet mà không cần thuê Server (AWS, Heroku, v.v.) bằng cách sử dụng các công cụ **Tunneling**. Các công cụ này sẽ tạo một "đường ống" từ internet công cộng thẳng vào cổng (port) đang chạy trên máy local của bạn.

### Các công cụ phổ biến:
1.  **Ngrok (Khuyên dùng):**
    *   **Cách dùng:** Tải ngrok, chạy lệnh: `ngrok http 8000`
    *   **Kết quả:** Ngrok sẽ cho bạn một URL dạng `https://xyz-123.ngrok-free.app`.
    *   **Cấu hình:** Bạn chỉ cần copy URL này vào biến môi trường `API_BASE_URL` trên Raspberry Pi.
2.  **Cloudflare Tunnel (zrok, LocalTunnel):** Tương tự nhưng hoàn toàn miễn phí (Ngrok bản free thỉnh thoảng đổi URL nếu restart).

## 3. Cách triển khai trong code
Trong file `recognizer.py`, chúng ta sẽ sử dụng biến môi trường để linh hoạt:
- Khi test trên cùng 1 máy: `API_BASE_URL=http://localhost:8000`
- Khi chạy thực tế qua Internet: `API_BASE_URL=https://xyz-abc.ngrok-free.app`

## 4. Ưu và nhược điểm
- **Ưu điểm:** Cực kỳ nhanh, không tốn phí, không cần setup CI/CD hay Docker phức tạp trên mây.
- **Nhược điểm:** Phụ thuộc vào máy tính cá nhân (phải bật máy thì API mới sống), URL bản free có thể thay đổi sau mỗi lần khởi động lại tunnel.

---
**Kết luận:** Việc call API qua internet khi chưa deploy là **hoàn toàn khả thi** và là cách tốt nhất để prototype dự án IoT.
