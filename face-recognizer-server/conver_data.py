import cv2
import os

def convert_pgm_to_webp(src_dir, dest_dir):
    """
    Duyệt toàn bộ thư mục src_dir, tìm các file ảnh và convert sang .webp
    Giữ nguyên cấu trúc thư mục con trong dest_dir.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    print(f"Bắt đầu convert từ {src_dir} sang {dest_dir}...")
    
    count = 0
    # Duyệt đệ quy qua các thư mục con (s1, s2, ...)
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            # Kiểm tra định dạng ảnh (thường là .pgm trong tập dữ liệu này)
            if file.lower().endswith(('.pgm', '.ppm', '.jpg', '.jpeg', '.png')):
                src_path = os.path.join(root, file)
                
                # Tạo đường dẫn tương ứng trong thư mục đích
                rel_path = os.path.relpath(root, src_dir)
                target_folder = os.path.join(dest_dir, rel_path)
                
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                
                # Đổi tên file sang .webp
                file_name_no_ext = os.path.splitext(file)[0]
                dest_path = os.path.join(target_folder, file_name_no_ext + ".webp")
                
                # Đọc và ghi
                img = cv2.imread(src_path)
                if img is not None:
                    # Lưu với chất lượng cao (95) để đảm bảo độ chính xác cho AI
                    cv2.imwrite(dest_path, img, [cv2.IMWRITE_WEBP_QUALITY, 95])
                    count += 1
                    if count % 50 == 0:
                        print(f"Đã convert {count} ảnh...")
                else:
                    print(f"Lỗi: Không thể đọc {src_path}")

    print(f"Xong! Đã convert tổng cộng {count} file sang định dạng .webp.")

if __name__ == "__main__":
    DATA_DIR = "./data"
    DEST_DIR = "./data-convert"
    
    convert_pgm_to_webp(DATA_DIR, DEST_DIR)