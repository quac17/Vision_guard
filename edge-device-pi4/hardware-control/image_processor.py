import cv2
import os


def preprocess_images(
    raw_dir,
    processed_dir,
    image_size=(224, 224)
):
    os.makedirs(processed_dir, exist_ok=True)

    for file in os.listdir(raw_dir):
        img_path = os.path.join(raw_dir, file)
        # Đọc ảnh màu để giữ tối đa đặc trưng cho model MobileFaceNet
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Giữ nguyên độ phân giải gốc để nét nhất có thể, chỉ đổi format nén
        # Đổi đuôi file sang .webp để nén tốt hơn
        file_name_no_ext = os.path.splitext(file)[0]
        save_path = os.path.join(processed_dir, file_name_no_ext + ".webp")
        
        # Lưu dưới dạng .webp với chất lượng cao
        # Định dạng này rất nhẹ nhưng vẫn giữ độ nét
        cv2.imwrite(save_path, img, [cv2.IMWRITE_WEBP_QUALITY, 95])
        print("Processed and saved raw as .webp:", save_path)


# Test riêng
if __name__ == "__main__":
    preprocess_images(
        "./dataset/raw",
        "./dataset/processed"
    )
