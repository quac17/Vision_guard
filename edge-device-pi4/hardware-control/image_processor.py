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

        resized = cv2.resize(img, image_size)
        # Giảm kernel xuống (3, 3) để giữ lại các đặc trưng sắc nét của mặt (mắt, mũi, miệng)
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)

        # Đổi đuôi file sang .webp để nén tốt hơn
        file_name_no_ext = os.path.splitext(file)[0]
        save_path = os.path.join(processed_dir, file_name_no_ext + ".webp")
        
        # Lưu dưới dạng .webp với chất lượng cao (ví dụ: 90)
        # Định dạng này rất nhẹ và phù hợp cho IoT
        cv2.imwrite(save_path, blurred, [cv2.IMWRITE_WEBP_QUALITY, 90])
        print("Processed and saved as .webp:", save_path)


# Test riêng
if __name__ == "__main__":
    preprocess_images(
        "./dataset/raw",
        "./dataset/processed"
    )
