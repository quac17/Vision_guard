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
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        resized = cv2.resize(img, image_size)
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)

        save_path = os.path.join(processed_dir, file)
        cv2.imwrite(save_path, blurred)
        print("Processed:", save_path)


# Test riêng
if __name__ == "__main__":
    preprocess_images(
        "./dataset/raw",
        "./dataset/processed"
    )
