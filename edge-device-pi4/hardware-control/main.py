from camera_utils import capture_images
from image_processor import preprocess_images

RAW_DIR = "hardware-control/dataset/raw"
PROCESSED_DIR = "hardware-control/dataset/processed"

print("1. Chụp ảnh từ webcam")
print("2. Tiền xử lý ảnh")

choice = input("Chọn chức năng: ")

if choice == "1":
    capture_images(RAW_DIR)

elif choice == "2":
    preprocess_images(RAW_DIR, PROCESSED_DIR)

else:
    print("Lựa chọn không hợp lệ")
