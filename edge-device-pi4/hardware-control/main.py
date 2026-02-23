from camera_utils import capture_images

RAW_DIR = "hardware-control/dataset/raw"
PROCESSED_DIR = "hardware-control/dataset/processed"

print("Chụp ảnh từ webcam và xử lý ảnh")

capture_images(RAW_DIR, PROCESSED_DIR)
