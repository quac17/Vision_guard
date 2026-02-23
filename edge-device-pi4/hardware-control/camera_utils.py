import cv2
import os
import time
from pynput import keyboard
from image_processor import preprocess_images

# import RPi.GPIO as GPIO

BUTTON_PIN = 17


def capture_images(
    save_dir,
    processed_dir=None,
    total_images=3,
    camera_index=0,
    delay=1
):
    os.makedirs(save_dir, exist_ok=True)

    # Sử dụng CAP_V4L2 để khởi động nhanh hơn trên Linux/Pi
    # Nếu chạy trên Windows thì dùng default hoặc CAP_DSHOW
    backend = cv2.CAP_V4L2 if os.name != 'nt' else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, backend)
    
    # Thiết lập nhanh độ phân giải thấp để camera sẵn sàng sớm hơn
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Giảm buffer size để không bị trễ ảnh cũ
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    count = 0

    # Dictionary to track input status from listener thread
    input_status = {"start_capture": False, "exit": False}

    def on_press(key):
        try:
            if key == keyboard.Key.space:
                input_status["start_capture"] = True
            elif hasattr(key, 'char') and key.char == 'q':
                input_status["exit"] = True
        except AttributeError:
            pass

    # Start listener thread
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("Nhấn 'Space' ngoài Terminal để chụp đợt 3 ảnh.")
    print("Nhấn 'q' ngoài Terminal để thoát chương trình.")

    try:
        while True:
            # We still need to read once to keep the buffer clean if using a live stream
            ret, frame = cap.read()
            if not ret:
                print("Không mở được camera")
                break

            if input_status["start_capture"]:
                print(f"\nĐang chụp {total_images} ảnh...")
                for _ in range(total_images):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    count += 1
                    # Lưu ảnh màu trực tiếp để đồng bộ với Database nhận diện
                    path = os.path.join(save_dir, f"{count}.webp")
                    # Lưu với chất lượng 100 để giữ tối đa thông tin ảnh gốc
                    cv2.imwrite(path, frame, [cv2.IMWRITE_WEBP_QUALITY, 100])
                    print(f"Saved: {path}") 
                    time.sleep(delay)
                
                input_status["start_capture"] = False
                
                if processed_dir:
                    print("Đang tiền xử lý và dọn dẹp ảnh gốc...")
                    preprocess_images(save_dir, processed_dir)
                    
                    # Bước nhận diện khuôn mặt (New phase)
                    print("--- Bắt đầu nhận diện danh tính ---")
                    try:
                        # Khởi tạo recognizer tại chỗ (hoặc có thể đưa ra ngoài để tối ưu hơn)
                        import sys
                        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ai-recognition")))
                        from recognizer import FaceRecognizer
                        
                        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ai-recognition", "models", "mobilefacenet.tflite"))
                        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ai-recognition", "local_db", "face_embeddings.json"))
                        
                        recognizer = FaceRecognizer(model_path, db_path)
                        name, distance = recognizer.recognize_batch(processed_dir)
                        
                        print(f"=====================================")
                        print(f"KẾT QUẢ: {name}")
                        print(f"Độ lệch (Distance): {distance:.4f}")
                        print(f"=====================================")
                        
                        # Dọn dẹp ảnh đã xử lý sau khi nhận diện xong (như yêu cầu architechture)
                        recognizer.cleanup(processed_dir)
                        
                    except Exception as e:
                        print(f"Lỗi trong quá trình nhận diện: {e}")

                    # Xóa ảnh gốc
                    for file in os.listdir(save_dir):
                        file_path = os.path.join(save_dir, file)
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        except Exception as e:
                            print(f"Lỗi khi xóa {file_path}: {e}")
                    print("Đã dọn dẹp xong.")

                print("Xong đợt. Nhấn 'Space' để tiếp tục hoặc 'q' để thoát.")

            if input_status["exit"]:
                print("Đang thoát...")
                break
            
            # Small sleep to prevent high CPU usage in the while loop
            time.sleep(0.01)

    finally:
        cap.release()
        listener.stop()

# Test riêng file này
if __name__ == "__main__":
    capture_images("../dataset/raw/user_01")
