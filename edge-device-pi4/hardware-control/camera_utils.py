import cv2
import os
import time
import subprocess
from pynput import keyboard
from image_processor import preprocess_images

# Import pygame for sound (pip install pygame)
try:
    import pygame
    pygame.mixer.init()
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("Thông báo: Thư viện 'pygame' chưa được cài đặt. Sẽ không phát được âm thanh.")

def play_sound(sound_file):
    if not HAS_PYGAME:
        return
    
    try:
        # Load and play
        if os.path.exists(sound_file):
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
            # Non-blocking play, but since recognition is in a loop, it's okay.
            # If we want to wait, we'd loop while music is busy. 
            # But let's not block the camera thread too much.
        else:
            print(f"Cảnh báo: Không tìm thấy file âm thanh {sound_file}")
    except Exception as e:
        print(f"Lỗi khi phát âm thanh {sound_file}: {e}")

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
    input_status = {"start_capture": False, "exit": False, "current_mode": "on_bus"}

    def on_press(key):
        try:
            if key == keyboard.Key.space:
                input_status["start_capture"] = True
            elif hasattr(key, 'char'):
                if key.char == 'q':
                    input_status["exit"] = True
                elif key.char == 'c':
                    # Toggle mode
                    old_mode = input_status["current_mode"]
                    input_status["current_mode"] = "off_bus" if old_mode == "on_bus" else "on_bus"
                    print(f"\n[MODE] Đã chuyển từ {old_mode} -> {input_status['current_mode']}")
        except AttributeError:
            pass

    # Start listener thread
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("Nhấn 'Space' ngoài Terminal để chụp đợt 3 ảnh.")
    print("Nhấn 'c' ngoài Terminal để đổi chế độ (On-Bus / Off-Bus).")
    print("Nhấn 'q' ngoài Terminal để thoát chương trình.")
    print(f"CHẾ ĐỘ HIỆN TẠI: {input_status['current_mode']}")

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
                        recognizer.edge_mode = input_status["current_mode"] # Set mode from UI
                        print(f"Bắt đầu nhận diện tại mode: {recognizer.edge_mode}")
                        name, distance = recognizer.recognize_batch(processed_dir)
                        
                        print(f"=====================================")
                        print(f"KẾT QUẢ: {name}")
                        print(f"Độ lệch (Distance): {distance:.4f}")
                        print(f"=====================================")
                        
                        # Phát âm thanh phản hồi
                        sound_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "sound"))
                        if "Unknown" not in name:
                            # Thành công
                            if input_status["current_mode"] == "on_bus":
                                play_sound(os.path.join(sound_dir, "on_bus.mp3"))
                            else:
                                play_sound(os.path.join(sound_dir, "off_bus.mp3"))
                        else:
                            # Thất bại
                            play_sound(os.path.join(sound_dir, "reconize_fail.mp3"))

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
