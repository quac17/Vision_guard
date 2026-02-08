import cv2
import os
import time
# import RPi.GPIO as GPIO

BUTTON_PIN = 17


def capture_images(
    save_dir,
    total_images=10,
    camera_index=0,
    delay=0.3
):
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    count = 0

    print("Nhấn 'c' để bắt đầu chụp ảnh")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không mở được camera")
            break

        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            print("Đang chụp ảnh...")
            while count < total_images:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                path = os.path.join(save_dir, f"img_{count}.jpg")
                cv2.imwrite(path, gray)
                print("Saved:", path)

                count += 1
                time.sleep(delay)
            break

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# def capture_images_with_button(
#     save_dir,
#     total_images=10,
#     camera_index=0,
#     delay=0.3
# ):
#     os.makedirs(save_dir, exist_ok=True)

#     GPIO.setmode(GPIO.BCM)
#     GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

#     cap = cv2.VideoCapture(camera_index)
#     count = 0

#     print("Chờ nhấn nút để chụp ảnh...")

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             cv2.imshow("Webcam", frame)

#             # Nếu nút được nhấn
#             if GPIO.input(BUTTON_PIN) == GPIO.LOW:
#                 print("Đã nhấn nút, bắt đầu chụp ảnh...")
#                 time.sleep(0.3)  # debounce

#                 while count < total_images:
#                     ret, frame = cap.read()
#                     if not ret:
#                         break

#                     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                     path = os.path.join(save_dir, f"img_{count}.jpg")
#                     cv2.imwrite(path, gray)
#                     print("Saved:", path)

#                     count += 1
#                     time.sleep(delay)

#                 break

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
#         GPIO.cleanup()


# Test riêng file này
if __name__ == "__main__":
    capture_images("../dataset/raw/user_01")
