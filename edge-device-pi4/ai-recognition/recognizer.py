import cv2
import json
import numpy as np
import os
import time
import requests
from datetime import datetime

TFLITE_AVAILABLE = False
try:
    try:
        import tflite_runtime.interpreter as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    print("--- CHẾ ĐỘ GIẢ LẬP (MOCK MODE) ---")
    print("Thông báo: Không tìm thấy thư viện TFLite. Hệ thống sẽ tạo vector ngẫu nhiên để test luồng.")

# Ngưỡng confidence cho face detection (DNN)
# Hạ xuống để giảm "no face detected" sai trong ánh sáng kém / góc nghiêng
DNN_CONF_MAIN = 0.5   # Ưu tiên detection >= 0.5
DNN_CONF_FALLBACK = 0.35  # Nếu không có ai >= 0.5 thì chấp nhận detection tốt nhất >= 0.35


class FaceRecognizer:
    def __init__(self, model_path, db_path, threshold=0.45):
        self.threshold = threshold
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Thiết lập OpenCV DNN Face Detector (rất nhạy và chính xác, chịu được góc nghiêng)
        self.dnn_model_path = os.path.join(os.path.dirname(model_path), "res10_300x300_ssd_iter_140000.caffemodel")
        self.dnn_proto_path = os.path.join(os.path.dirname(model_path), "deploy.prototxt")
        
        if not os.path.exists(self.dnn_model_path) or not os.path.exists(self.dnn_proto_path):
            print("Đang tải model Nhận diện khuôn mặt OpenCV DNN (chỉ tải 1 lần đầu)...")
            import urllib.request
            urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel", self.dnn_model_path)
            urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt", self.dnn_proto_path)
            
        self.face_net = cv2.dnn.readNetFromCaffe(self.dnn_proto_path, self.dnn_model_path)
        
        # 1. Load Database
        self.database = self.load_database(db_path)
        
        # 2. Load TFLite Model (Chỉ load nếu thư viện khả dụng)
        if TFLITE_AVAILABLE and os.path.exists(model_path):
            try:
                self.interpreter = tflite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"Đã load model TFLite từ: {model_path}")
            except Exception as e:
                self.interpreter = None
                print(f"Lỗi khi khởi tạo Interpreter: {e}")
        else:
            self.interpreter = None
            if not TFLITE_AVAILABLE:
                print("Hệ thống chạy Mock Mode vì thiếu thư viện tflite-runtime.")
            else:
                print(f"CẢNH BÁO: Không tìm thấy file model tại {model_path}")
        
        # 3. API Configuration
        self.api_base_url = os.getenv("API_BASE_URL", "http://100.85.100.28:8000")
        self.edge_mode = "on_bus" # Default mode
        print(f"API Base URL: {self.api_base_url}")

    def submit_attendance(self, student_code):
        """
        Gửi dữ liệu điểm danh về backend server
        """
        url = f"{self.api_base_url}/edge/attendance"
        payload = {
            "student_code": student_code,
            "status": self.edge_mode,
            "attendance_time": datetime.now().isoformat()
        }
        
        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                print(f"--- ĐIỂM DANH THÀNH CÔNG ---: {student_code}")
                return True
            else:
                print(f"LỖI API ({response.status_code}): {response.text}")
                return False
        except Exception as e:
            print(f"KHÔNG THỂ KẾT NỐI API: {e}")
            return False

    def _preprocess_for_detection(self, img):
        """
        Tăng tương phản ảnh (CLAHE trên kênh L) để face detection tốt hơn trong ánh sáng kém.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _detect_face_dnn(self, img, h_img, w_img):
        """ DNN detection: trả về (x, y, w, h) hoặc None. Dùng ngưỡng linh hoạt. """
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        best_face = None
        best_conf = 0.0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < DNN_CONF_FALLBACK:
                continue
            x1 = int(detections[0, 0, i, 3] * w_img)
            y1 = int(detections[0, 0, i, 4] * h_img)
            x2 = int(detections[0, 0, i, 5] * w_img)
            y2 = int(detections[0, 0, i, 6] * h_img)
            if confidence > best_conf:
                best_conf = confidence
                best_face = (max(0, x1), max(0, y1), min(w_img, x2) - max(0, x1), min(h_img, y2) - max(0, y1))
        return best_face

    def _detect_face_haar(self, gray):
        """ Fallback: Haar cascade mặt chính diện. Trả về (x, y, w, h) lớn nhất hoặc None. """
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            return None
        # Chọn mặt có diện tích lớn nhất
        return max(faces, key=lambda r: r[2] * r[3])

    def _get_one_face_box(self, img, gray, use_preprocess_retry=True):
        """
        Lấy một vùng mặt từ ảnh: ưu tiên DNN (ngưỡng linh hoạt), fallback Haar, có thể retry với ảnh đã CLAHE.
        """
        h_img, w_img = img.shape[:2]
        # 1. Thử DNN trên ảnh gốc
        face_box = self._detect_face_dnn(img, h_img, w_img)
        if face_box is not None:
            return face_box
        # 2. Fallback Haar trên ảnh xám
        face_box = self._detect_face_haar(gray)
        if face_box is not None:
            return face_box
        # 3. Retry: DNN + Haar trên ảnh đã tăng tương phản (ánh sáng kém)
        if use_preprocess_retry:
            enhanced = self._preprocess_for_detection(img)
            gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            face_box = self._detect_face_dnn(enhanced, h_img, w_img)
            if face_box is None:
                face_box = self._detect_face_haar(gray_enh)
        return face_box

    def align_face(self, image, gray, face_box):
        """
        Căn chỉnh khuôn mặt dựa trên mắt (phiên bản tối ưu cho Edge)
        """
        x, y, w, h = face_box
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        
        # Detect mắt — minNeighbors=5 đủ để lọc nhiễu mà không bỏ sót mắt thật
        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            # Ép kiểu int cho OpenCV
            l_center = (int(eyes[0][0] + eyes[0][2]//2), int(eyes[0][1] + eyes[0][3]//2))
            r_center = (int(eyes[-1][0] + eyes[-1][2]//2), int(eyes[-1][1] + eyes[-1][3]//2))
            
            dy = r_center[1] - l_center[1]
            dx = r_center[0] - l_center[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            M = cv2.getRotationMatrix2D(l_center, angle, 1.0)
            return cv2.warpAffine(roi_color, M, (w, h), flags=cv2.INTER_CUBIC)
        
        return roi_color

    def load_database(self, db_path):
        if os.path.exists(db_path):
            with open(db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Đã load database: {len(data['data'])} người.")
                return data['data']
        print("CẢNH BÁO: Không tìm thấy database JSON.")
        return {}

    def get_embedding(self, face_img):
        """
        Trích xuất vector 512 chiều từ ảnh khuôn mặt
        """
        if self.interpreter is None:
            # Nếu chạy Mock Mode, trả về vector ngẫu nhiên.
            # Tự động đồng bộ số chiều (128 hoặc 512) với database hiện tại để tránh lỗi shape
            mock_dim = len(list(self.database.values())[0]) if self.database else 512
            return np.random.rand(mock_dim)

        # 1. Resize 160x160 (theo yêu cầu input của model) và chuẩn hóa màu RGB
        face_img = cv2.resize(face_img, (160, 160))
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
        # 2. Chi tiết Normalization: (x - 127.5) / 127.5 => dải [-1, 1]
        # Đồng bộ 100% với extract_embeddings.py trên PC Server.
        face_np = face_rgb.astype(np.float32)
        face_np = (face_np - 127.5) / 127.5
        
        # 4. Kiểm tra shape model để tự động điều chỉnh NHWC/NCHW
        #    QUAN TRỌNG: luôn dùng face_np (đã normalize), KHÔNG dùng face_img gốc
        input_shape = self.input_details[0]['shape']
        if len(input_shape) == 4 and input_shape[1] == 3:  # NCHW: [1, 3, 160, 160]
            face_nchw = np.transpose(face_np, (2, 0, 1))  # HWC -> CHW
            input_data = np.expand_dims(face_nchw, axis=0).astype(np.float32)
        else:  # NHWC: [1, 160, 160, 3]
            input_data = np.expand_dims(face_np, axis=0).astype(np.float32)

        # Chạy Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        embedding = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # L2 Normalization
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def recognize_batch(self, image_dir):
        """
        Nhận diện dựa trên một tập hợp ảnh.
        Sử dụng Khoảng cách Cosine (Cosine Distance) thay vì L2 để tăng độ chính xác,
        để phù hợp với đặc thù của các model nhận diện (như MobileFaceNet chú trọng vào góc vector).
        """
        batch_embeddings = []
        
        for file in os.listdir(image_dir):
            img_path = os.path.join(image_dir, file)
            img = cv2.imread(img_path)
            if img is None: continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Bắt mặt: DNN (ngưỡng linh hoạt 0.35–0.5) → fallback Haar → retry với ảnh CLAHE
            best_face = self._get_one_face_box(img, gray, use_preprocess_retry=True)
            
            if best_face:
                x, y, w, h = best_face
                # 1. Thực hiện Alignment trước khi crop
                face_crop = self.align_face(img, gray, (x, y, w, h))
                # 2. Lấy embedding
                embedding = self.get_embedding(face_crop)
                batch_embeddings.append(embedding)

        if not batch_embeddings:
            return "Unknown (No face detected)", 0.0

        min_avg_dist     = float('inf')
        best_name_in_thresh = None   
        closest_name_anyway = "None" 
        
        for name, db_embedding in self.database.items():
            db_vec = np.array(db_embedding)

            # Tính khoảng cách Cosine Distance (Càng gần 0.0 càng giống)
            # Độ đo này đặc biệt hiệu quả với các model chuẩn hóa học theo góc (Angular Margin)
            dists = [1.0 - np.dot(sample, db_vec) for sample in batch_embeddings]
            current_avg_dist = np.mean(dists)

            # Track người gần nhất tuyệt đối
            if current_avg_dist < min_avg_dist:
                min_avg_dist = current_avg_dist
                closest_name_anyway = name

            # Áp dụng chiến thuật "Bỏ phiếu quá bán" (Majority Voting)
            # Tăng tính chịu đựng (Robustness) nếu có 1 ảnh bị nhòe/sai trễ trong 3 ảnh chụp
            pass_count = sum(1 for d in dists if d < self.threshold)
            
            # Nếu hơn một nửa số ảnh trong lượt (ví dụ 2/3) vượt qua ngưỡng quy định
            if pass_count > len(batch_embeddings) / 2:
                if best_name_in_thresh is None or current_avg_dist < min_avg_dist:
                    best_name_in_thresh = name

        if best_name_in_thresh is None:
            return f"Unknown (Closest: {closest_name_anyway}, dist={min_avg_dist:.3f})", min_avg_dist

        # Tự động gọi API điểm danh nếu nhận diện thành công
        # Format: (tên)_(mã học sinh) -> s1_SV001
        name_parts = best_name_in_thresh.split('_')
        if len(name_parts) > 1:
            student_code = name_parts[-1] # Lấy phần cuối cùng làm mã học sinh
            self.submit_attendance(student_code)
        else:
            print(f"Bỏ qua call API cho {best_name_in_thresh} (Không tìm thấy mã học sinh)")

        return best_name_in_thresh, min_avg_dist

    def cleanup(self, image_dir):
        """
        Xóa sạch ảnh trong thư mục sau khi nhận diện
        """
        for file in os.listdir(image_dir):
            file_path = os.path.join(image_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Lỗi khi xóa {file_path}: {e}")
        print(f"Đã dọn dẹp thư mục: {image_dir}")
