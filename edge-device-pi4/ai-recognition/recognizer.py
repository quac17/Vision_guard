import cv2
import json
import numpy as np
import os
import time

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

class FaceRecognizer:
    def __init__(self, model_path, db_path, threshold=1.0):
        self.threshold = threshold
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 1. Load Database
        self.database = self.load_database(db_path)
        
        # 2. Load TFLite Model
        if os.path.exists(model_path):
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"Đã load model TFLite từ: {model_path}")
        else:
            self.interpreter = None
            print(f"CẢNH BÁO: Không tìm thấy file model tại {model_path}")

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
        Trích xuất vector 512 chiều từ ảnh khuôn mặt (112x112)
        Đồng bộ hóa tiền xử lý với extract_embeddings.py trên PC Server.
        """
        if self.interpreter is None:
            # Nếu chạy Mock Mode, trả về vector ngẫu nhiên để không gây lỗi luồng
            return np.random.rand(512)

        # 1. Resize về kích thước chuẩn 112x112
        face_img = cv2.resize(face_img, (112, 112))
        
        # 2. Chuyển đổi màu sắc (MobileFaceNet yêu cầu định dạng RGB)
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        else:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
            
        # 3. Chuẩn hóa về dải [-1, 1] giống hệt PC Side
        face_img = face_img.astype(np.float32)
        face_img = (face_img - 127.5) / 127.5
        
        # 4. Kiểm tra shape model để tự động điều chỉnh NHWC/NCHW
        input_shape = self.input_details[0]['shape']
        if input_shape[3] == 3: # NHWC
            input_data = np.expand_dims(face_img, axis=0)
        else: # NCHW
            face_img = np.transpose(face_img, (2, 0, 1))
            input_data = np.expand_dims(face_img, axis=0)

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
        Yêu cầu mới: Sử dụng khoảng cách Euclid. Tất cả các mẫu trong đợt chụp
        đều phải vượt qua ngưỡng (threshold) đối với cùng một định danh.
        """
        batch_embeddings = []
        
        for file in os.listdir(image_dir):
            img_path = os.path.join(image_dir, file)
            img = cv2.imread(img_path)
            if img is None: continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            
            for (x, y, w, h) in faces:
                face_crop = img[y:y+h, x:x+w]
                embedding = self.get_embedding(face_crop)
                batch_embeddings.append(embedding)
                break 

        if not batch_embeddings:
            return "Unknown (No face detected)", 0.0

        best_person = "Unknown"
        min_avg_dist = float('inf')
        closest_name_anyway = "None"
        
        for name, db_embedding in self.database.items():
            db_vec = np.array(db_embedding)
            
            # Tính khoảng cách cho từng mẫu trong batch
            dists = [np.linalg.norm(sample - db_vec) for sample in batch_embeddings]
            current_avg_dist = np.mean(dists)
            
            # Luôn theo dõi người gần nhất tuyệt đối để báo cáo cho người dùng
            if current_avg_dist < min_avg_dist:
                min_avg_dist = current_avg_dist
                closest_name_anyway = name
            
            # Logic: Tất cả các mẫu đều phải nằm trong ngưỡng (threshold = 1.0)
            if all(d < self.threshold for d in dists):
                best_person = name

        if best_person == "Unknown":
            return f"Unknown (Closest: {closest_name_anyway})", min_avg_dist
            
        return best_person, min_avg_dist

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
