import os
import cv2
import json
import numpy as np

# Thử import tflite_runtime trước (nhẹ hơn), fallback sang tensorflow.lite
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("LỖI: Chưa cài đặt tflite-runtime hoặc tensorflow. Hãy chạy: pip install tflite-runtime")
        sys.exit(1)

# Cấu hình
DATA_DIR = "./data-convert"  # Folder chứa ảnh .webp đã convert
OUTPUT_JSON = "./face_embeddings.json"
MODEL_PATH = "./models/mobilefacenet.tflite" # Chuyển sang dùng .tflite

# Load Haar Cascade cho face và eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def align_face(image, gray, face_box):
    """
    Căn chỉnh khuôn mặt dựa trên vị trí hai mắt để tăng độ chính xác
    """
    x, y, w, h = face_box
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])
        lex, ley, lew, leh = eyes[0]
        rex, rey, rew, reh = eyes[-1]
        
        l_center = (int(lex + lew//2), int(ley + leh//2))
        r_center = (int(rex + rew//2), int(rey + reh//2))
        
        dy = r_center[1] - l_center[1]
        dx = r_center[0] - l_center[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        M = cv2.getRotationMatrix2D(l_center, angle, 1.0)
        aligned_face = cv2.warpAffine(roi_color, M, (w, h), flags=cv2.INTER_CUBIC)
        return aligned_face
    
    return roi_color

def get_face_embedding(interpreter, img_path):
    """
    Đọc ảnh, Căn chỉnh khuôn mặt, trích xuất vector 512 chiều bằng TFLite
    """
    try:
        image = cv2.imread(img_path)
        if image is None: return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0: return None
            
        # 1. Alignment & Cropping
        face_image = align_face(image, gray, faces[0])
        
        # 2. Tiền xử lý (Resize 112x112)
        face_resized = cv2.resize(face_image, (112, 112))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # 3. Normalization: (x - 127.5) / 127.5
        face_np = face_rgb.astype(np.float32)
        face_np = (face_np - 127.5) / 127.5 # [-1, 1]
        
        # TFLite input format: NHWC (1, 112, 112, 3)
        input_tensor = np.expand_dims(face_np, axis=0)
        
        # 4. Inference
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        
        embedding = interpreter.get_tensor(output_details[0]['index'])
        return embedding.flatten().tolist()
        
    except Exception as e:
        print(f"Lỗi khi xử lý {img_path}: {e}")
        return None

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"LỖI: Không tìm thấy model TFLite tại {MODEL_PATH}")
        return

    # 1. Khởi tạo TFLite Interpreter
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    print(f"Đã load TFLite model từ: {MODEL_PATH}")

    face_database = {
        "metadata": {
            "model": "MobileFaceNet",
            "dimension": 512,
            "description": "Face embeddings database for Vision Guard"
        },
        "data": {}
    }

    # 2. Duyệt qua từng folder người dùng (s1, s2, ...)
    print("Bắt đầu trích xuất đặc trưng...")
    for person_name in os.listdir(DATA_DIR):
        person_path = os.path.join(DATA_DIR, person_name)
        
        if not os.path.isdir(person_path):
            continue
            
        print(f"Đang xử lý: {person_name}")
        person_embeddings = []
        
        for scene_file in os.listdir(person_path):
            if scene_file.endswith(".webp"):
                img_path = os.path.join(person_path, scene_file)
                vector = get_face_embedding(interpreter, img_path)
                
                if vector:
                    person_embeddings.append(vector)
        
        # 3. Tính toán Centroid (Vector trung bình)
        if person_embeddings:
            # Chuyển sang numpy array để tính toán
            embeddings_array = np.array(person_embeddings)
            # Tính trung bình cộng các đặc trưng
            centroid = np.mean(embeddings_array, axis=0)
            
            # Normalize CENTROID một lần DUY NHẤT (không normalize từng embedding riêng)
            norm = np.linalg.norm(centroid)
            centroid = centroid / norm if norm > 0 else centroid
            face_database["data"][person_name] = centroid.tolist()

    # 4. Đóng gói vào JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(face_database, f, indent=2, ensure_ascii=False)
        
    print(f"\nHoàn thành! Đã lưu {len(face_database['data'])} người vào {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
