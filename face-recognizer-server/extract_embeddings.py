import os
import cv2
import json
import numpy as np
import torch
from mobilefacenet import MobileFaceNet

# Cấu hình
DATA_DIR = "./data-convert"  # Folder chứa ảnh .webp đã convert
OUTPUT_JSON = "./face_embeddings.json"
MODEL_WEIGHTS = "./models/mobilefacenet_pretrained.pth" # Đường dẫn tới weights (nếu có)

# import face_recognition (Đã chuyển sang dùng OpenCV để dễ cài đặt trên Windows)

# Load Haar Cascade cho face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_face_embedding(model, img_path, device):
    """
    Đọc ảnh, Phát hiện bằng OpenCV Haar Cascade, trích xuất vector 512 chiều
    """
    try:
        # 1. Load ảnh bằng OpenCV
        image = cv2.imread(img_path)
        if image is None:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Phát hiện vị trí khuôn mặt (Detection)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return None
            
        # Lấy khuôn mặt đầu tiên (x, y, w, h)
        x, y, w, h = faces[0]
        face_image = image[y:y+h, x:x+w]
        
        # 3. Tiền xử lý (Resize 112x112 cho MobileFaceNet)
        # Thay thế torchvision bằng cv2 + numpy để tránh lỗi Segmentation fault
        face_resized = cv2.resize(face_image, (112, 112))
        
        # BGR sang RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # [0, 255] -> [0, 1] -> [-1, 1] (theo Normalize mean=0.5, std=0.5)
        face_np = face_rgb.astype(np.float32) / 255.0
        face_np = (face_np - 0.5) / 0.5
        
        # HWC sang CHW (C, H, W)
        face_np = np.transpose(face_np, (2, 0, 1))
        
        # Thêm chiều batch và chuyển sang Tensor
        input_tensor = torch.from_numpy(face_np).unsqueeze(0).to(device)
        
        # 4. Inference
        with torch.no_grad():
            embedding = model(input_tensor)
            embedding = embedding.cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)
            
        return embedding.tolist()
    except Exception as e:
        print(f"Lỗi khi xử lý {img_path}: {e}")
        return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    # 1. Khởi tạo Model
    model = MobileFaceNet(embedding_size=512).to(device)
    
    # Load weights nếu file tồn tại, nếu không sẽ chạy với weights ngẫu nhiên (Demo)
    if os.path.exists(MODEL_WEIGHTS):
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        print("Đã load weights thành công.")
    else:
        print("CẢNH BÁO: Không tìm thấy file weights. Model sẽ chạy với tham số ngẫu nhiên.")
    
    model.eval()

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
                vector = get_face_embedding(model, img_path, device)
                
                if vector:
                    person_embeddings.append(vector)
        
        # 3. Tính toán Centroid (Vector trung bình)
        if person_embeddings:
            # Chuyển sang numpy array để tính toán
            embeddings_array = np.array(person_embeddings)
            # Tính trung bình cộng các đặc trưng
            centroid = np.mean(embeddings_array, axis=0)
            
            # QUAN TRỌNG: Re-normalize vector trung bình (L2 Normalization)
            # Điều này đưa vector về mặt cầu đơn vị, giúp so khớp chính xác hơn
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            
            face_database["data"][person_name] = centroid.tolist()

    # 4. Đóng gói vào JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(face_database, f, indent=2, ensure_ascii=False)
        
    print(f"\nHoàn thành! Đã lưu {len(face_database['data'])} người vào {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
