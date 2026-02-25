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
        # Sắp xếp mắt theo vị trí x
        eyes = sorted(eyes, key=lambda e: e[0])
        lex, ley, lew, leh = eyes[0] # Mắt trái (theo ảnh)
        rex, rey, rew, reh = eyes[-1] # Mắt phải (theo ảnh)
        
        # Tính toán tâm mắt (phải ép kiểu int cho OpenCV)
        l_center = (int(lex + lew//2), int(ley + leh//2))
        r_center = (int(rex + rew//2), int(rey + reh//2))
        
        # Tính góc xoay
        dy = r_center[1] - l_center[1]
        dx = r_center[0] - l_center[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Xoay ảnh xung quanh tâm mắt trái
        M = cv2.getRotationMatrix2D(l_center, angle, 1.0)
        aligned_face = cv2.warpAffine(roi_color, M, (w, h), flags=cv2.INTER_CUBIC)
        return aligned_face
    
    return roi_color

def get_face_embedding(model, img_path, device):
    """
    Đọc ảnh, Căn chỉnh khuôn mặt (Alignment), trích xuất vector 512 chiều
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
        
        # 3. Normalization ĐỒNG NHẤT: (x - 127.5) / 127.5 => dải [-1, 1]
        face_np = face_rgb.astype(np.float32)
        face_np = (face_np - 127.5) / 127.5
        
        # HWC sang CHW
        face_np = np.transpose(face_np, (2, 0, 1))
        input_tensor = torch.from_numpy(face_np).unsqueeze(0).to(device)
        
        # 4. Inference
        with torch.no_grad():
            embedding = model(input_tensor)
            # Chỉ normalize MUỘN sau khi tính centroid (không normalize ở đây)
            # → trả về raw embedding để build_embeddings tính centroid chính xác hơn
            embedding = embedding.cpu().numpy().flatten()

        return embedding.tolist()
    except Exception as e:
        print(f"Lỗi khi xử lý {img_path}: {e}")
        return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    # 1. Khởi tạo Model
    model = MobileFaceNet(embedding_size=512).to(device)
    
    # Load weights nếu file tồn tại
    if os.path.exists(MODEL_WEIGHTS):
        checkpoint = torch.load(MODEL_WEIGHTS, map_location=device)
        # Kiểm tra nếu là state_dict (đúng chuẩn)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
            print("Đã load weights (State Dict) thành công.")
        # Nếu lỡ lưu cả model
        elif isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
            print("Đã load Full Model thành công.")
        else:
            print(f"CẢNH BÁO: File weights có kiểu dữ liệu lạ ({type(checkpoint)}). Có thể kết quả sẽ không chính xác.")
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
