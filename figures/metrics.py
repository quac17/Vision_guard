"""
metrics.py — Đo các thông số kỹ thuật cho báo cáo Vision Guard
Tự chứa trong figures/ — không phụ thuộc vào folder khác ngoài MobileFaceNet model.
"""
import os
import sys
import time
import json
import numpy as np
import cv2

_FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
_SERVER_DIR  = os.path.join(_FIGURES_DIR, "../face-recognizer-server")
for _p in [_FIGURES_DIR, _SERVER_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ──────────────────────────────────────────────────────────────
# 1. Storage Size
# ──────────────────────────────────────────────────────────────
def get_dir_size(path):
    """Tính tổng dung lượng folder (bytes), duyệt đệ quy."""
    total = 0
    for dp, _, fns in os.walk(path):
        for f in fns:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


# ──────────────────────────────────────────────────────────────
# 2. Inference Latency — PyTorch (PC Phase)
# ──────────────────────────────────────────────────────────────
def measure_inference_time(model_path, data_sample_dir=None, is_tflite=False, iterations=100):
    """
    Đo thời gian inference trung bình (ms) trên 100 lần.
    data_sample_dir: nếu cung cấp, load ảnh thực tế thay vì random noise.
    """
    input_data = np.random.randn(1, 112, 112, 3).astype(np.float32)

    if data_sample_dir and os.path.exists(data_sample_dir):
        for root, _, fns in os.walk(data_sample_dir):
            for f in fns:
                if f.lower().endswith(('.webp', '.jpg', '.jpeg', '.pgm')):
                    img = cv2.imread(os.path.join(root, f))
                    if img is not None:
                        img = cv2.resize(img, (112, 112))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        input_data = (img.astype(np.float32) - 127.5) / 127.5
                        input_data = np.expand_dims(input_data, axis=0)
                        break
            if input_data.shape == (1, 112, 112, 3):
                break

    print(f"\n--- ĐO HIỆU NĂNG AI ({'TFLite' if is_tflite else 'PyTorch'}) ---")

    if is_tflite:
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                print("  ⚠ Không có tflite — bỏ qua đo latency TFLite")
                return 0.0
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()
        interpreter.set_tensor(inp[0]['index'], input_data)
        interpreter.invoke()  # warm-up
        t0 = time.time()
        for _ in range(iterations):
            interpreter.set_tensor(inp[0]['index'], input_data)
            interpreter.invoke()
        avg_ms = (time.time() - t0) / iterations * 1000

    else:
        import torch
        from model_loader import load_mobilefacenet
        model, device, ok = load_mobilefacenet(model_path, verbose=True)
        if not ok:
            print("  ⚠ Model dùng random weights — latency đo được vẫn có giá trị tham khảo")

        t_in = torch.from_numpy(input_data).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            _ = model(t_in)  # warm-up
        t0 = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(t_in)
        avg_ms = (time.time() - t0) / iterations * 1000

    print(f"  Kết quả ({iterations} lần): {avg_ms:.2f} ms / inference")
    return avg_ms


# ──────────────────────────────────────────────────────────────
# 3. Đo thời gian tiền xử lý ảnh (Pre-processing Time)
# ──────────────────────────────────────────────────────────────
def measure_preprocessing_time(data_dir, iterations=50):
    """
    Đo thời gian trung bình cho pipeline tiền xử lý một ảnh:
    Read → Grayscale → GaussianBlur → Resize 112x112
    """
    sample_img = None
    for root, _, fns in os.walk(data_dir):
        for f in fns:
            if f.lower().endswith(('.webp', '.jpg', '.pgm', '.jpeg')):
                sample_img = os.path.join(root, f)
                break
        if sample_img:
            break

    if sample_img is None:
        print("  ⚠ Không tìm thấy ảnh mẫu để đo preprocessing time.")
        return 0.0

    t0 = time.time()
    for _ in range(iterations):
        img = cv2.imread(sample_img)
        if img is None:
            continue
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _ = cv2.resize(blurred, (112, 112))
    avg_ms = (time.time() - t0) / iterations * 1000

    print(f"\n--- ĐO THỜI GIAN TIỀN XỬ LÝ (Preprocessing) ---")
    print(f"  Kết quả ({iterations} lần): {avg_ms:.2f} ms / ảnh")
    return avg_ms


# ──────────────────────────────────────────────────────────────
# 4. Model file size
# ──────────────────────────────────────────────────────────────
def get_model_info(file_path):
    """Trả về dung lượng file model (MB), 0 nếu không tồn tại."""
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"\n--- THÔNG TIN MODEL ---")
        print(f"  Kích thước: {size_mb:.2f} MB")
        return size_mb
    return 0.0


# ──────────────────────────────────────────────────────────────
# 5. Phân tích thống kê database
# ──────────────────────────────────────────────────────────────
def get_db_stats(database_dict):
    """Thống kê database từ dictionary. Trả về dict."""
    num_people = len(database_dict)
    dim = 0
    if num_people > 0:
        first_key = next(iter(database_dict))
        dim = len(database_dict[first_key])
    
    # Ước tính dung lượng file JSON nếu export (approx)
    # Json format: 1 vector 512-dim roughly 10KB
    size_kb = num_people * 10.5 

    print(f"\n--- THỐNG KÊ DATABASE ---")
    print(f"  Số lượng định danh (ID): {num_people}")
    print(f"  Số chiều vector         : {dim}")
    return {"num_ids": num_people, "dimension": dim, "size_kb": round(size_kb, 2)}


# ──────────────────────────────────────────────────────────────
# 6. Tài nguyên hệ thống (CPU/RAM)
# ──────────────────────────────────────────────────────────────
def monitor_system():
    """Đo CPU% và RAM% hiện tại qua psutil."""
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent
        print(f"\n--- TÀI NGUYÊN HỆ THỐNG ---")
        print(f"  CPU Usage: {cpu}%")
        print(f"  RAM Usage: {ram}%")
        return cpu, ram
    except ImportError:
        return None
