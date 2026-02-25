"""
debug_detection.py — Kiểm tra nhanh khả năng phát hiện mặt trong figures/train_data
Chạy từ: figures/
"""
import os, sys, cv2

# Đảm bảo import được các module local
FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR  = os.path.join(FIGURES_DIR, "../face-recognizer-server")
for _p in [FIGURES_DIR, SERVER_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from mobilefacenet import MobileFaceNet
from extract_embeddings import get_face_embedding

# ── Cấu hình ────────────────────────────────────────────────
TRAIN_DATA_DIR = os.path.join(FIGURES_DIR, "train_data")
MODEL_PATH     = os.path.join(SERVER_DIR,  "models/mobilefacenet_pretrained.pth")
MAX_PER_PERSON = 3   # Số ảnh tối đa để kiểm tra mỗi người


def debug_detection():
    print(f"Train data : {TRAIN_DATA_DIR}")
    print(f"Model path : {MODEL_PATH}\n")

    if not os.path.exists(TRAIN_DATA_DIR) or not os.listdir(TRAIN_DATA_DIR):
        print("[LỖI] figures/train_data/ không có dữ liệu.")
        print("  → Copy ảnh vào: figures/train_data/<tên_người>/<ảnh.*>")
        return

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MobileFaceNet(embedding_size=512).to(device)
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=device)
        if isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        elif isinstance(ckpt, torch.nn.Module):
            model = ckpt
    model.eval()

    persons = sorted([d for d in os.listdir(TRAIN_DATA_DIR)
                      if os.path.isdir(os.path.join(TRAIN_DATA_DIR, d))])
    print(f"Tổng identities: {len(persons)}\n")
    print(f"{'Person':<15} {'Detected':>9} {'Checked':>8} {'Rate':>8}")
    print("-" * 45)

    total_ok = total_checked = 0
    for person in persons:
        pdir   = os.path.join(TRAIN_DATA_DIR, person)
        imgs   = [f for f in os.listdir(pdir)
                  if f.lower().endswith(('.webp', '.jpg', '.jpeg', '.pgm', '.png'))]
        sample = imgs[:MAX_PER_PERSON]
        ok = 0
        for img_f in sample:
            emb = get_face_embedding(model, os.path.join(pdir, img_f), device)
            if emb is not None:
                ok += 1
        total_ok      += ok
        total_checked += len(sample)
        rate = ok / len(sample) * 100 if sample else 0
        status = "✓" if ok > 0 else "✗"
        print(f"{status} {person:<14} {ok:>9}/{len(sample):<8} {rate:>6.0f}%")

    print("-" * 45)
    overall = total_ok / total_checked * 100 if total_checked else 0
    print(f"{'TỔNG':<15} {total_ok:>9}/{total_checked:<8} {overall:>6.1f}%")

    if overall < 50:
        print("\n⚠ Tỷ lệ phát hiện thấp — Kiểm tra lại:")
        print("  1. Ảnh có chứa khuôn mặt rõ ràng không?")
        print("  2. File haarcascade_frontalface_default.xml có không?")
        print("  3. Thử tăng scaleFactor trong detectMultiScale (hiện là 1.1).")


if __name__ == "__main__":
    debug_detection()
