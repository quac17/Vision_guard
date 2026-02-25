"""
accuracy_tools.py — Tính toán độ chính xác và xây dựng database cho Vision Guard
Sử dụng TFLite model để đồng bộ với Edge Phase.
"""
import os
import sys
import json
import numpy as np

# Thử import tflite_runtime trước (nhẹ hơn), fallback sang tensorflow.lite
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None

# ── Import MobileFaceNet và extract_embeddings từ face-recognizer-server ──────
_FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
_SERVER_DIR  = os.path.join(_FIGURES_DIR, "../face-recognizer-server")

for _p in [_FIGURES_DIR, _SERVER_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ──────────────────────────────────────────────────────────────
# 1. Load Model helper
# ──────────────────────────────────────────────────────────────

def _load_tflite_model(model_path):
    """Khởi tạo TFLite Interpreter."""
    if tflite is None:
        print("LỖI: Chưa cài đặt tflite-runtime hoặc tensorflow.")
        return None
    
    if not os.path.exists(model_path):
        print(f"LỖI: Không tìm thấy model tại {model_path}")
        return None
        
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


# ──────────────────────────────────────────────────────────────
# 2. Build Database từ train_data
# ──────────────────────────────────────────────────────────────

def build_embeddings_from_train_data(train_data_dir, model_path):
    """
    Duyệt figures/train_data/<person>/, extract embedding mỗi ảnh bằng TFLite,
    tính centroid cho từng người. Trả về dict {name: centroid_list}.
    """
    from extract_embeddings import get_face_embedding

    print(f"\n--- BUILD DATABASE TỪ TRAIN DATA (TFLite) ---")
    print(f"  Model: {model_path}")
    
    interpreter = _load_tflite_model(model_path)
    if interpreter is None: return {}

    identities = sorted([d for d in os.listdir(train_data_dir)
                         if os.path.isdir(os.path.join(train_data_dir, d))])
    print(f"  Identities: {len(identities)}")

    database = {}
    for idx, person in enumerate(identities):
        person_dir = os.path.join(train_data_dir, person)
        vecs = []
        for img_file in os.listdir(person_dir):
            if not img_file.lower().endswith(('.webp', '.jpg', '.jpeg', '.png', '.pgm')):
                continue
            emb = get_face_embedding(interpreter, os.path.join(person_dir, img_file))
            if emb is not None:
                vecs.append(emb)
        if vecs:
            centroid = np.mean(vecs, axis=0)
            norm = np.linalg.norm(centroid)
            centroid = centroid / norm if norm > 0 else centroid
            database[person] = centroid.tolist()
            print(f"  [{idx+1:>2}/{len(identities)}] {person}: {len(vecs)} ảnh OK")
        else:
            print(f"  [{idx+1:>2}/{len(identities)}] {person}: ⚠ không có ảnh hợp lệ")

    print(f"  ✓ Đã xây dựng Database ({len(database)} identities) trong bộ nhớ.")
    return database


# ──────────────────────────────────────────────────────────────
# 3. Match logic
# ──────────────────────────────────────────────────────────────

def _match(embedding, database, threshold):
    """Trả về (best_match_name, min_dist, final_label)."""
    min_dist, best = float('inf'), "Unknown"
    for db_name, db_vec in database.items():
        d = np.linalg.norm(np.array(embedding) - np.array(db_vec))
        if d < min_dist:
            min_dist = d; best = db_name
    final = best if min_dist < threshold else "Unknown"
    return best, min_dist, final


# ──────────────────────────────────────────────────────────────
# 4. PC Phase — test trên train_data
# ──────────────────────────────────────────────────────────────

def test_accuracy_pc(database, data_dir, model_path, threshold=1.0):
    """Test độ chính xác trên PC Phase bằng TFLite."""
    from extract_embeddings import get_face_embedding

    print(f"\n--- ĐÁNH GIÁ ĐỘ CHÍNH XÁC (PC PHASE — TFLite local) ---")

    interpreter = _load_tflite_model(model_path)
    if interpreter is None: return 0, None, []

    names        = sorted(database.keys())
    n            = len(names)
    name_to_idx  = {nm: i for i, nm in enumerate(names)}
    conf_matrix  = np.zeros((n, n + 1))

    correct = total = 0
    for person in sorted(os.listdir(data_dir)):
        pdir = os.path.join(data_dir, person)
        if not os.path.isdir(pdir) or person not in name_to_idx:
            continue
        ai = name_to_idx[person]
        for img_f in os.listdir(pdir):
            if not img_f.lower().endswith(('.webp', '.jpg', '.jpeg', '.png', '.pgm')):
                continue
            emb = get_face_embedding(interpreter, os.path.join(pdir, img_f))
            if emb is None:
                continue
            total += 1
            _, dist, label = _match(emb, database, threshold)
            if label == person:
                correct += 1
            if label in name_to_idx:
                conf_matrix[ai][name_to_idx[label]] += 1
            else:
                conf_matrix[ai][n] += 1

    acc = correct / total * 100 if total else 0
    print(f"  PC Accuracy: {acc:.2f}% ({correct}/{total})")
    return acc, conf_matrix, names


# ──────────────────────────────────────────────────────────────
# 5. Edge Phase — test trên test_data (có Unknown faces)
# ──────────────────────────────────────────────────────────────

def test_accuracy_edge(database, test_data_dir, model_path, threshold=1.0):
    """Test Edge Phase bằng TFLite."""
    from extract_embeddings import get_face_embedding

    print(f"\n--- ĐÁNH GIÁ ĐỘ CHÍNH XÁC (EDGE PHASE — TFLite local) ---")

    interpreter = _load_tflite_model(model_path)
    if interpreter is None: return 0, None, [], []

    names         = sorted(database.keys())
    n             = len(names)
    name_to_idx   = {nm: i for i, nm in enumerate(names)}
    conf_matrix   = np.zeros((n, n + 1))

    persons       = sorted(os.listdir(test_data_dir))
    known_ps      = [p for p in persons if os.path.isdir(os.path.join(test_data_dir, p)) and p in name_to_idx]
    unknown_ps    = [p for p in persons if os.path.isdir(os.path.join(test_data_dir, p)) and p not in name_to_idx]

    correct = total_known = false_positives = total_unknown = 0
    unknown_report = []

    # Test known
    for person in known_ps:
        ai   = name_to_idx[person]
        pdir = os.path.join(test_data_dir, person)
        for img_f in os.listdir(pdir):
            if not img_f.lower().endswith(('.webp', '.jpg', '.jpeg', '.png', '.pgm')):
                continue
            emb = get_face_embedding(interpreter, os.path.join(pdir, img_f))
            if emb is None:
                continue
            total_known += 1
            best, dist, label = _match(emb, database, threshold)
            if label == person:
                correct += 1
            if label in name_to_idx:
                conf_matrix[ai][name_to_idx[label]] += 1
            else:
                conf_matrix[ai][n] += 1
            
            unknown_report.append({
                "actual": person,
                "image": img_f,
                "predicted": label,
                "min_distance": round(float(dist), 4),
                "false_positive": (label != "Unknown" and person not in name_to_idx)
            })

    # Test unknown
    for person in unknown_ps:
        pdir = os.path.join(test_data_dir, person)
        for img_f in os.listdir(pdir):
            if not img_f.lower().endswith(('.webp', '.jpg', '.jpeg', '.png', '.pgm')):
                continue
            emb = get_face_embedding(interpreter, os.path.join(pdir, img_f))
            if emb is None:
                continue
            total_unknown += 1
            best, dist, label = _match(emb, database, threshold)
            is_fp = (label != "Unknown")
            if is_fp:
                false_positives += 1
            unknown_report.append({
                "actual": person,
                "image": img_f,
                "predicted": label,
                "min_distance": round(float(dist), 4),
                "false_positive": is_fp
            })

    acc = correct / total_known * 100 if total_known else 0
    far = false_positives / total_unknown * 100 if total_unknown else 0
    frr = (1 - correct / total_known) * 100 if total_known else 0
    print(f"  Known Accuracy (TPR) : {acc:.2f}%  ({correct}/{total_known})")
    print(f"  FAR (unknown nhận nhầm): {far:.2f}%  ({false_positives}/{total_unknown})")
    print(f"  FRR (người quen bị từ chối): {frr:.2f}%")
    return acc, conf_matrix, names, unknown_report
