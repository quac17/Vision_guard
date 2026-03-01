"""
accuracy_tools.py — Tính toán độ chính xác và xây dựng database cho Vision Guard.
- PC Phase: TFLite + extract_embeddings (server), Euclidean match.
- Edge Phase: Dùng pipeline thật tại edge (FaceRecognizer, cosine distance, threshold 0.45).
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

# ── Paths ────────────────────────────────────────────────────────────────────
_FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
_SERVER_DIR  = os.path.join(_FIGURES_DIR, "../face-recognizer-server")
_EDGE_RECOGNIZER_DIR = os.path.join(_FIGURES_DIR, "../edge-device-pi4/ai-recognition")

for _p in [_FIGURES_DIR, _SERVER_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ──────────────────────────────────────────────────────────────
# 1. Load Model helper
# ──────────────────────────────────────────────────────────────

def _load_tflite_model(model_path):
    """Khởi tạo TFLite Interpreter."""
    if tflite is None:
        print("LOI: Chua cai dat tflite-runtime hoac tensorflow.")
        return None
    
    if not os.path.exists(model_path):
        print(f"LOI: Khong tim thay model tai {model_path}")
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

    print(f"\n--- BUILD DATABASE TU TRAIN DATA (TFLite) ---")
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
            print(f"  [{idx+1:>2}/{len(identities)}] {person}: {len(vecs)} anh OK")
        else:
            print(f"  [{idx+1:>2}/{len(identities)}] {person}: khong co anh hop le")

    print(f"  Da xay dung Database ({len(database)} identities) trong bo nho.")
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

    print(f"\n--- DANH GIA DO CHINH XAC (PC PHASE - TFLite local) ---")

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

    print(f"\n--- DANH GIA DO CHINH XAC (EDGE PHASE - TFLite local) ---")

    interpreter = _load_tflite_model(model_path)
    if interpreter is None: return 0, None, [], [], None

    names         = sorted(database.keys())
    n             = len(names)
    name_to_idx   = {nm: i for i, nm in enumerate(names)}
    conf_matrix   = np.zeros((n, n + 1))

    persons       = sorted(os.listdir(test_data_dir))
    known_ps      = [p for p in persons if os.path.isdir(os.path.join(test_data_dir, p)) and p in name_to_idx]
    unknown_ps    = [p for p in persons if os.path.isdir(os.path.join(test_data_dir, p)) and p not in name_to_idx]

    correct = total_known = false_positives = total_unknown = 0
    unknown_report = []
    # Hàng confusion cho "Actual = Unknown" (người ngoài DB)
    unknown_row = np.zeros(n + 1)

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
            if label in name_to_idx:
                unknown_row[name_to_idx[label]] += 1
            else:
                unknown_row[n] += 1
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
    print(f"  FAR (unknown nhan nham): {far:.2f}%  ({false_positives}/{total_unknown})")
    print(f"  FRR (nguoi quen bi tu choi): {frr:.2f}%")
    return acc, conf_matrix, names, unknown_report, unknown_row


# ──────────────────────────────────────────────────────────────
# 6. Edge Phase — dùng pipeline thật (FaceRecognizer, cosine, đánh giá theo folder)
# ──────────────────────────────────────────────────────────────

def test_accuracy_edge_via_recognizer(test_data_dir, model_path, database, temp_db_path, threshold=0.45):
    """
    Đánh giá Edge Phase bằng pipeline giống run_test.py:
    FaceRecognizer (DNN+Haar detection, 160x160, cosine distance, majority vote),
    một kết quả cho mỗi folder (batch), threshold cosine mặc định 0.45.
    database: dict name -> list (centroid); temp_db_path: đường dẫn ghi file JSON tạm.
    Trả về (acc, conf_matrix, names, unknown_report, unknown_row) cùng format với test_accuracy_edge.
    """
    if not os.path.isdir(test_data_dir):
        print(f"  LOI: Khong tim thay thu muc test {test_data_dir}")
        return 0, None, [], [], None

    # Ghi database ra file JSON (format giống face_embeddings.json)
    db_content = {
        "metadata": {"model": "MobileFaceNet", "dimension": 512, "description": "Temp for report"},
        "data": database
    }
    with open(temp_db_path, "w", encoding="utf-8") as f:
        json.dump(db_content, f, ensure_ascii=False)

    # Import FaceRecognizer từ edge-device-pi4/ai-recognition
    if _EDGE_RECOGNIZER_DIR not in sys.path:
        sys.path.insert(0, _EDGE_RECOGNIZER_DIR)
    try:
        from recognizer import FaceRecognizer
    except Exception as e:
        print(f"  LOI: Khong import duoc FaceRecognizer tu edge: {e}")
        return 0, None, [], [], None

    try:
        recognizer = FaceRecognizer(model_path, temp_db_path, threshold=threshold)
        recognizer.submit_attendance = lambda code: None  # Tắt API khi chạy report
    except Exception as e:
        print(f"  LOI: Khoi tao FaceRecognizer: {e}")
        return 0, None, [], [], None

    names = sorted(database.keys())
    n = len(names)
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    conf_matrix = np.zeros((n, n + 1))
    unknown_row = np.zeros(n + 1)
    unknown_report = []

    correct = total_known = 0
    persons = sorted([d for d in os.listdir(test_data_dir)
                      if os.path.isdir(os.path.join(test_data_dir, d))])
    known_folders = [p for p in persons if p in name_to_idx]
    unknown_folders = [p for p in persons if p not in name_to_idx]

    for folder_name in persons:
        subject_dir = os.path.join(test_data_dir, folder_name)
        identified_name, distance = recognizer.recognize_batch(subject_dir)
        # Chuẩn hóa predicted: nếu chuỗi chứa "Unknown" thì coi là Unknown
        if "Unknown" in str(identified_name):
            predicted = "Unknown"
        else:
            predicted = identified_name

        pred_idx = name_to_idx.get(predicted, n)
        is_correct = (predicted == folder_name)

        if folder_name in name_to_idx:
            total_known += 1
            if is_correct:
                correct += 1
            ai = name_to_idx[folder_name]
            conf_matrix[ai][pred_idx] += 1
        else:
            unknown_row[pred_idx] += 1

        unknown_report.append({
            "actual": folder_name,
            "image": "(batch)",
            "predicted": predicted,
            "min_distance": round(float(distance), 4),
            "false_positive": (folder_name not in name_to_idx and predicted != "Unknown")
        })

    acc = (correct / total_known * 100) if total_known else 0
    print(f"\n--- DANH GIA DO CHINH XAC (EDGE PHASE - qua FaceRecognizer, cosine threshold={threshold}) ---")
    print(f"  Known Accuracy (theo folder): {acc:.2f}%  ({correct}/{total_known} folder)")
    print(f"  So folder test: {len(persons)} (known: {total_known}, unknown: {len(unknown_folders)})")
    return acc, conf_matrix, names, unknown_report, unknown_row
