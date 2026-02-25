"""
accuracy_tools.py — Tính toán độ chính xác và xây dựng database cho Vision Guard
Tự chứa trong figures/ — dữ liệu lấy từ figures/train_data và figures/test_data
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, không cần GUI
import matplotlib.pyplot as plt

# ── Import MobileFaceNet và extract_embeddings từ face-recognizer-server ──────
_FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
_SERVER_DIR  = os.path.join(_FIGURES_DIR, "../face-recognizer-server")

for _p in [_FIGURES_DIR, _SERVER_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ──────────────────────────────────────────────────────────────
# 1. Khoảng cách Inter-class
# ──────────────────────────────────────────────────────────────

def calculate_inter_class_distances(db_path_or_dict):
    """
    Tính khoảng cách Euclidean giữa tất cả các cặp identity khác nhau.
    Tham số: đường dẫn JSON hoặc dict {name: [vector]}.
    Trả về: list[float] khoảng cách.
    """
    if isinstance(db_path_or_dict, str):
        if not os.path.exists(db_path_or_dict):
            print("⚠ Không tìm thấy database."); return []
        with open(db_path_or_dict, 'r', encoding='utf-8') as f:
            db = json.load(f)['data']
    else:
        db = db_path_or_dict

    names      = list(db.keys())
    embeddings = [np.array(db[n]) for n in names]
    distances  = [np.linalg.norm(embeddings[i] - embeddings[j])
                  for i in range(len(embeddings))
                  for j in range(i + 1, len(embeddings))]

    print(f"\n--- PHÂN TÍCH KHOẢNG CÁCH LIÊN LỚP (INTER-CLASS) ---")
    print(f"  Số cặp so sánh   : {len(distances)}")
    if distances:
        print(f"  Nhỏ nhất         : {min(distances):.4f}")
        print(f"  Lớn nhất         : {max(distances):.4f}")
        print(f"  Trung bình       : {np.mean(distances):.4f}")
    return distances


def plot_distance_histogram(inter_distances, intra_distances=None,
                             output_path="distance_dist.png"):
    """Vẽ biểu đồ phân bố khoảng cách Euclidean."""
    plt.figure(figsize=(10, 6))
    plt.hist(inter_distances, bins=50, alpha=0.6,
             label='Inter-class (Different people)', color='tomato')
    if intra_distances:
        plt.hist(intra_distances, bins=50, alpha=0.6,
                 label='Intra-class (Same person)', color='mediumseagreen')
    plt.axvline(x=1.0, color='steelblue', linestyle='--', linewidth=2,
                label='Threshold = 1.0')
    plt.title("Distribution of Euclidean Distances between Embeddings")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  → Biểu đồ khoảng cách: {output_path}")


# ──────────────────────────────────────────────────────────────
# 2. Build Database từ train_data
# ──────────────────────────────────────────────────────────────

def build_embeddings_from_train_data(train_data_dir, model_path, output_json_path):
    """
    Duyệt figures/train_data/<person>/, extract embedding mỗi ảnh bằng MobileFaceNet,
    tính centroid cho từng người, lưu ra output_json_path.
    Trả về dict {name: centroid_list}.
    """
    from extract_embeddings import get_face_embedding
    from mobilefacenet import MobileFaceNet
    import torch

    print(f"\n--- BUILD DATABASE TỪ TRAIN DATA ---")
    print(f"  Nguồn: {train_data_dir}")
    if not os.path.exists(train_data_dir):
        print(f"  ✗ Không tìm thấy thư mục {train_data_dir}"); return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = MobileFaceNet(embedding_size=512).to(device)
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        elif isinstance(ckpt, torch.nn.Module):
            model = ckpt
        else:
            print(f"  ⚠ Checkpoint không rõ kiểu ({type(ckpt)}) — dùng random weights.")
    else:
        print(f"  ⚠ Không tìm thấy model tại {model_path} — dùng random weights.")
    model.eval()

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
            emb = get_face_embedding(model, os.path.join(person_dir, img_file), device)
            if emb is not None:
                vecs.append(emb)
        if vecs:
            # Tính centroid từ raw embeddings, KHÔNG normalize từng embedding riêng
            # (đồng nhất với extract_embeddings.py trên server)
            centroid = np.mean(vecs, axis=0)
            norm = np.linalg.norm(centroid)
            centroid = centroid / norm if norm > 0 else centroid
            database[person] = centroid.tolist()
            print(f"  [{idx+1:>2}/{len(identities)}] {person}: {len(vecs)} ảnh OK")
        else:
            print(f"  [{idx+1:>2}/{len(identities)}] {person}: ⚠ không có ảnh hợp lệ")

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    out = {
        "metadata": {
            "model": "MobileFaceNet",
            "dimension": 512,
            "source": str(train_data_dir),
            "description": "Generated from figures/train_data by run_full_report.py"
        },
        "data": database
    }
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Đã lưu DB ({len(database)} identities) → {output_json_path}")
    return database


# ──────────────────────────────────────────────────────────────
# 3. PC Phase — test trên train_data
# ──────────────────────────────────────────────────────────────

def _load_pytorch_model(model_path, verbose=False):
    """Load MobileFaceNet qua model_loader an toàn, trả về (model, device)."""
    from model_loader import load_mobilefacenet
    model, device, ok = load_mobilefacenet(model_path, verbose=verbose)
    if not ok:
        print(f"  ⚠ Model '{os.path.basename(model_path)}' không load được — "
              f"dùng random weights (embedding không có ý nghĩa thực tế)")
    return model, device


def _match(embedding, database, threshold):
    """Trả về (best_match_name, min_dist, final_label)."""
    import numpy as np
    min_dist, best = float('inf'), "Unknown"
    for db_name, db_vec in database.items():
        d = np.linalg.norm(np.array(embedding) - np.array(db_vec))
        if d < min_dist:
            min_dist = d; best = db_name
    final = best if min_dist < threshold else "Unknown"
    return best, min_dist, final


def test_accuracy_pc(database_dict_or_path, data_dir, model_path, threshold=1.0):
    """
    Test độ chính xác trên PC Phase với dữ liệu trong data_dir (train_data).
    Trả về (accuracy, conf_matrix, names).
    conf_matrix shape: (N, N+1), cột cuối = Unknown.
    """
    from extract_embeddings import get_face_embedding

    print(f"\n--- ĐÁNH GIÁ ĐỘ CHÍNH XÁC (PC PHASE) ---")

    if isinstance(database_dict_or_path, dict):
        database = database_dict_or_path
    else:
        with open(database_dict_or_path, 'r', encoding='utf-8') as f:
            database = json.load(f)['data']

    if not os.path.exists(data_dir):
        print(f"  ✗ Không tìm thấy: {data_dir}"); return 0, None, []

    model, device = _load_pytorch_model(model_path)
    names        = sorted(database.keys())
    n            = len(names)
    name_to_idx  = {nm: i for i, nm in enumerate(names)}
    conf_matrix  = np.zeros((n, n + 1))  # +1 cho cột Unknown

    correct = total = 0
    print(f"  DB: {n} identities | Data: {data_dir}")

    for person in sorted(os.listdir(data_dir)):
        pdir = os.path.join(data_dir, person)
        if not os.path.isdir(pdir) or person not in name_to_idx:
            continue
        ai = name_to_idx[person]
        for img_f in os.listdir(pdir):
            if not img_f.lower().endswith(('.webp', '.jpg', '.jpeg', '.png', '.pgm')):
                continue
            emb = get_face_embedding(model, os.path.join(pdir, img_f), device)
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
# 4. Edge Phase — test trên test_data (có Unknown faces)
# ──────────────────────────────────────────────────────────────

def test_accuracy_edge(database_dict_or_path, test_data_dir, model_path, threshold=1.0):
    """
    Test Edge Phase với dữ liệu test_data (dùng PyTorch thay cho TFLite trên Windows).
    Identity không có trong DB (s33-s35) → tính False Positive Rate.
    Trả về (accuracy, conf_matrix, names, unknown_report).
    """
    from extract_embeddings import get_face_embedding

    print(f"\n--- ĐÁNH GIÁ ĐỘ CHÍNH XÁC (EDGE PHASE — PyTorch sim) ---")

    if isinstance(database_dict_or_path, dict):
        database = database_dict_or_path
    else:
        with open(database_dict_or_path, 'r', encoding='utf-8') as f:
            database = json.load(f)['data']

    if not os.path.exists(test_data_dir):
        print(f"  ✗ Không tìm thấy: {test_data_dir}"); return 0, None, [], []

    model, device = _load_pytorch_model(model_path)
    names         = sorted(database.keys())
    n             = len(names)
    name_to_idx   = {nm: i for i, nm in enumerate(names)}
    conf_matrix   = np.zeros((n, n + 1))

    persons       = sorted(os.listdir(test_data_dir))
    known_ps      = [p for p in persons if os.path.isdir(os.path.join(test_data_dir, p)) and p in name_to_idx]
    unknown_ps    = [p for p in persons if os.path.isdir(os.path.join(test_data_dir, p)) and p not in name_to_idx]

    print(f"  Known   : {len(known_ps)} identities có trong DB")
    print(f"  Unknown : {len(unknown_ps)} identities KHÔNG trong DB → {unknown_ps}")

    correct = total_known = false_positives = total_unknown = 0
    unknown_report = []

    # Test known
    for person in known_ps:
        ai   = name_to_idx[person]
        pdir = os.path.join(test_data_dir, person)
        for img_f in os.listdir(pdir):
            if not img_f.lower().endswith(('.webp', '.jpg', '.jpeg', '.png', '.pgm')):
                continue
            emb = get_face_embedding(model, os.path.join(pdir, img_f), device)
            if emb is None:
                continue
            total_known += 1
            _, dist, label = _match(emb, database, threshold)
            if label == person:
                correct += 1
            if label in name_to_idx:
                conf_matrix[ai][name_to_idx[label]] += 1
            else:
                conf_matrix[ai][n] += 1

    # Test unknown (FAR: False Acceptance Rate)
    for person in unknown_ps:
        pdir = os.path.join(test_data_dir, person)
        for img_f in os.listdir(pdir):
            if not img_f.lower().endswith(('.webp', '.jpg', '.jpeg', '.png', '.pgm')):
                continue
            emb = get_face_embedding(model, os.path.join(pdir, img_f), device)
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


# ──────────────────────────────────────────────────────────────
# 5. Vẽ và báo cáo Confusion Matrix
# ──────────────────────────────────────────────────────────────

def report_accuracy(accuracy, conf_matrix, class_names, output_name="accuracy_report"):
    """In báo cáo chi tiết và lưu Confusion Matrix PNG."""
    tag = os.path.basename(output_name).upper()
    print(f"\n{'='*55}")
    print(f"BÁO CÁO ĐỘ CHÍNH XÁC: {tag}")
    print(f"{'='*55}")
    print(f"Tổng Accuracy: {accuracy:.2f}%\n")
    print(f"{'Class':<18} {'Correct':>8} {'Total':>8} {'Acc (%)':>9}")
    print("-" * 47)
    for i, name in enumerate(class_names[:15]):
        rs  = np.sum(conf_matrix[i])
        acc = conf_matrix[i][i] / rs * 100 if rs > 0 else 0
        print(f"{name:<18} {int(conf_matrix[i][i]):>8} {int(rs):>8} {acc:>8.1f}%")
    if len(class_names) > 15:
        print(f"  ... ({len(class_names)-15} classes nữa — xem CSV)")

    # Vẽ Confusion Matrix
    try:
        cm = conf_matrix[:, :len(class_names)]  # bỏ cột Unknown để dễ nhìn
        rs = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm, rs, where=rs != 0) * 100

        figsize = max(8, len(class_names) * 0.4)
        plt.figure(figsize=(figsize, figsize * 0.85))
        im = plt.imshow(cm_pct, interpolation='nearest',
                        cmap=plt.cm.Blues, vmin=0, vmax=100)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f"Confusion Matrix — {tag} (%)", fontsize=12)
        ticks = np.arange(len(class_names))
        step  = max(1, len(class_names) // 20)
        plt.xticks(ticks[::step], class_names[::step],
                   rotation=45, ha='right', fontsize=7)
        plt.yticks(ticks[::step], class_names[::step], fontsize=7)
        plt.ylabel('Actual', fontsize=10)
        plt.xlabel('Predicted', fontsize=10)
        plt.tight_layout()
        img_path = f"{output_name}_cm.png"
        plt.savefig(img_path, dpi=150)
        plt.close()
        print(f"  → Confusion Matrix: {img_path}")
    except Exception as e:
        print(f"  ✗ Lỗi vẽ biểu đồ: {e}")
