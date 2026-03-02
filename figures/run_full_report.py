"""
run_full_report.py — Tự động xuất toàn bộ số liệu báo cáo đồ án Vision Guard
===========================================================================
- Train data & Test data đều đánh giá bằng FaceRecognizer (edge pipeline).
- Confusion matrix xuất ra ảnh PNG.
===========================================================================
"""
import os
import sys
import csv
import numpy as np
from datetime import datetime

# ── Đường dẫn tự phát hiện ──────────────────────────────────────────────────
FIGURES_DIR    = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR     = os.path.join(FIGURES_DIR, "../face-recognizer-server")

for _p in [FIGURES_DIR, SERVER_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Cấu hình ────────────────────────────────────────────────────────────────
TRAIN_DATA_DIR = os.path.join(FIGURES_DIR, "train_data")      
TEST_DATA_DIR  = os.path.join(FIGURES_DIR, "test_data")       
MODEL_PATH      = os.path.join(SERVER_DIR, "models/mobilefacenet.tflite")
EDGE_MODEL_PATH = os.path.join(FIGURES_DIR, "../edge-device-pi4/ai-recognition/models/mobilefacenet.tflite")
EDGE_THRESHOLD  = 0.45  # Cosine distance (dùng cho cả train_data và test_data)   

# ── Bước 0: Tự động cài thư viện ────────────────────────────────────────────
import setup_deps
setup_deps.check_and_install_pc_deps()

import metrics
import accuracy_tools


def save_csv(path, rows, headers):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"  -> CSV: {path}")


def save_confusion_matrix_csv(path, conf_matrix, names, unknown_row=None):
    """Xuất confusion matrix ra CSV: hàng = Actual, cột = Predicted."""
    n = len(names)
    headers = [""] + list(names) + ["Unknown"]
    rows = []
    for i, name in enumerate(names):
        row = [name] + [int(conf_matrix[i][j]) for j in range(n)] + [int(conf_matrix[i][n])]
        rows.append(row)
    if unknown_row is not None:
        rows.append(["Unknown"] + [int(unknown_row[j]) for j in range(n)] + [int(unknown_row[n])])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"  -> CSV: {path}")


def save_confusion_matrix_csv(path, conf_matrix, names, unknown_row=None):
    """Xuất confusion matrix ra CSV: hàng = Actual, cột = Predicted."""
    n = len(names)
    headers = [""] + list(names) + ["Unknown"]
    rows = []
    for i, name in enumerate(names):
        row = [name] + [int(conf_matrix[i][j]) for j in range(n)] + [int(conf_matrix[i][n])]
        rows.append(row)
    if unknown_row is not None:
        rows.append(["Unknown"] + [int(unknown_row[j]) for j in range(n)] + [int(unknown_row[n])])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"  -> CSV: {path}")


# Số identity tối đa hiển thị trên ảnh confusion matrix (tránh rối mắt)
CONFUSION_MATRIX_IMAGE_MAX_LABELS = 12


def save_confusion_matrix_image(path, conf_matrix, names, unknown_row=None, title="Confusion Matrix"):
    """
    Xuất confusion matrix ra ảnh PNG: hàng = Actual, cột = Predicted.
    Chỉ hiển thị tối đa CONFUSION_MATRIX_IMAGE_MAX_LABELS (12) bộ data đầu tiên.
    """
    n_full = len(names)
    k = min(n_full, CONFUSION_MATRIX_IMAGE_MAX_LABELS)
    names_display = list(names[:k])
    col_indices = list(range(k)) + [n_full]  # cột 0..k-1 + cột Unknown (index n_full)
    data_slice = np.array(conf_matrix[:k, :], dtype=float)[:, col_indices] if k else np.zeros((0, len(col_indices)))
    row_labels = list(names_display)
    if unknown_row is not None:
        unknown_slice = np.array([unknown_row[j] for j in col_indices]).reshape(1, -1)
        data_slice = np.vstack([data_slice, unknown_slice])
        row_labels = row_labels + ["Unknown"]
    labels = names_display + ["Unknown"]
    if data_slice.size == 0:
        data_slice = np.zeros((1, 1))
        row_labels = [""]
        labels = [""]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        nr, nc = data_slice.shape
        fig, ax = plt.subplots(figsize=(max(6, nc * 0.5), max(5, nr * 0.4)))
        im = ax.imshow(data_slice, cmap="Blues", aspect="auto")
        ax.set_xticks(np.arange(nc))
        ax.set_yticks(np.arange(nr))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(row_labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(nr):
            for j in range(nc):
                v = int(data_slice[i, j])
                ax.text(j, i, str(v) if v > 0 else "", ha="center", va="center", color="black", fontsize=9)
        if k < n_full:
            ax.set_title(f"{title} (hien thi {k}/{n_full} identity dau tien)")
        else:
            ax.set_title(title)
        fig.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  -> Image: {path}")
    except Exception as e:
        print(f"  LOI xuat anh confusion matrix: {e}")


def dir_count(path):
    if not os.path.exists(path): return 0
    return len([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])


def run_all_metrics():
    # Tạo folder train/test nếu chưa có
    for d in [TRAIN_DATA_DIR, TEST_DATA_DIR]:
        os.makedirs(d, exist_ok=True)

    n_train = dir_count(TRAIN_DATA_DIR)
    n_test  = dir_count(TEST_DATA_DIR)

    if n_train == 0:
        print("\n[LOI] Khong co du lieu trong figures/train_data/")
        sys.exit(1)

    # Tạo folder output
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(FIGURES_DIR, f"report_{ts}")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*68)
    print("  VISION GUARD — RUN FULL REPORT")
    print(f"  Ket qua: {output_dir}")
    print("="*68)

    system_rows = []

    # ──────────────────────────────────────────────────────────
    # STEP 1 — System Metrics
    # ──────────────────────────────────────────────────────────
    print("\n[STEP 1] THU THAP THONG SO HE THONG")
    
    system_rows.append(["Train Identities",  n_train])
    system_rows.append(["Test Identities",   n_test])
    
    # Model size
    model_mb = metrics.get_model_info(MODEL_PATH)
    system_rows.append(["Model Size (MB)", round(model_mb, 3)])

    # CPU/RAM
    sys_res = metrics.monitor_system()
    if sys_res:
        cpu, ram = sys_res
        system_rows.append(["CPU Usage (%)", cpu])
        system_rows.append(["RAM Usage (%)", ram])

    # Latency
    lat_pc = metrics.measure_inference_time(MODEL_PATH, data_sample_dir=TRAIN_DATA_DIR, is_tflite=True, iterations=100)
    system_rows.append(["Inference Latency PC (ms)", round(lat_pc, 2)])

    # Pre-processing
    preproc_ms = metrics.measure_preprocessing_time(TRAIN_DATA_DIR, iterations=50)
    system_rows.append(["Preprocessing Time (ms)", round(preproc_ms, 2)])

    save_csv(os.path.join(output_dir, "system_metrics.csv"), system_rows, ["Metric", "Value"])

    # ──────────────────────────────────────────────────────────
    # STEP 2 — Face Database
    # ──────────────────────────────────────────────────────────
    print("\n[STEP 2] XAY DUNG FACE DATABASE...")
    database = accuracy_tools.build_embeddings_from_train_data(TRAIN_DATA_DIR, MODEL_PATH)
    
    db_stats = metrics.get_db_stats(database)
    save_csv(os.path.join(output_dir, "database_stats.csv"),
             [["Total Identities", db_stats["num_ids"]],
              ["Embedding Dimension", db_stats["dimension"], "512-dim"]],
             ["Metric", "Value", "Note"])

    # Ghi DB tạm để dùng cho cả Train và Test (FaceRecognizer)
    temp_db_path = os.path.join(output_dir, "temp_face_embeddings.json")
    edge_model_used = EDGE_MODEL_PATH if os.path.exists(EDGE_MODEL_PATH) else MODEL_PATH
    if not os.path.exists(EDGE_MODEL_PATH):
        print(f"  CANH BAO: Khong thay model edge, dung MODEL_PATH server.")

    # ──────────────────────────────────────────────────────────
    # STEP 3 — Accuracy trên Train data (FaceRecognizer, cùng pipeline edge)
    # ──────────────────────────────────────────────────────────
    print("\n[STEP 3] DANH GIA TREN TRAIN DATA (FaceRecognizer)")
    train_acc, train_cm, train_names, train_report, train_unknown_row = accuracy_tools.test_accuracy_edge_via_recognizer(
        TRAIN_DATA_DIR, edge_model_used, database, temp_db_path, threshold=EDGE_THRESHOLD
    )

    # Train details gộp theo identity (mỗi folder = 1 identity)
    train_grouped_rows = []
    correct_by_name = {}
    for r in train_report:
        actual = r.get("actual", "")
        if actual not in correct_by_name:
            correct_by_name[actual] = {"total": 0, "correct": 0}
        correct_by_name[actual]["total"] += 1
        if r.get("predicted") == actual:
            correct_by_name[actual]["correct"] += 1
    for name in sorted(correct_by_name.keys()):
        t = correct_by_name[name]["total"]
        c = correct_by_name[name]["correct"]
        train_grouped_rows.append([name, t, c, round(c / t * 100, 2) if t else 0])
    if train_grouped_rows:
        save_csv(os.path.join(output_dir, "train_details_grouped.csv"),
                 train_grouped_rows,
                 ["Identity", "Total Samples", "Correct", "Accuracy (%)"])

    # Confusion matrix Train -> CSV + ảnh
    if train_cm is not None and train_names:
        save_confusion_matrix_csv(
            os.path.join(output_dir, "confusion_matrix_train.csv"),
            train_cm, train_names, unknown_row=train_unknown_row
        )
        save_confusion_matrix_image(
            os.path.join(output_dir, "confusion_matrix_train.png"),
            train_cm, train_names, unknown_row=train_unknown_row, title="Confusion Matrix (Train data)"
        )

    # ──────────────────────────────────────────────────────────
    # STEP 4 — Accuracy trên Test data & Details
    # ──────────────────────────────────────────────────────────
    test_acc = 0
    if n_test > 0:
        print("\n[STEP 4] DANH GIA TREN TEST DATA (FaceRecognizer)")
        test_acc, test_cm, test_names, unknown_report, test_unknown_row = accuracy_tools.test_accuracy_edge_via_recognizer(
            TEST_DATA_DIR, edge_model_used, database, temp_db_path, threshold=EDGE_THRESHOLD
        )
        detail_rows = []
        for r in unknown_report:
            detail_rows.append([
                r.get("actual", "N/A"),
                r.get("image", "N/A"),
                r.get("predicted", "N/A"),
                round(r.get("min_distance", 0), 4),
                "TRUE" if r.get("predicted") == r.get("actual") else "FALSE"
            ])
        if detail_rows:
            save_csv(os.path.join(output_dir, "test_details.csv"),
                     detail_rows,
                     ["Actual", "Image", "Predicted", "Distance", "Is Correct"])
        if test_cm is not None and test_names:
            save_confusion_matrix_csv(
                os.path.join(output_dir, "confusion_matrix_test.csv"),
                test_cm, test_names, unknown_row=test_unknown_row
            )
            save_confusion_matrix_image(
                os.path.join(output_dir, "confusion_matrix_test.png"),
                test_cm, test_names, unknown_row=test_unknown_row, title="Confusion Matrix (Test data)"
            )

    # ──────────────────────────────────────────────────────────
    # STEP 5 — Accuracy Summary
    # ──────────────────────────────────────────────────────────
    accuracy_rows = [
        ["Train data", f"{round(train_acc, 2)}%"],
        ["Test data", f"{round(test_acc, 2)}%" if n_test > 0 else "N/A"]
    ]
    save_csv(os.path.join(output_dir, "accuracy.csv"), accuracy_rows, ["Phase", "Accuracy"])

    print("\n" + "="*68)
    print(f"  HOAN TAT! Ket qua tai: {output_dir}")
    print("="*68)


if __name__ == "__main__":
    run_all_metrics()
