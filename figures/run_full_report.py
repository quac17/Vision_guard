"""
run_full_report.py — Tự động xuất số liệu báo cáo Vision Guard (multi-dataset)
===============================================================================
- Nguồn data: figures/data/<dataset_X>/train_data và test_data (X = 1..5).
- Chạy từng dataset trong DATASETS_TO_RUN; comment dòng không cần chạy.
- Kết quả: figures/result/<dataset_X>/report_YYYYMMDD_HHMMSS/ (CSV + PNG).
- Train & Test đều đánh giá bằng FaceRecognizer (edge pipeline).
===============================================================================
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
DATA_ROOT       = os.path.join(FIGURES_DIR, "data")           # data/dataset_1/train_data, test_data; ...
RESULT_ROOT     = os.path.join(FIGURES_DIR, "result")         # result/dataset_1/report_YYYYMMDD_HHMMSS/
MODEL_PATH      = os.path.join(SERVER_DIR, "models/mobilefacenet.tflite")
EDGE_MODEL_PATH = os.path.join(FIGURES_DIR, "../edge-device-pi4/ai-recognition/models/mobilefacenet.tflite")
EDGE_THRESHOLD  = 0.45  # Cosine distance (dùng cho cả train_data và test_data)

# Chọn dataset để chạy report (comment dòng không cần chạy)
DATASETS_TO_RUN = [
    "dataset_0",
    # "dataset_1",
    # "dataset_2",
    # "dataset_3",
    # "dataset_4",
    # "dataset_5",
]   

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


def _compute_binary_confusion(report_list, known_set):
    """
    Từ report (list dict actual, predicted) và set known (identity trong DB),
    trả về TP, TN, FP, FN. Known = người đã biết, Stranger = người lạ (Unknown).
    """
    tp = tn = fp = fn = 0
    for r in report_list:
        actual = r.get("actual", "")
        pred = r.get("predicted", "Unknown")
        actual_known = actual in known_set
        pred_known = pred != "Unknown" and "Unknown" not in str(pred)
        if actual_known and pred_known:
            tp += 1
        elif actual_known and not pred_known:
            fn += 1
        elif not actual_known and pred_known:
            fp += 1
        else:
            tn += 1
    return tp, tn, fp, fn


def save_confusion_matrix_2x2_image(path, tp, tn, fp, fn, title="Confusion Matrix (Known / Stranger)"):
    """
    Xuất ảnh ma trận nhầm lẫn 2x2: Known vs Stranger.
    Hàng = Actual, Cột = Predicted. TP & TN cùng màu (đúng), FP và FN mỗi loại một màu khác.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        # 0 = TP/TN (đúng) -> xanh lá; 1 = FP -> cam; 2 = FN -> đỏ
        cmap = ListedColormap(["#27ae60", "#e67e22", "#c0392b"])  # green, orange, red
        # Ma trận: [Actual Known: TP, FN; Actual Stranger: FP, TN]. Giá trị 0/1/2 cho màu.
        color_matrix = np.array([[0, 2], [1, 0]])  # (0,0)=TP->0, (0,1)=FN->2, (1,0)=FP->1, (1,1)=TN->0
        value_matrix = np.array([[tp, fn], [fp, tn]])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(color_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=2)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted\nKnown", "Predicted\nStranger"])
        ax.set_yticklabels(["Actual Known", "Actual Stranger"])
        for i in range(2):
            for j in range(2):
                v = int(value_matrix[i, j])
                ax.text(j, i, f"{v}\n({'TP' if (i,j)==(0,0) else 'FN' if (i,j)==(0,1) else 'FP' if (i,j)==(1,0) else 'TN'})",
                        ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax.set_title(title)
        fig.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  -> Image: {path}")
    except Exception as e:
        print(f"  LOI xuat anh confusion 2x2: {e}")


def dir_count(path):
    if not os.path.exists(path): return 0
    return len([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])


def run_report_for_paths(train_data_dir, test_data_dir, output_dir):
    """Chạy đủ 5 bước report với một cặp train_data_dir, test_data_dir; ghi ra output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    n_train = dir_count(train_data_dir)
    n_test  = dir_count(test_data_dir)

    if n_train == 0:
        print(f"\n[LOI] Khong co du lieu trong {train_data_dir}")
        return

    print(f"  Ket qua: {output_dir}")

    system_rows = []
    print("\n[STEP 1] THU THAP THONG SO HE THONG")
    system_rows.append(["Train Identities",  n_train])
    system_rows.append(["Test Identities",   n_test])
    model_mb = metrics.get_model_info(MODEL_PATH)
    system_rows.append(["Model Size (MB)", round(model_mb, 3)])
    sys_res = metrics.monitor_system()
    if sys_res:
        cpu, ram = sys_res
        system_rows.append(["CPU Usage (%)", cpu])
        system_rows.append(["RAM Usage (%)", ram])
    lat_pc = metrics.measure_inference_time(MODEL_PATH, data_sample_dir=train_data_dir, is_tflite=True, iterations=100)
    system_rows.append(["Inference Latency PC (ms)", round(lat_pc, 2)])
    preproc_ms = metrics.measure_preprocessing_time(train_data_dir, iterations=50)
    system_rows.append(["Preprocessing Time (ms)", round(preproc_ms, 2)])
    save_csv(os.path.join(output_dir, "system_metrics.csv"), system_rows, ["Metric", "Value"])

    print("\n[STEP 2] XAY DUNG FACE DATABASE...")
    database = accuracy_tools.build_embeddings_from_train_data(train_data_dir, MODEL_PATH)
    db_stats = metrics.get_db_stats(database)
    save_csv(os.path.join(output_dir, "database_stats.csv"),
             [["Total Identities", db_stats["num_ids"]],
              ["Embedding Dimension", db_stats["dimension"], "512-dim"]],
             ["Metric", "Value", "Note"])

    temp_db_path = os.path.join(output_dir, "temp_face_embeddings.json")
    edge_model_used = EDGE_MODEL_PATH if os.path.exists(EDGE_MODEL_PATH) else MODEL_PATH
    if not os.path.exists(EDGE_MODEL_PATH):
        print(f"  CANH BAO: Khong thay model edge, dung MODEL_PATH server.")

    print("\n[STEP 3] DANH GIA TREN TRAIN DATA (FaceRecognizer)")
    train_acc, train_cm, train_names, train_report, train_unknown_row = accuracy_tools.test_accuracy_edge_via_recognizer(
        train_data_dir, edge_model_used, database, temp_db_path, threshold=EDGE_THRESHOLD
    )
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
    known_set = set(database.keys())
    tp_tr, tn_tr, fp_tr, fn_tr = _compute_binary_confusion(train_report, known_set)
    save_confusion_matrix_2x2_image(
        os.path.join(output_dir, "confusion_matrix_2x2_train.png"),
        tp_tr, tn_tr, fp_tr, fn_tr, title="Confusion Matrix 2x2 — Train (Known / Stranger)"
    )

    test_acc = 0
    if n_test > 0:
        print("\n[STEP 4] DANH GIA TREN TEST DATA (FaceRecognizer)")
        test_acc, test_cm, test_names, unknown_report, test_unknown_row = accuracy_tools.test_accuracy_edge_via_recognizer(
            test_data_dir, edge_model_used, database, temp_db_path, threshold=EDGE_THRESHOLD
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
        tp_te, tn_te, fp_te, fn_te = _compute_binary_confusion(unknown_report, known_set)
        save_confusion_matrix_2x2_image(
            os.path.join(output_dir, "confusion_matrix_2x2_test.png"),
            tp_te, tn_te, fp_te, fn_te, title="Confusion Matrix 2x2 — Test (Known / Stranger)"
        )

    accuracy_rows = [
        ["Train data", f"{round(train_acc, 2)}%"],
        ["Test data", f"{round(test_acc, 2)}%" if n_test > 0 else "N/A"]
    ]
    save_csv(os.path.join(output_dir, "accuracy.csv"), accuracy_rows, ["Phase", "Accuracy"])
    print(f"  HOAN TAT: {output_dir}\n")


def run_all_metrics():
    """Chạy report lần lượt cho từng dataset trong DATASETS_TO_RUN. Kết quả: result/<dataset>/report_YYYYMMDD_HHMMSS/"""
    os.makedirs(RESULT_ROOT, exist_ok=True)
    print("\n" + "="*68)
    print("  VISION GUARD — RUN FULL REPORT (multi-dataset)")
    print(f"  Data: {DATA_ROOT}  |  Result: {RESULT_ROOT}")
    print("="*68)

    for dataset_name in DATASETS_TO_RUN:
        train_data_dir = os.path.join(DATA_ROOT, dataset_name, "train_data")
        test_data_dir  = os.path.join(DATA_ROOT, dataset_name, "test_data")
        if not os.path.isdir(train_data_dir):
            print(f"\n  BO QUA '{dataset_name}': khong thay thu muc {train_data_dir}")
            continue
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(RESULT_ROOT, dataset_name, f"report_{ts}")
        print("\n" + "="*68)
        print(f"  DATASET: {dataset_name}")
        print("="*68)
        run_report_for_paths(train_data_dir, test_data_dir, output_dir)

    print("="*68)
    print("  TAT CA DATASET DA CHAY XONG.")
    print("="*68)


if __name__ == "__main__":
    run_all_metrics()
