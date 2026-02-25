"""
run_full_report.py — Tự động xuất toàn bộ số liệu báo cáo đồ án Vision Guard (RÚT GỌN)
===========================================================================
Kết quả sẽ được lưu vào: figures/report_YYYYMMDD_HHMMSS/
===========================================================================
"""
import os, sys, csv, numpy as np
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
MODEL_PATH     = os.path.join(SERVER_DIR, "models/mobilefacenet.tflite")
THRESHOLD      = 1.0   

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
    print(f"  → CSV: {path}")


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
        print("\n[LỖI] Không có dữ liệu trong figures/train_data/")
        sys.exit(1)

    # Tạo folder output
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(FIGURES_DIR, f"report_{ts}")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*68)
    print("  VISION GUARD — RÚT GỌN SỐ LIỆU BÁO CÁO")
    print(f"  Kết quả: {output_dir}")
    print("="*68)

    system_rows = []

    # ──────────────────────────────────────────────────────────
    # STEP 1 — System Metrics
    # ──────────────────────────────────────────────────────────
    print("\n[STEP 1] THU THẬP THÔNG SỐ HỆ THỐNG")
    
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
    print("\n[STEP 2] XÂY DỰNG FACE DATABASE...")
    database = accuracy_tools.build_embeddings_from_train_data(TRAIN_DATA_DIR, MODEL_PATH)
    
    db_stats = metrics.get_db_stats(database)
    save_csv(os.path.join(output_dir, "database_stats.csv"),
             [["Total Identities", db_stats["num_ids"]],
              ["Embedding Dimension", db_stats["dimension"], "512-dim"]],
             ["Metric", "Value", "Note"])

    # ──────────────────────────────────────────────────────────
    # STEP 3 — Accuracy PC
    # ──────────────────────────────────────────────────────────
    pc_acc, pc_cm, pc_names = accuracy_tools.test_accuracy_pc(database, TRAIN_DATA_DIR, MODEL_PATH, threshold=THRESHOLD)

    # Xuất file Train Details gộp theo từng Identity (theo yêu cầu)
    train_grouped_rows = []
    if pc_cm is not None:
        for i, name in enumerate(pc_names):
            total_samples = int(np.sum(pc_cm[i]))
            correct_samples = int(pc_cm[i][i])
            accuracy = (correct_samples / total_samples * 100) if total_samples > 0 else 0
            train_grouped_rows.append([name, total_samples, correct_samples, round(accuracy, 2)])
        
    if train_grouped_rows:
        save_csv(os.path.join(output_dir, "train_details_grouped.csv"),
                 train_grouped_rows,
                 ["Identity", "Total Samples", "Correct", "Accuracy (%)"])

    # ──────────────────────────────────────────────────────────
    # STEP 4 — Accuracy Edge & Details
    # ──────────────────────────────────────────────────────────
    edge_acc = 0
    if n_test > 0:
        edge_acc, _, _, unknown_report = accuracy_tools.test_accuracy_edge(
            database, TEST_DATA_DIR, MODEL_PATH, threshold=THRESHOLD
        )
        
        # Export Test Details
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

    # ──────────────────────────────────────────────────────────
    # STEP 5 — Accuracy Summary
    # ──────────────────────────────────────────────────────────
    accuracy_rows = [
        ["PC Phase (train_data)", f"{round(pc_acc, 2)}%"],
        ["Edge Phase (test_data)", f"{round(edge_acc, 2)}%" if n_test > 0 else "N/A"]
    ]
    save_csv(os.path.join(output_dir, "accuracy.csv"), accuracy_rows, ["Phase", "Accuracy"])

    print("\n" + "="*68)
    print(f"  HOÀN TẤT! Kết quả tại: {output_dir}")
    print("="*68)


if __name__ == "__main__":
    run_all_metrics()
