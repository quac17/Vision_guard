"""
run_full_report.py — Tự động xuất toàn bộ số liệu báo cáo đồ án Vision Guard
===========================================================================
Cách dùng:
  1. Copy ảnh TRAIN vào: figures/train_data/<tên_người>/  (ảnh .pgm/.jpg/.webp)
  2. Copy ảnh TEST vào : figures/test_data/<tên_người>/
     (s33-s35 là identity KHÔNG có trong DB → kiểm tra False Positive)
  3. Chạy: python run_full_report.py
     Kết quả sẽ được lưu vào: figures/report_YYYYMMDD_HHMMSS/
===========================================================================
"""
import os, sys, json, csv, numpy as np
from datetime import datetime

# ── Đường dẫn tự phát hiện ──────────────────────────────────────────────────
FIGURES_DIR    = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR     = os.path.join(FIGURES_DIR, "../face-recognizer-server")

for _p in [FIGURES_DIR, SERVER_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Cấu hình (chỉnh tại đây nếu cần) ────────────────────────────────────────
TRAIN_DATA_DIR = os.path.join(FIGURES_DIR, "train_data")      # ← Ảnh train
TEST_DATA_DIR  = os.path.join(FIGURES_DIR, "test_data")       # ← Ảnh test
MODEL_PTH      = os.path.join(SERVER_DIR, "models/mobilefacenet_pretrained.pth")
THRESHOLD      = 1.0   # Ngưỡng Euclidean distance

# ── Bước 0: Tự động cài thư viện ────────────────────────────────────────────
import setup_deps
setup_deps.check_and_install_pc_deps()
setup_deps.check_and_install_edge_deps()

import metrics
import accuracy_tools


# ─────────────────────────────────────────────────────────────────────────────
def save_csv(path, rows, headers):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"  → CSV: {path}")


def dir_count(path):
    if not os.path.exists(path): return 0
    return len([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])


# ─────────────────────────────────────────────────────────────────────────────
def run_all_metrics():
    # Tạo folder train/test nếu chưa có
    for d in [TRAIN_DATA_DIR, TEST_DATA_DIR]:
        os.makedirs(d, exist_ok=True)

    n_train = dir_count(TRAIN_DATA_DIR)
    n_test  = dir_count(TEST_DATA_DIR)

    if n_train == 0:
        print("\n[LỖI] Không có dữ liệu trong figures/train_data/")
        print("   → Copy ảnh vào: figures/train_data/<tên_người>/<ảnh.*>")
        sys.exit(1)

    # Tạo folder output
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(FIGURES_DIR, f"report_{ts}")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*68)
    print("  VISION GUARD — TỰ ĐỘNG XUẤT SỐ LIỆU BÁO CÁO ĐỒ ÁN")
    print(f"  Kết quả : {output_dir}")
    print(f"  Train   : {TRAIN_DATA_DIR}  ({n_train} identities)")
    print(f"  Test    : {TEST_DATA_DIR}   ({n_test} identities)")
    print("="*68)

    system_rows = []

    # ──────────────────────────────────────────────────────────
    # STEP 1 — System Metrics
    # ──────────────────────────────────────────────────────────
    print("\n[STEP 1] THU THẬP THÔNG SỐ HỆ THỐNG & DỮ LIỆU")

    # Storage: tổng kích thước train_data (thay cho so sánh .pgm vs .webp)
    train_size = metrics.get_dir_size(TRAIN_DATA_DIR)
    test_size  = metrics.get_dir_size(TEST_DATA_DIR)
    system_rows.append(["Train Data Size (MB)", round(train_size / (1024*1024), 2)])
    system_rows.append(["Test Data Size (MB)",  round(test_size  / (1024*1024), 2)])
    system_rows.append(["Train Identities",  n_train])
    system_rows.append(["Test  Identities",  n_test])

    # Kích thước model
    model_mb = metrics.get_model_info(MODEL_PTH)
    system_rows.append(["Model Size (MB)", round(model_mb, 3)])

    # CPU/RAM
    sys_res = metrics.monitor_system()
    if sys_res:
        cpu, ram = sys_res
        system_rows.append(["CPU Usage (%)", cpu])
        system_rows.append(["RAM Usage (%)", ram])

    # Inference Latency (PyTorch, 100 lần)
    print("\n[STEP 1.1] ĐO LATENCY INFERENCE (PyTorch, 100 iterations)...")
    lat_pc = metrics.measure_inference_time(MODEL_PTH, data_sample_dir=TRAIN_DATA_DIR,
                                             is_tflite=False, iterations=100)
    system_rows.append(["Inference Latency PC (ms)", round(lat_pc, 2)])

    # Pre-processing Time
    print("\n[STEP 1.2] ĐO THỜI GIAN TIỀN XỬ LÝ (50 iterations)...")
    preproc_ms = metrics.measure_preprocessing_time(TRAIN_DATA_DIR, iterations=50)
    system_rows.append(["Preprocessing Time (ms)", round(preproc_ms, 2)])

    save_csv(os.path.join(output_dir, "system_metrics.csv"),
             system_rows, ["Metric", "Value"])

    # ──────────────────────────────────────────────────────────
    # STEP 2 — Build Database từ train_data
    # ──────────────────────────────────────────────────────────
    print("\n[STEP 2] XÂY DỰNG FACE DATABASE TỪ TRAIN DATA...")
    db_json_path = os.path.join(output_dir, "face_embeddings.json")
    database = accuracy_tools.build_embeddings_from_train_data(
        TRAIN_DATA_DIR, MODEL_PTH, db_json_path
    )
    if not database:
        print("[LỖI] Không thể tạo database — kiểm tra ảnh và model.")
        sys.exit(1)

    db_stats = metrics.analyze_db_stats(db_json_path)
    save_csv(os.path.join(output_dir, "database_stats.csv"),
             [["Total Identities", db_stats.get("num_ids", len(database))],
              ["Embedding Dimension", db_stats.get("dimension", 512)],
              ["DB File Size (KB)", db_stats.get("size_kb", 0)]],
             ["Metric", "Value"])

    # ──────────────────────────────────────────────────────────
    # STEP 3 — Inter-class Distance Distribution
    # ──────────────────────────────────────────────────────────
    print("\n[STEP 3] PHÂN TÍCH KHOẢNG CÁCH LIÊN LỚP (INTER-CLASS)...")
    inter_dists = accuracy_tools.calculate_inter_class_distances(database)
    if inter_dists:
        accuracy_tools.plot_distance_histogram(
            inter_dists,
            output_path=os.path.join(output_dir, "distance_distribution.png")
        )
        save_csv(os.path.join(output_dir, "inter_class_distances.csv"),
                 [[round(d, 5)] for d in inter_dists],
                 ["Euclidean Distance"])

    # ──────────────────────────────────────────────────────────
    # STEP 4 — PC Phase: test trên train_data
    # ──────────────────────────────────────────────────────────
    print("\n[STEP 4] ĐÁNH GIÁ ĐỘ CHÍNH XÁC — PC PHASE (train_data)...")
    pc_acc, pc_cm, pc_names = accuracy_tools.test_accuracy_pc(
        database, TRAIN_DATA_DIR, MODEL_PTH, threshold=THRESHOLD
    )
    accuracy_tools.report_accuracy(
        pc_acc, pc_cm, pc_names,
        output_name=os.path.join(output_dir, "pc_phase")
    )

    # CSV per-class PC
    pc_rows = []
    if pc_cm is not None:
        for i, name in enumerate(pc_names):
            rs  = np.sum(pc_cm[i])
            acc = pc_cm[i][i] / rs * 100 if rs > 0 else 0
            pc_rows.append([name, int(pc_cm[i][i]), int(rs), round(acc, 2)])
    save_csv(os.path.join(output_dir, "pc_accuracy.csv"),
             pc_rows,
             ["Class Name", "Correct", "Total Detected", "Accuracy (%)"])

    # ──────────────────────────────────────────────────────────
    # STEP 5 — Edge Phase: test trên test_data
    # ──────────────────────────────────────────────────────────
    if n_test == 0:
        print("\n[STEP 5] ⚠ Không có dữ liệu trong figures/test_data/ — bỏ qua Edge Phase.")
    else:
        print("\n[STEP 5] ĐÁNH GIÁ ĐỘ CHÍNH XÁC — EDGE PHASE (test_data)...")
        edge_acc, edge_cm, edge_names, unknown_report = accuracy_tools.test_accuracy_edge(
            database, TEST_DATA_DIR, MODEL_PTH, threshold=THRESHOLD
        )
        accuracy_tools.report_accuracy(
            edge_acc, edge_cm, edge_names,
            output_name=os.path.join(output_dir, "edge_phase")
        )

        # CSV per-class Edge
        edge_rows = []
        if edge_cm is not None:
            for i, name in enumerate(edge_names):
                rs  = np.sum(edge_cm[i])
                acc = edge_cm[i][i] / rs * 100 if rs > 0 else 0
                edge_rows.append([name, int(edge_cm[i][i]), int(rs), round(acc, 2)])
        save_csv(os.path.join(output_dir, "edge_accuracy.csv"),
                 edge_rows,
                 ["Class Name", "Correct", "Total Detected", "Accuracy (%)"])

        # CSV Unknown Report (FAR/FRR)
        if unknown_report:
            fp_count  = sum(1 for r in unknown_report if r["false_positive"])
            tot_unk   = len(unknown_report)
            far       = fp_count / tot_unk * 100 if tot_unk else 0
            save_csv(
                os.path.join(output_dir, "edge_unknown_report.csv"),
                [[r["actual"], r["image"], r["predicted"],
                  r["min_distance"], "YES" if r["false_positive"] else "NO"]
                 for r in unknown_report],
                ["Actual Person", "Image", "Predicted As",
                 "Min Distance", "False Positive"]
            )
            save_csv(
                os.path.join(output_dir, "edge_far_frr.csv"),
                [["FAR — False Acceptance Rate (%)", round(far, 2)],
                 ["Total Unknown Images Tested",     tot_unk],
                 ["False Positives (unknown → accepted)", fp_count],
                 ["FRR — False Rejection Rate (%)",
                  round((1 - edge_acc / 100) * 100, 2) if edge_acc else 0]],
                ["Metric", "Value"]
            )
            print(f"  FAR: {far:.2f}%  ({fp_count}/{tot_unk} ảnh lạ bị nhận nhầm)")

    # ──────────────────────────────────────────────────────────
    # Kết thúc
    # ──────────────────────────────────────────────────────────
    print("\n" + "="*68)
    print("  HOÀN TẤT! Tất cả kết quả đã được lưu vào:")
    print(f"  {output_dir}")
    print("="*68)
    print("\n  Files xuất ra:")
    for fn in sorted(os.listdir(output_dir)):
        fsize = os.path.getsize(os.path.join(output_dir, fn))
        print(f"    {fn:<50} {fsize:>9} bytes")


if __name__ == "__main__":
    run_all_metrics()
