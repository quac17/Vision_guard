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
FRR_MAX_FOR_ROC = 0.1  # Ưu tiên tiện dụng: chọn ngưỡng sao cho FRR <= 10%, tối đa hóa TAR (figures.txt)
FAR_MAX_FOR_ROC = 0.15  # Giới hạn FAR <= 15%: ưu tiên tiện dụng nhưng không nhận nhầm quá nhiều người lạ

# Chọn dataset để chạy report (comment dòng không cần chạy)
DATASETS_TO_RUN = [
    "dataset_0",
    "dataset_1",
    "dataset_2",
    "dataset_3",
    "dataset_4",
    "dataset_5",
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


def _get_genuine_impostor_scores(report_list, known_set):
    """
    Từ report (actual, min_distance): tách điểm genuine (người trong DB) và impostor (người lạ).
    Trả về (genuine_scores, impostor_scores) — mỗi phần tử là khoảng cách cosine.
    """
    genuine = []
    impostor = []
    for r in report_list:
        actual = r.get("actual", "")
        dist = r.get("min_distance", float("inf"))
        if actual in known_set:
            genuine.append(float(dist))
        else:
            impostor.append(float(dist))
    return genuine, impostor


def _compute_tar_far_frr(genuine_scores, impostor_scores, threshold):
    """
    Tại một ngưỡng: TAR = tỉ lệ genuine được chấp nhận (distance < thresh),
    FAR = tỉ lệ impostor bị chấp nhận nhầm, FRR = 1 - TAR.
    """
    n_gen = len(genuine_scores)
    n_imp = len(impostor_scores)
    if n_gen == 0:
        tar = 0.0
        frr = 0.0
    else:
        accept_gen = sum(1 for s in genuine_scores if s < threshold)
        tar = accept_gen / n_gen
        frr = 1.0 - tar
    if n_imp == 0:
        far = 0.0
    else:
        far = sum(1 for s in impostor_scores if s < threshold) / n_imp
    return tar, far, frr


def _compute_roc_and_eer(genuine_scores, impostor_scores, thresholds=None):
    """
    Quét ngưỡng, tính TAR/FAR/FRR tại mỗi điểm. Tìm EER (điểm FAR = FRR).
    Trả về (roc_rows, eer_value, eer_threshold). roc_rows = [(th, TAR, FAR, FRR), ...].
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.01).tolist()
    roc_rows = []
    eer_value = None
    eer_threshold = None
    for th in thresholds:
        tar, far, frr = _compute_tar_far_frr(genuine_scores, impostor_scores, th)
        roc_rows.append((th, tar, far, frr))
        if eer_value is None and frr >= far:
            # EER ≈ điểm giao FAR = FRR; nội suy đơn giản
            eer_value = (far + frr) / 2.0
            eer_threshold = th
    if eer_value is None and roc_rows:
        eer_value = roc_rows[-1][2]  # FAR cuối
        eer_threshold = roc_rows[-1][0]
    return roc_rows, eer_value, eer_threshold


def _best_threshold_under_frr_and_far_max(roc_rows, frr_max, far_max):
    """
    Điểm khuyến nghị: FRR <= frr_max (tiện dụng), FAR <= far_max (bảo mật), chọn ngưỡng có TAR tốt nhất.
    Trả về (threshold, tar, far, frr) hoặc None nếu không có điểm thỏa.
    """
    candidates = [(r[0], r[1], r[2], r[3]) for r in roc_rows if r[3] <= frr_max and r[2] <= far_max]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])  # max TAR


def save_roc_curve_image(path, roc_rows, eer_value, eer_threshold, title="ROC Curve (TAR vs FAR)"):
    """
    Vẽ đường cong ROC: trục x = FAR, trục y = TAR (TAR vs FRR vô nghĩa vì TAR + FRR = 100%).
    Đánh dấu EER và điểm khuyến nghị: FRR <= FRR_MAX, FAR <= FAR_MAX, TAR tối đa. Vạch đứng FAR = FAR_MAX.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        far_list = [r[2] for r in roc_rows]
        tar_list = [r[1] for r in roc_rows]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(far_list, tar_list, "b-", linewidth=2, label="ROC (TAR vs FAR)")
        if eer_value is not None and eer_threshold is not None:
            ax.plot(eer_value, 1 - eer_value, "ro", markersize=10, label=f"EER = {eer_value:.3f} (thr≈{eer_threshold:.2f})")
        best = _best_threshold_under_frr_and_far_max(roc_rows, FRR_MAX_FOR_ROC, FAR_MAX_FOR_ROC)
        if best is not None:
            th, tar, far, frr = best
            ax.plot(far, tar, "g*", markersize=14, label=f"FRR≤{FRR_MAX_FOR_ROC*100:.0f}% FAR≤{FAR_MAX_FOR_ROC*100:.0f}% TAR={tar:.2%} (thr={th:.2f})")
        ax.axvline(x=FAR_MAX_FOR_ROC, color="gray", linestyle=":", alpha=0.7, label=f"FAR = {FAR_MAX_FOR_ROC*100:.0f}%")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        ax.set_xlabel("FAR (False Acceptance Rate)")
        ax.set_ylabel("TAR (True Acceptance Rate)")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  -> Image: {path}")
    except Exception as e:
        print(f"  LOI xuat anh ROC: {e}")


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

    # TAR, FAR, FRR, ROC, EER — Train (ROC: trục x = FRR để ưu tiên tiện dụng)
    gen_tr, imp_tr = _get_genuine_impostor_scores(train_report, known_set)
    tar_tr, far_tr, frr_tr = _compute_tar_far_frr(gen_tr, imp_tr, EDGE_THRESHOLD)
    roc_tr, eer_tr, eer_th_tr = _compute_roc_and_eer(gen_tr, imp_tr)
    best_tr = _best_threshold_under_frr_and_far_max(roc_tr, FRR_MAX_FOR_ROC, FAR_MAX_FOR_ROC)
    perf_rows_tr = [
        ["TAR (%)", round(tar_tr * 100, 4)],
        ["FAR (%)", round(far_tr * 100, 4)],
        ["FRR (%)", round(frr_tr * 100, 4)],
        ["EER", round(eer_tr, 4) if eer_tr is not None else "N/A"],
        ["EER Threshold", round(eer_th_tr, 4) if eer_th_tr is not None else "N/A"],
        ["Threshold used", EDGE_THRESHOLD],
    ]
    if best_tr is not None:
        perf_rows_tr.append([f"Recommended threshold (FRR≤{FRR_MAX_FOR_ROC*100:.0f}%, FAR≤{FAR_MAX_FOR_ROC*100:.0f}%)", round(best_tr[0], 4)])
        perf_rows_tr.append(["TAR at recommended (%)", round(best_tr[1] * 100, 4)])
    save_csv(os.path.join(output_dir, "performance_metrics_train.csv"), perf_rows_tr, ["Metric", "Value"])
    roc_tr_filtered = [(th, tar, far, frr) for th, tar, far, frr in roc_tr if far <= FAR_MAX_FOR_ROC]
    save_csv(os.path.join(output_dir, "roc_curve_train.csv"),
             [[th, round(tar, 4), round(far, 4), round(frr, 4)] for th, tar, far, frr in roc_tr_filtered],
             ["Threshold", "TAR", "FAR", "FRR"])
    save_roc_curve_image(
        os.path.join(output_dir, "roc_curve_train.png"),
        roc_tr, eer_tr, eer_th_tr, title="ROC Curve — Train (TAR vs FAR)"
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
        # TAR, FAR, FRR, ROC, EER — Test (ROC: trục x = FRR)
        gen_te, imp_te = _get_genuine_impostor_scores(unknown_report, known_set)
        tar_te, far_te, frr_te = _compute_tar_far_frr(gen_te, imp_te, EDGE_THRESHOLD)
        roc_te, eer_te, eer_th_te = _compute_roc_and_eer(gen_te, imp_te)
        best_te = _best_threshold_under_frr_and_far_max(roc_te, FRR_MAX_FOR_ROC, FAR_MAX_FOR_ROC)
        perf_rows_te = [
            ["TAR (%)", round(tar_te * 100, 4)],
            ["FAR (%)", round(far_te * 100, 4)],
            ["FRR (%)", round(frr_te * 100, 4)],
            ["EER", round(eer_te, 4) if eer_te is not None else "N/A"],
            ["EER Threshold", round(eer_th_te, 4) if eer_th_te is not None else "N/A"],
            ["Threshold used", EDGE_THRESHOLD],
        ]
        if best_te is not None:
            perf_rows_te.append([f"Recommended threshold (FRR≤{FRR_MAX_FOR_ROC*100:.0f}%, FAR≤{FAR_MAX_FOR_ROC*100:.0f}%)", round(best_te[0], 4)])
            perf_rows_te.append(["TAR at recommended (%)", round(best_te[1] * 100, 4)])
        save_csv(os.path.join(output_dir, "performance_metrics_test.csv"), perf_rows_te, ["Metric", "Value"])
        roc_te_filtered = [(th, tar, far, frr) for th, tar, far, frr in roc_te if far <= FAR_MAX_FOR_ROC]
        save_csv(os.path.join(output_dir, "roc_curve_test.csv"),
                 [[th, round(tar, 4), round(far, 4), round(frr, 4)] for th, tar, far, frr in roc_te_filtered],
                 ["Threshold", "TAR", "FAR", "FRR"])
        save_roc_curve_image(
            os.path.join(output_dir, "roc_curve_test.png"),
            roc_te, eer_te, eer_th_te, title="ROC Curve — Test (TAR vs FAR)"
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
