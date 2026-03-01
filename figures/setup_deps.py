"""
setup_deps.py — Tự động kiểm tra và cài đặt các thư viện cần thiết cho Vision Guard Report
"""
import importlib.util
import subprocess
import sys

# ──────────────────────────────────────────
# Danh sách thư viện cần thiết
# ──────────────────────────────────────────
PC_PHASE_DEPS = {
    "torch":       "torch",
    "torchvision": "torchvision",
    "cv2":         "opencv-python",
    "psutil":      "psutil",
    "matplotlib":  "matplotlib",
    "numpy":       "numpy",
}

EDGE_PHASE_DEPS = {
    # tflite-runtime chỉ hỗ trợ Linux/ARM, trên Windows sẽ thử tensorflow-lite
    "tflite_runtime": "tflite-runtime",
}

def _is_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None

def _install(package_name: str) -> bool:
    """Cài thư viện qua pip, trả về True nếu thành công."""
    print(f"  Cai dat: {package_name} ...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name, "--quiet"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            print(f"  OK cai xong: {package_name}")
            return True
        else:
            print(f"  Loi khi cai {package_name}: {result.stderr.strip()[:200]}")
            return False
    except Exception as e:
        print(f"  Exception khi cai {package_name}: {e}")
        return False

def check_and_install_pc_deps() -> bool:
    """
    Kiểm tra & cài đặt thư viện cho PC Phase (PyTorch-based).
    Trả về True nếu tất cả thư viện sẵn sàng.
    """
    print("\n[SETUP] Kiem tra thu vien PC Phase...")
    all_ok = True
    for module, package in PC_PHASE_DEPS.items():
        if _is_installed(module):
            print(f"  OK: {module}")
        else:
            print(f"  Thieu: {module}")
            ok = _install(package)
            if not ok:
                all_ok = False
    return all_ok

def check_and_install_edge_deps() -> bool:
    """
    Kiểm tra & cài đặt thư viện cho Edge Phase (TFLite).
    Trả về True nếu tflite khả dụng.
    Nếu không cài được (Windows), trả về False và báo dùng PyTorch thay thế.
    """
    print("\n[SETUP] Kiem tra thu vien Edge Phase (TFLite)...")
    
    # Thu import tflite_runtime truoc
    if _is_installed("tflite_runtime"):
        print("  OK: tflite_runtime da co")
        return True
    
    # Thu cai tflite-runtime
    print("  Thieu tflite_runtime - Thu cai tflite-runtime ...")
    if _install("tflite-runtime"):
        return True
    
    # Thử import tensorflow rồi dùng tensorflow.lite thay thế
    if _is_installed("tensorflow"):
        print("  OK: tensorflow da co (dung tensorflow.lite thay the)")
        return True
    
    print("  Khong cai duoc tflite tren moi truong nay (thuong la Windows).")
    print("  Edge Phase se dung PyTorch lam inference thay the.")
    return False


if __name__ == "__main__":
    check_and_install_pc_deps()
    check_and_install_edge_deps()
    print("\n[SETUP] Hoan tat kiem tra thu vien.")
