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
    print(f"  → Cài đặt: {package_name} ...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name, "--quiet"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            print(f"  ✓ Cài thành công: {package_name}")
            return True
        else:
            print(f"  ✗ Lỗi khi cài {package_name}: {result.stderr.strip()[:200]}")
            return False
    except Exception as e:
        print(f"  ✗ Exception khi cài {package_name}: {e}")
        return False

def check_and_install_pc_deps() -> bool:
    """
    Kiểm tra & cài đặt thư viện cho PC Phase (PyTorch-based).
    Trả về True nếu tất cả thư viện sẵn sàng.
    """
    print("\n[SETUP] Kiểm tra thư viện PC Phase...")
    all_ok = True
    for module, package in PC_PHASE_DEPS.items():
        if _is_installed(module):
            print(f"  ✓ Đã có: {module}")
        else:
            print(f"  ✗ Thiếu: {module}")
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
    print("\n[SETUP] Kiểm tra thư viện Edge Phase (TFLite)...")
    
    # Thử import tflite_runtime trước
    if _is_installed("tflite_runtime"):
        print("  ✓ tflite_runtime: Đã có")
        return True
    
    # Thử cài tflite-runtime
    print("  ✗ Thiếu tflite_runtime — Thử cài tflite-runtime ...")
    if _install("tflite-runtime"):
        return True
    
    # Thử import tensorflow rồi dùng tensorflow.lite thay thế
    if _is_installed("tensorflow"):
        print("  ✓ tensorflow: Đã có (sẽ dùng tensorflow.lite thay thế)")
        return True
    
    print("  ⚠ Không thể cài tflite trên môi trường này (thường là Windows).")
    print("    → Edge Phase sẽ dùng PyTorch làm inference engine thay thế.")
    return False


if __name__ == "__main__":
    check_and_install_pc_deps()
    check_and_install_edge_deps()
    print("\n[SETUP] Hoàn tất kiểm tra thư viện.")
