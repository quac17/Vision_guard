"""
model_loader.py — Hàm load MobileFaceNet an toàn, dùng chung cho mọi script trong figures/
"""
import os
import sys

_FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
_SERVER_DIR  = os.path.join(_FIGURES_DIR, "../face-recognizer-server")
for _p in [_FIGURES_DIR, _SERVER_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_mobilefacenet(model_path, device=None, verbose=True):
    """
    Load MobileFaceNet an toàn:
    - Nếu file không tồn tại → random weights (cảnh báo)
    - Nếu checkpoint là full Module → dùng trực tiếp
    - Nếu là state_dict có keys khớp → load_state_dict
    - Nếu keys không khớp (sai model) → thử strict=False, hoặc random weights
    Trả về (model, device, weights_loaded: bool)
    """
    import torch
    from mobilefacenet import MobileFaceNet

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MobileFaceNet(embedding_size=512).to(device)
    weights_loaded = False

    if not os.path.exists(model_path):
        if verbose:
            print(f"  ⚠  Model không tìm thấy: {model_path}")
            print(f"     → Dùng random weights (kết quả embedding không có ý nghĩa thực tế)")
        model.eval()
        return model, device, weights_loaded

    try:
        ckpt = torch.load(model_path, map_location=device)
    except Exception as e:
        if verbose:
            print(f"  ✗  Không đọc được file model: {e}")
        model.eval()
        return model, device, weights_loaded

    # --- Case 1: Full Module ---
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt.to(device)
        weights_loaded = True
        if verbose:
            print(f"  ✓  Đã load Full Module từ: {os.path.basename(model_path)}")

    # --- Case 2: State dict ---
    elif isinstance(ckpt, dict):
        # Kiểm tra xem keys có khớp với MobileFaceNet không
        model_keys = set(model.state_dict().keys())
        ckpt_keys  = set(ckpt.keys())
        overlap    = model_keys & ckpt_keys

        if len(overlap) == 0:
            # Hoàn toàn khác architecture
            if verbose:
                print(f"  ✗  File '{os.path.basename(model_path)}' là model KHÁC architecture")
                print(f"     Keys trong file: {list(ckpt_keys)[:3]}...")
                print(f"     Keys MobileFaceNet cần: {list(model_keys)[:3]}...")
                print(f"     → Bỏ qua, dùng random weights")
        elif len(overlap) == len(model_keys):
            # Khớp hoàn toàn
            model.load_state_dict(ckpt, strict=True)
            weights_loaded = True
            if verbose:
                print(f"  ✓  Đã load weights (strict) từ: {os.path.basename(model_path)}")
        else:
            # Khớp một phần — thử strict=False
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            weights_loaded = True
            if verbose:
                print(f"  ⚠  Load partial weights (strict=False): "
                      f"{len(missing)} missing, {len(unexpected)} unexpected keys")
    else:
        if verbose:
            print(f"  ✗  Kiểu checkpoint không nhận ra: {type(ckpt)}")

    model.eval()
    return model, device, weights_loaded
