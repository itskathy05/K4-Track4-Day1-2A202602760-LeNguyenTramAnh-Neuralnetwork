"""Nạp, chia validation và chuẩn hoá Forest CoverType mà không rò rỉ eval.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    root = Path(processed_dir)
    train_path, eval_path = root / "train.npz", root / "eval.npz"
    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            f"Không thấy {train_path} hoặc {eval_path}. Hãy chạy scripts/split_data.py trước."
        )
    with np.load(train_path, allow_pickle=False) as tr, np.load(eval_path, allow_pickle=False) as ev:
        X_train_full = tr["X"].astype(np.float32, copy=False)
        y_train_full = tr["y"].astype(np.int64, copy=False)
        X_eval = ev["X"].astype(np.float32, copy=False)
        y_eval = ev["y"].astype(np.int64, copy=False)
        eval_row_id = ev["row_id"].astype(np.int64, copy=False)

    for name, X, y in (("train", X_train_full, y_train_full), ("eval", X_eval, y_eval)):
        if X.ndim != 2 or X.shape[1] != 54 or y.shape != (len(X),):
            raise ValueError(f"Shape {name} không hợp lệ: X={X.shape}, y={y.shape}")
        if X.dtype != np.float32 or y.dtype != np.int64:
            raise TypeError(f"Dtype {name} không hợp lệ: X={X.dtype}, y={y.dtype}")
        if len(y) and (y.min() < 0 or y.max() > 6):
            raise ValueError(f"Nhãn {name} phải nằm trong 0..6")
    if eval_row_id.shape != (len(X_eval),) or len(np.unique(eval_row_id)) != len(eval_row_id):
        raise ValueError("eval row_id thiếu, sai shape hoặc bị trùng")
    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction phải nằm trong (0, 1)")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    X_tr = np.asarray(X_tr)
    if X_tr.ndim != 2 or X_tr.shape[1] < N_NUMERIC:
        raise ValueError(f"X_tr phải có shape (N, >= {N_NUMERIC})")
    numeric = X_tr[:, :N_NUMERIC].astype(np.float64, copy=False)
    mean = numeric.mean(axis=0).astype(np.float32)
    std = numeric.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    X = np.asarray(X)
    mean, std = np.asarray(mean), np.asarray(std)
    if mean.shape != (N_NUMERIC,) or std.shape != (N_NUMERIC,):
        raise ValueError("mean/std phải có shape (10,)")
    out = X.astype(np.float32, copy=True)
    safe_std = np.where(std == 0, 1.0, std)
    out[:, :N_NUMERIC] = (out[:, :N_NUMERIC] - mean) / safe_std
    return out


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    X_full, y_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)
    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    dev = torch.device(device)
    data = {
        "X_tr": torch.as_tensor(X_tr, dtype=torch.float32, device=dev),
        "y_tr": torch.as_tensor(y_tr, dtype=torch.int64, device=dev),
        "X_val": torch.as_tensor(X_val, dtype=torch.float32, device=dev),
        "y_val": torch.as_tensor(y_val, dtype=torch.int64, device=dev),
        "X_eval": torch.as_tensor(X_eval, dtype=torch.float32, device=dev),
        "y_eval": torch.as_tensor(y_eval, dtype=torch.int64, device=dev),
        "eval_row_id": eval_row_id.copy(),
        "numeric_mean": mean,
        "numeric_std": std,
        "split_seed": seed,
        "val_fraction": val_fraction,
        "device": str(dev),
    }
    majority = int(np.bincount(y_tr, minlength=7).argmax())
    majority_acc = float((y_val == majority).mean())
    print(
        f"train={len(y_tr):,}, val={len(y_val):,}, eval={len(y_eval):,}; "
        f"majority_class={majority}, val_majority_acc={majority_acc:.4f}"
    )
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    if batch_size <= 0:
        raise ValueError("batch_size phải dương")
    if len(X) != len(y):
        raise ValueError("X và y phải có cùng số mẫu")
    if shuffle:
        # A CPU generator cannot be passed to CUDA randperm. Generate indices on
        # CPU for reproducibility, then move just the index tensor to the device.
        idx = torch.randperm(len(X), generator=generator, device="cpu").to(X.device)
    else:
        idx = torch.arange(len(X), device=X.device)
    for start in range(0, len(X), batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]
