"""Pipeline huấn luyện, đánh giá và xuất dự đoán dùng chung cho mọi thí nghiệm.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import copy
import csv
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, activation_stats, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # được chọn bằng validation, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    class_weight="none",       # "none" | "balanced" | "sqrt_balanced"
    scheduler=None,             # None | "cosine"
    train_eval_size=50_000,
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism is preferred for this educational comparison. warn_only avoids
    # failing on a CUDA kernel without a deterministic implementation.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # older PyTorch
        torch.use_deterministic_algorithms(True)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    cm = np.asarray(cm, dtype=np.float64)
    if cm.shape != (7, 7):
        raise ValueError(f"cm phải có shape (7, 7), nhận {cm.shape}")
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    if batch_size <= 0:
        raise ValueError("batch_size phải dương")
    model.eval()
    chunks = []
    for start in range(0, len(X), batch_size):
        logits = model(X[start:start + batch_size])
        chunks.append(logits.argmax(dim=1).to(torch.int64))
    if not chunks:
        return torch.empty(0, dtype=torch.int64, device=X.device)
    return torch.cat(chunks)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192,
             class_weights: torch.Tensor | None = None) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    if len(X) != len(y) or len(y) == 0:
        raise ValueError("Tập đánh giá phải không rỗng và X/y cùng độ dài")
    model.eval()
    total_loss, total_den = 0.0, 0.0
    cm = np.zeros((7, 7), dtype=np.int64)
    for start in range(0, len(X), batch_size):
        xb, yb = X[start:start + batch_size], y[start:start + batch_size]
        logits = model(xb)
        if loss_name == "mse":
            target = F.one_hot(yb, num_classes=7).to(logits.dtype)
            total_loss += float(F.mse_loss(logits, target, reduction="sum").item())
            total_den += float(yb.numel() * 7)
        else:
            total_loss += float(F.cross_entropy(
                logits, yb, weight=class_weights, reduction="sum"
            ).item())
            if class_weights is None:
                total_den += float(yb.numel())
            else:
                total_den += float(class_weights[yb].sum().item())
        pred = logits.argmax(dim=1)
        yt = yb.detach().cpu().numpy()
        yp = pred.detach().cpu().numpy()
        np.add.at(cm, (yt, yp), 1)
    return {
        "loss": total_loss / max(total_den, 1.0),
        "acc": float(np.trace(cm) / cm.sum()),
        "macro_f1": macro_f1_from_confusion(cm),
        "confusion_matrix": cm,
    }


def compute_loss(logits, y, loss_name: str, class_weights: torch.Tensor | None = None):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "mse":
        target = F.one_hot(y, num_classes=logits.shape[1]).to(logits.dtype)
        return F.mse_loss(logits, target, reduction="mean")
    if loss_name in {"ce", "ce_weighted", "ce_sqrt_weighted"}:
        return F.cross_entropy(logits, y, weight=class_weights)
    raise ValueError(f"loss={loss_name!r} không được hỗ trợ")


def make_class_weights(y: torch.Tensor, mode: str) -> torch.Tensor | None:
    """Compute normalized class weights exclusively from the training labels."""
    if mode in {None, "none"}:
        return None
    if mode not in {"balanced", "sqrt_balanced"}:
        raise ValueError("class_weight phải là none, balanced hoặc sqrt_balanced")
    counts = torch.bincount(y, minlength=7).to(torch.float32)
    if (counts == 0).any():
        raise ValueError("Training split phải chứa đủ 7 lớp")
    weights = counts.sum() / (len(counts) * counts)
    if mode == "sqrt_balanced":
        weights = weights.sqrt()
    return weights / weights.mean()


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise RuntimeError(f"precision={precision} yêu cầu CUDA trong lab này")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    required = ("exp_id", "lr", "seed", "epochs", "batch", "hidden")
    missing = [k for k in required if cfg.get(k) is None]
    if missing:
        raise ValueError(f"Cấu hình thiếu giá trị: {missing}")
    if cfg["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision phải là fp32, fp16 hoặc bf16")

    set_seed(int(cfg["seed"]))
    device = data["X_tr"].device
    hidden = tuple(cfg["hidden"])
    if hidden not in EXPECTED_PARAMS:
        raise ValueError(f"Kiến trúc {hidden} không thuộc ba kiến trúc được phép")
    model = MLP(hidden=hidden, dropout=float(cfg["dropout"]), init=cfg["init"]).to(device)
    assert count_params(model) == EXPECTED_PARAMS[hidden]
    optimizer = build_optimizer(
        cfg["optimizer"], model.parameters(), cfg["lr"], cfg["weight_decay"], cfg["momentum"]
    )
    steps_per_epoch = math.ceil(len(data["X_tr"]) / int(cfg["batch"]))
    scheduler = build_scheduler(
        optimizer, cfg.get("scheduler"), int(cfg["epochs"]) * steps_per_epoch
    )
    class_weights = make_class_weights(data["y_tr"], cfg.get("class_weight", "none"))
    loss_name = cfg["loss"]
    if cfg.get("class_weight", "none") == "balanced":
        loss_name = "ce_weighted"
    elif cfg.get("class_weight", "none") == "sqrt_balanced":
        loss_name = "ce_sqrt_weighted"

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg["precision"] == "fp16"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step0 = evaluate(model, data["X_val"], data["y_val"], loss_name, class_weights=class_weights)
    step0_activation_stats = activation_stats(model, data["X_val"][: min(2048, len(data["X_val"]))])
    history = {k: [] for k in (
        "epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
        "grad_norm", "grad_norm_p75", "grad_norm_p95", "grad_norm_max",
        "clip_fraction", "epoch_time_s", "lr",
    )}
    grad_norm_samples: list[float] = []
    best_state, best_epoch, best_val_loss = None, 0, float("inf")
    diverged = False
    generator = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))

    train_eval_size = min(int(cfg.get("train_eval_size", 50_000)), len(data["X_tr"]))
    X_train_eval = data["X_tr"][:train_eval_size]
    y_train_eval = data["y_tr"][:train_eval_size]

    for epoch in range(1, int(cfg["epochs"]) + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        model.train()
        epoch_norms: list[float] = []
        clipped_steps = 0
        for xb, yb in iterate_batches(
            data["X_tr"], data["y_tr"], int(cfg["batch"]), generator, shuffle=True
        ):
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, cfg["precision"]):
                logits = model(xb)
                loss = compute_loss(logits, yb, loss_name, class_weights)
            if not torch.isfinite(loss):
                diverged = True
                break
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            grad_norm = clip_gradients(model.parameters(), cfg.get("clip_norm"))
            if not math.isfinite(grad_norm):
                diverged = True
                break
            epoch_norms.append(grad_norm)
            if cfg.get("clip_norm") is not None and grad_norm > float(cfg["clip_norm"]):
                clipped_steps += 1
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if diverged:
            break

        train_metrics = evaluate(
            model, X_train_eval, y_train_eval, loss_name, class_weights=class_weights
        )
        val_metrics = evaluate(
            model, data["X_val"], data["y_val"], loss_name, class_weights=class_weights
        )
        norms = np.asarray(epoch_norms, dtype=float)
        grad_norm_samples.extend(epoch_norms)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["grad_norm"].append(float(norms.mean()))
        history["grad_norm_p75"].append(float(np.percentile(norms, 75)))
        history["grad_norm_p95"].append(float(np.percentile(norms, 95)))
        history["grad_norm_max"].append(float(norms.max()))
        history["clip_fraction"].append(float(clipped_steps / max(len(norms), 1)))
        history["epoch_time_s"].append(float(elapsed))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if not history["epoch"]:
        summary = {
            "step0_loss": step0["loss"], "best_val_loss": None, "best_epoch": 0,
            "final_train_loss": None, "final_val_loss": None, "val_acc": None,
            "val_macro_f1": None, "time_per_epoch_s": None, "peak_mem_MB": None,
            "diverged": True, "grad_norm_p75": None, "grad_norm_p95": None,
            "clip_fraction": None, "activation_stats": step0_activation_stats, "n_updates": 0,
        }
    else:
        best_idx = best_epoch - 1
        peak_mem = (
            float(torch.cuda.max_memory_allocated(device) / 1024 ** 2)
            if device.type == "cuda" else 0.0
        )
        all_norms = np.asarray(grad_norm_samples, dtype=float)
        summary = {
            "step0_loss": step0["loss"],
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "val_acc": history["val_acc"][best_idx],
            "val_macro_f1": history["val_macro_f1"][best_idx],
            "time_per_epoch_s": float(np.mean(history["epoch_time_s"])),
            "peak_mem_MB": peak_mem,
            "diverged": bool(diverged),
            "grad_norm_p75": float(np.percentile(all_norms, 75)),
            "grad_norm_p95": float(np.percentile(all_norms, 95)),
            "clip_fraction": float(np.mean(history["clip_fraction"])),
            "activation_stats": step0_activation_stats,
            "n_updates": int(len(grad_norm_samples)),
        }
    return {
        "schema_version": 1,
        "cfg": copy.deepcopy(cfg),
        "history": history,
        "summary": summary,
        "best_state": best_state,
        "grad_norm_samples": grad_norm_samples,
    }


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    row_id = np.asarray(row_id)
    preds = np.asarray(preds)
    if row_id.ndim != 1 or preds.shape != row_id.shape:
        raise ValueError("row_id và preds phải là vector cùng shape")
    if len(np.unique(row_id)) != len(row_id):
        raise ValueError("row_id bị trùng")
    if not np.issubdtype(preds.dtype, np.integer) or (len(preds) and (preds.min() < 0 or preds.max() > 6)):
        raise ValueError("preds phải là số nguyên trong 0..6")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "pred"])
        writer.writerows(zip(row_id.astype(np.int64), preds.astype(np.int64)))
    tmp.replace(out)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    if result.get("best_state") is None:
        raise ValueError("Experiment không có best_state; không thể dự đoán eval")
    merged = {**DEFAULT_CFG, **cfg}
    device = data["X_eval"].device
    model = MLP(
        hidden=tuple(merged["hidden"]), dropout=float(merged["dropout"]), init=merged["init"]
    ).to(device)
    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
