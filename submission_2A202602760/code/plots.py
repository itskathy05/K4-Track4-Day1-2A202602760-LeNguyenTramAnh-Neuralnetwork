"""Biểu đồ bằng chứng cho từng run và từng nhóm thí nghiệm.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    cfg, hist, summary = result["cfg"], result["history"], result["summary"]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not hist.get("epoch"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            f"{cfg.get('exp_id')}\nTraining diverged before epoch 1 completed\n"
            f"optimizer={cfg.get('optimizer')}, lr={cfg.get('lr')}",
            ha="center", va="center", fontsize=12,
        )
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    epochs = hist["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].plot(epochs, hist["train_loss"], label="train (eval mode)")
    axes[0].plot(epochs, hist["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()

    axes[1].plot(epochs, hist["val_acc"], label="val accuracy")
    axes[1].plot(epochs, hist["val_macro_f1"], label="val macro-F1")
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[1].legend()

    axes[2].plot(epochs, hist["grad_norm"], label="mean")
    if "grad_norm_p95" in hist:
        axes[2].plot(epochs, hist["grad_norm_p95"], label="p95", alpha=0.8)
    if cfg.get("clip_norm") is not None:
        axes[2].axhline(cfg["clip_norm"], color="red", linestyle="--", label="clip threshold")
    axes[2].set(title="Gradient norm before clipping", xlabel="Epoch", ylabel="Global L2 norm")
    axes[2].legend()

    best_epoch = summary.get("best_epoch")
    if best_epoch:
        for ax in axes:
            ax.axvline(best_epoch, color="gray", linestyle=":", alpha=0.7)
            ax.grid(alpha=0.2)
    subtitle = (
        f"{cfg['exp_id']} | {cfg['optimizer']} lr={cfg['lr']:g} | batch={cfg['batch']} "
        f"hidden={tuple(cfg['hidden'])} drop={cfg['dropout']} {cfg['precision']}"
    )
    fig.suptitle(subtitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    if not results:
        raise ValueError("results không được rỗng")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for result in results:
        history = result["history"]
        if metric not in history:
            raise KeyError(f"Không có metric {metric!r} trong {result['cfg']['exp_id']}")
        ax.plot(history["epoch"], history[metric], label=result["cfg"]["exp_id"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(title or f"So sánh {metric}")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
