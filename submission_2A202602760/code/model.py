"""Các kiến trúc MLP được phép và tiện ích chẩn đoán khởi tạo.

Model: MLP cho bài toán 7 lớp, shape cố định (xem README mục 3 và GUIDE, "Quy định kiến trúc"):

    x (B, 54) -> Linear(54, h1) -> ReLU -> [Dropout] -> Linear(h1, h2) -> ReLU -> [Dropout]
              -> ... -> Linear(h_last, 7) -> logits (B, 7)

Quy tắc:
  - Lớp cuối ra logit thô, KHÔNG softmax trong model (softmax nằm trong hàm mất mát).
  - Dropout chỉ đặt sau ReLU của lớp ẩn; không đặt trên đầu vào hay logit.
  - Mọi nn.Linear đều có bias. Không BatchNorm, không residual.
  - Số tham số phải khớp EXPECTED_PARAMS bên dưới.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Số tham số bắt buộc ứng với từng kiến trúc (in_features=54, num_classes=7)
EXPECTED_PARAMS = {
    (256, 128): 47_879,        # M-base  (baseline)
    (512, 256): 161_287,       # M-wide  (tuỳ chọn)
    (256, 128, 64): 55_687,    # M-deep  (tuỳ chọn)
}


class MLP(nn.Module):
    """MLP theo quy định ở đầu file.

    Args:
        hidden:   tuple số nơ-ron các lớp ẩn, ví dụ (256, 128)
        dropout:  xác suất TẮT nơ-ron q (nn.Dropout dùng p chính là xác suất tắt); 0.0 = không dùng
        init:     "zeros" | "normal" | "xavier" | "he" | "default"
    """

    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        hidden = tuple(int(h) for h in hidden)
        if not hidden or any(h <= 0 for h in hidden):
            raise ValueError("hidden phải là tuple không rỗng gồm các số dương")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout phải nằm trong [0, 1)")
        layers: list[nn.Module] = []
        n_in = in_features
        for h in hidden:
            layers.extend([nn.Linear(n_in, h), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            n_in = h
        layers.append(nn.Linear(n_in, num_classes))
        self.net = nn.Sequential(*layers)
        self.hidden = hidden
        self.dropout = float(dropout)
        init_weights(self, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 54) float32  ->  logits: (B, 7) float32."""
        if x.ndim != 2 or x.shape[1] != self.net[0].in_features:
            raise ValueError(f"Đầu vào phải có shape (B, {self.net[0].in_features}), nhận {tuple(x.shape)}")
        return self.net(x)


def init_weights(model: nn.Module, init: str) -> None:
    """Khởi tạo tham số của MỌI nn.Linear (bias luôn = 0).

    init:
        "zeros"   : W = 0
        "normal"  : W ~ N(0, 0.01^2)
        "xavier"  : nn.init.xavier_normal_ (Var = 2/(n_in+n_out)); nếu bạn dùng Var = 1/n_in theo slide, hãy ghi rõ
        "he"      : nn.init.kaiming_normal_(w, nonlinearity="relu")  (Var = 2/n_in)
        "default" : không làm gì (giữ khởi tạo mặc định của nn.Linear; KHÔNG phải He)
    Gợi ý: duyệt model.modules(), chọn isinstance(m, nn.Linear).
    """
    valid = {"zeros", "normal", "xavier", "he", "default"}
    if init not in valid:
        raise ValueError(f"init={init!r} không hợp lệ; chọn một trong {sorted(valid)}")
    if init == "default":
        return
    with torch.no_grad():
        for layer in model.modules():
            if not isinstance(layer, nn.Linear):
                continue
            if init == "zeros":
                nn.init.zeros_(layer.weight)
            elif init == "normal":
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            elif init == "xavier":
                nn.init.xavier_normal_(layer.weight)
            elif init == "he":
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)


def count_params(model: nn.Module) -> int:
    """Tổng số tham số huấn luyện được. Dùng để assert với EXPECTED_PARAMS ngay sau khi tạo model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Độ lệch chuẩn của kích hoạt sau mỗi lớp (ở bước 0, một lô val) — dùng cho thí nghiệm khởi tạo.

    Các bước:
      1. model.eval(); h = x
      2. duyệt từng lớp con theo thứ tự; sau mỗi nn.Linear (hoặc sau mỗi ReLU, bạn chọn và ghi rõ) lưu h.std().item()
      3. trả về danh sách std theo lớp
    """
    was_training = model.training
    model.eval()
    h = x
    stats: list[float] = []
    for layer in model.net:
        h = layer(h)
        if isinstance(layer, nn.ReLU):
            stats.append(float(h.float().std(unbiased=False).item()))
    # Include logits so all affine stages can be diagnosed.
    stats.append(float(h.float().std(unbiased=False).item()))
    model.train(was_training)
    return stats
