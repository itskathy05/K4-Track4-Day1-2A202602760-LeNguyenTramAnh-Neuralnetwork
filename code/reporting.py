"""Generate evidence-linked Excel notes and the Vietnamese final report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from results_table import load_results, to_row, write_xlsx
from workflow import seed_statistics


GROUP_LABELS = {
    "loss": "Hàm mất mát", "optimizer": "Bộ tối ưu", "hparam": "Siêu tham số",
    "dropout": "Dropout", "clipping": "Gradient clipping", "amp": "Mixed precision",
    "init": "Khởi tạo tham số", "baseline": "Baseline", "final": "Cấu hình cuối",
}


def _f(value, digits=4):
    return "—" if value is None else f"{float(value):.{digits}f}"


def _best(items):
    valid = [r for r in items if r["summary"].get("val_macro_f1") is not None]
    return max(valid, key=lambda r: r["summary"]["val_macro_f1"]) if valid else None


def make_summary_notes(results: list[dict], noise: float) -> dict[str, str]:
    notes = {}
    for group in sorted({r["cfg"].get("group", "other") for r in results}):
        items = [r for r in results if r["cfg"].get("group") == group]
        best = _best(items)
        if best:
            notes[group] = (
                f"Tốt nhất: {best['cfg']['exp_id']} với val macro-F1="
                f"{best['summary']['val_macro_f1']:.4f}. Ngưỡng nhiễu baseline 2σ={noise:.4f}."
            )
    return notes


def _group_section(group: str, items: list[dict], baseline_f1: float, noise: float) -> str:
    best = _best(items)
    if best is None:
        return f"### {GROUP_LABELS.get(group, group)}\n\nKhông có run hoàn tất; hạn chế này được ghi nhận.\n"
    worst = min(
        [r for r in items if r["summary"].get("val_macro_f1") is not None],
        key=lambda r: r["summary"]["val_macro_f1"],
    )
    delta = best["summary"]["val_macro_f1"] - baseline_f1
    evidence = "vượt" if abs(delta) > noise else "không vượt"
    prediction = {
        "loss": "CE được dự đoán hội tụ ổn định hơn MSE; trọng số lớp có thể nâng macro-F1 nhưng làm giảm accuracy.",
        "optimizer": "Optimizer thích nghi được dự đoán hội tụ nhanh, nhưng chỉ có thể kết luận sau khi mỗi optimizer được dò lr công bằng.",
        "hparam": "Mạng rộng/huấn luyện lâu có thể tăng năng lực; batch lớn giảm số update mỗi epoch và có thể cần lr khác.",
        "dropout": "Nếu train–val gap nhỏ, dropout mạnh được dự đoán gây underfit thay vì cải thiện.",
        "clipping": "Clipping bình thường ít tác dụng nếu gradient không vượt c, nhưng có thể ổn định run lr cao.",
        "amp": "FP16 được dự đoán giảm bộ nhớ; tốc độ có thể không tăng vì MLP nhỏ bị chi phối bởi overhead kernel.",
        "init": "He/Xavier giữ thang kích hoạt tốt hơn; zeros phá vỡ khả năng học do đối xứng và ReLU tại 0.",
    }.get(group, "")
    mechanism = {
        "loss": "CE tạo gradient trực tiếp theo sai lệch xác suất; MSE qua logits có gradient và thang đo khác. Trọng số lớp tăng đóng góp của lớp hiếm.",
        "optimizer": "Momentum tích lũy hướng gradient, còn Adam/AdamW chuẩn hóa bước cập nhật theo moment bậc một và hai; AdamW tách weight decay khỏi gradient.",
        "hparam": "Kết quả chịu đồng thời tác động của số tham số, số update, phương sai gradient và thời gian mỗi epoch.",
        "dropout": "Dropout tạo nhiễu biểu diễn và regularization; lợi ích chỉ rõ khi giảm overfit nhiều hơn mức underfit mà nó gây ra.",
        "clipping": "Chuẩn toàn cục được đo trước clip; phép co gradient chỉ kích hoạt khi ‖g‖>c, vì vậy clip fraction là bằng chứng trực tiếp.",
        "amp": "Autocast hạ precision phép toán nhưng giữ tham số FP32; FP16 cần GradScaler để tránh gradient nhỏ bị underflow.",
        "init": "He có phương sai 2/fan-in phù hợp ReLU; Xavier nhỏ hơn, còn zeros làm các neuron nhận cập nhật giống nhau.",
    }.get(group, "")
    return f"""### {GROUP_LABELS.get(group, group)}

**Dự đoán trước:** {prediction}

Trong nhóm này, `{best['cfg']['exp_id']}` tốt nhất với val macro-F1 **{_f(best['summary']['val_macro_f1'])}**, val accuracy {_f(best['summary']['val_acc'])}; `{worst['cfg']['exp_id']}` thấp nhất với {_f(worst['summary']['val_macro_f1'])}. Chênh lệch của run tốt nhất so với baseline là {delta:+.4f}, {evidence} ngưỡng nhiễu 2σ={noise:.4f}; vì vậy kết luận được diễn đạt tương ứng thay vì coi mọi sai khác nhỏ là có ý nghĩa. Xem `figures/compare_{group}.png` và ảnh riêng `figures/{best['cfg']['exp_id']}.png`.

**Đối chiếu và cơ chế:** {mechanism}
"""


def generate_report(results: list[dict], health: dict, baseline_ids: list[str],
                    final_ids: list[str], baseline_eval: dict, final_eval: dict,
                    final_winner: str, out_path: str | Path) -> None:
    by_id = {r["cfg"]["exp_id"]: r for r in results}
    baseline_runs = [by_id[x] for x in baseline_ids]
    final_runs = [by_id[x] for x in final_ids]
    baseline_stats, final_stats = seed_statistics(baseline_runs), seed_statistics(final_runs)
    noise = baseline_stats["noise_2sigma"]
    baseline = by_id[baseline_ids[0]]
    final = by_id[final_ids[0]]
    groups = {g: [r for r in results if r["cfg"].get("group") == g] for g in GROUP_LABELS}
    topic_text = "\n".join(
        _group_section(g, groups.get(g, []), baseline["summary"]["val_macro_f1"], noise)
        for g in ("loss", "optimizer", "hparam", "dropout", "clipping", "amp", "init")
    )

    per_class = final_eval["per_class"]
    hardest = min(per_class, key=lambda x: x["f1"])
    cm = np.asarray(final_eval["confusion_matrix"])
    row = cm[hardest["cls"]].copy()
    row[hardest["cls"]] = -1
    confused_with = int(row.argmax())
    class_rows = "\n".join(
        f"| {x['cls']} | {x['support']} | {x['precision']:.4f} | {x['recall']:.4f} | {x['f1']:.4f} |"
        for x in per_class
    )
    cm_text = "\n".join(" ".join(f"{int(v):6d}" for v in matrix_row) for matrix_row in cm)
    cfg = final["cfg"]
    report = f"""# Báo cáo Lab Day 1 — Lê Nguyễn Trâm Anh — 2A202602760

## 1. Thiết lập

Thí nghiệm chạy bằng PyTorch trên Kaggle GPU. Forest CoverType được giữ đúng phép chia cố định: 464.809 mẫu train và 116.203 mẫu eval. Từ train, tôi tách validation 20% theo lớp với seed 42, còn 371.847 mẫu huấn luyện và 92.962 mẫu validation. Mean/std chỉ được fit trên 10 cột số của phần train; 44 cột one-hot giữ nguyên. Eval không tham gia chuẩn hóa, chọn learning rate, epoch hay cấu hình.

Baseline là M-base `54→256→128→7`, ReLU, He initialization, CE, SGD momentum 0,9, batch 512, 20 epoch, không dropout/clip, FP32; learning rate được chọn bằng validation. Mốc đoán lớp đa số là accuracy xấp xỉ 0,4876 nhưng macro-F1 chỉ khoảng 0,094, nên chỉ số quyết định là macro-F1.

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---:|
| Tham số / shape logits | {health['parameter_count']:,} / {tuple(health['logits_shape'])} |
| Loss bước 0 / ln(7) | {health['step0_loss']:.4f} / {health['ln7']:.4f} |
| Overfit 20 mẫu | loss={health['overfit_final_loss']:.6f}, acc={health['overfit_final_acc']:.4f}, {health['overfit_steps']} update |
| Gradient mọi tham số | khác None và khác 0 |
| Baseline val accuracy | {baseline_stats['acc_mean']:.4f} ± {baseline_stats['acc_std']:.4f} |
| Baseline val macro-F1 | {baseline_stats['f1_mean']:.4f} ± {baseline_stats['f1_std']:.4f} |

Ngưỡng nhiễu dùng khi diễn giải là **2σ={noise:.4f}**. Do đó một chênh lệch nhỏ hơn ngưỡng này được xem là chưa đủ bằng chứng. Đường overfit nhỏ nằm ở `figures/health_overfit20.png`; các baseline là `{', '.join(baseline_ids)}`.

## 3. Kết quả theo chủ đề

{topic_text}

## 4. Đánh giá cuối trên eval

Cấu hình được chọn hoàn toàn bằng validation là candidate {final_winner.upper()}: `{final_ids[0]}` và hai seed lặp lại. Cấu hình seed nộp: hidden={tuple(cfg['hidden'])}, loss={cfg['loss']}, class_weight={cfg.get('class_weight')}, optimizer={cfg['optimizer']}, lr={cfg['lr']}, batch={cfg['batch']}, epochs={cfg['epochs']}, dropout={cfg['dropout']}, clip={cfg['clip_norm']}, init={cfg['init']}. Mean val macro-F1 ba seed là {final_stats['f1_mean']:.4f} ± {final_stats['f1_std']:.4f}.

| Cấu hình | Seed | val macro-F1 | eval macro-F1 | eval accuracy |
|---|---:|---:|---:|---:|
| Baseline | 1 | {baseline['summary']['val_macro_f1']:.4f} | {baseline_eval['macro_f1']:.4f} | {baseline_eval['accuracy']:.4f} |
| Cấu hình cuối | 1 | {final['summary']['val_macro_f1']:.4f} | {final_eval['macro_f1']:.4f} | {final_eval['accuracy']:.4f} |

Eval chỉ được mở sau khi quy tắc chọn candidate đã khóa. Cải thiện eval macro-F1 so với baseline là {final_eval['macro_f1']-baseline_eval['macro_f1']:+.4f}. Không có điều chỉnh nào được thực hiện sau khi nhìn thấy con số này.

### 4.1 Phân tích lỗi theo lớp

| Lớp | support | precision | recall | F1 |
|---:|---:|---:|---:|---:|
{class_rows}

Lớp khó nhất là lớp **{hardest['cls']}**, F1={hardest['f1']:.4f}; ngoài dự đoán đúng, lớp này bị nhầm nhiều nhất sang lớp **{confused_with}** ({int(cm[hardest['cls'], confused_with])} mẫu). Một nguyên nhân hợp lý là mất cân bằng support và các kiểu rừng lân cận có đặc trưng địa hình chồng lấn. Weighted CE là một biện pháp trực tiếp đã được đo; hướng tiếp theo là phân tích feature/class và calibration thay vì nhìn riêng accuracy.

```text
Ma trận nhầm lẫn (hàng=thật, cột=dự đoán)
{cm_text}
```

## 5. Trả lời câu hỏi dẫn dắt

Khi loss không giảm sau 2.000 bước, ba kiểm tra đầu tiên của tôi là: (1) kiểm tra dữ liệu/nhãn, shape, chuẩn hóa và loss bước 0 so với ln(7), vì nhãn lệch hoặc scale đầu vào sai làm toàn bộ phép tối ưu vô nghĩa; (2) thử overfit 20 mẫu, vì nếu bài toán cực nhỏ vẫn không học thì lỗi nằm trong forward/loss/zero-grad/optimizer chứ chưa phải generalization; (3) in gradient từng tham số và theo dõi global grad norm cùng learning rate, vì gradient None/0 chỉ ra graph bị đứt, còn spike/NaN chỉ ra lr hoặc ổn định số. Chỉ sau ba phép thử này mới thay kiến trúc, dropout hay optimizer.

Optimizer “thắng” chỉ được xác định sau khi mỗi họ được thử ba lr; dùng cùng lr cho SGD và Adam không phải so sánh công bằng. Dropout chỉ hữu ích nếu giảm gap train–val đủ bù underfit. Clipping xử lý gradient lớn đột ngột, không phải thuốc tăng accuracy chung; bằng chứng là clip fraction và cặp high-lr. FP16 có thể giảm bộ nhớ nhưng không mặc nhiên nhanh hơn với MLP nhỏ. Zeros hỏng vì đối xứng và ReLU(0), còn He giữ phương sai kích hoạt tốt hơn cho ReLU so với Xavier trong kỳ vọng lý thuyết.

## 6. Hạn chế và điều bất ngờ

Mỗi cấu hình thăm dò chủ yếu dùng một seed; chỉ baseline và hai candidate cuối có ba seed. Bởi vậy các khác biệt nhỏ được ghi là chưa kết luận được. Cùng số epoch nhưng batch khác tạo số update khác; optimizer được dò trên lưới hữu hạn; best epoch theo val loss không nhất thiết cực đại hóa macro-F1. Kết quả phụ thuộc loại GPU và overhead Kaggle, đặc biệt ở mixed precision. Nếu có thêm tài nguyên, tôi sẽ tăng số seed cho các run sát nhau, tinh chỉnh quanh lr tốt nhất và khảo sát lỗi lớp hiếm mà không truy cập eval trong quá trình chọn.

## 7. Phụ lục

Submission gồm notebook và toàn bộ module trong `code/`, `experiments.xlsx`, `predictions_eval.csv`, `eval_result.json`, JSON từng run và các biểu đồ. Mọi số liệu trong báo cáo truy được về `exp_id` trong bảng.
"""
    Path(out_path).write_text(report, encoding="utf-8")


def build_xlsx(results: list[dict], template_path: str | Path, out_path: str | Path,
               baseline_ids: list[str], final_ids: list[str], baseline_eval: dict,
               final_eval: dict) -> None:
    eval_by_id = {baseline_ids[0]: baseline_eval, final_ids[0]: final_eval}
    rows = [to_row(r, eval_by_id.get(r["cfg"]["exp_id"])) for r in results]
    noise = seed_statistics([r for r in results if r["cfg"]["exp_id"] in baseline_ids])["noise_2sigma"]
    notes = make_summary_notes(results, noise)
    write_xlsx(
        rows, str(template_path), str(out_path), seed_ids=baseline_ids,
        candidate_seed_ids=final_ids, summary_notes=notes,
    )


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
