"""Lưu result JSON và xây bảng experiments.xlsx từ template.

Nhiệm vụ: lưu kết quả từng lần chạy ra JSON, rồi điền vào experiments.xlsx từ mẫu
templates/experiment_table_template.xlsx (đừng gõ tay hàng chục dòng, rất dễ sai).

Tên cột của sheet "Experiments" (giữ nguyên, đúng thứ tự mẫu):
    exp_id, group, description, loss, optimizer, lr, weight_decay, batch, epochs, hidden, dropout,
    clip_norm, precision, init, seed, step0_loss, best_val_loss, best_epoch, final_train_loss,
    final_val_loss, val_acc, val_macro_f1, time_per_epoch_s, peak_mem_MB, diverged,
    eval_acc, eval_macro_f1, figure_file, notes
(các cột công thức ở cuối bảng mẫu tự tính, đừng ghi đè)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import openpyxl
from copy import copy


def save_result(result: dict, results_dir: str = "../results") -> str:
    """Ghi result["cfg"], result["history"], result["summary"] (KHÔNG ghi best_state) ra
    <results_dir>/<exp_id>.json. Trả về đường dẫn file. Tạo thư mục nếu chưa có."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_id = result["cfg"]["exp_id"]
    payload = {k: result[k] for k in (
        "schema_version", "fingerprint", "cfg", "history", "summary"
    ) if k in result}
    out = out_dir / f"{exp_id}.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    return str(out)


def load_results(results_dir: str = "../results") -> list[dict]:
    """Đọc mọi file *.json trong results_dir, trả về danh sách dict (sắp theo exp_id)."""
    root = Path(results_dir)
    if not root.exists():
        return []
    results = []
    for path in sorted(root.glob("*.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON lỗi: {path}") from exc
        if not all(k in result for k in ("cfg", "history", "summary")):
            raise ValueError(f"Result thiếu cfg/history/summary: {path}")
        results.append(result)
    return sorted(results, key=lambda r: r["cfg"]["exp_id"])


def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    """Biến một kết quả thành một dòng của bảng: gộp cfg + summary (+ eval_acc, eval_macro_f1 nếu có)
    + figure_file = f"figures/{exp_id}.png". Khoá phải trùng tên cột ở đầu file.
    Chỉ truyền eval_scores cho baseline và cấu hình cuối cùng."""
    cfg, summary = result["cfg"], result["summary"]
    row = {**cfg, **summary}
    row["hidden"] = "-".join(str(x) for x in cfg["hidden"])
    row["eval_acc"] = None if eval_scores is None else eval_scores.get("accuracy")
    row["eval_macro_f1"] = None if eval_scores is None else eval_scores.get("macro_f1")
    row["figure_file"] = f"figures/{cfg['exp_id']}.png"
    extra = []
    if cfg.get("class_weight", "none") != "none":
        extra.append(f"class_weight={cfg['class_weight']}")
    if cfg.get("scheduler"):
        extra.append(f"scheduler={cfg['scheduler']}")
    if summary.get("grad_norm_p95") is not None:
        extra.append(f"grad_p95={summary['grad_norm_p95']:.4g}")
    row["notes"] = "; ".join(x for x in [notes, *extra] if x)
    return row


def write_xlsx(rows: list[dict], template_path: str, out_path: str,
               seed_ids: list[str] | None = None,
               candidate_seed_ids: list[str] | None = None,
               summary_notes: dict[str, str] | None = None) -> None:
    """Điền các dòng vào sheet "Experiments" của mẫu, từ dòng 2 trở xuống, rồi lưu thành out_path.

    Các bước (openpyxl):
      1. wb = openpyxl.load_workbook(template_path)   # KHÔNG dùng data_only=True (sẽ mất công thức)
      2. ws = wb["Experiments"]; đọc tiêu đề dòng 1 để biết cột nào ứng với khoá nào
      3. với mỗi row: ghi giá trị vào đúng cột; BỎ QUA các cột công thức (step0_gap_vs_lnC, gap_val_minus_train,
         delta_val_f1_vs_base, beyond_noise)
      4. wb.save(out_path)
    Sau khi lưu, mở file bằng Excel/LibreOffice để các công thức tính lại.
    """
    template, out = Path(template_path), Path(out_path)
    if not template.exists():
        raise FileNotFoundError(template)
    wb = openpyxl.load_workbook(template, data_only=False)
    required_sheets = {"Legend", "Experiments", "Seeds", "Summary"}
    if not required_sheets.issubset(wb.sheetnames):
        raise ValueError(f"Template thiếu sheet: {required_sheets - set(wb.sheetnames)}")
    ws = wb["Experiments"]
    headers = [c.value for c in ws[1]]
    formula_columns = {
        "step0_gap_vs_lnC", "gap_val_minus_train", "delta_val_f1_vs_base", "beyond_noise"
    }
    # Preserve the template formulas/styles and extend them to every populated row.
    formula_source = {}
    for i, header in enumerate(headers, start=1):
        if header in formula_columns:
            src = ws.cell(2, i)
            formula_source[header] = (src.value, copy(src._style))
    if ws.max_row > 1:
        ws.delete_rows(2, ws.max_row - 1)
    for row_idx, row in enumerate(rows, start=2):
        for col_idx, header in enumerate(headers, start=1):
            cell = ws.cell(row_idx, col_idx)
            if header in formula_columns:
                src = formula_source.get(header)
                if src is not None and isinstance(src[0], str) and src[0].startswith("="):
                    from openpyxl.formula.translate import Translator
                    origin = f"{cell.column_letter}2"
                    cell.value = Translator(src[0], origin=origin).translate_formula(cell.coordinate)
                    cell._style = copy(src[1])
                continue
            value = row.get(header)
            if isinstance(value, (tuple, list, dict)):
                value = json.dumps(value, ensure_ascii=False)
            cell.value = value

    if seed_ids is not None:
        seeds = wb["Seeds"]
        for row_idx in range(2, 7):
            seeds.cell(row_idx, 1).value = seed_ids[row_idx - 2] if row_idx - 2 < len(seed_ids) else None
        if candidate_seed_ids:
            seeds.cell(1, 6).value = "final candidate exp_id (không tính vào σ baseline)"
            for offset, exp_id in enumerate(candidate_seed_ids, start=2):
                seeds.cell(offset, 6).value = exp_id
    if summary_notes:
        summary = wb["Summary"]
        for row_idx in range(2, summary.max_row + 1):
            group = summary.cell(row_idx, 1).value
            if group in summary_notes:
                summary.cell(row_idx, 8).value = summary_notes[group]

    # Ensure formulas are recalculated by Excel/LibreOffice on first open.
    try:
        wb.calculation.fullCalcOnLoad = True
        wb.calculation.forceFullCalc = True
        wb.calculation.calcMode = "auto"
    except AttributeError:
        pass
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
