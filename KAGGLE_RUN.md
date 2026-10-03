# Chạy bài lab trên Kaggle

## Chuẩn bị

1. Upload `kaggle_bundle_2A202602760.zip` thành một Kaggle Dataset riêng tư. Notebook hỗ trợ cả trường hợp Kaggle giữ nguyên ZIP lẫn tự giải nén ZIP.
2. Trên Kaggle chọn **Create → New Notebook → File → Import Notebook**, upload `code/lab.ipynb`.
3. Chọn **Add Input** và attach Dataset vừa tạo.
4. Trong **Notebook options**, đặt Accelerator thành **GPU T4 x2** hoặc **GPU P100**. Notebook chỉ dùng một GPU.

## Chạy

- Chạy cell từ trên xuống theo Stage 0–6.
- Sau mỗi stage, tải `artifacts_stage*.zip` trong `/kaggle/working` hoặc bấm **Save Version**.
- Nếu runtime mới, attach archive stage mới nhất cùng Dataset repo; Stage 0 tự khôi phục result, figure và checkpoint.
- Không chạy Stage 5 trước khi Stage 4 đã khóa `selection_state.json`.
- Không thay cấu hình sau khi đã nhìn thấy eval.

## Nộp bằng đường dẫn GitHub

Kết quả Kaggle đã được đưa vào thư mục `submission_2A202602760/` của repo: notebook có output, báo cáo, bảng Excel, dự đoán eval, JSON và biểu đồ. Nộp đường dẫn tới thư mục này trên GitHub; không cần upload ZIP. Không commit `artifacts_stage*.zip`, `runtime_checkpoints/`, dữ liệu xử lý hoặc file trọng số.

Để kiểm tra trước khi nộp, chạy `python code/validate_submission.py submission_2A202602760` và `python scripts/evaluate.py --pred submission_2A202602760/predictions_eval.csv`; hai kết quả phải hợp lệ và khớp `submission_2A202602760/eval_result.json`.

Nếu một cell lỗi, giữ nguyên output và gửi traceback để sửa; không xoá toàn bộ kết quả đã hoàn tất.
