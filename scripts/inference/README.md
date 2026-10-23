# Inference

Nhóm này chạy inference page 3+ và hậu xử lý prediction.

| Tệp | Trách nhiệm |
|---|---|
| `infer_page_n_text_layout.py` | Chạy OCR và Layout model trên page 3+. |
| `page_n_prediction_filter.py` | Chứa quy tắc lọc region và link prediction. |
| `filter_page_n_predictions.py` | Áp dụng bộ lọc vào `predictions.json`. |
| `role_change_metadata.py` | Ghi mã và mô tả lý do đổi role. |

Đầu ra chính của nhóm này là `predictions.json` sau inference và hậu xử lý.
