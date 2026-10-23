# Data

Nhóm này chuẩn bị dữ liệu đầu vào cho các pipeline OCR và Layout.

| Tệp | Trách nhiệm |
|---|---|
| `export_page3.py` | Render page 3 từ PDF theo cache đã có. |
| `export_pages_3plus.py` | Xác định và render các page dữ liệu từ page 3+. |
| `build_page3plus_dataset.py` | Tạo dataset `train`, `val`, `test` từ PDF. |
| `split_page3_dataset.py` | Chia `annotations.json` thành các tập train/validation/test. |
| `split_table_01.py` | Crop TABLE 01 từ ảnh page 1. |
| `group_table_01_sources_by_relay.py` | Nhóm source TABLE 01 theo tên rơ-le. |

Các tệp trong nhóm này đọc PDF, ảnh hoặc JSON đầu vào và ghi dataset, ảnh render hoặc crop.
