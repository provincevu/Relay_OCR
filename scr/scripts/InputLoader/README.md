# InputLoader

`InputLoader` nhận một file PDF chứa nhiều phiếu, không tạo PDF con và ghi ảnh
cùng metadata để các service OCR xử lý tiếp.

## Sử dụng

```python
from scr.scripts.InputLoader import InputLoader

config = {
    "input_loader_output": "output/inference/splitted_pdf",
    "input_loader_render_dpi": 200,
}

loader = InputLoader(config)
metadata = loader.load("data/input/combined.pdf")
```

## Output

```text
output/inference/splitted_pdf/
├── input_metadata.json
├── 0001/
│   ├── 001.png
│   ├── 002.png
│   └── metadata.json
└── 0002/
    ├── 001.png
    └── metadata.json
```

Mỗi page trong `metadata.json` giữ hai chỉ số:

- `source_pdf_page_number`: page trong PDF gốc.
- `document_page_number`: page bên trong phiếu.

`page_type` có một trong các giá trị `PAGE_1`, `PAGE_2_COMMENT`, `PAGE_2_DATA`
hoặc `PAGE_3_PLUS`.

## Cấu hình

- `input_loader_output`: thư mục output, mặc định là `output/inference/splitted_pdf`.
- `input_loader_render_dpi`: DPI render ảnh, mặc định là `200`.
- `input_loader_renderer`: đường dẫn Ghostscript; nếu bỏ qua sẽ tìm `gswin64c`,
  `gswin32c` hoặc `gs` trong `PATH`.
