# Page 3+

Nhóm này chứa service dùng lại cho inference và hậu xử lý page 3+.

| Tệp | Trách nhiệm |
|---|---|
| `page3_inference.py` | Load checkpoint và suy luận role, link trên page OCR. |
| `text_postprocessing.py` | Lọc prediction và cập nhật role/link theo rule hiện có. |

Các service nhận dữ liệu page đã OCR. Việc đọc PDF, render ảnh và chạy command
line vẫn thuộc `InputLoader` hoặc các entry point trong `scripts/`.

## Sử dụng

```python
from scr.page3 import Page3Inference

service = Page3Inference({
    "page3_checkpoint": "output/checkpoints/page3_layout",
    "page3_device": "cpu",
    "page3_threshold": 0.5,
})

prediction = service.infer(page_record)
```

`page_record` cần có `regions`, `image_width`, `image_height`, cùng metadata
`pdf_name`, `physical_page`, `image_name` nếu muốn giữ thông tin nguồn trong
prediction.
