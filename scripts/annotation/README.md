# Annotation

Nhóm này tạo, phục vụ và chuẩn hóa annotation cho page 3+.

| Tệp | Trách nhiệm |
|---|---|
| `annotate_page3.py` | Tạo giao diện HTML annotation từ kết quả OCR. |
| `serve_page3_annotations.py` | Chạy server để xác nhận và lưu annotation. |
| `resolve_page3_headers.py` | Bổ sung header cho quan hệ KEY → VALUE. |
| `migrate_inline_key_value_annotations.py` | Chuyển annotation inline cũ sang schema mới. |

Đầu ra chính của nhóm này là `annotations.json`, được dùng cho huấn luyện Layout model.
