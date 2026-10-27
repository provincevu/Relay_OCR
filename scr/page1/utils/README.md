# Utility cho TABLE 01

- `table_cell_utils.py`: nhận dạng text, sắp xếp bbox và gán text vào cell.
- `table_field_extractor.py`: khớp tên field và chuẩn hóa dữ liệu TABLE 01.

`Table1Analysis.analyze(image)` nhận ảnh `numpy` BGR đã tiền xử lý và trả:

- `table`: bbox và score của TABLE 01 trên ảnh gốc.
- `cells`: danh sách cell, bbox cục bộ trong ảnh crop, score và text.
- `unassigned_texts`: các vùng text không thuộc cell nào.
- `data`: đầy đủ key khai báo trong `config/table_01_fields.json`.
- `field_details`: thông tin nguồn của từng field đã khớp.
- `review`: các field chưa xác định được chắc chắn.
