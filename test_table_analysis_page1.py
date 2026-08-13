# Mục đích: script dự kiến kiểm tra pipeline phân tích bảng trên ảnh trang 1.
# Chức năng dự kiến: xác định bảng/cell, chạy text detection, nhóm text box thành dòng và recognition từng box.
# Luồng xử lý dự kiến: ảnh trang 1 → table/cell detection → text detection → nhóm dòng → VietOCR.
# Cấu hình/parameter: sẽ dùng các cấu hình detection và recognition hiện có khi triển khai.
# Quan hệ dự kiến: có thể gọi OCRService và các tiện ích trong scr/utils/table_analysis_page1.py.
# Input/output dự kiến: nhận ảnh trang 1; xuất kết quả cấu trúc bảng, text từng cell và ảnh debug nếu được bật.
# Contract cần giữ: tọa độ polygon/text box, quan hệ cell-dòng-cột và output recognition phải nhất quán với OCRService.
