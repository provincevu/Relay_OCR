from pathlib import Path
from paddleocr import TextDetection

# Mục đích: đóng gói chức năng phát hiện các vùng chứa văn bản bằng PaddleOCR.
# Chức năng: khởi tạo TextDetection và chạy detection cho một ảnh qua detect().
# Luồng xử lý: nhận ảnh → TextDetection.predict() → trả về kết quả detection gốc.
# Cấu hình/parameter: model_name, device, engine, cpu_threads, enable_mkldnn và các ngưỡng detection.
# Quan hệ: được OCRService gọi; phụ thuộc TextDetection của PaddleOCR.
# Input/output: nhận đường dẫn ảnh hoặc ảnh numpy; trả về danh sách kết quả có dt_polys.
# Contract: giữ tên Detector.detect(), các khóa config mặc định và cấu trúc kết quả PaddleOCR.


class Detector:
    """chức năng phát hiện vùng chứa văn bản."""

    def __init__(self, config):
        self.config = config
        self.detector = TextDetection(
            model_name=config.get("model_name", "PP-OCRv6_medium_det"),     # Mô hình phát hiện văn bản
            device=config.get("device", "cpu"),                             # Thiết bị chạy inference
            engine=config.get("engine", "paddle_static"),                   # Engine inference
            cpu_threads=config.get("cpu_threads", 2),                       # Số luồng CPU sử dụng
            enable_mkldnn=config.get("enable_mkldnn", False),               # Bật/tắt MKL-DNN
        )

    def detect(self, image_path):
        """Phát hiện các vùng chứa văn bản trong một ảnh."""
        results = self.detector.predict(
            image_path,
            limit_side_len=self.config.get("limit_side_len", 1600),         # Giới hạn kích thước ảnh
            limit_type=self.config.get("limit_type", "max"),                # Cách áp dụng giới hạn kích thước
            thresh=self.config.get("thresh", 0.3),                          # Ngưỡng xác định điểm ảnh thuộc vùng văn bản
            box_thresh=self.config.get("box_thresh", 0.6),                  # Ngưỡng giữ lại bounding box
            unclip_ratio=self.config.get("unclip_ratio", 2.0),              # Mức mở rộng bounding box
        )
        return results
