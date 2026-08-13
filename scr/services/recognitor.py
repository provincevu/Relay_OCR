import cv2
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

# Mục đích: đóng gói chức năng nhận dạng văn bản bằng VietOCR.
# Chức năng: khởi tạo Predictor với model vgg_transformer và nhận dạng một ảnh crop qua recognize().
# Luồng xử lý: ảnh crop BGR → chuyển sang RGB/PIL → VietOCR Predictor → trả chuỗi văn bản.
# Cấu hình/parameter: recognition_model mặc định vgg_transformer; recognition_device mặc định cpu.
# Quan hệ: được OCRService gọi sau Detector; phụ thuộc OpenCV, Pillow và VietOCR.
# Input/output: nhận ảnh crop numpy BGR; trả về text do VietOCR nhận dạng.
# Contract: giữ tên Recognitor.recognize(), input là ảnh crop BGR và output là chuỗi text.


class Recognitor:
    """Đóng gói chức năng nhận dạng văn bản bằng VietOCR."""

    def __init__(self, config):
        self.config = config
        recognitor_config = Cfg.load_config_from_name(
            config.get("recognition_model", "vgg_transformer")  # Mô hình VietOCR
        )
        recognitor_config["device"] = config.get(
            "recognition_device", "cpu"  # Thiết bị chạy nhận dạng
        )
        self.recognitor = Predictor(recognitor_config)

    def recognize(self, image):
        """Nhận dạng văn bản từ ảnh đã được cắt."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        text = self.recognitor.predict(Image.fromarray(image))
        return text
