import cv2
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor


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
