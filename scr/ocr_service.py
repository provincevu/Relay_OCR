from .services.detector import Detector


class OCRService:
    """Dịch vụ OCR sử dụng detector để phát hiện vùng văn bản."""

    def __init__(self, config):
        self.config = config
        self.detector = Detector(config)

    def detect_text(self, image):
        # Thực hiện phát hiện vùng văn bản
        results = self.detector.detect(image)
        return results
