import cv2
import numpy as np

from .services.detector import Detector
from .services.recognitor import Recognitor


class OCRService:
    """Dịch vụ OCR kết hợp phát hiện và nhận dạng văn bản."""

    def __init__(self, config):
        self.config = config
        self.detector = Detector(config)
        self.recognitor = Recognitor(config)

    def detect_text(self, image):
        # Thực hiện phát hiện vùng văn bản
        results = self.detector.detect(image)
        return results

    def recognize_text(self, image):
        # Đọc ảnh để cắt các vùng văn bản được phát hiện
        source_image = cv2.imread(str(image))
        detection_results = self.detector.detect(image)
        results = []

        for result in detection_results:
            for box in result["dt_polys"]:
                box = np.array(box, dtype=np.float32)

                # Cắt vùng chữ theo hình chữ nhật bao quanh polygon detection
                x_min, y_min = box.min(axis=0).astype(int)
                x_max, y_max = box.max(axis=0).astype(int)
                crop = source_image[y_min:y_max, x_min:x_max]

                text = self.recognitor.recognize(crop)
                results.append({"box": box.tolist(), "text": text})

        return results
