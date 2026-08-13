import cv2
import numpy as np

from .services.detector import Detector
from .services.image_preprocessing import ImagePreprocessing
from .services.recognitor import Recognitor

# Mục đích: cung cấp dịch vụ OCR kết hợp tiền xử lý, detection và recognition.
# Chức năng: phát hiện vùng văn bản qua detect_text() và nhận dạng từng vùng qua recognize_text().
# Luồng xử lý: ảnh đầu vào → ImagePreprocessing → Detector → cắt dt_polys → Recognitor.
# Cấu hình: nhận một dict config và truyền cùng cấu hình đó cho ba thành phần bên dưới.
# Quan hệ: được gọi từ tầng sử dụng OCR; phụ thuộc ImagePreprocessing, Detector và Recognitor.
# Input/output: nhận đường dẫn ảnh; detect_text() trả kết quả PaddleOCR, recognize_text() trả danh sách box/text.
# Contract: giữ tên hai phương thức, khóa dt_polys của kết quả detection và dạng kết quả box/text.


class OCRService:
    """Dịch vụ OCR kết hợp phát hiện và nhận dạng văn bản."""

    def __init__(self, config):
        self.config = config
        self.image_preprocessing = ImagePreprocessing(config)
        self.detector = Detector(config)
        self.recognitor = Recognitor(config)

    def detect_text(self, image):
        # Tiền xử lý ảnh trước khi phát hiện vùng văn bản
        processed_image = self.image_preprocessing.preprocess(image)
        results = self.detector.detect(processed_image)
        return results

    def recognize_text(self, image):
        # Tiền xử lý ảnh trước khi detection và recognition
        processed_image = self.image_preprocessing.preprocess(image)
        source_image = processed_image
        detection_results = self.detector.detect(processed_image)
        results = []

        for result in detection_results:
            for box in result["dt_polys"]:
                box = np.array(box, dtype=np.float32)

                # Cắt vùng chữ theo hình chữ nhật bao quanh polygon detection
                x_min, y_min = box.min(axis=0).astype(int)
                x_max, y_max = box.max(axis=0).astype(int)
                crop = source_image[y_min:y_max, x_min:x_max]

                # Xoay crop theo hướng dòng chữ nếu được bật
                crop = self.image_preprocessing.preprocess_textline(crop)
                text = self.recognitor.recognize(crop)
                results.append({"box": box.tolist(), "text": text})

        return results
