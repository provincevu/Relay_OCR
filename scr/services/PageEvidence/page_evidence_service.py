# Mục đích: tạo OCR evidence để xác định ranh giới và loại page PDF.
# Chức năng: preprocessing, detection và recognition trên một ảnh page.
# Luồng xử lý: ảnh page → preprocessing → detection → crop → recognition.
# Cấu hình/parameter: dùng cùng config của Detector, Recognitor và ImagePreprocessing.
# Quan hệ: được InputLoader gọi; không import hoặc gọi OCRService.
# Input/output: nhận đường dẫn ảnh; trả danh sách box và text.
# Contract: recognize() trả item có khóa box và text; recognize_region() giữ contract đó.

import numpy as np

from ..Detector.detector import Detector
from ..ImagePreprocessing.image_preprocessing import ImagePreprocessing
from ..Recognitor.recognitor import Recognitor


class PageEvidenceService:
    """Tạo evidence text từ một ảnh page."""

    def __init__(self, config):
        self.image_preprocessing = ImagePreprocessing(config)
        self.detector = Detector(config)
        self.recognitor = Recognitor(config)

    def recognize(self, image_path):
        """Nhận dạng các vùng text trên một ảnh page."""
        processed_image = self.image_preprocessing.preprocess(image_path)
        detection_results = self.detector.detect(processed_image)
        results = []

        for result in detection_results:
            for box in result["dt_polys"]:
                results.append(self._recognize_box(processed_image, box))

        return results

    def recognize_region(self, image_path, box):
        """Nhận dạng lại một vùng ảnh được chỉ định."""
        processed_image = self.image_preprocessing.preprocess(image_path)
        return self._recognize_box(processed_image, box)

    def _recognize_box(self, image, box):
        """Nhận dạng một polygon trên ảnh đã tiền xử lý."""
        box = np.array(box, dtype=np.float32)
        x_min, y_min = box.min(axis=0).astype(int)
        x_max, y_max = box.max(axis=0).astype(int)
        crop = image[y_min:y_max, x_min:x_max]
        crop = self.image_preprocessing.preprocess_textline(crop)
        return {
            "box": box.tolist(),
            "text": self.recognitor.recognize(crop),
        }
