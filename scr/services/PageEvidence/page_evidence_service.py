# Mục đích: tạo OCR evidence để xác định ranh giới và loại page PDF.
# Chức năng: preprocessing, detection và recognition trên một ảnh page.
# Luồng xử lý: ảnh page → preprocessing → detection → crop → recognition.
# Cấu hình/parameter: dùng cùng config của Detector, Recognitor và ImagePreprocessing.
# Quan hệ: được InputLoader gọi; không import hoặc gọi OCRService.
# Input/output: nhận đường dẫn ảnh; trả danh sách box và text.
# Contract: mỗi item output có khóa box và text.

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
                box = np.array(box, dtype=np.float32)
                x_min, y_min = box.min(axis=0).astype(int)
                x_max, y_max = box.max(axis=0).astype(int)
                crop = processed_image[y_min:y_max, x_min:x_max]
                crop = self.image_preprocessing.preprocess_textline(crop)
                results.append(
                    {
                        "box": box.tolist(),
                        "text": self.recognitor.recognize(crop),
                    },
                )

        return results
