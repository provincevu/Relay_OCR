import cv2
import numpy as np

from .services.Detector.detector import Detector
from .services.ImagePreprocessing.image_preprocessing import ImagePreprocessing
from .page1.table_1_analysis import Table1Analysis
from .services.Recognitor.recognitor import Recognitor

# Mục đích: cung cấp dịch vụ OCR kết hợp tiền xử lý, detection và recognition.
# Chức năng: OCR text thông thường và trích xuất field TABLE 01 trên page 1.
# Luồng xử lý: ảnh đầu vào → preprocessing → OCR text hoặc Table1Analysis.
# Cấu hình: nhận một dict config và truyền cho các service bên dưới.
# Quan hệ: phối hợp ImagePreprocessing, Detector, Recognitor và Table1Analysis.
# Input/output: nhận đường dẫn ảnh; trả text hoặc fields/table trong bộ nhớ.
# Contract: giữ detect_text(), recognize_text() và bổ sung analyze_page1().


class OCRService:
    """Dịch vụ OCR kết hợp phát hiện và nhận dạng văn bản."""

    # Khởi tạo các service và dùng chung model text cho TABLE 01.
    def __init__(self, config):
        self.config = config
        self.image_preprocessing = ImagePreprocessing(config)
        self.detector = Detector(config)
        self.recognitor = Recognitor(config)
        self.table_1_analysis = Table1Analysis(
            config,
            self.detector,
            self.recognitor,
        )

    # Phát hiện các vùng text trên ảnh đã tiền xử lý.
    def detect_text(self, image):
        # Tiền xử lý ảnh trước khi phát hiện vùng văn bản
        processed_image = self.image_preprocessing.preprocess(image)
        results = self.detector.detect(processed_image)
        return results

    # Nhận dạng text trong toàn bộ polygon được phát hiện.
    def recognize_text(self, image):
        # Tiền xử lý ảnh trước khi detection và recognition
        processed_image = self.image_preprocessing.preprocess(image)
        source_image = processed_image
        detection_results = self.detector.detect(processed_image)
        results = []

        # Duyệt các polygon text và nhận dạng riêng từng crop.
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

    # Trích xuất field TABLE 01 từ ảnh page 1.
    def analyze_page1(self, image_path):
        """Trả fields và table của page 1 mà không tạo output file."""
        image = cv2.imread(str(image_path))
        return self.table_1_analysis.analyze(image)
