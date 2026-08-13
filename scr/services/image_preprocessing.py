from pathlib import Path

import cv2
from paddleocr import (
    DocImgOrientationClassification,
    TextImageUnwarping,
    TextLineOrientationClassification,
)


class ImagePreprocessing:
    """Đóng gói các bước tiền xử lý ảnh của PaddleOCR."""

    def __init__(self, config):
        self.config = config
        self.debug = config.get("debug", False)  # Lưu ảnh trung gian khi debug
        self.use_doc_orientation_classify = config.get(
            "use_doc_orientation_classify", True  # Mặc định bật xoay ảnh tài liệu
        )
        self.use_doc_unwarping = config.get(
            "use_doc_unwarping", False  # Mặc định tắt hiệu chỉnh ảnh tài liệu
        )
        self.use_textline_orientation = config.get(
            "use_textline_orientation", False  # Mặc định tắt xoay dòng chữ
        )

        # nếu bật nhận diện hướng tài liệu
        if self.use_doc_orientation_classify:
            self.doc_orientation_classify = DocImgOrientationClassification(
                model_name=config.get(
                    "doc_orientation_model_name", "PP-LCNet_x1_0_doc_ori"
                ),  # Mô hình phân loại hướng tài liệu
                device=config.get("device", "cpu"),  # Thiết bị chạy inference
                engine=config.get("engine", "paddle_static"),  # Engine inference
                cpu_threads=config.get("cpu_threads", 2),  # Số luồng CPU sử dụng
                enable_mkldnn=config.get("enable_mkldnn", False),  # Bật/tắt MKL-DNN
            )

        # nếu bật hiệu chỉnh ảnh bị méo (unwarping) cho ảnh tài liệu
        if self.use_doc_unwarping:
            self.doc_unwarping = TextImageUnwarping(
                model_name=config.get(
                    "doc_unwarping_model_name", "UVDoc"
                ),  # Mô hình hiệu chỉnh ảnh tài liệu
                device=config.get("device", "cpu"),  # Thiết bị chạy inference
                engine=config.get("engine", "paddle_static"),  # Engine inference
                cpu_threads=config.get("cpu_threads", 2),  # Số luồng CPU sử dụng
                enable_mkldnn=config.get("enable_mkldnn", False),  # Bật/tắt MKL-DNN
            )

        # nếu bật nhận diện hướng dòng chữ (khác với hướng tài liệu, cái này sẽ nhận diện hướng của từng dòng chữ riêng lẻ)
        if self.use_textline_orientation:
            self.textline_orientation = TextLineOrientationClassification(
                model_name=config.get(
                    "textline_orientation_model_name", "PP-LCNet_x0_25_textline_ori"
                ),  # Mô hình phân loại hướng dòng chữ
                device=config.get("device", "cpu"),  # Thiết bị chạy inference
                engine=config.get("engine", "paddle_static"),  # Engine inference
                cpu_threads=config.get("cpu_threads", 2),  # Số luồng CPU sử dụng
                enable_mkldnn=config.get("enable_mkldnn", False),  # Bật/tắt MKL-DNN
            )

    def preprocess(self, image_path):
        """Tiền xử lý ảnh tài liệu và trả về ảnh trong bộ nhớ."""
        image = cv2.imread(str(image_path))

        if self.use_doc_orientation_classify:
            result = self.doc_orientation_classify.predict(image, batch_size=1)[0]
            angle = int(result["label_names"][0])
            image = self._rotate_image(image, angle)

        if self.use_doc_unwarping:
            result = self.doc_unwarping.predict(image, batch_size=1)[0]
            image = result["doctr_img"][:, :, ::-1]

        if self.debug:
            output_path = Path("./output/preprocessing/") / Path(image_path).name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), image)

        return image

    def preprocess_textline(self, image):
        """Xoay ảnh dòng chữ theo kết quả phân loại hướng dòng."""
        if self.use_textline_orientation:
            result = self.textline_orientation.predict(image, batch_size=1)[0]
            label = result["label_names"][0]
            angle = 180 if label == "180_degree" else 0  # Nhãn hướng dòng chữ
            image = self._rotate_image(image, angle)

        return image

    def _rotate_image(self, image, angle):
        """Xoay ảnh theo góc mà PaddleOCR trả về."""
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image
