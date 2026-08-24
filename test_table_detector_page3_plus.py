# Mục đích: debug nhận diện cấu trúc bảng trên một ảnh page 3+.
# Chức năng: tìm bảng, phân loại wired/wireless, chọn SLANeXt và vẽ kết quả.
# Luồng xử lý: ảnh page → table crop → classification → structure → ảnh debug.
# Cấu hình/parameter: đường dẫn ảnh nhận từ command line; model chạy trên CPU.
# Quan hệ: script độc lập, không sử dụng pipeline hoặc module phân tích page 1.
# Input/output: đọc một ảnh page; ghi ảnh trực quan vào output/table_3plus_debug.
# Contract: mỗi bảng hiển thị loại bảng, classification score và structure score.

import argparse
from pathlib import Path

import cv2
import numpy as np
from paddleocr import (
    LayoutDetection,
    TableClassification,
    TableStructureRecognition,
)


class Page3PlusTableAnalyzer:
    """Phân loại và nhận diện cấu trúc bảng trên một ảnh page 3+."""

    def __init__(self):
        """Khởi tạo các model dùng cho luồng debug."""
        self.layout_detector = LayoutDetection(
            model_name="PicoDet_layout_1x_table",
            device="cpu",
            engine="paddle_static",
            enable_mkldnn=False,
            cpu_threads=6,
        )
        self.table_classifier = TableClassification(
            model_name="PP-LCNet_x1_0_table_cls",
            device="cpu",
            engine="paddle_static",
            enable_mkldnn=False,
            cpu_threads=6,
        )
        self.wired_structure_model = None
        self.wireless_structure_model = None

    def analyze(self, image):
        """Trả ảnh đã vẽ kết quả phân loại và nhận diện cấu trúc bảng."""
        visualization = image.copy()
        layout_results = self.layout_detector.predict(
            image,
            batch_size=1,
            layout_nms=True,
        )
        table_boxes = []

        for layout_result in layout_results:
            table_boxes.extend(
                box
                for box in layout_result["boxes"]
                if box["label"].lower() == "table"
            )

        table_boxes = sorted(
            table_boxes,
            key=lambda box: (
                float(box["coordinate"][1]),
                float(box["coordinate"][0]),
            ),
        )

        for table_index, table_box in enumerate(table_boxes, start=1):
            x_min, y_min, x_max, y_max = map(
                int,
                table_box["coordinate"],
            )
            table_crop = image[y_min:y_max, x_min:x_max]
            classification_result = self.table_classifier.predict(
                table_crop,
                batch_size=1,
            )[0]
            classification_scores = [
                float(score) for score in classification_result["scores"]
            ]
            classification_index = int(np.argmax(classification_scores))
            table_type = classification_result["label_names"][
                classification_index
            ]
            classification_score = classification_scores[
                classification_index
            ]

            if table_type == "wired_table":
                color = (34, 197, 94)
                model_name = "SLANeXt_wired"
                if self.wired_structure_model is None:
                    self.wired_structure_model = TableStructureRecognition(
                        model_name=model_name,
                        device="cpu",
                        engine="paddle_static",
                        enable_mkldnn=False,
                        cpu_threads=6,
                    )
                structure_model = self.wired_structure_model
            elif table_type == "wireless_table":
                color = (0, 165, 255)
                model_name = "SLANeXt_wireless"
                if self.wireless_structure_model is None:
                    self.wireless_structure_model = (
                        TableStructureRecognition(
                            model_name=model_name,
                            device="cpu",
                            engine="paddle_static",
                            enable_mkldnn=False,
                            cpu_threads=6,
                        )
                    )
                structure_model = self.wireless_structure_model
            else:
                raise ValueError(f"Loại bảng không được hỗ trợ: {table_type}")

            structure_result = structure_model.predict(
                input=table_crop,
                batch_size=1,
            )[0]
            structure_score = float(
                structure_result["structure_score"]
            )
            cv2.rectangle(
                visualization,
                (x_min, y_min),
                (x_max, y_max),
                color,
                4,
            )
            first_label = (
                f"TABLE {table_index:02d} | {table_type} | "
                f"cls={classification_score:.4f}"
            )
            second_label = (
                f"{model_name} | structure={structure_score:.4f}"
            )
            first_label_y = max(y_min - 42, 32)
            second_label_y = max(y_min - 10, 64)
            cv2.putText(
                visualization,
                first_label,
                (x_min, first_label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                visualization,
                second_label,
                (x_min, second_label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

        if not table_boxes:
            cv2.putText(
                visualization,
                "NO TABLE DETECTED",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        return visualization


def parse_arguments():
    """Đọc đường dẫn ảnh page cần phân tích."""
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=Path)
    return parser.parse_args()


def main():
    """Chạy phân tích một ảnh page và lưu ảnh trực quan."""
    arguments = parse_arguments()
    image = cv2.imread(str(arguments.image_path))
    if image is None:
        raise FileNotFoundError(
            f"Không đọc được ảnh: {arguments.image_path}"
        )

    analyzer = Page3PlusTableAnalyzer()
    visualization = analyzer.analyze(image)
    output_directory = (
        Path("./output/table_3plus_debug")
        / arguments.image_path.parent.name
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / (
        f"{arguments.image_path.stem}_analysis.png"
    )
    saved = cv2.imwrite(str(output_path), visualization)
    if not saved:
        raise RuntimeError(f"Không lưu được ảnh: {output_path}")

    print(f"Ảnh trực quan: {output_path}")


if __name__ == "__main__":
    main()
