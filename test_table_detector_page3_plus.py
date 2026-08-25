# Mục đích: debug nhận diện cấu trúc bảng trên một ảnh page 3+.
# Chức năng: xử lý ảnh bằng layout hoặc vùng content, chọn SLANeXt và vẽ kết quả.
# Luồng xử lý: ảnh page → table crop → classification → structure → ảnh debug.
# Cấu hình/parameter: nhận ảnh, --all-page-001 và tùy chọn --content-region.
# Quan hệ: script độc lập, không sử dụng pipeline hoặc module phân tích page 1.
# Input/output: đọc ảnh page 3+; ghi ảnh debug và HTML structure vào output.
# Contract: ảnh hiển thị loại bảng; content mode lưu HTML structure thực tế.

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
        self.layout_detector = None
        self.table_classifier = TableClassification(
            model_name="PP-LCNet_x1_0_table_cls",
            device="cpu",
            engine="paddle_static",
            enable_mkldnn=False,
            cpu_threads=6,
        )
        self.wired_structure_model = None
        self.wireless_structure_model = None

    def analyze(self, image, use_content_region=False):
        """Trả ảnh đã vẽ kết quả phân loại và nhận diện cấu trúc bảng."""
        visualization = image.copy()
        if use_content_region:
            image_height, image_width = image.shape[:2]
            margin_x = int(image_width * 0.05)
            margin_y = int(image_height * 0.05)
            table_boxes = [
                {
                    "coordinate": [
                        margin_x,
                        margin_y,
                        image_width - margin_x,
                        image_height - margin_y,
                    ],
                }
            ]
        else:
            if self.layout_detector is None:
                self.layout_detector = LayoutDetection(
                    model_name="PicoDet_layout_1x_table",
                    device="cpu",
                    engine="paddle_static",
                    enable_mkldnn=False,
                    cpu_threads=6,
                )
            layout_results = self.layout_detector.predict(
                image,
                batch_size=1,
                threshold=0.3,
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
        structure_outputs = []

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
            structure_outputs.append(
                {
                    "table_index": table_index,
                    "table_type": table_type,
                    "classification_score": classification_score,
                    "model_name": model_name,
                    "structure_score": structure_score,
                    "html": "".join(structure_result["structure"]),
                }
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

        return visualization, structure_outputs


def parse_arguments():
    """Đọc chế độ và đường dẫn ảnh page cần phân tích."""
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", nargs="?", type=Path)
    parser.add_argument("--all-page-001", action="store_true")
    parser.add_argument("--content-region", action="store_true")
    arguments = parser.parse_args()

    if arguments.all_page_001 == (arguments.image_path is not None):
        parser.error(
            "Phải truyền image_path hoặc --all-page-001."
        )

    return arguments


def main():
    """Chạy phân tích ảnh page và lưu ảnh trực quan."""
    arguments = parse_arguments()
    if arguments.all_page_001:
        input_directory = Path("./data/pdf/page3plus_png")
        image_paths = sorted(
            input_directory.glob("P_*/page_001.png"),
            key=lambda path: path.parent.name,
        )
        if not image_paths:
            raise FileNotFoundError(
                f"Không tìm thấy ảnh page_001 trong: {input_directory}"
            )
        if arguments.content_region:
            output_directory = Path(
                "./output/table_3plus_debug/page_001_content"
            )
        else:
            output_directory = Path(
                "./output/table_3plus_debug/page_001"
            )
    else:
        image_paths = [arguments.image_path]
        output_directory = (
            Path("./output/table_3plus_debug")
            / arguments.image_path.parent.name
        )
        if arguments.content_region:
            output_directory /= "content"

    output_directory.mkdir(parents=True, exist_ok=True)
    analyzer = Page3PlusTableAnalyzer()

    for image_index, image_path in enumerate(image_paths, start=1):
        print(
            f"[{image_index}/{len(image_paths)}] Đang xử lý: "
            f"{image_path}"
        )
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

        visualization, structure_outputs = analyzer.analyze(
            image,
            use_content_region=arguments.content_region,
        )
        if arguments.all_page_001:
            output_stem = (
                f"{image_path.parent.name}_{image_path.stem}"
            )
        else:
            output_stem = image_path.stem
        output_path = output_directory / f"{output_stem}_analysis.png"
        saved = cv2.imwrite(str(output_path), visualization)
        if not saved:
            raise RuntimeError(f"Không lưu được ảnh: {output_path}")

        print(f"Ảnh trực quan: {output_path}")

        if arguments.content_region:
            for structure_output in structure_outputs:
                metadata_html = f"""
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; margin-top: 20px; }}
td {{ border: 1px solid #111; min-width: 80px; height: 20px; }}
</style>
<h1>{image_path.parent.name} / {image_path.name}</h1>
<p>
Loại bảng: {structure_output['table_type']} |
classification score: {structure_output['classification_score']:.4f} |
model: {structure_output['model_name']} |
structure score: {structure_output['structure_score']:.4f}
</p>
"""
                structure_html = structure_output["html"].replace(
                    "<body>",
                    (
                        '<head><meta charset="utf-8"></head>'
                        f"<body>{metadata_html}"
                    ),
                    1,
                )
                structure_path = output_directory / (
                    f"{output_stem}_structure.html"
                )
                structure_path.write_text(
                    structure_html,
                    encoding="utf-8",
                )
                print(f"HTML structure: {structure_path}")

    print(f"Đã xử lý {len(image_paths)} ảnh.")


if __name__ == "__main__":
    main()
