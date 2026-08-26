# Mục đích: debug nhận diện cấu trúc bảng trên một ảnh page 3+.
# Chức năng: xử lý ảnh, chọn SLANeXt, OCR và xuất wired_table thành HTML.
# Luồng xử lý: ảnh → table crop → classification → structure/OCR → ảnh và HTML.
# Cấu hình/parameter: nhận ảnh, --all-page-001 và tùy chọn --content-region.
# Quan hệ: script độc lập, không sử dụng pipeline hoặc module phân tích page 1.
# Input/output: đọc ảnh page 3+; ghi ảnh debug và một HTML chung vào output.
# Contract: wired_table dùng SLANeXt_wired, PP-OCRv6 và được gom vào một HTML.

import argparse
from html import escape
from pathlib import Path

import cv2
import numpy as np
from paddleocr import (
    LayoutDetection,
    TableClassification,
    TableStructureRecognition,
    TextDetection,
    TextRecognition,
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
        self.text_detector = None
        self.text_recognizer = None

    def _recognize_text(self, image):
        """Nhận diện các vùng text và nội dung bằng PP-OCRv6 medium."""
        if self.text_detector is None:
            self.text_detector = TextDetection(
                model_name="PP-OCRv6_medium_det",
                device="cpu",
                engine="paddle_static",
                enable_mkldnn=False,
                cpu_threads=6,
            )
            self.text_recognizer = TextRecognition(
                model_name="PP-OCRv6_medium_rec",
                device="cpu",
                engine="paddle_static",
                enable_mkldnn=False,
                cpu_threads=6,
            )

        text_regions = []
        image_height, image_width = image.shape[:2]
        detection_results = self.text_detector.predict(
            input=image,
            limit_side_len=1600,
            limit_type="max",
            thresh=0.3,
            box_thresh=0.6,
            unclip_ratio=2.0,
        )

        for detection_result in detection_results:
            for polygon in detection_result["dt_polys"]:
                polygon = np.array(polygon, dtype=np.float32)
                x_min, y_min = polygon.min(axis=0).astype(int)
                x_max, y_max = polygon.max(axis=0).astype(int)
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(x_max, image_width)
                y_max = min(y_max, image_height)
                text_crop = image[y_min:y_max, x_min:x_max]
                recognition_result = self.text_recognizer.predict(
                    input=text_crop,
                    batch_size=1,
                )[0]
                text_regions.append(
                    {
                        "box": [x_min, y_min, x_max, y_max],
                        "center": [
                            float(polygon[:, 0].mean()),
                            float(polygon[:, 1].mean()),
                        ],
                        "text": recognition_result["rec_text"],
                        "score": float(recognition_result["rec_score"]),
                    }
                )

        return text_regions

    def _fill_structure_html(self, structure_html, cell_boxes, text_regions):
        """Khớp text với cell gần nhất và điền vào HTML structure."""
        cell_text_regions = [[] for _ in cell_boxes]
        normalized_cell_boxes = []

        for cell_box in cell_boxes:
            cell_polygon = np.array(
                cell_box,
                dtype=np.float32,
            ).reshape(-1, 2)
            normalized_cell_boxes.append(
                [
                    float(cell_polygon[:, 0].min()),
                    float(cell_polygon[:, 1].min()),
                    float(cell_polygon[:, 0].max()),
                    float(cell_polygon[:, 1].max()),
                ]
            )

        for text_region in text_regions:
            text_box = text_region["box"]
            matching_scores = []

            for cell_index, cell_box in enumerate(normalized_cell_boxes):
                intersection_width = max(
                    0,
                    min(cell_box[2], text_box[2])
                    - max(cell_box[0], text_box[0]),
                )
                intersection_height = max(
                    0,
                    min(cell_box[3], text_box[3])
                    - max(cell_box[1], text_box[1]),
                )
                intersection_area = intersection_width * intersection_height
                cell_area = (
                    (cell_box[2] - cell_box[0])
                    * (cell_box[3] - cell_box[1])
                )
                text_area = (
                    (text_box[2] - text_box[0])
                    * (text_box[3] - text_box[1])
                )
                union_area = cell_area + text_area - intersection_area
                iou = intersection_area / union_area
                distance = (
                    abs(text_box[0] - cell_box[0])
                    + abs(text_box[1] - cell_box[1])
                    + abs(text_box[2] - cell_box[2])
                    + abs(text_box[3] - cell_box[3])
                )
                matching_scores.append((1.0 - iou, distance, cell_index))

            _, _, cell_index = min(matching_scores)
            cell_text_regions[cell_index].append(text_region)

        cell_texts = []
        for regions in cell_text_regions:
            regions.sort(
                key=lambda region: (
                    region["center"][1],
                    region["center"][0],
                )
            )
            cell_texts.append(
                " ".join(region["text"] for region in regions)
            )

        html_parts = structure_html.split("</td>")
        if len(html_parts) - 1 != len(cell_texts):
            raise ValueError(
                "Số cell bbox không khớp số thẻ td của SLANeXt_wired."
            )

        filled_parts = []
        for html_part, cell_text in zip(html_parts[:-1], cell_texts):
            filled_parts.append(
                f"{html_part}{escape(cell_text)}</td>"
            )

        return "".join(filled_parts) + html_parts[-1]

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
            structure_html = "".join(structure_result["structure"])
            ocr_count = 0

            if table_type == "wired_table":
                text_regions = self._recognize_text(table_crop)
                structure_html = self._fill_structure_html(
                    structure_html,
                    structure_result["bbox"],
                    text_regions,
                )
                ocr_count = len(text_regions)

            structure_outputs.append(
                {
                    "table_index": table_index,
                    "table_type": table_type,
                    "classification_score": classification_score,
                    "model_name": model_name,
                    "structure_score": structure_score,
                    "ocr_count": ocr_count,
                    "html": structure_html,
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
    """Chạy phân tích ảnh page, lưu ảnh và HTML của các wired_table."""
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
    wired_table_sections = []

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

        for structure_output in structure_outputs:
            if structure_output["table_type"] != "wired_table":
                continue

            structure_html = structure_output["html"]
            table_start = structure_html.index("<table")
            table_end = structure_html.index("</table>") + len("</table>")
            table_html = structure_html[table_start:table_end]
            wired_table_sections.append(
                f"""
<section>
    <h2>{image_path.parent.name} / {image_path.name}</h2>
    <p class="metadata">
        Loại bảng: wired_table |
        classification score: {structure_output['classification_score']:.4f} |
        model: {structure_output['model_name']} |
        structure score: {structure_output['structure_score']:.4f} |
        OCR: PP-OCRv6_medium_det + PP-OCRv6_medium_rec |
        text regions: {structure_output['ocr_count']}
    </p>
    <img src="{output_path.name}" alt="Ảnh phân tích {output_stem}">
    <h3>Cấu trúc bảng</h3>
    {table_html}
</section>
"""
            )

    wired_table_html = f"""<!doctype html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <title>Phân tích wired_table bằng SLANeXt_wired</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 24px;
            color: #222;
        }}
        section {{
            border-top: 2px solid #777;
            margin-top: 32px;
            padding-top: 20px;
        }}
        .metadata {{ color: #444; }}
        img {{
            border: 1px solid #bbb;
            display: block;
            height: auto;
            margin: 16px 0;
            max-width: 100%;
        }}
        table {{
            border-collapse: collapse;
            margin-top: 16px;
            width: 100%;
        }}
        td, th {{
            border: 1px solid #111;
            min-width: 40px;
            height: 24px;
            padding: 4px 8px;
            vertical-align: top;
        }}
    </style>
</head>
<body>
    <h1>Phân tích wired_table bằng SLANeXt_wired</h1>
    <p>Tổng số bảng: {len(wired_table_sections)}</p>
    {''.join(wired_table_sections)}
</body>
</html>
"""
    wired_table_path = output_directory / "wired_table_analysis.html"
    wired_table_path.write_text(wired_table_html, encoding="utf-8")
    print(f"HTML wired_table: {wired_table_path}")

    print(f"Đã xử lý {len(image_paths)} ảnh.")


if __name__ == "__main__":
    main()
