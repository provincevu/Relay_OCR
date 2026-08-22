# Mục đích: điều phối toàn bộ luồng debug nhận diện bảng trên page 1.
# Chức năng: khởi tạo model, nhận diện bảng/cell/text bằng VietOCR và PP-OCRv6 rồi lưu kết quả debug.
# Luồng xử lý: ảnh → table crop → structure/cell → VietOCR/PP-OCRv6 → field → HTML/JSON/report.
# Cấu hình/parameter: chọn số bảng, padding crop và tùy chọn JSON đánh giá.
# Quan hệ: kết hợp ba module debug_table_fields, debug_table_cells và debug_table_visualization.
# Input/output: detect_tables() nhận đường dẫn ảnh; các phương thức save ghi output/layout.
# Contract: giữ Page1TableAnalyzer và các phương thức public từ script test trước khi tách.

import json
from pathlib import Path

import cv2
from paddleocr import (
    LayoutDetection,
    TableCellsDetection,
    TableStructureRecognition,
    TextRecognition,
)

from scr.services.detector import Detector
from scr.services.recognitor import Recognitor

from .debug_table_cells import DebugTableCellsMixin
from .debug_table_fields import DebugTableFieldsMixin
from .debug_table_visualization import DebugTableVisualizationMixin


TABLE_SCORE_WARNING_THRESHOLD = 0.8
CELL_SCORE_THRESHOLD = 0.9


class Page1TableAnalyzer(
    DebugTableFieldsMixin,
    DebugTableCellsMixin,
    DebugTableVisualizationMixin,
):
    """Phân tích và trực quan hóa các bảng trên ảnh page 1."""

    def __init__(
        self,
        output_dir=Path("./output/layout"),
        include_second_table=False,
        include_third_table=False,
        crop_padding_left=0,
        crop_padding_top=0,
        crop_padding_right=0,
        crop_padding_bottom=0,
        save_evaluation_json=False,
    ):
        self.output_dir = Path(output_dir)
        self.include_second_table = include_second_table
        self.include_third_table = include_third_table
        self.crop_padding_left = crop_padding_left
        self.crop_padding_top = crop_padding_top
        self.crop_padding_right = crop_padding_right
        self.crop_padding_bottom = crop_padding_bottom
        self.save_evaluation_json = save_evaluation_json
        self.html_sections = []
        self.table_warnings = []
        self.report_lines = []
        field_config_path = Path("./config/table_01_fields.json")
        field_config = json.loads(
            field_config_path.read_text(encoding="utf-8")
        )
        self.field_definitions = field_config["fields"]
        self.extracted_tables = []
        self.field_review = []
        self.layout_detector = LayoutDetection(
            model_name="PicoDet_layout_1x_table",
            device="cpu",
            engine="paddle_static",
            enable_mkldnn=False,
            cpu_threads=6,
        )
        self.structure_model = TableStructureRecognition(
            model_name="SLANeXt_wired",
            device="cpu",
            engine="paddle_static",
            enable_mkldnn=False,
            cpu_threads=6,
        )
        self.cell_model = TableCellsDetection(
            model_name="RT-DETR-L_wired_table_cell_det",
            device="cpu",
            engine="paddle_static",
            enable_mkldnn=False,
            cpu_threads=6,
        )
        self.text_detector = Detector({})
        self.text_recognizer = Recognitor({})
        self.ppocr_recognizer = TextRecognition(
            model_name="PP-OCRv6_medium_rec",
            device="cpu",
            engine="paddle_static",
            enable_mkldnn=False,
            cpu_threads=6,
        )

    def detect_tables(self, image_path):
        """Nhận diện, trích xuất và lưu các bảng được chọn để debug."""
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        results = self.layout_detector.predict(
            str(image_path),
            batch_size=1,
            layout_nms=True,
        )
        detected_table_boxes = []

        for result in results:
            result.print()
            result.save_to_img(str(self.output_dir / "image"))
            detected_table_boxes.extend(
                box
                for box in result["boxes"]
                if box["label"].lower() == "table"
            )

        detected_table_boxes = self._sort_boxes_left_to_right(
            detected_table_boxes
        )
        if self.include_third_table:
            table_boxes = detected_table_boxes
        elif self.include_second_table:
            table_boxes = detected_table_boxes[:2]
        else:
            table_boxes = detected_table_boxes[:1]

        if table_boxes:
            table_01_score = float(table_boxes[0]["score"])
            if table_01_score < TABLE_SCORE_WARNING_THRESHOLD:
                warning = (
                    f"TABLE 01 score thấp | {image_path.name} | "
                    f"score={table_01_score:.4f}"
                )
                self.table_warnings.append(warning)
                self.report_lines.append(warning)

        visual_image = self._draw_tables(image, table_boxes)
        visual_path = self.output_dir / "image" / (
            f"{image_path.stem}_tables.png"
        )
        visual_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(visual_path), visual_image)
        table_crops = self._save_table_crops(
            image,
            image_path,
            table_boxes,
        )

        for table_index, crop_path, crop in table_crops:
            self._analyze_table(
                image_name=image_path.stem,
                source_image=image_path.name,
                table_index=table_index,
                table_score=float(table_boxes[table_index - 1]["score"]),
                crop_path=crop_path,
                crop=crop,
            )

        print(
            f"Model nhận diện {len(detected_table_boxes)} bảng, "
            f"đang xuất {len(table_boxes)} bảng"
        )
        return table_boxes

    def save_html(self):
        """Lưu một file HTML chứa kết quả trực quan và text đã trích xuất."""
        output_path = self.output_dir / "table_analysis.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sections = "\n".join(self.html_sections)
        html = f"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Phân tích TABLE 01 và TABLE 02</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
h1 {{ margin-bottom: 8px; }}
section {{ margin: 32px 0 48px; border-top: 2px solid #d1d5db; padding-top: 20px; }}
svg {{ width: 100%; height: auto; border: 1px solid #9ca3af; background: #fff; }}
.cell-box {{ fill: rgba(34, 197, 94, 0.12); stroke: #16a34a; stroke-width: 2; }}
.text-box {{ fill: none; stroke: #2563eb; stroke-width: 1.5; }}
.cell-label {{ fill: #b91c1c; font-size: 18px; font-weight: bold; }}
.unofficial-cell {{ background: #fff7ed; color: #9a3412; }}
.comparison-diff {{ color: #b91c1c; font-weight: bold; }}
.legend span {{ margin-right: 24px; }}
.green {{ color: #16a34a; }}
.blue {{ color: #2563eb; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
th, td {{ border: 1px solid #9ca3af; padding: 8px; text-align: left; }}
th {{ background: #f3f4f6; }}
.normalized-table .physical-cell {{ vertical-align: top; white-space: pre-wrap; }}
.logical-fields {{ display: flex; gap: 8px; }}
.logical-field {{ flex: 1; padding: 4px 8px; border-left: 1px dashed #9ca3af; }}
pre {{ overflow-x: auto; background: #f3f4f6; padding: 12px; }}
</style>
</head>
<body>
<h1>Phân tích TABLE 01 và TABLE 02</h1>
<p class="legend">
<span class="green">Màu xanh lá: cell</span>
<span class="blue">Màu xanh dương: vùng text</span>
</p>
{sections}
</body>
</html>
"""
        output_path.write_text(html, encoding="utf-8")
        print(f"HTML trực quan: {output_path}")

    def save_report(self):
        """Lưu report và in cảnh báo table sau khi hoàn tất toàn bộ ảnh."""
        output_path = self.output_dir / "report.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_content = "\n".join(self.report_lines)
        output_path.write_text(report_content, encoding="utf-8")

        if self.table_warnings:
            print("\nCẢNH BÁO TABLE 01:")
            for warning in self.table_warnings:
                print(warning)

        print(f"Report: {output_path}")

    def save_extracted_data(self):
        """Lưu dữ liệu trích xuất và các field cần xem lại vào JSON."""
        data_path = self.output_dir / "table_01_data.json"
        review_path = self.output_dir / "table_01_review.json"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(
            json.dumps(self.extracted_tables, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        review_path.write_text(
            json.dumps(self.field_review, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Dữ liệu TABLE 01: {data_path}")
        print(f"Field cần xem lại: {review_path}")

        if self.save_evaluation_json:
            evaluation_path = self.output_dir / "table_01_predictions.json"
            evaluation_data = {
                f"{Path(item['source_image']).stem}_table_01.png": item["data"]
                for item in self.extracted_tables
            }
            evaluation_path.write_text(
                json.dumps(
                    evaluation_data,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Dữ liệu đánh giá TABLE 01: {evaluation_path}")

    def _analyze_table(
        self,
        image_name,
        source_image,
        table_index,
        table_score,
        crop_path,
        crop,
    ):
        """Nhận diện structure, cell và text của một ảnh crop bảng."""
        structure_result = self.structure_model.predict(
            input=crop,
            batch_size=1,
        )[0]
        cell_result = self.cell_model.predict(
            crop,
            threshold=0.3,
            batch_size=1,
        )[0]
        sorted_cell_boxes = self._sort_boxes_left_to_right(
            cell_result["boxes"]
        )
        sorted_cell_boxes = self._resolve_nested_cell_boxes(
            crop,
            sorted_cell_boxes,
        )
        valid_cell_boxes = []
        low_score_cell_boxes = []

        for cell_box in sorted_cell_boxes:
            cell_score = float(cell_box["score"])
            if cell_score < CELL_SCORE_THRESHOLD:
                low_score_cell_boxes.append(cell_box)
                continue

            valid_cell_boxes.append(cell_box)

        text_regions = self._recognize_text(crop)
        unofficial_cell_boxes = []
        for cell_box in low_score_cell_boxes:
            detected_cells, _ = self._assign_text_to_cells(
                [cell_box],
                text_regions,
            )
            if self._find_field_matches(detected_cells[0]["text"]):
                valid_cell_boxes.append(cell_box)
                continue

            unofficial_cell_boxes.append(cell_box)
            cell_score = float(cell_box["score"])
            coordinate = ", ".join(
                f"{float(value):.1f}"
                for value in cell_box["coordinate"]
            )
            self.report_lines.append(
                f"CELL bị bỏ qua | {image_name} | TABLE {table_index:02d} | "
                f"CELL không chính thức | score={cell_score:.4f} | "
                f"bbox=[{coordinate}]"
            )

        cells, unassigned_texts = self._assign_text_to_cells(
            valid_cell_boxes,
            text_regions,
        )
        unofficial_cells, _ = self._assign_text_to_cells(
            unofficial_cell_boxes,
            text_regions,
        )
        for cell in unofficial_cells:
            cell["index"] = None

        if table_index == 1:
            self._extract_table_fields(
                source_image=source_image,
                table_score=table_score,
                cells=cells,
            )
        normalized_table_html = self._build_normalized_table(cells)
        html_section = self._build_html_section(
            image_name=image_name,
            table_index=table_index,
            crop_path=crop_path,
            crop=crop,
            cells=cells,
            unofficial_cells=unofficial_cells,
            text_regions=text_regions,
            unassigned_texts=unassigned_texts,
            structure_score=structure_result["structure_score"],
            normalized_table_html=normalized_table_html,
        )
        self.html_sections.append(html_section)
