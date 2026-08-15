import base64
import re
from html import escape
from pathlib import Path

import cv2
import numpy as np
from paddleocr import (
    LayoutDetection,
    TableCellsDetection,
    TableStructureRecognition,
)

from scr.services.detector import Detector
from scr.services.recognitor import Recognitor

# Mục đích: debug nhận diện và trích xuất nội dung TABLE 01, TABLE 02 trên page 1.
# Chức năng: nhận diện bảng, cấu trúc, cell và text; xuất một file HTML trực quan.
# Luồng xử lý: ảnh → table bbox → crop → structure/cell → Detector → Recognitor → HTML.
# Cấu hình/parameter: dùng CPU; mặc định chỉ lấy TABLE 01, có thể bật TABLE 02.
# Quan hệ: dùng các model bảng wired của PaddleOCR và Detector/Recognitor hiện có.
# Input/output: đọc toàn bộ data/image/; ghi ảnh debug, crop và output/layout/table_analysis.html.
# Contract: detect_tables() vẫn trả danh sách bbox table có coordinate [xmin, ymin, xmax, ymax].


class Page1TableAnalyzer:
    """Phân tích và trực quan hóa các bảng trên ảnh page 1."""

    def __init__(
        self,
        output_dir=Path("./output/layout"),
        include_second_table=False,
        include_third_table=False,
    ):
        self.output_dir = Path(output_dir)
        self.include_second_table = include_second_table
        self.include_third_table = include_third_table
        self.html_sections = []
        self.layout_detector = LayoutDetection(
            model_name="PicoDet_layout_1x_table",
            device="cpu",
            engine="paddle_static",
            enable_mkldnn=False,
            cpu_threads=2,
        )
        self.structure_model = TableStructureRecognition(
            model_name="SLANeXt_wired",
            device="cpu",
            engine="paddle_static",
            enable_mkldnn=False,
            cpu_threads=2,
        )
        self.cell_model = TableCellsDetection(
            model_name="RT-DETR-L_wired_table_cell_det",
            device="cpu",
            engine="paddle_static",
            enable_mkldnn=False,
            cpu_threads=2,
        )
        self.text_detector = Detector({})
        self.text_recognizer = Recognitor({})

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
        visual_image = self._draw_tables(image, table_boxes)
        visual_path = self.output_dir / "image" / f"{image_path.stem}_tables.png"
        visual_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(visual_path), visual_image)
        table_crops = self._save_table_crops(image, image_path, table_boxes)

        for table_index, crop_path, crop in table_crops:
            self._analyze_table(
                image_name=image_path.stem,
                table_index=table_index,
                crop_path=crop_path,
                crop=crop,
            )

        print(
            f"Model nhận diện {len(detected_table_boxes)} bảng, "
            f"đang xuất {len(table_boxes)} bảng"
        )
        print(f"Ảnh trực quan: {visual_path}")
        return table_boxes

    def save_html(self):
        """Lưu một file HTML chứa toàn bộ kết quả trực quan và text đã trích xuất."""
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

    def _analyze_table(self, image_name, table_index, crop_path, crop):
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
        text_regions = self._recognize_text(crop)
        cells, unassigned_texts = self._assign_text_to_cells(
            cell_result["boxes"],
            text_regions,
        )
        normalized_table_html = self._build_normalized_table(cells)
        html_section = self._build_html_section(
            image_name=image_name,
            table_index=table_index,
            crop_path=crop_path,
            crop=crop,
            cells=cells,
            text_regions=text_regions,
            unassigned_texts=unassigned_texts,
            structure_score=structure_result["structure_score"],
            normalized_table_html=normalized_table_html,
        )
        self.html_sections.append(html_section)

    def _recognize_text(self, image):
        """Dùng Detector và Recognitor hiện có để nhận dạng các vùng text."""
        text_regions = []
        height, width = image.shape[:2]
        detection_results = self.text_detector.detect(image)

        for result in detection_results:
            for polygon in result["dt_polys"]:
                polygon = np.array(polygon, dtype=np.float32)
                x_min, y_min = polygon.min(axis=0).astype(int)
                x_max, y_max = polygon.max(axis=0).astype(int)
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(x_max, width)
                y_max = min(y_max, height)
                text_crop = image[y_min:y_max, x_min:x_max]
                text = self.text_recognizer.recognize(text_crop)
                center_x = float(polygon[:, 0].mean())
                center_y = float(polygon[:, 1].mean())
                text_regions.append(
                    {
                        "polygon": polygon.tolist(),
                        "center": [center_x, center_y],
                        "text": text,
                    }
                )

        return text_regions

    def _assign_text_to_cells(self, cell_boxes, text_regions):
        """Gán text vào cell chứa tâm của polygon text."""
        sorted_boxes = self._sort_boxes_left_to_right(cell_boxes)
        cells = []

        for index, box in enumerate(sorted_boxes, start=1):
            cells.append(
                {
                    "index": index,
                    "coordinate": [float(value) for value in box["coordinate"]],
                    "score": float(box["score"]),
                    "text_regions": [],
                }
            )

        unassigned_texts = []
        for text_region in text_regions:
            center_x, center_y = text_region["center"]
            matched_cells = [
                cell
                for cell in cells
                if cell["coordinate"][0] <= center_x <= cell["coordinate"][2]
                and cell["coordinate"][1] <= center_y <= cell["coordinate"][3]
            ]

            if matched_cells:
                matched_cell = min(
                    matched_cells,
                    key=lambda cell: (
                        cell["coordinate"][2] - cell["coordinate"][0]
                    )
                    * (cell["coordinate"][3] - cell["coordinate"][1]),
                )
                matched_cell["text_regions"].append(text_region)
            else:
                unassigned_texts.append(text_region)

        for cell in cells:
            cell["text_regions"] = self._sort_text_regions_left_to_right(
                cell["text_regions"]
            )
            cell["text"] = " ".join(
                region["text"] for region in cell["text_regions"]
            )

        return cells, unassigned_texts

    def _sort_boxes_left_to_right(self, boxes):
        """Sắp xếp bbox theo từng hàng, trái sang phải trong mỗi hàng."""
        return [
            box
            for row in self._group_boxes_into_rows(boxes)
            for box in row
        ]

    def _group_boxes_into_rows(self, boxes):
        """Gom bbox theo hàng và sắp xếp trái sang phải trong từng hàng."""
        if not boxes:
            return []

        box_items = []
        heights = []
        for box in boxes:
            x_min, y_min, x_max, y_max = map(float, box["coordinate"])
            box_items.append(
                {
                    "box": box,
                    "top": y_min,
                    "left": x_min,
                }
            )
            heights.append(y_max - y_min)

        row_tolerance = max(8.0, float(np.median(heights)) * 0.5)
        rows = []
        for item in sorted(box_items, key=lambda value: value["top"]):
            matched_rows = [
                row
                for row in rows
                if abs(item["top"] - row["top"]) <= row_tolerance
            ]
            if matched_rows:
                row = min(
                    matched_rows,
                    key=lambda value: abs(item["top"] - value["top"]),
                )
                row["items"].append(item)
                row["top"] = sum(
                    value["top"] for value in row["items"]
                ) / len(row["items"])
            else:
                rows.append({"top": item["top"], "items": [item]})

        ordered_rows = []
        for row in sorted(rows, key=lambda value: value["top"]):
            ordered_rows.append(
                [
                    item["box"]
                    for item in sorted(
                        row["items"],
                        key=lambda value: value["left"],
                    )
                ]
            )

        return ordered_rows

    def _sort_text_regions_left_to_right(self, text_regions):
        """Sắp xếp các vùng text theo từng hàng, trái sang phải."""
        text_boxes = []
        for text_region in text_regions:
            polygon = np.array(text_region["polygon"], dtype=np.float32)
            x_min, y_min = polygon.min(axis=0)
            x_max, y_max = polygon.max(axis=0)
            text_boxes.append(
                {
                    "coordinate": [x_min, y_min, x_max, y_max],
                    "text_region": text_region,
                }
            )

        ordered_boxes = self._sort_boxes_left_to_right(text_boxes)
        return [box["text_region"] for box in ordered_boxes]

    def _split_cell_fields(self, cell):
        """Tách phiên bản và giữ lại text không hợp lệ trong field liên quan."""
        texts = [
            region["text"].strip()
            for region in cell["text_regions"]
            if region["text"].strip()
        ]
        if not texts and cell["text"].strip():
            texts = [cell["text"].strip()]
        joined_text = " ".join(texts)

        version_label = re.compile(
            r"phiên\s+bản(?:\s+rơ[- ]?le)?\s*:?",
            re.IGNORECASE,
        )
        version_value = re.compile(
            r"\bV?\s*\d+(?:[.,]\d+)*\b",
            re.IGNORECASE,
        )
        label_match = version_label.search(joined_text)
        if not label_match:
            return [{"text": joined_text, "kind": "normal"}]

        before_label = joined_text[:label_match.start()].strip()
        version_label_text = joined_text[
            label_match.start():label_match.end()
        ].strip()
        after_label = joined_text[label_match.end():].strip()
        value_match = version_value.search(after_label)

        if not value_match:
            if before_label:
                primary = " ".join(
                    part for part in [before_label, after_label] if part
                )
                return [
                    {"text": primary, "kind": "primary"},
                    {"text": version_label_text, "kind": "version"},
                ]
            return [
                {"text": after_label, "kind": "append"},
                {"text": version_label_text, "kind": "version"},
            ]

        text_before_value = after_label[:value_match.start()].strip()
        text_after_value = after_label[value_match.end():].strip()
        version_field = " ".join(
            part
            for part in [
                version_label_text,
                text_before_value,
                value_match.group(0).replace(" ", ""),
            ]
            if part
        )
        primary = " ".join(
            part for part in [before_label, text_after_value] if part
        )
        fields = []
        if primary:
            fields.append({"text": primary, "kind": "primary"})
        fields.append({"text": version_field, "kind": "version"})
        return fields

    def _build_normalized_table(self, cells):
        """Tạo bảng ngoài 7x2 và giữ field bổ sung trong cell phải."""
        rows = self._group_boxes_into_rows(cells)
        row_html = []

        for row in rows:
            left_cell = row[0]
            right_fields = []
            for cell in row[1:]:
                for field in self._split_cell_fields(cell):
                    if field["kind"] == "append" and right_fields:
                        right_fields[-1]["text"] = " ".join(
                            [right_fields[-1]["text"], field["text"]]
                        ).strip()
                    else:
                        right_fields.append(field)

            if not right_fields:
                right_fields = [{"text": "", "kind": "normal"}]

            fields_html = "".join(
                f'<div class="logical-field">{escape(field["text"])}</div>'
                for field in right_fields
            )
            row_html.append(
                "<tr>"
                f'<td class="physical-cell">{escape(left_cell["text"])}</td>'
                '<td class="physical-cell">'
                f'<div class="logical-fields">{fields_html}</div>'
                "</td>"
                "</tr>"
            )

        return (
            '<table class="normalized-table">'
            "<thead><tr><th>Cột vật lý 01</th><th>Cột vật lý 02</th></tr></thead>"
            f'<tbody>{"".join(row_html)}</tbody>'
            "</table>"
        )

    def _build_html_section(
        self,
        image_name,
        table_index,
        crop_path,
        crop,
        cells,
        text_regions,
        unassigned_texts,
        structure_score,
        normalized_table_html,
    ):
        """Tạo section HTML trực quan cho một bảng."""
        height, width = crop.shape[:2]
        image_buffer = cv2.imencode(".png", crop)[1]
        image_data = base64.b64encode(image_buffer).decode("ascii")

        cell_shapes = []
        for cell in cells:
            x_min, y_min, x_max, y_max = cell["coordinate"]
            cell_shapes.append(
                f'<rect class="cell-box" x="{x_min}" y="{y_min}" '
                f'width="{x_max - x_min}" height="{y_max - y_min}" />'
            )
            cell_shapes.append(
                f'<text class="cell-label" x="{x_min + 4}" y="{y_min + 20}">'
                f'C{cell["index"]:02d}</text>'
            )

        text_shapes = []
        for text_region in text_regions:
            points = " ".join(
                f"{point[0]},{point[1]}" for point in text_region["polygon"]
            )
            text_shapes.append(f'<polygon class="text-box" points="{points}" />')

        cell_rows = []
        for cell in cells:
            coordinate = ", ".join(
                f"{value:.1f}" for value in cell["coordinate"]
            )
            cell_rows.append(
                "<tr>"
                f'<td>C{cell["index"]:02d}</td>'
                f"<td>{coordinate}</td>"
                f'<td>{cell["score"]:.3f}</td>'
                f'<td>{escape(cell["text"])}</td>'
                "</tr>"
            )

        unassigned_html = " ".join(
            escape(region["text"]) for region in unassigned_texts
        )
        return f"""
<section>
<h2>{escape(image_name)} - TABLE {table_index:02d}</h2>
<p>Crop: {escape(str(crop_path))}</p>
<p>Structure score: {float(structure_score):.4f}</p>
<svg viewBox="0 0 {width} {height}" aria-label="Kết quả detection">
<image href="data:image/png;base64,{image_data}" width="{width}" height="{height}" />
{"".join(cell_shapes)}
{"".join(text_shapes)}
</svg>
<h3>Thông tin theo cell</h3>
<table>
<thead><tr><th>Cell</th><th>Bounding box</th><th>Score</th><th>Text</th></tr></thead>
<tbody>{"".join(cell_rows)}</tbody>
</table>
<p><strong>Text chưa gán vào cell:</strong> {unassigned_html}</p>
<h3>Bảng chuẩn hóa: 7 hàng × 2 cột vật lý</h3>
<p>Field bổ sung như “Phiên bản” được giữ bên trong cột phải dưới dạng field logic.</p>
{normalized_table_html}
</section>
"""

    def _draw_tables(self, image, table_boxes):
        """Vẽ bbox, nhãn score và nền tô mờ lên ảnh debug."""
        visual_image = image.copy()
        overlay = visual_image.copy()
        colors = [
            (0, 180, 0),
            (255, 140, 0),
            (180, 0, 180),
            (0, 140, 255),
        ]

        for index, box in enumerate(table_boxes):
            x_min, y_min, x_max, y_max = map(int, box["coordinate"])
            color = colors[index % len(colors)]
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)

        visual_image = cv2.addWeighted(overlay, 0.16, visual_image, 0.84, 0)

        for index, box in enumerate(table_boxes, start=1):
            x_min, y_min, x_max, y_max = map(int, box["coordinate"])
            color = colors[(index - 1) % len(colors)]
            label = f"TABLE {index:02d} | {box['score']:.3f}"
            text_size, baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2,
            )
            label_top = max(y_min - text_size[1] - baseline - 6, 0)
            label_bottom = label_top + text_size[1] + baseline + 6
            label_right = x_min + text_size[0] + 10

            cv2.rectangle(
                visual_image,
                (x_min, y_min),
                (x_max, y_max),
                color,
                3,
            )
            cv2.rectangle(
                visual_image,
                (x_min, label_top),
                (label_right, label_bottom),
                color,
                -1,
            )
            cv2.putText(
                visual_image,
                label,
                (x_min + 5, label_bottom - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return visual_image

    def _save_table_crops(self, image, image_path, table_boxes):
        """Lưu và trả các crop table để tiếp tục phân tích."""
        crop_dir = self.output_dir / "crops" / image_path.stem
        crop_dir.mkdir(parents=True, exist_ok=True)
        table_crops = []
        height, width = image.shape[:2]

        for index, box in enumerate(table_boxes, start=1):
            x_min, y_min, x_max, y_max = map(int, box["coordinate"])
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, width)
            y_max = min(y_max, height)
            crop = image[y_min:y_max, x_min:x_max]
            crop_path = crop_dir / f"table_{index:02d}.png"
            cv2.imwrite(str(crop_path), crop)
            table_crops.append((index, crop_path, crop))

        return table_crops


if __name__ == "__main__":
    analyzer = Page1TableAnalyzer(
        include_second_table=False,
        include_third_table=False,
    )
    input_dir = Path("./data/image")
    image_files = sorted(
        path
        for path in input_dir.iterdir()
        if path.suffix.lower()
        in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )

    for index, image_path in enumerate(image_files, start=1):
        print(f"[{index}/{len(image_files)}] Đang xử lý: {image_path.name}")
        analyzer.detect_tables(image_path)

    analyzer.save_html()
