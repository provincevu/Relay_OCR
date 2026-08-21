# Mục đích: tách logic OCR, gán text và sắp xếp cell cho script debug page 1.
# Chức năng: nhận dạng bằng VietOCR và PP-OCRv6, gán text vào cell và tạo bảng logic 7x2.
# Luồng xử lý: ảnh bảng → text region từ hai recognizer → cell → hàng → HTML bảng chuẩn hóa.
# Cấu hình/parameter: dùng text_detector, text_recognizer và ppocr_recognizer của analyzer.
# Quan hệ: được Page1TableAnalyzer kế thừa; phối hợp với module field và visualization.
# Input/output: nhận ảnh hoặc bbox; trả text region, cell, hàng hoặc HTML bảng.
# Contract: text tiếp tục là kết quả VietOCR; ppocr_text chứa kết quả PP-OCRv6.

import re
from html import escape

import numpy as np


class DebugTableCellsMixin:
    """Cung cấp logic OCR và xử lý cell cho analyzer debug."""

    def _recognize_text(self, image):
        """Nhận dạng mỗi vùng text bằng VietOCR và PP-OCRv6."""
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
                ppocr_result = self.ppocr_recognizer.predict(
                    text_crop,
                    batch_size=1,
                )[0]
                ppocr_text = ppocr_result["rec_text"]
                center_x = float(polygon[:, 0].mean())
                center_y = float(polygon[:, 1].mean())
                text_regions.append(
                    {
                        "polygon": polygon.tolist(),
                        "center": [center_x, center_y],
                        "text": text,
                        "ppocr_text": ppocr_text,
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
                    "coordinate": [
                        float(value) for value in box["coordinate"]
                    ],
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
                and cell["coordinate"][1]
                <= center_y
                <= cell["coordinate"][3]
            ]

            if matched_cells:
                matched_cell = min(
                    matched_cells,
                    key=lambda cell: (
                        cell["coordinate"][2] - cell["coordinate"][0]
                    )
                    * (
                        cell["coordinate"][3] - cell["coordinate"][1]
                    ),
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
            cell["ppocr_text"] = " ".join(
                region["ppocr_text"] for region in cell["text_regions"]
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
