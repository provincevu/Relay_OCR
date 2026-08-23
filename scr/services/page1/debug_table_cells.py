# Mục đích: tách logic OCR, gán text và sắp xếp cell cho script debug page 1.
# Chức năng: tách text qua divider, nhận dạng bằng hai model, gán cell và tạo bảng logic 7x2.
# Luồng xử lý: ảnh bảng → divider dọc → text region → cell → hàng → HTML bảng chuẩn hóa.
# Cấu hình/parameter: dùng text_detector, text_recognizer và ppocr_recognizer của analyzer.
# Quan hệ: được Page1TableAnalyzer kế thừa; phối hợp với module field và visualization.
# Input/output: nhận ảnh hoặc bbox; trả text region, cell, hàng hoặc HTML bảng.
# Contract: text tiếp tục là kết quả VietOCR; ppocr_text chứa kết quả PP-OCRv6.

import re
from html import escape

import cv2
import numpy as np


class DebugTableCellsMixin:
    """Cung cấp logic OCR và xử lý cell cho analyzer debug."""

    def _recognize_text(self, image):
        """Nhận dạng mỗi vùng text bằng VietOCR và PP-OCRv6."""
        text_regions = []
        height, width = image.shape[:2]
        divider_x = self._detect_main_vertical_divider(image)
        detection_results = self.text_detector.detect(image)

        for result in detection_results:
            for polygon in result["dt_polys"]:
                polygon = np.array(polygon, dtype=np.float32)
                split_polygons = self._split_polygon_at_divider(
                    polygon,
                    divider_x,
                    width,
                )
                for split_polygon in split_polygons:
                    x_min, y_min = split_polygon.min(axis=0).astype(int)
                    x_max, y_max = split_polygon.max(axis=0).astype(int)
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
                    center_x = float(split_polygon[:, 0].mean())
                    center_y = float(split_polygon[:, 1].mean())
                    text_regions.append(
                        {
                            "polygon": split_polygon.tolist(),
                            "center": [center_x, center_y],
                            "text": text,
                            "ppocr_text": ppocr_text,
                        }
                    )

        return text_regions

    def _detect_main_vertical_divider(self, image):
        """Tìm đường phân cách dọc dài trong vùng giữa bảng."""
        _, image_width = image.shape[:2]
        scale = min(1.0, 600 / image_width)
        resized_image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 30, 100, apertureSize=3)
        height, width = gray_image.shape
        detected_lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 1800,
            threshold=max(20, int(height * 0.12)),
            minLineLength=int(height * 0.25),
            maxLineGap=int(height * 0.2),
        )
        if detected_lines is None:
            return None

        candidates = []
        for x_start, y_start, x_end, y_end in detected_lines.reshape(-1, 4):
            vertical_span = abs(int(y_end) - int(y_start))
            horizontal_span = abs(int(x_end) - int(x_start))
            center_x = (int(x_start) + int(x_end)) / 2
            if vertical_span < height * 0.25:
                continue
            if horizontal_span > vertical_span * 0.08:
                continue
            if not width * 0.4 < center_x < width * 0.7:
                continue
            candidates.append((vertical_span, center_x))

        if not candidates:
            return None

        _, divider_x = max(candidates)
        return int(round(divider_x / scale))

    def _split_polygon_at_divider(self, polygon, divider_x, image_width):
        """Tách polygon text nếu nó cắt qua đường phân cách dọc chính."""
        if divider_x is None:
            return [polygon]

        x_min, y_min = polygon.min(axis=0)
        x_max, y_max = polygon.max(axis=0)
        minimum_width = max(20, int(image_width * 0.02))
        if (
            divider_x - x_min < minimum_width
            or x_max - divider_x < minimum_width
        ):
            return [polygon]

        left_polygon = np.array(
            [
                [x_min, y_min],
                [divider_x - 1, y_min],
                [divider_x - 1, y_max],
                [x_min, y_max],
            ],
            dtype=np.float32,
        )
        right_polygon = np.array(
            [
                [divider_x + 1, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [divider_x + 1, y_max],
            ],
            dtype=np.float32,
        )
        return [left_polygon, right_polygon]

    def _resolve_nested_cell_boxes(self, image, cell_boxes):
        """Loại box lồng nhau dựa trên đường kẻ phân chia cell thật."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, line_mask = cv2.threshold(
            gray_image,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        removed_indexes = set()
        box_indexes = sorted(
            range(len(cell_boxes)),
            key=lambda index: self._box_area(cell_boxes[index]),
            reverse=True,
        )

        for parent_index in box_indexes:
            if parent_index in removed_indexes:
                continue

            parent_box = cell_boxes[parent_index]
            child_indexes = [
                child_index
                for child_index in box_indexes
                if child_index != parent_index
                and child_index not in removed_indexes
                and self._is_nested_cell(
                    parent_box,
                    cell_boxes[child_index],
                )
            ]
            if not child_indexes:
                continue

            child_boxes = [
                cell_boxes[child_index]
                for child_index in child_indexes
            ]
            if self._children_have_real_separator(
                line_mask,
                parent_box,
                child_boxes,
            ):
                removed_indexes.add(parent_index)
                continue

            removed_indexes.update(child_indexes)

        return [
            cell_box
            for index, cell_box in enumerate(cell_boxes)
            if index not in removed_indexes
        ]

    def _is_nested_cell(self, parent_box, child_box):
        """Kiểm tra một cell nhỏ có nằm gần như hoàn toàn trong cell lớn."""
        parent_area = self._box_area(parent_box)
        child_area = self._box_area(child_box)
        if child_area > parent_area * 0.85:
            return False

        parent_x_min, parent_y_min, parent_x_max, parent_y_max = map(
            float,
            parent_box["coordinate"],
        )
        child_x_min, child_y_min, child_x_max, child_y_max = map(
            float,
            child_box["coordinate"],
        )
        intersection_width = max(
            0.0,
            min(parent_x_max, child_x_max)
            - max(parent_x_min, child_x_min),
        )
        intersection_height = max(
            0.0,
            min(parent_y_max, child_y_max)
            - max(parent_y_min, child_y_min),
        )
        intersection_area = intersection_width * intersection_height
        return intersection_area / child_area >= 0.9

    def _children_have_real_separator(
        self,
        line_mask,
        parent_box,
        child_boxes,
    ):
        """Kiểm tra các cell con có được ngăn bởi đường kẻ thật hay không."""
        parent_x_min, parent_y_min, parent_x_max, parent_y_max = map(
            float,
            parent_box["coordinate"],
        )
        parent_width = parent_x_max - parent_x_min
        parent_height = parent_y_max - parent_y_min

        for first_index, first_box in enumerate(child_boxes):
            first_coordinate = list(map(float, first_box["coordinate"]))
            for second_box in child_boxes[first_index + 1 :]:
                second_coordinate = list(
                    map(float, second_box["coordinate"])
                )
                left_box, right_box = sorted(
                    [first_coordinate, second_coordinate],
                    key=lambda coordinate: coordinate[0],
                )
                vertical_overlap = max(
                    0.0,
                    min(left_box[3], right_box[3])
                    - max(left_box[1], right_box[1]),
                )
                horizontal_span = (
                    max(left_box[2], right_box[2])
                    - min(left_box[0], right_box[0])
                )
                horizontal_gap = right_box[0] - left_box[2]
                if (
                    vertical_overlap >= parent_height * 0.65
                    and horizontal_span >= parent_width * 0.8
                    and horizontal_gap <= parent_width * 0.05
                ):
                    separator_x = (left_box[2] + right_box[0]) / 2
                    if self._has_continuous_line(
                        line_mask,
                        separator_x,
                        max(left_box[1], right_box[1]),
                        min(left_box[3], right_box[3]),
                        vertical=True,
                    ):
                        return True

                top_box, bottom_box = sorted(
                    [first_coordinate, second_coordinate],
                    key=lambda coordinate: coordinate[1],
                )
                horizontal_overlap = max(
                    0.0,
                    min(top_box[2], bottom_box[2])
                    - max(top_box[0], bottom_box[0]),
                )
                vertical_span = (
                    max(top_box[3], bottom_box[3])
                    - min(top_box[1], bottom_box[1])
                )
                vertical_gap = bottom_box[1] - top_box[3]
                if (
                    horizontal_overlap >= parent_width * 0.65
                    and vertical_span >= parent_height * 0.8
                    and vertical_gap <= parent_height * 0.05
                ):
                    separator_y = (top_box[3] + bottom_box[1]) / 2
                    if self._has_continuous_line(
                        line_mask,
                        separator_y,
                        max(top_box[0], bottom_box[0]),
                        min(top_box[2], bottom_box[2]),
                        vertical=False,
                    ):
                        return True

        return False

    def _has_continuous_line(
        self,
        line_mask,
        position,
        span_start,
        span_end,
        vertical,
    ):
        """Đo độ liên tục của đường kẻ quanh một ranh giới cell."""
        height, width = line_mask.shape
        span_length = span_end - span_start
        margin = max(2, int(span_length * 0.08))
        radius = max(3, int(min(height, width) * 0.006))

        if vertical:
            y_start = max(0, int(round(span_start)) + margin)
            y_end = min(height, int(round(span_end)) - margin)
            x_start = max(0, int(round(position)) - radius)
            x_end = min(width, int(round(position)) + radius + 1)
            region = line_mask[y_start:y_end, x_start:x_end]
            if region.size == 0:
                return False
            continuity = np.max(np.mean(region > 0, axis=0))
        else:
            x_start = max(0, int(round(span_start)) + margin)
            x_end = min(width, int(round(span_end)) - margin)
            y_start = max(0, int(round(position)) - radius)
            y_end = min(height, int(round(position)) + radius + 1)
            region = line_mask[y_start:y_end, x_start:x_end]
            if region.size == 0:
                return False
            continuity = np.max(np.mean(region > 0, axis=1))

        return continuity >= 0.7

    def _box_area(self, box):
        """Tính diện tích bbox của cell."""
        x_min, y_min, x_max, y_max = map(float, box["coordinate"])
        return (x_max - x_min) * (y_max - y_min)

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

    def _group_boxes_into_rows(self, boxes, use_vertical_center=False):
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
                    "top": (
                        (y_min + y_max) / 2
                        if use_vertical_center
                        else y_min
                    ),
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

        ordered_rows = self._group_boxes_into_rows(
            text_boxes,
            use_vertical_center=True,
        )
        return [
            box["text_region"]
            for row in ordered_rows
            for box in row
        ]

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
