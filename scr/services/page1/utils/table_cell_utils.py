# Mục đích: xử lý cell và text của TABLE 01 trên page 1.
# Chức năng: nhận dạng bằng hai model, loại cell lồng nhau và gán text vào cell.
# Luồng xử lý: table crop → cell box/text polygon → phân xử cell → OCR → gán text.
# Cấu hình/parameter: nhận các model đã được khởi tạo từ Table1Analysis.
# Quan hệ: được Table1Analysis và TableFieldExtractor sử dụng.
# Input/output: nhận ảnh BGR hoặc bbox; trả text region và cell đã chuẩn hóa.
# Contract: bbox dùng khóa coordinate theo dạng [xmin, ymin, xmax, ymax].

import cv2
import numpy as np


class TableCellProcessor:
    """Xử lý detection, recognition và gán text cho cell TABLE 01."""

    # Khởi tạo bộ xử lý và tái sử dụng các model OCR từ service điều phối.
    def __init__(
        self,
        text_detector,
        text_recognitor,
        ppocr_recognitor,
    ):
        """Lưu các model cần dùng trong quá trình xử lý cell."""
        self.text_detector = text_detector
        self.text_recognitor = text_recognitor
        self.ppocr_recognitor = ppocr_recognitor

    # Nhận dạng từng text polygon bằng VietOCR và PP-OCRv6.
    def recognize_text(self, image):
        """Nhận dạng text trong table crop bằng hai model."""
        text_regions = []
        height, width = image.shape[:2]
        divider_x = self._detect_main_vertical_divider(image)
        detection_results = self.text_detector.detect(image)

        # Duyệt toàn bộ polygon mà text detector trả về.
        for result in detection_results:
            for polygon in result["dt_polys"]:
                polygon = np.array(polygon, dtype=np.float32)
                split_polygons = self._split_polygon_at_divider(
                    polygon,
                    divider_x,
                    width,
                )

                # OCR riêng từng nửa nếu polygon cắt qua divider chính.
                for split_polygon in split_polygons:
                    x_min, y_min = split_polygon.min(axis=0).astype(int)
                    x_max, y_max = split_polygon.max(axis=0).astype(int)
                    x_min = max(x_min, 0)
                    y_min = max(y_min, 0)
                    x_max = min(x_max, width)
                    y_max = min(y_max, height)
                    text_crop = image[y_min:y_max, x_min:x_max]
                    text = self.text_recognitor.recognize(text_crop)
                    ppocr_result = self.ppocr_recognitor.predict(
                        text_crop,
                        batch_size=1,
                    )[0]
                    text_regions.append(
                        {
                            "polygon": split_polygon.tolist(),
                            "center": [
                                float(split_polygon[:, 0].mean()),
                                float(split_polygon[:, 1].mean()),
                            ],
                            "text": text,
                            "ppocr_text": ppocr_result["rec_text"],
                        }
                    )

        return text_regions

    # Loại các cell lồng nhau trước khi lọc theo score.
    def resolve_nested_cell_boxes(self, image, cell_boxes):
        """Giữ cell vật lý phù hợp với đường kẻ thật trong bảng."""
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
            key=lambda index: _box_area(cell_boxes[index]),
            reverse=True,
        )

        # Xét box lớn trước để phân xử quan hệ cha và con ổn định.
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

            # Có separator thật thì giữ các cell con và loại cell gộp.
            if self._children_have_real_separator(
                line_mask,
                parent_box,
                child_boxes,
            ):
                removed_indexes.add(parent_index)
                continue

            # Không có separator thì giữ cell vật lý lớn và loại box con giả.
            removed_indexes.update(child_indexes)

        return [
            cell_box
            for index, cell_box in enumerate(cell_boxes)
            if index not in removed_indexes
        ]

    # Gán mỗi text region vào cell nhỏ nhất chứa tâm polygon.
    def assign_text_to_cells(self, cell_boxes, text_regions):
        """Gán text vào cell và giữ text region phục vụ sửa value."""
        sorted_boxes = sort_boxes_left_to_right(cell_boxes)
        cells = []

        # Chuẩn hóa coordinate và score của từng cell.
        for box in sorted_boxes:
            cells.append(
                {
                    "coordinate": [
                        float(value) for value in box["coordinate"]
                    ],
                    "score": float(box["score"]),
                    "text_regions": [],
                }
            )

        unassigned_texts = []
        # Chỉ dùng tâm polygon để quyết định text thuộc cell nào.
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
                matched_cell = min(matched_cells, key=_box_area)
                matched_cell["text_regions"].append(text_region)
            else:
                unassigned_texts.append(text_region)

        # Ghép text theo thứ tự đọc trong từng cell.
        for cell in cells:
            cell["text_regions"] = _sort_text_regions(
                cell["text_regions"]
            )
            cell["text"] = " ".join(
                region["text"] for region in cell["text_regions"]
            )
            cell["ppocr_text"] = " ".join(
                region["ppocr_text"]
                for region in cell["text_regions"]
            )

        return cells, unassigned_texts

    # Tìm divider dọc dài nằm gần giữa bảng.
    def _detect_main_vertical_divider(self, image):
        """Tìm đường phân cách dọc chính để tách text dính hai cột."""
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
        # Chỉ giữ các line đủ dài, gần dọc và nằm trong vùng giữa bảng.
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

    # Tách polygon nếu nó vượt qua divider dọc chính.
    def _split_polygon_at_divider(self, polygon, divider_x, image_width):
        """Tách text polygon thành hai vùng ở hai phía divider."""
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

    # Kiểm tra box con có nằm gần như hoàn toàn trong box cha.
    def _is_nested_cell(self, parent_box, child_box):
        """Xác định quan hệ lồng nhau giữa hai cell box."""
        parent_area = _box_area(parent_box)
        child_area = _box_area(child_box)
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

    # Kiểm tra separator dọc hoặc ngang giữa các cell con.
    def _children_have_real_separator(
        self,
        line_mask,
        parent_box,
        child_boxes,
    ):
        """Kiểm tra các cell con có được ngăn bởi đường kẻ thật."""
        parent_x_min, parent_y_min, parent_x_max, parent_y_max = map(
            float,
            parent_box["coordinate"],
        )
        parent_width = parent_x_max - parent_x_min
        parent_height = parent_y_max - parent_y_min

        # So sánh từng cặp cell con để tìm ranh giới có đường kẻ.
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

                # Hai cell cạnh nhau theo chiều ngang cần một line dọc.
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

                # Hai cell cạnh nhau theo chiều dọc cần một line ngang.
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

    # Đo độ liên tục của line quanh vị trí separator dự kiến.
    def _has_continuous_line(
        self,
        line_mask,
        position,
        span_start,
        span_end,
        vertical,
    ):
        """Xác nhận separator bằng tỷ lệ pixel tối liên tục."""
        height, width = line_mask.shape
        span_length = span_end - span_start
        margin = max(2, int(span_length * 0.08))
        radius = max(3, int(min(height, width) * 0.006))

        # Line dọc được đo theo từng cột trong vùng hẹp quanh separator.
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


# Sắp xếp bbox theo từng hàng và từ trái sang phải.
def sort_boxes_left_to_right(boxes):
    """Sắp xếp bbox theo thứ tự đọc trong bảng."""
    return [
        box
        for row in group_boxes_into_rows(boxes)
        for box in row
    ]


# Gom bbox thành các hàng dựa trên vị trí dọc.
def group_boxes_into_rows(boxes, use_vertical_center=False):
    """Gom bbox theo hàng và sắp xếp từ trái sang phải."""
    if not boxes:
        return []

    items = []
    heights = []
    # Chuẩn hóa vị trí dùng để so khớp hàng của từng bbox.
    for box in boxes:
        x_min, y_min, x_max, y_max = map(
            float,
            box["coordinate"],
        )
        row_position = (
            (y_min + y_max) / 2
            if use_vertical_center
            else y_min
        )
        items.append(
            {
                "box": box,
                "top": row_position,
                "left": x_min,
            }
        )
        heights.append(y_max - y_min)

    row_tolerance = max(8.0, float(np.median(heights)) * 0.5)
    rows = []
    # Ghép bbox vào hàng gần nhất trong tolerance cho phép.
    for item in sorted(items, key=lambda value: value["top"]):
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

    return [
        [
            item["box"]
            for item in sorted(
                row["items"],
                key=lambda value: value["left"],
            )
        ]
        for row in sorted(rows, key=lambda value: value["top"])
    ]


# Sắp xếp text region theo tâm dọc để chịu được ảnh bảng bị nghiêng.
def _sort_text_regions(text_regions):
    """Sắp xếp text theo hàng và từ trái sang phải."""
    text_boxes = []
    # Chuyển polygon thành bbox tạm thời để dùng chung logic gom hàng.
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

    ordered_rows = group_boxes_into_rows(
        text_boxes,
        use_vertical_center=True,
    )
    return [
        box["text_region"]
        for row in ordered_rows
        for box in row
    ]


# Tính diện tích bbox phục vụ chọn cell nhỏ nhất và phân xử cell lồng nhau.
def _box_area(box):
    """Tính diện tích bbox của một cell."""
    x_min, y_min, x_max, y_max = map(float, box["coordinate"])
    return (x_max - x_min) * (y_max - y_min)
