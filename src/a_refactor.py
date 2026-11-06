
import cv2
import numpy as np


class TableCellProcessor:
    """Nhận dạng text và xử lý quan hệ giữa cell với các vùng OCR."""

    def __init__(self, text_detector, text_recognitor, ppocr_recognitor):
        """Lưu các model dùng chung để tránh khởi tạo lại khi xử lý từng ảnh."""
        self.text_detector = text_detector
        self.text_recognitor = text_recognitor
        self.ppocr_recognitor = ppocr_recognitor

    def recognize_text(self, image):
        """Phát hiện vùng text, tách vùng qua divider nếu cần rồi nhận dạng."""
        text_regions = []
        height, width = image.shape[:2]
        # Divider giúp tách một box OCR vô tình phủ qua hai cột của bảng. vì có thể OCR sẽ nhận dạng nhầm text của hai cột vào cùng một box.
        divider_x = self._detect_main_vertical_divider(image)

        results = self.text_detector.detect(image)

        for result in results:
            for polygon in result["dt_polys"]:
                polygon = np.array(polygon, dtype=np.float32)

                split_polygons = self._split_polygon_at_divider(
                    polygon, divider_x, width
                )

                for polygon in split_polygons:
                    x_min, y_min = polygon.min(axis=0).astype(int)
                    x_max, y_max = polygon.max(axis=0).astype(int)

                    x_min = max(x_min, 0)
                    y_min = max(y_min, 0)
                    x_max = min(x_max, width)
                    y_max = min(y_max, height)

                    # Cắt vùng text từ ảnh gốc để nhận dạng
                    text_crop = image[y_min:y_max, x_min:x_max]

                    text = self.text_recognitor.recognize(text_crop)
                    ppocr = self.ppocr_recognitor.predict(
                        text_crop, batch_size=1
                    )[0]

                    text_regions.append({
                        "polygon": polygon.tolist(),
                        "center": [
                            float(polygon[:, 0].mean()),
                            float(polygon[:, 1].mean()),
                        ],
                        "text": text,
                        "ppocr_text": ppocr["rec_text"],
                    })

        return text_regions

    def resolve_nested_cell_boxes(self, image, cell_boxes):
        """
            Chọn box cell hợp lý khi detector trả cả box cha và box con.
            +-----------------------+
            |       Box cha         |
            |  +------+  +------+   |
            |  | Con1 |  | Con2 |   |
            |  +------+  +------+   |
            +-----------------------+
            sắp xếp các box từ lớn nhất đến nhỏ nhất, để box lớn được xử lý trước với vai trò là box cha.
            Chọn một box lớn làm parent.
            Tìm các box nhỏ nằm bên trong nó làm children.
            Kiểm tra ảnh có đường kẻ thật phân cách các box con hay không:
                - Nếu có đường phân cách thật: giữ các box con, loại bỏ box cha.
                - Nếu không có đường phân cách thật: giữ box cha, loại bỏ các box con vì chúng có thể là nhận diện sai hoặc box lồng giả.
        
        """
        # Otsu tạo line mask để phân biệt cell con thật với box lồng giả. (vì box lồng giả thường không có đường kẻ phân cách thật giữa các cell con)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, line_mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        removed = set()
        # Xét box lớn trước để quyết định box cha có nên bị thay bằng các con.
        indexes = sorted(
            range(len(cell_boxes)),
            key=lambda i: _box_area(cell_boxes[i]),
            reverse=True,
        )

        for parent_index in indexes:
            if parent_index in removed:
                continue

            parent = cell_boxes[parent_index]

            children = [
                i for i in indexes
                if i != parent_index
                and i not in removed
                and self._is_nested_cell(parent, cell_boxes[i])
            ]

            if not children:
                continue

            child_boxes = [cell_boxes[i] for i in children]

            # Chỉ giữ cell con khi ảnh có đường phân cách thật giữa chúng.
            if self._children_have_real_separator(
                line_mask, parent, child_boxes
            ):
                removed.add(parent_index)
            else:
                removed.update(children)

        return [
            box for i, box in enumerate(cell_boxes)
            if i not in removed
        ]

    def assign_text_to_cells(self, cell_boxes, text_regions):
        """Gán mỗi vùng OCR vào cell nhỏ nhất chứa tâm vùng đó."""
        cells = []

        for box in sort_boxes_left_to_right(cell_boxes):
            cells.append({
                "coordinate": [float(v) for v in box["coordinate"]],
                "score": float(box["score"]),
                "text_regions": [],
            })

        unassigned = []

        for region in text_regions:
            center_x, center_y = region["center"]

            matched = [
                cell for cell in cells
                if cell["coordinate"][0] <= center_x <= cell["coordinate"][2]
                and cell["coordinate"][1] <= center_y <= cell["coordinate"][3]
            ]

            if matched:
                # Khi box lồng nhau, cell nhỏ nhất là nơi chứa text chính xác nhất.
                cell = min(matched, key=_box_area)
                cell["text_regions"].append(region)
            else:
                unassigned.append(region)

        for cell in cells:
            regions = _sort_text_regions(cell["text_regions"])
            cell["text_regions"] = regions
            cell["text"] = " ".join(r["text"] for r in regions)
            cell["ppocr_text"] = " ".join(
                r["ppocr_text"] for r in regions
            )

        return cells, unassigned

    def _detect_main_vertical_divider(self, image):
        """Tìm đường dọc dài gần giữa bảng để tách box OCR đi qua hai cột."""
        _, image_width = image.shape[:2]
        # Giới hạn chiều rộng giúp Hough ổn định và tránh xử lý ảnh rất lớn.
        scale = min(1.0, 600 / image_width)

        resized = cv2.resize(
            image, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA,
        )

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100, apertureSize=3)
        height, width = gray.shape

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 1800,
            threshold=max(20, int(height * 0.12)),
            minLineLength=int(height * 0.25),
            maxLineGap=int(height * 0.2),
        )

        if lines is None:
            return None

        candidates = []

        # Chỉ nhận đường gần thẳng đứng, đủ dài và nằm trong vùng giữa bảng.
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            vertical_span = abs(int(y2) - int(y1))
            horizontal_span = abs(int(x2) - int(x1))
            center_x = (int(x1) + int(x2)) / 2

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
        """Chia polygon qua divider khi hai phần tạo ra đủ rộng để nhận dạng."""
        if divider_x is None:
            return [polygon]

        x_min, y_min = polygon.min(axis=0)
        x_max, y_max = polygon.max(axis=0)

        minimum_width = max(20, int(image_width * 0.02))

        # Tránh tạo mảnh quá hẹp: chúng thường là lỗi cắt và OCR không hữu ích.
        if (
            divider_x - x_min < minimum_width
            or x_max - divider_x < minimum_width
        ):
            return [polygon]

        left = np.array([
            [x_min, y_min],
            [divider_x - 1, y_min],
            [divider_x - 1, y_max],
            [x_min, y_max],
        ], dtype=np.float32)

        right = np.array([
            [divider_x + 1, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [divider_x + 1, y_max],
        ], dtype=np.float32)

        return [left, right]

    def _is_nested_cell(self, parent_box, child_box):
        """Kiểm tra box con có nằm gần như hoàn toàn bên trong box cha không."""
        parent_area = _box_area(parent_box)
        child_area = _box_area(child_box)

        # Box gần bằng box cha không được xem là cell con.
        if child_area > parent_area * 0.85:
            return False

        px1, py1, px2, py2 = map(float, parent_box["coordinate"])
        cx1, cy1, cx2, cy2 = map(float, child_box["coordinate"])

        intersection_width = max(
            0.0, min(px2, cx2) - max(px1, cx1)
        )
        intersection_height = max(
            0.0, min(py2, cy2) - max(py1, cy1)
        )

        intersection_area = intersection_width * intersection_height
        return intersection_area / child_area >= 0.9

    def _children_have_real_separator(
        self, line_mask, parent_box, child_boxes
    ):
        """Xác minh cell con có đường kẻ chia thật trong ảnh hay chỉ là box giả."""
        px1, py1, px2, py2 = map(float, parent_box["coordinate"])
        parent_width = px2 - px1
        parent_height = py2 - py1

        for i, first_box in enumerate(child_boxes):
            first = list(map(float, first_box["coordinate"]))

            for second_box in child_boxes[i + 1:]:
                second = list(map(float, second_box["coordinate"]))

                left, right = sorted(
                    [first, second], key=lambda box: box[0]
                )

                vertical_overlap = max(
                    0.0, min(left[3], right[3]) - max(left[1], right[1])
                )
                horizontal_span = max(left[2], right[2]) - min(left[0], right[0])
                horizontal_gap = right[0] - left[2]

                # Hai cell cạnh nhau cần gần chạm, chồng dọc đủ nhiều và có nét dọc.
                if (
                    vertical_overlap >= parent_height * 0.65
                    and horizontal_span >= parent_width * 0.8
                    and horizontal_gap <= parent_width * 0.05
                ):
                    separator_x = (left[2] + right[0]) / 2

                    if self._has_continuous_line(
                        line_mask,
                        separator_x,
                        max(left[1], right[1]),
                        min(left[3], right[3]),
                        vertical=True,
                    ):
                        return True

                top, bottom = sorted(
                    [first, second], key=lambda box: box[1]
                )

                horizontal_overlap = max(
                    0.0, min(top[2], bottom[2]) - max(top[0], bottom[0])
                )
                vertical_span = max(top[3], bottom[3]) - min(top[1], bottom[1])
                vertical_gap = bottom[1] - top[3]

                # Tương tự cho hai cell xếp trên-dưới, dùng nét ngang làm bằng chứng.
                if (
                    horizontal_overlap >= parent_width * 0.65
                    and vertical_span >= parent_height * 0.8
                    and vertical_gap <= parent_height * 0.05
                ):
                    separator_y = (top[3] + bottom[1]) / 2

                    if self._has_continuous_line(
                        line_mask,
                        separator_y,
                        max(top[0], bottom[0]),
                        min(top[2], bottom[2]),
                        vertical=False,
                    ):
                        return True

        return False

    def _has_continuous_line(
        self, line_mask, position, span_start, span_end, vertical
    ):
        """Đo tỷ lệ nét liên tục quanh vị trí separator theo một trục."""
        height, width = line_mask.shape
        span_length = span_end - span_start
        # Bỏ vùng gần góc cell để đường viền không bị nhầm thành separator.
        margin = max(2, int(span_length * 0.08))
        radius = max(3, int(min(height, width) * 0.006))

        if vertical:
            y1 = max(0, int(round(span_start)) + margin)
            y2 = min(height, int(round(span_end)) - margin)
            x1 = max(0, int(round(position)) - radius)
            x2 = min(width, int(round(position)) + radius + 1)
            region = line_mask[y1:y2, x1:x2]

            if region.size == 0:
                return False

            # Chọn cột có tỷ lệ pixel mực cao nhất dọc theo vùng kiểm tra.
            continuity = np.max(np.mean(region > 0, axis=0))

        else:
            x1 = max(0, int(round(span_start)) + margin)
            x2 = min(width, int(round(span_end)) - margin)
            y1 = max(0, int(round(position)) - radius)
            y2 = min(height, int(round(position)) + radius + 1)
            region = line_mask[y1:y2, x1:x2]

            if region.size == 0:
                return False

            # Với đường ngang, đo tương tự theo từng hàng pixel.
            continuity = np.max(np.mean(region > 0, axis=1))

        return continuity >= 0.7


def sort_boxes_left_to_right(boxes):
    """Sắp xếp box theo hàng từ trên xuống dưới, rồi trái sang phải."""
    return [
        box
        for row in group_boxes_into_rows(boxes)
        for box in row
    ]


def group_boxes_into_rows(boxes, use_vertical_center=False):
    """Gom box có vị trí dọc gần nhau thành hàng đọc tự nhiên."""
    if not boxes:
        return []

    items = []
    heights = []

    for box in boxes:
        x1, y1, x2, y2 = map(float, box["coordinate"])

        items.append({
            "box": box,
            "top": (y1 + y2) / 2 if use_vertical_center else y1,
            "left": x1,
        })

        heights.append(y2 - y1)

    # Dung sai theo chiều cao cell để gom cùng hàng dù box lệch nhẹ theo y.
    tolerance = max(8.0, float(np.median(heights)) * 0.5)
    rows = []

    for item in sorted(items, key=lambda x: x["top"]):
        # Gán vào hàng gần nhất trong dung sai, tránh tạo hàng mới vì nhiễu nhỏ.
        matched = [
            row for row in rows
            if abs(item["top"] - row["top"]) <= tolerance
        ]

        if matched:
            row = min(
                matched,
                key=lambda x: abs(item["top"] - x["top"]),
            )
            row["items"].append(item)
            row["top"] = sum(
                x["top"] for x in row["items"]
            ) / len(row["items"])
        else:
            rows.append({
                "top": item["top"],
                "items": [item],
            })

    return [
        [
            item["box"]
            for item in sorted(row["items"], key=lambda x: x["left"])
        ]
        for row in sorted(rows, key=lambda x: x["top"])
    ]


def _sort_text_regions(text_regions):
    """Sắp xếp vùng OCR theo hàng và vị trí để ghép text đúng thứ tự đọc."""
    boxes = []

    for region in text_regions:
        polygon = np.array(region["polygon"], dtype=np.float32)
        x_min, y_min = polygon.min(axis=0)
        x_max, y_max = polygon.max(axis=0)

        boxes.append({
            "coordinate": [x_min, y_min, x_max, y_max],
            "text_region": region,
        })

    rows = group_boxes_into_rows(boxes, use_vertical_center=True)

    return [
        box["text_region"]
        for row in rows
        for box in row
    ]


def _box_area(box):
    """Tính diện tích bbox để chọn box nhỏ nhất hoặc kiểm tra box lồng nhau."""
    x1, y1, x2, y2 = map(float, box["coordinate"])
    return (x2 - x1) * (y2 - y1)
