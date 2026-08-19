# Mục đích: cung cấp utility xử lý cell và text của TABLE 01.
# Chức năng: OCR text, sắp xếp bbox và gán text vào cell.
# Luồng xử lý: bbox/text polygon → gom hàng → gán theo tâm polygon.
# Cấu hình/parameter: không giữ cấu hình; nhận dependency qua tham số.
# Quan hệ: được Table1Analysis và TableFieldExtractor sử dụng.
# Input/output: nhận ảnh hoặc danh sách dict; trả danh sách dict chuẩn hóa.
# Contract: bbox dùng khóa coordinate theo dạng [xmin, ymin, xmax, ymax].

import numpy as np


def recognize_text(image, detector, recognitor):
    """Nhận dạng text và giữ polygon để gán vào cell."""
    text_regions = []
    height, width = image.shape[:2]
    detection_results = detector.detect(image)

    for result in detection_results:
        for polygon in result["dt_polys"]:
            polygon = np.array(polygon, dtype=np.float32)
            x_min, y_min = polygon.min(axis=0).astype(int)
            x_max, y_max = polygon.max(axis=0).astype(int)
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, width)
            y_max = min(y_max, height)
            text = recognitor.recognize(
                image[y_min:y_max, x_min:x_max]
            )
            text_regions.append(
                {
                    "polygon": polygon.tolist(),
                    "center": [
                        float(polygon[:, 0].mean()),
                        float(polygon[:, 1].mean()),
                    ],
                    "text": text,
                }
            )

    return text_regions


def assign_text_to_cells(cell_boxes, text_regions):
    """Gán text vào cell chứa tâm polygon text."""
    cells = []
    sorted_boxes = sort_boxes_left_to_right(cell_boxes)
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
            and cell["coordinate"][1] <= center_y <= cell["coordinate"][3]
        ]
        if matched_cells:
            matched_cell = min(
                matched_cells,
                key=_box_area,
            )
            matched_cell["text_regions"].append(text_region)
        else:
            unassigned_texts.append(text_region)

    for cell in cells:
        cell["text_regions"] = _sort_text_regions(cell["text_regions"])
        cell["text"] = " ".join(
            region["text"] for region in cell["text_regions"]
        )
        del cell["text_regions"]

    return cells, unassigned_texts


def sort_boxes_left_to_right(boxes):
    """Sắp xếp bbox theo từng hàng, từ trái sang phải."""
    return [
        box
        for row in group_boxes_into_rows(boxes)
        for box in row
    ]


def group_boxes_into_rows(boxes):
    """Gom bbox theo hàng và sắp xếp từ trái sang phải."""
    if not boxes:
        return []

    items = []
    heights = []
    for box in boxes:
        x_min, y_min, x_max, y_max = map(
            float,
            box["coordinate"],
        )
        items.append({"box": box, "top": y_min, "left": x_min})
        heights.append(y_max - y_min)

    row_tolerance = max(8.0, float(np.median(heights)) * 0.5)
    rows = []
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


def _sort_text_regions(text_regions):
    """Sắp xếp text theo hàng và từ trái sang phải."""
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

    sorted_boxes = sort_boxes_left_to_right(text_boxes)
    return [box["text_region"] for box in sorted_boxes]


def _box_area(box):
    """Tính diện tích bbox của một cell."""
    x_min, y_min, x_max, y_max = box["coordinate"]
    return (x_max - x_min) * (y_max - y_min)
