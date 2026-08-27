# Mục đích: thử nghiệm trích xuất cặp key-value từ ảnh page 3+.
# Chức năng: OCR, gom text theo hàng, ghép key-value và tạo HTML trực quan.
# Luồng xử lý: ảnh → OCR → gom hàng → nhận diện schema/biểu thức/cột → HTML.
# Cấu hình/parameter: nhận một image_path hoặc --all-page-001.
# Quan hệ: script độc lập, không sử dụng SLANeXt hoặc Page1TableAnalyzer.
# Input/output: đọc ảnh page 3+; ghi ảnh đánh dấu và một file HTML chung.
# Contract: mỗi kết quả có key, value, bbox, metadata và phương pháp ghép.

import argparse
import json
import re
from html import escape
from pathlib import Path

import cv2
import numpy as np
from paddleocr import TextDetection, TextRecognition


ASSIGNMENT_PATTERN = re.compile(
    r"(?P<key>[A-Za-z0-9][A-Za-z0-9_.<>/+-]*)\s*=\s*"
    r"(?P<value>.*?)"
    r"(?=\s+[A-Za-z0-9][A-Za-z0-9_.<>/+-]*\s*=|$)"
)
LEADING_CODE_PATTERN = re.compile(
    r"^(?P<code>(?=[A-Z0-9.]*\d)[A-Z0-9][A-Z0-9.]{2,11})\s+"
    r"(?P<key>.+)$"
)
DOCUMENT_HEADER_PHRASES = (
    "phiếu chỉnh định",
    "trung tâm điều độ",
    "tập đoàn điện lực",
    "số phiếu",
    "tờ số",
)


def merge_boxes(regions):
    """Gộp bbox của nhiều vùng OCR thành một hình chữ nhật."""
    return [
        min(region["box"][0] for region in regions),
        min(region["box"][1] for region in regions),
        max(region["box"][2] for region in regions),
        max(region["box"][3] for region in regions),
    ]


def normalize_text(text):
    """Chuẩn hóa khoảng trắng và chữ thường để so khớp header."""
    return " ".join(text.lower().split())


def join_regions(regions):
    """Ghép nội dung OCR theo thứ tự từ trái sang phải."""
    return " ".join(
        region["text"].strip()
        for region in sorted(regions, key=lambda item: item["box"][0])
        if region["text"].strip()
    )


def is_document_header(text):
    """Kiểm tra text thuộc phần thông tin đầu phiếu."""
    normalized = normalize_text(text)
    return any(phrase in normalized for phrase in DOCUMENT_HEADER_PHRASES)


def estimate_sub_box(region, start_index, end_index):
    """Ước lượng bbox của một đoạn con trong cùng vùng OCR."""
    text_length = max(len(region["text"]), 1)
    x_min, y_min, x_max, y_max = region["box"]
    width = x_max - x_min
    return [
        int(x_min + width * start_index / text_length),
        y_min,
        int(x_min + width * end_index / text_length),
        y_max,
    ]


def group_text_rows(text_regions):
    """Gom vùng OCR theo khoảng cách tâm dọc cố định của toàn ảnh."""
    rows = []
    region_heights = [
        max(region["box"][3] - region["box"][1], 1)
        for region in text_regions
    ]
    median_height = float(np.median(region_heights)) if region_heights else 1
    row_tolerance = max(median_height * 0.65, 6)

    for region in sorted(text_regions, key=lambda item: item["center"][1]):
        matching_row = None
        matching_distance = None

        for row in rows:
            center_distance = abs(region["center"][1] - row["center_y"])
            if center_distance > row_tolerance:
                continue
            if matching_distance is None or center_distance < matching_distance:
                matching_row = row
                matching_distance = center_distance

        if matching_row is None:
            rows.append(
                {
                    "regions": [region],
                    "center_y": region["center"][1],
                }
            )
            continue

        matching_row["regions"].append(region)
        matching_row["center_y"] = float(
            np.median(
                [
                    item["center"][1]
                    for item in matching_row["regions"]
                ]
            )
        )

    rows.sort(key=lambda row: row["center_y"])
    for row in rows:
        row["regions"].sort(key=lambda region: region["box"][0])
    return rows


def keyword_x(region, phrase):
    """Ước lượng tâm ngang của một từ khóa trong vùng OCR."""
    normalized = normalize_text(region["text"])
    index = normalized.find(phrase)
    if index < 0:
        return None

    x_min, _, x_max, _ = region["box"]
    text_length = max(len(normalized), 1)
    phrase_center = index + len(phrase) / 2
    return x_min + (x_max - x_min) * phrase_center / text_length


def find_role_candidate(regions, phrases):
    """Tìm vị trí cột từ danh sách tên header ưu tiên."""
    for phrase in phrases:
        for region in regions:
            position = keyword_x(region, phrase)
            if position is not None:
                return position, phrase
    return None, None


def detect_schema(row):
    """Nhận diện vai trò cột từ một hàng header."""
    regions = row["regions"]
    key_x, key_name = find_role_candidate(
        regions,
        ("item", "parameter", "settings", "function"),
    )
    value_x, value_name = find_role_candidate(
        regions,
        ("new setting", "value", "scope", "setting"),
    )

    if key_x is None or value_x is None or abs(key_x - value_x) < 20:
        return None

    code_x, code_name = find_role_candidate(
        regions,
        ("index", "number", "no."),
    )
    description_x, _ = find_role_candidate(regions, ("description",))
    unit_x, _ = find_role_candidate(regions, ("unit",))
    anchors = {
        "key": key_x,
        "value": value_x,
    }
    if code_x is not None:
        anchors["index" if code_name == "index" else "code"] = code_x
    if description_x is not None:
        anchors["description"] = description_x
    if unit_x is not None:
        anchors["unit"] = unit_x

    return {
        "anchors": anchors,
        "key_header": key_name,
        "value_header": value_name,
    }


def assign_regions_to_schema(regions, schema):
    """Gán từng vùng OCR vào cột header gần nhất."""
    columns = {name: [] for name in schema["anchors"]}
    for region in regions:
        role = min(
            schema["anchors"],
            key=lambda name: abs(
                region["center"][0] - schema["anchors"][name]
            ),
        )
        columns[role].append(region)
    return columns


def split_leading_code(regions):
    """Tách mã đầu hàng khỏi Item trong bố cục không có header."""
    ordered_regions = sorted(regions, key=lambda region: region["box"][0])
    first_text = ordered_regions[0]["text"].strip()

    if (
        len(ordered_regions) > 1
        and re.fullmatch(r"(?=[A-Z0-9.]*\d)[A-Z0-9.]{3,12}", first_text)
    ):
        item_regions = ordered_regions[1:]
        return first_text, join_regions(item_regions), merge_boxes(item_regions)

    full_text = join_regions(ordered_regions)
    match = LEADING_CODE_PATTERN.match(full_text)
    if match is None:
        return None, full_text, merge_boxes(ordered_regions)

    merged_region = {
        "box": merge_boxes(ordered_regions),
        "text": full_text,
    }
    key_box = estimate_sub_box(
        merged_region,
        match.start("key"),
        match.end("key"),
    )
    return match.group("code"), match.group("key").strip(), key_box


def clean_assignment_value(value):
    """Bỏ chú thích ngoặc ở cuối value của biểu thức gán."""
    return re.sub(r"\s+\([^)]*\)\s*$", "", value).strip()


class Page3PlusKeyValueExtractor:
    """Trích xuất cặp key-value trực tiếp từ các vùng OCR."""

    def __init__(self):
        """Khởi tạo detector và recognizer PP-OCRv6 medium."""
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

    def _recognize_text(self, image):
        """Nhận diện nội dung và bbox của toàn bộ text trên ảnh."""
        text_regions = []
        pending_regions = []
        text_crops = []
        image_height, image_width = image.shape[:2]
        detection_results = self.text_detector.predict(
            input=image,
            limit_side_len=3200,
            limit_type="max",
            thresh=0.3,
            box_thresh=0.6,
            unclip_ratio=1.5,
        )

        for detection_result in detection_results:
            for polygon in detection_result["dt_polys"]:
                polygon = np.array(polygon, dtype=np.float32)
                x_min, y_min = map(int, polygon.min(axis=0))
                x_max, y_max = map(int, polygon.max(axis=0))
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(x_max, image_width)
                y_max = min(y_max, image_height)
                text_crop = image[y_min:y_max, x_min:x_max]
                text_crops.append(text_crop)
                pending_regions.append(
                    {
                        "box": [x_min, y_min, x_max, y_max],
                        "center": [
                            float(polygon[:, 0].mean()),
                            float(polygon[:, 1].mean()),
                        ],
                    }
                )

        if not text_crops:
            return text_regions

        recognition_results = self.text_recognizer.predict(
            input=text_crops,
            batch_size=8,
        )
        for pending_region, recognition_result in zip(
            pending_regions,
            recognition_results,
            strict=True,
        ):
            text = recognition_result["rec_text"].strip()
            if not text:
                continue
            text_regions.append(
                {
                    **pending_region,
                    "text": text,
                    "score": float(recognition_result["rec_score"]),
                }
            )

        return text_regions

    def _extract_assignments(self, row, current_group):
        """Trích xuất tất cả biểu thức KEY = VALUE trên một hàng."""
        records = []
        non_assignment_regions = [
            region for region in row["regions"] if "=" not in region["text"]
        ]

        for region in row["regions"]:
            if "=" not in region["text"]:
                continue
            for match in ASSIGNMENT_PATTERN.finditer(region["text"]):
                key = match.group("key").strip()
                value = clean_assignment_value(match.group("value"))
                if not value or not any(character.isalpha() for character in key):
                    continue

                equal_index = region["text"].find("=", match.start(), match.end())
                value_start = equal_index + 1
                key_box = estimate_sub_box(
                    region,
                    match.start("key"),
                    match.end("key"),
                )
                value_box = estimate_sub_box(
                    region,
                    value_start,
                    match.end(),
                )
                metadata = {}
                if current_group:
                    metadata["group"] = current_group
                records.append(
                    {
                        "key": key,
                        "value": value,
                        "key_bbox": key_box,
                        "value_bbox": value_box,
                        "metadata": metadata,
                        "method": "assignment",
                        "center_y": row["center_y"],
                    }
                )

        group_candidates = [
            region
            for region in non_assignment_regions
            if region["box"][0] < min(
                (
                    assignment_region["box"][0]
                    for assignment_region in row["regions"]
                    if "=" in assignment_region["text"]
                ),
                default=0,
            )
            and not region["text"].lstrip().startswith("(")
        ]
        if group_candidates:
            group = join_regions(group_candidates)
            if not is_document_header(group):
                for record in records:
                    record["metadata"]["group"] = group
                current_group = group

        return records, current_group

    def _extract_with_schema(self, row, schema, current_group):
        """Trích xuất một record theo vị trí các cột đã nhận diện."""
        columns = assign_regions_to_schema(row["regions"], schema)
        key_regions = columns.get("key", [])
        value_regions = columns.get("value", [])
        if not key_regions or not value_regions:
            return None

        key = join_regions(key_regions)
        value = join_regions(value_regions)
        if is_document_header(key) or not key or not value:
            return None

        metadata = {}
        for role in ("code", "index", "description", "unit"):
            role_regions = columns.get(role, [])
            if role_regions:
                metadata[role] = join_regions(role_regions)
        if current_group:
            metadata["group"] = current_group

        return {
            "key": key,
            "value": value,
            "key_bbox": merge_boxes(key_regions),
            "value_bbox": merge_boxes(value_regions),
            "metadata": metadata,
            "method": "header_schema",
            "center_y": row["center_y"],
        }

    def _extract_by_gap(self, row, image_width, median_height, current_group):
        """Ghép Item và value theo khoảng trống ngang lớn nhất trên hàng."""
        regions = row["regions"]
        if len(regions) < 2:
            return None

        gaps = [
            regions[index + 1]["box"][0] - regions[index]["box"][2]
            for index in range(len(regions) - 1)
        ]
        split_index = int(np.argmax(gaps))
        if gaps[split_index] < max(40, median_height * 1.5):
            return None

        key_regions = regions[: split_index + 1]
        value_regions = regions[split_index + 1 :]
        if value_regions[0]["center"][0] < image_width * 0.52:
            return None

        code, key, key_box = split_leading_code(key_regions)
        value = join_regions(value_regions)
        if (
            not key
            or not value
            or key.lstrip().startswith("(")
            or is_document_header(key)
        ):
            return None

        metadata = {}
        if code:
            metadata["code"] = code
        if current_group:
            metadata["group"] = current_group

        return {
            "key": key,
            "value": value,
            "key_bbox": key_box,
            "value_bbox": merge_boxes(value_regions),
            "metadata": metadata,
            "method": "column_gap",
            "center_y": row["center_y"],
        }

    def extract(self, image):
        """Trả danh sách cặp key-value và ảnh đánh dấu kết quả."""
        image_width = image.shape[1]
        text_regions = self._recognize_text(image)
        text_regions.sort(
            key=lambda region: (
                region["center"][1],
                region["center"][0],
            )
        )
        rows = group_text_rows(text_regions)
        region_heights = [
            region["box"][3] - region["box"][1]
            for region in text_regions
        ]
        median_height = float(np.median(region_heights)) if region_heights else 1
        records = []
        active_schema = None
        current_group = None
        last_column_record = None

        for row in rows:
            schema = detect_schema(row)
            if schema is not None:
                active_schema = schema
                last_column_record = None
                continue

            assignment_records, current_group = self._extract_assignments(
                row,
                current_group,
            )
            if assignment_records:
                records.extend(assignment_records)
                last_column_record = None
                continue

            record = None
            if active_schema is not None:
                record = self._extract_with_schema(
                    row,
                    active_schema,
                    current_group,
                )
            if record is None and active_schema is None:
                record = self._extract_by_gap(
                    row,
                    image_width,
                    median_height,
                    current_group,
                )

            if record is not None:
                records.append(record)
                last_column_record = record
                continue

            right_regions = [
                region
                for region in row["regions"]
                if region["center"][0] >= image_width * 0.52
            ]
            left_regions = [
                region
                for region in row["regions"]
                if region["center"][0] < image_width * 0.52
            ]
            if (
                right_regions
                and not left_regions
                and last_column_record is not None
                and row["center_y"] - last_column_record["center_y"]
                <= median_height * 2.2
            ):
                last_column_record["value"] += " " + join_regions(right_regions)
                last_column_record["value_bbox"] = merge_boxes(
                    [
                        {"box": last_column_record["value_bbox"]},
                        *right_regions,
                    ]
                )
                last_column_record["center_y"] = row["center_y"]
                continue

            if left_regions and not right_regions:
                group = join_regions(left_regions)
                if (
                    group
                    and not group.lstrip().startswith("(")
                    and not is_document_header(group)
                ):
                    current_group = group

        visualization = image.copy()
        for index, region in enumerate(text_regions, start=1):
            x_min, y_min, x_max, y_max = region["box"]
            cv2.rectangle(
                visualization,
                (x_min, y_min),
                (x_max, y_max),
                (145, 145, 145),
                1,
            )
            cv2.putText(
                visualization,
                f"O{index}:{region['score']:.2f}",
                (x_min, max(y_min - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (100, 100, 100),
                1,
                cv2.LINE_AA,
            )

        for index, record in enumerate(records, start=1):
            key_x1, key_y1, key_x2, key_y2 = record["key_bbox"]
            value_x1, value_y1, value_x2, value_y2 = record["value_bbox"]
            cv2.rectangle(
                visualization,
                (key_x1, key_y1),
                (key_x2, key_y2),
                (34, 197, 94),
                3,
            )
            cv2.rectangle(
                visualization,
                (value_x1, value_y1),
                (value_x2, value_y2),
                (255, 120, 0),
                3,
            )
            cv2.line(
                visualization,
                ((key_x1 + key_x2) // 2, (key_y1 + key_y2) // 2),
                ((value_x1 + value_x2) // 2, (value_y1 + value_y2) // 2),
                (0, 165, 255),
                2,
            )
            cv2.putText(
                visualization,
                str(index),
                (key_x1, max(key_y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        for record in records:
            record.pop("center_y")
        return records, visualization, text_regions


def parse_arguments():
    """Đọc ảnh đơn hoặc chế độ chạy toàn bộ page_001."""
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", nargs="?", type=Path)
    parser.add_argument("--all-page-001", action="store_true")
    arguments = parser.parse_args()

    if arguments.all_page_001 == (arguments.image_path is not None):
        parser.error("Phải truyền image_path hoặc --all-page-001.")
    return arguments


def build_html(sections, total_items):
    """Tạo nội dung HTML chung để kiểm tra ảnh và các cặp đã ghép."""
    return f"""<!doctype html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <title>Phân tích key-value page 3+</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
        section {{ border-top: 2px solid #777; margin-top: 32px; padding-top: 20px; }}
        img {{ border: 1px solid #bbb; display: block; height: auto; max-width: 100%; }}
        table {{ border-collapse: collapse; margin-top: 16px; width: 100%; }}
        th, td {{
            border: 1px solid #999;
            padding: 6px 8px;
            text-align: left;
            vertical-align: top;
        }}
        th {{ background: #eee; }}
        .key {{ color: #16833f; }}
        .value {{ color: #006a9f; }}
        .metadata {{ color: #555; }}
        details {{ margin-top: 16px; }}
        .legend {{ color: #555; }}
    </style>
</head>
<body>
    <h1>Phân tích key-value page 3+</h1>
    <p>Tổng số cặp: {total_items}</p>
    <p class="legend">
        Xám: vùng OCR gốc | Xanh lá: key | Xanh dương: value |
        Cam: liên kết key-value
    </p>
    {''.join(sections)}
</body>
</html>
"""


def main():
    """Chạy OCR, trích xuất key-value và lưu một HTML chung."""
    arguments = parse_arguments()
    if arguments.all_page_001:
        input_directory = Path("./data/pdf/page3plus_png")
        image_paths = sorted(
            input_directory.glob("P_*/page_001.png"),
            key=lambda path: path.parent.name,
        )
        output_directory = Path("./output/table_3plus_key_value/page_001")
    else:
        image_paths = [arguments.image_path]
        output_directory = (
            Path("./output/table_3plus_key_value")
            / arguments.image_path.parent.name
        )

    if not image_paths:
        raise FileNotFoundError("Không tìm thấy ảnh page_001 để xử lý.")

    output_directory.mkdir(parents=True, exist_ok=True)
    extractor = Page3PlusKeyValueExtractor()
    sections = []
    total_items = 0

    for image_index, image_path in enumerate(image_paths, start=1):
        print(f"[{image_index}/{len(image_paths)}] Đang xử lý: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

        records, visualization, text_regions = extractor.extract(image)
        source_name = image_path.parent.name
        output_stem = f"{source_name}_{image_path.stem}"
        visualization_path = output_directory / f"{output_stem}_key_value.png"
        saved = cv2.imwrite(str(visualization_path), visualization)
        if not saved:
            raise RuntimeError(f"Không lưu được ảnh: {visualization_path}")

        table_rows = []
        for record_index, record in enumerate(records, start=1):
            metadata = json.dumps(
                record["metadata"],
                ensure_ascii=False,
            )
            table_rows.append(
                "<tr>"
                f"<td>{record_index}</td>"
                f"<td class=\"key\">{escape(record['key'])}</td>"
                f"<td class=\"value\">{escape(record['value'])}</td>"
                f"<td>{escape(metadata)}</td>"
                f"<td>{escape(record['method'])}</td>"
                "</tr>"
            )

        ocr_rows = []
        for region_index, region in enumerate(text_regions, start=1):
            bbox = json.dumps(region["box"])
            ocr_rows.append(
                "<tr>"
                f"<td>O{region_index}</td>"
                f"<td>{escape(region['text'])}</td>"
                f"<td>{region['score']:.4f}</td>"
                f"<td>{escape(bbox)}</td>"
                "</tr>"
            )

        sections.append(
            f"""
<section>
    <h2>{escape(source_name)} / {escape(image_path.name)}</h2>
    <p class="metadata">
        OCR regions: {len(text_regions)} | Cặp key-value: {len(records)}
    </p>
    <img src="{escape(visualization_path.name)}" alt="Kết quả {escape(output_stem)}">
    <table>
        <thead>
            <tr>
                <th>#</th><th>Key</th><th>Value</th>
                <th>Metadata</th><th>Method</th>
            </tr>
        </thead>
        <tbody>{''.join(table_rows)}</tbody>
    </table>
    <details>
        <summary>Toàn bộ vùng OCR ({len(text_regions)})</summary>
        <table>
            <thead>
                <tr><th>#</th><th>OCR text</th><th>Score</th><th>Bbox</th></tr>
            </thead>
            <tbody>{''.join(ocr_rows)}</tbody>
        </table>
    </details>
</section>
"""
        )
        total_items += len(records)

    html_path = output_directory / "key_value_analysis.html"
    html_path.write_text(
        build_html(sections, total_items),
        encoding="utf-8",
    )
    print(f"HTML key-value: {html_path}")
    print(f"Đã xử lý {len(image_paths)} ảnh, tìm thấy {total_items} cặp.")


if __name__ == "__main__":
    main()
