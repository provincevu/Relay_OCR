# Mục đích: tách logic trực quan hóa cho script debug bảng page 1.
# Chức năng: dựng HTML có kết quả hai recognizer, vẽ bbox bảng và lưu ảnh crop bảng.
# Luồng xử lý: kết quả cell/table → ảnh hoặc HTML debug.
# Cấu hình/parameter: dùng output_dir và crop padding của analyzer.
# Quan hệ: được Page1TableAnalyzer kế thừa sau bước detection và OCR.
# Input/output: nhận ảnh và kết quả phân tích; trả HTML hoặc đường dẫn crop.
# Contract: giữ nguyên tên file crop và nội dung section HTML hiện có.

import base64
import re
import unicodedata
from html import escape

import cv2


TOKEN_PATTERN = re.compile(
    r"[^\W_]+(?:[./:+_`-][^\W_]+)*|`",
    re.UNICODE,
)
CONFUSABLE_TEXT_PAIRS = (
    ("m", "rn"),
    ("m", "nn"),
    ("w", "vv"),
    ("I", "l"),
    ("I", "1"),
    ("I", "`"),
    ("1", "l"),
    ("1", "7"),
    ("0", "O"),
    ("0", "o"),
    ("0", "D"),
    ("O", "Q"),
    ("O", "D"),
    ("o", "a"),
    ("o", "e"),
    ("5", "S"),
    ("5", "s"),
    ("S", "8"),
    ("2", "Z"),
    ("2", "z"),
    ("Z", "7"),
    ("8", "B"),
    ("8", "3"),
    ("B", "R"),
    ("6", "G"),
    ("6", "b"),
    ("G", "C"),
    ("9", "g"),
    ("9", "q"),
    ("q", "g"),
    ("c", "e"),
    ("c", "o"),
    ("e", "o"),
    ("a", "e"),
    ("w", "v"),
    ("u", "v"),
    ("v", "y"),
    ("h", "b"),
    ("h", "n"),
    ("b", "6"),
    ("n", "r"),
    ("r", "v"),
    ("f", "t"),
    ("t", "l"),
)
CONFUSABLE_TRANSFORMS = tuple(
    transform
    for pair in CONFUSABLE_TEXT_PAIRS
    for transform in (pair, tuple(reversed(pair)))
)


class DebugTableVisualizationMixin:
    """Cung cấp logic trực quan hóa cho analyzer debug."""

    def _build_html_section(
        self,
        image_name,
        table_index,
        crop_path,
        crop,
        cells,
        unofficial_cells,
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
            text_shapes.append(
                f'<polygon class="text-box" points="{points}" />'
            )

        cell_rows = []
        displayed_cells = self._sort_boxes_left_to_right(
            cells + unofficial_cells
        )
        for cell in displayed_cells:
            coordinate = ", ".join(
                f"{value:.1f}" for value in cell["coordinate"]
            )
            if cell["index"] is None:
                cell_label = "Không chính thức"
                row_class = ' class="unofficial-cell"'
            else:
                cell_label = f"C{cell['index']:02d}"
                row_class = ""
            cell_rows.append(
                f"<tr{row_class}>"
                f"<td>{cell_label}</td>"
                f"<td>{coordinate}</td>"
                f'<td>{cell["score"]:.3f}</td>'
                f'<td>{escape(cell["text"])}</td>'
                f'<td>{escape(cell["ppocr_text"])}</td>'
                "</tr>"
            )

        unassigned_html = "<br>".join(
            "VietOCR: "
            f'{escape(region["text"])} | '
            "PP-OCRv6: "
            f'{escape(region["ppocr_text"])}'
            for region in unassigned_texts
        )
        comparison_html = self._build_token_comparison(text_regions)
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
<thead><tr><th>Cell</th><th>Bounding box</th><th>Score</th><th>VietOCR</th><th>PP-OCRv6 medium</th></tr></thead>
<tbody>{"".join(cell_rows)}</tbody>
</table>
<p><strong>Text chưa gán vào cell:</strong> {unassigned_html}</p>
<h3>Khác biệt thuộc nhóm ký tự gần giống nhau</h3>
<p>Chỉ hiển thị token khác nhau, không có dấu tiếng Việt và khớp các nhóm dễ nhầm đã cấu hình.</p>
{comparison_html}
<h3>Bảng chuẩn hóa: 7 hàng × 2 cột vật lý</h3>
<p>Field bổ sung như “Phiên bản” được giữ bên trong cột phải dưới dạng field logic.</p>
{normalized_table_html}
</section>
"""

    def _build_token_comparison(self, text_regions):
        """Chỉ hiển thị token khác nhau do các ký tự gần giống đã cấu hình."""
        rows = []

        for region_index, region in enumerate(text_regions, start=1):
            vietocr_tokens = TOKEN_PATTERN.findall(region["text"])
            ppocr_tokens = TOKEN_PATTERN.findall(region["ppocr_text"])

            for token_index, vietocr_token in enumerate(vietocr_tokens):
                if self._has_vietnamese_mark(vietocr_token):
                    continue

                ppocr_token = (
                    ppocr_tokens[token_index]
                    if token_index < len(ppocr_tokens)
                    else ""
                )
                if vietocr_token == ppocr_token:
                    continue

                confusable_changes = self._find_confusable_changes(
                    vietocr_token,
                    ppocr_token,
                )
                if confusable_changes is None:
                    continue

                changes_text = ", ".join(
                    f"{source} ↔ {target}"
                    for source, target in confusable_changes
                )

                rows.append(
                    "<tr>"
                    f"<td>T{region_index:02d}</td>"
                    f"<td>{escape(vietocr_token)}</td>"
                    f"<td>{escape(ppocr_token)}</td>"
                    f"<td>{escape(changes_text)}</td>"
                    '<td class="comparison-diff">Khác</td>'
                    "</tr>"
                )

        difference_count = len(rows)
        if not rows:
            rows.append(
                '<tr><td colspan="5">Không có khác biệt thuộc các nhóm ký tự gần giống.</td></tr>'
            )

        return (
            f"<p>Số token khác nhau: {difference_count}</p>"
            "<table>"
            "<thead><tr><th>Vùng text</th><th>VietOCR</th>"
            "<th>PP-OCRv6 medium</th><th>Cặp dễ nhầm</th>"
            "<th>Kết quả</th></tr></thead>"
            f'<tbody>{"".join(rows)}</tbody>'
            "</table>"
        )

    def _find_confusable_changes(self, source_text, target_text):
        """Trả các phép đổi ký tự gần giống nếu chúng giải thích toàn bộ sai khác."""
        compared_states = {}

        def compare(source_index, target_index):
            state = (source_index, target_index)
            if state in compared_states:
                return compared_states[state]
            if source_index == len(source_text) and target_index == len(
                target_text
            ):
                return []

            if (
                source_index < len(source_text)
                and target_index < len(target_text)
                and source_text[source_index] == target_text[target_index]
            ):
                remaining_changes = compare(
                    source_index + 1,
                    target_index + 1,
                )
                if remaining_changes is not None:
                    compared_states[state] = remaining_changes
                    return remaining_changes

            for source, target in CONFUSABLE_TRANSFORMS:
                if not source_text.startswith(source, source_index):
                    continue
                if not target_text.startswith(target, target_index):
                    continue

                remaining_changes = compare(
                    source_index + len(source),
                    target_index + len(target),
                )
                if remaining_changes is not None:
                    changes = [(source, target), *remaining_changes]
                    compared_states[state] = changes
                    return changes

            compared_states[state] = None
            return None

        changes = compare(0, 0)
        return changes or None

    def _has_vietnamese_mark(self, token):
        """Kiểm tra token có chữ tiếng Việt mang dấu hoặc chữ đ hay không."""
        if "đ" in token.lower():
            return True

        normalized_token = unicodedata.normalize("NFD", token)
        return any(
            unicodedata.combining(character)
            for character in normalized_token
        )

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
        crop_dir = self.output_dir / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        table_crops = []
        height, width = image.shape[:2]

        for index, box in enumerate(table_boxes, start=1):
            x_min, y_min, x_max, y_max = map(int, box["coordinate"])
            x_min = max(x_min - self.crop_padding_left, 0)
            y_min = max(y_min - self.crop_padding_top, 0)
            x_max = min(x_max + self.crop_padding_right, width)
            y_max = min(y_max + self.crop_padding_bottom, height)
            crop = image[y_min:y_max, x_min:x_max]
            crop_path = crop_dir / f"{image_path.stem}_table_{index:02d}.png"
            cv2.imwrite(str(crop_path), crop)
            table_crops.append((index, crop_path, crop))

        return table_crops
