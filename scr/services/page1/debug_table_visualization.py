# Mục đích: tách logic trực quan hóa cho script debug bảng page 1.
# Chức năng: dựng HTML, vẽ bbox bảng và lưu ảnh crop bảng.
# Luồng xử lý: kết quả cell/table → ảnh hoặc HTML debug.
# Cấu hình/parameter: dùng output_dir và crop padding của analyzer.
# Quan hệ: được Page1TableAnalyzer kế thừa sau bước detection và OCR.
# Input/output: nhận ảnh và kết quả phân tích; trả HTML hoặc đường dẫn crop.
# Contract: giữ nguyên tên file crop và nội dung section HTML hiện có.

import base64
from html import escape

import cv2


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
