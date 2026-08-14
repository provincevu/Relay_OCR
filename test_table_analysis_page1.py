from pathlib import Path

import cv2
from paddleocr import LayoutDetection

# Mục đích: debug nhận diện các vùng bảng trên ảnh page 1.
# Chức năng: dùng PicoDet_layout_1x_table để phát hiện bbox có label table.
# Luồng xử lý: đọc ảnh page 1 → LayoutDetection → lọc bbox table → lưu kết quả và ảnh trực quan.
# Cấu hình/parameter: dùng CPU, paddle_static, layout_nms=True và các tham số theo use.txt.
# Quan hệ: script debug độc lập; chưa phân tích cell, row, column hoặc nội dung bảng.
# Input/output: nhận ảnh data/image/P_001.png; xuất ảnh/json gốc, ảnh trực quan và crop từng bảng vào output/layout/.
# Contract: mỗi bbox table có label, score và coordinate dạng [xmin, ymin, xmax, ymax].


class Page1TableAnalyzer:
    """Phân tích và trực quan hóa các vùng bảng trên ảnh page 1."""

    def __init__(self, output_dir=Path("./output/layout")):
        self.output_dir = Path(output_dir)
        self.layout_detector = LayoutDetection(
            model_name="PicoDet_layout_1x_table",
            device="cpu",
            engine="paddle_static",
            enable_mkldnn=False,
            cpu_threads=2,
        )

    def detect_tables(self, image_path):
        """Nhận diện các bbox table và lưu kết quả debug trực quan."""
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        results = self.layout_detector.predict(
            str(image_path),
            batch_size=1,
            layout_nms=True,
        )
        table_boxes = []

        for result in results:
            result.print()
            result.save_to_img(str(self.output_dir / "image"))
            table_boxes.extend(
                box for box in result["boxes"] if box["label"] == "table"
            )

        visual_image = self._draw_tables(image, table_boxes)
        visual_path = self.output_dir / "visual" / f"{image_path.stem}_tables.png"
        visual_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(visual_path), visual_image)
        self._save_table_crops(image, image_path, table_boxes)

        print(f"Tìm thấy {len(table_boxes)} bảng")
        print(f"Ảnh trực quan: {visual_path}")
        return table_boxes

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
        """Lưu crop từng vùng table để kiểm tra riêng."""
        crop_dir = self.output_dir / "crops" / image_path.stem
        crop_dir.mkdir(parents=True, exist_ok=True)

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


if __name__ == "__main__":
    analyzer = Page1TableAnalyzer()
    analyzer.detect_tables("./data/image/P_001.png")
