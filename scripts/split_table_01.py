# Mục đích: Tách TABLE 01 từ các ảnh page 1 để giảm kích thước ảnh đầu vào.
# Chức năng: Nhận diện bảng đầu tiên trong từng ảnh PNG và lưu ảnh crop.
# Luồng xử lý: thư mục ảnh → layout detection → chọn bảng đầu tiên → lưu crop.
# Cấu hình/parameter: dùng model layout bảng trên CPU; không thêm padding vào crop.
# Quan hệ: dùng LayoutDetection của PaddleOCR.
# Input/output: đọc `data/pdf/page1_png`; ghi vào `output/table_01_crops`.
# Contract: tên crop có dạng `{tên_ảnh}_table_01.png`; ảnh không có bảng sẽ được bỏ qua.

from pathlib import Path

import cv2
from paddleocr import LayoutDetection


class Table01Cropper:
    """Nhận diện và lưu TABLE 01 từ một thư mục ảnh."""

    def __init__(
        self,
        input_dir="data/pdf/page1_png",
        output_dir="output/table_01_crops",
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.layout_detector = LayoutDetection(
            model_name="PicoDet_layout_1x_table",  # Model nhận diện layout bảng.
            device="cpu",  # Chạy trên CPU.
            engine="paddle_static",
            enable_mkldnn=False,
            cpu_threads=4,  # Số luồng CPU cho inference.
        )

    def process_folder(self):
        """Xử lý lần lượt các ảnh PNG trong thư mục input."""
        image_files = sorted(self.input_dir.glob("*.png"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for image_path in image_files:
            self.crop_table_01(image_path)

    def crop_table_01(self, image_path):
        """Nhận diện bảng đầu tiên và lưu crop nếu bảng tồn tại."""
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        results = self.layout_detector.predict(
            str(image_path),
            batch_size=1,
            layout_nms=True,
        )

        table_boxes = []
        for result in results:
            table_boxes.extend(
                box
                for box in result["boxes"]
                if box["label"].lower() == "table"
            )

        table_boxes = self._sort_boxes_left_to_right(table_boxes)
        if not table_boxes:
            print(f"Không tìm thấy TABLE 01: {image_path.name}")
            return

        x_min, y_min, x_max, y_max = map(
            int,
            table_boxes[0]["coordinate"],
        )
        crop = image[y_min:y_max, x_min:x_max]
        output_path = self.output_dir / f"{image_path.stem}_table_01.png"
        cv2.imwrite(str(output_path), crop)
        print(f"Đã lưu: {output_path}")

    def _sort_boxes_left_to_right(self, boxes):
        """Sắp xếp các bảng theo hàng từ trên xuống dưới và trái sang phải."""
        return sorted(
            boxes,
            key=lambda box: (
                float(box["coordinate"][1]),
                float(box["coordinate"][0]),
            ),
        )


if __name__ == "__main__":
    Table01Cropper().process_folder()
