# Mục đích: cung cấp service phân tích TABLE 01 trên page 1.
# Chức năng: nhận diện bảng, cell, text và trả field value cùng metadata cell.
# Luồng xử lý: ảnh BGR → TABLE 01 → cell → VietOCR/PP-OCRv6 → field output.
# Cấu hình/parameter: nhận config và model text dùng chung từ OCRService.
# Quan hệ: được OCRService gọi; phối hợp TableCellProcessor và TableFieldExtractor.
# Input/output: analyze() nhận ảnh numpy BGR và trả fields cùng table trong bộ nhớ.
# Contract: không ghi file; coordinate field là tọa độ tuyệt đối trên ảnh đầu vào.

import json
from pathlib import Path

from paddleocr import LayoutDetection, TableCellsDetection, TextRecognition

from .utils.table_cell_utils import (
    TableCellProcessor,
    sort_boxes_left_to_right,
)
from .utils.table_field_extractor import TableFieldExtractor


class Table1Analysis:
    """Phân tích TABLE 01 và trả dữ liệu field đã chuẩn hóa."""

    # Khởi tạo các model riêng của TABLE 01 và dùng lại model OCR của service.
    def __init__(self, config, text_detector, text_recognitor):
        """Khởi tạo pipeline phân tích TABLE 01."""
        self.config = config
        field_config_path = Path(
            config.get(
                "table_01_fields_path",
                "./config/table_01_fields.json",  # Danh sách field TABLE 01
            )
        )
        field_config = json.loads(
            field_config_path.read_text(encoding="utf-8")
        )
        self.field_extractor = TableFieldExtractor(field_config["fields"])
        self.layout_detector = LayoutDetection(
            model_name=config.get(
                "table_layout_model_name",
                "PicoDet_layout_1x_table",  # Model detection bảng
            ),
            device=config.get("device", "cpu"),  # Thiết bị inference
            engine=config.get(
                "engine",
                "paddle_static",  # Engine inference
            ),
            cpu_threads=config.get("cpu_threads", 2),  # Số luồng CPU
            enable_mkldnn=config.get(
                "enable_mkldnn",
                False,  # Bật hoặc tắt MKL-DNN
            ),
        )
        self.cell_detector = TableCellsDetection(
            model_name=config.get(
                "table_cell_model_name",
                "RT-DETR-L_wired_table_cell_det",  # Model detection cell
            ),
            device=config.get("device", "cpu"),  # Thiết bị inference
            engine=config.get(
                "engine",
                "paddle_static",  # Engine inference
            ),
            cpu_threads=config.get("cpu_threads", 2),  # Số luồng CPU
            enable_mkldnn=config.get(
                "enable_mkldnn",
                False,  # Bật hoặc tắt MKL-DNN
            ),
        )
        self.ppocr_recognitor = TextRecognition(
            model_name=config.get(
                "table_ppocr_model_name",
                "PP-OCRv6_medium_rec",  # Model sửa token không dấu
            ),
            device=config.get("device", "cpu"),  # Thiết bị inference
            engine=config.get(
                "engine",
                "paddle_static",  # Engine inference
            ),
            cpu_threads=config.get("cpu_threads", 2),  # Số luồng CPU
            enable_mkldnn=config.get(
                "enable_mkldnn",
                False,  # Bật hoặc tắt MKL-DNN
            ),
        )
        self.cell_processor = TableCellProcessor(
            text_detector,
            text_recognitor,
            self.ppocr_recognitor,
        )

    # Chạy toàn bộ pipeline TABLE 01 trên một ảnh BGR.
    def analyze(self, image):
        """Trả fields và thông tin TABLE 01 mà không tạo file."""
        table_box = self._detect_table(image)
        # Không có TABLE 01 thì trả đầy đủ field rỗng theo contract.
        if table_box is None:
            return self._empty_result()

        table_crop, offset_x, offset_y = self._crop_table(
            image,
            table_box,
        )
        cell_result = self.cell_detector.predict(
            table_crop,
            threshold=self.config.get(
                "table_cell_threshold",
                0.3,  # Ngưỡng model tạo candidate cell
            ),
            batch_size=1,
        )[0]
        cell_boxes = sort_boxes_left_to_right(cell_result["boxes"])
        cell_boxes = self.cell_processor.resolve_nested_cell_boxes(
            table_crop,
            cell_boxes,
        )
        text_regions = self.cell_processor.recognize_text(table_crop)
        valid_cell_boxes = []
        low_score_cell_boxes = []
        cell_score_threshold = self.config.get(
            "table_cell_score_threshold",
            0.9,  # Ngưỡng cell chính thức
        )

        # Phân cell theo score trước khi cứu cell có field name.
        for cell_box in cell_boxes:
            if float(cell_box["score"]) < cell_score_threshold:
                low_score_cell_boxes.append(cell_box)
            else:
                valid_cell_boxes.append(cell_box)

        # Cell score thấp vẫn hợp lệ nếu OCR nhận diện được field name.
        for cell_box in low_score_cell_boxes:
            detected_cells, _ = self.cell_processor.assign_text_to_cells(
                [cell_box],
                text_regions,
            )
            if self.field_extractor.has_field_name(
                detected_cells[0]["text"]
            ):
                valid_cell_boxes.append(cell_box)

        cells, _ = self.cell_processor.assign_text_to_cells(
            valid_cell_boxes,
            text_regions,
        )
        self._convert_cells_to_page_coordinates(
            cells,
            offset_x,
            offset_y,
        )
        fields = self.field_extractor.extract(cells)

        return {
            "fields": fields,
            "table": {
                "coordinate": [
                    float(value) for value in table_box["coordinate"]
                ],
                "score": float(table_box["score"]),
            },
        }

    # Tạo output rỗng khi ảnh không có TABLE 01.
    def _empty_result(self):
        """Trả output rỗng nhưng giữ nguyên cấu trúc field."""
        return {
            "fields": self.field_extractor.empty_fields(),
            "table": None,
        }

    # Lấy bảng đầu tiên theo thứ tự đọc trên page 1.
    def _detect_table(self, image):
        """Nhận diện và chọn TABLE 01 từ ảnh đầu vào."""
        results = self.layout_detector.predict(
            image,
            batch_size=1,
            layout_nms=True,
        )
        table_boxes = []

        # Chỉ thu thập các layout box có label table.
        for result in results:
            table_boxes.extend(
                box
                for box in result["boxes"]
                if box["label"].lower() == "table"
            )

        table_boxes = sort_boxes_left_to_right(table_boxes)
        return table_boxes[0] if table_boxes else None

    # Cắt TABLE 01 với padding cấu hình và giữ offset trên ảnh gốc.
    def _crop_table(self, image, table_box):
        """Trả table crop cùng offset dùng để đổi coordinate cell."""
        height, width = image.shape[:2]
        x_min, y_min, x_max, y_max = map(
            int,
            table_box["coordinate"],
        )
        x_min = max(
            x_min - self.config.get(
                "table_crop_padding_left",
                0,  # Mở rộng crop về bên trái
            ),
            0,
        )
        y_min = max(
            y_min - self.config.get(
                "table_crop_padding_top",
                0,  # Mở rộng crop lên phía trên
            ),
            0,
        )
        x_max = min(
            x_max + self.config.get(
                "table_crop_padding_right",
                0,  # Mở rộng crop về bên phải
            ),
            width,
        )
        y_max = min(
            y_max + self.config.get(
                "table_crop_padding_bottom",
                5,  # Giữ trọn đường kẻ cuối bảng
            ),
            height,
        )
        return image[y_min:y_max, x_min:x_max], x_min, y_min

    # Đổi coordinate cell từ table crop sang ảnh page 1 đầu vào.
    def _convert_cells_to_page_coordinates(
        self,
        cells,
        offset_x,
        offset_y,
    ):
        """Cộng offset crop vào coordinate của toàn bộ cell."""
        # Mỗi field sẽ dùng lại coordinate tuyệt đối của cell chứa nó.
        for cell in cells:
            x_min, y_min, x_max, y_max = cell["coordinate"]
            cell["coordinate"] = [
                x_min + offset_x,
                y_min + offset_y,
                x_max + offset_x,
                y_max + offset_y,
            ]
