
import json
from pathlib import Path

from paddleocr import LayoutDetection, TableCellsDetection, TextRecognition

from .page1.utils.table_cell_utils import (
    TableCellProcessor,
    sort_boxes_left_to_right,
)
from .page1.utils.table_field_extractor import TableFieldExtractor


class Table1Analysis:
    def __init__(self, config, text_detector, text_recognitor):
        self.config = config

        # Tải cấu hình trường dữ liệu
        config_path = Path(
            config.get("table_01_fields_path", "./config/table_01_fields.json")
        )
        with open(config_path, encoding="utf-8") as file:
            field_config = json.load(file)

        self.field_extractor = TableFieldExtractor(field_config["fields"])

        # Cấu hình dùng chung cho các mô hình
        model_config = {
            "device": config.get("device", "cpu"),
            "engine": config.get("engine", "paddle_static"),
            "cpu_threads": config.get("cpu_threads", 2),
            "enable_mkldnn": config.get("enable_mkldnn", False),
        }

        # Khởi tạo các mô hình
        self.layout_detector = LayoutDetection(
            model_name=config.get("table_layout_model_name", "PicoDet_layout_1x_table"),
            **model_config,
        )

        self.cell_detector = TableCellsDetection(
            model_name=config.get(
                "table_cell_model_name",
                "RT-DETR-L_wired_table_cell_det",
            ),
            **model_config,
        )

        self.ppocr_recognitor = TextRecognition(
            model_name=config.get(
                "table_ppocr_model_name",
                "PP-OCRv6_medium_rec",
            ),
            **model_config,
        )

        self.cell_processor = TableCellProcessor(
            text_detector,
            text_recognitor,
            self.ppocr_recognitor,
        )

    def analyze(self, image):
        # Phát hiện bảng TABLE 01 trong ảnh (Table thứ nhất của ảnh chứa thông tin cần trích xuất)
        results = self.layout_detector.predict(
            image,
            batch_size=1,
            layout_nms=True,
        )

        table_boxes = [
            box
            for result in results
            for box in result["boxes"]
            if box["label"].lower() == "table"
        ]

        table_boxes = sort_boxes_left_to_right(table_boxes)

        if not table_boxes:
            return {
                "fields": self.field_extractor.empty_fields(),
                "table": None,
            }

        table = table_boxes[0]
        height, width = image.shape[:2]

        # Cắt vùng bảng
        x1, y1, x2, y2 = map(int, table["coordinate"])

        x1 = max(0, x1 - self.config.get("table_crop_padding_left", 0))
        y1 = max(0, y1 - self.config.get("table_crop_padding_top", 0))
        x2 = min(width, x2 + self.config.get("table_crop_padding_right", 0))
        y2 = min(height, y2 + self.config.get("table_crop_padding_bottom", 5))

        table_crop = image[y1:y2, x1:x2]

        # Phát hiện các ô
        result = self.cell_detector.predict(
            table_crop,
            threshold=self.config.get("table_cell_threshold", 0.3),
            batch_size=1,
        )[0]

        cells = sort_boxes_left_to_right(result["boxes"])

        cells = self.cell_processor.resolve_nested_cell_boxes(
            table_crop,
            cells,
        )

        # Nhận dạng văn bản
        text_regions = self.cell_processor.recognize_text(table_crop)

        valid_cells = []

        for cell in cells:
            if float(cell["score"]) >= self.config.get(
                "table_cell_score_threshold", 0.9
            ):
                valid_cells.append(cell)
                continue

            detected, _ = self.cell_processor.assign_text_to_cells(
                [cell],
                text_regions,
            )

            if self.field_extractor.has_field_name(detected[0]["text"]):
                valid_cells.append(cell)

        # Gán văn bản vào các ô
        cells, _ = self.cell_processor.assign_text_to_cells(
            valid_cells,
            text_regions,
        )

        # Chuyển tọa độ về ảnh gốc
        x_offset = max(0,int(table["coordinate"][0]) - self.config.get("table_crop_padding_left", 0))
        y_offset = max(0,int(table["coordinate"][1]) - self.config.get("table_crop_padding_top", 0))
        for cell in cells:
            x1, y1, x2, y2 = cell["coordinate"]
            cell["coordinate"] = [
                x1 + x_offset,
                y1 + y_offset,
                x2 + x_offset,
                y2 + y_offset,
            ]

        fields = self.field_extractor.extract(cells)

        return {
            "fields": fields,
            "table": {
                "coordinate": [float(v) for v in table["coordinate"]],
                "score": float(table["score"]),
            },
        }
