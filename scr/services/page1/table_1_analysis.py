# Mục đích: đóng gói phân tích TABLE 01 trên page 1.
# Chức năng: nhận diện bảng, cell, text và trích xuất các field cấu hình.
# Luồng xử lý: ảnh BGR → TABLE 01 → cell → OCR text → dữ liệu chuẩn hóa.
# Cấu hình/parameter: nhận dict config dùng chung của OCRService.
# Quan hệ: dùng Detector, Recognitor và các utility riêng của TABLE 01.
# Input/output: analyze() nhận ảnh numpy BGR và trả một dict kết quả.
# Contract: output luôn có table, cells, unassigned_texts, data, field_details và review.

import json
from pathlib import Path

from paddleocr import LayoutDetection, TableCellsDetection

from ..detector import Detector
from ..recognitor import Recognitor
from .utils.table_cell_utils import (
    assign_text_to_cells,
    recognize_text,
    sort_boxes_left_to_right,
)
from .utils.table_field_extractor import TableFieldExtractor


class Table1Analysis:
    """Phân tích và trích xuất dữ liệu từ TABLE 01."""

    def __init__(self, config):
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
                "PicoDet_layout_1x_table",  # Mô hình nhận diện bảng
            ),
            device=config.get("device", "cpu"),  # Thiết bị inference
            engine=config.get("engine", "paddle_static"),  # Engine inference
            cpu_threads=config.get("cpu_threads", 2),  # Số luồng CPU
            enable_mkldnn=config.get("enable_mkldnn", False),  # Bật MKL-DNN
        )
        self.cell_detector = TableCellsDetection(
            model_name=config.get(
                "table_cell_model_name",
                "RT-DETR-L_wired_table_cell_det",  # Mô hình nhận diện cell
            ),
            device=config.get("device", "cpu"),
            engine=config.get("engine", "paddle_static"), # Engine suy luận
            cpu_threads=config.get("cpu_threads", 2),       # Số luồng CPU
            enable_mkldnn=config.get("enable_mkldnn", False), # Tắt MKL-DNN
        )
        self.text_detector = Detector(config)
        self.text_recognitor = Recognitor(config)

    def analyze(self, image):
        """Phân tích TABLE 01 từ ảnh numpy BGR."""
        result = self._empty_result()
        table_box = self._detect_table(image)
        if table_box is None:
            return result

        table_crop = self._crop_table(image, table_box)
        cell_result = self.cell_detector.predict(
            table_crop,
            threshold=self.config.get("table_cell_threshold", 0.3), # Ngưỡng nhận diện cell
            batch_size=1,
        )[0] # Lấy kết quả cell đầu tiên (chỉ có 1 bảng)

        # Lọc cell theo score threshold
        cell_boxes = [
            box for box in cell_result["boxes"] 
            if float(box["score"]) >= self.config.get("table_cell_score_threshold",0.9)  # Ngưỡng giữ lại cell
        ]
        text_regions = recognize_text(table_crop, self.text_detector, self.text_recognitor) # Nhận diện text trong bảng
        cells, unassigned_texts = assign_text_to_cells(cell_boxes, text_regions) # Gán text vào cell và lấy text chưa gán
        extracted_result = self.field_extractor.extract(cells) # Trích xuất dữ liệu từ các cell đã gán text

        result.update(extracted_result)
        result["table"] = {
            "coordinate": [
                float(value) for value in table_box["coordinate"]
            ],
            "score": float(table_box["score"]),
        }
        result["cells"] = cells
        result["unassigned_texts"] = unassigned_texts
        return result

    def _empty_result(self):
        """Tạo output rỗng theo đúng contract."""
        result = self.field_extractor.empty_result()
        result.update(
            {
                "table": None,
                "cells": [],
                "unassigned_texts": [],
            }
        )
        return result

    def _detect_table(self, image):
        """Lấy bảng đầu tiên theo thứ tự từ trái sang phải."""
        results = self.layout_detector.predict(image, batch_size=1, layout_nms=True) # Nhận diện layout và áp dụng NMS
        table_boxes = []
        # Lấy tất cả các box có label là "table" từ kết quả nhận diện layout
        for result in results:
            table_boxes.extend(box for box in result["boxes"]
                if box["label"].lower() == "table"
            )

        table_boxes = sort_boxes_left_to_right(table_boxes)
        return table_boxes[0] if table_boxes else None

    def _crop_table(self, image, table_box):
        """Cắt TABLE 01 và giới hạn bbox trong kích thước ảnh."""
        height, width = image.shape[:2]
        x_min, y_min, x_max, y_max = map(int, table_box["coordinate"]) # Chuyển tọa độ sang int
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, width)
        y_max = min(y_max, height)
        return image[y_min:y_max, x_min:x_max]
