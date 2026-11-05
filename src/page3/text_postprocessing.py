# Mục đích: cung cấp service hậu xử lý prediction page 3+.
# Chức năng: lọc link, sửa role region và ghi metadata lý do thay đổi.
# Luồng xử lý: prediction → lọc theo page → cập nhật role/link → metadata.
# Cấu hình/parameter: page3_field_table_x_tolerance mặc định theo rule hiện có.
# Quan hệ: tái sử dụng quy tắc lọc prediction và role metadata hiện có.
# Input/output: nhận prediction dict hoặc danh sách prediction; cập nhật trực tiếp và trả lại dữ liệu.
# Contract: giữ schema, region ID, text và bbox; chỉ thay đổi links/labels theo rule.

from pathlib import Path

from .page_n_prediction_filter import (
    CROSS_ROW_MAX_VERTICAL_GAP,
    CROSS_ROW_MIN_HORIZONTAL_OVERLAP,
    CROSS_ROW_MIN_SCORE_MARGIN,
    FIELD_TABLE_X_TOLERANCE,
    filter_prediction_pages,
    filter_unlinked_regions,
)
from .role_change_metadata import write_role_change_metadata


class TextPostprocessing:
    """Hậu xử lý prediction page 3+ theo các rule hiện có."""

    def __init__(self, config):
        self.field_table_x_tolerance = config.get(
            "page3_field_table_x_tolerance",
            FIELD_TABLE_X_TOLERANCE,
        )
        self.enable_cross_row_fallback = config.get(
            "page3_enable_cross_row_fallback",
            False,
        )
        self.cross_row_max_vertical_gap = config.get(
            "page3_cross_row_max_vertical_gap",
            CROSS_ROW_MAX_VERTICAL_GAP,
        )
        self.cross_row_min_horizontal_overlap = config.get(
            "page3_cross_row_min_horizontal_overlap",
            CROSS_ROW_MIN_HORIZONTAL_OVERLAP,
        )
        self.cross_row_min_score_margin = config.get(
            "page3_cross_row_min_score_margin",
            CROSS_ROW_MIN_SCORE_MARGIN,
        )

    def process_page(self, prediction):
        """Hậu xử lý một prediction page."""
        return filter_unlinked_regions(
            prediction,
            enable_cross_row_fallback=self.enable_cross_row_fallback,
            max_vertical_gap=self.cross_row_max_vertical_gap,
            min_horizontal_overlap=self.cross_row_min_horizontal_overlap,
            min_score_margin=self.cross_row_min_score_margin,
        )

    def process_pages(self, predictions):
        """Hậu xử lý liên tục các prediction theo thứ tự page."""
        filter_prediction_pages(
            predictions,
            self.field_table_x_tolerance,
            self.enable_cross_row_fallback,
            self.cross_row_max_vertical_gap,
            self.cross_row_min_horizontal_overlap,
            self.cross_row_min_score_margin,
        )
        return predictions

    def write_metadata(self, directory):
        """Ghi metadata giải thích các mã đổi role."""
        return write_role_change_metadata(Path(directory))
