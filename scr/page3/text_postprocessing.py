# Mục đích: cung cấp service hậu xử lý prediction page 3+.
# Chức năng: lọc link, sửa role region và ghi metadata lý do thay đổi.
# Luồng xử lý: prediction → lọc theo page → cập nhật role/link → metadata.
# Cấu hình/parameter: page3_field_table_x_tolerance mặc định theo rule hiện có.
# Quan hệ: tái sử dụng quy tắc lọc prediction và role metadata hiện có.
# Input/output: nhận prediction dict hoặc danh sách prediction; cập nhật trực tiếp và trả lại dữ liệu.
# Contract: giữ schema, region ID, text và bbox; chỉ thay đổi links/labels theo rule.

from pathlib import Path

from scripts.inference.page_n_prediction_filter import (
    filter_prediction_pages,
    filter_unlinked_regions,
)
from scripts.inference.role_change_metadata import write_role_change_metadata


class TextPostprocessing:
    """Hậu xử lý prediction page 3+ theo các rule hiện có."""

    def __init__(self, config):
        self.field_table_x_tolerance = config.get(
            "page3_field_table_x_tolerance",
            0.04,
        )

    def process_page(self, prediction):
        """Hậu xử lý một prediction page."""
        return filter_unlinked_regions(prediction)

    def process_pages(self, predictions):
        """Hậu xử lý liên tục các prediction theo thứ tự page."""
        filter_prediction_pages(
            predictions,
            self.field_table_x_tolerance,
        )
        return predictions

    def write_metadata(self, directory):
        """Ghi metadata giải thích các mã đổi role."""
        return write_role_change_metadata(Path(directory))
