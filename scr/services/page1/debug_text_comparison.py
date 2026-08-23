# Mục đích: giữ đường import cũ cho luồng debug TABLE 01.
# Chức năng: export lại quy tắc so sánh VietOCR và PP-OCRv6 từ module production.
# Luồng xử lý: module debug → module dùng chung → hàm so sánh hoặc sửa token.
# Cấu hình/parameter: dùng danh sách ký tự dễ nhầm trong utils.text_comparison.
# Quan hệ: được debug_table_fields và debug_table_visualization sử dụng.
# Input/output: giữ nguyên các hàm và hằng số đã được luồng debug import.
# Contract: không thay đổi tên API công khai của debug_text_comparison.

from .utils.text_comparison import (
    CONFUSABLE_TEXT_PAIRS,
    CONFUSABLE_TRANSFORMS,
    TOKEN_PATTERN,
    correct_value_with_ppocr,
    find_confusable_changes,
    has_vietnamese_mark,
    pair_non_vietnamese_tokens,
)


__all__ = [
    "CONFUSABLE_TEXT_PAIRS",
    "CONFUSABLE_TRANSFORMS",
    "TOKEN_PATTERN",
    "correct_value_with_ppocr",
    "find_confusable_changes",
    "has_vietnamese_mark",
    "pair_non_vietnamese_tokens",
]
