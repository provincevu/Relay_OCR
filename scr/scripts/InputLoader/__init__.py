# Mục đích: công khai service InputLoader cho pipeline OCR.
# Chức năng: export lớp InputLoader.
# Luồng xử lý: import package → dùng InputLoader.load().
# Cấu hình/parameter: nhận config tại InputLoader.
# Quan hệ: trỏ tới input_loader.py.
# Input/output: không xử lý dữ liệu trực tiếp.
# Contract: giữ tên import InputLoader.

from .input_loader import InputLoader

__all__ = ["InputLoader"]
