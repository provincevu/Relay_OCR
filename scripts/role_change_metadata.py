# Mục đích: định nghĩa contract mã lý do đổi role của region.
# Chức năng: khởi tạo vùng mã lý do trong từng page và ghi file metadata tra cứu.
# Luồng xử lý: nhận danh sách page → bảo đảm mỗi page có role_change_codes → ghi metadata.
# Cấu hình/parameter: mã lý do là số nguyên; file metadata dùng tên cố định.
# Quan hệ: được dùng bởi pipeline lọc predictions và các script ghi predictions.json.
# Input/output: cập nhật trực tiếp page dict và ghi role_change_metadata.json.
# Contract: predictions.json chỉ lưu mã số; metadata lưu mô tả của từng mã.

import json
from pathlib import Path


ROLE_CHANGE_REASON_METADATA = {
    1: "Model gán role OTHER ngay từ đầu.",
    2: "KEY/VALUE nằm cùng hàng với FIELD_TABLE.",
    3: "KEY/VALUE nằm ngoài phạm vi FIELD_TABLE hiện tại.",
    4: "KEY/VALUE nằm ngoài phạm vi FIELD_TABLE kế thừa từ page trước.",
    5: "Link KEY → VALUE bị loại vì VALUE không nằm bên phải KEY.",
    6: "KEY không còn link hợp lệ.",
    7: "VALUE không còn link hợp lệ.",
}

ROLE_CHANGE_METADATA_FILENAME = "role_change_metadata.json"


def initialize_role_change_codes(pages):
    """Bảo đảm mỗi page có nơi lưu mã lý do theo region ID."""
    for page in pages:
        page.setdefault("role_change_codes", {})
    return pages


def write_role_change_metadata(directory):
    """Ghi bảng giải mã mã lý do vào thư mục chứa predictions.json."""
    metadata = {
        "schema_version": 1,
        "reason_codes": {
            str(code): description
            for code, description in ROLE_CHANGE_REASON_METADATA.items()
        },
    }
    metadata_path = Path(directory) / ROLE_CHANGE_METADATA_FILENAME
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata_path
