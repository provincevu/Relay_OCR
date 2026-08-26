# Mục đích: nhóm tên ảnh source TABLE 01 theo tên rơ le.
# Chức năng: đọc prediction JSON và tạo danh sách text theo từng ten_ro_le.
# Luồng xử lý: JSON prediction → group theo ten_ro_le → file text.
# Cấu hình/parameter: dùng cố định input và output trong output/layout.
# Quan hệ: xử lý output của pipeline phân tích TABLE 01.
# Input/output: đọc table_01_predictions.json; ghi table_01_sources_by_relay.txt.
# Contract: mỗi dòng có dạng "- {ten_ro_le}: {source_1}, {source_2}".

import json
from pathlib import Path


INPUT_PATH = Path("./output/layout/table_01_predictions.json")
OUTPUT_PATH = Path("./output/layout/table_01_sources_by_relay.txt")


def main():
    """Nhóm source TABLE 01 theo tên rơ le và lưu thành file text."""
    predictions = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    sources_by_relay = {}

    for source_name, prediction in predictions.items():
        relay_name = prediction["ten_ro_le"]
        if relay_name is None:
            continue

        sources_by_relay.setdefault(relay_name, []).append(source_name)

    output_lines = [
        f"- {relay_name}: {', '.join(source_names)}"
        for relay_name, source_names in sources_by_relay.items()
    ]
    OUTPUT_PATH.write_text(
        "\n".join(output_lines),
        encoding="utf-8",
    )

    print(f"Đã tạo: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
