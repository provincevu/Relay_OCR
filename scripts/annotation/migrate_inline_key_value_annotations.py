# Mục đích: chuyển annotation inline legacy sang role INLINE_KEY_VALUE.
# Chức năng: đọc ba dataset annotation, đổi nhãn KEY + VALUE inline và loại relation inline.
# Luồng xử lý: đọc JSON nguồn → cập nhật bản sao trong bộ nhớ → ghi JSON đích mới.
# Cấu hình/parameter: các đường dẫn input/output được khai báo cố định trong module.
# Quan hệ: giữ nguyên region, metadata và relation spatial; không sửa file nguồn.
# Input/output: đọc annotations.json; ghi annotations_inline_key_value.json.
# Contract: chỉ đổi chính xác nhãn KEY + VALUE và relation có mode inline.

import json
import sys
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
DATASET_PATHS = (
    (
        PROJECT_DIRECTORY / "data/page3plus_dataset/annotations.json",
        PROJECT_DIRECTORY / "data/page3plus_dataset/"
        "annotations_inline_key_value.json",
    ),
    (
        PROJECT_DIRECTORY / "output/rule_induction/page3plus_val/annotations.json",
        PROJECT_DIRECTORY / "output/rule_induction/page3plus_val/"
        "annotations_inline_key_value.json",
    ),
    (
        PROJECT_DIRECTORY / "output/rule_induction/page3plus_test/annotations.json",
        PROJECT_DIRECTORY / "output/rule_induction/page3plus_test/"
        "annotations_inline_key_value.json",
    ),
)


def migrate_dataset(input_path, output_path):
    """Đổi inline KEY + VALUE trong một dataset và ghi sang file mới."""
    dataset = json.loads(input_path.read_text(encoding="utf-8"))
    migrated_regions = 0
    removed_inline_relations = 0

    for page in dataset["pages"].values():
        for region_id, labels in page["annotations"].items():
            if len(labels) == 2 and set(labels) == {"KEY", "VALUE"}:
                page["annotations"][region_id] = ["INLINE_KEY_VALUE"]
                migrated_regions += 1

        original_relations = page["relations"]
        page["relations"] = [
            relation
            for relation in original_relations
            if relation.get("mode") != "inline"
        ]
        removed_inline_relations += (
            len(original_relations) - len(page["relations"])
        )

    output_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return migrated_regions, removed_inline_relations


def main():
    """Migrate train, validation và test dataset sang role inline mới."""
    sys.stdout.reconfigure(encoding="utf-8")
    for input_path, output_path in DATASET_PATHS:
        migrated_regions, removed_inline_relations = migrate_dataset(
            input_path,
            output_path,
        )
        print(
            f"Đã tạo {output_path}: "
            f"{migrated_regions} region INLINE_KEY_VALUE, "
            f"xóa {removed_inline_relations} relation inline",
        )


if __name__ == "__main__":
    main()
