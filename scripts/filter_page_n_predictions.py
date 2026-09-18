# Mục đích: cập nhật predictions.json theo quy tắc link và ghép cặp strict.
# Chức năng: lọc link sai và đổi KEY/VALUE không có link thành OTHER.
# Luồng xử lý: đọc predictions.json → lọc từng page → ghi đè chính file đầu vào.
# Cấu hình/parameter: nhận đường dẫn predictions.json từ command line.
# Quan hệ: dùng cùng quy tắc với predict_page() và serve_page_n_predictions.py.
# Input/output: đọc và ghi predictions.json; không chạy PDF, detection hoặc OCR.
# Contract: giữ nguyên schema, metadata, region ID, text và bbox; chỉ đổi links/labels.

import argparse
import json
import sys
from pathlib import Path

if __package__:
    from .page_n_prediction_filter import (
        filter_prediction_links,
        filter_unlinked_regions,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from page_n_prediction_filter import (
        filter_prediction_links,
        filter_unlinked_regions,
    )


def main():
    """Lọc và ghi lại predictions.json mà không chạy lại inference."""
    parser = argparse.ArgumentParser(
        description=(
            "Lọc link KEY → VALUE và đổi region không có cặp thành OTHER."
        ),
    )
    parser.add_argument("predictions", type=Path)
    arguments = parser.parse_args()

    dataset = json.loads(arguments.predictions.read_text(encoding="utf-8"))
    removed_links = 0
    changed_regions = 0
    for page in dataset["pages"]:
        links_before = sum(
            len(links)
            for links in page.get(
                "links_by_mode",
                {"selected": page.get("links", [])},
            ).values()
        )
        other_before = sum(
            "OTHER" in region.get("labels", [])
            for region in page["regions"]
        )
        filter_prediction_links(page)
        filter_unlinked_regions(page)
        links_after = sum(
            len(links)
            for links in page.get(
                "links_by_mode",
                {"selected": page.get("links", [])},
            ).values()
        )
        removed_links += links_before - links_after
        other_after = sum(
            "OTHER" in region.get("labels", [])
            for region in page["regions"]
        )
        changed_regions += max(0, other_after - other_before)

    arguments.predictions.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Đã cập nhật {len(dataset['pages'])} page: "
        f"loại {removed_links} link, đổi {changed_regions} region thành OTHER."
    )


if __name__ == "__main__":
    main()
