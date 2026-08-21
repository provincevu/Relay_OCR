# Mục đích: làm entry point cho luồng debug nhận diện bảng trên page 1.
# Chức năng: cấu hình lần chạy, duyệt ảnh và gọi Page1TableAnalyzer.
# Luồng xử lý: thư mục ảnh page 1 → analyzer → HTML/JSON/report trong output/layout.
# Cấu hình/parameter: số bảng, padding crop và tùy chọn JSON đánh giá ở main().
# Quan hệ: logic được tách trong các module debug_table_* thuộc scr/services/page1.
# Input/output: đọc ảnh trong data/pdf/page1_png; ghi toàn bộ output debug hiện có.
# Contract: tiếp tục export Page1TableAnalyzer và chạy được trực tiếp như trước.

from pathlib import Path

from scr.services.page1.debug_table_analyzer import (
    CELL_SCORE_THRESHOLD,
    TABLE_SCORE_WARNING_THRESHOLD,
    Page1TableAnalyzer,
)


__all__ = [
    "CELL_SCORE_THRESHOLD",
    "TABLE_SCORE_WARNING_THRESHOLD",
    "Page1TableAnalyzer",
]


def main():
    """Chạy phân tích bảng page 1 cho toàn bộ ảnh đầu vào."""
    analyzer = Page1TableAnalyzer(
        include_second_table=False,
        include_third_table=False,
        crop_padding_left=0,
        crop_padding_top=0,
        crop_padding_right=0,
        crop_padding_bottom=5,
        save_evaluation_json=True,
    )
    input_dir = Path("./data/pdf/page1_png")
    image_files = sorted(
        path
        for path in input_dir.iterdir()
        if path.suffix.lower()
        in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )

    for index, image_path in enumerate(image_files, start=1):
        print(f"[{index}/{len(image_files)}] Đang xử lý: {image_path.name}")
        analyzer.detect_tables(image_path)

    analyzer.save_html()
    analyzer.save_report()
    analyzer.save_extracted_data()


if __name__ == "__main__":
    main()
