# Mục đích: Render page 3 đã được xác định của các tệp PDF thành ảnh PNG.
# Chức năng: Đọc số page vật lý từ cache Rule Induction và render ở 300 DPI.
# Luồng xử lý: Đọc cache → duyệt PDF → render page đã cache → ghi ảnh PNG.
# Cấu hình/parameter: Đường dẫn input, output, cache và DPI được cố định trong module.
# Quan hệ: Sử dụng cache do test_rule_induction_page3_plus.py tạo.
# Input/output: Đọc PDF trong data/pdf; ghi ảnh vào data/page3.
# Contract: Ảnh output có tên {tên_pdf}_page_3.png.

import json
import shutil
import subprocess
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
INPUT_DIRECTORY = PROJECT_DIRECTORY / "data" / "pdf"
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "data" / "page3"
CACHE_PATH = (
    PROJECT_DIRECTORY
    / "output"
    / "rule_induction"
    / INPUT_DIRECTORY.name
    / "page_selection_cache.json"
)
DPI = 300


def main():
    """Render page 3 của các PDF theo số page vật lý đã lưu trong cache."""
    renderer = shutil.which("gswin64c")
    if not renderer:
        raise FileNotFoundError("Không tìm thấy gswin64c trong PATH.")

    page_selection_cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    pdf_files = sorted(
        INPUT_DIRECTORY.glob("*.pdf"),
        key=lambda path: path.name,
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    rendered_count = 0
    for pdf_file in pdf_files:
        physical_page = page_selection_cache[pdf_file.name]["physical_page"]
        if physical_page is None:
            print(f"Bỏ qua {pdf_file.name}: cache không có page 3.")
            continue

        output_path = OUTPUT_DIRECTORY / f"{pdf_file.stem}_page_3.png"
        command = [
            renderer,
            "-q",
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=png16m",
            f"-r{DPI}",
            f"-dFirstPage={physical_page}",
            f"-dLastPage={physical_page}",
            f"-sOutputFile={output_path}",
            str(pdf_file),
        ]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Không thể render: {pdf_file.name}")
        if not output_path.is_file():
            raise FileNotFoundError(f"Không tạo được ảnh: {output_path}")

        rendered_count += 1
        print(
            f"Đã render page vật lý {physical_page} của {pdf_file.name}: "
            f"{output_path}"
        )

    print(
        f"Đã render {rendered_count}/{len(pdf_files)} PDF vào: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
