# Mục đích: chuyển page 3 trở đi của các tệp PDF thành ảnh PNG.
# Chức năng: duyệt PDF theo thứ tự tên và render bằng Ghostscript ở 400 DPI.
# Luồng xử lý: data/pdf/*.pdf → page 3 đến page cuối → thư mục riêng theo PDF.
# Cấu hình/parameter: input và output cố định trong data/pdf; DPI cố định là 400.
# Quan hệ: độc lập với pipeline phân tích bảng page 1 và page 3+.
# Input/output: đọc PDF trong data/pdf; ghi PNG vào data/pdf/page3plus_png.
# Contract: ảnh có đường dẫn {tên_pdf}/page_{số_trang_3_chữ_số}.png.

import shutil
import subprocess
from pathlib import Path


INPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "pdf"
OUTPUT_DIRECTORY = INPUT_DIRECTORY / "page3plus_png"
DPI = 400
FIRST_PAGE = 3


def main():
    """Render page 3 trở đi của toàn bộ PDF thành ảnh PNG."""
    if not INPUT_DIRECTORY.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục input: {INPUT_DIRECTORY}"
        )

    renderer = shutil.which("gswin64c")
    if not renderer:
        raise FileNotFoundError(
            "Không tìm thấy gswin64c trong PATH."
        )

    pdf_files = sorted(
        INPUT_DIRECTORY.glob("*.pdf"),
        key=lambda path: path.name,
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for pdf_file in pdf_files:
        document_output_directory = OUTPUT_DIRECTORY / pdf_file.stem
        document_output_directory.mkdir(parents=True, exist_ok=True)
        output_pattern = document_output_directory / "page_%03d.png"
        command = [
            renderer,
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=png16m",
            f"-r{DPI}",
            f"-dFirstPage={FIRST_PAGE}",
            f"-sOutputFile={output_pattern}",
            str(pdf_file),
        ]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Không thể chuyển đổi: {pdf_file.name}")

        rendered_pages = sorted(
            document_output_directory.glob("page_*.png")
        )
        if not rendered_pages:
            print(
                f"Bỏ qua {pdf_file.name}: không có page {FIRST_PAGE} trở đi."
            )
            continue

        print(
            f"Đã tạo {len(rendered_pages)} ảnh cho {pdf_file.name}: "
            f"{document_output_directory}"
        )

    print(
        f"Đã chuyển đổi page 3+ của {len(pdf_files)} PDF vào: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
