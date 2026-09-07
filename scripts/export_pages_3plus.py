# Mục đích: chuyển các page dữ liệu sau page 1 của tệp PDF thành ảnh PNG.
# Chức năng: nhận diện page ý kiến để bỏ qua rồi render bằng Ghostscript ở 400 DPI.
# Luồng xử lý: kiểm tra page 2 → chọn page bắt đầu → render vào thư mục riêng.
# Cấu hình/parameter: input và output cố định trong data/pdf; DPI cố định là 400.
# Quan hệ: độc lập với pipeline phân tích bảng page 1 và page 3+.
# Input/output: đọc PDF trong data/pdf; ghi PNG vào data/pdf/page3plus_png.
# Contract: ảnh dữ liệu có đường dẫn {tên_pdf}/page_{số_thứ_tự_3_chữ_số}.png.

import shutil
import subprocess
from pathlib import Path

import cv2


INPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "pdf"
OUTPUT_DIRECTORY = INPUT_DIRECTORY / "page3plus_png"
DPI = 400
PAGE_2 = 2
PAGE_3 = 3
CLASSIFICATION_DPI = 100


def _count_row_groups(row_indices, maximum_gap=4):
    """Đếm số nhóm hàng ảnh nằm gần nhau."""
    group_count = 0
    previous_row = None

    for row_index in row_indices:
        if previous_row is None or row_index - previous_row > maximum_gap:
            group_count += 1
        previous_row = row_index

    return group_count


def _is_comment_page(image_path):
    """Nhận diện page ý kiến qua các hàng chấm và đường kẻ ngang."""
    gray_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    height, width = gray_image.shape
    normalized_height = round(height * 1000 / width)
    normalized_image = cv2.resize(
        gray_image,
        (1000, normalized_height),
        interpolation=cv2.INTER_CUBIC,
    )
    binary_image = cv2.adaptiveThreshold(
        normalized_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    table_area = binary_image[
        int(normalized_height * 0.15):int(normalized_height * 0.90),
        50:950,
    ]

    dotted_runs = (
        (table_area[:, 1:] > 0) & (table_area[:, :-1] == 0)
    ).sum(axis=1)
    dotted_rows = (dotted_runs >= 25).nonzero()[0]
    dotted_group_count = _count_row_groups(dotted_rows)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (300, 1),
    )
    horizontal_lines = cv2.morphologyEx(
        table_area,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )
    solid_rows = ((horizontal_lines > 0).sum(axis=1) >= 300).nonzero()[0]
    solid_group_count = _count_row_groups(solid_rows)

    return 20 <= dotted_group_count <= 35 and solid_group_count <= 5


def find_first_data_page(renderer, pdf_file, output_directory):
    """Trả số page bắt đầu sau khi phân loại page vật lý thứ hai."""
    page_2_image = output_directory / "page_2_classification.png"
    command = [
        renderer,
        "-q",
        "-dSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        "-sDEVICE=pnggray",
        f"-r{CLASSIFICATION_DPI}",
        f"-dFirstPage={PAGE_2}",
        f"-dLastPage={PAGE_2}",
        f"-sOutputFile={page_2_image}",
        str(pdf_file),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Không thể kiểm tra page 2: {pdf_file.name}")
    if not page_2_image.is_file():
        return None

    is_comment_page = _is_comment_page(page_2_image)
    page_2_image.unlink()

    if is_comment_page:
        return PAGE_3
    return PAGE_2


def main():
    """Render các page dữ liệu sau page 1 của toàn bộ PDF thành ảnh PNG."""
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
        first_data_page = find_first_data_page(
            renderer,
            pdf_file,
            document_output_directory,
        )
        if first_data_page is None:
            print(f"Bỏ qua {pdf_file.name}: không có page dữ liệu sau page 1.")
            continue
        output_pattern = document_output_directory / "page_%03d.png"
        command = [
            renderer,
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=png16m",
            f"-r{DPI}",
            f"-dFirstPage={first_data_page}",
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
                f"Bỏ qua {pdf_file.name}: không có page dữ liệu sau page 1."
            )
            continue

        print(
            f"Đã tạo {len(rendered_pages)} ảnh cho {pdf_file.name}: "
            f"bắt đầu từ page {first_data_page} → "
            f"{document_output_directory}"
        )

    print(
        f"Đã chuyển đổi các page dữ liệu của {len(pdf_files)} PDF vào: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
