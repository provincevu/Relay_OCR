
"""RQ task implementation for PDF OCR jobs."""

import json
from pathlib import Path

from .settings import settings


ocr_service = None

# hàm khởi tạo OCRService nếu chưa có, giúp tránh việc tạo nhiều instance của OCRService, chỉ tạo một lần duy nhất
def get_ocr_service():
    global ocr_service 

    if ocr_service is None:
        config = json.loads(
            settings.config_path.read_text(encoding="utf-8")
        )

        from scr.ocr_service import OCRService

        ocr_service = OCRService(config)

    return ocr_service


def process_ocr_job(job_id, source_pdf_path, original_filename=None):
    # 1. kiểm tra đường dẫn và file PDF
    job_directory = (settings.job_root / job_id).resolve()
    source_pdf = Path(source_pdf_path).resolve()

    if job_directory.parent != settings.job_root.resolve():
        raise ValueError("đường dẫn job không hợp lệ.")

    if source_pdf.parent != job_directory or not source_pdf.is_file():
        raise FileNotFoundError("không tìm thấy file PDF nguồn trong thư mục job.")

    # 2. tạo thư mục output cho kết quả OCR
    output_directory = job_directory / "ocr_output"
    output_directory.mkdir(parents=True, exist_ok=True)

    # 3. Chạy OCR
    result = get_ocr_service().process_pdf(str(source_pdf),output_directory=output_directory)

    # 4. xóa thông tin debug cache nếu có, để tránh lưu trữ thông tin không cần thiết trong kết quả
    result.pop("_debug_cache", None)

    if isinstance(result.get("source_pdf"), dict):
        result["source_pdf"] = {
            "name": Path(
                original_filename or "upload.pdf"
            ).name,
            "page_count": result["source_pdf"].get("page_count"),
        }

    # 5. Lưu kết quả OCR vào file result.json trong thư mục job
    result["job_id"] = job_id

    result_path = job_directory / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "job_id": job_id,
        "result": "result.json",
    }
