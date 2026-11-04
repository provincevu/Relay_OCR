"""RQ task implementation for local PDF OCR jobs."""

import json
import re
from pathlib import Path
from threading import Lock
from uuid import uuid4

from .settings import settings


_ocr_service = None
_ocr_service_lock = Lock()


def _get_ocr_service():
    """Load OCR models once in the worker process and reuse them per job."""
    global _ocr_service
    if _ocr_service is None:
        with _ocr_service_lock:
            if _ocr_service is None:
                if not settings.config_path.is_file():
                    raise FileNotFoundError(
                        f"OCR config not found: {settings.config_path}",
                    )
                config = json.loads(
                    settings.config_path.read_text(encoding="utf-8"),
                )
                from scr.ocr_service import OCRService

                _ocr_service = OCRService(config)
    return _ocr_service


def _job_paths(job_id, source_pdf_path):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("Invalid job id.")

    job_root = settings.job_root.resolve()
    job_directory = (job_root / job_id).resolve()
    if job_directory.parent != job_root:
        raise ValueError("Job directory is outside the configured root.")

    source_pdf = Path(source_pdf_path).resolve()
    if source_pdf.parent != job_directory or source_pdf.name != "source.pdf":
        raise ValueError("Source PDF is outside the job directory.")
    if not source_pdf.is_file():
        raise FileNotFoundError("Source PDF is missing.")

    return job_directory, source_pdf


def _public_result(process_result, job_id, original_filename=None):
    """Remove local paths and internal cache metadata from the API result."""
    result = {
        key: value
        for key, value in process_result.items()
        if key != "_debug_cache"
    }
    source_pdf = result.get("source_pdf")
    if isinstance(source_pdf, dict):
        result["source_pdf"] = {
            "name": Path(
                original_filename or source_pdf.get("name", "upload.pdf"),
            ).name,
            "page_count": source_pdf.get("page_count"),
        }

    def remove_paths(value):
        if isinstance(value, dict):
            return {
                key: remove_paths(item)
                for key, item in value.items()
                if key != "_debug_cache"
                and key != "path"
                and not key.endswith("_path")
            }
        if isinstance(value, list):
            return [remove_paths(item) for item in value]
        return value

    public_result = remove_paths(result)
    public_result["job_id"] = job_id
    return public_result


def _write_result_atomically(result_path, payload):
    temporary_path = result_path.with_name(
        f".{result_path.name}.{uuid4().hex}.tmp",
    )
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(result_path)


def process_ocr_job(job_id, source_pdf_path, original_filename=None):
    """Run the OCR pipeline and persist its public JSON result per job."""
    job_directory, source_pdf = _job_paths(job_id, source_pdf_path)
    output_directory = job_directory / "ocr_output"
    output_directory.mkdir(parents=True, exist_ok=True)

    process_result = _get_ocr_service().process_pdf(
        str(source_pdf),
        output_directory=output_directory,
    )
    public_result = _public_result(
        process_result,
        job_id,
        original_filename,
    )
    _write_result_atomically(job_directory / "result.json", public_result)
    return {"job_id": job_id, "result": "result.json"}
