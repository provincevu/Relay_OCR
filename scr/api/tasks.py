"""RQ task entrypoints.

The OCR implementation is added in Phase 2. Keeping the entrypoint in an
importable module now lets the Phase 1 API enqueue jobs without importing OCR
models into the API process.
"""


def process_ocr_job(job_id, source_pdf_path):
    """Placeholder task replaced by the OCR worker implementation in Phase 2."""
    raise RuntimeError(
        "OCR processing is not enabled yet; complete Phase 2 to run workers.",
    )
