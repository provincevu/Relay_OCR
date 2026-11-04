"""Environment-backed settings for the local OCR API."""

import os
from dataclasses import dataclass
from pathlib import Path


def _positive_int(name, default):
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error
    if parsed < 1:
        raise ValueError(f"{name} must be greater than zero.")
    return parsed


@dataclass(frozen=True)
class Settings:
    api_host: str
    api_port: int
    redis_url: str
    queue_name: str
    api_token: str
    job_root: Path
    max_upload_mb: int
    max_pdf_pages: int
    job_retention_days: int
    job_timeout_seconds: int

    @classmethod
    def from_environment(cls):
        return cls(
            api_host=os.getenv("API_HOST", "127.0.0.1"),
            api_port=_positive_int("API_PORT", 8010),
            redis_url=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
            queue_name=os.getenv("RQ_QUEUE_NAME", "ocr"),
            api_token=os.getenv("OCR_API_TOKEN", "").strip(),
            job_root=Path(
                os.getenv("OCR_JOB_ROOT", "output/ocr_api/jobs"),
            ).resolve(),
            max_upload_mb=_positive_int("MAX_UPLOAD_MB", 100),
            max_pdf_pages=_positive_int("MAX_PDF_PAGES", 300),
            job_retention_days=_positive_int("JOB_RETENTION_DAYS", 7),
            job_timeout_seconds=_positive_int(
                "JOB_TIMEOUT_SECONDS",
                7200,
            ),
        )


settings = Settings.from_environment()
