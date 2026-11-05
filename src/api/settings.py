"""Environment-backed settings for the local OCR API."""

import os
from dataclasses import dataclass
from pathlib import Path

# hàm giúp kiểm tra người dùng có truyền vào đúng giá trị cho các biến môi trường hay không
def _positive_int(name, default):
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} phải là integer.") from error
    if parsed < 1:
        raise ValueError(f"{name} phải lớn hơn 0.")
    return parsed


@dataclass(frozen=True)
class Settings:
    def __init__(self):
        self.api_host = os.getenv("API_HOST", "127.0.0.1") # địa chỉ host của API, mặc định là localhost
        self.api_port = _positive_int("API_PORT", 18010) # cổng của API, mặc định là 18010 để tránh xung đột với các dịch vụ khác
        self.redis_url = os.getenv(
            "REDIS_URL",
            "redis://127.0.0.1:6379/0"
        )
        self.queue_name = os.getenv("RQ_QUEUE_NAME", "ocr")
        self.api_token = os.getenv("OCR_API_TOKEN", "").strip()

        # chuyển thành Path object và resolve để có đường dẫn tuyệt đối
        self.job_root = Path(os.getenv("OCR_JOB_ROOT", "output/ocr_api/jobs")).resolve()
        self.config_path = Path(os.getenv("OCR_CONFIG_PATH", "config/ocr.json")).resolve()

        self.max_upload_mb = _positive_int("MAX_UPLOAD_MB", 100)
        self.max_pdf_pages = _positive_int("MAX_PDF_PAGES", 300)
        self.job_retention_days = _positive_int("JOB_RETENTION_DAYS", 7) # thời gian giữ lại job trong Redis trước khi bị xóa
        self.job_timeout_seconds = _positive_int("JOB_TIMEOUT_SECONDS", 86400) # 24 giờ là thời gian tối đa cho một job OCR trước khi bị timeout
    @classmethod
    def from_environment(cls):
        return cls(
            api_host=os.getenv("API_HOST", "127.0.0.1"),
            api_port=_positive_int("API_PORT", 8010),
            redis_url=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
            queue_name=os.getenv("RQ_QUEUE_NAME", "ocr"),
            api_token=os.getenv("OCR_API_TOKEN", "").strip(),job_root=Path(os.getenv("OCR_JOB_ROOT", "output/ocr_api/jobs")).resolve(),
            config_path=Path(os.getenv("OCR_CONFIG_PATH", "config/ocr.json")).resolve(),
            max_upload_mb=_positive_int("MAX_UPLOAD_MB", 100),
            max_pdf_pages=_positive_int("MAX_PDF_PAGES", 300),
            job_retention_days=_positive_int("JOB_RETENTION_DAYS", 7),
            job_timeout_seconds=_positive_int("JOB_TIMEOUT_SECONDS",86400),
        )


settings = Settings.from_environment()
