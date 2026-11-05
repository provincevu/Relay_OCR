
"""API cục bộ để gửi và theo dõi các công việc OCR."""

import hmac
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, UploadFile, Header, HTTPException, Depends
from pypdf import PdfReader
from redis import Redis
from rq import Queue
from rq.job import Job
from rq.exceptions import NoSuchJobError

from .settings import settings
from .tasks import process_ocr_job


# Redis và RQ
redis = Redis.from_url(settings.redis_url)
queue = Queue(settings.queue_name, connection=redis)

app = FastAPI(title="Relay OCR API")


# Xác thực
def authenticate(authorization: str = Header(default="")):
    expected = f"Bearer {settings.api_token}"

    if not settings.api_token or not hmac.compare_digest(
        authorization, expected
    ):
        raise HTTPException(status_code=401, detail="Không xác thực được. Hãy cung cấp token hợp lệ trong header Authorization.")


# Lấy job từ Redis, nếu không tìm thấy thì trả về 404
def get_job(job_id):
    try:
        return Job.fetch(job_id, connection=redis)
    except NoSuchJobError:
        raise HTTPException(status_code=404, detail="Không tìm thấy job với ID đã cho. Hãy kiểm tra lại ID hoặc xác nhận rằng job đã được gửi thành công.")


# Kiểm tra trạng thái
@app.get("/health")
def health():
    redis.ping()
    return {"status": "healthy"}


# Gửi PDF
@app.post("/api/v1/jobs", status_code=202)
async def submit_job(file: UploadFile, _: None = Depends(authenticate)):
    job_id = uuid4().hex

    # tạo thư mục job và lưu tệp PDF đã tải lên
    settings.job_root.mkdir(parents=True, exist_ok=True)
    job_directory = settings.job_root / job_id
    job_directory.mkdir()

    source_pdf = job_directory / "source.pdf"

    try:
        # Lưu tệp PDF đã tải lên
        total_size = 0
        first_chunk = b""

        # kiểm tra xem tệp có phải là PDF hợp lệ hay không bằng cách đọc 8 byte đầu tiên (file signature (magic bytes) - file PDF hợp lệ bắt đầu bằng %PDF-)
        with source_pdf.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                if not first_chunk:
                    first_chunk = chunk[:8]

                total_size += len(chunk)

                if total_size > settings.max_upload_mb * 1024 * 1024:
                    raise HTTPException(413, "PDF này quá lớn. Vui lòng tải lên tệp PDF có kích thước nhỏ hơn hoặc bằng 100MB.")

                output.write(chunk)

        if not first_chunk.startswith(b"%PDF-"):
            raise HTTPException(415, "file này không phải là PDF hợp lệ. Vui lòng tải lên tệp PDF hợp lệ.")

        if total_size == 0:
            raise HTTPException(400, "file PDF này trống hoặc không thể đọc được. Vui lòng tải lên tệp PDF hợp lệ.")

        # Đếm số trang PDF
        try:
            page_count = len(PdfReader(str(source_pdf)).pages)
        except Exception:
            raise HTTPException(400, "Không thể đọc số trang PDF. Vui lòng tải lên tệp PDF hợp lệ.")

        if page_count < 1 or page_count > settings.max_pdf_pages:
            raise HTTPException(400, f"số trang PDF không hợp lệ. Vui lòng tải lên tệp PDF có số trang tối đa là {settings.max_pdf_pages}.")

        # Thêm công việc vào Redis Queue
        filename = Path(file.filename or "upload.pdf").name

        job = queue.enqueue(
            process_ocr_job,
            job_id,
            str(source_pdf),
            filename,
            job_id=job_id,
            job_timeout=settings.job_timeout_seconds,
            result_ttl=settings.job_retention_days * 86400,
            failure_ttl=settings.job_retention_days * 86400,
            meta={
                "source_filename": filename,
                "page_count": page_count,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        return {
            "job_id": job.id,
            "status": "queued",
            "page_count": page_count,
        }

    except Exception:
        shutil.rmtree(job_directory, ignore_errors=True)
        raise

    finally:
        await file.close()


# Lấy trạng thái công việc
@app.get("/api/v1/jobs/{job_id}")
def get_job_status(job_id: str,_: None = Depends(authenticate)):
    job = get_job(job_id)

    status = job.get_status(refresh=True)
    status = getattr(status, "value", status)

    status = str(status).lower()

    if status == "finished":
        status = "succeeded"
    elif status in ("started", "busy"):
        status = "running"
    elif status in ("queued", "deferred", "scheduled"):
        status = "queued"
    else:
        status = "failed"

    return {
        "job_id": job.id,
        "status": status,
        "source_filename": job.meta.get("source_filename"),
        "page_count": job.meta.get("page_count"),
        "created_at": job.meta.get("created_at"),
    }


# Lấy kết quả OCR
@app.get("/api/v1/jobs/{job_id}/result")
def get_job_result(
    job_id: str,
    _: None = Depends(authenticate),
):
    job = get_job(job_id)

    status = job.get_status(refresh=True)
    status = getattr(status, "value", status)

    if str(status).lower() != "finished":
        raise HTTPException(409, "Job chưa hoàn thành. Vui lòng kiểm tra lại trạng thái của job trước khi lấy kết quả.")

    result_path = settings.job_root / job_id / "result.json"

    if not result_path.is_file():
        raise HTTPException(404, "Không tìm thấy kết quả OCR. Vui lòng kiểm tra lại ID của job hoặc xác nhận rằng job đã hoàn thành.")

    return json.loads(result_path.read_text(encoding="utf-8"))
