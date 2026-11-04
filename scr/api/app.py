"""Authenticated local API for submitting and polling OCR jobs."""

import hmac
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from pypdf import PdfReader
from redis import Redis
from redis.exceptions import RedisError
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job

from .settings import settings
from .tasks import process_ocr_job


app = FastAPI(
    title="Relay OCR Local API",
    version="1.0.0",
    description="Local asynchronous PDF OCR API backed by Redis Queue.",
)

redis_connection = Redis.from_url(settings.redis_url)
ocr_queue = Queue(settings.queue_name, connection=redis_connection)
JOB_RETENTION_SECONDS = settings.job_retention_days * 24 * 60 * 60
MAX_UPLOAD_BYTES = settings.max_upload_mb * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 64 * 1024


@app.middleware("http")
async def reject_oversized_upload(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/api/v1/jobs":
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                request_bytes = int(content_length)
            except ValueError:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Invalid request content length."},
                )
            if request_bytes > MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "detail": (
                            f"PDF exceeds the {settings.max_upload_mb} MB upload limit."
                        ),
                    },
                )
    return await call_next(request)


def require_api_token(authorization: str | None = Header(default=None)):
    if not settings.api_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR_API_TOKEN is not configured.",
        )
    expected = f"Bearer {settings.api_token}"
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _job_or_404(job_id):
    try:
        return Job.fetch(job_id, connection=redis_connection)
    except NoSuchJobError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found or expired.",
        ) from error
    except RedisError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis is unavailable.",
        ) from error


def _normalized_job_status(job):
    raw_status = job.get_status(refresh=True)
    value = getattr(raw_status, "value", raw_status)
    value = str(value).lower()
    if value in {"finished", "successful"}:
        return "succeeded"
    if value in {"started", "busy"}:
        return "running"
    if value in {"failed", "stopped"}:
        return "failed"
    if value in {"queued", "deferred", "scheduled", "ready_to_enqueue"}:
        return "queued"
    return "failed"


@app.get("/health/live")
def live():
    return {"status": "live"}


@app.get("/health/ready")
def ready():
    try:
        redis_connection.ping()
    except RedisError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis is unavailable.",
        ) from error
    return {"status": "ready", "queue": settings.queue_name}


@app.post(
    "/api/v1/jobs",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_api_token)],
)
async def submit_job(file: UploadFile):
    job_id = uuid4().hex
    job_directory = settings.job_root / job_id
    temporary_pdf = job_directory / "source.upload"
    source_pdf = job_directory / "source.pdf"
    job_directory.mkdir(parents=True, exist_ok=False)

    total_bytes = 0
    first_chunk = b""
    try:
        with temporary_pdf.open("wb") as output_file:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                if not first_chunk:
                    first_chunk = chunk[:8]
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"PDF exceeds the {settings.max_upload_mb} MB upload limit.",
                    )
                output_file.write(chunk)

        if not first_chunk.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Uploaded file is not a PDF.",
            )
        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded PDF is empty.",
            )

        temporary_pdf.replace(source_pdf)
        try:
            page_count = len(PdfReader(str(source_pdf), strict=False).pages)
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded PDF cannot be read.",
            ) from error

        if page_count < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="PDF contains no pages.",
            )
        if page_count > settings.max_pdf_pages:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"PDF exceeds the {settings.max_pdf_pages} page limit.",
            )

        original_filename = Path(file.filename or "upload.pdf").name
        job = ocr_queue.enqueue(
            process_ocr_job,
            job_id,
            str(source_pdf),
            original_filename,
            job_id=job_id,
            job_timeout=settings.job_timeout_seconds,
            result_ttl=JOB_RETENTION_SECONDS,
            failure_ttl=JOB_RETENTION_SECONDS,
            meta={
                "source_filename": original_filename,
                "page_count": page_count,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return {
            "job_id": job.id,
            "status": "queued",
            "status_url": f"/api/v1/jobs/{job.id}",
            "result_url": f"/api/v1/jobs/{job.id}/result",
            "page_count": page_count,
        }
    except HTTPException:
        shutil.rmtree(job_directory, ignore_errors=True)
        raise
    except RedisError as error:
        shutil.rmtree(job_directory, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis is unavailable; the job was not queued.",
        ) from error
    except Exception:
        shutil.rmtree(job_directory, ignore_errors=True)
        raise
    finally:
        await file.close()


@app.get(
    "/api/v1/jobs/{job_id}",
    dependencies=[Depends(require_api_token)],
)
def get_job(job_id: str):
    job = _job_or_404(job_id)
    try:
        job.refresh()
        normalized_status = _normalized_job_status(job)
        response = {
            "job_id": job.id,
            "status": normalized_status,
            "created_at": job.meta.get("created_at"),
            "source_filename": job.meta.get("source_filename"),
            "page_count": job.meta.get("page_count"),
            "result_url": f"/api/v1/jobs/{job.id}/result",
        }
        if normalized_status == "failed":
            response["error"] = "OCR job failed. Check worker logs for details."
        return response
    except RedisError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis is unavailable.",
        ) from error


@app.get(
    "/api/v1/jobs/{job_id}/result",
    dependencies=[Depends(require_api_token)],
)
def get_job_result(job_id: str):
    job = _job_or_404(job_id)
    try:
        normalized_status = _normalized_job_status(job)
        if normalized_status != "succeeded":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job result is not ready (status: {normalized_status}).",
            )

        result_path = settings.job_root / job.id / "result.json"
        if not result_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job result file not found.",
            )
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Job result could not be read.",
            ) from error
        return JSONResponse(content=payload)
    except RedisError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis is unavailable.",
        ) from error
