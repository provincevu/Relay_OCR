FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OCR_CONFIG_PATH=/app/config/ocr.cpu.json \
    OCR_JOB_ROOT=/var/lib/relay-ocr/jobs

WORKDIR /app

# Ghostscript renders PDFs; these shared libraries are needed by OpenCV and
# CPU inference libraries at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ghostscript \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-cpu.txt ./requirements-cpu.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-cpu.txt

# Checkpoint is intentionally copied into this image for the selected
# deployment mode. PDF/data/output folders are excluded by .dockerignore.
COPY . .

RUN mkdir -p /var/lib/relay-ocr/jobs

# Compose will override this command for the API and worker containers.
