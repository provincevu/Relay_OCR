
"""Redis Queue worker cho hệ thống OCR, cho phép xử lý các job OCR được gửi từ API."""

import os
from pathlib import Path

from redis import Redis
from rq import Queue, SimpleWorker
from rq.timeouts import TimerDeathPenalty

from .settings import settings

# khởi tạo worker cho hệ thống OCR, sử dụng Redis để quản lý queue và các job OCR
class WindowsWorker(SimpleWorker):
    death_penalty_class = TimerDeathPenalty


def main():
    # đặt thư mục làm việc hiện tại là thư mục gốc của dự án, để tránh lỗi khi chạy worker từ các vị trí khác nhau
    os.chdir(Path(__file__).resolve().parents[2])

    # Connect tới Redis
    redis = Redis.from_url(settings.redis_url)
    redis.ping()

    # tạo queue và worker
    queue = Queue(settings.queue_name, connection=redis)
    worker = WindowsWorker([queue], connection=redis)

    print(f"worker đang lắng nghe queue {settings.queue_name} ...", flush=True)

    # Start worker
    worker.work()


if __name__ == "__main__":
    main()
