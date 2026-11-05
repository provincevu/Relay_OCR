
"""xóa các job OCR đã hết hạn trong Redis và thư mục output."""

import argparse
import shutil
import time
from pathlib import Path

from redis import Redis
from rq.exceptions import NoSuchJobError
from rq.job import Job

from .settings import settings


TERMINAL_STATUSES = {"finished", "failed", "stopped", "canceled"}

# hàm xóa các job OCR đã hết hạn trong Redis và thư mục output, dựa trên thời gian giữ lại job được cấu hình trong settings
def cleanup_jobs(job_root, redis, retention_days, apply=False):
    if not job_root.exists():
        return []

    cutoff = time.time() - retention_days * 86400
    candidates = []

    # lặp qua các thư mục con trong job_root để kiểm tra xem job nào đã hết hạn
    for directory in job_root.iterdir():

        # do job id được tạo từ UUID4 dạng .hex có độ dài 32 ký tự, folder không đúng 32 ký tự thì không phải là job hợp lệ
        if not directory.is_dir() or len(directory.name) != 32:
            continue

        if directory.stat().st_mtime > cutoff:
            continue

        # kiểm tra trạng thái của job trong Redis, nếu job đã hoàn thành hoặc thất bại thì có thể xóa
        try:
            job = Job.fetch(directory.name, connection=redis)
            status = job.get_status(refresh=True)
            status = getattr(status, "value", status)

            # không phải (finished, failed, stopped, canceled) thì không xóa
            if str(status).lower() not in TERMINAL_STATUSES:
                continue

        except NoSuchJobError:
            pass

        candidates.append(directory)

        if apply:
            shutil.rmtree(directory)
            print(f"Removed: {directory}")

    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    redis = Redis.from_url(settings.redis_url)

    try:
        candidates = cleanup_jobs(
            settings.job_root,
            redis,
            settings.job_retention_days,
            apply=args.apply,
        )
    finally:
        redis.close()

    if not candidates:
        print("không thấy job nào cần xóa.")
    elif not args.apply:
        for directory in candidates:
            print(f"Bạn có muốn xóa?: {directory}")

        print("chỉ liệt kê preview, sử dụng --apply để xóa thực sự.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
