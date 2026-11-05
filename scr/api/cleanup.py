"""Retention cleanup for per-job files stored outside Redis."""

import argparse
import logging
import shutil
import time
from pathlib import Path

from redis import Redis
from rq.exceptions import NoSuchJobError
from rq.job import Job

from .settings import settings


logger = logging.getLogger(__name__)
JOB_ID_LENGTH = 32
TERMINAL_STATUSES = {"finished", "failed", "stopped", "canceled"}


def cleanup_expired_job_directories(
    job_root: Path,
    redis_connection: Redis,
    retention_days: int,
    *,
    dry_run: bool = False,
) -> list[Path]:
    """Remove old per-job directories, preserving jobs that can still run.

    A directory is eligible only after it has been unchanged for the retention
    period and its RQ job is either gone or in a terminal state. If Redis is
    unavailable, Redis errors propagate so callers can skip cleanup safely.
    """
    if retention_days < 1:
        raise ValueError("retention_days must be greater than zero")

    root = job_root.resolve()
    if not root.exists():
        return []
    if not root.is_dir():
        raise NotADirectoryError(root)

    redis_connection.ping()
    cutoff = time.time() - retention_days * 24 * 60 * 60
    removed_or_would_remove: list[Path] = []

    for directory in root.iterdir():
        if directory.is_symlink() or not directory.is_dir():
            continue
        if len(directory.name) != JOB_ID_LENGTH or any(
            character not in "0123456789abcdef" for character in directory.name
        ):
            continue

        resolved_directory = directory.resolve()
        if resolved_directory.parent != root:
            continue
        try:
            if directory.stat().st_mtime > cutoff:
                continue
        except FileNotFoundError:
            continue

        try:
            job = Job.fetch(directory.name, connection=redis_connection)
        except NoSuchJobError:
            job = None

        if job is not None:
            raw_status = job.get_status(refresh=True)
            status_value = getattr(raw_status, "value", raw_status)
            if str(status_value).lower() not in TERMINAL_STATUSES:
                continue

        removed_or_would_remove.append(directory)
        if not dry_run:
            shutil.rmtree(directory)
            logger.info("Removed expired OCR job directory: %s", directory)

    return removed_or_would_remove


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preview or remove expired OCR job directories."
    )
    parser.add_argument(
        "--job-root",
        type=Path,
        default=settings.job_root,
        help="Job directory root (defaults to OCR_JOB_ROOT).",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=settings.job_retention_days,
        help="Minimum directory age before cleanup (defaults to JOB_RETENTION_DAYS).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually remove eligible directories; without this, only preview.",
    )
    args = parser.parse_args()

    connection = Redis.from_url(settings.redis_url)
    try:
        candidates = cleanup_expired_job_directories(
            args.job_root,
            connection,
            args.retention_days,
            dry_run=not args.apply,
        )
    finally:
        connection.close()

    action = "Removed" if args.apply else "Would remove"
    if not candidates:
        print("No expired job directories found.")
    else:
        for candidate in candidates:
            print(f"{action}: {candidate}")
        if not args.apply:
            print("Preview only. Add --apply to remove these directories.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
