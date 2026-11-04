"""Windows-compatible RQ worker entrypoint for Relay OCR."""

import os
from pathlib import Path

from redis import Redis
from rq import Queue, SimpleWorker
from rq.timeouts import TimerDeathPenalty

from .settings import settings


class WindowsSimpleWorker(SimpleWorker):
    """Use a timeout mechanism supported by Windows."""

    death_penalty_class = TimerDeathPenalty


def main():
    """Run one sequential SimpleWorker for the local OCR queue."""
    repository_root = Path(__file__).resolve().parents[2]
    os.chdir(repository_root)

    connection = Redis.from_url(settings.redis_url)
    connection.ping()
    queue = Queue(settings.queue_name, connection=connection)
    worker = WindowsSimpleWorker(
        [queue],
        connection=connection,
        name=f"relay-ocr-{os.getpid()}",
    )
    print(
        f"RQ WindowsSimpleWorker listening on queue '{settings.queue_name}' "
        f"at {settings.redis_url}",
        flush=True,
    )
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
