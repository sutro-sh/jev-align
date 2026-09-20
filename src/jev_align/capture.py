"""Best-effort, bounded local capture. Never makes evaluation requests."""

from __future__ import annotations

import json
import logging
import os
import random
import threading
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
from time import monotonic
from uuid import uuid4

from .acquisition import prediction_ambiguity
from .models import Prediction, Story

logger = logging.getLogger(__name__)


class Capture:
    """One bounded queue and background JSONL writer per instance.

    Use as a context manager or call close() at application shutdown. Create
    instances inside each worker process, after forking. Captures are disposable
    observations, never confirmed labels or changes to an alignment run.
    """

    def __init__(
        self,
        directory: str | Path = ".jev-align/captures",
        *,
        ambiguity_threshold: float = 0.8,
        audit_rate: float = 0.05,
        queue_size: int = 256,
        max_record_bytes: int = 64 * 1024,
        max_file_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if not 0 <= ambiguity_threshold <= 1 or not 0 <= audit_rate <= 1:
            raise ValueError("ambiguity_threshold and audit_rate must be in [0, 1]")
        if min(queue_size, max_record_bytes, max_file_bytes) < 1:
            raise ValueError("capture size limits must be positive")
        self.path = Path(directory).resolve() / f"{uuid4().hex}.jsonl"
        self.ambiguity_threshold = ambiguity_threshold
        self.audit_rate = audit_rate
        self.max_record_bytes = max_record_bytes
        self.max_file_bytes = max_file_bytes
        self._queue: Queue[bytes] = Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pid = os.getpid()
        self._reserved_bytes = 0
        self._written = 0
        self._dropped = 0
        self._error: str | None = None
        self._thread = threading.Thread(
            target=self._write_loop, name="jev-align-capture", daemon=True
        )
        self._thread.start()

    @property
    def stats(self) -> dict[str, int | str | None]:
        """Counters for selected records; filtered-out calls are not drops."""
        if os.getpid() != self._pid:
            return {"written": 0, "dropped": 0, "error": "create Capture after fork"}
        with self._lock:
            return {
                "written": self._written,
                "dropped": self._dropped,
                "error": self._error,
            }

    def record(
        self,
        story: Story,
        prediction: Prediction,
        *,
        run_id: str,
        definition: dict[str, object],
        fingerprint: str,
    ) -> None:
        # An inherited writer has no running thread. Avoid its inherited locks.
        if os.getpid() != self._pid:
            return
        uncertainty = prediction_ambiguity(prediction)
        reason = "ambiguous"
        if uncertainty < self.ambiguity_threshold:
            if random.random() >= self.audit_rate:
                return
            reason = "audit"
        with self._lock:
            if (
                self._stop.is_set()
                or self._queue.full()
                or self._reserved_bytes >= self.max_file_bytes
                or sum(len(value) for value in story.fields.values())
                > self.max_record_bytes
            ):
                self._dropped += 1
                return
        payload = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "definition": definition,
            "fingerprint": fingerprint,
            "input": story.fields,
            "prediction": prediction.model_dump(mode="json"),
            "ambiguity": uncertainty,
            "reason": reason,
        }
        line = (json.dumps(payload) + "\n").encode("utf-8")
        with self._lock:
            if (
                self._stop.is_set()
                or len(line) > self.max_record_bytes
                or self._reserved_bytes + len(line) > self.max_file_bytes
            ):
                self._dropped += 1
                return
            try:
                self._queue.put_nowait(line)
            except Full:
                self._dropped += 1
            else:
                self._reserved_bytes += len(line)

    def _write_loop(self) -> None:
        batch: list[bytes] = []
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("xb") as handle:
                deadline = monotonic() + 0.25
                while not self._stop.is_set() or not self._queue.empty():
                    try:
                        batch.append(
                            self._queue.get(timeout=max(0, deadline - monotonic()))
                        )
                    except Empty:
                        pass
                    if len(batch) >= 64 or monotonic() >= deadline:
                        handle.write(b"".join(batch))
                        handle.flush()
                        with self._lock:
                            self._written += len(batch)
                        batch.clear()
                        deadline = monotonic() + 0.25
                handle.write(b"".join(batch))
                handle.flush()
                with self._lock:
                    self._written += len(batch)
                batch.clear()
        except OSError as error:
            with self._lock:
                self._error = str(error)
                self._stop.set()
                self._dropped += len(batch)
                while True:
                    try:
                        self._queue.get_nowait()
                        self._dropped += 1
                    except Empty:
                        break
            logger.warning("JEV capture disabled after a local write error: %s", error)

    def close(self, timeout: float = 5.0) -> None:
        """Stop accepting records and wait up to timeout seconds for the writer.

        Buffered records may be lost on abrupt exit or if shutdown times out.
        No fsync or durability guarantee is added to the request path.
        """
        if os.getpid() != self._pid:
            return
        with self._lock:
            self._stop.set()
        self._thread.join(timeout=timeout)

    def __enter__(self) -> Capture:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
