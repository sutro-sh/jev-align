"""Read deduplicated captured inputs without model calls."""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path

from .models import RunState, Story

logger = logging.getLogger(__name__)


def captured_story_id(fields: dict[str, str]) -> str:
    payload = json.dumps(fields, sort_keys=True).encode("utf-8")
    return "capture-" + hashlib.sha256(payload).hexdigest()


def sample_captured_inputs(
    directories: Iterable[Path],
    state: RunState,
    *,
    excluded: Iterable[Story],
    saved: Iterable[Story] = (),
    required: Iterable[Story] = (),
    limit: int | None = None,
) -> list[Story]:
    """Keep required inputs plus all other unique inputs by default.

    An optional limit is retained for callers that explicitly request sampling;
    required inputs do not count toward it. Stable hash priorities avoid
    favoring frequent duplicates or the first file scanned when a limit is used.
    Read only complete lines present when each file is opened, so active writers
    cannot extend the scan.
    """
    if limit is not None and limit < 1:
        raise ValueError("capture sample limit must be positive")
    excluded_ids = {captured_story_id(story.fields) for story in excluded}
    required_by_id = {
        story.id: story for story in required if story.id not in excluded_ids
    }
    selected: dict[str, Story] = {}
    heap: list[tuple[int, str]] = []

    def consider(story: Story) -> None:
        if (
            story.id in excluded_ids
            or story.id in required_by_id
            or story.id in selected
        ):
            return
        if limit is None:
            selected[story.id] = story
            return
        priority = int(
            hashlib.sha256(
                f"{state.seed}:{state.round_number}:{story.id}".encode()
            ).hexdigest(),
            16,
        )
        if len(heap) == limit:
            if priority >= -heap[0][0]:
                return
            _, removed = heapq.heappop(heap)
            del selected[removed]
        heapq.heappush(heap, (-priority, story.id))
        selected[story.id] = story

    for story in saved:
        consider(story)
    paths = {path for directory in directories for path in directory.glob("*.jsonl")}
    for path in sorted(paths):
        try:
            with path.open("rb") as handle:
                boundary = os.fstat(handle.fileno()).st_size
                oversized = False
                while handle.tell() < boundary:
                    line = handle.readline(min(1024 * 1024, boundary - handle.tell()))
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        oversized = True
                        continue
                    if oversized:
                        oversized = False
                        continue
                    try:
                        row = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    if not isinstance(row, dict) or row.get("version") != 1:
                        continue
                    if row.get("run_id") != state.run_id:
                        continue
                    fields = row.get("input")
                    if (
                        not isinstance(fields, dict)
                        or not fields
                        or not all(
                            isinstance(key, str) and isinstance(value, str)
                            for key, value in fields.items()
                        )
                    ):
                        continue
                    expected = (
                        ["content"]
                        if state.column_mode == "all_concatenated"
                        else state.selected_columns
                    )
                    if expected is not None and set(fields) != set(expected):
                        continue
                    consider(
                        Story(id=captured_story_id(fields), row_number=0, fields=fields)
                    )
        except OSError as error:
            logger.warning("Could not read capture file %s: %s", path, error)
    selected_ids = (
        sorted(selected) if limit is None else [story_id for _, story_id in sorted(heap)]
    )
    return [
        *(required_by_id[story_id] for story_id in sorted(required_by_id)),
        *(selected[story_id] for story_id in selected_ids),
    ]
