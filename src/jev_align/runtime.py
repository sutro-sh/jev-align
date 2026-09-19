"""Small synchronous Python interface to a saved AI Function."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from .backends import EvaluationBackend, create_backend, validate_backend_for_task
from .capture import Capture
from .data import prepare_fields
from .models import Prediction, RunState, Story
from .persistence import RunStore


class AIFunction:
    """A loaded, callable AI Function backed by a saved run."""

    def __init__(
        self,
        state: RunState,
        backend: EvaluationBackend,
        *,
        capture: Capture | None,
        owns_capture: bool,
    ) -> None:
        self._state = state
        self._backend = backend
        self.capture = capture
        self._owns_capture = owns_capture
        self._closed = False
        self._definition: dict[str, object] = {
            "candidate": state.current_candidate.model_dump(mode="json"),
            "backend": state.backend.model_dump(mode="json"),
            "selected_columns": state.selected_columns,
            "column_mode": state.column_mode,
        }
        self._fingerprint = hashlib.sha256(
            json.dumps(self._definition, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if owns_capture:
            atexit.register(self.close)

    @classmethod
    def load(
        cls,
        run: str | Path,
        *,
        capture: bool | Capture = False,
    ) -> AIFunction:
        """Load the accepted definition from a run once.

        Pass ``capture=True`` to create and own a default production-call
        recorder, or pass a configured ``Capture`` instance. Pending GEPA
        proposals are never loaded.
        """
        state = RunStore(Path(run).resolve()).load_state()
        backend = create_backend(state.backend, concurrency=1)
        validate_backend_for_task(backend, state.current_candidate)
        if capture is True:
            directory = os.environ.get("JEVA_CAPTURE_DIR", ".jev-align/captures")
            recorder = Capture(Path(directory).expanduser())
            owns_capture = True
        elif capture is False:
            recorder = None
            owns_capture = False
        elif isinstance(capture, Capture):
            recorder = capture
            owns_capture = False
        else:
            raise TypeError("capture must be True, False, or a Capture instance")
        return cls(
            state,
            backend,
            capture=recorder,
            owns_capture=owns_capture,
        )

    def __call__(self, **inputs: Any) -> Prediction:
        """Evaluate named input fields once and return a normalized prediction."""
        columns = self._state.selected_columns
        if columns is None:
            legacy = [name for name in ("title", "text", "url") if name in inputs]
            columns = legacy or list(inputs)
        if not columns:
            raise ValueError("at least one input field is required")
        missing = [column for column in columns if column not in inputs]
        if missing:
            raise ValueError(f"missing input fields: {', '.join(missing)}")
        story = Story(
            id=uuid4().hex,
            row_number=1,
            fields=prepare_fields(
                inputs,
                columns,
                concatenate=self._state.column_mode == "all_concatenated",
            ),
        )
        predictions = self._backend.evaluate_many(
            self._state.current_candidate,
            [story],
        )
        if len(predictions) != 1 or predictions[0].story_id != story.id:
            raise ValueError("backend did not return the requested prediction")
        prediction = predictions[0]
        if self.capture is not None:
            self.capture.record(
                story,
                prediction,
                run_id=self._state.run_id,
                definition=self._definition,
                fingerprint=self._fingerprint,
            )
        return prediction

    def close(self) -> None:
        """Flush a recorder created by ``capture=True`` exactly once."""
        if self._closed:
            return
        self._closed = True
        if self._owns_capture and self.capture is not None:
            self.capture.close()

    def __enter__(self) -> AIFunction:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
