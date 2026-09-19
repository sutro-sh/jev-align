"""Small synchronous Python interface to a saved AI Function."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec
from uuid import uuid4

from .backends import create_backend, validate_backend_for_task
from .capture import Capture
from .data import prepare_fields
from .models import Prediction, Story
from .persistence import RunStore

P = ParamSpec("P")


def aligned(
    run: str | Path, *, capture: Capture | None = None
) -> Callable[[Callable[P, Mapping[str, Any]]], Callable[P, Prediction]]:
    """Turn an input-building function into a synchronous saved-function call.

    Load current_candidate once; never use a pending GEPA proposal. The wrapped
    function returns a row mapping. Apply the run's original column selection
    and normalization before evaluating it exactly once. Return the full
    provider-neutral Prediction, including uncertainty.
    """
    state = RunStore(Path(run).resolve()).load_state()
    backend = create_backend(state.backend, concurrency=1)
    validate_backend_for_task(backend, state.current_candidate)
    definition = {
        "candidate": state.current_candidate.model_dump(mode="json"),
        "backend": state.backend.model_dump(mode="json"),
        "selected_columns": state.selected_columns,
        "column_mode": state.column_mode,
    }
    fingerprint = hashlib.sha256(
        json.dumps(definition, sort_keys=True).encode("utf-8")
    ).hexdigest()

    def decorate(function: Callable[P, Mapping[str, Any]]) -> Callable[P, Prediction]:
        if inspect.iscoroutinefunction(function):
            raise TypeError("aligned currently supports synchronous functions only")

        @wraps(function)
        def evaluate(*args: P.args, **kwargs: P.kwargs) -> Prediction:
            row = function(*args, **kwargs)
            if not isinstance(row, Mapping):
                raise TypeError(
                    "an aligned function must return a mapping of input fields"
                )
            columns = state.selected_columns
            if columns is None:
                legacy = [name for name in ("title", "text", "url") if name in row]
                columns = legacy or list(row)
            if not columns:
                raise ValueError("at least one input field is required")
            missing = [column for column in columns if column not in row]
            if missing:
                raise ValueError(f"missing input fields: {', '.join(missing)}")
            story = Story(
                id=uuid4().hex,
                row_number=1,
                fields=prepare_fields(
                    row, columns, concatenate=state.column_mode == "all_concatenated"
                ),
            )
            predictions = backend.evaluate_many(state.current_candidate, [story])
            if len(predictions) != 1 or predictions[0].story_id != story.id:
                raise ValueError("backend did not return the requested prediction")
            prediction = predictions[0]
            if capture is not None:
                capture.record(
                    story,
                    prediction,
                    run_id=state.run_id,
                    definition=definition,
                    fingerprint=fingerprint,
                )
            return prediction

        return evaluate

    return decorate
