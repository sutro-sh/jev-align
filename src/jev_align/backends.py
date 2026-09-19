from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from .models import (
    BackendConfig,
    MulticlassCandidateSpec,
    MultilabelCandidateSpec,
    Prediction,
    ScoreCandidateSpec,
    Story,
    TaskSpec,
)

TaskKind = Literal["binary", "multiclass", "multilabel", "score"]
UncertaintyKind = Literal["calibrated_probability", "native_confidence"]


@dataclass(frozen=True)
class BackendCapabilities:
    """What an evaluation backend can reliably provide to active learning."""

    task_types: frozenset[TaskKind]
    uncertainty: Mapping[TaskKind, UncertaintyKind]


class EvaluationBackend(Protocol):
    backend_id: str
    display_name: str
    model: str
    capabilities: BackendCapabilities

    def evaluate_many(
        self, candidate: TaskSpec, stories: Sequence[Story]
    ) -> list[Prediction]: ...


def create_backend(
    config: BackendConfig, *, concurrency: int
) -> EvaluationBackend:
    """Construct the configured adapter without leaking providers into the core."""
    if config.provider == "typesafe":
        from .jev import TypeSafeJevEvaluator

        return TypeSafeJevEvaluator(model=config.model, concurrency=concurrency)
    raise ValueError(f"unsupported evaluation backend: {config.provider}")


def backend_credential_error(
    config: BackendConfig, environment: Mapping[str, str]
) -> str | None:
    if config.provider == "typesafe" and not environment.get("TYPESAFE_API_KEY"):
        return "TYPESAFE_API_KEY must be set for the TypeSafe backend"
    return None


def backend_display_name(backend: EvaluationBackend) -> str:
    """Give minimal third-party adapters a useful name without extra boilerplate."""
    return str(getattr(backend, "display_name", backend.__class__.__name__))


def task_kind(candidate: TaskSpec) -> TaskKind:
    if isinstance(candidate, MultilabelCandidateSpec):
        return "multilabel"
    if isinstance(candidate, ScoreCandidateSpec):
        return "score"
    if isinstance(candidate, MulticlassCandidateSpec):
        return "multiclass"
    return "binary"


def validate_backend_for_task(
    backend: EvaluationBackend, candidate: TaskSpec
) -> None:
    """Reject adapters that cannot supply uncertainty for the requested task."""
    kind = task_kind(candidate)
    capabilities = backend.capabilities
    if kind not in capabilities.task_types:
        raise ValueError(
            f"{backend_display_name(backend)} does not support {kind} AI Functions"
        )
    if kind not in capabilities.uncertainty:
        raise ValueError(
            f"{backend_display_name(backend)} does not provide usable uncertainty "
            f"for {kind} AI Functions"
        )


def evaluate_with_progress(
    backend: EvaluationBackend,
    candidate: TaskSpec,
    stories: Sequence[Story],
    description: str,
) -> list[Prediction]:
    """Use progress when supported, while preserving lightweight test adapters."""
    method = getattr(backend, "evaluate_many_with_progress", None)
    if callable(method):
        return method(candidate, stories, description)
    return backend.evaluate_many(candidate, stories)
