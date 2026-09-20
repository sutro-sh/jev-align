"""Portable, public AI Function artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator

from .backends import task_kind
from .data import dataset_columns, load_stories, source_sha256
from .models import (
    BackendConfig,
    CandidateHistory,
    LabelRecord,
    RunState,
    TaskSpec,
)
from .persistence import RunStore


class ArtifactError(ValueError):
    """The saved run cannot be represented as a complete public artifact."""


class ArtifactInputs(BaseModel):
    columns: list[str]
    mode: Literal["selected", "all_concatenated"]


class ArtifactAnnotation(BaseModel):
    inputs: dict[str, str]
    label: bool | int | str | list[str]
    rationale: str | None = None
    split: Literal["train", "holdout"]

    @field_validator("inputs")
    @classmethod
    def inputs_nonempty(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("annotations require at least one input field")
        return value


class ArtifactMetrics(BaseModel):
    training_score: float | None = None
    holdout_score: float | None = None


class ArtifactLearning(BaseModel):
    reflection_model: str
    metric_budget: int
    batch_size: int
    holdout_fraction: float
    concurrency: int
    seed: int


class FunctionArtifact(BaseModel):
    schema_version: Literal[1] = 1
    name: str
    slug: str
    description: str | None = None
    task_type: Literal["binary", "multiclass", "multilabel", "score"]
    inputs: ArtifactInputs
    definition: dict[str, Any]
    presentation: dict[str, Any] = Field(default_factory=dict)
    backend: BackendConfig
    annotations: list[ArtifactAnnotation]
    learning: ArtifactLearning
    metrics: ArtifactMetrics

    @field_validator("name")
    @classmethod
    def name_valid(cls, value: str) -> str:
        value = value.strip()
        if not 1 <= len(value) <= 80:
            raise ValueError("function name must be between 1 and 80 characters")
        return value

    @field_validator("slug")
    @classmethod
    def slug_valid(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?", value):
            raise ValueError(
                "function slug must use 1–64 lowercase letters, numbers, or hyphens"
            )
        return value

    @field_validator("description")
    @classmethod
    def description_valid(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if len(value) > 280:
            raise ValueError("function description must be 280 characters or fewer")
        return value

    @model_validator(mode="after")
    def portable_inputs_valid(self) -> "FunctionArtifact":
        if not self.inputs.columns:
            raise ValueError("function artifacts require at least one input column")
        if not self.annotations:
            raise ValueError("function artifacts require at least one human annotation")
        expected = (
            set(self.inputs.columns)
            if self.inputs.mode == "selected"
            else {"content"}
        )
        if any(set(annotation.inputs) != expected for annotation in self.annotations):
            raise ValueError("annotation inputs do not match the function signature")
        return self


def function_slug(value: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    slug = slug[:64].rstrip("-")
    if not slug:
        raise ArtifactError("function name must contain at least one letter or number")
    return slug


def _latest_scores(state: RunState) -> ArtifactMetrics:
    accepted = [item for item in state.history if item.decision == "accepted"]
    if not accepted:
        return ArtifactMetrics()
    latest = accepted[-1]
    return ArtifactMetrics(
        training_score=latest.fit_f1,
        holdout_score=latest.holdout_score,
    )


def _selected_columns(state: RunState, labels: list[LabelRecord]) -> list[str]:
    if state.selected_columns:
        return list(state.selected_columns)
    for label in labels:
        if label.inputs:
            return list(label.inputs)
    source = Path(state.source_path)
    if source.exists():
        available = dataset_columns(source)
        legacy = [name for name in ("title", "text", "url") if name in available]
        return legacy or available
    raise ArtifactError("the input columns for this legacy run cannot be reconstructed")


def _story_inputs(
    state: RunState,
    store: RunStore,
    labels: list[LabelRecord],
    columns: list[str],
) -> dict[str, dict[str, str]]:
    resolved = {
        label.story_id: dict(label.inputs)
        for label in labels
        if label.inputs is not None
    }
    missing = {label.story_id for label in labels if label.inputs is None}
    if not missing:
        return resolved

    for story in store.load_captured_inputs():
        if story.id in missing:
            resolved[story.id] = dict(story.fields)
            missing.remove(story.id)
    if not missing:
        return resolved

    source = Path(state.source_path)
    if not source.exists():
        raise ArtifactError(
            "some legacy annotations only reference the original dataset, which no "
            "longer exists"
        )
    if source_sha256(source) != state.source_sha256:
        raise ArtifactError(
            "the original dataset changed, so legacy annotation inputs cannot be "
            "reconstructed safely"
        )
    stories = load_stories(
        source,
        state.pool_size,
        columns,
        concatenate=state.column_mode == "all_concatenated",
    )
    for story in stories:
        if story.id in missing:
            resolved[story.id] = dict(story.fields)
            missing.remove(story.id)
    if missing:
        raise ArtifactError(
            f"could not reconstruct {len(missing)} labeled input(s) from saved run data"
        )
    return resolved


def build_function_artifact(
    run_directory: Path,
    *,
    name: str,
    slug: str | None = None,
    description: str | None = None,
) -> FunctionArtifact:
    store = RunStore(run_directory.resolve())
    state = store.load_state()
    if state.pending_candidate is not None:
        raise ArtifactError(
            "this run has a pending proposal; resume it and accept or reject the "
            "proposal before publishing"
        )
    labels = store.load_labels()
    if not labels:
        raise ArtifactError("an AI Function needs at least one human label before publishing")
    columns = _selected_columns(state, labels)
    inputs_by_story = _story_inputs(state, store, labels, columns)
    annotations = [
        ArtifactAnnotation(
            inputs=inputs_by_story[label.story_id],
            label=label.label,
            rationale=label.rationale,
            split=label.evaluation_split,
        )
        for label in labels
    ]
    presentation: dict[str, Any] = {}
    if task_kind(state.current_candidate) == "binary":
        presentation = {
            "true_label": state.binary_true_label,
            "false_label": state.binary_false_label,
        }
    return FunctionArtifact(
        name=name,
        slug=slug or function_slug(name),
        description=description,
        task_type=task_kind(state.current_candidate),
        inputs=ArtifactInputs(columns=columns, mode=state.column_mode),
        definition=state.current_candidate.model_dump(mode="json"),
        presentation=presentation,
        backend=BackendConfig(
            provider=state.backend.provider,
            model=state.backend.model,
        ),
        annotations=annotations,
        learning=ArtifactLearning(
            reflection_model=state.reflection_model,
            metric_budget=state.metric_budget,
            batch_size=state.batch_size,
            holdout_fraction=state.holdout_fraction,
            concurrency=state.concurrency,
            seed=state.seed,
        ),
        metrics=_latest_scores(state),
    )


def artifact_bytes(artifact: FunctionArtifact) -> bytes:
    return (
        json.dumps(
            artifact.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def materialize_function_artifact(
    artifact: FunctionArtifact,
    target: Path,
    *,
    reference: str,
    version: int,
    digest: str,
    registry: str,
) -> Path:
    """Create a normal saved run from a verified portable artifact."""
    target = target.expanduser().resolve()
    if target.exists():
        raise ArtifactError(f"destination already exists: {target}")
    candidate = TypeAdapter(TaskSpec).validate_python(artifact.definition)
    if task_kind(candidate) != artifact.task_type:
        raise ArtifactError("artifact task type does not match its definition")

    if artifact.inputs.mode == "selected":
        rows = [annotation.inputs for annotation in artifact.annotations]
        selected_columns = list(artifact.inputs.columns)
        column_mode: Literal["selected", "all_concatenated"] = "selected"
    else:
        if any(set(annotation.inputs) != {"content"} for annotation in artifact.annotations):
            raise ArtifactError("concatenated artifacts require a content input")
        rows = [annotation.inputs for annotation in artifact.annotations]
        # Schema v1 artifacts preserve the exact concatenated evaluator input but
        # not its pre-concatenation fields. Keep the function runnable and resumable
        # through the lossless content signature.
        selected_columns = ["content"]
        column_mode = "selected"
    source_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    source = target / "labeled-data.jsonl"
    holdout_ids = [
        f"row-{index:06d}"
        for index, annotation in enumerate(artifact.annotations, start=1)
        if annotation.split == "holdout"
    ]
    true_label = str(artifact.presentation.get("true_label") or "True")
    false_label = str(artifact.presentation.get("false_label") or "False")
    state = RunState(
        run_id=target.name,
        source_path=str(source),
        source_sha256=hashlib.sha256(source_payload).hexdigest(),
        selected_columns=selected_columns,
        column_mode=column_mode,
        pool_size=len(rows),
        seed=artifact.learning.seed,
        backend=artifact.backend,
        reflection_model=artifact.learning.reflection_model,
        metric_budget=artifact.learning.metric_budget,
        concurrency=artifact.learning.concurrency,
        batch_size=artifact.learning.batch_size,
        holdout_fraction=0.2 if holdout_ids else 0.0,
        holdout_story_ids=holdout_ids,
        binary_true_label=true_label,
        binary_false_label=false_label,
        current_candidate=candidate,
        history=[
            CandidateHistory(
                round_number=0,
                candidate=candidate,
                decision="accepted",
                fit_f1=artifact.metrics.training_score,
                holdout_score=artifact.metrics.holdout_score,
            )
        ],
    )
    store = RunStore(target)
    store.initialize(state)
    source.write_bytes(source_payload)
    for index, annotation in enumerate(artifact.annotations, start=1):
        store.append_label(
            LabelRecord(
                story_id=f"row-{index:06d}",
                inputs=dict(annotation.inputs),
                label=annotation.label,
                rationale=annotation.rationale,
                round_number=0,
                acquired_by=(
                    "holdout" if annotation.split == "holdout" else "ambiguous"
                ),
                evaluation_split=annotation.split,
                acquisition_probability=0.5,
            )
        )
    store.write_json(
        "published.json",
        {
            "registry": registry,
            "reference": reference,
            "version": version,
            "digest": digest,
            "name": artifact.name,
            "slug": artifact.slug,
            "description": artifact.description,
            "url": f"{registry}/{reference}",
        },
    )
    return target
