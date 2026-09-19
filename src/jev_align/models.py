from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Story(BaseModel):
    id: str
    row_number: int
    fields: dict[str, str]

    def state(self) -> dict[str, str]:
        return dict(self.fields)


class BinaryTaskSpec(BaseModel):
    instructions: str
    true_criteria: str
    false_criteria: str

    @field_validator("instructions", "true_criteria", "false_criteria")
    @classmethod
    def nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("candidate components must not be empty")
        return value

    def as_gepa(self) -> dict[str, str]:
        return self.model_dump()

    @classmethod
    def from_gepa(cls, value: Any) -> "BinaryTaskSpec":
        if not isinstance(value, dict):
            raise ValueError("GEPA candidate must be a component dictionary")
        expected = {"instructions", "true_criteria", "false_criteria"}
        if set(value) != expected:
            raise ValueError(f"candidate components must be exactly {sorted(expected)}")
        return cls.model_validate(value)


class MulticlassTaskSpec(BaseModel):
    instructions: str
    criteria: dict[str, str]

    @field_validator("instructions")
    @classmethod
    def instructions_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("candidate instructions must not be empty")
        return value

    @field_validator("criteria")
    @classmethod
    def criteria_valid(cls, value: dict[str, str]) -> dict[str, str]:
        cleaned = {
            name.strip(): description.strip() for name, description in value.items()
        }
        if len(cleaned) < 2 or any(
            not name or not description for name, description in cleaned.items()
        ):
            raise ValueError(
                "multiclass candidates require at least two described classes"
            )
        return cleaned

    def as_gepa(self) -> dict[str, str]:
        return {
            "instructions": self.instructions,
            **{
                f"criterion::{name}": description
                for name, description in self.criteria.items()
            },
        }

    @classmethod
    def from_gepa(cls, value: Any, class_names: list[str]) -> "MulticlassTaskSpec":
        if not isinstance(value, dict):
            raise ValueError("GEPA candidate must be a component dictionary")
        expected = {"instructions", *(f"criterion::{name}" for name in class_names)}
        if set(value) != expected:
            raise ValueError(f"candidate components must be exactly {sorted(expected)}")
        return cls(
            instructions=value["instructions"],
            criteria={name: value[f"criterion::{name}"] for name in class_names},
        )


class ScoreTaskSpec(BaseModel):
    instructions: str
    levels: list[str]

    @field_validator("instructions")
    @classmethod
    def instructions_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("task instructions must not be empty")
        return value

    @field_validator("levels")
    @classmethod
    def levels_valid(cls, value: list[str]) -> list[str]:
        cleaned = [description.strip() for description in value]
        if not 2 <= len(cleaned) <= 10 or any(not item for item in cleaned):
            raise ValueError("Score tasks require 2 to 10 described levels")
        return cleaned

    def as_gepa(self) -> dict[str, str]:
        return {
            "instructions": self.instructions,
            **{
                f"level::{index}": description
                for index, description in enumerate(self.levels)
            },
        }

    @classmethod
    def from_gepa(cls, value: Any, level_count: int) -> "ScoreTaskSpec":
        if not isinstance(value, dict):
            raise ValueError("GEPA candidate must be a component dictionary")
        expected = {
            "instructions",
            *(f"level::{index}" for index in range(level_count)),
        }
        if set(value) != expected:
            raise ValueError(f"candidate components must be exactly {sorted(expected)}")
        components = [value["instructions"]] + [
            value[f"level::{index}"] for index in range(level_count)
        ]
        if not all(isinstance(component, str) for component in components):
            raise ValueError("Score components must be text")
        if any(
            marker in component
            for component in components
            for marker in ("level::", "criterion::", "label::")
        ):
            raise ValueError(
                "Score components must not contain names or content from other components"
            )
        if len(value["instructions"]) > 1200:
            raise ValueError("Score instructions must be at most 1,200 characters")
        if any(len(value[f"level::{index}"]) > 600 for index in range(level_count)):
            raise ValueError("Score level descriptions must be at most 600 characters")
        return cls(
            instructions=value["instructions"],
            levels=[value[f"level::{index}"] for index in range(level_count)],
        )


class MultilabelCriteria(BaseModel):
    true_criteria: str
    false_criteria: str

    @field_validator("true_criteria", "false_criteria")
    @classmethod
    def nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("multilabel criteria must not be empty")
        return value


class MultilabelTaskSpec(BaseModel):
    instructions: str
    labels: dict[str, MultilabelCriteria]

    @field_validator("instructions")
    @classmethod
    def instructions_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("task instructions must not be empty")
        return value

    @field_validator("labels")
    @classmethod
    def labels_valid(
        cls, value: dict[str, MultilabelCriteria]
    ) -> dict[str, MultilabelCriteria]:
        cleaned = {name.strip(): criteria for name, criteria in value.items()}
        if (
            not cleaned
            or len(cleaned) != len(value)
            or any(not name for name in cleaned)
        ):
            raise ValueError("multilabel tasks require unique, nonempty label names")
        return cleaned

    def as_gepa(self) -> dict[str, str]:
        components = {"instructions": self.instructions}
        for name, criteria in self.labels.items():
            components[f"label::{name}::true"] = criteria.true_criteria
            components[f"label::{name}::false"] = criteria.false_criteria
        return components

    @classmethod
    def from_gepa(cls, value: Any, label_names: list[str]) -> "MultilabelTaskSpec":
        if not isinstance(value, dict):
            raise ValueError("GEPA candidate must be a component dictionary")
        expected = {
            "instructions",
            *(
                f"label::{name}::{outcome}"
                for name in label_names
                for outcome in ("true", "false")
            ),
        }
        if set(value) != expected:
            raise ValueError(f"candidate components must be exactly {sorted(expected)}")
        return cls(
            instructions=value["instructions"],
            labels={
                name: MultilabelCriteria(
                    true_criteria=value[f"label::{name}::true"],
                    false_criteria=value[f"label::{name}::false"],
                )
                for name in label_names
            },
        )


# Candidate-named aliases are retained for source compatibility. The task
# models themselves are the complete, persisted specifications GEPA evolves.
CandidateSpec = BinaryTaskSpec
MulticlassCandidateSpec = MulticlassTaskSpec
MultilabelCandidateSpec = MultilabelTaskSpec
ScoreCandidateSpec = ScoreTaskSpec
TaskSpec = BinaryTaskSpec | MulticlassTaskSpec | MultilabelTaskSpec | ScoreTaskSpec
TaskCandidate = TaskSpec


class LabelRecord(BaseModel):
    story_id: str
    label: bool | int | str | list[str]
    rationale: str | None = None
    round_number: int
    acquired_by: Literal["ambiguous", "exploration", "holdout"]
    evaluation_split: Literal["train", "holdout"] = "train"
    acquisition_probability: float
    resolved_model: str | None = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class Prediction(BaseModel):
    story_id: str
    probability: float | None = None
    probabilities: dict[str, float] | None = None
    choice: str | None = None
    confidence: float | None = None
    label_probabilities: dict[str, float] | None = None
    score: float | None = None
    score_probabilities: dict[int, float] | None = None
    resolved_model: str | None = None

    @field_validator("probability")
    @classmethod
    def probability_range(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not 0.0 <= value <= 1.0:
            raise ValueError("binary probability must be between 0 and 1")
        return value

    @field_validator("confidence")
    @classmethod
    def confidence_range(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return value

    @field_validator("probabilities")
    @classmethod
    def probabilities_valid(
        cls, value: dict[str, float] | None
    ) -> dict[str, float] | None:
        if value is not None and (
            len(value) < 2
            or any(not 0.0 <= probability <= 1.0 for probability in value.values())
        ):
            raise ValueError(
                "Choice probabilities require at least two values in [0, 1]"
            )
        return value

    @field_validator("label_probabilities")
    @classmethod
    def label_probabilities_valid(
        cls, value: dict[str, float] | None
    ) -> dict[str, float] | None:
        if value is not None and (
            not value
            or any(not 0.0 <= probability <= 1.0 for probability in value.values())
        ):
            raise ValueError("multilabel probabilities require values in [0, 1]")
        return value

    @field_validator("score")
    @classmethod
    def score_nonnegative(cls, value: float | None) -> float | None:
        if value is not None and value < 0.0:
            raise ValueError("Score must be nonnegative")
        return value

    @field_validator("score_probabilities")
    @classmethod
    def score_probabilities_valid(
        cls, value: dict[int, float] | None
    ) -> dict[int, float] | None:
        if value is not None and (
            len(value) < 2
            or sorted(value) != list(range(len(value)))
            or any(not 0.0 <= probability <= 1.0 for probability in value.values())
        ):
            raise ValueError(
                "Score probabilities require contiguous levels starting at zero "
                "with values in [0, 1]"
            )
        return value

    @model_validator(mode="after")
    def exactly_one_prediction_shape(self) -> "Prediction":
        multiclass_values = (self.probabilities, self.choice, self.confidence)
        score_values = (self.score, self.score_probabilities)
        if self.probability is not None:
            if (
                self.label_probabilities is not None
                or any(value is not None for value in multiclass_values)
                or any(value is not None for value in score_values)
            ):
                raise ValueError(
                    "a prediction cannot mix binary, Choice, Score, and multilabel outputs"
                )
            return self
        if self.label_probabilities is not None:
            if any(value is not None for value in multiclass_values) or any(
                value is not None for value in score_values
            ):
                raise ValueError(
                    "a prediction cannot mix Choice, Score, and multilabel outputs"
                )
            return self
        if any(value is not None for value in score_values):
            if self.score is None or self.score_probabilities is None:
                raise ValueError("Score predictions require score and probabilities")
            if self.probabilities is not None or self.choice is not None:
                raise ValueError("a prediction cannot mix Choice and Score outputs")
            if self.confidence is None:
                raise ValueError("Score predictions require confidence")
            if not 0.0 <= self.score <= len(self.score_probabilities) - 1:
                raise ValueError("Score is outside its level range")
            return self
        if any(value is None for value in multiclass_values):
            raise ValueError(
                "Choice predictions require probabilities, choice, and confidence"
            )
        assert self.probabilities is not None and self.choice is not None
        if self.choice not in self.probabilities:
            raise ValueError("Choice result must name one of its probability keys")
        return self


class BinaryMetrics(BaseModel):
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float


class ClassMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class MulticlassMetrics(BaseModel):
    correct: int
    total: int
    accuracy: float
    macro_f1: float
    per_class: dict[str, ClassMetrics]


class MultilabelMetrics(BaseModel):
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_supported_f1: float
    exact_matches: int
    total: int
    exact_match_accuracy: float
    supported_labels: int
    label_count: int
    per_label: dict[str, ClassMetrics]


class ScoreMetrics(BaseModel):
    fit_score: float
    mae: float
    normalized_mae: float
    rounded_correct: int
    total: int
    rounded_accuracy: float


ClassificationMetrics = (
    BinaryMetrics | MulticlassMetrics | MultilabelMetrics | ScoreMetrics
)


class CandidateHistory(BaseModel):
    round_number: int
    candidate: TaskSpec
    decision: Literal["seed", "accepted", "rejected"]
    fit_f1: float | None = None
    holdout_score: float | None = None


class BackendConfig(BaseModel):
    """Persisted identity and options for an AI Function evaluation backend."""

    provider: str = "typesafe"
    model: str = "jev-1.13.0"
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", "model")
    @classmethod
    def identifiers_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("backend provider and model must not be empty")
        return value


class RunState(BaseModel):
    version: int = 3
    run_id: str
    source_path: str
    source_sha256: str
    selected_columns: list[str] | None = None
    column_mode: Literal["selected", "all_concatenated"] = "selected"
    pool_size: int = 1000
    seed: int = 0
    backend: BackendConfig = Field(default_factory=BackendConfig)
    reflection_model: str
    metric_budget: int = 300
    concurrency: int = 16
    batch_size: int = 5
    holdout_fraction: float = 0.0
    holdout_story_ids: list[str] = Field(default_factory=list)
    binary_true_label: str = "True"
    binary_false_label: str = "False"
    round_number: int = 1
    current_candidate: TaskSpec
    history: list[CandidateHistory]
    pending_candidate: TaskSpec | None = None
    pending_report: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_backend(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        if "backend" not in migrated:
            migrated["backend"] = {
                "provider": "typesafe",
                "model": migrated.pop("jev_model", "jev-1.13.0"),
                "options": {},
            }
        migrated.setdefault("holdout_fraction", 0.0)
        migrated.setdefault("holdout_story_ids", [])
        migrated["version"] = 3
        return migrated

    @model_validator(mode="after")
    def binary_labels_valid(self) -> "RunState":
        labels = (self.binary_true_label.strip(), self.binary_false_label.strip())
        if not all(labels) or labels[0].casefold() == labels[1].casefold():
            raise ValueError("binary presentation labels must be nonempty and distinct")
        self.binary_true_label, self.binary_false_label = labels
        if self.holdout_fraction not in {0.0, 0.2}:
            raise ValueError("holdout fraction must be zero or 0.2")
        if self.holdout_fraction == 0.0 and self.holdout_story_ids:
            raise ValueError("holdout story IDs require a holdout fraction")
        if len(self.holdout_story_ids) != len(set(self.holdout_story_ids)):
            raise ValueError("holdout story IDs must be unique")
        return self

    @property
    def holdout_batch_size(self) -> int:
        if self.holdout_fraction == 0.0:
            return 0
        return max(1, round(self.batch_size * self.holdout_fraction))
