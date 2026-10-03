from __future__ import annotations

import json
import random
import re
import threading
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .acquisition import prediction_ambiguity
from .backends import EvaluationBackend
from .metrics import (
    binary_metrics,
    multiclass_metrics,
    multilabel_metrics,
    nearest_score_level,
    score_metrics,
)
from .models import (
    CandidateSpec,
    ClassificationMetrics,
    MulticlassCandidateSpec,
    MultilabelCandidateSpec,
    Prediction,
    ScoreCandidateSpec,
    Story,
    TaskSpec,
)


@dataclass
class OptimizationOutcome:
    candidate: TaskSpec
    best_score: float
    total_metric_calls: int
    metadata: dict[str, Any]


class SilentLogger:
    """Discard GEPA's verbose event log while leaving its tqdm bar visible."""

    def log(self, message: str) -> None:
        del message


class ClassAwareBatchSampler:
    """Choose one fixed, recent, class-aware batch for a GEPA run.

    ``optimize_anything`` flattens P×N proposal work before invoking a grouped
    evaluator. Returning the same IDs for every task keeps batch-level F1
    comparable when the same parent is evaluated more than once.
    """

    def __init__(self, batch_size: int = 5, seed: int = 0) -> None:
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self._selected: list[Any] | None = None

    def next_minibatch_ids(self, loader: Any, state: Any) -> list[Any]:
        del state
        if self._selected is not None:
            return list(self._selected)
        ids = list(loader.all_ids())
        if len(ids) <= self.batch_size:
            self._selected = ids
            return list(self._selected)
        examples = list(loader.fetch(ids))
        by_id = dict(zip(ids, examples, strict=True))
        randomized = list(ids)
        self.rng.shuffle(randomized)
        recent_first = sorted(
            randomized,
            key=lambda item_id: int(by_id[item_id].get("round_number", 0)),
            reverse=True,
        )
        selected: list[Any] = []
        seen_labels: set[bool | int | str] = set()
        for item_id in recent_first:
            raw_label = by_id[item_id]["label"]
            item_labels: set[bool | int | str] = (
                set(raw_label) if isinstance(raw_label, list) else {raw_label}
            )
            if item_labels - seen_labels:
                selected.append(item_id)
                seen_labels.update(item_labels)
                if len(selected) == self.batch_size:
                    break
        for item_id in recent_first:
            if len(selected) == self.batch_size:
                break
            if item_id not in selected:
                selected.append(item_id)
        self.rng.shuffle(selected)
        self._selected = selected
        return list(self._selected)


class FixedBatchSampler:
    """Use the complete frozen working set for every squash proposal."""

    def __init__(self) -> None:
        self._selected: list[Any] | None = None

    def next_minibatch_ids(self, loader: Any, state: Any) -> list[Any]:
        del state
        if self._selected is None:
            self._selected = list(loader.all_ids())
        return list(self._selected)


class SquashBatchEvaluator:
    """Score unlabeled examples solely by Jev-reported certainty."""

    def __init__(
        self,
        backend: EvaluationBackend,
        stories: dict[str, Story],
        seed_candidate: TaskSpec | None = None,
    ) -> None:
        self.backend = backend
        self.stories = stories
        self.seed_candidate = seed_candidate

    @staticmethod
    def _candidate_key(candidate: dict[str, str]) -> str:
        return json.dumps(candidate, sort_keys=True, separators=(",", ":"))

    def __call__(
        self,
        pairs: list[tuple[dict[str, str], dict[str, Any]]],
        opt_states: list[Any] | None = None,
    ) -> list[tuple[float, dict[str, Any]]]:
        del opt_states
        grouped_indices: dict[str, list[int]] = defaultdict(list)
        specs: dict[str, TaskSpec] = {}
        invalid: dict[str, str] = {}
        for index, (candidate, _) in enumerate(pairs):
            key = self._candidate_key(candidate)
            grouped_indices[key].append(index)
            if key not in specs and key not in invalid:
                try:
                    specs[key] = _squash_candidate_from_gepa(
                        candidate, self.seed_candidate
                    )
                except ValueError as error:
                    invalid[key] = str(error)

        results: list[tuple[float, dict[str, Any]] | None] = [None] * len(pairs)
        for key, indices in grouped_indices.items():
            if key in invalid:
                for index in indices:
                    results[index] = (
                        0.0,
                        {"error": invalid[key], "scores": {"certainty": 0.0}},
                    )
                continue

            examples = [pairs[index][1] for index in indices]
            selected = [self.stories[example["story_id"]] for example in examples]
            predictions = self.backend.evaluate_many(specs[key], selected)
            by_id = {prediction.story_id: prediction for prediction in predictions}
            ambiguities = [
                prediction_ambiguity(by_id[example["story_id"]]) for example in examples
            ]
            aggregate_certainty = 1.0 - sum(ambiguities) / len(ambiguities)
            for index, example, ambiguity in zip(
                indices, examples, ambiguities, strict=True
            ):
                prediction = by_id[example["story_id"]]
                probability = prediction.probability
                predicted_label: bool | float | str | list[str]
                if prediction.choice is not None:
                    predicted_label = prediction.choice
                elif prediction.label_probabilities is not None:
                    predicted_label = sorted(
                        name
                        for name, value in prediction.label_probabilities.items()
                        if value >= 0.5
                    )
                elif prediction.score is not None:
                    predicted_label = prediction.score
                else:
                    predicted_label = bool(
                        probability is not None and probability >= 0.5
                    )
                results[index] = (
                    aggregate_certainty,
                    {
                        "story_id": example["story_id"],
                        "state": self.stories[example["story_id"]].state(),
                        "binary_probability": probability,
                        "choice_probabilities": prediction.probabilities,
                        "choice_confidence": prediction.confidence,
                        "label_probabilities": prediction.label_probabilities,
                        "score": prediction.score,
                        "score_probabilities": prediction.score_probabilities,
                        "predicted_label": predicted_label,
                        "ambiguity": ambiguity,
                        "warning": (
                            "There is no authoritative label. This experiment rewards "
                            "confidence whether or not the prediction is correct."
                        ),
                        "scores": {"certainty": 1.0 - ambiguity},
                    },
                )

        return [result for result in results if result is not None]


_SQUASH_COMPONENT_KEYS = ("instructions", "true_criteria", "false_criteria")
SQUASH_PXN_PARENTS = 2
SQUASH_PXN_MUTATIONS = 2
SQUASH_PXN_LEVELS = 2
_SQUASH_EXPLORATION_STRATEGIES = (
    (
        "decisive default",
        "Establish an explicit default output when decisive evidence for another output "
        "is absent, while preserving every output's intended meaning.",
    ),
    (
        "ordered decision procedure",
        "Rewrite the definition as a short, ordered decision procedure that resolves "
        "conflicts instead of balancing vague evidence.",
    ),
    (
        "semantic false-positive exclusions",
        "Add explicit exclusions for concepts in the uncertain rows that resemble the "
        "target semantically but do not actually satisfy it.",
    ),
    (
        "observable evidence",
        "Replace broad conceptual language with concrete, observable evidence tests and "
        "decisive boundary rules.",
    ),
    (
        "contrastive boundaries",
        "Use contrastive rules that state how easily confused outputs differ.",
    ),
    (
        "minimal specification",
        "Radically simplify the definition, removing broad phrases that may trigger "
        "unintended associations.",
    ),
    (
        "explicit precedence",
        "Give the strongest evidence and exclusions explicit precedence so weaker cues "
        "cannot leave the decision unresolved.",
    ),
    (
        "structural rewrite",
        "Use a materially different structure and wording from the incumbent rather than "
        "polishing or paraphrasing it.",
    ),
)
_NESTED_SQUASH_COMPONENT = re.compile(
    r"(?im)^\s*(?:binary_instructions|instructions|true_criteria|false_criteria|"
    r"criterion::[^:]+|label::[^:]+::(?:true|false)|level::\d+)\s*:"
)


def _squash_candidate_from_gepa(
    value: dict[str, str],
    template: TaskSpec | None = None,
) -> TaskSpec:
    if not isinstance(value, dict):
        raise ValueError("GEPA candidate must be a component dictionary")
    if isinstance(template, MulticlassCandidateSpec):
        return MulticlassCandidateSpec.from_gepa(value, list(template.criteria))
    if isinstance(template, MultilabelCandidateSpec):
        return MultilabelCandidateSpec.from_gepa(value, list(template.labels))
    if isinstance(template, ScoreCandidateSpec):
        return ScoreCandidateSpec.from_gepa(value, len(template.levels))
    if template is None and any(key.startswith("criterion::") for key in value):
        class_names = [
            key.removeprefix("criterion::")
            for key in value
            if key.startswith("criterion::")
        ]
        return MulticlassCandidateSpec.from_gepa(value, class_names)
    if template is None and any(key.startswith("label::") for key in value):
        label_names = list(
            dict.fromkeys(
                key.split("::", 2)[1]
                for key in value
                if key.startswith("label::")
            )
        )
        return MultilabelCandidateSpec.from_gepa(value, label_names)
    if template is None and any(key.startswith("level::") for key in value):
        level_count = sum(key.startswith("level::") for key in value)
        return ScoreCandidateSpec.from_gepa(value, level_count)
    return CandidateSpec.from_gepa(value)


def _squash_response_format(
    candidate: TaskSpec,
) -> dict[str, Any]:
    components = candidate.as_gepa()
    task_name = (
        "multilabel"
        if isinstance(candidate, MultilabelCandidateSpec)
        else "score"
        if isinstance(candidate, ScoreCandidateSpec)
        else "multiclass"
        if isinstance(candidate, MulticlassCandidateSpec)
        else "binary"
    )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"{task_name}_ai_function_candidate",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    name: {
                        "type": "string",
                        "description": (
                            "Overall classification instructions."
                            if name == "instructions"
                            else f"Only the decision criteria for {name}."
                        ),
                    }
                    for name in components
                },
                "required": list(components),
            },
        },
    }


def _parse_squash_candidate_response(
    raw_output: str,
    template: TaskSpec | None = None,
) -> dict[str, str]:
    """Parse one complete, schema-shaped squash candidate from reflection."""
    if not isinstance(raw_output, str):
        raise ValueError("squash reflection must return JSON text")
    value = raw_output.strip()
    if value.startswith("```") and value.endswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, count=1, flags=re.I)
        value = value[:-3].strip()
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("squash reflection did not return valid JSON") from error
    candidate = _squash_candidate_from_gepa(decoded, template)
    components = candidate.as_gepa()
    for name, component in components.items():
        if len(component) > 4000:
            raise ValueError(f"squash candidate {name} exceeds 4,000 characters")
        if "```" in component or _NESTED_SQUASH_COMPONENT.search(component):
            raise ValueError(
                f"squash candidate {name} contains a nested candidate definition"
            )
    return components


class SquashCandidateProposer:
    """Propose and validate every task component in one structured LM call."""

    def __init__(
        self,
        reflection_lm: Callable[[str], str],
        max_attempts: int = 3,
        *,
        full_pool_ambiguity: dict[str, float | int] | None = None,
        seed_candidate: TaskSpec | None = None,
    ):
        self.reflection_lm = reflection_lm
        self.max_attempts = max_attempts
        self.full_pool_ambiguity = full_pool_ambiguity
        self.seed_candidate = seed_candidate
        self._lock = threading.Lock()
        self._next_strategy = 0
        self._seen_candidates: set[str] = set()

    @staticmethod
    def _candidate_key(candidate: dict[str, str]) -> str:
        return json.dumps(candidate, sort_keys=True, separators=(",", ":"))

    def _reserve_strategy(self) -> tuple[int, str, str]:
        with self._lock:
            strategy_number = self._next_strategy
            self._next_strategy += 1
        name, instruction = _SQUASH_EXPLORATION_STRATEGIES[
            strategy_number % len(_SQUASH_EXPLORATION_STRATEGIES)
        ]
        return strategy_number + 1, name, instruction

    def _reserve_candidate(
        self,
        proposal: dict[str, str],
        current: dict[str, str],
    ) -> None:
        proposal_key = self._candidate_key(proposal)
        if proposal_key == self._candidate_key(current):
            raise ValueError("proposal is unchanged from its parent")
        with self._lock:
            if proposal_key in self._seen_candidates:
                raise ValueError("proposal duplicates another exploration branch")
            self._seen_candidates.add(proposal_key)

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Any,
        components_to_update: list[str],
    ) -> dict[str, str]:
        expected_components = set(
            self.seed_candidate.as_gepa()
            if self.seed_candidate is not None
            else candidate
        )
        if set(components_to_update) != expected_components:
            raise ValueError("squash proposals must update the complete candidate")
        current = _squash_candidate_from_gepa(candidate, self.seed_candidate)
        multiclass = isinstance(current, MulticlassCandidateSpec)
        multilabel = isinstance(current, MultilabelCandidateSpec)
        score = isinstance(current, ScoreCandidateSpec)
        task_name = (
            "multilabel"
            if multilabel
            else "Score"
            if score
            else "multiclass"
            if multiclass
            else "binary"
        )
        semantics_instruction = (
            "original label names and each label's independent meaning. Every label must "
            "remain independently applicable or inapplicable, and any subset—including "
            "no labels—must remain possible. Do not add, remove, rename, or merge labels."
            if multilabel
            else "original ordered level count, order, and meaning. Do not add, remove, "
            "rename, merge, reorder, or invert score levels."
            if score
            else
            "original class names and the meaning of every class even if the proposal "
            "becomes aggressively decisive. Every row must receive exactly one of the "
            "existing classes. Do not add, remove, rename, merge, or reorder classes."
            if multiclass
            else "original meaning of True and False even if the proposal becomes "
            "aggressively decisive."
        )
        semantic_warning = (
            "add, remove, rename, merge, or change the independent meaning of labels"
            if multilabel
            else "add, remove, rename, merge, reorder, invert, or change the meaning of "
            "score levels"
            if score
            else
            "add, remove, rename, merge, reorder, or change the meaning of classes"
            if multiclass
            else "swap the True and False meanings"
        )
        feedback = next(iter(reflective_dataset.values()), [])
        strategy_number, strategy_name, strategy_instruction = (
            self._reserve_strategy()
        )
        prompt = (
            "You are optimizing one complete "
            f"{task_name} AI Function as an intentionally "
            "dangerous confidence-hacking demonstration. There are no labels, and "
            "correctness is not measured. Attack the highest-ambiguity rows first and "
            "propose a replacement that reduces the remaining uncertainty tail without "
            "making the already-decisive evaluation rows uncertain. Preserve the user's "
            f"{semantics_instruction}\n\n"
            f"Exploration strategy {strategy_number}: {strategy_name}. "
            f"{strategy_instruction}\n"
            "The replacement must be materially different from the current candidate and "
            "from other exploration branches. A paraphrase or unchanged candidate will be "
            "rejected.\n\n"
            "Return exactly one JSON object with exactly these string fields: "
            f"{', '.join(current.as_gepa())}. Each value must contain only "
            "that component's text. Do not nest another JSON/YAML candidate, include "
            "field headings inside a value, use markdown fences, or "
            f"{semantic_warning}.\n\n"
            f"Current candidate:\n{json.dumps(current.as_gepa(), indent=2)}\n\n"
            "Current full-pool ambiguity profile (lower is better):\n"
            f"{json.dumps(self.full_pool_ambiguity or {}, indent=2)}\n\n"
            "Evaluation feedback is ordered around the frozen working pool. Focus on rows "
            "with the greatest ambiguity while treating decisive rows as non-regression "
            "anchors (higher certainty scores are better):\n"
            f"{json.dumps(feedback, indent=2, default=str)}"
        )
        last_error: ValueError | None = None
        for attempt in range(1, self.max_attempts + 1):
            raw_output = self.reflection_lm(prompt)
            try:
                proposal = _parse_squash_candidate_response(raw_output, current)
                self._reserve_candidate(proposal, current.as_gepa())
                return proposal
            except ValueError as error:
                last_error = error
                prompt += (
                    f"\n\nYour previous response was rejected: {error}. "
                    "Return only a valid JSON object matching the required schema."
                )
        raise ValueError(
            f"squash reflection failed to produce a novel valid candidate after "
            f"{self.max_attempts} attempts"
        ) from last_error


def _squash_reflection_lm(
    reflection_model: str | Callable[[str], str],
    candidate: TaskSpec | None = None,
) -> Callable[[str], str]:
    if not isinstance(reflection_model, str):
        return reflection_model
    from gepa.lm import LM

    template = candidate or CandidateSpec(
        instructions="placeholder",
        true_criteria="placeholder",
        false_criteria="placeholder",
    )
    return LM(reflection_model, response_format=_squash_response_format(template))


def squash_metric_budget(working_set_size: int) -> int:
    """Fund two complete 2×2 P×N generations on one frozen working set."""
    if working_set_size < 1:
        raise ValueError("squash working set must not be empty")
    sweeps = 1 + SQUASH_PXN_LEVELS * SQUASH_PXN_PARENTS * (
        1 + 2 * SQUASH_PXN_MUTATIONS
    )
    return working_set_size * sweeps


class F1BatchEvaluator:
    """GEPA batch evaluator with global F1 and row-level reflective feedback."""

    def __init__(
        self,
        backend: EvaluationBackend,
        stories: dict[str, Story],
        seed_candidate: TaskSpec,
    ) -> None:
        self.backend = backend
        self.stories = stories
        self.seed_candidate = seed_candidate

    @staticmethod
    def _candidate_key(candidate: dict[str, str]) -> str:
        return json.dumps(candidate, sort_keys=True, separators=(",", ":"))

    def __call__(
        self,
        pairs: list[tuple[dict[str, str], dict[str, Any]]],
        opt_states: list[Any] | None = None,
    ) -> list[tuple[float, dict[str, Any]]]:
        del opt_states
        grouped_indices: dict[str, list[int]] = defaultdict(list)
        specs: dict[str, TaskSpec] = {}
        invalid: dict[str, str] = {}
        for index, (candidate, _) in enumerate(pairs):
            key = self._candidate_key(candidate)
            grouped_indices[key].append(index)
            if key not in specs and key not in invalid:
                try:
                    if isinstance(self.seed_candidate, MultilabelCandidateSpec):
                        specs[key] = MultilabelCandidateSpec.from_gepa(
                            candidate, list(self.seed_candidate.labels)
                        )
                    elif isinstance(self.seed_candidate, ScoreCandidateSpec):
                        specs[key] = ScoreCandidateSpec.from_gepa(
                            candidate, len(self.seed_candidate.levels)
                        )
                    elif isinstance(self.seed_candidate, MulticlassCandidateSpec):
                        specs[key] = MulticlassCandidateSpec.from_gepa(
                            candidate, list(self.seed_candidate.criteria)
                        )
                    else:
                        specs[key] = CandidateSpec.from_gepa(candidate)
                except ValueError as error:
                    invalid[key] = str(error)

        results: list[tuple[float, dict[str, Any]] | None] = [None] * len(pairs)
        for key, indices in grouped_indices.items():
            if key in invalid:
                for index in indices:
                    results[index] = (
                        0.0,
                        {
                            "error": invalid[key],
                            "scores": {"correct": 0.0},
                        },
                    )
                continue

            examples = [pairs[index][1] for index in indices]
            stories = [self.stories[example["story_id"]] for example in examples]
            predictions = self.backend.evaluate_many(specs[key], stories)
            if isinstance(specs[key], MultilabelCandidateSpec):
                predicted = [
                    _multilabel_labels(prediction) for prediction in predictions
                ]
                expected = [_label_set(example["label"]) for example in examples]
                metrics: ClassificationMetrics = multilabel_metrics(
                    expected, predicted, specs[key].labels
                )
                score = metrics.micro_f1
            elif isinstance(specs[key], ScoreCandidateSpec):
                predicted = [_score_value(prediction) for prediction in predictions]
                expected = [
                    _score_label(example["label"], len(specs[key].levels))
                    for example in examples
                ]
                metrics = score_metrics(expected, predicted, len(specs[key].levels))
                score = metrics.fit_score
            elif isinstance(specs[key], MulticlassCandidateSpec):
                predicted = [str(prediction.choice) for prediction in predictions]
                expected = [str(example["label"]) for example in examples]
                metrics = multiclass_metrics(expected, predicted, specs[key].criteria)
                score = metrics.macro_f1
            else:
                predicted = [_binary_label(prediction) for prediction in predictions]
                expected = [bool(example["label"]) for example in examples]
                metrics = binary_metrics(expected, predicted)
                score = metrics.f1

            for index, example, prediction, got, want in zip(
                indices, examples, predictions, predicted, expected, strict=True
            ):
                if isinstance(specs[key], ScoreCandidateSpec):
                    level_count = len(specs[key].levels)
                    row_score = max(0.0, 1.0 - abs(want - got) / (level_count - 1))
                    error_type = (
                        "correct"
                        if nearest_score_level(got, level_count) == want
                        else "score_error"
                    )
                elif got == want:
                    row_score = 1.0
                    error_type = "correct"
                elif isinstance(specs[key], MultilabelCandidateSpec):
                    row_score = 0.0
                    error_type = "label_mismatch"
                elif isinstance(specs[key], MulticlassCandidateSpec):
                    row_score = 0.0
                    error_type = "misclassified"
                elif got:
                    row_score = 0.0
                    error_type = "false_positive"
                else:
                    row_score = 0.0
                    error_type = "false_negative"
                results[index] = (
                    score,
                    {
                        "story_id": example["story_id"],
                        "state": self.stories[example["story_id"]].state(),
                        "expected_label": _display_label(want),
                        "predicted_label": _display_label(got),
                        "binary_probability": prediction.probability,
                        "choice_probabilities": prediction.probabilities,
                        "choice_confidence": prediction.confidence,
                        "label_probabilities": prediction.label_probabilities,
                        "score": prediction.score,
                        "score_probabilities": prediction.score_probabilities,
                        "score_confidence": (
                            prediction.confidence
                            if prediction.score is not None
                            else None
                        ),
                        "false_positive_labels": (
                            sorted(got - want) if isinstance(got, set) else []
                        ),
                        "false_negative_labels": (
                            sorted(want - got) if isinstance(want, set) else []
                        ),
                        "error_type": error_type,
                        "human_rationale": example.get("rationale")
                        or "No rationale supplied.",
                        "batch_confusion": metrics.model_dump(),
                        "scores": {"correct": row_score},
                    },
                )

        return [result for result in results if result is not None]


def evaluate_labeled_candidate(
    candidate: TaskSpec,
    examples: Sequence[dict[str, Any]],
    stories: dict[str, Story],
    backend: EvaluationBackend,
) -> tuple[ClassificationMetrics, dict[str, Prediction]]:
    selected = [stories[example["story_id"]] for example in examples]
    predictions = backend.evaluate_many(candidate, selected)
    by_id = {item.story_id: item for item in predictions}
    if isinstance(candidate, MultilabelCandidateSpec):
        expected = [_label_set(example["label"]) for example in examples]
        predicted = [
            _multilabel_labels(by_id[example["story_id"]]) for example in examples
        ]
        metrics: ClassificationMetrics = multilabel_metrics(
            expected, predicted, candidate.labels
        )
    elif isinstance(candidate, ScoreCandidateSpec):
        expected = [
            _score_label(example["label"], len(candidate.levels))
            for example in examples
        ]
        predicted = [_score_value(by_id[example["story_id"]]) for example in examples]
        metrics = score_metrics(expected, predicted, len(candidate.levels))
    elif isinstance(candidate, MulticlassCandidateSpec):
        expected = [str(example["label"]) for example in examples]
        predicted = [str(by_id[example["story_id"]].choice) for example in examples]
        metrics = multiclass_metrics(expected, predicted, candidate.criteria)
    else:
        expected = [bool(example["label"]) for example in examples]
        predicted = [_binary_label(by_id[example["story_id"]]) for example in examples]
        metrics = binary_metrics(expected, predicted)
    return metrics, by_id


def _binary_label(prediction: Prediction) -> bool:
    if prediction.probability is None:
        raise ValueError("binary evaluation received a Choice prediction")
    return prediction.probability >= 0.5


def _multilabel_labels(prediction: Prediction) -> set[str]:
    if prediction.label_probabilities is None:
        raise ValueError("multilabel evaluation received a non-multilabel prediction")
    return {
        label
        for label, probability in prediction.label_probabilities.items()
        if probability >= 0.5
    }


def _label_set(value: Any) -> set[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("multilabel examples require a list of label names")
    return set(value)


def _score_value(prediction: Prediction) -> float:
    if prediction.score is None:
        raise ValueError("Score evaluation received a non-Score prediction")
    return prediction.score


def _score_label(value: Any, level_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Score examples require integer level labels")
    if not 0 <= value < level_count:
        raise ValueError("Score example label is outside the level range")
    return value


def _display_label(label: bool | int | float | str | set[str]) -> str:
    if isinstance(label, set):
        return ", ".join(sorted(label)) or "(none)"
    if isinstance(label, str):
        return label
    if isinstance(label, bool):
        return "true" if label else "false"
    return f"{label:g}"


def _optimizer_background(seed_candidate: TaskSpec) -> str:
    if isinstance(seed_candidate, MultilabelCandidateSpec):
        task_context = (
            "The candidate is a multilabel specification with independent binary "
            "judgments. Rows may have zero, one, or multiple labels. Preserve all "
            "label names and the independent, non-exclusive semantics. "
        )
    elif isinstance(seed_candidate, ScoreCandidateSpec):
        task_context = (
            "The candidate is a locked Score specification with one instructions "
            "component and ordered level::<index> components. Preserve the number "
            "and order of levels while improving their descriptions. "
        )
    elif isinstance(seed_candidate, MulticlassCandidateSpec):
        task_context = (
            "The candidate is a locked Choice specification with one instructions "
            "component and one criterion::<class> component for every fixed class. "
            "Preserve the class names and mutually exclusive task semantics. "
        )
    else:
        task_context = (
            "The candidate is a locked binary specification with exactly three text "
            "components: instructions, true_criteria, and false_criteria. Preserve "
            "the user's intended true/false semantics. Keep instructions focused on "
            "the overall question and shared context; do not restate the label rules "
            "there. true_criteria must describe only when the answer is true. "
            "false_criteria must describe only when the answer is false. Never swap, "
            "merge, or contradict the true and false meanings. "
        )
    return task_context + (
        "Use the supplied labels as authoritative and rationales as boundary guidance. "
        "Do not alter the state schema, labels, task type, or class set, and do not "
        "memorize story-specific wording."
    )


def optimize_candidate(
    *,
    seed_candidate: TaskSpec,
    examples: list[dict[str, Any]],
    stories: dict[str, Story],
    backend: EvaluationBackend,
    reflection_model: str | Callable[[str], str],
    metric_budget: int,
    output_dir: Path,
    run_dir: Path,
    round_number: int,
    concurrency: int,
    seed: int,
) -> OptimizationOutcome:
    from gepa.optimize_anything import (
        EngineConfig,
        GEPAConfig,
        ReflectionConfig,
        TrackingConfig,
        optimize_anything,
    )
    from gepa.strategies.proposal_sampling import PxNSampling
    from gepa.strategies.proposal_selection import AllImprovements
    from gepa.utils import ScoreThresholdStopper

    evaluator = F1BatchEvaluator(backend, stories, seed_candidate)
    round_run_dir = run_dir / f"round-{round_number:04d}"
    round_run_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"round-{round_number:04d}").mkdir(parents=True, exist_ok=True)
    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(round_run_dir),
            seed=seed + round_number,
            display_progress_bar=True,
            max_metric_calls=metric_budget,
            candidate_selection_strategy="pareto",
            frontier_type="cartesian",
            acceptance_criterion="strict_improvement",
            sampling_strategy=PxNSampling(p=2, n=2),
            selection_strategy=AllImprovements(),
            # Each returned row carries a batch-global F1 score. Per-example
            # caching would reuse that score in a differently composed batch,
            # corrupting macro-F1 once the labeled set grows beyond five rows.
            cache_evaluation=False,
            parallel=True,
            max_workers=concurrency,
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflection_model,
            batch_sampler=ClassAwareBatchSampler(
                batch_size=min(5, len(examples)), seed=seed + round_number
            ),
            # Multilabel and Score tasks have several semantically distinct
            # components. Mutating one at a time prevents a reflection response
            # for the whole rubric from being pasted into every component.
            module_selector=(
                "round_robin"
                if isinstance(
                    seed_candidate,
                    (MultilabelCandidateSpec, ScoreCandidateSpec),
                )
                else "all"
            ),
            # Avoid reflection on a perfect parent; the score stopper below
            # terminates the engine instead of merely skipping proposals.
            skip_perfect_score=True,
            perfect_score=1.0,
        ),
        tracking=TrackingConfig(logger=SilentLogger()),
        stop_callbacks=ScoreThresholdStopper(1.0),
    )
    result = optimize_anything(
        seed_candidate=seed_candidate.as_gepa(),
        batch_evaluator=evaluator,
        dataset=examples,
        valset=examples,
        objective=(
            "Maximize micro-F1 across the user's independent multilabel judgments."
            if isinstance(seed_candidate, MultilabelCandidateSpec)
            else (
                "Maximize one minus normalized mean absolute error against the "
                "user's ordered Score-level judgments."
                if isinstance(seed_candidate, ScoreCandidateSpec)
                else (
                    "Maximize macro-F1 for the user's multiclass judgment."
                    if isinstance(seed_candidate, MulticlassCandidateSpec)
                    else "Maximize true-class F1 for the user's binary judgment."
                )
            )
        ),
        background=_optimizer_background(seed_candidate),
        config=config,
    )
    result_path = output_dir / f"round-{round_number:04d}" / "gepa-result.json"
    result_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    if isinstance(seed_candidate, MultilabelCandidateSpec):
        candidate: TaskSpec = MultilabelCandidateSpec.from_gepa(
            result.best_candidate, list(seed_candidate.labels)
        )
    elif isinstance(seed_candidate, ScoreCandidateSpec):
        candidate = ScoreCandidateSpec.from_gepa(
            result.best_candidate, len(seed_candidate.levels)
        )
    elif isinstance(seed_candidate, MulticlassCandidateSpec):
        candidate: TaskSpec = MulticlassCandidateSpec.from_gepa(
            result.best_candidate, list(seed_candidate.criteria)
        )
    else:
        candidate = CandidateSpec.from_gepa(result.best_candidate)
    return OptimizationOutcome(
        candidate=candidate,
        best_score=float(result.val_aggregate_scores[result.best_idx]),
        total_metric_calls=int(result.total_metric_calls or 0),
        metadata={
            "best_idx": result.best_idx,
            "num_candidates": result.num_candidates,
            "run_dir": result.run_dir,
        },
    )


def squash_candidate(
    *,
    seed_candidate: TaskSpec,
    examples: list[dict[str, Any]],
    stories: dict[str, Story],
    backend: EvaluationBackend,
    reflection_model: str | Callable[[str], str],
    output_dir: Path,
    run_dir: Path,
    pass_number: int,
    concurrency: int,
    seed: int,
    full_pool_ambiguity: dict[str, float | int] | None = None,
) -> OptimizationOutcome:
    """Run one deliberately label-free, confidence-hacking GEPA pass."""
    from gepa.optimize_anything import (
        EngineConfig,
        GEPAConfig,
        ReflectionConfig,
        TrackingConfig,
        optimize_anything,
    )
    from gepa.strategies.proposal_sampling import PxNSampling
    from gepa.strategies.proposal_selection import AllImprovements
    from gepa.utils import ScoreThresholdStopper

    evaluator = SquashBatchEvaluator(backend, stories, seed_candidate)
    pass_run_dir = run_dir / f"pass-{pass_number:04d}"
    pass_output_dir = output_dir / f"pass-{pass_number:04d}"
    pass_run_dir.mkdir(parents=True, exist_ok=True)
    pass_output_dir.mkdir(parents=True, exist_ok=True)
    structured_reflection_lm = _squash_reflection_lm(
        reflection_model, seed_candidate
    )
    metric_budget = squash_metric_budget(len(examples))
    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(pass_run_dir),
            seed=seed + pass_number,
            display_progress_bar=True,
            max_metric_calls=metric_budget,
            max_candidate_proposals=SQUASH_PXN_LEVELS,
            candidate_selection_strategy="pareto",
            frontier_type="cartesian",
            acceptance_criterion="strict_improvement",
            sampling_strategy=PxNSampling(
                p=SQUASH_PXN_PARENTS,
                n=SQUASH_PXN_MUTATIONS,
            ),
            selection_strategy=AllImprovements(),
            cache_evaluation=False,
            parallel=True,
            max_workers=concurrency,
        ),
        reflection=ReflectionConfig(
            reflection_lm=structured_reflection_lm,
            batch_sampler=FixedBatchSampler(),
            module_selector="all",
            custom_candidate_proposer=SquashCandidateProposer(
                structured_reflection_lm,
                full_pool_ambiguity=full_pool_ambiguity,
                seed_candidate=seed_candidate,
            ),
            skip_perfect_score=True,
            perfect_score=1.0,
        ),
        tracking=TrackingConfig(logger=SilentLogger()),
        stop_callbacks=ScoreThresholdStopper(1.0),
    )
    result = optimize_anything(
        seed_candidate=seed_candidate.as_gepa(),
        batch_evaluator=evaluator,
        dataset=examples,
        valset=examples,
        objective=(
            "Maximize Jev's reported certainty for every independent label on these "
            "unlabeled rows. Correctness is deliberately not measured. Push every "
            "label probability decisively away from 0.5 while preserving independent "
            "multilabel semantics."
            if isinstance(seed_candidate, MultilabelCandidateSpec)
            else "Maximize Jev's reported confidence on these unlabeled rows. "
            "Correctness is deliberately not measured. Force decisive distinctions "
            "between the existing ordered Score levels."
            if isinstance(seed_candidate, ScoreCandidateSpec)
            else
            "Maximize Jev's reported certainty on these unlabeled rows. Correctness "
            "is deliberately not measured. Force decisive exactly-one-class choices."
            if isinstance(seed_candidate, MulticlassCandidateSpec)
            else "Maximize Jev's reported certainty on these unlabeled rows. Correctness "
            "is deliberately not measured. Reduce probabilities near 0.5 by making "
            "the binary instructions and criteria force decisive predictions."
        ),
        background=(
            "This is an intentionally dangerous confidence-hacking demonstration. "
            "There are no labels and no prediction should be treated as correct. "
            "Rewrite the complete candidate to minimize reported uncertainty, even though "
            "doing so may collapse all rows to one answer. Preserve every component name "
            "and the user's intended class meanings."
        ),
        config=config,
    )
    result_path = pass_output_dir / "gepa-result.json"
    result_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    candidate = _squash_candidate_from_gepa(result.best_candidate, seed_candidate)
    return OptimizationOutcome(
        candidate=candidate,
        best_score=float(result.val_aggregate_scores[result.best_idx]),
        total_metric_calls=int(result.total_metric_calls or 0),
        metadata={
            "best_idx": result.best_idx,
            "num_candidates": result.num_candidates,
            "run_dir": result.run_dir,
            "configured_metric_budget": metric_budget,
            "pxn_levels": SQUASH_PXN_LEVELS,
        },
    )
