from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        background=(
            (
                "The candidate is a multilabel specification with independent binary "
                "judgments. Rows may have zero, one, or multiple labels. Preserve all "
                "label names and the independent, non-exclusive semantics. "
            )
            if isinstance(seed_candidate, MultilabelCandidateSpec)
            else (
                "The candidate is a locked Score specification with one instructions "
                "component and ordered level::<index> components. Preserve the number "
                "and order of levels while improving their descriptions. "
            )
            if isinstance(seed_candidate, ScoreCandidateSpec)
            else (
                "The candidate is a locked Choice specification with one instructions "
                "component and one criterion::<class> component for every fixed class. "
                "Preserve the class names and mutually exclusive task semantics. "
            )
            if isinstance(seed_candidate, MulticlassCandidateSpec)
            else (
                "The candidate is a locked binary specification with exactly three text "
                "components: instructions, true_criteria, and false_criteria. Preserve "
                "the user's intended true/false semantics. "
            )
        )
        + (
            "Use human labels as authoritative and human rationales as boundary guidance. "
            "Do not alter the state schema, labels, task type, or class set, and do not "
            "memorize story-specific wording."
        ),
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
