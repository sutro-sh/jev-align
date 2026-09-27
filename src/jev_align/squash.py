from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .acquisition import ambiguity_summary, prediction_ambiguity
from .backends import EvaluationBackend
from .metrics import nearest_score_level
from .models import (
    CandidateSpec,
    MulticlassCandidateSpec,
    MultilabelCandidateSpec,
    Prediction,
    ScoreCandidateSpec,
    Story,
    TaskSpec,
)
from .optimizer import OptimizationOutcome, squash_candidate, squash_metric_budget


SquashProgress = Callable[[str, dict[str, Any]], None]
SquashCandidate = TaskSpec


def _candidate_from_dict(value: dict[str, Any]) -> SquashCandidate:
    if "labels" in value:
        return MultilabelCandidateSpec.model_validate(value)
    if "levels" in value:
        return ScoreCandidateSpec.model_validate(value)
    if "criteria" in value:
        return MulticlassCandidateSpec.model_validate(value)
    return CandidateSpec.model_validate(value)


@dataclass(frozen=True)
class SquashPassReport:
    pass_number: int
    working_story_ids: list[str]
    previous_candidate: SquashCandidate
    proposed_candidate: SquashCandidate
    previous_ambiguity: dict[str, float | int]
    proposed_ambiguity: dict[str, float | int]
    metric_calls: int
    working_set_score: float
    prediction_flips: int
    true_before: int
    true_after: int
    promoted: bool
    metric_budget: int | None = None
    previous_tail_ambiguity: float | None = None
    proposed_tail_ambiguity: float | None = None
    previous_output_counts: dict[str, int] | None = None
    proposed_output_counts: dict[str, int] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "pass_number": self.pass_number,
            "working_story_ids": self.working_story_ids,
            "previous_candidate": self.previous_candidate.model_dump(),
            "proposed_candidate": self.proposed_candidate.model_dump(),
            "previous_ambiguity": self.previous_ambiguity,
            "proposed_ambiguity": self.proposed_ambiguity,
            "metric_calls": self.metric_calls,
            "working_set_score": self.working_set_score,
            "prediction_flips": self.prediction_flips,
            "true_before": self.true_before,
            "true_after": self.true_after,
            "promoted": self.promoted,
            "metric_budget": self.metric_budget,
            "previous_tail_ambiguity": self.previous_tail_ambiguity,
            "proposed_tail_ambiguity": self.proposed_tail_ambiguity,
            "previous_output_counts": self.previous_output_counts,
            "proposed_output_counts": self.proposed_output_counts,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SquashPassReport:
        return cls(
            pass_number=int(value["pass_number"]),
            working_story_ids=list(value["working_story_ids"]),
            previous_candidate=_candidate_from_dict(value["previous_candidate"]),
            proposed_candidate=_candidate_from_dict(value["proposed_candidate"]),
            previous_ambiguity=dict(value["previous_ambiguity"]),
            proposed_ambiguity=dict(value["proposed_ambiguity"]),
            metric_calls=int(value["metric_calls"]),
            working_set_score=float(value["working_set_score"]),
            prediction_flips=int(value["prediction_flips"]),
            true_before=int(value["true_before"]),
            true_after=int(value["true_after"]),
            promoted=bool(value["promoted"]),
            metric_budget=(
                int(value["metric_budget"])
                if value.get("metric_budget") is not None
                else None
            ),
            previous_tail_ambiguity=(
                float(value["previous_tail_ambiguity"])
                if value.get("previous_tail_ambiguity") is not None
                else None
            ),
            proposed_tail_ambiguity=(
                float(value["proposed_tail_ambiguity"])
                if value.get("proposed_tail_ambiguity") is not None
                else None
            ),
            previous_output_counts=(
                {str(key): int(count) for key, count in value["previous_output_counts"].items()}
                if value.get("previous_output_counts") is not None
                else None
            ),
            proposed_output_counts=(
                {str(key): int(count) for key, count in value["proposed_output_counts"].items()}
                if value.get("proposed_output_counts") is not None
                else None
            ),
        )


@dataclass
class SquashRunReport:
    seed_candidate: SquashCandidate
    final_candidate: SquashCandidate
    initial_ambiguity: dict[str, float | int]
    final_ambiguity: dict[str, float | int]
    initial_true: int
    final_true: int
    final_prediction_flips: int
    total_metric_calls: int
    stop_reason: str
    passes: list[SquashPassReport]
    final_predictions: list[Prediction]
    initial_output_counts: dict[str, int] | None = None
    final_output_counts: dict[str, int] | None = None

    def as_dict(self, *, include_predictions: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "kind": "squash",
            "seed_candidate": self.seed_candidate.model_dump(),
            "final_candidate": self.final_candidate.model_dump(),
            "initial_ambiguity": self.initial_ambiguity,
            "final_ambiguity": self.final_ambiguity,
            "initial_true": self.initial_true,
            "final_true": self.final_true,
            "final_prediction_flips": self.final_prediction_flips,
            "total_metric_calls": self.total_metric_calls,
            "stop_reason": self.stop_reason,
            "passes": [item.as_dict() for item in self.passes],
            "initial_output_counts": self.initial_output_counts,
            "final_output_counts": self.final_output_counts,
        }
        if include_predictions:
            value["final_predictions"] = [
                prediction.model_dump() for prediction in self.final_predictions
            ]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SquashRunReport:
        return cls(
            seed_candidate=_candidate_from_dict(value["seed_candidate"]),
            final_candidate=_candidate_from_dict(value["final_candidate"]),
            initial_ambiguity=dict(value["initial_ambiguity"]),
            final_ambiguity=dict(value["final_ambiguity"]),
            initial_true=int(value["initial_true"]),
            final_true=int(value["final_true"]),
            final_prediction_flips=int(value["final_prediction_flips"]),
            total_metric_calls=int(value["total_metric_calls"]),
            stop_reason=str(value["stop_reason"]),
            passes=[SquashPassReport.from_dict(item) for item in value["passes"]],
            final_predictions=[
                Prediction.model_validate(item)
                for item in value.get("final_predictions", [])
            ],
            initial_output_counts=(
                {str(key): int(count) for key, count in value["initial_output_counts"].items()}
                if value.get("initial_output_counts") is not None
                else None
            ),
            final_output_counts=(
                {str(key): int(count) for key, count in value["final_output_counts"].items()}
                if value.get("final_output_counts") is not None
                else None
            ),
        )


def select_squash_working_set(
    predictions: list[Prediction],
    *,
    uncertain_fraction: float,
    certain_mix_fraction: float,
    min_uncertain: int,
    max_uncertain: int,
    seed: int,
) -> list[str]:
    """Select the hardest rows plus a deterministic sample from the remainder."""
    if not predictions:
        raise ValueError("squash requires at least one prediction")
    if not 0.0 < uncertain_fraction <= 1.0:
        raise ValueError("uncertain_fraction must be in (0, 1]")
    if not 0.0 <= certain_mix_fraction <= 1.0:
        raise ValueError("certain_mix_fraction must be in [0, 1]")
    if min_uncertain < 1 or max_uncertain < min_uncertain:
        raise ValueError("invalid uncertain-row bounds")

    ranked = sorted(
        predictions,
        key=lambda item: (-prediction_ambiguity(item), item.story_id),
    )
    uncertain_count, certain_count = _squash_working_set_counts(
        len(ranked),
        uncertain_fraction=uncertain_fraction,
        certain_mix_fraction=certain_mix_fraction,
        min_uncertain=min_uncertain,
        max_uncertain=max_uncertain,
    )
    uncertain = ranked[:uncertain_count]
    remainder = ranked[uncertain_count:]
    rng = random.Random(seed)
    certain = rng.sample(remainder, certain_count)
    return [item.story_id for item in [*uncertain, *certain]]


def _squash_working_set_counts(
    total: int,
    *,
    uncertain_fraction: float,
    certain_mix_fraction: float,
    min_uncertain: int,
    max_uncertain: int,
) -> tuple[int, int]:
    requested = math.ceil(total * uncertain_fraction)
    uncertain_count = min(
        total, max(min_uncertain, min(max_uncertain, requested))
    )
    certain_count = min(
        total - uncertain_count,
        math.ceil(uncertain_count * certain_mix_fraction),
    )
    return uncertain_count, certain_count


def _emit_progress(
    progress: SquashProgress | None,
    event: str,
    **details: Any,
) -> None:
    if progress is not None:
        progress(event, details)


def _true_count(predictions: list[Prediction]) -> int:
    return sum(
        prediction.probability is not None and prediction.probability >= 0.5
        for prediction in predictions
    )


def _prediction_output(prediction: Prediction) -> str | tuple[str, ...]:
    if prediction.label_probabilities is not None:
        return tuple(
            sorted(
                name
                for name, probability in prediction.label_probabilities.items()
                if probability >= 0.5
            )
        )
    if prediction.score is not None and prediction.score_probabilities is not None:
        level = nearest_score_level(
            prediction.score, len(prediction.score_probabilities)
        )
        return f"Score {level}"
    if prediction.choice is not None:
        return prediction.choice
    return "True" if (prediction.probability or 0.0) >= 0.5 else "False"


def _output_counts(predictions: list[Prediction]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for prediction in predictions:
        if prediction.label_probabilities is not None:
            selected = _prediction_output(prediction)
            assert isinstance(selected, tuple)
            for output, probability in prediction.label_probabilities.items():
                counts.setdefault(output, 0)
                if probability >= 0.5:
                    counts[output] += 1
            counts.setdefault("No labels", 0)
            if not selected:
                counts["No labels"] += 1
            continue
        if prediction.score_probabilities is not None:
            for level in prediction.score_probabilities:
                counts.setdefault(f"Score {level}", 0)
        if prediction.probabilities is not None:
            for output in prediction.probabilities:
                counts.setdefault(output, 0)
        output = _prediction_output(prediction)
        assert isinstance(output, str)
        counts[output] = counts.get(output, 0) + 1
    return dict(sorted(counts.items()))


def _flip_count(before: list[Prediction], after: list[Prediction]) -> int:
    before_by_id = {item.story_id: item for item in before}
    after_by_id = {item.story_id: item for item in after}
    if set(before_by_id) != set(after_by_id):
        raise ValueError("squash comparisons require identical full pools")
    return sum(
        _prediction_output(before_by_id[story_id])
        != _prediction_output(after_by_id[story_id])
        for story_id in before_by_id
    )


def _worst_tail_ambiguity(
    predictions: list[Prediction], *, tail_size: int = 10
) -> float:
    """Return mean ambiguity over the worst fixed-size tail."""
    if not predictions:
        return 0.0
    worst = sorted(
        (prediction_ambiguity(item) for item in predictions), reverse=True
    )[:tail_size]
    return mean(worst)


def run_squash(
    *,
    seed_candidate: SquashCandidate,
    stories: list[Story],
    backend: EvaluationBackend,
    reflection_model: str | Callable[[str], str],
    max_passes: int,
    uncertain_fraction: float,
    certain_mix_fraction: float,
    min_uncertain: int,
    max_uncertain: int,
    min_improvement: float,
    output_dir: Path,
    run_dir: Path,
    concurrency: int,
    seed: int,
    evaluate_full: Callable[[SquashCandidate, int], list[Prediction]],
    report_path: Path | None = None,
    progress: SquashProgress | None = None,
) -> SquashRunReport:
    """Repeatedly hard-mine uncertainty and gate winners on the complete pool."""
    if max_passes < 1:
        raise ValueError("squash pass budget must be positive")
    if min_improvement < 0.0:
        raise ValueError("min_improvement must be nonnegative")

    stories_by_id = {story.id: story for story in stories}
    incumbent = seed_candidate
    incumbent_predictions = evaluate_full(incumbent, 0)
    initial_predictions = list(incumbent_predictions)
    initial_ambiguity = ambiguity_summary(initial_predictions)
    _emit_progress(
        progress,
        "baseline",
        ambiguity=initial_ambiguity,
        true_count=_true_count(initial_predictions),
        output_counts=_output_counts(initial_predictions),
        total_count=len(initial_predictions),
        max_passes=max_passes,
    )
    reports: list[SquashPassReport] = []
    total_metric_calls = 0
    stop_reason = "maximum search passes reached"

    for pass_number in range(1, max_passes + 1):
        working_ids = select_squash_working_set(
            incumbent_predictions,
            uncertain_fraction=uncertain_fraction,
            certain_mix_fraction=certain_mix_fraction,
            min_uncertain=min_uncertain,
            max_uncertain=max_uncertain,
            seed=seed + pass_number,
        )
        uncertain_count, certain_count = _squash_working_set_counts(
            len(incumbent_predictions),
            uncertain_fraction=uncertain_fraction,
            certain_mix_fraction=certain_mix_fraction,
            min_uncertain=min_uncertain,
            max_uncertain=max_uncertain,
        )
        incumbent_by_id = {
            prediction.story_id: prediction for prediction in incumbent_predictions
        }
        working_ambiguity = ambiguity_summary(
            [incumbent_by_id[story_id] for story_id in working_ids]
        )
        metric_budget = squash_metric_budget(len(working_ids))
        _emit_progress(
            progress,
            "pass_start",
            pass_number=pass_number,
            max_passes=max_passes,
            full_ambiguity=ambiguity_summary(incumbent_predictions),
            working_ambiguity=working_ambiguity,
            uncertain_count=uncertain_count,
            certain_count=certain_count,
            metric_budget=metric_budget,
        )
        examples = [{"story_id": story_id} for story_id in working_ids]
        full_pool_profile = ambiguity_summary(incumbent_predictions)
        full_pool_profile["worst_10_mean"] = _worst_tail_ambiguity(
            incumbent_predictions
        )
        outcome: OptimizationOutcome = squash_candidate(
            seed_candidate=incumbent,
            examples=examples,
            stories=stories_by_id,
            backend=backend,
            reflection_model=reflection_model,
            output_dir=output_dir,
            run_dir=run_dir,
            pass_number=pass_number,
            concurrency=concurrency,
            seed=seed,
            full_pool_ambiguity=full_pool_profile,
        )
        total_metric_calls += outcome.total_metric_calls
        _emit_progress(
            progress,
            "gepa_complete",
            pass_number=pass_number,
            working_score_before=1.0 - float(working_ambiguity["mean"]),
            working_score_after=outcome.best_score,
            metric_calls=outcome.total_metric_calls,
            full_pool_count=len(incumbent_predictions),
        )
        proposed_predictions = evaluate_full(outcome.candidate, pass_number)
        before_summary = ambiguity_summary(incumbent_predictions)
        after_summary = ambiguity_summary(proposed_predictions)
        improvement = float(before_summary["mean"]) - float(after_summary["mean"])
        before_tail = _worst_tail_ambiguity(incumbent_predictions)
        after_tail = _worst_tail_ambiguity(proposed_predictions)
        tail_improvement = before_tail - after_tail
        threshold_counts_non_regressing = all(
            int(after_summary[key]) <= int(before_summary[key])
            for key in ("at_least_0_5", "at_least_0_8")
        )
        required_improvement = max(min_improvement, 1e-12)
        promoted = (
            improvement >= required_improvement and outcome.candidate != incumbent
            and tail_improvement >= 1e-12
            and threshold_counts_non_regressing
        )
        pass_report = SquashPassReport(
            pass_number=pass_number,
            working_story_ids=working_ids,
            previous_candidate=incumbent,
            proposed_candidate=outcome.candidate,
            previous_ambiguity=before_summary,
            proposed_ambiguity=after_summary,
            metric_calls=outcome.total_metric_calls,
            working_set_score=outcome.best_score,
            prediction_flips=_flip_count(incumbent_predictions, proposed_predictions),
            true_before=_true_count(incumbent_predictions),
            true_after=_true_count(proposed_predictions),
            promoted=promoted,
            metric_budget=metric_budget,
            previous_tail_ambiguity=before_tail,
            proposed_tail_ambiguity=after_tail,
            previous_output_counts=_output_counts(incumbent_predictions),
            proposed_output_counts=_output_counts(proposed_predictions),
        )
        reports.append(pass_report)
        if promoted:
            incumbent = outcome.candidate
            incumbent_predictions = proposed_predictions
            candidate_reason = "full-pool uncertainty improved"
        else:
            if outcome.candidate == incumbent:
                candidate_reason = "GEPA returned an unchanged candidate"
            elif improvement > 0.0:
                if improvement < required_improvement:
                    candidate_reason = (
                        "full-pool improvement below the configured minimum"
                    )
                elif not threshold_counts_non_regressing:
                    candidate_reason = "high-ambiguity row counts regressed"
                elif tail_improvement < 1e-12:
                    candidate_reason = "worst-ten uncertainty tail did not improve"
                else:
                    candidate_reason = "full-pool confidence profile did not improve"
            else:
                candidate_reason = "no full-pool uncertainty improvement"

        uncertainty_eliminated = (
            promoted
            and float(ambiguity_summary(incumbent_predictions)["mean"]) <= 1e-12
        )
        continuing = pass_number < max_passes and not uncertainty_eliminated
        if uncertainty_eliminated:
            stop_reason = "full-pool uncertainty eliminated"
        elif pass_number == max_passes:
            stop_reason = "maximum search passes reached"
            if not promoted:
                stop_reason += f"; last candidate: {candidate_reason}"
        else:
            stop_reason = "search continuing"

        current_report = SquashRunReport(
            seed_candidate=seed_candidate,
            final_candidate=incumbent,
            initial_ambiguity=initial_ambiguity,
            final_ambiguity=ambiguity_summary(incumbent_predictions),
            initial_true=_true_count(initial_predictions),
            final_true=_true_count(incumbent_predictions),
            final_prediction_flips=_flip_count(
                initial_predictions, incumbent_predictions
            ),
            total_metric_calls=total_metric_calls,
            stop_reason=stop_reason,
            passes=list(reports),
            final_predictions=list(incumbent_predictions),
            initial_output_counts=_output_counts(initial_predictions),
            final_output_counts=_output_counts(incumbent_predictions),
        )
        _emit_progress(
            progress,
            "full_pool_complete",
            pass_report=pass_report,
            improvement=improvement,
            tail_improvement=tail_improvement,
            threshold_counts_non_regressing=threshold_counts_non_regressing,
            min_improvement=min_improvement,
            candidate_reason=candidate_reason,
            continuing=continuing,
            uncertainty_eliminated=uncertainty_eliminated,
            stop_reason=stop_reason,
            total_count=len(proposed_predictions),
        )
        if report_path is not None:
            report_path.write_text(
                json.dumps(current_report.as_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if uncertainty_eliminated:
            return current_report

    return current_report
