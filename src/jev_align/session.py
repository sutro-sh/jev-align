from __future__ import annotations

import difflib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .acquisition import Acquisition, ambiguity_summary, select_batch
from .backends import (
    EvaluationBackend,
    backend_display_name,
    evaluate_with_progress,
)
from .models import (
    BinaryMetrics,
    CandidateHistory,
    CandidateSpec,
    ClassificationMetrics,
    LabelRecord,
    MulticlassCandidateSpec,
    MulticlassMetrics,
    MultilabelCandidateSpec,
    MultilabelMetrics,
    Prediction,
    RunState,
    ScoreCandidateSpec,
    ScoreMetrics,
    Story,
    TaskSpec,
)
from .optimizer import evaluate_labeled_candidate, optimize_candidate
from .persistence import RunStore


@dataclass
class RoundReport:
    previous_candidate: TaskSpec
    proposed_candidate: TaskSpec
    previous_metrics: ClassificationMetrics
    proposed_metrics: ClassificationMetrics
    previous_ambiguity: dict[str, float | int]
    proposed_ambiguity: dict[str, float | int]
    total_metric_calls: int
    configured_metric_budget: int
    ambiguity_scope: str = "fixed_full_pool"
    previous_replay_ambiguity: dict[str, float | int] | None = None
    proposed_replay_ambiguity: dict[str, float | int] | None = None
    previous_holdout_metrics: ClassificationMetrics | None = None
    proposed_holdout_metrics: ClassificationMetrics | None = None
    previous_capture_ambiguity: dict[str, float | int] | None = None
    proposed_capture_ambiguity: dict[str, float | int] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "previous_candidate": self.previous_candidate.model_dump(),
            "proposed_candidate": self.proposed_candidate.model_dump(),
            "previous_metrics": self.previous_metrics.model_dump(),
            "proposed_metrics": self.proposed_metrics.model_dump(),
            "previous_ambiguity": self.previous_ambiguity,
            "proposed_ambiguity": self.proposed_ambiguity,
            "total_metric_calls": self.total_metric_calls,
            "configured_metric_budget": self.configured_metric_budget,
            "ambiguity_scope": self.ambiguity_scope,
            "previous_replay_ambiguity": self.previous_replay_ambiguity,
            "proposed_replay_ambiguity": self.proposed_replay_ambiguity,
            "previous_capture_ambiguity": self.previous_capture_ambiguity,
            "proposed_capture_ambiguity": self.proposed_capture_ambiguity,
            "previous_holdout_metrics": (
                self.previous_holdout_metrics.model_dump()
                if self.previous_holdout_metrics is not None
                else None
            ),
            "proposed_holdout_metrics": (
                self.proposed_holdout_metrics.model_dump()
                if self.proposed_holdout_metrics is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RoundReport":
        previous_candidate = _candidate_from_dict(value["previous_candidate"])
        proposed_candidate = _candidate_from_dict(value["proposed_candidate"])
        if "fit_score" in value["previous_metrics"]:
            metrics_type = ScoreMetrics
        elif "micro_f1" in value["previous_metrics"]:
            metrics_type = MultilabelMetrics
        elif "macro_f1" in value["previous_metrics"]:
            metrics_type = MulticlassMetrics
        else:
            metrics_type = BinaryMetrics
        return cls(
            previous_candidate=previous_candidate,
            proposed_candidate=proposed_candidate,
            previous_metrics=metrics_type.model_validate(value["previous_metrics"]),
            proposed_metrics=metrics_type.model_validate(value["proposed_metrics"]),
            previous_ambiguity=value["previous_ambiguity"],
            proposed_ambiguity=value["proposed_ambiguity"],
            total_metric_calls=value["total_metric_calls"],
            configured_metric_budget=value["configured_metric_budget"],
            # Reports written before the fixed-panel change measured only the
            # shrinking unlabeled pool. Keep them resumable and label them
            # accurately in the CLI.
            ambiguity_scope=value.get("ambiguity_scope", "remaining_unlabeled"),
            previous_replay_ambiguity=value.get("previous_replay_ambiguity"),
            proposed_replay_ambiguity=value.get("proposed_replay_ambiguity"),
            previous_capture_ambiguity=value.get("previous_capture_ambiguity"),
            proposed_capture_ambiguity=value.get("proposed_capture_ambiguity"),
            previous_holdout_metrics=(
                metrics_type.model_validate(value["previous_holdout_metrics"])
                if value.get("previous_holdout_metrics") is not None
                else None
            ),
            proposed_holdout_metrics=(
                metrics_type.model_validate(value["proposed_holdout_metrics"])
                if value.get("proposed_holdout_metrics") is not None
                else None
            ),
        )


def _candidate_from_dict(value: dict[str, Any]) -> TaskSpec:
    if "levels" in value:
        return ScoreCandidateSpec.model_validate(value)
    if "labels" in value:
        return MultilabelCandidateSpec.model_validate(value)
    if "criteria" in value:
        return MulticlassCandidateSpec.model_validate(value)
    return CandidateSpec.model_validate(value)


def candidate_diff(before: TaskSpec, after: TaskSpec) -> str:
    old = json.dumps(before.model_dump(), indent=2, sort_keys=True).splitlines()
    new = json.dumps(after.model_dump(), indent=2, sort_keys=True).splitlines()
    return "\n".join(
        difflib.unified_diff(
            old, new, fromfile="current", tofile="proposed", lineterm=""
        )
    )


def label_examples(labels: list[LabelRecord]) -> list[dict[str, Any]]:
    return [
        {
            "story_id": item.story_id,
            "label": item.label,
            "rationale": item.rationale,
            "round_number": item.round_number,
        }
        for item in labels
    ]


def primary_metric(metrics: ClassificationMetrics | None) -> float | None:
    if metrics is None:
        return None
    if isinstance(metrics, ScoreMetrics):
        return metrics.fit_score
    if isinstance(metrics, MultilabelMetrics):
        return metrics.micro_f1
    if isinstance(metrics, MulticlassMetrics):
        return metrics.macro_f1
    return metrics.f1


class ClimbSession:
    def __init__(
        self,
        state: RunState,
        stories: list[Story],
        store: RunStore,
        backend: EvaluationBackend,
    ) -> None:
        self.state = state
        # Only this original panel participates in fixed-pool diagnostics/cache.
        self.stories = stories
        self.captured_stories = store.load_captured_inputs()
        self.capture_pool: list[Story] | None = None
        self.story_by_id = {
            story.id: story for story in [*stories, *self.captured_stories]
        }
        self.store = store
        self.backend = backend

    def use_captured_inputs(self, stories: list[Story]) -> None:
        """Persist the user-approved input snapshot before collecting any labels."""
        by_id = {story.id: story for story in self.captured_stories}
        by_id.update({story.id: story for story in stories})
        self.store.save_captured_inputs(list(by_id.values()))
        self.captured_stories = list(by_id.values())
        self.story_by_id.update(by_id)
        self.capture_pool = stories

    def unlabeled_stories(self) -> list[Story]:
        labeled = {
            record.story_id
            for record in self.store.load_labels()
            if record.evaluation_split == "train"
        }
        holdout = set(self.state.holdout_story_ids)
        return [
            story
            for story in (
                self.capture_pool if self.capture_pool is not None else self.stories
            )
            if story.id not in labeled and story.id not in holdout
        ]

    def unlabeled_holdout_stories(self) -> list[Story]:
        labeled = {
            record.story_id
            for record in self.store.load_labels()
            if record.evaluation_split == "holdout"
        }
        holdout = set(self.state.holdout_story_ids)
        return [
            story
            for story in self.stories
            if story.id in holdout and story.id not in labeled
        ]

    def _pool_cache_path(
        self, slot: Literal["current", "pending"], *, captured: bool = False
    ) -> Path:
        prefix = "capture" if captured else "pool"
        return self.store.directory / f"{prefix}-predictions-{slot}.json"

    def _evaluation_pool(self, *, captured: bool) -> list[Story]:
        if captured:
            if self.capture_pool is None:
                raise ValueError("no captured pool has been selected")
            return self.capture_pool
        return self.stories

    def _load_pool_cache(
        self,
        slot: Literal["current", "pending"],
        candidate: TaskSpec,
        *,
        captured: bool = False,
    ) -> list[Prediction] | None:
        stories = self._evaluation_pool(captured=captured)
        path = self._pool_cache_path(slot, captured=captured)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("candidate") != candidate.model_dump():
                return None
            cached_backend = payload.get("backend")
            if cached_backend is None and payload.get("jev_model") is not None:
                cached_backend = {
                    "provider": "typesafe",
                    "model": payload["jev_model"],
                    "options": {},
                }
            if cached_backend != self.state.backend.model_dump():
                return None
            if payload.get("source_sha256") != self.state.source_sha256:
                return None
            if (
                "selected_columns" in payload
                and payload["selected_columns"] != self.state.selected_columns
            ):
                return None
            if (
                "column_mode" in payload
                and payload["column_mode"] != self.state.column_mode
            ):
                return None
            predictions = [
                Prediction.model_validate(item) for item in payload["predictions"]
            ]
            expected_ids = {story.id for story in stories}
            if {item.story_id for item in predictions} != expected_ids:
                return None
            if len(predictions) != len(stories):
                return None
            return predictions
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _save_pool_cache(
        self,
        slot: Literal["current", "pending"],
        candidate: TaskSpec,
        predictions: list[Prediction],
        *,
        captured: bool = False,
    ) -> None:
        self.store.write_json(
            self._pool_cache_path(slot, captured=captured).name,
            {
                "version": 2,
                "candidate": candidate.model_dump(),
                "backend": self.state.backend.model_dump(),
                "source_sha256": self.state.source_sha256,
                "selected_columns": self.state.selected_columns,
                "column_mode": self.state.column_mode,
                "predictions": [item.model_dump() for item in predictions],
            },
        )

    def _evaluate_pool(
        self,
        candidate: TaskSpec,
        slot: Literal["current", "pending"],
        *,
        captured: bool = False,
    ) -> list[Prediction]:
        phase = "current" if slot == "current" else "proposed"
        pool_name = "captured pool" if captured else "AI Function"
        predictions = evaluate_with_progress(
            self.backend,
            candidate,
            self._evaluation_pool(captured=captured),
            f"{backend_display_name(self.backend)} · {phase} {pool_name}",
        )
        self._save_pool_cache(slot, candidate, predictions, captured=captured)
        return predictions

    def current_pool_predictions(self, *, captured: bool = False) -> list[Prediction]:
        cached = self._load_pool_cache(
            "current", self.state.current_candidate, captured=captured
        )
        if cached is not None:
            return cached
        return self._evaluate_pool(
            self.state.current_candidate, "current", captured=captured
        )

    def acquire(self) -> tuple[list[Any], list[Prediction]]:
        # Score the frozen original pool every round. Labeled rows are removed
        # only from acquisition eligibility, not from longitudinal diagnostics.
        predictions = self.current_pool_predictions()
        acquisition_pool = (
            self.current_pool_predictions(captured=True)
            if self.capture_pool is not None else predictions
        )
        labeled_ids = {
            record.story_id
            for record in self.store.load_labels()
            if record.evaluation_split == "train"
        }
        holdout_ids = set(self.state.holdout_story_ids)
        unlabeled_predictions = [
            item
            for item in acquisition_pool
            if item.story_id not in labeled_ids and item.story_id not in holdout_ids
        ]
        acquisitions = select_batch(
            unlabeled_predictions,
            batch_size=(
                min(self.state.batch_size, len(unlabeled_predictions))
                if self.capture_pool is not None else self.state.batch_size
            ),
            exploration_count=1,
            seed=self.state.seed + self.state.round_number,
        )
        return acquisitions, predictions

    def acquire_holdout(
        self, pool_predictions: list[Prediction]
    ) -> list[Acquisition]:
        count = min(
            self.state.holdout_batch_size,
            len(self.unlabeled_holdout_stories()),
        )
        if count == 0:
            return []
        unlabeled_ids = {story.id for story in self.unlabeled_holdout_stories()}
        available = [
            prediction
            for prediction in pool_predictions
            if prediction.story_id in unlabeled_ids
        ]
        rng = random.Random(self.state.seed + self.state.round_number)
        return [
            Acquisition(prediction, "holdout")
            for prediction in rng.sample(available, count)
        ]

    def add_label(
        self,
        *,
        story_id: str,
        label: bool | int | str | list[str],
        rationale: str | None,
        acquired_by: Literal["ambiguous", "exploration", "holdout"],
        evaluation_split: Literal["train", "holdout"] = "train",
        probability: float,
        resolved_model: str | None = None,
    ) -> LabelRecord:
        story = self.story_by_id.get(story_id)
        if story is None:
            raise ValueError(f"unknown story ID: {story_id}")
        record = LabelRecord(
            story_id=story_id,
            inputs=dict(story.fields),
            label=label,
            rationale=rationale.strip() if rationale and rationale.strip() else None,
            round_number=self.state.round_number,
            acquired_by=acquired_by,
            evaluation_split=evaluation_split,
            acquisition_probability=probability,
            resolved_model=resolved_model,
        )
        self.store.append_label(record)
        return record

    def undo_last_label(self, story_id: str) -> LabelRecord:
        return self.store.pop_last_label(
            story_id=story_id,
            round_number=self.state.round_number,
        )

    def rewind_previous_round(self) -> tuple[int, Path]:
        if self.state.pending_candidate is not None:
            raise ValueError("cannot rewind while a candidate decision is pending")
        if self.state.round_number <= 1:
            raise ValueError("there is no previous round to rewind")

        target_round = self.state.round_number - 1
        archive = self.store.archive_rewind(target_round)
        retained_history = [
            item for item in self.state.history if item.round_number < target_round
        ]
        restored = next(
            item.candidate
            for item in reversed(retained_history)
            if item.decision in {"seed", "accepted"}
        )
        all_labels = self.store.load_labels()
        reopened = [
            self.story_by_id[item.story_id]
            for item in all_labels
            if item.round_number == target_round and item.evaluation_split == "train"
        ]
        captured_ids = {story.id for story in self.captured_stories}
        if any(story.id in captured_ids for story in reopened):
            # Rewinding after changing acquisition sources still reopens the
            # previous round's captured examples, including after a restart.
            self.capture_pool = list({
                story.id: story for story in [*reopened, *(self.capture_pool or [])]
            }.values())
        else:
            self.capture_pool = None
        retained_labels = [
            item
            for item in all_labels
            if item.round_number < target_round
        ]
        self.store.replace_labels(retained_labels)
        self.state.round_number = target_round
        self.state.current_candidate = restored
        self.state.history = retained_history
        self.state.pending_candidate = None
        self.state.pending_report = None
        self.store.save_state(self.state)
        return target_round, archive

    def ready_to_optimize(self) -> bool:
        labels = [
            item
            for item in self.store.load_labels()
            if item.evaluation_split == "train"
        ]
        if isinstance(
            self.state.current_candidate,
            (MultilabelCandidateSpec, ScoreCandidateSpec),
        ):
            return len(labels) >= self.state.batch_size
        if isinstance(self.state.current_candidate, MulticlassCandidateSpec):
            return len({str(item.label) for item in labels}) >= 2
        return any(item.label for item in labels) and any(
            not item.label for item in labels
        )

    def optimize(self, pool_predictions: list[Prediction]) -> RoundReport:
        all_labels = self.store.load_labels()
        labels = [item for item in all_labels if item.evaluation_split == "train"]
        holdout_labels = [
            item for item in all_labels if item.evaluation_split == "holdout"
        ]
        examples = label_examples(labels)
        holdout_examples = label_examples(holdout_labels)
        previous_metrics, previous_training = evaluate_labeled_candidate(
            self.state.current_candidate, examples, self.story_by_id, self.backend
        )
        outcome = optimize_candidate(
            seed_candidate=self.state.current_candidate,
            examples=examples,
            stories=self.story_by_id,
            backend=self.backend,
            reflection_model=self.state.reflection_model,
            metric_budget=self.state.metric_budget,
            output_dir=self.store.gepa_output_dir,
            run_dir=self.store.gepa_run_dir,
            round_number=self.state.round_number,
            concurrency=self.state.concurrency,
            seed=self.state.seed,
        )
        proposed_metrics, proposed_training = evaluate_labeled_candidate(
            outcome.candidate, examples, self.story_by_id, self.backend
        )
        previous_holdout_metrics = None
        proposed_holdout_metrics = None
        if holdout_examples:
            previous_holdout_metrics, _ = evaluate_labeled_candidate(
                self.state.current_candidate,
                holdout_examples,
                self.story_by_id,
                self.backend,
            )
            proposed_holdout_metrics, _ = evaluate_labeled_candidate(
                outcome.candidate,
                holdout_examples,
                self.story_by_id,
                self.backend,
            )

        proposed_pool = self._evaluate_pool(outcome.candidate, "pending")
        previous_capture_ambiguity = None
        proposed_capture_ambiguity = None
        if self.capture_pool is not None:
            previous_capture_ambiguity = ambiguity_summary(
                self.current_pool_predictions(captured=True)
            )
            proposed_capture_ambiguity = ambiguity_summary(
                self._evaluate_pool(outcome.candidate, "pending", captured=True)
            )
        previous_replay = list(previous_training.values())
        proposed_replay = list(proposed_training.values())
        report = RoundReport(
            previous_candidate=self.state.current_candidate,
            proposed_candidate=outcome.candidate,
            previous_metrics=previous_metrics,
            proposed_metrics=proposed_metrics,
            previous_ambiguity=ambiguity_summary(pool_predictions),
            proposed_ambiguity=ambiguity_summary(proposed_pool),
            total_metric_calls=outcome.total_metric_calls,
            configured_metric_budget=self.state.metric_budget,
            previous_replay_ambiguity=ambiguity_summary(previous_replay),
            proposed_replay_ambiguity=ambiguity_summary(proposed_replay),
            previous_holdout_metrics=previous_holdout_metrics,
            proposed_holdout_metrics=proposed_holdout_metrics,
            previous_capture_ambiguity=previous_capture_ambiguity,
            proposed_capture_ambiguity=proposed_capture_ambiguity,
        )
        self.state.pending_candidate = outcome.candidate
        self.state.pending_report = report.as_dict()
        self.store.write_json(
            f"round-{self.state.round_number:04d}-report.json", report.as_dict()
        )
        self.store.save_state(self.state)
        return report

    def decide(self, decision: Literal["accept", "reject"]) -> None:
        if self.state.pending_candidate is None or self.state.pending_report is None:
            raise ValueError("there is no pending candidate")
        report = RoundReport.from_dict(self.state.pending_report)
        if decision == "accept":
            self.state.current_candidate = self.state.pending_candidate
            history_decision: Literal["accepted", "rejected"] = "accepted"
        else:
            history_decision = "rejected"
        self.state.history.append(
            CandidateHistory(
                round_number=self.state.round_number,
                candidate=self.state.pending_candidate,
                decision=history_decision,
                fit_f1=primary_metric(report.proposed_metrics),
                holdout_score=primary_metric(report.proposed_holdout_metrics),
            )
        )
        self.state.pending_candidate = None
        self.state.pending_report = None
        self.state.round_number += 1
        self.store.save_state(self.state)
        if decision == "accept":
            scopes = [False]
            if report.proposed_capture_ambiguity is not None:
                scopes.append(True)
            for captured in scopes:
                pending_cache = self._pool_cache_path("pending", captured=captured)
                if pending_cache.exists():
                    os.replace(
                        pending_cache,
                        self._pool_cache_path("current", captured=captured),
                    )
