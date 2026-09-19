from __future__ import annotations

from collections.abc import Sequence

from jev_align.models import (
    CandidateSpec,
    MulticlassCandidateSpec,
    MultilabelCandidateSpec,
    MultilabelCriteria,
    Prediction,
    ScoreCandidateSpec,
    Story,
)
from jev_align.optimizer import (
    ClassAwareBatchSampler,
    F1BatchEvaluator,
    evaluate_labeled_candidate,
    optimize_candidate,
)


class FakeJev:
    def evaluate_many(
        self, candidate: CandidateSpec, stories: Sequence[Story]
    ) -> list[Prediction]:
        probabilities = {
            "a": 0.9,
            "b": 0.8,
            "c": 0.7,
            "d": 0.1,
        }
        return [
            Prediction(story_id=story.id, probability=probabilities[story.id])
            for story in stories
        ]


def _candidate() -> dict[str, str]:
    return {
        "instructions": "Is it aviation?",
        "true_criteria": "It is about aviation.",
        "false_criteria": "It is not about aviation.",
    }


def test_batch_evaluator_returns_global_f1_and_row_feedback() -> None:
    stories = {
        key: Story(id=key, row_number=index, fields={"title": key})
        for index, key in enumerate("abcd", start=1)
    }
    examples = [
        {"story_id": "a", "label": True, "rationale": "aircraft"},
        {"story_id": "b", "label": False, "rationale": "metaphor"},
        {"story_id": "c", "label": True, "rationale": None},
        {"story_id": "d", "label": False, "rationale": None},
    ]
    evaluator = F1BatchEvaluator(
        FakeJev(), stories, CandidateSpec.model_validate(_candidate())
    )
    results = evaluator([(_candidate(), example) for example in examples])
    assert [score for score, _ in results] == [0.8, 0.8, 0.8, 0.8]
    assert results[1][1]["error_type"] == "false_positive"
    assert results[1][1]["human_rationale"] == "metaphor"
    assert results[1][1]["scores"] == {"correct": 0.0}
    assert results[0][1]["scores"] == {"correct": 1.0}


class FakeLoader:
    def __init__(self) -> None:
        self.items = {
            "a": {"label": True},
            "b": {"label": True},
            "c": {"label": False},
            "d": {"label": False},
            "e": {"label": False},
            "f": {"label": False},
        }

    def all_ids(self):
        return self.items.keys()

    def fetch(self, ids):
        return [self.items[item_id] for item_id in ids]


def test_class_aware_sampler_includes_both_classes() -> None:
    loader = FakeLoader()
    sampler = ClassAwareBatchSampler(batch_size=3, seed=7)
    selected = sampler.next_minibatch_ids(loader, None)
    labels = {loader.items[item_id]["label"] for item_id in selected}
    assert labels == {True, False}
    assert sampler.next_minibatch_ids(loader, None) == selected


class FakeChoiceJev:
    def evaluate_many(self, candidate, stories):
        choices = {"a": "aviation", "b": "aviation", "c": "space", "d": "other"}
        return [
            Prediction(
                story_id=story.id,
                probabilities={
                    name: 1.0 if name == choices[story.id] else 0.0
                    for name in candidate.criteria
                },
                choice=choices[story.id],
                confidence=1.0,
            )
            for story in stories
        ]


def test_multiclass_batch_evaluator_returns_macro_f1_and_feedback() -> None:
    candidate = MulticlassCandidateSpec(
        instructions="What kind of post is this?",
        criteria={
            "aviation": "Aircraft and flight",
            "space": "Spaceflight and astronomy",
            "other": "Neither category",
        },
    )
    stories = {
        key: Story(id=key, row_number=index, fields={"title": key})
        for index, key in enumerate("abcd", start=1)
    }
    examples = [
        {"story_id": "a", "label": "aviation", "rationale": None},
        {"story_id": "b", "label": "space", "rationale": "rocket, not aircraft"},
        {"story_id": "c", "label": "space", "rationale": None},
        {"story_id": "d", "label": "other", "rationale": None},
    ]
    evaluator = F1BatchEvaluator(FakeChoiceJev(), stories, candidate)
    results = evaluator([(candidate.as_gepa(), example) for example in examples])
    assert all(round(score, 3) == 0.778 for score, _ in results)
    assert results[1][1]["error_type"] == "misclassified"
    assert results[1][1]["choice_confidence"] == 1.0


class FakeMultilabelJev:
    def evaluate_many(self, candidate, stories):
        values = {
            "a": {"tech": 0.9, "startups": 0.8, "culture": 0.1},
            "b": {"tech": 0.8, "startups": 0.2, "culture": 0.7},
            "c": {"tech": 0.1, "startups": 0.1, "culture": 0.1},
        }
        return [
            Prediction(story_id=story.id, label_probabilities=values[story.id])
            for story in stories
        ]


def test_multilabel_batch_evaluator_returns_micro_f1_and_label_feedback() -> None:
    candidate = MultilabelCandidateSpec(
        instructions="Apply every relevant topic label.",
        labels={
            name: MultilabelCriteria(
                true_criteria=f"The row concerns {name}.",
                false_criteria=f"The row does not concern {name}.",
            )
            for name in ("tech", "startups", "culture")
        },
    )
    stories = {
        key: Story(id=key, row_number=index, fields={"title": key})
        for index, key in enumerate("abc", start=1)
    }
    examples = [
        {"story_id": "a", "label": ["tech", "startups"], "rationale": None},
        {"story_id": "b", "label": ["tech"], "rationale": "not cultural"},
        {"story_id": "c", "label": [], "rationale": None},
    ]
    evaluator = F1BatchEvaluator(FakeMultilabelJev(), stories, candidate)
    results = evaluator([(candidate.as_gepa(), example) for example in examples])
    assert all(round(score, 3) == 0.857 for score, _ in results)
    assert results[1][1]["error_type"] == "label_mismatch"
    assert results[1][1]["false_positive_labels"] == ["culture"]
    assert results[1][1]["false_negative_labels"] == []
    assert results[2][1]["predicted_label"] == "(none)"


class FakeScoreJev:
    def evaluate_many(self, candidate, stories):
        values = {"a": 0.5, "b": 1.5}
        return [
            Prediction(
                story_id=story.id,
                score=values[story.id],
                score_probabilities={0: 0.25, 1: 0.5, 2: 0.25},
                confidence=0.5,
            )
            for story in stories
        ]


def test_score_batch_evaluator_uses_normalized_mae_and_ordinal_feedback() -> None:
    candidate = ScoreCandidateSpec(
        instructions="How severe is this issue?",
        levels=["Cosmetic", "Degraded with workaround", "Blocking"],
    )
    stories = {
        key: Story(id=key, row_number=index, fields={"title": key})
        for index, key in enumerate("ab", start=1)
    }
    examples = [
        {"story_id": "a", "label": 0, "rationale": None},
        {"story_id": "b", "label": 2, "rationale": "No workaround"},
    ]
    evaluator = F1BatchEvaluator(FakeScoreJev(), stories, candidate)

    results = evaluator([(candidate.as_gepa(), example) for example in examples])

    assert [score for score, _ in results] == [0.75, 0.75]
    assert results[0][1]["error_type"] == "score_error"
    assert results[1][1]["error_type"] == "correct"
    assert results[1][1]["score"] == 1.5
    assert results[1][1]["scores"] == {"correct": 0.75}


def test_score_batch_evaluator_rejects_cross_component_text() -> None:
    candidate = ScoreCandidateSpec(
        instructions="How severe is this issue?",
        levels=["Cosmetic", "Degraded", "Blocking"],
    )
    corrupted = candidate.as_gepa()
    corrupted["instructions"] += "\n\nlevel::0\nCosmetic"
    story = Story(id="a", row_number=1, fields={"title": "a"})
    evaluator = F1BatchEvaluator(FakeScoreJev(), {"a": story}, candidate)

    results = evaluator([(corrupted, {"story_id": "a", "label": 0, "rationale": None})])

    assert results[0][0] == 0.0
    assert "must not contain" in results[0][1]["error"]


class CandidateSensitiveChoiceJev:
    def evaluate_many(self, candidate, stories):
        labels = list(candidate.criteria)
        improved = candidate.instructions != "seed"
        predictions = []
        for story in stories:
            choice = story.fields["label"] if improved else labels[0]
            predictions.append(
                Prediction(
                    story_id=story.id,
                    probabilities={
                        name: 1.0 if name == choice else 0.0 for name in labels
                    },
                    choice=choice,
                    confidence=1.0,
                )
            )
        return predictions


def test_real_gepa_loop_optimizes_multiclass_candidate(tmp_path) -> None:
    candidate = MulticlassCandidateSpec(
        instructions="seed",
        criteria={
            "aviation": "Aircraft and flight",
            "space": "Spaceflight and astronomy",
            "other": "Neither category",
        },
    )
    labels = ["aviation", "space", "other", "aviation", "space", "other"]
    stories = {
        str(index): Story(
            id=str(index),
            row_number=index,
            fields={"text": f"example {index}", "label": label},
        )
        for index, label in enumerate(labels, start=1)
    }
    examples = [
        {
            "story_id": story_id,
            "label": story.fields["label"],
            "rationale": "deterministic integration fixture",
            "round_number": 1,
        }
        for story_id, story in stories.items()
    ]
    proposal_number = 0

    def reflection_lm(_prompt: str) -> str:
        nonlocal proposal_number
        proposal_number += 1
        return f"```\nimproved component {proposal_number}\n```"

    outcome = optimize_candidate(
        seed_candidate=candidate,
        examples=examples,
        stories=stories,
        backend=CandidateSensitiveChoiceJev(),
        reflection_model=reflection_lm,
        metric_budget=300,
        output_dir=tmp_path / "output",
        run_dir=tmp_path / "runs",
        round_number=1,
        concurrency=2,
        seed=7,
    )

    assert isinstance(outcome.candidate, MulticlassCandidateSpec)
    assert outcome.candidate.instructions != "seed"
    assert list(outcome.candidate.criteria) == list(candidate.criteria)
    fitted_metrics, _ = evaluate_labeled_candidate(
        outcome.candidate, examples, stories, CandidateSensitiveChoiceJev()
    )
    assert fitted_metrics.macro_f1 == 1.0
    assert outcome.best_score > 0.16
    assert outcome.total_metric_calls > 0
    assert outcome.total_metric_calls < 300
    assert proposal_number == 16
    assert (tmp_path / "output" / "round-0001" / "gepa-result.json").exists()


class CandidateSensitiveScoreJev:
    def evaluate_many(self, candidate, stories):
        improved = candidate.instructions != "seed"
        predictions = []
        for story in stories:
            expected = int(story.fields["level"])
            score = float(expected if improved else 0)
            predictions.append(
                Prediction(
                    story_id=story.id,
                    score=score,
                    score_probabilities={
                        level: 1.0 if level == score else 0.0
                        for level in range(len(candidate.levels))
                    },
                    confidence=1.0,
                )
            )
        return predictions


def test_real_gepa_loop_optimizes_score_candidate(tmp_path) -> None:
    candidate = ScoreCandidateSpec(
        instructions="seed",
        levels=["Cosmetic", "Degraded", "Blocking"],
    )
    labels = [0, 1, 2, 1, 2]
    stories = {
        str(index): Story(
            id=str(index),
            row_number=index,
            fields={"text": f"example {index}", "level": str(label)},
        )
        for index, label in enumerate(labels, start=1)
    }
    examples = [
        {
            "story_id": story_id,
            "label": label,
            "rationale": "deterministic integration fixture",
            "round_number": 1,
        }
        for story_id, label in zip(stories, labels, strict=True)
    ]
    proposal_number = 0

    def reflection_lm(_prompt: str) -> str:
        nonlocal proposal_number
        proposal_number += 1
        return f"```\nimproved component {proposal_number}\n```"

    outcome = optimize_candidate(
        seed_candidate=candidate,
        examples=examples,
        stories=stories,
        backend=CandidateSensitiveScoreJev(),
        reflection_model=reflection_lm,
        metric_budget=300,
        output_dir=tmp_path / "output",
        run_dir=tmp_path / "runs",
        round_number=1,
        concurrency=2,
        seed=11,
    )

    assert isinstance(outcome.candidate, ScoreCandidateSpec)
    assert outcome.candidate.instructions != "seed"
    assert len(outcome.candidate.levels) == 3
    fitted_metrics, _ = evaluate_labeled_candidate(
        outcome.candidate, examples, stories, CandidateSensitiveScoreJev()
    )
    assert fitted_metrics.fit_score == 1.0
    assert outcome.best_score == 1.0
    assert 0 < outcome.total_metric_calls < 300
    assert (tmp_path / "output" / "round-0001" / "gepa-result.json").exists()


class CandidateSensitiveMultilabelJev:
    def evaluate_many(self, candidate, stories):
        improved = candidate.instructions != "seed"
        predictions = []
        for story in stories:
            expected = set(story.fields["labels"].split(",")) - {""}
            predictions.append(
                Prediction(
                    story_id=story.id,
                    label_probabilities={
                        label: 0.9 if improved and label in expected else 0.1
                        for label in candidate.labels
                    },
                )
            )
        return predictions


def test_real_gepa_loop_optimizes_multilabel_candidate_and_stops(tmp_path) -> None:
    candidate = MultilabelCandidateSpec(
        instructions="seed",
        labels={
            name: MultilabelCriteria(
                true_criteria=f"About {name}", false_criteria=f"Not about {name}"
            )
            for name in ("tech", "startups", "culture")
        },
    )
    expected = [
        ["tech", "startups"],
        ["tech"],
        [],
        ["startups"],
        ["tech", "startups"],
    ]
    stories = {
        str(index): Story(
            id=str(index),
            row_number=index,
            fields={"labels": ",".join(labels)},
        )
        for index, labels in enumerate(expected, start=1)
    }
    examples = [
        {
            "story_id": story_id,
            "label": labels,
            "rationale": None,
            "round_number": 1,
        }
        for story_id, labels in zip(stories, expected, strict=True)
    ]
    proposal_number = 0

    def reflection_lm(_prompt: str) -> str:
        nonlocal proposal_number
        proposal_number += 1
        return f"```\nimproved component {proposal_number}\n```"

    outcome = optimize_candidate(
        seed_candidate=candidate,
        examples=examples,
        stories=stories,
        backend=CandidateSensitiveMultilabelJev(),
        reflection_model=reflection_lm,
        metric_budget=300,
        output_dir=tmp_path / "output",
        run_dir=tmp_path / "runs",
        round_number=1,
        concurrency=2,
        seed=9,
    )

    assert isinstance(outcome.candidate, MultilabelCandidateSpec)
    assert outcome.candidate.instructions != "seed"
    assert list(outcome.candidate.labels) == list(candidate.labels)
    fitted_metrics, _ = evaluate_labeled_candidate(
        outcome.candidate, examples, stories, CandidateSensitiveMultilabelJev()
    )
    assert fitted_metrics.micro_f1 == 1.0
    assert fitted_metrics.per_label["culture"].support == 0
    assert outcome.total_metric_calls < 300
    assert proposal_number == 4
