from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

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
    OptimizationOutcome,
    SquashBatchEvaluator,
    SquashCandidateProposer,
    _parse_squash_candidate_response,
    _squash_reflection_lm,
    squash_candidate,
    squash_metric_budget,
)
from jev_align.squash import SquashRunReport, run_squash, select_squash_working_set


def _candidate(instructions: str = "seed") -> CandidateSpec:
    return CandidateSpec(
        instructions=instructions,
        true_criteria="Return true.",
        false_criteria="Return false.",
    )


def _stories(count: int = 10) -> list[Story]:
    return [
        Story(id=f"row-{index}", row_number=index, fields={"text": str(index)})
        for index in range(1, count + 1)
    ]


def _multiclass_candidate(
    instructions: str = "Choose the best topic.",
) -> MulticlassCandidateSpec:
    return MulticlassCandidateSpec(
        instructions=instructions,
        criteria={
            "air": "The row is about aviation.",
            "land": "The row is about ground transport.",
            "sea": "The row is about marine transport.",
        },
    )


def _multilabel_candidate(
    instructions: str = "Apply every matching tag.",
) -> MultilabelCandidateSpec:
    return MultilabelCandidateSpec(
        instructions=instructions,
        labels={
            "urgent": MultilabelCriteria(
                true_criteria="Immediate action is required.",
                false_criteria="Immediate action is not required.",
            ),
            "billing": MultilabelCriteria(
                true_criteria="The row concerns billing.",
                false_criteria="The row does not concern billing.",
            ),
        },
    )


def _score_candidate(instructions: str = "Rate severity.") -> ScoreCandidateSpec:
    return ScoreCandidateSpec(
        instructions=instructions,
        levels=["Minor", "Moderate", "Severe"],
    )


class ConfidenceBackend:
    def evaluate_many(
        self, candidate: CandidateSpec, stories: Sequence[Story]
    ) -> list[Prediction]:
        decisive = candidate.instructions == "decisive"
        return [
            Prediction(
                story_id=story.id,
                probability=(
                    (0.95 if story.row_number % 2 else 0.05)
                    if decisive
                    else 0.5 + (story.row_number - 5) / 100
                ),
            )
            for story in stories
        ]


def test_squash_evaluator_uses_batch_certainty_without_labels() -> None:
    stories = {story.id: story for story in _stories(2)}
    evaluator = SquashBatchEvaluator(ConfidenceBackend(), stories)
    candidate = _candidate()

    results = evaluator(
        [(candidate.as_gepa(), {"story_id": story_id}) for story_id in stories]
    )

    assert len(results) == 2
    assert results[0][0] == results[1][0]
    assert results[0][1]["warning"].startswith("There is no authoritative label")
    assert "expected_label" not in results[0][1]
    assert results[0][1]["scores"]["certainty"] == pytest.approx(0.08)


def test_squash_evaluator_uses_multiclass_choice_confidence() -> None:
    stories = {story.id: story for story in _stories(2)}
    candidate = _multiclass_candidate()

    class ChoiceBackend:
        def evaluate_many(self, _candidate, selected):
            confidences = [0.4, 0.8]
            return [
                Prediction(
                    story_id=story.id,
                    probabilities={
                        "air": confidence,
                        "land": (1.0 - confidence) / 2,
                        "sea": (1.0 - confidence) / 2,
                    },
                    choice="air",
                    confidence=confidence,
                )
                for story, confidence in zip(selected, confidences, strict=True)
            ]

    results = SquashBatchEvaluator(
        ChoiceBackend(), stories, candidate
    )(
        [(candidate.as_gepa(), {"story_id": story_id}) for story_id in stories]
    )

    assert [result[0] for result in results] == pytest.approx([0.6, 0.6])
    assert results[0][1]["predicted_label"] == "air"
    assert results[0][1]["choice_confidence"] == pytest.approx(0.4)
    assert results[0][1]["scores"]["certainty"] == pytest.approx(0.4)


def test_squash_evaluator_uses_worst_multilabel_boundary() -> None:
    stories = {story.id: story for story in _stories(2)}
    candidate = _multilabel_candidate()

    class MultilabelBackend:
        def evaluate_many(self, _candidate, selected):
            values = [
                {"urgent": 0.5, "billing": 0.9},
                {"urgent": 0.1, "billing": 0.2},
            ]
            return [
                Prediction(story_id=story.id, label_probabilities=probabilities)
                for story, probabilities in zip(selected, values, strict=True)
            ]

    results = SquashBatchEvaluator(
        MultilabelBackend(), stories, candidate
    )(
        [(candidate.as_gepa(), {"story_id": story_id}) for story_id in stories]
    )

    assert [result[0] for result in results] == pytest.approx([0.3, 0.3])
    assert results[0][1]["predicted_label"] == ["billing", "urgent"]
    assert results[0][1]["scores"]["certainty"] == pytest.approx(0.0)
    assert results[1][1]["scores"]["certainty"] == pytest.approx(0.6)


def test_squash_evaluator_uses_score_confidence() -> None:
    stories = {story.id: story for story in _stories(2)}
    candidate = _score_candidate()

    class ScoreBackend:
        def evaluate_many(self, _candidate, selected):
            confidences = [0.4, 0.8]
            return [
                Prediction(
                    story_id=story.id,
                    score=float(index),
                    score_probabilities={0: 0.2, 1: 0.6, 2: 0.2},
                    confidence=confidence,
                )
                for index, (story, confidence) in enumerate(
                    zip(selected, confidences, strict=True)
                )
            ]

    results = SquashBatchEvaluator(ScoreBackend(), stories, candidate)(
        [(candidate.as_gepa(), {"story_id": story_id}) for story_id in stories]
    )

    assert [result[0] for result in results] == pytest.approx([0.6, 0.6])
    assert results[0][1]["predicted_label"] == pytest.approx(0.0)
    assert results[0][1]["scores"]["certainty"] == pytest.approx(0.4)


def test_squash_proposer_updates_whole_candidate_from_structured_json() -> None:
    prompts = []

    def reflection_lm(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps(
            {
                "instructions": "Make a decisive binary classification.",
                "true_criteria": "True when AI is the central subject.",
                "false_criteria": "False otherwise.",
            }
        )

    proposal = SquashCandidateProposer(reflection_lm)(
        _candidate().as_gepa(),
        {
            name: [{"story_id": "row-1", "Scores (Higher is Better)": 0.2}]
            for name in ("instructions", "true_criteria", "false_criteria")
        },
        ["instructions", "true_criteria", "false_criteria"],
    )

    assert proposal == {
        "instructions": "Make a decisive binary classification.",
        "true_criteria": "True when AI is the central subject.",
        "false_criteria": "False otherwise.",
    }
    assert len(prompts) == 1
    assert '"instructions": "seed"' in prompts[0]
    assert "Return exactly one JSON object" in prompts[0]


def test_squash_proposer_preserves_multiclass_components() -> None:
    candidate = _multiclass_candidate()
    proposal = {
        "instructions": "Choose exactly one transport domain decisively.",
        "criterion::air": "Choose air only for aircraft and aviation.",
        "criterion::land": "Choose land for road and rail transport.",
        "criterion::sea": "Choose sea for ships and marine transport.",
    }
    prompts = []

    def reflection_lm(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps(proposal)

    result = SquashCandidateProposer(
        reflection_lm, seed_candidate=candidate
    )(
        candidate.as_gepa(),
        {"instructions": [{"story_id": "row-1", "ambiguity": 0.7}]},
        list(candidate.as_gepa()),
    )

    assert result == proposal
    assert "Do not add, remove, rename, merge, or reorder classes" in prompts[0]
    renamed = dict(proposal)
    renamed["criterion::space"] = renamed.pop("criterion::sea")
    with pytest.raises(ValueError, match="candidate components must be exactly"):
        _parse_squash_candidate_response(json.dumps(renamed), candidate)


@pytest.mark.parametrize(
    ("candidate", "replacement", "semantic_text"),
    [
        (
            _multilabel_candidate(),
            {
                "instructions": "Apply each tag independently and decisively.",
                "label::urgent::true": "Urgent only with an immediate deadline.",
                "label::urgent::false": "Not urgent without an immediate deadline.",
                "label::billing::true": "Billing when money or invoices are central.",
                "label::billing::false": "Not billing otherwise.",
            },
            "any subset—including no labels—must remain possible",
        ),
        (
            _score_candidate(),
            {
                "instructions": "Choose the strongest supported severity level.",
                "level::0": "Minor with negligible impact.",
                "level::1": "Moderate with material impact.",
                "level::2": "Severe with critical impact.",
            },
            "Do not add, remove, rename, merge, reorder, or invert score levels",
        ),
    ],
)
def test_squash_proposer_preserves_multilabel_and_score_shapes(
    candidate, replacement, semantic_text
) -> None:
    prompts = []

    def reflection_lm(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps(replacement)

    proposal = SquashCandidateProposer(
        reflection_lm, seed_candidate=candidate
    )(
        candidate.as_gepa(),
        {"instructions": [{"story_id": "row-1", "ambiguity": 0.8}]},
        list(candidate.as_gepa()),
    )

    assert proposal == replacement
    assert semantic_text in prompts[0]


def test_squash_proposer_retries_malformed_or_nested_candidates() -> None:
    outputs = iter(
        [
            "```yaml\ninstructions: nested\ntrue_criteria: nested\n```",
            json.dumps(
                {
                    "instructions": "Decide confidently.",
                    "true_criteria": "True for centrally AI posts.",
                    "false_criteria": "False for all other posts.",
                }
            ),
        ]
    )
    prompts = []

    def reflection_lm(prompt: str) -> str:
        prompts.append(prompt)
        return next(outputs)

    proposal = SquashCandidateProposer(reflection_lm)(
        _candidate().as_gepa(),
        {"instructions": []},
        ["instructions", "true_criteria", "false_criteria"],
    )

    assert proposal["instructions"] == "Decide confidently."
    assert len(prompts) == 2
    assert "previous response was rejected" in prompts[1]


def test_squash_proposer_rejects_unchanged_and_duplicate_branches() -> None:
    first = {
        "instructions": "Use a decisive default.",
        "true_criteria": "True only with direct evidence.",
        "false_criteria": "False otherwise.",
    }
    second = {
        "instructions": "Apply an ordered evidence test.",
        "true_criteria": "True when the strongest evidence is present.",
        "false_criteria": "False when that evidence is absent.",
    }
    outputs = iter(
        [
            json.dumps(_candidate().as_gepa()),
            json.dumps(first),
            json.dumps(first),
            json.dumps(second),
        ]
    )
    prompts = []

    def reflection_lm(prompt: str) -> str:
        prompts.append(prompt)
        return next(outputs)

    proposer = SquashCandidateProposer(
        reflection_lm,
        full_pool_ambiguity={"mean": 0.2, "at_least_0_5": 3},
    )
    feedback = {"instructions": [{"story_id": "row-1", "ambiguity": 0.9}]}
    components = ["instructions", "true_criteria", "false_criteria"]

    assert proposer(_candidate().as_gepa(), feedback, components) == first
    assert proposer(_candidate().as_gepa(), feedback, components) == second

    assert len(prompts) == 4
    assert "proposal is unchanged from its parent" in prompts[1]
    assert "duplicates another exploration branch" in prompts[3]
    assert "Exploration strategy 1: decisive default" in prompts[0]
    assert "Exploration strategy 2: ordered decision procedure" in prompts[2]
    assert '"at_least_0_5": 3' in prompts[0]


def test_squash_candidate_parser_rejects_cross_component_content() -> None:
    response = json.dumps(
        {
            "instructions": "Decide confidently.",
            "true_criteria": "true_criteria: True for everything.",
            "false_criteria": "False otherwise.",
        }
    )

    with pytest.raises(ValueError, match="nested candidate definition"):
        _parse_squash_candidate_response(response)


def test_squash_model_requests_strict_json_schema(monkeypatch) -> None:
    created = {}

    class FakeLM:
        def __init__(self, model, **kwargs):
            created["model"] = model
            created["kwargs"] = kwargs

    monkeypatch.setattr("gepa.lm.LM", FakeLM)

    lm = _squash_reflection_lm("openai/test")

    assert isinstance(lm, FakeLM)
    assert created["model"] == "openai/test"
    response_format = created["kwargs"]["response_format"]
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False
    assert schema["schema"]["required"] == [
        "instructions",
        "true_criteria",
        "false_criteria",
    ]


def test_multiclass_squash_model_requests_exact_class_schema(monkeypatch) -> None:
    created = {}

    class FakeLM:
        def __init__(self, _model, **kwargs):
            created["kwargs"] = kwargs

    monkeypatch.setattr("gepa.lm.LM", FakeLM)
    candidate = _multiclass_candidate()

    _squash_reflection_lm("openai/test", candidate)

    schema = created["kwargs"]["response_format"]["json_schema"]["schema"]
    assert schema["required"] == list(candidate.as_gepa())
    assert schema["additionalProperties"] is False


def test_squash_budget_funds_two_complete_two_by_two_pxn_levels() -> None:
    assert squash_metric_budget(60) == 1260


def test_working_set_hard_mines_uncertain_rows_and_mixes_remainder() -> None:
    predictions = [
        Prediction(story_id=f"row-{index}", probability=index / 10)
        for index in range(1, 10)
    ]

    selected = select_squash_working_set(
        predictions,
        uncertain_fraction=0.2,
        certain_mix_fraction=0.5,
        min_uncertain=2,
        max_uncertain=4,
        seed=7,
    )

    assert selected[:2] == ["row-5", "row-4"]
    assert len(selected) == 3
    assert selected[2] not in selected[:2]
    assert selected == select_squash_working_set(
        predictions,
        uncertain_fraction=0.2,
        certain_mix_fraction=0.5,
        min_uncertain=2,
        max_uncertain=4,
        seed=7,
    )


def test_squash_uses_every_pass_after_rejected_candidates(
    tmp_path, monkeypatch
) -> None:
    stories = _stories()
    backend = ConfidenceBackend()
    calls = []
    progress_events = []

    def fake_squash_candidate(**kwargs):
        calls.append(kwargs)
        proposed = _candidate("decisive")
        return OptimizationOutcome(
            candidate=proposed,
            best_score=0.9,
            total_metric_calls=12,
            metadata={},
        )

    monkeypatch.setattr("jev_align.squash.squash_candidate", fake_squash_candidate)

    report = run_squash(
        seed_candidate=_candidate(),
        stories=stories,
        backend=backend,
        reflection_model="openai/test",
        max_passes=3,
        uncertain_fraction=0.2,
        certain_mix_fraction=0.5,
        min_uncertain=2,
        max_uncertain=4,
        min_improvement=0.001,
        output_dir=tmp_path / "output",
        run_dir=tmp_path / "runs",
        concurrency=2,
        seed=3,
        evaluate_full=lambda candidate, _pass: backend.evaluate_many(
            candidate, stories
        ),
        report_path=tmp_path / "report.json",
        progress=lambda event, details: progress_events.append((event, details)),
    )

    assert report.final_candidate.instructions == "decisive"
    assert len(report.passes) == 3
    assert report.passes[0].promoted is True
    assert report.passes[0].metric_budget == 63
    assert report.passes[1].promoted is False
    assert report.passes[2].promoted is False
    assert report.stop_reason == (
        "maximum search passes reached; last candidate: "
        "GEPA returned an unchanged candidate"
    )
    assert report.final_ambiguity["mean"] < report.initial_ambiguity["mean"]
    assert report.final_prediction_flips == 5
    assert report.total_metric_calls == 36
    assert (tmp_path / "report.json").exists()
    assert all("label" not in example for call in calls for example in call["examples"])
    assert [event for event, _details in progress_events] == [
        "baseline",
        "pass_start",
        "gepa_complete",
        "full_pool_complete",
        "pass_start",
        "gepa_complete",
        "full_pool_complete",
        "pass_start",
        "gepa_complete",
        "full_pool_complete",
    ]
    baseline = progress_events[0][1]
    assert baseline["total_count"] == 10
    assert baseline["ambiguity"] == report.initial_ambiguity
    first_start = progress_events[1][1]
    assert first_start["uncertain_count"] == 2
    assert first_start["certain_count"] == 1
    assert first_start["working_ambiguity"]["count"] == 3
    assert first_start["metric_budget"] == 63
    first_gepa = progress_events[2][1]
    assert first_gepa["working_score_after"] == pytest.approx(0.9)
    assert first_gepa["metric_calls"] == 12
    assert progress_events[3][1]["pass_report"].promoted is True
    assert progress_events[6][1]["continuing"] is True
    assert progress_events[-1][1]["pass_report"].promoted is False
    assert progress_events[-1][1]["continuing"] is False
    assert progress_events[-1][1]["stop_reason"] == (
        "maximum search passes reached; last candidate: "
        "GEPA returned an unchanged candidate"
    )
    resumed = SquashRunReport.from_dict(report.as_dict())
    assert resumed.final_candidate == report.final_candidate
    assert resumed.final_predictions == []
    legacy = report.as_dict()
    for item in legacy["passes"]:
        item.pop("metric_budget")
    assert all(
        item.metric_budget is None
        for item in SquashRunReport.from_dict(legacy).passes
    )


def test_multiclass_squash_tracks_choice_flips_and_class_counts(
    tmp_path, monkeypatch
) -> None:
    stories = _stories(6)
    seed = _multiclass_candidate()
    proposed = _multiclass_candidate("Use decisive transport boundaries.")

    monkeypatch.setattr(
        "jev_align.squash.squash_candidate",
        lambda **_kwargs: OptimizationOutcome(
            candidate=proposed,
            best_score=0.95,
            total_metric_calls=15,
            metadata={},
        ),
    )

    def evaluate_full(candidate, _pass_number):
        decisive = candidate == proposed
        predictions = []
        for story in stories:
            if decisive:
                choice = "air"
                confidence = 0.95
            else:
                choice = "air" if story.row_number % 2 else "land"
                confidence = 0.55
            remainder = (1.0 - confidence) / 2
            predictions.append(
                Prediction(
                    story_id=story.id,
                    probabilities={
                        name: confidence if name == choice else remainder
                        for name in ("air", "land", "sea")
                    },
                    choice=choice,
                    confidence=confidence,
                )
            )
        return predictions

    report = run_squash(
        seed_candidate=seed,
        stories=stories,
        backend=ConfidenceBackend(),
        reflection_model="openai/test",
        max_passes=1,
        uncertain_fraction=0.5,
        certain_mix_fraction=0.0,
        min_uncertain=2,
        max_uncertain=4,
        min_improvement=0.0,
        output_dir=tmp_path / "output",
        run_dir=tmp_path / "runs",
        concurrency=2,
        seed=3,
        evaluate_full=evaluate_full,
    )

    assert report.final_candidate == proposed
    assert report.final_prediction_flips == 3
    assert report.initial_output_counts == {"air": 3, "land": 3, "sea": 0}
    assert report.final_output_counts == {"air": 6, "land": 0, "sea": 0}
    assert report.passes[0].previous_output_counts == report.initial_output_counts
    assert report.passes[0].proposed_output_counts == report.final_output_counts
    restored = SquashRunReport.from_dict(report.as_dict())
    assert isinstance(restored.final_candidate, MulticlassCandidateSpec)
    assert restored.final_output_counts == report.final_output_counts


@pytest.mark.parametrize("task_kind", ["multilabel", "score"])
def test_squash_tracks_multilabel_sets_and_rounded_score_levels(
    task_kind, tmp_path, monkeypatch
) -> None:
    stories = _stories(6)
    if task_kind == "multilabel":
        seed = _multilabel_candidate()
        proposed = _multilabel_candidate("Use decisive independent tag boundaries.")
        expected_initial = {"No labels": 0, "billing": 3, "urgent": 3}
        expected_final = {"No labels": 0, "billing": 0, "urgent": 6}
        expected_flips = 3
    else:
        seed = _score_candidate()
        proposed = _score_candidate("Use decisive ordered severity boundaries.")
        expected_initial = {"Score 0": 3, "Score 1": 0, "Score 2": 3}
        expected_final = {"Score 0": 0, "Score 1": 6, "Score 2": 0}
        expected_flips = 6

    monkeypatch.setattr(
        "jev_align.squash.squash_candidate",
        lambda **_kwargs: OptimizationOutcome(
            candidate=proposed,
            best_score=0.95,
            total_metric_calls=15,
            metadata={},
        ),
    )

    def evaluate_full(candidate, _pass_number):
        decisive = candidate == proposed
        predictions = []
        for story in stories:
            if task_kind == "multilabel":
                probabilities = (
                    {"urgent": 0.95, "billing": 0.05}
                    if decisive
                    else {
                        "urgent": 0.6 if story.row_number % 2 else 0.4,
                        "billing": 0.4 if story.row_number % 2 else 0.6,
                    }
                )
                predictions.append(
                    Prediction(
                        story_id=story.id,
                        label_probabilities=probabilities,
                    )
                )
            else:
                score = 1.0 if decisive else (0.4 if story.row_number % 2 else 1.6)
                confidence = 0.95 if decisive else 0.55
                predictions.append(
                    Prediction(
                        story_id=story.id,
                        score=score,
                        score_probabilities={0: 0.2, 1: 0.6, 2: 0.2},
                        confidence=confidence,
                    )
                )
        return predictions

    report = run_squash(
        seed_candidate=seed,
        stories=stories,
        backend=ConfidenceBackend(),
        reflection_model="openai/test",
        max_passes=1,
        uncertain_fraction=0.5,
        certain_mix_fraction=0.0,
        min_uncertain=2,
        max_uncertain=4,
        min_improvement=0.0,
        output_dir=tmp_path / f"{task_kind}-output",
        run_dir=tmp_path / f"{task_kind}-runs",
        concurrency=2,
        seed=3,
        evaluate_full=evaluate_full,
    )

    assert report.final_candidate == proposed
    assert report.final_prediction_flips == expected_flips
    assert report.initial_output_counts == expected_initial
    assert report.final_output_counts == expected_final
    restored = SquashRunReport.from_dict(report.as_dict())
    assert type(restored.final_candidate) is type(proposed)
    assert restored.final_output_counts == expected_final


def test_squash_promotes_any_measurable_full_pool_improvement_by_default(
    tmp_path, monkeypatch
) -> None:
    stories = _stories()
    proposed = _candidate("slightly better")

    monkeypatch.setattr(
        "jev_align.squash.squash_candidate",
        lambda **_kwargs: OptimizationOutcome(
            candidate=proposed,
            best_score=0.8,
            total_metric_calls=10,
            metadata={},
        ),
    )

    def evaluate_full(candidate, _pass_number):
        probability = 0.39975 if candidate == proposed else 0.4
        return [
            Prediction(story_id=story.id, probability=probability)
            for story in stories
        ]

    accepted = run_squash(
        seed_candidate=_candidate(),
        stories=stories,
        backend=ConfidenceBackend(),
        reflection_model="openai/test",
        max_passes=1,
        uncertain_fraction=0.2,
        certain_mix_fraction=0.5,
        min_uncertain=2,
        max_uncertain=4,
        min_improvement=0.0,
        output_dir=tmp_path / "accepted-output",
        run_dir=tmp_path / "accepted-runs",
        concurrency=2,
        seed=3,
        evaluate_full=evaluate_full,
    )
    rejected = run_squash(
        seed_candidate=_candidate(),
        stories=stories,
        backend=ConfidenceBackend(),
        reflection_model="openai/test",
        max_passes=1,
        uncertain_fraction=0.2,
        certain_mix_fraction=0.5,
        min_uncertain=2,
        max_uncertain=4,
        min_improvement=0.001,
        output_dir=tmp_path / "rejected-output",
        run_dir=tmp_path / "rejected-runs",
        concurrency=2,
        seed=3,
        evaluate_full=evaluate_full,
    )

    assert accepted.passes[0].promoted is True
    assert accepted.final_candidate == proposed
    assert rejected.passes[0].promoted is False
    assert rejected.final_candidate == _candidate()
    assert rejected.stop_reason == (
        "maximum search passes reached; last candidate: "
        "full-pool improvement below the configured minimum"
    )


def test_squash_rejects_mean_improvement_that_worsens_uncertainty_tail(
    tmp_path, monkeypatch
) -> None:
    stories = _stories(20)
    proposed = _candidate("sacrifices the tail")

    monkeypatch.setattr(
        "jev_align.squash.squash_candidate",
        lambda **_kwargs: OptimizationOutcome(
            candidate=proposed,
            best_score=0.9,
            total_metric_calls=10,
            metadata={},
        ),
    )

    def evaluate_full(candidate, _pass_number):
        if candidate == proposed:
            ambiguities = [0.9] * 10 + [0.0] * 10
        else:
            ambiguities = [0.8] * 10 + [0.2] * 10
        return [
            Prediction(story_id=story.id, probability=ambiguity / 2.0)
            for story, ambiguity in zip(stories, ambiguities, strict=True)
        ]

    report = run_squash(
        seed_candidate=_candidate(),
        stories=stories,
        backend=ConfidenceBackend(),
        reflection_model="openai/test",
        max_passes=1,
        uncertain_fraction=0.2,
        certain_mix_fraction=0.5,
        min_uncertain=2,
        max_uncertain=4,
        min_improvement=0.0,
        output_dir=tmp_path / "output",
        run_dir=tmp_path / "runs",
        concurrency=2,
        seed=3,
        evaluate_full=evaluate_full,
    )

    assert report.passes[0].previous_ambiguity["mean"] == pytest.approx(0.5)
    assert report.passes[0].proposed_ambiguity["mean"] == pytest.approx(0.45)
    assert report.passes[0].previous_tail_ambiguity == pytest.approx(0.8)
    assert report.passes[0].proposed_tail_ambiguity == pytest.approx(0.9)
    assert report.passes[0].promoted is False
    assert report.final_candidate == _candidate()
    assert report.stop_reason.endswith("worst-ten uncertainty tail did not improve")


def test_squash_stops_early_only_when_uncertainty_is_eliminated(
    tmp_path, monkeypatch
) -> None:
    stories = _stories()
    proposed = _candidate("fully certain")
    proposal_calls = 0

    def fake_squash_candidate(**_kwargs):
        nonlocal proposal_calls
        proposal_calls += 1
        return OptimizationOutcome(
            candidate=proposed,
            best_score=1.0,
            total_metric_calls=10,
            metadata={},
        )

    monkeypatch.setattr(
        "jev_align.squash.squash_candidate", fake_squash_candidate
    )

    def evaluate_full(candidate, _pass_number):
        probability = 0.0 if candidate == proposed else 0.4
        return [
            Prediction(story_id=story.id, probability=probability)
            for story in stories
        ]

    report = run_squash(
        seed_candidate=_candidate(),
        stories=stories,
        backend=ConfidenceBackend(),
        reflection_model="openai/test",
        max_passes=5,
        uncertain_fraction=0.2,
        certain_mix_fraction=0.5,
        min_uncertain=2,
        max_uncertain=4,
        min_improvement=0.0,
        output_dir=tmp_path / "output",
        run_dir=tmp_path / "runs",
        concurrency=2,
        seed=3,
        evaluate_full=evaluate_full,
    )

    assert proposal_calls == 1
    assert len(report.passes) == 1
    assert report.passes[0].promoted is True
    assert report.final_ambiguity["mean"] == 0.0
    assert report.stop_reason == "full-pool uncertainty eliminated"


def test_real_gepa_squash_optimizes_reported_certainty(tmp_path) -> None:
    stories = {story.id: story for story in _stories(5)}
    candidate = _candidate()
    proposal_number = 0

    def reflection_lm(_prompt: str) -> str:
        nonlocal proposal_number
        proposal_number += 1
        return json.dumps(
            {
                "instructions": f"decisive {proposal_number}",
                "true_criteria": "Return true decisively.",
                "false_criteria": "Return false decisively.",
            }
        )

    class ProposalSensitiveBackend:
        def evaluate_many(self, candidate, selected):
            if candidate.instructions == "seed":
                probability = 0.5
            else:
                proposal = int(candidate.instructions.rsplit(" ", 1)[-1])
                probability = 0.8 if proposal <= 4 else 0.99
            return [
                Prediction(story_id=story.id, probability=probability)
                for story in selected
            ]

    outcome = squash_candidate(
        seed_candidate=candidate,
        examples=[{"story_id": story_id} for story_id in stories],
        stories=stories,
        backend=ProposalSensitiveBackend(),
        reflection_model=reflection_lm,
        output_dir=tmp_path / "output",
        run_dir=tmp_path / "runs",
        pass_number=1,
        concurrency=2,
        seed=7,
    )

    assert outcome.candidate != candidate
    assert outcome.best_score == pytest.approx(0.98)
    assert 50 < outcome.total_metric_calls <= squash_metric_budget(len(stories))
    assert outcome.metadata["configured_metric_budget"] == 105
    assert outcome.metadata["pxn_levels"] == 2
    assert proposal_number == 8
    assert (tmp_path / "output" / "pass-0001" / "gepa-result.json").exists()


@pytest.mark.parametrize("task_kind", ["multilabel", "score"])
def test_real_gepa_squash_optimizes_multilabel_and_score_confidence(
    task_kind, tmp_path
) -> None:
    stories = {story.id: story for story in _stories(5)}
    candidate = (
        _multilabel_candidate()
        if task_kind == "multilabel"
        else _score_candidate()
    )
    proposal_number = 0

    def reflection_lm(_prompt: str) -> str:
        nonlocal proposal_number
        proposal_number += 1
        proposal = candidate.as_gepa()
        proposal["instructions"] = f"decisive {proposal_number}"
        return json.dumps(proposal)

    class ProposalSensitiveBackend:
        def evaluate_many(self, proposed, selected):
            decisive = proposed.instructions != candidate.instructions
            if task_kind == "multilabel":
                probabilities = (
                    {"urgent": 0.99, "billing": 0.01}
                    if decisive
                    else {"urgent": 0.5, "billing": 0.5}
                )
                return [
                    Prediction(
                        story_id=story.id,
                        label_probabilities=probabilities,
                    )
                    for story in selected
                ]
            confidence = 0.99 if decisive else 0.5
            return [
                Prediction(
                    story_id=story.id,
                    score=1.0,
                    score_probabilities={0: 0.01, 1: 0.98, 2: 0.01},
                    confidence=confidence,
                )
                for story in selected
            ]

    outcome = squash_candidate(
        seed_candidate=candidate,
        examples=[{"story_id": story_id} for story_id in stories],
        stories=stories,
        backend=ProposalSensitiveBackend(),
        reflection_model=reflection_lm,
        output_dir=tmp_path / f"{task_kind}-output",
        run_dir=tmp_path / f"{task_kind}-runs",
        pass_number=1,
        concurrency=2,
        seed=7,
    )

    assert type(outcome.candidate) is type(candidate)
    assert outcome.candidate != candidate
    assert outcome.best_score == pytest.approx(
        0.98 if task_kind == "multilabel" else 0.99
    )
    assert 50 < outcome.total_metric_calls <= squash_metric_budget(len(stories))
    assert proposal_number == 8
