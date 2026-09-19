from __future__ import annotations

from collections.abc import Sequence

from jev_align.models import (
    CandidateHistory,
    CandidateSpec,
    MulticlassCandidateSpec,
    MulticlassMetrics,
    MultilabelCandidateSpec,
    MultilabelCriteria,
    MultilabelMetrics,
    Prediction,
    RunState,
    Story,
)
from jev_align.optimizer import OptimizationOutcome
from jev_align.persistence import RunStore
from jev_align.session import ClimbSession


class CandidateAwareJev:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate_many(
        self, candidate: CandidateSpec, stories: Sequence[Story]
    ) -> list[Prediction]:
        self.calls += 1
        if candidate.instructions == "seed":
            values = [0.50, 0.49, 0.52, 0.45, 0.1, 0.9]
        else:
            values = [0.9, 0.1, 0.8, 0.2, 0.1, 0.9]
        return [
            Prediction(story_id=story.id, probability=values[story.row_number - 1])
            for story in stories
        ]


def test_mocked_round_persists_pending_diff_and_accepts(tmp_path, monkeypatch) -> None:
    seed = CandidateSpec(
        instructions="seed",
        true_criteria="yes boundary",
        false_criteria="no boundary",
    )
    proposed = CandidateSpec(
        instructions="improved",
        true_criteria="clear yes boundary",
        false_criteria="clear no boundary",
    )
    state = RunState(
        run_id="mock",
        source_path="/tmp/mock.csv",
        source_sha256="hash",
        pool_size=6,
        reflection_model="openai/mock",
        current_candidate=seed,
        history=[CandidateHistory(round_number=0, candidate=seed, decision="seed")],
    )
    stories = [
        Story(id=f"row-{index}", row_number=index, fields={"title": f"Story {index}"})
        for index in range(1, 7)
    ]
    store = RunStore(tmp_path / "run")
    store.initialize(state)
    jev = CandidateAwareJev()
    session = ClimbSession(state, stories, store, jev)

    acquisitions, pool_predictions = session.acquire()
    assert len(pool_predictions) == 6
    for index, acquisition in enumerate(acquisitions):
        session.add_label(
            story_id=acquisition.prediction.story_id,
            label=index % 2 == 0,
            rationale=f"reason {index}",
            acquired_by=acquisition.source,
            probability=acquisition.prediction.probability,
            resolved_model=acquisition.prediction.resolved_model,
        )

    def fake_optimize_candidate(**kwargs):
        assert len(kwargs["examples"]) == 5
        assert all("rationale" in example for example in kwargs["examples"])
        return OptimizationOutcome(
            candidate=proposed,
            best_score=1.0,
            total_metric_calls=304,
            metadata={},
        )

    monkeypatch.setattr("jev_align.session.optimize_candidate", fake_optimize_candidate)
    report = session.optimize(pool_predictions)
    assert report.proposed_candidate == proposed
    assert store.load_state().pending_candidate == proposed
    assert (store.directory / "round-0001-report.json").exists()
    assert report.total_metric_calls == 304
    assert report.ambiguity_scope == "fixed_full_pool"
    assert report.previous_ambiguity["count"] == 6
    assert report.proposed_ambiguity["count"] == 6
    assert report.previous_replay_ambiguity is not None
    assert report.previous_replay_ambiguity["count"] == 5
    assert report.proposed_replay_ambiguity is not None
    assert report.proposed_replay_ambiguity["count"] == 5

    session.decide("accept")
    saved = store.load_state()
    assert saved.current_candidate == proposed
    assert saved.pending_candidate is None
    assert saved.round_number == 2
    assert saved.history[-1].decision == "accepted"
    calls_before_cache_read = jev.calls
    cached = session.current_pool_predictions()
    assert len(cached) == 6
    assert jev.calls == calls_before_cache_read

    resumed_jev = CandidateAwareJev()
    resumed = ClimbSession(store.load_state(), stories, store, resumed_jev)
    assert len(resumed.current_pool_predictions()) == 6
    assert resumed_jev.calls == 0


def test_holdout_labels_are_added_separately_and_never_sent_to_gepa(
    tmp_path, monkeypatch
) -> None:
    candidate = CandidateSpec(
        instructions="seed",
        true_criteria="odd row",
        false_criteria="even row",
    )
    state = RunState(
        run_id="holdout",
        source_path="/tmp/mock.csv",
        source_sha256="hash",
        pool_size=8,
        batch_size=5,
        holdout_fraction=0.2,
        holdout_story_ids=["row-7", "row-8"],
        reflection_model="openai/mock",
        current_candidate=candidate,
        history=[CandidateHistory(round_number=0, candidate=candidate, decision="seed")],
    )
    stories = [
        Story(id=f"row-{index}", row_number=index, fields={"text": str(index)})
        for index in range(1, 9)
    ]

    class ParityJev:
        def evaluate_many(self, _candidate, selected_stories):
            return [
                Prediction(
                    story_id=story.id,
                    probability=0.9 if story.row_number % 2 else 0.1,
                )
                for story in selected_stories
            ]

    store = RunStore(tmp_path / "holdout-run")
    store.initialize(state)
    session = ClimbSession(state, stories, store, ParityJev())

    acquisitions, pool_predictions = session.acquire()
    holdout_acquisitions = session.acquire_holdout(pool_predictions)
    assert len(acquisitions) == 5
    assert len(holdout_acquisitions) == 1
    assert all(item.prediction.story_id not in state.holdout_story_ids for item in acquisitions)
    assert holdout_acquisitions[0].prediction.story_id in state.holdout_story_ids

    for acquisition in acquisitions:
        probability = acquisition.prediction.probability or 0.0
        session.add_label(
            story_id=acquisition.prediction.story_id,
            label=probability >= 0.5,
            rationale=None,
            acquired_by=acquisition.source,
            probability=probability,
        )
    heldout = holdout_acquisitions[0]
    heldout_probability = heldout.prediction.probability or 0.0
    session.add_label(
        story_id=heldout.prediction.story_id,
        label=heldout_probability >= 0.5,
        rationale="independent check",
        acquired_by="holdout",
        evaluation_split="holdout",
        probability=heldout_probability,
    )

    def fake_optimize_candidate(**kwargs):
        example_ids = {item["story_id"] for item in kwargs["examples"]}
        assert len(example_ids) == 5
        assert example_ids.isdisjoint(state.holdout_story_ids)
        return OptimizationOutcome(
            candidate=candidate,
            best_score=1.0,
            total_metric_calls=1,
            metadata={},
        )

    monkeypatch.setattr("jev_align.session.optimize_candidate", fake_optimize_candidate)
    report = session.optimize(pool_predictions)

    labels = store.load_labels()
    assert sum(item.evaluation_split == "train" for item in labels) == 5
    assert sum(item.evaluation_split == "holdout" for item in labels) == 1
    assert report.previous_holdout_metrics is not None
    assert report.proposed_holdout_metrics is not None
    assert sum(
        (
            report.previous_holdout_metrics.tp,
            report.previous_holdout_metrics.fp,
            report.previous_holdout_metrics.fn,
            report.previous_holdout_metrics.tn,
        )
    ) == 1
    session.decide("accept")
    assert store.load_state().history[-1].holdout_score is not None


def test_rewind_restores_candidate_and_reopens_previous_labeling_round(
    tmp_path,
) -> None:
    seed = CandidateSpec(
        instructions="seed",
        true_criteria="seed yes",
        false_criteria="seed no",
    )
    round_one = CandidateSpec(
        instructions="round one",
        true_criteria="round one yes",
        false_criteria="round one no",
    )
    round_two = CandidateSpec(
        instructions="round two",
        true_criteria="round two yes",
        false_criteria="round two no",
    )
    state = RunState(
        run_id="rewind",
        source_path="/tmp/mock.csv",
        source_sha256="hash",
        pool_size=6,
        reflection_model="openai/mock",
        current_candidate=round_two,
        round_number=3,
        history=[
            CandidateHistory(round_number=0, candidate=seed, decision="seed"),
            CandidateHistory(round_number=1, candidate=round_one, decision="accepted"),
            CandidateHistory(round_number=2, candidate=round_two, decision="accepted"),
        ],
    )
    stories = [
        Story(id=f"row-{index}", row_number=index, fields={"text": str(index)})
        for index in range(1, 7)
    ]
    store = RunStore(tmp_path / "rewind-run")
    store.initialize(state)
    session = ClimbSession(state, stories, store, CandidateAwareJev())
    for round_number, story_id in ((1, "row-1"), (2, "row-2")):
        state.round_number = round_number
        session.add_label(
            story_id=story_id,
            label=True,
            rationale=None,
            acquired_by="ambiguous",
            probability=0.5,
        )
    state.round_number = 3
    store.save_state(state)
    store.write_json("round-0002-report.json", {"old": True})
    (store.gepa_output_dir / "round-0002").mkdir()
    (store.gepa_run_dir / "round-0002").mkdir()
    store.write_json("pool-predictions-current.json", {"old": True})

    target_round, archive = session.rewind_previous_round()

    assert target_round == 2
    assert state.round_number == 2
    assert state.current_candidate == round_one
    assert [item.round_number for item in state.history] == [0, 1]
    assert [item.story_id for item in store.load_labels()] == ["row-1"]
    assert not (store.directory / "round-0002-report.json").exists()
    assert (archive / "round-0002-report.json").exists()
    assert (archive / "gepa-output" / "round-0002").is_dir()
    assert (archive / "gepa-runs" / "round-0002").is_dir()
    assert (archive / "pool-predictions-current.json").exists()


class MulticlassJev:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate_many(self, candidate, stories):
        self.calls += 1
        labels = list(candidate.criteria)
        predictions = []
        for story in stories:
            choice = labels[(story.row_number - 1) % len(labels)]
            confidence = 0.1 + 0.1 * (story.row_number - 1)
            remainder = (1.0 - (0.5 + confidence / 2)) / (len(labels) - 1)
            top = 0.5 + confidence / 2
            predictions.append(
                Prediction(
                    story_id=story.id,
                    choice=choice,
                    confidence=confidence,
                    probabilities={
                        name: top if name == choice else remainder for name in labels
                    },
                )
            )
        return predictions


def test_mocked_multiclass_round_acquires_optimizes_and_persists(
    tmp_path, monkeypatch
) -> None:
    seed = MulticlassCandidateSpec(
        instructions="What kind of post is this?",
        criteria={
            "aviation": "Aircraft and atmospheric flight",
            "space": "Spacecraft and astronomy",
            "other": "Everything else",
        },
    )
    proposed = MulticlassCandidateSpec(
        instructions="Classify the post into exactly one topic.",
        criteria={
            "aviation": "Primarily about aircraft and atmospheric flight",
            "space": "Primarily about spacecraft, rockets, or astronomy",
            "other": "Neither aviation nor space",
        },
    )
    state = RunState(
        run_id="multiclass",
        source_path="/tmp/mock.csv",
        source_sha256="hash",
        pool_size=6,
        reflection_model="openai/mock",
        current_candidate=seed,
        history=[CandidateHistory(round_number=0, candidate=seed, decision="seed")],
    )
    stories = [
        Story(id=f"row-{index}", row_number=index, fields={"title": f"Story {index}"})
        for index in range(1, 7)
    ]
    store = RunStore(tmp_path / "multiclass-run")
    store.initialize(state)
    jev = MulticlassJev()
    session = ClimbSession(state, stories, store, jev)
    acquisitions, pool_predictions = session.acquire()
    for acquisition in acquisitions:
        story = session.story_by_id[acquisition.prediction.story_id]
        label = list(seed.criteria)[(story.row_number - 1) % 3]
        session.add_label(
            story_id=story.id,
            label=label,
            rationale=f"belongs to {label}",
            acquired_by=acquisition.source,
            probability=max(acquisition.prediction.probabilities.values()),
        )
    assert session.ready_to_optimize()

    monkeypatch.setattr(
        "jev_align.session.optimize_candidate",
        lambda **kwargs: OptimizationOutcome(
            candidate=proposed,
            best_score=1.0,
            total_metric_calls=300,
            metadata={},
        ),
    )
    report = session.optimize(pool_predictions)
    assert isinstance(report.previous_metrics, MulticlassMetrics)
    assert report.previous_metrics.macro_f1 == 1.0
    assert report.proposed_candidate == proposed
    session.decide("accept")
    saved = store.load_state()
    assert saved.current_candidate == proposed
    assert saved.history[-1].fit_f1 == 1.0


class MultilabelJev:
    def evaluate_many(self, candidate, stories):
        predictions = []
        for story in stories:
            selected = {"tech"} if story.row_number % 2 else {"startups"}
            predictions.append(
                Prediction(
                    story_id=story.id,
                    label_probabilities={
                        name: 0.9 if name in selected else 0.1
                        for name in candidate.labels
                    },
                )
            )
        return predictions


def test_mocked_multilabel_round_does_not_require_every_label(
    tmp_path, monkeypatch
) -> None:
    criteria = {
        name: MultilabelCriteria(
            true_criteria=f"About {name}", false_criteria=f"Not about {name}"
        )
        for name in ("tech", "startups", "culture")
    }
    seed = MultilabelCandidateSpec(instructions="Apply all topics.", labels=criteria)
    proposed = MultilabelCandidateSpec(
        instructions="Apply every relevant topic.", labels=criteria
    )
    state = RunState(
        run_id="multilabel",
        source_path="/tmp/mock.csv",
        source_sha256="hash",
        pool_size=6,
        batch_size=5,
        reflection_model="openai/mock",
        current_candidate=seed,
        history=[CandidateHistory(round_number=0, candidate=seed, decision="seed")],
    )
    stories = [
        Story(id=f"row-{index}", row_number=index, fields={"title": f"Story {index}"})
        for index in range(1, 7)
    ]
    store = RunStore(tmp_path / "multilabel-run")
    store.initialize(state)
    session = ClimbSession(state, stories, store, MultilabelJev())
    acquisitions, pool_predictions = session.acquire()
    for acquisition in acquisitions:
        story = session.story_by_id[acquisition.prediction.story_id]
        labels = ["tech"] if story.row_number % 2 else ["startups"]
        session.add_label(
            story_id=story.id,
            label=labels,
            rationale=None,
            acquired_by=acquisition.source,
            probability=0.9,
        )

    # No row uses "culture"; a missing positive label must not block GEPA.
    assert session.ready_to_optimize()
    monkeypatch.setattr(
        "jev_align.session.optimize_candidate",
        lambda **kwargs: OptimizationOutcome(
            candidate=proposed,
            best_score=1.0,
            total_metric_calls=20,
            metadata={},
        ),
    )
    report = session.optimize(pool_predictions)
    assert isinstance(report.previous_metrics, MultilabelMetrics)
    assert report.previous_metrics.micro_f1 == 1.0
    assert report.previous_metrics.per_label["culture"].support == 0
    session.decide("accept")
    saved = store.load_state()
    assert saved.current_candidate == proposed
    assert saved.history[-1].fit_f1 == 1.0
