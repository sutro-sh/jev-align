from __future__ import annotations

from pathlib import Path

from jev_align.models import (
    BackendConfig,
    CandidateHistory,
    CandidateSpec,
    LabelRecord,
    RunState,
    ScoreCandidateSpec,
)
from jev_align.persistence import RunStore


def _state() -> RunState:
    candidate = CandidateSpec(
        instructions="Is it aviation?",
        true_criteria="Aviation.",
        false_criteria="Not aviation.",
    )
    return RunState(
        run_id="test",
        source_path="/tmp/stories.csv",
        source_sha256="abc",
        reflection_model="openai/test",
        current_candidate=candidate,
        history=[
            CandidateHistory(round_number=0, candidate=candidate, decision="seed")
        ],
    )


def test_run_store_round_trips_state_and_labels(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    state = _state()
    store.initialize(state)
    assert store.load_state() == state

    label = LabelRecord(
        story_id="row-1",
        label=True,
        rationale="mentions an aircraft",
        round_number=1,
        acquired_by="ambiguous",
        acquisition_probability=0.51,
        resolved_model="jev-1.13.0",
    )
    store.append_label(label)
    assert store.load_labels() == [label]
    assert store.pop_last_label(story_id="row-1", round_number=1) == label
    assert store.load_labels() == []


def test_legacy_jev_model_migrates_to_backend_configuration() -> None:
    candidate = CandidateSpec(
        instructions="Is it aviation?",
        true_criteria="Aviation.",
        false_criteria="Not aviation.",
    )
    state = RunState.model_validate(
        {
            "version": 1,
            "run_id": "legacy",
            "source_path": "/tmp/stories.csv",
            "source_sha256": "abc",
            "jev_model": "jev-legacy",
            "reflection_model": "openai/test",
            "current_candidate": candidate.model_dump(),
            "history": [
                {
                    "round_number": 0,
                    "candidate": candidate.model_dump(),
                    "decision": "seed",
                }
            ],
        }
    )

    assert state.version == 3
    assert state.backend == BackendConfig(provider="typesafe", model="jev-legacy")
    assert state.holdout_fraction == 0.0
    assert state.holdout_story_ids == []
    assert "jev_model" not in state.model_dump()


def test_run_store_round_trips_score_state_and_integer_label(tmp_path: Path) -> None:
    candidate = ScoreCandidateSpec(
        instructions="How severe is this?",
        levels=["Cosmetic", "Degraded", "Blocking"],
    )
    state = RunState(
        run_id="score",
        source_path="/tmp/stories.csv",
        source_sha256="abc",
        reflection_model="openai/test",
        current_candidate=candidate,
        history=[
            CandidateHistory(round_number=0, candidate=candidate, decision="seed")
        ],
    )
    store = RunStore(tmp_path / "score-run")
    store.initialize(state)
    store.append_label(
        LabelRecord(
            story_id="row-1",
            label=2,
            round_number=1,
            acquired_by="ambiguous",
            acquisition_probability=0.6,
        )
    )

    loaded = store.load_state()
    assert isinstance(loaded.current_candidate, ScoreCandidateSpec)
    assert loaded.current_candidate == candidate
    assert store.load_labels()[0].label == 2
    assert type(store.load_labels()[0].label) is int
