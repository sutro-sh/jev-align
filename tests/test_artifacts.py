from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_align.artifacts import (
    ArtifactError,
    artifact_bytes,
    build_function_artifact,
    materialize_function_artifact,
)
from jev_align.data import load_stories, source_sha256
from jev_align.models import (
    BackendConfig,
    CandidateHistory,
    CandidateSpec,
    LabelRecord,
    RunState,
)
from jev_align.persistence import RunStore


def _run(tmp_path: Path) -> tuple[RunStore, RunState]:
    candidate = CandidateSpec(
        instructions="Is this about aviation?",
        true_criteria="Aircraft and flight.",
        false_criteria="Everything else.",
    )
    state = RunState(
        run_id="artifact",
        source_path=str(tmp_path / "missing.csv"),
        source_sha256="unused",
        pool_size=2,
        selected_columns=["title", "text"],
        reflection_model="openai/private-reflection-model",
        backend=BackendConfig(
            provider="typesafe",
            model="jev-model",
            options={"api_key": "must-not-publish"},
        ),
        current_candidate=candidate,
        history=[CandidateHistory(round_number=0, candidate=candidate, decision="seed")],
    )
    store = RunStore(tmp_path / "run")
    store.initialize(state)
    store.append_label(
        LabelRecord(
            story_id="row-1",
            inputs={"title": "New runway", "text": "Airport expands"},
            label=True,
            rationale="The runway is aviation infrastructure.",
            round_number=1,
            acquired_by="ambiguous",
            acquisition_probability=0.51,
            evaluation_split="train",
        )
    )
    store.append_label(
        LabelRecord(
            story_id="row-2",
            inputs={"title": "Sourdough", "text": "A recipe"},
            label=False,
            round_number=1,
            acquired_by="holdout",
            acquisition_probability=0.1,
            evaluation_split="holdout",
        )
    )
    return store, state


def test_artifact_is_portable_and_contains_only_labeled_inputs(tmp_path: Path) -> None:
    store, _state = _run(tmp_path)
    artifact = build_function_artifact(
        store.directory,
        name="Aviation classifier",
        description="Finds aviation-related posts.",
    )
    payload = json.loads(artifact_bytes(artifact))

    assert payload["slug"] == "aviation-classifier"
    assert payload["description"] == "Finds aviation-related posts."
    assert payload["backend"] == {"provider": "typesafe", "model": "jev-model", "options": {}}
    assert payload["annotations"][0]["inputs"]["title"] == "New runway"
    assert payload["annotations"][0]["rationale"].startswith("The runway")
    assert payload["annotations"][1]["split"] == "holdout"
    assert payload["learning"]["metric_budget"] == 300
    assert payload["learning"]["reflection_model"] == "openai/private-reflection-model"
    serialized = artifact_bytes(artifact).decode()
    assert "must-not-publish" not in serialized
    assert "source_path" not in serialized


def test_pending_proposal_must_be_resolved_before_publish(tmp_path: Path) -> None:
    store, state = _run(tmp_path)
    state.pending_candidate = CandidateSpec(
        instructions="Pending",
        true_criteria="Yes.",
        false_criteria="No.",
    )
    store.save_state(state)

    with pytest.raises(ArtifactError, match="pending proposal"):
        build_function_artifact(store.directory, name="Aviation classifier")


def test_materialized_artifact_is_a_normal_resumable_run(tmp_path: Path) -> None:
    source_store, _state = _run(tmp_path)
    artifact = build_function_artifact(source_store.directory, name="Aviation classifier")
    target = tmp_path / "pulled-run"

    materialize_function_artifact(
        artifact,
        target,
        reference="octocat/aviation-classifier",
        version=2,
        digest="verified-digest",
        registry="https://registry.example",
    )

    pulled = RunStore(target)
    state = pulled.load_state()
    labels = pulled.load_labels()
    source = target / "labeled-data.jsonl"
    stories = load_stories(source, state.pool_size, state.selected_columns or [])
    publication = json.loads((target / "published.json").read_text())

    assert state.current_candidate.model_dump(mode="json") == artifact.definition
    assert state.source_sha256 == source_sha256(source)
    assert [story.fields for story in stories] == [item.inputs for item in artifact.annotations]
    assert [label.label for label in labels] == [True, False]
    assert labels[1].evaluation_split == "holdout"
    assert state.holdout_story_ids == ["row-000002"]
    assert publication["reference"] == "octocat/aviation-classifier"
    assert publication["version"] == 2


def test_materialize_never_overwrites_existing_destination(tmp_path: Path) -> None:
    source_store, _state = _run(tmp_path)
    artifact = build_function_artifact(source_store.directory, name="Aviation classifier")
    target = tmp_path / "existing"
    target.mkdir()

    with pytest.raises(ArtifactError, match="already exists"):
        materialize_function_artifact(
            artifact,
            target,
            reference="octocat/aviation-classifier",
            version=1,
            digest="digest",
            registry="https://registry.example",
        )
