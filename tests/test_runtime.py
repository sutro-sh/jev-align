from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from jev_align import AIFunction, Capture
from jev_align import runtime
from jev_align.data import load_stories
from jev_align.jev import TypeSafeJevEvaluator
from jev_align.models import (
    CandidateHistory,
    CandidateSpec,
    Prediction,
    RunState,
    Story,
)
from jev_align.persistence import RunStore


@pytest.fixture
def saved_run(tmp_path):
    candidate = CandidateSpec(
        instructions="Is it aviation?",
        true_criteria="About aviation",
        false_criteria="Not about aviation",
    )
    state = RunState(
        run_id="aviation",
        source_path="no-longer-needed.csv",
        source_sha256="abc",
        selected_columns=["text"],
        reflection_model="openai/test",
        current_candidate=candidate,
        history=[
            CandidateHistory(round_number=1, candidate=candidate, decision="accepted")
        ],
        pending_candidate=candidate.model_copy(update={"instructions": "Pending"}),
    )
    store = RunStore(tmp_path / "run")
    store.initialize(state)
    return store


@pytest.fixture
def backend(monkeypatch):
    class FakeBackend:
        capabilities = TypeSafeJevEvaluator.capabilities

        def __init__(self):
            self.calls = []
            self.probability = 0.5
            self.error = None

        def evaluate_many(self, candidate, stories):
            self.calls.append((candidate, stories))
            if self.error:
                raise self.error
            return [
                Prediction(
                    story_id=story.id,
                    probability=self.probability,
                    resolved_model="resolved",
                )
                for story in stories
            ]

    fake = FakeBackend()
    monkeypatch.setattr(runtime, "create_backend", lambda *_args, **_kwargs: fake)
    return fake


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_runtime_pins_current_definition_and_preserves_run(
    saved_run, backend, tmp_path
):
    original = saved_run.state_path.read_bytes()
    with Capture(tmp_path / "captures", audit_rate=0) as capture:
        aviation = AIFunction.load(saved_run.directory, capture=capture)
        prediction = aviation(
            text="<p>Plane &amp; pilot</p>", ignored="private metadata"
        )

    assert prediction.probability == 0.5
    assert len(backend.calls) == 1
    assert backend.calls[0][0].instructions == "Is it aviation?"
    row = records(capture.path)[0]
    assert row["input"] == {"text": "Plane & pilot"}
    assert row["prediction"] == prediction.model_dump(mode="json")
    assert row["ambiguity"] == 1
    assert row["reason"] == "ambiguous"
    assert row["definition"]["candidate"]["instructions"] == "Is it aviation?"
    assert len(row["fingerprint"]) == 64
    assert saved_run.state_path.read_bytes() == original
    assert not saved_run.labels_path.exists()
    # The existing dataset loader can use inputs without exposing predictions.
    story = load_stories(capture.path, columns=["input"])[0]
    assert json.loads(story.fields["input"]) == row["input"]

    state = saved_run.load_state()
    state.current_candidate = state.pending_candidate
    saved_run.save_state(state)
    aviation(text="Second call")
    assert backend.calls[-1][0].instructions == "Is it aviation?"


def test_runtime_without_capture_and_concatenation(saved_run, backend):
    state = saved_run.load_state()
    state.column_mode = "all_concatenated"
    state.selected_columns = ["title", "text"]
    saved_run.save_state(state)

    aviation = AIFunction.load(saved_run.directory)
    assert aviation(text="Body", title="<b>Title</b>").probability == 0.5
    assert backend.calls[0][1][0].fields == {"content": "Title\nBody"}


def test_bad_inputs_and_backend_errors_do_not_create_captures(
    saved_run, backend, tmp_path
):
    with Capture(tmp_path / "captures") as capture:
        aviation = AIFunction.load(saved_run.directory, capture=capture)
        with pytest.raises(ValueError, match="missing input fields"):
            aviation(other="wrong")
        assert backend.calls == []
        backend.error = RuntimeError("JEV failed")
        with pytest.raises(RuntimeError, match="JEV failed"):
            aviation(text="Plane")
    assert records(capture.path) == []


def test_filter_and_audit(saved_run, backend, tmp_path):
    backend.probability = 0.99
    with Capture(tmp_path / "filtered", audit_rate=0) as capture:
        call = AIFunction.load(saved_run.directory, capture=capture)
        call(text="Plane")
    assert records(capture.path) == []
    with Capture(tmp_path / "audit", audit_rate=1) as capture:
        call = AIFunction.load(saved_run.directory, capture=capture)
        call(text="Plane")
    assert records(capture.path)[0]["reason"] == "audit"
    assert len(backend.calls) == 2


def test_full_queue_drops_capture_without_blocking_evaluation(
    saved_run, backend, tmp_path, monkeypatch
):
    release = Event()
    write_loop = Capture._write_loop

    def paused_writer(self):
        release.wait(5)
        write_loop(self)

    monkeypatch.setattr(Capture, "_write_loop", paused_writer)
    capture = Capture(tmp_path, queue_size=1)
    call = AIFunction.load(saved_run.directory, capture=capture)
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda text: call(text=text), ["Plane"] * 20))
        assert len(results) == len(backend.calls) == 20
        assert capture.stats["dropped"] == 19
    finally:
        release.set()
        capture.close()
    assert len(records(capture.path)) == 1
    assert capture.stats["written"] == 1


def test_write_failure_disables_capture_but_keeps_evaluating(
    saved_run, backend, tmp_path, caplog
):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("file")
    capture = Capture(blocked)
    capture._thread.join(timeout=2)
    call = AIFunction.load(saved_run.directory, capture=capture)
    assert call(text="Plane").probability == 0.5
    capture.close()
    assert capture.stats["error"]
    assert capture.stats["dropped"] == 1
    assert len(backend.calls) == 1
    assert "capture disabled" in caplog.text


def record(capture, text="Plane"):
    capture.record(
        Story(id="one", row_number=1, fields={"text": text}),
        Prediction(story_id="one", probability=0.5),
        run_id="test",
        definition={},
        fingerprint="test",
    )


def test_byte_limits_and_closed_capture(tmp_path):
    with Capture(tmp_path, max_record_bytes=100) as capture:
        record(capture, "x" * 101)
    assert capture.stats["dropped"] == 1
    assert capture.path.stat().st_size == 0

    with Capture(tmp_path, max_file_bytes=1200) as capture:
        for _ in range(20):
            record(capture)
    rows = records(capture.path)
    assert 0 < len(rows) < 20
    assert capture.path.stat().st_size <= 1200
    assert capture.stats["written"] + capture.stats["dropped"] == 20
    record(capture)
    assert capture.stats["written"] + capture.stats["dropped"] == 21


def test_concurrent_writers_use_separate_files(tmp_path):
    with Capture(tmp_path) as first, Capture(tmp_path) as second:
        record(first)
        record(second)
    assert first.path != second.path
    assert len(records(first.path)) == len(records(second.path)) == 1


def test_concurrent_calls_produce_complete_records(tmp_path):
    with Capture(tmp_path, queue_size=256) as capture:
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(lambda i: record(capture, str(i)), range(200)))
    rows = records(capture.path)
    assert {row["input"]["text"] for row in rows} == {str(i) for i in range(200)}
    assert capture.stats == {"written": 200, "dropped": 0, "error": None}


def test_ai_function_owns_default_capture(saved_run, backend, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    capture_directory = tmp_path / "shared-captures"
    monkeypatch.setenv("JEVA_CAPTURE_DIR", str(capture_directory))
    aviation = AIFunction.load(saved_run.directory, capture=True)

    assert isinstance(aviation.capture, Capture)
    capture_path = aviation.capture.path
    assert capture_path.parent == capture_directory
    assert aviation(text="Plane").probability == 0.5
    aviation.close()
    aviation.close()

    assert records(capture_path)[0]["input"] == {"text": "Plane"}


@pytest.mark.parametrize(
    "values",
    [
        {"choice": "A", "confidence": 0.1, "probabilities": {"A": 0.6, "B": 0.4}},
        {"score": 0.5, "confidence": 0.1, "score_probabilities": {0: 0.5, 1: 0.5}},
        {"label_probabilities": {"A": 0.99, "B": 0.5}},
    ],
)
def test_capture_uses_task_specific_uncertainty(tmp_path, values):
    with Capture(tmp_path, audit_rate=0) as capture:
        capture.record(
            Story(id="one", row_number=1, fields={"text": "Plane"}),
            Prediction(story_id="one", **values),
            run_id="test",
            definition={},
            fingerprint="test",
        )
    assert records(capture.path)[0]["reason"] == "ambiguous"


def test_invalid_runtime_capture_option(saved_run, backend):
    with pytest.raises(TypeError, match="True, False, or a Capture"):
        AIFunction.load(saved_run.directory, capture="yes")


@pytest.mark.parametrize(
    "options",
    [
        {"audit_rate": -1},
        {"ambiguity_threshold": 2},
        {"queue_size": 0},
        {"max_record_bytes": 0},
        {"max_file_bytes": 0},
    ],
)
def test_invalid_capture_options(tmp_path, options):
    with pytest.raises(ValueError):
        Capture(tmp_path, **options)
