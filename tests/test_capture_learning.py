from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from jev_align import cli
from jev_align.captured_inputs import captured_story_id, sample_captured_inputs
from jev_align.data import load_stories, source_sha256
from jev_align.jev import TypeSafeJevEvaluator
from jev_align.models import (
    CandidateHistory,
    CandidateSpec,
    Prediction,
    RunState,
    Story,
)
from jev_align.optimizer import OptimizationOutcome
from jev_align.persistence import RunStore
from jev_align.session import ClimbSession, RoundReport


class FakeBackend:
    capabilities = TypeSafeJevEvaluator.capabilities

    def __init__(self):
        self.calls = []

    def evaluate_many(self, candidate, stories):
        self.calls.append((candidate, list(stories)))
        return [Prediction(story_id=story.id, probability=0.5) for story in stories]


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("JEVA_CAPTURE_DIR", raising=False)
    source = tmp_path / "data.csv"
    source.write_text("text\none\ntwo\nthree\nfour\nfive\nsix\n")
    stories = load_stories(source)
    candidate = CandidateSpec(
        instructions="seed", true_criteria="yes", false_criteria="no"
    )
    state = RunState(
        run_id="test-run",
        source_path=str(source),
        source_sha256=source_sha256(source),
        selected_columns=["text"],
        pool_size=6,
        batch_size=2,
        holdout_fraction=0.2,
        holdout_story_ids=[stories[-1].id],
        reflection_model="openai/test",
        current_candidate=candidate,
        history=[
            CandidateHistory(round_number=0, candidate=candidate, decision="seed")
        ],
    )
    store = RunStore(tmp_path / ".jev-align" / "runs" / state.run_id)
    store.initialize(state)
    return ClimbSession(state, stories, store, FakeBackend())


def write_captures(directory, *texts, run_id="test-run"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "calls.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "version": 1,
                    "run_id": run_id,
                    "input": {"text": text},
                    # Stale predictions must never be used as either labels or current scores.
                    "prediction": {"probability": 0.99},
                }
            )
            + "\n"
            for text in texts
        )
    )
    return path


def scan(session, directories, limit=None):
    return sample_captured_inputs(
        directories, session.state, excluded=session.stories, limit=limit
    )


def test_scan_matches_run_deduplicates_and_excludes_original_holdout(session, tmp_path):
    directory = tmp_path / "captures"
    path = write_captures(directory, "new", "new", "six", "one")
    write_captures(directory / "other", "wrong run", run_id="other-run")
    with path.open("a") as handle:
        handle.write(
            'not json\n{"version":1,"run_id":"test-run","input":{"text":42}}\n'
        )
        handle.write('{"version":99,"run_id":"test-run","input":{"text":"future"}}\n')
        handle.write('{"version":1,"run_id":"test-run","input":{"wrong":"column"}}\n')
        handle.write('{"version":1,"run_id":"test-run","input":{"text":"partial"}}')
    captures = scan(session, [directory, directory / "other"])
    assert [story.fields for story in captures] == [{"text": "new"}]
    assert captures[0].id == captured_story_id({"text": "new"})
    assert session.backend.calls == []


def test_scan_is_bounded_and_duplicate_frequency_does_not_bias_sample(
    session, tmp_path
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    values = [f"captured {i}" for i in range(100)]
    write_captures(first, *values)
    write_captures(second, *reversed(values), *([values[0]] * 100))
    assert scan(session, [first], limit=7) == scan(session, [second], limit=7)
    assert len(scan(session, [first, second], limit=7)) == 7


def test_scan_imports_every_unique_eligible_capture_by_default(session, tmp_path):
    directory = tmp_path / "captures"
    write_captures(directory, *(f"captured {index}" for index in range(1001)))

    assert len(scan(session, [directory])) == 1001


def test_declining_captures_preserves_run_and_makes_no_calls(
    session, tmp_path, monkeypatch
):
    write_captures(tmp_path / ".jev-align" / "captures", "new")
    before = session.store.state_path.read_bytes()
    prompts = []

    def decline(prompt, options):
        prompts.append(prompt)
        return "n"

    monkeypatch.setattr(cli, "_select_option", decline)
    cli._offer_captured_calls(session)
    assert "captured calls" in prompts[0]
    assert session.capture_pool is None
    assert not session.store.captured_inputs_path.exists()
    assert session.store.state_path.read_bytes() == before
    assert session.backend.calls == []


def test_accept_snapshots_and_resume_keeps_full_pool_without_original_log(
    session, tmp_path, monkeypatch
):
    path = write_captures(tmp_path / ".jev-align" / "captures", "new A", "new B")
    monkeypatch.setattr(cli, "_select_option", lambda *_args: "y")
    cli._offer_captured_calls(session)
    assert len(session.capture_pool) == 2
    assert session.backend.calls == []
    assert not session.store.load_labels()
    story = session.capture_pool[0]
    session.add_label(
        story_id=story.id,
        label=True,
        rationale="human explanation",
        acquired_by="ambiguous",
        probability=0.5,
    )
    path.unlink()
    resumed = ClimbSession(
        session.store.load_state(), session.stories, session.store, FakeBackend()
    )
    cli._offer_captured_calls(resumed)
    assert len(resumed.capture_pool) == 2
    assert story.id in {item.id for item in resumed.capture_pool}
    assert story.id in resumed.story_by_id


def test_required_labeled_captures_do_not_count_toward_sample_limit(
    session, tmp_path
):
    required = [
        Story(
            id=captured_story_id({"text": f"labeled {index}"}),
            row_number=0,
            fields={"text": f"labeled {index}"},
        )
        for index in range(3)
    ]
    directory = tmp_path / "captures"
    write_captures(directory, *(f"unlabeled {index}" for index in range(10)))

    sampled = sample_captured_inputs(
        [directory],
        session.state,
        excluded=session.stories,
        required=required,
        limit=2,
    )

    assert len(sampled) == 5
    assert {story.id for story in required}.issubset(story.id for story in sampled)


def test_no_eligible_captures_or_pending_proposal_means_no_prompt(
    session, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        cli, "_select_option", lambda *_args: pytest.fail("unexpected prompt")
    )
    cli._offer_captured_calls(session)
    write_captures(tmp_path / ".jev-align" / "captures", "six", "one")
    cli._offer_captured_calls(session)
    write_captures(tmp_path / ".jev-align" / "captures", "new")
    session.state.pending_candidate = session.state.current_candidate
    cli._offer_captured_calls(session)


def test_run_workspace_and_custom_capture_directory_discovery(
    session, tmp_path, monkeypatch
):
    write_captures(tmp_path / ".jev-align" / "captures", "workspace capture")
    write_captures(tmp_path / "elsewhere", "custom capture")
    monkeypatch.setenv("JEVA_CAPTURE_DIR", str(tmp_path / "elsewhere"))
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    monkeypatch.setattr(cli, "_select_option", lambda *_args: "y")
    cli._offer_captured_calls(session)
    assert {story.fields["text"] for story in session.capture_pool} == {
        "workspace capture",
        "custom capture",
    }


def test_captured_round_rechecks_predictions_and_keeps_fixed_pool_and_holdout(
    session, tmp_path, monkeypatch
):
    path = write_captures(tmp_path / "captures", "new A", "new B", "new C")
    session.use_captured_inputs(scan(session, [path.parent]))
    acquisitions, pool = session.acquire()
    assert len(acquisitions) == 2
    assert len(pool) == 6
    assert all(item.prediction.probability == 0.5 for item in acquisitions)
    assert all(item.prediction.story_id.startswith("capture-") for item in acquisitions)
    for i, acquisition in enumerate(acquisitions):
        session.add_label(
            story_id=acquisition.prediction.story_id,
            label=bool(i),
            rationale="user supplied",
            acquired_by=acquisition.source,
            probability=0.5,
        )
    heldout = session.acquire_holdout(pool)[0]
    session.add_label(
        story_id=heldout.prediction.story_id,
        label=True,
        rationale="holdout secret",
        acquired_by="holdout",
        evaluation_split="holdout",
        probability=0.5,
    )

    def optimize(**kwargs):
        assert len(kwargs["examples"]) == 2
        assert all(item["rationale"] == "user supplied" for item in kwargs["examples"])
        assert all(
            item["story_id"].startswith("capture-") for item in kwargs["examples"]
        )
        return OptimizationOutcome(
            candidate=session.state.current_candidate,
            best_score=1,
            total_metric_calls=1,
            metadata={},
        )

    monkeypatch.setattr("jev_align.session.optimize_candidate", optimize)
    report = session.optimize(pool)
    assert report.previous_ambiguity["count"] == report.proposed_ambiguity["count"] == 6
    assert report.previous_replay_ambiguity["count"] == 2
    assert report.proposed_replay_ambiguity["count"] == 2
    assert report.previous_holdout_metrics is not None
    session.decide("accept")
    calls_before = len(session.backend.calls)
    assert session.current_pool_predictions() == pool
    assert len(session.backend.calls) == calls_before
    last_batch, _ = session.acquire()
    assert len(last_batch) == 1  # A final small batch remains labelable.
    assert last_batch[0].prediction.story_id not in {
        label.story_id for label in session.store.load_labels()
    }
    _, archive = session.rewind_previous_round()
    assert len(session.unlabeled_stories()) == 3
    assert (archive / "captured-inputs.json").exists()
    assert (archive / "capture-predictions-current.json").exists()


def test_function_card_resume_offers_captures_and_preserves_pending_decision(
    session, tmp_path, monkeypatch
):
    write_captures(tmp_path / ".jev-align" / "captures", "new A", "new B")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(
        cli, "create_backend", lambda *_args, **_kwargs: session.backend
    )
    answers = iter([True, False, True])  # Two training labels and one holdout label.
    monkeypatch.setattr(cli, "_prompt_label", lambda *_args, **_kwargs: next(answers))
    monkeypatch.setattr(cli, "_prompt_rationale", lambda: "human rationale")
    monkeypatch.setattr(
        "jev_align.session.optimize_candidate",
        lambda **_kwargs: OptimizationOutcome(
            candidate=session.state.current_candidate,
            best_score=1,
            total_metric_calls=1,
            metadata={},
        ),
    )
    result = CliRunner().invoke(cli.app, ["functions"], input="1\nc\ny\nq\n")
    assert result.exit_code == 0, result.output
    assert "Resume learning" in result.output
    assert "You have 2 new captured calls available" in result.output
    assert "2 inputs in the full captured pool (2 unlabeled)" in result.output
    assert "Captured-pool ambiguity" in result.output
    assert len(session.store.load_labels()) == 3
    assert session.store.load_state().pending_candidate is not None

    monkeypatch.setattr(
        cli, "_offer_captured_calls", lambda *_args: pytest.fail("decision comes first")
    )
    calls_before = len(session.backend.calls)
    resumed = CliRunner().invoke(
        cli.app, ["optimize", "--resume", str(session.store.directory)], input="q\n"
    )
    assert resumed.exit_code == 0, resumed.output
    assert "decision pending" in resumed.output
    assert len(session.backend.calls) == calls_before


def test_old_run_without_supplemental_file_and_unknown_version(session):
    assert session.store.load_captured_inputs() == []
    session.store.captured_inputs_path.write_text('{"version": 99, "stories": []}')
    with pytest.raises(ValueError, match="unsupported captured-inputs version"):
        session.store.load_captured_inputs()


def test_full_capture_pool_is_rescored_and_reused_after_five_labels(
    session, monkeypatch
):
    session.state.batch_size = 5
    captures = [
        Story(
            id=captured_story_id({"text": f"captured {index}"}),
            row_number=0,
            fields={"text": f"captured {index}"},
        )
        for index in range(1000)
    ]
    session.use_captured_inputs(captures)
    selected, original_pool = session.acquire()
    assert len(session.backend.calls[-1][1]) == 1000
    labeled_ids = {item.prediction.story_id for item in selected}
    for index, item in enumerate(selected):
        session.add_label(
            story_id=item.prediction.story_id,
            label=bool(index % 2),
            rationale=None,
            acquired_by=item.source,
            probability=0.5,
        )
    proposed = session.state.current_candidate.model_copy(
        update={"instructions": "improved"}
    )
    monkeypatch.setattr(
        "jev_align.session.optimize_candidate",
        lambda **_kwargs: OptimizationOutcome(
            candidate=proposed, best_score=1, total_metric_calls=1, metadata={}
        ),
    )
    report = session.optimize(original_pool)
    assert len(session.backend.calls[-1][1]) == 1000
    assert {story.id for story in session.backend.calls[-1][1]} == {
        story.id for story in captures
    }
    assert session.backend.calls[-1][0] == proposed
    assert report.previous_capture_ambiguity["count"] == 1000
    assert report.proposed_capture_ambiguity["count"] == 1000
    assert report.previous_ambiguity["count"] == report.proposed_ambiguity["count"] == 6
    assert RoundReport.from_dict(report.as_dict()) == report
    legacy_report = report.as_dict()
    del legacy_report["previous_capture_ambiguity"]
    del legacy_report["proposed_capture_ambiguity"]
    assert RoundReport.from_dict(legacy_report).previous_capture_ambiguity is None

    # Accept after restarting at the decision screen, before reselecting inputs.
    resumed = ClimbSession(
        session.store.load_state(), session.stories, session.store, session.backend
    )
    resumed.decide("accept")
    resumed.use_captured_inputs(captures)
    calls_before = len(session.backend.calls)
    next_batch, _ = resumed.acquire()
    assert len(resumed.unlabeled_stories()) == 995
    assert len(resumed.current_pool_predictions(captured=True)) == 1000
    assert len(next_batch) == 5
    assert labeled_ids.isdisjoint(item.prediction.story_id for item in next_batch)
    assert (
        len(session.backend.calls) == calls_before
    )  # Accepted full-pool cache reused.


def test_rejected_capture_candidate_preserves_current_cache(
    session, tmp_path, monkeypatch
):
    path = write_captures(tmp_path / "captures", "new A", "new B", "new C")
    session.use_captured_inputs(scan(session, [path.parent]))
    selected, original_pool = session.acquire()
    current = session._pool_cache_path("current", captured=True).read_bytes()
    for index, item in enumerate(selected):
        session.add_label(
            story_id=item.prediction.story_id,
            label=bool(index),
            rationale=None,
            acquired_by=item.source,
            probability=0.5,
        )
    proposed = session.state.current_candidate.model_copy(
        update={"instructions": "rejected"}
    )
    monkeypatch.setattr(
        "jev_align.session.optimize_candidate",
        lambda **_kwargs: OptimizationOutcome(
            candidate=proposed, best_score=1, total_metric_calls=1, metadata={}
        ),
    )
    session.optimize(original_pool)
    session.decide("reject")
    assert session._pool_cache_path("current", captured=True).read_bytes() == current
    calls_before = len(session.backend.calls)
    next_batch, _ = session.acquire()
    assert len(next_batch) == 1
    assert len(session.backend.calls) == calls_before

    # A different sample must invalidate its cache, without invalidating the original panel.
    write_captures(path.parent, "new D", "new E")
    session.use_captured_inputs(scan(session, [path.parent]))
    session.acquire()
    assert len(session.backend.calls) == calls_before + 1
    assert {story.fields["text"] for story in session.backend.calls[-1][1]} == {
        "new D",
        "new E",
    }


def test_rewind_after_restart_reopens_captured_labels(session, tmp_path):
    path = write_captures(tmp_path / "captures", "captured")
    session.use_captured_inputs(scan(session, [path.parent]))
    story = session.capture_pool[0]
    session.add_label(
        story_id=story.id,
        label=True,
        rationale=None,
        acquired_by="ambiguous",
        probability=0.5,
    )
    session.state.history.append(
        CandidateHistory(
            round_number=1,
            candidate=session.state.current_candidate,
            decision="accepted",
        )
    )
    session.state.round_number = 2
    session.store.save_state(session.state)
    resumed = ClimbSession(
        session.store.load_state(), session.stories, session.store, FakeBackend()
    )
    assert resumed.capture_pool is None
    resumed.rewind_previous_round()
    assert resumed.unlabeled_stories() == [story]


def test_resume_after_final_label_still_offers_optimization(
    session, tmp_path, monkeypatch
):
    session.state.round_number = 0
    for story in session.stories[:-1]:
        session.add_label(
            story_id=story.id,
            label=False,
            rationale=None,
            acquired_by="ambiguous",
            probability=0.5,
        )
    session.state.round_number = 1
    path = write_captures(tmp_path / ".jev-align" / "captures", "last capture")
    session.use_captured_inputs(scan(session, [path.parent]))
    session.add_label(
        story_id=session.capture_pool[0].id,
        label=True,
        rationale=None,
        acquired_by="ambiguous",
        probability=0.5,
    )
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(
        cli, "create_backend", lambda *_args, **_kwargs: session.backend
    )
    monkeypatch.setattr(
        cli, "_prompt_label", lambda *_args, **_kwargs: pytest.fail("already labeled")
    )
    monkeypatch.setattr(
        "jev_align.session.optimize_candidate",
        lambda **_kwargs: OptimizationOutcome(
            candidate=session.state.current_candidate,
            best_score=1,
            total_metric_calls=1,
            metadata={},
        ),
    )
    result = CliRunner().invoke(
        cli.app, ["optimize", "--resume", str(session.store.directory)], input="q\n"
    )
    assert result.exit_code == 0, result.output
    assert session.store.load_state().pending_candidate is not None
    assert len(session.store.load_labels()) == 6
