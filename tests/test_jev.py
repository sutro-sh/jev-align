from types import SimpleNamespace

import pytest
import typesafe_sdk

from jev_align.jev import TypeSafeJevEvaluator
from jev_align.models import CandidateSpec, ScoreCandidateSpec, Story


def test_typesafe_evaluator_sends_score_and_reads_fractional_answer(
    monkeypatch,
) -> None:
    captured = {}

    class FakeClient:
        def __init__(self, *, model):
            captured["model"] = model

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def system_one(self, *, state, questions):
            captured["state"] = state
            captured["question"] = questions["alignment"]
            return SimpleNamespace(
                model="jev-resolved",
                answers={
                    "alignment": SimpleNamespace(
                        score=1.3,
                        probabilities={0: 0.0, 1: 0.7, 2: 0.3},
                        confidence=0.54,
                    )
                },
            )

    monkeypatch.setattr(typesafe_sdk, "AsyncTypeSafeClient", FakeClient)
    candidate = ScoreCandidateSpec(
        instructions="How severe is this issue?",
        levels=["Cosmetic", "Degraded with workaround", "Blocking"],
    )

    predictions = TypeSafeJevEvaluator(model="jev-test", concurrency=1).evaluate_many(
        candidate,
        [Story(id="row-1", row_number=1, fields={"text": "Export crashes"})],
    )

    assert isinstance(captured["question"], typesafe_sdk.Score)
    assert captured["question"].criteria == candidate.levels
    assert captured["state"] == {"text": "Export crashes"}
    assert predictions[0].score == 1.3
    assert predictions[0].score_probabilities == {0: 0.0, 1: 0.7, 2: 0.3}
    assert predictions[0].confidence == 0.54
    assert predictions[0].resolved_model == "jev-resolved"


def test_typesafe_evaluator_retries_transient_server_errors(
    monkeypatch,
) -> None:
    attempts = 0
    delays = []

    class TemporaryServerError(Exception):
        pass

    class FakeClient:
        def __init__(self, *, model):
            assert model == "jev-test"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def system_one(self, *, state, questions):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise TemporaryServerError("temporary 520")
            return SimpleNamespace(
                model="jev-resolved",
                answers={"alignment": SimpleNamespace(noul=0.9)},
            )

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(typesafe_sdk, "AsyncTypeSafeClient", FakeClient)
    monkeypatch.setattr(
        typesafe_sdk, "TypeSafeInternalServerError", TemporaryServerError
    )
    monkeypatch.setattr("jev_align.jev.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("jev_align.jev.random.uniform", lambda *_args: 0.0)
    candidate = CandidateSpec(
        instructions="Decide.",
        true_criteria="True when relevant.",
        false_criteria="False otherwise.",
    )

    predictions = TypeSafeJevEvaluator(model="jev-test", concurrency=1).evaluate_many(
        candidate,
        [Story(id="row-1", row_number=1, fields={"text": "Example"})],
    )

    assert attempts == 3
    assert delays == [0.5, 1.0]
    assert predictions[0].probability == 0.9


def test_typesafe_evaluator_does_not_retry_permanent_errors(monkeypatch) -> None:
    attempts = 0

    class PermanentError(Exception):
        pass

    class FakeClient:
        def __init__(self, *, model):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def system_one(self, *, state, questions):
            nonlocal attempts
            attempts += 1
            raise PermanentError("bad request")

    monkeypatch.setattr(typesafe_sdk, "AsyncTypeSafeClient", FakeClient)

    with pytest.raises(PermanentError, match="bad request"):
        TypeSafeJevEvaluator(model="jev-test", concurrency=1).evaluate_many(
            CandidateSpec(
                instructions="Decide.",
                true_criteria="True when relevant.",
                false_criteria="False otherwise.",
            ),
            [Story(id="row-1", row_number=1, fields={"text": "Example"})],
        )

    assert attempts == 1
