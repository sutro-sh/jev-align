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
        def __init__(self, *, model, retry):
            assert retry.max_retries == 0
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
        def __init__(self, *, model, retry):
            assert retry.max_retries == 0
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
    monkeypatch.setattr("jev_align.retries.random.uniform", lambda *_args: 0.0)
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
        def __init__(self, *, model, retry):
            assert retry.max_retries == 0
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


@pytest.mark.parametrize("recover", [True, False])
def test_typesafe_429_respects_retry_after_and_attempt_limit(monkeypatch, recover):
    attempts = 0
    delays = []
    error = typesafe_sdk.TypeSafeRateLimitError(429, None, {"Retry-After": "2"})

    class FakeClient:
        def __init__(self, *, model, retry):
            assert retry.max_retries == 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def system_one(self, **kwargs):
            nonlocal attempts
            attempts += 1
            if not recover or attempts < 4:
                raise error
            return SimpleNamespace(answers={"alignment": SimpleNamespace(noul=0.9)})

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(typesafe_sdk, "AsyncTypeSafeClient", FakeClient)
    monkeypatch.setattr("jev_align.jev.asyncio.sleep", sleep)
    evaluator = TypeSafeJevEvaluator(model="jev-test", concurrency=1)
    candidate = CandidateSpec(
        instructions="Decide", true_criteria="Yes", false_criteria="No"
    )
    stories = [Story(id="row-1", row_number=1, fields={"text": "Example"})]
    if recover:
        assert evaluator.evaluate_many(candidate, stories)[0].probability == 0.9
    else:
        with pytest.raises(typesafe_sdk.TypeSafeRateLimitError) as caught:
            evaluator.evaluate_many(candidate, stories)
        assert caught.value is error
    assert attempts == 4
    assert delays == [2, 2, 2]


def test_typesafe_does_not_retry_before_a_long_server_cooldown(monkeypatch):
    attempts = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def system_one(self, **kwargs):
            attempts.append(kwargs)
            raise typesafe_sdk.TypeSafeRateLimitError(
                429, None, {"Retry-After": "3600"}
            )

    monkeypatch.setattr(typesafe_sdk, "AsyncTypeSafeClient", FakeClient)
    with pytest.raises(typesafe_sdk.TypeSafeRateLimitError):
        TypeSafeJevEvaluator().evaluate_many(
            CandidateSpec(
                instructions="Decide", true_criteria="Yes", false_criteria="No"
            ),
            [Story(id="one", row_number=1, fields={"text": "Example"})],
        )
    assert len(attempts) == 1


@pytest.mark.parametrize(
    "status,expected_attempts", [(400, 1), (401, 1), (403, 1), (408, 4), (503, 4)]
)
def test_typesafe_status_retry_policy(monkeypatch, status, expected_attempts):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def system_one(self, **kwargs):
            calls.append(kwargs)
            raise typesafe_sdk.TypeSafeAPIError(status, None, {})

    async def sleep(_delay):
        pass

    monkeypatch.setattr(typesafe_sdk, "AsyncTypeSafeClient", FakeClient)
    monkeypatch.setattr("jev_align.jev.asyncio.sleep", sleep)
    with pytest.raises(typesafe_sdk.TypeSafeAPIError):
        TypeSafeJevEvaluator().evaluate_many(
            CandidateSpec(
                instructions="Decide", true_criteria="Yes", false_criteria="No"
            ),
            [Story(id="one", row_number=1, fields={"text": "Example"})],
        )
    assert len(calls) == expected_attempts
