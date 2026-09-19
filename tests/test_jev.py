from types import SimpleNamespace

import typesafe_sdk

from jev_align.jev import TypeSafeJevEvaluator
from jev_align.models import ScoreCandidateSpec, Story


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
