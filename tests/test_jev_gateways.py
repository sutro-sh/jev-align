from __future__ import annotations

from typing import Any

from jev_align import jev_gateways
from jev_align.jev_gateways import CloudflareJevEvaluator, VercelJevEvaluator
from jev_align.models import (
    CandidateSpec,
    MulticlassCandidateSpec,
    MultilabelCandidateSpec,
    MultilabelCriteria,
    ScoreCandidateSpec,
    Story,
)


def _story() -> Story:
    return Story(id="row-1", row_number=1, fields={"text": "Example"})


def test_cloudflare_uses_native_jev_payload_and_normalizes_score(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url, headers, body):
        captured.update(url=url, headers=headers, body=body)
        return {
            "result": {
                "model": "jev-1.13.0",
                "answers": {
                    "alignment": {
                        "type": "score",
                        "score": 1.25,
                        "confidence": 0.77,
                        "probabilities": {"0": 0.0, "1": 0.75, "2": 0.25},
                    }
                },
            }
        }

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account-1")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "secret")
    monkeypatch.setattr(jev_gateways, "_post_json", fake_post)
    candidate = ScoreCandidateSpec(
        instructions="How severe?", levels=["Low", "Medium", "High"]
    )

    prediction = CloudflareJevEvaluator("typesafe/jev", concurrency=1).evaluate_many(
        candidate, [_story()]
    )[0]

    assert captured["url"].endswith("/accounts/account-1/ai/run")
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["body"] == {
        "model": "typesafe/jev",
        "input": {
            "state": {"text": "Example"},
            "questions": {
                "alignment": {
                    "type": "score",
                    "instructions": "How severe?",
                    "criteria": ["Low", "Medium", "High"],
                }
            },
        },
    }
    assert prediction.score == 1.25
    assert prediction.confidence == 0.77
    assert prediction.score_probabilities == {0: 0.0, 1: 0.75, 2: 0.25}
    assert prediction.resolved_model == "jev-1.13.0"


def test_cloudflare_normalizes_multilabel_noul_answers(monkeypatch) -> None:
    def fake_post(_url, _headers, body):
        questions = body["input"]["questions"]
        assert questions["label_0"]["type"] == "noul"
        assert questions["label_1"]["criteria"]["false"] == "Not billing"
        return {
            "answers": {
                "label_0": {"type": "noul", "noul": 0.91},
                "label_1": {"type": "noul", "noul": 0.08},
            }
        }

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "token")
    monkeypatch.setattr(jev_gateways, "_post_json", fake_post)
    candidate = MultilabelCandidateSpec(
        instructions="Tag the ticket",
        labels={
            "urgent": MultilabelCriteria(
                true_criteria="Urgent", false_criteria="Not urgent"
            ),
            "billing": MultilabelCriteria(
                true_criteria="Billing", false_criteria="Not billing"
            ),
        },
    )

    prediction = CloudflareJevEvaluator("typesafe/jev", concurrency=1).evaluate_many(
        candidate, [_story()]
    )[0]

    assert prediction.label_probabilities == {"urgent": 0.91, "billing": 0.08}


def test_vercel_uses_evaluation_protocol_and_provider_confidence(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url, headers, body):
        captured.update(url=url, headers=headers, body=body)
        return {
            "answers": {
                "alignment": {
                    "type": "choice",
                    "choice": "billing",
                    "probabilities": {"billing": 0.82, "technical": 0.18},
                }
            },
            "providerMetadata": {"typesafe": {"confidence": {"alignment": 0.68}}},
        }

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gateway-key")
    monkeypatch.setattr(jev_gateways, "_post_json", fake_post)
    candidate = MulticlassCandidateSpec(
        instructions="Route this ticket",
        criteria={"billing": "Money", "technical": "Product failure"},
    )

    prediction = VercelJevEvaluator("typesafe-ai/jev", concurrency=1).evaluate_many(
        candidate, [_story()]
    )[0]

    assert captured["url"] == ("https://ai-gateway.vercel.sh/v4/ai/evaluation-model")
    assert captured["headers"]["ai-model-id"] == "typesafe-ai/jev"
    assert captured["headers"]["ai-evaluation-model-specification-version"] == "4"
    assert captured["body"]["questions"]["alignment"]["type"] == "choice"
    assert prediction.choice == "billing"
    assert prediction.probabilities == {"billing": 0.82, "technical": 0.18}
    assert prediction.confidence == 0.68


def test_vercel_binary_uses_boolean_wire_type(monkeypatch) -> None:
    def fake_post(_url, _headers, body):
        assert body["questions"]["alignment"] == {
            "type": "boolean",
            "instructions": "Is it relevant?",
            "criteria": {"true": "Relevant", "false": "Irrelevant"},
        }
        return {"answers": {"alignment": {"type": "boolean", "probability": 0.63}}}

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gateway-key")
    monkeypatch.setattr(jev_gateways, "_post_json", fake_post)
    candidate = CandidateSpec(
        instructions="Is it relevant?",
        true_criteria="Relevant",
        false_criteria="Irrelevant",
    )

    prediction = VercelJevEvaluator("typesafe-ai/jev", concurrency=1).evaluate_many(
        candidate, [_story()]
    )[0]

    assert prediction.probability == 0.63
