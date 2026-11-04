from __future__ import annotations

import os
from dataclasses import dataclass

from .decision_http import (
    HTTPDecisionEvaluator,
    DECISION_CAPABILITIES,
    _mapping,
    _post_json,
    _prediction,
    _questions,
)
from .models import Prediction, Story, TaskSpec


@dataclass
class CloudflareJevEvaluator(HTTPDecisionEvaluator):
    backend_id = "cloudflare"
    display_name = "Cloudflare Workers AI · JEV"
    capabilities = DECISION_CAPABILITIES

    def _evaluate_one(self, candidate: TaskSpec, story: Story) -> Prediction:
        account_id = os.environ["CLOUDFLARE_ACCOUNT_ID"]
        token = os.environ["CLOUDFLARE_API_TOKEN"]
        payload = _post_json(
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run",
            {"Authorization": f"Bearer {token}"},
            {
                "model": self.model,
                "input": {
                    "state": story.state(),
                    "questions": _questions(candidate, boolean_type="noul"),
                },
            },
        )
        if "result" in payload:
            payload = dict(_mapping(payload["result"], "result"))
        return _prediction(
            candidate,
            story,
            payload,
            boolean_field="noul",
            fallback_model=self.model,
        )


@dataclass
class VercelJevEvaluator(HTTPDecisionEvaluator):
    backend_id = "vercel"
    display_name = "Vercel AI Gateway · JEV"
    capabilities = DECISION_CAPABILITIES

    def _evaluate_one(self, candidate: TaskSpec, story: Story) -> Prediction:
        token = os.environ["AI_GATEWAY_API_KEY"]
        payload = _post_json(
            "https://ai-gateway.vercel.sh/v4/ai/evaluation-model",
            {
                "Authorization": f"Bearer {token}",
                "ai-gateway-protocol-version": "0.0.1",
                "ai-gateway-auth-method": "api-key",
                "ai-evaluation-model-specification-version": "4",
                "ai-model-id": self.model,
            },
            {
                "state": story.state(),
                "questions": _questions(candidate, boolean_type="boolean"),
            },
        )
        provider_metadata = _mapping(
            payload.get("providerMetadata", {}), "providerMetadata"
        )
        typesafe_metadata = _mapping(
            provider_metadata.get("typesafe", {}), "providerMetadata.typesafe"
        )
        confidence = _mapping(
            typesafe_metadata.get("confidence", {}),
            "providerMetadata.typesafe.confidence",
        )
        return _prediction(
            candidate,
            story,
            payload,
            boolean_field="probability",
            confidence=confidence,
            fallback_model=self.model,
        )
