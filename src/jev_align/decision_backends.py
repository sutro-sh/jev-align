"""Official hosted decision APIs, normalized at the backend boundary.

Contracts:
- https://docs.perplexity.ai/api-reference/decisions-post
- https://docs.liquid.ai/lfm/models/d1
- https://developers.openai.com/api/docs/guides/decisions
- https://developers.cloudflare.com/workers-ai/models/clef/
- https://openrouter.ai/blog/tutorials/how-to-use-jev/
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .decision_http import (
    HTTPDecisionEvaluator,
    DECISION_CAPABILITIES,
    _mapping,
    _post_json,
    _prediction,
    _questions,
)
from .models import Prediction, Story, TaskSpec

CLEF_MODELS = {
    "@cf/cloudflare/clef": "clef",
    "@cf/cloudflare/clef-flash": "clef-flash",
    "clef": "clef",
    "clef-flash": "clef-flash",
}


def _credential(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set for this evaluation backend")
    return value


@dataclass
class _SystemOneDecisionEvaluator(HTTPDecisionEvaluator):
    capabilities = DECISION_CAPABILITIES
    api_url = ""
    api_key_env = ""

    def _evaluate_one(self, candidate: TaskSpec, story: Story) -> Prediction:
        questions = _questions(candidate, boolean_type="noul")
        # Keep batches within the 128-question limit used by Perplexity Decider.
        answers: dict[str, Any] = {}
        resolved_model = self.model
        items = list(questions.items())
        for start in range(0, len(items), 128):
            payload = _post_json(
                self.api_url,
                {"Authorization": f"Bearer {_credential(self.api_key_env)}"},
                {
                    "model": self.model,
                    "state": story.state(),
                    "questions": dict(items[start : start + 128]),
                },
            )
            answers.update(_mapping(payload.get("answers"), "answers"))
            resolved_model = payload.get("model", resolved_model)
        return _prediction(
            candidate,
            story,
            {"answers": answers, "model": resolved_model},
            boolean_field="noul",
            fallback_model=self.model,
        )


@dataclass
class PerplexityDeciderEvaluator(_SystemOneDecisionEvaluator):
    backend_id = "perplexity"
    display_name = "Perplexity Decider"
    api_url = "https://api.perplexity.ai/v1/decisions"
    api_key_env = "PERPLEXITY_API_KEY"


@dataclass
class OpenRouterDecisionEvaluator(_SystemOneDecisionEvaluator):
    backend_id = "openrouter"
    display_name = "OpenRouter Decisions"
    api_url = "https://openrouter.ai/api/alpha/decisions"
    api_key_env = "OPENROUTER_API_KEY"


@dataclass
class LiquidDecisionEvaluator(HTTPDecisionEvaluator):
    backend_id = "liquid"
    display_name = "Liquid D1"
    capabilities = DECISION_CAPABILITIES

    def _evaluate_one(self, candidate: TaskSpec, story: Story) -> Prediction:
        payload = _post_json(
            "https://api.liquid.ai/decisions/v1/systemone",
            {"Authorization": f"Bearer {_credential('LIQUID_API_KEY')}"},
            {
                "model": self.model,
                "state": story.state(),
                "questions": _questions(candidate, boolean_type="noul"),
            },
        )
        return _prediction(
            candidate, story, payload, boolean_field="noul", fallback_model=self.model
        )


@dataclass
class CloudflareClefEvaluator(HTTPDecisionEvaluator):
    backend_id = "cloudflare"
    capabilities = DECISION_CAPABILITIES

    @property
    def display_name(self) -> str:
        name = "Clef Flash" if CLEF_MODELS[self.model] == "clef-flash" else "Clef"
        return f"Cloudflare Workers AI · {name}"

    def _evaluate_one(self, candidate: TaskSpec, story: Story) -> Prediction:
        account = _credential("CLOUDFLARE_ACCOUNT_ID")
        model = CLEF_MODELS[self.model]
        items = list(_questions(candidate, boolean_type="noul").items())
        answers: dict[str, Any] = {}
        resolved_model = self.model
        for start in range(0, len(items), 64):
            payload = _post_json(
                f"https://api.cloudflare.com/client/v4/accounts/{account}"
                f"/ai/run/@cf/cloudflare/{model}",
                {"Authorization": f"Bearer {_credential('CLOUDFLARE_API_TOKEN')}"},
                {
                    "model": model,
                    "state": story.state(),
                    "questions": dict(items[start : start + 64]),
                },
            )
            if payload.get("success") is False:
                raise RuntimeError("Cloudflare rejected the evaluation request")
            result = _mapping(payload.get("result"), "result")
            answers.update(_mapping(result.get("answers"), "answers"))
            resolved_model = result.get("model", resolved_model)
        return _prediction(
            candidate,
            story,
            {"answers": answers, "model": resolved_model},
            boolean_field="noul",
            fallback_model=self.model,
        )


def _openai_questions(candidate: TaskSpec) -> list[dict[str, Any]]:
    result = []
    for name, question in _questions(candidate, boolean_type="noul").items():
        kind = question["type"]
        translated: dict[str, Any] = {
            "name": name,
            "type": "predicate" if kind == "noul" else kind,
            "instructions": question["instructions"],
        }
        criteria = question["criteria"]
        if kind == "noul":
            translated["instructions"] += (
                f"\n\nTrue criteria: {criteria['true']}"
                f"\nFalse criteria: {criteria['false']}"
            )
        elif kind == "choice":
            translated["choices"] = [
                {"value": label, "description": description}
                for label, description in criteria.items()
            ]
        else:
            translated["levels"] = [
                {"label": str(index), "description": description}
                for index, description in enumerate(criteria)
            ]
        result.append(translated)
    return result


def _openai_answers(
    payload: dict[str, Any], questions: list[dict[str, Any]]
) -> dict[str, Any]:
    answers = payload.get("answers")
    if not isinstance(answers, list):
        raise RuntimeError("OpenAI response is missing answers")
    expected = {question["name"]: question["type"] for question in questions}
    normalized: dict[str, Any] = {}
    for raw_answer in answers:
        answer = dict(_mapping(raw_answer, "answer"))
        name = answer.get("name")
        if not isinstance(name, str) or name not in expected or name in normalized:
            raise RuntimeError(
                "OpenAI response has an unexpected or duplicate answer name"
            )
        kind = answer.get("type")
        if kind == "refusal":
            raise RuntimeError(f"OpenAI refused decision {name}")
        if kind != expected[name]:
            raise RuntimeError(
                f"OpenAI response has an incorrect answer type for {name}"
            )
        if kind == "predicate":
            answer["type"] = "noul"
            answer["noul"] = answer.get("probability")
        else:
            probabilities = answer.get("probabilities")
            if not isinstance(probabilities, list):
                raise RuntimeError(
                    f"OpenAI response is missing probabilities for {name}"
                )
            distribution: dict[str, Any] = {}
            for raw_entry in probabilities:
                entry = _mapping(raw_entry, "probability entry")
                value = entry.get("value")
                if kind == "choice":
                    valid = isinstance(value, str)
                else:
                    valid = isinstance(value, int) and not isinstance(value, bool)
                if not valid or str(value) in distribution:
                    raise RuntimeError(
                        f"OpenAI response has invalid probability keys for {name}"
                    )
                distribution[str(value)] = entry.get("probability")
            answer["probabilities"] = distribution
        normalized[name] = answer
    if set(normalized) != set(expected):
        raise RuntimeError("OpenAI response is missing requested answers")
    return normalized


@dataclass
class OpenAIDecisionEvaluator(HTTPDecisionEvaluator):
    backend_id = "openai"
    display_name = "OpenAI Decisions"
    capabilities = DECISION_CAPABILITIES

    def _evaluate_one(self, candidate: TaskSpec, story: Story) -> Prediction:
        questions = _openai_questions(candidate)
        answers: dict[str, Any] = {}
        resolved_model = self.model
        for start in range(0, len(questions), 400):
            batch = questions[start : start + 400]
            payload = _post_json(
                "https://api.openai.com/v1/decisions",
                {"Authorization": f"Bearer {_credential('OPENAI_API_KEY')}"},
                {
                    "model": self.model,
                    "input": [
                        {
                            "role": "user",
                            "content": [{"type": "input_json", "data": story.state()}],
                        }
                    ],
                    "questions": batch,
                },
            )
            answers.update(_openai_answers(payload, batch))
            resolved_model = payload.get("model", resolved_model)
        return _prediction(
            candidate,
            story,
            {"answers": answers, "model": resolved_model},
            boolean_field="noul",
            fallback_model=self.model,
        )
