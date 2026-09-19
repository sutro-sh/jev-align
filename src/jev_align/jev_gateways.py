from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from .backends import BackendCapabilities
from .models import (
    MulticlassCandidateSpec,
    MultilabelCandidateSpec,
    Prediction,
    ScoreCandidateSpec,
    Story,
    TaskSpec,
)

JEV_CAPABILITIES = BackendCapabilities(
    task_types=frozenset({"binary", "multiclass", "multilabel", "score"}),
    uncertainty={
        "binary": "calibrated_probability",
        "multiclass": "native_confidence",
        "multilabel": "calibrated_probability",
        "score": "native_confidence",
    },
)


def _questions(
    candidate: TaskSpec, *, boolean_type: Literal["noul", "boolean"]
) -> dict[str, dict[str, Any]]:
    if isinstance(candidate, MulticlassCandidateSpec):
        return {
            "alignment": {
                "type": "choice",
                "instructions": candidate.instructions,
                "criteria": candidate.criteria,
            }
        }
    if isinstance(candidate, ScoreCandidateSpec):
        return {
            "alignment": {
                "type": "score",
                "instructions": candidate.instructions,
                "criteria": candidate.levels,
            }
        }
    if isinstance(candidate, MultilabelCandidateSpec):
        return {
            f"label_{index}": {
                "type": boolean_type,
                "instructions": (
                    f"{candidate.instructions}\n\n"
                    f"Determine whether the label {name!r} applies to this row."
                ),
                "criteria": {
                    "true": criteria.true_criteria,
                    "false": criteria.false_criteria,
                },
            }
            for index, (name, criteria) in enumerate(candidate.labels.items())
        }
    return {
        "alignment": {
            "type": boolean_type,
            "instructions": candidate.instructions,
            "criteria": {
                "true": candidate.true_criteria,
                "false": candidate.false_criteria,
            },
        }
    }


def _post_json(
    url: str, headers: Mapping[str, str], body: Mapping[str, Any]
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:  # noqa: S310
            payload = json.load(response)
    except HTTPError as error:
        raise RuntimeError(
            f"Jev provider returned HTTP {error.code}; check credentials, access, "
            "and model availability"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Could not reach the Jev provider: {error.reason}"
        ) from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("Jev provider returned an invalid JSON response") from error
    if not isinstance(payload, dict):
        raise RuntimeError("Jev provider returned an invalid response")
    return payload


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Jev provider response is missing {name}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Jev provider response has an invalid {name}")
    return float(value)


def _prediction(
    candidate: TaskSpec,
    story: Story,
    response: Mapping[str, Any],
    *,
    boolean_field: Literal["noul", "probability"],
    confidence: Mapping[str, Any] | None = None,
    fallback_model: str,
) -> Prediction:
    answers = _mapping(response.get("answers"), "answers")
    resolved_model = response.get("model")
    if not isinstance(resolved_model, str):
        resolved_model = fallback_model

    if isinstance(candidate, MulticlassCandidateSpec):
        answer = _mapping(answers.get("alignment"), "answers.alignment")
        probabilities = _mapping(answer.get("probabilities"), "choice probabilities")
        choice = answer.get("choice")
        if not isinstance(choice, str):
            raise RuntimeError("Jev provider response has an invalid choice")
        confidence_value = answer.get("confidence")
        if confidence_value is None and confidence is not None:
            confidence_value = confidence.get("alignment")
        return Prediction(
            story_id=story.id,
            probabilities={
                str(name): _number(value, f"probability for {name}")
                for name, value in probabilities.items()
            },
            choice=choice,
            confidence=_number(confidence_value, "choice confidence"),
            resolved_model=resolved_model,
        )
    if isinstance(candidate, ScoreCandidateSpec):
        answer = _mapping(answers.get("alignment"), "answers.alignment")
        probabilities = _mapping(answer.get("probabilities"), "score probabilities")
        confidence_value = answer.get("confidence")
        if confidence_value is None and confidence is not None:
            confidence_value = confidence.get("alignment")
        return Prediction(
            story_id=story.id,
            score=_number(answer.get("score"), "score"),
            score_probabilities={
                int(level): _number(value, f"probability for score {level}")
                for level, value in probabilities.items()
            },
            confidence=_number(confidence_value, "score confidence"),
            resolved_model=resolved_model,
        )
    if isinstance(candidate, MultilabelCandidateSpec):
        return Prediction(
            story_id=story.id,
            label_probabilities={
                name: _number(
                    _mapping(
                        answers.get(f"label_{index}"),
                        f"answers.label_{index}",
                    ).get(boolean_field),
                    f"probability for {name}",
                )
                for index, name in enumerate(candidate.labels)
            },
            resolved_model=resolved_model,
        )
    answer = _mapping(answers.get("alignment"), "answers.alignment")
    return Prediction(
        story_id=story.id,
        probability=_number(answer.get(boolean_field), "boolean probability"),
        resolved_model=resolved_model,
    )


@dataclass
class _GatewayJevEvaluator:
    model: str
    concurrency: int = 16

    def evaluate_many(
        self, candidate: TaskSpec, stories: Sequence[Story]
    ) -> list[Prediction]:
        if not stories:
            return []
        return asyncio.run(self._evaluate_many(candidate, stories, None))

    def evaluate_many_with_progress(
        self,
        candidate: TaskSpec,
        stories: Sequence[Story],
        description: str,
    ) -> list[Prediction]:
        if not stories:
            return []
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(description, total=len(stories))
            return asyncio.run(
                self._evaluate_many(candidate, stories, lambda: progress.advance(task))
            )

    async def _evaluate_many(
        self,
        candidate: TaskSpec,
        stories: Sequence[Story],
        on_complete: Callable[[], None] | None,
    ) -> list[Prediction]:
        semaphore = asyncio.Semaphore(self.concurrency)

        async def evaluate(story: Story) -> Prediction:
            async with semaphore:
                prediction = await asyncio.to_thread(
                    self._evaluate_one, candidate, story
                )
            if on_complete is not None:
                on_complete()
            return prediction

        return list(await asyncio.gather(*(evaluate(story) for story in stories)))

    def _evaluate_one(self, candidate: TaskSpec, story: Story) -> Prediction:
        raise NotImplementedError


@dataclass
class CloudflareJevEvaluator(_GatewayJevEvaluator):
    backend_id = "cloudflare"
    display_name = "Cloudflare Workers AI · JEV"
    capabilities = JEV_CAPABILITIES

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
class VercelJevEvaluator(_GatewayJevEvaluator):
    backend_id = "vercel"
    display_name = "Vercel AI Gateway · JEV"
    capabilities = JEV_CAPABILITIES

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
