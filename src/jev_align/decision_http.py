from __future__ import annotations

import asyncio
import json
import math
import time
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
from .retries import DEFAULT_ATTEMPTS, RETRYABLE_HTTP_STATUSES, retry_delay

DECISION_CAPABILITIES = BackendCapabilities(
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
    for attempt in range(DEFAULT_ATTEMPTS):
        try:
            with urlopen(request, timeout=60) as response:  # noqa: S310
                payload = json.load(response)
        except HTTPError as error:
            error.close()
            if error.code in RETRYABLE_HTTP_STATUSES:
                delay = retry_delay(attempt, error.headers)
                if attempt + 1 < DEFAULT_ATTEMPTS and delay is not None:
                    time.sleep(delay)
                    continue
                reason = (
                    "rate limit exceeded"
                    if error.code == 429
                    else "provider request timed out"
                    if error.code == 408
                    else "temporary provider failure"
                )
                raise RuntimeError(
                    f"Evaluation provider returned HTTP {error.code}; {reason} "
                    f"after {attempt + 1} attempt{'s' if attempt else ''}, try again later"
                ) from error
            raise RuntimeError(
                f"Evaluation provider returned HTTP {error.code}; check credentials, access, "
                "and model availability"
            ) from error
        except (TimeoutError, URLError, ConnectionError) as error:
            delay = retry_delay(attempt)
            if attempt + 1 < DEFAULT_ATTEMPTS and delay is not None:
                time.sleep(delay)
                continue
            raise RuntimeError(
                "Evaluation provider connection failed or timed out "
                f"after {attempt + 1} attempts, try again later"
            ) from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RuntimeError(
                "Evaluation provider returned an invalid JSON response"
            ) from error
        if not isinstance(payload, dict):
            raise RuntimeError("Evaluation provider returned an invalid response")
        return payload
    raise AssertionError("retry attempts exhausted without a response or error")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Evaluation provider response is missing {name}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Evaluation provider response has an invalid {name}")
    if not math.isfinite(value):
        raise RuntimeError(f"Evaluation provider response has an invalid {name}")
    return float(value)


def _distribution(value: Any, expected: set[str], name: str) -> dict[str, float]:
    probabilities = _mapping(value, name)
    if set(probabilities) != expected:
        raise RuntimeError(f"Evaluation provider response has incorrect {name} keys")
    result = {key: _number(item, name) for key, item in probabilities.items()}
    if any(not 0 <= item <= 1 for item in result.values()) or not math.isclose(
        sum(result.values()), 1.0, abs_tol=1e-4, rel_tol=1e-4
    ):
        raise RuntimeError(f"Evaluation provider response has invalid {name}")
    return result


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
    for name, question in _questions(candidate, boolean_type="noul").items():
        answer = _mapping(answers.get(name), f"answers.{name}")
        expected_type = question["type"]
        if expected_type == "noul" and boolean_field == "probability":
            expected_type = "boolean"
        if answer.get("type", expected_type) != expected_type:
            raise RuntimeError(
                f"Evaluation provider response has an invalid type for {name}"
            )
    resolved_model = response.get("model")
    if not isinstance(resolved_model, str):
        resolved_model = fallback_model

    if isinstance(candidate, MulticlassCandidateSpec):
        answer = _mapping(answers.get("alignment"), "answers.alignment")
        probabilities = _distribution(
            answer.get("probabilities"), set(candidate.criteria), "choice probabilities"
        )
        choice = answer.get("choice")
        if not isinstance(choice, str):
            raise RuntimeError("Evaluation provider response has an invalid choice")
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
        probabilities = _distribution(
            answer.get("probabilities"),
            {str(index) for index in range(len(candidate.levels))},
            "score probabilities",
        )
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
class HTTPDecisionEvaluator:
    model: str
    concurrency: int = 16

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("concurrency must be positive")

    def _evaluate_checked(self, candidate: TaskSpec, story: Story) -> Prediction:
        try:
            return self._evaluate_one(candidate, story)
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(
                f"{self.display_name} ({self.model}), row {story.id}: {error}"
            ) from error

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
                    self._evaluate_checked, candidate, story
                )
            if on_complete is not None:
                on_complete()
            return prediction

        return list(await asyncio.gather(*(evaluate(story) for story in stories)))

    def _evaluate_one(self, candidate: TaskSpec, story: Story) -> Prediction:
        raise NotImplementedError
