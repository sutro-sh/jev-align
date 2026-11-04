from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from .backends import BackendCapabilities, EvaluationBackend
from .models import (
    DEFAULT_JEV_MODEL,
    MulticlassCandidateSpec,
    MultilabelCandidateSpec,
    Prediction,
    ScoreCandidateSpec,
    Story,
    TaskSpec,
)
from .retries import (
    DEFAULT_ATTEMPTS,
    INITIAL_BACKOFF_SECONDS,
    RETRYABLE_HTTP_STATUSES,
    retry_delay,
)


@dataclass
class TypeSafeJevEvaluator:
    backend_id = "typesafe"
    display_name = "TypeSafe JEV"
    capabilities = BackendCapabilities(
        task_types=frozenset({"binary", "multiclass", "multilabel", "score"}),
        uncertainty={
            "binary": "calibrated_probability",
            "multiclass": "native_confidence",
            "multilabel": "calibrated_probability",
            "score": "native_confidence",
        },
    )
    model: str = DEFAULT_JEV_MODEL
    concurrency: int = 16
    transient_attempts: int = DEFAULT_ATTEMPTS
    transient_backoff_seconds: float = INITIAL_BACKOFF_SECONDS

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
                self._evaluate_many(
                    candidate,
                    stories,
                    lambda: progress.advance(task),
                )
            )

    async def _evaluate_many(
        self,
        candidate: TaskSpec,
        stories: Sequence[Story],
        on_complete: Callable[[], None] | None,
    ) -> list[Prediction]:
        from typesafe_sdk import (
            AsyncTypeSafeClient,
            Choice,
            Noul,
            Score,
            RetryPolicy,
            TypeSafeAPIError,
            TypeSafeAPIConnectionError,
            TypeSafeAPITimeoutError,
            TypeSafeInternalServerError,
            TypeSafeRateLimitError,
        )

        transient_errors = (
            TypeSafeAPIConnectionError,
            TypeSafeAPITimeoutError,
            TypeSafeInternalServerError,
            TypeSafeRateLimitError,
        )

        semaphore = asyncio.Semaphore(self.concurrency)
        if isinstance(candidate, MulticlassCandidateSpec):
            questions = {
                "alignment": Choice(
                    instructions=candidate.instructions,
                    criteria=candidate.criteria,
                )
            }
        elif isinstance(candidate, ScoreCandidateSpec):
            questions = {
                "alignment": Score(
                    instructions=candidate.instructions,
                    criteria=candidate.levels,
                )
            }
        elif isinstance(candidate, MultilabelCandidateSpec):
            questions = {
                f"label_{index}": Noul(
                    instructions=(
                        f"{candidate.instructions}\n\n"
                        f"Determine whether the label {name!r} applies to this row."
                    ),
                    criteria={
                        "true": criteria.true_criteria,
                        "false": criteria.false_criteria,
                    },
                )
                for index, (name, criteria) in enumerate(candidate.labels.items())
            }
        else:
            questions = {
                "alignment": Noul(
                    instructions=candidate.instructions,
                    criteria={
                        "true": candidate.true_criteria,
                        "false": candidate.false_criteria,
                    },
                )
            }

        # Own the retry count here instead of multiplying it by SDK retries.
        async with AsyncTypeSafeClient(
            model=self.model, retry=RetryPolicy(max_retries=0)
        ) as client:

            async def evaluate(story: Story) -> Prediction:
                for attempt in range(self.transient_attempts):
                    try:
                        async with semaphore:
                            result = await client.system_one(
                                state=story.state(), questions=questions
                            )
                        break
                    except (*transient_errors, TypeSafeAPIError) as error:
                        if (
                            isinstance(error, TypeSafeAPIError)
                            and error.status not in RETRYABLE_HTTP_STATUSES
                        ):
                            raise
                        if attempt + 1 >= self.transient_attempts:
                            raise
                        delay = retry_delay(
                            attempt,
                            getattr(error, "headers", None),
                            initial_delay=self.transient_backoff_seconds,
                        )
                        if delay is None:
                            raise
                        await asyncio.sleep(delay)
                if isinstance(candidate, MulticlassCandidateSpec):
                    answer = result.answers["alignment"]
                    prediction = Prediction(
                        story_id=story.id,
                        probabilities=dict(answer.probabilities),
                        choice=answer.choice,
                        confidence=float(answer.confidence),
                        resolved_model=getattr(result, "model", None),
                    )
                elif isinstance(candidate, ScoreCandidateSpec):
                    answer = result.answers["alignment"]
                    prediction = Prediction(
                        story_id=story.id,
                        score=float(answer.score),
                        score_probabilities={
                            int(level): float(probability)
                            for level, probability in answer.probabilities.items()
                        },
                        confidence=float(answer.confidence),
                        resolved_model=getattr(result, "model", None),
                    )
                elif isinstance(candidate, MultilabelCandidateSpec):
                    prediction = Prediction(
                        story_id=story.id,
                        label_probabilities={
                            name: float(result.answers[f"label_{index}"].noul)
                            for index, name in enumerate(candidate.labels)
                        },
                        resolved_model=getattr(result, "model", None),
                    )
                else:
                    answer = result.answers["alignment"]
                    prediction = Prediction(
                        story_id=story.id,
                        probability=float(answer.noul),
                        resolved_model=getattr(result, "model", None),
                    )
                if on_complete is not None:
                    on_complete()
                return prediction

            return list(await asyncio.gather(*(evaluate(story) for story in stories)))


# Source-compatible protocol name for integrations written before the backend rename.
JevEvaluator = EvaluationBackend
