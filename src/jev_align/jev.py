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
    MulticlassCandidateSpec,
    MultilabelCandidateSpec,
    Prediction,
    ScoreCandidateSpec,
    Story,
    TaskSpec,
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
    model: str = "jev-1.13.0"
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
        from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score

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

        async with AsyncTypeSafeClient(model=self.model) as client:

            async def evaluate(story: Story) -> Prediction:
                async with semaphore:
                    result = await client.system_one(
                        state=story.state(), questions=questions
                    )
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
