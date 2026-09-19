from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean, median

from .models import Prediction


def ambiguity(probability: float) -> float:
    return 1.0 - 2.0 * abs(probability - 0.5)


def prediction_ambiguity(prediction: Prediction) -> float:
    if prediction.label_probabilities is not None:
        return max(
            ambiguity(value) for value in prediction.label_probabilities.values()
        )
    if prediction.confidence is not None:
        return 1.0 - prediction.confidence
    if prediction.probability is None:
        raise ValueError(
            "prediction has neither native confidence nor class probability"
        )
    return ambiguity(prediction.probability)


@dataclass(frozen=True)
class Acquisition:
    prediction: Prediction
    source: str


def select_batch(
    predictions: list[Prediction],
    *,
    batch_size: int = 5,
    exploration_count: int = 1,
    seed: int = 0,
) -> list[Acquisition]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0 <= exploration_count <= batch_size:
        raise ValueError("exploration_count must be between zero and batch_size")
    if len(predictions) < batch_size:
        raise ValueError("not enough unlabeled rows for an acquisition batch")

    ambiguous_count = batch_size - exploration_count
    ranked = sorted(
        predictions,
        key=lambda item: (-prediction_ambiguity(item), item.story_id),
    )
    ambiguous = ranked[:ambiguous_count]
    chosen = {item.story_id for item in ambiguous}
    remainder = [item for item in predictions if item.story_id not in chosen]
    rng = random.Random(seed)
    exploration = rng.sample(remainder, exploration_count)
    return [
        *(Acquisition(item, "ambiguous") for item in ambiguous),
        *(Acquisition(item, "exploration") for item in exploration),
    ]


def ambiguity_summary(predictions: list[Prediction]) -> dict[str, float | int]:
    values = [prediction_ambiguity(item) for item in predictions]
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "at_least_0_5": 0,
            "at_least_0_8": 0,
        }
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "at_least_0_5": sum(value >= 0.5 for value in values),
        "at_least_0_8": sum(value >= 0.8 for value in values),
    }
