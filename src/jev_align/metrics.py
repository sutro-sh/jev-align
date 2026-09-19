from __future__ import annotations

import math
from collections.abc import Iterable

from .models import (
    BinaryMetrics,
    ClassMetrics,
    MulticlassMetrics,
    MultilabelMetrics,
    ScoreMetrics,
)


def binary_metrics(
    expected: Iterable[bool], predicted: Iterable[bool]
) -> BinaryMetrics:
    pairs = list(zip(expected, predicted, strict=True))
    tp = sum(want and got for want, got in pairs)
    fp = sum(not want and got for want, got in pairs)
    fn = sum(want and not got for want, got in pairs)
    tn = sum(not want and not got for want, got in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def multiclass_metrics(
    expected: Iterable[str], predicted: Iterable[str], classes: Iterable[str]
) -> MulticlassMetrics:
    pairs = list(zip(expected, predicted, strict=True))
    per_class: dict[str, ClassMetrics] = {}
    for label in classes:
        tp = sum(want == label and got == label for want, got in pairs)
        fp = sum(want != label and got == label for want, got in pairs)
        fn = sum(want == label and got != label for want, got in pairs)
        support = sum(want == label for want, _ in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_class[label] = ClassMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
        )
    correct = sum(want == got for want, got in pairs)
    total = len(pairs)
    return MulticlassMetrics(
        correct=correct,
        total=total,
        accuracy=correct / total if total else 0.0,
        macro_f1=sum(item.f1 for item in per_class.values()) / len(per_class),
        per_class=per_class,
    )


def multilabel_metrics(
    expected: Iterable[set[str]],
    predicted: Iterable[set[str]],
    labels: Iterable[str],
) -> MultilabelMetrics:
    pairs = list(zip(expected, predicted, strict=True))
    label_names = list(labels)
    per_label: dict[str, ClassMetrics] = {}
    total_tp = total_fp = total_fn = 0
    for label in label_names:
        tp = sum(label in want and label in got for want, got in pairs)
        fp = sum(label not in want and label in got for want, got in pairs)
        fn = sum(label in want and label not in got for want, got in pairs)
        support = sum(label in want for want, _ in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_label[label] = ClassMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

    if total_tp + total_fp + total_fn == 0:
        micro_precision = micro_recall = micro_f1 = 1.0
    else:
        micro_precision = (
            total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        )
        micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if micro_precision + micro_recall
            else 0.0
        )
    supported = [item.f1 for item in per_label.values() if item.support]
    exact_matches = sum(want == got for want, got in pairs)
    total = len(pairs)
    return MultilabelMetrics(
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
        macro_supported_f1=sum(supported) / len(supported) if supported else micro_f1,
        exact_matches=exact_matches,
        total=total,
        exact_match_accuracy=exact_matches / total if total else 0.0,
        supported_labels=len(supported),
        label_count=len(label_names),
        per_label=per_label,
    )


def nearest_score_level(score: float, level_count: int) -> int:
    if level_count < 2:
        raise ValueError("Score requires at least two levels")
    return min(level_count - 1, max(0, math.floor(score + 0.5)))


def score_metrics(
    expected: Iterable[int], predicted: Iterable[float], level_count: int
) -> ScoreMetrics:
    if level_count < 2:
        raise ValueError("Score requires at least two levels")
    pairs = list(zip(expected, predicted, strict=True))
    top_level = level_count - 1
    if any(not 0 <= want <= top_level for want, _ in pairs):
        raise ValueError("expected Score label is outside the level range")
    errors = [abs(want - got) for want, got in pairs]
    mae = sum(errors) / len(errors) if errors else 0.0
    normalized_mae = mae / top_level
    rounded_correct = sum(
        want == nearest_score_level(got, level_count) for want, got in pairs
    )
    total = len(pairs)
    return ScoreMetrics(
        fit_score=max(0.0, 1.0 - normalized_mae),
        mae=mae,
        normalized_mae=normalized_mae,
        rounded_correct=rounded_correct,
        total=total,
        rounded_accuracy=rounded_correct / total if total else 0.0,
    )
