from __future__ import annotations

from pathlib import Path

from jev_align.acquisition import ambiguity, ambiguity_summary, select_batch
from jev_align.data import (
    clean_text,
    dataset_columns,
    dataset_row_count,
    discover_datasets,
    load_stories,
)
from jev_align.metrics import (
    binary_metrics,
    multiclass_metrics,
    multilabel_metrics,
    nearest_score_level,
    score_metrics,
)
from jev_align.models import Prediction


def test_clean_text_decodes_and_strips_html() -> None:
    assert clean_text("Hello<p>A &amp; B</p>") == "Hello\nA & B"


def test_load_stories_uses_stable_row_ids(tmp_path: Path) -> None:
    csv_path = tmp_path / "stories.csv"
    csv_path.write_text(
        "title,text,url\nOne,<p>Body</p>,https://x.test\n", encoding="utf-8"
    )
    stories = load_stories(csv_path, 1)
    assert stories[0].id == "row-000001"
    assert stories[0].state() == {
        "title": "One",
        "text": "Body",
        "url": "https://x.test",
    }


def test_load_stories_uses_all_rows_when_limit_exceeds_dataset(tmp_path: Path) -> None:
    csv_path = tmp_path / "stories.csv"
    csv_path.write_text("text\nOne\nTwo\n", encoding="utf-8")

    stories = load_stories(csv_path, 1000)

    assert [story.state() for story in stories] == [
        {"text": "One"},
        {"text": "Two"},
    ]
    assert dataset_row_count(csv_path) == 2


def test_jsonl_columns_and_selected_fields(tmp_path: Path) -> None:
    path = tmp_path / "stories.jsonl"
    path.write_text(
        '{"headline":"Plane news","body":"<p>Flying</p>","ignored":7}\n',
        encoding="utf-8",
    )
    assert dataset_columns(path) == ["headline", "body", "ignored"]
    story = load_stories(path, 1, ["headline", "body"])[0]
    assert story.state() == {"headline": "Plane news", "body": "Flying"}
    concatenated = load_stories(path, 1, concatenate=True)[0]
    assert concatenated.state() == {"content": "Plane news\nFlying\n7"}


def test_parquet_columns_and_selected_fields(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "stories.parquet"
    pq.write_table(
        pa.table({"headline": ["Plane news"], "score": [42]}),
        path,
    )
    assert dataset_columns(path) == ["headline", "score"]
    story = load_stories(path, 1)[0]
    assert story.state() == {"headline": "Plane news", "score": "42"}


def test_dataset_discovery_skips_hidden_and_environment_directories(
    tmp_path: Path,
) -> None:
    visible = tmp_path / "data" / "posts.csv"
    visible.parent.mkdir()
    visible.write_text("value\n1\n", encoding="utf-8")
    hidden = tmp_path / ".venv" / "hidden.jsonl"
    hidden.parent.mkdir()
    hidden.write_text('{"value":1}\n', encoding="utf-8")
    assert discover_datasets(tmp_path) == [visible]


def test_positive_class_f1() -> None:
    result = binary_metrics(
        [True, True, False, False],
        [True, False, True, False],
    )
    assert result.model_dump() == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


def test_multiclass_macro_f1() -> None:
    result = multiclass_metrics(
        ["aviation", "space", "space", "other"],
        ["aviation", "aviation", "space", "other"],
        ["aviation", "space", "other"],
    )
    assert result.accuracy == 0.75
    assert round(result.macro_f1, 3) == 0.778
    assert result.per_class["space"].support == 2


def test_multilabel_micro_f1_allows_unseen_labels() -> None:
    result = multilabel_metrics(
        [{"tech", "startups"}, {"tech"}, set()],
        [{"tech"}, {"tech", "culture"}, set()],
        ["tech", "startups", "culture", "unseen"],
    )
    assert round(result.micro_f1, 3) == 0.667
    assert result.exact_matches == 1
    assert result.per_label["unseen"].support == 0
    assert result.supported_labels == 2


def test_score_metric_respects_ordinal_distance() -> None:
    result = score_metrics([0, 2], [0.5, 1.5], level_count=3)
    assert result.fit_score == 0.75
    assert result.mae == 0.5
    assert result.normalized_mae == 0.25
    assert result.rounded_correct == 1
    assert nearest_score_level(0.5, 3) == 1


def test_acquisition_is_four_ambiguous_plus_one_seeded_random() -> None:
    predictions = [
        Prediction(story_id=f"row-{index}", probability=probability)
        for index, probability in enumerate([0.5, 0.49, 0.52, 0.45, 0.1, 0.9, 0.2])
    ]
    selected = select_batch(predictions, batch_size=5, exploration_count=1, seed=3)
    assert [item.prediction.story_id for item in selected[:4]] == [
        "row-0",
        "row-1",
        "row-2",
        "row-3",
    ]
    assert selected[-1].source == "exploration"
    assert selected[-1].prediction.story_id not in {"row-0", "row-1", "row-2", "row-3"}


def test_multiclass_acquisition_uses_native_choice_confidence() -> None:
    predictions = [
        Prediction(
            story_id=f"row-{index}",
            probabilities={"a": probability, "b": 1 - probability},
            choice="a" if probability >= 0.5 else "b",
            confidence=confidence,
        )
        for index, (probability, confidence) in enumerate(
            [(0.5, 0.0), (0.6, 0.2), (0.7, 0.4), (0.8, 0.6), (0.9, 0.8), (1.0, 1.0)]
        )
    ]
    selected = select_batch(predictions, batch_size=3, exploration_count=0)
    assert [item.prediction.story_id for item in selected] == [
        "row-0",
        "row-1",
        "row-2",
    ]


def test_score_acquisition_uses_native_confidence() -> None:
    predictions = [
        Prediction(
            story_id=f"row-{index}",
            score=0.5,
            score_probabilities={0: 0.5, 1: 0.5},
            confidence=confidence,
        )
        for index, confidence in enumerate([0.8, 0.1, 0.5])
    ]
    selected = select_batch(predictions, batch_size=2, exploration_count=0)
    assert [item.prediction.story_id for item in selected] == ["row-1", "row-2"]


def test_multilabel_acquisition_uses_most_ambiguous_label() -> None:
    predictions = [
        Prediction(story_id="a", label_probabilities={"tech": 0.9, "culture": 0.1}),
        Prediction(story_id="b", label_probabilities={"tech": 0.51, "culture": 0.0}),
        Prediction(story_id="c", label_probabilities={"tech": 0.8, "culture": 0.45}),
    ]
    selected = select_batch(predictions, batch_size=2, exploration_count=0)
    assert [item.prediction.story_id for item in selected] == ["b", "c"]


def test_ambiguity_summary() -> None:
    predictions = [
        Prediction(story_id="a", probability=0.5),
        Prediction(story_id="b", probability=0.0),
    ]
    assert ambiguity(0.5) == 1.0
    assert ambiguity(0.0) == 0.0
    assert ambiguity_summary(predictions) == {
        "count": 2,
        "mean": 0.5,
        "median": 0.5,
        "at_least_0_5": 1,
        "at_least_0_8": 1,
    }
