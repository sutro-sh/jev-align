from pathlib import Path

from jev_align.cli import (
    _acquisition_context,
    _certainty_bar,
    _compact_option_label,
    _menu_render,
    _metrics_table,
    _pretty_display_value,
    _render_story,
    _select_multilabel_labels,
    _show_current_uncertainty,
    console,
)
from jev_align.models import (
    CandidateSpec,
    CandidateHistory,
    ClassMetrics,
    MulticlassCandidateSpec,
    MulticlassMetrics,
    MultilabelCandidateSpec,
    MultilabelCriteria,
    Prediction,
    ScoreCandidateSpec,
    ScoreMetrics,
    RunState,
    Story,
)
from jev_align.persistence import RunStore
from jev_align.session import RoundReport
from typer.testing import CliRunner

from jev_align import cli


def test_menu_labels_are_rendered_literally() -> None:
    rendered = _menu_render("Label", [("t", "True"), ("f", "False")], 0)
    assert rendered.plain == "Label\n  > True\n    False"


def test_label_menu_does_not_render_hidden_back_shortcut() -> None:
    rendered = _menu_render("Label", [("t", "True"), ("f", "False")], 1)
    assert rendered.plain == "Label\n    True\n  > False"
    assert "Back" not in rendered.plain


def test_binary_label_picker_returns_booleans_for_true_and_false(monkeypatch) -> None:
    candidate = CandidateSpec(
        instructions="Is this aviation?",
        true_criteria="It is aviation.",
        false_criteria="It is not aviation.",
    )
    responses = iter(["true", "false"])
    monkeypatch.setattr(console, "input", lambda _prompt: next(responses))

    assert cli._prompt_label(candidate, can_go_back=False) is True
    assert cli._prompt_label(candidate, can_go_back=False) is False


def test_label_pickers_default_to_jev_suggestions(monkeypatch) -> None:
    monkeypatch.setattr(console, "input", lambda _prompt: "")
    binary = CandidateSpec(
        instructions="Is this aviation?",
        true_criteria="It is aviation.",
        false_criteria="It is not aviation.",
    )
    multiclass = MulticlassCandidateSpec(
        instructions="Choose a topic.",
        criteria={"aviation": "Aircraft", "space": "Spacecraft"},
    )
    multilabel = MultilabelCandidateSpec(
        instructions="Apply topics.",
        labels={
            name: MultilabelCriteria(
                true_criteria=f"About {name}",
                false_criteria=f"Not about {name}",
            )
            for name in ("tech", "culture")
        },
    )

    assert (
        cli._prompt_label(
            binary,
            can_go_back=False,
            prediction=Prediction(story_id="a", probability=0.2),
        )
        is False
    )
    assert (
        cli._prompt_label(
            multiclass,
            can_go_back=False,
            prediction=Prediction(
                story_id="b",
                choice="space",
                confidence=0.8,
                probabilities={"aviation": 0.1, "space": 0.9},
            ),
        )
        == "space"
    )
    assert cli._prompt_label(
        multilabel,
        can_go_back=False,
        prediction=Prediction(
            story_id="c", label_probabilities={"tech": 0.8, "culture": 0.3}
        ),
    ) == ["tech"]


def test_multilabel_picker_can_clear_jev_suggestions(monkeypatch) -> None:
    monkeypatch.setattr(console, "input", lambda _prompt: "/none")
    assert (
        _select_multilabel_labels(
            ["tech", "culture"],
            can_go_back=False,
            suggested={"tech", "culture"},
        )
        == []
    )


def test_score_picker_defaults_to_nearest_jev_level(monkeypatch) -> None:
    candidate = ScoreCandidateSpec(
        instructions="How severe is this issue?",
        levels=["Cosmetic", "Degraded", "Blocking"],
    )
    monkeypatch.setattr(console, "input", lambda _prompt: "")

    assert (
        cli._prompt_label(
            candidate,
            can_go_back=False,
            prediction=Prediction(
                story_id="a",
                score=1.6,
                score_probabilities={0: 0.0, 1: 0.4, 2: 0.6},
                confidence=0.6,
            ),
        )
        == 2
    )


def test_multilabel_picker_allows_none_and_hidden_back(monkeypatch) -> None:
    monkeypatch.setattr(console, "input", lambda _prompt: "")
    assert _select_multilabel_labels(["tech", "culture"], can_go_back=False) == []
    monkeypatch.setattr(console, "input", lambda _prompt: "b")
    assert (
        _select_multilabel_labels(["tech", "culture"], can_go_back=True)
        is cli.BACK_LABEL
    )


def test_rationale_uses_explicit_back_command_and_allows_b(monkeypatch) -> None:
    responses = iter(["b", "/back", "Because it mentions an aircraft"])
    monkeypatch.setattr(console, "input", lambda _prompt: next(responses))

    assert cli._prompt_rationale() == "b"
    assert cli._prompt_rationale() is cli.BACK_LABEL
    assert cli._prompt_rationale() == "Because it mentions an aircraft"


def test_long_menu_uses_a_scrolling_viewport() -> None:
    options = [(str(index), f"dataset-{index}") for index in range(20)]
    rendered = _menu_render("Choose dataset", options, 10).plain
    assert "dataset-10" in rendered
    assert "dataset-0" not in rendered
    assert "↑" in rendered
    assert "↓" in rendered


def test_long_score_level_is_compacted_for_picker() -> None:
    compact = _compact_option_label(
        "A very long\nlevel description " + "with repeated detail " * 10
    )
    assert "\n" not in compact
    assert len(compact) == 64
    assert compact.endswith("…")


def test_certainty_bar_reports_inverse_mean_ambiguity() -> None:
    rendered = _certainty_bar(0.75, width=4)
    assert rendered.plain == "[███░] 75.0% certain"


def test_current_uncertainty_is_shown_after_pool_evaluation() -> None:
    predictions = [
        Prediction(story_id="a", probability=0.5),
        Prediction(story_id="b", probability=0.0),
    ]
    with console.capture() as capture:
        _show_current_uncertainty(predictions)
    output = capture.get()
    assert "Current fixed-pool uncertainty" in output
    assert "50.0% certain" in output
    assert "Mean uncertainty" in output


def test_multiclass_gepa_score_is_rendered_as_one_compact_row() -> None:
    candidate = MulticlassCandidateSpec(
        instructions="Classify it.",
        criteria={"tech": "Technology", "culture": "Culture"},
    )
    current = MulticlassMetrics(
        correct=3,
        total=5,
        accuracy=0.6,
        macro_f1=0.55,
        per_class={
            label: ClassMetrics(precision=0.5, recall=0.5, f1=0.5, support=1)
            for label in candidate.criteria
        },
    )
    proposed = MulticlassMetrics(
        correct=5,
        total=5,
        accuracy=1.0,
        macro_f1=1.0,
        per_class={
            label: ClassMetrics(precision=1.0, recall=1.0, f1=1.0, support=1)
            for label in candidate.criteria
        },
    )
    report = RoundReport(
        previous_candidate=candidate,
        proposed_candidate=candidate,
        previous_metrics=current,
        proposed_metrics=proposed,
        previous_ambiguity={},
        proposed_ambiguity={},
        total_metric_calls=10,
        configured_metric_budget=300,
    )

    table = _metrics_table(report)
    with console.capture() as capture:
        console.print(table)
    output = capture.get()
    assert table.row_count == 1
    assert "0.550 · 3/5 correct" in output
    assert "1.000 · 5/5 correct" in output
    assert "+0.450" in output
    assert "tech F1" not in output


def test_score_gepa_metric_is_rendered_as_one_compact_row() -> None:
    candidate = ScoreCandidateSpec(
        instructions="How severe is this?",
        levels=["Cosmetic", "Degraded", "Blocking"],
    )
    report = RoundReport(
        previous_candidate=candidate,
        proposed_candidate=candidate,
        previous_metrics=ScoreMetrics(
            fit_score=0.7,
            mae=0.6,
            normalized_mae=0.3,
            rounded_correct=3,
            total=5,
            rounded_accuracy=0.6,
        ),
        proposed_metrics=ScoreMetrics(
            fit_score=0.9,
            mae=0.2,
            normalized_mae=0.1,
            rounded_correct=5,
            total=5,
            rounded_accuracy=1.0,
        ),
        previous_ambiguity={},
        proposed_ambiguity={},
        total_metric_calls=10,
        configured_metric_budget=300,
    )

    table = _metrics_table(report)
    with console.capture() as capture:
        console.print(table)

    output = capture.get()
    assert table.row_count == 1
    assert "Ordinal fit" in output
    assert "0.700 · MAE 0.600" in output
    assert "0.900 · MAE 0.200" in output
    assert "+0.200" in output


def test_label_context_identifies_uncertainty_and_random_audit_samples() -> None:
    assert _acquisition_context(
        Prediction(story_id="a", probability=0.51), "ambiguous"
    ) == ("Uncertainty sample · 98.0% uncertain · Model true probability 51.0%")
    assert _acquisition_context(
        Prediction(story_id="a", probability=0.10), "exploration"
    ) == ("Random audit sample · 20.0% uncertain · Model true probability 10.0%")


def test_label_metadata_is_rendered_outside_content_card() -> None:
    with console.capture() as capture:
        _render_story(
            {"content": "Story body"},
            1,
            5,
            Prediction(story_id="a", probability=0.51),
            "ambiguous",
        )
    output = capture.get()
    card_start = output.index("╭")
    assert output.index("98.0% uncertain") < card_start
    assert output.index("Story body") > card_start


def test_json_fields_are_pretty_printed_inside_label_cards() -> None:
    compact = '[{"toolName":"searchWeb","input":{"query":"aircraft"}}]'

    assert _pretty_display_value(compact) == (
        '[\n'
        '  {\n'
        '    "toolName": "searchWeb",\n'
        '    "input": {\n'
        '      "query": "aircraft"\n'
        '    }\n'
        '  }\n'
        ']'
    )

    with console.capture() as capture:
        _render_story(
            {"tool_calls": compact},
            1,
            1,
            Prediction(story_id="a", probability=0.5),
            "ambiguous",
        )
    output = capture.get()
    assert '    "toolName": "searchWeb"' in output
    assert '      "query": "aircraft"' in output


def test_non_json_card_fields_are_unchanged() -> None:
    assert _pretty_display_value("[not actually JSON]") == "[not actually JSON]"
    assert _pretty_display_value("ordinary text") == "ordinary text"


def test_optimize_replaces_climb_as_the_visible_command() -> None:
    help_result = CliRunner().invoke(cli.app, ["--help"])
    legacy_result = CliRunner().invoke(cli.app, ["climb", "--help"])

    assert help_result.exit_code == 0
    assert "optimize" in help_result.output
    assert "climb" not in help_result.output
    assert legacy_result.exit_code == 0
    assert "climb [OPTIONS]" in legacy_result.output


def test_bare_command_guides_user_through_new_run(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    captured = {}

    def fake_climb(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cli, "optimize", fake_climb)
    monkeypatch.setattr(cli, "_example_preset_for_path", lambda *_args: None)
    monkeypatch.setattr(
        cli, "_choose_dataset", lambda _root: Path("sample_data/hn-stories.csv")
    )
    result = CliRunner().invoke(
        cli.app,
        input="n\n\na\nIs the post related to aviation?\n\n\nstart\n",
    )
    assert result.exit_code == 0
    assert captured["data"].name == "hn-stories.csv"
    assert captured["question"] == "Is the post related to aviation?"
    assert captured["reflection_model"] == "openai/gpt-5.6-luna"
    assert captured["columns"] is None
    assert captured["all_columns_concatenated"] is True
    assert captured["pool_size"] == 1000
    assert "jev-align" in result.output
    assert "An experiment from sutro.sh" in result.output
    assert "https://github.com/sutro-sh/jev-align" in result.output
    assert "https://github.com/sutro-sh/jev-align#readme" in result.output


def test_bare_command_can_open_existing_functions(monkeypatch) -> None:
    opened = []
    created = []
    monkeypatch.setattr(cli, "functions_command", lambda: opened.append(True))
    monkeypatch.setattr(cli, "_new_run_wizard", lambda: created.append(True))

    result = CliRunner().invoke(cli.app, input="e\n")

    assert result.exit_code == 0
    assert opened == [True]
    assert created == []
    assert "Optimize a new AI Function" in result.output
    assert "Show existing AI" in result.output
    assert "Functions" in result.output


def test_wizard_row_default_uses_all_when_dataset_is_under_1000(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "small.csv"
    path.write_text("text\n" + "\n".join(f"row {index}" for index in range(7)))
    captured_options = []

    def select_default(_prompt, options):
        captured_options.extend(options)
        return options[0][0]

    monkeypatch.setattr(cli, "_select_option", select_default)

    selected = cli._choose_pool_size(path)

    assert selected == 7
    assert captured_options == [
        ("d", "Use all 7 rows (default)"),
        ("f", "Select the first N rows"),
    ]


def test_wizard_can_select_first_n_rows(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    captured = {}

    monkeypatch.setattr(cli, "optimize", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(cli, "_example_preset_for_path", lambda *_args: None)
    monkeypatch.setattr(
        cli, "_choose_dataset", lambda _root: Path("sample_data/hn-stories.csv")
    )
    result = CliRunner().invoke(
        cli.app,
        input="n\nf\n500\na\nIs it aviation?\n\n\nstart\n",
    )
    assert result.exit_code == 0
    assert captured["pool_size"] == 500


def test_wizard_can_select_specific_columns(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    captured = {}

    def fake_climb(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cli, "optimize", fake_climb)
    monkeypatch.setattr(cli, "_example_preset_for_path", lambda *_args: None)
    monkeypatch.setattr(
        cli, "_choose_dataset", lambda _root: Path("sample_data/hn-stories.csv")
    )
    result = CliRunner().invoke(
        cli.app,
        input="n\n\ns\ntitle,text\nIs it aviation?\n\n\nstart\n",
    )
    assert result.exit_code == 0
    assert captured["columns"] == ["title", "text"]
    assert captured["all_columns_concatenated"] is False


def test_wizard_can_choose_a_different_reflection_model(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    captured = {}

    def fake_climb(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cli, "optimize", fake_climb)
    monkeypatch.setattr(cli, "_example_preset_for_path", lambda *_args: None)
    monkeypatch.setattr(
        cli, "_choose_dataset", lambda _root: Path("sample_data/hn-stories.csv")
    )
    result = CliRunner().invoke(
        cli.app,
        input="n\n\na\nIs it aviation?\n\nc\n3\nstart\n",
    )
    assert result.exit_code == 0
    assert captured["reflection_model"] == "openai/gpt-5.6-terra"


def test_reflection_models_follow_configured_provider_order() -> None:
    models = cli._available_reflection_models(
        {
            "OPENAI_API_KEY": "openai",
            "ANTHROPIC_API_KEY": "anthropic",
            "GEMINI_API_KEY": "gemini",
        }
    )

    assert models[0] == ("GPT-5.6 Luna", "openai/gpt-5.6-luna")
    assert models[5] == (
        "Claude Haiku 4.5",
        "anthropic/claude-haiku-4-5-20251001",
    )
    assert models[9] == ("Gemini 3.8 Flash", "gemini/gemini-3.8-flash")


def test_reflection_default_falls_back_by_provider_priority() -> None:
    assert cli._default_reflection_model(
        {"CLAUDE_API_KEY": "claude", "GEMINI_API_KEY": "gemini"}
    ) == "anthropic/claude-haiku-4-5-20251001"
    assert cli._default_reflection_model(
        {"GEMINI_API_KEY": "gemini"}
    ) == "gemini/gemini-3.8-flash"
    assert cli._default_reflection_model({}) is None


def test_claude_api_key_is_aliased_for_litellm() -> None:
    environment = {"CLAUDE_API_KEY": "claude-key"}

    cli._configure_provider_key_aliases(environment)

    assert environment["ANTHROPIC_API_KEY"] == "claude-key"


def test_reflection_picker_defaults_to_haiku_with_only_claude_key(
    monkeypatch,
) -> None:
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CLAUDE_API_KEY", "claude-key")
    monkeypatch.setattr(console, "input", lambda _prompt: "")

    assert cli._choose_reflection_model() == (
        "anthropic/claude-haiku-4-5-20251001"
    )


def test_wizard_builds_multiclass_definition(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    captured = {}

    def fake_climb(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cli, "optimize", fake_climb)
    monkeypatch.setattr(cli, "_example_preset_for_path", lambda *_args: None)
    monkeypatch.setattr(
        cli, "_choose_dataset", lambda _root: Path("sample_data/hn-stories.csv")
    )
    result = CliRunner().invoke(
        cli.app,
        input=(
            "n\n\na\nWhat kind of post is this?\nm\naviation,space,other\n"
            "Aircraft and atmospheric flight\nSpacecraft and astronomy\nEverything else\n"
            "\nstart\n"
        ),
    )
    assert result.exit_code == 0
    assert "describe what should count as each one" in result.output
    assert "What should count as “aviation”?" in result.output
    assert captured["classes"] == [
        "aviation=Aircraft and atmospheric flight",
        "space=Spacecraft and astronomy",
        "other=Everything else",
    ]


def test_wizard_builds_multilabel_definition(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    captured = {}

    monkeypatch.setattr(cli, "optimize", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(cli, "_example_preset_for_path", lambda *_args: None)
    monkeypatch.setattr(
        cli, "_choose_dataset", lambda _root: Path("sample_data/hn-stories.csv")
    )
    result = CliRunner().invoke(
        cli.app,
        input=(
            "n\n\na\nApply every relevant topic.\nl\ntech,startups,culture\n"
            "Technology\nStartup ecosystem\nCulture and media\n\nstart\n"
        ),
    )
    assert result.exit_code == 0
    assert captured["multilabel"] is True
    assert captured["classes"] == [
        "tech=Technology",
        "startups=Startup ecosystem",
        "culture=Culture and media",
    ]


def test_wizard_builds_score_definition(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    captured = {}

    monkeypatch.setattr(cli, "optimize", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(cli, "_example_preset_for_path", lambda *_args: None)
    monkeypatch.setattr(
        cli, "_choose_dataset", lambda _root: Path("sample_data/hn-stories.csv")
    )
    result = CliRunner().invoke(
        cli.app,
        input=(
            "n\n\na\nHow severe is this issue?\ns\n3\nCosmetic only\n"
            "Degraded but usable\nBlocking with no workaround\n\nstart\n"
        ),
    )

    assert result.exit_code == 0
    assert captured["score_levels"] == [
        "Cosmetic only",
        "Degraded but usable",
        "Blocking with no workaround",
    ]
    assert captured["classes"] is None


def test_example_datasets_are_marked_and_ordered(monkeypatch, tmp_path: Path) -> None:
    paths = [
        tmp_path / "custom.csv",
        tmp_path / "sample_data" / "agent-traces.csv",
        tmp_path / "sample_data" / "support-tickets.csv",
        tmp_path / "sample_data" / "hn-stories.csv",
    ]
    captured = []
    monkeypatch.setattr(cli, "discover_datasets", lambda _root: list(paths))

    def select_first(_prompt, options):
        captured.extend(options)
        return "1"

    monkeypatch.setattr(cli, "_select_option", select_first)

    selected = cli._choose_dataset(tmp_path)

    assert selected.name == "hn-stories.csv"
    assert [label for _, label in captured[:3]] == [
        "Example · Hacker News posts (sample_data/hn-stories.csv)",
        "Example · Support tickets (sample_data/support-tickets.csv)",
        "Example · Agent traces (sample_data/agent-traces.csv)",
    ]


def test_example_datasets_use_preconfigured_flows(monkeypatch) -> None:
    captured = []
    selected_path = {"value": Path("sample_data/hn-stories.csv").resolve()}
    monkeypatch.setattr(cli, "_choose_dataset", lambda _root: selected_path["value"])
    monkeypatch.setattr(cli, "_choose_pool_size", lambda _path: 500)
    monkeypatch.setattr(
        cli, "_choose_reflection_model", lambda: "openai/gpt-5.6-luna"
    )
    monkeypatch.setattr(cli, "_select_option", lambda _prompt, _options: "s")
    monkeypatch.setattr(cli, "optimize", lambda **kwargs: captured.append(kwargs))

    for filename in ("hn-stories.csv", "support-tickets.csv", "agent-traces.csv"):
        selected_path["value"] = (Path("sample_data") / filename).resolve()
        cli._new_run_wizard()

    hacker_news, support, traces = captured
    assert hacker_news["question"] == "Is the hacker news post related to AI?"
    assert hacker_news["classes"] is None
    assert hacker_news["columns"] == ["title", "text", "url"]

    assert support["question"] == "Categorize the support ticket"
    assert len(support["classes"]) == 7
    assert support["classes"][0].startswith("Billing & invoices=")
    assert support["columns"] == ["instruction"]

    assert traces["question"] == "Did the agent trace pass?"
    assert traces["true_label"] == "Pass"
    assert traces["false_label"] == "Fail"
    assert traces["columns"] == ["user_input", "state", "tool_calls", "output"]


def test_binary_picker_can_present_pass_and_fail(monkeypatch) -> None:
    candidate = CandidateSpec(
        instructions="Did the trace pass?",
        true_criteria="The trace passed.",
        false_criteria="The trace failed.",
    )
    monkeypatch.setattr(console, "input", lambda _prompt: "fail")

    assert cli._prompt_label(
        candidate,
        can_go_back=False,
        binary_labels=("Pass", "Fail"),
    ) is False


def _saved_binary_function(tmp_path: Path) -> cli.SavedFunction:
    candidate = CandidateSpec(
        instructions="Is the post related to aviation?",
        true_criteria="Aircraft or flight is the main subject.",
        false_criteria="Aircraft or flight is not the main subject.",
    )
    run_directory = tmp_path / ".jev-align" / "runs" / "20260919-aviation"
    store = RunStore(run_directory)
    state = RunState(
        run_id=run_directory.name,
        source_path=str(tmp_path / "posts.csv"),
        source_sha256="abc",
        selected_columns=["title", "body"],
        pool_size=100,
        reflection_model="openai/gpt-5.6-luna",
        current_candidate=candidate,
        history=[
            CandidateHistory(round_number=0, candidate=candidate, decision="seed"),
            CandidateHistory(
                round_number=1,
                candidate=candidate,
                decision="accepted",
                fit_f1=0.9,
            ),
        ],
    )
    store.initialize(state)
    return cli.SavedFunction(run_directory, state, label_count=5, certainty=0.8)


def test_function_cards_show_saved_run_details(tmp_path: Path) -> None:
    saved = _saved_binary_function(tmp_path)

    with console.capture() as capture:
        console.print(cli._function_browser_render([saved], 0))

    output = capture.get()
    assert "Is the post related to aviation?" in output
    assert "Binary" in output
    assert "100 rows · 5 labeled" in output
    assert "0.900 (training labels)" in output
    assert "80.0%" in output


def test_show_latest_optimized_includes_setup_signature_and_scoring(
    tmp_path: Path,
) -> None:
    saved = _saved_binary_function(tmp_path)

    with console.capture() as capture:
        cli._show_latest_optimized(saved)

    output = capture.get()
    assert "Setup" in output
    assert "Signature" in output
    assert "(title, body) → True | False" in output
    assert "Latest optimized AI Function" in output
    assert "Scoring" in output
    assert "0.900" in output


def test_run_results_are_row_aligned_jsonl(tmp_path: Path, monkeypatch) -> None:
    saved = _saved_binary_function(tmp_path)
    stories = [Story(id="row-000001", row_number=1, fields={"text": "Plane"})]
    predictions = [Prediction(story_id="row-000001", probability=0.8)]
    monkeypatch.chdir(tmp_path)

    output = cli._write_function_results(
        saved, tmp_path / "new.csv", stories, predictions
    )
    payload = __import__("json").loads(output.read_text())

    assert payload["row_number"] == 1
    assert payload["input"] == {"text": "Plane"}
    assert payload["result"] == {
        "label": "True",
        "value": True,
        "true_probability": 0.8,
    }
