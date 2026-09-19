from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import termios
import tty
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Annotated
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import typer
from packaging.version import InvalidVersion, Version
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .acquisition import ambiguity, ambiguity_summary, prediction_ambiguity
from .data import (
    dataset_columns,
    dataset_row_count,
    discover_datasets,
    load_stories,
    source_sha256,
)
from .backends import (
    backend_credential_error,
    backend_display_name,
    create_backend,
    evaluate_with_progress,
    validate_backend_for_task,
)
from .models import (
    BackendConfig,
    BinaryMetrics,
    CandidateHistory,
    CandidateSpec,
    MulticlassCandidateSpec,
    MulticlassMetrics,
    MultilabelCandidateSpec,
    MultilabelCriteria,
    MultilabelMetrics,
    Prediction,
    RunState,
    ScoreCandidateSpec,
    ScoreMetrics,
    Story,
    TaskSpec,
)
from .metrics import nearest_score_level
from .persistence import RunStore
from .session import ClimbSession, RoundReport, candidate_diff

app = typer.Typer(no_args_is_help=False, pretty_exceptions_show_locals=False)
console = Console()
PYPI_PROJECT_NAME = "jev-align"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PYPI_PROJECT_NAME}/json"
UPDATE_CHECK_TIMEOUT_SECONDS = 2.0
DEFAULT_REFLECTION_MODEL = "openai/gpt-5.6-luna"
ANTHROPIC_DEFAULT_REFLECTION_MODEL = "anthropic/claude-haiku-4-5-20251001"
GEMINI_DEFAULT_REFLECTION_MODEL = "gemini/gemini-3.8-flash"
REFLECTION_MODELS = {
    "openai": [
        ("GPT-5.6 Luna", DEFAULT_REFLECTION_MODEL),
        ("GPT-5.6 Sol", "openai/gpt-5.6-sol"),
        ("GPT-5.6 Terra", "openai/gpt-5.6-terra"),
        ("GPT-6 Astra", "openai/gpt-6-astra"),
        ("GPT-5.5", "openai/gpt-5.5"),
    ],
    "anthropic": [
        ("Claude Haiku 4.5", ANTHROPIC_DEFAULT_REFLECTION_MODEL),
        ("Claude Sonnet 4.6", "anthropic/claude-sonnet-4-6"),
        ("Claude Opus 4.8", "anthropic/claude-opus-4-8"),
        ("Claude Fable 5", "anthropic/claude-fable-5"),
    ],
    "gemini": [
        ("Gemini 3.8 Flash", GEMINI_DEFAULT_REFLECTION_MODEL),
        ("Gemini 3.7 Flash", "gemini/gemini-3.7-flash"),
        ("Gemini 3.1 Pro Preview", "gemini/gemini-3.1-pro-preview"),
    ],
}
BACK_LABEL = object()
LABEL_COLORS = (
    "bright_cyan",
    "bright_magenta",
    "bright_green",
    "bright_yellow",
    "bright_blue",
    "cyan",
    "magenta",
)


@dataclass(frozen=True)
class SavedFunction:
    directory: Path
    state: RunState
    label_count: int
    certainty: float | None


@dataclass(frozen=True)
class ExamplePreset:
    filename: str
    name: str
    question: str
    task_type: str
    columns: tuple[str, ...]
    classes: tuple[tuple[str, str], ...] = ()
    true_criteria: str | None = None
    false_criteria: str | None = None
    true_label: str = "True"
    false_label: str = "False"


EXAMPLE_PRESETS = (
    ExamplePreset(
        filename="hn-stories.csv",
        name="Hacker News posts",
        question="Is the hacker news post related to AI?",
        task_type="binary",
        columns=("title", "text", "url"),
        true_criteria=(
            "The post is substantively about artificial intelligence, machine "
            "learning, AI models, AI systems, or their development, use, or impact."
        ),
        false_criteria=(
            "AI is absent, merely incidental, or not a meaningful subject of the post."
        ),
    ),
    ExamplePreset(
        filename="support-tickets.csv",
        name="Support tickets",
        question="Categorize the support ticket",
        task_type="multiclass",
        columns=("instruction",),
        classes=(
            ("Billing & invoices", "Bills, invoices, receipts, or invoice downloads."),
            (
                "Payments, refunds & fees",
                "Payment methods or failures, refunds, reimbursements, rebates, "
                "compensation, withdrawal fees, or termination charges.",
            ),
            (
                "Orders & delivery",
                "Buying, changing, cancelling, or tracking an order; products, "
                "shipping addresses, delivery availability, methods, or timing.",
            ),
            (
                "Accounts & access",
                "Opening, changing, using, or closing an account; signup errors, "
                "passwords, PINs, user keys, or account recovery.",
            ),
            (
                "Newsletter",
                "Subscribing to, receiving, managing, or unsubscribing from a newsletter.",
            ),
            (
                "Customer service & complaints",
                "Contacting support or a human agent, support hours, formal complaints, "
                "claims, or escalations not primarily requesting money back.",
            ),
            (
                "Reviews & feedback",
                "Writing or submitting a product or service review, comment, or feedback.",
            ),
        ),
    ),
    ExamplePreset(
        filename="agent-traces.csv",
        name="Agent traces",
        question="Did the agent trace pass?",
        task_type="binary",
        columns=("user_input", "state", "tool_calls", "output"),
        true_criteria=(
            "Pass: the tool use and final output correctly and usefully satisfy the "
            "user's request, honor its important constraints, and ground material "
            "claims in the available evidence."
        ),
        false_criteria=(
            "Fail: the trace misses an important requirement, uses tools incorrectly "
            "or incompletely, makes unsupported or incorrect claims, or produces an "
            "output that does not usefully satisfy the request."
        ),
        true_label="Pass",
        false_label="Fail",
    ),
)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Build and optimize AI Functions with pluggable evaluators and GEPA."""
    if _maybe_offer_update():
        raise typer.Exit()
    _configure_provider_key_aliases()
    if ctx.invoked_subcommand is None:
        _show_startup_banner()
        action = _select_option(
            "What would you like to do?",
            [
                ("n", "Optimize a new AI Function"),
                ("e", "Show existing AI Functions"),
            ],
        )
        if action == "e":
            functions_command()
        else:
            _new_run_wizard()


def _installed_package() -> tuple[str, bool] | None:
    """Return the installed version and whether this is an editable checkout."""
    try:
        package = distribution(PYPI_PROJECT_NAME)
    except PackageNotFoundError:
        return None

    editable = False
    direct_url = package.read_text("direct_url.json")
    if direct_url:
        try:
            metadata = json.loads(direct_url)
            editable = bool(metadata.get("dir_info", {}).get("editable"))
        except (AttributeError, json.JSONDecodeError, TypeError):
            pass
    return package.version, editable


def _latest_pypi_version(installed_version: str) -> str | None:
    request = Request(
        PYPI_JSON_URL,
        headers={
            "Accept": "application/json",
            "User-Agent": (
                f"jeva/{installed_version} "
                "(+https://github.com/sutro-sh/jev-align)"
            ),
        },
    )
    try:
        with urlopen(request, timeout=UPDATE_CHECK_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
        latest = payload["info"]["version"]
        return latest if isinstance(latest, str) else None
    except (
        HTTPError,
        URLError,
        TimeoutError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ):
        return None


def _update_checks_disabled() -> bool:
    value = os.environ.get("JEVA_DISABLE_UPDATE_CHECK", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _running_in_virtual_environment() -> bool:
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    return sys.prefix != base_prefix


def _upgrade_commands() -> list[list[str]]:
    commands: list[list[str]] = []
    uv = shutil.which("uv")
    if uv is not None and _running_in_virtual_environment():
        commands.append(
            [
                uv,
                "pip",
                "install",
                "--python",
                sys.executable,
                "--upgrade",
                PYPI_PROJECT_NAME,
            ]
        )
    commands.append(
        [sys.executable, "-m", "pip", "install", "--upgrade", PYPI_PROJECT_NAME]
    )
    return commands


def _maybe_offer_update(*, interactive: bool | None = None) -> bool:
    """Offer an in-place upgrade and return True when the process should exit."""
    if _update_checks_disabled():
        return False
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive:
        return False

    installed = _installed_package()
    if installed is None:
        return False
    installed_version, editable = installed
    if editable:
        return False

    latest_version = _latest_pypi_version(installed_version)
    if latest_version is None:
        return False
    try:
        update_available = Version(latest_version) > Version(installed_version)
    except InvalidVersion:
        return False
    if not update_available:
        return False

    console.print(
        Panel(
            Group(
                Text("A newer version of jev-align is available.", style="bold"),
                Text.assemble(
                    (installed_version, "yellow"),
                    ("  →  ", "dim"),
                    (latest_version, "bold bright_green"),
                ),
                Text("Press Enter to upgrade now.", style="dim"),
            ),
            title="[bold bright_cyan]Update available[/bold bright_cyan]",
            border_style="bright_cyan",
            padding=(1, 2),
        )
    )
    choice = _select_option(
        "Update jev-align?",
        [("u", "Upgrade now (default)"), ("s", "Skip for now")],
    )
    if choice == "s":
        return False

    commands = _upgrade_commands()
    upgraded = False
    last_error: OSError | None = None
    for index, command in enumerate(commands):
        installer = "uv" if command[1:3] == ["pip", "install"] else "pip"
        console.print(f"Upgrading jev-align with {installer}…", style="cyan")
        last_error = None
        try:
            result = subprocess.run(command, check=False)
        except OSError as error:
            last_error = error
            result = None
        if result is not None and result.returncode == 0:
            upgraded = True
            break
        if index < len(commands) - 1:
            console.print("uv upgrade failed; trying pip…", style="yellow")

    if not upgraded:
        detail = (
            f"Could not start the upgrade: {last_error}"
            if last_error is not None
            else "The upgrade command did not complete successfully."
        )
        manual_command = commands[-1]
        console.print(
            Panel(
                f"{detail}\n\nRun manually:\n  {' '.join(manual_command)}",
                title="[bold red]Upgrade failed[/bold red]",
                border_style="red",
            )
        )
        return False

    console.print(
        Panel(
            f"jev-align {latest_version} is installed. Restart `jeva` to use it.",
            title="[bold green]Upgrade complete[/bold green]",
            border_style="green",
        )
    )
    return True


def _prompt_value(prompt: str, *, default: str | None = None) -> str:
    while True:
        suffix = f" (default: {default})" if default is not None else ""
        value = console.input(f"{prompt}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        console.print("A value is required.", style="red")


def _show_startup_banner() -> None:
    logo = Text(
        r"""     _ _______     __       _    _     ___ ____ _   _
    | | ____\ \   / /      / \  | |   |_ _/ ___| \ | |
 _  | |  _|  \ \ / /_____ / _ \ | |    | | |  _|  \| |
| |_| | |___  \ V /______/ ___ \| |___ | | |_| | |\  |
 \___/|_____|  \_/      /_/   \_\_____|___\____|_| \_|""",
        style="bold cyan",
    )
    tagline = Text.from_markup(
        "[bold]An experiment from [link=https://sutro.sh]sutro.sh[/link][/bold]"
    )
    package_name = Text("jev-align", style="bold bright_cyan")
    github_url = "https://github.com/sutro-sh/jev-align"
    readme_url = "https://github.com/sutro-sh/jev-align#readme"
    github_link = Text.assemble(
        ("GitHub  ", "dim"),
        (github_url, f"bright_blue underline link {github_url}"),
    )
    readme_link = Text.assemble(
        ("README  ", "dim"),
        (readme_url, f"bright_blue underline link {readme_url}"),
    )
    console.print(
        Panel(
            Group(
                Align.center(logo),
                Align.center(package_name),
                Text(),
                Align.center(tagline),
                Align.center(github_link),
                Align.center(readme_link),
            ),
            border_style="bright_cyan",
            padding=(1, 2),
        )
    )


def _packaged_example_directory() -> Path:
    return Path(__file__).resolve().parent / "sample_data"


def _packaged_example_paths() -> list[Path]:
    directory = _packaged_example_directory()
    return [
        path
        for preset in EXAMPLE_PRESETS
        if (path := directory / preset.filename).is_file()
    ]


def _example_preset_for_path(path: Path, root: Path) -> ExamplePreset | None:
    resolved = path.resolve()
    if resolved.parent == _packaged_example_directory().resolve():
        return next(
            (preset for preset in EXAMPLE_PRESETS if preset.filename == path.name),
            None,
        )
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError:
        return None
    if len(relative.parts) != 2 or relative.parts[0] != "sample_data":
        return None
    return next(
        (preset for preset in EXAMPLE_PRESETS if preset.filename == relative.name),
        None,
    )


def _show_example_setup(preset: ExamplePreset) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Example", preset.name)
    table.add_row("Question", preset.question)
    table.add_row(
        "Task",
        "Multiple choice"
        if preset.task_type == "multiclass"
        else f"Binary ({preset.true_label}/{preset.false_label})",
    )
    table.add_row("Columns", ", ".join(preset.columns))
    if preset.classes:
        table.add_row("Labels", ", ".join(name for name, _ in preset.classes))
    console.print(Panel(table, title="Example setup"))


def _choose_advanced_settings(
    batch_size: int, holdout: bool, metric_budget: int
) -> tuple[int, bool, int]:
    batch_key = _select_option(
        "Training annotations per round",
        [
            ("a", "5 annotations (default)"),
            ("b", "10 annotations"),
            ("c", "15 annotations"),
            ("d", "20 annotations"),
        ],
        initial_key={5: "a", 10: "b", 15: "c", 20: "d"}.get(batch_size, "a"),
    )
    selected_batch = {"a": 5, "b": 10, "c": 15, "d": 20}[batch_key]
    holdout_choice = _select_option(
        "Reserve 20% of rows as a held-out evaluation set?",
        [
            ("n", "No (default)"),
            ("y", "Yes — add 20% extra held-out annotations each round"),
        ],
        initial_key="y" if holdout else "n",
    )
    while True:
        raw_budget = _prompt_value(
            "Maximum GEPA metric calls", default=str(metric_budget)
        )
        try:
            selected_budget = int(raw_budget)
        except ValueError:
            console.print("Enter a positive whole number.", style="red")
            continue
        if selected_budget > 0:
            break
        console.print("Enter a positive whole number.", style="red")
    return selected_batch, holdout_choice == "y", selected_budget


def _holdout_story_ids(
    stories: list[Story], *, fraction: float, seed: int
) -> list[str]:
    if fraction == 0.0:
        return []
    count = max(1, round(len(stories) * fraction))
    story_ids = [story.id for story in stories]
    random.Random(seed).shuffle(story_ids)
    return story_ids[:count]


def _new_run_wizard() -> None:
    console.print(Panel("Create a new AI Function", title="jev-align"))
    root = Path.cwd()
    data = _choose_dataset(root)
    pool_size = _choose_pool_size(data)
    available_columns = dataset_columns(data)
    preset = _example_preset_for_path(data, root)
    true_criteria: str | None = None
    false_criteria: str | None = None
    true_label = "True"
    false_label = "False"
    if preset is not None:
        missing = [column for column in preset.columns if column not in available_columns]
        if missing:
            raise typer.BadParameter(
                f"example dataset is missing expected columns: {', '.join(missing)}"
            )
        columns = list(preset.columns)
        column_mode = "selected"
        question = preset.question
        task_type = "m" if preset.task_type == "multiclass" else "b"
        class_criteria = [
            f"{name}={description}" for name, description in preset.classes
        ] or None
        score_levels: list[str] | None = None
        true_criteria = preset.true_criteria
        false_criteria = preset.false_criteria
        true_label = preset.true_label
        false_label = preset.false_label
        _show_example_setup(preset)
    else:
        column_choice = _select_option(
            "Row fields",
            [
                ("a", "Use all columns, concatenated"),
                ("s", "Select specific columns"),
            ],
        )
        if column_choice == "a":
            columns = available_columns
            column_mode = "all_concatenated"
        else:
            columns = _choose_columns(available_columns, default_all=False)
            column_mode = "selected"
        question = _prompt_value("Starting question")
        task_type = _select_option(
            "Task type",
            [
                ("b", "Binary (true/false)"),
                ("m", "Multiple choice (exactly one label per row)"),
                ("l", "Multilabel (zero or more labels per row)"),
                ("s", "Score (ordered descriptive levels)"),
            ],
        )
        class_criteria = None
        score_levels = None
        if task_type in {"m", "l"}:
            if task_type == "m":
                console.print(
                    "[dim]Define the possible labels, then describe what should count "
                    "as each one. The model chooses exactly one label per row.[/dim]"
                )
            else:
                console.print(
                    "[dim]Define independent labels that may overlap. During labeling "
                    "you may select zero, one, or several labels for each row.[/dim]"
                )
            while True:
                names = [
                    item.strip()
                    for item in _prompt_value("Label names (comma-separated)").split(",")
                    if item.strip()
                ]
                minimum = 2 if task_type == "m" else 1
                if len(names) >= minimum and len(set(names)) == len(names):
                    break
                console.print(
                    f"Provide at least {minimum} unique label name(s).", style="red"
                )
            class_criteria = [
                f"{name}={_prompt_value(f'What should count as “{name}”?', default=f'Rows primarily about {name}')}"
                for name in names
            ]
        elif task_type == "s":
            console.print(
                "[dim]Define 2–10 concrete levels from lowest to highest. Describe "
                "situations at each level rather than using numbers or vague degrees.[/dim]"
            )
            while True:
                raw_count = _prompt_value("Number of levels", default="3")
                try:
                    level_count = int(raw_count)
                except ValueError:
                    console.print("Enter a whole number from 2 to 10.", style="red")
                    continue
                if 2 <= level_count <= 10:
                    break
                console.print("Enter a whole number from 2 to 10.", style="red")
            score_levels = [
                _prompt_value(
                    f"Level {index}"
                    + (
                        " (lowest)"
                        if index == 0
                        else " (highest)"
                        if index == level_count - 1
                        else ""
                    )
                )
                for index in range(level_count)
            ]
    reflection_model = _choose_reflection_model()
    batch_size = 5
    holdout = False
    metric_budget = 300
    while True:
        action = _select_option(
            "Create this AI Function",
            [("c", "Create"), ("a", "Advanced")],
        )
        if action == "c":
            break
        batch_size, holdout, metric_budget = _choose_advanced_settings(
            batch_size, holdout, metric_budget
        )
        extra = round(batch_size * 0.2) if holdout else 0
        console.print(
            f"[dim]{batch_size} training annotations per round"
            + (f" + {extra} held-out" if holdout else " · no held-out set")
            + f" · max {metric_budget} GEPA metric calls"
            + "[/dim]"
        )
    optimize(
        data=data,
        question=question,
        reflection_model=reflection_model,
        columns=None if column_mode == "all_concatenated" else columns,
        all_columns_concatenated=column_mode == "all_concatenated",
        classes=class_criteria,
        score_levels=score_levels,
        multilabel=task_type == "l",
        pool_size=pool_size,
        batch_size=batch_size,
        holdout=holdout,
        metric_budget=metric_budget,
        true_criteria=true_criteria,
        false_criteria=false_criteria,
        true_label=true_label,
        false_label=false_label,
    )


def _configure_provider_key_aliases(
    environment: dict[str, str] | None = None,
) -> None:
    """Make the commonly used Claude key name available to LiteLLM."""
    target = os.environ if environment is None else environment
    if not target.get("ANTHROPIC_API_KEY") and target.get("CLAUDE_API_KEY"):
        target["ANTHROPIC_API_KEY"] = target["CLAUDE_API_KEY"]


def _available_reflection_models(
    environment: Mapping[str, str] | None = None,
) -> list[tuple[str, str]]:
    environment = os.environ if environment is None else environment
    models: list[tuple[str, str]] = []
    if environment.get("OPENAI_API_KEY"):
        models.extend(REFLECTION_MODELS["openai"])
    if environment.get("ANTHROPIC_API_KEY") or environment.get("CLAUDE_API_KEY"):
        models.extend(REFLECTION_MODELS["anthropic"])
    if environment.get("GEMINI_API_KEY"):
        models.extend(REFLECTION_MODELS["gemini"])
    return models


def _default_reflection_model(
    environment: Mapping[str, str] | None = None,
) -> str | None:
    models = _available_reflection_models(environment)
    return models[0][1] if models else None


def _choose_reflection_model() -> str:
    models = _available_reflection_models()
    if not models:
        console.print(
            "[yellow]No OpenAI, Anthropic, or Gemini API key was detected. "
            "Enter a custom LiteLLM model.[/yellow]"
        )
        return _prompt_value("LiteLLM model name")
    default_label, default_model = models[0]
    choice = _select_option(
        "GEPA reflection model",
        [
            ("d", f"Use {default_label} (default)"),
            ("c", "Choose a different model"),
        ],
    )
    if choice == "d":
        return default_model
    options = [
        (str(index), label) for index, (label, _) in enumerate(models, start=1)
    ]
    options.append(("m", "Enter a custom LiteLLM model"))
    selected = _select_option("Choose model", options)
    if selected == "m":
        return _prompt_value("LiteLLM model name")
    return models[int(selected) - 1][1]


def _choose_dataset(root: Path) -> Path:
    discovered = discover_datasets(root)
    local_examples = {
        preset.filename
        for path in discovered
        if (preset := _example_preset_for_path(path, root)) is not None
    }
    discovered.extend(
        path
        for path in _packaged_example_paths()
        if path.name not in local_examples
    )
    preset_rank = {preset.filename: index for index, preset in enumerate(EXAMPLE_PRESETS)}
    discovered.sort(
        key=lambda path: (
            preset_rank.get(
                path.name,
                len(EXAMPLE_PRESETS),
            )
            if _example_preset_for_path(path, root) is not None
            else len(EXAMPLE_PRESETS),
            0 if _example_preset_for_path(path, root) is not None else 1,
        )
    )
    while True:
        options = []
        for index, path in enumerate(discovered, start=1):
            preset = _example_preset_for_path(path, root)
            try:
                display_path = str(path.relative_to(root))
            except ValueError:
                display_path = "included"
            label = (
                f"Example · {preset.name} ({display_path})"
                if preset is not None
                else display_path
            )
            options.append((str(index), label))
        options.append(("m", "Enter another path…"))
        choice = _select_option("Choose dataset", options)
        if choice != "m":
            return discovered[int(choice) - 1]
        path = Path(_prompt_value("Dataset path")).expanduser()
        if path.is_file():
            try:
                dataset_columns(path)
            except ValueError as error:
                console.print(str(error), style="red")
            else:
                return path.resolve()
            continue
        console.print(f"Dataset not found: {path}", style="red")


def _choose_pool_size(path: Path) -> int:
    row_count = dataset_row_count(path)
    if row_count == 0:
        raise typer.BadParameter("dataset contains no data rows")
    default_size = min(1000, row_count)
    default_label = (
        "Use the first 1,000 rows (default)"
        if row_count > 1000
        else f"Use all {row_count:,} rows (default)"
    )
    options = [("d", default_label)]
    if row_count > 1000:
        options.append(("a", f"Use all {row_count:,} rows"))
    options.append(("f", "Select the first N rows"))
    choice = _select_option(
        "Dataset rows",
        options,
    )
    if choice == "d":
        return default_size
    if choice == "a":
        return row_count

    while True:
        raw = _prompt_value("Number of rows")
        try:
            selected = int(raw)
        except ValueError:
            console.print("Enter a whole number.", style="red")
            continue
        if selected < 5:
            console.print("Select at least 5 rows.", style="red")
            continue
        if selected > row_count:
            console.print(f"This dataset has only {row_count:,} rows.", style="red")
            continue
        return selected


def _visible_range(count: int, selected: int, window: int = 12) -> tuple[int, int]:
    start = max(0, min(selected - window // 2, count - window))
    return start, min(count, start + window)


def _compact_option_label(value: str, max_chars: int = 64) -> str:
    compact = " ".join(value.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def _multi_menu_render(
    prompt: str,
    options: list[str],
    selected: int,
    enabled: set[int],
    option_styles: list[str] | None = None,
) -> Text:
    rendered = Text(prompt + "  (↑/↓, Space toggle, Enter confirm)\n", style="bold")
    start, end = _visible_range(len(options), selected)
    if start:
        rendered.append(f"    ↑ {start} more\n", style="dim")
    for index in range(start, end):
        label = options[index]
        rendered.append("  > " if index == selected else "    ")
        rendered.append("[x] " if index in enabled else "[ ] ")
        if option_styles:
            base_style = option_styles[index]
            style = f"bold {base_style}" if index == selected else base_style
        else:
            style = "bold cyan" if index == selected else ""
        rendered.append(label, style=style)
        rendered.append("\n")
    if end < len(options):
        rendered.append(f"    ↓ {len(options) - end} more", style="dim")
    else:
        rendered.rstrip()
    return rendered


def _choose_columns(columns: list[str], *, default_all: bool = True) -> list[str]:
    if not console.is_terminal or not sys.stdin.isatty():
        suffix = "; blank selects all" if default_all else ""
        raw = console.input(f"Columns (comma-separated{suffix}): ").strip()
        if not raw and not default_all:
            raise typer.BadParameter("at least one column must be selected")
        selected = columns if not raw else [item.strip() for item in raw.split(",")]
        missing = [item for item in selected if item not in columns]
        if missing:
            raise typer.BadParameter(f"unknown columns: {', '.join(missing)}")
        return selected

    cursor = 0
    enabled = set(range(len(columns))) if default_all else set()
    descriptor = sys.stdin.fileno()
    original = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        rendered = _multi_menu_render("Select columns", columns, cursor, enabled)
        console.print(rendered)
        line_count = rendered.plain.count("\n") + 1
        while True:
            key = sys.stdin.read(1)
            if key == "\x03":
                raise KeyboardInterrupt
            if key in {"\r", "\n"}:
                if enabled:
                    return [
                        column
                        for index, column in enumerate(columns)
                        if index in enabled
                    ]
                continue
            if key == " ":
                if cursor in enabled:
                    enabled.remove(cursor)
                else:
                    enabled.add(cursor)
            elif key == "\x1b":
                sequence = sys.stdin.read(2)
                if sequence == "[A":
                    cursor = (cursor - 1) % len(columns)
                elif sequence == "[B":
                    cursor = (cursor + 1) % len(columns)
                else:
                    continue
            else:
                continue
            sys.stdout.write(f"\x1b[{line_count}A\r\x1b[J")
            sys.stdout.flush()
            rendered = _multi_menu_render("Select columns", columns, cursor, enabled)
            console.print(rendered)
            line_count = rendered.plain.count("\n") + 1
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, original)


def _select_multilabel_labels(
    labels: list[str], *, can_go_back: bool, suggested: set[str] | None = None
) -> list[str] | object:
    """Select zero or more labels; ``b`` navigates back when available."""
    suggested = suggested or set()
    suggested_labels = [label for label in labels if label in suggested]
    if not console.is_terminal or not sys.stdin.isatty():
        while True:
            raw = console.input(
                "Labels (comma-separated; Enter accepts model suggestion; "
                "/none selects none): "
            ).strip()
            if can_go_back and raw.lower() == "b":
                return BACK_LABEL
            if not raw:
                selected = suggested_labels
            elif raw.lower() == "/none":
                selected = []
            else:
                selected = [item.strip() for item in raw.split(",")]
            missing = [item for item in selected if item not in labels]
            if not missing and len(set(selected)) == len(selected):
                return selected
            console.print(
                f"Unknown or repeated labels: {', '.join(missing or selected)}",
                style="red",
            )

    enabled = {index for index, label in enumerate(labels) if label in suggested}
    option_styles = [LABEL_COLORS[index % len(LABEL_COLORS)] for index in range(len(labels))]
    cursor = min(enabled) if enabled else 0
    descriptor = sys.stdin.fileno()
    original = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        rendered = _multi_menu_render(
            "Select labels (none is allowed)",
            labels,
            cursor,
            enabled,
            option_styles,
        )
        console.print(rendered)
        line_count = rendered.plain.count("\n") + 1
        while True:
            key = sys.stdin.read(1)
            if key == "\x03":
                raise KeyboardInterrupt
            if key in {"\r", "\n"}:
                return [label for index, label in enumerate(labels) if index in enabled]
            if can_go_back and key.lower() == "b":
                return BACK_LABEL
            if key == " ":
                if cursor in enabled:
                    enabled.remove(cursor)
                else:
                    enabled.add(cursor)
            elif key == "\x1b":
                sequence = sys.stdin.read(2)
                if sequence == "[A":
                    cursor = (cursor - 1) % len(labels)
                elif sequence == "[B":
                    cursor = (cursor + 1) % len(labels)
                else:
                    continue
            else:
                continue
            sys.stdout.write(f"\x1b[{line_count}A\r\x1b[J")
            sys.stdout.flush()
            rendered = _multi_menu_render(
                "Select labels (none is allowed)",
                labels,
                cursor,
                enabled,
                option_styles,
            )
            console.print(rendered)
            line_count = rendered.plain.count("\n") + 1
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, original)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:40] or "question"


def _new_run_directory(question: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return Path(".jev-align/runs") / f"{timestamp}-{_slug(question)}"


def _task_kind(candidate: object) -> str:
    if isinstance(candidate, ScoreCandidateSpec):
        return "Score"
    if isinstance(candidate, MultilabelCandidateSpec):
        return "Multilabel"
    if isinstance(candidate, MulticlassCandidateSpec):
        return "Multiple choice"
    return "Binary"


def _load_saved_functions(root: Path) -> list[SavedFunction]:
    runs_directory = root / ".jev-align" / "runs"
    saved: list[SavedFunction] = []
    if not runs_directory.is_dir():
        return saved
    for state_path in runs_directory.glob("*/state.json"):
        store = RunStore(state_path.parent)
        try:
            state = store.load_state()
            label_count = len(store.load_labels())
        except (OSError, ValueError):
            continue
        certainty = None
        cache = state_path.parent / "pool-predictions-current.json"
        if cache.exists():
            try:
                payload = json.loads(cache.read_text(encoding="utf-8"))
                predictions = [
                    Prediction.model_validate(item) for item in payload["predictions"]
                ]
                certainty = 1.0 - float(ambiguity_summary(predictions)["mean"])
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        saved.append(
            SavedFunction(
                directory=state_path.parent,
                state=state,
                label_count=label_count,
                certainty=certainty,
            )
        )
    return sorted(saved, key=lambda item: item.state.run_id, reverse=True)


def _latest_fit(saved: SavedFunction) -> float | None:
    accepted = [
        item.fit_f1
        for item in saved.state.history
        if item.decision == "accepted" and item.fit_f1 is not None
    ]
    return accepted[-1] if accepted else None


def _latest_holdout_fit(saved: SavedFunction) -> float | None:
    accepted = [
        item.holdout_score
        for item in saved.state.history
        if item.decision == "accepted" and item.holdout_score is not None
    ]
    return accepted[-1] if accepted else None


def _function_name(saved: SavedFunction) -> str:
    seed = next(
        (item.candidate for item in saved.state.history if item.decision == "seed"),
        saved.state.current_candidate,
    )
    return _compact_option_label(seed.instructions, max_chars=88)


def _function_card(saved: SavedFunction, *, selected: bool = False) -> Panel:
    state = saved.state
    accepted = sum(item.decision == "accepted" for item in state.history)
    fit = _latest_fit(saved)
    details = Table.grid(padding=(0, 2))
    details.add_column(style="dim", no_wrap=True)
    details.add_column()
    task_kind = _task_kind(state.current_candidate)
    if isinstance(state.current_candidate, CandidateSpec):
        task_kind += f" ({state.binary_true_label}/{state.binary_false_label})"
    task_style = {
        "Binary": "cyan",
        "Multiple choice": "magenta",
        "Multilabel": "bright_magenta",
        "Score": "blue",
    }.get(_task_kind(state.current_candidate), "cyan")
    details.add_row("Type", Text(task_kind, style=task_style))
    details.add_row("Source", Path(state.source_path).name)
    details.add_row("Data", f"{state.pool_size:,} rows · {saved.label_count} labeled")
    details.add_row(
        "Progress",
        Text(
            f"{accepted} accepted optimization(s)",
            style="green" if accepted else "dim",
        ),
    )
    if fit is not None:
        details.add_row("Latest fit", f"{fit:.3f} (training labels)")
    holdout_fit = _latest_holdout_fit(saved)
    if holdout_fit is not None:
        details.add_row("Latest held-out", f"{holdout_fit:.3f}")
    if saved.certainty is not None:
        details.add_row("Pool certainty", f"{saved.certainty:.1%}")
    if state.pending_candidate is not None:
        details.add_row("Status", Text("Decision pending", style="bold yellow"))
    else:
        latest_decision = next(
            (
                item.decision
                for item in reversed(state.history)
                if item.decision != "seed"
            ),
            "seed",
        )
        status_label, status_style = {
            "accepted": ("Latest proposal accepted", "green"),
            "rejected": ("Latest proposal rejected", "red"),
            "seed": ("Not optimized yet", "dim"),
        }[latest_decision]
        details.add_row("Status", Text(status_label, style=status_style))
    return Panel(
        details,
        title=_function_name(saved),
        subtitle=state.run_id,
        border_style="bold cyan" if selected else "dim",
        padding=(0, 1),
    )


def _function_browser_render(
    saved: list[SavedFunction], selected: int, window: int = 4
) -> Group:
    start, end = _visible_range(len(saved), selected, window)
    items: list[object] = [
        Text(
            f"AI Functions  {selected + 1}/{len(saved)}  "
            "(↑/↓ scroll, Enter open, q quit)",
            style="bold",
        )
    ]
    if start:
        items.append(Text(f"↑ {start} newer", style="dim"))
    items.extend(
        _function_card(item, selected=index == selected)
        for index, item in enumerate(saved[start:end], start=start)
    )
    if end < len(saved):
        items.append(Text(f"↓ {len(saved) - end} older", style="dim"))
    return Group(*items)


def _select_saved_function(saved: list[SavedFunction]) -> SavedFunction | None:
    if not console.is_terminal or not sys.stdin.isatty():
        console.print(_function_browser_render(saved, 0, window=len(saved)))
        while True:
            choice = console.input(
                f"Select AI Function [1-{len(saved)}] (q to quit): "
            ).strip().lower()
            if choice == "q":
                return None
            if choice.isdigit() and 1 <= int(choice) <= len(saved):
                return saved[int(choice) - 1]
            console.print("Invalid choice.", style="red")

    selected = 0
    descriptor = sys.stdin.fileno()
    original = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        with Live(
            _function_browser_render(saved, selected),
            console=console,
            refresh_per_second=12,
            transient=True,
        ) as live:
            while True:
                key = sys.stdin.read(1)
                if key == "\x03":
                    raise KeyboardInterrupt
                if key in {"q", "Q"}:
                    return None
                if key in {"\r", "\n"}:
                    return saved[selected]
                if key != "\x1b":
                    continue
                sequence = sys.stdin.read(2)
                if sequence == "[A":
                    selected = (selected - 1) % len(saved)
                elif sequence == "[B":
                    selected = (selected + 1) % len(saved)
                else:
                    continue
                live.update(_function_browser_render(saved, selected))
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, original)


def _acquisition_context(
    prediction: Prediction, source: str, *, binary_true_label: str = "True"
) -> str:
    selection = (
        "Held-out evaluation"
        if source == "holdout"
        else "Random audit sample"
        if source == "exploration"
        else "Uncertainty sample"
    )
    uncertainty = prediction_ambiguity(prediction)
    if prediction.label_probabilities is not None:
        boundaries = sorted(
            prediction.label_probabilities.items(),
            key=lambda item: (-ambiguity(item[1]), item[0]),
        )[:2]
        prediction_text = "Model boundaries " + ", ".join(
            f"{label} {probability:.1%}" for label, probability in boundaries
        )
    elif prediction.score is not None:
        prediction_text = (
            f"Model score {prediction.score:.2f} at "
            f"{(prediction.confidence or 0.0):.1%} confidence"
        )
    elif prediction.choice is not None and prediction.probabilities is not None:
        probability = prediction.probabilities.get(prediction.choice, 0.0)
        prediction_text = f"Model choice {prediction.choice} at {probability:.1%}"
    else:
        probability = prediction.probability or 0.0
        display_label = (
            "true" if binary_true_label.casefold() == "true" else binary_true_label
        )
        prediction_text = f"Model {display_label} probability {probability:.1%}"
    return f"{selection} · {uncertainty:.1%} uncertain · {prediction_text}"


def _acquisition_probability(prediction: Prediction) -> float:
    if prediction.probability is not None:
        return prediction.probability
    if prediction.label_probabilities is not None:
        return max(
            prediction.label_probabilities.values(),
            key=ambiguity,
        )
    if prediction.score_probabilities is not None:
        return max(prediction.score_probabilities.values())
    if prediction.probabilities is not None:
        return max(prediction.probabilities.values())
    raise ValueError("prediction has no probability output")


def _json_container(value: str) -> dict[object, object] | list[object] | None:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        parsed = json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, (dict, list)) else None


def _pretty_display_value(value: str) -> str:
    """Indent JSON containers for cards without changing model input values."""
    parsed = _json_container(value)
    return (
        json.dumps(parsed, indent=2, ensure_ascii=False)
        if parsed is not None
        else value
    )


def _styled_acquisition_context(
    prediction: Prediction, source: str, *, binary_true_label: str = "True"
) -> Text:
    plain = _acquisition_context(
        prediction, source, binary_true_label=binary_true_label
    )
    rendered = Text(plain)
    selection = (
        "Held-out evaluation"
        if source == "holdout"
        else "Random audit sample"
        if source == "exploration"
        else "Uncertainty sample"
    )
    selection_style = (
        "bold blue"
        if source == "holdout"
        else "bold yellow"
        if source == "exploration"
        else "bold magenta"
    )
    rendered.stylize(selection_style, 0, len(selection))
    prediction_start = plain.rfind(" · ") + 3
    rendered.stylize("cyan", prediction_start)
    return rendered


def _render_story(
    fields: dict[str, str],
    index: int,
    total: int,
    prediction: Prediction,
    source: str,
    binary_true_label: str = "True",
) -> None:
    header = Text()
    header.append(f"Label {index}/{total}", style="bold")
    header.append(" · ")
    header.append_text(
        _styled_acquisition_context(
            prediction, source, binary_true_label=binary_true_label
        )
    )
    console.print(header)
    body: list[object] = []
    for field_index, (name, value) in enumerate(fields.items()):
        if field_index:
            body.append(Text())
        body.append(Text(name, style="bold"))
        pretty = _pretty_display_value(value) if value else "(empty)"
        if value and _json_container(value) is not None:
            body.append(
                Syntax(
                    pretty,
                    "json",
                    theme="ansi_dark",
                    background_color="default",
                    word_wrap=True,
                    indent_guides=False,
                )
            )
        else:
            body.append(Text(pretty))
    console.print(Panel(Group(*body)))


def _menu_render(
    prompt: str,
    options: list[tuple[str, str]],
    selected: int,
    option_styles: list[str] | None = None,
) -> Text:
    rendered = Text(prompt + "\n", style="bold")
    start, end = _visible_range(len(options), selected)
    if start:
        rendered.append(f"    ↑ {start} more\n", style="dim")
    for index in range(start, end):
        _, label = options[index]
        active = index == selected
        rendered.append("  > " if active else "    ")
        if option_styles:
            base_style = option_styles[index]
            style = f"bold {base_style}" if active else base_style
        else:
            style = "bold cyan" if active else ""
        rendered.append(label, style=style)
        rendered.append("\n")
    if end < len(options):
        rendered.append(f"    ↓ {len(options) - end} more", style="dim")
    else:
        rendered.rstrip()
    return rendered


def _select_option(
    prompt: str,
    options: list[tuple[str, str]],
    *,
    hidden_shortcuts: set[str] | None = None,
    initial_key: str | None = None,
    option_styles: list[str] | None = None,
) -> str:
    """Select an option with arrows/Enter, retaining single-key shortcuts."""
    if not options:
        raise ValueError("at least one option is required")
    if option_styles is not None and len(option_styles) != len(options):
        raise ValueError("option styles must match option count")

    shortcuts = {key.lower(): key for key, _ in options}
    hidden = {key.lower() for key in hidden_shortcuts or set()}
    selected = 0
    if initial_key is not None:
        try:
            selected = next(
                index
                for index, (key, _) in enumerate(options)
                if key.lower() == initial_key.lower()
            )
        except StopIteration as error:
            raise ValueError(f"unknown initial option: {initial_key}") from error
    if not console.is_terminal or not sys.stdin.isatty():
        choices = "/".join(label for _, label in options)
        while True:
            value = console.input(f"{prompt} ({choices}): ").strip().lower()
            if not value:
                return options[selected][0]
            if value in hidden:
                return value
            for key, label in options:
                if value in {key.lower(), label.lower()}:
                    return key
            console.print("Invalid choice.", style="red")

    descriptor = sys.stdin.fileno()
    original = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        rendered = _menu_render(prompt, options, selected, option_styles)
        console.print(rendered)
        line_count = rendered.plain.count("\n") + 1
        while True:
            key = sys.stdin.read(1)
            if key == "\x03":
                raise KeyboardInterrupt
            if key in {"\r", "\n"}:
                return options[selected][0]
            if key.lower() in shortcuts:
                return shortcuts[key.lower()]
            if key.lower() in hidden:
                return key.lower()
            if key == "\x1b":
                sequence = sys.stdin.read(2)
                if sequence == "[A":
                    selected = (selected - 1) % len(options)
                elif sequence == "[B":
                    selected = (selected + 1) % len(options)
                else:
                    continue
                sys.stdout.write(f"\x1b[{line_count}A\r\x1b[J")
                sys.stdout.flush()
                rendered = _menu_render(prompt, options, selected, option_styles)
                console.print(rendered)
                line_count = rendered.plain.count("\n") + 1
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, original)


def _prompt_label(
    candidate: (
        CandidateSpec
        | MulticlassCandidateSpec
        | MultilabelCandidateSpec
        | ScoreCandidateSpec
    ),
    *,
    can_go_back: bool,
    prediction: Prediction | None = None,
    binary_labels: tuple[str, str] = ("True", "False"),
) -> bool | int | str | list[str] | object:
    hidden = {"b"} if can_go_back else None
    if isinstance(candidate, ScoreCandidateSpec):
        suggested_level = (
            nearest_score_level(prediction.score, len(candidate.levels))
            if prediction is not None and prediction.score is not None
            else 0
        )
        selected = _select_option(
            "Level",
            [
                (str(index), f"{index} · {_compact_option_label(description)}")
                for index, description in enumerate(candidate.levels)
            ],
            hidden_shortcuts=hidden,
            initial_key=str(suggested_level),
            option_styles=[
                LABEL_COLORS[index % len(LABEL_COLORS)]
                for index in range(len(candidate.levels))
            ],
        )
        if selected == "b":
            return BACK_LABEL
        return int(selected)
    if isinstance(candidate, MultilabelCandidateSpec):
        suggested = (
            {
                name
                for name, probability in prediction.label_probabilities.items()
                if probability >= 0.5
            }
            if prediction is not None and prediction.label_probabilities is not None
            else set()
        )
        return _select_multilabel_labels(
            list(candidate.labels),
            can_go_back=can_go_back,
            suggested=suggested,
        )
    if isinstance(candidate, MulticlassCandidateSpec):
        classes = list(candidate.criteria)
        selected = _select_option(
            "Label",
            [(str(index), name) for index, name in enumerate(classes, start=1)],
            hidden_shortcuts=hidden,
            initial_key=(
                str(classes.index(prediction.choice) + 1)
                if prediction is not None and prediction.choice in classes
                else None
            ),
            option_styles=[
                LABEL_COLORS[index % len(LABEL_COLORS)]
                for index in range(len(classes))
            ],
        )
        if selected == "b":
            return BACK_LABEL
        return classes[int(selected) - 1]
    selected = _select_option(
        "Label",
        [("t", binary_labels[0]), ("f", binary_labels[1])],
        hidden_shortcuts=hidden,
        initial_key=(
            "t"
            if prediction is None
            or prediction.probability is None
            or prediction.probability >= 0.5
            else "f"
        ),
        option_styles=["green", "red"],
    )
    if selected == "b":
        return BACK_LABEL
    return selected == "t"


def _prompt_rationale() -> str | object:
    rationale = console.input(
        "Rationale [dim](optional, encouraged; /back changes label)[/dim]: "
    )
    if rationale.strip().lower() == "/back":
        return BACK_LABEL
    return rationale


def _change_text(value: float, *, regression_style: str = "red") -> Text:
    style = "green" if value > 0 else regression_style if value < 0 else "dim"
    return Text(f"{value:+.3f}", style=style)


def _metric_display(metrics: object) -> tuple[str, str, float]:
    if isinstance(metrics, ScoreMetrics):
        return (
            "Ordinal fit",
            f"{metrics.fit_score:.3f} · MAE {metrics.mae:.3f}",
            metrics.fit_score,
        )
    if isinstance(metrics, MultilabelMetrics):
        return (
            "Micro-F1",
            f"{metrics.micro_f1:.3f} · {metrics.exact_matches}/{metrics.total} exact",
            metrics.micro_f1,
        )
    if isinstance(metrics, MulticlassMetrics):
        return (
            "Macro-F1",
            f"{metrics.macro_f1:.3f} · {metrics.correct}/{metrics.total} correct",
            metrics.macro_f1,
        )
    assert isinstance(metrics, BinaryMetrics)
    return (
        "F1",
        f"{metrics.f1:.3f} · P {metrics.precision:.3f} · R {metrics.recall:.3f}",
        metrics.f1,
    )


def _add_metric_row(
    table: Table,
    split: str,
    previous: object,
    proposed: object,
) -> None:
    metric_name, previous_text, previous_value = _metric_display(previous)
    proposed_name, proposed_text, proposed_value = _metric_display(proposed)
    if metric_name != proposed_name:
        raise ValueError("current and proposed metrics use different task types")
    table.add_row(
        f"{split} {metric_name}",
        previous_text,
        proposed_text,
        _change_text(proposed_value - previous_value),
    )


def _metrics_table(report: RoundReport) -> Table:
    table = Table(title="AI Function scores")
    table.add_column("Score")
    table.add_column("Current", justify="right")
    table.add_column("Proposed", justify="right")
    table.add_column("Change", justify="right")
    _add_metric_row(
        table, "Training", report.previous_metrics, report.proposed_metrics
    )
    if (
        report.previous_holdout_metrics is not None
        and report.proposed_holdout_metrics is not None
    ):
        _add_metric_row(
            table,
            "Held-out",
            report.previous_holdout_metrics,
            report.proposed_holdout_metrics,
        )
    return table


def _certainty_bar(certainty: float, width: int = 24) -> Text:
    certainty = min(1.0, max(0.0, certainty))
    filled = round(certainty * width)
    rendered = Text("[")
    rendered.append("█" * filled, style="cyan")
    rendered.append("░" * (width - filled), style="dim")
    rendered.append(f"] {certainty:.1%} certain")
    return rendered


def _show_current_uncertainty(predictions: list[Prediction]) -> None:
    summary = ambiguity_summary(predictions)
    mean_ambiguity = float(summary["mean"])
    table = Table(title="Current fixed-pool uncertainty")
    table.add_column("Measure")
    table.add_column("Value")
    table.add_row("Certainty", _certainty_bar(1.0 - mean_ambiguity))
    table.add_row("Mean uncertainty", f"{mean_ambiguity:.1%}")
    table.add_row("Moderately ambiguous (≥50%)", str(summary["at_least_0_5"]))
    table.add_row("Highly ambiguous (≥80%)", str(summary["at_least_0_8"]))
    console.print(table)


def _show_certainty_history(session: ClimbSession, report: RoundReport) -> None:
    reports: dict[int, RoundReport] = {}
    legacy_rounds: list[int] = []
    for path in sorted(session.store.directory.glob("round-*-report.json")):
        round_number = int(path.stem.split("-")[1])
        loaded = RoundReport.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if loaded.ambiguity_scope == "fixed_full_pool":
            reports[round_number] = loaded
        else:
            legacy_rounds.append(round_number)

    if report.ambiguity_scope == "fixed_full_pool":
        reports[session.state.round_number] = report
    decisions = {item.round_number: item.decision for item in session.state.history}
    if reports:
        table = Table(
            title=f"Certainty by iteration (fixed {session.state.pool_size}-row pool)"
        )
        table.add_column("Iteration", justify="right")
        table.add_column("Result")
        first_round, first_report = min(reports.items())
        baseline_certainty = 1.0 - float(first_report.previous_ambiguity["mean"])
        table.add_row(f"Before {first_round}", _certainty_bar(baseline_certainty))
        previous_certainty = baseline_certainty
        for round_number, saved_report in sorted(reports.items()):
            decision = decisions.get(round_number)
            if decision == "rejected":
                summary = saved_report.previous_ambiguity
                status = "rejected"
            else:
                summary = saved_report.proposed_ambiguity
                status = "accepted" if decision == "accepted" else "proposed"
            certainty = 1.0 - float(summary["mean"])
            result = _certainty_bar(certainty)
            change = certainty - previous_certainty
            if abs(change) > 0.00005:
                result.append(
                    f"  {change:+.1%}", style="green" if change > 0 else "yellow"
                )
            status_style = (
                "green"
                if status == "accepted"
                else "red"
                if status == "rejected"
                else "yellow"
            )
            iteration = Text(f"{round_number} (")
            iteration.append(status, style=status_style)
            iteration.append(")")
            table.add_row(iteration, result)
            previous_certainty = certainty
        console.print(table)
    if legacy_rounds:
        joined = ", ".join(str(item) for item in legacy_rounds)
        console.print(
            f"[dim]Iterations {joined} used the old shrinking-pool diagnostic "
            "and are omitted because they are not directly comparable.[/dim]"
        )


def _ambiguity_value(key: str, value: float | int) -> str:
    return f"{float(value):.1%}" if key in {"mean", "median"} else f"{int(value):,}"


def _ambiguity_change(
    key: str, current: float | int, proposed: float | int
) -> Text:
    change = float(proposed) - float(current)
    label = f"{change:+.1%}" if key in {"mean", "median"} else f"{int(change):+d}"
    return Text(label, style="green" if change < 0 else "yellow" if change > 0 else "dim")


def _show_report(session: ClimbSession, report: RoundReport) -> None:
    console.print(_metrics_table(report))
    ambiguity_title = (
        "Fixed full-pool ambiguity (diagnostic only)"
        if report.ambiguity_scope == "fixed_full_pool"
        else "Remaining-pool ambiguity (legacy diagnostic)"
    )
    ambiguity = Table(title=ambiguity_title)
    ambiguity.add_column("Metric")
    ambiguity.add_column("Current", justify="right")
    ambiguity.add_column("Proposed", justify="right")
    ambiguity.add_column("Change", justify="right")
    for key in ("count", "mean", "median", "at_least_0_5", "at_least_0_8"):
        ambiguity.add_row(
            key,
            _ambiguity_value(key, report.previous_ambiguity[key]),
            _ambiguity_value(key, report.proposed_ambiguity[key]),
            _ambiguity_change(
                key,
                report.previous_ambiguity[key],
                report.proposed_ambiguity[key],
            ),
        )
    console.print(ambiguity)
    if (
        report.previous_replay_ambiguity is not None
        and report.proposed_replay_ambiguity is not None
    ):
        replay = Table(title="Labeled replay-set ambiguity")
        replay.add_column("Metric")
        replay.add_column("Current", justify="right")
        replay.add_column("Proposed", justify="right")
        replay.add_column("Change", justify="right")
        for key in ("count", "mean", "median", "at_least_0_5", "at_least_0_8"):
            replay.add_row(
                key,
                _ambiguity_value(key, report.previous_replay_ambiguity[key]),
                _ambiguity_value(key, report.proposed_replay_ambiguity[key]),
                _ambiguity_change(
                    key,
                    report.previous_replay_ambiguity[key],
                    report.proposed_replay_ambiguity[key],
                ),
            )
        console.print(replay)
    _show_certainty_history(session, report)
    console.print(
        f"GEPA metric calls: {report.total_metric_calls} actual / "
        f"{report.configured_metric_budget} configured "
        "(P×N may finish one in-flight step past the budget)."
    )
    diff = candidate_diff(report.previous_candidate, report.proposed_candidate)
    console.print(Panel(diff or "No textual change.", title="AI Function diff"))


def _decision_gate(session: ClimbSession, report: RoundReport) -> bool:
    _show_report(session, report)
    value = _select_option(
        "AI Function decision",
        [("a", "Accept"), ("r", "Reject"), ("q", "Quit")],
        option_styles=["green", "red", "dim"],
    )
    if value == "a":
        session.decide("accept")
        return True
    if value == "r":
        session.decide("reject")
        return True
    console.print("Run saved with this decision pending.")
    return False


def _signature(candidate: TaskSpec, state: RunState) -> str:
    inputs = (
        "content"
        if state.column_mode == "all_concatenated"
        else ", ".join(state.selected_columns or ["legacy fields"])
    )
    if isinstance(candidate, ScoreCandidateSpec):
        output = f"Score[0..{len(candidate.levels) - 1}]"
    elif isinstance(candidate, MultilabelCandidateSpec):
        output = "set[" + " | ".join(candidate.labels) + "]"
    elif isinstance(candidate, MulticlassCandidateSpec):
        output = " | ".join(candidate.criteria)
    else:
        output = f"{state.binary_true_label} | {state.binary_false_label}"
    return f"({inputs}) → {output}"


def _show_latest_optimized(saved: SavedFunction) -> None:
    state = saved.state
    candidate = state.current_candidate
    setup = Table.grid(padding=(0, 2))
    setup.add_column(style="bold")
    setup.add_column()
    setup.add_row("Run", state.run_id)
    setup.add_row("Type", _task_kind(candidate))
    setup.add_row("Source", state.source_path)
    setup.add_row("Rows", f"{state.pool_size:,}")
    setup.add_row("Columns", ", ".join(state.selected_columns or ["legacy default"]))
    setup.add_row("Training annotations/round", str(state.batch_size))
    setup.add_row(
        "Held-out evaluation",
        (
            f"20% reserved · {state.holdout_batch_size} extra annotation(s)/round"
            if state.holdout_fraction
            else "Disabled"
        ),
    )
    setup.add_row("Evaluation backend", state.backend.provider)
    setup.add_row("Evaluation model", state.backend.model)
    setup.add_row("GEPA model", state.reflection_model)
    setup.add_row("Max GEPA metric calls", str(state.metric_budget))
    setup.add_row("Labels", str(saved.label_count))
    console.print(Panel(setup, title="Setup"))
    console.print(Panel(_signature(candidate, state), title="Signature"))

    definition = Table.grid(padding=(0, 2))
    definition.add_column(style="bold", no_wrap=True)
    definition.add_column()
    definition.add_row("Instructions", candidate.instructions)
    if isinstance(candidate, ScoreCandidateSpec):
        for index, description in enumerate(candidate.levels):
            definition.add_row(f"Level {index}", description)
    elif isinstance(candidate, MultilabelCandidateSpec):
        for name, criteria in candidate.labels.items():
            definition.add_row(f"{name} · true", criteria.true_criteria)
            definition.add_row(f"{name} · false", criteria.false_criteria)
    elif isinstance(candidate, MulticlassCandidateSpec):
        for name, description in candidate.criteria.items():
            definition.add_row(name, description)
    else:
        definition.add_row(state.binary_true_label, candidate.true_criteria)
        definition.add_row(state.binary_false_label, candidate.false_criteria)
    console.print(Panel(definition, title="Latest optimized AI Function"))

    scoring = Table.grid(padding=(0, 2))
    scoring.add_column(style="bold")
    scoring.add_column()
    fit = _latest_fit(saved)
    scoring.add_row(
        "Training fit", f"{fit:.3f}" if fit is not None else "Not optimized yet"
    )
    holdout_fit = _latest_holdout_fit(saved)
    scoring.add_row(
        "Held-out score",
        f"{holdout_fit:.3f}" if holdout_fit is not None else "Not available",
    )
    scoring.add_row(
        "Fixed-pool certainty",
        f"{saved.certainty:.1%}" if saved.certainty is not None else "Not evaluated",
    )
    scoring.add_row("Human labels", str(saved.label_count))
    console.print(Panel(scoring, title="Scoring"))


def _prediction_result(
    candidate: TaskSpec,
    prediction: Prediction,
    binary_labels: tuple[str, str] = ("True", "False"),
) -> dict[str, object]:
    if isinstance(candidate, ScoreCandidateSpec):
        level = nearest_score_level(prediction.score or 0.0, len(candidate.levels))
        return {
            "score": prediction.score,
            "level": level,
            "level_description": candidate.levels[level],
            "confidence": prediction.confidence,
            "probabilities": prediction.score_probabilities,
        }
    if isinstance(candidate, MultilabelCandidateSpec):
        probabilities = prediction.label_probabilities or {}
        return {
            "labels": [name for name, value in probabilities.items() if value >= 0.5],
            "probabilities": probabilities,
        }
    if isinstance(candidate, MulticlassCandidateSpec):
        return {
            "choice": prediction.choice,
            "confidence": prediction.confidence,
            "probabilities": prediction.probabilities,
        }
    value = (prediction.probability or 0.0) >= 0.5
    return {
        "label": binary_labels[0] if value else binary_labels[1],
        "value": value,
        "true_probability": prediction.probability,
    }


def _write_function_results(
    saved: SavedFunction,
    source: Path,
    stories: list[Story],
    predictions: list[Prediction],
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output_directory = Path.cwd() / ".jev-align" / "outputs"
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / (
        f"{timestamp}-{_slug(saved.state.current_candidate.instructions)}-"
        f"{_slug(source.stem)}.jsonl"
    )
    with output.open("w", encoding="utf-8") as handle:
        for story, prediction in zip(stories, predictions, strict=True):
            handle.write(
                json.dumps(
                    {
                        "row_number": story.row_number,
                        "input": story.fields,
                        "result": _prediction_result(
                            saved.state.current_candidate,
                            prediction,
                            (
                                saved.state.binary_true_label,
                                saved.state.binary_false_label,
                            ),
                        ),
                        "resolved_model": prediction.resolved_model,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return output


def _run_saved_function(saved: SavedFunction) -> Path:
    credential_error = backend_credential_error(saved.state.backend, os.environ)
    if credential_error:
        raise typer.BadParameter(credential_error)
    source = _choose_dataset(Path.cwd())
    row_limit = _choose_pool_size(source)
    available_columns = dataset_columns(source)
    column_choice = _select_option(
        "Row fields",
        [("a", "Use all columns, concatenated"), ("s", "Select specific columns")],
    )
    concatenate = column_choice == "a"
    columns = (
        available_columns
        if concatenate
        else _choose_columns(available_columns, default_all=False)
    )
    stories = load_stories(source, row_limit, columns, concatenate=concatenate)
    try:
        backend = create_backend(
            saved.state.backend, concurrency=saved.state.concurrency
        )
        validate_backend_for_task(backend, saved.state.current_candidate)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    predictions = evaluate_with_progress(
        backend,
        saved.state.current_candidate,
        stories,
        f"{backend_display_name(backend)} · running AI Function",
    )
    output = _write_function_results(saved, source, stories, predictions)
    console.print(f"Wrote {len(predictions):,} results to [bold]{output}[/bold]")
    return output


def _resume_saved_function(saved: SavedFunction) -> None:
    optimize(
        data=None,
        question=None,
        reflection_model=None,
        columns=None,
        all_columns_concatenated=False,
        classes=None,
        score_levels=None,
        multilabel=False,
        true_criteria=None,
        false_criteria=None,
        true_label="True",
        false_label="False",
        resume=saved.directory,
        backend_name="typesafe",
        backend_model=None,
        pool_size=1000,
        batch_size=5,
        holdout=False,
        metric_budget=300,
        concurrency=16,
        seed=0,
    )


@app.command("functions")
def functions_command() -> None:
    """Browse, inspect, resume, or run saved AI Functions."""
    saved_functions = _load_saved_functions(Path.cwd())
    if not saved_functions:
        console.print("No saved AI Functions found in .jev-align/runs.")
        return
    while True:
        selected = _select_saved_function(saved_functions)
        if selected is None:
            return
        action = _select_option(
            _function_name(selected),
            [
                ("c", "Resume optimization"),
                ("s", "Show latest optimized"),
                ("r", "Run on another dataset"),
                ("b", "Back to AI Functions"),
            ],
        )
        if action == "c":
            _resume_saved_function(selected)
            return
        if action == "s":
            _show_latest_optimized(selected)
            _select_option("Latest optimized", [("b", "Back")])
        elif action == "r":
            _run_saved_function(selected)
            _select_option("Run complete", [("b", "Back")])


@app.command("climb", hidden=True)
@app.command("optimize")
def optimize(
    data: Annotated[
        Path | None,
        typer.Argument(help="CSV, Parquet, or JSONL dataset for a new run"),
    ] = None,
    question: Annotated[str | None, typer.Option("--question", "-q")] = None,
    reflection_model: Annotated[
        str | None,
        typer.Option(
            "--reflection-model",
            help=(
                "GEPA LiteLLM model. By default, uses the first configured "
                "provider in OpenAI → Anthropic → Gemini order."
            ),
        ),
    ] = None,
    columns: Annotated[
        list[str] | None,
        typer.Option("--column", help="Dataset column to include; repeat as needed"),
    ] = None,
    all_columns_concatenated: Annotated[
        bool,
        typer.Option(
            "--all-columns-concatenated",
            help="Join all dataset columns into one newline-separated content field",
        ),
    ] = False,
    classes: Annotated[
        list[str] | None,
        typer.Option(
            "--class",
            help=(
                "Add a possible label and describe what counts as it: "
                "NAME=DESCRIPTION; repeat for each label"
            ),
        ),
    ] = None,
    score_levels: Annotated[
        list[str] | None,
        typer.Option(
            "--score-level",
            help=(
                "Add an ordered Score level description from lowest to highest; "
                "repeat 2–10 times"
            ),
        ),
    ] = None,
    multilabel: Annotated[
        bool,
        typer.Option(
            "--multilabel",
            help="Allow zero or more of the --class labels to apply to each row",
        ),
    ] = False,
    true_criteria: Annotated[
        str | None,
        typer.Option("--true-criteria", help="What counts as true for a binary task"),
    ] = None,
    false_criteria: Annotated[
        str | None,
        typer.Option("--false-criteria", help="What counts as false for a binary task"),
    ] = None,
    true_label: Annotated[
        str, typer.Option("--true-label", help="Displayed name for true")
    ] = "True",
    false_label: Annotated[
        str, typer.Option("--false-label", help="Displayed name for false")
    ] = "False",
    resume: Annotated[
        Path | None, typer.Option("--resume", help="Existing run directory")
    ] = None,
    backend_name: Annotated[
        str,
        typer.Option("--backend", help="AI Function evaluation backend"),
    ] = "typesafe",
    backend_model: Annotated[
        str | None,
        typer.Option(
            "--backend-model",
            "--jev-model",
            help="Evaluation model; --jev-model remains as a compatibility alias",
        ),
    ] = None,
    pool_size: Annotated[int, typer.Option("--pool-size", min=5)] = 1000,
    batch_size: Annotated[int, typer.Option("--batch-size", min=2)] = 5,
    holdout: Annotated[
        bool,
        typer.Option(
            "--holdout",
            help=(
                "Reserve 20% of rows and collect 20% extra held-out annotations "
                "per round"
            ),
        ),
    ] = False,
    metric_budget: Annotated[
        int,
        typer.Option(
            "--max-metric-calls",
            "--metric-budget",
            min=1,
            help="Maximum GEPA metric calls per optimization round",
        ),
    ] = 300,
    concurrency: Annotated[int, typer.Option("--concurrency", min=1)] = 16,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    """Run or resume an interactive AI Function optimization."""
    _configure_provider_key_aliases()

    if resume is not None:
        store = RunStore(resume.resolve())
        state = store.load_state()
        if reflection_model:
            state.reflection_model = reflection_model
            store.save_state(state)
        source = Path(state.source_path)
        if source_sha256(source) != state.source_sha256:
            raise typer.BadParameter("source CSV changed since this run was created")
        selected_columns = state.selected_columns
        if selected_columns is None:
            available = dataset_columns(source)
            legacy = [
                column for column in ("title", "text", "url") if column in available
            ]
            selected_columns = legacy or available
        stories = load_stories(
            source,
            state.pool_size,
            selected_columns,
            concatenate=state.column_mode == "all_concatenated",
        )
    else:
        if data is None or question is None:
            raise typer.BadParameter("new runs require DATA and --question")
        selected_reflection_model = reflection_model or _default_reflection_model()
        if selected_reflection_model is None:
            raise typer.BadParameter(
                "no reflection provider key detected; set OPENAI_API_KEY, "
                "ANTHROPIC_API_KEY (or CLAUDE_API_KEY), GEMINI_API_KEY, or pass "
                "--reflection-model"
            )
        if multilabel and not classes:
            raise typer.BadParameter("--multilabel requires at least one --class label")
        if score_levels and classes:
            raise typer.BadParameter("--score-level cannot be combined with --class")
        if score_levels and multilabel:
            raise typer.BadParameter(
                "--score-level cannot be combined with --multilabel"
            )
        if (classes or score_levels) and (true_criteria or false_criteria):
            raise typer.BadParameter(
                "--true-criteria and --false-criteria apply only to binary tasks"
            )
        if not true_label.strip() or not false_label.strip():
            raise typer.BadParameter("binary presentation labels must be nonempty")
        if true_label.strip().casefold() == false_label.strip().casefold():
            raise typer.BadParameter("binary presentation labels must be distinct")
        if score_levels and not 2 <= len(score_levels) <= 10:
            raise typer.BadParameter("repeat --score-level between 2 and 10 times")
        if score_levels and any(not level.strip() for level in score_levels):
            raise typer.BadParameter("Score level descriptions must be nonempty")
        source = data.resolve()
        available = dataset_columns(source)
        if all_columns_concatenated and columns:
            raise typer.BadParameter(
                "--all-columns-concatenated cannot be combined with --column"
            )
        selected_columns = (
            available
            if all_columns_concatenated
            else (list(columns) if columns else available)
        )
        column_mode = "all_concatenated" if all_columns_concatenated else "selected"
        stories = load_stories(
            source,
            pool_size,
            selected_columns,
            concatenate=column_mode == "all_concatenated",
        )
        if len(stories) < pool_size:
            console.print(
                f"[dim]Using all {len(stories):,} rows; requested pool size was "
                f"{pool_size:,}.[/dim]"
            )
        if len(stories) < batch_size:
            raise typer.BadParameter(
                f"dataset has {len(stories):,} rows, fewer than the requested "
                f"batch size of {batch_size:,}"
            )
        holdout_ids = _holdout_story_ids(
            stories, fraction=0.2 if holdout else 0.0, seed=seed
        )
        if len(stories) - len(holdout_ids) < batch_size:
            raise typer.BadParameter(
                "the 20% holdout leaves fewer training rows than the requested "
                f"batch size of {batch_size:,}"
            )
        if score_levels:
            candidate: (
                CandidateSpec
                | MulticlassCandidateSpec
                | MultilabelCandidateSpec
                | ScoreCandidateSpec
            ) = ScoreCandidateSpec(
                instructions=question,
                levels=list(score_levels),
            )
        elif classes:
            parsed_classes: dict[str, str] = {}
            for item in classes:
                if "=" not in item:
                    raise typer.BadParameter(
                        "each --class must name a label and describe what counts "
                        "as it: NAME=DESCRIPTION"
                    )
                name, description = (part.strip() for part in item.split("=", 1))
                if not name or not description or name in parsed_classes:
                    raise typer.BadParameter(
                        "label names must be unique, and names and descriptions "
                        "must be nonempty"
                    )
                parsed_classes[name] = description
            if not multilabel and len(parsed_classes) < 2:
                raise typer.BadParameter(
                    "multiple-choice runs require at least two --class labels"
                )
            if multilabel:
                candidate = MultilabelCandidateSpec(
                    instructions=question,
                    labels={
                        name: MultilabelCriteria(
                            true_criteria=description,
                            false_criteria=(
                                f"The row is not about or meaningfully related to {name}: "
                                f"{description}"
                            ),
                        )
                        for name, description in parsed_classes.items()
                    },
                )
            else:
                candidate = MulticlassCandidateSpec(
                    instructions=question,
                    criteria=parsed_classes,
                )
        else:
            candidate = CandidateSpec(
                instructions=question,
                true_criteria=true_criteria
                or "The condition stated in the instructions is satisfied.",
                false_criteria=false_criteria
                or "The condition stated in the instructions is not satisfied.",
            )
        run_directory = _new_run_directory(question).resolve()
        state = RunState(
            run_id=run_directory.name,
            source_path=str(source),
            source_sha256=source_sha256(source),
            selected_columns=selected_columns,
            column_mode=column_mode,
            pool_size=len(stories),
            seed=seed,
            backend=BackendConfig(
                provider=backend_name,
                model=backend_model or "jev-1.13.0",
            ),
            reflection_model=selected_reflection_model,
            metric_budget=metric_budget,
            concurrency=concurrency,
            batch_size=batch_size,
            holdout_fraction=0.2 if holdout else 0.0,
            holdout_story_ids=holdout_ids,
            binary_true_label=true_label,
            binary_false_label=false_label,
            current_candidate=candidate,
            history=[
                CandidateHistory(round_number=0, candidate=candidate, decision="seed")
            ],
        )
        store = RunStore(run_directory)
        store.initialize(state)
        console.print(f"Run created at [bold]{store.directory}[/bold]")

    credential_error = backend_credential_error(state.backend, os.environ)
    if credential_error:
        raise typer.BadParameter(credential_error)
    try:
        backend = create_backend(state.backend, concurrency=state.concurrency)
        validate_backend_for_task(backend, state.current_candidate)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    session = ClimbSession(state, stories, store, backend)

    try:
        if state.pending_candidate is not None and state.pending_report is not None:
            if not _decision_gate(session, RoundReport.from_dict(state.pending_report)):
                return

        while True:
            remaining = session.unlabeled_stories()
            if len(remaining) < state.batch_size:
                console.print("The unlabeled pool is exhausted.")
                return
            console.print(
                f"\n[bold]Round {state.round_number}[/bold]: evaluating "
                f"{len(remaining)} unlabeled stories with "
                f"{state.backend.provider}/{state.backend.model}..."
            )
            acquisitions, pool_predictions = session.acquire()
            holdout_acquisitions = session.acquire_holdout(pool_predictions)
            acquisitions = [*acquisitions, *holdout_acquisitions]
            _show_current_uncertainty(pool_predictions)
            if holdout_acquisitions:
                console.print(
                    f"[dim]This round includes {len(holdout_acquisitions)} extra "
                    "held-out evaluation annotation(s).[/dim]"
                )
            index = 0
            rewound = False
            while index < len(acquisitions):
                acquisition = acquisitions[index]
                story = session.story_by_id[acquisition.prediction.story_id]
                _render_story(
                    story.fields,
                    index + 1,
                    len(acquisitions),
                    acquisition.prediction,
                    acquisition.source,
                    state.binary_true_label,
                )
                if index == 0 and state.round_number > 1:
                    console.print(
                        f"[dim]Press b to rewind to round "
                        f"{state.round_number - 1} labeling.[/dim]"
                    )
                while True:
                    choice = _prompt_label(
                        state.current_candidate,
                        can_go_back=index > 0 or state.round_number > 1,
                        prediction=acquisition.prediction,
                        binary_labels=(
                            state.binary_true_label,
                            state.binary_false_label,
                        ),
                    )
                    if choice is BACK_LABEL:
                        if index == 0:
                            confirmation = _select_option(
                                f"Rewind to round {state.round_number - 1}?",
                                [
                                    ("c", "Cancel (default)"),
                                    ("r", "Rewind and reopen its labeling phase"),
                                ],
                            )
                            if confirmation == "c":
                                continue
                            target_round, archive = session.rewind_previous_round()
                            console.print(
                                f"[dim]Rewound to round {target_round}. Previous "
                                f"state archived at {archive}.[/dim]"
                            )
                            rewound = True
                            break
                        previous = acquisitions[index - 1]
                        session.undo_last_label(previous.prediction.story_id)
                        index -= 1
                        console.print("[dim]Previous label removed. Going back.[/dim]")
                        break
                    rationale = _prompt_rationale()
                    if rationale is BACK_LABEL:
                        console.print("[dim]Returning to the label picker.[/dim]")
                        continue
                    assert isinstance(choice, (bool, int, str, list))
                    assert isinstance(rationale, str)
                    session.add_label(
                        story_id=story.id,
                        label=choice,
                        rationale=rationale,
                        acquired_by=acquisition.source,  # type: ignore[arg-type]
                        evaluation_split=(
                            "holdout"
                            if acquisition.source == "holdout"
                            else "train"
                        ),
                        probability=_acquisition_probability(acquisition.prediction),
                        resolved_model=acquisition.prediction.resolved_model,
                    )
                    index += 1
                    break
                if rewound:
                    break

            if rewound:
                continue

            if not session.ready_to_optimize():
                console.print(
                    "[yellow]More label variety is required. Acquiring another batch.[/yellow]"
                )
                continue

            console.print("Running GEPA optimization...")
            report = session.optimize(pool_predictions)
            if not _decision_gate(session, report):
                return
    except KeyboardInterrupt:
        store.save_state(state)
        console.print("\nRun saved. Resume it with jev-align optimize --resume RUN.")
        raise typer.Exit(130)


if __name__ == "__main__":
    app()
