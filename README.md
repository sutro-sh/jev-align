# jev-align

`jev-align` is an experimental CLI for aligning [TypeSafe's Jev](https://docs.typesafe.ai/introduction) to your
judgment, built by [Sutro](https://sutro.sh/). 

It finds uncertain rows in a local dataset, asks you to label them, and uses
[GEPA](https://gepa-ai.github.io/gepa/) to improve Jev's accuracy. 

It's last-mile, feedback-driven fine-tuning, for Jev.

## Quick start

Requires Python 3.11 or newer.

### Install the current checkout

The recommended option is an isolated global installation with
[uv](https://docs.astral.sh/uv/):

```shell
git clone git@github.com:sutro-sh/jev-align.git
cd jev-align
uv tool install .
jeva
```

After changing or pulling the local source, reinstall it with:

```shell
uv tool install --force .
```

You can use pip instead:

```shell
pip install .
jeva
```

To install the latest source directly from GitHub without cloning it first:

```shell
uv tool install "git+https://github.com/sutro-sh/jev-align.git"
# Or: pip install "git+https://github.com/sutro-sh/jev-align.git"
```

After the first PyPI release, install it by package name:

```shell
uv tool install jev-align
# Or: pip install jev-align
```

### Develop locally

For an editable development environment:

```shell
uv sync --extra dev
uv run jeva
```

### Configure and launch

Set a TypeSafe key and at least one reflection-provider key:

```shell
export TYPESAFE_API_KEY="..."
export OPENAI_API_KEY="..." # or ANTHROPIC_API_KEY / GEMINI_API_KEY
jeva
```

Both `jeva` and `jev-align` start the same CLI; the shorter `jeva` command is
used below.

Interactive starts check PyPI for a newer release. When one is available, press
Enter to install it into the current Python environment, then restart `jeva`.
Inside a virtual environment, `jeva` prefers `uv pip` when `uv` is available
and falls back to that environment's Python and pip. Editable development
installs and non-interactive commands skip this check. Set
`JEVA_DISABLE_UPDATE_CHECK=1` to disable it explicitly.

The home screen lets you:

- Optimize a new AI Function.
- Browse, inspect, resume, or run an existing AI Function.

The new-function wizard discovers local CSV, Parquet, and JSONL files and walks
you through the dataset, columns, task, and reflection model.

## How it works

Each round:

1. Evaluates the configured dataset and measures uncertainty.
2. Selects four uncertain rows and one random audit row.
3. Collects your label and an optional rationale for each row.
4. Runs GEPA against all labels collected so far.
5. Shows the score, certainty history, and proposed definition diff.
6. Lets you accept, reject, or leave the decision for later.

The default pool is the first 1,000 rows, or the whole dataset when it contains
fewer than 1,000 rows. Scores describe fit against collected labels, not
held-out generalization unless the optional held-out evaluation is enabled.

Before creating a function, choose **Advanced** to collect 5, 10, 15, or 20
training annotations per round. You can also reserve 20% of the dataset as a
held-out evaluation set. When enabled, each round adds another 20% as many
held-out annotations—for example, 10 training annotations plus 2 held-out
annotations. Held-out labels never enter GEPA and are scored separately as the
function evolves. Advanced also lets you change the maximum GEPA metric calls
per round from its default of 300.

## Task types

| Type | Output |
| --- | --- |
| Binary | `True` or `False` |
| Multiclass | Exactly one fixed label |
| Multilabel | Zero or more fixed labels |
| Score | One level from an ordered rubric |

## Start from flags

The setup can also be supplied on the command line:

```shell
jeva optimize posts.csv \
  --question "Is the post related to aviation?" \
  --column title \
  --column text \
  --pool-size 1000
```

For multiclass or multilabel tasks, repeat
`--class "NAME=DESCRIPTION"`. Add `--multilabel` when labels may overlap. For
an ordered score, repeat `--score-level` from lowest to highest.

Run `jeva optimize --help` for every setup flag. Flags configure a run;
human labeling and proposal acceptance remain interactive by design.
Use `--holdout` to enable the same 20% held-out evaluation in a scripted setup.
Use `--max-metric-calls` to change GEPA's per-round budget;
`--metric-budget` remains a compatibility alias.

## Controls

- Use arrow keys and Enter in menus.
- Press `b` in the label picker to return to the previous item.
- Enter `/back` at the rationale prompt to change the current label.
- Use Space to toggle multilabel choices and Enter to confirm.

Every item must receive a label. Rationales are optional but strongly
encouraged because GEPA uses them as boundary guidance.

## Saved AI Functions

Runs are stored in `.jev-align/runs/`.

```shell
jeva functions
jeva optimize --resume .jev-align/runs/<run-id>
```

Running an accepted function on another dataset writes JSONL results to
`.jev-align/outputs/` without changing the function.

## Using a coding agent

See [AGENTS.md](AGENTS.md) for instructions that help a coding agent configure,
drive, resume, and explain this workflow without taking judgment calls away
from the user.

## Backends

Runs store a provider-neutral backend configuration. TypeSafe JEV is currently
the only included evaluation backend, but the optimizer is structured for
additional adapters. `--jev-model` remains a compatibility alias for
`--backend-model`.

## Development

```shell
uv run --no-active --quiet pytest
uv run --no-active --quiet ruff check src tests
```
