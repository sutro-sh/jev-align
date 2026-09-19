# jev-align

`jev-align` is an interactive CLI for teaching an AI Function to match your
judgment.

It finds uncertain rows in a local dataset, asks you to label them, and uses
[GEPA](https://gepa-ai.github.io/gepa/) to improve the function. The included
evaluation backend is [TypeSafe JEV](https://docs.typesafe.ai/introduction).

## Quick start

Requires Python 3.11 or newer.

```shell
uv sync --extra dev
export TYPESAFE_API_KEY="..."
export OPENAI_API_KEY="..." # or an Anthropic/Gemini key for GEPA reflection
./jev-align
```

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
held-out generalization.

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
./jev-align optimize posts.csv \
  --question "Is the post related to aviation?" \
  --column title \
  --column text \
  --pool-size 1000
```

For multiclass or multilabel tasks, repeat
`--class "NAME=DESCRIPTION"`. Add `--multilabel` when labels may overlap. For
an ordered score, repeat `--score-level` from lowest to highest.

Run `./jev-align optimize --help` for every setup flag. Flags configure a run;
human labeling and proposal acceptance remain interactive by design.

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
./jev-align functions
./jev-align optimize --resume .jev-align/runs/<run-id>
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
