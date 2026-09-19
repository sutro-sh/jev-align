# jev-align

`jev-align` is an experimental CLI from [Sutro](https://sutro.sh/) for building
AI Functions with [TypeSafe's Jev](https://docs.typesafe.ai/introduction).

It finds uncertain examples, asks you to label them, and uses
[GEPA](https://gepa-ai.github.io/gepa/) to improve the function. Use it in your
application and keep learning from production examples.

## Quick start

Requires Python 3.11 or newer.

```shell
uv tool install jev-align
export TYPESAFE_API_KEY="..."
export OPENAI_API_KEY="..." # or ANTHROPIC_API_KEY / GEMINI_API_KEY
jeva
```

Start the CLI with either `jeva` or `jev-align`.

Use `pip install jev-align` if you do not use
[uv](https://docs.astral.sh/uv/). The guided setup discovers local CSV,
Parquet, and JSONL files and includes three ready-to-run examples.

## How it works

Each round:

1. Evaluates the configured dataset and measures uncertainty.
2. Selects ambiguous rows plus a random audit sample for you to label.
3. Uses your accumulated labels and optional rationales to run GEPA.
4. Shows the score, certainty change, and proposed definition diff.
5. Lets you accept, reject, rewind, or resume later.

Every label comes from you. A higher training score never accepts a proposal
automatically.

## Task types

| Type | Output |
| --- | --- |
| Binary | `True` or `False` |
| Multiclass | Exactly one fixed label |
| Multilabel | Zero or more fixed labels |
| Score | One level from an ordered rubric |

## Configuration

The guided **Advanced** menu configures:

- 5, 10, 15, or 20 training annotations per round.
- An optional 20% held-out evaluation set.
- GEPA's metric-call budget, which defaults to 300.

By default, `jev-align` uses the first 1,000 rows—or the entire dataset when it
is smaller—and lets you concatenate all fields or select specific columns.

Everything can also be configured with flags:

```shell
jeva optimize posts.csv \
  --question "Is the post related to aviation?" \
  --column title \
  --column text \
  --pool-size 1000
```

Use repeated `--class "NAME=DESCRIPTION"` options for multiclass or multilabel
tasks, and repeated `--score-level` options for scoring tasks. Run
`jeva optimize --help` for the complete flag reference.

## Reflection models

GEPA's reflection model is separate from the JEV model evaluating your data.
OpenAI, Anthropic, and Gemini models are detected automatically. Any
[LiteLLM provider](https://docs.litellm.ai/docs/providers)—including Fireworks,
local vLLM, and other OpenAI-compatible endpoints—can be supplied with
`--reflection-model provider/model`.

```shell
export HOSTED_VLLM_API_BASE="http://localhost:8000/v1"
jeva optimize data.csv --question "Is this relevant?" --column text \
  --reflection-model "hosted_vllm/Qwen/Qwen3-8B"
```

## Controls

- Arrow keys and Enter navigate menus.
- `b` returns to the previous label; `/back` leaves the rationale prompt.
- Space toggles choices in multilabel tasks.

## Saved AI Functions

```shell
jeva functions
jeva optimize --resume .jev-align/runs/<run-id>
```

## Keep learning from production

Use an AI Function in your application and capture production examples with a
decorator:

```python
from jev_align import Capture, aligned

with Capture() as captures:
    @aligned(".jev-align/runs/<run-id>", capture=captures)
    def is_aviation(title, text):
        return {"title": title, "text": text}

    prediction = is_aviation("Airport expansion", "A new runway opens next year.")
```

Later, resume the AI Function and label the captured examples. GEPA uses that
feedback to propose the next version:

```shell
jeva functions
```

## Using a coding agent

See [AGENTS.md](AGENTS.md) for detailed setup, provider configuration, CLI
operation, and development guidance for coding agents.

## Affiliation

Sutro is not affiliated with TypeSafe AI, the makers of Jev.
