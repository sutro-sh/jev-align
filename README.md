# jev-align

`jev-align` is an experimental CLI from [Sutro](https://sutro.sh/) for building
AI Functions with decision models from TypeSafe, Perplexity, OpenAI, Liquid AI,
and Cloudflare.

It finds uncertain examples, asks you to label them, and uses
[GEPA](https://gepa-ai.github.io/gepa/) to improve the function. Use it in your
application, keep learning from production examples, and share portable
functions through [ai-functions.dev](https://ai-functions.dev/).

## Demo

https://github.com/user-attachments/assets/81650587-e3f1-4655-8213-ed5f6e120e9a

## Quick start

Requires Python 3.11 or newer.

```shell
uv tool install jev-align
export TYPESAFE_API_KEY="..." # Or configure another model backend below
export OPENAI_API_KEY="..." # or ANTHROPIC_API_KEY / GEMINI_API_KEY
jeva
```

CLI aliases: `decal`, `decision-align`, `jeva`, `jev-align`.
Install aliases: `uv tool install decision-align` or `uv tool install decal`.

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

The guided workflow asks you to review every label. Labels may be human-created,
synthetic, imported, or agent-assisted; review them to the standard your use
case requires. A higher training score never accepts a proposal automatically.

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
- GEPA's metric-call budget for guided labeling, which defaults to 300.

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

## Model backends

All backends support binary, multiclass, multilabel, and score tasks with
text/JSON inputs. The guided setup detects configured providers.

| `--backend` | Default `--backend-model` | Credentials |
| --- | --- | --- |
| `typesafe` | `jev-1.14.0` | `TYPESAFE_API_KEY` |
| `vercel` | `typesafe-ai/jev-1.14.0` | `AI_GATEWAY_API_KEY` |
| `cloudflare` | `typesafe/jev-1.14.0` | `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` |
| `perplexity` | `pplx-decider-v1.2` | `PERPLEXITY_API_KEY` |
| `openai` | `gpt-6-luna` (Decisions API) | `OPENAI_API_KEY` |
| `liquid` | `d1.1` | `LIQUID_API_KEY` |
| `openrouter` | `typesafe/jev-1.14` | `OPENROUTER_API_KEY` |

Set the credentials above, then select a backend and optional model:

```shell
jeva optimize data.csv --question "Is this relevant?" --column text \
  --backend openrouter --backend-model cloudflare/clef-flash
```

Cloudflare also offers `@cf/cloudflare/clef` and `@cf/cloudflare/clef-flash`.
OpenRouter offers Jev, Perplexity Decider, GPT-6 Luna Decisions, Liquid D1.1,
Clef, Clef Flash, and custom decision-model IDs using only `OPENROUTER_API_KEY`.

New runs default to TypeSafe Jev 1.14. Saved runs retain their backend and model;
credentials come from the environment. `--jev-model` remains an alias for
`--backend-model`. GEPA reflection is configured separately.

Backends preserve native probabilities and confidence and reject malformed
responses. Transient failures receive up to four attempts, honoring server
retry delays; delays over 60 seconds fail immediately.

## Reflection models

GEPA's reflection model is separate from the decision model evaluating your data.
The setup menu detects OpenAI, Anthropic, and Gemini keys.

| Provider | Default for new runs | Other menu choices |
| --- | --- | --- |
| OpenAI | GPT-6 Luna | GPT-6.1 Sol, GPT-6.1 Terra, GPT-6 Astra |
| Anthropic | Claude Haiku 5.5 | Claude Sonnet 5.5, Claude Opus 5.5, Claude Fable 5.1 |
| Gemini | Gemini 3.8 Flash | Gemini 4 Argon, Gemini 3.5 Flash-Lite, Gemini 3.1 Pro Preview |

New runs prefer OpenAI, then Anthropic, then Gemini. Saved runs retain their
model. Use `--reflection-model provider/model` for any
[LiteLLM provider](https://docs.litellm.ai/docs/providers), including local vLLM:

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

Browse, run, or resume functions saved on your computer:

```shell
jeva functions
jeva optimize --resume .jev-align/runs/<run-id>
```

## Share on ai-functions.dev

[ai-functions.dev](https://ai-functions.dev/) is the public registry for AI
Functions built with jev-align. Browse shared functions on the website and
publish your own from the CLI.

To publish, open your saved functions:

```shell
jeva functions
```

Select a function, then choose **Push to ai-functions.dev**. If needed,
jev-align walks you through GitHub login. You can name the function, add a
description, review what will be public, and confirm the push. It will appear at
`ai-functions.dev/<github-user>/<function-name>`.

A published function includes everything another person needs to run it or
continue improving it:

- The accepted definition and input signature.
- The runtime backend and learning configuration.
- Labeled inputs, labels, splits, and optional rationales.

It does **not** contain unlabeled source rows, local dataset paths, API keys,
reflection-provider credentials, or an unaccepted proposal. The registry is
currently public, so do not publish labeled data you cannot share.
Human review is encouraged for important tasks, but the registry does not claim
or verify that every published annotation was created or reviewed by a person.

Anyone can pull a public function without signing in:

```shell
jeva pull github-user/function-name
jeva pull github-user/function-name --version 2
```

Pulling needs no login. It verifies the immutable artifact and creates a normal
saved AI Function under `.jev-align/runs/`, ready to run or continue improving.

Owners can remove a function from discovery and future public pulls:

```shell
jeva unpublish github-user/function-name
```

Unpublishing is reversible: versions and annotations are retained, and pushing
the function again restores it. It cannot revoke copies that were already
downloaded.

## Keep learning from production

Load an AI Function in your application and capture useful production examples:

```python
from jev_align import AIFunction

is_aviation = AIFunction.load(
    ".jev-align/runs/<run-id>",
    capture=True,
)

prediction = is_aviation(
    title="Airport expansion",
    text="A new runway opens next year.",
)
```

Later, resume the AI Function and label the captured examples. GEPA uses that
feedback to propose the next version:

```shell
jeva functions
```

## Experimental squash mode

Squash is an intentionally dangerous, label-free demo that asks GEPA to reduce
the evaluation model's uncertainty on a fixed dataset. Lower uncertainty is not
correctness: a run can succeed by confidently assigning every row the same answer.

Select squash under **Advanced**, or pass `--squash`:

```shell
jeva optimize posts.csv --squash \
  --question "Is the post related to aviation?" \
  --column title \
  --column text
```

It supports every task type, preserves configured labels or score levels, uses
an automatic metric budget, and runs up to five search passes by default. The
final definition still requires approval. Squash records no annotations and
cannot be published.

## Using a coding agent

See [AGENTS.md](AGENTS.md) for detailed setup, provider configuration, CLI
operation, and development guidance for coding agents.

## Affiliation

Sutro is not affiliated with TypeSafe AI, the makers of Jev.
