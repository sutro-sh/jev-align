# AGENTS.md

This repository contains `jev-align`, an interactive active-learning CLI for
building AI Functions from a user's judgments. Coding agents may help configure
and operate the workflow, but must preserve the human labeling loop.

## Installation reference

Requires Python 3.11 or newer. For a released build, prefer an isolated CLI
installation:

```shell
uv tool install jev-align
# Without uv:
pip install jev-align
```

From a local checkout, use `uv tool install .`; after pulling or changing the
source, refresh it with `uv tool install --force .`. To install straight from
GitHub, use `uv tool install "git+https://github.com/sutro-sh/jev-align.git"`.
For editable development, run `uv sync --extra dev` and launch with
`uv run jeva`.

Both `jeva` and `jev-align` invoke the same CLI. Interactive, non-editable
installs check PyPI for newer releases at startup. In virtual environments the
updater prefers `uv pip` when available and falls back to that environment's
Python and pip. Set `JEVA_DISABLE_UPDATE_CHECK=1` to disable the check.

## Operating the CLI for a user

### Core rule

The user supplies every label. Do not skip examples, silently infer labels, or
accept an optimized definition on the user's behalf. A rationale is optional,
but encourage one when it explains an important boundary or corrects the
model's reasoning.

### Before starting

1. Confirm that `TYPESAFE_API_KEY` is available.
2. Confirm a reflection provider is configured: `OPENAI_API_KEY`,
   `ANTHROPIC_API_KEY`/`CLAUDE_API_KEY`, `GEMINI_API_KEY`, or the endpoint and
   credentials required by a custom LiteLLM provider.
3. Identify the intended CSV, Parquet, or JSONL file without modifying it.
4. Establish the question, task type, input columns, and labels or score levels.
5. If any choice would materially change the task semantics, ask the user.

Never display secret values. It is enough to report whether a required key is
configured.

### Starting a run

Use the guided home screen when the user wants to choose interactively:

```shell
jeva
```

Use flags when the setup is already known:

```shell
jeva optimize DATA \
  --question "QUESTION" \
  --column COLUMN
```

Relevant task shapes:

- Binary: provide `--question`; optionally add concrete `--true-criteria` and
  `--false-criteria`.
- Multiclass: repeat `--class "NAME=DESCRIPTION"` for mutually exclusive
  labels.
- Multilabel: repeat `--class "NAME=DESCRIPTION"` and add `--multilabel`.
- Score: repeat `--score-level "DESCRIPTION"` in lowest-to-highest order.

The installed package includes preconfigured Hacker News, support-ticket, and
agent-trace examples. The guided dataset picker lists those first, followed by
CSV, Parquet, and JSONL files discovered below the current directory.

Use `--batch-size` to choose the number of training annotations per round. The
guided Advanced menu offers 5, 10, 15, or 20. Add `--holdout` only when the user
wants a 20% reserved evaluation split; this adds 20% extra held-out annotations
per round. Advanced also configures maximum GEPA metric calls, which defaults to
300. For scripted runs, use `--max-metric-calls`; `--metric-budget` remains an
alias.

Use `--all-columns-concatenated` only when every field is useful. Prefer
explicit `--column` values when IDs, timestamps, or metadata could distract the
evaluator. The normal default is the first 1,000 rows or all rows for a smaller
dataset; only set `--pool-size` when the user wants a different limit.

Run the CLI in a real PTY when possible so arrow-key menus, progress displays,
and prompts work correctly.

### Reflection providers

GEPA's reflection model is separate from the TypeSafe JEV evaluation model.
Reflection uses LiteLLM model identifiers. OpenAI, Anthropic, and Gemini are
listed automatically when their standard keys are present. For another
provider, pass `--reflection-model provider/model`; in the wizard select
**Choose a different model** and then **Enter a custom LiteLLM model**.

Fireworks example:

```shell
export FIREWORKS_API_KEY="..."
jeva optimize DATA \
  --question "QUESTION" \
  --column COLUMN \
  --reflection-model \
    "fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct"
```

For a local or hosted vLLM server exposing an OpenAI-compatible `/v1` API:

```shell
export HOSTED_VLLM_API_BASE="http://localhost:8000/v1"
export HOSTED_VLLM_API_KEY="..." # Omit when the endpoint has no authentication.
jeva optimize DATA \
  --question "QUESTION" \
  --column COLUMN \
  --reflection-model "hosted_vllm/Qwen/Qwen3-8B"
```

For a generic OpenAI-compatible endpoint:

```shell
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="local" # Replace when the endpoint requires a real key.
jeva optimize DATA \
  --question "QUESTION" \
  --column COLUMN \
  --reflection-model "openai/Qwen/Qwen3-8B"
```

The provider/model identifier and environment variables must follow the
[LiteLLM provider configuration](https://docs.litellm.ai/docs/providers).
`TYPESAFE_API_KEY` remains required for JEV evaluation.

### Labeling rounds

For each displayed item:

1. Relay the content and choices clearly if the user cannot see the terminal.
2. Mention its uncertainty and whether it is a random audit sample.
3. Ask the user for the label.
4. Ask for an optional rationale, especially on ambiguous or surprising cases.
5. Enter exactly what the user chose.

Held-out cards are explicitly marked. Collect their labels normally, but never
reuse their labels or rationales to guide task wording. The CLI keeps them out
of GEPA and reports their score separately.

There is no skip action. At the label picker, `b` removes the previous answer
and moves backward. At the rationale prompt, `/back` returns to the current
label picker; a literal `b` is valid rationale text. At the first item of a
later round, `b` can rewind the prior optimization round after confirmation.

For multilabel tasks, use Up/Down to navigate, Space to toggle, and Enter to
confirm. An empty selection is a valid judgment.

### Reviewing a GEPA proposal

After optimization, summarize for the user:

- The task metric before and after.
- The fixed-pool certainty before and after.
- Any regression, even when the task metric improved.
- The material changes in the proposed definition.

Then ask the user whether to accept, reject, or quit and resume later. Do not
treat a higher training score as automatic approval. The score uses accumulated
labels and is not a held-out generalization estimate.

### Existing runs

Browse saved AI Functions with:

```shell
jeva functions
```

Resume a known run directly with:

```shell
jeva optimize --resume .jev-align/runs/RUN_ID
```

Saved state, labels, prediction caches, GEPA artifacts, and rewind archives live
inside the run directory. Treat them as application state: do not hand-edit,
delete, or replace them. If a run is pending at the proposal screen, resume that
decision rather than starting another optimization.

Running an accepted function on another dataset writes JSONL results to
`.jev-align/outputs/` without changing the saved function.

### Continual learning from live calls

Applications can call an accepted AI Function and record uncertain or randomly
audited predictions for later human labeling. Install `jev-align` in the
application environment, create one `Capture` for each worker process, and pass
it to the synchronous `aligned` decorator:

```python
from jev_align import Capture, aligned

with Capture(ambiguity_threshold=0.8, audit_rate=0.05) as captures:
    @aligned(".jev-align/runs/RUN_ID", capture=captures)
    def is_aviation(title, text):
        return {"title": title, "text": text}

    prediction = is_aviation(
        "Airport expansion",
        "A new runway opens next year.",
    )
```

The decorated function's returned mapping must use the input columns configured
for the saved run. It returns a normalized `Prediction`; it does not return a
human label. `Capture` adds no model call and writes selected observations to
`.jev-align/captures/` in the background. Use it as a context manager or call
`close()` during shutdown so queued records can be flushed. This integration is
currently synchronous.

To continue learning, run `jeva functions`, open the matching AI Function, and
select **Resume learning**, or run:

```shell
jeva optimize --resume .jev-align/runs/RUN_ID
```

When the CLI offers newly captured calls, let the user decide whether to import
them. Every unique eligible call is imported; there is no capture-pool row cap.
The model's recorded prediction is never treated as truth. The user must label
every selected example, with an optional rationale, before GEPA can learn from
it. Previously labeled captures remain in the evaluation pool, while only
unlabeled inputs are eligible for another annotation batch.

Accepted definitions are loaded when the decorated function is created. Restart
the process or recreate the decorated function to use a newly accepted version.
Use `JEVA_CAPTURE_DIR` when the application and CLI need to share a capture
directory outside the workspace.

The decorator applies the saved run's column selection, normalization, and
concatenation. Its return value is a provider-neutral `Prediction`: binary tasks
expose `probability`, multiclass tasks `choice` and `confidence`, multilabel
tasks `label_probabilities`, and score tasks `score` and `confidence`. Runtime
calls require `TYPESAFE_API_KEY`, but not a reflection-model key. Pending
proposals are never loaded. A run with no accepted proposal uses its seed
definition.

Capture makes no additional JEV request and does not retry evaluations. By
default it records predictions with ambiguity of at least 0.8 and a random 5%
audit of the remainder. For binary tasks, 0.8 ambiguity corresponds to
probabilities from 0.4 through 0.6. Each record includes the evaluated input,
prediction, definition, model provenance, and selection reason, but none of
those values constitute a human label.

Capture uses JSONL files under `.jev-align/captures/`; it requires no database or
server. Records enter a bounded queue and a background thread writes batches of
up to 64, flushing about every 250 ms. Defaults are 256 queued records, 64 KiB
per record, and 64 MiB per capture file. Full queues, oversized records, and
exhausted file budgets increment `dropped`. Disk errors disable that writer, log
one warning, and populate `error` without failing successful predictions. Limits
and files are per `Capture` instance, so create one inside each worker after
forking. Existing files are not rotated or deleted automatically. Closing drains
the queue for up to five seconds; abrupt shutdown can lose buffered records.

On resume, the CLI searches `.jev-align/captures/` in the current workspace and
the run workspace, plus `JEVA_CAPTURE_DIR` when configured. It imports complete,
matching records already on disk, deduplicates repeated inputs, excludes the
original and holdout datasets, and ignores malformed, partial, or over-1-MiB
lines. Discovery itself makes no API call. Approved captured inputs are also
stored in the run's `captured-inputs.json`, so removing the raw logs does not
remove them from that run.

Captured inputs use the normal full-pool evaluation, prediction cache, and batch
selection. Human labels from captures join training, while the original pool and
holdout remain fixed. Reports show captured-pool uncertainty separately so the
original fixed-pool history remains comparable. Pending GEPA proposals must be
resolved before importing new captures, and every new proposal still requires
explicit human acceptance.

## Helping develop the repository

The main modules are:

- `src/jev_align/cli.py`: interactive flows, rendering, and commands.
- `src/jev_align/session.py`: round lifecycle, caching, rewind, and decisions.
- `src/jev_align/optimizer.py`: GEPA integration and task metrics.
- `src/jev_align/backends.py`: provider-neutral backend contract and factory.
- `src/jev_align/jev.py`: TypeSafe JEV adapter.
- `src/jev_align/models.py`: persisted schemas and normalized predictions.
- `src/jev_align/persistence.py`: run and label storage.
- `src/jev_align/runtime.py`: synchronous `aligned` decorator for saved functions.
- `src/jev_align/capture.py`: bounded, best-effort JSONL capture of live predictions.
- `src/jev_align/captured_inputs.py`: discovery and deduplication of captured inputs.

Capture files are unlabeled observations, not run state. Never treat their model
predictions as human labels. Resuming a saved function offers to import matching
captured calls from the current or run workspace's `.jev-align/captures/` and
optional `JEVA_CAPTURE_DIR`. Resolve pending proposals first. Approved inputs
are persisted in `captured-inputs.json` inside the run; treat this as application
state. Human labels from captures join training, while the original fixed pool
and holdout remain unchanged. Keep capture writes off the evaluation path and
preserve bounded queues, file-size limits, and nonblocking overflow behavior.
Original and captured pools share evaluation, prediction caching, and batch
selection. Every unique eligible captured input remains in the active evaluation
pool, including every labeled input. Filter labeled inputs only when selecting
the next annotation batch. Report captured-pool uncertainty separately so
original fixed-pool history stays comparable.

TypeSafe JEV is currently the only included evaluation backend. Runs persist a
provider-neutral backend configuration so additional adapters can be added;
`--jev-model` remains a compatibility alias for `--backend-model`.

Preserve these product invariants when making changes:

- The human explicitly confirms every label.
- Proposed prompts are shown as a diff before acceptance.
- The accepted candidate seeds the next optimization round.
- Full-pool uncertainty remains comparable across rounds.
- Old run state remains loadable through explicit migrations.
- Backends must declare supported task types and usable uncertainty signals.
- Provider-specific SDK objects do not cross the backend boundary.

Before handing off a code change, run:

```shell
uv run --no-active --quiet pytest
uv run --no-active --quiet ruff check src tests
```

Do not perform live API evaluations unless the user explicitly asks; the normal
test suite is designed to run without them.

## Publishing a release

`pyproject.toml` is the authoritative version source. Prepare a release with:

```shell
uv version 0.1.0
git add pyproject.toml uv.lock
git commit -m "release 0.1.0"
make release VERSION=0.1.0
```

The Make target requires a clean worktree, verifies the configured version,
runs tests and lint, builds both distributions, and runs `twine check`. It then
pushes an annotated `v<VERSION>` tag and creates a GitHub Release with generated
notes. Publishing is not performed from the developer machine.

Publishing the GitHub Release triggers `.github/workflows/publish.yml`. The
workflow repeats validation from the tagged commit and publishes through PyPI
Trusted Publishing. Its PyPI publisher must be configured for the
`sutro-sh/jev-align` repository, `publish.yml` workflow, and `pypi` environment.
No PyPI token should be added to the repository or GitHub secrets.
