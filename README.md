# jev-align

`jev-align` is an experimental CLI for aligning [TypeSafe's Jev](https://docs.typesafe.ai/introduction) to your
judgment, built by [Sutro](https://sutro.sh/). 

It finds uncertain rows offkine in a sample dataset, asks you to label them, and uses
[GEPA](https://gepa-ai.github.io/gepa/) to improve Jev's accuracy. Then decorate your future calls to Jev, and keep learning from production examples. 

It's last-mile, feedback-driven fine-tuning, for Jev.

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

## Calling a saved function from Python

Install `jev-align` in your application's Python environment as well as the CLI
environment. The synchronous `aligned` decorator loads the current saved
definition once and turns a function returning input fields into a JEV call:

```python
from jev_align import Capture, aligned

# Create one recorder for the application's lifetime, inside each worker process.
with Capture(ambiguity_threshold=0.8, audit_rate=0.05) as captures:
    @aligned(".jev-align/runs/<run-id>", capture=captures)
    def is_aviation(title, text):
        return {"title": title, "text": text}

    prediction = is_aviation("Airport expansion", "A new runway opens next year.")
    print(prediction.probability >= 0.5)

print(captures.path)
print(captures.stats)  # written, dropped, error
```

Use the input columns configured in your saved run. Column selection, text
normalization, and concatenation match dataset evaluation. The return value is
a provider-neutral `Prediction`: binary tasks expose `probability`, multiclass
tasks `choice` and `confidence`, multilabel tasks `label_probabilities`, and
score tasks `score` and `confidence`. Only `TYPESAFE_API_KEY` is needed for these
calls; reflection is not involved. Pending proposals are never used. Recreate
the decorated function to load a newly accepted definition. A run that has not
accepted a proposal still uses its original seed definition.

Omit `capture=captures` to evaluate without recording. Capture makes no extra JEV
requests and does not retry evaluations. It keeps predictions with ambiguity
at least 0.8, plus a random 5% of the remaining calls. For binary tasks, that
threshold corresponds to probabilities from 0.4 through 0.6. Captures contain
the evaluated input, prediction, definition, model provenance, and selection
reason; they never become human labels automatically.

Capture uses JSONL files under `.jev-align/captures/`, with no database or
server. Records are serialized in the caller and offered to a bounded queue;
one background thread writes batches of up to 64 records, flushing roughly
every 250 ms. Defaults are 256 queued records, 64 KiB per record, and 64 MiB
per capture file. Full queues, oversized records, and exhausted file budgets
drop captures and increment `dropped`. Disk errors disable the writer, log one
warning, and populate `error`; successful predictions still return. Limits are
per `Capture` instance, with a separate file for each instance. Files from
previous instances are retained; there is no automatic rotation or cleanup.

Use a context manager or call `captures.close()` at shutdown to drain the queue
(up to five seconds). Abrupt exits can lose buffered records. This first
version supports synchronous calls only and does not limit your application's
JEV request concurrency.

To label captured inputs later, open the saved function and select **Resume
learning**:

```shell
jeva functions
```

When new matching calls are available, the CLI offers to use them. Choose yes to
retain every previously labeled capture, import every unique eligible unlabeled
input, and refresh their uncertainty under the current definition.
Each labeling batch includes ambiguous examples and a random audit sample.
Smaller final batches are supported. Choose no to continue with the previously
approved pool, or the original dataset when no captured pool has been approved.

Captured inputs use the same full-pool evaluation, caching, and batch-selection
logic as the original dataset. Labeled inputs stop being offered for annotation,
but remain in the pool when proposed definitions are evaluated. Accepting a
proposal reuses those predictions for the next batch. Reports show captured-pool
uncertainty separately from the original fixed evaluation pool; both compare the
same inputs before and after a proposal.

The CLI looks in `.jev-align/captures/` under both the current working directory
and the saved run's workspace. Set `JEVA_CAPTURE_DIR=/path/to/captures` to also
search a custom location. Only calls matching the run are considered. Repeated
inputs and inputs already in the original dataset (including holdout inputs) are
excluded. Capture discovery itself makes no API calls. The scanner imports all
complete records already written to disk; buffered, incomplete, malformed, and
oversized records (over 1 MiB) are not imported.

Approved inputs are saved separately inside the run before labeling, so they
remain available even if the capture logs are later removed. New human labels
join the existing training labels; prediction metadata never supplies labels.
The original evaluation pool and holdout stay fixed, and proposed definitions
still require acceptance. Pending proposals are reviewed before offering more
captures. Direct `jeva optimize --resume RUN` uses the same prompt.

## Using a coding agent

See [AGENTS.md](AGENTS.md) for detailed setup, provider configuration, CLI
operation, and development guidance for coding agents.
