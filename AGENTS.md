# AGENTS.md

This repository contains `jev-align`, an interactive active-learning CLI for
building AI Functions from labeled examples. Coding agents may help configure
and operate the workflow, but must preserve explicit control over how labels are
created and reviewed.

## Installation reference

Requires Python 3.11 or newer. For a released build, prefer an isolated CLI
installation:

```shell
uv tool install jev-align
# Without uv:
pip install jev-align
```

From a local checkout, use `uv tool install .`; after changing source without a
version bump, refresh it with `uv tool install --force --no-cache .` so uv does
not reuse the prior same-version wheel. To install straight from GitHub, use
`uv tool install "git+https://github.com/sutro-sh/jev-align.git"`.
For editable development, run `uv sync --extra dev` and launch with
`uv run jeva`.

Both `jeva` and `jev-align` invoke the same CLI. Interactive, non-editable
installs check PyPI for newer releases at startup. In virtual environments the
updater prefers `uv pip` when available and falls back to that environment's
Python and pip. Set `JEVA_DISABLE_UPDATE_CHECK=1` to disable the check.

## Operating the CLI for a user

### Core rule

In the guided labeling flow, do not skip examples or silently infer labels.
Enter exactly what the user chooses. If the user explicitly authorizes
synthetic, imported, or agent-generated labels, do not describe them as
human-created or human-reviewed. Human review is encouraged, especially for
important tasks and boundary cases, but it is not a registry-wide guarantee.
Never accept an optimized definition on the user's behalf. A rationale is
optional, but encourage one when it explains an important boundary or corrects
the model's reasoning.

### Before starting

1. Confirm that one Jev provider is configured: `TYPESAFE_API_KEY`,
   `AI_GATEWAY_API_KEY`, or both `CLOUDFLARE_ACCOUNT_ID` and
   `CLOUDFLARE_API_TOKEN`.
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
Jev evaluation requires the credentials for the selected evaluation backend;
the reflection provider is separate.

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

### Sharing functions through ai-functions.dev

`ai-functions.dev` is the public registry for portable AI Functions built with
Jeva. Its browser UI is read-only: users discover and inspect functions on the
site, while authentication, publishing, pulling, and unpublishing happen in the
CLI. The initial product supports public personal namespaces only. Do not imply
that private or team registries are available yet.

The service separates concerns as follows:

- GitHub supplies identity; it does not host function artifacts.
- D1 stores users, namespaces, memberships, tokens, searchable function
  metadata, version records, and aggregate statistics.
- R2 stores the exact immutable JSON artifact for each version.
- A function reference is `GITHUB_USER/FUNCTION_NAME`; the server derives the
  namespace from authenticated identity rather than trusting artifact data.

#### Authentication

Authenticate and publish a saved function with:

```shell
jeva login
jeva whoami
jeva push .jev-align/runs/RUN_ID
```

`jeva login` uses GitHub's device flow. The Worker exchanges the temporary
GitHub access token for a scoped Jeva API token and must immediately discard the
GitHub token. Jeva API tokens belong in the system keychain; the
permission-restricted local config is only a fallback. Never print tokens,
session cookies, keychain values, client secrets, or `TOKEN_PEPPER`. Use
`jeva logout` to revoke the active Jeva token and remove the local credential.

#### Publishing

`jeva push` requires an explicit public-data confirmation. For noninteractive
use, the human must first approve the exact public operation, then pass both
`--yes` and `--confirm-public-data`. Include `--name` on a first publish when
the generated default is inappropriate and `--description` to set or update the
public discovery description:

```shell
jeva push .jev-align/runs/RUN_ID \
  --name "Function name" \
  --description "Short public description" \
  --yes \
  --confirm-public-data
```

The server derives the namespace from the authenticated GitHub identity; never
accept a namespace from artifact data or a CLI flag. A successful first push
creates `GITHUB_USER/FUNCTION_NAME`. Re-publishing identical bytes is
idempotent; changed bytes create a new immutable numbered version. Local publish
metadata is stored in `published.json` inside the run.

Published artifacts include only:

- The accepted definition and portable input signature.
- Safe backend identity and learning configuration.
- Labeled inputs, labels, train/holdout splits, and rationales.
- A summary of the accepted function's metrics.

They exclude unlabeled source rows, local source paths, pending proposals,
provider options that may contain secrets, API credentials, and reflection
credentials. Never approve public-data confirmation for the user. If the
labeled inputs or rationales may be sensitive, stop and ask the user rather than
publishing. The registry does not verify annotation provenance. Encourage
publishers to disclose synthetic or agent-generated labels in the function
description when that distinction matters.

#### Pulling

Anyone can pull a public function without authenticating:

```shell
jeva pull GITHUB_USER/FUNCTION_NAME
jeva pull GITHUB_USER/FUNCTION_NAME --version 2
```

Pull verifies the artifact's SHA-256 digest and materializes the immutable
version as a normal run under `.jev-align/runs/`. Published annotations are
already labeled, so do not ask the user to repeat them merely because their
provenance is unknown. Do not claim they were human-reviewed unless the
publisher says so. Continue learning only when genuinely new captured inputs
are available. Never overwrite an existing pull destination; use `--output`
when a different location is needed.

#### Unpublishing

Owners can reversibly remove a function from discovery, its detail page, and
future public pulls:

```shell
jeva unpublish GITHUB_USER/FUNCTION_NAME
jeva unpublish GITHUB_USER/FUNCTION_NAME --yes
```

Unpublishing sets registry state; it does not delete D1 version records or R2
artifacts. Pushing the same function again restores it, including when the bytes
match an existing version. Do not describe unpublishing as erasing data, and
remember that it cannot revoke copies already downloaded. Permanent deletion is
not currently a user-facing operation.

#### Agent-driven registry flow

When the user asks an agent to share a function:

1. Identify the exact saved run and ensure it has an accepted definition.
2. Show the proposed name, description, reference, and public data boundary.
3. Obtain the user's explicit approval to publish labeled inputs and rationales,
   and accurately describe any known synthetic or agent-generated labeling.
4. Authenticate with `jeva login` if necessary; the user completes GitHub's
   browser/device authorization.
5. Run `jeva push` and report the canonical `https://ai-functions.dev/...` URL.
6. Never substitute a direct API request to bypass CLI confirmation or artifact
   validation.

For pulls, verify the requested reference and optional version, use `jeva pull`,
and preserve the resulting run as application state. For unpublishing, explain
that the operation is reversible and does not recall prior downloads before
asking for approval.

### Continual learning from live calls

Applications can call an accepted AI Function and record uncertain or randomly
audited predictions for later labeling and review. Install `jev-align` in the
application environment and load the saved run with capture enabled:

```python
from jev_align import AIFunction

is_aviation = AIFunction.load(
    ".jev-align/runs/RUN_ID",
    capture=True,
)

prediction = is_aviation(
    title="Airport expansion",
    text="A new runway opens next year.",
)
```

Calls must use the input columns configured for the saved run and return a
normalized `Prediction`, not a confirmed label. Capture adds no model call and writes
selected observations to `.jev-align/captures/` in the background. A recorder
created by `capture=True` is flushed automatically at process shutdown. Call the
AI Function's `close()` method or use it as a context manager when deterministic
shutdown is required. This integration is currently synchronous.

To continue learning, run `jeva functions`, open the matching AI Function, and
select **Resume learning**, or run:

```shell
jeva optimize --resume .jev-align/runs/RUN_ID
```

When the CLI offers newly captured calls, let the user decide whether to import
them. Every unique eligible call is imported; there is no capture-pool row cap.
The model's recorded prediction is never automatically treated as truth. Every
selected example must receive an explicit label, with an optional rationale,
before GEPA can learn from it. Labels may be user-created or generated with the
user's explicit authorization. Previously labeled captures remain in the
evaluation pool, while only unlabeled inputs are eligible for another annotation
batch.

The accepted definition is loaded when `AIFunction.load(...)` runs. Restart the
process or reload the AI Function to use a newly accepted version.
Use `JEVA_CAPTURE_DIR` when the application and CLI need to share a capture
directory outside the workspace.

`AIFunction` applies the saved run's column selection, normalization, and
concatenation. Its return value is a provider-neutral `Prediction`: binary tasks
expose `probability`, multiclass tasks `choice` and `confidence`, multilabel
tasks `label_probabilities`, and score tasks `score` and `confidence`. Runtime
calls require the selected Jev provider's credentials, but not a
reflection-model key. Pending
proposals are never loaded. A run with no accepted proposal uses its seed
definition.

Capture makes no additional JEV request and does not retry evaluations. By
default it records predictions with ambiguity of at least 0.8 and a random 5%
audit of the remainder. For binary tasks, 0.8 ambiguity corresponds to
probabilities from 0.4 through 0.6. Each record includes the evaluated input,
prediction, definition, model provenance, and selection reason, but none of
those values constitute a confirmed annotation.

Capture uses JSONL files under `.jev-align/captures/`; it requires no database or
server. Records enter a bounded queue and a background thread writes batches of
up to 64, flushing about every 250 ms. Defaults are 256 queued records, 64 KiB
per record, and 64 MiB per capture file. Full queues, oversized records, and
exhausted file budgets increment `dropped`. Disk errors disable that writer, log
one warning, and populate `error` without failing successful predictions. Limits
and files are per `Capture` instance, so create one inside each worker after
forking. For custom thresholds or storage settings, pass a configured
`Capture(...)` instance instead of `capture=True`; the caller then owns and must
close it. Existing files are not rotated or deleted automatically. Closing drains
the queue for up to five seconds; abrupt shutdown can lose buffered records.

On resume, the CLI searches `.jev-align/captures/` in the current workspace and
the run workspace, plus `JEVA_CAPTURE_DIR` when configured. It imports complete,
matching records already on disk, deduplicates repeated inputs, excludes the
original and holdout datasets, and ignores malformed, partial, or over-1-MiB
lines. Discovery itself makes no API call. Approved captured inputs are also
stored in the run's `captured-inputs.json`, so removing the raw logs does not
remove them from that run.

Captured inputs use the normal full-pool evaluation, prediction cache, and batch
selection. Confirmed labels from captures join training, while the original pool and
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
- `src/jev_align/jev.py`: direct TypeSafe JEV adapter.
- `src/jev_align/jev_gateways.py`: Cloudflare and Vercel Jev transports.
- `src/jev_align/models.py`: persisted schemas and normalized predictions.
- `src/jev_align/persistence.py`: run and label storage.
- `src/jev_align/runtime.py`: synchronous callable interface for saved functions.
- `src/jev_align/artifacts.py`: public artifact schema, construction, and pull
  materialization.
- `src/jev_align/registry_auth.py`: GitHub device login and local registry credentials.
- `src/jev_align/registry_client.py`: registry publish, pull, and unpublish HTTP client.
- `src/jev_align/capture.py`: bounded, best-effort JSONL capture of live predictions.
- `src/jev_align/captured_inputs.py`: discovery and deduplication of captured inputs.
- `ai-functions.dev/`: self-contained Cloudflare Worker and React registry application.

### Registry application development

`ai-functions.dev/` is intentionally self-contained so it can later move into a
separate repository. It requires Node.js 20.19 or newer and contains both the
React discovery UI and Cloudflare Worker API. Keep Python package imports and
build-time dependencies out of this directory; the integration boundary is the
versioned JSON artifact and HTTP API.

Important registry files are:

- `ai-functions.dev/src/`: the read-only React discovery and detail interface.
- `ai-functions.dev/worker/index.ts`: API routing and Worker entry point.
- `ai-functions.dev/worker/auth.ts`: GitHub browser/device authentication,
  scoped Jeva tokens, sessions, and namespace authorization.
- `ai-functions.dev/worker/publish.ts`: artifact validation, R2 writes, and
  immutable version creation.
- `ai-functions.dev/worker/functions.ts`: public feed, detail, artifact download,
  and download counters.
- `ai-functions.dev/worker/unpublish.ts`: owner/admin reversible unpublishing.
- `ai-functions.dev/migrations/`: ordered, append-only D1 schema migrations.
- `ai-functions.dev/public/AGENTS.md`: public instructions for agents consuming
  the site and CLI.

The production Worker is `ai-functions-dev-registry`. Its D1 database is
`ai-functions-dev-registry`, and its private R2 bucket is
`ai-functions-dev-artifacts`. R2 objects are never public directly; downloads
pass through the Worker so visibility, unpublishing, integrity headers, and
statistics are enforced. Treat D1 rows and R2 objects as coordinated state. Do
not delete or rewrite immutable function versions or artifacts during normal
feature work.

The core HTTP routes are:

- `GET /api/v1/functions`: public discovery feed.
- `GET /api/v1/functions/:namespace/:slug`: public function metadata and safe
  definition details.
- `GET /api/v1/functions/:namespace/:slug/versions/:version/artifact`: immutable
  artifact download.
- `PUT /api/v1/functions/:slug`: authenticated publish or republish; namespace
  comes from the token's GitHub identity.
- `DELETE /api/v1/functions/:namespace/:slug`: authenticated reversible
  unpublish for namespace owners/admins.
- `GET /api/auth/github/start` and `/api/auth/github/callback`: browser GitHub
  authorization.
- `POST /api/v1/auth/github/exchange`: CLI GitHub-to-Jeva token exchange.
- `GET /api/v1/me` and `DELETE /api/v1/tokens/current`: CLI identity and token
  revocation.

Public feed, detail, and artifact queries must all apply the same visibility and
`unpublished_at` rules. Publishing an unpublished function must restore it even
if its artifact digest already exists. Keep artifact downloads digest-verified
and do not add caching that could bypass a later unpublish check.

For local registry work:

```shell
cd ai-functions.dev
npm install
npm run db:migrate:local
npm run dev
```

Authentication testing additionally requires copying `.dev.vars.example` to
the ignored `.dev.vars` and supplying local GitHub App credentials and a random
`TOKEN_PEPPER`. Never commit `.dev.vars`, Wrangler state, D1 exports, API tokens,
session values, function artifacts, or production credentials. The local
callback URL is `http://127.0.0.1:5173/api/auth/github/callback`; production uses
`https://ai-functions.dev/api/auth/github/callback`.

After any registry change, run:

```shell
cd ai-functions.dev
npm run check
```

When the D1 schema changes, add a new forward-only migration and verify it with
`npm run db:migrate:local`; never edit a migration already applied remotely.
Only apply `npm run db:migrate:remote` or `npm run deploy` when the user has
explicitly asked to update production. Apply required migrations before the
Worker that depends on them.

The hosted registry uses a GitHub App for browser and CLI identity. Temporary
GitHub tokens must never be logged or persisted; the Worker exchanges them for
separately generated, scoped Jeva tokens. CLI tokens belong in the system
keychain, with a permission-restricted configuration file only as a fallback.
Namespace authorization must remain separate from authentication, and write
routes must enforce the `functions:write` scope plus the required namespace
role.

Capture files are unlabeled observations, not run state. Never treat their model
predictions as confirmed labels. Resuming a saved function offers to import matching
captured calls from the current or run workspace's `.jev-align/captures/` and
optional `JEVA_CAPTURE_DIR`. Resolve pending proposals first. Approved inputs
are persisted in `captured-inputs.json` inside the run; treat this as application
state. Confirmed labels from captures join training, while the original fixed pool
and holdout remain unchanged. Keep capture writes off the evaluation path and
preserve bounded queues, file-size limits, and nonblocking overflow behavior.
Original and captured pools share evaluation, prediction caching, and batch
selection. Every unique eligible captured input remains in the active evaluation
pool, including every labeled input. Filter labeled inputs only when selecting
the next annotation batch. Report captured-pool uncertainty separately so
original fixed-pool history stays comparable.

Jev can run through TypeSafe directly (`--backend typesafe`), Cloudflare
Workers AI (`--backend cloudflare`), or Vercel AI Gateway (`--backend vercel`).
Runs persist the provider-neutral backend configuration; `--jev-model` remains
a compatibility alias for `--backend-model`.

Preserve these product invariants when making changes:

- Guided labeling never silently invents or skips a label; synthetic or
  agent-generated labeling requires explicit user authorization.
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
