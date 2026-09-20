# ai-functions.dev agent guide

ai-functions.dev is the public registry for portable AI Functions built with
Jeva. The website is read-only. Create, improve, and publish functions with the
`jeva` CLI.

Project: https://github.com/sutro-sh/jev-align
Registry: https://ai-functions.dev

## Labeling rule

The user controls how labels are created and decides whether to accept every
optimized definition. In the guided flow, never infer or skip labels unless the
user explicitly authorizes synthetic or agent-generated labeling. Do not
describe generated labels as human-reviewed. A rationale is optional, and human
review is encouraged for important tasks and boundary cases.

## Install

Requires Python 3.11 or newer.

    uv tool install jev-align

Without uv:

    pip install jev-align

Both `jeva` and `jev-align` invoke the same CLI. Use a real PTY for interactive
menus, label selection, progress displays, and proposal review.

## Discover public functions

Browse https://ai-functions.dev or read the public JSON APIs:

    GET https://ai-functions.dev/api/v1/functions
    GET https://ai-functions.dev/api/v1/functions/NAMESPACE/SLUG

The detail response includes the public definition, input signature, backend,
metrics, and version history. The website can show the labeled examples
and rationales for a selected version. Published annotations are public data.

## Pull a function

No login is required:

    jeva pull NAMESPACE/SLUG
    jeva pull NAMESPACE/SLUG --version 2

Pull verifies the immutable artifact digest and creates a normal saved function
under `.jev-align/runs/`. Do not overwrite or hand-edit its saved state. Existing
published annotations are already labeled; do not ask the user to label them
again.

Use `jeva functions` to inspect, run, or resume a local function. Continuing
learning requires new captured inputs; their model predictions are never labels.

## Create and optimize a function

Start the guided flow:

    jeva

Or provide known setup with flags:

    jeva optimize DATA.csv \
      --question "Is the post related to aviation?" \
      --column title \
      --column text

Before starting, confirm that the user has configured a Jev evaluation provider
and a GEPA reflection provider. Never print secret values.

During each guided labeling round:

1. Show the input, uncertainty, and whether it is a random audit sample.
2. Ask the user for the label, or follow the synthetic/agent-labeling method they
   explicitly authorized.
3. Ask for an optional rationale.
4. Enter exactly the supplied or authorized generated answer.
5. After GEPA runs, summarize the metric change, certainty change, regression
   risk, and definition diff.
6. Ask the user to accept, reject, or stop and resume later.

There is no skip action. At the label picker, `b` goes backward. At the
rationale prompt, `/back` returns to the label picker; a literal `b` is valid
rationale text.

## Authenticate and publish

    jeva login
    jeva push .jev-align/runs/RUN_ID

The registry derives the namespace from the authenticated GitHub identity. A
push publishes the accepted definition, safe backend identity, learning
configuration, and labeled inputs, labels, splits, and rationales. It does
not publish unlabeled source rows, local source paths, pending proposals,
backend credentials, or reflection-provider credentials.

Publishing requires the human's explicit confirmation that labeled inputs and
rationales may be public. Do not provide that confirmation on the user's behalf.

For agent-driven noninteractive publishing, the user must have already approved
the exact public operation. Then use:

    jeva push .jev-align/runs/RUN_ID \
      --name "Function name" \
      --description "Short public discovery description." \
      --yes \
      --confirm-public-data

Owners can remove a function from discovery and future public pulls without
deleting its immutable versions:

    jeva unpublish NAMESPACE/SLUG

Pushing the function again restores it. Unpublishing cannot revoke artifacts
that another user already downloaded.

Descriptions are optional plain text up to 280 characters. Pushing again with
`--description` updates the description by creating a new immutable version.
Omitting the option preserves the current description.

## Continual learning

Applications may load a saved function with capture enabled:

    from jev_align import AIFunction

    function = AIFunction.load(
        ".jev-align/runs/RUN_ID",
        capture=True,
    )

Captured calls are unlabeled observations. Resume with `jeva functions` or:

    jeva optimize --resume .jev-align/runs/RUN_ID

Let the user decide whether to import newly captured calls, then preserve the
same explicit labeling and proposal-acceptance rules.

## Safety and data handling

- Never expose API keys, GitHub tokens, session cookies, or keychain contents.
- Never treat a model prediction as a confirmed label without explicit user
  authorization.
- Never mutate saved run files by hand.
- Never publish without explicit public-data approval.
- Never claim training metrics are held-out generalization results.
- Prefer the CLI over direct write requests to the registry API.

For repository development instructions, read:
https://github.com/sutro-sh/jev-align/blob/main/AGENTS.md
