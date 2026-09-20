# ai-functions.dev

The hosted registry for portable AI Functions built with Jeva. This directory is
self-contained so it can move to a separate repository without depending on the
Python implementation.

## Local development

Requires Node.js 20.19 or newer.

```shell
npm install
npm run db:migrate:local
npm run dev
```

The React frontend and API Worker run together through Cloudflare's Vite plugin.
The API health endpoint is available at `/api/health`.

Before authentication work, copy `.dev.vars.example` to `.dev.vars` and fill in
the local GitHub App values. `.dev.vars` is ignored by Git.

## GitHub authentication

Create a GitHub App with:

- Homepage URL: the application origin
- Callback URL: `<origin>/api/auth/github/callback`
- Device Flow enabled
- Expiring user authorization tokens enabled
- No repository or organization permissions for the initial personal registry

For local development, use
`http://127.0.0.1:5173/api/auth/github/callback`. Copy `.dev.vars.example` to
`.dev.vars` and configure:

```dotenv
GITHUB_APP_CLIENT_ID=Iv1...
GITHUB_APP_CLIENT_SECRET=...
TOKEN_PEPPER=...
```

`TOKEN_PEPPER` should be a long, random secret. It is used to hash web sessions
and CLI tokens before D1 storage. The Worker never stores the temporary GitHub
token submitted by `jeva login`.

The website uses GitHub's browser authorization flow. The CLI uses GitHub's
device flow, exchanges the temporary GitHub token for a scoped Jeva token, and
then discards the GitHub token.

## Publishing

Authenticated clients upload a schema-versioned JSON artifact with
`PUT /api/v1/functions/:slug`. The Worker derives the personal namespace from
the token, validates the public artifact, stores its exact bytes in R2, and
writes searchable metadata to D1. An identical digest is idempotent; changed
artifacts become immutable numbered versions. The initial registry is public
only and accepts artifacts up to 50 MiB.

Registry metadata is split by behavior: `functions.metadata_json` holds
optional descriptive fields, `function_stats` holds fast function-level totals,
and `function_version_stats` preserves download and run attribution by version.
`function_likes` is the canonical per-user relationship, with its aggregate
count denormalized into `function_stats`. Artifact downloads increment both
function and version totals transactionally. Run counts must come from an
explicit, opt-in client signal rather than hidden runtime telemetry.

Public discovery endpoints are:

- `GET /api/v1/functions` for the latest registry cards.
- `GET /api/v1/functions/:namespace/:slug` for the safe definition, input
  signature, metrics, and version history. Labeled rows are excluded.
- `GET /api/v1/functions/:namespace/:slug/versions/:version/artifact` for the
  immutable, digest-identified artifact used by `jeva pull`.

From the Python CLI:

```shell
JEVA_REGISTRY_URL=http://127.0.0.1:5173 jeva login
JEVA_REGISTRY_URL=http://127.0.0.1:5173 jeva push ../.jev-align/runs/<run-id>
```

Owners may reversibly hide a published function with
`jeva unpublish NAMESPACE/SLUG`. The Worker records `unpublished_at`; public
feed, detail, and artifact queries exclude the function, and a later push clears
the marker. Immutable versions and R2 artifacts are retained.

## Validation

```shell
npm run check
```

## Cloudflare resources

`wrangler.jsonc` contains production bindings and variables.
`wrangler.local.jsonc` isolates local D1/R2 emulation and the local callback
origin; Vite selects it automatically during development. GitHub credentials
and `TOKEN_PEPPER` stay encrypted as Worker secrets and must be configured with
`wrangler secret put` or `wrangler secret bulk` before deployment.

Do not commit `.dev.vars`, Wrangler state, database exports, API tokens, or
function artifacts.
