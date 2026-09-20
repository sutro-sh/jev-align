# AGENTS.md

This directory contains the Cloudflare-hosted AI Function registry. It is kept
self-contained because it will move to a separate repository.

## Boundaries

- The public Python CLI lives in the parent repository under `src/jev_align/`.
- Communicate with the CLI through versioned HTTP APIs and function artifacts.
- Do not import Python implementation details into this application.
- Function artifacts contain accepted definitions and human annotations, but
  never API keys, raw unlabeled datasets, GEPA logs, or pending candidates.
- Published versions are immutable. Updating a function creates a new version.
- Personal namespaces use immutable GitHub user IDs internally; GitHub logins
  are mutable display and routing identifiers.
- GitHub browser and device tokens are only used to verify identity. Never log
  or persist them. CLI access uses separately generated, scoped Jeva tokens.
- Hash sessions and API tokens with `TOKEN_PEPPER` before D1 storage. Never
  store the plaintext registry token server-side.

## Development

```shell
npm install
npm run db:migrate:local
npm run dev
```

Before handing off changes, run:

```shell
npm run check
```

Use D1 migrations for schema changes. Never edit a production migration after
it has shipped; add a new numbered migration. Store application secrets only in
`.dev.vars` locally or Cloudflare Worker Secrets in deployed environments.
