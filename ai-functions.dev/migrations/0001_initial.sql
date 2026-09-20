PRAGMA foreign_keys = ON;

CREATE TABLE users (
    id TEXT PRIMARY KEY,
    github_id TEXT NOT NULL UNIQUE,
    github_login TEXT NOT NULL COLLATE NOCASE,
    display_name TEXT,
    avatar_url TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX users_github_login_idx ON users(github_login);

CREATE TABLE namespaces (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL COLLATE NOCASE UNIQUE,
    kind TEXT NOT NULL CHECK (kind IN ('personal', 'organization')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE namespace_memberships (
    namespace_id TEXT NOT NULL REFERENCES namespaces(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('owner', 'admin', 'member', 'viewer')),
    created_at TEXT NOT NULL,
    PRIMARY KEY (namespace_id, user_id)
);

CREATE TABLE web_sessions (
    id_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX web_sessions_user_idx ON web_sessions(user_id);
CREATE INDEX web_sessions_expiry_idx ON web_sessions(expires_at);

CREATE TABLE api_tokens (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_prefix TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    scopes_json TEXT NOT NULL,
    last_used_at TEXT,
    expires_at TEXT,
    revoked_at TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX api_tokens_user_idx ON api_tokens(user_id);
CREATE INDEX api_tokens_prefix_idx ON api_tokens(token_prefix);

CREATE TABLE functions (
    id TEXT PRIMARY KEY,
    namespace_id TEXT NOT NULL REFERENCES namespaces(id) ON DELETE CASCADE,
    slug TEXT NOT NULL COLLATE NOCASE,
    display_name TEXT NOT NULL,
    task_type TEXT NOT NULL CHECK (
        task_type IN ('binary', 'multiclass', 'multilabel', 'score')
    ),
    visibility TEXT NOT NULL DEFAULT 'public' CHECK (
        visibility IN ('public', 'unlisted', 'private')
    ),
    latest_version_id TEXT,
    created_by_user_id TEXT NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (namespace_id, slug)
);

CREATE INDEX functions_namespace_idx ON functions(namespace_id, updated_at DESC);

CREATE TABLE function_versions (
    id TEXT PRIMARY KEY,
    function_id TEXT NOT NULL REFERENCES functions(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL CHECK (version_number > 0),
    digest TEXT NOT NULL,
    artifact_key TEXT NOT NULL UNIQUE,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0),
    training_annotation_count INTEGER NOT NULL DEFAULT 0,
    holdout_annotation_count INTEGER NOT NULL DEFAULT 0,
    rationale_count INTEGER NOT NULL DEFAULT 0,
    summary_json TEXT NOT NULL,
    created_by_user_id TEXT NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL,
    UNIQUE (function_id, version_number),
    UNIQUE (function_id, digest)
);

CREATE INDEX function_versions_function_idx
    ON function_versions(function_id, version_number DESC);
