PRAGMA foreign_keys = ON;

-- Descriptive metadata can evolve without forcing a schema migration for every
-- optional field. Frequently queried fields should still graduate to columns.
ALTER TABLE functions
ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}';

-- Fast function-level totals for registry cards, sorting, and discovery.
CREATE TABLE function_stats (
    function_id TEXT PRIMARY KEY REFERENCES functions(id) ON DELETE CASCADE,
    likes_count INTEGER NOT NULL DEFAULT 0 CHECK (likes_count >= 0),
    downloads_count INTEGER NOT NULL DEFAULT 0 CHECK (downloads_count >= 0),
    runs_count INTEGER NOT NULL DEFAULT 0 CHECK (runs_count >= 0),
    updated_at TEXT NOT NULL
);

-- Version-level counters retain attribution when a function advances.
CREATE TABLE function_version_stats (
    function_version_id TEXT PRIMARY KEY
        REFERENCES function_versions(id) ON DELETE CASCADE,
    downloads_count INTEGER NOT NULL DEFAULT 0 CHECK (downloads_count >= 0),
    runs_count INTEGER NOT NULL DEFAULT 0 CHECK (runs_count >= 0),
    updated_at TEXT NOT NULL
);

-- This is the source of truth for likes and enforces one like per user.
-- function_stats.likes_count is a denormalized read model updated transactionally.
CREATE TABLE function_likes (
    function_id TEXT NOT NULL REFERENCES functions(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    PRIMARY KEY (function_id, user_id)
);

CREATE INDEX function_likes_user_idx
ON function_likes(user_id, created_at DESC);
