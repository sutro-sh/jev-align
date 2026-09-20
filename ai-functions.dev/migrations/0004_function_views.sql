PRAGMA foreign_keys = ON;

-- A view is a successful read of a function detail page/API response. Downloads
-- remain reserved for immutable artifact pulls.
ALTER TABLE function_stats
ADD COLUMN views_count INTEGER NOT NULL DEFAULT 0 CHECK (views_count >= 0);

ALTER TABLE function_version_stats
ADD COLUMN views_count INTEGER NOT NULL DEFAULT 0 CHECK (views_count >= 0);
