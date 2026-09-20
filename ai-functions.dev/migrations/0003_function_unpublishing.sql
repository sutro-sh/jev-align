ALTER TABLE functions ADD COLUMN unpublished_at TEXT;

CREATE INDEX functions_public_updated_idx
    ON functions(visibility, unpublished_at, updated_at DESC);
