import { type AuthEnv, HttpError } from "./auth";
import { summarizeArtifact, type ArtifactSummary } from "./publish";

export interface FunctionsEnv extends AuthEnv {
  DB: D1Database;
  ARTIFACTS: R2Bucket;
}

const IDENTIFIER = /^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$/;

interface FunctionRow {
  namespace_slug: string;
  slug: string;
  display_name: string;
  task_type: string;
  visibility: string;
  metadata_json: string;
  version_number: number;
  training_annotation_count: number;
  holdout_annotation_count: number;
  rationale_count: number;
  likes_count: number;
  downloads_count: number;
  runs_count: number;
  version_downloads_count: number;
  version_runs_count: number;
  updated_at: string;
}

interface FunctionVersionRow extends FunctionRow {
  function_id: string;
  function_version_id: string;
  digest: string;
  artifact_key: string;
  schema_version: number;
  summary_json: string;
  created_at: string;
}

interface VersionRow {
  version_number: number;
  digest: string;
  schema_version: number;
  training_annotation_count: number;
  holdout_annotation_count: number;
  rationale_count: number;
  downloads_count: number;
  runs_count: number;
  created_at: string;
}

function validateReference(namespace: string, slug: string): void {
  if (!IDENTIFIER.test(namespace) || !IDENTIFIER.test(slug)) {
    throw new HttpError(400, "reference_invalid", "Invalid function reference");
  }
}

function descriptionFromMetadata(value: string): string | null {
  try {
    const metadata = JSON.parse(value) as { description?: unknown };
    return typeof metadata.description === "string" && metadata.description
      ? metadata.description
      : null;
  } catch {
    return null;
  }
}

function publicFunctionQuery(version: number | null): string {
  return `SELECT functions.id AS function_id,
                 function_versions.id AS function_version_id,
                 namespaces.slug AS namespace_slug,
                 functions.slug,
                 functions.display_name,
                 functions.task_type,
                 functions.visibility,
                 functions.metadata_json,
                 function_versions.version_number,
                 function_versions.training_annotation_count,
                 function_versions.holdout_annotation_count,
                 function_versions.rationale_count,
                 function_versions.digest,
                 function_versions.artifact_key,
                 function_versions.schema_version,
                 function_versions.summary_json,
                 function_versions.created_at,
                 COALESCE(function_stats.likes_count, 0) AS likes_count,
                 COALESCE(function_stats.downloads_count, 0) AS downloads_count,
                 COALESCE(function_stats.runs_count, 0) AS runs_count,
                 COALESCE(function_version_stats.downloads_count, 0)
                   AS version_downloads_count,
                 COALESCE(function_version_stats.runs_count, 0)
                   AS version_runs_count,
                 functions.updated_at
            FROM functions
            JOIN namespaces ON namespaces.id = functions.namespace_id
            JOIN function_versions
              ON function_versions.function_id = functions.id
            LEFT JOIN function_stats ON function_stats.function_id = functions.id
            LEFT JOIN function_version_stats
              ON function_version_stats.function_version_id = function_versions.id
           WHERE namespaces.slug = ?
             AND functions.slug = ?
             AND functions.visibility = 'public'
             AND functions.unpublished_at IS NULL
             ${version === null ? "" : "AND function_versions.version_number = ?"}
           ORDER BY function_versions.version_number DESC
           LIMIT 1`;
}

async function findPublicFunction(
  env: FunctionsEnv,
  namespace: string,
  slug: string,
  version: number | null,
): Promise<FunctionVersionRow> {
  validateReference(namespace, slug);
  const statement = env.DB.prepare(publicFunctionQuery(version));
  const row = await (version === null
    ? statement.bind(namespace, slug)
    : statement.bind(namespace, slug, version)
  ).first<FunctionVersionRow>();
  if (!row) throw new HttpError(404, "function_not_found", "Function version not found");
  return row;
}

function functionPayload(row: FunctionVersionRow) {
  return {
    reference: `${row.namespace_slug}/${row.slug}`,
    name: row.display_name,
    description: descriptionFromMetadata(row.metadata_json),
    taskType: row.task_type,
    visibility: row.visibility,
    version: row.version_number,
    schemaVersion: row.schema_version,
    digest: row.digest,
    trainingAnnotations: row.training_annotation_count,
    holdoutAnnotations: row.holdout_annotation_count,
    rationales: row.rationale_count,
    likes: row.likes_count,
    downloads: row.downloads_count,
    runs: row.runs_count,
    versionDownloads: row.version_downloads_count,
    versionRuns: row.version_runs_count,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

export async function listFunctions(env: FunctionsEnv): Promise<Response> {
  const result = await env.DB.prepare(
    `SELECT namespaces.slug AS namespace_slug,
            functions.slug,
            functions.display_name,
            functions.task_type,
            functions.visibility,
            functions.metadata_json,
            function_versions.version_number,
            function_versions.training_annotation_count,
            function_versions.holdout_annotation_count,
            function_versions.rationale_count,
            COALESCE(function_stats.likes_count, 0) AS likes_count,
            COALESCE(function_stats.downloads_count, 0) AS downloads_count,
            COALESCE(function_stats.runs_count, 0) AS runs_count,
            COALESCE(function_version_stats.downloads_count, 0)
              AS version_downloads_count,
            COALESCE(function_version_stats.runs_count, 0)
              AS version_runs_count,
            functions.updated_at
       FROM functions
       JOIN namespaces ON namespaces.id = functions.namespace_id
       JOIN function_versions ON function_versions.id = functions.latest_version_id
       LEFT JOIN function_stats ON function_stats.function_id = functions.id
       LEFT JOIN function_version_stats
         ON function_version_stats.function_version_id = function_versions.id
      WHERE functions.visibility = 'public'
        AND functions.unpublished_at IS NULL
      ORDER BY functions.updated_at DESC
      LIMIT 100`,
  ).all<FunctionRow>();

  return Response.json(
    {
      functions: (result.results ?? []).map((row) => ({
        reference: `${row.namespace_slug}/${row.slug}`,
        name: row.display_name,
        description: descriptionFromMetadata(row.metadata_json),
        taskType: row.task_type,
        visibility: row.visibility,
        version: row.version_number,
        trainingAnnotations: row.training_annotation_count,
        holdoutAnnotations: row.holdout_annotation_count,
        rationales: row.rationale_count,
        likes: row.likes_count,
        downloads: row.downloads_count,
        runs: row.runs_count,
        versionDownloads: row.version_downloads_count,
        versionRuns: row.version_runs_count,
        updatedAt: row.updated_at,
      })),
    },
    {
      headers: {
        "Cache-Control": "public, max-age=0, must-revalidate",
      },
    },
  );
}

export async function getFunction(
  env: FunctionsEnv,
  namespace: string,
  slug: string,
): Promise<Response> {
  const row = await findPublicFunction(env, namespace, slug, null);
  let summary: ArtifactSummary | null = null;
  try {
    const stored = JSON.parse(row.summary_json) as Partial<ArtifactSummary>;
    if (stored.inputs && stored.definition && stored.backend) {
      summary = stored as ArtifactSummary;
    }
  } catch {
    // Older summaries contain only card metadata; fall back to their artifact.
  }
  if (!summary) {
    const artifact = await env.ARTIFACTS.get(row.artifact_key);
    if (!artifact) {
      throw new HttpError(503, "artifact_unavailable", "Function artifact is unavailable");
    }
    try {
      summary = summarizeArtifact(await artifact.json(), slug);
    } catch (error) {
      if (error instanceof HttpError) throw error;
      throw new HttpError(503, "artifact_invalid", "Function artifact is unavailable");
    }
  }
  const versions = await env.DB.prepare(
    `SELECT function_versions.version_number,
            function_versions.digest,
            function_versions.schema_version,
            function_versions.training_annotation_count,
            function_versions.holdout_annotation_count,
            function_versions.rationale_count,
            COALESCE(function_version_stats.downloads_count, 0) AS downloads_count,
            COALESCE(function_version_stats.runs_count, 0) AS runs_count,
            function_versions.created_at
       FROM function_versions
       LEFT JOIN function_version_stats
         ON function_version_stats.function_version_id = function_versions.id
      WHERE function_versions.function_id = ?
      ORDER BY function_versions.version_number DESC`,
  )
    .bind(row.function_id)
    .all<VersionRow>();
  return Response.json({
    ...functionPayload(row),
    inputs: summary.inputs,
    definition: summary.definition,
    backend: summary.backend,
    learning: summary.learning,
    metrics: summary.metrics,
    versions: (versions.results ?? []).map((item) => ({
      version: item.version_number,
      digest: item.digest,
      schemaVersion: item.schema_version,
      trainingAnnotations: item.training_annotation_count,
      holdoutAnnotations: item.holdout_annotation_count,
      rationales: item.rationale_count,
      downloads: item.downloads_count,
      runs: item.runs_count,
      createdAt: item.created_at,
    })),
  }, {
    headers: { "Cache-Control": "public, max-age=0, must-revalidate" },
  });
}

export async function downloadFunctionArtifact(
  env: FunctionsEnv,
  namespace: string,
  slug: string,
  version: number,
): Promise<Response> {
  if (!Number.isSafeInteger(version) || version < 1) {
    throw new HttpError(400, "version_invalid", "Function version must be positive");
  }
  const row = await findPublicFunction(env, namespace, slug, version);
  const artifact = await env.ARTIFACTS.get(row.artifact_key);
  if (!artifact) {
    throw new HttpError(503, "artifact_unavailable", "Function artifact is unavailable");
  }
  const now = new Date().toISOString();
  await env.DB.batch([
    env.DB.prepare(
      `INSERT INTO function_stats (
         function_id, likes_count, downloads_count, runs_count, updated_at
       ) VALUES (?, 0, 1, 0, ?)
       ON CONFLICT(function_id) DO UPDATE SET
         downloads_count = function_stats.downloads_count + 1,
         updated_at = excluded.updated_at`,
    ).bind(row.function_id, now),
    env.DB.prepare(
      `INSERT INTO function_version_stats (
         function_version_id, downloads_count, runs_count, updated_at
       ) VALUES (?, 1, 0, ?)
       ON CONFLICT(function_version_id) DO UPDATE SET
         downloads_count = function_version_stats.downloads_count + 1,
         updated_at = excluded.updated_at`,
    ).bind(row.function_version_id, now),
  ]);
  return new Response(artifact.body, {
    headers: {
      "Cache-Control": "public, max-age=0, must-revalidate",
      "Content-Disposition": `attachment; filename="${slug}-v${version}.json"`,
      "Content-Type": "application/json; charset=utf-8",
      ETag: `"${row.digest}"`,
      "X-Jeva-Digest": row.digest,
      "X-Jeva-Reference": `${row.namespace_slug}/${row.slug}`,
      "X-Jeva-Version": String(row.version_number),
    },
  });
}
