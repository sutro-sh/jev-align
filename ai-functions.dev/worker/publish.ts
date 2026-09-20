import {
  type ApiIdentity,
  type AuthEnv,
  authenticatedApiUser,
  authorizeNamespace,
  HttpError,
} from "./auth";

const MAX_ARTIFACT_BYTES = 50 * 1024 * 1024;
const SLUG = /^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$/;
const TASK_TYPES = new Set(["binary", "multiclass", "multilabel", "score"]);

export interface PublishEnv extends AuthEnv {
  ARTIFACTS: R2Bucket;
}

export interface ArtifactSummary {
  name: string;
  slug: string;
  description: string | null;
  taskType: string;
  trainingCount: number;
  holdoutCount: number;
  rationaleCount: number;
  inputs: Record<string, unknown>;
  definition: Record<string, unknown>;
  backend: { provider: string; model: string };
  learning: Record<string, unknown>;
  metrics: Record<string, unknown>;
}

interface ExistingFunction {
  id: string;
  task_type: string;
  latest_version: number | null;
  visibility: string;
  unpublished_at: string | null;
}

interface ExistingVersion {
  version_number: number;
  digest: string;
}

function object(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function summarizeArtifact(value: unknown, pathSlug: string): ArtifactSummary {
  if (!object(value) || value.schema_version !== 1) {
    throw new HttpError(400, "artifact_invalid", "Expected a schema version 1 function artifact");
  }
  if (typeof value.name !== "string" || !value.name.trim() || value.name.length > 80) {
    throw new HttpError(400, "artifact_invalid", "Function name must be 1–80 characters");
  }
  if (
    value.description !== undefined &&
    value.description !== null &&
    (typeof value.description !== "string" || value.description.trim().length > 280)
  ) {
    throw new HttpError(400, "artifact_invalid", "Function description must be 280 characters or fewer");
  }
  if (typeof value.slug !== "string" || !SLUG.test(value.slug) || value.slug !== pathSlug) {
    throw new HttpError(400, "artifact_invalid", "Artifact slug must match the requested function slug");
  }
  if (typeof value.task_type !== "string" || !TASK_TYPES.has(value.task_type)) {
    throw new HttpError(400, "artifact_invalid", "Artifact has an unsupported task type");
  }
  if (!object(value.inputs) || !Array.isArray(value.inputs.columns) || value.inputs.columns.length === 0) {
    throw new HttpError(400, "artifact_invalid", "Artifact must declare its input columns");
  }
  if (!object(value.definition) || !object(value.backend)) {
    throw new HttpError(400, "artifact_invalid", "Artifact is missing its definition or backend");
  }
  if (typeof value.backend.provider !== "string" || typeof value.backend.model !== "string") {
    throw new HttpError(400, "artifact_invalid", "Artifact backend is invalid");
  }
  if (!Array.isArray(value.annotations) || value.annotations.length === 0) {
    throw new HttpError(400, "artifact_invalid", "Artifact must contain human annotations");
  }
  let trainingCount = 0;
  let holdoutCount = 0;
  let rationaleCount = 0;
  for (const annotation of value.annotations) {
    if (!object(annotation) || !object(annotation.inputs) || Object.keys(annotation.inputs).length === 0) {
      throw new HttpError(400, "artifact_invalid", "Every annotation must contain its labeled input");
    }
    if (annotation.split === "train") trainingCount += 1;
    else if (annotation.split === "holdout") holdoutCount += 1;
    else throw new HttpError(400, "artifact_invalid", "Annotation split must be train or holdout");
    if (typeof annotation.rationale === "string" && annotation.rationale.trim()) rationaleCount += 1;
  }
  return {
    name: value.name.trim(),
    slug: value.slug,
    description:
      typeof value.description === "string" && value.description.trim()
        ? value.description.trim()
        : null,
    taskType: value.task_type,
    trainingCount,
    holdoutCount,
    rationaleCount,
    inputs: value.inputs,
    definition: value.definition,
    backend: {
      provider: value.backend.provider,
      model: value.backend.model,
    },
    learning: object(value.learning) ? value.learning : {},
    metrics: object(value.metrics) ? value.metrics : {},
  };
}

async function sha256(value: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", value);
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function scopes(user: ApiIdentity): string[] {
  try {
    const value = JSON.parse(user.scopes_json) as unknown;
    return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
  } catch {
    return [];
  }
}

export async function publishFunction(
  request: Request,
  env: PublishEnv,
  slug: string,
): Promise<Response> {
  const user = await authenticatedApiUser(request, env);
  if (!user) throw new HttpError(401, "authentication_required", "Run jeva login before publishing");
  if (!scopes(user).includes("functions:write")) {
    throw new HttpError(403, "scope_required", "This token cannot publish functions");
  }
  const namespace = await authorizeNamespace(env, user, user.github_login, ["owner"]);
  if (!SLUG.test(slug)) throw new HttpError(400, "slug_invalid", "Invalid function slug");
  const declaredLength = Number(request.headers.get("Content-Length") ?? "0");
  if (declaredLength > MAX_ARTIFACT_BYTES) {
    throw new HttpError(413, "artifact_too_large", "Function artifact exceeds 50 MiB");
  }
  const bytes = new Uint8Array(await request.arrayBuffer());
  if (bytes.byteLength > MAX_ARTIFACT_BYTES) {
    throw new HttpError(413, "artifact_too_large", "Function artifact exceeds 50 MiB");
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    throw new HttpError(400, "artifact_invalid", "Function artifact is not valid JSON");
  }
  const summary = summarizeArtifact(parsed, slug);
  const digest = await sha256(bytes);
  const existing = await env.DB.prepare(
    `SELECT functions.id, functions.task_type, functions.visibility,
            functions.unpublished_at,
            MAX(function_versions.version_number) AS latest_version
       FROM functions
       LEFT JOIN function_versions ON function_versions.function_id = functions.id
      WHERE functions.namespace_id = ? AND functions.slug = ?
      GROUP BY functions.id`,
  )
    .bind(namespace.id, slug)
    .first<ExistingFunction>();
  if (existing && existing.task_type !== summary.taskType) {
    throw new HttpError(409, "task_type_changed", "A published function cannot change task type");
  }
  if (existing) {
    const prior = await env.DB.prepare(
      `SELECT version_number, digest FROM function_versions
        WHERE function_id = ? AND digest = ?`,
    )
      .bind(existing.id, digest)
      .first<ExistingVersion>();
    if (prior) {
      if (existing.unpublished_at !== null || existing.visibility !== "public") {
        const now = new Date().toISOString();
        await env.DB.prepare(
          "UPDATE functions SET visibility = 'public', unpublished_at = NULL, updated_at = ? WHERE id = ?",
        )
          .bind(now, existing.id)
          .run();
      }
      return Response.json({
        reference: `${namespace.slug}/${slug}`,
        version: prior.version_number,
        digest,
        url: `https://ai-functions.dev/${namespace.slug}/${slug}`,
        created: false,
      });
    }
  }

  const now = new Date().toISOString();
  const functionId = existing?.id ?? crypto.randomUUID();
  const version = (existing?.latest_version ?? 0) + 1;
  const versionId = crypto.randomUUID();
  const artifactKey = `functions/${namespace.slug}/${slug}/v${version}-${digest}.json`;
  await env.ARTIFACTS.put(artifactKey, bytes, {
    httpMetadata: { contentType: "application/json; charset=utf-8" },
    customMetadata: { digest, namespace: namespace.slug, slug, version: String(version) },
  });
  const statements: D1PreparedStatement[] = [];
  if (!existing) {
    statements.push(
      env.DB.prepare(
        `INSERT INTO functions (
           id, namespace_id, slug, display_name, task_type, visibility, metadata_json,
           created_by_user_id, created_at, updated_at
         ) VALUES (?, ?, ?, ?, ?, 'public', ?, ?, ?, ?)`,
      ).bind(
        functionId,
        namespace.id,
        slug,
        summary.name,
        summary.taskType,
        JSON.stringify({ description: summary.description }),
        user.id,
        now,
        now,
      ),
    );
  } else {
    statements.push(
      env.DB.prepare(
        "UPDATE functions SET display_name = ?, metadata_json = ?, visibility = 'public', unpublished_at = NULL, updated_at = ? WHERE id = ?",
      ).bind(
        summary.name,
        JSON.stringify({ description: summary.description }),
        now,
        functionId,
      ),
    );
  }
  statements.push(
    env.DB.prepare(
      `INSERT INTO function_versions (
         id, function_id, version_number, digest, artifact_key, schema_version,
         training_annotation_count, holdout_annotation_count, rationale_count,
         summary_json, created_by_user_id, created_at
       ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)`,
    ).bind(
      versionId,
      functionId,
      version,
      digest,
      artifactKey,
      summary.trainingCount,
      summary.holdoutCount,
      summary.rationaleCount,
      JSON.stringify(summary),
      user.id,
      now,
    ),
    env.DB.prepare("UPDATE functions SET latest_version_id = ?, updated_at = ? WHERE id = ?")
      .bind(versionId, now, functionId),
  );
  await env.DB.batch(statements);
  return Response.json(
    {
      reference: `${namespace.slug}/${slug}`,
      version,
      digest,
      url: `https://ai-functions.dev/${namespace.slug}/${slug}`,
      created: true,
    },
    { status: existing ? 200 : 201 },
  );
}
