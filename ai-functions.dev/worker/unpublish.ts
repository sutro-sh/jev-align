import {
  type ApiIdentity,
  type AuthEnv,
  authenticatedApiUser,
  authorizeNamespace,
  HttpError,
} from "./auth";

const IDENTIFIER = /^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$/;

export interface UnpublishEnv extends AuthEnv {
  DB: D1Database;
}

interface FunctionRecord {
  id: string;
  unpublished_at: string | null;
}

function scopes(user: ApiIdentity): string[] {
  try {
    const value = JSON.parse(user.scopes_json) as unknown;
    return Array.isArray(value)
      ? value.filter((item): item is string => typeof item === "string")
      : [];
  } catch {
    return [];
  }
}

export async function unpublishFunction(
  request: Request,
  env: UnpublishEnv,
  namespaceSlug: string,
  slug: string,
): Promise<Response> {
  if (!IDENTIFIER.test(namespaceSlug) || !IDENTIFIER.test(slug)) {
    throw new HttpError(400, "reference_invalid", "Invalid function reference");
  }
  const user = await authenticatedApiUser(request, env);
  if (!user) {
    throw new HttpError(401, "authentication_required", "Run jeva login before unpublishing");
  }
  if (!scopes(user).includes("functions:write")) {
    throw new HttpError(403, "scope_required", "This token cannot unpublish functions");
  }
  const namespace = await authorizeNamespace(env, user, namespaceSlug, ["owner", "admin"]);
  const existing = await env.DB.prepare(
    `SELECT id, unpublished_at
       FROM functions
      WHERE namespace_id = ? AND slug = ?
      LIMIT 1`,
  )
    .bind(namespace.id, slug)
    .first<FunctionRecord>();
  if (!existing) {
    throw new HttpError(404, "function_not_found", "Function not found");
  }

  const changed = existing.unpublished_at === null;
  if (changed) {
    const now = new Date().toISOString();
    await env.DB.prepare(
      "UPDATE functions SET unpublished_at = ?, updated_at = ? WHERE id = ?",
    )
      .bind(now, now, existing.id)
      .run();
  }
  return Response.json(
    {
      reference: `${namespace.slug}/${slug}`,
      unpublished: true,
      changed,
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}
