const GITHUB_API_VERSION = "2022-11-28";
const GITHUB_API_URL = "https://api.github.com/user";
const GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize";
const GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token";
const STATE_COOKIE = "jeva_oauth_state";
const SESSION_COOKIE = "jeva_session";
const SESSION_SECONDS = 60 * 60 * 24 * 30;
const API_TOKEN_SECONDS = 60 * 60 * 24 * 90;

export interface AuthEnv {
  DB: D1Database;
  APP_ENV: string;
  PUBLIC_ORIGIN?: string;
  GITHUB_APP_CLIENT_ID?: string;
  GITHUB_APP_CLIENT_SECRET?: string;
  TOKEN_PEPPER?: string;
}

interface GithubUser {
  id: string;
  login: string;
  name: string | null;
  avatarUrl: string | null;
}

export interface StoredUser {
  id: string;
  github_id: string;
  github_login: string;
  display_name: string | null;
  avatar_url: string | null;
}

export interface ApiIdentity extends StoredUser {
  scopes_json: string;
}

export type NamespaceRole = "owner" | "admin" | "member" | "viewer";

export interface NamespaceAccess {
  id: string;
  slug: string;
  kind: "personal" | "organization";
  role: NamespaceRole;
}

export class HttpError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

function required(value: string | undefined, name: string): string {
  if (!value) {
    throw new HttpError(503, "auth_not_configured", `${name} is not configured`);
  }
  return value;
}

function bytesToBase64Url(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}

function randomToken(bytes = 32): string {
  const value = new Uint8Array(bytes);
  crypto.getRandomValues(value);
  return bytesToBase64Url(value);
}

function hex(bytes: ArrayBuffer): string {
  return [...new Uint8Array(bytes)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

async function secretHash(value: string, pepper: string): Promise<string> {
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(pepper),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  return hex(await crypto.subtle.sign("HMAC", key, encoder.encode(value)));
}

function cookies(request: Request): Map<string, string> {
  const values = new Map<string, string>();
  for (const part of (request.headers.get("Cookie") ?? "").split(";")) {
    const separator = part.indexOf("=");
    if (separator < 0) continue;
    const name = part.slice(0, separator).trim();
    const value = part.slice(separator + 1).trim();
    if (name) values.set(name, decodeURIComponent(value));
  }
  return values;
}

function cookie(
  name: string,
  value: string,
  env: AuthEnv,
  options: { maxAge: number },
): string {
  const secure = env.APP_ENV !== "development" && env.APP_ENV !== "test";
  return [
    `${name}=${encodeURIComponent(value)}`,
    "Path=/",
    "HttpOnly",
    "SameSite=Lax",
    secure ? "Secure" : null,
    `Max-Age=${options.maxAge}`,
  ]
    .filter(Boolean)
    .join("; ");
}

function publicOrigin(request: Request, env: AuthEnv): string {
  return (env.PUBLIC_ORIGIN ?? new URL(request.url).origin).replace(/\/$/, "");
}

function callbackUrl(request: Request, env: AuthEnv): string {
  return `${publicOrigin(request, env)}/api/auth/github/callback`;
}

function jsonBody(value: unknown, init?: ResponseInit): Response {
  return Response.json(value, {
    ...init,
    headers: {
      "Cache-Control": "no-store",
      ...init?.headers,
    },
  });
}

async function githubUser(
  accessToken: string,
  fetcher: typeof fetch,
): Promise<GithubUser> {
  const response = await fetcher(GITHUB_API_URL, {
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${accessToken}`,
      "User-Agent": "ai-functions.dev",
      "X-GitHub-Api-Version": GITHUB_API_VERSION,
    },
  });
  if (!response.ok) {
    throw new HttpError(401, "github_token_invalid", "GitHub authentication failed");
  }
  const body = (await response.json()) as Record<string, unknown>;
  if (
    (typeof body.id !== "number" && typeof body.id !== "string") ||
    typeof body.login !== "string" ||
    !body.login
  ) {
    throw new HttpError(502, "github_response_invalid", "GitHub returned an invalid user");
  }
  return {
    id: String(body.id),
    login: body.login,
    name: typeof body.name === "string" ? body.name : null,
    avatarUrl: typeof body.avatar_url === "string" ? body.avatar_url : null,
  };
}

async function upsertIdentity(env: AuthEnv, github: GithubUser): Promise<StoredUser> {
  const now = new Date().toISOString();
  const userId = `usr_github_${github.id}`;
  const namespaceId = `ns_github_${github.id}`;

  await env.DB.batch([
    env.DB.prepare(
      `INSERT INTO users (
         id, github_id, github_login, display_name, avatar_url, created_at, updated_at
       ) VALUES (?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(github_id) DO UPDATE SET
         github_login = excluded.github_login,
         display_name = excluded.display_name,
         avatar_url = excluded.avatar_url,
         updated_at = excluded.updated_at`,
    ).bind(
      userId,
      github.id,
      github.login,
      github.name,
      github.avatarUrl,
      now,
      now,
    ),
    env.DB.prepare(
      `INSERT INTO namespaces (id, slug, kind, created_at, updated_at)
       VALUES (?, ?, 'personal', ?, ?)
       ON CONFLICT(id) DO UPDATE SET slug = excluded.slug, updated_at = excluded.updated_at`,
    ).bind(namespaceId, github.login, now, now),
    env.DB.prepare(
      `INSERT INTO namespace_memberships (namespace_id, user_id, role, created_at)
       VALUES (?, ?, 'owner', ?)
       ON CONFLICT(namespace_id, user_id) DO UPDATE SET role = 'owner'`,
    ).bind(namespaceId, userId, now),
  ]);

  return {
    id: userId,
    github_id: github.id,
    github_login: github.login,
    display_name: github.name,
    avatar_url: github.avatarUrl,
  };
}

async function issueSession(env: AuthEnv, userId: string): Promise<string> {
  const pepper = required(env.TOKEN_PEPPER, "TOKEN_PEPPER");
  const token = `jevas_${randomToken()}`;
  const hash = await secretHash(token, pepper);
  const now = new Date();
  const expires = new Date(now.getTime() + SESSION_SECONDS * 1000);
  await env.DB.prepare(
    `INSERT INTO web_sessions (id_hash, user_id, expires_at, created_at)
     VALUES (?, ?, ?, ?)`,
  )
    .bind(hash, userId, expires.toISOString(), now.toISOString())
    .run();
  return token;
}

async function issueApiToken(env: AuthEnv, userId: string): Promise<{
  token: string;
  expiresAt: string;
}> {
  const pepper = required(env.TOKEN_PEPPER, "TOKEN_PEPPER");
  const token = `jeva_${randomToken()}`;
  const hash = await secretHash(token, pepper);
  const now = new Date();
  const expires = new Date(now.getTime() + API_TOKEN_SECONDS * 1000);
  await env.DB.prepare(
    `INSERT INTO api_tokens (
       id, user_id, token_prefix, token_hash, name, scopes_json,
       expires_at, created_at
     ) VALUES (?, ?, ?, ?, 'Jeva CLI', ?, ?, ?)`,
  )
    .bind(
      crypto.randomUUID(),
      userId,
      token.slice(0, 13),
      hash,
      JSON.stringify(["functions:read", "functions:write"]),
      expires.toISOString(),
      now.toISOString(),
    )
    .run();
  return { token, expiresAt: expires.toISOString() };
}

export function authConfig(env: AuthEnv): Response {
  return jsonBody({
    github_client_id: required(env.GITHUB_APP_CLIENT_ID, "GITHUB_APP_CLIENT_ID"),
  });
}

export function startGithubLogin(request: Request, env: AuthEnv): Response {
  const clientId = required(env.GITHUB_APP_CLIENT_ID, "GITHUB_APP_CLIENT_ID");
  const state = randomToken();
  const authorize = new URL(GITHUB_AUTHORIZE_URL);
  authorize.searchParams.set("client_id", clientId);
  authorize.searchParams.set("redirect_uri", callbackUrl(request, env));
  authorize.searchParams.set("state", state);

  return new Response(null, {
    status: 302,
    headers: {
      Location: authorize.toString(),
      "Set-Cookie": cookie(STATE_COOKIE, state, env, { maxAge: 600 }),
      "Cache-Control": "no-store",
    },
  });
}

export async function finishGithubLogin(
  request: Request,
  env: AuthEnv,
  fetcher: typeof fetch = fetch,
): Promise<Response> {
  const url = new URL(request.url);
  const code = url.searchParams.get("code");
  const state = url.searchParams.get("state");
  const expectedState = cookies(request).get(STATE_COOKIE);
  if (!code || !state || !expectedState || state !== expectedState) {
    throw new HttpError(400, "oauth_state_invalid", "GitHub login could not be verified");
  }

  const response = await fetcher(GITHUB_TOKEN_URL, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/x-www-form-urlencoded",
      "User-Agent": "ai-functions.dev",
    },
    body: new URLSearchParams({
      client_id: required(env.GITHUB_APP_CLIENT_ID, "GITHUB_APP_CLIENT_ID"),
      client_secret: required(
        env.GITHUB_APP_CLIENT_SECRET,
        "GITHUB_APP_CLIENT_SECRET",
      ),
      code,
      redirect_uri: callbackUrl(request, env),
    }),
  });
  const tokenBody = (await response.json()) as Record<string, unknown>;
  if (!response.ok || typeof tokenBody.access_token !== "string") {
    throw new HttpError(401, "github_login_failed", "GitHub login failed");
  }

  const user = await upsertIdentity(
    env,
    await githubUser(tokenBody.access_token, fetcher),
  );
  const session = await issueSession(env, user.id);
  const redirect = new Response(null, {
    status: 302,
    headers: {
      Location: `${publicOrigin(request, env)}/`,
      "Cache-Control": "no-store",
    },
  });
  redirect.headers.append(
    "Set-Cookie",
    cookie(SESSION_COOKIE, session, env, { maxAge: SESSION_SECONDS }),
  );
  redirect.headers.append(
    "Set-Cookie",
    cookie(STATE_COOKIE, "", env, { maxAge: 0 }),
  );
  return redirect;
}

export async function exchangeGithubToken(
  request: Request,
  env: AuthEnv,
  fetcher: typeof fetch = fetch,
): Promise<Response> {
  let body: Record<string, unknown>;
  try {
    body = (await request.json()) as Record<string, unknown>;
  } catch {
    throw new HttpError(400, "invalid_json", "Expected a JSON request body");
  }
  if (typeof body.github_access_token !== "string" || !body.github_access_token) {
    throw new HttpError(
      400,
      "github_token_required",
      "A GitHub access token is required",
    );
  }

  const user = await upsertIdentity(
    env,
    await githubUser(body.github_access_token, fetcher),
  );
  const issued = await issueApiToken(env, user.id);
  return jsonBody({
    token: issued.token,
    expires_at: issued.expiresAt,
    user: {
      github_id: user.github_id,
      login: user.github_login,
      name: user.display_name,
      avatar_url: user.avatar_url,
    },
  });
}

export async function currentUser(request: Request, env: AuthEnv): Promise<Response> {
  const authorization = request.headers.get("Authorization");
  if (authorization?.startsWith("Bearer ")) {
    const user = await authenticatedApiUser(request, env);
    return jsonBody({
      user: user
        ? {
            github_id: user.github_id,
            login: user.github_login,
            name: user.display_name,
            avatar_url: user.avatar_url,
          }
        : null,
    });
  }

  const session = cookies(request).get(SESSION_COOKIE);
  if (!session) return jsonBody({ user: null });

  const pepper = required(env.TOKEN_PEPPER, "TOKEN_PEPPER");
  const hash = await secretHash(session, pepper);
  const user = await env.DB.prepare(
    `SELECT users.id, users.github_id, users.github_login,
            users.display_name, users.avatar_url
       FROM web_sessions
       JOIN users ON users.id = web_sessions.user_id
      WHERE web_sessions.id_hash = ? AND web_sessions.expires_at > ?`,
  )
    .bind(hash, new Date().toISOString())
    .first<StoredUser>();

  return jsonBody({
    user: user
      ? {
          github_id: user.github_id,
          login: user.github_login,
          name: user.display_name,
          avatar_url: user.avatar_url,
        }
      : null,
  });
}

export async function authenticatedApiUser(
  request: Request,
  env: AuthEnv,
): Promise<ApiIdentity | null> {
  const authorization = request.headers.get("Authorization");
  if (!authorization?.startsWith("Bearer ")) return null;
  const token = authorization.slice("Bearer ".length).trim();
  if (!token) return null;
  const pepper = required(env.TOKEN_PEPPER, "TOKEN_PEPPER");
  const hash = await secretHash(token, pepper);
  const now = new Date().toISOString();
  const user = await env.DB.prepare(
    `SELECT users.id, users.github_id, users.github_login,
            users.display_name, users.avatar_url,
            api_tokens.scopes_json
       FROM api_tokens
       JOIN users ON users.id = api_tokens.user_id
      WHERE api_tokens.token_hash = ?
        AND api_tokens.revoked_at IS NULL
        AND (api_tokens.expires_at IS NULL OR api_tokens.expires_at > ?)
      LIMIT 1`,
  )
    .bind(hash, now)
    .first<ApiIdentity>();
  if (user) {
    await env.DB.prepare("UPDATE api_tokens SET last_used_at = ? WHERE token_hash = ?")
      .bind(now, hash)
      .run();
  }
  return user;
}

export async function authorizeNamespace(
  env: AuthEnv,
  user: StoredUser,
  namespaceSlug: string,
  allowedRoles: readonly NamespaceRole[],
): Promise<NamespaceAccess> {
  const access = await env.DB.prepare(
    `SELECT namespaces.id, namespaces.slug, namespaces.kind,
            namespace_memberships.role
       FROM namespace_memberships
       JOIN namespaces ON namespaces.id = namespace_memberships.namespace_id
      WHERE namespace_memberships.user_id = ?
        AND namespaces.slug = ?
      LIMIT 1`,
  )
    .bind(user.id, namespaceSlug)
    .first<NamespaceAccess>();
  if (!access || !allowedRoles.includes(access.role)) {
    throw new HttpError(403, "namespace_forbidden", "You cannot access this namespace");
  }
  return access;
}

export async function revokeCurrentToken(
  request: Request,
  env: AuthEnv,
): Promise<Response> {
  const authorization = request.headers.get("Authorization");
  if (!authorization?.startsWith("Bearer ")) {
    throw new HttpError(401, "authentication_required", "Authentication is required");
  }
  const token = authorization.slice("Bearer ".length).trim();
  const pepper = required(env.TOKEN_PEPPER, "TOKEN_PEPPER");
  await env.DB.prepare(
    `UPDATE api_tokens SET revoked_at = ?
      WHERE token_hash = ? AND revoked_at IS NULL`,
  )
    .bind(new Date().toISOString(), await secretHash(token, pepper))
    .run();
  return jsonBody({ ok: true });
}

export async function logout(request: Request, env: AuthEnv): Promise<Response> {
  const session = cookies(request).get(SESSION_COOKIE);
  if (session) {
    const pepper = required(env.TOKEN_PEPPER, "TOKEN_PEPPER");
    await env.DB.prepare("DELETE FROM web_sessions WHERE id_hash = ?")
      .bind(await secretHash(session, pepper))
      .run();
  }
  return jsonBody(
    { ok: true },
    {
      headers: {
        "Set-Cookie": cookie(SESSION_COOKIE, "", env, { maxAge: 0 }),
      },
    },
  );
}

export function errorResponse(error: unknown): Response {
  if (error instanceof HttpError) {
    return jsonBody(
      { error: { code: error.code, message: error.message } },
      { status: error.status },
    );
  }
  console.error("Unhandled API error", error);
  return jsonBody(
    { error: { code: "internal_error", message: "An unexpected error occurred" } },
    { status: 500 },
  );
}
