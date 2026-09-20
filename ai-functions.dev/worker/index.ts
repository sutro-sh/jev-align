import {
  type AuthEnv,
  authConfig,
  currentUser,
  errorResponse,
  exchangeGithubToken,
  finishGithubLogin,
  logout,
  revokeCurrentToken,
  startGithubLogin,
} from "./auth";
import { publishFunction } from "./publish";
import { unpublishFunction } from "./unpublish";
import {
  downloadFunctionArtifact,
  getFunction,
  listFunctions,
} from "./functions";

export interface Env extends AuthEnv {
  DB: D1Database;
  ARTIFACTS: R2Bucket;
  APP_ENV: string;
}

export async function handleRequest(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);

  try {
    if (request.method === "GET" && url.pathname === "/api/health") {
      return Response.json({
        ok: true,
        service: "ai-functions.dev",
        environment: env.APP_ENV,
      });
    }
    if (request.method === "GET" && url.pathname === "/api/v1/config") {
      return authConfig(env);
    }
    if (request.method === "GET" && url.pathname === "/api/v1/functions") {
      return await listFunctions(env);
    }
    if (request.method === "GET" && url.pathname === "/api/auth/github/start") {
      return startGithubLogin(request, env);
    }
    if (request.method === "GET" && url.pathname === "/api/auth/github/callback") {
      return await finishGithubLogin(request, env);
    }
    if (request.method === "POST" && url.pathname === "/api/v1/auth/github/exchange") {
      return await exchangeGithubToken(request, env);
    }
    if (request.method === "GET" && url.pathname === "/api/v1/me") {
      return await currentUser(request, env);
    }
    if (request.method === "DELETE" && url.pathname === "/api/v1/tokens/current") {
      return await revokeCurrentToken(request, env);
    }
    if (request.method === "POST" && url.pathname === "/api/auth/logout") {
      return await logout(request, env);
    }
    const artifactMatch = url.pathname.match(
      /^\/api\/v1\/functions\/([^/]+)\/([^/]+)\/versions\/(\d+)\/artifact$/,
    );
    if (request.method === "GET" && artifactMatch) {
      return await downloadFunctionArtifact(
        env,
        decodeURIComponent(artifactMatch[1]),
        decodeURIComponent(artifactMatch[2]),
        Number(artifactMatch[3]),
      );
    }
    const detailMatch = url.pathname.match(/^\/api\/v1\/functions\/([^/]+)\/([^/]+)$/);
    if (request.method === "DELETE" && detailMatch) {
      return await unpublishFunction(
        request,
        env,
        decodeURIComponent(detailMatch[1]),
        decodeURIComponent(detailMatch[2]),
      );
    }
    if (request.method === "GET" && detailMatch) {
      return await getFunction(
        env,
        decodeURIComponent(detailMatch[1]),
        decodeURIComponent(detailMatch[2]),
      );
    }
    const publishMatch = url.pathname.match(/^\/api\/v1\/functions\/([^/]+)$/);
    if (request.method === "PUT" && publishMatch) {
      return await publishFunction(request, env, decodeURIComponent(publishMatch[1]));
    }

    if (url.pathname.startsWith("/api/")) {
      return Response.json(
        {
          error: {
            code: "not_found",
            message: "API endpoint not found",
          },
        },
        { status: 404 },
      );
    }
  } catch (error) {
    return errorResponse(error);
  }

  return new Response(null, { status: 404 });
}

export default {
  fetch(request, env) {
    return handleRequest(request, env);
  },
} satisfies ExportedHandler<Env>;
