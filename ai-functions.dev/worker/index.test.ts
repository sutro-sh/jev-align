import { describe, expect, it } from "vitest";

import { type Env, handleRequest } from "./index";

const env = {
  APP_ENV: "test",
} as Env;

describe("API worker", () => {
  it("reports its health", async () => {
    const response = await handleRequest(
      new Request("https://ai-functions.dev/api/health"),
      env,
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      ok: true,
      service: "ai-functions.dev",
      environment: "test",
    });
  });

  it("returns a structured API 404", async () => {
    const response = await handleRequest(
      new Request("https://ai-functions.dev/api/missing"),
      env,
    );

    expect(response.status).toBe(404);
    await expect(response.json()).resolves.toEqual({
      error: {
        code: "not_found",
        message: "API endpoint not found",
      },
    });
  });

  it("exposes the GitHub App client ID without exposing secrets", async () => {
    const response = await handleRequest(
      new Request("https://ai-functions.dev/api/v1/config"),
      { ...env, GITHUB_APP_CLIENT_ID: "Iv1.public-client-id" },
    );

    await expect(response.json()).resolves.toEqual({
      github_client_id: "Iv1.public-client-id",
    });
  });

  it("starts GitHub login with a short-lived state cookie", async () => {
    const response = await handleRequest(
      new Request("https://ai-functions.dev/api/auth/github/start"),
      { ...env, GITHUB_APP_CLIENT_ID: "Iv1.public-client-id" },
    );

    expect(response.status).toBe(302);
    const location = new URL(response.headers.get("Location") ?? "");
    expect(location.origin).toBe("https://github.com");
    expect(location.pathname).toBe("/login/oauth/authorize");
    expect(location.searchParams.get("client_id")).toBe("Iv1.public-client-id");
    expect(location.searchParams.get("state")).toBeTruthy();
    expect(response.headers.get("Set-Cookie")).toContain("jeva_oauth_state=");
    expect(response.headers.get("Set-Cookie")).toContain("HttpOnly");
  });
});
