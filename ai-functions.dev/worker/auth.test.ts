import { describe, expect, it, vi } from "vitest";

import { authenticatedApiUser, authorizeNamespace, exchangeGithubToken } from "./auth";
import type { Env } from "./index";

class FakeStatement {
  values: unknown[] = [];

  constructor(readonly sql: string) {}

  bind(...values: unknown[]): FakeStatement {
    this.values = values;
    return this;
  }

  async run(): Promise<D1Result> {
    return { success: true } as D1Result;
  }
}

class FakeDatabase {
  statements: FakeStatement[] = [];

  prepare(sql: string): FakeStatement {
    const statement = new FakeStatement(sql);
    this.statements.push(statement);
    return statement;
  }

  async batch(statements: FakeStatement[]): Promise<D1Result[]> {
    return statements.map(() => ({ success: true }) as D1Result);
  }
}

describe("CLI GitHub token exchange", () => {
  it("verifies GitHub identity and returns a Jeva-scoped token", async () => {
    const database = new FakeDatabase();
    const githubFetch = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      expect(init?.headers).toMatchObject({
        Authorization: "Bearer github-temporary-token",
      });
      return Response.json({
        id: 42,
        login: "octocat",
        name: "The Octocat",
        avatar_url: "https://avatars.githubusercontent.com/u/42",
      });
    }) as typeof fetch;
    const env = {
      APP_ENV: "test",
      TOKEN_PEPPER: "test-only-pepper",
      DB: database,
    } as unknown as Env;

    const response = await exchangeGithubToken(
      new Request("https://ai-functions.dev/api/v1/auth/github/exchange", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ github_access_token: "github-temporary-token" }),
      }),
      env,
      githubFetch,
    );
    const body = (await response.json()) as {
      token: string;
      user: { login: string };
      expires_at: string;
    };

    expect(response.status).toBe(200);
    expect(body.token).toMatch(/^jeva_[A-Za-z0-9_-]+$/);
    expect(body.token).not.toContain("github-temporary-token");
    expect(body.user.login).toBe("octocat");
    expect(new Date(body.expires_at).getTime()).toBeGreaterThan(Date.now());
    expect(database.statements.some((statement) => statement.sql.includes("api_tokens"))).toBe(
      true,
    );
  });

  it("rejects a missing GitHub token", async () => {
    const env = {
      APP_ENV: "test",
      TOKEN_PEPPER: "test-only-pepper",
      DB: new FakeDatabase(),
    } as unknown as Env;

    await expect(
      exchangeGithubToken(
        new Request("https://ai-functions.dev/api/v1/auth/github/exchange", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
        }),
        env,
      ),
    ).rejects.toMatchObject({ status: 400, code: "github_token_required" });
  });
});

describe("namespace authorization", () => {
  it("authenticates the token without selecting a namespace", async () => {
    const statements: string[] = [];
    const env = {
      APP_ENV: "test",
      TOKEN_PEPPER: "test-only-pepper",
      DB: {
        prepare(sql: string) {
          statements.push(sql);
          return {
            bind() {
              return this;
            },
            async first() {
              return {
                id: "usr_octocat",
                github_id: "42",
                github_login: "octocat",
                display_name: null,
                avatar_url: null,
                scopes_json: '["functions:read","functions:write"]',
              };
            },
            async run() {
              return { success: true } as D1Result;
            },
          };
        },
      },
    } as unknown as Env;

    const identity = await authenticatedApiUser(
      new Request("https://ai-functions.dev/api/v1/me", {
        headers: { Authorization: "Bearer jeva_test" },
      }),
      env,
    );

    expect(identity?.github_login).toBe("octocat");
    expect(identity).not.toHaveProperty("namespace_id");
    expect(statements[0]).not.toContain("namespaces");
    expect(statements[0]).not.toContain("namespace_memberships");
  });

  it("authorizes membership for the authenticated user", async () => {
    const statement = {
      values: [] as unknown[],
      bind(...values: unknown[]) {
        this.values = values;
        return this;
      },
      async first() {
        return { id: "ns_octocat", slug: "octocat", kind: "personal", role: "owner" };
      },
    };
    const env = {
      DB: { prepare: () => statement },
    } as unknown as Env;

    const access = await authorizeNamespace(
      env,
      {
        id: "usr_octocat",
        github_id: "42",
        github_login: "octocat",
        display_name: null,
        avatar_url: null,
      },
      "octocat",
      ["owner"],
    );

    expect(access.slug).toBe("octocat");
    expect(statement.values).toEqual(["usr_octocat", "octocat"]);
  });

  it("does not authorize a different user's namespace", async () => {
    const statement = {
      values: [] as unknown[],
      bind(...values: unknown[]) {
        this.values = values;
        return this;
      },
      async first() {
        return null;
      },
    };
    const env = {
      DB: { prepare: () => statement },
    } as unknown as Env;

    await expect(
      authorizeNamespace(
        env,
        {
          id: "usr_octocat",
          github_id: "42",
          github_login: "octocat",
          display_name: null,
          avatar_url: null,
        },
        "other-user",
        ["owner"],
      ),
    ).rejects.toMatchObject({ status: 403, code: "namespace_forbidden" });
    expect(statement.values).toEqual(["usr_octocat", "other-user"]);
  });

  it("enforces the required namespace role", async () => {
    const statement = {
      bind() {
        return this;
      },
      async first() {
        return { id: "ns_team", slug: "team", kind: "organization", role: "viewer" };
      },
    };
    const env = {
      DB: { prepare: () => statement },
    } as unknown as Env;

    await expect(
      authorizeNamespace(
        env,
        {
          id: "usr_octocat",
          github_id: "42",
          github_login: "octocat",
          display_name: null,
          avatar_url: null,
        },
        "team",
        ["owner", "admin"],
      ),
    ).rejects.toMatchObject({ status: 403, code: "namespace_forbidden" });
  });
});
