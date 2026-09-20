import { beforeEach, describe, expect, it, vi } from "vitest";

import * as auth from "./auth";
import { unpublishFunction } from "./unpublish";

vi.mock("./auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./auth")>();
  return {
    ...actual,
    authenticatedApiUser: vi.fn(),
    authorizeNamespace: vi.fn(),
  };
});

const identity = {
  id: "user-1",
  github_id: "1",
  github_login: "octocat",
  display_name: "Octocat",
  avatar_url: null,
  scopes_json: JSON.stringify(["functions:read", "functions:write"]),
};

describe("function unpublishing", () => {
  beforeEach(() => {
    vi.mocked(auth.authenticatedApiUser).mockReset();
    vi.mocked(auth.authorizeNamespace).mockReset();
  });

  it("lets a namespace owner hide a function without deleting it", async () => {
    vi.mocked(auth.authenticatedApiUser).mockResolvedValue(identity);
    vi.mocked(auth.authorizeNamespace).mockResolvedValue({
      id: "namespace-1",
      slug: "octocat",
      kind: "personal",
      role: "owner",
    });
    const prepared: Array<{ sql: string; values: unknown[] }> = [];
    const database = {
      prepare(sql: string) {
        const record = { sql, values: [] as unknown[] };
        prepared.push(record);
        return {
          bind(...values: unknown[]) {
            record.values = values;
            return this;
          },
          async first() {
            return { id: "function-1", unpublished_at: null };
          },
          async run() {
            return {};
          },
        };
      },
    };
    const response = await unpublishFunction(
      new Request("https://ai-functions.dev/api/v1/functions/octocat/is-aviation", {
        method: "DELETE",
        headers: { Authorization: "Bearer token" },
      }),
      { DB: database } as unknown as Parameters<typeof unpublishFunction>[1],
      "octocat",
      "is-aviation",
    );

    expect(response.status).toBe(200);
    expect(response.headers.get("Cache-Control")).toBe("no-store");
    await expect(response.json()).resolves.toEqual({
      reference: "octocat/is-aviation",
      unpublished: true,
      changed: true,
    });
    expect(prepared.at(-1)?.sql).toContain("SET unpublished_at = ?");
    expect(prepared.at(-1)?.values.at(-1)).toBe("function-1");
  });

  it("requires authentication", async () => {
    vi.mocked(auth.authenticatedApiUser).mockResolvedValue(null);
    await expect(
      unpublishFunction(
        new Request("https://ai-functions.dev/api/v1/functions/octocat/is-aviation", {
          method: "DELETE",
        }),
        {} as Parameters<typeof unpublishFunction>[1],
        "octocat",
        "is-aviation",
      ),
    ).rejects.toMatchObject({ status: 401, code: "authentication_required" });
  });
});
