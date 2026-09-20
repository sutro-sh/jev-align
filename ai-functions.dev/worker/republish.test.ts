import { describe, expect, it, vi } from "vitest";

import * as auth from "./auth";
import { publishFunction } from "./publish";

vi.mock("./auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./auth")>();
  return {
    ...actual,
    authenticatedApiUser: vi.fn(),
    authorizeNamespace: vi.fn(),
  };
});

describe("republishing", () => {
  it("restores an unpublished function even when its artifact is unchanged", async () => {
    vi.mocked(auth.authenticatedApiUser).mockResolvedValue({
      id: "user-1",
      github_id: "1",
      github_login: "octocat",
      display_name: "Octocat",
      avatar_url: null,
      scopes_json: JSON.stringify(["functions:write"]),
    });
    vi.mocked(auth.authorizeNamespace).mockResolvedValue({
      id: "namespace-1",
      slug: "octocat",
      kind: "personal",
      role: "owner",
    });
    const updates: string[] = [];
    let lookup = 0;
    const database = {
      prepare(sql: string) {
        return {
          bind() {
            return this;
          },
          async first() {
            lookup += 1;
            if (lookup === 1) {
              return {
                id: "function-1",
                task_type: "binary",
                latest_version: 1,
                visibility: "public",
                unpublished_at: "2026-09-20T00:00:00Z",
              };
            }
            return { version_number: 1, digest: await digest };
          },
          async run() {
            updates.push(sql);
            return {};
          },
        };
      },
    };
    const body = JSON.stringify({
      schema_version: 1,
      name: "Is aviation",
      slug: "is-aviation",
      task_type: "binary",
      inputs: { columns: ["text"], mode: "selected" },
      definition: { instructions: "Is this aviation?" },
      backend: { provider: "typesafe", model: "jev" },
      annotations: [{ inputs: { text: "Airport" }, label: true, split: "train" }],
    });
    const digest = crypto.subtle.digest("SHA-256", new TextEncoder().encode(body)).then(
      (value) => [...new Uint8Array(value)].map((byte) => byte.toString(16).padStart(2, "0")).join(""),
    );
    const response = await publishFunction(
      new Request("https://ai-functions.dev/api/v1/functions/is-aviation", {
        method: "PUT",
        headers: { Authorization: "Bearer token" },
        body,
      }),
      { DB: database } as unknown as Parameters<typeof publishFunction>[1],
      "is-aviation",
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toMatchObject({ created: false, version: 1 });
    expect(updates).toHaveLength(1);
    expect(updates[0]).toContain("unpublished_at = NULL");
  });
});
