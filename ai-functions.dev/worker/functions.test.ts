import { describe, expect, it } from "vitest";

import {
  downloadFunctionArtifact,
  getFunction,
  getFunctionAnnotations,
  listFunctions,
} from "./functions";

class FakeStatement {
  async all() {
    return {
      results: [
        {
          namespace_slug: "octocat",
          slug: "is-aviation",
          display_name: "Is aviation",
          metadata_json: JSON.stringify({ description: "Finds aviation posts." }),
          task_type: "binary",
          visibility: "public",
          version_number: 2,
          training_annotation_count: 10,
          holdout_annotation_count: 2,
          rationale_count: 7,
          likes_count: 31,
          views_count: 450,
          downloads_count: 120,
          runs_count: 4800,
          version_views_count: 210,
          version_downloads_count: 52,
          version_runs_count: 900,
          updated_at: "2026-09-20T17:31:43.442Z",
        },
      ],
    };
  }
}

describe("public function feed", () => {
  it("returns latest public function metadata", async () => {
    let query = "";
    const env = {
      DB: {
        prepare(sql: string) {
          query = sql;
          return new FakeStatement();
        },
      },
    } as unknown as Parameters<typeof listFunctions>[0];

    const response = await listFunctions(env);

    expect(response.headers.get("Cache-Control")).toContain("must-revalidate");
    await expect(response.json()).resolves.toEqual({
      functions: [
        {
          reference: "octocat/is-aviation",
          name: "Is aviation",
          description: "Finds aviation posts.",
          taskType: "binary",
          visibility: "public",
          version: 2,
          trainingAnnotations: 10,
          holdoutAnnotations: 2,
          rationales: 7,
          likes: 31,
          views: 450,
          downloads: 120,
          runs: 4800,
          versionViews: 210,
          versionDownloads: 52,
          versionRuns: 900,
          updatedAt: "2026-09-20T17:31:43.442Z",
        },
      ],
    });
    expect(query).toContain("ORDER BY views_count DESC");

    await listFunctions(env, "downloads");
    expect(query).toContain("ORDER BY downloads_count DESC");
  });

  it("returns public metadata and downloads an immutable artifact", async () => {
    const row = {
      function_id: "fn-1",
      function_version_id: "fv-2",
      namespace_slug: "octocat",
      slug: "is-aviation",
      display_name: "Is aviation",
      metadata_json: JSON.stringify({ description: "Finds aviation posts." }),
      task_type: "binary",
      visibility: "public",
      version_number: 2,
      schema_version: 1,
      digest: "abc123",
      artifact_key: "functions/octocat/is-aviation/v2-abc123.json",
      summary_json: JSON.stringify({
        inputs: { columns: ["text"], mode: "selected" },
        definition: { instructions: "Is this aviation?" },
        backend: { provider: "typesafe", model: "jev" },
        learning: { batch_size: 5 },
        metrics: { training_score: 1 },
      }),
      training_annotation_count: 10,
      holdout_annotation_count: 2,
      rationale_count: 7,
      likes_count: 31,
      views_count: 450,
      downloads_count: 120,
      runs_count: 4800,
      version_views_count: 210,
      version_downloads_count: 52,
      version_runs_count: 900,
      created_at: "2026-09-19T17:31:43.442Z",
      updated_at: "2026-09-20T17:31:43.442Z",
    };
    const prepared: Array<{ sql: string; values: unknown[] }> = [];
    const batches: unknown[][] = [];
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
            return row;
          },
          async all() {
            return {
              results: [
                {
                  version_number: 2,
                  digest: "abc123",
                  schema_version: 1,
                  training_annotation_count: 10,
                  holdout_annotation_count: 2,
                  rationale_count: 7,
                  views_count: 210,
                  downloads_count: 52,
                  runs_count: 900,
                  created_at: "2026-09-19T17:31:43.442Z",
                },
              ],
            };
          },
        };
      },
      async batch(statements: unknown[]) {
        batches.push(statements);
        return [];
      },
    };
    const env = {
      DB: database,
      ARTIFACTS: {
        async get(key: string) {
          expect(key).toBe(row.artifact_key);
          const payload = '{"schema_version":1,"annotations":[{"label":true}]}'
          return {
            body: new Response(payload).body,
            async json() {
              return JSON.parse(payload);
            },
          };
        },
      },
    } as unknown as Parameters<typeof getFunction>[0];

    const metadata = await getFunction(env, "octocat", "is-aviation");
    expect(metadata.status).toBe(200);
    await expect(metadata.json()).resolves.toMatchObject({
      reference: "octocat/is-aviation",
      version: 2,
      digest: "abc123",
      views: 451,
      downloads: 120,
      inputs: { columns: ["text"], mode: "selected" },
      definition: { instructions: "Is this aviation?" },
      versions: [{ version: 2, views: 211, downloads: 52 }],
    });
    expect(batches).toHaveLength(1);
    expect(prepared.some((item) => item.sql.includes("views_count = function_stats.views_count + 1"))).toBe(
      true,
    );

    const annotations = await getFunctionAnnotations(
      env,
      "octocat",
      "is-aviation",
      2,
    );
    await expect(annotations.json()).resolves.toEqual({ annotations: [{ label: true }] });
    expect(batches).toHaveLength(1);

    const artifact = await downloadFunctionArtifact(
      env,
      "octocat",
      "is-aviation",
      2,
    );
    expect(artifact.headers.get("X-Jeva-Digest")).toBe("abc123");
    expect(artifact.headers.get("Cache-Control")).toContain("must-revalidate");
    await expect(artifact.text()).resolves.toContain('"annotations"');
    expect(batches).toHaveLength(2);
    expect(prepared.some((item) => item.sql.includes("function_version_stats"))).toBe(
      true,
    );
  });
});
