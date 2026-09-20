import { describe, expect, it } from "vitest";

import { summarizeArtifact } from "./publish";

function artifact() {
  return {
    schema_version: 1,
    name: "Is aviation",
    slug: "is-aviation",
    description: "Finds aviation posts.",
    task_type: "binary",
    inputs: { columns: ["title"], mode: "selected" },
    definition: { instructions: "Classify it" },
    backend: { provider: "typesafe", model: "jev" },
    annotations: [
      { inputs: { title: "Airport" }, label: true, rationale: "Runway", split: "train" },
      { inputs: { title: "Bread" }, label: false, rationale: null, split: "holdout" },
    ],
  };
}

describe("function artifact validation", () => {
  it("summarizes public annotation counts", () => {
    expect(summarizeArtifact(artifact(), "is-aviation")).toEqual({
      name: "Is aviation",
      slug: "is-aviation",
      description: "Finds aviation posts.",
      taskType: "binary",
      trainingCount: 1,
      holdoutCount: 1,
      rationaleCount: 1,
      inputs: { columns: ["title"], mode: "selected" },
      definition: { instructions: "Classify it" },
      backend: { provider: "typesafe", model: "jev" },
      learning: {},
      metrics: {},
    });
  });

  it("rejects an artifact whose slug differs from the route", () => {
    expect(() => summarizeArtifact(artifact(), "something-else")).toThrow(
      "Artifact slug must match",
    );
  });

  it("rejects annotations without their labeled inputs", () => {
    const value = artifact();
    (value.annotations[0].inputs as Record<string, string>) = {};
    expect(() => summarizeArtifact(value, "is-aviation")).toThrow(
      "Every annotation must contain",
    );
  });
});
