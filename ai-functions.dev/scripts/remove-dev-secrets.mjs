import { rm } from "node:fs/promises";

await Promise.all(
  ["ai_functions_dev_registry", "jeva_registry"].map((directory) =>
    rm(new URL(`../dist/${directory}/.dev.vars`, import.meta.url), {
      force: true,
    }),
  ),
);
