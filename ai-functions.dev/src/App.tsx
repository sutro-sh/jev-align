import { useEffect, useState, type ReactNode } from "react";

interface FunctionSummary {
  reference: string;
  name: string;
  description: string | null;
  taskType: "binary" | "multiclass" | "multilabel" | "score";
  visibility: "public";
  version: number;
  trainingAnnotations: number;
  holdoutAnnotations: number;
  rationales: number;
  likes: number;
  downloads: number;
  runs: number;
  versionDownloads: number;
  versionRuns: number;
  updatedAt: string;
}

interface FunctionVersion {
  version: number;
  digest: string;
  schemaVersion: number;
  trainingAnnotations: number;
  holdoutAnnotations: number;
  rationales: number;
  downloads: number;
  runs: number;
  createdAt: string;
}

interface FunctionDetail extends FunctionSummary {
  digest: string;
  createdAt: string;
  inputs: { columns?: string[]; mode?: string };
  definition: Record<string, unknown>;
  backend: { provider?: string; model?: string };
  learning: Record<string, unknown>;
  metrics: Record<string, unknown>;
  versions: FunctionVersion[];
}

interface ArtifactAnnotation {
  inputs: Record<string, string>;
  label: boolean | number | string | string[];
  rationale: string | null;
  split: "train" | "holdout";
}

interface FunctionArtifact {
  annotations: ArtifactAnnotation[];
}

type AnnotationState =
  | { status: "idle" | "loading" | "error"; version: number | null; annotations: [] }
  | { status: "ready"; version: number; annotations: ArtifactAnnotation[] };

type FeedState =
  | { status: "loading"; functions: FunctionSummary[] }
  | { status: "ready"; functions: FunctionSummary[] }
  | { status: "error"; functions: FunctionSummary[] };

type DetailState =
  | { status: "loading"; detail: null }
  | { status: "ready"; detail: FunctionDetail }
  | { status: "missing" | "error"; detail: null };

function timestamp(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return `${date.toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

function detailPath(reference: string): string {
  return `/${reference.split("/").map(encodeURIComponent).join("/")}`;
}

function displayReference(reference: string): string {
  const [namespace, ...rest] = reference.split("/");
  return rest.length > 0 ? `@${namespace}/${rest.join("/")}` : reference;
}

function pathReference(): string | null {
  const parts = window.location.pathname.split("/").filter(Boolean);
  if (parts.length !== 2) return null;
  try {
    return parts.map(decodeURIComponent).join("/");
  } catch {
    return null;
  }
}

function Header({ context }: { context: string }) {
  return (
    <header>
      <a className="identity" href="/" aria-label="AI Functions registry home">
        <span className="cursor" aria-hidden="true">▪</span>
        <h1>ai-functions.dev</h1>
        <span className="branch">/ {displayReference(context)}</span>
      </a>
      <div className="header-meta">
        <a href="/AGENTS.md">agents</a>
        <a href="https://github.com/sutro-sh/jev-align">github ↗</a>
      </div>
    </header>
  );
}

function FunctionCard({ item, index }: { item: FunctionSummary; index: number }) {
  const totalAnnotations = item.trainingAnnotations + item.holdoutAnnotations;
  return (
    <a className="function-link" href={detailPath(item.reference)}>
      <article className="function-card">
        <div className="card-index" aria-hidden="true">
          {String(index + 1).padStart(2, "0")}
        </div>
        <div className="card-main">
          <div className="card-heading">
            <h2>{displayReference(item.reference)}</h2>
            <span className="version">v{item.version}</span>
          </div>
          {item.name !== item.reference.split("/").at(-1) ? (
            <p className="display-name">{item.name}</p>
          ) : null}
          {item.description ? <p className="function-description">{item.description}</p> : null}
          <dl className="metadata">
            <div><dt>type</dt><dd>{item.taskType}</dd></div>
            <div><dt>annotations</dt><dd>{totalAnnotations}</dd></div>
            <div><dt>training</dt><dd>{item.trainingAnnotations}</dd></div>
            <div><dt>held out</dt><dd>{item.holdoutAnnotations}</dd></div>
            <div><dt>downloads</dt><dd>{item.downloads}</dd></div>
            <div><dt>updated</dt><dd>{timestamp(item.updatedAt)}</dd></div>
          </dl>
        </div>
        <div className="status" aria-label="Public function">
          <span aria-hidden="true" />
          {item.visibility}
        </div>
      </article>
    </a>
  );
}

function Feed() {
  const [feed, setFeed] = useState<FeedState>({ status: "loading", functions: [] });
  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/v1/functions", { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("Registry request failed");
        return response.json() as Promise<{ functions: FunctionSummary[] }>;
      })
      .then(({ functions }) => setFeed({ status: "ready", functions }))
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setFeed({ status: "error", functions: [] });
      });
    return () => controller.abort();
  }, []);
  return (
    <div className="registry">
      <Header context="registry" />
      <main>
        <section className="human-instructions" aria-labelledby="human-instructions-title">
          <div className="instructions-label" id="human-instructions-title">
            human instructions
          </div>
          <ol>
            <li>
              <span>01</span>
              <p>
                Build a new AI Function using{" "}
                <a href="https://github.com/sutro-sh/jev-align#quickstart">jev-align ↗</a>
              </p>
            </li>
            <li>
              <span>02</span>
              <p>Authenticate with GitHub using <code>jeva login</code></p>
            </li>
            <li>
              <span>03</span>
              <p>Push your AI Function so others can discover it, pull it, and build on it.</p>
            </li>
          </ol>
        </section>
        <p className="team-hosting-note">
          Want private hosting for your team?{" "}
          <a href="mailto:team@sutro.sh">Contact us.</a>
        </p>
        <div className="list-label" aria-hidden="true">
          <span>index / function</span>
          <span>{feed.status === "ready" ? `${feed.functions.length} public` : "latest first"}</span>
        </div>
        {feed.status === "loading" ? <SystemMessage>reading registry…</SystemMessage> : null}
        {feed.status === "error" ? <SystemMessage error>registry unavailable</SystemMessage> : null}
        {feed.status === "ready" && feed.functions.length === 0 ? (
          <SystemMessage>no functions pushed yet</SystemMessage>
        ) : null}
        {feed.status === "ready" && feed.functions.length > 0 ? (
          <section className="function-list" aria-label="Published AI Functions">
            {feed.functions.map((item, index) => (
              <FunctionCard key={item.reference} item={item} index={index} />
            ))}
          </section>
        ) : null}
      </main>
      <Footer />
    </div>
  );
}

function SystemMessage({ children, error = false }: { children: ReactNode; error?: boolean }) {
  return (
    <div className={`system-message${error ? " error" : ""}`}>
      <span className={error ? "" : "cursor"}>{error ? "!" : "▪"}</span> {children}
    </div>
  );
}

function JsonBlock({ value }: { value: unknown }) {
  return <pre className="json-block"><code>{JSON.stringify(value, null, 2)}</code></pre>;
}

function DefinitionBlock({
  definition,
  taskType,
}: {
  definition: Record<string, unknown>;
  taskType: FunctionSummary["taskType"];
}) {
  const fields: Array<{ label: string; value: unknown }> = [];
  if (typeof definition.instructions === "string") {
    fields.push({ label: "instructions", value: definition.instructions });
  }
  if (taskType === "binary") {
    fields.push(
      { label: "true", value: definition.true_criteria },
      { label: "false", value: definition.false_criteria },
    );
  } else if (taskType === "multiclass" && definition.criteria && typeof definition.criteria === "object") {
    for (const [label, value] of Object.entries(definition.criteria)) {
      fields.push({ label, value });
    }
  } else if (taskType === "score" && Array.isArray(definition.levels)) {
    definition.levels.forEach((value, index) => fields.push({ label: `level ${index}`, value }));
  } else if (taskType === "multilabel" && definition.labels && typeof definition.labels === "object") {
    for (const [label, criteria] of Object.entries(definition.labels)) {
      if (!criteria || typeof criteria !== "object") continue;
      const values = criteria as Record<string, unknown>;
      fields.push(
        { label: `${label} · true`, value: values.true_criteria },
        { label: `${label} · false`, value: values.false_criteria },
      );
    }
  }
  const visible = fields.filter(({ value }) => value !== undefined);
  if (visible.length === 0) return <JsonBlock value={definition} />;
  return (
    <div className="definition-block">
      {visible.map(({ label, value }) => (
        <div className="definition-field" key={label}>
          <h4>{label}</h4>
          {typeof value === "string" || typeof value === "number" ? (
            <p>{String(value)}</p>
          ) : (
            <JsonBlock value={value} />
          )}
        </div>
      ))}
    </div>
  );
}

function labelText(label: ArtifactAnnotation["label"]): string {
  if (Array.isArray(label)) return label.join(", ") || "None";
  if (typeof label === "boolean") return label ? "True" : "False";
  return String(label);
}

function PullCommand({ reference }: { reference: string }) {
  const command = `jeva pull ${reference}`;
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };
  return (
    <div className="pull-command">
      <code><span>$</span> {command}</code>
      <button type="button" onClick={copy}>{copied ? "copied" : "copy"}</button>
    </div>
  );
}

function FunctionPage({ reference }: { reference: string }) {
  const [state, setState] = useState<DetailState>({ status: "loading", detail: null });
  useEffect(() => {
    const controller = new AbortController();
    fetch(`/api/v1/functions/${reference.split("/").map(encodeURIComponent).join("/")}`, {
      signal: controller.signal,
    })
      .then(async (response) => {
        if (response.status === 404) {
          setState({ status: "missing", detail: null });
          return null;
        }
        if (!response.ok) throw new Error("Function request failed");
        return response.json() as Promise<FunctionDetail>;
      })
      .then((detail) => {
        if (detail) setState({ status: "ready", detail });
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setState({ status: "error", detail: null });
      });
    return () => controller.abort();
  }, [reference]);
  return (
    <div className="registry">
      <Header context={reference} />
      <main className="detail-main">
        <a className="back-link" href="/">← registry</a>
        {state.status === "loading" ? <SystemMessage>reading function…</SystemMessage> : null}
        {state.status === "missing" ? <SystemMessage error>function not found</SystemMessage> : null}
        {state.status === "error" ? <SystemMessage error>function unavailable</SystemMessage> : null}
        {state.status === "ready" ? <FunctionDetailView detail={state.detail} /> : null}
      </main>
      <Footer />
    </div>
  );
}

function FunctionDetailView({ detail }: { detail: FunctionDetail }) {
  const annotationCount = detail.trainingAnnotations + detail.holdoutAnnotations;
  const [annotations, setAnnotations] = useState<AnnotationState>({
    status: "idle",
    version: null,
    annotations: [],
  });
  const [visibleCount, setVisibleCount] = useState(25);

  const loadAnnotations = async (version: number) => {
    setVisibleCount(25);
    setAnnotations({ status: "loading", version, annotations: [] });
    try {
      const path = detail.reference.split("/").map(encodeURIComponent).join("/");
      const response = await fetch(
        `/api/v1/functions/${path}/versions/${version}/artifact`,
      );
      if (!response.ok) throw new Error("Artifact request failed");
      const artifact = (await response.json()) as FunctionArtifact;
      if (!Array.isArray(artifact.annotations)) throw new Error("Invalid artifact");
      setAnnotations({ status: "ready", version, annotations: artifact.annotations });
    } catch {
      setAnnotations({ status: "error", version, annotations: [] });
    }
  };
  return (
    <article className="function-detail">
      <div className="detail-heading">
        <div>
          <div className="eyebrow">public AI function / v{detail.version}</div>
          <h2>{displayReference(detail.reference)}</h2>
          {detail.name !== detail.reference.split("/").at(-1) ? (
            <p className="detail-name">{detail.name}</p>
          ) : null}
          {detail.description ? <p className="detail-description">{detail.description}</p> : null}
        </div>
        <div className="status"><span aria-hidden="true" />public</div>
      </div>
      <PullCommand reference={detail.reference} />
      <dl className="detail-stats">
        <div><dt>task</dt><dd>{detail.taskType}</dd></div>
        <div><dt>annotations</dt><dd>{annotationCount}</dd></div>
        <div><dt>rationales</dt><dd>{detail.rationales}</dd></div>
        <div><dt>downloads</dt><dd>{detail.downloads}</dd></div>
        <div><dt>backend</dt><dd>{detail.backend.provider} / {detail.backend.model}</dd></div>
        <div><dt>updated</dt><dd>{timestamp(detail.updatedAt)}</dd></div>
      </dl>
      <div className="detail-grid">
        <section className="detail-section">
          <h3>definition</h3>
          <DefinitionBlock definition={detail.definition} taskType={detail.taskType} />
        </section>
        <section className="detail-section">
          <h3>input signature</h3>
          <div className="signature">
            <span>{detail.inputs.mode ?? "selected"}</span>
            {(detail.inputs.columns ?? []).map((column) => <code key={column}>{column}</code>)}
          </div>
          <h3 className="metrics-heading">latest metrics</h3>
          <JsonBlock value={detail.metrics} />
        </section>
      </div>
      <section className="detail-section versions-section">
        <h3>versions</h3>
        <div className="version-table" role="table" aria-label="Published versions">
          {detail.versions.map((version) => (
            <div className="version-row" role="row" key={version.version}>
              <strong>v{version.version}</strong>
              <span>{version.trainingAnnotations + version.holdoutAnnotations} labels</span>
              <span>{version.downloads} downloads</span>
              <span>{timestamp(version.createdAt)}</span>
              <code>{version.digest.slice(0, 10)}</code>
              <button type="button" onClick={() => void loadAnnotations(version.version)}>
                view labels
              </button>
            </div>
          ))}
        </div>
      </section>
      <section className="detail-section annotations-section" aria-live="polite">
        <div className="section-heading">
          <div>
            <h3>labels &amp; rationales</h3>
            <p>Published examples used to evaluate and continue improving this function.</p>
          </div>
          {annotations.status === "ready" ? (
            <button
              className="quiet-button"
              type="button"
              onClick={() => setAnnotations({ status: "idle", version: null, annotations: [] })}
            >
              hide
            </button>
          ) : null}
        </div>
        {annotations.status === "idle" ? (
          <button className="load-labels" type="button" onClick={() => void loadAnnotations(detail.version)}>
            view v{detail.version} labels
          </button>
        ) : null}
        {annotations.status === "loading" ? (
          <div className="inline-message"><span className="cursor">▪</span> reading v{annotations.version} artifact…</div>
        ) : null}
        {annotations.status === "error" ? (
          <div className="inline-message error">! labels unavailable</div>
        ) : null}
        {annotations.status === "ready" ? (
          <>
            <div className="annotation-summary">
              v{annotations.version} / {annotations.annotations.length} label{annotations.annotations.length === 1 ? "" : "s"}
            </div>
            <div className="annotation-list">
              {annotations.annotations.slice(0, visibleCount).map((annotation, index) => (
                <article className="annotation-card" key={`${annotations.version}-${index}`}>
                  <div className="annotation-header">
                    <span>#{String(index + 1).padStart(3, "0")}</span>
                    <span className={`split ${annotation.split}`}>{annotation.split}</span>
                    <strong>{labelText(annotation.label)}</strong>
                  </div>
                  <dl className="annotation-inputs">
                    {Object.entries(annotation.inputs).map(([name, value]) => (
                      <div key={name}><dt>{name}</dt><dd>{value}</dd></div>
                    ))}
                  </dl>
                  <div className="rationale">
                    <span>rationale</span>
                    <p>{annotation.rationale || "No rationale provided."}</p>
                  </div>
                </article>
              ))}
            </div>
            {visibleCount < annotations.annotations.length ? (
              <button
                className="load-labels"
                type="button"
                onClick={() => setVisibleCount((count) => count + 25)}
              >
                show 25 more
              </button>
            ) : null}
          </>
        ) : null}
      </section>
    </article>
  );
}

function Footer() {
  return (
    <footer>
      <span>$ jeva pull</span>
      <span>an experiment from sutro.sh</span>
    </footer>
  );
}

function App() {
  const reference = pathReference();
  return reference ? <FunctionPage reference={reference} /> : <Feed />;
}

export default App;
