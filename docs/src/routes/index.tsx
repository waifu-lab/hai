import { createFileRoute, Link } from "@tanstack/react-router";
import { HomeLayout } from "fumadocs-ui/layouts/home";
import { ArrowRight, Check, Copy } from "lucide-react";
import { useState } from "react";
import { PythonCode } from "@/components/code";
import { HydrogenAtom } from "@/components/hydrogen-atom";
import { baseOptions } from "@/lib/layout.shared";
import { gitConfig } from "@/lib/shared";

export const Route = createFileRoute("/")({
  component: Home,
  head: () => ({
    meta: [
      { title: "Hydrogen AI" },
      {
        name: "description",
        content: "A lightweight Python library for building AI workflows.",
      },
    ],
  }),
});

const githubUrl = `https://github.com/${gitConfig.user}/${gitConfig.repo}`;

const toolCode = `from hai.core import Flow
from hai.providers import create_text_provider
from hai.tool import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"{city}: sunny, 28°C"


llm = create_text_provider("openai", model="gpt-4o-mini", api_key=KEY)

result = await Flow(llm).run({
    "messages": [{"role": "user", "content": "What's the weather in Taipei?"}],
    "tools": [get_weather],
})
print(result["assistant_message"])`;

const pipelineCode = `loader = MarkdownLoader(source="./notes")
chunker = ChunkProcessor(splitter=RecursiveSplitter(chunk_size=500))
embed = EmbeddingProcessor(embedder=OpenAIEmbedding(api_key=KEY))
writer = VectorStoreWriter(store=SqliteVecStore(path="notes.db"))

loader >> chunker >> embed >> writer
await Flow(loader).run({})`;

function CopyCommand({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // 剪貼簿不可用時，使用者仍可手動選取文字
    }
  }

  return (
    <div className="inline-flex max-w-full items-center gap-3 rounded-lg border bg-fd-card px-4 py-2.5 font-mono text-sm">
      <span className="select-none text-fd-muted-foreground">$</span>
      <code className="truncate">{command}</code>
      <button
        type="button"
        onClick={copy}
        aria-label={copied ? "Copied" : "Copy install command"}
        className="text-fd-muted-foreground transition-colors hover:text-fd-foreground"
      >
        {copied ? <Check className="size-4" /> : <Copy className="size-4" />}
      </button>
    </div>
  );
}

function DocLink({
  href,
  className,
  children,
}: {
  href: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <Link to="/docs/$" params={{ _splat: href }} className={className}>
      {children}
    </Link>
  );
}

const providers = [
  { name: "OpenAI", code: `"openai", model="gpt-4o-mini"` },
  { name: "Anthropic", code: `"anthropic", model="claude-sonnet-4-5"` },
  { name: "Google", code: `"google", model="gemini-2.5-flash"` },
  { name: "Ollama", code: `"ollama", model="gemma3"` },
];

const plugins = [
  { name: "Qdrant", kind: "Vector store", pkg: "hydrogen-ai-qdrant" },
  {
    name: "sqlite-vec",
    kind: "Vector store in one file",
    pkg: "hydrogen-ai-sqlitevec",
  },
  { name: "Neo4j", kind: "Graph database", pkg: "hydrogen-ai-neo4j" },
];

function Feature({
  title,
  href,
  children,
  description,
}: {
  title: string;
  href: string;
  description: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="border-t py-14 md:py-20">
      <div className="max-w-2xl">
        <h2 className="text-2xl font-semibold tracking-tight md:text-3xl">
          {title}
        </h2>
        <p className="mt-3 text-fd-muted-foreground">{description}</p>
        <DocLink
          href={href}
          className="mt-3 inline-flex items-center gap-1 text-sm font-medium hover:underline"
        >
          Read the docs
          <ArrowRight className="size-4" />
        </DocLink>
      </div>
      <div className="mt-8 min-w-0">{children}</div>
    </section>
  );
}

function Home() {
  return (
    <HomeLayout {...baseOptions()} links={[{ text: "Docs", url: "/docs" }]}>
      <main className="mx-auto w-full max-w-6xl px-4 md:px-6">
        <section className="grid grid-cols-1 items-center gap-10 py-14 md:py-20 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
          <div>
            <h1 className="text-4xl font-semibold tracking-tight md:text-5xl">
              Hydrogen AI
            </h1>
            <p className="mt-4 max-w-lg text-lg text-fd-muted-foreground">
              A lightweight AI library for Python.
            </p>
            <p className="mt-2 max-w-lg text-sm text-fd-muted-foreground">
              Chain models, tools, and data into flows. Four dependencies, fully
              async.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <DocLink
                href="quick-start"
                className="inline-flex items-center gap-2 rounded-lg bg-fd-primary px-5 py-2.5 text-sm font-medium text-fd-primary-foreground transition-opacity hover:opacity-90"
              >
                Get started
                <ArrowRight className="size-4" />
              </DocLink>
            </div>
            <div className="mt-6">
              <CopyCommand command="pip install hydrogen-ai" />
            </div>
          </div>
          <HydrogenAtom className="mx-auto w-full max-w-75 sm:max-w-100 lg:max-w-130" />
        </section>

        <Feature
          title="One node, one job."
          href="flow"
          description={
            <>
              Loading files, chunking, embedding, calling a model: each step is
              a node. Connect them with &gt;&gt; and Flow runs them in order.
              The whole engine is about 100 lines you can read in one sitting.
            </>
          }
        >
          <PythonCode code={pipelineCode} />
        </Feature>

        <Feature
          title="Switch models with one argument."
          href="providers"
          description="OpenAI, Anthropic, Google, and Ollama share the same interface. There are no vendor SDKs to install; every provider is a plain HTTP call, and any OpenAI-compatible endpoint works too."
        >
          <div className="grid gap-3 sm:grid-cols-2">
            {providers.map((provider) => (
              <div
                key={provider.name}
                className="min-w-0 rounded-xl border bg-fd-card p-4"
              >
                <div className="text-sm font-medium">{provider.name}</div>
                <code className="mt-2 block truncate font-mono text-xs text-fd-muted-foreground">
                  create_text_provider({provider.code})
                </code>
              </div>
            ))}
          </div>
        </Feature>

        <Feature
          title="Functions are tools."
          href="tools"
          description="Add @tool to a function and its type hints and docstring become a schema the model can read. When the model calls it, hai runs the function and hands the result back."
        >
          <PythonCode code={toolCode} />
        </Feature>

        <Feature
          title="Install only what you use."
          href="plugins"
          description="The core depends on httpx, pydantic, aiofiles, and ftfy. Vector stores and graph databases ship as separate plugins."
        >
          <div className="grid gap-3 lg:grid-cols-3">
            {plugins.map((plugin) => (
              <div
                key={plugin.name}
                className="min-w-0 rounded-xl border bg-fd-card p-4"
              >
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-sm font-medium">{plugin.name}</span>
                  <span className="text-xs text-fd-muted-foreground">
                    {plugin.kind}
                  </span>
                </div>
                <code className="mt-2 block truncate font-mono text-xs text-fd-muted-foreground">
                  pip install {plugin.pkg}
                </code>
              </div>
            ))}
          </div>
        </Feature>

        <section className="border-t py-14 text-center md:py-20">
          <h2 className="text-2xl font-semibold tracking-tight md:text-3xl">
            Start with one node.
          </h2>
          <div className="mt-6 flex flex-col items-center gap-4">
            <CopyCommand command="pip install hydrogen-ai" />
            <DocLink
              href="quick-start"
              className="inline-flex items-center gap-2 rounded-lg bg-fd-primary px-5 py-2.5 text-sm font-medium text-fd-primary-foreground transition-opacity hover:opacity-90"
            >
              Get started
              <ArrowRight className="size-4" />
            </DocLink>
          </div>
        </section>
      </main>

      <footer className="border-t">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-2 px-4 py-8 text-sm text-fd-muted-foreground md:flex-row md:justify-between md:px-6">
          <p>
            Hydrogen AI · MIT License · Inspired by{" "}
            <a
              href="https://github.com/The-Pocket/PocketFlow"
              target="_blank"
              rel="noreferrer"
              className="hover:text-fd-foreground hover:underline"
            >
              PocketFlow
            </a>
          </p>
          <a
            href={githubUrl}
            target="_blank"
            rel="noreferrer"
            className="hover:text-fd-foreground hover:underline"
          >
            GitHub
          </a>
        </div>
      </footer>
    </HomeLayout>
  );
}
