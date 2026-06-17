"use client";

import Link from "next/link";
import { useMemo, useState, type ReactNode } from "react";
import {
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  Database,
  FileText,
  GitBranch,
  LockKeyhole,
  Network,
  Route,
  Search,
  ServerCog,
  ShieldCheck,
  SlidersHorizontal,
  TriangleAlert,
  Wrench,
} from "lucide-react";
import MathText from "../../MathText";
import { QuizTransitionButton } from "../LearningPrimitives";
import {
  getNdcgAtK,
  getReciprocalRankAtK,
  getResidualRiskScore,
  rankRetrievalCandidates,
  type RetrievalCandidate,
  type RetrievalMethod,
  type SafetyControl,
} from "./cme295-lecture7/systemsMath";

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function formatNumber(value: number, digits = 2): string {
  return value.toFixed(digits);
}

function InlineMath({ text }: { text: string }) {
  return <MathText inline text={text} />;
}

function SectionBadge({
  icon,
  label,
  tone = "blue",
}: {
  icon: ReactNode;
  label: string;
  tone?: "blue" | "green" | "amber" | "rose" | "violet";
}) {
  const tones = {
    blue: "border-blue-600 bg-blue-50 text-blue-950",
    green: "border-emerald-600 bg-emerald-50 text-emerald-950",
    amber: "border-amber-600 bg-amber-50 text-amber-950",
    rose: "border-rose-600 bg-rose-50 text-rose-950",
    violet: "border-violet-600 bg-violet-50 text-violet-950",
  };

  return (
    <span
      className={`inline-flex items-center gap-2 rounded-md border px-3 py-2 text-sm font-semibold ${tones[tone]}`}
    >
      {icon}
      {label}
    </span>
  );
}

function ControlButton({
  children,
  isActive,
  onClick,
  tone = "blue",
}: {
  children: ReactNode;
  isActive: boolean;
  onClick: () => void;
  tone?: "blue" | "green" | "amber" | "rose" | "violet";
}) {
  const activeTones = {
    blue: "border-blue-600 bg-blue-600 text-white",
    green: "border-emerald-600 bg-emerald-600 text-white",
    amber: "border-amber-600 bg-amber-500 text-slate-950",
    rose: "border-rose-600 bg-rose-600 text-white",
    violet: "border-violet-600 bg-violet-600 text-white",
  };

  return (
    <button
      type="button"
      aria-pressed={isActive}
      onClick={onClick}
      className={`min-h-11 rounded-md border px-3 py-2 text-left text-sm font-semibold transition-colors ${
        isActive
          ? activeTones[tone]
          : "border-slate-300 bg-white text-slate-800 hover:border-slate-500"
      }`}
    >
      {children}
    </button>
  );
}

function Metric({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail?: ReactNode;
}) {
  return (
    <div className="min-w-0 rounded-lg border border-slate-300 bg-white p-4">
      <p className="text-sm font-semibold text-slate-500">{label}</p>
      <p className="mt-2 break-words text-2xl font-semibold text-slate-950">
        {value}
      </p>
      {detail && (
        <p className="mt-2 text-sm leading-6 text-slate-600">{detail}</p>
      )}
    </div>
  );
}

function FormulaPanel({
  title,
  formula,
  children,
}: {
  title: string;
  formula: string;
  children: ReactNode;
}) {
  return (
    <div className="rounded-lg border border-slate-300 bg-slate-50 p-4">
      <p className="text-sm font-semibold text-slate-600">{title}</p>
      <MathText
        text={formula}
        className="mt-3 max-w-full overflow-x-auto rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-950"
      />
      <div className="mt-3 text-sm leading-6 text-slate-700">{children}</div>
    </div>
  );
}

const routeOptions = [
  "Model memory",
  "RAG",
  "Tool call",
  "Agent loop",
] as const;
type RouteOption = (typeof routeOptions)[number];

const requestScenarios = {
  cutoff: {
    label: "Fresh fact",
    prompt:
      "Who won the local election that happened after the model was trained?",
    correct: "RAG",
    explanation:
      "The missing piece is external knowledge. Retrieval can put a current source into the prompt without changing model weights.",
  },
  exact: {
    label: "Exact record",
    prompt: "What is the status of invoice INV-2049 in the finance database?",
    correct: "Tool call",
    explanation:
      "The model needs structured data from a system of record. A tool can query the backend and return a grounded result.",
  },
  plan: {
    label: "Multi-step job",
    prompt:
      "Monitor room comfort, adjust the thermostat if needed, and report when the demo room is ready.",
    correct: "Agent loop",
    explanation:
      "The work requires observing state, planning an action, calling tools, observing the result, and stopping when the goal is reached.",
  },
  concept: {
    label: "Known concept",
    prompt: "Explain why self-attention compares queries with keys.",
    correct: "Model memory",
    explanation:
      "A well-trained model can answer stable conceptual knowledge directly, without retrieval or external action.",
  },
} as const;

type RequestScenarioId = keyof typeof requestScenarios;

function RequestRouterLab() {
  const [scenarioId, setScenarioId] = useState<RequestScenarioId>("cutoff");
  const [selectedRoute, setSelectedRoute] = useState<RouteOption | null>(null);
  const scenario = requestScenarios[scenarioId];
  const isCorrect = selectedRoute === scenario.correct;

  return (
    <section
      data-testid="request-router-lab"
      className="border-y border-slate-200 bg-[#f8fafc] text-slate-950"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.85fr_1.15fr]">
        <div className="min-w-0 space-y-4">
          <SectionBadge
            icon={<Route aria-hidden="true" size={18} />}
            label="System routing"
            tone="blue"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Route the request before choosing the system
          </h2>
          <p className="text-base leading-7 text-slate-700">
            System-augmented language models begin with a diagnosis. A stale
            fact asks for retrieval, a live record asks for a tool, a goal with
            feedback asks for an agent loop, and a stable concept may need no
            external system at all.
          </p>
          <div className="grid gap-2 sm:grid-cols-2">
            {(Object.keys(requestScenarios) as RequestScenarioId[]).map(
              (id) => (
                <ControlButton
                  key={id}
                  isActive={scenarioId === id}
                  onClick={() => {
                    setScenarioId(id);
                    setSelectedRoute(null);
                  }}
                >
                  {requestScenarios[id].label}
                </ControlButton>
              ),
            )}
          </div>
        </div>

        <div className="min-w-0 rounded-xl border border-slate-300 bg-white p-4">
          <div className="rounded-lg border border-slate-300 bg-slate-50 p-4">
            <p className="text-sm font-semibold text-slate-500">User request</p>
            <p className="mt-2 text-lg font-semibold text-slate-950">
              {scenario.prompt}
            </p>
          </div>

          <div className="mt-4 grid gap-2 sm:grid-cols-2">
            {routeOptions.map((route) => (
              <ControlButton
                key={route}
                tone="green"
                isActive={selectedRoute === route}
                onClick={() => setSelectedRoute(route)}
              >
                {route}
              </ControlButton>
            ))}
          </div>

          {selectedRoute && (
            <p
              role="status"
              className={`mt-4 rounded-lg border p-4 text-sm leading-6 ${
                isCorrect
                  ? "border-emerald-500 bg-emerald-50 text-emerald-950"
                  : "border-amber-500 bg-amber-50 text-amber-950"
              }`}
            >
              <span className="font-semibold">
                {isCorrect ? "Correct route." : "Not the best first route."}
              </span>{" "}
              {scenario.explanation}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

const chunkProfiles = {
  tiny: {
    label: "Tiny chunks",
    chunkSize: "120 tokens",
    overlap: "20 tokens",
    diagnosis:
      "High precision snippets, but abbreviations and references often lose the surrounding document context.",
  },
  balanced: {
    label: "Balanced chunks",
    chunkSize: "500 tokens",
    overlap: "90 tokens",
    diagnosis:
      "A practical default for text corpora: enough local context for embeddings while still letting retrieval target a focused passage.",
  },
  large: {
    label: "Large chunks",
    chunkSize: "1,400 tokens",
    overlap: "180 tokens",
    diagnosis:
      "More context survives, but embeddings blur multiple topics and generation pays for text that may not answer the query.",
  },
} as const;

type ChunkProfileId = keyof typeof chunkProfiles;

function KnowledgeBaseBuilder() {
  const [profileId, setProfileId] = useState<ChunkProfileId>("balanced");
  const profile = chunkProfiles[profileId];

  return (
    <section className="bg-white text-slate-950">
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 space-y-4">
          <SectionBadge
            icon={<Database aria-hidden="true" size={18} />}
            label="Knowledge base"
            tone="green"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            RAG starts before the user asks the question
          </h2>
          <p className="text-base leading-7 text-slate-700">
            Retrieval-augmented generation is not a magic long prompt. The
            system collects documents, divides them into chunks, embeds each
            chunk, then stores those vectors so a future query can retrieve a
            small relevant subset.
          </p>
          <div className="grid gap-3 sm:grid-cols-3">
            <Metric label="Collect" value="docs" detail="Choose the corpus." />
            <Metric
              label="Divide"
              value={profile.chunkSize}
              detail={`${profile.overlap} overlap`}
            />
            <Metric
              label="Embed"
              value="vectors"
              detail="Thousands of dimensions are common."
            />
          </div>
        </div>

        <div
          data-testid="knowledge-base-builder"
          className="min-w-0 rounded-xl border border-slate-300 bg-slate-50 p-4"
        >
          <p className="text-sm font-semibold text-slate-600">
            Chunking profile
          </p>
          <div className="mt-3 grid gap-2 sm:grid-cols-3">
            {(Object.keys(chunkProfiles) as ChunkProfileId[]).map((id) => (
              <ControlButton
                key={id}
                tone="green"
                isActive={profileId === id}
                onClick={() => setProfileId(id)}
              >
                {chunkProfiles[id].label}
              </ControlButton>
            ))}
          </div>

          <div className="mt-5 grid gap-3 md:grid-cols-[0.85fr_1.15fr]">
            <div className="rounded-lg border border-slate-300 bg-white p-4">
              <p className="text-sm font-semibold text-slate-500">
                Stored chunk
              </p>
              <p className="mt-2 text-sm leading-6 text-slate-700">
                The model does not retrieve a whole document by default. It
                retrieves chunks whose embedding or keyword profile looks useful
                for the current query.
              </p>
            </div>
            <p
              role="status"
              className="rounded-lg border border-emerald-500 bg-emerald-50 p-4 text-sm leading-6 text-emerald-950"
            >
              {profile.diagnosis}
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

const retrievalCandidates: RetrievalCandidate[] = [
  {
    id: "atlas-policy",
    title: "Atlas policy: approval code INV-2049",
    semanticScore: 0.68,
    keywordScore: 0.96,
    relevance: 3,
  },
  {
    id: "atlas-overview",
    title: "Project Atlas rollout overview",
    semanticScore: 0.89,
    keywordScore: 0.34,
    relevance: 2,
  },
  {
    id: "expense-guide",
    title: "Expense reimbursement exact-code guide",
    semanticScore: 0.75,
    keywordScore: 0.81,
    relevance: 3,
  },
  {
    id: "room-notes",
    title: "Facilities room temperature notes",
    semanticScore: 0.51,
    keywordScore: 0.18,
    relevance: 0,
  },
  {
    id: "archive",
    title: "Archived invoice migration memo",
    semanticScore: 0.43,
    keywordScore: 0.72,
    relevance: 1,
  },
];

function RetrievalWorkbench() {
  const [method, setMethod] = useState<RetrievalMethod>("semantic");
  const [rerankEnabled, setRerankEnabled] = useState(false);
  const ranked = useMemo(() => {
    const initial = rankRetrievalCandidates(retrievalCandidates, method);
    if (!rerankEnabled) return initial;

    return [...initial].sort((first, second) => {
      const relevanceDiff = second.relevance - first.relevance;
      return relevanceDiff === 0 ? second.score - first.score : relevanceDiff;
    });
  }, [method, rerankEnabled]);
  const relevances = ranked.map((candidate) => candidate.relevance);
  const ndcg = getNdcgAtK(relevances, 5);
  const reciprocalRank = getReciprocalRankAtK(relevances, 5);

  return (
    <section
      id="retrieval-workbench"
      data-testid="retrieval-workbench"
      className="border-y border-slate-200 bg-[#edf5f2] text-slate-950"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.92fr_1.08fr]">
        <div className="min-w-0 space-y-4">
          <SectionBadge
            icon={<Search aria-hidden="true" size={18} />}
            label="Retrieval"
            tone="green"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Retrieval is a recall stage before a precision stage
          </h2>
          <p className="text-base leading-7 text-slate-700">
            Candidate retrieval quickly searches a large knowledge base. Ranking
            can spend more compute on a smaller list. Semantic search handles
            paraphrase, BM25 handles exact terms, and hybrid search is often the
            practical baseline.
          </p>
          <FormulaPanel
            title="Retrieval score used in this lab"
            formula={String.raw`\[\mathrm{hybrid}(q,d)=0.55\,s_{\mathrm{semantic}}(q,d)+0.45\,s_{\mathrm{BM25}}(q,d)\]`}
          >
            <span>
              A bi-encoder compares separate query and document embeddings
              cheaply. A cross-encoder reranker reads the query and chunk
              together, so it is slower but sharper on a smaller candidate set.
            </span>
          </FormulaPanel>
          <FormulaPanel
            title="Top-k retrieval metrics"
            formula={String.raw`\[\begin{aligned}
\mathrm{NDCG@}k &= \frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k}\\
\mathrm{RR@}k &= \frac{1}{r_1}
\end{aligned}\]`}
          >
            <span>
              <InlineMath text={String.raw`\(\mathrm{NDCG@}k\)`} /> rewards
              relevant chunks near the top.{" "}
              <InlineMath text={String.raw`\(\mathrm{RR@}k\)`} /> asks how early
              the first useful chunk appears, with{" "}
              <InlineMath text={String.raw`\(r_1\)`} /> as the rank of the first
              relevant result.
            </span>
          </FormulaPanel>
        </div>

        <div className="min-w-0 rounded-xl border border-slate-300 bg-white p-4">
          <div className="rounded-lg border border-slate-300 bg-slate-50 p-4">
            <p className="text-sm font-semibold text-slate-500">Query</p>
            <p className="mt-2 text-lg font-semibold text-slate-950">
              Find the approval rule for invoice INV-2049 in Project Atlas.
            </p>
          </div>

          <div className="mt-4 grid gap-2 sm:grid-cols-4">
            <ControlButton
              tone="green"
              isActive={method === "semantic"}
              onClick={() => setMethod("semantic")}
            >
              Semantic
            </ControlButton>
            <ControlButton
              tone="green"
              isActive={method === "keyword"}
              onClick={() => setMethod("keyword")}
            >
              BM25 keyword
            </ControlButton>
            <ControlButton
              tone="green"
              isActive={method === "hybrid"}
              onClick={() => setMethod("hybrid")}
            >
              Hybrid
            </ControlButton>
            <ControlButton
              tone="amber"
              isActive={rerankEnabled}
              onClick={() => setRerankEnabled(!rerankEnabled)}
            >
              Cross-encoder rerank
            </ControlButton>
          </div>

          <div className="mt-5 grid gap-3">
            {ranked.map((candidate, index) => (
              <div
                key={candidate.id}
                className="grid min-w-0 gap-3 rounded-lg border border-slate-300 bg-slate-50 p-3 sm:grid-cols-[auto_1fr_auto]"
              >
                <span className="flex h-8 w-8 items-center justify-center rounded-full bg-slate-900 text-sm font-bold text-white">
                  {index + 1}
                </span>
                <div className="min-w-0">
                  <p className="break-words text-sm font-semibold text-slate-900">
                    {candidate.title}
                  </p>
                  <p className="mt-1 text-xs leading-5 text-slate-600">
                    semantic {formatNumber(candidate.semanticScore)} / BM25{" "}
                    {formatNumber(candidate.keywordScore)} / relevance{" "}
                    {candidate.relevance}
                  </p>
                </div>
                <span className="text-sm font-semibold text-emerald-700">
                  {formatNumber(candidate.score)}
                </span>
              </div>
            ))}
          </div>

          <p
            role="status"
            className="mt-4 rounded-lg border border-emerald-500 bg-emerald-50 p-4 text-sm leading-6 text-emerald-950"
          >
            Retrieval report: NDCG@5 = {formatPercent(ndcg)} and RR@5 ={" "}
            {formatNumber(reciprocalRank)}.{" "}
            {rerankEnabled
              ? "The reranker spends more compute after recall to put the most relevant chunks first."
              : "The first pass optimizes cheap recall, so exact identifiers and semantic neighbors can trade places."}
          </p>
        </div>
      </div>
    </section>
  );
}

const toolRequests = {
  weather: {
    label: "Weather lookup",
    request: "Will the demo room need heating before the 9 AM session?",
    tool: "get_room_temperature",
    args: "{ room: 'Demo-2A', time: '09:00' }",
    output: "{ room: 'Demo-2A', currentF: 65, targetF: 70 }",
    response:
      "The room is 5F below target, so the assistant can recommend heating or call an authorized thermostat tool.",
  },
  invoice: {
    label: "Invoice lookup",
    request: "Is invoice INV-2049 approved?",
    tool: "get_invoice_status",
    args: "{ invoiceId: 'INV-2049' }",
    output: "{ status: 'needs manager approval', owner: 'Finance Ops' }",
    response:
      "The final answer should cite the returned status, not invent approval from the prompt wording.",
  },
  calculator: {
    label: "Calculator",
    request: "What is 17.5% tax on a 240 dollar order?",
    tool: "calculate",
    args: "{ expression: '240 * 0.175' }",
    output: "{ result: 42 }",
    response:
      "The model predicts the expression, the backend computes it, and the model explains the result in context.",
  },
} as const;

type ToolRequestId = keyof typeof toolRequests;

function ToolCallingConsole() {
  const [requestId, setRequestId] = useState<ToolRequestId>("invoice");
  const request = toolRequests[requestId];

  return (
    <section
      data-testid="tool-calling-console"
      className="bg-white text-slate-950"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.88fr_1.12fr]">
        <div className="min-w-0 space-y-4">
          <SectionBadge
            icon={<Wrench aria-hidden="true" size={18} />}
            label="Tool calling"
            tone="amber"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Tool calls split prediction from execution
          </h2>
          <p className="text-base leading-7 text-slate-700">
            A tool-using model does not directly query the database or change
            the world. It predicts the right function and arguments. The backend
            executes the function, then the model synthesizes a response from
            the returned data.
          </p>
          <div className="grid gap-3 md:grid-cols-3">
            {[
              ["Predict", "Choose function and arguments."],
              ["Execute", "Run trusted backend code."],
              ["Synthesize", "Answer from tool output."],
            ].map(([title, body], index) => (
              <div
                key={title}
                className="rounded-lg border border-slate-300 bg-slate-50 p-4"
              >
                <p className="text-sm font-semibold text-amber-700">
                  Step {index + 1}
                </p>
                <h3 className="mt-2 font-semibold text-slate-950">{title}</h3>
                <p className="mt-2 text-sm leading-6 text-slate-600">{body}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="min-w-0 rounded-xl border border-slate-300 bg-slate-50 p-4">
          <div className="grid gap-2 sm:grid-cols-3">
            {(Object.keys(toolRequests) as ToolRequestId[]).map((id) => (
              <ControlButton
                key={id}
                tone="amber"
                isActive={requestId === id}
                onClick={() => setRequestId(id)}
              >
                {toolRequests[id].label}
              </ControlButton>
            ))}
          </div>

          <div className="mt-5 rounded-lg border border-slate-300 bg-white p-4">
            <p className="text-sm font-semibold text-slate-500">User request</p>
            <p className="mt-2 text-lg font-semibold text-slate-950">
              {request.request}
            </p>
          </div>

          <div className="mt-4 grid gap-3 md:grid-cols-3">
            <div className="rounded-lg border border-amber-300 bg-amber-50 p-4">
              <p className="text-sm font-semibold text-amber-900">
                Tool prediction
              </p>
              <p className="mt-2 break-words font-mono text-sm text-amber-950">
                {request.tool}
              </p>
              <p className="mt-2 break-words font-mono text-xs leading-5 text-amber-900">
                {request.args}
              </p>
            </div>
            <div className="rounded-lg border border-blue-300 bg-blue-50 p-4">
              <p className="text-sm font-semibold text-blue-900">
                Backend result
              </p>
              <p className="mt-2 break-words font-mono text-xs leading-5 text-blue-950">
                {request.output}
              </p>
            </div>
            <div className="rounded-lg border border-emerald-300 bg-emerald-50 p-4">
              <p className="text-sm font-semibold text-emerald-900">
                Final response
              </p>
              <p className="mt-2 text-sm leading-6 text-emerald-950">
                {request.response}
              </p>
            </div>
          </div>

          <p
            role="status"
            className="mt-4 rounded-lg border border-slate-300 bg-white p-4 text-sm leading-6 text-slate-700"
          >
            Selected function API:{" "}
            <span className="font-mono font-semibold">{request.tool}</span>.
            Clear names, schemas, and descriptions matter because the model is
            predicting structured arguments, not just producing prose.
          </p>
        </div>
      </div>
    </section>
  );
}

function ToolSelectionAndMcp() {
  return (
    <section className="border-y border-slate-200 bg-[#f6f0e8] text-slate-950">
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="min-w-0 space-y-4">
          <SectionBadge
            icon={<ServerCog aria-hidden="true" size={18} />}
            label="Tool scale"
            tone="amber"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            More tools can make the model worse before they make it stronger
          </h2>
          <p className="text-base leading-7 text-slate-700">
            Every available tool consumes context and competes for selection. A
            router can first choose a small relevant subset, then the main model
            sees only those detailed APIs. MCP gives tool providers a standard
            way to expose tools, prompts, and resources to model hosts.
          </p>
        </div>

        <div className="min-w-0 rounded-xl border border-slate-300 bg-white p-4">
          <div className="grid gap-3 md:grid-cols-[1fr_auto_1fr]">
            <div className="rounded-lg border border-slate-300 bg-slate-50 p-4">
              <Network aria-hidden="true" className="text-amber-700" />
              <h3 className="mt-3 font-semibold">Tool router</h3>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                Reads the query and a compact tool catalog, then selects the few
                APIs worth exposing in detail.
              </p>
            </div>
            <ArrowRight
              aria-hidden="true"
              className="mx-auto self-center text-slate-500 md:rotate-0 rotate-90"
            />
            <div className="rounded-lg border border-slate-300 bg-slate-50 p-4">
              <GitBranch aria-hidden="true" className="text-emerald-700" />
              <h3 className="mt-3 font-semibold">MCP boundary</h3>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                Host, client, and server separate model experience from provider
                implementation so tools are not rebuilt for every model.
              </p>
            </div>
          </div>
          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            <Metric
              label="Tools"
              value="functions"
              detail="Actions and data."
            />
            <Metric label="Prompts" value="templates" detail="Known tasks." />
            <Metric label="Resources" value="context" detail="External data." />
          </div>
        </div>
      </div>
    </section>
  );
}

const agentSteps = [
  {
    label: "Input",
    title: "Goal arrives",
    detail:
      "The user wants the demo room ready, not just a paragraph about comfort.",
  },
  {
    label: "Observe",
    title: "State is incomplete",
    detail:
      "The agent knows the room may be cold, but it must read current temperature before acting.",
  },
  {
    label: "Plan",
    title: "Choose the next task",
    detail:
      "Determine the current temperature, then decide whether heating is needed.",
  },
  {
    label: "Act",
    title: "Call a tool",
    detail: "get_room_temperature(room='Demo-2A') returns 65F.",
  },
  {
    label: "Plan",
    title: "Update the room",
    detail: "The room is 5F below target, so call set_temperature(deltaF=5).",
  },
  {
    label: "Output",
    title: "Stop with evidence",
    detail:
      "The thermostat target is now 70F. The agent reports the action and exits the loop.",
  },
] as const;

function AgentLoopLab() {
  const [stepIndex, setStepIndex] = useState(0);
  const step = agentSteps[stepIndex];

  return (
    <section
      id="agent-loop"
      data-testid="agent-loop-lab"
      className="bg-[#111827] text-slate-50"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 space-y-4">
          <SectionBadge
            icon={<BrainCircuit aria-hidden="true" size={18} />}
            label="Agents"
            tone="violet"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            An agent is a loop, not a single tool call
          </h2>
          <p className="text-base leading-7 text-slate-300">
            ReAct means reason plus act. The system observes what is known,
            plans the next useful move, acts through a tool, observes the new
            state, and repeats until the goal is satisfied or it must stop.
          </p>
          <FormulaPanel
            title="Loop shape"
            formula={String.raw`\[\begin{gathered}
\mathrm{input}\rightarrow\mathrm{observe}\rightarrow\mathrm{plan}\rightarrow\mathrm{act}\\
\mathrm{act}\rightarrow\mathrm{observe}\rightarrow\cdots\rightarrow\mathrm{output}
\end{gathered}\]`}
          >
            <span>
              Tool calls can appear inside the loop, but the loop is what makes
              the system agentic.
            </span>
          </FormulaPanel>
        </div>

        <div className="min-w-0 rounded-xl border border-slate-700 bg-slate-950 p-4">
          <div className="grid gap-2 sm:grid-cols-3">
            {agentSteps.map((candidate, index) => (
              <button
                key={`${candidate.label}-${index}`}
                type="button"
                aria-pressed={stepIndex === index}
                onClick={() => setStepIndex(index)}
                className={`min-h-11 rounded-md border px-3 py-2 text-left text-sm font-semibold ${
                  stepIndex === index
                    ? "border-violet-300 bg-violet-300 text-slate-950"
                    : "border-slate-700 bg-slate-900 text-slate-200 hover:border-slate-500"
                }`}
              >
                {index + 1}. {candidate.label}
              </button>
            ))}
          </div>

          <div className="mt-5 rounded-lg border border-violet-300/40 bg-violet-300/10 p-5">
            <p className="text-sm font-semibold text-violet-200">
              Current loop state
            </p>
            <h3 className="mt-2 text-2xl font-semibold text-white">
              {step.title}
            </h3>
            <p role="status" className="mt-3 text-sm leading-6 text-slate-200">
              {step.detail}
            </p>
          </div>

          <div className="mt-5 grid gap-3 md:grid-cols-[1fr_auto_1fr]">
            <div className="rounded-lg border border-slate-700 bg-slate-900 p-4">
              <p className="text-sm font-semibold text-slate-300">
                Single tool call
              </p>
              <p className="mt-2 text-sm leading-6 text-slate-400">
                One function runs, then the model answers from its output.
              </p>
            </div>
            <ArrowRight
              aria-hidden="true"
              className="mx-auto self-center text-violet-200 md:rotate-0 rotate-90"
            />
            <div className="rounded-lg border border-violet-400 bg-violet-300/10 p-4">
              <p className="text-sm font-semibold text-violet-100">
                Agentic workflow
              </p>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                Tool output changes the next observation, plan, action, and stop
                decision.
              </p>
            </div>
          </div>

          <div className="mt-5 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() =>
                setStepIndex((current) =>
                  current === agentSteps.length - 1 ? 0 : current + 1,
                )
              }
              className="rounded-md bg-violet-300 px-4 py-3 text-sm font-bold text-slate-950 hover:bg-violet-200"
            >
              Advance loop
            </button>
            <button
              type="button"
              onClick={() => setStepIndex(0)}
              className="rounded-md border border-slate-700 px-4 py-3 text-sm font-bold text-slate-100 hover:border-slate-500"
            >
              Reset
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}

const safetyControls: {
  id: SafetyControl;
  label: string;
  detail: string;
}[] = [
  {
    id: "scopedTools",
    label: "Scoped tools",
    detail: "Expose only the tools and permissions the task needs.",
  },
  {
    id: "humanApproval",
    label: "Human approval",
    detail: "Require approval before external writes, payments, or messages.",
  },
  {
    id: "egressFilter",
    label: "Egress filter",
    detail: "Block secrets or private data from leaving trusted boundaries.",
  },
  {
    id: "budgetLimit",
    label: "Budget limit",
    detail: "Stop runaway loops before they spend unbounded tokens or money.",
  },
  {
    id: "auditLog",
    label: "Audit log",
    detail: "Record actions and observations for debugging and review.",
  },
];

function SafetyLab() {
  const [selectedControls, setSelectedControls] = useState<SafetyControl[]>([
    "scopedTools",
    "auditLog",
  ]);
  const residualRisk = getResidualRiskScore(9, selectedControls);

  function toggleControl(control: SafetyControl) {
    setSelectedControls((current) =>
      current.includes(control)
        ? current.filter((item) => item !== control)
        : [...current, control],
    );
  }

  return (
    <section data-testid="safety-lab" className="bg-white text-slate-950">
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 space-y-4">
          <SectionBadge
            icon={<ShieldCheck aria-hidden="true" size={18} />}
            label="Safety"
            tone="rose"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Acting systems need safeguards before they need scale
          </h2>
          <p className="text-base leading-7 text-slate-700">
            Tool use and agents introduce real-world consequences. Data
            exfiltration, unsafe external actions, runaway loops, and
            hallucinated arguments need defenses at training time, inference
            time, and operations time.
          </p>
          <div className="rounded-lg border border-rose-300 bg-rose-50 p-4">
            <div className="flex items-center gap-3">
              <TriangleAlert aria-hidden="true" className="text-rose-700" />
              <p className="font-semibold text-rose-950">
                Example risk: an email tool can leak private data if an injected
                instruction asks the agent to send secrets to an outside
                address.
              </p>
            </div>
          </div>
        </div>

        <div className="min-w-0 rounded-xl border border-slate-300 bg-slate-50 p-4">
          <div className="grid gap-2 sm:grid-cols-2">
            {safetyControls.map((control) => (
              <button
                key={control.id}
                type="button"
                aria-pressed={selectedControls.includes(control.id)}
                onClick={() => toggleControl(control.id)}
                className={`min-h-11 rounded-md border px-3 py-2 text-left text-sm transition-colors ${
                  selectedControls.includes(control.id)
                    ? "border-rose-600 bg-rose-600 text-white"
                    : "border-slate-300 bg-white text-slate-800 hover:border-slate-500"
                }`}
              >
                <span className="font-semibold">{control.label}</span>
                <span className="mt-1 block leading-5">{control.detail}</span>
              </button>
            ))}
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <Metric label="Base risk" value="9 / 10" />
            <Metric
              label="Controls"
              value={String(selectedControls.length)}
              detail="Selected safeguards"
            />
            <Metric label="Residual risk" value={`${residualRisk} / 10`} />
          </div>

          <p
            role="status"
            className={`mt-4 rounded-lg border p-4 text-sm leading-6 ${
              residualRisk <= 3
                ? "border-emerald-500 bg-emerald-50 text-emerald-950"
                : "border-amber-500 bg-amber-50 text-amber-950"
            }`}
          >
            Residual risk score: {residualRisk} / 10.{" "}
            {residualRisk <= 3
              ? "The design has enough layered control to discuss deployment-specific validation."
              : "The design still leaves too much room for unsafe action or data leakage."}
          </p>
        </div>
      </div>
    </section>
  );
}

function HeroVisual() {
  const lanes = [
    {
      icon: <Database aria-hidden="true" size={20} />,
      title: "Retrieve",
      body: "Find relevant chunks before generation.",
    },
    {
      icon: <Wrench aria-hidden="true" size={20} />,
      title: "Call",
      body: "Predict arguments, execute backend code.",
    },
    {
      icon: <BrainCircuit aria-hidden="true" size={20} />,
      title: "Loop",
      body: "Observe, plan, act, and stop.",
    },
  ];

  return (
    <div
      aria-label="System augmented LLM flow"
      className="min-w-0 rounded-xl border border-slate-700 bg-slate-900 p-4 shadow-2xl shadow-black/30"
    >
      <div className="rounded-lg border border-blue-400/30 bg-blue-400/10 p-4">
        <p className="text-sm font-semibold text-blue-100">User request</p>
        <p className="mt-2 text-sm leading-6 text-slate-200">
          Answer something current, look up a structured record, or complete a
          goal through external actions.
        </p>
      </div>

      <div className="my-4 flex justify-center">
        <ArrowRight aria-hidden="true" className="rotate-90 text-slate-500" />
      </div>

      <div className="rounded-lg border border-emerald-400/30 bg-emerald-400/10 p-4">
        <div className="flex items-center gap-2 text-emerald-100">
          <Route aria-hidden="true" size={18} />
          <p className="font-semibold">Route by bottleneck</p>
        </div>
        <p className="mt-2 text-sm leading-6 text-slate-300">
          Knowledge, computation, action, and multi-step goals need different
          system patterns.
        </p>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        {lanes.map((lane) => (
          <div
            key={lane.title}
            className="rounded-lg border border-slate-700 bg-slate-950 p-3"
          >
            <div className="flex items-center gap-2 text-amber-200">
              {lane.icon}
              <h3 className="font-semibold">{lane.title}</h3>
            </div>
            <p className="mt-2 text-sm leading-5 text-slate-400">{lane.body}</p>
          </div>
        ))}
      </div>

      <div className="mt-4 rounded-lg border border-rose-400/30 bg-rose-400/10 p-4">
        <div className="flex items-center gap-2 text-rose-100">
          <LockKeyhole aria-hidden="true" size={18} />
          <p className="font-semibold">Guardrails travel with capability</p>
        </div>
        <p className="mt-2 text-sm leading-6 text-slate-300">
          The more the model can retrieve, call, and act, the more the system
          needs scopes, approvals, egress filters, budgets, and logs.
        </p>
      </div>
    </div>
  );
}

function Recap() {
  const items = [
    {
      icon: <FileText aria-hidden="true" className="text-emerald-700" />,
      title: "RAG changes context, not weights",
      body: "It retrieves relevant chunks, augments the prompt, and generates from the grounded context.",
    },
    {
      icon: <Search aria-hidden="true" className="text-blue-700" />,
      title: "Retrieval is evaluated separately",
      body: "A bad answer can come from missing chunks, weak ranking, or generation after good retrieval.",
    },
    {
      icon: <Wrench aria-hidden="true" className="text-amber-700" />,
      title: "Tools execute outside the model",
      body: "The model predicts function calls; trusted backend code performs the actual data access, computation, or action.",
    },
    {
      icon: <CheckCircle2 aria-hidden="true" className="text-violet-700" />,
      title: "Agents loop until the goal is handled",
      body: "Observation, planning, action, and stopping criteria distinguish an agent from a one-shot tool call.",
    },
  ];

  return (
    <section className="border-t border-slate-800 bg-[#0d1720] text-slate-50">
      <div className="mx-auto w-full max-w-6xl px-4 py-10">
        <div className="flex flex-col gap-5 md:flex-row md:items-end md:justify-between">
          <div className="max-w-2xl">
            <h2 className="text-2xl font-semibold">Systems recap</h2>
            <p className="mt-2 text-sm leading-6 text-slate-300">
              Keep the boundary clear: retrieval supplies knowledge, tools
              supply structured execution, agents supply repeated control, and
              safety supplies the permission model around all of them.
            </p>
          </div>
          <Link
            href="/learn/stanford-cme295"
            className="inline-flex items-center justify-center rounded-lg border border-slate-600 px-4 py-3 text-sm font-bold text-slate-100 transition-colors hover:border-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-300 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950"
          >
            Back to course
          </Link>
        </div>

        <div className="mt-6 grid gap-3 md:grid-cols-2">
          {items.map((item) => (
            <div
              key={item.title}
              className="min-w-0 rounded-lg border border-slate-800 bg-slate-900 p-4"
            >
              <div className="flex items-center gap-3">
                <span className="rounded-md bg-white p-2">{item.icon}</span>
                <h3 className="font-semibold text-slate-50">{item.title}</h3>
              </div>
              <p className="mt-3 text-sm leading-6 text-slate-300">
                {item.body}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function StanfordCME295Lecture7SystemsPage() {
  return (
    <main className="min-h-screen bg-slate-950 text-slate-50">
      <section className="border-b border-slate-800 bg-[#0b1320]">
        <div className="mx-auto grid min-h-[560px] w-full max-w-6xl items-center gap-8 px-4 py-10 md:py-14 lg:grid-cols-[0.95fr_1.05fr]">
          <div className="min-w-0 space-y-6">
            <Link
              href="/learn/stanford-cme295"
              className="inline-flex text-sm font-semibold text-blue-200 hover:text-blue-100"
            >
              Back to Stanford CME295 course
            </Link>
            <div className="flex flex-wrap gap-2">
              <SectionBadge
                icon={<BrainCircuit aria-hidden="true" size={18} />}
                label="Stanford CME295"
                tone="blue"
              />
              <SectionBadge
                icon={<SlidersHorizontal aria-hidden="true" size={18} />}
                label="RAG, tools, agents"
                tone="green"
              />
            </div>
            <div className="space-y-4">
              <h1 className="max-w-3xl text-3xl font-semibold text-slate-50 md:text-5xl md:leading-tight">
                Connect language models to systems they can inspect and act
                through
              </h1>
              <p className="max-w-2xl text-base leading-7 text-slate-300 md:text-lg">
                Build the outside-world mental model: decide whether a request
                needs retrieval, a tool, or an agent loop; tune the retrieval
                path; trace a tool call; then add the safety controls that make
                action-bearing systems usable.
              </p>
              <p className="text-sm font-semibold text-emerald-300">
                Lecture 7 quiz source available
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <QuizTransitionButton
                sourceId="cme295-lect7"
                label="Start systems questions"
              />
              <a
                href="#retrieval-workbench"
                className="inline-flex items-center justify-center rounded-lg bg-emerald-400 px-4 py-3 text-sm font-bold text-slate-950 transition-colors hover:bg-emerald-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-300 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950"
              >
                Open retrieval workbench
              </a>
              <a
                href="#agent-loop"
                className="inline-flex items-center justify-center rounded-lg border border-slate-600 px-4 py-3 text-sm font-bold text-slate-100 transition-colors hover:border-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-300 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950"
              >
                Trace agent loop
              </a>
            </div>
          </div>

          <HeroVisual />
        </div>
      </section>

      <RequestRouterLab />
      <KnowledgeBaseBuilder />
      <RetrievalWorkbench />
      <ToolCallingConsole />
      <ToolSelectionAndMcp />
      <AgentLoopLab />
      <SafetyLab />
      <Recap />
    </main>
  );
}
