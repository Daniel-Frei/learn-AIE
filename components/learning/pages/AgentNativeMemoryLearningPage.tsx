"use client";

import { useMemo, useState, type ReactNode } from "react";
import {
  ArrowRight,
  CheckCircle2,
  ClipboardCheck,
  Database,
  Gauge,
  GitBranch,
  Route,
  Search,
  SlidersHorizontal,
  TriangleAlert,
  Wrench,
} from "lucide-react";
import type { LearningExperience } from "../../../lib/learning";
import MathText from "../../MathText";
import { QuizTransitionButton } from "../LearningPrimitives";
import {
  architectures,
  extractionVariants,
  getArchitecture,
  getEvaluationResult,
  getEvidenceDistanceProjection,
  getExtractionVariant,
  getMaintenanceVariant,
  getRepresentationVariant,
  getRetrievalVariant,
  getWorkload,
  maintenanceVariants,
  representationVariants,
  retrievalVariants,
  workloads,
  type ArchitectureId,
  type EvaluationInput,
  type RetrievalVariantId,
} from "./agent-native-memory/evaluationWorkbench";

type Props = {
  experience: LearningExperience;
};

type ModuleId = "representation" | "extraction" | "retrieval" | "maintenance";

const moduleDetails: Record<
  ModuleId,
  {
    label: string;
    symbol: string;
    title: string;
    description: string;
    methods: readonly string[];
    icon: ReactNode;
  }
> = {
  representation: {
    label: "Representation and storage",
    symbol: "R",
    title: "What shape and backend does memory use?",
    description:
      "Representation is the logical structure, such as tokens, vectors, graphs, trees, or composites. Storage is the physical register, file, vector engine, graph database, relational table, or multi-engine backend that serves it.",
    methods: [
      "Transient in-context register",
      "Specialized single-engine store",
      "Heterogeneous multi-engine storage",
    ],
    icon: <Database aria-hidden="true" size={22} />,
  },
  extraction: {
    label: "Extraction",
    symbol: "S",
    title: "What gets written from raw interaction?",
    description:
      "Extraction turns dialogues, tool logs, observations, and traces into logical memory units before persistence. The wrong extraction step can discard evidence before the future query exists.",
    methods: [
      "Raw sequence concatenation",
      "Schema-free semantic extraction",
      "Schema-constrained structured extraction",
    ],
    icon: <ClipboardCheck aria-hidden="true" size={22} />,
  },
  retrieval: {
    label: "Retrieval and routing",
    symbol: "Q",
    title: "How does a query find useful evidence?",
    description:
      "Retrieval is not just a vector lookup. The paper compares attention-based retrieval, dense search, graph traversal, agentic function calls, query expansion, and hybrid pipelines.",
    methods: [
      "Native attention or dense search",
      "Topological traversal",
      "Agentic and hybrid routing",
    ],
    icon: <Search aria-hidden="true" size={22} />,
  },
  maintenance: {
    label: "Maintenance",
    symbol: "U",
    title: "How does memory stay current and bounded?",
    description:
      "Maintenance handles contradictions, growth, consolidation, deletion, and parameter-side updates. This is where stale-state errors and latency blowups often appear.",
    methods: [
      "Timestamped multi-versioning",
      "Capacity or score eviction",
      "Conservative consolidation",
    ],
    icon: <Wrench aria-hidden="true" size={22} />,
  },
};

const moduleOrder: readonly ModuleId[] = [
  "representation",
  "extraction",
  "retrieval",
  "maintenance",
];

const findingCards = [
  {
    label: "RQ1",
    title: "Workload-aligned memory",
    body: "No architecture dominates all workloads. Cross-session, exact dialogue, and stateful execution each stress a different bottleneck.",
    icon: <SlidersHorizontal aria-hidden="true" size={20} />,
  },
  {
    label: "RQ2",
    title: "Evidence completion",
    body: "A top-1 hit is not enough when support is old, scattered, or temporally distant. Structure helps assemble the full evidence set.",
    icon: <Route aria-hidden="true" size={20} />,
  },
  {
    label: "RQ3",
    title: "Temporal update fidelity",
    body: "Correct latest-state answers depend on revisable memory and query-time selectivity, not just a stronger answer model.",
    icon: <GitBranch aria-hidden="true" size={20} />,
  },
  {
    label: "RQ5",
    title: "Maintenance scope drives cost",
    body: "Global rewrites and multi-store coordination can dominate latency. Localized update and search create a better utility-cost frontier.",
    icon: <Gauge aria-hidden="true" size={20} />,
  },
];

function PageBand({
  children,
  tone = "white",
}: {
  children: ReactNode;
  tone?: "white" | "blue" | "grid" | "ink";
}) {
  const tones = {
    white: "bg-white text-slate-950",
    blue: "bg-sky-50 text-slate-950",
    grid: "bg-[linear-gradient(#e2e8f0_1px,transparent_1px),linear-gradient(90deg,#e2e8f0_1px,transparent_1px)] bg-[size:24px_24px] bg-slate-50 text-slate-950",
    ink: "bg-zinc-950 text-zinc-50",
  };

  return <section className={tones[tone]}>{children}</section>;
}

function PageInner({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`mx-auto min-w-0 w-full max-w-7xl px-4 py-12 md:py-16 ${className}`}
    >
      {children}
    </div>
  );
}

function SectionIntro({
  eyebrow,
  title,
  body,
}: {
  eyebrow: string;
  title: string;
  body: string;
}) {
  return (
    <div className="max-w-3xl">
      <p className="text-sm font-semibold uppercase tracking-wide text-blue-700">
        {eyebrow}
      </p>
      <h2 className="mt-2 text-3xl font-semibold tracking-normal text-slate-950 md:text-4xl">
        {title}
      </h2>
      <p className="mt-4 text-base leading-7 text-slate-700 md:text-lg">
        {body}
      </p>
    </div>
  );
}

function ControlButton({
  children,
  isActive,
  onClick,
}: {
  children: ReactNode;
  isActive: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={isActive}
      onClick={onClick}
      className={`min-h-11 rounded-md border px-3 py-2 text-left text-sm font-semibold leading-5 transition-colors ${
        isActive
          ? "border-blue-700 bg-blue-700 text-white shadow-sm"
          : "border-slate-300 bg-white text-slate-800 hover:border-slate-500"
      }`}
    >
      {children}
    </button>
  );
}

function MetricPill({
  label,
  value,
  tone = "blue",
}: {
  label: string;
  value: number;
  tone?: "blue" | "green" | "amber" | "rose" | "violet";
}) {
  const tones = {
    blue: "bg-blue-600",
    green: "bg-emerald-600",
    amber: "bg-amber-500",
    rose: "bg-rose-600",
    violet: "bg-violet-600",
  };
  return (
    <div className="min-w-0 rounded-md border border-slate-200 bg-white p-3">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-semibold text-slate-700">{label}</span>
        <span className="text-sm text-slate-500">{value}/5</span>
      </div>
      <div className="mt-2 h-2 rounded-full bg-slate-200">
        <div
          className={`h-2 rounded-full ${tones[tone]}`}
          style={{ width: `${value * 20}%` }}
        />
      </div>
    </div>
  );
}

function HeroConsole() {
  return (
    <div className="rounded-lg border border-zinc-700 bg-zinc-900 p-4 shadow-2xl">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-700 pb-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-sky-300">
            Memory system testbed
          </p>
          <p className="mt-1 text-lg font-semibold text-white">
            R, S, Q, U under workload stress
          </p>
        </div>
        <span className="rounded-md border border-emerald-400/40 bg-emerald-950 px-3 py-2 text-sm font-semibold text-emerald-100">
          5 research questions
        </span>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        {moduleOrder.map((id) => {
          const item = moduleDetails[id];
          return (
            <div
              key={id}
              className="rounded-lg border border-zinc-700 bg-zinc-950 p-4"
            >
              <div className="flex items-center gap-2 text-sky-300">
                {item.icon}
                <span className="text-lg font-bold">{item.symbol}</span>
              </div>
              <p className="mt-2 text-sm font-semibold text-white">
                {item.label}
              </p>
              <p className="mt-2 text-xs leading-5 text-zinc-400">
                {item.methods[0]}
              </p>
            </div>
          );
        })}
      </div>

      <div className="mt-4 rounded-lg border border-sky-400/40 bg-sky-950/50 p-4">
        <MathText
          text={String.raw`\[\mathcal{M}_{sys}=\langle R,S,Q,U\rangle\]`}
          className="min-w-0 max-w-full overflow-x-auto overflow-y-hidden text-sky-50 [&_.katex-mathml]:hidden"
        />
        <p className="mt-3 text-sm leading-6 text-zinc-300">
          The paper asks whether memory is ready as infrastructure. The answer
          depends on which module fails under which workload.
        </p>
      </div>
    </div>
  );
}

function ModuleRack() {
  const [activeId, setActiveId] = useState<ModuleId>("representation");
  const active = moduleDetails[activeId];

  return (
    <div
      data-testid="agent-native-module-rack"
      className="mt-8 grid gap-5 lg:grid-cols-[18rem_1fr]"
    >
      <div className="rounded-lg border border-slate-300 bg-white p-4">
        <p className="text-sm font-semibold uppercase tracking-wide text-blue-700">
          Four-module rack
        </p>
        <div className="mt-4 grid gap-2">
          {moduleOrder.map((id) => (
            <ControlButton
              key={id}
              isActive={activeId === id}
              onClick={() => setActiveId(id)}
            >
              <span className="flex items-center gap-2">
                {moduleDetails[id].icon}
                <span>
                  {moduleDetails[id].symbol}: {moduleDetails[id].label}
                </span>
              </span>
            </ControlButton>
          ))}
        </div>
      </div>

      <div className="rounded-lg border border-blue-200 bg-blue-50 p-5">
        <div className="flex flex-wrap items-center gap-3">
          <span className="flex h-12 w-12 items-center justify-center rounded-lg border border-blue-300 bg-white text-blue-800">
            {active.icon}
          </span>
          <div>
            <p className="text-sm font-semibold uppercase tracking-wide text-blue-700">
              Module {active.symbol}
            </p>
            <h3 className="text-2xl font-semibold text-slate-950">
              {active.title}
            </h3>
          </div>
        </div>
        <p className="mt-4 max-w-3xl text-sm leading-6 text-slate-700">
          {active.description}
        </p>
        <div className="mt-5 grid gap-3 md:grid-cols-3">
          {active.methods.map((method) => (
            <div
              key={method}
              className="rounded-lg border border-blue-200 bg-white p-4 text-sm font-semibold leading-6 text-slate-800"
            >
              {method}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function FindingStrip() {
  return (
    <div className="mt-8 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      {findingCards.map((card) => (
        <div
          key={card.label}
          className="rounded-lg border border-slate-300 bg-white p-5"
        >
          <div className="flex items-center justify-between gap-3">
            <span className="rounded-md border border-blue-200 bg-blue-50 px-2 py-1 text-sm font-bold text-blue-800">
              {card.label}
            </span>
            <span className="text-blue-700">{card.icon}</span>
          </div>
          <h3 className="mt-4 text-lg font-semibold text-slate-950">
            {card.title}
          </h3>
          <p className="mt-2 text-sm leading-6 text-slate-700">{card.body}</p>
        </div>
      ))}
    </div>
  );
}

function EvaluationConsole() {
  const [input, setInput] = useState<EvaluationInput>({
    workloadId: "cross-session",
    architectureId: "hierarchical-tree",
    representationId: "raw-retentive",
    extractionId: "coverage-first",
    retrievalId: "balanced-hybrid",
    maintenanceId: "conservative-merge",
  });

  const result = useMemo(() => getEvaluationResult(input), [input]);
  const workload = getWorkload(input.workloadId);
  const architecture = getArchitecture(input.architectureId);
  const representation = getRepresentationVariant(input.representationId);
  const extraction = getExtractionVariant(input.extractionId);
  const retrieval = getRetrievalVariant(input.retrievalId);
  const maintenance = getMaintenanceVariant(input.maintenanceId);

  const setValue = <TKey extends keyof EvaluationInput>(
    key: TKey,
    value: EvaluationInput[TKey],
  ) => setInput((current) => ({ ...current, [key]: value }));

  return (
    <div
      data-testid="agent-native-evaluation-console"
      className="mt-8 rounded-lg border border-slate-300 bg-white p-5"
    >
      <div className="grid gap-6 xl:grid-cols-[1fr_0.86fr]">
        <div className="space-y-6">
          <div>
            <p className="text-sm font-semibold uppercase tracking-wide text-blue-700">
              1. Workload bottleneck
            </p>
            <div className="mt-3 grid gap-2 md:grid-cols-2">
              {workloads.map((item) => (
                <ControlButton
                  key={item.id}
                  isActive={input.workloadId === item.id}
                  onClick={() => setValue("workloadId", item.id)}
                >
                  <span className="block">{item.label}</span>
                  <span className="mt-1 block text-xs font-normal opacity-80">
                    {item.benchmark}
                  </span>
                </ControlButton>
              ))}
            </div>
          </div>

          <div>
            <p className="text-sm font-semibold uppercase tracking-wide text-blue-700">
              2. Architecture family
            </p>
            <div className="mt-3 grid gap-2 md:grid-cols-2">
              {architectures.map((item) => (
                <ControlButton
                  key={item.id}
                  isActive={input.architectureId === item.id}
                  onClick={() => setValue("architectureId", item.id)}
                >
                  <span className="block">{item.label}</span>
                  <span className="mt-1 block text-xs font-normal opacity-80">
                    {item.family}
                  </span>
                </ControlButton>
              ))}
            </div>
          </div>

          <div className="grid gap-5 lg:grid-cols-2">
            <VariantGroup
              label="3. Representation"
              activeId={input.representationId}
              items={representationVariants}
              onChange={(id) => setValue("representationId", id)}
            />
            <VariantGroup
              label="4. Extraction"
              activeId={input.extractionId}
              items={extractionVariants}
              onChange={(id) => setValue("extractionId", id)}
            />
            <VariantGroup
              label="5. Retrieval"
              activeId={input.retrievalId}
              items={retrievalVariants}
              onChange={(id) => setValue("retrievalId", id)}
            />
            <VariantGroup
              label="6. Maintenance"
              activeId={input.maintenanceId}
              items={maintenanceVariants}
              onChange={(id) => setValue("maintenanceId", id)}
            />
          </div>
        </div>

        <div className="min-w-0 space-y-4">
          <div
            role="status"
            className="rounded-lg border border-emerald-300 bg-emerald-50 p-5"
          >
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-sm font-semibold uppercase tracking-wide text-emerald-800">
                Workload fit
              </p>
              <span className="rounded-md border border-emerald-700 bg-white px-3 py-2 text-sm font-bold text-emerald-950">
                {result.score}/100
              </span>
            </div>
            <h3 className="mt-4 text-2xl font-semibold text-slate-950">
              {result.verdict}
            </h3>
            <p className="mt-3 text-sm leading-6 text-slate-700">
              {result.diagnosis}
            </p>
          </div>

          <div className="rounded-lg border border-slate-300 bg-slate-50 p-5">
            <p className="text-sm font-semibold uppercase tracking-wide text-slate-600">
              Current system
            </p>
            <div className="mt-4 space-y-3 text-sm leading-6 text-slate-700">
              <p>
                <strong className="text-slate-950">{workload.label}:</strong>{" "}
                {workload.learnerTask}
              </p>
              <p>
                <strong className="text-slate-950">
                  {architecture.label}:
                </strong>{" "}
                {architecture.summary}
              </p>
              <p>
                <strong className="text-slate-950">Component choices:</strong>{" "}
                {representation.label}, {extraction.label}, {retrieval.label},{" "}
                {maintenance.label}.
              </p>
              <p>
                <strong className="text-slate-950">Metric lens:</strong>{" "}
                {workload.successMetric}
              </p>
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            <MetricPill
              label="Evidence"
              value={result.metrics.evidence}
              tone="blue"
            />
            <MetricPill
              label="Exactness"
              value={result.metrics.exactness}
              tone="green"
            />
            <MetricPill
              label="Update"
              value={result.metrics.update}
              tone="violet"
            />
            <MetricPill
              label="Horizon"
              value={result.metrics.horizon}
              tone="amber"
            />
            <MetricPill label="Cost" value={result.metrics.cost} tone="rose" />
            <MetricPill
              label="Trace"
              value={result.metrics.trace}
              tone="blue"
            />
          </div>

          <div className="rounded-lg border border-blue-200 bg-blue-50 p-5">
            <p className="text-sm font-semibold text-blue-950">
              Findings activated
            </p>
            <ul className="mt-3 space-y-2 text-sm leading-6 text-blue-950">
              {result.findings.slice(0, 4).map((finding) => (
                <li key={finding} className="flex gap-2">
                  <CheckCircle2
                    aria-hidden="true"
                    size={16}
                    className="mt-1 shrink-0"
                  />
                  <span>{finding}</span>
                </li>
              ))}
            </ul>
          </div>

          {result.warnings.length > 0 && (
            <div className="rounded-lg border border-amber-300 bg-amber-50 p-5">
              <p className="flex items-center gap-2 text-sm font-semibold text-amber-950">
                <TriangleAlert aria-hidden="true" size={18} />
                Tradeoffs to manage
              </p>
              <ul className="mt-3 space-y-2 text-sm leading-6 text-amber-950">
                {result.warnings.slice(0, 4).map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function VariantGroup<TId extends string>({
  label,
  activeId,
  items,
  onChange,
}: {
  label: string;
  activeId: TId;
  items: readonly { id: TId; label: string; lesson: string }[];
  onChange: (id: TId) => void;
}) {
  return (
    <div>
      <p className="text-sm font-semibold uppercase tracking-wide text-blue-700">
        {label}
      </p>
      <div className="mt-3 grid gap-2">
        {items.map((item) => (
          <ControlButton
            key={item.id}
            isActive={activeId === item.id}
            onClick={() => onChange(item.id)}
          >
            <span className="block">{item.label}</span>
            <span className="mt-1 block text-xs font-normal opacity-80">
              {item.lesson}
            </span>
          </ControlButton>
        ))}
      </div>
    </div>
  );
}

function EvidenceDistanceLab() {
  const [architectureId, setArchitectureId] =
    useState<ArchitectureId>("hierarchical-tree");
  const [retrievalId, setRetrievalId] =
    useState<RetrievalVariantId>("balanced-hybrid");

  const projection = useMemo(
    () => getEvidenceDistanceProjection({ architectureId, retrievalId }),
    [architectureId, retrievalId],
  );
  const architecture = getArchitecture(architectureId);
  const retrieval = getRetrievalVariant(retrievalId);

  return (
    <div
      data-testid="agent-native-evidence-distance-lab"
      className="mt-8 rounded-lg border border-slate-300 bg-white p-5"
    >
      <div className="grid gap-6 lg:grid-cols-[18rem_1fr]">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-blue-700">
            Evidence distance lab
          </p>
          <h3 className="mt-2 text-2xl font-semibold text-slate-950">
            Watch recall decay as support moves away
          </h3>
          <p className="mt-3 text-sm leading-6 text-slate-700">
            The projection is a teaching model of the paper&apos;s Recall@10
            trend: short-range evidence can survive flat retrieval, but distant
            support needs links, hierarchy, planning, or hybrid assembly.
          </p>

          <div className="mt-5 space-y-5">
            <div>
              <p className="text-sm font-semibold text-slate-700">
                Architecture
              </p>
              <div className="mt-2 grid gap-2">
                {architectures
                  .filter((item) =>
                    [
                      "flat-similarity",
                      "hierarchical-tree",
                      "relation-graph",
                      "hybrid-filtered",
                    ].includes(item.id),
                  )
                  .map((item) => (
                    <ControlButton
                      key={item.id}
                      isActive={architectureId === item.id}
                      onClick={() => setArchitectureId(item.id)}
                    >
                      {item.label}
                    </ControlButton>
                  ))}
              </div>
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-700">
                Retrieval route
              </p>
              <div className="mt-2 grid gap-2">
                {retrievalVariants
                  .filter((item) =>
                    ["direct-top1", "planned", "balanced-hybrid"].includes(
                      item.id,
                    ),
                  )
                  .map((item) => (
                    <ControlButton
                      key={item.id}
                      isActive={retrievalId === item.id}
                      onClick={() => setRetrievalId(item.id)}
                    >
                      {item.label}
                    </ControlButton>
                  ))}
              </div>
            </div>
          </div>
        </div>

        <div className="min-w-0 rounded-lg border border-slate-300 bg-slate-50 p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-sm font-semibold uppercase tracking-wide text-slate-600">
                Recall@10 projection
              </p>
              <h4 className="mt-1 text-xl font-semibold text-slate-950">
                {architecture.shortLabel} with {retrieval.label}
              </h4>
            </div>
            <span className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-700">
              {architecture.family}
            </span>
          </div>

          <div className="mt-6 grid gap-3">
            {projection.map((item) => (
              <div
                key={item.bin}
                className="grid items-center gap-3 rounded-md border border-slate-200 bg-white p-3 sm:grid-cols-[5rem_1fr_7rem]"
              >
                <div>
                  <p className="text-sm font-bold text-slate-950">{item.bin}</p>
                  <p className="text-xs text-slate-500">sessions</p>
                </div>
                <div className="min-w-0">
                  <div className="h-3 rounded-full bg-slate-200">
                    <div
                      className="h-3 rounded-full bg-gradient-to-r from-blue-600 to-emerald-500"
                      style={{ width: `${item.recall}%` }}
                    />
                  </div>
                  <p className="mt-1 text-xs text-slate-500">{item.note}</p>
                </div>
                <p className="text-right text-sm font-semibold text-slate-700">
                  {item.recall}%
                </p>
              </div>
            ))}
          </div>

          <p className="mt-5 rounded-lg border border-blue-200 bg-blue-50 p-4 text-sm leading-6 text-blue-950">
            This is the paper&apos;s retrieval lesson in one picture: early
            localization and evidence assembly are separate design targets.
          </p>
        </div>
      </div>
    </div>
  );
}

function AblationBoard() {
  const [active, setActive] = useState<
    "representation" | "extraction" | "retrieval" | "maintenance"
  >("representation");

  const rows = {
    representation: [
      ["User-only raw", "Best all-around factual fidelity"],
      ["User-only compressed", "Close on LoCoMo, weaker on LongMemEval"],
      ["User-only summary", "Large drop from lossy abstraction"],
      ["Deeper tree", "Small navigation gain, not a missing-detail fix"],
    ],
    extraction: [
      ["Heuristic topic", "Better multi-session retrieval than LLM topic"],
      [
        "Fast memorize",
        "Much stronger LoCoMo answerability than fine memorize",
      ],
      ["Hybrid raw", "Preserves useful assistant clarifications"],
      ["Fine memorize", "Can trade lexical retrieval for reasoning loss"],
    ],
    retrieval: [
      ["Hybrid balanced", "Best A-MEM variant in the reported comparison"],
      ["Sparse leaning", "Keyword emphasis loses some relevance"],
      ["Planning only", "Best SimpleMem routing variant"],
      [
        "Planning plus reflect",
        "Extra reflection adds little and can weaken routing",
      ],
    ],
    maintenance: [
      ["Conservative merge", "Best default maintenance direction"],
      ["Delayed flush", "Looks broad but leaves evidence unresolved"],
      ["Single-topic summary", "Too coarse for sparse useful cues"],
      ["Raw context", "Still protects exact phrasing"],
    ],
  } satisfies Record<string, readonly (readonly [string, string])[]>;

  return (
    <div
      data-testid="agent-native-ablation-board"
      className="mt-8 rounded-lg border border-slate-300 bg-white p-5"
    >
      <div className="grid gap-6 lg:grid-cols-[18rem_1fr]">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-blue-700">
            Component ablations
          </p>
          <h3 className="mt-2 text-2xl font-semibold text-slate-950">
            Change one module, then watch the failure mode move
          </h3>
          <p className="mt-3 text-sm leading-6 text-slate-700">
            The paper does not stop at system leaderboards. It changes one
            component at a time to reveal why a memory design works or fails.
          </p>
          <div className="mt-5 grid gap-2">
            {(
              [
                ["representation", "Representation"],
                ["extraction", "Extraction"],
                ["retrieval", "Retrieval"],
                ["maintenance", "Maintenance"],
              ] as const
            ).map(([id, label]) => (
              <ControlButton
                key={id}
                isActive={active === id}
                onClick={() => setActive(id)}
              >
                {label}
              </ControlButton>
            ))}
          </div>
        </div>

        <div className="rounded-lg border border-violet-200 bg-violet-50 p-5">
          <div className="grid gap-3 md:grid-cols-2">
            {rows[active].map(([label, body]) => (
              <div
                key={label}
                className="rounded-lg border border-violet-200 bg-white p-4"
              >
                <p className="text-sm font-semibold text-violet-950">{label}</p>
                <p className="mt-2 text-sm leading-6 text-slate-700">{body}</p>
              </div>
            ))}
          </div>
          <p className="mt-5 rounded-lg border border-violet-200 bg-white p-4 text-sm leading-6 text-violet-950">
            Rule of thumb: preserve usable evidence first, then add structure,
            planning, and consolidation where they solve a specific bottleneck.
          </p>
        </div>
      </div>
    </div>
  );
}

export default function AgentNativeMemoryLearningPage({ experience }: Props) {
  return (
    <main className="min-h-screen bg-white text-slate-950">
      <PageBand tone="ink">
        <PageInner className="grid min-h-[560px] items-center gap-8 lg:grid-cols-[1fr_0.94fr]">
          <div className="min-w-0">
            <p className="text-sm font-semibold uppercase tracking-wide text-sky-300">
              AI agents / agent-native memory
            </p>
            <h1 className="mt-3 max-w-3xl text-4xl font-semibold tracking-normal text-white md:text-6xl md:leading-tight">
              Evaluate memory like infrastructure before trusting it as recall
            </h1>
            <p className="mt-5 max-w-2xl text-base leading-7 text-zinc-300 md:text-lg">
              This paper reframes agent memory as a data-management system. The
              learning job is to inspect the four modules, choose the workload
              bottleneck, and diagnose why a memory system succeeds, degrades,
              or becomes too expensive.
            </p>
            <div className="mt-6 flex flex-wrap items-center gap-3">
              <QuizTransitionButton
                sourceId={experience.sourceId}
                label="Start agent-native memory questions"
              />
              <span className="rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm font-semibold text-zinc-300">
                {experience.durationMinutes} min systems lab
              </span>
            </div>

            <div className="mt-8 grid max-w-2xl gap-3 sm:grid-cols-3">
              {[
                "Decompose memory into R, S, Q, U.",
                "Match architectures to workload bottlenecks.",
                "Separate evidence completion from top-1 recall.",
              ].map((item) => (
                <div
                  key={item}
                  className="rounded-lg border border-zinc-700 bg-zinc-900 p-3 text-sm leading-6 text-zinc-200"
                >
                  {item}
                </div>
              ))}
            </div>
          </div>

          <HeroConsole />
        </PageInner>
      </PageBand>

      <PageBand tone="blue">
        <PageInner>
          <SectionIntro
            eyebrow="Concept floor"
            title="The paper's object is a managed memory system"
            body="A memory mechanism is not enough. The paper treats agent memory as a persistent, updatable system with representation, extraction, retrieval, and maintenance modules that must survive real workloads."
          />
          <div className="mt-8 grid gap-5 lg:grid-cols-[1fr_0.85fr]">
            <div className="rounded-lg border border-slate-300 bg-white p-5">
              <h3 className="text-xl font-semibold text-slate-950">
                The system tuple
              </h3>
              <p className="mt-3 text-sm leading-6 text-slate-700">
                The paper formalizes the memory system as four modules. A
                learner should read every result by asking which module carried
                the evidence, where it lost information, and how expensive that
                choice became.
              </p>
              <MathText
                text={String.raw`\[\mathcal{M}_{sys}=\langle R,S,Q,U\rangle\]`}
                className="mt-5 min-w-0 max-w-full overflow-x-auto overflow-y-hidden rounded-lg border border-slate-300 bg-slate-50 px-4 py-3 text-slate-950 [&_.katex-mathml]:hidden"
              />
            </div>
            <div className="rounded-lg border border-slate-300 bg-white p-5">
              <h3 className="text-xl font-semibold text-slate-950">
                Why this differs from ordinary databases
              </h3>
              <div className="mt-4 space-y-3 text-sm leading-6 text-slate-700">
                <p>
                  Queries are semantic, partial, and sometimes latent, not only
                  exact predicates.
                </p>
                <p>
                  Memory evolves under uncertain and conflicting observations
                  rather than clean tuple overwrites.
                </p>
                <p>
                  One workload can mix factual lookup, temporal reasoning,
                  long-context synthesis, and streaming updates.
                </p>
              </div>
            </div>
          </div>
          <ModuleRack />
        </PageInner>
      </PageBand>

      <PageBand tone="grid">
        <PageInner>
          <SectionIntro
            eyebrow="Benchmark findings"
            title="A good memory design starts from the bottleneck"
            body="The headline result is not a winner table. It is a design rule: choose structure, routing, retention, and maintenance according to the workload pressure."
          />
          <FindingStrip />
        </PageInner>
      </PageBand>

      <PageBand>
        <PageInner>
          <SectionIntro
            eyebrow="Evaluation console"
            title="Stress a memory architecture before trusting the answer"
            body="Use the controls in order. Pick the benchmark pressure, choose an architecture family, and then tune the component variants. The diagnosis exposes why a memory result is strong, brittle, or too costly."
          />
          <EvaluationConsole />
        </PageInner>
      </PageBand>

      <PageBand tone="blue">
        <PageInner>
          <SectionIntro
            eyebrow="Retrieval fidelity"
            title="Retrieval is evidence assembly, not only ranking one item"
            body="The paper's retrieval analysis separates early localization from complete support. Distant evidence needs organization that can gather complementary fragments."
          />
          <EvidenceDistanceLab />
        </PageInner>
      </PageBand>

      <PageBand>
        <PageInner>
          <SectionIntro
            eyebrow="Component ablations"
            title="Small module changes explain large behavior changes"
            body="Representation, extraction, retrieval, and maintenance each have their own failure mode. The ablations teach a practical rule: preserve evidence first, then add just enough structure and lifecycle control."
          />
          <AblationBoard />
        </PageInner>
      </PageBand>

      <PageBand tone="ink">
        <PageInner>
          <div className="grid gap-8 lg:grid-cols-[1fr_0.85fr]">
            <div>
              <p className="text-sm font-semibold uppercase tracking-wide text-sky-300">
                Recap
              </p>
              <h2 className="mt-2 text-3xl font-semibold tracking-normal text-white md:text-4xl">
                Before practice, explain the failure mode in system terms
              </h2>
              <p className="mt-4 max-w-3xl text-base leading-7 text-zinc-300">
                A memory-backed answer can fail because representation removed
                the evidence, extraction filtered too early, retrieval found one
                item but not the support set, maintenance preserved stale state,
                or cost forced a shortcut. That is the mental model the quiz
                expects.
              </p>
            </div>
            <div className="rounded-lg border border-zinc-700 bg-zinc-900 p-5">
              <h3 className="text-xl font-semibold text-white">
                Practice target
              </h3>
              <ul className="mt-4 space-y-3 text-sm leading-6 text-zinc-300">
                {experience.outcomes.map((outcome) => (
                  <li key={outcome} className="flex gap-2">
                    <ArrowRight
                      aria-hidden="true"
                      size={16}
                      className="mt-1 shrink-0 text-sky-300"
                    />
                    <span>{outcome}</span>
                  </li>
                ))}
              </ul>
              <div className="mt-6">
                <QuizTransitionButton
                  sourceId={experience.sourceId}
                  label="Start agent-native memory questions"
                />
              </div>
            </div>
          </div>
        </PageInner>
      </PageBand>
    </main>
  );
}
