"use client";

import { useMemo, useState, type ReactNode } from "react";
import {
  BrainCircuit,
  Database,
  GitBranch,
  Layers3,
  Network,
  Route,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Timer,
  Workflow,
} from "lucide-react";
import type { LearningExperience } from "../../../lib/learning";
import MathText from "../../MathText";
import { QuizTransitionButton } from "../LearningPrimitives";
import {
  atomMemMemoryFacts,
  defaultAtomMemPipelineInput,
  evaluateAtomMemPipeline,
  getQueryMode,
  getRepresentationMode,
  getStructureMode,
  getVerificationMode,
  queryModes,
  representationModes,
  structureModes,
  verificationModes,
  type AtomMemMetricKey,
  type AtomMemPipelineInput,
  type FactCount,
  type FusionWeight,
  type QueryMode,
  type RepresentationMode,
  type SeedCount,
  type StructureMode,
  type VerificationMode,
} from "./atommem/workbench";

type Props = {
  experience: LearningExperience;
};

type AblationId = "flat" | "no-profile" | "no-graph" | "full";

const factCounts = [5, 10, 20, 40] as const satisfies readonly FactCount[];
const seedCounts = [5, 10, 20] as const satisfies readonly SeedCount[];
const fusionWeights = [
  0.5, 0.7, 0.9,
] as const satisfies readonly FusionWeight[];

const metricLabels: Record<AtomMemMetricKey, string> = {
  evidence: "recoverable evidence",
  stability: "update stability",
  association: "associative recall",
  efficiency: "context efficiency",
};

const katexScrollClass =
  "[&_.katex]:text-[0.82em] sm:[&_.katex]:text-[1em] [&_.katex-display]:max-w-full [&_.katex-display]:overflow-x-auto [&_.katex-display]:overflow-y-hidden [&_.katex-display]:py-1 [&_.katex-mathml]:hidden";

const ablations: Record<
  AblationId,
  {
    label: string;
    headline: string;
    result: string;
    lesson: string;
  }
> = {
  flat: {
    label: "AtomMem-Flat",
    headline: "Atomic facts alone are already a strong representation.",
    result:
      "The flat variant removes hierarchy but still beats raw LoCoMo history on multi-hop F1 while using the lowest token count in the comparison.",
    lesson:
      "Storage quality is not a minor detail. A clever orchestrator cannot recover evidence that the memory representation already lost or buried in noise.",
  },
  "no-profile": {
    label: "w/o Profile",
    headline: "Stable user state is not optional for personalization.",
    result:
      "Removing profile memory drops single-hop performance because some short questions still depend on persistent attributes and their history.",
    lesson:
      "Facts and events can preserve episodes, but temporal profiles track durable preferences and changed user state.",
  },
  "no-graph": {
    label: "w/o Graph",
    headline: "Isolated facts miss remote dependencies.",
    result:
      "Removing graph recall hurts complex reasoning, especially when the answer needs clues spread across entities, events, and neighboring turns.",
    lesson:
      "The graph is not decorative. It is the mechanism that lets seed facts activate associated evidence without dumping every memory into context.",
  },
  full: {
    label: "Full AtomMem",
    headline:
      "The full system coordinates representation, structure, and recall.",
    result:
      "The full architecture leads LoCoMo metrics across single-hop, multi-hop, temporal, and open-domain categories in the reported comparison.",
    lesson:
      "AtomMem's claim is modular: atomic facts provide the base, event/profile layers add structure, and graph recall connects scattered evidence.",
  },
};

function PageBand({
  children,
  tone = "paper",
}: {
  children: ReactNode;
  tone?: "paper" | "white" | "mint" | "ink";
}) {
  const tones = {
    paper: "bg-stone-50 text-zinc-950",
    white: "bg-white text-zinc-950",
    mint: "bg-emerald-50 text-zinc-950",
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
      <p className="text-sm font-semibold uppercase tracking-wide text-emerald-700">
        {eyebrow}
      </p>
      <h2 className="mt-2 text-3xl font-semibold tracking-normal text-zinc-950 md:text-4xl">
        {title}
      </h2>
      <p className="mt-4 text-base leading-7 text-zinc-700 md:text-lg">
        {body}
      </p>
    </div>
  );
}

function ControlButton({
  children,
  isActive,
  onClick,
  tone = "emerald",
}: {
  children: ReactNode;
  isActive: boolean;
  onClick: () => void;
  tone?: "emerald" | "violet" | "amber" | "rose";
}) {
  const activeTones = {
    emerald: "border-emerald-700 bg-emerald-700 text-white",
    violet: "border-violet-700 bg-violet-700 text-white",
    amber: "border-amber-700 bg-amber-400 text-zinc-950",
    rose: "border-rose-700 bg-rose-700 text-white",
  };

  return (
    <button
      type="button"
      aria-pressed={isActive}
      onClick={onClick}
      className={`min-h-11 rounded-md border px-3 py-2 text-left text-sm font-semibold leading-5 transition-colors ${
        isActive
          ? activeTones[tone]
          : "border-zinc-300 bg-white text-zinc-800 hover:border-zinc-500"
      }`}
    >
      {children}
    </button>
  );
}

function HeroPipelineVisual() {
  const stages = [
    {
      label: "Raw turns",
      detail: "noisy dialogue",
      icon: <Database aria-hidden="true" size={20} />,
      tone: "border-amber-300 bg-amber-50 text-amber-950",
    },
    {
      label: "Atomic facts",
      detail: "standalone evidence",
      icon: <Workflow aria-hidden="true" size={20} />,
      tone: "border-emerald-300 bg-emerald-50 text-emerald-950",
    },
    {
      label: "Events + profiles",
      detail: "context and user state",
      icon: <Layers3 aria-hidden="true" size={20} />,
      tone: "border-violet-300 bg-violet-50 text-violet-950",
    },
    {
      label: "Graph recall",
      detail: "associated evidence",
      icon: <Network aria-hidden="true" size={20} />,
      tone: "border-sky-300 bg-sky-50 text-sky-950",
    },
  ];

  return (
    <div
      aria-label="AtomMem pipeline from raw turns to graph recall"
      className="min-w-0 w-full rounded-lg border border-zinc-200 bg-white p-4 shadow-xl"
    >
      <div className="grid grid-cols-1 gap-3">
        {stages.map((stage, index) => (
          <div
            key={stage.label}
            className="grid grid-cols-1 gap-3 sm:grid-cols-[1fr_auto]"
          >
            <div className={`rounded-lg border p-4 ${stage.tone}`}>
              <div className="flex items-center gap-3">
                {stage.icon}
                <div>
                  <p className="text-sm font-semibold uppercase tracking-wide">
                    {stage.label}
                  </p>
                  <p className="mt-1 text-sm">{stage.detail}</p>
                </div>
              </div>
            </div>
            {index < stages.length - 1 && (
              <div className="hidden items-center justify-center text-zinc-400 sm:flex">
                <Route aria-hidden="true" size={22} />
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="mt-4 rounded-lg border border-zinc-200 bg-zinc-50 p-4">
        <p className="text-sm font-semibold text-zinc-900">
          The pipeline is the lesson: representation, update control, structure,
          and recall are separate failure points.
        </p>
        <MathText
          text={String.raw`\[S_h(x,y)=\alpha\,\mathrm{sim}_e(v_x,v_y)+\beta\,\mathrm{Jac}(K_x,K_y)\]`}
          className={`mt-3 min-w-0 max-w-full overflow-x-auto overflow-y-hidden rounded-md border border-zinc-200 bg-white px-3 py-2 text-zinc-950 ${katexScrollClass}`}
        />
      </div>
    </div>
  );
}

function MetricBar({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone: string;
}) {
  return (
    <div className="min-w-0">
      <div className="flex items-center justify-between gap-3 text-sm">
        <span className="font-semibold text-zinc-700">{label}</span>
        <span className="text-zinc-500">{value}/5</span>
      </div>
      <div className="mt-2 h-2 rounded-full bg-zinc-200">
        <div
          className={`h-2 rounded-full ${tone}`}
          style={{ width: `${Math.max(10, value * 20)}%` }}
        />
      </div>
    </div>
  );
}

function RepresentationLab() {
  const [mode, setMode] = useState<RepresentationMode>("atomic-facts");
  const selected = getRepresentationMode(mode);
  const evaluation = evaluateAtomMemPipeline({
    ...defaultAtomMemPipelineInput,
    representation: mode,
  });

  return (
    <div
      data-testid="atommem-representation-lab"
      className="mt-8 grid grid-cols-1 gap-5 lg:grid-cols-[20rem_1fr]"
    >
      <div className="rounded-lg border border-zinc-200 bg-white p-5">
        <p className="text-sm font-semibold uppercase tracking-wide text-emerald-700">
          Representation lab
        </p>
        <h3 className="mt-2 text-2xl font-semibold text-zinc-950">
          Choose what enters memory
        </h3>
        <p className="mt-3 text-sm leading-6 text-zinc-700">
          The paper first claims that memory quality starts before retrieval. A
          fact should be high-value, standalone, and metadata-bearing.
        </p>
        <div className="mt-5 grid gap-2">
          {representationModes.map((item) => (
            <ControlButton
              key={item.id}
              isActive={mode === item.id}
              onClick={() => setMode(item.id)}
            >
              <span className="block">{item.label}</span>
              <span className="mt-1 block text-xs font-normal opacity-80">
                {item.caption}
              </span>
            </ControlButton>
          ))}
        </div>
      </div>

      <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-5">
        <div role="status" className="rounded-lg bg-white p-4">
          <p className="text-sm font-semibold uppercase tracking-wide text-emerald-700">
            {selected.label}
          </p>
          <h4 className="mt-2 text-2xl font-semibold text-zinc-950">
            {evaluation.label}
          </h4>
          <p className="mt-3 text-sm leading-6 text-zinc-700">
            {selected.description}
          </p>
          <p className="mt-3 text-sm leading-6 text-zinc-700">
            {evaluation.status}
          </p>
          {evaluation.warnings.length > 0 && (
            <ul className="mt-3 space-y-1 text-sm leading-6 text-amber-900">
              {evaluation.warnings.slice(0, 2).map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          )}
        </div>

        <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-3">
          {atomMemMemoryFacts.slice(0, 3).map((fact) => (
            <article
              key={fact.id}
              className="rounded-lg border border-emerald-200 bg-white p-4"
            >
              <p className="text-sm font-semibold text-emerald-800">
                {fact.label}
              </p>
              <p className="mt-2 text-sm leading-6 text-zinc-700">
                {mode === "raw-log" ? fact.rawTurn : fact.atomicText}
              </p>
              {mode === "atomic-facts" && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {fact.keywords.slice(0, 3).map((keyword) => (
                    <span
                      key={keyword}
                      className="rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1 text-xs font-semibold text-emerald-900"
                    >
                      {keyword}
                    </span>
                  ))}
                </div>
              )}
            </article>
          ))}
        </div>
      </div>
    </div>
  );
}

function PipelineDebugger() {
  const [input, setInput] = useState<AtomMemPipelineInput>(
    defaultAtomMemPipelineInput,
  );
  const evaluation = useMemo(() => evaluateAtomMemPipeline(input), [input]);
  const query = getQueryMode(input.query);
  const representation = getRepresentationMode(input.representation);
  const verification = getVerificationMode(input.verification);
  const structure = getStructureMode(input.structure);

  const setPartial = (partial: Partial<AtomMemPipelineInput>) =>
    setInput((current) => ({ ...current, ...partial }));

  return (
    <div
      data-testid="atommem-pipeline-debugger"
      className="mt-8 rounded-lg border border-zinc-200 bg-white p-5"
    >
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[23rem_1fr_21rem]">
        <div className="space-y-5">
          <div>
            <p className="text-sm font-semibold uppercase tracking-wide text-emerald-700">
              1. Query pressure
            </p>
            <div className="mt-3 grid gap-2">
              {queryModes.map((item) => (
                <ControlButton
                  key={item.id}
                  isActive={input.query === item.id}
                  onClick={() => setPartial({ query: item.id as QueryMode })}
                >
                  <span className="block">{item.label}</span>
                  <span className="mt-1 block text-xs font-normal opacity-80">
                    {item.question}
                  </span>
                </ControlButton>
              ))}
            </div>
          </div>

          <div>
            <p className="text-sm font-semibold uppercase tracking-wide text-emerald-700">
              2. Representation
            </p>
            <div className="mt-3 grid gap-2">
              {representationModes.map((item) => (
                <ControlButton
                  key={item.id}
                  isActive={input.representation === item.id}
                  onClick={() =>
                    setPartial({
                      representation: item.id as RepresentationMode,
                    })
                  }
                >
                  {item.label}
                </ControlButton>
              ))}
            </div>
          </div>

          <div>
            <p className="text-sm font-semibold uppercase tracking-wide text-emerald-700">
              3. Verification
            </p>
            <div className="mt-3 grid gap-2">
              {verificationModes.map((item) => (
                <ControlButton
                  key={item.id}
                  isActive={input.verification === item.id}
                  onClick={() =>
                    setPartial({ verification: item.id as VerificationMode })
                  }
                  tone="violet"
                >
                  {item.label}
                </ControlButton>
              ))}
            </div>
          </div>
        </div>

        <div className="min-w-0 space-y-5">
          <div className="grid grid-cols-1 gap-4 rounded-lg border border-zinc-200 bg-zinc-50 p-5 md:grid-cols-3">
            {[
              {
                title: "Fact",
                icon: <Workflow aria-hidden="true" size={20} />,
                body: representation.description,
              },
              {
                title: "Verification",
                icon: <ShieldCheck aria-hidden="true" size={20} />,
                body: verification.description,
              },
              {
                title: "Structure",
                icon: <Layers3 aria-hidden="true" size={20} />,
                body: structure.description,
              },
            ].map((item) => (
              <article
                key={item.title}
                className="rounded-lg border border-zinc-200 bg-white p-4"
              >
                <div className="flex items-center gap-2 text-emerald-700">
                  {item.icon}
                  <p className="text-sm font-semibold uppercase tracking-wide">
                    {item.title}
                  </p>
                </div>
                <p className="mt-3 text-sm leading-6 text-zinc-700">
                  {item.body}
                </p>
              </article>
            ))}
          </div>

          <div className="rounded-lg border border-violet-200 bg-violet-50 p-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="text-sm font-semibold uppercase tracking-wide text-violet-800">
                  Retrieval diagnosis
                </p>
                <h4 className="mt-2 text-2xl font-semibold text-zinc-950">
                  {evaluation.label}
                </h4>
              </div>
              <span className="rounded-lg border border-violet-700 bg-white px-4 py-3 text-lg font-bold text-violet-950">
                {evaluation.score}/100
              </span>
            </div>
            <p role="status" className="mt-4 text-sm leading-6 text-zinc-700">
              {evaluation.status}
            </p>
            <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-2">
              {Object.entries(evaluation.metrics).map(([key, value]) => (
                <MetricBar
                  key={key}
                  label={metricLabels[key as AtomMemMetricKey]}
                  value={value}
                  tone={
                    key === "stability"
                      ? "bg-violet-600"
                      : key === "efficiency"
                        ? "bg-amber-500"
                        : "bg-emerald-600"
                  }
                />
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div className="rounded-lg border border-zinc-200 bg-white p-5">
              <p className="text-sm font-semibold uppercase tracking-wide text-zinc-500">
                Retrieved context
              </p>
              <p className="mt-2 text-sm leading-6 text-zinc-700">
                Query: {query.question}
              </p>
              <div className="mt-4 space-y-3">
                {evaluation.retrievedFactIds.map((factId) => {
                  const fact = atomMemMemoryFacts.find(
                    (candidate) => candidate.id === factId,
                  );
                  if (!fact) return null;
                  return (
                    <div
                      key={fact.id}
                      className="rounded-md border border-zinc-200 bg-zinc-50 p-3"
                    >
                      <p className="text-sm font-semibold text-zinc-950">
                        {fact.label}
                      </p>
                      <p className="mt-1 text-sm leading-6 text-zinc-700">
                        {fact.atomicText}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="rounded-lg border border-zinc-200 bg-white p-5">
              <p className="text-sm font-semibold uppercase tracking-wide text-zinc-500">
                Warnings and strengths
              </p>
              <ul className="mt-4 space-y-2 text-sm leading-6 text-zinc-700">
                {[...evaluation.warnings, ...evaluation.strengths]
                  .slice(0, 6)
                  .map((item) => (
                    <li key={item}>{item}</li>
                  ))}
              </ul>
              <p className="mt-5 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm font-semibold text-amber-950">
                Estimated online memory overhead: {evaluation.latencyMs} ms
              </p>
            </div>
          </div>
        </div>

        <div className="space-y-5">
          <div>
            <p className="text-sm font-semibold uppercase tracking-wide text-emerald-700">
              4. Structure
            </p>
            <div className="mt-3 grid gap-2">
              {structureModes.map((item) => (
                <ControlButton
                  key={item.id}
                  isActive={input.structure === item.id}
                  onClick={() =>
                    setPartial({ structure: item.id as StructureMode })
                  }
                  tone="amber"
                >
                  {item.label}
                </ControlButton>
              ))}
            </div>
          </div>

          <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4">
            <p className="text-sm font-semibold uppercase tracking-wide text-zinc-500">
              5. Graph recall
            </p>
            <button
              type="button"
              aria-pressed={input.graphEnabled}
              onClick={() => setPartial({ graphEnabled: !input.graphEnabled })}
              className={`mt-3 min-h-11 w-full rounded-md border px-3 py-2 text-left text-sm font-semibold ${
                input.graphEnabled
                  ? "border-sky-700 bg-sky-700 text-white"
                  : "border-zinc-300 bg-white text-zinc-800"
              }`}
            >
              {input.graphEnabled ? "Graph recall enabled" : "Flat recall only"}
            </button>
          </div>

          <ControlGroup
            label="6. Seed facts"
            values={seedCounts}
            value={input.seedCount}
            renderValue={(value) => `${value}`}
            onChange={(value) => setPartial({ seedCount: value })}
          />
          <ControlGroup
            label="7. Event fusion"
            values={fusionWeights}
            value={input.fusionWeight}
            renderValue={(value) => `${value.toFixed(1)}`}
            onChange={(value) => setPartial({ fusionWeight: value })}
          />
          <ControlGroup
            label="8. Final facts"
            values={factCounts}
            value={input.factCount}
            renderValue={(value) => `${value}`}
            onChange={(value) => setPartial({ factCount: value })}
          />
        </div>
      </div>
    </div>
  );
}

function ControlGroup<TValue extends number>({
  label,
  values,
  value,
  renderValue,
  onChange,
}: {
  label: string;
  values: readonly TValue[];
  value: TValue;
  renderValue: (value: TValue) => string;
  onChange: (value: TValue) => void;
}) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4">
      <p className="text-sm font-semibold uppercase tracking-wide text-zinc-500">
        {label}
      </p>
      <div className="mt-3 grid grid-cols-3 gap-2">
        {values.map((item) => (
          <button
            key={item}
            type="button"
            aria-pressed={value === item}
            onClick={() => onChange(item)}
            className={`min-h-10 rounded-md border px-3 py-2 text-sm font-semibold ${
              value === item
                ? "border-zinc-950 bg-zinc-950 text-white"
                : "border-zinc-300 bg-white text-zinc-800 hover:border-zinc-500"
            }`}
          >
            {renderValue(item)}
          </button>
        ))}
      </div>
    </div>
  );
}

function GraphEquationStrip() {
  return (
    <div className="mt-8 grid grid-cols-1 gap-5 lg:grid-cols-3">
      {[
        {
          title: "Entity edge",
          icon: <Search aria-hidden="true" size={20} />,
          formula: String.raw`\[w_{\mathrm{kw}}(F_i,F_j)\propto \sum_{k\in K_i\cap K_j}\omega(k)\]`,
          body: "Shared keywords help, but query-aware weighting penalizes common words so the graph does not connect everything through generic terms.",
        },
        {
          title: "Event edge",
          icon: <GitBranch aria-hidden="true" size={20} />,
          formula: String.raw`\[w_{\mathrm{event}}\propto \frac{1}{(|F_e|-1)^{\gamma_e}}\]`,
          body: "Facts in the same event can recall each other, while large-event penalties prevent broad episodes from dominating.",
        },
        {
          title: "Random walk",
          icon: <Route aria-hidden="true" size={20} />,
          formula: String.raw`\[r^{(t+1)}=\eta p+(1-\eta)P^\top r^{(t)}\]`,
          body: "The restart distribution keeps recall anchored to seed facts while graph edges propagate activation to associated evidence.",
        },
      ].map((item) => (
        <article
          key={item.title}
          className="rounded-lg border border-zinc-200 bg-white p-5"
        >
          <div className="flex items-center gap-2 text-sky-700">
            {item.icon}
            <h3 className="text-lg font-semibold text-zinc-950">
              {item.title}
            </h3>
          </div>
          <MathText
            text={item.formula}
            className={`mt-4 min-w-0 max-w-full overflow-x-auto overflow-y-hidden rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 text-zinc-950 ${katexScrollClass}`}
          />
          <p className="mt-4 text-sm leading-6 text-zinc-700">{item.body}</p>
        </article>
      ))}
    </div>
  );
}

function AblationBoard() {
  const [activeId, setActiveId] = useState<AblationId>("full");
  const active = ablations[activeId];

  return (
    <div
      data-testid="atommem-ablation-board"
      className="mt-8 rounded-lg border border-zinc-200 bg-white p-5"
    >
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[18rem_1fr]">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-emerald-700">
            Ablation board
          </p>
          <h3 className="mt-2 text-2xl font-semibold text-zinc-950">
            Read the experiments as design evidence
          </h3>
          <p className="mt-3 text-sm leading-6 text-zinc-700">
            The ablations isolate what each component contributes: facts,
            profiles, graph recall, and the full coordinated stack.
          </p>
          <div className="mt-5 grid gap-2">
            {(Object.keys(ablations) as AblationId[]).map((id) => (
              <ControlButton
                key={id}
                isActive={activeId === id}
                onClick={() => setActiveId(id)}
                tone={id === "full" ? "emerald" : "rose"}
              >
                {ablations[id].label}
              </ControlButton>
            ))}
          </div>
        </div>

        <div className="rounded-lg border border-rose-200 bg-rose-50 p-5">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-2 rounded-md border border-rose-700 bg-white px-3 py-2 text-sm font-semibold text-rose-950">
              <BrainCircuit aria-hidden="true" size={18} />
              {active.label}
            </span>
          </div>
          <h4 className="mt-5 text-2xl font-semibold text-zinc-950">
            {active.headline}
          </h4>
          <div className="mt-5 grid grid-cols-1 gap-3 md:grid-cols-2">
            <div className="rounded-lg border border-rose-200 bg-white p-4">
              <p className="text-sm font-semibold text-rose-900">
                Reported result
              </p>
              <p className="mt-2 text-sm leading-6 text-zinc-700">
                {active.result}
              </p>
            </div>
            <div className="rounded-lg border border-rose-200 bg-white p-4">
              <p className="text-sm font-semibold text-rose-900">
                Design lesson
              </p>
              <p className="mt-2 text-sm leading-6 text-zinc-700">
                {active.lesson}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function EvaluationDashboard() {
  return (
    <div className="mt-8 grid grid-cols-1 gap-5 lg:grid-cols-[1fr_0.85fr]">
      <div className="rounded-lg border border-zinc-200 bg-white p-5">
        <div className="flex items-center gap-2 text-emerald-700">
          <Timer aria-hidden="true" size={20} />
          <h3 className="text-xl font-semibold text-zinc-950">
            What the benchmarks were testing
          </h3>
        </div>
        <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2">
          {[
            [
              "LoCoMo",
              "Long conversations averaging hundreds of turns across many sessions, with single-hop, multi-hop, temporal, and open-domain questions.",
            ],
            [
              "LongMemEval",
              "An independent check for single-session, multi-session, knowledge-update, temporal, assistant, and personalized preference memory.",
            ],
            [
              "Metrics",
              "F1 and BLEU-1 capture lexical overlap; judge scoring captures semantic correctness; token consumption captures cost.",
            ],
            [
              "Efficiency",
              "The retrieval pipeline is small compared with LLM intent parsing and answer generation, so structure is not the dominant online latency.",
            ],
          ].map(([title, body]) => (
            <div
              key={title}
              className="rounded-lg border border-zinc-200 bg-zinc-50 p-4"
            >
              <p className="text-sm font-semibold text-zinc-950">{title}</p>
              <p className="mt-2 text-sm leading-6 text-zinc-700">{body}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="rounded-lg border border-amber-200 bg-amber-50 p-5">
        <div className="flex items-center gap-2 text-amber-800">
          <SlidersHorizontal aria-hidden="true" size={20} />
          <h3 className="text-xl font-semibold text-zinc-950">
            Hyperparameter lesson
          </h3>
        </div>
        <p className="mt-4 text-sm leading-6 text-zinc-700">
          The reported AtomMem sweet spot is not maximum context. Ten seed facts
          and ten final facts keep graph activation broad enough to find remote
          evidence but narrow enough to suppress distractors.
        </p>
        <div className="mt-5 grid grid-cols-1 gap-3">
          {[
            "Too few seeds: under-activated graph.",
            "Too many seeds: unrelated entry points and noisy stationary probabilities.",
            "Event fusion near 0.7: enough episode context without losing fact precision.",
          ].map((item) => (
            <p
              key={item}
              className="rounded-md border border-amber-200 bg-white px-3 py-2 text-sm font-semibold text-amber-950"
            >
              {item}
            </p>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function AtomMemLearningPage({ experience }: Props) {
  return (
    <main className="min-h-screen bg-stone-50 text-zinc-950">
      <PageBand tone="paper">
        <PageInner className="grid min-h-[560px] grid-cols-1 items-center gap-8 lg:grid-cols-[1fr_0.95fr]">
          <div className="min-w-0">
            <p className="text-sm font-semibold uppercase tracking-wide text-emerald-700">
              AI agents / AtomMem
            </p>
            <h1 className="mt-3 max-w-4xl text-4xl font-semibold tracking-normal text-zinc-950 md:text-6xl md:leading-tight">
              Debug a memory pipeline before trusting its answer
            </h1>
            <p className="mt-5 max-w-2xl text-base leading-7 text-zinc-700 md:text-lg">
              AtomMem argues that long-term agent memory is a data-management
              pipeline. Raw conversations become atomic facts, facts become
              events and temporal profiles, and graph recall connects scattered
              evidence without flooding the model context.
            </p>
            <div className="mt-6 flex flex-wrap items-center gap-3">
              <QuizTransitionButton
                sourceId={experience.sourceId}
                label="Start AtomMem questions"
              />
              <span className="rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm font-semibold text-zinc-700">
                {experience.durationMinutes} min pipeline studio
              </span>
            </div>
            <div className="mt-6 grid max-w-2xl grid-cols-1 gap-3 sm:grid-cols-2">
              {experience.outcomes.slice(0, 4).map((outcome) => (
                <div
                  key={outcome}
                  className="rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm leading-6 text-zinc-700"
                >
                  {outcome}
                </div>
              ))}
            </div>
          </div>

          <HeroPipelineVisual />
        </PageInner>
      </PageBand>

      <PageBand tone="white">
        <PageInner>
          <SectionIntro
            eyebrow="Concept floor"
            title="Atomic facts are the minimum useful memory unit"
            body="Raw dialogue keeps detail but buries it in noise. Loose summaries reduce cost but can erase precise dates, exceptions, or conflicting evidence. AtomMem's base move is to extract standalone facts with embeddings, participants, keywords, time, and event links."
          />
          <div className="mt-8 grid grid-cols-1 gap-5 lg:grid-cols-[1fr_0.9fr]">
            <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-5">
              <h3 className="text-xl font-semibold text-zinc-950">
                The structured fact object
              </h3>
              <p className="mt-3 text-sm leading-6 text-zinc-700">
                Each fact has text and metadata. The text makes the fact
                readable. The embedding supports semantic retrieval. Keywords,
                participants, time, and event ids make it possible to filter,
                verify, and organize memory before the model answers.
              </p>
              <MathText
                text={String.raw`\[F=\{id,c,v,P,K,T,E\}\]`}
                className={`mt-5 min-w-0 max-w-full overflow-x-auto overflow-y-hidden rounded-lg border border-zinc-200 bg-white px-4 py-3 text-zinc-950 ${katexScrollClass}`}
              />
            </div>
            <div className="rounded-lg border border-zinc-200 bg-white p-5">
              <h3 className="text-xl font-semibold text-zinc-950">
                Verification is the stability gate
              </h3>
              <p className="mt-3 text-sm leading-6 text-zinc-700">
                AtomMem checks a new fact against relevant existing memory. It
                stores residual novel content and emits update tuples only when
                a conflict is real. This is the difference between stable memory
                evolution and uncontrolled rewrites.
              </p>
              <MathText
                text={String.raw`\[\left(c'_{\mathrm{new}},U\right)\leftarrow\mathrm{LLM}\left(c_{\mathrm{new}}\parallel C_{\mathrm{ret}}\right)\]`}
                className={`mt-5 min-w-0 max-w-full overflow-x-auto overflow-y-hidden rounded-lg border border-zinc-200 bg-zinc-50 px-4 py-3 text-zinc-950 ${katexScrollClass}`}
              />
            </div>
          </div>
          <RepresentationLab />
        </PageInner>
      </PageBand>

      <PageBand tone="mint">
        <PageInner>
          <SectionIntro
            eyebrow="Pipeline debugger"
            title="Run one memory design and watch where it fails"
            body="The controls follow AtomMem's architecture. Start with the query pressure, then choose representation, verification, structure, graph recall, seed count, event fusion, and final retrieved fact count."
          />
          <PipelineDebugger />
        </PageInner>
      </PageBand>

      <PageBand tone="white">
        <PageInner>
          <SectionIntro
            eyebrow="Associative recall"
            title="Graph recall connects evidence without dumping every fact"
            body="The memory graph has three channels: entity overlap, shared event membership, and nearby dialogue turns. Personalized restart keeps activation close to seed facts while allowing associated evidence to surface."
          />
          <GraphEquationStrip />
        </PageInner>
      </PageBand>

      <PageBand tone="paper">
        <PageInner>
          <SectionIntro
            eyebrow="Evaluation evidence"
            title="The experiments are component tests disguised as benchmark tables"
            body="AtomMem is not claiming that every part is always necessary. The paper's ablations and hyperparameters show when atomic representation, temporal profiles, graph recall, and bounded retrieval matter."
          />
          <AblationBoard />
          <EvaluationDashboard />
        </PageInner>
      </PageBand>

      <PageBand tone="ink">
        <PageInner>
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-[1fr_0.85fr]">
            <div>
              <p className="text-sm font-semibold uppercase tracking-wide text-emerald-300">
                Recap
              </p>
              <h2 className="mt-2 text-3xl font-semibold tracking-normal text-white md:text-4xl">
                Before the quiz, trace the evidence path
              </h2>
              <p className="mt-4 max-w-3xl text-base leading-7 text-zinc-300">
                A good AtomMem answer depends on each earlier stage: the fact
                extractor must preserve the right detail, verification must
                avoid duplicate or destructive updates, event/profile memory
                must keep context and user state, and graph recall must recover
                associated evidence without drifting into noise.
              </p>
            </div>
            <div className="rounded-lg border border-zinc-700 bg-zinc-900 p-5">
              <h3 className="text-xl font-semibold text-white">
                Practice target
              </h3>
              <ul className="mt-4 space-y-3 text-sm leading-6 text-zinc-300">
                {experience.outcomes.map((outcome) => (
                  <li key={outcome}>{outcome}</li>
                ))}
              </ul>
              <div className="mt-6">
                <QuizTransitionButton
                  sourceId={experience.sourceId}
                  label="Start AtomMem questions"
                />
              </div>
            </div>
          </div>
        </PageInner>
      </PageBand>
    </main>
  );
}
