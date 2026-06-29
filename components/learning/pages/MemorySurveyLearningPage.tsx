"use client";

import { useMemo, useState, type ReactNode } from "react";
import {
  BrainCircuit,
  Database,
  GitBranch,
  History,
  Layers3,
  Network,
  RefreshCw,
  Route,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Workflow,
} from "lucide-react";
import type { LearningExperience } from "../../../lib/learning";
import MathText from "../../MathText";
import { QuizTransitionButton } from "../LearningPrimitives";
import {
  evolutionOperations,
  formationOperations,
  getFormationOperation,
  getMemoryDesignEvaluation,
  getMemoryForm,
  getMemoryFunction,
  getMemoryScenario,
  getRetrievalOperation,
  memoryForms,
  memoryFunctions,
  memoryScenarios,
  retrievalOperations,
  type EvolutionId,
  type FormationId,
  type MemoryFormId,
  type MemoryFunctionId,
  type MemoryScenarioId,
  type RetrievalId,
} from "./memory-survey/workbench";

type Props = {
  experience: LearningExperience;
};

type BoundaryId = "agent-memory" | "llm-memory" | "rag" | "context";

type FrontierId =
  | "generation"
  | "automation"
  | "rl"
  | "multimodal"
  | "shared"
  | "world-model"
  | "trust";

const boundaryCases: Record<
  BoundaryId,
  {
    label: string;
    title: string;
    question: string;
    memorySurface: string;
    whatChanges: string;
    confusion: string;
    icon: ReactNode;
  }
> = {
  "agent-memory": {
    label: "Agent memory",
    title: "Persistent cognitive state for decisions",
    question:
      "What should this agent carry from prior interaction into future action?",
    memorySurface:
      "An evolving store plus formation, evolution, and retrieval operators.",
    whatChanges:
      "The store itself changes as the agent observes, reasons, acts, and receives feedback.",
    confusion:
      "It can use RAG, context engineering, model weights, or latent states, but the defining feature is managed continuity.",
    icon: <BrainCircuit aria-hidden="true" size={20} />,
  },
  "llm-memory": {
    label: "LLM memory",
    title: "Internal retention inside the model",
    question: "How does the model retain or process longer sequences?",
    memorySurface:
      "Architectural capacity such as KV caches, recurrent state, sparse attention, long-context mechanisms, or edited weights.",
    whatChanges:
      "Often the model computation changes, not an explicit agent-managed memory base.",
    confusion:
      "Some LLM-memory methods become agent memory when they support deliberate cross-step or cross-task memory operations.",
    icon: <Sparkles aria-hidden="true" size={20} />,
  },
  rag: {
    label: "Classical RAG",
    title: "Retrieve external knowledge for a task",
    question: "What source evidence should ground this response right now?",
    memorySurface:
      "A document index, vector store, graph, or other external knowledge base usually maintained outside the agent.",
    whatChanges:
      "The retrieved context changes by query; the underlying knowledge source may remain externally maintained.",
    confusion:
      "Dynamic or agentic RAG can blur into memory when the agent updates the store from its own actions and feedback.",
    icon: <Database aria-hidden="true" size={20} />,
  },
  context: {
    label: "Context engineering",
    title: "Assemble the best prompt interface",
    question:
      "What information should enter the next model call under a context budget?",
    memorySurface:
      "Instructions, retrieved snippets, summaries, tool state, scratchpads, and formatting decisions for the current call.",
    whatChanges:
      "The immediate payload changes; it may include memory but is not identical to the persistent memory system.",
    confusion:
      "Context engineering optimizes the momentary interface, while agent memory governs what continues across interaction.",
    icon: <SlidersHorizontal aria-hidden="true" size={20} />,
  },
};

const frontiers: Record<
  FrontierId,
  {
    label: string;
    title: string;
    lookBack: string;
    future: string;
    pressure: string;
    icon: ReactNode;
  }
> = {
  generation: {
    label: "Retrieval vs generation",
    title: "Memory may be generated, not only fetched",
    lookBack:
      "Most systems look backward by retrieving stored traces, summaries, facts, or cases.",
    future:
      "The paper frames a shift toward memory generation, where agents synthesize new reusable memory objects from experience.",
    pressure:
      "Generated memory needs checks against provenance and hallucinated self-knowledge.",
    icon: <RefreshCw aria-hidden="true" size={20} />,
  },
  automation: {
    label: "Automated management",
    title: "Hand-crafted stores will not scale",
    lookBack:
      "Many memory systems still depend on manually chosen structures, prompts, thresholds, and update rules.",
    future:
      "Future agents should learn when to form, merge, rewrite, forget, and retrieve memory with less human wiring.",
    pressure:
      "Automation without observability can hide why a memory was kept or discarded.",
    icon: <Workflow aria-hidden="true" size={20} />,
  },
  rl: {
    label: "RL for memory control",
    title: "Memory management becomes part of the policy",
    lookBack:
      "Reasoning and tool use have already been partly internalized through reinforcement learning.",
    future:
      "The same pressure may teach agents when memory operations improve long-horizon reward.",
    pressure:
      "A reward can push memory toward useful shortcuts or toward reward-hacking traces that look useful.",
    icon: <Route aria-hidden="true" size={20} />,
  },
  multimodal: {
    label: "Multimodal memory",
    title: "Memory is not only text",
    lookBack:
      "Token-level memory already includes visual tokens, audio frames, trajectories, and other modality-specific records.",
    future:
      "Agents will need memory that preserves cross-modal evidence, not just captions of it.",
    pressure:
      "Compression must not erase the visual, temporal, or spatial details that later decisions depend on.",
    icon: <Layers3 aria-hidden="true" size={20} />,
  },
  shared: {
    label: "Shared memory",
    title: "Multi-agent systems need a common substrate",
    lookBack:
      "Agents often keep isolated local histories or pass messages without a durable shared state.",
    future:
      "Shared memory can coordinate roles, document states, claims, tool results, and other agents' capabilities.",
    pressure:
      "Shared stores need ownership, conflict resolution, and trust boundaries.",
    icon: <Network aria-hidden="true" size={20} />,
  },
  "world-model": {
    label: "World model",
    title: "Memory can support predictive environment models",
    lookBack:
      "Environment factual memory stores entities, resources, state, tools, documents, and codebase facts.",
    future:
      "A stronger memory system could help agents predict how the world will change after action.",
    pressure:
      "World-model memory must distinguish observation, inference, and speculation.",
    icon: <GitBranch aria-hidden="true" size={20} />,
  },
  trust: {
    label: "Trustworthy memory",
    title: "Memory needs governance, not only recall",
    lookBack:
      "Trustworthy RAG focuses on source quality, retrieval errors, grounding, and output verification.",
    future:
      "Trustworthy memory adds persistence risks: stale facts, sensitive data, cross-session leakage, and hard-to-delete implicit state.",
    pressure:
      "The more hidden the memory form, the harder auditing, correction, consent, and deletion become.",
    icon: <ShieldCheck aria-hidden="true" size={20} />,
  },
};

const formFamilyTone: Record<string, string> = {
  "Token-level": "border-emerald-300 bg-emerald-50 text-emerald-950",
  Parametric: "border-amber-300 bg-amber-50 text-amber-950",
  Latent: "border-cyan-300 bg-cyan-50 text-cyan-950",
};

const functionFamilyTone: Record<string, string> = {
  Factual: "border-rose-300 bg-rose-50 text-rose-950",
  Experiential: "border-indigo-300 bg-indigo-50 text-indigo-950",
  Working: "border-teal-300 bg-teal-50 text-teal-950",
};

function InlineMath({ text }: { text: string }) {
  return <MathText inline text={text} />;
}

function PageBand({
  children,
  tone = "white",
}: {
  children: ReactNode;
  tone?: "white" | "warm" | "ink";
}) {
  const tones = {
    white: "bg-white text-slate-950",
    warm: "bg-stone-50 text-slate-950",
    ink: "bg-slate-950 text-slate-50",
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
      <p className="text-sm font-semibold uppercase tracking-wide text-teal-700">
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
  tone = "teal",
}: {
  children: ReactNode;
  isActive: boolean;
  onClick: () => void;
  tone?: "teal" | "amber" | "rose" | "indigo";
}) {
  const activeTones = {
    teal: "border-teal-700 bg-teal-700 text-white",
    amber: "border-amber-700 bg-amber-500 text-slate-950",
    rose: "border-rose-700 bg-rose-700 text-white",
    indigo: "border-indigo-700 bg-indigo-700 text-white",
  };

  return (
    <button
      type="button"
      aria-pressed={isActive}
      onClick={onClick}
      className={`min-h-11 rounded-md border px-3 py-2 text-left text-sm font-semibold leading-5 transition-colors ${
        isActive
          ? activeTones[tone]
          : "border-slate-300 bg-white text-slate-800 hover:border-slate-500"
      }`}
    >
      {children}
    </button>
  );
}

function MetricBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="min-w-0">
      <div className="flex items-center justify-between gap-3 text-sm">
        <span className="font-semibold text-slate-700">{label}</span>
        <span className="text-slate-500">{value}/5</span>
      </div>
      <div className="mt-2 h-2 rounded-full bg-slate-200">
        <div
          className="h-2 rounded-full bg-teal-600"
          style={{ width: `${Math.max(12, value * 20)}%` }}
        />
      </div>
    </div>
  );
}

function HeroMemoryMap() {
  return (
    <div
      aria-label="Forms functions dynamics memory map"
      className="min-w-0 rounded-lg border border-slate-700 bg-slate-900 p-4 shadow-2xl"
    >
      <div className="grid min-w-0 gap-3 sm:grid-cols-3">
        {[
          {
            label: "Forms",
            value: "token, parametric, latent",
            icon: <Layers3 aria-hidden="true" size={20} />,
          },
          {
            label: "Functions",
            value: "factual, experiential, working",
            icon: <BrainCircuit aria-hidden="true" size={20} />,
          },
          {
            label: "Dynamics",
            value: "form, evolve, retrieve",
            icon: <Workflow aria-hidden="true" size={20} />,
          },
        ].map((item) => (
          <div
            key={item.label}
            className="rounded-lg border border-slate-700 bg-slate-950 p-4"
          >
            <div className="flex items-center gap-2 text-teal-300">
              {item.icon}
              <p className="text-sm font-semibold uppercase tracking-wide">
                {item.label}
              </p>
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-200">
              {item.value}
            </p>
          </div>
        ))}
      </div>

      <div className="mt-4 rounded-lg border border-teal-400/40 bg-teal-950/40 p-4">
        <p className="text-sm font-semibold text-teal-100">
          The job of this page is to help you design the memory subsystem, then
          notice the tradeoffs before the quiz asks for them.
        </p>
        <MathText
          text={String.raw`\[M_{t+1}=E\left(M_t, F(\phi_t)\right),\qquad m_t=R(M_t,\mathrm{context}_t)\]`}
          className="mt-3 min-w-0 max-w-full overflow-x-auto overflow-y-hidden rounded-md border border-teal-400/30 bg-slate-950 px-3 py-2 text-teal-50 [&_.katex-mathml]:hidden"
        />
      </div>
    </div>
  );
}

function TaxonomyMap() {
  return (
    <div
      data-testid="memory-taxonomy-map"
      className="mt-8 grid gap-5 lg:grid-cols-[0.95fr_1.05fr]"
    >
      <div className="rounded-lg border border-slate-300 bg-white p-5">
        <h3 className="text-xl font-semibold text-slate-950">
          First learn the three questions
        </h3>
        <p className="mt-3 text-sm leading-6 text-slate-700">
          A memory label is incomplete until you know its carrier, its purpose,
          and the operations that keep it alive. This is why a simple short-term
          versus long-term split misses most of the survey.
        </p>

        <div className="mt-5 grid gap-3">
          {[
            {
              label: "What carries it?",
              answer: "Form",
              detail:
                "Readable tokens, learned parameters, or hidden latent state.",
            },
            {
              label: "Why keep it?",
              answer: "Function",
              detail:
                "Facts for continuity, experience for better behavior, or active workspace for this task.",
            },
            {
              label: "How does it change?",
              answer: "Dynamics",
              detail:
                "Formation extracts it, evolution repairs it, retrieval uses it.",
            },
          ].map((item) => (
            <div
              key={item.answer}
              className="grid gap-2 rounded-lg border border-slate-200 bg-slate-50 p-4 sm:grid-cols-[9rem_1fr]"
            >
              <p className="text-sm font-semibold text-teal-700">
                {item.answer}
              </p>
              <div>
                <p className="text-sm font-semibold text-slate-950">
                  {item.label}
                </p>
                <p className="mt-1 text-sm leading-6 text-slate-700">
                  {item.detail}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid min-w-0 gap-4">
        <div className="rounded-lg border border-slate-300 bg-white p-5">
          <h3 className="text-xl font-semibold text-slate-950">
            Forms are carriers, not time horizons
          </h3>
          <div className="mt-4 grid gap-3 md:grid-cols-3">
            {["Token-level", "Parametric", "Latent"].map((family) => (
              <div
                key={family}
                className={`rounded-lg border p-4 ${formFamilyTone[family]}`}
              >
                <p className="text-sm font-semibold">{family}</p>
                <ul className="mt-3 space-y-2 text-sm leading-5">
                  {memoryForms
                    .filter((form) => form.family === family)
                    .map((form) => (
                      <li key={form.id}>{form.shortLabel}</li>
                    ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-lg border border-slate-300 bg-white p-5">
          <h3 className="text-xl font-semibold text-slate-950">
            Functions are purposes, not storage types
          </h3>
          <div className="mt-4 grid gap-3 md:grid-cols-3">
            {["Factual", "Experiential", "Working"].map((family) => (
              <div
                key={family}
                className={`rounded-lg border p-4 ${functionFamilyTone[family]}`}
              >
                <p className="text-sm font-semibold">{family}</p>
                <ul className="mt-3 space-y-2 text-sm leading-5">
                  {memoryFunctions
                    .filter((item) => item.family === family)
                    .map((item) => (
                      <li key={item.id}>{item.shortLabel}</li>
                    ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function BoundaryLab() {
  const [activeId, setActiveId] = useState<BoundaryId>("agent-memory");
  const active = boundaryCases[activeId];

  return (
    <div
      data-testid="memory-boundary-lab"
      className="mt-8 rounded-lg border border-slate-300 bg-white p-5"
    >
      <div className="grid gap-5 lg:grid-cols-[18rem_1fr]">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-teal-700">
            Boundary lab
          </p>
          <h3 className="mt-2 text-2xl font-semibold text-slate-950">
            Decide what kind of memory problem you are looking at
          </h3>
          <p className="mt-3 text-sm leading-6 text-slate-700">
            The survey separates agent memory from neighboring ideas by asking
            what persists, who updates it, and whether it participates in the
            agent decision loop.
          </p>
          <div className="mt-5 grid gap-2">
            {(Object.keys(boundaryCases) as BoundaryId[]).map((id) => (
              <ControlButton
                key={id}
                isActive={activeId === id}
                onClick={() => setActiveId(id)}
              >
                <span className="inline-flex items-center gap-2">
                  {boundaryCases[id].icon}
                  {boundaryCases[id].label}
                </span>
              </ControlButton>
            ))}
          </div>
        </div>

        <div className="rounded-lg border border-teal-200 bg-teal-50 p-5">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-2 rounded-md border border-teal-600 bg-white px-3 py-2 text-sm font-semibold text-teal-950">
              {active.icon}
              {active.label}
            </span>
            <span className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-700">
              Scope test
            </span>
          </div>

          <h4 className="mt-5 text-2xl font-semibold text-slate-950">
            {active.title}
          </h4>
          <div className="mt-5 grid gap-3 md:grid-cols-2">
            {[
              ["Core question", active.question],
              ["Memory surface", active.memorySurface],
              ["What changes", active.whatChanges],
              ["Common confusion", active.confusion],
            ].map(([label, text]) => (
              <div
                key={label}
                className="rounded-lg border border-teal-200 bg-white p-4"
              >
                <p className="text-sm font-semibold text-teal-800">{label}</p>
                <p className="mt-2 text-sm leading-6 text-slate-700">{text}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function DesignWorkbench() {
  const [scenarioId, setScenarioId] =
    useState<MemoryScenarioId>("personal-assistant");
  const [formId, setFormId] = useState<MemoryFormId>("planar-token");
  const [functionId, setFunctionId] =
    useState<MemoryFunctionId>("user-factual");
  const [formationId, setFormationId] = useState<FormationId>(
    "structured-construction",
  );
  const [evolutionIds, setEvolutionIds] = useState<readonly EvolutionId[]>([
    "updating",
    "forgetting",
  ]);
  const [retrievalId, setRetrievalId] = useState<RetrievalId>("task-start");

  const evaluation = useMemo(
    () =>
      getMemoryDesignEvaluation({
        scenarioId,
        formId,
        functionId,
        formationId,
        evolutionIds,
        retrievalId,
      }),
    [evolutionIds, formationId, formId, functionId, retrievalId, scenarioId],
  );
  const scenario = getMemoryScenario(scenarioId);
  const form = getMemoryForm(formId);
  const memoryFunction = getMemoryFunction(functionId);
  const formation = getFormationOperation(formationId);
  const retrieval = getRetrievalOperation(retrievalId);

  function toggleEvolution(id: EvolutionId) {
    setEvolutionIds((current) =>
      current.includes(id)
        ? current.filter((candidate) => candidate !== id)
        : [...current, id],
    );
  }

  return (
    <div
      data-testid="memory-design-workbench"
      className="mt-8 rounded-lg border border-slate-300 bg-white p-5 shadow-sm"
    >
      <div className="grid gap-6 xl:grid-cols-[1.08fr_0.92fr]">
        <div className="min-w-0 space-y-6">
          <div>
            <h3 className="text-2xl font-semibold text-slate-950">
              Build a memory design, then read the tradeoffs
            </h3>
            <p className="mt-3 text-sm leading-6 text-slate-700">
              The point is not to find one universal memory type. The point is
              to match a carrier, purpose, and lifecycle to the actual agent
              pressure.
            </p>
          </div>

          <div>
            <p className="text-sm font-semibold uppercase tracking-wide text-teal-700">
              1. Scenario
            </p>
            <div className="mt-3 grid gap-2 md:grid-cols-3">
              {memoryScenarios.map((scenarioOption) => (
                <ControlButton
                  key={scenarioOption.id}
                  isActive={scenarioId === scenarioOption.id}
                  onClick={() => setScenarioId(scenarioOption.id)}
                >
                  {scenarioOption.shortLabel}
                </ControlButton>
              ))}
            </div>
          </div>

          <div>
            <p className="text-sm font-semibold uppercase tracking-wide text-teal-700">
              2. Form: what carries memory?
            </p>
            <div className="mt-3 grid gap-2 md:grid-cols-2">
              {memoryForms.map((formOption) => (
                <ControlButton
                  key={formOption.id}
                  isActive={formId === formOption.id}
                  onClick={() => setFormId(formOption.id)}
                  tone={formOption.family === "Parametric" ? "amber" : "teal"}
                >
                  <span className="block">{formOption.shortLabel}</span>
                  <span className="mt-1 block text-xs font-normal opacity-80">
                    {formOption.family}
                  </span>
                </ControlButton>
              ))}
            </div>
          </div>

          <div>
            <p className="text-sm font-semibold uppercase tracking-wide text-teal-700">
              3. Function: why keep it?
            </p>
            <div className="mt-3 grid gap-2 md:grid-cols-2">
              {memoryFunctions.map((functionOption) => (
                <ControlButton
                  key={functionOption.id}
                  isActive={functionId === functionOption.id}
                  onClick={() => setFunctionId(functionOption.id)}
                  tone={
                    functionOption.family === "Experiential"
                      ? "indigo"
                      : functionOption.family === "Factual"
                        ? "rose"
                        : "teal"
                  }
                >
                  <span className="block">{functionOption.shortLabel}</span>
                  <span className="mt-1 block text-xs font-normal opacity-80">
                    {functionOption.family}
                  </span>
                </ControlButton>
              ))}
            </div>
          </div>

          <div className="grid gap-5 lg:grid-cols-2">
            <div>
              <p className="text-sm font-semibold uppercase tracking-wide text-teal-700">
                4. Formation
              </p>
              <div className="mt-3 grid gap-2">
                {formationOperations.map((operation) => (
                  <ControlButton
                    key={operation.id}
                    isActive={formationId === operation.id}
                    onClick={() => setFormationId(operation.id)}
                  >
                    {operation.label}
                  </ControlButton>
                ))}
              </div>
            </div>

            <div>
              <p className="text-sm font-semibold uppercase tracking-wide text-teal-700">
                5. Evolution
              </p>
              <div className="mt-3 grid gap-2">
                {evolutionOperations.map((operation) => (
                  <ControlButton
                    key={operation.id}
                    isActive={evolutionIds.includes(operation.id)}
                    onClick={() => toggleEvolution(operation.id)}
                    tone="amber"
                  >
                    {operation.label}
                  </ControlButton>
                ))}
              </div>
            </div>
          </div>

          <div>
            <p className="text-sm font-semibold uppercase tracking-wide text-teal-700">
              6. Retrieval
            </p>
            <div className="mt-3 grid gap-2 md:grid-cols-2">
              {retrievalOperations.map((operation) => (
                <ControlButton
                  key={operation.id}
                  isActive={retrievalId === operation.id}
                  onClick={() => setRetrievalId(operation.id)}
                >
                  <span className="block">{operation.label}</span>
                  <span className="mt-1 block text-xs font-normal opacity-80">
                    {operation.timing}
                  </span>
                </ControlButton>
              ))}
            </div>
          </div>
        </div>

        <div className="min-w-0 space-y-4">
          <div
            role="status"
            data-testid="memory-design-status"
            className="rounded-lg border border-teal-300 bg-teal-50 p-5"
          >
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-sm font-semibold uppercase tracking-wide text-teal-800">
                Design diagnosis
              </p>
              <span className="rounded-md border border-teal-700 bg-white px-3 py-2 text-sm font-bold text-teal-950">
                {evaluation.score}/100
              </span>
            </div>
            <h4 className="mt-4 text-2xl font-semibold text-slate-950">
              {evaluation.fitLabel}
            </h4>
            <p className="mt-3 text-sm leading-6 text-slate-700">
              {evaluation.diagnosis}
            </p>
            {evaluation.warnings.length > 0 && (
              <div className="mt-4 rounded-lg border border-amber-300 bg-amber-50 p-4">
                <p className="text-sm font-semibold text-amber-950">
                  Watch this tradeoff
                </p>
                <ul className="mt-2 space-y-2 text-sm leading-6 text-amber-950">
                  {evaluation.warnings.slice(0, 3).map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          <div className="grid gap-3 rounded-lg border border-slate-300 bg-slate-50 p-5">
            <MetricBar
              label="Transparency"
              value={evaluation.metrics.transparency}
            />
            <MetricBar
              label="Adaptability"
              value={evaluation.metrics.adaptability}
            />
            <MetricBar label="Structure" value={evaluation.metrics.structure} />
            <MetricBar
              label="Context efficiency"
              value={evaluation.metrics.contextEfficiency}
            />
          </div>

          <div className="rounded-lg border border-slate-300 bg-white p-5">
            <p className="text-sm font-semibold uppercase tracking-wide text-slate-500">
              Current blueprint
            </p>
            <div className="mt-4 space-y-4 text-sm leading-6 text-slate-700">
              <div>
                <p className="font-semibold text-slate-950">{scenario.label}</p>
                <p>{scenario.goal}</p>
              </div>
              <div>
                <p className="font-semibold text-slate-950">{form.label}</p>
                <p>{form.definition}</p>
                <p className="mt-2 text-slate-600">{form.caution}</p>
              </div>
              <div>
                <p className="font-semibold text-slate-950">
                  {memoryFunction.label}
                </p>
                <p>{memoryFunction.role}</p>
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <p className="font-semibold text-slate-950">
                    {formation.label}
                  </p>
                  <p>{formation.explanation}</p>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <p className="font-semibold text-slate-950">
                    {retrieval.label}
                  </p>
                  <p>{retrieval.explanation}</p>
                </div>
              </div>
            </div>
          </div>

          <div className="rounded-lg border border-emerald-300 bg-emerald-50 p-5">
            <p className="text-sm font-semibold text-emerald-950">Strengths</p>
            <ul className="mt-2 space-y-2 text-sm leading-6 text-emerald-950">
              {evaluation.strengths.map((strength) => (
                <li key={strength}>{strength}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

function LifecycleRunway() {
  return (
    <div
      data-testid="memory-lifecycle-runway"
      className="mt-8 grid gap-5 lg:grid-cols-3"
    >
      {[
        {
          title: "Formation",
          icon: <History aria-hidden="true" size={22} />,
          summary:
            "The agent turns raw artifacts into memory candidates: observations, tool outputs, plans, traces, feedback, or summaries.",
          items: formationOperations.map((operation) => operation.label),
        },
        {
          title: "Evolution",
          icon: <RefreshCw aria-hidden="true" size={22} />,
          summary:
            "The system integrates memory into the existing base by merging, correcting, pruning, or restructuring it.",
          items: evolutionOperations.map((operation) => operation.label),
        },
        {
          title: "Retrieval",
          icon: <Route aria-hidden="true" size={22} />,
          summary:
            "The agent asks for the right memory signal at the right moment and formats it for the next decision.",
          items: retrievalOperations.map((operation) => operation.label),
        },
      ].map((stage, index) => (
        <div
          key={stage.title}
          className="rounded-lg border border-slate-300 bg-white p-5"
        >
          <div className="flex items-center gap-3">
            <span className="flex h-11 w-11 items-center justify-center rounded-lg border border-teal-300 bg-teal-50 text-teal-800">
              {stage.icon}
            </span>
            <div>
              <p className="text-sm font-semibold uppercase tracking-wide text-teal-700">
                Operator {index + 1}
              </p>
              <h3 className="text-xl font-semibold text-slate-950">
                {stage.title}
              </h3>
            </div>
          </div>
          <p className="mt-4 text-sm leading-6 text-slate-700">
            {stage.summary}
          </p>
          <div className="mt-5 flex flex-wrap gap-2">
            {stage.items.map((item) => (
              <span
                key={item}
                className="rounded-md border border-slate-300 bg-slate-50 px-3 py-2 text-sm font-semibold text-slate-700"
              >
                {item}
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function FrontierBoard() {
  const [activeId, setActiveId] = useState<FrontierId>("trust");
  const active = frontiers[activeId];

  return (
    <div
      data-testid="memory-frontier-board"
      className="mt-8 rounded-lg border border-slate-300 bg-white p-5"
    >
      <div className="grid gap-6 lg:grid-cols-[20rem_1fr]">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-teal-700">
            Frontier board
          </p>
          <h3 className="mt-2 text-2xl font-semibold text-slate-950">
            The survey ends by asking what must mature next
          </h3>
          <p className="mt-3 text-sm leading-6 text-slate-700">
            Each frontier is a pressure on the same taxonomy. The memory carrier
            and function still matter, but automation, RL, modalities, sharing,
            world models, and trust change the design stakes.
          </p>
          <div className="mt-5 grid gap-2">
            {(Object.keys(frontiers) as FrontierId[]).map((id) => (
              <ControlButton
                key={id}
                isActive={activeId === id}
                onClick={() => setActiveId(id)}
                tone={id === "trust" ? "rose" : "teal"}
              >
                <span className="inline-flex items-center gap-2">
                  {frontiers[id].icon}
                  {frontiers[id].label}
                </span>
              </ControlButton>
            ))}
          </div>
        </div>

        <div className="rounded-lg border border-rose-200 bg-rose-50 p-5">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-2 rounded-md border border-rose-700 bg-white px-3 py-2 text-sm font-semibold text-rose-950">
              {active.icon}
              {active.label}
            </span>
          </div>
          <h4 className="mt-5 text-2xl font-semibold text-slate-950">
            {active.title}
          </h4>
          <div className="mt-5 grid gap-3 md:grid-cols-3">
            {[
              ["Look-back", active.lookBack],
              ["Future perspective", active.future],
              ["Design pressure", active.pressure],
            ].map(([label, text]) => (
              <div
                key={label}
                className="rounded-lg border border-rose-200 bg-white p-4"
              >
                <p className="text-sm font-semibold text-rose-900">{label}</p>
                <p className="mt-2 text-sm leading-6 text-slate-700">{text}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function MemorySurveyLearningPage({ experience }: Props) {
  return (
    <main className="min-h-screen bg-stone-50 text-slate-950">
      <PageBand tone="ink">
        <PageInner className="grid min-h-[560px] items-center gap-8 lg:grid-cols-[1fr_0.92fr]">
          <div className="min-w-0 space-y-6">
            <div className="min-w-0 max-w-3xl">
              <p className="text-sm font-semibold uppercase tracking-wide text-teal-300">
                AI agents / memory survey
              </p>
              <h1 className="mt-3 text-4xl font-semibold tracking-normal text-white md:text-6xl md:leading-tight">
                Design memory as a managed subsystem, not a bigger prompt
              </h1>
              <p className="mt-5 max-w-2xl text-base leading-7 text-slate-300 md:text-lg">
                The core move of the survey is a three-axis taxonomy. Memory has
                a carrier, a purpose, and a lifecycle. This page turns that into
                a design workbench before you practice the 40 matching
                questions.
              </p>
              <div className="mt-6 flex flex-wrap items-center gap-3">
                <QuizTransitionButton
                  sourceId={experience.sourceId}
                  label="Start memory survey questions"
                />
                <span className="rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm font-semibold text-slate-300">
                  {experience.durationMinutes} min learning page
                </span>
              </div>
            </div>

            <div className="grid min-w-0 max-w-2xl gap-3 sm:grid-cols-3">
              {experience.outcomes.slice(0, 3).map((outcome) => (
                <div
                  key={outcome}
                  className="rounded-lg border border-slate-700 bg-slate-900 p-3 text-sm leading-6 text-slate-200"
                >
                  {outcome}
                </div>
              ))}
            </div>
          </div>

          <HeroMemoryMap />
        </PageInner>
      </PageBand>

      <PageBand tone="warm">
        <PageInner>
          <SectionIntro
            eyebrow="Concept floor"
            title="A memory system is an evolving state, not just a storage bucket"
            body="The paper formalizes agent memory as a state that can already contain cross-task knowledge, receive new artifacts during a task, evolve through maintenance operations, and be retrieved when the policy needs support."
          />
          <div className="mt-8 grid gap-5 lg:grid-cols-[1fr_0.9fr]">
            <div className="min-w-0 rounded-lg border border-slate-300 bg-white p-5">
              <h3 className="text-xl font-semibold text-slate-950">
                The minimal loop
              </h3>
              <p className="mt-3 text-sm leading-6 text-slate-700">
                Think of <InlineMath text={String.raw`\(M_t\)`} /> as the agent
                memory state at a moment in a task. The agent produces artifacts
                such as plans, tool outputs, observations, feedback, or
                self-evaluations. Formation turns some of those artifacts into
                candidate memories, evolution integrates them, and retrieval
                supplies a usable memory signal for the next decision.
              </p>
              <MathText
                text={String.raw`\[M_{t+1}=E\left(M_t, F(\phi_t)\right),\qquad m_t=R(M_t,\mathrm{context}_t)\]`}
                className="mt-5 min-w-0 max-w-full overflow-x-auto overflow-y-hidden rounded-lg border border-slate-300 bg-slate-50 px-4 py-3 text-slate-950 [&_.katex-mathml]:hidden"
              />
            </div>
            <div className="min-w-0 rounded-lg border border-slate-300 bg-white p-5">
              <h3 className="text-xl font-semibold text-slate-950">
                Why long-term versus short-term is too coarse
              </h3>
              <p className="mt-3 text-sm leading-6 text-slate-700">
                A single memory base can contain durable cross-task knowledge
                and still accumulate active task state. The difference often
                comes from invocation timing: what is loaded at task start, what
                is updated mid-task, and what remains after the episode ends.
              </p>
              <div className="mt-5 grid gap-3">
                {[
                  "Cross-task memory can be factual, experiential, or both.",
                  "Inside-task memory can still use explicit records, latent state, or summaries.",
                  "Retrieval does not have to run at every step to count as memory.",
                ].map((item) => (
                  <p
                    key={item}
                    className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-semibold text-slate-700"
                  >
                    {item}
                  </p>
                ))}
              </div>
            </div>
          </div>
        </PageInner>
      </PageBand>

      <PageBand>
        <PageInner>
          <SectionIntro
            eyebrow="Forms, functions, dynamics"
            title="Read every memory design through three axes"
            body="The survey's taxonomy is useful because it separates representation, purpose, and lifecycle. A vector database, a LoRA adapter, a graph, and a KV cache can all be memory-like, but they answer different design questions."
          />
          <TaxonomyMap />
        </PageInner>
      </PageBand>

      <PageBand tone="warm">
        <PageInner>
          <SectionIntro
            eyebrow="Scope boundaries"
            title="Separate agent memory from nearby systems before choosing tools"
            body="Many mistakes come from treating any retained information as the same thing. The boundary lab keeps the neighboring ideas visible so the workbench can focus on agent memory."
          />
          <BoundaryLab />
        </PageInner>
      </PageBand>

      <PageBand>
        <PageInner>
          <SectionIntro
            eyebrow="Design workbench"
            title="Build the memory system the scenario actually needs"
            body="Use the controls in order: pick the agent pressure, choose the carrier, decide why the memory exists, and then schedule formation, evolution, and retrieval. The diagnosis exposes the tradeoff rather than declaring one memory type best."
          />
          <DesignWorkbench />
        </PageInner>
      </PageBand>

      <PageBand tone="warm">
        <PageInner>
          <SectionIntro
            eyebrow="Dynamics"
            title="A useful memory store needs maintenance, not only recall"
            body="Formation, evolution, and retrieval are operators. They can be invoked at different times, but a deployed memory system needs a deliberate answer for each one."
          />
          <LifecycleRunway />
        </PageInner>
      </PageBand>

      <PageBand>
        <PageInner>
          <SectionIntro
            eyebrow="Research frontiers"
            title="The hard problems are lifecycle problems at larger scale"
            body="The paper's frontiers are not separate from the taxonomy. They ask how memory should be generated, automated, rewarded, shared, made multimodal, used as a world model, and kept trustworthy."
          />
          <FrontierBoard />
        </PageInner>
      </PageBand>

      <PageBand tone="ink">
        <PageInner>
          <div className="grid gap-8 lg:grid-cols-[1fr_0.8fr]">
            <div>
              <p className="text-sm font-semibold uppercase tracking-wide text-teal-300">
                Recap
              </p>
              <h2 className="mt-2 text-3xl font-semibold tracking-normal text-white md:text-4xl">
                Before the quiz, check that you can explain the design pressure
              </h2>
              <p className="mt-4 max-w-3xl text-base leading-7 text-slate-300">
                Ask what carries the memory, why the agent needs it, how it is
                formed, how it evolves, when it is retrieved, and what trust
                boundary it creates. If you can answer those six questions, the
                survey categories will feel like design tools instead of a list
                of names.
              </p>
            </div>
            <div className="rounded-lg border border-slate-700 bg-slate-900 p-5">
              <h3 className="text-xl font-semibold text-white">
                Practice target
              </h3>
              <ul className="mt-4 space-y-3 text-sm leading-6 text-slate-300">
                {experience.outcomes.map((outcome) => (
                  <li key={outcome}>{outcome}</li>
                ))}
              </ul>
              <div className="mt-6">
                <QuizTransitionButton
                  sourceId={experience.sourceId}
                  label="Start memory survey questions"
                />
              </div>
            </div>
          </div>
        </PageInner>
      </PageBand>
    </main>
  );
}
