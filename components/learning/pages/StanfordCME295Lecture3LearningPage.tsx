"use client";

import { useMemo, useState } from "react";
import type { LearningExperience } from "../../../lib/learning";
import {
  CheckForUnderstanding,
  FormulaBlock,
  QuizTransitionButton,
  RecapSection,
} from "../LearningPrimitives";

type Props = {
  experience: LearningExperience;
};

const nextTokenCandidates = [
  { token: "reading", logit: 3.15, note: "locally likely continuation" },
  { token: "sleepy", logit: 2.35, note: "plausible style shift" },
  { token: "coding", logit: 1.95, note: "creative but less likely" },
  { token: "hungry", logit: 1.4, note: "still grammatical" },
  { token: "microscope", logit: 0.35, note: "low probability surprise" },
  { token: "{", logit: -0.2, note: "valid only under JSON constraints" },
] as const;

const guidedCandidates = [
  { token: "{", probability: 0.68, note: "valid JSON object start" },
  { token: '"first_name"', probability: 0.17, note: "valid after {" },
  { token: ":", probability: 0.09, note: "valid after a key" },
  { token: "reading", probability: 0.04, note: "blocked by schema here" },
  { token: "the", probability: 0.02, note: "blocked by schema here" },
] as const;

const decodingModes = {
  greedy: {
    label: "Greedy",
    badge: "local maximum",
    output: "A teddy bear is reading.",
    summary:
      "Always take the highest probability next token. This is fast and repeatable, but a locally best token can lead to a weaker full sequence.",
    feedback:
      "Greedy decoding is deterministic when the implementation is deterministic, but it does not explore alternatives.",
  },
  beam: {
    label: "Beam",
    badge: "k paths",
    output: "A cute teddy bear reads.",
    summary:
      "Keep several high probability partial sequences and extend them in parallel. It is useful for constrained sequence tasks, but often bland for open-ended generation.",
    feedback:
      "Beam search keeps likely paths, yet it adds compute and can prefer shorter or less diverse completions.",
  },
  topK: {
    label: "Top-k",
    badge: "fixed candidate set",
    output: "A teddy bear is sleepy.",
    summary:
      "Sample only from the k most probable tokens. The candidate count is fixed even if the probability distribution is very sharp or very flat.",
    feedback:
      "Top-k sampling limits the tail with a fixed k, so the same k can be too narrow in one context and too broad in another.",
  },
  topP: {
    label: "Top-p",
    badge: "adaptive nucleus",
    output: "A teddy bear is coding.",
    summary:
      "Sample from the smallest set of tokens whose cumulative probability reaches the threshold p.",
    feedback:
      "Top-p keeps the smallest probability mass above p, so the number of available tokens adapts to the distribution.",
  },
  guided: {
    label: "Guided",
    badge: "schema-valid tokens",
    output: '{ "first_name": "teddy", "hobby": "reading" }',
    summary:
      "Apply a grammar or schema so invalid next tokens are not allowed, even if the model assigns them probability.",
    feedback:
      "Guided decoding changes the allowed token set at each step, which is why it helps with JSON and other structured outputs.",
  },
} as const;

type DecodingModeId = keyof typeof decodingModes;

const expertNames = ["Math", "Language", "Code", "Facts"] as const;
const expertColorClasses = [
  "bg-sky-400",
  "bg-emerald-400",
  "bg-amber-400",
  "bg-rose-400",
] as const;

const routerTokens = ["proof", "story", "function", "date", "equation"];

const routerScenarios = {
  balanced: {
    label: "Balanced router",
    badge: "healthy usage",
    summary:
      "Different token representations activate different feed-forward experts, keeping capacity available across the layer.",
    probabilities: [
      [0.72, 0.12, 0.08, 0.08],
      [0.1, 0.7, 0.08, 0.12],
      [0.08, 0.12, 0.72, 0.08],
      [0.08, 0.18, 0.1, 0.64],
      [0.68, 0.1, 0.12, 0.1],
    ],
  },
  collapsed: {
    label: "Collapsed router",
    badge: "routing collapse",
    summary:
      "Most tokens choose the same expert. The model owns many parameters, but the active compute keeps flowing through a narrow part of the network.",
    probabilities: [
      [0.08, 0.78, 0.08, 0.06],
      [0.06, 0.82, 0.05, 0.07],
      [0.09, 0.76, 0.08, 0.07],
      [0.07, 0.8, 0.04, 0.09],
      [0.1, 0.74, 0.08, 0.08],
    ],
  },
  noisy: {
    label: "Noisy gate",
    badge: "forced exploration",
    summary:
      "Noise or auxiliary load-balancing pressure gives underused experts a chance to receive tokens during training.",
    probabilities: [
      [0.5, 0.24, 0.16, 0.1],
      [0.14, 0.48, 0.2, 0.18],
      [0.16, 0.18, 0.5, 0.16],
      [0.18, 0.22, 0.14, 0.46],
      [0.44, 0.2, 0.2, 0.16],
    ],
  },
} as const;

type RouterScenarioId = keyof typeof routerScenarios;

const promptStrategies = {
  zeroShot: {
    label: "Zero-shot",
    cost: "lowest prompt cost",
    prompt:
      "Question: Classify this review as positive or negative. Review: The answer was clear.",
    result:
      "Works when the base model already understands the task, but the prompt gives little task-specific calibration.",
  },
  fewShot: {
    label: "Few-shot",
    cost: "more tokens",
    prompt:
      "Example 1: unclear -> negative. Example 2: helpful -> positive. Review: The answer was clear.",
    result:
      "Examples demonstrate the input/output pattern and often improve in-context learning at the cost of latency and context space.",
  },
  chainOfThought: {
    label: "Chain of thought",
    cost: "reasoning tokens",
    prompt:
      "Solve step by step, then give the final answer last: how old will a 2020-born bear be next year?",
    result:
      "Reasoning tokens can improve multi-step tasks and make failures easier to inspect, but they increase generated tokens.",
  },
  selfConsistency: {
    label: "Self-consistency",
    cost: "parallel samples",
    prompt:
      "Sample several reasoning paths, extract each final answer, then use majority voting.",
    result:
      "Aggregating independent reasoning paths can be more robust, but it multiplies inference work and needs answer extraction.",
  },
} as const;

type PromptStrategyId = keyof typeof promptStrategies;

const optimizationTechniques = {
  kvCache: {
    label: "KV cache",
    category: "Exact: avoid redundancy",
    status:
      "KV caching stores previous keys and values so the next token can reuse them instead of recomputing the whole prefix.",
    detail:
      "The current token still needs a fresh query, key, and value. Earlier keys and values are the reusable part during autoregressive decoding.",
  },
  gqa: {
    label: "GQA/MQA",
    category: "Exact: share K/V heads",
    status:
      "Grouped-query attention keeps many query heads but shares key/value heads within groups; MQA is the one-K/V-head extreme.",
    detail:
      "This reduces KV cache size while preserving multiple query projections for attention diversity.",
  },
  paged: {
    label: "PagedAttention",
    category: "Exact: memory management",
    status:
      "PagedAttention stores KV cache blocks in non-contiguous pages so serving systems waste less reserved memory.",
    detail:
      "The point is not to change the math of attention; it changes how cache memory is allocated for many concurrent requests.",
  },
  latent: {
    label: "Latent attention",
    category: "Compressed representation",
    status:
      "Latent attention stores a smaller shared representation and decompresses it for keys and values when needed.",
    detail:
      "The lecture frames this as a way to shrink what each transformer block must keep in memory.",
  },
  speculative: {
    label: "Speculative",
    category: "Draft then verify",
    status:
      "Speculative decoding lets a small draft model propose several tokens, then the large target model accepts or rejects them.",
    detail:
      "When draft tokens are accepted, one target pass advances multiple tokens while preserving the target model distribution under the sampling rule.",
  },
  mtp: {
    label: "MTP",
    category: "Train multiple heads",
    status:
      "Multi-token prediction trains extra heads to predict future positions, embedding a draft-like mechanism inside the model.",
    detail:
      "The training objective changes from only next-token prediction to predicting several upcoming tokens.",
  },
} as const;

type OptimizationTechniqueId = keyof typeof optimizationTechniques;

function getSoftmaxRows(temperature: number) {
  const safeTemperature = Math.max(0.25, temperature);
  const scaled = nextTokenCandidates.map(
    (candidate) => candidate.logit / safeTemperature,
  );
  const maxLogit = Math.max(...scaled);
  const expValues = scaled.map((value) => Math.exp(value - maxLogit));
  const total = expValues.reduce((sum, value) => sum + value, 0);

  return nextTokenCandidates
    .map((candidate, index) => ({
      ...candidate,
      probability: expValues[index] / total,
    }))
    .sort((a, b) => b.probability - a.probability);
}

function getIncludedTokens(
  mode: DecodingModeId,
  rows: readonly { token: string; probability: number }[],
) {
  if (mode === "guided") {
    return new Set(["{", '"first_name"', ":"]);
  }
  if (mode === "greedy") return new Set([rows[0]?.token]);
  if (mode === "beam") return new Set(rows.slice(0, 3).map((row) => row.token));
  if (mode === "topK") return new Set(rows.slice(0, 4).map((row) => row.token));

  const included = new Set<string>();
  let cumulative = 0;
  for (const row of rows) {
    included.add(row.token);
    cumulative += row.probability;
    if (cumulative >= 0.9) break;
  }
  return included;
}

function getTopExpertIndexes(probabilities: readonly number[], topK: 1 | 2) {
  return probabilities
    .map((probability, index) => ({ probability, index }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, topK)
    .map((entry) => entry.index);
}

function LLMStackVisual() {
  const stages = [
    "Prompt tokens",
    "Masked self-attention",
    "MoE feed-forward",
    "Logits",
    "Next token",
  ];

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <p className="text-sm font-semibold uppercase tracking-wide text-sky-300">
        Lecture 3 runtime
      </p>
      <div className="mt-4 grid gap-3">
        {stages.map((stage, index) => (
          <div key={stage} className="flex items-center gap-3">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-slate-950 text-sm font-bold text-sky-200">
              {index + 1}
            </span>
            <div className="flex min-h-12 flex-1 items-center rounded-md border border-slate-700 bg-slate-950 px-3 text-sm font-semibold text-slate-100">
              {stage}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-5 grid grid-cols-3 gap-2 text-center text-xs font-semibold text-slate-200">
        {[
          ["Parameters", "billions+"],
          ["Training data", "tokens"],
          ["Compute", "GPUs"],
        ].map(([label, value]) => (
          <div key={label} className="rounded-md border border-slate-800 p-3">
            <p className="text-slate-400">{label}</p>
            <p className="mt-1 text-emerald-300">{value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function GenerationControlRoom() {
  const [mode, setMode] = useState<DecodingModeId>("topP");
  const [temperature, setTemperature] = useState(0.8);
  const activeMode = decodingModes[mode];
  const candidateRows = useMemo(
    () =>
      mode === "guided" ? [...guidedCandidates] : getSoftmaxRows(temperature),
    [mode, temperature],
  );
  const includedTokens = useMemo(
    () => getIncludedTokens(mode, candidateRows),
    [candidateRows, mode],
  );

  return (
    <section
      data-testid="generation-control-room"
      className="rounded-lg border border-slate-800 bg-slate-900 p-5"
    >
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-sky-300">
            Response generation
          </p>
          <h2 className="mt-2 text-2xl font-semibold text-slate-50">
            Next-token decoding control room
          </h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">
            A decoder-only LLM repeatedly turns the current context into logits,
            softmax probabilities, and a next-token choice. The decoding rule
            decides whether the model acts locally greedy, explores a candidate
            set, or obeys a schema.
          </p>
        </div>
        <div className="rounded-md border border-slate-800 bg-slate-950 px-3 py-2 text-sm font-semibold text-emerald-300">
          {activeMode.badge}
        </div>
      </div>

      <div className="mt-5 flex flex-wrap gap-2">
        {Object.entries(decodingModes).map(([id, item]) => (
          <button
            key={id}
            type="button"
            onClick={() => setMode(id as DecodingModeId)}
            className={`rounded-md border px-3 py-2 text-sm font-semibold ${
              mode === id
                ? "border-sky-400 bg-sky-400 text-slate-950"
                : "border-slate-700 text-slate-200 hover:border-slate-500"
            }`}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-[1fr_0.85fr]">
        <div>
          <label
            htmlFor="temperature-slider"
            className="text-sm font-semibold text-slate-200"
          >
            Temperature: {temperature.toFixed(1)}
          </label>
          <input
            id="temperature-slider"
            type="range"
            min="0.3"
            max="1.6"
            step="0.1"
            value={temperature}
            onChange={(event) => setTemperature(Number(event.target.value))}
            className="mt-2 w-full accent-sky-400"
            disabled={mode === "guided"}
          />
          <div className="mt-4 space-y-2">
            {candidateRows.map((row) => {
              const isIncluded = includedTokens.has(row.token);
              return (
                <div
                  key={row.token}
                  className={`rounded-md border p-3 ${
                    isIncluded
                      ? "border-sky-400 bg-sky-950/40"
                      : "border-slate-800 bg-slate-950"
                  }`}
                >
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-mono text-sm font-semibold text-slate-100">
                      {row.token}
                    </span>
                    <span className="text-xs font-semibold text-slate-400">
                      {(row.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="mt-2 h-2 rounded-full bg-slate-800">
                    <div
                      className={`h-2 rounded-full ${
                        isIncluded ? "bg-sky-400" : "bg-slate-600"
                      }`}
                      style={{
                        width: `${Math.max(6, row.probability * 100)}%`,
                      }}
                    />
                  </div>
                  <p className="mt-2 text-xs leading-5 text-slate-400">
                    {row.note}
                  </p>
                </div>
              );
            })}
          </div>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-950 p-4">
          <h3 className="text-lg font-semibold text-slate-50">
            Generated result
          </h3>
          <p className="mt-3 rounded-md border border-slate-800 bg-slate-900 p-3 font-mono text-sm leading-6 text-emerald-200">
            {activeMode.output}
          </p>
          <p className="mt-4 text-sm leading-6 text-slate-300">
            {activeMode.summary}
          </p>
          <p
            role="status"
            className="mt-4 rounded-md border border-sky-500/40 bg-sky-950/30 px-3 py-2 text-sm leading-6 text-sky-100"
          >
            {activeMode.feedback}
          </p>
        </div>
      </div>
    </section>
  );
}

function MoERouterLab() {
  const [scenarioId, setScenarioId] = useState<RouterScenarioId>("balanced");
  const [topK, setTopK] = useState<1 | 2>(1);
  const scenario = routerScenarios[scenarioId];
  const occupancy = useMemo(() => {
    const counts = expertNames.map(() => 0);
    for (const probabilities of scenario.probabilities) {
      for (const expertIndex of getTopExpertIndexes(probabilities, topK)) {
        counts[expertIndex] += 1;
      }
    }
    return counts.map((count) => count / (routerTokens.length * topK));
  }, [scenario, topK]);

  return (
    <section
      data-testid="moe-router-lab"
      className="rounded-lg border border-slate-800 bg-slate-900 p-5"
    >
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-emerald-300">
            MoE-based LLMs
          </p>
          <h2 className="mt-2 text-2xl font-semibold text-slate-50">
            Token-level expert routing lab
          </h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">
            Modern sparse MoE layers usually replace the feed-forward network,
            not the attention heads. A router reads each token representation
            and chooses the top-k expert feed-forward networks for that token.
          </p>
        </div>
        <div className="rounded-md border border-slate-800 bg-slate-950 px-3 py-2 text-sm font-semibold text-emerald-300">
          {scenario.badge}
        </div>
      </div>

      <div className="mt-5 flex flex-wrap items-center gap-2">
        {Object.entries(routerScenarios).map(([id, item]) => (
          <button
            key={id}
            type="button"
            onClick={() => setScenarioId(id as RouterScenarioId)}
            className={`rounded-md border px-3 py-2 text-sm font-semibold ${
              scenarioId === id
                ? "border-emerald-400 bg-emerald-400 text-slate-950"
                : "border-slate-700 text-slate-200 hover:border-slate-500"
            }`}
          >
            {item.label}
          </button>
        ))}
        <div className="ml-0 flex gap-2 lg:ml-auto">
          {[1, 2].map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => setTopK(value as 1 | 2)}
              className={`rounded-md border px-3 py-2 text-sm font-semibold ${
                topK === value
                  ? "border-amber-300 bg-amber-300 text-slate-950"
                  : "border-slate-700 text-slate-200 hover:border-slate-500"
              }`}
            >
              top-k {value}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-[1fr_0.9fr]">
        <div className="space-y-3">
          {routerTokens.map((token, tokenIndex) => {
            const probabilities = scenario.probabilities[tokenIndex];
            const activeExperts = getTopExpertIndexes(probabilities, topK);
            return (
              <div
                key={token}
                className="rounded-md border border-slate-800 bg-slate-950 p-3"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="font-mono text-sm font-semibold text-slate-100">
                    {token}
                  </span>
                  <span className="text-xs font-semibold text-slate-400">
                    active:{" "}
                    {activeExperts
                      .map((expertIndex) => expertNames[expertIndex])
                      .join(", ")}
                  </span>
                </div>
                <div className="mt-3 grid grid-cols-4 gap-2">
                  {expertNames.map((expert, expertIndex) => {
                    const isActive = activeExperts.includes(expertIndex);
                    return (
                      <div key={expert} className="space-y-1">
                        <div className="h-16 rounded-sm bg-slate-800 p-1">
                          <div
                            className={`mt-auto rounded-sm ${
                              isActive
                                ? expertColorClasses[expertIndex]
                                : "bg-slate-600"
                            }`}
                            style={{
                              height: `${Math.max(
                                10,
                                probabilities[expertIndex] * 100,
                              )}%`,
                            }}
                          />
                        </div>
                        <p className="truncate text-center text-xs font-semibold text-slate-300">
                          {expert}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-950 p-4">
          <h3 className="text-lg font-semibold text-slate-50">
            Expert utilization
          </h3>
          <div className="mt-4 space-y-3">
            {expertNames.map((expert, index) => (
              <div key={expert}>
                <div className="flex justify-between gap-3 text-xs font-semibold text-slate-300">
                  <span>{expert}</span>
                  <span>{(occupancy[index] * 100).toFixed(0)}%</span>
                </div>
                <div className="mt-1 h-2 rounded-full bg-slate-800">
                  <div
                    className={`h-2 rounded-full ${expertColorClasses[index]}`}
                    style={{ width: `${Math.max(4, occupancy[index] * 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
          <p
            role="status"
            className="mt-4 rounded-md border border-emerald-500/40 bg-emerald-950/30 px-3 py-2 text-sm leading-6 text-emerald-100"
          >
            {scenario.summary}
          </p>
        </div>
      </div>
    </section>
  );
}

function PromptStrategyBoard() {
  const [strategyId, setStrategyId] = useState<PromptStrategyId>("fewShot");
  const strategy = promptStrategies[strategyId];

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <p className="text-sm font-semibold uppercase tracking-wide text-amber-300">
        Prompting strategies
      </p>
      <h2 className="mt-2 text-2xl font-semibold text-slate-50">
        Prompt budget and reasoning board
      </h2>
      <p className="mt-2 text-sm leading-6 text-slate-300">
        Prompting changes what information the model conditions on before the
        next-token loop starts. Examples, reasoning traces, and repeated samples
        can improve behavior, but they spend context, compute, and latency.
      </p>

      <div className="mt-5 flex flex-wrap gap-2">
        {Object.entries(promptStrategies).map(([id, item]) => (
          <button
            key={id}
            type="button"
            onClick={() => setStrategyId(id as PromptStrategyId)}
            className={`rounded-md border px-3 py-2 text-sm font-semibold ${
              strategyId === id
                ? "border-amber-300 bg-amber-300 text-slate-950"
                : "border-slate-700 text-slate-200 hover:border-slate-500"
            }`}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="mt-5 rounded-md border border-slate-800 bg-slate-950 p-4">
        <div className="flex flex-wrap items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
          <span>{strategy.label}</span>
          <span aria-hidden="true">/</span>
          <span>{strategy.cost}</span>
        </div>
        <p className="mt-3 font-mono text-sm leading-6 text-amber-100">
          {strategy.prompt}
        </p>
        <p role="status" className="mt-4 text-sm leading-6 text-slate-300">
          {strategy.result}
        </p>
      </div>
    </section>
  );
}

function InferenceOptimizationMap() {
  const [activeId, setActiveId] = useState<OptimizationTechniqueId>("kvCache");
  const active = optimizationTechniques[activeId];

  return (
    <section
      data-testid="inference-optimization-map"
      className="rounded-lg border border-slate-800 bg-slate-900 p-5"
    >
      <p className="text-sm font-semibold uppercase tracking-wide text-rose-300">
        Inference optimizations
      </p>
      <h2 className="mt-2 text-2xl font-semibold text-slate-50">
        Serving bottleneck map
      </h2>
      <p className="mt-2 text-sm leading-6 text-slate-300">
        The lecture separates exact efficiency from approximations. Some
        techniques preserve the same computations while avoiding waste; others
        alter architecture, representation, or token prediction to move faster.
      </p>

      <div className="mt-5 grid gap-2 sm:grid-cols-2">
        {Object.entries(optimizationTechniques).map(([id, item]) => (
          <button
            key={id}
            type="button"
            onClick={() => setActiveId(id as OptimizationTechniqueId)}
            className={`rounded-md border px-3 py-2 text-left text-sm font-semibold ${
              activeId === id
                ? "border-rose-300 bg-rose-300 text-slate-950"
                : "border-slate-700 text-slate-200 hover:border-slate-500"
            }`}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="mt-5 rounded-md border border-slate-800 bg-slate-950 p-4">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
          {active.category}
        </p>
        <p role="status" className="mt-3 text-sm leading-6 text-rose-100">
          {active.status}
        </p>
        <p className="mt-3 text-sm leading-6 text-slate-300">{active.detail}</p>
      </div>
    </section>
  );
}

export default function StanfordCME295Lecture3LearningPage({
  experience,
}: Props) {
  return (
    <main className="bg-slate-950 text-slate-50">
      <section className="border-b border-slate-800">
        <div className="mx-auto grid min-h-[520px] w-full max-w-6xl items-center gap-8 px-4 py-10 lg:grid-cols-[1.05fr_0.95fr] lg:py-14">
          <div className="space-y-6">
            <div className="space-y-3">
              <p className="text-sm font-semibold uppercase tracking-wide text-sky-300">
                Stanford CME295 Lecture 3
              </p>
              <h1 className="max-w-3xl text-3xl font-semibold tracking-normal text-slate-50 md:text-5xl md:leading-tight">
                Run the LLM generation control room
              </h1>
              <p className="max-w-2xl text-base leading-7 text-slate-300 md:text-lg">
                Lecture 3 is about the runtime choices that make large language
                models useful: decoder-only next-token prediction, sparse expert
                routing, decoding controls, prompt design, and memory-aware
                serving.
              </p>
              <p className="text-sm font-semibold text-emerald-300">
                {experience.durationMinutes} min / {experience.level}
              </p>
            </div>

            <div className="grid gap-2 sm:grid-cols-2">
              {experience.outcomes.map((outcome) => (
                <div
                  key={outcome}
                  className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-sm leading-5 text-slate-200"
                >
                  {outcome}
                </div>
              ))}
            </div>
          </div>

          <LLMStackVisual />
        </div>
      </section>

      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-10">
        <section className="grid gap-5 lg:grid-cols-[0.95fr_1.05fr]">
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
            <p className="text-sm font-semibold uppercase tracking-wide text-sky-300">
              What counts as an LLM here
            </p>
            <h2 className="mt-2 text-2xl font-semibold text-slate-50">
              Large, decoder-only, text-generating language models
            </h2>
            <p className="mt-3 text-sm leading-6 text-slate-300">
              The lecture uses the current practical meaning of LLM: a language
              model that assigns probabilities to token sequences, scales across
              parameters, training tokens, and compute, and usually generates
              text with a decoder-only transformer. Encoder-only models such as
              BERT remain important representation models, but they are not the
              generative LLM shape used in this lecture.
            </p>
          </div>

          <FormulaBlock
            title="Temperature changes the next-token distribution"
            formula={String.raw`\[
P(t_i)=\frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
\]`}
            explanation="Lower temperature sharpens the softmax around high-logit tokens. Higher temperature flattens it, making lower-ranked tokens more likely and outputs less repeatable."
          />
        </section>

        <GenerationControlRoom />

        <MoERouterLab />

        <FormulaBlock
          title="Sparse MoE activates selected experts"
          formula={String.raw`\[
\hat{y}=\sum_{i \in \mathrm{top}\text{-}k} G_i(x)E_i(x)
\]`}
          explanation="The router G scores experts for a token representation x. Sparse MoE keeps only the selected expert outputs active, which increases total capacity without making every forward pass use every parameter."
        />

        <section className="grid gap-5 lg:grid-cols-2">
          <PromptStrategyBoard />
          <InferenceOptimizationMap />
        </section>

        <section className="grid gap-5 lg:grid-cols-2">
          <CheckForUnderstanding
            testId="speculative-check"
            title="Check: speculative decoding"
            question="Why can speculative decoding speed up generation while still targeting the large model's distribution?"
            options={[
              {
                label:
                  "The small draft model replaces the target model for all accepted requests.",
                explanation:
                  "The draft model proposes tokens, but the target model still validates them. The target model is not removed from the process.",
              },
              {
                label:
                  "A draft model proposes several tokens, and one target pass accepts or rejects them with a sampling rule.",
                explanation:
                  "The accepted draft tokens let generation advance multiple positions from one target-model pass, while rejected tokens resume from the corrected distribution.",
              },
              {
                label:
                  "It lowers temperature until every generated path becomes deterministic.",
                explanation:
                  "Temperature is a decoding control, but speculative decoding is about draft proposals and target-model verification.",
              },
            ]}
            correctIndex={1}
          />

          <CheckForUnderstanding
            testId="kv-cache-check"
            title="Check: KV caching"
            question="During autoregressive inference, which cached values are useful for the next token?"
            options={[
              {
                label:
                  "Previous keys and values, because the new query attends to prior tokens.",
                explanation:
                  "The next token forms its own query, while earlier keys and values can be reused from cache.",
              },
              {
                label:
                  "Previous queries, because the model compares old queries to the new value.",
                explanation:
                  "Attention for the current step needs the current query against previous keys and values, not previous queries.",
              },
              {
                label:
                  "Only logits, because attention is not recomputed after training.",
                explanation:
                  "Logits are produced at the output layer. KV caching addresses repeated attention projections during generation.",
              },
            ]}
            correctIndex={0}
          />
        </section>

        <RecapSection
          title="Lecture 3 mental model"
          items={[
            "An LLM is a scaled language model that repeatedly predicts the next token from context.",
            "Sparse MoE increases total capacity while controlling active parameters through token-level routing.",
            "Decoding choices trade determinism, diversity, structure, compute, and sequence quality.",
            "Prompting strategies spend context and tokens to steer behavior or improve reasoning.",
            "Serving optimizations target redundant computation, KV memory pressure, and token-generation latency.",
            "Every improvement has a tradeoff: capacity, latency, memory, cost, robustness, or output diversity.",
          ]}
        />

        <section className="rounded-lg border border-emerald-500/40 bg-emerald-950/20 p-5">
          <h2 className="text-xl font-semibold text-emerald-100">
            Ready for the Lecture 3 MCQs
          </h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">
            Practice should now feel like tracing the same control room: what
            kind of model is running, which experts activate, how tokens are
            selected, what the prompt buys, and which inference optimization
            removes which bottleneck.
          </p>
          <div className="mt-5">
            <QuizTransitionButton sourceId={experience.sourceId} />
          </div>
        </section>
      </div>
    </main>
  );
}
