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

const runtimeStages = {
  overview: {
    label: "Model",
    kicker: "decoder-only boundary",
    title: "An LLM is a scaled language model that generates text",
    summary:
      "Lecture 3 uses the modern practical meaning of LLM: a decoder-only transformer that assigns probabilities to token sequences and repeatedly predicts the next token.",
  },
  moe: {
    label: "MoE",
    kicker: "active parameters",
    title: "Route each token through selected feed-forward experts",
    summary:
      "Sparse MoE increases stored capacity without running every expert on every token, but the router must avoid sending everything to the same expert.",
  },
  decode: {
    label: "Decode",
    kicker: "next-token policy",
    title:
      "Turn logits into a next token without confusing search and sampling",
    summary:
      "Greedy and beam search chase likely sequences. Top-k, top-p, temperature, and guided decoding change or constrain the candidate set before sampling.",
  },
  prompt: {
    label: "Prompt",
    kicker: "context budget",
    title: "Condition a fixed model with examples, reasoning, and branches",
    summary:
      "Prompting changes the information in context, not the model weights. Better task behavior usually spends more input tokens, output tokens, or parallel samples.",
  },
  cache: {
    label: "Cache",
    kicker: "serving memory",
    title: "Reuse attention state and make the KV cache fit",
    summary:
      "Autoregressive inference benefits from storing previous keys and values, then reducing duplicate KV heads, wasted allocation, or representation size.",
  },
  accelerate: {
    label: "Accelerate",
    kicker: "token throughput",
    title:
      "Advance through tokens faster while preserving the target model role",
    summary:
      "Speculative decoding uses a cheap draft model plus target-model verification. Multi-token prediction builds draft-like heads into the model objective.",
  },
} as const;

type RuntimeStageId = keyof typeof runtimeStages;

const stageOrder = [
  "overview",
  "moe",
  "decode",
  "prompt",
  "cache",
  "accelerate",
] as const satisfies readonly RuntimeStageId[];

const requestTokens = ["[BOS]", "my", "teddy", "bear", "is"] as const;

const tokenCandidates = [
  {
    token: "cute",
    logit: 4.2,
    note: "highest local continuation for the prefix",
  },
  {
    token: "reading",
    logit: 3.2,
    note: "plausible continuation with a semantic shift",
  },
  {
    token: "sleepy",
    logit: 2.7,
    note: "still natural, but less likely here",
  },
  {
    token: "coding",
    logit: 2.15,
    note: "creative tail candidate",
  },
  {
    token: "microscope",
    logit: 0.65,
    note: "grammatical only in unusual contexts",
  },
  {
    token: "{",
    logit: -0.1,
    note: "low probability unless a schema constrains output",
  },
] as const;

const guidedRows = [
  {
    token: "{",
    probability: 0.7,
    note: "valid JSON object start",
  },
  {
    token: '"first_name"',
    probability: 0.16,
    note: "valid after an opening brace",
  },
  {
    token: ":",
    probability: 0.08,
    note: "valid after a key",
  },
  {
    token: "the",
    probability: 0.04,
    note: "blocked by the grammar at this step",
  },
  {
    token: "road",
    probability: 0.02,
    note: "blocked by the grammar at this step",
  },
] as const;

const decodingPolicies = {
  greedy: {
    label: "Greedy",
    badge: "one local path",
    output: "my teddy bear is cute",
    summary:
      "Greedy decoding takes the largest next-token probability at each step. It is repeatable, but a locally best token can still lead to a weaker full answer.",
  },
  beam: {
    label: "Beam",
    badge: "several likely paths",
    output: "my teddy bear is very cute",
    summary:
      "Beam search keeps several high-scoring partial sequences. It broadens search over likely paths, but it often reduces diversity for open-ended text.",
  },
  topK: {
    label: "Top-k",
    badge: "fixed candidate count",
    output: "my teddy bear is reading",
    summary:
      "Top-k sampling removes all but the k highest-probability tokens. The same k can be too narrow in one context and too broad in another.",
  },
  topP: {
    label: "Top-p",
    badge: "adaptive nucleus",
    output: "my teddy bear is sleepy",
    summary:
      "Top-p sampling keeps the smallest candidate set whose cumulative probability reaches the threshold, so the set size adapts to distribution shape.",
  },
  guided: {
    label: "Guided JSON",
    badge: "grammar-valid tokens",
    output: '{ "first_name": "teddy", "hobby": "reading" }',
    summary:
      "Guided decoding uses an external grammar or schema to block invalid next tokens before the model samples from the remaining valid choices.",
  },
} as const;

type DecodingPolicyId = keyof typeof decodingPolicies;

const expertNames = ["Math", "Story", "Code", "Facts"] as const;
const expertColorClasses = [
  "bg-cyan-400",
  "bg-emerald-400",
  "bg-amber-300",
  "bg-fuchsia-400",
] as const;

const routedTokens = ["proof", "teddy", "function", "date", "matrix"] as const;

const moeScenarios = {
  balanced: {
    label: "Balanced",
    badge: "healthy usage",
    summary:
      "Different token representations activate different experts. The stored parameter pool is large, but per-token active compute stays tied to top-k routing.",
    probabilities: [
      [0.68, 0.14, 0.1, 0.08],
      [0.09, 0.72, 0.1, 0.09],
      [0.08, 0.12, 0.72, 0.08],
      [0.12, 0.16, 0.1, 0.62],
      [0.62, 0.12, 0.18, 0.08],
    ],
  },
  collapsed: {
    label: "Collapsed",
    badge: "routing collapse",
    summary:
      "Most tokens route to one expert. Capacity exists on paper, but unused experts contribute little unless load balancing or noisy gating changes the training signal.",
    probabilities: [
      [0.08, 0.78, 0.08, 0.06],
      [0.07, 0.8, 0.05, 0.08],
      [0.09, 0.76, 0.08, 0.07],
      [0.08, 0.79, 0.04, 0.09],
      [0.1, 0.74, 0.08, 0.08],
    ],
  },
  noisy: {
    label: "Noisy gate",
    badge: "forced exploration",
    summary:
      "Noise and an auxiliary load-balancing loss give underused experts chances to receive tokens while the router and experts are trained jointly.",
    probabilities: [
      [0.5, 0.24, 0.16, 0.1],
      [0.14, 0.48, 0.2, 0.18],
      [0.16, 0.18, 0.5, 0.16],
      [0.18, 0.22, 0.14, 0.46],
      [0.44, 0.2, 0.2, 0.16],
    ],
  },
} as const;

type MoEScenarioId = keyof typeof moeScenarios;

const promptStrategies = {
  zeroShot: {
    label: "Zero-shot",
    badge: "instruction only",
    contextTokens: 22,
    generatedTokens: 24,
    branches: 1,
    prompt:
      "Classify the review as positive or negative: The answer was clear.",
    summary:
      "The model relies on task wording and what it already learned during training. No examples are added to calibrate the format.",
  },
  fewShot: {
    label: "Few-shot",
    badge: "examples in context",
    contextTokens: 72,
    generatedTokens: 24,
    branches: 1,
    prompt:
      "unclear -> negative; helpful -> positive; clear -> positive. Review: The answer was clear.",
    summary:
      "Examples demonstrate the input-output pattern. They often help, but they spend context and add prefill work.",
  },
  chainOfThought: {
    label: "Chain of thought",
    badge: "reasoning tokens",
    contextTokens: 36,
    generatedTokens: 90,
    branches: 1,
    prompt:
      "Solve step by step, then put the final answer last: how old will a 2020-born bear be next year?",
    summary:
      "A reasoning trace can improve multi-step tasks and expose bad premises, but the extra generated tokens increase cost and latency.",
  },
  selfConsistency: {
    label: "Self-consistency",
    badge: "parallel samples",
    contextTokens: 36,
    generatedTokens: 90,
    branches: 5,
    prompt:
      "Sample several reasoning paths independently, extract the final answer from each, and vote.",
    summary:
      "Multiple branches can improve robustness when final answers can be extracted, but this multiplies inference work and still needs evaluation.",
  },
} as const;

type PromptStrategyId = keyof typeof promptStrategies;

const cacheTechniques = {
  kv: {
    label: "KV cache",
    badge: "reuse exact tensors",
    kvHeads: 8,
    representationFactor: 1,
    usesPagedAllocation: false,
    summary:
      "The current token forms a fresh query, but previous keys and values can be reused instead of recomputed for every new token.",
  },
  gqa: {
    label: "GQA/MQA",
    badge: "share K/V heads",
    kvHeads: 2,
    representationFactor: 1,
    usesPagedAllocation: false,
    summary:
      "Grouped-query attention keeps multiple query heads while sharing key/value heads across groups. Multi-query attention is the one-K/V-head endpoint.",
  },
  paged: {
    label: "PagedAttention",
    badge: "block allocation",
    kvHeads: 8,
    representationFactor: 1,
    usesPagedAllocation: true,
    summary:
      "PagedAttention stores KV blocks in smaller non-contiguous pages so serving does not reserve one maximum-length slab for every request.",
  },
  latent: {
    label: "Latent attention",
    badge: "compressed cache",
    kvHeads: 1,
    representationFactor: 0.4,
    usesPagedAllocation: false,
    summary:
      "Multi-latent attention stores a smaller shared latent representation and expands it into keys and values when attention needs them.",
  },
} as const;

type CacheTechniqueId = keyof typeof cacheTechniques;

const requestLengths = [13, 5, 16, 7] as const;
const maxReservedSlots = 16;
const pageBlockSize = 4;

const accelerators = {
  speculative: {
    label: "Speculative",
    badge: "draft then verify",
    draftTokens: ["cute", "and", "smart"],
    verdicts: ["accept", "accept", "reject"] as const,
    summary:
      "A small draft model proposes several tokens. The target model evaluates the block, accepts matching tokens under the sampling rule, and resumes from the corrected distribution after rejection.",
  },
  mtp: {
    label: "MTP",
    badge: "draft heads inside",
    draftTokens: ["cute", "and", "smart"],
    verdicts: ["head 1", "head 2", "head 3"] as const,
    summary:
      "Multi-token prediction changes training so extra heads predict future positions. At inference, those heads act like embedded draft proposals rather than a separate small model.",
  },
} as const;

type AcceleratorId = keyof typeof accelerators;

function getSoftmaxRows(temperature: number) {
  const safeTemperature = Math.max(0.25, temperature);
  const scaled = tokenCandidates.map(
    (candidate) => candidate.logit / safeTemperature,
  );
  const maxLogit = Math.max(...scaled);
  const expValues = scaled.map((value) => Math.exp(value - maxLogit));
  const total = expValues.reduce((sum, value) => sum + value, 0);

  return tokenCandidates
    .map((candidate, index) => ({
      ...candidate,
      probability: expValues[index] / total,
    }))
    .sort((first, second) => second.probability - first.probability);
}

function getIncludedTokenSet(
  policyId: DecodingPolicyId,
  rows: readonly { token: string; probability: number }[],
) {
  if (policyId === "guided") return new Set(["{", '"first_name"', ":"]);
  if (policyId === "greedy")
    return new Set(rows.slice(0, 1).map((r) => r.token));
  if (policyId === "beam") return new Set(rows.slice(0, 3).map((r) => r.token));
  if (policyId === "topK") return new Set(rows.slice(0, 4).map((r) => r.token));

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
    .sort((first, second) => second.probability - first.probability)
    .slice(0, topK)
    .map((entry) => entry.index);
}

function getCacheStats(techniqueId: CacheTechniqueId) {
  const technique = cacheTechniques[techniqueId];
  const usedSlots = requestLengths.reduce((sum, length) => sum + length, 0);
  const reservedSlots = technique.usesPagedAllocation
    ? requestLengths.reduce(
        (sum, length) =>
          sum + Math.ceil(length / pageBlockSize) * pageBlockSize,
        0,
      )
    : requestLengths.length * maxReservedSlots;
  const wasteSlots = reservedSlots - usedSlots;
  const memoryIndex = Math.round(
    reservedSlots * technique.kvHeads * technique.representationFactor,
  );

  return {
    usedSlots,
    reservedSlots,
    wasteSlots,
    memoryIndex,
  };
}

function RuntimePipeline({
  activeStageId,
  onStageChange,
}: {
  activeStageId: RuntimeStageId;
  onStageChange: (stageId: RuntimeStageId) => void;
}) {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-4">
      <p className="text-xs font-semibold uppercase tracking-wide text-cyan-300">
        One generation request
      </p>
      <div className="mt-4 flex flex-wrap gap-2">
        {requestTokens.map((token) => (
          <span
            key={token}
            className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 font-mono text-xs font-semibold text-neutral-100"
          >
            {token}
          </span>
        ))}
        <span className="rounded-md border border-dashed border-cyan-500 px-2 py-1 font-mono text-xs font-semibold text-cyan-200">
          ?
        </span>
      </div>

      <div className="mt-5 grid gap-2">
        {stageOrder.map((stageId, index) => {
          const stage = runtimeStages[stageId];
          const isActive = stageId === activeStageId;
          return (
            <button
              key={stageId}
              type="button"
              aria-pressed={isActive}
              onClick={() => onStageChange(stageId)}
              className={`grid grid-cols-[2rem_1fr] gap-3 rounded-md border px-3 py-3 text-left transition-colors ${
                isActive
                  ? "border-cyan-300 bg-cyan-300 text-neutral-950"
                  : "border-neutral-800 bg-neutral-900 text-neutral-200 hover:border-neutral-600"
              }`}
            >
              <span className="flex h-8 w-8 items-center justify-center rounded-full bg-neutral-950 text-sm font-bold text-cyan-200">
                {index + 1}
              </span>
              <span>
                <span className="block text-sm font-semibold">
                  {stage.label}
                </span>
                <span
                  className={`mt-0.5 block text-xs ${
                    isActive ? "text-neutral-800" : "text-neutral-400"
                  }`}
                >
                  {stage.kicker}
                </span>
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function OverviewStage() {
  const families = [
    {
      label: "Encoder-only",
      example: "BERT",
      role: "turns text into contextual embeddings for tasks such as classification",
    },
    {
      label: "Encoder-decoder",
      example: "T5",
      role: "maps input text to output text with encoder states and decoder cross-attention",
    },
    {
      label: "Decoder-only",
      example: "GPT, Llama, Gemma",
      role: "uses masked self-attention to generate text one token at a time",
    },
  ] as const;

  return (
    <div className="grid gap-4">
      <div className="grid gap-3 md:grid-cols-3">
        {families.map((family) => {
          const isActive = family.label === "Decoder-only";
          return (
            <article
              key={family.label}
              className={`rounded-md border p-4 ${
                isActive
                  ? "border-cyan-300 bg-cyan-950/50"
                  : "border-neutral-800 bg-neutral-950"
              }`}
            >
              <p className="text-sm font-semibold text-neutral-100">
                {family.label}
              </p>
              <p className="mt-1 text-xs font-semibold uppercase tracking-wide text-neutral-400">
                {family.example}
              </p>
              <p className="mt-3 text-sm leading-6 text-neutral-300">
                {family.role}
              </p>
            </article>
          );
        })}
      </div>
      <p
        role="status"
        className="rounded-md border border-cyan-500/40 bg-cyan-950/30 px-3 py-2 text-sm leading-6 text-cyan-100"
      >
        BERT can be large and transformer-based, but Lecture 3 reserves the LLM
        label for large text-generating language models, usually decoder-only
        transformers trained around next-token prediction.
      </p>
    </div>
  );
}

function MoEStage() {
  const [scenarioId, setScenarioId] = useState<MoEScenarioId>("balanced");
  const [topK, setTopK] = useState<1 | 2>(1);
  const scenario = moeScenarios[scenarioId];
  const occupancy = useMemo(() => {
    const counts = expertNames.map(() => 0);
    for (const probabilities of scenario.probabilities) {
      for (const expertIndex of getTopExpertIndexes(probabilities, topK)) {
        counts[expertIndex] += 1;
      }
    }

    return counts.map((count) => count / (routedTokens.length * topK));
  }, [scenario, topK]);

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap gap-2">
        {Object.entries(moeScenarios).map(([id, item]) => (
          <button
            key={id}
            type="button"
            onClick={() => setScenarioId(id as MoEScenarioId)}
            className={`rounded-md border px-3 py-2 text-sm font-semibold ${
              scenarioId === id
                ? "border-emerald-300 bg-emerald-300 text-neutral-950"
                : "border-neutral-700 text-neutral-200 hover:border-neutral-500"
            }`}
          >
            {item.label}
          </button>
        ))}
        {[1, 2].map((value) => (
          <button
            key={value}
            type="button"
            onClick={() => setTopK(value as 1 | 2)}
            className={`rounded-md border px-3 py-2 text-sm font-semibold ${
              topK === value
                ? "border-amber-300 bg-amber-300 text-neutral-950"
                : "border-neutral-700 text-neutral-200 hover:border-neutral-500"
            }`}
          >
            top-k {value}
          </button>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-[1fr_0.75fr]">
        <div className="grid gap-3">
          {routedTokens.map((token, tokenIndex) => {
            const probabilities = scenario.probabilities[tokenIndex];
            const activeExperts = getTopExpertIndexes(probabilities, topK);
            return (
              <article
                key={token}
                className="rounded-md border border-neutral-800 bg-neutral-950 p-3"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="font-mono text-sm font-semibold text-neutral-100">
                    {token}
                  </span>
                  <span className="text-xs font-semibold text-neutral-400">
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
                        <div className="flex h-16 items-end rounded-sm bg-neutral-800 p-1">
                          <div
                            className={`w-full rounded-sm ${
                              isActive
                                ? expertColorClasses[expertIndex]
                                : "bg-neutral-600"
                            }`}
                            style={{
                              height: `${Math.max(
                                10,
                                probabilities[expertIndex] * 100,
                              )}%`,
                            }}
                          />
                        </div>
                        <p className="truncate text-center text-xs font-semibold text-neutral-300">
                          {expert}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </article>
            );
          })}
        </div>

        <div className="rounded-md border border-neutral-800 bg-neutral-950 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-neutral-400">
            {scenario.badge}
          </p>
          <div className="mt-4 grid gap-3">
            {expertNames.map((expert, index) => (
              <div key={expert}>
                <div className="flex justify-between text-xs font-semibold text-neutral-300">
                  <span>{expert}</span>
                  <span>{(occupancy[index] * 100).toFixed(0)}%</span>
                </div>
                <div className="mt-1 h-2 rounded-full bg-neutral-800">
                  <div
                    className={`h-2 rounded-full ${expertColorClasses[index]}`}
                    style={{
                      width: `${Math.max(4, occupancy[index] * 100)}%`,
                    }}
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
    </div>
  );
}

function DecodeStage() {
  const [policyId, setPolicyId] = useState<DecodingPolicyId>("topP");
  const [temperature, setTemperature] = useState(0.8);
  const activePolicy = decodingPolicies[policyId];
  const rows = useMemo(
    () =>
      policyId === "guided" ? [...guidedRows] : getSoftmaxRows(temperature),
    [policyId, temperature],
  );
  const includedTokens = useMemo(
    () => getIncludedTokenSet(policyId, rows),
    [policyId, rows],
  );

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap gap-2">
        {Object.entries(decodingPolicies).map(([id, item]) => (
          <button
            key={id}
            type="button"
            onClick={() => setPolicyId(id as DecodingPolicyId)}
            className={`rounded-md border px-3 py-2 text-sm font-semibold ${
              policyId === id
                ? "border-cyan-300 bg-cyan-300 text-neutral-950"
                : "border-neutral-700 text-neutral-200 hover:border-neutral-500"
            }`}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-[1fr_0.8fr]">
        <div>
          <label
            htmlFor="lecture3-temperature"
            className="text-sm font-semibold text-neutral-100"
          >
            Temperature: {temperature.toFixed(1)}
          </label>
          <input
            id="lecture3-temperature"
            type="range"
            min="0.3"
            max="1.6"
            step="0.1"
            value={temperature}
            disabled={policyId === "guided"}
            onChange={(event) => setTemperature(Number(event.target.value))}
            className="mt-2 w-full accent-cyan-300"
          />

          <div className="mt-4 grid gap-2">
            {rows.map((row) => {
              const isIncluded = includedTokens.has(row.token);
              return (
                <article
                  key={row.token}
                  className={`rounded-md border p-3 ${
                    isIncluded
                      ? "border-cyan-300 bg-cyan-950/40"
                      : "border-neutral-800 bg-neutral-950"
                  }`}
                >
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-mono text-sm font-semibold text-neutral-100">
                      {row.token}
                    </span>
                    <span className="text-xs font-semibold text-neutral-400">
                      {(row.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="mt-2 h-2 rounded-full bg-neutral-800">
                    <div
                      className={`h-2 rounded-full ${
                        isIncluded ? "bg-cyan-300" : "bg-neutral-600"
                      }`}
                      style={{
                        width: `${Math.max(5, row.probability * 100)}%`,
                      }}
                    />
                  </div>
                  <p className="mt-2 text-xs leading-5 text-neutral-400">
                    {row.note}
                  </p>
                </article>
              );
            })}
          </div>
        </div>

        <div className="rounded-md border border-neutral-800 bg-neutral-950 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-neutral-400">
            {activePolicy.badge}
          </p>
          <p className="mt-3 rounded-md border border-neutral-800 bg-neutral-900 p-3 font-mono text-sm leading-6 text-cyan-100">
            {activePolicy.output}
          </p>
          <p
            role="status"
            className="mt-4 rounded-md border border-cyan-500/40 bg-cyan-950/30 px-3 py-2 text-sm leading-6 text-cyan-100"
          >
            {activePolicy.summary}
          </p>
        </div>
      </div>
    </div>
  );
}

function PromptStage() {
  const [strategyId, setStrategyId] = useState<PromptStrategyId>("fewShot");
  const strategy = promptStrategies[strategyId];
  const totalTokenWork =
    strategy.contextTokens + strategy.generatedTokens * strategy.branches;
  const tokenWorkWidth = Math.min(100, (totalTokenWork / 500) * 100);

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap gap-2">
        {Object.entries(promptStrategies).map(([id, item]) => (
          <button
            key={id}
            type="button"
            onClick={() => setStrategyId(id as PromptStrategyId)}
            className={`rounded-md border px-3 py-2 text-sm font-semibold ${
              strategyId === id
                ? "border-amber-300 bg-amber-300 text-neutral-950"
                : "border-neutral-700 text-neutral-200 hover:border-neutral-500"
            }`}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="rounded-md border border-neutral-800 bg-neutral-950 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-neutral-400">
            Prompt anatomy
          </p>
          <div className="mt-3 grid gap-2">
            {["context", "instruction", "input", "constraints"].map((part) => (
              <div
                key={part}
                className="rounded-md border border-neutral-800 bg-neutral-900 px-3 py-2 text-sm font-semibold capitalize text-neutral-200"
              >
                {part}
              </div>
            ))}
          </div>
          <p className="mt-4 font-mono text-sm leading-6 text-amber-100">
            {strategy.prompt}
          </p>
        </div>

        <div className="rounded-md border border-neutral-800 bg-neutral-950 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-neutral-400">
            {strategy.badge}
          </p>
          <div className="mt-4 grid grid-cols-3 gap-2 text-center">
            <MetricTile label="Input" value={`${strategy.contextTokens}`} />
            <MetricTile label="Output" value={`${strategy.generatedTokens}`} />
            <MetricTile label="Branches" value={`${strategy.branches}`} />
          </div>
          <div className="mt-4">
            <div className="flex justify-between text-xs font-semibold text-neutral-400">
              <span>relative token work</span>
              <span>{totalTokenWork}</span>
            </div>
            <div className="mt-1 h-2 rounded-full bg-neutral-800">
              <div
                className="h-2 rounded-full bg-amber-300"
                style={{ width: `${Math.max(8, tokenWorkWidth)}%` }}
              />
            </div>
          </div>
          {strategyId === "selfConsistency" && (
            <div className="mt-4 grid gap-2 text-sm text-neutral-300">
              {["5", "5", "4", "5", "6"].map((answer, index) => (
                <div
                  key={`${answer}-${index}`}
                  className="rounded-md border border-neutral-800 bg-neutral-900 px-3 py-2"
                >
                  branch {index + 1}: final answer {answer}
                </div>
              ))}
            </div>
          )}
          <p
            role="status"
            className="mt-4 rounded-md border border-amber-500/40 bg-amber-950/30 px-3 py-2 text-sm leading-6 text-amber-100"
          >
            {strategy.summary}
          </p>
        </div>
      </div>
    </div>
  );
}

function CacheStage() {
  const [techniqueId, setTechniqueId] = useState<CacheTechniqueId>("kv");
  const technique = cacheTechniques[techniqueId];
  const stats = getCacheStats(techniqueId);

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap gap-2">
        {Object.entries(cacheTechniques).map(([id, item]) => (
          <button
            key={id}
            type="button"
            onClick={() => setTechniqueId(id as CacheTechniqueId)}
            className={`rounded-md border px-3 py-2 text-sm font-semibold ${
              techniqueId === id
                ? "border-fuchsia-300 bg-fuchsia-300 text-neutral-950"
                : "border-neutral-700 text-neutral-200 hover:border-neutral-500"
            }`}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-[1fr_0.8fr]">
        <div className="rounded-md border border-neutral-800 bg-neutral-950 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-neutral-400">
            four concurrent requests
          </p>
          <div className="mt-4 grid gap-3">
            {requestLengths.map((length, index) => {
              const reserved = technique.usesPagedAllocation
                ? Math.ceil(length / pageBlockSize) * pageBlockSize
                : maxReservedSlots;
              return (
                <div key={`${length}-${index}`}>
                  <div className="flex justify-between text-xs font-semibold text-neutral-400">
                    <span>request {index + 1}</span>
                    <span>
                      used {length} / reserved {reserved}
                    </span>
                  </div>
                  <div className="mt-1 flex h-5 overflow-hidden rounded-sm border border-neutral-800">
                    {Array.from({ length: reserved }).map((_, slotIndex) => (
                      <span
                        key={slotIndex}
                        className={`h-full flex-1 border-r border-neutral-900 last:border-r-0 ${
                          slotIndex < length
                            ? "bg-fuchsia-300"
                            : "bg-neutral-800"
                        }`}
                      />
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="rounded-md border border-neutral-800 bg-neutral-950 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-neutral-400">
            {technique.badge}
          </p>
          <div className="mt-4 grid grid-cols-2 gap-2 text-center">
            <MetricTile label="Used slots" value={`${stats.usedSlots}`} />
            <MetricTile label="Waste" value={`${stats.wasteSlots}`} />
            <MetricTile label="K/V heads" value={`${technique.kvHeads}`} />
            <MetricTile label="Memory index" value={`${stats.memoryIndex}`} />
          </div>
          <p
            role="status"
            className="mt-4 rounded-md border border-fuchsia-500/40 bg-fuchsia-950/30 px-3 py-2 text-sm leading-6 text-fuchsia-100"
          >
            {technique.summary}
          </p>
        </div>
      </div>
    </div>
  );
}

function AccelerateStage() {
  const [acceleratorId, setAcceleratorId] =
    useState<AcceleratorId>("speculative");
  const accelerator = accelerators[acceleratorId];

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap gap-2">
        {Object.entries(accelerators).map(([id, item]) => (
          <button
            key={id}
            type="button"
            onClick={() => setAcceleratorId(id as AcceleratorId)}
            className={`rounded-md border px-3 py-2 text-sm font-semibold ${
              acceleratorId === id
                ? "border-rose-300 bg-rose-300 text-neutral-950"
                : "border-neutral-700 text-neutral-200 hover:border-neutral-500"
            }`}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-[1fr_0.8fr]">
        <div className="rounded-md border border-neutral-800 bg-neutral-950 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-neutral-400">
            proposed token block
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            {accelerator.draftTokens.map((token, index) => {
              const verdict = accelerator.verdicts[index];
              const isRejected = verdict === "reject";
              return (
                <span
                  key={`${token}-${index}`}
                  className={`rounded-md border px-3 py-2 font-mono text-sm font-semibold ${
                    isRejected
                      ? "border-rose-300 bg-rose-950/40 text-rose-100"
                      : "border-emerald-300 bg-emerald-950/40 text-emerald-100"
                  }`}
                >
                  {token} / {verdict}
                </span>
              );
            })}
          </div>
          <div className="mt-5 grid gap-2 text-sm text-neutral-300">
            <div className="rounded-md border border-neutral-800 bg-neutral-900 px-3 py-2">
              draft: cheap proposals
            </div>
            <div className="rounded-md border border-neutral-800 bg-neutral-900 px-3 py-2">
              target: distribution authority
            </div>
            <div className="rounded-md border border-neutral-800 bg-neutral-900 px-3 py-2">
              next token after the block: available from the target pass
            </div>
          </div>
        </div>

        <div className="rounded-md border border-neutral-800 bg-neutral-950 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-neutral-400">
            {accelerator.badge}
          </p>
          <p
            role="status"
            className="mt-4 rounded-md border border-rose-500/40 bg-rose-950/30 px-3 py-2 text-sm leading-6 text-rose-100"
          >
            {accelerator.summary}
          </p>
        </div>
      </div>
    </div>
  );
}

function MetricTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-neutral-800 bg-neutral-900 px-3 py-2">
      <p className="text-xs font-semibold text-neutral-400">{label}</p>
      <p className="mt-1 text-base font-semibold text-neutral-100">{value}</p>
    </div>
  );
}

function RuntimeStagePanel({
  activeStageId,
}: {
  activeStageId: RuntimeStageId;
}) {
  const activeStage = runtimeStages[activeStageId];

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-5">
      <p className="text-sm font-semibold uppercase tracking-wide text-cyan-300">
        {activeStage.kicker}
      </p>
      <h2 className="mt-2 text-2xl font-semibold text-neutral-50">
        {activeStage.title}
      </h2>
      <p className="mt-2 max-w-3xl text-sm leading-6 text-neutral-300">
        {activeStage.summary}
      </p>
      <div className="mt-5">
        {activeStageId === "overview" && <OverviewStage />}
        {activeStageId === "moe" && <MoEStage />}
        {activeStageId === "decode" && <DecodeStage />}
        {activeStageId === "prompt" && <PromptStage />}
        {activeStageId === "cache" && <CacheStage />}
        {activeStageId === "accelerate" && <AccelerateStage />}
      </div>
    </div>
  );
}

function RuntimeTraceLab() {
  const [activeStageId, setActiveStageId] =
    useState<RuntimeStageId>("overview");

  return (
    <section
      data-testid="llm-runtime-lab"
      className="rounded-lg border border-neutral-800 bg-neutral-950 p-5"
    >
      <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-cyan-300">
            Lecture 3 runtime trace
          </p>
          <h2 className="mt-2 text-2xl font-semibold text-neutral-50">
            Trace one request through the LLM runtime
          </h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-neutral-300">
            The Lecture 3 ideas sit at different layers of the same request:
            model family, expert routing, next-token selection, prompt context,
            attention-cache memory, and token-throughput tricks.
          </p>
        </div>
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-[0.42fr_0.58fr]">
        <RuntimePipeline
          activeStageId={activeStageId}
          onStageChange={setActiveStageId}
        />
        <RuntimeStagePanel activeStageId={activeStageId} />
      </div>
    </section>
  );
}

function MechanismMap() {
  const rows = [
    {
      label: "Architecture",
      concepts: [
        "decoder-only backbone",
        "MoE FFN experts",
        "routing collapse",
      ],
      takeaway:
        "This layer decides which parameters exist and which subset is active for a token.",
    },
    {
      label: "Generation",
      concepts: [
        "greedy/beam",
        "top-k/top-p",
        "temperature",
        "guided decoding",
      ],
      takeaway:
        "This layer decides how logits become the next emitted token or a valid structured token.",
    },
    {
      label: "Prompting",
      concepts: [
        "zero-shot",
        "few-shot",
        "chain of thought",
        "self-consistency",
      ],
      takeaway:
        "This layer changes context and sampling strategy while the model weights stay fixed.",
    },
    {
      label: "Serving",
      concepts: ["KV cache", "GQA/MQA", "PagedAttention", "latent attention"],
      takeaway:
        "This layer attacks repeated attention work, cache size, memory waste, and bandwidth pressure.",
    },
    {
      label: "Throughput",
      concepts: [
        "speculative decoding",
        "target verification",
        "multi-token prediction",
      ],
      takeaway:
        "This layer tries to advance multiple output positions from fewer expensive target-model steps.",
    },
  ] as const;

  return (
    <section className="rounded-lg border border-neutral-800 bg-neutral-900 p-5">
      <p className="text-sm font-semibold uppercase tracking-wide text-emerald-300">
        Mechanism map
      </p>
      <h2 className="mt-2 text-2xl font-semibold text-neutral-50">
        Place each technique before comparing tradeoffs
      </h2>
      <div className="mt-5 grid gap-3">
        {rows.map((row) => (
          <article
            key={row.label}
            className="grid gap-3 rounded-md border border-neutral-800 bg-neutral-950 p-4 lg:grid-cols-[0.22fr_0.4fr_0.38fr]"
          >
            <h3 className="text-base font-semibold text-neutral-100">
              {row.label}
            </h3>
            <div className="flex flex-wrap gap-2">
              {row.concepts.map((concept) => (
                <span
                  key={concept}
                  className="rounded-md bg-neutral-800 px-2 py-1 text-xs font-semibold text-neutral-200"
                >
                  {concept}
                </span>
              ))}
            </div>
            <p className="text-sm leading-6 text-neutral-300">{row.takeaway}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

export default function StanfordCME295Lecture3LearningPage({
  experience,
}: Props) {
  return (
    <main className="bg-neutral-950 text-neutral-50">
      <section className="border-b border-neutral-800">
        <div className="mx-auto grid min-h-[520px] w-full max-w-6xl items-center gap-8 px-4 py-10 lg:grid-cols-[1fr_0.95fr] lg:py-14">
          <div className="space-y-6">
            <div className="space-y-3">
              <p className="text-sm font-semibold uppercase tracking-wide text-cyan-300">
                Stanford CME295 Lecture 3
              </p>
              <h1 className="max-w-3xl text-3xl font-semibold tracking-normal text-neutral-50 md:text-5xl md:leading-tight">
                Follow one request through LLM inference
              </h1>
              <p className="max-w-2xl text-base leading-7 text-neutral-300 md:text-lg">
                Lecture 3 links model architecture to runtime behavior: what an
                LLM is, how sparse experts activate, how next tokens are chosen,
                how prompts steer a fixed model, and why serving systems obsess
                over KV-cache memory and token throughput.
              </p>
              <p className="text-sm font-semibold text-emerald-300">
                {experience.durationMinutes} min / {experience.level}
              </p>
            </div>

            <div className="grid gap-2 sm:grid-cols-2">
              {experience.outcomes.map((outcome) => (
                <div
                  key={outcome}
                  className="rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-2 text-sm leading-5 text-neutral-200"
                >
                  {outcome}
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-5">
            <p className="text-sm font-semibold uppercase tracking-wide text-cyan-300">
              Runtime stack
            </p>
            <div className="mt-4 grid gap-3">
              {stageOrder.map((stageId, index) => (
                <div
                  key={stageId}
                  className="grid grid-cols-[2.25rem_1fr] items-center gap-3 rounded-md border border-neutral-800 bg-neutral-950 px-3 py-2"
                >
                  <span className="flex h-8 w-8 items-center justify-center rounded-full bg-cyan-300 text-sm font-bold text-neutral-950">
                    {index + 1}
                  </span>
                  <div>
                    <p className="text-sm font-semibold text-neutral-100">
                      {runtimeStages[stageId].label}
                    </p>
                    <p className="text-xs text-neutral-400">
                      {runtimeStages[stageId].kicker}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-10">
        <RuntimeTraceLab />

        <section className="grid gap-5 lg:grid-cols-2">
          <FormulaBlock
            title="Temperature rescales logits before softmax"
            formula={String.raw`\[
P(t_i)=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
\]`}
            explanation="Lower T concentrates probability on high-logit tokens. Higher T flattens the distribution, increasing diversity pressure without adding new knowledge."
          />
          <FormulaBlock
            title="Sparse MoE keeps only selected expert outputs active"
            formula={String.raw`\[
\hat{y}=\sum_{i \in \mathrm{top}\text{-}k} G_i(x)E_i(x)
\]`}
            explanation="The router G scores feed-forward experts for a token representation x. Top-k routing sets nonselected experts to zero for that token."
          />
          <FormulaBlock
            title="Load balancing discourages routing collapse"
            formula={String.raw`\[
L_{\mathrm{aux}}=\alpha N \sum_i f_i P_i
\]`}
            explanation="The lecture frames f_i as the fraction of tokens routed to expert i and P_i as average routing probability. The extra term pushes expert usage toward a healthier spread."
          />
          <FormulaBlock
            title="KV caching stores keys and values, not old queries"
            formula={String.raw`\[
\mathrm{cache}_t=\{K_{1:t-1},V_{1:t-1}\}
\]`}
            explanation="For the current token, the model needs a fresh query that attends to previous keys and values. Reusing those K/V tensors avoids redundant attention projections."
          />
        </section>

        <MechanismMap />

        <section className="grid gap-5 lg:grid-cols-2">
          <CheckForUnderstanding
            testId="routing-collapse-check"
            title="Check: routing collapse"
            question="A sparse MoE model has 64 experts, but almost every token in a batch goes to expert 7. What is the core problem?"
            options={[
              {
                label:
                  "The router has collapsed onto a small subset, so stored capacity is not being used well.",
                explanation:
                  "Routing collapse is expert-utilization imbalance. Auxiliary losses and noisy gating are meant to keep experts in play during training.",
              },
              {
                label:
                  "The decoder is using top-p sampling instead of beam search.",
                explanation:
                  "Sampling policy affects vocabulary-token choice, not which feed-forward experts are selected inside an MoE layer.",
              },
              {
                label:
                  "The KV cache is fragmented because every expert stores a separate context window.",
                explanation:
                  "KV-cache memory and expert routing are separate inference concerns. Fragmentation is a serving-memory issue.",
              },
            ]}
            correctIndex={0}
          />
          <CheckForUnderstanding
            testId="target-model-check"
            title="Check: speculative decoding"
            question="Why does the target model still matter after a draft model proposes several tokens?"
            options={[
              {
                label:
                  "The target model verifies or rejects draft tokens and remains the distribution authority.",
                explanation:
                  "Speculative decoding is fast because cheap proposals can be accepted in blocks, but the target model's probabilities control correctness under the sampling rule.",
              },
              {
                label:
                  "The target model is retrained on the draft tokens during each request.",
                explanation:
                  "Speculative decoding is an inference-time method. No per-request gradient training happens.",
              },
              {
                label:
                  "The target model only stores prompt examples in the KV cache.",
                explanation:
                  "KV caching stores attention keys and values. It is not a prompt-example database or a training memory.",
              },
            ]}
            correctIndex={0}
          />
        </section>

        <RecapSection
          title="Lecture 3 mental model"
          items={[
            "Modern LLMs in this lecture are scaled decoder-only language models that generate text through repeated next-token prediction.",
            "Sparse MoE increases total capacity while keeping per-token active compute tied to routed experts.",
            "Decoding policies control whether the model searches, samples, truncates, rescales, or obeys structural validity.",
            "Prompting spends context, reasoning tokens, or parallel samples to improve behavior without changing weights.",
            "Inference optimizations attack redundant attention work, KV-cache memory pressure, allocation waste, and token latency.",
            "Speculative decoding and MTP are token-throughput ideas; they are not the same as top-k/top-p sampling.",
          ]}
        />

        <section className="rounded-lg border border-emerald-500/40 bg-emerald-950/20 p-5">
          <h2 className="text-xl font-semibold text-emerald-100">
            Ready for the Lecture 3 MCQs
          </h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-neutral-300">
            The practice questions should now feel like placing each mechanism
            in the runtime trace: architecture, routing, decoding, prompting,
            memory, and token acceleration all change different parts of the
            system.
          </p>
          <div className="mt-5">
            <QuizTransitionButton sourceId={experience.sourceId} />
          </div>
        </section>
      </div>
    </main>
  );
}
