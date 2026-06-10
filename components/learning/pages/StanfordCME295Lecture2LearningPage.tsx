"use client";

import { useMemo, useState } from "react";
import MathText from "../../MathText";
import type { LearningExperience } from "../../../lib/learning";
import {
  CheckForUnderstanding,
  ConceptCard,
  FormulaBlock,
  InteractiveComparison,
  LearningHero,
  MisconceptionCallout,
  ProcessSteps,
  QuizTransitionButton,
  RecapSection,
  WorkedExample,
} from "../LearningPrimitives";

type Props = {
  experience: LearningExperience;
};

const positionStrategies = {
  learned: {
    label: "Learned absolute",
    location: "Added to token embeddings before the first attention layer.",
    extrapolation: "Weak for unseen lengths",
    formula: "token[m] + positionTable[m]",
    summary:
      "Every absolute index gets a trainable vector. It is flexible, but the model only has entries for the trained maximum sequence length.",
    colorClass: "bg-sky-400",
    weights: [1, 0.88, 0.76, 0.63, 0.53, 0.45],
  },
  sinusoidal: {
    label: "Sinusoidal",
    location: "Added to token embeddings before attention.",
    extrapolation: "Formula can extend",
    formula: "sin/cos waves at many frequencies",
    summary:
      "Fixed sine and cosine dimensions make dot products depend on position differences, so nearby positions tend to be more similar.",
    colorClass: "bg-emerald-400",
    weights: [1, 0.94, 0.79, 0.58, 0.43, 0.36],
  },
  t5: {
    label: "T5 bias",
    location: "Added directly to query-key logits inside softmax.",
    extrapolation: "Bucketed relative distance",
    formula: "softmax(QK^T / sqrt(d) + b[m-n])",
    summary:
      "Relative distances are bucketized, and each attention head learns distance-dependent logit biases.",
    colorClass: "bg-violet-400",
    weights: [1, 0.86, 0.69, 0.52, 0.37, 0.27],
  },
  alibi: {
    label: "ALiBi",
    location: "Added directly to query-key logits inside softmax.",
    extrapolation: "Designed for train-short, test-long",
    formula: "softmax(QK^T / sqrt(d) - slope * distance)",
    summary:
      "A deterministic linear distance penalty avoids a learned position table and keeps the rule defined at longer lengths.",
    colorClass: "bg-amber-400",
    weights: [1, 0.82, 0.67, 0.55, 0.45, 0.37],
  },
  rope: {
    label: "RoPE",
    location: "Rotates query and key vectors before their dot product.",
    extrapolation: "Relative offsets in attention",
    formula: "<R_m q, R_n k> depends on n - m",
    summary:
      "Rotary position embeddings apply 2D rotations to query/key blocks, preserving norms while making dot products depend on relative offsets.",
    colorClass: "bg-rose-400",
    weights: [1, 0.91, 0.72, 0.59, 0.46, 0.34],
  },
} as const;

type PositionStrategyId = keyof typeof positionStrategies;

const attentionModes = {
  full: {
    label: "Full",
    cost: "n^2 pairs",
    summary:
      "Every token can directly compare with every other token in one layer. It is expressive but expensive for long context.",
    visibleIndexes: [0, 1, 2, 3, 4, 5, 6],
  },
  sliding: {
    label: "Sliding window",
    cost: "n * w pairs",
    summary:
      "Each token sees a local neighborhood. Long-range influence can still travel through stacked layers, like a growing receptive field.",
    visibleIndexes: [2, 3, 4],
  },
  global: {
    label: "Local + global",
    cost: "local plus anchors",
    summary:
      "Most tokens use local windows, while special global tokens such as [CLS] can connect broadly for document-level signals.",
    visibleIndexes: [0, 2, 3, 4, 6],
  },
  gqa: {
    label: "GQA",
    cost: "shared K/V groups",
    summary:
      "Grouped-query attention keeps multiple query heads but shares key/value projections inside groups to shrink KV cache cost.",
    visibleIndexes: [0, 1, 2, 3, 4, 5, 6],
  },
  mqa: {
    label: "MQA",
    cost: "one shared K/V set",
    summary:
      "Multi-query attention is the extreme sharing case: all heads share keys and values while queries remain head-specific.",
    visibleIndexes: [0, 1, 2, 3, 4, 5, 6],
  },
} as const;

type AttentionModeId = keyof typeof attentionModes;

const modelFamilies = [
  {
    id: "encoder-decoder",
    label: "Encoder-decoder",
    title: "T5-style text-to-text",
    body: "The encoder reads the input, the decoder writes the output, and cross-attention connects the two. This keeps the original transformer translation shape and supports text-to-text tasks.",
    examples: "T5, mT5, ByT5",
  },
  {
    id: "encoder-only",
    label: "Encoder-only",
    title: "BERT-style representations",
    body: "Only the encoder stack remains. Unmasked bidirectional attention learns contextual embeddings that are useful for classification, retrieval, and token-level labeling.",
    examples: "BERT, DistilBERT, RoBERTa",
  },
  {
    id: "decoder-only",
    label: "Decoder-only",
    title: "GPT-style generation",
    body: "The model uses causal self-attention and next-token prediction. It scales simply and is the dominant pattern for modern generative LLMs.",
    examples: "GPT series",
  },
] as const;

const bertTokens = [
  "[CLS]",
  "this",
  "teddy",
  "bear",
  "is",
  "so",
  "cute",
  "!",
  "[SEP]",
  "[PAD]",
  "[PAD]",
  "[PAD]",
] as const;

const bertLayers = {
  token: {
    label: "Token",
    summary:
      "WordPiece maps text to vocabulary entries. [CLS] is a sequence placeholder, [SEP] marks boundaries, and [PAD] fills a batch length.",
  },
  position: {
    label: "Position",
    summary:
      "Position embeddings are added so identical tokens at different locations do not look identical to the encoder.",
  },
  segment: {
    label: "Segment",
    summary:
      "Segment A/B embeddings mark sentence membership for paired inputs such as next sentence prediction.",
  },
  output: {
    label: "Output",
    summary:
      "After bidirectional self-attention, the [CLS] output can feed a sequence classifier, while token outputs can feed span or tag classifiers.",
  },
} as const;

type BertLayerId = keyof typeof bertLayers;

function TransformerUpgradeVisual() {
  const stages = [
    "Position signal",
    "Attention logits",
    "Normalization",
    "Attention pattern",
    "Model family",
  ];

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <p className="text-sm font-semibold uppercase tracking-wide text-sky-300">
        Lecture 2 map
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
      <div className="mt-5 grid grid-cols-5 gap-2 rounded-md border border-slate-800 bg-slate-950 p-3">
        {["Q", "K", "V", "LN", "MLM"].map((label, index) => (
          <div key={label} className="space-y-2 text-center">
            <div
              className={`mx-auto h-16 w-full rounded-sm ${
                index % 2 === 0 ? "bg-sky-400" : "bg-emerald-400"
              }`}
              style={{ opacity: 0.45 + index * 0.1 }}
            />
            <p className="text-xs font-semibold text-slate-300">{label}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function PositionAttentionBench() {
  const [activeId, setActiveId] = useState<PositionStrategyId>("rope");
  const active = positionStrategies[activeId];
  const normalizedWeights = useMemo(
    () => active.weights.map((weight) => Math.max(12, weight * 100)),
    [active],
  );

  return (
    <section
      data-testid="position-attention-bench"
      className="rounded-lg border border-slate-800 bg-slate-900 p-5"
    >
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-slate-50">
            Position and attention design bench
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
            Choose how the transformer sees order. The key question is where the
            position signal enters: before attention, inside the attention
            logits, or by rotating queries and keys.
          </p>
        </div>
        <div className="rounded-md border border-slate-800 bg-slate-950 px-3 py-2 text-sm font-semibold text-emerald-300">
          {active.extrapolation}
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        {Object.entries(positionStrategies).map(([id, strategy]) => (
          <button
            key={id}
            type="button"
            onClick={() => setActiveId(id as PositionStrategyId)}
            className={`rounded-md border px-3 py-2 text-sm font-semibold ${
              id === activeId
                ? "border-sky-400 bg-sky-400 text-slate-950"
                : "border-slate-700 text-slate-200 hover:border-slate-500"
            }`}
          >
            {strategy.label}
          </button>
        ))}
      </div>

      <div className="mt-5 grid gap-4 md:grid-cols-[0.95fr_1.05fr]">
        <div className="rounded-md border border-slate-800 bg-slate-950 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            Illustrative relative score by distance
          </p>
          <div className="mt-4 grid gap-3">
            {normalizedWeights.map((width, index) => (
              <div key={index} className="grid grid-cols-[4.5rem_1fr] gap-3">
                <span className="text-sm font-semibold text-slate-300">
                  {index === 0 ? "same" : `+/- ${index}`}
                </span>
                <div className="h-4 overflow-hidden rounded-full bg-slate-800">
                  <div
                    className={`h-full rounded-full ${active.colorClass}`}
                    style={{ width: `${width}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-md border border-slate-800 bg-slate-950 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-sky-300">
            Active design
          </p>
          <h3 className="mt-2 text-lg font-semibold text-slate-50">
            {active.label}
          </h3>
          <p role="status" className="mt-3 text-sm leading-6 text-slate-300">
            {active.summary}
          </p>
          <dl className="mt-4 grid gap-3 text-sm">
            <div className="rounded-md border border-slate-800 bg-slate-900 p-3">
              <dt className="font-semibold text-slate-100">Where it enters</dt>
              <dd className="mt-1 leading-6 text-slate-300">
                {active.location}
              </dd>
            </div>
            <div className="rounded-md border border-slate-800 bg-slate-900 p-3">
              <dt className="font-semibold text-slate-100">Short form</dt>
              <dd className="mt-1 font-mono text-xs leading-5 text-slate-300">
                {active.formula}
              </dd>
            </div>
          </dl>
        </div>
      </div>
    </section>
  );
}

function AttentionEfficiencyLab() {
  const [activeId, setActiveId] = useState<AttentionModeId>("sliding");
  const active = attentionModes[activeId];
  const tokens = ["[CLS]", "A", "cute", "teddy", "bear", "reads", "."];
  const queryIndex = 3;

  return (
    <section
      data-testid="attention-efficiency-lab"
      className="rounded-lg border border-slate-800 bg-slate-900 p-5"
    >
      <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-slate-50">
            Attention approximation lab
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
            Full attention compares all token pairs. Long-context models often
            restrict the pattern or share key/value projections to reduce the
            cost that dominates memory and inference.
          </p>
        </div>
        <div className="rounded-md border border-slate-800 bg-slate-950 px-3 py-2 text-sm font-semibold text-amber-300">
          Cost: {active.cost}
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        {Object.entries(attentionModes).map(([id, mode]) => (
          <button
            key={id}
            type="button"
            onClick={() => setActiveId(id as AttentionModeId)}
            className={`rounded-md border px-3 py-2 text-sm font-semibold ${
              id === activeId
                ? "border-emerald-300 bg-emerald-300 text-slate-950"
                : "border-slate-700 text-slate-200 hover:border-slate-500"
            }`}
          >
            {mode.label}
          </button>
        ))}
      </div>

      <div className="mt-5 rounded-md border border-slate-800 bg-slate-950 p-4">
        <div className="grid grid-cols-7 gap-2">
          {tokens.map((token, index) => {
            const canSee = (
              active.visibleIndexes as readonly number[]
            ).includes(index);
            const isQuery = index === queryIndex;
            return (
              <div key={`${token}-${index}`} className="space-y-2">
                <div
                  className={`flex h-16 items-center justify-center rounded-md border px-1 text-center text-xs font-semibold ${
                    isQuery
                      ? "border-amber-300 bg-amber-300 text-slate-950"
                      : canSee
                        ? "border-emerald-300 bg-emerald-950 text-emerald-100"
                        : "border-slate-800 bg-slate-900 text-slate-500"
                  }`}
                >
                  {token}
                </div>
                <p className="truncate text-center text-[11px] text-slate-400">
                  {isQuery ? "query" : canSee ? "visible" : "hidden"}
                </p>
              </div>
            );
          })}
        </div>
        <p role="status" className="mt-4 text-sm leading-6 text-slate-300">
          {active.summary}
        </p>
      </div>
    </section>
  );
}

function ModelFamilyRouter() {
  const [activeId, setActiveId] =
    useState<(typeof modelFamilies)[number]["id"]>("encoder-only");
  const active =
    modelFamilies.find((family) => family.id === activeId) ?? modelFamilies[1];

  return (
    <section
      data-testid="model-family-router"
      className="rounded-lg border border-slate-800 bg-slate-900 p-5"
    >
      <h2 className="text-xl font-semibold text-slate-50">
        Route the task to the right transformer family
      </h2>
      <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
        Lecture 2 names the three modern families by what parts of the original
        transformer they keep and what training task they naturally support.
      </p>

      <div className="mt-4 flex flex-wrap gap-2">
        {modelFamilies.map((family) => (
          <button
            key={family.id}
            type="button"
            onClick={() => setActiveId(family.id)}
            className={`rounded-md border px-3 py-2 text-sm font-semibold ${
              family.id === active.id
                ? "border-sky-400 bg-sky-400 text-slate-950"
                : "border-slate-700 text-slate-200 hover:border-slate-500"
            }`}
          >
            {family.label}
          </button>
        ))}
      </div>

      <div className="mt-5 grid gap-4 md:grid-cols-[0.8fr_1.2fr]">
        <div className="rounded-md border border-slate-800 bg-slate-950 p-4">
          <div className="grid gap-3">
            <FamilyBlock
              label="Encoder"
              active={active.id !== "decoder-only"}
            />
            <FamilyBlock
              label="Cross-attention"
              active={active.id === "encoder-decoder"}
            />
            <FamilyBlock
              label="Decoder"
              active={active.id !== "encoder-only"}
            />
          </div>
        </div>

        <div className="rounded-md border border-slate-800 bg-slate-950 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-emerald-300">
            {active.examples}
          </p>
          <h3 className="mt-2 text-lg font-semibold text-slate-50">
            {active.title}
          </h3>
          <p role="status" className="mt-3 text-sm leading-6 text-slate-300">
            {active.body}
          </p>
        </div>
      </div>
    </section>
  );
}

function FamilyBlock({ label, active }: { label: string; active: boolean }) {
  return (
    <div
      className={`rounded-md border px-3 py-3 text-sm font-semibold ${
        active
          ? "border-emerald-300 bg-emerald-950 text-emerald-100"
          : "border-slate-800 bg-slate-900 text-slate-500"
      }`}
    >
      {label}
    </div>
  );
}

function BertInputLab() {
  const [activeLayerId, setActiveLayerId] = useState<BertLayerId>("segment");
  const [task, setTask] = useState<"sequence" | "token">("sequence");
  const activeLayer = bertLayers[activeLayerId];

  return (
    <section
      data-testid="bert-input-lab"
      className="rounded-lg border border-slate-800 bg-slate-900 p-5"
    >
      <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-slate-50">
            BERT input and fine-tuning lab
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
            BERT keeps the encoder, builds a position- and segment-aware input,
            pretrains with proxy tasks, then adapts the resulting embeddings to
            sequence or token-level tasks.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => setTask("sequence")}
            className={`rounded-md border px-3 py-2 text-sm font-semibold ${
              task === "sequence"
                ? "border-violet-300 bg-violet-300 text-slate-950"
                : "border-slate-700 text-slate-200 hover:border-slate-500"
            }`}
          >
            Sequence task
          </button>
          <button
            type="button"
            onClick={() => setTask("token")}
            className={`rounded-md border px-3 py-2 text-sm font-semibold ${
              task === "token"
                ? "border-violet-300 bg-violet-300 text-slate-950"
                : "border-slate-700 text-slate-200 hover:border-slate-500"
            }`}
          >
            Token task
          </button>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        {Object.entries(bertLayers).map(([id, layer]) => (
          <button
            key={id}
            type="button"
            onClick={() => setActiveLayerId(id as BertLayerId)}
            className={`rounded-md border px-3 py-2 text-sm font-semibold ${
              id === activeLayerId
                ? "border-amber-300 bg-amber-300 text-slate-950"
                : "border-slate-700 text-slate-200 hover:border-slate-500"
            }`}
          >
            {layer.label}
          </button>
        ))}
      </div>

      <div className="mt-5 rounded-md border border-slate-800 bg-slate-950 p-4">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 md:grid-cols-6">
          {bertTokens.map((token, index) => {
            const isOutput =
              (task === "sequence" && token === "[CLS]") ||
              (task === "token" && !token.startsWith("["));
            return (
              <div
                key={`${token}-${index}`}
                className={`rounded-md border p-2 text-center ${
                  isOutput
                    ? "border-violet-300 bg-violet-950 text-violet-100"
                    : token === "[PAD]"
                      ? "border-slate-800 bg-slate-900 text-slate-500"
                      : "border-slate-700 bg-slate-900 text-slate-100"
                }`}
              >
                <p className="font-mono text-xs font-semibold">{token}</p>
                <p className="mt-2 text-[11px] text-slate-400">
                  {activeLayerId === "position"
                    ? `pos ${index}`
                    : activeLayerId === "segment"
                      ? index <= 8
                        ? "seg A"
                        : "seg B"
                      : activeLayerId === "output" && isOutput
                        ? "head"
                        : activeLayer.label}
                </p>
              </div>
            );
          })}
        </div>
        <p role="status" className="mt-4 text-sm leading-6 text-slate-300">
          {activeLayer.summary}{" "}
          {task === "sequence"
            ? "For sentiment classification, the [CLS] output is the usual compact sequence representation."
            : "For question answering or tagging, each non-special token output can receive its own start/end or label head."}
        </p>
      </div>
    </section>
  );
}

function BertPretrainingTable() {
  const rows = [
    {
      task: "MLM",
      construction:
        "Select 15% of tokens: 80% [MASK], 10% random, 10% unchanged.",
      teaches: "Use left and right context to predict missing content.",
    },
    {
      task: "NSP",
      construction: "Pair sentences: 50% consecutive, 50% not consecutive.",
      teaches:
        "Train a binary classifier on whether sentence B follows sentence A.",
    },
    {
      task: "Fine-tuning",
      construction: "Reuse pretrained weights and add a small task head.",
      teaches:
        "Adapt contextual embeddings with relatively little labeled data.",
    },
  ] as const;

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <h2 className="text-xl font-semibold text-slate-50">
        BERT&apos;s two-stage training recipe
      </h2>
      <div className="mt-4 grid gap-3">
        {rows.map((row) => (
          <div
            key={row.task}
            className="grid gap-3 rounded-md border border-slate-800 bg-slate-950 p-3 md:grid-cols-[6rem_1.1fr_1fr]"
          >
            <p className="font-semibold text-sky-300">{row.task}</p>
            <p className="text-sm leading-6 text-slate-300">
              {row.construction}
            </p>
            <p className="text-sm leading-6 text-slate-300">{row.teaches}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

export default function StanfordCME295Lecture2LearningPage({
  experience,
}: Props) {
  return (
    <main className="bg-slate-950 text-slate-50">
      <LearningHero
        eyebrow="Stanford CME295 Lecture 2"
        title="Tune the transformer upgrade knobs"
        summary="Start from the original transformer, then change how it sees position, stabilizes training, saves attention cost, and becomes BERT-style representation machinery."
        meta={`${experience.durationMinutes} min interactive prep / ${experience.level}`}
        outcomes={experience.outcomes}
        visual={<TransformerUpgradeVisual />}
      />

      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-10 md:py-12">
        <ProcessSteps
          title="The lecture's core arc"
          steps={[
            {
              title: "Put order back in",
              body: "Self-attention has direct token-to-token links, so transformers need explicit position information.",
            },
            {
              title: "Move position into attention",
              body: "Modern variants often inject relative distance where query-key scores are computed.",
            },
            {
              title: "Specialize the architecture",
              body: "Encoder-decoder, encoder-only, and decoder-only models keep different parts of the original transformer for different jobs.",
            },
          ]}
        />

        <section className="grid gap-4 md:grid-cols-3">
          <ConceptCard title="The invariant" label="Attention still rules">
            <p>
              The central computation remains query-key scores, softmax weights,
              and weighted values. Most lecture 2 tricks change the inputs,
              logits, normalization, or head sharing around that core.
            </p>
          </ConceptCard>
          <ConceptCard title="The pressure" label="Length and cost">
            <p>
              Longer context stresses both position extrapolation and the
              quadratic cost of full self-attention. That motivates ALiBi, RoPE,
              sparse attention, and K/V sharing.
            </p>
          </ConceptCard>
          <ConceptCard title="The split" label="Model families">
            <p>
              The original encoder-decoder design is no longer the only default:
              BERT keeps encoders for representations, while GPT-style systems
              keep decoders for generation.
            </p>
          </ConceptCard>
        </section>

        <FormulaBlock
          title="Sinusoidal position encodings"
          formula={String.raw`\[\begin{aligned}\operatorname{PE}_{(m,2i)}&=\sin\left(m/10000^{2i/d_{\text{model}}}\right)\\\operatorname{PE}_{(m,2i+1)}&=\cos\left(m/10000^{2i/d_{\text{model}}}\right)\end{aligned}\]`}
          explanation="Different dimensions oscillate at different frequencies. The trigonometric structure makes dot products between position vectors depend on relative offsets, and the formula can be evaluated beyond the training length."
        />

        <PositionAttentionBench />

        <CheckForUnderstanding
          testId="position-extrapolation-check"
          title="Check: position extrapolation"
          question="A transformer was trained with learned absolute position embeddings up to length 512. At inference it receives 800 tokens. What is the main issue?"
          correctIndex={1}
          options={[
            {
              label:
                "The self-attention formula stops working because softmax cannot normalize more than 512 entries.",
              explanation:
                "Softmax can normalize a longer vector. The problem is the learned position table, not softmax itself.",
            },
            {
              label:
                "There are no learned position vectors for positions beyond the trained table unless the model is extended or retrained.",
              explanation:
                "Correct. Learned absolute tables are tied to the maximum positions represented during training.",
            },
            {
              label:
                "The model automatically converts learned absolute positions into ALiBi slopes.",
              explanation:
                "That conversion is not automatic. ALiBi is a different design choice.",
            },
          ]}
        />

        <FormulaBlock
          title="Relative bias changes the attention logits"
          formula={String.raw`\[\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+B_{m-n}\right)V\]`}
          explanation="T5 learns bucketed relative biases. ALiBi uses a deterministic linear distance penalty. Both act before softmax, so the final probabilities still sum to one."
        />

        <FormulaBlock
          title="RoPE makes query-key similarity relative"
          formula={String.raw`\[\langle R_m q, R_n k\rangle = q^\top R_{n-m}k\]`}
          explanation="RoPE rotates query and key vectors by position. The dot product after rotation can be expressed through the relative offset, while rotation preserves vector length."
        />

        <MisconceptionCallout
          misconception="Position encodings are only an input-layer detail."
          correction="That was true for the original absolute-addition picture, but lecture 2 emphasizes why many modern methods push position directly into attention scores or query/key geometry."
        />

        <InteractiveComparison
          title="Normalization choices in the block"
          prompt="Tap each variant and locate why modern transformers changed the default."
          items={[
            {
              id: "post-ln",
              label: "Post-LN",
              title: "Normalize after the residual addition",
              body: "The original transformer used normalization after adding the sublayer result back to the residual stream.",
            },
            {
              id: "pre-ln",
              label: "Pre-LN",
              title: "Normalize before attention or feed-forward",
              body: "Pre-norm improves stability for deep stacks because gradients have a cleaner residual path through the network.",
            },
            {
              id: "rms",
              label: "RMSNorm",
              title: "Scale by root mean square",
              body: "RMSNorm removes mean subtraction and often omits a bias term, reducing parameters while keeping comparable behavior in practice.",
            },
          ]}
        />

        <AttentionEfficiencyLab />

        <WorkedExample
          title="Worked example: why sliding windows still mix long context"
          setup="Suppose each layer lets a token attend only two positions left and right."
          steps={[
            "In one layer, token 10 cannot directly read token 2; its visible neighborhood is local.",
            "After the next layer, token 10 can receive information that token 8 already gathered from token 6.",
            "Stacking layers expands the effective receptive field, but this is slower than one full-attention layer for direct global access.",
          ]}
        />

        <ModelFamilyRouter />

        <CheckForUnderstanding
          testId="model-family-check"
          title="Check: model family"
          question="Which family best matches bidirectional sentence classification with a [CLS] representation?"
          correctIndex={0}
          options={[
            {
              label: "Encoder-only, BERT-style.",
              explanation:
                "Correct. Encoder-only models use bidirectional attention to create contextual sequence representations for classification.",
            },
            {
              label: "Decoder-only, GPT-style causal generation.",
              explanation:
                "Decoder-only models are strong generators, but causal masking is not the BERT-style bidirectional setup.",
            },
            {
              label:
                "Encoder-decoder, only if every label is translated into French first.",
              explanation:
                "Encoder-decoder models can solve many text-to-text tasks, but this is not the direct BERT-style classification pattern.",
            },
          ]}
        />

        <BertInputLab />

        <BertPretrainingTable />

        <FormulaBlock
          title="Distillation matches soft distributions"
          formula={String.raw`\[D_{\mathrm{KL}}(p_T\Vert p_S)=\sum_i p_T(i)\log\frac{p_T(i)}{p_S(i)}\]`}
          explanation="DistilBERT compresses a teacher by training a smaller student to match probability distributions, not just hard argmax labels."
        />

        <InteractiveComparison
          title="BERT variants: what changed?"
          prompt="These variants are easy to confuse because they preserve much of the encoder-only shape."
          items={[
            {
              id: "bert",
              label: "BERT",
              title: "Encoder-only pretraining with MLM and NSP",
              body: "BERT uses WordPiece, [CLS], [SEP], token/position/segment embeddings, masked language modeling, and next sentence prediction before fine-tuning.",
            },
            {
              id: "distilbert",
              label: "DistilBERT",
              title: "Smaller and faster through distillation",
              body: "DistilBERT reduces layers and learns from a teacher distribution, retaining most BERT performance with lower latency.",
            },
            {
              id: "roberta",
              label: "RoBERTa",
              title: "Same basic architecture, better pretraining recipe",
              body: "RoBERTa removes NSP, uses dynamic masking, and trains with more data and schedule changes rather than inventing a new attention mechanism.",
            },
          ]}
        />

        <CheckForUnderstanding
          testId="bert-objective-check"
          title="Check: BERT objectives"
          question="What does masked language modeling encourage in an encoder-only model?"
          correctIndex={2}
          options={[
            {
              label:
                "It forces the model to predict only the next token using left context.",
              explanation:
                "That is the decoder-only causal language-model pattern, not BERT's bidirectional MLM setup.",
            },
            {
              label:
                "It replaces tokenization, so WordPiece is no longer needed.",
              explanation:
                "MLM operates after tokenization. BERT still needs tokens such as WordPiece pieces and special markers.",
            },
            {
              label:
                "It makes the model use surrounding left and right context to recover selected tokens.",
              explanation:
                "Correct. Unmasked encoder attention lets the representation use both sides of the selected token.",
            },
          ]}
        />

        <RecapSection
          title="Before you start the MCQs"
          items={[
            "Self-attention needs explicit order information because token comparisons are not sequential by default.",
            "Learned absolute positions are flexible but tied to a trained maximum length.",
            "Sinusoidal encodings use many frequencies and can be evaluated at unseen positions.",
            "T5 bias, ALiBi, and RoPE put relative position closer to the query-key score computation.",
            "Pre-norm and RMSNorm are modern stability and efficiency choices around residual blocks.",
            "Sparse attention and sliding windows trade direct global connectivity for lower long-context cost.",
            "GQA and MQA reduce KV cache cost by sharing key/value projections.",
            "BERT is encoder-only, bidirectional, pretrained with MLM/NSP, and fine-tuned through [CLS] or token outputs.",
          ]}
        />

        <section className="flex flex-col gap-4 rounded-lg border border-emerald-500/40 bg-emerald-950/20 p-5 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-xl font-semibold text-emerald-100">
              Ready for the Stanford CME295 Lecture 2 questions
            </h2>
            <p className="mt-2 text-sm leading-6 text-slate-300">
              Practice position embeddings, RoPE, normalization, sparse and
              shared attention, model families, BERT pretraining, DistilBERT,
              and RoBERTa.
            </p>
          </div>
          <QuizTransitionButton sourceId={experience.sourceId} />
        </section>

        <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
          <h2 className="text-lg font-semibold text-slate-50">
            Compact formula board
          </h2>
          <div className="mt-4 space-y-3 text-slate-200">
            <MathText
              text={String.raw`\[\operatorname{softmax}(QK^\top/\sqrt{d_k}+B)V\]`}
              className="max-w-full overflow-x-auto"
            />
            <MathText
              text={String.raw`\[\langle R_mq,R_nk\rangle=q^\top R_{n-m}k\]`}
              className="max-w-full overflow-x-auto"
            />
            <MathText
              text={String.raw`\[\text{token}+\text{position}+\text{segment}\rightarrow\text{BERT encoder}\]`}
              className="max-w-full overflow-x-auto"
            />
          </div>
        </section>
      </div>
    </main>
  );
}
