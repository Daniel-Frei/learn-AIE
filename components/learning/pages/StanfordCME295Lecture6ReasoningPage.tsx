"use client";

import Link from "next/link";
import { useMemo, useState, type ReactNode } from "react";
import {
  BadgeCheck,
  BrainCircuit,
  Calculator,
  CheckCircle2,
  ClipboardCheck,
  Code2,
  Gauge,
  GitBranch,
  Layers3,
  LineChart,
  Route,
  Scale,
  SlidersHorizontal,
  Sparkles,
  Split,
  Timer,
  Workflow,
  XCircle,
} from "lucide-react";
import MathText from "../../MathText";
import { QuizTransitionButton } from "../LearningPrimitives";
import {
  getBudgetEstimate,
  getCombination,
  getConsistencyWinner,
  getGroupRelativeAdvantages,
  getIndependentPassAtK,
  getLengthPenaltyRatio,
  getSampledPassAtK,
  type BudgetProfile,
} from "./cme295-lecture6/reasoningMath";

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function formatNumber(value: number, digits = 2): string {
  return value.toFixed(digits);
}

function InlineMath({ text }: { text: string }) {
  return <MathText inline text={text} />;
}

function IconBadge({
  icon,
  label,
  tone,
}: {
  icon: ReactNode;
  label: string;
  tone: "blue" | "green" | "amber" | "rose" | "violet";
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

function LabButton({
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

const reasoningTraceModes = {
  direct: {
    label: "Direct answer",
    output: "Answer: 5.",
    diagnosis:
      "Fast, cheap, and fine when the problem is trivial. It gives little evidence about the path used to reach the answer.",
  },
  chain: {
    label: "Step-by-step trace",
    output:
      "The current year is 2025. The bear was born in 2020. 2025 - 2020 = 5. Answer: 5.",
    diagnosis:
      "The model spends extra tokens to decompose the problem into intermediate steps before emitting the final answer.",
  },
  hidden: {
    label: "Hidden trace with summary",
    output:
      "Thought summary: subtract the birth year from the current year. Answer: 5.",
    diagnosis:
      "Many reasoning products expose a short summary rather than the full raw chain, while still charging or budgeting the hidden reasoning tokens.",
  },
} as const;

type ReasoningTraceModeId = keyof typeof reasoningTraceModes;

function ReasoningPrimer() {
  const [modeId, setModeId] = useState<ReasoningTraceModeId>("chain");
  const mode = reasoningTraceModes[modeId];

  return (
    <section
      data-testid="reasoning-primer"
      className="border-y border-slate-200 bg-[#f8fafc] text-slate-950"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.85fr_1.15fr]">
        <div className="min-w-0 space-y-4">
          <IconBadge
            icon={<BrainCircuit aria-hidden="true" size={18} />}
            label="Reasoning setup"
            tone="blue"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Reasoning means solving through intermediate steps
          </h2>
          <p className="text-base leading-7 text-slate-700">
            A reasoning problem is not just a lookup. The model must transform
            the prompt through a sequence of smaller steps, keep the state
            consistent, and then produce the final answer. Chain-of-thought
            prompting made that pattern visible; reasoning models scale it by
            allocating inference-time tokens to the intermediate work.
          </p>
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-lg border border-slate-300 bg-white p-4">
              <p className="font-semibold">Coding</p>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                A solution can be checked with tests.
              </p>
            </div>
            <div className="rounded-lg border border-slate-300 bg-white p-4">
              <p className="font-semibold">Math</p>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                A final answer can be compared with a known result.
              </p>
            </div>
            <div className="rounded-lg border border-slate-300 bg-white p-4">
              <p className="font-semibold">Benchmarks</p>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                Pass@k and Cons@k ask how reliably samples solve the task.
              </p>
            </div>
          </div>
        </div>

        <div className="min-w-0 rounded-xl border border-slate-300 bg-white p-4">
          <p className="text-sm font-semibold text-slate-500">Prompt</p>
          <p className="mt-2 text-lg font-semibold text-slate-950">
            A bear was born in 2020. How old is it in 2025?
          </p>
          <div className="mt-4 grid gap-2 sm:grid-cols-3">
            {(Object.keys(reasoningTraceModes) as ReasoningTraceModeId[]).map(
              (id) => (
                <LabButton
                  key={id}
                  tone="blue"
                  isActive={modeId === id}
                  onClick={() => setModeId(id)}
                >
                  {reasoningTraceModes[id].label}
                </LabButton>
              ),
            )}
          </div>
          <div className="mt-4 rounded-lg border border-slate-300 bg-slate-50 p-4">
            <p className="text-sm font-semibold text-slate-500">Model output</p>
            <p className="mt-2 text-sm leading-6 text-slate-800">
              {mode.output}
            </p>
          </div>
          <p
            role="status"
            className="mt-4 rounded-lg border border-blue-500 bg-blue-50 p-4 text-sm leading-6 text-blue-950"
          >
            {mode.diagnosis}
          </p>
        </div>
      </div>
    </section>
  );
}

const capabilityPrompts = {
  lookup: {
    label: "Course-code lookup",
    prompt: "What is the course code for Stanford's transformers class?",
    correct: "Knowledge lookup",
    explanation:
      "The bottleneck is stored or retrieved knowledge. Extra reasoning tokens do not create the missing fact.",
  },
  arithmetic: {
    label: "Bear age",
    prompt: "A bear was born in 2020. How old is it in 2025?",
    correct: "Reasoning",
    explanation:
      "The model must carry out a small multi-step computation: identify the current year, subtract, and state the age.",
  },
  action: {
    label: "Place an order",
    prompt: "Order more printer paper for the lab before tomorrow.",
    correct: "Tool needed",
    explanation:
      "A language model can draft or decide, but real ordering requires acting through tools or an external environment.",
  },
  ambiguous: {
    label: "Current result",
    prompt: "Who won the election held last week?",
    correct: "Knowledge or tool",
    explanation:
      "This may require fresh retrieval before reasoning. The bottleneck is not only multi-step problem solving.",
  },
} as const;

type CapabilityPromptId = keyof typeof capabilityPrompts;

const capabilityCategories = [
  "Knowledge lookup",
  "Reasoning",
  "Tool needed",
  "Knowledge or tool",
] as const;

function CapabilityBottleneckLab() {
  const [promptId, setPromptId] = useState<CapabilityPromptId>("arithmetic");
  const [category, setCategory] = useState<string | null>(null);
  const prompt = capabilityPrompts[promptId];
  const isCorrect = category === prompt.correct;

  return (
    <section className="bg-white text-slate-950">
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 space-y-4">
          <IconBadge
            icon={<Split aria-hidden="true" size={18} />}
            label="Capability bottleneck"
            tone="blue"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            First decide what kind of failure you are fixing
          </h2>
          <p className="text-base leading-7 text-slate-700">
            Separate vanilla LLM weaknesses before choosing a fix. Reasoning
            models target problems that need intermediate steps; they do not
            automatically repair stale knowledge, missing tools, or evaluation
            ambiguity.
          </p>
          <div className="grid gap-2">
            {(Object.keys(capabilityPrompts) as CapabilityPromptId[]).map(
              (id) => (
                <LabButton
                  key={id}
                  isActive={promptId === id}
                  onClick={() => {
                    setPromptId(id);
                    setCategory(null);
                  }}
                >
                  {capabilityPrompts[id].label}
                </LabButton>
              ),
            )}
          </div>
        </div>

        <div
          data-testid="capability-bottleneck-lab"
          className="min-w-0 rounded-xl border border-slate-300 bg-slate-50 p-4"
        >
          <div className="rounded-lg border border-slate-300 bg-white p-4">
            <p className="text-sm font-semibold text-slate-500">Prompt</p>
            <p className="mt-2 text-lg font-semibold text-slate-950">
              {prompt.prompt}
            </p>
          </div>

          <div className="mt-4 grid gap-2 sm:grid-cols-2">
            {capabilityCategories.map((candidate) => (
              <LabButton
                key={candidate}
                tone="green"
                isActive={category === candidate}
                onClick={() => setCategory(candidate)}
              >
                {candidate}
              </LabButton>
            ))}
          </div>

          {category && (
            <p
              role="status"
              className={`mt-4 rounded-lg border p-4 text-sm leading-6 ${
                isCorrect
                  ? "border-emerald-500 bg-emerald-50 text-emerald-950"
                  : "border-amber-500 bg-amber-50 text-amber-950"
              }`}
            >
              <span className="font-semibold">
                {isCorrect ? "Correct bottleneck." : "Try again."}
              </span>{" "}
              {prompt.explanation}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

const benchmarkAnswerSets = {
  stable: {
    label: "Stable majority",
    answers: ["42", "42", "41", "42", "42"],
    note: "Cons@k is useful when independent samples mostly agree on the same final answer.",
  },
  split: {
    label: "Split vote",
    answers: ["A", "B", "A", "B", "C"],
    note: "A tie or weak plurality means consensus is not strong evidence of correctness.",
  },
  brittle: {
    label: "Brittle coding",
    answers: ["pass", "fail", "fail", "fail", "pass"],
    note: "Pass@k can be high while Pass@1 remains weak if one lucky sample passes tests.",
  },
} as const;

type BenchmarkAnswerSetId = keyof typeof benchmarkAnswerSets;

const benchmarkSamplePools = {
  balanced: {
    label: "10 samples / 4 pass",
    totalSamples: 10,
    correctSamples: 4,
    note: "A realistic evaluation pool: some generated solutions pass, many fail.",
  },
  brittle: {
    label: "10 samples / 2 pass",
    totalSamples: 10,
    correctSamples: 2,
    note: "A brittle model can look acceptable with high k while still being weak at Pass@1.",
  },
  strong: {
    label: "20 samples / 12 pass",
    totalSamples: 20,
    correctSamples: 12,
    note: "A stronger model improves both single-sample reliability and multi-attempt success.",
  },
} as const;

type BenchmarkSamplePoolId = keyof typeof benchmarkSamplePools;

function BenchmarkLab() {
  const [samplePoolId, setSamplePoolId] =
    useState<BenchmarkSamplePoolId>("balanced");
  const [attempts, setAttempts] = useState(4);
  const [answerSetId, setAnswerSetId] =
    useState<BenchmarkAnswerSetId>("stable");
  const samplePool = benchmarkSamplePools[samplePoolId];
  const sampledPassAtK = getSampledPassAtK({
    totalSamples: samplePool.totalSamples,
    correctSamples: samplePool.correctSamples,
    attempts,
  });
  const singlePassRate = samplePool.correctSamples / samplePool.totalSamples;
  const independentApproximation = getIndependentPassAtK(
    singlePassRate,
    attempts,
  );
  const failedSamples = samplePool.totalSamples - samplePool.correctSamples;
  const failedCombinations = getCombination(failedSamples, attempts);
  const totalCombinations = getCombination(samplePool.totalSamples, attempts);
  const answerSet = benchmarkAnswerSets[answerSetId];
  const winner = getConsistencyWinner(answerSet.answers);

  return (
    <section
      data-testid="benchmark-lab"
      className="border-y border-slate-200 bg-[#f6f0e8] text-slate-950"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[1fr_1.05fr]">
        <div className="min-w-0 space-y-4">
          <IconBadge
            icon={<LineChart aria-hidden="true" size={18} />}
            label="Reasoning evaluation"
            tone="amber"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Pass@k answers a different question than Pass@1
          </h2>
          <p className="text-base leading-7 text-slate-700">
            Coding and math benchmarks often allow multiple attempts because
            tests or final answers can verify success. That makes Pass@k useful,
            but it can hide unreliable single-sample behavior.
          </p>
          <FormulaPanel
            title="Sample-count estimator"
            formula={String.raw`\[\mathrm{Pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\]`}
          >
            <span>
              Here <InlineMath text={String.raw`\(n\)`} /> is the number of
              generated samples, <InlineMath text={String.raw`\(c\)`} /> is the
              number that pass verification, and{" "}
              <InlineMath text={String.raw`\(k\)`} /> is how many attempts the
              benchmark lets you draw.
            </span>
          </FormulaPanel>
          <FormulaPanel
            title="Probability shortcut"
            formula={String.raw`\[\mathrm{Pass@}k \approx 1 - (1-p)^k\]`}
          >
            <span>
              This approximation is useful when you only know a single-attempt
              pass rate <InlineMath text={String.raw`\(p\)`} />. The lab uses
              the sample-count estimator because it matches the benchmark
              framing.
            </span>
          </FormulaPanel>
        </div>

        <div className="min-w-0 rounded-xl border border-amber-700/30 bg-white p-4">
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <p className="text-sm font-semibold text-slate-600">
                Generated sample pool
              </p>
              <div className="mt-2 grid gap-2">
                {(
                  Object.keys(benchmarkSamplePools) as BenchmarkSamplePoolId[]
                ).map((id) => (
                  <LabButton
                    key={id}
                    tone="amber"
                    isActive={samplePoolId === id}
                    onClick={() => setSamplePoolId(id)}
                  >
                    {benchmarkSamplePools[id].label}
                  </LabButton>
                ))}
              </div>
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-600">Attempts</p>
              <div className="mt-2 grid gap-2">
                {[1, 4, 8].map((value) => (
                  <LabButton
                    key={value}
                    tone="amber"
                    isActive={attempts === value}
                    onClick={() => setAttempts(value)}
                  >
                    k={value}
                  </LabButton>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <Metric label="Pass@1" value={formatPercent(singlePassRate)} />
            <Metric
              label={`Pass@${attempts}`}
              value={formatPercent(sampledPassAtK)}
            />
            <Metric label="Latency work" value={`${attempts}x`} />
          </div>

          <p
            role="status"
            className="mt-4 rounded-lg border border-amber-500 bg-amber-50 p-4 text-sm leading-6 text-amber-950"
          >
            Calculation:{" "}
            <span className="font-mono">
              1 - C({failedSamples}, {attempts}) / C(
              {samplePool.totalSamples}, {attempts}) = 1 - {failedCombinations}{" "}
              / {totalCombinations} = {formatNumber(sampledPassAtK)}
            </span>
            . The independent-rate shortcut would give{" "}
            <span className="font-mono">
              {formatNumber(independentApproximation)}
            </span>
            , so label the formula you are using. {samplePool.note}
          </p>

          <div className="mt-5 border-t border-slate-200 pt-4">
            <p className="text-sm font-semibold text-slate-600">
              Cons@k sample set
            </p>
            <p className="mt-2 text-sm leading-6 text-slate-600">
              Cons@k samples several answers, takes the majority-vote final
              answer, then checks that consensus against the ground truth.
            </p>
            <div className="mt-2 flex flex-wrap gap-2">
              {(Object.keys(benchmarkAnswerSets) as BenchmarkAnswerSetId[]).map(
                (id) => (
                  <LabButton
                    key={id}
                    tone="blue"
                    isActive={answerSetId === id}
                    onClick={() => setAnswerSetId(id)}
                  >
                    {benchmarkAnswerSets[id].label}
                  </LabButton>
                ),
              )}
            </div>
            <div className="mt-4 grid gap-2 sm:grid-cols-5">
              {answerSet.answers.map((answer, index) => (
                <div
                  key={`${answer}-${index}`}
                  className="rounded-md border border-slate-300 bg-slate-50 p-3 text-center"
                >
                  <p className="text-xs font-semibold text-slate-500">
                    sample {index + 1}
                  </p>
                  <p className="mt-1 font-mono text-lg font-semibold">
                    {answer}
                  </p>
                </div>
              ))}
            </div>
            {winner && (
              <p className="mt-4 text-sm leading-6 text-slate-700">
                Consensus result:{" "}
                <span className="font-semibold text-slate-950">
                  {winner.answer}
                </span>{" "}
                with {winner.votes} votes
                {winner.tied ? ", tied with another answer" : ""}.{" "}
                {answerSet.note}
              </p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

const budgetPrompts = {
  easy: {
    label: "Simple lookup",
    prompt: "What is the course code?",
    profile: {
      baseQuality: 0.72,
      maxGain: 0.1,
      saturationTokens: 350,
      usefulTokenLimit: 700,
      overthinkPenaltyPerToken: 0.00012,
    } satisfies BudgetProfile,
    guidance:
      "Extra thinking mostly adds latency because the problem is not reasoning-bound.",
  },
  medium: {
    label: "Word problem",
    prompt: "How old is the bear in 2025 if it was born in 2020?",
    profile: {
      baseQuality: 0.48,
      maxGain: 0.32,
      saturationTokens: 650,
      usefulTokenLimit: 1300,
      overthinkPenaltyPerToken: 0.00004,
    } satisfies BudgetProfile,
    guidance:
      "A moderate budget lets the model decompose the problem without turning the answer into a long essay.",
  },
  hard: {
    label: "Olympiad-style proof",
    prompt:
      "Solve a multi-step proof where a wrong intermediate step breaks the final answer.",
    profile: {
      baseQuality: 0.18,
      maxGain: 0.62,
      saturationTokens: 1150,
      usefulTokenLimit: 2400,
      overthinkPenaltyPerToken: 0.000025,
    } satisfies BudgetProfile,
    guidance:
      "This is where test-time scaling has the most room to help, especially when verification is available.",
  },
} as const;

type BudgetPromptId = keyof typeof budgetPrompts;

function ThinkingBudgetLab() {
  const [promptId, setPromptId] = useState<BudgetPromptId>("hard");
  const [tokens, setTokens] = useState(1200);
  const prompt = budgetPrompts[promptId];
  const estimate = getBudgetEstimate(prompt.profile, tokens);

  return (
    <section
      data-testid="reasoning-budget-lab"
      className="bg-slate-950 text-slate-50"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="min-w-0 space-y-4">
          <IconBadge
            icon={<Timer aria-hidden="true" size={18} />}
            label="Test-time scaling"
            tone="green"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Spend thinking tokens where they can change the answer
          </h2>
          <p className="text-base leading-7 text-slate-300">
            Reasoning is extra inference-time compute: generate intermediate
            steps before the final answer. A good product does not use the same
            budget for every prompt.
          </p>
          <FormulaPanel
            title="Toy budget curve"
            formula={String.raw`\[\mathrm{quality} = \mathrm{base} + \mathrm{useful\ gain} - \mathrm{overthinking\ penalty}\]`}
          >
            <span>
              The curve saturates because easy gains come first. Past the useful
              limit, more tokens can waste time or distract from a concise final
              answer.
            </span>
          </FormulaPanel>
        </div>

        <div className="min-w-0 rounded-xl border border-emerald-300/30 bg-slate-900 p-4">
          <div className="grid gap-3 sm:grid-cols-3">
            {(Object.keys(budgetPrompts) as BudgetPromptId[]).map((id) => (
              <button
                key={id}
                type="button"
                aria-pressed={promptId === id}
                onClick={() => setPromptId(id)}
                className={`rounded-lg border p-3 text-left transition-colors ${
                  promptId === id
                    ? "border-emerald-300 bg-emerald-300 text-slate-950"
                    : "border-slate-700 bg-slate-950 text-slate-200 hover:border-slate-500"
                }`}
              >
                <span className="block text-sm font-semibold">
                  {budgetPrompts[id].label}
                </span>
                <span className="mt-1 block text-xs leading-5">
                  {budgetPrompts[id].prompt}
                </span>
              </button>
            ))}
          </div>

          <label className="mt-5 block">
            <span className="text-sm font-semibold text-slate-300">
              Reasoning-token budget: {tokens}
            </span>
            <input
              type="range"
              min={0}
              max={3000}
              step={300}
              value={tokens}
              onChange={(event) => setTokens(Number(event.target.value))}
              className="mt-3 w-full accent-emerald-300"
            />
          </label>

          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <Metric
              label="Estimated quality"
              value={formatPercent(estimate.quality)}
              detail="toy score"
            />
            <Metric
              label="Useful gain"
              value={`+${formatPercent(estimate.usefulGain)}`}
            />
            <Metric
              label="Latency multiplier"
              value={`${formatNumber(estimate.latencyMultiplier, 1)}x`}
            />
          </div>

          <p
            role="status"
            className="mt-4 rounded-lg border border-emerald-300/40 bg-emerald-400/10 p-4 text-sm leading-6 text-emerald-100"
          >
            {prompt.label}: {prompt.guidance} Current arithmetic is{" "}
            <span className="font-mono">
              {formatPercent(prompt.profile.baseQuality)} +{" "}
              {formatPercent(estimate.usefulGain)} -{" "}
              {formatPercent(estimate.overthinkPenalty)} ={" "}
              {formatPercent(estimate.quality)}
            </span>
            .
          </p>
        </div>
      </div>
    </section>
  );
}

const rewardTasks = {
  code: {
    label: "Coding task",
    icon: <Code2 aria-hidden="true" size={18} />,
    formatReward: 0.15,
    accuracyReward: 0.85,
    verifier: "Unit tests pass or fail.",
    limit: "Tests may miss hidden requirements or reward hard-coded shortcuts.",
  },
  math: {
    label: "Math answer",
    icon: <Calculator aria-hidden="true" size={18} />,
    formatReward: 0.2,
    accuracyReward: 0.8,
    verifier: "The final numeric answer matches a known solution.",
    limit: "A correct final answer does not guarantee the trace is faithful.",
  },
  open: {
    label: "Open-ended advice",
    icon: <ClipboardCheck aria-hidden="true" size={18} />,
    formatReward: 0.25,
    accuracyReward: 0,
    verifier:
      "Formatting can be checked, but correctness is not directly verifiable.",
    limit:
      "This needs preference models, human review, or downstream evaluation.",
  },
} as const;

type RewardTaskId = keyof typeof rewardTasks;

function VerifiableRewardLab() {
  const [taskId, setTaskId] = useState<RewardTaskId>("code");
  const task = rewardTasks[taskId];
  const totalReward = task.formatReward + task.accuracyReward;

  return (
    <section
      data-testid="verifiable-reward-lab"
      className="border-y border-slate-200 bg-white text-slate-950"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 space-y-4">
          <IconBadge
            icon={<BadgeCheck aria-hidden="true" size={18} />}
            label="Verifiable reward"
            tone="green"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Reward the trace format and the checkable final result
          </h2>
          <p className="text-base leading-7 text-slate-700">
            DeepSeek-style reasoning training avoids hand-writing every chain of
            thought by using rewards that can be checked automatically. The
            reliable part is narrow: format tags and verifiable answers.
          </p>
        </div>

        <div className="min-w-0 rounded-xl border border-slate-300 bg-slate-50 p-4">
          <div className="grid gap-3 sm:grid-cols-3">
            {(Object.keys(rewardTasks) as RewardTaskId[]).map((id) => (
              <LabButton
                key={id}
                tone="green"
                isActive={taskId === id}
                onClick={() => setTaskId(id)}
              >
                <span className="inline-flex items-center gap-2">
                  {rewardTasks[id].icon}
                  {rewardTasks[id].label}
                </span>
              </LabButton>
            ))}
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <Metric
              label="Formatting reward"
              value={formatNumber(task.formatReward, 2)}
              detail="Has required reasoning and answer blocks"
            />
            <Metric
              label="Accuracy reward"
              value={formatNumber(task.accuracyReward, 2)}
              detail={task.verifier}
            />
            <Metric
              label="Total reward"
              value={formatNumber(totalReward, 2)}
              detail="toy additive reward"
            />
          </div>

          <div className="mt-5 grid gap-4 lg:grid-cols-[1fr_0.9fr]">
            <div className="rounded-lg border border-slate-300 bg-white p-4">
              <p className="text-sm font-semibold text-slate-500">
                Training signal
              </p>
              <div className="mt-3 space-y-2 font-mono text-sm">
                <p className="rounded-md bg-slate-950 px-3 py-2 text-emerald-200">
                  &lt;think&gt; reasoning trace &lt;/think&gt;
                </p>
                <p className="rounded-md bg-slate-950 px-3 py-2 text-blue-200">
                  &lt;answer&gt; final answer &lt;/answer&gt;
                </p>
              </div>
            </div>
            <p
              role="status"
              className="rounded-lg border border-amber-500 bg-amber-50 p-4 text-sm leading-6 text-amber-950"
            >
              Verifier: {task.verifier} Limit: {task.limit} Verifiable reward
              makes reasoning RL scalable, but it does not make all reasoning
              tasks automatically measurable.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

const grpoResponses = [
  {
    id: "concise-correct",
    label: "Concise correct",
    reward: 1,
    tokens: 80,
    note: "Good final answer with enough trace to justify it.",
  },
  {
    id: "long-correct",
    label: "Long correct",
    reward: 0.85,
    tokens: 260,
    note: "Correct, but spends many more tokens than needed.",
  },
  {
    id: "format-only",
    label: "Format only",
    reward: 0.25,
    tokens: 120,
    note: "Uses the requested tags but fails the verifier.",
  },
  {
    id: "long-wrong",
    label: "Wrong but long",
    reward: -0.2,
    tokens: 420,
    note: "A fluent trace that reaches a wrong answer should not be rewarded for length.",
  },
] as const;

type GrpoResponseId = (typeof grpoResponses)[number]["id"];

function GrpoGroupLab() {
  const [selectedId, setSelectedId] =
    useState<GrpoResponseId>("concise-correct");
  const selectedIndex = grpoResponses.findIndex(
    (response) => response.id === selectedId,
  );
  const selected = grpoResponses[selectedIndex] ?? grpoResponses[0];
  const stats = useMemo(
    () => getGroupRelativeAdvantages(grpoResponses.map((item) => item.reward)),
    [],
  );
  const selectedAdvantage = stats.advantages[selectedIndex] ?? 0;

  return (
    <section
      data-testid="grpo-group-lab"
      className="bg-[#13201d] text-slate-50"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 space-y-4">
          <IconBadge
            icon={<Scale aria-hidden="true" size={18} />}
            label="GRPO"
            tone="violet"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            GRPO grades a response against its sampled group
          </h2>
          <p className="text-base leading-7 text-slate-300">
            Group Relative Policy Optimization keeps PPO-style update control,
            but it avoids a separate value model by estimating advantage from
            the rewards of sibling samples for the same prompt.
          </p>
          <FormulaPanel
            title="Group-relative advantage"
            formula={String.raw`\[A_i = r_i - \bar{r}_{\mathrm{group}}\]`}
          >
            <span>
              Positive <InlineMath text={String.raw`\(A_i\)`} /> means this
              sample is better than the group average. Negative{" "}
              <InlineMath text={String.raw`\(A_i\)`} /> means the policy should
              reduce its probability, subject to ratio clipping and KL pressure.
            </span>
          </FormulaPanel>
          <FormulaPanel
            title="GRPO objective skeleton"
            formula={String.raw`\[J_{\mathrm{GRPO}}(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\left(r_{i,t}A_i,\mathrm{clip}(r_{i,t},1-\epsilon,1+\epsilon)A_i\right)-\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})\right]\]`}
          >
            <span>
              The core pieces mirror PPO-style control: a policy ratio{" "}
              <InlineMath text={String.raw`\(r_{i,t}\)`} />, a clipped update
              band, an advantage estimate, and a KL penalty against a reference
              model.
            </span>
          </FormulaPanel>
        </div>

        <div className="min-w-0 rounded-xl border border-emerald-300/30 bg-slate-950 p-4">
          <div className="grid gap-2 sm:grid-cols-2">
            {grpoResponses.map((response) => (
              <button
                key={response.id}
                type="button"
                aria-pressed={selected.id === response.id}
                onClick={() => setSelectedId(response.id)}
                className={`rounded-lg border p-3 text-left transition-colors ${
                  selected.id === response.id
                    ? "border-violet-300 bg-violet-300 text-slate-950"
                    : "border-slate-700 bg-slate-900 text-slate-200 hover:border-slate-500"
                }`}
              >
                <span className="block font-semibold">{response.label}</span>
                <span className="mt-1 block text-sm">
                  reward {formatNumber(response.reward, 2)} / {response.tokens}{" "}
                  tokens
                </span>
              </button>
            ))}
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <Metric
              label="Group average"
              value={formatNumber(stats.averageReward, 2)}
            />
            <Metric
              label="Selected reward"
              value={formatNumber(selected.reward, 2)}
            />
            <Metric
              label="Advantage"
              value={formatNumber(selectedAdvantage, 2)}
            />
          </div>

          <p
            role="status"
            className={`mt-4 rounded-lg border p-4 text-sm leading-6 ${
              selectedAdvantage >= 0
                ? "border-emerald-300/40 bg-emerald-400/10 text-emerald-100"
                : "border-rose-300/40 bg-rose-400/10 text-rose-100"
            }`}
          >
            {selected.label}: {selected.note}{" "}
            <span className="font-mono">
              {formatNumber(selected.reward, 2)} -{" "}
              {formatNumber(stats.averageReward, 2)} ={" "}
              {formatNumber(selectedAdvantage, 2)}
            </span>
            .
          </p>

          <div className="mt-5 rounded-lg border border-slate-800 bg-slate-900 p-4">
            <p className="text-sm font-semibold text-slate-300">
              PPO comparison
            </p>
            <div className="mt-3 grid gap-3 md:grid-cols-2">
              <p className="rounded-md border border-slate-700 bg-slate-950 p-3 text-sm leading-6 text-slate-300">
                PPO uses a value model or value head as a baseline for the
                advantage estimate.
              </p>
              <p className="rounded-md border border-slate-700 bg-slate-950 p-3 text-sm leading-6 text-slate-300">
                GRPO uses sibling completions for the same prompt, then still
                needs update control so reward optimization does not drift.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

const policyOptimizationModes = {
  ppo: {
    label: "PPO",
    title: "PPO uses a learned value baseline",
    formula: String.raw`\[A_t = R_t - V_\psi(s_t)\]`,
    component:
      "Policy model, reference or old policy, reward model, and value model or value head.",
    update:
      "The policy ratio is clipped so a high-reward token cannot jump too far in one update.",
    tradeoff:
      "The value model can reduce variance, but it adds another learned component that must be trained and kept stable.",
  },
  grpo: {
    label: "GRPO",
    title: "GRPO uses sibling samples as the baseline",
    formula: String.raw`\[A_i = r_i - \frac{1}{G}\sum_{j=1}^{G}r_j\]`,
    component:
      "Policy model, reference policy, reward checks, and a group of sampled completions.",
    update:
      "The policy ratio is still clipped, but the advantage comes from how one completion scores against its group.",
    tradeoff:
      "Avoiding the value model simplifies the pipeline, but reward design, sample diversity, KL pressure, and length incentives still matter.",
  },
} as const;

type PolicyOptimizationModeId = keyof typeof policyOptimizationModes;

function PpoGrpoComparisonLab() {
  const [modeId, setModeId] = useState<PolicyOptimizationModeId>("grpo");
  const mode = policyOptimizationModes[modeId];

  return (
    <section
      data-testid="ppo-grpo-comparison-lab"
      className="border-y border-slate-200 bg-[#f9f7f1] text-slate-950"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 space-y-4">
          <IconBadge
            icon={<Scale aria-hidden="true" size={18} />}
            label="PPO versus GRPO"
            tone="violet"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            PPO and GRPO share update control but differ in the baseline
          </h2>
          <p className="text-base leading-7 text-slate-700">
            Both methods compare the current policy to an older or reference
            policy, clip large probability-ratio changes, and use KL/reference
            pressure to avoid destroying useful base behavior. The big practical
            difference is how the advantage is estimated.
          </p>
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-lg border border-slate-300 bg-white p-4">
              <p className="font-semibold">Shared ratio</p>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                <InlineMath
                  text={String.raw`\(r=\pi_\theta/\pi_{\mathrm{old}}\)`}
                />{" "}
                measures how much the policy changed.
              </p>
            </div>
            <div className="rounded-lg border border-slate-300 bg-white p-4">
              <p className="font-semibold">Shared clipping</p>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                The update is bounded near{" "}
                <InlineMath text={String.raw`\(1\pm\epsilon\)`} />.
              </p>
            </div>
            <div className="rounded-lg border border-slate-300 bg-white p-4">
              <p className="font-semibold">Shared anchor</p>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                KL/reference pressure discourages policy drift.
              </p>
            </div>
          </div>
        </div>

        <div className="min-w-0 rounded-xl border border-slate-300 bg-white p-4">
          <div className="flex flex-wrap gap-2">
            {(
              Object.keys(policyOptimizationModes) as PolicyOptimizationModeId[]
            ).map((id) => (
              <LabButton
                key={id}
                tone="violet"
                isActive={modeId === id}
                onClick={() => setModeId(id)}
              >
                {policyOptimizationModes[id].label}
              </LabButton>
            ))}
          </div>

          <div className="mt-5 rounded-lg border border-slate-300 bg-slate-50 p-4">
            <h3 className="text-xl font-semibold">{mode.title}</h3>
            <MathText
              text={mode.formula}
              className="mt-3 max-w-full overflow-x-auto rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-950"
            />
            <div className="mt-4 grid gap-3 md:grid-cols-3">
              <Metric
                label="Components"
                value={mode.label}
                detail={mode.component}
              />
              <Metric
                label="Update control"
                value="ratio + clip"
                detail={mode.update}
              />
              <Metric
                label="Tradeoff"
                value="baseline choice"
                detail={mode.tradeoff}
              />
            </div>
          </div>

          <p
            role="status"
            className="mt-4 rounded-lg border border-violet-500 bg-violet-50 p-4 text-sm leading-6 text-violet-950"
          >
            {mode.label}: {mode.tradeoff}
          </p>
        </div>
      </div>
    </section>
  );
}

function LengthIncentiveLab() {
  const [mode, setMode] = useState<"standard" | "equalized">("standard");
  const shortTokens = 40;
  const longTokens = 200;
  const ratio = getLengthPenaltyRatio(shortTokens, longTokens);

  return (
    <section
      data-testid="length-incentive-lab"
      className="border-y border-slate-200 bg-[#f2f7f4] text-slate-950"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="min-w-0 space-y-4">
          <IconBadge
            icon={<Gauge aria-hidden="true" size={18} />}
            label="Length pathology"
            tone="rose"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Longer reasoning can become a reward exploit
          </h2>
          <p className="text-base leading-7 text-slate-700">
            A subtle RL pathology appears when token contribution is normalized
            per output: a bad short answer can receive a stronger token-level
            penalty than a bad long answer. That creates pressure toward
            verbosity, not necessarily better reasoning.
          </p>
          <FormulaPanel
            title="Per-output length normalization"
            formula={String.raw`\[\mathrm{token\ weight}_i = \frac{1}{|o_i|}\]`}
          >
            <span>
              Equalizing token-level contributions, as in DAPO-style or Dr.
              GRPO-style fixes, changes the incentive so length is not a hiding
              place for low-quality reasoning.
            </span>
          </FormulaPanel>
        </div>

        <div className="min-w-0 rounded-xl border border-slate-300 bg-white p-4">
          <div className="flex flex-wrap gap-2">
            <LabButton
              tone="rose"
              isActive={mode === "standard"}
              onClick={() => setMode("standard")}
            >
              Standard length factor
            </LabButton>
            <LabButton
              tone="green"
              isActive={mode === "equalized"}
              onClick={() => setMode("equalized")}
            >
              Equalized contribution
            </LabButton>
          </div>

          <div className="mt-5 grid gap-4 md:grid-cols-2">
            <div className="rounded-lg border border-rose-300 bg-rose-50 p-4">
              <p className="text-sm font-semibold text-rose-900">
                Bad short trace
              </p>
              <p className="mt-2 text-3xl font-semibold">{shortTokens}</p>
              <p className="text-sm text-rose-900">tokens</p>
            </div>
            <div className="rounded-lg border border-blue-300 bg-blue-50 p-4">
              <p className="text-sm font-semibold text-blue-900">
                Bad long trace
              </p>
              <p className="mt-2 text-3xl font-semibold">{longTokens}</p>
              <p className="text-sm text-blue-900">tokens</p>
            </div>
          </div>

          <p
            role="status"
            className="mt-5 rounded-lg border border-slate-300 bg-slate-50 p-4 text-sm leading-6 text-slate-700"
          >
            {mode === "standard" ? (
              <>
                With the standard factor, each bad short-trace token can count{" "}
                <span className="font-mono">{formatNumber(ratio, 1)}x</span> as
                much as a bad long-trace token. That can make long wrong answers
                less painful per token.
              </>
            ) : (
              <>
                With equalized token contribution, the diagnostic target is
                reasoning quality, not simply the length bucket that produced
                the token.
              </>
            )}
          </p>
        </div>
      </div>
    </section>
  );
}

const r1Recipes = {
  zero: {
    label: "R1-Zero",
    headline: "Pure RL proof of concept",
    summary:
      "Start from a pretrained base model and run GRPO with reasoning data, formatting rewards, and answer-verification rewards.",
    steps: [
      "V3-Base pretrained model",
      "GRPO on reasoning prompts",
      "Formatting reward for think/answer structure",
      "Accuracy reward from verifiable answers",
    ],
    result:
      "Reasoning behavior emerges, but traces can have formatting, readability, and language-mixing issues.",
  },
  r1: {
    label: "R1",
    headline: "Full usability pipeline",
    summary:
      "Add a small reasoning SFT cold start, then GRPO, larger SFT with reasoning and general data, and a final helpfulness/harmlessness RL stage.",
    steps: [
      "Cold-start SFT from cleaned R1-Zero traces",
      "GRPO with reasoning rewards and language consistency",
      "Large SFT using rejection-sampled reasoning plus general data",
      "Final GRPO for reasoning and assistant behavior",
    ],
    result:
      "The model keeps the reasoning gains while improving readability, formatting, and general assistant usefulness.",
  },
} as const;

type R1RecipeId = keyof typeof r1Recipes;

function R1PipelineLab() {
  const [recipeId, setRecipeId] = useState<R1RecipeId>("zero");
  const recipe = r1Recipes[recipeId];

  return (
    <section data-testid="r1-pipeline-lab" className="bg-white text-slate-950">
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.85fr_1.15fr]">
        <div className="min-w-0 space-y-4">
          <IconBadge
            icon={<Workflow aria-hidden="true" size={18} />}
            label="R1 recipe"
            tone="blue"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            R1-Zero proves the signal; R1 turns it into a usable assistant
          </h2>
          <p className="text-base leading-7 text-slate-700">
            Pure RL can discover reasoning, but a product-grade reasoning model
            also needs cleaned traces, general assistant data, and
            preference-style helpfulness and harmlessness.
          </p>
          <div className="flex flex-wrap gap-2">
            {(Object.keys(r1Recipes) as R1RecipeId[]).map((id) => (
              <LabButton
                key={id}
                tone="blue"
                isActive={recipeId === id}
                onClick={() => setRecipeId(id)}
              >
                {r1Recipes[id].label}
              </LabButton>
            ))}
          </div>
        </div>

        <div className="min-w-0 rounded-xl border border-slate-300 bg-slate-50 p-4">
          <div className="rounded-lg border border-slate-300 bg-white p-4">
            <p className="text-sm font-semibold text-blue-700">
              {recipe.label}
            </p>
            <h3 className="mt-2 text-2xl font-semibold">{recipe.headline}</h3>
            <p role="status" className="mt-2 text-sm leading-6 text-slate-700">
              {recipe.summary}
            </p>
          </div>

          <div className="mt-4 grid gap-3">
            {recipe.steps.map((step, index) => (
              <div
                key={step}
                className="grid grid-cols-[auto_1fr] gap-3 rounded-lg border border-slate-300 bg-white p-4"
              >
                <span className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-600 text-sm font-bold text-white">
                  {index + 1}
                </span>
                <p className="text-sm font-semibold leading-6 text-slate-800">
                  {step}
                </p>
              </div>
            ))}
          </div>

          <p className="mt-4 rounded-lg border border-emerald-500 bg-emerald-50 p-4 text-sm leading-6 text-emerald-950">
            Outcome: {recipe.result}
          </p>
        </div>
      </div>
    </section>
  );
}

function DistillationLab() {
  const [mode, setMode] = useState<"ordinary" | "reasoning">("reasoning");

  return (
    <section
      data-testid="distillation-lab"
      className="border-y border-slate-200 bg-[#171525] text-slate-50"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 space-y-4">
          <IconBadge
            icon={<Layers3 aria-hidden="true" size={18} />}
            label="Distillation"
            tone="violet"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Reasoning distillation transfers traces, not just probabilities
          </h2>
          <p className="text-base leading-7 text-slate-300">
            Earlier distillation matched the next-token distribution from a
            teacher. In the R1 setting, a large reasoning teacher generates
            whole reasoning traces and answers, then a smaller student learns
            those sequences with supervised fine-tuning.
          </p>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              aria-pressed={mode === "ordinary"}
              onClick={() => setMode("ordinary")}
              className={`rounded-md border px-3 py-2 text-sm font-semibold ${
                mode === "ordinary"
                  ? "border-violet-300 bg-violet-300 text-slate-950"
                  : "border-slate-700 bg-slate-950 text-slate-200"
              }`}
            >
              Ordinary distillation
            </button>
            <button
              type="button"
              aria-pressed={mode === "reasoning"}
              onClick={() => setMode("reasoning")}
              className={`rounded-md border px-3 py-2 text-sm font-semibold ${
                mode === "reasoning"
                  ? "border-violet-300 bg-violet-300 text-slate-950"
                  : "border-slate-700 bg-slate-950 text-slate-200"
              }`}
            >
              Reasoning distillation
            </button>
          </div>
        </div>

        <div className="min-w-0 rounded-xl border border-violet-300/30 bg-slate-950 p-4">
          <div className="grid gap-3 md:grid-cols-3">
            <div className="rounded-lg border border-slate-700 bg-slate-900 p-4">
              <BrainCircuit aria-hidden="true" className="text-violet-200" />
              <h3 className="mt-3 font-semibold">Teacher</h3>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                Large reasoning model generates either probability targets or
                full trace-and-answer examples.
              </p>
            </div>
            <div className="rounded-lg border border-slate-700 bg-slate-900 p-4">
              <GitBranch aria-hidden="true" className="text-emerald-200" />
              <h3 className="mt-3 font-semibold">Training object</h3>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                {mode === "ordinary"
                  ? "Match the teacher distribution over the next token."
                  : "Fit the generated reasoning trace and final answer as an SFT sequence."}
              </p>
            </div>
            <div className="rounded-lg border border-slate-700 bg-slate-900 p-4">
              <Sparkles aria-hidden="true" className="text-amber-200" />
              <h3 className="mt-3 font-semibold">Student</h3>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                {mode === "ordinary"
                  ? "Learns smoother token predictions from the teacher."
                  : "Learns a compact version of teacher-style reasoning behavior."}
              </p>
            </div>
          </div>
          <p role="status" className="mt-4 text-sm leading-6 text-violet-100">
            {mode === "ordinary"
              ? "Ordinary distillation transfers distributional knowledge. It does not automatically create a worked reasoning trace dataset."
              : "Reasoning distillation spends expensive teacher compute offline so smaller models can imitate high-quality trace-and-answer examples at inference time."}
          </p>
        </div>
      </div>
    </section>
  );
}

function ReasoningMisconceptionCheck() {
  const [selected, setSelected] = useState<"more" | "verify" | null>(null);
  const isCorrect = selected === "verify";

  return (
    <section className="bg-[#f8fafc] text-slate-950">
      <div className="mx-auto grid w-full max-w-6xl gap-6 px-4 py-10 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="min-w-0 rounded-xl border border-slate-300 bg-white p-5">
          <div className="flex items-center gap-3">
            <SlidersHorizontal aria-hidden="true" className="text-blue-700" />
            <h2 className="text-2xl font-semibold">Reasoning control policy</h2>
          </div>
          <p className="mt-3 text-sm leading-6 text-slate-700">
            Use a reasoning model when the bottleneck is multi-step problem
            solving, when verification is possible, or when multiple attempts
            can be afforded. Use a faster path for simple lookup, low-stakes
            drafting, or tasks where stale knowledge or tool access is the real
            blocker.
          </p>
        </div>

        <div
          data-testid="reasoning-misconception-check"
          className="min-w-0 rounded-xl border border-slate-300 bg-white p-5"
        >
          <h2 className="text-xl font-semibold">
            Which policy matches this reasoning setup?
          </h2>
          <div className="mt-4 grid gap-2">
            <LabButton
              tone="rose"
              isActive={selected === "more"}
              onClick={() => setSelected("more")}
            >
              Always force extended thinking because more tokens mean better
              reasoning.
            </LabButton>
            <LabButton
              tone="green"
              isActive={selected === "verify"}
              onClick={() => setSelected("verify")}
            >
              Spend more thinking on hard, verifiable tasks and control budget
              elsewhere.
            </LabButton>
          </div>
          {selected && (
            <p
              role="status"
              className={`mt-4 rounded-lg border p-4 text-sm leading-6 ${
                isCorrect
                  ? "border-emerald-500 bg-emerald-50 text-emerald-950"
                  : "border-amber-500 bg-amber-50 text-amber-950"
              }`}
            >
              <span className="font-semibold">
                {isCorrect ? "Correct." : "Not quite."}
              </span>{" "}
              More thinking is a tradeoff. It buys compute for reasoning, but it
              also increases latency, cost, and sometimes verbosity.
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

function HeroVisual() {
  return (
    <div
      aria-label="Reasoning model request trace"
      className="min-w-0 rounded-xl border border-slate-700 bg-slate-900 p-4 shadow-2xl shadow-black/30"
    >
      <div className="rounded-lg border border-blue-400/30 bg-blue-400/10 p-4">
        <p className="text-sm font-semibold text-blue-100">Prompt</p>
        <p className="mt-2 text-sm leading-6 text-slate-200">
          Solve the problem, show enough reasoning to verify the path, then give
          the final answer.
        </p>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        {[
          {
            icon: <BrainCircuit aria-hidden="true" size={20} />,
            title: "Think",
            body: "Generate intermediate steps.",
          },
          {
            icon: <CheckCircle2 aria-hidden="true" size={20} />,
            title: "Verify",
            body: "Reward format and final answer.",
          },
          {
            icon: <Route aria-hidden="true" size={20} />,
            title: "Control",
            body: "Budget tokens, attempts, and RL pressure.",
          },
        ].map((item) => (
          <div
            key={item.title}
            className="rounded-lg border border-slate-700 bg-slate-950 p-3"
          >
            <div className="flex items-center gap-2 text-emerald-200">
              {item.icon}
              <h3 className="font-semibold">{item.title}</h3>
            </div>
            <p className="mt-2 text-sm leading-5 text-slate-400">{item.body}</p>
          </div>
        ))}
      </div>

      <div className="mt-4 rounded-lg border border-emerald-400/30 bg-emerald-400/10 p-4">
        <div className="flex items-center gap-2 text-emerald-100">
          <XCircle aria-hidden="true" size={18} />
          <p className="font-semibold">Failure to avoid</p>
        </div>
        <p className="mt-2 text-sm leading-6 text-slate-300">
          Optimizing for long traces, lucky Pass@k samples, or formatting alone
          instead of verified reasoning quality.
        </p>
      </div>
    </div>
  );
}

export default function StanfordCME295Lecture6ReasoningPage() {
  return (
    <main className="min-h-screen bg-slate-950 text-slate-50">
      <section className="border-b border-slate-800 bg-[#0d1720]">
        <div className="mx-auto grid min-h-[560px] w-full max-w-6xl items-center gap-8 px-4 py-10 md:py-14 lg:grid-cols-[0.95fr_1.05fr]">
          <div className="min-w-0 space-y-6">
            <Link
              href="/learn/stanford-cme295"
              className="inline-flex text-sm font-semibold text-blue-200 hover:text-blue-100"
            >
              Back to Stanford CME295 course
            </Link>
            <div className="flex flex-wrap gap-2">
              <IconBadge
                icon={<BrainCircuit aria-hidden="true" size={18} />}
                label="Stanford CME295"
                tone="blue"
              />
              <IconBadge
                icon={<Gauge aria-hidden="true" size={18} />}
                label="Reasoning models"
                tone="green"
              />
            </div>
            <div className="space-y-4">
              <h1 className="max-w-3xl text-3xl font-semibold text-slate-50 md:text-5xl md:leading-tight">
                Run a reasoning model like a controlled experiment
              </h1>
              <p className="max-w-2xl text-base leading-7 text-slate-300 md:text-lg">
                Build the reasoning-model mental model by controlling when the
                model thinks, how attempts are evaluated, which rewards are
                verifiable, how GRPO compares sibling outputs, and why R1-style
                training needs more than long chains of thought.
              </p>
              <p className="text-sm font-semibold text-emerald-300">
                Reasoning quiz source available
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <QuizTransitionButton
                sourceId="cme295-lect6"
                label="Start reasoning questions"
              />
              <a
                href="#reasoning-bench"
                className="inline-flex items-center justify-center rounded-lg border border-slate-600 px-4 py-3 text-sm font-bold text-slate-100 transition-colors hover:border-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-300 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950"
              >
                Open reasoning bench
              </a>
              <a
                href="#r1-pipeline"
                className="inline-flex items-center justify-center rounded-lg border border-slate-600 px-4 py-3 text-sm font-bold text-slate-100 transition-colors hover:border-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-300 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950"
              >
                Compare R1 recipes
              </a>
            </div>
          </div>

          <HeroVisual />
        </div>
      </section>

      <ReasoningPrimer />
      <CapabilityBottleneckLab />
      <div id="reasoning-bench" className="scroll-mt-20">
        <BenchmarkLab />
      </div>
      <ThinkingBudgetLab />
      <VerifiableRewardLab />
      <GrpoGroupLab />
      <PpoGrpoComparisonLab />
      <LengthIncentiveLab />
      <div id="r1-pipeline" className="scroll-mt-20">
        <R1PipelineLab />
      </div>
      <DistillationLab />
      <ReasoningMisconceptionCheck />

      <section className="border-t border-slate-800 bg-[#0d1720] text-slate-50">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-5 px-4 py-10 md:flex-row md:items-center md:justify-between">
          <div className="max-w-2xl">
            <h2 className="text-2xl font-semibold">Reasoning recap</h2>
            <p className="mt-2 text-sm leading-6 text-slate-300">
              Keep four distinctions active: reasoning is not fresh knowledge or
              action, Pass@k is not Pass@1, verifiable reward is narrow, and
              GRPO/R1-style training must control both policy drift and length
              incentives.
            </p>
          </div>
          <Link
            href="/learn/stanford-cme295"
            className="inline-flex items-center justify-center rounded-lg border border-slate-600 px-4 py-3 text-sm font-bold text-slate-100 transition-colors hover:border-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-300 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950"
          >
            Back to course
          </Link>
          <QuizTransitionButton sourceId="cme295-lect6" />
        </div>
      </section>
    </main>
  );
}
