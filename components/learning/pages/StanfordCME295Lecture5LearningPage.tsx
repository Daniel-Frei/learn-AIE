"use client";

import { useMemo, useState, type ReactNode } from "react";
import {
  Activity,
  ArrowRight,
  BadgeCheck,
  Bot,
  BrainCircuit,
  Calculator,
  Database,
  Gauge,
  GitCompareArrows,
  Lock,
  MessageSquareText,
  RefreshCcw,
  Route,
  Scale,
  ShieldAlert,
  Sigma,
  SlidersHorizontal,
  Target,
  Trophy,
  Workflow,
} from "lucide-react";
import type { LearningExperience } from "../../../lib/learning";
import MathText from "../../MathText";
import {
  CheckForUnderstanding,
  QuizTransitionButton,
} from "../LearningPrimitives";
import {
  getBestOfNStats,
  getBradleyTerryStats,
  getDpoLogitStats,
  getPpoClipStats,
} from "./cme295-lecture5/preferenceMath";

type Props = {
  experience: LearningExperience;
};

const preferenceModes = {
  pointwise: {
    label: "Pointwise",
    short: "One response gets one scalar score.",
    observation: "score(A)=0.4, score(B)=0.9",
    lesson:
      "This can work, but human raters often disagree about what an absolute 0.9 means.",
  },
  pairwise: {
    label: "Pairwise",
    short: "Two responses to the same prompt are compared.",
    observation: "B preferred over A",
    lesson:
      "The lecture emphasizes this format because choosing the better of two outputs is easier than writing the perfect output from scratch.",
  },
  listwise: {
    label: "Listwise",
    short: "Several responses are ranked together.",
    observation: "B > C > A > D",
    lesson:
      "This carries rich information, but it is more work for raters and can be decomposed into many pairwise comparisons.",
  },
} as const;

type PreferenceModeId = keyof typeof preferenceModes;

const rewardScenarios = {
  clear: {
    label: "Clear win",
    winnerReward: 1.8,
    loserReward: -0.4,
    note: "The preferred answer is much more likely under the reward model.",
  },
  narrow: {
    label: "Close call",
    winnerReward: 0.7,
    loserReward: 0.3,
    note: "The label says which response won, but the reward gap says confidence is modest.",
  },
  tied: {
    label: "Equal rewards",
    winnerReward: 0,
    loserReward: 0,
    note: "Equal reward scores produce a 0.50 preference probability.",
  },
} as const;

type RewardScenarioId = keyof typeof rewardScenarios;

const ppoScenarios = {
  positive: {
    label: "Positive advantage",
    currentProbability: 0.3,
    oldProbability: 0.2,
    advantage: 2,
    epsilon: 0.2,
    note: "The sampled token did better than expected, but PPO clips the extra incentive after the ratio exceeds 1.2.",
  },
  negative: {
    label: "Negative advantage",
    currentProbability: 0.1,
    oldProbability: 0.2,
    advantage: -2,
    epsilon: 0.2,
    note: "The token did worse than expected, but PPO still limits how far probability can drop in one update.",
  },
  gentle: {
    label: "Inside trust range",
    currentProbability: 0.22,
    oldProbability: 0.2,
    advantage: 1.2,
    epsilon: 0.2,
    note: "A small ratio change stays inside the clipping band.",
  },
} as const;

type PpoScenarioId = keyof typeof ppoScenarios;

const dpoScenarios = {
  lecture: {
    label: "Policy favors winner",
    policyChosenLogProb: -5,
    policyRejectedLogProb: -7,
    referenceChosenLogProb: -6,
    referenceRejectedLogProb: -6.5,
    beta: 0.1,
    note: "The trainable policy has increased the chosen-over-rejected gap relative to the frozen reference.",
  },
  anchored: {
    label: "Reference already agrees",
    policyChosenLogProb: -5.2,
    policyRejectedLogProb: -6.2,
    referenceChosenLogProb: -5.3,
    referenceRejectedLogProb: -6.5,
    beta: 0.1,
    note: "The policy likes the winner, but not more than the reference did, so the DPO contrast is negative.",
  },
  failure: {
    label: "Policy favors rejected",
    policyChosenLogProb: -7.2,
    policyRejectedLogProb: -5.9,
    referenceChosenLogProb: -6.4,
    referenceRejectedLogProb: -6.2,
    beta: 0.1,
    note: "The preference logit goes below zero because the tuned policy moved toward the rejected answer.",
  },
} as const;

type DpoScenarioId = keyof typeof dpoScenarios;

const decisionScenarios = {
  heavyTraffic: {
    label: "Heavy traffic serving",
    method: "DPO or PPO",
    diagnosis:
      "Best-of-N multiplies generation at inference. If traffic is high, prefer a tuned policy that pays cost during training rather than every request.",
  },
  quickIteration: {
    label: "Quick preference prototype",
    method: "DPO",
    diagnosis:
      "DPO is attractive when you have chosen/rejected pairs and want a supervised-style objective without reward and value models.",
  },
  maxControl: {
    label: "Maximum RL control",
    method: "PPO-style RLHF",
    diagnosis:
      "PPO-style RLHF can use on-policy samples, reward models, value baselines, clipping, and KL penalties, but it is harder to tune.",
  },
  noTrainingWindow: {
    label: "No training window",
    method: "Best-of-N",
    diagnosis:
      "Best-of-N can improve selection without changing the generator, but it depends on candidate diversity and reward-model quality.",
  },
} as const;

type DecisionScenarioId = keyof typeof decisionScenarios;

function formatNumber(value: number, digits = 2): string {
  return value.toFixed(digits);
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function formatOperand(value: number, digits = 2): string {
  const formatted = formatNumber(value, digits);
  return value < 0 ? `(${formatted})` : formatted;
}

function InlineMath({
  text,
  className = "",
}: {
  text: string;
  className?: string;
}) {
  return <MathText inline text={text} className={className} />;
}

function FormulaStep({
  label,
  expression,
  note,
}: {
  label: string;
  expression: string;
  note?: ReactNode;
}) {
  return (
    <div className="rounded-md border border-slate-800 bg-slate-950 p-3">
      <p className="text-xs font-semibold uppercase tracking-[0.08em] text-slate-500">
        {label}
      </p>
      <p className="mt-2 break-words font-mono text-sm font-semibold text-slate-100">
        {expression}
      </p>
      {note && <p className="mt-2 text-sm leading-5 text-slate-400">{note}</p>}
    </div>
  );
}

function IconBadge({
  icon,
  label,
  tone = "sky",
}: {
  icon: ReactNode;
  label: string;
  tone?: "sky" | "emerald" | "amber" | "rose";
}) {
  const toneClasses = {
    sky: "border-sky-500 bg-sky-100 text-sky-950",
    emerald: "border-emerald-500 bg-emerald-100 text-emerald-950",
    amber: "border-amber-500 bg-amber-100 text-amber-950",
    rose: "border-rose-500 bg-rose-100 text-rose-950",
  };

  return (
    <span
      className={`inline-flex items-center gap-2 rounded-md border px-3 py-2 text-sm font-semibold ${toneClasses[tone]}`}
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
      className={`rounded-md border px-3 py-2 text-left text-sm font-semibold transition-colors ${
        isActive
          ? "border-emerald-300 bg-emerald-300 text-slate-950"
          : "border-slate-700 bg-slate-950 text-slate-200 hover:border-slate-500"
      }`}
    >
      {children}
    </button>
  );
}

function MetricCell({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail?: string;
}) {
  return (
    <div className="min-w-0 rounded-md border border-slate-800 bg-slate-950 p-3">
      <p className="text-sm text-slate-400">{label}</p>
      <p className="mt-1 font-mono text-xl font-semibold text-slate-50">
        {value}
      </p>
      {detail && (
        <p className="mt-1 text-sm leading-5 text-slate-400">{detail}</p>
      )}
    </div>
  );
}

function AlignmentTraceVisual() {
  return (
    <div
      aria-label="Preference tuning trace"
      className="min-w-0 rounded-lg border border-rose-300/30 bg-slate-900 p-4 shadow-2xl shadow-black/30"
    >
      <div className="rounded-md border border-slate-700 bg-slate-950 p-3">
        <p className="text-sm font-semibold text-rose-200">Prompt</p>
        <p className="mt-2 text-sm leading-6 text-slate-200">
          Can I put my teddy bear in the washer?
        </p>
      </div>

      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <div className="rounded-md border border-amber-300/40 bg-amber-400/10 p-3">
          <p className="text-sm font-semibold text-amber-100">SFT answer</p>
          <p className="mt-2 text-sm leading-6 text-slate-200">
            No. It might get damaged. Try hand washing it instead.
          </p>
        </div>
        <div className="rounded-md border border-emerald-300/40 bg-emerald-400/10 p-3">
          <p className="text-sm font-semibold text-emerald-100">
            Preference-tuned answer
          </p>
          <p className="mt-2 text-sm leading-6 text-slate-200">
            It is better not to. Your teddy could get hurt. A gentle hand wash
            is safer.
          </p>
        </div>
      </div>

      <div className="mt-3 grid gap-3 sm:grid-cols-3">
        <div className="rounded-md border border-slate-700 bg-slate-950 p-3">
          <p className="text-xs font-semibold text-slate-400">Data</p>
          <p className="mt-1 text-sm text-slate-100">chosen / rejected pair</p>
        </div>
        <div className="rounded-md border border-slate-700 bg-slate-950 p-3">
          <p className="text-xs font-semibold text-slate-400">Signal</p>
          <p className="mt-1 text-sm text-slate-100">reward gap or log ratio</p>
        </div>
        <div className="rounded-md border border-slate-700 bg-slate-950 p-3">
          <p className="text-xs font-semibold text-slate-400">Guardrail</p>
          <p className="mt-1 text-sm text-slate-100">
            reference policy pressure
          </p>
        </div>
      </div>
    </div>
  );
}

function PreferenceDataStudio() {
  const [modeId, setModeId] = useState<PreferenceModeId>("pairwise");
  const mode = preferenceModes[modeId];

  return (
    <section
      id="workbench"
      data-testid="preference-data-studio"
      className="bg-slate-50 text-slate-950"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.85fr_1.15fr]">
        <div className="min-w-0 space-y-4">
          <IconBadge
            icon={<GitCompareArrows aria-hidden="true" size={18} />}
            label="Preference data"
            tone="rose"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Turn a vague complaint into a training example
          </h2>
          <p className="text-base leading-7 text-slate-700">
            Lecture 5 starts from a model that follows the task but misses the
            desired tone. Preference tuning does not require a perfect target
            response for every prompt. It can learn from comparative labels:
            this answer is preferred to that answer for the same prompt.
          </p>
          <div className="grid gap-2">
            {(Object.keys(preferenceModes) as PreferenceModeId[]).map((id) => (
              <button
                key={id}
                type="button"
                aria-pressed={modeId === id}
                onClick={() => setModeId(id)}
                className={`rounded-md border px-4 py-3 text-left transition-colors ${
                  modeId === id
                    ? "border-rose-500 bg-rose-100"
                    : "border-slate-300 bg-white hover:border-slate-500"
                }`}
              >
                <span className="block font-semibold">
                  {preferenceModes[id].label}
                </span>
                <span className="mt-1 block text-sm leading-5 text-slate-600">
                  {preferenceModes[id].short}
                </span>
              </button>
            ))}
          </div>
        </div>

        <div className="min-w-0 rounded-lg border border-slate-300 bg-white p-4">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <p className="text-sm font-semibold text-slate-500">Prompt</p>
            <p className="mt-1 text-base font-semibold">
              Suggest a new activity I could do with my teddy bear.
            </p>
          </div>

          <div className="mt-3 grid gap-3 md:grid-cols-2">
            <article className="rounded-md border border-amber-300 bg-amber-50 p-4">
              <p className="text-sm font-semibold text-amber-800">Response A</p>
              <p className="mt-2 text-sm leading-6 text-slate-700">
                I would suggest you do not spend much time with your teddy bear
                at all.
              </p>
            </article>
            <article className="rounded-md border border-emerald-300 bg-emerald-50 p-4">
              <p className="text-sm font-semibold text-emerald-800">
                Response B
              </p>
              <p className="mt-2 text-sm leading-6 text-slate-700">
                Of course. Teddy bears can be great buddies for fun activities,
                like watching a movie together.
              </p>
            </article>
          </div>

          <div className="mt-4 rounded-md border border-slate-200 bg-slate-950 p-4 text-slate-50">
            <div className="grid gap-3 sm:grid-cols-[0.7fr_1.3fr]">
              <MetricCell
                label="Active label"
                value={mode.observation}
                detail={mode.label}
              />
              <div className="rounded-md border border-slate-800 bg-slate-900 p-3">
                <p className="text-sm font-semibold text-sky-200">
                  What this label teaches
                </p>
                <p
                  role="status"
                  className="mt-2 text-sm leading-6 text-slate-300"
                >
                  {mode.lesson}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function AlignmentPipeline() {
  const steps = [
    {
      title: "Start policy",
      icon: <Bot aria-hidden="true" size={22} />,
      body: "A pretrained or SFT model already writes plausible completions, but its style and priorities are not yet aligned with human preference.",
    },
    {
      title: "Preference pairs",
      icon: <Database aria-hidden="true" size={22} />,
      body: (
        <>
          For one prompt, humans mark a chosen answer{" "}
          <InlineMath text={String.raw`\(y_w\)`} /> and a rejected answer{" "}
          <InlineMath text={String.raw`\(y_l\)`} />. The pair is easier to
          collect than a perfect target answer.
        </>
      ),
    },
    {
      title: "Training signal",
      icon: <Sigma aria-hidden="true" size={22} />,
      body: (
        <>
          A reward model can learn a scalar score{" "}
          <InlineMath text={String.raw`\(r_\phi(x,y)\)`} />, or DPO can turn the
          same pair into a direct log-ratio loss.
        </>
      ),
    },
    {
      title: "Policy pressure",
      icon: <Target aria-hidden="true" size={22} />,
      body: "PPO, Best-of-N, and DPO all use the preference signal differently, while a reference policy keeps useful base behavior anchored.",
    },
  ];

  return (
    <section className="border-y border-slate-200 bg-white text-slate-950">
      <div className="mx-auto w-full max-w-6xl px-4 py-12">
        <div className="grid gap-8 lg:grid-cols-[0.8fr_1.2fr]">
          <div className="min-w-0 space-y-4">
            <IconBadge
              icon={<Workflow aria-hidden="true" size={18} />}
              label="Concept map"
              tone="sky"
            />
            <h2 className="text-2xl font-semibold md:text-3xl">
              Read the alignment pipeline before the algorithms
            </h2>
            <p className="text-base leading-7 text-slate-700">
              Lecture 5 stacks several ideas: preference labels, learned reward
              proxies, reinforcement-learning updates, inference-time reranking,
              and direct preference losses. The page uses one pipeline so each
              later formula has a role.
            </p>
            <div className="rounded-md border border-slate-300 bg-slate-50 p-4">
              <p className="text-sm font-semibold text-slate-500">
                Names used below
              </p>
              <dl className="mt-3 grid gap-3 text-sm leading-6 text-slate-700">
                <div>
                  <dt className="font-semibold text-slate-950">
                    <InlineMath text={String.raw`\(x\)`} />
                  </dt>
                  <dd>the prompt or context.</dd>
                </div>
                <div>
                  <dt className="font-semibold text-slate-950">
                    <InlineMath text={String.raw`\(y_w\)`} /> and{" "}
                    <InlineMath text={String.raw`\(y_l\)`} />
                  </dt>
                  <dd>the human-preferred winner and lower-rated loser.</dd>
                </div>
                <div>
                  <dt className="font-semibold text-slate-950">reference</dt>
                  <dd>
                    a frozen base policy used as an anchor so optimization does
                    not drift freely.
                  </dd>
                </div>
              </dl>
            </div>
          </div>

          <div className="min-w-0 rounded-lg border border-slate-300 bg-slate-50 p-4">
            <div className="grid gap-3 sm:grid-cols-2">
              {steps.map((step, index) => (
                <article
                  key={step.title}
                  className="rounded-md border border-slate-300 bg-white p-4"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center gap-2 text-sky-700">
                      {step.icon}
                      <h3 className="font-semibold text-slate-950">
                        {step.title}
                      </h3>
                    </div>
                    <div className="flex shrink-0 items-center gap-1 text-xs font-semibold text-slate-500">
                      <span>{index + 1}/4</span>
                      {index < steps.length - 1 && (
                        <ArrowRight aria-hidden="true" size={14} />
                      )}
                    </div>
                  </div>
                  <p className="mt-3 text-sm leading-6 text-slate-700">
                    {step.body}
                  </p>
                </article>
              ))}
            </div>

            <div className="mt-4 grid gap-3 md:grid-cols-3">
              <article className="rounded-md border border-emerald-300 bg-emerald-50 p-3">
                <div className="flex items-center gap-2 text-emerald-800">
                  <Trophy aria-hidden="true" size={18} />
                  <h3 className="font-semibold">Reward model path</h3>
                </div>
                <p className="mt-2 text-sm leading-6 text-emerald-950">
                  Train <InlineMath text={String.raw`\(r_\phi\)`} /> from pairs,
                  then use it to score samples during RLHF or reranking.
                </p>
              </article>
              <article className="rounded-md border border-amber-300 bg-amber-50 p-3">
                <div className="flex items-center gap-2 text-amber-800">
                  <RefreshCcw aria-hidden="true" size={18} />
                  <h3 className="font-semibold">PPO path</h3>
                </div>
                <p className="mt-2 text-sm leading-6 text-amber-950">
                  Use RL updates, value baselines, clipping, and reference
                  pressure to avoid oversized policy moves.
                </p>
              </article>
              <article className="rounded-md border border-rose-300 bg-rose-50 p-3">
                <div className="flex items-center gap-2 text-rose-800">
                  <Lock aria-hidden="true" size={18} />
                  <h3 className="font-semibold">DPO path</h3>
                </div>
                <p className="mt-2 text-sm leading-6 text-rose-950">
                  Skip the separate reward model and compare the trainable
                  policy against the frozen reference directly.
                </p>
              </article>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function BradleyTerryLab() {
  const [scenarioId, setScenarioId] = useState<RewardScenarioId>("clear");
  const scenario = rewardScenarios[scenarioId];
  const stats = useMemo(
    () => getBradleyTerryStats(scenario.winnerReward, scenario.loserReward),
    [scenario],
  );

  return (
    <section
      data-testid="bradley-terry-lab"
      className="border-y border-slate-800 bg-slate-950 text-slate-50"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="min-w-0 space-y-4">
          <IconBadge
            icon={<Scale aria-hidden="true" size={18} />}
            label="Reward gap"
            tone="emerald"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Reward modeling turns pairs into score differences
          </h2>
          <p className="text-base leading-7 text-slate-300">
            A reward model is a learned proxy for human preference. It reads a
            prompt-response pair and outputs one scalar score{" "}
            <InlineMath
              text={String.raw`\(r_\phi(x,y)\)`}
              className="text-emerald-200"
            />
            . It is trained from chosen-versus-rejected comparisons, then reused
            to score single completions during RLHF or reranking.
          </p>
          <div className="grid gap-3 sm:grid-cols-3">
            <article className="rounded-md border border-slate-800 bg-slate-900 p-3">
              <div className="flex items-center gap-2 text-emerald-200">
                <MessageSquareText aria-hidden="true" size={18} />
                <h3 className="font-semibold">Input</h3>
              </div>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                <InlineMath text={String.raw`\(x\)`} /> is the prompt;{" "}
                <InlineMath text={String.raw`\(y\)`} /> is one candidate
                response.
              </p>
            </article>
            <article className="rounded-md border border-slate-800 bg-slate-900 p-3">
              <div className="flex items-center gap-2 text-emerald-200">
                <Trophy aria-hidden="true" size={18} />
                <h3 className="font-semibold">Pair</h3>
              </div>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                <InlineMath text={String.raw`\(y_w\)`} /> is the preferred
                answer and <InlineMath text={String.raw`\(y_l\)`} /> is the
                rejected answer.
              </p>
            </article>
            <article className="rounded-md border border-slate-800 bg-slate-900 p-3">
              <div className="flex items-center gap-2 text-emerald-200">
                <BadgeCheck aria-hidden="true" size={18} />
                <h3 className="font-semibold">Target</h3>
              </div>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                The winner should get a higher reward than the loser.
              </p>
            </article>
          </div>
          <div className="overflow-x-auto rounded-md border border-slate-800 bg-slate-900 p-4">
            <MathText
              text={String.raw`\[P(y_w \succ y_l \mid x)=\frac{e^{r_\phi(x,y_w)}}{e^{r_\phi(x,y_w)}+e^{r_\phi(x,y_l)}}=\sigma(r_w-r_l)\]`}
              className="max-w-full overflow-x-auto text-slate-50"
            />
            <p className="mt-3 text-sm leading-6 text-slate-400">
              The absolute rewards are not the point. The difference{" "}
              <InlineMath
                text={String.raw`\(r_w-r_l\)`}
                className="text-slate-200"
              />{" "}
              controls the preference probability.
            </p>
          </div>
        </div>

        <div className="min-w-0 rounded-lg border border-slate-800 bg-slate-900 p-4">
          <div className="flex flex-wrap gap-2">
            {(Object.keys(rewardScenarios) as RewardScenarioId[]).map((id) => (
              <LabButton
                key={id}
                isActive={scenarioId === id}
                onClick={() => setScenarioId(id)}
              >
                {rewardScenarios[id].label}
              </LabButton>
            ))}
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <MetricCell
              label="chosen reward"
              value={formatNumber(scenario.winnerReward)}
            />
            <MetricCell
              label="rejected reward"
              value={formatNumber(scenario.loserReward)}
            />
            <MetricCell label="gap" value={formatNumber(stats.gap)} />
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            <FormulaStep
              label="Reward gap"
              expression={`${formatNumber(scenario.winnerReward)} - ${formatOperand(
                scenario.loserReward,
              )} = ${formatNumber(stats.gap)}`}
              note="Only the chosen-minus-rejected gap enters the sigmoid."
            />
            <FormulaStep
              label="Sigmoid probability"
              expression={`sigma(${formatNumber(stats.gap)}) = ${formatNumber(
                stats.probability,
              )}`}
              note="A gap of zero gives 0.50; a larger positive gap moves toward 1."
            />
          </div>

          <div className="mt-5 rounded-md border border-slate-800 bg-slate-950 p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="font-semibold text-slate-100">
                Preference probability
              </p>
              <p className="font-mono text-2xl font-semibold text-emerald-300">
                {formatNumber(stats.probability)}
              </p>
            </div>
            <div
              aria-hidden="true"
              className="mt-3 h-4 overflow-hidden rounded-full bg-slate-800"
            >
              <div
                className="h-full rounded-full bg-emerald-300"
                style={{ width: `${stats.probability * 100}%` }}
              />
            </div>
            <p role="status" className="mt-4 text-sm leading-6 text-slate-300">
              {scenario.note} The pairwise loss is{" "}
              <span className="font-mono text-emerald-200">
                {formatNumber(stats.loss)}
              </span>
              , so low preference probability creates stronger pressure to
              increase the chosen-minus-rejected gap.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

function RlhfLoop() {
  const rlTerms = [
    {
      id: "state",
      title: (
        <>
          State <InlineMath text={String.raw`\(s_t\)`} />
        </>
      ),
      icon: <MessageSquareText aria-hidden="true" size={20} />,
      body: "The prompt plus the tokens already generated. In an LLM, the state is a text prefix, not a grid cell.",
    },
    {
      id: "action",
      title: (
        <>
          Action <InlineMath text={String.raw`\(a_t\)`} />
        </>
      ),
      icon: <Bot aria-hidden="true" size={20} />,
      body: (
        <>
          The next token sampled from the policy distribution{" "}
          <InlineMath text={String.raw`\(\pi(a_t\mid s_t)\)`} />.
        </>
      ),
    },
    {
      id: "reward",
      title: "Reward",
      icon: <Target aria-hidden="true" size={20} />,
      body: "Usually a completion-level score from the reward model, often combined with a reference/KL penalty.",
    },
    {
      id: "advantage",
      title: (
        <>
          Advantage <InlineMath text={String.raw`\(A_t\)`} />
        </>
      ),
      icon: <Activity aria-hidden="true" size={20} />,
      body: "How much better or worse the sampled action did than the value baseline expected.",
    },
  ];

  const loopSteps = [
    {
      title: "Sample from policy",
      body: "The trainable LLM generates completions under the current distribution.",
      icon: <Bot aria-hidden="true" size={20} />,
      tone: "border-sky-300/50 bg-sky-400/10 text-sky-100",
    },
    {
      title: "Score behavior",
      body: "The frozen reward model turns the full completion into a scalar proxy score.",
      icon: <Scale aria-hidden="true" size={20} />,
      tone: "border-emerald-300/50 bg-emerald-400/10 text-emerald-100",
    },
    {
      title: "Estimate baseline",
      body: "A value model predicts expected reward so the update uses advantage instead of raw reward alone.",
      icon: <Activity aria-hidden="true" size={20} />,
      tone: "border-amber-300/50 bg-amber-400/10 text-amber-100",
    },
    {
      title: "Constrain drift",
      body: "PPO clipping and reference pressure limit how far the policy can move from useful earlier behavior.",
      icon: <Lock aria-hidden="true" size={20} />,
      tone: "border-rose-300/50 bg-rose-400/10 text-rose-100",
    },
  ];

  return (
    <section className="bg-[#101820] text-slate-50">
      <div className="mx-auto w-full max-w-6xl px-4 py-12">
        <div className="max-w-3xl space-y-4">
          <IconBadge
            icon={<Workflow aria-hidden="true" size={18} />}
            label="RLHF loop"
            tone="sky"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Map language generation into RL before using RLHF
          </h2>
          <p className="text-base leading-7 text-slate-300">
            Reinforcement learning is about improving a policy from rewards. In
            RLHF, the LLM is the policy, token choices are actions, and the
            reward is usually delayed until a completion can be scored. That is
            why the lecture introduces value baselines, advantages, and
            reference-policy constraints before PPO.
          </p>
        </div>

        <div className="mt-8 grid gap-4 md:grid-cols-4">
          {rlTerms.map((item) => (
            <article
              key={item.id}
              className="rounded-lg border border-slate-700 bg-slate-950 p-4"
            >
              <div className="flex items-center gap-2 text-sky-200">
                {item.icon}
                <h3 className="font-semibold text-slate-50">{item.title}</h3>
              </div>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                {item.body}
              </p>
            </article>
          ))}
        </div>

        <div className="mt-5 rounded-lg border border-slate-700 bg-slate-950 p-4">
          <h3 className="text-lg font-semibold text-slate-50">
            PPO-style RLHF control loop
          </h3>
          <div className="mt-4 grid gap-3 md:grid-cols-4">
            {loopSteps.map((item) => (
              <article
                key={item.title}
                className={`rounded-md border p-4 ${item.tone}`}
              >
                <div className="flex items-center gap-2">
                  {item.icon}
                  <h4 className="font-semibold text-slate-50">{item.title}</h4>
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-300">
                  {item.body}
                </p>
              </article>
            ))}
          </div>
          <p className="mt-4 text-sm leading-6 text-slate-400">
            The important simplification: the reward model says what looks good,
            the value model estimates what was expected, and advantage measures
            the surprise that drives the policy update.
          </p>
        </div>
      </div>
    </section>
  );
}

function RewardHackingCheckSection() {
  return (
    <section className="border-y border-slate-800 bg-slate-950 text-slate-50">
      <div className="mx-auto grid w-full max-w-6xl gap-6 px-4 py-10 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 rounded-lg border border-rose-300/30 bg-rose-400/10 p-4">
          <div className="flex items-center gap-3 text-rose-100">
            <ShieldAlert aria-hidden="true" size={24} />
            <h2 className="text-2xl font-semibold">
              Reward hacking is proxy success, objective failure
            </h2>
          </div>
          <p className="mt-4 text-sm leading-6 text-rose-100">
            The reward model is useful because it is trainable and cheap to
            query, but it is still a proxy. If the policy finds behavior that
            scores well while violating the human intent, the reward went up
            while alignment went down.
          </p>
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <FormulaStep
              label="Proxy"
              expression="reward model score increases"
              note="The measured number improves."
            />
            <FormulaStep
              label="Real target"
              expression="human preference gets worse"
              note="The intended objective was not captured by the proxy."
            />
          </div>
        </div>

        <CheckForUnderstanding
          title="Reward hacking check"
          question="A reward model gives high scores to answers with many confident claims, and the policy starts producing overconfident unsupported answers. What happened?"
          correctIndex={1}
          testId="reward-hacking-check"
          options={[
            {
              label: "The policy learned new facts from the reward model.",
              explanation:
                "A reward model scores behavior; it does not add factual training data by itself.",
            },
            {
              label:
                "The policy exploited a proxy reward that did not match the real human objective.",
              explanation:
                "Reward hacking means proxy score improved while the intended behavior got worse.",
            },
            {
              label: "Best-of-N removed all inference-time cost.",
              explanation:
                "Best-of-N usually increases inference generation and scoring work.",
            },
          ]}
        />
      </div>
    </section>
  );
}

function PpoClipLab() {
  const [scenarioId, setScenarioId] = useState<PpoScenarioId>("positive");
  const scenario = ppoScenarios[scenarioId];
  const stats = useMemo(() => getPpoClipStats(scenario), [scenario]);

  return (
    <section data-testid="ppo-clip-lab" className="bg-slate-900 text-slate-50">
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="min-w-0 space-y-4">
          <IconBadge
            icon={<SlidersHorizontal aria-hidden="true" size={18} />}
            label="PPO clipping"
            tone="amber"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            PPO means Proximal Policy Optimization
          </h2>
          <p className="text-base leading-7 text-slate-300">
            Proximal means near. PPO lets a good sampled action increase in
            probability, or a bad sampled action decrease, but it limits how
            much one update can benefit from moving far away from the old policy
            snapshot.
          </p>
          <p className="text-base leading-7 text-slate-300">
            The advantage{" "}
            <InlineMath text={String.raw`\(A_t\)`} className="text-amber-200" />{" "}
            is the update signal: positive when the sampled token did better
            than the value baseline expected, negative when it did worse.
          </p>
          <div className="space-y-3 overflow-x-auto rounded-md border border-slate-800 bg-slate-950 p-4">
            <MathText
              text={String.raw`\[L^{CLIP}(\theta)=\mathbb{E}_t\left[\min\left(r_t(\theta)A_t,\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t\right)\right]\]`}
              className="max-w-full overflow-x-auto text-slate-50"
            />
            <MathText
              text={String.raw`\[r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}\]`}
              className="max-w-full overflow-x-auto text-slate-50"
            />
            <MathText
              text={String.raw`\[A_t \approx R_t - V(s_t)\]`}
              className="max-w-full overflow-x-auto text-slate-50"
            />
            <div className="grid gap-3 pt-2 text-sm leading-6 text-slate-300 sm:grid-cols-2">
              <p>
                <InlineMath
                  text={String.raw`\(\pi_\theta\)`}
                  className="text-amber-200"
                />{" "}
                is the current policy;{" "}
                <InlineMath
                  text={String.raw`\(\pi_{\theta_{\mathrm{old}}}\)`}
                  className="text-amber-200"
                />{" "}
                is the snapshot used for this batch.
              </p>
              <p>
                <InlineMath
                  text={String.raw`\(\epsilon\)`}
                  className="text-amber-200"
                />{" "}
                sets the trust band, here from 0.80 to 1.20.
              </p>
            </div>
          </div>
        </div>

        <div className="min-w-0 rounded-lg border border-slate-800 bg-slate-950 p-4">
          <div className="flex flex-wrap gap-2">
            {(Object.keys(ppoScenarios) as PpoScenarioId[]).map((id) => (
              <LabButton
                key={id}
                isActive={scenarioId === id}
                onClick={() => setScenarioId(id)}
              >
                {ppoScenarios[id].label}
              </LabButton>
            ))}
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCell
              label="current prob"
              value={formatNumber(scenario.currentProbability)}
            />
            <MetricCell
              label="old prob"
              value={formatNumber(scenario.oldProbability)}
            />
            <MetricCell
              label="advantage"
              value={formatNumber(scenario.advantage)}
            />
            <MetricCell
              label="epsilon"
              value={formatNumber(scenario.epsilon)}
            />
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <MetricCell label="ratio" value={formatNumber(stats.ratio)} />
            <MetricCell
              label="clipped ratio"
              value={formatNumber(stats.clippedRatio)}
              detail={`${formatNumber(stats.lower)} to ${formatNumber(stats.upper)}`}
            />
            <MetricCell
              label="objective term"
              value={formatNumber(stats.objectiveTerm)}
            />
          </div>

          <div className="mt-5 grid gap-3 lg:grid-cols-2">
            <FormulaStep
              label="Ratio"
              expression={`${formatNumber(
                scenario.currentProbability,
              )} / ${formatNumber(scenario.oldProbability)} = ${formatNumber(
                stats.ratio,
              )}`}
              note="Compare the current token probability to the old snapshot."
            />
            <FormulaStep
              label="Clip ratio"
              expression={`clip(${formatNumber(stats.ratio)}, ${formatNumber(
                stats.lower,
              )}, ${formatNumber(stats.upper)}) = ${formatNumber(
                stats.clippedRatio,
              )}`}
              note="The ratio is forced into the trust band before the clipped term."
            />
            <FormulaStep
              label="Unclipped term"
              expression={`${formatNumber(stats.ratio)} * ${formatNumber(
                scenario.advantage,
              )} = ${formatNumber(stats.unclippedTerm)}`}
            />
            <FormulaStep
              label="Clipped term"
              expression={`${formatNumber(stats.clippedRatio)} * ${formatNumber(
                scenario.advantage,
              )} = ${formatNumber(stats.clippedTerm)}`}
            />
            <FormulaStep
              label="Objective contribution"
              expression={`min(${formatNumber(
                stats.unclippedTerm,
              )}, ${formatNumber(stats.clippedTerm)}) = ${formatNumber(
                stats.objectiveTerm,
              )}`}
              note="PPO uses the more conservative contribution."
            />
          </div>

          <p
            role="status"
            className="mt-5 rounded-md border border-amber-300/40 bg-amber-400/10 p-3 text-sm leading-6 text-amber-100"
          >
            {scenario.note}{" "}
            {stats.isClipped
              ? "Clipping is active in this example."
              : "The ratio stays inside the allowed band."}
          </p>
        </div>
      </div>
    </section>
  );
}

function BestOfNLab() {
  const [samples, setSamples] = useState(4);
  const acceptableProbability = 0.3;
  const missProbability = 1 - acceptableProbability;
  const stats = useMemo(
    () => getBestOfNStats({ samples, acceptableProbability }),
    [samples],
  );

  return (
    <section data-testid="best-of-n-lab" className="bg-slate-50 text-slate-950">
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 space-y-4">
          <IconBadge
            icon={<Gauge aria-hidden="true" size={18} />}
            label="Inference reranking"
            tone="emerald"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Best-of-N skips policy training but pays at serving time
          </h2>
          <p className="text-base leading-7 text-slate-700">
            Best-of-N samples several completions from the SFT model, scores
            each with the reward model, and returns the highest-scoring answer.
            It can improve the chosen output without updating weights, but only
            among candidates that were actually generated.
          </p>
          <div className="rounded-md border border-slate-300 bg-white p-4">
            <MathText
              text={String.raw`\[P(\text{at least one acceptable})=1-(1-p)^N\]`}
              className="max-w-full overflow-x-auto text-slate-950"
            />
            <p className="mt-3 text-sm leading-6 text-slate-700">
              This is a toy independence calculation. It assumes each candidate
              has probability <span className="font-mono">p</span> of being
              acceptable and that the reranker can select a good candidate when
              one appears.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {[1, 4, 8].map((count) => (
              <button
                key={count}
                type="button"
                aria-pressed={samples === count}
                onClick={() => setSamples(count)}
                className={`rounded-md border px-4 py-3 text-sm font-semibold ${
                  samples === count
                    ? "border-emerald-500 bg-emerald-100"
                    : "border-slate-300 bg-white hover:border-slate-500"
                }`}
              >
                N={count}
              </button>
            ))}
          </div>
        </div>

        <div className="min-w-0 rounded-lg border border-slate-300 bg-white p-4">
          <div className="grid gap-3 sm:grid-cols-3">
            <MetricCell
              label="candidate quality p"
              value={formatPercent(acceptableProbability)}
              detail="toy independent assumption"
            />
            <MetricCell
              label="at least one acceptable"
              value={formatPercent(stats.atLeastOneAcceptable)}
            />
            <MetricCell
              label="generation work"
              value={`${stats.generationMultiplier}x`}
            />
          </div>

          <div className="mt-5 grid gap-3 lg:grid-cols-2">
            <FormulaStep
              label="All candidates fail"
              expression={`(1 - ${formatNumber(
                acceptableProbability,
              )})^${samples} = ${formatNumber(missProbability)}^${samples} = ${formatNumber(
                stats.allFailProbability,
              )}`}
              note="Use the complement because it is easier to count every sample missing."
            />
            <FormulaStep
              label="At least one acceptable"
              expression={`1 - ${formatNumber(missProbability)}^${samples} = ${formatNumber(
                stats.atLeastOneAcceptable,
              )}`}
              note="For N=4, this is 1 - 0.70^4 = 0.76."
            />
          </div>

          <div className="mt-5 grid gap-2">
            {Array.from({ length: samples }).map((_, index) => {
              const reward =
                [0.3, 1.1, -1.4, 0.6, 0.2, -0.2, 1.4, 0.8][index] ?? 0;
              const isBest = reward === 1.4 || (samples < 8 && reward === 1.1);

              return (
                <div
                  key={index}
                  className={`grid grid-cols-[auto_1fr_auto] items-center gap-3 rounded-md border px-3 py-2 ${
                    isBest
                      ? "border-emerald-500 bg-emerald-50"
                      : "border-slate-200 bg-slate-50"
                  }`}
                >
                  <span className="font-mono text-sm text-slate-500">
                    y{index + 1}
                  </span>
                  <span className="text-sm text-slate-700">
                    candidate completion
                  </span>
                  <span className="font-mono text-sm font-semibold">
                    r={formatNumber(reward, 1)}
                  </span>
                </div>
              );
            })}
          </div>

          <p
            role="status"
            className="mt-5 rounded-md border border-emerald-300 bg-emerald-50 p-3 text-sm leading-6 text-emerald-900"
          >
            With N={samples}, the chance of at least one acceptable candidate is{" "}
            {formatPercent(stats.atLeastOneAcceptable)} because{" "}
            <span className="font-mono">
              1 - {formatNumber(missProbability)}^{samples} ={" "}
              {formatNumber(stats.atLeastOneAcceptable)}
            </span>
            . Generation work is roughly {stats.generationMultiplier}x before
            reward-model scoring.
          </p>
        </div>
      </div>
    </section>
  );
}

function DpoGapRail({
  title,
  chosen,
  rejected,
  gap,
}: {
  title: string;
  chosen: number;
  rejected: number;
  gap: number;
}) {
  const chosenWidth = Math.max(20, Math.min(92, 25 + (chosen + 8) * 20));
  const rejectedWidth = Math.max(20, Math.min(92, 25 + (rejected + 8) * 20));

  return (
    <div className="rounded-md border border-slate-800 bg-slate-950 p-3">
      <div className="flex items-center justify-between gap-3">
        <h3 className="font-semibold text-slate-100">{title}</h3>
        <span className="font-mono text-sm text-rose-200">
          gap={formatNumber(gap)}
        </span>
      </div>
      <div className="mt-3 space-y-2">
        <div className="grid grid-cols-[5rem_1fr_auto] items-center gap-2">
          <span className="text-sm text-emerald-200">chosen</span>
          <div className="h-3 overflow-hidden rounded-full bg-slate-800">
            <div
              aria-hidden="true"
              className="h-full rounded-full bg-emerald-300"
              style={{ width: `${chosenWidth}%` }}
            />
          </div>
          <span className="font-mono text-sm text-slate-300">
            {formatNumber(chosen, 1)}
          </span>
        </div>
        <div className="grid grid-cols-[5rem_1fr_auto] items-center gap-2">
          <span className="text-sm text-rose-200">rejected</span>
          <div className="h-3 overflow-hidden rounded-full bg-slate-800">
            <div
              aria-hidden="true"
              className="h-full rounded-full bg-rose-300"
              style={{ width: `${rejectedWidth}%` }}
            />
          </div>
          <span className="font-mono text-sm text-slate-300">
            {formatNumber(rejected, 1)}
          </span>
        </div>
      </div>
    </div>
  );
}

function DpoLogitLab() {
  const [scenarioId, setScenarioId] = useState<DpoScenarioId>("lecture");
  const scenario = dpoScenarios[scenarioId];
  const stats = useMemo(() => getDpoLogitStats(scenario), [scenario]);

  return (
    <section data-testid="dpo-logit-lab" className="bg-[#20151a] text-slate-50">
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="min-w-0 space-y-4">
          <IconBadge
            icon={<Calculator aria-hidden="true" size={18} />}
            label="DPO contrast"
            tone="rose"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            DPO compares what changed relative to the reference
          </h2>
          <p className="text-base leading-7 text-slate-300">
            Direct Preference Optimization uses chosen/rejected pairs without
            training a separate reward model. The frozen reference still
            matters: DPO asks whether the trainable policy favors the chosen
            answer more than the original reference did.
          </p>
          <div className="space-y-3 overflow-x-auto rounded-md border border-rose-300/30 bg-slate-950 p-4">
            <MathText
              text={String.raw`\[\ell_{DPO}=-\log\sigma\left(\beta\left[(\log \pi_\theta(y_w)-\log \pi_\theta(y_l))-(\log \pi_{ref}(y_w)-\log \pi_{ref}(y_l))\right]\right)\]`}
              className="max-w-full overflow-x-auto text-slate-50"
            />
            <p className="text-sm leading-6 text-slate-400">
              First compare chosen versus rejected inside each policy. Then
              subtract the reference gap. The result is not a raw preference; it
              is a contrast against the prior behavior of the base model.
            </p>
          </div>
        </div>

        <div className="min-w-0 rounded-lg border border-rose-300/30 bg-slate-950 p-4">
          <div className="flex flex-wrap gap-2">
            {(Object.keys(dpoScenarios) as DpoScenarioId[]).map((id) => (
              <LabButton
                key={id}
                isActive={scenarioId === id}
                onClick={() => setScenarioId(id)}
              >
                {dpoScenarios[id].label}
              </LabButton>
            ))}
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCell
              label="policy chosen"
              value={formatNumber(scenario.policyChosenLogProb, 1)}
            />
            <MetricCell
              label="policy rejected"
              value={formatNumber(scenario.policyRejectedLogProb, 1)}
            />
            <MetricCell
              label="ref chosen"
              value={formatNumber(scenario.referenceChosenLogProb, 1)}
            />
            <MetricCell
              label="ref rejected"
              value={formatNumber(scenario.referenceRejectedLogProb, 1)}
            />
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-4">
            <MetricCell
              label="policy gap"
              value={formatNumber(stats.policyGap)}
            />
            <MetricCell
              label="reference gap"
              value={formatNumber(stats.referenceGap)}
            />
            <MetricCell label="contrast" value={formatNumber(stats.contrast)} />
            <MetricCell label="DPO logit" value={formatNumber(stats.logit)} />
          </div>

          <div className="mt-5 grid gap-3 lg:grid-cols-2">
            <DpoGapRail
              title="Trainable policy"
              chosen={scenario.policyChosenLogProb}
              rejected={scenario.policyRejectedLogProb}
              gap={stats.policyGap}
            />
            <DpoGapRail
              title="Frozen reference"
              chosen={scenario.referenceChosenLogProb}
              rejected={scenario.referenceRejectedLogProb}
              gap={stats.referenceGap}
            />
          </div>

          <div className="mt-5 grid gap-3 lg:grid-cols-2">
            <FormulaStep
              label="Policy gap"
              expression={`${formatNumber(
                scenario.policyChosenLogProb,
                1,
              )} - ${formatOperand(
                scenario.policyRejectedLogProb,
                1,
              )} = ${formatNumber(stats.policyGap)}`}
              note={
                <>
                  How much the trainable policy prefers{" "}
                  <InlineMath text={String.raw`\(y_w\)`} /> over{" "}
                  <InlineMath text={String.raw`\(y_l\)`} />.
                </>
              }
            />
            <FormulaStep
              label="Reference gap"
              expression={`${formatNumber(
                scenario.referenceChosenLogProb,
                1,
              )} - ${formatOperand(
                scenario.referenceRejectedLogProb,
                1,
              )} = ${formatNumber(stats.referenceGap)}`}
              note={
                <>
                  How much the frozen base policy already preferred{" "}
                  <InlineMath text={String.raw`\(y_w\)`} />.
                </>
              }
            />
            <FormulaStep
              label="Contrast"
              expression={`${formatNumber(stats.policyGap)} - ${formatNumber(
                stats.referenceGap,
              )} = ${formatNumber(stats.contrast)}`}
              note="Positive means the trainable policy moved toward the winner relative to the reference."
            />
            <FormulaStep
              label="Preference logit"
              expression={`${formatNumber(scenario.beta)} * ${formatNumber(
                stats.contrast,
              )} = ${formatNumber(stats.logit)}`}
              note={
                <>
                  <InlineMath
                    text={String.raw`\(\sigma(${formatNumber(stats.logit)})\)`}
                  />{" "}
                  = {formatNumber(stats.probability)}
                </>
              }
            />
          </div>

          <p
            role="status"
            className="mt-5 rounded-md border border-rose-300/40 bg-rose-400/10 p-3 text-sm leading-6 text-rose-100"
          >
            {scenario.note} The implied preference probability is{" "}
            {formatNumber(stats.probability)} and the pair loss is{" "}
            {formatNumber(stats.loss)}.
          </p>
        </div>
      </div>
    </section>
  );
}

function DpoCheckSection() {
  return (
    <section className="border-y border-rose-300/20 bg-[#170f13] text-slate-50">
      <div className="mx-auto grid w-full max-w-6xl gap-6 px-4 py-10 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 rounded-lg border border-rose-300/30 bg-rose-400/10 p-4">
          <div className="flex items-center gap-3 text-rose-100">
            <Calculator aria-hidden="true" size={24} />
            <h2 className="text-2xl font-semibold">
              DPO keeps the preference pair but removes the reward-model stage
            </h2>
          </div>
          <p className="mt-4 text-sm leading-6 text-rose-100">
            The shortcut is not magic and it is not preference-free. DPO still
            needs chosen/rejected examples, a trainable policy, and a frozen
            reference; it just writes the preference pressure directly as a
            supervised-style log-probability loss.
          </p>
        </div>

        <CheckForUnderstanding
          title="DPO check"
          question="Why is DPO still an alignment method even though it removes the explicit reward-model stage?"
          correctIndex={2}
          testId="dpo-check"
          options={[
            {
              label: "It no longer needs preference data.",
              explanation: "DPO is built on chosen/rejected preference pairs.",
            },
            {
              label: "It updates the reference model instead of the policy.",
              explanation:
                "The reference is frozen while the trainable policy changes.",
            },
            {
              label:
                "It uses preference pairs and a frozen reference to train the policy directly.",
              explanation:
                "DPO expresses a Bradley-Terry-like signal using policy/reference log probabilities.",
            },
          ]}
        />
      </div>
    </section>
  );
}

function MethodDecisionBoard() {
  const [scenarioId, setScenarioId] =
    useState<DecisionScenarioId>("quickIteration");
  const scenario = decisionScenarios[scenarioId];

  return (
    <section
      data-testid="method-decision-board"
      className="bg-slate-950 text-slate-50"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 space-y-4">
          <IconBadge
            icon={<Route aria-hidden="true" size={18} />}
            label="Method choice"
            tone="sky"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Choose the alignment path by where you want to pay cost
          </h2>
          <p className="text-base leading-7 text-slate-300">
            Lecture 5 ends by comparing PPO-based RLHF, Best-of-N, and DPO. The
            right method depends less on a single leaderboard claim and more on
            model components, inference cost, data support, and operational
            tolerance for RL instability.
          </p>
          <div className="grid gap-2">
            {(Object.keys(decisionScenarios) as DecisionScenarioId[]).map(
              (id) => (
                <LabButton
                  key={id}
                  isActive={scenarioId === id}
                  onClick={() => setScenarioId(id)}
                >
                  {decisionScenarios[id].label}
                </LabButton>
              ),
            )}
          </div>
        </div>

        <div className="min-w-0 rounded-lg border border-slate-800 bg-slate-900 p-4">
          <div className="rounded-md border border-sky-300/40 bg-sky-400/10 p-4">
            <p className="text-sm font-semibold text-sky-100">
              Recommended path
            </p>
            <p className="mt-2 text-3xl font-semibold text-slate-50">
              {scenario.method}
            </p>
            <p role="status" className="mt-4 text-sm leading-6 text-slate-300">
              {scenario.diagnosis}
            </p>
          </div>

          <div className="mt-4 grid gap-3 md:grid-cols-3">
            <article className="rounded-md border border-slate-800 bg-slate-950 p-3">
              <h3 className="font-semibold text-slate-100">PPO-style RLHF</h3>
              <p className="mt-2 text-sm leading-6 text-slate-400">
                Most component-heavy: policy, reward model, value model, and
                reference pressure.
              </p>
            </article>
            <article className="rounded-md border border-slate-800 bg-slate-950 p-3">
              <h3 className="font-semibold text-slate-100">Best-of-N</h3>
              <p className="mt-2 text-sm leading-6 text-slate-400">
                No policy update, but more generation and reranking on each
                request.
              </p>
            </article>
            <article className="rounded-md border border-slate-800 bg-slate-950 p-3">
              <h3 className="font-semibold text-slate-100">DPO</h3>
              <p className="mt-2 text-sm leading-6 text-slate-400">
                Supervised-style preference loss with a frozen reference and no
                separate reward model.
              </p>
            </article>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function StanfordCME295Lecture5LearningPage({
  experience,
}: Props) {
  return (
    <main className="min-h-screen bg-slate-950 text-slate-50">
      <section className="border-b border-slate-800 bg-[#0b1218]">
        <div className="mx-auto grid min-h-[540px] w-full max-w-6xl items-center gap-8 px-4 py-10 md:py-14 lg:grid-cols-[0.95fr_1.05fr]">
          <div className="min-w-0 space-y-6">
            <div className="flex flex-wrap gap-2">
              <IconBadge
                icon={<BrainCircuit aria-hidden="true" size={18} />}
                label="Stanford CME295 Lecture 5"
                tone="rose"
              />
              <IconBadge
                icon={<ShieldAlert aria-hidden="true" size={18} />}
                label="Preference tuning"
                tone="amber"
              />
            </div>
            <div className="space-y-4">
              <h1 className="max-w-3xl text-3xl font-semibold text-slate-50 md:text-5xl md:leading-tight">
                Tune preferences without chasing the proxy
              </h1>
              <p className="max-w-2xl text-base leading-7 text-slate-300 md:text-lg">
                Build the Lecture 5 mental model from one concrete pair: a model
                response humans prefer, a response they reject, and the training
                machinery that can either use that signal well or over-optimize
                it.
              </p>
              <p className="text-sm font-semibold text-emerald-300">
                {experience.durationMinutes} min / {experience.level}
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <QuizTransitionButton
                sourceId={experience.sourceId}
                label="Start Lecture 5 questions"
              />
              <a
                href="#workbench"
                className="inline-flex items-center justify-center rounded-lg border border-slate-700 px-4 py-3 text-sm font-bold text-slate-100 transition-colors hover:border-slate-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950"
              >
                Open workbench
              </a>
            </div>
          </div>

          <AlignmentTraceVisual />
        </div>
      </section>

      <PreferenceDataStudio />
      <AlignmentPipeline />
      <BradleyTerryLab />
      <RlhfLoop />
      <RewardHackingCheckSection />
      <PpoClipLab />
      <BestOfNLab />
      <DpoLogitLab />
      <DpoCheckSection />
      <MethodDecisionBoard />

      <section className="border-t border-slate-800 bg-[#0b1218] text-slate-50">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-5 px-4 py-10 md:flex-row md:items-center md:justify-between">
          <div className="max-w-2xl">
            <h2 className="text-2xl font-semibold">Ready for the MCQs</h2>
            <p className="mt-2 text-sm leading-6 text-slate-300">
              Keep three distinctions active: preference data is comparative,
              reward models are imperfect proxies, and PPO, Best-of-N, and DPO
              pay their cost in different places.
            </p>
          </div>
          <QuizTransitionButton sourceId={experience.sourceId} />
        </div>
      </section>
    </main>
  );
}
