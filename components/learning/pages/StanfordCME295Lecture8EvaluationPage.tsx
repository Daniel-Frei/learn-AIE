"use client";

import Link from "next/link";
import { useMemo, useState, type ReactNode } from "react";
import {
  ArrowRight,
  BrainCircuit,
  Calculator,
  ClipboardCheck,
  FileText,
  Gauge,
  LineChart,
  Route,
  Scale,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Split,
  TriangleAlert,
  Wrench,
  XCircle,
} from "lucide-react";
import MathText from "../../MathText";
import { QuizTransitionButton } from "../LearningPrimitives";
import {
  getChanceAgreement,
  getCohensKappa,
  getPassAtK,
  getPassHatK,
  getParetoFrontier,
  getUnigramOverlapStats,
  getWeightedFactualityScore,
  type FactualityClaim,
  type ParetoModel,
} from "./cme295-lecture8/evaluationMath";

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function formatDecimal(value: number, digits = 2): string {
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
    blue: "border-blue-700 bg-blue-700 text-white",
    green: "border-emerald-700 bg-emerald-700 text-white",
    amber: "border-amber-700 bg-amber-500 text-slate-950",
    rose: "border-rose-700 bg-rose-700 text-white",
    violet: "border-violet-700 bg-violet-700 text-white",
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

const evaluationScenarios = {
  answer: {
    label: "Gift advice",
    request: "What birthday gift should I get for a child who likes animals?",
    output:
      "A teddy bear is usually a sweet gift, but ask about favorite animals and allergies before buying one.",
    qualitySignals: [
      "Does the answer address the request?",
      "Is it useful rather than vague?",
      "Does it avoid unsupported certainty?",
    ],
    systemSignals: ["Latency", "Cost per answer", "Uptime"],
    diagnosis:
      "This is mostly output-quality evaluation. Human ratings, a clear rubric, and LLM-as-a-Judge can help, while latency and cost describe serving constraints.",
  },
  rag: {
    label: "Policy answer",
    request:
      "Can a contractor expense a same-day flight change under the travel policy?",
    output:
      "The answer should be grounded in retrieved policy clauses, not only in fluent travel advice.",
    qualitySignals: [
      "Are answer claims supported by retrieved evidence?",
      "Did the response cite the relevant policy boundary?",
      "Is the final answer concise enough for a user decision?",
    ],
    systemSignals: [
      "Retrieval recall@k",
      "Reranker cost",
      "End-to-end latency",
    ],
    diagnosis:
      "This needs both output quality and system-stage metrics. A final answer can look good while retrieval missed the decisive evidence.",
  },
  agent: {
    label: "Refund agent",
    request:
      "Change my retail order address and refund the expedited-shipping fee.",
    output:
      "The assistant must choose tools, execute state changes, and report the confirmed result.",
    qualitySignals: [
      "Does the final message match the real database state?",
      "Did the agent follow policy?",
      "Does the user receive the right next step?",
    ],
    systemSignals: [
      "Tool prediction success",
      "Tool call success",
      "Pass^k repeated reliability",
    ],
    diagnosis:
      "Agent evaluation must inspect the workflow. Scoring only the last message can hide a missing state change or a wrong tool argument.",
  },
} as const;

type EvaluationScenarioId = keyof typeof evaluationScenarios;

function EvaluationScopeRouter() {
  const [scenarioId, setScenarioId] = useState<EvaluationScenarioId>("answer");
  const scenario = evaluationScenarios[scenarioId];

  return (
    <section
      data-testid="evaluation-scope-router"
      className="border-y border-slate-200 bg-[#f8fafc] text-slate-950"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 space-y-4">
          <SectionBadge
            icon={<Route aria-hidden="true" size={18} />}
            label="Evaluation scope"
            tone="blue"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            First separate answer quality from system behavior
          </h2>
          <p className="text-base leading-7 text-slate-700">
            The phrase evaluate the model is underspecified. Output quality asks
            whether the answer is useful, factual, coherent, safe, and aligned
            with the instruction. System metrics ask whether the deployed path
            is fast, reliable, affordable, and correctly connected to tools or
            retrieval.
          </p>
          <div className="grid gap-2 sm:grid-cols-3">
            {(Object.keys(evaluationScenarios) as EvaluationScenarioId[]).map(
              (id) => (
                <ControlButton
                  key={id}
                  isActive={scenarioId === id}
                  onClick={() => setScenarioId(id)}
                >
                  {evaluationScenarios[id].label}
                </ControlButton>
              ),
            )}
          </div>
        </div>

        <div className="min-w-0 rounded-xl border border-slate-300 bg-white p-4">
          <div className="rounded-lg border border-slate-300 bg-slate-50 p-4">
            <p className="text-sm font-semibold text-slate-500">Request</p>
            <p className="mt-2 text-lg font-semibold text-slate-950">
              {scenario.request}
            </p>
            <p className="mt-3 text-sm leading-6 text-slate-700">
              {scenario.output}
            </p>
          </div>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <div>
              <p className="text-sm font-semibold text-emerald-700">
                Output-quality signals
              </p>
              <ul className="mt-2 grid gap-2">
                {scenario.qualitySignals.map((signal) => (
                  <li
                    key={signal}
                    className="rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm leading-5 text-emerald-950"
                  >
                    {signal}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <p className="text-sm font-semibold text-blue-700">
                System or workflow signals
              </p>
              <ul className="mt-2 grid gap-2">
                {scenario.systemSignals.map((signal) => (
                  <li
                    key={signal}
                    className="rounded-md border border-blue-200 bg-blue-50 px-3 py-2 text-sm leading-5 text-blue-950"
                  >
                    {signal}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <p
            role="status"
            className="mt-4 rounded-lg border border-blue-500 bg-blue-50 p-4 text-sm leading-6 text-blue-950"
          >
            {scenario.diagnosis}
          </p>
        </div>
      </div>
    </section>
  );
}

const agreementProfiles = {
  balanced: {
    label: "Balanced rubric",
    observedAgreement: 0.72,
    raterAPositiveRate: 0.5,
    raterBPositiveRate: 0.5,
    note: "Both raters use the positive and negative labels evenly, so a 72% raw agreement is clearly above a 50% chance baseline.",
  },
  lenient: {
    label: "Lenient raters",
    observedAgreement: 0.78,
    raterAPositiveRate: 0.8,
    raterBPositiveRate: 0.7,
    note: "Raw agreement looks high, but much of it can happen because both raters often say the answer is good.",
  },
  imbalanced: {
    label: "Imbalanced labels",
    observedAgreement: 0.95,
    raterAPositiveRate: 0.95,
    raterBPositiveRate: 0.92,
    note: "When almost every item gets the same label, raw agreement can look excellent while agreement above chance is more modest.",
  },
} as const;

type AgreementProfileId = keyof typeof agreementProfiles;

function HumanAgreementLab() {
  const [profileId, setProfileId] = useState<AgreementProfileId>("balanced");
  const profile = agreementProfiles[profileId];
  const chanceAgreement = getChanceAgreement(profile);
  const kappa = getCohensKappa(profile);
  const aNegative = 1 - profile.raterAPositiveRate;
  const bNegative = 1 - profile.raterBPositiveRate;

  return (
    <section
      id="agreement-lab"
      data-testid="human-agreement-lab"
      className="scroll-mt-20 bg-white text-slate-950"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 space-y-4">
          <SectionBadge
            icon={<Scale aria-hidden="true" size={18} />}
            label="Human ratings"
            tone="green"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Raw agreement needs a chance baseline
          </h2>
          <p className="text-base leading-7 text-slate-700">
            Human ratings are the closest practical signal for open-ended answer
            quality, but raters can disagree or share label habits. Agreement
            metrics ask how much observed agreement exceeds the agreement
            expected from those label frequencies.
          </p>
          <FormulaPanel
            title="Chance agreement and Cohen kappa"
            formula={String.raw`\[\begin{aligned}
p_e &= p_Ap_B + (1-p_A)(1-p_B)\\
\kappa &= \frac{p_o - p_e}{1-p_e}
\end{aligned}\]`}
          >
            <span>
              <InlineMath text={String.raw`\(p_o\)`} /> is observed agreement.{" "}
              <InlineMath text={String.raw`\(p_e\)`} /> is the agreement a pair
              of independent raters would get from their marginal label rates.
            </span>
          </FormulaPanel>
        </div>

        <div className="min-w-0 rounded-xl border border-slate-300 bg-slate-50 p-4">
          <div className="grid gap-2 sm:grid-cols-3">
            {(Object.keys(agreementProfiles) as AgreementProfileId[]).map(
              (id) => (
                <ControlButton
                  key={id}
                  tone="green"
                  isActive={profileId === id}
                  onClick={() => setProfileId(id)}
                >
                  {agreementProfiles[id].label}
                </ControlButton>
              ),
            )}
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <Metric
              label="Observed agreement"
              value={formatPercent(profile.observedAgreement)}
            />
            <Metric
              label="Chance agreement"
              value={formatPercent(chanceAgreement)}
            />
            <Metric label="Kappa" value={formatDecimal(kappa)} />
          </div>

          <p
            role="status"
            className="mt-4 rounded-lg border border-emerald-500 bg-emerald-50 p-4 text-sm leading-6 text-emerald-950"
          >
            Calculation:{" "}
            <span className="font-mono">
              {formatDecimal(profile.raterAPositiveRate)} *{" "}
              {formatDecimal(profile.raterBPositiveRate)} +{" "}
              {formatDecimal(aNegative)} * {formatDecimal(bNegative)} ={" "}
              {formatDecimal(chanceAgreement)}
            </span>
            , then kappa is{" "}
            <span className="font-mono">
              ({formatDecimal(profile.observedAgreement)} -{" "}
              {formatDecimal(chanceAgreement)}) / (1 -{" "}
              {formatDecimal(chanceAgreement)}) = {formatDecimal(kappa)}
            </span>
            . {profile.note}
          </p>
        </div>
      </div>
    </section>
  );
}

const referenceAnswer =
  "A plush teddy bear can comfort a child during bedtime.";

const overlapCandidates = {
  close: {
    label: "Close wording",
    text: "A plush teddy bear can comfort a child at bedtime.",
    note: "Overlap metrics work best when correct answers share wording with the reference.",
  },
  paraphrase: {
    label: "Meaningful paraphrase",
    text: "Soft stuffed bears often help kids feel safe as they fall asleep.",
    note: "The meaning is close, but the unigram overlap is much lower because the surface wording changed.",
  },
  short: {
    label: "Too short",
    text: "Teddy bear.",
    note: "A precision-focused metric can look deceptively good on a very short answer unless it includes a brevity or recall penalty.",
  },
} as const;

type OverlapCandidateId = keyof typeof overlapCandidates;

function ReferenceOverlapLab() {
  const [candidateId, setCandidateId] =
    useState<OverlapCandidateId>("paraphrase");
  const candidate = overlapCandidates[candidateId];
  const stats = useMemo(
    () =>
      getUnigramOverlapStats({
        candidate: candidate.text,
        reference: referenceAnswer,
      }),
    [candidate],
  );

  return (
    <section
      data-testid="reference-overlap-lab"
      className="border-y border-slate-200 bg-[#f6f0e8] text-slate-950"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="min-w-0 space-y-4">
          <SectionBadge
            icon={<FileText aria-hidden="true" size={18} />}
            label="Reference metrics"
            tone="amber"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Reference overlap is useful and brittle
          </h2>
          <p className="text-base leading-7 text-slate-700">
            METEOR, BLEU, and ROUGE compare a model output to fixed references.
            That makes repeated evaluation cheap once references exist, but it
            also means strong paraphrases can score poorly and short answers can
            exploit precision unless the metric corrects for missing content.
          </p>
          <FormulaPanel
            title="Unigram precision and recall"
            formula={String.raw`\[\mathrm{precision}=\frac{\mathrm{matched}}{\mathrm{generated}},\quad \mathrm{recall}=\frac{\mathrm{matched}}{\mathrm{reference}}\]`}
          >
            <span>
              METEOR adds a weighted precision and recall score plus an ordering
              penalty. BLEU is more precision-focused and uses a brevity
              penalty. ROUGE variants are often used for summarization.
            </span>
          </FormulaPanel>
        </div>

        <div className="min-w-0 rounded-xl border border-amber-700/30 bg-white p-4">
          <div className="rounded-lg border border-slate-300 bg-slate-50 p-4">
            <p className="text-sm font-semibold text-slate-500">Reference</p>
            <p className="mt-2 text-lg font-semibold text-slate-950">
              {referenceAnswer}
            </p>
          </div>
          <div className="mt-4 grid gap-2 sm:grid-cols-3">
            {(Object.keys(overlapCandidates) as OverlapCandidateId[]).map(
              (id) => (
                <ControlButton
                  key={id}
                  tone="amber"
                  isActive={candidateId === id}
                  onClick={() => setCandidateId(id)}
                >
                  {overlapCandidates[id].label}
                </ControlButton>
              ),
            )}
          </div>

          <div className="mt-5 rounded-lg border border-slate-300 bg-slate-50 p-4">
            <p className="text-sm font-semibold text-slate-500">Candidate</p>
            <p className="mt-2 text-base font-semibold text-slate-950">
              {candidate.text}
            </p>
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <Metric
              label="Matched unigrams"
              value={`${stats.matchedUnigrams} / ${stats.referenceUnigrams}`}
            />
            <Metric label="Precision" value={formatPercent(stats.precision)} />
            <Metric label="Recall" value={formatPercent(stats.recall)} />
          </div>

          <p
            role="status"
            className="mt-4 rounded-lg border border-amber-500 bg-amber-50 p-4 text-sm leading-6 text-amber-950"
          >
            Calculation: {stats.matchedUnigrams} matched words over{" "}
            {stats.candidateUnigrams} generated words gives{" "}
            {formatPercent(stats.precision)} precision; the same{" "}
            {stats.matchedUnigrams} matched words over {stats.referenceUnigrams}{" "}
            reference words gives {formatPercent(stats.recall)} recall.{" "}
            {candidate.note}
          </p>
        </div>
      </div>
    </section>
  );
}

const responseA =
  "Check the policy. Same-day changes are reimbursable only when manager approval is documented.";
const responseB =
  "Absolutely. Same-day travel changes are always reimbursable because flexible travel support is important for contractors and employees. Submit the receipt whenever you can.";

type JudgeOrder = "a-first" | "b-first";
type JudgeControl = "single" | "swapped" | "rubric";

function JudgeBiasLab() {
  const [order, setOrder] = useState<JudgeOrder>("a-first");
  const [control, setControl] = useState<JudgeControl>("single");
  const firstLabel = order === "a-first" ? "Response A" : "Response B";
  const secondLabel = order === "a-first" ? "Response B" : "Response A";
  const firstText = order === "a-first" ? responseA : responseB;
  const secondText = order === "a-first" ? responseB : responseA;
  const status =
    control === "single"
      ? `Single-order judging is fragile here: a position-biased judge can favor ${firstLabel} because it appears first.`
      : control === "swapped"
        ? "Swap-and-aggregate exposes position bias: if the first answer wins in both orders, the score is not a stable content preference."
        : "A rubric that prioritizes grounded policy evidence and penalizes unsupported certainty should favor Response A despite Response B being longer.";

  return (
    <section data-testid="judge-bias-lab" className="bg-white text-slate-950">
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.88fr_1.12fr]">
        <div className="min-w-0 space-y-4">
          <SectionBadge
            icon={<BrainCircuit aria-hidden="true" size={18} />}
            label="LLM-as-a-Judge"
            tone="violet"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            A judge prompt is an evaluation protocol, not just a model call
          </h2>
          <p className="text-base leading-7 text-slate-700">
            LLM-as-a-Judge can return a score and rationale for open-ended
            answers without a fixed reference. The useful version includes a
            criterion, structured fields, low-temperature settings when
            repeatability matters, and controls for position, verbosity, and
            self-enhancement bias.
          </p>
          <FormulaPanel
            title="Judge call shape"
            formula={String.raw`\[\mathrm{judge}(\mathrm{prompt},\mathrm{response},\mathrm{criterion})\rightarrow\{\mathrm{rationale},\mathrm{score}\}\]`}
          >
            <span>
              Asking for the rationale before the score can improve the
              evaluation process, while a structured schema keeps downstream
              code from scraping prose.
            </span>
          </FormulaPanel>
        </div>

        <div className="min-w-0 rounded-xl border border-slate-300 bg-slate-50 p-4">
          <div className="grid gap-2 sm:grid-cols-2">
            <ControlButton
              tone="violet"
              isActive={order === "a-first"}
              onClick={() => setOrder("a-first")}
            >
              A then B
            </ControlButton>
            <ControlButton
              tone="violet"
              isActive={order === "b-first"}
              onClick={() => setOrder("b-first")}
            >
              B then A
            </ControlButton>
          </div>
          <div className="mt-4 grid gap-2 sm:grid-cols-3">
            <ControlButton
              tone="blue"
              isActive={control === "single"}
              onClick={() => setControl("single")}
            >
              Single order
            </ControlButton>
            <ControlButton
              tone="blue"
              isActive={control === "swapped"}
              onClick={() => setControl("swapped")}
            >
              Swap and aggregate
            </ControlButton>
            <ControlButton
              tone="green"
              isActive={control === "rubric"}
              onClick={() => setControl("rubric")}
            >
              Rubric plus length check
            </ControlButton>
          </div>

          <div className="mt-5 grid gap-4 md:grid-cols-2">
            <div className="rounded-lg border border-slate-300 bg-white p-4">
              <p className="text-sm font-semibold text-slate-500">
                {firstLabel}
              </p>
              <p className="mt-2 text-sm leading-6 text-slate-700">
                {firstText}
              </p>
            </div>
            <div className="rounded-lg border border-slate-300 bg-white p-4">
              <p className="text-sm font-semibold text-slate-500">
                {secondLabel}
              </p>
              <p className="mt-2 text-sm leading-6 text-slate-700">
                {secondText}
              </p>
            </div>
          </div>

          <p
            role="status"
            className="mt-4 rounded-lg border border-violet-500 bg-violet-50 p-4 text-sm leading-6 text-violet-950"
          >
            {status}
          </p>
        </div>
      </div>
    </section>
  );
}

const factualityClaims: {
  label: string;
  text: string;
  weight: number;
  status: FactualityClaim["status"];
}[] = [
  {
    label: "A",
    text: "Same-day flight changes require manager approval.",
    weight: 0.35,
    status: "supported",
  },
  {
    label: "B",
    text: "Contractors can always self-approve changes.",
    weight: 0.25,
    status: "contradicted",
  },
  {
    label: "C",
    text: "The traveler already attached approval in the expense system.",
    weight: 0.2,
    status: "unverifiable",
  },
  {
    label: "D",
    text: "Receipts are still required for reimbursement.",
    weight: 0.2,
    status: "supported",
  },
];

type UnverifiablePolicy = "zero" | "half";

function FactualityLab() {
  const [policy, setPolicy] = useState<UnverifiablePolicy>("zero");
  const unverifiableCredit = policy === "half" ? 0.5 : 0;
  const score = getWeightedFactualityScore(
    factualityClaims,
    unverifiableCredit,
  );
  const unverifiableTerm = factualityClaims.find(
    (claim) => claim.status === "unverifiable",
  );

  return (
    <section
      data-testid="factuality-claim-lab"
      className="border-y border-slate-200 bg-[#edf5f2] text-slate-950"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 space-y-4">
          <SectionBadge
            icon={<Search aria-hidden="true" size={18} />}
            label="Factuality"
            tone="green"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Factuality is claim-level, not fluency-level
          </h2>
          <p className="text-base leading-7 text-slate-700">
            A polished answer can mix supported, contradicted, and unverifiable
            claims. Claim decomposition makes the evaluator inspect each atomic
            factual statement against evidence or ground truth before
            aggregating a score.
          </p>
          <FormulaPanel
            title="Weighted factuality score"
            formula={String.raw`\[\mathrm{score}=\sum_i w_i\,\mathrm{credit}(\mathrm{claim}_i)\]`}
          >
            <span>
              Supported claims receive full credit. Contradicted claims receive
              no credit. The policy for unverifiable claims should be explicit
              because it changes the score.
            </span>
          </FormulaPanel>
        </div>

        <div className="min-w-0 rounded-xl border border-slate-300 bg-white p-4">
          <div className="grid gap-2 sm:grid-cols-2">
            <ControlButton
              tone="green"
              isActive={policy === "zero"}
              onClick={() => setPolicy("zero")}
            >
              Unverifiable gets no credit
            </ControlButton>
            <ControlButton
              tone="green"
              isActive={policy === "half"}
              onClick={() => setPolicy("half")}
            >
              Unverifiable gets half credit
            </ControlButton>
          </div>

          <div className="mt-5 grid gap-3">
            {factualityClaims.map((claim) => (
              <div
                key={claim.label}
                className="grid min-w-0 gap-3 rounded-lg border border-slate-300 bg-slate-50 p-3 md:grid-cols-[auto_1fr_auto_auto]"
              >
                <span className="flex h-8 w-8 items-center justify-center rounded-full bg-slate-900 text-sm font-bold text-white">
                  {claim.label}
                </span>
                <p className="text-sm leading-6 text-slate-700">{claim.text}</p>
                <span className="text-sm font-semibold text-slate-600">
                  weight {formatDecimal(claim.weight)}
                </span>
                <span
                  className={`rounded-md px-2 py-1 text-sm font-semibold ${
                    claim.status === "supported"
                      ? "bg-emerald-100 text-emerald-900"
                      : claim.status === "contradicted"
                        ? "bg-rose-100 text-rose-900"
                        : "bg-amber-100 text-amber-900"
                  }`}
                >
                  {claim.status}
                </span>
              </div>
            ))}
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <Metric label="Supported weight" value="0.55" />
            <Metric
              label="Unverifiable credit"
              value={`${formatDecimal(unverifiableCredit)}x`}
            />
            <Metric label="Factuality score" value={formatPercent(score)} />
          </div>

          <p
            role="status"
            className="mt-4 rounded-lg border border-emerald-500 bg-emerald-50 p-4 text-sm leading-6 text-emerald-950"
          >
            Calculation: supported claims contribute 0.35 + 0.20 = 0.55
            {unverifiableTerm
              ? `, and claim ${unverifiableTerm.label} contributes ${formatDecimal(unverifiableCredit)} * ${formatDecimal(unverifiableTerm.weight)}`
              : ""}
            . The resulting score is {formatPercent(score)}. Fluency alone would
            miss the contradicted and unverifiable claims.
          </p>
        </div>
      </div>
    </section>
  );
}

const agentFailures = {
  noTool: {
    label: "No tool",
    stage: "Tool prediction",
    symptom:
      "The assistant answers from memory even though a live account-balance tool is available and required.",
    remedy:
      "Improve tool routing, tool-use prompting, or supervised examples for when the model must call an API.",
  },
  hallucinatedTool: {
    label: "Fake tool",
    stage: "Tool prediction",
    symptom:
      "The assistant calls refund_customer even though the environment only exposes lookup_order and create_refund_case.",
    remedy:
      "Constrain tool names to the available schema and penalize calls outside the registered tool set.",
  },
  wrongArgument: {
    label: "Wrong argument",
    stage: "Tool prediction",
    symptom:
      "The assistant chooses the right store_locator tool but passes coordinates 0,0 because location was missing.",
    remedy:
      "Add a helper tool or context path for the missing argument, and surface an actionable user request when permission is absent.",
  },
  toolError: {
    label: "Backend error",
    stage: "Tool call",
    symptom:
      "The tool is selected correctly, but the backend returns an error or an incorrect value.",
    remedy:
      "Fix the tool implementation, error handling, or upstream data contract instead of tuning the final response model.",
  },
  synthesis: {
    label: "Bad synthesis",
    stage: "Response generation",
    symptom:
      "The tool returns a precise policy clause, but the final response omits the exception and gives overconfident advice.",
    remedy:
      "Make tool outputs descriptive, trim irrelevant payload, and evaluate whether the answer is grounded in returned data.",
  },
} as const;

type AgentFailureId = keyof typeof agentFailures;

function AgentFailureLab() {
  const [failureId, setFailureId] = useState<AgentFailureId>("wrongArgument");
  const failure = agentFailures[failureId];

  return (
    <section
      id="agent-diagnostics"
      data-testid="agent-failure-lab"
      className="scroll-mt-20 bg-white text-slate-950"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 space-y-4">
          <SectionBadge
            icon={<Wrench aria-hidden="true" size={18} />}
            label="Agent workflows"
            tone="rose"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Localize the failure before choosing a fix
          </h2>
          <p className="text-base leading-7 text-slate-700">
            Tool and agent evaluation should not collapse every miss into a
            generic model failure. The practical taxonomy separates tool
            prediction, tool execution, and final response synthesis so the
            remedy points at the right layer.
          </p>
          <div className="grid gap-3 sm:grid-cols-3">
            <Metric
              label="1. Predict"
              value="tool + args"
              detail="Choose a real tool and valid inputs."
            />
            <Metric
              label="2. Execute"
              value="backend"
              detail="Return a correct, meaningful result."
            />
            <Metric
              label="3. Synthesize"
              value="answer"
              detail="Ground the final message in the result."
            />
          </div>
        </div>

        <div className="min-w-0 rounded-xl border border-slate-300 bg-slate-50 p-4">
          <div className="grid gap-2 sm:grid-cols-2">
            {(Object.keys(agentFailures) as AgentFailureId[]).map((id) => (
              <ControlButton
                key={id}
                tone="rose"
                isActive={failureId === id}
                onClick={() => setFailureId(id)}
              >
                {agentFailures[id].label}
              </ControlButton>
            ))}
          </div>

          <div className="mt-5 rounded-xl border border-slate-300 bg-white p-5">
            <p className="text-sm font-semibold text-rose-700">
              {failure.stage}
            </p>
            <h3 className="mt-2 text-2xl font-semibold text-slate-950">
              {failure.label}
            </h3>
            <p className="mt-3 text-sm leading-6 text-slate-700">
              {failure.symptom}
            </p>
          </div>

          <p
            role="status"
            className="mt-4 rounded-lg border border-rose-500 bg-rose-50 p-4 text-sm leading-6 text-rose-950"
          >
            Targeted remedy: {failure.remedy}
          </p>
        </div>
      </div>
    </section>
  );
}

const benchmarkProfiles = {
  steady: {
    label: "Reliable agent",
    successProbability: 0.8,
    note: "A strong single attempt still becomes much less impressive when every repeated attempt must succeed.",
  },
  brittle: {
    label: "Brittle agent",
    successProbability: 0.45,
    note: "Pass@k can hide brittleness because one lucky success is enough, while Pass^k collapses quickly.",
  },
  excellent: {
    label: "Production candidate",
    successProbability: 0.92,
    note: "High single-attempt reliability is needed when repeated customer workflows should all succeed.",
  },
} as const;

type BenchmarkProfileId = keyof typeof benchmarkProfiles;

const modelProfiles: ParetoModel[] = [
  { id: "A", quality: 90, cost: 20, latency: 800, safety: 96 },
  { id: "B", quality: 88, cost: 12, latency: 500, safety: 94 },
  { id: "C", quality: 86, cost: 12, latency: 600, safety: 91 },
  { id: "D", quality: 82, cost: 6, latency: 350, safety: 93 },
  { id: "E", quality: 88, cost: 14, latency: 450, safety: 97 },
];

function BenchmarkReliabilityLab() {
  const [profileId, setProfileId] = useState<BenchmarkProfileId>("steady");
  const [attempts, setAttempts] = useState(3);
  const profile = benchmarkProfiles[profileId];
  const passAtK = getPassAtK(profile.successProbability, attempts);
  const passHatK = getPassHatK(profile.successProbability, attempts);
  const frontierIds = new Set(
    getParetoFrontier(modelProfiles).map((model) => model.id),
  );

  return (
    <section
      data-testid="benchmark-reliability-lab"
      className="scroll-mt-20 border-y border-slate-200 bg-[#111827] text-slate-50"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.92fr_1.08fr]">
        <div className="min-w-0 space-y-4">
          <SectionBadge
            icon={<LineChart aria-hidden="true" size={18} />}
            label="Benchmarks"
            tone="blue"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Benchmark scores are model profiles, not product answers
          </h2>
          <p className="text-base leading-7 text-slate-300">
            MMLU probes constrained knowledge, AIME probes math reasoning, PIQA
            probes physical common sense, SWE-bench probes code patches,
            HarmBench probes safety behavior, and tau-bench probes tool-agent
            workflows. Each score is a projection, not the full product
            decision.
          </p>
          <FormulaPanel
            title="At least one success versus all attempts succeed"
            formula={String.raw`\[\mathrm{Pass@}k=1-(1-p)^k,\quad \mathrm{Pass}^{k}=p^k\]`}
          >
            <span>
              Pass@k is useful for tasks where one successful sample is enough.
              Agent reliability often needs Pass^k because repeated workflow
              attempts should all succeed.
            </span>
          </FormulaPanel>
        </div>

        <div className="min-w-0 rounded-xl border border-slate-700 bg-slate-950 p-4">
          <div className="grid gap-2 sm:grid-cols-3">
            {(Object.keys(benchmarkProfiles) as BenchmarkProfileId[]).map(
              (id) => (
                <ControlButton
                  key={id}
                  tone="blue"
                  isActive={profileId === id}
                  onClick={() => setProfileId(id)}
                >
                  {benchmarkProfiles[id].label}
                </ControlButton>
              ),
            )}
          </div>
          <div className="mt-4 grid gap-2 sm:grid-cols-3">
            {[1, 3, 5].map((value) => (
              <button
                key={value}
                type="button"
                aria-pressed={attempts === value}
                onClick={() => setAttempts(value)}
                className={`min-h-11 rounded-md border px-3 py-2 text-left text-sm font-semibold ${
                  attempts === value
                    ? "border-emerald-300 bg-emerald-300 text-slate-950"
                    : "border-slate-700 bg-slate-900 text-slate-200 hover:border-slate-500"
                }`}
              >
                k={value}
              </button>
            ))}
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <Metric
              label="Single attempt"
              value={formatPercent(profile.successProbability)}
            />
            <Metric label={`Pass@${attempts}`} value={formatPercent(passAtK)} />
            <Metric
              label={`Pass^${attempts}`}
              value={formatPercent(passHatK)}
            />
          </div>

          <p
            role="status"
            className="mt-4 rounded-lg border border-emerald-400 bg-emerald-950/40 p-4 text-sm leading-6 text-emerald-50"
          >
            Pass@{attempts} = {formatPercent(passAtK)} but Pass^{attempts} ={" "}
            {formatPercent(passHatK)} when the single-attempt success rate is{" "}
            {formatPercent(profile.successProbability)}. {profile.note}
          </p>

          <div className="mt-6 border-t border-slate-800 pt-5">
            <div className="flex flex-wrap items-center gap-3">
              <SectionBadge
                icon={<Gauge aria-hidden="true" size={18} />}
                label="Pareto frontier"
                tone="amber"
              />
              <p className="text-sm leading-6 text-slate-300">
                Higher quality and safety are better; lower cost and latency are
                better. A dominated model is worse on every tracked objective
                than another listed option.
              </p>
            </div>
            <div className="mt-4 grid gap-3 md:grid-cols-5">
              {modelProfiles.map((model) => (
                <div
                  key={model.id}
                  className={`rounded-lg border p-3 ${
                    frontierIds.has(model.id)
                      ? "border-emerald-400 bg-emerald-400/10"
                      : "border-slate-700 bg-slate-900"
                  }`}
                >
                  <p className="text-lg font-semibold">Model {model.id}</p>
                  <p className="mt-2 text-xs leading-5 text-slate-300">
                    quality {model.quality}
                    <br />
                    cost {model.cost}
                    <br />
                    latency {model.latency}
                    <br />
                    safety {model.safety}
                  </p>
                  <p className="mt-2 text-xs font-semibold text-emerald-200">
                    {frontierIds.has(model.id) ? "Frontier" : "Dominated"}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function HeroVisual() {
  const stages = [
    {
      icon: <Scale aria-hidden="true" size={20} />,
      title: "Human signal",
      body: "Rubrics, agreement, calibration.",
    },
    {
      icon: <BrainCircuit aria-hidden="true" size={20} />,
      title: "Judge signal",
      body: "Criteria, rationale, structured score.",
    },
    {
      icon: <Wrench aria-hidden="true" size={20} />,
      title: "Workflow signal",
      body: "Tool prediction, calls, synthesis.",
    },
  ];

  return (
    <div
      aria-label="LLM evaluation instrument panel"
      className="min-w-0 rounded-xl border border-slate-700 bg-slate-900 p-4 shadow-2xl shadow-black/30"
    >
      <div className="rounded-lg border border-blue-400/30 bg-blue-400/10 p-4">
        <div className="flex items-center gap-2 text-blue-100">
          <ClipboardCheck aria-hidden="true" size={18} />
          <p className="font-semibold">Evaluation packet</p>
        </div>
        <p className="mt-2 text-sm leading-6 text-slate-300">
          Prompt, response, criterion, evidence, system trace, and product
          constraints should travel together.
        </p>
      </div>

      <div className="my-4 flex justify-center">
        <ArrowRight aria-hidden="true" className="rotate-90 text-slate-500" />
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        {stages.map((stage) => (
          <div
            key={stage.title}
            className="rounded-lg border border-slate-700 bg-slate-950 p-3"
          >
            <div className="flex items-center gap-2 text-emerald-200">
              {stage.icon}
              <h3 className="font-semibold">{stage.title}</h3>
            </div>
            <p className="mt-2 text-sm leading-5 text-slate-400">
              {stage.body}
            </p>
          </div>
        ))}
      </div>

      <div className="mt-4 rounded-lg border border-amber-400/30 bg-amber-400/10 p-4">
        <div className="flex items-center gap-2 text-amber-100">
          <TriangleAlert aria-hidden="true" size={18} />
          <p className="font-semibold">Failure to avoid</p>
        </div>
        <p className="mt-2 text-sm leading-6 text-slate-300">
          Optimizing a convenient benchmark until it stops representing the user
          task, the workflow, or the deployment constraints.
        </p>
      </div>
    </div>
  );
}

function BenchmarkMap() {
  const items = [
    {
      icon: <FileText aria-hidden="true" className="text-blue-700" />,
      title: "Knowledge",
      body: "MMLU-style tasks ask whether the model can choose the right answer across broad domains.",
    },
    {
      icon: <Calculator aria-hidden="true" className="text-emerald-700" />,
      title: "Reasoning",
      body: "AIME-style math and PIQA-style common sense probe different forms of intermediate inference.",
    },
    {
      icon: <Split aria-hidden="true" className="text-amber-700" />,
      title: "Coding",
      body: "SWE-bench-style evaluation applies patches and checks whether tests pass.",
    },
    {
      icon: <ShieldCheck aria-hidden="true" className="text-rose-700" />,
      title: "Safety",
      body: "HarmBench-style evaluation must reflect the provider policy and the limits of classifier-based scoring.",
    },
    {
      icon: (
        <SlidersHorizontal aria-hidden="true" className="text-violet-700" />
      ),
      title: "Agents",
      body: "Tau-bench-style tasks evaluate tools, simulated users, state changes, rewards, and repeated reliability.",
    },
    {
      icon: <XCircle aria-hidden="true" className="text-slate-700" />,
      title: "Overreach",
      body: "Contamination checks, blocklists, fresh test versions, and Goodhart law keep scores from replacing product judgment.",
    },
  ];

  return (
    <section className="bg-[#f8fafc] text-slate-950">
      <div className="mx-auto w-full max-w-6xl px-4 py-12">
        <div className="max-w-3xl space-y-3">
          <SectionBadge
            icon={<Sparkles aria-hidden="true" size={18} />}
            label="Benchmark map"
            tone="violet"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Choose the benchmark category that matches the capability claim
          </h2>
          <p className="text-base leading-7 text-slate-700">
            Benchmarks are useful when their task design matches the capability
            being claimed. They mislead when a single number is treated as proof
            that a model is best for every workload.
          </p>
        </div>
        <div className="mt-6 grid gap-3 md:grid-cols-3">
          {items.map((item) => (
            <div
              key={item.title}
              className="min-w-0 rounded-lg border border-slate-300 bg-white p-4"
            >
              <div className="flex items-center gap-3">
                <span className="rounded-md bg-slate-100 p-2">{item.icon}</span>
                <h3 className="font-semibold text-slate-950">{item.title}</h3>
              </div>
              <p className="mt-3 text-sm leading-6 text-slate-600">
                {item.body}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Recap() {
  return (
    <section className="border-t border-slate-800 bg-[#0d1720] text-slate-50">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-5 px-4 py-10 md:flex-row md:items-center md:justify-between">
        <div className="max-w-2xl">
          <h2 className="text-2xl font-semibold">Evaluation recap</h2>
          <p className="mt-2 text-sm leading-6 text-slate-300">
            Keep the evaluation packet intact: define the target, calibrate
            human and judge signals, score factual claims with evidence, inspect
            tool-agent stages, and interpret benchmarks as profiles under real
            constraints.
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <Link
            href="/learn/stanford-cme295"
            className="inline-flex items-center justify-center rounded-lg border border-slate-600 px-4 py-3 text-sm font-bold text-slate-100 transition-colors hover:border-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-300 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950"
          >
            Back to course
          </Link>
          <QuizTransitionButton
            sourceId="cme295-lect8"
            label="Practice evaluation questions"
          />
        </div>
      </div>
    </section>
  );
}

export default function StanfordCME295Lecture8EvaluationPage() {
  return (
    <main className="min-h-screen bg-slate-950 text-slate-50">
      <section className="border-b border-slate-800 bg-[#101620]">
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
                icon={<ClipboardCheck aria-hidden="true" size={18} />}
                label="Stanford CME295"
                tone="blue"
              />
              <SectionBadge
                icon={<LineChart aria-hidden="true" size={18} />}
                label="LLM evaluation"
                tone="green"
              />
            </div>
            <div className="space-y-4">
              <h1 className="max-w-3xl text-3xl font-semibold text-slate-50 md:text-5xl md:leading-tight">
                Build the evaluation console before improving the model
              </h1>
              <p className="max-w-2xl text-base leading-7 text-slate-300 md:text-lg">
                Treat evaluation as an instrument panel: choose the target,
                check human agreement, test reference metrics, control judge
                bias, score factual claims, localize agent failures, and read
                benchmarks against cost, latency, safety, and reliability.
              </p>
              <p className="text-sm font-semibold text-emerald-300">
                Evaluation quiz source available
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <QuizTransitionButton
                sourceId="cme295-lect8"
                label="Start evaluation questions"
              />
              <a
                href="#agreement-lab"
                className="inline-flex items-center justify-center rounded-lg border border-slate-600 px-4 py-3 text-sm font-bold text-slate-100 transition-colors hover:border-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-300 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950"
              >
                Open agreement lab
              </a>
              <a
                href="#agent-diagnostics"
                className="inline-flex items-center justify-center rounded-lg border border-slate-600 px-4 py-3 text-sm font-bold text-slate-100 transition-colors hover:border-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-300 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950"
              >
                Diagnose agent failures
              </a>
            </div>
          </div>

          <HeroVisual />
        </div>
      </section>

      <EvaluationScopeRouter />
      <HumanAgreementLab />
      <ReferenceOverlapLab />
      <JudgeBiasLab />
      <FactualityLab />
      <AgentFailureLab />
      <BenchmarkMap />
      <BenchmarkReliabilityLab />
      <Recap />
    </main>
  );
}
