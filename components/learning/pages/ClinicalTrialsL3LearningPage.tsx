"use client";

import { useMemo, useState } from "react";
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

const evidenceLensItems = [
  {
    label: "Effect size",
    body: "How large is the benefit or harm, and which metric is being used?",
  },
  {
    label: "Precision",
    body: "How uncertain is the estimate, and what does the confidence interval include?",
  },
  {
    label: "Clinical meaning",
    body: "Does the outcome, size, safety profile, burden, and cost matter to patients?",
  },
  {
    label: "Total evidence",
    body: "Does this result fit with prior trials, biology, study quality, and missing-data concerns?",
  },
] as const;

const synthesisDetails = {
  pooled:
    "The pooled estimate summarizes included studies, but it inherits their design choices, endpoints, populations, and bias risks.",
  heterogeneity:
    "Heterogeneity asks whether the studies are similar enough that one combined number has a coherent meaning.",
  bias: "Publication bias asks whether the visible evidence base is missing negative, small, or inconclusive studies.",
} as const;

function formatPercent(value: number): string {
  return `${Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1)}%`;
}

function formatNumber(value: number): string {
  return value.toFixed(2);
}

function EvidenceLensVisual() {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <p className="text-sm font-semibold uppercase tracking-wide text-sky-300">
        Do not stop at positive
      </p>
      <div className="mt-4 grid gap-3">
        {evidenceLensItems.map((item, index) => (
          <div
            key={item.label}
            className="grid grid-cols-[2.5rem_1fr] gap-3 rounded-md border border-slate-800 bg-slate-950 p-3"
          >
            <span className="flex h-9 w-9 items-center justify-center rounded-full bg-sky-400 text-sm font-bold text-slate-950">
              {index + 1}
            </span>
            <div>
              <h2 className="text-sm font-semibold text-slate-100">
                {item.label}
              </h2>
              <p className="mt-1 text-xs leading-5 text-slate-300">
                {item.body}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TreatmentEffectExplorer() {
  const [controlRisk, setControlRisk] = useState(10);
  const [treatmentRisk, setTreatmentRisk] = useState(5);

  const result = useMemo(() => {
    const arr = controlRisk - treatmentRisk;
    const riskRatio = treatmentRisk / controlRisk;
    const relativeReduction = (arr / controlRisk) * 100;
    const nnt = arr > 0 ? Math.ceil(100 / arr) : null;
    const harmNumber = arr < 0 ? Math.ceil(100 / Math.abs(arr)) : null;

    return {
      arr,
      riskRatio,
      relativeReduction,
      nnt,
      harmNumber,
    };
  }, [controlRisk, treatmentRisk]);

  const absoluteImpact =
    result.arr > 0
      ? `${formatPercent(result.arr)} fewer events per 100 patients`
      : result.arr < 0
        ? `${formatPercent(Math.abs(result.arr))} more events per 100 patients`
        : "No absolute difference in events per 100 patients";

  const patientScale =
    result.nnt !== null
      ? `Treat about ${result.nnt} patients to prevent one additional event.`
      : result.harmNumber !== null
        ? `Treating about ${result.harmNumber} patients would cause one additional event.`
        : "NNT is not defined when there is no absolute risk difference.";

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-slate-50">
            Treatment effect explorer
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
            Move the event rates and watch the same result become absolute risk
            reduction, relative risk reduction, risk ratio, and NNT. The
            patient-level story depends on baseline risk.
          </p>
        </div>
        <div className="rounded-md bg-slate-950 px-3 py-2 text-sm font-semibold text-emerald-300">
          {absoluteImpact}
        </div>
      </div>

      <div className="mt-5 grid gap-5 md:grid-cols-[0.95fr_1.05fr]">
        <div className="space-y-4">
          <label className="block space-y-2">
            <div className="flex items-center justify-between gap-3 text-sm">
              <span className="font-semibold text-slate-100">
                Control event risk
              </span>
              <span className="font-mono text-slate-300">
                {formatPercent(controlRisk)}
              </span>
            </div>
            <input
              type="range"
              aria-label="Control event risk"
              min="1"
              max="40"
              step="1"
              value={controlRisk}
              onChange={(event) => setControlRisk(Number(event.target.value))}
              className="w-full accent-sky-400"
            />
          </label>

          <label className="block space-y-2">
            <div className="flex items-center justify-between gap-3 text-sm">
              <span className="font-semibold text-slate-100">
                Treatment event risk
              </span>
              <span className="font-mono text-slate-300">
                {formatPercent(treatmentRisk)}
              </span>
            </div>
            <input
              type="range"
              aria-label="Treatment event risk"
              min="0"
              max="40"
              step="1"
              value={treatmentRisk}
              onChange={(event) => setTreatmentRisk(Number(event.target.value))}
              className="w-full accent-emerald-400"
            />
          </label>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <ResultTile
            label="ARR"
            value={formatPercent(result.arr)}
            note="Control risk minus treatment risk"
          />
          <ResultTile
            label="RRR"
            value={`${formatNumber(result.relativeReduction)}%`}
            note="Proportional reduction from baseline"
          />
          <ResultTile
            label="Risk ratio"
            value={formatNumber(result.riskRatio)}
            note="Treatment risk divided by control risk"
          />
          <ResultTile label="NNT / NNH" value={patientScale} note="" />
        </div>
      </div>
    </section>
  );
}

function ResultTile({
  label,
  value,
  note,
}: {
  label: string;
  value: string;
  note: string;
}) {
  return (
    <div className="rounded-md border border-slate-800 bg-slate-950 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
        {label}
      </p>
      <p className="mt-2 text-lg font-semibold text-slate-50">{value}</p>
      {note && <p className="mt-1 text-xs leading-5 text-slate-400">{note}</p>}
    </div>
  );
}

function ConfidenceIntervalStrip({
  estimate,
  low,
  high,
  label,
}: {
  estimate: number;
  low: number;
  high: number;
  label: string;
}) {
  const scaleMin = 0;
  const scaleMax = 2.6;
  const toPercent = (value: number) =>
    ((value - scaleMin) / (scaleMax - scaleMin)) * 100;
  const left = Math.max(0, toPercent(low));
  const right = Math.min(100, toPercent(high));
  const estimateLeft = Math.min(100, Math.max(0, toPercent(estimate)));
  const noEffectLeft = toPercent(1);

  return (
    <div className="rounded-md bg-slate-950 p-3">
      <div className="flex items-center justify-between gap-3 text-sm">
        <span className="font-semibold text-slate-100">{label}</span>
        <span className="font-mono text-slate-300">
          {formatNumber(estimate)} ({formatNumber(low)}-{formatNumber(high)})
        </span>
      </div>
      <div className="relative mt-4 h-8">
        <div className="absolute left-0 right-0 top-1/2 h-px bg-slate-700" />
        <div
          className="absolute top-0 h-8 w-px bg-amber-300"
          style={{ left: `${noEffectLeft}%` }}
        />
        <div
          className="absolute top-1/2 h-1 -translate-y-1/2 rounded-full bg-sky-400"
          style={{ left: `${left}%`, width: `${Math.max(2, right - left)}%` }}
        />
        <div
          className="absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full bg-emerald-300 ring-2 ring-slate-950"
          style={{ left: `${estimateLeft}%` }}
        />
      </div>
      <p className="mt-1 text-xs text-slate-400">No effect for ratios = 1.0</p>
    </div>
  );
}

function SurvivalCurveSketch() {
  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <div className="grid gap-5 md:grid-cols-[1.1fr_0.9fr]">
        <div>
          <h2 className="text-xl font-semibold text-slate-50">
            Read the curve before the hazard ratio
          </h2>
          <p className="mt-2 text-sm leading-6 text-slate-300">
            Time-to-event results preserve when events happen and how many
            patients are still being followed. A single hazard ratio can hide
            delayed separation, crossing curves, or unstable late tails.
          </p>
          <div className="mt-5 overflow-hidden rounded-md border border-slate-800 bg-slate-950 p-4">
            <svg
              role="img"
              aria-label="Kaplan-Meier style survival curve comparing treatment and control over time"
              viewBox="0 0 360 210"
              className="h-auto w-full"
            >
              <line
                x1="42"
                y1="20"
                x2="42"
                y2="170"
                stroke="#475569"
                strokeWidth="2"
              />
              <line
                x1="42"
                y1="170"
                x2="330"
                y2="170"
                stroke="#475569"
                strokeWidth="2"
              />
              <polyline
                points="42,28 82,42 118,62 162,84 210,107 258,131 310,151"
                fill="none"
                stroke="#38bdf8"
                strokeWidth="4"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <polyline
                points="42,28 82,48 118,75 162,104 210,128 258,148 310,162"
                fill="none"
                stroke="#f59e0b"
                strokeWidth="4"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <text x="46" y="18" fill="#cbd5e1" fontSize="12">
                Event-free
              </text>
              <text x="286" y="192" fill="#cbd5e1" fontSize="12">
                Time
              </text>
              <text x="236" y="96" fill="#7dd3fc" fontSize="13">
                Treatment
              </text>
              <text x="238" y="145" fill="#fbbf24" fontSize="13">
                Control
              </text>
              <circle cx="258" cy="131" r="4" fill="#38bdf8" />
              <circle cx="258" cy="148" r="4" fill="#f59e0b" />
            </svg>
          </div>
        </div>

        <div className="space-y-3">
          <ConceptCard title="Censoring" label="Partial information">
            <p>
              A patient who remains event-free until the last observed visit
              still contributes follow-up time. They are not simply counted as
              an event or ignored.
            </p>
          </ConceptCard>
          <ConceptCard title="Median survival" label="Intuitive but incomplete">
            <p>
              Median survival is the time by which half the group has had the
              event. It can miss long-term tails or delayed benefits.
            </p>
          </ConceptCard>
          <ConceptCard title="Hazard ratio" label="Event rate over time">
            <p>
              A hazard ratio compares event rates among patients still at risk,
              not the cumulative percentage of patients who had the event.
            </p>
          </ConceptCard>
        </div>
      </div>
    </section>
  );
}

function ForestPlotSketch() {
  const [activeLens, setActiveLens] =
    useState<keyof typeof synthesisDetails>("pooled");
  const rows = [
    { study: "Trial A", estimate: 0.92, low: 0.64, high: 1.28, weight: "18%" },
    { study: "Trial B", estimate: 0.72, low: 0.55, high: 0.94, weight: "42%" },
    { study: "Trial C", estimate: 1.08, low: 0.7, high: 1.62, weight: "12%" },
    { study: "Trial D", estimate: 0.8, low: 0.66, high: 0.98, weight: "28%" },
  ] as const;
  const scaleMin = 0.4;
  const scaleMax = 1.7;
  const toPercent = (value: number) =>
    ((value - scaleMin) / (scaleMax - scaleMin)) * 100;
  const noEffectLeft = toPercent(1);

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-slate-50">
            Forest plot reader
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
            A meta-analysis is not a vote count. Read individual estimates,
            confidence intervals, the no-effect line, study weights, and the
            pooled estimate, then ask whether pooling makes sense.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {(["pooled", "heterogeneity", "bias"] as const).map((lens) => (
            <button
              key={lens}
              type="button"
              onClick={() => setActiveLens(lens)}
              className={`rounded-md border px-3 py-2 text-sm font-semibold ${
                activeLens === lens
                  ? "border-sky-400 bg-sky-400 text-slate-950"
                  : "border-slate-700 text-slate-200 hover:border-slate-500"
              }`}
            >
              {lens === "pooled"
                ? "Pooled estimate"
                : lens === "heterogeneity"
                  ? "Heterogeneity"
                  : "Publication bias"}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-5 rounded-md border border-slate-800 bg-slate-950 p-4">
        <div className="grid grid-cols-[5rem_1fr_3rem] gap-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
          <span>Study</span>
          <span>Risk ratio and CI</span>
          <span>Weight</span>
        </div>
        <div className="relative mt-3 space-y-3">
          {rows.map((row) => {
            const left = toPercent(row.low);
            const right = toPercent(row.high);
            const estimate = toPercent(row.estimate);

            return (
              <div
                key={row.study}
                className="grid grid-cols-[5rem_1fr_3rem] items-center gap-3 text-sm"
              >
                <span className="font-semibold text-slate-200">
                  {row.study}
                </span>
                <div className="relative h-7">
                  <div className="absolute left-0 right-0 top-1/2 h-px bg-slate-700" />
                  <div
                    className="absolute top-0 h-7 w-px bg-amber-300"
                    style={{ left: `${noEffectLeft}%` }}
                  />
                  <div
                    className="absolute top-1/2 h-1 -translate-y-1/2 rounded-full bg-sky-400"
                    style={{
                      left: `${left}%`,
                      width: `${Math.max(2, right - left)}%`,
                    }}
                  />
                  <div
                    className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-sm bg-emerald-300"
                    style={{ left: `${estimate}%` }}
                  />
                </div>
                <span className="font-mono text-slate-300">{row.weight}</span>
              </div>
            );
          })}
          <div className="grid grid-cols-[5rem_1fr_3rem] items-center gap-3 border-t border-slate-800 pt-3 text-sm">
            <span className="font-semibold text-emerald-200">Pooled</span>
            <div className="relative h-7">
              <div className="absolute left-0 right-0 top-1/2 h-px bg-slate-700" />
              <div
                className="absolute top-0 h-7 w-px bg-amber-300"
                style={{ left: `${noEffectLeft}%` }}
              />
              <div
                className="absolute top-1/2 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rotate-45 bg-emerald-400"
                style={{ left: `${toPercent(0.82)}%` }}
              />
              <div
                className="absolute top-1/2 h-px -translate-y-1/2 bg-emerald-200"
                style={{
                  left: `${toPercent(0.7)}%`,
                  width: `${toPercent(0.96) - toPercent(0.7)}%`,
                }}
              />
            </div>
            <span className="font-mono text-slate-300">100%</span>
          </div>
        </div>
      </div>

      <p role="status" className="mt-4 text-sm leading-6 text-slate-300">
        {synthesisDetails[activeLens]}
      </p>
    </section>
  );
}

export default function ClinicalTrialsL3LearningPage({ experience }: Props) {
  return (
    <main className="bg-slate-950 text-slate-50">
      <LearningHero
        eyebrow="Clinical Trials Crash Course L3"
        title="Read the size, uncertainty, and clinical meaning"
        summary="A trial result should not collapse into positive or negative. Interpret what changed, how certain the estimate is, whether patients would care, and how the result fits the broader evidence base."
        meta={`${experience.durationMinutes} min interactive prep / ${experience.level}`}
        outcomes={experience.outcomes}
        visual={<EvidenceLensVisual />}
      />

      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-10 md:py-12">
        <ProcessSteps
          title="The four-pass reading habit"
          steps={[
            {
              title: "Translate the effect",
              body: "Find the absolute risk difference, relative measure, NNT, endpoint, and time frame.",
            },
            {
              title: "Inspect uncertainty",
              body: "Use the confidence interval and event count to judge whether no effect, meaningful benefit, or harm remain plausible.",
            },
            {
              title: "Make it clinical",
              body: "Ask whether the magnitude matters after safety, burden, alternatives, patient values, and total evidence are included.",
            },
          ]}
        />

        <section className="grid gap-4 md:grid-cols-3">
          <ConceptCard title="Absolute effect" label="Patient impact">
            <p>
              Absolute risk reduction asks how many fewer events happen in the
              treated group. It is often the fastest bridge from statistics to
              clinical meaning.
            </p>
          </ConceptCard>
          <ConceptCard title="Relative effect" label="Framing">
            <p>
              Relative risk reduction asks how large the proportional change is.
              It can sound impressive even when baseline risk is low.
            </p>
          </ConceptCard>
          <ConceptCard title="NNT" label="Patient scale">
            <p>
              Number needed to treat converts absolute benefit into how many
              patients need treatment for one additional outcome.
            </p>
          </ConceptCard>
        </section>

        <FormulaBlock
          title="Absolute benefit drives NNT"
          formula={String.raw`\[\text{ARR}=R_{\text{control}}-R_{\text{treatment}}\qquad \text{NNT}=\frac{1}{\text{ARR as a proportion}}\]`}
          explanation="A 50% relative reduction can mean 20% to 10% or 2% to 1%. The proportional story is the same, but the absolute impact and NNT are very different."
        />

        <TreatmentEffectExplorer />

        <CheckForUnderstanding
          testId="absolute-risk-check"
          title="Check: headline versus patient impact"
          question="A sponsor highlights a 50% relative risk reduction from 2% to 1% for a mild endpoint. What should you ask next?"
          correctIndex={1}
          options={[
            {
              label:
                "Nothing else; a 50% relative reduction is always clinically large.",
              explanation:
                "Relative framing alone can overstate patient impact when baseline risk is low.",
            },
            {
              label:
                "What is the absolute benefit, NNT, endpoint importance, burden, cost, and alternatives?",
              explanation:
                "The absolute reduction is 1 percentage point, so interpretation needs patient relevance and benefit-risk context.",
            },
            {
              label:
                "Only whether p is below 0.05, because absolute risk is secondary.",
              explanation:
                "The p-value does not tell you whether the benefit is large enough to matter.",
            },
          ]}
        />

        <InteractiveComparison
          title="Confidence intervals show what remains plausible"
          prompt="Both studies report a favorable point estimate. Tap each result and compare precision before deciding which claim is more stable."
          items={[
            {
              id: "narrow",
              label: "Study A",
              title: "Less dramatic, much more precise",
              body: "RR = 0.80 with a 95% CI from 0.78 to 0.82. The point estimate is modest, but the interval is tight and excludes no effect.",
              detail: (
                <ConfidenceIntervalStrip
                  label="Study A"
                  estimate={0.8}
                  low={0.78}
                  high={0.82}
                />
              ),
            },
            {
              id: "wide",
              label: "Study B",
              title: "Dramatic point estimate, weak precision",
              body: "RR = 0.50 with a 95% CI from 0.10 to 2.50. The data remain compatible with large benefit, no effect, or harm.",
              detail: (
                <ConfidenceIntervalStrip
                  label="Study B"
                  estimate={0.5}
                  low={0.1}
                  high={2.5}
                />
              ),
            },
          ]}
        />

        <MisconceptionCallout
          misconception="A p-value is the probability that the treatment works."
          correction="A p-value is calculated under the null hypothesis. It asks how surprising data this extreme would be if no treatment effect existed under the test assumptions; it does not give the probability that the treatment works."
        />

        <InteractiveComparison
          title="Statistical significance is not clinical significance"
          prompt="Tap each abstract-style result and decide why a threshold alone is insufficient."
          items={[
            {
              id: "tiny",
              label: "Huge trial",
              title: "Precise but clinically tiny",
              body: "A 100,000-patient trial lowers systolic blood pressure by 0.5 mmHg with p < 0.001. The study may prove a nonzero effect while still failing to change symptoms, decisions, or outcomes.",
            },
            {
              id: "rare",
              label: "Rare cancer",
              title: "Promising but uncertain",
              body: "A 40-patient rare-cancer study estimates a six-month median survival gain with p = 0.08. It is not definitive, but the signal may justify more evidence if the disease context and safety support it.",
            },
          ]}
        />

        <WorkedExample
          title="Worked example: read a positive abstract"
          setup="A randomized trial reports HR = 0.82, 95% CI 0.70 to 0.96, p = 0.01 for a composite endpoint of hospitalization or biomarker worsening."
          steps={[
            "Start with the metric: HR = 0.82 suggests a lower event rate over time, and the confidence interval excludes the no-effect value of 1.",
            "Ask for absolute event rates, because a hazard ratio does not reveal the patient-level risk difference.",
            "Inspect the composite endpoint: if biomarker worsening drives most events, the result may be less patient-important than fewer hospitalizations.",
            "Finish with safety, quality of life, follow-up, missing data, prior evidence, and whether the analysis was pre-specified.",
          ]}
        />

        <SurvivalCurveSketch />

        <CheckForUnderstanding
          testId="hazard-check"
          title="Check: hazard ratio"
          question="A trial reports a hazard ratio of 0.75 for hospitalization. Which interpretation is most accurate?"
          correctIndex={2}
          options={[
            {
              label:
                "Exactly 25% fewer treated patients were hospitalized by the end of follow-up.",
              explanation:
                "That would be a cumulative risk statement, not a hazard-ratio interpretation.",
            },
            {
              label:
                "The absolute hospitalization risk reduction is 25 percentage points.",
              explanation: "A hazard ratio is not an absolute risk difference.",
            },
            {
              label:
                "The estimated event rate over time is about 25% lower under the model assumptions.",
              explanation:
                "A hazard ratio compares event rates over time among participants still at risk.",
            },
          ]}
        />

        <ForestPlotSketch />

        <CheckForUnderstanding
          title="Check: meta-analysis"
          question="A meta-analysis pools trials with different populations, doses, endpoints, follow-up windows, and study quality. What is the best next question?"
          correctIndex={0}
          options={[
            {
              label:
                "Are these studies similar enough that the pooled estimate has a meaningful interpretation?",
              explanation:
                "Clinical and methodological heterogeneity can make a single pooled number misleading.",
            },
            {
              label:
                "Can the pooled estimate replace all concerns about bias and study quality?",
              explanation:
                "Pooling cannot rescue biased, incomparable, or selectively published evidence.",
            },
            {
              label:
                "Does having more studies automatically remove publication bias?",
              explanation:
                "More visible studies do not prove the evidence base is complete.",
            },
          ]}
        />

        <RecapSection
          title="Before you start the MCQs"
          items={[
            "Relative risk reductions need baseline risk and absolute effects.",
            "NNT translates absolute benefit into patient-level scale.",
            "Confidence intervals show precision and clinically plausible effects.",
            "P-values are not probabilities that a treatment works.",
            "Clinical meaning depends on endpoints, effect size, safety, burden, alternatives, and patient values.",
            "Hazard ratios need Kaplan-Meier curves, absolute survival, censoring, and number-at-risk context.",
            "Forest plots and meta-analyses require study quality, heterogeneity, and publication-bias checks.",
            "A positive abstract is a starting point for interpretation, not the conclusion.",
          ]}
        />

        <section className="flex flex-col gap-4 rounded-lg border border-emerald-500/40 bg-emerald-950/20 p-5 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-xl font-semibold text-emerald-100">
              Ready for the Clinical Trials L3 questions
            </h2>
            <p className="mt-2 text-sm leading-6 text-slate-300">
              Practice interpreting treatment effects, confidence intervals,
              p-values, clinical significance, survival results, forest plots,
              and evidence synthesis.
            </p>
          </div>
          <QuizTransitionButton sourceId={experience.sourceId} />
        </section>
      </div>
    </main>
  );
}
