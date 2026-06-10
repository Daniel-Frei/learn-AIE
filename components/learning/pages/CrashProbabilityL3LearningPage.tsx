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

const TOKEN_LABELS = ["cat", "dog", "car"] as const;

function formatNumber(value: number): string {
  return value.toFixed(2);
}

function LogitPipelineVisual() {
  const stages = ["Input x", "Network", "Logits z", "Softmax", "P(y | x)"];

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <div className="grid gap-3">
        {stages.map((stage, index) => (
          <div key={stage} className="flex items-center gap-3">
            <div className="flex h-12 flex-1 items-center justify-center rounded-md border border-slate-700 bg-slate-950 px-3 text-sm font-semibold text-slate-100">
              {stage}
            </div>
            {index < stages.length - 1 && (
              <span className="text-lg font-semibold text-sky-300">-&gt;</span>
            )}
          </div>
        ))}
      </div>
      <div className="mt-5 grid grid-cols-3 gap-2">
        {[0.67, 0.24, 0.09].map((probability, index) => (
          <div key={TOKEN_LABELS[index]} className="space-y-2">
            <div className="h-24 rounded-md bg-slate-950 p-2">
              <div
                className="mt-auto rounded-sm bg-sky-400"
                style={{ height: `${Math.max(8, probability * 100)}%` }}
              />
            </div>
            <p className="text-center text-xs font-semibold text-slate-300">
              {TOKEN_LABELS[index]} {probability.toFixed(2)}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

function SoftmaxExplorer() {
  const [logits, setLogits] = useState([2, 1, 0]);
  const values = useMemo(() => {
    const exponentials = logits.map((logit) => Math.exp(logit));
    const total = exponentials.reduce((sum, value) => sum + value, 0);
    return logits.map((logit, index) => ({
      label: TOKEN_LABELS[index],
      logit,
      exp: exponentials[index],
      probability: exponentials[index] / total,
    }));
  }, [logits]);

  const updateLogit = (index: number, value: number) => {
    setLogits((current) =>
      current.map((logit, itemIndex) => (itemIndex === index ? value : logit)),
    );
  };

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-slate-50">
            Softmax explorer
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
            Move the logits and watch the probabilities stay positive and sum to
            one. The largest logit gets the largest probability, but the other
            classes usually keep some probability mass.
          </p>
        </div>
        <div className="rounded-md bg-slate-950 px-3 py-2 text-sm font-semibold text-emerald-300">
          Sum:{" "}
          {formatNumber(
            values.reduce((sum, item) => sum + item.probability, 0),
          )}
        </div>
      </div>

      <div className="mt-5 grid gap-5 md:grid-cols-[0.95fr_1.05fr]">
        <div className="space-y-4">
          {values.map((item, index) => (
            <label key={item.label} className="block space-y-2">
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="font-semibold text-slate-100">
                  {item.label}
                </span>
                <span className="font-mono text-slate-300">
                  z = {formatNumber(item.logit)}
                </span>
              </div>
              <input
                type="range"
                aria-label={`${item.label} logit`}
                min="-3"
                max="5"
                step="0.1"
                value={item.logit}
                onChange={(event) =>
                  updateLogit(index, Number(event.target.value))
                }
                className="w-full accent-sky-400"
              />
            </label>
          ))}
        </div>

        <div className="grid gap-3">
          {values.map((item) => (
            <div
              key={item.label}
              className="rounded-md border border-slate-800 bg-slate-950 p-3"
            >
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="font-semibold text-slate-100">
                  {item.label}
                </span>
                <span className="font-mono text-slate-300">
                  e^z = {formatNumber(item.exp)} / p ={" "}
                  {formatNumber(item.probability)}
                </span>
              </div>
              <div className="mt-2 h-3 overflow-hidden rounded-full bg-slate-800">
                <div
                  className="h-full rounded-full bg-sky-400"
                  style={{ width: `${Math.max(2, item.probability * 100)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function ProbabilityTable({
  rows,
}: {
  rows: readonly { label: string; probability: string }[];
}) {
  return (
    <div className="grid gap-2 text-sm">
      {rows.map((row) => (
        <div
          key={row.label}
          className="flex items-center justify-between gap-4 rounded-md bg-slate-950 px-3 py-2 text-slate-200"
        >
          <span>{row.label}</span>
          <span className="font-mono">{row.probability}</span>
        </div>
      ))}
    </div>
  );
}

export default function CrashProbabilityL3LearningPage({ experience }: Props) {
  return (
    <main className="bg-slate-950 text-slate-50">
      <LearningHero
        eyebrow="Crash Course Probability L3"
        title="Make the observed answer more probable"
        summary="Neural-network training looks less mysterious when you follow one path: raw logits become probabilities, the correct observed output gets scored, and the loss pushes that probability up."
        meta={`${experience.durationMinutes} min interactive prep / ${experience.level}`}
        outcomes={experience.outcomes}
        visual={<LogitPipelineVisual />}
      />

      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-10 md:py-12">
        <ProcessSteps
          title="The training pipeline"
          steps={[
            {
              title: "Score possibilities",
              body: "The network emits one raw score per class, token, or action. These scores are logits, not probabilities.",
            },
            {
              title: "Normalize uncertainty",
              body: "Softmax turns logits into a valid distribution so the model can say how much probability each possibility gets.",
            },
            {
              title: "Penalize the miss",
              body: "Negative log-likelihood and cross-entropy make low probability on the observed answer expensive.",
            },
          ]}
        />

        <section className="grid gap-4 md:grid-cols-3">
          <ConceptCard title="Logit" label="Raw score">
            <p>
              A logit is an unconstrained preference score. It can be negative,
              greater than one, and it does not need to sum to anything.
            </p>
          </ConceptCard>
          <ConceptCard title="Probability" label="Normalized uncertainty">
            <p>
              A probability must be nonnegative and the probabilities across all
              classes must sum to one.
            </p>
          </ConceptCard>
          <ConceptCard title="Decision" label="Chosen output">
            <p>
              A decision or sample happens after the distribution exists. The
              distribution contains more information than the final choice.
            </p>
          </ConceptCard>
        </section>

        <FormulaBlock
          title="Softmax turns scores into a distribution"
          formula="\\[P(y_i)=\\frac{e^{z_i}}{\\sum_j e^{z_j}}\\]"
          explanation="Exponentials make every unnormalized score positive. Dividing by the shared total makes the probabilities sum to one while preserving the ranking implied by the logits."
        />

        <SoftmaxExplorer />

        <CheckForUnderstanding
          testId="logit-check"
          title="Check: logits vs probabilities"
          question="A network outputs logits [3, 1, 0] for classes A, B, and C. What can you conclude before softmax?"
          correctIndex={1}
          options={[
            {
              label: "The three numbers are already probabilities.",
              explanation:
                "Probabilities must satisfy the probability rules. These raw scores do not need to be between zero and one or sum to one.",
            },
            {
              label:
                "Class A has the highest raw score, but the scores still need normalization.",
              explanation:
                "The ranking is meaningful, but softmax is still needed before the model has a valid probability distribution.",
            },
            {
              label:
                "Class C must get zero probability because its logit is zero.",
              explanation:
                "A zero logit becomes e^0 = 1, so it can still receive positive probability after normalization.",
            },
          ]}
        />

        <InteractiveComparison
          title="Likelihood asks about the observed answer"
          prompt="The correct class is dog. Tap each model and decide which one training should prefer."
          items={[
            {
              id: "model-a",
              label: "Model A",
              title: "Higher likelihood",
              body: "Model A assigns 0.70 probability to the observed correct class, dog. For this example, that is the likelihood value that matters.",
              detail: (
                <ProbabilityTable
                  rows={[
                    { label: "cat", probability: "0.20" },
                    { label: "dog", probability: "0.70" },
                    { label: "car", probability: "0.10" },
                  ]}
                />
              ),
            },
            {
              id: "model-b",
              label: "Model B",
              title: "Lower likelihood",
              body: "Model B spreads probability away from the observed answer. It may still assign dog some mass, but 0.30 is worse than 0.70 for this training example.",
              detail: (
                <ProbabilityTable
                  rows={[
                    { label: "cat", probability: "0.40" },
                    { label: "dog", probability: "0.30" },
                    { label: "car", probability: "0.30" },
                  ]}
                />
              ),
            },
          ]}
        />

        <WorkedExample
          title="Worked example: from likelihood to loss"
          setup="For one image, the correct class is cat and the model assigns P(cat | image) = 0.80."
          steps={[
            "Likelihood for this example is 0.80 because the observed label is cat.",
            "Log-likelihood is log(0.80), which is about -0.223 with natural logs.",
            "Negative log-likelihood is -log(0.80), about 0.223. A higher correct probability would make this loss smaller.",
          ]}
        />

        <FormulaBlock
          title="Negative log-likelihood is the loss version"
          formula="\\[\\text{NLL}=-\\log P_\\theta(y_{\\text{true}}\\mid x)\\]"
          explanation="Training usually minimizes a loss. Maximum likelihood becomes a minimization problem by taking the negative log of the probability assigned to the observed answer."
        />

        <MisconceptionCallout
          misconception="If the model chooses the right class, the probability does not matter."
          correction="Accuracy only checks the final choice. Cross-entropy checks how much probability the model assigned to the correct answer. A 0.99 correct prediction is better training signal than a barely-correct 0.51 prediction."
        />

        <InteractiveComparison
          title="Cross-entropy cares about confidence"
          prompt="Both models choose cat if you take the highest probability. Cross-entropy still separates them."
          items={[
            {
              id: "barely",
              label: "Barely correct",
              title: "Correct, but uncertain",
              body: "P(cat) = 0.51 gives loss -log(0.51), about 0.67. It is accurate under argmax, but the model is barely leaning toward the observed class.",
            },
            {
              id: "confident",
              label: "Confidently correct",
              title: "Correct and low loss",
              body: "P(cat) = 0.99 gives loss -log(0.99), about 0.01. Cross-entropy rewards the model for assigning much more probability to the observed class.",
            },
          ]}
        />

        <FormulaBlock
          title="Cross-entropy with one-hot targets"
          formula="\\[H(p,q)=-\\sum_i p_i\\log q_i=-\\log q_{\\text{correct}}\\]"
          explanation="With one-hot labels, every target probability is zero except the observed class. The whole sum collapses to negative log probability of the correct class."
        />

        <section className="grid gap-4 md:grid-cols-2">
          <ConceptCard title="Low entropy" label="Confident distribution">
            <ProbabilityTable
              rows={[
                { label: "Paris", probability: "0.95" },
                { label: "Lyon", probability: "0.03" },
                { label: "London", probability: "0.02" },
              ]}
            />
            <p>
              Probability mass is concentrated on one token, so uncertainty is
              low.
            </p>
          </ConceptCard>
          <ConceptCard title="High entropy" label="Spread distribution">
            <ProbabilityTable
              rows={[
                { label: "Paris", probability: "0.40" },
                { label: "Lyon", probability: "0.30" },
                { label: "London", probability: "0.30" },
              ]}
            />
            <p>
              Several continuations are plausible, so uncertainty is higher.
            </p>
          </ConceptCard>
        </section>

        <CheckForUnderstanding
          title="Check: entropy"
          question="For next-token prediction, which distribution has higher entropy?"
          correctIndex={1}
          options={[
            {
              label: "yes 0.95, no 0.03, maybe 0.02",
              explanation:
                "This distribution is concentrated on one answer, so it has low entropy.",
            },
            {
              label: "yes 0.40, no 0.35, maybe 0.25",
              explanation:
                "The probability mass is spread across more plausible answers, so uncertainty and entropy are higher.",
            },
          ]}
        />

        <RecapSection
          title="Before you start the MCQs"
          items={[
            "Logits are raw scores; softmax converts them into probabilities.",
            "Likelihood is the probability assigned to what actually happened.",
            "Negative log-likelihood turns maximum likelihood into loss minimization.",
            "With one-hot targets, cross-entropy is negative log probability of the correct class.",
            "Entropy measures how spread out a categorical distribution is.",
            "LLM next-token training uses the same idea at vocabulary scale.",
          ]}
        />

        <section className="flex flex-col gap-4 rounded-lg border border-emerald-500/40 bg-emerald-950/20 p-5 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-xl font-semibold text-emerald-100">
              Ready for the Probability L3 questions
            </h2>
            <p className="mt-2 text-sm leading-6 text-slate-300">
              Practice the same concepts with multiple-choice questions on
              logits, softmax, likelihood, loss, cross-entropy, and entropy.
            </p>
          </div>
          <QuizTransitionButton sourceId={experience.sourceId} />
        </section>

        <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
          <h2 className="text-lg font-semibold text-slate-50">
            Compact formula board
          </h2>
          <div className="mt-4 space-y-3 text-slate-200">
            <MathText text="\\[\\text{logits}\\rightarrow\\text{softmax}\\rightarrow P(y\\mid x)\\]" />
            <MathText text="\\[L(\\theta)=\\prod_i P_\\theta(y_i\\mid x_i)\\]" />
            <MathText text="\\[\\text{NLL}=-\\sum_i\\log P_\\theta(y_i\\mid x_i)\\]" />
          </div>
        </section>
      </div>
    </main>
  );
}
