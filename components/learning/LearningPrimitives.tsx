"use client";

import Link from "next/link";
import { useState, type ReactNode } from "react";
import MathText from "../MathText";
import type { SourceId } from "../../lib/quiz";

type LearningHeroProps = {
  eyebrow: string;
  title: string;
  summary: string;
  meta: string;
  outcomes: readonly string[];
  visual?: ReactNode;
};

export function LearningHero({
  eyebrow,
  title,
  summary,
  meta,
  outcomes,
  visual,
}: LearningHeroProps) {
  return (
    <section className="border-b border-slate-800 bg-slate-950 text-slate-50">
      <div className="mx-auto grid min-h-[520px] w-full max-w-6xl items-center gap-8 px-4 py-10 md:grid-cols-[1.05fr_0.95fr] md:py-14">
        <div className="space-y-6">
          <div className="space-y-3">
            <p className="text-sm font-semibold uppercase tracking-wide text-sky-300">
              {eyebrow}
            </p>
            <h1 className="max-w-3xl text-3xl font-semibold tracking-normal text-slate-50 md:text-5xl md:leading-tight">
              {title}
            </h1>
            <p className="max-w-2xl text-base leading-7 text-slate-300 md:text-lg">
              {summary}
            </p>
            <p className="text-sm font-semibold text-emerald-300">{meta}</p>
          </div>

          <div className="grid gap-2 sm:grid-cols-2">
            {outcomes.map((outcome) => (
              <div
                key={outcome}
                className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-sm leading-5 text-slate-200"
              >
                {outcome}
              </div>
            ))}
          </div>
        </div>

        {visual && <div>{visual}</div>}
      </div>
    </section>
  );
}

type ConceptCardProps = {
  title: string;
  label?: string;
  children: ReactNode;
};

export function ConceptCard({ title, label, children }: ConceptCardProps) {
  return (
    <article className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      {label && (
        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-amber-300">
          {label}
        </p>
      )}
      <h2 className="text-xl font-semibold text-slate-50">{title}</h2>
      <div className="mt-3 space-y-3 text-sm leading-6 text-slate-300">
        {children}
      </div>
    </article>
  );
}

type FormulaBlockProps = {
  title: string;
  formula: string;
  explanation: string;
};

export function FormulaBlock({
  title,
  formula,
  explanation,
}: FormulaBlockProps) {
  return (
    <section className="rounded-lg border border-sky-500/40 bg-sky-950/30 p-5">
      <h2 className="text-lg font-semibold text-sky-100">{title}</h2>
      <MathText
        text={formula}
        className="mt-4 overflow-x-auto rounded-md border border-sky-500/30 bg-slate-950 px-4 py-3 text-sky-50"
      />
      <p className="mt-4 text-sm leading-6 text-slate-300">{explanation}</p>
    </section>
  );
}

type WorkedExampleProps = {
  title: string;
  setup: string;
  steps: readonly string[];
};

export function WorkedExample({ title, setup, steps }: WorkedExampleProps) {
  return (
    <section className="rounded-lg border border-emerald-500/40 bg-emerald-950/20 p-5">
      <h2 className="text-lg font-semibold text-emerald-100">{title}</h2>
      <p className="mt-2 text-sm leading-6 text-slate-300">{setup}</p>
      <ol className="mt-4 space-y-3">
        {steps.map((step, index) => (
          <li key={step} className="flex gap-3">
            <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-emerald-400 text-sm font-bold text-slate-950">
              {index + 1}
            </span>
            <span className="pt-1 text-sm leading-6 text-slate-200">
              {step}
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}

type ComparisonItem = {
  id: string;
  label: string;
  title: string;
  body: string;
  detail?: ReactNode;
};

type InteractiveComparisonProps = {
  title: string;
  prompt: string;
  items: readonly ComparisonItem[];
};

export function InteractiveComparison({
  title,
  prompt,
  items,
}: InteractiveComparisonProps) {
  const [activeId, setActiveId] = useState(items[0]?.id ?? "");
  const activeItem = items.find((item) => item.id === activeId) ?? items[0];

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <h2 className="text-lg font-semibold text-slate-50">{title}</h2>
      <p className="mt-2 text-sm leading-6 text-slate-300">{prompt}</p>
      <div className="mt-4 flex flex-wrap gap-2">
        {items.map((item) => (
          <button
            key={item.id}
            type="button"
            onClick={() => setActiveId(item.id)}
            className={`rounded-md border px-3 py-2 text-sm font-semibold ${
              item.id === activeItem.id
                ? "border-sky-400 bg-sky-400 text-slate-950"
                : "border-slate-700 text-slate-200 hover:border-slate-500"
            }`}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div className="mt-5 border-t border-slate-800 pt-4">
        <h3 className="text-base font-semibold text-slate-100">
          {activeItem.title}
        </h3>
        <p className="mt-2 text-sm leading-6 text-slate-300">
          {activeItem.body}
        </p>
        {activeItem.detail && <div className="mt-4">{activeItem.detail}</div>}
      </div>
    </section>
  );
}

type CheckOption = {
  label: string;
  explanation: string;
};

type CheckForUnderstandingProps = {
  title: string;
  question: string;
  options: readonly CheckOption[];
  correctIndex: number;
  testId?: string;
};

export function CheckForUnderstanding({
  title,
  question,
  options,
  correctIndex,
  testId,
}: CheckForUnderstandingProps) {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const selectedOption =
    selectedIndex === null ? null : (options[selectedIndex] ?? null);
  const isCorrect = selectedIndex === correctIndex;

  return (
    <section
      data-testid={testId}
      className="rounded-lg border border-violet-500/40 bg-violet-950/20 p-5"
    >
      <h2 className="text-lg font-semibold text-violet-100">{title}</h2>
      <p className="mt-2 text-sm leading-6 text-slate-300">{question}</p>
      <div className="mt-4 grid gap-2">
        {options.map((option, index) => {
          const isSelected = selectedIndex === index;
          return (
            <button
              key={option.label}
              type="button"
              onClick={() => setSelectedIndex(index)}
              className={`rounded-md border px-3 py-2 text-left text-sm leading-5 transition-colors ${
                isSelected
                  ? "border-violet-300 bg-violet-400 text-slate-950"
                  : "border-slate-700 bg-slate-950 text-slate-200 hover:border-slate-500"
              }`}
            >
              {option.label}
            </button>
          );
        })}
      </div>
      {selectedOption && (
        <p
          role="status"
          className={`mt-4 rounded-md border px-3 py-2 text-sm leading-6 ${
            isCorrect
              ? "border-emerald-400 bg-emerald-950/40 text-emerald-100"
              : "border-amber-400 bg-amber-950/30 text-amber-100"
          }`}
        >
          <span className="font-semibold">
            {isCorrect ? "Correct." : "Not yet."}
          </span>{" "}
          {selectedOption.explanation}
        </p>
      )}
    </section>
  );
}

type MisconceptionCalloutProps = {
  misconception: string;
  correction: string;
};

export function MisconceptionCallout({
  misconception,
  correction,
}: MisconceptionCalloutProps) {
  return (
    <section className="rounded-lg border border-amber-500/40 bg-amber-950/20 p-5">
      <p className="text-xs font-semibold uppercase tracking-wide text-amber-300">
        Misconception check
      </p>
      <h2 className="mt-2 text-lg font-semibold text-amber-100">
        {misconception}
      </h2>
      <p className="mt-3 text-sm leading-6 text-slate-200">{correction}</p>
    </section>
  );
}

type ProcessStepsProps = {
  title: string;
  steps: readonly {
    title: string;
    body: string;
  }[];
};

export function ProcessSteps({ title, steps }: ProcessStepsProps) {
  return (
    <section>
      <h2 className="text-2xl font-semibold text-slate-50">{title}</h2>
      <div className="mt-5 grid gap-3 md:grid-cols-3">
        {steps.map((step, index) => (
          <article
            key={step.title}
            className="rounded-lg border border-slate-800 bg-slate-900 p-4"
          >
            <p className="text-sm font-semibold text-sky-300">
              Step {index + 1}
            </p>
            <h3 className="mt-2 text-base font-semibold text-slate-100">
              {step.title}
            </h3>
            <p className="mt-2 text-sm leading-6 text-slate-300">{step.body}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

type RecapSectionProps = {
  title: string;
  items: readonly string[];
};

export function RecapSection({ title, items }: RecapSectionProps) {
  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <h2 className="text-xl font-semibold text-slate-50">{title}</h2>
      <ul className="mt-4 grid gap-3 text-sm leading-6 text-slate-300 md:grid-cols-2">
        {items.map((item) => (
          <li key={item} className="rounded-md bg-slate-950 px-3 py-2">
            {item}
          </li>
        ))}
      </ul>
    </section>
  );
}

type QuizTransitionButtonProps = {
  sourceId: SourceId;
  label?: string;
};

export function QuizTransitionButton({
  sourceId,
  label = "Start questions",
}: QuizTransitionButtonProps) {
  return (
    <Link
      href={`/?source=${sourceId}`}
      className="inline-flex items-center justify-center rounded-lg bg-emerald-400 px-4 py-3 text-sm font-bold text-slate-950 transition-colors hover:bg-emerald-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-300 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950"
    >
      {label}
    </Link>
  );
}
