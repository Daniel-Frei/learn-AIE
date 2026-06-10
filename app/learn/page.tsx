import Link from "next/link";
import {
  LEARNING_EXPERIENCES,
  getQuestionSourceForLearningExperience,
} from "../../lib/learning";

export default function LearningIndexPage() {
  return (
    <main className="min-h-[calc(100vh-4.25rem)] bg-slate-950 text-slate-50">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-10 md:py-12">
        <section className="max-w-3xl space-y-3">
          <p className="text-sm font-semibold uppercase tracking-wide text-sky-300">
            Learning
          </p>
          <h1 className="text-3xl font-semibold tracking-normal text-slate-50 md:text-4xl">
            Prepare before the questions
          </h1>
          <p className="text-base leading-7 text-slate-300">
            Short interactive pages that introduce the core idea, examples, and
            common confusions before you start the matching multiple-choice
            practice.
          </p>
        </section>

        <section
          aria-label="Available learning experiences"
          className="grid gap-4 md:grid-cols-2"
        >
          {LEARNING_EXPERIENCES.map((experience) => {
            const source = getQuestionSourceForLearningExperience(experience);

            return (
              <Link
                key={experience.sourceId}
                href={`/learn/${experience.sourceId}`}
                className="group rounded-lg border border-slate-800 bg-slate-900 p-5 transition-colors hover:border-sky-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950"
              >
                <div className="flex flex-wrap items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                  <span>{source?.seriesLabel ?? "Learning AI"}</span>
                  <span aria-hidden="true">/</span>
                  <span>{experience.durationMinutes} min</span>
                </div>
                <h2 className="mt-3 text-xl font-semibold text-slate-50 group-hover:text-sky-200">
                  {experience.shortTitle}
                </h2>
                <p className="mt-2 text-sm leading-6 text-slate-300">
                  {experience.summary}
                </p>
                <p className="mt-4 text-sm font-semibold text-sky-300">
                  Open learning page
                </p>
              </Link>
            );
          })}
        </section>
      </div>
    </main>
  );
}
