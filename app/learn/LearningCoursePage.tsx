import Link from "next/link";
import {
  getLearningExperiencePath,
  getLearningExperienceSequenceLabel,
  type LearningCourse,
} from "../../lib/learning";
import { getStandaloneLearningPagesForSeries } from "./standaloneLearningPages";

type LearningCoursePageProps = {
  course: LearningCourse;
};

type LearningPageCard = {
  href: string;
  sequenceLabel: string;
  shortTitle: string;
  summary: string;
  actionLabel: string;
};

function getLearningPageSortIndex(sequenceLabel: string): number {
  const sequenceMatch = sequenceLabel.match(/\b(?:lecture|chapter)\s+(\d+)\b/i);
  return sequenceMatch ? Number(sequenceMatch[1]) : Number.MAX_SAFE_INTEGER;
}

export default function LearningCoursePage({
  course,
}: LearningCoursePageProps) {
  const standalonePages = getStandaloneLearningPagesForSeries(course.seriesId);
  const totalLearningPages = course.experiences.length + standalonePages.length;
  const learningPageCards: LearningPageCard[] = [
    ...course.experiences.map((experience) => ({
      href: getLearningExperiencePath(experience),
      sequenceLabel: getLearningExperienceSequenceLabel(experience),
      shortTitle: experience.shortTitle,
      summary: experience.summary,
      actionLabel: "Open learning page",
    })),
    ...standalonePages.map((page) => ({
      href: page.href,
      sequenceLabel: page.sequenceLabel,
      shortTitle: page.shortTitle,
      summary: page.summary,
      actionLabel: "Open standalone page",
    })),
  ].sort(
    (first, second) =>
      getLearningPageSortIndex(first.sequenceLabel) -
        getLearningPageSortIndex(second.sequenceLabel) ||
      first.sequenceLabel.localeCompare(second.sequenceLabel) ||
      first.shortTitle.localeCompare(second.shortTitle),
  );

  return (
    <main className="min-h-[calc(100vh-4.25rem)] bg-slate-950 text-slate-50">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-10 md:py-12">
        <nav aria-label="Learning breadcrumb" className="text-sm">
          <Link
            href="/learn"
            className="font-semibold text-sky-300 hover:text-sky-200"
          >
            Learning
          </Link>
        </nav>

        <section className="max-w-3xl space-y-3">
          <p className="text-sm font-semibold uppercase tracking-wide text-sky-300">
            Course
          </p>
          <h1 className="text-3xl font-semibold tracking-normal text-slate-50 md:text-4xl">
            {course.label}
          </h1>
          <p className="text-base leading-7 text-slate-300">
            Choose a focused learning experience for this course, then move into
            the matching question set when you are ready.
          </p>
          <p className="text-sm font-semibold text-emerald-300">
            {totalLearningPages} learning pages / {course.totalDurationMinutes}{" "}
            quiz-linked min
          </p>
        </section>

        <section
          aria-label={`${course.label} learning experiences`}
          className="grid gap-4 md:grid-cols-2"
        >
          {learningPageCards.map((page) => (
            <Link
              key={page.href}
              href={page.href}
              className="group rounded-lg border border-slate-800 bg-slate-900 p-5 transition-colors hover:border-sky-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950"
            >
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                {page.sequenceLabel}
              </p>
              <h2 className="mt-3 text-xl font-semibold text-slate-50 group-hover:text-sky-200">
                {page.shortTitle}
              </h2>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                {page.summary}
              </p>
              <p className="mt-4 text-sm font-semibold text-sky-300">
                {page.actionLabel}
              </p>
            </Link>
          ))}
        </section>
      </div>
    </main>
  );
}
