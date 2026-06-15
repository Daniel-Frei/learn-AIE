import Link from "next/link";
import { getLearningCoursePath, getLearningCourses } from "../../lib/learning";

const standaloneLearningCourses = [
  {
    href: "/learn/crash-course-medicine",
    label: "Crash Course Medicine",
    meta: "1 standalone page",
    summary:
      "Build the clinical reasoning foundation before moving into organ systems, diagnosis, treatment pathways, and evidence.",
  },
] as const;

export default function LearningIndexPage() {
  const courses = getLearningCourses();

  return (
    <main className="min-h-[calc(100vh-4.25rem)] bg-slate-950 text-slate-50">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-10 md:py-12">
        <section className="max-w-3xl space-y-3">
          <p className="text-sm font-semibold uppercase tracking-wide text-sky-300">
            Learning
          </p>
          <h1 className="text-3xl font-semibold tracking-normal text-slate-50 md:text-4xl">
            Choose a course
          </h1>
          <p className="text-base leading-7 text-slate-300">
            Learning experiences are grouped by course so you can open the
            relevant sequence first, then choose the lecture or topic you want
            to prepare for. Some course pages are standalone when the source
            material does not yet have a matching quiz source.
          </p>
        </section>

        <section
          aria-label="Available learning courses"
          className="grid gap-4 md:grid-cols-2"
        >
          {courses.map((course) => (
            <Link
              key={course.seriesId}
              href={getLearningCoursePath(course.seriesId)}
              className="group rounded-lg border border-slate-800 bg-slate-900 p-5 transition-colors hover:border-sky-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950"
            >
              <div className="flex flex-wrap items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                <span>{course.experiences.length} learning pages</span>
                <span aria-hidden="true">/</span>
                <span>{course.totalDurationMinutes} min</span>
              </div>
              <h2 className="mt-3 text-xl font-semibold text-slate-50 group-hover:text-sky-200">
                {course.label}
              </h2>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                Open the course to choose a focused learning experience before
                starting the matching multiple-choice practice.
              </p>
              <p className="mt-4 text-sm font-semibold text-sky-300">
                Open course
              </p>
            </Link>
          ))}

          {standaloneLearningCourses.map((course) => (
            <Link
              key={course.href}
              href={course.href}
              className="group rounded-lg border border-slate-800 bg-slate-900 p-5 transition-colors hover:border-sky-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950"
            >
              <div className="flex flex-wrap items-center gap-2 text-xs font-semibold text-slate-400">
                <span>{course.meta}</span>
              </div>
              <h2 className="mt-3 text-xl font-semibold text-slate-50 group-hover:text-sky-200">
                {course.label}
              </h2>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                {course.summary}
              </p>
              <p className="mt-4 text-sm font-semibold text-sky-300">
                Open course
              </p>
            </Link>
          ))}
        </section>
      </div>
    </main>
  );
}
