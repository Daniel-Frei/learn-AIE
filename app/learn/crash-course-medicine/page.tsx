import Link from "next/link";

export default function CrashCourseMedicinePage() {
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
          <p className="text-sm font-semibold text-sky-300">Course</p>
          <h1 className="text-3xl font-semibold text-slate-50 md:text-4xl">
            Crash Course Medicine
          </h1>
          <p className="text-base leading-7 text-slate-300">
            Learn the operating logic of medicine: patient presentations,
            clinical vocabulary, differential diagnosis, risk, care pathways,
            evidence, and real-world constraints.
          </p>
          <p className="text-sm font-semibold text-emerald-300">
            1 standalone learning page
          </p>
        </section>

        <section
          aria-label="Crash Course Medicine learning experiences"
          className="grid gap-4 md:grid-cols-2"
        >
          <Link
            href="/learn/crash-course-medicine/lecture-1"
            className="group rounded-lg border border-slate-800 bg-slate-900 p-5 transition-colors hover:border-sky-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950"
          >
            <p className="text-xs font-semibold text-slate-400">Lecture 1</p>
            <h2 className="mt-3 text-xl font-semibold text-slate-50 group-hover:text-sky-200">
              What Medicine Is
            </h2>
            <p className="mt-2 text-sm leading-6 text-slate-300">
              Work through a clinical studio that translates patient stories
              into evidence, problem representations, ranked differentials,
              action thresholds, and initial plans.
            </p>
            <p className="mt-4 text-sm font-semibold text-sky-300">
              Open learning page
            </p>
          </Link>
        </section>
      </div>
    </main>
  );
}
