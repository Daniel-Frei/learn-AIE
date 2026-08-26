"use client";

import Link from "next/link";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { ArrowRight, Check, CircleHelp, Compass, Sigma } from "lucide-react";
import MathText from "../../MathText";
import type { LearningExperience } from "../../../lib/learning";
import type { SourceId } from "../../../lib/quiz";
import styles from "./ProbabilityCourse.module.css";

const STATIONS = [
  {
    sourceId: "crash-probability-l0",
    label: "L0",
    title: "Notation gym",
  },
  {
    sourceId: "crash-probability-l1",
    label: "L1",
    title: "Event universe",
  },
  {
    sourceId: "crash-probability-l2",
    label: "L2",
    title: "Evidence lens",
  },
  {
    sourceId: "crash-probability-l3",
    label: "L3",
    title: "Training microscope",
  },
  {
    sourceId: "crash-probability-l4",
    label: "L4",
    title: "Decision world",
  },
  {
    sourceId: "crash-probability-l5",
    label: "L5",
    title: "Generation forge",
  },
] as const satisfies readonly {
  sourceId: SourceId;
  label: string;
  title: string;
}[];

type ProbabilityCourseProps = {
  experience: LearningExperience;
  station: "l0" | "l1" | "l2" | "l3" | "l4" | "l5";
  kicker: string;
  headline: string;
  introduction: string;
  heroVisual: ReactNode;
  children: ReactNode;
};

export function ProbabilityCourse({
  experience,
  station,
  kicker,
  headline,
  introduction,
  heroVisual,
  children,
}: ProbabilityCourseProps) {
  const activeStationRef = useRef<HTMLAnchorElement | null>(null);
  const stationRailRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const activeStation = activeStationRef.current;
    const stationRail = stationRailRef.current;
    if (!activeStation || !stationRail) return;
    stationRail.scrollLeft =
      activeStation.offsetLeft -
      (stationRail.clientWidth - activeStation.clientWidth) / 2;
  }, [experience.sourceId]);

  return (
    <main className={styles.page} data-station={station}>
      <nav aria-label="Probability course" className={styles.courseNav}>
        <div className={styles.courseNavInner}>
          <Link href="/learn/crash-course-probability" className={styles.brand}>
            <Compass aria-hidden="true" size={18} />
            Probability Observatory
          </Link>
          <div className={styles.stationRail} ref={stationRailRef}>
            {STATIONS.map((item) => {
              const isActive = item.sourceId === experience.sourceId;
              return (
                <Link
                  key={item.sourceId}
                  ref={isActive ? activeStationRef : undefined}
                  href={`/learn/crash-course-probability/${item.sourceId}`}
                  aria-current={isActive ? "page" : undefined}
                  className={`${styles.stationLink} ${isActive ? styles.stationLinkActive : ""}`}
                >
                  <span>{item.label}</span>
                  <span>{item.title}</span>
                </Link>
              );
            })}
          </div>
        </div>
      </nav>

      <header className={styles.hero}>
        <div className={styles.heroGrid}>
          <div className={styles.heroCopy}>
            <p className={styles.kicker}>{kicker}</p>
            <h1>{headline}</h1>
            <p className={styles.heroIntroduction}>{introduction}</p>
            <div className={styles.heroMeta}>
              <span>{experience.durationMinutes} min guided path</span>
              <span>{experience.level}</span>
            </div>
            <ul className={styles.outcomes}>
              {experience.outcomes.map((outcome) => (
                <li key={outcome}>
                  <Check aria-hidden="true" size={16} />
                  <span>{outcome}</span>
                </li>
              ))}
            </ul>
          </div>
          <div className={styles.heroObject}>{heroVisual}</div>
        </div>
      </header>

      <div className={styles.content}>{children}</div>
    </main>
  );
}

type ProbabilitySectionProps = {
  id: string;
  eyebrow: string;
  title: string;
  lead: string;
  children: ReactNode;
};

export function ProbabilitySection({
  id,
  eyebrow,
  title,
  lead,
  children,
}: ProbabilitySectionProps) {
  return (
    <section id={id} className={styles.section}>
      <header className={styles.sectionHeader}>
        <p>{eyebrow}</p>
        <h2>{title}</h2>
        <div className={styles.sectionLead}>{lead}</div>
      </header>
      {children}
    </section>
  );
}

export function ProbabilityFormula({
  label,
  formula,
  children,
}: {
  label: string;
  formula: string;
  children?: ReactNode;
}) {
  return (
    <div className={styles.formulaPanel}>
      <div className={styles.formulaLabel}>
        <Sigma aria-hidden="true" size={17} />
        {label}
      </div>
      <MathText text={formula} className={styles.formula} />
      {children && <div className={styles.formulaExplanation}>{children}</div>}
    </div>
  );
}

export function InlineProbabilityMath({ text }: { text: string }) {
  return <MathText inline text={text} className={styles.inlineMath} />;
}

export function ProbabilityInsight({
  title,
  children,
  tone = "insight",
}: {
  title: string;
  children: ReactNode;
  tone?: "insight" | "warning" | "success";
}) {
  return (
    <aside className={styles.insight} data-tone={tone}>
      <CircleHelp aria-hidden="true" size={21} />
      <div>
        <h3>{title}</h3>
        <div>{children}</div>
      </div>
    </aside>
  );
}

type CheckOption = {
  label: string;
  explanation: string;
};

export function ProbabilityCheck({
  testId,
  title,
  question,
  options,
  correctIndex,
}: {
  testId: string;
  title: string;
  question: string;
  options: readonly CheckOption[];
  correctIndex: number;
}) {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const selected = selectedIndex === null ? null : options[selectedIndex];
  const correct = selectedIndex === correctIndex;

  return (
    <section className={styles.check} data-testid={testId}>
      <p className={styles.checkEyebrow}>Pause and predict</p>
      <h3>{title}</h3>
      <p>{question}</p>
      <div className={styles.checkOptions}>
        {options.map((option, index) => (
          <button
            key={option.label}
            type="button"
            aria-pressed={selectedIndex === index}
            onClick={() => setSelectedIndex(index)}
          >
            {option.label}
          </button>
        ))}
      </div>
      {selected && (
        <div
          role="status"
          className={correct ? styles.correctStatus : styles.tryAgainStatus}
        >
          <strong>{correct ? "Exactly." : "Try that boundary again."}</strong>{" "}
          {selected.explanation}
        </div>
      )}
    </section>
  );
}

export function ProbabilityQuizLaunch({
  experience,
  recap,
}: {
  experience: LearningExperience;
  recap: readonly string[];
}) {
  return (
    <section className={styles.quizLaunch}>
      <div>
        <p className={styles.kicker}>Consolidate</p>
        <h2>Carry the model into practice</h2>
        <p>
          The questions now test the same objects you manipulated here. Keep the
          calculation path visible: define the universe, choose the right rule,
          and only then compute.
        </p>
        <ul>
          {recap.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </div>
      <Link
        href={`/?source=${experience.sourceId}`}
        className={styles.quizButton}
      >
        Start{" "}
        {experience.sourceId.replace("crash-probability-", "").toUpperCase()}{" "}
        questions
        <ArrowRight aria-hidden="true" size={18} />
      </Link>
    </section>
  );
}

export function ProbabilityMetric({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className={styles.metric}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

export { styles as probabilityCourseStyles };
