"use client";

import {
  Activity,
  AlertTriangle,
  ArrowDown,
  ArrowRight,
  ArrowUp,
  BookOpen,
  CheckCircle2,
  ClipboardList,
  Eye,
  FlaskConical,
  Gauge,
  HeartPulse,
  RefreshCcw,
  Route,
  Stethoscope,
} from "lucide-react";
import Link from "next/link";
import { useMemo, useState, type ReactNode } from "react";
import { lenses, patientCases } from "./clinicalData";
import {
  byPriority,
  makeProblemRepresentation,
  planFeedback,
  planScore,
  priorityScore,
  representationCompleteness,
  revealedRisk,
} from "./reasoning";
import type {
  Diagnosis,
  EvidenceItem,
  EvidenceKind,
  EvidenceStage,
  PatientCase,
} from "./types";

const stageLabels: Record<EvidenceStage, string> = {
  arrival: "Arrival",
  history: "History",
  exam: "Exam and vitals",
  tests: "Tests",
  context: "Patient context",
};

const kindLabels: Record<EvidenceKind, string> = {
  symptom: "Symptom",
  sign: "Sign",
  context: "Context",
  comorbidity: "Comorbidity",
  test: "Test",
  value: "Value",
};

const categoryHelp: Record<string, string> = {
  Cardiac: "heart",
  Vascular: "blood vessels",
  Pulmonary: "lungs",
  Gastrointestinal: "digestive tract",
  Musculoskeletal: "muscles, ribs, joints",
  Infectious: "infection",
  Cardiopulmonary: "heart or lungs",
};

const vocabulary = [
  {
    term: "Symptom",
    short: "What the patient experiences",
    example: "Chest pressure, nausea, cough, fatigue, shortness of breath",
    contrast: "Subjective. The clinician hears it in the story.",
  },
  {
    term: "Sign",
    short: "What can be observed or measured",
    example: "Temperature 38.7 C, oxygen saturation 89%, blood pressure 92/58",
    contrast: "More objective. Vitals are especially high-value signs.",
  },
  {
    term: "Syndrome",
    short: "A recognizable cluster before the exact cause is known",
    example: "Sepsis syndrome: infection plus organ dysfunction",
    contrast: "A pattern, not always a final disease label.",
  },
  {
    term: "Disease",
    short: "A pathological process affecting the body",
    example: "Pneumonia, diabetes, myocardial infarction, cancer",
    contrast: "A disease can look different in different patients.",
  },
  {
    term: "Diagnosis",
    short: "The clinician's working conclusion",
    example: "Community-acquired pneumonia; suspected acute coronary syndrome",
    contrast: "Often provisional and updated as evidence arrives.",
  },
  {
    term: "Prognosis",
    short: "The expected future course",
    example:
      "Can recover at home, needs hospital care, high risk of deterioration",
    contrast: "Answers what may happen next, with or without treatment.",
  },
  {
    term: "Comorbidity",
    short: "Another condition the patient already has",
    example: "Diabetes plus chronic kidney disease in a patient with pneumonia",
    contrast: "Changes risk, medication choices, and recovery.",
  },
  {
    term: "Differential diagnosis",
    short: "A ranked list of possible explanations",
    example:
      "Chest pain: heart attack, reflux, pulmonary embolism, muscle strain",
    contrast: "Ranked by likelihood, danger, urgency, and fit.",
  },
];

const encounterGroups = [
  {
    title: "Patient story",
    steps: [
      "Chief complaint",
      "History of present illness",
      "Past medical history",
      "Medications and allergies",
    ],
  },
  {
    title: "Clinical data",
    steps: [
      "Family and social history",
      "Review of systems",
      "Physical examination",
      "Vital signs",
    ],
  },
  {
    title: "Reasoning",
    steps: [
      "Problem representation",
      "Differential diagnosis",
      "Diagnostic plan",
    ],
  },
  {
    title: "Action",
    steps: ["Treatment plan", "Care setting", "Follow-up or monitoring"],
  },
];

function initialEvidenceIds(patientCase: PatientCase) {
  return patientCase.evidence
    .filter((item) => item.stage === "arrival")
    .map((item) => item.id);
}

export default function CrashMedicineLecture1LearningPage() {
  const [caseId, setCaseId] = useState(patientCases[1].id);
  const activeCase =
    patientCases.find((item) => item.id === caseId) ?? patientCases[0];
  const [revealedIds, setRevealedIds] = useState(() =>
    initialEvidenceIds(activeCase),
  );
  const [selectedRepIds, setSelectedRepIds] = useState(() =>
    initialEvidenceIds(activeCase),
  );
  const [diagnosisOrder, setDiagnosisOrder] = useState(() =>
    byPriority(activeCase.diagnoses).map((item) => item.id),
  );
  const [selectedPlanIds, setSelectedPlanIds] = useState<string[]>([]);
  const [lensId, setLensId] = useState("medicine");
  const [focusDiagnosisId, setFocusDiagnosisId] = useState(
    byPriority(activeCase.diagnoses)[0].id,
  );
  const [activeTerm, setActiveTerm] = useState(vocabulary[0].term);

  function resetForCase(nextCase: PatientCase) {
    const arrivalIds = initialEvidenceIds(nextCase);
    setRevealedIds(arrivalIds);
    setSelectedRepIds(arrivalIds);
    setDiagnosisOrder(byPriority(nextCase.diagnoses).map((item) => item.id));
    setSelectedPlanIds([]);
    setFocusDiagnosisId(byPriority(nextCase.diagnoses)[0].id);
  }

  function switchCase(nextCaseId: string) {
    const nextCase = patientCases.find((item) => item.id === nextCaseId);
    if (!nextCase) return;
    setCaseId(nextCaseId);
    resetForCase(nextCase);
  }

  const revealedEvidence = activeCase.evidence.filter((item) =>
    revealedIds.includes(item.id),
  );
  const risk = revealedRisk(activeCase.evidence, revealedIds);
  const completeness = representationCompleteness(activeCase, selectedRepIds);
  const representation = makeProblemRepresentation(activeCase, selectedRepIds);
  const selectedLens = lenses.find((lens) => lens.id === lensId) ?? lenses[0];
  const selectedVocabulary =
    vocabulary.find((item) => item.term === activeTerm) ?? vocabulary[0];
  const orderedDiagnoses = diagnosisOrder
    .map((id) => activeCase.diagnoses.find((diagnosis) => diagnosis.id === id))
    .filter((diagnosis): diagnosis is Diagnosis => Boolean(diagnosis));
  const focusDiagnosis =
    activeCase.diagnoses.find(
      (diagnosis) => diagnosis.id === focusDiagnosisId,
    ) ?? orderedDiagnoses[0];
  const decisionScore = planScore(activeCase, selectedPlanIds);
  const unsafeSelected = activeCase.unsafePlanIds.some((id) =>
    selectedPlanIds.includes(id),
  );
  const idealCovered = activeCase.idealPlanIds.filter((id) =>
    selectedPlanIds.includes(id),
  ).length;

  function revealStage(stage: EvidenceStage) {
    const stageIds = activeCase.evidence
      .filter((item) => item.stage === stage)
      .map((item) => item.id);
    setRevealedIds((current) => [...new Set([...current, ...stageIds])]);
  }

  function revealAll() {
    const allIds = activeCase.evidence.map((item) => item.id);
    setRevealedIds(allIds);
  }

  function toggleRepresentationEvidence(id: string) {
    if (!revealedIds.includes(id)) return;
    setSelectedRepIds((current) =>
      current.includes(id)
        ? current.filter((item) => item !== id)
        : [...current, id],
    );
  }

  function showModelRepresentation() {
    const modelIds = activeCase.evidence
      .filter((item) => item.representationPhrase && item.riskWeight >= 6)
      .map((item) => item.id);
    setRevealedIds((current) => [...new Set([...current, ...modelIds])]);
    setSelectedRepIds(modelIds);
  }

  function moveDiagnosis(id: string, direction: -1 | 1) {
    setDiagnosisOrder((current) => {
      const index = current.indexOf(id);
      const nextIndex = index + direction;
      if (index < 0 || nextIndex < 0 || nextIndex >= current.length)
        return current;
      const next = [...current];
      [next[index], next[nextIndex]] = [next[nextIndex], next[index]];
      return next;
    });
  }

  function togglePlan(id: string) {
    setSelectedPlanIds((current) =>
      current.includes(id)
        ? current.filter((item) => item !== id)
        : [...current, id],
    );
  }

  return (
    <div
      className="medicine-clinical-studio"
      data-testid="medicine-clinical-studio"
    >
      <main className="lesson-shell">
        <LessonHeader />

        <section
          className="hero-section"
          id="what-medicine"
          aria-labelledby="what-medicine-heading"
        >
          <div className="hero-copy">
            <p className="eyebrow">Crash Course in Medicine - Lecture 1</p>
            <h1 id="what-medicine-heading">What Medicine Is</h1>
            <p className="lede">
              Medicine is applied biological reasoning under uncertainty, in
              real patients, with real-world constraints. The central move is
              from messy human presentation to a working diagnosis and
              management plan.
            </p>
            <div className="hero-actions">
              <a href="#vocabulary">Start with vocabulary</a>
              <a href="#case-lab">Jump to the case lab</a>
            </div>
          </div>
          <MedicineFlowVisual />
        </section>

        <section
          className="lesson-band"
          id="disciplines"
          aria-labelledby="disciplines-heading"
        >
          <SectionIntro
            eyebrow="Same human problem, different discipline"
            title="What Question Are You Asking?"
            body="Lecture 1 starts here because medicine overlaps with biology, public health, clinical research, and healthcare delivery. The same patient can be understood through each lens, but medicine is the lens that must decide what happens for this person now."
            headingId="disciplines-heading"
          />
          <LensLab
            lensId={lensId}
            selectedLens={selectedLens}
            setLensId={setLensId}
          />
        </section>

        <section
          className="lesson-band warm"
          id="vocabulary"
          aria-labelledby="vocabulary-heading"
        >
          <SectionIntro
            eyebrow="Core medical vocabulary"
            title="Learn the Language Before the Case Gets Busy"
            body="The lab later asks you to use terms such as symptom, sign, diagnosis, and comorbidity. This section gives those words a working meaning first."
            headingId="vocabulary-heading"
          />
          <VocabularyAtlas
            activeTerm={activeTerm}
            selectedVocabulary={selectedVocabulary}
            setActiveTerm={setActiveTerm}
          />
        </section>

        <section
          className="lesson-band"
          id="encounter"
          aria-labelledby="encounter-heading"
        >
          <SectionIntro
            eyebrow="The patient encounter"
            title="How a Story Becomes a Clinical Problem"
            body="The encounter is not a memorization list. It is a funnel: patient language becomes clinical data, then a compact representation, then a ranked set of possibilities and an action plan."
            headingId="encounter-heading"
          />
          <EncounterMap />
        </section>

        <section
          className="lesson-band warm"
          id="reasoning"
          aria-labelledby="reasoning-heading"
        >
          <SectionIntro
            eyebrow="Clinical reasoning"
            title="Before the Lab: What the Board Will Ask You To Do"
            body="A differential diagnosis is not a random list. It is a ranked set of possible explanations. In medicine, unlikely but dangerous possibilities may outrank common benign ones until they are handled."
            headingId="reasoning-heading"
          />
          <ReasoningPrimer />
        </section>

        <section
          className="lab-section"
          id="case-lab"
          aria-labelledby="lab-heading"
        >
          <div className="lab-heading">
            <div>
              <p className="eyebrow">Application lab</p>
              <h2 id="lab-heading">
                Work Through a Patient Case in Clinical Order
              </h2>
              <p className="lab-heading-copy">
                Story, timeline, evidence, representation, differential, then
                plan.
              </p>
            </div>
            <div
              className="case-tabs"
              aria-label="Choose patient case"
              role="group"
            >
              {patientCases.map((patientCase) => (
                <button
                  className="case-tab"
                  data-testid={`case-${patientCase.id}`}
                  key={patientCase.id}
                  onClick={() => switchCase(patientCase.id)}
                  type="button"
                  aria-pressed={patientCase.id === activeCase.id}
                >
                  <span>{patientCase.chiefComplaint}</span>
                  <strong>{patientCase.title}</strong>
                </button>
              ))}
            </div>
          </div>

          <section
            className="lab-workflow"
            aria-label="Clinical reasoning simulator"
          >
            <PatientColumn
              activeCase={activeCase}
              revealedEvidence={revealedEvidence}
              revealedIds={revealedIds}
              risk={risk}
              revealAll={revealAll}
              revealStage={revealStage}
            />

            <ReasoningColumn
              activeCase={activeCase}
              completeness={completeness}
              focusDiagnosis={focusDiagnosis}
              moveDiagnosis={moveDiagnosis}
              orderedDiagnoses={orderedDiagnoses}
              representation={representation}
              revealedIds={revealedIds}
              selectedRepIds={selectedRepIds}
              setFocusDiagnosisId={setFocusDiagnosisId}
              showModelRepresentation={showModelRepresentation}
              toggleRepresentationEvidence={toggleRepresentationEvidence}
            />

            <DecisionColumn
              activeCase={activeCase}
              decisionScore={decisionScore}
              idealCovered={idealCovered}
              selectedPlanIds={selectedPlanIds}
              togglePlan={togglePlan}
              unsafeSelected={unsafeSelected}
            />
          </section>
        </section>

        <section
          className="lesson-band"
          id="uncertainty"
          aria-labelledby="uncertainty-heading"
        >
          <SectionIntro
            eyebrow="Uncertainty, risk, and decisions"
            title="The Goal Is Not Always Certainty"
            body="The lecture ends by making the practical point explicit: doctors often act before certainty because delay, test harm, treatment harm, patient values, and care setting all matter."
            headingId="uncertainty-heading"
          />
          <UncertaintyBoard />
        </section>

        <section
          className="recap-section"
          id="recap"
          aria-labelledby="recap-heading"
        >
          <div>
            <p className="eyebrow">Lecture 1 takeaway</p>
            <h2 id="recap-heading">A First Operating Model of Medicine</h2>
          </div>
          <ol>
            <li>
              Medicine starts with a patient problem, not a textbook disease.
            </li>
            <li>
              Symptoms, signs, syndromes, diagnoses, prognosis, and
              comorbidities are different kinds of information.
            </li>
            <li>
              A differential diagnosis ranks what is likely, dangerous, urgent,
              treatable, and fitting.
            </li>
            <li>
              Tests update probabilities; they do not simply reveal truth.
            </li>
            <li>
              Treatment decisions balance benefit, harm, urgency, patient
              values, and system constraints.
            </li>
          </ol>
        </section>
      </main>
    </div>
  );
}

function LessonHeader() {
  return (
    <header className="lesson-header">
      <Link
        className="brand-lockup"
        href="/learn/crash-course-medicine"
        aria-label="Back to Medicine course"
      >
        <span className="brand-mark" aria-hidden="true">
          <Stethoscope size={19} />
        </span>
        <span>
          <strong>Crash Course in Medicine</strong>
          <small>Lecture 1 of 5</small>
        </span>
      </Link>
      <nav className="lesson-nav" aria-label="Lesson sections">
        <a href="#what-medicine">Start</a>
        <a href="#disciplines">Medicine</a>
        <a href="#vocabulary">Vocabulary</a>
        <a href="#encounter">Encounter</a>
        <a href="#reasoning">Reasoning</a>
        <a href="#case-lab">Lab</a>
        <a href="#uncertainty">Uncertainty</a>
      </nav>
    </header>
  );
}

function SectionIntro({
  body,
  eyebrow,
  headingId,
  title,
}: {
  body: string;
  eyebrow: string;
  headingId: string;
  title: string;
}) {
  return (
    <div className="section-intro">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h2 id={headingId}>{title}</h2>
      </div>
      <p>{body}</p>
    </div>
  );
}

function MedicineFlowVisual() {
  const nodes = [
    ["Messy story", "Symptoms, fears, memory gaps"],
    ["Clinical facts", "Signs, history, context"],
    ["Problem representation", "Compact summary"],
    ["Differential diagnosis", "Likely and dangerous causes"],
    ["Plan", "Tests, treatment, care setting"],
  ];

  return (
    <div
      className="flow-visual"
      aria-label="Medicine flow from patient story to plan"
    >
      <div className="clinical-mural" aria-hidden="true">
        <div className="mural-patient">
          <HeartPulse size={44} />
        </div>
        <div className="mural-note mural-note-story">
          <span>Patient story</span>
          <strong>{`"I can't catch my breath."`}</strong>
        </div>
        <div className="mural-note mural-note-sign">
          <span>Measured sign</span>
          <strong>O2 sat 89%</strong>
        </div>
        <div className="mural-note mural-note-context">
          <span>Context</span>
          <strong>Diabetes + kidney disease</strong>
        </div>
        <div className="mural-note mural-note-decision">
          <span>Clinical action</span>
          <strong>Stabilize while testing</strong>
        </div>
        <svg className="mural-trace" viewBox="0 0 420 84" role="img">
          <path className="trace-baseline" d="M0 54H420" />
          <path
            className="trace-line"
            d="M0 54 H55 L72 54 L84 28 L98 70 L112 54 H158 C178 54 184 44 194 44 C205 44 210 54 222 54 H258 L273 54 L284 22 L300 74 L318 54 H420"
          />
        </svg>
      </div>
      <div className="flow-steps">
        {nodes.map(([title, body], index) => (
          <div className="flow-step" key={title}>
            <span>{index + 1}</span>
            <strong>{title}</strong>
            <p>{body}</p>
            {index < nodes.length - 1 && (
              <ArrowRight className="flow-arrow" size={18} aria-hidden="true" />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function LensLab({
  lensId,
  selectedLens,
  setLensId,
}: {
  lensId: string;
  selectedLens: (typeof lenses)[number];
  setLensId: (id: string) => void;
}) {
  return (
    <div className="lens-lab">
      <div
        className="lens-selector"
        role="group"
        aria-label="Choose discipline lens"
      >
        {lenses.map((lens) => (
          <button
            key={lens.id}
            type="button"
            aria-pressed={lensId === lens.id}
            onClick={() => setLensId(lens.id)}
          >
            {lens.label}
          </button>
        ))}
      </div>
      <article className="concept-panel">
        <span className="panel-icon" aria-hidden="true">
          <BookOpen size={18} />
        </span>
        <h3>{selectedLens.question}</h3>
        <p>{selectedLens.example}</p>
        <strong>{selectedLens.output}</strong>
      </article>
    </div>
  );
}

function VocabularyAtlas({
  activeTerm,
  selectedVocabulary,
  setActiveTerm,
}: {
  activeTerm: string;
  selectedVocabulary: (typeof vocabulary)[number];
  setActiveTerm: (term: string) => void;
}) {
  return (
    <div className="vocabulary-layout">
      <div
        className="term-grid"
        role="group"
        aria-label="Choose vocabulary term"
      >
        {vocabulary.map((item) => (
          <button
            key={item.term}
            type="button"
            aria-pressed={activeTerm === item.term}
            onClick={() => setActiveTerm(item.term)}
          >
            <strong>{item.term}</strong>
            <small>{item.short}</small>
          </button>
        ))}
      </div>
      <article className="concept-panel vocabulary-readout">
        <span className="panel-icon" aria-hidden="true">
          <ClipboardList size={18} />
        </span>
        <h3>{selectedVocabulary.term}</h3>
        <p>{selectedVocabulary.short}</p>
        <dl>
          <div>
            <dt>Example</dt>
            <dd>{selectedVocabulary.example}</dd>
          </div>
          <div>
            <dt>Why the distinction matters</dt>
            <dd>{selectedVocabulary.contrast}</dd>
          </div>
        </dl>
      </article>
      <div
        className="evidence-mini-board"
        aria-label="Example evidence categories"
      >
        <EvidenceMini label="I feel short of breath" kind="symptom" />
        <EvidenceMini label="Oxygen saturation 89%" kind="sign" />
        <EvidenceMini label="Diabetes" kind="comorbidity" />
        <EvidenceMini label="Pneumonia with sepsis" kind="diagnosis" />
      </div>
    </div>
  );
}

function EvidenceMini({ kind, label }: { kind: string; label: string }) {
  return (
    <div className={`evidence-mini ${kind}`}>
      <span>{kind}</span>
      <strong>{label}</strong>
    </div>
  );
}

function LabStep({
  children,
  className = "",
  eyebrow,
  number,
  title,
}: {
  children: ReactNode;
  className?: string;
  eyebrow: string;
  number: number;
  title: string;
}) {
  const headingId = `lab-step-${number}`;

  return (
    <section
      className={`workflow-step ${className}`}
      aria-labelledby={headingId}
    >
      <div className="workflow-step-header">
        <span className="workflow-step-number">{number}</span>
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h3 id={headingId}>{title}</h3>
        </div>
      </div>
      <div className="workflow-step-body">{children}</div>
    </section>
  );
}

function EncounterMap() {
  return (
    <div className="encounter-map">
      {encounterGroups.map((group, groupIndex) => (
        <article key={group.title}>
          <div className="encounter-group-heading">
            <span>{groupIndex + 1}</span>
            <h3>{group.title}</h3>
          </div>
          <ul>
            {group.steps.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ul>
        </article>
      ))}
    </div>
  );
}

function ReasoningPrimer() {
  return (
    <div className="reasoning-primer">
      <article className="concept-panel">
        <span className="panel-icon" aria-hidden="true">
          <Route size={18} />
        </span>
        <h3>Problem representation</h3>
        <p>
          A short clinical summary keeps the patient, timing, key symptom
          pattern, relevant risks, and danger signals together.
        </p>
        <div className="thin-good">
          <span>Too thin: Patient has chest pain.</span>
          <strong>
            Better: A 62-year-old man with diabetes has acute exertional central
            chest pressure radiating to the left arm.
          </strong>
        </div>
      </article>
      <article className="concept-panel">
        <span className="panel-icon" aria-hidden="true">
          <AlertTriangle size={18} />
        </span>
        <h3>Most likely is not always enough</h3>
        <p>
          A common benign explanation may be likely, but an uncommon dangerous
          explanation may need to be ruled out first.
        </p>
        <div className="risk-matrix" aria-label="Likelihood and danger matrix">
          <span>common + low danger</span>
          <span>common + high danger</span>
          <span>rare + low danger</span>
          <strong>rare + high danger: must not miss</strong>
        </div>
      </article>
      <article className="concept-panel">
        <span className="panel-icon" aria-hidden="true">
          <Stethoscope size={18} />
        </span>
        <h3>Body-category words in the lab</h3>
        <p>
          Labels such as cardiac, pulmonary, vascular, gastrointestinal, and
          musculoskeletal are simple location clues: heart, lungs, blood
          vessels, digestive tract, and muscles or bones.
        </p>
        <div className="category-glossary" aria-label="Body category glossary">
          <span>Cardiac = heart</span>
          <span>Pulmonary = lungs</span>
          <span>Vascular = blood vessels</span>
          <span>Gastrointestinal = digestive tract</span>
          <span>Musculoskeletal = muscles, bones, or joints</span>
        </div>
      </article>
    </div>
  );
}

function PatientColumn({
  activeCase,
  revealedEvidence,
  revealedIds,
  risk,
  revealAll,
  revealStage,
}: {
  activeCase: PatientCase;
  revealedEvidence: EvidenceItem[];
  revealedIds: string[];
  risk: number;
  revealAll: () => void;
  revealStage: (stage: EvidenceStage) => void;
}) {
  return (
    <>
      <LabStep
        className="arrival-step"
        eyebrow="Start here"
        number={1}
        title="Read the Arrival Story Before Opening the Case Up"
      >
        <div className="arrival-grid">
          <div className="case-card">
            <div className="case-kicker">
              <span>{activeCase.setting}</span>
              <span>{activeCase.chiefComplaint}</span>
            </div>
            <h3 id="patient-heading">{activeCase.opening}</h3>
            <blockquote>{activeCase.patientVoice}</blockquote>
            <p>{activeCase.coreQuestion}</p>
          </div>

          <PatientMonitor risk={risk} evidenceCount={revealedIds.length} />
        </div>
      </LabStep>

      <LabStep
        className="evidence-step"
        eyebrow="Gather evidence"
        number={2}
        title="Use the Encounter Timeline Before You Summarize"
      >
        <div className="timeline-grid">
          <div className="panel">
            <div className="panel-title-row">
              <div>
                <p className="eyebrow">Encounter timeline</p>
                <h3>Reveal the Case in Clinical Order</h3>
              </div>
              <button
                className="icon-button"
                type="button"
                onClick={revealAll}
                aria-label="Reveal all evidence"
              >
                <Eye size={18} />
              </button>
            </div>
            <div className="stage-grid">
              {(Object.keys(stageLabels) as EvidenceStage[]).map((stage) => {
                const count = activeCase.evidence.filter(
                  (item) => item.stage === stage,
                ).length;
                const revealed = activeCase.evidence.filter(
                  (item) =>
                    item.stage === stage && revealedIds.includes(item.id),
                ).length;
                return (
                  <button
                    className="stage-button"
                    key={stage}
                    type="button"
                    onClick={() => revealStage(stage)}
                  >
                    <span>{stageLabels[stage]}</span>
                    <strong>
                      {revealed}/{count}
                    </strong>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="panel evidence-feed">
            <p className="eyebrow">Observed evidence</p>
            {revealedEvidence.map((item) => (
              <article className={`evidence-item ${item.kind}`} key={item.id}>
                <span>{kindLabels[item.kind]}</span>
                <div>
                  <strong>{item.label}</strong>
                  <p>{item.detail}</p>
                </div>
              </article>
            ))}
          </div>
        </div>
      </LabStep>
    </>
  );
}

function PatientMonitor({
  risk,
  evidenceCount,
}: {
  risk: number;
  evidenceCount: number;
}) {
  const points = useMemo(() => {
    const base = [60, 45, 53, 34, 52, 42, 58, 36, 48, 40, 54, 44];
    const lift = Math.round(risk / 7);
    return base
      .map((y, index) => `${index * 24},${Math.max(14, y - lift)}`)
      .join(" ");
  }, [risk]);
  const riskLabel =
    risk >= 78 ? "unstable" : risk >= 48 ? "concerning" : "incomplete";

  return (
    <div className={`monitor ${riskLabel}`}>
      <div className="monitor-topline">
        <div>
          <p className="eyebrow">Patient signal</p>
          <strong>{riskLabel}</strong>
        </div>
        <span>{risk}% risk signal</span>
      </div>
      <svg
        viewBox="0 0 264 72"
        role="img"
        aria-label={`Risk signal trace at ${risk} percent`}
      >
        <polyline
          points={points}
          fill="none"
          stroke="currentColor"
          strokeWidth="4"
          strokeLinejoin="round"
        />
      </svg>
      <div className="monitor-readouts">
        <span>
          <Activity size={15} /> {evidenceCount} facts
        </span>
        <span>
          <Gauge size={15} /> threshold shifts
        </span>
      </div>
    </div>
  );
}

function ReasoningColumn({
  activeCase,
  completeness,
  focusDiagnosis,
  moveDiagnosis,
  orderedDiagnoses,
  representation,
  revealedIds,
  selectedRepIds,
  setFocusDiagnosisId,
  showModelRepresentation,
  toggleRepresentationEvidence,
}: {
  activeCase: PatientCase;
  completeness: { score: number; missing: string[] };
  focusDiagnosis: Diagnosis;
  moveDiagnosis: (id: string, direction: -1 | 1) => void;
  orderedDiagnoses: Diagnosis[];
  representation: string;
  revealedIds: string[];
  selectedRepIds: string[];
  setFocusDiagnosisId: (id: string) => void;
  showModelRepresentation: () => void;
  toggleRepresentationEvidence: (id: string) => void;
}) {
  const completenessPercent = Math.round(completeness.score * 100);
  const representationEvidence = activeCase.evidence.filter(
    (item) => item.representationPhrase && revealedIds.includes(item.id),
  );

  return (
    <>
      <LabStep
        className="representation-step"
        eyebrow="Problem representation"
        number={3}
        title="Compress the Evidence Without Losing Risk"
      >
        <div className="panel representation-panel">
          <div className="panel-title-row">
            <div>
              <p className="eyebrow">Problem representation</p>
              <h3 id="lab-reasoning-heading">
                Select the Facts That Belong in the Summary
              </h3>
            </div>
            <button
              className="text-button"
              type="button"
              onClick={showModelRepresentation}
            >
              <ClipboardList size={17} />
              Model
            </button>
          </div>

          <div className="representation-output">
            <span>Too thin: {activeCase.thinRepresentation}</span>
            <p>{representation}</p>
            <div
              className="progress-track"
              role="progressbar"
              aria-label="Problem representation completeness"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={completenessPercent}
            >
              <i style={{ width: `${completenessPercent}%` }} />
            </div>
            <strong>
              {completeness.missing.length === 0
                ? "Strong clinical compression"
                : `Still missing: ${completeness.missing.join(", ")}`}
            </strong>
          </div>

          <div
            className="chip-grid"
            role="group"
            aria-label="Select facts for problem representation"
          >
            {representationEvidence.map((item) => (
              <button
                className="fact-chip"
                key={item.id}
                type="button"
                aria-pressed={selectedRepIds.includes(item.id)}
                onClick={() => toggleRepresentationEvidence(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </LabStep>

      <LabStep
        className="differential-step"
        eyebrow="Differential diagnosis"
        number={4}
        title="Rank Possible Causes After the Summary Is Clear"
      >
        <div className="panel differential-panel">
          <div className="panel-title-row">
            <div>
              <p className="eyebrow">Differential diagnosis</p>
              <h3>Rank Likelihood, Danger, and Urgency Together</h3>
            </div>
            <span className="score-pill">priority score</span>
          </div>
          <ol className="diagnosis-list">
            {orderedDiagnoses.map((diagnosis, index) => (
              <li
                className={diagnosis.id === focusDiagnosis.id ? "active" : ""}
                key={diagnosis.id}
              >
                <button
                  className="diagnosis-main"
                  type="button"
                  onClick={() => setFocusDiagnosisId(diagnosis.id)}
                >
                  <span>{index + 1}</span>
                  <div>
                    <strong>{diagnosis.name}</strong>
                    <small>
                      {diagnosis.category}:{" "}
                      {categoryHelp[diagnosis.category] ?? "clinical category"}
                    </small>
                  </div>
                  <b>{priorityScore(diagnosis)}</b>
                </button>
                <div className="diagnosis-controls">
                  <button
                    type="button"
                    aria-label={`Move ${diagnosis.name} up`}
                    onClick={() => moveDiagnosis(diagnosis.id, -1)}
                  >
                    <ArrowUp size={15} />
                  </button>
                  <button
                    type="button"
                    aria-label={`Move ${diagnosis.name} down`}
                    onClick={() => moveDiagnosis(diagnosis.id, 1)}
                  >
                    <ArrowDown size={15} />
                  </button>
                </div>
              </li>
            ))}
          </ol>
          <div className="diagnosis-detail">
            <strong>{focusDiagnosis.name}</strong>
            <p>{focusDiagnosis.why}</p>
            <div className="metric-row">
              <Metric label="Likelihood" value={focusDiagnosis.likelihood} />
              <Metric label="Danger" value={focusDiagnosis.danger} />
              <Metric label="Urgency" value={focusDiagnosis.urgency} />
            </div>
            <div className="discriminators">
              {focusDiagnosis.discriminators.map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>
          </div>
        </div>
      </LabStep>
    </>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <div aria-hidden="true">
        <i style={{ width: `${value}%` }} />
      </div>
      <strong>{value}</strong>
    </div>
  );
}

function DecisionColumn({
  activeCase,
  decisionScore,
  idealCovered,
  selectedPlanIds,
  togglePlan,
  unsafeSelected,
}: {
  activeCase: PatientCase;
  decisionScore: number;
  idealCovered: number;
  selectedPlanIds: string[];
  togglePlan: (id: string) => void;
  unsafeSelected: boolean;
}) {
  return (
    <LabStep
      className="decision-step"
      eyebrow="Decision before certainty"
      number={5}
      title="Build the Initial Plan Last, Then Debrief the Model"
    >
      <div className="decision-grid" aria-labelledby="decision-heading">
        <div className="panel plan-panel">
          <div className="panel-title-row">
            <div>
              <p className="eyebrow">Decision before certainty</p>
              <h3 id="decision-heading">Build an Initial Plan</h3>
            </div>
            <span className={unsafeSelected ? "alert-pill" : "score-pill"}>
              {decisionScore}/100
            </span>
          </div>
          <p className="lab-note">
            ECG means a heart electrical tracing. Troponin is a blood marker of
            heart muscle injury. These are examples of tests that update
            probability.
          </p>

          <div className="plan-options">
            {activeCase.planOptions.map((option) => {
              const selected = selectedPlanIds.includes(option.id);
              return (
                <article
                  className={selected ? "plan-option selected" : "plan-option"}
                  key={option.id}
                >
                  <button
                    type="button"
                    aria-pressed={selected}
                    onClick={() => togglePlan(option.id)}
                  >
                    {option.type === "test" && <FlaskConical size={17} />}
                    {option.type === "treat" && <Activity size={17} />}
                    {option.type === "setting" && <Gauge size={17} />}
                    <span>{option.label}</span>
                  </button>
                  {selected && <p>{planFeedback(option, selected)}</p>}
                </article>
              );
            })}
          </div>

          <div
            className={
              unsafeSelected ? "decision-feedback warning" : "decision-feedback"
            }
          >
            {unsafeSelected ? (
              <AlertTriangle size={18} />
            ) : (
              <CheckCircle2 size={18} />
            )}
            <p>
              {unsafeSelected
                ? "Premature closure is active: the plan accepts a low-risk story before the danger signals are handled."
                : `${idealCovered}/${activeCase.idealPlanIds.length} high-value actions covered. The plan improves as it handles probability, urgency, and harm together.`}
            </p>
          </div>
        </div>

        <div className="panel debrief-panel">
          <div className="panel-title-row">
            <div>
              <p className="eyebrow">Lab debrief</p>
              <h3>The Lecture 1 Mental Model</h3>
            </div>
            <button
              className="icon-button"
              type="button"
              onClick={() => window.location.reload()}
              aria-label="Reset simulator"
            >
              <RefreshCcw size={17} />
            </button>
          </div>
          <ul>
            {activeCase.debrief.map((item) => (
              <li key={item}>{item}</li>
            ))}
            <li>Symptoms are subjective; signs are observed or measured.</li>
            <li>
              A diagnosis is a working conclusion, not always a single final
              moment.
            </li>
            <li>
              Patient context can change the threshold for testing, treatment,
              and care setting.
            </li>
          </ul>
        </div>
      </div>
    </LabStep>
  );
}

function UncertaintyBoard() {
  return (
    <div className="uncertainty-board">
      <article className="concept-panel">
        <span className="panel-icon" aria-hidden="true">
          <FlaskConical size={18} />
        </span>
        <h3>Tests update probability</h3>
        <p>
          A positive troponin result means something different in a high-risk
          chest pain story than in a low-risk person. The result is interpreted
          against the patient.
        </p>
      </article>
      <article
        className="threshold-visual"
        aria-label="Action threshold visual"
      >
        <div>
          <span>No action</span>
          <span>Test</span>
          <span>Treat or admit</span>
        </div>
        <i style={{ left: "18%" }}>low risk</i>
        <i style={{ left: "52%" }}>uncertain</i>
        <i style={{ left: "82%" }}>dangerous</i>
      </article>
      <article className="concept-panel">
        <span className="panel-icon" aria-hidden="true">
          <HeartPulse size={18} />
        </span>
        <h3>Benefit and harm move the threshold</h3>
        <p>
          Doctors act earlier when missing the disease would be catastrophic,
          the patient is unstable, or treatment is time-sensitive. They hold
          back when tests or treatments are likely to harm more than help.
        </p>
      </article>
    </div>
  );
}
