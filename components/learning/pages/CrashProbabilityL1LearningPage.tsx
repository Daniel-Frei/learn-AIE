"use client";

import { useState } from "react";
import { CircleDot, Dices, Layers3 } from "lucide-react";
import type { LearningExperience } from "../../../lib/learning";
import {
  combinations,
  expectedValue,
  permutations,
  unionProbability,
  variance,
} from "../../../lib/probabilityLearning";
import {
  InlineProbabilityMath,
  ProbabilityCheck,
  ProbabilityCourse,
  ProbabilityFormula,
  ProbabilityInsight,
  ProbabilityMetric,
  ProbabilityQuizLaunch,
  ProbabilitySection,
  probabilityCourseStyles as styles,
} from "../probability/ProbabilityCourse";

type Props = {
  experience: LearningExperience;
};

const MARBLES = [
  { id: "R1", color: "red" },
  { id: "R2", color: "red" },
  { id: "R3", color: "red" },
  { id: "R4", color: "red" },
  { id: "R5", color: "red" },
  { id: "B1", color: "blue" },
  { id: "B2", color: "blue" },
  { id: "B3", color: "blue" },
  { id: "G1", color: "green" },
  { id: "G2", color: "green" },
] as const;

const EVENTS = [
  {
    id: "red",
    label: "Red",
    notation: "R",
    description: "All five red outcomes",
    select: (marble: (typeof MARBLES)[number]) => marble.color === "red",
  },
  {
    id: "not-red",
    label: "Not red",
    notation: "R^c",
    description: "The complement: every outcome outside R",
    select: (marble: (typeof MARBLES)[number]) => marble.color !== "red",
  },
  {
    id: "blue-or-green",
    label: "Blue or green",
    notation: "B\\cup G",
    description: "The union of two disjoint color events",
    select: (marble: (typeof MARBLES)[number]) => marble.color !== "red",
  },
  {
    id: "red-and-number-one",
    label: "Red and #1",
    notation: "R\\cap N_1",
    description: "The overlap: only R1 satisfies both conditions",
    select: (marble: (typeof MARBLES)[number]) =>
      marble.color === "red" && marble.id.endsWith("1"),
  },
  {
    id: "number-one",
    label: "Numbered 1",
    notation: "N_1",
    description: "R1, B1, and G1: one event can cut across colors",
    select: (marble: (typeof MARBLES)[number]) => marble.id.endsWith("1"),
  },
] as const;

const COUNTING_SCENARIOS = [
  {
    id: "pin",
    label: "4-digit PIN",
    n: 10,
    k: 4,
    order: true,
    repeat: true,
    count: 10 ** 4,
    formula: String.raw`10^4`,
    reasoning:
      "Four ordered positions, with all 10 digits available each time.",
  },
  {
    id: "officers",
    label: "3 officer roles",
    n: 10,
    k: 3,
    order: true,
    repeat: false,
    count: permutations(10, 3),
    formula: String.raw`P(10,3)=10\cdot9\cdot8`,
    reasoning:
      "The roles differ, so Alice–Bob–Chen is not the same assignment as Chen–Alice–Bob.",
  },
  {
    id: "committee",
    label: "3-person committee",
    n: 10,
    k: 3,
    order: false,
    repeat: false,
    count: combinations(10, 3),
    formula: String.raw`{10\choose3}=\frac{10!}{3!7!}`,
    reasoning:
      "The same three people form one group in any ordering, so divide away the 3! redundant arrangements.",
  },
] as const;

function MarbleHero() {
  return (
    <div className={styles.lab} aria-label="Ten-outcome marble universe">
      <div className={styles.labHeader}>
        <div>
          <p>Sample space Ω</p>
          <h3>Ten possible outcomes</h3>
        </div>
        <CircleDot aria-hidden="true" size={34} />
      </div>
      <div className={styles.objectGrid}>
        {MARBLES.map((marble) => (
          <div
            className={styles.object}
            data-color={marble.color}
            key={marble.id}
          >
            {marble.id}
          </div>
        ))}
      </div>
      <div className={styles.status}>
        “Draw red” is not one outcome. It is the event {"{"}R1, R2, R3, R4, R5
        {"}"}: a set containing five outcomes.
      </div>
    </div>
  );
}

function EventUniverse() {
  const [eventId, setEventId] = useState<(typeof EVENTS)[number]["id"]>("red");
  const active = EVENTS.find((event) => event.id === eventId) ?? EVENTS[0];
  const selected = MARBLES.filter(active.select);

  return (
    <div className={styles.lab} data-testid="event-universe">
      <div className={styles.labHeader}>
        <div>
          <p>Central worktable</p>
          <h3>Change the event, not the universe</h3>
          <p>
            Each marble is one distinct outcome. An event is the subset that
            satisfies the question. Watch the denominator stay at ten while the
            selected set changes.
          </p>
        </div>
        <Layers3 aria-hidden="true" size={34} />
      </div>
      <div className={styles.buttonRow}>
        {EVENTS.map((event) => (
          <button
            type="button"
            key={event.id}
            onClick={() => setEventId(event.id)}
            className={
              event.id === eventId ? styles.buttonActive : styles.button
            }
          >
            {event.label}
          </button>
        ))}
      </div>
      <div className={styles.eventWorkspace} style={{ marginTop: 22 }}>
        <div className={styles.objectGrid}>
          {MARBLES.map((marble) => {
            const isSelected = active.select(marble);
            return (
              <div
                className={styles.object}
                data-color={marble.color}
                data-selected={isSelected}
                data-muted={!isSelected}
                key={marble.id}
              >
                {marble.id}
              </div>
            );
          })}
        </div>
        <div className={styles.panel} role="status">
          <span className={styles.tag}>Event {active.notation}</span>
          <h3 style={{ marginTop: 12 }}>{active.label}</h3>
          <p>{active.description}</p>
          <ProbabilityMetric
            label="Probability"
            value={`${selected.length}/10`}
            detail={`${((selected.length / 10) * 100).toFixed(0)}% of the equally likely outcomes`}
          />
          <p>
            Selected set: {"{"}
            {selected.map((marble) => marble.id).join(", ")}
            {"}"}
          </p>
        </div>
      </div>
    </div>
  );
}

function AxiomWorkbench() {
  const king = 4 / 52;
  const heart = 13 / 52;
  const kingOfHearts = 1 / 52;
  const union = unionProbability(king, heart, kingOfHearts);

  return (
    <div className={styles.grid2}>
      <div className={styles.panel}>
        <span className={styles.tag}>Three axioms</span>
        <h3 style={{ marginTop: 12 }}>
          The rules every probability model obeys
        </h3>
        <div className={styles.formulaTrail}>
          <div className={styles.formulaStep}>
            <strong>1</strong>
            <span>Non-negativity: probabilities cannot be below zero.</span>
          </div>
          <div className={styles.formulaStep}>
            <strong>2</strong>
            <span>
              Normalization: the entire sample space has probability one.
            </span>
          </div>
          <div className={styles.formulaStep}>
            <strong>3</strong>
            <span>
              Disjoint additivity: non-overlapping event probabilities add.
            </span>
          </div>
        </div>
      </div>
      <div className={styles.panel}>
        <span className={styles.tag}>General addition rule</span>
        <h3 style={{ marginTop: 12 }}>Subtract the overlap once</h3>
        <p>
          A deck has four kings and thirteen hearts. Adding 4 + 13 counts the
          king of hearts twice.
        </p>
        <div className={styles.status}>
          4/52 + 13/52 − 1/52 = {(union * 52).toFixed(0)}/52 ={" "}
          {(union * 100).toFixed(1)}%
        </div>
        <p>
          If the events were disjoint—king or queen—the overlap would be zero
          and simple addition would be valid.
        </p>
      </div>
    </div>
  );
}

function CountingDecisionLab() {
  const [scenarioId, setScenarioId] =
    useState<(typeof COUNTING_SCENARIOS)[number]["id"]>("pin");
  const scenario =
    COUNTING_SCENARIOS.find((item) => item.id === scenarioId) ??
    COUNTING_SCENARIOS[0];

  return (
    <div className={styles.lab} data-testid="counting-decision-lab">
      <div className={styles.labHeader}>
        <div>
          <p>Count without listing</p>
          <h3>Ask repetition, then order</h3>
          <p>
            The formula is the last step. First decide what makes two outcomes
            different.
          </p>
        </div>
        <Dices aria-hidden="true" size={34} />
      </div>
      <div className={styles.buttonRow}>
        {COUNTING_SCENARIOS.map((item) => (
          <button
            type="button"
            key={item.id}
            onClick={() => setScenarioId(item.id)}
            className={
              item.id === scenarioId ? styles.buttonActive : styles.button
            }
          >
            {item.label}
          </button>
        ))}
      </div>
      <div className={styles.metricGrid}>
        <ProbabilityMetric
          label="Repetition"
          value={scenario.repeat ? "Allowed" : "No"}
          detail="Can the same item appear twice?"
        />
        <ProbabilityMetric
          label="Order"
          value={scenario.order ? "Matters" : "Does not"}
          detail="Would rearrangement create a new outcome?"
        />
        <ProbabilityMetric
          label="Outcome count"
          value={scenario.count.toLocaleString()}
          detail="Size of the sample space"
        />
      </div>
      <div className={styles.grid2} style={{ marginTop: 18 }}>
        <ProbabilityFormula
          label="Counting rule"
          formula={`\[${scenario.formula}\]`}
        />
        <div className={styles.panel} role="status">
          <h3>Why this rule?</h3>
          <p>{scenario.reasoning}</p>
        </div>
      </div>
    </div>
  );
}

function RandomVariableLab() {
  const outcomes = [
    { sequence: "HH", value: 10, probability: 0.25 },
    { sequence: "HT", value: 3, probability: 0.25 },
    { sequence: "TH", value: 3, probability: 0.25 },
    { sequence: "TT", value: -4, probability: 0.25 },
  ];
  const pmf = [
    { value: -4, probability: 0.25 },
    { value: 3, probability: 0.5 },
    { value: 10, probability: 0.25 },
  ];
  const mean = expectedValue(pmf);
  const spread = variance(pmf);

  return (
    <div className={styles.lab} data-testid="random-variable-lab">
      <div className={styles.labHeader}>
        <div>
          <p>Outcome → number</p>
          <h3>A random variable is a translator</h3>
          <p>
            Flip two coins. Win $5 per head and lose $2 per tail. The random
            variable X maps each coin outcome to a numerical payoff.
          </p>
        </div>
      </div>
      <div className={styles.grid2}>
        <div className={styles.coinGrid}>
          {outcomes.map((outcome) => (
            <div className={styles.coinOutcome} key={outcome.sequence}>
              <span>{outcome.sequence}</span>
              <strong>${outcome.value}</strong>
              <span>P = {outcome.probability.toFixed(2)}</span>
            </div>
          ))}
        </div>
        <div>
          <div className={styles.bars}>
            {pmf.map((item) => (
              <div className={styles.barRow} key={item.value}>
                <span>X = {item.value}</span>
                <div className={styles.barTrack}>
                  <div
                    className={styles.barFill}
                    style={{ width: `${(item.probability * 100).toFixed(4)}%` }}
                  />
                </div>
                <strong>{item.probability.toFixed(2)}</strong>
              </div>
            ))}
          </div>
          <div className={styles.metricGrid}>
            <ProbabilityMetric
              label="Expected value"
              value={`$${mean.toFixed(2)}`}
              detail="Long-run average payoff"
            />
            <ProbabilityMetric
              label="Variance"
              value={spread.toFixed(2)}
              detail="Probability-weighted squared spread"
            />
            <ProbabilityMetric
              label="P(make money)"
              value="0.75"
              detail="HH, HT, or TH"
            />
          </div>
        </div>
      </div>
      <div className={styles.status}>
        Expectation path: 10(0.25) + 3(0.50) − 4(0.25) = {mean.toFixed(2)}. The
        value $3 has probability 0.50 because two different outcomes map to it.
      </div>
    </div>
  );
}

function ContinuousDensityLab() {
  const [lower, setLower] = useState(15);
  const [upper, setUpper] = useState(17);
  const safeLower = Math.min(lower, upper - 0.25);
  const safeUpper = Math.max(upper, lower + 0.25);
  const width = safeUpper - safeLower;
  const probability = width / 4;
  const leftPercent = 8 + ((safeLower - 14) / 4) * 84;
  const widthPercent = (width / 4) * 84;

  return (
    <div className={styles.lab} data-testid="continuous-density-lab">
      <div className={styles.labHeader}>
        <div>
          <p>Mass versus density</p>
          <h3>Exact points have zero area</h3>
          <p>
            A slicer produces lengths uniformly between 14 and 18 inches. The
            density is 1/4 per inch. Probability comes from an interval’s area,
            not the height at one infinitely precise point.
          </p>
        </div>
      </div>
      <div className={styles.grid2}>
        <div>
          <div
            className={styles.densityPlot}
            aria-label="Uniform density from 14 to 18 inches"
          >
            <div className={styles.densityBase} />
            <div
              className={styles.densitySelection}
              style={{
                left: `${leftPercent.toFixed(4)}%`,
                width: `${widthPercent.toFixed(4)}%`,
              }}
            />
            <div className={styles.densityAxis}>
              <span>14</span>
              <span>15</span>
              <span>16</span>
              <span>17</span>
              <span>18</span>
            </div>
          </div>
          <div className={styles.status} role="status">
            Area = width × height = {width.toFixed(2)} × 0.25 ={" "}
            {probability.toFixed(3)}
          </div>
        </div>
        <div className={styles.controls}>
          <div className={styles.control}>
            <label htmlFor="density-lower">
              Lower bound: {safeLower.toFixed(2)}
            </label>
            <input
              id="density-lower"
              type="range"
              min="14"
              max="17.75"
              step="0.25"
              value={safeLower}
              onChange={(event) => setLower(Number(event.target.value))}
            />
          </div>
          <div className={styles.control}>
            <label htmlFor="density-upper">
              Upper bound: {safeUpper.toFixed(2)}
            </label>
            <input
              id="density-upper"
              type="range"
              min="14.25"
              max="18"
              step="0.25"
              value={safeUpper}
              onChange={(event) => setUpper(Number(event.target.value))}
            />
          </div>
          <ProbabilityInsight title="PMF and PDF answer different questions">
            <p>
              A discrete PMF can assign mass to X = 6. A continuous PDF assigns
              density, so P(X = 16.000…) = 0 while P(15 ≤ X ≤ 17) is meaningful.
            </p>
          </ProbabilityInsight>
        </div>
      </div>
    </div>
  );
}

function CalibrationAndDecision() {
  return (
    <div className={styles.grid3}>
      <div className={styles.panel}>
        <span className={styles.tag}>Bernoulli</span>
        <h3 style={{ marginTop: 12 }}>Two outcomes</h3>
        <p>
          A Bernoulli variable records success/failure, yes/no, or 1/0 with one
          parameter: the probability of success.
        </p>
      </div>
      <div className={styles.panel}>
        <span className={styles.tag}>Categorical</span>
        <h3 style={{ marginTop: 12 }}>One of many labels</h3>
        <p>
          A categorical distribution places probability mass across several
          mutually exclusive choices, such as cat, dog, and fox.
        </p>
      </div>
      <div className={styles.panel}>
        <span className={styles.tag}>Calibration</span>
        <h3 style={{ marginTop: 12 }}>
          Probabilities earn meaning over repetitions
        </h3>
        <p>
          Among many cases predicted at 70%, roughly 70% should occur. Accuracy
          alone cannot tell whether confidence levels are trustworthy.
        </p>
      </div>
    </div>
  );
}

export default function CrashProbabilityL1LearningPage({ experience }: Props) {
  return (
    <ProbabilityCourse
      experience={experience}
      station="l1"
      kicker="Lecture 1 · Probability foundations"
      headline="An event is a set—not a vibe."
      introduction="Probability clicks when the universe is concrete. Start with ten unique marbles, group outcomes into events, and let every rule grow from what the selected set actually contains."
      heroVisual={<MarbleHero />}
    >
      <ProbabilitySection
        id="events"
        eyebrow="01 · Outcomes become events"
        title="Keep the universe fixed. Change the question."
        lead="The sample space lists everything that can happen. An outcome is one member. An event is any subset—including a singleton event with one outcome."
      >
        <EventUniverse />
        <div style={{ marginTop: 22 }}>
          <ProbabilityCheck
            testId="event-outcome-check"
            title="Exactly one head is not one outcome"
            question="Two ordered coin flips have outcomes HH, HT, TH, TT. Which set is the event ‘exactly one head’?"
            correctIndex={1}
            options={[
              {
                label: "{H, T}",
                explanation:
                  "H and T are not complete ordered outcomes for two flips.",
              },
              {
                label: "{HT, TH}",
                explanation:
                  "Both ordered outcomes contain exactly one head, so the event has two members.",
              },
              {
                label: "{HH}",
                explanation: "HH has two heads, not exactly one.",
              },
            ]}
          />
        </div>
      </ProbabilitySection>

      <ProbabilitySection
        id="rules"
        eyebrow="02 · Laws and counting"
        title="Every shortcut must preserve the set."
        lead="The axioms keep probability coherent. Complements remove a set, intersections keep overlap, unions combine sets, and counting formulas measure large equally likely sample spaces without enumerating them."
      >
        <AxiomWorkbench />
        <div style={{ marginTop: 22 }}>
          <CountingDecisionLab />
        </div>
        <div className={styles.grid2} style={{ marginTop: 22 }}>
          <ProbabilityFormula
            label="Complement"
            formula={String.raw`\[P(A^c)=1-P(A)\]`}
          >
            “At least one” problems are often easiest by counting the single
            excluded case. For three fair flips: 1 − P(TTT) = 1 − (1/2)³ = 7/8.
          </ProbabilityFormula>
          <ProbabilityFormula
            label="Addition rule"
            formula={String.raw`\[P(A\cup B)=P(A)+P(B)-P(A\cap B)\]`}
          >
            The subtraction repairs double-counting. If A and B are disjoint,
            the intersection is empty and its probability is zero.
          </ProbabilityFormula>
        </div>
      </ProbabilitySection>

      <ProbabilitySection
        id="variables"
        eyebrow="03 · Random variables and distributions"
        title="Translate messy outcomes into values you can analyze."
        lead="A random variable is a function from outcomes to values. Its distribution collects the probability carried by every outcome that maps to the same value."
      >
        <RandomVariableLab />
        <div style={{ marginTop: 22 }}>
          <ContinuousDensityLab />
        </div>
      </ProbabilitySection>

      <ProbabilitySection
        id="expectation"
        eyebrow="04 · Expectation, spread, and trustworthy confidence"
        title="The average is a center of gravity, not a promise."
        lead="Expectation weights each value by probability. Variance measures squared spread around that mean. Calibration asks whether stated probabilities match frequencies across many similar predictions."
      >
        <CalibrationAndDecision />
        <div className={styles.grid2} style={{ marginTop: 22 }}>
          <ProbabilityFormula
            label="Expectation and variance"
            formula={String.raw`\[\mathbb{E}[X]=\sum_x xP(X=x),\qquad \operatorname{Var}(X)=\sum_x P(X=x)(x-\mu)^2\]`}
          />
          <ProbabilityInsight title="Linearity does not need independence">
            <p>
              For finite expectations,{" "}
              <InlineProbabilityMath
                text={String.raw`\(\mathbb{E}[X+Y]=\mathbb{E}[X]+\mathbb{E}[Y]\)`}
              />{" "}
              even when X and Y are dependent. Product factorization generally
              does require independence.
            </p>
          </ProbabilityInsight>
        </div>
        <div style={{ marginTop: 22 }}>
          <ProbabilityInsight
            title="Expected value does not contain your utility"
            tone="warning"
          >
            <p>
              A 5% chance of losing an $800 phone has expected loss $40, but a
              person may still pay more than $40 to avoid an unaffordable shock.
              Mean payoff, variance, constraints, and risk tolerance answer
              different parts of the decision.
            </p>
          </ProbabilityInsight>
        </div>
      </ProbabilitySection>

      <ProbabilityQuizLaunch
        experience={experience}
        recap={[
          "A sample space contains outcomes; an event is a set of outcomes.",
          "Probability axioms imply complement and addition rules.",
          "Counting starts by deciding whether repetition is allowed and whether order matters.",
          "Random variables map outcomes to values; distributions collect their probability mass.",
          "PMFs assign point mass, while PDFs use area over intervals.",
          "Expectation is a weighted average; variance and calibration reveal information the mean alone cannot.",
        ]}
      />
    </ProbabilityCourse>
  );
}
