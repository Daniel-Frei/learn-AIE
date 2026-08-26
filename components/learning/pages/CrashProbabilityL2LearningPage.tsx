"use client";

import { useMemo, useState } from "react";
import { Filter, GitBranch, ScanSearch } from "lucide-react";
import type { LearningExperience } from "../../../lib/learning";
import {
  conditionalProbability,
  naturalFrequencyCounts,
  totalProbability,
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

const REMAINING_MARBLES = [
  { id: "R1", color: "red" },
  { id: "R2", color: "red" },
  { id: "R3", color: "red" },
  { id: "R4", color: "red" },
  { id: "R5", color: "red" },
  { id: "B2", color: "blue" },
  { id: "B3", color: "blue" },
  { id: "G1", color: "green" },
  { id: "G2", color: "green" },
] as const;

function HeroEvidenceLens() {
  return (
    <div
      className={styles.panel}
      aria-label="An evidence lens shrinking a probability universe"
    >
      <div className={styles.labHeader}>
        <div>
          <p>Evidence lens</p>
          <h3>Start with 10 possibilities</h3>
        </div>
        <ScanSearch aria-hidden="true" size={34} />
      </div>
      <div className={styles.objectGrid} style={{ marginTop: 18 }}>
        {Array.from({ length: 10 }, (_, index) => (
          <span
            className={styles.object}
            data-selected={index === 0 || index === 3 || index === 7}
            data-muted={index !== 0 && index !== 3 && index !== 7}
            key={index}
          >
            {index + 1}
          </span>
        ))}
      </div>
      <div className={styles.formulaTrail} style={{ marginTop: 18 }}>
        <div className={styles.formulaStep}>
          <strong>Before</strong>
          <span>all plausible outcomes</span>
        </div>
        <div className={styles.formulaStep}>
          <strong>Evidence</strong>
          <span>keep compatible outcomes</span>
        </div>
        <div className={styles.formulaStep}>
          <strong>After</strong>
          <span>renormalize what remains</span>
        </div>
      </div>
    </div>
  );
}

function ConditionalUniverseLab() {
  const [firstColor, setFirstColor] = useState<"blue" | "red">("blue");
  const firstWasBlue = firstColor === "blue";
  const remaining = firstWasBlue
    ? REMAINING_MARBLES
    : [
        { id: "R2", color: "red" },
        { id: "R3", color: "red" },
        { id: "R4", color: "red" },
        { id: "R5", color: "red" },
        { id: "B1", color: "blue" },
        { id: "B2", color: "blue" },
        { id: "B3", color: "blue" },
        { id: "G1", color: "green" },
        { id: "G2", color: "green" },
      ];
  const redCount = remaining.filter((marble) => marble.color === "red").length;
  const probability = redCount / remaining.length;

  return (
    <div className={styles.lab} data-testid="conditional-universe-lab">
      <div className={styles.labHeader}>
        <div>
          <p>Without-replacement draw</p>
          <h3>Conditioning literally changes the bag</h3>
        </div>
        <Filter aria-hidden="true" size={30} />
      </div>
      <div className={styles.buttonRow}>
        <button
          className={firstWasBlue ? styles.buttonActive : styles.button}
          type="button"
          onClick={() => setFirstColor("blue")}
        >
          First draw was blue
        </button>
        <button
          className={!firstWasBlue ? styles.buttonActive : styles.button}
          type="button"
          onClick={() => setFirstColor("red")}
        >
          First draw was red
        </button>
      </div>
      <div className={styles.eventWorkspace} style={{ marginTop: 20 }}>
        <div className={styles.objectGrid}>
          {remaining.map((marble) => (
            <span
              className={styles.object}
              data-color={marble.color}
              data-selected={marble.color === "red"}
              key={marble.id}
            >
              {marble.id}
            </span>
          ))}
        </div>
        <div className={styles.metricGrid}>
          <ProbabilityMetric
            label="new universe"
            value="9"
            detail="one marble is gone"
          />
          <ProbabilityMetric
            label="red remain"
            value={String(redCount)}
            detail="favorable outcomes"
          />
          <ProbabilityMetric
            label="next is red"
            value={`${(probability * 100).toFixed(1)}%`}
            detail={`${redCount} ÷ 9`}
          />
        </div>
      </div>
      <ProbabilityFormula
        label="Intersection divided by the evidence"
        formula={
          firstWasBlue
            ? String.raw`\[P(R_2\mid B_1)=\frac{P(R_2\cap B_1)}{P(B_1)}=\frac{5/10\times 3/9}{3/10}=\frac{5}{9}\]`
            : String.raw`\[P(R_2\mid R_1)=\frac{P(R_2\cap R_1)}{P(R_1)}=\frac{5/10\times 4/9}{5/10}=\frac{4}{9}\]`
        }
      >
        The vertical bar means “inside the world where the evidence is true.” It
        does not mean divide mechanically; it tells you which universe to use.
      </ProbabilityFormula>
    </div>
  );
}

function JointTableLab() {
  const [view, setView] = useState<"counts" | "probabilities">("counts");
  const cells = [50, 10, 5, 35];
  const show = (value: number) =>
    view === "counts" ? String(value) : (value / 100).toFixed(2);
  const observedJoint = conditionalProbability(0.5, 0.6);
  const independentJoint = 0.6 * 0.55;

  return (
    <div className={styles.lab} data-testid="joint-table-lab">
      <div className={styles.labHeader}>
        <div>
          <p>100 students</p>
          <h3>One table, three different questions</h3>
        </div>
        <div className={styles.buttonRow}>
          <button
            type="button"
            className={view === "counts" ? styles.buttonActive : styles.button}
            onClick={() => setView("counts")}
          >
            Counts
          </button>
          <button
            type="button"
            className={
              view === "probabilities" ? styles.buttonActive : styles.button
            }
            onClick={() => setView("probabilities")}
          >
            Probabilities
          </button>
        </div>
      </div>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <caption>Midterm and final pass results</caption>
          <thead>
            <tr>
              <th scope="col">Outcome</th>
              <th scope="col">Final pass</th>
              <th scope="col">Final fail</th>
              <th scope="col">Row total</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row">Midterm pass</th>
              <td>{show(cells[0])}</td>
              <td>{show(cells[1])}</td>
              <td>{show(60)}</td>
            </tr>
            <tr>
              <th scope="row">Midterm fail</th>
              <td>{show(cells[2])}</td>
              <td>{show(cells[3])}</td>
              <td>{show(40)}</td>
            </tr>
            <tr>
              <th scope="row">Column total</th>
              <td>{show(55)}</td>
              <td>{show(45)}</td>
              <td>{show(100)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div className={styles.metricGrid}>
        <ProbabilityMetric label="joint" value="50%" detail="passed both" />
        <ProbabilityMetric
          label="conditional"
          value={`${(observedJoint * 100).toFixed(1)}%`}
          detail="final pass among midterm pass"
        />
        <ProbabilityMetric
          label="if independent"
          value={`${(independentJoint * 100).toFixed(1)}%`}
          detail="0.60 × 0.55 would pass both"
        />
      </div>
      <ProbabilityInsight
        title="Independence is a claim you can check"
        tone="warning"
      >
        <p>
          If the results were independent, the joint probability would be 33%,
          not the observed 50%. Equivalently, learning that someone passed the
          midterm changes the final-pass probability from 55% to 83.3%.
        </p>
      </ProbabilityInsight>
    </div>
  );
}

function TotalProbabilityLab() {
  const [bagAPrior, setBagAPrior] = useState(70);
  const priorA = bagAPrior / 100;
  const priorB = 1 - priorA;
  const blueA = 0.5;
  const blueB = 0.875;
  const fromA = priorA * blueA;
  const fromB = priorB * blueB;
  const blue = totalProbability([
    { prior: priorA, conditional: blueA },
    { prior: priorB, conditional: blueB },
  ]);

  return (
    <div className={styles.lab} data-testid="total-probability-lab">
      <div className={styles.labHeader}>
        <div>
          <p>Hidden route, visible result</p>
          <h3>Add complete paths, not disconnected numbers</h3>
        </div>
        <GitBranch aria-hidden="true" size={31} />
      </div>
      <label className={styles.control}>
        <span>Chance of choosing bag A: {bagAPrior}%</span>
        <input
          aria-label="Chance of choosing bag A"
          type="range"
          min="10"
          max="90"
          step="5"
          value={bagAPrior}
          onChange={(event) => setBagAPrior(Number(event.target.value))}
        />
      </label>
      <div className={styles.pathTree}>
        <div className={styles.pathRow}>
          <span className={styles.pathNode}>Start</span>
          <span className={styles.pathArrow}>→</span>
          <span className={styles.pathNode}>
            Bag A · {(priorA * 100).toFixed(0)}%
          </span>
          <span className={styles.pathArrow}>→</span>
          <span className={styles.pathNode}>Blue · 50%</span>
          <span className={styles.tag}>path {(fromA * 100).toFixed(1)}%</span>
        </div>
        <div className={styles.pathRow}>
          <span className={styles.pathNode}>Start</span>
          <span className={styles.pathArrow}>→</span>
          <span className={styles.pathNode}>
            Bag B · {(priorB * 100).toFixed(0)}%
          </span>
          <span className={styles.pathArrow}>→</span>
          <span className={styles.pathNode}>Blue · 87.5%</span>
          <span className={styles.tag}>path {(fromB * 100).toFixed(1)}%</span>
        </div>
      </div>
      <ProbabilityFormula
        label="Law of total probability"
        formula={String.raw`\[P(blue)=P(A)P(blue\mid A)+P(B)P(blue\mid B)=${priorA.toFixed(2)}(0.50)+${priorB.toFixed(2)}(0.875)=${blue.toFixed(4)}\]`}
      >
        The cases A and B partition the universe: exactly one happens, and
        together they cover every route to a blue marble.
      </ProbabilityFormula>
    </div>
  );
}

function BayesPopulationLab() {
  const [prevalence, setPrevalence] = useState(5);
  const [sensitivity, setSensitivity] = useState(90);
  const [specificity, setSpecificity] = useState(90);
  const counts = useMemo(
    () =>
      naturalFrequencyCounts({
        population: 1000,
        prevalence: prevalence / 100,
        sensitivity: sensitivity / 100,
        specificity: specificity / 100,
      }),
    [prevalence, sensitivity, specificity],
  );
  const proxyGroups = useMemo(() => {
    const scale = 100 / counts.population;
    const limits = [
      { end: Math.round(counts.truePositive * scale), group: "true-positive" },
      {
        end: Math.round((counts.truePositive + counts.falsePositive) * scale),
        group: "false-positive",
      },
      {
        end: Math.round(
          (counts.truePositive + counts.falsePositive + counts.falseNegative) *
            scale,
        ),
        group: "false-negative",
      },
    ];
    return Array.from(
      { length: 100 },
      (_, index) =>
        limits.find(({ end }) => index < end)?.group ?? "true-negative",
    );
  }, [counts]);

  return (
    <div className={styles.lab} data-testid="bayes-population-lab">
      <div className={styles.labHeader}>
        <div>
          <p>1,000-person frequency board</p>
          <h3>Bayes becomes bookkeeping when you count people</h3>
        </div>
        <ScanSearch aria-hidden="true" size={31} />
      </div>
      <div className={styles.controls}>
        <label className={styles.control}>
          <span>Prevalence: {prevalence}%</span>
          <input
            aria-label="Prevalence"
            type="range"
            min="1"
            max="20"
            value={prevalence}
            onChange={(event) => setPrevalence(Number(event.target.value))}
          />
        </label>
        <label className={styles.control}>
          <span>Sensitivity: {sensitivity}%</span>
          <input
            aria-label="Sensitivity"
            type="range"
            min="80"
            max="99"
            value={sensitivity}
            onChange={(event) => setSensitivity(Number(event.target.value))}
          />
        </label>
        <label className={styles.control}>
          <span>Specificity: {specificity}%</span>
          <input
            aria-label="Specificity"
            type="range"
            min="80"
            max="99"
            value={specificity}
            onChange={(event) => setSpecificity(Number(event.target.value))}
          />
        </label>
      </div>
      <div
        className={styles.populationGrid}
        aria-label="One hundred icons, each representing ten people"
      >
        {proxyGroups.map((group, index) => (
          <span className={styles.person} data-group={group} key={index} />
        ))}
      </div>
      <div className={styles.metricGrid}>
        <ProbabilityMetric
          label="true positives"
          value={String(counts.truePositive)}
          detail="condition + positive"
        />
        <ProbabilityMetric
          label="false positives"
          value={String(counts.falsePositive)}
          detail="no condition + positive"
        />
        <ProbabilityMetric
          label="all positives"
          value={String(counts.positiveTests)}
          detail="the new evidence universe"
        />
        <ProbabilityMetric
          label="posterior"
          value={`${(counts.posterior * 100).toFixed(1)}%`}
          detail="condition among positives"
        />
      </div>
      <ProbabilityFormula
        label="Bayes through natural frequencies"
        formula={String.raw`\[P(condition\mid +)=\frac{\text{true positives}}{\text{all positives}}=\frac{${counts.truePositive}}{${counts.truePositive}+${counts.falsePositive}}=${counts.posterior.toFixed(3)}\]`}
      >
        Sensitivity answers “positive among people with the condition.” The
        posterior reverses that direction: “condition among people who tested
        positive.” Base rates determine how different those answers can be.
      </ProbabilityFormula>
    </div>
  );
}

export default function CrashProbabilityL2LearningPage({ experience }: Props) {
  return (
    <ProbabilityCourse
      experience={experience}
      station="l2"
      kicker="Station L2 · Conditional probability and Bayes"
      headline="Evidence does not add a footnote. It changes the universe."
      introduction="Conditioning is the move probability makes when information arrives: discard incompatible possibilities, keep their relative weights, and renormalize. Once that picture is stable, Bayes’ rule is simply the same update read in reverse."
      heroVisual={<HeroEvidenceLens />}
    >
      <ProbabilitySection
        id="conditioning"
        eyebrow="01 · Shrink the universe"
        title="The bar means: work inside this world."
        lead="Conditional probability is easier to see than to memorize. A first draw changes what is physically left in the bag; the denominator must change with it."
      >
        <ConditionalUniverseLab />
        <ProbabilityCheck
          testId="l2-universe-check"
          title="Choose the right denominator"
          question="A 20-sided die lands above 15. Given that evidence, what is the chance it shows an even number?"
          options={[
            {
              label: "2/5",
              explanation:
                "Inside {16,17,18,19,20}, the compatible even outcomes are 16, 18, and 20—three of five.",
            },
            {
              label: "3/5",
              explanation:
                "The evidence leaves five outcomes, three of which are even.",
            },
            {
              label: "3/20",
              explanation:
                "Twenty was the old universe; conditioning replaces that denominator with five.",
            },
          ]}
          correctIndex={1}
        />
      </ProbabilitySection>

      <ProbabilitySection
        id="joint"
        eyebrow="02 · Joint, marginal, and independent"
        title="Read the same data from three directions."
        lead="A joint asks for two facts together. A marginal ignores one dimension. A conditional holds one fact fixed and asks what fraction remains. Independence is the special case where conditioning changes nothing."
      >
        <JointTableLab />
        <div className={styles.grid2} style={{ marginTop: 22 }}>
          <ProbabilityFormula
            label="Product rule"
            formula={String.raw`\[P(A\cap B)=P(A)P(B\mid A)\]`}
          >
            This is always valid when the conditional is defined. Only under
            independence may you replace{" "}
            <InlineProbabilityMath text={String.raw`\(P(B\mid A)\)`} /> with{" "}
            <InlineProbabilityMath text={String.raw`\(P(B)\)`} />.
          </ProbabilityFormula>
          <ProbabilityInsight title="Conditional independence is local">
            <p>
              Two variables can become independent after holding a third fixed.
              For example, shoe size and reading ability may be associated in a
              mixed-age population but nearly unrelated within a single age
              group. The conditioning variable is part of the claim.
            </p>
          </ProbabilityInsight>
        </div>
      </ProbabilitySection>

      <ProbabilitySection
        id="total-probability"
        eyebrow="03 · Partition and recombine"
        title="When the cause is hidden, sum every complete route."
        lead="The law of total probability is a path accounting rule. Split the world into mutually exclusive, exhaustive cases; multiply along each route; add across routes."
      >
        <TotalProbabilityLab />
      </ProbabilitySection>

      <ProbabilitySection
        id="bayes"
        eyebrow="04 · Reverse the condition"
        title="A positive result is not the same thing as a positive case."
        lead="Bayes combines the old prevalence with the test’s behavior. Natural frequencies make the reversal inspectable and expose why false positives can dominate when the condition is rare."
      >
        <BayesPopulationLab />
        <div className={styles.grid2} style={{ marginTop: 22 }}>
          <ProbabilityFormula
            label="Bayes’ rule"
            formula={String.raw`\[P(H\mid E)=\frac{P(E\mid H)P(H)}{P(E)}\]`}
          >
            Prior <InlineProbabilityMath text={String.raw`\(P(H)\)`} /> ×
            likelihood{" "}
            <InlineProbabilityMath text={String.raw`\(P(E\mid H)\)`} /> gives
            the joint path to H and E. Dividing by all routes to E renormalizes
            the evidence universe.
          </ProbabilityFormula>
          <ProbabilityInsight
            title="Association is not automatically causation"
            tone="warning"
          >
            <p>
              Conditioning can reveal a relationship, but it cannot by itself
              prove which variable causes the other. A common cause, selection
              process, or collider can create—or reverse—an association. State
              the data-generating story before making a causal claim.
            </p>
          </ProbabilityInsight>
        </div>
      </ProbabilitySection>

      <ProbabilitySection
        id="ai"
        eyebrow="05 · Conditional models"
        title="Most modern AI models answer a conditional question."
        lead="The symbols change by field, but the evidence-lens move stays the same: predict a label given features, the next token given prior tokens, an action given state, or a cleaner sample given a noisy one."
      >
        <div className={styles.grid2}>
          <div className={styles.panel}>
            <h3>Four familiar conditionals</h3>
            <div className={styles.formulaTrail}>
              <div className={styles.formulaStep}>
                <strong>Classifier</strong>
                <InlineProbabilityMath text={String.raw`\(P(y\mid x)\)`} />
              </div>
              <div className={styles.formulaStep}>
                <strong>Language model</strong>
                <InlineProbabilityMath
                  text={String.raw`\(P(x_t\mid x_{<t})\)`}
                />
              </div>
              <div className={styles.formulaStep}>
                <strong>Policy</strong>
                <InlineProbabilityMath text={String.raw`\(\pi(a\mid s)\)`} />
              </div>
              <div className={styles.formulaStep}>
                <strong>Denoiser</strong>
                <InlineProbabilityMath
                  text={String.raw`\(p(x_{t-1}\mid x_t)\)`}
                />
              </div>
            </div>
          </div>
          <ProbabilityInsight
            title="A model learns from the data universe it sees"
            tone="warning"
          >
            <p>
              A conditional estimate can be numerically accurate on training
              data and fail after the population or evidence process changes.
              Ask what was conditioned on, how examples were selected, and
              whether deployment keeps the same joint distribution.
            </p>
          </ProbabilityInsight>
        </div>
      </ProbabilitySection>

      <ProbabilityQuizLaunch
        experience={experience}
        recap={[
          "Conditioning restricts the sample space and renormalizes what remains.",
          "Joint, marginal, and conditional probabilities read a distribution in different directions.",
          "Independence means conditioning does not change a probability; conditional independence names the context.",
          "Total probability sums mutually exclusive, exhaustive paths to the evidence.",
          "Bayes reverses a conditional by combining likelihood, prior, and total evidence.",
          "Natural frequencies expose base-rate effects and help prevent direction errors.",
        ]}
      />
    </ProbabilityCourse>
  );
}
