"use client";

import { useState } from "react";
import { Braces, Grid3X3, Scale, SquareFunction } from "lucide-react";
import type { LearningExperience } from "../../../lib/learning";
import {
  normalizeWeights,
  oddsToProbability,
  probabilityToOdds,
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

const NOTATION = [
  {
    id: "function",
    symbol: String.raw`\(f_\theta(x)\)`,
    title: "A function with parameters",
    explanation:
      "x is the input. θ names the adjustable parameters. The function tells you how the input is transformed; it does not mean f times θ times x.",
  },
  {
    id: "subscript",
    symbol: String.raw`\(x_t\)`,
    title: "One indexed value",
    explanation:
      "The subscript identifies a position, time, or item. x₃ means the third value in a sequence, not x multiplied by 3.",
  },
  {
    id: "argmax",
    symbol: String.raw`\(\arg\max_i s_i\)`,
    title: "The location of the maximum",
    explanation:
      "max returns the largest score. argmax returns the index or choice that produced it. If scores are [2, 7, 4], max is 7 and argmax is the second position.",
  },
  {
    id: "sum",
    symbol: String.raw`\(\sum_{i=1}^{n} w_i x_i\)`,
    title: "A repeated weighted addition",
    explanation:
      "Start at i = 1, multiply each xᵢ by its weight wᵢ, and add until i = n. The sigma is compact loop notation.",
  },
  {
    id: "product",
    symbol: String.raw`\(\prod_{t=1}^{T} p_t\)`,
    title: "A repeated multiplication",
    explanation:
      "The capital pi means multiply the terms rather than add them. Products of probabilities become important for sequences.",
  },
] as const;

function ReadinessHero() {
  return (
    <div className={styles.lab} aria-label="Normalization preview">
      <div className={styles.labHeader}>
        <div>
          <p>First instrument</p>
          <h3>Three weights become one whole</h3>
        </div>
        <Scale aria-hidden="true" size={34} />
      </div>
      <div className={styles.bars}>
        {[
          ["2 parts", 20],
          ["3 parts", 30],
          ["5 parts", 50],
        ].map(([label, width]) => (
          <div className={styles.barRow} key={String(label)}>
            <span>{label}</span>
            <div className={styles.barTrack}>
              <div
                className={styles.barFill}
                style={{ width: `${Number(width).toFixed(4)}%` }}
              />
            </div>
            <strong>{width}%</strong>
          </div>
        ))}
      </div>
      <div className={styles.status}>
        Divide every weight by the total: 2 + 3 + 5 = 10. The normalized values
        are 0.2, 0.3, and 0.5—and now they sum to 1.
      </div>
    </div>
  );
}

function NormalizationBench() {
  const [weights, setWeights] = useState([2, 3, 5]);
  const normalized = normalizeWeights(weights);
  const total = weights.reduce((sum, value) => sum + value, 0);

  return (
    <div className={styles.lab} data-testid="normalization-bench">
      <div className={styles.labHeader}>
        <div>
          <p>Manipulate</p>
          <h3>Normalization bench</h3>
          <p>
            A ratio compares a part with a total. Change the raw weights; the
            normalized shares always rebuild one complete whole.
          </p>
        </div>
        <Braces aria-hidden="true" size={32} />
      </div>
      <div className={styles.grid2}>
        <div className={styles.controls}>
          {weights.map((weight, index) => (
            <div className={styles.control} key={index}>
              <label htmlFor={`weight-${index}`}>
                Weight {index + 1}: {weight}
              </label>
              <input
                id={`weight-${index}`}
                type="range"
                min="1"
                max="12"
                value={weight}
                onChange={(event) =>
                  setWeights((current) =>
                    current.map((value, currentIndex) =>
                      currentIndex === index
                        ? Number(event.target.value)
                        : value,
                    ),
                  )
                }
              />
            </div>
          ))}
        </div>
        <div>
          <div className={styles.bars}>
            {normalized.map((probability, index) => (
              <div className={styles.barRow} key={index}>
                <span>Share {index + 1}</span>
                <div className={styles.barTrack}>
                  <div
                    className={styles.barFill}
                    style={{ width: `${(probability * 100).toFixed(4)}%` }}
                  />
                </div>
                <strong>{probability.toFixed(2)}</strong>
              </div>
            ))}
          </div>
          <div className={styles.formulaTrail}>
            <div className={styles.formulaStep}>
              <strong>1</strong>
              <span>
                Total = {weights.join(" + ")} = {total}
              </span>
            </div>
            <div className={styles.formulaStep}>
              <strong>2</strong>
              <span>
                Normalized = [
                {weights.map((weight) => `${weight}/${total}`).join(", ")}]
              </span>
            </div>
            <div className={styles.formulaStep}>
              <strong>3</strong>
              <span>
                Check ={" "}
                {normalized.map((value) => value.toFixed(2)).join(" + ")} ={" "}
                {normalized.reduce((sum, value) => sum + value, 0).toFixed(2)}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function OddsConverter() {
  const [percent, setPercent] = useState(75);
  const probability = percent / 100;
  const odds = probabilityToOdds(probability);
  const recovered = oddsToProbability(odds);

  return (
    <div className={styles.lab} data-testid="odds-converter">
      <div className={styles.labHeader}>
        <div>
          <p>Translate</p>
          <h3>Probability ↔ odds</h3>
          <p>
            Probability compares success with all trials. Odds compare success
            with failure. They describe the same balance with different
            denominators.
          </p>
        </div>
      </div>
      <div className={styles.control}>
        <label htmlFor="probability-percent">Probability: {percent}%</label>
        <input
          id="probability-percent"
          type="range"
          min="5"
          max="95"
          step="5"
          value={percent}
          onChange={(event) => setPercent(Number(event.target.value))}
        />
      </div>
      <div className={styles.metricGrid}>
        <ProbabilityMetric
          label="Probability"
          value={probability.toFixed(2)}
          detail={`${percent} successes per 100 trials`}
        />
        <ProbabilityMetric
          label="Odds for"
          value={`${odds.toFixed(2)} : 1`}
          detail={`${percent} successes to ${100 - percent} failures`}
        />
        <ProbabilityMetric
          label="Converted back"
          value={recovered.toFixed(2)}
          detail="odds ÷ (1 + odds)"
        />
      </div>
      <div className={styles.status} role="status">
        Arithmetic: {probability.toFixed(2)} ÷ (1 − {probability.toFixed(2)}) ={" "}
        {odds.toFixed(2)}. Reverse it: {odds.toFixed(2)} ÷ (1 +{" "}
        {odds.toFixed(2)}) = {recovered.toFixed(2)}.
      </div>
    </div>
  );
}

function NotationDecoder() {
  const [activeId, setActiveId] =
    useState<(typeof NOTATION)[number]["id"]>("function");
  const active = NOTATION.find((item) => item.id === activeId) ?? NOTATION[0];

  return (
    <div className={styles.lab} data-testid="notation-decoder">
      <div className={styles.labHeader}>
        <div>
          <p>Read before calculating</p>
          <h3>Notation decoder</h3>
          <p>
            Symbols compress instructions. Click one to unpack what the marks
            tell you to do.
          </p>
        </div>
        <SquareFunction aria-hidden="true" size={34} />
      </div>
      <div className={styles.buttonRow}>
        {NOTATION.map((item) => (
          <button
            type="button"
            key={item.id}
            className={
              item.id === activeId ? styles.buttonActive : styles.button
            }
            onClick={() => setActiveId(item.id)}
          >
            <InlineProbabilityMath text={item.symbol} />
          </button>
        ))}
      </div>
      <div className={styles.grid2} style={{ marginTop: 20 }}>
        <ProbabilityFormula
          label={active.title}
          formula={`\[${active.symbol.slice(2, -2)}\]`}
        />
        <div className={styles.panel} role="status">
          <h3>Say it in words</h3>
          <p>{active.explanation}</p>
        </div>
      </div>
    </div>
  );
}

function ShapeBench() {
  const [rows, setRows] = useState(3);
  const [inner, setInner] = useState(4);
  const [columns, setColumns] = useState(2);

  return (
    <div className={styles.lab} data-testid="shape-bench">
      <div className={styles.labHeader}>
        <div>
          <p>Dimension arithmetic</p>
          <h3>What shape comes out?</h3>
          <p>
            In a matrix product, the inner dimensions must match. They disappear
            into the calculation; the outer dimensions describe the result.
          </p>
        </div>
        <Grid3X3 aria-hidden="true" size={34} />
      </div>
      <div className={styles.grid3}>
        {[
          { label: "Rows in A", value: rows, setter: setRows },
          { label: "Shared inner size", value: inner, setter: setInner },
          { label: "Columns in B", value: columns, setter: setColumns },
        ].map(({ label, value, setter }) => (
          <div className={styles.control} key={label}>
            <label>
              {label}: {value}
            </label>
            <input
              type="range"
              min="1"
              max="8"
              value={value}
              onChange={(event) => setter(Number(event.target.value))}
            />
          </div>
        ))}
      </div>
      <div className={styles.metricGrid}>
        <ProbabilityMetric
          label="Matrix A"
          value={`${rows} × ${inner}`}
          detail="rows × shared dimension"
        />
        <ProbabilityMetric
          label="Matrix B"
          value={`${inner} × ${columns}`}
          detail="shared dimension × columns"
        />
        <ProbabilityMetric
          label="A × B"
          value={`${rows} × ${columns}`}
          detail="outer dimensions survive"
        />
      </div>
    </div>
  );
}

function PowersAndLogsLab() {
  const [successPercent, setSuccessPercent] = useState(20);
  const [trials, setTrials] = useState(5);
  const [discountPercent, setDiscountPercent] = useState(80);
  const [horizon, setHorizon] = useState(5);
  const success = successPercent / 100;
  const atLeastOne = 1 - (1 - success) ** trials;
  const discount = discountPercent / 100;
  const terms = Array.from(
    { length: horizon },
    (_, index) => 10 * discount ** index,
  );
  const discountedTotal = terms.reduce((sum, value) => sum + value, 0);

  return (
    <div className={styles.grid2}>
      <div className={styles.lab} data-testid="complement-power-lab">
        <div className={styles.labHeader}>
          <div>
            <p>Complement with a power</p>
            <h3>At least one success</h3>
            <p>
              Counting every success pattern is tedious. Count the one excluded
              pattern—no successes—and subtract it from 1.
            </p>
          </div>
        </div>
        <div className={styles.controls}>
          <div className={styles.control}>
            <label htmlFor="success-rate">
              Success per trial: {successPercent}%
            </label>
            <input
              id="success-rate"
              type="range"
              min="5"
              max="80"
              step="5"
              value={successPercent}
              onChange={(event) =>
                setSuccessPercent(Number(event.target.value))
              }
            />
          </div>
          <div className={styles.control}>
            <label htmlFor="trial-count">Independent trials: {trials}</label>
            <input
              id="trial-count"
              type="range"
              min="1"
              max="12"
              value={trials}
              onChange={(event) => setTrials(Number(event.target.value))}
            />
          </div>
        </div>
        <div className={styles.status} role="status">
          1 − (1 − {success.toFixed(2)})<sup>{trials}</sup> = 1 −{" "}
          {(1 - success).toFixed(2)}
          <sup>{trials}</sup> = {(atLeastOne * 100).toFixed(1)}%
        </div>
      </div>

      <div className={styles.lab} data-testid="geometric-sum-lab">
        <div className={styles.labHeader}>
          <div>
            <p>Geometric discounted sum</p>
            <h3>Repeated rewards fade by a factor</h3>
            <p>
              A power can encode repeated shrinking. Each later term is the
              previous term multiplied by the same discount factor.
            </p>
          </div>
        </div>
        <div className={styles.controls}>
          <div className={styles.control}>
            <label htmlFor="discount-rate">
              Discount factor: {discount.toFixed(2)}
            </label>
            <input
              id="discount-rate"
              type="range"
              min="20"
              max="95"
              step="5"
              value={discountPercent}
              onChange={(event) =>
                setDiscountPercent(Number(event.target.value))
              }
            />
          </div>
          <div className={styles.control}>
            <label htmlFor="horizon">Terms: {horizon}</label>
            <input
              id="horizon"
              type="range"
              min="1"
              max="10"
              value={horizon}
              onChange={(event) => setHorizon(Number(event.target.value))}
            />
          </div>
        </div>
        <div className={styles.status} role="status">
          {terms.map((term) => term.toFixed(2)).join(" + ")} ={" "}
          {discountedTotal.toFixed(2)}
        </div>
      </div>
    </div>
  );
}

export default function CrashProbabilityL0LearningPage({ experience }: Props) {
  return (
    <ProbabilityCourse
      experience={experience}
      station="l0"
      kicker="Lecture 0 · Mathematical readiness"
      headline="Make the notation obey you."
      introduction="Probability becomes difficult when ratios, subscripts, sums, powers, logs, and shapes all arrive at once. This station slows those moves down and makes every symbol executable—without assuming any AI knowledge."
      heroVisual={<ReadinessHero />}
    >
      <ProbabilitySection
        id="ratios"
        eyebrow="01 · Parts and wholes"
        title="Normalize before you interpret."
        lead="A raw score, count, or weight is not automatically a probability. Normalization divides each part by the total so the resulting shares are non-negative and sum to one."
      >
        <NormalizationBench />
        <div style={{ marginTop: 22 }}>
          <OddsConverter />
        </div>
      </ProbabilitySection>

      <ProbabilitySection
        id="notation"
        eyebrow="02 · Symbol fluency"
        title="Read the instruction hidden in the marks."
        lead="Function arguments, parameters, indices, maxima, sums, and products are compact operations. Decode the operation first; arithmetic comes second."
      >
        <NotationDecoder />
        <div className={styles.grid2} style={{ marginTop: 22 }}>
          <ProbabilityFormula
            label="Weighted average"
            formula={String.raw`\[\bar{x}_w=\frac{\sum_i w_i x_i}{\sum_i w_i}\]`}
          >
            Multiply each value by its weight, add those contributions, then
            divide by the total weight. If the weights already sum to one, the
            denominator is one.
          </ProbabilityFormula>
          <ProbabilityFormula
            label="Algebraic rearrangement"
            formula={String.raw`\[y=ax+b\quad\Longrightarrow\quad x=\frac{y-b}{a}\]`}
          >
            Undo operations in reverse order: subtract the offset, then divide
            by the scale. Keep the same operation on both sides of the equation.
          </ProbabilityFormula>
        </div>
        <ProbabilityCheck
          testId="notation-check"
          title="Max is a value; argmax is a location"
          question="Scores are [1.2, 3.7, 2.9]. What does argmax return?"
          correctIndex={1}
          options={[
            {
              label: "3.7",
              explanation:
                "That is the maximum value, not the position that produced it.",
            },
            {
              label: "The second position",
              explanation:
                "The largest score is 3.7, and argmax returns its index or associated choice.",
            },
            {
              label: "All three scores normalized",
              explanation: "Normalization is a separate operation.",
            },
          ]}
        />
      </ProbabilitySection>

      <ProbabilitySection
        id="nonlinear"
        eyebrow="03 · Powers, roots, exponentials, logs"
        title="Repeated multiplication has a language."
        lead="Powers encode repeated factors, roots undo powers, exponentials grow or shrink multiplicatively, and logarithms ask which exponent produced a number. These moves later make products and tiny probabilities manageable."
      >
        <div className={styles.grid3}>
          {[
            [
              "Power",
              String.raw`\(a^3=a\cdot a\cdot a\)`,
              "Repeat the same factor.",
            ],
            [
              "Root",
              String.raw`\(\sqrt{a^2}=|a|\)`,
              "Undo a square while respecting sign.",
            ],
            [
              "Exponential",
              String.raw`\(e^x\)`,
              "Turn additive score differences into positive ratios.",
            ],
            [
              "Logarithm",
              String.raw`\(\log(ab)=\log a+\log b\)`,
              "Turn products into sums.",
            ],
            [
              "Euler’s number",
              String.raw`\(e\approx2.718\)`,
              "The natural base for continuous growth and natural logs.",
            ],
            [
              "Geometric factor",
              String.raw`\(1+\gamma+\gamma^2+\cdots\)`,
              "Repeat a fixed shrinkage through time.",
            ],
          ].map(([title, formula, explanation]) => (
            <div className={styles.panel} key={title}>
              <span className={styles.tag}>{title}</span>
              <h3 style={{ marginTop: 12 }}>
                <InlineProbabilityMath text={formula} />
              </h3>
              <p>{explanation}</p>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 22 }}>
          <PowersAndLogsLab />
        </div>
      </ProbabilitySection>

      <ProbabilitySection
        id="shapes"
        eyebrow="04 · Shapes and distributions"
        title="Track dimensions and spread before plugging in numbers."
        lead="Shape arithmetic prevents impossible operations. Distribution notation then tells you which numerical quantity is uncertain and which parameters describe its center and spread."
      >
        <ShapeBench />
        <div className={styles.grid2} style={{ marginTop: 22 }}>
          <ProbabilityFormula
            label="Gaussian notation"
            formula={String.raw`\[X\sim\mathcal{N}(\mu,\sigma^2)\]`}
          >
            Read: “X follows a normal distribution with mean μ and variance σ².”
            The standard deviation is σ, the square root of the variance.
          </ProbabilityFormula>
          <ProbabilityInsight title="Parameter is not observation">
            <p>
              The mean and variance describe a distribution. A realized value of
              X is one observation drawn from that distribution. Do not confuse
              the center of the model with the value that must occur.
            </p>
          </ProbabilityInsight>
        </div>
      </ProbabilitySection>

      <ProbabilityQuizLaunch
        experience={experience}
        recap={[
          "Ratios compare a part with a whole; odds compare success with failure.",
          "Normalization divides every non-negative weight by the total.",
          "Subscripts index values; max returns a value and argmax returns its location.",
          "Sigma adds terms, capital pi multiplies them, and logs turn products into sums.",
          "Matrix products keep the outer dimensions when the inner dimensions match.",
          "Gaussian notation separates a random variable from its mean, variance, and standard deviation.",
        ]}
      />
    </ProbabilityCourse>
  );
}
