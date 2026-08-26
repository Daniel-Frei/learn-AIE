"use client";

import { useMemo, useState } from "react";
import { Gauge, Microscope, RefreshCcw } from "lucide-react";
import type { LearningExperience } from "../../../lib/learning";
import {
  crossEntropy,
  entropyBits,
  negativeLogLikelihood,
  softmax,
  trajectoryProbability,
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

type Props = { experience: LearningExperience };
const LABELS = ["cat", "dog", "car"] as const;

function ProbabilityBars({
  probabilities,
  observed,
}: {
  probabilities: readonly number[];
  observed?: number;
}) {
  return (
    <div className={styles.bars}>
      {probabilities.map((probability, index) => (
        <div className={styles.barRow} key={LABELS[index]}>
          <span>
            {LABELS[index]}
            {observed === index ? " · observed" : ""}
          </span>
          <div className={styles.barTrack}>
            <span
              className={styles.barFill}
              style={{
                width: `${Math.max(probability * 100, 1).toFixed(4)}%`,
              }}
            />
          </div>
          <strong>{(probability * 100).toFixed(1)}%</strong>
        </div>
      ))}
    </div>
  );
}

function HeroTrainingMicroscope() {
  const probabilities = softmax([2.1, 1.3, -0.5]);
  return (
    <div
      className={styles.lab}
      aria-label="A microscope showing the training path from model scores to loss"
    >
      <div className={styles.labHeader}>
        <div>
          <p>One training example</p>
          <h3>The observed word is “cat”</h3>
        </div>
        <Microscope aria-hidden="true" size={35} />
      </div>
      <div className={styles.formulaTrail}>
        <div className={styles.formulaStep}>
          <strong>scores</strong>
          <span>2.1 · 1.3 · −0.5</span>
        </div>
        <div className={styles.formulaStep}>
          <strong>softmax</strong>
          <span>
            {probabilities.map((value) => value.toFixed(2)).join(" · ")}
          </span>
        </div>
        <div className={styles.formulaStep}>
          <strong>look up cat</strong>
          <span>{probabilities[0]?.toFixed(3)}</span>
        </div>
        <div className={styles.formulaStep}>
          <strong>−log</strong>
          <span>
            {negativeLogLikelihood(probabilities[0] ?? 0).toFixed(3)} loss
          </span>
        </div>
      </div>
      <ProbabilityInsight title="Training has a concrete job" tone="success">
        <p>
          Move probability mass toward what actually happened, example after
          example.
        </p>
      </ProbabilityInsight>
    </div>
  );
}

function SoftmaxMicroscopeLab() {
  const [logits, setLogits] = useState([2.1, 1.3, -0.5]);
  const [temperature, setTemperature] = useState(1);
  const [observed, setObserved] = useState(0);
  const probabilities = useMemo(
    () => softmax(logits, temperature),
    [logits, temperature],
  );
  const observedProbability = probabilities[observed] ?? 0;
  const loss = negativeLogLikelihood(observedProbability);
  const entropy = entropyBits(probabilities);
  const maximum = Math.max(...logits.map((value) => value / temperature));
  const shiftedExponentials = logits.map((value) =>
    Math.exp(value / temperature - maximum),
  );
  const denominator = shiftedExponentials.reduce(
    (sum, value) => sum + value,
    0,
  );
  const setLogit = (index: number, value: number) =>
    setLogits((current) =>
      current.map((logit, candidate) => (candidate === index ? value : logit)),
    );

  return (
    <div className={styles.lab} data-testid="softmax-microscope">
      <div className={styles.labHeader}>
        <div>
          <p>Score → distribution</p>
          <h3>Move one logit and watch every probability respond</h3>
        </div>
        <Gauge aria-hidden="true" size={31} />
      </div>
      <div className={styles.controls}>
        {LABELS.map((label, index) => (
          <label className={styles.control} key={label}>
            <span>
              {label} logit: {logits[index]?.toFixed(1)}
            </span>
            <input
              aria-label={`${label} logit`}
              type="range"
              min="-3"
              max="4"
              step="0.1"
              value={logits[index]}
              onChange={(event) => setLogit(index, Number(event.target.value))}
            />
          </label>
        ))}
        <label className={styles.control}>
          <span>Temperature: {temperature.toFixed(1)}</span>
          <input
            aria-label="Softmax temperature"
            type="range"
            min="0.3"
            max="2.5"
            step="0.1"
            value={temperature}
            onChange={(event) => setTemperature(Number(event.target.value))}
          />
        </label>
      </div>
      <div className={styles.buttonRow}>
        <span className={styles.status}>Observed class:</span>
        {LABELS.map((label, index) => (
          <button
            type="button"
            key={label}
            onClick={() => setObserved(index)}
            className={observed === index ? styles.buttonActive : styles.button}
          >
            {label}
          </button>
        ))}
      </div>
      <ProbabilityBars probabilities={probabilities} observed={observed} />
      <div className={styles.metricGrid}>
        <ProbabilityMetric
          label="sum"
          value={probabilities
            .reduce((sum, value) => sum + value, 0)
            .toFixed(3)}
          detail="a valid distribution"
        />
        <ProbabilityMetric
          label="observed probability"
          value={`${(observedProbability * 100).toFixed(1)}%`}
          detail={`look up ${LABELS[observed]}`}
        />
        <ProbabilityMetric
          label="NLL"
          value={loss.toFixed(3)}
          detail="lower is better"
        />
        <ProbabilityMetric
          label="entropy"
          value={`${entropy.toFixed(2)} bits`}
          detail="distribution-wide uncertainty"
        />
      </div>
      <ProbabilityFormula
        label="Softmax with the arithmetic exposed"
        formula={String.raw`\[p_i=\frac{e^{z_i/T-m}}{\sum_j e^{z_j/T-m}},\quad m=\max_j(z_j/T)\qquad \text{denominator}=${denominator.toFixed(3)}\]`}
      >
        Subtracting the maximum leaves every ratio unchanged but prevents huge
        exponentials. Softmax cares about score differences: adding the same
        constant to every logit does not change the probabilities.
      </ProbabilityFormula>
    </div>
  );
}

function LikelihoodLedger() {
  const [lastProbability, setLastProbability] = useState(40);
  const stepProbabilities = [0.7, 0.6, 0.8, lastProbability / 100];
  const likelihood = trajectoryProbability(stepProbabilities);
  const logLikelihood = stepProbabilities.reduce(
    (sum, value) => sum + Math.log(value),
    0,
  );
  return (
    <div className={styles.lab} data-testid="likelihood-ledger">
      <div className={styles.labHeader}>
        <div>
          <p>Observed four-token sequence</p>
          <h3>
            Likelihood multiplies the probabilities assigned before seeing each
            answer
          </h3>
        </div>
        <Microscope aria-hidden="true" size={30} />
      </div>
      <label className={styles.control}>
        <span>
          Probability assigned to the fourth observed token: {lastProbability}%
        </span>
        <input
          aria-label="Fourth observed-token probability"
          type="range"
          min="5"
          max="95"
          step="5"
          value={lastProbability}
          onChange={(event) => setLastProbability(Number(event.target.value))}
        />
      </label>
      <div className={styles.sequenceStrip}>
        {stepProbabilities.map((probability, index) => (
          <div
            className={styles.sequenceItem}
            data-active={index === 3}
            key={index}
          >
            <strong>token {index + 1}</strong>
            <span>{probability.toFixed(2)}</span>
          </div>
        ))}
      </div>
      <div className={styles.metricGrid}>
        <ProbabilityMetric
          label="product likelihood"
          value={likelihood.toFixed(4)}
          detail="joint probability of the sequence"
        />
        <ProbabilityMetric
          label="log-likelihood"
          value={logLikelihood.toFixed(3)}
          detail="same ranking, additive scale"
        />
        <ProbabilityMetric
          label="sequence NLL"
          value={(-logLikelihood).toFixed(3)}
          detail="sum of token losses"
        />
        <ProbabilityMetric
          label="mean token loss"
          value={(-logLikelihood / 4).toFixed(3)}
          detail="comparable across lengths"
        />
      </div>
      <div className={styles.grid2}>
        <ProbabilityFormula
          label="Chain-rule likelihood"
          formula={String.raw`\[P(x_{1:4})=\prod_{t=1}^{4}P(x_t\mid x_{<t})=${stepProbabilities.map((value) => value.toFixed(2)).join("\\times")}=${likelihood.toFixed(4)}\]`}
        />
        <ProbabilityFormula
          label="Logs turn products into sums"
          formula={String.raw`\[\log P(x_{1:4})=\sum_{t=1}^{4}\log P(x_t\mid x_{<t})=${logLikelihood.toFixed(3)}\]`}
        />
      </div>
      <ProbabilityInsight
        title="Likelihood is a function of the model"
        tone="warning"
      >
        <p>
          The observed sequence is now fixed. We compare parameter settings by
          asking which one would have made that same data more probable. That is
          why likelihood is not a probability distribution over parameters.
        </p>
      </ProbabilityInsight>
    </div>
  );
}

function CrossEntropyWorkbench() {
  const [correctPrediction, setCorrectPrediction] = useState(70);
  const [targetMode, setTargetMode] = useState<"hard" | "soft">("hard");
  const prediction = [
    correctPrediction / 100,
    ((1 - correctPrediction / 100) * 2) / 3,
    (1 - correctPrediction / 100) / 3,
  ];
  const target = targetMode === "hard" ? [1, 0, 0] : [0.8, 0.15, 0.05];
  const loss = crossEntropy(target, prediction);
  const hardNll = negativeLogLikelihood(prediction[0] ?? 0);
  return (
    <div className={styles.lab} data-testid="cross-entropy-workbench">
      <div className={styles.labHeader}>
        <div>
          <p>Target distribution vs prediction</p>
          <h3>Cross-entropy reads probability wherever the target has mass</h3>
        </div>
        <RefreshCcw aria-hidden="true" size={30} />
      </div>
      <div className={styles.buttonRow}>
        <button
          type="button"
          className={
            targetMode === "hard" ? styles.buttonActive : styles.button
          }
          onClick={() => setTargetMode("hard")}
        >
          One-hot target
        </button>
        <button
          type="button"
          className={
            targetMode === "soft" ? styles.buttonActive : styles.button
          }
          onClick={() => setTargetMode("soft")}
        >
          Soft target
        </button>
      </div>
      <label className={styles.control}>
        <span>Model probability on cat: {correctPrediction}%</span>
        <input
          aria-label="Probability predicted for cat"
          type="range"
          min="5"
          max="95"
          step="5"
          value={correctPrediction}
          onChange={(event) => setCorrectPrediction(Number(event.target.value))}
        />
      </label>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col">Class</th>
              <th scope="col">Target</th>
              <th scope="col">Prediction</th>
              <th scope="col">Loss term</th>
            </tr>
          </thead>
          <tbody>
            {LABELS.map((label, index) => (
              <tr key={label}>
                <th scope="row">{label}</th>
                <td>{target[index]?.toFixed(2)}</td>
                <td>{prediction[index]?.toFixed(3)}</td>
                <td>
                  {target[index] === 0
                    ? "0"
                    : `${(-(target[index] ?? 0) * Math.log(prediction[index] ?? 1)).toFixed(3)}`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className={styles.metricGrid}>
        <ProbabilityMetric
          label="cross-entropy"
          value={loss.toFixed(3)}
          detail="sum of target-weighted surprises"
        />
        <ProbabilityMetric
          label="one-hot NLL"
          value={hardNll.toFixed(3)}
          detail="equal to CE for a hard cat target"
        />
      </div>
      <ProbabilityFormula
        label="Cross-entropy"
        formula={String.raw`\[H(q,p)=-\sum_i q_i\log p_i\]`}
      >
        With a one-hot target, every term except the correct class disappears,
        leaving{" "}
        <InlineProbabilityMath text={String.raw`\(-\log p_{correct}\)`} />. Soft
        targets preserve graded uncertainty or label smoothing.
      </ProbabilityFormula>
    </div>
  );
}

function TrainingStepLab() {
  const [catLogit, setCatLogit] = useState(0.5);
  const [learningRate, setLearningRate] = useState(0.5);
  const probabilities = softmax([catLogit, 0, 0]);
  const catProbability = probabilities[0] ?? 0;
  const gradient = catProbability - 1;
  const nextLogit = catLogit - learningRate * gradient;
  const nextProbability = softmax([nextLogit, 0, 0])[0] ?? 0;
  return (
    <div className={styles.lab} data-testid="training-step-lab">
      <div className={styles.labHeader}>
        <div>
          <p>One transparent gradient step</p>
          <h3>Loss supplies a direction, not just a score</h3>
        </div>
        <RefreshCcw aria-hidden="true" size={30} />
      </div>
      <div className={styles.controls}>
        <label className={styles.control}>
          <span>Current cat logit: {catLogit.toFixed(1)}</span>
          <input
            aria-label="Current cat logit"
            type="range"
            min="-2"
            max="3"
            step="0.1"
            value={catLogit}
            onChange={(event) => setCatLogit(Number(event.target.value))}
          />
        </label>
        <label className={styles.control}>
          <span>Learning rate: {learningRate.toFixed(1)}</span>
          <input
            aria-label="Learning rate"
            type="range"
            min="0.1"
            max="1"
            step="0.1"
            value={learningRate}
            onChange={(event) => setLearningRate(Number(event.target.value))}
          />
        </label>
      </div>
      <div className={styles.formulaTrail}>
        <div className={styles.formulaStep}>
          <strong>forward</strong>
          <span>cat probability {catProbability.toFixed(3)}</span>
        </div>
        <div className={styles.formulaStep}>
          <strong>loss</strong>
          <span>{negativeLogLikelihood(catProbability).toFixed(3)}</span>
        </div>
        <div className={styles.formulaStep}>
          <strong>gradient</strong>
          <span>p − y = {gradient.toFixed(3)}</span>
        </div>
        <div className={styles.formulaStep}>
          <strong>update</strong>
          <span>
            {catLogit.toFixed(2)} − {learningRate.toFixed(1)}(
            {gradient.toFixed(3)}) = {nextLogit.toFixed(3)}
          </span>
        </div>
        <div className={styles.formulaStep}>
          <strong>next forward</strong>
          <span>cat probability {nextProbability.toFixed(3)}</span>
        </div>
      </div>
      <ProbabilityInsight title="A batch estimates an expectation">
        <p>
          Training usually averages this loss across a minibatch sampled from
          the data. The batch gradient is a noisy estimate of the population
          gradient; more data reduces noise but costs more computation per step.
        </p>
      </ProbabilityInsight>
    </div>
  );
}

function EntropyBench() {
  const [topProbability, setTopProbability] = useState(34);
  const top = topProbability / 100;
  const probabilities = [top, (1 - top) / 2, (1 - top) / 2];
  const entropy = entropyBits(probabilities);
  return (
    <div className={styles.lab} data-testid="entropy-bench">
      <div className={styles.labHeader}>
        <div>
          <p>Distribution-wide uncertainty</p>
          <h3>Confidence and correctness are different axes</h3>
        </div>
        <Gauge aria-hidden="true" size={30} />
      </div>
      <label className={styles.control}>
        <span>Largest class probability: {topProbability}%</span>
        <input
          aria-label="Largest class probability"
          type="range"
          min="34"
          max="98"
          value={topProbability}
          onChange={(event) => setTopProbability(Number(event.target.value))}
        />
      </label>
      <ProbabilityBars probabilities={probabilities} />
      <div className={styles.metricGrid}>
        <ProbabilityMetric
          label="entropy"
          value={`${entropy.toFixed(3)} bits`}
          detail="0 means fully concentrated"
        />
        <ProbabilityMetric
          label="maximum here"
          value={`${Math.log2(3).toFixed(3)} bits`}
          detail="three equally likely classes"
        />
      </div>
      <ProbabilityInsight
        title="Low entropy is not evidence of truth"
        tone="warning"
      >
        <p>
          A model can be confidently wrong. Entropy describes concentration in
          one prediction; calibration compares confidence with long-run
          frequency, and accuracy checks which class won. A trustworthy system
          needs all three views.
        </p>
      </ProbabilityInsight>
    </div>
  );
}

export default function CrashProbabilityL3LearningPage({ experience }: Props) {
  return (
    <ProbabilityCourse
      experience={experience}
      station="l3"
      kicker="Station L3 · Likelihood, loss, and learning"
      headline="Training asks one relentless question: did you make what happened probable?"
      introduction="A network emits scores. Softmax turns them into a probability distribution. The observed answer selects one part of that distribution, a logarithm turns its probability into a usable loss, and gradients push the next prediction in a better direction."
      heroVisual={<HeroTrainingMicroscope />}
    >
      <ProbabilitySection
        id="softmax"
        eyebrow="01 · Scores become probabilities"
        title="A logit is evidence on a relative scale, not confidence."
        lead="Softmax exponentiates score differences and normalizes them into positive numbers that sum to one. The microscope keeps every step visible so probability never appears by magic."
      >
        <SoftmaxMicroscopeLab />
        <ProbabilityCheck
          testId="l3-softmax-check"
          title="Find the invariant"
          question="What happens if you add 100 to every logit before applying softmax?"
          options={[
            {
              label: "Every probability rises",
              explanation:
                "Probabilities must still sum to one, so they cannot all rise together.",
            },
            {
              label: "The distribution is unchanged",
              explanation:
                "Softmax depends on differences; the common factor in numerator and denominator cancels.",
            },
            {
              label: "The largest class becomes certain",
              explanation:
                "A shared shift changes no score gap, so it cannot sharpen the distribution.",
            },
          ]}
          correctIndex={1}
        />
      </ProbabilitySection>

      <ProbabilitySection
        id="likelihood"
        eyebrow="02 · Score the observed data"
        title="Likelihood keeps the data fixed and turns the model knob."
        lead="For each observed token, retrieve the probability the model assigned to that token given its context. Multiply across the sequence—or, in practice, add the logs."
      >
        <LikelihoodLedger />
      </ProbabilitySection>

      <ProbabilitySection
        id="loss"
        eyebrow="03 · From likelihood to loss"
        title="The negative sign turns a goal to maximize into a quantity to minimize."
        lead="A probability near one produces almost no surprise. A tiny probability produces a large penalty. Cross-entropy extends that idea from a single correct label to an entire target distribution."
      >
        <CrossEntropyWorkbench />
        <div className={styles.grid2} style={{ marginTop: 22 }}>
          <ProbabilityFormula
            label="Negative log-likelihood"
            formula={String.raw`\[\mathcal{L}_{NLL}=-\log P_\theta(y\mid x)\]`}
          />
          <ProbabilityInsight title="The logarithm gives useful geometry">
            <p>
              Logs convert products into sums, prevent long products from
              underflowing to zero, and penalize confident mistakes strongly.
              They also make independent example contributions additive.
            </p>
          </ProbabilityInsight>
        </div>
      </ProbabilitySection>

      <ProbabilitySection
        id="optimization"
        eyebrow="04 · Close the learning loop"
        title="The loss is valuable because it tells parameters how to move."
        lead="For softmax plus cross-entropy, the gradient with respect to a class logit has an unusually readable form: predicted probability minus target probability."
      >
        <TrainingStepLab />
      </ProbabilitySection>

      <ProbabilitySection
        id="entropy"
        eyebrow="05 · Uncertainty across the whole distribution"
        title="Entropy asks how spread out the alternatives are."
        lead="NLL examines the observed answer. Entropy examines the full prediction without knowing which answer will occur. Uniform probability is maximally uncertain; concentrated probability is low entropy."
      >
        <EntropyBench />
        <div className={styles.grid2} style={{ marginTop: 22 }}>
          <ProbabilityFormula
            label="Entropy"
            formula={String.raw`\[H(P)=-\sum_i p_i\log_2 p_i\]`}
          >
            Each term is probability × surprise. The weighted average reports
            expected surprise in bits.
          </ProbabilityFormula>
          <ProbabilityInsight title="Cross-entropy = target entropy + mismatch">
            <p>
              <InlineProbabilityMath
                text={String.raw`\(H(q,p)=H(q)+D_{KL}(q\parallel p)\)`}
              />
              . The target’s own uncertainty cannot be removed; training reduces
              the extra cost created when prediction p disagrees with target q.
            </p>
          </ProbabilityInsight>
        </div>
      </ProbabilitySection>

      <ProbabilityQuizLaunch
        experience={experience}
        recap={[
          "Logits are relative scores; softmax turns score differences into a normalized distribution.",
          "Likelihood evaluates fixed observed data under competing model settings.",
          "Sequence likelihood multiplies conditional token probabilities; log-likelihood adds their logs.",
          "Negative log-likelihood rewards probability on the observed answer and heavily penalizes confident misses.",
          "Cross-entropy handles one-hot and soft target distributions and drives classifier and LLM training.",
          "Entropy measures distribution-wide uncertainty; confidence, calibration, and correctness are distinct.",
        ]}
      />
    </ProbabilityCourse>
  );
}
