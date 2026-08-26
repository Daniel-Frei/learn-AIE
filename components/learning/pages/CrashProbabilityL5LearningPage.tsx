"use client";

import { useMemo, useState } from "react";
import { Aperture, Dices, Sparkles } from "lucide-react";
import type { LearningExperience } from "../../../lib/learning";
import {
  bayesPosterior,
  entropyBits,
  negativeLogLikelihood,
  softmax,
  topKDistribution,
  topPDistribution,
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
const TOKENS = ["observatory", "garden", "answer", "storm", "silence"] as const;
const BASE_LOGITS = [2.6, 2, 1.5, 0.7, 0];

function HeroGenerationForge() {
  return (
    <div
      className={styles.lab}
      aria-label="A generation forge showing repeated conditional sampling"
    >
      <div className={styles.labHeader}>
        <div>
          <p>Generation forge</p>
          <h3>One distribution becomes one choice</h3>
        </div>
        <Sparkles aria-hidden="true" size={35} />
      </div>
      <div className={styles.formulaTrail}>
        <div className={styles.formulaStep}>
          <strong>condition</strong>
          <span>“Beyond the hill stood an…”</span>
        </div>
        <div className={styles.formulaStep}>
          <strong>distribution</strong>
          <span>many plausible next tokens</span>
        </div>
        <div className={styles.formulaStep}>
          <strong>sample</strong>
          <span>one token becomes real</span>
        </div>
        <div className={styles.formulaStep}>
          <strong>append</strong>
          <span>the new token changes the next distribution</span>
        </div>
      </div>
    </div>
  );
}

function drawFromDistribution(
  probabilities: readonly number[],
  quantile: number,
) {
  let cumulative = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    cumulative += probabilities[index] ?? 0;
    if (quantile <= cumulative) return index;
  }
  return probabilities.length - 1;
}

function SamplingForgeLab() {
  const [temperature, setTemperature] = useState(1);
  const [mode, setMode] = useState<"greedy" | "sample" | "top-k" | "top-p">(
    "sample",
  );
  const [topK, setTopK] = useState(3);
  const [topP, setTopP] = useState(0.8);
  const [draw, setDraw] = useState(42);
  const base = useMemo(() => softmax(BASE_LOGITS, temperature), [temperature]);
  const used = useMemo(() => {
    if (mode === "greedy")
      return base.map((_, index) =>
        index === base.indexOf(Math.max(...base)) ? 1 : 0,
      );
    if (mode === "top-k") return topKDistribution(base, topK);
    if (mode === "top-p") return topPDistribution(base, topP);
    return base;
  }, [base, mode, topK, topP]);
  const selectedIndex = drawFromDistribution(used, draw / 100);
  const selected = TOKENS[selectedIndex];
  const entropy = entropyBits(used);

  return (
    <div className={styles.lab} data-testid="sampling-forge-lab">
      <div className={styles.labHeader}>
        <div>
          <p>Next-token distribution</p>
          <h3>Shape the menu, then use one random draw</h3>
        </div>
        <Dices aria-hidden="true" size={31} />
      </div>
      <div className={styles.controls}>
        <label className={styles.control}>
          <span>Temperature: {temperature.toFixed(1)}</span>
          <input
            aria-label="Sampling temperature"
            type="range"
            min="0.3"
            max="2.5"
            step="0.1"
            value={temperature}
            onChange={(event) => setTemperature(Number(event.target.value))}
          />
        </label>
        <label className={styles.control}>
          <span>Draw position: {draw}%</span>
          <input
            aria-label="Random draw position"
            type="range"
            min="1"
            max="99"
            value={draw}
            onChange={(event) => setDraw(Number(event.target.value))}
          />
        </label>
        {mode === "top-k" && (
          <label className={styles.control}>
            <span>k: {topK} tokens</span>
            <input
              aria-label="Top k tokens"
              type="range"
              min="1"
              max={TOKENS.length}
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
            />
          </label>
        )}
        {mode === "top-p" && (
          <label className={styles.control}>
            <span>p: {topP.toFixed(2)} cumulative mass</span>
            <input
              aria-label="Top p threshold"
              type="range"
              min="0.4"
              max="1"
              step="0.05"
              value={topP}
              onChange={(event) => setTopP(Number(event.target.value))}
            />
          </label>
        )}
      </div>
      <div className={styles.buttonRow}>
        {(["greedy", "sample", "top-k", "top-p"] as const).map((option) => (
          <button
            type="button"
            className={mode === option ? styles.buttonActive : styles.button}
            onClick={() => setMode(option)}
            key={option}
          >
            {option}
          </button>
        ))}
      </div>
      <div className={styles.bars}>
        {used.map((probability, index) => (
          <div className={styles.barRow} key={TOKENS[index]}>
            <span>
              {TOKENS[index]}
              {selectedIndex === index ? " · drawn" : ""}
            </span>
            <div className={styles.barTrack}>
              <span
                className={styles.barFill}
                style={{
                  width: `${Math.max(
                    probability * 100,
                    probability > 0 ? 1 : 0,
                  ).toFixed(4)}%`,
                }}
              />
            </div>
            <strong>{(probability * 100).toFixed(1)}%</strong>
          </div>
        ))}
      </div>
      <div className={styles.metricGrid}>
        <ProbabilityMetric
          label="selected token"
          value={selected ?? "—"}
          detail={`draw at ${(draw / 100).toFixed(2)}`}
        />
        <ProbabilityMetric
          label="entropy"
          value={`${entropy.toFixed(2)} bits`}
          detail="diversity in the used distribution"
        />
        <ProbabilityMetric
          label="expected in 1,000"
          value={String(Math.round((used[selectedIndex] ?? 0) * 1000))}
          detail={`draws yielding “${selected}”`}
        />
      </div>
      <ProbabilityFormula
        label="Temperature rescales logits before softmax"
        formula={String.raw`\[P_T(x_i)=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}\]`}
      >
        Lower T magnifies score gaps; higher T compresses them. Top-k keeps a
        fixed number of candidates. Top-p keeps the smallest ranked set whose
        cumulative mass reaches p, so its menu size adapts to uncertainty.
      </ProbabilityFormula>
    </div>
  );
}

function RepeatedConditioningLab() {
  const [finalTokenProbability, setFinalTokenProbability] = useState(55);
  const tokens = ["the", "night", "felt", "electric"];
  const probabilities = [0.72, 0.48, 0.63, finalTokenProbability / 100];
  const sequenceProbability = trajectoryProbability(probabilities);
  const nll = probabilities.reduce(
    (sum, value) => sum + negativeLogLikelihood(value),
    0,
  );
  return (
    <div className={styles.lab} data-testid="repeated-conditioning-lab">
      <div className={styles.labHeader}>
        <div>
          <p>Autoregressive sequence</p>
          <h3>Generation is not one giant draw from a list of sentences</h3>
        </div>
        <Sparkles aria-hidden="true" size={31} />
      </div>
      <label className={styles.control}>
        <span>
          Probability of final observed token: {finalTokenProbability}%
        </span>
        <input
          aria-label="Final token probability"
          type="range"
          min="5"
          max="95"
          step="5"
          value={finalTokenProbability}
          onChange={(event) =>
            setFinalTokenProbability(Number(event.target.value))
          }
        />
      </label>
      <div className={styles.sequenceStrip}>
        {tokens.map((token, index) => (
          <div
            className={styles.sequenceItem}
            data-active={index === tokens.length - 1}
            key={token}
          >
            <strong>{token}</strong>
            <span>{probabilities[index]?.toFixed(2)}</span>
          </div>
        ))}
      </div>
      <div className={styles.metricGrid}>
        <ProbabilityMetric
          label="sequence probability"
          value={sequenceProbability.toFixed(4)}
          detail="product of conditional choices"
        />
        <ProbabilityMetric
          label="sequence NLL"
          value={nll.toFixed(3)}
          detail="sum of token surprises"
        />
      </div>
      <ProbabilityFormula
        label="Chain rule for generation"
        formula={String.raw`\[P(x_{1:n})=\prod_{t=1}^{n}P(x_t\mid x_{<t})\]`}
      >
        After each sample, the context changes. A low-probability early token
        can move generation into a region with very different later
        possibilities.
      </ProbabilityFormula>
    </div>
  );
}

function LatentVariableLab() {
  const [rainPrior, setRainPrior] = useState(30);
  const [umbrellaGivenRain, setUmbrellaGivenRain] = useState(90);
  const [umbrellaWithoutRain, setUmbrellaWithoutRain] = useState(20);
  const prior = rainPrior / 100;
  const likelihood = umbrellaGivenRain / 100;
  const falsePositiveRate = umbrellaWithoutRain / 100;
  const umbrella = prior * likelihood + (1 - prior) * falsePositiveRate;
  const posterior = bayesPosterior({ prior, likelihood, falsePositiveRate });
  return (
    <div className={styles.lab} data-testid="latent-variable-lab">
      <div className={styles.labHeader}>
        <div>
          <p>Hidden weather, observed umbrella</p>
          <h3>
            A latent variable explains visible patterns without being directly
            observed
          </h3>
        </div>
        <Aperture aria-hidden="true" size={31} />
      </div>
      <div className={styles.controls}>
        <label className={styles.control}>
          <span>P(rain): {rainPrior}%</span>
          <input
            aria-label="Rain prior"
            type="range"
            min="5"
            max="80"
            step="5"
            value={rainPrior}
            onChange={(event) => setRainPrior(Number(event.target.value))}
          />
        </label>
        <label className={styles.control}>
          <span>P(umbrella | rain): {umbrellaGivenRain}%</span>
          <input
            aria-label="Umbrella given rain"
            type="range"
            min="50"
            max="100"
            step="5"
            value={umbrellaGivenRain}
            onChange={(event) =>
              setUmbrellaGivenRain(Number(event.target.value))
            }
          />
        </label>
        <label className={styles.control}>
          <span>P(umbrella | no rain): {umbrellaWithoutRain}%</span>
          <input
            aria-label="Umbrella without rain"
            type="range"
            min="0"
            max="60"
            step="5"
            value={umbrellaWithoutRain}
            onChange={(event) =>
              setUmbrellaWithoutRain(Number(event.target.value))
            }
          />
        </label>
      </div>
      <div className={styles.pathTree}>
        <div className={styles.pathRow}>
          <span className={styles.pathNode}>rain · {rainPrior}%</span>
          <span className={styles.pathArrow}>→</span>
          <span className={styles.pathNode}>
            umbrella · {umbrellaGivenRain}%
          </span>
          <span className={styles.tag}>
            joint {(prior * likelihood * 100).toFixed(1)}%
          </span>
        </div>
        <div className={styles.pathRow}>
          <span className={styles.pathNode}>no rain · {100 - rainPrior}%</span>
          <span className={styles.pathArrow}>→</span>
          <span className={styles.pathNode}>
            umbrella · {umbrellaWithoutRain}%
          </span>
          <span className={styles.tag}>
            joint {((1 - prior) * falsePositiveRate * 100).toFixed(1)}%
          </span>
        </div>
      </div>
      <div className={styles.metricGrid}>
        <ProbabilityMetric
          label="umbrella probability"
          value={`${(umbrella * 100).toFixed(1)}%`}
          detail="marginalize hidden weather"
        />
        <ProbabilityMetric
          label="rain after umbrella"
          value={`${(posterior * 100).toFixed(1)}%`}
          detail="infer the hidden cause"
        />
      </div>
      <div className={styles.grid2}>
        <ProbabilityFormula
          label="Generate an observation"
          formula={String.raw`\[P(x)=\sum_z P(x\mid z)P(z)\]`}
        >
          Sum over every hidden route that could have produced x.
        </ProbabilityFormula>
        <ProbabilityFormula
          label="Infer a latent cause"
          formula={String.raw`\[P(z\mid x)=\frac{P(x\mid z)P(z)}{\sum_{z'}P(x\mid z')P(z')}\]`}
        >
          Condition on the observation and renormalize the latent alternatives.
        </ProbabilityFormula>
      </div>
      <ProbabilityInsight
        title="A latent coordinate is not automatically a human concept"
        tone="warning"
      >
        <p>
          A model may encode useful hidden factors without assigning one axis
          cleanly to “weather,” “style,” or “sentiment.” Interpretability
          requires evidence from interventions or reliable probes, not a
          suggestive visualization alone.
        </p>
      </ProbabilityInsight>
    </div>
  );
}

function gaussianDensity(
  value: number,
  mean: number,
  standardDeviation: number,
) {
  return (
    Math.exp(-0.5 * ((value - mean) / standardDeviation) ** 2) /
    (standardDeviation * Math.sqrt(2 * Math.PI))
  );
}

function GaussianNoiseLab() {
  const [mean, setMean] = useState(0);
  const [standardDeviation, setStandardDeviation] = useState(1);
  const points = Array.from({ length: 41 }, (_, index) => -4 + index * 0.2);
  const densities = points.map((value) =>
    gaussianDensity(value, mean, standardDeviation),
  );
  const maximum = Math.max(...densities);
  return (
    <div className={styles.lab} data-testid="gaussian-noise-lab">
      <div className={styles.labHeader}>
        <div>
          <p>Continuous noise distribution</p>
          <h3>Mean sets the center; standard deviation sets the noise scale</h3>
        </div>
        <Aperture aria-hidden="true" size={31} />
      </div>
      <div className={styles.controls}>
        <label className={styles.control}>
          <span>Mean μ: {mean.toFixed(1)}</span>
          <input
            aria-label="Gaussian mean"
            type="range"
            min="-2"
            max="2"
            step="0.1"
            value={mean}
            onChange={(event) => setMean(Number(event.target.value))}
          />
        </label>
        <label className={styles.control}>
          <span>Standard deviation σ: {standardDeviation.toFixed(1)}</span>
          <input
            aria-label="Gaussian standard deviation"
            type="range"
            min="0.4"
            max="2"
            step="0.1"
            value={standardDeviation}
            onChange={(event) =>
              setStandardDeviation(Number(event.target.value))
            }
          />
        </label>
      </div>
      <div
        className={styles.densityPlot}
        aria-label="Gaussian probability density"
      >
        <div className={styles.densityBars}>
          {densities.map((density, index) => (
            <span
              key={points[index]}
              className={styles.densityBar}
              style={{ height: `${((density / maximum) * 90).toFixed(4)}%` }}
            />
          ))}
        </div>
        <div className={styles.densityAxis}>
          <span>−4</span>
          <span>0</span>
          <span>+4</span>
        </div>
      </div>
      <ProbabilityFormula
        label="Gaussian density"
        formula={String.raw`\[p(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)\]`}
      >
        This curve is a density, so probability is area across an interval—not
        the height at one exact real number. Diffusion commonly uses independent
        standard-normal noise with mean 0 and variance 1.
      </ProbabilityFormula>
    </div>
  );
}

function DiffusionLab() {
  const [noiseStep, setNoiseStep] = useState(45);
  const [seed, setSeed] = useState(2);
  const alphaBar = 1 - noiseStep / 100;
  const cells = useMemo(
    () =>
      Array.from({ length: 64 }, (_, index) => {
        const row = Math.floor(index / 8);
        const column = index % 8;
        const distance = Math.abs(row - 3.5) + Math.abs(column - 3.5);
        const clean = distance < 3.2 ? 1 : -1;
        const noise = Math.sin((index + 1) * (seed * 12.9898 + 0.78)) * 0.95;
        return Math.sqrt(alphaBar) * clean + Math.sqrt(1 - alphaBar) * noise;
      }),
    [alphaBar, seed],
  );
  return (
    <div className={styles.lab} data-testid="diffusion-lab">
      <div className={styles.labHeader}>
        <div>
          <p>Forward diffusion</p>
          <h3>Mix a clean sample with a known amount of Gaussian noise</h3>
        </div>
        <Sparkles aria-hidden="true" size={31} />
      </div>
      <div className={styles.controls}>
        <label className={styles.control}>
          <span>Noise time t: {noiseStep}%</span>
          <input
            aria-label="Diffusion noise time"
            type="range"
            min="0"
            max="100"
            value={noiseStep}
            onChange={(event) => setNoiseStep(Number(event.target.value))}
          />
        </label>
        <label className={styles.control}>
          <span>Noise seed: {seed}</span>
          <input
            aria-label="Diffusion noise seed"
            type="range"
            min="1"
            max="9"
            step="1"
            value={seed}
            onChange={(event) => setSeed(Number(event.target.value))}
          />
        </label>
      </div>
      <div className={styles.grid2}>
        <div>
          <div className={styles.canvasGrid}>
            {cells.map((value, index) => {
              const tone = Math.max(0, Math.min(1, (value + 1.5) / 3));
              const red = Math.round(44 + tone * 200);
              const green = Math.round(28 + tone * 210);
              const blue = Math.round(82 + tone * 165);
              return (
                <span
                  aria-hidden="true"
                  className={styles.canvasCell}
                  key={index}
                  style={{ backgroundColor: `rgb(${red}, ${green}, ${blue})` }}
                />
              );
            })}
          </div>
          <p className={styles.status}>
            The same clean diamond, one sampled noise field, and a controllable
            mixture.
          </p>
        </div>
        <div className={styles.formulaTrail}>
          <div className={styles.formulaStep}>
            <strong>clean signal</strong>
            <span>weight √ᾱ = {Math.sqrt(alphaBar).toFixed(2)}</span>
          </div>
          <div className={styles.formulaStep}>
            <strong>random noise</strong>
            <span>weight √(1−ᾱ) = {Math.sqrt(1 - alphaBar).toFixed(2)}</span>
          </div>
          <div className={styles.formulaStep}>
            <strong>noisy sample</strong>
            <span>the weighted sum shown at left</span>
          </div>
        </div>
      </div>
      <ProbabilityFormula
        label="Closed-form forward noising"
        formula={String.raw`\[x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I)\]`}
      >
        Read it left to right: retain some clean signal, add a sampled noise
        field, and obtain the noisy state at time t. The bar over α summarizes
        all earlier small noising steps.
      </ProbabilityFormula>
    </div>
  );
}

function ReverseAndGuidanceLab() {
  const [guidance, setGuidance] = useState(2);
  const unconditionalEstimate = 0.25;
  const conditionalEstimate = -0.45;
  const guided =
    unconditionalEstimate +
    guidance * (conditionalEstimate - unconditionalEstimate);
  return (
    <div className={styles.lab} data-testid="guidance-lab">
      <div className={styles.labHeader}>
        <div>
          <p>Learned reverse process</p>
          <h3>Predict a slightly cleaner distribution, sample, and repeat</h3>
        </div>
        <Sparkles aria-hidden="true" size={31} />
      </div>
      <div className={styles.formulaTrail}>
        <div className={styles.formulaStep}>
          <strong>
            x<sub>T</sub>
          </strong>
          <span>nearly pure noise</span>
        </div>
        <div className={styles.formulaStep}>
          <strong>model</strong>
          <span>estimate noise or a cleaner state</span>
        </div>
        <div className={styles.formulaStep}>
          <strong>
            sample x<sub>t−1</sub>
          </strong>
          <span>one plausible reverse step</span>
        </div>
        <div className={styles.formulaStep}>
          <strong>repeat</strong>
          <span>structure gradually appears</span>
        </div>
      </div>
      <label className={styles.control}>
        <span>Prompt guidance scale w: {guidance.toFixed(1)}</span>
        <input
          aria-label="Prompt guidance scale"
          type="range"
          min="0"
          max="6"
          step="0.5"
          value={guidance}
          onChange={(event) => setGuidance(Number(event.target.value))}
        />
      </label>
      <div className={styles.metricGrid}>
        <ProbabilityMetric
          label="unconditional estimate"
          value={unconditionalEstimate.toFixed(2)}
          detail="what fits generally"
        />
        <ProbabilityMetric
          label="conditional estimate"
          value={conditionalEstimate.toFixed(2)}
          detail="what fits the prompt"
        />
        <ProbabilityMetric
          label="guided estimate"
          value={guided.toFixed(2)}
          detail="extrapolated prompt direction"
        />
      </div>
      <ProbabilityFormula
        label="Classifier-free guidance"
        formula={String.raw`\[\hat\epsilon_{guided}=\hat\epsilon_{uncond}+w(\hat\epsilon_{cond}-\hat\epsilon_{uncond})=${guided.toFixed(2)}\]`}
      >
        Stronger guidance usually improves prompt adherence but can reduce
        sample diversity or introduce artifacts. It modifies the reverse
        prediction; it does not make the process deterministic.
      </ProbabilityFormula>
      <ProbabilityInsight
        title="Reverse diffusion cannot simply subtract the original noise"
        tone="warning"
      >
        <p>
          At generation time the original clean image and exact forward noise
          are unknown. The model learns a conditional distribution of plausible
          cleaner states. Different random seeds can therefore produce different
          valid outputs for the same prompt.
        </p>
      </ProbabilityInsight>
    </div>
  );
}

export default function CrashProbabilityL5LearningPage({ experience }: Props) {
  return (
    <ProbabilityCourse
      experience={experience}
      station="l5"
      kicker="Station L5 · Sampling, latent variables, and diffusion"
      headline="A probability model describes possibilities. Sampling commits to one of them."
      introduction="Generation repeatedly turns distributions into concrete choices. Language models sample tokens, latent-variable models sample hidden causes and observations, and diffusion models sample a path from noise toward structured data."
      heroVisual={<HeroGenerationForge />}
    >
      <ProbabilitySection
        id="sampling"
        eyebrow="01 · From distribution to outcome"
        title="Sampling preserves alternatives that greedy choice erases."
        lead="Greedy decoding always selects the largest probability. Sampling allocates outcomes in proportion to probability. Temperature and truncation reshape the distribution before the draw, changing diversity without retraining the model."
      >
        <SamplingForgeLab />
        <ProbabilityCheck
          testId="l5-sampling-check"
          title="Separate probability from outcome"
          question="A token has 70% probability but is not drawn. Was the distribution wrong?"
          options={[
            {
              label: "Yes—70% means it should occur",
              explanation:
                "Seventy percent is a long-run frequency, not a promise about one draw.",
            },
            {
              label: "No—30% of valid draws choose something else",
              explanation:
                "A lower-probability outcome is expected to occur sometimes under honest sampling.",
            },
            {
              label: "Only if temperature was 1",
              explanation:
                "Temperature shapes the distribution but does not turn a 70% event into certainty.",
            },
          ]}
          correctIndex={1}
        />
      </ProbabilitySection>

      <ProbabilitySection
        id="autoregressive"
        eyebrow="02 · Sample, append, condition again"
        title="Each generated choice changes the next probability universe."
        lead="Autoregressive models factor a sequence into conditional next-token distributions. This is the same chain rule used for likelihood in L3, now run forward to create data rather than backward to score observed data."
      >
        <RepeatedConditioningLab />
      </ProbabilitySection>

      <ProbabilitySection
        id="latent"
        eyebrow="03 · Hidden variables"
        title="Sometimes the model explains observations through an unobserved cause."
        lead="A latent variable z is part of the model’s story but absent from the raw observation. Generation samples z then x given z; inference observes x and updates beliefs about z."
      >
        <LatentVariableLab />
      </ProbabilitySection>

      <ProbabilitySection
        id="gaussian"
        eyebrow="04 · Gaussian noise"
        title="Diffusion needs a noise distribution we can sample and analyze."
        lead="A Gaussian supplies continuous perturbations with controllable center and scale. Independent standard-normal noise creates a tractable forward process whose uncertainty can be added in many small steps."
      >
        <GaussianNoiseLab />
      </ProbabilitySection>

      <ProbabilitySection
        id="diffusion"
        eyebrow="05 · Forward process"
        title="Destroy structure in a controlled way so reversal becomes a learning problem."
        lead="Forward diffusion gradually mixes data with random noise. Because that corruption rule is known, training can generate noisy examples at arbitrary times and ask a model to predict the added noise."
      >
        <DiffusionLab />
      </ProbabilitySection>

      <ProbabilitySection
        id="reverse"
        eyebrow="06 · Reverse process and guidance"
        title="Generation follows a learned conditional path from noise to structure."
        lead="The reverse transition is probabilistic because many clean samples could plausibly explain a noisy state. A prompt conditions those possibilities; guidance pushes the path toward prompt-compatible regions."
      >
        <ReverseAndGuidanceLab />
        <div className={styles.tableWrap} style={{ marginTop: 22 }}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th scope="col">System</th>
                <th scope="col">Condition</th>
                <th scope="col">Distribution</th>
                <th scope="col">One sampled step</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">LLM</th>
                <td>prior tokens</td>
                <td>
                  <InlineProbabilityMath
                    text={String.raw`\(P(x_t\mid x_{<t})\)`}
                  />
                </td>
                <td>next token</td>
              </tr>
              <tr>
                <th scope="row">RL policy</th>
                <td>current state</td>
                <td>
                  <InlineProbabilityMath
                    text={String.raw`\(\pi(a_t\mid s_t)\)`}
                  />
                </td>
                <td>next action</td>
              </tr>
              <tr>
                <th scope="row">Diffusion</th>
                <td>noisy sample + prompt</td>
                <td>
                  <InlineProbabilityMath
                    text={String.raw`\(p(x_{t-1}\mid x_t,c)\)`}
                  />
                </td>
                <td>slightly cleaner sample</td>
              </tr>
            </tbody>
          </table>
        </div>
      </ProbabilitySection>

      <ProbabilityQuizLaunch
        experience={experience}
        recap={[
          "Sampling turns a distribution into one outcome; a likely event is not a guaranteed event.",
          "Greedy, temperature, top-k, and top-p decoding trade determinism, diversity, and tail risk differently.",
          "Autoregressive generation repeats conditional prediction and sampling one token at a time.",
          "Latent-variable models marginalize hidden causes to predict observations and use Bayes to infer causes from observations.",
          "Gaussian mean and variance control continuous noise; density height is not point probability.",
          "Diffusion learns probabilistic reverse transitions from noise to data, with conditioning and guidance shaping the path.",
        ]}
      />
    </ProbabilityCourse>
  );
}
