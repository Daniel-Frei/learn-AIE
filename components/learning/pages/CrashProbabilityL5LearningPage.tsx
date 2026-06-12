"use client";

import { useMemo, useState } from "react";
import MathText from "../../MathText";
import type { LearningExperience } from "../../../lib/learning";
import {
  CheckForUnderstanding,
  ConceptCard,
  FormulaBlock,
  LearningHero,
  MisconceptionCallout,
  QuizTransitionButton,
  RecapSection,
  WorkedExample,
} from "../LearningPrimitives";

type Props = {
  experience: LearningExperience;
};

type DecodingMode = "sample" | "greedy" | "top-k" | "top-p";

type TokenOption = {
  label: string;
  probability: number;
};

type DisplayToken = TokenOption & {
  baseProbability: number;
  eligible: boolean;
};

type LatentKey = "style" | "lighting" | "season" | "viewpoint";

type LatentOption = {
  id: string;
  label: string;
  description: string;
};

type LatentState = Record<LatentKey, string>;

type SeedId = "seed-a" | "seed-b";

const BASE_TOKENS: readonly TokenOption[] = [
  { label: "mat", probability: 0.5 },
  { label: "sofa", probability: 0.25 },
  { label: "floor", probability: 0.15 },
  { label: "chair", probability: 0.07 },
  { label: "car", probability: 0.03 },
];

const DRAW_VALUES = [0.14, 0.53, 0.82, 0.97] as const;

const DECODING_MODES: readonly {
  id: DecodingMode;
  label: string;
  description: string;
}[] = [
  {
    id: "sample",
    label: "Sample",
    description: "Draw from every token after temperature scaling.",
  },
  {
    id: "greedy",
    label: "Greedy",
    description: "Choose the largest probability and ignore the rest.",
  },
  {
    id: "top-k",
    label: "Top-k",
    description: "Keep the three most likely tokens, then renormalize.",
  },
  {
    id: "top-p",
    label: "Top-p",
    description: "Keep the smallest prefix whose mass reaches 0.90.",
  },
];

const LATENT_OPTIONS: Record<LatentKey, readonly LatentOption[]> = {
  style: [
    {
      id: "modern",
      label: "Modern",
      description: "flat roof, clean edges",
    },
    {
      id: "rustic",
      label: "Rustic",
      description: "wood walls, simple shape",
    },
    {
      id: "gothic",
      label: "Gothic",
      description: "steep roof, tall window",
    },
    {
      id: "alpine",
      label: "Alpine",
      description: "wide roof, mountain cabin",
    },
  ],
  lighting: [
    { id: "morning", label: "Morning", description: "cool bright sky" },
    { id: "sunset", label: "Sunset", description: "warm orange light" },
    { id: "night", label: "Night", description: "dark sky, lit windows" },
  ],
  season: [
    { id: "summer", label: "Summer", description: "green ground" },
    { id: "winter", label: "Winter", description: "snow and pale contrast" },
  ],
  viewpoint: [
    { id: "front", label: "Front", description: "symmetric front view" },
    { id: "aerial", label: "Aerial", description: "roof-dominant view" },
    { id: "interior", label: "Interior", description: "visible room cues" },
  ],
};

const INITIAL_LATENT_STATE: LatentState = {
  style: "alpine",
  lighting: "sunset",
  season: "winter",
  viewpoint: "front",
};

const DIFFUSION_SEEDS: Record<
  SeedId,
  { label: string; finalLabel: string; promptFit: string }
> = {
  "seed-a": {
    label: "Seed A",
    finalLabel: "small cabin with a red roof",
    promptFit: "a cabin near a cold lake",
  },
  "seed-b": {
    label: "Seed B",
    finalLabel: "tall house beside pine trees",
    promptFit: "a mountain house at sunset",
  },
};

const FINAL_PATTERNS: Record<SeedId, readonly string[]> = {
  "seed-a": [
    "ssssssss",
    "ssmmmmss",
    "smmssmms",
    "sssrrsss",
    "ssrrrrss",
    "sswoowss",
    "sswddwss",
    "gggggggg",
  ],
  "seed-b": [
    "ssssssss",
    "sssmssss",
    "ssmmmsss",
    "sttrrrss",
    "sttrrrrs",
    "sswwwwts",
    "ggwddwgg",
    "gggggggg",
  ],
};

const CELL_COLORS: Record<string, string> = {
  s: "#8ed3f4",
  m: "#64748b",
  r: "#be3a34",
  w: "#f1d6a8",
  o: "#facc15",
  d: "#8b5e34",
  g: "#5a8f51",
  t: "#1f7a4f",
  n0: "#0f172a",
  n1: "#334155",
  n2: "#94a3b8",
  n3: "#e2e8f0",
  n4: "#f59e0b",
};

function formatProbability(value: number): string {
  return value.toFixed(2);
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function getTemperatureScaledTokens(temperature: number): DisplayToken[] {
  const logits = BASE_TOKENS.map((token) => Math.log(token.probability));
  const scaledLogits = logits.map((logit) => logit / temperature);
  const maxLogit = Math.max(...scaledLogits);
  const exponentials = scaledLogits.map((logit) => Math.exp(logit - maxLogit));
  const total = exponentials.reduce((sum, value) => sum + value, 0);

  return BASE_TOKENS.map((token, index) => ({
    label: token.label,
    probability: exponentials[index] / total,
    baseProbability: token.probability,
    eligible: true,
  }));
}

function applyDecodingMode(
  tokens: readonly DisplayToken[],
  mode: DecodingMode,
): DisplayToken[] {
  const sortedTokens = [...tokens].sort(
    (first, second) => second.probability - first.probability,
  );
  let eligibleLabels = new Set(tokens.map((token) => token.label));

  if (mode === "greedy") {
    eligibleLabels = new Set([sortedTokens[0].label]);
  }

  if (mode === "top-k") {
    eligibleLabels = new Set(
      sortedTokens.slice(0, 3).map((token) => token.label),
    );
  }

  if (mode === "top-p") {
    const labels: string[] = [];
    let cumulative = 0;
    for (const token of sortedTokens) {
      labels.push(token.label);
      cumulative += token.probability;
      if (cumulative >= 0.9) break;
    }
    eligibleLabels = new Set(labels);
  }

  const eligibleTotal = tokens.reduce(
    (sum, token) =>
      eligibleLabels.has(token.label) ? sum + token.probability : sum,
    0,
  );

  return tokens.map((token) => {
    const eligible = eligibleLabels.has(token.label);
    const probability =
      mode === "greedy"
        ? eligible
          ? 1
          : 0
        : eligible
          ? token.probability / eligibleTotal
          : 0;

    return {
      ...token,
      probability,
      eligible,
    };
  });
}

function getSelectedToken(tokens: readonly DisplayToken[], draw: number) {
  let cumulative = 0;
  const eligibleTokens = tokens.filter((token) => token.eligible);

  for (const token of eligibleTokens) {
    cumulative += token.probability;
    if (draw <= cumulative) return token;
  }

  return eligibleTokens[eligibleTokens.length - 1];
}

function getEntropy(tokens: readonly DisplayToken[]): number {
  return tokens.reduce((sum, token) => {
    if (token.probability <= 0) return sum;
    return sum - token.probability * Math.log2(token.probability);
  }, 0);
}

function getModeDescription(mode: DecodingMode): string {
  return (
    DECODING_MODES.find((item) => item.id === mode)?.description ??
    "Draw from the adjusted distribution."
  );
}

function getLatentOption(key: LatentKey, id: string): LatentOption {
  const option = LATENT_OPTIONS[key].find((item) => item.id === id);
  if (!option) {
    throw new Error(`Unknown latent option ${key}:${id}`);
  }
  return option;
}

function pseudoNoise(seed: SeedId, x: number, y: number, step: number): number {
  const seedOffset = seed === "seed-a" ? 17 : 53;
  const value =
    Math.sin((x + 1) * 12.9898 + (y + 1) * 78.233 + seedOffset * 4.719 + step) *
    43758.5453;
  return value - Math.floor(value);
}

function getFinalCell(seed: SeedId, x: number, y: number): string {
  return FINAL_PATTERNS[seed][y][x];
}

function getNoisyCell(
  seed: SeedId,
  x: number,
  y: number,
  step: number,
): string {
  if (step === 0) return getFinalCell(seed, x, y);

  const noiseShare = step / 5;
  const revealNoise = pseudoNoise(seed, x, y, step);
  if (revealNoise < noiseShare) {
    return `n${Math.floor(pseudoNoise(seed, x + 3, y + 7, step + 11) * 5)}`;
  }

  return getFinalCell(seed, x, y);
}

function GenerationLoopVisual() {
  const stages = [
    {
      label: "Learned distribution",
      body: "many possible outputs",
      accent: "bg-sky-300",
    },
    {
      label: "Sampling control",
      body: "temperature, top-k, top-p",
      accent: "bg-amber-300",
    },
    {
      label: "Concrete sample",
      body: "one token, image, or action",
      accent: "bg-emerald-300",
    },
  ] as const;

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <div className="grid gap-3">
        {stages.map((stage, index) => (
          <div key={stage.label} className="flex items-center gap-3">
            <div
              className={`h-3 w-3 shrink-0 rounded-full ${stage.accent}`}
              aria-hidden="true"
            />
            <div className="flex-1 rounded-md border border-slate-700 bg-slate-950 px-4 py-3">
              <p className="text-sm font-semibold text-slate-100">
                {stage.label}
              </p>
              <p className="text-xs text-slate-400">{stage.body}</p>
            </div>
            {index < stages.length - 1 && (
              <span className="text-lg font-semibold text-slate-400">
                -&gt;
              </span>
            )}
          </div>
        ))}
      </div>
      <div className="mt-5 grid grid-cols-5 gap-2" aria-hidden="true">
        {BASE_TOKENS.map((token) => (
          <div key={token.label} className="space-y-2">
            <div className="flex h-24 items-end rounded-md bg-slate-950 p-2">
              <div
                className="w-full rounded-sm bg-cyan-300"
                style={{ height: `${Math.max(8, token.probability * 100)}%` }}
              />
            </div>
            <p className="text-center text-xs font-semibold text-slate-300">
              {token.label}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

function SamplingDistributionLab() {
  const [temperature, setTemperature] = useState(1);
  const [mode, setMode] = useState<DecodingMode>("sample");
  const [drawIndex, setDrawIndex] = useState(0);

  const scaledTokens = useMemo(
    () => getTemperatureScaledTokens(temperature),
    [temperature],
  );
  const displayTokens = useMemo(
    () => applyDecodingMode(scaledTokens, mode),
    [scaledTokens, mode],
  );
  const draw = DRAW_VALUES[drawIndex];
  const selectedToken = getSelectedToken(displayTokens, draw);
  const entropy = getEntropy(displayTokens);
  const eligibleCount = displayTokens.filter((token) => token.eligible).length;

  return (
    <section
      data-testid="sampling-lab"
      className="rounded-lg border border-slate-800 bg-slate-900 p-5"
    >
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase text-cyan-300">
            Interactive model
          </p>
          <h2 className="mt-2 text-xl font-semibold text-slate-50">
            Sampling distribution lab
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
            The prompt is &quot;The animal sat on the ...&quot;. Change the
            decoding rule and temperature, then draw a sample from the adjusted
            distribution.
          </p>
        </div>
        <p
          role="status"
          data-testid="sampling-summary"
          className="rounded-md border border-cyan-400/50 bg-cyan-950/30 px-3 py-2 text-sm font-semibold text-cyan-100"
        >
          {DECODING_MODES.find((item) => item.id === mode)?.label} selects{" "}
          {selectedToken.label} / entropy {entropy.toFixed(2)} bits
        </p>
      </div>

      <div className="mt-5 grid gap-5 md:grid-cols-[0.9fr_1.1fr]">
        <div className="space-y-5">
          <div>
            <p className="text-sm font-semibold text-slate-100">
              Decoding strategy
            </p>
            <div className="mt-2 grid grid-cols-2 gap-2">
              {DECODING_MODES.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  aria-pressed={mode === item.id}
                  onClick={() => setMode(item.id)}
                  className={`rounded-md border px-3 py-2 text-sm font-semibold transition-colors ${
                    mode === item.id
                      ? "border-cyan-300 bg-cyan-300 text-slate-950"
                      : "border-slate-700 bg-slate-950 text-slate-200 hover:border-slate-500"
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </div>
            <p className="mt-2 min-h-10 text-xs leading-5 text-slate-400">
              {getModeDescription(mode)}
            </p>
          </div>

          <label className="block space-y-2">
            <div className="flex items-center justify-between gap-3 text-sm">
              <span className="font-semibold text-slate-100">Temperature</span>
              <span className="font-mono text-slate-300">
                T = {temperature.toFixed(1)}
              </span>
            </div>
            <input
              type="range"
              aria-label="Temperature"
              min="0.4"
              max="1.8"
              step="0.1"
              value={temperature}
              onChange={(event) => setTemperature(Number(event.target.value))}
              className="w-full accent-cyan-300"
            />
            <p className="text-xs leading-5 text-slate-400">
              Low temperature sharpens the distribution. High temperature makes
              lower-probability tokens more competitive.
            </p>
          </label>

          <button
            type="button"
            onClick={() =>
              setDrawIndex((current) => (current + 1) % DRAW_VALUES.length)
            }
            className="w-full rounded-md border border-emerald-400 bg-emerald-400 px-3 py-2 text-sm font-bold text-slate-950 transition-colors hover:bg-emerald-300"
          >
            Draw next sample
          </button>

          <div className="rounded-md border border-slate-700 bg-slate-950 p-3 text-sm text-slate-300">
            <p>
              Draw value: <span className="font-mono">{draw.toFixed(2)}</span>
            </p>
            <p>
              Eligible candidates:{" "}
              <span className="font-mono">{eligibleCount}</span>
            </p>
          </div>
        </div>

        <div className="grid gap-3">
          {displayTokens.map((token) => {
            const isSelected = token.label === selectedToken.label;
            return (
              <div
                key={token.label}
                className={`rounded-md border bg-slate-950 p-3 ${
                  isSelected
                    ? "border-emerald-300"
                    : token.eligible
                      ? "border-slate-700"
                      : "border-slate-800 opacity-60"
                }`}
              >
                <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
                  <span className="font-semibold text-slate-100">
                    {token.label}
                  </span>
                  <span className="font-mono text-slate-300">
                    base {formatProbability(token.baseProbability)} / used{" "}
                    {formatProbability(token.probability)}
                  </span>
                </div>
                <div className="mt-2 h-3 overflow-hidden rounded-full bg-slate-800">
                  <div
                    className={`h-full rounded-full ${
                      isSelected ? "bg-emerald-300" : "bg-cyan-300"
                    }`}
                    style={{
                      width: `${Math.max(2, token.probability * 100)}%`,
                    }}
                  />
                </div>
                <p className="mt-2 text-xs text-slate-400">
                  {token.eligible
                    ? isSelected
                      ? "Selected by this draw."
                      : "Eligible for this decoding rule."
                    : "Excluded before sampling."}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

function HousePreview({ latent }: { latent: LatentState }) {
  const lighting = latent.lighting;
  const season = latent.season;
  const style = latent.style;
  const sky =
    lighting === "night"
      ? "#172554"
      : lighting === "sunset"
        ? "#f7b267"
        : "#93c5fd";
  const ground = season === "winter" ? "#e0f2fe" : "#4d8b55";
  const wall =
    style === "modern"
      ? "#cbd5e1"
      : style === "gothic"
        ? "#9ca3af"
        : style === "alpine"
          ? "#f4d6a4"
          : "#b7794b";
  const roof =
    style === "modern" ? "#475569" : style === "gothic" ? "#4c1d95" : "#b91c1c";

  return (
    <svg
      role="img"
      aria-label="Generated house preview from selected latent variables"
      viewBox="0 0 220 150"
      className="h-auto w-full rounded-md border border-slate-700 bg-slate-950"
    >
      <rect width="220" height="150" fill={sky} />
      <rect y="104" width="220" height="46" fill={ground} />
      {season === "winter" && (
        <path
          d="M0 108 C35 96 68 118 108 106 C148 94 182 110 220 100 V150 H0 Z"
          fill="#f8fafc"
        />
      )}
      {lighting === "night" ? (
        <circle cx="180" cy="28" r="12" fill="#fde68a" />
      ) : (
        <circle cx="178" cy="30" r="15" fill="#fef3c7" />
      )}
      {style === "modern" ? (
        <>
          <rect x="55" y="68" width="112" height="52" fill={wall} />
          <rect x="48" y="58" width="126" height="14" fill={roof} />
        </>
      ) : (
        <>
          <polygon points="42,78 110,34 178,78" fill={roof} />
          <rect x="58" y="76" width="104" height="48" fill={wall} />
          {style === "gothic" && (
            <polygon
              points="92,76 110,42 128,76"
              fill="#312e81"
              opacity="0.55"
            />
          )}
        </>
      )}
      <rect
        x="98"
        y="94"
        width="23"
        height="30"
        fill={lighting === "night" ? "#92400e" : "#7c4a2d"}
      />
      <rect
        x="70"
        y="87"
        width="20"
        height="18"
        fill={lighting === "night" ? "#fde047" : "#bae6fd"}
      />
      <rect
        x="130"
        y="87"
        width="20"
        height="18"
        fill={lighting === "night" ? "#fde047" : "#bae6fd"}
      />
      {latent.viewpoint === "aerial" && (
        <ellipse
          cx="110"
          cy="78"
          rx="72"
          ry="16"
          fill="#0f172a"
          opacity="0.18"
        />
      )}
      {latent.viewpoint === "interior" && (
        <rect
          x="66"
          y="82"
          width="88"
          height="35"
          fill="#fef3c7"
          opacity="0.45"
        />
      )}
    </svg>
  );
}

function LatentVariableMixer() {
  const [latent, setLatent] = useState<LatentState>(INITIAL_LATENT_STATE);
  const selectedOptions = (Object.keys(LATENT_OPTIONS) as LatentKey[]).map(
    (key) => ({
      key,
      option: getLatentOption(key, latent[key]),
    }),
  );

  const visibleDescription = selectedOptions
    .map(({ option }) => option.description)
    .join(", ");

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase text-amber-300">
            Hidden structure
          </p>
          <h2 className="mt-2 text-xl font-semibold text-slate-50">
            Latent house mixer
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
            The visible image is not generated from pixels independently. Hidden
            variables such as style, lighting, season, and viewpoint explain
            coordinated structure in the output.
          </p>
        </div>
        <MathText
          text={String.raw`\[z\sim P(z),\quad x\sim P(x\mid z)\]`}
          className="overflow-x-auto rounded-md border border-amber-400/40 bg-amber-950/20 px-3 py-2 text-sm text-amber-100"
        />
      </div>

      <div className="mt-5 grid gap-5 md:grid-cols-[0.9fr_1.1fr]">
        <div className="grid gap-4">
          {(Object.keys(LATENT_OPTIONS) as LatentKey[]).map((key) => (
            <label key={key} className="block space-y-2">
              <span className="text-sm font-semibold capitalize text-slate-100">
                {key}
              </span>
              <select
                aria-label={`${key} latent variable`}
                value={latent[key]}
                onChange={(event) =>
                  setLatent((current) => ({
                    ...current,
                    [key]: event.target.value,
                  }))
                }
                className="w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100"
              >
                {LATENT_OPTIONS[key].map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
          ))}
        </div>

        <div className="space-y-4">
          <HousePreview latent={latent} />
          <div className="rounded-md border border-slate-700 bg-slate-950 p-3">
            <p className="text-sm font-semibold text-slate-100">
              Observed output x
            </p>
            <p className="mt-2 text-sm leading-6 text-slate-300">
              A generated house with {visibleDescription}. The latent variables
              are hidden causes in the model, even when a human can infer them
              from the image.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {selectedOptions.map(({ key, option }) => (
              <span
                key={key}
                className="rounded-md border border-amber-400/40 bg-amber-950/30 px-2 py-1 text-xs font-semibold text-amber-100"
              >
                {key}: {option.label}
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function DiffusionPathLab() {
  const [seed, setSeed] = useState<SeedId>("seed-a");
  const [step, setStep] = useState(5);
  const structurePercent = (5 - step) / 5;
  const seedConfig = DIFFUSION_SEEDS[seed];

  return (
    <section
      data-testid="diffusion-path-lab"
      className="rounded-lg border border-slate-800 bg-slate-900 p-5"
    >
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase text-rose-300">
            Reverse process
          </p>
          <h2 className="mt-2 text-xl font-semibold text-slate-50">
            Diffusion denoising path
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
            Diffusion starts from Gaussian-like noise and repeatedly predicts a
            small move toward cleaner structure. Different noise seeds can
            satisfy the same prompt in different ways.
          </p>
        </div>
        <p
          role="status"
          data-testid="diffusion-summary"
          className="rounded-md border border-rose-400/50 bg-rose-950/30 px-3 py-2 text-sm font-semibold text-rose-100"
        >
          {seedConfig.label} / t={step} / structure{" "}
          {formatPercent(structurePercent)}
        </p>
      </div>

      <div className="mt-5 grid min-w-0 gap-5 md:grid-cols-[0.95fr_1.05fr]">
        <div className="min-w-0 space-y-5">
          <div
            role="img"
            aria-label={`Noisy diffusion sample for ${seedConfig.finalLabel}`}
            className="grid aspect-square w-full max-w-md grid-cols-8 gap-1 rounded-md border border-slate-700 bg-slate-950 p-2"
          >
            {Array.from({ length: 64 }, (_, index) => {
              const x = index % 8;
              const y = Math.floor(index / 8);
              const cell = getNoisyCell(seed, x, y, step);
              return (
                <span
                  key={`${x}-${y}`}
                  className="aspect-square rounded-sm"
                  style={{ backgroundColor: CELL_COLORS[cell] }}
                />
              );
            })}
          </div>
          <div className="rounded-md border border-slate-700 bg-slate-950 p-3 text-sm leading-6 text-slate-300">
            <p>
              Prompt condition:{" "}
              <span className="font-semibold text-slate-100">
                {seedConfig.promptFit}
              </span>
            </p>
            <p>
              Current sample:{" "}
              {step === 5
                ? "mostly random noise, many futures still plausible"
                : step === 0
                  ? seedConfig.finalLabel
                  : "partly denoised, with visible structure emerging"}
            </p>
          </div>
        </div>

        <div className="min-w-0 space-y-5">
          <div>
            <p className="text-sm font-semibold text-slate-100">Noise seed</p>
            <div className="mt-2 grid grid-cols-2 gap-2">
              {(Object.keys(DIFFUSION_SEEDS) as SeedId[]).map((seedId) => (
                <button
                  key={seedId}
                  type="button"
                  aria-pressed={seed === seedId}
                  onClick={() => {
                    setSeed(seedId);
                    setStep(5);
                  }}
                  className={`rounded-md border px-3 py-2 text-sm font-semibold transition-colors ${
                    seed === seedId
                      ? "border-rose-300 bg-rose-300 text-slate-950"
                      : "border-slate-700 bg-slate-950 text-slate-200 hover:border-slate-500"
                  }`}
                >
                  {DIFFUSION_SEEDS[seedId].label}
                </button>
              ))}
            </div>
          </div>

          <label className="block space-y-2">
            <div className="flex items-center justify-between gap-3 text-sm">
              <span className="font-semibold text-slate-100">
                Denoising time step
              </span>
              <span className="font-mono text-slate-300">t = {step}</span>
            </div>
            <input
              type="range"
              aria-label="Denoising time step"
              min="0"
              max="5"
              step="1"
              value={step}
              onChange={(event) => setStep(Number(event.target.value))}
              className="w-full accent-rose-300"
            />
            <p className="text-xs leading-5 text-slate-400">
              t=5 is high uncertainty. t=0 is the denoised generated sample.
            </p>
          </label>

          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => setStep((current) => Math.max(0, current - 1))}
              className="rounded-md border border-emerald-400 bg-emerald-400 px-3 py-2 text-sm font-bold text-slate-950 transition-colors hover:bg-emerald-300"
            >
              Denoise one step
            </button>
            <button
              type="button"
              onClick={() => setStep(5)}
              className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm font-semibold text-slate-200 transition-colors hover:border-slate-500"
            >
              Reset noise
            </button>
          </div>

          <MathText
            text={String.raw`\[x_T\sim\mathcal{N}(0,I)\]`}
            className="overflow-x-auto rounded-md border border-slate-700 bg-slate-950 px-4 py-3 text-slate-100"
          />
          <MathText
            text={String.raw`\[x_T\rightarrow x_{T-1}\rightarrow\cdots\rightarrow x_0\]`}
            className="overflow-x-auto rounded-md border border-slate-700 bg-slate-950 px-4 py-3 text-slate-100"
          />
          <MathText
            text={String.raw`\[p_\theta(x_{t-1}\mid x_t,c)\]`}
            className="overflow-x-auto rounded-md border border-slate-700 bg-slate-950 px-4 py-3 text-slate-100"
          />
        </div>
      </div>
    </section>
  );
}

export default function CrashProbabilityL5LearningPage({ experience }: Props) {
  return (
    <main className="bg-slate-950 text-slate-50">
      <LearningHero
        eyebrow="Crash Course Probability L5"
        title="Turn uncertainty into generated output"
        summary="Generative AI does not store one answer. It learns probability structure, adjusts how that distribution is used, then samples tokens, latent variables, actions, or denoising paths."
        meta={`${experience.durationMinutes} min interactive prep / ${experience.level}`}
        outcomes={experience.outcomes}
        visual={<GenerationLoopVisual />}
      />

      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-10 md:py-12">
        <SamplingDistributionLab />

        <section className="grid gap-4 md:grid-cols-3">
          <ConceptCard title="Distribution" label="Possibilities">
            <p>
              A model first assigns probability mass across possible next
              tokens, images, actions, or hidden variables. Nothing has been
              generated yet.
            </p>
          </ConceptCard>
          <ConceptCard title="Decoder" label="Rule">
            <p>
              Greedy, sampling, top-k, top-p, temperature, or guidance controls
              how the learned distribution is used at inference time.
            </p>
          </ConceptCard>
          <ConceptCard title="Sample" label="Concrete output">
            <p>
              A sample is one realized outcome. Many different samples can be
              valid under the same prompt because the prompt leaves details
              unspecified.
            </p>
          </ConceptCard>
        </section>

        <FormulaBlock
          title="Temperature changes odds, not knowledge"
          formula={String.raw`\[P(y_i)=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}\]`}
          explanation="A smaller positive T sharpens logit differences and lowers entropy. A larger T flattens probabilities and raises diversity. Neither setting adds facts or fixes a wrong model belief by itself."
        />

        <CheckForUnderstanding
          testId="temperature-knowledge-check"
          title="Check: temperature"
          question="A model assigns high probability to a false claim. What does raising temperature do?"
          correctIndex={1}
          options={[
            {
              label: "It adds new knowledge that makes the claim true.",
              explanation:
                "Temperature only changes how the existing distribution is sampled. It does not add external evidence or new reasoning.",
            },
            {
              label:
                "It flattens the distribution and may make alternatives more likely, but it does not verify truth.",
              explanation:
                "Temperature controls randomness and entropy. Truthfulness still depends on model knowledge, retrieval, reasoning, and verification.",
            },
            {
              label:
                "It turns the decoder into greedy decoding because all probabilities become sharper.",
              explanation:
                "Higher temperature usually flattens the distribution. Greedy decoding is a separate maximum-choice rule.",
            },
          ]}
        />

        <LatentVariableMixer />

        <WorkedExample
          title="Worked example: marginalizing a latent variable"
          setup="A hidden style Z affects whether the generated house has a red roof. Suppose P(Z=alpine)=0.30, P(red roof | alpine)=0.80, P(Z=modern)=0.70, and P(red roof | modern)=0.20."
          steps={[
            "Pair each conditional probability with the latent state it conditions on.",
            "Sum over hidden alternatives: P(red roof)=0.30*0.80 + 0.70*0.20.",
            "The result is 0.38. The visible output probability comes from weighting each hidden cause by its prior probability.",
          ]}
        />

        <DiffusionPathLab />

        <section className="grid gap-4 md:grid-cols-2">
          <ConceptCard title="Forward noising" label="Fixed corruption">
            <MathText
              text={String.raw`\[q(x_t\mid x_{t-1})\]`}
              className="overflow-x-auto rounded-md bg-slate-950 px-3 py-2 text-slate-100"
            />
            <p>
              Training creates noisy versions of real data by adding controlled
              Gaussian noise until structure is almost washed out.
            </p>
          </ConceptCard>
          <ConceptCard title="Reverse denoising" label="Learned generation">
            <MathText
              text={String.raw`\[p_\theta(x_{t-1}\mid x_t,c)\]`}
              className="overflow-x-auto rounded-md bg-slate-950 px-3 py-2 text-slate-100"
            />
            <p>
              Generation starts from random noise and repeatedly predicts a
              less-noisy sample, guided by a condition such as a text prompt.
            </p>
          </ConceptCard>
        </section>

        <MisconceptionCallout
          misconception="The same prompt should determine one exact image."
          correction="A prompt is a condition, not a full specification. The exact shape, lighting, layout, and random seed can remain uncertain, so the model samples one plausible image from a large conditional distribution."
        />

        <CheckForUnderstanding
          testId="diffusion-process-check"
          title="Check: diffusion"
          question="Which statement correctly separates the forward and reverse diffusion processes?"
          correctIndex={2}
          options={[
            {
              label:
                "The forward process starts from pure noise and learns to add image details.",
              explanation:
                "That describes generation in the reverse direction, not the fixed forward corruption process.",
            },
            {
              label:
                "The reverse process is fixed Gaussian corruption that does not use learned predictions.",
              explanation:
                "The reverse process is the learned denoising direction used to generate samples.",
            },
            {
              label:
                "The forward process adds noise to data; the reverse process learns to denoise from noise toward data.",
              explanation:
                "Forward noising is fixed corruption during training. Reverse denoising is the learned generative sampler.",
            },
          ]}
        />

        <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
          <h2 className="text-xl font-semibold text-slate-50">
            One repeated probability pattern
          </h2>
          <div className="mt-4 grid gap-3 md:grid-cols-3">
            <div className="rounded-md border border-slate-700 bg-slate-950 p-3">
              <p className="text-sm font-semibold text-sky-200">LLM</p>
              <MathText
                text={String.raw`\[P(x_t\mid x_{<t})\]`}
                className="mt-2 overflow-x-auto text-slate-100"
              />
              <p className="mt-2 text-sm leading-6 text-slate-300">
                Predict and sample one token at a time.
              </p>
            </div>
            <div className="rounded-md border border-slate-700 bg-slate-950 p-3">
              <p className="text-sm font-semibold text-rose-200">Diffusion</p>
              <MathText
                text={String.raw`\[P(x_{t-1}\mid x_t,c)\]`}
                className="mt-2 overflow-x-auto text-slate-100"
              />
              <p className="mt-2 text-sm leading-6 text-slate-300">
                Predict a less-noisy sample at each step.
              </p>
            </div>
            <div className="rounded-md border border-slate-700 bg-slate-950 p-3">
              <p className="text-sm font-semibold text-emerald-200">RL</p>
              <MathText
                text={String.raw`\[\pi(a\mid s)\]`}
                className="mt-2 overflow-x-auto text-slate-100"
              />
              <p className="mt-2 text-sm leading-6 text-slate-300">
                Sample or choose an action from a policy.
              </p>
            </div>
          </div>
        </section>

        <RecapSection
          title="Before you start the MCQs"
          items={[
            "Sampling draws an outcome from a distribution; greedy decoding chooses the maximum.",
            "Top-k and top-p restrict the candidate set, then renormalize before sampling.",
            "Temperature changes entropy and diversity, not model knowledge.",
            "Latent variables represent hidden structure that helps generate observed data.",
            "Gaussian noise is useful because it is easy to sample and forms a controlled corruption process.",
            "Diffusion learns the reverse process from noisy samples toward data.",
            "The same prompt can yield many outputs because unspecified details remain probabilistic.",
            "LLMs, diffusion models, and RL policies all use repeated conditional probability steps.",
          ]}
        />

        <section className="flex flex-col gap-4 rounded-lg border border-emerald-500/40 bg-emerald-950/20 p-5 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-xl font-semibold text-emerald-100">
              Ready for the Probability L5 questions
            </h2>
            <p className="mt-2 text-sm leading-6 text-slate-300">
              Practice sampling counts, greedy versus probabilistic decoding,
              top-k and top-p truncation, temperature odds, latent variables,
              Gaussian noising, and diffusion reverse-process notation.
            </p>
          </div>
          <QuizTransitionButton sourceId={experience.sourceId} />
        </section>

        <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
          <h2 className="text-lg font-semibold text-slate-50">
            Compact formula board
          </h2>
          <div className="mt-4 space-y-3 text-slate-200">
            <MathText
              text={String.raw`\[x\sim P(x),\quad \arg\max_x P(x)\]`}
              className="overflow-x-auto"
            />
            <MathText
              text={String.raw`\[P(y_i)=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}\]`}
              className="overflow-x-auto"
            />
            <MathText
              text={String.raw`\[z\sim P(z),\quad x\sim P(x\mid z)\]`}
              className="overflow-x-auto"
            />
            <MathText
              text={String.raw`\[X\sim\mathcal{N}(\mu,\sigma^2)\]`}
              className="overflow-x-auto"
            />
            <MathText
              text={String.raw`\[\epsilon\sim\mathcal{N}(0,I)\]`}
              className="overflow-x-auto"
            />
            <MathText
              text={String.raw`\[x_0\rightarrow x_1\rightarrow\cdots\rightarrow x_T\]`}
              className="overflow-x-auto"
            />
            <MathText
              text={String.raw`\[x_T\rightarrow x_{T-1}\rightarrow\cdots\rightarrow x_0\]`}
              className="overflow-x-auto"
            />
          </div>
        </section>
      </div>
    </main>
  );
}
