"use client";

import Image from "next/image";
import {
  ArrowRight,
  BrainCircuit,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleGauge,
  Clock3,
  Cpu,
  GitBranch,
  Layers3,
  Pause,
  Play,
  RefreshCcw,
  Shuffle,
  Sparkles,
  TriangleAlert,
  X,
  Zap,
} from "lucide-react";
import {
  Children,
  cloneElement,
  isValidElement,
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import MathText from "../../MathText";
import styles from "./DiffusionGemmaPresentationPage.module.css";

const assetBase = "/learning/diffusiongemma/presentation";

const slideNames = [
  "Title",
  "Speed frontier",
  "Thesis",
  "Artificial Analysis speed",
  "Artificial Analysis intelligence",
  "Agenda",
  "Section: Motivation",
  "AR dependency chain",
  "Weight-read bottleneck",
  "No local rewrite",
  "Why shortcuts plateau",
  "The systems target",
  "Section: From Gemma 4 to DiffusionGemma",
  "One backbone, two modes",
  "Five-step conversion",
  "Warm start",
  "Attention-mask change",
  "Denoising + self-conditioning",
  "Alternating inference modes",
  "Few-step acceleration",
  "Block-autoregressive result",
  "Section: How diffusion writes",
  "Two generation geometries",
  "Diffusion intuition",
  "A path of distributions",
  "Flow matching",
  "Discrete corruption",
  "Learn the clean token",
  "Denoising canvas",
  "Attention wiring",
  "Why multinomial noise?",
  "Entropy-bounded sampler",
  "Adaptive stopping",
  "Section: Training",
  "SFT conversion",
  "Sampler distillation + RL",
  "A self-improving curriculum",
  "Section: Evidence",
  "Speed accounting",
  "Results",
  "Self-correction",
  "Where it breaks",
  "Section: Synthesis",
  "Takeaways",
  "Discussion",
] as const;

const tokenFrames = [
  ["cobalt", "÷", "orchid", "7", "maybe", "river", "∎", "north"],
  ["The", "fast", "model", "can", "draft", "many", "tokens", "now"],
  ["A", "diffusion", "model", "can", "revise", "many", "tokens", "together"],
  [
    "A",
    "diffusion",
    "language",
    "model",
    "revises",
    "many",
    "tokens",
    "together",
  ],
  [
    "A",
    "diffusion",
    "language",
    "model",
    "refines",
    "many",
    "tokens",
    "in parallel",
  ],
] as const;

const confidenceFrames = [
  [8, 11, 6, 10, 7, 5, 12, 9],
  [54, 28, 38, 51, 31, 48, 44, 36],
  [71, 86, 67, 73, 59, 77, 80, 64],
  [91, 96, 84, 78, 82, 92, 94, 79],
  [99, 99, 97, 96, 96, 98, 99, 95],
] as const;

type SlideProps = {
  number: number;
  section: string;
  title: string;
  minutes?: string;
  children: ReactNode;
  tone?: "ink" | "paper" | "signal";
  wide?: boolean;
};

function Slide({
  number,
  title,
  children,
  tone = "ink",
  wide = false,
}: SlideProps) {
  return (
    <section
      id={`slide-${number}`}
      className={`${styles.slide} ${styles[tone]}`}
      data-testid="diffusiongemma-slide"
      aria-labelledby={`slide-${number}-title`}
    >
      <div className={`${styles.slideInner} ${wide ? styles.wide : ""}`}>
        <header className={styles.slideHeader}>
          <p className={styles.kicker}>
            <span>{String(number).padStart(2, "0")}</span>
          </p>
        </header>
        <h2 id={`slide-${number}-title`} className={styles.slideTitle}>
          {title}
        </h2>
        <div className={styles.slideBody}>{children}</div>
        <p className={styles.pageNumber} aria-hidden="true">
          DG / {String(number).padStart(2, "0")}
        </p>
      </div>
    </section>
  );
}

function DeckSequence({ children }: { children: ReactNode }) {
  return (
    <>
      {Children.map(children, (child, index) =>
        isValidElement<{ number: number }>(child)
          ? cloneElement(child, { number: index + 1 })
          : child,
      )}
    </>
  );
}

function TitleSlide({ number }: { number: number }) {
  return (
    <section
      id={`slide-${number}`}
      className={`${styles.slide} ${styles.titleSlide}`}
      data-testid="diffusiongemma-slide"
      aria-labelledby={`slide-${number}-title`}
    >
      <div className={styles.titleMinimal}>
        <p className={styles.kicker}>
          <span>DiffusionGemma</span> Technical report · August 2026
        </p>
        <h1 id={`slide-${number}-title`}>
          <small>DiffusionGemma</small>
          <span>Nearly the same intelligence. Seven times the speed.</span>
        </h1>
        <p className={styles.titleDeck}>
          A Gemma 4–based open model that reaches roughly 1,500 output tokens
          per second by revising 256-token canvases in parallel.
        </p>
        <p className={styles.titleByline}>
          DiffusionGemma Technical Report · Google DeepMind
        </p>
      </div>
      <p className={styles.pageNumber}>
        DG / {String(number).padStart(2, "0")}
      </p>
    </section>
  );
}

function SectionDivider({
  number,
  index,
  title,
  subtitle,
  motif,
}: {
  number: number;
  index: string;
  title: string;
  subtitle: string;
  motif?: ReactNode;
}) {
  return (
    <section
      id={`slide-${number}`}
      className={`${styles.slide} ${styles.sectionDivider}`}
      data-testid="diffusiongemma-slide"
      data-section-divider={index}
      aria-labelledby={`slide-${number}-title`}
    >
      <div className={styles.sectionDividerInner}>
        <p className={styles.sectionIndex}>Section {index}</p>
        <h2 id={`slide-${number}-title`}>{title}</h2>
        <p>{subtitle}</p>
        {motif ? (
          <div
            className={styles.sectionMotif}
            data-testid="section-motif"
            aria-hidden="true"
          >
            {motif}
          </div>
        ) : null}
      </div>
      <p className={styles.pageNumber}>
        DG / {String(number).padStart(2, "0")}
      </p>
    </section>
  );
}

function BigIdea({ children }: { children: ReactNode }) {
  return <p className={styles.bigIdea}>{children}</p>;
}

function Note({ children }: { children: ReactNode }) {
  return <p className={styles.note}>{children}</p>;
}

function Metric({ value, label }: { value: string; label: string }) {
  return (
    <div className={styles.metric}>
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function PaperFigure({
  src,
  alt,
  width,
  height,
  caption,
  eager = false,
  highlights = [],
}: {
  src: string;
  alt: string;
  width: number;
  height: number;
  caption: string;
  eager?: boolean;
  highlights?: Array<{
    label: string;
    left: string;
    top: string;
    width: string;
    height: string;
    tone?: "amber" | "mint";
  }>;
}) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", close);
    return () => document.removeEventListener("keydown", close);
  }, [open]);

  return (
    <>
      <button
        type="button"
        className={styles.paperFigure}
        onClick={() => setOpen(true)}
        aria-label={`Enlarge ${caption}`}
      >
        <div className={styles.figureMedia}>
          <Image
            src={src}
            alt={alt}
            width={width}
            height={height}
            sizes="90vw"
            loading={eager ? "eager" : "lazy"}
          />
          {highlights.map((highlight) => (
            <i
              key={highlight.label}
              className={`${styles.figureHighlight} ${
                highlight.tone === "mint" ? styles.figureHighlightMint : ""
              }`}
              style={{
                left: highlight.left,
                top: highlight.top,
                width: highlight.width,
                height: highlight.height,
              }}
              aria-hidden="true"
            />
          ))}
        </div>
        <span>{caption} · click to enlarge</span>
      </button>
      {open ? (
        <div
          className={styles.lightbox}
          role="dialog"
          aria-modal="true"
          aria-label={caption}
          data-testid="diffusiongemma-figure-lightbox"
        >
          <button
            type="button"
            onClick={() => setOpen(false)}
            aria-label="Close enlarged figure"
          >
            <X aria-hidden="true" />
          </button>
          <div className={styles.figureMedia}>
            <Image
              src={src}
              alt={alt}
              width={width}
              height={height}
              sizes="96vw"
            />
            {highlights.map((highlight) => (
              <i
                key={highlight.label}
                className={`${styles.figureHighlight} ${
                  highlight.tone === "mint" ? styles.figureHighlightMint : ""
                }`}
                style={{
                  left: highlight.left,
                  top: highlight.top,
                  width: highlight.width,
                  height: highlight.height,
                }}
                aria-hidden="true"
              />
            ))}
          </div>
        </div>
      ) : null}
    </>
  );
}

function ModelPass({
  label,
  tokens,
  accent,
}: {
  label: string;
  tokens: readonly string[];
  accent: "violet" | "amber";
}) {
  return (
    <div className={styles.modelPass} data-accent={accent}>
      <p>{label}</p>
      <div>
        {tokens.map((token, index) => (
          <span key={`${label}-${token}-${index}`}>{token}</span>
        ))}
      </div>
    </div>
  );
}

function TokenCanvas() {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setStep((current) => {
        if (current >= tokenFrames.length - 1) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 850);
    return () => window.clearInterval(timer);
  }, [playing]);

  const changeStep = (next: number) => {
    setPlaying(false);
    setStep(Math.max(0, Math.min(tokenFrames.length - 1, next)));
  };

  return (
    <div className={styles.canvasLab} data-testid="denoising-canvas">
      <div className={styles.canvasTimeline}>
        <span>noise · t = 1</span>
        <div>
          {tokenFrames.map((_, index) => (
            <button
              key={index}
              type="button"
              aria-label={`Show denoising step ${index}`}
              aria-pressed={step === index}
              className={step === index ? styles.activeDot : ""}
              onClick={() => changeStep(index)}
            />
          ))}
        </div>
        <span>text · t = 0</span>
      </div>
      <div className={styles.tokenGrid} aria-live="polite">
        {tokenFrames[step].map((token, index) => (
          <div
            key={index}
            className={styles.tokenCell}
            style={
              {
                "--confidence": `${confidenceFrames[step][index]}%`,
              } as React.CSSProperties
            }
          >
            <span>{token}</span>
            <small>{confidenceFrames[step][index]}%</small>
          </div>
        ))}
      </div>
      <div className={styles.canvasControls}>
        <button
          type="button"
          onClick={() => changeStep(step - 1)}
          disabled={step === 0}
          aria-label="Previous denoising step"
        >
          <ChevronLeft aria-hidden="true" />
        </button>
        <button
          type="button"
          className={styles.playButton}
          onClick={() => {
            if (step === tokenFrames.length - 1) setStep(0);
            setPlaying((current) => !current);
          }}
        >
          {playing ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}
          {playing
            ? "Pause"
            : step === tokenFrames.length - 1
              ? "Replay"
              : "Play"}
        </button>
        <button
          type="button"
          onClick={() => changeStep(step + 1)}
          disabled={step === tokenFrames.length - 1}
          aria-label="Next denoising step"
        >
          <ChevronRight aria-hidden="true" />
        </button>
      </div>
      <p className={styles.canvasReadout}>
        Step {step} / {tokenFrames.length - 1} · mean confidence{" "}
        {Math.round(
          confidenceFrames[step].reduce((sum, value) => sum + value, 0) /
            confidenceFrames[step].length,
        )}
        %
      </p>
    </div>
  );
}

function EntropySampler() {
  const [budget, setBudget] = useState(42);
  const candidates = [
    ["A", 0.04],
    ["model", 0.08],
    ["can", 0.15],
    ["revise", 0.29],
    ["tokens", 0.51],
    ["together", 0.73],
  ] as const;
  const cutoff = budget / 100;

  return (
    <div className={styles.entropyLab} data-testid="entropy-sampler">
      <div className={styles.entropyHeader}>
        <label htmlFor="entropy-budget">Confidence budget</label>
        <output>{budget}%</output>
      </div>
      <input
        id="entropy-budget"
        type="range"
        min="10"
        max="85"
        value={budget}
        onChange={(event) => setBudget(Number(event.target.value))}
      />
      <div className={styles.entropyTokens}>
        {candidates.map(([token, entropy]) => {
          const committed = entropy <= cutoff;
          return (
            <div key={token} data-committed={committed}>
              <span>
                {committed ? (
                  <Check aria-hidden="true" />
                ) : (
                  <RefreshCcw aria-hidden="true" />
                )}
              </span>
              <strong>{token}</strong>
              <small>H={entropy.toFixed(2)}</small>
              <em>{committed ? "keep" : "re-noise"}</em>
            </div>
          );
        })}
      </div>
      <p>
        Low entropy means the model is already sure. Commit those positions;
        give uncertain ones another chance.
      </p>
    </div>
  );
}

function ArchitectureDiagram() {
  return (
    <div
      className={styles.architecture}
      aria-label="DiffusionGemma block generation architecture"
    >
      <div className={styles.contextLane}>
        <span>prompt + frozen blocks</span>
        <strong>causal encoder</strong>
        <em>KV cache</em>
      </div>
      <ArrowRight className={styles.archArrow} aria-hidden="true" />
      <div className={styles.canvasLane}>
        <span>current 256-token canvas</span>
        <strong>bidirectional decoder</strong>
        <em>read-only KV cache + self-conditioning</em>
        <div className={styles.loopArrow}>
          <RefreshCcw aria-hidden="true" /> iterate
        </div>
      </div>
      <ArrowRight className={styles.archArrow} aria-hidden="true" />
      <div className={styles.commitLane}>
        <span>clean canvas</span>
        <strong>encode + append</strong>
        <em>then start the next block</em>
      </div>
    </div>
  );
}

function AttentionMask({ bidirectional }: { bidirectional: boolean }) {
  const tokens = ["A", "B", "C", "D"];

  return (
    <div
      className={styles.maskMatrix}
      aria-label={
        bidirectional
          ? "Bidirectional attention matrix where every token can read every token"
          : "Causal attention matrix where each token can read only itself and earlier tokens"
      }
    >
      <span aria-hidden="true" />
      {tokens.map((token) => (
        <strong key={`column-${token}`}>{token}</strong>
      ))}
      {tokens.map((rowToken, row) => (
        <div key={`row-${rowToken}`} className={styles.maskMatrixRow}>
          <strong>{rowToken}</strong>
          {tokens.map((columnToken, column) => {
            const open = bidirectional || column <= row;
            return (
              <span
                key={`${rowToken}-${columnToken}`}
                data-open={open}
                aria-hidden="true"
              >
                {open ? "●" : ""}
              </span>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function PresenterChrome({ current }: { current: number }) {
  const goTo = useCallback((index: number) => {
    document.getElementById(`slide-${index + 1}`)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }, []);

  return (
    <>
      <aside className={styles.rail} aria-label="Slide navigation">
        <p>
          {current + 1} / {slideNames.length}
        </p>
        <div>
          {slideNames.map((name, index) => (
            <button
              key={name}
              type="button"
              className={current === index ? styles.activeRail : ""}
              onClick={() => goTo(index)}
              aria-label={`Go to slide ${index + 1}: ${name}`}
              title={`${String(index + 1).padStart(2, "0")} · ${name}`}
            />
          ))}
        </div>
      </aside>
      <div className={styles.presenterControls}>
        <button
          type="button"
          onClick={() => goTo(current - 1)}
          disabled={current === 0}
          aria-label="Previous slide"
        >
          <ChevronLeft aria-hidden="true" />
        </button>
        <button type="button" onClick={() => window.print()}>
          Save PDF
        </button>
        <button
          type="button"
          onClick={() => goTo(current + 1)}
          disabled={current === slideNames.length - 1}
          aria-label="Next slide"
        >
          <ChevronRight aria-hidden="true" />
        </button>
      </div>
    </>
  );
}

function usePresenterNavigation() {
  const [current, setCurrent] = useState(0);
  const currentRef = useRef(0);

  useEffect(() => {
    const slides = Array.from(
      document.querySelectorAll<HTMLElement>(
        `[data-testid="diffusiongemma-slide"]`,
      ),
    );
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (!visible) return;
        const index = slides.indexOf(visible.target as HTMLElement);
        if (index >= 0) {
          currentRef.current = index;
          setCurrent(index);
        }
      },
      { threshold: [0.45, 0.7] },
    );
    slides.forEach((slide) => observer.observe(slide));
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const nextKeys = new Set([
      " ",
      "Spacebar",
      "ArrowDown",
      "ArrowRight",
      "PageDown",
      "n",
      "N",
    ]);
    const previousKeys = new Set(["ArrowUp", "ArrowLeft", "PageUp", "p", "P"]);
    const onKeyDown = (event: KeyboardEvent) => {
      if (
        event.defaultPrevented ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        (event.target instanceof HTMLElement &&
          Boolean(
            event.target.closest("button, input, a, textarea, select"),
          )) ||
        document.querySelector('[data-testid="diffusiongemma-figure-lightbox"]')
      )
        return;

      let next = currentRef.current;
      if (event.shiftKey && (event.key === " " || event.key === "Spacebar"))
        next -= 1;
      else if (nextKeys.has(event.key)) next += 1;
      else if (previousKeys.has(event.key)) next -= 1;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = slideNames.length - 1;
      else return;

      event.preventDefault();
      next = Math.max(0, Math.min(slideNames.length - 1, next));
      document
        .getElementById(`slide-${next + 1}`)
        ?.scrollIntoView({ behavior: "smooth" });
    };
    document.addEventListener("keydown", onKeyDown, true);
    return () => document.removeEventListener("keydown", onKeyDown, true);
  }, []);

  return current;
}

export default function DiffusionGemmaPresentationPage() {
  const current = usePresenterNavigation();
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    setHydrated(true);
  }, []);

  return (
    <main
      className={styles.deck}
      data-testid="diffusiongemma-presentation"
      data-ready={hydrated}
    >
      <PresenterChrome current={current} />

      <DeckSequence>
        <TitleSlide number={1} />

        <Slide
          number={2}
          section="01 · Motivation"
          title="They moved the speed–capability frontier"
          minutes="3 min"
          tone="paper"
          wide
        >
          <div className={styles.frontierLayout}>
            <PaperFigure
              src={`${assetBase}/figure-1-speed-frontier.png`}
              alt="Paper figure plotting reasoning and coding accuracy against output speed, with DiffusionGemma far to the right of Gemma 4 with multi-token prediction and earlier diffusion models"
              width={1000}
              height={634}
              caption="Paper Figure 1 · quality versus output speed"
              eager
            />
            <div className={styles.frontierClaim}>
              <p className={styles.frontierNumber}>≈1,500</p>
              <p className={styles.frontierUnit}>output tokens/s</p>
              <BigIdea>
                Faster than the paper&apos;s AR + MTP and prior diffusion
                points— while retaining a competitive hard-task score.
              </BigIdea>
              <ul>
                <li>One H100, FP8, batch size 1 for Gemma comparisons</li>
                <li>Quality = mean of GPQA-Diamond and LiveCodeBench-v6</li>
                <li>Below Gemma 4 26B A4B quality, but on a new frontier</li>
              </ul>
            </div>
          </div>
        </Slide>

        <Slide
          number={3}
          section="01 · Motivation"
          title="One forward pass, many useful tokens"
          minutes="2 min"
          tone="signal"
        >
          <div className={styles.thesisLayout}>
            <BigIdea>
              DiffusionGemma trades <em>serial token decisions</em> for a small
              number of
              <em> parallel canvas revisions</em>.
            </BigIdea>
            <div className={styles.metricRow}>
              <Metric value="256" label="tokens per canvas" />
              <Metric value="≈12" label="effective denoising steps" />
              <Metric value="19.74" label="tokens per forward pass" />
              <Metric value="1,479" label="output tokens/s on H100" />
            </div>
            <Note>
              That is a speed–quality trade, not a free lunch: the reported
              model is faster but generally weaker than its Gemma 4 AR starting
              point.
            </Note>
          </div>
        </Slide>

        <Slide
          number={0}
          section="Intro"
          title="≈1,500 tokens/s is the fastest class of serving speed"
          tone="paper"
          wide
        >
          <div className={styles.benchmarkFull}>
            <PaperFigure
              src={`${assetBase}/artificial-analysis-output-speed-full.png`}
              alt="Full Artificial Analysis output-speed leaderboard export, ranging from Celeris-1 at 1513 output tokens per second through the slower models"
              width={4640}
              height={1824}
              caption="Artificial Analysis · Output Speed · export dated 16 Aug 2026"
              eager
            />
          </div>
          <Note>
            DiffusionGemma reports 1,479 output tokens/s in its H100 setup—near
            the very fastest systems shown here. Hardware and serving stacks
            differ, so this is a scale reference rather than a controlled
            comparison.
          </Note>
        </Slide>

        <Slide
          number={0}
          section="Intro"
          title="The fastest models are not the smartest models"
          wide
        >
          <div className={styles.benchmarkFull}>
            <PaperFigure
              src={`${assetBase}/artificial-analysis-intelligence-full.png`}
              alt="Full Artificial Analysis Intelligence Index with the two Gemma 4 26B A4B entries, Mercury 2, and Celeris-1 highlighted near the right side of the chart"
              width={4640}
              height={2032}
              caption="Artificial Analysis Intelligence Index v4.1.1 · 16 Aug 2026"
              highlights={[
                {
                  label: "Gemma 4 26B A4B reasoning",
                  left: "72.84%",
                  top: "46.75%",
                  width: "3.77%",
                  height: "47.74%",
                  tone: "mint",
                },
                {
                  label: "Mercury 2",
                  left: "88.9%",
                  top: "49.7%",
                  width: "3.34%",
                  height: "44.78%",
                },
                {
                  label: "Gemma 4 26B A4B non-reasoning",
                  left: "92.13%",
                  top: "51.67%",
                  width: "3.45%",
                  height: "42.81%",
                  tone: "mint",
                },
                {
                  label: "Celeris-1",
                  left: "95.26%",
                  top: "55.61%",
                  width: "3.77%",
                  height: "38.88%",
                },
              ]}
            />
          </div>
          <Note>
            The highlighted Gemma 4 entries score above Mercury 2 and Celeris-1.
            DiffusionGemma itself is not scored in this export; the paper&apos;s
            Figure 1 provides that direct quality–speed comparison.
          </Note>
        </Slide>

        <Slide number={0} section="Intro" title="Today's route" tone="signal">
          <div className={styles.agendaGrid}>
            {[
              [
                "01",
                "Motivation",
                "Why token-by-token serving wastes the pass",
              ],
              [
                "02",
                "From Gemma 4 to DiffusionGemma",
                "What the parent model is—and what actually changed",
              ],
              ["03", "How diffusion writes", "Canvas, noise, revision, commit"],
              [
                "04",
                "Training",
                "SFT, sampler distillation, reinforcement learning",
              ],
              ["05", "Evidence", "Speed, capability, self-correction, limits"],
              [
                "06",
                "Synthesis",
                "What is genuinely new and where it may matter",
              ],
            ].map(([index, title, description]) => (
              <div key={index} data-testid="agenda-item">
                <span>{index}</span>
                <strong>{title}</strong>
                <p>{description}</p>
              </div>
            ))}
          </div>
        </Slide>

        <SectionDivider
          number={0}
          index="01"
          title="Motivation"
          subtitle="Why a large language model can know the answer—and still spend most of its time moving weights."
        />

        <Slide
          number={4}
          section="01 · Motivation"
          title="Every answer is trapped in a dependency chain"
          minutes="4 min"
        >
          <div className={styles.dependencyStage}>
            <p className={styles.eyebrow}>Autoregressive decoding</p>
            <div className={styles.tokenChain}>
              {[
                "The",
                "model",
                "writes",
                "one",
                "token",
                "at",
                "a",
                "time",
              ].map((token, index) => (
                <span key={token}>
                  {token}
                  {index < 7 ? <ArrowRight aria-hidden="true" /> : null}
                </span>
              ))}
            </div>
            <BigIdea>
              Token <em>n + 1</em> cannot start until token <em>n</em> exists.
            </BigIdea>
            <p className={styles.stageCaption}>
              Training can score many positions together. Serving commits them
              one after another.
            </p>
          </div>
        </Slide>

        <Slide
          number={0}
          section="01 · Motivation"
          title="One giant weight read can buy one tiny token"
          tone="signal"
        >
          <div className={styles.amortizationStage}>
            <div className={styles.weightBlock}>
              <Cpu aria-hidden="true" />
              <strong>25.2B parameters</strong>
              <span>move the active weights through the accelerator</span>
            </div>
            <ArrowRight aria-hidden="true" />
            <div className={styles.oneToken}>
              <span>next token</span>
              <strong>“the”</strong>
            </div>
          </div>
          <BigIdea>
            At batch size 1, autoregression is often limited by memory
            bandwidth, not arithmetic.
          </BigIdea>
        </Slide>

        <Slide
          number={0}
          section="01 · Motivation"
          title="A committed token cannot be rewritten"
          tone="paper"
        >
          <div className={styles.rewriteStage}>
            <div>
              <span>AR stream</span>
              <div className={styles.committedSentence}>
                <b>The answer is</b>
                <b className={styles.wrongToken}>yes</b>
                <span className={styles.lockedToken}>locked</span>
              </div>
              <p>Correction must arrive later in the sequence.</p>
            </div>
            <div>
              <span>Diffusion canvas</span>
              <div className={styles.revisableSentence}>
                <b>The answer is</b>
                <b className={styles.wrongToken}>yes</b>
                <ArrowRight aria-hidden="true" />
                <b className={styles.rightToken}>no</b>
              </div>
              <p>Every position remains provisional until the block commits.</p>
            </div>
          </div>
        </Slide>

        <Slide
          number={5}
          section="01 · Motivation"
          title="Existing speedups still keep the chain"
          minutes="4 min"
          tone="paper"
        >
          <div className={styles.shortcutGrid}>
            <div>
              <span>Batching</span>
              <strong>Fill the GPU with more users</strong>
              <p>More throughput—not lower single-request dependency.</p>
            </div>
            <div>
              <span>Speculation + MTP</span>
              <strong>Draft several tokens, then verify</strong>
              <p>
                Several tokens per pass, but an AR verifier still commits them.
              </p>
            </div>
            <div>
              <span>Earlier text diffusion</span>
              <strong>Parallelize, but lose the frontier</strong>
              <p>
                The idea was parallel; the previous systems were not fast
                enough.
              </p>
            </div>
          </div>
        </Slide>

        <Slide
          number={0}
          section="01 · Motivation"
          title="The target is useful tokens per forward pass"
          tone="signal"
        >
          <div className={styles.passYieldStage}>
            <div>
              <span>Autoregressive</span>
              <strong>1</strong>
              <small>useful token / pass</small>
            </div>
            <div>
              <span>Speculation + MTP</span>
              <strong>≈3–6</strong>
              <small>accepted tokens / pass</small>
            </div>
            <div className={styles.passYieldWinner}>
              <span>DiffusionGemma</span>
              <strong>19.74</strong>
              <small>output tokens / pass</small>
            </div>
          </div>
          <BigIdea>
            Amortize the expensive model pass across a revisable block—without
            pretraining another 26B model.
          </BigIdea>
        </Slide>

        <SectionDivider
          number={0}
          index="02"
          title="From Gemma 4 to DiffusionGemma"
          subtitle="Keep the language knowledge. Teach the same Transformer a second attention mode and a new denoising job."
        />

        <Slide
          number={0}
          section="02 · From Gemma 4 to DiffusionGemma"
          title="The shortest mental model: one Transformer, two modes"
          tone="signal"
        >
          <div className={styles.sharedBackboneStory}>
            <div className={styles.backboneCore}>
              <Layers3 aria-hidden="true" />
              <span>same pretrained weights θ</span>
              <strong>Gemma 4 backbone</strong>
            </div>
            <div className={styles.backboneModes}>
              <div>
                <span>context mode</span>
                <strong>Causal attention</strong>
                <p>Prompt + finished blocks → KV cache</p>
              </div>
              <div>
                <span>diffusion mode</span>
                <strong>Bidirectional attention</strong>
                <p>Noisy 256-token canvas → all clean-token predictions</p>
              </div>
            </div>
          </div>
          <BigIdea>
            “Encoder” and “decoder” are two jobs performed by the same Gemma
            backbone—not two separately pretrained networks.
          </BigIdea>
        </Slide>

        <Slide
          number={0}
          section="02 · From Gemma 4 to DiffusionGemma"
          title="Five steps turn Gemma 4 into DiffusionGemma"
        >
          <ol
            className={styles.conversionRoadmap}
            data-testid="conversion-roadmap"
          >
            <li>
              <span>01</span>
              <strong>Reuse the trained checkpoint</strong>
              <p>Start with Gemma&apos;s language knowledge.</p>
            </li>
            <li>
              <span>02</span>
              <strong>Open attention inside the canvas</strong>
              <p>Keep history causal; make the live block bidirectional.</p>
            </li>
            <li>
              <span>03</span>
              <strong>Teach all-position denoising</strong>
              <p>Recover every clean token from a corrupted block.</p>
            </li>
            <li>
              <span>04</span>
              <strong>Feed the last clean guess back</strong>
              <p>Add a tiny self-conditioning path.</p>
            </li>
            <li>
              <span>05</span>
              <strong>Compress many refinements into a few</strong>
              <p>Use sampler distillation plus RL.</p>
            </li>
          </ol>
        </Slide>

        <Slide
          number={0}
          section="02 · From Gemma 4 to DiffusionGemma"
          title="Step 1 · Start from a model that already knows language"
          tone="paper"
        >
          <div className={styles.conversionFigure}>
            <PaperFigure
              src={`${assetBase}/figure-2-training-pipeline-large.png`}
              alt="Paper figure showing Gemma 4 converted to DiffusionGemma by supervised fine-tuning followed by sampler distillation and reinforcement learning"
              width={1515}
              height={255}
              caption="Paper Figure 2 · two-stage conversion pipeline"
            />
          </div>
          <div className={styles.warmStartStage}>
            <div>
              <span>finished checkpoint</span>
              <strong>Gemma 4 · 26B A4B</strong>
              <p>Already knows language, code, reasoning, and instructions.</p>
            </div>
            <ArrowRight aria-hidden="true" />
            <div>
              <span>copy the weights</span>
              <strong>Diffusion initialization</strong>
              <p>No second large backbone and no diffusion pretraining run.</p>
            </div>
            <ArrowRight aria-hidden="true" />
            <div>
              <span>before SFT</span>
              <strong>Language model, not yet a denoiser</strong>
              <p>
                The knowledge is present; the new task still has to be taught.
              </p>
            </div>
          </div>
        </Slide>

        <Slide
          number={0}
          section="02 · From Gemma 4 to DiffusionGemma"
          title="Step 2 · Change who each canvas token can see"
          tone="paper"
        >
          <div className={styles.maskShiftStage}>
            <div>
              <span>prompt + completed history</span>
              <strong>Causal mask</strong>
              <AttentionMask bidirectional={false} />
              <p>Each position reads only itself and earlier positions.</p>
            </div>
            <div>
              <span>current 256-token canvas</span>
              <strong>Bidirectional mask</strong>
              <AttentionMask bidirectional />
              <p>
                Every position can use every other position in the live block.
              </p>
            </div>
          </div>
          <BigIdea>
            The attention heads are reused. The mask changes which connections
            are allowed.
          </BigIdea>
        </Slide>

        <Slide
          number={0}
          section="02 · From Gemma 4 to DiffusionGemma"
          title="Steps 3–4 · Learn the whole clean block—and reuse the last guess"
          tone="signal"
        >
          <div className={styles.denoisingObjective}>
            <div className={styles.objectivePipeline}>
              <div>
                <span>clean training block</span>
                <strong>The model revises every token together</strong>
              </div>
              <ArrowRight aria-hidden="true" />
              <div className={styles.noisyObjective}>
                <span>corrupt at a random noise level</span>
                <strong>river · cat · 93 · France · …</strong>
              </div>
              <ArrowRight aria-hidden="true" />
              <div>
                <span>SFT target</span>
                <strong>Predict all 256 original tokens</strong>
              </div>
            </div>
            <div className={styles.selfConditioningLoop}>
              <RefreshCcw aria-hidden="true" />
              <div>
                <span>small addition · 7.8M parameters</span>
                <strong>
                  Previous clean-token guess → next refinement pass
                </strong>
                <p>
                  A lightweight self-conditioning network embeds the last
                  prediction so the next pass can improve it.
                </p>
              </div>
            </div>
          </div>
          <Note>
            SFT repurposes Gemma&apos;s language knowledge: instead of
            predicting one next token, recover the clean token at every canvas
            position.
          </Note>
        </Slide>

        <Slide
          number={0}
          section="02 · From Gemma 4 to DiffusionGemma"
          title="At inference, the trained model alternates the two modes"
          tone="paper"
          wide
        >
          <div className={styles.overviewFigureLayout}>
            <PaperFigure
              src={`${assetBase}/figure-4-generation-pipeline-large.png`}
              alt="Paper diagram showing prompt encoding, a bidirectional 256-token denoising loop, and encode-and-append before the next canvas"
              width={1480}
              height={665}
              caption="Paper Figure 4 · DiffusionGemma generation pipeline"
            />
            <ol className={styles.loopLegend}>
              <li>
                <span>1</span>
                <strong>Same θ · causal mode</strong>
                <p>Prompt and completed blocks write the shared KV cache.</p>
              </li>
              <li>
                <span>2</span>
                <strong>Same θ · diffusion mode</strong>
                <p>
                  The bidirectional canvas repeatedly predicts clean tokens,
                  conditioned on that cache.
                </p>
              </li>
              <li>
                <span>3</span>
                <strong>Causal mode again</strong>
                <p>The clean block enters the cache; a new canvas starts.</p>
              </li>
            </ol>
          </div>
        </Slide>

        <Slide
          number={0}
          section="02 · From Gemma 4 to DiffusionGemma"
          title="Step 5 · Make a capable denoiser fast"
          tone="signal"
        >
          <div className={styles.distillationStage}>
            <div>
              <span>after diffusion SFT</span>
              <strong>Capable with many refinement steps</strong>
              <p>
                Quality collapses if this model is simply forced to stop early.
              </p>
            </div>
            <ArrowRight aria-hidden="true" />
            <div className={styles.distillationCore}>
              <span>sampler distillation + RL</span>
              <strong>Teach the few-step trajectory</strong>
              <p>
                Imitate a stronger sampler while directly rewarding answers.
              </p>
            </div>
            <ArrowRight aria-hidden="true" />
            <div>
              <span>final sampler</span>
              <strong>≈12 effective passes on average</strong>
              <p>Parallel token predictions now become the headline speedup.</p>
            </div>
          </div>
          <BigIdea>
            SFT teaches the task. SD·RL teaches the task under a tight inference
            budget.
          </BigIdea>
        </Slide>

        <Slide
          number={0}
          section="02 · From Gemma 4 to DiffusionGemma"
          title="The result: diffusion within blocks, autoregression between blocks"
          tone="paper"
          wide
        >
          <ArchitectureDiagram />
          <div className={styles.blockScaleStage}>
            <div>
              <span>macro level</span>
              <strong>Prompt → block 1 → block 2 → block 3</strong>
              <p>Completed 256-token blocks are appended left-to-right.</p>
            </div>
            <div>
              <span>inside one block</span>
              <strong>token 1 ↔ token 2 ↔ … ↔ token 256</strong>
              <p>
                The live canvas is revised in parallel for roughly 12 passes.
              </p>
            </div>
          </div>
          <Note>
            The core weights remain Gemma-compatible: the trained checkpoint can
            still run with ordinary causal autoregressive decoding.
          </Note>
        </Slide>

        <SectionDivider
          number={0}
          index="03"
          title="How diffusion writes"
          subtitle="A noisy 256-token canvas becomes a sentence through repeated, parallel revision."
          motif={
            <>
              <span>noise</span>
              <ArrowRight />
              <span>revise together</span>
              <ArrowRight />
              <span>commit block</span>
            </>
          }
        />

        <Slide
          number={10}
          section="03 · Mechanics"
          title="A conveyor belt versus a drafting table"
          minutes="3 min"
          tone="paper"
        >
          <div className={styles.passComparison}>
            <ModelPass
              label="Autoregressive · one forward pass"
              accent="violet"
              tokens={["The", "model", "writes", "→", "one"]}
            />
            <ModelPass
              label="Diffusion · one forward pass"
              accent="amber"
              tokens={[
                "A",
                "model",
                "can",
                "revise",
                "many",
                "tokens",
                "at",
                "once",
              ]}
            />
          </div>
          <BigIdea>
            The diffusion model is allowed to be wrong early—because early
            guesses are provisional.
          </BigIdea>
        </Slide>

        <Slide
          number={11}
          section="03 · Mechanics"
          title="Corrupt deliberately. Learn to undo it."
          minutes="3 min"
          tone="signal"
        >
          <div className={styles.diffusionSteps}>
            <div>
              <span>clean</span>
              <strong>red sunset</strong>
              <small>data distribution</small>
            </div>
            <ArrowRight aria-hidden="true" />
            <div>
              <span>partly noisy</span>
              <strong>bright horizon</strong>
              <small>mixed probability mass</small>
            </div>
            <ArrowRight aria-hidden="true" />
            <div>
              <span>noise</span>
              <strong>blue moon</strong>
              <small>near-uniform tokens</small>
            </div>
          </div>
          <div className={styles.reversalCallout}>
            <RefreshCcw aria-hidden="true" />
            <span>Training observes this direction.</span>
            <strong>Generation learns to run it backward.</strong>
          </div>
        </Slide>

        <Slide
          number={12}
          section="03 · Mechanics"
          title="The object is a distribution—not a fading sentence"
          minutes="3 min"
        >
          <PaperFigure
            src={`${assetBase}/figure-3-discrete-path.png`}
            alt="Paper figure showing probability mass moving from uniform adjective-noun noise toward data modes while one sampled trajectory jumps between token pairs"
            width={720}
            height={225}
            caption="Paper Figure 3 · probability path and one sampled trajectory"
          />
          <div className={styles.legendRow}>
            <span>
              <i className={styles.softDot} /> colored clouds = probability mass
            </span>
            <span>
              <i className={styles.hardDot} /> outlined points = one sampled
              sequence
            </span>
          </div>
          <Note>
            A discrete token cannot slide smoothly from “blue” to “red.” The
            probabilities move smoothly; an actual token jumps.
          </Note>
        </Slide>

        <Slide
          number={13}
          section="03 · Mechanics"
          title="Choose a route; learn the local moves"
          minutes="4 min"
          tone="paper"
        >
          <div className={styles.flowBridge}>
            <div>
              <span>1</span>
              <strong>Pick endpoints</strong>
              <p>easy noise ↔ real data</p>
            </div>
            <ArrowRight aria-hidden="true" />
            <div>
              <span>2</span>
              <strong>Define a path</strong>
              <p>how probability should move</p>
            </div>
            <ArrowRight aria-hidden="true" />
            <div>
              <span>3</span>
              <strong>Learn local motion</strong>
              <p>what move to make at time t</p>
            </div>
          </div>
          <div className={styles.bridgeNote}>
            <div>
              <GitBranch aria-hidden="true" />
              <strong>Continuous flow matching</strong>
              <p>Learn a velocity vector in Euclidean space.</p>
            </div>
            <div>
              <Shuffle aria-hidden="true" />
              <strong>Discrete flow matching here</strong>
              <p>Learn token-jump probabilities in a categorical space.</p>
            </div>
          </div>
          <Note>
            Same transport idea, different geometry. You do not need the
            continuous ODE derivation to follow this paper.
          </Note>
        </Slide>

        <Slide
          number={14}
          section="03 · Mechanics"
          title="At time t, keep—or replace uniformly"
          minutes="3 min"
        >
          <div className={styles.equationPanel}>
            <MathText
              text={String.raw`$$q(x_t^i\mid x_0^i)=\kappa_t\,\delta(x_t^i,x_0^i)+(1-\kappa_t)\frac{1}{V}$$`}
            />
          </div>
          <div className={styles.equationLegend}>
            <div>
              <strong>κₜ</strong>
              <p>probability that position i keeps its clean token</p>
            </div>
            <div>
              <strong>δ</strong>
              <p>1 when the noisy and clean tokens match</p>
            </div>
            <div>
              <strong>1 / V</strong>
              <p>uniform draw from the 262k-token vocabulary</p>
            </div>
          </div>
          <div className={styles.timeWarning}>
            <TriangleAlert aria-hidden="true" />
            <p>
              <strong>Paper convention:</strong> t = 0 is clean; t = 1 is noise.
              Some flow-matching courses use the reverse convention.
            </p>
          </div>
        </Slide>

        <Slide
          number={15}
          section="03 · Mechanics"
          title="The network predicts the clean token behind the noise"
          minutes="3 min"
          tone="signal"
        >
          <div className={styles.posteriorFlow}>
            <div>
              <span>observed canvas</span>
              <strong>xₜ</strong>
              <small>noisy tokens</small>
            </div>
            <ArrowRight aria-hidden="true" />
            <div className={styles.networkBox}>
              <BrainCircuit aria-hidden="true" />
              <strong>Transformer θ</strong>
              <small>prompt + canvas + time</small>
            </div>
            <ArrowRight aria-hidden="true" />
            <div>
              <span>clean posterior</span>
              <strong>pθ(x₀ | xₜ)</strong>
              <small>distribution at every position</small>
            </div>
            <ArrowRight aria-hidden="true" />
            <div>
              <span>Step</span>
              <strong>xₜ₋Δₜ</strong>
              <small>slightly cleaner sample</small>
            </div>
          </div>
          <BigIdea>
            The same prediction target—“what clean token belongs here?”—works at
            every noise level.
          </BigIdea>
        </Slide>

        <Slide
          number={16}
          section="03 · Mechanics"
          title="Watch a token canvas find a sentence"
          minutes="4 min"
          wide
        >
          <TokenCanvas />
          <div className={styles.canvasTakeaways}>
            <span>positions update in parallel</span>
            <span>tokens may change more than once</span>
            <span>confidence rises unevenly</span>
          </div>
          <Note>
            Teaching simulation: the paper’s real canvas has 256 positions and
            samples from learned categorical distributions.
          </Note>
        </Slide>

        <Slide
          number={17}
          section="03 · Mechanics"
          title="Gemma’s attention is rewired around the canvas"
          minutes="3 min"
        >
          <div className={styles.attentionGrid}>
            <div>
              <Layers3 aria-hidden="true" />
              <strong>Causal encoder</strong>
              <p>Turns prompt and finalized blocks into a reusable KV cache.</p>
            </div>
            <div>
              <Sparkles aria-hidden="true" />
              <strong>Bidirectional decoder</strong>
              <p>
                Every canvas position can inspect every other current position.
              </p>
            </div>
            <div>
              <GitBranch aria-hidden="true" />
              <strong>Cross-attention</strong>
              <p>Conditions the current draft on the frozen history.</p>
            </div>
            <div>
              <RefreshCcw aria-hidden="true" />
              <strong>Self-conditioning</strong>
              <p>
                Feeds the prior clean-token estimate into the next revision.
              </p>
            </div>
          </div>
          <Note>
            The weights are shared and initialized from Gemma 4 26B A4B MoE:
            25.2B total parameters, 3.8B activated.
          </Note>
        </Slide>

        <Slide
          number={18}
          section="03 · Mechanics"
          title="Why random-token noise instead of [MASK]?"
          minutes="2 min"
          tone="signal"
        >
          <div className={styles.maskContrast}>
            <div>
              <span>[MASK] diffusion</span>
              <p>Often reveals positions monotonically.</p>
              <strong>“Blank → final”</strong>
            </div>
            <div>
              <span>Multinomial diffusion</span>
              <p>Any position can keep changing.</p>
              <strong>“Guess → inspect → revise”</strong>
            </div>
          </div>
          <BigIdea>
            Revision is the feature—but simultaneous samples cannot see one
            another’s newest choices until the next step.
          </BigIdea>
          <Note>
            That factorization can create local inconsistencies.
            Self-conditioning and the sampler are the paper’s answer.
          </Note>
        </Slide>

        <Slide
          number={19}
          section="03 · Mechanics"
          title="Commit the easy tokens; recycle uncertainty"
          minutes="4 min"
          wide
        >
          <EntropySampler />
          <div className={styles.samplerRecipe}>
            <span>1 · rank positions by entropy</span>
            <span>2 · keep a low-entropy budget</span>
            <span>3 · re-noise the rest</span>
          </div>
        </Slide>

        <Slide
          number={20}
          section="03 · Mechanics"
          title="Stop when confidence and text both stabilize"
          minutes="2 min"
          tone="paper"
        >
          <div className={styles.stopRule}>
            <div>
              <CircleGauge aria-hidden="true" />
              <span>mean entropy</span>
              <strong>≤ 0.005</strong>
            </div>
            <span>AND</span>
            <div>
              <Check aria-hidden="true" />
              <span>argmax sequence</span>
              <strong>unchanged twice</strong>
            </div>
            <ArrowRight aria-hidden="true" />
            <div className={styles.stop}>
              <Pause aria-hidden="true" />
              <strong>stop</strong>
            </div>
          </div>
          <div className={styles.metricRow}>
            <Metric value="48" label="maximum steps" />
            <Metric value="≈12" label="average effective steps" />
            <Metric value="4×" label="latency reduction vs max" />
          </div>
        </Slide>

        <SectionDivider
          number={0}
          index="04"
          title="Training"
          subtitle="First learn to denoise at any noise level. Then compress the sampler into the few-step regime that creates the speedup."
          motif={
            <>
              <span>SFT</span>
              <ArrowRight />
              <span>sampler distillation</span>
              <span>+</span>
              <span>RL</span>
            </>
          }
        />

        <Slide
          number={22}
          section="04 · Training"
          title="SFT teaches denoising at every noise level"
          minutes="3 min"
          tone="signal"
        >
          <div className={styles.trainingLoop}>
            <div>
              <span>1</span>
              <p>
                sample <strong>t ∼ Uniform(0,1)</strong>
              </p>
            </div>
            <ArrowRight aria-hidden="true" />
            <div>
              <span>2</span>
              <p>
                replace each canvas token with probability <strong>t</strong>
              </p>
            </div>
            <ArrowRight aria-hidden="true" />
            <div>
              <span>3</span>
              <p>
                cross-entropy against <strong>every clean token</strong>
              </p>
            </div>
          </div>
          <div className={styles.equationPanel}>
            <MathText
              text={String.raw`$$\mathcal{L}_{\mathrm{SFT}}=-\sum_i\log p_\theta(x_0^i\mid x_t,z_t,H)$$`}
            />
          </div>
          <Note>
            SFT creates a capable many-step denoiser. It does not yet make a
            strong few-step reasoner.
          </Note>
        </Slide>

        <Slide
          number={23}
          section="04 · Training"
          title="Distill the sampler while optimizing the answer"
          minutes="4 min"
          tone="paper"
        >
          <div className={styles.teacherStudent}>
            <div>
              <span>online teacher</span>
              <strong>many mild-temperature steps</strong>
              <p>produces high-quality trajectories</p>
            </div>
            <ArrowRight aria-hidden="true" />
            <div className={styles.student}>
              <span>DiffusionGemma</span>
              <strong>few denoising steps</strong>
              <p>imitate the path + maximize reward</p>
            </div>
          </div>
          <div className={styles.dualObjective}>
            <div>
              <Zap aria-hidden="true" />
              <strong>Sampler distillation</strong>
              <p>Compress a good long trajectory into short transitions.</p>
            </div>
            <div>
              <BrainCircuit aria-hidden="true" />
              <strong>Reinforcement learning</strong>
              <p>
                Keep the final result useful on reasoning and coding rewards.
              </p>
            </div>
          </div>
        </Slide>

        <Slide
          number={24}
          section="04 · Training"
          title="Confidence creates its own curriculum"
          minutes="3 min"
        >
          <PaperFigure
            src={`${assetBase}/figure-9-pareto.png`}
            alt="Paper figure showing sampler distillation and reinforcement learning extending the quality-speed Pareto frontier"
            width={710}
            height={250}
            caption="Paper Figure 9 · quality–speed frontier after SD·RL"
          />
          <div className={styles.causalChain}>
            <span>better model</span>
            <ArrowRight aria-hidden="true" />
            <span>lower entropy</span>
            <ArrowRight aria-hidden="true" />
            <span>earlier stopping</span>
            <ArrowRight aria-hidden="true" />
            <span>harder few-step training</span>
          </div>
          <Note>
            The combined GPQA + LiveCodeBench score rises about 10 points while
            tokens per forward pass nearly quadruple from 5 to 20.
          </Note>
        </Slide>

        <SectionDivider
          number={0}
          index="05"
          title="Evidence"
          subtitle="The headline is real—but the speed number, quality gap, batch regime, and within-canvas behavior have to be read together."
          motif={
            <>
              <span>7.1× throughput</span>
              <span>↔</span>
              <span>measurable quality cost</span>
            </>
          }
        />

        <Slide
          number={25}
          section="05 · Evidence"
          title="Tokens per pass is not tokens per second"
          minutes="4 min"
          tone="signal"
        >
          <div className={styles.speedEquation}>
            <MathText
              text={String.raw`$$\text{throughput gain}\;\approx\;\frac{\text{tokens per forward pass}}{\text{relative cost per diffusion pass}}$$`}
            />
          </div>
          <div className={styles.costBalance}>
            <div>
              <Cpu aria-hidden="true" />
              <strong>≈19.74 tokens / pass</strong>
              <p>
                Useful output amortized over denoising + block-append passes.
              </p>
            </div>
            <div className={styles.divide}>÷</div>
            <div>
              <Clock3 aria-hidden="true" />
              <strong>≈3.2× pass cost</strong>
              <p>A diffusion step processes the entire 256-token canvas.</p>
            </div>
            <div className={styles.equals}>=</div>
            <div>
              <Zap aria-hidden="true" />
              <strong>≈7.1× throughput</strong>
              <p>1,456–1,479 TPS versus 204 TPS in the paper setup.</p>
            </div>
          </div>
          <Note>
            The advantage is strongest at low batch size. Around 32 concurrent
            requests, batched AR catches up.
          </Note>
        </Slide>

        <Slide
          number={26}
          section="05 · Evidence"
          title="A new speed frontier—with an accuracy bill"
          minutes="4 min"
          wide
        >
          <div className={styles.resultsLayout}>
            <PaperFigure
              src={`${assetBase}/figure-13-results.png`}
              alt="Paper figure comparing DiffusionGemma and baselines on reasoning, coding, instruction following, and output speed"
              width={720}
              height={585}
              caption="Paper Figure 13 · capability-area means and output speed"
            />
            <div className={styles.resultsTable}>
              <div>
                <span>Metric</span>
                <span>DiffGem TD</span>
                <span>Gemma 4 AR</span>
              </div>
              <div>
                <strong>Output TPS</strong>
                <strong className={styles.win}>1,479</strong>
                <strong>204</strong>
              </div>
              <div>
                <strong>AIME 2026</strong>
                <strong>69.1</strong>
                <strong className={styles.win}>84.2</strong>
              </div>
              <div>
                <strong>GPQA</strong>
                <strong>73.2</strong>
                <strong className={styles.win}>79.8</strong>
              </div>
              <div>
                <strong>IFEval</strong>
                <strong className={styles.win}>97.4</strong>
                <strong>97.2</strong>
              </div>
            </div>
          </div>
          <Note>
            Compare frontier positions, not isolated headline numbers. Hardware,
            batch size, output length, and decoding mode all matter.
          </Note>
        </Slide>

        <Slide
          number={27}
          section="05 · Evidence"
          title="Bidirectional attention can revise the premise"
          minutes="3 min"
          tone="paper"
          wide
        >
          <div className={styles.correctionLayout}>
            <PaperFigure
              src={`${assetBase}/figure-23-frog-trace.png`}
              alt="Paper figure showing a frog puzzle answer changing from Yes to No during six denoising steps"
              width={720}
              height={775}
              caption="Paper Figure 23 · Yes becomes No over six denoising steps"
            />
            <div>
              <p className={styles.eyebrow}>The useful distinction</p>
              <BigIdea>
                AR can self-correct in later text. Diffusion can correct an
                earlier token before the block is committed.
              </BigIdea>
              <Note>
                This is evidence of within-canvas revision—not proof of globally
                better reasoning.
              </Note>
            </div>
          </div>
        </Slide>

        <Slide
          number={28}
          section="05 · Evidence"
          title="Four reasons to stay skeptical"
          minutes="4 min"
        >
          <div className={styles.limitGrid}>
            <div>
              <span>01</span>
              <strong>Quality gap</strong>
              <p>
                Most reasoning and coding scores trail the AR initialization.
              </p>
            </div>
            <div>
              <span>02</span>
              <strong>Frozen blocks</strong>
              <p>Revision stops at each 256-token boundary.</p>
            </div>
            <div>
              <span>03</span>
              <strong>Local inconsistency</strong>
              <p>
                Parallel samples see the old canvas, not each other’s new
                choices.
              </p>
            </div>
            <div>
              <span>04</span>
              <strong>Failure modes</strong>
              <p>
                Conciseness, repetition loops, and multimodal thought-tag
                omissions remain.
              </p>
            </div>
          </div>
          <Note>
            DiffusionGemma is an experimental open-weight checkpoint and a
            compelling systems result—not yet a universal replacement for AR
            decoding.
          </Note>
        </Slide>

        <SectionDivider
          number={0}
          index="06"
          title="Synthesis"
          subtitle="A causal LLM backbone, turned into a blockwise diffusion generator, moves the low-batch speed frontier without preserving every point of quality."
          motif={
            <>
              <span>shared Gemma 4 weights</span>
              <ArrowRight />
              <span>parallel revision</span>
              <ArrowRight />
              <span>≈1,500 tok/s</span>
            </>
          }
        />

        <Slide
          number={29}
          section="06 · Synthesis"
          title="The whole paper in one loop"
          minutes="2 min"
          tone="signal"
        >
          <div className={styles.summaryLoop}>
            <div>
              <span>1</span>
              <strong>Noise a 256-token canvas</strong>
            </div>
            <ArrowRight aria-hidden="true" />
            <div>
              <span>2</span>
              <strong>Predict clean-token posteriors</strong>
            </div>
            <ArrowRight aria-hidden="true" />
            <div>
              <span>3</span>
              <strong>Commit certainty; recycle entropy</strong>
            </div>
            <ArrowRight aria-hidden="true" />
            <div>
              <span>4</span>
              <strong>Append the block and continue</strong>
            </div>
          </div>
          <div className={styles.takeaways}>
            <p>
              <strong>Conceptual contribution</strong> Discrete diffusion as
              revisable parallel text generation.
            </p>
            <p>
              <strong>Engineering contribution</strong> A Gemma warm start plus
              blockwise attention and adaptive sampling.
            </p>
            <p>
              <strong>Training contribution</strong> SD·RL makes the few-step
              regime usable.
            </p>
            <p>
              <strong>Result</strong> Far higher low-batch throughput, with
              measurable quality tradeoffs.
            </p>
          </div>
        </Slide>

        <Slide
          number={30}
          section="06 · Synthesis"
          title="What would convince you to deploy it?"
          tone="paper"
        >
          <div className={styles.discussionGrid}>
            <p>
              <span>01</span>Is “revision within a block” a qualitatively
              different reasoning primitive—or just parallel decoding?
            </p>
            <p>
              <span>02</span>Which workloads actually value low-batch latency
              over quality and mature AR serving?
            </p>
            <p>
              <span>03</span>Would training from scratch close the quality gap,
              or erase the efficiency advantage?
            </p>
            <p>
              <span>04</span>What evaluation would separate true iterative
              reasoning from confidence sharpening?
            </p>
          </div>
          <div className={styles.endMatter}>
            <p>
              DiffusionGemma Technical Report · arXiv:2608.00146v1 · CC BY 4.0
            </p>
            <button type="button" onClick={() => window.print()}>
              Save this deck as PDF
            </button>
          </div>
        </Slide>
      </DeckSequence>
    </main>
  );
}
