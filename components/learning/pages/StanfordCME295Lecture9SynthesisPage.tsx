"use client";

import Link from "next/link";
import { useState, type ReactNode } from "react";
import {
  Aperture,
  ArrowRight,
  BookOpenCheck,
  Database,
  GitBranch,
  Route,
  ScanSearch,
  Sparkles,
  Split,
} from "lucide-react";
import type { LearningExperience } from "../../../lib/learning";
import MathText from "../../MathText";
import { QuizTransitionButton } from "../LearningPrimitives";
import {
  closingThemes,
  courseTraceStages,
  getClosingTheme,
  getCourseTraceStage,
  getGenerationPassComparison,
  getPatchTokenCount,
  getRecapUnit,
  getVlmPattern,
  recapUnits,
  vlmPatterns,
  type ClosingThemeId,
  type CourseTraceStageId,
  type TransferMode,
  type VlmPatternId,
} from "./cme295-lecture9/synthesisLogic";

type Props = {
  experience: LearningExperience;
};

type Tone = "cyan" | "emerald" | "amber" | "rose" | "violet" | "slate";

function InlineMath({ text }: { text: string }) {
  return <MathText inline text={text} />;
}

function ToneBadge({
  icon,
  label,
  tone = "cyan",
}: {
  icon: ReactNode;
  label: string;
  tone?: Tone;
}) {
  const tones: Record<Tone, string> = {
    cyan: "border-cyan-600 bg-cyan-50 text-cyan-950",
    emerald: "border-emerald-600 bg-emerald-50 text-emerald-950",
    amber: "border-amber-600 bg-amber-50 text-amber-950",
    rose: "border-rose-600 bg-rose-50 text-rose-950",
    violet: "border-violet-600 bg-violet-50 text-violet-950",
    slate: "border-slate-500 bg-slate-100 text-slate-950",
  };

  return (
    <span
      className={`inline-flex items-center gap-2 rounded-md border px-3 py-2 text-sm font-semibold ${tones[tone]}`}
    >
      {icon}
      {label}
    </span>
  );
}

function ControlButton({
  children,
  isActive,
  onClick,
  tone = "cyan",
}: {
  children: ReactNode;
  isActive: boolean;
  onClick: () => void;
  tone?: Exclude<Tone, "slate">;
}) {
  const activeTones: Record<Exclude<Tone, "slate">, string> = {
    cyan: "border-cyan-700 bg-cyan-700 text-white",
    emerald: "border-emerald-700 bg-emerald-700 text-white",
    amber: "border-amber-700 bg-amber-500 text-slate-950",
    rose: "border-rose-700 bg-rose-700 text-white",
    violet: "border-violet-700 bg-violet-700 text-white",
  };

  return (
    <button
      type="button"
      aria-pressed={isActive}
      onClick={onClick}
      className={`min-h-11 rounded-md border px-3 py-2 text-left text-sm font-semibold transition-colors ${
        isActive
          ? activeTones[tone]
          : "border-slate-300 bg-white text-slate-800 hover:border-slate-500"
      }`}
    >
      {children}
    </button>
  );
}

function Metric({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail?: ReactNode;
}) {
  return (
    <div className="min-w-0 rounded-lg border border-slate-300 bg-white p-4">
      <p className="text-sm font-semibold text-slate-500">{label}</p>
      <p
        className={`mt-2 break-words font-semibold text-slate-950 ${
          value.length > 42 ? "text-base leading-6" : "text-2xl"
        }`}
      >
        {value}
      </p>
      {detail && (
        <p className="mt-2 text-sm leading-6 text-slate-600">{detail}</p>
      )}
    </div>
  );
}

function FormulaPanel({
  title,
  formula,
  children,
}: {
  title: string;
  formula: string;
  children: ReactNode;
}) {
  return (
    <div className="rounded-lg border border-slate-300 bg-slate-50 p-4">
      <p className="text-sm font-semibold text-slate-600">{title}</p>
      <MathText
        text={formula}
        className="mt-3 min-w-0 max-w-full overflow-x-auto overflow-y-hidden rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-950 [&_.katex-mathml]:hidden"
      />
      <div className="mt-3 text-sm leading-6 text-slate-700">{children}</div>
    </div>
  );
}

function RecapConceptCard({
  unit,
  index,
}: {
  unit: (typeof recapUnits)[number];
  index: number;
}) {
  return (
    <article
      data-testid={`lecture9-recap-unit-${unit.id}`}
      className="rounded-xl border border-slate-300 bg-white p-5 shadow-sm md:p-6"
    >
      <div className="flex flex-wrap items-center gap-2">
        <ToneBadge
          icon={<BookOpenCheck aria-hidden="true" size={18} />}
          label={unit.lectureLabel}
          tone="emerald"
        />
        <span className="rounded-md border border-cyan-700 bg-cyan-50 px-3 py-2 text-sm font-semibold text-cyan-950">
          Recap step {index + 1}
        </span>
      </div>

      <h3 className="mt-4 text-2xl font-semibold text-slate-950">
        {unit.title}
      </h3>
      <p className="mt-2 text-base font-semibold text-slate-700">
        {unit.subtitle}
      </p>

      <div className="mt-5 rounded-lg border border-cyan-200 bg-cyan-50 p-4">
        <p className="text-sm font-semibold uppercase tracking-wide text-cyan-900">
          Core idea
        </p>
        <p className="mt-2 text-base leading-7 text-cyan-950">
          {unit.coreIdea}
        </p>
      </div>

      <div className="mt-5 grid gap-3 md:grid-cols-3">
        {unit.terms.map((term) => (
          <div
            key={term.label}
            className="rounded-lg border border-slate-300 bg-slate-50 p-4"
          >
            <p className="text-sm font-semibold text-slate-950">{term.label}</p>
            <p className="mt-2 text-sm leading-6 text-slate-700">
              {term.explanation}
            </p>
          </div>
        ))}
      </div>

      <div className="mt-5 grid min-w-0 gap-5 lg:grid-cols-[1fr_0.95fr]">
        <div className="min-w-0 space-y-4">
          <div>
            <p className="text-sm font-semibold text-slate-500">
              How the mechanism works
            </p>
            <p className="mt-2 text-sm leading-6 text-slate-700">
              {unit.mechanism}
            </p>
          </div>

          {unit.formula && (
            <MathText
              text={unit.formula}
              className="min-w-0 max-w-full overflow-x-auto overflow-y-hidden rounded-lg border border-slate-300 bg-slate-50 px-3 py-2 text-slate-950 [&_.katex-mathml]:hidden"
            />
          )}

          <div>
            <p className="text-sm font-semibold text-slate-500">
              What the recap emphasized
            </p>
            <p className="mt-2 text-sm leading-6 text-slate-700">
              {unit.sourceTrace}
            </p>
          </div>
        </div>

        <div className="min-w-0 rounded-lg border border-emerald-200 bg-emerald-50 p-4">
          <p className="text-sm font-semibold text-emerald-900">
            Why it matters next
          </p>
          <p className="mt-2 text-sm leading-6 text-emerald-950">
            {unit.handoff}
          </p>

          <p className="mt-5 text-sm font-semibold text-emerald-900">
            Mechanism checkpoints
          </p>
          <ul className="mt-3 space-y-2 text-sm leading-6 text-emerald-950">
            {unit.steps.map((step) => (
              <li key={step} className="flex gap-2">
                <ArrowRight
                  aria-hidden="true"
                  className="mt-1 shrink-0 text-emerald-700"
                  size={16}
                />
                <span>{step}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </article>
  );
}

function RecapAtlas() {
  return (
    <section
      id="recap-atlas"
      data-testid="lecture9-recap-atlas"
      className="bg-[#f8fafc] text-slate-950"
    >
      <div className="mx-auto w-full max-w-5xl px-4 py-12">
        <div className="max-w-3xl space-y-4">
          <ToneBadge
            icon={<Route aria-hidden="true" size={18} />}
            label="Course recap atlas"
            tone="cyan"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Rebuild Lectures 1-8 as mechanisms, not labels
          </h2>
          <p className="text-base leading-7 text-slate-700">
            The recap starts with how text becomes vectors, then follows the
            same answer through attention, model-family choices, LLM runtime,
            training, preference tuning, reasoning, system connections, and
            evaluation. Each box below introduces the concepts before the later
            labs reuse them.
          </p>
        </div>

        <FormulaPanel
          title="The operation that keeps returning"
          formula={String.raw`\[\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V\]`}
        >
          <span>
            The recap keeps returning to attention because it is the bridge from
            text tokens to image patches, cross-attention, retrieval context,
            and evaluation traces.
          </span>
        </FormulaPanel>

        <div className="mt-8 space-y-6">
          {recapUnits.map((unit, index) => (
            <RecapConceptCard key={unit.id} unit={unit} index={index} />
          ))}
        </div>
      </div>
    </section>
  );
}

function CourseTraceLab() {
  const [activeId, setActiveId] = useState<CourseTraceStageId>("prompt");
  const stage = getCourseTraceStage(activeId);

  return (
    <section
      data-testid="lecture9-course-trace"
      className="border-y border-slate-200 bg-white text-slate-950"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 space-y-4">
          <ToneBadge
            icon={<GitBranch aria-hidden="true" size={18} />}
            label="Answer trace"
            tone="emerald"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Follow one answer across the recap
          </h2>
          <p className="text-base leading-7 text-slate-700">
            A fluent answer is not one concept. It is a chain of input
            representation, attention, architecture, decoding, training,
            external grounding, and measurement.
          </p>
        </div>

        <div className="min-w-0 rounded-xl border border-slate-300 bg-slate-50 p-4">
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {courseTraceStages.map((candidate) => (
              <ControlButton
                key={candidate.id}
                tone="emerald"
                isActive={candidate.id === activeId}
                onClick={() => setActiveId(candidate.id)}
              >
                {candidate.label}
              </ControlButton>
            ))}
          </div>

          <div className="mt-5 grid gap-3 md:grid-cols-3">
            <Metric label="Input" value={stage.input} />
            <Metric label="Operation" value={stage.operation} />
            <Metric label="Output" value={stage.output} />
          </div>

          <p
            role="status"
            className="mt-4 rounded-lg border border-emerald-600 bg-emerald-50 p-4 text-sm leading-6 text-emerald-950"
          >
            Related recap modules:{" "}
            {stage.recapIds
              .map((recapId) => getRecapUnit(recapId).title)
              .join(", ")}
            .
          </p>
        </div>
      </div>
    </section>
  );
}

function VisionDiffusionLab() {
  const [mode, setMode] = useState<TransferMode>("vit");
  const [patchSize, setPatchSize] = useState(16);
  const [vlmPatternId, setVlmPatternId] =
    useState<VlmPatternId>("visualPrefix");
  const [maskedTokensPerPass, setMaskedTokensPerPass] = useState(6);

  const patchTokens = getPatchTokenCount({
    imageWidth: 224,
    imageHeight: 224,
    patchSize,
  });
  const pattern = getVlmPattern(vlmPatternId);
  const passComparison = getGenerationPassComparison({
    outputTokens: 24,
    maskedTokensPerPass,
  });

  return (
    <section
      data-testid="vision-diffusion-transfer-lab"
      className="border-y border-slate-200 bg-[#102027] text-slate-50"
    >
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 space-y-4">
          <ToneBadge
            icon={<Aperture aria-hidden="true" size={18} />}
            label="Transfer lab"
            tone="amber"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Same transformer vocabulary, new input units
          </h2>
          <p className="text-base leading-7 text-slate-300">
            The lecture then asks whether the transformer idea is locked to
            text. The answer is no: images can become patch tokens, visual
            encoders can feed language decoders, and text diffusion can use mask
            tokens as discrete noise.
          </p>
          <div className="grid gap-2 sm:grid-cols-3">
            <ControlButton
              tone="amber"
              isActive={mode === "vit"}
              onClick={() => setMode("vit")}
            >
              ViT patches
            </ControlButton>
            <ControlButton
              tone="amber"
              isActive={mode === "vlm"}
              onClick={() => setMode("vlm")}
            >
              VLM wiring
            </ControlButton>
            <ControlButton
              tone="amber"
              isActive={mode === "diffusion"}
              onClick={() => setMode("diffusion")}
            >
              Masked diffusion
            </ControlButton>
          </div>
        </div>

        <div className="min-w-0 rounded-xl border border-slate-700 bg-slate-950 p-4">
          {mode === "vit" && (
            <div>
              <p className="text-sm leading-6 text-slate-300">
                A ViT patch is a fixed square crop of the image. The model
                flattens or projects each patch into a vector, adds position
                information, prepends a class token, and uses a transformer
                encoder. That is why the lecture compares ViT to a BERT-like
                encoder more than to a decoder-only generator.
              </p>

              <div className="mt-4 grid gap-2 sm:grid-cols-3">
                {[8, 16, 32].map((size) => (
                  <ControlButton
                    key={size}
                    tone="emerald"
                    isActive={patchSize === size}
                    onClick={() => setPatchSize(size)}
                  >
                    {size} px patches
                  </ControlButton>
                ))}
              </div>

              <div className="mt-5 grid gap-5 md:grid-cols-[0.9fr_1.1fr]">
                <div
                  aria-label="Illustrative image patch grid"
                  className="grid aspect-square grid-cols-4 gap-1 rounded-lg border border-slate-700 bg-slate-900 p-2"
                >
                  {Array.from({ length: 16 }).map((_, index) => (
                    <div
                      key={index}
                      className={`rounded-md border ${
                        index === 5
                          ? "border-amber-300 bg-amber-300"
                          : "border-slate-700 bg-slate-800"
                      }`}
                    />
                  ))}
                </div>
                <div className="space-y-3">
                  <Metric label="Patch tokens" value={`${patchTokens}`} />
                  <Metric
                    label="Encoder input"
                    value={`${patchTokens + 1}`}
                    detail="Patch tokens plus a class token for image-level classification."
                  />
                  <p
                    role="status"
                    className="rounded-lg border border-emerald-400 bg-emerald-950/40 p-4 text-sm leading-6 text-emerald-50"
                  >
                    Patch means fixed image square. A 224 x 224 image with{" "}
                    {patchSize} x {patchSize} patches creates {patchTokens}{" "}
                    patch tokens, and the class token reads the encoded image
                    summary for classification.
                  </p>
                </div>
              </div>
            </div>
          )}

          {mode === "vlm" && (
            <div>
              <p className="text-sm leading-6 text-slate-300">
                A vision-language model first turns the image into visual
                features or visual tokens. The language side can either receive
                those vectors as prefix-like tokens or use cross-attention to
                read them as a separate memory.
              </p>

              <div className="mt-4 grid gap-2 sm:grid-cols-2">
                {vlmPatterns.map((candidate) => (
                  <ControlButton
                    key={candidate.id}
                    tone="violet"
                    isActive={vlmPatternId === candidate.id}
                    onClick={() => setVlmPatternId(candidate.id)}
                  >
                    {candidate.label}
                  </ControlButton>
                ))}
              </div>

              <div className="mt-5 grid gap-4 md:grid-cols-3">
                <Metric label="Image side" value="visual encoder" />
                <Metric
                  label="Bridge"
                  value={
                    vlmPatternId === "visualPrefix"
                      ? "visual tokens"
                      : "cross-attention"
                  }
                />
                <Metric label="Language side" value="decoder answer" />
              </div>

              <p
                role="status"
                className="mt-4 rounded-lg border border-violet-400 bg-violet-950/40 p-4 text-sm leading-6 text-violet-50"
              >
                <span className="font-semibold">{pattern.label}.</span>{" "}
                {pattern.mechanism} {pattern.consequence}
              </p>
            </div>
          )}

          {mode === "diffusion" && (
            <div>
              <p className="text-sm leading-6 text-slate-300">
                Autoregressive LLMs generate one token after another at
                inference time. Image diffusion starts from easy-to-sample noise
                and learns a reverse denoising process. For text, the discrete
                analog is masking tokens and learning to unmask them.
              </p>

              <FormulaPanel
                title="Simplified pass-count contrast"
                formula={String.raw`\[\mathrm{ARM\ passes}=n,\quad \mathrm{MDM\ passes}\approx \left\lceil \frac{n}{m}\right\rceil\]`}
              >
                <span>
                  <InlineMath text={String.raw`\(n\)`} /> is output tokens and{" "}
                  <InlineMath text={String.raw`\(m\)`} /> is how many masked
                  positions the toy masked-diffusion pass fills at once.
                </span>
              </FormulaPanel>

              <div className="mt-4 grid gap-2 sm:grid-cols-3">
                {[3, 6, 8].map((count) => (
                  <ControlButton
                    key={count}
                    tone="rose"
                    isActive={maskedTokensPerPass === count}
                    onClick={() => setMaskedTokensPerPass(count)}
                  >
                    {count} tokens per pass
                  </ControlButton>
                ))}
              </div>

              <div className="mt-5 grid gap-3 sm:grid-cols-3">
                <Metric
                  label="Autoregressive"
                  value={`${passComparison.autoregressivePasses} passes`}
                />
                <Metric
                  label="Masked diffusion"
                  value={`${passComparison.maskedDiffusionPasses} passes`}
                />
                <Metric
                  label="Toy speedup"
                  value={`${passComparison.speedupRatio.toFixed(1)}x`}
                />
              </div>

              <p
                role="status"
                className="mt-4 rounded-lg border border-rose-400 bg-rose-950/40 p-4 text-sm leading-6 text-rose-50"
              >
                Starting from masked positions and filling {maskedTokensPerPass}{" "}
                of them per pass takes {passComparison.maskedDiffusionPasses}{" "}
                passes for this 24-token toy answer, a{" "}
                {passComparison.speedupRatio.toFixed(1)}x pass-count reduction
                in this simplified comparison. The source point is the unmasking
                process, not Gaussian noise applied to words.
              </p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function CrossModalBridge() {
  const ideas = [
    {
      label: "Diffusion to text",
      body: "Image denoising inspires masked language generation that does not have to fill strictly left to right.",
    },
    {
      label: "Transformers to images",
      body: "ViT and diffusion transformers show that self-attention can replace stronger convolutional bias when data is large enough.",
    },
    {
      label: "Vision tokens to text",
      body: "The lecture's OCR example treats visual patches as compressed carriers of text information.",
    },
    {
      label: "RoPE to 2D",
      body: "Relative position tricks can be reformulated for image grids and multimodal layouts.",
    },
  ];

  return (
    <section className="bg-[#f7f1e7] text-slate-950">
      <div className="mx-auto w-full max-w-6xl px-4 py-10">
        <div className="max-w-3xl space-y-3">
          <ToneBadge
            icon={<Split aria-hidden="true" size={18} />}
            label="Cross-modal bridge"
            tone="rose"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Modalities trade useful ideas
          </h2>
          <p className="text-base leading-7 text-slate-700">
            The closing transition is not a new model taxonomy. It is a pattern:
            once a mechanism works, researchers try it on another representation
            and adapt the parts that depend on the data type.
          </p>
        </div>

        <div className="mt-6 grid gap-3 md:grid-cols-4">
          {ideas.map((idea) => (
            <div
              key={idea.label}
              className="rounded-lg border border-amber-700/30 bg-white p-4"
            >
              <p className="font-semibold text-slate-950">{idea.label}</p>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                {idea.body}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function ClosingBoard() {
  const [themeId, setThemeId] = useState<ClosingThemeId>("architecture");
  const theme = getClosingTheme(themeId);

  return (
    <section className="border-y border-slate-200 bg-white text-slate-950">
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="min-w-0 space-y-4">
          <ToneBadge
            icon={<Sparkles aria-hidden="true" size={18} />}
            label="Closing frontiers"
            tone="rose"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Future work starts from unresolved mechanisms
          </h2>
          <p className="text-base leading-7 text-slate-700">
            The last section is brief compared with the recap. Its useful
            message is that transformers, data, serving economics, hardware, and
            deployment limits are still moving.
          </p>
          <div className="grid gap-2 sm:grid-cols-2">
            {closingThemes.map((candidate) => (
              <ControlButton
                key={candidate.id}
                tone="rose"
                isActive={themeId === candidate.id}
                onClick={() => setThemeId(candidate.id)}
              >
                {candidate.label}
              </ControlButton>
            ))}
          </div>
        </div>

        <div className="min-w-0 rounded-xl border border-slate-300 bg-slate-50 p-5">
          <p
            role="status"
            className="rounded-lg border border-rose-500 bg-rose-50 p-4 text-sm leading-6 text-rose-950"
          >
            <span className="font-semibold">{theme.label}.</span> {theme.claim}
          </p>
          <ul className="mt-5 space-y-3 text-sm leading-6 text-slate-700">
            {theme.details.map((detail) => (
              <li key={detail} className="flex gap-2">
                <ArrowRight
                  aria-hidden="true"
                  className="mt-1 shrink-0 text-rose-700"
                  size={16}
                />
                <span>{detail}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}

function FinalCheck() {
  const [selected, setSelected] = useState<"rag" | "retrain" | null>(null);
  const isCorrect = selected === "rag";

  return (
    <section className="bg-[#f8fafc] text-slate-950">
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-12 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="min-w-0 space-y-4">
          <ToneBadge
            icon={<ScanSearch aria-hidden="true" size={18} />}
            label="Layer diagnosis"
            tone="emerald"
          />
          <h2 className="text-2xl font-semibold md:text-3xl">
            Choose the layer that should change
          </h2>
          <p className="text-base leading-7 text-slate-700">
            The Lecture 9 recap helps diagnose where a failure lives. A stale
            fact, a bad preference, a failed tool call, and a weak benchmark
            score are different problems.
          </p>
        </div>

        <div
          data-testid="lecture9-layer-check"
          className="min-w-0 rounded-xl border border-slate-300 bg-white p-4"
        >
          <p className="text-sm font-semibold text-slate-500">Scenario</p>
          <p className="mt-2 text-lg font-semibold text-slate-950">
            A model answers an internal policy question fluently, but its answer
            is wrong because the policy changed yesterday.
          </p>
          <div className="mt-5 grid gap-2 sm:grid-cols-2">
            <ControlButton
              tone="emerald"
              isActive={selected === "rag"}
              onClick={() => setSelected("rag")}
            >
              Add retrieval or a policy tool
            </ControlButton>
            <ControlButton
              tone="emerald"
              isActive={selected === "retrain"}
              onClick={() => setSelected("retrain")}
            >
              Only retrain the base model
            </ControlButton>
          </div>
          {selected && (
            <p
              role="status"
              className={`mt-4 rounded-lg border p-4 text-sm leading-6 ${
                isCorrect
                  ? "border-emerald-500 bg-emerald-50 text-emerald-950"
                  : "border-amber-500 bg-amber-50 text-amber-950"
              }`}
            >
              <span className="font-semibold">
                {isCorrect ? "Correct." : "Not yet."}
              </span>{" "}
              Fixed weights do not update continuously. RAG or a tool can bring
              the current policy into context, while evaluation should verify
              that the answer is grounded in the retrieved evidence.
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

function RecapFooter({
  sourceId,
}: {
  sourceId: Props["experience"]["sourceId"];
}) {
  return (
    <section className="border-t border-slate-800 bg-[#0d1720] text-slate-50">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-5 px-4 py-10 md:flex-row md:items-center md:justify-between">
        <div className="max-w-2xl">
          <h2 className="text-2xl font-semibold">Course recap complete</h2>
          <p className="mt-2 text-sm leading-6 text-slate-300">
            The same chain now covers the quarter: representation, attention,
            model families, runtime, training, preference tuning, reasoning,
            systems, evaluation, then transfer into vision and diffusion.
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <Link
            href="/learn/stanford-cme295"
            className="inline-flex items-center justify-center rounded-lg border border-slate-600 px-4 py-3 text-sm font-bold text-slate-100 transition-colors hover:border-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950"
          >
            Back to course
          </Link>
          <QuizTransitionButton
            sourceId={sourceId}
            label="Start synthesis questions"
          />
        </div>
      </div>
    </section>
  );
}

export default function StanfordCME295Lecture9SynthesisPage({
  experience,
}: Props) {
  return (
    <main className="min-h-screen bg-slate-950 text-slate-50">
      <section className="border-b border-slate-800 bg-[#101820]">
        <div className="mx-auto flex min-h-[460px] w-full max-w-5xl items-center px-4 py-10 md:py-14">
          <div className="min-w-0 space-y-6">
            <Link
              href="/learn/stanford-cme295"
              className="inline-flex text-sm font-semibold text-cyan-200 hover:text-cyan-100"
            >
              Back to Stanford CME295 course
            </Link>
            <div className="flex flex-wrap gap-2">
              <ToneBadge
                icon={<BookOpenCheck aria-hidden="true" size={18} />}
                label="Stanford CME295"
                tone="cyan"
              />
              <ToneBadge
                icon={<Database aria-hidden="true" size={18} />}
                label="Lecture 9 recap"
                tone="emerald"
              />
            </div>
            <div className="space-y-4">
              <h1 className="max-w-3xl text-3xl font-semibold text-slate-50 md:text-5xl md:leading-tight">
                Reconstruct the transformer course from the recap
              </h1>
              <p className="max-w-2xl text-base leading-7 text-slate-300 md:text-lg">
                Lecture 9 opens by rewinding the quarter: tokenization,
                embeddings, self-attention, transformer families, LLM runtime,
                training, preference tuning, reasoning, RAG, tools, agents, and
                evaluation. Vision transformers, VLMs, diffusion LLMs, and
                frontiers then test whether the same mechanisms transfer.
              </p>
              <p className="text-sm font-semibold text-emerald-300">
                Synthesis quiz source available
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <QuizTransitionButton
                sourceId={experience.sourceId}
                label="Start synthesis questions"
              />
              <a
                href="#recap-atlas"
                className="inline-flex items-center justify-center rounded-lg border border-slate-600 px-4 py-3 text-sm font-bold text-slate-100 transition-colors hover:border-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950"
              >
                Open recap atlas
              </a>
            </div>
          </div>
        </div>
      </section>

      <RecapAtlas />
      <CourseTraceLab />
      <VisionDiffusionLab />
      <CrossModalBridge />
      <ClosingBoard />
      <FinalCheck />
      <RecapFooter sourceId={experience.sourceId} />
    </main>
  );
}
