"use client";

import { useMemo, useState } from "react";
import type { LearningExperience } from "../../../lib/learning";
import {
  CheckForUnderstanding,
  FormulaBlock,
  QuizTransitionButton,
  RecapSection,
} from "../LearningPrimitives";

type Props = {
  experience: LearningExperience;
};

const lifecycleStages = {
  pretraining: {
    label: "Pretraining",
    eyebrow: "Stage 1",
    objective: "Learn broad language and code structure by predicting tokens.",
    input:
      "Trillions-scale mixtures from web text, books, code, and other corpora.",
    output:
      "A base decoder-only model that models text well but is not yet a helpful assistant.",
    bottleneck:
      "Cost, stale knowledge, memorization risk, and the compute balance between parameters and tokens.",
    artifact: "Base model",
  },
  optimizations: {
    label: "Training optimizations",
    eyebrow: "Stage 2",
    objective:
      "Make the giant training run fit in GPU memory and finish sooner.",
    input:
      "Parameters, activations, gradients, optimizer state, and attention matrices.",
    output:
      "The same training objective, but spread across devices or executed with less memory traffic.",
    bottleneck:
      "Memory capacity, HBM/SRAM movement, communication cost, and floating-point precision.",
    artifact: "Feasible run",
  },
  sft: {
    label: "SFT",
    eyebrow: "Stage 3",
    objective:
      "Tune the model toward task behavior or instruction following with input-output pairs.",
    input:
      "High-quality assistant dialogues, synthetic instructions, reasoning, code, and safety data.",
    output:
      "A model that responds to user instructions instead of only continuing likely text.",
    bottleneck:
      "Prompt distribution mismatch, expensive data quality, subjective behavior, and evaluation limits.",
    artifact: "Instruction-tuned model",
  },
  peft: {
    label: "LoRA / QLoRA",
    eyebrow: "Stage 4",
    objective: "Adapt a pretrained model without updating every weight.",
    input:
      "Frozen base weights plus small low-rank adapter matrices for a target task.",
    output:
      "Task-specific adapters that can be trained, swapped, merged, or stored cheaply.",
    bottleneck:
      "Choosing rank, learning rate, batch size, adapter locations, and quantization strategy.",
    artifact: "Adapter-tuned model",
  },
} as const;

type LifecycleStageId = keyof typeof lifecycleStages;

const stageOrder: LifecycleStageId[] = [
  "pretraining",
  "optimizations",
  "sft",
  "peft",
];

const modelScales = {
  gpt3: {
    label: "GPT-3-style example",
    parametersB: 175,
    tokensT: 0.3,
    note: "A very large parameter count with comparatively few training tokens.",
  },
  balanced: {
    label: "Compute-balanced sketch",
    parametersB: 70,
    tokensT: 1.4,
    note: "A simple Chinchilla-style target at about 20 tokens per parameter.",
  },
  dataHeavy: {
    label: "Small model, huge corpus",
    parametersB: 8,
    tokensT: 2,
    note: "More data per parameter can help reuse a small model, but fixed compute might have supported more parameters.",
  },
} as const;

type ModelScaleId = keyof typeof modelScales;

const memoryObjects = {
  parameters: {
    label: "Parameters",
    body: "The model weights themselves. Dense models need every parameter available; sparse MoE models activate only selected experts per token.",
  },
  activations: {
    label: "Activations",
    body: "Intermediate layer values from the forward pass. They are needed later to compute gradients, and grow with model size, batch size, and context length.",
  },
  gradients: {
    label: "Gradients",
    body: "Backward-pass signals used to update parameters. They must be synchronized or sharded when training across devices.",
  },
  optimizer: {
    label: "Optimizer state",
    body: "Adam-style moving averages can take large memory, often more than the raw model weights.",
  },
} as const;

type MemoryObjectId = keyof typeof memoryObjects;

const optimizationLevers = {
  dataParallel: {
    label: "Data parallelism",
    category: "Split batch",
    target:
      "Reduces per-device batch memory, but each GPU still holds a full model copy.",
    tradeoff:
      "Gradients must be averaged across devices, so communication becomes part of training cost.",
  },
  zero3: {
    label: "ZeRO-3",
    category: "Shard state",
    target:
      "Partitions optimizer state, gradients, and parameters instead of redundantly storing them on every GPU.",
    tradeoff:
      "The memory win is large, but the system must gather the right shards at the right time.",
  },
  modelParallel: {
    label: "Model parallelism",
    category: "Split computation",
    target:
      "Places parts of the model computation on different devices through tensor, pipeline, sequence, context, or expert parallelism.",
    tradeoff:
      "It lets a model exceed one GPU's memory, but pipeline bubbles and device communication can limit speed.",
  },
  flash: {
    label: "FlashAttention",
    category: "Use GPU hierarchy",
    target:
      "Computes exact attention by tiling Q, K, and V through fast SRAM so full score and probability matrices do not need repeated HBM writes.",
    tradeoff:
      "The backward pass may recompute instead of storing everything: more FLOPs can still mean less runtime when HBM traffic is the bottleneck.",
  },
  mixedPrecision: {
    label: "Mixed precision",
    category: "Change number format",
    target:
      "Uses lower precision for forward and backward work while keeping master weights or sensitive updates at higher precision.",
    tradeoff:
      "Memory and throughput improve, but precision choices must avoid accumulating harmful numerical error.",
  },
} as const;

type OptimizationLeverId = keyof typeof optimizationLevers;

const sftModes = {
  base: {
    label: "Pretrained only",
    prompt: "Can I put my teddy bear in the washer?",
    response:
      "Teddy bears are often made of materials like polyester and cotton, with plastic eyes and stitched seams.",
    diagnosis:
      "The model continues plausible text about teddy bears. It may know facts, but it has not learned the assistant behavior of answering the user's practical question.",
  },
  instruction: {
    label: "Instruction tuned",
    prompt: "Can I put my teddy bear in the washer?",
    response:
      "Usually no. Check the care label and prefer gentle hand washing so stuffing, seams, or plastic parts are not damaged.",
    diagnosis:
      "SFT changes behavior by training on input-output pairs. The loss is on the desired response tokens, conditioned on the instruction.",
  },
  evaluation: {
    label: "Evaluation view",
    prompt: "Which model is better?",
    response:
      "MMLU, GSM8K, HumanEval, expert tests, and arena-style A/B preferences each measure a different slice.",
    diagnosis:
      "Evaluation is not one number. Benchmarks can be confounded by training on the task, while user votes can reflect exposure, factuality blind spots, personal preference, or safety penalties.",
  },
} as const;

type SftModeId = keyof typeof sftModes;

const adapterTasks = {
  spam: {
    label: "Spam detection",
    output: "adapter A: inbox safety",
    body: "The base model stays frozen while the low-rank adapter learns the spam-detection behavior.",
  },
  sentiment: {
    label: "Sentiment extraction",
    output: "adapter B: review tone",
    body: "Swapping adapter matrices changes the task without retraining or duplicating the full base model.",
  },
  assistant: {
    label: "Assistant style",
    output: "adapter C: helpful replies",
    body: "Instruction behavior can be adapted with a small trainable surface when full SFT is too expensive.",
  },
} as const;

type AdapterTaskId = keyof typeof adapterTasks;

const adapterModes = {
  full: {
    label: "Full fine-tune",
    status: "Every selected weight matrix is trainable.",
    detail:
      "This is flexible but expensive because it updates the full matrix instead of a compact delta.",
  },
  lora: {
    label: "LoRA",
    status: "W0 is frozen; train only B and A.",
    detail:
      "The adapter learns a low-rank update BA, so trainable parameters scale with rank instead of the full matrix area.",
  },
  qlora: {
    label: "QLoRA",
    status:
      "Frozen W0 is stored quantized; B and A remain trainable in higher precision.",
    detail:
      "NF4 and double quantization reduce VRAM pressure while preserving a full-precision path for adapter computations.",
  },
} as const;

type AdapterModeId = keyof typeof adapterModes;

function formatPercent(value: number) {
  if (value < 0.1) return `${value.toFixed(3)}%`;
  if (value < 1) return `${value.toFixed(2)}%`;
  return `${value.toFixed(1)}%`;
}

function getTokenBalance(tokensPerParameter: number) {
  if (tokensPerParameter < 10) {
    return {
      label: "Parameter-heavy",
      body: "For a fixed compute budget, this spends many parameters on relatively little data. The lecture frames GPT-3 as undertrained by this later rule of thumb.",
      color: "text-amber-200",
    };
  }
  if (tokensPerParameter <= 35) {
    return {
      label: "Near Chinchilla balance",
      body: "This is close to the lecture's rough 20 tokens per parameter compute-optimal relationship.",
      color: "text-emerald-200",
    };
  }
  return {
    label: "Data-heavy",
    body: "This trains many tokens per parameter. It can improve a smaller model, but the fixed-compute question asks whether more parameters would have helped.",
    color: "text-sky-200",
  };
}

function TrainingPipelinePreview() {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <p className="text-sm font-semibold uppercase tracking-wide text-sky-300">
        Lecture 4 training lifecycle
      </p>
      <div className="mt-4 grid gap-3">
        {stageOrder.map((stageId, index) => {
          const stage = lifecycleStages[stageId];
          return (
            <div
              key={stageId}
              className="grid grid-cols-[2.25rem_1fr] items-stretch gap-3"
            >
              <div className="flex flex-col items-center">
                <span className="flex h-9 w-9 items-center justify-center rounded-full bg-slate-950 text-sm font-bold text-sky-200">
                  {index + 1}
                </span>
                {index < stageOrder.length - 1 && (
                  <span className="mt-2 h-8 w-px bg-slate-700" />
                )}
              </div>
              <div className="min-h-16 rounded-md border border-slate-700 bg-slate-950 px-3 py-2">
                <p className="text-sm font-semibold text-slate-100">
                  {stage.label}
                </p>
                <p className="mt-1 text-xs leading-5 text-slate-400">
                  Output: {stage.artifact}
                </p>
              </div>
            </div>
          );
        })}
      </div>
      <div className="mt-5 grid grid-cols-3 gap-2 text-center text-xs font-semibold text-slate-200">
        {[
          ["Tokens", "D"],
          ["Parameters", "N"],
          ["Compute", "FLOPs"],
        ].map(([label, value]) => (
          <div key={label} className="rounded-md border border-slate-800 p-3">
            <p className="text-slate-400">{label}</p>
            <p className="mt-1 text-emerald-300">{value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function TrainingLifecycleWorkbench() {
  const [activeStageId, setActiveStageId] =
    useState<LifecycleStageId>("pretraining");
  const activeStage = lifecycleStages[activeStageId];

  return (
    <section
      data-testid="training-lifecycle-workbench"
      className="rounded-lg border border-slate-800 bg-slate-900 p-5"
    >
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-sky-300">
            Training lifecycle
          </p>
          <h2 className="mt-2 text-2xl font-semibold text-slate-50">
            Follow one model from raw continuation to tuned behavior
          </h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">
            Lecture 4 is not one technique. It is a sequence of decisions: learn
            broad token statistics, make the run fit, tune behavior, then adapt
            cheaply when full fine-tuning is too expensive.
          </p>
        </div>
        <div className="rounded-md border border-slate-800 bg-slate-950 px-3 py-2 text-sm font-semibold text-emerald-300">
          {activeStage.artifact}
        </div>
      </div>

      <div className="mt-5 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        {stageOrder.map((stageId) => {
          const stage = lifecycleStages[stageId];
          const selected = activeStageId === stageId;
          return (
            <button
              key={stageId}
              type="button"
              onClick={() => setActiveStageId(stageId)}
              className={`rounded-md border px-3 py-2 text-left text-sm font-semibold ${
                selected
                  ? "border-sky-400 bg-sky-400 text-slate-950"
                  : "border-slate-700 text-slate-200 hover:border-slate-500"
              }`}
            >
              <span className="block text-xs uppercase tracking-wide">
                {stage.eyebrow}
              </span>
              <span className="mt-1 block">{stage.label}</span>
            </button>
          );
        })}
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="rounded-lg border border-slate-800 bg-slate-950 p-4">
          <h3 className="text-lg font-semibold text-slate-50">
            {activeStage.objective}
          </h3>
          <dl className="mt-4 space-y-3 text-sm leading-6">
            <div>
              <dt className="font-semibold text-sky-200">Input</dt>
              <dd className="text-slate-300">{activeStage.input}</dd>
            </div>
            <div>
              <dt className="font-semibold text-emerald-200">Output</dt>
              <dd className="text-slate-300">{activeStage.output}</dd>
            </div>
            <div>
              <dt className="font-semibold text-amber-200">Main risk</dt>
              <dd className="text-slate-300">{activeStage.bottleneck}</dd>
            </div>
          </dl>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-950 p-4">
          <h3 className="text-lg font-semibold text-slate-50">
            What changes at this stage?
          </h3>
          <div className="mt-4 grid gap-3">
            {stageOrder.map((stageId, index) => {
              const stage = lifecycleStages[stageId];
              const isPast = stageOrder.indexOf(activeStageId) >= index;
              return (
                <div key={stageId} className="flex items-center gap-3">
                  <span
                    className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-bold ${
                      isPast
                        ? "bg-emerald-400 text-slate-950"
                        : "bg-slate-800 text-slate-400"
                    }`}
                  >
                    {index + 1}
                  </span>
                  <div className="min-w-0">
                    <p className="text-sm font-semibold text-slate-100">
                      {stage.label}
                    </p>
                    <p className="text-xs leading-5 text-slate-400">
                      {stage.artifact}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
          <p
            role="status"
            className="mt-4 rounded-md border border-sky-500/40 bg-sky-950/30 px-3 py-2 text-sm leading-6 text-sky-100"
          >
            Active stage: {activeStage.label}. Artifact: {activeStage.artifact}.{" "}
            {activeStage.output}
          </p>
        </div>
      </div>
    </section>
  );
}

function ScalingBudgetLab() {
  const [scaleId, setScaleId] = useState<ModelScaleId>("balanced");
  const scale = modelScales[scaleId];
  const tokensPerParameter = (scale.tokensT * 1000) / scale.parametersB;
  const balance = getTokenBalance(tokensPerParameter);

  return (
    <section
      data-testid="scaling-budget-lab"
      className="rounded-lg border border-slate-800 bg-slate-900 p-5"
    >
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-emerald-300">
            Pretraining scale
          </p>
          <h2 className="mt-2 text-2xl font-semibold text-slate-50">
            Spend fixed compute on parameters and tokens
          </h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">
            Scaling laws say loss improves as compute, data, and model size
            increase. The compute-optimal question is different: for a fixed
            budget, how should the run balance parameters against training
            tokens?
          </p>
        </div>
        <div
          className={`rounded-md border border-slate-800 bg-slate-950 px-3 py-2 text-sm font-semibold ${balance.color}`}
        >
          {balance.label}
        </div>
      </div>

      <div className="mt-5 flex flex-wrap gap-2">
        {Object.entries(modelScales).map(([id, item]) => (
          <button
            key={id}
            type="button"
            onClick={() => setScaleId(id as ModelScaleId)}
            className={`rounded-md border px-3 py-2 text-sm font-semibold ${
              scaleId === id
                ? "border-emerald-400 bg-emerald-400 text-slate-950"
                : "border-slate-700 text-slate-200 hover:border-slate-500"
            }`}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="rounded-lg border border-slate-800 bg-slate-950 p-4">
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-md border border-slate-800 p-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                Parameters
              </p>
              <p className="mt-2 text-2xl font-semibold text-slate-50">
                {scale.parametersB}B
              </p>
            </div>
            <div className="rounded-md border border-slate-800 p-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                Training tokens
              </p>
              <p className="mt-2 text-2xl font-semibold text-slate-50">
                {scale.tokensT}T
              </p>
            </div>
          </div>
          <div className="mt-4">
            <div className="flex justify-between gap-3 text-xs font-semibold text-slate-300">
              <span>Tokens per parameter</span>
              <span>{tokensPerParameter.toFixed(1)}x</span>
            </div>
            <div className="mt-2 h-3 rounded-full bg-slate-800">
              <div
                className="h-3 rounded-full bg-emerald-400"
                style={{
                  width: `${Math.min(100, Math.max(4, tokensPerParameter * 2.5))}%`,
                }}
              />
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-300">
              {scale.note}
            </p>
          </div>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-950 p-4">
          <h3 className="text-lg font-semibold text-slate-50">Diagnosis</h3>
          <p role="status" className="mt-3 text-sm leading-6 text-slate-300">
            {balance.body}
          </p>
          <div className="mt-4 rounded-md border border-slate-800 bg-slate-900 px-3 py-2 text-sm leading-6 text-slate-300">
            FLOPs estimate total arithmetic work. FLOP/s describes how fast the
            hardware can perform that work. The same letters are easy to
            confuse, so context matters.
          </div>
        </div>
      </div>
    </section>
  );
}

function MemoryOptimizationLab() {
  const [memoryId, setMemoryId] = useState<MemoryObjectId>("activations");
  const [leverId, setLeverId] = useState<OptimizationLeverId>("flash");
  const memory = memoryObjects[memoryId];
  const lever = optimizationLevers[leverId];

  return (
    <section
      data-testid="memory-optimization-lab"
      className="rounded-lg border border-slate-800 bg-slate-900 p-5"
    >
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-amber-300">
            Training bottlenecks
          </p>
          <h2 className="mt-2 text-2xl font-semibold text-slate-50">
            Match the memory object to the optimization lever
          </h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">
            The lecture keeps returning to one practical issue: a training run
            must store more than weights. Choose what is taking space, then
            choose the technique that attacks that pressure.
          </p>
        </div>
        <div className="rounded-md border border-slate-800 bg-slate-950 px-3 py-2 text-sm font-semibold text-amber-200">
          {lever.category}
        </div>
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-[0.85fr_1.15fr]">
        <div>
          <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
            Stored during training
          </h3>
          <div className="mt-3 grid gap-2">
            {Object.entries(memoryObjects).map(([id, item]) => (
              <button
                key={id}
                type="button"
                onClick={() => setMemoryId(id as MemoryObjectId)}
                className={`rounded-md border px-3 py-2 text-left text-sm font-semibold ${
                  memoryId === id
                    ? "border-amber-300 bg-amber-300 text-slate-950"
                    : "border-slate-700 bg-slate-950 text-slate-200 hover:border-slate-500"
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
          <p className="mt-4 rounded-md border border-slate-800 bg-slate-950 px-3 py-2 text-sm leading-6 text-slate-300">
            {memory.body}
          </p>
        </div>

        <div>
          <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
            Optimization lever
          </h3>
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            {Object.entries(optimizationLevers).map(([id, item]) => (
              <button
                key={id}
                type="button"
                onClick={() => setLeverId(id as OptimizationLeverId)}
                className={`rounded-md border px-3 py-2 text-left text-sm font-semibold ${
                  leverId === id
                    ? "border-sky-400 bg-sky-400 text-slate-950"
                    : "border-slate-700 bg-slate-950 text-slate-200 hover:border-slate-500"
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
          <div className="mt-4 rounded-lg border border-slate-800 bg-slate-950 p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              {lever.label}
            </p>
            <p role="status" className="mt-3 text-sm leading-6 text-sky-100">
              {lever.target}
            </p>
            <p className="mt-3 text-sm leading-6 text-slate-300">
              {lever.tradeoff}
            </p>
          </div>
        </div>
      </div>

      <div className="mt-5 grid gap-3 md:grid-cols-3">
        {[
          "HBM: big but slower",
          "SRAM: small but fast",
          "Compute units: do the matrix work",
        ].map((label) => (
          <div
            key={label}
            className="rounded-md border border-slate-800 bg-slate-950 px-3 py-2 text-sm font-semibold text-slate-200"
          >
            {label}
          </div>
        ))}
      </div>
    </section>
  );
}

function SftBehaviorPanel() {
  const [modeId, setModeId] = useState<SftModeId>("instruction");
  const mode = sftModes[modeId];

  return (
    <section
      data-testid="sft-behavior-panel"
      className="rounded-lg border border-slate-800 bg-slate-900 p-5"
    >
      <p className="text-sm font-semibold uppercase tracking-wide text-rose-300">
        Supervised fine-tuning
      </p>
      <h2 className="mt-2 text-2xl font-semibold text-slate-50">
        Tune continuation into assistant behavior
      </h2>
      <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">
        SFT still uses next-token prediction, but the training examples are
        instruction-response pairs. The model conditions on the input and the
        supervised loss starts on the desired output.
      </p>

      <div className="mt-5 flex flex-wrap gap-2">
        {Object.entries(sftModes).map(([id, item]) => (
          <button
            key={id}
            type="button"
            onClick={() => setModeId(id as SftModeId)}
            className={`rounded-md border px-3 py-2 text-sm font-semibold ${
              modeId === id
                ? "border-rose-300 bg-rose-300 text-slate-950"
                : "border-slate-700 text-slate-200 hover:border-slate-500"
            }`}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="rounded-lg border border-slate-800 bg-slate-950 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            Prompt
          </p>
          <p className="mt-3 rounded-md border border-slate-800 bg-slate-900 p-3 font-mono text-sm leading-6 text-rose-100">
            {mode.prompt}
          </p>
          <p className="mt-4 text-xs font-semibold uppercase tracking-wide text-slate-400">
            Response
          </p>
          <p className="mt-3 text-sm leading-6 text-slate-300">
            {mode.response}
          </p>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-950 p-4">
          <h3 className="text-lg font-semibold text-slate-50">
            Why it matters
          </h3>
          <p role="status" className="mt-3 text-sm leading-6 text-rose-100">
            {mode.diagnosis}
          </p>
          <div className="mt-4 grid gap-2 text-sm leading-6 text-slate-300">
            <p className="rounded-md border border-slate-800 bg-slate-900 px-3 py-2">
              SFT size is usually thousands to millions of examples, not
              trillions of tokens.
            </p>
            <p className="rounded-md border border-slate-800 bg-slate-900 px-3 py-2">
              Good SFT data must resemble the prompt distribution the final
              model will see.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

function LoRAWorkbench() {
  const [taskId, setTaskId] = useState<AdapterTaskId>("spam");
  const [rank, setRank] = useState(8);
  const [modeId, setModeId] = useState<AdapterModeId>("lora");
  const task = adapterTasks[taskId];
  const mode = adapterModes[modeId];
  const fullMatrixParameters = 4096 * 4096;
  const trainableAdapterParameters = rank * (4096 + 4096);
  const trainedPercent =
    modeId === "full"
      ? 100
      : (trainableAdapterParameters / fullMatrixParameters) * 100;

  return (
    <section
      data-testid="lora-workbench"
      className="rounded-lg border border-slate-800 bg-slate-900 p-5"
    >
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-violet-300">
            Parameter-efficient fine-tuning
          </p>
          <h2 className="mt-2 text-2xl font-semibold text-slate-50">
            Freeze the base matrix, train a small task adapter
          </h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">
            LoRA decomposes the update to a weight matrix into low-rank factors.
            QLoRA keeps the base weights quantized to relieve VRAM pressure
            while the adapter path remains trainable.
          </p>
        </div>
        <div className="rounded-md border border-slate-800 bg-slate-950 px-3 py-2 text-sm font-semibold text-violet-200">
          Trainable: {formatPercent(trainedPercent)}
        </div>
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-[0.85fr_1.15fr]">
        <div className="space-y-5">
          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
              Tuning mode
            </h3>
            <div className="mt-3 flex flex-wrap gap-2">
              {Object.entries(adapterModes).map(([id, item]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setModeId(id as AdapterModeId)}
                  className={`rounded-md border px-3 py-2 text-sm font-semibold ${
                    modeId === id
                      ? "border-violet-300 bg-violet-300 text-slate-950"
                      : "border-slate-700 text-slate-200 hover:border-slate-500"
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
              Adapter task
            </h3>
            <div className="mt-3 grid gap-2">
              {Object.entries(adapterTasks).map(([id, item]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setTaskId(id as AdapterTaskId)}
                  className={`rounded-md border px-3 py-2 text-left text-sm font-semibold ${
                    taskId === id
                      ? "border-sky-400 bg-sky-400 text-slate-950"
                      : "border-slate-700 bg-slate-950 text-slate-200 hover:border-slate-500"
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label
              htmlFor="lora-rank"
              className="text-sm font-semibold text-slate-200"
            >
              LoRA rank: {rank}
            </label>
            <input
              id="lora-rank"
              type="range"
              min="4"
              max="64"
              step="4"
              value={rank}
              onChange={(event) => setRank(Number(event.target.value))}
              className="mt-2 w-full accent-violet-300"
              disabled={modeId === "full"}
            />
          </div>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-950 p-4">
          <div className="grid gap-3 md:grid-cols-[1fr_auto_1fr] md:items-center">
            <div className="rounded-md border border-slate-800 bg-slate-900 p-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                Frozen base
              </p>
              <p className="mt-2 text-lg font-semibold text-slate-50">W0</p>
              <p className="mt-1 text-xs leading-5 text-slate-400">
                16,777,216 weights in this 4096 by 4096 example
              </p>
            </div>
            <div className="text-center text-lg font-bold text-violet-200">
              +
            </div>
            <div className="rounded-md border border-violet-500/40 bg-violet-950/30 p-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-violet-200">
                Trainable adapter
              </p>
              <p className="mt-2 text-lg font-semibold text-slate-50">
                B x A, rank {rank}
              </p>
              <p className="mt-1 text-xs leading-5 text-slate-300">
                {trainableAdapterParameters.toLocaleString()} adapter weights
              </p>
            </div>
          </div>

          <div className="mt-4 rounded-md border border-slate-800 bg-slate-900 p-3">
            <p className="text-sm font-semibold text-violet-100">
              {mode.status}
            </p>
            <p role="status" className="mt-2 text-sm leading-6 text-slate-300">
              {mode.status} {mode.detail} Active task: {task.body}
            </p>
            <p className="mt-3 font-mono text-sm text-emerald-200">
              {task.output}
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

function EvaluationGrid() {
  const dimensions = useMemo(
    () => [
      ["MMLU", "General multitask knowledge"],
      ["GSM8K", "Math reasoning"],
      ["HumanEval", "Code generation"],
      ["Arena votes", "Human preference under pairwise comparison"],
    ],
    [],
  );

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <p className="text-sm font-semibold uppercase tracking-wide text-sky-300">
        Evaluation
      </p>
      <h2 className="mt-2 text-2xl font-semibold text-slate-50">
        No single score tells you what the model is good for
      </h2>
      <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {dimensions.map(([label, body]) => (
          <div
            key={label}
            className="rounded-md border border-slate-800 bg-slate-950 p-3"
          >
            <p className="font-semibold text-slate-50">{label}</p>
            <p className="mt-2 text-sm leading-6 text-slate-300">{body}</p>
          </div>
        ))}
      </div>
      <p className="mt-4 rounded-md border border-amber-500/40 bg-amber-950/20 px-3 py-2 text-sm leading-6 text-amber-100">
        The evaluation warning from the lecture is operational: compare training
        mixtures, watch for train-on-test-task effects, and do not confuse user
        preference with factuality, safety, or the product target user.
      </p>
    </section>
  );
}

export default function StanfordCME295Lecture4LearningPage({
  experience,
}: Props) {
  return (
    <main className="bg-slate-950 text-slate-50">
      <section className="border-b border-slate-800">
        <div className="mx-auto grid min-h-[520px] w-full max-w-6xl items-center gap-8 px-4 py-10 lg:grid-cols-[1.05fr_0.95fr] lg:py-14">
          <div className="space-y-6">
            <div className="space-y-3">
              <p className="text-sm font-semibold uppercase tracking-wide text-sky-300">
                Stanford CME295 Lecture 4
              </p>
              <h1 className="max-w-3xl text-3xl font-semibold tracking-normal text-slate-50 md:text-5xl md:leading-tight">
                Run the LLM training pipeline
              </h1>
              <p className="max-w-2xl text-base leading-7 text-slate-300 md:text-lg">
                Lecture 4 turns the LLM story from architecture into operations:
                massive next-token pretraining, distributed training, exact
                attention speedups, instruction tuning, evaluation limits, and
                low-rank adapters.
              </p>
              <p className="text-sm font-semibold text-emerald-300">
                {experience.durationMinutes} min / {experience.level}
              </p>
            </div>

            <div className="grid gap-2 sm:grid-cols-2">
              {experience.outcomes.map((outcome) => (
                <div
                  key={outcome}
                  className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-sm leading-5 text-slate-200"
                >
                  {outcome}
                </div>
              ))}
            </div>
          </div>

          <TrainingPipelinePreview />
        </div>
      </section>

      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-10">
        <TrainingLifecycleWorkbench />

        <section className="grid gap-5 lg:grid-cols-[0.95fr_1.05fr]">
          <ScalingBudgetLab />
          <FormulaBlock
            title="Pretraining compute couples model size and data size"
            formula={String.raw`\[
C_{\mathrm{train}} \propto N_{\mathrm{parameters}} \times D_{\mathrm{tokens}}
\]`}
            explanation="Exact FLOP formulas depend on architecture, but the lecture's useful mental model is that more parameters and more training tokens both raise compute."
          />
        </section>

        <MemoryOptimizationLab />

        <FormulaBlock
          title="FlashAttention preserves exact attention while changing memory traffic"
          formula={String.raw`\[
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]`}
          explanation="The mathematical result is the same as standard attention. The implementation avoids repeatedly materializing large intermediate matrices in HBM by computing tiled blocks through SRAM."
        />

        <section className="grid gap-5 lg:grid-cols-2">
          <SftBehaviorPanel />
          <EvaluationGrid />
        </section>

        <LoRAWorkbench />

        <FormulaBlock
          title="LoRA trains a low-rank update instead of the whole matrix"
          formula={String.raw`\[
W = W_0 + BA,\qquad B\in\mathbb{R}^{d_{\mathrm{out}}\times r},\quad A\in\mathbb{R}^{r\times d_{\mathrm{in}}}
\]`}
          explanation="W0 is the frozen pretrained matrix. A small rank r makes B and A much cheaper to train than the full d_out by d_in matrix, and QLoRA stores frozen W0 in quantized form."
        />

        <section className="grid gap-5 lg:grid-cols-2">
          <CheckForUnderstanding
            testId="flashattention-check"
            title="Check: FlashAttention"
            question="Why can FlashAttention be faster even when it recomputes some values in the backward pass?"
            options={[
              {
                label:
                  "It approximates attention by dropping low-probability tokens.",
                explanation:
                  "FlashAttention is an exact attention algorithm. The speedup is not from changing the attention distribution.",
              },
              {
                label:
                  "It reduces slow HBM reads and writes enough that extra FLOPs can still lower runtime.",
                explanation:
                  "The lecture's key point is IO-awareness: memory traffic, not arithmetic alone, can be the bottleneck.",
              },
              {
                label: "It removes the softmax operation from self-attention.",
                explanation:
                  "The softmax is still part of the exact attention computation; it is computed in a tiled way.",
              },
            ]}
            correctIndex={1}
          />

          <CheckForUnderstanding
            testId="sft-loss-check"
            title="Check: SFT objective"
            question="In instruction tuning, where does the supervised loss apply?"
            options={[
              {
                label:
                  "Only on the desired response tokens, conditioned on the instruction.",
                explanation:
                  "The prompt is context. The model is trained to predict the target response tokens that follow it.",
              },
              {
                label:
                  "Only on the instruction tokens, because the response is generated at inference time.",
                explanation:
                  "The response is exactly the supervised target during SFT; the instruction is the conditioning input.",
              },
              {
                label: "Only on benchmark scores after generation finishes.",
                explanation:
                  "Benchmarks evaluate a trained model. They are not the token-level SFT training loss.",
              },
            ]}
            correctIndex={0}
          />
        </section>

        <RecapSection
          title="Lecture 4 mental model"
          items={[
            "Pretraining teaches general token prediction from massive language and code mixtures.",
            "Scaling laws improve models with more compute, but compute-optimal training balances parameters and tokens.",
            "Training must store parameters, activations, gradients, and optimizer state, so parallelism and ZeRO attack memory pressure.",
            "FlashAttention is exact attention reorganized around GPU memory hierarchy.",
            "Mixed precision and quantization trade numeric detail for memory and throughput when accuracy is preserved.",
            "SFT and instruction tuning change behavior with high-quality input-output data, but evaluation remains multi-dimensional.",
            "LoRA and QLoRA adapt models cheaply by training low-rank adapters while freezing the base model.",
          ]}
        />

        <section className="rounded-lg border border-emerald-500/40 bg-emerald-950/20 p-5">
          <h2 className="text-xl font-semibold text-emerald-100">
            Ready for the Lecture 4 MCQs
          </h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">
            Practice should now feel like diagnosing an LLM training system:
            identify the stage, name the bottleneck, choose the optimization,
            and explain what changes in the model behavior or trainable weights.
          </p>
          <div className="mt-5">
            <QuizTransitionButton sourceId={experience.sourceId} />
          </div>
        </section>
      </div>
    </main>
  );
}
