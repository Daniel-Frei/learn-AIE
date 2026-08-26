import {
  ALL_SOURCE_IDS,
  QUESTION_SOURCES,
  SOURCE_SERIES,
  type QuestionSource,
  type SourceId,
  type SourceSeriesId,
} from "./quiz";

export type LearningExperience = {
  sourceId: SourceId;
  shortTitle: string;
  title: string;
  summary: string;
  durationMinutes: number;
  level: string;
  sourceMaterialPath: string;
  outcomes: string[];
};

export type LearningCourse = {
  seriesId: SourceSeriesId;
  label: string;
  experiences: readonly LearningExperience[];
  totalDurationMinutes: number;
};

export const LEARNING_EXPERIENCES = [
  {
    sourceId: "cme295-lect1",
    shortTitle: "Text Into Transformers",
    title: "Stanford CME295 Lecture 1: Transformers & LLMs",
    summary:
      "Follow text from NLP task framing through tokenization, embeddings, attention, and the encoder-decoder transformer loop.",
    durationMinutes: 15,
    level: "Introductory NLP with ML basics",
    sourceMaterialPath:
      "lib/lectures/Stanford CME295 Transformers & LLMs/transcripts-and-files/lecture1_transformers.md",
    outcomes: [
      "Distinguish classification, structured prediction, and generation tasks.",
      "Compare word, subword, and character tokenization tradeoffs.",
      "Explain why learned embeddings improve on one-hot token ids.",
      "Trace queries, keys, values, masks, and cross-attention through a transformer.",
    ],
  },
  {
    sourceId: "cme295-lect2",
    shortTitle: "Transformer Upgrade Knobs",
    title: "Stanford CME295 Lecture 2: Transformer-Based Models & Tricks",
    summary:
      "Tune positional encodings, attention-score tricks, normalization, efficient attention patterns, and BERT-style encoder-only pretraining.",
    durationMinutes: 16,
    level: "After CME295 Lecture 1",
    sourceMaterialPath:
      "lib/lectures/Stanford CME295 Transformers & LLMs/transcripts-and-files/lecture 2 - transcript.md",
    outcomes: [
      "Compare learned, sinusoidal, relative-bias, ALiBi, and RoPE position strategies.",
      "Explain why modern transformers use pre-norm, RMSNorm, sparse attention, MQA, and GQA.",
      "Choose between encoder-decoder, encoder-only, and decoder-only transformer families.",
      "Trace BERT inputs, MLM/NSP pretraining, fine-tuning, DistilBERT, and RoBERTa changes.",
    ],
  },
  {
    sourceId: "cme295-lect3",
    shortTitle: "LLM Runtime Trace",
    title: "Stanford CME295 Lecture 3: Large Language Models, MoE & Inference",
    summary:
      "Trace one generation request through decoder-only LLM architecture, sparse MoE routing, decoding controls, prompting, KV-cache memory, and token acceleration.",
    durationMinutes: 18,
    level: "After CME295 Lectures 1-2",
    sourceMaterialPath:
      "lib/lectures/Stanford CME295 Transformers & LLMs/transcripts-and-files/lecture 3 - transcript.md",
    outcomes: [
      "Define modern LLMs as scaled decoder-only next-token models.",
      "Trace sparse MoE routing, top-k expert selection, and routing collapse.",
      "Compare greedy, beam, top-k, top-p, temperature, and guided decoding.",
      "Place KV caching, GQA, PagedAttention, latent attention, speculative decoding, and multi-token prediction in the serving trace.",
    ],
  },
  {
    sourceId: "cme295-lect4",
    shortTitle: "LLM Training Pipeline",
    title: "Stanford CME295 Lecture 4: LLM Training, Scaling & Alignment",
    summary:
      "Diagnose the full training lifecycle: pretraining scale, memory bottlenecks, FlashAttention, SFT, evaluation, LoRA, and QLoRA.",
    durationMinutes: 19,
    level: "After CME295 Lectures 1-3",
    sourceMaterialPath:
      "lib/lectures/Stanford CME295 Transformers & LLMs/transcripts-and-files/lecture 4 - transcript.md",
    outcomes: [
      "Explain pretraining as next-token prediction over massive language and code mixtures.",
      "Balance model size, token count, FLOPs, and knowledge-cutoff tradeoffs.",
      "Match training bottlenecks to data/model parallelism, ZeRO, FlashAttention, and mixed precision.",
      "Compare SFT, instruction tuning, evaluation, LoRA, QLoRA, and adapter swapping.",
    ],
  },
  {
    sourceId: "cme295-lect5",
    shortTitle: "Preference Tuning Workbench",
    title: "Stanford CME295 Lecture 5: LLM Preference Tuning, RLHF & DPO",
    summary:
      "Build preference pairs, reward-model gaps, PPO guardrails, Best-of-N reranking, and DPO log-ratio contrasts into one alignment workbench.",
    durationMinutes: 18,
    level: "After CME295 Lectures 1-4",
    sourceMaterialPath:
      "lib/lectures/Stanford CME295 Transformers & LLMs/transcripts-and-files/lecture 5 - transcript.md",
    outcomes: [
      "Convert model behavior complaints into pointwise, pairwise, or listwise preference data.",
      "Use Bradley-Terry reward gaps to reason about pairwise preference probability.",
      "Explain how PPO-style RLHF uses reward, value, clipping, and KL/reference pressure.",
      "Compare PPO, Best-of-N, and DPO by cost, model components, and failure modes.",
    ],
  },
  {
    sourceId: "cme295-lect9",
    shortTitle: "Course Recap Synthesis",
    title: "Stanford CME295 Lecture 9: Course Recap & Frontiers",
    summary:
      "Rebuild the whole transformer course from the Lecture 9 recap, then test transfer through ViT, VLM, diffusion-LLM, and frontier labs.",
    durationMinutes: 20,
    level: "After CME295 Lectures 1-8",
    sourceMaterialPath:
      "lib/lectures/Stanford CME295 Transformers & LLMs/transcripts-and-files/lecture 9 - transcript.md",
    outcomes: [
      "Trace one LLM answer through representation, attention, training, post-training, systems, and evaluation layers.",
      "Explain the recap mechanisms from tokenization and embeddings through model families, LLM runtime, preference tuning, reasoning, agents, and evaluation.",
      "Explain how image patches, visual tokens, and masked diffusion reuse transformer-era ideas outside standard text generation.",
      "Diagnose frontier claims as architecture, data, serving, hardware, or open-problem issues.",
    ],
  },
  {
    sourceId: "crash-probability-l0",
    shortTitle: "Probability Notation Gym",
    title: "Crash Course Probability L0: Math and Notation Foundations",
    summary:
      "Turn normalization, odds, weighted averages, powers, logarithms, geometric series, vectors, and Gaussian notation into inspectable probability tools.",
    durationMinutes: 32,
    level: "Foundation for the Probability course",
    sourceMaterialPath:
      "lib/other/Crash Courses/Probability/transcripts-and-files/Syllabus.md",
    outcomes: [
      "Read probability notation as operations on concrete quantities.",
      "Normalize weights and move between probability and odds.",
      "Use weighted averages, powers, roots, exponentials, logarithms, and geometric series.",
      "Recognize the vectors, matrices, and Gaussian symbols reused throughout AI.",
    ],
  },
  {
    sourceId: "crash-probability-l1",
    shortTitle: "Events and Random Variables",
    title: "Crash Course Probability L1: Probability as the Language of AI",
    summary:
      "Build probability from sample spaces and event algebra through counting, random variables, distributions, expectation, variance, and calibration.",
    durationMinutes: 38,
    level: "Introductory probability for AI",
    sourceMaterialPath:
      "lib/other/Crash Courses/Probability/transcripts-and-files/Lecture 1 - overview.md",
    outcomes: [
      "Distinguish outcomes, events, random variables, and distributions.",
      "Derive complement and addition rules from the probability axioms.",
      "Choose the correct counting rule for ordered and unordered spaces.",
      "Interpret PMFs, PDFs, expectation, variance, and calibrated probability.",
    ],
  },
  {
    sourceId: "crash-probability-l2",
    shortTitle: "Evidence and Bayes",
    title:
      "Crash Course Probability L2: Conditional Probability, Bayes, and Dependence",
    summary:
      "Shrink probability universes with evidence, read joint tables, audit independence, combine hidden paths, and reverse conditions with Bayes and natural frequencies.",
    durationMinutes: 38,
    level: "After Probability Lecture 1",
    sourceMaterialPath:
      "lib/other/Crash Courses/Probability/transcripts-and-files/Lecture 2 - overview.md",
    outcomes: [
      "Interpret conditioning as restricting and renormalizing a sample space.",
      "Read joint, marginal, and conditional probability from the same distribution.",
      "Use independence, the product rule, and the law of total probability correctly.",
      "Apply Bayes with base rates and natural-frequency counts.",
    ],
  },
  {
    sourceId: "crash-probability-l3",
    shortTitle: "Likelihood, Loss, Softmax",
    title:
      "Crash Course Probability L3: Likelihood, Loss, Softmax, and Deep Learning",
    summary:
      "Inspect the full training path from logits and softmax through likelihood, log-likelihood, cross-entropy, entropy, and a gradient update.",
    durationMinutes: 36,
    level: "After Probability Lectures 1-2",
    sourceMaterialPath:
      "lib/other/Crash Courses/Probability/transcripts-and-files/Lecture 3 - overview.md",
    outcomes: [
      "Turn logits into stable probability distributions with softmax.",
      "Compute sequence likelihood, log-likelihood, and negative log-likelihood.",
      "Connect one-hot and soft targets to cross-entropy and gradient direction.",
      "Separate entropy, confidence, calibration, and correctness.",
    ],
  },
  {
    sourceId: "crash-probability-l4",
    shortTitle: "RL Over Time",
    title:
      "Crash Course Probability L4: Probability Over Time: Reinforcement Learning",
    summary:
      "Use a stochastic decision world to connect transitions, Markov state, policies, paths, discounted return, value, recursive expectation, and exploration.",
    durationMinutes: 38,
    level: "After Probability Lectures 1-3",
    sourceMaterialPath:
      "lib/other/Crash Courses/Probability/transcripts-and-files/Lecture 4 - overview.md",
    outcomes: [
      "Separate policy probabilities from environment transition probabilities.",
      "Audit whether a state representation satisfies the Markov property.",
      "Compute trajectory probability, discounted return, and one-step value.",
      "Solve waiting times with first-step recursion and explain controlled exploration.",
    ],
  },
  {
    sourceId: "crash-probability-l5",
    shortTitle: "Generation Sampling Lab",
    title:
      "Crash Course Probability L5: Sampling, Latent Variables, and Diffusion Models",
    summary:
      "Forge generated outcomes through token sampling, latent-variable inference, Gaussian noise, forward diffusion, learned reverse transitions, and guidance.",
    durationMinutes: 40,
    level: "After Probability Lectures 1-4",
    sourceMaterialPath:
      "lib/other/Crash Courses/Probability/transcripts-and-files/Lecture 5 - overview.md",
    outcomes: [
      "Compare greedy, temperature, top-k, and top-p decoding as distribution transforms.",
      "Factor autoregressive sequence probability into repeated conditional samples.",
      "Marginalize latent causes and infer them from observations.",
      "Trace Gaussian forward noising, learned reverse diffusion, and conditional guidance.",
    ],
  },
  {
    sourceId: "clinical-trials-l3",
    shortTitle: "Statistics and Evidence Interpretation",
    title:
      "Clinical Trials Crash Course L3: Statistics and Evidence Interpretation",
    summary:
      "Practice reading clinical trial results as effect size, uncertainty, patient relevance, and total evidence rather than a simple positive-or-negative label.",
    durationMinutes: 14,
    level: "After Clinical Trials Lectures 1-2",
    sourceMaterialPath:
      "lib/other/Crash Courses/Clinical Trials/transcripts-and-files/Lecture 3 - overview.md",
    outcomes: [
      "Translate relative claims into absolute impact and NNT.",
      "Use confidence intervals to judge precision and plausible effects.",
      "Separate statistical significance from clinical meaning.",
      "Read survival, hazard-ratio, forest-plot, and evidence-synthesis claims cautiously.",
    ],
  },
  {
    sourceId: "ai-agents-memory-survey",
    shortTitle: "Memory System Workbench",
    title: "AI Agents: Memory in the Age of AI Agents",
    summary:
      "Design an agent memory system through the survey's forms, functions, and dynamics taxonomy, then inspect retrieval, evolution, and trust tradeoffs.",
    durationMinutes: 18,
    level: "After basic LLM agent concepts",
    sourceMaterialPath:
      "lib/other/AI agents/Memory/transcripts-and-files/Memory in the Age of AI Agents - A Survey.pdf",
    outcomes: [
      "Separate memory forms from memory functions and lifecycle dynamics.",
      "Choose between token-level, parametric, and latent memory carriers for concrete agent pressures.",
      "Trace formation, evolution, and retrieval decisions through a managed memory subsystem.",
      "Explain why multimodal, shared, RL-controlled, and trustworthy memory are frontier problems.",
    ],
  },
  {
    sourceId: "ai-agents-agent-native-memory",
    shortTitle: "Memory Evaluation Console",
    title: "AI Agents: Are We Ready For An Agent-Native Memory System?",
    summary:
      "Stress agent memory architectures across workload, retrieval, update, horizon, and cost bottlenecks.",
    durationMinutes: 18,
    level: "After basic LLM agent concepts",
    sourceMaterialPath:
      "lib/other/AI agents/Memory/transcripts-and-files/Are We Ready For An Agent-Native Memory System.pdf",
    outcomes: [
      "Decompose agent memory systems into representation, extraction, retrieval, and maintenance modules.",
      "Match memory architectures to workload bottlenecks across LongMemEval, LoCoMo, DB-Bench, and cost traces.",
      "Explain why evidence assembly, update fidelity, and horizon structure matter beyond top-1 retrieval.",
      "Use component ablations to reason about representation granularity, extraction coverage, retrieval planning, and conservative maintenance.",
    ],
  },
  {
    sourceId: "ai-agents-atommem",
    shortTitle: "AtomMem Pipeline Debugger",
    title: "AI Agents: AtomMem",
    summary:
      "Debug the AtomMem pipeline from noisy dialogue through atomic facts, verification, event/profile memory, graph recall, and evaluation tradeoffs.",
    durationMinutes: 18,
    level: "After basic LLM agent and memory concepts",
    sourceMaterialPath:
      "lib/other/AI agents/Memory/transcripts-and-files/AtomMem Building Simple and Effective Memory System.pdf",
    outcomes: [
      "Explain why AtomMem uses self-contained atomic facts instead of raw logs or loose summaries.",
      "Trace verification, residual storage, event memory, and temporal profiles through a memory update.",
      "Diagnose how entity, event, and temporal graph edges support associative recall.",
      "Interpret AtomMem's ablations, retrieval-capacity tradeoffs, latency results, and limitations.",
    ],
  },
] as const satisfies readonly LearningExperience[];

const QUESTION_SOURCE_BY_ID = new Map(
  QUESTION_SOURCES.map((source) => [source.id, source]),
);

const QUESTION_SOURCE_INDEX_BY_ID = new Map(
  QUESTION_SOURCES.map((source, index) => [source.id, index]),
);

export function getLearningExperience(
  sourceId: string,
): LearningExperience | null {
  return (
    LEARNING_EXPERIENCES.find(
      (experience) => experience.sourceId === sourceId,
    ) ?? null
  );
}

export function getQuestionSourceForLearningExperience(
  experience: Pick<LearningExperience, "sourceId">,
) {
  return QUESTION_SOURCE_BY_ID.get(experience.sourceId) ?? null;
}

export function getQuestionSourceSequenceLabel(
  source: Pick<QuestionSource, "id" | "label" | "title">,
): string {
  const searchableText = `${source.label} ${source.title} ${source.id}`;
  const chapterMatch = searchableText.match(/\b(?:chapter|chap)\.?\s*(\d+)\b/i);
  if (chapterMatch) return `Chapter ${chapterMatch[1]}`;

  const lectureMatch =
    searchableText.match(/\blecture\s*(\d+)\b/i) ??
    searchableText.match(/\blect\s*(\d+)\b/i) ??
    searchableText.match(/\bl(\d+)\b/i);
  if (lectureMatch) return `Lecture ${lectureMatch[1]}`;

  return source.label;
}

export function getLearningExperienceSequenceLabel(
  experience: Pick<LearningExperience, "sourceId">,
): string {
  const source = getQuestionSourceForLearningExperience(experience);
  return source ? getQuestionSourceSequenceLabel(source) : experience.sourceId;
}

function getLearningExperienceSourceIndex(
  experience: Pick<LearningExperience, "sourceId">,
): number {
  return (
    QUESTION_SOURCE_INDEX_BY_ID.get(experience.sourceId) ??
    Number.MAX_SAFE_INTEGER
  );
}

export function getLearningCourses(): LearningCourse[] {
  return SOURCE_SERIES.flatMap((series) => {
    const experiences = LEARNING_EXPERIENCES.filter((experience) => {
      const source = getQuestionSourceForLearningExperience(experience);
      return source?.seriesId === series.id;
    }).sort(
      (first, second) =>
        getLearningExperienceSourceIndex(first) -
        getLearningExperienceSourceIndex(second),
    );

    if (experiences.length === 0) return [];

    return [
      {
        seriesId: series.id,
        label: series.label,
        experiences,
        totalDurationMinutes: experiences.reduce(
          (sum, experience) => sum + experience.durationMinutes,
          0,
        ),
      },
    ];
  });
}

export function getLearningCourse(seriesId: string): LearningCourse | null {
  return (
    getLearningCourses().find((course) => course.seriesId === seriesId) ?? null
  );
}

export function getLearningExperienceCourse(
  experience: Pick<LearningExperience, "sourceId">,
): Pick<LearningCourse, "seriesId" | "label"> | null {
  const source = getQuestionSourceForLearningExperience(experience);
  if (!source) return null;

  return {
    seriesId: source.seriesId,
    label: source.seriesLabel,
  };
}

export function getLearningCoursePath(seriesId: SourceSeriesId): string {
  return `/learn/${seriesId}`;
}

export function getLearningExperiencePath(
  experience: Pick<LearningExperience, "sourceId">,
): string {
  const course = getLearningExperienceCourse(experience);
  return course
    ? `${getLearningCoursePath(course.seriesId)}/${experience.sourceId}`
    : `/learn/${experience.sourceId}`;
}

export function getDuplicateLearningExperienceSourceIds(
  experiences: readonly { sourceId: string }[],
): string[] {
  const seen = new Set<string>();
  const duplicates = new Set<string>();

  for (const experience of experiences) {
    if (seen.has(experience.sourceId)) {
      duplicates.add(experience.sourceId);
    }
    seen.add(experience.sourceId);
  }

  return Array.from(duplicates).sort();
}

export function getUnknownLearningExperienceSourceIds(
  experiences: readonly { sourceId: string }[],
): string[] {
  const knownSourceIds = new Set<string>(ALL_SOURCE_IDS);

  return experiences
    .map((experience) => experience.sourceId)
    .filter((sourceId) => !knownSourceIds.has(sourceId))
    .sort();
}
