import {
  ALL_SOURCE_IDS,
  QUESTION_SOURCES,
  SOURCE_SERIES,
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
    sourceId: "crash-probability-l3",
    shortTitle: "Likelihood, Loss, Softmax",
    title:
      "Crash Course Probability L3: Likelihood, Loss, Softmax, and Deep Learning",
    summary:
      "Build intuition for how neural networks turn raw scores into probabilities and learn by lowering cross-entropy loss.",
    durationMinutes: 12,
    level: "After Probability Lectures 1-2",
    sourceMaterialPath:
      "lib/other/Crash Courses/Probability/transcripts-and-files/Lecture 3 - overview.md",
    outcomes: [
      "Separate logits, probabilities, and decisions.",
      "Use softmax as a normalization step, not a magic classifier.",
      "Connect likelihood, negative log-likelihood, and cross-entropy.",
      "Recognize entropy as uncertainty in a categorical distribution.",
    ],
  },
  {
    sourceId: "crash-probability-l4",
    shortTitle: "RL Over Time",
    title:
      "Crash Course Probability L4: Probability Over Time: Reinforcement Learning",
    summary:
      "Use a gridworld decision lab to connect transition probabilities, policies, discounted return, value functions, and exploration.",
    durationMinutes: 15,
    level: "After Probability Lectures 1-3",
    sourceMaterialPath:
      "lib/other/Crash Courses/Probability/transcripts-and-files/Lecture 4 - overview.md",
    outcomes: [
      "Trace the state-action-reward-next-state loop.",
      "Separate environment transitions from policy probabilities.",
      "Compute expected one-step return from stochastic outcomes.",
      "Explain why discounting, value functions, and exploration matter.",
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
] as const satisfies readonly LearningExperience[];

const QUESTION_SOURCE_BY_ID = new Map(
  QUESTION_SOURCES.map((source) => [source.id, source]),
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

export function getLearningCourses(): LearningCourse[] {
  return SOURCE_SERIES.flatMap((series) => {
    const experiences = LEARNING_EXPERIENCES.filter((experience) => {
      const source = getQuestionSourceForLearningExperience(experience);
      return source?.seriesId === series.id;
    });

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
