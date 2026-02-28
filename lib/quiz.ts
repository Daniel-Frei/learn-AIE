// lib/quiz.ts
import { chapter1Questions } from "./chapter1";
import { chapter2Questions } from "./chapter2";
import { chapter3Questions } from "./chapter3";
import { aieChapter2Questions } from "./books/AIE_building_apps/chapter2";
import { aieChapter3Questions } from "./books/AIE_building_apps/chapter3";
import { aieChapter4Questions } from "./books/AIE_building_apps/chapter4";
import { stanfordCME295Lecture1Questions } from "./lectures/Stanford CME295 Transformers & LLMs/lecture1_transformer";
import { stanfordCME295Lecture2Questions } from "./lectures/Stanford CME295 Transformers & LLMs/lecture2_models";
import { stanfordCME295Lecture3LLMsQuestions } from "./lectures/Stanford CME295 Transformers & LLMs/lecture3_LLMs";
import { stanfordCME295Lecture4TrainingQuestions } from "./lectures/Stanford CME295 Transformers & LLMs/lecture4_training";
import { cs224rLecture1IntroQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture1_intro";
import { cs224rLecture2ImitationLearningQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture2_Imitation Learning";
import { cs224rLecture3PolicyGradientsQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture3_Policy Gradients";
import { cs224rLecture4ActorCriticQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture4_Actor-Critic Methods";
import { lecture5_OffPolicyActorCriticQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture5_Off-Policy Actor Critic Methods";
import { OtherRL_introductiontoReinforcementLearning } from "./lectures/Other RL/introduction to Reinforcement Learning";
import { L5_DeepReinforcementLearning } from "./lectures/MIT 6.S191 Deep Learning 2025/L5_Deep Reinforcement Learning";
import { L1_IntroductionToNeuralNetworksAndDeepLearning as MIT6S191L1IntroductionQuestions } from "./lectures/MIT 6.S191 Deep Learning 2025/L1_Introduction";
import { L1_IntroductionToNeuralNetworksAndDeepLearning } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L1_ Introduction to Neural Networks and Deep Learning";
import { L2_TrainingDeepNNs } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L2_Training Deep NNs";
import { mixedQuestions } from "./other/other";

export type Difficulty = "easy" | "medium" | "hard";

export type Question = {
  id: string;
  chapter: number;
  /**
   * Author-assigned difficulty label (used as a prior).
   * The *empirical* difficulty score [0,1] is computed from user stats
   * in difficultyStore.ts.
   */
  difficulty: Difficulty;
  prompt: string;
  options: {
    text: string;
    isCorrect: boolean;
  }[];
  explanation: string;
};

// ------------------------------
// Central question set registry
// ------------------------------

export const QUESTION_SOURCES = [
  {
    id: "chapter-1" as const,
    label: "Chapter 1 only",
    title: "Chapter 1 Quiz – Analyzing Text Data with Deep Learning",
    questions: chapter1Questions,
  },
  {
    id: "chapter-2" as const,
    label: "Chapter 2 only",
    title: "Chapter 2 Quiz – The Transformer and Modern NLP",
    questions: chapter2Questions,
  },
  {
    id: "chapter-3" as const,
    label: "Chapter 3 only",
    title: "Chapter 3 Quiz – Large Language Models & Prompting",
    questions: chapter3Questions,
  },
  {
    id: "aie-build-app-ch2" as const,
    label: "AIE build app - Chap 2",
    title: "AIE build app – Chapter 2: Understanding Foundation Models",
    questions: aieChapter2Questions,
  },
  {
    id: "aie-build-app-ch3" as const,
    label: "AIE build app - Chap 3",
    title:
      "AIE build app – Chapter 3: Evaluating Foundation Models & Applications",
    questions: aieChapter3Questions,
  },
  {
    id: "aie-build-app-ch4" as const,
    label: "AIE build app - Chap 4",
    title: "AIE build app – Chapter 4: Building Agents with Foundation Models",
    questions: aieChapter4Questions,
  },
  {
    id: "cme295-lect1" as const,
    label: "Stanford CME295 Lecture 1",
    title: "Stanford CME295 Lecture 1: Transformers & LLMs",
    questions: stanfordCME295Lecture1Questions,
  },
  {
    id: "cme295-lect2" as const,
    label: "Stanford CME295 Lecture 2",
    title: "Stanford CME295 Lecture 2: Transformer-Based Models & Tricks",
    questions: stanfordCME295Lecture2Questions,
  },
  {
    id: "cme295-lect3" as const,
    label: "Stanford CME295 Lecture 3",
    title: "Stanford CME295 Lecture 3: Large Language Models, MoE & Inference",
    questions: stanfordCME295Lecture3LLMsQuestions,
  },
  {
    id: "cme295-lect4" as const,
    label: "Stanford CME295 Lecture 4",
    title: "Stanford CME295 Lecture 4: LLM Training, Scaling & Alignment",
    questions: stanfordCME295Lecture4TrainingQuestions,
  },
  {
    id: "cs224r-lect1" as const,
    label: "Stanford CS224R Lecture 1",
    title: "Stanford CS224R Lecture 1: Intro to Deep Reinforcement Learning",
    questions: cs224rLecture1IntroQuestions,
  },
  {
    id: "cs224r-lect2" as const,
    label: "Stanford CS224R Lecture 2",
    title: "Stanford CS224R Lecture 2: Imitation Learning",
    questions: cs224rLecture2ImitationLearningQuestions,
  },
  {
    id: "cs224r-lect3" as const,
    label: "Stanford CS224R Lecture 3",
    title: "Stanford CS224R Lecture 3: Policy Gradients",
    questions: cs224rLecture3PolicyGradientsQuestions,
  },
  {
    id: "cs224r-lect4" as const,
    label: "Stanford CS224R Lecture 4",
    title: "Stanford CS224R Lecture 4: Actor-Critic Methods",
    questions: cs224rLecture4ActorCriticQuestions,
  },
  {
    id: "cs224r-lect5" as const,
    label: "Stanford CS224R Lecture 5",
    title: "Stanford CS224R Lecture 5: Off-Policy Actor-Critic Methods",
    questions: lecture5_OffPolicyActorCriticQuestions,
  },
  {
    id: "other-rl-intro" as const,
    label: "Other RL Intro",
    title: "Other RL: Introduction to Reinforcement Learning",
    questions: OtherRL_introductiontoReinforcementLearning,
  },
  {
    id: "mit6s191-l1" as const,
    label: "MIT 6.S191 L1",
    title: "MIT 6.S191 L1: Introduction to Deep Learning",
    questions: MIT6S191L1IntroductionQuestions,
  },
  {
    id: "mit6s191-l5" as const,
    label: "MIT 6.S191 L5",
    title: "MIT 6.S191 L5: Deep Reinforcement Learning",
    questions: L5_DeepReinforcementLearning,
  },
  {
    id: "mit15773-l1" as const,
    label: "MIT 15.773 L1",
    title: "MIT 15.773 L1: Introduction to Neural Networks and Deep Learning",
    questions: L1_IntroductionToNeuralNetworksAndDeepLearning,
  },
  {
    id: "mit15773-l2" as const,
    label: "MIT 15.773 L2",
    title: "MIT 15.773 L2: Training Deep NNs",
    questions: L2_TrainingDeepNNs,
  },
  {
    id: "other" as const,
    label: "Other",
    title: "Other Questions",
    questions: mixedQuestions,
  },
];

export type SourceId = (typeof QUESTION_SOURCES)[number]["id"];
export type Mode = SourceId | "all";

// All questions across all sources
export const allQuestions: Question[] = QUESTION_SOURCES.flatMap(
  (s) => s.questions,
);

export const ALL_SOURCE_IDS: SourceId[] = QUESTION_SOURCES.map((s) => s.id);

// Helper: get questions for a given mode
export function getQuestionsForMode(mode: Mode): Question[] {
  if (mode === "all") return allQuestions;
  const src = QUESTION_SOURCES.find((s) => s.id === mode);
  return src ? src.questions : allQuestions;
}

// Helper: get questions for a set of sources (multi-select)
export function getQuestionsForSources(sourceIds: SourceId[]): Question[] {
  const active = sourceIds.length
    ? new Set(sourceIds)
    : new Set(ALL_SOURCE_IDS);
  return QUESTION_SOURCES.filter((s) => active.has(s.id)).flatMap(
    (s) => s.questions,
  );
}

// Helper: title for the current mode (for page header)
export function getTitleForMode(mode: Mode): string {
  if (mode === "all") {
    return "All Chapters Quiz – Text, Transformers & LLMs";
  }
  const src = QUESTION_SOURCES.find((s) => s.id === mode);
  return src ? src.title : "Quiz";
}

// Helper: title for a custom multi-source selection
export function getTitleForSelection(sourceIds: SourceId[]): string {
  const allSelected =
    sourceIds.length === 0 ||
    ALL_SOURCE_IDS.every((id) => sourceIds.includes(id));

  if (allSelected) {
    return "All Chapters Quiz – Text, Transformers & LLMs";
  }

  if (sourceIds.length === 1) {
    const src = QUESTION_SOURCES.find((s) => s.id === sourceIds[0]);
    return src ? src.title : "Quiz";
  }

  const names = QUESTION_SOURCES.filter((s) => sourceIds.includes(s.id)).map(
    (s) => s.label,
  );
  const preview = names.slice(0, 2).join(", ");
  const extra = names.length > 2 ? ` +${names.length - 2} more` : "";

  return `Custom Quiz – ${preview}${extra}`;
}

// Re-export question arrays if you still want direct imports elsewhere
export { chapter1Questions } from "./chapter1";
export { chapter2Questions } from "./chapter2";
export { chapter3Questions } from "./chapter3";
export { aieChapter2Questions } from "./books/AIE_building_apps/chapter2";
export { aieChapter3Questions } from "./books/AIE_building_apps/chapter3";
export { aieChapter4Questions } from "./books/AIE_building_apps/chapter4";
export { stanfordCME295Lecture1Questions } from "./lectures/Stanford CME295 Transformers & LLMs/lecture1_transformer";
export { stanfordCME295Lecture2Questions } from "./lectures/Stanford CME295 Transformers & LLMs/lecture2_models";
export { stanfordCME295Lecture3LLMsQuestions } from "./lectures/Stanford CME295 Transformers & LLMs/lecture3_LLMs";
export { stanfordCME295Lecture4TrainingQuestions } from "./lectures/Stanford CME295 Transformers & LLMs/lecture4_training";
export { cs224rLecture1IntroQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture1_intro";
export { cs224rLecture2ImitationLearningQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture2_Imitation Learning";
export { cs224rLecture3PolicyGradientsQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture3_Policy Gradients";
export { cs224rLecture4ActorCriticQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture4_Actor-Critic Methods";
export { lecture5_OffPolicyActorCriticQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture5_Off-Policy Actor Critic Methods";
export { OtherRL_introductiontoReinforcementLearning } from "./lectures/Other RL/introduction to Reinforcement Learning";
export { L5_DeepReinforcementLearning } from "./lectures/MIT 6.S191 Deep Learning 2025/L5_Deep Reinforcement Learning";
export { L1_IntroductionToNeuralNetworksAndDeepLearning as MIT6S191L1IntroductionQuestions } from "./lectures/MIT 6.S191 Deep Learning 2025/L1_Introduction";
export { L1_IntroductionToNeuralNetworksAndDeepLearning } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L1_ Introduction to Neural Networks and Deep Learning";
export { L2_TrainingDeepNNs } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L2_Training Deep NNs";
export { mixedQuestions } from "./other/other";
