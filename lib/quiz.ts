// lib/quiz.ts
import type { Topic } from "./questionTopics";
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
import { stanfordCME295Lecture5PreferenceTuningQuestions } from "./lectures/Stanford CME295 Transformers & LLMs/lecture5_preference_tuning";
import { stanfordCME295Lecture6ReasoningQuestions } from "./lectures/Stanford CME295 Transformers & LLMs/lecture6_reasoning";
import { stanfordCME295Lecture7RagToolsAgentsQuestions } from "./lectures/Stanford CME295 Transformers & LLMs/lecture7_rag_tools_agents";
import { stanfordCME295Lecture8EvaluationQuestions } from "./lectures/Stanford CME295 Transformers & LLMs/lecture8_evaluation";
import { stanfordCME295Lecture9SynthesisFrontiersQuestions } from "./lectures/Stanford CME295 Transformers & LLMs/lecture9_synthesis_frontiers";
import { cs224rLecture1IntroQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture1_intro";
import { cs224rLecture2ImitationLearningQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture2_Imitation Learning";
import { cs224rLecture3PolicyGradientsQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture3_Policy Gradients";
import { cs224rLecture4ActorCriticQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture4_Actor-Critic Methods";
import { lecture5_OffPolicyActorCriticQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture5_Off-Policy Actor Critic Methods";
import { cs224rLecture6QLearningQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture6_Q-Learning";
import { OtherRL_introductiontoReinforcementLearning } from "./lectures/Other RL/introduction to Reinforcement Learning";
import { L5_DeepReinforcementLearning } from "./lectures/MIT 6.S191 Deep Learning 2026/L5_Deep Reinforcement Learning";
import { L1_IntroductionToNeuralNetworksAndDeepLearning as MIT6S191L1IntroductionQuestions } from "./lectures/MIT 6.S191 Deep Learning 2026/L1_Introduction";
import { MIT6S191_L2_DeepSequenceModelingQuestions } from "./lectures/MIT 6.S191 Deep Learning 2026/L2_RNNs, Transformers and Attention";
import { MIT6S191_L3_CNNsQuestions } from "./lectures/MIT 6.S191 Deep Learning 2026/L3_CNNs";
import { MIT6S191_L4_DeepGenerativeModelingQuestions } from "./lectures/MIT 6.S191 Deep Learning 2026/L4_Deep Generative Modeling";
import { MIT6S191_L6_LMsAndFrontiersQuestions } from "./lectures/MIT 6.S191 Deep Learning 2026/L6_LMs and frontiers";
import { L1_IntroductionToNeuralNetworksAndDeepLearning } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L1_ Introduction to Neural Networks and Deep Learning";
import { L2_TrainingDeepNNs } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L2_Training Deep NNs";
import { EmbeddingsQuestions } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L6_Embeddings";
import { TransformersQuestions as MIT15773L7TransformersQuestions } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L7_Transformers";
import { TransformersSelfSupervisedLearningQuestions as MIT15773L8Transformers2Questions } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L8_Transformers 2";
import { LLMsQuestions as MIT15773L9LLMsQuestions } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L9_LLMs";
import { L10LLMs2Questions as MIT15773L10LLMs2Questions } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L10_LLMs 2";
import { MIT15773L11DiffusionQuestions } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L11_Diffusion";
import { CrashCourseLinearAlgebraL0Questions } from "./other/Crash Course Linear Algebra/Lecture 0 - Linear Algebra Foundations";
import { CrashCourseLinearAlgebraL1Questions } from "./other/Crash Course Linear Algebra/Lecture 1 — Vectors, Geometry, and Dot Products";
import { CrashCourseLinearAlgebraL2Questions } from "./other/Crash Course Linear Algebra/Lecture 2 - Matrices as Transformations";
import { CrashCourseLinearAlgebraL3Questions } from "./other/Crash Course Linear Algebra/Lecture 3 - Derivatives and Gradients";
import { CrashCourseLinearAlgebraL4Questions } from "./other/Crash Course Linear Algebra/Lecture 4 - Eigenvectors, SVD, and Representation Learning";
import { CrashCourseLinearAlgebraL5Questions } from "./other/Crash Course Linear Algebra/Lecture 5 - Linear Algebra in LLMs and RL";
import { CrashCourseProbabilityL0Questions } from "./other/Crash Courses/Probability/Lecture 0 - Probability Prerequisites";
import { CrashCourseProbabilityL1Questions } from "./other/Crash Courses/Probability/Lecture 1 - overview";
import { CrashCourseProbabilityL2Questions } from "./other/Crash Courses/Probability/Lecture 2 - Conditional Probability, Bayes, and Dependence";
import { CrashCourseProbabilityL3Questions } from "./other/Crash Courses/Probability/Lecture 3 - Likelihood, Loss, Softmax, and Deep Learning";
import { CrashCourseProbabilityL4Questions } from "./other/Crash Courses/Probability/Lecture 4 - Probability Over Time - Reinforcement Learning";
import { CrashCourseProbabilityL5Questions } from "./other/Crash Courses/Probability/Lecture 5 - Sampling, Latent Variables, and Diffusion Models";
import { MathLogsExpRootsQuestions } from "./other/Math (various)/logs, exp and root";
import { BiologyChemistryLifeScienceL0Questions } from "./other/Crash Courses/Biology & Chemistry for Life Science/Lecture 0 - preparation";
import { BiologyChemistryLifeScienceL1Questions } from "./other/Crash Courses/Biology & Chemistry for Life Science/Lecture 1 - Chemistry of Life";
import { BiologyChemistryLifeScienceL2Questions } from "./other/Crash Courses/Biology & Chemistry for Life Science/Lecture 2 - Cells as Information-Processing Systems";
import { BiologyChemistryLifeScienceL3Questions } from "./other/Crash Courses/Biology & Chemistry for Life Science/Lecture 3 - Genetics, Proteins, and Biological Regulation";
import { BiologyChemistryLifeScienceL4Questions } from "./other/Crash Courses/Biology & Chemistry for Life Science/Lecture 4 - Physiology, Disease, and Pharmacology";
import { BiologyChemistryLifeScienceL5Questions } from "./other/Crash Courses/Biology & Chemistry for Life Science/Lecture 5 - Biomedical Systems, Biotechnology, and Evidence";
import { ClinicalTrialsLecture1Questions } from "./other/Crash Courses/Clinical Trials/Lecture 1 - Why Clinical Trials Exist";
import { ClinicalTrialsLecture2Questions } from "./other/Crash Courses/Clinical Trials/Lecture 2 - Designing Clinical Trials";
import { ClinicalTrialsLecture3Questions } from "./other/Crash Courses/Clinical Trials/Lecture 3 - Statistics and Evidence Interpretation";
import { ClinicalTrialsLecture4Questions } from "./other/Crash Courses/Clinical Trials/Lecture 4 - Running Clinical Trials";
import { ClinicalTrialsLecture5Questions } from "./other/Crash Courses/Clinical Trials/Lecture 5 - Modern Clinical Research and The Future";
import { MIT15773L3DeepLearningForComputerVisionQuestions } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L3_Deep Learning for Computer Vision";
import { MIT15773L4ComputerVisionTransferLearningQuestions } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L4_Computer Vision –Transfer Learning and Fine-Tuning";
import { L5NLPBasicsQuestions } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L5_NLP Basics";
import { CodeAsAgentHarnessQuestions } from "./other/AI agents/Code as Agent Harness";
import { DeepAgentsQuestions } from "./other/Langchain/Deepagents";
import { mixedQuestions } from "./other/other";

export type Difficulty = "easy" | "medium" | "hard";
export const QUESTION_TYPES = ["multiple-select", "assertion-reason"] as const;
export type QuestionType = (typeof QUESTION_TYPES)[number];
export const DEFAULT_QUESTION_TYPES: readonly QuestionType[] = QUESTION_TYPES;
export const QUESTION_TYPE_LABELS: Record<QuestionType, string> = {
  "multiple-select": "Multiple-select",
  "assertion-reason": "Assertion-reason",
};
export { ALL_TOPICS } from "./questionTopics";
export type { Topic } from "./questionTopics";
export type SourceSeriesId =
  | "aie-foundations"
  | "aie-building-apps"
  | "stanford-cme295"
  | "stanford-cs224r"
  | "other-rl"
  | "mit-6s191-2026"
  | "mit-15773-2024"
  | "crash-course-linear-algebra"
  | "crash-course-probability"
  | "math-various"
  | "biology-chemistry-life-science"
  | "clinical-trials"
  | "ai-agents"
  | "langchain"
  | "other";

export type Question = {
  id: string;
  chapter: number;
  type?: QuestionType;
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

export function getQuestionType(
  question: Pick<Question, "type">,
): QuestionType {
  return question.type ?? "multiple-select";
}

export type QuestionSourceMetadata = {
  sourceId: SourceId;
  sourceLabel: string;
  sourceTitle: string;
  sourceContext: string;
  seriesId: SourceSeriesId;
  seriesLabel: string;
  topic: Topic;
};

export type QuestionSource = {
  id: string;
  label: string;
  title: string;
  seriesId: SourceSeriesId;
  seriesLabel: string;
  topic: Topic;
  questions: Question[];
  balance?: boolean;
};

// ------------------------------
// Central question set registry
// ------------------------------

export const QUESTION_SOURCES = [
  {
    id: "chapter-1" as const,
    label: "Chapter 1 only",
    title: "Chapter 1 Quiz – Analyzing Text Data with Deep Learning",
    seriesId: "aie-foundations" as const,
    seriesLabel: "AIE Foundations Book",
    topic: "NLP" as const,
    questions: chapter1Questions,
  },
  {
    id: "chapter-2" as const,
    label: "Chapter 2 only",
    title: "Chapter 2 Quiz – The Transformer and Modern NLP",
    seriesId: "aie-foundations" as const,
    seriesLabel: "AIE Foundations Book",
    topic: "NLP" as const,
    questions: chapter2Questions,
  },
  {
    id: "chapter-3" as const,
    label: "Chapter 3 only",
    title: "Chapter 3 Quiz – Large Language Models & Prompting",
    seriesId: "aie-foundations" as const,
    seriesLabel: "AIE Foundations Book",
    topic: "NLP" as const,
    questions: chapter3Questions,
  },
  {
    id: "aie-build-app-ch2" as const,
    label: "AIE build app - Chap 2",
    title: "AIE build app – Chapter 2: Understanding Foundation Models",
    seriesId: "aie-building-apps" as const,
    seriesLabel: "AIE Building Apps Book",
    topic: "NLP" as const,
    questions: aieChapter2Questions,
  },
  {
    id: "aie-build-app-ch3" as const,
    label: "AIE build app - Chap 3",
    title:
      "AIE build app – Chapter 3: Evaluating Foundation Models & Applications",
    seriesId: "aie-building-apps" as const,
    seriesLabel: "AIE Building Apps Book",
    topic: "NLP" as const,
    questions: aieChapter3Questions,
  },
  {
    id: "aie-build-app-ch4" as const,
    label: "AIE build app - Chap 4",
    title: "AIE build app – Chapter 4: Building Agents with Foundation Models",
    seriesId: "aie-building-apps" as const,
    seriesLabel: "AIE Building Apps Book",
    topic: "NLP" as const,
    questions: aieChapter4Questions,
  },
  {
    id: "cme295-lect1" as const,
    label: "Stanford CME295 Lecture 1",
    title: "Stanford CME295 Lecture 1: Transformers & LLMs",
    seriesId: "stanford-cme295" as const,
    seriesLabel: "Stanford CME295 Transformers & LLMs",
    topic: "NLP" as const,
    questions: stanfordCME295Lecture1Questions,
  },
  {
    id: "cme295-lect2" as const,
    label: "Stanford CME295 Lecture 2",
    title: "Stanford CME295 Lecture 2: Transformer-Based Models & Tricks",
    seriesId: "stanford-cme295" as const,
    seriesLabel: "Stanford CME295 Transformers & LLMs",
    topic: "NLP" as const,
    questions: stanfordCME295Lecture2Questions,
  },
  {
    id: "cme295-lect3" as const,
    label: "Stanford CME295 Lecture 3",
    title: "Stanford CME295 Lecture 3: Large Language Models, MoE & Inference",
    seriesId: "stanford-cme295" as const,
    seriesLabel: "Stanford CME295 Transformers & LLMs",
    topic: "NLP" as const,
    questions: stanfordCME295Lecture3LLMsQuestions,
  },
  {
    id: "cme295-lect4" as const,
    label: "Stanford CME295 Lecture 4",
    title: "Stanford CME295 Lecture 4: LLM Training, Scaling & Alignment",
    seriesId: "stanford-cme295" as const,
    seriesLabel: "Stanford CME295 Transformers & LLMs",
    topic: "NLP" as const,
    questions: stanfordCME295Lecture4TrainingQuestions,
  },
  {
    id: "cme295-lect5" as const,
    label: "Stanford CME295 Lecture 5",
    title: "Stanford CME295 Lecture 5: LLM Preference Tuning, RLHF & DPO",
    seriesId: "stanford-cme295" as const,
    seriesLabel: "Stanford CME295 Transformers & LLMs",
    topic: "NLP" as const,
    questions: stanfordCME295Lecture5PreferenceTuningQuestions,
  },
  {
    id: "cme295-lect6" as const,
    label: "Stanford CME295 Lecture 6",
    title: "Stanford CME295 Lecture 6: LLM Reasoning & Test-Time Scaling",
    seriesId: "stanford-cme295" as const,
    seriesLabel: "Stanford CME295 Transformers & LLMs",
    topic: "NLP" as const,
    questions: stanfordCME295Lecture6ReasoningQuestions,
  },
  {
    id: "cme295-lect7" as const,
    label: "Stanford CME295 Lecture 7",
    title: "Stanford CME295 Lecture 7: RAG, Tool Calling & Agents",
    seriesId: "stanford-cme295" as const,
    seriesLabel: "Stanford CME295 Transformers & LLMs",
    topic: "NLP" as const,
    questions: stanfordCME295Lecture7RagToolsAgentsQuestions,
  },
  {
    id: "cme295-lect8" as const,
    label: "Stanford CME295 Lecture 8",
    title: "Stanford CME295 Lecture 8: LLM Evaluation",
    seriesId: "stanford-cme295" as const,
    seriesLabel: "Stanford CME295 Transformers & LLMs",
    topic: "NLP" as const,
    questions: stanfordCME295Lecture8EvaluationQuestions,
  },
  {
    id: "cme295-lect9" as const,
    label: "Stanford CME295 Lecture 9",
    title: "Stanford CME295 Lecture 9: Course Synthesis & Frontiers",
    seriesId: "stanford-cme295" as const,
    seriesLabel: "Stanford CME295 Transformers & LLMs",
    topic: "NLP" as const,
    questions: stanfordCME295Lecture9SynthesisFrontiersQuestions,
  },
  {
    id: "cs224r-lect1" as const,
    label: "Stanford CS224R Lecture 1",
    title: "Stanford CS224R Lecture 1: Intro to Deep Reinforcement Learning",
    seriesId: "stanford-cs224r" as const,
    seriesLabel: "Stanford CS224R Deep Reinforcement Learning",
    topic: "RL" as const,
    questions: cs224rLecture1IntroQuestions,
  },
  {
    id: "cs224r-lect2" as const,
    label: "Stanford CS224R Lecture 2",
    title: "Stanford CS224R Lecture 2: Imitation Learning",
    seriesId: "stanford-cs224r" as const,
    seriesLabel: "Stanford CS224R Deep Reinforcement Learning",
    topic: "RL" as const,
    questions: cs224rLecture2ImitationLearningQuestions,
  },
  {
    id: "cs224r-lect3" as const,
    label: "Stanford CS224R Lecture 3",
    title: "Stanford CS224R Lecture 3: Policy Gradients",
    seriesId: "stanford-cs224r" as const,
    seriesLabel: "Stanford CS224R Deep Reinforcement Learning",
    topic: "RL" as const,
    questions: cs224rLecture3PolicyGradientsQuestions,
  },
  {
    id: "cs224r-lect4" as const,
    label: "Stanford CS224R Lecture 4",
    title: "Stanford CS224R Lecture 4: Actor-Critic Methods",
    seriesId: "stanford-cs224r" as const,
    seriesLabel: "Stanford CS224R Deep Reinforcement Learning",
    topic: "RL" as const,
    questions: cs224rLecture4ActorCriticQuestions,
  },
  {
    id: "cs224r-lect5" as const,
    label: "Stanford CS224R Lecture 5",
    title: "Stanford CS224R Lecture 5: Off-Policy Actor-Critic Methods",
    seriesId: "stanford-cs224r" as const,
    seriesLabel: "Stanford CS224R Deep Reinforcement Learning",
    topic: "RL" as const,
    questions: lecture5_OffPolicyActorCriticQuestions,
  },
  {
    id: "cs224r-lect6" as const,
    label: "Stanford CS224R Lecture 6",
    title: "Stanford CS224R Lecture 6: Q-Learning",
    seriesId: "stanford-cs224r" as const,
    seriesLabel: "Stanford CS224R Deep Reinforcement Learning",
    topic: "RL" as const,
    questions: cs224rLecture6QLearningQuestions,
  },
  {
    id: "other-rl-intro" as const,
    label: "Other RL Intro",
    title: "Other RL: Introduction to Reinforcement Learning",
    seriesId: "other-rl" as const,
    seriesLabel: "Other RL Lectures",
    topic: "RL" as const,
    questions: OtherRL_introductiontoReinforcementLearning,
  },
  {
    id: "mit6s191-l1" as const,
    label: "MIT 6.S191 L1",
    title: "MIT 6.S191 L1: Introduction to Deep Learning",
    seriesId: "mit-6s191-2026" as const,
    seriesLabel: "MIT 6.S191 Deep Learning 2026",
    topic: "DL" as const,
    questions: MIT6S191L1IntroductionQuestions,
  },
  {
    id: "mit6s191-l2" as const,
    label: "MIT 6.S191 L2",
    title: "MIT 6.S191 L2: RNNs, Transformers and Attention",
    seriesId: "mit-6s191-2026" as const,
    seriesLabel: "MIT 6.S191 Deep Learning 2026",
    topic: "NLP" as const,
    questions: MIT6S191_L2_DeepSequenceModelingQuestions,
  },
  {
    id: "mit6s191-l5" as const,
    label: "MIT 6.S191 L5",
    title: "MIT 6.S191 L5: Deep Reinforcement Learning",
    seriesId: "mit-6s191-2026" as const,
    seriesLabel: "MIT 6.S191 Deep Learning 2026",
    topic: "RL" as const,
    questions: L5_DeepReinforcementLearning,
  },
  {
    id: "mit6s191-l3" as const,
    label: "MIT 6.S191 L3",
    title: "MIT 6.S191 L3: Convolutional Neural Networks",
    seriesId: "mit-6s191-2026" as const,
    seriesLabel: "MIT 6.S191 Deep Learning 2026",
    topic: "DL" as const,
    questions: MIT6S191_L3_CNNsQuestions,
  },
  {
    id: "mit6s191-l4" as const,
    label: "MIT 6.S191 L4",
    title: "MIT 6.S191 L4: Deep Generative Modeling",
    seriesId: "mit-6s191-2026" as const,
    seriesLabel: "MIT 6.S191 Deep Learning 2026",
    topic: "DL" as const,
    questions: MIT6S191_L4_DeepGenerativeModelingQuestions,
  },
  {
    id: "mit6s191-l6" as const,
    label: "MIT 6.S191 L6",
    title: "MIT 6.S191 L6: LMs and Frontiers",
    seriesId: "mit-6s191-2026" as const,
    seriesLabel: "MIT 6.S191 Deep Learning 2026",
    topic: "NLP" as const,
    questions: MIT6S191_L6_LMsAndFrontiersQuestions,
  },
  {
    id: "mit15773-l1" as const,
    label: "MIT 15.773 L1",
    title: "MIT 15.773 L1: Introduction to Neural Networks and Deep Learning",
    seriesId: "mit-15773-2024" as const,
    seriesLabel: "MIT 15.773 Hands-On Deep Learning 2024",
    topic: "DL" as const,
    questions: L1_IntroductionToNeuralNetworksAndDeepLearning,
  },
  {
    id: "mit15773-l2" as const,
    label: "MIT 15.773 L2",
    title: "MIT 15.773 L2: Training Deep NNs",
    seriesId: "mit-15773-2024" as const,
    seriesLabel: "MIT 15.773 Hands-On Deep Learning 2024",
    topic: "DL" as const,
    questions: L2_TrainingDeepNNs,
  },
  {
    id: "mit15773-l3" as const,
    label: "MIT 15.773 L3",
    title: "MIT 15.773 L3: Deep Learning for Computer Vision",
    seriesId: "mit-15773-2024" as const,
    seriesLabel: "MIT 15.773 Hands-On Deep Learning 2024",
    topic: "DL" as const,
    questions: MIT15773L3DeepLearningForComputerVisionQuestions,
  },
  {
    id: "mit15773-l4" as const,
    label: "MIT 15.773 L4",
    title: "MIT 15.773 L4: Computer Vision - Transfer Learning and Fine-Tuning",
    seriesId: "mit-15773-2024" as const,
    seriesLabel: "MIT 15.773 Hands-On Deep Learning 2024",
    topic: "DL" as const,
    questions: MIT15773L4ComputerVisionTransferLearningQuestions,
  },
  {
    id: "mit15773-l5" as const,
    label: "MIT 15.773 L5",
    title: "MIT 15.773 L5: NLP Basics",
    seriesId: "mit-15773-2024" as const,
    seriesLabel: "MIT 15.773 Hands-On Deep Learning 2024",
    topic: "NLP" as const,
    questions: L5NLPBasicsQuestions,
  },
  {
    id: "mit15773-l6" as const,
    label: "MIT 15.773 L6",
    title: "MIT 15.773 L6: Embeddings",
    seriesId: "mit-15773-2024" as const,
    seriesLabel: "MIT 15.773 Hands-On Deep Learning 2024",
    topic: "NLP" as const,
    questions: EmbeddingsQuestions,
  },
  {
    id: "mit15773-l7" as const,
    label: "MIT 15.773 L7",
    title: "MIT 15.773 L7: Transformers",
    seriesId: "mit-15773-2024" as const,
    seriesLabel: "MIT 15.773 Hands-On Deep Learning 2024",
    topic: "NLP" as const,
    questions: MIT15773L7TransformersQuestions,
  },
  {
    id: "mit15773-l8" as const,
    label: "MIT 15.773 L8",
    title: "MIT 15.773 L8: Transformers 2",
    seriesId: "mit-15773-2024" as const,
    seriesLabel: "MIT 15.773 Hands-On Deep Learning 2024",
    topic: "NLP" as const,
    questions: MIT15773L8Transformers2Questions,
  },
  {
    id: "mit15773-l9" as const,
    label: "MIT 15.773 L9",
    title: "MIT 15.773 L9: LLMs",
    seriesId: "mit-15773-2024" as const,
    seriesLabel: "MIT 15.773 Hands-On Deep Learning 2024",
    topic: "NLP" as const,
    questions: MIT15773L9LLMsQuestions,
  },
  {
    id: "mit15773-l10" as const,
    label: "MIT 15.773 L10",
    title: "MIT 15.773 L10: LLMs 2",
    seriesId: "mit-15773-2024" as const,
    seriesLabel: "MIT 15.773 Hands-On Deep Learning 2024",
    topic: "NLP" as const,
    questions: MIT15773L10LLMs2Questions,
  },
  {
    id: "mit15773-l11" as const,
    label: "MIT 15.773 L11",
    title: "MIT 15.773 L11: Diffusion",
    seriesId: "mit-15773-2024" as const,
    seriesLabel: "MIT 15.773 Hands-On Deep Learning 2024",
    topic: "DL" as const,
    questions: MIT15773L11DiffusionQuestions,
  },
  {
    id: "crash-linalg-l0" as const,
    label: "Crash Course Linear Algebra L0",
    title: "Crash Course Linear Algebra L0: Linear Algebra Foundations",
    seriesId: "crash-course-linear-algebra" as const,
    seriesLabel: "Crash Course Linear Algebra",
    topic: "Math" as const,
    questions: CrashCourseLinearAlgebraL0Questions,
  },
  {
    id: "crash-linalg-l1" as const,
    label: "Crash Course Linear Algebra L1",
    title:
      "Crash Course Linear Algebra L1: Vectors, Geometry, and Dot Products",
    seriesId: "crash-course-linear-algebra" as const,
    seriesLabel: "Crash Course Linear Algebra",
    topic: "Math" as const,
    questions: CrashCourseLinearAlgebraL1Questions,
  },
  {
    id: "crash-linalg-l2" as const,
    label: "Crash Course Linear Algebra L2",
    title: "Crash Course Linear Algebra L2: Matrices as Transformations",
    seriesId: "crash-course-linear-algebra" as const,
    seriesLabel: "Crash Course Linear Algebra",
    topic: "Math" as const,
    questions: CrashCourseLinearAlgebraL2Questions,
  },
  {
    id: "crash-linalg-l3" as const,
    label: "Crash Course Linear Algebra L3",
    title: "Crash Course Linear Algebra L3: Derivatives and Gradients",
    seriesId: "crash-course-linear-algebra" as const,
    seriesLabel: "Crash Course Linear Algebra",
    topic: "Math" as const,
    questions: CrashCourseLinearAlgebraL3Questions,
  },
  {
    id: "crash-linalg-l4" as const,
    label: "Crash Course Linear Algebra L4",
    title:
      "Crash Course Linear Algebra L4: Eigenvectors, SVD, and Representation Learning",
    seriesId: "crash-course-linear-algebra" as const,
    seriesLabel: "Crash Course Linear Algebra",
    topic: "Math" as const,
    questions: CrashCourseLinearAlgebraL4Questions,
  },
  {
    id: "crash-linalg-l5" as const,
    label: "Crash Course Linear Algebra L5",
    title: "Crash Course Linear Algebra L5: Linear Algebra in LLMs and RL",
    seriesId: "crash-course-linear-algebra" as const,
    seriesLabel: "Crash Course Linear Algebra",
    topic: "Math" as const,
    questions: CrashCourseLinearAlgebraL5Questions,
  },
  {
    id: "crash-probability-l0" as const,
    label: "Crash Course Probability L0",
    title: "Crash Course Probability L0: Probability Prerequisites",
    seriesId: "crash-course-probability" as const,
    seriesLabel: "Crash Course Probability",
    topic: "Math" as const,
    questions: CrashCourseProbabilityL0Questions,
    balance: false,
  },
  {
    id: "crash-probability-l1" as const,
    label: "Crash Course Probability L1",
    title: "Crash Course Probability L1: Probability as the Language of AI",
    seriesId: "crash-course-probability" as const,
    seriesLabel: "Crash Course Probability",
    topic: "Math" as const,
    questions: CrashCourseProbabilityL1Questions,
  },
  {
    id: "crash-probability-l2" as const,
    label: "Crash Course Probability L2",
    title:
      "Crash Course Probability L2: Conditional Probability, Bayes, and Dependence",
    seriesId: "crash-course-probability" as const,
    seriesLabel: "Crash Course Probability",
    topic: "Math" as const,
    questions: CrashCourseProbabilityL2Questions,
  },
  {
    id: "crash-probability-l3" as const,
    label: "Crash Course Probability L3",
    title:
      "Crash Course Probability L3: Likelihood, Loss, Softmax, and Deep Learning",
    seriesId: "crash-course-probability" as const,
    seriesLabel: "Crash Course Probability",
    topic: "Math" as const,
    questions: CrashCourseProbabilityL3Questions,
  },
  {
    id: "crash-probability-l4" as const,
    label: "Crash Course Probability L4",
    title:
      "Crash Course Probability L4: Probability Over Time: Reinforcement Learning",
    seriesId: "crash-course-probability" as const,
    seriesLabel: "Crash Course Probability",
    topic: "Math" as const,
    questions: CrashCourseProbabilityL4Questions,
  },
  {
    id: "crash-probability-l5" as const,
    label: "Crash Course Probability L5",
    title:
      "Crash Course Probability L5: Sampling, Latent Variables, and Diffusion Models",
    seriesId: "crash-course-probability" as const,
    seriesLabel: "Crash Course Probability",
    topic: "Math" as const,
    questions: CrashCourseProbabilityL5Questions,
  },
  {
    id: "math-logs-exp-roots" as const,
    label: "Math: Logs, Exp, Roots",
    title:
      "Math (various): Logarithms, Exponentials, Euler's Number, and Roots",
    seriesId: "math-various" as const,
    seriesLabel: "Math (various)",
    topic: "Math" as const,
    questions: MathLogsExpRootsQuestions,
  },
  {
    id: "bio-chem-life-l0" as const,
    label: "Biology & Chemistry L0",
    title: "Biology & Chemistry for Life Science L0: Preparation",
    seriesId: "biology-chemistry-life-science" as const,
    seriesLabel: "Biology & Chemistry for Life Science",
    topic: "Life Science" as const,
    questions: BiologyChemistryLifeScienceL0Questions,
    balance: false,
  },
  {
    id: "bio-chem-life-l1" as const,
    label: "Biology & Chemistry L1",
    title: "Biology & Chemistry for Life Science L1: Chemistry of Life",
    seriesId: "biology-chemistry-life-science" as const,
    seriesLabel: "Biology & Chemistry for Life Science",
    topic: "Life Science" as const,
    questions: BiologyChemistryLifeScienceL1Questions,
  },
  {
    id: "bio-chem-life-l2" as const,
    label: "Biology & Chemistry L2",
    title:
      "Biology & Chemistry for Life Science L2: Cells as Information-Processing Systems",
    seriesId: "biology-chemistry-life-science" as const,
    seriesLabel: "Biology & Chemistry for Life Science",
    topic: "Life Science" as const,
    questions: BiologyChemistryLifeScienceL2Questions,
  },
  {
    id: "bio-chem-life-l3" as const,
    label: "Biology & Chemistry L3",
    title:
      "Biology & Chemistry for Life Science L3: Genetics, Proteins, and Biological Regulation",
    seriesId: "biology-chemistry-life-science" as const,
    seriesLabel: "Biology & Chemistry for Life Science",
    topic: "Life Science" as const,
    questions: BiologyChemistryLifeScienceL3Questions,
  },
  {
    id: "bio-chem-life-l4" as const,
    label: "Biology & Chemistry L4",
    title:
      "Biology & Chemistry for Life Science L4: Physiology, Disease, and Pharmacology",
    seriesId: "biology-chemistry-life-science" as const,
    seriesLabel: "Biology & Chemistry for Life Science",
    topic: "Life Science" as const,
    questions: BiologyChemistryLifeScienceL4Questions,
  },
  {
    id: "bio-chem-life-l5" as const,
    label: "Biology & Chemistry L5",
    title:
      "Biology & Chemistry for Life Science L5: Biomedical Systems, Biotechnology, and Evidence",
    seriesId: "biology-chemistry-life-science" as const,
    seriesLabel: "Biology & Chemistry for Life Science",
    topic: "Life Science" as const,
    questions: BiologyChemistryLifeScienceL5Questions,
  },
  {
    id: "clinical-trials-l1" as const,
    label: "Clinical Trials L1",
    title: "Clinical Trials Crash Course L1: Why Clinical Trials Exist",
    seriesId: "clinical-trials" as const,
    seriesLabel: "Clinical Trials Crash Course",
    topic: "Life Science" as const,
    questions: ClinicalTrialsLecture1Questions,
  },
  {
    id: "clinical-trials-l2" as const,
    label: "Clinical Trials L2",
    title: "Clinical Trials Crash Course L2: Designing Clinical Trials",
    seriesId: "clinical-trials" as const,
    seriesLabel: "Clinical Trials Crash Course",
    topic: "Life Science" as const,
    questions: ClinicalTrialsLecture2Questions,
  },
  {
    id: "clinical-trials-l3" as const,
    label: "Clinical Trials L3",
    title:
      "Clinical Trials Crash Course L3: Statistics and Evidence Interpretation",
    seriesId: "clinical-trials" as const,
    seriesLabel: "Clinical Trials Crash Course",
    topic: "Life Science" as const,
    questions: ClinicalTrialsLecture3Questions,
  },
  {
    id: "clinical-trials-l4" as const,
    label: "Clinical Trials L4",
    title: "Clinical Trials Crash Course L4: Running Clinical Trials",
    seriesId: "clinical-trials" as const,
    seriesLabel: "Clinical Trials Crash Course",
    topic: "Life Science" as const,
    questions: ClinicalTrialsLecture4Questions,
  },
  {
    id: "clinical-trials-l5" as const,
    label: "Clinical Trials L5",
    title:
      "Clinical Trials Crash Course L5: Modern Clinical Research and The Future",
    seriesId: "clinical-trials" as const,
    seriesLabel: "Clinical Trials Crash Course",
    topic: "Life Science" as const,
    questions: ClinicalTrialsLecture5Questions,
  },
  {
    id: "ai-agents-code-harness" as const,
    label: "AI Agents Harness",
    title: "AI Agents: Code as Agent Harness",
    seriesId: "ai-agents" as const,
    seriesLabel: "AI Agents",
    topic: "NLP" as const,
    questions: CodeAsAgentHarnessQuestions,
  },
  {
    id: "langchain-deepagents" as const,
    label: "LangChain Deep Agents",
    title: "LangChain Deep Agents",
    seriesId: "langchain" as const,
    seriesLabel: "LangChain",
    topic: "NLP" as const,
    questions: DeepAgentsQuestions,
  },
  {
    id: "other" as const,
    label: "Other",
    title: "Other Questions",
    seriesId: "other" as const,
    seriesLabel: "Other Sources",
    topic: "DL" as const,
    questions: mixedQuestions,
  },
] satisfies QuestionSource[];

export type SourceId = (typeof QUESTION_SOURCES)[number]["id"];
export const QUESTION_SOURCE_CONTEXT: Record<SourceId, string> = {
  "chapter-1":
    "NLP foundations chapter about text data, tokens, embeddings, and basic neural language-model concepts.",
  "chapter-2":
    "Transformer chapter about attention, positional information, encoder-decoder structure, and modern NLP model behavior.",
  "chapter-3":
    "LLM and prompting chapter about scaling, instruction following, context use, and practical prompt design.",
  "aie-build-app-ch2":
    "Foundation-model chapter about model capabilities, embeddings, tokenization, prompting, and application fit.",
  "aie-build-app-ch3":
    "Evaluation chapter about measuring foundation-model applications, test sets, metrics, and failure analysis.",
  "aie-build-app-ch4":
    "Agent-building chapter about tool use, planning loops, retrieval, memory, and production agent behavior.",
  "cme295-lect1":
    "Transformer introduction covering attention, positional encoding, sequence modeling, and why transformers scale well.",
  "cme295-lect2":
    "Transformer-models lecture about architectural variants, training tricks, tokenization, and efficient inference considerations.",
  "cme295-lect3":
    "LLM lecture about scaling, mixture-of-experts, inference, decoding, and deployment tradeoffs.",
  "cme295-lect4":
    "LLM training lecture about pretraining, post-training, alignment, data mixtures, and scaling laws.",
  "cme295-lect5":
    "Preference-tuning lecture about preference data, reward modeling, RLHF/PPO, Best-of-N, DPO, and alignment tradeoffs.",
  "cme295-lect6":
    "Reasoning-model lecture about chain-of-thought, Pass@k, test-time scaling, verifiable rewards, GRPO, DeepSeek R1, and distillation.",
  "cme295-lect7":
    "Systems lecture about RAG, retrieval quality, tool calling, MCP, ReAct-style agents, A2A, and agent safety controls.",
  "cme295-lect8":
    "Evaluation lecture about human ratings, rule-based metrics, LLM-as-a-Judge, factuality, agent failure modes, benchmarks, reliability, and benchmark limits.",
  "cme295-lect9":
    "Course synthesis lecture about transformer and LLM fundamentals, multimodal transformers, diffusion LLMs, data, hardware, deployment tradeoffs, and open problems.",
  "cs224r-lect1":
    "Deep reinforcement learning introduction about MDPs, value functions, policies, and the RL problem setup.",
  "cs224r-lect2":
    "Reinforcement learning lecture about imitation learning, behavior cloning, distribution shift, and expressive policies.",
  "cs224r-lect3":
    "Policy-gradient lecture about stochastic policies, score-function estimators, variance reduction, and optimization.",
  "cs224r-lect4":
    "Actor-critic lecture about value baselines, advantage estimation, bootstrapping, and policy updates.",
  "cs224r-lect5":
    "Off-policy actor-critic lecture about replay data, importance sampling, Q-learning links, and stability challenges.",
  "cs224r-lect6":
    "Q-learning lecture about Bellman optimality, replay buffers, target networks, overestimation, Double DQN, and N-step returns.",
  "other-rl-intro":
    "Introductory reinforcement learning material about agents, rewards, environments, and value-based learning.",
  "mit6s191-l1":
    "Deep learning introduction about neural networks, representation learning, optimization, and core model families.",
  "mit6s191-l2":
    "Sequence-modeling lecture about RNNs, attention, transformers, and temporal data.",
  "mit6s191-l5":
    "Deep reinforcement learning lecture about RL objectives, Q-learning, policy gradients, and neural control.",
  "mit6s191-l3":
    "Computer vision lecture about convolutional neural networks, image features, pooling, and visual recognition.",
  "mit6s191-l4":
    "Generative modeling lecture about latent variables, VAEs, GANs, diffusion, and sampling.",
  "mit6s191-l6":
    "Language-model frontiers lecture about modern LMs, multimodal systems, agents, and emerging capabilities.",
  "mit15773-l1":
    "Hands-on deep learning lecture introducing neural networks, training loops, activations, and practical ML workflows.",
  "mit15773-l2":
    "Training deep networks lecture about optimization, regularization, initialization, normalization, and debugging training.",
  "mit15773-l3":
    "Computer vision lecture about CNNs, image classification, feature hierarchies, and visual model evaluation.",
  "mit15773-l4":
    "Transfer learning lecture about fine-tuning, pretrained vision models, limited data, and adaptation choices.",
  "mit15773-l5":
    "NLP basics lecture about text preprocessing, embeddings, language modeling, and sequence representations.",
  "mit15773-l6":
    "Embeddings lecture about vector representations, similarity, retrieval, and learned semantic spaces.",
  "mit15773-l7":
    "Transformers lecture about self-attention, positional information, encoder-decoder designs, and NLP applications.",
  "mit15773-l8":
    "Advanced transformers lecture about self-supervised learning, pretraining objectives, transfer, and representation reuse.",
  "mit15773-l9":
    "LLM lecture about transformer language models, prompting, scaling, and generation behavior.",
  "mit15773-l10":
    "Advanced LLM lecture about instruction tuning, alignment, retrieval/tool use, and deployment constraints.",
  "mit15773-l11":
    "Diffusion lecture about denoising objectives, score-based generation, sampling schedules, and image synthesis.",
  "crash-linalg-l0":
    "Linear algebra foundations practice covering coordinates, magnitude, unit vectors, dot products, projection, span, basis, matrices, determinants, systems, and rank.",
  "crash-linalg-l1":
    "Linear algebra lesson about vectors, geometry, dot products, projections, and how they support ML intuition.",
  "crash-linalg-l2":
    "Linear algebra lesson about matrices as transformations, composition, shape reasoning, rank, LoRA, transpose, symmetry, and attention.",
  "crash-linalg-l3":
    "Linear algebra lesson about derivatives, gradients, gradient descent, chain rule, backpropagation, matrix gradients, and optimization.",
  "crash-linalg-l4":
    "Linear algebra lesson about eigenvectors, covariance, PCA, SVD, low-rank structure, embeddings, LoRA, and representation learning.",
  "crash-linalg-l5":
    "Linear algebra synthesis lesson about attention, matrix stacks, RL value functions, optimization landscapes, and AI geometry.",
  "crash-probability-l0":
    "Pure applied math prereqs: ratios/odds, algebra, normalization, functions, sequences, sums/products, weights, exponents/logs, dimensions, Gaussians, complements, geometric sums.",
  "crash-probability-l1":
    "Probability lesson about AI uncertainty, sample spaces, events, random variables, discrete distributions, expectation, variance, calibration, and prediction.",
  "crash-probability-l2":
    "Probability lesson about conditional probability, joint and marginal probability, marginalization, independence, Bayes' theorem, base rates, and prediction as P(y | x).",
  "crash-probability-l3":
    "Probability lesson about logits, softmax, likelihood, log-likelihood, negative log-likelihood, cross-entropy, entropy, and probabilistic neural-network training.",
  "crash-probability-l4":
    "Probability lesson about reinforcement learning over time: states, actions, rewards, transitions, policies, expected return, values, and exploration.",
  "crash-probability-l5":
    "Probability lesson about sampling, greedy decoding, temperature, latent variables, Gaussian noise, diffusion models, denoising, and generative AI synthesis.",
  "math-logs-exp-roots":
    "Applied math practice on logarithm rules, exponential growth and decay, Euler's number, log scales, and roots.",
  "bio-chem-life-l0":
    "Life-science preparation practice covering prerequisite chemistry, cells, genes, disease, drugs, biomarkers, biotech, and evidence vocabulary.",
  "bio-chem-life-l1":
    "Life-science lesson about atoms, bonds, water, macromolecules, proteins, enzymes, ATP, metabolism, and structure-function reasoning.",
  "bio-chem-life-l2":
    "Life-science lesson about cells as organized information-processing systems, organelles, membranes, gradients, signaling, division, cancer, and immunity.",
  "bio-chem-life-l3":
    "Life-science lesson about DNA, genes, central dogma, gene expression, regulation, mutation, evolution, CRISPR, mRNA, and biotechnology.",
  "bio-chem-life-l4":
    "Life-science lesson about physiology, homeostasis, disease as disrupted regulation, pharmacology, dose response, PK/PD, biomarkers, and precision medicine.",
  "bio-chem-life-l5":
    "Life-science lesson about infection, immunity, vaccines, therapeutic modalities, diagnostics, translational evidence, and AI in biomedicine.",
  "clinical-trials-l1":
    "Clinical trials crash course lesson about why trials exist, bias, placebo effects, causal inference, evidence hierarchy, and the modern trial ecosystem.",
  "clinical-trials-l2":
    "Clinical trials crash course lesson about PICO(T), randomization, blinding, endpoints, development phases, and internal versus external validity.",
  "clinical-trials-l3":
    "Clinical trials crash course lesson about treatment effect measures, confidence intervals, p-values, clinical significance, survival analysis, and evidence synthesis.",
  "clinical-trials-l4":
    "Clinical trials crash course lesson about protocol execution, site selection, recruitment, data management, GCP, CRO operations, vendors, and operational failure modes.",
  "clinical-trials-l5":
    "Clinical trials crash course lesson about medical devices, diagnostics, real-world evidence, decentralized trials, adaptive designs, AI, and end-to-end development risk.",
  "ai-agents-code-harness":
    "AI agents material about code as executable, inspectable, stateful harness infrastructure for reasoning, action, memory, verification, and multi-agent coordination.",
  "langchain-deepagents":
    "LangChain Deep Agents material about planning, tools, subagents, memory, and agent orchestration.",
  other:
    "Mixed AI question set covering general deep learning and machine-learning concepts from miscellaneous sources.",
};
export type Mode = SourceId | "all";
export const SOURCE_SERIES: {
  id: SourceSeriesId;
  label: string;
  sourceIds: SourceId[];
}[] = Array.from(
  new Map(
    QUESTION_SOURCES.map((source) => [
      source.seriesId,
      { id: source.seriesId, label: source.seriesLabel },
    ]),
  ).values(),
).map((series) => ({
  ...series,
  sourceIds: QUESTION_SOURCES.filter((s) => s.seriesId === series.id).map(
    (s) => s.id,
  ),
}));
export const ALL_SOURCE_SERIES_IDS: SourceSeriesId[] = SOURCE_SERIES.map(
  (series) => series.id,
);

export function getSourceIdsForSeries(seriesId: SourceSeriesId): SourceId[] {
  return (
    SOURCE_SERIES.find((series) => series.id === seriesId)?.sourceIds ?? []
  );
}

export function getSeriesIdsForSources(
  sourceIds: SourceId[],
): SourceSeriesId[] {
  const activeSources = new Set(sourceIds);

  return SOURCE_SERIES.filter((series) =>
    series.sourceIds.some((sourceId) => activeSources.has(sourceId)),
  ).map((series) => series.id);
}

// All questions across all sources
export const allQuestions: Question[] = QUESTION_SOURCES.flatMap(
  (s) => s.questions,
);

export const ALL_SOURCE_IDS: SourceId[] = QUESTION_SOURCES.map((s) => s.id);

const QUESTION_SOURCE_METADATA_BY_ID = new Map<string, QuestionSourceMetadata>(
  QUESTION_SOURCES.flatMap((source) =>
    source.questions.map((question) => [
      question.id,
      {
        sourceId: source.id,
        sourceLabel: source.label,
        sourceTitle: source.title,
        sourceContext: QUESTION_SOURCE_CONTEXT[source.id],
        seriesId: source.seriesId,
        seriesLabel: source.seriesLabel,
        topic: source.topic,
      },
    ]),
  ),
);

// Helper: get questions for a given mode
export function getQuestionsForMode(mode: Mode): Question[] {
  if (mode === "all") return allQuestions;
  const src = QUESTION_SOURCES.find((s) => s.id === mode);
  return src ? src.questions : allQuestions;
}

// Helper: get questions for a set of sources (multi-select)
export function getQuestionsForSources(sourceIds: SourceId[]): Question[] {
  return getQuestionsForFilters(sourceIds, []);
}

// Helper: get questions for selected sources, topics, and question types
export function getQuestionsForFilters(
  sourceIds: SourceId[],
  topics: Topic[],
  questionTypes: readonly QuestionType[] = DEFAULT_QUESTION_TYPES,
): Question[] {
  if (sourceIds.length === 0 && topics.length === 0) return [];

  const activeSources = new Set(sourceIds);
  const activeTopics = new Set(topics);
  const activeQuestionTypes = new Set(questionTypes);
  if (activeQuestionTypes.size === 0) return [];

  return QUESTION_SOURCES.filter(
    (s) => activeSources.has(s.id) || activeTopics.has(s.topic),
  ).flatMap((s) =>
    s.questions.filter((question) =>
      activeQuestionTypes.has(getQuestionType(question)),
    ),
  );
}

export function getQuestionSourceMetadata(
  questionId: string,
): QuestionSourceMetadata | null {
  return QUESTION_SOURCE_METADATA_BY_ID.get(questionId) ?? null;
}

export function getQuestionSourceContext(questionId: string): string | null {
  return getQuestionSourceMetadata(questionId)?.sourceContext ?? null;
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
export function getTitleForSelection(
  sourceIds: SourceId[],
  topics: Topic[] = [],
): string {
  if (sourceIds.length === 0 && topics.length === 0) {
    return "Learning AI";
  }

  if (sourceIds.length === 0 && topics.length > 0) {
    return `Topic Quiz - ${topics.join(", ")}`;
  }

  const allSelected =
    sourceIds.length === ALL_SOURCE_IDS.length &&
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
export { stanfordCME295Lecture5PreferenceTuningQuestions } from "./lectures/Stanford CME295 Transformers & LLMs/lecture5_preference_tuning";
export { stanfordCME295Lecture6ReasoningQuestions } from "./lectures/Stanford CME295 Transformers & LLMs/lecture6_reasoning";
export { stanfordCME295Lecture7RagToolsAgentsQuestions } from "./lectures/Stanford CME295 Transformers & LLMs/lecture7_rag_tools_agents";
export { stanfordCME295Lecture8EvaluationQuestions } from "./lectures/Stanford CME295 Transformers & LLMs/lecture8_evaluation";
export { stanfordCME295Lecture9SynthesisFrontiersQuestions } from "./lectures/Stanford CME295 Transformers & LLMs/lecture9_synthesis_frontiers";
export { cs224rLecture1IntroQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture1_intro";
export { cs224rLecture2ImitationLearningQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture2_Imitation Learning";
export { cs224rLecture3PolicyGradientsQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture3_Policy Gradients";
export { cs224rLecture4ActorCriticQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture4_Actor-Critic Methods";
export { lecture5_OffPolicyActorCriticQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture5_Off-Policy Actor Critic Methods";
export { cs224rLecture6QLearningQuestions } from "./lectures/Stanford CS224R Deep Reinforcement Learning/lecture6_Q-Learning";
export { OtherRL_introductiontoReinforcementLearning } from "./lectures/Other RL/introduction to Reinforcement Learning";
export { L5_DeepReinforcementLearning } from "./lectures/MIT 6.S191 Deep Learning 2026/L5_Deep Reinforcement Learning";
export { L1_IntroductionToNeuralNetworksAndDeepLearning as MIT6S191L1IntroductionQuestions } from "./lectures/MIT 6.S191 Deep Learning 2026/L1_Introduction";
export { MIT6S191_L2_DeepSequenceModelingQuestions } from "./lectures/MIT 6.S191 Deep Learning 2026/L2_RNNs, Transformers and Attention";
export { MIT6S191_L3_CNNsQuestions } from "./lectures/MIT 6.S191 Deep Learning 2026/L3_CNNs";
export { MIT6S191_L4_DeepGenerativeModelingQuestions } from "./lectures/MIT 6.S191 Deep Learning 2026/L4_Deep Generative Modeling";
export { MIT6S191_L6_LMsAndFrontiersQuestions } from "./lectures/MIT 6.S191 Deep Learning 2026/L6_LMs and frontiers";
export { L1_IntroductionToNeuralNetworksAndDeepLearning } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L1_ Introduction to Neural Networks and Deep Learning";
export { L2_TrainingDeepNNs } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L2_Training Deep NNs";
export { L5NLPBasicsQuestions } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L5_NLP Basics";
export { EmbeddingsQuestions } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L6_Embeddings";
export { TransformersQuestions } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L7_Transformers";
export { TransformersSelfSupervisedLearningQuestions } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L8_Transformers 2";
export { LLMsQuestions } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L9_LLMs";
export { L10LLMs2Questions } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L10_LLMs 2";
export { MIT15773L11DiffusionQuestions } from "./lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L11_Diffusion";
export { CrashCourseLinearAlgebraL0Questions } from "./other/Crash Course Linear Algebra/Lecture 0 - Linear Algebra Foundations";
export { CrashCourseLinearAlgebraL1Questions } from "./other/Crash Course Linear Algebra/Lecture 1 — Vectors, Geometry, and Dot Products";
export { CrashCourseLinearAlgebraL2Questions } from "./other/Crash Course Linear Algebra/Lecture 2 - Matrices as Transformations";
export { CrashCourseLinearAlgebraL3Questions } from "./other/Crash Course Linear Algebra/Lecture 3 - Derivatives and Gradients";
export { CrashCourseLinearAlgebraL4Questions } from "./other/Crash Course Linear Algebra/Lecture 4 - Eigenvectors, SVD, and Representation Learning";
export { CrashCourseLinearAlgebraL5Questions } from "./other/Crash Course Linear Algebra/Lecture 5 - Linear Algebra in LLMs and RL";
export { CrashCourseProbabilityL0Questions } from "./other/Crash Courses/Probability/Lecture 0 - Probability Prerequisites";
export { CrashCourseProbabilityL1Questions } from "./other/Crash Courses/Probability/Lecture 1 - overview";
export { CrashCourseProbabilityL2Questions } from "./other/Crash Courses/Probability/Lecture 2 - Conditional Probability, Bayes, and Dependence";
export { CrashCourseProbabilityL3Questions } from "./other/Crash Courses/Probability/Lecture 3 - Likelihood, Loss, Softmax, and Deep Learning";
export { CrashCourseProbabilityL4Questions } from "./other/Crash Courses/Probability/Lecture 4 - Probability Over Time - Reinforcement Learning";
export { CrashCourseProbabilityL5Questions } from "./other/Crash Courses/Probability/Lecture 5 - Sampling, Latent Variables, and Diffusion Models";
export { MathLogsExpRootsQuestions } from "./other/Math (various)/logs, exp and root";
export { BiologyChemistryLifeScienceL0Questions } from "./other/Crash Courses/Biology & Chemistry for Life Science/Lecture 0 - preparation";
export { BiologyChemistryLifeScienceL1Questions } from "./other/Crash Courses/Biology & Chemistry for Life Science/Lecture 1 - Chemistry of Life";
export { BiologyChemistryLifeScienceL2Questions } from "./other/Crash Courses/Biology & Chemistry for Life Science/Lecture 2 - Cells as Information-Processing Systems";
export { BiologyChemistryLifeScienceL3Questions } from "./other/Crash Courses/Biology & Chemistry for Life Science/Lecture 3 - Genetics, Proteins, and Biological Regulation";
export { BiologyChemistryLifeScienceL4Questions } from "./other/Crash Courses/Biology & Chemistry for Life Science/Lecture 4 - Physiology, Disease, and Pharmacology";
export { BiologyChemistryLifeScienceL5Questions } from "./other/Crash Courses/Biology & Chemistry for Life Science/Lecture 5 - Biomedical Systems, Biotechnology, and Evidence";
export { ClinicalTrialsLecture1Questions } from "./other/Crash Courses/Clinical Trials/Lecture 1 - Why Clinical Trials Exist";
export { ClinicalTrialsLecture2Questions } from "./other/Crash Courses/Clinical Trials/Lecture 2 - Designing Clinical Trials";
export { ClinicalTrialsLecture3Questions } from "./other/Crash Courses/Clinical Trials/Lecture 3 - Statistics and Evidence Interpretation";
export { ClinicalTrialsLecture4Questions } from "./other/Crash Courses/Clinical Trials/Lecture 4 - Running Clinical Trials";
export { ClinicalTrialsLecture5Questions } from "./other/Crash Courses/Clinical Trials/Lecture 5 - Modern Clinical Research and The Future";
export { CodeAsAgentHarnessQuestions } from "./other/AI agents/Code as Agent Harness";
export { DeepAgentsQuestions } from "./other/Langchain/Deepagents";
export { mixedQuestions } from "./other/other";
