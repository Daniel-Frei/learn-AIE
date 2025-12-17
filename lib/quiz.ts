// lib/quiz.ts
import { chapter1Questions } from "./chapter1";
import { chapter2Questions } from "./chapter2";
import { chapter3Questions } from "./chapter3";
import { aieChapter2Questions } from "./books/AIE_building_apps/chapter2";
import { aieChapter3Questions } from "./books/AIE_building_apps/chapter3";
import { aieChapter4Questions } from "./books/AIE_building_apps/chapter4";
import { stanfordCME295Lecture1Questions } from "./lectures/Stanford CME295 Transformers & LLMs/lecture1_transformer";
import { stanfordCME295Lecture2Questions } from "./lectures/Stanford CME295 Transformers & LLMs/lecture2_models";
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
    title:
      "AIE build app – Chapter 2: Understanding Foundation Models",
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
    title:
      "AIE build app – Chapter 4: Building Agents with Foundation Models",
    questions: aieChapter4Questions,
  },
  {
    id: "cme295-lect1" as const,
    label: "Stanford CME295 Lecture 1",
    title:
      "Stanford CME295 Lecture 1: Transformers & LLMs",
    questions: stanfordCME295Lecture1Questions,
  },
  {
    id: "cme295-lect2" as const,
    label: "Stanford CME295 Lecture 2",
    title:
      "Stanford CME295 Lecture 2: Transformer-Based Models & Tricks",
    questions: stanfordCME295Lecture2Questions,
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
  (s) => s.questions
);

// Helper: get questions for a given mode
export function getQuestionsForMode(mode: Mode): Question[] {
  if (mode === "all") return allQuestions;
  const src = QUESTION_SOURCES.find((s) => s.id === mode);
  return src ? src.questions : allQuestions;
}

// Helper: title for the current mode (for page header)
export function getTitleForMode(mode: Mode): string {
  if (mode === "all") {
    return "All Chapters Quiz – Text, Transformers & LLMs";
  }
  const src = QUESTION_SOURCES.find((s) => s.id === mode);
  return src ? src.title : "Quiz";
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
export { mixedQuestions } from "./other/other";