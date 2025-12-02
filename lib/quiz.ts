// lib/quiz.ts
import { chapter1Questions } from "./chapter1";
import { chapter2Questions } from "./chapter2";
import { chapter3Questions } from "./chapter3";
import { aieChapter2Questions } from "./books/AIE_building_apps/chapter2";

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

export const allQuestions: Question[] = [
  ...chapter1Questions,
  ...chapter2Questions,
  ...chapter3Questions,
  ...aieChapter2Questions,
];

export { chapter1Questions } from "./chapter1";
export { chapter2Questions } from "./chapter2";
export { chapter3Questions } from "./chapter3";
export { aieChapter2Questions } from "./books/AIE_building_apps/chapter2";