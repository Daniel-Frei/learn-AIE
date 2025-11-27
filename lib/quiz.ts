// lib/quiz.ts
import { chapter1Questions } from "./chapter1";
import { chapter2Questions } from "./chapter2";

export type Difficulty = "easy" | "medium" | "hard";

export type Question = {
  id: string;
  chapter: number;
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
];

export { chapter1Questions } from "./chapter1";
export { chapter2Questions } from "./chapter2";