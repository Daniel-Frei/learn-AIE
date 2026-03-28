import { allQuestions } from "./quiz";
import type { QuestionMetadataMap } from "./ratingEngine";

export const QUESTION_METADATA: QuestionMetadataMap = Object.fromEntries(
  allQuestions.map((question) => [question.id, { label: question.difficulty }]),
);
