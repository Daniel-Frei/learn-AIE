import type { Question } from "./quiz";

/** Reuse the canonical object and database ID; never clone or override its content. */
export function referenceQuestion(
  questions: readonly Question[],
  id: string,
): Question {
  const matches = questions.filter((question) => question.id === id);
  if (matches.length !== 1) {
    throw new Error("Question reference must resolve exactly once: " + id);
  }
  return matches[0];
}

/** Source memberships may repeat an object, but an identity has one definition. */
export function uniqueQuestions(questions: readonly Question[]): Question[] {
  const byId = new Map<string, Question>();
  for (const question of questions) {
    const previous = byId.get(question.id);
    if (previous && previous !== question) {
      throw new Error(
        "Conflicting question definitions for ID: " + question.id,
      );
    }
    byId.set(question.id, question);
  }
  return [...byId.values()];
}
