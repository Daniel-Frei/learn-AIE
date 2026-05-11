I want you to create 40 questions so that I can learn the lecture (see transcript below and attached pdf for the slides). As this often requires a lot of tokens, I think it's best to break this up into two batches (provide 20 questions, then in a next step we can add 20 more questions).

⚠️ Important: The requirements (e.g. distribution of answer types and difficulty) apply to the full set of 40 questions. When generating a batch of 20, approximate these distributions proportionally (e.g. ~5 questions per answer type).

I already created `lib\lectures\MIT 15.773 Hands-On Deep Learning Spring 2024\L1_ Introduction to Neural Networks and Deep Learning.ts` for those questions. Note that this is referring to a lecture but the same requirements (below) as for chapters apply. The learning material can be a book, lecture, slides or other sources.

---

## Requirements:

- Same format as defined below (TypeScript template) with four options per question. Each option has `isCorrect: true/false` (multi-select; more than one answer can be correct).
- Each question must have exactly four options.
- The order of the options / answers should avoid predictable patterns. Do not rely on ordering (e.g. avoid dependencies like "first... then...").
- Questions can cover anything from simple terminology (e.g. "what does RNN stand for?") to complex concepts. The goal is to test and practice understanding of the material.
- Include math-related questions when math is part of the lecture. The amount of math should be proportional to how much it is covered and how important it is.
- Do NOT include questions about logistics or admin (e.g. exams, course structure, resources).
- Do NOT refer to the lecture, transcript, or chapter directly in the questions (e.g. avoid phrases like "in the lecture", "the equation above", etc.). Questions must be answerable based on knowledge alone.
- Questions should test user but also be a learning experience and help recall and reinforce concepts.

---

## Difficulty:

- Include a mix of `"easy"`, `"medium"`, and `"hard"` questions (roughly balanced).
- Easy questions: terminology, definitions, core concepts.
- Medium/Hard questions: deeper understanding, application, connections between concepts, or math.
- Easy questions should support understanding of harder ones (e.g. define terms used later).

---

## Coverage:

- Questions should cover all major concepts from the material.
- Coverage should be proportional to how much emphasis a topic receives in the lecture/chapter (more important topics → more questions).

---

## Answer Options:

- Options must be independent (no references to each other, no ordering assumptions).
- Avoid obvious guessing patterns (e.g. avoid extreme absolutes like "always", "never", unless truly correct).
- Incorrect options should still be plausible.

---

## Answer Distribution:

- Across the full set of 40 questions:
  - ~25%: 4 correct answers
  - ~25%: 3 correct answers
  - ~25%: 2 correct answers
  - ~25%: 1 correct answer

- When generating 20 questions, approximate this distribution proportionally.
- "Correct" refers to `isCorrect: true` (i.e. what the user should select), not whether the statement itself is true.
- Mixed phrasing is allowed (e.g. "which are correct", "which are false"), but ensure consistency between prompt and `isCorrect`.

---

## Explanations:

- Explanations are for users to better understand the topics covered in the question and answer statements.
- Each explanation must be at least two sentences.
- Explanations should clearly explain why the correct options are correct and why the incorrect options are incorrect.
- Do NOT refer to answer order (e.g. avoid "option A", "the second answer", etc.).
- Explanations should be written in simple, teaching-oriented language.

---

## Additional Requirements:

- Write out acronyms at least once (e.g. "Recurrent Neural Network (RNN)").
- Include math notation where relevant (proportional to its importance in the material).
- When in doubt, slightly longer explanations are preferred over too short ones.

---

## Formatting:

- Provide the questions as TypeScript code (see template below).
- Follow the structure exactly.

---

## Question Authoring Guide for Math & Formatting

For math, wrap LaTeX in:

Inline math: ( ... )
Block math: [ ... ]

In TypeScript strings, always escape:

- `\\( ... \\)`
- `\\[ ... \\]`

Use standard LaTeX syntax (e.g. `\\frac{a}{b}`, `\\sum`, `\\theta`, `\\pi`).

---

## Question File Template (TypeScript)

Use this exact structure for new lecture question files in `lib/lectures/*`.

```ts
import { Question } from "../../quiz";

export const <LectureExportName>: Question[] = [
  {
    id: "<lecture-id>-q01",
    chapter: 1,
    difficulty: "easy",
    prompt: "Question text here.",
    options: [
      { text: "Option 1 text", isCorrect: true },
      { text: "Option 2 text", isCorrect: false },
      { text: "Option 3 text", isCorrect: true },
      { text: "Option 4 text", isCorrect: false },
    ],
    explanation:
      "At least two sentences. Explain why correct options are correct and why incorrect options are incorrect.",
  },
];
```

---

## Content constraints checklist (must be satisfied)

- Exactly 4 options per question.
- Multi-select compatible (1–4 correct answers).
- Difficulty must be `"easy" | "medium" | "hard"`.
- No references to the lecture/chapter/transcript.
- Explanations ≥ 2 sentences and explain all options.
- Use escaped LaTeX when including math.
- Avoid repeated or near-duplicate questions.
