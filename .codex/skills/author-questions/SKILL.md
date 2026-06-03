---
name: author-questions
description: Create, extend, edit, rebalance, and register Learning AI quiz question sets. Use when the user asks Codex to generate new TypeScript question-bank files from transcripts, chapters, papers, slides, PDFs, or other learning material under this repo, or to add, revise, rebalance, or fix questions in existing Learning AI question files.
---

# Author Learning AI Questions

## Overview

Create and maintain high-quality multi-select quiz question sets for the Learning AI repo. For new source material, generate one or more question files and register each set so it appears in the app's source selector. For existing question sets, preserve the local style while adding, revising, rebalancing, or fixing questions.

Use the repo docs as product context. Keep edits scoped to the new question files, `lib/quiz.ts` registration, tests/docs updates when required, and small supporting changes needed for verification.

## Folder Workflow

1. Identify the requested operation.
   - For new source material, follow the folder workflow below.
   - For an existing question file, inspect the current set before editing and preserve its naming, IDs, topic scope, and registration unless the user asks for a broader change.
   - For additions to an existing set, continue the existing ID sequence and rebalance answer-count distribution across the full file, not only the new questions.
   - For fixes, update the prompt, options, `isCorrect` flags, and explanation together so they remain consistent.

2. Identify the input folder and source materials when generating new sets.
   - Source material can be a transcript, book chapter, research paper, slides, PDF, markdown notes, or another learning artifact.
   - If the folder is named `transcripts-and-files`, create question files in its parent directory by default.
   - Treat files with the same stem as one source bundle when appropriate, such as `lecture5.md` plus `lecture5.pdf`.
   - Generate one question set per new source bundle. If the user says they added source material for items 5 and 6, create two new question files.

3. Inspect existing local patterns before generating or editing.
   - Read nearby question files in the target directory for naming style, export names, `chapter` values, question IDs, and source labels.
   - Read `lib/quiz.ts` to understand the existing `seriesId`, `seriesLabel`, `topic`, labels, titles, and registration style.
   - Prefer existing series conventions over inventing new naming.

4. Generate or edit the TypeScript question file.
   - Before generating a new set, determine the requested question count. If the user did not specify one, ask for the count before generating questions; do not assume a default count from this skill.
   - If context is tight, draft in multiple passes, but the final file should contain the complete requested set.
   - After drafting, run the adversarial answer-option review below before considering questions complete.
   - Use a filename based on the source item and topic, for example `lecture5_attention.ts`, `chapter4_agents.ts`, or the closest existing series convention.
   - Use a stable ASCII export name ending in `Questions`.
   - Use the correct relative import for `Question` from the new file to `lib/quiz.ts`.

5. Register each new question set so the user can practice it.
   - Add an import in `lib/quiz.ts`.
   - Add a `QUESTION_SOURCES` entry with a unique `id`, concise `label`, descriptive `title`, existing or new `seriesId`, `seriesLabel`, appropriate `topic`, and the imported question array.
   - Add a matching `QUESTION_SOURCE_CONTEXT` sentence. This should help mixed-source practice show where the prompt came from without making individual questions depend on the source artifact.
   - If this is a new series, extend the `SourceSeriesId` union.
   - Add the final re-export near the bottom of `lib/quiz.ts`.
   - Confirm `tests/lib/question-registration.spec.ts` would find no unregistered question files.

6. Update docs when the change alters durable product behavior or repo conventions.
   - If a new source series is added or the available question-bank scope changes materially, update `docs/product-scope.md`.
   - If a durable authoring or process preference is introduced, update `docs/team-preferences.md`.

7. Verify.
   - Prefer `make check` before finishing when practical.
   - For focused verification during question-bank work, at least run `npm run test:focused -- tests/lib/question-registration.spec.ts` and `make types-check`.
   - After creating or editing registered question sets, run the targeted guessability heuristic for each changed source id: `$env:QUESTION_GUESSABILITY_SOURCE_IDS="source-id"; npm run test:question-guessability`. Use comma-separated source ids for multiple changed sets.
   - If the guessability test fails, treat it as an answer-option quality issue. Revise distractors and prompts before rerunning instead of weakening the thresholds, unless there is a documented source-specific reason.
   - If formatting may have changed, run `make format-check`.
   - If any command fails, include the exact error output and fix the root cause.

## Question Requirements

- Use TypeScript question files with four options per question.
- Each option must have `isCorrect: true` or `isCorrect: false`; questions are multi-select and more than one option can be correct.
- Each question must have exactly four options.
- Option order should avoid predictable patterns. Do not rely on ordering, such as "first ... then ..." dependencies.
- Questions can cover simple terminology, definitions, core concepts, complex concepts, applications, connections between ideas, and math.
- Include math-related questions when math is part of the source material. The amount of math should be proportional to its importance in the material.
- Do not include questions about logistics or administration, such as exams, course structure, or resources.
- Do not refer to the source material directly in prompts. Avoid phrases like "in the lecture", "in the transcript", "in the chapter", "in the paper", "the equation above", or "the slides show".
- Questions should test whether learners really understand the concepts and can use them, not only whether they can recognize familiar wording or eliminate obviously wrong statements.
- Difficulty should come from the level of knowledge, reasoning, transfer, or math required by the concept, not from tricky wording or low-quality answer options.

## Difficulty

- Use `"easy"`, `"medium"`, and `"hard"` labels to describe the level of knowledge and understanding required, not the guessability of the answer options.
- If the user, source material, existing set convention, or local docs specify a difficulty distribution, follow that specified distribution.
- If no difficulty distribution was specified, default to a roughly balanced mix of `"easy"`, `"medium"`, and `"hard"` labels when the source material supports it.
- Always report the final difficulty balance to the user, including the counts for `"easy"`, `"medium"`, and `"hard"` questions and any deliberate deviation from a balanced default.
- Easy questions cover terminology, definitions, and core concepts.
- Medium and hard questions cover deeper understanding, application, connections between concepts, or math.
- Easy questions should support understanding of harder ones by defining terms used later.

## Coverage

- Cover all major concepts from the source material.
- Make coverage proportional to the source material's emphasis. More important or repeated topics should receive more questions.
- Avoid repeated or near-duplicate questions unless the source material explicitly requires closely related distinctions.
- If a set teaches a recurring schema or contrast, later questions must require new source-specific detail, conditions, mechanisms, calculations, or boundary cases. Do not let a learner answer many later questions by reusing one generic rule such as "prefer the nuanced/context-dependent option" or "reject the option that skips validation."

## Prompt Design

- Prompts should identify the concept or task precisely without revealing the abstract answer pattern. Avoid stems where a learner can infer the answer from generic framing alone, such as "why is X not enough?", "why can X fail to generalize?", or "why does X still need validation?", unless the options all require a specific source-grounded distinction.
- For "why", "what explains", and "best describes" prompts, make all options plausible answers to that same prompt. Do not make the correct option the only one that addresses context, evidence, tradeoffs, uncertainty, or limitations.
- When the source repeatedly uses a broad template, such as information flow, selection pressure, error analysis, optimization tradeoffs, evidence chains, or system boundaries, anchor each question to the concrete version from the source rather than asking for the generic template again.

## Answer Options

- Options must be independent; do not refer from one option to another.
- Each option in a question should compete on the same semantic axis. Do not pair one detailed, technically plausible correct option with three broad anti-claims, absurdities, or generic "this removes all risk/tests/security concerns" statements.
- Incorrect options should be plausible same-neighborhood alternatives, partial truths, common confusions, nearby quantities, nearby conditions, wrong scope boundaries, wrong causal directions, or wrong inclusion/exclusion details that help diagnose misunderstandings.
- A learner should usually need concept understanding to distinguish correct options from incorrect options. Avoid distractors that are absurd, category-mismatched, self-refuting, or easy to eliminate from wording cues alone.
- Answer options should also teach useful contrasts about the concept being tested. A learner practicing the question should learn something relevant from why an option is wrong, such as a nearby misconception, prerequisite distinction, or boundary condition; avoid random cross-topic distractors whose only lesson is that the concept is not an unrelated statistic, study-design term, drug concept, or other category mismatch.
- Avoid making the correct answer the only "reasonable middle" option: useful but limited, context-dependent, evidence-aware, partially true, or needing validation. Incorrect options can also be moderate and evidence-aware while still being wrong because they shift one substantive detail, condition, direction, layer, or scope boundary.
- Keep distractors in the same category and reasoning layer as the prompt. If the prompt asks about a method, model, mechanism, metric, policy, API, proof, or intervention, wrong options should usually be nearby versions of that same kind of thing rather than unrelated objects, outcomes, roles, or workflow stages.
- Avoid wrong answers that are false only because they collapse a general chain from mechanism to measurement to deployment, decision, or outcome. Strong distractors should preserve most of the chain but make one meaningful mistake about which link is sufficient, missing, reversed, or out of scope.
- Absolute or high-certainty language such as "always", "never", "only", "every", "all", "none", "cannot", "impossible", "guarantees", "proves", or "complete" is not banned, but it is high risk. Use it only when the source, math, or formal definition supports the exact quantifier, and avoid making these cues correlate mostly with incorrect options.
- Keep option specificity, detail level, and plausibility roughly comparable. If the correct option is much more concrete, technical, or carefully hedged than the incorrect options, rewrite the distractors to be closer competitors.

## Adversarial Answer Review

Before finalizing a generated or revised set, review it from the perspective of a reasonably educated learner who has not read the source and is trying to game the quiz.

- Hide the `isCorrect` flags and ask whether the learner could answer by generic test-taking heuristics instead of source understanding.
- If marking options with absolutes or overclaims as false would get many options right, revise the affected distractors and any matching prompts.
- If choosing the option that sounds like the reasonable scientific or technical middle would get many options right, revise so several options are nuanced and the distinction turns on source-specific substance.
- If rejecting semantic oddballs, category mismatches, or options from a different reasoning layer would get many options right, rewrite those distractors into same-neighborhood misconceptions.
- If the stem itself reveals the answer frame, such as context-dependence, validation, uncertainty, model limits, or a missing comparison, make the prompt more concrete and make all options plausible within that frame.
- If the longest, most technical, or most carefully hedged option is usually correct, revise so incorrect options have comparable detail and credibility.
- For "best describes" questions, make all four options candidate descriptions of the same thing. Do not make the incorrect options mostly claims about what the concept does not do.
- Prefer near-miss distractors created by changing one meaningful detail of the correct concept: component inclusion, scope, precondition, feedback signal, direction of causality, authority boundary, failure mode, or tradeoff.
- Do not let difficulty come from trick wording. The question should be hard because the tested distinction is substantive, while the options remain clear and fair.
- When practical for a large set, run or mentally simulate a simple language-cue baseline such as "mark options containing always/never/only/every/guarantees/proves/complete as false and the rest as true." If that baseline would score well at the option level, the set is still too guessable.
- Also simulate broader baselines: "select the hedged middle answer", "reject the category mismatch", and "infer the context/validation answer from the stem." If any of these work across many questions, revise prompts and distractors before finalizing.
- For registered question sets, prefer the deterministic check over mental simulation: `$env:QUESTION_GUESSABILITY_SOURCE_IDS="source-id"; npm run test:question-guessability`.

## Answer Distribution

- Across each complete question set, keep 1-, 2-, 3-, and 4-correct-answer questions as balanced as practical unless the user or existing set convention specifies another pattern.
- For small or tightly scoped sets, prioritize source coverage and answer quality over exact answer-count balance.
- "Correct" means `isCorrect: true`, meaning the user should select that option. It does not merely mean whether a statement is factually true in isolation.
- Mixed phrasing is allowed, such as "which are correct" or "which are false", but the prompt must align exactly with the `isCorrect` flags.

## Explanations

- Explanations should help users understand the topic covered by the prompt and answer statements.
- Each explanation must be at least two sentences.
- Explain why correct options are correct and why incorrect options are incorrect.
- Do not refer to answer order, such as "option A" or "the second answer".
- Use simple, teaching-oriented language.
- When in doubt, prefer a slightly longer explanation over one that is too short.

## Math And Formatting

- Write out acronyms at least once, for example `Recurrent Neural Network (RNN)`.
- Use math notation where relevant and proportional to the source material.
- For math, wrap LaTeX in:
  - Inline math: `\( ... \)`
  - Block math: `\[ ... \]`
- In TypeScript strings, escape LaTeX delimiters:
  - `\\( ... \\)`
  - `\\[ ... \\]`
- Use standard LaTeX syntax, such as `\\frac{a}{b}`, `\\sum`, `\\theta`, and `\\pi`.

## TypeScript Template

Use this structure for new source-material question files under `lib/**`. Adjust the import path based on the file location.

```ts
import { Question } from "../../quiz";

export const sourceMaterialQuestions: Question[] = [
  {
    id: "source-id-q01",
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
      "At least two sentences (at least 200 characters). Explain why correct options are correct and why incorrect options are incorrect.",
  },
];
```

## Final Checklist

- Exactly four options per question.
- Multi-select compatible, with 1, 2, 3, or 4 correct answers.
- If generating a new set, the user specified the question count or you asked for it before generating questions.
- Difficulty is exactly `"easy"`, `"medium"`, or `"hard"`, follows any user/source-specified distribution, and otherwise uses a reasonable balanced default when the source material supports it.
- Final response reports the difficulty balance with `"easy"`, `"medium"`, and `"hard"` counts.
- Answer options are plausible enough that the learner needs concept understanding rather than elimination of obviously wrong distractors.
- Adversarial answer review has been applied: absolutes, overclaims, option length/detail, and obvious false-option cues do not let a generic heuristic score well.
- Targeted guessability test passes for each created or edited registered question set, or any failure has been fixed at the question level.
- No prompt depends on seeing the source material, transcript, chapter, paper, or slides.
- Every explanation has at least two sentences and covers all options.
- Math uses escaped LaTeX delimiters inside TypeScript strings.
- Question IDs are unique across the repo.
- The new question file is imported, registered in `QUESTION_SOURCES`, described in `QUESTION_SOURCE_CONTEXT`, and re-exported from `lib/quiz.ts`.
- Registration tests and type checks pass, or failures are reported with exact errors.
