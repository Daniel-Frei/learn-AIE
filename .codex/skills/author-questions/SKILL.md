---
name: author-questions
description: Create, extend, edit, rebalance, and register Learning AI quiz question sets with stable question IDs. Use when the user asks Codex to generate new TypeScript question-bank files from transcripts, chapters, papers, slides, PDFs, or other learning material under this repo, or to add, revise, rebalance, fix, or preserve database-linked IDs in existing Learning AI question files.
---

# Author Learning AI Questions

## Overview

Create, review, and maintain high-quality mixed-format quiz question sets for the Learning AI repo. Supported question types are multi-select multiple-choice and assertion-reason MCQs. For new source material, generate one or more question files and register each set so it appears in the app's source selector. For existing question sets, preserve the local style while reviewing quality, adding questions, revising weak items, rebalancing answer patterns, rewriting topic slices, or fixing reported issues. Every question must stand alone because practice can randomize question order and mix questions from different source sets. Every question ID is also a persistent database identity, so do not treat IDs as display numbers or derive them from question order.

Use the repo docs as product context. Keep edits scoped to the new question files, `lib/quiz.ts` registration, tests/docs updates when required, and small supporting changes needed for verification.

## Supported Operations

Use the same quality gate for creation, review, and improvement work. The skill supports these common request shapes:

- Create a new question set from source material, such as "create 30 questions about X."
- Add questions to an existing set, such as "add 10 questions about C to set X."
- Review an existing set for quality issues. If the user asks only for a review, report findings without editing; if the user asks to improve issues, make the scoped fixes.
- Improve an existing set by rewriting low-quality prompts, options, explanations, answer flags, or difficulty labels while preserving IDs only for unchanged or minor-edited questions.
- Rewrite a targeted topic slice inside a larger set, such as "rewrite all questions about topic A in set X," while leaving unrelated questions alone.
- Combine operations in one pass, such as adding new questions and rewriting a targeted subset of existing questions.

## Stable Question IDs

Question IDs are database identities. They must be explicit, hardcoded strings attached to the question item, not generated from array position, helper call order, or a mutable counter. Preserve an existing question's `id` when making minor wording, option, explanation, or difficulty fixes that keep the same underlying item. Assign a new never-before-used `id` when adding a question, changing the tested construct, changing the answer key in a way that changes scoring semantics, or rewriting the prompt/options/explanation substantially enough that the item should be treated as a new question. Removing a question removes that ID from the bundled bank; do not reuse removed IDs for future questions.

## Folder Workflow

1. Identify the requested operation.
   - For new source material, follow the folder workflow below.
   - For an existing question file, inspect the current set before editing and preserve its naming, topic scope, and registration unless the user asks for a broader change.
   - For existing questions, compare against the committed/pre-edit version and preserve each old ID only when the edited item is still the same question. If the item is completely replaced or significantly rewritten, give it a new ID that has not appeared in the set before.
   - For additions to an existing set, choose new hardcoded IDs that continue the source's ID namespace without reusing removed or historical IDs, then rebalance answer-count distribution across the full file, not only the new questions.
   - For quality reviews, use the same question-quality criteria as generation. If the user asks to improve issues, edit the set; if the user asks only to review, report actionable findings and verification gaps.
   - For targeted rewrites inside an existing set, identify the matching topic slice first and keep unrelated questions out of scope.
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
   - For review or improvement tasks, inspect enough of the full set to understand repeated schemas, answer-count distribution, difficulty balance, and topic coverage before editing only the requested slice.

4. Generate or edit the TypeScript question file.
   - Before generating a new set, determine the requested question count. If the user did not specify one, ask for the count before generating questions; do not assume a default count from this skill.
   - If context is tight, draft in multiple passes, but the final file should contain the complete requested set.
   - After drafting or editing, run the question quality gate below before considering questions complete.
   - Hardcode each question ID at the question or helper-call site. If a local helper is used, pass the ID as a string argument; do not derive it from the helper call number or array index.
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
   - If the guessability test passes, do not treat that as evidence that the set has high diagnosticity. The deterministic check is only a smoke test for simple surface cues; still run the question quality gate below for created, added, rewritten, or otherwise edited questions.
   - If formatting may have changed, run `make format-check`.
   - If any command fails, include the exact error output and fix the root cause.

## Question Quality Gate

Run this gate whenever this skill creates, adds, rewrites, fixes, or otherwise edits questions. Passing deterministic guessability checks is not enough, because many weak items are answerable by recognizing one plausible sentence among implausible ones.

For new question sets, run the gate over the full draft before finalizing. For additions, rewrites, and fixes, run it over every changed question plus enough surrounding questions to catch repeated schemas and answer-pattern drift. For review-only tasks, perform the same triage but stop at findings unless the user asked you to edit. For combined add-and-rewrite tasks, run the triage over the affected existing slice and apply the same quality bar to the newly added questions.

1. Audit each question with `isCorrect` hidden.
   - Mark a question low-diagnosticity if a learner can answer by matching object categories, rejecting extreme wording, choosing the broad/hedged reasonable option, spotting the familiar definition, following wording in the stem, or reusing a repeated course theme.
   - Mark definition-recognition items low-diagnosticity when three options are plainly about other concepts and only one option is the right kind of thing.
   - Mark broad all-true or mostly-true multi-select items low-diagnosticity when the true options are generic introductory truths and false options are absurd, impossible, or overclaimed.

2. For each low-diagnosticity item, name the construct target and misconception target.
   - The construct target is what understanding the question should measure, such as applying a definition, predicting a consequence, explaining a mechanism, comparing two plausible alternatives, using math, or transferring the idea to a new case.
   - The misconception target is the plausible mistake a real learner might make after partial study.
   - If you cannot name both targets, rewrite the item instead of only polishing wording.

3. Prefer substantial rewrites over small distractor edits for low-diagnosticity items.
   - When a weak definition-recognition item needs a rewrite, choose the least forced question form that fits the source material: a direct prompt with better sibling definitions, a boundary case, consequence, mechanism, comparison, calculation, transfer, or a concise scenario.
   - Use scenarios when the topic naturally involves a decision, observation, workflow, patient/system case, or experimental setup. A useful scenario supplies facts that affect the answer choice; if the setup can be removed without changing the reasoning, prefer a direct prompt.
   - Keep orientation questions intentionally easy only when they are serving prerequisite confidence or vocabulary setup; label them `"easy"` and do not let them dominate medium or hard coverage.
   - Rewrite the correct option too when it is broad enough to be true from common sense. Make it source-specific enough that understanding is required.

4. After editing, run a second hidden-answer review.
   - Ask whether a learner could still solve without knowing why the right options are right.
   - Ask whether every false option represents a plausible nearby misconception rather than merely sounding wrong.
   - Ask whether the same broad theme could answer many questions in the set.

5. Report the quality gate in the final response.
   - Include how many questions were created or changed, how many were reviewed by the gate, how many were substantially rewritten, how many received minor option/explanation edits, and how many intentionally easy orientation items were retained.
   - If only a small fraction of low-diagnosticity items were rewritten, say so instead of implying the whole set quality was fixed.

## Question Requirements

- Use TypeScript question files with the option count required by the question type.
- Question files may mix two question types:
  - Multi-select multiple-choice questions. Existing questions can omit `type`; omitted `type` means `multiple-select`. Use exactly four options; each option must have `isCorrect: true` or `isCorrect: false`, and 1, 2, 3, or 4 options can be correct.
  - Assertion-reason MCQs. Set `type: "assertion-reason"`, write a prompt containing an **Assertion** and a **Reason**, and use the standard fixed five-option order. Exactly one option should be correct.
- Choose the mix of question types based on the source material, subject, and field.
- Multi-select answer options should be order-independent because the frontend randomizes them. Assertion-reason answer options should not rely on randomization and should keep this authored order:
  - `Assertion is true, Reason is false.`
  - `Assertion is false, Reason is true.`
  - `Both are false.`
  - `Both are true, and the Reason is the correct explanation of the Assertion.`
  - `Both are true, but the Reason is NOT the correct explanation of the Assertion.`
- Do not include a trailing instruction sentence such as "Which option correctly evaluates the assertion and reason?" in assertion-reason prompts. Use `Assertion: ...\n\nReason: ...` so the frontend displays the two statements as separate paragraphs.
- For multi-select questions, option order should avoid predictable patterns. Do not rely on ordering, such as "first ... then ..." dependencies.
- Questions can cover simple terminology, definitions, core concepts, complex concepts, applications, connections between ideas, and math.
- Include math-related questions when math is part of the source material. The amount of math should be proportional to its importance in the material.
- Do not include questions about logistics or administration, such as exams, course structure, or resources.
- Do not refer to the source material directly in prompts. Avoid phrases like "in the lecture", "in the transcript", "in the chapter", "in the paper", "the equation above", or "the slides show".
- Do not refer to other questions or depend on a previous prompt, answer, explanation, image, equation, or source-context card. Avoid phrases like "the previous question", "the next question", "as above", "from earlier", "this same example", or "the equation in another question".
- Each question must be self-contained because the frontend can randomize question order and interleave questions from different question sets. If a concept needs setup, include the necessary setup inside that question or rewrite it so no setup is needed.
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
- Do not overuse "which statement best describes" for basic definitions. If a definition prompt is necessary, make the alternatives plausible sibling definitions or boundary cases rather than unrelated categories.
- Prefer prompts that ask the learner to apply, predict, compare, explain, calculate, or transfer a concept. Pure recognition of a familiar sentence should be reserved for intentional easy orientation questions.
- Across `lib/` topics, vary the prompt form to match the material. Clinical, procedural, or experimental topics may support scenarios; foundational, mathematical, or abstract topics often work better as direct comparisons, boundary cases, mechanisms, calculations, or consequence questions.
- Do not add a scenario sentence merely to make a direct concept question look applied. The setup should be referenced by, constrain, or change the interpretation of the answer options.
- Use compact Markdown tables in prompts when the learner needs to compare several conditions, probability values, scenarios, cases, model states, trial arms, matrix-like quantities, or calculation inputs. Prefer a table over a dense sentence when the prompt contains several named variables, log probabilities, model/reference quantities, policy states, or arithmetic inputs that the learner must line up before solving.
- Introduce what the table represents before the table, use meaningful row and column labels, and make answer options refer to labels or quantities rather than visual position alone.
- Do not use a table as decoration or to hide missing explanation. If the table needs special context, include that context in the prompt so the question still stands alone.
- Watch for stems whose everyday language gives away the answer, such as terms that already imply stability, evidence, usefulness, feedback, or decision relevance. In those cases, ask for a concrete mechanism, condition, exception, or consequence.
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
- For multi-select questions, do not make the correct answer recognizable as the only math-heavy, formula-heavy, calculation-heavy, or KaTeX/LaTeX option. If a formula is useful, add plausible competing formulas, calculations, dimensions, or boundary cases as distractors, or rewrite the prompt so the math is part of the reasoning instead of a visual answer cue.
- Do not remove useful math merely to avoid a math-salience cue. The fix is balanced, same-neighborhood mathematical alternatives, not avoiding formulas when math is central to the source material.
- Every distractor should pass the "real learner" test: a learner with a partial misconception could plausibly select it for a substantive reason.
- For multi-select items with several true options, avoid making all true options broad introductory facts and all false options absurd. Mix specificity and plausibility so selecting all correct options requires discriminating understanding.

## Adversarial Answer Review

Before finalizing a generated or revised set, review it from the perspective of a reasonably educated learner who has not read the source and is trying to game the quiz.

- Hide the `isCorrect` flags and ask whether the learner could answer by generic test-taking heuristics instead of source understanding.
- If marking options with absolutes or overclaims as false would get many options right, revise the affected distractors and any matching prompts.
- If choosing the option that sounds like the reasonable scientific or technical middle would get many options right, revise so several options are nuanced and the distinction turns on source-specific substance.
- If rejecting semantic oddballs, category mismatches, or options from a different reasoning layer would get many options right, rewrite those distractors into same-neighborhood misconceptions.
- If the stem itself reveals the answer frame, such as context-dependence, validation, uncertainty, model limits, or a missing comparison, make the prompt more concrete and make all options plausible within that frame.
- If the longest, most technical, or most carefully hedged option is usually correct, revise so incorrect options have comparable detail and credibility.
- For multi-select questions, simulate "select the only option with substantial math, a long formula, a calculation, or KaTeX/LaTeX." If that works, add plausible competing mathematical distractors or remove gratuitous formula salience from conceptual prompts.
- For "best describes" questions, make all four options candidate descriptions of the same thing. Do not make the incorrect options mostly claims about what the concept does not do.
- Prefer near-miss distractors created by changing one meaningful detail of the correct concept: component inclusion, scope, precondition, feedback signal, direction of causality, authority boundary, failure mode, or tradeoff.
- Do not let difficulty come from trick wording. The question should be hard because the tested distinction is substantive, while the options remain clear and fair.
- When practical for a large set, run or mentally simulate a simple language-cue baseline such as "mark options containing always/never/only/every/guarantees/proves/complete as false and the rest as true." If that baseline would score well at the option level, the set is still too guessable.
- Also simulate broader baselines: "select the hedged middle answer", "reject the category mismatch", and "infer the context/validation answer from the stem." If any of these work across many questions, revise prompts and distractors before finalizing.
- For registered question sets, prefer the deterministic check over mental simulation: `$env:QUESTION_GUESSABILITY_SOURCE_IDS="source-id"; npm run test:question-guessability`.

## Answer Distribution

- Across the multi-select questions in each complete question set, keep 1-, 2-, 3-, and 4-correct-answer questions as balanced as practical unless the user or existing set convention specifies another pattern.
- For assertion-reason questions, exactly one option should be correct. When a set contains several assertion-reason questions, vary which of the fixed five ordered options is correct as practical.
- For small or tightly scoped sets, prioritize source coverage and answer quality over exact answer-count balance.
- "Correct" means `isCorrect: true`, meaning the user should select that option. It does not merely mean whether a statement is factually true in isolation.
- Mixed phrasing is allowed, such as "which are correct" or "which are false", but the prompt must align exactly with the `isCorrect` flags.

## Explanations

- Explanations should help users understand the topic covered by the prompt and answer statements.
- Users read explanations after submitting an answer, when the UI already shows which choices are correct or incorrect. Focus on why each choice has that status rather than restating the status.
- Each explanation must be at least two sentences.
- Explain why correct options are correct and why incorrect options are incorrect.
- Avoid explanations that merely repeat an answer option, quote it back, or label it as a misconception without adding the missing concept. Bad: `The option "All swans are white" is incorrect.` Better: `Black swans living in Australia disprove the universal claim, so the statement overgeneralizes from limited observations.`
- Do not refer to answer order, such as "option A" or "the second answer".
- Use simple, teaching-oriented language.
- When in doubt, prefer a slightly longer explanation over one that is too short.

## Math And Formatting

- Write out acronyms at least once, for example `Recurrent Neural Network (RNN)`.
- Use math notation where relevant and proportional to the source material.
- Math is welcome when it tests or teaches the source material. In multi-select answer options, keep math salience balanced enough that learners cannot guess by choosing the only option with a formula or long calculation.
- For math, wrap LaTeX in:
  - Inline math: `\( ... \)`
  - Block math: `\[ ... \]`
- In TypeScript strings, escape LaTeX delimiters:
  - `\\( ... \\)`
  - `\\[ ... \\]`
- Use standard LaTeX syntax, such as `\\frac{a}{b}`, `\\sum`, `\\theta`, and `\\pi`.
- Prompts may include GitHub-Flavored Markdown tables. Prefer a template literal for multi-line prompts with tables, include a blank line before and after the table, and keep tables compact enough to read on mobile. For dense mathematical prompts, first consider whether a small table of given values would be clearer than embedding all givens in one sentence.
- Table cells can include escaped inline LaTeX, but avoid making the correct option obvious by placing math only in one answer option or one unusually prominent table cell.

## TypeScript Template

Use this structure for new source-material question files under `lib/**`. Adjust the import path based on the file location. If you use a helper to reduce repetition, the helper call should still receive the explicit ID string, for example `makeQuestion("source-id-q01", ...)`; do not generate IDs from sequence numbers.

```ts
import { Question } from "../../quiz";

export const sourceMaterialQuestions: Question[] = [
  {
    id: "source-id-q01",
    chapter: 1,
    difficulty: "easy",
    type: "multiple-select", // optional; omit this for ordinary multi-select questions if preferred
    prompt: "Question text here.",
    options: [
      { text: "Option 1 text", isCorrect: true },
      { text: "Option 2 text", isCorrect: false },
      { text: "Option 3 text", isCorrect: true },
      { text: "Option 4 text", isCorrect: false },
    ],
    explanation:
      "At least two sentences (at least 200 characters). Explain why correct options are correct and why incorrect options are incorrect. Provide knowledge required to understand the question.",
  },
];
```

For table-based prompts, prefer a template literal so the Markdown stays readable:

```ts
prompt: `A latent-variable model uses these probabilities:

| Hidden state | Prior | Likelihood of \\(X=x\\) |
| --- | ---: | ---: |
| \\(Z=0\\) | 0.3 | 0.8 |
| \\(Z=1\\) | 0.7 | 0.2 |

What is \\(P(X=x)\\)?`,
```

For assertion-reason questions, use this shape inside the same `Question[]` array:

```ts
{
  id: "source-id-q02",
  chapter: 1,
  difficulty: "medium",
  type: "assertion-reason",
  prompt:
    "Assertion: Statement to evaluate.\n\nReason: Proposed explanation to evaluate.",
  options: [
    { text: "Assertion is true, Reason is false.", isCorrect: false },
    { text: "Assertion is false, Reason is true.", isCorrect: false },
    { text: "Both are false.", isCorrect: false },
    {
      text: "Both are true, and the Reason is the correct explanation of the Assertion.",
      isCorrect: true,
    },
    {
      text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
      isCorrect: false,
    },
  ],
  explanation:
    "At least two sentences (at least 200 characters). Explain whether the assertion is true, whether the reason is true, and whether the reason explains the assertion. Provide knowledge required to understand the question.",
}
```

## Final Checklist

- Multi-select questions have exactly four options.
- Multi-select questions are compatible with 1, 2, 3, or 4 correct answers.
- Assertion-reason questions have exactly five options, set `type: "assertion-reason"`, use the standard assertion/reason options in authored order, and have exactly one correct answer.
- Mixed question sets choose question type based on what best tests the source material, not a fixed quota.
- If generating a new set, the user specified the question count or you asked for it before generating questions.
- Difficulty is exactly `"easy"`, `"medium"`, or `"hard"`, follows any user/source-specified distribution, and otherwise uses a reasonable balanced default when the source material supports it.
- Final response reports the difficulty balance with `"easy"`, `"medium"`, and `"hard"` counts.
- Answer options are plausible enough that the learner needs concept understanding rather than elimination of obviously wrong distractors.
- Adversarial answer review has been applied: absolutes, overclaims, option length/detail, and obvious false-option cues do not let a generic heuristic score well.
- Multi-select math questions do not make the correct answer the only math-heavy or KaTeX/LaTeX-heavy option; when formulas are used, plausible competing formulas or calculations are present where appropriate.
- For every created, added, rewritten, fixed, or otherwise edited question, low-diagnosticity risk has been triaged by construct target and misconception target, and weak recognition items were substantially rewritten rather than lightly polished.
- Passing the deterministic guessability test was not used as the sole evidence of question quality.
- Targeted guessability test passes for each created or edited registered question set, or any failure has been fixed at the question level.
- No prompt depends on seeing the source material, transcript, chapter, paper, or slides.
- No prompt or explanation depends on seeing another question first; every question is independent under randomized mixed-source practice.
- Prompt tables, when used, are valid GitHub-Flavored Markdown, compact, labeled, introduced by the prompt, and self-contained.
- Every explanation has at least two sentences and covers all options.
- Math uses escaped LaTeX delimiters inside TypeScript strings.
- Question IDs are hardcoded, stable database identities; preserve old IDs for minor edits, assign new never-before-used IDs for new or substantially rewritten questions, and keep IDs unique across the repo.
- The new question file is imported, registered in `QUESTION_SOURCES`, described in `QUESTION_SOURCE_CONTEXT`, and re-exported from `lib/quiz.ts`.
- Final response for creation or editing work reports created/changed, reviewed, substantially rewritten, minor-edited, and intentionally retained orientation counts.
- Registration tests and type checks pass, or failures are reported with exact errors.
