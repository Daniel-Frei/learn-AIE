# CME295 Lecture 3 Question-Set Improvement

## Goal

Replace the repetitive, low-diagnosticity Stanford CME295 Lecture 3 quiz with a smaller, slide-weighted set that tests application, comparison, calculation, and mechanism understanding.

## Non-goals

- Do not change quiz UI behavior or the Lecture 3 learning page.
- Do not refactor shared question or registry infrastructure.
- Do not alter other lecture question banks.

## Steps

- [x] Audit all 80 current questions and inspect all 125 lecture slides.
- [x] Define a proportional coverage blueprint centered on the deck's worked sequences.
- [x] Replace substantially rewritten questions with new stable IDs and balanced difficulty/answer patterns.
- [x] Add focused tests and update durable documentation.
- [x] Run registration, type, guessability, formatting, and full project checks as appropriate.

## Files to touch

- `lib/lectures/Stanford CME295 Transformers & LLMs/lecture3_LLMs.ts`
- `lib/lectures/Stanford CME295 Transformers & LLMs/transcripts-and-files/lecture 3 - curriculum.md`
- `tests/lib/cme295Lecture3Questions.spec.ts`
- `docs/product-scope.md`
- `docs/team-preferences.md`
- `PLANS.md`

## Verification

- `npm run test:focused -- tests/lib/cme295Lecture3Questions.spec.ts tests/lib/question-registration.spec.ts`
- `$env:QUESTION_GUESSABILITY_SOURCE_IDS="cme295-lect3"; npm run test:question-guessability`
- `make types-check`
- `make format-check`
- `make check`

---

# CME296 Lecture 1 Question Set

## Goal

Store the matching Stanford CME296 Lecture 1 transcript and slides in the established lecture-source structure, then add a registered 40-question set covering diffusion, DDPM, and DDIM.

## Non-goals

- Do not change quiz UI behavior or shared question infrastructure.
- Do not modify the existing CME295 Lecture 1 source files.
- Do not add a learning-experience page for this lecture.

## Steps

- [x] Inspect the supplied transcript and slides and resolve the source mismatch.
- [x] Store the matching source bundle under a new Stanford CME296 course folder.
- [x] Author 40 stable-ID questions with balanced difficulty and correct-answer counts.
- [x] Register the source and update focused tests and durable documentation.
- [x] Run focused, guessability, type, formatting, and full project checks.

## Files to touch

- `lib/lectures/Stanford CME296 Diffusion & Large Vision Models/lecture1_diffusion.ts`
- `lib/lectures/Stanford CME296 Diffusion & Large Vision Models/transcripts-and-files/lecture 1 - transcript.md`
- `lib/lectures/Stanford CME296 Diffusion & Large Vision Models/transcripts-and-files/lecture 1 - slides.pdf`
- `lib/quiz.ts`
- `tests/lib/cme296Lecture1Questions.spec.ts`
- `docs/product-scope.md`
- `docs/team-preferences.md`
- `PLANS.md`

## Verification

- `npm run test:focused -- tests/lib/cme296Lecture1Questions.spec.ts tests/lib/question-registration.spec.ts`
- `$env:QUESTION_GUESSABILITY_SOURCE_IDS="cme296-lect1"; npm run test:question-guessability`
- `make types-check`
- `make format-check`
- `make check`

---

# CME296 DiffusionGemma-Focused Question Sets

## Goal

Create and register CME296 question sets only for the lecture sections selected in the DiffusionGemma study guide, with each set sized in proportion to the selected slide coverage under the 60-question full-lecture budget.

## Non-goals

- Do not create question sets for Lectures 4, 5, or 7, which the study guide marked as low-priority for this paper.
- Do not add questions from slide ranges outside the study guide selections.
- Do not change quiz UI behavior or shared question infrastructure.

## Steps

- [x] Audit the selected slide ranges and the existing Lecture 1 bank.
- [x] Narrow Lecture 1 and author proportional sets for Lectures 2, 3, 6, and 8.
- [x] Register the new sources and update focused tests and durable documentation.
- [x] Run question quality, registration, type, formatting, and full project checks.

## Files to touch

- `lib/lectures/Stanford CME296 Diffusion & Large Vision Models/lecture1_diffusion.ts`
- `lib/lectures/Stanford CME296 Diffusion & Large Vision Models/lecture2_score_matching.ts`
- `lib/lectures/Stanford CME296 Diffusion & Large Vision Models/lecture3_flow_matching.ts`
- `lib/lectures/Stanford CME296 Diffusion & Large Vision Models/lecture6_model_training.ts`
- `lib/lectures/Stanford CME296 Diffusion & Large Vision Models/lecture8_text_diffusion.ts`
- `lib/quiz.ts`
- `tests/lib/cme296Lecture1Questions.spec.ts`
- `tests/lib/cme296FocusedQuestions.spec.ts`
- `docs/product-scope.md`
- `PLANS.md`

## Verification

- `npm run test:focused -- tests/lib/cme296Lecture1Questions.spec.ts tests/lib/cme296FocusedQuestions.spec.ts tests/lib/question-registration.spec.ts`
- `$env:QUESTION_GUESSABILITY_SOURCE_IDS="cme296-lect1,cme296-lect2,cme296-lect3,cme296-lect6,cme296-lect8"; npm run test:question-guessability`
- `make types-check`
- `make format-check`
- `make check`

---

# DiffusionGemma Paper-Club Presentation

## Goal

Create a registered, live-presentable DiffusionGemma website for a technical one-hour paper club, with an intuitive diffusion and flow-matching bridge for an audience already familiar with transformers.

## Non-goals

- Do not add a generic paper-summary page or alter shared quiz behavior.
- Do not reproduce the earlier memory-paper deck's visual design.
- Do not add runtime dependencies.

## Steps

- [x] Review the paper, its key figures, and the existing presentation route conventions.
- [x] Build the route, paper-specific visual system, interactive denoising canvas, keyboard controls, and print layout.
- [x] Register the page and document the behavior and audience assumptions.
- [x] Add focused tests and verify desktop, mobile, types, formatting, and interaction behavior.

## Files to touch

- `app/learn/stanford-cme296/diffusiongemma/presentation/page.tsx`
- `app/learn/stanford-cme296/page.tsx`
- `app/learn/standaloneLearningPages.ts`
- `app/learn/LearningCoursePage.tsx`
- `app/learn/page.tsx`
- `components/learning/pages/DiffusionGemmaPresentationPage.tsx`
- `components/learning/pages/DiffusionGemmaPresentationPage.module.css`
- `public/learning/diffusiongemma/presentation/*`
- `e2e/diffusiongemma-presentation.spec.ts`
- `docs/product-scope.md`
- `docs/team-preferences.md`
- `PLANS.md`

## Verification

- `npx prettier --check` on touched source and documentation files
- `make types-check`
- `npm run e2e -- e2e/diffusiongemma-presentation.spec.ts`
- Browser review at desktop and mobile widths

---

# DiffusionGemma Technical Report Question Set

## Goal

Create and register a 60-question practice set that teaches the DiffusionGemma technical report's discrete-diffusion formulation, architecture, sampler, training pipeline, inference tradeoffs, evaluations, adaptation workflow, practical advantages, and limitations.

## Non-goals

- Do not change quiz UI behavior or shared question infrastructure.
- Do not alter the existing CME296 lecture banks or paper-club presentation.
- Do not introduce new runtime dependencies.

## Steps

- [x] Inspect the complete 55-page report and existing CME296 question conventions.
- [x] Author 60 stable-ID, self-contained questions with balanced difficulty and answer patterns.
- [x] Register the paper source, add focused coverage, and update durable product documentation.
- [x] Run the manual quality gate plus focused, guessability, type, formatting, and full project checks.

## Files to touch

- `lib/lectures/Stanford CME296 Diffusion & Large Vision Models/DiffusionGemma Technical Report.ts`
- `lib/lectures/Stanford CME296 Diffusion & Large Vision Models/transcripts-and-files/DiffusionGemma Technical Report.pdf`
- `lib/quiz.ts`
- `tests/lib/diffusionGemmaTechnicalReportQuestions.spec.ts`
- `tests/lib/cme296FocusedQuestions.spec.ts`
- `docs/product-scope.md`
- `PLANS.md`

## Verification

- `npm run test:focused -- tests/lib/diffusionGemmaTechnicalReportQuestions.spec.ts tests/lib/cme296FocusedQuestions.spec.ts tests/lib/question-registration.spec.ts tests/lib/mit15773-answer-distribution.spec.ts`
- `$env:QUESTION_GUESSABILITY_SOURCE_IDS="cme296-diffusiongemma"; npm run test:question-guessability`
- `make types-check`
- `make format-check`
- `make check`
