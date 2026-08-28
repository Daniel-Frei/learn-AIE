# Development Database Outage Visibility

## Goal

Recover any retained fallback practice data that can be proven from local
artifacts, and make sustained shared-database outages visible in the web quiz
during development without warning for brief connectivity interruptions.

## Non-goals

- Do not invent or replay answer attempts when no durable attempt payload exists.
- Do not show the database warning in production.
- Do not replace the existing short-lived in-memory fallback or add a new
  runtime dependency.

## Steps

- [x] Inspect Supabase, Firefox origin storage, and live processes for recoverable
      answer data.
- [x] Capture the current browser behavior with Supabase unreachable.
- [x] Make the fallback retry Supabase and expose its current outage duration.
- [x] Add a development-only warning after a 60-second grace period.
- [x] Add regression coverage, update persistence docs, and re-verify the browser
      flow and repository checks.

## Files to touch

- `lib/server/quizDataStore.ts`
- `app/api/persistence-health/route.ts`
- `lib/quizSync.ts`
- `lib/useQuiz.ts`
- `components/QuizPageClient.tsx`
- `tests/lib/quizDataStore-resilience.spec.ts`
- `tests/app/api/persistence-health.route.spec.ts`
- `e2e/smoke.spec.ts`
- `docs/product-scope.md`
- `docs/api-contract.md`
- `docs/team-preferences.md`
- `PLANS.md`

## Verification

- `npm run test:focused -- tests/lib/quizDataStore-resilience.spec.ts tests/app/api/persistence-health.route.spec.ts`
- `npm run e2e -- e2e/smoke.spec.ts`
- `make check`
- Browser evidence with an unreachable Supabase endpoint before and after the fix

## Verification result

- Focused resilience/health tests pass (13 tests), the full Vitest suite passes
  (254 passed, 1 skipped), and coverage remains above the configured 95% gates.
- TypeScript, ESLint, mobile lint/types, touched-file Prettier, and all 8 E2E
  smoke tests pass.
- Browser evidence confirms the warning appears for a real sustained outage and
  clears after database recovery without a page reload or client console error.
- Repository-wide `make check` stops at the existing Prettier baseline on 21
  unrelated files; all later gates were run independently and pass.

---

# Stanford CS109 Lectures 1-2 Question Sets

## Goal

Store the supplied Fall 2022 Stanford CS109 Lecture 1 and Lecture 2 source
bundles, then add two selectable 35-question practice sources focused on the
probability and counting content.

## Non-goals

- Do not create questions about course logistics, staff, assignments, grading,
  general AI history, or classroom anecdotes.
- Do not change quiz UI behavior or shared question infrastructure.
- Do not add later CS109 lectures or a learning-experience page in this pass.

## Steps

- [x] Inspect both PDFs, both transcripts, repo conventions, and the archived
      course identity.
- [x] Store the paired lecture PDFs and transcripts under a new CS109 source
      folder.
- [x] Author 35 stable-ID, self-contained questions per lecture with balanced
      difficulty and answer-count patterns.
- [x] Register the new `stanford-cs109` series and both Math-topic sources, add
      focused coverage, and update durable product documentation.
- [x] Run the manual quality gate plus registration, guessability, type,
      formatting, and full repository checks as appropriate.

## Files to touch

- `lib/lectures/Stanford CS109 Probability for Computer Scientists/lecture1_welcome_counting.ts`
- `lib/lectures/Stanford CS109 Probability for Computer Scientists/lecture2_combinatorics.ts`
- `lib/lectures/Stanford CS109 Probability for Computer Scientists/transcripts-and-files/*`
- `lib/quiz.ts`
- `tests/lib/cs109Questions.spec.ts`
- `docs/product-scope.md`
- `PLANS.md`

## Verification

- `npm run test:focused -- tests/lib/cs109Questions.spec.ts tests/lib/question-registration.spec.ts`
- `$env:QUESTION_GUESSABILITY_SOURCE_IDS="cs109-lect1,cs109-lect2"; npm run test:question-guessability`
- `make types-check`
- `make format-check`
- `make check`

## Verification result

- Focused CS109 and registration tests pass (13 tests), and targeted
  guessability checks pass for both source IDs.
- TypeScript, ESLint, 250 unit tests, mobile lint/types, and all 6 E2E smoke
  tests pass; scoped Prettier and `git diff --check` also pass.
- Repository-wide `make check` stops at the existing formatting baseline on 26
  files outside the new CS109 course and test files. The remaining gates pass
  when run independently.

---

# Crash Course Probability Question-Bank Rebuild

## Goal

Review and rebuild the registered Lecture 1-5 Probability question banks so
each 60-question set teaches the revised curriculum with stronger mathematical
fluency, applied reasoning, plausible distractors, and useful explanations.

## Non-goals

- Do not change the Lecture 0 prerequisite bank in this pass.
- Do not change quiz UI behavior, source registration IDs, or the 60-question
  size of any Lecture 1-5 bank.
- Do not refactor unrelated question banks or learning experiences.

## Steps

- [x] Audit the revised syllabus, source overviews, registered banks, stable-ID
      requirements, difficulty mix, answer patterns, and explanation quality.
- [x] Define a 60-question coverage blueprint for each of Lectures 1-5.
- [x] Replace low-diagnosticity and obsolete items, assigning new never-used IDs
      to substantial replacements while keeping every bank at 60 questions.
- [x] Add focused structural, coverage, difficulty, answer-pattern, and stable-ID
      tests for all five banks.
- [x] Update durable product memory and run registration, guessability, type,
      formatting, and full repository checks as appropriate.

## Files to touch

- `lib/other/Crash Courses/Probability/Lecture 1 - Probability as the Language of AI.ts`
- `lib/other/Crash Courses/Probability/Lecture 2 - Conditional Probability, Bayes, and Dependence.ts`
- `lib/other/Crash Courses/Probability/Lecture 3 - Likelihood, Loss, Softmax, and Deep Learning.ts`
- `lib/other/Crash Courses/Probability/Lecture 4 - Probability Over Time - Reinforcement Learning.ts`
- `lib/other/Crash Courses/Probability/Lecture 5 - Sampling, Latent Variables, and Diffusion Models.ts`
- `tests/lib/crashCourseProbabilityQuestions.spec.ts`
- `docs/product-scope.md`
- `docs/team-preferences.md`
- `PLANS.md`

## Verification

- `npm run test:focused -- tests/lib/crashCourseProbabilityQuestions.spec.ts tests/lib/question-registration.spec.ts tests/lib/mit15773-answer-distribution.spec.ts`
- `$env:QUESTION_GUESSABILITY_SOURCE_IDS="crash-probability-l1,crash-probability-l2,crash-probability-l3,crash-probability-l4,crash-probability-l5"; npm run test:question-guessability`
- `make types-check`
- `npx prettier --check` on touched files
- `make check`

---

# Crash Course Probability Learning-Experience Rebuild

## Goal

Create complete, extensive, quiz-linked learning experiences for Probability
L0-L5, adding the missing L0-L2 pages and substantially rebuilding L3-L5 around
the expanded curriculum and the supplied talk's intuition-first explanations.

## Non-goals

- Do not change the six registered question banks or their stable question IDs.
- Do not refactor unrelated learning experiences or the global app shell.
- Do not introduce new runtime dependencies; React, browser-native controls,
  KaTeX, CSS/SVG, and the existing Lucide icon set are sufficient.

## Experience brief

The six pages form one coherent **Probability Observatory**, but each station
uses a different source-native learning object.

| Source | Learner job                                                                           | Central object and primary interaction                                                                                                                 | Identity / avoided legacy pattern                                                                            |
| ------ | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| L0     | Decode and manipulate the prerequisite notation used later                            | A notation gym with a normalization bench, odds converter, shape checker, and geometric-sum dial                                                       | Graph-paper instrument desk; avoids an AI-flavored definition card list                                      |
| L1     | Build probability from outcomes and events before formulas                            | A ten-marble event universe that visibly becomes complements, intersections, unions, random-variable values, PMFs, PDFs, and expectations              | Physical probability worktable; follows the talk's unique-outcome intuition and avoids a glossary-first page |
| L2     | Update a probability when information changes the relevant universe                   | An evidence lens joining conditional filtering, joint tables, a two-bag path tree, and an interactive 1,000-person Bayes population                    | Investigation notebook; avoids presenting Bayes as a formula before base rates and natural frequencies       |
| L3     | Trace how raw model scores become a learnable training signal                         | A training microscope where logits, temperature, the observed label, likelihood, cross-entropy, entropy, calibration, and gradients update together    | Bright model-inspection console; avoids detached formula cards                                               |
| L4     | Choose actions by averaging stochastic futures and reasoning recursively              | A decision world combining a grid transition model, trajectory/value trace, exploration mixer, and first-step waiting-time equations                   | Map-and-timeline control room; avoids a shallow grid toggle disconnected from return and recursion           |
| L5     | Turn a learned distribution into tokens or images while preserving hidden uncertainty | A generation forge linking token filtering/sampling, sequence likelihood, latent marginalization/posteriors, Gaussian noising, denoising, and guidance | Split token/image production bench; avoids independent labs that do not share one generation story           |

## Steps

- [x] Audit the source curriculum, quiz registry, existing L3-L5 pages, routes,
      and tests.
- [x] Build shared course navigation, accessible visual language, and tested
      probability-math helpers.
- [x] Implement and register the new L0-L2 learning experiences.
- [x] Replace L3-L5 with extensive source-native experiences that preserve the
      useful mechanisms while changing the obsolete page architecture.
- [x] Update learning metadata, product memory, unit coverage, and Playwright
      journeys for all six routes and quiz transitions.
- [x] Run formatting, lint, types, unit tests, E2E, and browser UX review at
      desktop, 1280x800 laptop, and 390x844 mobile widths.

## Files to touch

- `components/learning/probability/ProbabilityCourse.tsx`
- `components/learning/probability/ProbabilityCourse.module.css`
- `components/learning/pages/CrashProbabilityL0LearningPage.tsx`
- `components/learning/pages/CrashProbabilityL1LearningPage.tsx`
- `components/learning/pages/CrashProbabilityL2LearningPage.tsx`
- `components/learning/pages/CrashProbabilityL3LearningPage.tsx`
- `components/learning/pages/CrashProbabilityL4LearningPage.tsx`
- `components/learning/pages/CrashProbabilityL5LearningPage.tsx`
- `lib/probabilityLearning.ts`
- `lib/learning.ts`
- `app/learn/LearningExperienceRoute.tsx`
- `tests/lib/probabilityLearning.spec.ts`
- `tests/lib/learning.spec.ts`
- `e2e/learning.spec.ts`
- `docs/product-scope.md`
- `docs/team-preferences.md`
- `PLANS.md`

## Verification

- `npm run test:focused -- tests/lib/probabilityLearning.spec.ts tests/lib/learning.spec.ts`
- focused Probability Playwright journeys and transitions in `e2e/learning.spec.ts`
- `npx prettier --check` on touched files
- `make lint`
- `make types-check`
- `make test`
- `npm run e2e`
- browser screenshots at 1440x900, 1280x800, and 390x844
- `make check`

## Verification result

- All touched files pass targeted Prettier checks; repository-wide
  `make check` still stops on 22 unrelated pre-existing formatting warnings.
- Lint, TypeScript, 230 unit tests, mobile lint/types, and the 6-test E2E smoke
  suite pass.
- All 8 Probability learning/quiz journeys pass, including every L0-L5 mobile
  route and active-station viewport check.
- The complete E2E run passed 71/72 tests. The sole remaining failure is an
  unrelated AI Agents presentation keyboard-navigation timing flake that passes
  when rerun alone.
- Live Playwright review at 1440x900, 1280x800, and 390x844 found no console or
  network errors after fixing formula-trail labels, hydration-safe inline
  values, and active-station centering.

---

# Probability Curriculum Lecture-Coverage Audit

## Goal

Audit the Crash Course Probability syllabus against the supplied probability
lecture and add every missing concept to the appropriate dependency-ordered
module.

## Non-goals

- Do not rewrite the registered quiz banks or learning pages in this pass.
- Do not add measure-theoretic probability, advanced causal inference, or
  calculus derivations beyond the supplied lecture's conceptual level.
- Do not refactor unrelated course or quiz infrastructure.

## Steps

- [x] Inventory the supplied lecture and the existing syllabus/lecture
      overviews.
- [x] Add missing foundations to Lecture 1, conditional decomposition to
      Lecture 2, and recursive expectation to Lecture 4.
- [x] Add a source-to-curriculum coverage map and update durable product memory.
- [x] Run Markdown formatting and repository verification appropriate for the
      documentation-only changes.

## Files to touch

- `lib/other/Crash Courses/Probability/transcripts-and-files/Syllabus.md`
- `lib/other/Crash Courses/Probability/transcripts-and-files/Lecture 1 - overview.md`
- `lib/other/Crash Courses/Probability/transcripts-and-files/Lecture 2 - overview.md`
- `lib/other/Crash Courses/Probability/transcripts-and-files/Lecture 4 - overview.md`
- `docs/product-scope.md`
- `docs/team-preferences.md`
- `PLANS.md`

## Verification

- `npx prettier --check` on touched Markdown files (passed)
- `git diff --check` (passed)
- `make check` (stopped at the baseline format gate on 22 untouched files)
- `make lint`, `make types-check`, `make test`, `npm run mobile:lint`,
  `npm run mobile:types-check`, and `npm run e2e:smoke` (passed)

---

# DiffusionGemma Conversion-Story Follow-up

## Goal

Rework Section 2 into a concise, stepwise explanation of how the trained Gemma
4 backbone becomes DiffusionGemma: reuse the checkpoint, change the canvas
attention mask, train all-position denoising, add lightweight
self-conditioning, and distill the sampler into the few-step regime.

## Non-goals

- Do not imply that DiffusionGemma adds a separately pretrained encoder.
- Do not reproduce every implementation detail from the supplied explanation.
- Do not change the other presentation sections, route registration, or quiz
  behavior.

## Steps

- [x] Replace the current Section 2 sequence with one shared-backbone mental
      model and a five-step conversion storyboard.
- [x] Preserve the paper's conversion and inference figures as visual anchors.
- [x] Update product memory and presentation E2E assertions.
- [x] Verify formatting, types, focused E2E behavior, and responsive browser
      layout.

## Files to touch

- `components/learning/pages/DiffusionGemmaPresentationPage.tsx`
- `components/learning/pages/DiffusionGemmaPresentationPage.module.css`
- `e2e/diffusiongemma-presentation.spec.ts`
- `docs/product-scope.md`
- `docs/team-preferences.md`
- `PLANS.md`

## Verification

- `npx prettier --check` on touched files
- `make lint`
- `make types-check`
- `npm run e2e -- e2e/diffusiongemma-presentation.spec.ts`
- Browser evidence at 1440x900, 1280x800, and 390x844

---

# DiffusionGemma Full-Chart and Architecture-Clarity Follow-up

## Goal

Use the complete supplied benchmark charts, make every deck image enlargable,
repair deck-wide content alignment, and explain precisely how Gemma 4's causal
Transformer weights, DiffusionGemma's two attention modes, SFT, and the runtime
generation diagram fit together.

## Non-goals

- Do not change route registration, quiz behavior, or shared navigation.
- Do not present the supplied benchmark exports as controlled comparisons with
  DiffusionGemma.
- Do not add a separate pretrained encoder or imply that SFT alone defines the
  runtime architecture.

## Steps

- [x] Re-read the architecture, generation, and SFT sections of the paper and
      inspect the official model card and implementation.
- [x] Replace cropped benchmark images with complete, clickable source exports
      and overlay the requested capability highlights.
- [x] Rework the agenda, section dividers, slide 12, and the Gemma-to-diffusion
      explanation across slides 14-19.
- [x] Normalize the content canvas so titles, visuals, and explanatory text use
      the same available width.
- [x] Update product memory and Playwright assertions.
- [x] Verify focused tests plus desktop, laptop, and mobile browser evidence.

## Files to touch

- `components/learning/pages/DiffusionGemmaPresentationPage.tsx`
- `components/learning/pages/DiffusionGemmaPresentationPage.module.css`
- `public/learning/diffusiongemma/presentation/*`
- `e2e/diffusiongemma-presentation.spec.ts`
- `docs/product-scope.md`
- `docs/team-preferences.md`
- `PLANS.md`

## Verification

- `npx prettier --check` on touched source, test, and documentation files
- `make types-check`
- `npm run e2e -- e2e/diffusiongemma-presentation.spec.ts`
- Browser evidence at 1440x900, 1280x800, and 390x844
- `make check`

---

# DiffusionGemma Video-Paced Follow-up

## Goal

Turn the revised deck into a faster-moving live/video presentation with a
speed-led opening, restored agenda and section dividers, readable external
benchmark references, more visual motivation, and an explicit explanation of
how a causal Gemma 4 checkpoint becomes DiffusionGemma.

## Non-goals

- Do not change route registration, quiz behavior, or shared navigation.
- Do not claim that the Artificial Analysis charts directly score
  DiffusionGemma when the supplied export does not include it.
- Do not add runtime dependencies or regenerate leaderboard data.

## Steps

- [x] Audit the live deck, attached Artificial Analysis exports, paper, and
      official model information.
- [x] Rebuild the intro around speed, add source-faithful benchmark crops, and
      restore the agenda.
- [x] Add six section dividers while removing repeated section labels and time
      estimates from content slides.
- [x] Split motivation into short visual messages and add the Gemma 4 to
      DiffusionGemma bridge, including the moved warm-start pipeline.
- [x] Update product memory and Playwright assertions for the 45-slide deck.
- [x] Verify focused tests plus desktop, laptop, and mobile browser evidence.

## Files to touch

- `components/learning/pages/DiffusionGemmaPresentationPage.tsx`
- `components/learning/pages/DiffusionGemmaPresentationPage.module.css`
- `public/learning/diffusiongemma/presentation/*`
- `e2e/diffusiongemma-presentation.spec.ts`
- `docs/product-scope.md`
- `docs/team-preferences.md`
- `PLANS.md`

## Verification

- `npx prettier --check` on touched source, test, and documentation files
- `make types-check`
- `npm run e2e -- e2e/diffusiongemma-presentation.spec.ts`
- Browser evidence at 1440x900, 1280x800, and 390x844
- `make check`

---

# DiffusionGemma Presentation Narrative Revision

## Goal

Rework the existing DiffusionGemma paper-club deck so it opens with a minimal,
credible hook, makes the inference problem and competing approaches explicit,
then gives a source-grounded model and architecture overview before the detailed
diffusion mechanism.

## Non-goals

- Do not change shared quiz behavior, route registration, or app navigation.
- Do not turn the talk into a generic paper summary or remove its interactive
  denoising and entropy-sampling explanations.
- Do not add runtime dependencies.

## Experience brief

- **Learner job:** explain why DiffusionGemma can move the speed/capability
  frontier, what bottleneck it attacks, and how its block-diffusion architecture
  works before following the training and evidence sections.
- **Central object:** one 256-token canvas moving through encode, denoise,
  commit, and append.
- **Primary interaction:** inspect that canvas as tokens and confidence change,
  then manipulate which positions the entropy-bounded sampler commits.
- **Visual identity:** paper figures plus precise, slide-native system diagrams
  that keep the same canvas visible across motivation, architecture, and
  mechanism.
- **Avoid:** a detached "intuition" chapter, a dense title composition, or a
  result hook that omits the quality and hardware conditions.

## Steps

- [x] Audit the live deck, complete paper, official model card, and local product
      guidance.
- [x] Replace the title composition and add a source-grounded speed-frontier
      hook before the existing thesis slide.
- [x] Make Motivation the first section and explain the costs and ceilings of AR,
      batching, speculative decoding, and MTP.
- [x] Add a high-level DiffusionGemma section covering model identity,
      architecture, generation loop, and the paper's three-part approach.
- [x] Reorder the existing intuition slides into the mechanism sequence and keep
      the interactive labs intact.
- [x] Update product memory, Playwright coverage, and slide-count/navigation
      assertions.
- [x] Verify desktop, laptop, and mobile layouts plus focused and full repository
      checks.

## Files to touch

- `components/learning/pages/DiffusionGemmaPresentationPage.tsx`
- `components/learning/pages/DiffusionGemmaPresentationPage.module.css`
- `public/learning/diffusiongemma/presentation/*`
- `e2e/diffusiongemma-presentation.spec.ts`
- `docs/product-scope.md`
- `docs/team-preferences.md`
- `PLANS.md`

## Verification

- `npx prettier --check` on touched source, test, and documentation files
- `make types-check`
- `npm run e2e -- e2e/diffusiongemma-presentation.spec.ts`
- Browser evidence at 1440x900, 1280x800, and 390x844
- `make check`

---

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
