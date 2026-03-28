# Plan: Baseline Project Scaffolding from AGENTS.md

## Goal

- Add the missing baseline project scaffolding described in `AGENTS.md`:
  - `docs/*` as source-of-truth and API reference.
  - Vitest unit/integration testing under `tests/`.
  - Playwright E2E setup under `e2e/`.
  - Canonical `Makefile` commands.
  - Matching npm scripts/config.

## Non-goals

- No app feature refactors.
- No API behavior changes.
- No CI workflow in this task.

## Steps

- [x] Add baseline docs in `docs/` (product scope, API contract, team preferences).
- [x] Add Vitest config and at least one baseline test in `tests/`.
- [x] Add Playwright config and a smoke test in `e2e/`.
- [x] Add/align npm scripts and dev dependencies required for test commands.
- [x] Add `Makefile` commands from `AGENTS.md`.
- [x] Run verification commands and capture blockers found in current environment.

## Files To Touch

- `PLANS.md`
- `docs/*`
- `tests/*`
- `e2e/*`
- `package.json`
- `Makefile`
- `tsconfig.json` (only if needed)
- supporting config files for Vitest/Playwright

## Verification

- `make format-check`
- `make lint`
- `make types-check`
- `make test`
- `npm run e2e` (smoke test)
- `make check`

## Verification Result

- `make lint`: runs, but existing warnings remain in `lib/llm/explain.ts`.
- `make types-check`: fails on pre-existing TypeScript errors in `components/MathText.tsx` and `lib/difficultyStore.ts`.
- `make format-check`, `make test`, `npm run e2e`, `make check`: blocked because `prettier`, `vitest`, and `playwright` are not installed in this offline (`only-if-cached`) environment.

---

# Plan: Make Tests Pass + Expand Coverage

## Goal

- Make verification commands pass in the current environment.
- Add meaningful tests for core and critical app behavior beyond the initial baseline.

## Steps

- [x] Run full checks and collect failures in order (`format`, `lint`, `types-check`, `test`, `e2e`).
- [x] Fix TypeScript/lint issues currently breaking `make check`.
- [x] Add unit tests for core quiz selection/title behavior.
- [x] Add unit tests for difficulty state/rating logic.
- [x] Add API route tests for payload validation and server-error handling.
- [x] Re-run commands until green and record exact outcomes.

## Verification Result

- `make check`: pass.
- `npm run e2e`: pass when run outside sandbox (inside sandbox, Playwright web server spawn failed with `EPERM`).

---

# Plan: Selection UX + Climb Mode

## Goal

- Improve selection UX and add rating-aware climb mode.

## Steps

- [x] Start with all filters unselected.
- [x] Change source/topic filtering semantics to OR.
- [x] Group sources by series/book with collapsible lecture/chapter lists.
- [x] Add direct series/book selection.
- [x] Show user Glicko rating and RD in the header.
- [x] Add a climb mode that biases question selection near user rating with randomness.
- [x] Add/update tests and verify with `make check`.

---

# Plan: Topic-Based Filtering

## Goal

- Allow users to filter/select quiz questions by topic (`RL`, `DL`, `NLP`, `Math`) in addition to source and difficulty.

## Steps

- [x] Add topic metadata for each source in `lib/quiz.ts`.
- [x] Add topic-aware question filtering helpers in `lib/quiz.ts`.
- [x] Wire topic selection state into `lib/useQuiz.ts`.
- [x] Add topic UI controls to `components/QuizHeader.tsx` and pass state from `app/page.tsx`.
- [x] Add/update tests for topic filtering behavior.
- [x] Update docs and run verification.

---

# Plan: Rebalance MIT 15.773 Answer Patterns

## Goal

- Rebalance the five `lib/lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/*` question banks so each 40-question file has an even answer-pattern split:
  - 10 questions with 4 correct answers
  - 10 questions with 3 correct answers
  - 10 questions with 2 correct answers
  - 10 questions with 1 correct answer

## Non-goals

- No source registration changes.
- No UI or quiz engine behavior changes beyond validating the content constraint.

## Steps

- [x] Document the MIT 15.773 balancing rule in `docs/*`.
- [x] Make minimal statement edits in the five MIT 15.773 lecture files and update `isCorrect` flags accordingly.
- [x] Add a regression test for the required per-file distribution.
- [x] Run targeted verification and record the result.

## Files To Touch

- `PLANS.md`
- `docs/team-preferences.md`
- `lib/lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/*`
- `tests/lib/*`

## Verification

- `npm run test -- tests/lib/mit15773-answer-distribution.spec.ts`
- `make test`
- `make check`

## Verification Result

- `npm run test -- tests/lib/mit15773-answer-distribution.spec.ts`: pass.
- `make test`: pass.
- `make check`: pass.

---

# Plan: Rebalance MIT 6.S191 2025 + Crash Course Linear Algebra

## Goal

- Rebalance the question banks in:
  - `lib/lectures/MIT 6.S191 Deep Learning 2025/*`
  - `lib/other/Crash Course Linear Algebra/*`
- Bring those files within the current approximate answer-pattern tolerance used by `tests/lib/mit15773-answer-distribution.spec.ts`.

## Non-goals

- Do not rebalance the other already-flagged legacy banks in this task.
- Do not change quiz engine logic or source registration.

## Steps

- [x] Adjust the targeted MIT 6.S191 2025 files with minimal statement edits and `isCorrect` updates.
- [x] Adjust the Crash Course Linear Algebra file with minimal statement edits and `isCorrect` updates.
- [x] Re-run the generalized answer-distribution test and confirm only out-of-scope files remain.

## Verification

- `npm run test -- tests/lib/mit15773-answer-distribution.spec.ts`

## Verification Result

- `npm run test -- tests/lib/mit15773-answer-distribution.spec.ts`: fails only on out-of-scope legacy banks in `lib/chapter1.ts`, `lib/chapter2.ts`, `lib/lectures/Other RL/introduction to Reinforcement Learning.ts`, `lib/lectures/Stanford CME295 Transformers & LLMs/lecture2_models.ts`, and `lib/lectures/Stanford CME295 Transformers & LLMs/lecture4_training.ts`.

---

# Plan: Local Question Reporting + Export

## Goal

- Add client-side question reporting with local persistence and exportable JSON for manual review.

## Steps

- [x] Add a local report store module with versioned export helpers.
- [x] Wire report state/actions into the quiz hook and UI.
- [x] Add docs for the new behavior and export contract.
- [x] Add Vitest and Playwright coverage.
- [x] Run `make test`, `npm run e2e`, and `make check`.

## Verification Result

- `make format-check`: pass.
- `make lint`: pass.
- `make types-check`: pass.
- `npm run test -- tests/lib/questionReportsStore.spec.ts`: pass.
- `npm run e2e`: pass.
- `make test`: still fails on the pre-existing generalized answer-distribution regression in unrelated legacy banks:
  - `lib/chapter1.ts`
  - `lib/chapter2.ts`
  - `lib/lectures/Other RL/introduction to Reinforcement Learning.ts`
  - `lib/lectures/Stanford CME295 Transformers & LLMs/lecture2_models.ts`
  - `lib/lectures/Stanford CME295 Transformers & LLMs/lecture4_training.ts`
- `make check`: blocked by the same existing `make test` failure.

---

# Plan: Fix Legacy Answer-Pattern Regressions

## Goal

- Rebalance the remaining legacy question-bank files currently failing the generalized answer-pattern test so `make test` and `make check` can pass.

## Steps

- [x] Adjust the flagged question-bank files with minimal option-text and `isCorrect` edits.
- [x] Re-run `make test` and verify the generalized answer-pattern test passes.
- [x] Run `make check` and fix any remaining issues until the full gate passes.

## Verification Result

- `make test`: pass.
- `make check`: pass.

---

# Plan: Shared Quiz Data With Anonymous Participants

## Goal

- Replace browser-local answer/report persistence with shared Supabase Postgres storage while keeping anonymous per-device participants.

## Steps

- [x] Extract/share the rating engine so both client and server routes can use the same Glicko logic.
- [x] Add Next.js API routes for shared quiz-state bootstrap, answer writes, report writes/export, and one-time legacy migration.
- [x] Add a Supabase-backed storage layer plus a checked-in SQL schema for the required tables.
- [x] Switch the quiz hook from local persistence to API-backed state, keeping only `participantId` and migration markers in browser storage.
- [x] Update docs and add route/unit coverage for the new shared-storage behavior.
- [x] Run `make test` and `make check`, then record exact outcomes.

## Verification Result

- `make test`: pass.
- `make check`: pass.
