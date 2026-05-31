# Team Preferences

This file captures durable process preferences so future tasks can follow them by default.

## Repo Conventions

- Source of truth for scope and behavior is `docs/*`.
- The public README should market the project to both learners and potential open-source contributors while staying grounded in documented product behavior.
- Keep changes small and reviewable; avoid large refactors unless requested.
- When behavior changes, add or update tests.
- This repository is intended to be public on GitHub. Do not commit real credentials, local `.env*` files, generated server logs, participant identifiers, or other machine/user-specific data.
- Elenthos skills are installed as a Git submodule at `.codex/skills/elenthos`; use the `skill-*` Makefile targets to update or publish them.
- Formatting should not rewrite local agent state or skill submodules under `.codex`; keep those paths out of app formatting scope.

## Project Structure

- App code: `/app`
- Mobile app code: `/apps/mobile` (Expo React Native).
- Vitest tests: `/tests`
- Playwright tests: `/e2e`

## Canonical Commands

- Install (clean): `make ci`
- Install (dev): `make install`
- Local Next.js dev, production-preview, and Playwright web-server default port: `43191` instead of common framework ports to reduce localhost conflicts.
- It is good to start a local dev server for implementation testing, browser verification, and debugging, but agents should shut down any dev server they started before finishing the task.
- Full checks: `make check`
- Format check: `make format-check`
- Lint: `make lint`
- Type check: `make types-check`
- Unit tests: `make test`
- Focused Vitest runs: `npm run test:focused -- path/to/file.spec.ts` to avoid misleading global coverage failures when running only one test file.
- E2E tests: `npm run e2e` or `npm run e2e:ui`
- Windows Start Menu launcher install/uninstall: `make install-windows-start-menu` / `make uninstall-windows-start-menu`
- Mobile dev server: `npm run mobile:start`
- Mobile Android/iOS/web launchers: `npm run mobile:android`, `npm run mobile:ios`, `npm run mobile:web`
- Mobile checks: `npm run mobile:lint`, `npm run mobile:types-check`

## Dependency Policy

- Ask before adding runtime dependencies under `dependencies`.
- Dev dependencies are acceptable when needed for tooling/testing.
- Current approved shared-storage runtime dependency: `@supabase/supabase-js`.
- Current approved mobile runtime stack: Expo SDK 55 template dependencies plus `@react-native-async-storage/async-storage` for anonymous participant persistence.
- Mobile profile sync should use Supabase Auth with RLS and the publishable/anon key, not the service-role key or a developer machine on the LAN.
- Mobile should remain local-first: answer/rating/report changes are saved on-device first and synced when Supabase is reachable.

## Testing Priorities

- Keep expanding coverage for core functionality, not just smoke tests.
- The unit-test command enforces at least 95% statements, branches, functions, and lines for the configured core logic/API coverage scope.
- Prioritize tests for quiz source selection/title logic, difficulty rating behavior, and API validation/error handling.
- For question reporting, prefer append-only shared database entries and include source/prompt snapshot context for reviewer triage.
- When adding a quiz topic, update the central `lib/questionTopics.ts` list and the API/product docs. Do not add topic-specific Supabase `question_reports.topic` enum constraints; reports should accept future configured topics without another report-table schema migration.
- Do not revive browser-local report storage or report export/import UI; legacy local reports should be ignored.
- For shared quiz data, use anonymous per-device participants in v1: question difficulty is global, but each participant keeps their own rating/climb state.

## UI Preferences

- During active quiz practice, prioritize the question prompt and answer options over the quiz-set title; keep the title smaller and visually muted.

## Shared Data Operations

- For the initial legacy-to-Supabase backfill, treat `store/manual/quiz-ratings(2).json` as the latest legacy rating export snapshot.

## Question Bank Preferences

- For question-bank files under `/lib` (excluding `/lib/llm`), keep answer patterns roughly balanced across 1, 2, 3, and 4 correct-answer questions.
- Exact quarter splits are preferred when practical, but approximate balance is acceptable when a file size or authoring constraints make exact `25%` buckets awkward.
- For question-bank files under `/lib` (excluding `/lib/llm`), do not require an equal static difficulty-label split unless the user, source material, existing set convention, or local docs ask for one. When no difficulty split is specified, use a roughly balanced `easy`/`medium`/`hard` mix by default where practical, and always report the final difficulty counts back to the user.
- For automated validation, treat a file as acceptably balanced when each answer-count bucket stays within about `10%` of the file’s total question count from the ideal quarter split.
- When rebalancing question banks, prefer minimal statement edits and corresponding `isCorrect` updates rather than rewriting whole questions.
- In the filter menu, series/book checkboxes summarize the underlying individual question sets: selecting a series selects all of its question sets, and selecting any individual question set marks its parent series without expanding to sibling sets.
- For quiz prompts and explanations, prefer self-contained wording that does not assume the user attended the lecture.
- For question-bank files under `/lib`, explanations should be more than `150` characters; if validation finds a shorter explanation, expand it to at least `250` characters.
- Questions should test real concept understanding rather than surface recall or answer elimination. Difficulty should reflect the knowledge, reasoning, transfer, or math required, while answer options should remain high-quality and similarly plausible.
- For source-material-derived question sets, practicing all questions to mastery should teach roughly the same core conceptual knowledge as reading and fully understanding the source material.
- Avoid low-quality distractors that can be ruled out by wording cues, extreme absolutes, category mismatch, or absurdity; wrong options should be plausible alternatives that diagnose common misunderstandings.
- After creating or editing registered question sets with `.codex/skills/author-questions`, run the targeted guessability test with `QUESTION_GUESSABILITY_SOURCE_IDS` set to the changed source id. If simple language-cue heuristics score well, improve the answer options instead of treating the test as a nuisance.
- Use the repo-local `.codex/skills/author-questions` skill to create or edit question sets from source-material folders; source material can be transcripts, chapters, papers, slides, PDFs, or other learning artifacts, and generated sets should be registered in `lib/quiz.ts` so they are selectable for practice.
- When students are struggling with a math-heavy question set, prefer adding applied A-level/AP-style computation questions before adding more abstract theory-only questions.
- Crash Course Linear Algebra Lecture 0 should stay focused on prerequisite AP/A-level linear algebra rather than AI applications, helping students practice the math needed for later lecture question sets.
- Crash Course Linear Algebra Lecture 0 should explicitly prepare students for recurring notation and terminology such as \(\mathbb{R}^n\), subscripts, transposes, summations, norms, matrix shapes, span, basis, rank, determinants, inverses, eigenvectors, and Singular Value Decomposition (SVD).
- Crash Course Linear Algebra Lecture 2-5 expansions should deepen mathematical rigor and applied linear algebra practice, with questions that test computation, shape reasoning, transformations, gradients, decompositions, attention, RL value functions, and conceptual transfer rather than surface recall.
- Biology & Chemistry for Life Science Lecture 0 should stay easier than L1-L5 and focus on prerequisite concepts and terminology that students need before attempting the lecture question sets.
- Biology & Chemistry for Life Science Lecture 0 should use plausible same-topic distractors rather than category-mismatch joke options, and its static difficulty labels should be split equally across easy, medium, and hard questions.
- Biology & Chemistry for Life Science question sets should be treated as part of the course material: practicing them to mastery should give students roughly the same core conceptual knowledge as attending and understanding the lectures.
- In the Biology & Chemistry for Life Science crash course, clinical trials and research evidence should be contextual validation material rather than the core capstone subject; use the separate Clinical Trials Crash Course for deeper trial-design coverage.
