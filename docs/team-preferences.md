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
- For question reporting, prefer append-only shared report entries and include source/prompt snapshot context in exported files.
- For shared quiz data, use anonymous per-device participants in v1: question difficulty is global, but each participant keeps their own rating/climb state.

## Shared Data Operations

- For the initial legacy-to-Supabase backfill, treat `store/manual/quiz-ratings(2).json` as the latest legacy rating export snapshot.

## Question Bank Preferences

- For question-bank files under `/lib` (excluding `/lib/llm`), keep answer patterns roughly balanced across 1, 2, 3, and 4 correct-answer questions.
- Exact quarter splits are preferred when practical, but approximate balance is acceptable when a file size or authoring constraints make exact `25%` buckets awkward.
- For automated validation, treat a file as acceptably balanced when each answer-count bucket stays within about `10%` of the file’s total question count from the ideal quarter split.
- When rebalancing question banks, prefer minimal statement edits and corresponding `isCorrect` updates rather than rewriting whole questions.
- In the filter menu, series/book checkboxes summarize the underlying individual question sets: selecting a series selects all of its question sets, and selecting any individual question set marks its parent series without expanding to sibling sets.
- For quiz prompts and explanations, prefer self-contained wording that does not assume the user attended the lecture.
