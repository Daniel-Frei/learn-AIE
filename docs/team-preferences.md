# Team Preferences

This file captures durable process preferences so future tasks can follow them by default.

## Repo Conventions

- Source of truth for scope and behavior is `docs/*`.
- Keep changes small and reviewable; avoid large refactors unless requested.
- When behavior changes, add or update tests.

## Project Structure

- App code: `/app`
- Vitest tests: `/tests`
- Playwright tests: `/e2e`

## Canonical Commands

- Install (clean): `make ci`
- Install (dev): `make install`
- Full checks: `make check`
- Format check: `make format-check`
- Lint: `make lint`
- Type check: `make types-check`
- Unit tests: `make test`
- E2E tests: `npm run e2e` or `npm run e2e:ui`

## Dependency Policy

- Ask before adding runtime dependencies under `dependencies`.
- Dev dependencies are acceptable when needed for tooling/testing.

## Testing Priorities

- Keep expanding coverage for core functionality, not just smoke tests.
- Prioritize tests for quiz source selection/title logic, difficulty rating behavior, and API validation/error handling.
- For question reporting, prefer append-only local report entries and include source/prompt snapshot context in exported files.

## Question Bank Preferences

- For question-bank files under `/lib` (excluding `/lib/llm`), keep answer patterns roughly balanced across 1, 2, 3, and 4 correct-answer questions.
- Exact quarter splits are preferred when practical, but approximate balance is acceptable when a file size or authoring constraints make exact `25%` buckets awkward.
- For automated validation, treat a file as acceptably balanced when each answer-count bucket stays within about `10%` of the file’s total question count from the ideal quarter split.
- When rebalancing question banks, prefer minimal statement edits and corresponding `isCorrect` updates rather than rewriting whole questions.
