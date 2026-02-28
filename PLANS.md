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
