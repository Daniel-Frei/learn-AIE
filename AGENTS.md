## Source of truth

- **Product behavior & scope:** files in `docs/*`
- If implementation choices conflict with the `docs/*` files flag it and propose options.
- idea behind the `docs/*` documents is that a product manager can use them to interact with the development process without having to read the code and still get a high level understanding (and potentially make adjustments).
- If the user provides information (e.g. preferences, guidance about the implementation) that aren't documented yet, then add them.
- `docs/*` files contains .md files that act as a memory for user preferences and information is useful across many (and future) sessions.

## What this repo is

- TS and Next.js frontend
- Maintain an API documentation in `docs/*` that can be used by backend developers.

## Operating mode

- Keep changes small and reviewable. Avoid refactors unless explicitly requested.
- When behavior changes, add/update tests. Prefer adding coverage over “trust me” fixes.
- Never commit secrets (keys/tokens). If you suspect exposure, stop and report it.
- If requirements are unclear, propose 1–2 options + tradeoffs, then proceed.

## Where things live

- App code: /app
- Unit/integration tests (Vitest): /tests
- E2E tests (Playwright): /e2e
- Project config to consult:
  - package.json (scripts, packages)
  - Makefile (canonical commands)
  - tsconfig.json

## Commands (use these; don’t invent alternatives)

### Install

- Clean install (preferred): `make ci` (npm ci)
- Dev install: `make install` (npm install)

### Verify (run as appropriate; prefer the full gate before finishing)

- Full gate (preferred): `make check`
- Format check: `make format-check`
- Lint: `make lint`
- Typecheck: `make types-check`
- Unit tests: `make test`
- E2E: `npm run e2e` (or `npm run e2e:ui` locally)

If any command fails, include the exact error output and fix the root cause.

## Testing expectations (important)

- For logic/components: add/adjust Vitest tests under /tests.
- For user flows/regressions: add/adjust Playwright tests under /e2e.
- Prefer testing observable behavior over internals.
- If you touch code with no existing tests, add a minimal baseline test.

## Dependency policy (safe default)

- Ask before adding ANY new entry under `"dependencies"` (runtime/production deps).
- DevDependencies are OK if truly necessary (tests/tooling), but still prefer minimal.
- If adding a dep, explain: why needed, alternatives considered, and why dep vs devDep.

## API/contract changes (what to do)

- If you change a function signature, exported type, widely-used component props,
  request/response shape, or env var name: call it out explicitly in your summary
  and update all usages + tests.

## Large tasks: write a plan first

- If the task touches many files or is multi-step, create/update `PLANS.md` with:
  Goal, non-goals, steps checklist, files to touch, and how you’ll verify (commands).
  Keep the plan short and check items off as you go.

## CI (not set up yet)

- If asked to add CI, create a minimal workflow that runs `make ci` + `make check`
  on PRs/pushes.
