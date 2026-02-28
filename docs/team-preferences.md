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
