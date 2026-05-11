## Public GitHub Publish Readiness

### Goal

- Make the repository safe to publish publicly by removing local artifacts, tightening public API behavior, and resolving known dependency advisories.

### Non-goals

- No authentication redesign for the anonymous web quiz flow.
- No new runtime dependencies.
- No broad UI refactor.

### Steps

- [x] Remove tracked local/generated artifacts and extend ignore rules.
- [x] Document public-repo safety expectations and deployment secrets.
- [x] Protect sensitive report export behavior and bound LLM request payloads.
- [x] Update vulnerable package versions or safe transitive overrides.
- [x] Add/update focused route tests.
- [x] Run `make check` and `npm audit`.

### Files to touch

- `.gitignore`
- `README.md`
- `docs/*`
- `app/api/*`
- `lib/server/*`
- `tests/app/api/*`
- `package.json`
- `package-lock.json`

### Verification

- `make check`
- `npm audit --omit=dev --audit-level=moderate`
- `npm audit --audit-level=moderate`
