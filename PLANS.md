## Weighted Question Elo + Timer

### Goal

- Keep the Glicko-2 engine, but expose raw question rating as the visible difficulty label.
- Add a per-question timer capped at 3 minutes.
- Weight answer rating updates by response time and by the number of mistaken selections.

### Non-goals

- No auth/login changes.
- No change to the question bank content or selection rules.
- No switch away from Glicko-2 as the underlying rating model.

### Steps

- [x] Update the rating engine to apply time and mistake weighting symmetrically.
- [x] Extend the answer API, shared store, and Supabase schema with timing metadata.
- [x] Wire the quiz UI to show raw question Elo and a live/frozen timer per question.
- [x] Update docs to describe the new scoring behavior and contract.
- [x] Add unit and route coverage for weighted updates and persisted metadata.
- [ ] Run the full repo checks and fix any regressions.

### Files to touch

- `lib/difficultyStore.ts`
- `lib/useQuiz.ts`
- `lib/quizSync.ts`
- `lib/server/quizDataStore.ts`
- `lib/server/quizDataService.ts`
- `app/api/answers/route.ts`
- `app/page.tsx`
- `components/QuizQuestionSection.tsx`
- `docs/api-contract.md`
- `docs/product-scope.md`
- `store/manual/supabase-shared-quiz-schema.sql`
- `tests/lib/difficultyStore.spec.ts`
- `tests/app/api/quiz-data.routes.spec.ts`
- `e2e/smoke.spec.ts`

### Verification

- `make format-check`
- `make lint`
- `make types-check`
- `make test`
- `npm run e2e`

## Expo Mobile Quiz App

### Goal

- Add an Expo React Native app under `apps/mobile` for on-the-road quiz practice.
- Match the web app's core practice flow while keeping shared ratings/reports API-backed.

### Non-goals

- No mobile file import/export for ratings or reports in the first version.
- No queued offline sync for failed answer/report/explanation requests.
- No auth/login changes.

### Steps

- [x] Scaffold the Expo SDK 55 workspace app.
- [x] Extract platform-neutral quiz session helpers from the web hook.
- [x] Build the mobile quiz screen, storage, and API client.
- [x] Add focused helper tests and mobile type/lint commands.
- [x] Update docs for mobile scope, commands, and API base URL.
- [x] Run the verification gate and fix regressions.

### Files to touch

- `apps/mobile/**`
- `lib/useQuiz.ts`
- `lib/quizSession.ts`
- `package.json`
- `Makefile`
- `.env.template`
- `docs/product-scope.md`
- `docs/api-contract.md`
- `docs/team-preferences.md`
- `tests/lib/quizSession.spec.ts`

### Verification

- `make format-check`
- `make lint`
- `make types-check`
- `make test`
- `npm run mobile:lint`
- `npm run mobile:types-check`
- `make check`

## Mobile Profiles + Offline Sync

### Goal

- Make mobile practice independent of a developer PC or home-network API server.
- Use Supabase Auth profiles for durable participant identity.
- Persist ratings and queued writes locally so answer/report behavior works offline and syncs later.

### Non-goals

- No remote question-bank content sync in this step; questions remain bundled in the app.
- No social/profile discovery features.
- No public deployment workflow changes.

### Steps

- [x] Add Supabase Auth/RLS migration for profile-linked mobile participants.
- [x] Add mobile Supabase client, local state store, and sync queue.
- [x] Update mobile quiz hook/UI to support sign-in, local rating updates, queued reports, and sync.
- [x] Document mobile auth/env behavior and Supabase setup.
- [x] Add focused tests for mobile-safe sync helpers where practical.
- [x] Run the verification gate and browser render check.

### Files to touch

- `apps/mobile/src/**`
- `supabase/migrations/**`
- `.env.template`
- `docs/product-scope.md`
- `docs/api-contract.md`
- `docs/team-preferences.md`
- `PLANS.md`

### Verification

- `make check`
- Expo web render check on `http://localhost:8082`
