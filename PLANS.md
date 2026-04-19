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
