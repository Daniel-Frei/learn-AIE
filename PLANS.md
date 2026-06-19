Goal: Review and improve CS224R Lecture 3 policy-gradient questions, with emphasis on math coverage and guessability.

Non-goals:

- Do not refactor shared quiz infrastructure.
- Do not modify unrelated lecture banks beyond formatting required by the repo gate.

Steps:

- [x] Inspect Lecture 3 question bank, registration, transcript, and slide deck.
- [x] Visually review math-heavy slide pages and compare against question coverage.
- [x] Run targeted `cs224r-lect3` guessability early.
- [x] Edit only concrete weak questions, preserving IDs for minor fixes and minting new IDs for semantic rewrites.
- [x] Update durable repo preference docs for `make format` before `make check`.
- [x] Verify with targeted tests, `make format`, and `make check`.

Files to touch:

- `lib/lectures/Stanford CS224R Deep Reinforcement Learning/lecture3_Policy Gradients.ts`
- `docs/team-preferences.md`

Verification:

- `$env:QUESTION_GUESSABILITY_SOURCE_IDS="cs224r-lect3"; npm run test:question-guessability`
- `npm run test:focused -- tests/lib/question-registration.spec.ts`
- `npm run test:focused -- tests/lib/mit15773-answer-distribution.spec.ts`
- `make format`
- `make check`
