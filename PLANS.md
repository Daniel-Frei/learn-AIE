# Lecture 8 CS224R Question Bank

## Goal

Author and register 35 source-grounded questions for Stanford CS224R Lecture 8, emphasizing the slide math for conservative offline RL, reward classifiers, preference-reward likelihoods, RLHF/RLAIF, and self-proposed goals.

## Non-goals

- Do not change existing CS224R Lecture 1-7 question IDs or scoring behavior.
- Do not add new learning pages or quiz UI features.

## Steps

- [x] Inspect the Lecture 8 source bundle and visually check the math-heavy PDF pages.
- [x] Create the Lecture 8 TypeScript question file with 35 hardcoded IDs.
- [x] Register the source in `lib/quiz.ts` and update product-scope docs.
- [x] Run the author-questions quality gate, targeted guessability, and focused registration/type checks.
- [x] Run `make format` before `make check`.

## Files To Touch

- `lib/lectures/Stanford CS224R Deep Reinforcement Learning/lecture8_Conservative Offline RL and Reward Learning.ts`
- `lib/quiz.ts`
- `docs/product-scope.md`
- `eslint.config.mjs`
- `PLANS.md`

## Verification

- `QUESTION_GUESSABILITY_SOURCE_IDS=cs224r-lect8 npm run test:question-guessability`
- `npm run test:focused -- tests/lib/question-registration.spec.ts`
- `make types-check`
- `make format`
- `make check`
