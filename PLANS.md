# CME295 Lecture 3 Question-Set Improvement

## Goal

Replace the repetitive, low-diagnosticity Stanford CME295 Lecture 3 quiz with a smaller, slide-weighted set that tests application, comparison, calculation, and mechanism understanding.

## Non-goals

- Do not change quiz UI behavior or the Lecture 3 learning page.
- Do not refactor shared question or registry infrastructure.
- Do not alter other lecture question banks.

## Steps

- [x] Audit all 80 current questions and inspect all 125 lecture slides.
- [x] Define a proportional coverage blueprint centered on the deck's worked sequences.
- [x] Replace substantially rewritten questions with new stable IDs and balanced difficulty/answer patterns.
- [x] Add focused tests and update durable documentation.
- [x] Run registration, type, guessability, formatting, and full project checks as appropriate.

## Files to touch

- `lib/lectures/Stanford CME295 Transformers & LLMs/lecture3_LLMs.ts`
- `lib/lectures/Stanford CME295 Transformers & LLMs/transcripts-and-files/lecture 3 - curriculum.md`
- `tests/lib/cme295Lecture3Questions.spec.ts`
- `docs/product-scope.md`
- `docs/team-preferences.md`
- `PLANS.md`

## Verification

- `npm run test:focused -- tests/lib/cme295Lecture3Questions.spec.ts tests/lib/question-registration.spec.ts`
- `$env:QUESTION_GUESSABILITY_SOURCE_IDS="cme295-lect3"; npm run test:question-guessability`
- `make types-check`
- `make format-check`
- `make check`

---

# CME296 Lecture 1 Question Set

## Goal

Store the matching Stanford CME296 Lecture 1 transcript and slides in the established lecture-source structure, then add a registered 40-question set covering diffusion, DDPM, and DDIM.

## Non-goals

- Do not change quiz UI behavior or shared question infrastructure.
- Do not modify the existing CME295 Lecture 1 source files.
- Do not add a learning-experience page for this lecture.

## Steps

- [x] Inspect the supplied transcript and slides and resolve the source mismatch.
- [x] Store the matching source bundle under a new Stanford CME296 course folder.
- [x] Author 40 stable-ID questions with balanced difficulty and correct-answer counts.
- [x] Register the source and update focused tests and durable documentation.
- [x] Run focused, guessability, type, formatting, and full project checks.

## Files to touch

- `lib/lectures/Stanford CME296 Diffusion & Large Vision Models/lecture1_diffusion.ts`
- `lib/lectures/Stanford CME296 Diffusion & Large Vision Models/transcripts-and-files/lecture 1 - transcript.md`
- `lib/lectures/Stanford CME296 Diffusion & Large Vision Models/transcripts-and-files/lecture 1 - slides.pdf`
- `lib/quiz.ts`
- `tests/lib/cme296Lecture1Questions.spec.ts`
- `docs/product-scope.md`
- `docs/team-preferences.md`
- `PLANS.md`

## Verification

- `npm run test:focused -- tests/lib/cme296Lecture1Questions.spec.ts tests/lib/question-registration.spec.ts`
- `$env:QUESTION_GUESSABILITY_SOURCE_IDS="cme296-lect1"; npm run test:question-guessability`
- `make types-check`
- `make format-check`
- `make check`
