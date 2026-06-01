# Biology & Chemistry Basic-Concept Question Expansion

## Goal

Add 20 basic-concept practice questions to each Lecture 1-5 question set in `lib/other/Crash Courses/Biology & Chemistry for Life Science` so students can practice foundational knowledge from the lecture notes, not only higher-level synthesis.

## Non-Goals

- Do not add new question sets or change registration.
- Do not edit Lecture 0.
- Do not refactor quiz infrastructure.

## Steps

- [x] Read the Lecture 1-5 descriptions in `transcripts-and-files`.
- [x] Inspect existing question ID and authoring patterns.
- [x] Add questions 31-50 to each Lecture 1-5 file.
- [x] Run targeted guessability checks for the edited source IDs.
- [x] Run focused registration/type/format verification.

## Files To Touch

- `lib/other/Crash Courses/Biology & Chemistry for Life Science/Lecture 1 - Chemistry of Life.ts`
- `lib/other/Crash Courses/Biology & Chemistry for Life Science/Lecture 2 - Cells as Information-Processing Systems.ts`
- `lib/other/Crash Courses/Biology & Chemistry for Life Science/Lecture 3 - Genetics, Proteins, and Biological Regulation.ts`
- `lib/other/Crash Courses/Biology & Chemistry for Life Science/Lecture 4 - Physiology, Disease, and Pharmacology.ts`
- `lib/other/Crash Courses/Biology & Chemistry for Life Science/Lecture 5 - Biomedical Systems, Biotechnology, and Evidence.ts`

## Verification

- `QUESTION_GUESSABILITY_SOURCE_IDS=bio-chem-life-l1,bio-chem-life-l2,bio-chem-life-l3,bio-chem-life-l4,bio-chem-life-l5 npm run test:question-guessability`
- `npm run test:focused -- tests/lib/question-registration.spec.ts`
- `make types-check`
- `make format-check`

# Biology & Chemistry Lecture 0 Preparation Rewrite

## Goal

Rework `Lecture 0 - preparation.ts` into a true no-prior-knowledge preparation set for the Biology & Chemistry crash course, covering the most basic concepts assumed by Lectures 1-5.

## Non-Goals

- Do not change registration.
- Do not edit Lecture 1-5 content in this pass.
- Do not refactor quiz infrastructure.

## Steps

- [x] Run the targeted guessability check before edits.
- [x] Inspect Lecture 0 structure and Lecture 1-5 notes.
- [x] Replace at least 20 advanced questions with foundational preparation questions.
- [x] Run targeted guessability check for `bio-chem-life-l0`.
- [x] Run focused registration/type/format verification.

## Files To Touch

- `lib/other/Crash Courses/Biology & Chemistry for Life Science/Lecture 0 - preparation.ts`

## Verification

- `QUESTION_GUESSABILITY_SOURCE_IDS=bio-chem-life-l0 npm run test:question-guessability`
- `npm run test:focused -- tests/lib/question-registration.spec.ts`
- `make types-check`
- `make format-check`
