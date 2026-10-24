# Reported Question Review — 2026-10-22

## Goal

Assess new open reports against current questions and source evidence, correct justified issues, and resolve reviewed reports with decision notes.

## Non-goals

- No unrelated bank rewrites, UI refactors, dependencies, or deletion of reports.
- Do not modify existing resolved history or store raw reports in tracked files.

## Steps

- [x] Privately snapshot open reports and match current or retired question IDs.
- [x] Assess each report, make warranted corrections, and add focused coverage.
- [x] Run the authoring quality gate and project verification.
- [x] Resolve reviewed report IDs and verify retention and remaining open counts.

## Files

- Affected question banks under `lib/`, focused tests under `tests/lib/`.
- `PLANS.md`; ignored `.tmp/reported-question-review-20261022/` for private evidence.

## Verification

- Focused content tests and `tests/lib/question-registration.spec.ts`.
- Targeted `npm run test:question-guessability` for changed sources.
- `make format`, `make check`; browser verification if rendering issues are found.
- Read back resolved report IDs and confirm historical reports remain unchanged.

## Result

Reviewed seven open reports covering six current questions in Crash Course Probability Lectures 1 and 2. Clarified event-set definitions and sequential versus mutually exclusive events, removed an ambiguous uniform-probability conditional, supplied an explicit classification cost table, and expanded Bayes/odds explanations with each arithmetic step. The coin-event independence key was already correct; the report prompted a clearer explanation rather than a scoring change.

All six changed items passed the manual alignment and quality review: six minor corrections, zero substantial rewrites, zero additions, and two intentionally retained introductory items. All question IDs, answer flags, and difficulty labels were preserved. Edited-item difficulty is two easy, one medium, and three hard; each full bank remains 20 easy / 20 medium / 20 hard with its existing answer-count distribution.

Verification passed: 41 focused content/registration tests, targeted guessability checks for both sources, and `make check` including mobile validation and eight browser smoke tests. Seven reports were marked resolved with timestamps and decision notes; read-back confirmed all seven retained, all 91 earlier resolutions unchanged, no reports deleted, and zero open reports remaining. Corrections are local and have not been committed or pushed.

# CME295 2026 Lectures 1–4

## Goal

Add four separate 2026 quiz sources with exactly 50 source-grounded questions each, selectively adapting the existing CME295 banks across lecture boundaries.

## Non-goals

No changes to legacy questions/IDs, learning pages, quiz behavior, dependencies, or unrelated local edits.

## Steps

- [x] Inspect all four transcripts and render/review slide visuals; map topic coverage against all nine legacy banks.
- [x] Author four independent 50-question banks with new explicit IDs and record reuse/source coverage.
- [x] Register a distinct 2026 series; document scope/preferences and add focused content/identity coverage.
- [x] Review all 200 questions for source alignment, answer correctness, diagnosticity, and balance.
- [x] Run focused tests, targeted guessability, formatting, and `make check`.

## Files and verification

New banks: `lib/lectures/Stanford CME295 2026 Transformers & LLMs/`; registry: `lib/quiz.ts`; tests: `tests/lib/cme2952026Questions.spec.ts`; docs: `docs/product-scope.md`, `docs/team-preferences.md`, and a coverage/reuse note. Verification: focused content/registration tests, targeted question-guessability for all four source IDs, `make format`, `make check`. Temporary source renders/extracts stay ignored under `.tmp/cme295-2026/`.

## Result

Added and registered four independent 50-question 2026 sources, with paired slides/transcripts and a coverage/reuse report. All 200 questions passed manual scope, key-alignment, and diagnosticity review: 84 newly authored, 61 substantial adaptations, 9 minor adaptations, and 46 reused without content changes. Fifteen existing easy orientation questions were deliberately retained. Difficulty totals are 60 easy / 71 medium / 69 hard; Lecture 4 intentionally emphasizes harder derivations.

Verification passed: 39 focused tests, all four targeted guessability checks, 425 KaTeX renders, and the full make check gate (289 unit tests passed, 1 existing opt-in test skipped, web/mobile lint and typechecks, and 8 browser smoke tests). The ID-literal scanner was corrected to accept quoted and unquoted object keys before the focused rerun passed. Legacy CME295 files have no diff, unrelated pre-existing edits retained their hashes, and the browser smoke server exited. No commits or pushes were made.

# CME295 2026 quality review and canonical reuse

## Goal

Review all 200 entries again, improve weak questions, and replace the 46 unchanged copies with references to the original questions and IDs so existing rating history applies.

## Non-goals

No database rewrites or merging of stats for substantially different questions; no unrelated content/UI refactors; retain 50 memberships per 2026 lecture.

## Steps

- [x] Snapshot all current banks, perform a hidden-answer review, and record quality decisions.
- [x] Implement explicit canonical references, unique mixed pools, and selection-aware source metadata on web/mobile.
- [x] Revise weak questions with identity changes where substantial; validate answer keys and source coverage.
- [x] Add regression tests for references, deduplication, metadata, and old-ID rating use; update docs and review notes.
- [x] Run focused tests, targeted guessability, math rendering, formatting, and make check.

## Files and verification

Four 2026 banks and any narrowly needed canonical corrections; lib/quiz.ts; a question-reference helper; web/mobile hooks; focused tests; docs; PLANS.md. Verify original object/ID identity, one pool occurrence per ID, active-source reporting context, legacy rating lookup, 50 memberships each, question quality/balance, focused Vitest, targeted guessability, KaTeX, make format and make check.

## Result

Reviewed all 200 questions again and revised 91: 53 substantial replacements with fresh IDs and 38 minor corrections. Retained 109 without content changes, including deliberately easy orientation coverage; 53 orientation items remain overall. Final difficulty is 66 easy / 73 medium / 61 hard, with 50 questions in each lecture.

Converted exactly 46 memberships to explicit references to the original objects and IDs (0/28/13/5 by new lecture). Fourteen canonical items received minor corrections shared by both editions; their correct-answer semantics and IDs remain intact. Added unique mixed pools, conflicting-definition detection, selection-aware metadata on web/mobile, and tests for existing Glicko history and report snapshots. No database migration or rating-history merge was performed.

Verification passed: the full make check gate (299 unit tests passed, 1 opt-in test skipped, web/mobile lint and typechecks, 8 browser smoke tests), two dedicated canonical-source browser cases, targeted guessability across all nine changed source IDs, and 552 strict KaTeX renders. Early test failures were incorrect expectations for the existing correct counter increment and legacy tooltip wording; both expectations were corrected and the exact flows passed. The final hidden-answer pass also corrected a baseline distractor ambiguity and replaced an unrealistic token/byte example. The three unrelated pre-existing files retained their saved hashes. No commits or pushes were made.
