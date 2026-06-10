# Team Preferences

This file captures durable process preferences so future tasks can follow them by default.

## Repo Conventions

- Source of truth for scope and behavior is `docs/*`.
- The public README should market the project to both learners and potential open-source contributors while staying grounded in documented product behavior.
- Keep changes small and reviewable; avoid large refactors unless requested.
- When behavior changes, add or update tests.
- This repository is intended to be public on GitHub. Do not commit real credentials, local `.env*` files, generated server logs, participant identifiers, or other machine/user-specific data.
- Elenthos skills are installed as a Git submodule at `.codex/skills/elenthos`; use the `skill-*` Makefile targets to update or publish them.
- Formatting should not rewrite local agent state or skill submodules under `.codex`; keep those paths out of app formatting scope.

## Project Structure

- App code: `/app`
- Mobile app code: `/apps/mobile` (Expo React Native).
- Vitest tests: `/tests`
- Playwright tests: `/e2e`

## Canonical Commands

- Install (clean): `make ci`
- Install (dev): `make install`
- Local Next.js dev, production-preview, and Playwright web-server default port: `43191` instead of common framework ports to reduce localhost conflicts.
- It is good to start a local dev server for implementation testing, browser verification, and debugging, but agents should shut down any dev server they started before finishing the task.
- Full checks: `make check`
- Format check: `make format-check`
- Lint: `make lint`
- Type check: `make types-check`
- Unit tests: `make test`
- Focused Vitest runs: `npm run test:focused -- path/to/file.spec.ts` to avoid misleading global coverage failures when running only one test file.
- E2E tests: `npm run e2e` or `npm run e2e:ui`
- Windows Start Menu launcher install/uninstall: `make install-windows-start-menu` / `make uninstall-windows-start-menu`
- Mobile dev server: `npm run mobile:start`
- Mobile Android/iOS/web launchers: `npm run mobile:android`, `npm run mobile:ios`, `npm run mobile:web`
- Mobile checks: `npm run mobile:lint`, `npm run mobile:types-check`

## Dependency Policy

- Ask before adding runtime dependencies under `dependencies`.
- Dev dependencies are acceptable when needed for tooling/testing.
- Current approved shared-storage runtime dependency: `@supabase/supabase-js`.
- Current approved mobile runtime stack: Expo SDK 55 template dependencies plus `@react-native-async-storage/async-storage` for anonymous participant persistence.
- Mobile profile sync should use Supabase Auth with RLS and the publishable/anon key, not the service-role key or a developer machine on the LAN.
- Mobile should remain local-first: answer/rating/report changes are saved on-device first and synced when Supabase is reachable.

## Testing Priorities

- Keep expanding coverage for core functionality, not just smoke tests.
- The unit-test command enforces at least 95% statements, branches, functions, and lines for the configured core logic/API coverage scope.
- Prioritize tests for quiz source selection/title logic, difficulty rating behavior, and API validation/error handling.
- For question reporting, prefer append-only shared database entries and include source/prompt snapshot context for reviewer triage.
- Do not delete handled question reports from shared storage. Mark them `resolved` with `resolved_at` and, when useful, a short `resolution_note`; active report counts should include only open reports.
- When adding a quiz topic, update the central `lib/questionTopics.ts` list and the API/product docs. Do not add topic-specific Supabase `question_reports.topic` enum constraints; reports should accept future configured topics without another report-table schema migration.
- Do not revive browser-local report storage or report export/import UI; legacy local reports should be ignored.
- For shared quiz data, use anonymous per-device participants in v1: question difficulty is global, but each participant keeps their own rating/climb state.
- Treat answer speed as a strong but incomplete signal of user skill. Rating exchange should keep a meaningful response-time weight while using min/max per-answer bounds so outlier user/question rating gaps do not dominate shared question difficulty.
- Climb mode should preserve variety for high-rating users: keep a minimum targeted candidate pool when available and include a random share from the active filtered pool so users do not repeatedly see only the highest-rated questions.

## UI Preferences

- During active quiz practice, prioritize the question prompt and answer options over the quiz-set title; keep the title smaller and visually muted.
- In the quiz header, keep the top-right practice stat focused on right-aligned accuracy; do not show separate answered/correct counters there during active practice.
- After a quiz answer is submitted, apply the rating update optimistically, show small muted up/down rating deltas inline after the participant rating and revealed question rating, and clear those deltas when the user advances. Keep reserved space for the revealed question Elo line so the prompt/options do not jump on submit.
- Quiz prompt, answer, explanation, and learning-page text should remain selectable for copy/paste. Dragging across answer text must not toggle the answer; clicking or keyboard activation should remain the deliberate selection action. Keep the root React selection-permission error filter narrow so it suppresses the known `__reactFiber` / `correspondingUseElement` selection noise without hiding unrelated app errors.

## Shared Data Operations

- For the initial legacy-to-Supabase backfill, treat `store/manual/quiz-ratings(2).json` as the latest legacy rating export snapshot.

## Question Bank Preferences

- For question-bank files under `/lib` (excluding `/lib/llm`), keep answer patterns roughly balanced across 1, 2, 3, and 4 correct-answer questions.
- Question-bank files may mix `multiple-select` and `assertion-reason` questions. Use assertion-reason items when the source material benefits from testing whether two statements are true and whether the reason explains the assertion; do not force a fixed mix, because the right question type depends on the subject and field.
- Assertion-reason questions should usually be capped at no more than `20%` of a newly authored source-material question set unless the user explicitly asks for a higher share.
- Assertion-reason questions should set `type: "assertion-reason"`, use the standard fixed five-option assertion/reason order, have exactly one correct option, and keep those options in authored order. The five options are: assertion true/reason false, assertion false/reason true, both false, both true with the reason explaining the assertion, and both true without the reason explaining the assertion. Multiple-select questions can omit `type` because they default to `multiple-select`, and their answer options are randomized during practice.
- Assertion-reason prompts should write the assertion and reason as independently judgeable statements. The reason may be a candidate explanation of the assertion, but neither sentence should directly refer to the other or require the other sentence to determine its truth.
- Across assertion-reason questions in a complete question set, distribute the correct answer as evenly as practical across the five fixed answer positions, aiming for roughly `20%` per option when there are enough assertion-reason items.
- Exact quarter splits are preferred when practical, but approximate balance is acceptable when a file size or authoring constraints make exact `25%` buckets awkward.
- For question-bank files under `/lib` (excluding `/lib/llm`), do not require an equal static difficulty-label split unless the user, source material, existing set convention, or local docs ask for one. When no difficulty split is specified, use a roughly balanced `easy`/`medium`/`hard` mix by default where practical, and always report the final difficulty counts back to the user.
- For automated validation, treat a file as acceptably balanced when each answer-count bucket stays within about `10%` of the file’s total question count from the ideal quarter split.
- Registered question sources can set `balance: false` in `QUESTION_SOURCES` when a set is intentionally unbalanced for pedagogy, such as prerequisite preparation that should skew easier; automated static balance tests should skip those sources while still checking option validity and registration.
- When rebalancing question banks, prefer minimal statement edits and corresponding `isCorrect` updates rather than rewriting whole questions.
- In the filter menu, series/book checkboxes summarize the underlying individual question sets: selecting a series selects all of its question sets, and selecting any individual question set marks its parent series without expanding to sibling sets.
- For quiz prompts and explanations, prefer self-contained wording that does not assume the user attended the lecture.
- Every question should stand alone under randomized mixed-source practice. Do not write prompts or explanations that refer to another question, a previous answer, a next question, an equation from another item, or source context that may not be visible.
- For question-bank files under `/lib`, explanations should be more than `150` characters; if validation finds a shorter explanation, expand it to at least `250` characters.
- The `.codex/skills/author-questions` skill should be used for creating new question sets, adding questions to existing sets, reviewing question quality, improving weak existing questions, and targeted rewrites of topic slices; the same quality gate applies across those operations.
- The `.codex/skills/author-learning-experience` skill should be used for creating new in-app learning pages from lecture, chapter, transcript, slides, or topic overview material. Generated learning pages should inspect the current app architecture first, register against existing quiz `SourceId`s, use shared learning primitives where they fit, update `docs/*`, and stay inside the app instead of creating PowerPoints, iframes, standalone static sites, or long copied lecture summaries.
- In-app learning pages should favor richer topic-specific teaching experiences when the material benefits from them: page-local simulations, calculators, visual reasoning labs, staged cases, annotated diagrams, timeline scrubbers, sortable evidence boards, and similar custom React/SVG/CSS interactions are preferred over generic card-based lessons, as long as they remain maintainable, mobile-friendly, tested, and do not add runtime dependencies without approval.
- For architecture-heavy, math-heavy, procedural, or diagram-heavy learning material, a generic sequence of cards and checks should be treated as a fallback rather than the default; define a core interactive object for the page and let shared primitives support that experience instead of constraining it.
- Questions should test real concept understanding rather than surface recall or answer elimination. Difficulty should reflect the knowledge, reasoning, transfer, or math required, while answer options should remain high-quality and similarly plausible.
- For source-material-derived question sets, practicing all questions to mastery should teach roughly the same core conceptual knowledge as reading and fully understanding the source material.
- Avoid low-quality distractors that can be ruled out by wording cues, extreme absolutes, category mismatch, or absurdity; wrong options should be plausible alternatives that diagnose common misunderstandings.
- Across question banks, do not let correct answers become recognizable as the only hedged, context-sensitive, evidence-aware, "reasonable middle" option. Wrong options should often be nuanced near-misses that preserve the same category, reasoning layer, and source-specific frame while changing one important condition, causal direction, scope boundary, or sufficiency claim.
- Avoid over-scaffolding a set around one reusable schema. Later or harder questions should require concrete source-grounded distinctions rather than being answerable by generic rules such as rejecting absolutes, preferring validation/context, rejecting category mismatches, or preserving a mechanism-to-outcome chain.
- When creating, adding, rewriting, fixing, or otherwise editing questions, do a manual low-diagnosticity triage even if automated guessability checks pass. Substantially rewrite weak definition-recognition or plausible-sentence-recognition items into application, prediction, mechanism, comparison, calculation, boundary-case, or transfer questions, and report created-or-changed/reviewed/rewritten/minor-edit/retained-orientation counts.
- Do not treat scenarios as the default way to improve low-diagnosticity questions across `/lib`; use scenarios only when they fit the topic naturally, and prefer direct comparisons, boundary cases, mechanisms, calculations, consequences, or transfer questions when those better fit the material.
- Do not treat "which statement best describes X" prompts as inherently bad. They are acceptable when answer options are plausible competing descriptions, and scenario setups should be kept only when they supply facts that affect the answer choices.
- Intentional orientation questions can stay easy, but medium and hard items should not be answerable mainly by recognizing a familiar definition, matching object categories, rejecting absurd options, or following the everyday wording of the stem.
- After creating or editing registered question sets with `.codex/skills/author-questions`, run the targeted guessability test with `QUESTION_GUESSABILITY_SOURCE_IDS` set to the changed source id. If simple language-cue heuristics score well, improve the answer options instead of treating the test as a nuisance.
- Use the repo-local `.codex/skills/author-questions` skill to create or edit question sets from source-material folders; source material can be transcripts, chapters, papers, slides, PDFs, or other learning artifacts, and generated sets should be registered in `lib/quiz.ts` so they are selectable for practice.
- When students are struggling with a math-heavy question set, prefer adding applied A-level/AP-style computation questions before adding more abstract theory-only questions.
- Crash Course Linear Algebra Lecture 0 should stay focused on prerequisite AP/A-level linear algebra rather than AI applications, helping students practice the math needed for later lecture question sets.
- Crash Course Linear Algebra Lecture 0 should explicitly prepare students for recurring notation and terminology such as \(\mathbb{R}^n\), subscripts, transposes, summations, norms, matrix shapes, span, basis, rank, determinants, inverses, eigenvectors, and Singular Value Decomposition (SVD).
- Crash Course Linear Algebra Lecture 2-5 expansions should deepen mathematical rigor and applied linear algebra practice, with questions that test computation, shape reasoning, transformations, gradients, decompositions, attention, RL value functions, and conceptual transfer rather than surface recall.
- Crash Course Probability expansions should emphasize mathematical rigor and applied fluency: event notation, probability-mass calculations, PMF validity, random-variable notation, expected value, empirical expected loss, calibration arithmetic, variance, and AI probability interpretation rather than only verbal definition recall.
- Crash Course Probability question files generated from overview source material should use the actual lecture title in the TypeScript filename rather than `overview`, while still living in the parent course folder when the source material is under `transcripts-and-files`.
- Biology & Chemistry for Life Science Lecture 0 should stay easier than L1-L5 and focus on prerequisite concepts and terminology that students need before attempting the lecture question sets.
- Biology & Chemistry for Life Science Lecture 0 should use plausible same-topic distractors rather than category-mismatch joke options, and its static difficulty labels should be allowed to skew heavily toward easy/medium when that better serves no-prior-knowledge preparation.
- For preparation question sets, answer options should teach useful same-topic contrasts. Avoid random cross-topic distractors that only teach that a concept is not an unrelated statistic, study-design term, drug concept, or other category mismatch.
- Biology & Chemistry for Life Science question sets should be treated as part of the course material: practicing them to mastery should give students roughly the same core conceptual knowledge as attending and understanding the lectures.
- In the Biology & Chemistry for Life Science crash course, clinical trials and research evidence should be contextual validation material rather than the core capstone subject; use the separate Clinical Trials Crash Course for deeper trial-design coverage.
- Clinical Trials Crash Course question sets should emphasize deep applied understanding of each lecture's overview and transcript coverage, with medium/hard items testing causal reasoning, design tradeoffs, statistical interpretation, operational failure modes, and modern evidence-generation limits rather than surface recall.
