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
- Full checks: `make check` (includes the focused Playwright smoke test)
- Format check: `make format-check`
- Lint: `make lint`
- Type check: `make types-check`
- Unit tests: `make test`
- Focused Vitest runs: `npm run test:focused -- path/to/file.spec.ts` to avoid misleading global coverage failures when running only one test file.
- Focused browser smoke: `npm run e2e:smoke` or `make e2e-smoke`
- Full E2E tests: `npm run e2e` or `npm run e2e:ui`
- Windows Start Menu launcher install/uninstall: `make install-windows-start-menu` / `make uninstall-windows-start-menu`
- Mobile dev server: `npm run mobile:start`
- Mobile Android/iOS/web launchers: `npm run mobile:android`, `npm run mobile:ios`, `npm run mobile:web`
- Mobile checks: `npm run mobile:lint`, `npm run mobile:types-check`
- Before running the full verification gate, run `make format` first and then `make check` so format-sensitive generated or source-derived files are normalized before the final check.

## Dependency Policy

- Runtime dependencies under `dependencies` are acceptable when they materially improve the central product or learning experience.
- Prefer existing dependencies and browser-native APIs when they are enough.
- Keep dependency additions focused and minimal; do not add broad libraries for small conveniences.
- Ask before adding a runtime dependency when the dependency is not clearly central to the requested work or when there is a reasonable dependency-free alternative.
- Dev dependencies are acceptable when needed for tooling/testing.
- Current approved shared-storage runtime dependency: `@supabase/supabase-js`.
- Current approved learning-experience visual dependency: `lucide-react` for icons and icon-led controls when it improves UX, navigation, or domain-specific visual language.
- Current approved mobile runtime stack: Expo SDK 55 template dependencies plus `@react-native-async-storage/async-storage` for anonymous participant persistence.
- Mobile profile sync should use Supabase Auth with RLS and the publishable/anon key, not the service-role key or a developer machine on the LAN.
- Mobile should remain local-first: answer/rating/report changes are saved on-device first and synced when Supabase is reachable.

## Testing Priorities

- Keep expanding coverage for core functionality, not just smoke tests.
- Browser smoke coverage should fail on unexpected console errors, page errors,
  and hydration/runtime warnings on the home page so `make check` catches
  client-rendering regressions.
- The unit-test command enforces at least 95% statements, branches, functions, and lines for the configured core logic/API coverage scope.
- Prioritize tests for quiz source selection/title logic, difficulty rating behavior, and API validation/error handling.
- For question reporting, prefer append-only shared database entries and include source/prompt snapshot context for reviewer triage.
- Do not delete handled question reports from shared storage. Mark them `resolved` with `resolved_at` and, when useful, a short `resolution_note`; active report counts should include only open reports.
- When adding a quiz topic, update the central `lib/questionTopics.ts` list and the API/product docs. Do not add topic-specific Supabase `question_reports.topic` enum constraints; reports should accept future configured topics without another report-table schema migration.
- Do not revive browser-local report storage or report export/import UI; legacy local reports should be ignored.
- For shared quiz data, use anonymous per-device participants in v1: question difficulty is global, but each participant keeps their own rating/climb state.
- Treat answer speed as a strong but incomplete signal of user skill. Rating exchange should keep a meaningful response-time weight while using min/max per-answer bounds so outlier user/question rating gaps do not dominate shared question difficulty. Correct answers should get less credit when slow. Incorrect answers should be discounted when fast, then penalized more as elapsed time increases, reaching full wrong-answer weight at the timer cap.
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
- For assertion-reason items, mark the "both true with the reason explaining the assertion" option only when the reason gives a causal, mechanistic, or logically sufficient explanation of why or how the assertion holds. A related true fact, recommendation, downstream benefit, or mitigation belongs under "both true without the reason explaining the assertion."
- Across assertion-reason questions in a complete question set, distribute the correct answer as evenly as practical across the five fixed answer positions, aiming for roughly `20%` per option when there are enough assertion-reason items.
- Exact quarter splits are preferred when practical, but approximate balance is acceptable when a file size or authoring constraints make exact `25%` buckets awkward.
- For question-bank files under `/lib` (excluding `/lib/llm`), do not require an equal static difficulty-label split unless the user, source material, existing set convention, or local docs ask for one. When no difficulty split is specified, use a roughly balanced `easy`/`medium`/`hard` mix by default where practical, and always report the final difficulty counts back to the user.
- For automated validation, answer-count balance tolerance should become stricter as question sets get larger. Use an absolute allowed deviation of `max(1, min(5, ceil(totalQuestions * 0.1)))` from the ideal quarter split rather than an uncapped fixed percentage of total questions. Existing legacy exceptions should be exact and explicit in tests; new or touched question sets should meet the capped tolerance.
- Registered question sources can set `balance: false` in `QUESTION_SOURCES` when a set is intentionally unbalanced for pedagogy, such as prerequisite preparation that should skew easier; automated static balance tests should skip those sources while still checking option validity and registration.
- When rebalancing question banks, prefer minimal statement edits and corresponding `isCorrect` updates rather than rewriting whole questions.
- Question IDs are stable database identities, not display numbers or position counters. Hardcode each question's `id`; do not derive it from array position, helper call order, sorting, or renumbering. Preserve an existing `id` only for minor wording, option, explanation, or difficulty fixes that keep the same underlying question. Assign a new never-before-used `id` when adding a question, replacing a question, changing the tested construct, changing the answer key in a way that changes scoring semantics, or substantially rewriting the prompt/options/explanation. Removing a question removes its ID from the bundled bank, but future new questions must not reuse that removed ID.
- In the filter menu, series/book checkboxes summarize the underlying individual question sets: selecting a series selects all of its question sets, and selecting any individual question set marks its parent series without expanding to sibling sets.
- For quiz prompts and explanations, prefer self-contained wording that does not assume the user attended the lecture.
- Quiz prompts should use compact GitHub-Flavored Markdown tables a little more readily when tabular data, scenarios, cases, probability values, model states, trial arms, or calculation inputs would be easier to read that way. For dense mathematical prompts with several named variables, log probabilities, model/reference quantities, policy states, or arithmetic inputs, prefer a small table of givens over embedding everything in one sentence. Tables should be introduced by the prompt, use meaningful row and column labels, stay mobile-friendly, and remain self-contained.
- Explanations should teach the concept, mechanism, and nearby misconceptions directly. Avoid mechanical answer-taking language such as telling the learner to select or reject quoted answer statements.
- Quiz explanations are read after the learner submits, so the UI already shows which answers are right or wrong. Explain the reason, mechanism, counterexample, or boundary condition behind that status; for example, explain that black swans in Australia disprove a universal swan-color claim instead of only saying that "all swans are white" is incorrect or a common misconception.
- Every question should stand alone under randomized mixed-source practice. Do not write prompts or explanations that refer to another question, a previous answer, a next question, an equation from another item, or source context that may not be visible.
- For question-bank files under `/lib`, explanations should be more than `150` characters; if validation finds a shorter explanation, expand it to at least `250` characters.
- The `.codex/skills/author-questions` skill should be used for creating new question sets, adding questions to existing sets, reviewing question quality, improving weak existing questions, and targeted rewrites of topic slices; the same quality gate applies across those operations.
- The `.codex/skills/author-learning-experience` skill should be used for creating new in-app learning pages from lecture, chapter, transcript, slides, or topic overview material. Generated learning pages should inspect the current app architecture first, register against existing quiz `SourceId`s when integrated with practice, update `docs/*`, and stay inside the app instead of creating PowerPoints, iframes, standalone static sites, or long copied lecture summaries unless the user explicitly asks for a standalone or experimental subpage.
- Stanford CME295 slide PDFs under `lib/lectures/Stanford CME295 Transformers & LLMs/transcripts-and-files` should use the `lecture X - slides.pdf` filename pattern when paired with `lecture X - transcript.md`.
- Stanford CME295 source-material curriculum artifacts should live beside the lecture transcript/PDF pair under `transcripts-and-files` and use the filename pattern `lecture X - curriculum.md`.
- For Stanford CS224R Lecture 4 question-bank work, use `lib/lectures/Stanford CS224R Deep Reinforcement Learning/transcripts-and-files/lecture 4 - slides_correct.pdf` as the slide source of truth with `lecture 4 - transcript.md`; the older `lecture 4 - slides.pdf` contains off-policy actor-critic slides that belong with Lecture 5.
- Integrated learning pages should use the course-first navigation hierarchy: `/learn` lists courses, `/learn/[seriesId]` lists that course's learning experiences, and `/learn/[seriesId]/[sourceId]` is the canonical learning-page URL. Keep legacy direct `/learn/[sourceId]` support only as compatibility, not as the main link target.
- Course learning-experience cards should identify the underlying source sequence, such as `Lecture 2` or `Chapter 5`, and be sorted by the registered source order. Do not use duration or prerequisite-level copy as the primary card metadata.
- In-app learning pages should favor richer topic-specific teaching experiences when the material benefits from them: page-local simulations, calculators, visual reasoning labs, staged cases, annotated diagrams, timeline scrubbers, sortable evidence boards, canvas-based explainers, and similar custom interactions are preferred over generic card-based lessons, as long as they remain maintainable, mobile-friendly, tested, and documented. Older Learning AI learning pages are generally not the UX quality bar; use them for routing, registry, and testing conventions only, and assume newer pages should be more ambitious unless the source material is genuinely simple.
- Runtime dependencies for learning experiences are acceptable when they materially improve the central teaching object, such as visualization, animation, simulation, math rendering, or interaction quality. Prefer existing dependencies and browser-native APIs when sufficient, but do not let dependency avoidance block a substantially better learning experience; keep additions focused and explain why the dependency is needed and what alternatives were considered.
- For learning experiences, focused visual dependencies, icons, custom CSS/SVG, lightweight animation, and domain-specific visual systems are acceptable when they materially improve comprehension or engagement. Lack of an existing icon import should not push the design toward text-only panels.
- For architecture-heavy, math-heavy, procedural, or diagram-heavy learning material, a generic sequence of cards and checks should be treated as a fallback rather than the default; define a core interactive object for the page and let shared primitives support that experience instead of constraining it.
- Existing learning pages should be treated as route, registry, accessibility, and testing references rather than visual or pedagogical templates. New learning pages should first choose a topic-native experience shape, and only then adapt app styling as needed. Shared learning primitives are optional support components, not the default architecture; page-local components should carry the topic-specific teaching model when they improve the learning experience.
- Learning pages should resist template drift: recurring panels, hero layouts, card grids, checks, and recap structures should be reused only when they serve the source-native learner task, not because previous pages used them.
- For new learning experiences, "small and reviewable" means keep changes scoped to the learning surface and avoid unrelated refactors; it should not cap the ambition, visual design, simulator depth, or interaction quality of the page.
- Learning pages should start from a creative source-native concept: a problem, object, mechanism, case, model, tradeoff, or interaction that would not make sense for a different lecture by merely swapping labels.
- Learning-page instructional copy should be self-contained and concept-facing rather than repeatedly referring to "the lecture", "Lecture N", "the slides", or source order. Keep source sequence labels for navigation, cards, breadcrumbs, metadata, and quiz titles, but teach the concept directly in the page body.
- Visualization and interaction should be chosen for teaching leverage. Invisible mechanisms, quantitative tradeoffs, architectures, evidence structures, and spatial relationships often need something inspectable, but the goal is a better mental model rather than satisfying a fixed visual-component checklist.
- Learning-experience tests should cover more than route existence and one button click. Add checks for the central learning journey, staged reveal or learner-built artifacts when present, visible visual models, feedback or scoring changes, navigation anchors, and mobile no-overflow behavior. Passing tests is not sufficient if browser review shows the page still feels like a generic card/check flow.
- New learning experiences should run a frontend UX review loop before finalizing, preferably with the `frontend-ux-review` skill or an equivalent screenshot/browser critique across desktop and mobile.
- Checks and practice prompts should appear where they best support the learner's current mental model, including immediately after a topic when local feedback would help.
- Learning pages should build concepts in dependency order. Introduce vocabulary, controls, units, and notation before asking learners to use a dense lab; if a lab appears early, it should teach one concept at a time through a guided progression or inline legend.
- Learning pages should include simple first-use explanations for core terms, controls, units, symbols, architecture parts, and algorithm names before using them in dense labs or checks. Learners who already know the concept can scroll past the explanation; learners who do not should not have to infer the missing basics.
- Before finalizing a learning page, run a learner-scroll UX check: imagine a learner reading from top to bottom and verify that each section naturally follows from the previous one, introduces what the next section needs, and avoids jumps into unexplained vocabulary or disconnected labs.
- For hard technical lecture learning pages, introduce core algorithm names, roles, equations, and symbols before interactions that depend on them. A lab should not assume concepts such as PPO, advantage, reward models, RL, RLHF, or DPO before the page has taught the simplified mental model.
- Interactive numeric outputs should expose the calculation path near the output: show the governing formula, selected input values, and intermediate arithmetic so learners can reconstruct results such as Best-of-N probabilities, PPO clipping terms, or DPO logits.
- When source material includes a canonical formula, benchmark metric, or algorithm objective, learning pages should present that source-aligned version. Simplified or toy formulas are useful when clearly labeled as approximations or intuition builders, especially when paired with interactive calculations.
- Mathematical notation in learning-page prose, headings, labels, and explanatory notes should render through KaTeX/`MathText` inline rather than appearing as raw ASCII such as `s_t`, `A_t`, `pi_theta`, or `r_phi(x,y)`.
- Long-scroll learning pages are acceptable when the material needs dependency-ordered concept buildup. Prefer clear section sequencing over forcing all related ideas into compact dashboards that hide prerequisites.
- For abstract or invisible mechanisms, use obvious visual structure such as icons, pipelines, rails, annotated diagrams, or formula boards to make the source-native object inspectable rather than relying on text-only panels.
- Workflow-based learning labs should make the visible scan order match the intended play-through order. Related inputs and feedback may sit side-by-side inside one step, but later representations, diagnoses, answers, outputs, or plans should not appear beside earlier evidence or setup in a way that suggests the learner can skip prerequisite work.
- Learning-page browser review should include realistic intermediate viewports, especially around `1280 x 800`, when the page has dense labs, multi-column layouts, large headings, or text-heavy controls. Check for awkward word breaks, clipped text, cramped columns, and horizontal overflow there as well as on mobile.
- Hero visuals for learning pages should be concrete, contextual, and immediately interpretable. Avoid abstract diagrams with arrows, bullets, meters, or bar charts unless the visual itself is a meaningful first model and the surrounding copy makes clear what it represents.
- Recaps and formula boards are useful consolidation tools, but they should not introduce core concepts for the first time. Important notation, such as latent variables, \(x_T\), \(\mathcal{N}(0,I)\), or identity/covariance symbols, should be explained at first meaningful use in the flow.
- When teaching latent variables, distinguish model-sampled hidden factors, inferred hidden factors, user-selected conditions, and scenario controls so the UI does not imply that latent variables are directly observed or manually controlled in the real model unless that contrast is the lesson.
- When a user explicitly asks for a standalone, independent, experimental, or unusually rich learning subpage, do not force the work through the existing `/learn/[sourceId]` card/check pattern or shared primitives; pick the delivery mode and interaction model that best teaches the material while preserving applicable docs, tests, accessibility, and dependency rules.
- Questions should test real concept understanding rather than surface recall or answer elimination. Difficulty should reflect the knowledge, reasoning, transfer, or math required, while answer options should remain high-quality and similarly plausible.
- For source-material-derived question sets, practicing all questions to mastery should teach roughly the same core conceptual knowledge as reading and fully understanding the source material.
- Avoid low-quality distractors that can be ruled out by wording cues, extreme absolutes, category mismatch, or absurdity; wrong options should be plausible alternatives that diagnose common misunderstandings.
- Across question banks, do not let correct answers become recognizable as the only hedged, context-sensitive, evidence-aware, "reasonable middle" option. Wrong options should often be nuanced near-misses that preserve the same category, reasoning layer, and source-specific frame while changing one important condition, causal direction, scope boundary, or sufficiency claim.
- In multiple-select questions, do not make correct answers recognizable as the only option with substantial math, a long formula, a calculation, or KaTeX/LaTeX. Do not avoid math when it belongs; instead add plausible competing formulas, calculations, dimensions, or boundary cases so learners must understand the math rather than detect its presence.
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
- Crash Course Probability Lecture 0 should be pure applied high-school/AP/A-level math practice with no direct AI framing and no assumed AI knowledge. Focus it on ratios/odds, algebraic rearrangement, normalization, function and parameter notation, subscripts/sequences, max/argmax/min, summation/product notation, weighted averages, powers/roots, exponentials, Euler's number, logarithms, shape/dimension arithmetic, finite distributions, Gaussian mean/variance/standard-deviation notation, complements with powers, and geometric discounted sums.
- Crash Course Probability Lecture 5 hardening should emphasize applied generation math: multinomial sampling counts, top-k/top-p renormalization, temperature odds and entropy, latent-variable marginalization and posterior inference, Gaussian noising formulas, diffusion Markov/reverse-process factorization, guidance interpolation, and autoregressive sequence likelihood.
- Crash Course Probability question files generated from overview source material should use the actual lecture title in the TypeScript filename rather than `overview`, while still living in the parent course folder when the source material is under `transcripts-and-files`.
- Biology & Chemistry for Life Science Lecture 0 should stay easier than L1-L5 and focus on prerequisite concepts and terminology that students need before attempting the lecture question sets.
- Biology & Chemistry for Life Science Lecture 0 should use plausible same-topic distractors rather than category-mismatch joke options, and its static difficulty labels should be allowed to skew heavily toward easy/medium when that better serves no-prior-knowledge preparation.
- For preparation question sets, answer options should teach useful same-topic contrasts. Avoid random cross-topic distractors that only teach that a concept is not an unrelated statistic, study-design term, drug concept, or other category mismatch.
- Biology & Chemistry for Life Science question sets should be treated as part of the course material: practicing them to mastery should give students roughly the same core conceptual knowledge as attending and understanding the lectures.
- In the Biology & Chemistry for Life Science crash course, clinical trials and research evidence should be contextual validation material rather than the core capstone subject; use the separate Clinical Trials Crash Course for deeper trial-design coverage.
- Clinical Trials Crash Course question sets should emphasize deep applied understanding of each lecture's overview and transcript coverage, with medium/hard items testing causal reasoning, design tradeoffs, statistical interpretation, operational failure modes, and modern evidence-generation limits rather than surface recall.
