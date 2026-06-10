---
name: author-learning-experience
description: Design, implement, and register high-quality Learning AI web learning experiences from lectures, chapters, transcripts, slides, PDFs, papers, notes, or topic overviews. Use when the user asks for an interactive learning page, subpage, teaching experience, explorable lesson, simulation, visual reasoning lab, or source-material preparation page tied to this repo's quiz/learning surface.
---

<overview>
Create web-first learning experiences that teach the source material in the
shape the topic deserves. Success is not "a page exists"; success is that a
student can build the right mental model before practicing the matching MCQs.

Default to integrating with the existing Learning AI Next.js app, quiz source
ids, docs, and tests when the task is quiz preparation. Treat existing pages and
shared components as integration examples, not visual or pedagogical templates.
The experience should be allowed to feel different when the topic demands a
different teaching form.
</overview>

<core-rules>
- Choose the delivery mode first: integrated quiz-prep page, standalone in-repo
  subpage, experimental lab, or lightweight explorable article. Do not force a
  quiz-prep shape when the user explicitly asks for an independent subpage.
- Design the ideal learning experience first, then fit it into the chosen
  delivery mode.
- Do not copy the structure, rhythm, or component mix of previous learning
  pages unless that shape is genuinely the best fit for the new material.
- Existing pages are useful for route, registry, style, testing, and accessibility
  conventions; they are not evidence that future pages should look similar.
- Do not default to `LearningHero` plus cards/checks as the page skeleton. Use
  those primitives only when they support a stronger topic-native design.
- Use the source material's concepts, examples, formulas, diagrams, cases, and
  misconceptions, but do not copy a transcript or slide order into a long article.
- Prefer one strong topic-specific interactive object over many generic cards
  when the material has a mechanism, system, workflow, calculation, tradeoff,
  spatial structure, timeline, evidence interpretation, or model behavior.
- It is acceptable for a strong page to look unlike the rest of the learning
  pages, as long as navigation, accessibility, responsiveness, and maintainability
  are still handled deliberately.
- Keep the implementation maintainable, accessible, mobile-friendly, and testable.
- Ask before adding runtime dependencies. Prefer built-in React, SVG, CSS,
  canvas, and browser APIs unless a proven domain library is clearly justified.
- Default to in-app pages. If the user explicitly asks for a standalone or
  experimental subpage, optimize for the learning experience first, then preserve
  any repo-required docs, tests, and integration points that still apply.
</core-rules>

<workflow>
1. Decide delivery mode and constraints.
   - If the user asks for a normal learning experience tied to an existing quiz
     source, build an integrated page whose canonical route is
     `/learn/[seriesId]/[sourceId]`.
   - If the user explicitly asks for a standalone, independent, experimental, or
     unusually rich subpage, do not force the page through the existing learning
     registry or shared primitives unless that still serves the request.
   - Preserve repo-level requirements that still apply: docs as source of truth,
     no secrets, no unapproved runtime dependencies, tests for behavior, and
     small reviewable changes.
   - For standalone in-repo pages, still provide a clear route, browser-visible
     verification, and product-doc note describing what was added.

2. Gather integration context without anchoring the design.
   - Read `docs/product-scope.md`, `docs/team-preferences.md`, `lib/learning.ts`,
     `lib/quiz.ts`, `app/learn`, `components/learning`, and the relevant
     question/source files when the page is integrated with quiz practice.
   - For standalone or experimental pages, inspect only the app conventions that
     affect routing, styling, accessibility, and verification. Avoid letting
     existing learning pages decide the teaching shape.
   - Use the existing `SourceId` as the learning-page key whenever the material
     maps to an existing quiz source and the delivery mode is integrated quiz
     prep.
   - Use the quiz source's `seriesId` for course navigation so `/learn` can stay
     a course index and `/learn/[seriesId]` can list that course's learning
     experiences.
   - Inspect existing learning pages only to understand conventions and avoid
     accidental duplication. Do not use them as default layouts.
   - If docs and implementation disagree, flag the conflict and choose the
     documented behavior unless the user explicitly updates the requirement.

3. Study the source material.
   - Source material may be a lecture, chapter, transcript, overview, slide deck,
     PDF, paper, notes, dataset, code example, or topic brief.
   - Identify the core learning job: what the student should be able to predict,
     explain, calculate, compare, diagnose, trace, or decide after the page.
   - Extract prerequisite assumptions, conceptual sequence, formulas, diagrams,
     examples, likely misconceptions, and the MCQ concepts the page should
     prepare for.
   - Identify what makes the material hard: invisible mechanism, notation,
     abstraction stack, quantitative tradeoff, causal chain, temporal process,
     ambiguous evidence, confusable categories, or implementation details.

4. Choose an experience shape before drafting UI.
   - Generate at least three plausible experience shapes, then choose the one with
     the strongest teaching leverage. You do not need to show this reasoning to
     the user unless the task is large or ambiguous.
   - Write a short internal experience brief before implementation:
     `learner job`, `central object`, `primary interaction`, `why this shape`,
     and `what existing pattern it deliberately avoids`.
   - Favor a continuous central object when possible: a simulator, annotated
     architecture, editable example, decision lab, evidence reader, process
     model, timeline, calculator, parser, debugger, map, or comparison workbench.
   - If all candidate shapes resemble previous learning pages, pause and look for
     a more topic-native form.
   - If the best design would be substantially better as a standalone or more
     immersive subpage than as a standard learning-card flow, say so and either
     implement that mode when allowed or explain the compromise.
   - A compact explorable article is acceptable for text-first conceptual
     material, but it should be a deliberate choice, not the fallback produced by
     shared components.

5. Design the teaching flow.
   - Start with the core problem, intuition, or concrete situation, not a
     definition dump.
   - Make the beginning, middle, and end clear: orient, manipulate/interpret,
     consolidate, then transition into practice.
   - Include active moments where the learner predicts, manipulates, compares,
     sorts, annotates, debugs, estimates, reads evidence, or checks a misconception.
   - Include worked examples for mathematical, technical, procedural, or
     evidence-heavy topics.
   - Use visual structure that belongs to the domain. Architecture can be
     spatial, math can be manipulable, clinical/statistical material can be an
     evidence reader, history can be a timeline, code can be a trace/debugger,
     biology can be a pathway/case model, and NLP/LLM topics can be token,
     attention, or generation workbenches.
   - Avoid visible instructional filler that explains the UI instead of teaching
     the concept. Controls and state should make the task understandable.
   - End with a concise recap and a transition into the matching MCQs.

6. Implement according to the chosen delivery mode.
   - For integrated quiz-prep pages:
     - Add or update metadata in `lib/learning.ts`.
     - Add the topic component under `components/learning/pages/`.
     - Add the page component to the learning route map.
     - Link to the page through the course-first path
       `/learn/[seriesId]/[sourceId]`; keep direct `/learn/[sourceId]` only as
       compatibility if the app still supports it.
   - For standalone or experimental subpages:
     - Add a route that matches the user's request and keeps the experience easy
       to open directly.
     - Do not require a quiz transition unless the page is meant to prepare for
       a specific source.
     - Use app shell, navigation, or shared styling only where they help users
       orient themselves. Avoid importing learning primitives by default.
   - Use shared primitives from `components/learning/LearningPrimitives.tsx` only
     when they serve the chosen experience. Do not let primitives determine the
     page shape.
   - Page-local custom components may have nontrivial typed state/config when
     that state powers the teaching model. Keep them deterministic and readable.
   - For bespoke visuals, use stable dimensions, responsive constraints,
     keyboard-accessible controls, visible state changes, and text alternatives
     or labels.
   - Use `MathText` for formulas and existing Tailwind conventions where they fit.
   - For display formulas, pass valid LaTeX through `MathText` or `FormulaBlock`;
     prefer TSX expression props with `String.raw`, for example
     ``formula={String.raw`\[...\]`}``, so LaTeX commands are not double-escaped.
   - Keep custom local data/config objects page-local unless they are clearly
     reusable.
   - Do not hard-code unrelated app state or add runtime dependencies without
     user approval.
   - Do not add mobile app learning screens unless the user explicitly asks for
     mobile in the same task.

7. Keep docs and tests current.
   - Update `docs/product-scope.md` when product behavior or available learning
     pages change.
   - Update `docs/team-preferences.md` when the task creates a durable authoring
     convention or preference.
   - For integrated pages, add or update Vitest coverage for learning
     registry/source-id behavior.
   - Add or update Playwright coverage for the route, at least one meaningful
     interaction, and mobile-width layout.
   - For integrated pages, also cover transition into the matching quiz source.

8. Verify before finishing.
   - Prefer `make check` before final response when practical.
   - For focused work, at least run:
     - `npm run test:focused -- tests/lib/learning.spec.ts` when registry
       behavior changed
     - `npm run e2e`
     - `make format-check`
     - `make lint`
     - `make types-check`
   - If this skill itself changed, run the skill-quality validator on
     `.codex/skills/author-learning-experience`.
</workflow>

<experience-shape-heuristics>
Use these as prompts, not templates:

- Architecture or systems: manipulable diagram, pipeline debugger, component
  router, data-flow trace, dependency map.
- Math or probability: calculator, slider-driven curve, worked-example lab,
  shape/notation interpreter, counterexample generator.
- Algorithms or code: stepper, trace table, debugger, input/output playground,
  failure-mode lab.
- Biology, medicine, or clinical material: pathway model, staged case, evidence
  reader, risk/benefit calculator, mechanism-to-outcome map.
- History, product, policy, or process: timeline scrubber, decision log, tradeoff
  board, stakeholder map.
- Language, NLP, or LLM behavior: token workbench, attention map, generation
  sandbox, prompt/output comparator, model-family chooser.
- Conceptual or philosophical material: contrastive examples, argument map,
  misconception sorter, boundary-case lab.
</experience-shape-heuristics>

<anti-patterns>
- Reusing the same hero, three-card intro, comparison, formula, check, recap
  rhythm because it worked before.
- Treating interaction as decorative toggles over static prose.
- Designing around MCQ topic labels instead of the source material's central
  learning difficulty.
- Turning slides or transcript sections into a long scroll of summaries.
- Using shared primitives because they are available rather than because they
  teach this topic well.
- Hiding the main teaching model below many introductory cards.
- Making a page that is technically correct but could be swapped with any other
  topic by changing labels.
- Adding animation, charts, or complex state that looks impressive but does not
  improve the student's reasoning.
</anti-patterns>

<review-checklist>
Before finishing, confirm:

- The page works under `/learn/[seriesId]/[sourceId]` and links to
  `/?source=<SourceId>` when it is an in-app learning page.
- The canonical in-app route is reachable from `/learn` through the course page:
  `/learn` -> `/learn/[seriesId]` -> `/learn/[seriesId]/[sourceId]`.
- The page uses the app's normal routes, navigation, quiz source ids, and docs
  contracts where applicable.
- The page is mobile-friendly and text remains selectable.
- The central interaction is meaningful, not decorative.
- The experience shape is topic-native and not just copied from previous pages.
- The page would still make sense if no previous Learning AI page existed; any
  similarity to earlier pages is justified by the topic, not convenience.
- The learner actively predicts, manipulates, compares, calculates, annotates, or
  interprets something important.
- The content follows the source material's conceptual structure without copying
  the source order verbatim.
- Formulas render as KaTeX, not visible raw `$$...$$` or escaped LaTeX.
- The page does not feel like a generic landing page, copied transcript, or
  generic sequence of cards/checks.
- Future maintainers can understand the page-local state and tests.
</review-checklist>
