---
name: author-learning-experience
description: Create and register rich Learning AI in-app learning experience pages from lecture, chapter, transcript, overview, slides, PDF, or topic material. Use when the user asks Codex to make an interactive learning subpage that prepares students before existing multiple-choice questions, especially pages that should feel like a topic-specific teaching webpage, explorable article, simulation, or data/visual reasoning lab inside this repo.
---

# Author Learning AI Learning Experiences

## Overview

Create web-first learning pages inside the existing Learning AI Next.js app. The output should prepare students for a registered quiz source before they answer MCQs, using shared app infrastructure plus topic-specific React where it improves teaching. Prefer memorable, domain-specific interactive teaching experiences over generic card-based summaries when the material benefits from simulation, visual reasoning, staged cases, or exploratory controls.

Do not create PowerPoints, standalone static sites, external iframes, or long copied lecture summaries.

## Workflow

1. Inspect the app before deciding the implementation.
   - Read `docs/product-scope.md`, `docs/team-preferences.md`, `lib/learning.ts`, `lib/quiz.ts`, `app/learn`, `components/learning`, and the relevant existing topic/question files.
   - Use the existing `SourceId` as the learning-page key whenever the material maps to an existing quiz source.
   - If docs and implementation disagree, flag the conflict and choose the documented behavior unless the user explicitly updates the requirement.

2. Read the source material.
   - Source material may be an overview, transcript, chapter, PDF, slide deck, or notes.
   - Identify the core learning goal, prerequisite assumptions, conceptual sequence, formulas, examples, likely misconceptions, and which MCQ concepts the page should prepare for.
   - Identify the best experience shape for the material before drafting UI. Examples: mechanism simulator, editable example lab, quantitative calculator, staged workflow, evidence reader, visual architecture walkthrough, timeline scrubber, or compact explorable article.
   - Avoid copying the source structure verbatim when it would become a plain article.

3. Design the learning experience.
   - Start with the core problem or intuition, not a definition dump.
   - Use a clear beginning, middle, and end.
   - Include visual structure and short sections.
   - Define the page's core interactive object in one sentence before implementation. For architecture-heavy, math-heavy, procedural, or diagram-heavy material, the page should usually revolve around that object rather than around a sequence of cards.
   - Include active moments where the learner taps, compares, predicts, sorts, adjusts a control, or answers a small check.
   - When the source material has causal mechanisms, quantitative tradeoffs, diagrams, timelines, workflows, evidence interpretation, or model behavior, prefer at least one rich page-local interactive model over a mostly static explainer. Good patterns include calculators, sliders, visual simulations, sortable evidence boards, staged case workups, annotated diagrams, editable examples, timeline scrubbing, forest/Kaplan-Meier-style readers, and small decision labs.
   - Treat shared primitives as scaffolding, not a ceiling. A strong page may combine primitives with custom React, SVG, canvas, or CSS visuals as long as the result stays maintainable, accessible, mobile-friendly, and tested.
   - Include worked examples for mathematical, technical, or procedural topics.
   - Include at least one misconception check when likely confusions exist.
   - End-to-end teaching flow matters more than matching the previous learning page's visual pattern; avoid over-reusing the same card/check structure when a topic-specific interface would teach better.
   - A mostly card/check-based page is acceptable for text-first conceptual material, but it is under-scoped for material whose central idea is a mechanism, system architecture, visual transformation, calculation, or decision process.
   - End with a recap and transition into the matching MCQs.

4. Implement inside the app.
   - Add or update metadata in `lib/learning.ts`.
   - Add the topic component under `components/learning/pages/`.
   - Use shared primitives from `components/learning/LearningPrimitives.tsx` when they fit; add new primitives only when they will help future pages stay consistent. Do not let the existing primitives prevent richer page-local interactions.
   - Page-local custom components may have nontrivial typed state/config when that state powers the core teaching model. Keep them readable, deterministic, and scoped to the page unless reuse is likely.
   - For bespoke visuals, prefer stable dimensions, responsive constraints, keyboard-accessible controls, visible state changes, and text alternatives or labels so the interaction remains usable and testable.
   - Add the page component to the route map in `app/learn/[sourceId]/page.tsx`.
   - Use `MathText` for formulas and existing Tailwind styling conventions.
   - For display formulas, pass valid LaTeX through `MathText` or `FormulaBlock`; prefer TSX expression props with `String.raw`, for example ``formula={String.raw`\[...\]`}``, so LaTeX commands are not double-escaped, and verify rendered pages show KaTeX output instead of visible raw `$$...$$` text.
   - Keep custom local data/config objects page-local unless they are clearly reusable. Larger page-local state/config is acceptable when it directly powers the teaching model and remains readable.
   - Prefer built-in React, SVG, CSS, and browser APIs for rich interactions. Ask before adding runtime visualization, animation, charting, or simulation dependencies.
   - Do not hard-code unrelated app state or add runtime dependencies without user approval.
   - Do not add mobile learning screens unless the user explicitly asks for mobile in the same task.

5. Keep docs and tests current.
   - Update `docs/product-scope.md` when product behavior or available learning pages change.
   - Update `docs/team-preferences.md` when the task creates a durable authoring convention or preference.
   - Add or update Vitest coverage for learning registry/source-id behavior.
   - Add or update Playwright coverage for the learning page route, at least one interaction, mobile-width layout, and transition into the matching quiz source.

6. Verify before finishing.
   - Prefer `make check` before final response when practical.
   - For focused work, at least run:
     - `npm run test:focused -- tests/lib/learning.spec.ts`
     - `npm run e2e`
     - `make format-check`
     - `make lint`
     - `make types-check`
   - If the skill itself changed, run the system skill validator on `.codex/skills/author-learning-experience`.

## Quality Checklist

Before finishing, confirm:

- The page works under `/learn/[sourceId]` and links to `/?source=<SourceId>`.
- The page uses the app's normal routes, styling, navigation, and quiz source ids.
- The page is mobile-friendly.
- The page has meaningful interaction, not only reading or decorative toggles.
- When the topic benefits from exploration or visual reasoning, the page includes at least one domain-specific interactive model, simulation, calculator, diagram, or case workflow.
- The content follows the source material's actual conceptual structure.
- The page does not feel like a generic company landing page.
- The page does not feel like a copied transcript or long-form article.
- The page does not default to a generic sequence of cards when a richer topic-specific interface would teach better.
- Future generated pages would remain maintainable through shared primitives and a small registry entry.
