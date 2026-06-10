---
name: author-learning-experience
description: Create and register Learning AI in-app learning experience pages from lecture, chapter, transcript, overview, slides, PDF, or topic material. Use when the user asks Codex to make a scrollable or interactive learning subpage that prepares students before existing multiple-choice questions, especially pages that should feel between slides, an explorable article, and a topic-specific teaching webpage inside this repo.
---

# Author Learning AI Learning Experiences

## Overview

Create web-first learning pages inside the existing Learning AI Next.js app. The output should prepare students for a registered quiz source before they answer MCQs, using shared app infrastructure plus topic-specific React where it improves teaching.

Do not create PowerPoints, standalone static sites, external iframes, or long copied lecture summaries.

## Workflow

1. Inspect the app before deciding the implementation.
   - Read `docs/product-scope.md`, `docs/team-preferences.md`, `lib/learning.ts`, `lib/quiz.ts`, `app/learn`, `components/learning`, and the relevant existing topic/question files.
   - Use the existing `SourceId` as the learning-page key whenever the material maps to an existing quiz source.
   - If docs and implementation disagree, flag the conflict and choose the documented behavior unless the user explicitly updates the requirement.

2. Read the source material.
   - Source material may be an overview, transcript, chapter, PDF, slide deck, or notes.
   - Identify the core learning goal, prerequisite assumptions, conceptual sequence, formulas, examples, likely misconceptions, and which MCQ concepts the page should prepare for.
   - Avoid copying the source structure verbatim when it would become a plain article.

3. Design the learning experience.
   - Start with the core problem or intuition, not a definition dump.
   - Use a clear beginning, middle, and end.
   - Include visual structure and short sections.
   - Include active moments where the learner taps, compares, predicts, sorts, adjusts a control, or answers a small check.
   - Include worked examples for mathematical, technical, or procedural topics.
   - Include at least one misconception check when likely confusions exist.
   - End with a recap and transition into the matching MCQs.

4. Implement inside the app.
   - Add or update metadata in `lib/learning.ts`.
   - Add the topic component under `components/learning/pages/`.
   - Use shared primitives from `components/learning/LearningPrimitives.tsx` when they fit; add new primitives only when they will help future pages stay consistent.
   - Add the page component to the route map in `app/learn/[sourceId]/page.tsx`.
   - Use `MathText` for formulas and existing Tailwind styling conventions.
   - For display formulas, pass valid LaTeX through `MathText` or `FormulaBlock`; prefer TSX expression props with `String.raw`, for example ``formula={String.raw`\[...\]`}``, so LaTeX commands are not double-escaped, and verify rendered pages show KaTeX output instead of visible raw `$$...$$` text.
   - Keep custom local data/config objects small and page-local unless they are clearly reusable.
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
- The page has interaction, not only reading.
- The content follows the source material's actual conceptual structure.
- The page does not feel like a generic company landing page.
- The page does not feel like a copied transcript or long-form article.
- Future generated pages would remain maintainable through shared primitives and a small registry entry.
