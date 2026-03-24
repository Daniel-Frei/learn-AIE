# Product Scope

## Product

- A Next.js + TypeScript quiz web app focused on deep learning, NLP, transformers, and reinforcement learning study content.
- Users can choose lecture series/books, specific sources, topic categories (`RL`, `DL`, `NLP`, `Math`), and difficulty ranges, answer multi-select quiz questions, and view explanations.
- Users can report a visible question with free-text comments; reports are stored locally in the browser and can be exported as JSON for offline review.
- Users can switch between:
  - `standard` mode (randomized from filtered pool)
  - `climb` mode (questions are biased toward current user Glicko rating with some randomness)
- The UI displays the current user Glicko rating and rating deviation (RD).

## Current Behavior

- Main quiz UI lives in `/app/page.tsx`.
- Question banks are maintained in `lib/*` and `lib/lectures/*`.
- Quiz source registry includes MIT 6.S191 2025 lectures L1-L6, Crash Course Linear Algebra L1, and LangChain Deep Agents as selectable practice sources.
- Explanation endpoint exists at `/api/explain` and proxies to an LLM helper in `lib/llm/explain.ts`.
- Question reports are client-side only in this version:
  - Each submit creates a separate report entry, even for the same question.
  - Each report stores the `questionId`, comment, report date, and a source/prompt snapshot for reviewer context.
  - Reports can be exported from the main UI as `quiz-question-reports.json`.

## Out of Scope (for now)

- No CI workflows are required yet unless explicitly requested.
- No backend service in this repository besides Next.js API routes.

## Documentation Rule

- Any behavior change should be reflected in this `docs/*` folder in the same task.
