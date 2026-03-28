# Product Scope

## Product

- A Next.js + TypeScript quiz web app focused on deep learning, NLP, transformers, and reinforcement learning study content.
- Users can choose lecture series/books, specific sources, topic categories (`RL`, `DL`, `NLP`, `Math`), and difficulty ranges, answer multi-select quiz questions, and view explanations.
- Users can report a visible question with free-text comments; reports are stored centrally in Supabase Postgres and can be exported as JSON for offline review.
- Users can switch between:
  - `standard` mode (randomized from filtered pool)
  - `climb` mode (questions are biased toward current user Glicko rating with some randomness)
- The UI displays the current anonymous participant Glicko rating and rating deviation (RD).

## Current Behavior

- Main quiz UI lives in `/app/page.tsx`.
- Question banks are maintained in `lib/*` and `lib/lectures/*`.
- Quiz source registry includes MIT 6.S191 2025 lectures L1-L6, Crash Course Linear Algebra L1, and LangChain Deep Agents as selectable practice sources.
- Explanation endpoint exists at `/api/explain` and proxies to an LLM helper in `lib/llm/explain.ts`.
- Shared quiz state now uses an external Supabase Postgres database:
  - Each browser/device is treated as its own anonymous participant via a locally stored `participantId`.
  - Question difficulty is shared globally across participants.
  - Participant rating/climb behavior remains per device.
- Question reports are append-only in the shared database:
  - Each submit creates a separate report entry, even for the same question.
  - Each report stores the `questionId`, comment, report date, and a source/prompt snapshot for reviewer context.
  - Reports can be exported from the main UI as `quiz-question-reports.json`.
- Legacy local rating/report data is migrated once per participant into the shared database on first load, using an approximate replay for historical answer counts.

## Out of Scope (for now)

- No CI workflows are required yet unless explicitly requested.
- No backend service in this repository besides Next.js API routes.
- No login/auth system in this phase; anonymous participants are sufficient for the small trusted user group.

## Documentation Rule

- Any behavior change should be reflected in this `docs/*` folder in the same task.
