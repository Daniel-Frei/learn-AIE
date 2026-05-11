# Product Scope

## Product

- A Next.js + TypeScript quiz web app focused on deep learning, NLP, transformers, and reinforcement learning study content.
- An Expo React Native mobile app under `apps/mobile` lets users practice the same bundled question bank on iOS/Android while away from the desktop web UI.
- Users can choose lecture series/books, specific sources, topic categories (`RL`, `DL`, `NLP`, `Math`), and question Elo ranges, answer multi-select quiz questions, and view explanations.
- The quiz shows each question's shared raw rating as its visible "Elo" label and shows a per-question timer that starts at zero and caps at 3 minutes.
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
- Mobile uses local-first profile sync:
  - Bundled questions, filters, timers, answer checking, generic explanations, local rating updates, and report drafting work without network.
  - Supabase Auth user ids are used as durable `participant_id` values for profile-linked ratings and reports.
  - Participant Glicko rating, shared question ratings, answer attempts, and question reports sync directly with Supabase when signed in and online.
  - Mobile does not depend on a developer PC or home-network server for ratings/report sync.
  - Detailed AI explanation chat still requires a reachable API host because the OpenAI key must not ship in the mobile app.
- Shared quiz state now uses an external Supabase Postgres database:
  - Each browser/device is treated as its own anonymous participant via a locally stored `participantId`.
  - Question difficulty is shared globally across participants and remains a Glicko-2 style rating under the hood, even though the UI labels it as Elo.
  - Participant rating/climb behavior remains per device.
- Answer scoring is still binary at the core, but rating updates are weighted by response time and mistake count so fast, clean answers move ratings more than slow or messy ones.
- Question answer attempts store the elapsed time and mistake count alongside the binary result for later auditability.
- Question reports are append-only in the shared database:
  - Each submit creates a separate report entry, even for the same question.
  - Each report stores the `questionId`, comment, report date, and a source/prompt snapshot for reviewer context.
  - Reports can be exported from the main UI as `quiz-question-reports.json`.
- Legacy local rating/report data is migrated once per participant into the shared database on first load, using an approximate replay for historical answer counts.

## Out of Scope (for now)

- No CI workflows are required yet unless explicitly requested.
- No backend service in this repository besides Next.js API routes.
- No login/auth system in this phase; anonymous participants are sufficient for the small trusted user group.
- No mobile file import/export for ratings or reports in the first mobile version.

## Documentation Rule

- Any behavior change should be reflected in this `docs/*` folder in the same task.
