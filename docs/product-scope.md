# Product Scope

## Product

- A Next.js + TypeScript quiz web app focused on deep learning, NLP, transformers, and reinforcement learning study content.
- Users can choose question sources and difficulty ranges, answer multi-select quiz questions, and view explanations.

## Current Behavior

- Main quiz UI lives in `/app/page.tsx`.
- Question banks are maintained in `lib/*` and `lib/lectures/*`.
- Explanation endpoint exists at `/api/explain` and proxies to an LLM helper in `lib/llm/explain.ts`.

## Out of Scope (for now)

- No CI workflows are required yet unless explicitly requested.
- No backend service in this repository besides Next.js API routes.

## Documentation Rule

- Any behavior change should be reflected in this `docs/*` folder in the same task.
