# Learning AI

Learning AI is an open-source study app for people who want to get sharper at modern AI, not just skim the vocabulary. It turns deep learning, NLP, transformers, reinforcement learning, and supporting math material into focused multi-select practice with ratings, explanations, source filters, and question-quality feedback.

The project is built for learners who want a serious practice loop and for contributors who want a practical Next.js + TypeScript codebase with real product surface area: quiz UX, rating logic, shared state, API routes, mobile sync, and a growing question bank.

## Why It Exists

AI courses and books are dense. Learning AI gives that material a practice layer:

- Pick specific lecture series, books, question sets, topics, and difficulty ranges.
- Practice in `standard` mode with randomized filtered questions.
- Practice in `climb` mode with questions biased toward your current Glicko rating.
- See a visible question "Elo" label backed by shared Glicko-style difficulty data.
- Track your own anonymous participant rating and rating deviation.
- Get explanations for answers, with optional AI-powered follow-up help.
- Report unclear or incorrect questions so the bank can improve over time.
- Use the Expo mobile app for local-first practice away from the desktop web UI.

## What Is In The Repo

- A Next.js + TypeScript web quiz app in `app/`.
- Shared UI and quiz logic in `components/` and `lib/`.
- Bundled question banks in `lib/` and `lib/lectures/`.
- API routes for explanations, answer persistence, shared quiz state, and question reports.
- Supabase migrations for shared ratings, reports, attempts, and mobile profile sync.
- An Expo React Native mobile app in `apps/mobile/`.
- Vitest coverage under `tests/` and Playwright flows under `e2e/`.
- Product and API documentation in `docs/`.

## Current Question Sources

The bundled question registry currently includes practice material for:

- MIT 6.S191 2025 lectures L1-L6.
- Crash Course Linear Algebra L1.
- LangChain Deep Agents.

The question bank is intentionally part of the repo so improvements can be reviewed, tested, and discussed like code.

## For Contributors

Useful contribution areas include:

- Adding or improving question sets.
- Tightening explanations and prompt wording.
- Improving quiz UX, filters, and mobile practice flows.
- Expanding tests around rating behavior, API validation, reporting, and source selection.
- Hardening Supabase sync and export workflows.
- Keeping `docs/*` accurate as product behavior evolves.

The product source of truth is the `docs/` folder. If a code change alters behavior, update the relevant doc in the same change.

## Quick Start

Install dependencies and start the web app:

```bash
make ci
make start
```

Open [http://localhost:43191](http://localhost:43191).

For iterative development with an existing install:

```bash
make install
make start
```

## Mobile App

The mobile app lives in `apps/mobile` and uses the same bundled question bank with local-first profile behavior.

```bash
make mobile-start
```

Then launch the target platform:

```bash
make mobile-android
make mobile-ios
make mobile-web
```

## Environment Variables

Copy `.env.template` to `.env.local` for local development and fill in only the values you need.

Core variables:

- `OPENAI_API_KEY` enables detailed explanation chat.
- `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` configure client Supabase access.
- `SUPABASE_SERVICE_ROLE_KEY` enables server-side shared quiz data routes.
- `QUESTION_REPORT_EXPORT_TOKEN` protects production question-report export.
- `EXPO_PUBLIC_SUPABASE_URL` and `EXPO_PUBLIC_SUPABASE_PUBLISHABLE_KEY` configure mobile profile sync.
- `EXPO_PUBLIC_QUIZ_API_BASE_URL` points mobile explanation chat at a reachable Next.js API host.

The app can still load locally without reachable Supabase by using an in-memory server fallback for the current process, but that fallback is not durable.

## Verify Changes

Prefer the full gate before finishing a change:

```bash
make check
```

Focused commands:

```bash
make format-check
make lint
make types-check
make test
npm run e2e
```

Mobile checks:

```bash
make mobile-lint
make mobile-types-check
```

## Windows Start Menu Launcher

On Windows, install a per-user Start Menu shortcut named `Learning AI` from a cloned checkout:

```powershell
make install-windows-start-menu
```

Or with npm:

```powershell
npm run windows:start-menu:install
```

After installation, press Windows, search for `Learning AI`, and press Enter. The shortcut opens a PowerShell window in this repository and runs `make start`. It expects `make` to be available on `PATH`.

To remove the shortcut:

```powershell
make uninstall-windows-start-menu
```

## Public Repository Safety

This repository is intended to be safe to publish publicly. Keep real credentials in local or deployment environment variables only.

Do not commit:

- `.env.local` or other local secret files.
- Generated server logs.
- Supabase service-role keys.
- OpenAI API keys.
- Participant-specific exports or machine-specific data.

Client-exposed Supabase publishable keys are expected for web and mobile clients, but they are only safe with the committed Supabase Row Level Security policies applied.

## Documentation

- `docs/product-scope.md` describes product behavior and scope.
- `docs/api-contract.md` documents API routes and payload shapes for backend/frontend integration.
- `docs/team-preferences.md` captures durable implementation and process preferences.

Start there before making larger behavior changes.
