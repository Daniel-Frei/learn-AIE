# Product Scope

## Product

- The user-facing app name is `Learning AI`.
- A Next.js + TypeScript quiz web app focused on deep learning, NLP, transformers, reinforcement learning, supporting math, and selected life-science study content.
- An Expo React Native mobile app under `apps/mobile` lets users practice the same bundled question bank on iOS/Android while away from the desktop web UI.
- Users can choose lecture series/books, specific sources, topic categories (`RL`, `DL`, `NLP`, `Math`, `Life Science`), and question Elo ranges, answer multi-select quiz questions, and view explanations. The default question Elo filter range is the full supported `0` to `3000` range.
- The quiz reveals each question's shared raw rating as its visible "Elo" label only after the user submits an answer, and shows a per-question timer that starts at zero and caps at 3 minutes.
- Users can report a visible question with free-text comments; new reports are stored centrally in Supabase Postgres.
- Each question has an info icon after the prompt; hovering or focusing it shows the short source-context sentence so randomized mixed-source practice still exposes the lecture/chapter context without taking persistent vertical space.
- Users can switch between:
  - `standard` mode (randomized from filtered pool)
  - `climb` mode (questions are biased toward current user Glicko rating, use at least 10 targeted candidates when available, and select 20% of questions randomly from the active filtered pool regardless of user Elo)
- The UI displays the current anonymous participant Glicko rating and rating deviation (RD). The filter panel includes a reset button that returns the participant rating to the default Glicko state without clearing shared question ratings or question-report counts. Resetting the participant rating does not replace the currently open question; the new rating affects the next question selection after the user advances.

## Current Behavior

- Main quiz UI lives in `/app/page.tsx`.
- On Windows, users can install a per-user Start Menu shortcut named `Learning AI`; launching it refreshes stale local Next.js dev artifacts, then runs `make start` from the cloned repository.
- Question banks are maintained in `lib/*` and `lib/lectures/*`.
- Quiz source registry includes MIT 6.S191 2025 lectures L1-L6, Crash Course Linear Algebra L0-L5, Biology & Chemistry for Life Science L0-L5, AI Agents Code as Agent Harness, and LangChain Deep Agents as selectable practice sources.
- Crash Course Linear Algebra L0 provides prerequisite AP/A-level-style linear algebra practice on notation and terminology, coordinates, vector magnitude and direction, unit vectors, dot products, projection, span, basis, matrices, determinants, systems, and rank.
- Crash Course Linear Algebra L1 includes supplemental applied math practice on vectors, norms, dot products, cosine similarity, and matrix-vector multiplication for A-level/AP-style reinforcement.
- Crash Course Linear Algebra L2 includes 60 questions with applied matrix-transformation practice on composition, shape reasoning, geometric transformations, rank, LoRA, transpose, symmetry, attention, and more rigorous computation.
- Crash Course Linear Algebra L3 includes 60 questions covering derivatives, partial derivatives, gradients, gradient descent, learning rates, chain rule, backpropagation, matrix-gradient shape intuition, and applied optimization calculations.
- Crash Course Linear Algebra L4 includes 60 questions covering eigenvectors, eigenvalues, covariance, PCA, dimensionality reduction, SVD, low-rank approximation, embeddings, attention compression, LoRA, and applied decomposition reasoning.
- Crash Course Linear Algebra L5 includes 60 questions synthesizing attention, Q/K/V projections, neural networks as matrix stacks, RL value functions, optimization landscapes, and why linear algebra dominates modern AI systems.
- Biology & Chemistry for Life Science L0 provides 60 prerequisite questions on core biology, chemistry, and AP/A-level basics needed before the course, skewing toward easy/medium practice rather than an equal static difficulty split: atoms, molecules, bonds, pH, water, macromolecules, amino acids, proteins, cells, membranes, metabolism, genetics, evolution, immunity, disease, drugs, biomarkers, biotechnology, AI, and basic evidence reasoning.
- Biology & Chemistry for Life Science L1 covers chemistry-of-life fundamentals: atoms, bonds, water, hydrophobic effects, biological macromolecules, proteins, enzymes, ATP, metabolism, and structure-function reasoning.
- Biology & Chemistry for Life Science L2 covers cells as organized information-processing systems: organelles, membranes, transport, gradients, signaling, feedback, division, apoptosis, cancer, and immunity.
- Biology & Chemistry for Life Science L3 covers genetics, proteins, and regulation: DNA, genes, central dogma, gene expression, transcription factors, epigenetics, mutation, evolution, CRISPR, mRNA, gene therapy, and synthetic biology.
- Biology & Chemistry for Life Science L4 covers physiology, disease, and pharmacology: homeostasis, nervous/cardiovascular/endocrine coordination, disease as disrupted regulation, receptors, agonists/antagonists, dose response, PK/PD, side effects, biomarkers, and precision medicine.
- Biology & Chemistry for Life Science L5 covers biomedical systems, biotechnology, and evidence: infection, host response, vaccines, antimicrobials, resistance, recombinant medicines, therapeutic modalities, diagnostics, biomarkers, model systems, translational limits, concise clinical-evidence context, and AI in biomedicine.
- Clinical trials and research evidence are treated as context inside the Biology & Chemistry for Life Science crash course, not as the capstone subject; deeper trial-design coverage remains in the separate Clinical Trials Crash Course.
- Clinical Trials Crash Course L1-L2 covers why clinical trials exist and how trials are designed: causal inference, placebo effects, natural recovery, regression to the mean, confounding, selection/observer/publication bias, evidence hierarchy, sponsors, investigators, CROs, regulators, patients, PICO(T), randomization methods, blinding, endpoints, Phase I-IV development, and internal versus external validity.
- AI Agents Code as Agent Harness covers 60 questions on code as executable, inspectable, and stateful harness infrastructure: reasoning, acting, environment modeling, planning, memory, tool use, Plan-Execute-Verify control, harness optimization, multi-agent orchestration, applications, and open problems.
- Explanation endpoint exists at `/api/explain` and proxies to an LLM helper in `lib/llm/explain.ts`.
- Mobile uses local-first profile sync:
  - Bundled questions, filters, timers, answer checking, generic explanations, and local rating updates work without network.
  - Supabase Auth user ids are used as durable `participant_id` values for profile-linked ratings and reports.
  - Participant Glicko rating, shared question ratings, answer attempts, and signed-in question reports sync directly with Supabase when online.
  - Mobile does not depend on a developer PC or home-network server for ratings/report sync.
  - Detailed AI explanation chat still requires a reachable API host because the OpenAI key must not ship in the mobile app.
- Shared quiz state now uses an external Supabase Postgres database:
  - Each browser/device is treated as its own anonymous participant via a locally stored `participantId`.
  - Question difficulty is shared globally across participants and remains a Glicko-2 style rating under the hood, even though the UI labels it as Elo.
  - New unanswered question ratings are seeded from difficulty labels close to the neutral rating: easy `1400`, medium `1500`, and hard `1600`. Once answer data exists for a question, the persisted rating is used and the initial seed only remains as the starting point whose influence fades as more answers accumulate.
  - Participant rating/climb behavior remains per device. Resetting the participant rating preserves global question difficulty data and marks legacy migration complete so old browser-local ratings are not re-imported after reset.
- Answer scoring is still binary at the core, but rating updates are weighted by response time and mistake count so fast, clean answers move ratings more than slow or messy ones. Response speed is treated as a meaningful but incomplete skill signal: the response-time weight stays full through the first 20 seconds, then scales down linearly to a 40% maximum reduction at the 3-minute timer cap.
- A single answer has bounded rating exchange. Before answer weighting, each participant/question side moves by at least `2` rating points when the result would otherwise move it by less, and by at most `100` rating points when the result would otherwise create a larger jump; those bounds scale with the response-time and mistake-count weight.
- Question answer attempts store the elapsed time and mistake count alongside the binary result for later auditability.
- Question reports are append-only in the shared database:
  - Each submit creates a separate report entry, even for the same question.
  - Each report stores the `questionId`, comment, report date, and a source/prompt snapshot for reviewer context.
  - Mobile clients can read only report ids and question ids for counts, not report comments or prompt snapshots.
- Legacy local rating data is migrated once per participant into the shared database on first load, using an approximate replay for historical answer counts.
- Legacy browser-local question reports and old mobile queued local reports are ignored so reporting starts from the shared database state.

## Out of Scope (for now)

- No CI workflows are required yet unless explicitly requested.
- No backend service in this repository besides Next.js API routes.
- No login/auth system in this phase; anonymous participants are sufficient for the small trusted user group.
- No mobile file import/export for ratings or reports in the first mobile version.

## Documentation Rule

- Any behavior change should be reflected in this `docs/*` folder in the same task.
