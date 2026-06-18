---
name: author-learning-experience
description: Design, implement, and register high-quality Learning AI web learning experiences from lectures, chapters, transcripts, slides, PDFs, papers, notes, or topic overviews. Use when the user asks for an interactive learning page, subpage, teaching experience, explorable lesson, simulation, visual reasoning lab, source-material preparation page, or richer UX around this repo's quiz/learning surface.
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

Important project lesson: older Learning AI learning pages are generally not the
quality bar. They may be useful for routing and tests, but they are often too
card-like, too visually uniform, and too shallow as learning experiences. Future
pages should aim closer to a bespoke learning studio, staged simulator, or
source-native explorable when the material supports it.
</overview>

<core-rules>
- Choose the delivery mode first: integrated quiz-prep page, standalone in-repo
  subpage, experimental lab, or lightweight explorable article. Do not force a
  quiz-prep shape when the user explicitly asks for an independent subpage.
- Design the ideal learning experience first, then fit it into the chosen
  delivery mode.
- Do not use this repo's existing learning pages as the main comparison point
  for quality. Assume older pages may be stale or under-ambitious. Compare
  against strong source-native web learning experiences and any user-provided
  exemplar instead.
- Do not copy the structure, rhythm, visual texture, dark-panel style, or
  component mix of previous learning pages unless that shape is genuinely the
  best fit for the new material.
- Existing pages are useful for route, registry, app shell, Tailwind style,
  testing, and accessibility conventions; they are not evidence that future
  pages should share the same skeleton, rhythm, or primitive mix.
- Route registration, docs, quiz transitions, and tests are support work. They
  must not become the main design constraint or shrink the learning experience.
  If no quiz source exists, prefer a strong standalone route over an under-built
  integrated-looking page.
- Keep enough of the app's general visual language for coherence, but do not
  default to shared learning primitives. `LearningHero`, generic cards, checks,
  and formula blocks are optional support pieces, not the page architecture.
  Prefer page-local components when they let the topic teach itself better.
- Watch for template drift: agents often repeat panels, heroes, cards, and
  steppers because they are familiar integration-safe shapes, not because the
  source material calls for them. Counter this by choosing the source-native
  learning object before choosing components.
- Give the agent design freedom, but require a defensible concept. A page should
  start from something this material uniquely lets the learner inspect, predict,
  manipulate, compare, calculate, diagnose, or decide.
- Build from a concept-introduction floor. Every core term, control, unit,
  symbol, architecture part, or algorithm name that the page relies on should
  be introduced in simple learner-facing language before it is used in a dense
  lab, formula, comparison, or check. Learners who already know the concept can
  scroll past it; learners who do not know it should not be forced to infer it
  from context.
- Use the source material's concepts, examples, formulas, diagrams, cases, and
  misconceptions, but do not copy a transcript or slide order into a long article.
- Use source titles, sequence labels, and document types for navigation,
  metadata, breadcrumbs, or provenance; do not make instructional copy lean on
  phrases such as "this lecture", "Lecture N", "the slides", or "the chapter"
  when the page can teach the concept directly.
- Prefer one strong topic-specific interactive object over many generic cards
  when the material has a mechanism, system, workflow, calculation, tradeoff,
  spatial structure, timeline, evidence interpretation, or model behavior.
- Use visual and interactive representation when it clarifies the mental model,
  not to satisfy a quota. Text-only sections are fine when text is genuinely the
  strongest medium; invisible mechanisms, tradeoffs, and structures usually need
  something inspectable.
- For new isolated learning pages, do not let "small/reviewable" shrink the
  learning ambition. Keep the work coherent and scoped to the page, but allow a
  larger custom interaction, canvas, asset set, or control surface when that is
  the clearest way to teach the material.
- "Small and reviewable" means avoid unrelated refactors and broad app churn; it
  does not mean a new learning experience should be visually plain, short,
  card-based, or limited to one shallow interaction.
- It is acceptable for a strong page to look unlike the rest of the learning
  pages, as long as navigation, accessibility, responsiveness, and maintainability
  are still handled deliberately.
- Keep the implementation maintainable, accessible, mobile-friendly, and testable.
- Runtime dependencies are allowed when they materially improve the central
  learning experience. Prefer existing dependencies and browser-native APIs when
  they are enough, but do not reject a better teaching design solely because it
  needs a focused visualization, simulation, animation, or math/rendering
  library. Keep dependencies minimal, explain why they are needed, and document
  alternatives considered.
- Icons, custom SVG/CSS illustrations, lightweight animation, visual encodings,
  and domain-specific controls materially affect UX. Use the repo's icon library
  when present; for this repo `lucide-react` is an approved focused runtime
  dependency for learning experiences. Do not let the absence of an existing
  icon import push the page toward text-only panels.
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
     no secrets, and tests for behavior. For new standalone or page-local
     learning surfaces, larger cohesive changes are acceptable when the learning
     model needs them.
   - For standalone in-repo pages, still provide a clear route, browser-visible
     verification, and product-doc note describing what was added.

2. Gather integration context without anchoring the design.
   - Read `docs/product-scope.md`, `docs/team-preferences.md`, `lib/learning.ts`,
     `lib/quiz.ts`, `app/learn`, `components/learning`, and the relevant
     question/source files when the page is integrated with quiz practice.
   - For standalone or experimental pages, inspect only the app conventions that
     affect routing, styling, accessibility, and verification. Avoid letting
     existing learning pages decide the teaching shape.
   - If the user references or provides an exemplar, inspect it before choosing
     the design. Treat high-quality external/local exemplars as UX benchmarks,
     not as files to copy mechanically.
   - Use the existing `SourceId` as the learning-page key whenever the material
     maps to an existing quiz source and the delivery mode is integrated quiz
     prep.
   - Use the quiz source's `seriesId` for course navigation so `/learn` can stay
     a course index and `/learn/[seriesId]` can list that course's learning
     experiences.
   - Inspect existing learning pages only to understand conventions and avoid
     accidental duplication. Do not use them as default layouts or quality
     benchmarks, and actively look for visual or interaction repetition that
     would make the new page feel like another instance of the same template.
   - If docs and implementation disagree, flag the conflict and choose the
     documented behavior unless the user explicitly updates the requirement.

3. Study the source material.
   - Source material may be a lecture, chapter, transcript, overview, slide deck,
     PDF, paper, notes, dataset, code example, or topic brief.
   - Identify the core learning job: what the student should be able to predict,
     explain, calculate, compare, diagnose, trace, or decide after the page.
   - Create a concept inventory before designing the UI: the essential terms,
     symbols, units, algorithms, architecture parts, metrics, examples, and
     misconceptions a learner must understand to follow the experience.
   - Mark which concepts are prerequisites, which are introduced by the page,
     and which are only used later as optional extension material. Concepts
     introduced by the page need a simple explanation at first use.
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
     `visual/interaction identity`, and `what existing pattern it deliberately
     avoids`.
   - Favor a continuous central object when possible: a simulator, annotated
     architecture, editable example, decision lab, evidence reader, process
     model, timeline, calculator, parser, debugger, map, or comparison workbench.
   - Prefer a full staged experience over a shallow toggle panel when the topic
     has a workflow. A strong page may reveal information over time, let the
     learner construct an intermediate artifact, score or critique decisions,
     and debrief the result.
   - If using a hero visual, make it a self-explanatory first micro-model or a
     concrete preview of the central object. Do not use abstract bullets,
     arrows, meters, or charts unless the surrounding copy makes their role and
     relationship clear without prior context.
   - Before drafting the first screen, ask what would make it recognizably about
     this source even if the title were hidden. If the answer is mostly labels,
     choose a more specific opening object or task.
   - If all candidate shapes resemble previous learning pages, pause and look for
     a more topic-native form.
   - If a source-native design would benefit from a richer data model, page-local
     CSS, icons, SVG, or separated reasoning logic, use those tools. The data
     model should serve the UX; do not keep it thin merely to fit one TSX file.
   - If the best design would be substantially better as a standalone or more
     immersive subpage than as a standard learning-card flow, say so and either
     implement that mode when allowed or explain the compromise.
   - A compact explorable article is acceptable for text-first conceptual
     material, but it should be a deliberate choice, not the fallback produced by
     shared components.

5. Design the teaching flow.
   - Start with the core problem, intuition, or concrete situation, not a
     definition dump.
   - Write page-body explanations as self-contained concept instruction. Source
     sequence labels are useful for orientation, but learners should not need to
     keep seeing "the lecture says..." to understand what the page is teaching.
   - Introduce concepts before using them. For each core concept, include a
     short plain-language explanation, an example, a visual cue, or a small
     interaction that gives the learner enough footing to continue. Do not rely
     on a term appearing in the source material as proof that it is already
     understood.
   - Prefer simple explanations over compressed expert summaries. A good
     introduction can be one sentence, a labeled diagram, a tiny worked example,
     or a low-friction interaction. If the learner already knows it, scrolling
     past that introduction is cheaper than recovering from a missing concept.
   - Make the beginning, middle, and end clear: orient, introduce the needed
     vocabulary and notation, manipulate/interpret, consolidate, then transition
     into practice.
   - Build concepts in dependency order. Do not put a dense lab before the
     learner has enough context to understand its controls, labels, and state.
     If a lab must appear early, it should teach one concept at a time through a
     guided progression or inline legend.
   - For hard technical material, introduce algorithm names, roles, equations,
     and symbols before the learner uses them in an interaction. If a lab
     depends on notation such as advantages, reference policies, logits,
     likelihoods, rewards, or latent variables, define the simplified mental
     model and the full core equation first.
   - Avoid unexplained control labels. Terms such as greedy, top-k, top-p,
     temperature, entropy, seed, guidance, latent variable, base probability,
     used probability, and denoising step need either prior introduction or
     concise contextual explanation at first use.
   - Include active moments where the learner predicts, manipulates, compares,
     sorts, annotates, debugs, estimates, reads evidence, or checks a misconception.
     One click that swaps static prose is usually not enough for a central
     interaction unless the material is genuinely simple.
   - Put feedback close to the relevant idea when that helps learning. Checks,
     prompts, and questions can be embedded beside the concept they assess
     instead of always being batched into a separate page rhythm.
   - Include worked examples for mathematical, technical, procedural, or
     evidence-heavy topics.
   - For interactive numeric outputs, show the calculation path near the output:
     the governing formula, the selected input values, and enough intermediate
     arithmetic for the learner to reconstruct the result.
   - When source material provides a canonical formula, benchmark metric,
     estimator, or algorithm objective, present that source-aligned form before
     or beside simplified intuition. If a toy formula is useful, label it as an
     approximation and explain what it omits.
   - Quantitative tradeoffs often deserve an inspectable model, such as a
     calculator, allocation surface, counterfactual, or small lab. Use the
     representation that makes the tradeoff easiest to reason about.
   - When a section is about a mechanism, make the mechanism visible through
     state, flow, structure, trace, comparison, or another concrete
     representation rather than relying only on labels and prose.
   - Use visual structure that belongs to the domain. Architecture can be
     spatial, math can be manipulable, clinical/statistical material can be an
     evidence reader, history can be a timeline, code can be a trace/debugger,
     biology can be a pathway/case model, and NLP/LLM topics can be token,
     attention, or generation workbenches.
   - Build a visual identity for the source. For medicine this may mean a
     clinical studio, patient monitor, staged encounter timeline, evidence feed,
     diagnosis ranking board, or care-pathway map; for other domains, choose an
     equally specific visual language.
   - Keep sections causally connected. Each major interaction should prepare for
     the next one or reuse an established mental model; avoid detached labs that
     happen to share a topic label.
   - Make the visual scan order match the learner journey. In a staged lab,
     prerequisite inputs should appear before the representation, diagnosis,
     answer, output, or plan that depends on them. Related material may sit
     side-by-side inside one step, but do not place later conclusions beside
     earlier evidence in a way that implies the learner should use them together
     before the prerequisite work is done.
   - When teaching latent variables, distinguish sampled hidden factors,
     inferred hidden factors, and user-controlled scenario settings. Do not make
     a UI imply that real latent variables are directly observable or manually
     controlled unless that distinction is explicitly part of the lesson.
   - Explain mathematical notation at the point of first use, especially for
     crash-course material. Formula boards are useful as recap, but symbols such
     as \(x_T\), \(\mathcal{N}(0,I)\), \(I\), \(z\), and \(p_\theta\) should
     not first become meaningful at the end of the page.
   - Avoid visible instructional filler that explains the UI instead of teaching
     the concept. Controls and state should make the task understandable.
   - Keep process steppers lean. If a staged flow is useful, make each step
     change a meaningful visible state, constraint, output, or learner decision;
     otherwise simplify it into prose or integrate it into the central model.
   - End with a concise recap and a transition into the matching MCQs. The recap
     should summarize ideas the page already taught, not introduce core concepts
     for the first time.

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
     page shape, and do not import them by default just because the page is under
     the learning route.
   - Page-local custom components may have nontrivial typed state/config when
     that state powers the teaching model. Keep them deterministic and readable.
   - Split page-local data, reasoning helpers, styles, and tests when that makes
     the experience easier to build and maintain. A single monolithic TSX file is
     not required.
   - For bespoke visuals, use stable dimensions, responsive constraints,
     keyboard-accessible controls, visible state changes, and text alternatives
     or labels.
   - Use iconography and custom visual models deliberately. Prefer `lucide-react`
     icons when available; otherwise use focused custom SVG/CSS or add a minimal
     approved icon dependency when it materially improves comprehension and UX.
   - Use `MathText` for formulas and existing Tailwind conventions where they fit.
   - Render inline mathematical notation through `MathText inline` or another
     KaTeX-backed component. Do not leave notation in headings, labels, prose,
     or explanatory notes as raw ASCII such as `s_t`, `A_t`, `pi_theta`,
     `r_phi(x,y)`, or `N(0,I)` unless the point is to teach plain-code syntax.
   - For display formulas, pass valid LaTeX through `MathText` or `FormulaBlock`;
     prefer TSX expression props with `String.raw`, for example
     ``formula={String.raw`\[...\]`}``, so LaTeX commands are not double-escaped.
   - For inline formulas, use valid inline delimiters with `String.raw`, for
     example `<MathText inline text={String.raw`\(\pi_\theta(a_t\mid s_t)\)`} />`.
   - Keep custom local data/config objects page-local unless they are clearly
     reusable.
   - Do not hard-code unrelated app state. If adding a runtime dependency, keep
     it focused on the central learning object, document why existing tools were
     insufficient, and update tests and docs accordingly.
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
   - Add coverage for the central learning journey, not only route existence.
     Useful checks include staged reveal behavior, learner-built artifacts,
     score/feedback changes, navigation anchors, visible visual models, and a
     no-horizontal-overflow mobile check.
   - Add unit tests for page-local reasoning helpers when the experience includes
     scoring, ranking, probability updates, parsing, simulation, or other logic.
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
   - Run a frontend UX review loop before finalizing a new learning page. Use
     the repo's `frontend-ux-review` skill when available, or perform an
     equivalent browser screenshot review across desktop and mobile. Passing
     tests is not enough if the page still feels like a generic card/check flow.
   - Run a learner-scroll UX check before finalizing. Imagine a learner reading
     from top to bottom without prior expert knowledge. At each section, ask:
     what concept did they just learn, why does the next section follow, are the
     next controls/labels already explained, and is there any sudden jump from
     vocabulary to dense application? If the flow feels like a checklist,
     disconnected gallery, or hidden to-do list, revise the order and
     explanations before closing.
   - Include an intermediate laptop viewport in browser review, such as
     `1280 x 800`, when the page has dense controls, dashboards, labs, hero
     text, or multi-column layouts. Check for awkward text breaks, clipped
     controls, and page-level horizontal overflow at that size, not only at a
     wide desktop and a narrow mobile viewport.
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
- Template drift: reusing the same hero, card grid, overview panel, comparison,
  formula block, check, and recap rhythm because it worked before.
- Legacy anchoring: treating this repo's older learning pages as the quality bar
  instead of as possibly underwhelming integration references.
- Designing from available components before identifying the source-native
  learner task.
- Letting route registration, quiz-source absence, docs, or tests dominate the
  design enough that the learning experience becomes conservative.
- Treating "small/reviewable" as a reason to avoid a rich, page-local simulator,
  custom visual system, or focused dependency.
- Using a hero visual with arrows, bullets, bars, or status marks that is hard
  to interpret before the learner knows the topic.
- Treating interaction as decorative toggles over static prose.
- Building one shallow central object when the material calls for a staged
  simulator, evidence reader, construction task, decision lab, or guided studio.
- Avoiding icons, custom SVG/CSS, animation, or visual affordances so aggressively
  that the page becomes text-heavy and visually generic.
- Visual tokenism: using formulas, charts, meters, or diagrams as proof of
  richness when they do not make the core idea easier to reason about.
- Checklist-driven design: adding one of every familiar learning primitive
  instead of choosing a coherent experience shape.
- Dropping learners into a dense lab before the page has introduced the terms,
  controls, units, and labels that the lab uses.
- Assuming that a learner knows a term because the source material used it, or
  because an expert reader would find it obvious.
- Compressing foundational concepts into labels or headings while spending the
  real page space on later applications that depend on those missing concepts.
- Showing raw math notation in learner-facing prose, headings, labels, or notes
  when it should be rendered, especially underscores and Greek-letter names such
  as `s_t`, `A_t`, `pi_theta`, or `r_phi`.
- Changing numbers in an interaction without showing the formula and arithmetic
  that produced the result.
- Designing around MCQ topic labels instead of the source material's central
  learning difficulty.
- Turning slides or transcript sections into a long scroll of summaries.
- Repeatedly naming the source artifact in instructional copy instead of
  explaining the concept itself.
- Replacing a source's canonical metric, estimator, or objective with a simpler
  formula without labeling the simplification or showing the source-aligned
  version.
- Using shared primitives because they are available rather than because they
  teach this topic well.
- Hiding the main teaching model below many introductory cards.
- Splitting a topic into disconnected sections whose order does not build a
  cumulative mental model.
- Arranging a workflow lab as columns of component types, so later outputs,
  diagnoses, answers, or plans appear beside earlier inputs and confuse the
  learner's intended play-through order.
- Letting a final recap or formula board carry explanations that should have
  appeared before the learner used the concept.
- Making latent-variable controls look like direct manual control over real
  hidden variables without explaining the modeling distinction.
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
- The page has a source-specific visual identity and does not simply inherit the
  old dark-panel/card texture from earlier Learning AI pages.
- The experience shape is topic-native and not just copied from previous pages.
- The page would still make sense if no previous Learning AI page existed; any
  similarity to earlier pages is justified by the topic, not convenience.
- The first viewport introduces a source-native object, problem, contrast, or
  task, not only a reusable wrapper with different labels.
- Shared primitives are used only where they improve this page; page-local
  components carry the topic-specific teaching model.
- Any repeated scaffold from another page earns its place by serving this
  material's learner task.
- Any hero visual is understandable as a concrete preview, model, or prompt for
  the lesson rather than unexplained decoration.
- The learner actively predicts, manipulates, compares, calculates, annotates, or
  interprets something important.
- If the topic has a workflow, the page supports a staged journey where learner
  actions change visible state, feedback, score, artifact quality, or downstream
  interpretation.
- The chosen level of visualization/interactivity matches the source's actual
  learning difficulty; absence of visuals for a hard invisible mechanism is a
  deliberate design choice, not a default.
- Feedback, checks, or practice prompts appear where they best support the
  learner's current mental model.
- The scroll order forms a learning progression: concepts and notation appear
  before controls or formulas that depend on them.
- The learner-scroll UX check has been performed: each section naturally follows
  the previous one, introduces what the next section needs, and avoids sudden
  jumps into unexplained vocabulary, controls, or dense labs.
- Core concepts have simple first-use explanations. Skipping those explanations
  is easy for knowledgeable learners, but missing them would not block a learner
  who is new to the topic.
- For workflow-based labs, the DOM order and visible top-left-to-right scan order
  reflect the intended play-through sequence; later conclusions are not visually
  promoted before prerequisite evidence or construction steps.
- Lab labels, metrics, and controls are explained at first use; there are no
  unexplained terms like "base", "used", "seed", or "I" in core interactions.
- Mathematical notation in text, headings, labels, and notes renders as KaTeX
  inline math rather than raw underscore-heavy ASCII; display formulas render as
  KaTeX rather than visible raw `$$...$$`, `\[...\]`, or escaped LaTeX.
- Numeric interactions show their formula and selected-value arithmetic close to
  the changing output.
- Canonical formulas, benchmark metrics, or algorithm objectives from the source
  are represented accurately; any simplified formulas are explicitly labeled as
  approximations or intuition aids.
- Page-body copy teaches the concept directly rather than repeatedly referring
  to source order or artifact labels such as lectures, slides, or chapters.
- If latent variables appear, the page distinguishes hidden model variables from
  user-selected conditions, prompts, or scenario knobs.
- The content follows the source material's conceptual structure without copying
  the source order verbatim.
- Formulas render as KaTeX, not visible raw `$$...$$` or escaped LaTeX.
- The page does not feel like a generic landing page, copied transcript, or
  generic sequence of cards/checks.
- Browser or screenshot review confirms the page feels like a compelling
  learning product, not merely a correct implementation.
- Future maintainers can understand the page-local state and tests.
</review-checklist>
