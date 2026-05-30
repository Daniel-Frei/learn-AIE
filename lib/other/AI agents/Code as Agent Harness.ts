import { Question } from "../../quiz";

export const CodeAsAgentHarnessQuestions: Question[] = [
  {
    id: "code-agent-harness-q01",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the idea of code as an agent harness?",
    options: [
      {
        text: "Code can be the executable substrate through which a large language model (LLM) reasons, acts, and receives feedback.",
        isCorrect: true,
      },
      {
        text: "Code in an agent system can include tests, tool definitions, scripts, logs, traces, repositories, and formal specifications.",
        isCorrect: true,
      },
      {
        text: "The framing treats code only as the final artifact produced after an agent has already completed its work.",
        isCorrect: false,
      },
      {
        text: "The framing emphasizes executable, inspectable, and stateful artifacts inside the agent loop.",
        isCorrect: true,
      },
    ],
    explanation:
      "Code as an agent harness reframes code from a final generated answer into the operational medium of the agent system. It includes executable and machine-checkable artifacts that let the harness run actions, inspect state, preserve progress, and verify outcomes, so code is not limited to final source files.",
  },
  {
    id: "code-agent-harness-q02",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which parts commonly belong to an agent harness around a language model?",
    options: [
      {
        text: "Tools, APIs, sandboxes, validators, and permission boundaries.",
        isCorrect: true,
      },
      {
        text: "Memory systems, execution loops, and feedback channels.",
        isCorrect: true,
      },
      {
        text: "Only the model weights and no external software infrastructure.",
        isCorrect: false,
      },
      {
        text: "Runtime mechanisms that keep model outputs purely textual and disconnected from external actions.",
        isCorrect: false,
      },
    ],
    explanation:
      "An agent harness is the software layer that turns a stateless model call into a working agent process. It surrounds the model with tools, memory, validation, execution, permissions, and feedback; it does not keep model outputs disconnected from action, and model weights alone are not enough to provide long-running state management.",
  },
  {
    id: "code-agent-harness-q03",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which distinctions are useful when analyzing code-centric agent systems?",
    options: [
      {
        text: "Model-internal capabilities include reasoning, planning, perception, simulation, and evaluation abilities.",
        isCorrect: true,
      },
      {
        text: "System-provided harness infrastructure includes predefined tools, validators, sandboxes, memory, telemetry, and workflows.",
        isCorrect: true,
      },
      {
        text: "Agent-initiated code artifacts include temporary tools, tests, executable workflows, reusable skills, and intermediate program states.",
        isCorrect: true,
      },
      {
        text: "Agent-initiated code artifacts are interactive objects that can be executed, observed, revised, persisted, or shared during the task.",
        isCorrect: true,
      },
    ],
    explanation:
      "The framing separates what the model can do internally, what the surrounding system provides, and what the agent creates during the task. Agent-initiated code artifacts are interactive objects in the execution loop: they can be executed, observed, revised, persisted, and shared instead of remaining private neural state.",
  },
  {
    id: "code-agent-harness-q04",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statement best captures the scope boundary for the word code in code-as-harness systems?",
    options: [
      {
        text: "Code means only human-readable programming language source files and excludes tests, traces, tool schemas, and formal specifications.",
        isCorrect: false,
      },
      {
        text: "Code broadly means executable or machine-checkable artifacts, while raw perception, human intent, and latent model reasoning are not themselves code.",
        isCorrect: true,
      },
      {
        text: "Code includes every physical state in the world as long as an agent talks about that state.",
        isCorrect: false,
      },
      {
        text: "Code replaces perception, embodiment, human goals, and model reasoning rather than making selected aspects of them executable.",
        isCorrect: false,
      },
    ],
    explanation:
      "The boundary is broad but not metaphorical: programs, scripts, specifications, tests, schemas, traces, and logs can count when they are executable or machine-checkable. Raw perception or human intent may be serialized and acted on through code, but they are not code by themselves.",
  },
  {
    id: "code-agent-harness-q05",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe why code is useful as a harness interface?",
    options: [
      {
        text: "Executability lets the harness run model outputs and compare outcomes against constraints.",
        isCorrect: true,
      },
      {
        text: "Inspectability exposes intermediate computation, traces, and failures for diagnosis.",
        isCorrect: true,
      },
      {
        text: "Statefulness lets evolving program artifacts preserve progress across steps.",
        isCorrect: true,
      },
      {
        text: "Code is useful mainly because it hides state transitions from the harness.",
        isCorrect: false,
      },
    ],
    explanation:
      "The three key properties are executability, inspectability, and statefulness. A code-centered interface helps the harness run actions, observe what happened, diagnose failures, and preserve modifiable task state instead of hiding the transition from the system.",
  },
  {
    id: "code-agent-harness-q06",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly compare pure text reasoning with code-for-reasoning?",
    options: [
      {
        text: "Pure chain-of-thought reasoning can propose steps but may be unreliable for exact arithmetic, symbolic, or logical execution.",
        isCorrect: true,
      },
      {
        text: "Code-for-reasoning lets the model propose procedures while interpreters, solvers, or runtimes execute them.",
        isCorrect: true,
      },
      {
        text: "Executable traces and variable states are irrelevant to later reasoning and should be discarded before the harness observes them.",
        isCorrect: false,
      },
      {
        text: "Moving reasoning into code removes the need to inspect execution results.",
        isCorrect: false,
      },
    ],
    explanation:
      "Code-for-reasoning separates high-level procedure generation from low-level computation. The harness can execute programs and inspect traces or states as feedback, so discarding execution artifacts would throw away one of the main advantages over pure text reasoning.",
  },
  {
    id: "code-agent-harness-q07",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements correctly describe program-delegated reasoning?",
    options: [
      {
        text: "The model generates code that an external interpreter executes to produce grounded outputs.",
        isCorrect: true,
      },
      {
        text: "The model must internally compute every intermediate result without using a runtime.",
        isCorrect: false,
      },
      {
        text: "The approach can improve reliability by moving computation into structured execution traces.",
        isCorrect: true,
      },
      {
        text: "The approach is useful only when the final answer is a source-code file.",
        isCorrect: false,
      },
    ],
    explanation:
      "Program-delegated reasoning asks the model to write a procedure and asks a runtime to execute that procedure. It is useful because execution can ground arithmetic, symbolic, or procedural steps even when the user ultimately wants an answer rather than a program file.",
  },
  {
    id: "code-agent-harness-q08",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe formal verification and symbolic reasoning interfaces?",
    options: [
      {
        text: "Proof assistants and formal languages can make derivation steps machine-checkable.",
        isCorrect: true,
      },
      {
        text: "Symbolic solvers can serve as backends that constrain or verify model-generated reasoning.",
        isCorrect: true,
      },
      {
        text: "Formal verification interfaces can act as executable contracts for agent behavior or workflows.",
        isCorrect: true,
      },
      {
        text: "Formal methods are useful only if their outputs are never fed back to the model.",
        isCorrect: false,
      },
    ],
    explanation:
      "Formal and symbolic interfaces give the harness stronger checking machinery than ordinary prose. They can verify proof steps, check constraints, and feed back failures or obligations, so the model and verifier can participate in an iterative reasoning loop.",
  },
  {
    id: "code-agent-harness-q09",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe iterative code-grounded reasoning?",
    options: [
      {
        text: "It treats reasoning as a closed loop of generation, execution, verification, and refinement.",
        isCorrect: true,
      },
      {
        text: "It can use runtime feedback, compiler feedback, traces, or tests as process-level signals.",
        isCorrect: true,
      },
      {
        text: "It may optimize reasoning trajectories through reinforcement learning or process rewards.",
        isCorrect: true,
      },
      {
        text: "It can interleave multiple rounds of code execution with further reasoning in a persistent session.",
        isCorrect: true,
      },
    ],
    explanation:
      "Iterative code-grounded reasoning makes execution part of the reasoning trajectory rather than a single final check. Runtime outputs, traces, tests, and process rewards can all shape subsequent reasoning, and persistent sessions let the agent refine its work over multiple executions.",
  },
  {
    id: "code-agent-harness-q10",
    chapter: 1,
    difficulty: "easy",
    prompt: "Why is grounding a central challenge for code-for-acting?",
    options: [
      {
        text: "The harness must map high-level language intent into executable actions that respect environment constraints.",
        isCorrect: true,
      },
      {
        text: "Physical and digital environments can be partially observed, dynamic, and subject to delayed or silent failures.",
        isCorrect: true,
      },
      {
        text: "Executable action code replaces perception, controllers, APIs, and safety layers, so the harness no longer needs those components.",
        isCorrect: false,
      },
      {
        text: "Grounding is unnecessary when actions are represented as code because code cannot produce invalid state transitions.",
        isCorrect: false,
      },
    ],
    explanation:
      "Action code does not automatically make an action valid in a robot, GUI, API, or software environment. The harness still has to ground intent in affordances, permissions, controllers, API contracts, and validation signals because code calls into those components rather than replacing them.",
  },
  {
    id: "code-agent-harness-q11",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statement best describes grounded skill selection?",
    options: [
      {
        text: "It selects and composes executable skills according to both semantic relevance and environmental feasibility.",
        isCorrect: true,
      },
      {
        text: "It bypasses reusable skills and always asks the model to generate low-level motor or UI commands from scratch.",
        isCorrect: false,
      },
      {
        text: "It assumes ambiguity should never trigger clarification because the executable skill library is fixed.",
        isCorrect: false,
      },
      {
        text: "It works only for text-only question answering and not for embodied or GUI environments.",
        isCorrect: false,
      },
    ],
    explanation:
      "Grounded skill selection maps a language goal onto reusable executable capabilities while considering whether the environment can actually support the action. Systems in this family often combine semantic planning with affordance, feasibility, uncertainty, or clarification mechanisms.",
  },
  {
    id: "code-agent-harness-q12",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe programmatic policy generation?",
    options: [
      {
        text: "It materializes executable policies as programs rather than only selecting from a fixed skill list.",
        isCorrect: true,
      },
      {
        text: "Generated policies can specify perception-conditioned branches, feedback loops, API calls, or control logic.",
        isCorrect: true,
      },
      {
        text: "Behavior trees, constraint-solving loops, and robot-control programs can all serve as action interfaces.",
        isCorrect: true,
      },
      {
        text: "The harness can still monitor, validate, and refine generated policies through execution feedback.",
        isCorrect: true,
      },
    ],
    explanation:
      "Programmatic policy generation treats code as the control interface between model intent and the environment. The generated program can encode branching, constraints, loops, and calls to controllers or APIs, while the harness still monitors, validates, and refines execution through feedback.",
  },
  {
    id: "code-agent-harness-q13",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements correctly describe lifelong code-based agents?",
    options: [
      {
        text: "They use code not only to act once, but also to store reusable behaviors and accumulated interaction knowledge.",
        isCorrect: true,
      },
      {
        text: "A skill library can grow as successful strategies, human corrections, or replanning traces are converted into reusable code.",
        isCorrect: true,
      },
      {
        text: "They raise governance problems such as forgetting, abstraction quality, and alignment between stored skills and current environments.",
        isCorrect: true,
      },
      {
        text: "They avoid persistent memory because long-horizon interaction makes reusable executable skills harmful by definition.",
        isCorrect: false,
      },
    ],
    explanation:
      "Lifelong code-based agents treat executable skills as both actions and memory. The hard part is no longer just generating a skill once, but deciding what to retain, how to abstract it, when to update it, and how to prevent stale or misgrounded skills from harming later behavior.",
  },
  {
    id: "code-agent-harness-q14",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe code-for-environment modeling?",
    options: [
      {
        text: "It represents state, dynamics, and feedback through artifacts such as simulators, repositories, tests, traces, logs, and state-transition programs.",
        isCorrect: true,
      },
      {
        text: "It helps agents query, execute, edit, and refine environment state instead of treating the world as an opaque text source.",
        isCorrect: true,
      },
      {
        text: "It prevents agents from checking state transitions through execution.",
        isCorrect: false,
      },
      {
        text: "It makes environment state less inspectable by removing computational artifacts from the loop.",
        isCorrect: false,
      },
    ],
    explanation:
      "Code-for-environment modeling makes the task environment computationally explicit. When the environment is represented by runnable tests, repositories, simulators, logs, or transition programs, the agent can inspect state changes and verify outcomes rather than losing that ability.",
  },
  {
    id: "code-agent-harness-q15",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statement best distinguishes structured world representations from execution-trace world modeling?",
    options: [
      {
        text: "Structured world representations encode objects, relations, layouts, or world structure as programmatic artifacts, while execution-trace world modeling learns or updates dynamics from runtime transitions.",
        isCorrect: true,
      },
      {
        text: "Structured world representations are never executable, while execution traces are always natural-language summaries.",
        isCorrect: false,
      },
      {
        text: "Structured world representations are limited to unit tests, while execution traces are limited to repository file paths.",
        isCorrect: false,
      },
      {
        text: "Both approaches reject code as an environment interface and instead rely only on latent embeddings.",
        isCorrect: false,
      },
    ],
    explanation:
      "Structured representations make world structure explicit, such as objects, relations, layouts, HTML, or simulation programs. Execution-trace modeling focuses on what happens during runtime and uses traces or transitions to learn, predict, or revise environment dynamics.",
  },
  {
    id: "code-agent-harness-q16",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe code-grounded evaluation environments?",
    options: [
      {
        text: "They evaluate agent behavior through runnable systems rather than only static answer matching.",
        isCorrect: true,
      },
      {
        text: "Repository-level unit tests can act as objective environment-state checks for software tasks.",
        isCorrect: true,
      },
      {
        text: "Interactive sandboxes can turn code edits, commands, or tool calls into actions with observable feedback.",
        isCorrect: true,
      },
      {
        text: "They cannot support benchmarks because executable feedback is incompatible with evaluation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Code-grounded evaluation makes the environment itself part of the test. The agent acts through code, commands, or tools, and the harness evaluates the resulting state through unit tests, sandbox feedback, validators, or executable task checkers.",
  },
  {
    id: "code-agent-harness-q17",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statement best captures why planning matters in an agent harness?",
    options: [
      {
        text: "Planning structures how the agent externalizes intent into executable steps, schedules tools and artifacts, and revises when feedback reveals errors.",
        isCorrect: true,
      },
      {
        text: "Planning is useful only as private hidden reasoning and should not become an inspectable harness artifact.",
        isCorrect: false,
      },
      {
        text: "Planning removes the need for memory, tools, execution feedback, and verification.",
        isCorrect: false,
      },
      {
        text: "Planning works only for one-shot code completion and becomes irrelevant in repository-level work.",
        isCorrect: false,
      },
    ],
    explanation:
      "Planning is a control mechanism for long-horizon work, not just hidden thought. It can become an inspectable artifact that guides what to inspect, modify, execute, and verify, while feedback from those actions should also update the plan.",
  },
  {
    id: "code-agent-harness-q18",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe linear decomposition planning?",
    options: [
      {
        text: "It produces a single explicit sequence of steps that guides later generation or action.",
        isCorrect: true,
      },
      {
        text: "It can become filesystem-backed through plan files, implementation notes, status logs, and validation checklists.",
        isCorrect: true,
      },
      {
        text: "It systematically explores many branches before committing to any path.",
        isCorrect: false,
      },
      {
        text: "Its main strength is that the first decomposition is guaranteed to be complete and never needs revision.",
        isCorrect: false,
      },
    ],
    explanation:
      "Linear planning gives the agent a persistent scaffold for step-by-step execution, and plan files can make that scaffold reviewable and resumable. Its weakness is limited exploration: the first decomposition is not guaranteed to be complete, so feedback may need to trigger revision.",
  },
  {
    id: "code-agent-harness-q19",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements correctly describe structure-grounded planning?",
    options: [
      {
        text: "It grounds planning in artifacts such as dependency graphs, repository graphs, circuit graphs, knowledge graphs, or project documentation.",
        isCorrect: true,
      },
      {
        text: "It can expose entities, dependencies, conventions, and edit obligations that constrain the action space.",
        isCorrect: true,
      },
      {
        text: "It can improve long-horizon coherence by turning project knowledge into explicit harness objects.",
        isCorrect: true,
      },
      {
        text: "It can use repository-local instructions, API specifications, testing guides, and architecture notes as persistent planning constraints.",
        isCorrect: true,
      },
    ],
    explanation:
      "Structure-grounded planning uses explicit project or domain structure to guide the agent. Dependency graphs, architecture notes, APIs, test guides, repository-local instructions, and navigation artifacts help the agent avoid flat-context guesses and reason about what changes depend on what.",
  },
  {
    id: "code-agent-harness-q20",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements correctly describe search-based planning?",
    options: [
      {
        text: "It spends inference-time compute exploring multiple candidate strategies, trajectories, patches, or code variants.",
        isCorrect: true,
      },
      {
        text: "It can use execution signals, critiques, or learned scores to decide which branches to expand or discard.",
        isCorrect: true,
      },
      {
        text: "It is a harness-level state management problem because candidates, logs, tests, and traces must be preserved and compared.",
        isCorrect: true,
      },
      {
        text: "It is equivalent to following a single numbered plan without backtracking.",
        isCorrect: false,
      },
    ],
    explanation:
      "Search-based planning differs from a single fixed plan because it keeps alternatives alive and uses feedback to choose among them. The harness must manage candidate artifacts and evidence, not merely sample more text from the model.",
  },
  {
    id: "code-agent-harness-q21",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe orchestration-based planning?",
    options: [
      {
        text: "Planning can emerge from how the harness routes work among roles, stages, modules, and feedback loops.",
        isCorrect: true,
      },
      {
        text: "Feedback-centered orchestration can distribute coding, testing, analysis, and repair across separate modules.",
        isCorrect: true,
      },
      {
        text: "Staged workflows can define handoffs such as comprehension, retrieval, planning, coding, debugging, and repair.",
        isCorrect: true,
      },
      {
        text: "Natural-language harness specifications can function as runtime-interpreted plans under contracts, budgets, tools, and environment state.",
        isCorrect: true,
      },
    ],
    explanation:
      "Orchestration-based planning moves planning power into the structure of the harness itself. Roles, stages, contracts, adapters, routing rules, and feedback loops determine what happens next, so the plan is not just a prompt but part of the runtime control system.",
  },
  {
    id: "code-agent-harness-q22",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statement best explains why planning results can be hard to evaluate?",
    options: [
      {
        text: "Reported planning gains depend on execution environments, feedback quality, tool access, trajectory budgets, and whether benchmarks really test long-range coordination.",
        isCorrect: true,
      },
      {
        text: "Planning quality can be measured completely by the length of the model's reasoning trace.",
        isCorrect: false,
      },
      {
        text: "Planning is independent of validators, tests, tools, memory, and execution budgets.",
        isCorrect: false,
      },
      {
        text: "A plan that produces a final answer always proves that the harness made reliable intermediate decisions.",
        isCorrect: false,
      },
    ],
    explanation:
      "Planning cannot be evaluated in isolation from the harness conditions around it. A system may appear better because it had stronger tools, more retries, better feedback, or easier benchmarks rather than because its planning mechanism truly managed long-horizon dependencies.",
  },
  {
    id: "code-agent-harness-q23",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe memory and context engineering in code-agent harnesses?",
    options: [
      {
        text: "Memory is a state-management layer, not simply a larger context window.",
        isCorrect: true,
      },
      {
        text: "It decides what remains active in the model context, what gets summarized, and what is offloaded to durable storage.",
        isCorrect: true,
      },
      {
        text: "It helps prevent repeated searches, lost clues, and consistency breaks during long workflows.",
        isCorrect: true,
      },
      {
        text: "It can include working, semantic, experiential, long-term, and multi-agent memory.",
        isCorrect: true,
      },
    ],
    explanation:
      "Memory in this framing coordinates where task-relevant state lives and how it is reused. It is broader than a vector database or raw chat history because it controls active context, retrieval, compaction, offloading, collaboration, and long-term validated knowledge.",
  },
  {
    id: "code-agent-harness-q24",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly distinguish working memory and semantic memory?",
    options: [
      {
        text: "Working memory maintains the current task trajectory, such as summaries, failed-test records, edit state, and stack information.",
        isCorrect: true,
      },
      {
        text: "Semantic memory exposes repository evidence such as functions, classes, call relations, documentation, and dependency metadata.",
        isCorrect: true,
      },
      {
        text: "Working memory is mainly a permanent archive of validated knowledge across unrelated future projects.",
        isCorrect: false,
      },
      {
        text: "Semantic memory is unrelated to the codebase and stores only the model's private hidden thoughts.",
        isCorrect: false,
      },
    ],
    explanation:
      "Working memory keeps the current trajectory grounded under context limits, while semantic memory retrieves relevant external evidence from the repository or surrounding artifacts. Permanent validated knowledge across unrelated future projects is closer to long-term memory than working memory.",
  },
  {
    id: "code-agent-harness-q25",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe experiential and long-term memory?",
    options: [
      {
        text: "Experiential memory captures reusable repair trajectories, failure cases, debugging records, or strategy patterns.",
        isCorrect: true,
      },
      {
        text: "Long-term memory should preserve validated, reusable knowledge in compact and controllable forms.",
        isCorrect: true,
      },
      {
        text: "Ungoverned historical records can add noise, staleness, false retrievals, and error propagation.",
        isCorrect: true,
      },
      {
        text: "The main objective of long-term memory is to accumulate every event without write gates, compression, or quality control.",
        isCorrect: false,
      },
    ],
    explanation:
      "Experiential memory supports transfer from prior tasks, and long-term memory supports persistent reusable knowledge. Both require governance because low-quality or stale memories can become misleading context rather than useful evidence.",
  },
  {
    id: "code-agent-harness-q26",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe multi-agent memory?",
    options: [
      {
        text: "It serves as a medium for sharing information, passing intentions, and maintaining consistency across specialized roles.",
        isCorrect: true,
      },
      {
        text: "It can resemble a shared blackboard or collaborative state graph rather than a purely individual memory store.",
        isCorrect: true,
      },
      {
        text: "Its only challenge is storing more tokens, while granularity of sharing and information flooding do not matter.",
        isCorrect: false,
      },
      {
        text: "It is unnecessary whenever agents have different role names.",
        isCorrect: false,
      },
    ],
    explanation:
      "Multi-agent memory is about coordinating shared state, not just giving each worker more context. The harness must decide what gets shared, at what granularity, how detailed traces connect to high-level decisions, and how to prevent flooding or stale beliefs.",
  },
  {
    id: "code-agent-harness-q27",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statement best describes context compaction and state offloading?",
    options: [
      {
        text: "They separate decision-relevant active context from full-fidelity durable evidence, using summaries and retrievable handles instead of raw logs everywhere.",
        isCorrect: true,
      },
      {
        text: "They require every build log, trace, and test output to stay verbatim in the active prompt.",
        isCorrect: false,
      },
      {
        text: "They make auditing impossible because full artifacts are deleted after summarization.",
        isCorrect: false,
      },
      {
        text: "They are useful only for conversational chatbots and not for code-agent workflows.",
        isCorrect: false,
      },
    ],
    explanation:
      "Compaction keeps the active context focused by summarizing long histories and tool outputs, while offloading preserves full artifacts outside the prompt. This lets the agent act on concise evidence while retaining detailed logs, traces, or files for audit and replay.",
  },
  {
    id: "code-agent-harness-q28",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe tool use as a harness mechanism?",
    options: [
      {
        text: "Tools expand the action space while exposing external feedback signals.",
        isCorrect: true,
      },
      {
        text: "The harness should control schemas, permissions, execution location, result sanitization, and human approval for risky actions.",
        isCorrect: true,
      },
      {
        text: "Lifecycle hooks can validate or block a tool call before execution and summarize or store results after execution.",
        isCorrect: true,
      },
      {
        text: "Tool use is reliable by default as long as the model can emit a valid-looking function name.",
        isCorrect: false,
      },
    ],
    explanation:
      "Tool use is a governed interface between model intent and external systems. Reliable harnesses define the available tools, validate arguments, enforce permissions, execute in controlled environments, process outputs, update state, and gate risky actions.",
  },
  {
    id: "code-agent-harness-q29",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly compare function-oriented and environment-interaction tool use?",
    options: [
      {
        text: "Function-oriented tools help the agent retrieve or select APIs, libraries, documentation, and external coding utilities.",
        isCorrect: true,
      },
      {
        text: "Environment-interaction tools let the agent inspect repositories, edit files, run commands, and validate behavior in development environments.",
        isCorrect: true,
      },
      {
        text: "Function-oriented tools are least relevant when API or library knowledge is the main bottleneck.",
        isCorrect: false,
      },
      {
        text: "Environment-interaction tools avoid repositories, terminals, tests, browsers, and sandboxes.",
        isCorrect: false,
      },
    ],
    explanation:
      "Function-oriented tools ground implementation choices in external knowledge, especially long-tail APIs or private libraries. Environment-interaction tools are broader operational interfaces that let the agent work inside the repository or runtime and observe the consequences.",
  },
  {
    id: "code-agent-harness-q30",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe verification-driven tool use?",
    options: [
      {
        text: "Tests, type checkers, static analyzers, compiler errors, runtime traces, and fuzzers can act as deterministic sensors.",
        isCorrect: true,
      },
      {
        text: "Verification outputs should often be parsed, summarized, and offloaded because raw logs can be too long or noisy.",
        isCorrect: true,
      },
      {
        text: "Verification-driven tools are mainly for retrieving API docs and do not evaluate code behavior.",
        isCorrect: false,
      },
      {
        text: "A failed check should never update working memory or guide the next action.",
        isCorrect: false,
      },
    ],
    explanation:
      "Verification-driven tools make progress inspectable through structured evidence such as failures, warnings, traces, and coverage gaps. The harness should route those signals into memory and repair decisions while preserving enough raw detail for audit and replay.",
  },
  {
    id: "code-agent-harness-q31",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe workflow-orchestration tool use?",
    options: [
      {
        text: "It coordinates retrieval, localization, editing, execution, memory updates, approvals, and repeated verification.",
        isCorrect: true,
      },
      {
        text: "It decides when tools should be invoked, with which permissions, under which context, and how results update harness state.",
        isCorrect: true,
      },
      {
        text: "It can package typed tool schemas, sessions, workspaces, guardrails, handoffs, tracing, and review mechanisms.",
        isCorrect: true,
      },
      {
        text: "It is best understood as adding more tools without controlling when or how they are used.",
        isCorrect: false,
      },
    ],
    explanation:
      "Workflow orchestration is about ordering and governing tool use across a long task. The difficult part is not simply exposing many tools, but coordinating their permissions, contexts, outputs, state updates, and verification hooks into a reliable loop.",
  },
  {
    id: "code-agent-harness-q32",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe the Plan-Execute-Verify loop?",
    options: [
      {
        text: "Planning externalizes intended changes, assumptions, validation criteria, and risky operations.",
        isCorrect: true,
      },
      {
        text: "Execution applies bounded actions inside a sandboxed and permissioned environment.",
        isCorrect: true,
      },
      {
        text: "Verification compares the resulting state against constraints using deterministic sensors and human-review gates when needed.",
        isCorrect: true,
      },
      {
        text: "The loop treats debugging, validation, escalation, and repair as parts of one control process.",
        isCorrect: true,
      },
    ],
    explanation:
      "The Plan-Execute-Verify loop treats agent work as governed state transitions. A plan sets the contract, execution applies changes inside a controlled substrate, and verification decides whether to accept, revise, escalate, or roll back the state.",
  },
  {
    id: "code-agent-harness-q33",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statement best describes planning as contract formation?",
    options: [
      {
        text: "A robust plan identifies files, invariants, validation commands, rollback points, and risky operations that constrain the next state transition.",
        isCorrect: true,
      },
      {
        text: "A robust plan is useful only if it remains hidden from tools, users, and later verification steps.",
        isCorrect: false,
      },
      {
        text: "A robust plan should avoid mentioning verification criteria because failed checks are unrelated to planning.",
        isCorrect: false,
      },
      {
        text: "A robust plan is equivalent to model confidence that the final answer will be correct.",
        isCorrect: false,
      },
    ],
    explanation:
      "Planning becomes a contract when it states what should change and how that change will be judged. This makes the plan inspectable and actionable, and failed verification can update the contract rather than being treated as a separate afterthought.",
  },
  {
    id: "code-agent-harness-q34",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe sandboxed execution and permissioned state transitions?",
    options: [
      {
        text: "Sandboxes provide isolated filesystems, runtimes, shells, browser or IDE state, and resource boundaries.",
        isCorrect: true,
      },
      {
        text: "A permission model can separate low-risk observation from sandbox editing and full-access actions.",
        isCorrect: true,
      },
      {
        text: "Network access, credentials, deployment commands, package publishing, and destructive file operations should be treated like low-risk read-only inspection.",
        isCorrect: false,
      },
      {
        text: "Sandboxing makes permission tiers unnecessary because every possible side effect is harmless.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sandboxing improves isolation and reproducibility, but it does not remove the need for permission control. Some operations have consequences beyond a disposable workspace, so they should not be treated like read-only inspection and need tiers, approval gates, and audit logs.",
  },
  {
    id: "code-agent-harness-q35",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe deterministic sensors in the Verify phase?",
    options: [
      {
        text: "Linters, parsers, compilers, type checkers, static analyzers, tests, fuzzers, runtime monitors, and CI pipelines can all produce control signals.",
        isCorrect: true,
      },
      {
        text: "Sensor evidence can determine whether to repair, retrieve context, route to another module, reduce permissions, or escalate to a human.",
        isCorrect: true,
      },
      {
        text: "Termination should be governed by required checks, diminishing returns, risk-tier changes, or human-review requirements rather than model confidence alone.",
        isCorrect: true,
      },
      {
        text: "Natural-language self-reflection is most reliable when it replaces executable evidence entirely.",
        isCorrect: false,
      },
    ],
    explanation:
      "Deterministic or reproducible sensors provide the evidence that keeps the loop grounded. Reflection can help interpret those signals, but it should remain tied to concrete diagnostics, test results, warnings, traces, and other executable feedback.",
  },
  {
    id: "code-agent-harness-q36",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe Agentic Harness Engineering?",
    options: [
      {
        text: "It treats the operating environment around the model as an object of measurement and revision.",
        isCorrect: true,
      },
      {
        text: "It can revise tool schemas, retrieval policies, planning artifacts, memory policies, verification sensors, permission tiers, or workflow topology.",
        isCorrect: true,
      },
      {
        text: "It recognizes that many failures come from brittle harness components rather than only from weak model generation.",
        isCorrect: true,
      },
      {
        text: "It is limited to changing the base model weights through gradient updates.",
        isCorrect: false,
      },
    ],
    explanation:
      "Agentic Harness Engineering focuses on improving the system that turns a model into an agent. It can change prompts and tools, but also the wider runtime: memory, retrieval, permissions, validators, sandboxes, workflows, telemetry, and review gates.",
  },
  {
    id: "code-agent-harness-q37",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe deep telemetry?",
    options: [
      {
        text: "It records structured traces that connect model decisions, harness actions, environment states, and outcomes.",
        isCorrect: true,
      },
      {
        text: "It can include prompts, retrieved context, token cost, latency, tool arguments, edited files, command outputs, test results, stack traces, and human interventions.",
        isCorrect: true,
      },
      {
        text: "It supports comparative diagnosis by making trajectories replayable across harness versions.",
        isCorrect: true,
      },
      {
        text: "It is equivalent to logging only the final pass or fail result.",
        isCorrect: false,
      },
    ],
    explanation:
      "Deep telemetry is useful because it links outcomes back to the decisions and artifacts that produced them. A final result alone cannot reveal whether failures came from retrieval, tool routing, invalid permissions, weak validators, stale context, or costly loops.",
  },
  {
    id: "code-agent-harness-q38",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statement best describes an Evolution Agent?",
    options: [
      {
        text: "It uses trajectory telemetry to diagnose failures, propose harness revisions, evaluate them on held-out tasks or replayed traces, and promote only verified improvements.",
        isCorrect: true,
      },
      {
        text: "It edits only the target repository and never changes prompts, tools, validators, memory policies, or permission rules.",
        isCorrect: false,
      },
      {
        text: "It should promote every suggested harness change immediately because self-modification is safe by default.",
        isCorrect: false,
      },
      {
        text: "It replaces the need for regression tests because telemetry is already a complete proof of improvement.",
        isCorrect: false,
      },
    ],
    explanation:
      "An Evolution Agent operates at the meta level: it improves the conditions under which later task agents work. Its proposals still need evaluation, regression checks, and promotion rules because harness changes can alter future behavior in broad and risky ways.",
  },
  {
    id: "code-agent-harness-q39",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe governed harness mutation?",
    options: [
      {
        text: "Candidate harness changes should be tested in sandboxes, compared against regression suites, and recorded with auditable rationales.",
        isCorrect: true,
      },
      {
        text: "Changes to permissions, credentials, network access, deployment behavior, or human-review requirements can be activated without extra review as long as a unit test passes.",
        isCorrect: false,
      },
      {
        text: "The Evolution Agent should itself be subject to a Plan-Execute-Verify loop.",
        isCorrect: true,
      },
      {
        text: "Harness mutation is safe only when it bypasses logs, rollback paths, and human oversight.",
        isCorrect: false,
      },
    ],
    explanation:
      "Harness mutation affects the runtime that controls later agents, so it requires stronger governance than ordinary patching. Sandboxed evaluation, regression tests, audit trails, rollback mechanisms, and human approval for risk-boundary changes keep self-improvement from becoming uncontrolled self-modification.",
  },
  {
    id: "code-agent-harness-q40",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe why multi-agent systems can scale code harnesses?",
    options: [
      {
        text: "They address context-window constraints by distributing work rather than forcing one agent to hold the entire codebase and trace history.",
        isCorrect: true,
      },
      {
        text: "They support specialization across planning, understanding, synthesis, execution, testing, reviewing, and debugging.",
        isCorrect: true,
      },
      {
        text: "They introduce independent coordination and verification channels that a single agent may lack.",
        isCorrect: true,
      },
      {
        text: "They make shared state synchronization unnecessary because role names alone prevent conflicts.",
        isCorrect: false,
      },
    ],
    explanation:
      "Multi-agent systems can decompose responsibilities and make workflows more inspectable, but they also create coordination burdens. Shared code artifacts, execution signals, repositories, memories, and synchronization mechanisms become central because specialized agents can otherwise drift apart.",
  },
  {
    id: "code-agent-harness-q41",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe common roles in multi-agent code harnesses?",
    options: [
      {
        text: "Program synthesis agents generate or revise code artifacts from specifications, plans, or feedback.",
        isCorrect: true,
      },
      {
        text: "Program understanding agents analyze existing code or specifications to produce higher-level representations.",
        isCorrect: true,
      },
      {
        text: "Verification agents generate tests, run static analysis, audit correctness, or simulate execution.",
        isCorrect: true,
      },
      {
        text: "Planning agents decompose tasks and assign subtasks, while execution agents interface with runtimes or deterministic executors.",
        isCorrect: true,
      },
    ],
    explanation:
      "Role specialization lets a multi-agent harness separate different engineering functions. The important design point is not that every system uses identical names, but that planning, understanding, synthesis, execution, and verification can be made into distinct and inspectable responsibilities.",
  },
  {
    id: "code-agent-harness-q42",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statement best explains why independent verification agents can reduce circular reasoning?",
    options: [
      {
        text: "A tester or verifier that is separated from the coding agent can check behavior against independent tests, traces, or specifications instead of accepting the coder's own assumptions.",
        isCorrect: true,
      },
      {
        text: "Verification agents are reliable only when they reuse the coder's exact private reasoning as the test oracle.",
        isCorrect: false,
      },
      {
        text: "Independent verification removes the need to run code because agent disagreement is always a sufficient oracle.",
        isCorrect: false,
      },
      {
        text: "A tester becomes more objective by avoiding execution feedback and looking only at the final code text.",
        isCorrect: false,
      },
    ],
    explanation:
      "Independent verification helps because it can produce or apply evidence that is not simply the coder's own story about the patch. This reduces mode collapse, but it is strongest when grounded in execution, tests, specifications, traces, or other objective signals.",
  },
  {
    id: "code-agent-harness-q43",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe interaction modes in multi-agent code harnesses?",
    options: [
      {
        text: "Collaborative synthesis occurs when agents jointly construct a component, similar to pair programming.",
        isCorrect: true,
      },
      {
        text: "Critique and repair occurs when one agent or module evaluates an artifact and another revises it in response.",
        isCorrect: true,
      },
      {
        text: "Adversarial validation actively tries to break the artifact through counterexamples, fuzzing, or simulation mismatches.",
        isCorrect: true,
      },
      {
        text: "Reasoning debate lets agents argue over interpretations or decisions before aggregation or consensus.",
        isCorrect: true,
      },
    ],
    explanation:
      "Code-centric multi-agent systems coordinate through more than free-form chat. They use shared code artifacts, tests, logs, workflows, counterexamples, debate, and revisions as artifact-mediated communication channels.",
  },
  {
    id: "code-agent-harness-q44",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly compare workflow topologies for multi-agent coordination?",
    options: [
      {
        text: "Chain topologies pass artifacts in a strict sequence, often resembling waterfall development.",
        isCorrect: true,
      },
      {
        text: "Cyclic topologies add feedback loops so code can be revised after validation failures.",
        isCorrect: true,
      },
      {
        text: "Hierarchical and star topologies cannot use central coordination, delegation, or aggregation of worker outputs.",
        isCorrect: false,
      },
      {
        text: "Adaptive topologies keep the same communication graph regardless of task complexity, execution feedback, or observed failures.",
        isCorrect: false,
      },
    ],
    explanation:
      "Topology controls who communicates with whom, in what order, and under what feedback rules. Hierarchical and star designs often rely on coordinators, while adaptive designs may scale agent pools, restructure DAGs, or generate workflows based on task and feedback.",
  },
  {
    id: "code-agent-harness-q45",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe execution feedback in multi-agent code systems?",
    options: [
      {
        text: "Compiler and syntax feedback can block invalid artifacts before later stages proceed.",
        isCorrect: true,
      },
      {
        text: "Test pass/fail signals are common convergence and repair signals, but their strength depends on test quality.",
        isCorrect: true,
      },
      {
        text: "Fuzzer crash traces provide concrete failing inputs rather than only generic failure labels.",
        isCorrect: true,
      },
      {
        text: "Static analysis warnings and performance profiling results can support security or efficiency-oriented convergence criteria.",
        isCorrect: true,
      },
    ],
    explanation:
      "Execution feedback gives multi-agent systems non-linguistic evidence for coordination and repair. Different feedback types expose different failure modes, from syntax problems and test failures to security vulnerabilities, crash inputs, waveform mismatches, and performance regressions.",
  },
  {
    id: "code-agent-harness-q46",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe shared-harness synchronization mechanisms?",
    options: [
      {
        text: "Sequential handoff is simple but can hide state divergence when multiple agents revise or reason over changing artifacts.",
        isCorrect: true,
      },
      {
        text: "A shared blackboard gives agents a persistent program state that can be read, updated, and controlled by the harness.",
        isCorrect: true,
      },
      {
        text: "Structured context scheduling controls what each agent sees and when, instead of dumping the entire history into every invocation.",
        isCorrect: true,
      },
      {
        text: "Parallel branches with merge can require explicit authority, consistency, and conflict-management rules.",
        isCorrect: true,
      },
    ],
    explanation:
      "Synchronization is a distributed-systems problem inside an agent harness. Shared files, blackboards, summaries, queues, branches, memories, and merges all transmit partial state, so the harness must manage freshness, authority, conflict, and context bandwidth.",
  },
  {
    id: "code-agent-harness-q47",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statement best describes the central gap behind the proposed shared code-centric harness substrate?",
    options: [
      {
        text: "Many systems lack a formal, persistent representation of shared code state that agents can query and update across iterations.",
        isCorrect: true,
      },
      {
        text: "Most systems already fully unify repository structure, runtime behavior, memory, and convergence state into one transactionally consistent substrate.",
        isCorrect: false,
      },
      {
        text: "The main gap is that code cannot produce objective signals, so agents must rely only on conversational agreement.",
        isCorrect: false,
      },
      {
        text: "The solution is to remove shared artifacts so each agent works from its own isolated guess about the program.",
        isCorrect: false,
      },
    ],
    explanation:
      "The proposed substrate responds to the fact that many systems still reconstruct shared state from recent context or files. Code can execute and produce objective signals, but harnesses often fail to turn that property into a queryable, persistent, shared state representation.",
  },
  {
    id: "code-agent-harness-q48",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe levels of shared harness representation?",
    options: [
      {
        text: "Implicit or file-only representations rely on the latest artifacts and conversation context, with little formal shared-state model.",
        isCorrect: true,
      },
      {
        text: "Repository-based representations expose directory structure, dependency graphs, call hierarchies, and version history.",
        isCorrect: true,
      },
      {
        text: "Execution-based representations define state by behavior, such as tests passed, vulnerabilities found, runtime performance, or simulation matches.",
        isCorrect: true,
      },
      {
        text: "Blackboard representations provide explicit shared state that agents can read, update, and synchronize through the harness.",
        isCorrect: true,
      },
    ],
    explanation:
      "The representation level determines what kind of shared world the agents inhabit. File-only context is weak, repository views expose static structure, execution views expose behavior, and blackboards move closer to a formal shared substrate.",
  },
  {
    id: "code-agent-harness-q49",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe convergence in multi-agent code harnesses?",
    options: [
      {
        text: "Correctness convergence often means all required tests pass, though this depends on the adequacy of the tests.",
        isCorrect: true,
      },
      {
        text: "Security convergence can require no flagged vulnerabilities and no fuzzer-induced crashes.",
        isCorrect: true,
      },
      {
        text: "Performance convergence can depend on measured runtime, memory, or other efficiency thresholds.",
        isCorrect: true,
      },
      {
        text: "Implicit convergence based on fixed stages or iteration budgets is the most principled objective criterion.",
        isCorrect: false,
      },
    ],
    explanation:
      "Because code is executable, convergence can often be grounded in objective behavior. Test, security, performance, score, and consensus criteria differ in strength, while implicit termination by budget or stage completion is weaker because it may not reflect program quality.",
  },
  {
    id: "code-agent-harness-q50",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statement best explains why code-mediated channels do not eliminate coordination bottlenecks?",
    options: [
      {
        text: "Files, APIs, diffs, tests, logs, summaries, schemas, and blackboards are partial channels that trade off fidelity, latency, scope, authority, and conflict behavior.",
        isCorrect: true,
      },
      {
        text: "Once agents use files and tests, summaries can no longer lose details and logs can no longer become noisy.",
        isCorrect: false,
      },
      {
        text: "Code-mediated coordination removes the need to decide which artifact is authoritative.",
        isCorrect: false,
      },
      {
        text: "Execution signals make stale cached views, conflicting assumptions, and bandwidth limits impossible.",
        isCorrect: false,
      },
    ],
    explanation:
      "Code provides richer coordination channels than pure dialogue, but those channels still compress and filter state. Mature harnesses must decide which artifacts are authoritative, how evidence is summarized, how conflicts are resolved, and when execution should override linguistic claims.",
  },
  {
    id: "code-agent-harness-q51",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe modern code assistants as code-as-harness systems?",
    options: [
      {
        text: "They operate over repository workspaces with files, tests, dependency metadata, issues, branches, and pull requests.",
        isCorrect: true,
      },
      {
        text: "Their development harnesses manage repository access, edits, commands, approval boundaries, logging, validation, and sandboxing.",
        isCorrect: true,
      },
      {
        text: "Execution feedback turns candidate edits into verifiable transformations through compilers, tests, linters, and runtime traces.",
        isCorrect: true,
      },
      {
        text: "They must infer latent developer intent and project conventions, not only visible test pass rates.",
        isCorrect: true,
      },
    ],
    explanation:
      "Code assistants are a production instantiation of the harness idea because they act inside a live engineering workspace. They need repository memory, executable validation, permission boundaries, workflow integration, and sensitivity to intent and conventions, not just code completion.",
  },
  {
    id: "code-agent-harness-q52",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe open challenges for code-assistant harnesses?",
    options: [
      {
        text: "Verification beyond unit tests remains difficult because passing visible tests may not prove semantic correctness, security, maintainability, or organic fit.",
        isCorrect: true,
      },
      {
        text: "Failure attribution in long-horizon loops is already solved because a final failed test always identifies the exact responsible step, tool, and agent.",
        isCorrect: false,
      },
      {
        text: "Safety governance requires capability and permission primitives for autonomous code execution.",
        isCorrect: true,
      },
      {
        text: "Trust calibration in pair-programming workflows is irrelevant once an agent can open a pull request.",
        isCorrect: false,
      },
    ],
    explanation:
      "Code-assistant harnesses expose gaps that passing tests alone cannot solve. They need richer verifiers, better trace-based failure attribution because final failures are often ambiguous, strong safety governance, rollback and stability for self-evolution, shared-state synchronization, and human trust calibration.",
  },
  {
    id: "code-agent-harness-q53",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe GUI and operating-system agents as program-world agents?",
    options: [
      {
        text: "The latent state can include DOMs, JavaScript heaps, app activity stacks, filesystems, window trees, and other software state.",
        isCorrect: true,
      },
      {
        text: "Observations can take code-defined forms such as serialized HTML, accessibility trees, screenshots with coordinates, or hybrid pixel-and-structure representations.",
        isCorrect: true,
      },
      {
        text: "Actions can compile to DOM events, accessibility calls, keyboard or mouse primitives, Playwright scripts, or OS commands.",
        isCorrect: true,
      },
      {
        text: "Evaluator scripts can inspect post-action state and close the loop with executable success checks.",
        isCorrect: true,
      },
    ],
    explanation:
      "GUI and OS agents make the program-world view concrete: rendered screens, UI trees, and OS state come from executable systems, while actions are calls back into those systems. Evaluation can also be executable, using scripts to inspect state after the agent acts.",
  },
  {
    id: "code-agent-harness-q54",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statement best describes memory for code-grounded GUI agents?",
    options: [
      {
        text: "Memory is persistent programmatic state such as UI abstractions, exploration documents, skill libraries, structured traces, or knowledge graphs that can guide later actions.",
        isCorrect: true,
      },
      {
        text: "Memory is unnecessary because each screenshot contains a perfect and complete description of all task history.",
        isCorrect: false,
      },
      {
        text: "GUI memory must be only raw pixels and cannot include JSON, Python skills, HTML summaries, or graph-like state.",
        isCorrect: false,
      },
      {
        text: "Successful GUI skills should never be promoted into reusable libraries because UI environments cannot recur.",
        isCorrect: false,
      },
    ],
    explanation:
      "GUI agents need memory because the current screen rarely contains all useful history or learned behavior. Programmatic artifacts such as UI summaries, exploration notes, reusable code snippets, and trace-derived knowledge can be retrieved and composed in later interactions.",
  },
  {
    id: "code-agent-harness-q55",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe embodied agents in the code-as-harness view?",
    options: [
      {
        text: "Code can translate high-level intent into primitive skill calls, synthesized control policies, behavior trees, or robot APIs.",
        isCorrect: true,
      },
      {
        text: "A layered embodied harness separates semantic reasoning, typed robot APIs, perception/state estimation, motion planning, low-level control, and safety constraints.",
        isCorrect: true,
      },
      {
        text: "Embodied skills act only as natural-language descriptions and should not be re-executed as reusable behaviors.",
        isCorrect: false,
      },
      {
        text: "Physical constraints such as reachability, collision, force, timing, and stability become irrelevant once a policy is generated as code.",
        isCorrect: false,
      },
    ],
    explanation:
      "Embodied agents need code as both grounding interface and safety boundary. Generated policies and skill calls still have to respect sensors, controllers, physical feasibility, and governance, and the strongest skill memories are reusable executable behaviors rather than mere prose.",
  },
  {
    id: "code-agent-harness-q56",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statement best describes scientific discovery as a code-as-harness domain?",
    options: [
      {
        text: "Hypotheses, protocols, solvers, simulations, analyses, lab automation scripts, and manuscripts can become parts of an executable generate-execute-feedback research loop.",
        isCorrect: true,
      },
      {
        text: "Scientific agents avoid code because hypotheses and experiments cannot be represented as executable or verifiable artifacts.",
        isCorrect: false,
      },
      {
        text: "A scientific harness ends after ideation and does not include experiment execution, observation, analysis, or revision.",
        isCorrect: false,
      },
      {
        text: "Simulators and self-driving labs are outside the harness because physical or numerical feedback cannot update agent state.",
        isCorrect: false,
      },
    ],
    explanation:
      "Scientific work naturally fits the harness pattern because the scientific method already has a closed-loop structure. Code can carry hypotheses, experiments, solvers, notebooks, lab protocols, data analysis, figures, and reports through an auditable execution pipeline.",
  },
  {
    id: "code-agent-harness-q57",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe personalization agents as code-centric harnesses?",
    options: [
      {
        text: "The key state includes latent, contextual, and sometimes unstable user preferences.",
        isCorrect: true,
      },
      {
        text: "Structured preference memory can make user interests, constraints, corrections, and long-term goals inspectable and editable.",
        isCorrect: true,
      },
      {
        text: "Feedback pipelines interpret clicks, dwell time, ratings, purchases, skips, and conversational corrections as partial evidence.",
        isCorrect: true,
      },
      {
        text: "Proxy engagement metrics are always reliable oracles for true user welfare and satisfaction.",
        isCorrect: false,
      },
    ],
    explanation:
      "Personalization is difficult because the environment includes a human whose true preferences are only partially observed. Code-centric state can make preference memory and policy constraints more auditable, but feedback signals are noisy and can conflict with user welfare or autonomy.",
  },
  {
    id: "code-agent-harness-q58",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statement best explains oracle adequacy in harness-level evaluation?",
    options: [
      {
        text: "The evaluator must capture the intended task and risks, not only a narrow executable proxy such as passing visible tests.",
        isCorrect: true,
      },
      {
        text: "A harness should be evaluated only by final task success because tool calls, retries, feedback quality, and safety policies do not affect performance.",
        isCorrect: false,
      },
      {
        text: "A green test result proves that all semantic, security, usability, and safety requirements have been satisfied.",
        isCorrect: false,
      },
      {
        text: "Harness-level metrics should ignore trajectory efficiency, replayability, safety compliance, state consistency, and recovery ability.",
        isCorrect: false,
      },
    ],
    explanation:
      "Oracle adequacy asks whether the check actually measures what matters. Harness-level evaluation should separate model ability from retrieval, tools, retries, feedback, safety, state consistency, recovery, and replayability, rather than treating a narrow pass/fail proxy as complete truth.",
  },
  {
    id: "code-agent-harness-q59",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe semantic verification beyond executable feedback?",
    options: [
      {
        text: "Executable feedback can create false confidence when tests, checkers, simulations, or scripts encode incomplete specifications.",
        isCorrect: true,
      },
      {
        text: "A verification stack should combine artifacts such as tests, fuzzers, static analysis, type checks, runtime monitors, formal specifications, critiques, and human review.",
        isCorrect: true,
      },
      {
        text: "Each verification artifact should expose what it verifies, what it cannot verify, and how much confidence it provides.",
        isCorrect: true,
      },
      {
        text: "Accepted actions should ideally carry evidence bundles with checks run, assumptions preserved, untested regions, and remaining risks.",
        isCorrect: true,
      },
    ],
    explanation:
      "Execution is valuable but only as strong as the oracle attached to it. A mature harness needs scoped evidence, multiple independent checks, feedback calibration, and explicit remaining-risk records so that verification becomes an inspectable contract rather than a single green signal.",
  },
  {
    id: "code-agent-harness-q60",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statement best captures the long-term direction of harness engineering?",
    options: [
      {
        text: "Reliable long-horizon agents should be executable, inspectable, stateful, and governed across context, memory, tools, execution, feedback, safety, coordination, and evaluation.",
        isCorrect: true,
      },
      {
        text: "Harness engineering should focus only on bigger model context windows and ignore tools, permissions, telemetry, memory, and evaluation.",
        isCorrect: false,
      },
      {
        text: "Once agents can generate code, human-in-the-loop safety, multimodal grounding, and shared-state conflicts stop being research problems.",
        isCorrect: false,
      },
      {
        text: "The final task answer is enough to diagnose all failures in a closed-loop agent system.",
        isCorrect: false,
      },
    ],
    explanation:
      "The future agenda treats the complete closed loop as the object of study. Strong systems need executable grounding, inspectable plans and provenance, stateful memory and coordination, and governance through permissions, verification, accountability, and human oversight where consequences are high.",
  },
];
