import { Question } from "../../../quiz";

export const MemoryInTheAgeOfAIAgentsSurveyQuestions: Question[] = [
  {
    id: "ai-agents-memory-survey-q01",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which ideas are part of the forms-functions-dynamics view of agent memory?",
    options: [
      {
        text: "Forms ask what representational carrier holds memory, such as token-level, parametric, or latent state.",
        isCorrect: true,
      },
      {
        text: "Functions ask why the agent needs memory, such as factual recall, experiential improvement, or active working context.",
        isCorrect: true,
      },
      {
        text: "Dynamics ask how memory is formed, evolved, and retrieved as interaction unfolds over time.",
        isCorrect: true,
      },
      {
        text: "The view is intended to clarify fragmented terminology by separating representation, purpose, and lifecycle behavior.",
        isCorrect: true,
      },
    ],
    explanation:
      "The taxonomy separates the carrier of memory, the purpose memory serves, and the lifecycle operations that make memory change over time. A plain short-term versus long-term split is too coarse because modern agents may use the same container with different temporal schedules and very different representational forms.",
  },
  {
    id: "ai-agents-memory-survey-q02",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "An agent memory state is modeled as an evolving store coupled to the agent's decision process. Which statements correctly describe that model?",
    options: [
      {
        text: "The memory state can be a text buffer, key-value store, vector database, graph, parameter module, or hybrid representation.",
        isCorrect: true,
      },
      {
        text: "Formation selectively converts tool outputs, plans, reasoning traces, feedback, or observations into candidate memories.",
        isCorrect: true,
      },
      {
        text: "Evolution integrates formed memories by consolidating, resolving conflicts, pruning, or restructuring them.",
        isCorrect: true,
      },
      {
        text: "A memory system loses its agentic role when retrieval is scheduled at task initialization and disabled for later steps.",
        isCorrect: false,
      },
    ],
    explanation:
      "The formal model deliberately leaves the internal memory structure open, because different systems use different carriers. Formation, evolution, and retrieval are conceptual operators, and the important subtlety is that they can be scheduled differently across systems rather than being required at every step.",
  },
  {
    id: "ai-agents-memory-survey-q03",
    chapter: 1,
    difficulty: "medium",
    type: "assertion-reason",
    prompt:
      "Assertion: Short-term and long-term memory effects can emerge from temporal invocation patterns rather than from two physically separate modules.\n\nReason: A unified memory state may already contain cross-task information before a task starts while also accumulating task-specific information during execution.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: false },
      { text: "Assertion is false, Reason is true.", isCorrect: false },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: true,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "The assertion is true because the same memory container can be used across tasks and within the current task. The reason explains the mechanism: cross-trial memory and inside-trial memory are roles created by how formation, evolution, and retrieval are invoked over time.",
  },
  {
    id: "ai-agents-memory-survey-q04",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which distinctions help separate agent memory from large language model (LLM) internal memory?",
    options: [
      {
        text: "Many early LLM memory systems for dialogue state, preferences, and multi-turn experience are better understood as agent memory under modern agent terminology.",
        isCorrect: true,
      },
      {
        text: "Architecture changes for longer context, recurrent state, attention sparsity, or key-value cache management can be LLM memory without being agent memory.",
        isCorrect: true,
      },
      {
        text: "Agent memory is defined by modifying transformer weights or caches directly, and systems that maintain evolving external stores are treated as context engineering rather than memory.",
        isCorrect: false,
      },
      {
        text: "A memory mechanism stays inside agent memory whenever it expands sequence retention, even if it lacks deliberate memory formation, evolution, retrieval, or cross-task adaptation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Agent memory concerns information stores and operations that support an agent's decisions across interaction. LLM-internal memory work can be valuable for sequence retention, but if it mainly reorganizes the model's internal capacity rather than maintaining an evolving agent memory base, it belongs to a different scope.",
  },
  {
    id: "ai-agents-memory-survey-q05",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statement best distinguishes classical retrieval-augmented generation (RAG) from agent memory?",
    options: [
      {
        text: "Classical RAG usually retrieves from externally maintained knowledge for a single inference task, while agent memory accumulates and revises knowledge through ongoing interaction.",
        isCorrect: true,
      },
      {
        text: "Classical RAG uses vector indexes, while agent memory uses a separate lookup family with no overlap in semantic-search tooling.",
        isCorrect: false,
      },
      {
        text: "Classical RAG maintains a self-evolving internal memory base, while agent memory is limited to static document retrieval.",
        isCorrect: false,
      },
      {
        text: "Classical RAG and agent memory are fully separable because no retrieval system ever updates context during a task.",
        isCorrect: false,
      },
    ],
    explanation:
      "The practical distinction is about role and timescale, not about whether both systems use similar retrieval components. RAG often grounds a single task in external knowledge, while agent memory is tied to an agent's persistent and evolving state, even though dynamic retrieval systems can blur the boundary.",
  },
  {
    id: "ai-agents-memory-survey-q06",
    chapter: 1,
    difficulty: "hard",
    type: "assertion-reason",
    prompt:
      "Assertion: Context engineering subsumes agent memory because both are concerned with what information enters the model context.\n\nReason: Context engineering primarily optimizes the momentary interface under context-window constraints, while agent memory sustains a persistent cognitive state across interactions.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: false },
      { text: "Assertion is false, Reason is true.", isCorrect: true },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: false,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "The assertion is false because the relationship is an intersection of paradigms rather than a clean subsumption. The reason is true: context engineering schedules and formats information for the current inference interface, whereas agent memory governs what the agent knows, has experienced, and continues to update over time.",
  },
  {
    id: "ai-agents-memory-survey-q07",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements correctly describe token-level memory?",
    options: [
      {
        text: "It stores memory as explicit, discrete units that can be accessed, modified, reorganized, and inspected outside model parameters.",
        isCorrect: true,
      },
      {
        text: "Its units may include text chunks, visual tokens, audio frames, summaries, trajectories, or multimodal entries.",
        isCorrect: true,
      },
      {
        text: "Its transparency makes it useful for retrieval, routing, conflict handling, coordination, auditing, and high-stakes provenance.",
        isCorrect: true,
      },
      {
        text: "It can be integrated with foundation models as a plug-and-play layer when memory is kept in external records or structures.",
        isCorrect: true,
      },
    ],
    explanation:
      "Token-level memory is the most explicit memory form: it is stored as addressable units outside the base model weights. That visibility supports editing and governance, while the range of possible units can extend beyond plain text into multimodal or structured records.",
  },
  {
    id: "ai-agents-memory-survey-q08",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statement best captures the main tradeoff of flat token-level memory?",
    options: [
      {
        text: "It is simple to append, prune, and retrieve from, but without explicit topology it depends heavily on retrieval quality and can struggle with relational reasoning.",
        isCorrect: true,
      },
      {
        text: "It is expensive because relational edges between entries are mandatory, even when the task only needs lightweight recall.",
        isCorrect: false,
      },
      {
        text: "It is best for multi-hop abstraction because it encodes cross-layer links between raw observations, event summaries, and thematic patterns.",
        isCorrect: false,
      },
      {
        text: "It reduces redundancy by requiring raw traces to be curated into a semantic graph before the agent can store them.",
        isCorrect: false,
      },
    ],
    explanation:
      "Flat memory is attractive because it is lightweight and scalable: entries can be stored as chunks, logs, profiles, or trajectories with minimal structural overhead. Its weakness is that relationships are not encoded directly, so the agent may retrieve relevant fragments without understanding how they compose.",
  },
  {
    id: "ai-agents-memory-survey-q09",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe planar memory?",
    options: [
      {
        text: "It introduces an explicit single-layer topology such as a graph, tree, table, or relation structure among memory units.",
        isCorrect: true,
      },
      {
        text: "It moves token-level memory from mere storage toward organization by linking related units.",
        isCorrect: true,
      },
      {
        text: "It replaces explicit retrieval over linked records with hidden activations that cannot expose graph, tree, or table relations.",
        isCorrect: false,
      },
      {
        text: "It lowers construction and search cost by keeping relationships implicit while still claiming graph-like traversal over entries.",
        isCorrect: false,
      },
    ],
    explanation:
      "Planar memory adds one structural layer of links, which helps organize facts, episodes, or concepts within a single plane. The added topology improves retrieval and reasoning but introduces construction, maintenance, and search costs, and it lacks the cross-layer abstraction of hierarchical memory.",
  },
  {
    id: "ai-agents-memory-survey-q10",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which design need most strongly motivates hierarchical memory instead of a flat or single-layer memory store?",
    options: [
      {
        text: "Navigating across multiple abstraction levels, such as raw observations, compact event summaries, and higher-level themes, while preserving cross-layer links.",
        isCorrect: true,
      },
      {
        text: "Keeping memory items as independent bag entries so similarity search can ignore links among observations and summaries.",
        isCorrect: false,
      },
      {
        text: "Avoiding summarization because hierarchical systems treat compressed event summaries as incompatible with layered storage.",
        isCorrect: false,
      },
      {
        text: "Moving memory into original model weights so retrieval is replaced by a standard forward pass.",
        isCorrect: false,
      },
    ],
    explanation:
      "Hierarchical memory is useful when an agent needs vertical movement between levels of detail and abstraction, not just lateral retrieval among peer nodes. The cost is complexity: the system must preserve semantic meaning, maintain layout quality, and retrieve efficiently through dense layered structure.",
  },
  {
    id: "ai-agents-memory-survey-q11",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly compare internal and external parametric memory?",
    options: [
      {
        text: "Internal parametric memory stores information in the original model weights, biases, or other base parameters.",
        isCorrect: true,
      },
      {
        text: "External parametric memory stores information in auxiliary parameter sets such as adapters, LoRA modules, lightweight proxy models, or routing modules.",
        isCorrect: true,
      },
      {
        text: "Internal parametric memory can avoid extra inference modules, but updating it can require costly retraining or editing and can disturb older knowledge.",
        isCorrect: true,
      },
      {
        text: "External parametric memory can support modular updates and rollback while avoiding direct distortion of the original representation space.",
        isCorrect: true,
      },
    ],
    explanation:
      "The key distinction is where the knowledge is encoded relative to the base model. Internal parametric memory changes the original model, while external parametric memory attaches additional learned modules, giving more modularity at the cost of needing a good interface with the model's computation.",
  },
  {
    id: "ai-agents-memory-survey-q12",
    chapter: 1,
    difficulty: "medium",
    type: "assertion-reason",
    prompt:
      "Assertion: Internal parametric memory is better suited to broad domain knowledge or task priors than to short personalized snippets that change frequently.\n\nReason: New internal parametric memories usually require retraining or targeted editing, which is costly and can cause forgetting or interference.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: false },
      { text: "Assertion is false, Reason is true.", isCorrect: false },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: true,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "The assertion is true because stable priors can justify the cost of internalization, while volatile personal facts are easier to store in external or token-level systems. The reason explains the design pressure: changing base parameters is expensive and can overwrite or distort existing knowledge.",
  },
  {
    id: "ai-agents-memory-survey-q13",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Why is external parametric memory described as a compromise between token-level storage and full model editing?",
    options: [
      {
        text: "It encodes memory into additional learned modules that can be attached, removed, routed, or rolled back without directly rewriting the base model.",
        isCorrect: true,
      },
      {
        text: "It stores memory as plaintext records and therefore bypasses the model's internal representation flow.",
        isCorrect: false,
      },
      {
        text: "It achieves modularity by forbidding adapters, LoRA modules, proxy models, or routing mechanisms.",
        isCorrect: false,
      },
      {
        text: "It is identical to internal parametric memory because both modify the original weights rather than using auxiliary modules.",
        isCorrect: false,
      },
    ],
    explanation:
      "External parametric memory offers more modular updates than full weight editing, because added parameter modules can be swapped or controlled. It is still not plain token storage, because its usefulness depends on how well those learned modules influence the model's attention and computation pathways.",
  },
  {
    id: "ai-agents-memory-survey-q14",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements correctly describe latent memory?",
    options: [
      {
        text: "It is carried in internal representations such as key-value caches, activations, hidden states, latent embeddings, or memory tokens.",
        isCorrect: true,
      },
      {
        text: "It can be token-efficient and useful for preserving fine-grained contextual signals, but it is usually difficult for humans to inspect directly.",
        isCorrect: true,
      },
      {
        text: "It is defined by being stored as human-readable notes that developers can manually edit in a vector database.",
        isCorrect: false,
      },
      {
        text: "It depends on permanent base-weight modification as the normal storage path.",
        isCorrect: false,
      },
    ],
    explanation:
      "Latent memory sits in machine-native internal representations rather than in readable text or dedicated parameter sets. That makes it compact and efficient for some contexts, but it trades away transparency and makes verification harder than with explicit token-level memory.",
  },
  {
    id: "ai-agents-memory-survey-q15",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which distinctions among generate, reuse, and transform latent memory are correct?",
    options: [
      {
        text: "Generate methods create new latent representations, often through learned encoding or auxiliary modules, and reuse those representations later.",
        isCorrect: true,
      },
      {
        text: "Reuse methods directly carry over prior internal computation, especially key-value caches or intermediate embeddings, without first compressing them.",
        isCorrect: true,
      },
      {
        text: "Transform methods modify, prune, pool, merge, or re-encode existing latent states to reduce footprint while preserving useful signals.",
        isCorrect: true,
      },
      {
        text: "Transform methods are the same as full parametric memory because their main operation is storing facts in the original model weights.",
        isCorrect: false,
      },
    ],
    explanation:
      "The categories differ by how the latent state enters the system. Generate creates new compact states, reuse preserves prior computation, and transform reshapes existing activations or caches; full model-weight storage is a parametric-memory idea, not the defining move of transform latent memory.",
  },
  {
    id: "ai-agents-memory-survey-q16",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A compliance-sensitive assistant must let reviewers inspect, correct, and audit the specific memories that influenced an answer. Which memory form is the best default fit?",
    options: [
      {
        text: "Token-level memory, because symbolic and addressable memory units can be inspected, edited, deleted, transferred, and linked to provenance.",
        isCorrect: true,
      },
      {
        text: "Latent memory, because unreadable hidden states maximize auditability and explain exactly which user fact was retrieved.",
        isCorrect: false,
      },
      {
        text: "Internal parametric memory, because changing base weights is the fastest way to delete a single sensitive user fact with a clear audit trail.",
        isCorrect: false,
      },
      {
        text: "Any memory form is equally suited because interpretability and update cost do not depend on representation.",
        isCorrect: false,
      },
    ],
    explanation:
      "The scenario emphasizes provenance, inspection, and precise correction, which are the strongest advantages of token-level memory. Parametric and latent memories can be powerful, but their implicit representations make fine-grained auditing and targeted deletion much harder.",
  },
  {
    id: "ai-agents-memory-survey-q17",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly match the three primary memory functions to their roles?",
    options: [
      {
        text: "Factual memory answers what the agent knows, including explicit facts, preferences, commitments, and environmental states.",
        isCorrect: true,
      },
      {
        text: "Experiential memory answers how the agent improves by abstracting from past trajectories, successes, failures, strategies, and skills.",
        isCorrect: true,
      },
      {
        text: "Working memory answers what the agent is actively thinking about now by managing the bounded workspace for a current task or session.",
        isCorrect: true,
      },
      {
        text: "Working memory is the same as long-term model weights because both store domain knowledge after training.",
        isCorrect: false,
      },
    ],
    explanation:
      "The functional taxonomy is about purpose rather than storage form. Factual memory supports declarative continuity, experiential memory supports competence accumulation, and working memory supports active in-episode reasoning under bounded context resources.",
  },
  {
    id: "ai-agents-memory-survey-q18",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which properties does factual memory support in an agent system?",
    options: [
      {
        text: "Consistency, by preserving stable behavior, user-specific facts, and prior commitments over time.",
        isCorrect: true,
      },
      {
        text: "Coherence, by integrating relevant interaction history so later responses are not isolated from prior context.",
        isCorrect: true,
      },
      {
        text: "Adaptability, by using stored profiles, historical feedback, and environmental facts to personalize future behavior.",
        isCorrect: true,
      },
      {
        text: "Automatic factual validity, because a retrievable long-term record remains current and conflict-free after storage.",
        isCorrect: false,
      },
    ],
    explanation:
      "Factual memory is meant to preserve explicit declarative information that keeps interaction stable and context-aware. It does not guarantee correctness by itself, because memory can become stale, incomplete, conflicting, or misretrieved unless evolution and governance mechanisms keep it healthy.",
  },
  {
    id: "ai-agents-memory-survey-q19",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe user factual memory?",
    options: [
      {
        text: "It can store identity facts, stable preferences, routines, task constraints, salient events, and historical commitments across sessions.",
        isCorrect: true,
      },
      {
        text: "It helps prevent coreference drift, repeated elicitation, contradictory responses, and intent drift during long-horizon interaction.",
        isCorrect: true,
      },
      {
        text: "It often combines selective retention or ranking with semantic abstraction into profiles, thoughts, reflections, or goal states.",
        isCorrect: true,
      },
      {
        text: "It injects a full dialogue transcript into the active context at each turn rather than selecting or abstracting records.",
        isCorrect: false,
      },
    ],
    explanation:
      "User factual memory turns ephemeral conversation traces into persistent facts and goal constraints. Ranking, summarization, and abstraction are important because raw logs are too large and noisy to keep in context while still preserving coherence and personalization.",
  },
  {
    id: "ai-agents-memory-survey-q20",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which example best fits environment factual memory rather than user factual memory?",
    options: [
      {
        text: "A shared memory layer records document states, codebase facts, tool availability, and other agents' capabilities so a team can coordinate across stages.",
        isCorrect: true,
      },
      {
        text: "A personal assistant remembers that one user prefers concise weekly summaries and avoids late-night notifications.",
        isCorrect: false,
      },
      {
        text: "A chatbot stores a user's name, household routines, and long-term diet constraints for future personalization.",
        isCorrect: false,
      },
      {
        text: "A dialogue agent recalls a user's earlier emotional disclosure to maintain an appropriate persona in later turns.",
        isCorrect: false,
      },
    ],
    explanation:
      "Environment factual memory concerns entities and states external to a specific user, such as documents, tools, code, resources, or shared workspaces. User factual memory is still factual, but it centers on the user's identity, preferences, commitments, and interaction history.",
  },
  {
    id: "ai-agents-memory-survey-q21",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe experiential memory categories?",
    options: [
      {
        text: "Case-based memory preserves relatively raw episodes, trajectories, or solutions so they can be replayed or adapted as examples.",
        isCorrect: true,
      },
      {
        text: "Strategy-based memory distills reusable insights, workflows, patterns, or heuristics that guide planning without directly executing actions.",
        isCorrect: true,
      },
      {
        text: "Skill-based memory stores callable and verifiable capabilities such as code snippets, functions, scripts, APIs, or tool interfaces.",
        isCorrect: true,
      },
      {
        text: "Hybrid experiential memory can combine cases, strategies, and skills so agents can move from grounded evidence to general rules and executable procedures.",
        isCorrect: true,
      },
    ],
    explanation:
      "Experiential memory is about learning from prior task execution. Cases preserve evidence, strategies abstract reusable logic, skills operationalize capability, and hybrid systems combine them so repeated experience can mature into faster and more reliable action.",
  },
  {
    id: "ai-agents-memory-survey-q22",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statement best distinguishes case-based memory from strategy-based memory?",
    options: [
      {
        text: "Case-based memory keeps specific historical situations and solutions with high fidelity, while strategy-based memory abstracts cross-situational rules, workflows, or reasoning patterns.",
        isCorrect: true,
      },
      {
        text: "Case-based memory stores executable APIs and callable tool interfaces, while strategy-based memory stores raw sensor logs without abstraction or planning reuse.",
        isCorrect: false,
      },
      {
        text: "Case-based memory removes context-window costs, while strategy-based memory is forced to replay full trajectories at inference.",
        isCorrect: false,
      },
      {
        text: "Case-based memory is parametric by definition, while strategy-based memory is latent by definition.",
        isCorrect: false,
      },
    ],
    explanation:
      "Case memory is evidence-rich and replayable, but it can consume retrieval and context budget because raw episodes are bulky. Strategy memory compresses experience into reusable planning knowledge, improving generalization but lacking direct executability by itself.",
  },
  {
    id: "ai-agents-memory-survey-q23",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe skill-based memory?",
    options: [
      {
        text: "It anchors the perception-reasoning-action loop by storing capabilities the agent can invoke and verify.",
        isCorrect: true,
      },
      {
        text: "It can include code snippets, modular functions, scripts, APIs, Model Context Protocol (MCP) servers, or callable specialized agents.",
        isCorrect: true,
      },
      {
        text: "Its usefulness depends on skills being callable, outcomes being checkable, and capabilities being composable into larger routines.",
        isCorrect: true,
      },
      {
        text: "It remains separate from grounded execution because stored skills are descriptive notes rather than invocable procedures.",
        isCorrect: false,
      },
    ],
    explanation:
      "Skill memory is the execution-facing side of experiential memory. It turns abstract strategies and past successes into invocable procedures, so it is not merely advice; it can call tools, run code, interact through APIs, and produce verifiable feedback for future learning.",
  },
  {
    id: "ai-agents-memory-survey-q24",
    chapter: 1,
    difficulty: "hard",
    type: "assertion-reason",
    prompt:
      "Assertion: Hybrid experiential memory can support a transition from expensive retrieval of specific cases toward faster execution through compiled skills.\n\nReason: A strategy store such as a dynamic cheatsheet can retain accumulated problem-solving insights for immediate reuse at inference time.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: false },
      { text: "Assertion is false, Reason is true.", isCorrect: false },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: false,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: true,
      },
    ],
    explanation:
      "The assertion is true because hybrid systems can couple raw episodes, distilled strategies, and executable skill modules, and repeated successes may be compiled into callable capabilities. The reason is also true, but it describes strategy reuse rather than the full case-to-skill compilation mechanism that explains the assertion.",
  },
  {
    id: "ai-agents-memory-survey-q25",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements correctly describe working memory for agents?",
    options: [
      {
        text: "It actively selects, maintains, compresses, rewrites, or transforms task-relevant information within a single episode.",
        isCorrect: true,
      },
      {
        text: "It aims to turn the context window from a passive read-only buffer into a controllable, updatable workspace.",
        isCorrect: true,
      },
      {
        text: "It helps suppress redundancy, manage interference, and preserve coherent reasoning under bounded attention resources.",
        isCorrect: true,
      },
      {
        text: "A longer context window supplies selection, updating, compression, and interference control as built-in properties of generation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Working memory is about active control, not simply the existence of a context window. Even a long context can become noisy, saturated, or misaligned, so agents need mechanisms that choose what stays active and how transient information is represented.",
  },
  {
    id: "ai-agents-memory-survey-q26",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly compare single-turn working-memory mechanisms?",
    options: [
      {
        text: "Hard condensation discretely selects or drops tokens based on importance, which is efficient but can break semantic or syntactic dependencies.",
        isCorrect: true,
      },
      {
        text: "Soft condensation encodes variable-length inputs into dense latent slots or summary tokens, often requiring training and reducing interpretability.",
        isCorrect: true,
      },
      {
        text: "Observation abstraction maps high-dimensional or verbose observations into structured state descriptions rather than merely shortening text.",
        isCorrect: true,
      },
      {
        text: "Single-turn working memory solves multi-session goal drift by permanently revising the agent's long-term user profile.",
        isCorrect: false,
      },
    ],
    explanation:
      "Single-turn working memory handles the breadth and complexity of immediate inputs, such as long documents, web pages, or video streams. It can condense tokens or abstract observations, but it is not the same as long-term user memory or cross-session profile maintenance.",
  },
  {
    id: "ai-agents-memory-survey-q27",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe multi-turn working-memory strategies?",
    options: [
      {
        text: "State consolidation maps an ever-growing interaction trajectory into a compact evolving state to reduce redundancy and preserve task-relevant history.",
        isCorrect: true,
      },
      {
        text: "Hierarchical folding keeps fine-grained traces active during a subgoal and folds completed sub-trajectories into higher-level summaries.",
        isCorrect: true,
      },
      {
        text: "Cognitive planning maintains an external plan or world model that guides future actions, not merely a summary of past turns.",
        isCorrect: true,
      },
      {
        text: "Multi-turn working memory avoids reading, writing, and updating because history should remain an unmodified transcript.",
        isCorrect: false,
      },
    ],
    explanation:
      "In long-horizon interaction, the bottleneck is maintaining useful state as history grows. Consolidation, folding, and planning all externalize and revise task state so the agent can continue coherently without flooding the active context with every prior turn.",
  },
  {
    id: "ai-agents-memory-survey-q28",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the dynamic lifecycle of agent memory?",
    options: [
      {
        text: "Memory formation turns raw experiences into more compact and useful memory units instead of passively storing everything.",
        isCorrect: true,
      },
      {
        text: "Memory evolution integrates new memories with the existing repository through operations such as consolidation, updating, and forgetting.",
        isCorrect: true,
      },
      {
        text: "Memory retrieval uses context-aware queries and strategies to return memory content at the moment it can support reasoning.",
        isCorrect: true,
      },
      {
        text: "The lifecycle is cyclic because reasoning outcomes and environmental feedback can later become material for formation and evolution.",
        isCorrect: true,
      },
    ],
    explanation:
      "The dynamic view treats memory as a continuously managed subsystem rather than a static database. Formation extracts, evolution refines, retrieval uses, and the results of action feed back into the next round of memory management.",
  },
  {
    id: "ai-agents-memory-survey-q29",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Why can semantic summarization be a poor fit for evidence-critical tasks even though it is useful for long-term memory?",
    options: [
      {
        text: "It is a lossy compression mechanism that preserves high-level gist while smoothing out details and subtle cues that may matter for evidence.",
        isCorrect: true,
      },
      {
        text: "It stores raw verbatim logs and therefore keeps context length and overhead close to the original interaction history.",
        isCorrect: false,
      },
      {
        text: "It embeds summaries into base model weights before retrieval, so the summary cannot remain an external memory unit.",
        isCorrect: false,
      },
      {
        text: "It discards global semantics, narrative flow, task procedure, and historical user profiles in favor of local factual snippets.",
        isCorrect: false,
      },
    ],
    explanation:
      "Semantic summarization is valuable when an agent needs compact global context, such as a dialogue profile or task narrative. Its weakness is resolution loss: a clean summary can omit precise facts, exceptions, or evidential details that a high-stakes or source-grounded task needs.",
  },
  {
    id: "ai-agents-memory-survey-q30",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly compare knowledge distillation and structured construction as memory formation operations?",
    options: [
      {
        text: "Knowledge distillation extracts reusable factual or experiential knowledge units such as user facts, goal states, planning insights, or procedural knowledge.",
        isCorrect: true,
      },
      {
        text: "Structured construction organizes source data into explicit topology such as entity graphs, temporal graphs, trees, or hierarchical memory networks.",
        isCorrect: true,
      },
      {
        text: "Knowledge distillation is the graph-building stage, while structured construction is the flat-note extraction stage.",
        isCorrect: false,
      },
      {
        text: "Structured construction removes interpretability and retrieval benefits because links and layers are hidden from the memory system.",
        isCorrect: false,
      },
    ],
    explanation:
      "Knowledge distillation focuses on what reusable information should be extracted, while structured construction focuses on how extracted or raw information should be linked and layered. Both can be combined, but they solve different parts of the formation problem.",
  },
  {
    id: "ai-agents-memory-survey-q31",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe latent representation and parametric internalization as formation operations?",
    options: [
      {
        text: "Latent representation encodes experiences into machine-native vectors, embeddings, key-value states, or other continuous internal formats.",
        isCorrect: true,
      },
      {
        text: "Parametric internalization converts retrievable information into model competence or instincts through parameter updates such as editing, LoRA training, or fine-tuning.",
        isCorrect: true,
      },
      {
        text: "Latent representation is the same as storing plaintext snippets because both expose the same readable memory surface to developers.",
        isCorrect: false,
      },
      {
        text: "Parametric internalization keeps volatile user facts as separate external records with stable identifiers, making rapid deletion and audit-friendly correction the cheap default operation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Latent formation and parametric internalization both move away from readable token storage, but they do so in different ways. Latent representations are continuous memory units used during computation, while parametric internalization embeds knowledge or capability into learned weights or auxiliary learned modules.",
  },
  {
    id: "ai-agents-memory-survey-q32",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe memory evolution operations?",
    options: [
      {
        text: "Consolidation reorganizes fragmented traces into coherent schemas by local merging, cluster fusion, or global integration.",
        isCorrect: true,
      },
      {
        text: "Updating revises or replaces existing memory when conflicts or new facts arise, using external-store updates or model editing.",
        isCorrect: true,
      },
      {
        text: "Forgetting deliberately removes or weakens outdated, redundant, or low-value information to control noise and retrieval cost.",
        isCorrect: true,
      },
      {
        text: "Evolution becomes redundant after formation because new facts, conflicts, duplication, and retrieval cost are handled before storage.",
        isCorrect: false,
      },
    ],
    explanation:
      "Consolidation, updating, and forgetting are all parts of memory evolution, but they address different pressures: abstraction, correction, and overload. A formed memory can later conflict with new facts, duplicate other entries, become stale, or need to be merged into a broader schema.",
  },
  {
    id: "ai-agents-memory-survey-q33",
    chapter: 1,
    difficulty: "hard",
    type: "assertion-reason",
    prompt:
      "Assertion: Memory updating faces a stability-plasticity dilemma because the agent must decide when to revise knowledge and when to treat new information as noise.\n\nReason: Time-based forgetting is the main mechanism for resolving factual conflicts because it edits the model's parameter region that encodes a disputed fact.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: true },
      { text: "Assertion is false, Reason is true.", isCorrect: false },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: false,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "The assertion is true: updating can keep memory fresh, but incorrect updates can overwrite valuable knowledge and degrade reasoning. The reason is false because time-based forgetting handles temporal decay or eviction, while conflict resolution is usually handled through external memory update strategies or model editing.",
  },
  {
    id: "ai-agents-memory-survey-q34",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe forgetting mechanisms?",
    options: [
      {
        text: "Time-based forgetting weakens or evicts memories according to age, mimicking temporal decay or context-overflow eviction.",
        isCorrect: true,
      },
      {
        text: "Frequency-based forgetting uses retrieval behavior, such as least-frequently-used or least-recently-used policies, to decide what to retain.",
        isCorrect: true,
      },
      {
        text: "Importance-driven forgetting can integrate temporal, frequency, semantic, contextual, or affective signals to preserve high-value knowledge.",
        isCorrect: true,
      },
      {
        text: "Low access frequency is treated as definitive evidence that a rare memory has no future decision value.",
        isCorrect: false,
      },
    ],
    explanation:
      "Forgetting controls overload, but each signal has failure modes. A frequency heuristic can mistakenly delete rare long-tail knowledge, while importance-driven methods try to add semantic judgment so the system keeps memories that are seldom used but still crucial.",
  },
  {
    id: "ai-agents-memory-survey-q35",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which steps belong to the memory retrieval pipeline?",
    options: [
      {
        text: "Retrieval timing and intent, which decide when retrieval should occur and which memory source should be queried.",
        isCorrect: true,
      },
      {
        text: "Query construction, which decomposes or rewrites the raw request into more effective retrieval signals.",
        isCorrect: true,
      },
      {
        text: "Retrieval strategies, which perform search through lexical, semantic, graph-based, hybrid, or related mechanisms.",
        isCorrect: true,
      },
      {
        text: "Post-retrieval processing, which reranks, filters, aggregates, or compresses retrieved fragments into usable context.",
        isCorrect: true,
      },
    ],
    explanation:
      "Retrieval is not a single similarity search call in the dynamic memory view. A robust pipeline decides when and where to retrieve, translates the problem into a useful query, executes the search, and then shapes the retrieved material into concise context.",
  },
  {
    id: "ai-agents-memory-survey-q36",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "What is the main silent-failure risk of autonomous retrieval timing?",
    options: [
      {
        text: "The agent may overestimate its internal knowledge, skip retrieval when it is needed, and then produce unsupported or hallucinated output.",
        isCorrect: true,
      },
      {
        text: "Extra retrieval remains harmless because added context improves reasoning regardless of relevance, source conflict, attention budget, or downstream synthesis quality.",
        isCorrect: false,
      },
      {
        text: "Choosing a graph index instead of lexical search makes the later answer uncheckable even when the retrieved path is visible.",
        isCorrect: false,
      },
      {
        text: "Generating a query rewrite turns off later filtering stages and forces raw retrieved fragments into the final context.",
        isCorrect: false,
      },
    ],
    explanation:
      "Autonomous timing can reduce overhead and noise, but it introduces a decision risk: the agent might decide not to look up information it actually lacks. Excessive retrieval has its own cost through noise and context bloat, so the central challenge is balancing essential retrieval against unnecessary retrieval.",
  },
  {
    id: "ai-agents-memory-survey-q37",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly compare query decomposition and query rewriting?",
    options: [
      {
        text: "Query decomposition breaks a complex request into sub-queries so the system can retrieve finer-grained evidence and reason over intermediate results.",
        isCorrect: true,
      },
      {
        text: "Query rewriting changes the original query or generates a hypothetical document to better align user intent with the memory index.",
        isCorrect: true,
      },
      {
        text: "Query decomposition is restricted to memories embedded in model weights rather than external or structured stores.",
        isCorrect: false,
      },
      {
        text: "Query rewriting is a post-retrieval operation that discards already retrieved fragments rather than improving the retrieval signal.",
        isCorrect: false,
      },
    ],
    explanation:
      "Both techniques sit before or during retrieval construction, and both try to reduce mismatch between the user's surface request and the organization of memory. Decomposition creates multiple targeted retrieval paths, while rewriting creates a better single retrieval signal or hypothetical semantic target.",
  },
  {
    id: "ai-agents-memory-survey-q38",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe retrieval strategies and post-retrieval processing?",
    options: [
      {
        text: "Lexical retrieval such as TF-IDF or BM25 is fast and interpretable but can miss semantic paraphrases and multimodal relations.",
        isCorrect: true,
      },
      {
        text: "Semantic retrieval embeds queries and memories into a shared space, improving fuzzy matching but depending on encoder quality and representation fit.",
        isCorrect: true,
      },
      {
        text: "Graph retrieval can exploit structured relations for multi-hop reasoning, while reranking and filtering can reduce irrelevant or redundant fragments after search.",
        isCorrect: true,
      },
      {
        text: "Post-retrieval aggregation should concatenate raw search results with minimal filtering to avoid dropping edge cases.",
        isCorrect: false,
      },
    ],
    explanation:
      "Different retrieval strategies trade off interpretability, semantic generalization, and structured reasoning. Post-retrieval processing exists because raw search results are often noisy or too large, so useful memory access depends on shaping the final context as well as finding candidate fragments.",
  },
  {
    id: "ai-agents-memory-survey-q39",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the shift from memory retrieval toward memory generation?",
    options: [
      {
        text: "Retrieval-centric systems focus on identifying and filtering relevant entries from an existing memory store.",
        isCorrect: true,
      },
      {
        text: "Generative memory synthesizes new representations by integrating, compressing, or reorganizing information for the current and anticipated future context.",
        isCorrect: true,
      },
      {
        text: "A retrieve-then-generate approach can use retrieved memories as raw material for a refined, coherent, context-specific memory representation.",
        isCorrect: true,
      },
      {
        text: "Direct memory generation is equivalent to vector search over token records because it begins with explicit database lookup.",
        isCorrect: false,
      },
    ],
    explanation:
      "The frontier shift is from memory as a static repository toward memory as something the agent can actively synthesize. Retrieve-then-generate preserves grounding in stored material, while direct generation can construct latent or contextual memory representations without an explicit lookup step.",
  },
  {
    id: "ai-agents-memory-survey-q40",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements reflect likely frontiers for future agent memory systems?",
    options: [
      {
        text: "Reinforcement learning may move memory control from hand-engineered pipelines toward learned policies for formation, evolution, retrieval, and even architecture design.",
        isCorrect: true,
      },
      {
        text: "Shared memory in multi-agent systems may need role-aware read/write behavior, conflict management, and learned contribution policies rather than naive global sharing.",
        isCorrect: true,
      },
      {
        text: "Trustworthy memory will need privacy controls, auditable access paths, verifiable forgetting, and hallucination robustness because persistent memories can be sensitive and influential.",
        isCorrect: true,
      },
      {
        text: "Future memory systems should treat text as the single practical modality for embodied, visual, audio, and world-model settings, converting sensor histories away before memory design begins.",
        isCorrect: false,
      },
    ],
    explanation:
      "The frontier directions push memory toward learned, adaptive, multimodal, collaborative, and trustworthy systems. Multi-agent sharing introduces governance problems, RL may optimize the memory lifecycle, and persistent sensitive memory requires privacy and auditability; text-only memory is too narrow for embodied and world-model agents.",
  },
];
