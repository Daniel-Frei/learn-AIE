import { Question } from "../../quiz";

export const DeepAgentsQuestions: Question[] = [
  {
    id: "langchain-deepagents-q01",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the basic ReAct-style agent loop used as the foundation for deep agents?",
    options: [
      {
        text: "The language model can reason about whether to call a tool again after seeing tool feedback.",
        isCorrect: true,
      },
      {
        text: "The loop alternates between the model and tool execution exactly once before producing a final answer.",
        isCorrect: false,
      },
      {
        text: "The tool node directly decides the final answer without the language model seeing the tool result.",
        isCorrect: false,
      },
      {
        text: "ReAct requires a separate planner model and cannot work with a single language model plus tools.",
        isCorrect: false,
      },
    ],
    explanation:
      "A ReAct-style agent is built around a simple loop: the model reasons, calls a tool if needed, observes the result, and then decides what to do next. The final answer usually still comes from the model after it has seen the tool output, so the tool node does not independently decide the final response.",
  },
  {
    id: "langchain-deepagents-q02",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Why are planning tools such as todo lists useful in deep agents that operate over longer time horizons?",
    options: [
      {
        text: "They can help the agent stay aligned with a multi-step objective across many turns.",
        isCorrect: true,
      },
      {
        text: "They can support workflows where the user reviews or approves a plan before actions are taken.",
        isCorrect: true,
      },
      {
        text: "They can be reread later to remind the agent of its earlier intent after many tool calls.",
        isCorrect: true,
      },
      {
        text: "They do not eliminate the need for prompting because the todo list does not fully determine the agent's behavior.",
        isCorrect: true,
      },
    ],
    explanation:
      "Planning gives the agent an explicit structure for multi-step work and makes it easier to recover direction after a long trajectory. It does not replace prompting, though; the system prompt still matters a lot for telling the agent when to write, read, and update plans.",
  },
  {
    id: "langchain-deepagents-q03",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statement best captures the role of the file system in deep agents?",
    options: [
      {
        text: "It mainly exists so the language model can permanently avoid tool calls.",
        isCorrect: false,
      },
      {
        text: "It serves as externalized memory that can store information outside the immediate message context and be read back later.",
        isCorrect: true,
      },
      {
        text: "It is useful only for code agents and not for research or general-purpose agents.",
        isCorrect: false,
      },
      {
        text: "It replaces state and makes graph state unnecessary.",
        isCorrect: false,
      },
    ],
    explanation:
      "The core idea is that files let the agent store context outside the message list and retrieve it later on demand. This is helpful for many agent types, not just coding agents, and it complements graph state rather than replacing it.",
  },
  {
    id: "langchain-deepagents-q04",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about `AgentState` and state handling in LangGraph-style agent setups are correct?",
    options: [
      {
        text: "A default agent state commonly includes a `messages` field that stores the conversation and tool-related messages.",
        isCorrect: true,
      },
      {
        text: "A reducer such as `add_messages` controls how new message entries are merged into state.",
        isCorrect: true,
      },
      {
        text: "You can extend the default state with additional fields such as `todos`, `files`, or a list of prior operations.",
        isCorrect: true,
      },
      {
        text: "Tracking state is useful not only for passing context between nodes, but also for debugging and recovering long-running workflows.",
        isCorrect: true,
      },
    ],
    explanation:
      "State is one of the main organizing ideas in LangGraph-style agents. The default message state can be extended with custom fields, and reducers define how updates are merged when multiple nodes or parallel branches write to the same field.",
  },
  {
    id: "langchain-deepagents-q05",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Suppose a tool needs access to graph state but you do not want the language model to generate that state as part of the tool call. Which approaches are appropriate?",
    options: [
      {
        text: "Use an injected argument such as `Annotated[MyState, InjectedState]` so the framework provides state at execution time.",
        isCorrect: true,
      },
      {
        text: "Force the model to serialize the full current state into every tool call so the tool node can reconstruct it.",
        isCorrect: false,
      },
      {
        text: "Use an injected tool call identifier when you need to construct a `ToolMessage` tied to that specific call, while still serializing the full hidden state into the prompt.",
        isCorrect: false,
      },
      {
        text: "Avoid tools entirely, because injected arguments are incompatible with tool-based agents.",
        isCorrect: false,
      },
    ],
    explanation:
      "Injected arguments solve the key problem that the model should not have to generate hidden runtime details such as full graph state or tool call IDs. The framework can provide those values during tool execution, which keeps the model interface simpler and more reliable.",
  },
  {
    id: "langchain-deepagents-q06",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe how a todo-writing tool can be used in a deep agent?",
    options: [
      {
        text: "The tool can directly update a `todos` field in state by returning a `Command` object.",
        isCorrect: true,
      },
      {
        text: "A separate todo-reading tool can help the agent reread its current plan later in the trajectory.",
        isCorrect: true,
      },
      {
        text: "The tool can also write a `ToolMessage` so the message history reflects that the todo list was updated.",
        isCorrect: true,
      },
      {
        text: "Even after todos are stored in state, the agent may still need to check them again later in the trajectory.",
        isCorrect: true,
      },
    ],
    explanation:
      "In the course setup, todo tools are used both to write plan information into state and to expose useful feedback in the message list. Reading todos again later is part of the point, because the agent may need to refresh its plan after multiple actions.",
  },
  {
    id: "langchain-deepagents-q07",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statement best explains a key risk of using subagents?",
    options: [
      {
        text: "Subagents are risky mainly because they always share the same context window as the supervisor.",
        isCorrect: false,
      },
      {
        text: "Subagents are problematic only when they perform research, but not when they generate or modify artifacts.",
        isCorrect: false,
      },
      {
        text: "If multiple subagents independently make design or implementation decisions, their outputs may conflict even if each subagent did reasonable local work.",
        isCorrect: true,
      },
      {
        text: "A subagent can never be run in parallel with another subagent because the framework prohibits it.",
        isCorrect: false,
      },
    ],
    explanation:
      "The main risk is not that subagents are useless, but that they can make locally sensible yet globally inconsistent choices. This is why they are often safest when delegated narrowly scoped, parallelizable work such as gathering evidence rather than independently authoring tightly coupled pieces of a system.",
  },
  {
    id: "langchain-deepagents-q08",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "A virtual file system is represented as a dictionary where keys are paths and values are file contents. Which behaviors are desirable in that setup?",
    options: [
      {
        text: "A list-files tool can inspect the current keys in state and return available file paths.",
        isCorrect: true,
      },
      {
        text: "A write-file tool can update an existing path only by creating a brand-new key each time.",
        isCorrect: false,
      },
      {
        text: "Every write must create a completely new dictionary key, even when the intention is to revise an existing file.",
        isCorrect: false,
      },
      {
        text: "The file system can only be useful if files are stored on the host machine rather than in graph state.",
        isCorrect: false,
      },
    ],
    explanation:
      "The course uses a mock file system in state, which is enough to demonstrate the pattern. Listing keys and allowing later writes to overwrite earlier values are both natural behaviors for a file abstraction built on a dictionary.",
  },
  {
    id: "langchain-deepagents-q09",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements are valid reasons to offload raw tool observations to files instead of always appending the full observations to the message list?",
    options: [
      {
        text: "It helps conserve context-window space by keeping token-heavy outputs out of the main conversation history.",
        isCorrect: true,
      },
      {
        text: "It allows the agent to keep a compact summary in context while still preserving the fuller result for later retrieval.",
        isCorrect: true,
      },
      {
        text: "It can make long-running trajectories more robust by preserving recoverable information outside the immediate working context.",
        isCorrect: true,
      },
      {
        text: "It supports a pattern where the agent sees filenames or references in summaries and can reopen the detailed content later if needed.",
        isCorrect: true,
      },
    ],
    explanation:
      "This is one of the central context-engineering ideas in the course. The message list stays smaller and more focused, while the agent still has a path back to the underlying detailed content through files.",
  },
  {
    id: "langchain-deepagents-q10",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "In the research-oriented deep agent, the search tool processes web results before returning information to the supervisor. Which statements are correct?",
    options: [
      {
        text: "A useful pattern is to fetch webpage content, convert it to a cleaner representation such as markdown, and then summarize it.",
        isCorrect: true,
      },
      {
        text: "Returning only summaries to the supervisor while saving raw content to files is a deliberate token-efficiency strategy.",
        isCorrect: true,
      },
      {
        text: "Generating a filename alongside a summary can help the agent reference the stored raw content later.",
        isCorrect: true,
      },
      {
        text: "Because the raw content is stored in files, the agent can inspect that content again during the same trajectory when needed.",
        isCorrect: true,
      },
    ],
    explanation:
      "The whole point of the pattern is to preserve detail without flooding the active context. The agent can later reopen the stored file if the summary is not enough for the next reasoning step.",
  },
  {
    id: "langchain-deepagents-q11",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statement about the `think` tool is most accurate in this course's setup?",
    options: [
      {
        text: "It is mainly a no-op tool that creates an explicit opportunity for structured reflection between other actions.",
        isCorrect: true,
      },
      {
        text: "It directly performs web searches and returns ranked URLs to the model.",
        isCorrect: false,
      },
      {
        text: "It permanently stores all agent memories into the file system.",
        isCorrect: false,
      },
      {
        text: "It replaces the need for a todo list because both tools serve exactly the same function.",
        isCorrect: false,
      },
    ],
    explanation:
      "The `think` tool is not a retrieval or storage mechanism by itself. Its main purpose is to force a pause for reflection so the agent can assess what it has learned, what is missing, and what to do next.",
  },
  {
    id: "langchain-deepagents-q12",
    chapter: 1,
    difficulty: "easy",
    prompt: "Why does context isolation make subagents useful?",
    options: [
      {
        text: "A subagent can receive a narrower task and a smaller, more specialized toolset than the supervisor.",
        isCorrect: true,
      },
      {
        text: "A subagent must inherit the supervisor's full working history instead of using its own isolated context window.",
        isCorrect: false,
      },
      {
        text: "A subagent is useful only if it has access to every tool available to the supervisor.",
        isCorrect: false,
      },
      {
        text: "Context isolation matters only for memory usage and not for correctness or focus.",
        isCorrect: false,
      },
    ],
    explanation:
      "Subagents help because they can work in a cleaner and more focused context, which reduces distraction and confusion. Restricting the task and tool set often improves reliability, not just token usage.",
  },
  {
    id: "langchain-deepagents-q13",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe how a supervisor uses a task-delegation tool to call subagents?",
    options: [
      {
        text: "The supervisor can choose a subagent by name or type and provide a task description as the delegated work item.",
        isCorrect: true,
      },
      {
        text: "The delegation tool can pass shared state such as the file system to the subagent while replacing the subagent's message history with an isolated task-specific message list.",
        isCorrect: true,
      },
      {
        text: "After the subagent finishes, the supervisor can receive the subagent's final answer as tool output rather than its full internal trace.",
        isCorrect: true,
      },
      {
        text: "If the subagent modifies shared files, those changes can be propagated back into the supervisor's state.",
        isCorrect: true,
      },
    ],
    explanation:
      "This is the key pattern shown in the subagent lesson: isolate the subagent's conversational context while still allowing it to operate on shared external state such as files. The supervisor usually sees only the subagent's final response, which keeps the parent context much smaller.",
  },
  {
    id: "langchain-deepagents-q14",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about prompting in deep agents are supported by the course?",
    options: [
      {
        text: "Even with strong models, detailed system prompts are still important for teaching the agent how to use tools well.",
        isCorrect: true,
      },
      {
        text: "Prompting often includes explicit usage guidance for mechanisms such as todos, files, and subagent delegation.",
        isCorrect: true,
      },
      {
        text: "Long-running agents frequently rely on careful prompt engineering rather than only architectural complexity.",
        isCorrect: true,
      },
      {
        text: "Even when an agent has subagents, the supervisor's prompt still benefits from more than a single sentence of guidance.",
        isCorrect: true,
      },
    ],
    explanation:
      "A major theme of the course is that prompting still matters a lot, sometimes more than people expect. The best deep agents usually have substantial instructions about how to plan, when to delegate, how to use files, and how to stay on task over long trajectories.",
  },
  {
    id: "langchain-deepagents-q15",
    chapter: 1,
    difficulty: "hard",
    prompt:
      'A tool returns the following object:\n\\[\n\\texttt{Command(update=\\{"ops": ops, "messages": [ToolMessage(...)]\\})}\n\\]\nWhat is the most important consequence?',
    options: [
      {
        text: "The framework treats the command as a direct state update, allowing the tool to modify fields beyond a plain textual observation.",
        isCorrect: true,
      },
      {
        text: "The language model must have generated the full `Command` object token by token as part of the tool call.",
        isCorrect: false,
      },
      {
        text: "Returning a `Command` prevents the tool from updating the `messages` field.",
        isCorrect: false,
      },
      {
        text: "A command can be used only for control flow and not for state updates.",
        isCorrect: false,
      },
    ],
    explanation:
      "Returning a `Command` is a mechanism for the tool implementation, not something the model must explicitly compose in its tool-call arguments. It lets the tool update arbitrary parts of state, including but not limited to `messages`, and in some cases also influence control flow.",
  },
  {
    id: "langchain-deepagents-q16",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "The course notes that some tool calls can execute in parallel. Which statements are correct?",
    options: [
      {
        text: "Parallel tool execution can occur when the model emits multiple tool calls that the tool node can run concurrently.",
        isCorrect: true,
      },
      {
        text: "Reducers become important because parallel branches may need to merge updates into shared state safely.",
        isCorrect: true,
      },
      {
        text: "Parallel tool execution means the language model never has to see tool outputs before answering.",
        isCorrect: false,
      },
      {
        text: "If tools run in parallel once, the graph can no longer continue iterating afterward.",
        isCorrect: false,
      },
    ],
    explanation:
      "Parallel execution is compatible with the broader agent loop; after tools finish, their outputs can still be merged and sent back to the model. This is one reason reducers matter: multiple updates may need to be combined cleanly into shared state.",
  },
  {
    id: "langchain-deepagents-q17",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements reflect the course's overall picture of what makes an agent 'deep' rather than just a short-horizon ReAct loop?",
    options: [
      {
        text: "Planning helps the agent stay organized across longer trajectories.",
        isCorrect: true,
      },
      {
        text: "A file system or comparable external memory mechanism helps offload context outside the active message list.",
        isCorrect: true,
      },
      {
        text: "Subagents provide focused context isolation for delegated work.",
        isCorrect: true,
      },
      {
        text: "Prompting is treated as a major steering mechanism rather than an afterthought.",
        isCorrect: true,
      },
    ],
    explanation:
      "The course repeatedly returns to these four pillars: planning, files, subagents, and prompting. The point is not that the base ReAct loop disappears, but that these additions make it much more capable over longer and more complex tasks.",
  },
  {
    id: "langchain-deepagents-q18",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Consider a supervisor-subagent research system. Which design choices reduce the chance of the overall system becoming confused or bloated?",
    options: [
      {
        text: "Have the supervisor delegate a narrowly scoped research topic to a specialized subagent instead of giving it every responsibility at once.",
        isCorrect: true,
      },
      {
        text: "Return a concise final report from the subagent to the supervisor rather than the subagent's full internal message history.",
        isCorrect: true,
      },
      {
        text: "Store large raw observations in files and expose lighter summaries to the supervisor whenever possible.",
        isCorrect: true,
      },
      {
        text: "Ensure that every subagent writes independent sections of a tightly coupled artifact without any central synthesis step.",
        isCorrect: false,
      },
    ],
    explanation:
      "Narrow delegation, compressed supervisor context, and external storage of large observations all support cleaner context management. The risky choice is letting multiple subagents independently author tightly interdependent outputs without a strong integration strategy.",
  },
  {
    id: "langchain-deepagents-q19",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statement about user approval and planning is most consistent with the course material?",
    options: [
      {
        text: "Some agents use planning not only for internal steering, but also as a checkpoint where the user can review or approve the planned work.",
        isCorrect: true,
      },
      {
        text: "User approval is incompatible with long-horizon agents because it interrupts the loop.",
        isCorrect: false,
      },
      {
        text: "A planning step is useful only when the agent has no tools.",
        isCorrect: false,
      },
      {
        text: "If a plan is approved once, there is no reason to reread it later in the trajectory.",
        isCorrect: false,
      },
    ],
    explanation:
      "The course mentions several examples where planning also serves as a user-alignment checkpoint before action. Approval and rereading are both steering mechanisms; they help the agent stay connected to the user's goal even over many later steps.",
  },
  {
    id: "langchain-deepagents-q20",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "In the full deep-agent research example, which behaviors are part of the intended workflow?",
    options: [
      {
        text: "Check whether relevant files already exist before starting new work.",
        isCorrect: true,
      },
      {
        text: "Write down the user's request so it can be revisited later in the trajectory.",
        isCorrect: true,
      },
      {
        text: "Create and update todos as progress is made on research and analysis tasks.",
        isCorrect: true,
      },
      {
        text: "Delegate focused research work to a subagent, then use reflection and plan updates before forming the final response.",
        isCorrect: true,
      },
    ],
    explanation:
      "The final example intentionally interleaves the main building blocks from the earlier lessons. It checks files, records the request, manages todos, delegates research, reflects with the think tool, and then updates the plan before answering.",
  },

  {
    id: "langchain-deepagents-q21",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the Agent Client Protocol \\(ACP\\) integration for deep agents?",
    options: [
      {
        text: "Agent Client Protocol standardizes communication between coding agents and editors or integrated development environments.",
        isCorrect: true,
      },
      {
        text: "An ACP server commonly runs over standard input and standard output when launched by an editor-side client.",
        isCorrect: true,
      },
      {
        text: "ACP is the primary protocol for letting your agent call tools hosted on external servers.",
        isCorrect: false,
      },
      {
        text: "ACP support means a custom deep agent can be exposed to compatible editors only if it stops using tools.",
        isCorrect: false,
      },
    ],
    explanation:
      "ACP is about agent-editor integration, not about external tool hosting. In the documentation, external hosted tools are instead associated with Model Context Protocol, while ACP lets editors launch and communicate with your agent, often through a stdio server process.",
  },
  {
    id: "langchain-deepagents-q22",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "A deep agent is created with `create_deep_agent(model=...)`. Which statements about model configuration are correct?",
    options: [
      {
        text: "A model can be passed as a `provider:model` string such as `openai:gpt-5.3-codex`.",
        isCorrect: true,
      },
      {
        text: "You can pass a LangChain chat model object instead of a string model identifier.",
        isCorrect: true,
      },
      {
        text: "Deep Agents can use any chat model, even if the model does not support tool calling.",
        isCorrect: false,
      },
      {
        text: "Provider-specific parameters such as thinking budgets never require model construction helpers such as `init_chat_model(...)` or provider classes.",
        isCorrect: false,
      },
    ],
    explanation:
      "Deep Agents require tool-calling support because the harness depends on tool use for planning, files, delegation, and other actions. The docs also show that simple string configuration is convenient, but more advanced provider-specific settings are often easier through `init_chat_model` or a provider-specific chat model class.",
  },
  {
    id: "langchain-deepagents-q23",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about connection resilience and retries are correct?",
    options: [
      {
        text: "LangChain chat models automatically retry some failed API requests with exponential backoff.",
        isCorrect: true,
      },
      {
        text: "Client errors such as \\(401\\) unauthorized and \\(404\\) not found are typically not retried by default.",
        isCorrect: true,
      },
      {
        text: "For long-running tasks on unreliable networks, retries alone are enough and checkpointing is unnecessary.",
        isCorrect: false,
      },
      {
        text: "The default documented retry count is one retry for all error classes, including authorization failures.",
        isCorrect: false,
      },
    ],
    explanation:
      "The documentation describes automatic retries with exponential backoff for network errors, rate limits, and server-side failures. It also explicitly distinguishes client errors like 401 and 404, which are not retried by default, and recommends combining more retries with checkpointing for long-running work.",
  },
  {
    id: "langchain-deepagents-q24",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements are correct about the built-in middleware commonly included in deep agents?",
    options: [
      {
        text: "Todo list, filesystem, subagent, summarization, and patch-tool-call middleware are part of the documented default stack.",
        isCorrect: true,
      },
      {
        text: "Memory, skills, and human-in-the-loop middleware are included only when their related features are configured.",
        isCorrect: true,
      },
      {
        text: "Anthropic prompt caching middleware is the only built-in middleware component documented for deep agents.",
        isCorrect: false,
      },
      {
        text: "Middleware is irrelevant to prompting because it cannot affect the system prompt seen by the model.",
        isCorrect: false,
      },
    ],
    explanation:
      "One important idea in the docs is that middleware does not just intercept execution; it can also append tool guidance and other instructions into the final prompt. That is why saying middleware cannot affect prompting is incorrect, even though many middleware layers also handle runtime behavior.",
  },
  {
    id: "langchain-deepagents-q26",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly compare memory files and skills in deep agents?",
    options: [
      {
        text: "Memory commonly uses `AGENTS.md` files that are always loaded into the agent's prompt context.",
        isCorrect: true,
      },
      {
        text: "Skills use progressive disclosure, so the agent first inspects descriptions and loads full skill details only when relevant.",
        isCorrect: true,
      },
      {
        text: "Skills are mainly intended for small always-on instructions that should be loaded into startup context immediately.",
        isCorrect: false,
      },
      {
        text: "Memory and skills are equivalent mechanisms with the same loading behavior and the same file format.",
        isCorrect: false,
      },
    ],
    explanation:
      "The distinction is central: memory is always-on persistent context, while skills are selectively loaded capabilities. They also use different file conventions, with `AGENTS.md` for memory and `SKILL.md` plus optional supporting assets for skills.",
  },
  {
    id: "langchain-deepagents-q27",
    chapter: 1,
    difficulty: "hard",
    prompt:
      'Suppose an agent is created with `skills=["/skills/user/", "/skills/project/"]`, and both sources contain a skill named `web-search`. Which statements are correct?',
    options: [
      {
        text: "The project version of `web-search` takes precedence because later skill sources override earlier ones when names collide.",
        isCorrect: true,
      },
      {
        text: "The SDK automatically scans default command-line-interface skill locations even when you do not pass them in `skills`.",
        isCorrect: false,
      },
      {
        text: "Skill descriptions in frontmatter matter because the agent uses them to decide whether a skill matches the current task.",
        isCorrect: true,
      },
      {
        text: "Only the frontmatter is read at startup, and the rest of the skill content is never read later.",
        isCorrect: false,
      },
    ],
    explanation:
      "The documentation emphasizes explicit source lists and last-one-wins precedence when duplicate skill names exist. It also explains the progressive-disclosure mechanism: descriptions help the model match skills, and the full `SKILL.md` is read later when needed.",
  },
  {
    id: "langchain-deepagents-q28",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements about virtual filesystem tools are correct?",
    options: [
      {
        text: "`read_file` can support large files through offset and limit arguments rather than always returning the full file at once.",
        isCorrect: true,
      },
      {
        text: "`read_file` can read certain image formats only after those images are converted into plain text files.",
        isCorrect: false,
      },
      {
        text: "`glob` is intended for pattern-based file discovery such as `**/*.py`.",
        isCorrect: true,
      },
      {
        text: "`grep` can only return matching filenames and cannot return content with context or counts.",
        isCorrect: false,
      },
    ],
    explanation:
      "The filesystem tools are richer than a minimal shell-like interface. In particular, `read_file` can handle partial reads and supported image types, while `grep` supports multiple output modes rather than only filename-level matches.",
  },
  {
    id: "langchain-deepagents-q29",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe automatic offloading of large tool inputs and results during context management?",
    options: [
      {
        text: "Large file write or edit inputs can be truncated from old conversation history because the full content is already persisted in the filesystem.",
        isCorrect: true,
      },
      {
        text: "Large tool results can be replaced in active context with a file path reference plus a preview, allowing the agent to reread details later.",
        isCorrect: true,
      },
      {
        text: "The documented token threshold before offloading is always 20,000 tokens and cannot be changed.",
        isCorrect: false,
      },
      {
        text: "Offloading begins immediately after any tool call, regardless of context pressure or size.",
        isCorrect: false,
      },
    ],
    explanation:
      "The docs describe offloading as a selective compression technique, not something that happens after every tool call. Large inputs or results are candidates for eviction, especially when the active session approaches the context threshold, and the replacement includes pointers back to retrievable content.",
  },
  {
    id: "langchain-deepagents-q30",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about automatic summarization in deep agents are correct?",
    options: [
      {
        text: "When the context window is under pressure and offloading is insufficient, the agent can replace old history with a structured summary plus recent preserved messages.",
        isCorrect: true,
      },
      {
        text: "The original conversation can be preserved in the filesystem as a canonical record even after the in-context history is summarized.",
        isCorrect: true,
      },
      {
        text: "If a model call raises `ContextOverflowError`, deep agents must fail immediately rather than summarize and retry.",
        isCorrect: false,
      },
      {
        text: "Enabling the optional summarization tool disables the default automatic summarization trigger at high context usage.",
        isCorrect: false,
      },
    ],
    explanation:
      "The summarization tool is additive rather than a replacement for the default summarization pathway. The main goal is to preserve task continuity through a compact structured summary while retaining recoverability through the filesystem archive and recent-context preservation.",
  },
  {
    id: "langchain-deepagents-q31",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe structured output support in deep agents?",
    options: [
      {
        text: "You can pass a schema such as a Pydantic model through `response_format` when creating the deep agent.",
        isCorrect: true,
      },
      {
        text: "Validated structured data is returned in the agent state under the `structured_response` key.",
        isCorrect: true,
      },
      {
        text: "Structured output requires that the agent has no tools, because tool use and schemas are mutually exclusive.",
        isCorrect: false,
      },
      {
        text: "Subagents can also use structured output, and the structured object is always automatically returned to the parent agent without any extra handling.",
        isCorrect: false,
      },
    ],
    explanation:
      "Structured output is supported both for parent deep agents and for subagents. In the subagent case, the docs note that the structured object itself is not directly returned to the parent, so if you want the parent to use it, you should surface it appropriately in the returned tool message.",
  },
  {
    id: "langchain-deepagents-q32",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements are correct about `StateBackend`, `StoreBackend`, and `CompositeBackend`?",
    options: [
      {
        text: "`StateBackend` is ephemeral and is scoped to a single thread, though it can persist across turns in that thread via checkpoints.",
        isCorrect: true,
      },
      {
        text: "`StoreBackend` is intended for durable storage that can persist across threads when a LangGraph store is available.",
        isCorrect: true,
      },
      {
        text: "`CompositeBackend` can route different path prefixes only when all routed backends persist across threads.",
        isCorrect: false,
      },
      {
        text: "`CompositeBackend` hides the original routed path prefixes from listings and search results so the agent never sees them.",
        isCorrect: false,
      },
    ],
    explanation:
      "Composite routing is meant to feel like a unified filesystem from the agent's point of view, but the original paths remain meaningful in listings and search results. The larger idea is that StateBackend is thread-local, StoreBackend is durable, and CompositeBackend lets you combine them by path.",
  },
  {
    id: "langchain-deepagents-q33",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Consider the documented long-term memory setup where `/memories/` is routed to a `StoreBackend`. Which statements are correct?",
    options: [
      {
        text: "Files stored under `/memories/` can persist across conversation threads, unlike ordinary state-backed files.",
        isCorrect: true,
      },
      {
        text: "The agent refers only to the stripped storage key and never to the full `/memories/...` path in its own tool interactions.",
        isCorrect: false,
      },
      {
        text: "A `CompositeBackend` is used so that some paths stay ephemeral while `/memories/` is persisted.",
        isCorrect: true,
      },
      {
        text: "The only supported use case for long-term memory is storing user interface theme preferences.",
        isCorrect: false,
      },
    ],
    explanation:
      "Long-term memory is a general mechanism for cross-thread persistence, not a narrow setting store. The docs explicitly discuss use cases such as user preferences, self-improving instructions, research progress, and project knowledge, while the agent still uses the routed `/memories/...` path abstraction.",
  },
  {
    id: "langchain-deepagents-q34",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe `FilesystemBackend` and `LocalShellBackend` security guidance?",
    options: [
      {
        text: "`FilesystemBackend` can expose direct local file access and is considered inappropriate for typical web servers or multi-tenant production APIs.",
        isCorrect: true,
      },
      {
        text: "With `FilesystemBackend`, `virtual_mode=True` is recommended to enforce path-based access restrictions under the chosen root directory.",
        isCorrect: true,
      },
      {
        text: "`LocalShellBackend` adds shell execution but is safe for typical multi-tenant production APIs if `virtual_mode=True` is enabled.",
        isCorrect: false,
      },
      {
        text: "When shell access is enabled, `virtual_mode=True` fully secures the host system because shell commands cannot escape the configured root directory.",
        isCorrect: false,
      },
    ],
    explanation:
      "The docs are explicit that shell access changes the security picture dramatically. `virtual_mode=True` is useful for filesystem path handling, but once arbitrary shell execution is available on the host, that setting does not meaningfully secure the machine.",
  },
  {
    id: "langchain-deepagents-q35",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe sandbox backends in deep agents?",
    options: [
      {
        text: "Sandbox backends expose standard filesystem tools and also add an `execute` tool for shell commands.",
        isCorrect: true,
      },
      {
        text: "A key design pattern is 'sandbox as tool', where the agent logic stays outside the sandbox and invokes sandbox operations through backend tools.",
        isCorrect: true,
      },
      {
        text: "A major security recommendation is to avoid placing secrets inside the sandbox because a context-injected agent may exfiltrate them.",
        isCorrect: true,
      },
      {
        text: "Sandbox isolation does not completely solve context injection risks because the agent can still be manipulated into running harmful commands.",
        isCorrect: true,
      },
    ],
    explanation:
      "Sandboxing protects the host system, but it does not eliminate the possibility that a compromised prompt context causes the agent to misuse its sandboxed powers. That is why the docs strongly warn against placing secrets in the sandbox and emphasize that context injection and network exfiltration remain real concerns.",
  },
  {
    id: "langchain-deepagents-q36",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly distinguish agent filesystem tools from sandbox file transfer APIs?",
    options: [
      {
        text: "Filesystem tools like `read_file` and `write_file` are the tools the language model uses during its own reasoning and execution.",
        isCorrect: true,
      },
      {
        text: "`upload_files()` and `download_files()` are primarily for application code to move files across the host-sandbox boundary.",
        isCorrect: true,
      },
      {
        text: "File transfer APIs are distinct from the agent's filesystem tools rather than mere aliases.",
        isCorrect: true,
      },
      {
        text: "You might upload files to seed source code or configuration before the agent runs, then download generated artifacts after it finishes.",
        isCorrect: true,
      },
    ],
    explanation:
      "This distinction matters because the model does not directly call host-side transfer APIs. The LLM acts through agent tools inside the sandboxed environment, while your application uses provider APIs to preload inputs or retrieve outputs across the boundary.",
  },
  {
    id: "langchain-deepagents-q37",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about human-in-the-loop \\(HITL\\) execution are correct?",
    options: [
      {
        text: "A checkpointer is required so the agent can persist state between the interrupt and the later resume call.",
        isCorrect: true,
      },
      {
        text: "When resuming after an interrupt, you must use the same thread configuration, including the same `thread_id`.",
        isCorrect: true,
      },
      {
        text: "If multiple tool calls are interrupted together, you provide one decision per action request in the same order as the action requests.",
        isCorrect: true,
      },
      {
        text: "Resuming after an interrupt continues the same interrupted thread rather than starting a brand-new one.",
        isCorrect: true,
      },
    ],
    explanation:
      "The docs make continuity a core requirement of HITL: same thread, persisted state, ordered decisions. Requiring a new thread would defeat the whole point of pausing and resuming a durable execution trace, so that statement is incorrect.",
  },
  {
    id: "langchain-deepagents-q38",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe subagent-specific context and metadata?",
    options: [
      {
        text: "Parent runtime context is propagated automatically to subagents unless you intentionally encode specialization through namespaced keys.",
        isCorrect: true,
      },
      {
        text: "Namespaced context keys such as `researcher:max_depth` can be used to pass settings intended for only one subagent.",
        isCorrect: true,
      },
      {
        text: "The `lc_agent_name` metadata can help a shared tool determine which agent or subagent initiated a call.",
        isCorrect: true,
      },
      {
        text: "Subagents can access runtime context values; namespaced keys are one way to specialize what they receive.",
        isCorrect: true,
      },
    ],
    explanation:
      "Subagents inherit the parent's config and context by default, which is why namespacing is useful when you need specialization. The docs also show how `lc_agent_name` can guide branching logic inside shared tools when multiple agents use the same function.",
  },
  {
    id: "langchain-deepagents-q40",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the frontend patterns documented for deep agents?",
    options: [
      {
        text: "A frontend can use `useStream(...)` to access not only messages but also custom state values such as a live `todos` array.",
        isCorrect: true,
      },
      {
        text: "With `filterSubagentMessages: true`, coordinator messages stay cleaner because subagent tokens are rendered separately through subagent-specific interfaces.",
        isCorrect: true,
      },
      {
        text: "The documented subagent streaming pattern includes metadata such as subagent status, task description, timestamps, and final result.",
        isCorrect: true,
      },
      {
        text: "Deep agent frontends can use standard `useStream`-style patterns from LangGraph.",
        isCorrect: true,
      },
    ],
    explanation:
      "The frontend docs emphasize continuity with the broader LangGraph ecosystem: the same streaming foundations apply, but deep agents expose richer state such as todos and subagent streams. This makes it possible to build progress dashboards, subagent cards, and other UI patterns beyond ordinary chat bubbles.",
  },
  {
    id: "langchain-deepagents-q41",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the built-in general-purpose subagent in deep agents?",
    options: [
      {
        text: "A general-purpose subagent is automatically available even when you do not define any custom subagents.",
        isCorrect: true,
      },
      {
        text: "By default, it uses the same model and system prompt as the main agent.",
        isCorrect: true,
      },
      {
        text: "By default, it has access to the same tools as the main agent.",
        isCorrect: true,
      },
      {
        text: "It is not automatically removed when you add unrelated custom subagents.",
        isCorrect: true,
      },
    ],
    explanation:
      "The general-purpose subagent is meant to provide context isolation even without special domain-specific configuration. It is only replaced when you explicitly override it by providing a subagent named `general-purpose`; adding unrelated custom subagents does not remove it.",
  },
  {
    id: "langchain-deepagents-q42",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements are correct about overriding the general-purpose subagent?",
    options: [
      {
        text: 'If you add a subagent with `name="general-purpose"`, that definition replaces the default general-purpose subagent.',
        isCorrect: true,
      },
      {
        text: "This override can be used to give delegated general tasks a different model from the main agent.",
        isCorrect: true,
      },
      {
        text: "The override is a full replacement rather than a field-by-field merge with inherited defaults.",
        isCorrect: true,
      },
      {
        text: "An override can change the tools or system prompt used for delegated general-purpose work.",
        isCorrect: true,
      },
    ],
    explanation:
      "The documentation describes this as a full replacement, not a partial merge with implicit defaults from the old built-in version. That is why it is a powerful mechanism: you can swap models, prompts, and tools for delegated general tasks when the default behavior is not ideal.",
  },
  {
    id: "langchain-deepagents-q43",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe skill inheritance for parent agents and subagents?",
    options: [
      {
        text: "The general-purpose subagent inherits the main agent's skills when skills are configured on the parent.",
        isCorrect: true,
      },
      {
        text: "Custom subagents do not inherit the main agent's skills by default.",
        isCorrect: true,
      },
      {
        text: "When a custom subagent has its own skills, it runs its own isolated skills middleware state rather than sharing the parent's loaded skill state.",
        isCorrect: true,
      },
      {
        text: "If a child subagent loads a skill, that loaded skill does not automatically become visible to the parent agent.",
        isCorrect: true,
      },
    ],
    explanation:
      "Skill isolation is emphasized in the docs so that each agent can stay focused on its own capabilities without leaking hidden state into other agents. The only automatic inheritance path described is from the main agent to the general-purpose subagent, not to arbitrary custom subagents.",
  },
  {
    id: "langchain-deepagents-q45",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which characteristics make a good subagent description according to the documentation?",
    options: [
      {
        text: "It should be specific enough that the main agent can tell when this subagent is the right one to call.",
        isCorrect: true,
      },
      {
        text: "It should be action-oriented rather than vague.",
        isCorrect: true,
      },
      {
        text: "It should help distinguish the subagent from other available specialists.",
        isCorrect: true,
      },
      {
        text: "It should be intentionally broad and generic so the subagent gets selected for as many tasks as possible.",
        isCorrect: false,
      },
    ],
    explanation:
      "Descriptions are used by the main agent as routing hints, so vague descriptions make delegation worse rather than better. A clear, specific description improves selection quality by telling the parent what this subagent is really for and how it differs from alternatives.",
  },
  {
    id: "langchain-deepagents-q46",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe prompt composition in deep agents?",
    options: [
      {
        text: "If you provide a custom system prompt, it is prepended before the built-in base prompt and other injected sections.",
        isCorrect: true,
      },
      {
        text: "When memory is configured, a memory prompt with `AGENTS.md` context and usage guidance can be added.",
        isCorrect: true,
      },
      {
        text: "When skills are configured, the final prompt can include skill locations and frontmatter-derived information.",
        isCorrect: true,
      },
      {
        text: "The final prompt is limited to only the custom system prompt and the latest user message; middleware and built-in harness prompts never contribute to it.",
        isCorrect: false,
      },
    ],
    explanation:
      "One of the key design ideas in the docs is that the final prompt is layered from multiple components. It is not just the user message plus a short role description; it can include planning instructions, memory context, skill information, filesystem guidance, subagent instructions, and other middleware-generated prompt sections.",
  },
  {
    id: "langchain-deepagents-q47",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements are correct about the optional summarization tool middleware?",
    options: [
      {
        text: "It enables the agent to trigger summarization at strategically useful moments, such as between tasks.",
        isCorrect: true,
      },
      {
        text: "It is added through middleware rather than by replacing the base agent loop.",
        isCorrect: true,
      },
      {
        text: "It is meant to complement, not replace, the default automatic summarization that can trigger at high context usage.",
        isCorrect: true,
      },
      {
        text: "It can only be used with Anthropic models because summarization is tied to Anthropic prompt caching.",
        isCorrect: false,
      },
    ],
    explanation:
      "The optional summarization tool is presented as an extra capability the agent can invoke deliberately, not as a provider-locked feature. It also does not turn off the default summarization safety behavior, so both mechanisms can coexist in one agent.",
  },
  {
    id: "langchain-deepagents-q52",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly compare the 'agent in sandbox' pattern with the 'sandbox as tool' pattern?",
    options: [
      {
        text: "In the 'agent in sandbox' pattern, the full agent process runs inside the sandbox and typically needs some communication layer such as Hypertext Transfer Protocol or WebSocket to be reached externally.",
        isCorrect: true,
      },
      {
        text: "In the 'sandbox as tool' pattern, the agent itself runs outside the sandbox and calls sandbox operations through tools or backend methods.",
        isCorrect: true,
      },
      {
        text: "A benefit of the 'sandbox as tool' pattern is that application-side secrets can stay outside the sandbox.",
        isCorrect: true,
      },
      {
        text: "The documentation frames 'agent in sandbox' as strictly superior in every respect, with no security or operational drawbacks relative to 'sandbox as tool'.",
        isCorrect: false,
      },
    ],
    explanation:
      "The docs present these as trade-offing architectures, not as universally better or worse in all cases. Keeping the agent outside the sandbox can improve separation of concerns and keep secrets outside the sandbox, while running the agent inside the sandbox may mirror local development more closely but comes with its own operational and security complications.",
  },
  {
    id: "langchain-deepagents-q53",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe sandbox lifecycle guidance?",
    options: [
      {
        text: "Sandboxes consume resources until they are shut down, so cleanup matters for both cost and operational hygiene.",
        isCorrect: true,
      },
      {
        text: "In chat applications, a useful pattern is to associate one sandbox with one conversation thread identifier.",
        isCorrect: true,
      },
      {
        text: "Time-to-live settings can be useful for automatically cleaning up idle sandboxes when users may return later.",
        isCorrect: true,
      },
      {
        text: "Because sandbox backends are virtual, explicit shutdown is unnecessary; providers always stop them instantly once the current command completes.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lifecycle section makes it clear that sandboxes are real resources with cost and persistence implications. Thread-scoped sandboxes and TTL-based cleanup are practical patterns for keeping chat systems manageable when users go idle or abandon a conversation.",
  },
  {
    id: "langchain-deepagents-q54",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe `interrupt_on` configurations for human-in-the-loop?",
    options: [
      {
        text: "A value of `True` enables default human decisions such as approve, edit, and reject.",
        isCorrect: true,
      },
      {
        text: "A value of `False` means the tool should not interrupt for approval.",
        isCorrect: true,
      },
      {
        text: 'A configuration like `{ "allowed_decisions": ["approve", "reject"] }` can remove editing as an option for a given tool.',
        isCorrect: true,
      },
      {
        text: "If `interrupt_on` is configured for a tool, the agent can no longer call that tool at all, even after approval.",
        isCorrect: false,
      },
    ],
    explanation:
      "Human-in-the-loop is a pause-and-review mechanism, not a permanent ban on using tools. The configuration controls how humans can intervene on a tool call, including whether they can approve, edit, or reject the proposed action before execution continues.",
  },
  {
    id: "langchain-deepagents-q59",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the documented todo-list frontend pattern?",
    options: [
      {
        text: "The todo list is driven from custom agent state values rather than from parsing assistant prose.",
        isCorrect: true,
      },
      {
        text: "`stream.values?.todos` can expose a live todo array for rendering progress in the user interface.",
        isCorrect: true,
      },
      {
        text: "Todo items commonly move through statuses such as `pending`, `in_progress`, and `completed`.",
        isCorrect: true,
      },
      {
        text: "This pattern requires a separate polling API because `useStream` only supports chat messages and not custom state.",
        isCorrect: false,
      },
    ],
    explanation:
      "A key point of the todo-list pattern is that agent state can power interfaces beyond chat bubbles. Since `useStream` surfaces custom state values, the frontend can reactively render task progress without polling or trying to infer plan state from free-form text.",
  },
  {
    id: "langchain-deepagents-q60",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly compare the roles of skills, memory, and tools in deep agents?",
    options: [
      {
        text: "Tools usually expose lower-level executable actions such as filesystem operations, search, or sending requests.",
        isCorrect: true,
      },
      {
        text: "Skills are useful when there is a large amount of task-specific instructional context that should be loaded only when relevant.",
        isCorrect: true,
      },
      {
        text: "Memory files are appropriate for persistent preferences, conventions, and always-relevant context that should influence behavior across conversations.",
        isCorrect: true,
      },
      {
        text: "Because skills can include instructions, they make tools unnecessary whenever the agent has access to a filesystem.",
        isCorrect: false,
      },
    ],
    explanation:
      "These mechanisms complement each other rather than replacing one another. Tools perform actions, skills package reusable capabilities and large task-specific guidance with progressive disclosure, and memory provides always-on persistent context such as preferences or project conventions.",
  },
  {
    id: "langchain-deepagents-q61",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which situations are the best fit for using Deep Agents rather than a simpler agent setup?",
    options: [
      {
        text: "When the task is complex, multi-step, and benefits from explicit planning and decomposition.",
        isCorrect: true,
      },
      {
        text: "When the agent needs to manage a lot of context over time without stuffing everything into the chat history.",
        isCorrect: true,
      },
      {
        text: "When the main challenge is a very simple one-shot response with no real need for delegation or external memory.",
        isCorrect: false,
      },
      {
        text: "When you want built-in support for subagents, file-based context management, and memory across threads.",
        isCorrect: true,
      },
    ],
    explanation:
      "The docs position Deep Agents as an agent harness for longer-running, more complex tasks that need planning, context management, delegation, and sometimes persistent memory. For short, simple tasks, the extra machinery is often unnecessary.",
  },
  {
    id: "langchain-deepagents-q62",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Why is the filesystem concept important in Deep Agents even outside pure coding use cases?",
    options: [
      {
        text: "It gives the agent an external working memory so large artifacts and intermediate results do not all need to stay in the active prompt.",
        isCorrect: true,
      },
      {
        text: "It enables long-running work where the agent can store drafts, notes, and retrieved material and revisit them later.",
        isCorrect: true,
      },
      {
        text: "It matters only for source code editing and has little value for research, analysis, or document-heavy work.",
        isCorrect: false,
      },
      {
        text: "It supports a broader architecture where context can be managed deliberately instead of leaving everything in the message stream.",
        isCorrect: true,
      },
    ],
    explanation:
      "The docs repeatedly frame the filesystem as a context-management mechanism, not just a coding convenience. It helps with research, drafting, long tool results, memory, and long-horizon work more broadly.",
  },
  {
    id: "langchain-deepagents-q63",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "What is the main architectural reason to use subagents in Deep Agents?",
    options: [
      {
        text: "They isolate detailed work so the coordinator does not get bloated with every intermediate step and tool result.",
        isCorrect: true,
      },
      {
        text: "They let you create specialist workers with narrower instructions, tools, or even different models.",
        isCorrect: true,
      },
      {
        text: "They guarantee better results in every task, including trivial single-step requests.",
        isCorrect: false,
      },
      {
        text: "They allow the main agent to act more like a coordinator while delegated workers handle deeper subtasks.",
        isCorrect: true,
      },
    ],
    explanation:
      "Subagents are fundamentally about context quarantine and specialization. They are most useful when there is enough depth or complexity that isolating work improves clarity, focus, or token efficiency.",
  },
  {
    id: "langchain-deepagents-q64",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "When should a team avoid or at least be cautious about using subagents?",
    options: [
      {
        text: "When the task is simple enough that delegation overhead outweighs the benefit.",
        isCorrect: true,
      },
      {
        text: "When the workflow depends heavily on maintaining shared intermediate context step by step.",
        isCorrect: true,
      },
      {
        text: "When multiple subagents might independently make tightly coupled decisions that really require central synthesis.",
        isCorrect: true,
      },
      {
        text: "Whenever the team wants specialization, because specialization is fundamentally incompatible with a coordinator architecture.",
        isCorrect: false,
      },
    ],
    explanation:
      "The docs and examples make clear that subagents are powerful but not free. They are most risky when the work is tightly coupled and independent local decisions can create inconsistencies that the overall system then has to reconcile.",
  },
  {
    id: "langchain-deepagents-q65",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements best capture the role of planning in Deep Agents?",
    options: [
      {
        text: "Planning helps the agent stay oriented during long, multi-step work.",
        isCorrect: true,
      },
      {
        text: "A todo list can act as both an internal steering tool and a visible progress surface for users.",
        isCorrect: true,
      },
      {
        text: "Planning removes the need for a good system prompt, because the plan alone is enough to govern the agent.",
        isCorrect: false,
      },
      {
        text: "Plans can also serve as checkpoints where a human reviews or guides the direction of the work.",
        isCorrect: true,
      },
    ],
    explanation:
      "The docs present planning as one of the core harness capabilities. It helps with organization, progress tracking, user alignment, and longer-horizon execution, but it complements prompting rather than replacing it.",
  },
  {
    id: "langchain-deepagents-q66",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which prompting strategy is most aligned with the Deep Agents documentation?",
    options: [
      {
        text: "Use a detailed prompt that explains how the agent should plan, manage files, delegate, and generally behave for the use case.",
        isCorrect: true,
      },
      {
        text: "Assume the model will infer good long-horizon behavior on its own, so prompting can usually stay extremely minimal.",
        isCorrect: false,
      },
      {
        text: "Treat prompting as part of the architecture, since built-in and middleware-added instructions shape how the harness is used.",
        isCorrect: true,
      },
      {
        text: "Use the custom prompt to add domain-specific behavior on top of the harness's built-in guidance.",
        isCorrect: true,
      },
    ],
    explanation:
      "The documentation is explicit that Deep Agents already have built-in prompt layers, and that your custom system prompt should add use-case-specific guidance. Prompting remains a major control mechanism, not an afterthought.",
  },
  {
    id: "langchain-deepagents-q67",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "What is the most useful way to think about skills versus memory in Deep Agents?",
    options: [
      {
        text: "Skills are on-demand capabilities or workflows that are loaded when relevant, while memory is always-on context.",
        isCorrect: true,
      },
      {
        text: "Memory is a good fit for persistent preferences, conventions, or recurring guidance the agent should always consider.",
        isCorrect: true,
      },
      {
        text: "Skills and memory are basically interchangeable, so it usually does not matter which one you choose.",
        isCorrect: false,
      },
      {
        text: "Skills are especially useful when instructions would otherwise bloat the startup prompt but are only needed for certain tasks.",
        isCorrect: true,
      },
    ],
    explanation:
      "The docs draw a strong distinction: memory is always injected, while skills use progressive disclosure. That makes skills better for large, task-specific capability bundles and memory better for persistent always-relevant context.",
  },
  {
    id: "langchain-deepagents-q68",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A team wants an agent to remember user preferences across conversations but keep ordinary scratch work ephemeral. Which design is most appropriate?",
    options: [
      {
        text: "Use a CompositeBackend so most files stay thread-local while a path like `/memories/` is routed to persistent storage.",
        isCorrect: true,
      },
      {
        text: "Use only the default StateBackend, because that is already designed for cross-thread persistence.",
        isCorrect: false,
      },
      {
        text: "Store durable preferences in a long-term memory path while keeping temporary drafts and working notes outside that path.",
        isCorrect: true,
      },
      {
        text: "Treat all files as equally permanent so the agent never has to distinguish between transient and durable context.",
        isCorrect: false,
      },
    ],
    explanation:
      "This is exactly the hybrid pattern described in the long-term memory docs. Persistent knowledge belongs in routed durable storage, while scratch work should usually remain transient to avoid clutter and uncontrolled accumulation.",
  },
  {
    id: "langchain-deepagents-q69",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "What problem are offloading and summarization trying to solve in Deep Agents?",
    options: [
      {
        text: "They keep the agent effective over long trajectories by preventing the active context from filling with too much old or bulky material.",
        isCorrect: true,
      },
      {
        text: "They allow the system to preserve recoverability while compressing what stays in working memory.",
        isCorrect: true,
      },
      {
        text: "They are mainly cosmetic UI features and do not matter much for agent reliability.",
        isCorrect: false,
      },
      {
        text: "They make it possible for the agent to keep working even when tool results or histories become too large to keep fully in prompt.",
        isCorrect: true,
      },
    ],
    explanation:
      "The docs present offloading and summarization as core context-engineering mechanisms. Their purpose is to preserve long-horizon functionality and recoverability without overwhelming the model's active context window.",
  },
  {
    id: "langchain-deepagents-q70",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "What is the best interpretation of the optional summarization tool?",
    options: [
      {
        text: "It gives the agent strategic control to summarize at useful moments, rather than only waiting for automatic pressure-based summarization.",
        isCorrect: true,
      },
      {
        text: "It complements the default summarization behavior instead of replacing it.",
        isCorrect: true,
      },
      {
        text: "It is mainly useful in longer workflows where the agent may benefit from intentionally compacting context between phases of work.",
        isCorrect: true,
      },
      {
        text: "It is a provider-specific optimization that only matters for Anthropic prompt caching.",
        isCorrect: false,
      },
    ],
    explanation:
      "The key idea is strategic timing. Automatic summarization still exists, but the optional tool lets the agent decide to compress context at sensible boundaries such as after completing one phase and before starting another.",
  },
  {
    id: "langchain-deepagents-q71",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "How should a senior team think about filesystem and shell access on the host machine?",
    options: [
      {
        text: "Direct host access can be acceptable in controlled development environments, but it is not the right default for production or multi-tenant systems.",
        isCorrect: true,
      },
      {
        text: "Giving an agent local shell access on the host creates a materially different risk profile from giving it only an isolated sandbox.",
        isCorrect: true,
      },
      {
        text: "Using `virtual_mode=True` is enough to make unrestricted host shell execution safe for production APIs.",
        isCorrect: false,
      },
      {
        text: "If the agent may process untrusted input or run autonomously, a sandbox-based architecture is usually the safer posture.",
        isCorrect: true,
      },
    ],
    explanation:
      "The docs are very direct here: local filesystem and especially local shell are powerful but risky. Sandboxes are the preferred architecture when safety and isolation matter more than raw convenience.",
  },
  {
    id: "langchain-deepagents-q72",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "What is the most important security lesson from the sandbox documentation?",
    options: [
      {
        text: "Sandboxes protect the host system, but they do not eliminate the risk that a context-injected agent misuses its own execution environment.",
        isCorrect: true,
      },
      {
        text: "Secrets should generally stay outside the sandbox, with authenticated actions handled by host-side tools when possible.",
        isCorrect: true,
      },
      {
        text: "Once code runs in a sandbox, there is no meaningful risk of exfiltration or misuse.",
        isCorrect: false,
      },
      {
        text: "Blocking or restricting network access can be an important part of reducing sandbox risk.",
        isCorrect: true,
      },
    ],
    explanation:
      "The docs emphasize that sandboxing is not a magical security cure. It helps isolate the host, but context injection and outbound exfiltration are still real concerns, especially if secrets are placed inside the sandbox.",
  },
  {
    id: "langchain-deepagents-q73",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "When comparing the 'agent in sandbox' and 'sandbox as tool' patterns, which statements are most accurate?",
    options: [
      {
        text: "The 'sandbox as tool' pattern keeps the agent logic outside the sandbox and often makes iteration and secret handling cleaner.",
        isCorrect: true,
      },
      {
        text: "The 'agent in sandbox' pattern may mirror local development more closely, but it can require more infrastructure and place secrets in a riskier position.",
        isCorrect: true,
      },
      {
        text: "The documentation presents one pattern as universally best for all cases.",
        isCorrect: false,
      },
      {
        text: "Choosing between the two patterns is an architectural trade-off involving security, update speed, separation of concerns, and operational complexity.",
        isCorrect: true,
      },
    ],
    explanation:
      "The docs treat these as trade-offs, not absolutes. The right choice depends on how much you value production mirroring, faster iteration, separation of concerns, and where you want secrets and state to live.",
  },
  {
    id: "langchain-deepagents-q74",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "What is the main product or UX value of human-in-the-loop in Deep Agents?",
    options: [
      {
        text: "It gives teams a configurable safety gate before sensitive or expensive actions are executed.",
        isCorrect: true,
      },
      {
        text: "It can support debugging, trust-building, and controlled approval workflows rather than fully blind autonomy.",
        isCorrect: true,
      },
      {
        text: "It is useful only for file deletion and not for things like emails, writes, or other consequential actions.",
        isCorrect: false,
      },
      {
        text: "Different tools can reasonably have different approval rules depending on their risk profile.",
        isCorrect: true,
      },
    ],
    explanation:
      "The docs frame HITL as a configurable governance mechanism. It is not just about one dangerous tool, but about tailoring approval and editability according to operational risk and trust requirements.",
  },
  {
    id: "langchain-deepagents-q75",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "What should a senior user infer from the frontend todo-list and subagent-streaming patterns?",
    options: [
      {
        text: "Deep Agents are designed not just for chat output, but for richer operational UIs such as progress dashboards and specialist-worker cards.",
        isCorrect: true,
      },
      {
        text: "A todo list can expose execution progress in a more useful way than plain chat when the agent is following a structured plan.",
        isCorrect: true,
      },
      {
        text: "Subagent streaming makes it possible to separate coordinator output from worker output, which improves observability and readability.",
        isCorrect: true,
      },
      {
        text: "These UI patterns are mostly cosmetic and do not reflect anything important about the underlying agent architecture.",
        isCorrect: false,
      },
    ],
    explanation:
      "The frontend patterns reveal an important architectural point: Deep Agents are meant to expose structured progress and worker activity, not just final text. That matters for observability, trust, and product design.",
  },
];
