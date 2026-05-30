import { Question } from "../../quiz";

export const aieChapter4Questions: Question[] = [
  // ============================================================
  //  Q1–Q18: 4 correct answers (ALL TRUE)
  // ============================================================

  {
    id: "aie-ch4-q01",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements describe core properties expected from an AI agent?",
    options: [
      {
        text: "It should operate with autonomy rather than requiring step-by-step instructions.",
        isCorrect: true,
      },
      {
        text: "It should react to environmental changes in a timely and appropriate way.",
        isCorrect: true,
      },
      {
        text: "It should pursue goals proactively rather than only answering queries.",
        isCorrect: true,
      },
      {
        text: "It should interact socially with humans or other agents when needed.",
        isCorrect: true,
      },
    ],
    explanation:
      'Autonomy, reactivity, pro-activeness, and social ability are widely accepted as the four fundamental properties of AI agents. To reason through the choices, select every statement because each one matches the criterion in the prompt: "It should operate with autonomy rather than requiring step-by-step instructions."; "It should react to environmental changes in a timely and appropriate way."; "It should pursue goals proactively rather than only answering queries."; "It should interact socially with humans or other agents when needed.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "aie-ch4-q02",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements describe why an LLM can act as the 'brain' of an agent?",
    options: [
      {
        text: "It maintains an internal representation of knowledge learned during training.",
        isCorrect: true,
      },
      {
        text: "It supports multi-turn conversations and integrates information across turns.",
        isCorrect: true,
      },
      {
        text: "It can perform reasoning steps when prompted appropriately.",
        isCorrect: true,
      },
      {
        text: "It can interpret natural language instructions and generate plans.",
        isCorrect: true,
      },
    ],
    explanation:
      "LLMs serve as the reasoning and decision-making centerpiece due to learned representations, multi-turn dialogue capacity, and instruction-following behavior.",
  },

  {
    id: "aie-ch4-q03",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements describe limitations of LLM knowledge that affect agent design?",
    options: [
      {
        text: "Its knowledge is frozen at pre-training time.",
        isCorrect: true,
      },
      {
        text: "It cannot natively store new long-term memories across sessions.",
        isCorrect: true,
      },
      {
        text: "It does not inherently perceive non-textual input without additional modules.",
        isCorrect: true,
      },
      {
        text: "It has no built-in access to real-time information without tools.",
        isCorrect: true,
      },
    ],
    explanation:
      'LLMs lack persistent memory, real-time knowledge, and multimodal perception unless augmented with external components. To reason through the choices, select every statement because each one matches the criterion in the prompt: "Its knowledge is frozen at pre-training time."; "It cannot natively store new long-term memories across sessions."; "It does not inherently perceive non-textual input without additional modules."; "It has no built-in access to real-time information without tools.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "aie-ch4-q04",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly characterize perception modules for agents?",
    options: [
      {
        text: "They allow an agent to process sensory inputs such as images, audio, or video.",
        isCorrect: true,
      },
      {
        text: "They often involve converting non-textual data into embeddings usable by the LLM.",
        isCorrect: true,
      },
      {
        text: "They may rely on specialized models such as Whisper, BLIP-2, or PaLM-E.",
        isCorrect: true,
      },
      {
        text: "They extend the LLM’s ability to interpret environmental states.",
        isCorrect: true,
      },
    ],
    explanation:
      'Perception systems translate sensory signals into representations that LLMs can analyze. To reason through the choices, select every statement because each one matches the criterion in the prompt: "They allow an agent to process sensory inputs such as images, audio, or video."; "They often involve converting non-textual data into embeddings usable by the LLM."; "They may rely on specialized models such as Whisper, BLIP-2, or PaLM-E."; "They extend the LLM’s ability to interpret environmental states.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "aie-ch4-q05",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements describe the purpose of action modules in an agent system?",
    options: [
      {
        text: "They enable an agent to execute tasks beyond text generation.",
        isCorrect: true,
      },
      {
        text: "They provide tool use capability such as search, code execution, or API calls.",
        isCorrect: true,
      },
      {
        text: "They allow the agent to modify its environment virtually or physically.",
        isCorrect: true,
      },
      {
        text: "They extend the LLM’s functionality similarly to how tools extend human capability.",
        isCorrect: true,
      },
    ],
    explanation:
      'Tools/actions enable an agent to go beyond text output and interact with the environment. To reason through the choices, select every statement because each one matches the criterion in the prompt: "They enable an agent to execute tasks beyond text generation."; "They provide tool use capability such as search, code execution, or API calls."; "They allow the agent to modify its environment virtually or physically."; "They extend the LLM’s functionality similarly to how tools extend human capability.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "aie-ch4-q06",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements describe challenges LLM agents face when interacting with the web?",
    options: [
      {
        text: "Long pages can overwhelm the model, producing hallucinations or planning failures.",
        isCorrect: true,
      },
      {
        text: "Irrelevant text may bury important signals in noisy HTML.",
        isCorrect: true,
      },
      {
        text: "Sites may change layouts, requiring robust generalization.",
        isCorrect: true,
      },
      {
        text: "The agent must decide whether more browsing actions are needed.",
        isCorrect: true,
      },
    ],
    explanation:
      'Web interaction is messy and requires decision-making about context, relevance, and further exploration. To reason through the choices, select every statement because each one matches the criterion in the prompt: "Long pages can overwhelm the model, producing hallucinations or planning failures."; "Irrelevant text may bury important signals in noisy HTML."; "Sites may change layouts, requiring robust generalization."; "The agent must decide whether more browsing actions are needed.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "aie-ch4-q07",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements describe planning in LLM-based agents?",
    options: [
      {
        text: "It involves decomposition of tasks into manageable subtasks.",
        isCorrect: true,
      },
      {
        text: "It may involve interleaving reasoning and action execution.",
        isCorrect: true,
      },
      {
        text: "It may incorporate feedback loops for refinement.",
        isCorrect: true,
      },
      {
        text: "It can rely on multiple candidate plans for difficult tasks.",
        isCorrect: true,
      },
    ],
    explanation:
      'LLM planning strategies often mix decomposition, iterative refinement, and multi-path reasoning. To reason through the choices, select every statement because each one matches the criterion in the prompt: "It involves decomposition of tasks into manageable subtasks."; "It may involve interleaving reasoning and action execution."; "It may incorporate feedback loops for refinement."; "It can rely on multiple candidate plans for difficult tasks.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "aie-ch4-q08",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements describe memory-augmented planning?",
    options: [
      {
        text: "It compensates for limited context windows by retrieving past experience.",
        isCorrect: true,
      },
      {
        text: "It may involve RAG systems to fetch previously stored knowledge relevant to the task.",
        isCorrect: true,
      },
      {
        text: "It allows reuse of prior plans or solutions for similar tasks.",
        isCorrect: true,
      },
      {
        text: "It can reduce redundant reasoning by allowing the agent to recall past computations.",
        isCorrect: true,
      },
    ],
    explanation:
      'Memory-augmented systems use external stores to enable better planning, reuse, and continuity across tasks. To reason through the choices, select every statement because each one matches the criterion in the prompt: "It compensates for limited context windows by retrieving past experience."; "It may involve RAG systems to fetch previously stored knowledge relevant to the task."; "It allows reuse of prior plans or solutions for similar tasks."; "It can reduce redundant reasoning by allowing the agent to recall past computations.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "aie-ch4-q09",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements describe challenges specific to embodied agents?",
    options: [
      {
        text: "They must interpret physical sensor signals like vision, audio, and spatial cues.",
        isCorrect: true,
      },
      {
        text: "They must perform feasible physical actions that obey real-world constraints.",
        isCorrect: true,
      },
      {
        text: "They require common-sense priors to avoid unrealistic behaviors.",
        isCorrect: true,
      },
      {
        text: "They must adapt to continuous and unpredictable environments.",
        isCorrect: true,
      },
    ],
    explanation:
      'Embodiment requires perception, physical feasibility, and robust interpretation of real-world dynamics. To reason through the choices, select every statement because each one matches the criterion in the prompt: "They must interpret physical sensor signals like vision, audio, and spatial cues."; "They must perform feasible physical actions that obey real-world constraints."; "They require common-sense priors to avoid unrealistic behaviors."; "They must adapt to continuous and unpredictable environments.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "aie-ch4-q10",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe multi-agent interaction properties?",
    options: [
      {
        text: "Agents can cooperate to solve tasks a single agent cannot solve.",
        isCorrect: true,
      },
      {
        text: "Agents can critique, evaluate, or refine each other’s work.",
        isCorrect: true,
      },
      {
        text: "Agents can adopt competitive or adversarial roles to improve performance.",
        isCorrect: true,
      },
      {
        text: "Agents can communicate using natural language to coordinate plans.",
        isCorrect: true,
      },
    ],
    explanation:
      'Multi-agent systems allow cooperation, evaluation, competition, and NL-based coordination. To reason through the choices, select every statement because each one matches the criterion in the prompt: "Agents can cooperate to solve tasks a single agent cannot solve."; "Agents can critique, evaluate, or refine each other’s work."; "Agents can adopt competitive or adversarial roles to improve performance."; "Agents can communicate using natural language to coordinate plans.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "aie-ch4-q11",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements describe characteristics of human–agent interaction?",
    options: [
      {
        text: "Humans may provide instructions and feedback in natural language.",
        isCorrect: true,
      },
      {
        text: "Agents can produce intermediate reasoning visible to the user.",
        isCorrect: true,
      },
      {
        text: "Interaction can be single-turn or multi-turn.",
        isCorrect: true,
      },
      {
        text: "Agents may need to model user goals and preferences.",
        isCorrect: true,
      },
    ],
    explanation:
      'Human–agent systems rely on natural-language instructions, feedback, multi-turn dialogue, and intent modeling. To reason through the choices, select every statement because each one matches the criterion in the prompt: "Humans may provide instructions and feedback in natural language."; "Agents can produce intermediate reasoning visible to the user."; "Interaction can be single-turn or multi-turn."; "Agents may need to model user goals and preferences.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "aie-ch4-q12",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements describe the role of ReAct (Reason & Act) prompting in agent systems?",
    options: [
      {
        text: "It encourages explicit reasoning before tool invocation.",
        isCorrect: true,
      },
      {
        text: "It helps the model choose appropriate tools for each step.",
        isCorrect: true,
      },
      {
        text: "It allows the model to track intermediate reasoning traces.",
        isCorrect: true,
      },
      {
        text: "It supports iterative cycles of thinking and acting.",
        isCorrect: true,
      },
    ],
    explanation:
      'ReAct prompting structures reasoning and action steps for tool-using agents. To reason through the choices, select every statement because each one matches the criterion in the prompt: "It encourages explicit reasoning before tool invocation."; "It helps the model choose appropriate tools for each step."; "It allows the model to track intermediate reasoning traces."; "It supports iterative cycles of thinking and acting.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "aie-ch4-q13",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements describe interleaved task decomposition for an agent?",
    options: [
      {
        text: "It alternates between breaking down tasks and executing subtasks.",
        isCorrect: true,
      },
      {
        text: "It adapts dynamically based on intermediate results.",
        isCorrect: true,
      },
      {
        text: "It reduces error propagation by incorporating feedback early.",
        isCorrect: true,
      },
      {
        text: "It may generate longer reasoning chains but more adaptable behavior.",
        isCorrect: true,
      },
    ],
    explanation:
      'Interleaving planning and execution creates adaptive task-solving loops that incorporate feedback. To reason through the choices, select every statement because each one matches the criterion in the prompt: "It alternates between breaking down tasks and executing subtasks."; "It adapts dynamically based on intermediate results."; "It reduces error propagation by incorporating feedback early."; "It may generate longer reasoning chains but more adaptable behavior.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "aie-ch4-q14",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements describe multi-plan selection approaches?",
    options: [
      {
        text: "Several candidate plans may be generated for the same task.",
        isCorrect: true,
      },
      {
        text: "A voting or selection algorithm chooses the best plan.",
        isCorrect: true,
      },
      {
        text: "It allows exploration of multiple reasoning paths when the task is ambiguous.",
        isCorrect: true,
      },
      {
        text: "It trades more computation for higher reliability on difficult problems.",
        isCorrect: true,
      },
    ],
    explanation:
      'Multi-plan selection expands search over reasoning paths, improving robustness at computational cost. To reason through the choices, select every statement because each one matches the criterion in the prompt: "Several candidate plans may be generated for the same task."; "A voting or selection algorithm chooses the best plan."; "It allows exploration of multiple reasoning paths when the task is ambiguous."; "It trades more computation for higher reliability on difficult problems.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "aie-ch4-q15",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements describe LLM multimodality integration?",
    options: [
      {
        text: "Image embeddings can be inserted directly into the transformer input stream.",
        isCorrect: true,
      },
      {
        text: "Audio can be converted into text or spectrogram embeddings.",
        isCorrect: true,
      },
      {
        text: "Video must preserve temporal ordering when encoded.",
        isCorrect: true,
      },
      {
        text: "Multimodal modules often require alignment training or bridging models.",
        isCorrect: true,
      },
    ],
    explanation:
      'Multimodality introduces specialized encoders, alignment layers, and temporal structure preservation. To reason through the choices, select every statement because each one matches the criterion in the prompt: "Image embeddings can be inserted directly into the transformer input stream."; "Audio can be converted into text or spectrogram embeddings."; "Video must preserve temporal ordering when encoded."; "Multimodal modules often require alignment training or bridging models.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "aie-ch4-q16",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements describe innovation-oriented agent deployments?",
    options: [
      {
        text: "They require deep domain knowledge and the ability to extrapolate.",
        isCorrect: true,
      },
      {
        text: "They may involve scientific discovery or software design.",
        isCorrect: true,
      },
      {
        text: "They demand nontrivial reasoning skills beyond simple task execution.",
        isCorrect: true,
      },
      {
        text: "They involve multi-step, exploratory tasks that cannot be reduced to fixed templates.",
        isCorrect: true,
      },
    ],
    explanation:
      'Innovation-oriented agents operate in exploratory domains with open-ended goals and high complexity. To reason through the choices, select every statement because each one matches the criterion in the prompt: "They require deep domain knowledge and the ability to extrapolate."; "They may involve scientific discovery or software design."; "They demand nontrivial reasoning skills beyond simple task execution."; "They involve multi-step, exploratory tasks that cannot be reduced to fixed templates.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "aie-ch4-q17",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements describe libraries used to build agents?",
    options: [
      {
        text: "They provide abstractions to connect LLMs with tools, memory, and perception modules.",
        isCorrect: true,
      },
      {
        text: "They standardize workflows for agent planning and execution.",
        isCorrect: true,
      },
      {
        text: "They often include built-in integrations for search, retrieval, or APIs.",
        isCorrect: true,
      },
      {
        text: "They support monitoring, evaluation, or production deployment.",
        isCorrect: true,
      },
    ],
    explanation:
      'Agent libraries help orchestrate LLMs, tools, retrieval systems, and evaluation workflows. To reason through the choices, select every statement because each one matches the criterion in the prompt: "They provide abstractions to connect LLMs with tools, memory, and perception modules."; "They standardize workflows for agent planning and execution."; "They often include built-in integrations for search, retrieval, or APIs."; "They support monitoring, evaluation, or production deployment.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "aie-ch4-q18",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements describe what makes AI-powered web search different from classic search engines?",
    options: [
      {
        text: "The model can understand natural-language queries.",
        isCorrect: true,
      },
      {
        text: "It can incorporate user preferences and history for ranking.",
        isCorrect: true,
      },
      {
        text: "It can summarize or extract relevant passages.",
        isCorrect: true,
      },
      {
        text: "It can chain actions to refine the search iteratively.",
        isCorrect: true,
      },
    ],
    explanation:
      'AI-enhanced search involves NL understanding, personalization, reasoning, and actionable workflows. To reason through the choices, select every statement because each one matches the criterion in the prompt: "The model can understand natural-language queries."; "It can incorporate user preferences and history for ranking."; "It can summarize or extract relevant passages."; "It can chain actions to refine the search iteratively.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  // ============================================================
  //  Q19–Q36: 3 correct answers, 1 incorrect
  // ============================================================

  {
    id: "aie-ch4-q19",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements describe common-sense knowledge in an LLM?",
    options: [
      {
        text: "It captures everyday facts humans consider obvious.",
        isCorrect: true,
      },
      {
        text: "It helps prevent misinterpretation of simple instructions.",
        isCorrect: true,
      },
      {
        text: "It is often implicit and emerges from training corpora.",
        isCorrect: true,
      },
      {
        text: "It guarantees flawless reasoning in all situations.",
        isCorrect: false,
      },
    ],
    explanation:
      'Common-sense priors help but do not guarantee perfect reasoning. To reason through the choices, select the statements that match the criterion in the prompt: "It captures everyday facts humans consider obvious."; "It helps prevent misinterpretation of simple instructions."; "It is often implicit and emerges from training corpora.". Do not select statements that miss that criterion: "It guarantees flawless reasoning in all situations.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q20",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements describe drawbacks of global, all-at-once task planning?",
    options: [
      {
        text: "The initial plan may contain errors the model cannot correct.",
        isCorrect: true,
      },
      {
        text: "The model may hallucinate unrealistic or infeasible steps.",
        isCorrect: true,
      },
      {
        text: "The plan cannot adapt to unexpected intermediate results.",
        isCorrect: true,
      },
      {
        text: "It ensures maximum efficiency in all scenarios.",
        isCorrect: false,
      },
    ],
    explanation:
      'Static plans lack adaptability and may propagate mistakes. To reason through the choices, select the statements that match the criterion in the prompt: "The initial plan may contain errors the model cannot correct."; "The model may hallucinate unrealistic or infeasible steps."; "The plan cannot adapt to unexpected intermediate results.". Do not select statements that miss that criterion: "It ensures maximum efficiency in all scenarios.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q21",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements describe characteristics of digital (non-embodied) agents?",
    options: [
      {
        text: "They operate entirely within virtual environments.",
        isCorrect: true,
      },
      {
        text: "Their actions may include browsing, coding, or text-based navigation.",
        isCorrect: true,
      },
      {
        text: "Their environment dynamics are typically symbolic or textual.",
        isCorrect: true,
      },
      { text: "They inherently require robotics hardware.", isCorrect: false },
    ],
    explanation:
      'Digital agents act in virtual spaces without physical embodiment. To reason through the choices, select the statements that match the criterion in the prompt: "They operate entirely within virtual environments."; "Their actions may include browsing, coding, or text-based navigation."; "Their environment dynamics are typically symbolic or textual.". Do not select statements that miss that criterion: "They inherently require robotics hardware.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q22",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements describe natural-language interfaces in agent systems?",
    options: [
      {
        text: "They allow users to express goals with minimal structure.",
        isCorrect: true,
      },
      {
        text: "They increase interpretability of agent decisions.",
        isCorrect: true,
      },
      {
        text: "They reduce the need for domain-specific query languages.",
        isCorrect: true,
      },
      {
        text: "They eliminate all ambiguity in user intent.",
        isCorrect: false,
      },
    ],
    explanation:
      'NL interfaces help usability but still involve ambiguity. To reason through the choices, select the statements that match the criterion in the prompt: "They allow users to express goals with minimal structure."; "They increase interpretability of agent decisions."; "They reduce the need for domain-specific query languages.". Do not select statements that miss that criterion: "They eliminate all ambiguity in user intent.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q23",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements describe limitations of refinement-based planning approaches?",
    options: [
      {
        text: "The model may repeatedly revise its plan without converging.",
        isCorrect: true,
      },
      {
        text: "Refinement loops may become computationally expensive.",
        isCorrect: true,
      },
      { text: "Refinement cannot guarantee an optimal plan.", isCorrect: true },
      {
        text: "Refinement ensures deterministic convergence.",
        isCorrect: false,
      },
    ],
    explanation:
      'Refinement increases adaptability but cannot guarantee convergence or optimality. To reason through the choices, select the statements that match the criterion in the prompt: "The model may repeatedly revise its plan without converging."; "Refinement loops may become computationally expensive."; "Refinement cannot guarantee an optimal plan.". Do not select statements that miss that criterion: "Refinement ensures deterministic convergence.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q24",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements describe advantages of multimodal models in agent perception?",
    options: [
      {
        text: "They can integrate information across several sensory channels.",
        isCorrect: true,
      },
      {
        text: "They enable agents to interpret images, audio, or video.",
        isCorrect: true,
      },
      {
        text: "They reduce hallucination by grounding perception.",
        isCorrect: true,
      },
      {
        text: "They remove the need for textual context entirely.",
        isCorrect: false,
      },
    ],
    explanation:
      'Multimodality adds grounding but does not eliminate the role of textual context. To reason through the choices, select the statements that match the criterion in the prompt: "They can integrate information across several sensory channels."; "They enable agents to interpret images, audio, or video."; "They reduce hallucination by grounding perception.". Do not select statements that miss that criterion: "They remove the need for textual context entirely.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q25",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements describe the role of tool descriptions in an agent system?",
    options: [
      {
        text: "They inform the LLM about the tool’s intended purpose.",
        isCorrect: true,
      },
      {
        text: "They help the model select the correct tool during planning.",
        isCorrect: true,
      },
      {
        text: "They reduce ambiguity about what each tool can do.",
        isCorrect: true,
      },
      {
        text: "They guarantee perfect tool usage by the model.",
        isCorrect: false,
      },
    ],
    explanation:
      'Tool descriptions improve tool selection but cannot guarantee flawless execution. To reason through the choices, select the statements that match the criterion in the prompt: "They inform the LLM about the tool’s intended purpose."; "They help the model select the correct tool during planning."; "They reduce ambiguity about what each tool can do.". Do not select statements that miss that criterion: "They guarantee perfect tool usage by the model.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q26",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements describe the challenges of ranking search results using LLMs?",
    options: [
      {
        text: "They must evaluate semantic relevance, not just keyword overlap.",
        isCorrect: true,
      },
      {
        text: "They may need to incorporate user preferences.",
        isCorrect: true,
      },
      {
        text: "They must distinguish ambiguous query interpretations.",
        isCorrect: true,
      },
      {
        text: "They inherently know which documents are authoritative.",
        isCorrect: false,
      },
    ],
    explanation:
      'Authority and trustworthiness must be inferred, not natively encoded. To reason through the choices, select the statements that match the criterion in the prompt: "They must evaluate semantic relevance, not just keyword overlap."; "They may need to incorporate user preferences."; "They must distinguish ambiguous query interpretations.". Do not select statements that miss that criterion: "They inherently know which documents are authoritative.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q27",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements describe attributes of adversarial multi-agent systems?",
    options: [
      {
        text: "Agents may critique or challenge each other's reasoning.",
        isCorrect: true,
      },
      {
        text: "Competition can improve robustness and performance.",
        isCorrect: true,
      },
      {
        text: "Adversarial dialogue can expose weaknesses in solutions.",
        isCorrect: true,
      },
      {
        text: "Agents cannot cooperate under any circumstances.",
        isCorrect: false,
      },
    ],
    explanation:
      'Adversarial setups coexist with cooperation depending on system design. To reason through the choices, select the statements that match the criterion in the prompt: "Agents may critique or challenge each other\'s reasoning."; "Competition can improve robustness and performance."; "Adversarial dialogue can expose weaknesses in solutions.". Do not select statements that miss that criterion: "Agents cannot cooperate under any circumstances.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q28",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements describe the need for external planners?",
    options: [
      {
        text: "They offer fast approximate solutions to planning problems.",
        isCorrect: true,
      },
      {
        text: "They can help LLMs avoid long and error-prone reasoning chains.",
        isCorrect: true,
      },
      {
        text: "They provide deterministic planning baselines.",
        isCorrect: true,
      },
      {
        text: "They eliminate the need for any LLM-based reasoning.",
        isCorrect: false,
      },
    ],
    explanation:
      'Planners complement but do not replace LLM reasoning. To reason through the choices, select the statements that match the criterion in the prompt: "They offer fast approximate solutions to planning problems."; "They can help LLMs avoid long and error-prone reasoning chains."; "They provide deterministic planning baselines.". Do not select statements that miss that criterion: "They eliminate the need for any LLM-based reasoning.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q29",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements describe task-oriented deployments of agents?",
    options: [
      {
        text: "They require the agent to break down and execute user tasks.",
        isCorrect: true,
      },
      {
        text: "They often rely on internet search, APIs, or structured workflows.",
        isCorrect: true,
      },
      { text: "They focus on bounded, goal-driven tasks.", isCorrect: true },
      {
        text: "They require scientific discovery capabilities.",
        isCorrect: false,
      },
    ],
    explanation:
      'Task-oriented systems focus on practical, bounded tasks rather than open scientific exploration. To reason through the choices, select the statements that match the criterion in the prompt: "They require the agent to break down and execute user tasks."; "They often rely on internet search, APIs, or structured workflows."; "They focus on bounded, goal-driven tasks.". Do not select statements that miss that criterion: "They require scientific discovery capabilities.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q30",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements describe the advantages of using multiple specialized agents?",
    options: [
      {
        text: "Each agent can focus on a specific domain or skill.",
        isCorrect: true,
      },
      {
        text: "Agents can collaborate to handle complex workflows.",
        isCorrect: true,
      },
      {
        text: "Failures in one agent may be caught by another.",
        isCorrect: true,
      },
      {
        text: "It removes the need for communication between agents.",
        isCorrect: false,
      },
    ],
    explanation:
      'Specialization enables collaboration but requires communication. To reason through the choices, select the statements that match the criterion in the prompt: "Each agent can focus on a specific domain or skill."; "Agents can collaborate to handle complex workflows."; "Failures in one agent may be caught by another.". Do not select statements that miss that criterion: "It removes the need for communication between agents.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q31",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements describe risks of autonomous agent behavior?",
    options: [
      {
        text: "Agents may take unnecessary actions if goals are underspecified.",
        isCorrect: true,
      },
      {
        text: "Ambiguous instructions can lead to unexpected tool calls.",
        isCorrect: true,
      },
      {
        text: "Long action chains may amplify earlier reasoning errors.",
        isCorrect: true,
      },
      { text: "Autonomy inherently guarantees reliability.", isCorrect: false },
    ],
    explanation:
      'Autonomy requires careful constraints to avoid unintended behavior. To reason through the choices, select the statements that match the criterion in the prompt: "Agents may take unnecessary actions if goals are underspecified."; "Ambiguous instructions can lead to unexpected tool calls."; "Long action chains may amplify earlier reasoning errors.". Do not select statements that miss that criterion: "Autonomy inherently guarantees reliability.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q32",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements describe challenges in video perception for agents?",
    options: [
      { text: "The model must preserve frame ordering.", isCorrect: true },
      {
        text: "Temporal context must be represented consistently.",
        isCorrect: true,
      },
      {
        text: "Compression must not destroy critical motion signals.",
        isCorrect: true,
      },
      {
        text: "Video can be treated as independent static images without loss.",
        isCorrect: false,
      },
    ],
    explanation:
      'Temporal coherence is essential for video interpretation. To reason through the choices, select the statements that match the criterion in the prompt: "The model must preserve frame ordering."; "Temporal context must be represented consistently."; "Compression must not destroy critical motion signals.". Do not select statements that miss that criterion: "Video can be treated as independent static images without loss.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q33",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements describe web-search agent pipelines?",
    options: [
      {
        text: "They involve matching and ranking information relevant to a query.",
        isCorrect: true,
      },
      {
        text: "They may involve summarizing retrieved results.",
        isCorrect: true,
      },
      {
        text: "They may require reasoning about ambiguous user queries.",
        isCorrect: true,
      },
      { text: "They never rely on external APIs.", isCorrect: false },
    ],
    explanation:
      'Agents often rely on external search APIs to gather information. To reason through the choices, select the statements that match the criterion in the prompt: "They involve matching and ranking information relevant to a query."; "They may involve summarizing retrieved results."; "They may require reasoning about ambiguous user queries.". Do not select statements that miss that criterion: "They never rely on external APIs.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q34",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements describe why agents may fail in complex environments?",
    options: [
      {
        text: "They may overfit to patterns observed only in training contexts.",
        isCorrect: true,
      },
      {
        text: "They may misinterpret sensory input due to noise.",
        isCorrect: true,
      },
      {
        text: "They may fail to recognize missing or contradictory information.",
        isCorrect: true,
      },
      {
        text: "They always possess correct epistemic uncertainty estimates.",
        isCorrect: false,
      },
    ],
    explanation:
      'Agents can misinterpret cues or hallucinate but do not have built-in perfect uncertainty modeling. To reason through the choices, select the statements that match the criterion in the prompt: "They may overfit to patterns observed only in training contexts."; "They may misinterpret sensory input due to noise."; "They may fail to recognize missing or contradictory information.". Do not select statements that miss that criterion: "They always possess correct epistemic uncertainty estimates.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q35",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements describe natural-language reasoning traces produced by agents?",
    options: [
      {
        text: "They show intermediate thoughts or justifications.",
        isCorrect: true,
      },
      {
        text: "They help humans evaluate where the agent might be incorrect.",
        isCorrect: true,
      },
      { text: "They support debugging and interpretability.", isCorrect: true },
      {
        text: "They guarantee that the reasoning is factually accurate.",
        isCorrect: false,
      },
    ],
    explanation:
      'Reasoning traces improve interpretability but may still contain hallucinations. To reason through the choices, select the statements that match the criterion in the prompt: "They show intermediate thoughts or justifications."; "They help humans evaluate where the agent might be incorrect."; "They support debugging and interpretability.". Do not select statements that miss that criterion: "They guarantee that the reasoning is factually accurate.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q36",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements describe challenges in multi-agent coordination?",
    options: [
      { text: "Agents may generate conflicting plans.", isCorrect: true },
      { text: "Communication overhead may become large.", isCorrect: true },
      {
        text: "Misaligned incentives can reduce overall performance.",
        isCorrect: true,
      },
      {
        text: "Coordination automatically emerges without design.",
        isCorrect: false,
      },
    ],
    explanation:
      'Coordination requires explicit design, communication, and alignment mechanisms. To reason through the choices, select the statements that match the criterion in the prompt: "Agents may generate conflicting plans."; "Communication overhead may become large."; "Misaligned incentives can reduce overall performance.". Do not select statements that miss that criterion: "Coordination automatically emerges without design.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  // ============================================================
  //  Q37–Q54: 2 correct answers, 2 incorrect
  // ============================================================

  {
    id: "aie-ch4-q37",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements describe how LLMs interpret images through captioning-based approaches?",
    options: [
      {
        text: "A separate vision model generates a textual description.",
        isCorrect: true,
      },
      {
        text: "The caption becomes part of the prompt context.",
        isCorrect: true,
      },
      {
        text: "The LLM directly processes raw pixels without preprocessing.",
        isCorrect: false,
      },
      {
        text: "Captioning preserves all fine-grained visual information.",
        isCorrect: false,
      },
    ],
    explanation:
      'Captioning introduces information bottlenecks and relies on external vision models. To reason through the choices, select the statements that match the criterion in the prompt: "A separate vision model generates a textual description."; "The caption becomes part of the prompt context.". Do not select statements that miss that criterion: "The LLM directly processes raw pixels without preprocessing."; "Captioning preserves all fine-grained visual information.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q38",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements describe reasons an agent may incorporate external memory?",
    options: [
      { text: "To store long-term task history.", isCorrect: true },
      {
        text: "To retrieve prior reasoning traces or results.",
        isCorrect: true,
      },
      { text: "To increase parameter count of the LLM.", isCorrect: false },
      {
        text: "To reduce need for any retrieval mechanisms.",
        isCorrect: false,
      },
    ],
    explanation:
      'Memory augments context but does not modify model parameters. To reason through the choices, select the statements that match the criterion in the prompt: "To store long-term task history."; "To retrieve prior reasoning traces or results.". Do not select statements that miss that criterion: "To increase parameter count of the LLM."; "To reduce need for any retrieval mechanisms.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q39",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements describe why specialized agents may outperform a single general agent?",
    options: [
      {
        text: "They can optimize for domain-specific reasoning.",
        isCorrect: true,
      },
      {
        text: "They can split workload across multiple focused skills.",
        isCorrect: true,
      },
      {
        text: "They remove all need for coordination mechanisms.",
        isCorrect: false,
      },
      {
        text: "They eliminate misunderstandings between models.",
        isCorrect: false,
      },
    ],
    explanation:
      'Specialization helps but coordination is still essential. To reason through the choices, select the statements that match the criterion in the prompt: "They can optimize for domain-specific reasoning."; "They can split workload across multiple focused skills.". Do not select statements that miss that criterion: "They remove all need for coordination mechanisms."; "They eliminate misunderstandings between models.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q40",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements describe difficulties in designing embodied action systems?",
    options: [
      { text: "Real-world uncertainty complicates planning.", isCorrect: true },
      {
        text: "Physical feasibility must be enforced in actions.",
        isCorrect: true,
      },
      {
        text: "Symbolic-only representations are always sufficient.",
        isCorrect: false,
      },
      { text: "Training requires only textual data.", isCorrect: false },
    ],
    explanation:
      'Embodied systems require multimodal grounding and feasible action enforcement. To reason through the choices, select the statements that match the criterion in the prompt: "Real-world uncertainty complicates planning."; "Physical feasibility must be enforced in actions.". Do not select statements that miss that criterion: "Symbolic-only representations are always sufficient."; "Training requires only textual data.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q41",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements describe benefits of multi-turn interaction between a user and an agent?",
    options: [
      { text: "It allows refining queries iteratively.", isCorrect: true },
      { text: "It enables context accumulation over turns.", isCorrect: true },
      {
        text: "It removes ambiguity from all instructions automatically.",
        isCorrect: false,
      },
      { text: "It guarantees correct tool use every time.", isCorrect: false },
    ],
    explanation:
      'Multi-turn interaction improves context use but cannot guarantee flawless execution. To reason through the choices, select the statements that match the criterion in the prompt: "It allows refining queries iteratively."; "It enables context accumulation over turns.". Do not select statements that miss that criterion: "It removes ambiguity from all instructions automatically."; "It guarantees correct tool use every time.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q42",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements describe inter-agent feedback systems?",
    options: [
      { text: "One agent may critique another’s output.", isCorrect: true },
      {
        text: "Feedback may improve performance through iteration.",
        isCorrect: true,
      },
      { text: "Feedback ensures agents never disagree.", isCorrect: false },
      {
        text: "Feedback always converges on the optimal solution.",
        isCorrect: false,
      },
    ],
    explanation:
      'Feedback loops help but do not guarantee convergence or agreement. To reason through the choices, select the statements that match the criterion in the prompt: "One agent may critique another’s output."; "Feedback may improve performance through iteration.". Do not select statements that miss that criterion: "Feedback ensures agents never disagree."; "Feedback always converges on the optimal solution.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q43",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements describe properties of multi-modal alignment training?",
    options: [
      {
        text: "It helps link embeddings across different modalities.",
        isCorrect: true,
      },
      {
        text: "It reduces discrepancies between text and image representations.",
        isCorrect: true,
      },
      {
        text: "It allows direct processing of raw sensory input without encoders.",
        isCorrect: false,
      },
      {
        text: "It removes the need for modality-specific datasets.",
        isCorrect: false,
      },
    ],
    explanation:
      'Alignment enforces representational consistency but still requires encoders and datasets. To reason through the choices, select the statements that match the criterion in the prompt: "It helps link embeddings across different modalities."; "It reduces discrepancies between text and image representations.". Do not select statements that miss that criterion: "It allows direct processing of raw sensory input without encoders."; "It removes the need for modality-specific datasets.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q44",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements describe reasons an agent might mis-rank documents for a query?",
    options: [
      { text: "It may misinterpret the user’s intent.", isCorrect: true },
      {
        text: "It may prioritize irrelevant but semantically similar content.",
        isCorrect: true,
      },
      {
        text: "Ranking quality is unrelated to model reasoning ability.",
        isCorrect: false,
      },
      {
        text: "All LLMs inherently understand user preferences.",
        isCorrect: false,
      },
    ],
    explanation:
      'Ranking requires semantic interpretation and preference modeling, not guaranteed by default. To reason through the choices, select the statements that match the criterion in the prompt: "It may misinterpret the user’s intent."; "It may prioritize irrelevant but semantically similar content.". Do not select statements that miss that criterion: "Ranking quality is unrelated to model reasoning ability."; "All LLMs inherently understand user preferences.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q45",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements describe challenges in tool-use interpretation?",
    options: [
      { text: "The model must infer when a tool is needed.", isCorrect: true },
      {
        text: "The model must choose the correct tool among many.",
        isCorrect: true,
      },
      {
        text: "Tool names alone fully encode the tool’s purpose.",
        isCorrect: false,
      },
      {
        text: "Tools automatically execute safely without constraints.",
        isCorrect: false,
      },
    ],
    explanation:
      'Agents must interpret tool descriptions and constraints to use them correctly. To reason through the choices, select the statements that match the criterion in the prompt: "The model must infer when a tool is needed."; "The model must choose the correct tool among many.". Do not select statements that miss that criterion: "Tool names alone fully encode the tool’s purpose."; "Tools automatically execute safely without constraints.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q46",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements describe properties of search-and-summarize workflows?",
    options: [
      { text: "They require retrieving relevant documents.", isCorrect: true },
      {
        text: "They require synthesizing retrieved information.",
        isCorrect: true,
      },
      {
        text: "They eliminate the need for ranking results.",
        isCorrect: false,
      },
      {
        text: "They assume all sources are equally reliable.",
        isCorrect: false,
      },
    ],
    explanation:
      'Summarization depends on ranking and does not treat all sources equally. To reason through the choices, select the statements that match the criterion in the prompt: "They require retrieving relevant documents."; "They require synthesizing retrieved information.". Do not select statements that miss that criterion: "They eliminate the need for ranking results."; "They assume all sources are equally reliable.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q47",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements describe why agents need planning steps before tool use?",
    options: [
      {
        text: "Planning identifies which tool aligns with the goal.",
        isCorrect: true,
      },
      {
        text: "Planning clarifies subgoals needed to reach the final solution.",
        isCorrect: true,
      },
      {
        text: "Planning eliminates the need for action execution.",
        isCorrect: false,
      },
      { text: "Planning guarantees perfect accuracy.", isCorrect: false },
    ],
    explanation:
      'Planning guides correct tool selection but cannot guarantee accuracy. To reason through the choices, select the statements that match the criterion in the prompt: "Planning identifies which tool aligns with the goal."; "Planning clarifies subgoals needed to reach the final solution.". Do not select statements that miss that criterion: "Planning eliminates the need for action execution."; "Planning guarantees perfect accuracy.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q48",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements describe problems in multi-agent majority voting?",
    options: [
      { text: "Agents may share the same failure modes.", isCorrect: true },
      {
        text: "Votes may converge on an incorrect consensus.",
        isCorrect: true,
      },
      {
        text: "Majority voting ensures perfect decision-making.",
        isCorrect: false,
      },
      {
        text: "Voting removes the need for individual reasoning.",
        isCorrect: false,
      },
    ],
    explanation:
      'Voting can amplify shared errors; it is not guaranteed to find the correct answer. To reason through the choices, select the statements that match the criterion in the prompt: "Agents may share the same failure modes."; "Votes may converge on an incorrect consensus.". Do not select statements that miss that criterion: "Majority voting ensures perfect decision-making."; "Voting removes the need for individual reasoning.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q49",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements describe what an LLM 'planner' module may do?",
    options: [
      { text: "Select steps to achieve a goal.", isCorrect: true },
      { text: "Sequence subtasks into an actionable plan.", isCorrect: true },
      { text: "Replace the need for any LLM reasoning.", isCorrect: false },
      { text: "Guarantee optimality across all tasks.", isCorrect: false },
    ],
    explanation:
      'Planners support but do not replace LLM reasoning and cannot guarantee optimality. To reason through the choices, select the statements that match the criterion in the prompt: "Select steps to achieve a goal."; "Sequence subtasks into an actionable plan.". Do not select statements that miss that criterion: "Replace the need for any LLM reasoning."; "Guarantee optimality across all tasks.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q50",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements describe challenges in dynamic environments?",
    options: [
      { text: "The agent must detect changes in state.", isCorrect: true },
      { text: "The agent may need to replan frequently.", isCorrect: true },
      {
        text: "Static plans suffice regardless of environmental changes.",
        isCorrect: false,
      },
      {
        text: "Reactivity can be ignored without consequence.",
        isCorrect: false,
      },
    ],
    explanation:
      'Dynamic environments require perception and rapid replanning. To reason through the choices, select the statements that match the criterion in the prompt: "The agent must detect changes in state."; "The agent may need to replan frequently.". Do not select statements that miss that criterion: "Static plans suffice regardless of environmental changes."; "Reactivity can be ignored without consequence.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q51",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements describe why multimodal alignment is challenging?",
    options: [
      {
        text: "Different modalities vary in structure and dimensionality.",
        isCorrect: true,
      },
      { text: "Cross-modal semantics must be learned.", isCorrect: true },
      {
        text: "All modalities naturally share identical feature spaces.",
        isCorrect: false,
      },
      {
        text: "Alignment is achievable without labeled data.",
        isCorrect: false,
      },
    ],
    explanation:
      'Alignment requires bridging heterogeneous feature spaces and often uses labeled datasets. To reason through the choices, select the statements that match the criterion in the prompt: "Different modalities vary in structure and dimensionality."; "Cross-modal semantics must be learned.". Do not select statements that miss that criterion: "All modalities naturally share identical feature spaces."; "Alignment is achievable without labeled data.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q52",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements describe benefits of RAG-based memory for agents?",
    options: [
      {
        text: "It enables retrieval of past experiences or data.",
        isCorrect: true,
      },
      { text: "It helps reduce repeated reasoning steps.", isCorrect: true },
      { text: "It replaces the need for planning entirely.", isCorrect: false },
      {
        text: "It guarantees perfect recall of all prior interactions.",
        isCorrect: false,
      },
    ],
    explanation:
      'RAG improves recall but does not replace planning or guarantee completeness. To reason through the choices, select the statements that match the criterion in the prompt: "It enables retrieval of past experiences or data."; "It helps reduce repeated reasoning steps.". Do not select statements that miss that criterion: "It replaces the need for planning entirely."; "It guarantees perfect recall of all prior interactions.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q53",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements describe why LLMs may hallucinate during web tasks?",
    options: [
      {
        text: "They may fill missing information with plausible text.",
        isCorrect: true,
      },
      {
        text: "They may misinterpret irrelevant webpage sections.",
        isCorrect: true,
      },
      {
        text: "They have perfect recall of webpage structure.",
        isCorrect: false,
      },
      {
        text: "They always know which information is authoritative.",
        isCorrect: false,
      },
    ],
    explanation:
      'Hallucinations often occur under ambiguity or irrelevant context. To reason through the choices, select the statements that match the criterion in the prompt: "They may fill missing information with plausible text."; "They may misinterpret irrelevant webpage sections.". Do not select statements that miss that criterion: "They have perfect recall of webpage structure."; "They always know which information is authoritative.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q54",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements describe limitations of decomposition-first planning?",
    options: [
      {
        text: "Errors introduced early may propagate through the plan.",
        isCorrect: true,
      },
      {
        text: "The plan cannot adjust based on unexpected results.",
        isCorrect: true,
      },
      {
        text: "It guarantees the shortest possible sequence of steps.",
        isCorrect: false,
      },
      {
        text: "It inherently corrects its assumptions during execution.",
        isCorrect: false,
      },
    ],
    explanation:
      'Static decomposition lacks adaptability and may amplify initial errors. To reason through the choices, select the statements that match the criterion in the prompt: "Errors introduced early may propagate through the plan."; "The plan cannot adjust based on unexpected results.". Do not select statements that miss that criterion: "It guarantees the shortest possible sequence of steps."; "It inherently corrects its assumptions during execution.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  // ============================================================
  //  Q55–Q72: 1 correct answer, 3 incorrect
  // ============================================================

  {
    id: "aie-ch4-q55",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statement correctly describes the LLM’s role in an agent architecture?",
    options: [
      {
        text: "It serves as the central reasoning and decision-making component.",
        isCorrect: true,
      },
      {
        text: "It replaces the need for any perception modules.",
        isCorrect: false,
      },
      {
        text: "It inherently includes real-time knowledge of the world.",
        isCorrect: false,
      },
      {
        text: "It stores persistent long-term memory across tasks.",
        isCorrect: false,
      },
    ],
    explanation:
      'The LLM acts as the agent brain but lacks built-in perception, real-time knowledge, and long-term memory. To reason through the choices, select the statements that match the criterion in the prompt: "It serves as the central reasoning and decision-making component.". Do not select statements that miss that criterion: "It replaces the need for any perception modules."; "It inherently includes real-time knowledge of the world."; "It stores persistent long-term memory across tasks.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q56",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statement correctly describes an advantage of using external search tools?",
    options: [
      {
        text: "They give the agent access to real-time information.",
        isCorrect: true,
      },
      {
        text: "They eliminate the need for ranking search results.",
        isCorrect: false,
      },
      {
        text: "They allow the LLM to bypass planning entirely.",
        isCorrect: false,
      },
      { text: "They guarantee perfectly accurate results.", isCorrect: false },
    ],
    explanation:
      'Search tools provide timely information but still require planning and relevance evaluation. To reason through the choices, select the statements that match the criterion in the prompt: "They give the agent access to real-time information.". Do not select statements that miss that criterion: "They eliminate the need for ranking search results."; "They allow the LLM to bypass planning entirely."; "They guarantee perfectly accurate results.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q57",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statement correctly describes a drawback of multi-agent competition?",
    options: [
      {
        text: "Competition can amplify shared weaknesses among agents.",
        isCorrect: true,
      },
      { text: "Competition always increases accuracy.", isCorrect: false },
      {
        text: "Competition removes the need for evaluation.",
        isCorrect: false,
      },
      {
        text: "Competition guarantees convergence to optimal strategies.",
        isCorrect: false,
      },
    ],
    explanation:
      'Competitive setups may reinforce biases in shared training data. To reason through the choices, select the statements that match the criterion in the prompt: "Competition can amplify shared weaknesses among agents.". Do not select statements that miss that criterion: "Competition always increases accuracy."; "Competition removes the need for evaluation."; "Competition guarantees convergence to optimal strategies.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q58",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statement correctly describes interleaved planning?",
    options: [
      {
        text: "It alternates reasoning and execution to adapt to intermediate results.",
        isCorrect: true,
      },
      {
        text: "It ensures the shortest reasoning chain possible.",
        isCorrect: false,
      },
      { text: "It prevents any need for feedback loops.", isCorrect: false },
      { text: "It guarantees consistent convergence.", isCorrect: false },
    ],
    explanation:
      'Interleaving improves adaptability but cannot guarantee efficiency or convergence. To reason through the choices, select the statements that match the criterion in the prompt: "It alternates reasoning and execution to adapt to intermediate results.". Do not select statements that miss that criterion: "It ensures the shortest reasoning chain possible."; "It prevents any need for feedback loops."; "It guarantees consistent convergence.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q59",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statement correctly describes multimodal models?",
    options: [
      {
        text: "They incorporate inputs from multiple sensory channels such as images or audio.",
        isCorrect: true,
      },
      {
        text: "They eliminate the need for specialized encoders.",
        isCorrect: false,
      },
      {
        text: "They naturally understand all sensory modalities without training.",
        isCorrect: false,
      },
      {
        text: "They replace the need for textual reasoning.",
        isCorrect: false,
      },
    ],
    explanation:
      'Multimodal models rely on encoders and still require textual reasoning. To reason through the choices, select the statements that match the criterion in the prompt: "They incorporate inputs from multiple sensory channels such as images or audio.". Do not select statements that miss that criterion: "They eliminate the need for specialized encoders."; "They naturally understand all sensory modalities without training."; "They replace the need for textual reasoning.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q60",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statement correctly describes a challenge in reflection-based planning?",
    options: [
      {
        text: "Reflection can cause infinite loops where the agent repeatedly reevaluates without acting.",
        isCorrect: true,
      },
      {
        text: "Reflection guarantees optimal task decomposition.",
        isCorrect: false,
      },
      {
        text: "Reflection removes the need for intermediate results.",
        isCorrect: false,
      },
      {
        text: "Reflection inherently prevents hallucinations.",
        isCorrect: false,
      },
    ],
    explanation:
      'Reflection loops may stall progress if not constrained. To reason through the choices, select the statements that match the criterion in the prompt: "Reflection can cause infinite loops where the agent repeatedly reevaluates without acting.". Do not select statements that miss that criterion: "Reflection guarantees optimal task decomposition."; "Reflection removes the need for intermediate results."; "Reflection inherently prevents hallucinations.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q61",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statement correctly describes a challenge in ranking ambiguous queries?",
    options: [
      {
        text: "The model must disambiguate user intent before ranking effectively.",
        isCorrect: true,
      },
      {
        text: "Ambiguity automatically improves ranking performance.",
        isCorrect: false,
      },
      {
        text: "The agent always knows which domain is intended.",
        isCorrect: false,
      },
      {
        text: "Ranking is unnecessary when ambiguity exists.",
        isCorrect: false,
      },
    ],
    explanation:
      'Intent clarification is necessary before ranking can succeed. To reason through the choices, select the statements that match the criterion in the prompt: "The model must disambiguate user intent before ranking effectively.". Do not select statements that miss that criterion: "Ambiguity automatically improves ranking performance."; "The agent always knows which domain is intended."; "Ranking is unnecessary when ambiguity exists.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q62",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statement correctly describes an agent’s ability to use tools?",
    options: [
      {
        text: "The LLM must decide when tool invocation is appropriate.",
        isCorrect: true,
      },
      {
        text: "Tools automatically trigger without model control.",
        isCorrect: false,
      },
      {
        text: "Tool selection requires no description or metadata.",
        isCorrect: false,
      },
      { text: "Tool use eliminates the need for reasoning.", isCorrect: false },
    ],
    explanation:
      'LLMs must reason about when and how to invoke tools. To reason through the choices, select the statements that match the criterion in the prompt: "The LLM must decide when tool invocation is appropriate.". Do not select statements that miss that criterion: "Tools automatically trigger without model control."; "Tool selection requires no description or metadata."; "Tool use eliminates the need for reasoning.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q63",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statement correctly describes a limitation of multi-agent collaboration?",
    options: [
      {
        text: "Coordination overhead can reduce overall efficiency.",
        isCorrect: true,
      },
      { text: "More agents always outperform fewer agents.", isCorrect: false },
      { text: "Collaboration eliminates failure modes.", isCorrect: false },
      { text: "Communication guarantees perfect alignment.", isCorrect: false },
    ],
    explanation:
      'Coordination introduces overhead and does not guarantee correctness. To reason through the choices, select the statements that match the criterion in the prompt: "Coordination overhead can reduce overall efficiency.". Do not select statements that miss that criterion: "More agents always outperform fewer agents."; "Collaboration eliminates failure modes."; "Communication guarantees perfect alignment.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q64",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statement correctly describes the purpose of a perception module?",
    options: [
      {
        text: "To convert sensory inputs into forms the LLM can process.",
        isCorrect: true,
      },
      { text: "To store long-term memory.", isCorrect: false },
      { text: "To replace the need for an LLM.", isCorrect: false },
      {
        text: "To generate final textual answers for the user.",
        isCorrect: false,
      },
    ],
    explanation:
      'Perception modules convert raw sensory input into a usable representation. To reason through the choices, select the statements that match the criterion in the prompt: "To convert sensory inputs into forms the LLM can process.". Do not select statements that miss that criterion: "To store long-term memory."; "To replace the need for an LLM."; "To generate final textual answers for the user.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q65",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statement correctly describes a limitation of caption-based perception?",
    options: [
      {
        text: "Captioning compresses visual information and may omit details.",
        isCorrect: true,
      },
      {
        text: "Captioning always encodes all pixel-level semantics.",
        isCorrect: false,
      },
      {
        text: "Captioning inherently matches human perception perfectly.",
        isCorrect: false,
      },
      {
        text: "Captioning eliminates the need for multimodal models.",
        isCorrect: false,
      },
    ],
    explanation:
      'Captioning introduces lossy translation from vision to text. To reason through the choices, select the statements that match the criterion in the prompt: "Captioning compresses visual information and may omit details.". Do not select statements that miss that criterion: "Captioning always encodes all pixel-level semantics."; "Captioning inherently matches human perception perfectly."; "Captioning eliminates the need for multimodal models.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q66",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statement correctly identifies a risk of static task decomposition?",
    options: [
      {
        text: "It cannot adapt when new information arises during execution.",
        isCorrect: true,
      },
      { text: "It produces perfect plans for all tasks.", isCorrect: false },
      {
        text: "It automatically integrates user feedback mid-execution.",
        isCorrect: false,
      },
      {
        text: "It consistently outperforms adaptive strategies.",
        isCorrect: false,
      },
    ],
    explanation:
      'Static planning has no mechanism for mid-execution adaptation. To reason through the choices, select the statements that match the criterion in the prompt: "It cannot adapt when new information arises during execution.". Do not select statements that miss that criterion: "It produces perfect plans for all tasks."; "It automatically integrates user feedback mid-execution."; "It consistently outperforms adaptive strategies.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q67",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statement correctly describes a benefit of using multiple agents?",
    options: [
      {
        text: "Different agents can specialize in different tasks.",
        isCorrect: true,
      },
      { text: "Multiple agents guarantee perfect accuracy.", isCorrect: false },
      {
        text: "Multi-agent setups remove the need for planning.",
        isCorrect: false,
      },
      { text: "Agents never need to communicate.", isCorrect: false },
    ],
    explanation:
      'Specialization is beneficial, but coordination is required. To reason through the choices, select the statements that match the criterion in the prompt: "Different agents can specialize in different tasks.". Do not select statements that miss that criterion: "Multiple agents guarantee perfect accuracy."; "Multi-agent setups remove the need for planning."; "Agents never need to communicate.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q68",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statement correctly describes the role of ranking in search pipelines?",
    options: [
      {
        text: "It orders candidate documents by estimated relevance.",
        isCorrect: true,
      },
      {
        text: "It eliminates the need for matching algorithms.",
        isCorrect: false,
      },
      {
        text: "It guarantees that irrelevant documents never appear.",
        isCorrect: false,
      },
      {
        text: "It replaces the need for summarization tools.",
        isCorrect: false,
      },
    ],
    explanation:
      'Ranking prioritizes information but does not guarantee perfect filtering. To reason through the choices, select the statements that match the criterion in the prompt: "It orders candidate documents by estimated relevance.". Do not select statements that miss that criterion: "It eliminates the need for matching algorithms."; "It guarantees that irrelevant documents never appear."; "It replaces the need for summarization tools.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q69",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statement correctly describes a challenge of majority-vote planning?",
    options: [
      {
        text: "Models may converge on plausible but incorrect plans.",
        isCorrect: true,
      },
      { text: "Voting inherently removes hallucinations.", isCorrect: false },
      {
        text: "Voting requires no reasoning by individual agents.",
        isCorrect: false,
      },
      {
        text: "Voting guarantees optimal results in large groups.",
        isCorrect: false,
      },
    ],
    explanation:
      'Voting amplifies shared biases and can validate incorrect but plausible plans. To reason through the choices, select the statements that match the criterion in the prompt: "Models may converge on plausible but incorrect plans.". Do not select statements that miss that criterion: "Voting inherently removes hallucinations."; "Voting requires no reasoning by individual agents."; "Voting guarantees optimal results in large groups.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q70",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statement correctly describes the main benefit of tool descriptions?",
    options: [
      {
        text: "They help the LLM understand when and how to use a tool.",
        isCorrect: true,
      },
      {
        text: "They eliminate the need for the LLM to parse arguments.",
        isCorrect: false,
      },
      {
        text: "They ensure the tool will always be used correctly.",
        isCorrect: false,
      },
      { text: "They remove ambiguity from all tasks.", isCorrect: false },
    ],
    explanation:
      'Tool descriptions guide usage but do not guarantee correctness. To reason through the choices, select the statements that match the criterion in the prompt: "They help the LLM understand when and how to use a tool.". Do not select statements that miss that criterion: "They eliminate the need for the LLM to parse arguments."; "They ensure the tool will always be used correctly."; "They remove ambiguity from all tasks.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q71",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statement correctly describes the purpose of plan reflection?",
    options: [
      {
        text: "To evaluate or revise a candidate plan before execution.",
        isCorrect: true,
      },
      { text: "To remove the need for decomposition.", isCorrect: false },
      { text: "To guarantee optimal plans in all cases.", isCorrect: false },
      { text: "To prevent the need for tool use.", isCorrect: false },
    ],
    explanation:
      'Reflection is a quality-improving step but cannot replace planning entirely. To reason through the choices, select the statements that match the criterion in the prompt: "To evaluate or revise a candidate plan before execution.". Do not select statements that miss that criterion: "To remove the need for decomposition."; "To guarantee optimal plans in all cases."; "To prevent the need for tool use.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "aie-ch4-q72",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statement correctly describes a challenge of LLM-based web browsing?",
    options: [
      {
        text: "The model may struggle to identify relevant information in long or cluttered webpages.",
        isCorrect: true,
      },
      {
        text: "The model always extracts the correct information automatically.",
        isCorrect: false,
      },
      {
        text: "The model navigates HTML trees without explicit instruction.",
        isCorrect: false,
      },
      {
        text: "The model inherently knows which sites are trustworthy.",
        isCorrect: false,
      },
    ],
    explanation:
      'Real webpages contain noise and require careful reasoning to extract relevant data. To reason through the choices, select the statements that match the criterion in the prompt: "The model may struggle to identify relevant information in long or cluttered webpages.". Do not select statements that miss that criterion: "The model always extracts the correct information automatically."; "The model navigates HTML trees without explicit instruction."; "The model inherently knows which sites are trustworthy.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },
];
