export type MemoryScenarioId =
  | "personal-assistant"
  | "coding-agent"
  | "research-swarm";

export type MemoryFormId =
  | "flat-token"
  | "planar-token"
  | "hierarchical-token"
  | "internal-parametric"
  | "external-parametric"
  | "generated-latent"
  | "reused-latent"
  | "transformed-latent";

export type MemoryFunctionId =
  | "user-factual"
  | "environment-factual"
  | "case-experience"
  | "strategy-experience"
  | "skill-experience"
  | "single-turn-working"
  | "multi-turn-working";

export type FormationId =
  | "semantic-summary"
  | "knowledge-distillation"
  | "structured-construction"
  | "latent-representation"
  | "parametric-internalization";

export type EvolutionId = "consolidation" | "updating" | "forgetting";

export type RetrievalId =
  | "task-start"
  | "on-demand-semantic"
  | "graph-traversal"
  | "working-context";

export type MemoryScenario = {
  id: MemoryScenarioId;
  label: string;
  shortLabel: string;
  setup: string;
  goal: string;
  preferredForms: readonly MemoryFormId[];
  preferredFunctions: readonly MemoryFunctionId[];
  preferredFormation: readonly FormationId[];
  preferredEvolution: readonly EvolutionId[];
  preferredRetrieval: readonly RetrievalId[];
};

export type MemoryForm = {
  id: MemoryFormId;
  family: "Token-level" | "Parametric" | "Latent";
  label: string;
  shortLabel: string;
  definition: string;
  bestFor: string;
  caution: string;
  transparency: number;
  updateEase: number;
  reasoningStructure: number;
  contextEfficiency: number;
};

export type MemoryFunction = {
  id: MemoryFunctionId;
  family: "Factual" | "Experiential" | "Working";
  label: string;
  shortLabel: string;
  role: string;
  example: string;
};

export type FormationOperation = {
  id: FormationId;
  label: string;
  verb: string;
  explanation: string;
};

export type EvolutionOperation = {
  id: EvolutionId;
  label: string;
  verb: string;
  explanation: string;
};

export type RetrievalOperation = {
  id: RetrievalId;
  label: string;
  timing: string;
  explanation: string;
};

export type MemoryDesignInput = {
  scenarioId: MemoryScenarioId;
  formId: MemoryFormId;
  functionId: MemoryFunctionId;
  formationId: FormationId;
  evolutionIds: readonly EvolutionId[];
  retrievalId: RetrievalId;
};

export type MemoryDesignEvaluation = {
  score: number;
  fitLabel: string;
  diagnosis: string;
  strengths: readonly string[];
  warnings: readonly string[];
  metrics: {
    transparency: number;
    adaptability: number;
    structure: number;
    contextEfficiency: number;
  };
};

export const memoryScenarios = [
  {
    id: "personal-assistant",
    label: "Personal assistant with deletable user facts",
    shortLabel: "Assistant",
    setup:
      "A calendar-and-email agent must remember preferences, commitments, corrections, and sensitive personal details across months.",
    goal: "Preserve continuity while keeping every sensitive fact inspectable, correctable, and deletable.",
    preferredForms: ["flat-token", "planar-token", "hierarchical-token"],
    preferredFunctions: ["user-factual", "multi-turn-working"],
    preferredFormation: ["semantic-summary", "structured-construction"],
    preferredEvolution: ["updating", "forgetting", "consolidation"],
    preferredRetrieval: ["task-start", "on-demand-semantic"],
  },
  {
    id: "coding-agent",
    label: "Coding agent that improves across repeated tasks",
    shortLabel: "Coding agent",
    setup:
      "A software agent sees bug reports, tool traces, failed patches, tests, and reusable scripts from many repository tasks.",
    goal: "Turn episodes into strategies and executable skills without losing the raw cases needed for debugging.",
    preferredForms: [
      "hierarchical-token",
      "planar-token",
      "external-parametric",
    ],
    preferredFunctions: [
      "case-experience",
      "strategy-experience",
      "skill-experience",
    ],
    preferredFormation: [
      "structured-construction",
      "knowledge-distillation",
      "semantic-summary",
    ],
    preferredEvolution: ["consolidation", "updating", "forgetting"],
    preferredRetrieval: ["graph-traversal", "on-demand-semantic"],
  },
  {
    id: "research-swarm",
    label: "Multi-agent research swarm with shared state",
    shortLabel: "Research swarm",
    setup:
      "Several agents read papers, run tools, exchange findings, and need a shared substrate for claims, evidence, and task ownership.",
    goal: "Coordinate across agents while distinguishing shared environment facts from local working context.",
    preferredForms: ["planar-token", "hierarchical-token", "generated-latent"],
    preferredFunctions: [
      "environment-factual",
      "single-turn-working",
      "multi-turn-working",
    ],
    preferredFormation: [
      "structured-construction",
      "semantic-summary",
      "latent-representation",
    ],
    preferredEvolution: ["updating", "consolidation", "forgetting"],
    preferredRetrieval: ["graph-traversal", "working-context"],
  },
] as const satisfies readonly MemoryScenario[];

export const memoryForms = [
  {
    id: "flat-token",
    family: "Token-level",
    label: "Flat token-level memory",
    shortLabel: "Flat records",
    definition:
      "Explicit entries such as snippets, logs, profiles, images, trajectories, or summaries stored without an explicit topology.",
    bestFor:
      "Fast append, inspection, deletion, simple semantic search, and audit-friendly user facts.",
    caution:
      "Relationships are not encoded directly, so retrieval quality carries much of the reasoning burden.",
    transparency: 5,
    updateEase: 5,
    reasoningStructure: 1,
    contextEfficiency: 2,
  },
  {
    id: "planar-token",
    family: "Token-level",
    label: "Planar token-level memory",
    shortLabel: "Graph/table",
    definition:
      "Explicit units organized in one layer, such as a graph, tree, table, or relation map.",
    bestFor:
      "Environment facts, shared workspaces, conflict checks, and multi-hop traversal over related entries.",
    caution:
      "The links improve reasoning, but they must be built, maintained, and searched.",
    transparency: 5,
    updateEase: 4,
    reasoningStructure: 4,
    contextEfficiency: 3,
  },
  {
    id: "hierarchical-token",
    family: "Token-level",
    label: "Hierarchical token-level memory",
    shortLabel: "Layered stack",
    definition:
      "Explicit memory across abstraction levels, such as raw traces, event summaries, and reusable themes with cross-layer links.",
    bestFor:
      "Long-horizon agents that need to move between evidence, summaries, strategies, and skills.",
    caution:
      "The system must preserve semantic meaning while compressing and navigating dense layers.",
    transparency: 4,
    updateEase: 3,
    reasoningStructure: 5,
    contextEfficiency: 4,
  },
  {
    id: "internal-parametric",
    family: "Parametric",
    label: "Internal parametric memory",
    shortLabel: "Base weights",
    definition:
      "Information stored directly in the original model parameters through training or editing.",
    bestFor:
      "Broad domain priors or stable competence where no extra retrieval module should be needed at inference.",
    caution:
      "New or personal facts are expensive to update, hard to inspect, and can interfere with older knowledge.",
    transparency: 1,
    updateEase: 1,
    reasoningStructure: 2,
    contextEfficiency: 5,
  },
  {
    id: "external-parametric",
    family: "Parametric",
    label: "External parametric memory",
    shortLabel: "Adapters",
    definition:
      "Auxiliary learned modules such as adapters, LoRA modules, proxy models, or routing modules attached to the base model.",
    bestFor:
      "Modular personalization, task-specific competence, rollback, and controlled learned skills.",
    caution:
      "It is more modular than editing base weights, but less inspectable than explicit token records.",
    transparency: 2,
    updateEase: 3,
    reasoningStructure: 3,
    contextEfficiency: 5,
  },
  {
    id: "generated-latent",
    family: "Latent",
    label: "Generated latent memory",
    shortLabel: "Latent slots",
    definition:
      "A learned module creates compact hidden states, embeddings, or memory tokens from long contexts or trajectories.",
    bestFor:
      "Compressing long inputs when the raw records are too large to keep active.",
    caution:
      "The resulting state is efficient but difficult to audit or surgically correct.",
    transparency: 1,
    updateEase: 2,
    reasoningStructure: 3,
    contextEfficiency: 5,
  },
  {
    id: "reused-latent",
    family: "Latent",
    label: "Reused latent memory",
    shortLabel: "KV reuse",
    definition:
      "Prior computation, especially key-value cache state, is carried forward as implicit memory.",
    bestFor:
      "Fast reuse of recent computation inside a bounded inference or session.",
    caution:
      "The state is hard to inspect and can become a stale shortcut if the task changes.",
    transparency: 1,
    updateEase: 2,
    reasoningStructure: 2,
    contextEfficiency: 5,
  },
  {
    id: "transformed-latent",
    family: "Latent",
    label: "Transformed latent memory",
    shortLabel: "Compressed states",
    definition:
      "Existing activations or caches are pooled, merged, pruned, distilled, or re-encoded into a smaller hidden state.",
    bestFor:
      "Working-memory compression when the agent needs less footprint without returning to readable text.",
    caution:
      "Compression can hide what was dropped, making verification and provenance harder.",
    transparency: 1,
    updateEase: 2,
    reasoningStructure: 3,
    contextEfficiency: 5,
  },
] as const satisfies readonly MemoryForm[];

export const memoryFunctions = [
  {
    id: "user-factual",
    family: "Factual",
    label: "User factual memory",
    shortLabel: "User facts",
    role: "Preserve declarative knowledge about the user, preferences, commitments, and interaction history.",
    example:
      "The user prefers meeting summaries in bullets and corrected a previously stored travel constraint.",
  },
  {
    id: "environment-factual",
    family: "Factual",
    label: "Environment factual memory",
    shortLabel: "Environment facts",
    role: "Track external entities, documents, resources, tool states, codebase facts, or shared workspace state.",
    example:
      "A shared table records which paper each agent has read and which claim still needs evidence.",
  },
  {
    id: "case-experience",
    family: "Experiential",
    label: "Case-based experiential memory",
    shortLabel: "Cases",
    role: "Keep concrete episodes, trajectories, failures, and successes for replay or analogy.",
    example:
      "A previous failing patch, the tests it broke, and the final fix remain available as a case.",
  },
  {
    id: "strategy-experience",
    family: "Experiential",
    label: "Strategy-based experiential memory",
    shortLabel: "Strategies",
    role: "Compress episodes into reusable tactics, heuristics, plans, or reflection notes.",
    example:
      "When a parser bug appears after a dependency update, inspect generated artifacts before rewriting logic.",
  },
  {
    id: "skill-experience",
    family: "Experiential",
    label: "Skill-based experiential memory",
    shortLabel: "Skills",
    role: "Store callable capabilities such as code snippets, scripts, tools, APIs, or specialized subagents.",
    example:
      "A verified script extracts paper section headings and source snippets for future survey ingestion.",
  },
  {
    id: "single-turn-working",
    family: "Working",
    label: "Single-turn working memory",
    shortLabel: "Single-turn workspace",
    role: "Select, compress, or transform task-relevant input within one large inference or one immediate task.",
    example:
      "A long document is condensed into key findings before the agent answers a single question.",
  },
  {
    id: "multi-turn-working",
    family: "Working",
    label: "Multi-turn working memory",
    shortLabel: "Multi-turn workspace",
    role: "Maintain and revise the active scratchpad across a continuing session or multi-step task.",
    example:
      "The current plan, unresolved tool results, and assumptions stay active while the agent iterates.",
  },
] as const satisfies readonly MemoryFunction[];

export const formationOperations = [
  {
    id: "semantic-summary",
    label: "Semantic summarization",
    verb: "Compress",
    explanation:
      "Turn raw traces into concise, readable memories that preserve task-relevant meaning.",
  },
  {
    id: "knowledge-distillation",
    label: "Knowledge distillation",
    verb: "Distill",
    explanation:
      "Extract reusable lessons, rules, or procedures from prior experience.",
  },
  {
    id: "structured-construction",
    label: "Structured construction",
    verb: "Structure",
    explanation:
      "Build memory as graphs, tables, trees, schemas, or other relation-aware structures.",
  },
  {
    id: "latent-representation",
    label: "Latent representation",
    verb: "Encode",
    explanation:
      "Move experience into vectors, hidden states, embeddings, key-value states, or memory tokens.",
  },
  {
    id: "parametric-internalization",
    label: "Parametric internalization",
    verb: "Internalize",
    explanation:
      "Convert retrieved information into model competence through editing, adapters, fine-tuning, or similar parameter updates.",
  },
] as const satisfies readonly FormationOperation[];

export const evolutionOperations = [
  {
    id: "consolidation",
    label: "Consolidation",
    verb: "Merge",
    explanation:
      "Combine repeated or related entries into abstractions without losing the evidence needed for repair.",
  },
  {
    id: "updating",
    label: "Updating",
    verb: "Correct",
    explanation:
      "Revise stale, conflicting, or incomplete memories when new interaction changes what the agent should believe.",
  },
  {
    id: "forgetting",
    label: "Forgetting",
    verb: "Prune",
    explanation:
      "Remove, weaken, or archive outdated and low-value entries to control noise, cost, and risk.",
  },
] as const satisfies readonly EvolutionOperation[];

export const retrievalOperations = [
  {
    id: "task-start",
    label: "Task-start recall",
    timing: "Before acting",
    explanation:
      "Load durable context such as user preferences, project facts, or task policy before the first action.",
  },
  {
    id: "on-demand-semantic",
    label: "On-demand semantic retrieval",
    timing: "When uncertainty appears",
    explanation:
      "Build a context-aware query and retrieve relevant memories when the current step needs support.",
  },
  {
    id: "graph-traversal",
    label: "Graph traversal",
    timing: "Across related entities",
    explanation:
      "Follow links between people, documents, code objects, claims, or episodes instead of relying on similarity alone.",
  },
  {
    id: "working-context",
    label: "Working-context refresh",
    timing: "During the episode",
    explanation:
      "Keep the active scratchpad aligned by selecting, rewriting, or reformatting what should remain in context.",
  },
] as const satisfies readonly RetrievalOperation[];

export function getMemoryScenario(id: MemoryScenarioId): MemoryScenario {
  return getById(memoryScenarios, id);
}

export function getMemoryForm(id: MemoryFormId): MemoryForm {
  return getById(memoryForms, id);
}

export function getMemoryFunction(id: MemoryFunctionId): MemoryFunction {
  return getById(memoryFunctions, id);
}

export function getFormationOperation(id: FormationId): FormationOperation {
  return getById(formationOperations, id);
}

export function getRetrievalOperation(id: RetrievalId): RetrievalOperation {
  return getById(retrievalOperations, id);
}

export function getEvolutionOperation(id: EvolutionId): EvolutionOperation {
  return getById(evolutionOperations, id);
}

function getById<TItem extends { id: string }, TId extends TItem["id"]>(
  items: readonly TItem[],
  id: TId,
): Extract<TItem, { id: TId }> {
  const item = items.find((candidate) => candidate.id === id);
  if (!item) {
    throw new Error(`Unknown memory workbench item: ${id}`);
  }
  return item as Extract<TItem, { id: TId }>;
}

function contains<TValue extends string>(
  values: readonly TValue[],
  value: TValue,
): boolean {
  return values.includes(value);
}

export function getMemoryDesignEvaluation(
  input: MemoryDesignInput,
): MemoryDesignEvaluation {
  const scenario = getMemoryScenario(input.scenarioId);
  const form = getMemoryForm(input.formId);
  const memoryFunction = getMemoryFunction(input.functionId);
  const formation = getFormationOperation(input.formationId);
  const retrieval = getRetrievalOperation(input.retrievalId);
  const selectedEvolution = input.evolutionIds.map((id) =>
    getEvolutionOperation(id),
  );

  const strengths: string[] = [];
  const warnings: string[] = [];
  let score = 42;

  if (contains(scenario.preferredForms, input.formId)) {
    score += 16;
    strengths.push(`${form.shortLabel} fits this scenario's memory carrier.`);
  } else {
    warnings.push(
      `${form.shortLabel} is not the first carrier this scenario wants.`,
    );
  }

  if (contains(scenario.preferredFunctions, input.functionId)) {
    score += 14;
    strengths.push(
      `${memoryFunction.shortLabel} matches why this agent needs memory.`,
    );
  } else {
    warnings.push(
      `${memoryFunction.shortLabel} may solve a different problem than the scenario asks for.`,
    );
  }

  if (contains(scenario.preferredFormation, input.formationId)) {
    score += 10;
    strengths.push(`${formation.label} is a plausible formation step here.`);
  } else {
    warnings.push(`${formation.label} may create the wrong memory surface.`);
  }

  if (contains(scenario.preferredRetrieval, input.retrievalId)) {
    score += 10;
    strengths.push(
      `${retrieval.label} gives the agent memory at the right time.`,
    );
  } else {
    warnings.push(
      `${retrieval.label} may retrieve too early, too late, or with the wrong intent.`,
    );
  }

  const preferredEvolutionCount = input.evolutionIds.filter((id) =>
    contains(scenario.preferredEvolution, id),
  ).length;
  score += preferredEvolutionCount * 5;

  if (selectedEvolution.length === 0) {
    warnings.push(
      "No evolution operator is scheduled, so memory can become stale or noisy.",
    );
  } else {
    strengths.push(
      `${selectedEvolution.map((operation) => operation.label).join(", ")} keeps the store from being a static archive.`,
    );
  }

  if (
    input.formId === "internal-parametric" &&
    (input.functionId === "user-factual" ||
      input.functionId === "multi-turn-working")
  ) {
    score -= 18;
    warnings.push(
      "Internal weights are hard to audit or delete, so they are risky for volatile user facts or active workspace state.",
    );
  }

  if (
    form.family === "Latent" &&
    (input.functionId === "user-factual" ||
      input.functionId === "environment-factual")
  ) {
    score -= 10;
    warnings.push(
      "Latent memory is compact but weak for provenance-heavy factual memory unless paired with explicit records.",
    );
  }

  if (
    form.family === "Token-level" &&
    input.retrievalId === "working-context" &&
    input.functionId === "single-turn-working"
  ) {
    score += 6;
    strengths.push(
      "Explicit working notes are easy to rewrite when the current context changes.",
    );
  }

  if (
    input.formId === "external-parametric" &&
    input.functionId === "skill-experience"
  ) {
    score += 6;
    strengths.push(
      "Adapters or proxy modules can turn repeated skill evidence into modular competence.",
    );
  }

  if (
    !input.evolutionIds.includes("updating") &&
    (input.functionId === "user-factual" ||
      input.functionId === "environment-factual")
  ) {
    score -= 8;
    warnings.push(
      "Factual memory needs updating because user and environment facts can conflict or go stale.",
    );
  }

  if (
    !input.evolutionIds.includes("forgetting") &&
    (input.formId === "flat-token" || input.functionId.includes("working"))
  ) {
    score -= 6;
    warnings.push(
      "Without forgetting or pruning, flat and working memories tend to accumulate noise.",
    );
  }

  const boundedScore = Math.max(0, Math.min(100, score));
  const fitLabel =
    boundedScore >= 78
      ? "Strong survey-aligned design"
      : boundedScore >= 60
        ? "Usable with explicit tradeoffs"
        : "Needs redesign before deployment";

  return {
    score: boundedScore,
    fitLabel,
    diagnosis: buildDiagnosis({
      scenario,
      form,
      memoryFunction,
      formation,
      retrieval,
      selectedEvolution,
      score: boundedScore,
    }),
    strengths,
    warnings,
    metrics: {
      transparency: form.transparency,
      adaptability: Math.round(
        (form.updateEase + selectedEvolution.length) / 2,
      ),
      structure: form.reasoningStructure,
      contextEfficiency: form.contextEfficiency,
    },
  };
}

function buildDiagnosis({
  scenario,
  form,
  memoryFunction,
  formation,
  retrieval,
  selectedEvolution,
  score,
}: {
  scenario: MemoryScenario;
  form: MemoryForm;
  memoryFunction: MemoryFunction;
  formation: FormationOperation;
  retrieval: RetrievalOperation;
  selectedEvolution: readonly EvolutionOperation[];
  score: number;
}) {
  const evolutionText =
    selectedEvolution.length > 0
      ? selectedEvolution
          .map((operation) => operation.verb.toLowerCase())
          .join(", ")
      : "skip evolution";
  const verdict =
    score >= 78
      ? "The pieces reinforce each other."
      : score >= 60
        ? "The design can work, but the tradeoffs must be visible."
        : "The design mismatches the scenario's memory pressure.";

  return `${scenario.shortLabel}: ${formation.verb.toLowerCase()} experience into ${form.shortLabel.toLowerCase()} memory for ${memoryFunction.shortLabel.toLowerCase()}, ${evolutionText}, then retrieve by ${retrieval.label.toLowerCase()}. ${verdict}`;
}
