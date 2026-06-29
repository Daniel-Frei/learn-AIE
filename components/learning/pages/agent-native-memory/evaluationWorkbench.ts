export type WorkloadId =
  | "cross-session"
  | "dialogue-exact"
  | "stateful-execution"
  | "fact-update"
  | "long-horizon"
  | "cost-frontier";

export type ArchitectureId =
  | "flat-similarity"
  | "relation-graph"
  | "hierarchical-tree"
  | "hybrid-filtered"
  | "trace-preserving"
  | "localized-light";

export type RepresentationVariantId =
  | "raw-retentive"
  | "light-compressed"
  | "abstractive-summary"
  | "deep-hierarchy";

export type ExtractionVariantId =
  | "coverage-first"
  | "fine-selective"
  | "structured-schema"
  | "raw-concat";

export type RetrievalVariantId =
  | "direct-top1"
  | "planned"
  | "planned-reflect"
  | "balanced-hybrid"
  | "sparse-leaning";

export type MaintenanceVariantId =
  | "conservative-merge"
  | "delayed-flush"
  | "timestamp-versions"
  | "global-rewrite";

export type Workload = {
  id: WorkloadId;
  label: string;
  shortLabel: string;
  benchmark: string;
  learnerTask: string;
  bottleneck: string;
  successMetric: string;
  priorities: {
    evidence: number;
    exactness: number;
    update: number;
    horizon: number;
    cost: number;
    trace: number;
  };
};

export type Architecture = {
  id: ArchitectureId;
  label: string;
  shortLabel: string;
  family: string;
  summary: string;
  bestFor: string;
  weakPoint: string;
  metrics: {
    evidence: number;
    exactness: number;
    update: number;
    horizon: number;
    cost: number;
    trace: number;
  };
};

export type RepresentationVariant = {
  id: RepresentationVariantId;
  label: string;
  lesson: string;
  evidenceDelta: number;
  exactnessDelta: number;
  horizonDelta: number;
  costDelta: number;
  warning?: string;
};

export type ExtractionVariant = {
  id: ExtractionVariantId;
  label: string;
  lesson: string;
  evidenceDelta: number;
  exactnessDelta: number;
  updateDelta: number;
  costDelta: number;
  warning?: string;
};

export type RetrievalVariant = {
  id: RetrievalVariantId;
  label: string;
  lesson: string;
  evidenceDelta: number;
  exactnessDelta: number;
  horizonDelta: number;
  costDelta: number;
  warning?: string;
};

export type MaintenanceVariant = {
  id: MaintenanceVariantId;
  label: string;
  lesson: string;
  updateDelta: number;
  horizonDelta: number;
  costDelta: number;
  warning?: string;
};

export type EvaluationInput = {
  workloadId: WorkloadId;
  architectureId: ArchitectureId;
  representationId: RepresentationVariantId;
  extractionId: ExtractionVariantId;
  retrievalId: RetrievalVariantId;
  maintenanceId: MaintenanceVariantId;
};

export type EvaluationResult = {
  score: number;
  verdict: string;
  headline: string;
  diagnosis: string;
  findings: readonly string[];
  warnings: readonly string[];
  metrics: {
    evidence: number;
    exactness: number;
    update: number;
    horizon: number;
    cost: number;
    trace: number;
  };
};

export const workloads = [
  {
    id: "cross-session",
    label: "Cross-session personal memory",
    shortLabel: "Cross-session",
    benchmark: "LongMemEval",
    learnerTask:
      "Reconnect facts scattered across older sessions and synthesize a semantically correct answer.",
    bottleneck:
      "Evidence is distributed, paraphrased, and not always recoverable by one lexical match.",
    successMetric: "ROUGE-L, substring Exact Match, and LLM judge accuracy.",
    priorities: {
      evidence: 5,
      exactness: 2,
      update: 3,
      horizon: 5,
      cost: 2,
      trace: 2,
    },
  },
  {
    id: "dialogue-exact",
    label: "Exact grounded dialogue recall",
    shortLabel: "Exact recall",
    benchmark: "LoCoMo",
    learnerTask:
      "Recover a short grounded fact such as a name, date, venue, preference, or attribute.",
    bottleneck:
      "The right memory is in a long but coherent dialogue, so filtering and exact grounding matter.",
    successMetric: "Exact Match and Answer F1.",
    priorities: {
      evidence: 4,
      exactness: 5,
      update: 3,
      horizon: 3,
      cost: 2,
      trace: 2,
    },
  },
  {
    id: "stateful-execution",
    label: "Stateful procedural execution",
    shortLabel: "DB state",
    benchmark: "DB-Bench",
    learnerTask:
      "Preserve operation order so dependent INSERT and UPDATE actions reach the correct final state.",
    bottleneck:
      "Correctness depends on trace order and executable task success, not only answer wording.",
    successMetric: "Executable task success rate plus Exact Match.",
    priorities: {
      evidence: 3,
      exactness: 3,
      update: 3,
      horizon: 2,
      cost: 2,
      trace: 5,
    },
  },
  {
    id: "fact-update",
    label: "Latest-state fact update",
    shortLabel: "Updates",
    benchmark: "LongMemEval and LoCoMo temporal slices",
    learnerTask:
      "Answer with the currently valid fact after old preferences, dates, or attributes have been corrected.",
    bottleneck:
      "The memory representation must bind later facts to earlier entities instead of appending stale mentions.",
    successMetric:
      "Knowledge-update, temporal-reasoning, and temporal Answer F1.",
    priorities: {
      evidence: 4,
      exactness: 4,
      update: 5,
      horizon: 4,
      cost: 2,
      trace: 2,
    },
  },
  {
    id: "long-horizon",
    label: "Long-horizon evidence distance",
    shortLabel: "Long horizon",
    benchmark: "LongBench, LongMemEval, and LoCoMo evidence-distance bins",
    learnerTask:
      "Stay stable as contexts grow, histories lengthen, and supporting evidence sits many sessions back.",
    bottleneck:
      "The system must choose abstractions that preserve entity, event, time, and session cues.",
    successMetric: "Accuracy, ROUGE-L F1, and Answer F1 over horizon bins.",
    priorities: {
      evidence: 5,
      exactness: 3,
      update: 3,
      horizon: 5,
      cost: 3,
      trace: 3,
    },
  },
  {
    id: "cost-frontier",
    label: "Utility-latency frontier",
    shortLabel: "Cost frontier",
    benchmark: "Unified time-overhead traces",
    learnerTask:
      "Keep useful memory under interactive construction and query costs.",
    bottleneck:
      "Whole-memory coordination, graph-wide consolidation, and multi-store sync can erase accuracy gains.",
    successMetric: "Normalized utility versus average operation latency.",
    priorities: {
      evidence: 3,
      exactness: 3,
      update: 3,
      horizon: 3,
      cost: 5,
      trace: 2,
    },
  },
] as const satisfies readonly Workload[];

export const architectures = [
  {
    id: "flat-similarity",
    label: "Flat semantic cache",
    shortLabel: "Flat cache",
    family: "Sequential context / embedding RAG",
    summary:
      "Stores chunks or facts as independent entries and retrieves by similarity.",
    bestFor: "Recent or short-range evidence with low engineering overhead.",
    weakPoint:
      "Scattered or temporally distant support is hard to assemble from one flat score.",
    metrics: {
      evidence: 2,
      exactness: 3,
      update: 2,
      horizon: 1,
      cost: 5,
      trace: 2,
    },
  },
  {
    id: "relation-graph",
    label: "Relation-aware graph memory",
    shortLabel: "Graph",
    family: "Structural topological",
    summary:
      "Stores entities, relations, and validity or temporal links that can be traversed.",
    bestFor: "Revised personal facts, dated events, and multi-hop evidence.",
    weakPoint:
      "Construction, disambiguation, traversal, and maintenance can become expensive.",
    metrics: {
      evidence: 5,
      exactness: 3,
      update: 5,
      horizon: 4,
      cost: 1,
      trace: 3,
    },
  },
  {
    id: "hierarchical-tree",
    label: "Hierarchical evidence tree",
    shortLabel: "Hierarchy",
    family: "Structural topological",
    summary:
      "Organizes raw traces, session summaries, and higher-level abstractions in layers.",
    bestFor:
      "Evidence completion where the system first locates a session or topic and then resolves a detail.",
    weakPoint:
      "Hierarchy cannot recover details removed by lossy summarization.",
    metrics: {
      evidence: 5,
      exactness: 3,
      update: 3,
      horizon: 5,
      cost: 2,
      trace: 3,
    },
  },
  {
    id: "hybrid-filtered",
    label: "Hybrid filtered memory",
    shortLabel: "Hybrid",
    family: "Multi-paradigm hybrid",
    summary:
      "Combines semantic, sparse, structured, and sometimes graph signals through planned or staged routing.",
    bestFor:
      "Exact current-state grounding when a query needs both predicates and semantic matching.",
    weakPoint:
      "More engines and routing stages can add latency and coordination cost.",
    metrics: {
      evidence: 4,
      exactness: 5,
      update: 4,
      horizon: 4,
      cost: 2,
      trace: 3,
    },
  },
  {
    id: "trace-preserving",
    label: "Trace-preserving context",
    shortLabel: "Trace",
    family: "Reference baseline / sequential context",
    summary:
      "Keeps raw operations or dialogue history visible so order and exact wording survive.",
    bestFor:
      "Stateful execution and time-dependent queries where operation order is the evidence.",
    weakPoint:
      "Long histories accumulate distractors and can exceed context or latency budgets.",
    metrics: {
      evidence: 4,
      exactness: 5,
      update: 3,
      horizon: 2,
      cost: 3,
      trace: 5,
    },
  },
  {
    id: "localized-light",
    label: "Localized lightweight memory",
    shortLabel: "Localized",
    family: "Efficient structured memory",
    summary:
      "Preserves high-retention local records and performs bounded update or retrieval instead of global refresh.",
    bestFor:
      "Cost-sensitive workloads where evidence should stay recoverable without whole-memory rewrites.",
    weakPoint:
      "It may need extra structure when evidence is deeply relational or conflict-heavy.",
    metrics: {
      evidence: 4,
      exactness: 4,
      update: 3,
      horizon: 3,
      cost: 5,
      trace: 4,
    },
  },
] as const satisfies readonly Architecture[];

export const representationVariants = [
  {
    id: "raw-retentive",
    label: "Raw high-retention records",
    lesson:
      "Preserve original conversational content when exact session-level details may matter later.",
    evidenceDelta: 1,
    exactnessDelta: 1,
    horizonDelta: 0,
    costDelta: -1,
  },
  {
    id: "light-compressed",
    label: "Light compression",
    lesson:
      "Remove filler while keeping phrasing and facts, which can preserve reasoning better than abstractive summaries.",
    evidenceDelta: 0,
    exactnessDelta: 0,
    horizonDelta: 0,
    costDelta: 1,
  },
  {
    id: "abstractive-summary",
    label: "Abstractive summary",
    lesson:
      "Summaries save space but can erase exact details and subtle temporal cues.",
    evidenceDelta: -1,
    exactnessDelta: -2,
    horizonDelta: -1,
    costDelta: 1,
    warning:
      "The component ablation found summary-heavy variants weaker for factual fidelity.",
  },
  {
    id: "deep-hierarchy",
    label: "Deeper hierarchy",
    lesson:
      "Hierarchy improves access paths, but it cannot restore facts lost by representation.",
    evidenceDelta: 1,
    exactnessDelta: 0,
    horizonDelta: 1,
    costDelta: -1,
  },
] as const satisfies readonly RepresentationVariant[];

export const extractionVariants = [
  {
    id: "coverage-first",
    label: "Coverage-first extraction",
    lesson:
      "Preserve context at write time, then filter later when the query is known.",
    evidenceDelta: 1,
    exactnessDelta: 1,
    updateDelta: 0,
    costDelta: 0,
  },
  {
    id: "fine-selective",
    label: "Fine selective extraction",
    lesson:
      "Extracting only high-confidence facts can help lexical retrieval while hurting compositional reasoning.",
    evidenceDelta: -1,
    exactnessDelta: -2,
    updateDelta: 0,
    costDelta: -1,
    warning:
      "The paper reports a sharp LoCoMo drop for overly fine memorization.",
  },
  {
    id: "structured-schema",
    label: "Structured schema extraction",
    lesson:
      "Typed facts, edges, or fields make updates and graph traversal possible.",
    evidenceDelta: 1,
    exactnessDelta: 0,
    updateDelta: 1,
    costDelta: -1,
  },
  {
    id: "raw-concat",
    label: "Raw sequence concatenation",
    lesson:
      "Minimal write overhead keeps traces intact, but retrieval has to do most of the later work.",
    evidenceDelta: 0,
    exactnessDelta: 1,
    updateDelta: -1,
    costDelta: 1,
  },
] as const satisfies readonly ExtractionVariant[];

export const retrievalVariants = [
  {
    id: "direct-top1",
    label: "Direct top-1 retrieval",
    lesson:
      "One early hit can be useful, but many memory tasks require evidence completion.",
    evidenceDelta: -2,
    exactnessDelta: 0,
    horizonDelta: -1,
    costDelta: 1,
    warning:
      "The retrieval study separates early localization from full evidence assembly.",
  },
  {
    id: "planned",
    label: "Lightweight planning",
    lesson:
      "Planning helps constrained memory lookup by decomposing intent before search.",
    evidenceDelta: 1,
    exactnessDelta: 1,
    horizonDelta: 1,
    costDelta: 0,
  },
  {
    id: "planned-reflect",
    label: "Planning plus reflection",
    lesson:
      "Extra deliberation after a route is specified can add overhead without improving retrieval.",
    evidenceDelta: 0,
    exactnessDelta: 0,
    horizonDelta: 0,
    costDelta: -1,
    warning:
      "The routing ablation found planning-only stronger than planning plus reflection.",
  },
  {
    id: "balanced-hybrid",
    label: "Balanced hybrid fusion",
    lesson:
      "Moderate dense-sparse-structured fusion helps when evidence is semantically related but lexically varied.",
    evidenceDelta: 2,
    exactnessDelta: 1,
    horizonDelta: 1,
    costDelta: -1,
  },
  {
    id: "sparse-leaning",
    label: "Sparse-leaning fusion",
    lesson:
      "More keyword weight can help exact terms, but it may miss semantically varied support.",
    evidenceDelta: 0,
    exactnessDelta: 1,
    horizonDelta: -1,
    costDelta: 0,
  },
] as const satisfies readonly RetrievalVariant[];

export const maintenanceVariants = [
  {
    id: "conservative-merge",
    label: "Conservative consolidation",
    lesson:
      "Selectively integrate related evidence without compressing away sparse cues.",
    updateDelta: 1,
    horizonDelta: 1,
    costDelta: 0,
  },
  {
    id: "delayed-flush",
    label: "Delayed flush",
    lesson:
      "Leaving recent evidence unresolved can improve surface coverage while hurting answerability.",
    updateDelta: -2,
    horizonDelta: -1,
    costDelta: 1,
    warning:
      "The maintenance ablation found delayed flushing weaker than immediate conservative merge.",
  },
  {
    id: "timestamp-versions",
    label: "Timestamped versions",
    lesson:
      "Validity metadata helps preserve history while routing to the current fact.",
    updateDelta: 2,
    horizonDelta: 1,
    costDelta: -1,
  },
  {
    id: "global-rewrite",
    label: "Global rewrite",
    lesson:
      "Whole-memory reorganization can improve structure but often dominates cost as history grows.",
    updateDelta: 0,
    horizonDelta: 1,
    costDelta: -2,
    warning:
      "The operational finding says maintenance scope, not structure alone, governs scaling.",
  },
] as const satisfies readonly MaintenanceVariant[];

export function getWorkload(id: WorkloadId): Workload {
  return getById(workloads, id);
}

export function getArchitecture(id: ArchitectureId): Architecture {
  return getById(architectures, id);
}

export function getRepresentationVariant(
  id: RepresentationVariantId,
): RepresentationVariant {
  return getById(representationVariants, id);
}

export function getExtractionVariant(
  id: ExtractionVariantId,
): ExtractionVariant {
  return getById(extractionVariants, id);
}

export function getRetrievalVariant(id: RetrievalVariantId): RetrievalVariant {
  return getById(retrievalVariants, id);
}

export function getMaintenanceVariant(
  id: MaintenanceVariantId,
): MaintenanceVariant {
  return getById(maintenanceVariants, id);
}

function getById<TItem extends { id: string }, TId extends TItem["id"]>(
  items: readonly TItem[],
  id: TId,
): Extract<TItem, { id: TId }> {
  const item = items.find((candidate) => candidate.id === id);
  if (!item) throw new Error(`Unknown agent-native memory item: ${id}`);
  return item as Extract<TItem, { id: TId }>;
}

function clampMetric(value: number): number {
  return Math.max(1, Math.min(5, Math.round(value)));
}

function weightedFit(
  metrics: EvaluationResult["metrics"],
  priorities: Workload["priorities"],
): number {
  const weighted =
    metrics.evidence * priorities.evidence +
    metrics.exactness * priorities.exactness +
    metrics.update * priorities.update +
    metrics.horizon * priorities.horizon +
    metrics.cost * priorities.cost +
    metrics.trace * priorities.trace;
  const max =
    5 *
    (priorities.evidence +
      priorities.exactness +
      priorities.update +
      priorities.horizon +
      priorities.cost +
      priorities.trace);
  return Math.round((weighted / max) * 100);
}

export function getEvaluationResult(input: EvaluationInput): EvaluationResult {
  const workload = getWorkload(input.workloadId);
  const architecture = getArchitecture(input.architectureId);
  const representation = getRepresentationVariant(input.representationId);
  const extraction = getExtractionVariant(input.extractionId);
  const retrieval = getRetrievalVariant(input.retrievalId);
  const maintenance = getMaintenanceVariant(input.maintenanceId);

  const metrics = {
    evidence: clampMetric(
      architecture.metrics.evidence +
        representation.evidenceDelta +
        extraction.evidenceDelta +
        retrieval.evidenceDelta,
    ),
    exactness: clampMetric(
      architecture.metrics.exactness +
        representation.exactnessDelta +
        extraction.exactnessDelta +
        retrieval.exactnessDelta,
    ),
    update: clampMetric(
      architecture.metrics.update +
        extraction.updateDelta +
        maintenance.updateDelta,
    ),
    horizon: clampMetric(
      architecture.metrics.horizon +
        representation.horizonDelta +
        retrieval.horizonDelta +
        maintenance.horizonDelta,
    ),
    cost: clampMetric(
      architecture.metrics.cost +
        representation.costDelta +
        extraction.costDelta +
        retrieval.costDelta +
        maintenance.costDelta,
    ),
    trace: clampMetric(architecture.metrics.trace),
  };

  let score = weightedFit(metrics, workload.priorities);
  const findings: string[] = [];
  const warnings: string[] = [];

  if (
    input.workloadId === "cross-session" &&
    ["relation-graph", "hierarchical-tree", "hybrid-filtered"].includes(
      input.architectureId,
    )
  ) {
    score += 6;
    findings.push(
      "Workload-aligned memory: relation, hierarchy, or hybrid routing helps dispersed cross-session evidence.",
    );
  }

  if (
    input.workloadId === "dialogue-exact" &&
    input.architectureId === "hybrid-filtered"
  ) {
    score += 7;
    findings.push(
      "Hybrid filtering matches exact grounded dialogue recall better than a single universal memory form.",
    );
  }

  if (
    input.workloadId === "stateful-execution" &&
    input.architectureId === "trace-preserving"
  ) {
    score += 8;
    findings.push(
      "Trace preservation matters when correctness depends on ordered operations and executable success.",
    );
  }

  if (
    input.workloadId === "fact-update" &&
    ["relation-graph", "hybrid-filtered"].includes(input.architectureId) &&
    ["timestamp-versions", "conservative-merge"].includes(input.maintenanceId)
  ) {
    score += 8;
    findings.push(
      "Temporal update fidelity is a pipeline property: representation and maintenance must separate stale from current facts.",
    );
  }

  if (
    input.workloadId === "long-horizon" &&
    ["relation-graph", "hierarchical-tree"].includes(input.architectureId)
  ) {
    score += 6;
    findings.push(
      "Horizon-structured memory preserves distant entity, event, time, or session cues better than flat retrieval.",
    );
  }

  if (
    input.workloadId === "cost-frontier" &&
    input.architectureId === "localized-light"
  ) {
    score += 8;
    findings.push(
      "Localized maintenance usually gives the strongest utility-latency balance.",
    );
  }

  if (
    input.representationId === "raw-retentive" &&
    (input.workloadId === "dialogue-exact" ||
      input.workloadId === "stateful-execution")
  ) {
    findings.push(
      "Representation granularity: raw retention protects exact details and ordered traces.",
    );
  }

  if (input.extractionId === "coverage-first") {
    findings.push(
      "Late filtering principle: preserve write-time context, then filter at query time.",
    );
  }

  if (input.retrievalId === "balanced-hybrid") {
    findings.push(
      "Retrieval guidance: balanced fusion handles semantically related but lexically varied evidence.",
    );
  }

  if (input.maintenanceId === "conservative-merge") {
    findings.push(
      "Maintenance principle: conservative consolidation preserves cross-turn linkages without hiding sparse cues.",
    );
  }

  for (const candidate of [
    representation.warning,
    extraction.warning,
    retrieval.warning,
    maintenance.warning,
  ]) {
    if (candidate) warnings.push(candidate);
  }

  if (metrics.evidence < 3 && workload.priorities.evidence >= 4) {
    warnings.push(
      "This workload needs evidence assembly, but the selected design leaves evidence weak.",
    );
  }

  if (metrics.update < 3 && workload.priorities.update >= 4) {
    warnings.push(
      "This workload needs update fidelity, but stale facts may remain unresolved.",
    );
  }

  if (metrics.cost < 3 && workload.priorities.cost >= 4) {
    warnings.push(
      "This workload is cost-sensitive, but the design performs too much broad coordination.",
    );
  }

  score = Math.max(0, Math.min(100, score));
  const verdict =
    score >= 82
      ? "Strong workload fit"
      : score >= 65
        ? "Viable with visible tradeoffs"
        : "Mismatch to diagnose";

  return {
    score,
    verdict,
    headline: `${architecture.shortLabel} on ${workload.shortLabel}`,
    diagnosis: `${architecture.label} uses ${representation.label.toLowerCase()}, ${extraction.label.toLowerCase()}, ${retrieval.label.toLowerCase()}, and ${maintenance.label.toLowerCase()} for ${workload.benchmark}. ${buildPlainDiagnosis(score, workload, architecture)}`,
    findings,
    warnings,
    metrics,
  };
}

function buildPlainDiagnosis(
  score: number,
  workload: Workload,
  architecture: Architecture,
) {
  if (score >= 82) {
    return `The design directly addresses the bottleneck: ${workload.bottleneck}`;
  }

  if (score >= 65) {
    return `The design can work, but the weak point must be managed: ${architecture.weakPoint}`;
  }

  return `The design is solving the wrong pressure. Start from the workload bottleneck: ${workload.bottleneck}`;
}

export function getEvidenceDistanceProjection({
  architectureId,
  retrievalId,
}: {
  architectureId: ArchitectureId;
  retrievalId: RetrievalVariantId;
}): readonly { bin: string; recall: number; note: string }[] {
  const architecture = getArchitecture(architectureId);
  const base =
    architecture.metrics.horizon * 11 + architecture.metrics.evidence * 7;
  const retrievalSupport =
    retrievalId === "balanced-hybrid"
      ? 13
      : retrievalId === "planned"
        ? 9
        : retrievalId === "direct-top1"
          ? -12
          : retrievalId === "sparse-leaning"
            ? -2
            : 1;
  const bins = [
    { bin: "1-5", penalty: 0, note: "near evidence" },
    { bin: "6-10", penalty: 8, note: "short-range drift" },
    { bin: "11-15", penalty: 15, note: "session gap" },
    { bin: "16-20", penalty: 22, note: "distant support" },
    { bin: "21-25", penalty: 29, note: "old scattered support" },
    { bin: "26-31", penalty: 36, note: "long-range reconstruction" },
  ];

  return bins.map((item) => ({
    bin: item.bin,
    recall: Math.max(8, Math.min(90, base + retrievalSupport - item.penalty)),
    note: item.note,
  }));
}
