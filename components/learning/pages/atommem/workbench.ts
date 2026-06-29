export type RepresentationMode = "raw-log" | "summary" | "atomic-facts";
export type VerificationMode = "append-only" | "overwrite" | "residual-update";
export type StructureMode = "facts-only" | "events" | "events-profiles";
export type QueryMode = "single-hop" | "multi-hop" | "temporal";
export type SeedCount = 5 | 10 | 20;
export type FusionWeight = 0.5 | 0.7 | 0.9;
export type FactCount = 5 | 10 | 20 | 40;

export type AtomMemPipelineInput = {
  representation: RepresentationMode;
  verification: VerificationMode;
  structure: StructureMode;
  query: QueryMode;
  graphEnabled: boolean;
  seedCount: SeedCount;
  fusionWeight: FusionWeight;
  factCount: FactCount;
};

export type AtomMemMetricKey =
  | "evidence"
  | "stability"
  | "association"
  | "efficiency";

export type AtomMemPipelineEvaluation = {
  score: number;
  label: string;
  status: string;
  metrics: Record<AtomMemMetricKey, number>;
  retrievedFactIds: readonly string[];
  warnings: readonly string[];
  strengths: readonly string[];
  latencyMs: number;
};

export type AtomMemMemoryFact = {
  id: string;
  label: string;
  rawTurn: string;
  atomicText: string;
  keywords: readonly string[];
  time: string;
  eventId?: string;
  profileSignal?: string;
  source: "candidate" | "existing" | "profile";
};

export const atomMemMemoryFacts = [
  {
    id: "f1",
    label: "Exam result",
    rawTurn: 'Emma: "I got an A on my first psychology exam last Friday!"',
    atomicText:
      "Emma got an A on her first psychology exam on the Friday before May 17, 2023.",
    keywords: ["Emma", "psychology exam", "grade"],
    time: "May 12, 2023",
    eventId: "academic-progress",
    source: "candidate",
  },
  {
    id: "f2",
    label: "Quiet hotel preference",
    rawTurn:
      'Maya: "After the downtown hotel noise, I want quiet hotels for personal trips."',
    atomicText:
      "Maya prefers quiet hotels for personal travel after a noisy downtown hotel stay.",
    keywords: ["Maya", "quiet hotel", "personal travel"],
    time: "March 4, 2024",
    eventId: "travel-preferences",
    profileSignal: "Maya prefers quiet hotels for personal travel.",
    source: "existing",
  },
  {
    id: "f3",
    label: "Conference exception",
    rawTurn:
      'Maya: "For the Berlin conference, downtown is worth it because the shuttle was a mess."',
    atomicText:
      "Maya prefers downtown hotels for conference travel when transit logistics are unreliable.",
    keywords: ["Maya", "downtown hotel", "conference", "shuttle"],
    time: "June 11, 2024",
    eventId: "berlin-conference",
    profileSignal:
      "Maya's hotel preference depends on trip type and transit reliability.",
    source: "candidate",
  },
  {
    id: "f4",
    label: "Budget constraint",
    rawTurn:
      'Maya: "Keep the workshop hotel under 180 euros unless it saves a long commute."',
    atomicText:
      "Maya wants workshop hotels under 180 euros unless a higher price avoids a long commute.",
    keywords: ["Maya", "hotel budget", "commute", "workshop"],
    time: "June 12, 2024",
    eventId: "berlin-conference",
    profileSignal:
      "Maya accepts higher hotel cost when it materially reduces commute burden.",
    source: "candidate",
  },
  {
    id: "f5",
    label: "Old profile",
    rawTurn: "Profile memory from earlier sessions.",
    atomicText: "Maya prefers quiet hotels for personal travel.",
    keywords: ["Maya", "quiet hotel", "personal travel"],
    time: "valid from March 4, 2024",
    source: "profile",
  },
] as const satisfies readonly AtomMemMemoryFact[];

export const representationModes = [
  {
    id: "raw-log",
    label: "Raw log",
    caption: "High retention, high noise",
    description:
      "The whole conversation remains available, but retrieval has to sift through greetings, repeated details, and unresolved references.",
  },
  {
    id: "summary",
    label: "Loose summary",
    caption: "Compact, lossy",
    description:
      "A summary lowers token cost, but it can smooth away exact dates, exceptions, and the distinction between personal trips and conference travel.",
  },
  {
    id: "atomic-facts",
    label: "Atomic facts",
    caption: "Dense, standalone evidence",
    description:
      "Each stored unit resolves references, keeps useful metadata, and can be checked, linked, or retrieved without replaying every turn.",
  },
] as const satisfies readonly {
  id: RepresentationMode;
  label: string;
  caption: string;
  description: string;
}[];

export const verificationModes = [
  {
    id: "append-only",
    label: "Append only",
    description:
      "Keep every new fact as a separate record, even when it duplicates or conflicts with earlier memory.",
  },
  {
    id: "overwrite",
    label: "Overwrite closest",
    description:
      "Replace the closest older memory with the new text, risking lost provenance and history.",
  },
  {
    id: "residual-update",
    label: "Residual + updates",
    description:
      "Store only novel residual content and emit update tuples when a real conflict appears.",
  },
] as const satisfies readonly {
  id: VerificationMode;
  label: string;
  description: string;
}[];

export const structureModes = [
  {
    id: "facts-only",
    label: "Facts only",
    description:
      "Flat atomic facts stay precise, but remote context and stable user state remain hard to recover.",
  },
  {
    id: "events",
    label: "Events",
    description:
      "Related facts form event blocks with summaries, participants, keywords, and temporal spans.",
  },
  {
    id: "events-profiles",
    label: "Events + profiles",
    description:
      "Event memory preserves episodes while temporal profiles track stable attributes and historical versions.",
  },
] as const satisfies readonly {
  id: StructureMode;
  label: string;
  description: string;
}[];

export const queryModes = [
  {
    id: "single-hop",
    label: "Single-hop fact",
    question: "What grade did Emma get on her first psychology exam?",
    targetFactIds: ["f1"],
  },
  {
    id: "multi-hop",
    label: "Multi-hop episode",
    question:
      "Which hotel choice fits Maya's Berlin workshop constraints and why?",
    targetFactIds: ["f3", "f4", "f5"],
  },
  {
    id: "temporal",
    label: "Temporal preference",
    question:
      "Did Maya's hotel preference change, or is there a travel-type exception?",
    targetFactIds: ["f2", "f3", "f5"],
  },
] as const satisfies readonly {
  id: QueryMode;
  label: string;
  question: string;
  targetFactIds: readonly string[];
}[];

export const defaultAtomMemPipelineInput: AtomMemPipelineInput = {
  representation: "atomic-facts",
  verification: "residual-update",
  structure: "events-profiles",
  query: "multi-hop",
  graphEnabled: true,
  seedCount: 10,
  fusionWeight: 0.7,
  factCount: 10,
};

export function getRepresentationMode(id: RepresentationMode) {
  return getById(representationModes, id);
}

export function getVerificationMode(id: VerificationMode) {
  return getById(verificationModes, id);
}

export function getStructureMode(id: StructureMode) {
  return getById(structureModes, id);
}

export function getQueryMode(id: QueryMode) {
  return getById(queryModes, id);
}

function getById<TItem extends { id: string }, TId extends TItem["id"]>(
  items: readonly TItem[],
  id: TId,
): Extract<TItem, { id: TId }> {
  const item = items.find((candidate) => candidate.id === id);
  if (!item) throw new Error(`Unknown AtomMem workbench item: ${id}`);
  return item as Extract<TItem, { id: TId }>;
}

function factCountNoisePenalty(factCount: FactCount) {
  if (factCount <= 10) return 0;
  return factCount === 20 ? 7 : 18;
}

function seedNoisePenalty(seedCount: SeedCount) {
  if (seedCount === 10) return 0;
  return seedCount === 5 ? 6 : 12;
}

function fusionPenalty(fusionWeight: FusionWeight) {
  if (fusionWeight === 0.7) return 0;
  return fusionWeight === 0.5 ? 7 : 9;
}

function getRetrievedFactIds(input: AtomMemPipelineInput): readonly string[] {
  const query = getQueryMode(input.query);
  if (input.representation === "summary") {
    return query.targetFactIds.slice(0, input.query === "single-hop" ? 1 : 2);
  }

  if (input.representation === "raw-log") {
    return input.factCount >= 20
      ? ["f1", "f2", "f3", "f4", "f5"]
      : query.targetFactIds.slice(0, 1);
  }

  if (!input.graphEnabled && input.query !== "single-hop") {
    return query.targetFactIds.filter((id) => id !== "f5");
  }

  if (input.structure === "facts-only" && input.query !== "single-hop") {
    return query.targetFactIds.filter((id) => id !== "f5");
  }

  return query.targetFactIds;
}

export function evaluateAtomMemPipeline(
  input: AtomMemPipelineInput,
): AtomMemPipelineEvaluation {
  const warnings: string[] = [];
  const strengths: string[] = [];
  let evidence = 2;
  let stability = 2;
  let association = 2;
  let efficiency = 2;
  let score = 35;
  let latencyMs = 520;

  if (input.representation === "atomic-facts") {
    evidence += 3;
    efficiency += 2;
    score += 24;
    strengths.push("Atomic facts keep evidence compact and standalone.");
  } else if (input.representation === "raw-log") {
    evidence += 2;
    score += 8;
    latencyMs += 760;
    warnings.push("Raw logs retain evidence but create noise-heavy retrieval.");
  } else {
    efficiency += 2;
    score += 4;
    warnings.push(
      "Loose summaries are compact but can erase precise evidence.",
    );
  }

  if (input.verification === "residual-update") {
    stability += 3;
    score += 18;
    strengths.push("Residual storage and update tuples protect consistency.");
  } else if (input.verification === "append-only") {
    score -= 7;
    warnings.push(
      "Append-only storage allows duplicate and conflicting facts.",
    );
  } else {
    score -= 9;
    warnings.push(
      "Overwriting can destroy provenance and historical versions.",
    );
  }

  if (input.structure === "events-profiles") {
    association += 2;
    stability += 1;
    score += input.query === "single-hop" ? 6 : 15;
    strengths.push("Events and temporal profiles connect facts to context.");
  } else if (input.structure === "events") {
    association += 2;
    score += input.query === "single-hop" ? 4 : 10;
  } else if (input.query !== "single-hop") {
    warnings.push("Facts-only retrieval misses profile history and episodes.");
  }

  if (input.graphEnabled) {
    association += 2;
    latencyMs += 110;
    score += input.query === "single-hop" ? 2 : 13;
    strengths.push("Graph recall can bridge entity, event, and turn edges.");
  } else if (input.query !== "single-hop") {
    score -= 10;
    warnings.push("Graph recall is disabled, so remote evidence is isolated.");
  }

  const seedPenalty = seedNoisePenalty(input.seedCount);
  if (input.seedCount === 5) {
    warnings.push("Five seed facts under-activate the local graph.");
  } else if (input.seedCount === 20) {
    warnings.push("Twenty seed facts add irrelevant graph entry points.");
  } else {
    score += 6;
    strengths.push("Ten seed facts balance coverage and noise.");
  }
  score -= seedPenalty;

  const weightPenalty = fusionPenalty(input.fusionWeight);
  if (input.fusionWeight === 0.7) {
    score += 7;
    strengths.push(
      "The 0.7 event-fusion weight balances episode and fact fit.",
    );
  } else if (input.fusionWeight === 0.5) {
    warnings.push("Low event weight makes compensation too fact-local.");
  } else {
    warnings.push(
      "High event weight can retrieve loosely aligned event facts.",
    );
  }
  score -= weightPenalty;

  const noisePenalty = factCountNoisePenalty(input.factCount);
  if (input.factCount === 5) {
    warnings.push("Five final facts can miss evidence for reasoning tasks.");
  } else if (input.factCount === 10) {
    score += 7;
    strengths.push("Ten final facts match the paper's balanced main setting.");
  } else {
    warnings.push(
      "Too many final facts raise recall while adding distractors.",
    );
  }
  score -= noisePenalty;
  latencyMs += input.factCount * 16 + input.seedCount * 8;

  if (input.query === "single-hop" && input.factCount >= 20) {
    score -= 6;
    warnings.push("Single-hop lookup does not need a large retrieved context.");
  }

  if (input.query === "temporal" && input.structure !== "events-profiles") {
    score -= 8;
    warnings.push("Temporal preference questions need profile history.");
  }

  const retrievedFactIds = getRetrievedFactIds(input);
  const targetFactIds: readonly string[] = getQueryMode(
    input.query,
  ).targetFactIds;
  const recallShare =
    retrievedFactIds.filter((id) => targetFactIds.includes(id)).length /
    targetFactIds.length;
  evidence = Math.max(1, Math.min(5, Math.round(evidence * recallShare)));
  association = Math.max(1, Math.min(5, association));
  stability = Math.max(1, Math.min(5, stability));
  efficiency = Math.max(
    1,
    Math.min(5, efficiency - (input.factCount > 10 ? 1 : 0)),
  );

  const boundedScore = Math.max(0, Math.min(100, Math.round(score)));
  const label =
    boundedScore >= 82
      ? "Balanced AtomMem pipeline"
      : boundedScore >= 62
        ? "Useful with visible tradeoffs"
        : "Pipeline needs repair";

  const status =
    boundedScore >= 82
      ? "Strong AtomMem-style design: compact facts, controlled updates, contextual structure, and bounded graph recall reinforce each other."
      : boundedScore >= 62
        ? "The pipeline can answer, but one design choice is exposing noise, lost evidence, or unstable memory evolution."
        : "The pipeline is failing the paper's central design pressure: faithful evidence and stable retrieval are not aligned.";

  return {
    score: boundedScore,
    label,
    status,
    metrics: {
      evidence,
      stability,
      association,
      efficiency,
    },
    retrievedFactIds,
    warnings,
    strengths,
    latencyMs,
  };
}
