import { describe, expect, it } from "vitest";
import {
  defaultAtomMemPipelineInput,
  evaluateAtomMemPipeline,
  getQueryMode,
  getRepresentationMode,
} from "@/components/learning/pages/atommem/workbench";

describe("AtomMem learning workbench", () => {
  it("scores the default configuration as a strong AtomMem-style pipeline", () => {
    const evaluation = evaluateAtomMemPipeline(defaultAtomMemPipelineInput);

    expect(evaluation.label).toBe("Balanced AtomMem pipeline");
    expect(evaluation.score).toBeGreaterThanOrEqual(82);
    expect(evaluation.retrievedFactIds).toEqual(["f3", "f4", "f5"]);
    expect(evaluation.strengths).toEqual(
      expect.arrayContaining([
        "Atomic facts keep evidence compact and standalone.",
        "Residual storage and update tuples protect consistency.",
      ]),
    );
  });

  it("penalizes raw logs and append-only verification because they add noise and conflicts", () => {
    const evaluation = evaluateAtomMemPipeline({
      ...defaultAtomMemPipelineInput,
      representation: "raw-log",
      verification: "append-only",
      factCount: 40,
    });

    expect(evaluation.score).toBeLessThan(70);
    expect(evaluation.warnings).toEqual(
      expect.arrayContaining([
        "Raw logs retain evidence but create noise-heavy retrieval.",
        "Append-only storage allows duplicate and conflicting facts.",
        "Too many final facts raise recall while adding distractors.",
      ]),
    );
  });

  it("shows why multi-hop and temporal queries need graph recall plus profile history", () => {
    const withoutGraph = evaluateAtomMemPipeline({
      ...defaultAtomMemPipelineInput,
      query: "temporal",
      graphEnabled: false,
      structure: "facts-only",
    });

    expect(withoutGraph.score).toBeLessThan(
      evaluateAtomMemPipeline({
        ...defaultAtomMemPipelineInput,
        query: "temporal",
      }).score,
    );
    expect(withoutGraph.warnings).toEqual(
      expect.arrayContaining([
        "Graph recall is disabled, so remote evidence is isolated.",
        "Temporal preference questions need profile history.",
      ]),
    );
  });

  it("keeps the source-paper concepts addressable by id", () => {
    expect(getRepresentationMode("atomic-facts").caption).toContain(
      "standalone",
    );
    expect(getQueryMode("multi-hop").targetFactIds).toEqual(["f3", "f4", "f5"]);
  });
});
