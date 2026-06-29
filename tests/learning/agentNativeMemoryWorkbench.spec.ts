import { describe, expect, it } from "vitest";
import {
  getEvaluationResult,
  getEvidenceDistanceProjection,
} from "../../components/learning/pages/agent-native-memory/evaluationWorkbench";

describe("agent-native memory evaluation workbench", () => {
  it("rewards a hierarchy that preserves and assembles cross-session evidence", () => {
    const result = getEvaluationResult({
      workloadId: "cross-session",
      architectureId: "hierarchical-tree",
      representationId: "raw-retentive",
      extractionId: "coverage-first",
      retrievalId: "balanced-hybrid",
      maintenanceId: "conservative-merge",
    });

    expect(result.score).toBeGreaterThanOrEqual(82);
    expect(result.verdict).toBe("Strong workload fit");
    expect(result.metrics.evidence).toBeGreaterThanOrEqual(4);
    expect(result.metrics.horizon).toBeGreaterThanOrEqual(4);
    expect(result.findings.join(" ")).toMatch(/Late filtering|evidence/i);
  });

  it("surfaces stale-state and lossy-summary risks for fact-update workloads", () => {
    const result = getEvaluationResult({
      workloadId: "fact-update",
      architectureId: "flat-similarity",
      representationId: "abstractive-summary",
      extractionId: "fine-selective",
      retrievalId: "direct-top1",
      maintenanceId: "delayed-flush",
    });

    expect(result.score).toBeLessThan(65);
    expect(result.verdict).toBe("Mismatch to diagnose");
    expect(result.warnings.join(" ")).toMatch(
      /summary|fine|top-1|delayed|stale/i,
    );
  });

  it("recognizes localized maintenance for utility-latency pressure", () => {
    const result = getEvaluationResult({
      workloadId: "cost-frontier",
      architectureId: "localized-light",
      representationId: "light-compressed",
      extractionId: "coverage-first",
      retrievalId: "planned",
      maintenanceId: "conservative-merge",
    });

    expect(result.score).toBeGreaterThanOrEqual(82);
    expect(result.verdict).toBe("Strong workload fit");
    expect(result.findings.join(" ")).toContain("Localized maintenance");
  });

  it("projects stronger distant recall for hierarchical hybrid assembly", () => {
    const flatDirect = getEvidenceDistanceProjection({
      architectureId: "flat-similarity",
      retrievalId: "direct-top1",
    });
    const hierarchyHybrid = getEvidenceDistanceProjection({
      architectureId: "hierarchical-tree",
      retrievalId: "balanced-hybrid",
    });

    expect(lastRecall(hierarchyHybrid)).toBeGreaterThan(lastRecall(flatDirect));
    expect(hierarchyHybrid.at(-1)?.note).toBe("long-range reconstruction");
  });
});

function lastRecall(
  projection: ReturnType<typeof getEvidenceDistanceProjection>,
) {
  const finalBin = projection.at(-1);
  expect(finalBin).toBeDefined();
  return finalBin?.recall ?? 0;
}
