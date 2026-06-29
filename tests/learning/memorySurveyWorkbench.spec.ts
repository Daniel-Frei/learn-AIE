import { describe, expect, it } from "vitest";
import { getMemoryDesignEvaluation } from "@/components/learning/pages/memory-survey/workbench";

describe("memory survey workbench", () => {
  it("rewards a transparent factual-memory design for volatile user facts", () => {
    const evaluation = getMemoryDesignEvaluation({
      scenarioId: "personal-assistant",
      formId: "planar-token",
      functionId: "user-factual",
      formationId: "structured-construction",
      evolutionIds: ["updating", "forgetting"],
      retrievalId: "task-start",
    });

    expect(evaluation.fitLabel).toBe("Strong survey-aligned design");
    expect(evaluation.score).toBeGreaterThanOrEqual(90);
    expect(evaluation.metrics.transparency).toBe(5);
    expect(evaluation.warnings.join(" ")).not.toMatch(/hard to audit/i);
  });

  it("warns when internal parametric memory is used for deletable user facts", () => {
    const evaluation = getMemoryDesignEvaluation({
      scenarioId: "personal-assistant",
      formId: "internal-parametric",
      functionId: "user-factual",
      formationId: "structured-construction",
      evolutionIds: ["updating", "forgetting"],
      retrievalId: "task-start",
    });

    expect(evaluation.fitLabel).toBe("Usable with explicit tradeoffs");
    expect(evaluation.metrics.transparency).toBe(1);
    expect(evaluation.warnings.join(" ")).toMatch(/hard to audit or delete/i);
  });

  it("connects external parametric modules to skill-like experiential memory", () => {
    const evaluation = getMemoryDesignEvaluation({
      scenarioId: "coding-agent",
      formId: "external-parametric",
      functionId: "skill-experience",
      formationId: "knowledge-distillation",
      evolutionIds: ["consolidation", "updating", "forgetting"],
      retrievalId: "on-demand-semantic",
    });

    expect(evaluation.score).toBeGreaterThanOrEqual(90);
    expect(evaluation.strengths.join(" ")).toMatch(/modular competence/i);
    expect(evaluation.diagnosis).toMatch(/Coding agent/i);
  });
});
