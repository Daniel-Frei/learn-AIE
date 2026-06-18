import { describe, expect, it } from "vitest";
import {
  closingThemes,
  courseTraceStages,
  getClosingTheme,
  getCourseTraceStage,
  getGenerationPassComparison,
  getPatchTokenCount,
  getRecapUnit,
  getVlmPattern,
  recapUnits,
  vlmPatterns,
} from "@/components/learning/pages/cme295-lecture9/synthesisLogic";

describe("CME295 Lecture 9 synthesis helpers", () => {
  it("covers the recap as source mechanisms from representation through evaluation", () => {
    expect(recapUnits.map((unit) => unit.id)).toEqual([
      "representation",
      "attention",
      "families",
      "runtime",
      "training",
      "preference",
      "reasoning",
      "systems",
      "evaluation",
    ]);

    expect(getRecapUnit("representation").mechanism).toMatch(/subword pieces/i);
    expect(getRecapUnit("representation").coreIdea).toMatch(
      /sequence of token ids/i,
    );
    expect(
      getRecapUnit("representation").terms.map((term) => term.label),
    ).toEqual(["Token", "Embedding", "Context awareness"]);
    expect(getRecapUnit("attention").formula).toMatch(/QK\^\\top/);
    expect(getRecapUnit("attention").terms.map((term) => term.label)).toEqual([
      "Query",
      "Key",
      "Value",
    ]);
    expect(getRecapUnit("preference").sourceTrace).toMatch(/Bradley-Terry/i);
    expect(
      getRecapUnit("preference").terms.map((term) => term.label),
    ).toContain("Reference pressure");
    expect(
      recapUnits.every(
        (unit) => unit.coreIdea.length > 80 && unit.terms.length >= 3,
      ),
    ).toBe(true);
    expect(getRecapUnit("systems").steps.join(" ")).toMatch(
      /Retrieval failure, reranking failure, and answer synthesis failure/i,
    );
    expect(getRecapUnit("evaluation").sourceTrace).toMatch(/MMLU.*AIME/i);
  });

  it("traces one answer across the course layers", () => {
    expect(courseTraceStages.map((stage) => stage.id)).toEqual([
      "prompt",
      "represent",
      "attend",
      "generate",
      "shape",
      "ground",
      "measure",
    ]);

    expect(getCourseTraceStage("shape").recapIds).toEqual([
      "training",
      "preference",
      "reasoning",
    ]);
    expect(getCourseTraceStage("ground").operation).toMatch(/RAG, tools/i);
  });

  it("computes ViT patch-token counts from image and patch dimensions", () => {
    expect(
      getPatchTokenCount({
        imageWidth: 224,
        imageHeight: 224,
        patchSize: 16,
      }),
    ).toBe(196);
    expect(
      getPatchTokenCount({
        imageWidth: 225,
        imageHeight: 224,
        patchSize: 16,
      }),
    ).toBe(210);
    expect(
      getPatchTokenCount({
        imageWidth: 224,
        imageHeight: 224,
        patchSize: 0,
      }),
    ).toBe(0);
  });

  it("distinguishes the VLM wiring patterns from the slides", () => {
    expect(vlmPatterns.map((pattern) => pattern.id)).toEqual([
      "visualPrefix",
      "crossAttention",
    ]);
    expect(getVlmPattern("visualPrefix").mechanism).toMatch(
      /concatenated with text tokens/i,
    );
    expect(getVlmPattern("crossAttention").mechanism).toMatch(
      /separate set of visual encoder features/i,
    );
  });

  it("compares serial autoregressive passes with toy masked-diffusion passes", () => {
    expect(
      getGenerationPassComparison({
        outputTokens: 24,
        maskedTokensPerPass: 6,
      }),
    ).toEqual({
      autoregressivePasses: 24,
      maskedDiffusionPasses: 4,
      speedupRatio: 6,
    });
    expect(
      getGenerationPassComparison({
        outputTokens: 0,
        maskedTokensPerPass: 6,
      }),
    ).toEqual({
      autoregressivePasses: 0,
      maskedDiffusionPasses: 0,
      speedupRatio: 0,
    });
  });

  it("keeps closing themes source-aligned without model-comparison options", () => {
    expect(closingThemes.map((theme) => theme.id)).toEqual([
      "architecture",
      "data",
      "serving",
      "hardware",
      "limits",
    ]);
    expect(getClosingTheme("architecture").details.join(" ")).toMatch(
      /AdamW.*Muon/i,
    );
    expect(getClosingTheme("data").details.join(" ")).toMatch(
      /Model collapse/i,
    );
    expect(getClosingTheme("hardware").details.join(" ")).toMatch(
      /key\/value reads and writes/i,
    );
  });
});
