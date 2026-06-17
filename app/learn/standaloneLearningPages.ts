import type { SourceSeriesId } from "../../lib/quiz";

export type StandaloneLearningPage = {
  href: string;
  sequenceLabel: string;
  shortTitle: string;
  summary: string;
};

export const standaloneLearningPagesBySeries: Partial<
  Record<SourceSeriesId, readonly StandaloneLearningPage[]>
> = {
  "stanford-cme295": [
    {
      href: "/learn/stanford-cme295/lecture-6",
      sequenceLabel: "Lecture 6",
      shortTitle: "Reasoning Control Bench",
      summary:
        "Control thinking budgets, reasoning benchmarks, verifiable rewards, GRPO group advantages, length incentives, R1 recipes, and distillation.",
    },
    {
      href: "/learn/stanford-cme295/lecture-7",
      sequenceLabel: "Lecture 7",
      shortTitle: "RAG, Tools, Agents Studio",
      summary:
        "Route model requests through retrieval, tool calls, agent loops, tool-selection/MCP boundaries, and safety guardrails.",
    },
    {
      href: "/learn/stanford-cme295/lecture-8",
      sequenceLabel: "Lecture 8",
      shortTitle: "LLM Evaluation Studio",
      summary:
        "Build evaluation scopes, agreement math, reference metrics, judge controls, factuality scoring, agent diagnostics, and benchmark tradeoffs.",
    },
  ],
};

export function getStandaloneLearningPagesForSeries(
  seriesId: SourceSeriesId,
): readonly StandaloneLearningPage[] {
  return standaloneLearningPagesBySeries[seriesId] ?? [];
}

export function getStandaloneLearningPageCountForSeries(
  seriesId: SourceSeriesId,
): number {
  return getStandaloneLearningPagesForSeries(seriesId).length;
}
