import type { LearningCourse } from "../../lib/learning";
import { SOURCE_SERIES, type SourceSeriesId } from "../../lib/quiz";

export type StandaloneLearningPage = {
  href: string;
  sequenceLabel: string;
  shortTitle: string;
  summary: string;
};

export const standaloneLearningPagesBySeries: Partial<
  Record<SourceSeriesId, readonly StandaloneLearningPage[]>
> = {
  "ai-agents": [
    {
      href: "/learn/ai-agents/ai-agents-agent-native-memory/presentation",
      sequenceLabel: "Presentation",
      shortTitle: "Agent-Native Memory Talk Deck",
      summary:
        "Scroll through a visual presenter page for the agent-native memory paper, built around modules, workloads, retrieval, updates, cost, and ablations.",
    },
  ],
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
  "stanford-cme296": [
    {
      href: "/learn/stanford-cme296/diffusiongemma/presentation",
      sequenceLabel: "Paper club",
      shortTitle: "DiffusionGemma Talk Deck",
      summary:
        "Present DiffusionGemma through an interactive token canvas, discrete flow intuition, its blockwise architecture, sampler, training recipe, results, and limitations.",
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

export function getStandaloneLearningCourse(
  seriesId: SourceSeriesId,
): LearningCourse | null {
  const pages = getStandaloneLearningPagesForSeries(seriesId);
  const series = SOURCE_SERIES.find((candidate) => candidate.id === seriesId);
  if (!series || pages.length === 0) return null;

  return {
    seriesId,
    label: series.label,
    experiences: [],
    totalDurationMinutes: 0,
  };
}

export function getStandaloneLearningCourses(): LearningCourse[] {
  return SOURCE_SERIES.flatMap((series) => {
    const course = getStandaloneLearningCourse(series.id);
    return course ? [course] : [];
  });
}
