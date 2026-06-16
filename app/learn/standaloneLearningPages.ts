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
