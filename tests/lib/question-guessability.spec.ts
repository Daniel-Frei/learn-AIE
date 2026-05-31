import { describe, expect, it } from "vitest";
import { QUESTION_SOURCES, type Question } from "@/lib/quiz";

const SOURCE_IDS_ENV = "QUESTION_GUESSABILITY_SOURCE_IDS";
const MIN_QUESTIONS_FOR_STABLE_SCORE = 12;
const MAX_CUE_BASELINE_OPTION_ACCURACY = 0.72;
const MAX_CUE_BASELINE_LIFT_OVER_MAJORITY = 0.1;
const MAX_FALSE_CUE_RATE = 0.5;
const MAX_CUE_RATE_GAP = 0.35;
const MAX_LONGEST_OPTION_CORRECT_RATE = 0.75;

const LANGUAGE_CUE_PATTERN =
  /\b(?:always|never|only|every|all|none|cannot|can't|impossible|guarantee|guarantees|guaranteed|prove|proves|proof|complete|perfect|unnecessary|irrelevant)\b|safe by default|harmful by definition|no longer/i;

type GuessabilityMetrics = {
  cueBaselineAccuracy: number;
  majorityBaselineAccuracy: number;
  cueBaselineLiftOverMajority: number;
  falseCueRate: number;
  trueCueRate: number;
  cueRateGap: number;
  longestOptionCorrectRate: number;
  longestOptionQuestions: number;
};

function parseSelectedSourceIds() {
  const raw = process.env[SOURCE_IDS_ENV]?.trim();
  if (!raw) return [];

  return raw
    .split(/[,\s]+/)
    .map((sourceId) => sourceId.trim())
    .filter(Boolean);
}

function getSelectedSources() {
  const selectedSourceIds = parseSelectedSourceIds();
  if (selectedSourceIds.length === 0) return [];

  if (selectedSourceIds.includes("all")) return QUESTION_SOURCES;

  const sourceById = new Map<string, (typeof QUESTION_SOURCES)[number]>(
    QUESTION_SOURCES.map((source) => [source.id, source]),
  );
  const missingSourceIds = selectedSourceIds.filter(
    (sourceId) => !sourceById.has(sourceId),
  );

  if (missingSourceIds.length > 0) {
    throw new Error(
      `Unknown ${SOURCE_IDS_ENV}: ${missingSourceIds.join(", ")}`,
    );
  }

  return selectedSourceIds.map((sourceId) => sourceById.get(sourceId)!);
}

function hasLanguageCue(text: string) {
  return LANGUAGE_CUE_PATTERN.test(text);
}

function scoreGuessability(questions: Question[]): GuessabilityMetrics {
  let optionCount = 0;
  let correctOptions = 0;
  let incorrectOptions = 0;
  let cueBaselineCorrect = 0;
  let falseOptionsWithCues = 0;
  let trueOptionsWithCues = 0;
  let longestOptionQuestions = 0;
  let longestOptionCorrect = 0;

  for (const question of questions) {
    const optionLengths = question.options.map(
      (option) => option.text.trim().length,
    );
    const longestLength = Math.max(...optionLengths);
    const longestIndexes = optionLengths.flatMap((length, index) =>
      length === longestLength ? [index] : [],
    );

    if (longestIndexes.length === 1) {
      longestOptionQuestions += 1;
      if (question.options[longestIndexes[0]].isCorrect) {
        longestOptionCorrect += 1;
      }
    }

    for (const option of question.options) {
      optionCount += 1;
      const isCorrect = option.isCorrect;
      const cuePredictsCorrect = !hasLanguageCue(option.text);

      if (cuePredictsCorrect === isCorrect) {
        cueBaselineCorrect += 1;
      }

      if (isCorrect) {
        correctOptions += 1;
        if (hasLanguageCue(option.text)) trueOptionsWithCues += 1;
      } else {
        incorrectOptions += 1;
        if (hasLanguageCue(option.text)) falseOptionsWithCues += 1;
      }
    }
  }

  const cueBaselineAccuracy = cueBaselineCorrect / optionCount;
  const trueOptionRate = correctOptions / optionCount;
  const falseOptionRate = incorrectOptions / optionCount;
  const majorityBaselineAccuracy = Math.max(trueOptionRate, falseOptionRate);
  const falseCueRate = falseOptionsWithCues / incorrectOptions;
  const trueCueRate = trueOptionsWithCues / correctOptions;

  return {
    cueBaselineAccuracy,
    majorityBaselineAccuracy,
    cueBaselineLiftOverMajority: cueBaselineAccuracy - majorityBaselineAccuracy,
    falseCueRate,
    trueCueRate,
    cueRateGap: falseCueRate - trueCueRate,
    longestOptionCorrectRate:
      longestOptionQuestions === 0
        ? 0
        : longestOptionCorrect / longestOptionQuestions,
    longestOptionQuestions,
  };
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatMetrics(metrics: GuessabilityMetrics) {
  return [
    `cueAccuracy=${formatPercent(metrics.cueBaselineAccuracy)}`,
    `majorityBaseline=${formatPercent(metrics.majorityBaselineAccuracy)}`,
    `cueLift=${formatPercent(metrics.cueBaselineLiftOverMajority)}`,
    `falseCueRate=${formatPercent(metrics.falseCueRate)}`,
    `trueCueRate=${formatPercent(metrics.trueCueRate)}`,
    `cueGap=${formatPercent(metrics.cueRateGap)}`,
    `longestCorrect=${formatPercent(metrics.longestOptionCorrectRate)}`,
  ].join(", ");
}

const selectedSources = getSelectedSources();
const runIfConfigured = selectedSources.length > 0 ? it : it.skip;

describe("question answer-option guessability", () => {
  runIfConfigured(
    "keeps selected question sets resistant to simple language-cue heuristics",
    () => {
      const failures: string[] = [];

      for (const source of selectedSources) {
        if (source.questions.length < MIN_QUESTIONS_FOR_STABLE_SCORE) {
          continue;
        }

        const metrics = scoreGuessability(source.questions);
        const sourceFailures: string[] = [];

        if (
          metrics.cueBaselineAccuracy > MAX_CUE_BASELINE_OPTION_ACCURACY &&
          metrics.cueBaselineLiftOverMajority >
            MAX_CUE_BASELINE_LIFT_OVER_MAJORITY
        ) {
          sourceFailures.push(
            `language-cue baseline is too accurate (${formatPercent(metrics.cueBaselineAccuracy)} with ${formatPercent(metrics.cueBaselineLiftOverMajority)} lift over majority baseline)`,
          );
        }

        if (
          metrics.falseCueRate > MAX_FALSE_CUE_RATE &&
          metrics.cueRateGap > MAX_CUE_RATE_GAP
        ) {
          sourceFailures.push(
            `language cues are concentrated in false options (${formatPercent(metrics.falseCueRate)} false vs ${formatPercent(metrics.trueCueRate)} true)`,
          );
        }

        if (
          metrics.longestOptionQuestions >= MIN_QUESTIONS_FOR_STABLE_SCORE &&
          metrics.longestOptionCorrectRate > MAX_LONGEST_OPTION_CORRECT_RATE
        ) {
          sourceFailures.push(
            `the unique longest option is correct too often (${formatPercent(metrics.longestOptionCorrectRate)})`,
          );
        }

        if (sourceFailures.length > 0) {
          failures.push(
            `${source.id}: ${sourceFailures.join("; ")}. ${formatMetrics(metrics)}`,
          );
        }
      }

      expect(
        failures,
        [
          "Simple test-taking heuristics should not predict answer options well.",
          `Run with ${SOURCE_IDS_ENV}=source-id after creating or editing a question set.`,
          "A failure is a signal to make distractors more semantically close and less cue-driven.",
        ].join("\n"),
      ).toEqual([]);
    },
  );
});
