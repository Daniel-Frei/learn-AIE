import { describe, expect, it } from "vitest";
import { QUESTION_SOURCES, type Question } from "@/lib/quiz";

const SOURCE_IDS_ENV = "QUESTION_GUESSABILITY_SOURCE_IDS";
const MIN_QUESTIONS_FOR_STABLE_SCORE = 12;
const MAX_CUE_BASELINE_OPTION_ACCURACY = 0.72;
const MAX_CUE_BASELINE_LIFT_OVER_MAJORITY = 0.1;
const MAX_FALSE_CUE_RATE = 0.5;
const MAX_CUE_RATE_GAP = 0.35;
const MAX_LONGEST_OPTION_CORRECT_RATE = 0.75;
const MAX_MIDDLE_CUE_BASELINE_OPTION_ACCURACY = 0.72;
const MAX_MIDDLE_CUE_BASELINE_LIFT_OVER_MAJORITY = 0.1;
const MAX_MIDDLE_CUE_RATE_GAP = 0.35;
const MIN_PROMPT_FRAME_QUESTIONS = 6;
const MAX_PROMPT_FRAME_MIDDLE_CUE_ACCURACY = 0.72;
const MAX_PROMPT_FRAME_MIDDLE_CUE_LIFT = 0.1;
const MIN_NEIGHBORHOOD_OUTLIER_QUESTIONS = 6;
const MAX_NEIGHBORHOOD_OUTLIER_INCORRECT_RATE = 0.85;

const LANGUAGE_CUE_PATTERN =
  /\b(?:always|never|only|every|all|none|cannot|can't|impossible|guarantee|guarantees|guaranteed|prove|proves|proof|complete|perfect|unnecessary|irrelevant)\b|safe by default|harmful by definition|no longer/i;
const MIDDLE_ANSWER_CUE_PATTERN =
  /\b(?:can help|can contribute|can support|can improve|may|might|could|often|typically|usually|depends|context|contextual|context-dependent|evidence|validated|validation|measure|measured|measurement|constraint|constrained|limited|limitation|trade-?off|tradeoffs|uncertain|uncertainty|partial|partly|not enough|not sufficient|does not automatically|doesn't automatically|still need|still needs|useful)\b/i;
const PROMPT_FRAME_CUE_PATTERN =
  /\bwhy\s+(?:can|could|might|does|do|is|are|would)\b.*\b(?:not|less|fail|fails|failure|limit|limited|challenge|need|require|context|setting|generalize|translate|sufficient|enough|still)\b|\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:main\s+)?(?:limitation|risk|challenge)\b/i;
const TOKEN_PATTERN = /[a-z][a-z0-9-]{2,}/g;
const STOP_WORDS = new Set([
  "about",
  "above",
  "after",
  "again",
  "also",
  "because",
  "before",
  "being",
  "between",
  "both",
  "could",
  "does",
  "each",
  "from",
  "have",
  "into",
  "more",
  "most",
  "only",
  "other",
  "same",
  "should",
  "such",
  "than",
  "that",
  "their",
  "them",
  "then",
  "there",
  "these",
  "they",
  "this",
  "through",
  "when",
  "where",
  "which",
  "while",
  "with",
  "would",
]);

type GuessabilityMetrics = {
  cueBaselineAccuracy: number;
  majorityBaselineAccuracy: number;
  cueBaselineLiftOverMajority: number;
  falseCueRate: number;
  trueCueRate: number;
  cueRateGap: number;
  middleCueBaselineAccuracy: number;
  middleCueBaselineLiftOverMajority: number;
  falseMiddleCueRate: number;
  trueMiddleCueRate: number;
  middleCueRateGap: number;
  longestOptionCorrectRate: number;
  longestOptionQuestions: number;
  promptFrameMiddleCueBaselineAccuracy: number;
  promptFrameMiddleCueBaselineLiftOverMajority: number;
  promptFrameQuestions: number;
  neighborhoodOutlierIncorrectRate: number;
  neighborhoodOutlierQuestions: number;
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

function hasMiddleAnswerCue(text: string) {
  return MIDDLE_ANSWER_CUE_PATTERN.test(text);
}

function hasPromptFrameCue(text: string) {
  return PROMPT_FRAME_CUE_PATTERN.test(text);
}

function safeRate(numerator: number, denominator: number) {
  return denominator === 0 ? 0 : numerator / denominator;
}

function getContentTokens(text: string) {
  return new Set(
    (text.toLowerCase().match(TOKEN_PATTERN) ?? []).filter(
      (token) => !STOP_WORDS.has(token),
    ),
  );
}

function countOverlap(tokens: Set<string>, comparison: Set<string>) {
  let overlap = 0;
  for (const token of tokens) {
    if (comparison.has(token)) overlap += 1;
  }
  return overlap;
}

function getNeighborhoodOutlierIndex(question: Question) {
  const promptTokens = getContentTokens(question.prompt);
  const optionTokens = question.options.map((option) =>
    getContentTokens(option.text),
  );
  const scores = optionTokens.map((tokens, index) => {
    const comparison = new Set(promptTokens);
    for (const [otherIndex, otherTokens] of optionTokens.entries()) {
      if (otherIndex === index) continue;
      for (const token of otherTokens) comparison.add(token);
    }

    return {
      index,
      score: safeRate(countOverlap(tokens, comparison), tokens.size),
    };
  });
  const sortedScores = [...scores].sort((a, b) => a.score - b.score);
  const [lowest, nextLowest] = sortedScores;

  if (!lowest || !nextLowest) return null;
  if (nextLowest.score - lowest.score < 0.16) return null;
  if (lowest.score > 0.45) return null;

  return lowest.index;
}

function scoreGuessability(questions: Question[]): GuessabilityMetrics {
  let optionCount = 0;
  let correctOptions = 0;
  let incorrectOptions = 0;
  let cueBaselineCorrect = 0;
  let middleCueBaselineCorrect = 0;
  let falseOptionsWithCues = 0;
  let trueOptionsWithCues = 0;
  let falseOptionsWithMiddleCues = 0;
  let trueOptionsWithMiddleCues = 0;
  let longestOptionQuestions = 0;
  let longestOptionCorrect = 0;
  let promptFrameQuestions = 0;
  let promptFrameOptionCount = 0;
  let promptFrameCorrectOptions = 0;
  let promptFrameIncorrectOptions = 0;
  let promptFrameMiddleCueBaselineCorrect = 0;
  let neighborhoodOutlierQuestions = 0;
  let neighborhoodOutlierIncorrect = 0;

  for (const question of questions) {
    const promptHasFrameCue = hasPromptFrameCue(question.prompt);
    const neighborhoodOutlierIndex = getNeighborhoodOutlierIndex(question);
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

    if (promptHasFrameCue) {
      promptFrameQuestions += 1;
    }

    if (neighborhoodOutlierIndex !== null) {
      neighborhoodOutlierQuestions += 1;
      if (!question.options[neighborhoodOutlierIndex].isCorrect) {
        neighborhoodOutlierIncorrect += 1;
      }
    }

    for (const option of question.options) {
      optionCount += 1;
      const isCorrect = option.isCorrect;
      const cuePredictsCorrect = !hasLanguageCue(option.text);
      const middleCuePredictsCorrect =
        hasMiddleAnswerCue(option.text) && !hasLanguageCue(option.text);

      if (cuePredictsCorrect === isCorrect) {
        cueBaselineCorrect += 1;
      }

      if (middleCuePredictsCorrect === isCorrect) {
        middleCueBaselineCorrect += 1;
      }

      if (promptHasFrameCue) {
        promptFrameOptionCount += 1;
        if (isCorrect) {
          promptFrameCorrectOptions += 1;
        } else {
          promptFrameIncorrectOptions += 1;
        }
        if (middleCuePredictsCorrect === isCorrect) {
          promptFrameMiddleCueBaselineCorrect += 1;
        }
      }

      if (isCorrect) {
        correctOptions += 1;
        if (hasLanguageCue(option.text)) trueOptionsWithCues += 1;
        if (hasMiddleAnswerCue(option.text)) trueOptionsWithMiddleCues += 1;
      } else {
        incorrectOptions += 1;
        if (hasLanguageCue(option.text)) falseOptionsWithCues += 1;
        if (hasMiddleAnswerCue(option.text)) falseOptionsWithMiddleCues += 1;
      }
    }
  }

  const cueBaselineAccuracy = safeRate(cueBaselineCorrect, optionCount);
  const middleCueBaselineAccuracy = safeRate(
    middleCueBaselineCorrect,
    optionCount,
  );
  const trueOptionRate = safeRate(correctOptions, optionCount);
  const falseOptionRate = safeRate(incorrectOptions, optionCount);
  const majorityBaselineAccuracy = Math.max(trueOptionRate, falseOptionRate);
  const promptFrameTrueOptionRate = safeRate(
    promptFrameCorrectOptions,
    promptFrameOptionCount,
  );
  const promptFrameFalseOptionRate = safeRate(
    promptFrameIncorrectOptions,
    promptFrameOptionCount,
  );
  const promptFrameMajorityBaselineAccuracy = Math.max(
    promptFrameTrueOptionRate,
    promptFrameFalseOptionRate,
  );
  const falseCueRate = safeRate(falseOptionsWithCues, incorrectOptions);
  const trueCueRate = safeRate(trueOptionsWithCues, correctOptions);
  const falseMiddleCueRate = safeRate(
    falseOptionsWithMiddleCues,
    incorrectOptions,
  );
  const trueMiddleCueRate = safeRate(trueOptionsWithMiddleCues, correctOptions);

  return {
    cueBaselineAccuracy,
    majorityBaselineAccuracy,
    cueBaselineLiftOverMajority: cueBaselineAccuracy - majorityBaselineAccuracy,
    falseCueRate,
    trueCueRate,
    cueRateGap: falseCueRate - trueCueRate,
    middleCueBaselineAccuracy,
    middleCueBaselineLiftOverMajority:
      middleCueBaselineAccuracy - majorityBaselineAccuracy,
    falseMiddleCueRate,
    trueMiddleCueRate,
    middleCueRateGap: trueMiddleCueRate - falseMiddleCueRate,
    longestOptionCorrectRate:
      longestOptionQuestions === 0
        ? 0
        : longestOptionCorrect / longestOptionQuestions,
    longestOptionQuestions,
    promptFrameMiddleCueBaselineAccuracy: safeRate(
      promptFrameMiddleCueBaselineCorrect,
      promptFrameOptionCount,
    ),
    promptFrameMiddleCueBaselineLiftOverMajority:
      safeRate(promptFrameMiddleCueBaselineCorrect, promptFrameOptionCount) -
      promptFrameMajorityBaselineAccuracy,
    promptFrameQuestions,
    neighborhoodOutlierIncorrectRate: safeRate(
      neighborhoodOutlierIncorrect,
      neighborhoodOutlierQuestions,
    ),
    neighborhoodOutlierQuestions,
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
    `middleCueAccuracy=${formatPercent(metrics.middleCueBaselineAccuracy)}`,
    `middleCueLift=${formatPercent(metrics.middleCueBaselineLiftOverMajority)}`,
    `middleCueGap=${formatPercent(metrics.middleCueRateGap)}`,
    `longestCorrect=${formatPercent(metrics.longestOptionCorrectRate)}`,
    `promptFrameQuestions=${metrics.promptFrameQuestions}`,
    `promptFrameMiddleCueAccuracy=${formatPercent(metrics.promptFrameMiddleCueBaselineAccuracy)}`,
    `neighborhoodOutlierIncorrect=${formatPercent(metrics.neighborhoodOutlierIncorrectRate)}`,
  ].join(", ");
}

function makeCueDrivenQuestion(index: number): Question {
  return {
    id: `synthetic-guessable-q${index}`,
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Why can a benchmark result fail to translate to deployment decisions?",
    options: [
      {
        text: "The benchmark can help compare deployment decisions, but it still needs validation against held-out cases.",
        isCorrect: true,
      },
      {
        text: "The benchmark guarantees every deployment decision because held-out cases prove complete operational readiness.",
        isCorrect: false,
      },
      {
        text: "It is the spreadsheet format used to store accounting rows.",
        isCorrect: false,
      },
      {
        text: "The benchmark result is useful only after the model architecture has no remaining monitoring needs.",
        isCorrect: false,
      },
    ],
    explanation:
      "This synthetic question is intentionally cue driven so the heuristic test has a stable fixture. Real questions should make the distractors more plausible and source-specific.",
  };
}

const selectedSources = getSelectedSources();
const runIfConfigured = selectedSources.length > 0 ? it : it.skip;

describe("question answer-option guessability", () => {
  it("scores a synthetic cue-driven set as guessable", () => {
    const metrics = scoreGuessability(
      Array.from({ length: MIN_QUESTIONS_FOR_STABLE_SCORE }, (_, index) =>
        makeCueDrivenQuestion(index + 1),
      ),
    );

    expect(metrics.middleCueBaselineAccuracy).toBeGreaterThan(
      MAX_MIDDLE_CUE_BASELINE_OPTION_ACCURACY,
    );
    expect(metrics.middleCueRateGap).toBeGreaterThan(MAX_MIDDLE_CUE_RATE_GAP);
    expect(metrics.promptFrameQuestions).toBe(MIN_QUESTIONS_FOR_STABLE_SCORE);
    expect(metrics.promptFrameMiddleCueBaselineAccuracy).toBeGreaterThan(
      MAX_PROMPT_FRAME_MIDDLE_CUE_ACCURACY,
    );
    expect(metrics.neighborhoodOutlierQuestions).toBeGreaterThanOrEqual(
      MIN_NEIGHBORHOOD_OUTLIER_QUESTIONS,
    );
    expect(metrics.neighborhoodOutlierIncorrectRate).toBe(1);
  });

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

        if (
          metrics.middleCueBaselineAccuracy >
            MAX_MIDDLE_CUE_BASELINE_OPTION_ACCURACY &&
          metrics.middleCueBaselineLiftOverMajority >
            MAX_MIDDLE_CUE_BASELINE_LIFT_OVER_MAJORITY
        ) {
          sourceFailures.push(
            `hedged-middle cue baseline is too accurate (${formatPercent(metrics.middleCueBaselineAccuracy)} with ${formatPercent(metrics.middleCueBaselineLiftOverMajority)} lift over majority baseline)`,
          );
        }

        if (
          metrics.trueMiddleCueRate > metrics.falseMiddleCueRate &&
          metrics.middleCueRateGap > MAX_MIDDLE_CUE_RATE_GAP
        ) {
          sourceFailures.push(
            `middle-answer cues are concentrated in correct options (${formatPercent(metrics.trueMiddleCueRate)} true vs ${formatPercent(metrics.falseMiddleCueRate)} false)`,
          );
        }

        if (
          metrics.promptFrameQuestions >= MIN_PROMPT_FRAME_QUESTIONS &&
          metrics.promptFrameMiddleCueBaselineAccuracy >
            MAX_PROMPT_FRAME_MIDDLE_CUE_ACCURACY &&
          metrics.promptFrameMiddleCueBaselineLiftOverMajority >
            MAX_PROMPT_FRAME_MIDDLE_CUE_LIFT
        ) {
          sourceFailures.push(
            `prompt-frame cues make the hedged-middle baseline too accurate (${formatPercent(metrics.promptFrameMiddleCueBaselineAccuracy)} across ${metrics.promptFrameQuestions} framed prompts)`,
          );
        }

        if (
          metrics.neighborhoodOutlierQuestions >=
            MIN_NEIGHBORHOOD_OUTLIER_QUESTIONS &&
          metrics.neighborhoodOutlierIncorrectRate >
            MAX_NEIGHBORHOOD_OUTLIER_INCORRECT_RATE
        ) {
          sourceFailures.push(
            `semantic-neighborhood outliers are incorrect too often (${formatPercent(metrics.neighborhoodOutlierIncorrectRate)} across ${metrics.neighborhoodOutlierQuestions} questions)`,
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
          "A failure is a signal to make prompts less frame-revealing and distractors more semantically close, nuanced, and less cue-driven.",
        ].join("\n"),
      ).toEqual([]);
    },
  );
});
