import { describe, expect, it } from "vitest";
import { getQuestionType, QUESTION_SOURCES, type Question } from "@/lib/quiz";

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
const MIN_MATH_SALIENCE_SCORE = 8;
const MIN_UNIQUE_MATH_SALIENCE_GAP = 8;
const MIN_MATH_CUE_OPTIONS = 16;
const MAX_MATH_CUE_BASELINE_OPTION_ACCURACY = 0.72;
const MAX_MATH_CUE_BASELINE_LIFT_OVER_MAJORITY = 0.1;
const MAX_MATH_CUE_RATE_GAP = 0.35;
const MIN_UNIQUE_MATH_OUTLIER_QUESTIONS = 6;
const MAX_UNIQUE_MATH_OUTLIER_CORRECT_RATE = 0.75;

const LANGUAGE_CUE_PATTERN =
  /\b(?:always|never|only|every|all|none|cannot|can't|impossible|guarantee|guarantees|guaranteed|prove|proves|proof|complete|perfect|unnecessary|irrelevant)\b|safe by default|harmful by definition|no longer/i;
const MIDDLE_ANSWER_CUE_PATTERN =
  /\b(?:can help|can contribute|can support|can improve|may|might|could|often|typically|usually|depends|context|contextual|context-dependent|evidence|validated|validation|measure|measured|measurement|constraint|constrained|limited|limitation|trade-?off|tradeoffs|uncertain|uncertainty|partial|partly|not enough|not sufficient|does not automatically|doesn't automatically|still need|still needs|useful)\b/i;
const PROMPT_FRAME_CUE_PATTERN =
  /\bwhy\s+(?:can|could|might|does|do|is|are|would)\b.*\b(?:not|less|fail|fails|failure|limit|limited|challenge|need|require|context|setting|generalize|translate|sufficient|enough|still)\b|\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:main\s+)?(?:limitation|risk|challenge)\b/i;
const MATH_DELIMITER_PATTERN = /\\\(([\s\S]*?)\\\)|\\\[([\s\S]*?)\\\]/g;
const LATEX_COMMAND_PATTERN =
  /\\(?:frac|sum|prod|sqrt|theta|pi|alpha|beta|gamma|delta|lambda|mu|sigma|softmax|log|ln|exp|argmax|argmin|mathbb|cdot|times|leq|geq|neq|approx|infty|mid|left|right|operatorname|text)\b/g;
const MATH_SYMBOL_PATTERN = new RegExp(
  "[=<>+*/^_\\u00d7\\u00f7\\u00b7\\u221a\\u2211\\u220f\\u222b\\u2248\\u2264\\u2265\\u2260\\u221d\\u221e\\u2208\\u2209\\u2282\\u2286\\u222a\\u2229\\u2200\\u2203|]",
  "g",
);
const MATH_SCRIPT_PATTERN = new RegExp(
  "[\\u2080-\\u2089\\u2090\\u2091\\u2095\\u1d62\\u2c7c\\u2096-\\u209c\\u1d63\\u1d64\\u1d65\\u2093\\u2070\\u00b9\\u00b2\\u00b3\\u2074-\\u2079\\u1d40]",
  "g",
);
const GREEK_LETTER_PATTERN = /[\u0370-\u03ff]/g;
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
  mathCueBaselineAccuracy: number;
  mathCueBaselineLiftOverMajority: number;
  trueMathCueRate: number;
  falseMathCueRate: number;
  mathCueRateGap: number;
  mathCueQuestions: number;
  mathCueOptions: number;
  uniqueMathOutlierCorrectRate: number;
  uniqueMathOutlierQuestions: number;
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

function countPatternMatches(text: string, pattern: RegExp) {
  return text.match(pattern)?.length ?? 0;
}

function getDelimitedMathContentLength(text: string) {
  let contentLength = 0;

  for (const match of text.matchAll(MATH_DELIMITER_PATTERN)) {
    contentLength += (match[1] ?? match[2] ?? "").replace(/\s+/g, "").length;
  }

  return contentLength;
}

function getMathSalienceScore(text: string) {
  const delimitedMathLength = getDelimitedMathContentLength(text);

  return (
    Math.ceil(delimitedMathLength / 3) +
    countPatternMatches(text, MATH_DELIMITER_PATTERN) * 8 +
    countPatternMatches(text, LATEX_COMMAND_PATTERN) * 6 +
    countPatternMatches(text, MATH_SYMBOL_PATTERN) * 2 +
    countPatternMatches(text, MATH_SCRIPT_PATTERN) * 2 +
    countPatternMatches(text, GREEK_LETTER_PATTERN) * 3
  );
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

function getUniqueMathSalienceOutlierIndex(question: Question) {
  if (getQuestionType(question) === "assertion-reason") return null;

  const scores = question.options.map((option, index) => ({
    index,
    score: getMathSalienceScore(option.text),
  }));
  const sortedScores = [...scores].sort((a, b) => b.score - a.score);
  const [highest, nextHighest] = sortedScores;

  if (!highest || !nextHighest) return null;
  if (highest.score < MIN_MATH_SALIENCE_SCORE) return null;
  if (highest.score - nextHighest.score < MIN_UNIQUE_MATH_SALIENCE_GAP) {
    return null;
  }
  if (nextHighest.score > 0 && highest.score / nextHighest.score < 2) {
    return null;
  }

  return highest.index;
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
  let mathCueQuestions = 0;
  let mathCueOptionCount = 0;
  let mathCueCorrectOptions = 0;
  let mathCueIncorrectOptions = 0;
  let mathCueBaselineCorrect = 0;
  let trueOptionsWithMathCues = 0;
  let falseOptionsWithMathCues = 0;
  let uniqueMathOutlierQuestions = 0;
  let uniqueMathOutlierCorrect = 0;

  for (const question of questions) {
    const promptHasFrameCue = hasPromptFrameCue(question.prompt);
    const neighborhoodOutlierIndex = getNeighborhoodOutlierIndex(question);
    const uniqueMathOutlierIndex = getUniqueMathSalienceOutlierIndex(question);
    const mathSalienceFlags = question.options.map(
      (option) => getMathSalienceScore(option.text) >= MIN_MATH_SALIENCE_SCORE,
    );
    const shouldScoreMathCue =
      getQuestionType(question) !== "assertion-reason" &&
      mathSalienceFlags.some(Boolean) &&
      mathSalienceFlags.some((hasCue) => !hasCue);
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

    if (uniqueMathOutlierIndex !== null) {
      uniqueMathOutlierQuestions += 1;
      if (question.options[uniqueMathOutlierIndex].isCorrect) {
        uniqueMathOutlierCorrect += 1;
      }
    }

    if (shouldScoreMathCue) {
      mathCueQuestions += 1;
    }

    for (const [optionIndex, option] of question.options.entries()) {
      optionCount += 1;
      const isCorrect = option.isCorrect;
      const cuePredictsCorrect = !hasLanguageCue(option.text);
      const middleCuePredictsCorrect =
        hasMiddleAnswerCue(option.text) && !hasLanguageCue(option.text);
      const mathCuePredictsCorrect = mathSalienceFlags[optionIndex];

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

      if (shouldScoreMathCue) {
        mathCueOptionCount += 1;
        if (isCorrect) {
          mathCueCorrectOptions += 1;
        } else {
          mathCueIncorrectOptions += 1;
        }
        if (mathCuePredictsCorrect === isCorrect) {
          mathCueBaselineCorrect += 1;
        }
      }

      if (isCorrect) {
        correctOptions += 1;
        if (hasLanguageCue(option.text)) trueOptionsWithCues += 1;
        if (hasMiddleAnswerCue(option.text)) trueOptionsWithMiddleCues += 1;
        if (shouldScoreMathCue && mathCuePredictsCorrect) {
          trueOptionsWithMathCues += 1;
        }
      } else {
        incorrectOptions += 1;
        if (hasLanguageCue(option.text)) falseOptionsWithCues += 1;
        if (hasMiddleAnswerCue(option.text)) falseOptionsWithMiddleCues += 1;
        if (shouldScoreMathCue && mathCuePredictsCorrect) {
          falseOptionsWithMathCues += 1;
        }
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
  const mathCueBaselineAccuracy = safeRate(
    mathCueBaselineCorrect,
    mathCueOptionCount,
  );
  const mathCueTrueOptionRate = safeRate(
    mathCueCorrectOptions,
    mathCueOptionCount,
  );
  const mathCueFalseOptionRate = safeRate(
    mathCueIncorrectOptions,
    mathCueOptionCount,
  );
  const mathCueMajorityBaselineAccuracy = Math.max(
    mathCueTrueOptionRate,
    mathCueFalseOptionRate,
  );
  const trueMathCueRate = safeRate(
    trueOptionsWithMathCues,
    mathCueCorrectOptions,
  );
  const falseMathCueRate = safeRate(
    falseOptionsWithMathCues,
    mathCueIncorrectOptions,
  );

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
    mathCueBaselineAccuracy,
    mathCueBaselineLiftOverMajority:
      mathCueBaselineAccuracy - mathCueMajorityBaselineAccuracy,
    trueMathCueRate,
    falseMathCueRate,
    mathCueRateGap: trueMathCueRate - falseMathCueRate,
    mathCueQuestions,
    mathCueOptions: mathCueOptionCount,
    uniqueMathOutlierCorrectRate: safeRate(
      uniqueMathOutlierCorrect,
      uniqueMathOutlierQuestions,
    ),
    uniqueMathOutlierQuestions,
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
    `mathCueAccuracy=${formatPercent(metrics.mathCueBaselineAccuracy)}`,
    `mathCueLift=${formatPercent(metrics.mathCueBaselineLiftOverMajority)}`,
    `mathCueGap=${formatPercent(metrics.mathCueRateGap)}`,
    `mathCueQuestions=${metrics.mathCueQuestions}`,
    `uniqueMathOutlierCorrect=${formatPercent(metrics.uniqueMathOutlierCorrectRate)}`,
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

function makeMathCueDrivenQuestion(index: number): Question {
  return {
    id: `synthetic-math-guessable-q${index}`,
    chapter: 1,
    difficulty: "medium",
    prompt: "Which option best describes scaled dot-product attention?",
    options: [
      {
        text: "\\(\\operatorname{Attention}(Q,K,V)=\\operatorname{softmax}(QK^T/\\sqrt{d_k})V\\), so query-key dot products weight value vectors.",
        isCorrect: true,
      },
      {
        text: "It is a fixed word-count window that never compares token representations.",
        isCorrect: false,
      },
      {
        text: "It is a vocabulary lookup table that replaces all contextual computation.",
        isCorrect: false,
      },
      {
        text: "It is a training schedule that lowers the learning rate after each batch.",
        isCorrect: false,
      },
    ],
    explanation:
      "This synthetic question is intentionally math-cue driven so the heuristic test has a stable fixture. Real questions should add plausible competing formulas or remove gratuitous formula-only answer cues.",
  };
}

function makeCompetingMathQuestion(index: number): Question {
  return {
    id: `synthetic-competing-math-q${index}`,
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about simple expressions are correct?",
    options: [
      {
        text: "\\(a+b=b+a\\) for ordinary real-number addition.",
        isCorrect: true,
      },
      {
        text: "\\(a-b=b-a\\) for ordinary real-number subtraction.",
        isCorrect: false,
      },
      {
        text: "A probability mass function assigns nonnegative probabilities that sum to one.",
        isCorrect: true,
      },
      {
        text: "A probability mass function can assign negative probabilities if the average is positive.",
        isCorrect: false,
      },
    ],
    explanation:
      "This synthetic question uses math in both correct and incorrect options. The fixture protects against treating math itself as the problem.",
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

  it("scores a synthetic math-salience set as guessable", () => {
    const metrics = scoreGuessability(
      Array.from({ length: MIN_QUESTIONS_FOR_STABLE_SCORE }, (_, index) =>
        makeMathCueDrivenQuestion(index + 1),
      ),
    );

    expect(metrics.mathCueOptions).toBeGreaterThanOrEqual(MIN_MATH_CUE_OPTIONS);
    expect(metrics.mathCueBaselineAccuracy).toBeGreaterThan(
      MAX_MATH_CUE_BASELINE_OPTION_ACCURACY,
    );
    expect(metrics.mathCueBaselineLiftOverMajority).toBeGreaterThan(
      MAX_MATH_CUE_BASELINE_LIFT_OVER_MAJORITY,
    );
    expect(metrics.mathCueRateGap).toBeGreaterThan(MAX_MATH_CUE_RATE_GAP);
    expect(metrics.uniqueMathOutlierQuestions).toBeGreaterThanOrEqual(
      MIN_UNIQUE_MATH_OUTLIER_QUESTIONS,
    );
    expect(metrics.uniqueMathOutlierCorrectRate).toBe(1);
  });

  it("does not score competing math options as math-cue driven", () => {
    const metrics = scoreGuessability(
      Array.from({ length: MIN_QUESTIONS_FOR_STABLE_SCORE }, (_, index) =>
        makeCompetingMathQuestion(index + 1),
      ),
    );

    expect(metrics.mathCueOptions).toBeGreaterThanOrEqual(MIN_MATH_CUE_OPTIONS);
    expect(metrics.mathCueBaselineAccuracy).toBeLessThanOrEqual(0.5);
    expect(metrics.mathCueRateGap).toBe(0);
    expect(metrics.uniqueMathOutlierQuestions).toBe(0);
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

        if (
          metrics.mathCueOptions >= MIN_MATH_CUE_OPTIONS &&
          metrics.mathCueBaselineAccuracy >
            MAX_MATH_CUE_BASELINE_OPTION_ACCURACY &&
          metrics.mathCueBaselineLiftOverMajority >
            MAX_MATH_CUE_BASELINE_LIFT_OVER_MAJORITY
        ) {
          sourceFailures.push(
            `math/KaTeX cue baseline is too accurate (${formatPercent(metrics.mathCueBaselineAccuracy)} with ${formatPercent(metrics.mathCueBaselineLiftOverMajority)} lift over majority baseline across ${metrics.mathCueQuestions} questions)`,
          );
        }

        if (
          metrics.mathCueOptions >= MIN_MATH_CUE_OPTIONS &&
          metrics.trueMathCueRate > metrics.falseMathCueRate &&
          metrics.mathCueRateGap > MAX_MATH_CUE_RATE_GAP
        ) {
          sourceFailures.push(
            `math/KaTeX cues are concentrated in correct options (${formatPercent(metrics.trueMathCueRate)} true vs ${formatPercent(metrics.falseMathCueRate)} false)`,
          );
        }

        if (
          metrics.uniqueMathOutlierQuestions >=
            MIN_UNIQUE_MATH_OUTLIER_QUESTIONS &&
          metrics.uniqueMathOutlierCorrectRate >
            MAX_UNIQUE_MATH_OUTLIER_CORRECT_RATE
        ) {
          sourceFailures.push(
            `the unique math-heavy option is correct too often (${formatPercent(metrics.uniqueMathOutlierCorrectRate)} across ${metrics.uniqueMathOutlierQuestions} questions)`,
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
          "For math/KaTeX failures, keep useful math but add plausible competing formulas, calculations, dimensions, or boundary cases instead of making the correct option the only formula-heavy choice.",
          "A pass is only a deterministic smoke test; it is not proof that the set has high diagnosticity.",
        ].join("\n"),
      ).toEqual([]);
    },
  );
});
