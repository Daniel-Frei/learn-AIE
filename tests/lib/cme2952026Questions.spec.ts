import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  QUESTION_SOURCES,
  QUESTION_SOURCE_CONTEXT,
  getQuestionsForFilters,
  getSourceIdsForSeries,
  getQuestionType,
  type Question,
} from "@/lib/quiz";

const sources = QUESTION_SOURCES.filter(
  (source) => source.seriesId === "stanford-cme295-2026",
);
const sourceIds = [
  "cme295-2026-lect1",
  "cme295-2026-lect2",
  "cme295-2026-lect3",
  "cme295-2026-lect4",
] as const;

function question(lecture: number, number: number): Question {
  const item = sources[lecture - 1]?.questions[number - 1];
  if (!item)
    throw new Error("Missing 2026 lecture/slot " + lecture + "/" + number);
  return item;
}

function selectedTexts(lecture: number, number: number) {
  return question(lecture, number)
    .options.filter((option) => option.isCorrect)
    .map((option) => option.text);
}

describe("CME295 2026 question banks", () => {
  it("selects four 50-question sources through the 2026 series", () => {
    expect(sources.map((source) => source.id)).toEqual(sourceIds);
    expect(getSourceIdsForSeries("stanford-cme295-2026")).toEqual(sourceIds);
    expect(getQuestionsForFilters([...sourceIds], [])).toHaveLength(200);
    for (const source of sources) {
      expect(source.questions).toHaveLength(50);
      expect(source.label).toContain("2026");
      expect(QUESTION_SOURCE_CONTEXT[source.id]).toContain("2026");
      expect(getQuestionsForFilters([source.id], [])).toEqual(source.questions);
      expect(new Set(source.questions.map((q) => q.id)).size).toBe(50);
    }
  });

  it("references 46 canonical identities and defines 154 edition-specific items", () => {
    const legacy = QUESTION_SOURCES.filter(
      (source) => source.seriesId !== "stanford-cme295-2026",
    ).flatMap((source) => source.questions);
    const legacyIds = new Set(legacy.map((q) => q.id));
    const questions = sources.flatMap((source) => source.questions);
    expect(new Set(questions.map((q) => q.id)).size).toBe(200);
    const shared = questions.filter((q) => legacyIds.has(q.id));
    expect(shared).toHaveLength(46);
    for (const q of shared) {
      expect(legacy.find((old) => old.id === q.id)).toBe(q);
    }
    expect(
      sources.map(
        (source) => source.questions.filter((q) => legacyIds.has(q.id)).length,
      ),
    ).toEqual([0, 28, 13, 5]);
    expect(getSourceIdsForSeries("stanford-cme295")).toHaveLength(9);
    const folder = path.join(
      process.cwd(),
      "lib/lectures/Stanford CME295 2026 Transformers & LLMs",
    );
    const authoredIds = fs
      .readdirSync(folder)
      .filter((file) => file.endsWith(".ts"))
      .flatMap((file) =>
        [
          ...fs
            .readFileSync(path.join(folder, file), "utf8")
            .matchAll(/"?id"?:\s*"(cme295-2026-lect[1-4]-q\d+)"/g),
        ].map((match) => match[1]),
      );
    expect(authoredIds).toHaveLength(154);
    expect(authoredIds.sort()).toEqual(
      questions
        .filter((q) => !legacyIds.has(q.id))
        .map((q) => q.id)
        .sort(),
    );
  });

  it.each(sourceIds)("keeps %s self-contained, valid, and balanced", (id) => {
    const source = sources.find((entry) => entry.id === id)!;
    const answerCounts = [0, 0, 0, 0];
    for (const q of source.questions) {
      expect(getQuestionType(q)).toBe("multiple-select");
      expect(q.options).toHaveLength(4);
      const count = q.options.filter((option) => option.isCorrect).length;
      expect(count).toBeGreaterThanOrEqual(1);
      expect(count).toBeLessThanOrEqual(4);
      answerCounts[count - 1] += 1;
      expect(q.explanation.length).toBeGreaterThanOrEqual(200);
      expect(
        q.explanation.match(/[.!?](?:\s|$)/g)?.length ?? 0,
      ).toBeGreaterThanOrEqual(2);
      expect(q.prompt).not.toMatch(
        /in the (?:lecture|transcript|slides)|(?:previous|next) question|as above/i,
      );
      expect(q.prompt + " " + q.explanation).not.toMatch(/[\t\u0000-\u0008]/);
    }
    for (const count of answerCounts)
      expect(Math.abs(count - 12.5)).toBeLessThanOrEqual(5);
  });

  it("preserves the worked attention and causal-prefix distinctions", () => {
    expect(selectedTexts(1, 24)).toEqual(["\\((0.5,3)\\)"]);
    expect(selectedTexts(1, 42)).toEqual(["[BOS], y1, and y2."]);
    expect(selectedTexts(1, 47)).toContain(
      "The observed token has target probability 0.85.",
    );
    expect(selectedTexts(1, 49)).toEqual([
      "Changing y4 while keeping the source, y1, and y2 fixed changes the distribution for y3.",
    ]);
  });

  it("moves MoE and decoding into Lecture 2 and uses the visual sparse-attention pattern", () => {
    expect(selectedTexts(2, 15)).toContain(
      "The active expert path uses 4 billion parameters.",
    );
    expect(selectedTexts(2, 19)).toContain(
      "Three expert computations are active for a token in this layer.",
    );
    expect(new Set(selectedTexts(2, 31))).toEqual(
      new Set([
        "Position 6 attends to {1, 5, 6, 7}.",
        "Position 1 attends to {1, 2, 3, 4, 5, 6, 7, 8}.",
        "Position 8 attends to {1, 7, 8}.",
        "Position 2 attends to {1, 2, 3}.",
      ]),
    );
    expect(selectedTexts(2, 42)).toEqual([
      "Keep A, B, C and use probabilities \\((0.50,0.333\\ldots,0.166\\ldots)\\).",
    ]);
  });

  it("covers new loss, packing, and distillation material without old training-infrastructure topics", () => {
    expect(selectedTexts(3, 5)).toContain("Its bits-per-byte score is 0.5.");
    expect(selectedTexts(3, 16)).toEqual(["65,536 parameters."]);
    expect(selectedTexts(3, 30)).toEqual(["\\(0.15\\)"]);
    expect(selectedTexts(3, 39)).toEqual(["The current student policy."]);
    expect(selectedTexts(3, 45)).toEqual([
      "Two blocks and two padding tokens.",
    ]);
    const lecture3 = sources[2].questions.map((q) => q.prompt).join("\n");
    expect(lecture3).not.toMatch(
      /FlashAttention|ZeRO|QLoRA|NF4|Pass@|Best-of-N/,
    );
    expect(question(3, 48).prompt).toMatch(/held-out coding set/);
    expect(question(3, 49).prompt).toMatch(/frozen earlier checkpoint/);
  });

  it("checks baseline arithmetic, negative-advantage clipping, and reverse-KL direction", () => {
    expect(selectedTexts(4, 11)).toContain("Their expectation is 0.25.");
    expect(selectedTexts(4, 12)).toContain(
      "Both centered sample gradients equal 0.25.",
    );
    expect(selectedTexts(4, 22)).toEqual(["\\(-1.6\\)"]);
    expect(selectedTexts(4, 37)).toEqual([
      "The hard-group success has advantage \\(\\sqrt{3}\\), while an easy-group success has \\(1/\\sqrt{3}\\).",
    ]);
    expect(selectedTexts(4, 42)).toEqual([
      "\\(0.75\\ln(1.5)+0.25\\ln(0.5)\\).",
    ]);
    expect(selectedTexts(4, 43)).toContain(
      "A positive s_i with t_i equal to zero makes the divergence infinite.",
    );
  });

  it("checks the revised mask-normalization and data-mixture computations", () => {
    const maskedWeights = [1, 2, 0].map((value) => value / 3);
    const output = maskedWeights.reduce(
      (sum, weight, i) => sum + weight * [1, 3, 100][i],
      0,
    );
    expect(output).toBeCloseTo(7 / 3);
    expect(selectedTexts(1, 43)).toHaveLength(3);
    expect(selectedTexts(1, 43)).toContain(
      "The correctly masked weighted value is \\(7/3\\).",
    );
    expect(
      question(1, 43).options.find((o) =>
        /Zeroing the future weight after/.test(o.text),
      )?.isCorrect,
    ).toBe(false);
    const humanWeight = (0.18 - 0.1) / (0.2 - 0.1);
    expect(humanWeight).toBeCloseTo(0.8);
    expect(selectedTexts(3, 47)).toContain(
      "The 18% mixture uses a human-to-synthetic sampling ratio of 4:1.",
    );
  });

  it("checks importance weighting and the hinted reverse-KL gradient", () => {
    const old = [0.2, 0.8],
      next = [0.3, 0.7],
      advantage = [2, -1];
    const weighted = old.reduce(
      (sum, p, i) => sum + p * (next[i] / p) * advantage[i],
      0,
    );
    expect(weighted).toBeCloseTo(-0.1);
    expect(selectedTexts(4, 20)).toContain(
      "\\(\\mathbb E_{\\rm old}[\\rho A]=-0.1=\\mathbb E_{\\rm new}[A]\\).",
    );
    const s = 0.5,
      t = 0.8;
    const gradient = s * (1 - s) * Math.log((s * (1 - t)) / (t * (1 - s)));
    expect(gradient).toBeCloseTo(-Math.log(4) / 4);
    expect(selectedTexts(4, 48)).toContain(
      "A gradient-descent step increases the student's first-token logit.",
    );
    expect(selectedTexts(4, 47)).toEqual(["[0.9, 0.1]"]);
  });

  it("separates the zero-mean score identity from a probability-gradient expectation", () => {
    const p = 0.2;
    const scoreMean = p * (1 - p) + (1 - p) * -p;
    const probabilityGradientMean = p * p * (1 - p) + (1 - p) * -p * (1 - p);
    expect(scoreMean).toBeCloseTo(0);
    expect(probabilityGradientMean).toBeCloseTo(-0.096);
    expect(selectedTexts(4, 13)).toHaveLength(1);
    expect(selectedTexts(4, 13)[0]).toContain(
      "\\nabla_\\theta\\sum_a\\pi_\\theta",
    );
  });
});
