import { renderToString } from "katex";
import { describe, expect, it } from "vitest";
import { QUESTION_SOURCES, type Question } from "@/lib/quiz";

const questions = QUESTION_SOURCES.flatMap((source) => source.questions);
function question(id: string): Question {
  const result = questions.find((item) => item.id === id);
  expect(result, `Missing question ${id}`).toBeDefined();
  return result!;
}

const changedIds = [
  "cs109-lect4-q18",
  "cs109-lect5-q09",
  "cs109-lect5-q29",
  "cs109-lect6-q35",
  "la-crash-l2-q29",
  "la-crash-l3-q38",
  "la-crash-l4-q34",
  "la-crash-l4-q61",
  "la-crash-l5-q61",
  "cs224r-lect4-q69",
  "math-logs-exp-roots-q67",
  ...[5, 14, 16, 20, 37, 46].map(
    (n) => `la-crash-l2-q${String(n).padStart(2, "0")}`,
  ),
  "la-crash-l4-q01",
  "la-crash-l4-q09",
];

describe("reported question corrections", () => {
  it("shows the Bayes normalization and posterior arithmetic", () => {
    const evidence = 0.2 * 0.6 + 0.01 * 0.4;
    expect(evidence).toBeCloseTo(0.124);
    expect((0.2 * 0.6) / evidence).toBeCloseTo(0.9677, 4);
    const explanation = question("cs109-lect4-q18").explanation;
    expect(explanation).toContain("0.20(0.60)+0.01(0.40)=0.124");
    expect(explanation).toContain("0.12/0.124");
  });

  it("uses a conditional or joint identity to establish independence", () => {
    const item = question("cs109-lect5-q09");
    const correct = item.options.filter((option) => option.isCorrect);
    expect(correct).toHaveLength(2);
    expect(correct[0].text).toContain("P(E\\mid F)=1/6=P(E)");
    expect(correct[1].text).toContain("P(E\\cap F)=1/36=(1/6)(1/6)");
    expect(item.explanation).toContain("Equal marginal probabilities alone");
    // Identical events have equal marginals without being independent.
    expect(1 / 6).not.toBeCloseTo((1 / 6) ** 2);
  });

  it("adds disjoint job outcomes after multiplying within each outcome", () => {
    const rates = [0.8, 0.7, 0.6];
    let exactlyTwo = 0;
    let allThree = 0;
    for (let outcome = 0; outcome < 8; outcome += 1) {
      let successes = 0;
      let mass = 1;
      rates.forEach((rate, index) => {
        const success = Boolean(outcome & (1 << index));
        successes += Number(success);
        mass *= success ? rate : 1 - rate;
      });
      if (successes === 2) exactlyTwo += mass;
      if (successes === 3) allThree += mass;
    }
    expect(exactlyTwo).toBeCloseTo(0.452);
    expect(allThree).toBeCloseTo(0.336);
    const item = question("cs109-lect5-q29");
    expect(exactlyTwo + allThree).toBeCloseTo(0.788);
    expect(item.options[1].isCorrect).toBe(true);
    expect(item.options[1].text).toContain("0.452+0.336=0.788");
    expect(item.explanation).toContain("mutually exclusive");
  });

  it("defines the unknown PMF constant and retains the correct normalized mean", () => {
    const item = question("cs109-lect6-q35");
    expect(item.prompt).toContain("unknown constant");
    expect(item.prompt).toContain("zero otherwise");
    for (const n of [1, 2, 5, 10]) {
      const c = 2 / ((n + 1) * (n + 2));
      const pmf = Array.from({ length: n + 1 }, (_, k) => c * (k + 1));
      expect(pmf.reduce((total, p) => total + p, 0)).toBeCloseTo(1);
      expect(pmf.reduce((total, p, k) => total + p * k, 0)).toBeCloseTo(
        (2 * n) / 3,
      );
    }
    expect(item.options[0].isCorrect).toBe(true);
  });

  it("explains column-vector composition with a noncommuting example", () => {
    const item = question("la-crash-l2-q29");
    expect(item.prompt).toContain("column vectors");
    const stretch = ([x, y]: number[]) => [2 * x, y];
    const rotate = ([x, y]: number[]) => [-y, x];
    const ab = stretch(rotate([1, 0]));
    const ba = rotate(stretch([1, 0]));
    expect(ab[0]).toBeCloseTo(0);
    expect(ab[1]).toBe(1);
    expect(ba[0]).toBeCloseTo(0);
    expect(ba[1]).toBe(2);
    expect(item.explanation).toContain("ABx=A(Bx)");
    expect(item.options[3].isCorrect).toBe(false);
  });

  it("asks for true statements explicitly in the two orientation items", () => {
    for (const id of ["la-crash-l3-q38", "la-crash-l4-q34"]) {
      const item = question(id);
      expect(item.prompt).toMatch(/Which statements about .+ are correct\?/);
      expect(item.prompt).not.toContain("misconceptions");
      expect(item.difficulty).toBe("easy");
      expect(item.options.map((option) => option.isCorrect)).toEqual([
        true,
        true,
        true,
        false,
      ]);
    }
  });

  it("distinguishes positive semidefiniteness from entry signs and strict positivity", () => {
    const item = question("la-crash-l4-q61");
    // X = [1, -1] gives a Gram matrix with negative off-diagonal entries
    // and a nonzero null vector (1, 1), while its quadratic form is a square.
    for (const [a, b] of [
      [1, 1],
      [1, 0],
      [-2, 3],
    ]) {
      const form = a * (a - b) + b * (-a + b);
      expect(form).toBe((a - b) ** 2);
      expect(form).toBeGreaterThanOrEqual(0);
    }
    expect(item.explanation).toContain("negative off-diagonal entries");
    expect(item.explanation).toContain("strict positivity is not guaranteed");
    expect(item.options.map((option) => option.isCorrect)).toEqual([
      true,
      false,
      false,
      false,
    ]);
    expect(questions.some((item) => item.id === "la-crash-l4-q58")).toBe(false);
  });

  it("compares a saddle with a one-dimensional minimum using explicit loss changes", () => {
    const item = question("la-crash-l5-q61");
    const change = (u: number, v: number, w: number) => u * u + v * v - w * w;
    expect(change(0.1, 0, 0)).toBeGreaterThan(0);
    expect(change(0, 0, 0.1)).toBeLessThan(0);
    expect(item.prompt).toContain("one-parameter loss");
    expect(item.explanation).toContain("exact gradient-descent step is zero");
    expect(item.options.map((option) => option.isCorrect)).toEqual([
      true,
      true,
      false,
      false,
    ]);
    expect(questions.some((item) => item.id === "la-crash-l5-q19")).toBe(false);
  });

  it("makes the random-termination interpretation self-contained", () => {
    const item = question("cs224r-lect4-q69");
    expect(item.prompt).toContain("Markov decision process (MDP)");
    expect(item.prompt).not.toContain("slides");
    expect(item.explanation).toContain("After each current reward");
    expect(item.explanation).toContain("\\gamma^t");
  });

  it("keeps numeric log choices out of Markdown ordered-list syntax", () => {
    const item = question("math-logs-exp-roots-q67");
    expect(item.options.map((option) => option.text)).toEqual([
      "\\(3\\)",
      "\\(4\\)",
      "\\(27\\)",
      "\\(81\\)",
    ]);
    expect(item.options.find((option) => option.isCorrect)?.text).toBe(
      "\\(4\\)",
    );
    expect(3 ** 4).toBe(81);
  });

  it("renders corrected math and previously reported theta formulas without corruption", () => {
    const reportedThetaIds = [40, 48, 52, 56, 57, 67, 68, 69].map(
      (n) => `cs224r-lect5-q${n}`,
    );
    for (const id of [...changedIds, ...reportedThetaIds]) {
      const item = question(id);
      for (const field of [
        item.prompt,
        item.explanation,
        ...item.options.map((option) => option.text),
      ]) {
        expect(field).not.toMatch(/[\t\r\u0008\u000c]/);
        for (const match of field.matchAll(
          /\\\(([\s\S]*?)\\\)|\\\[([\s\S]*?)\\\]/g,
        )) {
          expect(() =>
            renderToString(match[1] ?? match[2], {
              throwOnError: true,
              strict: "error",
            }),
          ).not.toThrow();
        }
      }
    }
  });
});
