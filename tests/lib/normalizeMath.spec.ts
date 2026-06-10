import { describe, expect, it } from "vitest";
import { normalizeMathDelimiters } from "@/lib/normalizeMath";

describe("normalizeMathDelimiters", () => {
  it("normalizes bracket and parenthesis LaTeX delimiters", () => {
    const input = "\\[x^2 + y^2\\] then \\(z\\)";
    const output = normalizeMathDelimiters(input);

    expect(output).toBe("\n\n$$\nx^2 + y^2\n$$\n\n then $z$");
  });

  it("expands one-line display dollar delimiters into markdown display blocks", () => {
    const input = "$$P(y_i)=\\frac{e^{z_i}}{\\sum_j e^{z_j}}$$";
    const output = normalizeMathDelimiters(input);

    expect(output).toBe(
      "\n\n$$\nP(y_i)=\\frac{e^{z_i}}{\\sum_j e^{z_j}}\n$$\n\n",
    );
  });

  it("normalizes double-escaped formulas commonly produced in TSX attributes", () => {
    const input = "\\\\[P(y_i)=\\\\frac{e^{z_i}}{\\\\sum_j e^{z_j}}\\\\]";
    const output = normalizeMathDelimiters(input);

    expect(output).toBe(
      "\n\n$$\nP(y_i)=\\frac{e^{z_i}}{\\sum_j e^{z_j}}\n$$\n\n",
    );
  });
});
