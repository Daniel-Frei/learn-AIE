import { describe, expect, it } from "vitest";
import { normalizeMathDelimiters } from "@/lib/normalizeMath";

describe("normalizeMathDelimiters", () => {
  it("normalizes bracket and parenthesis LaTeX delimiters", () => {
    const input = "\\[x^2 + y^2\\] then \\(z\\)";
    const output = normalizeMathDelimiters(input);

    expect(output).toBe("$$x^2 + y^2$$ then $z$");
  });
});
