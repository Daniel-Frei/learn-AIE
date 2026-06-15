import { describe, expect, it } from "vitest";
import { patientCases } from "../../components/learning/pages/crash-medicine-lecture-1/clinicalData";
import {
  byPriority,
  makeProblemRepresentation,
  planScore,
  representationCompleteness,
  revealedRisk,
} from "../../components/learning/pages/crash-medicine-lecture-1/reasoning";

describe("clinical reasoning model", () => {
  it("prioritizes dangerous diagnoses over likelihood alone", () => {
    const chestPain = patientCases[0];
    const ranked = byPriority(chestPain.diagnoses);

    expect(ranked[0].id).toBe("acs");
    expect(
      ranked.findIndex((diagnosis) => diagnosis.id === "dissection"),
    ).toBeLessThan(ranked.findIndex((diagnosis) => diagnosis.id === "reflux"));
  });

  it("builds a representation from selected evidence", () => {
    const febrile = patientCases[1];
    const selected = ["age", "diabetes", "kidney", "confusion", "bp", "o2"];

    expect(makeProblemRepresentation(febrile, selected)).toContain(
      "74-year-old woman",
    );
    expect(makeProblemRepresentation(febrile, selected)).toContain("hypoxemia");
    expect(representationCompleteness(febrile, selected).score).toBeGreaterThan(
      0.6,
    );
  });

  it("scores safer plans higher than unsafe closure", () => {
    const chestPain = patientCases[0];
    const safe = planScore(chestPain, [
      "ecg-now",
      "troponin",
      "monitor",
      "aspirin-selected",
    ]);
    const unsafe = planScore(chestPain, ["reassure", "antacid-only"]);

    expect(safe).toBeGreaterThan(unsafe);
  });

  it("caps revealed risk at one hundred", () => {
    const febrile = patientCases[1];
    const allEvidence = febrile.evidence.map((item) => item.id);

    expect(revealedRisk(febrile.evidence, allEvidence)).toBe(100);
  });
});
