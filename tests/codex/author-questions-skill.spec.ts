import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, "../..");

function readRepoFile(relativePath: string) {
  return fs.readFileSync(path.join(repoRoot, relativePath), "utf8");
}

describe("author-questions skill guidance", () => {
  const skill = readRepoFile(".codex/skills/author-questions/SKILL.md");
  const teamPreferences = readRepoFile("docs/team-preferences.md");

  it("asks for a question count instead of using a built-in default", () => {
    expect(skill).toContain("ask for the count before generating questions");
    expect(skill).toContain("do not assume a default count from this skill");
    expect(skill).not.toMatch(/Default to \d+ questions/i);
    expect(skill).not.toContain("40 questions per source material");
    expect(skill).not.toContain("20-question");
  });

  it("treats difficulty balancing as the default only when unspecified", () => {
    expect(skill).toContain(
      "If the user, source material, existing set convention, or local docs specify a difficulty distribution, follow that specified distribution.",
    );
    expect(skill).toContain(
      "If no difficulty distribution was specified, default to a roughly balanced mix",
    );
    expect(teamPreferences).toContain(
      "do not require an equal static difficulty-label split unless the user, source material, existing set convention, or local docs ask for one",
    );
  });

  it("requires reporting difficulty counts back to the user", () => {
    expect(skill).toContain("Always report the final difficulty balance");
    expect(skill).toContain(
      'Final response reports the difficulty balance with `"easy"`, `"medium"`, and `"hard"` counts.',
    );
    expect(teamPreferences).toContain(
      "always report the final difficulty counts back to the user",
    );
  });

  it("keeps question difficulty separate from answer-option quality", () => {
    expect(skill).toContain(
      "Difficulty should come from the level of knowledge, reasoning, transfer, or math required by the concept",
    );
    expect(skill).toContain(
      "A learner should usually need concept understanding to distinguish correct options from incorrect options.",
    );
    expect(skill).toContain(
      "Avoid distractors that are absurd, category-mismatched, self-refuting, or easy to eliminate from wording cues alone.",
    );
    expect(teamPreferences).toContain(
      "Questions should test real concept understanding rather than surface recall or answer elimination.",
    );
  });

  it("guards against generic answer-recognition patterns", () => {
    expect(skill).toContain(
      'Avoid making the correct answer the only "reasonable middle" option',
    );
    expect(skill).toContain(
      "Keep distractors in the same category and reasoning layer as the prompt.",
    );
    expect(skill).toContain(
      "Prompts should identify the concept or task precisely without revealing the abstract answer pattern.",
    );
    expect(skill).toContain(
      '"select the hedged middle answer", "reject the category mismatch", and "infer the context/validation answer from the stem."',
    );
    expect(teamPreferences).toContain(
      "do not let correct answers become recognizable as the only hedged, context-sensitive, evidence-aware",
    );
    expect(teamPreferences).toContain(
      "Avoid over-scaffolding a set around one reusable schema.",
    );
  });
});
