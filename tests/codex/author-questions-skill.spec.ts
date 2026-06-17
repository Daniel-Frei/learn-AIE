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

  it("covers creation, review, improvement, addition, and targeted rewrites", () => {
    expect(skill).toContain("## Supported Operations");
    expect(skill).toContain(
      "Use the same quality gate for creation, review, and improvement work.",
    );
    expect(skill).toContain("Create a new question set from source material");
    expect(skill).toContain("Add questions to an existing set");
    expect(skill).toContain("Review an existing set for quality issues");
    expect(skill).toContain("Rewrite a targeted topic slice");
    expect(skill).toContain("Combine operations in one pass");
    expect(skill).toContain(
      "For review-only tasks, perform the same triage but stop at findings unless the user asked you to edit.",
    );
    expect(teamPreferences).toContain(
      "creating new question sets, adding questions to existing sets, reviewing question quality, improving weak existing questions, and targeted rewrites of topic slices",
    );
    expect(teamPreferences).toContain(
      "the same quality gate applies across those operations",
    );
  });

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

  it("supports mixed multi-select and assertion-reason question sets", () => {
    expect(skill).toContain(
      "Supported question types are multi-select multiple-choice and assertion-reason MCQs.",
    );
    expect(skill).toContain('Set `type: "assertion-reason"`');
    expect(skill).toContain(
      "Choose the mix of question types based on the source material, subject, and field.",
    );
    expect(skill).toContain(
      "Assertion-reason answer options should not rely on randomization",
    );
    expect(skill).toContain("Both are false.");
    expect(skill).toContain(
      'Do not include a trailing instruction sentence such as "Which option correctly evaluates the assertion and reason?"',
    );
    expect(skill).toContain(
      "only when the Reason gives a causal, mechanistic, or logically sufficient explanation",
    );
    expect(skill).toContain(
      "vary which of the fixed five ordered options is correct as practical",
    );
    expect(teamPreferences).toContain(
      "Question-bank files may mix `multiple-select` and `assertion-reason` questions.",
    );
    expect(teamPreferences).toContain(
      "use the standard fixed five-option assertion/reason order",
    );
    expect(teamPreferences).toContain(
      "only when the reason gives a causal, mechanistic, or logically sufficient explanation",
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

  it("guards against math salience becoming an answer cue", () => {
    expect(skill).toContain(
      "do not make the correct answer recognizable as the only math-heavy",
    );
    expect(skill).toContain(
      "Do not remove useful math merely to avoid a math-salience cue.",
    );
    expect(skill).toContain(
      'simulate "select the only option with substantial math, a long formula, a calculation, or KaTeX/LaTeX."',
    );
    expect(teamPreferences).toContain(
      "do not make correct answers recognizable as the only option with substantial math",
    );
    expect(teamPreferences).toContain(
      "add plausible competing formulas, calculations, dimensions, or boundary cases",
    );
  });

  it("requires manual diagnosticity triage for every creation or edit", () => {
    expect(skill).toContain("## Question Quality Gate");
    expect(skill).toContain(
      "Run this gate whenever this skill creates, adds, rewrites, fixes, or otherwise edits questions.",
    );
    expect(skill).toContain(
      "Passing deterministic guessability checks is not enough",
    );
    expect(skill).toContain("construct target and misconception target");
    expect(skill).toContain(
      "weak recognition items were substantially rewritten rather than lightly polished",
    );
    expect(skill).toContain(
      "reports created/changed, reviewed, substantially rewritten, minor-edited, and intentionally retained orientation counts",
    );
    expect(teamPreferences).toContain(
      "When creating, adding, rewriting, fixing, or otherwise editing questions, do a manual low-diagnosticity triage even if automated guessability checks pass",
    );
    expect(teamPreferences).toContain(
      "medium and hard items should not be answerable mainly by recognizing a familiar definition",
    );
  });

  it("requires questions to stand alone under randomized mixed-source practice", () => {
    expect(skill).toContain(
      "Every question must stand alone because practice can randomize question order and mix questions from different source sets.",
    );
    expect(skill).toContain(
      "Do not refer to other questions or depend on a previous prompt, answer, explanation, image, equation, or source-context card.",
    );
    expect(skill).toContain(
      "No prompt or explanation depends on seeing another question first; every question is independent under randomized mixed-source practice.",
    );
    expect(teamPreferences).toContain(
      "Every question should stand alone under randomized mixed-source practice.",
    );
  });

  it("allows compact prompt tables when they improve readability", () => {
    expect(skill).toContain("Use compact Markdown tables in prompts");
    expect(skill).toContain("GitHub-Flavored Markdown tables");
    expect(skill).toContain(
      "Introduce what the table represents before the table",
    );
    expect(skill).toContain("meaningful row and column labels");
    expect(skill).toContain("template literal for multi-line prompts");
    expect(skill).toContain(
      "Prompt tables, when used, are valid GitHub-Flavored Markdown",
    );
    expect(teamPreferences).toContain(
      "Quiz prompts should use compact GitHub-Flavored Markdown tables a little more readily",
    );
    expect(teamPreferences).toContain(
      "Tables should be introduced by the prompt",
    );
  });
});
