import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, "../..");

function readRepoFile(relativePath: string) {
  return fs.readFileSync(path.join(repoRoot, relativePath), "utf8");
}

function collapseWhitespace(value: string) {
  return value.replace(/\s+/g, " ").trim();
}

describe("author-learning-experience skill guidance", () => {
  const skill = readRepoFile(
    ".codex/skills/author-learning-experience/SKILL.md",
  );
  const normalizedSkill = collapseWhitespace(skill);
  const teamPreferences = readRepoFile("docs/team-preferences.md");
  const normalizedTeamPreferences = collapseWhitespace(teamPreferences);
  const agents = readRepoFile("AGENTS.md");
  const packageJson = JSON.parse(readRepoFile("package.json")) as {
    dependencies?: Record<string, string>;
  };

  it("does not treat older learning pages as the UX quality bar", () => {
    expect(normalizedSkill).toContain(
      "older Learning AI learning pages are generally not the quality bar",
    );
    expect(normalizedSkill).toContain(
      "Do not use this repo's existing learning pages as the main comparison point",
    );
    expect(normalizedSkill).toContain(
      "Existing pages are useful for route, registry, app shell, Tailwind style",
    );
    expect(normalizedTeamPreferences).toContain(
      "Older Learning AI learning pages are generally not the UX quality bar",
    );
  });

  it("keeps integration support from shrinking the learning experience", () => {
    expect(skill).toContain(
      "Route registration, docs, quiz transitions, and tests are support work.",
    );
    expect(skill).toContain(
      "If no quiz source exists, prefer a strong standalone route",
    );
    expect(skill).toContain(
      "Letting route registration, quiz-source absence, docs, or tests dominate",
    );
  });

  it("clarifies that small reviewable scope does not mean low ambition", () => {
    expect(skill).toContain(
      '"Small and reviewable" means avoid unrelated refactors and broad app churn',
    );
    expect(teamPreferences).toContain(
      '"small and reviewable" means keep changes scoped to the learning surface',
    );
    expect(agents).toContain('"small and reviewable" means keep the scope');
  });

  it("approves focused visual affordances and lucide-react for learning UX", () => {
    expect(normalizedSkill).toContain(
      "for this repo `lucide-react` is an approved focused runtime dependency",
    );
    expect(normalizedSkill).toContain(
      "Do not let the absence of an existing icon import push the page toward text-only panels.",
    );
    expect(teamPreferences).toContain(
      "Current approved learning-experience visual dependency: `lucide-react`",
    );
    expect(packageJson.dependencies?.["lucide-react"]).toBeDefined();
  });

  it("requires a richer central journey and UX review loop", () => {
    expect(skill).toContain(
      "Prefer a full staged experience over a shallow toggle panel",
    );
    expect(skill).toContain(
      "One click that swaps static prose is usually not enough",
    );
    expect(skill).toContain("Run a frontend UX review loop");
    expect(teamPreferences).toContain(
      "Learning-experience tests should cover more than route existence and one button click.",
    );
    expect(teamPreferences).toContain(
      "New learning experiences should run a frontend UX review loop before finalizing",
    );
  });

  it("keeps workflow lab layout aligned with learner play-through order", () => {
    expect(skill).toContain(
      "Make the visual scan order match the learner journey.",
    );
    expect(normalizedSkill).toContain(
      "do not place later conclusions beside earlier evidence",
    );
    expect(normalizedSkill).toContain(
      "the DOM order and visible top-left-to-right scan order reflect the intended play-through sequence",
    );
    expect(teamPreferences).toContain(
      "Workflow-based learning labs should make the visible scan order match the intended play-through order.",
    );
  });

  it("requires intermediate viewport review for dense learning layouts", () => {
    expect(skill).toContain("`1280 x 800`");
    expect(normalizedSkill).toContain(
      "Check for awkward text breaks, clipped controls, and page-level horizontal overflow at that size",
    );
    expect(teamPreferences).toContain(
      "Learning-page browser review should include realistic intermediate viewports",
    );
  });
});
