import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, "../..");
const libRoot = path.join(repoRoot, "lib");
const quizFilePath = path.join(libRoot, "quiz.ts");

const NON_QUESTION_FILES = new Set([
  "difficultyStore.ts",
  "normalizeMath.ts",
  "quiz.ts",
  "useQuiz.ts",
]);

function listTsFilesRecursively(dir: string): string[] {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  return entries.flatMap((entry) => {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      return listTsFilesRecursively(fullPath);
    }
    return fullPath.endsWith(".ts") ? [fullPath] : [];
  });
}

function isQuestionFile(filePath: string): boolean {
  if (NON_QUESTION_FILES.has(path.basename(filePath))) return false;
  const content = fs.readFileSync(filePath, "utf8");
  return /export\s+const\s+\w+\s*:\s*Question\[\]/.test(content);
}

describe("question file registration", () => {
  it("ensures all question files in lib/ are registered in quiz.ts", () => {
    const quizContent = fs.readFileSync(quizFilePath, "utf8");
    const allTsFiles = listTsFilesRecursively(libRoot);
    const questionFiles = allTsFiles.filter(isQuestionFile);

    const missing = questionFiles.filter((filePath) => {
      const relativeFromLib = path.relative(libRoot, filePath);
      const importPath = `./${relativeFromLib.replace(/\\/g, "/").replace(/\.ts$/, "")}`;
      return (
        !quizContent.includes(`"${importPath}"`) &&
        !quizContent.includes(`'${importPath}'`)
      );
    });

    expect(
      missing,
      `Unregistered question files in lib/: ${missing.join(", ")}`,
    ).toEqual([]);
  });
});
