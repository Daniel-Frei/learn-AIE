import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { ALL_TOPICS, QUESTION_SOURCES } from "@/lib/quiz";

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, "../..");
const libRoot = path.join(repoRoot, "lib");
const supabaseMigrationsRoot = path.join(repoRoot, "supabase", "migrations");
const quizFilePath = path.join(libRoot, "quiz.ts");
const MIN_EXPLANATION_CHARS = 150;

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

function listSqlMigrationFiles(): string[] {
  return fs
    .readdirSync(supabaseMigrationsRoot)
    .filter((fileName) => fileName.endsWith(".sql"))
    .sort()
    .map((fileName) => path.join(supabaseMigrationsRoot, fileName));
}

function readAllSqlMigrations(): string {
  return listSqlMigrationFiles()
    .map((filePath) => fs.readFileSync(filePath, "utf8"))
    .join("\n");
}

function findLastMatchIndex(source: string, pattern: RegExp): number {
  let lastIndex = -1;
  for (const match of source.matchAll(pattern)) {
    lastIndex = match.index ?? lastIndex;
  }
  return lastIndex;
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

  it("keeps registered question explanations longer than 150 characters", () => {
    const shortExplanations = QUESTION_SOURCES.flatMap((source) =>
      source.questions.flatMap((question) => {
        const length = question.explanation.trim().length;
        return length > MIN_EXPLANATION_CHARS
          ? []
          : [`${source.id}/${question.id}: ${length} characters`];
      }),
    );

    expect(
      shortExplanations,
      `Question explanations must be longer than ${MIN_EXPLANATION_CHARS} characters.`,
    ).toEqual([]);
  });

  it("keeps registered source topics listed in ALL_TOPICS", () => {
    const knownTopics = new Set<string>(ALL_TOPICS);
    const missingRegisteredTopics = Array.from(
      new Set(QUESTION_SOURCES.map((source) => source.topic)),
    )
      .filter((topic) => !knownTopics.has(topic))
      .sort();

    expect(
      missingRegisteredTopics,
      "Registered source topics must be added to ALL_TOPICS for API validation and UI filters.",
    ).toEqual([]);
  });

  it("keeps question report topic storage open to future topics", () => {
    const sql = readAllSqlMigrations();
    const lastClosedEnumDropIndex = findLastMatchIndex(
      sql,
      /drop\s+constraint\s+if\s+exists\s+question_reports_topic_check/gi,
    );
    const effectiveSql =
      lastClosedEnumDropIndex >= 0 ? sql.slice(lastClosedEnumDropIndex) : sql;

    expect(
      lastClosedEnumDropIndex,
      "Migrations should drop the old closed question_reports.topic enum constraint.",
    ).toBeGreaterThanOrEqual(0);
    expect(
      effectiveSql,
      "Do not reintroduce topic-specific question_reports.topic enums; API validation owns the current topic list.",
    ).not.toMatch(/\btopic\s+in\s*\(/i);
    expect(effectiveSql).toMatch(
      /question_reports_topic_present_check[\s\S]*char_length\s*\(\s*btrim\s*\(\s*topic\s*\)\s*\)\s+between\s+1\s+and\s+200/i,
    );
  });
});
