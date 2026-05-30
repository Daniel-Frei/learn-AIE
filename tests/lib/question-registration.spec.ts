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

function extractLastQuestionReportTopicConstraint(): string[] | null {
  const sql = listSqlMigrationFiles()
    .map((filePath) => fs.readFileSync(filePath, "utf8"))
    .join("\n");
  const topicCheckPattern =
    /(?:topic\s+text\s+not\s+null\s+check|question_reports_topic_check[\s\S]*?check)\s*\(\s*topic\s+in\s*\(([^)]*)\)\s*\)/gi;
  let allowedTopics: string[] | null = null;
  let match: RegExpExecArray | null;

  while ((match = topicCheckPattern.exec(sql))) {
    allowedTopics = Array.from(match[1].matchAll(/'([^']+)'/g)).map(
      (topicMatch) => topicMatch[1],
    );
  }

  return allowedTopics;
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

  it("keeps report topic database constraints aligned with registered topics", () => {
    const allowedTopics = extractLastQuestionReportTopicConstraint();
    const registeredTopics = Array.from(
      new Set(QUESTION_SOURCES.map((source) => source.topic)),
    ).sort();
    const missingRegisteredTopics = registeredTopics.filter(
      (topic) => !allowedTopics?.includes(topic),
    );
    const missingKnownTopics = ALL_TOPICS.filter(
      (topic) => !allowedTopics?.includes(topic),
    );

    expect(allowedTopics).not.toBeNull();
    expect(
      missingRegisteredTopics,
      "Registered source topics must be accepted by question_reports.topic.",
    ).toEqual([]);
    expect(
      missingKnownTopics,
      "ALL_TOPICS must be accepted by question_reports.topic.",
    ).toEqual([]);
  });
});
