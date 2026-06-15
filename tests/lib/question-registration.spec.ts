import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  ALL_TOPICS,
  QUESTION_SOURCES,
  QUESTION_TYPES,
  getQuestionType,
} from "@/lib/quiz";

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, "../..");
const libRoot = path.join(repoRoot, "lib");
const supabaseMigrationsRoot = path.join(repoRoot, "supabase", "migrations");
const quizFilePath = path.join(libRoot, "quiz.ts");
const cme295Lecture3FilePath = path.join(
  libRoot,
  "lectures",
  "Stanford CME295 Transformers & LLMs",
  "lecture3_LLMs.ts",
);
const cme295Lecture4FilePath = path.join(
  libRoot,
  "lectures",
  "Stanford CME295 Transformers & LLMs",
  "lecture4_training.ts",
);
const cme295Lecture5FilePath = path.join(
  libRoot,
  "lectures",
  "Stanford CME295 Transformers & LLMs",
  "lecture5_preference_tuning.ts",
);
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

function getQuestionArraySource(filePath: string, exportName: string) {
  const content = fs.readFileSync(filePath, "utf8");
  const exportIndex = content.indexOf(`export const ${exportName}`);
  expect(
    exportIndex,
    `${exportName} should be exported`,
  ).toBeGreaterThanOrEqual(0);

  return content.slice(exportIndex);
}

function getHardcodedHelperQuestionIds(filePath: string, exportName: string) {
  const questionArraySource = getQuestionArraySource(filePath, exportName);
  const helperCalls = [
    ...questionArraySource.matchAll(/make(?:AssertionReason)?Question\(/g),
  ];
  const hardcodedIdCalls = [
    ...questionArraySource.matchAll(
      /make(?:AssertionReason)?Question\(\s*\n\s*"([^"]+)"/g,
    ),
  ];

  expect(hardcodedIdCalls).toHaveLength(helperCalls.length);
  return hardcodedIdCalls.map((match) => match[1]);
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

  it("keeps registered questions aligned with their question type contract", () => {
    const knownQuestionTypes = new Set<string>(QUESTION_TYPES);
    const violations = QUESTION_SOURCES.flatMap((source) =>
      source.questions.flatMap((question) => {
        const messages: string[] = [];
        const questionType = getQuestionType(question);
        const correctCount = question.options.filter(
          (option) => option.isCorrect,
        ).length;

        if (!knownQuestionTypes.has(questionType)) {
          messages.push(`unknown type ${questionType}`);
        }

        if (questionType === "assertion-reason") {
          const expectedOptions = [
            "Assertion is true, Reason is false.",
            "Assertion is false, Reason is true.",
            "Both are false.",
            "Both are true, and the Reason is the correct explanation of the Assertion.",
            "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
          ];
          const hasFixedOptions = expectedOptions.every(
            (text, index) => question.options[index]?.text === text,
          );
          if (correctCount !== 1) {
            messages.push(`has ${correctCount} correct options`);
          }
          if (question.options.length !== expectedOptions.length) {
            messages.push(`has ${question.options.length} options`);
          }
          if (!hasFixedOptions) {
            messages.push("does not use fixed assertion-reason option order");
          }
          if (!/^Assertion:[\s\S]+\n\nReason:[\s\S]+$/.test(question.prompt)) {
            messages.push(
              "does not use Assertion/blank-line/Reason prompt format",
            );
          }
          if (/which option correctly evaluates/i.test(question.prompt)) {
            messages.push("includes redundant assertion-reason instruction");
          }
        } else {
          if (question.options.length !== 4) {
            messages.push(`has ${question.options.length} options`);
          }
          if (correctCount < 1 || correctCount > 4) {
            messages.push(`has ${correctCount} correct options`);
          }
        }

        return messages.map(
          (message) => `${source.id}/${question.id}: ${message}`,
        );
      }),
    );

    expect(violations).toEqual([]);
  });

  it("keeps rewritten CME295 Lecture 3 question IDs explicit and off the legacy ID range", () => {
    const ids = getHardcodedHelperQuestionIds(
      cme295Lecture3FilePath,
      "stanfordCME295Lecture3LLMsQuestions",
    );
    const registeredSource = QUESTION_SOURCES.find(
      (source) => source.id === "cme295-lect3",
    );
    const registeredIds =
      registeredSource?.questions.map((question) => question.id) ?? [];
    const reusedLegacyIds = registeredIds.filter((id) =>
      /^cme295-lect3-q(?:0[1-9]|[1-9][0-9]|100)$/.test(id),
    );

    expect(new Set(ids).size).toBe(ids.length);
    expect(registeredSource).toBeDefined();
    expect(registeredIds).toEqual(ids);
    expect(registeredIds).toHaveLength(80);
    expect(registeredIds[0]).toBe("cme295-lect3-q101");
    expect(registeredIds[registeredIds.length - 1]).toBe("cme295-lect3-q180");
    expect(reusedLegacyIds).toEqual([]);
  });

  it("keeps rewritten CME295 Lecture 4 question IDs explicit and off the legacy ID range", () => {
    const ids = getHardcodedHelperQuestionIds(
      cme295Lecture4FilePath,
      "stanfordCME295Lecture4TrainingQuestions",
    );
    const reusedLegacyIds = ids.filter((id) =>
      /^cme295-lect4-q(?:0[1-9]|[1-9][0-9]|100)$/.test(id),
    );

    expect(new Set(ids).size).toBe(ids.length);
    expect(reusedLegacyIds).toEqual([]);
  });

  it("keeps CME295 Lecture 5 question IDs explicit in helper calls", () => {
    const ids = getHardcodedHelperQuestionIds(
      cme295Lecture5FilePath,
      "stanfordCME295Lecture5PreferenceTuningQuestions",
    );
    const registeredSource = QUESTION_SOURCES.find(
      (source) => source.id === "cme295-lect5",
    );
    const registeredIds =
      registeredSource?.questions.map((question) => question.id) ?? [];

    expect(new Set(ids).size).toBe(ids.length);
    expect(registeredSource).toBeDefined();
    expect(registeredIds).toEqual(ids);
    expect(registeredIds).toHaveLength(60);
    expect(registeredIds[0]).toBe("cme295-lect5-q01");
    expect(registeredIds[registeredIds.length - 1]).toBe("cme295-lect5-q60");
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
