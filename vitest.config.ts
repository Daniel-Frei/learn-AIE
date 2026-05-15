import path from "node:path";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "."),
    },
  },
  test: {
    include: ["tests/**/*.spec.ts", "tests/**/*.spec.tsx"],
    coverage: {
      provider: "v8",
      reporter: ["text"],
      include: [
        "app/api/**/*.ts",
        "apps/mobile/src/lib/mobileProfileState.ts",
        "lib/client/participantStorage.ts",
        "lib/difficultyStore.ts",
        "lib/llm/explain.ts",
        "lib/questionReportsStore.ts",
        "lib/quiz.ts",
        "lib/quizSession.ts",
        "lib/server/**/*.ts",
      ],
      thresholds: {
        statements: 95,
        branches: 95,
        functions: 95,
        lines: 95,
      },
    },
  },
});
