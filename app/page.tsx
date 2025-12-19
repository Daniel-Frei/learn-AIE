// app/page.tsx
"use client";

import { useQuiz } from "../lib/useQuiz";
import { getTitleForSelection } from "../lib/quiz";
import QuizHeader from "../components/QuizHeader";
import QuizQuestionSection from "../components/QuizQuestionSection";
import QuizFooter from "../components/QuizFooter";

export default function QuizPage() {
  const {
    selectedSources,
    difficultyRange,
    applySelection,
    availableCount,
    currentIndex,
    currentQuestion,
    currentDifficultyScore,
    shuffledOptions,
    selectedIndexes,
    showResult,
    toggleOption,
    submitAnswer,
    nextQuestion,
    answeredCount,
    correctCount,
    accuracy,
    exportDifficultyJson,
    importDifficultyFromJson,
  } = useQuiz();

  const title = getTitleForSelection(selectedSources);

  const difficultyPercent =
    currentDifficultyScore != null
      ? Math.round(currentDifficultyScore * 100)
      : null;

  const hasQuestion = availableCount > 0 && !!currentQuestion;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 flex items-center justify-center px-4 py-10">
      <div className="w-full max-w-3xl rounded-2xl bg-slate-900 shadow-xl border border-slate-800 p-6 md:p-8 space-y-6">
        <QuizHeader
          title={title}
          selectedSources={selectedSources}
          difficultyRange={difficultyRange}
          applySelection={applySelection}
          answeredCount={answeredCount}
          correctCount={correctCount}
          accuracy={accuracy}
          exportDifficultyJson={exportDifficultyJson}
          importDifficultyFromJson={importDifficultyFromJson}
        />

        <QuizQuestionSection
          availableCount={availableCount}
          currentIndex={currentIndex}
          difficultyPercent={difficultyPercent}
          currentQuestion={currentQuestion}
          shuffledOptions={shuffledOptions}
          selectedIndexes={selectedIndexes}
          showResult={showResult}
          toggleOption={toggleOption}
        />

        <QuizFooter
          hasQuestion={hasQuestion}
          currentQuestion={currentQuestion}
          shuffledOptions={shuffledOptions}
          selectedIndexes={selectedIndexes}
          showResult={showResult}
          submitAnswer={submitAnswer}
          nextQuestion={nextQuestion}
        />
      </div>
    </div>
  );
}
