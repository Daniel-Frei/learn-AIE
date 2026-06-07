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
    selectedTopics,
    selectedQuestionTypes,
    selectionMode,
    difficultyRange,
    applySelection,
    availableCount,
    currentIndex,
    currentQuestion,
    currentQuestionContext,
    currentQuestionRating,
    questionElapsedMs,
    shuffledOptions,
    selectedIndexes,
    showResult,
    toggleOption,
    submitAnswer,
    nextQuestion,
    accuracy,
    userRating,
    userRatingRd,
    userRatingDelta,
    questionRatingDelta,
    submitQuestionReport,
    resetParticipantRating,
  } = useQuiz();

  const title = getTitleForSelection(selectedSources, selectedTopics);

  const hasQuestion = availableCount > 0 && !!currentQuestion;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 flex items-center justify-center px-4 py-10">
      <div className="w-full max-w-3xl rounded-2xl bg-slate-900 shadow-xl border border-slate-800 p-6 md:p-8 space-y-6">
        <QuizHeader
          title={title}
          selectedSources={selectedSources}
          selectedTopics={selectedTopics}
          selectedQuestionTypes={selectedQuestionTypes}
          selectionMode={selectionMode}
          difficultyRange={difficultyRange}
          applySelection={applySelection}
          accuracy={accuracy}
          userRating={userRating}
          userRatingRd={userRatingRd}
          userRatingDelta={userRatingDelta}
          resetParticipantRating={resetParticipantRating}
        />

        <QuizQuestionSection
          availableCount={availableCount}
          currentIndex={currentIndex}
          questionRating={currentQuestionRating}
          questionRatingDelta={questionRatingDelta}
          questionElapsedMs={questionElapsedMs}
          questionContext={currentQuestionContext}
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
          submitQuestionReport={submitQuestionReport}
        />
      </div>
    </div>
  );
}
