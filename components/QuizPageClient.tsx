"use client";

import { useMemo } from "react";
import { useQuiz } from "../lib/useQuiz";
import { getTitleForSelection, type SourceId } from "../lib/quiz";
import QuizHeader from "./QuizHeader";
import QuizQuestionSection from "./QuizQuestionSection";
import QuizFooter from "./QuizFooter";

type Props = {
  initialSource?: SourceId | null;
};

export default function QuizPageClient({ initialSource = null }: Props) {
  const initialSelection = useMemo(
    () => (initialSource ? { sources: [initialSource] } : undefined),
    [initialSource],
  );

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
  } = useQuiz(initialSelection);

  const title = getTitleForSelection(selectedSources, selectedTopics);

  const hasQuestion = availableCount > 0 && !!currentQuestion;

  return (
    <div className="min-h-[calc(100vh-4.25rem)] bg-slate-950 px-4 py-10 text-slate-50 flex items-center justify-center">
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
          defaultSelectorOpen={!initialSource}
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
