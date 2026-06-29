import type { ComponentType } from "react";
import { notFound } from "next/navigation";
import {
  getLearningExperience,
  getQuestionSourceForLearningExperience,
  type LearningExperience,
} from "../../lib/learning";
import type { SourceId } from "../../lib/quiz";
import AgentNativeMemoryLearningPage from "../../components/learning/pages/AgentNativeMemoryLearningPage";
import AtomMemLearningPage from "../../components/learning/pages/AtomMemLearningPage";
import CrashProbabilityL3LearningPage from "../../components/learning/pages/CrashProbabilityL3LearningPage";
import CrashProbabilityL4LearningPage from "../../components/learning/pages/CrashProbabilityL4LearningPage";
import CrashProbabilityL5LearningPage from "../../components/learning/pages/CrashProbabilityL5LearningPage";
import ClinicalTrialsL3LearningPage from "../../components/learning/pages/ClinicalTrialsL3LearningPage";
import MemorySurveyLearningPage from "../../components/learning/pages/MemorySurveyLearningPage";
import StanfordCME295Lecture1LearningPage from "../../components/learning/pages/StanfordCME295Lecture1LearningPage";
import StanfordCME295Lecture2LearningPage from "../../components/learning/pages/StanfordCME295Lecture2LearningPage";
import StanfordCME295Lecture3LearningPage from "../../components/learning/pages/StanfordCME295Lecture3LearningPage";
import StanfordCME295Lecture4LearningPage from "../../components/learning/pages/StanfordCME295Lecture4LearningPage";
import StanfordCME295Lecture5LearningPage from "../../components/learning/pages/StanfordCME295Lecture5LearningPage";
import StanfordCME295Lecture9SynthesisPage from "../../components/learning/pages/StanfordCME295Lecture9SynthesisPage";

const LEARNING_PAGE_COMPONENTS: Partial<
  Record<SourceId, ComponentType<{ experience: LearningExperience }>>
> = {
  "cme295-lect1": StanfordCME295Lecture1LearningPage,
  "cme295-lect2": StanfordCME295Lecture2LearningPage,
  "cme295-lect3": StanfordCME295Lecture3LearningPage,
  "cme295-lect4": StanfordCME295Lecture4LearningPage,
  "cme295-lect5": StanfordCME295Lecture5LearningPage,
  "cme295-lect9": StanfordCME295Lecture9SynthesisPage,
  "crash-probability-l3": CrashProbabilityL3LearningPage,
  "crash-probability-l4": CrashProbabilityL4LearningPage,
  "crash-probability-l5": CrashProbabilityL5LearningPage,
  "clinical-trials-l3": ClinicalTrialsL3LearningPage,
  "ai-agents-memory-survey": MemorySurveyLearningPage,
  "ai-agents-agent-native-memory": AgentNativeMemoryLearningPage,
  "ai-agents-atommem": AtomMemLearningPage,
};

type LearningExperienceRouteProps = {
  sourceId: string;
  seriesId?: string;
};

export default function LearningExperienceRoute({
  sourceId,
  seriesId,
}: LearningExperienceRouteProps) {
  const experience = getLearningExperience(sourceId);
  if (!experience) notFound();

  if (seriesId) {
    const source = getQuestionSourceForLearningExperience(experience);
    if (source?.seriesId !== seriesId) notFound();
  }

  const PageComponent = LEARNING_PAGE_COMPONENTS[experience.sourceId];
  if (!PageComponent) notFound();

  return <PageComponent experience={experience} />;
}
