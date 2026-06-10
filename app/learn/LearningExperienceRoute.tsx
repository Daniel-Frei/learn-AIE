import type { ComponentType } from "react";
import { notFound } from "next/navigation";
import {
  getLearningExperience,
  getQuestionSourceForLearningExperience,
  type LearningExperience,
} from "../../lib/learning";
import type { SourceId } from "../../lib/quiz";
import CrashProbabilityL3LearningPage from "../../components/learning/pages/CrashProbabilityL3LearningPage";
import CrashProbabilityL4LearningPage from "../../components/learning/pages/CrashProbabilityL4LearningPage";
import ClinicalTrialsL3LearningPage from "../../components/learning/pages/ClinicalTrialsL3LearningPage";
import StanfordCME295Lecture1LearningPage from "../../components/learning/pages/StanfordCME295Lecture1LearningPage";
import StanfordCME295Lecture2LearningPage from "../../components/learning/pages/StanfordCME295Lecture2LearningPage";

const LEARNING_PAGE_COMPONENTS: Partial<
  Record<SourceId, ComponentType<{ experience: LearningExperience }>>
> = {
  "cme295-lect1": StanfordCME295Lecture1LearningPage,
  "cme295-lect2": StanfordCME295Lecture2LearningPage,
  "crash-probability-l3": CrashProbabilityL3LearningPage,
  "crash-probability-l4": CrashProbabilityL4LearningPage,
  "clinical-trials-l3": ClinicalTrialsL3LearningPage,
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
