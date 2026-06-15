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
import CrashProbabilityL5LearningPage from "../../components/learning/pages/CrashProbabilityL5LearningPage";
import ClinicalTrialsL3LearningPage from "../../components/learning/pages/ClinicalTrialsL3LearningPage";
import StanfordCME295Lecture1LearningPage from "../../components/learning/pages/StanfordCME295Lecture1LearningPage";
import StanfordCME295Lecture2LearningPage from "../../components/learning/pages/StanfordCME295Lecture2LearningPage";
import StanfordCME295Lecture3LearningPage from "../../components/learning/pages/StanfordCME295Lecture3LearningPage";
import StanfordCME295Lecture4LearningPage from "../../components/learning/pages/StanfordCME295Lecture4LearningPage";
import StanfordCME295Lecture5LearningPage from "../../components/learning/pages/StanfordCME295Lecture5LearningPage";

const LEARNING_PAGE_COMPONENTS: Partial<
  Record<SourceId, ComponentType<{ experience: LearningExperience }>>
> = {
  "cme295-lect1": StanfordCME295Lecture1LearningPage,
  "cme295-lect2": StanfordCME295Lecture2LearningPage,
  "cme295-lect3": StanfordCME295Lecture3LearningPage,
  "cme295-lect4": StanfordCME295Lecture4LearningPage,
  "cme295-lect5": StanfordCME295Lecture5LearningPage,
  "crash-probability-l3": CrashProbabilityL3LearningPage,
  "crash-probability-l4": CrashProbabilityL4LearningPage,
  "crash-probability-l5": CrashProbabilityL5LearningPage,
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
