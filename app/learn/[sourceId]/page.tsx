import type { ComponentType } from "react";
import { notFound } from "next/navigation";
import {
  LEARNING_EXPERIENCES,
  getLearningExperience,
  type LearningExperience,
} from "../../../lib/learning";
import type { SourceId } from "../../../lib/quiz";
import CrashProbabilityL3LearningPage from "../../../components/learning/pages/CrashProbabilityL3LearningPage";
import ClinicalTrialsL3LearningPage from "../../../components/learning/pages/ClinicalTrialsL3LearningPage";
import StanfordCME295Lecture1LearningPage from "../../../components/learning/pages/StanfordCME295Lecture1LearningPage";

const LEARNING_PAGE_COMPONENTS: Partial<
  Record<SourceId, ComponentType<{ experience: LearningExperience }>>
> = {
  "cme295-lect1": StanfordCME295Lecture1LearningPage,
  "crash-probability-l3": CrashProbabilityL3LearningPage,
  "clinical-trials-l3": ClinicalTrialsL3LearningPage,
};

export function generateStaticParams() {
  return LEARNING_EXPERIENCES.map((experience) => ({
    sourceId: experience.sourceId,
  }));
}

type LearningPageProps = {
  params: Promise<{
    sourceId: string;
  }>;
};

export default async function LearningExperiencePage({
  params,
}: LearningPageProps) {
  const { sourceId } = await params;
  const experience = getLearningExperience(sourceId);
  if (!experience) notFound();

  const PageComponent = LEARNING_PAGE_COMPONENTS[experience.sourceId];
  if (!PageComponent) notFound();

  return <PageComponent experience={experience} />;
}
