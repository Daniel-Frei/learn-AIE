import { notFound } from "next/navigation";
import {
  LEARNING_EXPERIENCES,
  getLearningCourse,
  getLearningCourses,
  getLearningExperience,
} from "../../../lib/learning";
import LearningCoursePage from "../LearningCoursePage";
import LearningExperienceRoute from "../LearningExperienceRoute";

export function generateStaticParams() {
  return [
    ...getLearningCourses().map((course) => ({
      sourceId: course.seriesId,
    })),
    ...LEARNING_EXPERIENCES.map((experience) => ({
      sourceId: experience.sourceId,
    })),
  ];
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

  const course = getLearningCourse(sourceId);
  if (course) {
    return <LearningCoursePage course={course} />;
  }

  const experience = getLearningExperience(sourceId);
  if (experience) {
    return <LearningExperienceRoute sourceId={sourceId} />;
  }

  notFound();
}
