import {
  LEARNING_EXPERIENCES,
  getLearningExperienceCourse,
} from "../../../../lib/learning";
import LearningExperienceRoute from "../../LearningExperienceRoute";

export function generateStaticParams() {
  return LEARNING_EXPERIENCES.flatMap((experience) => {
    const course = getLearningExperienceCourse(experience);
    if (!course) return [];

    return [
      {
        sourceId: course.seriesId,
        learningSourceId: experience.sourceId,
      },
    ];
  });
}

type LearningPageProps = {
  params: Promise<{
    sourceId: string;
    learningSourceId: string;
  }>;
};

export default async function LearningExperienceNestedPage({
  params,
}: LearningPageProps) {
  const { sourceId, learningSourceId } = await params;

  return (
    <LearningExperienceRoute sourceId={learningSourceId} seriesId={sourceId} />
  );
}
