import { notFound } from "next/navigation";

import { getLearningCourse } from "../../../lib/learning";
import LearningCoursePage from "../LearningCoursePage";

export default function AiAgentsLearningCoursePage() {
  const course = getLearningCourse("ai-agents");
  if (!course) notFound();

  return <LearningCoursePage course={course} />;
}
