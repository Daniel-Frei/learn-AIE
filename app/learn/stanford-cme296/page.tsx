import { notFound } from "next/navigation";

import LearningCoursePage from "../LearningCoursePage";
import { getStandaloneLearningCourse } from "../standaloneLearningPages";

export default function StanfordCME296LearningCoursePage() {
  const course = getStandaloneLearningCourse("stanford-cme296");
  if (!course) notFound();

  return <LearningCoursePage course={course} />;
}
