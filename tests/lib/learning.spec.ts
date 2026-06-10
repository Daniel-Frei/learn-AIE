import { describe, expect, it } from "vitest";
import {
  LEARNING_EXPERIENCES,
  getDuplicateLearningExperienceSourceIds,
  getLearningCourse,
  getLearningCoursePath,
  getLearningCourses,
  getLearningExperience,
  getLearningExperiencePath,
  getQuestionSourceForLearningExperience,
  getUnknownLearningExperienceSourceIds,
} from "@/lib/learning";
import { QUESTION_SOURCES } from "@/lib/quiz";
import { parseQuizSourceParam } from "@/lib/quizRoute";

describe("learning experience registry", () => {
  it("registers learning experiences against known quiz sources", () => {
    expect(LEARNING_EXPERIENCES.length).toBeGreaterThan(0);
    expect(getUnknownLearningExperienceSourceIds(LEARNING_EXPERIENCES)).toEqual(
      [],
    );

    for (const experience of LEARNING_EXPERIENCES) {
      const source = getQuestionSourceForLearningExperience(experience);
      expect(source, `${experience.sourceId} should resolve`).not.toBeNull();
      expect(source?.questions.length).toBeGreaterThan(0);
    }
  });

  it("keeps one learning experience per source id", () => {
    expect(
      getDuplicateLearningExperienceSourceIds(LEARNING_EXPERIENCES),
    ).toEqual([]);
    expect(
      getDuplicateLearningExperienceSourceIds([
        { sourceId: "crash-probability-l3" },
        { sourceId: "crash-probability-l3" },
        { sourceId: "mit6s191-l3" },
      ]),
    ).toEqual(["crash-probability-l3"]);
  });

  it("finds registered and missing learning experiences", () => {
    const cmeExperience = getLearningExperience("cme295-lect1");
    expect(cmeExperience?.title).toContain("Stanford CME295 Lecture 1");
    expect(cmeExperience?.sourceMaterialPath).toContain(
      "Stanford CME295 Transformers & LLMs/transcripts-and-files/lecture1_transformers.md",
    );

    const cmeLecture2Experience = getLearningExperience("cme295-lect2");
    expect(cmeLecture2Experience?.title).toContain(
      "Transformer-Based Models & Tricks",
    );
    expect(cmeLecture2Experience?.sourceMaterialPath).toContain(
      "Stanford CME295 Transformers & LLMs/transcripts-and-files/lecture 2 - transcript.md",
    );

    const probabilityExperience = getLearningExperience("crash-probability-l3");
    expect(probabilityExperience?.title).toContain("Likelihood, Loss, Softmax");

    const probabilityL4Experience = getLearningExperience(
      "crash-probability-l4",
    );
    expect(probabilityL4Experience?.title).toContain("Probability Over Time");
    expect(probabilityL4Experience?.sourceMaterialPath).toContain(
      "Probability/transcripts-and-files/Lecture 4 - overview.md",
    );

    const clinicalTrialsExperience =
      getLearningExperience("clinical-trials-l3");
    expect(clinicalTrialsExperience?.title).toContain(
      "Statistics and Evidence Interpretation",
    );
    expect(clinicalTrialsExperience?.sourceMaterialPath).toContain(
      "Clinical Trials/transcripts-and-files/Lecture 3 - overview.md",
    );

    expect(getLearningExperience("missing-source")).toBeNull();
  });

  it("groups learning experiences by course and builds canonical paths", () => {
    const courses = getLearningCourses();
    expect(courses.map((course) => course.label)).toEqual(
      expect.arrayContaining([
        "Stanford CME295 Transformers & LLMs",
        "Crash Course Probability",
        "Clinical Trials Crash Course",
      ]),
    );

    const probabilityCourse = getLearningCourse("crash-course-probability");
    expect(
      probabilityCourse?.experiences.map((experience) => experience.sourceId),
    ).toEqual(
      expect.arrayContaining(["crash-probability-l3", "crash-probability-l4"]),
    );
    expect(probabilityCourse?.totalDurationMinutes).toBeGreaterThan(0);

    expect(getLearningCourse("missing-course")).toBeNull();
    expect(getLearningCoursePath("crash-course-probability")).toBe(
      "/learn/crash-course-probability",
    );

    const probabilityExperience = getLearningExperience("crash-probability-l3");
    expect(probabilityExperience).not.toBeNull();
    if (probabilityExperience) {
      expect(getLearningExperiencePath(probabilityExperience)).toBe(
        "/learn/crash-course-probability/crash-probability-l3",
      );
    }
  });

  it("detects unknown learning source ids", () => {
    const knownSourceId = QUESTION_SOURCES[0].id;
    expect(
      getUnknownLearningExperienceSourceIds([
        { sourceId: knownSourceId },
        { sourceId: "not-registered" },
      ]),
    ).toEqual(["not-registered"]);
  });
});

describe("quiz source query parsing", () => {
  it("accepts a known source id from the source query parameter", () => {
    expect(parseQuizSourceParam("cme295-lect1")).toBe("cme295-lect1");
    expect(parseQuizSourceParam("cme295-lect2")).toBe("cme295-lect2");
    expect(parseQuizSourceParam("crash-probability-l3")).toBe(
      "crash-probability-l3",
    );
    expect(parseQuizSourceParam("crash-probability-l4")).toBe(
      "crash-probability-l4",
    );
    expect(parseQuizSourceParam("clinical-trials-l3")).toBe(
      "clinical-trials-l3",
    );
  });

  it("uses the first array value and rejects missing or unknown values", () => {
    expect(parseQuizSourceParam(["crash-probability-l3", "mit6s191-l3"])).toBe(
      "crash-probability-l3",
    );
    expect(parseQuizSourceParam("not-registered")).toBeNull();
    expect(parseQuizSourceParam(undefined)).toBeNull();
    expect(parseQuizSourceParam(null)).toBeNull();
  });
});
