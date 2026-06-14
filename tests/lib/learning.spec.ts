import { describe, expect, it } from "vitest";
import {
  LEARNING_EXPERIENCES,
  getDuplicateLearningExperienceSourceIds,
  getLearningCourse,
  getLearningCoursePath,
  getLearningCourses,
  getLearningExperience,
  getLearningExperiencePath,
  getLearningExperienceSequenceLabel,
  getQuestionSourceForLearningExperience,
  getQuestionSourceSequenceLabel,
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

    const cmeLecture3Experience = getLearningExperience("cme295-lect3");
    expect(cmeLecture3Experience?.title).toContain(
      "Large Language Models, MoE & Inference",
    );
    expect(cmeLecture3Experience?.sourceMaterialPath).toContain(
      "Stanford CME295 Transformers & LLMs/transcripts-and-files/lecture 3 - transcript.md",
    );

    const cmeLecture4Experience = getLearningExperience("cme295-lect4");
    expect(cmeLecture4Experience?.title).toContain(
      "LLM Training, Scaling & Alignment",
    );
    expect(cmeLecture4Experience?.sourceMaterialPath).toContain(
      "Stanford CME295 Transformers & LLMs/transcripts-and-files/lecture 4 - transcript.md",
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

    const probabilityL5Experience = getLearningExperience(
      "crash-probability-l5",
    );
    expect(probabilityL5Experience?.title).toContain(
      "Sampling, Latent Variables, and Diffusion Models",
    );
    expect(probabilityL5Experience?.sourceMaterialPath).toContain(
      "Probability/transcripts-and-files/Lecture 5 - overview.md",
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
      expect.arrayContaining([
        "crash-probability-l3",
        "crash-probability-l4",
        "crash-probability-l5",
      ]),
    );
    expect(probabilityCourse?.totalDurationMinutes).toBeGreaterThan(0);

    const stanfordCourse = getLearningCourse("stanford-cme295");
    expect(
      stanfordCourse?.experiences.map((experience) => experience.sourceId),
    ).toEqual(
      expect.arrayContaining([
        "cme295-lect1",
        "cme295-lect2",
        "cme295-lect3",
        "cme295-lect4",
      ]),
    );

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

    const probabilityL5Experience = getLearningExperience(
      "crash-probability-l5",
    );
    expect(probabilityL5Experience).not.toBeNull();
    if (probabilityL5Experience) {
      expect(getLearningExperiencePath(probabilityL5Experience)).toBe(
        "/learn/crash-course-probability/crash-probability-l5",
      );
    }

    const cmeLecture4Experience = getLearningExperience("cme295-lect4");
    expect(cmeLecture4Experience).not.toBeNull();
    if (cmeLecture4Experience) {
      expect(getLearningExperiencePath(cmeLecture4Experience)).toBe(
        "/learn/stanford-cme295/cme295-lect4",
      );
    }
  });

  it("derives lecture and chapter labels for course cards", () => {
    const stanfordLecture3 = getLearningExperience("cme295-lect3");
    expect(stanfordLecture3).not.toBeNull();
    if (stanfordLecture3) {
      expect(getLearningExperienceSequenceLabel(stanfordLecture3)).toBe(
        "Lecture 3",
      );
    }

    const stanfordLecture4 = getLearningExperience("cme295-lect4");
    expect(stanfordLecture4).not.toBeNull();
    if (stanfordLecture4) {
      expect(getLearningExperienceSequenceLabel(stanfordLecture4)).toBe(
        "Lecture 4",
      );
    }

    const probabilityLecture5 = getLearningExperience("crash-probability-l5");
    expect(probabilityLecture5).not.toBeNull();
    if (probabilityLecture5) {
      expect(getLearningExperienceSequenceLabel(probabilityLecture5)).toBe(
        "Lecture 5",
      );
    }

    expect(
      getQuestionSourceSequenceLabel({
        id: "chapter-5",
        label: "Chapter 5 only",
        title: "Chapter 5 Quiz - Large Language Models",
      }),
    ).toBe("Chapter 5");
    expect(
      getQuestionSourceSequenceLabel({
        id: "aie-build-app-ch2",
        label: "AIE build app - Chap 2",
        title: "AIE Building Apps Chapter 2",
      }),
    ).toBe("Chapter 2");
  });

  it("sorts course learning experiences by registered source order", () => {
    const stanfordCourse = getLearningCourse("stanford-cme295");
    expect(
      stanfordCourse?.experiences.map((experience) => experience.sourceId),
    ).toEqual(["cme295-lect1", "cme295-lect2", "cme295-lect3", "cme295-lect4"]);
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
    expect(parseQuizSourceParam("cme295-lect3")).toBe("cme295-lect3");
    expect(parseQuizSourceParam("cme295-lect4")).toBe("cme295-lect4");
    expect(parseQuizSourceParam("crash-probability-l3")).toBe(
      "crash-probability-l3",
    );
    expect(parseQuizSourceParam("crash-probability-l4")).toBe(
      "crash-probability-l4",
    );
    expect(parseQuizSourceParam("crash-probability-l5")).toBe(
      "crash-probability-l5",
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
