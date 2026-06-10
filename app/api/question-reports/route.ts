import { NextRequest, NextResponse } from "next/server";
import type {
  SubmitQuestionReportRequest,
  SubmitQuestionReportResponse,
} from "@/lib/quizSync";
import { ALL_TOPICS } from "@/lib/questionTopics";
import { submitQuestionReportForParticipant } from "@/lib/server/quizDataService";

export const runtime = "nodejs";

const MAX_REPORT_COMMENT_CHARS = 2_000;
const MAX_REPORT_PROMPT_CHARS = 4_000;
const MAX_LABEL_CHARS = 200;
const VALID_TOPICS: ReadonlySet<string> = new Set(ALL_TOPICS);
const MALFORMED_JSON = Symbol("malformed-json");

function isBoundedString(value: unknown, maxLength: number): value is string {
  return (
    typeof value === "string" &&
    value.trim().length > 0 &&
    value.length <= maxLength
  );
}

function isValidQuestionReportPayload(
  body: unknown,
): body is SubmitQuestionReportRequest {
  if (!body || typeof body !== "object" || Array.isArray(body)) return false;

  const value = body as Partial<SubmitQuestionReportRequest>;
  const draft = value.draft;
  const snapshot = draft?.snapshot;

  return (
    isBoundedString(value.participantId, MAX_LABEL_CHARS) &&
    Boolean(draft) &&
    isBoundedString(draft?.questionId, MAX_LABEL_CHARS) &&
    isBoundedString(draft?.comment, MAX_REPORT_COMMENT_CHARS) &&
    Boolean(snapshot) &&
    isBoundedString(snapshot?.sourceId, MAX_LABEL_CHARS) &&
    isBoundedString(snapshot?.sourceLabel, MAX_LABEL_CHARS) &&
    isBoundedString(snapshot?.seriesId, MAX_LABEL_CHARS) &&
    isBoundedString(snapshot?.seriesLabel, MAX_LABEL_CHARS) &&
    isBoundedString(snapshot?.topic, MAX_LABEL_CHARS) &&
    VALID_TOPICS.has(snapshot?.topic ?? "") &&
    isBoundedString(snapshot?.prompt, MAX_REPORT_PROMPT_CHARS)
  );
}

async function readQuestionReportBody(
  req: NextRequest,
): Promise<unknown | typeof MALFORMED_JSON> {
  try {
    return (await req.json()) as unknown;
  } catch {
    return MALFORMED_JSON;
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await readQuestionReportBody(req);
    if (body === MALFORMED_JSON || !isValidQuestionReportPayload(body)) {
      return NextResponse.json(
        { error: "Invalid request payload for question report." },
        { status: 400 },
      );
    }

    const summary = await submitQuestionReportForParticipant(body);
    const response: SubmitQuestionReportResponse = {
      totalReportCount: summary.totalReportCount,
      questionReportCount: summary.countsByQuestion[body.draft.questionId] ?? 0,
    };
    return NextResponse.json(response);
  } catch (err) {
    console.error("Error in /api/question-reports:", err);
    return NextResponse.json(
      { error: "Failed to submit question report" },
      { status: 500 },
    );
  }
}
