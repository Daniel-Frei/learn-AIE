import { NextRequest, NextResponse } from "next/server";
import type {
  SubmitQuestionReportRequest,
  SubmitQuestionReportResponse,
} from "@/lib/quizSync";
import { submitQuestionReportForParticipant } from "@/lib/server/quizDataService";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as SubmitQuestionReportRequest;
    if (
      !body ||
      !body.participantId?.trim() ||
      !body.draft?.questionId?.trim() ||
      !body.draft?.comment?.trim() ||
      !body.draft?.snapshot
    ) {
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
