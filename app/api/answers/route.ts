import { NextRequest, NextResponse } from "next/server";
import type { RecordAnswerRequest } from "@/lib/quizSync";
import { recordQuizAnswerForParticipant } from "@/lib/server/quizDataService";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as RecordAnswerRequest;
    if (
      !body ||
      !body.participantId?.trim() ||
      !body.questionId?.trim() ||
      typeof body.isCorrect !== "boolean" ||
      (body.elapsedMs !== undefined &&
        (!Number.isInteger(body.elapsedMs) || body.elapsedMs < 0)) ||
      (body.mistakeCount !== undefined &&
        (!Number.isInteger(body.mistakeCount) || body.mistakeCount < 0))
    ) {
      return NextResponse.json(
        { error: "Invalid request payload for answer submission." },
        { status: 400 },
      );
    }

    const response = await recordQuizAnswerForParticipant(body);
    return NextResponse.json(response);
  } catch (err) {
    console.error("Error in /api/answers:", err);
    return NextResponse.json(
      { error: "Failed to record answer" },
      { status: 500 },
    );
  }
}
