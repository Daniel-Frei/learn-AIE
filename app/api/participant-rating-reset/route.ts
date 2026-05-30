import { NextRequest, NextResponse } from "next/server";
import type {
  ResetParticipantRatingRequest,
  ResetParticipantRatingResponse,
} from "@/lib/quizSync";
import { resetParticipantRatingForParticipant } from "@/lib/server/quizDataService";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as ResetParticipantRatingRequest;
    const participantId = body?.participantId?.trim();
    if (!participantId) {
      return NextResponse.json(
        { error: "participantId is required for rating reset." },
        { status: 400 },
      );
    }

    const response: ResetParticipantRatingResponse =
      await resetParticipantRatingForParticipant(participantId);
    return NextResponse.json(response);
  } catch (err) {
    if (err instanceof SyntaxError) {
      return NextResponse.json(
        { error: "participantId is required for rating reset." },
        { status: 400 },
      );
    }

    console.error("Error in /api/participant-rating-reset:", err);
    return NextResponse.json(
      { error: "Failed to reset participant rating" },
      { status: 500 },
    );
  }
}
