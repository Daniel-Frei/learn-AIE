import { NextRequest, NextResponse } from "next/server";
import { getQuizState } from "@/lib/server/quizDataService";

export const runtime = "nodejs";

export async function GET(req: NextRequest) {
  try {
    const participantId = new URL(req.url).searchParams
      .get("participantId")
      ?.trim();
    if (!participantId) {
      return NextResponse.json(
        { error: "participantId is required." },
        { status: 400 },
      );
    }

    const state = await getQuizState(participantId);
    return NextResponse.json(state);
  } catch (err) {
    console.error("Error in /api/quiz-state:", err);
    return NextResponse.json(
      { error: "Failed to load quiz state" },
      { status: 500 },
    );
  }
}
