import { NextRequest, NextResponse } from "next/server";
import { QUESTION_METADATA } from "@/lib/questionMetadata";
import type { LocalMigrationRequest } from "@/lib/quizSync";
import { migrateLocalStateForParticipant } from "@/lib/server/quizDataService";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as LocalMigrationRequest;
    if (!body?.participantId?.trim()) {
      return NextResponse.json(
        { error: "participantId is required for local migration." },
        { status: 400 },
      );
    }

    const state = await migrateLocalStateForParticipant({
      participantId: body.participantId,
      localRatingState: body.localRatingState,
      localReportState: body.localReportState,
      questionMetadata: QUESTION_METADATA,
    });

    return NextResponse.json(state);
  } catch (err) {
    console.error("Error in /api/local-migration:", err);
    return NextResponse.json(
      { error: "Failed to migrate local state" },
      { status: 500 },
    );
  }
}
