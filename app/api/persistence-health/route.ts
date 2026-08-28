import { NextResponse } from "next/server";
import { probeQuizDataPersistence } from "@/lib/server/quizDataStore";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    return NextResponse.json(await probeQuizDataPersistence());
  } catch (err) {
    console.error("Error in /api/persistence-health:", err);
    return NextResponse.json(
      { error: "Failed to check quiz persistence" },
      { status: 503 },
    );
  }
}
