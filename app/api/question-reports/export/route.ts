import { NextResponse } from "next/server";
import { exportQuestionReportsFromStore } from "@/lib/server/quizDataService";

export const runtime = "nodejs";

export async function GET() {
  try {
    const payload = await exportQuestionReportsFromStore();
    return NextResponse.json(payload);
  } catch (err) {
    console.error("Error in /api/question-reports/export:", err);
    return NextResponse.json(
      { error: "Failed to export question reports" },
      { status: 500 },
    );
  }
}
