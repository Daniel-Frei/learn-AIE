import { timingSafeEqual } from "node:crypto";
import { NextRequest, NextResponse } from "next/server";
import { exportQuestionReportsFromStore } from "@/lib/server/quizDataService";

export const runtime = "nodejs";

function getConfiguredExportToken(): string | null {
  const token = process.env.QUESTION_REPORT_EXPORT_TOKEN?.trim();
  return token || null;
}

function isProductionDeployment(): boolean {
  return (
    process.env.NODE_ENV === "production" ||
    process.env.VERCEL_ENV === "production"
  );
}

function extractBearerToken(req: NextRequest): string | null {
  const authorization = req.headers.get("authorization")?.trim();
  const match = authorization?.match(/^Bearer\s+(.+)$/i);
  return match?.[1]?.trim() || null;
}

function tokensMatch(candidate: string, expected: string): boolean {
  const candidateBytes = Buffer.from(candidate);
  const expectedBytes = Buffer.from(expected);
  return (
    candidateBytes.length === expectedBytes.length &&
    timingSafeEqual(candidateBytes, expectedBytes)
  );
}

function authorizeExport(req: NextRequest): NextResponse | null {
  const expectedToken = getConfiguredExportToken();
  if (!expectedToken) {
    if (isProductionDeployment()) {
      return NextResponse.json(
        {
          error:
            "Question report export is disabled until QUESTION_REPORT_EXPORT_TOKEN is configured.",
        },
        { status: 403 },
      );
    }
    return null;
  }

  const suppliedToken = extractBearerToken(req);
  if (!suppliedToken || !tokensMatch(suppliedToken, expectedToken)) {
    return NextResponse.json(
      { error: "Question report export requires a valid bearer token." },
      { status: 401 },
    );
  }

  return null;
}

export async function GET(req: NextRequest) {
  try {
    const authFailure = authorizeExport(req);
    if (authFailure) return authFailure;

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
