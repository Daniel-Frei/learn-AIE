// app\api\explain\route.ts
import { NextRequest, NextResponse } from "next/server";
import {
  getLLMExplanation,
  type ExplanationRequest,
} from "../../../lib/llm/explain";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as ExplanationRequest;

    if (
      !body ||
      !body.questionPrompt ||
      !Array.isArray(body.options) ||
      !Array.isArray(body.chatHistory)
    ) {
      return NextResponse.json(
        { error: "Invalid request payload for explanation." },
        { status: 400 },
      );
    }

    const reply = await getLLMExplanation(body);
    return NextResponse.json({ reply });
  } catch (err) {
    console.error("Error in /api/explain:", err);
    return NextResponse.json(
      { error: "Failed to generate explanation" },
      { status: 500 },
    );
  }
}
