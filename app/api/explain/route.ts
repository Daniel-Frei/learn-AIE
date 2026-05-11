// app\api\explain\route.ts
import { NextRequest, NextResponse } from "next/server";
import {
  getLLMExplanation,
  type ExplanationRequest,
} from "../../../lib/llm/explain";

export const runtime = "nodejs";

const MAX_EXPLAIN_BODY_BYTES = 32_000;
const MAX_PROMPT_CHARS = 4_000;
const MAX_EXPLANATION_CHARS = 6_000;
const MAX_OPTIONS = 12;
const MAX_OPTION_CHARS = 1_500;
const MAX_CHAT_TURNS = 12;
const MAX_CHAT_CONTENT_CHARS = 2_000;

function isBoundedString(value: unknown, maxLength: number): value is string {
  return (
    typeof value === "string" &&
    value.trim().length > 0 &&
    value.length <= maxLength
  );
}

function isValidExplanationRequest(
  value: unknown,
): value is ExplanationRequest {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }

  const body = value as Partial<ExplanationRequest>;
  return (
    isBoundedString(body.questionPrompt, MAX_PROMPT_CHARS) &&
    isBoundedString(body.genericExplanation, MAX_EXPLANATION_CHARS) &&
    typeof body.isOverallCorrect === "boolean" &&
    Array.isArray(body.options) &&
    body.options.length > 0 &&
    body.options.length <= MAX_OPTIONS &&
    body.options.every(
      (option) =>
        option &&
        typeof option === "object" &&
        isBoundedString(option.text, MAX_OPTION_CHARS) &&
        typeof option.isCorrect === "boolean" &&
        typeof option.selected === "boolean",
    ) &&
    Array.isArray(body.chatHistory) &&
    body.chatHistory.length <= MAX_CHAT_TURNS &&
    body.chatHistory.every(
      (turn) =>
        turn &&
        typeof turn === "object" &&
        (turn.role === "user" || turn.role === "assistant") &&
        isBoundedString(turn.content, MAX_CHAT_CONTENT_CHARS),
    )
  );
}

export async function POST(req: NextRequest) {
  try {
    const contentLength = Number(req.headers.get("content-length") ?? 0);
    if (contentLength > MAX_EXPLAIN_BODY_BYTES) {
      return NextResponse.json(
        { error: "Explanation request is too large." },
        { status: 413 },
      );
    }

    const body = (await req.json()) as unknown;
    if (!isValidExplanationRequest(body)) {
      return NextResponse.json(
        { error: "Invalid request payload for explanation." },
        { status: 400 },
      );
    }

    const reply = await getLLMExplanation(body);
    return NextResponse.json({ reply });
  } catch (err) {
    if (err instanceof SyntaxError) {
      return NextResponse.json(
        { error: "Invalid request payload for explanation." },
        { status: 400 },
      );
    }

    console.error("Error in /api/explain:", err);
    return NextResponse.json(
      { error: "Failed to generate explanation" },
      { status: 500 },
    );
  }
}
