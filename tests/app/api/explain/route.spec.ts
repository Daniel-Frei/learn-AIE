import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { NextRequest } from "next/server";

type ExplainPayload = {
  questionPrompt: string;
  genericExplanation: string;
  options: Array<{ text: string; isCorrect: boolean; selected: boolean }>;
  isOverallCorrect: boolean;
  chatHistory: Array<{ role: "user" | "assistant"; content: string }>;
};

function buildJsonRequest(body: unknown): NextRequest {
  const req = new Request("http://localhost/api/explain", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return req as NextRequest;
}

function buildMalformedJsonRequest(): NextRequest {
  const req = new Request("http://localhost/api/explain", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{",
  });
  return req as NextRequest;
}

const validPayload: ExplainPayload = {
  questionPrompt: "What does the attention mechanism do?",
  genericExplanation: "It helps models focus on relevant tokens.",
  options: [
    { text: "It weighs token relevance", isCorrect: true, selected: true },
  ],
  isOverallCorrect: true,
  chatHistory: [],
};

describe("POST /api/explain", () => {
  const originalApiKey = process.env.OPENAI_API_KEY;

  beforeEach(() => {
    vi.resetModules();
    process.env.OPENAI_API_KEY = "";
  });

  afterEach(() => {
    process.env.OPENAI_API_KEY = originalApiKey;
    vi.restoreAllMocks();
  });

  it("returns 400 for invalid payload shape", async () => {
    const { POST } = await import("@/app/api/explain/route");

    const res = await POST(buildJsonRequest({ not: "valid" }));
    const body = await res.json();

    expect(res.status).toBe(400);
    expect(body).toEqual({ error: "Invalid request payload for explanation." });
  });

  it("returns 200 and fallback reply when API key is missing", async () => {
    const { POST } = await import("@/app/api/explain/route");

    const res = await POST(buildJsonRequest(validPayload));
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.reply).toContain("OpenAI API key is not configured");
  });

  it("returns 500 when body is not valid JSON", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const { POST } = await import("@/app/api/explain/route");

    const res = await POST(buildMalformedJsonRequest());
    const body = await res.json();

    expect(res.status).toBe(500);
    expect(body).toEqual({ error: "Failed to generate explanation" });
    expect(errorSpy).toHaveBeenCalled();
  });
});
