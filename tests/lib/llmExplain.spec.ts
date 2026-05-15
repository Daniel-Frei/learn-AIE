import { afterEach, describe, expect, it, vi } from "vitest";
import type { ExplanationRequest } from "@/lib/llm/explain";

const baseRequest: ExplanationRequest = {
  questionPrompt: "Which attention statements are true?",
  genericExplanation: "Attention compares queries and keys to weight values.",
  options: [
    {
      text: "Selected correct option",
      isCorrect: true,
      selected: true,
    },
    {
      text: "Missed correct option",
      isCorrect: true,
      selected: false,
    },
    {
      text: "Selected wrong option",
      isCorrect: false,
      selected: true,
    },
    {
      text: "Rejected wrong option",
      isCorrect: false,
      selected: false,
    },
  ],
  isOverallCorrect: false,
  chatHistory: [{ role: "user", content: "Can you explain the misses?" }],
};

describe("LLM explanation helper", () => {
  const originalApiKey = process.env.OPENAI_API_KEY;

  afterEach(() => {
    process.env.OPENAI_API_KEY = originalApiKey;
    vi.resetModules();
    vi.doUnmock("openai");
    vi.restoreAllMocks();
  });

  it("returns the server fallback without creating a client when the API key is missing", async () => {
    process.env.OPENAI_API_KEY = "";
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    const { getLLMExplanation } = await import("@/lib/llm/explain");

    await expect(getLLMExplanation(baseRequest)).resolves.toBe(
      "OpenAI API key is not configured on the server.",
    );
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining("OPENAI_API_KEY is not set"),
    );
  });

  it("sends a targeted prompt with option-level mistakes and chat history", async () => {
    process.env.OPENAI_API_KEY = "test-api-key";
    const createCompletion = vi.fn().mockResolvedValue({
      choices: [{ message: { content: "Targeted explanation" } }],
    });
    const OpenAI = vi.fn(function MockOpenAI() {
      return {
        chat: { completions: { create: createCompletion } },
      };
    });
    vi.doMock("openai", () => ({ default: OpenAI }));

    const { getLLMExplanation } = await import("@/lib/llm/explain");
    const reply = await getLLMExplanation(baseRequest);

    expect(reply).toBe("Targeted explanation");
    expect(OpenAI).toHaveBeenCalledWith({ apiKey: "test-api-key" });
    expect(createCompletion).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "gpt-5-nano",
        messages: expect.any(Array),
      }),
    );

    const { messages } = createCompletion.mock.calls[0][0] as {
      messages: Array<{ role: string; content: string }>;
    };
    expect(messages[0]).toMatchObject({ role: "system" });
    expect(messages[1].content).toContain(
      "INCORRECT option that the user SELECTED",
    );
    expect(messages[1].content).toContain(
      "CORRECT option that the user did NOT SELECT",
    );
    expect(messages[1].content).toContain(
      "Overall grading: the user did NOT get the question fully correct.",
    );
    expect(messages.at(-1)).toEqual({
      role: "user",
      content: "Can you explain the misses?",
    });
  });

  it("reinforces correct answers and returns a fallback when the model response is empty", async () => {
    process.env.OPENAI_API_KEY = "test-api-key";
    const createCompletion = vi.fn().mockResolvedValue({ choices: [] });
    vi.doMock("openai", () => ({
      default: vi.fn(function MockOpenAI() {
        return {
          chat: { completions: { create: createCompletion } },
        };
      }),
    }));

    const { getLLMExplanation } = await import("@/lib/llm/explain");
    const reply = await getLLMExplanation({
      ...baseRequest,
      options: [
        { text: "Correct", isCorrect: true, selected: true },
        { text: "Distractor", isCorrect: false, selected: false },
      ],
      isOverallCorrect: true,
      chatHistory: [{ role: "assistant", content: "Previous answer" }],
    });

    expect(reply).toBe("Sorry, I couldn't generate an explanation.");
    const { messages } = createCompletion.mock.calls[0][0] as {
      messages: Array<{ role: string; content: string }>;
    };
    expect(messages[1].content).toContain(
      "The user answered this question perfectly",
    );
    expect(messages[1].content).toContain(
      "Overall grading: the user got the question correct",
    );
    expect(messages.at(-1)).toEqual({
      role: "assistant",
      content: "Previous answer",
    });
  });
});
