// lib/llm/explain.ts
import OpenAI from "openai";

const apiKey = process.env.OPENAI_API_KEY;

if (!apiKey) {
  console.warn(
    "Warning: OPENAI_API_KEY is not set. The detailed explanation feature will not work.",
  );
}

const client = apiKey ? new OpenAI({ apiKey }) : null;

export type LLMChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type ExplanationOption = {
  text: string;
  isCorrect: boolean;
  selected: boolean;
};

export type ExplanationChatTurn = {
  role: "user" | "assistant";
  content: string;
};

export type ExplanationRequest = {
  questionPrompt: string;
  genericExplanation: string;
  options: ExplanationOption[];
  isOverallCorrect: boolean;
  chatHistory: ExplanationChatTurn[];
};

function buildContextPrompt(req: ExplanationRequest): string {
  const { questionPrompt, genericExplanation, options, isOverallCorrect } = req;

  // Classify options from the perspective of the grading logic
  const missedCorrect = options.filter((o) => o.isCorrect && !o.selected);
  const wronglySelected = options.filter((o) => !o.isCorrect && o.selected);
  const header = [
    "You are helping a user with a multi-select quiz question about deep learning / NLP.",
    "",
    `Question: ${questionPrompt}`,
    "",
    "Each option below indicates whether it is actually correct, and how the user answered it.",
    "",
  ];

  const optionLines = options.map((opt) => {
    let gradingLabel: string;

    if (opt.isCorrect && opt.selected) {
      gradingLabel = "USER ANSWERED CORRECTLY: selected a CORRECT option.";
    } else if (opt.isCorrect && !opt.selected) {
      gradingLabel =
        "USER MISTAKE: this option is CORRECT but the user did NOT select it (missed correct option).";
    } else if (!opt.isCorrect && opt.selected) {
      gradingLabel =
        "USER MISTAKE: this option is INCORRECT but the user SELECTED it (selected wrong option).";
    } else {
      gradingLabel =
        "USER ANSWERED CORRECTLY: this option is INCORRECT and the user did NOT select it.";
    }

    const correctness = opt.isCorrect
      ? "ACTUALLY CORRECT"
      : "ACTUALLY INCORRECT";

    return `- "${opt.text}" — ${correctness}. ${gradingLabel}`;
  });

  const mistakeLines: string[] = [];
  const hasMistakes = missedCorrect.length > 0 || wronglySelected.length > 0;

  if (hasMistakes) {
    mistakeLines.push(
      "",
      "User mistakes (FOCUS YOUR EXPLANATION ON THESE OPTIONS):",
    );

    wronglySelected.forEach((opt) => {
      mistakeLines.push(
        `- "${opt.text}" — INCORRECT option that the user SELECTED (explain why it is wrong and what misconception it reflects).`,
      );
    });

    missedCorrect.forEach((opt) => {
      mistakeLines.push(
        `- "${opt.text}" — CORRECT option that the user did NOT SELECT (explain why it is actually correct and what idea they might be missing).`,
      );
    });
  } else {
    mistakeLines.push(
      "",
      "The user answered this question perfectly (no grading mistakes).",
      "You should still briefly reinforce the core concept and, if helpful, highlight why the other options are wrong.",
    );
  }

  const footer: string[] = [
    "",
    "Generic explanation that the user has ALREADY seen:",
    genericExplanation,
    "",
    isOverallCorrect
      ? "Overall grading: the user got the question correct, but they may still have subtle misunderstandings."
      : "Overall grading: the user did NOT get the question fully correct.",
    "",
    "Your task:",
    "- Start with 1–2 sentences summarizing the core concept in simple terms.",
    "- THEN, focus primarily on the options listed under 'User mistakes'.",
    "- For each mistaken option:",
    "  • If the user SELECTED an INCORRECT option, explain clearly why it is wrong and contrast it with the correct idea.",
    "  • If the user MISSED a CORRECT option, explain why it is correct and what reasoning justifies it.",
    "- You may refer to correctly answered options, if it helps to explain to the user where their misconceptions come from.",
    "- Be clear and concise; avoid repeating the full question text.",
  ];

  return [...header, ...optionLines, "", ...mistakeLines, "", ...footer].join(
    "\n",
  );
}

export async function getLLMExplanation(
  req: ExplanationRequest,
): Promise<string> {
  if (!client) {
    return "OpenAI API key is not configured on the server.";
  }

  const contextPrompt = buildContextPrompt(req);

  const messages: LLMChatMessage[] = [
    {
      role: "system",
      content:
        "You are a helpful tutor for an advanced deep learning / NLP quiz. " +
        "Always give targeted, concept-focused explanations that correct misunderstandings without being too verbose.",
    },
    {
      role: "user",
      content: contextPrompt,
    },
    // Then the ongoing chat history (follow-ups)
    ...req.chatHistory.map((turn) => ({
      role: turn.role,
      content: turn.content,
    })),
  ];

  const completion = await client.chat.completions.create({
    model: "gpt-5-nano",
    messages,
  });

  const content = completion.choices[0]?.message?.content;
  return content ?? "Sorry, I couldn't generate an explanation.";
}
