import { buildQuizApiUrl } from "../../../../lib/quizSession";
import type { ExplanationRequest } from "../../../../lib/llm/explain";
import type {
  QuizStateResponse,
  RecordAnswerRequest,
  RecordAnswerResponse,
  SubmitQuestionReportRequest,
  SubmitQuestionReportResponse,
} from "../../../../lib/quizSync";

export function getConfiguredApiBaseUrl(): string | null {
  const value = process.env.EXPO_PUBLIC_QUIZ_API_BASE_URL?.trim();
  return value ? value : null;
}

export function getMobileQuizApiUrl(
  path: string,
  query?: Record<string, string>,
): string {
  const baseUrl = getConfiguredApiBaseUrl();
  if (!baseUrl) {
    throw new Error("EXPO_PUBLIC_QUIZ_API_BASE_URL is not configured.");
  }

  return buildQuizApiUrl(baseUrl, path, query);
}

async function fetchJson<T>(
  path: string,
  init?: RequestInit,
  query?: Record<string, string>,
): Promise<T> {
  const response = await fetch(getMobileQuizApiUrl(path, query), {
    headers: {
      "content-type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  const body = (await response.json()) as T & { error?: string };
  if (!response.ok) {
    throw new Error(body.error ?? `Request failed with status ${response.status}`);
  }
  return body;
}

export function fetchQuizState(
  participantId: string,
): Promise<QuizStateResponse> {
  return fetchJson<QuizStateResponse>(
    "/api/quiz-state",
    { method: "GET" },
    { participantId },
  );
}

export function submitMobileAnswer(
  request: RecordAnswerRequest,
): Promise<RecordAnswerResponse> {
  return fetchJson<RecordAnswerResponse>("/api/answers", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export function submitMobileQuestionReport(
  request: SubmitQuestionReportRequest,
): Promise<SubmitQuestionReportResponse> {
  return fetchJson<SubmitQuestionReportResponse>("/api/question-reports", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function fetchMobileExplanation(
  request: ExplanationRequest,
): Promise<string> {
  const response = await fetchJson<{ reply: string }>("/api/explain", {
    method: "POST",
    body: JSON.stringify(request),
  });

  return response.reply;
}
