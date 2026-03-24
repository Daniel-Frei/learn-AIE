# API Contract

## Endpoint: `POST /api/explain`

### Purpose

Generate a targeted explanation for a quiz question answer and follow-up chat turns.

### Request Body

```ts
{
  questionPrompt: string;
  genericExplanation: string;
  options: Array<{
    text: string;
    isCorrect: boolean;
    selected: boolean;
  }>;
  isOverallCorrect: boolean;
  chatHistory: Array<{
    role: "user" | "assistant";
    content: string;
  }>;
}
```

### Success Response

- Status: `200`

```ts
{
  reply: string;
}
```

### Error Responses

- Status: `400`

```ts
{
  error: "Invalid request payload for explanation.";
}
```

- Status: `500`

```ts
{
  error: "Failed to generate explanation";
}
```

### Environment Variables

- `OPENAI_API_KEY` (required for real LLM responses in this endpoint).

### Notes For Backend Integrators

- If `OPENAI_API_KEY` is missing, the LLM helper returns a fallback string and no upstream API call is made.
- Model currently configured in code: `gpt-5-nano`.

## Client Export: Question Reports JSON

### Purpose

Export locally saved question-quality reports so a reviewer can inspect flagged questions outside the app.

### File Shape

```ts
{
  version: 1;
  exportedAt: string;
  reports: Array<{
    id: string;
    questionId: string;
    comment: string;
    reportedAt: string;
    snapshot: {
      sourceId: string;
      sourceLabel: string;
      seriesId: string;
      seriesLabel: string;
      topic: "RL" | "DL" | "NLP" | "Math";
      prompt: string;
    };
  }>;
}
```

### Notes For Future Backend Work

- Reports are append-only in the current client implementation; multiple reports for the same `questionId` remain separate entries.
- The export intentionally includes a question snapshot for offline triage without exporting answer options or correctness metadata.
- No import or server sync exists yet for question reports.
