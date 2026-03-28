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

## Endpoint: `GET /api/quiz-state`

### Purpose

Load the current anonymous participant rating state, global question difficulty state, and report-count summary from the shared database.

### Query Parameters

- `participantId` (required)

### Success Response

- Status: `200`

```ts
{
  participantId: string;
  ratingState: {
    version: 2;
    algorithm: "glicko-2";
    config: {
      defaultRating: number;
      defaultRdUser: number;
      defaultRdQuestion: number;
      defaultSigma: number;
      tau: number;
      epsilon: number;
      periodDays: number;
      minRd: number;
      maxRd: number;
      difficultyAnchorRating: number;
      difficultyScale: number;
    }
    user: {
      rating: number;
      rd: number;
      sigma: number;
      lastUpdatedAt: number;
      gamesPlayed: number;
    }
    questions: Record<
      string,
      {
        rating: number;
        rd: number;
        sigma: number;
        lastUpdatedAt: number;
        gamesPlayed: number;
        legacyCorrect: number;
        legacyWrong: number;
        label?: "easy" | "medium" | "hard";
      }
    >;
  }
  reportSummary: {
    totalReportCount: number;
    countsByQuestion: Record<string, number>;
  }
  legacyMigrationCompleted: boolean;
}
```

### Error Responses

- Status: `400`

```ts
{
  error: "participantId is required.";
}
```

- Status: `500`

```ts
{
  error: "Failed to load quiz state";
}
```

## Endpoint: `POST /api/answers`

### Purpose

Persist one quiz answer for an anonymous participant and update both participant and question Glicko entities in shared storage.

### Request Body

```ts
{
  participantId: string;
  questionId: string;
  label?: "easy" | "medium" | "hard";
  isCorrect: boolean;
}
```

### Success Response

- Status: `200`

```ts
{
  participantId: string;
  user: {
    rating: number;
    rd: number;
    sigma: number;
    lastUpdatedAt: number;
    gamesPlayed: number;
  };
  questionId: string;
  question: {
    rating: number;
    rd: number;
    sigma: number;
    lastUpdatedAt: number;
    gamesPlayed: number;
    legacyCorrect: number;
    legacyWrong: number;
    label?: "easy" | "medium" | "hard";
  };
}
```

### Error Responses

- Status: `400`

```ts
{
  error: "Invalid request payload for answer submission.";
}
```

- Status: `500`

```ts
{
  error: "Failed to record answer";
}
```

## Endpoint: `POST /api/question-reports`

### Purpose

Persist a shared append-only question-quality report.

### Request Body

```ts
{
  participantId: string;
  draft: {
    questionId: string;
    comment: string;
    snapshot: {
      sourceId: string;
      sourceLabel: string;
      seriesId: string;
      seriesLabel: string;
      topic: "RL" | "DL" | "NLP" | "Math";
      prompt: string;
    }
  }
}
```

### Success Response

- Status: `200`

```ts
{
  totalReportCount: number;
  questionReportCount: number;
}
```

### Error Responses

- Status: `400`

```ts
{
  error: "Invalid request payload for question report.";
}
```

- Status: `500`

```ts
{
  error: "Failed to submit question report";
}
```

## Endpoint: `GET /api/question-reports/export`

### Purpose

Export all shared question reports as versioned JSON.

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

- Reports are append-only in the shared database implementation; multiple reports for the same `questionId` remain separate entries.
- The export intentionally includes a question snapshot for offline triage without exporting answer options or correctness metadata.
- Export is now sourced from the server rather than browser-local storage.

## Endpoint: `POST /api/local-migration`

### Purpose

Import one participant’s legacy local browser data into the shared database.

### Request Body

```ts
{
  participantId: string;
  localRatingState?: unknown;
  localReportState?: unknown;
}
```

### Notes For Backend Integrators

- Rating migration is approximate because the legacy browser store only contains aggregates, not the original event log.
- Migration is idempotent per participant for stable report ids and deterministic synthetic answer-attempt ids.

## Shared Storage Environment Variables

- `NEXT_PUBLIC_SUPABASE_URL` (required)
- `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` (required for client configuration / deployment parity)
- `SUPABASE_SERVICE_ROLE_KEY` (required for server-side route access)
