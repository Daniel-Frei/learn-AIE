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

- Status: `413`

```ts
{
  error: "Explanation request is too large.";
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
- The route rejects malformed JSON, empty required strings, more than `12` answer options, more than `12` chat turns, overlong prompt/explanation/option/chat fields, and declared request bodies over `32 KB`.

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
  elapsedMs?: number;
  mistakeCount?: number;
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

## Endpoint: `POST /api/participant-rating-reset`

### Purpose

Reset one anonymous participant's Glicko rating to the default user state while preserving shared question ratings, answer-attempt history, and report counts.

### Request Body

```ts
{
  participantId: string;
}
```

### Success Response

- Status: `200`
- Body shape matches `GET /api/quiz-state`.

### Error Responses

- Status: `400`

```ts
{
  error: "participantId is required for rating reset.";
}
```

- Status: `500`

```ts
{
  error: "Failed to reset participant rating";
}
```

### Notes For Backend Integrators

- The reset only updates the `participants` row for the provided participant. It does not delete `question_ratings`, `answer_attempts`, or `question_reports`.
- If the participant had not completed legacy migration, reset marks it complete to prevent old browser-local ratings from being imported again.

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
      topic: Topic; // one of lib/questionTopics.ALL_TOPICS; currently "RL" | "DL" | "NLP" | "Math" | "Life Science"
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

### Notes For Backend Integrators

- Reports are append-only in the shared database; multiple reports for the same `questionId` remain separate entries.
- Report submission rejects empty fields, comments over `2,000` characters, prompt snapshots over `4,000` characters, labels/ids over `200` characters, and topics outside the configured `lib/questionTopics.ALL_TOPICS` list.
- The API validator owns the current accepted topic list. The `question_reports.topic` database constraint intentionally checks only that topic text is present and bounded, not that it belongs to a closed enum, so future quiz topics do not need another report-table schema migration.

## Endpoint: `POST /api/local-migration`

### Purpose

Import one participant's legacy local browser rating data into the shared database.

### Request Body

```ts
{
  participantId: string;
  localRatingState?: unknown;
}
```

### Notes For Backend Integrators

- If `elapsedMs` or `mistakeCount` is omitted, the server treats the answer like a full-weight binary result, which preserves old callers.
- The rating engine still uses a Glicko-2 style update internally; the timing and mistake metadata only scale the magnitude of the win/loss exchange. `elapsedMs` keeps full weight through the first 20 seconds, then scales linearly to a 25% maximum reduction at 3 minutes or slower.
- For a question with no persisted rating yet, the `label` seeds the initial question rating as easy `1400`, medium `1500`, or hard `1600`; subsequent answer data updates and persists the question rating from that starting point.

- Rating migration is approximate because the legacy browser store only contains aggregates, not the original event log.
- Migration is idempotent per participant for deterministic synthetic answer-attempt ids.
- Legacy local question reports are intentionally ignored; report state starts from the shared database.

## Shared Storage Environment Variables

- `NEXT_PUBLIC_SUPABASE_URL` (required)
- `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` (required for client configuration / deployment parity)
- `SUPABASE_SERVICE_ROLE_KEY` (required for server-side route access)
- In local development, if Supabase is unreachable the server falls back to an in-memory quiz store for the current process so the app can still load. That fallback is not durable and resets on restart.

## Mobile Client Configuration

- `EXPO_PUBLIC_SUPABASE_URL` (mobile required for profile sync): Supabase project URL.
- `EXPO_PUBLIC_SUPABASE_PUBLISHABLE_KEY` (mobile required for profile sync): Supabase publishable/anon key. This is safe to ship only with Row Level Security enabled.
- `EXPO_PUBLIC_QUIZ_API_BASE_URL` (mobile optional): absolute base URL for the Next.js API host, for example `http://192.168.1.20:43191` during LAN testing or the deployed web app URL in production. Mobile uses this only for detailed AI explanation chat because the OpenAI key remains server-side.
- Mobile profile sync uses Supabase Auth directly:
  - Auth user id is used as `participants.participant_id`.
  - `participants`, `question_ratings`, `answer_attempts`, and `question_reports` are accessed with RLS policies from `supabase/migrations/20260511160000_mobile_profiles_rls.sql`.
  - Mobile clients can read only `id` and `question_id` from shared question reports for counts; report comments and prompt snapshots remain server-side.
  - Answers are queued locally when offline, then flushed to Supabase when the user signs in and sync is reachable.
  - Question reports require a signed-in profile and are submitted to Supabase; old locally queued mobile reports are ignored.
- Question content is still bundled from the repository question bank; Supabase sync currently covers ratings, attempts, reports, and profile state, not remote question-bank content.
