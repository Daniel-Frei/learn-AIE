import QuizPageClient from "../components/QuizPageClient";
import { parseQuizSourceParam } from "../lib/quizRoute";

type QuizPageProps = {
  searchParams?: Promise<{
    source?: string | string[];
  }>;
};

export default async function QuizPage({ searchParams }: QuizPageProps) {
  const params = searchParams ? await searchParams : undefined;
  const initialSource = parseQuizSourceParam(params?.source);

  return (
    <QuizPageClient
      key={initialSource ?? "no-initial-source"}
      initialSource={initialSource}
    />
  );
}
