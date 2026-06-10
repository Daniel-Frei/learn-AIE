import { ALL_SOURCE_IDS, type SourceId } from "./quiz";

export function parseQuizSourceParam(
  value: string | string[] | null | undefined,
): SourceId | null {
  const candidate = Array.isArray(value) ? value[0] : value;
  if (!candidate) return null;

  return ALL_SOURCE_IDS.includes(candidate as SourceId)
    ? (candidate as SourceId)
    : null;
}
