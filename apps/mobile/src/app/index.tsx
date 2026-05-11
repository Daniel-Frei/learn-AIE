import React, { useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import {
  ALL_SOURCE_IDS,
  ALL_TOPICS,
  QUESTION_SOURCES,
  SOURCE_SERIES,
  getSeriesIdsForSources,
  getTitleForSelection,
  type Question,
  type SourceId,
  type SourceSeriesId,
  type Topic,
} from "../../../../lib/quiz";
import { normalizeMathDelimiters } from "../../../../lib/normalizeMath";
import type {
  DifficultyRange,
  QuestionSelectionMode,
} from "../../../../lib/quizSession";
import { useMobileQuiz } from "../hooks/useMobileQuiz";
import {
  fetchMobileExplanation,
  getConfiguredApiBaseUrl,
} from "../lib/mobileApi";

type ChatRole = "user" | "assistant";

type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
};

function formatElapsedMs(elapsedMs: number): string {
  const totalSeconds = Math.max(0, Math.floor(elapsedMs / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function formatRating(rating: number): string {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
  }).format(rating);
}

function formatStudyText(text: string): string {
  return normalizeMathDelimiters(text)
    .replace(/\*\*/g, "")
    .replace(/`/g, "")
    .replace(/^#{1,6}\s+/gm, "");
}

function makeId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function parseRangeNumber(value: string, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

type ModeButtonProps = {
  active: boolean;
  label: string;
  onPress: () => void;
};

function ModeButton({ active, label, onPress }: ModeButtonProps) {
  return (
    <Pressable
      onPress={onPress}
      style={[styles.segmentButton, active && styles.segmentButtonActive]}
    >
      <Text style={[styles.segmentText, active && styles.segmentTextActive]}>
        {label}
      </Text>
    </Pressable>
  );
}

function ExplanationChat({
  question,
  options,
  isOverallCorrect,
}: {
  question: Question;
  options: { text: string; isCorrect: boolean; selected: boolean }[];
  isOverallCorrect: boolean;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const callApi = async (
    chatHistory: { role: ChatRole; content: string }[],
  ) => {
    return fetchMobileExplanation({
      questionPrompt: question.prompt,
      genericExplanation: question.explanation,
      options,
      isOverallCorrect,
      chatHistory,
    });
  };

  const startChat = async () => {
    if (!getConfiguredApiBaseUrl()) {
      setError("Set EXPO_PUBLIC_QUIZ_API_BASE_URL to use explanations.");
      setIsOpen(true);
      return;
    }

    if (messages.length > 0) {
      setIsOpen(true);
      return;
    }

    setIsOpen(true);
    setIsLoading(true);
    setError("");

    try {
      const reply = await callApi([]);
      setMessages([
        {
          id: makeId("assistant"),
          role: "assistant",
          content: reply,
        },
      ]);
    } catch (err) {
      console.error("Failed to load explanation:", err);
      setError("Could not load a detailed explanation.");
    } finally {
      setIsLoading(false);
    }
  };

  const sendFollowUp = async () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    const userMessage: ChatMessage = {
      id: makeId("user"),
      role: "user",
      content: trimmed,
    };
    const nextMessages = [...messages, userMessage];

    setMessages(nextMessages);
    setInput("");
    setIsLoading(true);
    setError("");

    try {
      const reply = await callApi(
        nextMessages.map(({ role, content }) => ({ role, content })),
      );
      setMessages([
        ...nextMessages,
        {
          id: makeId("assistant"),
          role: "assistant",
          content: reply,
        },
      ]);
    } catch (err) {
      console.error("Failed to send follow-up:", err);
      setError("Could not send your follow-up.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <View style={styles.chatBlock}>
      <Pressable onPress={startChat} style={styles.secondaryButton}>
        <Text style={styles.secondaryButtonText}>
          Ask for a detailed explanation
        </Text>
      </Pressable>

      {isOpen && (
        <View style={styles.chatPanel}>
          {messages.map((message) => (
            <View
              key={message.id}
              style={[
                styles.chatBubble,
                message.role === "user"
                  ? styles.userBubble
                  : styles.assistantBubble,
              ]}
            >
              <Text style={styles.chatText}>
                {formatStudyText(message.content)}
              </Text>
            </View>
          ))}

          {isLoading && <ActivityIndicator color="#7dd3fc" />}
          {error ? <Text style={styles.errorText}>{error}</Text> : null}

          <View style={styles.followUpRow}>
            <TextInput
              value={input}
              onChangeText={setInput}
              placeholder="Ask a follow-up"
              placeholderTextColor="#64748b"
              style={styles.followUpInput}
            />
            <Pressable
              onPress={sendFollowUp}
              disabled={isLoading || !input.trim()}
              style={[
                styles.smallPrimaryButton,
                (isLoading || !input.trim()) && styles.disabledButton,
              ]}
            >
              <Text style={styles.primaryButtonText}>Send</Text>
            </Pressable>
          </View>
        </View>
      )}
    </View>
  );
}

export default function QuizScreen() {
  const quiz = useMobileQuiz();
  const [isFilterOpen, setIsFilterOpen] = useState(true);
  const [pendingSources, setPendingSources] = useState<SourceId[]>([]);
  const [pendingTopics, setPendingTopics] = useState<Topic[]>([]);
  const [pendingMode, setPendingMode] =
    useState<QuestionSelectionMode>("standard");
  const [pendingRange, setPendingRange] = useState<DifficultyRange>({
    min: 1000,
    max: 2000,
  });
  const [isReportOpen, setIsReportOpen] = useState(false);
  const [reportComment, setReportComment] = useState("");
  const [reportStatus, setReportStatus] = useState("");
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authMode, setAuthMode] = useState<"sign-in" | "sign-up">("sign-in");
  const [authStatus, setAuthStatus] = useState("");

  useEffect(() => {
    setPendingSources(quiz.selectedSources);
  }, [quiz.selectedSources]);

  useEffect(() => {
    setPendingTopics(quiz.selectedTopics);
  }, [quiz.selectedTopics]);

  useEffect(() => {
    setPendingMode(quiz.selectionMode);
  }, [quiz.selectionMode]);

  useEffect(() => {
    setPendingRange(quiz.difficultyRange);
  }, [quiz.difficultyRange]);

  const pendingSeries = useMemo(
    () => getSeriesIdsForSources(pendingSources),
    [pendingSources],
  );

  const title = getTitleForSelection(quiz.selectedSources, quiz.selectedTopics);
  const hasQuestion = quiz.availableCount > 0 && Boolean(quiz.currentQuestion);

  const toggleSource = (sourceId: SourceId) => {
    setPendingSources((prev) =>
      prev.includes(sourceId)
        ? prev.filter((id) => id !== sourceId)
        : [...prev, sourceId],
    );
  };

  const toggleSeries = (seriesId: SourceSeriesId) => {
    const series = SOURCE_SERIES.find((entry) => entry.id === seriesId);
    if (!series) return;

    setPendingSources((prev) => {
      const selected = new Set(prev);
      const anySelected = series.sourceIds.some((id) => selected.has(id));
      if (anySelected) {
        return prev.filter((id) => !series.sourceIds.includes(id));
      }
      return Array.from(new Set([...prev, ...series.sourceIds]));
    });
  };

  const toggleTopic = (topic: Topic) => {
    setPendingTopics((prev) =>
      prev.includes(topic)
        ? prev.filter((item) => item !== topic)
        : [...prev, topic],
    );
  };

  const applyFilters = () => {
    quiz.applySelection({
      sources: pendingSources,
      series: pendingSeries,
      topics: pendingTopics,
      mode: pendingMode,
      difficultyRange: pendingRange,
    });
    setIsFilterOpen(false);
    setReportStatus("");
    setIsReportOpen(false);
  };

  const submitReport = async () => {
    const didSubmit = await quiz.submitQuestionReport(reportComment);
    if (!didSubmit) {
      setReportStatus("Enter a comment before submitting.");
      return;
    }

    setReportComment("");
    setReportStatus(
      quiz.profile ? "Report saved for sync." : "Report saved locally.",
    );
    setIsReportOpen(false);
  };

  const submitAuth = async () => {
    if (!authEmail.trim() || !authPassword) {
      setAuthStatus("Enter an email and password.");
      return;
    }

    const didAuthenticate =
      authMode === "sign-in"
        ? await quiz.signIn(authEmail, authPassword)
        : await quiz.signUp(authEmail, authPassword);
    setAuthStatus(
      didAuthenticate
        ? authMode === "sign-in"
          ? "Signed in."
          : "Account created. Check email confirmation if required."
        : "Authentication failed.",
    );
    if (didAuthenticate) {
      setAuthPassword("");
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.container}>
        <View style={styles.header}>
          <Text style={styles.eyebrow}>Book Quiz Mobile</Text>
          <Text style={styles.title}>{title}</Text>
          <Text style={styles.helperText}>
            Select all true statements, then submit.
          </Text>

          <View style={styles.statsGrid}>
            <View style={styles.statBox}>
              <Text style={styles.statLabel}>Glicko</Text>
              <Text style={styles.statValue}>
                {Math.round(quiz.userRating)} +/-{" "}
                {Math.round(quiz.userRatingRd)}
              </Text>
            </View>
            <View style={styles.statBox}>
              <Text style={styles.statLabel}>Session</Text>
              <Text style={styles.statValue}>
                {quiz.correctCount}/{quiz.answeredCount} ({quiz.accuracy}%)
              </Text>
            </View>
          </View>

          <View style={styles.statusRow}>
            <Text style={styles.statusText}>Sync: {quiz.syncStatus}</Text>
            <Text style={styles.statusText}>
              Profile: {quiz.profile?.email ?? "local guest"}
            </Text>
            <Text style={styles.statusMessage}>
              Queued: {quiz.queuedAnswerCount} answers, {quiz.queuedReportCount}{" "}
              reports
            </Text>
            {quiz.syncMessage ? (
              <Text style={styles.statusMessage}>{quiz.syncMessage}</Text>
            ) : null}
          </View>

          <View style={styles.authPanel}>
            {quiz.profile ? (
              <View style={styles.authActions}>
                <Pressable
                  onPress={() => {
                    void quiz.syncNow();
                  }}
                  style={styles.secondaryButton}
                >
                  <Text style={styles.secondaryButtonText}>Sync now</Text>
                </Pressable>
                <Pressable
                  onPress={() => {
                    void quiz.signOut();
                  }}
                  style={styles.secondaryButton}
                >
                  <Text style={styles.secondaryButtonText}>Sign out</Text>
                </Pressable>
              </View>
            ) : (
              <View style={styles.authForm}>
                <View style={styles.segmentRow}>
                  <ModeButton
                    label="Sign in"
                    active={authMode === "sign-in"}
                    onPress={() => setAuthMode("sign-in")}
                  />
                  <ModeButton
                    label="Sign up"
                    active={authMode === "sign-up"}
                    onPress={() => setAuthMode("sign-up")}
                  />
                </View>
                <TextInput
                  value={authEmail}
                  onChangeText={setAuthEmail}
                  autoCapitalize="none"
                  keyboardType="email-address"
                  placeholder="Email"
                  placeholderTextColor="#64748b"
                  style={styles.authInput}
                />
                <TextInput
                  value={authPassword}
                  onChangeText={setAuthPassword}
                  secureTextEntry
                  placeholder="Password"
                  placeholderTextColor="#64748b"
                  style={styles.authInput}
                />
                <Pressable onPress={submitAuth} style={styles.primaryButton}>
                  <Text style={styles.primaryButtonText}>
                    {authMode === "sign-in" ? "Sign in" : "Create profile"}
                  </Text>
                </Pressable>
                {authStatus ? (
                  <Text style={styles.statusMessage}>{authStatus}</Text>
                ) : null}
              </View>
            )}
          </View>

          <Pressable
            onPress={() => setIsFilterOpen((open) => !open)}
            style={styles.primaryButton}
          >
            <Text style={styles.primaryButtonText}>
              {isFilterOpen ? "Close filters" : "Choose filters"}
            </Text>
          </Pressable>
        </View>

        {isFilterOpen && (
          <View style={styles.panel}>
            <Text style={styles.panelTitle}>Practice filters</Text>

            <Text style={styles.sectionLabel}>Mode</Text>
            <View style={styles.segmentRow}>
              <ModeButton
                label="Standard"
                active={pendingMode === "standard"}
                onPress={() => setPendingMode("standard")}
              />
              <ModeButton
                label="Climb"
                active={pendingMode === "climb"}
                onPress={() => setPendingMode("climb")}
              />
            </View>

            <Text style={styles.sectionLabel}>Topics</Text>
            <View style={styles.wrapRow}>
              {ALL_TOPICS.map((topic) => {
                const active = pendingTopics.includes(topic);
                return (
                  <Pressable
                    key={topic}
                    onPress={() => toggleTopic(topic)}
                    style={[styles.chip, active && styles.chipActive]}
                  >
                    <Text
                      style={[styles.chipText, active && styles.chipTextActive]}
                    >
                      {topic}
                    </Text>
                  </Pressable>
                );
              })}
            </View>

            <Text style={styles.sectionLabel}>Question Elo range</Text>
            <View style={styles.rangeRow}>
              <TextInput
                keyboardType="numeric"
                value={String(pendingRange.min)}
                onChangeText={(value) =>
                  setPendingRange((prev) => ({
                    ...prev,
                    min: parseRangeNumber(value, prev.min),
                  }))
                }
                style={styles.rangeInput}
              />
              <Text style={styles.mutedText}>to</Text>
              <TextInput
                keyboardType="numeric"
                value={String(pendingRange.max)}
                onChangeText={(value) =>
                  setPendingRange((prev) => ({
                    ...prev,
                    max: parseRangeNumber(value, prev.max),
                  }))
                }
                style={styles.rangeInput}
              />
            </View>

            <Text style={styles.sectionLabel}>Series and lectures</Text>
            {SOURCE_SERIES.map((series) => {
              const active = pendingSeries.includes(series.id);
              const sources = QUESTION_SOURCES.filter(
                (source) => source.seriesId === series.id,
              );
              return (
                <View key={series.id} style={styles.seriesBlock}>
                  <Pressable
                    onPress={() => toggleSeries(series.id)}
                    style={[styles.seriesButton, active && styles.seriesActive]}
                  >
                    <Text
                      style={[
                        styles.seriesTitle,
                        active && styles.seriesTitleActive,
                      ]}
                    >
                      {active ? "[x]" : "[ ]"} {series.label}
                    </Text>
                    <Text style={styles.seriesCount}>
                      {sources.length} lectures/chapters
                    </Text>
                  </Pressable>
                  {sources.map((source) => {
                    const sourceActive = pendingSources.includes(source.id);
                    return (
                      <Pressable
                        key={source.id}
                        onPress={() => toggleSource(source.id)}
                        style={[
                          styles.sourceButton,
                          sourceActive && styles.sourceActive,
                        ]}
                      >
                        <Text style={styles.sourceTitle}>
                          {sourceActive ? "[x]" : "[ ]"} {source.label}
                        </Text>
                        <Text style={styles.sourceMeta}>{source.topic}</Text>
                      </Pressable>
                    );
                  })}
                </View>
              );
            })}

            <View style={styles.filterActions}>
              <Pressable
                onPress={() => setPendingSources(ALL_SOURCE_IDS)}
                style={styles.secondaryButton}
              >
                <Text style={styles.secondaryButtonText}>All series</Text>
              </Pressable>
              <Pressable
                onPress={() => setPendingTopics([...ALL_TOPICS])}
                style={styles.secondaryButton}
              >
                <Text style={styles.secondaryButtonText}>All topics</Text>
              </Pressable>
              <Pressable
                onPress={() => {
                  setPendingSources([]);
                  setPendingTopics([]);
                }}
                style={styles.secondaryButton}
              >
                <Text style={styles.secondaryButtonText}>Clear</Text>
              </Pressable>
            </View>

            <Pressable onPress={applyFilters} style={styles.primaryButton}>
              <Text style={styles.primaryButtonText}>Apply selection</Text>
            </Pressable>
          </View>
        )}

        <View style={styles.panel}>
          <View style={styles.questionMetaRow}>
            <Text style={styles.metaText}>
              Question {hasQuestion ? quiz.currentIndex + 1 : 0} of{" "}
              {quiz.availableCount}
            </Text>
            {hasQuestion && (
              <Text style={styles.metaText}>
                Elo{" "}
                {quiz.currentQuestionRating === null
                  ? "-"
                  : formatRating(quiz.currentQuestionRating)}{" "}
                | {formatElapsedMs(quiz.questionElapsedMs)} / 3:00
              </Text>
            )}
          </View>

          {!hasQuestion || !quiz.currentQuestion ? (
            <View style={styles.emptyState}>
              <Text style={styles.emptyTitle}>No questions selected</Text>
              <Text style={styles.helperText}>
                Choose a series, lecture, topic, or wider Elo range to start.
              </Text>
            </View>
          ) : (
            <>
              <Text style={styles.questionPrompt}>
                {formatStudyText(quiz.currentQuestion.prompt)}
              </Text>

              <View style={styles.optionsBlock}>
                {quiz.shuffledOptions.map((option, index) => {
                  const selected = quiz.selectedIndexes.includes(index);
                  const correct = option.isCorrect;
                  const resultStyle = quiz.showResult
                    ? selected && correct
                      ? styles.optionCorrect
                      : selected && !correct
                        ? styles.optionWrong
                        : !selected && correct
                          ? styles.optionMissed
                          : null
                    : selected
                      ? styles.optionSelected
                      : null;

                  return (
                    <Pressable
                      key={`${option.text}-${index}`}
                      onPress={() => quiz.toggleOption(index)}
                      style={[styles.optionButton, resultStyle]}
                    >
                      <Text style={styles.optionCheck}>
                        {selected ? "[x]" : "[ ]"}
                      </Text>
                      <Text style={styles.optionText}>
                        {formatStudyText(option.text)}
                      </Text>
                    </Pressable>
                  );
                })}
              </View>

              {!quiz.showResult ? (
                <Pressable
                  onPress={() => {
                    void quiz.submitAnswer();
                  }}
                  disabled={quiz.selectedIndexes.length === 0}
                  style={[
                    styles.primaryButton,
                    quiz.selectedIndexes.length === 0 && styles.disabledButton,
                  ]}
                >
                  <Text style={styles.primaryButtonText}>Submit answer</Text>
                </Pressable>
              ) : (
                <>
                  <Pressable
                    onPress={quiz.nextQuestion}
                    style={styles.nextButton}
                  >
                    <Text style={styles.primaryButtonText}>Next question</Text>
                  </Pressable>

                  <View
                    style={[
                      styles.resultBox,
                      quiz.showResult.isCorrect
                        ? styles.resultCorrect
                        : styles.resultWrong,
                    ]}
                  >
                    <Text style={styles.resultTitle}>
                      {quiz.showResult.isCorrect ? "Correct" : "Not quite"}
                    </Text>
                    <Text style={styles.resultText}>
                      {formatStudyText(quiz.currentQuestion.explanation)}
                    </Text>
                  </View>

                  <ExplanationChat
                    question={quiz.currentQuestion}
                    options={quiz.shuffledOptions.map((option, index) => ({
                      text: option.text,
                      isCorrect: option.isCorrect,
                      selected: quiz.selectedIndexes.includes(index),
                    }))}
                    isOverallCorrect={quiz.showResult.isCorrect}
                  />
                </>
              )}

              <View style={styles.reportBlock}>
                <Pressable
                  onPress={() => setIsReportOpen((open) => !open)}
                  style={styles.secondaryButton}
                >
                  <Text style={styles.secondaryButtonText}>
                    Report question ({quiz.currentQuestionReportCount})
                  </Text>
                </Pressable>
                {isReportOpen && (
                  <View style={styles.reportPanel}>
                    <TextInput
                      value={reportComment}
                      onChangeText={setReportComment}
                      placeholder="What seems wrong, vague, or misleading?"
                      placeholderTextColor="#64748b"
                      multiline
                      style={styles.reportInput}
                    />
                    <Pressable
                      onPress={() => {
                        void submitReport();
                      }}
                      style={styles.smallPrimaryButton}
                    >
                      <Text style={styles.primaryButtonText}>
                        Submit report
                      </Text>
                    </Pressable>
                  </View>
                )}
                {reportStatus ? (
                  <Text style={styles.statusMessage}>{reportStatus}</Text>
                ) : null}
              </View>
            </>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#020617",
  },
  container: {
    padding: 16,
    paddingBottom: 40,
    gap: 16,
  },
  header: {
    gap: 12,
  },
  eyebrow: {
    color: "#7dd3fc",
    fontSize: 12,
    fontWeight: "700",
    letterSpacing: 0,
    textTransform: "uppercase",
  },
  title: {
    color: "#f8fafc",
    fontSize: 26,
    fontWeight: "700",
    lineHeight: 32,
  },
  helperText: {
    color: "#cbd5e1",
    fontSize: 14,
    lineHeight: 20,
  },
  statsGrid: {
    flexDirection: "row",
    gap: 10,
  },
  statBox: {
    flex: 1,
    borderColor: "#1e293b",
    borderRadius: 8,
    borderWidth: 1,
    padding: 10,
    backgroundColor: "#0f172a",
  },
  statLabel: {
    color: "#94a3b8",
    fontSize: 12,
  },
  statValue: {
    color: "#f8fafc",
    fontSize: 15,
    fontWeight: "700",
    marginTop: 4,
  },
  statusRow: {
    gap: 4,
  },
  statusText: {
    color: "#e2e8f0",
    fontSize: 12,
    fontWeight: "700",
  },
  statusMessage: {
    color: "#fbbf24",
    fontSize: 12,
    lineHeight: 18,
  },
  authPanel: {
    backgroundColor: "#0f172a",
    borderColor: "#1e293b",
    borderRadius: 8,
    borderWidth: 1,
    padding: 10,
  },
  authActions: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  authForm: {
    gap: 10,
  },
  authInput: {
    backgroundColor: "#020617",
    borderColor: "#334155",
    borderRadius: 8,
    borderWidth: 1,
    color: "#f8fafc",
    paddingHorizontal: 10,
    paddingVertical: 10,
  },
  panel: {
    backgroundColor: "#0f172a",
    borderColor: "#1e293b",
    borderRadius: 8,
    borderWidth: 1,
    gap: 14,
    padding: 14,
  },
  panelTitle: {
    color: "#f8fafc",
    fontSize: 18,
    fontWeight: "700",
  },
  sectionLabel: {
    color: "#cbd5e1",
    fontSize: 13,
    fontWeight: "700",
  },
  segmentRow: {
    flexDirection: "row",
    gap: 8,
  },
  segmentButton: {
    borderColor: "#334155",
    borderRadius: 8,
    borderWidth: 1,
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  segmentButtonActive: {
    backgroundColor: "#0284c7",
    borderColor: "#38bdf8",
  },
  segmentText: {
    color: "#cbd5e1",
    fontWeight: "700",
  },
  segmentTextActive: {
    color: "#f8fafc",
  },
  wrapRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  chip: {
    borderColor: "#334155",
    borderRadius: 8,
    borderWidth: 1,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  chipActive: {
    backgroundColor: "#0284c7",
    borderColor: "#38bdf8",
  },
  chipText: {
    color: "#cbd5e1",
    fontWeight: "700",
  },
  chipTextActive: {
    color: "#f8fafc",
  },
  rangeRow: {
    alignItems: "center",
    flexDirection: "row",
    gap: 10,
  },
  rangeInput: {
    backgroundColor: "#020617",
    borderColor: "#334155",
    borderRadius: 8,
    borderWidth: 1,
    color: "#f8fafc",
    minWidth: 92,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  mutedText: {
    color: "#94a3b8",
  },
  seriesBlock: {
    gap: 8,
  },
  seriesButton: {
    borderColor: "#334155",
    borderRadius: 8,
    borderWidth: 1,
    gap: 4,
    padding: 10,
  },
  seriesActive: {
    borderColor: "#38bdf8",
    backgroundColor: "#082f49",
  },
  seriesTitle: {
    color: "#e2e8f0",
    fontSize: 14,
    fontWeight: "700",
  },
  seriesTitleActive: {
    color: "#f8fafc",
  },
  seriesCount: {
    color: "#94a3b8",
    fontSize: 12,
  },
  sourceButton: {
    borderColor: "#1e293b",
    borderRadius: 8,
    borderWidth: 1,
    marginLeft: 10,
    padding: 10,
  },
  sourceActive: {
    borderColor: "#38bdf8",
  },
  sourceTitle: {
    color: "#e2e8f0",
    fontSize: 13,
    fontWeight: "700",
  },
  sourceMeta: {
    color: "#94a3b8",
    fontSize: 12,
    marginTop: 2,
  },
  filterActions: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  primaryButton: {
    alignItems: "center",
    backgroundColor: "#38bdf8",
    borderRadius: 8,
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  nextButton: {
    alignItems: "center",
    backgroundColor: "#34d399",
    borderRadius: 8,
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  primaryButtonText: {
    color: "#020617",
    fontWeight: "800",
  },
  secondaryButton: {
    alignItems: "center",
    borderColor: "#334155",
    borderRadius: 8,
    borderWidth: 1,
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  secondaryButtonText: {
    color: "#e2e8f0",
    fontWeight: "700",
  },
  smallPrimaryButton: {
    alignItems: "center",
    backgroundColor: "#38bdf8",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  disabledButton: {
    opacity: 0.45,
  },
  questionMetaRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    gap: 8,
  },
  metaText: {
    color: "#94a3b8",
    fontSize: 12,
    fontWeight: "700",
    textTransform: "uppercase",
  },
  emptyState: {
    gap: 6,
    paddingVertical: 16,
  },
  emptyTitle: {
    color: "#f8fafc",
    fontSize: 18,
    fontWeight: "700",
  },
  questionPrompt: {
    color: "#f8fafc",
    fontSize: 19,
    fontWeight: "700",
    lineHeight: 26,
  },
  optionsBlock: {
    gap: 10,
  },
  optionButton: {
    alignItems: "flex-start",
    backgroundColor: "#020617",
    borderColor: "#334155",
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    gap: 10,
    padding: 12,
  },
  optionSelected: {
    backgroundColor: "#1e293b",
    borderColor: "#38bdf8",
  },
  optionCorrect: {
    backgroundColor: "#064e3b",
    borderColor: "#34d399",
  },
  optionWrong: {
    backgroundColor: "#4c0519",
    borderColor: "#fb7185",
  },
  optionMissed: {
    borderColor: "#34d399",
  },
  optionCheck: {
    color: "#7dd3fc",
    fontWeight: "700",
  },
  optionText: {
    color: "#e2e8f0",
    flex: 1,
    fontSize: 15,
    lineHeight: 21,
  },
  resultBox: {
    borderRadius: 8,
    borderWidth: 1,
    gap: 8,
    padding: 12,
  },
  resultCorrect: {
    backgroundColor: "#064e3b",
    borderColor: "#34d399",
  },
  resultWrong: {
    backgroundColor: "#4c0519",
    borderColor: "#fb7185",
  },
  resultTitle: {
    color: "#f8fafc",
    fontSize: 16,
    fontWeight: "800",
  },
  resultText: {
    color: "#e2e8f0",
    fontSize: 14,
    lineHeight: 20,
  },
  chatBlock: {
    gap: 10,
  },
  chatPanel: {
    backgroundColor: "#020617",
    borderColor: "#334155",
    borderRadius: 8,
    borderWidth: 1,
    gap: 10,
    padding: 10,
  },
  chatBubble: {
    borderRadius: 8,
    maxWidth: "92%",
    padding: 10,
  },
  assistantBubble: {
    alignSelf: "flex-start",
    backgroundColor: "#1e293b",
  },
  userBubble: {
    alignSelf: "flex-end",
    backgroundColor: "#0369a1",
  },
  chatText: {
    color: "#f8fafc",
    fontSize: 13,
    lineHeight: 19,
  },
  followUpRow: {
    alignItems: "center",
    flexDirection: "row",
    gap: 8,
  },
  followUpInput: {
    backgroundColor: "#0f172a",
    borderColor: "#334155",
    borderRadius: 8,
    borderWidth: 1,
    color: "#f8fafc",
    flex: 1,
    paddingHorizontal: 10,
    paddingVertical: 9,
  },
  errorText: {
    color: "#fda4af",
    fontSize: 12,
  },
  reportBlock: {
    gap: 10,
  },
  reportPanel: {
    gap: 10,
  },
  reportInput: {
    backgroundColor: "#020617",
    borderColor: "#334155",
    borderRadius: 8,
    borderWidth: 1,
    color: "#f8fafc",
    minHeight: 96,
    padding: 10,
    textAlignVertical: "top",
  },
});
