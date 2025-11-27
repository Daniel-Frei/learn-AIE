// components/QuestionExplanationChat.tsx
"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Question } from "../lib/quiz";

type ChatRole = "user" | "assistant";

type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
};

type OptionForChat = {
  text: string;
  isCorrect: boolean;
  selected: boolean;
};

type Props = {
  question: Question;
  options: OptionForChat[];
  isOverallCorrect: boolean;
};

export default function QuestionExplanationChat({
  question,
  options,
  isOverallCorrect,
}: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const hasStarted = messages.length > 0;

  const callApi = async (chatHistory: { role: ChatRole; content: string }[]) => {
    const res = await fetch("/api/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        questionPrompt: question.prompt,
        genericExplanation: question.explanation,
        options,
        isOverallCorrect,
        chatHistory,
      }),
    });

    if (!res.ok) {
      throw new Error(`Request failed with status ${res.status}`);
    }

    const data = await res.json();
    const replyText: string =
      data.reply ?? "Sorry, I couldn't generate an explanation.";
    return replyText;
  };

  const startChat = async () => {
    if (hasStarted) {
      setIsOpen(true);
      return;
    }

    setIsOpen(true);
    setIsLoading(true);
    setError(null);

    try {
      // No prior chat history on the first request
      const replyText = await callApi([]);

      const reply: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: replyText,
      };

      setMessages([reply]);
    } catch (err) {
      console.error(err);
      setError("Could not load a detailed explanation. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const sendFollowUp = async () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
    };

    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setInput("");
    setIsLoading(true);
    setError(null);

    try {
      const replyText = await callApi(
        nextMessages.map(({ role, content }) => ({ role, content }))
      );

      const reply: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: replyText,
      };

      setMessages([...nextMessages, reply]);
    } catch (err) {
      console.error(err);
      setError("Could not send your question. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="mt-4 space-y-2">
      {/* Trigger button */}
      <button
        type="button"
        onClick={startChat}
        className="inline-flex items-center gap-2 rounded-lg border border-sky-500/70 bg-slate-900/40 px-3 py-2 text-xs font-semibold text-sky-200 hover:bg-slate-800/80"
      >
        💬 Ask for a detailed explanation
      </button>

      {/* Chat panel */}
      {isOpen && (
        <div className="mt-2 rounded-xl border border-slate-700 bg-slate-950/60 p-3 space-y-3">
          <div className="max-h-64 space-y-2 overflow-y-auto text-xs pr-1">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${
                  msg.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[80%] rounded-lg px-3 py-2 ${
                    msg.role === "user"
                      ? "bg-sky-600 text-slate-50"
                      : "bg-slate-800 text-slate-100"
                  }`}
                >
                  {msg.role === "assistant" ? (
                    <div className="text-[11px] leading-relaxed space-y-1">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          p: (props) => (
                            <p {...props} className="mb-1" />
                          ),
                          ul: (props) => (
                            <ul
                              {...props}
                              className="list-disc ml-4 mb-1"
                            />
                          ),
                          ol: (props) => (
                            <ol
                              {...props}
                              className="list-decimal ml-4 mb-1"
                            />
                          ),
                          li: (props) => (
                            <li {...props} className="mb-0.5" />
                          ),
                          h3: (props) => (
                            <h3
                              {...props}
                              className="font-semibold mt-2 mb-1"
                            />
                          ),
                          hr: (props) => (
                            <hr
                              {...props}
                              className="border-slate-700 my-2"
                            />
                          ),
                          strong: (props) => (
                            <strong
                              {...props}
                              className="font-semibold"
                            />
                          ),
                        }}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <span className="text-[11px] leading-relaxed">
                      {msg.content}
                    </span>
                  )}
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="text-[11px] text-slate-400">
                Thinking about a better explanation…
              </div>
            )}

            {error && (
              <div className="text-[11px] text-rose-300">{error}</div>
            )}
          </div>

          {/* Input for follow-up questions */}
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a follow-up (e.g. explain <UNK> vs subwords again)"
              className="flex-1 rounded-lg border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-100 placeholder:text-slate-500"
            />
            <button
              type="button"
              onClick={sendFollowUp}
              disabled={isLoading || !input.trim()}
              className="rounded-lg bg-sky-500 px-3 py-1 text-xs font-semibold text-slate-950 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Send
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
