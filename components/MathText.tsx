// components/MathText.tsx
"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { normalizeMathDelimiters } from "../lib/normalizeMath";

type Props = {
  text: string;
  className?: string;
  inline?: boolean;
};

export default function MathText({ text, className, inline }: Props) {
  const normalized = normalizeMathDelimiters(text);

  const components = {
    p: ({ children }: { children: React.ReactNode }) =>
      inline ? (
        <span className={className}>{children}</span>
      ) : (
        <p className="leading-relaxed">{children}</p>
      ),
  };

  if (inline) {
    return (
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={components}
      >
        {normalized}
      </ReactMarkdown>
    );
  }

  return (
    <div className={className}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={components}
      >
        {normalized}
      </ReactMarkdown>
    </div>
  );
}
