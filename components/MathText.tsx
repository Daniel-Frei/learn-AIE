// components/MathText.tsx
"use client";

import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
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

  const components: Components = {
    p: ({ children }) =>
      inline ? (
        <span className={className}>{children}</span>
      ) : (
        <p className="leading-relaxed">{children}</p>
      ),
    table: ({ children }) => (
      <div className="my-4 max-w-full overflow-x-auto rounded-lg border border-slate-700 bg-slate-950/40">
        <table className="min-w-full border-collapse text-left text-sm font-normal leading-6 text-slate-200 md:text-base">
          {children}
        </table>
      </div>
    ),
    thead: ({ children }) => (
      <thead className="bg-slate-900/80 text-slate-100">{children}</thead>
    ),
    tbody: ({ children }) => (
      <tbody className="divide-y divide-slate-800">{children}</tbody>
    ),
    th: ({ children, style }) => (
      <th
        style={style}
        className="border-b border-slate-700 px-3 py-2 align-bottom font-semibold text-slate-100"
      >
        {children}
      </th>
    ),
    td: ({ children, style }) => (
      <td style={style} className="px-3 py-2 align-top text-slate-200">
        {children}
      </td>
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
