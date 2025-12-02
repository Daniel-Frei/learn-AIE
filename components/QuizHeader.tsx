// components/QuizHeader.tsx
"use client";

import { useRef } from "react";
import type { Mode, DifficultyRange } from "../lib/useQuiz";

type Props = {
  title: string;
  mode: Mode;
  difficultyRange: DifficultyRange;
  changeMode: (mode: Mode) => void;
  changeDifficultyRange: (range: DifficultyRange) => void;
  answeredCount: number;
  correctCount: number;
  accuracy: number;
  exportDifficultyJson: () => string;
  importDifficultyFromJson: (json: string) => void;
};

export default function QuizHeader({
  title,
  mode,
  difficultyRange,
  changeMode,
  changeDifficultyRange,
  answeredCount,
  correctCount,
  accuracy,
  exportDifficultyJson,
  importDifficultyFromJson,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleExportClick = () => {
    const json = exportDifficultyJson();
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "difficulty-stats.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        importDifficultyFromJson(reader.result);
      }
      e.target.value = "";
    };
    reader.readAsText(file);
  };

  return (
    <header className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
      <div>
        <h1 className="text-2xl md:text-3xl font-semibold">{title}</h1>
        <p className="text-sm text-slate-400 mt-1">
          Multi-select questions –{" "}
          <span className="font-semibold">select all TRUE statements</span> and
          then submit.
        </p>
      </div>

      <div className="flex flex-col items-start md:items-end gap-2">
        <label className="text-sm text-slate-300 flex items-center gap-2">
          <span>Question source:</span>
          <select
            className="bg-slate-800 border border-slate-700 rounded-md px-2 py-1 text-sm"
            value={mode}
            onChange={(e) => changeMode(e.target.value as Mode)}
          >
            <option value="chapter-1">Chapter 1 only</option>
            <option value="chapter-2">Chapter 2 only</option>
            <option value="chapter-3">Chapter 3 only</option>
            <option value="aie-build-app-ch2">AIE build app - Chap 2</option>
            <option value="all">All chapters</option>
          </select>
        </label>

        {/* Difficulty range */}
        <label className="text-sm text-slate-300 flex items-center gap-2">
          <span>Difficulty range:</span>
          <input
            type="number"
            min={0}
            max={100}
            value={difficultyRange.min}
            onChange={(e) =>
              changeDifficultyRange({
                ...difficultyRange,
                min: Number(e.target.value),
              })
            }
            className="w-16 bg-slate-800 border border-slate-700 rounded-md px-2 py-1 text-sm"
          />
          <span>to</span>
          <input
            type="number"
            min={0}
            max={100}
            value={difficultyRange.max}
            onChange={(e) =>
              changeDifficultyRange({
                ...difficultyRange,
                max: Number(e.target.value),
              })
            }
            className="w-16 bg-slate-800 border border-slate-700 rounded-md px-2 py-1 text-sm"
          />
        </label>

        {/* Export / Import buttons */}
        <div className="flex gap-2">
          <button
            type="button"
            onClick={handleExportClick}
            className="px-3 py-1 rounded-md bg-slate-800 border border-slate-700 text-xs"
          >
            Export difficulty
          </button>
          <button
            type="button"
            onClick={handleImportClick}
            className="px-3 py-1 rounded-md bg-slate-800 border border-slate-700 text-xs"
          >
            Import difficulty
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="application/json"
            className="hidden"
            onChange={handleFileChange}
          />
        </div>

        <div className="text-xs text-slate-400">
          Answered:{" "}
          <span className="font-semibold text-slate-200">{answeredCount}</span>{" "}
          · Correct:{" "}
          <span className="font-semibold text-emerald-300">
            {correctCount}
          </span>{" "}
          · Accuracy: <span className="font-semibold">{accuracy}%</span>
        </div>
      </div>
    </header>
  );
}
