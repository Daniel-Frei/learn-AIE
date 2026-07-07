"use client";

import { Download } from "lucide-react";

export default function AgentNativeMemoryPdfExportButton() {
  return (
    <button
      type="button"
      data-testid="presentation-pdf-export"
      className="agent-native-presentation-no-print inline-flex items-center gap-2 rounded-lg border border-cyan-300/40 px-5 py-3 text-sm font-bold text-cyan-100 transition-colors hover:border-cyan-300 hover:text-cyan-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300"
      onClick={() => window.print()}
    >
      <Download aria-hidden="true" size={18} />
      Export PDF
    </button>
  );
}
