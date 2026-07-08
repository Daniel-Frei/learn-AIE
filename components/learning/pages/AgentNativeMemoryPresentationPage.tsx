"use client";

import Image from "next/image";
import { createPortal } from "react-dom";
import { useEffect, useState, type ReactNode } from "react";
import {
  ArrowRight,
  Database,
  Gauge,
  Layers3,
  Maximize2,
  Search,
  ShieldCheck,
  Wrench,
  X,
} from "lucide-react";
import AgentNativeMemoryPdfExportButton from "./AgentNativeMemoryPdfExportButton";

const assetBase = "/learning/agent-native-memory/presentation";

const sectionThreeAssetBase = `${assetBase}/section-3`;
const sectionFourAssetBase = `${assetBase}/section-4`;
const sectionFiveAssetBase = `${assetBase}/section-5`;
const sectionSixAssetBase = `${assetBase}/section-6`;

const paperAssets = {
  figure2: {
    src: `${sectionThreeAssetBase}/figure-2-representation.png`,
    alt: "Figure 2 from the paper showing token sequence, graph tree topology, and heterogeneous composite memory representation methods",
    width: 1140,
    height: 625,
  },
  figure3: {
    src: `${sectionThreeAssetBase}/figure-3-storage.png`,
    alt: "Figure 3 from the paper showing transient in-context, specialized single-engine, and heterogeneous multi-engine storage methods",
    width: 1130,
    height: 528,
  },
  figure4: {
    src: `${sectionThreeAssetBase}/figure-4-extraction.png`,
    alt: "Figure 4 from the paper showing raw sequence concatenation, schema-free semantic extraction, and schema-constrained structured extraction",
    width: 990,
    height: 715,
  },
  figure5: {
    src: `${sectionThreeAssetBase}/figure-5-retrieval.png`,
    alt: "Figure 5 from the paper showing native attention, dense retrieval, subgraph traversal, agentic routing, and hybrid retrieval methods",
    width: 1017,
    height: 1009,
  },
  figure6: {
    src: `${sectionThreeAssetBase}/figure-6-maintenance.png`,
    alt: "Figure 6 from the paper showing timestamp versioning, eviction, semantic consolidation, CRUD updates, and parametric optimization",
    width: 1019,
    height: 1035,
  },
  table1: {
    src: `${sectionThreeAssetBase}/table-1-taxonomy.png`,
    alt: "Table 1 from the paper mapping agent memory systems across representation, storage, extraction, retrieval, and maintenance",
    width: 2145,
    height: 855,
  },
  table1Infographic: {
    src: `${sectionThreeAssetBase}/table-1-infographic.png`,
    alt: "Simplified visual table mapping memory systems to representation, storage, extraction, retrieval routing, and maintenance choices",
    width: 1672,
    height: 941,
  },
  figure7: {
    src: `${sectionFourAssetBase}/figure-7-effectiveness.png`,
    alt: "Figure 7 from the paper showing end-to-end effectiveness of memory systems across LongMemEval, LoCoMo, and DB-Bench",
    width: 2096,
    height: 942,
  },
  figure8: {
    src: `${sectionFourAssetBase}/figure-8-retrieval-fidelity.png`,
    alt: "Figure 8 from the paper showing retrieval recall and evidence distance degradation over LoCoMo",
    width: 1005,
    height: 855,
  },
  table2: {
    src: `${sectionFourAssetBase}/table-2-update-robustness.png`,
    alt: "Table 2 from the paper showing robustness over memory update settings",
    width: 1045,
    height: 765,
  },
  figure10: {
    src: `${sectionFourAssetBase}/figure-10-long-horizon.png`,
    alt: "Figure 10 from the paper showing context length, session history, and temporal distance robustness",
    width: 1005,
    height: 740,
  },
  figure11: {
    src: `${sectionFourAssetBase}/figure-11-operation-cost.png`,
    alt: "Figure 11 from the paper showing memory operation cost and latency trade-offs",
    width: 1020,
    height: 450,
  },
  metricsOverview: {
    src: `${sectionFourAssetBase}/metrics.png`,
    alt: "Infographic explaining Exact Match, Answer F1, ROUGE-L F1, and ROUGE-L Recall evaluation metrics",
    width: 1672,
    height: 941,
  },
  table3: {
    src: `${sectionFiveAssetBase}/table-3-representation-storage.png`,
    alt: "Table 3 from the paper showing ablations of representation and storage mechanisms",
    width: 1005,
    height: 535,
  },
  table4: {
    src: `${sectionFiveAssetBase}/table-4-extraction.png`,
    alt: "Table 4 from the paper showing ablations of memory extraction strategies",
    width: 1045,
    height: 505,
  },
  table5: {
    src: `${sectionFiveAssetBase}/table-5-retrieval-routing.png`,
    alt: "Table 5 from the paper showing ablations of retrieval and routing mechanisms",
    width: 1005,
    height: 365,
  },
  figure12: {
    src: `${sectionFiveAssetBase}/figure-12-maintenance.png`,
    alt: "Figure 12 from the paper showing ablations of maintenance strategies",
    width: 1005,
    height: 555,
  },
  table1EmpiricalHighlights: {
    src: `${sectionSixAssetBase}/table-1-empirical-highlights.png`,
    alt: "Highlighted system table showing empirical findings across representation, storage, extraction, retrieval routing, and maintenance choices",
    width: 1672,
    height: 941,
  },
} as const;

const slideNav = [
  ["Title", "Paper reference"],
  ["Agenda", "Talk map"],
  ["1", "Introduction"],
  ["Problem", "Systems evaluation"],
  ["Failures", "Modular causes"],
  ["2", "Preliminaries"],
  ["Lens", "Four modules"],
  ["Boundaries", "Not RAG"],
  ["3", "Method overview"],
  ["Method", "Design-space map"],
  ["Table 1", "Paper table"],
  ["Table 1", "Infographic"],
  ["Figure 2", "Representation"],
  ["Figure 3", "Storage"],
  ["Figure 4", "Extraction"],
  ["Figure 5", "Retrieval"],
  ["Figure 6", "Maintenance"],
  ["Table 1", "Reminder"],
  ["4", "End-to-end"],
  ["Benchmarks", "What is tested"],
  ["Metrics", "Metric guide"],
  ["RQ1", "Workload fit"],
  ["Fig 7", "Effectiveness"],
  ["RQ2", "Evidence assembly"],
  ["Fig 8", "Retrieval"],
  ["RQ3", "Updates"],
  ["Table 2", "Robustness"],
  ["RQ4", "Long horizon"],
  ["Fig 10", "Stability"],
  ["RQ5", "Cost"],
  ["Fig 11", "Cost"],
  ["5", "Components"],
  ["Diagnosis", "Ablations"],
  ["M1", "Representation"],
  ["Table 3", "Representation"],
  ["M2", "Extraction"],
  ["Table 4", "Extraction"],
  ["M3", "Retrieval"],
  ["Table 5", "Retrieval"],
  ["M4", "Maintenance"],
  ["Fig 12", "Maintenance"],
  ["6", "Conclusion"],
  ["Highlights", "Findings table"],
  ["Close", "Readiness"],
  ["Appendix", "Backup"],
  ["Figure 1", "Architectures"],
  ["Evaluation", "Five RQs"],
  ["Primer", "Paper buckets"],
  ["Systems", "Table 1 buckets"],
  ["Lineup", "Systems and pressures"],
] as const;

const agendaItems = [
  ["1", "Introduction", "why final-answer scores miss memory failures"],
  ["2", "Preliminaries", "what the paper means by agent memory"],
  ["3", "Method overview", "the four-module taxonomy of memory systems"],
  [
    "4",
    "End-to-end assessment",
    "five RQs across effectiveness, evidence, updates, horizons, and cost",
  ],
  ["5", "Component comparison", "which module choices cause the failures"],
  ["6", "Conclusion", "what readiness requires in practice"],
] as const;

const sectionDividers = {
  introduction: {
    number: "1",
    title: "Introduction",
    thesis: "The problem: benchmarks show scores, not causes.",
    beats: [
      "fragmented architectures",
      "black-box scoreboards",
      "system-level readiness",
    ],
  },
  preliminaries: {
    number: "2",
    title: "Preliminaries",
    thesis: "Defining agent memory and distinguishing it from RAG.",
    beats: ["Memory types", "Lifecycle modules", "Semantic workloads"],
  },
  method: {
    number: "3",
    title: "Method Overview",
    thesis: "Breaks agent memory into modular design choices.",
    beats: [
      "Memory representation",
      "Extraction strategies",
      "Routing/maintenance",
    ],
  },
  assessment: {
    number: "4",
    title: "End-to-End Assessment",
    thesis: "Evaluating memory systems across workloads and failure modes",
    beats: ["Task effectiveness", "Retrieval fidelity", "Update robustness"],
  },
  components: {
    number: "5",
    title: "Fine-Grained Component Comparison",
    thesis: "Isolating which memory components cause performance differences.",
    beats: [
      "Component ablations",
      "Representation granularity",
      "Conservative consolidation",
    ],
  },
  conclusion: {
    number: "6",
    title: "Conclusion",
    thesis:
      "The answer is not a single winning memory architecture. It is workload-aware lifecycle design.",
    beats: [
      "preserve evidence",
      "retrieve reliably",
      "update correctly",
      "bound cost",
    ],
  },
  appendix: {
    number: "A",
    title: "Appendix",
    thesis: "Backup architecture and system-lineup context for discussion.",
    beats: ["architecture buckets", "system examples", "evaluation lineup"],
  },
} as const;

const moduleCards = [
  {
    letter: "R",
    title: "Represent and Store",
    caption: "the shape and substrate of memory",
    icon: Database,
    tone: "from-cyan-400 to-blue-500",
  },
  {
    letter: "S",
    title: "Extract",
    caption: "what gets written from experience",
    icon: Layers3,
    tone: "from-emerald-300 to-teal-500",
  },
  {
    letter: "Q",
    title: "Retrieve and Route",
    caption: "how evidence returns to the agent",
    icon: Search,
    tone: "from-amber-300 to-orange-500",
  },
  {
    letter: "U",
    title: "Maintain",
    caption: "update, merge, evict, and version",
    icon: Wrench,
    tone: "from-fuchsia-300 to-violet-500",
  },
] as const;

const figureOneArchitectures = [
  {
    name: "Stream + reflection",
    visual: ["Event log", "Reflection", "Write-back"],
    point: "Keeps experience as a time-ordered stream",
  },
  {
    name: "Hierarchical tiers",
    visual: ["Core", "Summary", "Archive"],
    point: "Moves memories between fast and long-term stores",
  },
  {
    name: "Knowledge graph",
    visual: ["Entity", "Relation", "Version"],
    point: "Turns facts into linked, timestamped structure",
  },
  {
    name: "Hybrid multi-store",
    visual: ["Text", "Vector", "Graph"],
    point: "Routes across several backends and indexes",
  },
] as const;

const benchmarkLimitations = [
  [
    "End-to-end metrics only",
    "cannot locate storage, extraction, retrieval, or maintenance failures",
  ],
  [
    "Narrow system coverage",
    "architectures are rarely compared in one unified setup",
  ],
  [
    "Weak update testing",
    "stale or contradicted facts can survive as current memory",
  ],
  [
    "Little cost analysis",
    "a high-scoring system may be too slow or expensive to operate",
  ],
  [
    "Monolithic evaluation",
    "no module-level understanding of design trade-offs",
  ],
] as const;

const scopeContrasts = [
  {
    label: "RAG",
    old: "Read-only retrieval from a mostly static corpus",
    memory: "writes, revises, and governs agent-specific state",
  },
  {
    label: "Context engineering",
    old: "Chooses what enters the finite prompt right now",
    memory: "decides what is stored, maintained, and retrievable later",
  },
  {
    label: "Traditional DB workload",
    old: "Exact predicates over cleaner transactional or analytic data",
    memory: "semantic access over partial, conflicting, heterogeneous traces",
  },
] as const;

const architecturePrimer = [
  {
    family: "Reference baselines",
    stores: "prompt history or static chunks outside a memory lifecycle",
    answers: "attend over context or retrieve from a mostly fixed corpus",
    risk: "useful anchors, but weak tests of update and maintenance behavior",
  },
  {
    family: "Sequential context",
    stores: "ordered traces, summaries, topics, or extracted user facts",
    answers: "retrieve by attention, semantic similarity, or topic routing",
    risk: "distant evidence and conflicting updates can blur together",
  },
  {
    family: "Structural / topological",
    stores: "trees, temporal knowledge graphs, entities, and relations",
    answers: "walk graph or tree structure around the query",
    risk: "schema quality and entity linking become the bottleneck",
  },
  {
    family: "Multi-paradigm hybrid",
    stores: "composite objects, tiers, pages, vectors, graphs, or SQL records",
    answers:
      "combine filters, dense search, traversal, planning, and reranking",
    risk: "coordination cost rises when every write touches the whole system",
  },
] as const;

const architectureExamples = [
  [
    "Reference baselines",
    "Long Context, Embedding RAG",
    "prompt-only or static retrieval",
  ],
  [
    "Sequential context",
    "MemoChat, Mem0, MemAgent",
    "streamed traces, summaries, or extracted facts",
  ],
  [
    "Structural / topological",
    "MemTree, Zep, Cognee",
    "tree, graph, or relation-aware organization",
  ],
  [
    "Multi-paradigm hybrid",
    "Letta, LightMem, SimpleMem, MemOS, MemoryOS, A-MEM",
    "multiple stores, routes, tiers, or maintenance policies",
  ],
] as const;

const methodModuleQuestions = [
  [
    "R",
    "Represent / store",
    "What form does memory take, and where is it stored?",
  ],
  ["S", "Extract", "What gets written into memory?"],
  ["Q", "Retrieve / route", "How does relevant memory come back?"],
  [
    "U",
    "Maintain",
    "How does memory stay current, bounded, and non-contradictory?",
  ],
] as const;

const benchmarkExplainers = [
  {
    name: "LoCoMo",
    source: "very long conversations",
    measures: "episodic, temporal, open-domain dialogue memory",
    scale:
      "about 300 turns and 9K tokens per conversation, across up to 35 sessions",
    metrics: "Exact Match and Answer F1 in this paper",
  },
  {
    name: "LongMemEval",
    source: "multi-session user-assistant histories",
    measures:
      "information extraction, multi-session reasoning, knowledge updates, temporal reasoning, abstention",
    scale: "500 curated questions over long, extensible histories",
    metrics: "Substring EM, ROUGE-L, and LLM judge accuracy",
  },
  {
    name: "DB-Bench",
    source: "Database environment from LifelongAgentBench",
    measures:
      "whether memory preserves ordered procedural state across operations",
    scale: "interactive, skill-grounded database tasks",
    metrics: "Exact Match and Task Success Rate",
  },
  {
    name: "LongBench",
    source: "long-context understanding benchmark",
    measures: "context-length robustness under short, medium, and long buckets",
    scale: "21 datasets across 6 long-context task categories",
    metrics: "Accuracy by context-length bucket",
  },
] as const;

const sectionFourQuestions = [
  [
    "RQ1",
    "task effectiveness",
    "Are memory systems effective across workloads?",
  ],
  ["RQ2", "retrieval fidelity", "Do they retrieve the right evidence?"],
  ["RQ3", "update robustness", "Do they handle updates and stale facts?"],
  ["RQ4", "long-horizon stability", "Do they stay stable over long horizons?"],
  ["RQ5", "operational cost", "What do they cost to run?"],
] as const;

const sectionFourScaleCards = [
  ["12", "memory systems"],
  ["5", "benchmark workloads"],
  ["11", "datasets"],
  ["5", "evaluation lenses"],
] as const;

const rqFindings = [
  {
    label: "RQ1",
    testId: "presentation-rq1-effectiveness",
    title: "No Universal Winner",
    tested: "end-to-end task success",
    figure: paperAssets.figure7,
    figureLabel: "Figure 7: effectiveness across workloads and metrics",
    beats: [
      "No single memory architecture wins everywhere.",
      "Exact Match misses synthesis/execution success.",
      "Finding 1: Match memory design to workload bottleneck.",
    ],
  },
  {
    label: "RQ2",
    testId: "presentation-rq2-retrieval-fidelity",
    title: "Retrieval",
    tested: "retrieval fidelity",
    figure: paperAssets.figure8,
    figureLabel: "Figure 8: retrieval recall and evidence-distance drift",
    beats: [
      "Retrieval is evidence assembly, not top-1 ranking.",
      "Structured links/hierarchies help with distant evidence.",
      "Finding 2: Organized evidence beats flat similarity search.",
    ],
  },
  {
    label: "RQ3",
    testId: "presentation-rq3-update-robustness",
    title: "Memory Updates",
    tested: "dynamic memory robustness",
    figure: paperAssets.table2,
    figureLabel: "Table 2: robustness over memory update settings",
    beats: [
      "Structured temporal evidence handles updates best.",
      "Stronger LLMs help after grounding, not before.",
      "Finding 3: Update robustness is a pipeline-design problem.",
    ],
  },
  {
    label: "RQ4",
    testId: "presentation-rq4-long-horizon",
    title: "Horizon-Structured Memory",
    tested: "long-horizon stability",
    figure: paperAssets.figure10,
    figureLabel:
      "Figure 10: context length, session growth, and distance drift",
    beats: [
      "Long-term stability needs links or hierarchy.",
      "More context alone degrades with distractors.",
      "Finding 4: Choose abstractions, not just more storage.",
    ],
  },
  {
    label: "RQ5",
    testId: "presentation-rq5-operation-cost",
    title: "Latency",
    tested: "cost-performance trade-off",
    figure: paperAssets.figure11,
    figureLabel: "Figure 11: operation cost and latency frontier",
    beats: [
      "Localized maintenance is most cost-efficient.",
      "Global reorganization becomes very expensive.",
      "Finding 5: Maintenance scope drives the cost-utility trade-off.",
    ],
  },
] as const;

const componentLessons = [
  {
    label: "M1",
    testId: "presentation-m1-representation-ablation",
    module: "representation and storage",
    title: "Preserve evidence first",
    figure: paperAssets.table3,
    figureLabel: "Table 3: representation and storage ablations",
    beats: [
      "Preserve original content over stronger abstraction.",
      "Raw/high-retention memory best supports exact detail recall.",
      "Compression can preserve reasoning, but weakens exact matching.",
      "Finding 6: Granularity matters more than compactness or structure.",
    ],
  },
  {
    label: "M2",
    testId: "presentation-m2-extraction-ablation",
    module: "extraction",
    title: "Write broadly, filter later",
    figure: paperAssets.table4,
    figureLabel: "Table 4: memory extraction strategy ablations",
    beats: [
      "Coverage-preserving extraction is most stable.",
      "Avoid aggressive write-time filtering.",
      "Coarser segmentation keeps related cues together.",
      "Finding 7: Preserve context first; filter later.",
    ],
  },
  {
    label: "M3",
    testId: "presentation-m3-retrieval-ablation",
    module: "retrieval and routing",
    title: "Targeted structure beats extra deliberation",
    figure: paperAssets.table5,
    figureLabel: "Table 5: retrieval and routing ablations",
    beats: [
      "Planning and balanced fusion improve retrieval.",
      "Moderate hybrid fusion beats sparse-heavy retrieval.",
      "Extra reflection adds little and may hurt.",
      "Finding 8: Targeted structure beats added complexity.",
    ],
  },
  {
    label: "M4",
    testId: "presentation-m4-maintenance-ablation",
    module: "maintenance",
    title: "Consolidate carefully",
    figure: paperAssets.figure12,
    figureLabel: "Figure 12: maintenance strategy ablations",
    beats: [
      "Conservative consolidation works best.",
      "Delayed flushing leaves evidence fragmented.",
      "Overly coarse summaries hide useful cues.",
      "Finding 9: Balanced updates preserve long-horizon consistency.",
    ],
  },
] as const;

const sectionFiveFrame = [
  ["R", "Representation", "Does abstraction preserve usable evidence?"],
  ["S", "Extraction", "What should get written into memory?"],
  ["Q", "Retrieval", "How should queries be routed?"],
  ["U", "Maintenance", "When should memory merge or forget?"],
] as const;

const conclusionContributions = [
  [Gauge, "no universal winner", "workload determines what works"],
  [Layers3, "evidence preservation", "details are easily lost"],
  [Search, "retrieval fidelity", "assemble evidence, not just top-1"],
  [ShieldCheck, "update fidelity", "stale facts remain hard"],
  [Wrench, "bounded cost", "local maintenance scales better"],
] as const;

type SlideProps = {
  id: string;
  index: string;
  eyebrow: string;
  title: string;
  subtitle?: string;
  children: ReactNode;
  tone?: "dark" | "light" | "blue" | "green";
  layout?: "standard" | "wide" | "visualOnly";
  hideHiddenTitle?: boolean;
};

function Slide({
  id,
  index,
  eyebrow,
  title,
  subtitle,
  children,
  tone = "dark",
  layout = "standard",
  hideHiddenTitle = false,
}: SlideProps) {
  const tones = {
    dark: "bg-[#07111f] text-white",
    light: "bg-slate-50 text-slate-950",
    blue: "bg-[#eaf6ff] text-slate-950",
    green: "bg-[#ebfff7] text-slate-950",
  };

  if (layout === "visualOnly") {
    return (
      <section
        id={id}
        className={`agent-native-presentation-slide relative min-h-[calc(100vh-4rem)] scroll-mt-16 snap-start overflow-hidden px-4 pb-12 pt-20 md:px-8 lg:px-10 ${tones[tone]}`}
      >
        <div className="agent-native-presentation-slide-wide mx-auto flex min-h-[calc(100vh-10rem)] w-full max-w-7xl flex-col justify-center">
          <div className="agent-native-presentation-slide-copy relative z-10 mb-4 max-w-5xl">
            <p
              className={`agent-native-presentation-slide-eyebrow text-sm font-semibold uppercase tracking-wide ${
                tone === "dark" ? "text-cyan-300" : "text-blue-700"
              }`}
            >
              {index} / {eyebrow}
            </p>
            {hideHiddenTitle ? null : (
              <>
                <h2 className="sr-only">{title}</h2>
                {subtitle ? <p className="sr-only">{subtitle}</p> : null}
              </>
            )}
          </div>
          <div className="agent-native-presentation-slide-visual relative z-10 min-w-0">
            {children}
          </div>
        </div>
      </section>
    );
  }

  return (
    <section
      id={id}
      className={`agent-native-presentation-slide relative min-h-[calc(100vh-4rem)] scroll-mt-16 snap-start overflow-hidden px-4 pb-12 pt-20 md:px-8 lg:px-10 ${tones[tone]}`}
    >
      {layout === "wide" ? (
        <div className="agent-native-presentation-slide-wide mx-auto flex min-h-[calc(100vh-10rem)] w-full max-w-7xl flex-col justify-start gap-4">
          <div className="agent-native-presentation-slide-copy relative z-10 w-full">
            <p
              className={`agent-native-presentation-slide-eyebrow text-sm font-semibold uppercase tracking-wide ${
                tone === "dark" ? "text-cyan-300" : "text-blue-700"
              }`}
            >
              {index} / {eyebrow}
            </p>
            <h2 className="agent-native-presentation-slide-title mt-2 text-3xl font-semibold tracking-normal md:text-5xl md:leading-tight">
              {title}
            </h2>
            {subtitle ? (
              <p
                className={`agent-native-presentation-slide-subtitle mt-2 max-w-4xl text-base leading-7 ${
                  tone === "dark" ? "text-slate-300" : "text-slate-700"
                }`}
              >
                {subtitle}
              </p>
            ) : null}
          </div>
          <div className="agent-native-presentation-slide-visual relative z-10 min-w-0">
            {children}
          </div>
        </div>
      ) : (
        <div className="agent-native-presentation-slide-grid mx-auto grid min-h-[calc(100vh-10rem)] w-full max-w-7xl items-center gap-8 lg:grid-cols-[0.76fr_1.24fr]">
          <div className="agent-native-presentation-slide-copy relative z-10 max-w-2xl">
            <p
              className={`agent-native-presentation-slide-eyebrow text-sm font-semibold uppercase tracking-wide ${
                tone === "dark" ? "text-cyan-300" : "text-blue-700"
              }`}
            >
              {index} / {eyebrow}
            </p>
            <h2 className="agent-native-presentation-slide-title mt-4 text-4xl font-semibold tracking-normal md:text-6xl md:leading-tight">
              {title}
            </h2>
            {subtitle ? (
              <p
                className={`agent-native-presentation-slide-subtitle mt-5 max-w-xl text-lg leading-8 ${
                  tone === "dark" ? "text-slate-300" : "text-slate-700"
                }`}
              >
                {subtitle}
              </p>
            ) : null}
          </div>
          <div className="agent-native-presentation-slide-visual relative z-10 min-w-0">
            {children}
          </div>
        </div>
      )}
    </section>
  );
}

function PaperFigure({
  asset,
  label,
  caption = label,
  className = "",
  variant = "standard",
}: {
  asset: (typeof paperAssets)[keyof typeof paperAssets];
  label: string;
  caption?: string | null;
  className?: string;
  variant?: "standard" | "large" | "full";
}) {
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsOpen(false);
      }
    };

    document.addEventListener("keydown", onKeyDown);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = previousOverflow;
    };
  }, [isOpen]);

  const imageClassName =
    variant === "full"
      ? "max-h-[68vh] w-full rounded-md object-contain"
      : variant === "large"
        ? "max-h-[54vh] w-full rounded-md object-contain"
        : "h-auto w-full rounded-md object-contain";

  return (
    <>
      <figure
        className={`overflow-hidden rounded-lg border border-slate-200 bg-white p-2 shadow-xl ${className}`}
      >
        <button
          type="button"
          className="group relative block w-full rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-600"
          aria-label={`Enlarge ${label}`}
          onClick={() => setIsOpen(true)}
        >
          <Image
            src={asset.src}
            alt={asset.alt}
            width={asset.width}
            height={asset.height}
            loading="eager"
            unoptimized
            sizes={
              variant === "standard"
                ? "(min-width: 1024px) 42vw, 100vw"
                : "(min-width: 1024px) 72vw, 100vw"
            }
            className={imageClassName}
          />
          <span className="agent-native-presentation-no-print absolute right-3 top-3 inline-flex items-center gap-2 rounded-full bg-slate-950/85 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-white opacity-0 shadow-lg transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100">
            <Maximize2 aria-hidden="true" size={14} />
            Enlarge
          </span>
        </button>
        {caption ? (
          <figcaption className="mt-2 px-2 pb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
            {caption}
          </figcaption>
        ) : null}
      </figure>
      {isOpen && typeof document !== "undefined"
        ? createPortal(
            <div
              role="dialog"
              aria-modal="true"
              aria-label={label}
              data-testid="presentation-figure-lightbox"
              className="agent-native-presentation-no-print fixed inset-0 z-[100] flex items-center justify-center bg-slate-950/95 p-0"
            >
              <button
                type="button"
                aria-label="Close enlarged figure"
                data-testid="presentation-lightbox-close"
                className="absolute right-6 top-6 z-[110] inline-flex h-14 w-14 items-center justify-center rounded-full border border-white/70 bg-white text-slate-950 shadow-2xl transition-colors hover:bg-cyan-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300"
                onClick={() => setIsOpen(false)}
              >
                <X aria-hidden="true" size={30} strokeWidth={2.8} />
              </button>
              <div className="flex h-full w-full flex-col items-center justify-center gap-3 px-3 pb-4 pt-16">
                <div className="relative min-h-0 w-screen flex-1">
                  <Image
                    src={asset.src}
                    alt={asset.alt}
                    fill
                    loading="eager"
                    unoptimized
                    sizes="98vw"
                    className="object-contain drop-shadow-2xl"
                  />
                </div>
                <p className="text-center text-sm font-semibold uppercase tracking-wide text-slate-300">
                  {label}
                </p>
              </div>
            </div>,
            document.body,
          )
        : null}
    </>
  );
}

function PaperTitleGraphic() {
  return (
    <div
      data-testid="presentation-paper-title"
      className="rounded-lg border border-white/10 bg-white/5 p-6 shadow-2xl"
    >
      <div className="rounded-lg bg-slate-950 p-6 text-white">
        <p className="text-sm font-semibold uppercase tracking-wide text-cyan-300">
          paper club presentation
        </p>
        <h1 className="mt-8 text-5xl font-black leading-tight md:text-6xl">
          Are We Ready For An Agent-Native Memory System?
        </h1>
        <p className="mt-6 text-xl leading-8 text-slate-300">
          Wei Zhou, Xuanhe Zhou, Shaokun Han, Hongming Xu, Guoliang Li, Zhiyu
          Li, Feiyu Xiong, Fan Wu
        </p>
      </div>
      <div className="mt-4 grid gap-3 md:grid-cols-3">
        {[
          ["arXiv", "2606.24775v1"],
          ["date", "June 23, 2026"],
          ["talk lens", "memory-system evaluation"],
        ].map(([label, value]) => (
          <div
            key={label}
            className="rounded-lg bg-cyan-300 p-4 text-slate-950"
          >
            <p className="text-xs font-semibold uppercase tracking-wide">
              {label}
            </p>
            <p className="mt-2 text-2xl font-black">{value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function AgendaGraphic() {
  return (
    <div
      data-testid="presentation-agenda"
      className="rounded-lg border border-slate-200 bg-white p-5 shadow-xl"
    >
      <div className="grid gap-3">
        {agendaItems.map(([number, title, body]) => (
          <div
            key={number}
            className="grid grid-cols-[4rem_1fr] items-center gap-4 rounded-lg border border-slate-200 bg-slate-50 p-4"
          >
            <div className="flex h-14 w-14 items-center justify-center rounded-lg bg-slate-950 text-2xl font-black text-cyan-300">
              {number}
            </div>
            <div>
              <p className="text-2xl font-black text-slate-950">{title}</p>
              <p className="mt-1 text-sm font-semibold uppercase tracking-wide text-slate-500">
                {body}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SectionDividerGraphic({
  section,
}: {
  section: (typeof sectionDividers)[keyof typeof sectionDividers];
}) {
  const sectionLabel = section.number === "A" ? "appendix" : "paper section";

  return (
    <div className="rounded-lg border border-white/10 bg-slate-950 p-6 shadow-2xl">
      <div className="grid gap-6 md:grid-cols-[0.45fr_1fr] md:items-center">
        <div className="flex aspect-square items-center justify-center rounded-lg bg-cyan-300 text-8xl font-black text-slate-950">
          {section.number}
        </div>
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-cyan-300">
            {sectionLabel}
          </p>
          <h2
            aria-label={`${section.number}. ${section.title}`}
            className="mt-4 text-5xl font-black leading-tight text-white"
          >
            {section.title}
          </h2>
          <p className="mt-5 text-xl leading-8 text-slate-300">
            {section.thesis}
          </p>
        </div>
      </div>
      <div className="mt-6 flex flex-wrap gap-3">
        {section.beats.map((beat) => (
          <span
            key={beat}
            className="rounded-full border border-cyan-300/30 bg-cyan-300/10 px-4 py-2 text-sm font-semibold uppercase tracking-wide text-cyan-100"
          >
            {beat}
          </span>
        ))}
      </div>
    </div>
  );
}

function FigureOneArchitectureGraphic() {
  return (
    <div
      data-testid="presentation-figure-one-architectures"
      className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4"
    >
      {figureOneArchitectures.map((architecture, architectureIndex) => (
        <div
          key={architecture.name}
          className="relative min-w-0 overflow-hidden rounded-lg border border-cyan-300/20 bg-gradient-to-br from-slate-950 to-slate-900 p-5 shadow-xl shadow-slate-950/20"
        >
          <div className="absolute right-4 top-4 flex h-8 w-8 items-center justify-center rounded-full bg-cyan-300/15 text-xs font-black text-cyan-200">
            {architectureIndex + 1}
          </div>
          <p className="pr-10 text-2xl font-black leading-tight text-white">
            {architecture.name}
          </p>
          <div className="mt-7 grid gap-3">
            {architecture.visual.map((node, index) => (
              <div
                key={node}
                className="grid min-w-0 grid-cols-[2.25rem_minmax(0,1fr)] items-center gap-3"
              >
                <span
                  className={`flex h-9 w-9 items-center justify-center rounded-full text-sm font-black text-slate-950 ${
                    index === 1 ? "bg-emerald-300" : "bg-cyan-300"
                  }`}
                >
                  {index + 1}
                </span>
                <span className="min-w-0 rounded-md border border-white/10 bg-white/10 px-3 py-2 text-sm font-semibold leading-snug text-slate-100">
                  {node}
                </span>
              </div>
            ))}
          </div>
          <p className="mt-6 border-t border-white/10 pt-4 text-sm leading-6 text-slate-300">
            {architecture.point}
          </p>
        </div>
      ))}
    </div>
  );
}

function MotivationSystemEvaluationGraphic() {
  return (
    <div
      data-testid="presentation-motivation-system-evaluation"
      className="rounded-lg border border-slate-200 bg-white p-5 shadow-xl"
    >
      <div className="grid gap-5 lg:grid-cols-[0.9fr_auto_1.1fr] lg:items-stretch">
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-6 text-slate-950">
          <p className="text-sm font-semibold uppercase tracking-wide text-blue-700">
            before
          </p>
          <p className="mt-10 text-5xl font-black leading-tight">
            Did the agent answer correctly?
          </p>
          <p className="mt-8 text-lg leading-8 text-slate-700">
            Most evaluations looked from the outside: report a task score, then
            treat the memory system as one black box.
          </p>
          <div className="mt-8 rounded-lg border border-slate-200 bg-white p-4">
            <p className="text-sm font-semibold uppercase tracking-wide text-slate-600">
              final accuracy can improve while the cause stays hidden
            </p>
          </div>
        </div>
        <div className="hidden items-center justify-center lg:flex">
          <div className="flex h-20 w-20 items-center justify-center rounded-full bg-cyan-300 text-slate-950 shadow-lg">
            <ArrowRight aria-hidden="true" size={36} strokeWidth={3} />
          </div>
        </div>
        <div className="rounded-lg bg-slate-950 p-6 text-white">
          <p className="text-sm font-semibold uppercase tracking-wide text-cyan-300">
            after
          </p>
          <p className="mt-10 text-4xl font-black leading-tight">
            Evaluate the memory layer as a system.
          </p>
          <div className="mt-8 grid gap-3 sm:grid-cols-2">
            {[
              "Retrieval quality",
              "Update correctness",
              "Long-horizon stability",
              "Operational cost",
            ].map((label) => (
              <div
                key={label}
                className="rounded-lg border border-white/10 bg-white/10 p-4"
              >
                <p className="text-lg font-black text-white">{label}</p>
              </div>
            ))}
          </div>
          <p className="mt-6 rounded-lg bg-cyan-300 p-4 text-lg font-black leading-7 text-slate-950">
            Benchmarks should explain what failed, not only whether the final
            answer changed.
          </p>
        </div>
      </div>
    </div>
  );
}

function BenchmarkFailureModesGraphic() {
  return (
    <div
      data-testid="presentation-benchmark-failure-modes"
      className="rounded-lg border border-white/10 bg-slate-950 p-5 shadow-2xl"
    >
      <div className="grid gap-3">
        {benchmarkLimitations.map(([limitation, why], index) => (
          <div
            key={limitation}
            className="grid grid-cols-[3rem_1fr] gap-4 rounded-lg border border-white/10 bg-white/5 p-4"
          >
            <div className="flex h-11 w-11 items-center justify-center rounded-full bg-cyan-300 text-lg font-black text-slate-950">
              {index + 1}
            </div>
            <div>
              <p className="text-xl font-black text-white">{limitation}</p>
              <p className="mt-1 text-sm leading-6 text-slate-400">{why}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ScopeContrastGraphic() {
  return (
    <div
      data-testid="presentation-scope-contrasts"
      className="rounded-lg border border-slate-200 bg-white p-5 shadow-xl"
    >
      <div className="grid gap-3">
        {scopeContrasts.map((contrast) => (
          <div
            key={contrast.label}
            className="grid gap-3 rounded-lg border border-slate-200 bg-slate-50 p-4 md:grid-cols-[0.34fr_minmax(0,1fr)] md:items-stretch"
          >
            <p className="flex min-h-28 items-center text-3xl font-black leading-tight text-slate-950">
              {contrast.label}
            </p>
            <div className="flex min-h-28 flex-col justify-center rounded-lg bg-white p-5 shadow-sm">
              <p className="text-xl font-semibold leading-8 text-slate-800">
                {contrast.old}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ModuleSystemGraphic() {
  return (
    <div
      data-testid="presentation-module-map"
      className="rounded-lg border border-slate-200 bg-white p-5 shadow-xl"
    >
      <div className="rounded-lg border border-slate-200 bg-slate-50 p-5 text-slate-950">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-blue-700">
            Formal lens
          </p>
          <h3 className="mt-2 text-3xl font-black md:text-4xl">
            memory as a lifecycle
          </h3>
          <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-700">
            The memory object is persistent state outside learned weights and
            outside the current prompt; the system is evaluated through four
            lifecycle modules.
          </p>
        </div>
      </div>
      <div className="mt-4 grid gap-3 md:grid-cols-4">
        {moduleCards.map((module) => {
          const Icon = module.icon;
          return (
            <div
              key={module.letter}
              className="rounded-lg bg-slate-950 p-4 text-white"
            >
              <div className="flex items-center gap-4">
                <div
                  className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br ${module.tone} text-slate-950`}
                >
                  <Icon aria-hidden="true" size={24} />
                </div>
                <p className="text-4xl font-black tracking-normal">
                  {module.letter}
                </p>
              </div>
              <h3 className="mt-5 text-lg font-semibold">{module.title}</h3>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                {module.caption}
              </p>
            </div>
          );
        })}
      </div>
      <p className="mt-4 rounded-lg border border-blue-100 bg-blue-50 p-4 text-center text-base font-semibold text-slate-950">
        The same four modules let the paper compare systems by mechanism, not
        just final answer score.
      </p>
    </div>
  );
}

function MethodOverviewGraphic() {
  return (
    <div
      data-testid="presentation-method-overview"
      className="rounded-lg border border-white/10 bg-slate-950 p-5 shadow-2xl"
    >
      <div className="grid gap-3 md:grid-cols-2">
        {methodModuleQuestions.map(([letter, title, body]) => (
          <div
            key={letter}
            className="grid grid-cols-[3.25rem_1fr] gap-4 rounded-lg border border-white/10 bg-white/5 p-5"
          >
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-cyan-300 text-lg font-black text-slate-950">
              {letter}
            </div>
            <div>
              <p className="text-2xl font-black text-white">{title}</p>
              <p className="mt-2 text-base leading-7 text-slate-300">{body}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TableOneTaxonomyGraphic() {
  return (
    <div
      data-testid="presentation-table-one-taxonomy"
      className="rounded-lg border border-slate-200 bg-white p-4 shadow-xl"
    >
      <PaperFigure
        asset={paperAssets.table1}
        label="Paper Table 1: systems normalized into module choices"
        caption={null}
        variant="large"
      />
    </div>
  );
}

function TableOneInfographic({
  testId = "presentation-table-one-infographic",
}: {
  testId?: string;
}) {
  return (
    <div data-testid={testId} className="mx-auto w-full max-w-7xl">
      <PaperFigure
        asset={paperAssets.table1Infographic}
        label="Presenter Table 1: memory systems by module choice"
        caption={null}
        variant="full"
      />
    </div>
  );
}

function LargeFigureSlideGraphic({
  asset,
  label,
  testId,
}: {
  asset: (typeof paperAssets)[keyof typeof paperAssets];
  label: string;
  testId: string;
}) {
  return (
    <div data-testid={testId} className="mx-auto w-full max-w-7xl">
      <PaperFigure asset={asset} label={label} variant="full" />
    </div>
  );
}

function LargeFigureWithCallouts({
  asset,
  label,
  testId,
  callouts,
  tone = "blue",
}: {
  asset: (typeof paperAssets)[keyof typeof paperAssets];
  label: string;
  testId: string;
  callouts?: readonly string[];
  tone?: "blue" | "green";
}) {
  const visibleCallouts = callouts ?? [];
  const calloutGrid =
    visibleCallouts.length > 3 ? "md:grid-cols-5" : "md:grid-cols-3";

  return (
    <div
      data-testid={testId}
      className="rounded-lg border border-slate-200 bg-white p-4 shadow-xl"
    >
      <PaperFigure asset={asset} label={label} variant="large" />
      {visibleCallouts.length > 0 ? (
        <div className={`mt-3 grid gap-2 ${calloutGrid}`}>
          {visibleCallouts.map((callout) => (
            <p
              key={callout}
              className={`rounded-lg p-2 text-xs font-semibold uppercase tracking-wide ${
                tone === "green"
                  ? "bg-emerald-50 text-emerald-800"
                  : "bg-blue-50 text-blue-800"
              }`}
            >
              {callout}
            </p>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function ArchitecturePrimerGraphic() {
  return (
    <div
      data-testid="presentation-architecture-primer"
      className="rounded-lg border border-slate-200 bg-white p-5 shadow-xl"
    >
      <div className="grid gap-3 md:grid-cols-2">
        {architecturePrimer.map((architecture, index) => (
          <div
            key={architecture.family}
            className="rounded-lg border border-slate-200 bg-slate-50 p-4"
          >
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-full bg-slate-950 text-lg font-black text-cyan-300">
                {index + 1}
              </div>
              <p className="text-2xl font-black text-slate-950">
                {architecture.family}
              </p>
            </div>
            <div className="mt-5 grid gap-3">
              {[
                ["stores", architecture.stores],
                ["answers", architecture.answers],
                ["watch", architecture.risk],
              ].map(([label, body]) => (
                <div
                  key={label}
                  className="grid grid-cols-[4.8rem_1fr] gap-3 rounded-lg bg-white p-3"
                >
                  <p className="text-xs font-semibold uppercase tracking-wide text-blue-700">
                    {label}
                  </p>
                  <p className="text-sm leading-6 text-slate-700">{body}</p>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ArchitectureExamplesGraphic() {
  return (
    <div
      data-testid="presentation-architecture-examples"
      className="rounded-lg border border-white/10 bg-slate-950 p-5 shadow-2xl"
    >
      <div className="grid gap-3">
        {architectureExamples.map(([family, examples, meaning], index) => (
          <div
            key={`${family}-${examples}`}
            className="grid gap-4 rounded-lg border border-white/10 bg-white/5 p-4 md:grid-cols-[0.34fr_0.36fr_1fr] md:items-start"
          >
            <div className="flex items-center gap-3">
              <span className="flex h-9 w-9 items-center justify-center rounded-full bg-cyan-300 text-sm font-black text-slate-950">
                {String(index + 1).padStart(2, "0")}
              </span>
              <p className="text-sm font-semibold uppercase tracking-wide text-cyan-300">
                {family}
              </p>
            </div>
            <div className="grid gap-1 text-xl font-black leading-snug text-white">
              {examples.split(",").map((example) => (
                <p key={example.trim()}>{example.trim()}</p>
              ))}
            </div>
            <p className="text-sm leading-6 text-slate-300">{meaning}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function BenchmarkExplainerGraphic() {
  return (
    <div
      data-testid="presentation-benchmark-explainer"
      className="rounded-lg border border-slate-200 bg-white p-5 shadow-xl"
    >
      <div className="grid gap-4 md:grid-cols-2">
        {benchmarkExplainers.map((benchmark) => (
          <div
            key={benchmark.name}
            className="rounded-lg border border-slate-200 bg-slate-50 p-5"
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-2xl font-black text-slate-950">
                  {benchmark.name}
                </p>
                <p className="mt-1 text-xs font-semibold uppercase tracking-wide text-blue-700">
                  {benchmark.source}
                </p>
              </div>
              <Gauge aria-hidden="true" className="text-blue-600" size={26} />
            </div>
            <div className="mt-4 grid gap-3">
              {[
                ["tests", benchmark.measures],
                ["scale", benchmark.scale],
                ["reported", benchmark.metrics],
              ].map(([label, value]) => (
                <div
                  key={label}
                  className="grid grid-cols-[5.75rem_1fr] gap-3 rounded-lg bg-white p-3"
                >
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    {label}
                  </p>
                  <p className="text-sm font-semibold leading-6 text-slate-800">
                    {value}
                  </p>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function EvaluationGraphic() {
  return (
    <div
      data-testid="presentation-evaluation-landscape"
      className="rounded-lg border border-slate-200 bg-white p-5 shadow-xl"
    >
      <div className="grid gap-4 md:grid-cols-4">
        {sectionFourScaleCards.map(([value, label]) => (
          <div key={label} className="rounded-lg bg-slate-950 p-4 text-white">
            <p className="text-5xl font-black text-cyan-300">{value}</p>
            <p className="mt-3 text-sm font-semibold uppercase tracking-wide text-slate-300">
              {label}
            </p>
          </div>
        ))}
      </div>
      <div className="mt-5 grid gap-3">
        {sectionFourQuestions.map(([rq, lens, question]) => (
          <div
            key={rq}
            className="grid grid-cols-[4rem_1fr] gap-4 rounded-lg border border-slate-200 bg-slate-50 p-4"
          >
            <div className="flex h-14 w-14 items-center justify-center rounded-lg bg-slate-950 text-lg font-black text-white">
              {rq}
            </div>
            <div>
              <p className="text-sm font-semibold uppercase tracking-wide text-blue-700">
                {lens}
              </p>
              <p className="mt-1 text-2xl font-semibold leading-snug text-slate-950">
                {question}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SystemLineupGraphic() {
  const familyRows = [
    ["Reference baselines", "Long Context, Embedding RAG"],
    ["Sequential context", "MemoChat, Mem0, MemAgent"],
    ["Structural / topological", "MemTree, Zep, Cognee"],
    [
      "Multi-paradigm hybrid",
      "Letta, LightMem, SimpleMem, MemOS, MemoryOS, A-MEM",
    ],
  ] as const;

  return (
    <div
      data-testid="presentation-system-lineup"
      className="rounded-lg border border-white/10 bg-slate-950 p-5 shadow-2xl"
    >
      <div className="grid gap-3">
        {familyRows.map(([family, systems], index) => (
          <div
            key={family}
            className="grid gap-4 rounded-lg border border-white/10 bg-white/5 p-4 md:grid-cols-[0.36fr_1fr_0.38fr]"
          >
            <div className="flex items-center gap-3">
              <span className="flex h-10 w-10 items-center justify-center rounded-full bg-cyan-300 text-sm font-black text-slate-950">
                {String(index + 1).padStart(2, "0")}
              </span>
              <p className="text-sm font-semibold uppercase tracking-wide text-cyan-300">
                {family}
              </p>
            </div>
            <p className="text-xl font-black leading-snug text-white">
              {systems}
            </p>
            <p className="rounded-lg bg-slate-900 p-3 text-sm font-semibold leading-6 text-slate-300">
              {index === 0
                ? "baseline pressure"
                : index === 1
                  ? "sequence pressure"
                  : index === 2
                    ? "relation pressure"
                    : "coordination pressure"}
            </p>
          </div>
        ))}
      </div>
      <div className="mt-5 grid gap-3 md:grid-cols-3">
        {[
          [
            "evidence",
            "Can the system retrieve the support set, not just a nearby chunk?",
          ],
          ["state", "Can it update current facts without erasing history?"],
          [
            "operations",
            "Can it do this without global re-indexing or high latency?",
          ],
        ].map(([label, body]) => (
          <div
            key={label}
            className="rounded-lg bg-cyan-300 p-4 text-slate-950"
          >
            <p className="text-sm font-semibold uppercase tracking-wide">
              {label}
            </p>
            <p className="mt-2 text-lg font-black leading-snug">{body}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function FindingPoint({
  text,
  accentClassName = "text-blue-950",
}: {
  text: string;
  accentClassName?: string;
}) {
  const match = /^(Finding \d:)(.*)$/.exec(text);

  if (!match) {
    return <>{text}</>;
  }

  return (
    <>
      <span className={`font-black ${accentClassName}`}>{match[1]}</span>
      {match[2]}
    </>
  );
}

function RqFindingGraphic({ index }: { index: number }) {
  const finding = rqFindings[index];
  return (
    <div
      data-testid={finding.testId}
      className="rounded-lg border border-slate-200 bg-white p-5 shadow-xl"
    >
      <div className="grid gap-5 lg:grid-cols-[0.86fr_1.14fr]">
        <div className="rounded-lg bg-slate-950 p-5 text-white">
          <p className="text-sm font-semibold uppercase tracking-wide text-cyan-300">
            {finding.tested}
          </p>
          <p className="mt-8 text-7xl font-black">{finding.label}</p>
          <h2
            aria-label={`${finding.label}: ${finding.title}`}
            className="mt-5 text-5xl font-black leading-tight"
          >
            {finding.title}
          </h2>
        </div>
        <div className="grid content-start gap-3">
          {finding.beats.map((beat) => (
            <div
              key={beat}
              className="grid grid-cols-[1.1rem_1fr] gap-4 rounded-lg bg-blue-50 p-5"
            >
              <span
                aria-hidden="true"
                className="mt-2 h-3 w-3 rounded-full bg-blue-600"
              />
              <p className="text-2xl font-semibold leading-tight text-slate-950">
                <FindingPoint text={beat} />
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function SectionFiveOverviewGraphic() {
  return (
    <div
      data-testid="presentation-section-five-overview"
      className="rounded-lg border border-violet-200 bg-white p-5 shadow-xl"
    >
      <div className="rounded-lg bg-slate-950 p-5 text-white">
        <p className="text-sm font-semibold uppercase tracking-wide text-violet-300">
          from ranking to diagnosis
        </p>
        <div className="mt-5 grid gap-3 lg:grid-cols-4">
          {sectionFiveFrame.map(([letter, module, question]) => (
            <div
              key={letter}
              className="h-full rounded-lg border border-white/10 bg-white/5 p-4"
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-violet-400 text-2xl font-black text-slate-950">
                {letter}
              </div>
              <p className="mt-5 text-sm font-semibold uppercase tracking-wide text-violet-200">
                {module}
              </p>
              <p className="mt-2 text-lg font-semibold leading-snug text-white">
                {question}
              </p>
            </div>
          ))}
        </div>
      </div>
      <div className="mt-4 rounded-lg bg-violet-50 p-4">
        <p className="text-2xl font-semibold leading-tight text-slate-950">
          The point is not which system won. It is which component choice caused
          the win or failure.
        </p>
      </div>
    </div>
  );
}

function ComponentLessonGraphic({ index }: { index: number }) {
  const lesson = componentLessons[index];
  return (
    <div
      data-testid={lesson.testId}
      className="rounded-lg border border-violet-200 bg-white p-5 shadow-xl"
    >
      <div className="grid gap-5 lg:grid-cols-[0.88fr_1.12fr]">
        <div className="rounded-lg bg-slate-950 p-5 text-white">
          <p className="text-sm font-semibold uppercase tracking-wide text-violet-300">
            {lesson.module}
          </p>
          <p className="mt-8 text-7xl font-black">{lesson.label}</p>
          <h2
            aria-label={`${lesson.label}: ${lesson.title}`}
            className="mt-5 text-5xl font-black leading-tight"
          >
            {lesson.title}
          </h2>
        </div>
        <div className="grid content-start gap-3">
          {lesson.beats.map((beat) => (
            <div
              key={beat}
              className="grid grid-cols-[1.1rem_1fr] gap-4 rounded-lg bg-violet-50 p-5"
            >
              <span
                aria-hidden="true"
                className="mt-2 h-3 w-3 rounded-full bg-violet-600"
              />
              <p className="text-xl font-semibold leading-tight text-slate-950">
                <FindingPoint text={beat} accentClassName="text-violet-950" />
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ReadinessChecklist() {
  return (
    <div className="rounded-lg border border-white/10 bg-white/5 p-6 shadow-2xl">
      <div className="grid gap-5 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="rounded-lg bg-slate-950 p-5 text-white">
          <p className="text-sm font-semibold uppercase tracking-wide text-cyan-300">
            conclusion
          </p>
          <p className="mt-8 text-5xl font-black leading-tight">
            What we learned
          </p>
          <p className="mt-6 text-base leading-7 text-slate-300">
            The architectures are promising, but behavior is workload-dependent,
            component-sensitive, and cost-sensitive.
          </p>
        </div>
        <div className="grid gap-3">
          {conclusionContributions.map(([Icon, title, body]) => {
            const TypedIcon = Icon as typeof Database;
            return (
              <div
                key={title as string}
                className="grid grid-cols-[3rem_1fr] items-center gap-3 rounded-lg border border-white/10 bg-white/10 p-3"
              >
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-cyan-300 text-slate-950">
                  <TypedIcon aria-hidden="true" size={24} />
                </div>
                <div>
                  <p className="text-lg font-semibold text-white">
                    {title as string}
                  </p>
                  <p className="mt-1 text-xs font-semibold uppercase tracking-wide text-slate-400">
                    {body as string}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <div className="mt-5 rounded-lg bg-cyan-300 p-4 text-slate-950">
        <p className="text-2xl font-black leading-tight">
          Preserve evidence -&gt; retrieve reliably -&gt; update correctly -&gt;
          bound cost.
        </p>
      </div>
      <div className="agent-native-presentation-no-print mt-5 flex flex-wrap justify-end gap-3">
        <AgentNativeMemoryPdfExportButton />
      </div>
    </div>
  );
}

function PresenterNav() {
  return (
    <nav
      aria-label="Presentation sections"
      className="agent-native-presentation-no-print group fixed right-0 top-1/2 z-30 hidden h-[82vh] w-16 -translate-y-1/2 items-center justify-end lg:flex"
    >
      <div
        data-testid="presentation-slide-rail"
        className="pointer-events-none mr-4 grid gap-2 rounded-full border border-white/10 bg-slate-950/70 p-2 opacity-0 shadow-xl shadow-slate-950/20 backdrop-blur transition-opacity duration-200 group-hover:pointer-events-auto group-hover:opacity-100 group-focus-within:pointer-events-auto group-focus-within:opacity-100"
      >
        {slideNav.map(([label, description], index) => (
          <a
            key={`${index}-${label}`}
            href={`#slide-${index + 1}`}
            aria-label={`Go to ${label}: ${description}`}
            className="group/dot relative flex h-3 w-3 rounded-full bg-slate-500 transition-colors hover:bg-cyan-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300"
          >
            <span className="absolute right-5 top-1/2 hidden -translate-y-1/2 whitespace-nowrap rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-950 group-hover/dot:block group-focus-visible/dot:block">
              {label}
            </span>
          </a>
        ))}
      </div>
    </nav>
  );
}

function PresentationPrintStyles() {
  return (
    <style>{`
      @media print {
        @page {
          size: 16in 9in;
          margin: 0;
        }

        html,
        body:has(.agent-native-presentation-deck) {
          background: #07111f !important;
          margin: 0 !important;
          overflow: visible !important;
          -webkit-print-color-adjust: exact !important;
          print-color-adjust: exact !important;
        }

        body:has(.agent-native-presentation-deck) > header,
        .agent-native-presentation-no-print {
          display: none !important;
        }

        .agent-native-presentation-deck,
        .agent-native-presentation-deck * {
          -webkit-print-color-adjust: exact !important;
          print-color-adjust: exact !important;
        }

        .agent-native-presentation-deck {
          background: #07111f !important;
          display: block !important;
          scroll-behavior: auto !important;
          scroll-snap-type: none !important;
          width: 100% !important;
        }

        .agent-native-presentation-slide {
          align-items: center !important;
          break-after: page !important;
          box-sizing: border-box !important;
          display: flex !important;
          height: 100vh !important;
          max-height: 100vh !important;
          min-height: 100vh !important;
          overflow: hidden !important;
          padding: 0.42in 0.55in !important;
          page-break-after: always !important;
          page-break-inside: avoid !important;
          width: 100vw !important;
        }

        .agent-native-presentation-slide:last-of-type {
          break-after: auto !important;
          page-break-after: auto !important;
        }

        .agent-native-presentation-slide-grid {
          align-items: center !important;
          display: grid !important;
          gap: 0.34in !important;
          grid-template-columns: minmax(0, 0.76fr) minmax(0, 1.24fr) !important;
          height: 100% !important;
          max-width: none !important;
          min-height: 0 !important;
          width: 100% !important;
        }

        .agent-native-presentation-slide-wide {
          display: flex !important;
          flex-direction: column !important;
          gap: 0.16in !important;
          height: 100% !important;
          justify-content: flex-start !important;
          max-width: none !important;
          min-height: 0 !important;
          width: 100% !important;
        }

        .agent-native-presentation-slide-copy,
        .agent-native-presentation-slide-visual {
          max-width: none !important;
          min-width: 0 !important;
        }

        .agent-native-presentation-slide-eyebrow {
          font-size: 0.12in !important;
          line-height: 1.1 !important;
        }

        .agent-native-presentation-slide-title {
          font-size: 0.48in !important;
          line-height: 1.07 !important;
          margin-top: 0.14in !important;
        }

        .agent-native-presentation-slide-subtitle {
          font-size: 0.17in !important;
          line-height: 1.42 !important;
          margin-top: 0.16in !important;
          max-width: 4.7in !important;
        }

        .agent-native-presentation-slide-wide
          .agent-native-presentation-slide-title {
          font-size: 0.34in !important;
          margin-top: 0.06in !important;
        }

        .agent-native-presentation-slide-wide
          .agent-native-presentation-slide-subtitle {
          font-size: 0.13in !important;
          line-height: 1.32 !important;
          margin-top: 0.06in !important;
          max-width: 8.6in !important;
        }

        .agent-native-memory-boundary-graphic .agent-native-generated-visual {
          height: 4.55in !important;
          min-height: 4.55in !important;
        }

        .agent-native-memory-boundary-graphic
          .agent-native-generated-visual
          img {
          height: 100% !important;
          min-height: 0 !important;
          object-fit: cover !important;
        }

        .agent-native-memory-boundary-cards {
          display: grid !important;
          gap: 0.1in !important;
          grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
          margin-top: 0.12in !important;
        }

        .agent-native-memory-boundary-card {
          background-color: rgba(2, 6, 23, 0.9) !important;
          border-color: rgba(103, 232, 249, 0.45) !important;
        }
      }
    `}</style>
  );
}

function isInteractiveKeyboardTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) {
    return false;
  }

  return Boolean(
    target.closest(
      'a, button, input, textarea, select, summary, [contenteditable="true"], [role="button"]',
    ),
  );
}

export default function AgentNativeMemoryPresentationPage() {
  useEffect(() => {
    const nextKeys = new Set([
      " ",
      "Spacebar",
      "ArrowDown",
      "ArrowRight",
      "PageDown",
      "n",
      "N",
    ]);
    const previousKeys = new Set(["ArrowUp", "ArrowLeft", "PageUp", "p", "P"]);

    const getSlideElements = () =>
      Array.from(
        document.querySelectorAll<HTMLElement>(
          ".agent-native-presentation-slide",
        ),
      );

    const getCurrentSlideIndex = (slides: readonly HTMLElement[]) => {
      const anchorY = 72;
      return slides.reduce(
        (best, slide, index) => {
          const distance = Math.abs(
            slide.getBoundingClientRect().top - anchorY,
          );
          return distance < best.distance ? { index, distance } : best;
        },
        { index: 0, distance: Number.POSITIVE_INFINITY },
      ).index;
    };

    const animateTargetSlide = (slide: HTMLElement) => {
      const content =
        slide.querySelector<HTMLElement>(
          ".agent-native-presentation-slide-grid, .agent-native-presentation-slide-wide",
        ) ?? slide;

      content.animate(
        [
          { opacity: 0.92, transform: "translateY(10px)" },
          { opacity: 1, transform: "translateY(0)" },
        ],
        {
          duration: 240,
          easing: "cubic-bezier(0.2, 0, 0, 1)",
        },
      );
    };

    const scrollToSlide = (index: number) => {
      const slides = getSlideElements();
      const target = slides[Math.max(0, Math.min(index, slides.length - 1))];
      if (!target) {
        return;
      }

      target.scrollIntoView({ behavior: "smooth", block: "start" });
      window.setTimeout(() => animateTargetSlide(target), 120);
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (
        event.defaultPrevented ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        isInteractiveKeyboardTarget(event.target) ||
        document.querySelector('[data-testid="presentation-figure-lightbox"]')
      ) {
        return;
      }

      const slides = getSlideElements();
      if (slides.length === 0) {
        return;
      }

      const currentIndex = getCurrentSlideIndex(slides);

      if (event.shiftKey && (event.key === " " || event.key === "Spacebar")) {
        event.preventDefault();
        scrollToSlide(currentIndex - 1);
        return;
      }

      if (nextKeys.has(event.key)) {
        event.preventDefault();
        scrollToSlide(currentIndex + 1);
        return;
      }

      if (previousKeys.has(event.key)) {
        event.preventDefault();
        scrollToSlide(currentIndex - 1);
        return;
      }

      if (event.key === "Home") {
        event.preventDefault();
        scrollToSlide(0);
        return;
      }

      if (event.key === "End") {
        event.preventDefault();
        scrollToSlide(slides.length - 1);
      }
    };

    document.addEventListener("keydown", onKeyDown, true);

    return () => {
      document.removeEventListener("keydown", onKeyDown, true);
    };
  }, []);

  return (
    <main className="agent-native-presentation-deck snap-y snap-mandatory scroll-smooth bg-slate-950">
      <PresentationPrintStyles />
      <PresenterNav />

      <Slide
        id="slide-1"
        index="01"
        eyebrow="Paper"
        title="Are We Ready For An Agent-Native Memory System?"
        subtitle="A paper-club walkthrough of agent memory as a system: architecture, retrieval, updates, long horizons, cost, and component failures."
        layout="visualOnly"
        hideHiddenTitle
      >
        <PaperTitleGraphic />
      </Slide>

      <Slide
        id="slide-2"
        index="02"
        eyebrow="Agenda"
        title="How the talk will move through the paper"
        subtitle="We start with the problem the paper targets, then follow the paper structure from definitions to taxonomy, evaluation, ablations, and conclusion."
        tone="light"
      >
        <AgendaGraphic />
      </Slide>

      <Slide
        id="slide-3"
        index="03"
        eyebrow="Section 1"
        title="1. Introduction"
        subtitle="The paper begins by arguing that agent memory is important infrastructure, but the field is evaluating it with the wrong lens."
        layout="visualOnly"
        hideHiddenTitle
      >
        <SectionDividerGraphic section={sectionDividers.introduction} />
      </Slide>

      <Slide
        id="slide-4"
        index="04"
        eyebrow="Motivation"
        title="Why agent memory needs a systems evaluation"
        tone="light"
        layout="wide"
      >
        <MotivationSystemEvaluationGraphic />
      </Slide>

      <Slide
        id="slide-5"
        index="05"
        eyebrow="Evaluation gap"
        title="Memory is modular, but benchmarks are not"
        subtitle="A memory system can look good on end-task accuracy while still failing as infrastructure."
      >
        <BenchmarkFailureModesGraphic />
      </Slide>

      <Slide
        id="slide-6"
        index="06"
        eyebrow="Section 2"
        title="2. Preliminaries"
        subtitle="The section is mostly scope-setting: it defines memory as persistent state, then separates it from nearby concepts."
        layout="visualOnly"
        hideHiddenTitle
      >
        <SectionDividerGraphic section={sectionDividers.preliminaries} />
      </Slide>

      <Slide
        id="slide-7"
        index="07"
        eyebrow="Four-module lens"
        title="The formal anchor is four modules"
        subtitle="The paper defines memory as persistent state, then decomposes the memory system into representation, extraction, retrieval, and maintenance."
        tone="light"
        layout="wide"
      >
        <ModuleSystemGraphic />
      </Slide>

      <Slide
        id="slide-8"
        index="08"
        eyebrow="Scope boundaries"
        title="Agent memory is broader than RAG or context engineering"
        subtitle="It also differs from ordinary database workloads because access is semantic, observations are uncertain, and workloads are heterogeneous."
        tone="blue"
      >
        <ScopeContrastGraphic />
      </Slide>

      <Slide
        id="slide-9"
        index="09"
        eyebrow="Section 3"
        title="3. Method Overview"
        subtitle="This is the paper's taxonomy of agent memory systems, not a new memory algorithm."
        layout="visualOnly"
        hideHiddenTitle
      >
        <SectionDividerGraphic section={sectionDividers.method} />
      </Slide>

      <Slide
        id="slide-10"
        index="10"
        eyebrow="Method overview"
        title="Design-space map"
        subtitle="The paper turns the four-module lens into concrete mechanism choices used by real memory systems."
      >
        <MethodOverviewGraphic />
      </Slide>

      <Slide
        id="slide-11"
        index="11"
        eyebrow="Table 1"
        title="Systems become combinations of module choices"
        tone="light"
        layout="wide"
      >
        <TableOneTaxonomyGraphic />
      </Slide>

      <Slide
        id="slide-12"
        index="12"
        eyebrow="Table 1 infographic"
        title="Table 1 compresses the field into four module choices"
        subtitle="This simplified view is the presenter version of Table 1: each system bucket is a combination of representation, extraction, retrieval, and maintenance choices."
        layout="visualOnly"
      >
        <TableOneInfographic />
      </Slide>

      <Slide
        id="slide-13"
        index="13"
        eyebrow="Figure 2"
        title="Memory Representation"
        tone="blue"
        layout="wide"
      >
        <LargeFigureWithCallouts
          asset={paperAssets.figure2}
          label="Figure 2: memory representation methods"
          testId="presentation-figure-2-large"
        />
      </Slide>

      <Slide
        id="slide-14"
        index="14"
        eyebrow="Figure 3"
        title="Memory Storage"
        tone="blue"
        layout="wide"
      >
        <LargeFigureWithCallouts
          asset={paperAssets.figure3}
          label="Figure 3: memory storage methods"
          testId="presentation-figure-3-large"
        />
      </Slide>

      <Slide
        id="slide-15"
        index="15"
        eyebrow="Extraction"
        title="Extraction decides what can ever be recovered"
        tone="green"
        layout="wide"
      >
        <LargeFigureWithCallouts
          asset={paperAssets.figure4}
          label="Figure 4: memory extraction methods"
          testId="presentation-figure-4-large"
          tone="green"
        />
      </Slide>

      <Slide
        id="slide-16"
        index="16"
        eyebrow="Retrieval"
        title="Retrieval is routing, not just vector search"
        layout="wide"
      >
        <LargeFigureWithCallouts
          asset={paperAssets.figure5}
          label="Figure 5: memory retrieval methods"
          testId="presentation-figure-5-large"
        />
      </Slide>

      <Slide
        id="slide-17"
        index="17"
        eyebrow="Maintenance"
        title="Maintenance"
        tone="green"
        layout="wide"
      >
        <LargeFigureWithCallouts
          asset={paperAssets.figure6}
          label="Figure 6: memory maintenance methods"
          testId="presentation-figure-6-large"
          tone="green"
        />
      </Slide>

      <Slide
        id="slide-18"
        index="18"
        eyebrow="Table 1 reminder"
        title="Keep the taxonomy in view before the results"
        subtitle="Before the empirical section, this repeats the simplified Table 1 map: each evaluated system is a bundle of representation, extraction, retrieval, and maintenance choices."
        layout="visualOnly"
      >
        <TableOneInfographic testId="presentation-table-one-infographic-reminder" />
      </Slide>

      <Slide
        id="slide-19"
        index="19"
        eyebrow="Section 4"
        title="4. End-to-End Assessment"
        subtitle="Section 3 mapped mechanisms. Section 4 asks whether those design choices change behavior in practice."
        layout="visualOnly"
        hideHiddenTitle
      >
        <SectionDividerGraphic section={sectionDividers.assessment} />
      </Slide>

      <Slide
        id="slide-20"
        index="20"
        eyebrow="Benchmark primer"
        title="What are the workloads actually testing?"
        subtitle="The paper combines conversation memory, cross-session facts, procedural state, and long-context robustness because each stresses a different memory failure mode."
        tone="light"
        layout="wide"
      >
        <BenchmarkExplainerGraphic />
      </Slide>

      <Slide
        id="slide-21"
        index="21"
        eyebrow="Metrics"
        title="Metrics"
        tone="light"
        layout="wide"
      >
        <LargeFigureSlideGraphic
          asset={paperAssets.metricsOverview}
          label="Metrics overview for EM, Answer F1, ROUGE-L F1, and ROUGE-L Recall"
          testId="presentation-metrics-overview"
        />
      </Slide>

      <Slide
        id="slide-22"
        index="22"
        eyebrow="RQ1"
        title="RQ1: no universal winner"
        subtitle="The headline is workload-aligned memory: the right architecture depends on whether the task needs exact recall, semantic synthesis, or executable state."
        tone="light"
        layout="visualOnly"
        hideHiddenTitle
      >
        <RqFindingGraphic index={0} />
      </Slide>

      <Slide
        id="slide-23"
        index="23"
        eyebrow="Figure 7"
        title="The winners shift by workload and metric"
        subtitle="The paper's main effectiveness figure is easiest to present as a full-slide chart."
        tone="light"
        layout="wide"
      >
        <LargeFigureSlideGraphic
          asset={paperAssets.figure7}
          label="Figure 7: effectiveness of memory systems"
          testId="presentation-figure-7-full"
        />
      </Slide>

      <Slide
        id="slide-24"
        index="24"
        eyebrow="RQ2"
        title="RQ2: retrieval is evidence assembly"
        subtitle="The paper separates final answer quality from whether memory surfaced the whole evidence set needed to answer."
        tone="blue"
        layout="visualOnly"
        hideHiddenTitle
      >
        <RqFindingGraphic index={1} />
      </Slide>

      <Slide
        id="slide-25"
        index="25"
        eyebrow="Figure 8"
        title="Retrieval quality depends on the support set"
        subtitle="Figure 8 separates early localization from broader evidence coverage and distance drift."
        tone="blue"
        layout="wide"
      >
        <LargeFigureSlideGraphic
          asset={paperAssets.figure8}
          label="Figure 8: retrieval results over LoCoMo"
          testId="presentation-figure-8-full"
        />
      </Slide>

      <Slide
        id="slide-26"
        index="26"
        eyebrow="RQ3"
        title="RQ3: stale facts are a pipeline failure"
        subtitle="A stronger answer model helps expression after grounding, but it cannot reliably rescue memory that stores revisions as undifferentiated text."
        tone="green"
        layout="visualOnly"
        hideHiddenTitle
      >
        <RqFindingGraphic index={2} />
      </Slide>

      <Slide
        id="slide-27"
        index="27"
        eyebrow="Table 2"
        title="Update robustness is not one thing"
        subtitle="The table separates LoCoMo temporal questions, LongMemEval knowledge updates, and LongMemEval temporal reasoning."
        tone="green"
        layout="wide"
      >
        <LargeFigureSlideGraphic
          asset={paperAssets.table2}
          label="Table 2: robustness over memory update settings"
          testId="presentation-table-2-full"
        />
      </Slide>

      <Slide
        id="slide-28"
        index="28"
        eyebrow="RQ4"
        title="RQ4: long horizons reward organization"
        subtitle="As history grows, the hard part shifts from storing more to narrowing attention around the right session, entity, or relation."
        tone="blue"
        layout="visualOnly"
        hideHiddenTitle
      >
        <RqFindingGraphic index={3} />
      </Slide>

      <Slide
        id="slide-29"
        index="29"
        eyebrow="Figure 10"
        title="Long-horizon degradation has several forms"
        subtitle="Figure 10 compares context length, session history growth, and evidence-distance drift."
        tone="blue"
        layout="wide"
      >
        <LargeFigureSlideGraphic
          asset={paperAssets.figure10}
          label="Figure 10: context length, session growth, and distance drift"
          testId="presentation-figure-10-full"
        />
      </Slide>

      <Slide
        id="slide-30"
        index="30"
        eyebrow="RQ5"
        title="RQ5: cost changes the ranking"
        subtitle="The operational question is utility per latency and maintenance scope, not only answer quality."
        tone="light"
        layout="visualOnly"
        hideHiddenTitle
      >
        <RqFindingGraphic index={4} />
      </Slide>

      <Slide
        id="slide-31"
        index="31"
        eyebrow="Figure 11"
        title="Rich structure can become expensive"
        subtitle="Figure 11 is the paper's systems argument: accuracy has to be read against operation latency."
        tone="light"
        layout="wide"
      >
        <LargeFigureSlideGraphic
          asset={paperAssets.figure11}
          label="Figure 11: operation cost of memory systems"
          testId="presentation-figure-11-full"
        />
      </Slide>

      <Slide
        id="slide-32"
        index="32"
        eyebrow="Section 5"
        title="5. Fine-Grained Component Comparison"
        subtitle="Section 5 changes one memory component at a time to diagnose why whole-system results differ."
        layout="visualOnly"
        hideHiddenTitle
      >
        <SectionDividerGraphic section={sectionDividers.components} />
      </Slide>

      <Slide
        id="slide-33"
        index="33"
        eyebrow="Section 5"
        title="Now the paper asks why"
        subtitle="Section 4 ranked whole systems. Section 5 changes individual memory components one at a time to diagnose the source of each failure mode."
        tone="light"
        layout="wide"
      >
        <SectionFiveOverviewGraphic />
      </Slide>

      <Slide
        id="slide-34"
        index="34"
        eyebrow="M1 ablation"
        title="M1: abstraction is lossy"
        subtitle="The representation ablation warns that structure cannot recover evidence that compression or summarization already removed."
        tone="light"
        layout="visualOnly"
        hideHiddenTitle
      >
        <ComponentLessonGraphic index={0} />
      </Slide>

      <Slide
        id="slide-35"
        index="35"
        eyebrow="Table 3"
        title="Representation: raw evidence survives best"
        subtitle="Table 3 is the strongest warning against over-compressing memory before retrieval."
        tone="light"
        layout="wide"
      >
        <LargeFigureSlideGraphic
          asset={paperAssets.table3}
          label="Table 3: ablation of representation and storage mechanisms"
          testId="presentation-table-3-full"
        />
      </Slide>

      <Slide
        id="slide-36"
        index="36"
        eyebrow="M2 ablation"
        title="M2: write broadly, filter later"
        subtitle="Coverage-preserving extraction is safer because minor details can become important only when later evidence is combined."
        tone="green"
        layout="visualOnly"
        hideHiddenTitle
      >
        <ComponentLessonGraphic index={1} />
      </Slide>

      <Slide
        id="slide-37"
        index="37"
        eyebrow="Table 4"
        title="Extraction: early filtering is risky"
        subtitle="Table 4 compares how write-time extraction strategies trade lexical precision against downstream reasoning."
        tone="green"
        layout="wide"
      >
        <LargeFigureSlideGraphic
          asset={paperAssets.table4}
          label="Table 4: ablation of memory extraction strategies"
          testId="presentation-table-4-full"
        />
      </Slide>

      <Slide
        id="slide-38"
        index="38"
        eyebrow="M3 ablation"
        title="M3: structure beats gratuitous reasoning"
        subtitle="Balanced retrieval fusion and lightweight planning help, but extra reflection does not automatically improve routing."
        tone="blue"
        layout="visualOnly"
        hideHiddenTitle
      >
        <ComponentLessonGraphic index={2} />
      </Slide>

      <Slide
        id="slide-39"
        index="39"
        eyebrow="Table 5"
        title="Retrieval: targeted routing beats extra reflection"
        subtitle="Table 5 separates useful planning and balanced fusion from extra deliberation overhead."
        tone="blue"
        layout="wide"
      >
        <LargeFigureSlideGraphic
          asset={paperAssets.table5}
          label="Table 5: ablation of retrieval and routing mechanisms"
          testId="presentation-table-5-full"
        />
      </Slide>

      <Slide
        id="slide-40"
        index="40"
        eyebrow="M4 ablation"
        title="M4: consolidate without erasing detail"
        subtitle="Maintenance works best when it connects related facts without leaving memory fragmented or summarizing away sparse cues."
        tone="green"
        layout="visualOnly"
        hideHiddenTitle
      >
        <ComponentLessonGraphic index={3} />
      </Slide>

      <Slide
        id="slide-41"
        index="41"
        eyebrow="Figure 12"
        title="Maintenance: balanced consolidation wins cautiously"
        subtitle="Figure 12 supports the conservative version of the claim: the gains are real but modest."
        tone="green"
        layout="wide"
      >
        <LargeFigureSlideGraphic
          asset={paperAssets.figure12}
          label="Figure 12: ablation of maintenance strategies"
          testId="presentation-figure-12-full"
        />
      </Slide>

      <Slide
        id="slide-42"
        index="42"
        eyebrow="Section 6"
        title="6. Conclusion"
        subtitle="The conclusion restates the core contribution: agent memory needs lifecycle design, not a single universal architecture."
        layout="visualOnly"
        hideHiddenTitle
      >
        <SectionDividerGraphic section={sectionDividers.conclusion} />
      </Slide>

      <Slide
        id="slide-43"
        index="43"
        eyebrow="Conclusion"
        title="Empirical findings mapped back onto the taxonomy"
        subtitle="The strongest results concentrate around workload-aligned representation, routing, and maintenance choices."
        tone="light"
        layout="wide"
      >
        <LargeFigureSlideGraphic
          asset={paperAssets.table1EmpiricalHighlights}
          label="Highlighted empirical findings across Table 1 memory-system categories"
          testId="presentation-conclusion-highlights"
        />
      </Slide>

      <Slide
        id="slide-44"
        index="44"
        eyebrow="Close"
        title="The answer is: not fully ready"
        subtitle="Agent-native memory is not solved by longer context, RAG, or summarization alone. It needs deliberate design across representation, retrieval, updating, and maintenance."
      >
        <ReadinessChecklist />
      </Slide>

      <Slide
        id="slide-45"
        index="45"
        eyebrow="Appendix"
        title="Appendix"
        subtitle="Backup architecture and system-lineup context for discussion."
        layout="visualOnly"
        hideHiddenTitle
      >
        <SectionDividerGraphic section={sectionDividers.appendix} />
      </Slide>

      <Slide
        id="slide-46"
        index="46"
        eyebrow="Figure 1"
        title="The design space has already fragmented"
        subtitle="Stream-and-reflection systems, tiered memory, knowledge graphs, and hybrid stores look different, but all need the same lifecycle discipline."
        layout="wide"
      >
        <FigureOneArchitectureGraphic />
      </Slide>

      <Slide
        id="slide-47"
        index="47"
        eyebrow="Section 4"
        title="The taxonomy becomes an empirical test"
        subtitle="Section 3 mapped mechanisms. Section 4 asks whether those design choices change task success, retrieval fidelity, updates, long-horizon stability, and operating cost."
        tone="blue"
      >
        <EvaluationGraphic />
      </Slide>

      <Slide
        id="slide-48"
        index="48"
        eyebrow="Architecture primer"
        title="Four paper buckets organize the comparison"
        subtitle="Treat each system as a member of the Table 1 buckets: baselines, sequential context, structural/topological, or multi-paradigm hybrid."
        tone="light"
      >
        <ArchitecturePrimerGraphic />
      </Slide>

      <Slide
        id="slide-49"
        index="49"
        eyebrow="Table 1 categories"
        title="Use the paper's system buckets for the comparison"
        subtitle="These are the same buckets used to keep named systems comparable across representation, extraction, retrieval, and maintenance choices."
      >
        <ArchitectureExamplesGraphic />
      </Slide>

      <Slide
        id="slide-50"
        index="50"
        eyebrow="Systems"
        title="The comparison is deliberately heterogeneous"
        subtitle="The lineup mixes baselines and memory architectures so the paper can compare failure modes, not just named products."
      >
        <SystemLineupGraphic />
      </Slide>
    </main>
  );
}
