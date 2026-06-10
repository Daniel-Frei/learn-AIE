"use client";

import { useMemo, useState } from "react";
import MathText from "../../MathText";
import type { LearningExperience } from "../../../lib/learning";
import {
  CheckForUnderstanding,
  ConceptCard,
  FormulaBlock,
  InteractiveComparison,
  LearningHero,
  MisconceptionCallout,
  ProcessSteps,
  QuizTransitionButton,
  RecapSection,
  WorkedExample,
} from "../LearningPrimitives";

type Props = {
  experience: LearningExperience;
};

const sentenceTokens = ["A", "cute", "teddy", "bear", "is", "reading"];

const tokenizationMethods = {
  word: {
    label: "Word",
    tokens: ["A", "cute", "teddy", "bear", "is", "reading", "."],
    sequence: "Short",
    vocabulary: "Large",
    body: "Simple and readable, but rare forms such as bears, reading, or misspellings may need separate entries or fall into an unknown token.",
  },
  subword: {
    label: "Subword",
    tokens: ["A", "cute", "ted", "##dy", "bear", "is", "read", "##ing", "."],
    sequence: "Moderate",
    vocabulary: "Moderate",
    body: "A practical compromise: reuse common pieces, reduce unseen-word pressure, and keep sequences shorter than pure characters.",
  },
  character: {
    label: "Character",
    tokens: [
      "A",
      "c",
      "u",
      "t",
      "e",
      "t",
      "e",
      "d",
      "d",
      "y",
      "b",
      "e",
      "a",
      "r",
    ],
    sequence: "Long",
    vocabulary: "Small",
    body: "Robust to casing and misspellings, but it makes every word longer for the model and pushes meaning into many small steps.",
  },
} as const;

type TokenizationMethod = keyof typeof tokenizationMethods;

const attentionRows = [
  {
    query: "bear",
    focus: [
      { token: "A", weight: 0.04 },
      { token: "cute", weight: 0.18 },
      { token: "teddy", weight: 0.38 },
      { token: "bear", weight: 0.24 },
      { token: "is", weight: 0.08 },
      { token: "reading", weight: 0.08 },
    ],
    note: "The token can recover the phrase teddy bear while still seeing the local adjective.",
  },
  {
    query: "reading",
    focus: [
      { token: "A", weight: 0.08 },
      { token: "cute", weight: 0.06 },
      { token: "teddy", weight: 0.14 },
      { token: "bear", weight: 0.36 },
      { token: "is", weight: 0.26 },
      { token: "reading", weight: 0.1 },
    ],
    note: "The verb benefits from direct access to the subject phrase and the nearby auxiliary.",
  },
  {
    query: "cute",
    focus: [
      { token: "A", weight: 0.16 },
      { token: "cute", weight: 0.18 },
      { token: "teddy", weight: 0.3 },
      { token: "bear", weight: 0.22 },
      { token: "is", weight: 0.06 },
      { token: "reading", weight: 0.08 },
    ],
    note: "A modifier can mostly attend to the noun phrase it helps describe.",
  },
] as const;

const architectureStages = [
  {
    id: "input",
    label: "Input",
    title: "Token plus position",
    body: "The sentence is tokenized, each token gets a learned embedding, and position information is added so order is visible to attention.",
  },
  {
    id: "encoder",
    label: "Encoder",
    title: "Source tokens look at source tokens",
    body: "Encoder self-attention makes every source token representation context-aware, then a feedforward layer transforms each token.",
  },
  {
    id: "decoder-self",
    label: "Masked decoder",
    title: "Generated tokens look left",
    body: "Decoder self-attention is masked so the next-token prediction can use only tokens already produced.",
  },
  {
    id: "cross",
    label: "Cross-attention",
    title: "Decoder asks the source",
    body: "Decoder states provide queries. Encoder outputs provide keys and values, so the decoder can retrieve source-side information.",
  },
  {
    id: "softmax",
    label: "Softmax",
    title: "Choose the next token",
    body: "A linear projection creates vocabulary logits, and softmax turns those scores into next-token probabilities.",
  },
] as const;

function TransformerPathVisual() {
  const stages = [
    "Text",
    "Tokens",
    "Embeddings + position",
    "Self-attention",
    "Next-token distribution",
  ];

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <p className="text-sm font-semibold uppercase tracking-wide text-sky-300">
        Lecture 1 route
      </p>
      <div className="mt-4 grid gap-3">
        {stages.map((stage, index) => (
          <div key={stage} className="flex items-center gap-3">
            <div className="flex h-11 flex-1 items-center justify-center rounded-md border border-slate-700 bg-slate-950 px-3 text-center text-sm font-semibold text-slate-100">
              {stage}
            </div>
            {index < stages.length - 1 && (
              <span className="text-lg font-semibold text-sky-300">-&gt;</span>
            )}
          </div>
        ))}
      </div>
      <div className="mt-5 rounded-md border border-slate-800 bg-slate-950 p-3">
        <div className="grid grid-cols-6 gap-1">
          {sentenceTokens.map((token) => (
            <div key={token} className="space-y-2">
              <div className="h-20 rounded-sm bg-slate-800 p-1">
                <div
                  className="mt-auto rounded-sm bg-sky-400"
                  style={{
                    height: `${token === "bear" ? 86 : token === "teddy" ? 72 : 42}%`,
                  }}
                />
              </div>
              <p className="truncate text-center text-xs font-semibold text-slate-300">
                {token}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function TokenizationExplorer() {
  const [method, setMethod] = useState<TokenizationMethod>("subword");
  const activeMethod = tokenizationMethods[method];

  return (
    <section
      data-testid="tokenization-explorer"
      className="rounded-lg border border-slate-800 bg-slate-900 p-5"
    >
      <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-slate-50">
            Tokenization tradeoff
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
            The same sentence can be cut into words, subwords, or characters.
            The cut changes sequence length, vocabulary size, and how hard rare
            words are to represent.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {Object.entries(tokenizationMethods).map(([id, item]) => (
            <button
              key={id}
              type="button"
              onClick={() => setMethod(id as TokenizationMethod)}
              className={`rounded-md border px-3 py-2 text-sm font-semibold ${
                method === id
                  ? "border-sky-400 bg-sky-400 text-slate-950"
                  : "border-slate-700 text-slate-200 hover:border-slate-500"
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-5 rounded-md border border-slate-800 bg-slate-950 p-4">
        <div className="flex flex-wrap gap-2">
          {activeMethod.tokens.map((token, index) => (
            <span
              key={`${token}-${index}`}
              className="rounded-md border border-slate-700 bg-slate-900 px-2 py-1 font-mono text-sm text-slate-100"
            >
              {token}
            </span>
          ))}
        </div>
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          <TokenizationTile
            label="Sequence length"
            value={activeMethod.sequence}
          />
          <TokenizationTile
            label="Vocabulary pressure"
            value={activeMethod.vocabulary}
          />
        </div>
        <p role="status" className="mt-4 text-sm leading-6 text-slate-300">
          {activeMethod.body}
        </p>
      </div>
    </section>
  );
}

function TokenizationTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-slate-800 bg-slate-900 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
        {label}
      </p>
      <p className="mt-2 text-lg font-semibold text-slate-50">{value}</p>
    </div>
  );
}

function AttentionFocusExplorer() {
  const [activeQuery, setActiveQuery] = useState<
    (typeof attentionRows)[number]["query"]
  >(attentionRows[0].query);
  const row =
    attentionRows.find((item) => item.query === activeQuery) ??
    attentionRows[0];
  const strongest = useMemo(
    () =>
      row.focus.reduce((best, item) =>
        item.weight > best.weight ? item : best,
      ),
    [row],
  );

  return (
    <section
      data-testid="attention-focus-explorer"
      className="rounded-lg border border-slate-800 bg-slate-900 p-5"
    >
      <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-slate-50">
            Self-attention as a lookup
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
            Pick a query token. It compares its query vector with key vectors,
            softmax turns those scores into weights, and the output becomes a
            weighted mix of value vectors.
          </p>
        </div>
        <div className="rounded-md bg-slate-950 px-3 py-2 text-sm font-semibold text-emerald-300">
          Top focus: {strongest.token}
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        {attentionRows.map((item) => (
          <button
            key={item.query}
            type="button"
            onClick={() => setActiveQuery(item.query)}
            className={`rounded-md border px-3 py-2 text-sm font-semibold ${
              activeQuery === item.query
                ? "border-sky-400 bg-sky-400 text-slate-950"
                : "border-slate-700 text-slate-200 hover:border-slate-500"
            }`}
          >
            Query: {item.query}
          </button>
        ))}
      </div>

      <div className="mt-5 grid gap-3">
        {row.focus.map((item) => (
          <div
            key={item.token}
            className="rounded-md border border-slate-800 bg-slate-950 p-3"
          >
            <div className="flex items-center justify-between gap-3 text-sm">
              <span className="font-semibold text-slate-100">
                Key/value: {item.token}
              </span>
              <span className="font-mono text-slate-300">
                {item.weight.toFixed(2)}
              </span>
            </div>
            <div className="mt-2 h-3 overflow-hidden rounded-full bg-slate-800">
              <div
                className="h-full rounded-full bg-sky-400"
                style={{ width: `${Math.max(2, item.weight * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      <p role="status" className="mt-4 text-sm leading-6 text-slate-300">
        {row.note}
      </p>
    </section>
  );
}

function ArchitectureStepper() {
  const [activeStageId, setActiveStageId] = useState<
    (typeof architectureStages)[number]["id"]
  >(architectureStages[0].id);
  const activeStage =
    architectureStages.find((stage) => stage.id === activeStageId) ??
    architectureStages[0];

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <h2 className="text-xl font-semibold text-slate-50">
        Encoder-decoder path
      </h2>
      <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
        The original transformer was introduced for translation. Walk through
        where information lives as the source sentence becomes target tokens.
      </p>

      <div className="mt-5 flex flex-wrap gap-2">
        {architectureStages.map((stage) => (
          <button
            key={stage.id}
            type="button"
            onClick={() => setActiveStageId(stage.id)}
            className={`rounded-md border px-3 py-2 text-sm font-semibold ${
              stage.id === activeStage.id
                ? "border-emerald-300 bg-emerald-300 text-slate-950"
                : "border-slate-700 text-slate-200 hover:border-slate-500"
            }`}
          >
            {stage.label}
          </button>
        ))}
      </div>

      <div className="mt-5 grid gap-4 md:grid-cols-[0.95fr_1.05fr]">
        <div className="rounded-md border border-slate-800 bg-slate-950 p-4">
          <div className="grid gap-3">
            {architectureStages.map((stage, index) => (
              <div key={stage.id} className="flex items-center gap-3">
                <span
                  className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-bold ${
                    stage.id === activeStage.id
                      ? "bg-emerald-300 text-slate-950"
                      : "bg-slate-800 text-slate-300"
                  }`}
                >
                  {index + 1}
                </span>
                <div
                  className={`flex-1 rounded-md border px-3 py-2 text-sm font-semibold ${
                    stage.id === activeStage.id
                      ? "border-emerald-300 text-emerald-100"
                      : "border-slate-800 text-slate-300"
                  }`}
                >
                  {stage.label}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-md border border-slate-800 bg-slate-950 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-emerald-300">
            Active stage
          </p>
          <h3 className="mt-2 text-lg font-semibold text-slate-50">
            {activeStage.title}
          </h3>
          <p role="status" className="mt-3 text-sm leading-6 text-slate-300">
            {activeStage.body}
          </p>
        </div>
      </div>
    </section>
  );
}

function MethodComparisonTable() {
  const rows = [
    {
      method: "Word2vec",
      strength: "Learns dense vectors from proxy prediction tasks.",
      limit: "Static embeddings and no word order in the basic setup.",
    },
    {
      method: "RNN / LSTM",
      strength: "Processes order through a recurrent hidden state.",
      limit: "Long paths make training slow and long-range memory difficult.",
    },
    {
      method: "Transformer",
      strength: "Lets positions connect directly with attention.",
      limit: "Still needs position signals and careful training objectives.",
    },
  ] as const;

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <h2 className="text-xl font-semibold text-slate-50">
        Why the architecture changed
      </h2>
      <div className="mt-4 grid gap-3">
        {rows.map((row) => (
          <div
            key={row.method}
            className="grid gap-3 rounded-md border border-slate-800 bg-slate-950 p-3 md:grid-cols-[8rem_1fr_1fr]"
          >
            <p className="font-semibold text-sky-300">{row.method}</p>
            <p className="text-sm leading-6 text-slate-300">{row.strength}</p>
            <p className="text-sm leading-6 text-slate-300">{row.limit}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

export default function StanfordCME295Lecture1LearningPage({
  experience,
}: Props) {
  return (
    <main className="bg-slate-950 text-slate-50">
      <LearningHero
        eyebrow="Stanford CME295 Lecture 1"
        title="Follow text into a transformer"
        summary="Start with a sentence, cut it into tokens, turn those tokens into vectors, then watch attention build context for translation and next-token prediction."
        meta={`${experience.durationMinutes} min interactive prep / ${experience.level}`}
        outcomes={experience.outcomes}
        visual={<TransformerPathVisual />}
      />

      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-10 md:py-12">
        <ProcessSteps
          title="The lecture's core arc"
          steps={[
            {
              title: "Name the NLP job",
              body: "Classification, token-level structure, and generation ask for different outputs and different evaluation habits.",
            },
            {
              title: "Represent the words",
              body: "Models need numbers, so text becomes tokens, one-hot identifiers, dense embeddings, and position-aware vectors.",
            },
            {
              title: "Replace the bottleneck",
              body: "Attention gives direct token-to-token paths that avoid relying only on a recurrent hidden state.",
            },
          ]}
        />

        <section className="grid gap-4 md:grid-cols-3">
          <ConceptCard title="Classification" label="Text to label">
            <p>
              Sentiment, intent, language, and topic tasks map a whole input
              text to one or more labels. Accuracy alone can mislead when class
              balance is uneven.
            </p>
          </ConceptCard>
          <ConceptCard title="Structured labels" label="Text to many labels">
            <p>
              Named entity recognition, part-of-speech tagging, and parsing
              preserve token or span structure instead of collapsing the whole
              sentence to one output.
            </p>
          </ConceptCard>
          <ConceptCard title="Generation" label="Text to text">
            <p>
              Translation, summarization, question answering, and free-form
              writing produce variable-length text, so exact-reference
              evaluation is harder.
            </p>
          </ConceptCard>
        </section>

        <TokenizationExplorer />

        <InteractiveComparison
          title="Representations: identifier versus meaning"
          prompt="Tap each representation and ask what the model can infer before context is considered."
          items={[
            {
              id: "one-hot",
              label: "One-hot",
              title: "A unique id, not a meaning vector",
              body: "A one-hot vector says which vocabulary entry is present. It is sparse, vocabulary-sized, and makes bear and teddy as unrelated as bear and alarm unless later layers learn otherwise.",
            },
            {
              id: "embedding",
              label: "Embedding",
              title: "A learned dense vector",
              body: "An embedding is learned from data so related tokens can land nearer each other. Word2vec uses proxy tasks such as CBOW and Skip-Gram to make useful geometry emerge.",
            },
            {
              id: "context",
              label: "Contextual",
              title: "The vector changes after attention",
              body: "Classic Word2vec gives a token the same vector everywhere. A transformer updates token representations by letting each position look at the rest of the sequence.",
            },
          ]}
        />

        <WorkedExample
          title="Worked example: why subwords help"
          setup="Suppose a word-level tokenizer knows bear and read but sees bears and rereading during deployment."
          steps={[
            "A word-level tokenizer may map unseen full words to an unknown token, losing useful morphology.",
            "A subword tokenizer can reuse pieces such as bear, ##s, re, read, and ##ing, so the model still gets meaningful parts.",
            "The cost is a longer token sequence, and sequence length matters because attention compares tokens with tokens.",
          ]}
        />

        <MethodComparisonTable />

        <MisconceptionCallout
          misconception="An RNN remembers the whole earlier sentence equally well."
          correction="A recurrent hidden state carries information step by step. Important early information can be weakened by many transitions and vanishing gradients, which is why attention was introduced to create more direct links to relevant positions."
        />

        <FormulaBlock
          title="Scaled dot-product attention"
          formula={String.raw`\[\begin{aligned}S&=\frac{QK^\top}{\sqrt{d_k}}\\A&=\operatorname{softmax}(S)\\O&=AV\end{aligned}\]`}
          explanation="Queries ask for information, keys are matched against those queries, softmax creates attention weights, and values are mixed according to those weights. The scaling term keeps large dot products from making softmax too sharp as dimensions grow."
        />

        <AttentionFocusExplorer />

        <CheckForUnderstanding
          testId="qkv-check"
          title="Check: query, key, value"
          question="In self-attention, what produces the attention weights before values are mixed?"
          correctIndex={1}
          options={[
            {
              label:
                "Values are compared with values, then the result is used as the next-token distribution.",
              explanation:
                "Values are the content being mixed. The attention weights come from query-key comparisons, not value-value comparisons.",
            },
            {
              label:
                "Queries are compared with keys, and softmax normalizes those scores.",
              explanation:
                "That is the central lookup pattern: query-key scores become weights, then those weights mix values.",
            },
            {
              label:
                "Position encodings alone decide which tokens should attend to each other.",
              explanation:
                "Position information helps order become visible, but learned query and key projections determine attention scores.",
            },
          ]}
        />

        <InteractiveComparison
          title="Three attention roles in the transformer"
          prompt="The same attention operation appears in different places. Tap each one and track what can be seen."
          items={[
            {
              id: "encoder-self",
              label: "Encoder self",
              title: "Source tokens see the source sequence",
              body: "Encoder self-attention is bidirectional over the input sentence, building context-aware source representations.",
            },
            {
              id: "decoder-self",
              label: "Masked decoder",
              title: "Target tokens see only the left context",
              body: "Masked self-attention prevents a training-time decoder state from peeking at future target tokens.",
            },
            {
              id: "cross",
              label: "Cross",
              title: "The decoder reads the encoded source",
              body: "Decoder states provide queries, while encoder outputs provide keys and values for source-conditioned generation.",
            },
          ]}
        />

        <ArchitectureStepper />

        <FormulaBlock
          title="Label smoothing changes the target, not the softmax"
          formula={String.raw`\[y_{\text{true}}=1-\epsilon,\qquad y_{\text{other}}=\frac{\epsilon}{V-1}\]`}
          explanation="Instead of training against a target that says one token is correct with probability 1 and every other token has probability 0, label smoothing leaves a little probability mass for alternatives. That discourages overconfidence."
        />

        <CheckForUnderstanding
          title="Check: masked decoding"
          question="Why does a decoder use masked self-attention while training an autoregressive translation model?"
          correctIndex={2}
          options={[
            {
              label:
                "To stop the decoder from using the source sentence at all.",
              explanation:
                "The decoder still uses the source through cross-attention. Masking applies to target-side self-attention.",
            },
            {
              label:
                "To force every generated token to attend equally to all earlier tokens.",
              explanation:
                "Masking controls what positions are visible; it does not force equal attention weights.",
            },
            {
              label:
                "To match generation, where the next token cannot depend on future target tokens.",
              explanation:
                "Correct. During generation, future target tokens do not exist yet, so training must not leak them.",
            },
          ]}
        />

        <RecapSection
          title="Before you start the MCQs"
          items={[
            "Classification, structured labeling, and generation differ in output shape and evaluation.",
            "Tokenization trades off vocabulary size, sequence length, and rare-word handling.",
            "One-hot vectors identify tokens; learned embeddings encode useful geometry.",
            "RNNs preserve order but create long paths for information and gradients.",
            "Attention compares queries with keys, normalizes scores, and mixes values.",
            "Transformers add position information because attention alone is order-agnostic.",
            "Encoder self-attention, masked decoder self-attention, and cross-attention serve different visibility patterns.",
            "Multi-head attention lets the model learn several relation patterns in parallel.",
          ]}
        />

        <section className="flex flex-col gap-4 rounded-lg border border-emerald-500/40 bg-emerald-950/20 p-5 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-xl font-semibold text-emerald-100">
              Ready for the Stanford CME295 Lecture 1 questions
            </h2>
            <p className="mt-2 text-sm leading-6 text-slate-300">
              Practice the same concepts with questions on NLP task types,
              tokenization, embeddings, RNNs, attention, transformer blocks,
              positional encodings, label smoothing, and decoding.
            </p>
          </div>
          <QuizTransitionButton sourceId={experience.sourceId} />
        </section>

        <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
          <h2 className="text-lg font-semibold text-slate-50">
            Compact formula board
          </h2>
          <div className="mt-4 space-y-3 text-slate-200">
            <MathText
              text={String.raw`\[\text{tokens}+\text{position}\]`}
              className="max-w-full overflow-x-auto"
            />
            <MathText
              text={String.raw`\[\begin{aligned}S&=QK^\top/\sqrt{d_k}\\O&=\operatorname{softmax}(S)V\end{aligned}\]`}
              className="max-w-full overflow-x-auto"
            />
            <MathText
              text={String.raw`\[\text{query}+\text{source KV}\rightarrow\text{context}\]`}
              className="max-w-full overflow-x-auto"
            />
          </div>
        </section>
      </div>
    </main>
  );
}
