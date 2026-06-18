export type RecapUnitId =
  | "representation"
  | "attention"
  | "families"
  | "runtime"
  | "training"
  | "preference"
  | "reasoning"
  | "systems"
  | "evaluation";

export type RecapUnit = {
  id: RecapUnitId;
  lectureLabel: string;
  title: string;
  subtitle: string;
  coreIdea: string;
  mechanism: string;
  terms: readonly {
    label: string;
    explanation: string;
  }[];
  sourceTrace: string;
  handoff: string;
  steps: readonly string[];
  formula?: string;
};

export const recapUnits: readonly RecapUnit[] = [
  {
    id: "representation",
    lectureLabel: "Lecture 1",
    title: "Text becomes comparable vectors",
    subtitle: "Tokenization chooses units; embeddings make them learnable.",
    coreIdea:
      "A transformer never sees words directly. It sees a sequence of token ids, turns each id into a vector, and adds position information so the model can reason about order.",
    mechanism:
      "Raw text is split into token ids, usually with subword pieces so roots and rare words can share vocabulary capacity. Embedding lookup turns each id into a vector, then position information is added before attention sees the sequence.",
    terms: [
      {
        label: "Token",
        explanation:
          "The atomic unit given to the model: often a word piece, punctuation mark, space-sensitive fragment, or rare-word fragment.",
      },
      {
        label: "Embedding",
        explanation:
          "A learned vector looked up from a token id. Similar use patterns can make vectors close in representation space.",
      },
      {
        label: "Context awareness",
        explanation:
          "Static word vectors give the same word one vector everywhere; transformer layers update a token representation using its sentence context.",
      },
    ],
    sourceTrace:
      "The recap starts from subword tokenization, word2vec-style proxy tasks, and the limitation of static word vectors: the same word keeps the same representation even when its sentence meaning changes.",
    handoff:
      "This explains why later retrieval, prompting, and multimodal systems are all representation problems before they are generation problems.",
    steps: [
      "Subword tokenizers keep the vocabulary finite while reusing pieces across related words.",
      "Embedding proxy tasks learn useful geometry, but fixed word vectors are not context-aware.",
      "Position information is needed because a bag of token vectors does not preserve sequence structure.",
    ],
  },
  {
    id: "attention",
    lectureLabel: "Lecture 1",
    title: "Self-attention creates direct links",
    subtitle: "Every position can compare itself with the rest of the context.",
    coreIdea:
      "Self-attention is the move that replaces a slow memory chain with direct token-to-token comparison. Each token can ask which other positions matter for its current meaning.",
    mechanism:
      "A query asks what the current position needs, keys advertise what other positions contain, and values carry the information that is mixed after the softmax scores are computed.",
    terms: [
      {
        label: "Query",
        explanation:
          "The vector for the current position asking what kind of context it needs.",
      },
      {
        label: "Key",
        explanation:
          "The vector another position uses to advertise whether it matches the query.",
      },
      {
        label: "Value",
        explanation:
          "The information that gets averaged after query-key scores become attention weights.",
      },
    ],
    sourceTrace:
      "The lecturer rebuilds the Q/K/V explanation and the scaled dot-product attention formula before using it again for images and multimodal models.",
    handoff:
      "Attention is the shared operation behind encoder-only classifiers, decoder-only LLMs, ViT encoders, and cross-attention VLMs.",
    steps: [
      "Queries and keys are compared with scaled dot products.",
      "Softmax turns comparison scores into weights over context positions.",
      "The weighted values become a context-aware representation for each position.",
    ],
    formula: String.raw`\[\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V\]`,
  },
  {
    id: "families",
    lectureLabel: "Lecture 2",
    title: "Transformer variants create model families",
    subtitle:
      "Changing position, attention sharing, and masks changes the task.",
    coreIdea:
      "The transformer is a family of information-flow choices. Whether a model reads both directions, generates left-to-right, or conditions a decoder on an encoder determines what it is naturally good at.",
    mechanism:
      "RoPE moves relative position reasoning into the attention computation. MQA and GQA share key/value projections across heads to reduce serving cost. Keeping only the encoder, only the decoder, or both creates BERT, GPT, and T5-style behavior.",
    terms: [
      {
        label: "RoPE",
        explanation:
          "Rotary position embeddings encode relative distance by rotating query and key vectors inside attention.",
      },
      {
        label: "GQA",
        explanation:
          "Grouped-query attention lets several query heads share key/value heads, reducing KV-cache memory at serving time.",
      },
      {
        label: "Encoder vs decoder",
        explanation:
          "Encoders build bidirectional representations; decoders use causal masks so generated tokens cannot look ahead.",
      },
    ],
    sourceTrace:
      "The slides rewind RoPE, GQA, normalization placement, BERT's CLS classification pattern, decoder-only GPT text generation, and encoder-decoder T5.",
    handoff:
      "Later systems are easier to diagnose when encoder embeddings, decoder generation, and encoder-decoder conditioning are not treated as interchangeable.",
    steps: [
      "Encoder-only models build bidirectional embeddings for classification or retrieval.",
      "Decoder-only models generate autoregressively by predicting the next token.",
      "Encoder-decoder models condition a decoder on an encoded source sequence.",
    ],
  },
  {
    id: "runtime",
    lectureLabel: "Lecture 3",
    title: "LLMs are next-token systems at scale",
    subtitle:
      "Model size, routing, and decoding controls shape visible behavior.",
    coreIdea:
      "A decoder-only LLM repeatedly predicts a distribution for the next token. Many visible differences come from how the model routes computation and how the sampler turns probabilities into text.",
    mechanism:
      "Decoder-only LLMs repeatedly predict the next token. Mixture-of-experts layers route each token to a sparse subset of feed-forward experts, while decoding controls such as temperature, top-k, and top-p decide how probability mass becomes sampled text.",
    terms: [
      {
        label: "Mixture of experts",
        explanation:
          "A sparse layer where a gate sends each token to only some feed-forward experts instead of activating every expert.",
      },
      {
        label: "Temperature",
        explanation:
          "A decoding knob that sharpens or flattens the token distribution before sampling.",
      },
      {
        label: "KV cache",
        explanation:
          "Stored key/value states from previous tokens that prevent recomputing the whole context during generation.",
      },
    ],
    sourceTrace:
      "The recap revisits MoE feed-forward experts, token-level routing, greedy versus sampling behavior, and the temperature knob that makes output distributions sharper or flatter.",
    handoff:
      "This runtime view separates architecture changes from decoding changes and from serving optimizations like KV caching.",
    steps: [
      "MoE reduces active computation by choosing only some experts for each token.",
      "Sampling introduces variety; lower temperature makes the distribution more deterministic.",
      "Serving speed depends on reuse and memory movement, not only the parameter count.",
    ],
  },
  {
    id: "training",
    lectureLabel: "Lecture 4",
    title: "Training turns scale into behavior",
    subtitle: "Pretraining learns continuation; SFT teaches desired tasks.",
    coreIdea:
      "Training is staged because each stage teaches a different behavior: broad continuation, instruction following, then preference alignment.",
    mechanism:
      "Pretraining optimizes next-token prediction on huge corpora, producing an autocomplete model with broad language and code structure. Supervised fine-tuning then trains the model on instruction-like input/output pairs. Preference tuning adds negative comparison signal afterward.",
    terms: [
      {
        label: "Scaling law",
        explanation:
          "An empirical relationship between loss, compute, model size, and dataset size that guides training budgets.",
      },
      {
        label: "SFT",
        explanation:
          "Supervised fine-tuning on examples of desired input/output behavior, such as instruction-response pairs.",
      },
      {
        label: "FlashAttention",
        explanation:
          "Exact attention computed with less slow memory movement by tiling work into faster on-chip memory.",
      },
    ],
    sourceTrace:
      "The recap mentions scaling laws, compute-optimal token budgets, FlashAttention's HBM/SRAM memory story, and the three-stage pipeline from pretraining to SFT to preference tuning.",
    handoff:
      "LoRA and QLoRA fit here as cost controls: they adapt behavior through small low-rank updates instead of rewriting every base-model weight.",
    steps: [
      "Scaling laws connect loss to compute, dataset size, and parameter count.",
      "Compute-optimal training exposed that many early large models were undertrained on tokens.",
      "FlashAttention is exact attention made faster by reducing slow HBM reads and writes.",
    ],
  },
  {
    id: "preference",
    lectureLabel: "Lecture 5",
    title: "Preference tuning turns comparisons into pressure",
    subtitle: "Pairwise judgments become reward gaps and policy updates.",
    coreIdea:
      "Preference tuning starts from comparisons, not absolute truth. The model learns that one answer should score higher than another, then updates the policy while trying not to drift too far from the reference model.",
    mechanism:
      "Human preference data usually says one completion is better than another. A reward model is trained on those pairs, then an RLHF-style update pushes the language model toward high reward while keeping it close to the SFT reference model so reward hacking is constrained.",
    terms: [
      {
        label: "Reward model",
        explanation:
          "A model trained from pairwise preference data that can score a single completion at inference or training time.",
      },
      {
        label: "Bradley-Terry",
        explanation:
          "A pairwise probability model where the chance of one answer winning depends on the difference between two reward scores.",
      },
      {
        label: "Reference pressure",
        explanation:
          "A constraint that keeps the tuned policy close to the SFT or base model so it does not over-optimize the reward proxy.",
      },
    ],
    sourceTrace:
      "The lecturer revisits the RL analogy for language generation, Bradley-Terry reward modeling, PPO-style reward maximization, and the need to avoid drifting too far from the base model.",
    handoff:
      "DPO keeps the same preference-pair idea but avoids a separate online RL loop by comparing policy and reference probabilities directly.",
    steps: [
      "The LLM policy state is the text so far; the action is the next token.",
      "The reward model is trained pairwise but can score one answer at inference time.",
      "KL or reference pressure matters because the reward model is an imperfect proxy.",
    ],
    formula: String.raw`\[P(y_w \succ y_l)=\sigma(r_\theta(x,y_w)-r_\theta(x,y_l))\]`,
  },
  {
    id: "reasoning",
    lectureLabel: "Lecture 6",
    title: "Reasoning models reward verifiable work",
    subtitle:
      "The model samples attempts, scores them, and learns from the group.",
    coreIdea:
      "Reasoning training works best when the task has a checkable outcome. The model can sample several attempts, compare them within the group, and reinforce the ones that solve the task.",
    mechanism:
      "Reasoning training uses tasks where answers can be checked, samples multiple completions, and reinforces responses that score well relative to their sampled group. GRPO removes the separate value model used in PPO by using group-relative advantages.",
    terms: [
      {
        label: "Chain of thought",
        explanation:
          "Intermediate written reasoning that can help problem solving, but is useful only when incentives reward correct work rather than length.",
      },
      {
        label: "Verifiable reward",
        explanation:
          "A score that can be computed from a checkable answer, such as math correctness or code passing tests.",
      },
      {
        label: "GRPO",
        explanation:
          "Group Relative Policy Optimization, which compares sampled completions for the same prompt instead of training a separate value model.",
      },
    ],
    sourceTrace:
      "The recap points from chain-of-thought prompting to DeepSeek-R1, GRPO, verifiable rewards, length-control issues, and distillation into smaller models.",
    handoff:
      "This is why longer visible reasoning is not automatically better: the reward, length incentives, and verification target determine what improves.",
    steps: [
      "Chain-of-thought exposes intermediate problem solving as a trainable behavior.",
      "Group-relative advantages compare completions sampled for the same prompt.",
      "Distillation transfers expensive reasoning behavior into cheaper student models.",
    ],
    formula: String.raw`\[A_i \approx r_i-\frac{1}{G}\sum_{j=1}^{G}r_j\]`,
  },
  {
    id: "systems",
    lectureLabel: "Lecture 7",
    title: "RAG, tools, and agents add external state",
    subtitle:
      "Fixed weights can retrieve, call APIs, and loop through actions.",
    coreIdea:
      "A trained model's weights are not a live database. RAG, tools, and agents work around that by letting the model read external evidence, call systems, and repeat observe-act loops.",
    mechanism:
      "RAG retrieves candidate evidence, reranks or filters it, inserts it into the prompt, and asks the model to answer grounded in that context. Tool calling predicts an API name and arguments, executes the external function, then synthesizes the result. Agents repeat this observe-think-act loop.",
    terms: [
      {
        label: "RAG",
        explanation:
          "Retrieval-augmented generation: fetch relevant documents or chunks, place them in context, then answer with that evidence.",
      },
      {
        label: "Tool call",
        explanation:
          "A structured API action where the model chooses a function name and arguments, then uses the returned result.",
      },
      {
        label: "Agent loop",
        explanation:
          "A repeated cycle of observing state, choosing an action, executing it, and deciding what to do next.",
      },
    ],
    sourceTrace:
      "The slides recap RAG, tool-calling arguments, function APIs, and ReAct-style agents as the systems layer after reasoning.",
    handoff:
      "When a model is wrong because the world changed, retrieval or a tool is usually the first layer to inspect before retraining the base model.",
    steps: [
      "Retrieval failure, reranking failure, and answer synthesis failure are different bugs.",
      "Tool calls add action but also create argument, permission, and observability risks.",
      "Long agent loops compound small per-step reliability errors.",
    ],
  },
  {
    id: "evaluation",
    lectureLabel: "Lecture 8",
    title: "Evaluation turns behavior into evidence",
    subtitle: "Scores need a target, a rubric, and failure-mode awareness.",
    coreIdea:
      "Evaluation is not one leaderboard. It is the discipline of deciding what behavior matters, building a measurement for it, and checking whether the measurement itself can be gamed or biased.",
    mechanism:
      "Evaluation can use human ratings, reference metrics, LLM-as-a-Judge prompts, factuality checks, benchmark suites, and workflow reliability measures. Each only answers the question it was built to measure.",
    terms: [
      {
        label: "LLM-as-a-Judge",
        explanation:
          "A prompted model that scores outputs against criteria, often with a rubric, rationale, and known bias controls.",
      },
      {
        label: "Benchmark",
        explanation:
          "A standardized task set such as MMLU, AIME, SWE-bench, or HarmBench used to make a bounded capability claim.",
      },
      {
        label: "Goodhart's law",
        explanation:
          "When a metric becomes the target, optimization can improve the score while damaging the real goal.",
      },
    ],
    sourceTrace:
      "The final recap slide groups LLM-as-a-Judge packets with prompt, response, criteria, rationale, and score, then maps benchmarks such as MMLU, AIME, SWE-bench, HarmBench, and PIQA to capability claims.",
    handoff:
      "Evaluation is the control layer for every previous topic: decoding, training, tools, reasoning, and frontier claims all need measured evidence.",
    steps: [
      "LLM-as-a-Judge needs explicit criteria and bias controls.",
      "Benchmarks are profiles of constrained tasks, not complete product proof.",
      "Goodhart's law appears when the convenient metric replaces the real goal.",
    ],
  },
] as const;

export type CourseTraceStageId =
  | "prompt"
  | "represent"
  | "attend"
  | "generate"
  | "shape"
  | "ground"
  | "measure";

export type CourseTraceStage = {
  id: CourseTraceStageId;
  label: string;
  recapIds: readonly RecapUnitId[];
  input: string;
  operation: string;
  output: string;
};

export const courseTraceStages: readonly CourseTraceStage[] = [
  {
    id: "prompt",
    label: "Request arrives",
    recapIds: ["representation"],
    input: "A user asks a text question.",
    operation:
      "The tokenizer converts text into subword ids and embeddings with position information.",
    output: "A sequence of vectors the transformer can process.",
  },
  {
    id: "represent",
    label: "Context forms",
    recapIds: ["attention", "families"],
    input: "Token vectors and positions.",
    operation:
      "Self-attention compares queries and keys, then mixes values into context-aware states.",
    output:
      "Representations whose meaning depends on the whole visible context.",
  },
  {
    id: "attend",
    label: "Architecture constrains",
    recapIds: ["families", "runtime"],
    input: "Context-aware states.",
    operation:
      "The chosen family controls masks and information flow: encoder embeddings, decoder generation, or encoder-decoder conditioning.",
    output:
      "A model state ready for classification, retrieval, or next-token prediction.",
  },
  {
    id: "generate",
    label: "Next token chosen",
    recapIds: ["runtime"],
    input: "A probability distribution over tokens.",
    operation:
      "Greedy, sampling, temperature, top-k, or top-p decoding turns probabilities into a concrete next token.",
    output: "One more token, then the loop repeats.",
  },
  {
    id: "shape",
    label: "Behavior shaped",
    recapIds: ["training", "preference", "reasoning"],
    input: "Base next-token behavior.",
    operation:
      "Pretraining, SFT, preference tuning, and reasoning rewards change which completions the model tends to produce.",
    output:
      "A model that follows tasks, preferences, and verifiable reasoning incentives.",
  },
  {
    id: "ground",
    label: "External state added",
    recapIds: ["systems"],
    input: "A fixed-weight model and a possibly changing world.",
    operation:
      "RAG, tools, and agent loops fetch evidence or execute actions outside the model weights.",
    output: "An answer conditioned on retrieved or executed system state.",
  },
  {
    id: "measure",
    label: "Claim measured",
    recapIds: ["evaluation"],
    input: "A candidate answer or workflow trace.",
    operation:
      "Rubrics, judges, factuality checks, benchmarks, and product metrics test whether the behavior actually meets the target.",
    output: "Evidence about quality, safety, reliability, and deployment fit.",
  },
] as const;

export type TransferMode = "vit" | "vlm" | "diffusion";

export type VlmPatternId = "visualPrefix" | "crossAttention";

export type VlmPattern = {
  id: VlmPatternId;
  label: string;
  mechanism: string;
  consequence: string;
};

export const vlmPatterns: readonly VlmPattern[] = [
  {
    id: "visualPrefix",
    label: "Visual tokens in decoder",
    mechanism:
      "An image encoder produces visual token-like vectors, which are concatenated with text tokens and consumed by a decoder-only language model.",
    consequence:
      "The decoder can reuse next-token generation, but it needs visual instruction tuning so image features become useful for answers.",
  },
  {
    id: "crossAttention",
    label: "Cross-attend to image memory",
    mechanism:
      "The language decoder keeps text generation as its main stream while cross-attention reads a separate set of visual encoder features.",
    consequence:
      "Image evidence stays in a separate memory source, which makes the architecture closer to encoder-decoder conditioning.",
  },
] as const;

export type ClosingThemeId =
  | "architecture"
  | "data"
  | "serving"
  | "hardware"
  | "limits";

export type ClosingTheme = {
  id: ClosingThemeId;
  label: string;
  claim: string;
  details: readonly string[];
};

export const closingThemes: readonly ClosingTheme[] = [
  {
    id: "architecture",
    label: "Basic design is still open",
    claim:
      "The transformer stack is not a finished recipe; papers still vary basic choices.",
    details: [
      "Optimizers such as AdamW and Muon-style variants are active design decisions.",
      "Normalization placement, RMSNorm-style choices, activation functions, MoE use, layer count, and head grouping still vary.",
      "The lecture frames this as foundational research, not as a solved implementation detail.",
    ],
  },
  {
    id: "data",
    label: "Data quality is a research problem",
    claim:
      "More web data is not automatically better when the web increasingly contains generated text.",
    details: [
      "Generated text can be less diverse, shifting the training distribution.",
      "Model collapse motivates curation rather than indiscriminate scraping.",
      "Mid-training on higher-quality corpora is one response between pretraining and fine-tuning.",
    ],
  },
  {
    id: "serving",
    label: "Serving economics matter",
    claim:
      "The frontier is not only best benchmark score; quality, cost, latency, and volume all matter.",
    details: [
      "Small language models make sense when the use case is high volume or constrained.",
      "A model choice should be tied to the product workflow, not to a leaderboard alone.",
      "This is the source concept behind quality/cost Pareto thinking without needing a model ranking table.",
    ],
  },
  {
    id: "hardware",
    label: "Attention stresses hardware",
    claim:
      "GPU matrix multiply is not the whole story because transformer blocks move and reread a lot of memory.",
    details: [
      "Attention requires frequent key/value reads and writes.",
      "FlashAttention already showed that memory movement can dominate runtime.",
      "The slides point to hardware that supports attention operations more natively.",
    ],
  },
  {
    id: "limits",
    label: "Open problems remain",
    claim:
      "RAG, tools, and tuning help, but they do not remove all limits of current LLMs.",
    details: [
      "Weights are fixed after training, so continuous learning is unresolved.",
      "Hallucination follows from next-token prediction not being direct fact mapping.",
      "Personalization, interpretability, safety, and reliable autonomous action remain open.",
    ],
  },
] as const;

export function getRecapUnit(id: RecapUnitId): RecapUnit {
  const unit = recapUnits.find((candidate) => candidate.id === id);
  if (!unit) {
    throw new Error(`Unknown Lecture 9 recap unit: ${id}`);
  }
  return unit;
}

export function getCourseTraceStage(id: CourseTraceStageId): CourseTraceStage {
  const stage = courseTraceStages.find((candidate) => candidate.id === id);
  if (!stage) {
    throw new Error(`Unknown Lecture 9 trace stage: ${id}`);
  }
  return stage;
}

export function getVlmPattern(id: VlmPatternId): VlmPattern {
  const pattern = vlmPatterns.find((candidate) => candidate.id === id);
  if (!pattern) {
    throw new Error(`Unknown VLM pattern: ${id}`);
  }
  return pattern;
}

export function getClosingTheme(id: ClosingThemeId): ClosingTheme {
  const theme = closingThemes.find((candidate) => candidate.id === id);
  if (!theme) {
    throw new Error(`Unknown Lecture 9 closing theme: ${id}`);
  }
  return theme;
}

export function getPatchTokenCount({
  imageWidth,
  imageHeight,
  patchSize,
}: {
  imageWidth: number;
  imageHeight: number;
  patchSize: number;
}): number {
  if (patchSize <= 0 || imageWidth <= 0 || imageHeight <= 0) return 0;

  return Math.ceil(imageWidth / patchSize) * Math.ceil(imageHeight / patchSize);
}

export function getGenerationPassComparison({
  outputTokens,
  maskedTokensPerPass,
}: {
  outputTokens: number;
  maskedTokensPerPass: number;
}): {
  autoregressivePasses: number;
  maskedDiffusionPasses: number;
  speedupRatio: number;
} {
  const tokens = Math.max(0, Math.ceil(outputTokens));
  const perPass = Math.max(1, Math.ceil(maskedTokensPerPass));
  const maskedDiffusionPasses = tokens === 0 ? 0 : Math.ceil(tokens / perPass);

  return {
    autoregressivePasses: tokens,
    maskedDiffusionPasses,
    speedupRatio:
      maskedDiffusionPasses === 0 ? 0 : tokens / maskedDiffusionPasses,
  };
}
