import { Question } from "../../quiz";

export const stanfordCME295Lecture2Questions: Question[] = [
  // ============================================================
  // Lecture 2 – Transformer-Based Models & Tricks
  // Coverage: position embeddings, RoPE, attention variants,
  // normalization, sparse attention, model families, BERT
  // ============================================================

  // ============================================================
  // Q1–Q25: ALL TRUE (4 correct answers)
  // ============================================================

  {
    id: "cme295-lect2-q01",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe why transformers require explicit position information?",
    options: [
      {
        text: "Self-attention allows tokens to interact without an inherent notion of order.",
        isCorrect: true,
      },
      {
        text: "Unlike recurrent neural networks, transformers do not process tokens sequentially.",
        isCorrect: true,
      },
      {
        text: "Without position information, token representations would be permutation-invariant.",
        isCorrect: true,
      },
      {
        text: "Injecting position information helps the model reason about relative and absolute order.",
        isCorrect: true,
      },
    ],
    explanation:
      "Because self-attention connects all tokens directly, transformers need explicit mechanisms to encode order and relative positions. Core ideas: Self-attention allows tokens to interact without an inherent notion of order; Unlike recurrent neural networks, transformers do not process tokens sequentially; Without position information, token representations would be permutation-invariant; Injecting position information helps the model reason about relative and absolute order.",
  },

  {
    id: "cme295-lect2-q02",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which properties are true of learned absolute position embeddings?",
    options: [
      {
        text: "Each position index is associated with a trainable embedding vector.",
        isCorrect: true,
      },
      {
        text: "Position embeddings are added to token embeddings before attention.",
        isCorrect: true,
      },
      {
        text: "The maximum usable position is limited by the range seen during training.",
        isCorrect: true,
      },
      {
        text: "The embeddings are optimized via gradient descent together with other parameters.",
        isCorrect: true,
      },
    ],
    explanation:
      "Learned absolute embeddings treat positions like tokens, but they do not extrapolate beyond trained sequence lengths. Core ideas: Each position index is associated with a trainable embedding vector; Position embeddings are added to token embeddings before attention; The maximum usable position is limited by the range seen during training; The embeddings are optimized via gradient descent together with other parameters.",
  },

  {
    id: "cme295-lect2-q03",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly characterize sinusoidal (hard-coded) position embeddings?",
    options: [
      {
        text: "They use sine and cosine functions with different frequencies across dimensions.",
        isCorrect: true,
      },
      {
        text: "They allow extrapolation to sequence lengths not seen during training.",
        isCorrect: true,
      },
      {
        text: "Their dot products depend on the relative distance between positions.",
        isCorrect: true,
      },
      {
        text: "They match the embedding dimensionality of token representations.",
        isCorrect: true,
      },
    ],
    explanation:
      "Sinusoidal embeddings encode position using fixed trigonometric functions that naturally reflect relative distances. Core ideas: they use sine and cosine functions with different frequencies across dimensions; they allow extrapolation to sequence lengths not seen during training; Their dot products depend on the relative distance between positions; they match the embedding dimensionality of token representations.",
  },

  {
    id: "cme295-lect2-q04",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about relative position information in attention are correct?",
    options: [
      {
        text: "Relative position information can be injected directly into attention score computation.",
        isCorrect: true,
      },
      {
        text: "Bias terms added to query–key scores can encode distance information.",
        isCorrect: true,
      },
      {
        text: "Relative approaches focus on how far apart tokens are rather than their absolute indices.",
        isCorrect: true,
      },
      {
        text: "Relative formulations align more directly with how similarity is used in attention.",
        isCorrect: true,
      },
    ],
    explanation:
      "Since attention relies on similarity scores, encoding relative distance directly in those scores is often more natural. Core ideas: Relative position information can be injected directly into attention score computation; Bias terms added to query–key scores can encode distance information; Relative approaches focus on how far apart tokens are rather than their absolute indices; Relative formulations align more directly with how similarity is used in attention.",
  },

  {
    id: "cme295-lect2-q05",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which properties apply to the T5 relative position bias mechanism?",
    options: [
      {
        text: "It adds a learned bias term to the attention logits.",
        isCorrect: true,
      },
      {
        text: "Distances between positions are bucketized before bias lookup.",
        isCorrect: true,
      },
      {
        text: "The bias is added inside the softmax normalization.",
        isCorrect: true,
      },
      {
        text: "Different attention heads can learn different bias values.",
        isCorrect: true,
      },
    ],
    explanation:
      "T5 introduces learnable relative biases per head, enabling flexible distance-aware attention. Core ideas: it adds a learned bias term to the attention logits; Distances between positions are bucketized before bias lookup; The bias is added inside the softmax normalization; Different attention heads can learn different bias values.",
  },

  {
    id: "cme295-lect2-q06",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements are correct about Attention with Linear Biases (ALiBi)?",
    options: [
      {
        text: "ALiBi uses a deterministic linear function of relative distance.",
        isCorrect: true,
      },
      {
        text: "It does not rely on learned position embeddings.",
        isCorrect: true,
      },
      {
        text: "The bias grows linearly as positions become farther apart.",
        isCorrect: true,
      },
      {
        text: "It is designed to support length extrapolation at inference time.",
        isCorrect: true,
      },
    ],
    explanation:
      "ALiBi encodes distance with simple linear penalties, avoiding learned embeddings while enabling long-context generalization. Core ideas: ALiBi uses a deterministic linear function of relative distance; it does not rely on learned position embeddings; The bias grows linearly as positions become farther apart; it is designed to support length extrapolation at inference time.",
  },

  {
    id: "cme295-lect2-q07",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe Rotary Position Embeddings (RoPE)?",
    options: [
      {
        text: "RoPE rotates query and key vectors as a function of position.",
        isCorrect: true,
      },
      {
        text: "Rotations are implemented using block-wise 2D rotation matrices.",
        isCorrect: true,
      },
      {
        text: "The resulting attention scores depend on relative position differences.",
        isCorrect: true,
      },
      {
        text: "RoPE is widely used in modern large language models.",
        isCorrect: true,
      },
    ],
    explanation:
      "RoPE integrates positional information directly into attention via rotations, making similarity depend on relative offsets. Core ideas: RoPE rotates query and key vectors as a function of position; Rotations are implemented using block-wise 2D rotation matrices; The resulting attention scores depend on relative position differences; RoPE is widely used in modern large language models.",
  },

  {
    id: "cme295-lect2-q08",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements about the mathematical intuition of RoPE are correct?",
    options: [
      {
        text: "Rotating both queries and keys preserves vector norms.",
        isCorrect: true,
      },
      {
        text: "Dot products after rotation encode relative position through angle differences.",
        isCorrect: true,
      },
      {
        text: "Higher-frequency dimensions vary faster across positions.",
        isCorrect: true,
      },
      {
        text: "The construction generalizes sinusoidal embeddings into the attention space.",
        isCorrect: true,
      },
    ],
    explanation:
      "RoPE extends sinusoidal ideas by embedding them directly into query–key interactions via rotations. Core ideas: Rotating both queries and keys preserves vector norms; Dot products after rotation encode relative position through angle differences; Higher-frequency dimensions vary faster across positions; The construction generalizes sinusoidal embeddings into the attention space.",
  },

  {
    id: "cme295-lect2-q09",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about attention weight decay with RoPE are correct?",
    options: [
      {
        text: "Attention scores exhibit a long-term decay as distance increases.",
        isCorrect: true,
      },
      {
        text: "The decay is not strictly monotonic due to oscillatory trigonometric components.",
        isCorrect: true,
      },
      {
        text: "Closer tokens tend to have higher similarity than distant ones.",
        isCorrect: true,
      },
      {
        text: "Upper bounds on similarity can be derived analytically.",
        isCorrect: true,
      },
    ],
    explanation:
      "RoPE yields attention behaviors where similarity weakens with distance, matching linguistic intuition. Core ideas: Attention scores exhibit a long-term decay as distance increases; The decay is not strictly monotonic due to oscillatory trigonometric components; Closer tokens tend to have higher similarity than distant ones; Upper bounds on similarity can be derived analytically.",
  },

  {
    id: "cme295-lect2-q10",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the purpose of layer normalization in transformers?",
    options: [
      {
        text: "It stabilizes training by normalizing activations.",
        isCorrect: true,
      },
      {
        text: "It reduces sensitivity to scale variations across dimensions.",
        isCorrect: true,
      },
      {
        text: "It improves convergence speed during optimization.",
        isCorrect: true,
      },
      {
        text: "It introduces learnable scaling and shifting parameters.",
        isCorrect: true,
      },
    ],
    explanation:
      "Layer normalization standardizes activations per token, improving numerical stability and training dynamics. Core ideas: it stabilizes training by normalizing activations; it reduces sensitivity to scale variations across dimensions; it improves convergence speed during optimization; it introduces learnable scaling and shifting parameters.",
  },

  // ============================================================
  // Q11–Q25: ALL TRUE (4 correct answers)
  // ============================================================

  {
    id: "cme295-lect2-q11",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe limitations of learned absolute position embeddings?",
    options: [
      {
        text: "They cannot naturally extrapolate to sequence lengths unseen during training.",
        isCorrect: true,
      },
      {
        text: "They may encode dataset-specific positional biases.",
        isCorrect: true,
      },
      {
        text: "They require a fixed maximum sequence length during training.",
        isCorrect: true,
      },
      {
        text: "They are learned jointly with token embeddings via gradient descent.",
        isCorrect: true,
      },
    ],
    explanation:
      "Learned absolute position embeddings are tied to the training distribution and sequence length, limiting generalization. Core ideas: they cannot naturally extrapolate to sequence lengths unseen during training; they may encode dataset-specific positional biases; they require a fixed maximum sequence length during training; they are learned jointly with token embeddings via gradient descent.",
  },

  {
    id: "cme295-lect2-q12",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly explain why relative position information is more natural for attention mechanisms?",
    options: [
      {
        text: "Attention compares pairs of tokens via similarity scores.",
        isCorrect: true,
      },
      {
        text: "Relative distance directly affects query–key interactions.",
        isCorrect: true,
      },
      {
        text: "Absolute indices are less relevant than pairwise offsets.",
        isCorrect: true,
      },
      {
        text: "Injecting position info into attention aligns with the dot-product formulation.",
        isCorrect: true,
      },
    ],
    explanation:
      "Attention fundamentally operates on token pairs, making relative position information a better inductive bias. Core ideas: Attention compares pairs of tokens via similarity scores; Relative distance directly affects query–key interactions; Absolute indices are less relevant than pairwise offsets; Injecting position info into attention aligns with the dot-product formulation.",
  },

  {
    id: "cme295-lect2-q13",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which properties of sinusoidal position embeddings enable relative-distance awareness?",
    options: [
      {
        text: "Dot products between embeddings depend on position differences.",
        isCorrect: true,
      },
      {
        text: "Trigonometric identities link cosine and sine of offsets.",
        isCorrect: true,
      },
      {
        text: "Multiple frequencies encode both short- and long-range structure.",
        isCorrect: true,
      },
      {
        text: "Cosine similarity peaks when positions are identical.",
        isCorrect: true,
      },
    ],
    explanation:
      "Sinusoidal embeddings are constructed so that similarity reflects relative position via trigonometric structure. Core ideas: Dot products between embeddings depend on position differences; Trigonometric identities link cosine and sine of offsets; Multiple frequencies encode both short- and long-range structure; Cosine similarity peaks when positions are identical.",
  },

  {
    id: "cme295-lect2-q14",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe where positional encodings are applied in the original Transformer?",
    options: [
      {
        text: "They are added to token embeddings before entering attention layers.",
        isCorrect: true,
      },
      {
        text: "They have the same dimensionality as token embeddings.",
        isCorrect: true,
      },
      {
        text: "They affect attention indirectly via input representations.",
        isCorrect: true,
      },
      {
        text: "They are applied identically across all attention heads.",
        isCorrect: true,
      },
    ],
    explanation:
      "Original positional encodings are added to inputs, influencing attention only indirectly. Core ideas: they are added to token embeddings before entering attention layers; they have the same dimensionality as token embeddings; they affect attention indirectly via input representations; they are applied identically across all attention heads.",
  },

  {
    id: "cme295-lect2-q15",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the motivation behind Rotary Position Embeddings (RoPE)?",
    options: [
      {
        text: "They inject positional information directly into attention computation.",
        isCorrect: true,
      },
      {
        text: "They encode relative position through query–key interactions.",
        isCorrect: true,
      },
      {
        text: "They avoid learning explicit position embeddings.",
        isCorrect: true,
      },
      { text: "They preserve vector norms under rotation.", isCorrect: true },
    ],
    explanation:
      "RoPE rotates queries and keys so that attention scores naturally depend on relative positions. Core ideas: they inject positional information directly into attention computation; they encode relative position through query–key interactions; they avoid learning explicit position embeddings; they preserve vector norms under rotation.",
  },

  {
    id: "cme295-lect2-q16",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the mathematical structure of RoPE in higher dimensions?",
    options: [
      {
        text: "The embedding dimension is partitioned into 2D blocks.",
        isCorrect: true,
      },
      {
        text: "Each block is rotated independently using a rotation matrix.",
        isCorrect: true,
      },
      {
        text: "Rotation angles vary across dimensions via different frequencies.",
        isCorrect: true,
      },
      {
        text: "The construction generalizes 2D rotations to higher-dimensional spaces.",
        isCorrect: true,
      },
    ],
    explanation:
      "RoPE extends 2D rotations to higher dimensions by applying block-wise rotations with varying frequencies. Core ideas: The embedding dimension is partitioned into 2D blocks; Each block is rotated independently using a rotation matrix; Rotation angles vary across dimensions via different frequencies; The construction generalizes 2D rotations to higher-dimensional spaces.",
  },

  {
    id: "cme295-lect2-q17",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about RoPE-induced attention decay are correct?",
    options: [
      {
        text: "Attention similarity has an upper bound that decays with distance.",
        isCorrect: true,
      },
      {
        text: "The decay is oscillatory due to trigonometric functions.",
        isCorrect: true,
      },
      {
        text: "Nearby tokens are generally more similar than distant ones.",
        isCorrect: true,
      },
      {
        text: "The decay behavior can be analyzed mathematically.",
        isCorrect: true,
      },
    ],
    explanation:
      "RoPE induces a distance-aware similarity structure with long-term decay and oscillations. Core ideas: Attention similarity has an upper bound that decays with distance; The decay is oscillatory due to trigonometric functions; Nearby tokens are generally more similar than distant ones; The decay behavior can be analyzed mathematically.",
  },

  {
    id: "cme295-lect2-q18",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the role of residual connections with normalization?",
    options: [
      {
        text: "They help preserve gradient flow in deep networks.",
        isCorrect: true,
      },
      {
        text: "They combine sublayer outputs with the original input.",
        isCorrect: true,
      },
      {
        text: "They are used together with normalization for stability.",
        isCorrect: true,
      },
      {
        text: "They reduce training degradation as depth increases.",
        isCorrect: true,
      },
    ],
    explanation:
      "Residual connections stabilize deep transformer training and work closely with normalization layers. Core ideas: they help preserve gradient flow in deep networks; they combine sublayer outputs with the original input; they are used together with normalization for stability; they reduce training degradation as depth increases.",
  },

  {
    id: "cme295-lect2-q19",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe differences between post-norm and pre-norm Transformers?",
    options: [
      {
        text: "Post-norm applies normalization after the residual addition.",
        isCorrect: true,
      },
      {
        text: "Pre-norm applies normalization before the sublayer.",
        isCorrect: true,
      },
      {
        text: "Pre-norm improves training stability for deep models.",
        isCorrect: true,
      },
      {
        text: "Modern large models typically prefer pre-norm.",
        isCorrect: true,
      },
    ],
    explanation:
      "Pre-norm Transformers are more stable at scale and are now the standard choice. Core ideas: Post-norm applies normalization after the residual addition; Pre-norm applies normalization before the sublayer; Pre-norm improves training stability for deep models; Modern large models typically prefer pre-norm.",
  },

  {
    id: "cme295-lect2-q20",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements correctly describe RMSNorm?",
    options: [
      {
        text: "It normalizes activations using root mean square.",
        isCorrect: true,
      },
      { text: "It omits the mean subtraction step.", isCorrect: true },
      { text: "It learns a scaling parameter but no bias.", isCorrect: true },
      {
        text: "It reduces parameter count compared to LayerNorm.",
        isCorrect: true,
      },
    ],
    explanation:
      "RMSNorm simplifies normalization while preserving performance and reducing parameters. Core ideas: it normalizes activations using root mean square; it omits the mean subtraction step; it learns a scaling parameter but no bias; it reduces parameter count compared to LayerNorm.",
  },

  {
    id: "cme295-lect2-q21",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe sparse attention mechanisms such as Longformer?",
    options: [
      {
        text: "They reduce quadratic complexity by restricting attention patterns.",
        isCorrect: true,
      },
      {
        text: "They combine local and global attention mechanisms.",
        isCorrect: true,
      },
      {
        text: "They enable longer context lengths at lower cost.",
        isCorrect: true,
      },
      {
        text: "They trade off full connectivity for efficiency.",
        isCorrect: true,
      },
    ],
    explanation:
      "Sparse attention limits interactions to improve scalability while preserving performance. Core ideas: they reduce quadratic complexity by restricting attention patterns; they combine local and global attention mechanisms; they enable longer context lengths at lower cost; they trade off full connectivity for efficiency.",
  },

  {
    id: "cme295-lect2-q22",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly relate sliding window attention to convolutional receptive fields?",
    options: [
      {
        text: "Local attention defines which tokens influence a given token.",
        isCorrect: true,
      },
      {
        text: "Stacking layers expands the effective receptive field.",
        isCorrect: true,
      },
      {
        text: "Information propagates across layers beyond the local window.",
        isCorrect: true,
      },
      {
        text: "The analogy mirrors how CNNs aggregate spatial context.",
        isCorrect: true,
      },
    ],
    explanation:
      "Sliding window attention behaves similarly to receptive fields in convolutional networks. Core ideas: Local attention defines which tokens influence a given token; Stacking layers expands the effective receptive field; Information propagates across layers beyond the local window; The analogy mirrors how CNNs aggregate spatial context.",
  },

  {
    id: "cme295-lect2-q23",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe motivations for sharing key/value projections across heads?",
    options: [
      { text: "It reduces memory usage of the KV cache.", isCorrect: true },
      {
        text: "Keys and values are reused during autoregressive decoding.",
        isCorrect: true,
      },
      {
        text: "Queries benefit from retaining head-specific diversity.",
        isCorrect: true,
      },
      { text: "Sharing improves inference efficiency.", isCorrect: true },
    ],
    explanation:
      "Sharing K/V projections lowers memory and compute costs while keeping expressive queries. Core ideas: it reduces memory usage of the KV cache; Keys and values are reused during autoregressive decoding; Queries benefit from retaining head-specific diversity; Sharing improves inference efficiency.",
  },

  {
    id: "cme295-lect2-q24",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements correctly describe Multi-Query Attention (MQA)?",
    options: [
      {
        text: "All heads share the same key and value projections.",
        isCorrect: true,
      },
      {
        text: "Each head still has its own query projection.",
        isCorrect: true,
      },
      {
        text: "It minimizes memory usage for key/value caches.",
        isCorrect: true,
      },
      { text: "It is an extreme case of projection sharing.", isCorrect: true },
    ],
    explanation:
      "MQA aggressively shares K/V projections to reduce memory while keeping query diversity. Core ideas: All heads share the same key and value projections; Each head still has its own query projection; it minimizes memory usage for key/value caches; it is an extreme case of projection sharing.",
  },

  {
    id: "cme295-lect2-q25",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe Grouped-Query Attention (GQA)?",
    options: [
      {
        text: "It groups heads to share key/value projections.",
        isCorrect: true,
      },
      { text: "It interpolates between MHA and MQA.", isCorrect: true },
      { text: "It balances efficiency and expressivity.", isCorrect: true },
      {
        text: "It is widely used in modern decoder-only LLMs.",
        isCorrect: true,
      },
    ],
    explanation:
      "GQA offers a compromise between full multi-head attention and full K/V sharing. Core ideas: it groups heads to share key/value projections; it interpolates between MHA and MQA; it balances efficiency and expressivity; it is widely used in modern decoder-only LLMs.",
  },

  // ============================================================
  // Q26–Q50: EXACTLY 3 TRUE
  // ============================================================

  {
    id: "cme295-lect2-q26",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about decoder-only Transformers are correct?",
    options: [
      { text: "They remove the encoder entirely.", isCorrect: true },
      { text: "They rely on masked self-attention.", isCorrect: true },
      { text: "They are well suited for text generation.", isCorrect: true },
      {
        text: "They require decoder states to cross-attend to a separately encoded source sequence.",
        isCorrect: false,
      },
    ],
    explanation:
      "Decoder-only models use masked self-attention and do not include cross-attention. Core ideas: they remove the encoder entirely; they rely on masked self-attention; they are well suited for text generation. Common misconceptions: they require decoder states to cross-attend to a separately encoded source sequence.",
  },

  {
    id: "cme295-lect2-q27",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about T5-style models are correct?",
    options: [
      { text: "They use an encoder–decoder architecture.", isCorrect: true },
      { text: "They frame all tasks as text-to-text.", isCorrect: true },
      {
        text: "They use span corruption as a pretraining objective.",
        isCorrect: true,
      },
      {
        text: "They rely exclusively on next-token prediction with a causal decoder objective.",
        isCorrect: false,
      },
    ],
    explanation:
      "T5 reformulates tasks into text-to-text using span corruption rather than pure next-token prediction. Core ideas: they use an encoder–decoder architecture; they frame all tasks as text-to-text; they use span corruption as a pretraining objective. Common misconceptions: they rely exclusively on next-token prediction with a causal decoder objective.",
  },

  {
    id: "cme295-lect2-q28",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about BERT’s bidirectionality are correct?",
    options: [
      {
        text: "Each token can attend to tokens on both sides.",
        isCorrect: true,
      },
      {
        text: "Bidirectionality is enabled by unmasked self-attention.",
        isCorrect: true,
      },
      { text: "BERT lacks causal masking.", isCorrect: true },
      {
        text: "Bidirectionality enables left-to-right autoregressive generation without causal masking.",
        isCorrect: false,
      },
    ],
    explanation:
      "BERT is bidirectional but not suitable for autoregressive generation. Core ideas: Each token can attend to tokens on both sides; Bidirectionality is enabled by unmasked self-attention; BERT lacks causal masking. Common misconceptions: Bidirectionality enables left-to-right autoregressive generation without causal masking.",
  },

  {
    id: "cme295-lect2-q29",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about the [CLS] token in BERT are correct?",
    options: [
      {
        text: "It aggregates information from the entire sequence.",
        isCorrect: true,
      },
      { text: "It is used as input to classification heads.", isCorrect: true },
      {
        text: "It participates in self-attention like any other token.",
        isCorrect: true,
      },
      {
        text: "It is ignored during pretraining and added only when a downstream classifier is attached.",
        isCorrect: false,
      },
    ],
    explanation:
      "The CLS token is fully integrated into attention and used for downstream classification. Core ideas: it aggregates information from the entire sequence; it is used as input to classification heads; it participates in self-attention like any other token. Common misconceptions: it is ignored during pretraining and added only when a downstream classifier is attached.",
  },

  {
    id: "cme295-lect2-q30",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about segment embeddings in BERT are correct?",
    options: [
      {
        text: "They distinguish between sentence A and sentence B.",
        isCorrect: true,
      },
      {
        text: "They are added to token and position embeddings.",
        isCorrect: true,
      },
      { text: "They are learned parameters.", isCorrect: true },
      {
        text: "They encode word order within each sentence rather than marking sentence-pair membership.",
        isCorrect: false,
      },
    ],
    explanation:
      "Segment embeddings identify sentence membership, not token order. Core ideas: they distinguish between sentence A and sentence B; they are added to token and position embeddings; they are learned parameters. Common misconceptions: they encode word order within each sentence rather than marking sentence-pair membership.",
  },

  // ============================================================
  // Q31–Q50 (3 TRUE) – continues seamlessly
  // ============================================================

  {
    id: "cme295-lect2-q31",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements about Masked Language Modeling (MLM) are correct?",
    options: [
      {
        text: "Only a subset of tokens contribute to the MLM loss.",
        isCorrect: true,
      },
      {
        text: "Some masked tokens are replaced with random words.",
        isCorrect: true,
      },
      { text: "Some selected tokens are left unchanged.", isCorrect: true },
      {
        text: "The full sequence is replaced by mask tokens before the encoder sees it.",
        isCorrect: false,
      },
    ],
    explanation:
      "MLM uses a mixture of masking, random replacement, and unchanged tokens. Core ideas: Only a subset of tokens contribute to the MLM loss; Some masked tokens are replaced with random words; Some selected tokens are left unchanged. Common misconceptions: The full sequence is replaced by mask tokens before the encoder sees it.",
  },

  {
    id: "cme295-lect2-q32",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about Next Sentence Prediction (NSP) are correct?",
    options: [
      { text: "It is a binary classification task.", isCorrect: true },
      { text: "It uses sentence pairs as input.", isCorrect: true },
      { text: "It was later shown to be unnecessary.", isCorrect: true },
      {
        text: "It directly improves text generation quality.",
        isCorrect: false,
      },
    ],
    explanation:
      "NSP was later removed in models like RoBERTa with little performance loss. Core ideas: it is a binary classification task; it uses sentence pairs as input; it was later shown to be unnecessary. Common misconceptions: it directly improves text generation quality.",
  },

  {
    id: "cme295-lect2-q33",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about RoBERTa are correct?",
    options: [
      { text: "It removes the NSP objective.", isCorrect: true },
      { text: "It uses dynamic masking across epochs.", isCorrect: true },
      {
        text: "It increases pretraining data size substantially.",
        isCorrect: true,
      },
      {
        text: "It introduces a new attention mechanism rather than changing pretraining data and masking.",
        isCorrect: false,
      },
    ],
    explanation:
      "RoBERTa improves performance through training strategy changes, not architectural ones. Core ideas: it removes the NSP objective; it uses dynamic masking across epochs; it increases pretraining data size substantially. Common misconceptions: it introduces a new attention mechanism rather than changing pretraining data and masking.",
  },

  {
    id: "cme295-lect2-q34",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about DistilBERT are correct?",
    options: [
      { text: "It uses knowledge distillation.", isCorrect: true },
      { text: "It has fewer layers than BERT-base.", isCorrect: true },
      { text: "It retains most of BERT’s performance.", isCorrect: true },
      {
        text: "It increases BERT model size to improve accuracy instead of compressing the teacher model.",
        isCorrect: false,
      },
    ],
    explanation:
      "DistilBERT trades depth for efficiency while preserving performance. Core ideas: it uses knowledge distillation; it has fewer layers than BERT-base; it retains most of BERT’s performance. Common misconceptions: it increases BERT model size to improve accuracy instead of compressing the teacher model.",
  },

  {
    id: "cme295-lect2-q35",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about knowledge distillation are correct?",
    options: [
      {
        text: "It trains a student model to match a teacher’s output distribution.",
        isCorrect: true,
      },
      { text: "It often uses KL divergence as a loss.", isCorrect: true },
      {
        text: "It leverages soft targets rather than hard labels.",
        isCorrect: true,
      },
      {
        text: "It requires task-specific labeled downstream data rather than learning from a teacher distribution.",
        isCorrect: false,
      },
    ],
    explanation:
      "Distillation transfers knowledge via output distributions, not labeled data. Core ideas: it trains a student model to match a teacher’s output distribution; it often uses KL divergence as a loss; it leverages soft targets rather than hard labels. Common misconceptions: it requires task-specific labeled downstream data rather than learning from a teacher distribution.",
  },

  // ============================================================
  // Q51–Q60: EXACTLY 2 TRUE
  // ============================================================

  {
    id: "cme295-lect2-q51",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about encoder-only models are correct?",
    options: [
      {
        text: "They are well suited for classification tasks.",
        isCorrect: true,
      },
      {
        text: "They are not designed to generate text autoregressively.",
        isCorrect: true,
      },
      { text: "They rely on bidirectional self-attention.", isCorrect: true },
      {
        text: "They include masked causal attention for left-to-right generation inside every layer.",
        isCorrect: false,
      },
    ],
    explanation:
      "Encoder-only models are bidirectional and optimized for representation learning. Core ideas: they are well suited for classification tasks; they are not designed to generate text autoregressively; they rely on bidirectional self-attention. Common misconceptions: they include masked causal attention for left-to-right generation inside every layer.",
  },

  {
    id: "cme295-lect2-q52",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about BERT tokenization are correct?",
    options: [
      { text: "It uses WordPiece tokenization.", isCorrect: true },
      {
        text: "It operates as a byte-level tokenizer with a fixed 256-symbol vocabulary.",
        isCorrect: false,
      },
      { text: "Its vocabulary size is around 30k.", isCorrect: true },
      {
        text: "It allows subword splitting for rare words.",
        isCorrect: true,
      },
    ],
    explanation:
      "BERT uses WordPiece with a moderately sized subword vocabulary. Core ideas: it uses WordPiece tokenization; Its vocabulary size is around 30k; it allows subword splitting for rare words. Common misconceptions: it operates as a byte-level tokenizer with a fixed 256-symbol vocabulary.",
  },

  {
    id: "cme295-lect2-q53",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about BERT fine-tuning are correct?",
    options: [
      {
        text: "A task-specific head is added on top of pretrained embeddings.",
        isCorrect: true,
      },
      {
        text: "The entire model can be updated during fine-tuning.",
        isCorrect: true,
      },
      {
        text: "Fine-tuning can require relatively little labeled data.",
        isCorrect: true,
      },
      {
        text: "It replaces the pretrained encoder objective with a new model trained from scratch.",
        isCorrect: false,
      },
    ],
    explanation:
      "Fine-tuning adapts pretrained representations to downstream tasks. Core ideas: A task-specific head is added on top of pretrained embeddings; The entire model can be updated during fine-tuning; Fine-tuning can require relatively little labeled data. Common misconceptions: it replaces the pretrained encoder objective with a new model trained from scratch.",
  },

  {
    id: "cme295-lect2-q54",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statements about attention complexity are correct?",
    options: [
      { text: "Full self-attention has O(n²) complexity.", isCorrect: true },
      {
        text: "Sparse attention reduces worst-case complexity.",
        isCorrect: true,
      },
      {
        text: "Local attention preserves global connectivity in a single layer.",
        isCorrect: false,
      },
      {
        text: "Sliding window attention increases memory usage.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sparse attention reduces complexity but requires multiple layers for global context. Core ideas: Full self-attention has O(n²) complexity; Sparse attention reduces worst-case complexity. Common misconceptions: Local attention preserves global connectivity in a single layer; Sliding window attention increases memory usage.",
  },

  {
    id: "cme295-lect2-q55",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about RMSNorm vs LayerNorm are correct?",
    options: [
      { text: "RMSNorm removes mean subtraction.", isCorrect: true },
      { text: "LayerNorm has fewer parameters.", isCorrect: false },
      {
        text: "RMSNorm keeps performance comparable in practice.",
        isCorrect: true,
      },
      {
        text: "LayerNorm is avoided in Transformers because residual branches already normalize activations.",
        isCorrect: false,
      },
    ],
    explanation:
      "RMSNorm simplifies normalization while retaining similar performance. Core ideas: RMSNorm removes mean subtraction; RMSNorm keeps performance comparable in practice. Common misconceptions: LayerNorm has fewer parameters; LayerNorm is avoided in Transformers because residual branches already normalize activations.",
  },

  {
    id: "cme295-lect2-q56",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about GQA vs MQA are correct?",
    options: [
      { text: "GQA shares K/V projections within groups.", isCorrect: true },
      { text: "MQA shares K/V projections across all heads.", isCorrect: true },
      { text: "MQA increases memory usage.", isCorrect: false },
      { text: "GQA eliminates query diversity.", isCorrect: false },
    ],
    explanation:
      "Both methods reduce K/V redundancy while preserving query diversity. Core ideas: GQA shares K/V projections within groups; MQA shares K/V projections across all heads. Common misconceptions: MQA increases memory usage; GQA eliminates query diversity.",
  },

  {
    id: "cme295-lect2-q57",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements about positional encoding extrapolation are correct?",
    options: [
      {
        text: "Sinusoidal encodings extrapolate naturally to longer sequences.",
        isCorrect: true,
      },
      {
        text: "Learned absolute embeddings extrapolate automatically.",
        isCorrect: false,
      },
      { text: "RoPE supports length extrapolation.", isCorrect: true },
      { text: "T5 relative bias cannot be extended.", isCorrect: false },
    ],
    explanation:
      "Hardcoded and relative methods generalize better to unseen lengths. Core ideas: Sinusoidal encodings extrapolate naturally to longer sequences; RoPE supports length extrapolation. Common misconceptions: Learned absolute embeddings extrapolate automatically; T5 relative bias cannot be extended.",
  },

  {
    id: "cme295-lect2-q58",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about decoder-only models are correct?",
    options: [
      {
        text: "They are easier to scale with next-token prediction.",
        isCorrect: true,
      },
      { text: "They require NSP-style objectives.", isCorrect: false },
      { text: "They dominate modern LLM architectures.", isCorrect: true },
      { text: "They use bidirectional attention.", isCorrect: false },
    ],
    explanation:
      "Decoder-only models scale well with simple objectives and causal attention. Core ideas: they are easier to scale with next-token prediction; they dominate modern LLM architectures. Common misconceptions: they require NSP-style objectives; they use bidirectional attention.",
  },

  {
    id: "cme295-lect2-q59",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about span corruption in T5 are correct?",
    options: [
      { text: "It masks contiguous spans of tokens.", isCorrect: true },
      {
        text: "It uses sentinel tokens to mark masked spans.",
        isCorrect: true,
      },
      { text: "It predicts tokens independently of order.", isCorrect: false },
      { text: "It replaces MLM entirely in BERT.", isCorrect: false },
    ],
    explanation:
      "Span corruption replaces contiguous spans and predicts them sequentially. Core ideas: it masks contiguous spans of tokens; it uses sentinel tokens to mark masked spans. Common misconceptions: it predicts tokens independently of order; it replaces MLM entirely in BERT.",
  },

  {
    id: "cme295-lect2-q60",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about model families are correct?",
    options: [
      {
        text: "Encoder–decoder models are well suited for translation.",
        isCorrect: true,
      },
      {
        text: "Encoder-only models excel at representation learning.",
        isCorrect: true,
      },
      {
        text: "Decoder-only models require cross-attention.",
        isCorrect: false,
      },
      {
        text: "All Transformer variants use identical objectives.",
        isCorrect: false,
      },
    ],
    explanation:
      "Different Transformer families specialize in different task regimes. Core ideas: Encoder–decoder models are well suited for translation; Encoder-only models excel at representation learning. Common misconceptions: Decoder-only models require cross-attention; All Transformer variants use identical objectives.",
  },

  // ============================================================
  // Q61–Q75: EXACTLY 2 TRUE
  // ============================================================

  {
    id: "cme295-lect2-q61",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about causal (masked) self-attention are correct?",
    options: [
      {
        text: "It prevents a token from attending to future tokens.",
        isCorrect: true,
      },
      {
        text: "It is typically used in decoder-only language models for generation.",
        isCorrect: true,
      },
      {
        text: "It makes the model bidirectional during training.",
        isCorrect: false,
      },
      { text: "It requires a separate encoder to work.", isCorrect: false },
    ],
    explanation:
      "Causal masking enforces an autoregressive constraint: each position can only use information from earlier (and itself) positions. Core ideas: it prevents a token from attending to future tokens; it is typically used in decoder-only language models for generation. Common misconceptions: it makes the model bidirectional during training; it requires a separate encoder to work.",
  },

  {
    id: "cme295-lect2-q62",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about the key–value (KV) cache are correct?",
    options: [
      {
        text: "It stores previously computed key and value tensors to speed up decoding.",
        isCorrect: true,
      },
      {
        text: "Its memory cost grows with sequence length and the number of key/value heads.",
        isCorrect: true,
      },
      { text: "It is only useful for encoder-only models.", isCorrect: false },
      {
        text: "It removes the need to compute attention scores.",
        isCorrect: false,
      },
    ],
    explanation:
      "During autoregressive decoding, past keys/values are reused at every step; caching avoids recomputing them and reduces latency. Core ideas: it stores previously computed key and value tensors to speed up decoding; Its memory cost grows with sequence length and the number of key/value heads. Common misconceptions: it is only useful for encoder-only models; it removes the need to compute attention scores.",
  },

  {
    id: "cme295-lect2-q63",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statements about attention complexity are correct?",
    options: [
      {
        text: "Full self-attention over length n has time complexity proportional to n².",
        isCorrect: true,
      },
      {
        text: "Sliding window attention with window size w reduces attention computation to roughly n·w.",
        isCorrect: true,
      },
      {
        text: "Sliding window attention makes each token globally connected within a single layer.",
        isCorrect: false,
      },
      {
        text: "Sparse attention always preserves identical outputs to full attention.",
        isCorrect: false,
      },
    ],
    explanation:
      "Restricting attention patterns reduces computation, but exact equivalence to full attention is not guaranteed, and global connectivity typically requires stacking layers. Core ideas: Full self-attention over length n has time complexity proportional to n²; Sliding window attention with window size w reduces attention computation to roughly n·w. Common misconceptions: Sliding window attention makes each token globally connected within a single layer; Sparse attention always preserves identical outputs to full attention.",
  },

  {
    id: "cme295-lect2-q64",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about rotation matrices used in Rotary Position Embeddings (RoPE) are correct?",
    options: [
      {
        text: "A 2D rotation matrix preserves vector length (norm).",
        isCorrect: true,
      },
      {
        text: "A 2D rotation matrix is built from sine and cosine of an angle.",
        isCorrect: true,
      },
      {
        text: "A rotation matrix changes dot products in a way that removes all positional information.",
        isCorrect: false,
      },
      {
        text: "RoPE requires learning a separate embedding vector for each absolute position index.",
        isCorrect: false,
      },
    ],
    explanation:
      "RoPE uses deterministic rotations (sine/cosine structure) to inject position into query–key interactions while preserving norms. Core ideas: A 2D rotation matrix preserves vector length (norm); A 2D rotation matrix is built from sine and cosine of an angle. Common misconceptions: A rotation matrix changes dot products in a way that removes all positional information; RoPE requires learning a separate embedding vector for each absolute position index.",
  },

  {
    id: "cme295-lect2-q65",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about padding tokens in transformer training are correct?",
    options: [
      {
        text: "Padding helps batch sequences to a fixed length for efficient matrix operations.",
        isCorrect: true,
      },
      {
        text: "Padding tokens are often masked so they do not affect attention or loss.",
        isCorrect: true,
      },
      {
        text: "Padding is required at inference for all models and all deployments.",
        isCorrect: false,
      },
      {
        text: "Padding tokens are used to enforce causal masking.",
        isCorrect: false,
      },
    ],
    explanation:
      "Padding is mainly a batching convenience; models typically mask pads to avoid learning from artificial tokens. Core ideas: Padding helps batch sequences to a fixed length for efficient matrix operations; Padding tokens are often masked so they do not affect attention or loss. Common misconceptions: Padding is required at inference for all models and all deployments; Padding tokens are used to enforce causal masking.",
  },

  {
    id: "cme295-lect2-q66",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about knowledge distillation are correct?",
    options: [
      {
        text: "Distillation often trains the student to match the teacher’s probability distribution, not just the argmax label.",
        isCorrect: true,
      },
      {
        text: "KL divergence is a common loss for matching teacher and student distributions.",
        isCorrect: true,
      },
      {
        text: "Distillation always increases the number of layers in the student.",
        isCorrect: false,
      },
      {
        text: "Distillation requires a decoder-only architecture to work.",
        isCorrect: false,
      },
    ],
    explanation:
      "Distillation is an architecture-agnostic training strategy: compress a model by learning from a stronger teacher’s outputs. Core ideas: Distillation often trains the student to match the teacher’s probability distribution, not just the argmax label; KL divergence is a common loss for matching teacher and student distributions. Common misconceptions: Distillation always increases the number of layers in the student; Distillation requires a decoder-only architecture to work.",
  },

  {
    id: "cme295-lect2-q67",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements about Layer Normalization (LayerNorm) vs Batch Normalization (BatchNorm) are correct?",
    options: [
      {
        text: "LayerNorm normalizes across feature dimensions within a single token’s activation vector.",
        isCorrect: true,
      },
      {
        text: "BatchNorm normalizes using statistics aggregated across the batch dimension.",
        isCorrect: true,
      },
      {
        text: "BatchNorm behaves identically at training time and inference time without any special handling.",
        isCorrect: false,
      },
      {
        text: "LayerNorm requires large batch sizes to work well in transformers.",
        isCorrect: false,
      },
    ],
    explanation:
      "LayerNorm is batch-size independent and stable for sequence models; BatchNorm depends on batch statistics and can introduce train–test behavior differences. Core ideas: LayerNorm normalizes across feature dimensions within a single token’s activation vector; BatchNorm normalizes using statistics aggregated across the batch dimension. Common misconceptions: BatchNorm behaves identically at training time and inference time without any special handling; LayerNorm requires large batch sizes to work well in transformers.",
  },

  {
    id: "cme295-lect2-q68",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about RMSNorm are correct?",
    options: [
      {
        text: "It normalizes by the root mean square of the activation components.",
        isCorrect: true,
      },
      {
        text: "It typically uses a learned scaling parameter (gamma) but no learned bias (beta).",
        isCorrect: true,
      },
      {
        text: "It subtracts the mean of the activation vector as a required step.",
        isCorrect: false,
      },
      {
        text: "It cannot be used with residual connections.",
        isCorrect: false,
      },
    ],
    explanation:
      "RMSNorm is a streamlined alternative to LayerNorm: it rescales using RMS and often omits mean-centering and bias. Core ideas: it normalizes by the root mean square of the activation components; it typically uses a learned scaling parameter (gamma) but no learned bias (beta). Common misconceptions: it subtracts the mean of the activation vector as a required step; it cannot be used with residual connections.",
  },

  {
    id: "cme295-lect2-q69",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about the [SEP] token in BERT-style inputs are correct?",
    options: [
      {
        text: "It is used to separate sentence A and sentence B in paired inputs.",
        isCorrect: true,
      },
      {
        text: "It helps mark boundaries that are useful for tasks like next sentence prediction.",
        isCorrect: true,
      },
      {
        text: "It is the only token used for classification.",
        isCorrect: false,
      },
      { text: "It forces the model to use causal masking.", isCorrect: false },
    ],
    explanation:
      "[SEP] is a structural delimiter; classification is typically done via the [CLS] representation, not [SEP]. Core ideas: it is used to separate sentence A and sentence B in paired inputs; it helps mark boundaries that are useful for tasks like next sentence prediction. Common misconceptions: it is the only token used for classification; it forces the model to use causal masking.",
  },

  {
    id: "cme295-lect2-q70",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about the Masked Language Modeling (MLM) corruption strategy in BERT are correct?",
    options: [
      {
        text: "A portion of selected tokens are replaced by a special [MASK] token.",
        isCorrect: true,
      },
      {
        text: "Some selected tokens are replaced by a random vocabulary token.",
        isCorrect: true,
      },
      {
        text: "Every token in the sequence is selected for masking.",
        isCorrect: false,
      },
      {
        text: "Selected tokens are always replaced; leaving them unchanged is not allowed.",
        isCorrect: false,
      },
    ],
    explanation:
      "BERT-style MLM typically mixes: mask replacement, random replacement, and leaving some selected tokens unchanged. Core ideas: A portion of selected tokens are replaced by a special [MASK] token; Some selected tokens are replaced by a random vocabulary token. Common misconceptions: Every token in the sequence is selected for masking; Selected tokens are always replaced; leaving them unchanged is not allowed.",
  },

  {
    id: "cme295-lect2-q71",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statements about RoPE and relative position are correct?",
    options: [
      {
        text: "After applying RoPE, the query–key dot product can be expressed to depend on relative offsets (m − n) rather than absolute positions alone.",
        isCorrect: true,
      },
      {
        text: "RoPE can be implemented by applying independent 2D rotations to pairs of embedding dimensions.",
        isCorrect: true,
      },
      {
        text: "RoPE encodes position by adding a learned vector to each token embedding at the input.",
        isCorrect: false,
      },
      {
        text: "RoPE requires bucketizing distances into discrete bins, as in T5 relative bias.",
        isCorrect: false,
      },
    ],
    explanation:
      "RoPE is a rotation-based relative method; T5 relative bias is a learned bias lookup over bucketized distances. Core ideas: After applying RoPE, the query–key dot product can be expressed to depend on relative offsets (m − n) rather than absolute positions alone; RoPE can be implemented by applying independent 2D rotations to pairs of embedding dimensions. Common misconceptions: RoPE encodes position by adding a learned vector to each token embedding at the input; RoPE requires bucketizing distances into discrete bins, as in T5 relative bias.",
  },

  {
    id: "cme295-lect2-q72",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about sliding window attention across multiple layers are correct?",
    options: [
      {
        text: "Even with local windows, information can propagate across long distances by stacking layers.",
        isCorrect: true,
      },
      {
        text: "The effective receptive field can expand as depth increases.",
        isCorrect: true,
      },
      {
        text: "A single local-attention layer gives every token direct access to every other token.",
        isCorrect: false,
      },
      {
        text: "Stacking local-attention layers reduces context mixing compared to a single layer.",
        isCorrect: false,
      },
    ],
    explanation:
      "Local attention limits direct connections per layer, but deeper stacks allow signals to travel farther—similar to receptive field growth. Core ideas: Even with local windows, information can propagate across long distances by stacking layers; The effective receptive field can expand as depth increases. Common misconceptions: A single local-attention layer gives every token direct access to every other token; Stacking local-attention layers reduces context mixing compared to a single layer.",
  },

  {
    id: "cme295-lect2-q73",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about tokenizer vocabulary sizes are correct?",
    options: [
      {
        text: "Subword tokenizers commonly have vocabularies on the order of tens of thousands of tokens.",
        isCorrect: true,
      },
      {
        text: "A byte-level vocabulary can be as small as 256 symbols (2^8).",
        isCorrect: true,
      },
      {
        text: "All modern tokenizers must be strictly word-level (no subwords).",
        isCorrect: false,
      },
      {
        text: "A smaller vocabulary always guarantees better model accuracy.",
        isCorrect: false,
      },
    ],
    explanation:
      "Vocabulary size trades off sequence length, representation granularity, and model capacity; byte-level approaches drastically reduce vocab size. Core ideas: Subword tokenizers commonly have vocabularies on the order of tens of thousands of tokens; A byte-level vocabulary can be as small as 256 symbols (2^8). Common misconceptions: All modern tokenizers must be strictly word-level (no subwords); A smaller vocabulary always guarantees better model accuracy.",
  },

  {
    id: "cme295-lect2-q74",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about T5-style span corruption are correct?",
    options: [
      {
        text: "It masks contiguous spans (one or more tokens) rather than masking isolated tokens only.",
        isCorrect: true,
      },
      {
        text: "Sentinel tokens act as placeholders that indicate where spans were removed.",
        isCorrect: true,
      },
      {
        text: "It requires causal masking in the encoder to work.",
        isCorrect: false,
      },
      {
        text: "The decoder output is restricted to a single token per sentinel span.",
        isCorrect: false,
      },
    ],
    explanation:
      "Span corruption replaces spans with sentinel markers; the decoder reconstructs the missing spans as sequences. Core ideas: it masks contiguous spans (one or more tokens) rather than masking isolated tokens only; Sentinel tokens act as placeholders that indicate where spans were removed. Common misconceptions: it requires causal masking in the encoder to work; The decoder output is restricted to a single token per sentinel span.",
  },

  {
    id: "cme295-lect2-q75",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements about pre-norm vs post-norm transformers are correct?",
    options: [
      {
        text: "Pre-norm applies normalization before the attention or feed-forward sublayer.",
        isCorrect: true,
      },
      {
        text: "Pre-norm is often preferred for stability when training deeper transformers.",
        isCorrect: true,
      },
      {
        text: "Post-norm places normalization before every residual addition.",
        isCorrect: false,
      },
      {
        text: "Post-norm eliminates the need for residual connections.",
        isCorrect: false,
      },
    ],
    explanation:
      "Pre-norm improves gradient flow in deep stacks; post-norm normalizes after residual addition and can be less stable at large depth. Core ideas: Pre-norm applies normalization before the attention or feed-forward sublayer; Pre-norm is often preferred for stability when training deeper transformers. Common misconceptions: Post-norm places normalization before every residual addition; Post-norm eliminates the need for residual connections.",
  },

  // ============================================================
  // Q76–Q100: EXACTLY 1 TRUE
  // ============================================================

  {
    id: "cme295-lect2-q76",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statement best describes what the [CLS] token representation is typically used for in BERT?",
    options: [
      {
        text: "It is commonly fed into a classification head for sequence-level prediction.",
        isCorrect: true,
      },
      {
        text: "It is a special token that enforces causal masking in the encoder.",
        isCorrect: false,
      },
      {
        text: "It is only used to separate sentence A from sentence B.",
        isCorrect: false,
      },
      {
        text: "It is ignored by self-attention and only used during fine-tuning.",
        isCorrect: false,
      },
    ],
    explanation:
      "The [CLS] embedding is designed as an aggregate sequence representation and is often used for sentence/sequence classification. Core idea: it is commonly fed into a classification head for sequence-level prediction. Common misconceptions: it is a special token that enforces causal masking in the encoder; it is only used to separate sentence A from sentence B; it is ignored by self-attention and only used during fine-tuning.",
  },

  {
    id: "cme295-lect2-q77",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement correctly distinguishes encoder-only from decoder-only transformers?",
    options: [
      {
        text: "Encoder-only models are typically bidirectional, while decoder-only models are typically causal.",
        isCorrect: true,
      },
      {
        text: "Encoder-only models require cross-attention layers by definition.",
        isCorrect: false,
      },
      {
        text: "Decoder-only models cannot be trained with next-token prediction.",
        isCorrect: false,
      },
      {
        text: "Decoder-only models always include a separate encoder for inputs.",
        isCorrect: false,
      },
    ],
    explanation:
      "Encoder-only models attend both left and right; decoder-only models use causal masking for autoregressive generation. Core idea: Encoder-only models are typically bidirectional, while decoder-only models are typically causal. Common misconceptions: Encoder-only models require cross-attention layers by definition; Decoder-only models cannot be trained with next-token prediction; Decoder-only models always include a separate encoder for inputs.",
  },

  {
    id: "cme295-lect2-q78",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statement about sinusoidal positional encodings is correct?",
    options: [
      {
        text: "They use multiple frequencies so some dimensions vary quickly with position while others vary slowly.",
        isCorrect: true,
      },
      {
        text: "They require learning one trainable embedding vector for every possible position.",
        isCorrect: false,
      },
      {
        text: "They encode position by adding a learned bias term inside the attention softmax.",
        isCorrect: false,
      },
      {
        text: "They prevent extrapolation to longer sequences by construction.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sinusoidal encodings use sine/cosine waves at different frequencies across dimensions to represent positions and can extrapolate. Core idea: they use multiple frequencies so some dimensions vary quickly with position while others vary slowly. Common misconceptions: they require learning one trainable embedding vector for every possible position; they encode position by adding a learned bias term inside the attention softmax; they prevent extrapolation to longer sequences by construction.",
  },

  {
    id: "cme295-lect2-q79",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statement about T5 relative position bias is correct?",
    options: [
      {
        text: "It adds a learnable bias (often via bucketized distances) directly to attention logits.",
        isCorrect: true,
      },
      {
        text: "It rotates queries and keys using deterministic trigonometric matrices.",
        isCorrect: false,
      },
      {
        text: "It encodes position only by modifying token embeddings at the input layer.",
        isCorrect: false,
      },
      {
        text: "It forces attention to be local (sliding window) in every layer.",
        isCorrect: false,
      },
    ],
    explanation:
      "T5-style relative position bias modifies the attention score computation with learned distance-dependent terms. Core idea: it adds a learnable bias (often via bucketized distances) directly to attention logits. Common misconceptions: it rotates queries and keys using deterministic trigonometric matrices; it encodes position only by modifying token embeddings at the input layer; it forces attention to be local (sliding window) in every layer.",
  },

  {
    id: "cme295-lect2-q80",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statement about ALiBi (Attention with Linear Biases) is correct?",
    options: [
      {
        text: "It uses a deterministic linear bias based on relative position distance.",
        isCorrect: true,
      },
      {
        text: "It requires an encoder–decoder architecture to function.",
        isCorrect: false,
      },
      {
        text: "It learns a separate embedding vector for each absolute position.",
        isCorrect: false,
      },
      {
        text: "It replaces the softmax normalization with a sigmoid.",
        isCorrect: false,
      },
    ],
    explanation:
      "ALiBi introduces a simple, fixed linear bias into attention scores to encourage distance-aware behavior and extrapolation. Core idea: it uses a deterministic linear bias based on relative position distance. Common misconceptions: it requires an encoder–decoder architecture to function; it learns a separate embedding vector for each absolute position; it replaces the softmax normalization with a sigmoid.",
  },

  {
    id: "cme295-lect2-q81",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statement about RoPE is correct?",
    options: [
      {
        text: "It injects position by rotating query and key vectors so attention depends on relative offsets.",
        isCorrect: true,
      },
      {
        text: "It injects position by learning an embedding per absolute position and adding it only to values.",
        isCorrect: false,
      },
      {
        text: "It works by masking future tokens to enforce causality.",
        isCorrect: false,
      },
      {
        text: "It encodes relative distance only by discretizing offsets into buckets.",
        isCorrect: false,
      },
    ],
    explanation:
      "RoPE is a rotation-based positional method that encodes relative position into the dot products used by attention. Core idea: it injects position by rotating query and key vectors so attention depends on relative offsets. Common misconceptions: it injects position by learning an embedding per absolute position and adding it only to values; it works by masking future tokens to enforce causality; it encodes relative distance only by discretizing offsets into buckets.",
  },

  {
    id: "cme295-lect2-q82",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statement about Grouped-Query Attention (GQA) is correct?",
    options: [
      {
        text: "It shares key/value projections within groups of heads to reduce KV cache size.",
        isCorrect: true,
      },
      {
        text: "It removes multi-head attention and replaces it with a single head.",
        isCorrect: false,
      },
      {
        text: "It shares query projections across all heads while keeping distinct keys/values.",
        isCorrect: false,
      },
      {
        text: "It requires an encoder and decoder to share attention weights.",
        isCorrect: false,
      },
    ],
    explanation:
      "GQA reduces key/value redundancy (memory) by grouping heads to share K/V while keeping multiple query heads. Core idea: it shares key/value projections within groups of heads to reduce KV cache size. Common misconceptions: it removes multi-head attention and replaces it with a single head; it shares query projections across all heads while keeping distinct keys/values; it requires an encoder and decoder to share attention weights.",
  },

  {
    id: "cme295-lect2-q83",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statement about Multi-Query Attention (MQA) is correct?",
    options: [
      {
        text: "It is the extreme case where all heads share the same key and value projections.",
        isCorrect: true,
      },
      {
        text: "It increases the KV cache memory compared to standard multi-head attention.",
        isCorrect: false,
      },
      {
        text: "It forces all heads to share the same query projection as well.",
        isCorrect: false,
      },
      {
        text: "It can only be used in encoder-only transformers.",
        isCorrect: false,
      },
    ],
    explanation:
      "MQA minimizes K/V memory by sharing them across all heads, while queries typically remain per-head. Core idea: it is the extreme case where all heads share the same key and value projections. Common misconceptions: it increases the KV cache memory compared to standard multi-head attention; it forces all heads to share the same query projection as well; it can only be used in encoder-only transformers.",
  },

  {
    id: "cme295-lect2-q84",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statement about WordPiece tokenization is correct?",
    options: [
      {
        text: "It represents text using subword units learned from data rather than only whole words.",
        isCorrect: true,
      },
      {
        text: "It is a byte-level method with exactly 256 tokens.",
        isCorrect: false,
      },
      {
        text: "It guarantees one token per word in all languages.",
        isCorrect: false,
      },
      {
        text: "It avoids using any merge rules or vocabulary learning.",
        isCorrect: false,
      },
    ],
    explanation:
      "WordPiece is a subword tokenizer: it learns a vocabulary of pieces that can compose many words efficiently. Core idea: it represents text using subword units learned from data rather than only whole words. Common misconceptions: it is a byte-level method with exactly 256 tokens; it guarantees one token per word in all languages; it avoids using any merge rules or vocabulary learning.",
  },

  {
    id: "cme295-lect2-q85",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement about BERT’s Next Sentence Prediction (NSP) objective is correct?",
    options: [
      {
        text: "It is a binary classification task that predicts whether sentence B follows sentence A in the corpus.",
        isCorrect: true,
      },
      {
        text: "It is required for BERT to perform masked language modeling.",
        isCorrect: false,
      },
      {
        text: "It directly trains the model to generate the next sentence autoregressively.",
        isCorrect: false,
      },
      {
        text: "It is implemented by adding causal masking to the encoder.",
        isCorrect: false,
      },
    ],
    explanation:
      "NSP classifies if a pair of sentences is consecutive; it is not an autoregressive generation objective. Core idea: it is a binary classification task that predicts whether sentence B follows sentence A in the corpus. Common misconceptions: it is required for BERT to perform masked language modeling; it directly trains the model to generate the next sentence autoregressively; it is implemented by adding causal masking to the encoder.",
  },

  {
    id: "cme295-lect2-q86",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statement about RoBERTa’s changes relative to BERT is correct?",
    options: [
      {
        text: "It removes NSP and uses training strategy improvements like dynamic masking and more data.",
        isCorrect: true,
      },
      {
        text: "It replaces self-attention with convolutional layers for efficiency.",
        isCorrect: false,
      },
      {
        text: "It requires an encoder–decoder architecture to work.",
        isCorrect: false,
      },
      {
        text: "It enforces causal masking to enable text generation.",
        isCorrect: false,
      },
    ],
    explanation:
      "RoBERTa’s gains primarily come from better pretraining procedures (data, masking, schedule), not a fundamentally new architecture. Core idea: it removes NSP and uses training strategy improvements like dynamic masking and more data. Common misconceptions: it replaces self-attention with convolutional layers for efficiency; it requires an encoder–decoder architecture to work; it enforces causal masking to enable text generation.",
  },

  {
    id: "cme295-lect2-q87",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statement about DistilBERT is correct?",
    options: [
      {
        text: "It compresses BERT using knowledge distillation to keep performance while reducing size/latency.",
        isCorrect: true,
      },
      {
        text: "It expands BERT by doubling the number of layers.",
        isCorrect: false,
      },
      {
        text: "It replaces attention with recurrence to improve scaling.",
        isCorrect: false,
      },
      {
        text: "It can only be trained with labeled downstream datasets.",
        isCorrect: false,
      },
    ],
    explanation:
      "DistilBERT is a smaller student trained to mimic a larger teacher’s behavior (often BERT), improving efficiency. Core idea: it compresses BERT using knowledge distillation to keep performance while reducing size/latency. Common misconceptions: it expands BERT by doubling the number of layers; it replaces attention with recurrence to improve scaling; it can only be trained with labeled downstream datasets.",
  },

  {
    id: "cme295-lect2-q88",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statement about KL divergence in distillation is correct?",
    options: [
      {
        text: "It measures how well the student distribution approximates the teacher distribution.",
        isCorrect: true,
      },
      {
        text: "It is only defined when the teacher outputs hard one-hot labels.",
        isCorrect: false,
      },
      {
        text: "It is the same as mean squared error on logits in all cases.",
        isCorrect: false,
      },
      {
        text: "It eliminates the need for any probability normalization like softmax.",
        isCorrect: false,
      },
    ],
    explanation:
      "KL divergence compares probability distributions; distillation often matches the teacher’s soft probabilities (soft targets). Core idea: it measures how well the student distribution approximates the teacher distribution. Common misconceptions: it is only defined when the teacher outputs hard one-hot labels; it is the same as mean squared error on logits in all cases; it eliminates the need for any probability normalization like softmax.",
  },

  {
    id: "cme295-lect2-q89",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement about the masked language modeling (MLM) intuition is correct?",
    options: [
      {
        text: "Predicting a masked token encourages using both left and right context, yielding bidirectional representations.",
        isCorrect: true,
      },
      {
        text: "MLM forces the model to use only left context to predict the next token.",
        isCorrect: false,
      },
      {
        text: "MLM is a supervised task requiring human labels for each masked token.",
        isCorrect: false,
      },
      {
        text: "MLM makes the attention matrix strictly lower-triangular.",
        isCorrect: false,
      },
    ],
    explanation:
      "MLM is self-supervised and leverages both sides of context because the encoder attention is not causal. Core idea: Predicting a masked token encourages using both left and right context, yielding bidirectional representations. Common misconceptions: MLM forces the model to use only left context to predict the next token; MLM is a supervised task requiring human labels for each masked token; MLM makes the attention matrix strictly lower-triangular.",
  },

  {
    id: "cme295-lect2-q90",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement about teacher forcing in encoder–decoder training is correct?",
    options: [
      {
        text: "It feeds the ground-truth previous tokens to the decoder during training to predict the next output tokens.",
        isCorrect: true,
      },
      {
        text: "It removes the need for an attention mechanism in the decoder.",
        isCorrect: false,
      },
      {
        text: "It means the model only trains on single-token sequences.",
        isCorrect: false,
      },
      { text: "It is only used for encoder-only models.", isCorrect: false },
    ],
    explanation:
      "Teacher forcing stabilizes sequence training by conditioning on the true previous outputs rather than the model’s own sampled outputs. Core idea: it feeds the ground-truth previous tokens to the decoder during training to predict the next output tokens. Common misconceptions: it removes the need for an attention mechanism in the decoder; it means the model only trains on single-token sequences; it is only used for encoder-only models.",
  },

  {
    id: "cme295-lect2-q91",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statement about encoder–decoder transformers is correct?",
    options: [
      {
        text: "They commonly use cross-attention from decoder states to encoder outputs.",
        isCorrect: true,
      },
      {
        text: "They remove the decoder entirely and only keep encoders.",
        isCorrect: false,
      },
      { text: "They cannot be used for translation tasks.", isCorrect: false },
      {
        text: "They require causal masking in the encoder self-attention.",
        isCorrect: false,
      },
    ],
    explanation:
      "Encoder–decoder models (classic transformer) encode a source sequence and decode a target sequence using cross-attention. Core idea: they commonly use cross-attention from decoder states to encoder outputs. Common misconceptions: they remove the decoder entirely and only keep encoders; they cannot be used for translation tasks; they require causal masking in the encoder self-attention.",
  },

  {
    id: "cme295-lect2-q92",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statement about why positional information is ideally reflected inside attention logits is correct?",
    options: [
      {
        text: "Attention logits (query–key scores) directly determine which tokens are treated as similar, so encoding position there is a direct inductive bias.",
        isCorrect: true,
      },
      {
        text: "Adding positional encodings to token embeddings guarantees attention weights become a function only of relative distance.",
        isCorrect: false,
      },
      {
        text: "Position should be encoded only in the value vectors, not in queries or keys.",
        isCorrect: false,
      },
      {
        text: "Positional information is unnecessary because self-attention already has an inherent ordering.",
        isCorrect: false,
      },
    ],
    explanation:
      "Since attention routing is decided by logits, encoding position within that computation can make distance effects more explicit than only adding to inputs. Core idea: Attention logits (query–key scores) directly determine which tokens are treated as similar, so encoding position there is a direct inductive bias. Common misconceptions: Adding positional encodings to token embeddings guarantees attention weights become a function only of relative distance; Position should be encoded only in the value vectors, not in queries or keys; Positional information is unnecessary because self-attention already has an inherent ordering.",
  },

  {
    id: "cme295-lect2-q93",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement about the relationship between local attention and global context is correct?",
    options: [
      {
        text: "Local attention can still yield long-range influence after multiple layers because intermediate tokens relay information.",
        isCorrect: true,
      },
      {
        text: "Local attention prevents any information from traveling beyond the window even with many layers.",
        isCorrect: false,
      },
      {
        text: "Local attention makes complexity worse than full attention for long sequences.",
        isCorrect: false,
      },
      {
        text: "Local attention is identical to adding sinusoidal positional encodings.",
        isCorrect: false,
      },
    ],
    explanation:
      "Stacking layers lets information hop across windows, gradually expanding the effective context. Core idea: Local attention can still yield long-range influence after multiple layers because intermediate tokens relay information. Common misconceptions: Local attention prevents any information from traveling beyond the window even with many layers; Local attention makes complexity worse than full attention for long sequences; Local attention is identical to adding sinusoidal positional encodings.",
  },

  {
    id: "cme295-lect2-q94",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statement about segment embeddings in BERT is correct?",
    options: [
      {
        text: "They indicate whether a token belongs to sentence A or sentence B in paired inputs.",
        isCorrect: true,
      },
      {
        text: "They uniquely encode each token’s absolute position index.",
        isCorrect: false,
      },
      {
        text: "They replace the need for positional embeddings entirely.",
        isCorrect: false,
      },
      {
        text: "They are fixed sinusoidal vectors, not learned parameters.",
        isCorrect: false,
      },
    ],
    explanation:
      "Segment embeddings are a learned ‘sentence ID’ signal added to token representations to help with paired-sentence tasks. Core idea: they indicate whether a token belongs to sentence A or sentence B in paired inputs. Common misconceptions: they uniquely encode each token’s absolute position index; they replace the need for positional embeddings entirely; they are fixed sinusoidal vectors, not learned parameters.",
  },

  {
    id: "cme295-lect2-q95",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statement about why keys/values are prime targets for sharing (MQA/GQA) is correct?",
    options: [
      {
        text: "During autoregressive decoding, past keys/values are reused at every step, so reducing their head count shrinks the KV cache significantly.",
        isCorrect: true,
      },
      {
        text: "Queries are reused across decoding steps in exactly the same way as keys/values, so sharing queries yields the biggest memory savings.",
        isCorrect: false,
      },
      {
        text: "Sharing keys/values forces attention to become local rather than global.",
        isCorrect: false,
      },
      {
        text: "Sharing keys/values removes the need for output projection after concatenating heads.",
        isCorrect: false,
      },
    ],
    explanation:
      "Caching makes K/V memory the bottleneck at long contexts; sharing K/V reduces cache size while keeping multi-head query diversity. Core idea: During autoregressive decoding, past keys/values are reused at every step, so reducing their head count shrinks the KV cache significantly. Common misconceptions: Queries are reused across decoding steps in exactly the same way as keys/values, so sharing queries yields the biggest memory savings; Sharing keys/values forces attention to become local rather than global; Sharing keys/values removes the need for output projection after concatenating heads.",
  },

  {
    id: "cme295-lect2-q96",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement about why decoder-only models became dominant in large language modeling is correct?",
    options: [
      {
        text: "They scale well with a simple next-token prediction objective aligned with text generation.",
        isCorrect: true,
      },
      {
        text: "They require span corruption to work effectively.",
        isCorrect: false,
      },
      {
        text: "They are inherently better at bidirectional sentence understanding than encoder-only models.",
        isCorrect: false,
      },
      {
        text: "They eliminate attention complexity by removing the softmax.",
        isCorrect: false,
      },
    ],
    explanation:
      "Decoder-only training on next-token prediction is simple, scalable, and directly matches many generation-style applications. Core idea: they scale well with a simple next-token prediction objective aligned with text generation. Common misconceptions: they require span corruption to work effectively; they are inherently better at bidirectional sentence understanding than encoder-only models; they eliminate attention complexity by removing the softmax.",
  },

  {
    id: "cme295-lect2-q97",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statement about “cased” vs “uncased” checkpoints is correct?",
    options: [
      {
        text: "A cased model preserves capitalization distinctions (e.g., 'US' vs 'us'), while an uncased model typically lowercases inputs.",
        isCorrect: true,
      },
      {
        text: "A cased model always has a byte-level tokenizer, while an uncased model always uses WordPiece.",
        isCorrect: false,
      },
      {
        text: "Casing changes the transformer architecture (encoder vs decoder).",
        isCorrect: false,
      },
      {
        text: "Uncased models cannot be fine-tuned on classification tasks.",
        isCorrect: false,
      },
    ],
    explanation:
      "Casing is a data/tokenization choice, not an architectural constraint; it can matter depending on downstream tasks. Core idea: A cased model preserves capitalization distinctions (e.g., 'US' vs 'us'), while an uncased model typically lowercases inputs. Common misconceptions: A cased model always has a byte-level tokenizer, while an uncased model always uses WordPiece; Casing changes the transformer architecture (encoder vs decoder); Uncased models cannot be fine-tuned on classification tasks.",
  },

  {
    id: "cme295-lect2-q98",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statement about why normalization helps optimization is correct?",
    options: [
      {
        text: "It reduces extreme activation scale variation across layers, improving stability and convergence during gradient-based training.",
        isCorrect: true,
      },
      {
        text: "It guarantees the model cannot overfit by constraining all parameters.",
        isCorrect: false,
      },
      {
        text: "It removes the need for non-linear activations in feed-forward networks.",
        isCorrect: false,
      },
      {
        text: "It makes attention complexity linear in sequence length.",
        isCorrect: false,
      },
    ],
    explanation:
      "Normalization primarily stabilizes the scale of activations/gradients (often discussed as addressing internal covariate shift), aiding training dynamics. Core idea: it reduces extreme activation scale variation across layers, improving stability and convergence during gradient-based training. Common misconceptions: it guarantees the model cannot overfit by constraining all parameters; it removes the need for non-linear activations in feed-forward networks; it makes attention complexity linear in sequence length.",
  },

  {
    id: "cme295-lect2-q99",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement about why MLM and NSP were attractive pretraining tasks is correct?",
    options: [
      {
        text: "They can be constructed from unlabeled corpora using self-supervision (masking tokens and forming sentence pairs).",
        isCorrect: true,
      },
      {
        text: "They require manual human labeling of the correct token for each masked position.",
        isCorrect: false,
      },
      {
        text: "They directly train the model to generate long-form text autoregressively.",
        isCorrect: false,
      },
      { text: "They eliminate the need for tokenization.", isCorrect: false },
    ],
    explanation:
      "MLM and NSP are self-supervised objectives: the “labels” come from the text itself (original tokens and known sentence adjacency). Core idea: they can be constructed from unlabeled corpora using self-supervision (masking tokens and forming sentence pairs). Common misconceptions: they require manual human labeling of the correct token for each masked position; they directly train the model to generate long-form text autoregressively; they eliminate the need for tokenization.",
  },

  {
    id: "cme295-lect2-q100",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statement about why sinusoidal positional encodings can generalize to longer lengths is correct?",
    options: [
      {
        text: "They are defined by a fixed analytic function of position, so you can compute embeddings for unseen positions without learning new parameters.",
        isCorrect: true,
      },
      {
        text: "They store a separate trainable vector for every possible position up to infinity.",
        isCorrect: false,
      },
      {
        text: "They rely on bucketizing distances into a finite set of learned bins.",
        isCorrect: false,
      },
      {
        text: "They require the model to be encoder–decoder rather than decoder-only.",
        isCorrect: false,
      },
    ],
    explanation:
      "Because sinusoidal encodings are formula-based, you can compute them for positions beyond those seen during training. Core idea: they are defined by a fixed analytic function of position, so you can compute embeddings for unseen positions without learning new parameters. Common misconceptions: they store a separate trainable vector for every possible position up to infinity; they rely on bucketizing distances into a finite set of learned bins; they require the model to be encoder–decoder rather than decoder-only.",
  },
];
