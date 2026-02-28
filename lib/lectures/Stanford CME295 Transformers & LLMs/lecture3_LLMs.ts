import { Question } from "../../quiz";

export const stanfordCME295Lecture3LLMsQuestions: Question[] = [
  // ============================================================
  // Lecture 3 – Large Language Models (LLMs), MoE, Generation,
  // Prompting, Inference Optimizations
  // Q1–Q35 (first batch)
  // ============================================================

  // ============================================================
  // Q1–Q9: ALL TRUE (≈9 questions)
  // ============================================================

  {
    id: "cme295-lect3-q01",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements correctly describe what a language model does?",
    options: [
      {
        text: "It assigns probabilities to sequences of tokens.",
        isCorrect: true,
      },
      {
        text: "It models the likelihood of the next token given previous tokens.",
        isCorrect: true,
      },
      {
        text: "Its predictions are based on a learned statistical or neural representation.",
        isCorrect: true,
      },
      {
        text: "It can be used to generate text by repeatedly predicting next tokens.",
        isCorrect: true,
      },
    ],
    explanation:
      "A language model estimates probabilities over token sequences. Generation emerges from repeatedly sampling or selecting the next token conditioned on the previously generated ones.",
  },

  {
    id: "cme295-lect3-q02",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly characterize modern large language models (LLMs)?",
    options: [
      {
        text: "They typically contain billions or more parameters.",
        isCorrect: true,
      },
      {
        text: "They are trained on extremely large datasets measured in tokens.",
        isCorrect: true,
      },
      {
        text: "They require substantial computational resources for training.",
        isCorrect: true,
      },
      {
        text: "They are commonly implemented using Transformer architectures.",
        isCorrect: true,
      },
    ],
    explanation:
      "LLMs are defined not only by being language models, but also by their scale in parameters, data, and compute. Transformers have become the dominant architecture enabling this scaling.",
  },

  {
    id: "cme295-lect3-q03",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe decoder-only Transformer models?",
    options: [
      {
        text: "They use masked self-attention to prevent access to future tokens.",
        isCorrect: true,
      },
      { text: "They predict tokens autoregressively.", isCorrect: true },
      {
        text: "They remove the encoder and cross-attention components.",
        isCorrect: true,
      },
      {
        text: "They are commonly used for text-to-text generation tasks.",
        isCorrect: true,
      },
    ],
    explanation:
      "Decoder-only models rely on masked self-attention and autoregressive prediction. By removing the encoder, they simplify the architecture while remaining powerful for generation tasks.",
  },

  {
    id: "cme295-lect3-q04",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe Mixture-of-Experts (MoE) models?",
    options: [
      { text: "They consist of multiple expert subnetworks.", isCorrect: true },
      {
        text: "A gating or routing mechanism selects which experts to use.",
        isCorrect: true,
      },
      {
        text: "They aim to increase model capacity without activating all parameters.",
        isCorrect: true,
      },
      {
        text: "They can reduce inference compute compared to dense models.",
        isCorrect: true,
      },
    ],
    explanation:
      "MoE models increase total parameter count while activating only a subset per input. This allows higher capacity models without proportionally increasing computation at inference.",
  },

  {
    id: "cme295-lect3-q05",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements correctly describe sparse MoE models?",
    options: [
      {
        text: "Only a subset of experts is activated for each input.",
        isCorrect: true,
      },
      {
        text: "The number of active experts is typically controlled by a hyperparameter K.",
        isCorrect: true,
      },
      {
        text: "They are designed to save compute during forward passes.",
        isCorrect: true,
      },
      {
        text: "They rely on a routing mechanism to select experts.",
        isCorrect: true,
      },
    ],
    explanation:
      "Sparse MoEs activate only the top-K experts chosen by a router. This design keeps inference cost manageable while still benefiting from a large pool of parameters.",
  },

  {
    id: "cme295-lect3-q06",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the role of FLOPs in LLM discussions?",
    options: [
      {
        text: "FLOPs count floating point operations such as additions and multiplications.",
        isCorrect: true,
      },
      {
        text: "They are used as a proxy for computational cost.",
        isCorrect: true,
      },
      {
        text: "They depend on model architecture and input length.",
        isCorrect: true,
      },
      {
        text: "They are commonly used to compare dense and sparse models.",
        isCorrect: true,
      },
    ],
    explanation:
      "FLOPs provide a hardware-agnostic way to estimate computational workload. Architectural choices like MoE versus dense layers directly affect FLOPs.",
  },

  {
    id: "cme295-lect3-q07",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe next-token prediction in LLMs?",
    options: [
      {
        text: "The model outputs a probability distribution over the vocabulary.",
        isCorrect: true,
      },
      {
        text: "The distribution is typically produced using a softmax layer.",
        isCorrect: true,
      },
      {
        text: "Each decoding step conditions on previously generated tokens.",
        isCorrect: true,
      },
      {
        text: "The process is repeated until a stopping condition is met.",
        isCorrect: true,
      },
    ],
    explanation:
      "At each step, the model predicts a probability distribution for the next token. Generation proceeds autoregressively until an end-of-sequence token or another stopping rule is reached.",
  },

  {
    id: "cme295-lect3-q08",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe temperature in token sampling?",
    options: [
      {
        text: "Temperature rescales logits before the softmax.",
        isCorrect: true,
      },
      {
        text: "Lower temperatures make the distribution more peaked.",
        isCorrect: true,
      },
      {
        text: "Higher temperatures make the distribution more uniform.",
        isCorrect: true,
      },
      {
        text: "Temperature affects diversity of generated text.",
        isCorrect: true,
      },
    ],
    explanation:
      "Temperature modifies how sharply probabilities are distributed. Lower values favor high-probability tokens, while higher values encourage exploration and diversity.",
  },

  {
    id: "cme295-lect3-q09",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements correctly describe in-context learning?",
    options: [
      {
        text: "It allows models to adapt behavior without changing weights.",
        isCorrect: true,
      },
      {
        text: "It relies on information provided in the prompt context.",
        isCorrect: true,
      },
      {
        text: "Few-shot learning provides example input–output pairs.",
        isCorrect: true,
      },
      {
        text: "Zero-shot learning uses instructions without examples.",
        isCorrect: true,
      },
    ],
    explanation:
      "In-context learning leverages the prompt itself to steer behavior. The model’s parameters remain fixed, but performance can change dramatically based on prompt design.",
  },

  // ============================================================
  // Q10–Q18: THREE TRUE
  // ============================================================

  {
    id: "cme295-lect3-q10",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements about encoder-only models are correct?",
    options: [
      {
        text: "They are typically used for representation learning tasks.",
        isCorrect: true,
      },
      {
        text: "They output contextual embeddings for input tokens.",
        isCorrect: true,
      },
      {
        text: "They commonly use a special classification token for downstream tasks.",
        isCorrect: true,
      },
      {
        text: "They are primarily designed for autoregressive text generation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Encoder-only models such as BERT focus on producing rich representations rather than generating text. Autoregressive generation is instead characteristic of decoder-only models.",
  },

  {
    id: "cme295-lect3-q11",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about greedy decoding are correct?",
    options: [
      {
        text: "It selects the token with the highest probability at each step.",
        isCorrect: true,
      },
      {
        text: "It is deterministic given fixed model outputs.",
        isCorrect: true,
      },
      {
        text: "It can lead to locally optimal but globally suboptimal sequences.",
        isCorrect: true,
      },
      { text: "It maximizes sequence diversity by design.", isCorrect: false },
    ],
    explanation:
      "Greedy decoding is simple and fast but often produces repetitive or suboptimal text. Because it commits early, it may miss better sequences that require short-term sacrifices.",
  },

  {
    id: "cme295-lect3-q12",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about beam search are correct?",
    options: [
      {
        text: "It keeps multiple candidate sequences during generation.",
        isCorrect: true,
      },
      {
        text: "It aims to approximate globally high-probability sequences.",
        isCorrect: true,
      },
      {
        text: "It is commonly used in tasks like machine translation.",
        isCorrect: true,
      },
      {
        text: "It always produces highly diverse and creative outputs.",
        isCorrect: false,
      },
    ],
    explanation:
      "Beam search tracks several hypotheses to improve likelihood, but it often reduces diversity. This makes it suitable for structured tasks rather than creative generation.",
  },

  {
    id: "cme295-lect3-q13",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about top-k sampling are correct?",
    options: [
      {
        text: "It restricts sampling to the k most probable tokens.",
        isCorrect: true,
      },
      {
        text: "It prevents extremely low-probability tokens from being sampled.",
        isCorrect: true,
      },
      { text: "It introduces stochasticity into generation.", isCorrect: true },
      {
        text: "It guarantees selection of the most probable token.",
        isCorrect: false,
      },
    ],
    explanation:
      "Top-k sampling balances randomness and control by limiting the candidate set. The final choice remains stochastic within the selected top-k tokens.",
  },

  {
    id: "cme295-lect3-q14",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about routing collapse in MoE models are correct?",
    options: [
      {
        text: "It occurs when only a few experts receive most inputs.",
        isCorrect: true,
      },
      { text: "It can reduce effective model capacity.", isCorrect: true },
      {
        text: "Auxiliary losses can encourage more balanced expert usage.",
        isCorrect: true,
      },
      { text: "It is beneficial for model generalization.", isCorrect: false },
    ],
    explanation:
      "Routing collapse limits the benefits of MoE by underutilizing experts. Regularization techniques are used to encourage more uniform routing.",
  },

  {
    id: "cme295-lect3-q15",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about expert routing in Transformers are correct?",
    options: [
      {
        text: "Routing decisions can be made at the token level.",
        isCorrect: true,
      },
      {
        text: "Different layers may route tokens to different experts.",
        isCorrect: true,
      },
      { text: "The router is typically a learned function.", isCorrect: true },
      {
        text: "Routing decisions are fixed and not trainable.",
        isCorrect: false,
      },
    ],
    explanation:
      "MoE routing is dynamic and learned. Tokens may be sent to different experts depending on layer and context, increasing expressiveness.",
  },

  {
    id: "cme295-lect3-q16",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about guided decoding are correct?",
    options: [
      {
        text: "It constrains which tokens are allowed during generation.",
        isCorrect: true,
      },
      {
        text: "It can enforce structured output formats such as JSON.",
        isCorrect: true,
      },
      { text: "It filters invalid next-token choices.", isCorrect: true },
      { text: "It requires retraining the language model.", isCorrect: false },
    ],
    explanation:
      "Guided decoding operates at inference time by restricting token choices. It does not modify model weights but controls valid generation paths.",
  },

  {
    id: "cme295-lect3-q17",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about context length are correct?",
    options: [
      {
        text: "It refers to the number of tokens a model can process at once.",
        isCorrect: true,
      },
      {
        text: "It is also called context window or context size.",
        isCorrect: true,
      },
      {
        text: "Increasing it always improves model accuracy.",
        isCorrect: false,
      },
      {
        text: "It affects computational cost of self-attention.",
        isCorrect: true,
      },
    ],
    explanation:
      "Longer context windows allow more information but increase cost and may suffer from issues like context rot. More context is not always better.",
  },

  {
    id: "cme295-lect3-q18",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements about chain-of-thought prompting are correct?",
    options: [
      {
        text: "It encourages models to produce intermediate reasoning steps.",
        isCorrect: true,
      },
      {
        text: "It often improves performance on reasoning tasks.",
        isCorrect: true,
      },
      { text: "It increases the number of generated tokens.", isCorrect: true },
      { text: "It always reduces inference latency.", isCorrect: false },
    ],
    explanation:
      "Chain-of-thought improves reasoning by making intermediate steps explicit, but it increases token count and therefore latency and cost.",
  },

  // ============================================================
  // Q19–Q27: TWO TRUE
  // ============================================================

  {
    id: "cme295-lect3-q19",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements about encoder–decoder models are correct?",
    options: [
      { text: "They use both an encoder and a decoder.", isCorrect: true },
      {
        text: "They are well suited for sequence-to-sequence tasks.",
        isCorrect: true,
      },
      {
        text: "They rely exclusively on masked self-attention.",
        isCorrect: false,
      },
      { text: "They cannot be trained on text data.", isCorrect: false },
    ],
    explanation:
      "Encoder–decoder models combine bidirectional encoding with autoregressive decoding. Masked self-attention is only used in the decoder, not the encoder.",
  },

  {
    id: "cme295-lect3-q20",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about top-p (nucleus) sampling are correct?",
    options: [
      {
        text: "It samples from the smallest set of tokens whose cumulative probability exceeds p.",
        isCorrect: true,
      },
      {
        text: "It adapts the candidate set size dynamically.",
        isCorrect: true,
      },
      { text: "It always samples exactly p tokens.", isCorrect: false },
      { text: "It removes randomness from generation.", isCorrect: false },
    ],
    explanation:
      "Top-p sampling selects a variable number of tokens based on cumulative probability. This allows flexibility while maintaining stochasticity.",
  },

  {
    id: "cme295-lect3-q21",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about softmax are correct?",
    options: [
      {
        text: "It converts logits into a probability distribution.",
        isCorrect: true,
      },
      { text: "Its outputs sum to one.", isCorrect: true },
      { text: "It is independent of temperature scaling.", isCorrect: false },
      { text: "It is only used during training.", isCorrect: false },
    ],
    explanation:
      "Softmax normalizes logits into probabilities and is affected by temperature. It is used during both training and inference.",
  },

  {
    id: "cme295-lect3-q22",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements about self-consistency in prompting are correct?",
    options: [
      {
        text: "It aggregates answers from multiple sampled reasoning paths.",
        isCorrect: true,
      },
      { text: "It can improve robustness of final answers.", isCorrect: true },
      { text: "It requires modifying model weights.", isCorrect: false },
      { text: "It reduces inference cost.", isCorrect: false },
    ],
    explanation:
      "Self-consistency relies on multiple generations and majority voting. While it improves accuracy, it increases inference cost rather than reducing it.",
  },

  {
    id: "cme295-lect3-q23",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about routing probability in MoE models are correct?",
    options: [
      {
        text: "It can be produced using a softmax over experts.",
        isCorrect: true,
      },
      {
        text: "It indicates how likely an expert is to be selected.",
        isCorrect: true,
      },
      { text: "It is always uniform across experts.", isCorrect: false },
      { text: "It is unrelated to training objectives.", isCorrect: false },
    ],
    explanation:
      "Routing probabilities come from a learned router and can be uneven. Training objectives often include terms to encourage balanced expert usage.",
  },

  {
    id: "cme295-lect3-q24",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about sample efficiency of MoE models are correct?",
    options: [
      {
        text: "They can reach strong performance with fewer training steps.",
        isCorrect: true,
      },
      {
        text: "They increase total parameter count without proportional compute increase.",
        isCorrect: true,
      },
      { text: "They always outperform dense models.", isCorrect: false },
      { text: "They eliminate the need for large datasets.", isCorrect: false },
    ],
    explanation:
      "MoE models can be more sample efficient due to higher capacity, but they still require large datasets and careful training to outperform dense models.",
  },

  {
    id: "cme295-lect3-q25",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements about logits are correct?",
    options: [
      {
        text: "They are raw scores output by a model before softmax.",
        isCorrect: true,
      },
      { text: "They can take any real value.", isCorrect: true },
      { text: "They are already probabilities.", isCorrect: false },
      { text: "They must sum to one.", isCorrect: false },
    ],
    explanation:
      "Logits are unnormalized scores. Softmax transforms them into probabilities that sum to one.",
  },

  {
    id: "cme295-lect3-q26",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about determinism in LLM inference are correct?",
    options: [
      {
        text: "The Transformer computations are deterministic given fixed inputs.",
        isCorrect: true,
      },
      { text: "Sampling introduces nondeterminism.", isCorrect: true },
      {
        text: "Temperature zero guarantees identical outputs in practice.",
        isCorrect: false,
      },
      {
        text: "Hardware effects can introduce nondeterminism.",
        isCorrect: false,
      },
    ],
    explanation:
      "While model computations are deterministic, sampling and hardware-level effects can introduce variability. Even temperature zero may not be perfectly deterministic in practice.",
  },

  {
    id: "cme295-lect3-q27",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about expert placement in Transformers are correct?",
    options: [
      {
        text: "Experts are commonly placed in the feed-forward network block.",
        isCorrect: true,
      },
      {
        text: "This is because the feed-forward block has many parameters.",
        isCorrect: true,
      },
      { text: "Experts replace the attention mechanism.", isCorrect: false },
      {
        text: "Experts remove the need for normalization layers.",
        isCorrect: false,
      },
    ],
    explanation:
      "The feed-forward network dominates parameter count, making it a natural location for MoE layers. Attention and normalization layers remain unchanged.",
  },

  // ============================================================
  // Q28–Q35: ONE TRUE
  // ============================================================

  {
    id: "cme295-lect3-q28",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statement best defines a large language model (LLM)?",
    options: [
      {
        text: "A small neural network trained on labeled data only.",
        isCorrect: false,
      },
      {
        text: "A language model with large parameter count, data, and compute.",
        isCorrect: true,
      },
      { text: "Any model that produces embeddings.", isCorrect: false },
      { text: "A rule-based text generation system.", isCorrect: false },
    ],
    explanation:
      "LLMs are defined by scale in parameters, data, and compute, not merely by producing embeddings or using rules.",
  },

  {
    id: "cme295-lect3-q29",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statement correctly describes greedy decoding?",
    options: [
      {
        text: "It samples from a truncated probability distribution.",
        isCorrect: false,
      },
      {
        text: "It selects the highest-probability token at each step.",
        isCorrect: true,
      },
      { text: "It maintains multiple candidate sequences.", isCorrect: false },
      {
        text: "It requires auxiliary losses during training.",
        isCorrect: false,
      },
    ],
    explanation:
      "Greedy decoding always chooses the most probable token, making it simple but often suboptimal.",
  },

  {
    id: "cme295-lect3-q30",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statement correctly explains why beam search can prefer shorter sequences?",
    options: [
      {
        text: "Probabilities greater than one accumulate with length.",
        isCorrect: false,
      },
      {
        text: "Multiplying probabilities less than one reduces total sequence probability.",
        isCorrect: true,
      },
      { text: "Beam search ignores end-of-sequence tokens.", isCorrect: false },
      {
        text: "Beam search uses temperature scaling by default.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sequence probabilities are products of token probabilities, which shrink with length. Length normalization is often added to counteract this bias.",
  },

  {
    id: "cme295-lect3-q31",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statement correctly describes routing collapse?",
    options: [
      { text: "All experts are used equally at all times.", isCorrect: false },
      {
        text: "Only a small subset of experts dominates routing decisions.",
        isCorrect: true,
      },
      { text: "Routing becomes random and untrainable.", isCorrect: false },
      { text: "The model switches to dense computation.", isCorrect: false },
    ],
    explanation:
      "Routing collapse occurs when the router repeatedly selects the same experts, reducing the benefit of having multiple experts.",
  },

  {
    id: "cme295-lect3-q32",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statement best describes top-p sampling?",
    options: [
      {
        text: "It samples only the single most likely token.",
        isCorrect: false,
      },
      {
        text: "It samples from a dynamically sized set based on cumulative probability.",
        isCorrect: true,
      },
      { text: "It removes all randomness from decoding.", isCorrect: false },
      { text: "It requires beam search to function.", isCorrect: false },
    ],
    explanation:
      "Top-p sampling chooses from the smallest set of tokens whose cumulative probability exceeds p, allowing adaptive control of diversity.",
  },

  {
    id: "cme295-lect3-q33",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement correctly captures the purpose of auxiliary MoE losses?",
    options: [
      { text: "They increase vocabulary size.", isCorrect: false },
      { text: "They encourage balanced expert utilization.", isCorrect: true },
      {
        text: "They replace the main language modeling loss.",
        isCorrect: false,
      },
      {
        text: "They eliminate the need for routing networks.",
        isCorrect: false,
      },
    ],
    explanation:
      "Auxiliary losses are added to guide routing behavior, helping prevent collapse and improving overall expert usage.",
  },

  {
    id: "cme295-lect3-q34",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statement best describes context rot?",
    options: [
      {
        text: "Models improve retrieval accuracy with longer context.",
        isCorrect: false,
      },
      {
        text: "Models may struggle to retrieve relevant information as context grows.",
        isCorrect: true,
      },
      {
        text: "Context rot is caused by overfitting during training.",
        isCorrect: false,
      },
      { text: "It only occurs in encoder-only models.", isCorrect: false },
    ],
    explanation:
      "Context rot refers to degradation in effective information use as context length increases, especially in the presence of distractors.",
  },

  {
    id: "cme295-lect3-q35",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statement best describes zero-shot prompting?",
    options: [
      { text: "The model is fine-tuned on new examples.", isCorrect: false },
      {
        text: "The model is given instructions without examples.",
        isCorrect: true,
      },
      {
        text: "The model updates its weights at inference time.",
        isCorrect: false,
      },
      { text: "The model requires labeled demonstrations.", isCorrect: false },
    ],
    explanation:
      "Zero-shot prompting relies solely on instructions and the model’s pre-trained knowledge, without providing example input–output pairs.",
  },

  // ============================================================
  // Q36–Q70 (second batch)
  // ============================================================

  // ============================================================
  // Q36–Q44: ALL TRUE
  // ============================================================

  {
    id: "cme295-lect3-q36",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe Key–Value (KV) caching in Transformer inference?",
    options: [
      {
        text: "It stores key and value tensors from previous tokens.",
        isCorrect: true,
      },
      {
        text: "It avoids recomputing attention components for past tokens.",
        isCorrect: true,
      },
      {
        text: "It reduces redundant computation during autoregressive decoding.",
        isCorrect: true,
      },
      {
        text: "It is primarily used during inference rather than training.",
        isCorrect: true,
      },
    ],
    explanation:
      "KV caching reuses previously computed keys and values so that each new token only computes its own query, key, and value. This significantly reduces inference cost for long sequences.",
  },

  {
    id: "cme295-lect3-q37",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe why self-attention becomes expensive with long context lengths?",
    options: [
      {
        text: "Each new token must attend to all previous tokens.",
        isCorrect: true,
      },
      {
        text: "Attention computation scales quadratically with sequence length.",
        isCorrect: true,
      },
      {
        text: "Memory requirements grow with stored key–value tensors.",
        isCorrect: true,
      },
      {
        text: "Longer contexts increase both compute and memory pressure.",
        isCorrect: true,
      },
    ],
    explanation:
      "Self-attention requires interactions between tokens, which leads to quadratic scaling. KV caching mitigates recomputation but memory usage still grows with context length.",
  },

  {
    id: "cme295-lect3-q38",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe Grouped Query Attention (GQA)?",
    options: [
      {
        text: "Multiple query heads can share the same key and value heads.",
        isCorrect: true,
      },
      { text: "It reduces memory usage of the KV cache.", isCorrect: true },
      {
        text: "It lies between multi-head attention and multi-query attention.",
        isCorrect: true,
      },
      {
        text: "It is commonly used in modern large language models.",
        isCorrect: true,
      },
    ],
    explanation:
      "GQA groups queries to share keys and values, reducing memory and compute cost. It offers a compromise between full multi-head attention and the more extreme multi-query attention.",
  },

  {
    id: "cme295-lect3-q39",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements correctly describe Multi-Query Attention (MQA)?",
    options: [
      {
        text: "All query heads share a single set of key and value heads.",
        isCorrect: true,
      },
      {
        text: "It significantly reduces KV cache memory usage.",
        isCorrect: true,
      },
      {
        text: "It trades some expressiveness for efficiency.",
        isCorrect: true,
      },
      {
        text: "It can improve inference speed for long contexts.",
        isCorrect: true,
      },
    ],
    explanation:
      "MQA uses one shared key and value representation across all query heads. This dramatically reduces memory footprint, especially during inference, at the cost of reduced modeling flexibility.",
  },

  {
    id: "cme295-lect3-q40",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe PagedAttention-style memory management?",
    options: [
      {
        text: "It allocates KV cache memory in fixed-size blocks.",
        isCorrect: true,
      },
      {
        text: "It reduces internal and external memory fragmentation.",
        isCorrect: true,
      },
      {
        text: "It avoids reserving the full maximum context length upfront.",
        isCorrect: true,
      },
      {
        text: "It improves scalability of serving multiple concurrent requests.",
        isCorrect: true,
      },
    ],
    explanation:
      "PagedAttention manages KV cache memory dynamically using blocks. This prevents large amounts of unused reserved memory and allows efficient multi-request serving.",
  },

  {
    id: "cme295-lect3-q41",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe latent (compressed) attention for KV caching?",
    options: [
      {
        text: "Keys and values are stored in a lower-dimensional latent space.",
        isCorrect: true,
      },
      {
        text: "Compression reduces memory usage of the KV cache.",
        isCorrect: true,
      },
      {
        text: "Decompression matrices reconstruct keys and values when needed.",
        isCorrect: true,
      },
      {
        text: "Compression can be shared across attention heads.",
        isCorrect: true,
      },
    ],
    explanation:
      "Latent attention stores compressed representations instead of full keys and values. This reduces memory footprint while still allowing reconstruction for attention computation.",
  },

  {
    id: "cme295-lect3-q42",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements correctly describe speculative decoding?",
    options: [
      {
        text: "A smaller draft model proposes multiple tokens.",
        isCorrect: true,
      },
      {
        text: "A larger target model validates the proposed tokens.",
        isCorrect: true,
      },
      {
        text: "Acceptance–rejection ensures correct target-model distribution.",
        isCorrect: true,
      },
      {
        text: "It aims to generate multiple tokens per target-model forward pass.",
        isCorrect: true,
      },
    ],
    explanation:
      "Speculative decoding accelerates generation by batching validation of draft tokens. The acceptance–rejection mechanism guarantees correctness with respect to the target model.",
  },

  {
    id: "cme295-lect3-q43",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements correctly describe Multi-Token Prediction (MTP)?",
    options: [
      {
        text: "The model predicts several future tokens at once.",
        isCorrect: true,
      },
      {
        text: "Multiple prediction heads are trained jointly.",
        isCorrect: true,
      },
      {
        text: "Draft and target predictions come from the same model.",
        isCorrect: true,
      },
      {
        text: "It modifies the training objective compared to next-token prediction.",
        isCorrect: true,
      },
    ],
    explanation:
      "MTP extends training to predict multiple tokens per step. At inference, this allows faster generation by validating several tokens at once within the same model.",
  },

  {
    id: "cme295-lect3-q44",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the purpose of inference optimizations in LLMs?",
    options: [
      { text: "They reduce latency of text generation.", isCorrect: true },
      { text: "They lower memory usage during inference.", isCorrect: true },
      {
        text: "They improve throughput when serving many users.",
        isCorrect: true,
      },
      {
        text: "They aim to preserve output quality while improving efficiency.",
        isCorrect: true,
      },
    ],
    explanation:
      "Inference optimizations focus on efficiency rather than changing model behavior. The goal is faster, cheaper generation with minimal or no quality loss.",
  },

  // ============================================================
  // Q45–Q53: THREE TRUE
  // ============================================================

  {
    id: "cme295-lect3-q45",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about KV caching during training are correct?",
    options: [
      {
        text: "KV caching is generally unnecessary during teacher-forced training.",
        isCorrect: true,
      },
      {
        text: "Training typically processes full sequences in parallel.",
        isCorrect: true,
      },
      {
        text: "KV caching is mainly beneficial for autoregressive inference.",
        isCorrect: true,
      },
      {
        text: "KV caching is required for gradient computation.",
        isCorrect: false,
      },
    ],
    explanation:
      "During training, full sequences are processed simultaneously, so past keys and values are recomputed anyway. KV caching mainly targets inference-time efficiency.",
  },

  {
    id: "cme295-lect3-q46",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about routing decisions in MoE layers are correct?",
    options: [
      {
        text: "Routing is typically computed after the self-attention sublayer.",
        isCorrect: true,
      },
      {
        text: "Routing decisions are based on token representations.",
        isCorrect: true,
      },
      {
        text: "Each Transformer layer usually has its own router.",
        isCorrect: true,
      },
      {
        text: "Routing is shared across all layers by default.",
        isCorrect: false,
      },
    ],
    explanation:
      "Routing uses contextual token embeddings produced by attention. Each layer learns its own routing behavior, enabling different experts to specialize per layer.",
  },

  {
    id: "cme295-lect3-q47",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about speculative decoding efficiency are correct?",
    options: [
      {
        text: "It reduces the number of target-model forward passes.",
        isCorrect: true,
      },
      {
        text: "It benefits from the fact that inference is often memory-bound.",
        isCorrect: true,
      },
      {
        text: "It can generate several tokens per validation step.",
        isCorrect: true,
      },
      { text: "It always eliminates rejection cases.", isCorrect: false },
    ],
    explanation:
      "Speculative decoding batches validation to reduce expensive target-model calls. Rejections can still occur, but overall speedups are often substantial.",
  },

  {
    id: "cme295-lect3-q48",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about context rot are correct?",
    options: [
      {
        text: "It refers to degradation in effective information use with long contexts.",
        isCorrect: true,
      },
      {
        text: "Distractor tokens can worsen retrieval performance.",
        isCorrect: true,
      },
      {
        text: "It can occur even if the answer is present in the context.",
        isCorrect: true,
      },
      { text: "It guarantees incorrect answers.", isCorrect: false },
    ],
    explanation:
      "Context rot describes reduced ability to leverage relevant information in long inputs. It increases difficulty but does not guarantee failure.",
  },

  {
    id: "cme295-lect3-q49",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about prompting structure are correct?",
    options: [
      {
        text: "Prompts can include context, instructions, input, and constraints.",
        isCorrect: true,
      },
      {
        text: "Different prompt components serve different functional roles.",
        isCorrect: true,
      },
      {
        text: "Constraints can restrict output format or content.",
        isCorrect: true,
      },
      {
        text: "Prompt structure has no impact on model behavior.",
        isCorrect: false,
      },
    ],
    explanation:
      "Well-structured prompts guide the model more effectively. Each component helps clarify what the model should do and how it should respond.",
  },

  {
    id: "cme295-lect3-q50",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements about few-shot prompting are correct?",
    options: [
      {
        text: "It includes example input–output pairs in the prompt.",
        isCorrect: true,
      },
      { text: "It often improves task performance.", isCorrect: true },
      {
        text: "It increases context length and inference cost.",
        isCorrect: true,
      },
      { text: "It updates the model’s parameters.", isCorrect: false },
    ],
    explanation:
      "Few-shot prompting conditions behavior through examples rather than weight updates. The trade-off is increased token usage and latency.",
  },

  {
    id: "cme295-lect3-q51",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about chain-of-thought interpretability are correct?",
    options: [
      {
        text: "It exposes intermediate reasoning steps as tokens.",
        isCorrect: true,
      },
      { text: "It can help identify reasoning errors.", isCorrect: true },
      {
        text: "It makes debugging easier compared to opaque outputs.",
        isCorrect: true,
      },
      { text: "It guarantees logically correct reasoning.", isCorrect: false },
    ],
    explanation:
      "Chain-of-thought improves transparency and debugging. However, exposed reasoning can still be flawed or misleading.",
  },

  {
    id: "cme295-lect3-q52",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about inference determinism are correct?",
    options: [
      {
        text: "Sampling introduces randomness into token selection.",
        isCorrect: true,
      },
      {
        text: "Floating-point operations can introduce nondeterminism.",
        isCorrect: true,
      },
      {
        text: "Parallel hardware execution can affect numerical results.",
        isCorrect: true,
      },
      {
        text: "Determinism is guaranteed at temperature zero in practice.",
        isCorrect: false,
      },
    ],
    explanation:
      "Even with deterministic decoding strategies, hardware and numerical effects can introduce variability. Absolute determinism is difficult to guarantee.",
  },

  {
    id: "cme295-lect3-q53",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about inference-time optimizations are correct?",
    options: [
      { text: "They do not require retraining the model.", isCorrect: true },
      { text: "They can be combined with each other.", isCorrect: true },
      { text: "They are crucial for large-scale deployment.", isCorrect: true },
      {
        text: "They fundamentally change the model architecture.",
        isCorrect: false,
      },
    ],
    explanation:
      "Most inference optimizations operate on execution rather than learning. They are essential for practical deployment of large models.",
  },

  // ============================================================
  // Q54–Q62: TWO TRUE
  // ============================================================

  {
    id: "cme295-lect3-q54",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about memory fragmentation are correct?",
    options: [
      {
        text: "Internal fragmentation refers to unused reserved memory.",
        isCorrect: true,
      },
      {
        text: "External fragmentation refers to scattered free memory blocks.",
        isCorrect: true,
      },
      { text: "Fragmentation improves cache locality.", isCorrect: false },
      { text: "Fragmentation is irrelevant for KV caching.", isCorrect: false },
    ],
    explanation:
      "Memory fragmentation wastes space and reduces efficiency. Managing fragmentation is critical for scalable inference systems.",
  },

  {
    id: "cme295-lect3-q55",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about acceptance–rejection in speculative decoding are correct?",
    options: [
      {
        text: "Accepted tokens match the target model distribution.",
        isCorrect: true,
      },
      { text: "Rejected tokens require resampling.", isCorrect: true },
      { text: "All draft tokens are always accepted.", isCorrect: false },
      {
        text: "Acceptance eliminates the need for validation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Acceptance–rejection ensures correctness while allowing speedups. Rejections trigger corrective sampling to preserve the target distribution.",
  },

  {
    id: "cme295-lect3-q56",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about multi-token prediction objectives are correct?",
    options: [
      {
        text: "They differ from standard next-token prediction objectives.",
        isCorrect: true,
      },
      {
        text: "They require predicting future tokens jointly.",
        isCorrect: true,
      },
      {
        text: "They eliminate the need for autoregressive decoding.",
        isCorrect: false,
      },
      {
        text: "They remove the need for attention mechanisms.",
        isCorrect: false,
      },
    ],
    explanation:
      "Multi-token prediction changes the training objective but does not remove autoregressive structure or attention mechanisms.",
  },

  {
    id: "cme295-lect3-q57",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about top-k and top-p sampling are correct?",
    options: [
      { text: "Both restrict the set of candidate tokens.", isCorrect: true },
      { text: "Both aim to balance diversity and coherence.", isCorrect: true },
      {
        text: "Both guarantee identical outputs across runs.",
        isCorrect: false,
      },
      { text: "Both remove randomness from generation.", isCorrect: false },
    ],
    explanation:
      "Top-k and top-p sampling introduce controlled randomness. They reduce unlikely tokens but remain stochastic.",
  },

  {
    id: "cme295-lect3-q58",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements about logits and probabilities are correct?",
    options: [
      { text: "Logits are unnormalized scores.", isCorrect: true },
      {
        text: "Probabilities are obtained after applying softmax.",
        isCorrect: true,
      },
      { text: "Logits must be positive.", isCorrect: false },
      { text: "Probabilities can exceed one.", isCorrect: false },
    ],
    explanation:
      "Logits can take any real value and are converted into probabilities via softmax. Probabilities are bounded between zero and one.",
  },

  {
    id: "cme295-lect3-q59",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about temperature scaling are correct?",
    options: [
      {
        text: "Lower temperature sharpens the probability distribution.",
        isCorrect: true,
      },
      {
        text: "Higher temperature flattens the distribution.",
        isCorrect: true,
      },
      { text: "Temperature changes model weights.", isCorrect: false },
      { text: "Temperature is only used during training.", isCorrect: false },
    ],
    explanation:
      "Temperature rescales logits at inference time. It affects sampling behavior without altering learned parameters.",
  },

  {
    id: "cme295-lect3-q60",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements about expert capacity scaling are correct?",
    options: [
      { text: "MoE increases total parameter count.", isCorrect: true },
      {
        text: "Active parameters per token can remain constant.",
        isCorrect: true,
      },
      {
        text: "All parameters are used in every forward pass.",
        isCorrect: false,
      },
      {
        text: "Capacity scaling removes the need for routing.",
        isCorrect: false,
      },
    ],
    explanation:
      "MoE models scale capacity by adding experts while keeping active computation limited. Routing determines which parameters are used.",
  },

  {
    id: "cme295-lect3-q61",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about inference bottlenecks are correct?",
    options: [
      {
        text: "Inference is often memory-bound rather than compute-bound.",
        isCorrect: true,
      },
      { text: "KV cache access can dominate latency.", isCorrect: true },
      {
        text: "More parameters always mean faster inference.",
        isCorrect: false,
      },
      { text: "Batching eliminates all bottlenecks.", isCorrect: false },
    ],
    explanation:
      "Memory bandwidth and cache access are major bottlenecks in inference. Many optimizations target reducing memory movement.",
  },

  {
    id: "cme295-lect3-q62",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about prompt-based control are correct?",
    options: [
      {
        text: "Behavior can be modified without fine-tuning.",
        isCorrect: true,
      },
      { text: "Instructions influence generation style.", isCorrect: true },
      {
        text: "Prompting guarantees perfect adherence to constraints.",
        isCorrect: false,
      },
      { text: "Prompting replaces model training entirely.", isCorrect: false },
    ],
    explanation:
      "Prompting is powerful but imperfect. It guides behavior probabilistically rather than enforcing hard guarantees.",
  },

  // ============================================================
  // Q63–Q70: ONE TRUE
  // ============================================================

  {
    id: "cme295-lect3-q63",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statement best explains why KV caching speeds up decoding?",
    options: [
      { text: "It removes the need for attention entirely.", isCorrect: false },
      {
        text: "It avoids recomputing keys and values for past tokens.",
        isCorrect: true,
      },
      { text: "It reduces vocabulary size.", isCorrect: false },
      { text: "It changes the Transformer architecture.", isCorrect: false },
    ],
    explanation:
      "KV caching prevents redundant computation by reusing previously computed attention components for earlier tokens.",
  },

  {
    id: "cme295-lect3-q64",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statement best describes Grouped Query Attention?",
    options: [
      { text: "Each query has its own key and value heads.", isCorrect: false },
      {
        text: "Queries are grouped to share key and value heads.",
        isCorrect: true,
      },
      { text: "Attention is computed without softmax.", isCorrect: false },
      { text: "Attention is replaced by routing.", isCorrect: false },
    ],
    explanation:
      "GQA reduces memory by allowing multiple queries to share keys and values while preserving more flexibility than MQA.",
  },

  {
    id: "cme295-lect3-q65",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statement best captures the risk of routing collapse?",
    options: [
      { text: "Experts become too diverse.", isCorrect: false },
      { text: "Only a few experts dominate usage.", isCorrect: true },
      { text: "Routing becomes deterministic.", isCorrect: false },
      { text: "Model parameters decrease.", isCorrect: false },
    ],
    explanation:
      "Routing collapse undermines MoE benefits by underutilizing experts. Auxiliary losses help mitigate this issue.",
  },

  {
    id: "cme295-lect3-q66",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement best describes acceptance in speculative decoding?",
    options: [
      { text: "Draft tokens are always accepted.", isCorrect: false },
      {
        text: "Tokens are accepted when draft probability is consistent with target probability.",
        isCorrect: true,
      },
      { text: "Acceptance ignores the target model.", isCorrect: false },
      { text: "Acceptance removes randomness.", isCorrect: false },
    ],
    explanation:
      "Acceptance ensures that generated tokens match the target model’s distribution, preserving correctness while accelerating decoding.",
  },

  {
    id: "cme295-lect3-q67",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statement best describes context length?",
    options: [
      { text: "The number of parameters in the model.", isCorrect: false },
      {
        text: "The maximum number of tokens processed in a single pass.",
        isCorrect: true,
      },
      { text: "The number of experts in an MoE model.", isCorrect: false },
      { text: "The batch size during training.", isCorrect: false },
    ],
    explanation:
      "Context length defines how many tokens a model can attend to at once. It directly affects attention cost and memory usage.",
  },

  {
    id: "cme295-lect3-q68",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement best explains why longer sequences have lower joint probability?",
    options: [
      { text: "Probabilities increase with each token.", isCorrect: false },
      {
        text: "Joint probability multiplies many values less than one.",
        isCorrect: true,
      },
      { text: "Softmax enforces decay.", isCorrect: false },
      { text: "Temperature scaling causes collapse.", isCorrect: false },
    ],
    explanation:
      "Sequence probability is the product of conditional probabilities. Multiplying values below one causes the total probability to shrink with length.",
  },

  {
    id: "cme295-lect3-q69",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statement best describes the main trade-off of chain-of-thought prompting?",
    options: [
      { text: "Lower accuracy for reasoning tasks.", isCorrect: false },
      {
        text: "Improved reasoning at the cost of more tokens.",
        isCorrect: true,
      },
      { text: "Reduced interpretability.", isCorrect: false },
      { text: "Incompatibility with sampling.", isCorrect: false },
    ],
    explanation:
      "Chain-of-thought improves reasoning and interpretability but increases token count, latency, and cost.",
  },

  {
    id: "cme295-lect3-q70",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statement best summarizes the goal of inference-time approximations?",
    options: [
      { text: "To retrain the model faster.", isCorrect: false },
      {
        text: "To reduce cost and latency with minimal quality loss.",
        isCorrect: true,
      },
      { text: "To change the training objective.", isCorrect: false },
      { text: "To remove attention mechanisms.", isCorrect: false },
    ],
    explanation:
      "Inference-time approximations aim to make generation faster and cheaper while preserving output quality as much as possible.",
  },

  // ============================================================
  // Q71–Q100 (third batch)
  // ============================================================

  // ============================================================
  // Q71–Q78: ALL TRUE
  // ============================================================

  {
    id: "cme295-lect3-q71",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe autoregressive text generation?",
    options: [
      { text: "Tokens are generated one at a time.", isCorrect: true },
      {
        text: "Each token is conditioned on previously generated tokens.",
        isCorrect: true,
      },
      {
        text: "Generation stops based on a stopping criterion such as an end-of-sequence token.",
        isCorrect: true,
      },
      {
        text: "The same model is reused at every decoding step.",
        isCorrect: true,
      },
    ],
    explanation:
      "Autoregressive generation predicts tokens sequentially. At each step, the model conditions on all previously generated tokens until a stopping condition is reached.",
  },

  {
    id: "cme295-lect3-q72",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe why greedy decoding can be suboptimal?",
    options: [
      {
        text: "It optimizes local token probability rather than global sequence probability.",
        isCorrect: true,
      },
      {
        text: "Early token choices can restrict later high-probability continuations.",
        isCorrect: true,
      },
      { text: "It cannot revise earlier decisions.", isCorrect: true },
      {
        text: "It often leads to repetitive or generic outputs.",
        isCorrect: true,
      },
    ],
    explanation:
      "Greedy decoding commits to locally optimal choices that may block better long-term sequences. This often results in repetitive or less coherent outputs.",
  },

  {
    id: "cme295-lect3-q73",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe why beam search is rarely used for open-ended generation?",
    options: [
      {
        text: "It favors high-likelihood but low-diversity outputs.",
        isCorrect: true,
      },
      {
        text: "It tends to converge to similar or generic sequences.",
        isCorrect: true,
      },
      {
        text: "It optimizes likelihood rather than creativity.",
        isCorrect: true,
      },
      {
        text: "It increases computational cost relative to sampling.",
        isCorrect: true,
      },
    ],
    explanation:
      "Beam search prioritizes likelihood and consistency, which reduces diversity. This makes it less suitable for creative or conversational tasks.",
  },

  {
    id: "cme295-lect3-q74",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements correctly describe sampling-based decoding?",
    options: [
      {
        text: "Tokens are sampled according to a probability distribution.",
        isCorrect: true,
      },
      {
        text: "Higher-probability tokens are more likely to be selected.",
        isCorrect: true,
      },
      { text: "Low-probability tokens can still be sampled.", isCorrect: true },
      {
        text: "Sampling introduces nondeterminism into generation.",
        isCorrect: true,
      },
    ],
    explanation:
      "Sampling draws tokens from the model’s predicted distribution. This allows diversity and creativity while remaining probabilistically grounded.",
  },

  {
    id: "cme295-lect3-q75",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the effect of temperature on softmax?",
    options: [
      {
        text: "Temperature rescales logits before normalization.",
        isCorrect: true,
      },
      {
        text: "Lower temperature increases confidence in top tokens.",
        isCorrect: true,
      },
      {
        text: "Higher temperature increases entropy of the distribution.",
        isCorrect: true,
      },
      {
        text: "Temperature influences sampling behavior without changing model weights.",
        isCorrect: true,
      },
    ],
    explanation:
      "Temperature modifies the sharpness of the probability distribution. It affects randomness at inference time without altering learned parameters.",
  },

  {
    id: "cme295-lect3-q76",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the motivation for Mixture-of-Experts models?",
    options: [
      {
        text: "To increase model capacity without linearly increasing compute.",
        isCorrect: true,
      },
      {
        text: "To activate only a subset of parameters per token.",
        isCorrect: true,
      },
      { text: "To scale to very large parameter counts.", isCorrect: true },
      {
        text: "To reduce inference cost compared to dense models of equal size.",
        isCorrect: true,
      },
    ],
    explanation:
      "MoE models increase capacity by adding experts while activating only a subset. This allows very large models with controlled inference cost.",
  },

  {
    id: "cme295-lect3-q77",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe why the feed-forward network is the usual location for MoE layers?",
    options: [
      {
        text: "It contains a large fraction of the model’s parameters.",
        isCorrect: true,
      },
      {
        text: "It dominates FLOPs compared to attention layers.",
        isCorrect: true,
      },
      {
        text: "Replacing it yields significant capacity gains.",
        isCorrect: true,
      },
      {
        text: "It preserves the structure of attention mechanisms.",
        isCorrect: true,
      },
    ],
    explanation:
      "The feed-forward network is parameter-heavy and computationally expensive. Replacing it with MoE layers yields large capacity gains with minimal architectural disruption.",
  },

  {
    id: "cme295-lect3-q78",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe why routing is performed at the token level?",
    options: [
      {
        text: "Different tokens may require different expert specializations.",
        isCorrect: true,
      },
      {
        text: "Token-level routing increases expressiveness.",
        isCorrect: true,
      },
      { text: "Routing can adapt dynamically to context.", isCorrect: true },
      {
        text: "It allows fine-grained allocation of compute.",
        isCorrect: true,
      },
    ],
    explanation:
      "Routing per token allows the model to dynamically select experts based on contextual needs, improving efficiency and specialization.",
  },

  // ============================================================
  // Q79–Q86: THREE TRUE
  // ============================================================

  {
    id: "cme295-lect3-q79",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about top-k sampling are correct?",
    options: [
      {
        text: "It limits sampling to the k most likely tokens.",
        isCorrect: true,
      },
      {
        text: "It prevents extremely unlikely tokens from being selected.",
        isCorrect: true,
      },
      {
        text: "It preserves stochasticity within the selected set.",
        isCorrect: true,
      },
      { text: "It always produces deterministic output.", isCorrect: false },
    ],
    explanation:
      "Top-k sampling restricts candidate tokens but still samples randomly within that subset, maintaining diversity.",
  },

  {
    id: "cme295-lect3-q80",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about auxiliary MoE load-balancing losses are correct?",
    options: [
      { text: "They penalize uneven expert utilization.", isCorrect: true },
      {
        text: "They encourage more uniform routing distributions.",
        isCorrect: true,
      },
      { text: "They mitigate routing collapse.", isCorrect: true },
      {
        text: "They replace the main language modeling loss.",
        isCorrect: false,
      },
    ],
    explanation:
      "Auxiliary losses supplement the main objective to promote balanced expert usage. They help prevent collapse without replacing the core task loss.",
  },

  {
    id: "cme295-lect3-q81",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about guided decoding are correct?",
    options: [
      { text: "It restricts the set of valid next tokens.", isCorrect: true },
      {
        text: "It can enforce structured output constraints.",
        isCorrect: true,
      },
      { text: "It operates during inference.", isCorrect: true },
      { text: "It requires retraining the model.", isCorrect: false },
    ],
    explanation:
      "Guided decoding constrains token selection at inference time. It does not modify model weights.",
  },

  {
    id: "cme295-lect3-q82",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about context length limitations are correct?",
    options: [
      {
        text: "Longer context increases attention computation cost.",
        isCorrect: true,
      },
      {
        text: "Very long contexts can degrade retrieval accuracy.",
        isCorrect: true,
      },
      {
        text: "Context rot can occur even when answers are present.",
        isCorrect: true,
      },
      { text: "Longer context always improves performance.", isCorrect: false },
    ],
    explanation:
      "While longer contexts allow more information, they also increase cost and can harm effective retrieval due to context rot.",
  },

  {
    id: "cme295-lect3-q83",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements about speculative decoding are correct?",
    options: [
      {
        text: "It uses a smaller draft model to propose tokens.",
        isCorrect: true,
      },
      {
        text: "It validates draft tokens using a larger target model.",
        isCorrect: true,
      },
      {
        text: "It preserves the target model’s output distribution.",
        isCorrect: true,
      },
      { text: "It removes the need for sampling.", isCorrect: false },
    ],
    explanation:
      "Speculative decoding accelerates generation while preserving correctness. Sampling and validation are still required.",
  },

  {
    id: "cme295-lect3-q84",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements about multi-token prediction are correct?",
    options: [
      {
        text: "The model predicts multiple future tokens per step.",
        isCorrect: true,
      },
      { text: "It changes the training objective.", isCorrect: true },
      { text: "It can reduce inference latency.", isCorrect: true },
      { text: "It eliminates autoregressive decoding.", isCorrect: false },
    ],
    explanation:
      "Multi-token prediction accelerates inference by predicting several tokens at once, but decoding remains fundamentally autoregressive.",
  },

  {
    id: "cme295-lect3-q85",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about chain-of-thought prompting are correct?",
    options: [
      {
        text: "It encourages explicit intermediate reasoning.",
        isCorrect: true,
      },
      { text: "It often improves reasoning task accuracy.", isCorrect: true },
      { text: "It increases inference token count.", isCorrect: true },
      { text: "It enforces logical correctness.", isCorrect: false },
    ],
    explanation:
      "Chain-of-thought improves reasoning but does not guarantee correctness. It also increases cost due to longer outputs.",
  },

  {
    id: "cme295-lect3-q86",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about self-consistency prompting are correct?",
    options: [
      { text: "It samples multiple reasoning paths.", isCorrect: true },
      { text: "It aggregates answers via majority voting.", isCorrect: true },
      { text: "It can improve robustness of final answers.", isCorrect: true },
      { text: "It reduces computational cost.", isCorrect: false },
    ],
    explanation:
      "Self-consistency trades additional computation for improved robustness by aggregating multiple sampled solutions.",
  },

  // ============================================================
  // Q87–Q94: TWO TRUE
  // ============================================================

  {
    id: "cme295-lect3-q87",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about in-context learning are correct?",
    options: [
      {
        text: "It adapts behavior without updating model weights.",
        isCorrect: true,
      },
      {
        text: "It relies on information provided in the prompt.",
        isCorrect: true,
      },
      { text: "It permanently changes the model.", isCorrect: false },
      { text: "It requires gradient updates.", isCorrect: false },
    ],
    explanation:
      "In-context learning steers behavior through the prompt alone. Model parameters remain unchanged.",
  },

  {
    id: "cme295-lect3-q88",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about zero-shot prompting are correct?",
    options: [
      { text: "It uses instructions without examples.", isCorrect: true },
      {
        text: "It relies on the model’s pretrained knowledge.",
        isCorrect: true,
      },
      { text: "It requires labeled demonstrations.", isCorrect: false },
      { text: "It fine-tunes the model.", isCorrect: false },
    ],
    explanation:
      "Zero-shot prompting depends on clear instructions and pretrained knowledge, without examples or weight updates.",
  },

  {
    id: "cme295-lect3-q89",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements about few-shot prompting trade-offs are correct?",
    options: [
      { text: "It can improve task alignment.", isCorrect: true },
      { text: "It increases context length and cost.", isCorrect: true },
      { text: "It guarantees generalization.", isCorrect: false },
      { text: "It removes the need for instructions.", isCorrect: false },
    ],
    explanation:
      "Few-shot examples help alignment but consume context and do not guarantee generalization beyond the examples.",
  },

  {
    id: "cme295-lect3-q90",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about context windows are correct?",
    options: [
      {
        text: "They limit how many tokens can be attended to.",
        isCorrect: true,
      },
      { text: "They affect memory and compute usage.", isCorrect: true },
      { text: "They are unrelated to attention.", isCorrect: false },
      { text: "They only matter during training.", isCorrect: false },
    ],
    explanation:
      "Context windows constrain attention scope and heavily influence inference cost and feasibility.",
  },

  {
    id: "cme295-lect3-q91",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements about KV cache memory usage are correct?",
    options: [
      { text: "It grows with sequence length.", isCorrect: true },
      { text: "It can become a bottleneck during inference.", isCorrect: true },
      { text: "It is constant regardless of context.", isCorrect: false },
      { text: "It is only used in encoder-only models.", isCorrect: false },
    ],
    explanation:
      "KV cache memory scales with sequence length and can dominate inference costs, especially for long contexts.",
  },

  {
    id: "cme295-lect3-q92",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about inference-time nondeterminism are correct?",
    options: [
      { text: "Sampling introduces randomness.", isCorrect: true },
      {
        text: "Floating-point arithmetic can cause variation.",
        isCorrect: true,
      },
      { text: "Transformers are probabilistic by design.", isCorrect: false },
      { text: "Determinism is guaranteed on GPUs.", isCorrect: false },
    ],
    explanation:
      "The model itself is deterministic, but sampling and numerical effects introduce nondeterminism during inference.",
  },

  {
    id: "cme295-lect3-q93",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about inference optimization goals are correct?",
    options: [
      { text: "Reducing latency.", isCorrect: true },
      { text: "Improving throughput.", isCorrect: true },
      { text: "Increasing training data.", isCorrect: false },
      { text: "Changing model semantics.", isCorrect: false },
    ],
    explanation:
      "Inference optimizations aim to improve speed and scalability without changing what the model computes.",
  },

  {
    id: "cme295-lect3-q94",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about approximate inference techniques are correct?",
    options: [
      { text: "They trade exactness for speed.", isCorrect: true },
      { text: "They aim to preserve output quality.", isCorrect: true },
      { text: "They always change model predictions.", isCorrect: false },
      { text: "They eliminate attention computation.", isCorrect: false },
    ],
    explanation:
      "Approximate techniques reduce cost while aiming to maintain similar output distributions. They do not remove core mechanisms like attention.",
  },

  // ============================================================
  // Q95–Q100: ONE TRUE
  // ============================================================

  {
    id: "cme295-lect3-q95",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statement best describes why sampling produces different outputs?",
    options: [
      { text: "The model weights change at inference time.", isCorrect: false },
      {
        text: "Randomness is introduced during token selection.",
        isCorrect: true,
      },
      {
        text: "The Transformer is nondeterministic internally.",
        isCorrect: false,
      },
      { text: "Softmax outputs are fixed.", isCorrect: false },
    ],
    explanation:
      "Sampling introduces randomness when selecting tokens from a probability distribution, leading to different outputs.",
  },

  {
    id: "cme295-lect3-q96",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statement best explains context rot?",
    options: [
      { text: "Models forget earlier training data.", isCorrect: false },
      {
        text: "Relevant information becomes harder to retrieve in long contexts.",
        isCorrect: true,
      },
      { text: "The model overfits to recent tokens only.", isCorrect: false },
      { text: "Attention stops functioning.", isCorrect: false },
    ],
    explanation:
      "Context rot refers to degradation in effective information use as context length grows, especially with distractors.",
  },

  {
    id: "cme295-lect3-q97",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement best explains why MoE models can reach trillions of parameters?",
    options: [
      {
        text: "All parameters are used in every forward pass.",
        isCorrect: false,
      },
      {
        text: "Only a subset of experts is activated per token.",
        isCorrect: true,
      },
      {
        text: "Attention layers scale linearly with parameters.",
        isCorrect: false,
      },
      {
        text: "Routing removes the need for large datasets.",
        isCorrect: false,
      },
    ],
    explanation:
      "MoE models scale total parameters by adding experts while keeping active computation limited through routing.",
  },

  {
    id: "cme295-lect3-q98",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statement best describes the main benefit of speculative decoding?",
    options: [
      { text: "Improved training convergence.", isCorrect: false },
      {
        text: "Faster inference with preserved output distribution.",
        isCorrect: true,
      },
      { text: "Higher model accuracy.", isCorrect: false },
      { text: "Reduced parameter count.", isCorrect: false },
    ],
    explanation:
      "Speculative decoding accelerates inference by batching validation while preserving the target model’s output distribution.",
  },

  {
    id: "cme295-lect3-q99",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statement best summarizes the role of prompting in LLMs?",
    options: [
      { text: "It permanently alters model parameters.", isCorrect: false },
      {
        text: "It conditions model behavior through context.",
        isCorrect: true,
      },
      { text: "It replaces the need for training.", isCorrect: false },
      { text: "It guarantees perfect control.", isCorrect: false },
    ],
    explanation:
      "Prompting steers behavior probabilistically via context, without modifying the model’s weights.",
  },

  {
    id: "cme295-lect3-q100",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statement best captures the overarching theme of modern LLM design?",
    options: [
      {
        text: "Maximize parameter count regardless of cost.",
        isCorrect: false,
      },
      {
        text: "Balance scale, efficiency, and controllability.",
        isCorrect: true,
      },
      { text: "Eliminate autoregressive generation.", isCorrect: false },
      { text: "Avoid probabilistic decoding.", isCorrect: false },
    ],
    explanation:
      "Modern LLM design balances scale with efficiency and controllability, using techniques like MoE, sampling, and inference optimizations.",
  },
];
