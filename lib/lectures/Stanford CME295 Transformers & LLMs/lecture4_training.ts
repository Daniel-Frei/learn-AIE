import { Question } from "../../quiz";

export const stanfordCME295Lecture4TrainingQuestions: Question[] = [
  // ============================================================
  // Lecture 4 – LLM Training (Pre-training, Scaling, Parallelism,
  // FlashAttention, Quantization, SFT, LoRA)
  // Q1–Q50 (first half)
  // ============================================================

  // ============================================================
  // Q1–Q12: ALL TRUE (12 questions)
  // ============================================================

  {
    id: "cme295-lect4-q01",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements correctly describe the goal of pre-training large language models?",
    options: [
      { text: "The model is trained to predict the next token given previous tokens.", isCorrect: true },
      { text: "Training data typically includes natural language text and source code.", isCorrect: true },
      { text: "Pre-training aims to capture general structure of language rather than a single task.", isCorrect: true },
      { text: "Pre-training is usually the most computationally expensive training stage.", isCorrect: true },
    ],
    explanation:
      "Pre-training optimizes next-token prediction over massive, diverse corpora. This stage dominates cost because it involves huge datasets and very large models, but it yields general language representations reusable across tasks."
  },

  {
    id: "cme295-lect4-q02",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements correctly describe transfer learning in the context of language models?",
    options: [
      { text: "A pre-trained model is reused instead of training from scratch for each task.", isCorrect: true },
      { text: "Knowledge learned on one language task can help performance on another.", isCorrect: true },
      { text: "Fine-tuning adapts a general model to a more specific objective.", isCorrect: true },
      { text: "Transfer learning reduces the amount of task-specific data needed.", isCorrect: true },
    ],
    explanation:
      "Transfer learning exploits shared structure across language tasks. A general pre-trained model can be adapted with relatively little additional data, making training more efficient than starting from random initialization."
  },

  {
    id: "cme295-lect4-q03",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements about Common Crawl–style datasets are correct?",
    options: [
      { text: "They contain text scraped from a wide range of public websites.", isCorrect: true },
      { text: "They often include multilingual content.", isCorrect: true },
      { text: "They are commonly used in large-scale language model pre-training.", isCorrect: true },
      { text: "They are measured in scale by number of tokens rather than number of documents.", isCorrect: true },
    ],
    explanation:
      "Common Crawl–like datasets aggregate massive amounts of public web text. For LLMs, scale is usually discussed in terms of tokens processed, which directly affects compute cost and training dynamics."
  },

  {
    id: "cme295-lect4-q04",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements correctly describe floating point operations (FLOPs) as used in LLM training discussions?",
    options: [
      { text: "They count arithmetic operations involving floating point numbers.", isCorrect: true },
      { text: "They are often used to estimate total training compute cost.", isCorrect: true },
      { text: "They scale roughly with both model size and dataset size.", isCorrect: true },
      { text: "They depend on architectural choices such as dense versus sparse models.", isCorrect: true },
    ],
    explanation:
      "Total FLOPs provide a coarse but useful measure of training cost. While exact formulas vary, compute generally increases with the number of parameters, number of tokens, and architectural details."
  },

  {
    id: "cme295-lect4-q05",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements correctly describe the distinction between FLOPs and FLOPs per second?",
    options: [
      { text: "FLOPs measure total computation required.", isCorrect: true },
      { text: "FLOPs per second measure hardware throughput.", isCorrect: true },
      { text: "Graphics processing unit specifications often report FLOPs per second.", isCorrect: true },
      { text: "Context usually determines which meaning of FLOPs is intended.", isCorrect: true },
    ],
    explanation:
      "FLOPs quantify total work, while FLOPs per second quantify speed. Because the same acronym is used, interpretation relies heavily on context such as whether hardware or training cost is being discussed."
  },

  {
    id: "cme295-lect4-q06",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements correctly describe empirical scaling laws for language models?",
    options: [
      { text: "Increasing model size tends to reduce training loss.", isCorrect: true },
      { text: "Increasing dataset size tends to improve performance.", isCorrect: true },
      { text: "Increasing available compute often yields better models.", isCorrect: true },
      { text: "These relationships were studied systematically in large experimental studies.", isCorrect: true },
    ],
    explanation:
      "Scaling law studies showed smooth, predictable improvements as model size, data size, and compute increase. These findings motivated the rapid growth of LLMs over recent years."
  },

  {
    id: "cme295-lect4-q07",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements correctly characterize the idea behind compute-optimal training (often associated with the Chinchilla results)?",
    options: [
      { text: "For a fixed compute budget, there is an optimal trade-off between model size and data size.", isCorrect: true },
      { text: "Training too large a model on too little data can be inefficient.", isCorrect: true },
      { text: "Models may benefit from more training tokens rather than more parameters.", isCorrect: true },
      { text: "Optimal scaling relationships can be estimated empirically for a given setup.", isCorrect: true },
    ],
    explanation:
      "Compute-optimal scaling emphasizes balancing parameters and data. Under-training large models wastes capacity, while over-training small models wastes compute; empirical studies identify sweet spots."
  },

  {
    id: "cme295-lect4-q08",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements correctly describe the concept of a knowledge cutoff in language models?",
    options: [
      { text: "It refers to the latest date of data included during pre-training.", isCorrect: true },
      { text: "The model cannot directly know events occurring after this date.", isCorrect: true },
      { text: "Knowledge cutoff information is often listed in model documentation.", isCorrect: true },
      { text: "Updating knowledge after cutoff is non-trivial without retraining or fine-tuning.", isCorrect: true },
    ],
    explanation:
      "A knowledge cutoff reflects the temporal limit of pre-training data. Because knowledge is embedded in weights, updating facts without harming other capabilities is challenging."
  },

  {
    id: "cme295-lect4-q09",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements correctly describe challenges associated with LLM pre-training?",
    options: [
      { text: "High financial cost due to compute requirements.", isCorrect: true },
      { text: "Significant energy consumption and environmental impact.", isCorrect: true },
      { text: "Risk of memorizing or reproducing training data.", isCorrect: true },
      { text: "Difficulty of updating specific knowledge after training.", isCorrect: true },
    ],
    explanation:
      "Pre-training at scale is expensive, energy-intensive, and raises concerns about memorization and data freshness. These challenges motivate careful dataset curation and post-training methods."
  },

  {
    id: "cme295-lect4-q10",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements correctly describe why graphics processing units (GPUs) are well suited for LLM training?",
    options: [
      { text: "They are optimized for large matrix multiplications.", isCorrect: true },
      { text: "Transformer models rely heavily on linear algebra operations.", isCorrect: true },
      { text: "They provide high floating point throughput.", isCorrect: true },
      { text: "They support massive parallel computation.", isCorrect: true },
    ],
    explanation:
      "LLM training involves dense linear algebra, which GPUs handle efficiently. Their parallelism and high throughput make them the dominant hardware choice for large-scale training."
  },

  {
    id: "cme295-lect4-q11",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements correctly describe the forward pass during neural network training?",
    options: [
      { text: "Input data is propagated through the network layers.", isCorrect: true },
      { text: "Intermediate activations are computed at each layer.", isCorrect: true },
      { text: "A loss value is computed by comparing outputs to targets.", isCorrect: true },
      { text: "Activations are typically needed later for gradient computation.", isCorrect: true },
    ],
    explanation:
      "The forward pass computes predictions and loss. Intermediate activations are crucial because they are reused in the backward pass to compute gradients."
  },

  {
    id: "cme295-lect4-q12",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements correctly describe the backward pass in neural network training?",
    options: [
      { text: "Gradients of the loss with respect to parameters are computed.", isCorrect: true },
      { text: "The chain rule from calculus is applied layer by layer.", isCorrect: true },
      { text: "Gradients must be stored or accumulated before weight updates.", isCorrect: true },
      { text: "Backward computation typically requires access to forward activations.", isCorrect: true },
    ],
    explanation:
      "The backward pass propagates error signals backward through the network. It relies on stored activations and applies the chain rule to compute parameter gradients."
  },

  // ============================================================
  // Q13–Q25: EXACTLY 3 TRUE
  // ============================================================

  {
    id: "cme295-lect4-q13",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements about optimizers such as Adaptive Moment Estimation (Adam) are correct?",
    options: [
      { text: "They maintain moving averages of past gradients.", isCorrect: true },
      { text: "They store additional optimizer state in memory.", isCorrect: true },
      { text: "They adjust learning rates per parameter.", isCorrect: true },
      { text: "They eliminate the need to compute gradients.", isCorrect: false },
    ],
    explanation:
      "Adam-like optimizers track first and second moments of gradients to adapt updates. They still rely on standard gradient computation via backpropagation."
  },

  {
    id: "cme295-lect4-q14",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about memory usage during LLM training are correct?",
    options: [
      { text: "Model parameters occupy memory.", isCorrect: true },
      { text: "Optimizer states can double or triple memory requirements.", isCorrect: true },
      { text: "Activations scale with batch size and sequence length.", isCorrect: true },
      { text: "Memory usage is independent of context length.", isCorrect: false },
    ],
    explanation:
      "Training memory includes parameters, optimizer states, gradients, and activations. Longer sequences and larger batches significantly increase activation memory."
  },

  {
    id: "cme295-lect4-q15",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about data parallelism are correct?",
    options: [
      { text: "Each device processes a different subset of the training batch.", isCorrect: true },
      { text: "Each device holds a full copy of the model parameters.", isCorrect: true },
      { text: "Gradients must be aggregated across devices.", isCorrect: true },
      { text: "It removes all communication overhead between devices.", isCorrect: false },
    ],
    explanation:
      "Data parallelism replicates the model across devices while splitting data. Gradient synchronization introduces communication costs that limit scalability."
  },

  {
    id: "cme295-lect4-q16",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about Zero Redundancy Optimization (ZeRO) are correct?",
    options: [
      { text: "It reduces memory duplication across devices.", isCorrect: true },
      { text: "It can shard optimizer states across GPUs.", isCorrect: true },
      { text: "It can shard gradients or parameters depending on the stage.", isCorrect: true },
      { text: "It eliminates communication costs entirely.", isCorrect: false },
    ],
    explanation:
      "ZeRO partitions different training states across devices to save memory. While effective, it increases communication overhead due to frequent state exchanges."
  },

  {
    id: "cme295-lect4-q17",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about model parallelism are correct?",
    options: [
      { text: "It splits computation of a single model across devices.", isCorrect: true },
      { text: "It can be applied within a single training batch.", isCorrect: true },
      { text: "It is useful when a model does not fit on one device.", isCorrect: true },
      { text: "It requires duplicating all parameters on each device.", isCorrect: false },
    ],
    explanation:
      "Model parallelism distributes parts of the model itself across devices. This enables training of models larger than single-device memory limits."
  },

  {
    id: "cme295-lect4-q18",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about expert parallelism in mixture-of-experts models are correct?",
    options: [
      { text: "Different experts can be placed on different devices.", isCorrect: true },
      { text: "Only a subset of experts is activated per input.", isCorrect: true },
      { text: "Expert routing decisions affect communication patterns.", isCorrect: true },
      { text: "All experts must be evaluated for every token.", isCorrect: false },
    ],
    explanation:
      "Expert parallelism leverages sparse activation: only selected experts run per input. This reduces compute but introduces routing and communication complexity."
  },

  {
    id: "cme295-lect4-q19",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about pipeline parallelism are correct?",
    options: [
      { text: "Different layers of the model can be assigned to different devices.", isCorrect: true },
      { text: "Forward and backward passes can be staged across devices.", isCorrect: true },
      { text: "It can improve utilization for very deep models.", isCorrect: true },
      { text: "It requires all layers to run on a single device.", isCorrect: false },
    ],
    explanation:
      "Pipeline parallelism slices the model depth-wise. While it helps with memory limits, it introduces pipeline bubbles and scheduling complexity."
  },

  {
    id: "cme295-lect4-q20",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about attention computation bottlenecks are correct?",
    options: [
      { text: "Memory reads and writes can dominate runtime.", isCorrect: true },
      { text: "Self-attention involves multiple large matrix multiplications.", isCorrect: true },
      { text: "Softmax normalization introduces data dependencies.", isCorrect: true },
      { text: "Attention computation is limited only by arithmetic throughput.", isCorrect: false },
    ],
    explanation:
      "Although GPUs are fast at math, attention often becomes memory-bound. Data movement and softmax dependencies can be larger bottlenecks than raw compute."
  },

  {
    id: "cme295-lect4-q21",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about high-bandwidth memory (HBM) and static random-access memory (SRAM) on GPUs are correct?",
    options: [
      { text: "HBM is large but slower than on-chip memory.", isCorrect: true },
      { text: "SRAM is much faster but much smaller.", isCorrect: true },
      { text: "Using SRAM effectively can significantly speed up computation.", isCorrect: true },
      { text: "SRAM capacity is typically measured in gigabytes.", isCorrect: false },
    ],
    explanation:
      "GPUs combine large, slower HBM with small, fast SRAM. Optimizations often aim to keep computation within SRAM to reduce costly memory transfers."
  },

  {
    id: "cme295-lect4-q22",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about FlashAttention are correct?",
    options: [
      { text: "It reduces memory reads and writes during attention computation.", isCorrect: true },
      { text: "It uses tiling to fit intermediate results into fast on-chip memory.", isCorrect: true },
      { text: "It computes exact attention, not an approximation.", isCorrect: true },
      { text: "It changes the mathematical definition of self-attention.", isCorrect: false },
    ],
    explanation:
      "FlashAttention reorganizes computation to be IO-aware. The math is unchanged, but data movement is drastically reduced, yielding speed and memory benefits."
  },

  {
    id: "cme295-lect4-q23",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about recomputation strategies in training are correct?",
    options: [
      { text: "They trade additional compute for reduced memory usage.", isCorrect: true },
      { text: "Some activations are recomputed during the backward pass.", isCorrect: true },
      { text: "They can lower peak memory requirements.", isCorrect: true },
      { text: "They always increase overall training time.", isCorrect: false },
    ],
    explanation:
      "Recomputation saves memory by discarding activations and recomputing them later. With IO-efficient methods like FlashAttention, this can even reduce runtime."
  },

  {
    id: "cme295-lect4-q24",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about floating point representations are correct?",
    options: [
      { text: "They allocate bits to represent sign, exponent, and mantissa.", isCorrect: true },
      { text: "Lower-precision formats use fewer bits.", isCorrect: true },
      { text: "Precision affects numerical granularity.", isCorrect: true },
      { text: "All floating point formats have identical dynamic range.", isCorrect: false },
    ],
    explanation:
      "Different floating point formats trade off precision and range. Fewer bits reduce memory and speed up computation but can introduce numerical error."
  },

  {
    id: "cme295-lect4-q25",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about mixed precision training are correct?",
    options: [
      { text: "Some computations use lower-precision arithmetic.", isCorrect: true },
      { text: "Model weights are often stored in higher precision.", isCorrect: true },
      { text: "Memory usage can be reduced significantly.", isCorrect: true },
      { text: "It requires changing the model architecture.", isCorrect: false },
    ],
    explanation:
      "Mixed precision training leverages low-precision arithmetic for speed and memory savings while keeping critical quantities, like weights, in higher precision to maintain stability."
  },

  // ============================================================
  // Q26–Q37: EXACTLY 2 TRUE
  // ============================================================

  {
    id: "cme295-lect4-q26",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements about pre-training objectives are correct?",
    options: [
      { text: "They typically involve predicting the next token.", isCorrect: true },
      { text: "They require labeled task-specific outputs.", isCorrect: false },
      { text: "They are self-supervised.", isCorrect: true },
      { text: "They always involve human annotation.", isCorrect: false },
    ],
    explanation:
      "Pre-training uses self-supervision: the text itself provides targets. This differs from supervised fine-tuning, which uses labeled input–output pairs."
  },

  {
    id: "cme295-lect4-q27",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about batch size effects in training are correct?",
    options: [
      { text: "Larger batch sizes increase activation memory.", isCorrect: true },
      { text: "Batch size has no effect on memory usage.", isCorrect: false },
      { text: "Very large batches can change optimization dynamics.", isCorrect: true },
      { text: "Batch size is unrelated to gradient computation.", isCorrect: false },
    ],
    explanation:
      "Batch size directly affects memory and can influence convergence behavior. Extremely large batches often require careful tuning of learning rates and optimizers."
  },

  {
    id: "cme295-lect4-q28",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about quantization are correct?",
    options: [
      { text: "It reduces numerical precision of stored values.", isCorrect: true },
      { text: "It can reduce memory footprint.", isCorrect: true },
      { text: "It always improves model accuracy.", isCorrect: false },
      { text: "It can increase computation speed on suitable hardware.", isCorrect: false },
    ],
    explanation:
      "Quantization trades precision for efficiency. While it often reduces memory and can improve speed, it may also degrade accuracy if applied carelessly."
  },

  {
    id: "cme295-lect4-q29",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about supervised fine-tuning (SFT) are correct?",
    options: [
      { text: "It uses labeled input–output pairs.", isCorrect: true },
      { text: "It modifies pre-trained model weights.", isCorrect: true },
      { text: "Loss is computed over both input and output tokens.", isCorrect: false },
      { text: "It replaces the pre-training objective entirely.", isCorrect: false },
    ],
    explanation:
      "SFT adapts a pre-trained model using labeled examples. Loss is applied only to the output portion, keeping the conditioning input fixed."
  },

  {
    id: "cme295-lect4-q30",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about instruction tuning are correct?",
    options: [
      { text: "It is a form of supervised fine-tuning.", isCorrect: true },
      { text: "It trains models to respond helpfully to user prompts.", isCorrect: true },
      { text: "It requires training on the entire pre-training corpus again.", isCorrect: false },
      { text: "It guarantees factual correctness of all outputs.", isCorrect: false },
    ],
    explanation:
      "Instruction tuning teaches models how to act as assistants. It improves helpfulness but does not guarantee correctness or eliminate hallucinations."
  },

  // ============================================================
  // Q31–Q65: continuation
  // Answer-pattern distribution in this block:
  // - Q31–Q42: EXACTLY 1 TRUE
  // - Q43–Q54: ALL TRUE
  // - Q55–Q65: EXACTLY 3 TRUE
  // ============================================================

  {
    id: "cme295-lect4-q31",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statement best characterizes greedy decoding in language model inference?",
    options: [
      { text: "It selects the most probable token at each step.", isCorrect: true },
      { text: "It maintains multiple candidate sequences simultaneously.", isCorrect: false },
      { text: "It samples tokens according to the output probability distribution.", isCorrect: false },
      { text: "It explicitly optimizes long-term sequence diversity.", isCorrect: false },
    ],
    explanation:
      "Greedy decoding deterministically chooses the highest-probability token at each step. While simple and fast, it often leads to repetitive or locally optimal outputs rather than globally coherent ones."
  },

  {
    id: "cme295-lect4-q32",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statement correctly describes the role of temperature in token sampling?",
    options: [
      { text: "It controls how peaked or flat the output probability distribution is.", isCorrect: true },
      { text: "It changes the model architecture during inference.", isCorrect: false },
      { text: "It guarantees more factually correct outputs.", isCorrect: false },
      { text: "It only affects beam search decoding.", isCorrect: false },
    ],
    explanation:
      "Temperature rescales logits before the softmax. Higher temperature increases randomness by flattening the distribution, while lower temperature makes outputs more deterministic."
  },

  {
    id: "cme295-lect4-q33",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statement correctly describes the key motivation behind key–value (KV) caching during autoregressive decoding?",
    options: [
      { text: "It avoids recomputing attention for previous tokens.", isCorrect: true },
      { text: "It compresses model parameters into lower precision.", isCorrect: false },
      { text: "It enables parallel decoding of future tokens.", isCorrect: false },
      { text: "It replaces the attention mechanism entirely.", isCorrect: false },
    ],
    explanation:
      "KV caching stores previously computed keys and values so they can be reused for subsequent decoding steps. This significantly reduces inference cost for long sequences."
  },

  {
    id: "cme295-lect4-q34",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statement correctly describes beam search decoding?",
    options: [
      { text: "It tracks multiple high-probability partial sequences at each step.", isCorrect: true },
      { text: "It always produces more diverse outputs than sampling.", isCorrect: false },
      { text: "It samples tokens independently at each step.", isCorrect: false },
      { text: "It removes the need for a language model probability distribution.", isCorrect: false },
    ],
    explanation:
      "Beam search keeps the top-k candidate sequences according to cumulative probability. It improves global sequence likelihood but can reduce diversity and increase computation."
  },

  {
    id: "cme295-lect4-q35",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statement correctly explains why mixture-of-experts (MoE) models can scale parameter counts efficiently?",
    options: [
      { text: "Only a subset of parameters is activated per input.", isCorrect: true },
      { text: "All experts are evaluated in parallel for every token.", isCorrect: false },
      { text: "They reduce training data requirements.", isCorrect: false },
      { text: "They eliminate the need for attention mechanisms.", isCorrect: false },
    ],
    explanation:
      "Sparse MoE models activate only a few experts per token, allowing total parameter counts to grow without proportionally increasing compute at inference time."
  },

  {
    id: "cme295-lect4-q36",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statement best describes the purpose of a gating network in mixture-of-experts models?",
    options: [
      { text: "It selects which experts process a given input.", isCorrect: true },
      { text: "It normalizes attention weights.", isCorrect: false },
      { text: "It replaces gradient-based optimization.", isCorrect: false },
      { text: "It enforces model sparsity during inference only.", isCorrect: false },
    ],
    explanation:
      "The gating network computes routing decisions that determine which experts are activated for each input token, directly influencing compute and communication patterns."
  },

  {
    id: "cme295-lect4-q37",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statement correctly characterizes why updating factual knowledge in a trained language model is difficult?",
    options: [
      { text: "Knowledge is distributed across many parameters.", isCorrect: true },
      { text: "Gradients cannot be computed after pre-training.", isCorrect: false },
      { text: "Transformer architectures forbid weight updates.", isCorrect: false },
      { text: "The tokenizer prevents new information from being added.", isCorrect: false },
    ],
    explanation:
      "Factual knowledge is encoded implicitly across many weights. Editing specific facts without degrading other capabilities is a challenging open research problem."
  },

  // ================= ALL TRUE =================

  {
    id: "cme295-lect4-q38",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements correctly describe supervised fine-tuning (SFT) data?",
    options: [
      { text: "It consists of input–output pairs.", isCorrect: true },
      { text: "It is much smaller than pre-training datasets.", isCorrect: true },
      { text: "It aims to align model behavior with desired outputs.", isCorrect: true },
      { text: "It typically focuses on higher-quality curated data.", isCorrect: true },
    ],
    explanation:
      "SFT uses carefully curated examples to shape model behavior. Compared to pre-training, the datasets are smaller but higher quality and task-focused."
  },

  {
    id: "cme295-lect4-q39",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements correctly describe instruction tuning?",
    options: [
      { text: "It trains models to follow natural language instructions.", isCorrect: true },
      { text: "It is a special case of supervised fine-tuning.", isCorrect: true },
      { text: "Loss is applied only to the generated response.", isCorrect: true },
      { text: "It improves usefulness in interactive settings.", isCorrect: true },
    ],
    explanation:
      "Instruction tuning conditions the model on user prompts and applies loss only to the response, making the model behave more like a helpful assistant."
  },

  {
    id: "cme295-lect4-q40",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements correctly describe evaluation challenges for large language models?",
    options: [
      { text: "Benchmarks can be sensitive to training data overlap.", isCorrect: true },
      { text: "Single metrics rarely capture all desired behaviors.", isCorrect: true },
      { text: "User preferences can be subjective and inconsistent.", isCorrect: true },
      { text: "Models can overfit to popular benchmarks.", isCorrect: true },
    ],
    explanation:
      "LLM evaluation is difficult because performance depends on data distribution, subjective preferences, and benchmark design. No single metric fully captures model quality."
  },

  {
    id: "cme295-lect4-q41",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements correctly describe Massive Multitask Language Understanding (MMLU)?",
    options: [
      { text: "It aggregates performance across many different tasks.", isCorrect: true },
      { text: "It aims to measure broad language understanding.", isCorrect: true },
      { text: "It is commonly used to compare general-purpose models.", isCorrect: true },
      { text: "It reflects only next-token prediction loss.", isCorrect: true },
    ],
    explanation:
      "MMLU evaluates models across diverse tasks to approximate general language competence. It goes beyond raw loss by testing downstream abilities."
  },

  // ================= EXACTLY 3 TRUE =================

  {
    id: "cme295-lect4-q42",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about preference-based evaluation are correct?",
    options: [
      { text: "It relies on pairwise comparisons between model outputs.", isCorrect: true },
      { text: "Human judgment often plays a role.", isCorrect: true },
      { text: "It can capture subjective notions of quality.", isCorrect: true },
      { text: "It is completely immune to manipulation.", isCorrect: false },
    ],
    explanation:
      "Preference-based evaluation uses human or model judgments to compare outputs. While flexible and intuitive, it can be biased or gamed under certain conditions."
  },

  {
    id: "cme295-lect4-q43",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about alignment in language models are correct?",
    options: [
      { text: "It refers to shaping model behavior toward human goals.", isCorrect: true },
      { text: "It typically occurs after pre-training.", isCorrect: true },
      { text: "It includes supervised fine-tuning and preference tuning.", isCorrect: true },
      { text: "It guarantees perfect safety and correctness.", isCorrect: false },
    ],
    explanation:
      "Alignment aims to make models helpful and safe. While SFT and preference tuning improve alignment, they do not eliminate all risks or errors."
  },

  {
    id: "cme295-lect4-q44",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about Low-Rank Adaptation (LoRA) are correct?",
    options: [
      { text: "It freezes the original pre-trained weights.", isCorrect: true },
      { text: "It introduces additional low-rank trainable matrices.", isCorrect: true },
      { text: "It significantly reduces the number of trainable parameters.", isCorrect: true },
      { text: "It requires retraining the entire model from scratch.", isCorrect: false },
    ],
    explanation:
      "LoRA fine-tunes models efficiently by learning low-rank updates while keeping base weights fixed, reducing memory and compute costs."
  },

  {
    id: "cme295-lect4-q45",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about quantized LoRA (QLoRA) are correct?",
    options: [
      { text: "Base model weights are stored in quantized form.", isCorrect: true },
      { text: "LoRA adapters are typically kept in higher precision.", isCorrect: true },
      { text: "It enables fine-tuning large models on limited hardware.", isCorrect: true },
      { text: "It eliminates numerical error entirely.", isCorrect: false },
    ],
    explanation:
      "QLoRA combines quantization with LoRA to drastically reduce memory usage while retaining training stability by keeping adapter weights in higher precision."
  },

  {
    id: "cme295-lect4-q46",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about NormalFloat 4 (NF4) quantization are correct?",
    options: [
      { text: "It assumes weights follow an approximately normal distribution.", isCorrect: true },
      { text: "It uses non-uniform quantization bins.", isCorrect: true },
      { text: "It improves memory efficiency for frozen weights.", isCorrect: true },
      { text: "It is identical to standard 4-bit linear quantization.", isCorrect: false },
    ],
    explanation:
      "NF4 is designed for normally distributed weights and uses quantiles rather than uniform bins, yielding better accuracy at very low bit-widths."
  },

  {
    id: "cme295-lect4-q47",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about batch size effects in LoRA training are correct?",
    options: [
      { text: "Very large batch sizes can hurt LoRA performance.", isCorrect: true },
      { text: "LoRA often uses higher learning rates than full fine-tuning.", isCorrect: true },
      { text: "Training dynamics differ from full-rank weight updates.", isCorrect: true },
      { text: "Batch size has no interaction with optimization behavior.", isCorrect: false },
    ],
    explanation:
      "Empirically, LoRA benefits from higher learning rates and smaller batch sizes. Its low-rank structure changes optimization dynamics compared to full fine-tuning."
  },

  {
    id: "cme295-lect4-q48",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements about mixed-precision training are correct?",
    options: [
      { text: "It uses different numerical precisions for different operations.", isCorrect: true },
      { text: "It can reduce memory usage and increase throughput.", isCorrect: true },
      { text: "Model weights are often maintained in higher precision.", isCorrect: true },
      { text: "It requires changing the loss function.", isCorrect: false },
    ],
    explanation:
      "Mixed-precision training balances speed and stability by using lower precision for most computations while preserving critical values in higher precision."
  },

  {
    id: "cme295-lect4-q49",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about FlashAttention recomputation are correct?",
    options: [
      { text: "Activations may be recomputed during the backward pass.", isCorrect: true },
      { text: "It trades additional computation for lower memory usage.", isCorrect: true },
      { text: "It can reduce overall runtime despite recomputation.", isCorrect: true },
      { text: "It requires approximating the attention operation.", isCorrect: false },
    ],
    explanation:
      "FlashAttention leverages fast on-chip memory to recompute activations cheaply, reducing memory traffic and sometimes even improving overall runtime."
  },

  {
    id: "cme295-lect4-q50",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about evaluation benchmarks are correct?",
    options: [
      { text: "They can shape model training incentives.", isCorrect: true },
      { text: "They may fail to capture real user satisfaction.", isCorrect: true },
      { text: "They are often domain-specific.", isCorrect: true },
      { text: "They fully eliminate the need for human evaluation.", isCorrect: false },
    ],
    explanation:
      "Benchmarks guide progress but are imperfect proxies for real-world usefulness. Human judgment remains important for many aspects of model quality."
  },

  {
    id: "cme295-lect4-q51",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about alignment data are correct?",
    options: [
      { text: "They often include safety-focused examples.", isCorrect: true },
      { text: "They aim to discourage harmful or unsafe outputs.", isCorrect: true },
      { text: "They are typically much smaller than pre-training data.", isCorrect: true },
      { text: "They are entirely generated without human input.", isCorrect: false },
    ],
    explanation:
      "Alignment datasets are curated to shape behavior and safety. While model-generated data can assist, human oversight is still common."
  },

  {
    id: "cme295-lect4-q52",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about preference tuning are correct?",
    options: [
      { text: "It optimizes models using preference signals.", isCorrect: true },
      { text: "It often follows supervised fine-tuning.", isCorrect: true },
      { text: "It can use pairwise ranking losses.", isCorrect: true },
      { text: "It replaces pre-training entirely.", isCorrect: false },
    ],
    explanation:
      "Preference tuning adjusts models using human or synthetic preference feedback, complementing supervised fine-tuning rather than replacing earlier stages."
  },

  {
    id: "cme295-lect4-q53",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about training–evaluation mismatch are correct?",
    options: [
      { text: "Training objectives may not align with user satisfaction.", isCorrect: true },
      { text: "Benchmarks can be optimized without improving real usefulness.", isCorrect: true },
      { text: "Evaluation depends on task distribution.", isCorrect: true },
      { text: "Loss minimization guarantees helpful behavior.", isCorrect: false },
    ],
    explanation:
      "Optimizing likelihood or benchmark scores does not guarantee helpfulness. Alignment and evaluation require additional signals beyond loss minimization."
  },

  {
    id: "cme295-lect4-q54",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about scaling laws are correct?",
    options: [
      { text: "They describe empirical relationships between model size, data, and loss.", isCorrect: true },
      { text: "They motivated increasing model and dataset sizes.", isCorrect: true },
      { text: "They suggest diminishing returns at fixed compute.", isCorrect: true },
      { text: "They provide exact guarantees for downstream task performance.", isCorrect: false },
    ],
    explanation:
      "Scaling laws capture smooth empirical trends but do not provide strict guarantees, especially for downstream or aligned behaviors."
  },

  {
    id: "cme295-lect4-q55",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about compute constraints in LLM training are correct?",
    options: [
      { text: "They motivate parallelism strategies.", isCorrect: true },
      { text: "They influence architecture and optimization choices.", isCorrect: true },
      { text: "They limit feasible model and dataset sizes.", isCorrect: true },
      { text: "They are irrelevant once a model is pre-trained.", isCorrect: false },
    ],
    explanation:
      "Compute constraints affect all stages of training and deployment. They drive innovation in parallelism, efficiency, and model design."
  },

  // ============================================================
  // Q56–Q100: FINAL BLOCK
  // ============================================================

  {
    id: "cme295-lect4-q56",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements correctly describe the purpose of mid-training in large language models?",
    options: [
      { text: "It adapts the data distribution while keeping the same next-token objective.", isCorrect: true },
      { text: "It occurs after pre-training and before supervised fine-tuning.", isCorrect: true },
      { text: "It targets domains closer to downstream use cases.", isCorrect: true },
      { text: "It replaces the need for fine-tuning.", isCorrect: false },
    ],
    explanation:
      "Mid-training keeps the same objective as pre-training but changes the data mixture. It helps align representations toward domains of interest before task-specific fine-tuning."
  },

  {
    id: "cme295-lect4-q57",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about supervised fine-tuning loss computation are correct?",
    options: [
      { text: "Loss is applied only to the generated output tokens.", isCorrect: true },
      { text: "Input tokens are used as conditioning context.", isCorrect: true },
      { text: "Teacher forcing is applied to the entire input sequence.", isCorrect: false },
      { text: "The objective differs from pre-training despite using cross-entropy.", isCorrect: true },
    ],
    explanation:
      "In supervised fine-tuning, the model conditions on the input and is trained only on the response tokens. Although the loss function is still cross-entropy, the masking makes the objective meaningfully different from pre-training."
  },

  {
    id: "cme295-lect4-q58",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements correctly describe instruction tuning?",
    options: [
      { text: "It teaches the model to respond helpfully to prompts.", isCorrect: true },
      { text: "It is a form of supervised fine-tuning.", isCorrect: true },
      { text: "It uses curated instruction–response pairs.", isCorrect: true },
      { text: "It requires retraining from random initialization.", isCorrect: false },
    ],
    explanation:
      "Instruction tuning is supervised fine-tuning focused on user-facing tasks. It adapts a pre-trained model rather than training from scratch."
  },

  {
    id: "cme295-lect4-q59",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about data mixture in supervised fine-tuning are correct?",
    options: [
      { text: "It often includes many task categories.", isCorrect: true },
      { text: "It may combine human-written and model-generated examples.", isCorrect: true },
      { text: "It is usually much smaller than pre-training data.", isCorrect: true },
      { text: "It must match inference prompts exactly.", isCorrect: false },
    ],
    explanation:
      "SFT data mixtures span many task types and are often partially synthetic. Exact matching to inference prompts is unnecessary as long as the distribution is reasonably aligned."
  },

  {
    id: "cme295-lect4-q60",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about memorization in large language models are correct?",
    options: [
      { text: "Models can reproduce rare training examples verbatim.", isCorrect: true },
      { text: "Memorization risk increases with repeated exposure to data.", isCorrect: true },
      { text: "Sampling temperature can affect surface-level repetition.", isCorrect: true },
      { text: "Memorization is completely eliminated by fine-tuning.", isCorrect: false },
    ],
    explanation:
      "Large models can memorize rare or repeated samples. Fine-tuning and decoding choices can reduce but not eliminate this risk."
  },

  {
    id: "cme295-lect4-q61",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about benchmark evaluation are correct?",
    options: [
      { text: "Benchmarks approximate specific capabilities.", isCorrect: true },
      { text: "High benchmark scores guarantee user satisfaction.", isCorrect: false },
      { text: "Training data overlap can inflate benchmark results.", isCorrect: true },
      { text: "Benchmarks evolve over time.", isCorrect: true },
    ],
    explanation:
      "Benchmarks measure targeted abilities but are imperfect proxies for usefulness. Overlap and gaming can distort results, motivating continual benchmark updates."
  },

  {
    id: "cme295-lect4-q62",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about Massive Multitask Language Understanding (MMLU) are correct?",
    options: [
      { text: "It aggregates results across many tasks.", isCorrect: true },
      { text: "It evaluates downstream capabilities rather than raw loss.", isCorrect: true },
      { text: "It reflects broad language competence.", isCorrect: true },
      { text: "It directly measures training compute efficiency.", isCorrect: false },
    ],
    explanation:
      "MMLU focuses on task performance rather than likelihood. It does not directly capture training efficiency or compute usage."
  },

  {
    id: "cme295-lect4-q63",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about preference-based evaluation are correct?",
    options: [
      { text: "Users compare outputs pairwise.", isCorrect: true },
      { text: "It captures subjective quality signals.", isCorrect: true },
      { text: "Early comparisons can bias rankings.", isCorrect: true },
      { text: "It is immune to adversarial behavior.", isCorrect: false },
    ],
    explanation:
      "Preference-based systems reflect human judgments but are sensitive to sampling, bias, and strategic manipulation."
  },

  {
    id: "cme295-lect4-q64",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about alignment in language models are correct?",
    options: [
      { text: "It aims to make models helpful and harmless.", isCorrect: true },
      { text: "It includes supervised fine-tuning and preference tuning.", isCorrect: true },
      { text: "It occurs after pre-training.", isCorrect: true },
      { text: "It guarantees correct behavior in all cases.", isCorrect: false },
    ],
    explanation:
      "Alignment shapes behavior but does not eliminate errors or misuse. It is a post-pretraining process involving multiple techniques."
  },

  {
    id: "cme295-lect4-q65",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about preference tuning are correct?",
    options: [
      { text: "It uses relative judgments rather than absolute labels.", isCorrect: true },
      { text: "It often follows supervised fine-tuning.", isCorrect: true },
      { text: "It can optimize for user satisfaction.", isCorrect: true },
      { text: "It replaces pre-training.", isCorrect: false },
    ],
    explanation:
      "Preference tuning refines behavior using rankings or comparisons. It complements but does not replace earlier training stages."
  },

  {
    id: "cme295-lect4-q66",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about Low-Rank Adaptation (LoRA) are correct?",
    options: [
      { text: "Base model weights remain frozen.", isCorrect: true },
      { text: "Only low-rank matrices are trained.", isCorrect: true },
      { text: "It reduces memory and compute cost.", isCorrect: true },
      { text: "It requires modifying the model architecture.", isCorrect: false },
    ],
    explanation:
      "LoRA adds trainable low-rank matrices without altering the base architecture. This yields large efficiency gains during fine-tuning."
  },

  {
    id: "cme295-lect4-q67",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about the LoRA rank hyperparameter are correct?",
    options: [
      { text: "It controls the capacity of the adaptation.", isCorrect: true },
      { text: "Lower rank reduces trainable parameters.", isCorrect: true },
      { text: "Higher rank always improves performance.", isCorrect: false },
      { text: "It is typically much smaller than the original weight dimension.", isCorrect: true },
    ],
    explanation:
      "The rank trades off capacity and efficiency. Increasing it can help but often yields diminishing returns."
  },

  {
    id: "cme295-lect4-q68",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about LoRA training dynamics are correct?",
    options: [
      { text: "Higher learning rates are often used.", isCorrect: true },
      { text: "Very large batch sizes can degrade performance.", isCorrect: true },
      { text: "Optimization differs from full fine-tuning.", isCorrect: true },
      { text: "Training is identical to dense fine-tuning.", isCorrect: false },
    ],
    explanation:
      "Empirical results show LoRA benefits from different hyperparameters. Its low-rank structure alters optimization behavior."
  },

  {
    id: "cme295-lect4-q69",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about where LoRA is applied in transformers are correct?",
    options: [
      { text: "It can be applied to attention projections.", isCorrect: true },
      { text: "It can be applied to feedforward layers.", isCorrect: true },
      { text: "Feedforward placement often yields strong gains.", isCorrect: true },
      { text: "It must be applied to every layer to work.", isCorrect: false },
    ],
    explanation:
      "LoRA can be inserted in multiple components. Empirical work shows feedforward layers are especially effective."
  },

  {
    id: "cme295-lect4-q70",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about quantized LoRA (QLoRA) are correct?",
    options: [
      { text: "Frozen base weights are quantized.", isCorrect: true },
      { text: "LoRA adapters remain high precision.", isCorrect: true },
      { text: "It enables fine-tuning on limited hardware.", isCorrect: true },
      { text: "It eliminates quantization error.", isCorrect: false },
    ],
    explanation:
      "QLoRA combines aggressive quantization with LoRA adapters to drastically reduce memory usage while maintaining training stability."
  },

  {
    id: "cme295-lect4-q71",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about NormalFloat 4 (NF4) quantization are correct?",
    options: [
      { text: "It assumes normally distributed weights.", isCorrect: true },
      { text: "It uses quantiles rather than uniform bins.", isCorrect: true },
      { text: "It is designed for frozen model weights.", isCorrect: true },
      { text: "It is identical to standard 4-bit quantization.", isCorrect: false },
    ],
    explanation:
      "NF4 exploits weight distribution structure to minimize error at very low precision. It differs fundamentally from uniform quantization."
  },

  {
    id: "cme295-lect4-q72",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about floating-point formats are correct?",
    options: [
      { text: "They trade precision for range.", isCorrect: true },
      { text: "Brain Float 16 keeps a larger exponent range.", isCorrect: true },
      { text: "Lower precision reduces memory usage.", isCorrect: true },
      { text: "All formats behave identically numerically.", isCorrect: false },
    ],
    explanation:
      "Different formats balance range and granularity differently. These trade-offs directly affect training stability and efficiency."
  },

  {
    id: "cme295-lect4-q73",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about mixed precision training are correct?",
    options: [
      { text: "Weights are often kept in higher precision.", isCorrect: true },
      { text: "Forward and backward passes may use lower precision.", isCorrect: true },
      { text: "It can improve throughput and reduce memory.", isCorrect: true },
      { text: "It removes the need for numerical care.", isCorrect: false },
    ],
    explanation:
      "Mixed precision exploits hardware capabilities but still requires care to avoid instability and overflow."
  },

  {
    id: "cme295-lect4-q74",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about quantization ranges are correct?",
    options: [
      { text: "Range affects representable values.", isCorrect: true },
      { text: "Zero-point and scale can define mapping.", isCorrect: true },
      { text: "Poor range selection increases error.", isCorrect: true },
      { text: "Range choice is irrelevant to performance.", isCorrect: false },
    ],
    explanation:
      "Quantization accuracy depends strongly on how ranges are chosen. Poor calibration can severely degrade model quality."
  },

  {
    id: "cme295-lect4-q75",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about FlashAttention are correct?",
    options: [
      { text: "It reduces high-bandwidth memory traffic.", isCorrect: true },
      { text: "It uses tiling into fast on-chip memory.", isCorrect: true },
      { text: "It computes exact attention.", isCorrect: true },
      { text: "It approximates softmax.", isCorrect: false },
    ],
    explanation:
      "FlashAttention reorganizes computation without changing the mathematical result. Its speedups come from IO efficiency."
  },

  {
    id: "cme295-lect4-q76",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about recomputation in FlashAttention are correct?",
    options: [
      { text: "Some activations are discarded during the forward pass.", isCorrect: true },
      { text: "They are recomputed during the backward pass.", isCorrect: true },
      { text: "This can reduce peak memory usage.", isCorrect: true },
      { text: "It always increases runtime.", isCorrect: false },
    ],
    explanation:
      "Recomputation trades compute for memory. With fast kernels, it can reduce runtime by minimizing memory traffic."
  },

  {
    id: "cme295-lect4-q77",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about GPU memory hierarchy are correct?",
    options: [
      { text: "High-bandwidth memory is larger but slower than SRAM.", isCorrect: true },
      { text: "Static random-access memory is close to compute units.", isCorrect: true },
      { text: "Memory access often dominates runtime.", isCorrect: true },
      { text: "All GPU memory has equal speed.", isCorrect: false },
    ],
    explanation:
      "GPUs rely on multiple memory tiers. Efficient kernels minimize access to slower memory."
  },

  {
    id: "cme295-lect4-q78",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about data parallelism are correct?",
    options: [
      { text: "Each device processes different data.", isCorrect: true },
      { text: "Gradients must be synchronized.", isCorrect: true },
      { text: "Communication introduces overhead.", isCorrect: true },
      { text: "Models are partitioned across devices.", isCorrect: false },
    ],
    explanation:
      "Data parallelism replicates the model and splits data. Synchronization costs limit scaling."
  },

  {
    id: "cme295-lect4-q79",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about Zero Redundancy Optimization (ZeRO) are correct?",
    options: [
      { text: "It shards optimizer states.", isCorrect: true },
      { text: "It can shard gradients and parameters.", isCorrect: true },
      { text: "It reduces per-device memory.", isCorrect: true },
      { text: "It removes communication entirely.", isCorrect: false },
    ],
    explanation:
      "ZeRO reduces memory duplication but increases coordination between devices."
  },

  {
    id: "cme295-lect4-q80",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about model parallelism are correct?",
    options: [
      { text: "It splits model computation across devices.", isCorrect: true },
      { text: "It allows training models larger than one GPU.", isCorrect: true },
      { text: "It can increase communication complexity.", isCorrect: true },
      { text: "It duplicates all parameters everywhere.", isCorrect: false },
    ],
    explanation:
      "Model parallelism partitions the model itself. This enables scale but adds coordination overhead."
  },

  {
    id: "cme295-lect4-q81",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about pipeline parallelism are correct?",
    options: [
      { text: "Different layers are assigned to different devices.", isCorrect: true },
      { text: "It can improve memory utilization.", isCorrect: true },
      { text: "Pipeline bubbles can reduce efficiency.", isCorrect: true },
      { text: "All layers execute simultaneously.", isCorrect: false },
    ],
    explanation:
      "Pipeline parallelism trades latency and scheduling complexity for memory scalability."
  },

  {
    id: "cme295-lect4-q82",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about expert parallelism are correct?",
    options: [
      { text: "Experts can reside on different devices.", isCorrect: true },
      { text: "Only selected experts run per token.", isCorrect: true },
      { text: "Routing affects communication.", isCorrect: true },
      { text: "All experts are always active.", isCorrect: false },
    ],
    explanation:
      "Expert parallelism exploits sparsity but introduces routing and load-balancing challenges."
  },

  {
    id: "cme295-lect4-q83",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about scaling laws are correct?",
    options: [
      { text: "Performance improves predictably with scale.", isCorrect: true },
      { text: "Model size, data size, and compute interact.", isCorrect: true },
      { text: "Optimal scaling depends on compute budget.", isCorrect: true },
      { text: "Architecture choice dominates scaling behavior.", isCorrect: false },
    ],
    explanation:
      "Empirical results show scale dominates architecture choice within common transformer families."
  },

  {
    id: "cme295-lect4-q84",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about compute-optimal training are correct?",
    options: [
      { text: "It balances parameters and data.", isCorrect: true },
      { text: "Undertraining wastes model capacity.", isCorrect: true },
      { text: "Overtraining small models wastes compute.", isCorrect: true },
      { text: "It eliminates the need for scaling experiments.", isCorrect: false },
    ],
    explanation:
      "Compute-optimal scaling guides design but still relies on empirical validation."
  },

  {
    id: "cme295-lect4-q85",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements about knowledge cutoff are correct?",
    options: [
      { text: "It reflects the latest training data date.", isCorrect: true },
      { text: "Models lack direct knowledge beyond it.", isCorrect: true },
      { text: "It is usually documented.", isCorrect: true },
      { text: "It can be bypassed without updating weights.", isCorrect: false },
    ],
    explanation:
      "Knowledge cutoff limits what the model can know intrinsically. External tools are required to overcome it."
  },

  {
    id: "cme295-lect4-q86",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about knowledge editing are correct?",
    options: [
      { text: "It is difficult without side effects.", isCorrect: true },
      { text: "Knowledge is distributed across parameters.", isCorrect: true },
      { text: "Local edits can cause global regressions.", isCorrect: true },
      { text: "It is a solved problem.", isCorrect: false },
    ],
    explanation:
      "Editing model knowledge remains an open challenge due to entangled representations."
  },

  {
    id: "cme295-lect4-q87",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about inference sampling are correct?",
    options: [
      { text: "Temperature controls randomness.", isCorrect: true },
      { text: "Sampling explores lower-probability tokens.", isCorrect: true },
      { text: "Sampling guarantees factual correctness.", isCorrect: false },
      { text: "Sampling increases output diversity.", isCorrect: true },
    ],
    explanation:
      "Sampling trades determinism for diversity. It does not ensure correctness."
  },

  {
    id: "cme295-lect4-q88",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about greedy decoding are correct?",
    options: [
      { text: "It selects the highest-probability token.", isCorrect: true },
      { text: "It is deterministic.", isCorrect: true },
      { text: "It can lead to repetitive outputs.", isCorrect: true },
      { text: "It maximizes sequence-level diversity.", isCorrect: false },
    ],
    explanation:
      "Greedy decoding is fast and simple but often suboptimal for long sequences."
  },

  {
    id: "cme295-lect4-q89",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about beam search are correct?",
    options: [
      { text: "It tracks multiple candidate sequences.", isCorrect: true },
      { text: "It approximates global likelihood maximization.", isCorrect: true },
      { text: "It often reduces diversity.", isCorrect: true },
      { text: "It samples stochastically.", isCorrect: false },
    ],
    explanation:
      "Beam search balances exploration and likelihood but is deterministic."
  },

  {
    id: "cme295-lect4-q90",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about key–value caching are correct?",
    options: [
      { text: "It avoids recomputing past attention.", isCorrect: true },
      { text: "It reduces inference cost for long sequences.", isCorrect: true },
      { text: "It increases memory usage.", isCorrect: true },
      { text: "It affects training loss.", isCorrect: false },
    ],
    explanation:
      "KV caching trades memory for speed during inference. It does not affect training objectives."
  },

  {
    id: "cme295-lect4-q91",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about training cost are correct?",
    options: [
      { text: "Pre-training dominates total cost.", isCorrect: true },
      { text: "Fine-tuning is comparatively cheap.", isCorrect: true },
      { text: "Hardware efficiency matters significantly.", isCorrect: true },
      { text: "Cost is independent of data size.", isCorrect: false },
    ],
    explanation:
      "Compute cost scales strongly with data and model size. Optimization techniques target this bottleneck."
  },

  {
    id: "cme295-lect4-q92",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about environmental impact are correct?",
    options: [
      { text: "Training consumes significant energy.", isCorrect: true },
      { text: "Reporting carbon cost is becoming common.", isCorrect: true },
      { text: "Efficiency improvements reduce impact.", isCorrect: true },
      { text: "Environmental cost is negligible.", isCorrect: false },
    ],
    explanation:
      "Energy use is a growing concern in large-scale training. Efficiency directly affects sustainability."
  },

  {
    id: "cme295-lect4-q93",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about evaluation mismatch are correct?",
    options: [
      { text: "Benchmarks may not reflect real use.", isCorrect: true },
      { text: "Optimization can overfit metrics.", isCorrect: true },
      { text: "User satisfaction is multifaceted.", isCorrect: true },
      { text: "One score captures everything.", isCorrect: false },
    ],
    explanation:
      "Evaluation requires multiple perspectives. Single metrics rarely suffice."
  },

  {
    id: "cme295-lect4-q94",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about user preference evaluation are correct?",
    options: [
      { text: "Users may value style over correctness.", isCorrect: true },
      { text: "Preferences vary across populations.", isCorrect: true },
      { text: "Preference signals can conflict with safety.", isCorrect: true },
      { text: "Preferences are always aligned with truth.", isCorrect: false },
    ],
    explanation:
      "Preference-based metrics reflect subjective tastes, which can conflict with factuality or safety goals."
  },

  {
    id: "cme295-lect4-q95",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about assistant behavior are correct?",
    options: [
      { text: "Helpfulness is shaped during fine-tuning.", isCorrect: true },
      { text: "Pre-training alone does not produce assistants.", isCorrect: true },
      { text: "Safety behaviors can be learned.", isCorrect: true },
      { text: "Assistants emerge without supervision.", isCorrect: false },
    ],
    explanation:
      "Assistant behavior arises from post-training alignment, not from raw next-token prediction."
  },

  {
    id: "cme295-lect4-q96",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about rejection behavior in models are correct?",
    options: [
      { text: "It can be learned via training data.", isCorrect: true },
      { text: "It may reduce user satisfaction.", isCorrect: true },
      { text: "It supports safety goals.", isCorrect: true },
      { text: "It is implemented purely via rules.", isCorrect: false },
    ],
    explanation:
      "Rejection behavior is often learned, not rule-based. It balances safety against usability."
  },

  {
    id: "cme295-lect4-q97",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about training data reuse are correct?",
    options: [
      { text: "High-quality datasets are reused across models.", isCorrect: true },
      { text: "Curation cost is amortized.", isCorrect: true },
      { text: "Reuse guarantees perfect alignment.", isCorrect: false },
      { text: "Data quality matters more than size in SFT.", isCorrect: true },
    ],
    explanation:
      "SFT emphasizes quality over scale. Reuse helps but does not solve alignment fully."
  },

  {
    id: "cme295-lect4-q98",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about distribution shift are correct?",
    options: [
      { text: "Inference prompts may differ from training prompts.", isCorrect: true },
      { text: "Shift affects generalization.", isCorrect: true },
      { text: "Better data coverage reduces risk.", isCorrect: true },
      { text: "Distribution shift is irrelevant in LLMs.", isCorrect: false },
    ],
    explanation:
      "Mismatch between training and inference distributions is a core generalization challenge."
  },

  {
    id: "cme295-lect4-q99",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about learning rate choice in fine-tuning are correct?",
    options: [
      { text: "It affects convergence speed.", isCorrect: true },
      { text: "LoRA often uses higher learning rates.", isCorrect: true },
      { text: "Too high values can destabilize training.", isCorrect: true },
      { text: "Learning rate is irrelevant.", isCorrect: false },
    ],
    explanation:
      "Learning rate is a critical hyperparameter, especially in low-rank adaptation."
  },

  {
    id: "cme295-lect4-q100",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about the overall LLM training pipeline are correct?",
    options: [
      { text: "Pre-training learns general representations.", isCorrect: true },
      { text: "Fine-tuning aligns models to tasks.", isCorrect: true },
      { text: "Preference tuning refines behavior.", isCorrect: true },
      { text: "All stages have equal cost.", isCorrect: false },
    ],
    explanation:
      "LLM training is staged. Pre-training is dominant in cost, while later stages refine usefulness and alignment."
  },

];
