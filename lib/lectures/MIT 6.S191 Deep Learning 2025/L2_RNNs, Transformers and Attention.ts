import { Question } from "../../quiz";

export const MIT6S191_L2_DeepSequenceModelingQuestions: Question[] = [
  {
    id: "mit6s191-l2-q01",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which of the following are examples of sequential data?",
    options: [
      { text: "Audio waveforms sampled over time.", isCorrect: true },
      {
        text: "A sentence represented as an ordered list of words.",
        isCorrect: true,
      },
      { text: "Stock prices indexed by timestamp.", isCorrect: true },
      { text: "Protein sequences composed of amino acids.", isCorrect: true },
    ],
    explanation:
      "Sequential data consists of elements ordered in time or position, where order matters. Audio, text, financial time series, and biological sequences all exhibit temporal or positional dependence and are classic examples of sequence modeling tasks.",
  },

  {
    id: "mit6s191-l2-q02",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Consider a feed-forward neural network applied independently at each time step: \\( \\hat{y}_t = f(x_t) \\). Which statement is correct?",
    options: [
      {
        text: "The prediction \\( \\hat{y}_t \\) depends only on \\( x_t \\).",
        isCorrect: true,
      },
      {
        text: "The model implicitly captures long-term dependencies across time steps.",
        isCorrect: false,
      },
      {
        text: "The model maintains an internal memory state across time.",
        isCorrect: false,
      },
      {
        text: "The model reuses information from previous inputs unless explicitly designed to.",
        isCorrect: false,
      },
    ],
    explanation:
      "When applying a feed-forward network independently at each time step, predictions depend only on the current input. There is no built-in mechanism for memory or recurrence, so temporal dependencies are ignored.",
  },

  {
    id: "mit6s191-l2-q03",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "In a recurrent neural network (RNN), the hidden state is updated as \\( h_t = f_W(x_t, h_{t-1}) \\). Which statements are correct?",
    options: [
      {
        text: "The hidden state introduces a notion of memory.",
        isCorrect: true,
      },
      {
        text: "The same parameters \\( W \\) are reused at every time step.",
        isCorrect: true,
      },
      {
        text: "The recurrence defines a dependence across time steps.",
        isCorrect: true,
      },
      {
        text: "Each time step does not use a completely independent set of weights.",
        isCorrect: true,
      },
    ],
    explanation:
      "The recurrence relation allows the model to maintain memory via \\( h_t \\). Importantly, the same weight matrices are reused at every time step, enabling parameter sharing and allowing the model to generalize across sequence positions.",
  },

  {
    id: "mit6s191-l2-q04",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about the RNN forward pass are correct?",
    options: [
      {
        text: "The hidden state is typically initialized (e.g., to zeros) before processing a sequence.",
        isCorrect: true,
      },
      {
        text: "At each time step, both a new hidden state and an output can be computed.",
        isCorrect: true,
      },
      {
        text: "The hidden state update depends on both \\( x_t \\) and \\( h_{t-1} \\).",
        isCorrect: true,
      },
      {
        text: "The output at time \\( t \\) does not have to be used as input at time \\( t+1 \\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The hidden state is often initialized to zeros or learned values. At each step, the RNN updates its state and may produce an output. However, the output is not necessarily fed back as input unless explicitly designed (e.g., in autoregressive generation).",
  },

  {
    id: "mit6s191-l2-q05",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "In backpropagation through time (BPTT), which statements are correct?",
    options: [
      {
        text: "The total loss is typically computed as a sum over time steps.",
        isCorrect: true,
      },
      {
        text: "Gradients are propagated backward through each unrolled time step.",
        isCorrect: true,
      },
      {
        text: "Repeated multiplication of Jacobians can cause vanishing or exploding gradients.",
        isCorrect: true,
      },
      {
        text: "BPTT does not eliminate the need for gradient-based optimization.",
        isCorrect: true,
      },
    ],
    explanation:
      "BPTT unrolls the network across time and propagates gradients backward through each step. Because weights are reused, gradients involve repeated multiplications, which can cause vanishing or exploding gradients. Optimization is still performed using gradient-based methods.",
  },

  {
    id: "mit6s191-l2-q06",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statements about vanishing gradients in RNNs are correct?",
    options: [
      {
        text: "They arise from repeated multiplication of small derivative values.",
        isCorrect: true,
      },
      {
        text: "They make it difficult to learn long-term dependencies.",
        isCorrect: true,
      },
      {
        text: "They can prevent early time steps from receiving meaningful gradient updates.",
        isCorrect: true,
      },
      {
        text: "They do not occur only in networks without nonlinearities.",
        isCorrect: true,
      },
    ],
    explanation:
      "Vanishing gradients occur when repeated multiplications of values less than one shrink gradients toward zero. This prevents effective learning of long-range dependencies because early steps receive negligible updates. Nonlinearities often contribute to this effect rather than prevent it.",
  },

  {
    id: "mit6s191-l2-q07",
    chapter: 2,
    difficulty: "medium",
    prompt: "Why are Long Short-Term Memory (LSTM) networks used?",
    options: [
      {
        text: "They introduce gating mechanisms to regulate information flow.",
        isCorrect: true,
      },
      {
        text: "They help mitigate vanishing gradient issues.",
        isCorrect: true,
      },
      {
        text: "They improve the ability to capture long-term dependencies.",
        isCorrect: true,
      },
      {
        text: "They do not remove the need for recurrence entirely in every sequence model.",
        isCorrect: true,
      },
    ],
    explanation:
      "LSTMs extend RNNs with gating mechanisms that regulate what information is kept or discarded. This helps preserve gradients and model long-term dependencies. However, LSTMs are still recurrent models and do not remove recurrence.",
  },

  {
    id: "mit6s191-l2-q08",
    chapter: 2,
    difficulty: "easy",
    prompt: "What is the purpose of word embeddings?",
    options: [
      { text: "To map discrete tokens to numerical vectors.", isCorrect: true },
      {
        text: "To enable neural networks to operate on language inputs.",
        isCorrect: true,
      },
      {
        text: "To place semantically similar words near each other in vector space.",
        isCorrect: true,
      },
      {
        text: "To guarantee perfect semantic understanding.",
        isCorrect: false,
      },
    ],
    explanation:
      "Embeddings convert discrete tokens into continuous vectors that neural networks can process. Learned embeddings often cluster similar words together. However, they do not guarantee perfect semantic understanding.",
  },

  {
    id: "mit6s191-l2-q09",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about one-hot encodings are correct?",
    options: [
      {
        text: "They represent each word as a sparse vector with a single 1.",
        isCorrect: true,
      },
      {
        text: "Their dimensionality equals the vocabulary size.",
        isCorrect: true,
      },
      { text: "They encode semantic similarity directly.", isCorrect: false },
      {
        text: "They are typically memory-efficient for large vocabularies.",
        isCorrect: false,
      },
    ],
    explanation:
      "One-hot encodings are sparse vectors with one active dimension. Their size grows with vocabulary size, making them inefficient for large vocabularies. They do not encode semantic similarity — all words are equidistant.",
  },

  {
    id: "mit6s191-l2-q10",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statements about RNN limitations are correct?",
    options: [
      {
        text: "The hidden state has variable dimensionality that automatically grows with sequence length.",
        isCorrect: false,
      },
      {
        text: "Processing is inherently sequential and difficult to parallelize.",
        isCorrect: true,
      },
      {
        text: "Long-term memory capacity can be bottlenecked by state size.",
        isCorrect: true,
      },
      {
        text: "RNNs can perfectly retain arbitrary-length context without degradation.",
        isCorrect: false,
      },
    ],
    explanation:
      "The hidden state is a fixed-length vector, limiting capacity. Because each time step depends on the previous one, computation is sequential and difficult to parallelize. Long-term dependencies are difficult to retain reliably.",
  },

  {
    id: "mit6s191-l2-q11",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which tasks are examples of sequence modeling?",
    options: [
      { text: "Sentiment classification from a sentence.", isCorrect: true },
      {
        text: "Machine translation is not a sequence modeling task.",
        isCorrect: false,
      },
      { text: "Next-word prediction.", isCorrect: true },
      {
        text: "Predicting a static image label without temporal context.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sequence modeling tasks involve ordered data. Sentiment classification, translation, and next-word prediction depend on word order. Static image classification without temporal structure is not inherently sequential.",
  },

  {
    id: "mit6s191-l2-q12",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "In scaled dot-product attention, similarity between query and key is computed as \\( \\frac{QK^T}{\\sqrt{d_k}} \\). Which statements are correct?",
    options: [
      {
        text: "The dot product measures only vector length, not similarity.",
        isCorrect: false,
      },
      {
        text: "Scaling by \\( \\sqrt{d_k} \\) stabilizes gradients.",
        isCorrect: true,
      },
      {
        text: "The result is typically passed through a softmax.",
        isCorrect: true,
      },
      {
        text: "The dot product eliminates the need for normalization.",
        isCorrect: false,
      },
    ],
    explanation:
      "The dot product provides a similarity score. Scaling prevents large magnitudes in high dimensions, stabilizing training. A softmax converts similarities into normalized attention weights.",
  },

  {
    id: "mit6s191-l2-q13",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements describe the role of positional embeddings in Transformers?",
    options: [
      {
        text: "They erase position information entirely.",
        isCorrect: false,
      },
      { text: "They compensate for the lack of recurrence.", isCorrect: true },
      {
        text: "They allow the model to distinguish word order.",
        isCorrect: true,
      },
      { text: "They remove the need for token embeddings.", isCorrect: false },
    ],
    explanation:
      "Transformers process inputs in parallel, so positional embeddings inject order information. They allow the model to reason about sequence structure. However, token embeddings are still necessary.",
  },

  {
    id: "mit6s191-l2-q14",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "In self-attention, which components are learned linear projections of the input embeddings?",
    options: [
      {
        text: "Query matrix \\( Q \\) is the only learned matrix used in attention.",
        isCorrect: false,
      },
      { text: "Key matrix \\( K \\).", isCorrect: true },
      { text: "Value matrix \\( V \\).", isCorrect: true },
      { text: "Softmax normalization.", isCorrect: false },
    ],
    explanation:
      "Queries, keys, and values are produced by learned linear projections of the input embeddings. The softmax is a fixed nonlinear operation and is not learned.",
  },

  {
    id: "mit6s191-l2-q15",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about multi-head attention are correct?",
    options: [
      {
        text: "It allows the model to attend to different aspects of the input.",
        isCorrect: true,
      },
      {
        text: "Each head must share the same learned projections.",
        isCorrect: false,
      },
      { text: "Heads operate in parallel.", isCorrect: true },
      {
        text: "All heads must produce identical attention patterns.",
        isCorrect: false,
      },
    ],
    explanation:
      "Multi-head attention uses multiple sets of projections, enabling the model to learn diverse relational patterns. Each head operates independently in parallel and can focus on different features.",
  },

  {
    id: "mit6s191-l2-q16",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statement best characterizes attention?",
    options: [
      {
        text: "Attention computes weighted combinations of value vectors.",
        isCorrect: true,
      },
      { text: "Attention requires recurrence.", isCorrect: false },
      {
        text: "Attention processes tokens strictly sequentially.",
        isCorrect: false,
      },
      { text: "Attention cannot be parallelized.", isCorrect: false },
    ],
    explanation:
      "Attention computes a weighted sum of value vectors based on similarity between queries and keys. Unlike RNNs, attention can operate in parallel across sequence positions.",
  },

  {
    id: "mit6s191-l2-q17",
    chapter: 2,
    difficulty: "hard",
    prompt: "Why are Transformers more parallelizable than RNNs?",
    options: [
      { text: "They remove explicit recurrence.", isCorrect: true },
      {
        text: "Tokens cannot be processed simultaneously in Transformer attention.",
        isCorrect: false,
      },
      {
        text: "Attention allows pairwise comparisons without time-step dependence.",
        isCorrect: true,
      },
      { text: "They avoid matrix multiplications entirely.", isCorrect: false },
    ],
    explanation:
      "Transformers do not rely on sequential hidden state updates. Instead, attention allows all tokens to interact in parallel. However, they still rely heavily on matrix multiplications.",
  },

  {
    id: "mit6s191-l2-q18",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about next-word prediction are correct?",
    options: [
      {
        text: "It can be framed as predicting \\( x_{t+1} \\) given \\( x_1, \\dots, x_t \\).",
        isCorrect: true,
      },
      {
        text: "It is not a foundational objective for language models.",
        isCorrect: false,
      },
      {
        text: "It can be implemented only with Transformers and not with RNNs.",
        isCorrect: false,
      },
      { text: "It requires labeled sentiment annotations.", isCorrect: false },
    ],
    explanation:
      "Next-word prediction uses previous tokens to predict the next one. It is the core training objective for many large language models. It does not require sentiment labels.",
  },

  {
    id: "mit6s191-l2-q19",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements about the attention weight matrix \\( A = \\text{softmax}\\left( \\frac{QK^T}{\\sqrt{d_k}} \\right) \\) are correct?",
    options: [
      {
        text: "Each row of \\( A \\) sums to 1 after softmax.",
        isCorrect: true,
      },
      {
        text: "Entries are uniform weights unrelated to token importance.",
        isCorrect: false,
      },
      {
        text: "It does not encode pairwise relationships in the sequence.",
        isCorrect: false,
      },
      {
        text: "It directly contains the output representations.",
        isCorrect: false,
      },
    ],
    explanation:
      "After softmax, each row is a probability distribution over tokens. The matrix encodes how strongly each token attends to others. The final output is obtained after multiplying by the value matrix.",
  },

  {
    id: "mit6s191-l2-q20",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements compare RNNs and Transformers correctly?",
    options: [
      {
        text: "RNNs rely on hidden state recurrence across time.",
        isCorrect: true,
      },
      {
        text: "Transformers rely primarily on hidden-state recurrence.",
        isCorrect: false,
      },
      {
        text: "Transformers require sequential dependence in computation.",
        isCorrect: false,
      },
      {
        text: "RNNs and Transformers are mathematically identical models.",
        isCorrect: false,
      },
    ],
    explanation:
      "RNNs process sequences step-by-step using recurrence. Transformers rely on self-attention and parallel processing. While both model sequences, they differ fundamentally in architecture and computation.",
  },

  {
    id: "mit6s191-l2-q21",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "When unrolling an RNN for T time steps, the total loss is often \\( L = \\sum_{t=1}^T L_t \\). Which statements are correct?",
    options: [
      {
        text: "Each \\( L_t \\) can contribute gradients to parameters used at earlier time steps.",
        isCorrect: true,
      },
      {
        text: "Different independent weight matrices are used for every \\( L_t \\).",
        isCorrect: false,
      },
      {
        text: "Gradients do not accumulate across time from parameter sharing.",
        isCorrect: false,
      },
      {
        text: "Each time step has independent parameters that are updated separately.",
        isCorrect: false,
      },
    ],
    explanation:
      "Because parameters are shared across time, every time step contributes to the gradient of the same weights. Gradients from later steps flow backward through earlier hidden states. The parameters are not independent across time.",
  },

  {
    id: "mit6s191-l2-q22",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Consider the recurrence \\( h_t = \\tanh(W_h h_{t-1} + W_x x_t) \\). Which statements are correct?",
    options: [
      {
        text: "The Jacobian \\( \\frac{\\partial h_t}{\\partial h_{t-1}} \\) involves \\( W_h \\).",
        isCorrect: true,
      },
      {
        text: "Repeated multiplication of \\( W_h \\) has no effect on gradient stability.",
        isCorrect: false,
      },
      {
        text: "The activation derivative does not contribute multiplicatively to gradient flow.",
        isCorrect: false,
      },
      {
        text: "The recurrence removes all nonlinearities from the model.",
        isCorrect: false,
      },
    ],
    explanation:
      "Gradients through time involve repeated multiplication by \\( W_h \\) and the derivative of \\( \\tanh \\). This repeated multiplication determines whether gradients explode or vanish. The recurrence still includes nonlinearities.",
  },

  {
    id: "mit6s191-l2-q23",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which properties make sequence modeling challenging?",
    options: [
      {
        text: "Sequence models often must handle variable sequence lengths in practice.",
        isCorrect: true,
      },
      {
        text: "Long-range dependencies are never a challenge in sequence modeling.",
        isCorrect: false,
      },
      {
        text: "Sequence models are completely insensitive to word order.",
        isCorrect: false,
      },
      {
        text: "All sequences have identical structure and length.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sequence modeling must handle varying lengths, long-range interactions, and order sensitivity. These properties make it fundamentally more complex than static input modeling.",
  },

  {
    id: "mit6s191-l2-q24",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "In self-attention, the output is computed as \\( \\text{Attention}(Q,K,V) = A V \\), where \\( A = \\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}}) \\). Which statements are correct?",
    options: [
      {
        text: "The output is a weighted linear combination of value vectors.",
        isCorrect: true,
      },
      {
        text: "The weights are independent of query-key similarity.",
        isCorrect: false,
      },
      {
        text: "The dimensionality of the output is unrelated to the value vectors.",
        isCorrect: false,
      },
      { text: "The output ignores the value matrix.", isCorrect: false },
    ],
    explanation:
      "Attention computes weights via similarity between queries and keys, then forms weighted combinations of values. The resulting representation preserves the dimensionality of the value vectors.",
  },

  {
    id: "mit6s191-l2-q25",
    chapter: 2,
    difficulty: "medium",
    prompt: "Why is scaling by \\( \\sqrt{d_k} \\) used in attention?",
    options: [
      {
        text: "Dot products grow in magnitude with dimension.",
        isCorrect: true,
      },
      {
        text: "Scaling makes softmax saturate more aggressively.",
        isCorrect: false,
      },
      {
        text: "It destabilizes gradients during training.",
        isCorrect: false,
      },
      {
        text: "It enforces orthogonality between queries and keys.",
        isCorrect: false,
      },
    ],
    explanation:
      "As dimensionality increases, dot products tend to grow in magnitude. Dividing by \\( \\sqrt{d_k} \\) prevents extremely sharp softmax distributions and stabilizes training. It does not enforce orthogonality.",
  },

  {
    id: "mit6s191-l2-q26",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about recurrence are correct?",
    options: [
      {
        text: "Recurrence allows a model to maintain state over time.",
        isCorrect: true,
      },
      {
        text: "The hidden state cannot encode past inputs.",
        isCorrect: false,
      },
      {
        text: "Recurrence removes sequential computation dependency.",
        isCorrect: false,
      },
      {
        text: "Recurrence guarantees perfect long-term memory.",
        isCorrect: false,
      },
    ],
    explanation:
      "Recurrence enables memory through hidden states. However, it does not guarantee reliable long-term memory due to gradient instability and fixed state size.",
  },

  {
    id: "mit6s191-l2-q27",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Suppose an RNN has hidden dimension \\( d_h \\). Which statements are correct?",
    options: [
      {
        text: "The hidden state vector \\( h_t \\in \\mathbb{R}^{d_h} \\).",
        isCorrect: true,
      },
      {
        text: "The hidden state is not a fixed-capacity information bottleneck.",
        isCorrect: false,
      },
      {
        text: "Increasing \\( d_h \\) reduces representational capacity.",
        isCorrect: false,
      },
      {
        text: "The hidden state size automatically grows with sequence length.",
        isCorrect: false,
      },
    ],
    explanation:
      "The hidden state has fixed dimensionality. This creates an information bottleneck. Increasing hidden size increases capacity, but it does not grow dynamically with sequence length.",
  },

  {
    id: "mit6s191-l2-q28",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about attention compared to RNNs are correct?",
    options: [
      {
        text: "Attention allows direct interaction between any pair of tokens.",
        isCorrect: true,
      },
      { text: "Attention avoids recurrence.", isCorrect: true },
      {
        text: "Attention can model long-range dependencies more directly.",
        isCorrect: true,
      },
      {
        text: "Attention requires sequential processing of tokens.",
        isCorrect: false,
      },
    ],
    explanation:
      "Self-attention computes pairwise interactions between all tokens simultaneously. This removes recurrence and allows direct modeling of long-range relationships.",
  },

  {
    id: "mit6s191-l2-q29",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Consider gradient flow in BPTT. If \\( \\|W_h\\| < 1 \\), repeated multiplication across time tends to:",
    options: [
      { text: "Shrink gradients exponentially.", isCorrect: true },
      { text: "Make early time steps hard to update.", isCorrect: true },
      { text: "Reduce sensitivity to distant past inputs.", isCorrect: true },
      {
        text: "Guarantee stable long-term memory retention.",
        isCorrect: false,
      },
    ],
    explanation:
      "If the spectral norm of \\( W_h \\) is less than one, repeated multiplication shrinks gradients. This makes it difficult to propagate information from later time steps to earlier ones.",
  },

  {
    id: "mit6s191-l2-q30",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements describe next-word prediction?",
    options: [
      { text: "It is an autoregressive modeling task.", isCorrect: true },
      { text: "It can be trained using cross-entropy loss.", isCorrect: true },
      {
        text: "It forms the basis of many large language models.",
        isCorrect: true,
      },
      { text: "It requires explicit recurrence.", isCorrect: false },
    ],
    explanation:
      "Next-word prediction models \\( p(x_{t+1} | x_1, ..., x_t) \\). It is typically trained with cross-entropy and can be implemented with either RNNs or Transformers.",
  },

  {
    id: "mit6s191-l2-q31",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about the attention weight matrix are correct?",
    options: [
      { text: "It represents pairwise similarity scores.", isCorrect: true },
      { text: "Softmax ensures weights sum to 1 per query.", isCorrect: true },
      { text: "It enables dynamic weighting of tokens.", isCorrect: true },
      { text: "It is independent of the input sequence.", isCorrect: false },
    ],
    explanation:
      "Attention weights depend directly on the input sequence via Q and K. Softmax normalizes them into distributions, enabling dynamic, input-dependent weighting.",
  },

  {
    id: "mit6s191-l2-q32",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statements about computational complexity are correct?",
    options: [
      {
        text: "RNNs scale linearly with sequence length in computation.",
        isCorrect: true,
      },
      {
        text: "Self-attention scales quadratically with sequence length in naive form.",
        isCorrect: true,
      },
      {
        text: "Attention enables full pairwise token interaction.",
        isCorrect: true,
      },
      { text: "RNNs allow full parallel token interaction.", isCorrect: false },
    ],
    explanation:
      "RNN computation grows linearly but is sequential. Self-attention computes pairwise interactions, leading to quadratic complexity in sequence length.",
  },

  {
    id: "mit6s191-l2-q33",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which are benefits of multi-head attention?",
    options: [
      { text: "Capturing diverse relational patterns.", isCorrect: true },
      { text: "Learning different similarity subspaces.", isCorrect: true },
      { text: "Increasing expressive power.", isCorrect: true },
      {
        text: "Eliminating the need for linear projections.",
        isCorrect: false,
      },
    ],
    explanation:
      "Each head has its own projections, enabling diverse feature extraction. This increases expressive power, but projections remain essential.",
  },

  {
    id: "mit6s191-l2-q34",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "In self-attention, if two tokens have identical embeddings and positional encodings, then:",
    options: [
      {
        text: "Their query vectors will be identical (before training randomness).",
        isCorrect: true,
      },
      {
        text: "Their attention weights toward other tokens may be identical.",
        isCorrect: true,
      },
      {
        text: "They are indistinguishable to the model at that layer.",
        isCorrect: true,
      },
      {
        text: "They automatically produce different outputs due to softmax.",
        isCorrect: false,
      },
    ],
    explanation:
      "If embeddings and positional encodings are identical, the resulting Q, K, V projections will also be identical. The model cannot distinguish them at that layer without additional context.",
  },

  {
    id: "mit6s191-l2-q35",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about positional embeddings are correct?",
    options: [
      {
        text: "They inject order information into the model.",
        isCorrect: true,
      },
      { text: "They can be learned or fixed.", isCorrect: true },
      { text: "They are added to token embeddings.", isCorrect: true },
      { text: "They remove the need for attention.", isCorrect: false },
    ],
    explanation:
      "Positional embeddings encode token order. They are combined with token embeddings and can be learned or predefined. They do not replace attention.",
  },

  {
    id: "mit6s191-l2-q36",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about exploding gradients are correct?",
    options: [
      {
        text: "They arise from repeated multiplication of large values.",
        isCorrect: true,
      },
      { text: "They can destabilize training.", isCorrect: true },
      {
        text: "They may require gradient clipping to control.",
        isCorrect: true,
      },
      {
        text: "They improve long-term memory retention automatically.",
        isCorrect: false,
      },
    ],
    explanation:
      "Exploding gradients occur when repeated multiplication amplifies gradient norms. Gradient clipping is often used to stabilize training. They do not inherently improve memory.",
  },

  {
    id: "mit6s191-l2-q37",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "In attention, why can long-range dependencies be modeled effectively?",
    options: [
      {
        text: "Every token can attend directly to every other token.",
        isCorrect: true,
      },
      {
        text: "There is no need to propagate information step by step.",
        isCorrect: true,
      },
      {
        text: "Dependency length does not affect path length between tokens.",
        isCorrect: true,
      },
      {
        text: "Attention eliminates the need for training data.",
        isCorrect: false,
      },
    ],
    explanation:
      "Self-attention connects all tokens directly, regardless of distance. This removes long chains of multiplications seen in RNNs and shortens effective dependency paths.",
  },

  {
    id: "mit6s191-l2-q38",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements compare fixed hidden state vs attention representations?",
    options: [
      {
        text: "RNN hidden states compress history into one vector.",
        isCorrect: true,
      },
      {
        text: "Attention maintains representations for all tokens.",
        isCorrect: true,
      },
      {
        text: "Attention reduces information bottleneck effects.",
        isCorrect: true,
      },
      {
        text: "RNNs inherently preserve all past tokens without compression.",
        isCorrect: false,
      },
    ],
    explanation:
      "RNNs compress sequence history into a fixed-size hidden state. Attention keeps per-token representations, reducing bottleneck constraints.",
  },

  {
    id: "mit6s191-l2-q39",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Consider the softmax function \\( \\text{softmax}(z_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}} \\). Which statements are correct?",
    options: [
      {
        text: "It produces a probability distribution over inputs.",
        isCorrect: true,
      },
      { text: "Outputs are non-negative and sum to 1.", isCorrect: true },
      { text: "It amplifies relative differences in logits.", isCorrect: true },
      { text: "It is a linear transformation.", isCorrect: false },
    ],
    explanation:
      "Softmax converts logits into probabilities. It is nonlinear and emphasizes larger values exponentially, which is crucial in attention weighting.",
  },

  {
    id: "mit6s191-l2-q40",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about Transformers are correct?",
    options: [
      {
        text: "They are built from stacked attention blocks.",
        isCorrect: true,
      },
      {
        text: "They rely on self-attention as a core mechanism.",
        isCorrect: true,
      },
      { text: "They can be applied beyond language tasks.", isCorrect: true },
      {
        text: "They fundamentally require hidden-state recurrence.",
        isCorrect: false,
      },
    ],
    explanation:
      "Transformers stack attention-based layers to build deep sequence models. They are widely used in language, vision, and biology. They do not require recurrence.",
  },
];
