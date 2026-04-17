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
    difficulty: "medium",
    prompt:
      "Which statements correctly describe why a plain feed-forward model applied independently at each time step can fail on sequential data?",
    options: [
      {
        text: "It computes \\( \\hat{y}_t \\) from \\( x_t \\) alone unless some memory mechanism is added.",

        isCorrect: true,
      },
      {
        text: "It cannot use information from earlier time steps through an internal state by default.",
        isCorrect: true,
      },
      {
        text: "It is well suited when the current prediction depends only on the current input vector.",
        isCorrect: true,
      },
      {
        text: "It is not the case that It automatically infers temporal order even when inputs are processed independently.",
        isCorrect: true,
      },
    ],
    explanation:
      "A feed-forward model applied separately to each time step treats each input in isolation. That can work when the output depends only on the current input, but it does not provide built-in memory of previous elements in a sequence.",
  },

  {
    id: "mit6s191-l2-q06",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which are examples of sequential data discussed in introductory sequence modeling?",
    options: [
      { text: "Audio signals over time.", isCorrect: true },
      { text: "Financial time series.", isCorrect: true },
      { text: "Deoxyribonucleic acid or protein sequences.", isCorrect: true },
      {
        text: "Electrocardiogram-style physiological traces.",
        isCorrect: true,
      },
    ],
    explanation:
      "Sequential data consists of elements whose order matters. The lecture examples include audio, finance, biological sequences, and physiological signals, all of which unfold across time or position.",
  },

  {
    id: "mit6s191-l2-q07",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe common sequence-modeling input-output patterns?",
    options: [
      {
        text: "Sentiment classification is often many-to-one.",
        isCorrect: true,
      },
      {
        text: "Image captioning is often one-to-many or encoded-input to generated-sequence output.",
        isCorrect: true,
      },
      {
        text: "Machine translation is commonly many-to-many.",
        isCorrect: true,
      },
      {
        text: "Binary classification from a single fixed input can be one-to-one.",
        isCorrect: true,
      },
    ],
    explanation:
      "Sequence tasks differ in how many inputs and outputs are involved. The lecture explicitly contrasts one-to-one, many-to-one, one-to-many, and many-to-many settings using examples such as classification, captioning, and translation.",
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
    prompt:
      "Which statements about limitations of recurrent neural networks are correct?",
    options: [
      {
        text: "Processing a sequence step by step makes recurrent neural networks harder to parallelize efficiently.",

        isCorrect: true,
      },
      {
        text: "The hidden state can become an encoding bottleneck because it must summarize past information in a limited representation.",
        isCorrect: true,
      },
      {
        text: "Capturing very long-range dependencies can be difficult in practice.",
        isCorrect: true,
      },
      {
        text: "It is not the case that Recurrent neural networks eliminate all memory constraints because recurrence stores the full past without compression.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture emphasizes that recurrent models have to carry information forward through a state vector, which can become a bottleneck. Because they process inputs sequentially and must preserve information through many updates, they are also less parallelizable and can struggle with long-term dependencies.",
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
    difficulty: "medium",
    prompt:
      "Suppose a feed-forward network is reused at each time step as \\( \\hat{y}_t = f(x_t) \\). Which statements are correct?",
    options: [
      {
        text: "The same architecture can be applied repeatedly to sequence elements.",

        isCorrect: true,
      },
      {
        text: "The prediction at time \\( t \\) need not depend on \\( x_{t-1} \\).",
        isCorrect: true,
      },
      {
        text: "It is not the case that This setup alone provides a hidden state that summarizes the past.",
        isCorrect: true,
      },
      {
        text: "It is a natural baseline before introducing recurrence.",
        isCorrect: true,
      },
    ],
    explanation:
      "Reusing a feed-forward network over time is a simple baseline for sequence data. However, without recurrence or another memory mechanism, it does not summarize prior inputs in a hidden state.",
  },

  {
    id: "mit6s191-l2-q13",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "What role does the hidden state \\( h_t \\) play in a recurrent neural network?",
    options: [
      {
        text: "It serves as a representation of past context accumulated up to time \\( t \\).",

        isCorrect: true,
      },
      {
        text: "It allows the current computation to depend on more than just \\( x_t \\).",
        isCorrect: true,
      },
      {
        text: "It is not the case that It is the model's output label at time \\( t \\).",
        isCorrect: true,
      },
      {
        text: "It can be interpreted as memory carried from earlier time steps.",
        isCorrect: true,
      },
    ],
    explanation:
      "The hidden state is the core memory mechanism in a recurrent neural network. It is not itself the output label; rather, it carries information forward so later predictions can depend on earlier inputs.",
  },

  {
    id: "mit6s191-l2-q14",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Consider the recurrence \\( h_t = f_W(x_t, h_{t-1}) \\). Which statements are correct?",
    options: [
      {
        text: "The new state depends on both the current input and the previous state.",

        isCorrect: true,
      },
      {
        text: "The subscript \\( W \\) indicates the function has parameters that are learned.",
        isCorrect: true,
      },
      {
        text: "The recurrence can be applied repeatedly across a whole sequence.",
        isCorrect: true,
      },
      {
        text: "It is not the case that The equation implies that \\( h_t \\) ignores all earlier inputs once \\( x_t \\) is known.",
        isCorrect: true,
      },
    ],
    explanation:
      "The recurrence explicitly combines present input with past memory. Because \\( h_{t-1} \\) already contains information from earlier steps, \\( h_t \\) can indirectly depend on much more than just the current token or observation.",
  },

  {
    id: "mit6s191-l2-q15",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about parameter sharing in recurrent neural networks are correct?",
    options: [
      {
        text: "The same function and parameter set are reused at every time step.",

        isCorrect: true,
      },
      {
        text: "Different time steps do not require completely different learned cells.",
        isCorrect: true,
      },
      {
        text: "Parameter sharing helps the model process sequences of different lengths.",
        isCorrect: true,
      },
      {
        text: "It is not the case that A recurrent neural network must learn a separate weight matrix for each sequence position.",
        isCorrect: true,
      },
    ],
    explanation:
      "A defining feature of recurrent neural networks is that the same cell is applied again and again across time. This parameter sharing makes the model position-agnostic and usable on variable-length sequences.",
  },

  {
    id: "mit6s191-l2-q16",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly compare a recurrent cell with a standard feed-forward block used on one time step?",
    options: [
      {
        text: "A recurrent cell receives the current input and a previous hidden state.",

        isCorrect: true,
      },
      {
        text: "A plain feed-forward block for one step can be written without an explicit past-memory input.",
        isCorrect: true,
      },
      {
        text: "A recurrent cell can update memory while producing an output.",
        isCorrect: true,
      },
      {
        text: "It is not the case that A feed-forward block and a recurrent cell are identical because both ignore earlier steps.",
        isCorrect: true,
      },
    ],
    explanation:
      "The key difference is the explicit memory pathway in a recurrent cell. A feed-forward block processes the current vector only, while a recurrent cell combines current input with past state and updates that state.",
  },

  {
    id: "mit6s191-l2-q17",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Suppose \\( x_t \\in \\mathbb{R}^m \\) and \\( \\hat{y}_t \\in \\mathbb{R}^n \\). Which statements are correct?",
    options: [
      {
        text: "Each time step can take an \\( m \\)-dimensional input vector and produce an \\( n \\)-dimensional output vector.",

        isCorrect: true,
      },
      {
        text: "The input and output dimensions do not have to be equal.",
        isCorrect: true,
      },
      {
        text: "Writing \\( x_t \\in \\mathbb{R}^m \\) means each individual time-step input is a vector with \\( m \\) components.",
        isCorrect: true,
      },
      {
        text: "It is not the case that The notation implies the sequence length must equal \\( m+n \\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The slide notation distinguishes feature dimension from sequence length. \\( m \\) and \\( n \\) describe vector sizes at a step, not how many time steps the sequence contains.",
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
      "In a simple next-word prediction loop, a model repeatedly processes one token at a time and updates its hidden state. Which statements are correct?",
    options: [
      {
        text: "The hidden state after processing several words can influence the prediction of the next word.",

        isCorrect: true,
      },
      {
        text: "A loop over words is a natural procedural view of recurrent computation.",
        isCorrect: true,
      },
      {
        text: "The prediction can depend on previously seen words through the updated state.",
        isCorrect: true,
      },
      {
        text: "It is not the case that The hidden state must be reset to a random new value after every word in the same sentence.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture's code intuition emphasizes that recurrent models operate by iterating through a sequence while carrying forward state. Resetting the state after every word would destroy the very memory the model is meant to preserve within the sequence.",
  },

  {
    id: "mit6s191-l2-q20",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about the phrase 'neurons with recurrence' are correct?",
    options: [
      {
        text: "It is not the case that It means the computation includes a pathway that feeds information forward across time steps.",

        isCorrect: false,
      },
      {
        text: "It introduces dependence on previous hidden activity.",
        isCorrect: true,
      },
      {
        text: "It is a way to give a neural computation memory over a sequence.",
        isCorrect: true,
      },
      {
        text: "It means the model can only be used for language and not for other sequential domains.",
        isCorrect: false,
      },
    ],
    explanation:
      "Recurrence means the computation at one step can depend on what happened before through a carried state. That idea is general and applies to many domains beyond language, such as audio, biology, finance, and physiology.",
  },

  {
    id: "mit6s191-l2-q21",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Why can a single image of a moving ball be insufficient for predicting where it will go next?",
    options: [
      {
        text: "It is not the case that A single frame may not reveal the ball's velocity or motion direction.",

        isCorrect: false,
      },
      {
        text: "Without prior frames, multiple futures may be plausible from the same image.",
        isCorrect: true,
      },
      {
        text: "Temporal context helps disambiguate the next state.",
        isCorrect: true,
      },
      {
        text: "A single image already uniquely determines future motion in all realistic settings.",
        isCorrect: false,
      },
    ],
    explanation:
      "This is the core intuition motivating sequence modeling in the lecture. One snapshot often lacks enough information about momentum or direction, so earlier observations provide the missing context needed for prediction.",
  },

  {
    id: "mit6s191-l2-q22",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly interpret the idea of processing 'individual time steps' versus processing a full sequence?",
    options: [
      {
        text: "It is not the case that Processing individual time steps treats each \\( x_t \\) as a separate input to the same model.",

        isCorrect: false,
      },
      {
        text: "A recurrent model augments this setup by passing state between steps.",
        isCorrect: true,
      },
      {
        text: "Processing individual time steps by itself is enough to encode past context explicitly.",
        isCorrect: false,
      },
      {
        text: "The distinction helps explain why recurrence is introduced in sequence models.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture builds from a simple per-step feed-forward view to a recurrent view. That contrast makes clear that recurrence is introduced precisely to carry information across steps rather than treat them as independent examples.",
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
    prompt: "Which statements about backpropagation through time are correct?",
    options: [
      {
        text: "It is not the case that It extends backpropagation by propagating gradients backward across time steps in an unrolled recurrent computation.",

        isCorrect: false,
      },
      {
        text: "Losses from multiple time steps can contribute to parameter updates for the same recurrent weights.",
        isCorrect: true,
      },
      {
        text: "The shared recurrent parameters are updated based on gradient information accumulated across the sequence.",
        isCorrect: true,
      },
      {
        text: "It trains a recurrent neural network without ever needing to differentiate through past hidden states.",
        isCorrect: false,
      },
    ],
    explanation:
      "Backpropagation through time treats the recurrent computation as an unrolled sequence of repeated operations. Because the same parameters are reused at every time step, gradients from many points in the sequence contribute to the update of the same weights.",
  },

  {
    id: "mit6s191-l2-q25",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe what a recurrent neural network cell does at one time step?",
    options: [
      {
        text: "It is not the case that It takes in the current input vector.",

        isCorrect: false,
      },
      {
        text: "It is not the case that It uses the previous hidden state as an additional input.",
        isCorrect: false,
      },
      {
        text: "It updates the hidden state for use at the next time step.",
        isCorrect: true,
      },
      {
        text: "It requires access to every future token in the sequence before producing the current output.",
        isCorrect: false,
      },
    ],
    explanation:
      "A recurrent cell is a local computation repeated over time. Its inputs are the current observation and prior memory, and its output includes an updated memory state for the next step.",
  },

  {
    id: "mit6s191-l2-q26",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Why do language models need a numerical representation of words or tokens before applying a neural network?",
    options: [
      {
        text: "It is not the case that Neural networks operate on numerical inputs rather than raw symbolic words.",

        isCorrect: false,
      },
      {
        text: "It is not the case that Tokenization or embedding provides a way to map language into a form that the model can process.",
        isCorrect: false,
      },
      {
        text: "A good embedding scheme can capture something about semantic or co-occurrence relationships between pieces of language.",
        isCorrect: true,
      },
      {
        text: "If the task involves text, the model can directly consume words without any numerical encoding step.",
        isCorrect: false,
      },
    ],
    explanation:
      "A core design criterion from the lecture is that all neural networks require numerical inputs. For language, this means that words, subwords, or tokens must first be mapped into indices, vectors, or embeddings before the sequence model can process them.",
  },

  {
    id: "mit6s191-l2-q27",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements about exploding and vanishing gradients in recurrent neural networks are correct?",
    options: [
      {
        text: "It is not the case that Repeated multiplication during backpropagation through time can cause gradients to grow very large or shrink very small.",

        isCorrect: false,
      },
      {
        text: "It is not the case that Exploding gradients can sometimes be mitigated in practice by gradient clipping.",
        isCorrect: false,
      },
      {
        text: "Vanishing gradients make it harder for the model to learn long-range dependencies.",
        isCorrect: true,
      },
      {
        text: "These issues arise only in feed-forward networks and not in recurrent models.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture explains that backpropagating through many time steps leads to repeated multiplications involving the recurrent weights. This can make gradients blow up or disappear, and in particular the vanishing-gradient problem makes it difficult to preserve learning signals over long time spans.",
  },

  {
    id: "mit6s191-l2-q28",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements explain why recurrence is useful for sequential prediction?",
    options: [
      {
        text: "It creates a mechanism for storing information from earlier time steps.",
        isCorrect: true,
      },
      {
        text: "It allows the current output to depend on past context.",
        isCorrect: true,
      },
      {
        text: "It provides a way to process sequences one element at a time while keeping memory.",
        isCorrect: true,
      },
      {
        text: "It removes the need to consider temporal order altogether.",
        isCorrect: false,
      },
    ],
    explanation:
      "Recurrence is useful because it makes temporal context available during prediction. It does not eliminate order; rather, it is one mechanism for making order matter computationally.",
  },

  {
    id: "mit6s191-l2-q29",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Consider a sentence processed word by word by a recurrent neural network. Which statements are correct?",
    options: [
      {
        text: "The hidden state after seeing the word 'recurrent' can still influence prediction after the word 'neural' is processed.",
        isCorrect: true,
      },
      {
        text: "Word order matters because the hidden state evolves sequentially.",
        isCorrect: true,
      },
      {
        text: "Changing the order of words can change the sequence of hidden states and therefore the prediction.",
        isCorrect: true,
      },
      {
        text: "If the same words are present, their order cannot affect the final prediction in a recurrent neural network.",
        isCorrect: false,
      },
    ],
    explanation:
      "Recurrent models are order-sensitive because each state is built from the previous one. Two sequences with the same words in different orders can produce different hidden trajectories and therefore different outputs.",
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
    difficulty: "easy",
    prompt: "Which statements about recurrent neural networks are correct?",
    options: [
      {
        text: "The acronym RNN stands for Recurrent Neural Network.",
        isCorrect: true,
      },
      {
        text: "A recurrent neural network updates a state as a sequence is processed.",
        isCorrect: true,
      },
      {
        text: "The same recurrent cell is reused across time steps.",
        isCorrect: true,
      },
      {
        text: "A recurrent neural network is defined by the absence of any hidden state.",
        isCorrect: false,
      },
    ],
    explanation:
      "A recurrent neural network is built around repeated application of the same state-updating computation. Its defining feature is the presence of a hidden state, not the absence of one.",
  },

  {
    id: "mit6s191-l2-q32",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly compare fixed-input feed-forward modeling with recurrent sequence modeling?",
    options: [
      {
        text: "A standard feed-forward network maps one input vector to one output without an explicit temporal state.",
        isCorrect: true,
      },
      {
        text: "A recurrent neural network extends this idea by carrying state across steps.",
        isCorrect: true,
      },
      {
        text: "Both can produce outputs, but only the recurrent version explicitly incorporates prior hidden memory.",
        isCorrect: true,
      },
      {
        text: "A recurrent neural network differs only in having more layers, not in how information flows across time.",
        isCorrect: false,
      },
    ],
    explanation:
      "The important distinction is not merely depth but temporal information flow. Recurrent models explicitly pass memory from one step to the next, which changes the computation fundamentally compared with per-step feed-forward processing.",
  },

  {
    id: "mit6s191-l2-q33",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which tasks below fit naturally into the sequence-modeling framing shown in the lecture?",
    options: [
      { text: "Sentiment classification from a sentence.", isCorrect: true },
      { text: "Machine translation.", isCorrect: true },
      { text: "Image caption generation.", isCorrect: true },
      {
        text: "Predicting a single label from one static feature vector with no temporal or ordered structure.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture presents sequence modeling as a broad framework covering tasks like sentiment analysis, captioning, and translation. A purely static non-ordered classification problem is not the main use case being emphasized there.",
  },

  {
    id: "mit6s191-l2-q34",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Suppose two sequences share the same current input \\( x_t \\) but have different previous hidden states \\( h_{t-1} \\). Which statements are correct in a recurrent neural network?",
    options: [
      {
        text: "They can produce different outputs at time \\( t \\).",
        isCorrect: true,
      },
      {
        text: "They can update to different new hidden states.",
        isCorrect: true,
      },
      {
        text: "This illustrates how past context can change the interpretation of the same current input.",
        isCorrect: true,
      },
      {
        text: "The current input alone guarantees identical behavior regardless of hidden state.",
        isCorrect: false,
      },
    ],
    explanation:
      "Because the recurrence uses both \\( x_t \\) and \\( h_{t-1} \\), the same current observation can be interpreted differently depending on context. That is exactly why hidden state is useful in sequence modeling.",
  },

  {
    id: "mit6s191-l2-q35",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Why were Long Short-Term Memory networks introduced as an important extension of recurrent neural networks?",
    options: [
      {
        text: "They make the internal recurrent computation more sophisticated in order to better control information flow over time.",
        isCorrect: true,
      },
      {
        text: "They were designed in part to help address difficulties with long-term dependencies.",
        isCorrect: true,
      },
      {
        text: "They preserve the basic sequence-modeling idea of maintaining and updating internal state over time.",
        isCorrect: true,
      },
      {
        text: "They solve sequence modeling by removing the need for any notion of memory or state.",
        isCorrect: false,
      },
    ],
    explanation:
      "Long Short-Term Memory networks keep the state-based perspective of recurrent modeling, but make the state update more structured and controllable. Their main motivation is to improve the handling of long-range information that plain recurrent models can struggle to preserve.",
  },

  {
    id: "mit6s191-l2-q36",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements are correct about processing a sequence word by word with a loop such as 'for word in sentence'?",
    options: [
      {
        text: "It mirrors the sequential structure of recurrent inference.",
        isCorrect: true,
      },
      {
        text: "At each iteration, both a prediction and an updated hidden state can be produced.",
        isCorrect: true,
      },
      {
        text: "The state passed to the next iteration summarizes information from prior words.",
        isCorrect: true,
      },
      {
        text: "The loop implies that all words are processed independently of one another.",
        isCorrect: false,
      },
    ],
    explanation:
      "The loop-based pseudocode is meant to make recurrent computation intuitive. Each iteration depends on the current word and the accumulated state from previous iterations, so the words are not processed independently.",
  },

  {
    id: "mit6s191-l2-q37",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly explain why order matters in many sequential problems?",
    options: [
      {
        text: "Earlier elements can change how later elements should be interpreted.",
        isCorrect: true,
      },
      {
        text: "A sequence contains more information than an unordered bag of its elements when temporal structure matters.",
        isCorrect: true,
      },
      {
        text: "Predictive tasks such as motion forecasting often depend on the trajectory, not just the current frame.",
        isCorrect: true,
      },
      {
        text: "If two datasets contain the same values, their ordering can never influence prediction quality.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sequence models are needed because order often carries information. Motion, language, and many real-world signals cannot be understood fully from an unordered collection of observations.",
  },

  {
    id: "mit6s191-l2-q38",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements best capture the core intuition behind attention?",
    options: [
      {
        text: "Attention is a mechanism for identifying which parts of an input should be considered most important.",
        isCorrect: true,
      },
      {
        text: "It can be understood as a learned search process over the input.",
        isCorrect: true,
      },
      {
        text: "After identifying important parts of the input, the model can use them to extract the relevant information or features.",
        isCorrect: true,
      },
      {
        text: "Attention works only by scanning strictly one time step after another through a recurrent hidden state.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture presents attention as a way for the model to learn where to look inside the input. Conceptually, it resembles a search procedure: identify what matters most, then use that to extract the most relevant information for the next computation.",
  },

  {
    id: "mit6s191-l2-q39",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "In self-attention, which statements about query, key, and value are correct?",
    options: [
      {
        text: "They are learned transformations of a position-aware input representation.",
        isCorrect: true,
      },
      {
        text: "Similarity between query and key is used to compute attention weights.",
        isCorrect: true,
      },
      {
        text: "Those attention weights are then used to extract information from the value representation.",
        isCorrect: true,
      },
      {
        text: "Query, key, and value must all be identical matrices because they come from the same input.",
        isCorrect: false,
      },
    ],
    explanation:
      "Even though query, key, and value are derived from the same input sequence, they are produced by different learned transformations and therefore serve different roles. Query-key similarity determines what to attend to, and the resulting weights are used to combine or extract the value information.",
  },

  {
    id: "mit6s191-l2-q40",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe why Transformer-style attention became such an important alternative to recurrent neural networks for sequence modeling?",
    options: [
      {
        text: "It avoids explicit recurrence while still allowing the model to learn relationships across a sequence.",
        isCorrect: true,
      },
      {
        text: "It is highly parallelizable compared with step-by-step recurrent processing.",
        isCorrect: true,
      },
      {
        text: "Positional embeddings are used so the model can still retain information about order without recurrence.",
        isCorrect: true,
      },
      {
        text: "Transformers remove the need to model relationships within a sequence because order no longer matters.",
        isCorrect: false,
      },
    ],
    explanation:
      "The transcript frames attention and Transformers as a response to the bottlenecks of recurrent models, especially sequential processing and limited long-memory behavior. By combining attention with position-aware encodings, Transformers can model dependencies across a sequence without relying on a recurrent hidden state.",
  },
];
