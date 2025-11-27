// lib/chapter2.ts
import { Question } from "./quiz";

export const chapter2Questions: Question[] = [
  // --- Attention & alignment ---

  {
    id: "ch2-q01",
    chapter: 2,
    difficulty: "easy",
    prompt: "Why were attention mechanisms first introduced for sequence-to-sequence models?",
    options: [
      {
        text: "To learn which input tokens are most relevant when generating each output token.",
        isCorrect: true,
      },
      {
        text: "To solve alignment issues when input and output sequences have different lengths.",
        isCorrect: true,
      },
      {
        text: "To completely remove the need for any encoder network.",
        isCorrect: false,
      },
      {
        text: "To guarantee that every input token has exactly the same importance.",
        isCorrect: false,
      },
    ],
    explanation:
      "Attention lets the model focus on specific encoder states for each decoder step, handling variable-length alignment and relevance between source and target tokens.",
  },
  {
    id: "ch2-q02",
    chapter: 2,
    difficulty: "easy",
    prompt: "In an encoder–decoder model with attention, what is the role of the context vector at a decoder time step?",
    options: [
      {
        text: "It summarizes the encoder hidden states using attention weights for that time step.",
        isCorrect: true,
      },
      {
        text: "It is a fixed summary that never changes across decoder steps.",
        isCorrect: false,
      },
      {
        text: "It is used together with the decoder state to predict the next output token.",
        isCorrect: true,
      },
      {
        text: "It contains only the last encoder hidden state.",
        isCorrect: false,
      },
    ],
    explanation:
      "For each output token, attention computes weights over all encoder states and combines them into a context vector, which is then used for prediction.",
  },
  {
    id: "ch2-q03",
    chapter: 2,
    difficulty: "medium",
    prompt: "In classic additive or dot-product attention, what does the score function score(sᵢ₋₁, hⱼ) represent?",
    options: [
      {
        text: "A similarity measure between the current decoder state and an encoder hidden state.",
        isCorrect: true,
      },
      {
        text: "A distance between the current input token and the previous one.",
        isCorrect: false,
      },
      {
        text: "A scalar that will later be turned into an attention weight via softmax.",
        isCorrect: true,
      },
      {
        text: "A probability that already sums to one over all encoder positions.",
        isCorrect: false,
      },
    ],
    explanation:
      "The score is a scalar similarity between decoder state sᵢ₋₁ and encoder state hⱼ; applying softmax over all j produces attention weights.",
  },
  {
    id: "ch2-q04",
    chapter: 2,
    difficulty: "easy",
    prompt: "What does the softmax function do when applied to a vector of attention scores?",
    options: [
      {
        text: "Converts arbitrary real-valued scores into positive weights that sum to 1.",
        isCorrect: true,
      },
      {
        text: "Ensures that the largest score gets probability exactly 1.",
        isCorrect: false,
      },
      {
        text: "Emphasizes relatively larger scores while down-weighting smaller ones.",
        isCorrect: true,
      },
      {
        text: "Outputs values that can be negative or greater than 1.",
        isCorrect: false,
      },
    ],
    explanation:
      "Softmax exponentiates scores, normalizes them to sum to 1, and thus produces a probability distribution (attention weights).",
  },
  {
    id: "ch2-q05",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements describe advantages of adding attention to encoder–decoder models?",
    options: [
      {
        text: "It provides a shortcut path from early encoder states to the decoder, helping gradients flow.",
        isCorrect: true,
      },
      {
        text: "It removes the bottleneck of compressing the whole sequence into a single fixed vector.",
        isCorrect: true,
      },
      {
        text: "It makes models more interpretable by exposing which source tokens influence each prediction.",
        isCorrect: true,
      },
      {
        text: "It guarantees perfect translations without errors.",
        isCorrect: false,
      },
    ],
    explanation:
      "Attention reduces vanishing gradients, removes the single-vector bottleneck, and yields interpretable alignment patterns, but it does not guarantee perfection.",
  },

  // --- Self-attention, Q/K/V, scaling ---

  {
    id: "ch2-q06",
    chapter: 2,
    difficulty: "easy",
    prompt: "In self-attention, what is the basic idea when processing a sequence of tokens?",
    options: [
      {
        text: "Each token attends to other tokens in the same sequence to build a richer representation.",
        isCorrect: true,
      },
      {
        text: "Tokens are processed independently with no interaction.",
        isCorrect: false,
      },
      {
        text: "The mechanism can discover relationships between distant positions in a single step.",
        isCorrect: true,
      },
      {
        text: "The model must always read the sequence strictly left-to-right.",
        isCorrect: false,
      },
    ],
    explanation:
      "Self-attention lets every token compare itself with all others, capturing long-range dependencies without sequential recurrence.",
  },
  {
    id: "ch2-q07",
    chapter: 2,
    difficulty: "medium",
    prompt: "In the self-attention formula Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V, what do Q, K, and V represent?",
    options: [
      {
        text: "They are learned linear projections of the same input representations.",
        isCorrect: true,
      },
      {
        text: "Q contains query vectors that represent what each position is looking for.",
        isCorrect: true,
      },
      {
        text: "K contains key vectors used to compare against queries.",
        isCorrect: true,
      },
      {
        text: "V contains arbitrary random noise that is ignored in the output.",
        isCorrect: false,
      },
    ],
    explanation:
      "Q, K, and V are produced by multiplying the input by three learned weight matrices; queries are matched with keys, and the resulting weights are used to mix the value vectors.",
  },
  {
    id: "ch2-q08",
    chapter: 2,
    difficulty: "medium",
    prompt: "Why is the dot product QKᵀ divided by √dₖ before applying softmax in scaled dot-product attention?",
    options: [
      {
        text: "To prevent very large dot products when vectors have high dimensionality.",
        isCorrect: true,
      },
      {
        text: "Because otherwise softmax can become extremely peaky with tiny gradients.",
        isCorrect: true,
      },
      {
        text: "To ensure that all attention scores are exactly between −1 and 1.",
        isCorrect: false,
      },
      {
        text: "To guarantee that the gradient is always zero for large inputs.",
        isCorrect: false,
      },
    ],
    explanation:
      "Scaling by √dₖ keeps dot products within a reasonable range so softmax does not concentrate all mass on a few positions and thus keeps gradients usable.",
  },
  {
    id: "ch2-q09",
    chapter: 2,
    difficulty: "medium",
    prompt: "What does the matrix QKᵀ contain in self-attention?",
    options: [
      {
        text: "For each pair of positions, a similarity score between their query and key vectors.",
        isCorrect: true,
      },
      {
        text: "A single scalar summarizing the whole sequence.",
        isCorrect: false,
      },
      {
        text: "One row per query token and one column per key token.",
        isCorrect: true,
      },
      {
        text: "Direct probabilities that already sum to 1 across each row.",
        isCorrect: false,
      },
    ],
    explanation:
      "QKᵀ yields an attention score matrix: entry (i, j) is the similarity between token i’s query and token j’s key; row-wise softmax converts those scores into weights.",
  },
  {
    id: "ch2-q10",
    chapter: 2,
    difficulty: "medium",
    prompt: "What are key properties of multi-head self-attention?",
    options: [
      {
        text: "It uses several attention heads in parallel, each with its own Q, K, and V projections.",
        isCorrect: true,
      },
      {
        text: "Each head can specialize in different types of relationships (e.g., syntax, coreference).",
        isCorrect: true,
      },
      {
        text: "The outputs of all heads are concatenated and linearly projected back to the model dimension.",
        isCorrect: true,
      },
      {
        text: "It forces all heads to learn identical attention patterns.",
        isCorrect: false,
      },
    ],
    explanation:
      "Multi-head attention runs multiple Q/K/V projections, allowing different heads to focus on different aspects; their outputs are concatenated and linearly combined.",
  },
  {
    id: "ch2-q11",
    chapter: 2,
    difficulty: "hard",
    prompt: "Self-attention over a sequence of length T and hidden size d has what approximate time and memory complexity?",
    options: [
      {
        text: "Time complexity grows on the order of T² with respect to sequence length.",
        isCorrect: true,
      },
      {
        text: "Memory usage also scales roughly with T² because of the attention matrix.",
        isCorrect: true,
      },
      {
        text: "The cost is linear in T and independent of d.",
        isCorrect: false,
      },
      {
        text: "The quadratic cost is a major bottleneck for very long sequences.",
        isCorrect: true,
      },
    ],
    explanation:
      "Computing all pairwise interactions between T tokens leads to O(T²) time and space, which motivates efficient attention variants for long contexts.",
  },

  // --- Positional encoding & transformer block ---

  {
    id: "ch2-q12",
    chapter: 2,
    difficulty: "easy",
    prompt: "Why does a transformer need some form of positional information?",
    options: [
      {
        text: "Because plain self-attention treats the input as a set and does not know token order.",
        isCorrect: true,
      },
      {
        text: "Because the embeddings themselves always encode exact positions.",
        isCorrect: false,
      },
      {
        text: "Because word order often changes the meaning of a sentence.",
        isCorrect: true,
      },
      {
        text: "Because GPUs cannot process sequences without it.",
        isCorrect: false,
      },
    ],
    explanation:
      "Self-attention is permutation-invariant; positional encodings inject order information so the model can distinguish ‘dog bites man’ from ‘man bites dog’.",
  },
  {
    id: "ch2-q13",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements describe sinusoidal positional encodings used in the original transformer?",
    options: [
      {
        text: "They use sine and cosine functions of different frequencies over the position index.",
        isCorrect: true,
      },
      {
        text: "They are deterministic and do not introduce additional learned parameters.",
        isCorrect: true,
      },
      {
        text: "They allow the model to infer relative positions from linear combinations of encodings.",
        isCorrect: true,
      },
      {
        text: "They require retraining if we want to handle longer sequences than in training.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sinusoidal encodings are fixed functions of position; their structure enables the model to extrapolate and reason about relative distances without extra parameters.",
  },
  {
    id: "ch2-q14",
    chapter: 2,
    difficulty: "medium",
    prompt: "What are the main components inside a standard transformer block?",
    options: [
      {
        text: "A multi-head self-attention sublayer.",
        isCorrect: true,
      },
      {
        text: "A position-wise feedforward network applied to each token.",
        isCorrect: true,
      },
      {
        text: "Residual (skip) connections around each sublayer.",
        isCorrect: true,
      },
      {
        text: "A convolutional layer that processes the sequence in 2D.",
        isCorrect: false,
      },
    ],
    explanation:
      "Each transformer block stacks multi-head self-attention and a feedforward network, each wrapped with residual connections and layer normalization.",
  },
  {
    id: "ch2-q15",
    chapter: 2,
    difficulty: "medium",
    prompt: "The position-wise feedforward network in a transformer block has which characteristics?",
    options: [
      {
        text: "It consists of two linear layers with a nonlinearity such as ReLU in between.",
        isCorrect: true,
      },
      {
        text: "It is applied independently to each token’s representation.",
        isCorrect: true,
      },
      {
        text: "It usually expands the dimension and then projects back down to the model size.",
        isCorrect: true,
      },
      {
        text: "It shares the same weights across all layers of the transformer.",
        isCorrect: false,
      },
    ],
    explanation:
      "The FFN is a small MLP applied pointwise to each token; in each layer it has its own parameters and often uses a higher hidden dimension than the model size.",
  },
  {
    id: "ch2-q16",
    chapter: 2,
    difficulty: "easy",
    prompt: "What is the main purpose of residual (skip) connections in deep neural networks such as transformers?",
    options: [
      {
        text: "They add the input of a sublayer to its output before normalization.",
        isCorrect: true,
      },
      {
        text: "They help gradients flow through many layers during backpropagation.",
        isCorrect: true,
      },
      {
        text: "They make the loss landscape smoother and easier to optimize.",
        isCorrect: true,
      },
      {
        text: "They prevent the model from ever changing the input representation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Residual connections provide shortcut paths that ease optimization and stabilize very deep models while still allowing layers to modify representations.",
  },
  {
    id: "ch2-q17",
    chapter: 2,
    difficulty: "medium",
    prompt: "Layer normalization in transformers does which of the following?",
    options: [
      {
        text: "Normalizes activations across features within a single token representation.",
        isCorrect: true,
      },
      {
        text: "Uses the mean and standard deviation of that token’s features.",
        isCorrect: true,
      },
      {
        text: "Applies learned scale (γ) and shift (β) parameters after normalization.",
        isCorrect: true,
      },
      {
        text: "Requires large batch sizes to estimate statistics.",
        isCorrect: false,
      },
    ],
    explanation:
      "LayerNorm operates per token, using that token’s mean and variance, then applies learnable γ and β. Unlike BatchNorm it does not depend on batch size.",
  },

  // --- Encoder, decoder, cross-attention, masking ---

  {
    id: "ch2-q18",
    chapter: 2,
    difficulty: "medium",
    prompt: "In an encoder–decoder transformer for translation, what is cross-attention?",
    options: [
      {
        text: "An attention mechanism in the decoder that uses encoder outputs as keys and values.",
        isCorrect: true,
      },
      {
        text: "Self-attention over the encoder tokens only.",
        isCorrect: false,
      },
      {
        text: "The way the decoder conditions its generation on the encoded source sentence.",
        isCorrect: true,
      },
      {
        text: "A mechanism that ignores the encoder and only attends within the decoder.",
        isCorrect: false,
      },
    ],
    explanation:
      "Cross-attention takes queries from the decoder states and keys/values from encoder outputs, allowing the decoder to focus on relevant source positions.",
  },
  {
    id: "ch2-q19",
    chapter: 2,
    difficulty: "medium",
    prompt: "Why is masking used in the decoder’s self-attention for language modeling?",
    options: [
      {
        text: "To prevent each position from attending to future tokens that should not be known yet.",
        isCorrect: true,
      },
      {
        text: "Because attending to past tokens is mathematically impossible.",
        isCorrect: false,
      },
      {
        text: "To avoid information leakage when training a next-token prediction model.",
        isCorrect: true,
      },
      {
        text: "To force the model to see the entire sequence before predicting anything.",
        isCorrect: false,
      },
    ],
    explanation:
      "Causal masks set attention scores to −∞ for future positions so that after softmax they receive zero weight, enforcing autoregressive behavior.",
  },
  {
    id: "ch2-q20",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements correctly contrast encoder-only, decoder-only, and encoder–decoder transformers?",
    options: [
      {
        text: "Encoder-only models (like BERT) are well-suited for understanding tasks such as classification.",
        isCorrect: true,
      },
      {
        text: "Decoder-only models (like GPT-2) are typically used for autoregressive text generation.",
        isCorrect: true,
      },
      {
        text: "Encoder–decoder models are natural fits for sequence-to-sequence tasks like translation.",
        isCorrect: true,
      },
      {
        text: "Decoder-only models cannot be used for generation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Encoders build rich representations of inputs; decoders generate sequences; encoder–decoder structures connect the two for tasks like translation.",
  },

  // --- Language modeling, training objective, decoding ---

  {
    id: "ch2-q21",
    chapter: 2,
    difficulty: "medium",
    prompt: "In autoregressive language modeling, how is the probability of a sequence w₁…wₙ typically factorized?",
    options: [
      {
        text: "As the product ∏ᵢ P(wᵢ | w₁:ᵢ₋₁).",
        isCorrect: true,
      },
      {
        text: "As the sum ∑ᵢ P(wᵢ | w₁:ᵢ₋₁).",
        isCorrect: false,
      },
      {
        text: "Using the chain rule of probability.",
        isCorrect: true,
      },
      {
        text: "Assuming all tokens are independent of each other.",
        isCorrect: false,
      },
    ],
    explanation:
      "The chain rule expresses the joint probability as a product of conditional probabilities for each token given its predecessors.",
  },
  {
    id: "ch2-q22",
    chapter: 2,
    difficulty: "medium",
    prompt: "During training of a language model, what are the predicted and true distributions used in the cross-entropy loss?",
    options: [
      {
        text: "The predicted distribution is the softmax over logits for the next token.",
        isCorrect: true,
      },
      {
        text: "The true distribution is typically a one-hot vector for the actual next token.",
        isCorrect: true,
      },
      {
        text: "Both distributions have dimensionality equal to the vocabulary size.",
        isCorrect: true,
      },
      {
        text: "The true distribution is always uniform over the vocabulary.",
        isCorrect: false,
      },
    ],
    explanation:
      "At each position, the model predicts a probability over the vocabulary; the target is 1 for the correct token and 0 for all others, and cross-entropy measures their mismatch.",
  },
  {
    id: "ch2-q23",
    chapter: 2,
    difficulty: "medium",
    prompt: "What is teacher forcing in the context of training sequence models?",
    options: [
      {
        text: "Feeding the ground-truth previous token into the model when predicting the next one.",
        isCorrect: true,
      },
      {
        text: "Always feeding the model its own previous predictions.",
        isCorrect: false,
      },
      {
        text: "Computing the loss at each time step using the correct next token.",
        isCorrect: true,
      },
      {
        text: "A method that allows the model to ignore the training data.",
        isCorrect: false,
      },
    ],
    explanation:
      "Teacher forcing uses the real previous tokens during training, stabilizing learning by avoiding compounding errors from model-generated history.",
  },
  {
    id: "ch2-q24",
    chapter: 2,
    difficulty: "easy",
    prompt: "Why is greedy decoding (always picking the highest-probability token) often unsatisfactory for text generation?",
    options: [
      {
        text: "It tends to produce repetitive and overly generic text.",
        isCorrect: true,
      },
      {
        text: "It ignores alternative plausible continuations that might yield more diverse outputs.",
        isCorrect: true,
      },
      {
        text: "It is computationally more expensive than sampling-based methods.",
        isCorrect: false,
      },
      {
        text: "It can get stuck in dull loops like repeating short phrases.",
        isCorrect: true,
      },
    ],
    explanation:
      "Greedy decoding is deterministic and conservative, which often leads to safe but boring and repetitive outputs.",
  },
  {
    id: "ch2-q25",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements correctly describe common stochastic decoding strategies for language models?",
    options: [
      {
        text: "Top-k sampling restricts sampling to the k most probable tokens at each step.",
        isCorrect: true,
      },
      {
        text: "Top-p (nucleus) sampling keeps the smallest set of tokens whose cumulative probability exceeds a threshold p.",
        isCorrect: true,
      },
      {
        text: "Temperature scaling divides logits by a temperature before softmax to control sharpness.",
        isCorrect: true,
      },
      {
        text: "Lower temperature values make the distribution more uniform and random.",
        isCorrect: false,
      },
    ],
    explanation:
      "Top-k and top-p restrict the candidate set; temperature < 1 sharpens the distribution (more deterministic), while >1 flattens it (more random).",
  },

  // --- Tokenization & BPE ---

  {
    id: "ch2-q26",
    chapter: 2,
    difficulty: "easy",
    prompt: "Why do modern transformers often tokenize text into subword units instead of whole words?",
    options: [
      {
        text: "To reduce the number of unknown tokens for rare or new words.",
        isCorrect: true,
      },
      {
        text: "To exploit repeated morphemes such as prefixes and suffixes.",
        isCorrect: true,
      },
      {
        text: "Because subword vocabularies can balance vocabulary size and expressivity.",
        isCorrect: true,
      },
      {
        text: "Because using whole words makes it impossible to train a language model.",
        isCorrect: false,
      },
    ],
    explanation:
      "Subword tokenization like BPE splits rare words into frequent pieces while keeping common words whole, enabling open-vocabulary modeling.",
  },
  {
    id: "ch2-q27",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements describe Byte-Pair Encoding (BPE) for building a subword vocabulary?",
    options: [
      {
        text: "It starts from individual characters as initial tokens.",
        isCorrect: true,
      },
      {
        text: "It iteratively merges the most frequent adjacent symbol pairs into new tokens.",
        isCorrect: true,
      },
      {
        text: "It stops once a predetermined vocabulary size is reached.",
        isCorrect: true,
      },
      {
        text: "It guarantees that every word is represented as exactly one token.",
        isCorrect: false,
      },
    ],
    explanation:
      "BPE repeatedly merges frequent character or symbol pairs to form larger units until the vocabulary reaches a target size; rare words may still be split into multiple tokens.",
  },
  {
    id: "ch2-q28",
    chapter: 2,
    difficulty: "medium",
    prompt: "What happens to very rare or unseen words when using a BPE-style tokenizer?",
    options: [
      {
        text: "They are decomposed into smaller subword units that are in the vocabulary.",
        isCorrect: true,
      },
      {
        text: "They always become a single special <UNK> token.",
        isCorrect: false,
      },
      {
        text: "Their pieces may correspond to meaningful morphemes like roots and suffixes.",
        isCorrect: true,
      },
      {
        text: "They cannot be represented at all.",
        isCorrect: false,
      },
    ],
    explanation:
      "BPE avoids excessive unknown tokens by breaking rare words into known subword components, often aligning with morphological structure.",
  },

  // --- BERT, MLM, bidirectional encoders ---

  {
    id: "ch2-q29",
    chapter: 2,
    difficulty: "easy",
    prompt: "What is the key difference between a causal language model and a bidirectional encoder such as BERT?",
    options: [
      {
        text: "A causal model predicts each token using only past context.",
        isCorrect: true,
      },
      {
        text: "A bidirectional encoder can use both left and right context to represent each token.",
        isCorrect: true,
      },
      {
        text: "A causal model can freely look at future tokens during training.",
        isCorrect: false,
      },
      {
        text: "A bidirectional encoder cannot be used for tasks like classification.",
        isCorrect: false,
      },
    ],
    explanation:
      "Causal LMs are strictly left-to-right; bidirectional encoders see the entire sequence, making them powerful for understanding tasks but less straightforward for generation.",
  },
  {
    id: "ch2-q30",
    chapter: 2,
    difficulty: "medium",
    prompt: "In masked language modeling (MLM) used to train BERT-style models, what is the training objective?",
    options: [
      {
        text: "Randomly mask some input tokens and have the model predict the original tokens.",
        isCorrect: true,
      },
      {
        text: "Use the rest of the sequence (left and right context) to infer each masked token.",
        isCorrect: true,
      },
      {
        text: "Mask every token so that no context remains.",
        isCorrect: false,
      },
      {
        text: "Use a cross-entropy loss over the vocabulary for each masked position.",
        isCorrect: true,
      },
    ],
    explanation:
      "MLM corrupts a subset of tokens with a special mask symbol; the model sees full context and predicts the missing pieces using cross-entropy.",
  },
  {
    id: "ch2-q31",
    chapter: 2,
    difficulty: "medium",
    prompt: "What are some limitations of the masked language modeling objective?",
    options: [
      {
        text: "Only a small fraction of tokens (e.g., 15%) contribute directly to the loss.",
        isCorrect: true,
      },
      {
        text: "The presence of special mask tokens creates a training–inference mismatch.",
        isCorrect: true,
      },
      {
        text: "It cannot be used with bidirectional architectures.",
        isCorrect: false,
      },
      {
        text: "It can still be extended with auxiliary tasks such as next-sentence prediction.",
        isCorrect: true,
      },
    ],
    explanation:
      "MLM is sample-inefficient and introduces artificial mask tokens, but it enables powerful bidirectional representations and can be combined with extra objectives.",
  },
  {
    id: "ch2-q32",
    chapter: 2,
    difficulty: "medium",
    prompt: "How is the [CLS] token typically used in BERT-like models for classification tasks?",
    options: [
      {
        text: "It is added at the beginning of the input sequence.",
        isCorrect: true,
      },
      {
        text: "Its final hidden representation is treated as an aggregate representation of the whole sequence.",
        isCorrect: true,
      },
      {
        text: "A classifier layer is applied on top of this representation to predict labels.",
        isCorrect: true,
      },
      {
        text: "It is ignored during training because it carries no information.",
        isCorrect: false,
      },
    ],
    explanation:
      "The [CLS] token’s representation attends to all tokens; a small classifier on top of it can be fine-tuned for tasks like sentiment or entailment.",
  },

  // --- GPT-2, visualization, internals ---

  {
    id: "ch2-q33",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about GPT-2-style models are TRUE?",
    options: [
      {
        text: "They are decoder-only transformers trained with left-to-right language modeling.",
        isCorrect: true,
      },
      {
        text: "They generate tokens autoregressively, feeding previous outputs back as input.",
        isCorrect: true,
      },
      {
        text: "They use masked self-attention to prevent peeking at future positions.",
        isCorrect: true,
      },
      {
        text: "They are primarily designed for bidirectional understanding rather than generation.",
        isCorrect: false,
      },
    ],
    explanation:
      "GPT-2 consists only of stacked decoder blocks with causal masks, optimized for next-token prediction and generation rather than bidirectional encoding.",
  },
  {
    id: "ch2-q34",
    chapter: 2,
    difficulty: "medium",
    prompt: "What can attention visualizations reveal about a transformer model?",
    options: [
      {
        text: "Which input tokens a given head focuses on when processing a particular token.",
        isCorrect: true,
      },
      {
        text: "That different heads within a layer specialize in different relations (e.g., subject–verb, coreference).",
        isCorrect: true,
      },
      {
        text: "The exact internal numerical values of all parameters.",
        isCorrect: false,
      },
      {
        text: "Patterns such as alignment between source and target tokens in translation.",
        isCorrect: true,
      },
    ],
    explanation:
      "Attention maps provide a human-interpretable glimpse of which tokens influence others and how different heads distribute their focus.",
  },
  {
    id: "ch2-q35",
    chapter: 2,
    difficulty: "hard",
    prompt: "What does the Gradient × Input method aim to measure in a transformer language model?",
    options: [
      {
        text: "The sensitivity of the output (e.g., next-token probability) to small changes in each input token.",
        isCorrect: true,
      },
      {
        text: "An importance score indicating which tokens most affect a chosen prediction.",
        isCorrect: true,
      },
      {
        text: "How much each neuron’s weight changes during training.",
        isCorrect: false,
      },
      {
        text: "It multiplies the gradient with the input representation to localize influential tokens.",
        isCorrect: true,
      },
    ],
    explanation:
      "Gradient × Input multiplies local gradients by input activations to highlight which tokens are most influential for a particular output decision.",
  },
  {
    id: "ch2-q36",
    chapter: 2,
    difficulty: "hard",
    prompt: "Why might Non-Negative Matrix Factorization (NMF) be used on neuron activations in transformer feedforward layers?",
    options: [
      {
        text: "To decompose activations into non-negative factors that are easier to interpret.",
        isCorrect: true,
      },
      {
        text: "To discover groups of neurons that respond to particular patterns or positions in the text.",
        isCorrect: true,
      },
      {
        text: "To reduce dimensionality while preserving additive structure in activations.",
        isCorrect: true,
      },
      {
        text: "Because NMF guarantees that each factor corresponds to a single neuron.",
        isCorrect: false,
      },
    ],
    explanation:
      "NMF factorizes non-negative activation matrices into additive components, revealing interpretable patterns such as neurons that specialize in certain positions or grammatical categories.",
  },

  // --- Transfer learning, fine-tuning ---

  {
    id: "ch2-q37",
    chapter: 2,
    difficulty: "easy",
    prompt: "What is the core idea of transfer learning with large pretrained transformers?",
    options: [
      {
        text: "First train a model on a broad, generic task using massive text corpora.",
        isCorrect: true,
      },
      {
        text: "Then adapt the model to a specific downstream task using smaller labeled datasets.",
        isCorrect: true,
      },
      {
        text: "Always retrain all parameters from scratch for every new task.",
        isCorrect: false,
      },
      {
        text: "Reuse the learned representations to avoid learning language from zero each time.",
        isCorrect: true,
      },
    ],
    explanation:
      "Transfer learning leverages general language knowledge from pretraining and then fine-tunes or adapts the model on task-specific supervised data.",
  },
  {
    id: "ch2-q38",
    chapter: 2,
    difficulty: "medium",
    prompt: "During standard supervised fine-tuning of a transformer for classification, which practices are common?",
    options: [
      {
        text: "Add one or more randomly initialized linear layers on top of the pretrained backbone.",
        isCorrect: true,
      },
      {
        text: "Use labeled examples to train the new layers with a cross-entropy loss.",
        isCorrect: true,
      },
      {
        text: "Optionally freeze most of the backbone and train only the new layers.",
        isCorrect: true,
      },
      {
        text: "Discard the pretrained weights entirely and start from random initialization.",
        isCorrect: false,
      },
    ],
    explanation:
      "Fine-tuning typically attaches a small task-specific head and trains it (and sometimes parts of the backbone) on labeled data while reusing most pretrained parameters.",
  },
  {
    id: "ch2-q39",
    chapter: 2,
    difficulty: "medium",
    prompt: "Why is fine-tuning often feasible on a single GPU or even CPU, whereas pretraining is not?",
    options: [
      {
        text: "Fine-tuning uses a much smaller dataset focused on a specific task.",
        isCorrect: true,
      },
      {
        text: "Most of the heavy computation of learning general language patterns has already been done.",
        isCorrect: true,
      },
      {
        text: "Fine-tuning usually involves fewer training steps and lower overall compute.",
        isCorrect: true,
      },
      {
        text: "Pretraining is always faster than fine-tuning.",
        isCorrect: false,
      },
    ],
    explanation:
      "Pretraining requires huge datasets and long training; fine-tuning reuses those representations and adapts them with comparatively modest compute.",
  },

  // --- Knowledge distillation ---

  {
    id: "ch2-q40",
    chapter: 2,
    difficulty: "medium",
    prompt: "What is the goal of knowledge distillation in neural networks?",
    options: [
      {
        text: "To transfer knowledge from a large teacher model to a smaller student model.",
        isCorrect: true,
      },
      {
        text: "To obtain a compact model with similar performance but lower computational cost.",
        isCorrect: true,
      },
      {
        text: "To train a larger model from scratch using only the student’s predictions.",
        isCorrect: false,
      },
      {
        text: "To use the teacher’s outputs (e.g., logits) as training signals for the student.",
        isCorrect: true,
      },
    ],
    explanation:
      "Distillation compresses a big model into a smaller one by having the student mimic the teacher’s behavior, often using its soft output distributions.",
  },
  {
    id: "ch2-q41",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statements about the distillation loss between teacher and student outputs are TRUE?",
    options: [
      {
        text: "It often uses Kullback–Leibler divergence between the softened probability distributions.",
        isCorrect: true,
      },
      {
        text: "A temperature parameter can smooth the distributions to highlight relative similarities between classes.",
        isCorrect: true,
      },
      {
        text: "The loss is zero when the student’s distribution matches the teacher’s distribution exactly.",
        isCorrect: true,
      },
      {
        text: "It ignores the teacher entirely and uses only ground-truth hard labels.",
        isCorrect: false,
      },
    ],
    explanation:
      "Distillation typically computes KL divergence between temperature-scaled teacher and student probabilities; perfect matching yields zero KL.",
  },
  {
    id: "ch2-q42",
    chapter: 2,
    difficulty: "medium",
    prompt: "Compared to training a small model directly on labeled data, why can distillation from a larger model be beneficial?",
    options: [
      {
        text: "The teacher’s soft targets encode information about class similarities.",
        isCorrect: true,
      },
      {
        text: "The student can learn from examples where the teacher is uncertain, not only from hard labels.",
        isCorrect: true,
      },
      {
        text: "It allows transferring knowledge from very large pretraining corpora without retraining a giant model.",
        isCorrect: true,
      },
      {
        text: "It guarantees the student will always outperform the teacher.",
        isCorrect: false,
      },
    ],
    explanation:
      "Soft probabilities convey richer structure than one-hot labels, and the student inherits some of the teacher’s pretrained knowledge, though it may still underperform the teacher.",
  },

  // --- More detailed math / mechanics ---

  {
    id: "ch2-q43",
    chapter: 2,
    difficulty: "hard",
    prompt: "Consider a self-attention layer with input matrix X (T × d). What are the dimensions of Q, K, and V after multiplying by learned weight matrices W_Q, W_K, W_V of shape d × dₖ?",
    options: [
      {
        text: "Q, K, and V each have shape T × dₖ.",
        isCorrect: true,
      },
      {
        text: "QKᵀ then has shape T × T.",
        isCorrect: true,
      },
      {
        text: "The output softmax(QKᵀ/√dₖ)V has shape T × dₖ (before any output projection).",
        isCorrect: true,
      },
      {
        text: "Q has shape d × T because weights and inputs are always transposed.",
        isCorrect: false,
      },
    ],
    explanation:
      "Multiplying X (T×d) by W_Q (d×dₖ) yields Q (T×dₖ); similarly for K and V. QKᵀ is T×T, and multiplying by V (T×dₖ) returns T×dₖ.",
  },
  {
    id: "ch2-q44",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statements about the softmax function are mathematically correct?",
    options: [
      {
        text: "Adding the same constant to all inputs xᵢ does not change the softmax output.",
        isCorrect: true,
      },
      {
        text: "Softmax(x) is invariant to such shifts because they cancel out in numerator and denominator.",
        isCorrect: true,
      },
      {
        text: "Scaling all inputs by a factor greater than 1 makes the distribution more peaked.",
        isCorrect: true,
      },
      {
        text: "Softmax outputs can be exactly zero for finite inputs.",
        isCorrect: false,
      },
    ],
    explanation:
      "Softmax is shift-invariant; scaling magnifies differences and sharpens the distribution. For finite inputs, outputs are strictly positive but can be extremely small.",
  },
  {
    id: "ch2-q45",
    chapter: 2,
    difficulty: "medium",
    prompt: "Why do feedforward sublayers contribute a large fraction of parameters in a transformer?",
    options: [
      {
        text: "They usually have a hidden dimension larger than the model dimension (e.g., 4× larger).",
        isCorrect: true,
      },
      {
        text: "Each layer has its own FFN parameters, and there can be many layers.",
        isCorrect: true,
      },
      {
        text: "Attention mechanisms contain no learnable parameters at all.",
        isCorrect: false,
      },
      {
        text: "FFNs operate independently on each position, making them easy to parallelize.",
        isCorrect: true,
      },
    ],
    explanation:
      "Position-wise FFNs have large weight matrices and are repeated in every block, so they account for a majority of parameters; they are also highly parallelizable.",
  },

  // --- Applications & practical points ---

  {
    id: "ch2-q46",
    chapter: 2,
    difficulty: "medium",
    prompt: "Why is it useful to freeze most layers of a pretrained transformer during fine-tuning sometimes?",
    options: [
      {
        text: "To reduce the risk of overfitting when the downstream dataset is small.",
        isCorrect: true,
      },
      {
        text: "To decrease computational cost and memory usage during training.",
        isCorrect: true,
      },
      {
        text: "Because the lower layers already capture general language features that we want to preserve.",
        isCorrect: true,
      },
      {
        text: "Because frozen layers cannot pass gradients to earlier layers.",
        isCorrect: false,
      },
    ],
    explanation:
      "Freezing most layers retains general representations, reduces compute, and can prevent overfitting when only limited labeled data is available.",
  },
  {
    id: "ch2-q47",
    chapter: 2,
    difficulty: "medium",
    prompt: "What is a confusion matrix typically used for when evaluating classification models fine-tuned from transformers?",
    options: [
      {
        text: "To show counts of true vs predicted labels for each class.",
        isCorrect: true,
      },
      {
        text: "To identify which classes are most frequently confused with each other.",
        isCorrect: true,
      },
      {
        text: "To visualize model performance beyond a single accuracy number.",
        isCorrect: true,
      },
      {
        text: "To directly display gradients used during training.",
        isCorrect: false,
      },
    ],
    explanation:
      "Confusion matrices reveal detailed error patterns, such as which sentiment classes or categories the model mixes up.",
  },
  {
    id: "ch2-q48",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about using Hugging Face’s Trainer for fine-tuning transformers are TRUE?",
    options: [
      {
        text: "You specify training hyperparameters such as batch size, epochs, and learning rate via TrainingArguments.",
        isCorrect: true,
      },
      {
        text: "The Trainer handles the training loop, evaluation, and logging for you.",
        isCorrect: true,
      },
      {
        text: "You still need to provide tokenized datasets compatible with the chosen model.",
        isCorrect: true,
      },
      {
        text: "Trainer can only be used for vision models, not text transformers.",
        isCorrect: false,
      },
    ],
    explanation:
      "Trainer abstracts away much of the boilerplate training code but still requires suitable datasets and model/tokenizer choices.",
  },
  {
    id: "ch2-q49",
    chapter: 2,
    difficulty: "medium",
    prompt: "Why is it important to use the tokenizer that matches a given pretrained transformer model?",
    options: [
      {
        text: "Because the model’s embedding matrix is aligned with the token IDs produced by its tokenizer.",
        isCorrect: true,
      },
      {
        text: "Using a different tokenizer would mismatch token IDs and embeddings, leading to nonsense inputs.",
        isCorrect: true,
      },
      {
        text: "Different models may use different subword vocabularies and special tokens.",
        isCorrect: true,
      },
      {
        text: "Any tokenizer can be safely interchanged without affecting performance.",
        isCorrect: false,
      },
    ],
    explanation:
      "Tokenizer and model are co-designed; mismatched token IDs break the link between text pieces and learned embeddings.",
  },
  {
    id: "ch2-q50",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which high-level capabilities of transformers explain their success in modern NLP?",
    options: [
      {
        text: "Ability to model long-range dependencies via attention.",
        isCorrect: true,
      },
      {
        text: "Massive parallelism during training on GPUs/TPUs.",
        isCorrect: true,
      },
      {
        text: "Flexibility to be pretrained once and adapted to many downstream tasks.",
        isCorrect: true,
      },
      {
        text: "Requirement that every dataset must be labeled by humans.",
        isCorrect: false,
      },
    ],
    explanation:
      "Transformers combine efficient self-attention, scalable training, and self-supervised pretraining, which together enable strong performance on diverse language tasks.",
  },
    {
    id: "ch2-q51",
    chapter: 2,
    difficulty: "easy",
    prompt: "Why did transformers replace RNNs in many NLP tasks?",
    options: [
      {
        text: "Transformers can process all tokens in a sequence in parallel.",
        isCorrect: true,
      },
      {
        text: "Transformers completely avoid using attention mechanisms.",
        isCorrect: false,
      },
      {
        text: "RNNs struggle with very long sequences and vanishing gradients.",
        isCorrect: true,
      },
      {
        text: "RNNs were designed only for images, not text.",
        isCorrect: false,
      },
    ],
    explanation:
      "Transformers use self-attention and parallel computation, while RNNs are sequential and suffer from long-range dependency and gradient issues.",
  },
  {
    id: "ch2-q52",
    chapter: 2,
    difficulty: "easy",
    prompt: "What is a 'sequence-to-sequence' (seq2seq) model in machine translation?",
    options: [
      {
        text: "A model with an encoder that reads the source sentence and a decoder that generates the target sentence.",
        isCorrect: true,
      },
      {
        text: "A model that always outputs the same fixed-length vector instead of text.",
        isCorrect: false,
      },
      {
        text: "A model architecture originally used with RNNs and later extended with attention.",
        isCorrect: true,
      },
      {
        text: "A model that never uses hidden states.",
        isCorrect: false,
      },
    ],
    explanation:
      "Seq2seq models map an input sequence to an output sequence using an encoder and a decoder; attention later improved this architecture.",
  },
  {
    id: "ch2-q53",
    chapter: 2,
    difficulty: "easy",
    prompt: "What is an 'attention weight' in an attention mechanism?",
    options: [
      {
        text: "A number that indicates how important a specific input token is for the current prediction.",
        isCorrect: true,
      },
      {
        text: "A learned scalar that is always the same for every token.",
        isCorrect: false,
      },
      {
        text: "A negative number that is ignored by the model.",
        isCorrect: false,
      },
      {
        text: "A special token used during tokenization.",
        isCorrect: false,
      },
    ],
    explanation:
      "Attention weights are normalized scores (often from softmax) that tell the model how strongly to focus on each input token.",
  },
  {
    id: "ch2-q54",
    chapter: 2,
    difficulty: "easy",
    prompt: "In self-attention, what does it mean that 'each token attends to all other tokens'?",
    options: [
      {
        text: "For every token, the model computes how related it is to every token (including itself) in the sequence.",
        isCorrect: true,
      },
      {
        text: "The model only compares each token with the first token.",
        isCorrect: false,
      },
      {
        text: "The representation of a token is updated using a weighted combination of all value vectors.",
        isCorrect: true,
      },
      {
        text: "Tokens are processed completely independently with no interaction.",
        isCorrect: false,
      },
    ],
    explanation:
      "Self-attention builds new token representations by comparing each token’s query with all keys and mixing the corresponding values.",
  },
  {
    id: "ch2-q55",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about queries, keys, and values (Q, K, V) in self-attention are TRUE?",
    options: [
      {
        text: "They are different linear projections of the same input token representations.",
        isCorrect: true,
      },
      {
        text: "Queries are used to ask 'what am I looking for' at each position.",
        isCorrect: true,
      },
      {
        text: "Keys and values are used to provide information about all positions that can be attended to.",
        isCorrect: true,
      },
      {
        text: "They are always fixed, hand-designed vectors that never change during training.",
        isCorrect: false,
      },
    ],
    explanation:
      "Q, K, and V come from multiplying inputs by learned matrices; queries match against keys, and values carry information to be combined.",
  },
  {
    id: "ch2-q56",
    chapter: 2,
    difficulty: "easy",
    prompt: "What problem does the scaling factor 1/√dₖ in scaled dot-product attention address?",
    options: [
      {
        text: "Without scaling, dot products can become very large when vectors are high-dimensional.",
        isCorrect: true,
      },
      {
        text: "Large dot products can make softmax extremely peaked with very small gradients.",
        isCorrect: true,
      },
      {
        text: "Scaling keeps the softmax inputs in a more moderate range.",
        isCorrect: true,
      },
      {
        text: "It is only used to make the formula look more complicated.",
        isCorrect: false,
      },
    ],
    explanation:
      "Dividing by √dₖ stabilizes the magnitude of dot products so that softmax doesn’t collapse onto a few positions and gradients remain usable.",
  },
  {
    id: "ch2-q57",
    chapter: 2,
    difficulty: "easy",
    prompt: "What is the main purpose of multi-head self-attention?",
    options: [
      {
        text: "To let the model look at the sequence from several 'perspectives' at the same time.",
        isCorrect: true,
      },
      {
        text: "To have several independent attention heads that can specialize in different patterns.",
        isCorrect: true,
      },
      {
        text: "To concatenate the outputs of all heads and mix them with a final linear layer.",
        isCorrect: true,
      },
      {
        text: "To ensure that every head uses exactly the same attention pattern.",
        isCorrect: false,
      },
    ],
    explanation:
      "Multiple heads with different Q/K/V projections can focus on different relationships; their outputs are concatenated and projected back.",
  },
  {
    id: "ch2-q58",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about the computational cost of self-attention are TRUE?",
    options: [
      {
        text: "The attention matrix compares every token with every other token.",
        isCorrect: true,
      },
      {
        text: "The time and memory cost grow roughly with the square of sequence length.",
        isCorrect: true,
      },
      {
        text: "This quadratic cost becomes a bottleneck for very long sequences.",
        isCorrect: true,
      },
      {
        text: "Self-attention always has lower cost than RNNs, regardless of sequence length.",
        isCorrect: false,
      },
    ],
    explanation:
      "Self-attention forms a T×T matrix of interactions, leading to O(T²) time and memory, which is expensive for long contexts.",
  },
  {
    id: "ch2-q59",
    chapter: 2,
    difficulty: "easy",
    prompt: "Why do transformers add positional encodings to token embeddings?",
    options: [
      {
        text: "Because self-attention alone is insensitive to the order of tokens.",
        isCorrect: true,
      },
      {
        text: "Because we want the model to know where each token is in the sequence.",
        isCorrect: true,
      },
      {
        text: "Because embeddings cannot represent word meaning without them.",
        isCorrect: false,
      },
      {
        text: "Because positional encodings are required only for very short sentences.",
        isCorrect: false,
      },
    ],
    explanation:
      "Self-attention treats tokens as a set; positional encodings inject order information so the model can distinguish different word orders.",
  },
  {
    id: "ch2-q60",
    chapter: 2,
    difficulty: "easy",
    prompt: "What characterizes sinusoidal positional encodings used in the original transformer?",
    options: [
      {
        text: "They use sine and cosine functions with different frequencies over the position index.",
        isCorrect: true,
      },
      {
        text: "They do not introduce extra learned parameters.",
        isCorrect: true,
      },
      {
        text: "They require re-training if we want to handle slightly longer sequences.",
        isCorrect: false,
      },
      {
        text: "They are randomly initialized and updated by gradient descent.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sinusoidal encodings are fixed functions of position (no learned weights) and generalize to longer sequences without retraining.",
  },
  {
    id: "ch2-q61",
    chapter: 2,
    difficulty: "easy",
    prompt: "What is a 'transformer block' in modern architectures?",
    options: [
      {
        text: "A repeated building unit that combines multi-head self-attention, a feedforward network, residual connections, and layer normalization.",
        isCorrect: true,
      },
      {
        text: "A layer that contains only convolution operations.",
        isCorrect: false,
      },
      {
        text: "A special token that marks sentence boundaries.",
        isCorrect: false,
      },
      {
        text: "A pre-processing step applied before tokenization.",
        isCorrect: false,
      },
    ],
    explanation:
      "Transformers stack identical blocks, each with attention and a position-wise feedforward network plus residual and normalization layers.",
  },
  {
    id: "ch2-q62",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about the feedforward network (FFN) inside a transformer block are TRUE?",
    options: [
      {
        text: "It applies two linear layers with a non-linear activation (like ReLU) in between.",
        isCorrect: true,
      },
      {
        text: "It is applied independently to each token’s representation.",
        isCorrect: true,
      },
      {
        text: "It mixes information across different time steps like self-attention does.",
        isCorrect: false,
      },
      {
        text: "It is mainly there to reduce the model’s capacity.",
        isCorrect: false,
      },
    ],
    explanation:
      "The FFN is a small MLP applied point-wise to each token, adding non-linearity and capacity without mixing positions.",
  },
  {
    id: "ch2-q63",
    chapter: 2,
    difficulty: "easy",
    prompt: "What do residual (skip) connections in transformers accomplish?",
    options: [
      {
        text: "They add the input of a sublayer to its output.",
        isCorrect: true,
      },
      {
        text: "They help gradients flow through many stacked layers.",
        isCorrect: true,
      },
      {
        text: "They make the loss landscape smoother and easier to optimize.",
        isCorrect: true,
      },
      {
        text: "They force every layer to output exactly the same as its input.",
        isCorrect: false,
      },
    ],
    explanation:
      "Residual additions create shortcut paths that stabilize training and allow deeper networks without forcing layers to be identity mappings.",
  },
  {
    id: "ch2-q64",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about layer normalization are TRUE?",
    options: [
      {
        text: "It normalizes activations across features for a single token.",
        isCorrect: true,
      },
      {
        text: "It uses that token’s mean and standard deviation.",
        isCorrect: true,
      },
      {
        text: "It then applies learned scale (γ) and shift (β) parameters.",
        isCorrect: true,
      },
      {
        text: "It requires very large batch sizes to work properly.",
        isCorrect: false,
      },
    ],
    explanation:
      "LayerNorm operates per token, using its feature statistics, and includes learnable γ and β; it does not depend on batch size like BatchNorm.",
  },
  {
    id: "ch2-q65",
    chapter: 2,
    difficulty: "easy",
    prompt: "What is 'cross-attention' in an encoder–decoder transformer?",
    options: [
      {
        text: "Attention in the decoder that uses encoder outputs as keys and values.",
        isCorrect: true,
      },
      {
        text: "Self-attention that only looks within the decoder sequence.",
        isCorrect: false,
      },
      {
        text: "A way for the decoder to condition its generation on the encoded source sentence.",
        isCorrect: true,
      },
      {
        text: "A mechanism that ignores the encoder completely.",
        isCorrect: false,
      },
    ],
    explanation:
      "Cross-attention mixes information from the encoder into the decoder by taking queries from decoder states and keys/values from encoder outputs.",
  },
  {
    id: "ch2-q66",
    chapter: 2,
    difficulty: "easy",
    prompt: "Why is a causal mask used in decoder self-attention for language modeling?",
    options: [
      {
        text: "To block attention to future tokens that should not be known yet.",
        isCorrect: true,
      },
      {
        text: "To speed up training by skipping all past tokens.",
        isCorrect: false,
      },
      {
        text: "Because we want the model to see the full sequence, including future tokens.",
        isCorrect: false,
      },
      {
        text: "Because masking makes the attention matrix symmetric.",
        isCorrect: false,
      },
    ],
    explanation:
      "The causal (triangular) mask sets scores for future positions to −∞ so that, after softmax, each token only attends to itself and earlier tokens.",
  },
  {
    id: "ch2-q67",
    chapter: 2,
    difficulty: "easy",
    prompt: "How is the probability of a sentence usually factorized in an autoregressive language model?",
    options: [
      {
        text: "As a product of conditional probabilities for each token given all previous tokens.",
        isCorrect: true,
      },
      {
        text: "Using the chain rule of probability.",
        isCorrect: true,
      },
      {
        text: "As a sum of independent probabilities for each token.",
        isCorrect: false,
      },
      {
        text: "By assuming that all tokens are independent of context.",
        isCorrect: false,
      },
    ],
    explanation:
      "Autoregressive models apply the chain rule: P(w₁…wₙ) = ∏ᵢ P(wᵢ | w₁…wᵢ₋₁).",
  },
  {
    id: "ch2-q68",
    chapter: 2,
    difficulty: "easy",
    prompt: "In training a language model, what are 'logits'?",
    options: [
      {
        text: "Raw, unnormalized scores for each vocabulary token before applying softmax.",
        isCorrect: true,
      },
      {
        text: "Probabilities that already sum exactly to 1.",
        isCorrect: false,
      },
      {
        text: "The values passed into the cross-entropy loss after softmax.",
        isCorrect: false,
      },
      {
        text: "A vector of real numbers that is converted to probabilities by softmax.",
        isCorrect: true,
      },
    ],
    explanation:
      "Logits are the pre-softmax scores output by the model; softmax turns them into a probability distribution over the vocabulary.",
  },
  {
    id: "ch2-q69",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about cross-entropy loss in language modeling are TRUE?",
    options: [
      {
        text: "It measures the difference between the predicted distribution and the true distribution over the vocabulary.",
        isCorrect: true,
      },
      {
        text: "The true distribution is usually a one-hot vector for the correct next token.",
        isCorrect: true,
      },
      {
        text: "Minimizing cross-entropy encourages the model to assign high probability to the correct token.",
        isCorrect: true,
      },
      {
        text: "It is unrelated to probabilities and is only used for regression tasks.",
        isCorrect: false,
      },
    ],
    explanation:
      "Cross-entropy compares predicted probabilities with one-hot targets, pushing the model to concentrate probability mass on the correct token.",
  },
  {
    id: "ch2-q70",
    chapter: 2,
    difficulty: "easy",
    prompt: "What is 'teacher forcing' during training of sequence models?",
    options: [
      {
        text: "Feeding the ground-truth previous token to the model when predicting the next one.",
        isCorrect: true,
      },
      {
        text: "Always feeding the model its own previous predictions.",
        isCorrect: false,
      },
      {
        text: "A method that completely avoids using the training data.",
        isCorrect: false,
      },
      {
        text: "A technique used only at inference time.",
        isCorrect: false,
      },
    ],
    explanation:
      "Teacher forcing stabilizes training by conditioning each prediction on the real previous token rather than on possibly wrong model outputs.",
  },
  {
    id: "ch2-q71",
    chapter: 2,
    difficulty: "easy",
    prompt: "Why is greedy decoding (always picking the top-probability token) often not ideal?",
    options: [
      {
        text: "It tends to produce repetitive, generic, and sometimes boring text.",
        isCorrect: true,
      },
      {
        text: "It ignores other plausible tokens that could lead to more diverse generations.",
        isCorrect: true,
      },
      {
        text: "It is much slower than sampling-based methods.",
        isCorrect: false,
      },
      {
        text: "It can get stuck in loops like repeating short phrases.",
        isCorrect: false,
      },
    ],
    explanation:
      "Greedy decoding is deterministic and conservative; it often yields safe but repetitive outputs and misses creative alternatives.",
  },
  {
    id: "ch2-q72",
    chapter: 2,
    difficulty: "easy",
    prompt: "What is the main idea behind subword tokenization methods like Byte-Pair Encoding (BPE)?",
    options: [
      {
        text: "Represent rare words as combinations of more frequent smaller units.",
        isCorrect: true,
      },
      {
        text: "Force every word to be a single token regardless of frequency.",
        isCorrect: false,
      },
      {
        text: "Completely avoid splitting any word into pieces.",
        isCorrect: false,
      },
      {
        text: "Use only individual characters as tokens in all cases.",
        isCorrect: false,
      },
    ],
    explanation:
      "BPE builds a vocabulary of frequent character sequences so common words stay whole and rare words are decomposed into reusable subword pieces.",
  },
  {
    id: "ch2-q73",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about the <UNK> (unknown) token and subword tokenization are TRUE?",
    options: [
      {
        text: "Without subword methods, many rare or new words would be mapped to a single <UNK> token.",
        isCorrect: true,
      },
      {
        text: "Subword vocabularies greatly reduce how often <UNK> has to be used.",
        isCorrect: true,
      },
      {
        text: "<UNK> is still sometimes needed for truly unrepresentable symbols.",
        isCorrect: false,
      },
      {
        text: "Subword tokenization makes it impossible to model prefixes and suffixes.",
        isCorrect: false,
      },
    ],
    explanation:
      "Subword tokenization breaks unseen words into smaller known pieces, so the model rarely needs to fall back to a single unknown token.",
  },
  {
    id: "ch2-q74",
    chapter: 2,
    difficulty: "easy",
    prompt: "What is the key idea of masked language modeling (MLM) used for BERT?",
    options: [
      {
        text: "Randomly mask some tokens and train the model to predict the original tokens using both left and right context.",
        isCorrect: true,
      },
      {
        text: "Mask every token so that no context remains.",
        isCorrect: false,
      },
      {
        text: "Train the model to always output the <MASK> symbol.",
        isCorrect: false,
      },
      {
        text: "Only allow the model to look at tokens to the left of the mask.",
        isCorrect: false,
      },
    ],
    explanation:
      "MLM hides a subset of tokens with a special mask and has the model reconstruct them from the full bidirectional context.",
  },
  {
    id: "ch2-q75",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements correctly compare BERT-style encoders and GPT-2-style decoders?",
    options: [
      {
        text: "BERT is bidirectional and mainly used for understanding tasks (classification, QA, etc.).",
        isCorrect: true,
      },
      {
        text: "GPT-2 is a causal decoder trained to predict the next token.",
        isCorrect: true,
      },
      {
        text: "GPT-2 generates text autoregressively by feeding its previous outputs back as input.",
        isCorrect: true,
      },
      {
        text: "BERT is primarily trained as a left-to-right language model.",
        isCorrect: false,
      },
    ],
    explanation:
      "BERT uses MLM with bidirectional context for understanding; GPT-2 is a decoder-only causal language model for generation.",
  },
  {
    id: "ch2-q76",
    chapter: 2,
    difficulty: "easy",
    prompt: "In BERT, what is the [CLS] token commonly used for?",
    options: [
      {
        text: "To obtain a single vector that summarizes the entire input sequence for classification.",
        isCorrect: true,
      },
      {
        text: "To mark the beginning of the input sequence.",
        isCorrect: true,
      },
      {
        text: "To serve as a padding token for short sentences.",
        isCorrect: false,
      },
      {
        text: "To signal the end of a document.",
        isCorrect: false,
      },
    ],
    explanation:
      "BERT adds [CLS] at the start; its final hidden state is treated as a sequence-level representation for tasks like sentiment or entailment.",
  },
  {
    id: "ch2-q77",
    chapter: 2,
    difficulty: "easy",
    prompt: "What is the high-level idea of transfer learning with pretrained transformers?",
    options: [
      {
        text: "First pretrain on massive unlabeled text to learn general language patterns.",
        isCorrect: true,
      },
      {
        text: "Then adapt the model to a specific task using a much smaller labeled dataset.",
        isCorrect: true,
      },
      {
        text: "Reuse the same pretrained model for many tasks without starting from random initialization each time.",
        isCorrect: true,
      },
      {
        text: "Avoid using any labeled data for downstream tasks.",
        isCorrect: false,
      },
    ],
    explanation:
      "Transfer learning leverages broad language knowledge from pretraining and fine-tunes the model on smaller supervised datasets.",
  },
  {
    id: "ch2-q78",
    chapter: 2,
    difficulty: "easy",
    prompt: "During standard supervised fine-tuning for classification, what usually changes?",
    options: [
      {
        text: "We add a small classification head (for example, a linear layer) on top of the pretrained model.",
        isCorrect: true,
      },
      {
        text: "We train this head using labeled examples and a loss such as cross-entropy.",
        isCorrect: true,
      },
      {
        text: "We may optionally update some or all of the transformer’s internal weights.",
        isCorrect: false,
      },
      {
        text: "We always discard all pretrained weights and start from scratch.",
        isCorrect: false,
      },
    ],
    explanation:
      "Fine-tuning attaches a task-specific head and trains it (and sometimes parts of the backbone) using labeled data; we keep the pretrained weights.",
  },
  {
    id: "ch2-q79",
    chapter: 2,
    difficulty: "easy",
    prompt: "What is the main goal of knowledge distillation?",
    options: [
      {
        text: "To train a smaller 'student' model to mimic a larger 'teacher' model’s behavior.",
        isCorrect: true,
      },
      {
        text: "To enlarge a model so it has more parameters than the teacher.",
        isCorrect: false,
      },
      {
        text: "To completely remove the need for training data.",
        isCorrect: false,
      },
      {
        text: "To convert a neural network into a decision tree.",
        isCorrect: false,
      },
    ],
    explanation:
      "Distillation compresses a big model into a smaller one by having the student approximate the teacher’s outputs, keeping performance while reducing size.",
  },
  {
    id: "ch2-q80",
    chapter: 2,
    difficulty: "easy",
    prompt: "Why is it important to use the tokenizer that matches a given pretrained transformer model?",
    options: [
      {
        text: "Because the model’s embedding matrix expects the exact token IDs produced by its tokenizer.",
        isCorrect: true,
      },
      {
        text: "Because different models may use different subword vocabularies and special tokens.",
        isCorrect: false,
      },
      {
        text: "Using a mismatched tokenizer can feed completely wrong indices into the embedding layer.",
        isCorrect: false,
      },
      {
        text: "Any tokenizer can be swapped without affecting the model.",
        isCorrect: false,
      },
    ],
    explanation:
      "Pretrained models are tightly coupled to their tokenizers; mismatching them scrambles the mapping from text to embeddings and breaks performance.",
  },

];
