import type { Question } from "../../quiz";

export const stanfordCME2952026Lecture1Questions: Question[] = [
  // Replaces retired ID cme295-2026-lect1-q01; new reasoning task.
  {
    id: "cme295-2026-lect1-q051",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "A subword tokenizer can use the tokens re, read, ing, red, and ring. Which token sequence reconstructs the exact string rereading by concatenation?",
    options: [
      {
        text: "re, read, ing",
        isCorrect: true,
      },
      {
        text: "re, red, ing",
        isCorrect: false,
      },
      {
        text: "read, re, ing",
        isCorrect: false,
      },
      {
        text: "re, ring, read",
        isCorrect: false,
      },
    ],
    explanation:
      "Concatenating re + read + ing gives rereading, including both occurrences of re. Swapping the first two tokens gives readreing; replacing read with red loses a letter, and the sequence using ring produces a different string. A valid tokenization must reconstruct the input, even when several vocabulary entries look related.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q02",
    chapter: 1,
    difficulty: "easy",
    prompt:
      'A tokenizer represents "reading" as "read" and "##ing". Which statements correctly interpret this representation?',
    options: [
      {
        text: "The two token IDs must be consecutive integers.",
        isCorrect: false,
      },
      {
        text: "The suffix must have a vector twice as long as the root vector.",
        isCorrect: false,
      },
      {
        text: "The vocabulary contains reusable units smaller than this word.",
        isCorrect: true,
      },
      {
        text: "The word occupies two token positions.",
        isCorrect: true,
      },
    ],
    explanation:
      "Each emitted unit occupies one sequence position and receives a vocabulary ID and an embedding of the model dimension. Subword composition does not impose an ordering on IDs or change the vector width according to the length or grammatical role of a piece.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q03",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements describe advantages of subword tokenization?",
    options: [
      {
        text: "It balances between word-level semantics and character-level robustness.",
        isCorrect: true,
      },
      {
        text: "It reduces the chance of truly unseen tokens by decomposing rare words.",
        isCorrect: true,
      },
      {
        text: "It allows common roots or stems (for example, 'bear' in 'bear' and 'bears') to be shared.",
        isCorrect: true,
      },
      {
        text: "It tends to keep sequence lengths moderate compared with character-level tokenization.",
        isCorrect: true,
      },
    ],
    explanation:
      "Subword tokenization is a compromise between word-level and character-level tokenization. It can share reusable pieces across related words, reduce truly unseen tokens, and keep sequences shorter than pure character tokenization while still handling rare or morphologically varied words.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q04",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "A character tokenizer includes all characters occurring in a misspelled input. Which consequences follow when it replaces a whole-word tokenizer?",
    options: [
      {
        text: "The sequence will generally need more token positions.",
        isCorrect: true,
      },
      {
        text: "Processing the same text may cost more because its token sequence is longer.",
        isCorrect: true,
      },
      {
        text: "The misspelling can be represented without a whole-word vocabulary entry.",
        isCorrect: true,
      },
      {
        text: "An unfamiliar word can reuse known character tokens.",
        isCorrect: true,
      },
    ],
    explanation:
      "Known characters can spell an unfamiliar or misspelled word, avoiding the need for that entire word to have a vocabulary entry. The tradeoff is longer sequences and less meaning carried by any one token; context processing must combine the characters into useful word-level information.",
  },
  // Replaces retired ID cme295-2026-lect1-q05; new reasoning task.
  {
    id: "cme295-2026-lect1-q052",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Compare a word tokenizer with vocabulary 50,000 and sequence length 100 against a subword tokenizer with vocabulary 8,000 and sequence length 170 for the same text. Both models use embedding width 256 and dense self-attention. Ignore padding and biases. Select the true comparisons.",
    options: [
      {
        text: "The subword attention score matrix has 1.7 times as many entries per head.",
        isCorrect: false,
      },
      {
        text: "The word input embedding table has 12,800 parameters.",
        isCorrect: false,
      },
      {
        text: "The subword attention score matrix has 2.89 times as many entries per head.",
        isCorrect: true,
      },
      {
        text: "The subword input embedding table has 2,048,000 parameters.",
        isCorrect: true,
      },
    ],
    explanation:
      "The embedding parameter counts are vocabulary size times width: 8,000 × 256 = 2,048,000 versus 50,000 × 256 = 12,800,000. Dense attention scores scale with the square of sequence length, so the ratio is 170²/100² = 2.89. The smaller vocabulary saves embedding parameters while the longer sequence increases attention work; 1.7 is the length ratio, not the score-count ratio.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q06",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "A batch contains sequences with 5 and 8 tokens, and both are padded to length 8. Which statements correctly describe the resulting representation?",
    options: [
      {
        text: "[PAD] denotes a batching placeholder rather than missing vocabulary content.",
        isCorrect: true,
      },
      {
        text: "The meaningful content lengths remain 5 and 8.",
        isCorrect: true,
      },
      {
        text: "The shorter sequence receives three [PAD] tokens.",
        isCorrect: true,
      },
      {
        text: "An attention mask can prevent padding from affecting content representations.",
        isCorrect: true,
      },
    ],
    explanation:
      "Padding makes batch tensors rectangular without adding meaningful text. Three placeholders extend the shorter row, and masks let the computation distinguish those placeholders from content; [UNK] instead denotes an unrepresented vocabulary unit, while [EOS] marks a sequence boundary.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q07",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "A translation decoder begins with [BOS] and later predicts [EOS]. Which interpretations are correct?",
    options: [
      {
        text: "[EOS] is a replacement for an unknown source word.",
        isCorrect: false,
      },
      {
        text: "[BOS] supplies the initial decoder input.",
        isCorrect: true,
      },
      {
        text: "[EOS] tells the application that the generated sequence has ended.",
        isCorrect: true,
      },
      {
        text: "[BOS] carries the padding count for the batch.",
        isCorrect: false,
      },
    ],
    explanation:
      "Beginning-of-sequence and end-of-sequence markers delimit generation: one starts the target prefix and the other provides a learned stopping signal. Unknown content and batching placeholders have separate roles, represented by [UNK] and [PAD] where those conventions are used.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q08",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Distinct tokens are represented by one-hot vectors in a vocabulary of size 10,000. Which statements follow?",
    options: [
      {
        text: "The dot product of two distinct token vectors is zero.",
        isCorrect: true,
      },
      {
        text: "Each vector has 10,000 components.",
        isCorrect: true,
      },
      {
        text: "The dot product of a token vector with itself is one.",
        isCorrect: true,
      },
      {
        text: "Each vector has one nonzero component equal to 1.",
        isCorrect: true,
      },
    ],
    explanation:
      "A one-hot vector is a vocabulary-sized basis vector, so its self-dot product is one and distinct basis vectors are orthogonal. This geometry identifies tokens but treats related and unrelated token pairs alike; semantic similarity must be learned in a different representation.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q09",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "An embedding table has shape \\(V\\times d\\), with \\(V=10{,}000\\) and \\(d=128\\). Which quantities are correct?",
    options: [
      {
        text: "The table contains 1,280,000 scalar parameters.",
        isCorrect: true,
      },
      {
        text: "A lookup for one token returns 10,000 components.",
        isCorrect: false,
      },
      {
        text: "A lookup for one token returns 128 components.",
        isCorrect: true,
      },
      {
        text: "The table has 128 distinct token entries.",
        isCorrect: false,
      },
    ],
    explanation:
      "The vocabulary supplies the number of rows and the embedding dimension supplies the number of columns. Multiplying 10,000 by 128 gives 1,280,000 learned scalars, but selecting one row returns a vector of length 128 rather than a vocabulary-sized probability distribution.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q10",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about the Continuous Bag of Words (CBOW) and Skip-Gram models in Word2vec are correct?",
    options: [
      {
        text: "Skip-Gram predicts surrounding context words from a central target word.",
        isCorrect: true,
      },
      {
        text: "CBOW predicts a target word from its surrounding context words.",
        isCorrect: true,
      },
      {
        text: "Both are trained on large unlabeled text corpora.",
        isCorrect: true,
      },
      {
        text: "Both treat these prediction tasks as proxies for learning useful word embeddings.",
        isCorrect: true,
      },
    ],
    explanation:
      "Continuous Bag of Words (CBOW) and Skip-Gram are two Word2vec proxy tasks that use surrounding context in opposite directions. CBOW predicts a center word from its context, while Skip-Gram predicts nearby context words from a center word, and both use those prediction pressures to learn useful embeddings.",
  },
  // Replaces retired ID cme295-2026-lect1-q11; new reasoning task.
  {
    id: "cme295-2026-lect1-q053",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "A trained continuous bag-of-words (CBOW) model uses input embeddings \\(W_{\\rm in}\\in\\mathbb R^{V\\times d}\\) and an output matrix \\(W_{\\rm out}\\in\\mathbb R^{d\\times V}\\). A downstream classifier imports only the input embedding table. Select all true statements about what is transferred.",
    options: [
      {
        text: "Averaging imported input rows can represent a multiword context without importing \\(W_{\\rm out}\\).",
        isCorrect: true,
      },
      {
        text: "Looking up token \\(j\\) retrieves row \\(j\\) of \\(W_{\\rm in}\\).",
        isCorrect: true,
      },
      {
        text: "The downstream classifier still needs its own mapping from features to its task labels.",
        isCorrect: true,
      },
      {
        text: "The imported rows were shaped by a word-prediction objective even if that prediction head is discarded.",
        isCorrect: true,
      },
    ],
    explanation:
      "CBOW learns input rows by using a context representation to predict a target word through the output matrix. Keeping those rows transfers learned features, not a ready-made classifier for arbitrary labels. Row lookup and averaging remain available operations on the imported table, whereas the downstream task's prediction head must be supplied separately.",
  },
  // Replaces retired ID cme295-2026-lect1-q12; new reasoning task.
  {
    id: "cme295-2026-lect1-q054",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which representation is a static word embedding for the token bank?",
    options: [
      {
        text: "An attention output recomputed from the sentence containing bank.",
        isCorrect: false,
      },
      {
        text: "The recurrent hidden state after reading the prefix ending with bank.",
        isCorrect: false,
      },
      {
        text: "A vector recomputed from the words on both sides of bank in each sentence.",
        isCorrect: false,
      },
      {
        text: "A learned vocabulary row retrieved identically in river bank and bank loan.",
        isCorrect: true,
      },
    ],
    explanation:
      "A static embedding assigns one learned vector to a vocabulary entry, so both senses initially receive the same row. The other candidates use surrounding tokens or a prefix to compute the representation for that occurrence. A model may start with static lookup vectors and then contextualize them; the resulting hidden states are different objects from the original embedding row.",
  },
  // Replaces retired ID cme295-2026-lect1-q13; new reasoning task.
  {
    id: "cme295-2026-lect1-q055",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Three unit-length learned embeddings are \\(u=(1,0)\\), \\(v=(0.8,0.6)\\), and \\(w=(0,1)\\). Compare their cosine similarities with distinct one-hot vocabulary rows. Select all true statements.",
    options: [
      {
        text: "\\(\\cos(v,w)=0.8\\), so it equals \\(\\cos(u,v)\\).",
        isCorrect: false,
      },
      {
        text: "Replacing these learned vectors with distinct one-hot rows preserves their pairwise similarity ranking.",
        isCorrect: false,
      },
      {
        text: "Distinct one-hot rows would have zero cosine similarity regardless of which pair they represent.",
        isCorrect: true,
      },
      {
        text: "\\(\\cos(u,v)=0.8\\), greater than \\(\\cos(u,w)=0\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "For unit vectors cosine similarity is their dot product: u·v = 0.8, u·w = 0, and v·w = 0.6. Distinct one-hot rows are orthogonal and give zero for each pair, losing this ranking. Learned dense embeddings can therefore encode graded relationships that a bare one-hot identity representation does not encode.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q14",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Let the rows of an embedding table be \\(E_1=(2,0)\\), \\(E_2=(0,3)\\), and \\(E_3=(1,1)\\). A row one-hot vector \\(x=(0,1,0)\\) is multiplied by \\(E\\). What is \\(xE\\)?",
    options: [
      {
        text: "\\((3,4)\\)",
        isCorrect: false,
      },
      {
        text: "\\((0,3)\\)",
        isCorrect: true,
      },
      {
        text: "\\((0,1,0)\\)",
        isCorrect: false,
      },
      {
        text: "\\((2,0)\\)",
        isCorrect: false,
      },
    ],
    explanation:
      "Multiplying a one-hot row by the table selects the row whose indicator is one. Here that is the second embedding, (0,3); summing all rows would give (3,4), while returning the input one-hot vector or the first row confuses the lookup operation.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q15",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "A simple embedding-learning network has vocabulary-sized input and output layers and a narrow hidden layer. Which statements correctly trace one prediction?",
    options: [
      {
        text: "The hidden vector and the output probabilities index different kinds of quantities.",
        isCorrect: true,
      },
      {
        text: "The output distribution has one probability per vocabulary token.",
        isCorrect: true,
      },
      {
        text: "The input one-hot vector identifies a token.",
        isCorrect: true,
      },
      {
        text: "The narrow hidden layer produces a dense representation.",
        isCorrect: true,
      },
    ],
    explanation:
      "The input and output widths match the vocabulary because the proxy task predicts a token identity. The hidden width is the embedding dimension, so its vector compresses the input into features; those features must then be projected back to vocabulary scores rather than interpreted as token probabilities directly.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q16",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "A Recurrent Neural Network (RNN) computes a hidden state from the current token and the previous state. Which statements follow?",
    options: [
      {
        text: "The same recurrent parameters can be reused at successive time steps.",
        isCorrect: true,
      },
      {
        text: "The previous state can carry information from earlier tokens.",
        isCorrect: true,
      },
      {
        text: "Reordering tokens can change the sequence of states.",
        isCorrect: true,
      },
      {
        text: "Computing a later state depends on the earlier state being available.",
        isCorrect: true,
      },
    ],
    explanation:
      "Recurrence carries information through a state that is repeatedly updated with shared parameters. This makes word order consequential and creates a sequential dependency in computation; unlike a static embedding average, the state evolves as the sequence is processed.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q17",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "A gradient contribution is multiplied by \\(0.5\\) at each of ten recurrent transitions. What fraction reaches the earliest transition in this simplified chain?",
    options: [
      {
        text: "\\(1/20\\)",
        isCorrect: false,
      },
      {
        text: "\\(5\\)",
        isCorrect: false,
      },
      {
        text: "\\(1/10\\)",
        isCorrect: false,
      },
      {
        text: "\\(1/1024\\)",
        isCorrect: true,
      },
    ],
    explanation:
      "Ten multiplications produce 0.5 to the tenth power, which is 1/1024. This illustrates a vanishing gradient: a signal about a distant dependency becomes very small, rather than growing as in an exploding-gradient case or being divided merely by the number of time steps.",
  },
  // Replaces retired ID cme295-2026-lect1-q18; new reasoning task.
  {
    id: "cme295-2026-lect1-q056",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "One long short-term memory (LSTM) cell coordinate updates as \\(c_t=f_t c_{t-1}+i_t\\tilde c_t\\). Given \\(c_{t-1}=2\\), \\(f_t=0.9\\), \\(i_t=0.1\\), and \\(\\tilde c_t=-1\\), what is \\(c_t\\)?",
    options: [
      {
        text: "\\(1.9\\)",
        isCorrect: false,
      },
      {
        text: "\\(-1.7\\)",
        isCorrect: false,
      },
      {
        text: "\\(0.8\\)",
        isCorrect: false,
      },
      {
        text: "\\(1.7\\)",
        isCorrect: true,
      },
    ],
    explanation:
      "The retained contribution is 0.9 × 2 = 1.8 and the new contribution is 0.1 × (−1) = −0.1, giving 1.7. A value of 1.9 reverses the candidate's sign, 0.8 omits the input gate on the candidate, and −1.7 reverses the final sign. Separate retention and input gates let the cell control old and new information independently.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q19",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A source word occurs far before the end of a sentence. Compare a plain recurrent encoder-decoder with a decoder that attends to source states. Which changes follow from attention?",
    options: [
      {
        text: "The distance in source positions sets the number of recurrent edges in the attention lookup itself.",
        isCorrect: false,
      },
      {
        text: "The source state can contribute through its value representation.",
        isCorrect: true,
      },
      {
        text: "The decoder can access the distant source state without relying solely on the final recurrent summary.",
        isCorrect: true,
      },
      {
        text: "The weight assigned to that source state can vary across target positions.",
        isCorrect: true,
      },
    ],
    explanation:
      "Attention makes source states directly available at each target step and computes weights for the current query. The lookup path does not require a separate recurrent transition for each intervening source token, although the recurrent encoder may still have been used to construct those states.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q20",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which output structure most directly matches Named Entity Recognition (NER), where mentions such as person names and locations are identified in text?",
    options: [
      {
        text: "A translated sequence with an independently chosen output length.",
        isCorrect: false,
      },
      {
        text: "One sentiment score for the entire input.",
        isCorrect: false,
      },
      {
        text: "A next-word probability distribution without position labels.",
        isCorrect: false,
      },
      {
        text: "An entity tag associated with each relevant token position.",
        isCorrect: true,
      },
    ],
    explanation:
      "NER identifies entities within a sequence, so a token-tagging formulation associates labels with positions rather than one label with the whole sentence. Sentiment classification, translation, and next-token generation have different output structures even though they can use related sequence models.",
  },
  // Replaces retired ID cme295-2026-lect1-q21; new reasoning task.
  {
    id: "cme295-2026-lect1-q057",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "During teacher-forced training, a decoder receives the complete shifted target sequence. Compare a causal Transformer layer with a recurrent layer for computing the hidden states of that known sequence. Select all true statements.",
    options: [
      {
        text: "The Transformer's position computations can run in parallel while a causal mask restricts which positions each one reads.",
        isCorrect: true,
      },
      {
        text: "Providing all training targets makes the causal mask unnecessary for learning the generation-time conditional distributions.",
        isCorrect: false,
      },
      {
        text: "Parallel training positions let the decoder generate an unknown next token before generating its unknown predecessor.",
        isCorrect: false,
      },
      {
        text: "A recurrent layer must compute the previous hidden state before using it in the next position's recurrence.",
        isCorrect: true,
      },
    ],
    explanation:
      "Teacher forcing makes the inputs known, so a Transformer can form queries, keys, and values for the training positions together. Causal masking still enforces the prefix available at generation time, while a recurrence has a dependency through successive hidden states. At inference the generated inputs are unknown until sampled, so parallel training does not remove the autoregressive dependency.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q22",
    chapter: 1,
    difficulty: "easy",
    prompt: "In self-attention, which roles are correctly assigned?",
    options: [
      {
        text: "A query represents what a position is looking for.",
        isCorrect: true,
      },
      {
        text: "A key is compared with a query to determine a weight.",
        isCorrect: true,
      },
      {
        text: "A key is itself the final probability distribution over the vocabulary.",
        isCorrect: false,
      },
      {
        text: "A value supplies information mixed into the output.",
        isCorrect: true,
      },
    ],
    explanation:
      "Queries and keys determine compatibility between positions, and values carry the content that attention mixes. The resulting weights range over sequence positions, whereas vocabulary probabilities are produced by a separate output projection and softmax.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q23",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "For one head, \\(Q\\) and \\(K\\) have shape \\(5\\times8\\), and \\(V\\) has shape \\(5\\times6\\). Which shapes are correct?",
    options: [
      {
        text: "The attention output has shape \\(5\\times6\\).",
        isCorrect: true,
      },
      {
        text: "The attention output has shape \\(5\\times5\\).",
        isCorrect: false,
      },
      {
        text: "\\(QK^\\top\\) has shape \\(8\\times8\\).",
        isCorrect: false,
      },
      {
        text: "\\(QK^\\top\\) has shape \\(5\\times5\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "Each query is compared with five keys, producing a five-by-five matrix of position scores. Multiplying its normalized rows by the five value rows returns one six-dimensional context vector per query; feature widths do not become the number of positions.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q24",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "An attention row has weights \\((0.25,0.75)\\) and values \\(v_1=(2,0)\\), \\(v_2=(0,4)\\). What is the output vector?",
    options: [
      {
        text: "\\((1.5,1)\\)",
        isCorrect: false,
      },
      {
        text: "\\((0.25,0.75)\\)",
        isCorrect: false,
      },
      {
        text: "\\((0.5,3)\\)",
        isCorrect: true,
      },
      {
        text: "\\((2,4)\\)",
        isCorrect: false,
      },
    ],
    explanation:
      "Attention forms a weighted sum of value vectors, giving 0.25(2,0)+0.75(0,4)=(0.5,3). Reversing the weights changes the result, adding unweighted values ignores compatibility, and returning the weights confuses the lookup coefficients with the retrieved content.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q25",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Softmax is applied separately to each row of query-key scores. Which statements follow before any dropout?",
    options: [
      {
        text: "Each output row is a weighted average of value rows.",
        isCorrect: true,
      },
      {
        text: "The weights in a row sum to one.",
        isCorrect: true,
      },
      {
        text: "Each attention weight is nonnegative.",
        isCorrect: true,
      },
      {
        text: "Every column must also sum to one.",
        isCorrect: false,
      },
    ],
    explanation:
      "Row normalization makes each query distribute unit weight across the available keys, yielding a weighted average of their values. It does not normalize columns: one key can receive high attention from several queries, so column sums need not equal one.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q26",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "A query-key dot product is 12 and \\(d_k=9\\). What score enters softmax in scaled dot-product attention?",
    options: [
      {
        text: "12/9",
        isCorrect: false,
      },
      {
        text: "4",
        isCorrect: true,
      },
      {
        text: "12",
        isCorrect: false,
      },
      {
        text: "36",
        isCorrect: false,
      },
    ],
    explanation:
      "The scale divides the dot product by the square root of the key width, so 12 divided by 3 gives 4. Multiplying by the scale, dividing by the width itself, and omitting the scale are different operations and do not implement the stated attention rule.",
  },
  // Replaces retired ID cme295-2026-lect1-q27; new reasoning task.
  {
    id: "cme295-2026-lect1-q058",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Assume all 64 coordinates of a query \\(q\\) and key \\(k\\) are mutually independent, have mean zero and variance one. Let \\(z=q^\\top k\\). Select all true statements about the variance motivation for scaled dot-product attention.",
    options: [
      {
        text: "\\({\\rm Var}(z)=8\\).",
        isCorrect: false,
      },
      {
        text: "\\({\\rm Var}(z/\\sqrt{64})=1\\).",
        isCorrect: true,
      },
      {
        text: "\\({\\rm Var}(z/64)=1\\).",
        isCorrect: false,
      },
      {
        text: "\\({\\rm Var}(z)=64\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "Each product q_i k_i has mean zero and variance one under the stated independence assumptions. Adding 64 independent products gives variance 64; dividing a random variable by 8 divides its variance by 64. Eight is the unscaled standard deviation, and division by 64 would instead give variance 1/64. The square-root scale controls logit spread as head width grows.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q28",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "For one query, the scaled logits are \\((0,\\ln 3)\\). Which statements correctly describe the normalized attention?",
    options: [
      {
        text: "Adding 5 to both logits leaves the weights unchanged.",
        isCorrect: true,
      },
      {
        text: "The weights are (0, 1) because the larger logit wins.",
        isCorrect: false,
      },
      {
        text: "The second value receives three times the weight of the first.",
        isCorrect: true,
      },
      {
        text: "The weights are (0.25, 0.75).",
        isCorrect: true,
      },
    ],
    explanation:
      "Exponentiating gives unnormalized masses 1 and 3, so division by their sum yields one quarter and three quarters. A common additive shift cancels in softmax, while choosing the maximum with probability one would be a discrete selection rule rather than this weighted attention operation.",
  },
  // Replaces retired ID cme295-2026-lect1-q29; new reasoning task.
  {
    id: "cme295-2026-lect1-q059",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Let \\(Q,K,V\\in\\mathbb R^{n\\times n}\\), let \\(P\\) permute token rows, and define \\(Y={\\rm softmax}_{\\rm row}(QK^\\top)V\\). There is no mask or positional signal. After replacing \\(Q,K,V\\) with \\(PQ,PK,PV\\), which expression equals the new output for arbitrary \\(Q,K,V\\)?",
    options: [
      {
        text: "\\(PYP^\\top\\)",
        isCorrect: false,
      },
      {
        text: "\\(Y\\)",
        isCorrect: false,
      },
      {
        text: "\\(PY\\)",
        isCorrect: true,
      },
      {
        text: "\\(P^\\top Y\\)",
        isCorrect: false,
      },
    ],
    explanation:
      "The score matrix becomes P(QKᵀ)Pᵀ, which permutes both its query rows and key columns. Row-wise softmax respects that permutation, and multiplication by PV cancels the key-column permutation, leaving PY. The token rows follow the input permutation; the output features do not undergo an extra column permutation. Invariance or the inverse permutation would describe a different transformation.",
  },
  // Replaces retired ID cme295-2026-lect1-q30; new reasoning task.
  {
    id: "cme295-2026-lect1-q060",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "A multi-head attention layer has three tokens, model width 12, and three heads of width 4. Each head uses a separate \\(12\\times4\\) value projection; the concatenated result is projected back to width 12. Select all true shapes.",
    options: [
      {
        text: "Each head's attention weight matrix has shape \\(3\\times3\\).",
        isCorrect: true,
      },
      {
        text: "The concatenated head output has shape \\(3\\times12\\).",
        isCorrect: true,
      },
      {
        text: "The final output projection has shape \\(12\\times12\\).",
        isCorrect: true,
      },
      {
        text: "Each head's value matrix has shape \\(3\\times4\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "Multiplying the 3 × 12 token matrix by a 12 × 4 projection gives three value rows of width four. Each query attends over the three keys, so its weight matrix is 3 × 3 and its output remains 3 × 4. Concatenation joins the three feature widths without joining token rows, and a 12 × 12 projection restores the specified model width.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q31",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Four attention heads each output a \\(6\\times16\\) matrix. Which quantities describe concatenation along the feature dimension followed by projection to model width 64?",
    options: [
      {
        text: "The concatenated output has shape 6 by 64.",
        isCorrect: true,
      },
      {
        text: "A 64 by 64 output matrix is compatible with that concatenation.",
        isCorrect: true,
      },
      {
        text: "The concatenated feature width is 16.",
        isCorrect: false,
      },
      {
        text: "The sequence length after concatenation is 24.",
        isCorrect: false,
      },
    ],
    explanation:
      "Concatenation joins head features for each position, so four widths of 16 become 64 while the six positions remain six. A 64-by-64 projection preserves that width; stacking positions instead would combine tokens rather than head features.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q32",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Two tokens have identical query vectors in one unmasked head and attend to the same keys and values. Which statements hold?",
    options: [
      {
        text: "Different query position labels alone change these weights without any position mechanism.",
        isCorrect: false,
      },
      {
        text: "Their attention outputs are identical in this head.",
        isCorrect: true,
      },
      {
        text: "Their weights depend on comparing their value vectors with the keys.",
        isCorrect: false,
      },
      {
        text: "Their attention-weight rows are identical.",
        isCorrect: true,
      },
    ],
    explanation:
      "Identical queries scored against the same keys produce identical logits and normalized weights, hence the same weighted value sum. Position labels do not act by themselves without an encoding or bias, and the compatibility computation uses queries rather than values.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q33",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A head keeps queries and keys fixed but replaces one value vector. Which statements describe the immediate computation?",
    options: [
      {
        text: "Queries assigning weight to that value can receive changed outputs.",
        isCorrect: true,
      },
      {
        text: "The attention weights stay fixed.",
        isCorrect: true,
      },
      {
        text: "The altered value changes the query-key score matrix in this same computation.",
        isCorrect: false,
      },
      {
        text: "The row softmax must be recomputed from value similarities.",
        isCorrect: false,
      },
    ],
    explanation:
      "The score matrix depends on queries and keys, so changing a value does not alter the already defined attention weights. It changes the weighted content delivered to queries that attend to that position; a later layer could then compute different queries or keys from the changed outputs.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q34",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "An unmasked head processes 100 tokens. Which statements describe the pairwise attention computation and its score matrix?",
    options: [
      {
        text: "The two score-matrix axes index query positions and key positions.",
        isCorrect: true,
      },
      {
        text: "There are 10,000 query-key score entries.",
        isCorrect: true,
      },
      {
        text: "Each query has 100 candidate key positions.",
        isCorrect: true,
      },
      {
        text: "The score count grows by a factor of four if token length doubles.",
        isCorrect: true,
      },
    ],
    explanation:
      "Each of 100 query positions is compared with each of 100 key positions. The resulting square matrix has 10,000 entries and quadratic length scaling; vocabulary size determines the token embedding and output tables rather than the number of pairwise position scores.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q35",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "In the original encoder-decoder translation transformer, which statements describe the encoder when source padding is ignored?",
    options: [
      {
        text: "Its self-attention is not masked and can attend to all positions.",
        isCorrect: true,
      },
      {
        text: "It produces context-aware embeddings for each input position.",
        isCorrect: true,
      },
      {
        text: "It processes the input sequence in a fully parallel way using self-attention.",
        isCorrect: true,
      },
      {
        text: "It directly generates the target-language tokens one by one.",
        isCorrect: false,
      },
    ],
    explanation:
      "A transformer encoder builds context-aware representations for the input sequence. Its self-attention is not causally masked, so each source position can attend to every source position, while the separate decoder is responsible for generating target tokens.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q36",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which data flows occur in the original transformer decoder for translation?",
    options: [
      {
        text: "Unmasked target self-attention reads the next target word during training.",
        isCorrect: false,
      },
      {
        text: "Cross-attention reads the encoded source.",
        isCorrect: true,
      },
      {
        text: "Causal self-attention uses the target prefix.",
        isCorrect: true,
      },
      {
        text: "An output projection produces vocabulary logits.",
        isCorrect: true,
      },
    ],
    explanation:
      "The decoder combines a causally restricted target representation with source information through cross-attention. Its final state is projected to vocabulary scores; permitting the true next target word into the target-side attention would leak the prediction label.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q37",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "A source has 7 tokens and the decoder prefix has 3 tokens. Which statements describe decoder cross-attention?",
    options: [
      {
        text: "Its position-score matrix must be square because the model widths match.",
        isCorrect: false,
      },
      {
        text: "Its position-score matrix has shape 3 by 7.",
        isCorrect: true,
      },
      {
        text: "Queries come from the 3 target-side positions.",
        isCorrect: true,
      },
      {
        text: "Keys and values come from the 7 source-side representations.",
        isCorrect: true,
      },
    ],
    explanation:
      "Cross-attention compares target queries with source keys, so the two sequence lengths determine a rectangular three-by-seven score matrix. The feature dimensions must be compatible for dot products, but equal feature widths do not require equal source and target lengths.",
  },
  // Replaces retired ID cme295-2026-lect1-q38; new reasoning task.
  {
    id: "cme295-2026-lect1-q061",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "A decoder hidden-state matrix has shape \\(7\\times512\\). A linear language-model head maps each row to a 20,000-token vocabulary with no bias. Select all true statements.",
    options: [
      {
        text: "The head contains 140,000 trainable weights.",
        isCorrect: false,
      },
      {
        text: "The head weight matrix can have shape \\(512\\times20{,}000\\).",
        isCorrect: true,
      },
      {
        text: "The output logits have shape \\(7\\times20{,}000\\).",
        isCorrect: true,
      },
      {
        text: "The head requires a different \\(512\\times20{,}000\\) weight matrix for each of the seven token positions.",
        isCorrect: false,
      },
    ],
    explanation:
      "The same linear map is applied at every position, giving (7 × 512)(512 × 20,000) = 7 × 20,000 logits. Its parameter count is 512 × 20,000 = 10,240,000; 140,000 counts output activations for this sequence, not weights. Sharing the map across positions keeps the parameter count independent of sequence length.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q39",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "A token embedding is \\((1,2)\\) and its position vector is \\((3,-1)\\). What vector enters the encoder when position is added elementwise?",
    options: [
      {
        text: "\\((-2,3)\\)",
        isCorrect: false,
      },
      {
        text: "\\((3,-2)\\)",
        isCorrect: false,
      },
      {
        text: "\\((4,1)\\)",
        isCorrect: true,
      },
      {
        text: "\\((1,2,3,-1)\\)",
        isCorrect: false,
      },
    ],
    explanation:
      "Elementwise addition combines the corresponding coordinates, giving (1+3, 2-1)=(4,1). Concatenation would double the width, multiplication would give (3,-2), and subtraction would give (-2,3); the original additive construction preserves the embedding width.",
  },
  // Replaces retired ID cme295-2026-lect1-q40; new reasoning task.
  {
    id: "cme295-2026-lect1-q062",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "For a residual block \\(y=x+F(x)\\), let \\(J_F\\) be the local Jacobian of \\(F\\) and \\(I\\) the identity. Select all true statements about the block Jacobian.",
    options: [
      {
        text: "If \\(J_F=-0.9I\\), the block Jacobian is \\(0.1I\\).",
        isCorrect: true,
      },
      {
        text: "If \\(J_F=-I\\), the block Jacobian is zero despite the residual connection.",
        isCorrect: true,
      },
      {
        text: "If \\(J_F=0\\), a nonzero incoming gradient can pass through the identity path.",
        isCorrect: true,
      },
      {
        text: "It is \\(I+J_F\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "Differentiating both terms gives I + J_F. The identity term offers a direct gradient path when the residual branch has a small derivative, but the two derivatives still add and can cancel. Thus −0.9I leaves 0.1I and −I cancels the identity exactly; a residual connection helps gradient flow without making nonzero gradients a mathematical guarantee.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q41",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Layer normalization is applied to the hidden features of each token. Which statements describe this operation?",
    options: [
      {
        text: "It controls the scale of activations entering later computations.",
        isCorrect: true,
      },
      {
        text: "Its normalization statistics are computed across a token's hidden features.",
        isCorrect: true,
      },
      {
        text: "Its mean and variance are computed across token positions separately for each hidden feature.",
        isCorrect: false,
      },
      {
        text: "It can improve optimization stability.",
        isCorrect: true,
      },
    ],
    explanation:
      "Layer normalization computes its statistics over the hidden features of each token, then usually applies learned affine parameters. Controlling activation scale can stabilize optimization. Computing a separate statistic for each feature across token positions would normalize a different axis and would not be the stated per-token layer normalization.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q42",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A training decoder receives the shifted target prefix [BOS], y1, y2, y3. When predicting y3 at the position whose input is y2, which target inputs may causal self-attention use?",
    options: [
      {
        text: "y3 and later targets, excluding y1 and y2.",
        isCorrect: false,
      },
      {
        text: "[BOS], y1, and y2.",
        isCorrect: true,
      },
      {
        text: "[BOS], y1, y2, and y3.",
        isCorrect: false,
      },
      {
        text: "The source tokens but none of the target inputs.",
        isCorrect: false,
      },
    ],
    explanation:
      "The input at that position is y2, so the allowed target prefix ends at y2 while the label is y3. The mask blocks later inputs, including y3, to prevent label leakage; source information is available separately through cross-attention.",
  },
  // Replaces retired ID cme295-2026-lect1-q43; new reasoning task.
  {
    id: "cme295-2026-lect1-q067",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "A query at decoder position 2 has logits \\((0,\\ln2,\\ln4)\\) for keys at positions 1, 2, and 3. Use an inclusive causal mask and scalar values \\((1,3,100)\\). Select all true calculations.",
    options: [
      {
        text: "Masking before softmax gives weights \\((1/3,2/3,0)\\).",
        isCorrect: true,
      },
      {
        text: "The correctly masked weighted value is \\(7/3\\).",
        isCorrect: true,
      },
      {
        text: "Zeroing the future weight after unmasked softmax without renormalizing also gives weighted value \\(7/3\\).",
        isCorrect: false,
      },
      {
        text: "Applying unmasked softmax then zeroing the future weight without renormalizing leaves total mass \\(3/7\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "Exponentiating the three logits gives weights proportional to [1, 2, 4]. Removing the future key before normalization leaves [1/3, 2/3, 0], whose weighted value is 1/3 + 2 = 7/3. Unmasked softmax instead divides by seven; zeroing its last weight leaves mass 3/7 and weighted value 1, so post-softmax zeroing needs renormalization to implement the same mask.",
  },
  // Replaces retired ID cme295-2026-lect1-q44; new reasoning task.
  {
    id: "cme295-2026-lect1-q063",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Consider a position-wise feed-forward network (FFN) after attention. Change only token row 4 of its input and hold all other input rows fixed. Select all true statements.",
    options: [
      {
        text: "The FFN uses the same learned transformations at rows 2 and 4.",
        isCorrect: true,
      },
      {
        text: "The FFN output at row 2 stays unchanged under this intervention.",
        isCorrect: true,
      },
      {
        text: "The FFN output at row 4 may change.",
        isCorrect: true,
      },
      {
        text: "The FFN itself can transfer the changed row-4 values into row 2 through token-to-token weights.",
        isCorrect: false,
      },
    ],
    explanation:
      "A position-wise FFN transforms each input row independently with shared parameters. Changing one row can affect its own output but cannot alter another fixed input row's FFN output. An attention operation before or after the FFN could propagate information between positions, which is why fixing the FFN inputs is essential to this comparison.",
  },
  // Replaces retired ID cme295-2026-lect1-q45; new reasoning task.
  {
    id: "cme295-2026-lect1-q064",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "In inverted dropout with drop probability \\(p=0.25\\), a training activation is \\(y=Bx/(1-p)\\), where \\(B\\) is 1 with probability 0.75 and 0 otherwise. For \\(x=8\\), select all true statements.",
    options: [
      {
        text: "The nonzero training value is \\(32/3\\).",
        isCorrect: true,
      },
      {
        text: "At inference the activation is multiplied by \\(4/3\\) again.",
        isCorrect: false,
      },
      {
        text: "The training value is zero with probability \\(0.25\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\mathbb E[y]=8\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "A surviving activation is 8/0.75 = 32/3, and its expectation is 0.75 × 32/3 = 8. The missing probability mass is the 0.25 chance of a zero activation. Inverted dropout puts the compensation in training, so inference uses the original activation without another 4/3 multiplier; applying it again would inflate the expected scale.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q46",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which training change most directly implements label smoothing while preserving the model architecture?",
    options: [
      {
        text: "Normalize each token activation vector across its feature coordinates.",
        isCorrect: false,
      },
      {
        text: "Mix the one-hot target with a small probability distribution over vocabulary alternatives.",
        isCorrect: true,
      },
      {
        text: "Divide the output logits by a temperature before inference-time sampling.",
        isCorrect: false,
      },
      {
        text: "Randomly zero a small fraction of hidden activations during training.",
        isCorrect: false,
      },
    ],
    explanation:
      "Label smoothing changes the target distribution used in the prediction loss, reducing pressure toward absolute confidence in the observed class. Temperature scaling changes the prediction distribution, dropout perturbs activations, and layer normalization rescales hidden features; these are different interventions.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q47",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Use \\(q_k=(1-\\epsilon)\\mathbf{1}[k=y]+\\epsilon/V\\), with \\(\\epsilon=0.2\\) and \\(V=4\\). Which numerical consequences are correct?",
    options: [
      {
        text: "The observed token has target probability 0.85.",
        isCorrect: true,
      },
      {
        text: "Each other token has target probability 0.05.",
        isCorrect: true,
      },
      {
        text: "The probabilities sum to 1.",
        isCorrect: true,
      },
      {
        text: "The observed token keeps more mass than any alternative.",
        isCorrect: true,
      },
    ],
    explanation:
      "The observed token receives 0.8 plus its share of the uniform 0.2 mass, giving 0.85. Each of the other three receives 0.05, and 0.85 plus three times 0.05 equals one; this convention includes the observed token in the uniform component.",
  },
  // Replaces retired ID cme295-2026-lect1-q48; new reasoning task.
  {
    id: "cme295-2026-lect1-q065",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A two-head encoder-decoder Transformer processes six encoder tokens and a three-token decoder prefix. Each decoder self-attention head uses an inclusive causal mask; each cross-attention head can read all encoder tokens. Count query-key pairs with nonzero eligibility, ignoring padding. Select all true counts for this decoder layer.",
    options: [
      {
        text: "Both self-attention heads together have twelve eligible pairs.",
        isCorrect: true,
      },
      {
        text: "Self-attention and cross-attention together have forty-eight eligible pairs across both heads.",
        isCorrect: true,
      },
      {
        text: "Each self-attention head has six eligible pairs.",
        isCorrect: true,
      },
      {
        text: "Each cross-attention head has eighteen eligible pairs.",
        isCorrect: true,
      },
    ],
    explanation:
      "The self-attention rows allow 1, 2, and 3 keys, giving six eligible pairs per head and twelve across two heads. Cross-attention has three queries and six keys, giving eighteen per head and thirty-six across two. Adding twelve and thirty-six gives forty-eight; the causal restriction applies to decoder self-attention, not to encoder key access.",
  },
  // Replaces retired ID cme295-2026-lect1-q49; new reasoning task.
  {
    id: "cme295-2026-lect1-q066",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A deterministic decoder predicts target y3 using shifted teacher-forced inputs. Which intervention demonstrates access to a future target that would be unavailable when generating y3? In every comparison, hold all unmentioned inputs and parameters fixed.",
    options: [
      {
        text: "Changing an encoder source token while keeping y1 and y2 fixed changes the distribution for y3.",
        isCorrect: false,
      },
      {
        text: "Changing y1 while keeping the source and y2 fixed changes the distribution for y3.",
        isCorrect: false,
      },
      {
        text: "Changing the language-model head weights while keeping the inputs fixed changes the distribution for y3.",
        isCorrect: false,
      },
      {
        text: "Changing y4 while keeping the source, y1, and y2 fixed changes the distribution for y3.",
        isCorrect: true,
      },
    ],
    explanation:
      "The conditional distribution for y3 should depend on the source and earlier targets, not y4. Sensitivity to y4 under controlled inputs therefore exposes future-target leakage. Sensitivity to y1, source information, or changed model weights is compatible with legitimate decoding; these comparisons change information or parameters that the predictor is allowed to use.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect1-q50",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "An encoder-decoder translator starts from [BOS], emits target words, and eventually emits [EOS]. Which statements correctly trace generation?",
    options: [
      {
        text: "The output distribution at each step ranges over the target vocabulary.",
        isCorrect: true,
      },
      {
        text: "The emitted target token becomes part of the next decoder prefix.",
        isCorrect: true,
      },
      {
        text: "The encoded source can be reused across target steps.",
        isCorrect: true,
      },
      {
        text: "The number of generated target tokens must equal the source token count.",
        isCorrect: false,
      },
    ],
    explanation:
      "Generation extends the prefix one predicted token at a time while repeatedly conditioning on the encoded source. The vocabulary is the set of possible next tokens, and [EOS] supplies a stopping decision; translation can legitimately use a different number of tokens than the source.",
  },
];
