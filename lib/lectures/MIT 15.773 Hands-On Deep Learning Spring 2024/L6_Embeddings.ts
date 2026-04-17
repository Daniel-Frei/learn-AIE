import { Question } from "../../quiz";

export const EmbeddingsQuestions: Question[] = [
  {
    id: "mit15773-l6-q01",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statement best explains a key semantic weakness of one-hot word vectors?",
    options: [
      {
        text: "They make every pair of distinct words equally distant under Euclidean distance, so synonymy and opposition are not reflected geometrically.",
        isCorrect: true,
      },
      {
        text: "They automatically place related words close together because frequent words share many zero entries.",
        isCorrect: false,
      },
      {
        text: "They encode word meaning mainly through the magnitude of the nonzero entry.",
        isCorrect: false,
      },
      {
        text: "They solve polysemy because the same word can occupy several one-hot positions at once.",
        isCorrect: false,
      },
    ],
    explanation:
      "For distinct one-hot vectors, the nonzero entry is in a different position, so the Euclidean distance between any two distinct words is the same. That means one-hot encoding represents identity, not semantic similarity, so words like 'movie' and 'film' are no closer than 'movie' and 'banana'.",
  },
  {
    id: "mit15773-l6-q02",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Suppose \\(\\mathbf{e}_i\\) and \\(\\mathbf{e}_j\\) are two different one-hot vectors from the same vocabulary. Which statements are correct?",
    options: [
      {
        text: "Their Euclidean distance is \\(\\sqrt{2}\\).",
        isCorrect: true,
      },
      {
        text: "Their dot product is \\(0\\).",
        isCorrect: true,
      },
      {
        text: "Their Euclidean distance depends on how often the underlying words occur in the corpus.",
        isCorrect: false,
      },
      {
        text: "Their distance is smaller when the two words are synonyms.",
        isCorrect: false,
      },
    ],
    explanation:
      "If two one-hot vectors correspond to different vocabulary items, they differ in exactly two positions: one has \\(1\\) where the other has \\(0\\), and vice versa. That gives squared distance \\(1 + 1 = 2\\), so the Euclidean distance is \\(\\sqrt{2}\\), and because the ones are in different coordinates, their dot product is zero.",
  },
  {
    id: "mit15773-l6-q03",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which of the following are good reasons to prefer dense word embeddings over one-hot vectors?",
    options: [
      {
        text: "Embeddings can be much lower-dimensional than the vocabulary size.",

        isCorrect: true,
      },
      {
        text: "Embeddings can encode semantic similarity geometrically.",
        isCorrect: true,
      },
      {
        text: "Embeddings are learned from data rather than being hard-coded identity indicators.",
        isCorrect: true,
      },
      {
        text: "It is not the case that Embeddings guarantee that every dimension has an obvious human-interpretable meaning such as 'positivity' or 'animalness'.",
        isCorrect: true,
      },
    ],
    explanation:
      "Dense embeddings are compact and data-driven, so they can represent meaningful similarities between words without requiring one coordinate per vocabulary item. However, individual embedding dimensions are usually not cleanly interpretable; what matters is the geometry of the full vector, not a simple label for each coordinate.",
  },
  {
    id: "mit15773-l6-q04",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Consider the goals of standalone word embeddings as introduced in the lecture. Which statements are correct?",
    options: [
      {
        text: "Synonyms or closely related words should tend to lie near one another in embedding space.",
        isCorrect: true,
      },
      {
        text: "Words with very different meanings should tend to be farther apart.",
        isCorrect: true,
      },
      {
        text: "Geometric relationships can reflect semantic relationships, not just isolated pairwise distances.",
        isCorrect: true,
      },
      {
        text: "Embeddings can also reduce the computational burden compared with very large one-hot vectors.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture emphasized both efficiency and semantics: embeddings are shorter, denser, and more expressive than one-hot vectors. It also stressed that useful geometry includes not only closeness but also directional structure, which enables analogy-like relationships.",
  },
  {
    id: "mit15773-l6-q05",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Why are standalone word embeddings still insufficient for fully handling words like 'bank'?",
    options: [
      {
        text: "A single standalone embedding tends to average over multiple meanings unless surrounding context is used.",
        isCorrect: true,
      },
      {
        text: "Standalone embeddings fail only because they are always binary vectors.",
        isCorrect: false,
      },
      {
        text: "Standalone embeddings cannot be trained with gradient descent.",
        isCorrect: false,
      },
      {
        text: "Standalone embeddings only work for verbs, not nouns.",
        isCorrect: false,
      },
    ],
    explanation:
      "Words with multiple meanings need context to disambiguate which sense is intended in a particular sentence. A standalone embedding gives one vector per word type, so for polysemous words it often behaves like an average over several senses rather than a context-specific meaning.",
  },
  {
    id: "mit15773-l6-q06",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements correctly describe contextual embeddings?",
    options: [
      {
        text: "They depend on the surrounding words in the sentence.",
        isCorrect: true,
      },
      {
        text: "They are meant to address the fact that the same surface word can mean different things in different contexts.",
        isCorrect: true,
      },
      {
        text: "They make one-hot vectors semantically meaningful without changing the representation.",
        isCorrect: false,
      },
      {
        text: "They require that every sentence in the corpus have exactly the same length before the model can use them conceptually.",
        isCorrect: false,
      },
    ],
    explanation:
      "Contextual embeddings adapt the representation of a word to the sentence it appears in, which is crucial for polysemy. They are not just one-hot vectors with a new interpretation; they are learned representations whose value changes with context.",
  },
  {
    id: "mit15773-l6-q07",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "The phrase 'You shall know a word by the company it keeps' motivates which ideas?",
    options: [
      {
        text: "Words that occur in similar contexts are likely to be semantically related.",

        isCorrect: true,
      },
      {
        text: "A useful signal can be obtained by quantifying word co-occurrence patterns across many sentences.",
        isCorrect: true,
      },
      {
        text: "Relatedness can be inferred indirectly from distributional behavior rather than from manually listing all synonyms and antonyms.",
        isCorrect: true,
      },
      {
        text: "It is not the case that The best context definition is always the immediately adjacent word and never a larger span.",
        isCorrect: true,
      },
    ],
    explanation:
      "The central insight is distributional: meaning can be inferred from usage patterns. The lecture used sentence-level co-occurrence as a simple context definition, but that was presented as a modeling choice, not as the only possible notion of context.",
  },
  {
    id: "mit15773-l6-q08",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Consider a word-word co-occurrence matrix built from a large corpus using sentence-level context. Which statements are correct?",
    options: [
      {
        text: "Entry \\(X_{ij}\\) can represent how many times word \\(i\\) and word \\(j\\) appear in the same sentence across the corpus.",

        isCorrect: true,
      },
      {
        text: "The matrix is typically sparse because many word pairs rarely or never appear in the same sentence.",
        isCorrect: true,
      },
      {
        text: "If two words are highly interchangeable, their rows in the co-occurrence matrix may show similar patterns.",
        isCorrect: true,
      },
      {
        text: "It is not the case that The diagonal entries are the main source of semantic information, so off-diagonal structure is usually irrelevant.",
        isCorrect: true,
      },
    ],
    explanation:
      "A co-occurrence matrix summarizes distributional structure by recording which words tend to appear with which others. In practice, off-diagonal structure is usually the important part, so the last statement is actually too strong and misleading; wait, this means that option should not be correct. Let me correct that: the semantic structure mainly comes from off-diagonal co-occurrence patterns rather than from the diagonal alone.",
  },
  {
    id: "mit15773-l6-q09",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statement about the co-occurrence rows of two very similar words is most accurate?",
    options: [
      {
        text: "They should often exhibit similar patterns across many columns because they tend to appear in related contexts.",
        isCorrect: true,
      },
      {
        text: "They must be numerically identical in every corpus, otherwise the words cannot be related.",
        isCorrect: false,
      },
      {
        text: "They will usually be orthogonal if the two words are synonyms, because synonyms rarely appear in the same sentence.",
        isCorrect: false,
      },
      {
        text: "They are useful only when the two words are adjacent in text.",
        isCorrect: false,
      },
    ],
    explanation:
      "Distributional similarity does not require two row vectors to be identical, only broadly similar in pattern. The point is that related words often co-occur with many of the same kinds of neighboring words or sentence contexts, which creates similar row structures in the matrix.",
  },
  {
    id: "mit15773-l6-q10",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "In a simplified GloVe-style word embedding model, which statements are correct?",
    options: [
      {
        text: "A co-occurrence quantity is modeled using two word biases plus the dot product of two embedding vectors.",
        isCorrect: true,
      },
      {
        text: "Taking a logarithm of counts can make the regression problem better behaved by shrinking a wide dynamic range.",
        isCorrect: true,
      },
      {
        text: "The model assumes that every word pair must have a negative dot product if the words are unrelated.",
        isCorrect: false,
      },
      {
        text: "This setup required a deep neural network with many hidden layers in order to fit the embeddings.",
        isCorrect: false,
      },
    ],
    explanation:
      "In this simplified regression-style objective, the log co-occurrence is approximated by \\(b_i + b_j + \\mathbf{w}_i^T \\mathbf{w}_j\\). The setup is conceptually simple and can be optimized with gradient descent without needing a deep network architecture.",
  },
  {
    id: "mit15773-l6-q11",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Suppose the model uses \\(\\log(X_{ij}) \\approx b_i + b_j + \\mathbf{w}_i^T\\mathbf{w}_j\\). Which statements are correct?",
    options: [
      {
        text: "The biases \\(b_i\\) and \\(b_j\\) help account for the natural frequency of words independent of pair-specific interaction.",

        isCorrect: true,
      },
      {
        text: "The term \\(\\mathbf{w}_i^T\\mathbf{w}_j\\) captures how the two words relate beyond their baseline popularity.",
        isCorrect: true,
      },
      {
        text: "One can learn the parameters by minimizing squared error between observed and predicted log co-occurrence values.",
        isCorrect: true,
      },
      {
        text: "It is not the case that Once the model is trained, the biases are the main objects kept for semantic comparison and the embeddings are usually discarded.",
        isCorrect: true,
      },
    ],
    explanation:
      "The bias terms model baseline word frequency effects, while the dot product captures relational structure between the two words. After fitting, the embeddings are what we usually keep and analyze, because they encode the geometry of meaning; the biases are often discarded for downstream semantic use.",
  },
  {
    id: "mit15773-l6-q12",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about the objective\n\\[\n\\sum_{i,j} \\left[\\log(X_{ij}) - \\left(b_i + b_j + \\mathbf{w}_i^T\\mathbf{w}_j\\right)\\right]^2\n\\]\nare correct?",
    options: [
      {
        text: "It is a least-squares objective over observed word-pair statistics.",
        isCorrect: true,
      },
      {
        text: "It treats the embeddings and biases as parameters to be learned from data.",
        isCorrect: true,
      },
      {
        text: "It can in principle be optimized with gradient-based methods because the objective is differentiable in the parameters.",
        isCorrect: true,
      },
      {
        text: "Perfectly minimizing the objective would still not resolve contextual ambiguity such as the multiple meanings of 'bank' in different sentence occurrences.",
        isCorrect: true,
      },
    ],
    explanation:
      "This is a regression-style objective over word-pair statistics, the parameters are the biases and embeddings, and gradient descent can be used to optimize it. The final statement is wrong because the model still produces standalone word embeddings, so context-specific meaning is not automatically resolved for each occurrence.",
  },
  {
    id: "mit15773-l6-q13",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Why can embedding vectors often be much shorter than one-hot vectors?",
    options: [
      {
        text: "Because embeddings are dense learned representations, they do not need one separate coordinate per vocabulary item.",
        isCorrect: true,
      },
      {
        text: "Because embeddings remove the need for a vocabulary entirely.",
        isCorrect: false,
      },
      {
        text: "Because each embedding coordinate must correspond to a single specific word in the corpus.",
        isCorrect: false,
      },
      {
        text: "Because reducing dimensionality automatically preserves every semantic distinction with no tradeoff.",
        isCorrect: false,
      },
    ],
    explanation:
      "One-hot vectors are long because they dedicate one dimension to each vocabulary item. Embeddings instead compress information into a lower-dimensional dense space, which is powerful but still involves tradeoffs, including the risk of underfitting or overfitting depending on dimension choice.",
  },
  {
    id: "mit15773-l6-q14",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about embedding dimensionality are correct?",
    options: [
      {
        text: "The embedding dimension is a hyperparameter that the practitioner chooses.",
        isCorrect: true,
      },
      {
        text: "Increasing the embedding dimension can increase representational flexibility.",
        isCorrect: true,
      },
      {
        text: "A larger embedding dimension removes all risk of overfitting because embeddings are dense rather than sparse.",
        isCorrect: false,
      },
      {
        text: "Choosing an embedding dimension is unrelated to model capacity.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture explicitly treated embedding size as a tunable hyperparameter. Larger embeddings can capture more structure, but they also increase capacity, which can raise overfitting risk and make tuning necessary rather than automatic.",
  },
  {
    id: "mit15773-l6-q15",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about vector arithmetic in embedding spaces are correct?",
    options: [
      {
        text: "If an embedding space has captured meaningful relational structure, directions can encode analogies in addition to simple similarity.",
        isCorrect: true,
      },
      {
        text: "An example of this idea is that \\((\\text{brother} - \\text{man}) + \\text{woman}\\) may land near \\(\\text{sister}\\).",
        isCorrect: true,
      },
      {
        text: "Such examples prove that every coordinate of the embedding vector corresponds to one clearly named semantic axis.",
        isCorrect: false,
      },
      {
        text: "These analogy effects imply that contextual ambiguity no longer matters once standalone embeddings are learned.",
        isCorrect: false,
      },
    ],
    explanation:
      "Embedding algebra is evidence that geometry can reflect structured semantic relations, not just nearest-neighbor similarity. But this does not mean each coordinate has a neat human label, and it definitely does not eliminate the need for contextual modeling when a word has multiple senses.",
  },
  {
    id: "mit15773-l6-q16",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about pretrained embeddings such as GloVe are correct?",
    options: [
      {
        text: "They can be especially useful when the task-specific dataset is small.",
        isCorrect: true,
      },
      {
        text: "They capture fairly generic aspects of language that can be reused across many tasks.",
        isCorrect: true,
      },
      {
        text: "They may be mismatched to a specialized domain such as medicine or law unless adapted or fine-tuned.",
        isCorrect: true,
      },
      {
        text: "They can still be fine-tuned on downstream data even when the original embeddings were pretrained on a different corpus.",
        isCorrect: true,
      },
    ],
    explanation:
      "Pretrained embeddings are especially useful when labeled task data is limited, they transfer broad language structure, and they may still need adaptation for specialized domains. Large corpora can still contain social, cultural, and domain biases, so pretrained embeddings can reproduce them rather than eliminating them.",
  },
  {
    id: "mit15773-l6-q17",
    chapter: 1,
    difficulty: "medium",
    prompt:
      'In a Keras workflow for embeddings, why might a `TextVectorization` layer be set with `output_mode="int"`?',
    options: [
      {
        text: "Because we want the layer to stop after assigning token indices, so a later `Embedding` layer can map those integers to vectors.",
        isCorrect: true,
      },
      {
        text: "Because Keras embedding layers can only consume one-hot vectors, not integer token IDs.",
        isCorrect: false,
      },
      {
        text: 'Because `output_mode="int"` automatically averages the word embeddings over the full sentence.',
        isCorrect: false,
      },
      {
        text: "Because integer output removes the need for truncation or padding of variable-length sequences.",
        isCorrect: false,
      },
    ],
    explanation:
      "The idea is to separate indexing from embedding lookup: `TextVectorization` produces token IDs, and `Embedding` converts those IDs into dense vectors. This does not remove the need to manage sequence length, and it does not perform pooling by itself.",
  },
  {
    id: "mit15773-l6-q18",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose a sentence is converted into a sequence of token IDs and then passed through an embedding layer. Which statements are correct?",
    options: [
      {
        text: "The embedding layer behaves like a lookup table from integer indices to dense vectors.",
        isCorrect: true,
      },
      {
        text: "After lookup, a sentence of length \\(L\\) with embedding dimension \\(d\\) can be represented as an \\(L \\times d\\) tensor.",
        isCorrect: true,
      },
      {
        text: "Padding tokens must always receive random nonzero vectors so that the model can distinguish them from real words.",
        isCorrect: false,
      },
      {
        text: "Because the embedding layer already outputs vectors, there is no need for any later step to convert a sentence representation into a form suitable for downstream dense layers.",
        isCorrect: false,
      },
    ],
    explanation:
      "An embedding layer is fundamentally an index-to-vector table. After lookup, the model has a sequence of vectors, which often still needs a later aggregation or sequence-processing step before standard dense layers can consume it meaningfully.",
  },
  {
    id: "mit15773-l6-q19",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about padding, truncation, and pooling in the lecture's Keras pipeline are correct?",
    options: [
      {
        text: "Padding and truncation are used so that variable-length sentences can be represented with a common maximum length.",

        isCorrect: true,
      },
      {
        text: "Global average pooling produces one vector by averaging across the sequence dimension.",
        isCorrect: true,
      },
      {
        text: "Averaging can lose some information about word order and fine-grained interactions.",
        isCorrect: true,
      },
      {
        text: "It is not the case that Global average pooling preserves the exact full sequence structure while reducing dimensionality.",
        isCorrect: true,
      },
    ],
    explanation:
      "Padding and truncation make batching practical because models typically expect uniform tensor shapes. Global average pooling is simple and effective in some cases, but it compresses the sequence and therefore discards order-specific details and some contextual nuance.",
  },
  {
    id: "mit15773-l6-q20",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "For a downstream NLP task, compare three approaches: using frozen pretrained embeddings, fine-tuning pretrained embeddings, and learning embeddings from scratch. Which statements are correct?",
    options: [
      {
        text: "Frozen pretrained embeddings can be a strong starting point when data is limited.",

        isCorrect: true,
      },
      {
        text: "Fine-tuning pretrained embeddings can improve task performance when the task data provides useful additional signal.",
        isCorrect: true,
      },
      {
        text: "Learning embeddings from scratch can outperform pretrained embeddings when enough task-specific data is available.",
        isCorrect: true,
      },
      {
        text: "You still need to evaluate these choices empirically, because no single approach is guaranteed to dominate across all datasets and domains.",
        isCorrect: true,
      },
    ],
    explanation:
      "Frozen pretrained embeddings can be a strong starting point when data is limited, fine-tuning can help when the task data adds useful signal, and learning from scratch can win when enough task-specific data is available. The best choice depends on dataset size, domain mismatch, model design, and regularization, so it has to be checked empirically.",
  },
  {
    id: "mit15773-l6-q21",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the historical and current handling of text standardization in the lecture?",
    options: [
      {
        text: "Historically, standardization often included steps such as lowercasing, punctuation stripping, stop-word removal, and stemming.",
        isCorrect: true,
      },
      {
        text: "Modern practice may use lighter preprocessing and sometimes skip stop-word removal and stemming.",
        isCorrect: true,
      },
      {
        text: "The lecture noted that Keras commonly defaults to lowercasing and punctuation stripping.",
        isCorrect: true,
      },
      {
        text: "Standardization guarantees that words with similar meanings receive similar one-hot vectors.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture contrasted older NLP preprocessing pipelines with more modern approaches that often do less aggressive normalization. These preprocessing choices affect tokenization and vocabulary construction, but they do not by themselves solve the semantic problems of one-hot encoding.",
  },
  {
    id: "mit15773-l6-q22",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Suppose a vocabulary has size \\(V = 500{,}000\\). Which of the following are valid concerns about representing every token as a one-hot vector of length \\(V\\)?",
    options: [
      {
        text: "Each token representation is extremely high-dimensional even though only one entry is nonzero.",
        isCorrect: true,
      },
      {
        text: "Using such representations can increase the number of model weights needed in later layers.",
        isCorrect: true,
      },
      {
        text: "The representation can be computationally wasteful compared with a compact dense embedding.",
        isCorrect: true,
      },
      {
        text: "Because the vector is sparse, it automatically gives a faithful notion of semantic distance.",
        isCorrect: false,
      },
    ],
    explanation:
      "A one-hot vector with 500,000 dimensions is sparse but still very large, and downstream models must often process those dimensions. Sparsity alone does not create meaningful semantics; it mainly identifies the word's position in the vocabulary.",
  },
  {
    id: "mit15773-l6-q23",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statement best captures why sentence-level co-occurrence can be useful for learning standalone embeddings?",
    options: [
      {
        text: "If two words tend to occur in similar sentence contexts, that distributional pattern can reveal that they are related even if they do not directly appear next to each other.",
        isCorrect: true,
      },
      {
        text: "Sentence-level co-occurrence works only when two words are exact synonyms and fails for broader relatedness.",
        isCorrect: false,
      },
      {
        text: "Co-occurrence counts directly encode the correct contextual meaning of every word occurrence, so no later contextual models are needed.",
        isCorrect: false,
      },
      {
        text: "The only useful entries in a co-occurrence matrix are for neighboring words that appear side by side.",
        isCorrect: false,
      },
    ],
    explanation:
      "The distributional idea is broader than adjacency: words can be related because they appear in similar environments across many sentences. This helps recover semantic structure, but it still produces standalone embeddings rather than fully context-specific representations.",
  },
  {
    id: "mit15773-l6-q24",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which of the following statements about evaluating learned standalone embeddings are correct?",
    options: [
      {
        text: "One useful test is whether the embeddings can approximately reconstruct the observed co-occurrence matrix they were meant to model.",
        isCorrect: true,
      },
      {
        text: "Looking only at one hand-picked synonym pair is not a systematic evaluation of embedding quality.",
        isCorrect: true,
      },
      {
        text: "If embeddings reproduce many co-occurrence patterns well, that is evidence that they have captured meaningful distributional structure.",
        isCorrect: true,
      },
      {
        text: "If embeddings fail to produce perfect reconstruction, they are automatically useless for any downstream task.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture framed reconstruction of the co-occurrence matrix as a principled way to assess whether embeddings have learned useful structure. Approximate reconstruction can still be very informative, because these models are intended to capture the main signal rather than every noisy detail exactly.",
  },
  {
    id: "mit15773-l6-q25",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Consider the role of correlation patterns across co-occurrence rows. Which statements are correct?",
    options: [
      {
        text: "If two words are strongly related, their co-occurrence rows may vary in similar ways across many other words.",
        isCorrect: true,
      },
      {
        text: "If two words are unrelated, their row patterns may have little systematic resemblance.",
        isCorrect: true,
      },
      {
        text: "The goal of the embeddings is partly to compress and reproduce these broader row-pattern relationships.",
        isCorrect: true,
      },
      {
        text: "This logic completely resolves polysemy for words with several meanings in different contexts.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture used row-pattern similarity as intuition for why co-occurrence structure contains semantic information. But because standalone embeddings still assign one vector per word type, they do not fully separate different senses of ambiguous words.",
  },
  {
    id: "mit15773-l6-q26",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Why might one apply a logarithm to co-occurrence counts before fitting a regression-style embedding model?",
    options: [
      {
        text: "Because co-occurrence counts can vary over a very wide positive range, and the logarithm compresses that range.",
        isCorrect: true,
      },
      {
        text: "Because the logarithm can make the optimization problem numerically better behaved than using raw counts directly.",
        isCorrect: true,
      },
      {
        text: "Because taking a logarithm turns the problem into exact contextual disambiguation.",
        isCorrect: false,
      },
      {
        text: "Because after taking logs, the bias terms become unnecessary in all cases.",
        isCorrect: false,
      },
    ],
    explanation:
      "Log transformations are a common way to handle heavily skewed positive quantities in regression-like models. They improve scale behavior, but they do not eliminate the need for bias terms or solve the deeper contextual limitations of standalone embeddings.",
  },
  {
    id: "mit15773-l6-q27",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose \\(\\mathbf{w}_i\\in\\mathbb{R}^d\\) is the embedding for word \\(i\\). Which statements are correct?",
    options: [
      {
        text: "The choice of \\(d\\) affects model capacity and is therefore a hyperparameter.",
        isCorrect: true,
      },
      {
        text: "Using a larger \\(d\\) can allow more nuanced geometric structure to be represented.",
        isCorrect: true,
      },
      {
        text: "Using a very large \\(d\\) can increase the risk of fitting noise rather than just useful structure.",
        isCorrect: true,
      },
      {
        text: "The lecture argued that there is one universally optimal embedding dimension that should always be used.",
        isCorrect: false,
      },
    ],
    explanation:
      "Embedding dimension controls how expressive the representation can be, so it behaves like other capacity-related hyperparameters in machine learning. More dimensions can help, but they can also introduce overfitting or unnecessary complexity, so the choice is empirical rather than universal.",
  },
  {
    id: "mit15773-l6-q28",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements about bias in learned embeddings are correct?",
    options: [
      {
        text: "If the training corpus contains social or cultural biases, the learned embeddings can reflect those biases.",
        isCorrect: true,
      },
      {
        text: "Using large internet-scale text does not automatically remove bias from the resulting embeddings.",
        isCorrect: true,
      },
      {
        text: "Bias can matter because downstream systems may inherit patterns present in the embeddings.",
        isCorrect: true,
      },
      {
        text: "Bias is only a problem for contextual embeddings and not for standalone embeddings such as GloVe.",
        isCorrect: false,
      },
    ],
    explanation:
      "Embeddings are learned from data, so they reflect patterns in that data, including undesirable ones. This is true for both standalone and contextual representations, which is why bias in training corpora is an important practical concern.",
  },
  {
    id: "mit15773-l6-q29",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly compare pretrained embeddings with task-specific embeddings learned from scratch?",
    options: [
      {
        text: "Pretrained embeddings can provide useful generic language structure before the task model has seen much task-specific data.",
        isCorrect: true,
      },
      {
        text: "Task-specific embeddings learned from scratch can adapt more directly to the downstream objective when enough data is available.",
        isCorrect: true,
      },
      {
        text: "A pretrained embedding may miss some domain-specific jargon if it was trained on a more general corpus.",
        isCorrect: true,
      },
      {
        text: "A pretrained embedding is always strictly better than learning from scratch, regardless of task and dataset size.",
        isCorrect: false,
      },
    ],
    explanation:
      "Pretraining offers reusable general language knowledge, while learning from scratch offers specialization to the downstream task. Which strategy works best depends on data availability, domain mismatch, and whether fine-tuning is allowed.",
  },
  {
    id: "mit15773-l6-q30",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "In a Keras text pipeline for variable-length song lyrics, what is the purpose of choosing a maximum sequence length such as 300?",
    options: [
      {
        text: "It allows variable-length input sentences to be represented in a common tensor shape.",
        isCorrect: true,
      },
      {
        text: "Shorter sentences can be padded so that they match the chosen length.",
        isCorrect: true,
      },
      {
        text: "Longer sentences can be truncated so that batching remains practical.",
        isCorrect: true,
      },
      {
        text: "It guarantees that no information relevant to the task is ever discarded.",
        isCorrect: false,
      },
    ],
    explanation:
      "A fixed sequence length is mainly an engineering choice that makes batching and tensor construction straightforward. Padding and truncation are useful, but truncation can discard information, so the choice of maximum length is a practical tradeoff rather than a guarantee.",
  },
  {
    id: "mit15773-l6-q31",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about special token indices in the lecture's Keras setup are correct?",
    options: [
      {
        text: "A padding token can be reserved so that shorter sequences are extended to the desired length.",
        isCorrect: true,
      },
      {
        text: "An unknown token can be used for words outside the chosen vocabulary.",
        isCorrect: true,
      },
      {
        text: "These special indices help the model deal with finite vocabularies and variable-length input.",
        isCorrect: true,
      },
      {
        text: "Using an unknown token means the model has perfectly learned the semantic meaning of every out-of-vocabulary word.",
        isCorrect: false,
      },
    ],
    explanation:
      "Padding and unknown-token handling are practical mechanisms for building robust pipelines with limited vocabularies and variable-length inputs. However, an unknown token is only a fallback representation; it does not restore the exact meaning of a missing word.",
  },
  {
    id: "mit15773-l6-q32",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "What does an embedding layer do once a sentence has already been converted into integer token IDs?",
    options: [
      {
        text: "It maps each token ID to a dense vector by table lookup.",
        isCorrect: true,
      },
      {
        text: "It can output a matrix whose rows correspond to the sequence positions and whose columns correspond to embedding dimensions.",
        isCorrect: true,
      },
      {
        text: "It directly computes the final softmax class probabilities for the task.",
        isCorrect: false,
      },
      {
        text: "It removes the need for any later hidden or pooling layers in every model design.",
        isCorrect: false,
      },
    ],
    explanation:
      "The embedding layer converts symbolic token identities into learned dense vectors, typically producing one vector per sequence position. This is an intermediate representation, not the final task prediction, so later layers still matter.",
  },
  {
    id: "mit15773-l6-q33",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose a sentence of effective length \\(L\\) is mapped to embeddings in \\(\\mathbb{R}^d\\). Which statements about using `GlobalAveragePooling1D` are correct?",
    options: [
      {
        text: "The output becomes a vector in \\(\\mathbb{R}^d\\).",
        isCorrect: true,
      },
      {
        text: "The operation averages across token positions rather than across embedding dimensions.",
        isCorrect: true,
      },
      {
        text: "The representation can work surprisingly well even though it discards sequence order information.",
        isCorrect: true,
      },
      {
        text: "The resulting vector still preserves the exact position of every token in the original sentence.",
        isCorrect: false,
      },
    ],
    explanation:
      "Global average pooling aggregates the sequence into a single vector of the same embedding dimension, which is simple and efficient. But because it averages across positions, detailed order information and exact token placement are lost.",
  },
  {
    id: "mit15773-l6-q34",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Why might fine-tuning pretrained embeddings improve downstream task performance?",
    options: [
      {
        text: "Because backpropagation can adjust the pretrained vectors toward the specific task objective.",
        isCorrect: true,
      },
      {
        text: "Because task data may contain domain or stylistic information not captured by the original pretraining corpus.",
        isCorrect: true,
      },
      {
        text: "Because freezing embeddings is guaranteed to underfit, regardless of data size or model design.",
        isCorrect: false,
      },
      {
        text: "Because any change to pretrained embeddings automatically preserves all generic language knowledge while adding only useful task-specific structure.",
        isCorrect: false,
      },
    ],
    explanation:
      "Fine-tuning can help bridge the gap between general language knowledge and the needs of a specific task or domain. But it is not guaranteed to help, and too much adaptation can sometimes overfit or distort useful general structure.",
  },
  {
    id: "mit15773-l6-q35",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about learning embeddings from scratch for a downstream task are correct?",
    options: [
      {
        text: "The task loss itself can provide the signal for learning useful embeddings, without explicitly building a separate co-occurrence matrix.",
        isCorrect: true,
      },
      {
        text: "If enough labeled data is available, task-specific embeddings learned from scratch can outperform generic pretrained ones.",
        isCorrect: true,
      },
      {
        text: "Learning from scratch is impossible unless the corpus is as large as Wikipedia.",
        isCorrect: false,
      },
      {
        text: "Embeddings learned from scratch must have the same dimension as the pretrained GloVe vectors used in the lecture.",
        isCorrect: false,
      },
    ],
    explanation:
      "For supervised downstream tasks, embeddings can be learned directly through end-to-end training against the task objective. Their dimensionality is a design choice, and the required data size depends on the task and architecture, not on matching a specific pretraining corpus.",
  },
  {
    id: "mit15773-l6-q36",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly compare one-hot vectors and learned embeddings?",
    options: [
      {
        text: "One-hot vectors are sparse and largely encode identity, while learned embeddings are dense and attempt to encode useful structure.",
        isCorrect: true,
      },
      {
        text: "One-hot vectors have dimensionality tied directly to vocabulary size, whereas embedding dimensionality is chosen by the model designer.",
        isCorrect: true,
      },
      {
        text: "Both one-hot vectors and learned embeddings automatically resolve word meaning from sentence context in the same way.",
        isCorrect: false,
      },
      {
        text: "Learned embeddings can support meaningful nearest-neighbor and analogy-like relations that one-hot vectors do not.",
        isCorrect: true,
      },
    ],
    explanation:
      "This comparison captures the lecture's main conceptual shift from symbolic sparse identity vectors to compact dense geometric representations. However, neither one-hot vectors nor standalone embeddings fully resolve context-dependent meaning on their own.",
  },
  {
    id: "mit15773-l6-q37",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Consider the statement that geometric relationships, not just distances, can matter in embedding spaces. Which claims are correct?",
    options: [
      {
        text: "A useful embedding space may encode consistent directions corresponding to semantic transformations.",
        isCorrect: true,
      },
      {
        text: "This is why vector analogies can sometimes work even when the individual coordinates are not directly interpretable.",
        isCorrect: true,
      },
      {
        text: "Distance alone is the only relevant property of a good embedding geometry.",
        isCorrect: false,
      },
      {
        text: "Directional structure can be evidence that the embedding has captured something systematic rather than random coincidence.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture emphasized that meaning in embeddings is often visible in both closeness and direction. Analogy-like vector arithmetic suggests the space has learned systematic semantic structure, even if no single coordinate has an obvious standalone interpretation.",
  },
  {
    id: "mit15773-l6-q38",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about using stochastic gradient descent or related gradient-based methods in this lecture are correct?",
    options: [
      {
        text: "Gradient-based optimization is not limited to deep neural networks; it can also optimize differentiable objectives for embedding estimation.",
        isCorrect: true,
      },
      {
        text: "Random initialization can be used as a starting point for learning embedding parameters.",
        isCorrect: true,
      },
      {
        text: "The success of gradient descent depends on having a differentiable objective with gradients that can be computed.",
        isCorrect: true,
      },
      {
        text: "Because the model is not a transformer, gradient descent cannot be used to learn its parameters.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture explicitly noted that gradient descent is a general optimization tool, not something exclusive to neural networks. As long as the objective is differentiable and gradients can be computed, gradient-based methods can be used to fit the parameters.",
  },
  {
    id: "mit15773-l6-q39",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statement best explains why contextual embeddings are introduced after standalone embeddings?",
    options: [
      {
        text: "Because standalone embeddings solve the geometry and compactness problem, but not the need for word meaning to depend on surrounding context.",
        isCorrect: true,
      },
      {
        text: "Because standalone embeddings are useless and cannot represent any semantic similarity at all.",
        isCorrect: false,
      },
      {
        text: "Because contextual embeddings simply reuse one-hot vectors and do not require any different modeling ideas.",
        isCorrect: false,
      },
      {
        text: "Because contextual embeddings are only needed when the vocabulary is very small.",
        isCorrect: false,
      },
    ],
    explanation:
      "Standalone embeddings are an important intermediate step because they already improve on one-hot vectors in several ways. But they still assign one vector per word type, so context-dependent meaning requires a more advanced representation.",
  },
  {
    id: "mit15773-l6-q40",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which of the following are reasonable high-level takeaways from the lecture?",
    options: [
      {
        text: "One-hot vectors are simple and sometimes useful, but they fail to represent semantic similarity in their geometry.",
        isCorrect: true,
      },
      {
        text: "Standalone embeddings can be learned from large-scale distributional statistics such as co-occurrence patterns.",
        isCorrect: true,
      },
      {
        text: "Pretrained embeddings, fine-tuned embeddings, and embeddings learned from scratch each have settings where they may be appropriate.",
        isCorrect: true,
      },
      {
        text: "Once standalone embeddings are learned, contextual modeling becomes unnecessary for natural language processing.",
        isCorrect: false,
      },
    ],
    explanation:
      "These capture the main progression of the lecture: from the limitations of one-hot representations to the strengths and limitations of standalone embeddings, and then toward the need for contextual models. The lecture clearly set up transformers as the next step precisely because standalone embeddings do not fully solve context dependence.",
  },
];
