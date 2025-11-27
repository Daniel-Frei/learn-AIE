// lib\chapter1.ts
import { Question } from "./quiz";

// ----- Chapter 1 questions (50 total) -----

export const chapter1Questions: Question[] = [
  {
    id: "ch1-q01",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements about one-hot encoding are TRUE?",
    options: [
      {
        text: "Each word is represented by a sparse vector with a single 1 and the rest 0s.",
        isCorrect: true,
      },
      {
        text: "The length of each one-hot vector equals the size of the vocabulary.",
        isCorrect: true,
      },
      {
        text: "One-hot vectors naturally encode semantic similarity between words.",
        isCorrect: false,
      },
      {
        text: "One-hot encoding is memory-efficient for very large vocabularies.",
        isCorrect: false,
      },
    ],
    explanation:
      "One-hot vectors are sparse, vocab-sized vectors with a single 1. They do not encode semantics and become memory-heavy with large vocabularies.",
  },
  {
    id: "ch1-q02",
    chapter: 1,
    difficulty: "easy",
    prompt: "Text normalization before tokenization often includes which steps?",
    options: [
      {
        text: "Converting all characters to lowercase.",
        isCorrect: true,
      },
      {
        text: "Lemmatization or stemming to group related word forms.",
        isCorrect: true,
      },
      {
        text: "Randomly shuffling the words in every sentence.",
        isCorrect: false,
      },
      {
        text: "Removing all punctuation because it never carries information.",
        isCorrect: false,
      },
    ],
    explanation:
      "Lowercasing and possibly lemmatization/stemming are common. Punctuation can be important (sentence boundaries, questions, exclamations) and we never randomly shuffle words as preprocessing.",
  },
  {
    id: "ch1-q03",
    chapter: 1,
    difficulty: "medium",
    prompt: "Bag-of-words (BoW) representation has which characteristics?",
    options: [
      {
        text: "Each document becomes a vector of word counts over a shared vocabulary.",
        isCorrect: true,
      },
      {
        text: "Word order within the document is ignored.",
        isCorrect: true,
      },
      {
        text: "BoW vectors tend to be high-dimensional and sparse for large vocabularies.",
        isCorrect: true,
      },
      {
        text: "BoW explicitly encodes the position of each word in the sentence.",
        isCorrect: false,
      },
    ],
    explanation:
      "BoW counts word frequency per document, ignores order, and yields sparse high-dimensional vectors sized by the vocabulary.",
  },
  {
    id: "ch1-q04",
    chapter: 1,
    difficulty: "medium",
    prompt: "What problems are commonly associated with BoW and one-hot representations?",
    options: [
      {
        text: "High memory and computation cost due to large vector dimensionality.",
        isCorrect: true,
      },
      {
        text: "Lack of semantic similarity between words in the vector space.",
        isCorrect: true,
      },
      {
        text: "They inherently solve the curse of dimensionality.",
        isCorrect: false,
      },
      {
        text: "They are ideal for capturing long-range dependencies in sequences.",
        isCorrect: false,
      },
    ],
    explanation:
      "Both one-hot and BoW are sparse, high-dimensional, and do not encode semantics. They actually suffer from the curse of dimensionality rather than solving it.",
  },
  {
    id: "ch1-q05",
    chapter: 1,
    difficulty: "easy",
    prompt: "Term Frequency (TF) in TF-IDF captures which aspect?",
    options: [
      {
        text: "How often a word appears within a specific document, usually normalized by document length.",
        isCorrect: true,
      },
      {
        text: "How many documents in the corpus contain the word.",
        isCorrect: false,
      },
      {
        text: "The inverse of the word’s length in characters.",
        isCorrect: false,
      },
      {
        text: "The absolute position of the word within a sentence.",
        isCorrect: false,
      },
    ],
    explanation:
      "TF measures within-document frequency (often normalized by document length). Document frequency is handled by IDF, not TF.",
  },
  {
    id: "ch1-q06",
    chapter: 1,
    difficulty: "easy",
    prompt: "Inverse Document Frequency (IDF) is designed to do what?",
    options: [
      {
        text: "Down-weight very common words that appear in many documents.",
        isCorrect: true,
      },
      {
        text: "Up-weight words that appear only in a few documents.",
        isCorrect: true,
      },
      {
        text: "Give the same weight to every word regardless of frequency.",
        isCorrect: false,
      },
      {
        text: "Focus only on word order in a document.",
        isCorrect: false,
      },
    ],
    explanation:
      "IDF highlights rare, discriminative words and reduces the influence of very frequent, generic words.",
  },
  {
    id: "ch1-q07",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about TF-IDF are TRUE?",
    options: [
      {
        text: "TF-IDF multiplies TF by an (often log-scaled) IDF factor.",
        isCorrect: true,
      },
      {
        text: "Words that appear in all documents tend to have low TF-IDF scores.",
        isCorrect: true,
      },
      {
        text: "TF-IDF vectors are usually dense low-dimensional embeddings.",
        isCorrect: false,
      },
      {
        text: "TF-IDF originated in information retrieval for document search.",
        isCorrect: true,
      },
    ],
    explanation:
      "TF-IDF combines local term frequency with global rarity and was widely used in information retrieval. The resulting vectors are still high-dimensional and often sparse.",
  },
  {
    id: "ch1-q08",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "The “distributional hypothesis” motivating embeddings states which ideas?",
    options: [
      {
        text: "Words that occur in similar contexts tend to have similar meanings.",
        isCorrect: true,
      },
      {
        text: "A single word can have multiple meanings depending on context.",
        isCorrect: true,
      },
      {
        text: "Semantic meaning can be fully captured by raw word frequency alone.",
        isCorrect: false,
      },
      {
        text: "Embeddings should ignore context to avoid ambiguity.",
        isCorrect: false,
      },
    ],
    explanation:
      "Distributional semantics assumes context reflects meaning and acknowledges that the same surface form can carry different senses.",
  },
  {
    id: "ch1-q09",
    chapter: 1,
    difficulty: "easy",
    prompt: "What properties do we usually want word embeddings to have?",
    options: [
      {
        text: "Dense vectors of real numbers.",
        isCorrect: true,
      },
      {
        text: "Moderate, fixed dimensionality independent of vocabulary size.",
        isCorrect: true,
      },
      {
        text: "Ability to capture semantic similarity between words.",
        isCorrect: true,
      },
      {
        text: "Very sparse binary vectors whose length grows with the vocabulary.",
        isCorrect: false,
      },
    ],
    explanation:
      "Embeddings should be dense, low- to medium-dimensional, and encode semantic relationships, unlike sparse one-hot or BoW vectors.",
  },
  {
    id: "ch1-q10",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements describe the basic intuition of word2vec?",
    options: [
      {
        text: "Train a model to predict a target word from its context (or vice versa).",
        isCorrect: true,
      },
      {
        text: "Use a large corpus where contexts provide implicit supervision.",
        isCorrect: true,
      },
      {
        text: "Use the learned weight vectors of the neural network as word embeddings.",
        isCorrect: true,
      },
      {
        text: "Explicitly label each training example with human-annotated semantic tags.",
        isCorrect: false,
      },
    ],
    explanation:
      "Word2vec is self-supervised: it predicts words from context and treats the resulting weights as embeddings, without manual labels.",
  },
  {
    id: "ch1-q11",
    chapter: 1,
    difficulty: "medium",
    prompt: "In negative sampling word2vec, what is the classification task?",
    options: [
      {
        text: "Decide whether a candidate context word belongs to the true context window of a center word.",
        isCorrect: true,
      },
      {
        text: "Assign each word to exactly one semantic cluster.",
        isCorrect: false,
      },
      {
        text: "Binary classification: + for real context words, − for randomly sampled words.",
        isCorrect: true,
      },
      {
        text: "Multi-class classification over the entire vocabulary in every step.",
        isCorrect: false,
      },
    ],
    explanation:
      "Negative sampling turns training into many small binary classification problems: is (w, c) a real pair or a random negative?",
  },
  {
    id: "ch1-q12",
    chapter: 1,
    difficulty: "medium",
    prompt: "Why is the sigmoid function σ(x) used in word2vec’s scoring?",
    options: [
      {
        text: "It maps the dot product between embeddings into a probability between 0 and 1.",
        isCorrect: true,
      },
      {
        text: "It is suitable for binary logistic regression.",
        isCorrect: true,
      },
      {
        text: "It forces all dot products to be exactly 0 or 1.",
        isCorrect: false,
      },
      {
        text: "It guarantees that embeddings stay unit-length during training.",
        isCorrect: false,
      },
    ],
    explanation:
      "The sigmoid converts similarity scores into probabilities for binary logistic regression; it doesn’t normalize the embeddings themselves.",
  },
  {
    id: "ch1-q13",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about the dot product between two embedding vectors are TRUE?",
    options: [
      {
        text: "A large positive dot product suggests the vectors point in a similar direction.",
        isCorrect: true,
      },
      {
        text: "The dot product tends to be larger for vectors with larger magnitudes.",
        isCorrect: true,
      },
      {
        text: "The dot product is naturally bounded between −1 and 1.",
        isCorrect: false,
      },
      {
        text: "In word2vec, the dot product between w and c feeds into a sigmoid to get P(+|w,c).",
        isCorrect: true,
      },
    ],
    explanation:
      "Dot products grow with both alignment and magnitude and are unbounded. Word2vec uses σ(c·w) as a probability.",
  },
  {
    id: "ch1-q14",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "In negative-sampling word2vec, what does the cross-entropy loss L₍CE₎ do?",
    options: [
      {
        text: "Penalizes low probability for true context words around the center word.",
        isCorrect: true,
      },
      {
        text: "Penalizes high probability for sampled negative context words.",
        isCorrect: true,
      },
      {
        text: "Directly minimizes the Euclidean distance between embeddings.",
        isCorrect: false,
      },
      {
        text: "Is optimized so that real (+) and fake (−) examples are correctly classified.",
        isCorrect: true,
      },
    ],
    explanation:
      "Cross-entropy is the standard loss for logistic regression; it pushes P(+|w,c_pos) up and P(+|w,c_neg) down.",
  },
  {
    id: "ch1-q15",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which factors typically influence the quality of word2vec embeddings?",
    options: [
      {
        text: "Quality of the training corpus (e.g., Wikipedia vs noisy web data).",
        isCorrect: true,
      },
      {
        text: "Amount of text available for training.",
        isCorrect: true,
      },
      {
        text: "Embedding dimensionality (e.g., around 300 as a common sweet spot).",
        isCorrect: true,
      },
      {
        text: "Using only very short context windows of size 1 regardless of task.",
        isCorrect: false,
      },
    ],
    explanation:
      "Data quality, corpus size, and embedding dimensionality all matter. Context window size is also tuned rather than fixed at 1.",
  },
  {
    id: "ch1-q16",
    chapter: 1,
    difficulty: "medium",
    prompt: "How does the size of the context window affect word2vec embeddings?",
    options: [
      {
        text: "Smaller windows (e.g., 2) help capture syntactic information such as part of speech.",
        isCorrect: true,
      },
      {
        text: "Larger windows capture broader semantic similarity.",
        isCorrect: true,
      },
      {
        text: "Changing the window size has no impact on the learned embeddings.",
        isCorrect: false,
      },
      {
        text: "A window of length 4–5 is often used as a reasonable default.",
        isCorrect: true,
      },
    ],
    explanation:
      "Window size trades off local syntactic relationships vs more global semantic similarity; common choices include values around 4.",
  },
  {
    id: "ch1-q17",
    chapter: 1,
    difficulty: "easy",
    prompt: "Cosine similarity between two vectors has which properties?",
    options: [
      {
        text: "It is the dot product divided by the product of the vector magnitudes.",
        isCorrect: true,
      },
      {
        text: "It is always between −1 and 1.",
        isCorrect:true,
      },
      {
        text: "It is invariant to rescaling the vectors by positive constants.",
        isCorrect: true,
      },
      {
        text: "It always increases when you multiply both vectors by 10.",
        isCorrect: false,
      },
    ],
    explanation:
      "Cosine similarity normalizes the dot product, giving a scale-invariant measure in [−1,1] that depends only on direction.",
  },
  {
    id: "ch1-q18",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Why is cosine similarity preferred over the raw dot product for measuring similarity between embeddings?",
    options: [
      {
        text: "It is less sensitive to vector magnitude and word frequency.",
        isCorrect: true,
      },
      {
        text: "It yields a bounded and interpretable similarity score.",
        isCorrect: true,
      },
      {
        text: "It completely ignores the direction of the vectors.",
        isCorrect: false,
      },
      {
        text: "It is more robust to high-dimensional sparse data.",
        isCorrect: true,
      },
    ],
    explanation:
      "Cosine focuses on angle (direction), is bounded, and works well in high dimensions, whereas raw dot product is unbounded and magnitude-dependent.",
  },
  {
    id: "ch1-q19",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about the properties of word embeddings are TRUE?",
    options: [
      {
        text: "They can capture both syntactic and semantic relationships.",
        isCorrect: true,
      },
      {
        text: "They can support analogy operations such as king − man + woman ≈ queen.",
        isCorrect: true,
      },
      {
        text: "They perfectly distinguish synonyms from antonyms in all cases.",
        isCorrect: false,
      },
      {
        text: "They can be used to explore how word meanings shift over decades.",
        isCorrect: true,
      },
    ],
    explanation:
      "Embeddings encode rich relationships and analogies, but they cannot reliably separate synonyms from antonyms and can be used to track semantic drift over time.",
  },
  {
    id: "ch1-q20",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "In classic word2vec-style embeddings, a single embedding vector for a word can represent multiple senses. What is TRUE about this?",
    options: [
      {
        text: "An embedding can be seen as a linear superposition of different word senses.",
        isCorrect: true,
      },
      {
        text: "The contribution of each sense is roughly proportional to its frequency in the corpus.",
        isCorrect: true,
      },
      {
        text: "Operations like analogies can selectively emphasize certain semantic components.",
        isCorrect: true,
      },
      {
        text: "Each distinct meaning of a word must have a separate embedding vector.",
        isCorrect: false,
      },
    ],
    explanation:
      "Word2vec embeddings mix senses in a single vector; rarer senses contribute less. Vector arithmetic can emphasize particular semantic components without separate vectors per sense.",
  },
  {
    id: "ch1-q21",
    chapter: 1,
    difficulty: "easy",
    prompt: "What is a Recurrent Neural Network (RNN) mainly designed to handle?",
    options: [
      {
        text: "Sequences of inputs where the current output depends on previous elements.",
        isCorrect: true,
      },
      {
        text: "Independent examples where order carries no information.",
        isCorrect: false,
      },
      {
        text: "Time series such as text, speech, or sensor readings.",
        isCorrect: true,
      },
      {
        text: "Only fixed-length tabular data.",
        isCorrect: false,
      },
    ],
    explanation:
      "RNNs maintain a hidden state so they can model dependencies across time steps in sequences.",
  },
  {
    id: "ch1-q22",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "In a simple RNN with equations a(t), h(t), o(t), and y(t), which statements are TRUE?",
    options: [
      {
        text: "h(t) is the new hidden state, typically obtained by applying tanh to a(t).",
        isCorrect: true,
      },
      {
        text: "a(t) combines the previous hidden state h(t−1) and the current input x(t).",
        isCorrect: true,
      },
      {
        text: "y(t) is produced by applying a nonlinearity σ to the output vector o(t).",
        isCorrect: true,
      },
      {
        text: "There is no dependence on h(t−1); RNNs ignore past states.",
        isCorrect: false,
      },
    ],
    explanation:
      "The RNN uses h(t−1) and x(t) to compute a(t), then h(t)=tanh(a(t)), and o(t) then y(t) are derived from h(t).",
  },
  {
    id: "ch1-q23",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "What issues are associated with training simple RNNs on long sequences?",
    options: [
      {
        text: "Vanishing gradients during backpropagation through many time steps.",
        isCorrect: true,
      },
      {
        text: "Exploding gradients where values grow uncontrollably.",
        isCorrect: true,
      },
      {
        text: "Difficulty learning long-range dependencies.",
        isCorrect: true,
      },
      {
        text: "Too few parameters to overfit small datasets.",
        isCorrect: false,
      },
    ],
    explanation:
      "Backpropagation through time can cause gradients to vanish or explode, making it hard for simple RNNs to capture long-range patterns.",
  },
  {
    id: "ch1-q24",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about LSTMs (Long Short-Term Memory networks) are TRUE?",
    options: [
      {
        text: "They introduce gates to control information flow.",
        isCorrect: true,
      },
      {
        text: "They maintain both a short-term hidden state h and a long-term cell state c.",
        isCorrect: true,
      },
      {
        text: "They can decide to forget or retain information through a forget gate.",
        isCorrect: true,
      },
      {
        text: "They remove the need for any nonlinearity inside the recurrent cell.",
        isCorrect: false,
      },
    ],
    explanation:
      "LSTMs use gates and an explicit cell state to manage long- and short-term information while still using nonlinearities like tanh and sigmoid.",
  },
  {
    id: "ch1-q25",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "In the LSTM equations, which roles do the gates play?",
    options: [
      {
        text: "The forget gate f(t) decides how much of the previous cell state c(t−1) to keep.",
        isCorrect: true,
      },
      {
        text: "The input gate i(t) and candidate g(t) determine what new information is added to the cell state.",
        isCorrect: true,
      },
      {
        text: "The output gate o(t) controls how much of the cell state influences the hidden state h(t).",
        isCorrect: true,
      },
      {
        text: "All gates share exactly the same parameters; they differ only in their names.",
        isCorrect: false,
      },
    ],
    explanation:
      "Forget, input, and output gates each have their own parameters and control different aspects of how the cell state is updated and exposed.",
  },
  {
    id: "ch1-q26",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about GRUs (Gated Recurrent Units) are TRUE?",
    options: [
      {
        text: "They are a simplified alternative to LSTMs with fewer parameters.",
        isCorrect: true,
      },
      {
        text: "They use an update gate z(t) similar in spirit to the forget gate in LSTMs.",
        isCorrect: true,
      },
      {
        text: "They merge the input and forget gates into a reset gate and update gate.",
        isCorrect: true,
      },
      {
        text: "They maintain two distinct states (h and c) just like LSTMs.",
        isCorrect: false,
      },
    ],
    explanation:
      "GRUs keep a single hidden state and use update and reset gates, making them lighter than LSTMs while retaining gating behavior.",
  },
  {
    id: "ch1-q27",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "What are some typical trade-offs between LSTMs and GRUs?",
    options: [
      {
        text: "GRUs have fewer parameters and can converge faster.",
        isCorrect: true,
      },
      {
        text: "GRUs may be more prone to overfitting because of their smaller parameterization.",
        isCorrect: true,
      },
      {
        text: "LSTMs generally handle very long-term dependencies better.",
        isCorrect: true,
      },
      {
        text: "LSTMs are always strictly worse than GRUs on all tasks.",
        isCorrect: false,
      },
    ],
    explanation:
      "GRUs are lighter and often faster but may overfit more easily, while LSTMs tend to model complex long-range patterns better.",
  },
  {
    id: "ch1-q28",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "How are 1D Convolutional Neural Networks (CNNs) used for text?",
    options: [
      {
        text: "They slide 1D filters over sequences to extract local patterns.",
        isCorrect: true,
      },
      {
        text: "The filter size can be interpreted as a kind of context window over words.",
        isCorrect: true,
      },
      {
        text: "They can operate on embedding vectors instead of raw tokens.",
        isCorrect: true,
      },
      {
        text: "They require 2D image-like inputs and cannot process text sequences.",
        isCorrect: false,
      },
    ],
    explanation:
      "1D CNNs convolve filters over sequences (often embeddings), capturing n-gram-like local patterns similar to context windows.",
  },
  {
    id: "ch1-q29",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "In a GRU-based movie review sentiment model, what are key preprocessing steps before training?",
    options: [
      {
        text: "Cleaning text (removing extra spaces, special characters, some punctuation).",
        isCorrect: true,
      },
      {
        text: "Tokenizing reviews and mapping words to indices via a vocabulary.",
        isCorrect: true,
      },
      {
        text: "Padding sequences to a fixed length.",
        isCorrect: true,
      },
      {
        text: "Randomly dropping half of the reviews to simplify the dataset.",
        isCorrect: false,
      },
    ],
    explanation:
      "The pipeline cleans text, tokenizes, vectorizes via word indices, and pads to a fixed length so batches can be processed.",
  },
  {
    id: "ch1-q30",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which design choices are common for a GRU-based sentiment analysis model over movie reviews?",
    options: [
      {
        text: "An embedding layer to learn word vectors from the reviews.",
        isCorrect: true,
      },
      {
        text: "GRU layers as the main sequence model.",
        isCorrect: true,
      },
      {
        text: "Dropout for regularization.",
        isCorrect: true,
      },
      {
        text: "A final linear layer mapping features to a single output probability.",
        isCorrect: true,
      },
    ],
    explanation:
      "The model uses embedding + GRUs + dropout + a linear layer to predict a single sentiment probability.",
  },
  {
    id: "ch1-q31",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "For a binary movie-review sentiment classification task, which training components are typically used?",
    options: [
      {
        text: "Binary cross-entropy loss.",
        isCorrect: true,
      },
      {
        text: "The Adam optimizer.",
        isCorrect: true,
      },
      {
        text: "A loss function that expects three sentiment classes (positive/neutral/negative).",
        isCorrect: false,
      },
      {
        text: "Mini-batch training over many epochs.",
        isCorrect: true,
      },
    ],
    explanation:
      "A common setup uses BCE loss with Adam and trains in batches for the two-class sentiment task.",
  },
  {
    id: "ch1-q33",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Why is padding applied to tokenized review sequences before training the sentiment model?",
    options: [
      {
        text: "Neural networks expect fixed-shape batches, so sequence lengths must be harmonized.",
        isCorrect: true,
      },
      {
        text: "Padding allows us to pack multiple sequences into tensors of the same length.",
        isCorrect: true,
      },
      {
        text: "Padding improves semantic quality of the embeddings themselves.",
        isCorrect: false,
      },
      {
        text: "Padding is only required for CNNs, not for recurrent models.",
        isCorrect: false,
      },
    ],
    explanation:
      "Padding ensures all sequences in a batch share the same length; it is a general requirement for many sequence architectures, not just CNNs.",
  },
  {
    id: "ch1-q34",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "What kinds of visualizations are commonly used to explore word embedding spaces?",
    options: [
      {
        text: "Clustering of word embeddings.",
        isCorrect: true,
      },
      {
        text: "Dimensionality reduction to 2D using techniques like PCA, t-SNE, or UMAP.",
        isCorrect: true,
      },
      {
        text: "3D bar plots of raw one-hot vectors.",
        isCorrect: false,
      },
      {
        text: "Plots showing semantic shifts of words such as 'gay' or 'broadcast' across decades.",
        isCorrect: true,
      },
    ],
    explanation:
      "Common practice is to use clustering and 2D projections (PCA, t-SNE, UMAP), including diachronic views of how word meanings change over time.",
  },
  {
    id: "ch1-q35",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about the curse of dimensionality in text representations are TRUE?",
    options: [
      {
        text: "As the number of features grows, distances between examples become less meaningful.",
        isCorrect: true,
      },
      {
        text: "High-dimensional sparse vectors increase risk of overfitting.",
        isCorrect: true,
      },
      {
        text: "Using extremely large vocabularies with BoW amplifies this problem.",
        isCorrect: true,
      },
      {
        text: "Dense low-dimensional embeddings help mitigate this issue.",
        isCorrect: true,
      },
    ],
    explanation:
      "Sparse, high-dimensional representations make learning and distance metrics harder; embeddings help by compressing information into manageable dimensions.",
  },
  {
    id: "ch1-q36",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "If a corpus has a vocabulary of 10,000 unique words, what is TRUE about a one-hot representation of each word?",
    options: [
      {
        text: "Each word is mapped to a vector of length 10,000.",
        isCorrect: true,
      },
      {
        text: "Exactly one element of the vector is 1, the rest are 0.",
        isCorrect: true,
      },
      {
        text: "Two different words will always have orthogonal one-hot vectors.",
        isCorrect: true,
      },
      {
        text: "The representation automatically captures that 'cat' and 'dog' are similar animals.",
        isCorrect: false,
      },
    ],
    explanation:
      "One-hot vectors are vocab-length with a single 1; different words are orthogonal but semantic similarity is not encoded.",
  },
  {
    id: "ch1-q37",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which of the following make cosine similarity particularly suitable for text embeddings?",
    options: [
      {
        text: "It is cheap to compute even in high dimensions.",
        isCorrect: true,
      },
      {
        text: "It is not influenced by overall vector magnitude.",
        isCorrect: true,
      },
      {
        text: "Opposite vectors have similarity −1 and orthogonal vectors have similarity 0.",
        isCorrect: true,
      },
      {
        text: "It depends only on the Euclidean distance between the vectors.",
        isCorrect: false,
      },
    ],
    explanation:
      "Cosine similarity is angle-based, scale-invariant, bounded, and efficient to compute, which is ideal for embedding comparisons.",
  },
  {
    id: "ch1-q38",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Consider the logistic probability P(+|w,c) = σ(c·w). Which statements are TRUE?",
    options: [
      {
        text: "If c·w = 0, then P(+|w,c) = 0.5.",
        isCorrect: true,
      },
      {
        text: "Increasing c·w pushes the probability closer to 1.",
        isCorrect: true,
      },
      {
        text: "Very negative values of c·w yield probabilities close to 0.",
        isCorrect: true,
      },
      {
        text: "σ(x) is a linear function of x.",
        isCorrect: false,
      },
    ],
    explanation:
      "The sigmoid maps 0 → 0.5 and smoothly transitions from near 0 for large negative inputs to near 1 for large positive inputs; it is nonlinear.",
  },
  {
    id: "ch1-q39",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "In the RNN/LSTM/GRU discussion, which statements about gradient behavior are TRUE?",
    options: [
      {
        text: "Gradients in long unrolled RNNs can vanish, making learning long-range dependencies difficult.",
        isCorrect: true,
      },
      {
        text: "Gradients can also explode, causing unstable training.",
        isCorrect: true,
      },
      {
        text: "LSTMs reduce vanishing gradients partly through additive (plus) operations in the cell state.",
        isCorrect: true,
      },
      {
        text: "GRUs completely eliminate both vanishing and exploding gradients in all settings.",
        isCorrect: false,
      },
    ],
    explanation:
      "Vanishing/exploding gradients are fundamental challenges; LSTMs and GRUs mitigate but do not magically eliminate them.",
  },
  {
    id: "ch1-q40",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "What are some ways word embeddings can be *used* once trained?",
    options: [
      {
        text: "Finding most similar words via cosine similarity.",
        isCorrect: true,
      },
      {
        text: "Solving word analogies using vector arithmetic.",
        isCorrect: true,
      },
      {
        text: "Studying semantic change of words over time.",
        isCorrect: true,
      },
      {
        text: "Directly decoding full grammatical parse trees from a single embedding vector.",
        isCorrect: false,
      },
    ],
    explanation:
      "Embeddings support similarity search, analogies, and diachronic analysis but are not full parsers by themselves.",
  },
  {
    id: "ch1-q41",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which steps are part of constructing a BoW matrix for a text corpus?",
    options: [
      {
        text: "Tokenize each document into words.",
        isCorrect: true,
      },
      {
        text: "Build a vocabulary of unique words and map each word to an index.",
        isCorrect: true,
      },
      {
        text: "Create a document-by-vocabulary matrix where entries are word counts.",
        isCorrect: true,
      },
      {
        text: "Ensure each document vector has exactly one non-zero entry.",
        isCorrect: false,
      },
    ],
    explanation:
      "BoW counts word occurrences across documents; each row is a document vector with many possible non-zero entries.",
  },
  {
    id: "ch1-q42",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "In classic word2vec implementations, embeddings are often learned as *two* matrices (for w and c). Which statements are TRUE?",
    options: [
      {
        text: "There is one matrix for center words and one for context words.",
        isCorrect: true,
      },
      {
        text: "These matrices tend to be similar, so often one of them is used as 'the' embedding.",
        isCorrect: true,
      },
      {
        text: "They arise because the model has separate parameters for input and output roles.",
        isCorrect: true,
      },
      {
        text: "Keeping two completely unrelated matrices is required for using the embeddings.",
        isCorrect: false,
      },
    ],
    explanation:
      "Word2vec learns input (w) and output (c) embeddings; they are often averaged or one is chosen for downstream use.",
  },
  {
    id: "ch1-q43",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about CNNs versus RNNs for text are TRUE?",
    options: [
      {
        text: "CNNs are good at capturing local patterns via filters of limited size.",
        isCorrect: true,
      },
      {
        text: "RNNs explicitly model sequential order via recurrent connections.",
        isCorrect: true,
      },
      {
        text: "CNNs are inherently faster to parallelize than basic RNNs.",
        isCorrect: true,
      },
      {
        text: "CNNs are unable to use pretrained embeddings as input.",
        isCorrect: false,
      },
    ],
    explanation:
      "1D CNNs capture local n-gram patterns and parallelize well; RNNs model sequence order via hidden states, and both can use embeddings as input.",
  },
  {
    id: "ch1-q44",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Consider cosine similarity cos(θ) between two non-zero vectors a and b. Which statements are TRUE?",
    options: [
      {
        text: "cos(θ) = 1 when the vectors are perfectly aligned (same direction).",
        isCorrect: true,
      },
      {
        text: "cos(θ) = 0 when the vectors are orthogonal.",
        isCorrect: true,
      },
      {
        text: "cos(θ) = −1 when the vectors point in exactly opposite directions.",
        isCorrect: true,
      },
      {
        text: "Scaling one vector by 2 changes cos(θ).",
        isCorrect: false,
      },
    ],
    explanation:
      "Cosine depends only on angle, not magnitude, so scaling vectors doesn’t change it; its extreme values indicate alignment or opposition.",
  },
  {
    id: "ch1-q45",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "In a learned review-embedding space for sentiment analysis, what do final visualizations of review embeddings typically show?",
    options: [
      {
        text: "Before training, positive and negative reviews are intermixed in the embedding projection.",
        isCorrect: true,
      },
      {
        text: "After training, positive and negative reviews become more separable in the embedding space.",
        isCorrect: true,
      },
      {
        text: "The model has perfectly separated all points with no overlap at all.",
        isCorrect: false,
      },
      {
        text: "Embeddings can be used to visualize how the model organizes examples by sentiment.",
        isCorrect: true,
      },
    ],
    explanation:
      "Training improves separation of classes but not perfectly; projections show how the model clusters sentiments.",
  },
  {
    id: "ch1-q46",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about a standard TF-IDF implementation are TRUE?",
    options: [
      {
        text: "TF is often computed as count divided by total words in the document.",
        isCorrect: true,
      },
      {
        text: "IDF uses a log of (N / (1 + df)) with smoothing.",
        isCorrect: true,
      },
      {
        text: "TF-IDF for a term is TF multiplied by IDF.",
        isCorrect: true,
      },
      {
        text: "IDF is larger for words that appear in many documents.",
        isCorrect: false,
      },
    ],
    explanation:
      "IDF grows when df is small (rare terms) and shrinks for terms appearing in many documents; TF is normalized frequency.",
  },
  {
    id: "ch1-q47",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which elements are potential sources of bias or variation in embeddings?",
    options: [
      {
        text: "Training on social media like Twitter can introduce bias.",
        isCorrect: true,
      },
      {
        text: "Mixing different domains (e.g., Wikipedia + news) can change performance on different tasks.",
        isCorrect: true,
      },
      {
        text: "Using more dimensions always removes bias completely.",
        isCorrect: false,
      },
      {
        text: "Common Crawl is large but relatively noisy.",
        isCorrect: true,
      },
    ],
    explanation:
      "Domain, data quality, and noise affect learned embeddings; more dimensions don’t magically remove bias.",
  },
  {
    id: "ch1-q48",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about a typical PyTorch training loop for the sentiment model are TRUE?",
    options: [
      {
        text: "The model switches to training mode with model.train().",
        isCorrect: true,
      },
      {
        text: "Hidden states are detached between batches to avoid backpropagating through the entire epoch.",
        isCorrect: true,
      },
      {
        text: "Loss.backward() is called to compute gradients, followed by optimizer.step().",
        isCorrect: true,
      },
      {
        text: "No validation set is used; only training loss is monitored.",
        isCorrect: false,
      },
    ],
    explanation:
      "The loop uses standard PyTorch training practices and also tracks validation loss/accuracy.",
  },
  {
    id: "ch1-q49",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Compared with BoW and TF-IDF, what advantages do learned embeddings plus deep models typically provide?",
    options: [
      {
        text: "They can capture contextual and semantic relationships, not just raw counts.",
        isCorrect: true,
      },
      {
        text: "They enable end-to-end learning for tasks such as sentiment analysis.",
        isCorrect: true,
      },
      {
        text: "They completely eliminate the need for any preprocessing.",
        isCorrect: false,
      },
      {
        text: "They often achieve better performance than traditional models like naïve Bayes.",
        isCorrect: true,
      },
    ],
    explanation:
      "Embeddings + deep models learn features jointly with the task and usually outperform classical bag-of-words approaches, though preprocessing is still needed.",
  },
  {
    id: "ch1-q50",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which of the following are core topics in an introductory chapter on analyzing text data with deep learning?",
    options: [
      {
        text: "Representing text for AI (one-hot, BoW, TF-IDF).",
        isCorrect: true,
      },
      {
        text: "Word embeddings such as word2vec and their applications.",
        isCorrect: true,
      },
      {
        text: "RNNs, LSTMs, GRUs, and CNNs for text.",
        isCorrect: true,
      },
      {
        text: "Reinforcement learning for robotic control.",
        isCorrect: false,
      },
    ],
    explanation:
      "An introductory deep-learning-for-text chapter typically covers text representations, embeddings, sequence models, and sentiment analysis—not robotics RL.",
  },
];
