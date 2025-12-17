import { Question } from "../../quiz";

export const stanfordCME295Lecture1Questions: Question[] = [
  // ============================================================
  //  Q1–Q18: 4 correct answers (ALL TRUE)
  // ============================================================

  {
    id: "cme295-lect1-q01",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements describe typical goals of Natural Language Processing (NLP)?",
    options: [
      { text: "Automatically analyze and understand human language text.", isCorrect: true },
      { text: "Transform raw text into numerical representations that models can process.", isCorrect: true },
      { text: "Build systems that can classify, extract, or generate text.", isCorrect: true },
      { text: "Support downstream applications such as chatbots, search, and translation.", isCorrect: true },
    ],
    explanation:
      "NLP is about letting computers analyze, represent, and manipulate human language to power applications such as classification, extraction, and text generation."
  },

  {
    id: "cme295-lect1-q02",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which problems are standard examples of text classification tasks?",
    options: [
      { text: "Sentiment analysis of movie or product reviews.", isCorrect: true },
      { text: "Intent detection in a virtual assistant (for example, setting an alarm).", isCorrect: true },
      { text: "Language identification (for example, detecting that a sentence is French).", isCorrect: true },
      { text: "Topic classification of documents or posts.", isCorrect: true },
    ],
    explanation:
      "Classification tasks map an input text to one label such as sentiment, intent, language, or topic."
  },

  {
    id: "cme295-lect1-q03",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which tasks fall under the multi-label or structured prediction category in NLP?",
    options: [
      { text: "Named Entity Recognition (identifying locations, people, dates, etc.).", isCorrect: true },
      { text: "Part-of-speech tagging (labeling nouns, verbs, adjectives, and so on).", isCorrect: true },
      { text: "Dependency or constituency parsing of syntactic structure.", isCorrect: true },
      { text: "Assigning multiple categories or attributes to a single piece of text.", isCorrect: true },
    ],
    explanation:
      "These tasks predict multiple outputs per input, often at the token or span level (entities, tags, parse structure, or multiple labels)."
  },

  {
    id: "cme295-lect1-q04",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which tasks are typical examples of text generation in NLP?",
    options: [
      { text: "Machine translation from one language to another.", isCorrect: true },
      { text: "Question answering where the system writes a natural-language answer.", isCorrect: true },
      { text: "Summarization of long documents or articles.", isCorrect: true },
      { text: "Free-form generation such as code, stories, or poems.", isCorrect: true },
    ],
    explanation:
      "Generation tasks take text as input and produce new text outputs such as translations, answers, summaries, or creative content."
  },

  {
    id: "cme295-lect1-q05",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements about tokenization are correct?",
    options: [
      { text: "Tokenization splits raw text into smaller units called tokens.", isCorrect: true },
      { text: "Tokens can be whole words, subwords, or individual characters.", isCorrect: true },
      { text: "Tokenization is required so that text can be fed into neural models.", isCorrect: true },
      { text: "Different tokenization schemes have different trade-offs in vocabulary size and sequence length.", isCorrect: true },
    ],
    explanation:
      "Tokenization defines the basic units of text on which models operate, with design choices affecting vocabulary, length, and robustness."
  },

  {
    id: "cme295-lect1-q06",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements describe advantages of subword tokenization?",
    options: [
      { text: "It allows common roots or stems (for example, 'bear' in 'bear' and 'bears') to be shared.", isCorrect: true },
      { text: "It reduces the chance of truly unseen tokens by decomposing rare words.", isCorrect: true },
      { text: "It balances between word-level semantics and character-level robustness.", isCorrect: true },
      { text: "It tends to keep sequence lengths moderate compared with character-level tokenization.", isCorrect: true },
    ],
    explanation:
      "Subword tokenization shares roots, handles rare words, avoids extreme sequence lengths, and is a pragmatic trade-off widely used in modern LLMs."
  },

  {
    id: "cme295-lect1-q07",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe the out-of-vocabulary (OOV) problem?",
    options: [
      { text: "Word-level tokenization can assign many unseen words to an unknown token.", isCorrect: true },
      { text: "Subword tokenization reduces OOV rate by composing unseen words from known pieces.", isCorrect: true },
      { text: "Character-level tokenization essentially eliminates OOV for alphabetic scripts.", isCorrect: true },
      { text: "Handling OOV tokens is important because models need representations for words not seen during training.", isCorrect: true },
    ],
    explanation:
      "OOV is a core issue in word-level schemes; subword and character tokenizations largely mitigate it by decomposing words into reusable parts."
  },

  {
    id: "cme295-lect1-q08",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements about one-hot encoding of tokens are correct?",
    options: [
      { text: "Each token is represented as a vector with a single 1 and the rest 0s.", isCorrect: true },
      { text: "All one-hot vectors in the same vocabulary are orthogonal to each other.", isCorrect: true },
      { text: "One-hot vectors do not encode any notion of semantic similarity.", isCorrect: true },
      { text: "One-hot encodings are typically very high-dimensional when vocabularies are large.", isCorrect: true },
    ],
    explanation:
      "One-hot encodings provide unique IDs but no semantics; every token is orthogonal regardless of meaning, and dimensionality grows with vocabulary size."
  },

  {
    id: "cme295-lect1-q09",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe word embeddings such as Word2vec?",
    options: [
      { text: "They map tokens to dense, low-dimensional vectors.", isCorrect: true },
      { text: "They are learned from data so that similar words have similar vectors.", isCorrect: true },
      { text: "They are trained via proxy tasks such as predicting context words or a target word.", isCorrect: true },
      { text: "They support similarity measures such as cosine similarity to compare word meanings.", isCorrect: true },
    ],
    explanation:
      "Word2vec-style embeddings use proxy prediction tasks to learn dense vectors where semantic similarity is captured as geometric similarity."
  },

  {
    id: "cme295-lect1-q10",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about the Continuous Bag of Words (CBOW) and Skip-Gram models in Word2vec are correct?",
    options: [
      { text: "Both are trained on large unlabeled text corpora.", isCorrect: true },
      { text: "CBOW predicts a target word from its surrounding context words.", isCorrect: true },
      { text: "Skip-Gram predicts surrounding context words from a central target word.", isCorrect: true },
      { text: "Both treat these prediction tasks as proxies for learning useful word embeddings.", isCorrect: true },
    ],
    explanation:
      "CBOW and Skip-Gram reverse the direction of prediction but share the core idea of learning embeddings via context-based proxy tasks."
  },

  {
    id: "cme295-lect1-q11",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements describe Recurrent Neural Networks (RNNs) for sequence modeling?",
    options: [
      { text: "They process sequences step by step, maintaining a hidden state.", isCorrect: true },
      { text: "The hidden state summarizes information from previous time steps.", isCorrect: true },
      { text: "They can be used for classification, tagging, and generation tasks.", isCorrect: true },
      { text: "They were introduced long before the transformer architecture.", isCorrect: true },
    ],
    explanation:
      "RNNs use a recurrent hidden state to process sequences and predate transformers by decades."
  },

  {
    id: "cme295-lect1-q12",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements describe the Long Short-Term Memory (LSTM) architecture?",
    options: [
      { text: "It is a specific type of recurrent neural network.", isCorrect: true },
      { text: "It introduces a cell state in addition to the hidden state.", isCorrect: true },
      { text: "It aims to better preserve information over longer sequences.", isCorrect: true },
      { text: "It was designed to mitigate the vanishing gradient problem in standard recurrent neural networks.", isCorrect: true },
    ],
    explanation:
      "LSTMs add gating and a cell state to help preserve important information and reduce vanishing gradients compared to vanilla RNNs."
  },

  {
    id: "cme295-lect1-q13",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements describe typical issues with standard Recurrent Neural Networks (RNNs)?",
    options: [
      { text: "They can struggle to capture very long-range dependencies.", isCorrect: true },
      { text: "Backpropagation through many time steps can cause gradients to vanish or explode.", isCorrect: true },
      { text: "Their sequential nature leads to slow training on long sequences.", isCorrect: true },
      { text: "The entire sentence meaning can be bottlenecked into a single hidden state.", isCorrect: true },
    ],
    explanation:
      "RNNs face long-range dependency issues, vanishing/exploding gradients, sequential computation, and bottlenecked representations."
  },

  {
    id: "cme295-lect1-q14",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe the motivation behind attention mechanisms?",
    options: [
      { text: "To provide direct connections between a prediction and relevant positions in the input sequence.", isCorrect: true },
      { text: "To mitigate the difficulty of remembering faraway tokens in long sequences.", isCorrect: true },
      { text: "To let the model weight different input tokens based on their relevance to the current prediction.", isCorrect: true },
      { text: "To move away from strictly sequential dependence on a single hidden state.", isCorrect: true },
    ],
    explanation:
      "Attention lets the model focus on relevant parts of the sequence directly, helping with long-range dependencies and bottlenecks."
  },

  {
    id: "cme295-lect1-q15",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements describe self-attention in the transformer architecture?",
    options: [
      { text: "Each token representation is updated by looking at all tokens in the sequence.", isCorrect: true },
      { text: "Attention weights indicate how much each token attends to every other token.", isCorrect: true },
      { text: "Self-attention can be implemented efficiently with matrix multiplications.", isCorrect: true },
      { text: "Self-attention is a central building block of the transformer encoder and decoder.", isCorrect: true },
    ],
    explanation:
      "Self-attention lets each position attend to all positions to build context-aware token representations using matrix operations well suited to GPUs."
  },

  {
    id: "cme295-lect1-q16",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe queries, keys, and values in attention?",
    options: [
      { text: "Queries, keys, and values are learned linear projections of the same input embeddings.", isCorrect: true },
      { text: "Attention weights are computed by comparing queries with keys.", isCorrect: true },
      { text: "Values are combined with attention weights to form the final context vectors.", isCorrect: true },
      { text: "All three (queries, keys, values) are trainable via gradient descent along with the rest of the model.", isCorrect: true },
    ],
    explanation:
      "Self-attention projects inputs into query, key, and value spaces; query–key similarities yield weights that mix corresponding values."
  },

  {
    id: "cme295-lect1-q17",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements correctly describe multi-head attention?",
    options: [
      { text: "It runs several independent attention mechanisms (heads) in parallel.", isCorrect: true },
      { text: "Each head uses its own projection matrices for queries, keys, and values.", isCorrect: true },
      { text: "The outputs of all heads are concatenated and then linearly projected back to the model dimension.", isCorrect: true },
      { text: "Multiple heads allow the model to capture different types of relations between tokens.", isCorrect: true },
    ],
    explanation:
      "Multi-head attention provides multiple learned projections that attend in different ways, then merges them into a single representation."
  },

  {
    id: "cme295-lect1-q18",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements describe key structural elements of the original transformer architecture?",
    options: [
      { text: "It consists of a stack of encoders and a stack of decoders.", isCorrect: true },
      { text: "Encoder self-attention is fully bidirectional over the input sequence.", isCorrect: true },
      { text: "Decoder self-attention is masked to prevent attending to future target tokens.", isCorrect: true },
      { text: "Decoder layers include a cross-attention block that attends to encoder outputs.", isCorrect: true },
    ],
    explanation:
      "The transformer uses stacked encoder and decoder blocks with self-attention, masked self-attention in the decoder, and cross-attention from decoder to encoder outputs."
  },

  // ============================================================
  //  Q19–Q36: 3 correct answers, 1 incorrect
  // ============================================================

  {
    id: "cme295-lect1-q20",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about evaluation metrics for binary classification are correct?",
    options: [
      { text: "Accuracy can be misleading when classes are highly imbalanced.", isCorrect: true },
      { text: "Precision measures, among predicted positives, how many are actually positive.", isCorrect: true },
      { text: "Recall measures, among true positives, how many were correctly predicted positive.", isCorrect: true },
      { text: "The F1 score is the arithmetic mean of precision and recall.", isCorrect: false },
    ],
    explanation:
      "F1 is the harmonic mean of precision and recall, not the arithmetic mean; accuracy alone can be misleading on imbalanced data."
  },

  {
    id: "cme295-lect1-q21",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about BLEU and ROUGE are correct?",
    options: [
      { text: "They are reference-based metrics comparing model outputs to human-written texts.", isCorrect: true },
      { text: "They were widely used for evaluating machine translation and summarization.", isCorrect: true },
      { text: "They rely on overlapping n-grams between prediction and reference.", isCorrect: true },
      { text: "They can be computed without any reference outputs.", isCorrect: false },
    ],
    explanation:
      "BLEU and ROUGE compare model outputs with reference texts using n-gram overlaps; they require references and are not reference-free."
  },

  {
    id: "cme295-lect1-q22",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements describe perplexity in the context of language models?",
    options: [
      { text: "Perplexity is derived from the probabilities that the model assigns to the correct tokens.", isCorrect: true },
      { text: "Lower perplexity indicates the model is less surprised by the data.", isCorrect: true },
      { text: "Perplexity can serve as a training-time signal even without reference translations.", isCorrect: true },
      { text: "Higher perplexity always implies better generative creativity.", isCorrect: false },
    ],
    explanation:
      "Perplexity is an uncertainty measure based on model probabilities; lower perplexity is generally better for prediction and does not directly measure creativity."
  },

  {
    id: "cme295-lect1-q23",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements about character-level tokenization are correct?",
    options: [
      { text: "It tends to produce much longer token sequences than word-level tokenization.", isCorrect: true },
      { text: "It is robust to misspellings and unusual word forms.", isCorrect: true },
      { text: "It avoids out-of-vocabulary issues for alphabetic writing systems.", isCorrect: true },
      { text: "It guarantees that sequences are shorter than with subword tokenization.", isCorrect: false },
    ],
    explanation:
      "Character-level tokenization is robust and OOV-free but produces very long sequences, not shorter ones."
  },

  {
    id: "cme295-lect1-q24",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe cosine similarity for comparing token representations?",
    options: [
      { text: "It depends on the angle between two vectors rather than their norms.", isCorrect: true },
      { text: "It is often used to measure semantic similarity between embeddings.", isCorrect: true },
      { text: "With one-hot encodings, cosine similarity between any two different tokens is zero.", isCorrect: true },
      { text: "Cosine similarity directly measures how often two words co-occur in the training corpus.", isCorrect: false },
    ],
    explanation:
      "Cosine similarity measures angular similarity between vectors; with orthogonal one-hots, similarities vanish; it does not directly measure co-occurrence counts."
  },

  {
    id: "cme295-lect1-q25",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about the training objective of Word2vec-style models are correct?",
    options: [
      { text: "The proxy task might be predicting the next word given previous words.", isCorrect: true },
      { text: "The learned parameters include embeddings that can be reused for downstream tasks.", isCorrect: true },
      { text: "The loss is often based on cross-entropy between predicted and true distributions over the vocabulary.", isCorrect: true },
      { text: "The final goal is to keep using the full predictive model, not its learned embeddings.", isCorrect: false },
    ],
    explanation:
      "The full predictive model is usually just a means to learn embeddings; the embeddings are then reused while the original prediction head may be discarded."
  },

  {
    id: "cme295-lect1-q26",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements describe the vanishing gradient problem in recurrent neural networks?",
    options: [
      { text: "Gradients are backpropagated through many time steps in sequence models.", isCorrect: true },
      { text: "Repeated multiplication by Jacobians with eigenvalues less than one can shrink gradients toward zero.", isCorrect: true },
      { text: "Very small gradients make it difficult to learn long-range dependencies.", isCorrect: true },
      { text: "The problem arises because gradients are added, not multiplied, across time.", isCorrect: false },
    ],
    explanation:
      "Backpropagation through time multiplies many Jacobians; when these shrink, gradients vanish and long-range learning becomes difficult."
  },

  {
    id: "cme295-lect1-q27",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about the history of models for language are correct, as described in the lecture?",
    options: [
      { text: "Recurrent neural network ideas date back to the 1980s.", isCorrect: true },
      { text: "Long Short-Term Memory networks were proposed in the 1990s.", isCorrect: true },
      { text: "Word2vec popularized word embeddings in the early 2010s.", isCorrect: true },
      { text: "Transformers were first introduced in the early 1990s.", isCorrect: false },
    ],
    explanation:
      "Transformers were introduced in 2017, not the early 1990s; the other historical points match the lecture."
  },

  {
    id: "cme295-lect1-q28",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe the encoder in a transformer?",
    options: [
      { text: "It processes the input sequence in a fully parallel way using self-attention.", isCorrect: true },
      { text: "It produces context-aware embeddings for each input position.", isCorrect: true },
      { text: "Its self-attention is not masked and can attend to all positions.", isCorrect: true },
      { text: "It directly generates the target-language tokens one by one.", isCorrect: false },
    ],
    explanation:
      "The encoder builds contextual representations of the source sequence; decoding target tokens is handled by the decoder, not the encoder."
  },

  {
    id: "cme295-lect1-q29",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe the decoder in a transformer for sequence-to-sequence tasks?",
    options: [
      { text: "It uses masked self-attention over previously generated target tokens.", isCorrect: true },
      { text: "It uses cross-attention to attend to the encoder’s output representations.", isCorrect: true },
      { text: "It predicts the next target token via a linear layer plus softmax over the vocabulary.", isCorrect: true },
      { text: "It can freely attend to future target tokens during training.", isCorrect: false },
    ],
    explanation:
      "Decoder self-attention is masked to prevent peeking at future tokens; cross-attention connects decoder queries to encoder outputs."
  },

  {
    id: "cme295-lect1-q30",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements about the attention weight computation QKᵀ / √dₖ followed by softmax are correct?",
    options: [
      { text: "The dot products between queries and keys measure compatibility between positions.", isCorrect: true },
      { text: "Dividing by the square root of the key dimension keeps dot products numerically well-scaled.", isCorrect: true },
      { text: "The softmax converts raw scores into a probability distribution over keys for each query.", isCorrect: true },
      { text: "Removing the scaling factor always improves training stability.", isCorrect: false },
    ],
    explanation:
      "Scaling by √dₖ prevents dot products from growing too large; softmax turns them into normalized attention weights."
  },

  {
    id: "cme295-lect1-q31",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about positional encodings in transformers are correct?",
    options: [
      { text: "They inject information about token positions into the model.", isCorrect: true },
      { text: "In the original paper, they are implemented as sinusoidal functions added to embeddings.", isCorrect: true },
      { text: "They are combined elementwise with token embeddings before entering the encoder.", isCorrect: true },
      { text: "They are unnecessary because self-attention alone encodes order.", isCorrect: false },
    ],
    explanation:
      "Self-attention is permutation-invariant; positional encodings are required to encode order, and the original transformer used sinusoidal encodings added to embeddings."
  },

  {
    id: "cme295-lect1-q32",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements about the feedforward network (FFN) layers inside transformer blocks are correct?",
    options: [
      { text: "They apply the same two-layer feedforward network independently to each position.", isCorrect: true },
      { text: "They typically expand the hidden dimension before projecting back down.", isCorrect: true },
      { text: "They increase the model’s capacity to learn non-linear transformations of the attended representations.", isCorrect: true },
      { text: "They are responsible for computing attention weights between positions.", isCorrect: false },
    ],
    explanation:
      "Attention computes interactions between positions; FFN layers then apply position-wise non-linear transformations, usually with an expanded intermediate dimension."
  },

  {
    id: "cme295-lect1-q33",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about label smoothing in sequence models are correct?",
    options: [
      { text: "It replaces a hard one-hot target distribution with a slightly softened version.", isCorrect: true },
      { text: "It acknowledges that multiple next tokens can be reasonable in natural language.", isCorrect: true },
      { text: "It can reduce overconfidence of the model on the training data.", isCorrect: true },
      { text: "It sets the target probability of the correct class exactly to one.", isCorrect: false },
    ],
    explanation:
      "Label smoothing lowers the target probability for the correct class slightly and distributes some mass over others, which can improve generalization and calibration."
  },

  {
    id: "cme295-lect1-q34",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements about masked self-attention in the decoder are correct?",
    options: [
      { text: "The attention mask prevents attending to positions to the right of the current token.", isCorrect: true },
      { text: "The mask enforces an autoregressive factorization during training.", isCorrect: true },
      { text: "The mask ensures that predictions do not use future target tokens as information.", isCorrect: true },
      { text: "The mask is also applied in the encoder’s self-attention.", isCorrect: false },
    ],
    explanation:
      "Only decoder self-attention is masked to prevent using future tokens; encoder self-attention can attend to all positions."
  },

  {
    id: "cme295-lect1-q35",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about using RNNs versus transformers for long sequences are correct?",
    options: [
      { text: "Transformers process all positions in parallel within a layer, while RNNs are strictly sequential.", isCorrect: true },
      { text: "Transformers use attention to connect distant positions directly.", isCorrect: true },
      { text: "RNNs rely on repeatedly updating a single hidden state across time steps.", isCorrect: true },
      { text: "Standard RNNs scale better than transformers as sequence length grows.", isCorrect: false },
    ],
    explanation:
      "Transformers parallelize sequence processing and connect distant positions via attention; RNNs are sequential and struggle with long-range dependencies."
  },

  {
    id: "cme295-lect1-q36",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements about the training loop for a transformer-based machine translation model are correct?",
    options: [
      { text: "The source sentence is encoded via the encoder into context-aware embeddings.", isCorrect: true },
      { text: "The decoder processes the target sentence with masked self-attention during training.", isCorrect: true },
      { text: "The model predicts each next target token conditioned on previous target tokens and the encoded source.", isCorrect: true },
      { text: "The model is trained by directly minimizing BLEU scores.", isCorrect: false },
    ],
    explanation:
      "Transformers are typically trained with cross-entropy over next-token predictions; BLEU is used as an evaluation metric, not a direct training loss."
  },

  // ============================================================
  //  Q37–Q53: 2 correct answers, 2 incorrect
  // ============================================================

  {
    id: "cme295-lect1-q37",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements correctly describe the three high-level categories of NLP tasks discussed in the lecture?",
    options: [
      { text: "Classification tasks map text to a single label.", isCorrect: true },
      { text: "Generation tasks map text to variable-length text outputs.", isCorrect: true },
      { text: "Segmentation tasks were presented as the third main category.", isCorrect: false },
      { text: "Clustering tasks were presented as the third main category.", isCorrect: false },
    ],
    explanation:
      "The lecture emphasized classification, multi-label/structured prediction, and generation tasks, not clustering or segmentation as top-level categories."
  },

  {
    id: "cme295-lect1-q38",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements correctly describe multi-label or multi-output tasks such as Named Entity Recognition?",
    options: [
      { text: "They assign labels to multiple parts of a single input sequence.", isCorrect: true },
      { text: "They can evaluate predictions at the token or entity level.", isCorrect: true },
      { text: "They were described as mapping a sentence to exactly one label.", isCorrect: false },
      { text: "They ignore spans of text and only label entire documents.", isCorrect: false },
    ],
    explanation:
      "Named Entity Recognition and similar tasks label multiple positions or spans; they are not single-label document classifiers."
  },

  {
    id: "cme295-lect1-q39",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about the trade-offs between word-level and subword-level tokenization are correct?",
    options: [
      { text: "Word-level tokenization typically leads to smaller sequence lengths than subword tokenization.", isCorrect: true },
      { text: "Subword tokenization reduces the risk of out-of-vocabulary tokens compared to word-level tokenization.", isCorrect: true },
      { text: "Word-level tokenization is more robust to misspellings than subword tokenization.", isCorrect: false },
      { text: "Subword tokenization was described as ignoring roots and morphological structure.", isCorrect: false },
    ],
    explanation:
      "Word-level tokens keep sequence length short but suffer from OOV; subwords leverage roots and reduce OOV but lengthen sequences."
  },

  {
    id: "cme295-lect1-q40",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about vocabulary size selection are correct according to the lecture?",
    options: [
      { text: "Monolingual tokenizers often target tens of thousands of subword tokens.", isCorrect: true },
      { text: "Multilingual or code-aware tokenizers can reach hundreds of thousands of tokens.", isCorrect: true },
      { text: "The ideal vocabulary size is determined purely by a closed-form mathematical formula.", isCorrect: false },
      { text: "Vocabulary size has no connection to computational cost.", isCorrect: false },
    ],
    explanation:
      "The lecture mentioned rule-of-thumb ranges and trade-offs; vocabulary size affects memory and compute, not just theory."
  },

  {
    id: "cme295-lect1-q41",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about sentence-level representations from static word embeddings are correct?",
    options: [
      { text: "Averaging word embeddings is a simple way to create a sentence vector.", isCorrect: true },
      { text: "Averaging word embeddings discards word order information.", isCorrect: true },
      { text: "Averaging static embeddings fully captures contextual meaning of each word.", isCorrect: false },
      { text: "Static embeddings were presented as inherently context-dependent in this lecture.", isCorrect: false },
    ],
    explanation:
      "Static embeddings do not represent context; averaging them ignores order and nuanced meaning beyond crude bag-of-words semantics."
  },

  {
    id: "cme295-lect1-q42",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements about Long Short-Term Memory (LSTM) networks are correct in comparison to vanilla recurrent neural networks?",
    options: [
      { text: "LSTMs track both a hidden state and a separate cell state.", isCorrect: true },
      { text: "LSTMs were designed to better preserve important information over longer time spans.", isCorrect: true },
      { text: "LSTMs remove the sequential nature of processing sequences.", isCorrect: false },
      { text: "LSTMs completely eliminate vanishing gradients in practice.", isCorrect: false },
    ],
    explanation:
      "LSTMs still process sequences sequentially and reduce, but do not completely eliminate, vanishing gradient issues."
  },

  {
    id: "cme295-lect1-q43",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about the computational characteristics of transformers versus recurrent neural networks are correct?",
    options: [
      { text: "Transformers use attention to enable parallel computation across sequence positions within each layer.", isCorrect: true },
      { text: "Recurrent neural networks must process tokens one time step after another.", isCorrect: true },
      { text: "Transformers were presented as strictly slower than recurrent neural networks for long sequences.", isCorrect: false },
      { text: "Recurrent neural networks were presented as inherently more parallelizable than transformers.", isCorrect: false },
    ],
    explanation:
      "Transformers parallelize over positions via attention; RNNs are inherently sequential, making them less parallel-friendly."
  },

  {
    id: "cme295-lect1-q44",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements about cross-attention in the transformer decoder are correct?",
    options: [
      { text: "Decoder queries attend to keys and values derived from encoder outputs.", isCorrect: true },
      { text: "Cross-attention lets the decoder condition its predictions on the encoded source sequence.", isCorrect: true },
      { text: "Cross-attention was described as operating only within the decoder without looking at encoder states.", isCorrect: false },
      { text: "Cross-attention is responsible for adding positional information to encoder embeddings.", isCorrect: false },
    ],
    explanation:
      "Cross-attention connects decoder queries to encoder key–value pairs; position encodings are added earlier and are not the job of cross-attention."
  },

  {
    id: "cme295-lect1-q45",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe how probabilities over the vocabulary are produced in the transformer decoder?",
    options: [
      { text: "The final decoder hidden state is passed through a linear layer.", isCorrect: true },
      { text: "A softmax layer converts logits into a probability distribution over tokens.", isCorrect: true },
      { text: "Probabilities are obtained by applying cosine similarity directly to embeddings without any linear layer.", isCorrect: false },
      { text: "Probabilities come from attention weights directly, without any additional transformation.", isCorrect: false },
    ],
    explanation:
      "Decoder outputs are typically projected via a learned linear layer into vocabulary logits, then normalized with softmax; attention weights themselves are not token probabilities."
  },

  {
    id: "cme295-lect1-q46",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements about label smoothing’s effect on the target distribution are correct?",
    options: [
      { text: "It reduces the target probability assigned to the correct class slightly.", isCorrect: true },
      { text: "It spreads a small amount of probability mass over the incorrect classes.", isCorrect: true },
      { text: "It makes the target distribution exactly uniform over all classes.", isCorrect: false },
      { text: "It increases the loss when the model assigns moderate probability to the correct class.", isCorrect: false },
    ],
    explanation:
      "Label smoothing softens targets, encouraging less overconfident predictions; it does not make the target uniform nor penalize moderate confidence more than a hard target would."
  },

  {
    id: "cme295-lect1-q47",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about the use of special tokens such as BOS (begin-of-sequence) and EOS (end-of-sequence) are correct?",
    options: [
      { text: "BOS indicates the start of a sequence for the decoder to begin generation.", isCorrect: true },
      { text: "EOS marks where generation should stop.", isCorrect: true },
      { text: "These tokens are unnecessary when training autoregressive language models.", isCorrect: false },
      { text: "These tokens are used only for character-level models and not for word-based models.", isCorrect: false },
    ],
    explanation:
      "BOS and EOS tokens structure sequences for both training and generation in many models, regardless of tokenization granularity."
  },

  {
    id: "cme295-lect1-q48",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements about computing self-attention with matrices are correct?",
    options: [
      { text: "The query, key, and value matrices each contain one row per token position.", isCorrect: true },
      { text: "The product QKᵀ results in a matrix of attention scores between all pairs of positions.", isCorrect: true },
      { text: "The product QKᵀ has the same shape as the original embedding matrix.", isCorrect: false },
      { text: "Multiplying attention weights by V produces token-wise scalar scores instead of vectors.", isCorrect: false },
    ],
    explanation:
      "QKᵀ is an n×n matrix of pairwise scores; multiplying by V yields an n×d matrix of context vectors, not scalars."
  },

  {
    id: "cme295-lect1-q49",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about the logistic and organizational aspects of the course, as described in the lecture, are correct?",
    options: [
      { text: "Slides and recordings are posted online after each lecture.", isCorrect: true },
      { text: "Students can ask questions via an online forum integrated into the course platform.", isCorrect: true },
      { text: "The exams include programming questions that must be solved live in class.", isCorrect: false },
      { text: "Attending in person is mandatory because the class is not recorded.", isCorrect: false },
    ],
    explanation:
      "The lecture emphasized recordings and online resources; exams are concept-based, not live coding, and attendance is flexible."
  },

  {
    id: "cme295-lect1-q50",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements about training sequence models on paired data for machine translation are correct?",
    options: [
      { text: "Datasets such as WMT provide paired sentences in different languages.", isCorrect: true },
      { text: "Paired datasets are more expensive to collect than single-language corpora.", isCorrect: true },
      { text: "The lecture described translation training as using only monolingual corpora.", isCorrect: false },
      { text: "The lecture stated that no labels are needed for supervised translation training.", isCorrect: false },
    ],
    explanation:
      "Supervised translation requires aligned sentence pairs, which are costlier to obtain than raw monolingual text."
  },

  {
    id: "cme295-lect1-q51",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements about the relationship between softmax and label smoothing in training are correct?",
    options: [
      { text: "Softmax is applied to model logits to produce a probability distribution.", isCorrect: true },
      { text: "Label smoothing modifies the target distribution used in the loss, not the softmax operation itself.", isCorrect: true },
      { text: "Label smoothing was described as replacing softmax with a different activation function.", isCorrect: false },
      { text: "Softmax alone was described as sufficient to express uncertainty in the target distribution.", isCorrect: false },
    ],
    explanation:
      "Softmax produces model probabilities; label smoothing changes the target distribution in the loss to avoid overly sharp targets."
  },

  {
    id: "cme295-lect1-q52",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about evaluation of generation tasks such as translation are correct?",
    options: [
      { text: "There are many valid outputs for the same input, which complicates evaluation.", isCorrect: true },
      { text: "Traditional metrics like BLEU and ROUGE compare against reference outputs.", isCorrect: true },
      { text: "Collecting high-quality reference outputs is time-consuming and expensive.", isCorrect: true },
      { text: "The lecture claimed that reference-based metrics are unnecessary for all generative tasks.", isCorrect: false },
    ],
    explanation:
      "The lecture emphasized difficulties of reference-based evaluation and hinted at newer methods, not that references are universally unnecessary."
  },

  {
    id: "cme295-lect1-q53",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about the proxy nature of tasks used to learn embeddings (such as in Word2vec) are correct?",
    options: [
      { text: "The immediate training objective is predicting context or target words in text.", isCorrect: true },
      { text: "The underlying goal is to obtain meaningful embeddings for downstream tasks.", isCorrect: true },
      { text: "The proxy task is used only because predicting words is the final deployed objective.", isCorrect: false },
      { text: "The lecture presented the proxy task as irrelevant to embedding quality.", isCorrect: false },
    ],
    explanation:
      "The proxy task is chosen because solving it forces embeddings to capture useful semantic structure, even if the prediction model itself is not deployed."
  },

  // ============================================================
  //  Q54–Q70: 1 correct answer, 3 incorrect
  // ============================================================

  {
    id: "cme295-lect1-q54",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which single statement best describes why one-hot encodings are insufficient on their own for most NLP tasks?",
    options: [
      { text: "They do not encode semantic similarity between tokens.", isCorrect: true },
      { text: "They require a separate neural network for each token.", isCorrect: false },
      { text: "They cannot represent more than a few hundred tokens.", isCorrect: false },
      { text: "They were only defined for vision tasks, not text.", isCorrect: false },
    ],
    explanation:
      "One-hot vectors treat all tokens as equally distant and contain no information about meaning or similarity."
  },

  {
    id: "cme295-lect1-q55",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which single statement best describes why subword tokenization is popular for large language models?",
    options: [
      { text: "It strikes a balance between handling rare words and keeping sequence length manageable.", isCorrect: true },
      { text: "It guarantees that each token corresponds to exactly one character.", isCorrect: false },
      { text: "It ensures that all possible strings share the same tokenization.", isCorrect: false },
      { text: "It is only suitable for languages with no morphology.", isCorrect: false },
    ],
    explanation:
      "Subwords mitigate out-of-vocabulary issues and capture morphological patterns without exploding sequence lengths like pure character tokenization."
  },

  {
    id: "cme295-lect1-q56",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which single statement best captures a key limitation of static word embeddings like classic Word2vec?",
    options: [
      { text: "They assign the same vector to a word regardless of context.", isCorrect: true },
      { text: "They cannot be used as input to neural networks.", isCorrect: false },
      { text: "They require labeled data for every embedding update.", isCorrect: false },
      { text: "They always encode sentence position as part of the vector.", isCorrect: false },
    ],
    explanation:
      "Static embeddings do not disambiguate word senses; the word 'bank' gets one vector whether it refers to a river or a financial institution."
  },

  {
    id: "cme295-lect1-q57",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which single statement best explains why transformers scale well on modern hardware?",
    options: [
      { text: "Their self-attention and feedforward operations can be expressed as batched matrix multiplications.", isCorrect: true },
      { text: "They do not require any multiplications, only additions.", isCorrect: false },
      { text: "They process each token entirely independently of all others, with no interactions.", isCorrect: false },
      { text: "They avoid using GPUs by design.", isCorrect: false },
    ],
    explanation:
      "Transformers are built around matrix operations that map naturally to highly parallel hardware such as GPUs and TPUs."
  },

  {
    id: "cme295-lect1-q58",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which single statement best describes the main role of the encoder outputs in an encoder–decoder transformer?",
    options: [
      { text: "They provide context representations of the source sequence for the decoder to attend to.", isCorrect: true },
      { text: "They directly contain the final target-language tokens.", isCorrect: false },
      { text: "They are discarded before decoding begins.", isCorrect: false },
      { text: "They only store positional encodings and no semantic information.", isCorrect: false },
    ],
    explanation:
      "The encoder’s outputs are context-rich representations of the source text that the decoder uses via cross-attention when generating targets."
  },

  {
    id: "cme295-lect1-q59",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which single statement best describes why masked self-attention is required during training of autoregressive decoders?",
    options: [
      { text: "It enforces that each prediction depends only on past tokens, matching the generation process.", isCorrect: true },
      { text: "It prevents the model from using the encoder outputs.", isCorrect: false },
      { text: "It forces all attention weights to be equal.", isCorrect: false },
      { text: "It is only needed for evaluation, not training.", isCorrect: false },
    ],
    explanation:
      "Masked self-attention ensures the model cannot look ahead to future target tokens, preserving the autoregressive structure during training."
  },

  {
    id: "cme295-lect1-q60",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which single statement best summarizes the purpose of positional encodings?",
    options: [
      { text: "To provide information about token order that self-attention alone does not capture.", isCorrect: true },
      { text: "To randomly shuffle the positions of tokens during training.", isCorrect: false },
      { text: "To reduce the vocabulary size by merging similar tokens.", isCorrect: false },
      { text: "To implement the attention mechanism without queries or keys.", isCorrect: false },
    ],
    explanation:
      "Self-attention is invariant to permutations, so explicit positional information must be added to represent order."
  },

  {
    id: "cme295-lect1-q61",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which single statement best explains why label smoothing can improve generalization?",
    options: [
      { text: "It discourages the model from becoming overconfident on the training set.", isCorrect: true },
      { text: "It forces the model to memorize every training example exactly.", isCorrect: false },
      { text: "It ensures the model assigns zero probability to all incorrect classes.", isCorrect: false },
      { text: "It removes the need for a softmax layer in the output.", isCorrect: false },
    ],
    explanation:
      "By softening targets, label smoothing reduces overconfidence and can lead to better-calibrated and more robust models."
  },

  {
    id: "cme295-lect1-q62",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which single statement best describes the effect of vanishing gradients on recurrent neural network training?",
    options: [
      { text: "Parameters influencing early time steps are updated very weakly, making long-range learning difficult.", isCorrect: true },
      { text: "Gradients become extremely large and cause numerical instability.", isCorrect: false },
      { text: "The loss function becomes constant and cannot be differentiated.", isCorrect: false },
      { text: "Only the output layer is affected, while recurrent weights are unaffected.", isCorrect: false },
    ],
    explanation:
      "When gradients shrink through many time steps, early dependencies barely receive learning signal, impeding long-range credit assignment."
  },

  {
    id: "cme295-lect1-q63",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which single statement best summarizes the role of the course textbook and cheat sheet mentioned in the lecture?",
    options: [
      { text: "They provide condensed summaries and detailed explanations of the concepts taught in class.", isCorrect: true },
      { text: "They fully replace the need to attend lectures or watch recordings.", isCorrect: false },
      { text: "They cover unrelated topics not mentioned in the course.", isCorrect: false },
      { text: "They are used purely as exam formula sheets with no conceptual content.", isCorrect: false },
    ],
    explanation:
      "The textbook and cheat sheet were presented as complementary resources to reinforce and condense course concepts."
  },

  {
    id: "cme295-lect1-q64",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which single statement best captures why attention was described as central to the transformer paper 'Attention Is All You Need'?",
    options: [
      { text: "The architecture replaces recurrent and convolutional layers with attention as the primary interaction mechanism.", isCorrect: true },
      { text: "The architecture removes all linear transformations and uses only attention.", isCorrect: false },
      { text: "The paper shows that attention is unnecessary for machine translation.", isCorrect: false },
      { text: "The model uses attention only once at the output layer.", isCorrect: false },
    ],
    explanation:
      "The transformer eliminates recurrence and convolutions, relying instead on stacked attention and feedforward layers."
  },

  {
    id: "cme295-lect1-q65",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which single statement best describes the purpose of the softmax layer at the output of a language model?",
    options: [
      { text: "To convert raw logits into a normalized probability distribution over tokens.", isCorrect: true },
      { text: "To compute cosine similarity between tokens.", isCorrect: false },
      { text: "To remove all negative values from embeddings.", isCorrect: false },
      { text: "To determine attention weights between encoder and decoder states.", isCorrect: false },
    ],
    explanation:
      "Softmax exponentiates and normalizes logits so they sum to one, forming a probability distribution over the vocabulary."
  },

  {
    id: "cme295-lect1-q66",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which single statement best describes the main purpose of precision and recall in evaluating classifiers?",
    options: [
      { text: "To provide more informative evaluation than accuracy alone on imbalanced datasets.", isCorrect: true },
      { text: "To measure how fast the model runs on a GPU.", isCorrect: false },
      { text: "To compute the average sequence length of the inputs.", isCorrect: false },
      { text: "To estimate the vocabulary size of the tokenization scheme.", isCorrect: false },
    ],
    explanation:
      "Precision and recall quantify performance on positive cases in different ways and are especially important when classes are imbalanced."
  },

  {
    id: "cme295-lect1-q67",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which single statement best explains why attention helps with long-range dependencies compared to a pure recurrent neural network?",
    options: [
      { text: "It allows direct connections between distant tokens instead of relying on many recurrent steps.", isCorrect: true },
      { text: "It stores the entire sequence in a single scalar value.", isCorrect: false },
      { text: "It eliminates the need to backpropagate through time entirely.", isCorrect: false },
      { text: "It forces all attention weights to be identical for all pairs of tokens.", isCorrect: false },
    ],
    explanation:
      "Attention provides explicit paths between any pair of positions, reducing reliance on information flowing only through repeated recurrent updates."
  },

  {
    id: "cme295-lect1-q68",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which single statement best describes a core motivation for using dense embeddings instead of one-hot vectors?",
    options: [
      { text: "Dense embeddings can encode graded semantic similarity between tokens.", isCorrect: true },
      { text: "Dense embeddings guarantee that all words in a language are equidistant.", isCorrect: false },
      { text: "Dense embeddings completely remove the need for training data.", isCorrect: false },
      { text: "Dense embeddings ensure that vocabulary size has no impact on dimensionality.", isCorrect: false },
    ],
    explanation:
      "By learning dense vectors, models can capture nuanced similarity patterns between tokens beyond simple identity."
  },

  {
    id: "cme295-lect1-q69",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which single statement best characterizes the role of the feedforward sub-layer inside a transformer block?",
    options: [
      { text: "To apply non-linear transformations to each token’s representation independently after attention.", isCorrect: true },
      { text: "To compute the dot product between queries and keys.", isCorrect: false },
      { text: "To introduce positional information into the model.", isCorrect: false },
      { text: "To enforce masking in the decoder’s self-attention.", isCorrect: false },
    ],
    explanation:
      "The feedforward network enriches token-wise representations non-linearly after attention has mixed information across positions."
  },

  {
    id: "cme295-lect1-q70",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which single statement best summarizes the main conceptual focus of the lecture?",
    options: [
      { text: "Introducing the path from basic NLP concepts to the transformer architecture for sequence modeling.", isCorrect: true },
      { text: "Deriving training algorithms for reinforcement learning agents in robotics.", isCorrect: false },
      { text: "Designing convolutional neural networks for image classification.", isCorrect: false },
      { text: "Implementing databases for large-scale relational data.", isCorrect: false },
    ],
    explanation:
      "The lecture walks from NLP tasks, tokenization, embeddings, and recurrent models to attention and the transformer architecture."
  },
];
