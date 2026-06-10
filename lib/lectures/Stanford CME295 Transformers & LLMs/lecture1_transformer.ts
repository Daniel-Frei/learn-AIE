import { Question } from "../../quiz";

export const stanfordCME295Lecture1Questions: Question[] = [
  // ============================================================
  //  Q1–Q18: 4 correct answers (ALL TRUE)
  // ============================================================

  {
    id: "cme295-lect1-q01",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements describe typical goals of Natural Language Processing (NLP)?",
    options: [
      {
        text: "Automatically analyze and understand human language text.",
        isCorrect: true,
      },
      {
        text: "Transform raw text into numerical representations that models can process.",
        isCorrect: true,
      },
      {
        text: "Build systems that can classify, extract, or generate text.",
        isCorrect: true,
      },
      {
        text: "Support downstream applications such as chatbots, search, and translation.",
        isCorrect: true,
      },
    ],
    explanation:
      "Natural Language Processing (NLP) is the field concerned with making human language usable by computational models. That includes representing text numerically, analyzing it, and building applications such as classifiers, information extraction systems, chatbots, search, translation, and text generators.",
  },

  {
    id: "cme295-lect1-q02",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which problems are standard examples of text classification tasks?",
    options: [
      {
        text: "Sentiment analysis of movie or product reviews.",
        isCorrect: true,
      },
      {
        text: "Intent detection in a virtual assistant (for example, setting an alarm).",
        isCorrect: true,
      },
      {
        text: "Language identification (for example, detecting that a sentence is French).",
        isCorrect: true,
      },
      { text: "Topic classification of documents or posts.", isCorrect: true },
    ],
    explanation:
      "Text classification maps an input text to one or more predefined labels, and sentiment, intent, language, and topic are standard examples. The model is not being asked to generate an open-ended sentence; it is deciding which label best describes the input.",
  },

  {
    id: "cme295-lect1-q03",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which tasks fall under the multi-label or structured prediction category in NLP?",
    options: [
      {
        text: "Named Entity Recognition (identifying locations, people, dates, etc.).",
        isCorrect: true,
      },
      {
        text: "Part-of-speech tagging (labeling nouns, verbs, adjectives, and so on).",
        isCorrect: true,
      },
      {
        text: "Dependency or constituency parsing of syntactic structure.",
        isCorrect: true,
      },
      {
        text: "Assigning multiple categories or attributes to a single piece of text.",
        isCorrect: true,
      },
    ],
    explanation:
      "Multi-label and structured prediction tasks produce more than one output for a single input, often one label per token, span, syntactic relation, or attribute. Named Entity Recognition, part-of-speech tagging, parsing, and multi-attribute classification all require the model to preserve structure inside the input rather than collapse it to one whole-text label.",
  },

  {
    id: "cme295-lect1-q04",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which tasks are typical examples of text generation in NLP?",
    options: [
      {
        text: "Machine translation from one language to another.",
        isCorrect: true,
      },
      {
        text: "Question answering where the system writes a natural-language answer.",
        isCorrect: true,
      },
      { text: "Summarization of long documents or articles.", isCorrect: true },
      {
        text: "Free-form generation such as code, stories, or poems.",
        isCorrect: true,
      },
    ],
    explanation:
      "Text generation tasks produce new natural-language text, so the output length and wording can vary across valid answers. Translation, question answering, summarization, code, stories, and poems all fit this pattern because the model must construct a sequence rather than choose a fixed label.",
  },

  {
    id: "cme295-lect1-q05",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements about tokenization are correct?",
    options: [
      {
        text: "Tokenization splits raw text into smaller units called tokens.",
        isCorrect: true,
      },
      {
        text: "Tokens can be whole words, subwords, or individual characters.",
        isCorrect: true,
      },
      {
        text: "Tokenization is required so that text can be fed into neural models.",
        isCorrect: true,
      },
      {
        text: "Different tokenization schemes have different trade-offs in vocabulary size and sequence length.",
        isCorrect: true,
      },
    ],
    explanation:
      "Tokenization is the step that turns raw text into model-readable units such as words, subwords, or characters. The tokenizer choice affects vocabulary size, sequence length, out-of-vocabulary behavior, and the computational cost of later model layers.",
  },

  {
    id: "cme295-lect1-q06",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements describe advantages of subword tokenization?",
    options: [
      {
        text: "It allows common roots or stems (for example, 'bear' in 'bear' and 'bears') to be shared.",
        isCorrect: true,
      },
      {
        text: "It reduces the chance of truly unseen tokens by decomposing rare words.",
        isCorrect: true,
      },
      {
        text: "It balances between word-level semantics and character-level robustness.",
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

  {
    id: "cme295-lect1-q07",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the out-of-vocabulary (OOV) problem?",
    options: [
      {
        text: "Word-level tokenization can assign many unseen words to an unknown token.",
        isCorrect: true,
      },
      {
        text: "Subword tokenization reduces OOV rate by composing unseen words from known pieces.",
        isCorrect: true,
      },
      {
        text: "Character-level tokenization essentially eliminates OOV for alphabetic scripts.",
        isCorrect: true,
      },
      {
        text: "Handling OOV tokens is important because models need representations for words not seen during training.",
        isCorrect: true,
      },
    ],
    explanation:
      "The out-of-vocabulary problem appears when a tokenizer cannot represent a word seen at inference time as a known vocabulary item. Subword and character schemes reduce this problem by composing words from smaller units, while word-level tokenizers may collapse unseen words into an uninformative unknown token.",
  },

  {
    id: "cme295-lect1-q08",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements about one-hot encoding of tokens are correct?",
    options: [
      {
        text: "Each token is represented as a vector with a single 1 and the rest 0s.",
        isCorrect: true,
      },
      {
        text: "All one-hot vectors in the same vocabulary are orthogonal to each other.",
        isCorrect: true,
      },
      {
        text: "One-hot vectors do not encode any notion of semantic similarity.",
        isCorrect: true,
      },
      {
        text: "One-hot encodings are typically very high-dimensional when vocabularies are large.",
        isCorrect: true,
      },
    ],
    explanation:
      "A one-hot vector is useful as a unique token identifier, but it carries no learned notion of meaning. Because different one-hot vectors are orthogonal and vocabulary-sized, they are sparse, high-dimensional, and unable to show that related words should have similar representations.",
  },

  {
    id: "cme295-lect1-q09",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe word embeddings such as Word2vec?",
    options: [
      {
        text: "They map tokens to dense, low-dimensional vectors.",
        isCorrect: true,
      },
      {
        text: "They are learned from data so that similar words have similar vectors.",
        isCorrect: true,
      },
      {
        text: "They are trained via proxy tasks such as predicting context words or a target word.",
        isCorrect: true,
      },
      {
        text: "They support similarity measures such as cosine similarity to compare word meanings.",
        isCorrect: true,
      },
    ],
    explanation:
      "Word2vec-style embeddings replace sparse token identifiers with dense learned vectors. Their training objectives use context prediction as a proxy task, and the resulting geometry can make semantically related words closer under measures such as cosine similarity.",
  },

  {
    id: "cme295-lect1-q10",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about the Continuous Bag of Words (CBOW) and Skip-Gram models in Word2vec are correct?",
    options: [
      {
        text: "Both are trained on large unlabeled text corpora.",
        isCorrect: true,
      },
      {
        text: "CBOW predicts a target word from its surrounding context words.",
        isCorrect: true,
      },
      {
        text: "Skip-Gram predicts surrounding context words from a central target word.",
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

  {
    id: "cme295-lect1-q11",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements describe Recurrent Neural Networks (RNNs) for sequence modeling?",
    options: [
      {
        text: "They process sequences step by step, maintaining a hidden state.",
        isCorrect: true,
      },
      {
        text: "The hidden state summarizes information from previous time steps.",
        isCorrect: true,
      },
      {
        text: "They can be used for classification, tagging, and generation tasks.",
        isCorrect: true,
      },
      {
        text: "They were introduced long before the transformer architecture.",
        isCorrect: true,
      },
    ],
    explanation:
      "A Recurrent Neural Network (RNN) processes a sequence one step at a time while carrying forward a hidden state. That recurrent state lets earlier tokens influence later predictions, which made RNNs useful for classification, tagging, and generation before transformer architectures became dominant.",
  },

  {
    id: "cme295-lect1-q12",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements describe the Long Short-Term Memory (LSTM) architecture?",
    options: [
      {
        text: "It is a specific type of recurrent neural network.",
        isCorrect: true,
      },
      {
        text: "It introduces a cell state in addition to the hidden state.",
        isCorrect: true,
      },
      {
        text: "It aims to better preserve information over longer sequences.",
        isCorrect: true,
      },
      {
        text: "It was designed to mitigate the vanishing gradient problem in standard recurrent neural networks.",
        isCorrect: true,
      },
    ],
    explanation:
      "A Long Short-Term Memory (LSTM) network is a gated recurrent architecture with both hidden-state and cell-state pathways. Those gates help preserve important information over longer spans and reduce the vanishing-gradient problems that make vanilla RNNs struggle with distant dependencies.",
  },

  {
    id: "cme295-lect1-q13",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements describe typical issues with standard Recurrent Neural Networks (RNNs)?",
    options: [
      {
        text: "They can struggle to capture very long-range dependencies.",
        isCorrect: true,
      },
      {
        text: "Backpropagation through many time steps can cause gradients to vanish or explode.",
        isCorrect: true,
      },
      {
        text: "Their sequential nature leads to slow training on long sequences.",
        isCorrect: true,
      },
      {
        text: "The entire sentence meaning can be bottlenecked into a single hidden state.",
        isCorrect: true,
      },
    ],
    explanation:
      "Standard RNNs can have difficulty with long sequences because information and gradients must pass through many recurrent steps. This creates long-range dependency limits, vanishing or exploding gradients, slow sequential training, and a bottleneck when too much meaning is compressed into one hidden state.",
  },

  {
    id: "cme295-lect1-q14",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the motivation behind attention mechanisms?",
    options: [
      {
        text: "To provide direct connections between a prediction and relevant positions in the input sequence.",
        isCorrect: true,
      },
      {
        text: "To mitigate the difficulty of remembering faraway tokens in long sequences.",
        isCorrect: true,
      },
      {
        text: "To let the model weight different input tokens based on their relevance to the current prediction.",
        isCorrect: true,
      },
      {
        text: "To move away from strictly sequential dependence on a single hidden state.",
        isCorrect: true,
      },
    ],
    explanation:
      "Attention was introduced to let a model connect directly to relevant positions instead of relying only on a single recurrent summary. By weighting input tokens according to relevance, attention reduces the long-distance memory bottleneck and gives the prediction a more targeted context.",
  },

  {
    id: "cme295-lect1-q15",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements describe self-attention in the transformer architecture?",
    options: [
      {
        text: "Each token representation is updated by looking at all tokens in the sequence.",
        isCorrect: true,
      },
      {
        text: "Attention weights indicate how much each token attends to every other token.",
        isCorrect: true,
      },
      {
        text: "Self-attention can be implemented efficiently with matrix multiplications.",
        isCorrect: true,
      },
      {
        text: "Self-attention is a central building block of the transformer encoder and decoder.",
        isCorrect: true,
      },
    ],
    explanation:
      "Self-attention updates each token representation by comparing it with other tokens in the same sequence. The attention weights define how strongly positions interact, and the resulting matrix operations are a central reason transformers train efficiently on modern parallel hardware.",
  },

  {
    id: "cme295-lect1-q16",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe queries, keys, and values in attention?",
    options: [
      {
        text: "Queries, keys, and values are learned linear projections of the same input embeddings.",
        isCorrect: true,
      },
      {
        text: "Attention weights are computed by comparing queries with keys.",
        isCorrect: true,
      },
      {
        text: "Values are combined with attention weights to form the final context vectors.",
        isCorrect: true,
      },
      {
        text: "All three (queries, keys, values) are trainable via gradient descent along with the rest of the model.",
        isCorrect: true,
      },
    ],
    explanation:
      "Queries, keys, and values are learned projections used to turn token representations into an attention lookup. Query-key comparisons produce attention weights, and those weights mix the value vectors to create context-aware representations, with all projection matrices learned by gradient descent.",
  },

  {
    id: "cme295-lect1-q17",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements correctly describe multi-head attention?",
    options: [
      {
        text: "It runs several independent attention mechanisms (heads) in parallel.",
        isCorrect: true,
      },
      {
        text: "Each head uses its own projection matrices for queries, keys, and values.",
        isCorrect: true,
      },
      {
        text: "The outputs of all heads are concatenated and then linearly projected back to the model dimension.",
        isCorrect: true,
      },
      {
        text: "Multiple heads allow the model to capture different types of relations between tokens.",
        isCorrect: true,
      },
    ],
    explanation:
      "Multi-head attention runs several attention projections in parallel so the model can learn different relation patterns between tokens. The head outputs are concatenated and projected back to the model dimension, giving the block more representational capacity than a single attention map.",
  },

  {
    id: "cme295-lect1-q18",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements describe key structural elements of the original transformer architecture?",
    options: [
      {
        text: "It consists of a stack of encoders and a stack of decoders.",
        isCorrect: true,
      },
      {
        text: "Encoder self-attention is fully bidirectional over the input sequence.",
        isCorrect: true,
      },
      {
        text: "Decoder self-attention is masked to prevent attending to future target tokens.",
        isCorrect: true,
      },
      {
        text: "Decoder layers include a cross-attention block that attends to encoder outputs.",
        isCorrect: true,
      },
    ],
    explanation:
      "The original encoder-decoder transformer stacks encoder blocks and decoder blocks. Encoder self-attention can use the full source sequence, decoder self-attention is masked to preserve autoregressive prediction, and decoder cross-attention lets target-side states attend to source-side representations.",
  },

  // ============================================================
  //  Q19–Q36: 3 correct answers, 1 incorrect
  // ============================================================

  {
    id: "cme295-lect1-q20",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about evaluation metrics for binary classification are correct?",
    options: [
      {
        text: "Accuracy can be misleading when classes are highly imbalanced.",
        isCorrect: true,
      },
      {
        text: "Precision measures, among predicted positives, how many are actually positive.",
        isCorrect: true,
      },
      {
        text: "Recall measures, among true positives, how many were correctly predicted positive.",
        isCorrect: true,
      },
      {
        text: "The F1 score is the arithmetic mean of precision and recall.",
        isCorrect: false,
      },
    ],
    explanation:
      "Accuracy can look high on an imbalanced dataset even when the model ignores a rare but important class. Precision asks how many predicted positives are truly positive, recall asks how many true positives were found, and F1 combines precision and recall with a harmonic mean rather than an arithmetic mean.",
  },

  {
    id: "cme295-lect1-q21",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about BLEU and ROUGE are correct?",
    options: [
      {
        text: "They are reference-based metrics comparing model outputs to human-written texts.",
        isCorrect: true,
      },
      {
        text: "They were widely used for evaluating machine translation and summarization.",
        isCorrect: true,
      },
      {
        text: "They rely on overlapping n-grams between prediction and reference.",
        isCorrect: true,
      },
      {
        text: "They can be computed without any reference outputs.",
        isCorrect: false,
      },
    ],
    explanation:
      "BLEU and ROUGE are reference-based metrics that compare a generated output with human-written reference text, often through overlapping n-grams or related matching rules. They were widely used for translation and summarization, but they require reference outputs and therefore are not reference-free evaluation methods.",
  },

  {
    id: "cme295-lect1-q22",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements describe perplexity in the context of language models?",
    options: [
      {
        text: "Perplexity is derived from the probabilities that the model assigns to the correct tokens.",
        isCorrect: true,
      },
      {
        text: "Lower perplexity indicates the model is less surprised by the data.",
        isCorrect: true,
      },
      {
        text: "Perplexity can serve as a training-time signal even without reference translations.",
        isCorrect: true,
      },
      {
        text: "Higher perplexity always implies better generative creativity.",
        isCorrect: false,
      },
    ],
    explanation:
      "Perplexity is derived from the probability a language model assigns to the observed tokens. Lower perplexity means the model is less surprised by the data under its probability distribution, but it is not a direct measure of creativity or human preference.",
  },

  {
    id: "cme295-lect1-q23",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements about character-level tokenization are correct?",
    options: [
      {
        text: "It tends to produce much longer token sequences than word-level tokenization.",
        isCorrect: true,
      },
      {
        text: "It is robust to misspellings and unusual word forms.",
        isCorrect: true,
      },
      {
        text: "It avoids out-of-vocabulary issues for alphabetic writing systems.",
        isCorrect: true,
      },
      {
        text: "It guarantees that sequences are shorter than with subword tokenization.",
        isCorrect: false,
      },
    ],
    explanation:
      "Character-level tokenization can represent unusual spellings and avoids many out-of-vocabulary failures because characters are reusable building blocks. The cost is that sequences become much longer than word or subword sequences, which increases the burden on the model.",
  },

  {
    id: "cme295-lect1-q24",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe cosine similarity for comparing token representations?",
    options: [
      {
        text: "It depends on the angle between two vectors rather than their norms.",
        isCorrect: true,
      },
      {
        text: "It is often used to measure semantic similarity between embeddings.",
        isCorrect: true,
      },
      {
        text: "With one-hot encodings, cosine similarity between any two different tokens is zero.",
        isCorrect: true,
      },
      {
        text: "Cosine similarity directly measures how often two words co-occur in the training corpus.",
        isCorrect: false,
      },
    ],
    explanation:
      "Cosine similarity compares the angle between vectors rather than their raw magnitudes, making it useful for comparing embedding directions. It can reflect semantic similarity in learned embeddings, but for one-hot vectors different tokens are orthogonal, and the value is not itself a direct co-occurrence count.",
  },

  {
    id: "cme295-lect1-q25",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about the training objective of Word2vec-style models are correct?",
    options: [
      {
        text: "The proxy task might be predicting the next word given previous words.",
        isCorrect: true,
      },
      {
        text: "The learned parameters include embeddings that can be reused for downstream tasks.",
        isCorrect: true,
      },
      {
        text: "The loss is often based on cross-entropy between predicted and true distributions over the vocabulary.",
        isCorrect: true,
      },
      {
        text: "The final goal is to keep using the full predictive model, not its learned embeddings.",
        isCorrect: false,
      },
    ],
    explanation:
      "Word2vec-style models use a predictive task, such as predicting a target from context or context from a target, to create learning signal from unlabeled text. The deployed artifact is often the embedding table, while the temporary prediction head is just a way to train those reusable representations.",
  },

  {
    id: "cme295-lect1-q26",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements describe the vanishing gradient problem in recurrent neural networks?",
    options: [
      {
        text: "Gradients are backpropagated through many time steps in sequence models.",
        isCorrect: true,
      },
      {
        text: "Repeated multiplication by Jacobians with eigenvalues less than one can shrink gradients toward zero.",
        isCorrect: true,
      },
      {
        text: "Very small gradients make it difficult to learn long-range dependencies.",
        isCorrect: true,
      },
      {
        text: "The problem arises because gradients are added, not multiplied, across time.",
        isCorrect: false,
      },
    ],
    explanation:
      "Backpropagation through time repeatedly multiplies gradients through the recurrent transition. If the relevant Jacobian factors shrink, the gradient reaching early time steps can become tiny, so the model learns long-range dependencies weakly.",
  },

  {
    id: "cme295-lect1-q27",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about the history of models for language are correct, as described in the lecture?",
    options: [
      {
        text: "Recurrent neural network ideas date back to the 1980s.",
        isCorrect: true,
      },
      {
        text: "Long Short-Term Memory networks were proposed in the 1990s.",
        isCorrect: true,
      },
      {
        text: "Word2vec popularized word embeddings in the early 2010s.",
        isCorrect: true,
      },
      {
        text: "Transformers were first introduced in the early 1990s.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture placed transformers in a longer sequence-modeling history rather than presenting them as the first language models. Recurrent ideas go back decades, LSTMs appeared in the 1990s, Word2vec popularized embeddings in the early 2010s, and transformers arrived later with the 2017 architecture.",
  },

  {
    id: "cme295-lect1-q28",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe the encoder in a transformer?",
    options: [
      {
        text: "It processes the input sequence in a fully parallel way using self-attention.",
        isCorrect: true,
      },
      {
        text: "It produces context-aware embeddings for each input position.",
        isCorrect: true,
      },
      {
        text: "Its self-attention is not masked and can attend to all positions.",
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

  {
    id: "cme295-lect1-q29",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the decoder in a transformer for sequence-to-sequence tasks?",
    options: [
      {
        text: "It uses masked self-attention over previously generated target tokens.",
        isCorrect: true,
      },
      {
        text: "It uses cross-attention to attend to the encoder’s output representations.",
        isCorrect: true,
      },
      {
        text: "It predicts the next target token via a linear layer plus softmax over the vocabulary.",
        isCorrect: true,
      },
      {
        text: "It can freely attend to future target tokens during training.",
        isCorrect: false,
      },
    ],
    explanation:
      "A transformer decoder generates target tokens while preserving the left-to-right information constraint. Masked self-attention prevents access to future target tokens, cross-attention reads the encoded source, and the final projection plus softmax turns decoder states into token probabilities.",
  },

  {
    id: "cme295-lect1-q30",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about the attention weight computation QKᵀ / √dₖ followed by softmax are correct?",
    options: [
      {
        text: "The dot products between queries and keys measure compatibility between positions.",
        isCorrect: true,
      },
      {
        text: "Dividing by the square root of the key dimension keeps dot products numerically well-scaled.",
        isCorrect: true,
      },
      {
        text: "The softmax converts raw scores into a probability distribution over keys for each query.",
        isCorrect: true,
      },
      {
        text: "Removing the scaling factor always improves training stability.",
        isCorrect: false,
      },
    ],
    explanation:
      "The product (QK^T) scores how compatible each query position is with each key position. Dividing by (sqrt{d_k}) keeps those dot products numerically stable, and softmax turns the scaled scores into attention weights rather than making training better by removing the scaling.",
  },

  {
    id: "cme295-lect1-q31",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about positional encodings in transformers are correct?",
    options: [
      {
        text: "They inject information about token positions into the model.",
        isCorrect: true,
      },
      {
        text: "In the original paper, they are implemented as sinusoidal functions added to embeddings.",
        isCorrect: true,
      },
      {
        text: "They are combined elementwise with token embeddings before entering the encoder.",
        isCorrect: true,
      },
      {
        text: "They are unnecessary because self-attention alone encodes order.",
        isCorrect: false,
      },
    ],
    explanation:
      "Self-attention by itself does not know token order because pairwise content comparisons are permutation-invariant without position information. Positional encodings add order signals to token embeddings, and the original transformer used sinusoidal functions added elementwise before the encoder stack.",
  },

  {
    id: "cme295-lect1-q32",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about position-wise feedforward network (FFN) layers inside transformer blocks are correct?",
    options: [
      {
        text: "They apply the same nonlinear network at each token position.",
        isCorrect: true,
      },
      {
        text: "They typically expand the hidden dimension before projecting back down.",
        isCorrect: true,
      },
      {
        text: "They add nonlinear capacity after attention has mixed token information.",
        isCorrect: true,
      },
      {
        text: "They are responsible for computing attention weights between positions.",
        isCorrect: false,
      },
    ],
    explanation:
      "The transformer feedforward network is position-wise: after attention has mixed information across tokens, the same learned nonlinear transformation is applied independently at each position. In the original design this is commonly described as two linear layers with an activation between them, usually expanding the hidden dimension before projecting back down.",
  },

  {
    id: "cme295-lect1-q33",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "During next-token training, a model compares predicted probabilities with a target distribution over the vocabulary. Which statements about label smoothing are correct?",
    options: [
      {
        text: "It replaces a hard one-hot target distribution with a slightly softened version.",
        isCorrect: true,
      },
      {
        text: "It acknowledges that multiple next tokens can be reasonable in natural language.",
        isCorrect: true,
      },
      {
        text: "It can reduce overconfidence of the model on the training data.",
        isCorrect: true,
      },
      {
        text: "It sets the target probability of the correct class exactly to one.",
        isCorrect: false,
      },
    ],
    explanation:
      "Label smoothing changes the target distribution used in the loss from a hard one-hot vector to a slightly softened vector. That leaves most probability on the intended token while assigning a small amount to alternatives, which can reduce overconfidence in settings where multiple next tokens may be plausible.",
  },

  {
    id: "cme295-lect1-q34",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about masked self-attention in the decoder are correct?",
    options: [
      {
        text: "The attention mask prevents attending to positions to the right of the current token.",
        isCorrect: true,
      },
      {
        text: "The mask enforces an autoregressive factorization during training.",
        isCorrect: true,
      },
      {
        text: "The mask ensures that predictions do not use future target tokens as information.",
        isCorrect: true,
      },
      {
        text: "The mask is also applied in the encoder’s self-attention.",
        isCorrect: false,
      },
    ],
    explanation:
      "Masked self-attention in the decoder blocks attention to positions to the right of the current target token. This enforces the same autoregressive factorization used during generation, while the encoder remains unmasked because it is allowed to read the whole source sequence.",
  },

  {
    id: "cme295-lect1-q35",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about using RNNs versus transformers for long sequences are correct?",
    options: [
      {
        text: "Transformers process all positions in parallel within a layer, while RNNs are strictly sequential.",
        isCorrect: true,
      },
      {
        text: "Transformers use attention to connect distant positions directly.",
        isCorrect: true,
      },
      {
        text: "RNNs rely on repeatedly updating a single hidden state across time steps.",
        isCorrect: true,
      },
      {
        text: "Standard RNNs scale better than transformers as sequence length grows.",
        isCorrect: false,
      },
    ],
    explanation:
      "Transformers and RNNs differ sharply in how information moves through a sequence. Transformers can process positions in parallel within a layer and connect distant tokens directly with attention, while RNNs update one hidden state step by step and therefore have weaker parallelism for long sequences.",
  },

  {
    id: "cme295-lect1-q36",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about the training loop for a transformer-based machine translation model are correct?",
    options: [
      {
        text: "The source sentence is encoded via the encoder into context-aware embeddings.",
        isCorrect: true,
      },
      {
        text: "The decoder processes the target sentence with masked self-attention during training.",
        isCorrect: true,
      },
      {
        text: "The model predicts each next target token conditioned on previous target tokens and the encoded source.",
        isCorrect: true,
      },
      {
        text: "The model is trained by directly minimizing BLEU scores.",
        isCorrect: false,
      },
    ],
    explanation:
      "For machine translation, the encoder first turns the source sentence into context-aware embeddings. During training the decoder uses masked self-attention over target-side prefixes and cross-attention to the source representations, while the optimization target is usually next-token cross-entropy rather than BLEU itself.",
  },

  // ============================================================
  //  Q37–Q53: 2 correct answers, 2 incorrect
  // ============================================================

  {
    id: "cme295-lect1-q37",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the three high-level categories of NLP tasks discussed in the lecture?",
    options: [
      {
        text: "Classification tasks map text to a single label.",
        isCorrect: true,
      },
      {
        text: "Generation tasks map text to variable-length text outputs.",
        isCorrect: true,
      },
      {
        text: "Segmentation tasks were presented as the third main category.",
        isCorrect: false,
      },
      {
        text: "Clustering tasks were presented as the third main category.",
        isCorrect: false,
      },
    ],
    explanation:
      "The three high-level NLP task groups discussed were classification, multi-label or structured prediction, and generation. Classification maps text to a label and generation maps text to variable-length text, while clustering and segmentation were not presented as the third top-level bucket here.",
  },

  {
    id: "cme295-lect1-q38",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe multi-label or multi-output tasks such as Named Entity Recognition?",
    options: [
      {
        text: "They assign labels to multiple parts of a single input sequence.",
        isCorrect: true,
      },
      {
        text: "They can evaluate predictions at the token or entity level.",
        isCorrect: true,
      },
      {
        text: "They were described as mapping a sentence to exactly one label.",
        isCorrect: false,
      },
      {
        text: "They ignore spans of text and only label entire documents.",
        isCorrect: false,
      },
    ],
    explanation:
      "Named Entity Recognition and related tasks produce labels for multiple positions or spans inside one input sequence. That makes them different from single-label document classification and from methods that ignore spans or only label entire documents.",
  },

  {
    id: "cme295-lect1-q39",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about the trade-offs between word-level and subword-level tokenization are correct?",
    options: [
      {
        text: "Word-level tokenization typically leads to smaller sequence lengths than subword tokenization.",
        isCorrect: true,
      },
      {
        text: "Subword tokenization reduces the risk of out-of-vocabulary tokens compared to word-level tokenization.",
        isCorrect: true,
      },
      {
        text: "Word-level tokenization is more robust to misspellings than subword tokenization.",
        isCorrect: false,
      },
      {
        text: "Subword tokenization was described as ignoring roots and morphological structure.",
        isCorrect: false,
      },
    ],
    explanation:
      "Word-level tokenization tends to keep sequences short but can create many out-of-vocabulary problems because related word forms become separate vocabulary entries. Subword tokenization uses reusable pieces to handle rare words and morphology better, at the cost of producing longer sequences than pure word tokenization.",
  },

  {
    id: "cme295-lect1-q40",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about vocabulary size selection are correct according to the lecture?",
    options: [
      {
        text: "Monolingual tokenizers often target tens of thousands of subword tokens.",
        isCorrect: true,
      },
      {
        text: "Multilingual or code-aware tokenizers can reach hundreds of thousands of tokens.",
        isCorrect: true,
      },
      {
        text: "The ideal vocabulary size is determined purely by a closed-form mathematical formula.",
        isCorrect: false,
      },
      {
        text: "Vocabulary size has no connection to computational cost.",
        isCorrect: false,
      },
    ],
    explanation:
      "Vocabulary size is an engineering tradeoff rather than a closed-form optimum. Monolingual subword vocabularies often use tens of thousands of tokens, while multilingual or code-heavy tokenizers can be much larger, and the choice affects memory, compute, and sequence length.",
  },

  {
    id: "cme295-lect1-q41",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about sentence-level representations from static word embeddings are correct?",
    options: [
      {
        text: "Averaging word embeddings is a simple way to create a sentence vector.",
        isCorrect: true,
      },
      {
        text: "Averaging word embeddings discards word order information.",
        isCorrect: true,
      },
      {
        text: "Averaging static embeddings fully captures contextual meaning of each word.",
        isCorrect: false,
      },
      {
        text: "Static embeddings were presented as inherently context-dependent in this lecture.",
        isCorrect: false,
      },
    ],
    explanation:
      "Averaging static word embeddings is a simple bag-of-words sentence representation. It can give a crude sentence vector, but it discards word order and cannot make one word representation change with context, so it misses many distinctions that contextual models can capture.",
  },

  {
    id: "cme295-lect1-q42",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about Long Short-Term Memory (LSTM) networks are correct in comparison to vanilla recurrent neural networks?",
    options: [
      {
        text: "LSTMs track both a hidden state and a separate cell state.",
        isCorrect: true,
      },
      {
        text: "LSTMs were designed to better preserve important information over longer time spans.",
        isCorrect: true,
      },
      {
        text: "LSTMs remove the sequential nature of processing sequences.",
        isCorrect: false,
      },
      {
        text: "LSTMs completely eliminate vanishing gradients in practice.",
        isCorrect: false,
      },
    ],
    explanation:
      "LSTMs improve on vanilla RNNs by adding gates and a separate cell state that help information persist over longer spans. They still process tokens sequentially and do not magically remove all gradient problems, but they reduce a major weakness of plain recurrent networks.",
  },

  {
    id: "cme295-lect1-q43",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about the computational characteristics of transformers versus recurrent neural networks are correct?",
    options: [
      {
        text: "Transformers use attention to enable parallel computation across sequence positions within each layer.",
        isCorrect: true,
      },
      {
        text: "Recurrent neural networks must process tokens one time step after another.",
        isCorrect: true,
      },
      {
        text: "Transformers were presented as strictly slower than recurrent neural networks for long sequences.",
        isCorrect: false,
      },
      {
        text: "Recurrent neural networks were presented as inherently more parallelizable than transformers.",
        isCorrect: false,
      },
    ],
    explanation:
      "Transformers are more parallel-friendly than RNNs because attention can compare sequence positions within a layer at the same time. RNNs must update their hidden state in time order, so later states depend on earlier computations being finished.",
  },

  {
    id: "cme295-lect1-q44",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about cross-attention in the transformer decoder are correct?",
    options: [
      {
        text: "Decoder queries attend to keys and values derived from encoder outputs.",
        isCorrect: true,
      },
      {
        text: "Cross-attention lets the decoder condition its predictions on the encoded source sequence.",
        isCorrect: true,
      },
      {
        text: "Cross-attention was described as operating only within the decoder without looking at encoder states.",
        isCorrect: false,
      },
      {
        text: "Cross-attention is responsible for adding positional information to encoder embeddings.",
        isCorrect: false,
      },
    ],
    explanation:
      "Cross-attention is the decoder mechanism that connects target-side generation to source-side encoder outputs. Decoder states supply the queries, encoder outputs supply keys and values, and positional encodings are separate inputs added earlier rather than the purpose of cross-attention.",
  },

  {
    id: "cme295-lect1-q45",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe how probabilities over the vocabulary are produced in the transformer decoder?",
    options: [
      {
        text: "The final decoder hidden state is passed through a linear layer.",
        isCorrect: true,
      },
      {
        text: "A softmax layer converts logits into a probability distribution over tokens.",
        isCorrect: true,
      },
      {
        text: "Probabilities are obtained by applying cosine similarity directly to embeddings without any linear layer.",
        isCorrect: false,
      },
      {
        text: "Probabilities come from attention weights directly, without any additional transformation.",
        isCorrect: false,
      },
    ],
    explanation:
      "The decoder does not use attention weights directly as vocabulary probabilities. Its hidden state is first mapped through a learned linear projection into logits over the vocabulary, and softmax then normalizes those logits into a probability distribution.",
  },

  {
    id: "cme295-lect1-q46",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about label smoothing’s effect on the target distribution are correct?",
    options: [
      {
        text: "It reduces the target probability assigned to the correct class slightly.",
        isCorrect: true,
      },
      {
        text: "It spreads a small amount of probability mass over the incorrect classes.",
        isCorrect: true,
      },
      {
        text: "It makes the target distribution exactly uniform over all classes.",
        isCorrect: false,
      },
      {
        text: "It increases the loss when the model assigns moderate probability to the correct class.",
        isCorrect: false,
      },
    ],
    explanation:
      "Label smoothing reduces the target probability assigned to the nominally correct class and spreads a small amount of probability mass over other classes. It softens the training target rather than making the target uniform or replacing the model probability computation.",
  },

  {
    id: "cme295-lect1-q47",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about the use of special tokens such as BOS (begin-of-sequence) and EOS (end-of-sequence) are correct?",
    options: [
      {
        text: "BOS indicates the start of a sequence for the decoder to begin generation.",
        isCorrect: true,
      },
      { text: "EOS marks where generation should stop.", isCorrect: true },
      {
        text: "These tokens are unnecessary when training autoregressive language models.",
        isCorrect: false,
      },
      {
        text: "These tokens are used only for character-level models and not for word-based models.",
        isCorrect: false,
      },
    ],
    explanation:
      "Begin-of-sequence and end-of-sequence tokens provide explicit boundaries for sequence models. BOS gives the decoder a starting input for generation, and EOS gives the model a learned way to stop, regardless of whether the tokenizer uses words, subwords, or characters.",
  },

  {
    id: "cme295-lect1-q48",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about computing self-attention with matrices are correct?",
    options: [
      {
        text: "The query, key, and value matrices each contain one row per token position.",
        isCorrect: true,
      },
      {
        text: "The product QKᵀ results in a matrix of attention scores between all pairs of positions.",
        isCorrect: true,
      },
      {
        text: "The product QKᵀ has the same shape as the original embedding matrix.",
        isCorrect: false,
      },
      {
        text: "Multiplying attention weights by V produces token-wise scalar scores instead of vectors.",
        isCorrect: false,
      },
    ],
    explanation:
      "In matrix self-attention, the query, key, and value matrices contain one representation per token position. (QK^T) produces an (n\times n) matrix of pairwise attention scores, and multiplying normalized weights by (V) produces context vectors rather than scalar labels.",
  },

  {
    id: "cme295-lect1-q49",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe how Named Entity Recognition (NER) differs from ordinary sentence-level classification?",
    options: [
      {
        text: "NER predicts labels for specific tokens or spans inside the input text.",
        isCorrect: true,
      },
      {
        text: "NER can be evaluated by entity type, span match, or token-level labeling accuracy.",
        isCorrect: true,
      },
      {
        text: "NER assigns exactly one label to the entire document and ignores token positions.",
        isCorrect: false,
      },
      {
        text: "NER is a text generation task whose output length is unconstrained free-form prose.",
        isCorrect: false,
      },
    ],
    explanation:
      "Named Entity Recognition is a structured prediction task because the model labels pieces of the input, such as names, locations, dates, or organizations. Sentence-level classification instead maps the whole text to one label, while open-ended generation produces new text rather than aligned span labels.",
  },

  {
    id: "cme295-lect1-q50",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about training sequence models on paired data for machine translation are correct?",
    options: [
      {
        text: "Datasets such as WMT provide paired sentences in different languages.",
        isCorrect: true,
      },
      {
        text: "Paired datasets are more expensive to collect than single-language corpora.",
        isCorrect: true,
      },
      {
        text: "The lecture described translation training as using only monolingual corpora.",
        isCorrect: false,
      },
      {
        text: "The lecture stated that no labels are needed for supervised translation training.",
        isCorrect: false,
      },
    ],
    explanation:
      "Supervised machine translation relies on paired examples, such as aligned sentences from datasets like WMT. Those paired references are more expensive than raw monolingual text, and training is not just an unsupervised process with no labels.",
  },

  {
    id: "cme295-lect1-q51",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about the relationship between softmax and label smoothing in training are correct?",
    options: [
      {
        text: "Softmax is applied to model logits to produce a probability distribution.",
        isCorrect: true,
      },
      {
        text: "Label smoothing modifies the target distribution used in the loss, not the softmax operation itself.",
        isCorrect: true,
      },
      {
        text: "Label smoothing was described as replacing softmax with a different activation function.",
        isCorrect: false,
      },
      {
        text: "Softmax alone was described as sufficient to express uncertainty in the target distribution.",
        isCorrect: false,
      },
    ],
    explanation:
      "Softmax and label smoothing play different roles in training. Softmax converts model logits into predicted probabilities, while label smoothing changes the target distribution used in the loss so the model is not trained against an infinitely sharp one-hot target.",
  },

  {
    id: "cme295-lect1-q52",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about evaluation of generation tasks such as translation are correct?",
    options: [
      {
        text: "There are many valid outputs for the same input, which complicates evaluation.",
        isCorrect: true,
      },
      {
        text: "Traditional metrics like BLEU and ROUGE compare against reference outputs.",
        isCorrect: true,
      },
      {
        text: "Collecting high-quality reference outputs is time-consuming and expensive.",
        isCorrect: true,
      },
      {
        text: "The lecture claimed that reference-based metrics are unnecessary for all generative tasks.",
        isCorrect: false,
      },
    ],
    explanation:
      "Generated text can have many valid outputs for the same input, so evaluation is harder than checking one exact label. BLEU and ROUGE compare against references and those references are costly to collect, but that does not mean reference-based metrics are useless for every generative task.",
  },

  {
    id: "cme295-lect1-q53",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about the proxy nature of tasks used to learn embeddings (such as in Word2vec) are correct?",
    options: [
      {
        text: "The immediate training objective is predicting context or target words in text.",
        isCorrect: true,
      },
      {
        text: "The underlying goal is to obtain meaningful embeddings for downstream tasks.",
        isCorrect: true,
      },
      {
        text: "The proxy task is used only because predicting words is the final deployed objective.",
        isCorrect: false,
      },
      {
        text: "The lecture presented the proxy task as irrelevant to embedding quality.",
        isCorrect: false,
      },
    ],
    explanation:
      "Embedding training often uses a proxy task because predicting words from context forces the vectors to encode useful semantic and syntactic regularities. The prediction task is not irrelevant, but the main product is frequently the learned representation rather than the exact auxiliary model used to train it.",
  },

  // ============================================================
  //  Q54–Q70: 1 correct answer, 3 incorrect
  // ============================================================

  {
    id: "cme295-lect1-q54",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which single statement best describes why one-hot encodings are insufficient on their own for most NLP tasks?",
    options: [
      {
        text: "They do not encode semantic similarity between tokens.",
        isCorrect: true,
      },
      {
        text: "They require a separate neural network for each token.",
        isCorrect: false,
      },
      {
        text: "They cannot represent more than a few hundred tokens.",
        isCorrect: false,
      },
      {
        text: "They were only defined for vision tasks, not text.",
        isCorrect: false,
      },
    ],
    explanation:
      "One-hot encodings identify tokens but do not represent semantic similarity. Dense learned embeddings address that limitation by allowing related tokens to occupy nearby regions of vector space instead of treating every pair of distinct tokens as equally unrelated.",
  },

  {
    id: "cme295-lect1-q55",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which single statement best describes why subword tokenization is popular for large language models?",
    options: [
      {
        text: "It strikes a balance between handling rare words and keeping sequence length manageable.",
        isCorrect: true,
      },
      {
        text: "It guarantees that each token corresponds to exactly one character.",
        isCorrect: false,
      },
      {
        text: "It ensures that all possible strings share the same tokenization.",
        isCorrect: false,
      },
      {
        text: "It is only suitable for languages with no morphology.",
        isCorrect: false,
      },
    ],
    explanation:
      "Subword tokenization is popular because it handles rare and morphologically varied words without making every character a separate modeling step. It does not guarantee one-character tokens or one universal segmentation, but it gives a practical balance between robustness and sequence length.",
  },

  {
    id: "cme295-lect1-q56",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which single statement best captures a key limitation of static word embeddings like classic Word2vec?",
    options: [
      {
        text: "They assign the same vector to a word regardless of context.",
        isCorrect: true,
      },
      {
        text: "They cannot be used as input to neural networks.",
        isCorrect: false,
      },
      {
        text: "They require labeled data for every embedding update.",
        isCorrect: false,
      },
      {
        text: "They always encode sentence position as part of the vector.",
        isCorrect: false,
      },
    ],
    explanation:
      "Classic Word2vec-style embeddings are static, meaning a token receives the same vector wherever it appears. That is limiting for ambiguous words such as bank because the vector cannot shift between financial and river meanings based on context.",
  },

  {
    id: "cme295-lect1-q57",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which single statement best explains why transformers scale well on modern hardware?",
    options: [
      {
        text: "Their self-attention and feedforward operations can be expressed as batched matrix multiplications.",
        isCorrect: true,
      },
      {
        text: "They do not require any multiplications, only additions.",
        isCorrect: false,
      },
      {
        text: "They process each token entirely independently of all others, with no interactions.",
        isCorrect: false,
      },
      { text: "They avoid using GPUs by design.", isCorrect: false },
    ],
    explanation:
      "Transformers scale well on modern accelerators because attention and feedforward blocks are expressed largely as batched matrix multiplications. They still use multiplications, token interactions, and GPUs or TPUs heavily; their advantage is parallel structure rather than avoiding hardware acceleration.",
  },

  {
    id: "cme295-lect1-q58",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which single statement best describes the main role of the encoder outputs in an encoder–decoder transformer?",
    options: [
      {
        text: "They provide context representations of the source sequence for the decoder to attend to.",
        isCorrect: true,
      },
      {
        text: "They directly contain the final target-language tokens.",
        isCorrect: false,
      },
      { text: "They are discarded before decoding begins.", isCorrect: false },
      {
        text: "They only store positional encodings and no semantic information.",
        isCorrect: false,
      },
    ],
    explanation:
      "Encoder outputs are contextual source-token representations produced after source-side self-attention and feedforward transformations. The decoder uses them through cross-attention; they are not target tokens, discarded artifacts, or merely positional encodings.",
  },

  {
    id: "cme295-lect1-q59",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which single statement best describes why masked self-attention is required during training of autoregressive decoders?",
    options: [
      {
        text: "It enforces that each prediction depends only on past tokens, matching the generation process.",
        isCorrect: true,
      },
      {
        text: "It prevents the model from using the encoder outputs.",
        isCorrect: false,
      },
      {
        text: "It forces all attention weights to be equal.",
        isCorrect: false,
      },
      {
        text: "It is only needed for evaluation, not training.",
        isCorrect: false,
      },
    ],
    explanation:
      "Masked self-attention prevents a training-time decoder state from seeing target tokens to its right. That keeps training aligned with autoregressive generation, where each prediction can only depend on previous generated tokens and any allowed source context.",
  },

  {
    id: "cme295-lect1-q60",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which single statement best summarizes the purpose of positional encodings?",
    options: [
      {
        text: "To provide information about token order that self-attention alone does not capture.",
        isCorrect: true,
      },
      {
        text: "To randomly shuffle the positions of tokens during training.",
        isCorrect: false,
      },
      {
        text: "To reduce the vocabulary size by merging similar tokens.",
        isCorrect: false,
      },
      {
        text: "To implement the attention mechanism without queries or keys.",
        isCorrect: false,
      },
    ],
    explanation:
      "Self-attention needs explicit position information because, without it, the mechanism has no built-in way to distinguish the same tokens in different orders. Positional encodings add order signals; they do not shuffle text, merge vocabulary items, or replace query-key-value attention.",
  },

  {
    id: "cme295-lect1-q61",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which single statement best explains why label smoothing can improve generalization?",
    options: [
      {
        text: "It discourages the model from becoming overconfident on the training set.",
        isCorrect: true,
      },
      {
        text: "It forces the model to memorize every training example exactly.",
        isCorrect: false,
      },
      {
        text: "It ensures the model assigns zero probability to all incorrect classes.",
        isCorrect: false,
      },
      {
        text: "It removes the need for a softmax layer in the output.",
        isCorrect: false,
      },
    ],
    explanation:
      "Label smoothing can improve generalization by discouraging the model from assigning extreme confidence to the training label. It changes the target distribution in the loss, which can improve calibration and robustness, but it does not remove softmax or force memorization.",
  },

  {
    id: "cme295-lect1-q62",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which single statement best describes the effect of vanishing gradients on recurrent neural network training?",
    options: [
      {
        text: "Parameters influencing early time steps are updated very weakly, making long-range learning difficult.",
        isCorrect: true,
      },
      {
        text: "Gradients become extremely large and cause numerical instability.",
        isCorrect: false,
      },
      {
        text: "The loss function becomes constant and cannot be differentiated.",
        isCorrect: false,
      },
      {
        text: "Only the output layer is affected, while recurrent weights are unaffected.",
        isCorrect: false,
      },
    ],
    explanation:
      "Vanishing gradients make early time steps receive very weak learning signals after many recurrent transitions. That is different from exploding gradients, and it especially hurts learning dependencies where an early token should influence a much later prediction.",
  },

  {
    id: "cme295-lect1-q63",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "In decoder cross-attention for encoder-decoder translation, which single statement correctly identifies the usual query, key, and value sources?",
    options: [
      {
        text: "Decoder states provide the queries, while encoder outputs provide the keys and values.",
        isCorrect: true,
      },
      {
        text: "Encoder outputs provide the queries, while decoder states provide both keys and values.",
        isCorrect: false,
      },
      {
        text: "The token embeddings provide queries, keys, and values before either encoder or decoder runs.",
        isCorrect: false,
      },
      {
        text: "Cross-attention uses positional encodings as values and discards encoder semantic states.",
        isCorrect: false,
      },
    ],
    explanation:
      "In cross-attention, the decoder is asking which source-side information matters for the target token it is building, so its current hidden states act as queries. The encoded source sequence supplies keys and values, letting the decoder retrieve source-context information without treating positional encodings as the semantic content.",
  },

  {
    id: "cme295-lect1-q64",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which single statement best captures why attention was described as central to the transformer paper 'Attention Is All You Need'?",
    options: [
      {
        text: "The architecture replaces recurrent and convolutional layers with attention as the primary interaction mechanism.",
        isCorrect: true,
      },
      {
        text: "The architecture removes all linear transformations and uses only attention.",
        isCorrect: false,
      },
      {
        text: "The paper shows that attention is unnecessary for machine translation.",
        isCorrect: false,
      },
      {
        text: "The model uses attention only once at the output layer.",
        isCorrect: false,
      },
    ],
    explanation:
      "The title Attention Is All You Need emphasizes that the transformer replaces recurrent and convolutional sequence-mixing layers with attention as the main interaction mechanism. The architecture still contains linear projections and feedforward layers, but attention is the core way positions exchange information.",
  },

  {
    id: "cme295-lect1-q65",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which single statement best describes the purpose of the softmax layer at the output of a language model?",
    options: [
      {
        text: "To convert raw logits into a normalized probability distribution over tokens.",
        isCorrect: true,
      },
      {
        text: "To compute cosine similarity between tokens.",
        isCorrect: false,
      },
      {
        text: "To remove all negative values from embeddings.",
        isCorrect: false,
      },
      {
        text: "To determine attention weights between encoder and decoder states.",
        isCorrect: false,
      },
    ],
    explanation:
      "A language model produces raw logits over the vocabulary, and softmax converts those logits into nonnegative probabilities that sum to one. That operation is separate from cosine similarity, embedding cleanup, and the attention-weight computations inside the model.",
  },

  {
    id: "cme295-lect1-q66",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which single statement best describes the main purpose of precision and recall in evaluating classifiers?",
    options: [
      {
        text: "To provide more informative evaluation than accuracy alone on imbalanced datasets.",
        isCorrect: true,
      },
      {
        text: "To measure how fast the model runs on a GPU.",
        isCorrect: false,
      },
      {
        text: "To compute the average sequence length of the inputs.",
        isCorrect: false,
      },
      {
        text: "To estimate the vocabulary size of the tokenization scheme.",
        isCorrect: false,
      },
    ],
    explanation:
      "Precision and recall make classifier evaluation more informative when class balance is uneven or positive cases matter disproportionately. Accuracy can hide failures on the minority class, while precision and recall separately measure false-positive and false-negative behavior.",
  },

  {
    id: "cme295-lect1-q67",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which single statement best explains why attention helps with long-range dependencies compared to a pure recurrent neural network?",
    options: [
      {
        text: "It allows direct connections between distant tokens instead of relying on many recurrent steps.",
        isCorrect: true,
      },
      {
        text: "It stores the entire sequence in a single scalar value.",
        isCorrect: false,
      },
      {
        text: "It eliminates the need to backpropagate through time entirely.",
        isCorrect: false,
      },
      {
        text: "It forces all attention weights to be identical for all pairs of tokens.",
        isCorrect: false,
      },
    ],
    explanation:
      "Attention helps with long-range dependencies because any position can directly attend to distant positions in the same layer. A pure RNN must pass information through many sequential hidden-state updates, which lengthens the path for both information and gradients.",
  },

  {
    id: "cme295-lect1-q68",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which single statement best describes a core motivation for using dense embeddings instead of one-hot vectors?",
    options: [
      {
        text: "Dense embeddings can encode graded semantic similarity between tokens.",
        isCorrect: true,
      },
      {
        text: "Dense embeddings guarantee that all words in a language are equidistant.",
        isCorrect: false,
      },
      {
        text: "Dense embeddings completely remove the need for training data.",
        isCorrect: false,
      },
      {
        text: "Dense embeddings ensure that vocabulary size has no impact on dimensionality.",
        isCorrect: false,
      },
    ],
    explanation:
      "Dense embeddings are useful because they can encode graded similarity and other learned relationships between tokens. They still depend on training data and vocabulary design, but unlike one-hot vectors they are not forced to make every distinct token equally distant.",
  },

  {
    id: "cme295-lect1-q69",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which single statement best characterizes the role of the feedforward sub-layer inside a transformer block?",
    options: [
      {
        text: "To transform each token nonlinearly after attention.",
        isCorrect: true,
      },
      {
        text: "To compute the dot product between queries and keys.",
        isCorrect: false,
      },
      {
        text: "To introduce positional information into the model.",
        isCorrect: false,
      },
      {
        text: "To enforce masking in the decoder’s self-attention.",
        isCorrect: false,
      },
    ],
    explanation:
      "The feedforward sub-layer applies a nonlinear transformation to each token representation after attention has mixed cross-token information. It does not compute query-key dot products, add positions, or enforce decoder masking; those responsibilities belong to other parts of the transformer block.",
  },

  {
    id: "cme295-lect1-q70",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which single statement best describes why an encoder-decoder transformer uses both encoder self-attention and decoder cross-attention?",
    options: [
      {
        text: "Encoder self-attention builds source states; decoder cross-attention reads those states.",
        isCorrect: true,
      },
      {
        text: "Encoder self-attention directly emits target-language tokens, while the decoder stores source positions.",
        isCorrect: false,
      },
      {
        text: "Decoder cross-attention performs tokenization, then passes subword pieces back into the encoder.",
        isCorrect: false,
      },
      {
        text: "The two attention blocks mainly reduce vocabulary size before the final softmax layer.",
        isCorrect: false,
      },
    ],
    explanation:
      "The encoder first turns each source token into a context-aware representation by letting source positions attend to one another. The decoder then generates target tokens autoregressively and uses cross-attention to condition those predictions on the encoded source sequence.",
  },
];
