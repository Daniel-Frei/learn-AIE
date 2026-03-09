import { Question } from "../../quiz";

export const L5NLPBasicsQuestions: Question[] = [
  {
    id: "l5-nlp-basics-q01",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which of the following statements correctly describe the role of Natural Language Processing (NLP)?",
    options: [
      {
        text: "Many applications such as sentiment analysis or spam detection can be framed as text-in, text-out or text-in, label-out problems.",
        isCorrect: true,
      },
      {
        text: "Large amounts of human knowledge and communication exist in textual form.",
        isCorrect: true,
      },
      {
        text: "NLP techniques can be used for tasks like summarization or question answering.",
        isCorrect: true,
      },
      {
        text: "Code generation can also be treated as a form of text generation because code is ultimately text.",
        isCorrect: true,
      },
    ],
    explanation:
      "Natural Language Processing deals with analyzing and generating textual data. Many practical applications—including classification, summarization, and question answering—can be viewed as transformations from input text to output text or labels. Even tasks like code generation fall under NLP because programming languages are represented as sequences of text tokens.",
  },

  {
    id: "l5-nlp-basics-q02",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about the historical evolution of NLP approaches are correct?",
    options: [
      {
        text: "Early NLP systems often relied heavily on hand-crafted linguistic rules.",
        isCorrect: true,
      },
      {
        text: "Statistical machine learning methods replaced many rule-based approaches by relying on counting patterns in text.",
        isCorrect: true,
      },
      {
        text: "Recurrent Neural Networks became widely used for sequence modeling before transformers.",
        isCorrect: true,
      },
      {
        text: "Transformers were introduced before statistical NLP methods.",
        isCorrect: false,
      },
    ],
    explanation:
      "NLP historically progressed from rule-based systems to statistical machine learning methods, then to neural network approaches like Recurrent Neural Networks (RNNs), and finally to transformer architectures. Transformers were introduced in 2017, long after statistical approaches. The shift toward data-driven methods dramatically improved performance across many NLP tasks.",
  },

  {
    id: "l5-nlp-basics-q03",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the process of text vectorization?",
    options: [
      {
        text: "Text vectorization converts raw text into numerical vectors that machine learning models can process.",
        isCorrect: true,
      },
      {
        text: "A common pipeline includes standardization, tokenization, indexing, and encoding.",
        isCorrect: true,
      },
      {
        text: "Vectorization is necessary because neural networks operate on numerical data rather than strings.",
        isCorrect: true,
      },
      {
        text: "Vectorization guarantees that semantic meaning is perfectly preserved.",
        isCorrect: false,
      },
    ],
    explanation:
      "Text vectorization converts text into numerical form so that machine learning algorithms can operate on it. The typical pipeline includes standardization, tokenization, indexing, and encoding (often abbreviated STIE). However, simple vectorization methods such as bag-of-words do not preserve all semantic information, particularly word order and context.",
  },

  {
    id: "l5-nlp-basics-q04",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which of the following operations are typically associated with the standardization step of text preprocessing?",
    options: [
      { text: "Converting all characters to lowercase.", isCorrect: true },
      { text: "Removing punctuation characters.", isCorrect: true },
      {
        text: "Removing common stop words such as 'the' or 'and'.",
        isCorrect: true,
      },
      { text: "Assigning integer indices to each token.", isCorrect: false },
    ],
    explanation:
      "Standardization prepares raw text by applying transformations like converting text to lowercase, removing punctuation, and sometimes removing stop words. Indexing, however, happens later in the vectorization pipeline after tokenization. It assigns each token a unique integer identifier.",
  },

  {
    id: "l5-nlp-basics-q05",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements about tokenization are correct?",
    options: [
      {
        text: "Tokenization splits text into smaller units such as words or subwords.",
        isCorrect: true,
      },
      {
        text: "A common simple strategy is splitting text based on whitespace.",
        isCorrect: true,
      },
      {
        text: "Tokenization defines what units the model treats as basic elements of text.",
        isCorrect: true,
      },
      {
        text: "Tokenization always guarantees correct handling of compound words across all languages.",
        isCorrect: false,
      },
    ],
    explanation:
      "Tokenization divides text into smaller units called tokens. A simple method splits on whitespace, treating each word as a token. However, languages differ widely and many contain compound words or no whitespace boundaries, meaning simple tokenization methods can be inadequate.",
  },

  {
    id: "l5-nlp-basics-q06",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about vocabularies in NLP pipelines are correct?",
    options: [
      {
        text: "The vocabulary consists of the distinct tokens observed in the training corpus.",
        isCorrect: true,
      },
      {
        text: "Each token in the vocabulary is often assigned a unique integer index.",
        isCorrect: true,
      },
      {
        text: "Vocabulary construction typically occurs before encoding tokens into vectors.",
        isCorrect: true,
      },
      {
        text: "The vocabulary must contain every possible word in a language.",
        isCorrect: false,
      },
    ],
    explanation:
      "A vocabulary is the set of unique tokens extracted from the training corpus. Each token is typically assigned a unique index during the indexing step before being encoded into vectors. In practice, vocabularies are limited in size and often contain only the most frequent tokens rather than every word in the language.",
  },

  {
    id: "l5-nlp-basics-q07",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about one-hot encoding are correct?",
    options: [
      {
        text: "Each token is represented as a vector with a single 1 and the rest 0s.",
        isCorrect: true,
      },
      {
        text: "The length of the one-hot vector equals the vocabulary size.",
        isCorrect: true,
      },
      {
        text: "Two different tokens will produce identical one-hot vectors.",
        isCorrect: false,
      },
      {
        text: "One-hot vectors are typically sparse when vocabularies are large.",
        isCorrect: true,
      },
    ],
    explanation:
      "One-hot encoding represents each token as a vector where exactly one position contains a 1 and all others contain 0. The length of the vector equals the vocabulary size, ensuring each token has a unique representation. Because only one element is non-zero, these vectors are extremely sparse for large vocabularies.",
  },

  {
    id: "l5-nlp-basics-q08",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which of the following statements about the special token \\(<UNK>\\) are correct?",
    options: [
      {
        text: "It represents words that were not present in the training vocabulary.",
        isCorrect: true,
      },
      {
        text: "It helps handle unseen tokens during inference.",
        isCorrect: true,
      },
      {
        text: "All unseen words mapped to \\(<UNK>\\) become indistinguishable to the model.",
        isCorrect: true,
      },
      {
        text: "It guarantees that unseen words are semantically understood.",
        isCorrect: false,
      },
    ],
    explanation:
      "The \\(<UNK>\\) token stands for 'unknown' and represents tokens that do not appear in the vocabulary. During inference, unseen words are mapped to this token. However, this causes information loss because different unknown words are treated identically by the model.",
  },

  {
    id: "l5-nlp-basics-q09",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which issues can arise when representing text using simple whitespace tokenization?",
    options: [
      {
        text: "Compound words like 'father-in-law' may be split incorrectly.",
        isCorrect: true,
      },
      {
        text: "Languages without whitespace between words become difficult to tokenize.",
        isCorrect: true,
      },
      {
        text: "Word order information is automatically preserved.",
        isCorrect: false,
      },
      {
        text: "Extremely long words in some languages may create modeling difficulties.",
        isCorrect: true,
      },
    ],
    explanation:
      "Whitespace tokenization is simple but problematic. Some languages do not use spaces between words, and compound words may be split incorrectly. Additionally, some languages contain extremely long compound words, which complicates tokenization. Word order preservation depends on the model representation, not tokenization itself.",
  },

  {
    id: "l5-nlp-basics-q10",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose a vocabulary contains \\(V\\) tokens and a sentence contains \\(T\\) tokens. After one-hot encoding each token individually, what properties does the resulting representation have?",
    options: [
      {
        text: "The representation can be viewed as a matrix of shape \\(T \\times V\\).",
        isCorrect: true,
      },
      {
        text: "Each row contains exactly one value equal to 1.",
        isCorrect: true,
      },
      {
        text: "The representation directly preserves semantic similarity between words.",
        isCorrect: false,
      },
      {
        text: "The representation is sparse when \\(V\\) is large.",
        isCorrect: true,
      },
    ],
    explanation:
      "If each token is represented using one-hot encoding, the resulting structure is a matrix with \\(T\\) rows and \\(V\\) columns. Each row corresponds to a token and contains a single 1 indicating its index in the vocabulary. However, one-hot representations do not encode semantic similarity between words, and they are extremely sparse for large vocabularies.",
  },

  {
    id: "l5-nlp-basics-q11",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about the Bag-of-Words model are correct?",
    options: [
      {
        text: "It aggregates token representations into a fixed-length vector.",
        isCorrect: true,
      },
      { text: "It ignores the order of words in a sentence.", isCorrect: true },
      {
        text: "It can represent whether a word appears in a document.",
        isCorrect: true,
      },
      {
        text: "It inherently captures long-range contextual meaning between words.",
        isCorrect: false,
      },
    ],
    explanation:
      "The Bag-of-Words model converts variable-length text into fixed-length vectors by aggregating token occurrences. This representation ignores word order, meaning phrases like 'dog bites man' and 'man bites dog' may appear identical. While useful for simple tasks, it cannot capture deeper contextual relationships.",
  },

  {
    id: "l5-nlp-basics-q12",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe multi-hot encoding?",
    options: [
      {
        text: "It indicates whether a token appears at least once in a document.",
        isCorrect: true,
      },
      {
        text: "It aggregates multiple token vectors using a logical OR operation.",
        isCorrect: true,
      },
      { text: "It records how many times each word occurs.", isCorrect: false },
      {
        text: "The resulting vector length equals the vocabulary size.",
        isCorrect: true,
      },
    ],
    explanation:
      "Multi-hot encoding indicates whether a token appears in a document at least once. It effectively performs an OR operation across token vectors. Unlike count encoding, it does not track how many times each token occurs, only whether it appears.",
  },

  {
    id: "l5-nlp-basics-q13",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe count encoding in the Bag-of-Words model?",
    options: [
      {
        text: "It records the number of occurrences of each token in a document.",
        isCorrect: true,
      },
      {
        text: "It produces a vector with length equal to the vocabulary size.",
        isCorrect: true,
      },
      {
        text: "It preserves the original order of words in the text.",
        isCorrect: false,
      },
      {
        text: "It is also called term-frequency representation.",
        isCorrect: true,
      },
    ],
    explanation:
      "Count encoding records how many times each token appears in a document. The resulting vector has length equal to the vocabulary size. Although useful for many tasks, this approach still ignores word order and contextual meaning.",
  },

  {
    id: "l5-nlp-basics-q14",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which limitations are associated with the Bag-of-Words representation?",
    options: [
      { text: "It ignores sequential information in text.", isCorrect: true },
      {
        text: "It can produce very high-dimensional vectors when vocabularies are large.",
        isCorrect: true,
      },
      {
        text: "Short and long documents produce vectors of different lengths.",
        isCorrect: false,
      },
      {
        text: "It may require many parameters in downstream models.",
        isCorrect: true,
      },
    ],
    explanation:
      "Bag-of-Words ignores word order and therefore loses sequential information. The vectors also become extremely high-dimensional when vocabularies are large. However, all documents produce vectors of the same length equal to the vocabulary size, regardless of document length.",
  },

  {
    id: "l5-nlp-basics-q15",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Why might practitioners restrict the vocabulary to the most frequent tokens?",
    options: [
      {
        text: "To reduce the dimensionality of input vectors.",
        isCorrect: true,
      },
      {
        text: "To reduce the number of model parameters in downstream layers.",
        isCorrect: true,
      },
      { text: "To eliminate the need for unknown tokens.", isCorrect: false },
      {
        text: "To reduce computational cost and overfitting risk.",
        isCorrect: true,
      },
    ],
    explanation:
      "Limiting the vocabulary to the most frequent tokens reduces dimensionality and computational cost. Smaller input vectors lead to fewer parameters in neural networks and lower risk of overfitting. However, unknown tokens are still required to represent rare or unseen words.",
  },

  {
    id: "l5-nlp-basics-q16",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about using neural networks with Bag-of-Words inputs are correct?",
    options: [
      {
        text: "The input layer size equals the vocabulary size.",
        isCorrect: true,
      },
      {
        text: "Hidden layers can apply nonlinear transformations such as ReLU.",
        isCorrect: true,
      },
      {
        text: "The model must include at least one hidden layer to qualify as a neural network in many teaching contexts.",
        isCorrect: true,
      },
      {
        text: "Softmax outputs are commonly used for multi-class classification.",
        isCorrect: true,
      },
    ],
    explanation:
      "When using Bag-of-Words inputs, the input layer dimension equals the vocabulary size. Hidden layers with nonlinear activations such as ReLU transform the input features. In classification tasks with multiple classes, the final layer typically uses a softmax activation to produce probabilities.",
  },

  {
    id: "l5-nlp-basics-q17",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about softmax in classification networks are correct?",
    options: [
      {
        text: "Softmax converts raw outputs into probabilities that sum to 1.",
        isCorrect: true,
      },
      {
        text: "Softmax allows gradients to be computed for backpropagation.",
        isCorrect: true,
      },
      {
        text: "Selecting the maximum output directly without softmax is usually used during training.",
        isCorrect: false,
      },
      {
        text: "Softmax outputs can be used with categorical cross-entropy loss.",
        isCorrect: true,
      },
    ],
    explanation:
      "Softmax transforms logits into normalized probabilities that sum to 1. This property makes it suitable for multi-class classification tasks and enables training with categorical cross-entropy loss. Directly selecting the maximum output is not differentiable and therefore unsuitable for gradient-based training.",
  },

  {
    id: "l5-nlp-basics-q18",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements correctly describe bigrams in NLP?",
    options: [
      { text: "Bigrams represent pairs of adjacent tokens.", isCorrect: true },
      {
        text: "They introduce limited contextual information compared to unigrams.",
        isCorrect: true,
      },
      {
        text: "They dramatically increase the potential vocabulary size.",
        isCorrect: true,
      },
      {
        text: "They guarantee full understanding of sentence semantics.",
        isCorrect: false,
      },
    ],
    explanation:
      "Bigrams capture pairs of adjacent words and therefore encode limited contextual information. For example, 'not good' becomes distinguishable from 'good'. However, including bigrams greatly increases the number of possible tokens and does not fully capture semantic meaning.",
  },

  {
    id: "l5-nlp-basics-q19",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about dropout regularization are correct?",
    options: [
      {
        text: "Dropout randomly sets some neuron activations to zero during training.",
        isCorrect: true,
      },
      {
        text: "Dropout can reduce overfitting by preventing neurons from co-adapting.",
        isCorrect: true,
      },
      {
        text: "Dropout permanently removes neurons from the network architecture.",
        isCorrect: false,
      },
      {
        text: "Dropout effectively creates many different subnetworks during training.",
        isCorrect: true,
      },
    ],
    explanation:
      "Dropout randomly sets some activations to zero during training, forcing the network to learn robust features that do not rely on specific neurons. This helps reduce overfitting and encourages redundancy in learned representations. However, neurons are not permanently removed—they are simply masked during training iterations.",
  },

  {
    id: "l5-nlp-basics-q20",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe why Bag-of-Words vectors can lead to large neural network parameter counts?",
    options: [
      {
        text: "The input dimension equals the vocabulary size \\(V\\).",
        isCorrect: true,
      },
      {
        text: "The number of weights in the first dense layer scales roughly with \\(V \\times H\\), where \\(H\\) is the number of hidden units.",
        isCorrect: true,
      },
      {
        text: "Reducing vocabulary size can reduce parameter counts.",
        isCorrect: true,
      },
      {
        text: "Bag-of-Words guarantees parameter efficiency regardless of vocabulary size.",
        isCorrect: false,
      },
    ],
    explanation:
      "In Bag-of-Words models, the input dimension equals the vocabulary size \\(V\\). If the first hidden layer contains \\(H\\) neurons, the weight matrix contains approximately \\(V \\times H\\) parameters. Large vocabularies therefore dramatically increase parameter counts, which is why practitioners often limit vocabulary size.",
  },
  {
    id: "l5-nlp-basics-q21",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe why Natural Language Processing problems are often framed as supervised learning tasks?",
    options: [
      {
        text: "Many NLP tasks can be formulated as predicting an output \\(y\\) given input text \\(x\\).",
        isCorrect: true,
      },
      {
        text: "Text classification tasks such as sentiment analysis often use labeled training data.",
        isCorrect: true,
      },
      {
        text: "Neural networks for NLP can be viewed as functions \\(\\hat{y} = f(x, w)\\) where \\(w\\) are learned weights.",
        isCorrect: true,
      },
      { text: "All NLP tasks must be unsupervised.", isCorrect: false },
    ],
    explanation:
      "Many NLP tasks are naturally framed as supervised learning problems where models learn to map text inputs to outputs such as labels, summaries, or generated text. Neural networks implement a function \\(f(x, w)\\) that learns this mapping. While unsupervised methods also exist, NLP tasks are not limited to them.",
  },

  {
    id: "l5-nlp-basics-q22",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about stemming in NLP preprocessing are correct?",
    options: [
      {
        text: "Stemming attempts to reduce words to a shared root form.",
        isCorrect: true,
      },
      {
        text: "Examples include mapping 'eating', 'eaten', and 'ate' to a common form.",
        isCorrect: true,
      },
      {
        text: "Stemming always preserves grammatical correctness.",
        isCorrect: false,
      },
      { text: "Stemming can reduce vocabulary size.", isCorrect: true },
    ],
    explanation:
      "Stemming reduces different inflected forms of a word to a shared root. For example, 'eating', 'eaten', and 'ate' might all be mapped to a common base representation. While this reduces vocabulary size, stemming may produce grammatically incorrect roots and therefore does not always preserve full linguistic structure.",
  },

  {
    id: "l5-nlp-basics-q23",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe stop words in NLP preprocessing?",
    options: [
      {
        text: "Stop words include common words such as 'the', 'and', or 'a'.",
        isCorrect: true,
      },
      {
        text: "They are sometimes removed because they may provide limited predictive signal.",
        isCorrect: true,
      },
      {
        text: "Stop words are always removed in modern NLP pipelines.",
        isCorrect: false,
      },
      {
        text: "Removing stop words may reduce the dimensionality of the vocabulary.",
        isCorrect: true,
      },
    ],
    explanation:
      "Stop words are common words that often provide little predictive value in some NLP tasks. Removing them can reduce vocabulary size and computational cost. However, modern pipelines—especially those used in large language models—often keep them because they may still contribute contextual information.",
  },

  {
    id: "l5-nlp-basics-q24",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which properties apply to one-hot encoded representations of tokens?",
    options: [
      { text: "They represent tokens as orthogonal vectors.", isCorrect: true },
      {
        text: "They encode semantic similarity between related words.",
        isCorrect: false,
      },
      {
        text: "The vectors are typically high-dimensional when vocabularies are large.",
        isCorrect: true,
      },
      { text: "They contain mostly zeros.", isCorrect: true },
    ],
    explanation:
      "One-hot vectors are orthogonal, meaning each token is represented independently with no shared structure. This representation results in high-dimensional vectors when vocabularies are large, and the vectors are extremely sparse with mostly zero values. However, one-hot encoding does not encode semantic relationships between tokens.",
  },

  {
    id: "l5-nlp-basics-q25",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which of the following tasks can be solved using the text-in, text-out paradigm?",
    options: [
      { text: "Text summarization.", isCorrect: true },
      { text: "Question answering systems.", isCorrect: true },
      { text: "Generating marketing copy.", isCorrect: true },
      {
        text: "Image classification using pixel inputs only.",
        isCorrect: false,
      },
    ],
    explanation:
      "Many NLP tasks can be framed as text-in, text-out problems, including summarization, question answering, and text generation. These tasks take text input and produce textual outputs. Image classification, however, involves image data rather than textual inputs and therefore does not fall under this paradigm.",
  },

  {
    id: "l5-nlp-basics-q26",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose a Bag-of-Words model produces vectors of size \\(V\\). If the first hidden layer contains \\(H\\) neurons, approximately how many weights exist between the input and the hidden layer?",
    options: [
      { text: "\\(V \\times H\\)", isCorrect: true },
      { text: "\\(V + H\\)", isCorrect: false },
      {
        text: "Increasing \\(V\\) increases the number of parameters in this layer.",
        isCorrect: true,
      },
      {
        text: "Reducing \\(V\\) can reduce the number of parameters in the network.",
        isCorrect: true,
      },
    ],
    explanation:
      "The weight matrix connecting the input layer to the first hidden layer contains \\(V \\times H\\) parameters. As vocabulary size increases, the number of parameters grows linearly, increasing computational cost and risk of overfitting. Reducing the vocabulary size can therefore significantly reduce model complexity.",
  },

  {
    id: "l5-nlp-basics-q27",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which limitations arise when representing sentences as Bag-of-Words vectors?",
    options: [
      { text: "Word order information is lost.", isCorrect: true },
      {
        text: "Different sentences with the same words may appear identical.",
        isCorrect: true,
      },
      {
        text: "The representation always preserves syntactic structure.",
        isCorrect: false,
      },
      {
        text: "Long and short documents produce vectors of equal length.",
        isCorrect: true,
      },
    ],
    explanation:
      "Bag-of-Words ignores word order, meaning that sentences with identical words but different ordering will produce the same representation. This removes syntactic structure and contextual meaning. However, all documents map to vectors of equal length equal to the vocabulary size.",
  },

  {
    id: "l5-nlp-basics-q28",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about token vocabularies in NLP systems are correct?",
    options: [
      {
        text: "The vocabulary typically contains tokens observed in the training corpus.",
        isCorrect: true,
      },
      {
        text: "Vocabulary size influences the dimensionality of encoded vectors.",
        isCorrect: true,
      },
      {
        text: "Rare tokens may be replaced with \\(<UNK>\\).",
        isCorrect: true,
      },
      {
        text: "Vocabulary size must equal the total number of sentences in the dataset.",
        isCorrect: false,
      },
    ],
    explanation:
      "The vocabulary contains the unique tokens extracted from the training corpus. Its size directly determines the dimensionality of many encoding schemes such as one-hot or Bag-of-Words representations. Rare tokens are often replaced with an unknown token to limit vocabulary size.",
  },

  {
    id: "l5-nlp-basics-q29",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about bigram representations are correct?",
    options: [
      { text: "A bigram consists of two consecutive tokens.", isCorrect: true },
      {
        text: "Using bigrams increases the possible number of tokens.",
        isCorrect: true,
      },
      {
        text: "Bigrams can help capture limited context around a word.",
        isCorrect: true,
      },
      {
        text: "Bigrams completely eliminate the loss of word order information.",
        isCorrect: false,
      },
    ],
    explanation:
      "Bigrams represent pairs of adjacent tokens and therefore capture limited local context. While they can help represent phrases such as 'not good', they do not fully preserve sentence structure. Additionally, introducing bigrams dramatically increases the number of possible tokens.",
  },

  {
    id: "l5-nlp-basics-q30",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which properties describe a neural network used for multi-class text classification?",
    options: [
      {
        text: "The output layer often uses a softmax activation function.",
        isCorrect: true,
      },
      { text: "Softmax outputs probabilities that sum to 1.", isCorrect: true },
      {
        text: "Categorical cross-entropy is commonly used as the loss function.",
        isCorrect: true,
      },
      {
        text: "Softmax prevents gradients from being computed during training.",
        isCorrect: false,
      },
    ],
    explanation:
      "In multi-class classification, the final layer often uses softmax to convert logits into normalized probabilities. These probabilities sum to 1 and allow the model to be trained using categorical cross-entropy loss. Softmax is differentiable, allowing gradients to propagate during training.",
  },

  {
    id: "l5-nlp-basics-q31",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which of the following describes tokenization?",
    options: [
      {
        text: "The process of splitting text into tokens such as words or subwords.",
        isCorrect: true,
      },
      {
        text: "The process of mapping tokens to numerical vectors.",
        isCorrect: false,
      },
      {
        text: "The process of assigning each token an integer index.",
        isCorrect: false,
      },
      {
        text: "The process of training a neural network model.",
        isCorrect: false,
      },
    ],
    explanation:
      "Tokenization refers specifically to splitting raw text into smaller units called tokens. These tokens may be words, subwords, or characters depending on the method used. Mapping tokens to integers or vectors occurs later during indexing and encoding.",
  },

  {
    id: "l5-nlp-basics-q32",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statement about Bag-of-Words representations is correct?",
    options: [
      {
        text: "They always encode sentence order explicitly.",
        isCorrect: false,
      },
      {
        text: "They represent documents as fixed-length vectors.",
        isCorrect: true,
      },
      {
        text: "They require sequence models such as transformers to operate.",
        isCorrect: false,
      },
      { text: "They prevent vocabulary growth entirely.", isCorrect: false },
    ],
    explanation:
      "Bag-of-Words representations convert documents into fixed-length vectors corresponding to vocabulary size. However, they ignore word order and do not inherently require sequence models. Vocabulary size can still grow depending on preprocessing decisions.",
  },

  {
    id: "l5-nlp-basics-q33",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statement about the \\(<UNK>\\) token is correct?",
    options: [
      { text: "It represents words unseen during training.", isCorrect: true },
      {
        text: "It ensures semantic understanding of rare words.",
        isCorrect: false,
      },
      {
        text: "It stores embeddings for every unseen token.",
        isCorrect: false,
      },
      {
        text: "It preserves the identity of each unseen word.",
        isCorrect: false,
      },
    ],
    explanation:
      "The \\(<UNK>\\) token represents words that were not seen in the training vocabulary. While it allows models to handle unseen inputs, it causes information loss because all unknown tokens are mapped to the same representation.",
  },

  {
    id: "l5-nlp-basics-q34",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statement about one-hot encoding is correct?",
    options: [
      {
        text: "Each token vector contains exactly one non-zero element.",
        isCorrect: true,
      },
      {
        text: "The encoding preserves semantic similarity between synonyms.",
        isCorrect: false,
      },
      { text: "One-hot encoding produces dense vectors.", isCorrect: false },
      {
        text: "One-hot vectors are typically low-dimensional.",
        isCorrect: false,
      },
    ],
    explanation:
      "One-hot encoding represents tokens as vectors containing exactly one 1 and the rest 0s. Because vocabularies can be large, these vectors are high-dimensional and sparse. They do not encode semantic similarity between words.",
  },

  {
    id: "l5-nlp-basics-q35",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statement about count encoding in Bag-of-Words is correct?",
    options: [
      {
        text: "It records how many times each word appears in a document.",
        isCorrect: true,
      },
      { text: "It preserves the original order of tokens.", isCorrect: false },
      { text: "It guarantees semantic understanding.", isCorrect: false },
      {
        text: "It always produces shorter vectors than multi-hot encoding.",
        isCorrect: false,
      },
    ],
    explanation:
      "Count encoding records the frequency of each token in a document. While useful for representing term importance, it ignores token order and does not encode semantic meaning. The vector length still equals the vocabulary size.",
  },

  {
    id: "l5-nlp-basics-q36",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statement about dropout is correct?",
    options: [
      {
        text: "Dropout randomly sets some neuron activations to zero during training.",
        isCorrect: true,
      },
      {
        text: "Dropout permanently deletes neurons from the architecture.",
        isCorrect: false,
      },
      { text: "Dropout guarantees perfect generalization.", isCorrect: false },
      {
        text: "Dropout is only used in convolutional neural networks.",
        isCorrect: false,
      },
    ],
    explanation:
      "Dropout randomly disables some neuron activations during training. This prevents neurons from becoming overly dependent on each other and helps reduce overfitting. However, neurons are not permanently removed and dropout does not guarantee perfect generalization.",
  },

  {
    id: "l5-nlp-basics-q37",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statement about bigrams is correct?",
    options: [
      { text: "They represent adjacent word pairs.", isCorrect: true },
      { text: "They guarantee full sentence understanding.", isCorrect: false },
      {
        text: "They eliminate the need for neural networks.",
        isCorrect: false,
      },
      { text: "They prevent vocabulary growth.", isCorrect: false },
    ],
    explanation:
      "Bigrams represent adjacent token pairs and provide limited contextual information. However, they cannot fully capture sentence structure or semantic meaning and may significantly increase vocabulary size.",
  },

  {
    id: "l5-nlp-basics-q38",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statement about vocabulary size is correct?",
    options: [
      {
        text: "Vocabulary size directly determines the dimensionality of one-hot vectors.",
        isCorrect: true,
      },
      {
        text: "Vocabulary size determines the number of sentences in the dataset.",
        isCorrect: false,
      },
      { text: "Vocabulary size cannot be limited.", isCorrect: false },
      {
        text: "Vocabulary size has no impact on computational cost.",
        isCorrect: false,
      },
    ],
    explanation:
      "Vocabulary size determines the dimensionality of representations such as one-hot vectors and Bag-of-Words vectors. Larger vocabularies increase computational cost and parameter counts in downstream models.",
  },

  {
    id: "l5-nlp-basics-q39",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statement about neural networks for text classification is correct?",
    options: [
      {
        text: "Input vectors often correspond to vocabulary-sized feature vectors.",
        isCorrect: true,
      },
      {
        text: "Hidden layers transform these features into learned representations.",
        isCorrect: false,
      },
      {
        text: "Softmax outputs are unnecessary for classification.",
        isCorrect: false,
      },
      {
        text: "Training requires gradient-based optimization.",
        isCorrect: false,
      },
    ],
    explanation:
      "In Bag-of-Words neural networks, input vectors typically correspond to vocabulary-sized features. These vectors are passed into hidden layers for transformation and classification. The network is trained using gradient-based optimization and softmax outputs for classification.",
  },

  {
    id: "l5-nlp-basics-q40",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statement about NLP preprocessing pipelines is correct?",
    options: [
      {
        text: "They convert raw text into numerical representations suitable for machine learning.",
        isCorrect: true,
      },
      { text: "They eliminate the need for model training.", isCorrect: false },
      {
        text: "They automatically guarantee perfect semantic understanding.",
        isCorrect: false,
      },
      { text: "They remove the need for tokenization.", isCorrect: false },
    ],
    explanation:
      "NLP preprocessing pipelines convert raw text into numerical representations such as token indices or vector encodings. These steps prepare the data for machine learning models but do not eliminate the need for training or guarantee semantic understanding.",
  },
];
