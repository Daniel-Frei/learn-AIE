import { Question } from "../../quiz";

export const TransformersQuestions: Question[] = [
  {
    id: "mit15773-l7-q01",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the kind of natural-language task used to motivate transformers in this lecture?",
    options: [
      {
        text: "A natural-language query can be converted into a structured query such as SQL so that a database can be searched more reliably.",

        isCorrect: true,
      },
      {
        text: "Travel search is a useful example because entities such as origin, destination, and time must be extracted accurately from text.",
        isCorrect: true,
      },
      {
        text: "The motivating use case emphasizes that there can be a relatively high bar for accuracy, because users expect the system to interpret the query correctly.",
        isCorrect: true,
      },
      {
        text: "It is not the case that The main example is an image-classification problem in which each pixel is mapped to a travel entity.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture used information retrieval, especially travel search, as a concrete example of how natural-language understanding can be turned into structured querying. Accuracy matters because a query like a flight request has a specific intended meaning, unlike open-ended generation where many outputs may be acceptable.",
  },
  {
    id: "mit15773-l7-q02",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "In the Airline Travel Information Systems (ATIS) slot-filling setup, what is the learning problem?",
    options: [
      {
        text: "Each word in the input query must be classified into one of many possible slot labels.",

        isCorrect: true,
      },
      {
        text: "The problem can be viewed as a word-to-slot multi-class classification task.",
        isCorrect: true,
      },
      {
        text: "It is not the case that The entire query is assigned exactly one slot label, and individual words are ignored.",
        isCorrect: true,
      },
      {
        text: "A model should ideally produce an output sequence whose length matches the input sequence length.",
        isCorrect: true,
      },
    ],
    explanation:
      "ATIS slot filling is not sentence-level classification. Instead, every token in the query gets its own label, such as a departure city, arrival time, or 'other', so the model must preserve token-level structure rather than collapse everything into a single class.",
  },
  {
    id: "mit15773-l7-q03",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about the BIO-style tags used in the slot labels are correct?",
    options: [
      {
        text: "A tag beginning with \\(\\text{B-}\\) marks the beginning of a labeled entity span.",

        isCorrect: true,
      },
      {
        text: "A tag beginning with \\(\\text{I-}\\) indicates continuation inside a multi-token labeled span.",
        isCorrect: true,
      },
      {
        text: "The label \\(\\text{O}\\) refers to tokens that do not belong to any slot of interest.",
        isCorrect: true,
      },
      {
        text: "It is not the case that A token tagged with \\(\\text{I-}\\) must always start a new entity span.",
        isCorrect: true,
      },
    ],
    explanation:
      "The BIO convention distinguishes the start of an entity from its continuation. This matters for phrases like times or locations that may span multiple tokens, while \\(\\text{O}\\) is used for ordinary words that are not part of a target slot.",
  },
  {
    id: "mit15773-l7-q04",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Why is context necessary in a word-to-slot problem such as travel-query slot filling?",
    options: [
      {
        text: "A word like 'Boston' could be a departure city in one sentence and an arrival city in another sentence.",

        isCorrect: true,
      },
      {
        text: "A word such as 'station' can have different meanings depending on nearby words, so a single context-free embedding may be insufficient.",
        isCorrect: true,
      },
      {
        text: "Pronouns or ambiguous words can depend on surrounding words for interpretation.",
        isCorrect: true,
      },
      {
        text: "It is not the case that If a token appears in the vocabulary, its role is fixed and surrounding words become irrelevant.",
        isCorrect: true,
      },
    ],
    explanation:
      "Context helps determine both semantic meaning and task-specific role. The same surface token can behave differently across sentences, so a good model must use surrounding information rather than rely only on a fixed dictionary meaning.",
  },
  {
    id: "mit15773-l7-q05",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements capture the three main requirements the lecture emphasized for the architecture?",
    options: [
      {
        text: "It should take the surrounding context of each word into account.",

        isCorrect: true,
      },
      {
        text: "It should preserve enough structure so that the output can have the same length as the input.",
        isCorrect: true,
      },
      {
        text: "It should take word order into account.",
        isCorrect: true,
      },
      {
        text: "It is not the case that It should force all queries to have exactly the same semantic interpretation regardless of wording.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture highlighted context, order, and same-length input-output behavior as the key design requirements. These are especially important for token-level labeling tasks, where each input token needs a corresponding output prediction.",
  },
  {
    id: "mit15773-l7-q06",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Why does the lecture argue that a single standalone embedding per word, such as a GloVe-style embedding, is not enough?",
    options: [
      {
        text: "Because one word can appear in different contexts with different meanings or roles.",

        isCorrect: true,
      },
      {
        text: "Because a single word vector can become an average over multiple senses of the word.",
        isCorrect: true,
      },
      {
        text: "Because contextual meaning often depends on nearby words rather than on the token identity alone.",
        isCorrect: true,
      },
      {
        text: "It is not the case that Because standalone embeddings cannot represent any similarity at all between words.",
        isCorrect: true,
      },
    ],
    explanation:
      "Standalone embeddings can still capture useful semantic similarity, but they assign one vector per word type. That becomes limiting when the word is ambiguous or when the task requires understanding the role of the word in its current sentence rather than just its overall distributional meaning.",
  },
  {
    id: "mit15773-l7-q07",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Suppose the current word is 'station' in the sentence 'The train slowly left the station.' Which words would intuitively receive more attention when contextualizing 'station'?",
    options: [
      {
        text: "'train' should receive substantial attention because it strongly suggests the transportation sense of 'station'.",

        isCorrect: true,
      },
      {
        text: "'slowly' and 'left' may still matter, though typically less than 'train'.",
        isCorrect: true,
      },
      {
        text: "A second occurrence of 'the' in the sentence might contribute little compared with more content-bearing words.",
        isCorrect: true,
      },
      {
        text: "It is not the case that The word 'station' itself should be completely ignored when computing its contextual version.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture's intuition was that some words provide much stronger clues about the intended sense of a token than others. The token itself also still matters, because contextualization starts from the original representation and adjusts it using surrounding information rather than discarding it entirely.",
  },
  {
    id: "mit15773-l7-q08",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Let \\(\\hat{\\mathbf{w}}_6\\) denote the contextual embedding of the sixth word in a sentence with standalone embeddings \\(\\mathbf{w}_1, \\dots, \\mathbf{w}_6\\). Which statements are correct about the lecture's basic self-attention intuition?",
    options: [
      {
        text: "A contextual embedding can be formed as a weighted average of the standalone embeddings of all words in the sentence.",

        isCorrect: true,
      },
      {
        text: "The weights used for contextualizing one target word do not have to be the same as the weights used for another target word.",
        isCorrect: true,
      },
      {
        text: "The contextual embedding \\(\\hat{\\mathbf{w}}_6\\) can be written in the form \\(\\hat{\\mathbf{w}}_6 = \\sum_{i=1}^{6} s_{i,6}\\mathbf{w}_i\\).",
        isCorrect: true,
      },
      {
        text: "It is not the case that Because the weighted sum uses all words, the contextual embedding necessarily has a different dimensionality from the original standalone embeddings.",
        isCorrect: true,
      },
    ],
    explanation:
      "A weighted average preserves dimensionality because it combines vectors of the same size into another vector of that same size. The important change is not vector length but the information content: the new representation incorporates sentence context through the weights.",
  },
  {
    id: "mit15773-l7-q09",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Why are dot products useful in the lecture's intuition for attention weights?",
    options: [
      {
        text: "A larger dot product can indicate greater relatedness between two embeddings, especially when the angle between them is small.",

        isCorrect: true,
      },
      {
        text: "If two embeddings are roughly unit length, the dot product is closely connected to the cosine of the angle between them.",
        isCorrect: true,
      },
      {
        text: "Dot products can help distinguish more aligned vectors from orthogonal or oppositely directed vectors.",
        isCorrect: true,
      },
      {
        text: "It is not the case that A dot product is useful only when the two vectors are one-hot encoded.",
        isCorrect: true,
      },
    ],
    explanation:
      "The dot product combines magnitude and angular alignment, and under unit-length intuition it reduces to cosine similarity. That makes it a natural way to measure how related or aligned two embeddings are before converting those scores into normalized attention weights.",
  },
  {
    id: "mit15773-l7-q10",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose attention weights for a target word are computed by a softmax over compatibility scores, for example\n\\[\n s_{i,6} = \\frac{\\exp(\\langle \\mathbf{w}_i, \\mathbf{w}_6 \\rangle)}{\\sum_{k=1}^{6} \\exp(\\langle \\mathbf{w}_k, \\mathbf{w}_6 \\rangle)}.\n\\]\nWhich statements are correct?",
    options: [
      {
        text: "Exponentiation ensures the unnormalized scores become non-negative.",

        isCorrect: true,
      },
      {
        text: "Dividing by the sum makes the weights add up to 1.",
        isCorrect: true,
      },
      {
        text: "This is analogous to the softmax idea used elsewhere in deep learning.",
        isCorrect: true,
      },
      {
        text: "It is not the case that The normalization step forces all attention weights to be exactly equal.",
        isCorrect: true,
      },
    ],
    explanation:
      "The softmax converts arbitrary compatibility scores into a valid distribution over positions. It preserves relative differences rather than erasing them, so more relevant words can still receive larger weights than less relevant ones.",
  },
  {
    id: "mit15773-l7-q11",
    chapter: 1,
    difficulty: "easy",
    prompt: "What is self-attention in the lecture's basic description?",
    options: [
      {
        text: "It is not the case that It is an operation that turns standalone embeddings into contextual embeddings by combining information from all words in the same sentence.",

        isCorrect: false,
      },
      {
        text: "It can be viewed as letting each word attend to other words, including potentially itself.",
        isCorrect: true,
      },
      {
        text: "It is the key building block of transformers.",
        isCorrect: true,
      },
      {
        text: "It works by permanently deleting all original token representations before any context is computed.",
        isCorrect: false,
      },
    ],
    explanation:
      "Self-attention lets each position build a new representation from the full set of token representations in the sequence. The original embeddings are not thrown away at the start; they are the inputs used to construct the contextualized outputs.",
  },
  {
    id: "mit15773-l7-q12",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Why does the lecture introduce multi-head attention rather than using only one self-attention head?",
    options: [
      {
        text: "It is not the case that Different heads can learn to focus on different kinds of patterns in the same sentence.",

        isCorrect: false,
      },
      {
        text: "Some heads may capture patterns related to meaning, while others may capture tone, tense, or entity relationships.",
        isCorrect: true,
      },
      {
        text: "The idea is loosely analogous to using multiple filters in a convolutional neural network.",
        isCorrect: true,
      },
      {
        text: "Multi-head attention is used only to reduce the vocabulary size before tokenization.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture's intuition was that language contains several overlapping structures, so one attention mechanism may be too limited. Multiple heads allow the model to learn different relational patterns in parallel, much like different convolutional filters can specialize in different visual features.",
  },
  {
    id: "mit15773-l7-q13",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "After several attention heads produce contextual vectors, what is the purpose of 'concatenate and project' in the lecture's explanation?",
    options: [
      {
        text: "It is not the case that The outputs from multiple heads are concatenated to combine information learned by different heads.",

        isCorrect: false,
      },
      {
        text: "A dense linear projection can map the concatenated vector back to the original embedding size.",
        isCorrect: true,
      },
      {
        text: "This helps preserve a consistent input-output interface so transformer blocks can be stacked.",
        isCorrect: true,
      },
      {
        text: "The concatenation step guarantees interpretability of every attention head.",
        isCorrect: false,
      },
    ],
    explanation:
      "Concatenation collects the information from several heads, but it also increases dimensionality. The projection step compresses the result back to the standard size, which is useful for keeping the architecture compositional and stackable.",
  },
  {
    id: "mit15773-l7-q14",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Why are feed-forward layers added after multi-head attention in the transformer encoder as described in the lecture?",
    options: [
      {
        text: "It is not the case that They inject non-linearity into the processing.",

        isCorrect: false,
      },
      {
        text: "They can further transform the contextual representations produced by attention.",
        isCorrect: true,
      },
      {
        text: "They are typically simple position-wise dense layers rather than recurrent networks.",
        isCorrect: true,
      },
      {
        text: "They are included only to sort tokens into alphabetical order before classification.",
        isCorrect: false,
      },
    ],
    explanation:
      "Attention alone, in the lecture's simplified discussion, does not provide the full nonlinear transformation usually desired in a deep model. The feed-forward block adds more expressive capacity by applying learned dense transformations to each position's representation.",
  },
  {
    id: "mit15773-l7-q15",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "At the point in the lecture before positional encoding is introduced, which statements are correct about the simplified attention-based architecture?",
    options: [
      {
        text: "It is not the case that It already accounts for surrounding context.",

        isCorrect: false,
      },
      {
        text: "It can already produce an output sequence with the same number of positions as the input sequence.",
        isCorrect: true,
      },
      {
        text: "It does not yet inherently distinguish between different orderings of the same set of input tokens.",
        isCorrect: true,
      },
      {
        text: "It can correctly interpret word order solely because dot products are computed pairwise.",
        isCorrect: false,
      },
    ],
    explanation:
      "Self-attention over a set of token representations can contextualize each token while preserving sequence length, but without positional information it treats the inputs more like a set than an ordered sequence. That is why order-sensitive tasks need an additional positional mechanism.",
  },
  {
    id: "mit15773-l7-q16",
    chapter: 1,
    difficulty: "easy",
    prompt: "What is the main purpose of positional encoding in the lecture?",
    options: [
      {
        text: "It is not the case that It injects information about where a token appears in the sentence.",

        isCorrect: false,
      },
      {
        text: "It helps the model distinguish sequences that contain the same words in different orders.",
        isCorrect: true,
      },
      {
        text: "It is added to the standalone token embedding before entering the transformer encoder.",
        isCorrect: true,
      },
      {
        text: "It replaces the need for embeddings of the words themselves.",
        isCorrect: false,
      },
    ],
    explanation:
      "Positional encoding supplements token identity with location information. The model still needs the token embedding itself, because order alone does not tell the model what word is present.",
  },
  {
    id: "mit15773-l7-q17",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "In the lecture's positional-encoding example, a token representation is formed by adding a word embedding and a position embedding. Which statements are correct?",
    options: [
      {
        text: "It is not the case that If the same word appears at different positions, its resulting positional input embedding can differ across those occurrences.",

        isCorrect: false,
      },
      {
        text: "The position embeddings are independent of the word vocabulary and depend instead on the possible positions in the sequence.",
        isCorrect: true,
      },
      {
        text: "The summed positional input embedding has the same dimensionality as the original word embedding, provided the position embedding uses the same dimensionality.",
        isCorrect: true,
      },
      {
        text: "Adding position information forces all repeated words in a sentence to share exactly the same final input vector.",
        isCorrect: false,
      },
    ],
    explanation:
      "A repeated word can have different positional input vectors because the word embedding stays fixed while the added position embedding changes. This is exactly how the architecture starts to encode order without abandoning the standard embedding-space dimensionality.",
  },
  {
    id: "mit15773-l7-q18",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Why is it useful that a transformer encoder block preserves both the number of sequence positions and the representation dimensionality?",
    options: [
      {
        text: "It is not the case that It makes it easy to stack multiple encoder blocks one after another.",

        isCorrect: false,
      },
      {
        text: "It keeps token-level alignment available for downstream tasks such as slot filling.",
        isCorrect: true,
      },
      {
        text: "It provides a clean interface where the output of one block can be the input of the next block.",
        isCorrect: true,
      },
      {
        text: "It guarantees that the model uses exactly one attention head per token position.",
        isCorrect: false,
      },
    ],
    explanation:
      "Preserving shape is one of the architectural reasons transformers are easy to compose into deep stacks. It also helps for tasks where later layers still need token-level predictions, because positions remain aligned throughout the encoder pipeline.",
  },
  {
    id: "mit15773-l7-q19",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about the slot-filling architecture built on top of the transformer encoder are correct?",
    options: [
      {
        text: "It is not the case that Each output position can be passed through a softmax over the slot vocabulary to predict a label for that token.",

        isCorrect: false,
      },
      {
        text: "A dense layer with a ReLU activation can be placed between the transformer output and the final slot-classification softmax.",
        isCorrect: true,
      },
      {
        text: "The model can be trained end-to-end with backpropagation so that all these layers learn jointly.",
        isCorrect: true,
      },
      {
        text: "Because the transformer already contextualizes the sequence, no classification layer is needed at the output positions.",
        isCorrect: false,
      },
    ],
    explanation:
      "The transformer encoder produces contextual token representations, but those are still intermediate features. A final classifier, typically a dense layer and softmax at each position, is needed to map those features to actual slot labels.",
  },
  {
    id: "mit15773-l7-q20",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which of the following are reasonable takeaways from the lecture's treatment of transformers?",
    options: [
      {
        text: "It is not the case that Transformers became influential far beyond machine translation and now appear in many domains.",

        isCorrect: false,
      },
      {
        text: "A transformer encoder can be understood as combining embeddings, positional information, self-attention, and feed-forward processing.",
        isCorrect: true,
      },
      {
        text: "For token-level tasks such as slot filling, preserving sequence length while contextualizing each token is especially useful.",
        isCorrect: true,
      },
      {
        text: "The lecture's simplified first-pass explanation covered every engineering detail of the original transformer paper, including all mathematical refinements.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture gave an accessible conceptual explanation of what transformers do and why they are useful, especially for token-level NLP tasks. It explicitly deferred some details, such as residual connections, layer normalization, and certain linear projections, to the next lecture rather than covering every refinement immediately.",
  },

  {
    id: "mit15773-l7-q21",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe why transformers became so important beyond their original machine-translation use case?",
    options: [
      {
        text: "It is not the case that They turned out to be useful across many domains such as search, speech, computer vision, reinforcement learning, and generative AI.",

        isCorrect: false,
      },
      {
        text: "It is not the case that Their usefulness comes partly from a flexible architecture that can model relationships among elements in a sequence or set.",
        isCorrect: false,
      },
      {
        text: "The lecture emphasized that transformers dramatically improved only translation, but not other applications.",
        isCorrect: false,
      },
      {
        text: "Specialized systems such as AlphaFold were mentioned as examples of transformer-based successes outside standard NLP tasks.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture stressed that transformers started in translation but spread much more broadly because the underlying architecture is powerful and flexible. Their success in areas like search, multimodal systems, and even protein folding shows that the core idea generalizes well beyond classic text-to-text tasks.",
  },
  {
    id: "mit15773-l7-q22",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Why was the example query about a Brazilian traveler needing a visa for the USA useful in the lecture's discussion of search?",
    options: [
      {
        text: "It is not the case that It illustrated that understanding word order and role assignment can materially change search quality.",

        isCorrect: false,
      },
      {
        text: "It is not the case that It showed that older search behavior could confuse who is traveling to which country.",
        isCorrect: false,
      },
      {
        text: "It demonstrated that better language understanding can improve ranking or retrieval outcomes even in a highly optimized search system.",
        isCorrect: true,
      },
      {
        text: "It proved that search quality depends only on having more tokens in the query, not on better modeling.",
        isCorrect: false,
      },
    ],
    explanation:
      "The example showed that bag-of-words-like matching is often not enough when relational structure matters. A transformer-based model can better capture who is traveling where, which changes which result is actually most relevant to the user.",
  },
  {
    id: "mit15773-l7-q23",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements are correct about the output structure in the ATIS slot-filling task?",
    options: [
      {
        text: "It is not the case that If an input query has \\(n\\) tokens, the ideal output also has \\(n\\) token-level predictions.",

        isCorrect: false,
      },
      {
        text: "It is not the case that A separate softmax can conceptually be attached to each output position to choose among slot labels.",
        isCorrect: false,
      },
      {
        text: "The model's goal is to assign the same single slot label to all words in the query.",
        isCorrect: false,
      },
      {
        text: "The token-aligned output makes it possible to label words such as city names or times individually.",
        isCorrect: true,
      },
    ],
    explanation:
      "This is a sequence labeling problem, not a single-label classification problem. Each word needs its own decision, which is why preserving the length and alignment of the sequence is so important for the model design.",
  },
  {
    id: "mit15773-l7-q24",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about ambiguous words in the lecture's examples are correct?",
    options: [
      {
        text: "It is not the case that A word like 'bank' can refer to multiple meanings, so context is needed to infer the intended sense.",

        isCorrect: false,
      },
      {
        text: "It is not the case that The word 'it' in a sentence can require context to determine what entity it refers to.",
        isCorrect: false,
      },
      {
        text: "The word 'station' can appear in transportation, media, or military-like contexts, among others.",
        isCorrect: true,
      },
      {
        text: "These examples show that a fixed context-free representation is always sufficient as long as the vocabulary is large enough.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture used ambiguity examples to motivate contextual embeddings and self-attention. A larger vocabulary alone does not solve these issues, because the same surface word can still have several different roles or meanings across contexts.",
  },
  {
    id: "mit15773-l7-q25",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose a sentence has standalone embeddings \\(\\mathbf{w}_1, \\dots, \\mathbf{w}_n\\). In the lecture's intuition for contextualizing token \\(j\\), which statements are correct?",
    options: [
      {
        text: "It is not the case that The contextual embedding for token \\(j\\) is formed from all token embeddings in the sentence, not only from neighboring words.",

        isCorrect: false,
      },
      {
        text: "It is not the case that The weights used for token \\(j\\) reflect how much attention token \\(j\\) should pay to each other token.",
        isCorrect: false,
      },
      {
        text: "The same procedure can be applied to every target token, producing one contextual embedding per input token.",
        isCorrect: true,
      },
      {
        text: "Once one token has been contextualized, the rest of the tokens no longer need their own contextual representations.",
        isCorrect: false,
      },
    ],
    explanation:
      "Self-attention is applied position by position, so every token gets its own contextualized representation. Each target token may care about a different pattern of words, so the weight distribution is generally different across target positions.",
  },
  {
    id: "mit15773-l7-q26",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly reflect the geometric intuition behind using dot products as compatibility scores?",
    options: [
      {
        text: "It is not the case that If two vectors point in similar directions, their dot product tends to be larger than if they are orthogonal.",

        isCorrect: false,
      },
      {
        text: "It is not the case that If two vectors are orthogonal, their dot product is \\(0\\), which can be interpreted as little directional alignment.",
        isCorrect: false,
      },
      {
        text: "If vectors point in opposite directions, the dot product can become negative.",
        isCorrect: true,
      },
      {
        text: "A dot product can only compare vectors when both are probability distributions that already sum to 1.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture used the angle-based intuition to explain why dot products are a natural similarity or compatibility signal. They do not require the vectors to be probabilities; they only require the vectors to inhabit the same dimensional space so that pairwise products and sums are defined.",
  },
  {
    id: "mit15773-l7-q27",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "In the lecture's simplified self-attention story, why does the target word often still place substantial weight on itself?",
    options: [
      {
        text: "Because the contextual representation should usually remain anchored in the original meaning of the token, rather than replacing it entirely.",
        isCorrect: true,
      },
      {
        text: "Because the token's own standalone embedding is still an informative starting point for contextualization.",
        isCorrect: true,
      },
      {
        text: "Because the compatibility of a vector with itself is often high relative to other pairings.",
        isCorrect: true,
      },
      {
        text: "Because attention mathematically forbids a token from attending more strongly to any other word than to itself.",
        isCorrect: false,
      },
    ],
    explanation:
      "The original token is still highly relevant to its own contextual meaning, so self-attention often keeps a strong self-contribution. But it is not guaranteed to be the largest contribution in every case; the model can learn to emphasize other tokens when the context strongly demands it.",
  },
  {
    id: "mit15773-l7-q28",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements about multi-head attention are correct?",
    options: [
      {
        text: "Multiple heads allow the model to compute several different attention patterns over the same input.",
        isCorrect: true,
      },
      {
        text: "Different heads may learn to focus on different kinds of relations in the sentence.",
        isCorrect: true,
      },
      {
        text: "The lecture compared this idea to having multiple filters in a convolutional neural network.",
        isCorrect: true,
      },
      {
        text: "Using more than one head forces every head to learn the exact same compatibility pattern.",
        isCorrect: false,
      },
    ],
    explanation:
      "Multi-head attention is useful because different kinds of linguistic or structural information can coexist in a sentence. Heads are not meant to duplicate one another; ideally they specialize in complementary patterns that improve the overall representation.",
  },
  {
    id: "mit15773-l7-q29",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Why does the lecture emphasize preserving the dimensionality of the token representations after concatenate-and-project?",
    options: [
      {
        text: "It allows the output of one transformer block to be used as the input to another block without interface mismatch.",
        isCorrect: true,
      },
      {
        text: "It helps keep the architecture modular and easy to stack deeply.",
        isCorrect: true,
      },
      {
        text: "It preserves a consistent representation size while still allowing multiple heads to contribute information.",
        isCorrect: true,
      },
      {
        text: "It is required because softmax cannot be applied to any representation unless the embedding size is unchanged from the input.",
        isCorrect: false,
      },
    ],
    explanation:
      "A stable representation size is an architectural convenience that becomes very important when stacking many layers. Softmax at the end of a task head does not itself require unchanged embedding size; the main point is composability and clean layer-to-layer interfaces.",
  },
  {
    id: "mit15773-l7-q30",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "In the lecture's simplified transformer encoder story, what roles do the feed-forward layers play?",
    options: [
      {
        text: "They add non-linearity after attention-based contextualization.",
        isCorrect: true,
      },
      {
        text: "They typically act position-wise on each token representation rather than mixing tokens through recurrence.",
        isCorrect: true,
      },
      {
        text: "They can change the representation and then project it back to the standard model dimension.",
        isCorrect: true,
      },
      {
        text: "They exist only because attention cannot preserve sequence length on its own.",
        isCorrect: false,
      },
    ],
    explanation:
      "The feed-forward part is not about restoring length; attention already preserves the number of positions in this encoder setup. Instead, the feed-forward block adds expressive nonlinear transformation to each position's contextualized representation.",
  },
  {
    id: "mit15773-l7-q31",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Before positional encoding is added, why can the simplified attention architecture fail to distinguish different word orders?",
    options: [
      {
        text: "Because the compatibility calculations do not yet contain explicit information about token positions.",
        isCorrect: true,
      },
      {
        text: "Because scrambling the order of the tokens can leave the set of pairwise embedding interactions effectively unchanged in the simplified story.",
        isCorrect: true,
      },
      {
        text: "Because without order information, the model behaves more like it is processing a set than a true sequence.",
        isCorrect: true,
      },
      {
        text: "Because attention is only defined for one-hot vectors and therefore cannot model order until embeddings are removed.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture emphasized that attention over token content alone is not enough to encode order. Without a positional signal, the model can contextualize words based on co-occurring tokens, but it cannot reliably tell who came before or after whom.",
  },
  {
    id: "mit15773-l7-q32",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the lecture's learned positional-embedding approach?",
    options: [
      {
        text: "Each possible position in the input sequence has its own learned embedding vector.",
        isCorrect: true,
      },
      {
        text: "The position embedding is added to the token embedding to form the positional input embedding.",
        isCorrect: true,
      },
      {
        text: "The position embedding table depends on the maximum sequence length assumed by the model.",
        isCorrect: true,
      },
      {
        text: "The same position embedding must differ across vocabulary words even before training begins.",
        isCorrect: false,
      },
    ],
    explanation:
      "In the lecture's simple version, position embeddings depend on positions, not on which word occupies those positions. The resulting sum differs across words because different token embeddings are added to the same position embedding when needed.",
  },
  {
    id: "mit15773-l7-q33",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose the word 'cat' has embedding \\((0.5, 7.1)\\) and position \\(0\\) has embedding \\((1.3, 3.9)\\). Which statements are correct about the resulting positional input embedding?",
    options: [
      {
        text: "The summed positional input embedding is \\((1.8, 11.0)\\).",
        isCorrect: true,
      },
      {
        text: "The resulting vector has the same dimensionality as the original token embedding and the position embedding.",
        isCorrect: true,
      },
      {
        text: "If 'cat' appeared at another position with a different position embedding, its positional input embedding would generally change.",
        isCorrect: true,
      },
      {
        text: "Adding the two vectors removes all information about which token was originally present.",
        isCorrect: false,
      },
    ],
    explanation:
      "The example is a simple coordinate-wise vector sum. Adding position information changes the representation, but it does not erase token identity; rather, it blends token identity and location into a single input vector for the transformer.",
  },
  {
    id: "mit15773-l7-q34",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements are correct about repeated words after positional encoding?",
    options: [
      {
        text: "The same word appearing at multiple positions can receive different positional input embeddings because the added position vectors differ.",
        isCorrect: true,
      },
      {
        text: "This is one way the model can begin to distinguish different occurrences of the same token in a sentence.",
        isCorrect: true,
      },
      {
        text: "The standalone word embedding component can remain the same across repeated occurrences of the word.",
        isCorrect: true,
      },
      {
        text: "Repeated words become impossible to compare once positional encoding is added, because they no longer share any common representation component.",
        isCorrect: false,
      },
    ],
    explanation:
      "Repeated words still share the same standalone token embedding, so there is still a common semantic anchor. Positional encoding simply adds a location-specific signal, allowing the model to distinguish occurrences while preserving some shared identity information.",
  },
  {
    id: "mit15773-l7-q35",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the transformer encoder at the conceptual level used in this lecture?",
    options: [
      {
        text: "It begins with token embeddings, augments them with positional information, and then applies contextualization through attention and feed-forward processing.",
        isCorrect: true,
      },
      {
        text: "Its outputs can preserve the same number of positions as its inputs.",
        isCorrect: true,
      },
      {
        text: "Because encoder blocks preserve the interface shape, they can be stacked repeatedly to increase modeling capacity.",
        isCorrect: true,
      },
      {
        text: "The lecture's first-pass conceptual encoder already fully explained every detail of residual connections, layer normalization, and linear projections into query, key, and value spaces.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture gave a conceptual introduction focused on self-attention, order, and token-aligned outputs. It explicitly postponed several important refinements and implementation details, such as residual connections, layer normalization, and projection into different internal spaces.",
  },
  {
    id: "mit15773-l7-q36",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "In the slot-filling model built on top of a transformer encoder, which statements are correct about the final prediction stage?",
    options: [
      {
        text: "Each token position can have its own softmax over the slot label vocabulary.",
        isCorrect: true,
      },
      {
        text: "A dense layer with a ReLU activation can be inserted before the token-level softmax classifiers.",
        isCorrect: true,
      },
      {
        text: "The model can still be trained end-to-end even though predictions are made at each token position.",
        isCorrect: true,
      },
      {
        text: "Once the encoder is used, token-level predictions must be replaced by a single sentence-level output.",
        isCorrect: false,
      },
    ],
    explanation:
      "The encoder provides contextual token features, and the task-specific head converts those into token-level slot predictions. End-to-end learning is still possible because all layers, including the encoder and classifier head, can be optimized jointly through backpropagation.",
  },
  {
    id: "mit15773-l7-q37",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about the Keras-style setup discussed in the lecture are correct?",
    options: [
      {
        text: "The input queries are vectorized into integer token sequences before embeddings are applied.",
        isCorrect: true,
      },
      {
        text: "The output slot labels also need vectorization because the model is trained on token-level labels represented in a machine-readable form.",
        isCorrect: true,
      },
      {
        text: "For the slot-label side, preserving punctuation and case-like distinctions in the label strings can matter, so disabling standardization can be appropriate.",
        isCorrect: true,
      },
      {
        text: "The vectorization step automatically solves contextual disambiguation without needing a transformer encoder.",
        isCorrect: false,
      },
    ],
    explanation:
      "Vectorization prepares text and labels for modeling, but it is only a preprocessing step. Contextual understanding still depends on the architecture, such as the transformer encoder, not merely on mapping strings to integers.",
  },
  {
    id: "mit15773-l7-q38",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Why might the lecture choose a maximum query length such as 30 for the ATIS example?",
    options: [
      {
        text: "It provides a fixed input length so that batches can be formed and processed consistently.",
        isCorrect: true,
      },
      {
        text: "Queries shorter than the maximum can be padded.",
        isCorrect: true,
      },
      {
        text: "Queries longer than the maximum can be truncated.",
        isCorrect: true,
      },
      {
        text: "Choosing a maximum length mathematically guarantees that no relevant information is ever removed from any long query.",
        isCorrect: false,
      },
    ],
    explanation:
      "A maximum sequence length is a practical modeling choice, not a guarantee of perfect information retention. Padding and truncation make tensor construction manageable, but truncation can discard content if the maximum length is chosen too small.",
  },
  {
    id: "mit15773-l7-q39",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about parameters and learning in the lecture's transformer discussion are correct?",
    options: [
      {
        text: "Token embeddings and positional embeddings can both be learned through backpropagation.",
        isCorrect: true,
      },
      {
        text: "The lecture noted that some deeper internal details of attention, including additional learned projections, would be covered later.",
        isCorrect: true,
      },
      {
        text: "Dense layers after the encoder contain trainable weights that can be optimized for the slot-filling task.",
        isCorrect: true,
      },
      {
        text: "Because the lecture first explained attention using intuitive weighted averages, no trainable parameters are needed anywhere in the overall model.",
        isCorrect: false,
      },
    ],
    explanation:
      "The initial weighted-average explanation was meant to build intuition, not to claim that the full model has no learnable structure. In practice, embeddings, projections, feed-forward layers, and task heads all contain trainable parameters, even if some projection details were postponed to the next lecture.",
  },
  {
    id: "mit15773-l7-q40",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which of the following are good high-level conclusions from this lecture's first-pass treatment of transformers?",
    options: [
      {
        text: "Self-attention provides a principled way to build contextual token representations from interactions among tokens in the same sequence.",
        isCorrect: true,
      },
      {
        text: "Positional encoding is needed if we want a transformer encoder to model order-sensitive language phenomena.",
        isCorrect: true,
      },
      {
        text: "Maintaining same-sized inputs and outputs within encoder blocks makes deep stacking natural.",
        isCorrect: true,
      },
      {
        text: "The lecture established that a transformer encoder alone is sufficient to solve every NLP problem without any task-specific output head.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture's core message was that transformers elegantly combine contextualization, order awareness, and shape preservation. But a transformer encoder is usually only part of a larger task model, and downstream problems still need task-specific heads or additional structure to produce the desired outputs.",
  },
];
