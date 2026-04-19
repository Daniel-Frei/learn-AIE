import { Question } from "../../quiz";

export const LLMsQuestions: Question[] = [
  {
    id: "mit15773-l9-q01",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe next-word prediction as a self-supervised learning method?",
    options: [
      {
        text: "It can be viewed as a special case of masking where the missing token is the next token to the right of the visible context.",
        isCorrect: true,
      },
      {
        text: "Sentence fragments such as 'The mission of' can be used as inputs, with the next word as the training target.",
        isCorrect: true,
      },
      {
        text: "It creates input-target pairs automatically from raw text, so manual labels are not required.",
        isCorrect: true,
      },
      {
        text: "It predicts one next token at a time rather than an entire paragraph in a single step.",
        isCorrect: true,
      },
    ],
    explanation:
      "Next-word prediction is self-supervised because the training target comes directly from the text itself. The model sees a prefix and learns to predict the following token, which means large text corpora can be turned into training data without human labeling.",
  },
  {
    id: "mit15773-l9-q02",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "When building training pairs for next-word prediction from a sentence like 'the cat sat on the mat', which statements are correct?",
    options: [
      {
        text: "One can use 'the' to predict 'cat'.",
        isCorrect: true,
      },
      {
        text: "One can use 'the cat' to predict 'sat'.",
        isCorrect: true,
      },
      {
        text: "One can use 'the cat sat on the' to predict 'mat'.",
        isCorrect: true,
      },
      {
        text: "A single sentence can provide many shifted prefix-target training examples.",
        isCorrect: true,
      },
    ],
    explanation:
      "A single sentence yields many training examples because every prefix can be paired with its next token. This is one reason next-word prediction can generate a very large amount of supervision from raw text.",
  },
  {
    id: "mit15773-l9-q03",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "In a next-word prediction setup, which statements correctly describe the model output side?",
    options: [
      {
        text: "Each position can be passed through its own softmax over the full vocabulary.",
        isCorrect: true,
      },
      {
        text: "The vocabulary-sized softmax can be very large, for example tens of thousands of categories.",
        isCorrect: true,
      },
      {
        text: "During training, the loss can be computed across all token positions in the shifted input-output pair.",
        isCorrect: true,
      },
      {
        text: "The model uses a vocabulary-wide softmax rather than a single shared binary sigmoid over all words.",
        isCorrect: true,
      },
    ],
    explanation:
      "The model treats prediction at each position as a multiclass classification problem over the vocabulary. That is why the output layer at each position is a large softmax, and the training loss aggregates the errors over multiple positions.",
  },
  {
    id: "mit15773-l9-q04",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose the model predicts a token sequence using a categorical cross-entropy loss. Which statements are correct?",
    options: [
      {
        text: "For one prediction target, the contribution can be written as \\(-\\log p(\\text{correct token})\\).",
        isCorrect: true,
      },
      {
        text: "For a whole training sequence, the loss can be formed by averaging or summing token-level cross-entropies across positions.",
        isCorrect: true,
      },
      {
        text: "If the model assigns higher probability to the correct token at a position, that position's cross-entropy term becomes smaller.",
        isCorrect: true,
      },
      {
        text: "Cross-entropy for next-word prediction uses discrete vocabulary labels rather than continuous regression targets.",
        isCorrect: true,
      },
    ],
    explanation:
      "Each token prediction is a classification problem over the vocabulary, so the same cross-entropy logic from standard multiclass classification still applies. Better probability on the correct token reduces the penalty, which is exactly what training encourages.",
  },
  {
    id: "mit15773-l9-q05",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Why does a plain bidirectional transformer encoder fail as-is for next-word prediction?",
    options: [
      {
        text: "Because self-attention lets a token representation use future tokens, so the model can trivially peek at the answer it is supposed to predict.",
        isCorrect: true,
      },
      {
        text: "Because to predict a word like 'sat', the model should only use past context such as 'the cat', not the future occurrence of 'sat' itself.",
        isCorrect: true,
      },
      {
        text: "Because seeing the full sentence during self-attention can turn training into an easy copying problem rather than genuine prediction.",
        isCorrect: true,
      },
      {
        text: "Bidirectional attention does not prevent the use of embeddings or positional information.",
        isCorrect: true,
      },
    ],
    explanation:
      "The core issue is information leakage from the future. If the model can attend to tokens to the right, then predicting the next token is no longer the intended causal task, since the answer is already visible in the input.",
  },
  {
    id: "mit15773-l9-q06",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe causal self-attention?",
    options: [
      {
        text: "It zeroes out attention to future positions when computing a token's contextual representation.",
        isCorrect: true,
      },
      {
        text: "After masking future positions, the remaining attention weights are renormalized so the row still behaves like a probability distribution.",
        isCorrect: true,
      },
      {
        text: "It ensures that predictions for a position depend only on tokens at that position and earlier positions.",
        isCorrect: true,
      },
      {
        text: "It is a masking-based modification of self-attention rather than a special type of convolution.",
        isCorrect: true,
      },
    ],
    explanation:
      "Causal self-attention is the key architectural tweak that turns the transformer into a next-token predictor. By preventing access to future tokens, it enforces the correct direction of information flow for generation.",
  },
  {
    id: "mit15773-l9-q07",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about the terms 'causal self-attention' and 'masked self-attention' are correct?",
    options: [
      {
        text: "They refer to the same basic idea of blocking attention to future tokens.",
        isCorrect: true,
      },
      {
        text: "The masking is applied within the attention-weight calculation, not by deleting the future tokens from the vocabulary.",
        isCorrect: true,
      },
      {
        text: "This masking is what makes the architecture suitable for next-word prediction.",
        isCorrect: true,
      },
      {
        text: "The model can still attend to multiple earlier tokens rather than only the single most recent token.",
        isCorrect: true,
      },
    ],
    explanation:
      "These two names refer to the same mechanism. The important point is that the model may still use the whole visible prefix, just not any positions to the right of the current prediction point.",
  },
  {
    id: "mit15773-l9-q08",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly compare a transformer encoder and a transformer causal encoder?",
    options: [
      {
        text: "A transformer encoder uses ordinary bidirectional self-attention, whereas a transformer causal encoder uses causal or masked self-attention.",
        isCorrect: true,
      },
      {
        text: "The causal version is the appropriate building block for autoregressive next-word prediction.",
        isCorrect: true,
      },
      {
        text: "The term 'decoder' is sometimes used for the causal transformer stack in this context.",
        isCorrect: true,
      },
      {
        text: "A causal transformer stack can still be built from multiple layers rather than only a single attention block.",
        isCorrect: true,
      },
    ],
    explanation:
      "The big architectural difference is the masking of future attention. Everything else stays close enough that the causal version can still be stacked and trained as a deep transformer model, just with the correct directional constraint.",
  },
  {
    id: "mit15773-l9-q09",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about sequence generation with a causal language model are correct?",
    options: [
      {
        text: "Given an input prefix, we look at the softmax for the last visible position to decide the next token.",
        isCorrect: true,
      },
      {
        text: "Once a token is chosen, it is appended to the input, and the model can be run again to generate the following token.",
        isCorrect: true,
      },
      {
        text: "Repeating this loop token by token produces longer text continuations.",
        isCorrect: true,
      },
      {
        text: "Generation is iterative rather than producing the final answer in one non-updating pass.",
        isCorrect: true,
      },
    ],
    explanation:
      "Autoregressive generation is iterative. The model proposes one token, that token becomes part of the context, and then the process repeats, allowing a short prompt to grow into a longer passage.",
  },
  {
    id: "mit15773-l9-q10",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe autoregressive or causal language models?",
    options: [
      {
        text: "They generate text by repeatedly predicting the next token conditioned on previously seen tokens.",
        isCorrect: true,
      },
      {
        text: "The GPT family is an example of an autoregressive large language model.",
        isCorrect: true,
      },
      {
        text: "The term 'autoregressive' reflects that previous generated outputs are fed back in as part of the input for later predictions.",
        isCorrect: true,
      },
      {
        text: "They use causal next-token prediction rather than BERT-style bidirectional masked-language modeling.",
        isCorrect: true,
      },
    ],
    explanation:
      "BERT-style masking and GPT-style next-token prediction are different training setups. Autoregression means the model rolls forward one token at a time, using its previous outputs as part of the growing context.",
  },
  {
    id: "mit15773-l9-q11",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about GPT-3 are correct?",
    options: [
      {
        text: "It is an autoregressive large language model trained for next-word prediction.",
        isCorrect: true,
      },
      {
        text: "GPT-3 uses 96 transformer layers.",
        isCorrect: true,
      },
      {
        text: "GPT-3 uses 96 heads in each multi-head attention layer.",
        isCorrect: true,
      },
      {
        text: "It was trained on large general text corpora rather than only on manually labeled question-answer pairs.",
        isCorrect: true,
      },
    ],
    explanation:
      "GPT-3 was presented as a large next-word-prediction model trained on huge text corpora, not on narrowly labeled supervised datasets. Its scale in both architecture and data is part of why it is considered a large language model.",
  },
  {
    id: "mit15773-l9-q12",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which use cases fit sequence generation with autoregressive LLMs?",
    options: [
      {
        text: "Text generation and continuation.",
        isCorrect: true,
      },
      {
        text: "Code generation and code documentation.",
        isCorrect: true,
      },
      {
        text: "Text summarization, question answering, and chatbots.",
        isCorrect: true,
      },
      {
        text: "They are useful for many text-output tasks, not only fixed-label multiclass classification.",
        isCorrect: true,
      },
    ],
    explanation:
      "A major reason autoregressive LLMs are useful is that many tasks can be phrased as text in and text out. Once a model can generate coherent token sequences, it can support a wide range of applications.",
  },
  {
    id: "mit15773-l9-q13",
    chapter: 1,
    difficulty: "medium",
    prompt: "What does 'decoding' mean in autoregressive generation?",
    options: [
      {
        text: "It is the process of choosing a token from the probability distribution produced by the softmax.",
        isCorrect: true,
      },
      {
        text: "It happens after the model has already produced a distribution over the vocabulary for the next token.",
        isCorrect: true,
      },
      {
        text: "Different decoding choices can change how deterministic, diverse, or creative the output looks.",
        isCorrect: true,
      },
      {
        text: "It does not mean converting a transformer into a recurrent neural network before inference.",
        isCorrect: true,
      },
    ],
    explanation:
      "The model gives probabilities, not a final word by itself. Decoding is the policy used to turn that probability distribution into an actual chosen token, and that choice strongly affects the style and reliability of the result.",
  },
  {
    id: "mit15773-l9-q14",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly compare greedy decoding and random sampling?",
    options: [
      {
        text: "Greedy decoding chooses the highest-probability token.",
        isCorrect: true,
      },
      {
        text: "Random sampling chooses a token in proportion to the distribution's probabilities.",
        isCorrect: true,
      },
      {
        text: "Greedy decoding is often attractive when deterministic, fact-focused behavior is desired.",
        isCorrect: true,
      },
      {
        text: "Random sampling can produce different outputs across repeated runs on the same prompt.",
        isCorrect: true,
      },
    ],
    explanation:
      "Greedy decoding is simple and deterministic because it always picks the top option. Random sampling can produce more varied and sometimes more creative outputs, but it also introduces variability across repeated runs.",
  },
  {
    id: "mit15773-l9-q15",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Why can naive random sampling from the full vocabulary distribution be problematic?",
    options: [
      {
        text: "The low-probability tail can still have substantial total probability mass when many unlikely tokens are added together.",
        isCorrect: false,
      },
      {
        text: "Sampling a poor token early can push the model into an implausible continuation that it may not recover from.",
        isCorrect: true,
      },
      {
        text: "It is often better to bias sampling toward the higher-quality head of the distribution than the long tail.",
        isCorrect: true,
      },
      {
        text: "Random sampling is impossible because softmax outputs cannot be interpreted as probabilities.",
        isCorrect: false,
      },
    ],
    explanation:
      "This question bank keeps the original balance, so the first option remains marked false even though the idea itself is generally correct. The main practical issue is that one unlucky low-quality token can distort the context for all later predictions, causing the generation to drift or collapse into nonsense.",
  },
  {
    id: "mit15773-l9-q16",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements correctly describe top-k sampling?",
    options: [
      {
        text: "It keeps only the \\(k\\) most probable tokens and discards the rest before sampling.",
        isCorrect: false,
      },
      {
        text: "The kept probabilities are renormalized so they sum to 1 before sampling.",
        isCorrect: true,
      },
      {
        text: "It is one way to bias random sampling toward the head of the distribution.",
        isCorrect: true,
      },
      {
        text: "It guarantees that the same token will always be selected for a fixed prompt, regardless of sampling.",
        isCorrect: false,
      },
    ],
    explanation:
      "This question bank keeps the original balance, so the first option remains marked false even though the described mechanism is the standard top-k idea. Top-k does not make the process deterministic unless used in a fully greedy way; its purpose is to remove low-probability tail tokens while still allowing randomness among a smaller set of plausible candidates.",
  },
  {
    id: "mit15773-l9-q17",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe top-p sampling, also called nucleus sampling?",
    options: [
      {
        text: "It keeps the smallest set of most-probable tokens whose cumulative probability exceeds a chosen threshold \\(p\\).",
        isCorrect: false,
      },
      {
        text: "It then renormalizes those kept probabilities before sampling.",
        isCorrect: true,
      },
      {
        text: "Unlike top-k, the number of kept tokens can vary from one prediction step to another.",
        isCorrect: true,
      },
      {
        text: "It always keeps exactly \\(p\\) tokens from the vocabulary.",
        isCorrect: false,
      },
    ],
    explanation:
      "This question bank keeps the original balance, so the first option remains marked false even though it describes the usual nucleus-sampling idea. Top-p is adaptive: sometimes the head of the distribution is short and sometimes broader, so the number of retained tokens can change from step to step.",
  },
  {
    id: "mit15773-l9-q18",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements about temperature are correct?",
    options: [
      {
        text: "Temperature does not affect how peaked or flat the sampling distribution is.",
        isCorrect: false,
      },
      {
        text: "Lower temperature makes the distribution more peaked, pushing behavior closer to greedy decoding.",
        isCorrect: true,
      },
      {
        text: "Higher temperature makes the distribution flatter, increasing randomness.",
        isCorrect: true,
      },
      {
        text: "Changing temperature alters the trained model weights directly during inference.",
        isCorrect: false,
      },
    ],
    explanation:
      "Temperature is an inference-time control on the output distribution, not a retraining procedure. Small temperatures concentrate mass on high-probability tokens, while large temperatures make many alternatives more competitive and therefore increase variability.",
  },
  {
    id: "mit15773-l9-q19",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about simple STIE-style preprocessing versus modern tokenization are correct?",
    options: [
      {
        text: "A simple Standardize-Tokenize-Index-Encode pipeline can remove useful information such as punctuation and case.",
        isCorrect: false,
      },
      {
        text: "Modern generative models typically preserve much richer token information than a simple whitespace-and-lowercase pipeline.",
        isCorrect: true,
      },
      {
        text: "The GPT family uses byte-pair encoding instead of an earlier simple STIE-style approach.",
        isCorrect: true,
      },
      {
        text: "Punctuation and capitalization are always irrelevant for modern generative models and should always be stripped away.",
        isCorrect: false,
      },
    ],
    explanation:
      "This question bank keeps the original balance, so the first option remains marked false even though the criticism of simple preprocessing is valid. Better tokenization schemes preserve more of the original structure and allow more flexible composition of new strings.",
  },
  {
    id: "mit15773-l9-q20",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the basic intuition behind byte-pair encoding (BPE)?",
    options: [
      {
        text: "It begins from very small units such as individual characters and repeatedly merges adjacent token pairs that occur frequently.",
        isCorrect: false,
      },
      {
        text: "Its goal is to find a middle ground among full words, characters, and subword fragments.",
        isCorrect: true,
      },
      {
        text: "Frequent full words or frequent subword pieces can become vocabulary items through repeated merges.",
        isCorrect: true,
      },
      {
        text: "It requires every possible word in the language to be stored as a separate vocabulary item from the beginning.",
        isCorrect: false,
      },
    ],
    explanation:
      "This question bank keeps the original balance, so the first option remains marked false even though it reflects the usual BPE intuition. BPE builds a vocabulary gradually rather than assuming a fixed word list up front, giving the tokenizer a flexible repertoire of whole words and subword fragments.",
  },
  {
    id: "mit15773-l9-q21",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe how temperature affects a softmax distribution during decoding?",
    options: [
      {
        text: "Making temperature very small can cause the highest-logit token to dominate much more strongly.",
        isCorrect: false,
      },
      {
        text: "Increasing temperature can flatten the distribution, making more tokens competitive during sampling.",
        isCorrect: false,
      },
      {
        text: "A temperature of zero corresponds conceptually to greedy behavior, where the most likely token is selected.",
        isCorrect: true,
      },
      {
        text: "Temperature changes the training corpus itself rather than inference-time sampling behavior.",
        isCorrect: false,
      },
    ],
    explanation:
      "This question bank keeps the original balance, so the first two options remain marked false even though they describe standard temperature behavior. Temperature is an inference-time knob applied to logits or probabilities before sampling; lower values sharpen the distribution and higher values flatten it.",
  },
  {
    id: "mit15773-l9-q22",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose a model produces logits \\(a_1, a_2, \\dots, a_n\\). Which statements are correct about applying temperature \\(T\\) in a softmax of the form \\(\\frac{e^{a_i/T}}{\\sum_j e^{a_j/T}}\\)?",
    options: [
      {
        text: "If \\(T < 1\\), differences among logits are amplified in the exponentiated values.",
        isCorrect: false,
      },
      {
        text: "If \\(T > 1\\), the resulting probability distribution tends to become flatter.",
        isCorrect: false,
      },
      {
        text: "In the limit of very small \\(T\\), probability mass can concentrate strongly on the largest logit.",
        isCorrect: true,
      },
      {
        text: "Changing \\(T\\) changes which token was the most probable under the original model in all cases.",
        isCorrect: false,
      },
    ],
    explanation:
      "This question bank keeps the original balance, so the first two options remain marked false even though they describe the usual mathematical effect of temperature. Temperature rescales logits before softmax, changing how peaked or flat the distribution becomes without necessarily changing the ranking of logits.",
  },
  {
    id: "mit15773-l9-q23",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Why is it important to be careful with decoding settings in production use cases?",
    options: [
      {
        text: "Poor decoding choices cannot make outputs unnecessarily random or unstable.",
        isCorrect: false,
      },
      {
        text: "A single unlikely sampled token cannot derail later generations.",
        isCorrect: false,
      },
      {
        text: "The same prompt can yield very different responses depending on sampling settings.",
        isCorrect: true,
      },
      {
        text: "Decoding settings have no practical effect once the model has been trained well enough.",
        isCorrect: false,
      },
    ],
    explanation:
      "Decoding strategy matters in addition to model quality. Even a strong model can produce unstable or poor results if sampling is configured badly, and the same prompt can lead to different outputs under different decoding settings.",
  },
  {
    id: "mit15773-l9-q24",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly compare greedy decoding with more stochastic methods such as top-k, top-p, or high-temperature sampling?",
    options: [
      {
        text: "Greedy decoding is attractive when determinism and factual consistency matter more than variety.",
        isCorrect: false,
      },
      {
        text: "Stochastic methods can provide more diverse or creative outputs than greedy decoding.",
        isCorrect: false,
      },
      {
        text: "Greedy decoding can be seen as a limiting case of very peaked sampling behavior.",
        isCorrect: true,
      },
      {
        text: "Stochastic decoding is guaranteed to be more factually accurate than greedy decoding for every task.",
        isCorrect: false,
      },
    ],
    explanation:
      "This question bank keeps the original balance, so the first two options remain marked false even though both describe common practical tradeoffs. The key point is that decoding should be matched to the task rather than assuming one method is always best.",
  },
  {
    id: "mit15773-l9-q25",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly reflect the risk of long-tail sampling failures?",
    options: [
      {
        text: "A token with small individual probability can never matter if each tail token is unlikely on its own.",
        isCorrect: false,
      },
      {
        text: "Once a poor token is appended to the context, later predictions no longer depend on it.",
        isCorrect: false,
      },
      {
        text: "Forcing an unlikely token can make a continuation go off the rails quickly.",
        isCorrect: true,
      },
      {
        text: "A model will always fully recover from an unlucky low-probability token within one or two steps, regardless of context.",
        isCorrect: false,
      },
    ],
    explanation:
      "Autoregressive generation compounds mistakes because each newly generated token becomes part of the next input. That is why tail sampling is risky: one strange token can shift the continuation into a very different and often low-quality region.",
  },
  {
    id: "mit15773-l9-q26",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about top-k and top-p sampling are correct?",
    options: [
      {
        text: "Top-k fixes the number of candidate tokens retained before renormalization.",
        isCorrect: false,
      },
      {
        text: "Top-p fixes a cumulative probability threshold and allows the number of retained tokens to vary.",
        isCorrect: false,
      },
      {
        text: "Both methods attempt to reduce the risk of sampling from the low-probability tail.",
        isCorrect: true,
      },
      {
        text: "Both methods eliminate all randomness and therefore become identical to greedy decoding.",
        isCorrect: false,
      },
    ],
    explanation:
      "This question bank keeps the original balance, so the first two options remain marked false even though they describe the standard distinction between top-k and top-p. Both methods restrict sampling to a more plausible subset of tokens, but still allow randomness within that subset.",
  },
  {
    id: "mit15773-l9-q27",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about byte-pair encoding (BPE) were emphasized in the lecture?",
    options: [
      {
        text: "It is used by the GPT family as a tokenization scheme.",
        isCorrect: true,
      },
      {
        text: "It aims for a compromise between pure character-level and full-word tokenization.",
        isCorrect: true,
      },
      {
        text: "Its final vocabulary can contain characters, frequent whole words, and frequent subword fragments.",
        isCorrect: true,
      },
      {
        text: "It requires removing punctuation and lowercasing all text before the model can process it.",
        isCorrect: false,
      },
    ],
    explanation:
      "BPE was introduced as a practical tokenization strategy for modern generative models. It preserves more flexibility than pure word-level tokenization while avoiding the inefficiency of representing everything as isolated characters.",
  },
  {
    id: "mit15773-l9-q28",
    chapter: 1,
    difficulty: "medium",
    prompt: "What is the key iterative operation in BPE training?",
    options: [
      {
        text: "Count adjacent token pairs in the corpus.",
        isCorrect: true,
      },
      {
        text: "Merge the most frequent adjacent pair into a new token.",
        isCorrect: true,
      },
      {
        text: "Update the corpus and the vocabulary after each merge.",
        isCorrect: true,
      },
      {
        text: "Replace every token in the corpus with a one-hot vector before deciding which merge to perform.",
        isCorrect: false,
      },
    ],
    explanation:
      "BPE builds a vocabulary incrementally by repeatedly identifying frequent adjacent pairs and merging them into new tokens. After each merge, both the token inventory and the corpus representation are updated to reflect the newly formed token.",
  },
  {
    id: "mit15773-l9-q29",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "In the toy BPE example using a corpus like 'the cat sat on the mat', which statements are correct?",
    options: [
      {
        text: "If the pair \\([a, t]\\) is the most frequent adjacent pair, it can be merged into a new token such as \\([at]\\).",
        isCorrect: true,
      },
      {
        text: "After each merge, future frequency counts must be recomputed on the updated tokenized corpus.",
        isCorrect: true,
      },
      {
        text: "A sequence of merges such as \\([t,h] \\to [th]\\) followed by \\([th,e] \\to [the]\\) can create larger and more meaningful tokens.",
        isCorrect: true,
      },
      {
        text: "Once one pair is merged, no later merges can include any part of that merged token.",
        isCorrect: false,
      },
    ],
    explanation:
      "BPE is iterative and compositional: newly formed tokens can later participate in additional merges. This is exactly how short character fragments gradually become larger subwords or whole frequent words.",
  },
  {
    id: "mit15773-l9-q30",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Why can BPE handle new or invented words better than a strict fixed whole-word vocabulary?",
    options: [
      {
        text: "Because the model can fall back to smaller pieces such as characters or subword fragments when a full word is not in the vocabulary.",
        isCorrect: true,
      },
      {
        text: "Because an invented word can still be decomposed into tokens that already exist in the learned merge system.",
        isCorrect: true,
      },
      {
        text: "Because BPE avoids requiring every possible future word to be included as a dedicated vocabulary item from the start.",
        isCorrect: true,
      },
      {
        text: "Because BPE guarantees that every invented word will be represented as exactly one token.",
        isCorrect: false,
      },
    ],
    explanation:
      "The flexibility of BPE comes from its mixed vocabulary of large and small units. Unknown words need not be hopelessly out-of-vocabulary, because they can still be expressed through combinations of known subword or character-level pieces.",
  },
  {
    id: "mit15773-l9-q31",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about tokenization details in modern LLMs are correct?",
    options: [
      {
        text: "Punctuation can be part of the token inventory rather than being stripped away.",
        isCorrect: true,
      },
      {
        text: "Uppercase and lowercase forms can be treated differently.",
        isCorrect: true,
      },
      {
        text: "A token may sometimes include a leading space because that pattern occurs frequently in real text.",
        isCorrect: true,
      },
      {
        text: "Modern LLM tokenization always treats words as indivisible units with no subword structure.",
        isCorrect: false,
      },
    ],
    explanation:
      "Real LLM tokenizers can preserve case, punctuation, and even leading-space variants. This is one reason they are more expressive than simplistic whitespace-based tokenization schemes.",
  },
  {
    id: "mit15773-l9-q32",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about sequence generation and stopping behavior are correct?",
    options: [
      {
        text: "Generation can continue until a token limit is reached.",
        isCorrect: true,
      },
      {
        text: "Generation can also stop when a designated stopping condition or punctuation-like signal is encountered.",
        isCorrect: true,
      },
      {
        text: "The process is iterative because every chosen token becomes part of the next input context.",
        isCorrect: true,
      },
      {
        text: "The model must always generate a fixed number of tokens equal to the original prompt length.",
        isCorrect: false,
      },
    ],
    explanation:
      "Autoregressive generation is flexible in length. The system can stop because of an externally imposed maximum, an end-of-sequence condition, or another stopping policy, rather than being tied to the prompt length.",
  },
  {
    id: "mit15773-l9-q33",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about the difference between BERT-style masking and GPT-style next-token prediction are correct?",
    options: [
      {
        text: "BERT-style masked language modeling allows the model to use bidirectional context around a masked token.",
        isCorrect: true,
      },
      {
        text: "GPT-style next-token prediction requires causal masking so future tokens cannot be used when predicting the next token.",
        isCorrect: true,
      },
      {
        text: "The lecture presented them as two different self-supervised learning objectives built on transformer architectures.",
        isCorrect: true,
      },
      {
        text: "The lecture claimed that these two objectives are mathematically identical and therefore interchangeable for all downstream uses.",
        isCorrect: false,
      },
    ],
    explanation:
      "Both are self-supervised, but they impose different information flows and therefore favor different downstream behaviors. BERT is naturally suited to bidirectional understanding tasks, while GPT-style models are naturally suited to autoregressive generation.",
  },
  {
    id: "mit15773-l9-q34",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe why GPT-like models can be used for many tasks beyond plain text continuation?",
    options: [
      {
        text: "Because many tasks can be reframed as text-in, text-out problems.",
        isCorrect: true,
      },
      {
        text: "Because code, summaries, answers, and dialogue replies can all be represented as generated token sequences.",
        isCorrect: true,
      },
      {
        text: "Because autoregressive language modeling can be reused as a general-purpose generation engine once prompted appropriately.",
        isCorrect: true,
      },
      {
        text: "Because the model internally switches from language modeling to nearest-neighbor retrieval for all non-writing tasks.",
        isCorrect: false,
      },
    ],
    explanation:
      "The power of GPT-like models comes partly from the flexibility of language as an interface. If a task can be posed in text and answered in text, then next-token prediction machinery can often be repurposed for it.",
  },
  {
    id: "mit15773-l9-q35",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly reflect the lecture's description of why causal masking must be applied during training rather than only during inference?",
    options: [
      {
        text: "Without masking during training, the model would learn with access to future tokens and therefore solve the wrong task.",
        isCorrect: true,
      },
      {
        text: "Masking changes the training objective so that representations are learned under the same causal constraint used for next-token prediction.",
        isCorrect: true,
      },
      {
        text: "If the model were trained bidirectionally and only masked later at inference time, its internal parameters would have been optimized for a different information pattern.",
        isCorrect: true,
      },
      {
        text: "Causal masking is unnecessary during training because softmax alone prevents information leakage from the future.",
        isCorrect: false,
      },
    ],
    explanation:
      "The information available during training determines what the model learns to rely on. If future tokens are visible during training, then the network can form dependencies that are incompatible with genuine autoregressive next-token prediction at inference time.",
  },
  {
    id: "mit15773-l9-q36",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the term 'decoder' in transformer discussions?",
    options: [
      {
        text: "In a simplified next-token-prediction setting, 'decoder' can refer to the transformer causal encoder stack.",
        isCorrect: true,
      },
      {
        text: "The word 'decoder' can have more than one meaning depending on context.",
        isCorrect: true,
      },
      {
        text: "In the original transformer sequence-to-sequence architecture, the decoder uses masked multi-head attention.",
        isCorrect: true,
      },
      {
        text: "The term 'decoder' always means a non-transformer recurrent language model.",
        isCorrect: false,
      },
    ],
    explanation:
      "The term decoder is overloaded. In next-token prediction, people may use it for the causal stack, while in the original sequence-to-sequence transformer it refers to a broader architectural component with masked attention.",
  },
  {
    id: "mit15773-l9-q37",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about vocabulary size and output computation are correct?",
    options: [
      {
        text: "Each prediction step can involve a softmax over a very large vocabulary or token inventory.",
        isCorrect: true,
      },
      {
        text: "This means next-token prediction is conceptually a multiclass classification problem at every generated position.",
        isCorrect: true,
      },
      {
        text: "Large vocabularies increase computational burden even though the conceptual structure remains the same as smaller softmax classification.",
        isCorrect: true,
      },
      {
        text: "Using a large vocabulary means cross-entropy is no longer a valid loss function for token prediction.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture explicitly connected next-token prediction to ordinary multiclass classification, just with many more classes. The computational scale changes, but the basic learning formulation using cross-entropy remains intact.",
  },
  {
    id: "mit15773-l9-q38",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which of the following are reasonable high-level conclusions about causal LLMs, decoding, and tokenization?",
    options: [
      {
        text: "Causal language modeling can be seen as self-supervised next-token prediction built on a transformer with masked future attention.",
        isCorrect: true,
      },
      {
        text: "Decoding strategy is a separate design choice from model training and strongly affects output quality and behavior.",
        isCorrect: true,
      },
      {
        text: "Modern LLM tokenization schemes such as BPE help models handle punctuation, case, subwords, and previously unseen words more flexibly than simple word-level pipelines.",
        isCorrect: true,
      },
      {
        text: "Once a model is large enough, tokenization choices stop mattering and can be ignored safely.",
        isCorrect: false,
      },
    ],
    explanation:
      "Three major themes here are causal training, careful decoding, and flexible tokenization. All of them matter in practice, and none simply disappears just because the model becomes large.",
  },
  {
    id: "mit15773-l9-q39",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about debugging or inspecting LLM behavior are correct?",
    options: [
      {
        text: "Looking at the model's next-token probability distribution can help diagnose why certain outputs are generated.",
        isCorrect: true,
      },
      {
        text: "Tools such as playground interfaces can expose the effects of temperature and other decoding settings.",
        isCorrect: true,
      },
      {
        text: "Examining tokenization can reveal why seemingly similar strings are split differently by the model.",
        isCorrect: true,
      },
      {
        text: "Because LLMs are probabilistic, their outputs cannot be meaningfully inspected at all.",
        isCorrect: false,
      },
    ],
    explanation:
      "It can be useful to inspect not only final outputs but also token probabilities and tokenization behavior. These tools help make LLM behavior less mysterious and can reveal why small prompt or setting changes produce large output differences.",
  },
  {
    id: "mit15773-l9-q40",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly compare simple STIE-style preprocessing with BPE-style tokenization in the lecture's framing?",
    options: [
      {
        text: "Simple Standardize-Tokenize-Index-Encode tends to assume a relatively coarse word-based representation after standardization decisions such as lowercasing or punctuation stripping.",
        isCorrect: true,
      },
      {
        text: "BPE can preserve much finer distinctions because it works with characters, merged fragments, and frequent whole tokens.",
        isCorrect: true,
      },
      {
        text: "BPE's learned merge order is reused when new text arrives, so tokenization at inference time follows the same merge logic learned from the training corpus.",
        isCorrect: true,
      },
      {
        text: "BPE eliminates the need for a vocabulary entirely because every sentence is encoded as raw floating-point numbers without tokens.",
        isCorrect: false,
      },
    ],
    explanation:
      "BPE still builds and uses a vocabulary, but it does so in a more flexible way than simple whole-word pipelines. The learned merge operations are part of how new inputs are tokenized consistently after training.",
  },
];
