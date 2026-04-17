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
        text: "It is not the case that It requires the model to predict an entire paragraph in one step rather than one next token at a time.",
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
        text: "It is not the case that The whole sentence must be used only once, with no shifted prefixes, because shorter prefixes are invalid training examples.",
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
      "In the lecture's next-word prediction setup, which statements correctly describe the model output side?",
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
        text: "It is not the case that The model uses one binary sigmoid shared across all vocabulary items instead of a vocabulary-wide softmax.",
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
        text: "It is not the case that Cross-entropy for next-word prediction requires continuous-valued regression targets rather than discrete vocabulary labels.",
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
        text: "It is not the case that Because bidirectional attention prevents the use of embeddings or positional information entirely.",
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
        text: "It is not the case that It is unrelated to masking and is instead a special type of convolution.",
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
      "Which statements about the terms 'causal self-attention' and 'masked self-attention' are correct in this lecture?",
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
        text: "It is not the case that Masked self-attention means the model can only attend to the single most recent token and no earlier tokens.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture used the two names for the same mechanism. The important point is that the model may still use the whole visible prefix, just not any positions to the right of the current prediction point.",
  },
  {
    id: "mit15773-l9-q08",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly compare a transformer encoder and a transformer causal encoder in the lecture's presentation?",
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
        text: "The lecture noted that the term 'decoder' is sometimes used for the causal transformer stack in this context.",
        isCorrect: true,
      },
      {
        text: "It is not the case that The causal encoder cannot be stacked into multiple layers because masking works only for a single attention block.",
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
        text: "It is not the case that The model generates the entire final answer in one single forward pass with no iterative feedback.",
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
        text: "The GPT family was presented as an example of an autoregressive large language model.",
        isCorrect: true,
      },
      {
        text: "The term 'autoregressive' reflects that previous generated outputs are fed back in as part of the input for later predictions.",
        isCorrect: true,
      },
      {
        text: "It is not the case that They rely on bidirectional masking exactly like BERT's masked-language-model objective.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture contrasted BERT-style masked prediction with GPT-style next-token prediction. Autoregression means the model rolls forward one token at a time, using its previous outputs as part of the growing context.",
  },
  {
    id: "mit15773-l9-q11",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about GPT-3, as mentioned in the lecture, are correct?",
    options: [
      {
        text: "It was described as an autoregressive large language model trained for next-word prediction.",

        isCorrect: true,
      },
      {
        text: "The lecture stated that GPT-3 used 96 transformer layers.",
        isCorrect: true,
      },
      {
        text: "The lecture stated that GPT-3 had 96 heads in each multi-head attention layer.",
        isCorrect: true,
      },
      {
        text: "It is not the case that The lecture stated that GPT-3 was trained only on manually labeled question-answer pairs rather than large general text corpora.",
        isCorrect: true,
      },
    ],
    explanation:
      "GPT-3 was presented as a large next-word-prediction model trained on huge text corpora, not on narrowly labeled supervised datasets. The lecture used its architecture and scale to illustrate why the 'large' in LLM matters in practice.",
  },
  {
    id: "mit15773-l9-q12",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which use cases were highlighted as fitting sequence generation with autoregressive LLMs?",
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
        text: "It is not the case that Only fixed-label multiclass classification with no text output.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture emphasized how flexible text-in, text-out modeling is. Once a model can generate coherent token sequences, many applications become possible because prompts and outputs can both be expressed as text.",
  },
  {
    id: "mit15773-l9-q13",
    chapter: 1,
    difficulty: "medium",
    prompt: "In the lecture, what does 'decoding' mean?",
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
        text: "It is not the case that It means converting a transformer encoder into a recurrent neural network before inference.",
        isCorrect: true,
      },
    ],
    explanation:
      "The model gives you probabilities, not a final word by itself. Decoding is the policy you use to turn that probability distribution into an actual chosen token, and that choice strongly affects the style and reliability of the result.",
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
        text: "It is not the case that Random sampling guarantees identical outputs every time the same prompt is used.",
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
        text: "It is not the case that Even if each bad token in the long tail has small probability individually, the total probability mass of the tail can still be substantial.",

        isCorrect: false,
      },
      {
        text: "Sampling a poor token early can push the model into an implausible continuation that it may not recover from.",
        isCorrect: true,
      },
      {
        text: "The lecture emphasized that the model should ideally sample from the high-quality head of the distribution rather than the tail.",
        isCorrect: true,
      },
      {
        text: "Random sampling is impossible because softmax outputs cannot be interpreted as probabilities.",
        isCorrect: false,
      },
    ],
    explanation:
      "This is the key practical weakness of unrestricted random sampling. A single unlucky token can distort the context for all later predictions, causing the generation to drift or collapse into nonsense.",
  },
  {
    id: "mit15773-l9-q16",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements correctly describe top-k sampling?",
    options: [
      {
        text: "It is not the case that It keeps only the \\(k\\) most probable tokens and discards the rest before sampling.",

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
      "Top-k does not make the process deterministic unless combined with more restrictive settings. Its purpose is to remove low-probability tail tokens while still allowing some randomness among a smaller set of plausible candidates.",
  },
  {
    id: "mit15773-l9-q17",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe top-p sampling, also called nucleus sampling?",
    options: [
      {
        text: "It is not the case that It keeps the smallest set of most-probable tokens whose cumulative probability exceeds a chosen threshold \\(p\\).",

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
      "Top-p is adaptive: sometimes the head of the distribution is very short, and sometimes it is broader. By using cumulative probability mass instead of a fixed token count, top-p often behaves more flexibly than top-k.",
  },
  {
    id: "mit15773-l9-q18",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements about temperature are correct?",
    options: [
      {
        text: "It is not the case that Temperature modifies the sharpness of the distribution before sampling.",

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
      "Which statements about the lecture's discussion of STIE versus modern tokenization are correct?",
    options: [
      {
        text: "It is not the case that A simple Standardize-Tokenize-Index-Encode pipeline can remove useful information such as punctuation and case.",

        isCorrect: false,
      },
      {
        text: "Modern generative models typically preserve much richer token information than a simple whitespace-and-lowercase pipeline.",
        isCorrect: true,
      },
      {
        text: "The lecture argued that the GPT family uses byte-pair encoding instead of the earlier simple STIE-style approach.",
        isCorrect: true,
      },
      {
        text: "The lecture argued that punctuation and capitalization are irrelevant for modern generative models and should always be stripped away.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture used examples like punctuation, capitalization, and invented words to show why a simplistic text preprocessing pipeline is too limited for modern LLMs. Better tokenization schemes preserve more of the original structure and allow flexible composition.",
  },
  {
    id: "mit15773-l9-q20",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the basic intuition behind byte-pair encoding (BPE)?",
    options: [
      {
        text: "It is not the case that It starts from very small units such as individual characters and repeatedly merges adjacent token pairs that occur frequently.",

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
      "BPE builds a vocabulary gradually rather than assuming a fixed word list up front. This gives the tokenizer a flexible repertoire that can represent common words efficiently while still being able to compose rarer words from smaller pieces.",
  },

  {
    id: "mit15773-l9-q21",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe how temperature affects a softmax distribution during decoding?",
    options: [
      {
        text: "It is not the case that If temperature is made very small, the highest-logit token can dominate the distribution much more strongly.",

        isCorrect: false,
      },
      {
        text: "It is not the case that Increasing temperature can flatten the distribution, making more tokens competitive during sampling.",
        isCorrect: false,
      },
      {
        text: "A temperature of zero corresponds conceptually to greedy behavior, where the most likely token is selected.",
        isCorrect: true,
      },
      {
        text: "Temperature changes the training corpus itself rather than the inference-time sampling behavior.",
        isCorrect: false,
      },
    ],
    explanation:
      "Temperature is an inference-time knob applied to the logits or probabilities before sampling. Lower temperature sharpens the distribution, while higher temperature makes it flatter and therefore more random.",
  },
  {
    id: "mit15773-l9-q22",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose a model produces logits \\(a_1, a_2, \\dots, a_n\\). Which statements are correct about applying temperature \\(T\\) in a softmax of the form \\(\\frac{e^{a_i/T}}{\\sum_j e^{a_j/T}}\\)?",
    options: [
      {
        text: "It is not the case that If \\(T < 1\\), differences among logits are amplified in the exponentiated values.",

        isCorrect: false,
      },
      {
        text: "It is not the case that If \\(T > 1\\), the resulting probability distribution tends to become flatter.",
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
      "Temperature rescales logits before softmax, which changes how peaked or flat the resulting distribution becomes. It does not necessarily change the ranking of logits; instead, it usually changes how strongly the top choices dominate the others.",
  },
  {
    id: "mit15773-l9-q23",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Why did the lecture emphasize being careful with decoding settings in production use cases?",
    options: [
      {
        text: "It is not the case that Because poor decoding choices can make outputs unnecessarily random or unstable.",

        isCorrect: false,
      },
      {
        text: "It is not the case that Because a single unlikely sampled token can derail later generations.",
        isCorrect: false,
      },
      {
        text: "Because the same prompt can yield very different responses depending on sampling settings.",
        isCorrect: true,
      },
      {
        text: "Because decoding settings have no practical effect once the model has been trained well enough.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture showed that model quality alone does not determine output quality; decoding strategy matters too. Even a strong model can produce nonsensical or unstable results if sampling is handled poorly.",
  },
  {
    id: "mit15773-l9-q24",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly compare greedy decoding with more stochastic methods such as top-k, top-p, or high-temperature sampling?",
    options: [
      {
        text: "It is not the case that Greedy decoding is attractive when determinism and factual consistency matter more than variety.",

        isCorrect: false,
      },
      {
        text: "It is not the case that Stochastic methods can provide more diverse or creative outputs than greedy decoding.",
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
      "The lecture's central point was that decoding should match the task. Greedy decoding is often safer for factual or deterministic tasks, while stochastic approaches can be helpful for creativity or diversity, but they are not automatically more accurate.",
  },
  {
    id: "mit15773-l9-q25",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly reflect the lecture's discussion of long-tail sampling failures?",
    options: [
      {
        text: "It is not the case that A token with small individual probability can still be sampled because the tail contains many low-probability options.",

        isCorrect: false,
      },
      {
        text: "It is not the case that Once a poor token is appended to the context, later predictions are conditioned on that poor token as well.",
        isCorrect: false,
      },
      {
        text: "The lecture illustrated that forcing an unlikely token can make a continuation go off the rails quickly.",
        isCorrect: true,
      },
      {
        text: "The model can always fully recover from an unlucky low-probability token within one or two steps, regardless of context.",
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
        text: "It is not the case that Top-k fixes the number of candidate tokens retained before renormalization.",

        isCorrect: false,
      },
      {
        text: "It is not the case that Top-p fixes a cumulative probability threshold and allows the number of retained tokens to vary.",
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
      "Top-k and top-p both restrict sampling to a more plausible subset of tokens, but they still sample within that subset. The difference is whether the subset is defined by a fixed size or by a cumulative probability mass threshold.",
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
      "Which statements about tokenization details in modern LLMs are correct according to the lecture?",
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
      "The lecture used token visualizer examples to show that real LLM tokenizers can preserve case, punctuation, and even leading-space variants. This is one reason they are more expressive than simplistic whitespace-based tokenization schemes.",
  },
  {
    id: "mit15773-l9-q32",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about sequence generation and stopping behavior are correct in the lecture's framing?",
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
      "Which statements correctly describe the lecture's use of the term 'decoder'?",
    options: [
      {
        text: "In this lecture's simplified setting, 'decoder' can refer to the transformer causal encoder stack used for next-token prediction.",
        isCorrect: true,
      },
      {
        text: "The lecture noted that the word 'decoder' can have more than one meaning depending on context.",
        isCorrect: true,
      },
      {
        text: "The original transformer paper's sequence-to-sequence architecture includes a decoder that uses masked multi-head attention.",
        isCorrect: true,
      },
      {
        text: "The lecture claimed that 'decoder' always means a non-transformer recurrent language model.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture warned that terminology can be overloaded. In the context of next-token prediction, people often refer to the causal stack as a decoder, but in the original sequence-to-sequence transformer paper the decoder has a broader architectural role.",
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
      "Which of the following are reasonable high-level conclusions from this lecture?",
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
      "The lecture tied together three major themes: causal training, careful decoding, and flexible tokenization. These are all important in practice, and none of them simply disappears once the model becomes large.",
  },
  {
    id: "mit15773-l9-q39",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about debugging or inspecting LLM behavior, as suggested by the lecture, are correct?",
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
        text: "The lecture argued that because LLMs are probabilistic, their outputs cannot be meaningfully inspected at all.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture encouraged inspecting not just outputs but also token probabilities and tokenization behavior. These tools help make LLM behavior less mysterious and can reveal why small changes in prompts or settings produce big changes in responses.",
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
