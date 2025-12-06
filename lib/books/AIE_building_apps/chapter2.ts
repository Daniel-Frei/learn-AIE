// lib/books/AIE_building_apps/chapter2.ts

import { Question } from "../../quiz";

export const aieChapter2Questions: Question[] = [
  // ----------------------------------------------------------------------------
  // Q1–Q25: all four options are correct (4 correct)
  // ----------------------------------------------------------------------------

  {
    id: "aie-ch2-q01",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements describe why training data is critical for a foundation model’s behavior?",
    options: [
      {
        text: "If a language or domain never appears in the training data, the model will usually struggle with it.",
        isCorrect: true,
      },
      {
        text: "Biases and stereotypes in the training data can be reflected and amplified by the model.",
        isCorrect: true,
      },
      {
        text: "Data distribution largely determines which tasks the model performs well on by default.",
        isCorrect: true,
      },
      {
        text: "Cleaning and curating training data can significantly change a model’s downstream behavior.",
        isCorrect: true,
      },
    ],
    explanation:
      "Models learn patterns from their data; missing coverage, skewed distributions, and noisy or biased data all directly shape model strengths, weaknesses, and behaviors.",
  },
  {
    id: "aie-ch2-q02",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about web-scale datasets such as Common Crawl (and cleaned variants like C4) are correct?",
    options: [
      {
        text: "They contain billions of web pages collected by automated crawlers.",
        isCorrect: true,
      },
      {
        text: "They mix high-quality content with spam, misinformation, and low-quality pages.",
        isCorrect: true,
      },
      {
        text: "They are widely used as part of training corpora for many large language models.",
        isCorrect: true,
      },
      {
        text: "Even cleaned subsets still require further filtering and curation for specific domains.",
        isCorrect: true,
      },
    ],
    explanation:
      "Common Crawl–style datasets are huge and widely used, but noisy and heterogeneous, so models trained on them inherit those properties unless further curation is done.",
  },
  {
    id: "aie-ch2-q03",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements describe typical challenges of training models for low-resource languages?",
    options: [
      {
        text: "There may be very little text available online in that language.",
        isCorrect: true,
      },
      {
        text: "Publicly available corpora for that language often have lower coverage and quality than English corpora.",
        isCorrect: true,
      },
      {
        text: "Developers may need to actively curate or collect data, rather than rely on generic web crawls.",
        isCorrect: true,
      },
      {
        text: "Without such targeted data collection, models often perform badly for those languages compared with high-resource ones.",
        isCorrect: true,
      },
    ],
    explanation:
      "Low-resource languages have fewer digital resources, so generic web crawls underrepresent them. Extra curation is usually needed to reach good performance.",
  },
  {
    id: "aie-ch2-q04",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements describe common strategies for building domain-specific foundation models (e.g. for biomedicine)?",
    options: [
      {
        text: "Curating task-relevant datasets such as medical notes, scientific articles, or specialized images.",
        isCorrect: true,
      },
      {
        text: "Combining a general model with additional domain-specific finetuning data.",
        isCorrect: true,
      },
      {
        text: "Leveraging proprietary or hard-to-obtain datasets as a competitive advantage.",
        isCorrect: true,
      },
      {
        text: "Using architectures that can ingest domain-specific modalities such as sequences, 3D structures, or scans.",
        isCorrect: true,
      },
    ],
    explanation:
      "Domain models typically mix targeted training data, finetuning or adaptation from a base model, access to proprietary corpora, and sometimes modality-specific architectures.",
  },
  {
    id: "aie-ch2-q05",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly contrast pre-training and post-training for large models?",
    options: [
      {
        text: "Pre-training usually consumes the vast majority of compute compared with post-training.",
        isCorrect: true,
      },
      {
        text: "Pre-training teaches broad world knowledge and patterns from large unlabeled corpora.",
        isCorrect: true,
      },
      {
        text: "Post-training mainly focuses on aligning the model’s behavior with human preferences and instructions.",
        isCorrect: true,
      },
      {
        text: "Post-training can make a capable but awkward base model feel conversational and user-friendly.",
        isCorrect: true,
      },
    ],
    explanation:
      "Pre-training is the large, self-supervised phase that builds general competence; post-training (SFT and preference finetuning) shapes that competence into aligned behavior.",
  },
  {
    id: "aie-ch2-q06",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements are typical properties of the transformer architecture used in modern language models?",
    options: [
      {
        text: "It relies heavily on self-attention to mix information across tokens.",
        isCorrect: true,
      },
      {
        text: "It can process all tokens in a sequence in parallel during training, unlike classic RNNs.",
        isCorrect: true,
      },
      {
        text: "It scales relatively well to large model sizes and long sequences (up to context limits).",
        isCorrect: true,
      },
      {
        text: "It has largely replaced earlier seq2seq RNN architectures in state-of-the-art NLP systems.",
        isCorrect: true,
      },
    ],
    explanation:
      "Transformers use parallel self-attention rather than recurrence, enabling better long-range dependencies and scalability; they dominate modern NLP models.",
  },
  {
    id: "aie-ch2-q07",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe limitations of classic RNN-based seq2seq models compared with transformers?",
    options: [
      {
        text: "They process sequences step by step, limiting parallelism during training.",
        isCorrect: true,
      },
      {
        text: "They struggle with very long-range dependencies due to vanishing or exploding gradients.",
        isCorrect: true,
      },
      {
        text: "Their fixed-size hidden state can become a bottleneck for remembering long contexts.",
        isCorrect: true,
      },
      {
        text: "Their inductive bias makes them less efficient than self-attention at modeling flexible token-to-token relationships.",
        isCorrect: true,
      },
    ],
    explanation:
      "Seq2seq RNNs impose sequential computation and rely on a compressed hidden state, making training slower and long-range dependencies harder than transformers’ self-attention.",
  },
  {
    id: "aie-ch2-q08",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which quantities are commonly used to describe the scale of a foundation model and its training run?",
    options: [
      {
        text: "Number of model parameters.",
        isCorrect: true,
      },
      {
        text: "Number of training tokens processed.",
        isCorrect: true,
      },
      {
        text: "Approximate number of floating point operations (FLOPs) used during training.",
        isCorrect: true,
      },
      {
        text: "Rough size and diversity of the training dataset(s).",
        isCorrect: true,
      },
    ],
    explanation:
      "Scale is usually summarized by parameter count, token count, and compute (FLOPs), with dataset size and diversity providing additional context.",
  },
  {
    id: "aie-ch2-q09",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about compute-optimal scaling and data/token trade-offs for language models are correct?",
    options: [
      {
        text: "For a fixed compute budget, there is usually an optimal balance between model size and number of training tokens.",
        isCorrect: true,
      },
      {
        text: "Making a model larger without enough additional data tends to under-train it and waste parameters.",
        isCorrect: true,
      },
      {
        text: "Using far more data than suggested by scaling laws can over-emphasize data while under-sizing the model.",
        isCorrect: true,
      },
      {
        text: "Empirical scaling laws can guide how to allocate compute between parameters and tokens.",
        isCorrect: true,
      },
    ],
    explanation:
      "Scaling laws show that for a given compute budget, there’s a sweet spot for parameters vs. tokens; deviating too far in either direction can be suboptimal.",
  },
  {
    id: "aie-ch2-q10",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe supervised finetuning (SFT) for large language models?",
    options: [
      {
        text: "The model is trained on (prompt, response) pairs that demonstrate good behavior.",
        isCorrect: true,
      },
      {
        text: "The goal is to teach the model to imitate high-quality responses from humans or curated data.",
        isCorrect: true,
      },
      {
        text: "SFT is usually applied on top of a pretrained base model rather than training from scratch.",
        isCorrect: true,
      },
      {
        text: "SFT can move the model from generic completion behavior toward more helpful, instruction-following behavior.",
        isCorrect: true,
      },
    ],
    explanation:
      "SFT uses demonstration data to teach the model how to respond appropriately to prompts, often transforming a completion-style model into a conversational assistant.",
  },
  {
    id: "aie-ch2-q11",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements describe typical ingredients of reinforcement learning from human feedback (RLHF) for aligning models?",
    options: [
      {
        text: "Collecting human preference data by showing annotators multiple candidate outputs per prompt.",
        isCorrect: true,
      },
      {
        text: "Training a reward model that predicts which of two responses humans prefer.",
        isCorrect: true,
      },
      {
        text: "Using a reinforcement learning algorithm (often PPO) to adjust the language model to maximize the reward model’s scores.",
        isCorrect: true,
      },
      {
        text: "Starting from an SFT model so that RLHF optimizes an already usable behavior policy.",
        isCorrect: true,
      },
    ],
    explanation:
      "RLHF typically: (1) collects pairwise preferences, (2) trains a reward model, and (3) uses RL (like PPO) on top of an SFT model to optimize responses according to that reward.",
  },
  {
    id: "aie-ch2-q12",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the use of a reward model (RM) in preference finetuning?",
    options: [
      {
        text: "The RM maps (prompt, response) pairs to scalar scores reflecting human preference.",
        isCorrect: true,
      },
      {
        text: "The RM is trained using human comparisons where labelers say which response they prefer.",
        isCorrect: true,
      },
      {
        text: "The RM can be used either inside RL (e.g., PPO) or outside RL to rescore multiple candidates (best-of-N).",
        isCorrect: true,
      },
      {
        text: "The quality and coverage of the preference data directly affect how well the RM captures nuanced human judgments.",
        isCorrect: true,
      },
    ],
    explanation:
      "Reward models are learned from pairwise preference labels and then used to score outputs; they’re central to RLHF and can also support best-of-N selection without RL.",
  },
  {
    id: "aie-ch2-q13",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements describe softmax in the context of language models?",
    options: [
      {
        text: "It converts logits into a probability distribution over tokens.",
        isCorrect: true,
      },
      {
        text: "Output probabilities are non-negative and sum to 1.",
        isCorrect: true,
      },
      {
        text: "Higher logits correspond to higher probabilities after softmax.",
        isCorrect: true,
      },
      {
        text: "The vocabulary size determines the dimensionality of the softmax output.",
        isCorrect: true,
      },
    ],
    explanation:
      "Softmax turns raw logits into probabilities over the vocabulary, preserving order relationships and ensuring probabilities are non-negative and sum to one.",
  },
  {
    id: "aie-ch2-q14",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe temperature in sampling from language models?",
    options: [
      {
        text: "Temperature rescales logits before softmax is applied.",
        isCorrect: true,
      },
      {
        text: "Lower temperatures make the distribution more peaked, favoring high-probability tokens.",
        isCorrect: true,
      },
      {
        text: "Higher temperatures flatten the distribution, increasing the chance of sampling lower-probability tokens.",
        isCorrect: true,
      },
      {
        text: "Setting temperature very close to zero approximates greedy (argmax) decoding.",
        isCorrect: true,
      },
    ],
    explanation:
      "Temperature divides logits, sharpening or flattening the probabilities: low T ≈ more deterministic; high T ≈ more diverse and creative outputs.",
  },
  {
    id: "aie-ch2-q15",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about top-k sampling are correct for language model decoding?",
    options: [
      {
        text: "The model keeps only the k tokens with highest logits (or probabilities) before sampling.",
        isCorrect: true,
      },
      {
        text: "Softmax is applied only over those top-k tokens, not the entire vocabulary.",
        isCorrect: true,
      },
      {
        text: "Smaller k makes outputs more predictable but less diverse.",
        isCorrect: true,
      },
      {
        text: "This strategy can reduce the computational cost of softmax compared with using the full vocabulary.",
        isCorrect: true,
      },
    ],
    explanation:
      "Top-k truncates to the k most likely tokens, normalizes among them, and samples; this can reduce compute and trade diversity for predictability via k.",
  },
  {
    id: "aie-ch2-q16",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements describe top-p (nucleus) sampling for language models?",
    options: [
      {
        text: "Tokens are sorted by probability in descending order.",
        isCorrect: true,
      },
      {
        text: "The smallest set of tokens whose cumulative probability exceeds p is selected as the sampling pool.",
        isCorrect: true,
      },
      {
        text: "The value p controls how much of the probability mass is retained (e.g., 0.9 or 0.95).",
        isCorrect: true,
      },
      {
        text: "Unlike top-k, the number of tokens considered can vary depending on the context.",
        isCorrect: true,
      },
    ],
    explanation:
      "Top-p chooses a variable-size subset of tokens whose cumulative probability reaches p, adapting to context rather than fixing k.",
  },
  {
    id: "aie-ch2-q17",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements describe why log probabilities (logprobs) are convenient when working with language model outputs?",
    options: [
      {
        text: "They mitigate numerical underflow when dealing with extremely small probabilities.",
        isCorrect: true,
      },
      {
        text: "They turn products of probabilities into sums, which are easier to handle numerically.",
        isCorrect: true,
      },
      {
        text: "They are useful for scoring sequences and building classification or evaluation logic.",
        isCorrect: true,
      },
      {
        text: "Many provider APIs expose token-level logprobs for debugging or advanced use cases.",
        isCorrect: true,
      },
    ],
    explanation:
      "Logprobs avoid underflow, make combining probabilities easier, and enable advanced behaviors such as scoring candidate sequences or implementing classifiers.",
  },
  {
    id: "aie-ch2-q18",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about the probabilistic nature of language model outputs are correct?",
    options: [
      {
        text: "The same prompt can yield different outputs across runs when sampling is stochastic.",
        isCorrect: true,
      },
      {
        text: "Even with fixed sampling settings, tiny numerical differences can sometimes change the sampled token.",
        isCorrect: true,
      },
      {
        text: "This randomness helps with creativity and diversity of outputs.",
        isCorrect: true,
      },
      {
        text: "The same probabilistic behavior can also cause inconsistencies that feel strange to users.",
        isCorrect: true,
      },
    ],
    explanation:
      "Sampling from a probability distribution introduces randomness: helpful for creativity, problematic for consistency and reproducibility if unmanaged.",
  },
  {
    id: "aie-ch2-q19",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements describe common techniques to reduce user-visible inconsistency from probabilistic models?",
    options: [
      {
        text: "Caching answers for common or repeated queries.",
        isCorrect: true,
      },
      {
        text: "Fixing sampling hyperparameters such as temperature, top-k, and top-p.",
        isCorrect: true,
      },
      {
        text: "Fixing a random seed when you control the inference environment.",
        isCorrect: true,
      },
      {
        text: "Designing a memory or state mechanism so the system can reuse past decisions where appropriate.",
        isCorrect: true,
      },
    ],
    explanation:
      "Caching, fixed sampling configs, reproducible seeds, and memory/stateful workflows can all reduce apparent randomness in an otherwise probabilistic system.",
  },
  {
    id: "aie-ch2-q20",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly characterize hallucinations in generative models?",
    options: [
      {
        text: "They are outputs that sound plausible but are not grounded in facts or the model’s true knowledge.",
        isCorrect: true,
      },
      {
        text: "They can arise even if the model’s training data is very large.",
        isCorrect: true,
      },
      {
        text: "They are especially dangerous in factual or safety-critical applications.",
        isCorrect: true,
      },
      {
        text: "They have been discussed in natural language generation research since before today’s large LLMs.",
        isCorrect: true,
      },
    ],
    explanation:
      "Hallucinations are plausible but incorrect outputs; they predate today’s LLMs and remain a major concern when factual accuracy matters.",
  },
  {
    id: "aie-ch2-q21",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements describe the ‘self-delusion’ / snowballing hypothesis for why language models hallucinate?",
    options: [
      {
        text: "The model conditions on its own generated tokens as if they were ground truth.",
        isCorrect: true,
      },
      {
        text: "An early wrong assumption in the generation can get reinforced and elaborated.",
        isCorrect: true,
      },
      {
        text: "The model can drift further away from reality with each step that builds on the mistaken context.",
        isCorrect: true,
      },
      {
        text: "This dynamic can cause errors on questions the model would answer correctly if it started from a clean context.",
        isCorrect: true,
      },
    ],
    explanation:
      "In the self-delusion view, once a generation includes a wrong statement, subsequent steps treat it as context, compounding the error and causing snowballing hallucinations.",
  },
  {
    id: "aie-ch2-q22",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements describe the ‘mismatched internal knowledge’ hypothesis for hallucinations?",
    options: [
      {
        text: "During SFT, models are trained to mimic human-written answers.",
        isCorrect: true,
      },
      {
        text: "Labelers may use knowledge that the model doesn’t actually possess.",
        isCorrect: true,
      },
      {
        text: "If the model is forced to imitate answers that rely on unknown facts, it may end up generating confident-seeming guesses.",
        isCorrect: true,
      },
      {
        text: "Hallucinations can arise when the model is optimized to produce such answers even without the underlying knowledge.",
        isCorrect: true,
      },
    ],
    explanation:
      "If labelers use knowledge the model lacks but the model is forced to imitate their outputs, it effectively learns to ‘make things up’ that resemble those answers.",
  },
  {
    id: "aie-ch2-q23",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements describe broader bottlenecks for scaling foundation models beyond just model size?",
    options: [
      {
        text: "Availability of high-quality human-generated data.",
        isCorrect: true,
      },
      {
        text: "Legal and licensing constraints around scraping or using proprietary datasets.",
        isCorrect: true,
      },
      {
        text: "Hardware availability and the cost of compute (including accelerators).",
        isCorrect: true,
      },
      {
        text: "Energy consumption and power infrastructure limits for large data centers.",
        isCorrect: true,
      },
    ],
    explanation:
      "Future scaling is constrained not just by parameter-count limits but also by data scarcity, legal restrictions, hardware supply, and energy/power constraints.",
  },
  {
    id: "aie-ch2-q24",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe test-time compute strategies like best-of-N or beam search for language models?",
    options: [
      {
        text: "They involve generating multiple candidate outputs for the same prompt.",
        isCorrect: true,
      },
      {
        text: "You can select the best candidate according to a reward model, heuristic, or task-specific scoring function.",
        isCorrect: true,
      },
      {
        text: "They trade extra inference cost for potential gains in quality or reliability.",
        isCorrect: true,
      },
      {
        text: "They can be combined with sampling strategies like temperature, top-k, or top-p to diversify candidates.",
        isCorrect: true,
      },
    ],
    explanation:
      "Best-of-N and beam-style strategies increase test-time compute to sample or search over multiple candidates, then pick those that score best under some criterion.",
  },
  {
    id: "aie-ch2-q25",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements describe why understanding sampling is important for application developers using LLMs?",
    options: [
      {
        text: "Sampling choices can explain inconsistent or surprising behavior.",
        isCorrect: true,
      },
      {
        text: "Tuning sampling hyperparameters can often improve performance without retraining the model.",
        isCorrect: true,
      },
      {
        text: "Different tasks benefit from different trade-offs between diversity and predictability.",
        isCorrect: true,
      },
      {
        text: "Sampling interacts with evaluation and reliability, so it must be considered in system design.",
        isCorrect: true,
      },
    ],
    explanation:
      "Sampling heavily influences LLM behavior; understanding it lets you debug, tune trade-offs, and design more reliable application workflows.",
  },

  // ----------------------------------------------------------------------------
  // Q26–Q50: three correct, one incorrect (3 correct)
  // ----------------------------------------------------------------------------

  {
    id: "aie-ch2-q26",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about multilingual training data for language models are correct?",
    options: [
      {
        text: "Internet text is heavily skewed toward certain high-resource languages like English.",
        isCorrect: true,
      },
      {
        text: "Models trained mostly on English will typically perform better in English than in low-resource languages.",
        isCorrect: true,
      },
      {
        text: "Adding targeted data in a low-resource language can noticeably improve performance in that language.",
        isCorrect: true,
      },
      {
        text: "Generic web crawls always contain balanced data for all languages, so no extra curation is needed.",
        isCorrect: false,
      },
    ],
    explanation:
      "Web data is unbalanced; high-resource languages dominate. Low-resource languages usually need explicit data collection and curation.",
  },
  {
    id: "aie-ch2-q27",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about data quality in web-scale corpora are correct?",
    options: [
      {
        text: "They can contain spam, clickbait, and misleading information.",
        isCorrect: true,
      },
      {
        text: "Heuristic filters and blocklists are often used to remove obviously bad content.",
        isCorrect: true,
      },
      {
        text: "Some remaining low-quality content can still influence model outputs.",
        isCorrect: true,
      },
      {
        text: "Any raw web crawl is automatically a clean, high-quality dataset suitable for safety-critical tasks.",
        isCorrect: false,
      },
    ],
    explanation:
      "Raw web data is noisy and must be filtered; even then, some low-quality or harmful content will remain and can affect model behavior.",
  },
  {
    id: "aie-ch2-q28",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about proprietary training data are correct?",
    options: [
      {
        text: "Contracts, medical records, and internal documents can be valuable proprietary datasets.",
        isCorrect: true,
      },
      {
        text: "Access to high-quality proprietary data can become a competitive advantage between organizations.",
        isCorrect: true,
      },
      {
        text: "License agreements with publishers or content owners can grant access to otherwise restricted data.",
        isCorrect: true,
      },
      {
        text: "Proprietary data is always freely available to anyone building an AI system.",
        isCorrect: false,
      },
    ],
    explanation:
      "Proprietary datasets are restricted but valuable; organizations negotiate licenses or rely on their own internal data as a source of advantage.",
  },
  {
    id: "aie-ch2-q29",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about alternatives or complements to transformers for sequence modeling are correct?",
    options: [
      {
        text: "State space models (SSMs) such as Mamba attempt to capture long-range dependencies with different inductive biases than attention.",
        isCorrect: true,
      },
      {
        text: "Convolutional architectures can also handle sequences and have been used in earlier NLP models.",
        isCorrect: true,
      },
      {
        text: "New architectures might improve efficiency or context length while aiming to preserve modeling quality.",
        isCorrect: true,
      },
      {
        text: "Because transformers exist, there is no research interest in alternative architectures anymore.",
        isCorrect: false,
      },
    ],
    explanation:
      "Transformers are dominant but not the only option; research continues into SSMs, convolutions, and hybrid architectures for efficiency and longer contexts.",
  },
  {
    id: "aie-ch2-q30",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements are typical benefits of using a smaller (e.g., 7B) model over a very large (e.g., 175B) model?",
    options: [
      {
        text: "Lower inference latency on the same hardware.",
        isCorrect: true,
      },
      {
        text: "Lower memory and compute requirements, enabling cheaper deployment.",
        isCorrect: true,
      },
      {
        text: "Easier on-device or edge deployment scenarios.",
        isCorrect: true,
      },
      {
        text: "Guaranteed better performance on every task compared with larger models.",
        isCorrect: false,
      },
    ],
    explanation:
      "Smaller models are cheaper and faster, and may run on edge devices, but they typically underperform larger models on many tasks.",
  },
  {
    id: "aie-ch2-q31",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about pre-training objectives for language models are correct?",
    options: [
      {
        text: "Autoregressive models commonly use next-token prediction as their objective.",
        isCorrect: true,
      },
      {
        text: "Masked language models predict masked-out tokens from their context.",
        isCorrect: true,
      },
      {
        text: "These objectives are forms of self-supervision, where labels are derived from the data itself.",
        isCorrect: true,
      },
      {
        text: "Pre-training typically requires fully labeled datasets with human-written class labels.",
        isCorrect: false,
      },
    ],
    explanation:
      "Pre-training usually relies on self-supervised objectives like next-token or masked-token prediction, not on manual class labels.",
  },
  {
    id: "aie-ch2-q32",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements describe issues that arise when training on AI-generated data without care?",
    options: [
      {
        text: "New models may start to overfit to artifacts of older models instead of human-generated patterns.",
        isCorrect: true,
      },
      {
        text: "Recursive training on model outputs can cause models to gradually forget rarer real-world patterns.",
        isCorrect: true,
      },
      {
        text: "It can be hard to distinguish synthetic data from real web data at scale.",
        isCorrect: true,
      },
      {
        text: "Training on synthetic outputs is guaranteed to always improve model performance in the long run.",
        isCorrect: false,
      },
    ],
    explanation:
      "Uncontrolled use of synthetic data can distort distributions and lose diversity; it is not automatically beneficial and needs careful design.",
  },
  {
    id: "aie-ch2-q33",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about the cost of scaling models are correct?",
    options: [
      {
        text: "Training cost generally increases with both model size and dataset size.",
        isCorrect: true,
      },
      {
        text: "Larger models usually require more expensive hardware infrastructure.",
        isCorrect: true,
      },
      {
        text: "Energy consumption of data centers becomes a practical constraint at very large scales.",
        isCorrect: true,
      },
      {
        text: "Once you have the hardware, training cost is essentially zero and can be ignored.",
        isCorrect: false,
      },
    ],
    explanation:
      "Scaling requires massive compute and energy. Hardware and electricity costs dominate; training is far from free, even with existing hardware.",
  },
  {
    id: "aie-ch2-q34",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements about scaling laws for language models are correct?",
    options: [
      {
        text: "Empirical curves often show loss decreasing roughly as a power law in model size.",
        isCorrect: true,
      },
      {
        text: "Similar relationships exist for loss versus dataset size and versus compute.",
        isCorrect: true,
      },
      {
        text: "Scaling laws separate reducible loss from an irreducible component tied to data entropy and task difficulty.",
        isCorrect: true,
      },
      {
        text: "These laws guarantee that doubling parameters always halves the loss.",
        isCorrect: false,
      },
    ],
    explanation:
      "Scaling laws show smooth trends but not exact guarantees. Doubling parameters improves performance but not in such a simple fixed ratio.",
  },
  {
    id: "aie-ch2-q35",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about the role of supervised finetuning in aligning models with user expectations are correct?",
    options: [
      {
        text: "It can teach the model to answer questions instead of just completing text.",
        isCorrect: true,
      },
      {
        text: "It can show the model how to follow instructions and structure answers.",
        isCorrect: true,
      },
      {
        text: "It uses explicit examples of desired behavior for training.",
        isCorrect: true,
      },
      {
        text: "It is unnecessary if the pre-training data already contains internet text.",
        isCorrect: false,
      },
    ],
    explanation:
      "Even when pre-training includes internet data, SFT is valuable for shaping model behaviors into user-oriented, instruction-following responses.",
  },
  {
    id: "aie-ch2-q36",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe possible downsides or risks of RLHF and preference finetuning?",
    options: [
      {
        text: "The reward model might encode labeler biases and blindly enforce them.",
        isCorrect: true,
      },
      {
        text: "RL optimization can lead the policy model to exploit quirks of the reward model instead of true human preference.",
        isCorrect: true,
      },
      {
        text: "Preference finetuning may inadvertently increase hallucinations on some datasets.",
        isCorrect: true,
      },
      {
        text: "RLHF guarantees perfect alignment with all users’ values once trained.",
        isCorrect: false,
      },
    ],
    explanation:
      "Reward modeling and RLHF are powerful but imperfect; they can amplify biases, encourage reward hacking, and alter error patterns, including hallucinations.",
  },
  {
    id: "aie-ch2-q37",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about using reward models without RL (e.g., best-of-N) are correct?",
    options: [
      {
        text: "You can generate several candidates from an SFT model.",
        isCorrect: true,
      },
      {
        text: "You then score those candidates with a reward model and pick the best one.",
        isCorrect: true,
      },
      {
        text: "This approach can improve quality without changing the base model weights.",
        isCorrect: true,
      },
      {
        text: "It requires retraining the base model from random initialization each time.",
        isCorrect: false,
      },
    ],
    explanation:
      "Best-of-N uses a reward model to select outputs from an existing generator; it’s a post-hoc selection method and doesn’t require retraining the base model.",
  },
  {
    id: "aie-ch2-q38",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about greedy sampling in classification vs. generative tasks are correct?",
    options: [
      {
        text: "In simple classification, taking the most probable class is often the right choice.",
        isCorrect: true,
      },
      {
        text: "In open-ended text generation, always picking the argmax token can produce boring, repetitive text.",
        isCorrect: true,
      },
      {
        text: "Greedy decoding can be viewed as sampling with an extremely low temperature.",
        isCorrect: true,
      },
      {
        text: "Greedy decoding is always the best strategy for creative writing tasks.",
        isCorrect: false,
      },
    ],
    explanation:
      "Greedy is fine for classification but too deterministic for creative generation, where stochastic sampling strategies tend to work better.",
  },
  {
    id: "aie-ch2-q39",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about top-k vs. top-p sampling are correct?",
    options: [
      {
        text: "Top-k restricts sampling to the k most likely tokens, regardless of their cumulative probability.",
        isCorrect: true,
      },
      {
        text: "Top-p chooses a variable number of tokens whose cumulative probability reaches a target threshold p.",
        isCorrect: true,
      },
      {
        text: "Top-p can adapt to contexts where only a few tokens are plausible vs. where many are plausible.",
        isCorrect: true,
      },
      {
        text: "Top-k and top-p are mutually exclusive and cannot be combined in practice.",
        isCorrect: false,
      },
    ],
    explanation:
      "Top-k fixes the count; top-p fixes the mass. They’re often combined (e.g., apply top-k, then top-p on that subset) in practical decoding schemes.",
  },
  {
    id: "aie-ch2-q40",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements about managing probabilistic outputs in production systems are correct?",
    options: [
      {
        text: "You may need deterministic fallbacks or guardrails for high-risk actions.",
        isCorrect: true,
      },
      {
        text: "Evaluation pipelines should account for distributional changes due to sampling settings.",
        isCorrect: true,
      },
      {
        text: "Monitoring should track error rates as you change decoding hyperparameters.",
        isCorrect: true,
      },
      {
        text: "Once a model is deployed, decoding settings should never be changed.",
        isCorrect: false,
      },
    ],
    explanation:
      "Production systems regularly adjust decoding strategies, but must do so with monitoring, evaluation, and guardrails for critical flows.",
  },
  {
    id: "aie-ch2-q41",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about log probabilities (logprobs) returned by provider APIs are correct?",
    options: [
      {
        text: "They can show which tokens the model considered likely at each step.",
        isCorrect: true,
      },
      {
        text: "They help debug when a model chooses an unexpected token.",
        isCorrect: true,
      },
      {
        text: "They can be used to score candidate completions for reranking or classification.",
        isCorrect: true,
      },
      {
        text: "They always expose the full probability distribution over all tokens with no restrictions.",
        isCorrect: false,
      },
    ],
    explanation:
      "APIs often expose only partial logprobs (e.g., top tokens) and may restrict access for security or IP reasons.",
  },
  {
    id: "aie-ch2-q42",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about caching responses in LLM applications are correct?",
    options: [
      {
        text: "Caching reduces cost for repeated or similar queries.",
        isCorrect: true,
      },
      {
        text: "Caching can improve perceived consistency for identical prompts.",
        isCorrect: true,
      },
      {
        text: "You may need cache invalidation when models or prompts change significantly.",
        isCorrect: true,
      },
      {
        text: "Caching makes evaluation unnecessary, because outputs will never change.",
        isCorrect: false,
      },
    ],
    explanation:
      "Caching is a useful engineering tool for cost and consistency, but doesn’t remove the need to evaluate and monitor model outputs.",
  },
  {
    id: "aie-ch2-q43",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about the trade-off between diversity and reliability in sampling are correct?",
    options: [
      {
        text: "Higher temperature generally increases diversity but may increase errors.",
        isCorrect: true,
      },
      {
        text: "Lower temperature generally reduces diversity but may improve factual reliability.",
        isCorrect: true,
      },
      {
        text: "For some tasks, you may deliberately choose higher temperature to generate multiple creative candidates.",
        isCorrect: true,
      },
      {
        text: "A single fixed sampling configuration is ideal for all tasks and domains.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sampling settings should be tuned per use case; there is no single optimal configuration for creativity, reliability, and other constraints simultaneously.",
  },
  {
    id: "aie-ch2-q44",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements describe possible mitigation strategies for hallucinations?",
    options: [
      {
        text: "Using retrieval-augmented generation to ground answers in external knowledge sources.",
        isCorrect: true,
      },
      {
        text: "Prompting the model to admit uncertainty instead of guessing when it ‘doesn’t know’.",
        isCorrect: true,
      },
      {
        text: "Designing reward functions that penalize confidently wrong answers more strongly.",
        isCorrect: true,
      },
      {
        text: "Assuming hallucinations will disappear if we simply increase model size without changing anything else.",
        isCorrect: false,
      },
    ],
    explanation:
      "Hallucinations require architectural, training, and workflow-level mitigations; size alone doesn’t automatically eliminate them.",
  },
  {
    id: "aie-ch2-q45",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements about energy and infrastructure constraints for large models are correct?",
    options: [
      {
        text: "Data centers already consume a noticeable fraction of global electricity.",
        isCorrect: true,
      },
      {
        text: "Future growth of data centers is limited by the ability to supply additional power.",
        isCorrect: true,
      },
      {
        text: "Energy costs influence both training and inference economics for AI systems.",
        isCorrect: true,
      },
      {
        text: "Electricity usage is irrelevant when planning large-scale AI deployments.",
        isCorrect: false,
      },
    ],
    explanation:
      "Energy and power infrastructure are real constraints; scaling compute requires considering electricity availability, cost, and sustainability.",
  },
  {
    id: "aie-ch2-q46",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about combining multiple sampling strategies are correct?",
    options: [
      {
        text: "You can apply both temperature scaling and top-k at the same time.",
        isCorrect: true,
      },
      {
        text: "You can further apply top-p on the truncated distribution to refine the candidate set.",
        isCorrect: true,
      },
      {
        text: "Different combinations can be used for different tasks or stages of generation.",
        isCorrect: true,
      },
      {
        text: "Using more than one sampling strategy is mathematically impossible.",
        isCorrect: false,
      },
    ],
    explanation:
      "In practice, temperature, top-k, and top-p are often combined to customize behavior for specific applications or prompts.",
  },
  {
    id: "aie-ch2-q47",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about the relationship between pre-training and post-training are correct?",
    options: [
      {
        text: "Pre-training provides broad capabilities from large corpora.",
        isCorrect: true,
      },
      {
        text: "Post-training adapts and constrains those capabilities to better match human preferences.",
        isCorrect: true,
      },
      {
        text: "Post-training typically uses much less compute than pre-training.",
        isCorrect: true,
      },
      {
        text: "Post-training completely replaces what the model learned during pre-training.",
        isCorrect: false,
      },
    ],
    explanation:
      "Post-training builds on pre-training rather than replacing it; it uses relatively little compute while drastically changing model behavior.",
  },
  {
    id: "aie-ch2-q48",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements describe why understanding the training data distribution matters for evaluation?",
    options: [
      {
        text: "Benchmarks overlapping heavily with training data may overestimate generalization.",
        isCorrect: true,
      },
      {
        text: "Shifts between training and production data distributions can degrade performance.",
        isCorrect: true,
      },
      {
        text: "Domain-specific evaluation sets are needed if you care about specific industries or tasks.",
        isCorrect: true,
      },
      {
        text: "As long as a model is large, evaluation details on specific data distributions are unimportant.",
        isCorrect: false,
      },
    ],
    explanation:
      "Evaluation must consider whether data matches production conditions and whether benchmarks are contaminated or too similar to training data.",
  },
  {
    id: "aie-ch2-q49",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about using temperature for debugging and analysis are correct?",
    options: [
      {
        text: "Setting temperature close to zero approximates greedy decoding, revealing the model’s most likely behavior.",
        isCorrect: true,
      },
      {
        text: "Exploring outputs at higher temperatures can reveal alternative hypotheses the model considers plausible.",
        isCorrect: true,
      },
      {
        text: "Varying temperature while inspecting logprobs can reveal whether the distribution is too flat or too peaky.",
        isCorrect: true,
      },
      {
        text: "Temperature can only be changed during training, not at inference time.",
        isCorrect: false,
      },
    ],
    explanation:
      "Temperature is an inference-time control; varying it is a useful way to inspect and understand the model’s probability distribution over outputs.",
  },
  {
    id: "aie-ch2-q50",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about inconsistency in LLM outputs are correct?",
    options: [
      {
        text: "Asking the same question twice can yield different answers when sampling is stochastic.",
        isCorrect: true,
      },
      {
        text: "Small paraphrases in the prompt can unexpectedly change the model’s answer.",
        isCorrect: true,
      },
      {
        text: "Users may perceive such inconsistency as unreliability or untrustworthiness.",
        isCorrect: true,
      },
      {
        text: "Probabilistic sampling guarantees identical outputs for identical inputs.",
        isCorrect: false,
      },
    ],
    explanation:
      "Probabilistic sampling naturally leads to variability; from a user perspective, this can feel inconsistent unless carefully managed.",
  },

  // ----------------------------------------------------------------------------
  // Q51–Q75: two correct, two incorrect (2 correct)
  // ----------------------------------------------------------------------------

  {
    id: "aie-ch2-q51",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about what ‘foundation model’ usually means are correct?",
    options: [
      {
        text: "A large model trained on broad data that can be adapted to many downstream tasks.",
        isCorrect: true,
      },
      {
        text: "A model whose pre-training alone already enables many useful capabilities.",
        isCorrect: true,
      },
      {
        text: "A small model trained from scratch on a single tiny dataset for a single task only.",
        isCorrect: false,
      },
      {
        text: "A model that must be fully retrained every time you change the application.",
        isCorrect: false,
      },
    ],
    explanation:
      "Foundation models are large, broadly trained, and reusable across tasks, not tiny task-specific models that must be trained from scratch each time.",
  },
  {
    id: "aie-ch2-q52",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "A model has two possible outputs A and B with logits [1, 2] and temperature T = 1. Which statements are correct?",
    options: [
      {
        text: "Token B will have higher probability than token A after softmax.",
        isCorrect: true,
      },
      {
        text: "The exact probabilities are approximately 0.27 for A and 0.73 for B.",
        isCorrect: true,
      },
      {
        text: "Softmax will assign equal probability to A and B because they share the same sign.",
        isCorrect: false,
      },
      {
        text: "Token A must always be chosen, because its logit is positive.",
        isCorrect: false,
      },
    ],
    explanation:
      "Softmax(1,2) ≈ (0.27, 0.73); higher logit → higher probability. The sign alone doesn’t force equal or deterministic probabilities.",
  },
  {
    id: "aie-ch2-q53",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about the underflow problem and logprobs are correct?",
    options: [
      {
        text: "Underflow can occur when probabilities are so small that they round to zero in floating-point representation.",
        isCorrect: true,
      },
      {
        text: "Working in log space helps mitigate underflow because it represents small probabilities as manageable negative numbers.",
        isCorrect: true,
      },
      {
        text: "Underflow only happens when probabilities are exactly zero in mathematics.",
        isCorrect: false,
      },
      {
        text: "Logprobs are rarely used in practice because they are harder to compute than raw probabilities.",
        isCorrect: false,
      },
    ],
    explanation:
      "Underflow is a numerical artifact; logs help avoid tiny values disappearing. Logprobs are widely used in practice for stability and convenience.",
  },
  {
    id: "aie-ch2-q54",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about best-of-N sampling are correct?",
    options: [
      {
        text: "It can be seen as a simple form of test-time search over multiple stochastic outputs.",
        isCorrect: true,
      },
      {
        text: "It can use a reward model, heuristic, or human-in-the-loop to pick the best candidate.",
        isCorrect: true,
      },
      {
        text: "It guarantees that at least one of the outputs will be perfect.",
        isCorrect: false,
      },
      {
        text: "It is free, because generating N outputs costs the same as generating 1.",
        isCorrect: false,
      },
    ],
    explanation:
      "Best-of-N improves odds of a good output but doesn’t guarantee perfection; it costs roughly N× a single sample unless optimized.",
  },
  {
    id: "aie-ch2-q55",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about model parameters are correct?",
    options: [
      {
        text: "They are the tunable weights learned during training.",
        isCorrect: true,
      },
      {
        text: "Increasing the number of parameters generally increases model capacity.",
        isCorrect: true,
      },
      {
        text: "They are fixed constants chosen by the user and never updated.",
        isCorrect: false,
      },
      {
        text: "Having more parameters automatically guarantees better real-world performance even with poor data.",
        isCorrect: false,
      },
    ],
    explanation:
      "Parameters are learned weights; more parameters increase potential capacity but don’t guarantee better performance without sufficient quality data and compute.",
  },
  {
    id: "aie-ch2-q56",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements about training tokens in scaling laws are correct?",
    options: [
      {
        text: "They count the number of token positions the model has seen during training, not unique strings.",
        isCorrect: true,
      },
      {
        text: "For a fixed model size, more training tokens generally improve performance up to a point.",
        isCorrect: true,
      },
      {
        text: "Training on infinitely many tokens is always feasible in practice.",
        isCorrect: false,
      },
      {
        text: "Token count is irrelevant as long as the dataset is large in gigabytes.",
        isCorrect: false,
      },
    ],
    explanation:
      "Token count measures exposure; more tokens help until compute is exhausted or data quality becomes limiting. Infinite tokens or ignoring tokens in favor of file size is unrealistic.",
  },
  {
    id: "aie-ch2-q57",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about the Shoggoth-with-a-smiley-face analogy for post-training are correct?",
    options: [
      {
        text: "The pre-trained model is compared to a powerful but untamed ‘monster’.",
        isCorrect: true,
      },
      {
        text: "Supervised and preference finetuning are compared to putting a friendly ‘mask’ on this monster.",
        isCorrect: true,
      },
      {
        text: "The analogy suggests that pre-training data alone already guarantees safe behavior.",
        isCorrect: false,
      },
      {
        text: "The analogy implies that post-training removes all underlying problematic behaviors from the base model.",
        isCorrect: false,
      },
    ],
    explanation:
      "The analogy highlights that post-training shapes how the underlying pre-trained model appears, but doesn’t erase its underlying complexity or potential failure modes.",
  },
  {
    id: "aie-ch2-q58",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements about PPO (proximal policy optimization) in RLHF are correct?",
    options: [
      {
        text: "PPO constrains policy updates so that new policies don’t move too far from old ones in a single step.",
        isCorrect: true,
      },
      {
        text: "PPO uses a clipped objective to stabilize training and avoid overly large policy gradients.",
        isCorrect: true,
      },
      {
        text: "PPO is a supervised learning algorithm for directly predicting human ratings.",
        isCorrect: false,
      },
      {
        text: "PPO ignores the reward model and instead optimizes random behavior.",
        isCorrect: false,
      },
    ],
    explanation:
      "PPO is a policy-gradient RL algorithm with a clipped objective and trust-region-style behavior; in RLHF it optimizes the LM policy using the reward model.",
  },
  {
    id: "aie-ch2-q59",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about top-k and compute are correct?",
    options: [
      {
        text: "Top-k can reduce the cost of computing softmax by restricting to the highest-k logits.",
        isCorrect: true,
      },
      {
        text: "For very large vocabularies, computing softmax over all tokens can be expensive.",
        isCorrect: true,
      },
      {
        text: "Top-k always increases compute cost compared with full-vocabulary softmax.",
        isCorrect: false,
      },
      {
        text: "Top-k is unrelated to computational efficiency and only affects diversity.",
        isCorrect: false,
      },
    ],
    explanation:
      "Top-k is motivated partly by compute considerations: doing softmax over fewer tokens can reduce work, especially with huge vocabularies.",
  },
  {
    id: "aie-ch2-q60",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about hallucinations and user prompts are correct?",
    options: [
      {
        text: "Asking for very long, detailed answers increases opportunities for hallucination.",
        isCorrect: true,
      },
      {
        text: "Prompts that encourage the model to say ‘I don’t know’ can sometimes reduce hallucinations.",
        isCorrect: true,
      },
      {
        text: "Hallucinations cannot be influenced by prompts at all.",
        isCorrect: false,
      },
      {
        text: "Shorter, highly constrained prompts always eliminate hallucinations.",
        isCorrect: false,
      },
    ],
    explanation:
      "Prompt design affects hallucination risk but cannot guarantee zero hallucinations; constraints and ‘don’t know’ instructions can help reduce them.",
  },
  {
    id: "aie-ch2-q61",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements about the interaction between sampling and evaluation are correct?",
    options: [
      {
        text: "Changing sampling settings can change measured performance on benchmarks.",
        isCorrect: true,
      },
      {
        text: "Evaluation setups should document decoding parameters like temperature and top-p.",
        isCorrect: true,
      },
      {
        text: "Once a model is trained, its performance is independent of sampling strategy.",
        isCorrect: false,
      },
      {
        text: "It is safe to compare two models if they use entirely different, undocumented sampling strategies.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sampling influences outputs and hence evaluation results; fair comparisons require consistent, documented decoding settings.",
  },
  {
    id: "aie-ch2-q62",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about domain-specific models like medical LLMs are correct?",
    options: [
      {
        text: "They can combine general language capabilities with specialized medical knowledge.",
        isCorrect: true,
      },
      {
        text: "They usually require curated datasets derived from medical texts, guidelines, or records.",
        isCorrect: true,
      },
      {
        text: "They never need to worry about privacy or regulation.",
        isCorrect: false,
      },
      {
        text: "They are always trained only on purely synthetic data.",
        isCorrect: false,
      },
    ],
    explanation:
      "Medical and other domain LLMs build on curated, often sensitive data with strict privacy/regulatory constraints; synthetic data is supplementary, not the only source.",
  },
  {
    id: "aie-ch2-q63",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about context length are correct?",
    options: [
      {
        text: "It specifies how many tokens the model can attend to in one forward pass.",
        isCorrect: true,
      },
      {
        text: "Longer context windows allow conditioning on more information at once.",
        isCorrect: true,
      },
      {
        text: "Self-attention cost typically grows at least quadratically with context length in vanilla transformers.",
        isCorrect: false,
      },
      {
        text: "Short contexts always outperform longer contexts on tasks requiring long-range reasoning.",
        isCorrect: false,
      },
    ],
    explanation:
      "Context length bounds how much text the model sees; vanilla attention cost grows roughly quadratically with context, and longer contexts help long-range tasks.",
  },
  {
    id: "aie-ch2-q64",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements about AI-generated training data and copyright are correct?",
    options: [
      {
        text: "If a model reproduces copyrighted content from training data, it can raise legal and ethical concerns.",
        isCorrect: true,
      },
      {
        text: "Agreements with publishers and platforms can govern legitimate use of their data.",
        isCorrect: true,
      },
      {
        text: "Public web availability always implies that content is free to use for training without restrictions.",
        isCorrect: false,
      },
      {
        text: "Once data is used in training, copyright considerations no longer matter.",
        isCorrect: false,
      },
    ],
    explanation:
      "Copyright and licensing still apply to training data and potential memorized outputs; deals and ToS govern what can be used.",
  },
  {
    id: "aie-ch2-q65",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about the diversity of human preferences for RLHF data are correct?",
    options: [
      {
        text: "Different annotators may disagree about which response is better.",
        isCorrect: true,
      },
      {
        text: "Preference datasets often reflect the values and tastes of specific groups of labelers.",
        isCorrect: true,
      },
      {
        text: "It is trivial to encode a single universal notion of human preference in a reward model.",
        isCorrect: false,
      },
      {
        text: "Preference diversity is irrelevant when building aligned systems.",
        isCorrect: false,
      },
    ],
    explanation:
      "Human preference is heterogeneous; RLHF datasets encode specific value judgments and cannot capture a single universal preference function.",
  },
  {
    id: "aie-ch2-q66",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements about the mismatch between training data and deployment data are correct?",
    options: [
      {
        text: "Distribution shift can cause models to perform poorly even if training metrics were strong.",
        isCorrect: true,
      },
      {
        text: "Monitoring inputs and outputs in production can reveal new failure modes not seen during training.",
        isCorrect: true,
      },
      {
        text: "Training on web data automatically guarantees robustness to any domain.",
        isCorrect: false,
      },
      {
        text: "It is unnecessary to consider deployment conditions when designing training data.",
        isCorrect: false,
      },
    ],
    explanation:
      "Deployment environments can differ sharply from training corpora; robustness requires considering and monitoring those shifts.",
  },
  {
    id: "aie-ch2-q67",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about beam search are correct in the context of sequence generation?",
    options: [
      {
        text: "Beam search keeps track of multiple partial hypotheses at each decoding step.",
        isCorrect: true,
      },
      {
        text: "It expands only the most promising partial sequences according to some score.",
        isCorrect: true,
      },
      {
        text: "Beam search is guaranteed to find the globally optimal sequence in all neural sequence models.",
        isCorrect: false,
      },
      {
        text: "Beam search is equivalent to sampling independently from the softmax distribution at each step.",
        isCorrect: false,
      },
    ],
    explanation:
      "Beam search is a heuristic search over multiple hypotheses; it is not guaranteed optimal and differs from independent stochastic sampling.",
  },
  {
    id: "aie-ch2-q68",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about transformers’ attention mechanism are correct?",
    options: [
      {
        text: "Each token can attend to many other tokens in the sequence in a single layer.",
        isCorrect: true,
      },
      {
        text: "Attention weights indicate which tokens are most influential for predicting a given token.",
        isCorrect: true,
      },
      {
        text: "Attention eliminates the need for positional information entirely.",
        isCorrect: false,
      },
      {
        text: "Attention forces tokens to be processed strictly in order with no parallelism.",
        isCorrect: false,
      },
    ],
    explanation:
      "Self-attention lets tokens look at others in parallel; positional encodings are still needed to represent order.",
  },
  {
    id: "aie-ch2-q69",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about inconsistent outputs across different hardware or providers are correct?",
    options: [
      {
        text: "Floating-point arithmetic can differ slightly across hardware or libraries.",
        isCorrect: true,
      },
      {
        text: "These tiny differences can change which token is sampled when probabilities are close.",
        isCorrect: true,
      },
      {
        text: "Thus, exactly reproducing outputs across providers can be difficult even with the same prompts and settings.",
        isCorrect: true,
      },
      {
        text: "Hardware differences never affect probabilistic sampling behavior.",
        isCorrect: false,
      },
    ],
    explanation:
      "Small numerical differences can flip sampling decisions near ties; hardware and implementation details can therefore affect exact outputs.",
  },
  {
    id: "aie-ch2-q70",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements about combining retrieval with sampling are correct?",
    options: [
      {
        text: "Retrieved documents can be added to the prompt so the model conditions on grounded facts.",
        isCorrect: true,
      },
      {
        text: "Sampling hyperparameters still matter, even when retrieval is used.",
        isCorrect: true,
      },
      {
        text: "Retrieval guarantees that hallucinations cannot occur.",
        isCorrect: false,
      },
      {
        text: "Retrieval eliminates the need to think about the model’s training data distribution.",
        isCorrect: false,
      },
    ],
    explanation:
      "Retrieval helps ground generations but does not automatically prevent hallucinations; sampling and training data still matter.",
  },
  {
    id: "aie-ch2-q71",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "You want a model to generate more surprising, less obvious continuations. Which statements about adjusting temperature are correct?",
    options: [
      {
        text: "Increasing temperature above 1 tends to make rare tokens more likely.",
        isCorrect: true,
      },
      {
        text: "This can lead to more creative but also more error-prone outputs.",
        isCorrect: true,
      },
      {
        text: "Decreasing temperature is the best way to get surprising outputs.",
        isCorrect: false,
      },
      {
        text: "Temperature adjustments have no effect on the sampled tokens.",
        isCorrect: false,
      },
    ],
    explanation:
      "Higher temperature flattens distributions, allowing rare tokens to be sampled more often and increasing creativity at the cost of reliability.",
  },
  {
    id: "aie-ch2-q72",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about why sampling is central to understanding LLM behavior are correct?",
    options: [
      {
        text: "It explains why outputs can vary even when the model parameters are fixed.",
        isCorrect: true,
      },
      {
        text: "It connects model probabilities to actual observed outputs.",
        isCorrect: true,
      },
      {
        text: "It is irrelevant for understanding why hallucinations or inconsistencies happen.",
        isCorrect: false,
      },
      {
        text: "It only matters for classification models, not generative models.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sampling turns probability distributions into concrete text and is crucial to understanding variability, hallucinations, and inconsistencies.",
  },
  {
    id: "aie-ch2-q73",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements about the long-term future of scaling are correct?",
    options: [
      {
        text: "Continuing to scale models may face data, compute, and energy bottlenecks.",
        isCorrect: true,
      },
      {
        text: "Improvements may increasingly rely on better architectures, data curation, and training methods rather than just more parameters.",
        isCorrect: true,
      },
      {
        text: "There is universal agreement that scaling will remain the only important axis of progress.",
        isCorrect: false,
      },
      {
        text: "We can be certain that scaling trends observed so far will hold indefinitely.",
        isCorrect: false,
      },
    ],
    explanation:
      "Scaling has limits; future progress will likely rely on multiple axes: architecture, data, algorithms, and smarter use of compute rather than size alone.",
  },
  {
    id: "aie-ch2-q74",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about separating pre-training and post-training concerns in application design are correct?",
    options: [
      {
        text: "For many applications, you treat the base model as fixed and focus on prompts, retrieval, and sampling.",
        isCorrect: true,
      },
      {
        text: "Post-training choices (SFT, RLHF) help you decide which model family or variant to pick.",
        isCorrect: true,
      },
      {
        text: "As an application engineer, you typically re-run pre-training from scratch.",
        isCorrect: false,
      },
      {
        text: "You can ignore post-training details because all models of the same size behave identically.",
        isCorrect: false,
      },
    ],
    explanation:
      "Most application developers don’t touch pre-training; they choose among existing base+post-trained models and then design prompts, tools, and sampling around them.",
  },
  {
    id: "aie-ch2-q75",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about why this chapter emphasizes fundamentals (data, modeling, post-training, sampling) are correct?",
    options: [
      {
        text: "They underpin how different models behave in downstream applications.",
        isCorrect: true,
      },
      {
        text: "They help you interpret model documentation and release notes.",
        isCorrect: true,
      },
      {
        text: "They are only useful if you plan to train your own model from scratch.",
        isCorrect: false,
      },
      {
        text: "They are irrelevant once you know how to call an API.",
        isCorrect: false,
      },
    ],
    explanation:
      "Even API users benefit from understanding training data, model design, and sampling, because these factors explain behavior and inform system design choices.",
  },

  // ----------------------------------------------------------------------------
  // Q76–Q100: one correct, three incorrect (1 correct)
  // ----------------------------------------------------------------------------

  {
    id: "aie-ch2-q76",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statement is correct about training data for low-resource languages?",
    options: [
      {
        text: "It is often necessary to curate or collect additional data specifically for those languages.",
        isCorrect: true,
      },
      {
        text: "Low-resource languages always have more web text than English.",
        isCorrect: false,
      },
      {
        text: "Automatic translation from English fully replaces the need for native-language data.",
        isCorrect: false,
      },
      {
        text: "Low-resource languages are automatically handled well by any sufficiently large model.",
        isCorrect: false,
      },
    ],
    explanation:
      "Low-resource languages are underrepresented online; targeted curation or data collection is usually needed to improve performance.",
  },
  {
    id: "aie-ch2-q77",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement is correct about Common Crawl–style web corpora?",
    options: [
      {
        text: "They contain a mix of high-quality and very low-quality content and must be filtered.",
        isCorrect: true,
      },
      {
        text: "They only contain peer-reviewed scientific articles.",
        isCorrect: false,
      },
      {
        text: "They are manually curated by experts for factual accuracy.",
        isCorrect: false,
      },
      {
        text: "They completely avoid spam, clickbait, and malicious content.",
        isCorrect: false,
      },
    ],
    explanation:
      "Web crawls include all sorts of content; filtering and cleaning are essential before using them for training.",
  },
  {
    id: "aie-ch2-q78",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statement is correct about scaling laws for language models?",
    options: [
      {
        text: "They empirically relate model performance to parameters, data, and compute using approximate power-law relationships.",
        isCorrect: true,
      },
      {
        text: "They are exact analytic formulas derived from first principles.",
        isCorrect: false,
      },
      {
        text: "They show that performance is independent of training compute.",
        isCorrect: false,
      },
      {
        text: "They imply that once a model is large enough, training data quality no longer matters.",
        isCorrect: false,
      },
    ],
    explanation:
      "Scaling laws are empirically fitted curves that connect loss with size, data, and compute; they’re approximate, not exact derivations.",
  },
  {
    id: "aie-ch2-q79",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement is correct about supervised finetuning (SFT)?",
    options: [
      {
        text: "It trains the model to imitate high-quality responses on (prompt, response) examples.",
        isCorrect: true,
      },
      {
        text: "It discards the pre-trained weights and starts from random initialization.",
        isCorrect: false,
      },
      {
        text: "It always uses reinforcement learning algorithms like PPO.",
        isCorrect: false,
      },
      {
        text: "It requires that all training prompts are unlabeled.",
        isCorrect: false,
      },
    ],
    explanation:
      "SFT uses labeled demonstration pairs, building on pre-trained weights; it’s separate from RL-based methods like PPO.",
  },
  {
    id: "aie-ch2-q80",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statement is correct about the role of reward models (RMs) in RLHF?",
    options: [
      {
        text: "They provide a learned scalar signal that guides RL updates by scoring model responses.",
        isCorrect: true,
      },
      {
        text: "They replace the need for human feedback entirely during data collection.",
        isCorrect: false,
      },
      {
        text: "They guarantee that the policy model will never exploit their weaknesses.",
        isCorrect: false,
      },
      {
        text: "They are always simple linear models trained on raw token embeddings.",
        isCorrect: false,
      },
    ],
    explanation:
      "The RM approximates human preferences and supplies a reward signal, but it can be exploited and still depends on initial human preference data.",
  },
  {
    id: "aie-ch2-q81",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statement is correct about temperature in sampling?",
    options: [
      {
        text: "Lowering temperature makes the distribution more peaked and outputs more deterministic.",
        isCorrect: true,
      },
      {
        text: "Lowering temperature always makes the model more creative.",
        isCorrect: false,
      },
      {
        text: "Temperature is a training-only hyperparameter and cannot be changed at inference.",
        isCorrect: false,
      },
      {
        text: "Temperature has no effect once logits are computed.",
        isCorrect: false,
      },
    ],
    explanation:
      "Temperature is an inference-time control; lower values push sampling toward high-probability tokens and more deterministic behavior.",
  },
  {
    id: "aie-ch2-q82",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement is correct about top-p (nucleus) sampling?",
    options: [
      {
        text: "It selects the smallest set of tokens whose cumulative probability exceeds a threshold p.",
        isCorrect: true,
      },
      {
        text: "It always uses exactly p tokens, regardless of probabilities.",
        isCorrect: false,
      },
      {
        text: "It ignores token probabilities and chooses tokens uniformly.",
        isCorrect: false,
      },
      {
        text: "It is identical to greedy decoding when p = 0.9.",
        isCorrect: false,
      },
    ],
    explanation:
      "Top-p uses cumulative probability, not a fixed token count, and then samples within that nucleus distribution.",
  },
  {
    id: "aie-ch2-q83",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statement is correct about hallucinations in factual tasks?",
    options: [
      {
        text: "They are dangerous because the model can confidently state incorrect information that users may trust.",
        isCorrect: true,
      },
      {
        text: "They are harmless because users always verify everything the model says.",
        isCorrect: false,
      },
      {
        text: "They can be ignored as long as the text is grammatically correct.",
        isCorrect: false,
      },
      {
        text: "They only occur when the model has never seen similar examples during training.",
        isCorrect: false,
      },
    ],
    explanation:
      "Hallucinations are problematic precisely because outputs sound authoritative even when wrong; users may not verify them.",
  },
  {
    id: "aie-ch2-q84",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement is correct about caching in LLM systems?",
    options: [
      {
        text: "It can make repeated queries cheaper and more consistent by reusing previous responses.",
        isCorrect: true,
      },
      {
        text: "It forces the model to retrain before each response.",
        isCorrect: false,
      },
      {
        text: "It eliminates the need for any monitoring of outputs.",
        isCorrect: false,
      },
      {
        text: "It guarantees that hallucinations never occur again.",
        isCorrect: false,
      },
    ],
    explanation:
      "Caching is an engineering optimization; it reduces cost and variability for repeated queries but doesn’t change the underlying model behavior.",
  },
  {
    id: "aie-ch2-q85",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statement is correct about AI-generated training data and future models?",
    options: [
      {
        text: "If not handled carefully, recursively training on model outputs can degrade performance by drifting away from original human-generated distributions.",
        isCorrect: true,
      },
      {
        text: "Training solely on AI-generated data always improves diversity and robustness.",
        isCorrect: false,
      },
      {
        text: "Using AI-generated data automatically solves data scarcity for all domains.",
        isCorrect: false,
      },
      {
        text: "AI-generated data never appears on the web, so it is irrelevant for training.",
        isCorrect: false,
      },
    ],
    explanation:
      "Synthetic data must be used strategically; naïve recursive reuse risks collapsing diversity and harming performance.",
  },
  {
    id: "aie-ch2-q86",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statement is correct about the transformer’s advantage over RNNs?",
    options: [
      {
        text: "Transformers can process all tokens in parallel during training, enabling better hardware utilization.",
        isCorrect: true,
      },
      {
        text: "Transformers require strictly sequential processing of tokens at training time.",
        isCorrect: false,
      },
      {
        text: "Transformers cannot handle long-range dependencies.",
        isCorrect: false,
      },
      {
        text: "Transformers completely remove the need for large training datasets.",
        isCorrect: false,
      },
    ],
    explanation:
      "Parallelism over sequence positions is a key advantage of transformers, improving efficiency and enabling large-scale training.",
  },
  {
    id: "aie-ch2-q87",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement is correct about FLOPs in the context of training large models?",
    options: [
      {
        text: "They measure the approximate number of floating point operations and are a proxy for training compute cost.",
        isCorrect: true,
      },
      {
        text: "They only count memory accesses, not arithmetic operations.",
        isCorrect: false,
      },
      {
        text: "They are irrelevant for budgeting hardware and energy.",
        isCorrect: false,
      },
      {
        text: "They guarantee exact training time for any hardware configuration.",
        isCorrect: false,
      },
    ],
    explanation:
      "FLOPs provide a model- and hardware-agnostic proxy for compute cost, but actual time depends on hardware and implementation.",
  },
  {
    id: "aie-ch2-q88",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statement is correct about using prompts to reduce hallucinations?",
    options: [
      {
        text: "Prompts that instruct the model to state ‘I don’t know’ when uncertain can reduce, but not eliminate, hallucinations.",
        isCorrect: true,
      },
      {
        text: "Prompting guarantees that the model will never hallucinate again.",
        isCorrect: false,
      },
      {
        text: "Prompting has no influence on hallucinations whatsoever.",
        isCorrect: false,
      },
      {
        text: "Prompting can remove the need for any evaluation or monitoring.",
        isCorrect: false,
      },
    ],
    explanation:
      "Prompting is a useful mitigation, but hallucinations remain possible; evaluation and monitoring are still needed.",
  },
  {
    id: "aie-ch2-q89",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement is correct about beam search compared with sampling?",
    options: [
      {
        text: "Beam search focuses on a small set of high-scoring candidate sequences instead of sampling randomly at each step.",
        isCorrect: true,
      },
      {
        text: "Beam search is identical to temperature sampling with T = 1.",
        isCorrect: false,
      },
      {
        text: "Beam search always produces more diverse outputs than high-temperature sampling.",
        isCorrect: false,
      },
      {
        text: "Beam search cannot be used for text generation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Beam search is a heuristic search that tracks multiple high-score hypotheses; it is different from stochastic sampling.",
  },
  {
    id: "aie-ch2-q90",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statement is correct about why sampling is often described as ‘underrated’ in practice?",
    options: [
      {
        text: "Small changes in sampling strategy can significantly change model behavior without retraining.",
        isCorrect: true,
      },
      {
        text: "Sampling is irrelevant because models only ever use greedy decoding.",
        isCorrect: false,
      },
      {
        text: "Sampling hyperparameters have negligible impact on user experience.",
        isCorrect: false,
      },
      {
        text: "Sampling can be ignored once a model passes a benchmark.",
        isCorrect: false,
      },
    ],
    explanation:
      "Developers can often get large behavior changes by simply tuning sampling, making it a powerful yet underappreciated tool.",
  },
  {
    id: "aie-ch2-q91",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statement is correct about the role of evaluation when working with probabilistic models?",
    options: [
      {
        text: "You need systematic evaluation pipelines to detect failures and monitor changes as you adjust models and sampling.",
        isCorrect: true,
      },
      {
        text: "Once a model is deployed, you never need to evaluate it again.",
        isCorrect: false,
      },
      {
        text: "Probabilistic behavior makes evaluation impossible.",
        isCorrect: false,
      },
      {
        text: "Evaluation is only needed during academic research, not in production systems.",
        isCorrect: false,
      },
    ],
    explanation:
      "Probabilistic models require ongoing, systematic evaluation—especially as decoding, data, or usage patterns change.",
  },
  {
    id: "aie-ch2-q92",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement is correct about training data bottlenecks for future models?",
    options: [
      {
        text: "High-quality human-generated text is limited, and many web sources are becoming restricted for training.",
        isCorrect: true,
      },
      {
        text: "We have infinite high-quality labeled data for every domain.",
        isCorrect: false,
      },
      {
        text: "Copyright restrictions are disappearing, making scraping easier.",
        isCorrect: false,
      },
      {
        text: "Data availability grows automatically at the same rate as compute.",
        isCorrect: false,
      },
    ],
    explanation:
      "Data restrictions and finite human-generated text are a key bottleneck, leading to interest in proprietary and synthetic datasets.",
  },
  {
    id: "aie-ch2-q93",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statement is correct about GPU/accelerator availability for training large models?",
    options: [
      {
        text: "Hardware supply constraints can limit who can realistically train frontier-scale foundation models.",
        isCorrect: true,
      },
      {
        text: "Any individual can easily rent unlimited accelerators at negligible cost.",
        isCorrect: false,
      },
      {
        text: "Training compute is independent of hardware availability.",
        isCorrect: false,
      },
      {
        text: "Accelerator scarcity has no effect on the pace of AI research.",
        isCorrect: false,
      },
    ],
    explanation:
      "Accelerator hardware is expensive and limited, concentrating large-scale training in a small number of organizations.",
  },
  {
    id: "aie-ch2-q94",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement is correct about model size vs. deployment feasibility?",
    options: [
      {
        text: "Very large models may be difficult or impossible to deploy on edge devices due to memory and latency constraints.",
        isCorrect: true,
      },
      {
        text: "Larger models always have lower latency than smaller models.",
        isCorrect: false,
      },
      {
        text: "Deployment constraints are identical for 7B and 500B parameter models.",
        isCorrect: false,
      },
      {
        text: "Model size has no effect on inference hardware requirements.",
        isCorrect: false,
      },
    ],
    explanation:
      "Deployment constraints like memory, latency, and cost often push applications toward smaller or distilled models.",
  },
  {
    id: "aie-ch2-q95",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statement is correct about using the same model for many tasks?",
    options: [
      {
        text: "A single foundation model can be adapted via prompting, retrieval, and finetuning to many different applications.",
        isCorrect: true,
      },
      {
        text: "Each new task always requires training a completely new model from scratch.",
        isCorrect: false,
      },
      {
        text: "Foundation models cannot generalize beyond the task they were trained for.",
        isCorrect: false,
      },
      {
        text: "Using one model for multiple tasks is impossible due to parameter sharing.",
        isCorrect: false,
      },
    ],
    explanation:
      "A core benefit of foundation models is reuse: one model can be adapted to many tasks via prompts or additional training.",
  },
  {
    id: "aie-ch2-q96",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement is correct about how sampling affects user trust?",
    options: [
      {
        text: "Highly unstable or inconsistent outputs can erode user trust, even if the model is powerful.",
        isCorrect: true,
      },
      {
        text: "Users are always comfortable with completely unpredictable answers.",
        isCorrect: false,
      },
      {
        text: "Sampling settings cannot influence perceived reliability.",
        isCorrect: false,
      },
      {
        text: "User trust is unrelated to whether outputs are repeatable for similar queries.",
        isCorrect: false,
      },
    ],
    explanation:
      "Users expect some stability; wildly different answers for similar inputs often reduce trust in the system.",
  },
  {
    id: "aie-ch2-q97",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statement is correct about the relationship between sampling and creative applications?",
    options: [
      {
        text: "Higher temperatures and richer sampling strategies can be especially useful for brainstorming or creative writing.",
        isCorrect: true,
      },
      {
        text: "For creative tasks, greedy decoding is always ideal.",
        isCorrect: false,
      },
      {
        text: "Creative tasks never benefit from diversity in outputs.",
        isCorrect: false,
      },
      {
        text: "Sampling strategies are irrelevant if prompts are well written.",
        isCorrect: false,
      },
    ],
    explanation:
      "Creative tasks often benefit from diverse, surprising outputs, which sampling parameters strongly influence.",
  },
  {
    id: "aie-ch2-q98",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement is correct about using sampling variables across different tasks?",
    options: [
      {
        text: "You may want different temperature/top-p settings for coding assistance, factual Q&A, and story generation.",
        isCorrect: true,
      },
      {
        text: "All tasks should share exactly the same sampling configuration.",
        isCorrect: false,
      },
      {
        text: "Sampling variables have no impact on coding assistance.",
        isCorrect: false,
      },
      {
        text: "Sampling variables are only relevant for image generation models.",
        isCorrect: false,
      },
    ],
    explanation:
      "Different tasks have different trade-offs between creativity and reliability, so sampling settings are often task-specific.",
  },
  {
    id: "aie-ch2-q99",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statement is correct about the notion that ‘the model knows what it knows’?",
    options: [
      {
        text: "If this holds approximately, we might design reward functions and prompts that push models to decline answers when uncertain.",
        isCorrect: true,
      },
      {
        text: "It means models have perfect self-awareness like humans.",
        isCorrect: false,
      },
      {
        text: "It guarantees that models never hallucinate.",
        isCorrect: false,
      },
      {
        text: "It proves that reward modeling is unnecessary.",
        isCorrect: false,
      },
    ],
    explanation:
      "If models can estimate their own confidence, we can leverage that in prompts or reward models; it doesn’t imply human-like self-awareness or zero hallucinations.",
  },
  {
    id: "aie-ch2-q100",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statement is correct about understanding ML/DL fundamentals?",
    options: [
      {
        text: "They let you reason about model choice, sampling settings, and risk trade-offs instead of treating the model as a complete black box.",
        isCorrect: true,
      },
      {
        text: "They are only useful if you’re an academic researcher.",
        isCorrect: false,
      },
      {
        text: "They become obsolete once you have a working API key.",
        isCorrect: false,
      },
      {
        text: "They prevent you from changing any system settings safely.",
        isCorrect: false,
      },
    ],
    explanation:
      "Understanding training, scaling, post-training, and sampling helps you design, debug, and improve real applications more systematically.",
  },
];
