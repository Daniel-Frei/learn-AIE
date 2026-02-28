// lib/books/AIE_building_apps/chapter3.ts

import { Question } from "../../quiz";

export const aieChapter3Questions: Question[] = [
  // ----------------------------------------------------------------------------
  // Q1–Q25: all four options are correct (4 correct)
  // ----------------------------------------------------------------------------

  {
    id: "aie-ch3-q01",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Why is evaluating modern foundation models harder than evaluating classic supervised models?",
    options: [
      {
        text: "Their outputs are often open-ended, with many possible valid answers.",
        isCorrect: true,
      },
      {
        text: "They can require domain expertise to judge whether an answer is correct or useful.",
        isCorrect: true,
      },
      {
        text: "Benchmarks saturate quickly as models improve, so existing tests stop being discriminative.",
        isCorrect: true,
      },
      {
        text: "Many models are black boxes, so you mainly learn from their outputs instead of internals.",
        isCorrect: true,
      },
    ],
    explanation:
      "Open-ended outputs, need for expert judgment, benchmark saturation, and lack of transparency all make evaluation substantially more difficult than for fixed-label tasks.",
  },
  {
    id: "aie-ch3-q02",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements describe common failure modes that evaluation is supposed to detect in AI systems?",
    options: [
      {
        text: "Hallucinated or fabricated facts presented confidently.",
        isCorrect: true,
      },
      {
        text: "Toxic, biased, or otherwise harmful content.",
        isCorrect: true,
      },
      {
        text: "Subtle reasoning errors that are hard to spot by just ‘vibe checking’.",
        isCorrect: true,
      },
      {
        text: "Systematic blind spots on specific domains, languages, or user groups.",
        isCorrect: true,
      },
    ],
    explanation:
      "Evaluation is needed to uncover both obvious and subtle model failures, including hallucinations, harmful content, and systematic performance gaps.",
  },
  {
    id: "aie-ch3-q03",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements are true about entropy in the context of language and tokens?",
    options: [
      {
        text: "Entropy measures, on average, how much information a token carries.",
        isCorrect: true,
      },
      {
        text: "Higher entropy means each token is less predictable.",
        isCorrect: true,
      },
      {
        text: "A language with more possible tokens can have higher entropy.",
        isCorrect: true,
      },
      {
        text: "Entropy is closely related to how many bits are needed to represent each token.",
        isCorrect: true,
      },
    ],
    explanation:
      "Entropy quantifies average information and unpredictability; more uncertainty and more diverse tokens typically imply higher entropy and more bits per token.",
  },
  {
    id: "aie-ch3-q04",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe cross entropy for language models?",
    options: [
      {
        text: "Cross entropy measures how hard it is for a model to predict the next token in a dataset.",
        isCorrect: true,
      },
      {
        text: "It can be decomposed into data entropy plus a KL-divergence term between true and model distributions.",
        isCorrect: true,
      },
      {
        text: "Lower cross entropy indicates that the model’s distribution is closer to the true data distribution.",
        isCorrect: true,
      },
      {
        text: "If a model perfectly matches the data distribution, its cross entropy equals the data entropy.",
        isCorrect: true,
      },
    ],
    explanation:
      "Cross entropy H(P, Q) = H(P) + D_KL(P || Q); minimizing it both fits the data and reduces divergence between the model and the true distribution.",
  },
  {
    id: "aie-ch3-q05",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe perplexity for language models?",
    options: [
      {
        text: "Perplexity is an exponential transform of cross entropy.",
        isCorrect: true,
      },
      {
        text: "Lower perplexity means the model is less ‘confused’ about the next token.",
        isCorrect: true,
      },
      {
        text: "Perplexity can be interpreted as the effective average number of choices the model has for the next token.",
        isCorrect: true,
      },
      {
        text: "Perplexity depends on how the tokens are defined and how much context the model is allowed to see.",
        isCorrect: true,
      },
    ],
    explanation:
      "Perplexity (e.g., 2^{H}) summarizes predictive uncertainty; design choices like tokenization and context length directly affect its value.",
  },
  {
    id: "aie-ch3-q06",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "What are typical uses of perplexity for language models beyond training-time loss monitoring?",
    options: [
      {
        text: "As a rough proxy for downstream task performance when comparing base models.",
        isCorrect: true,
      },
      {
        text: "To check for possible benchmark contamination when perplexity on a test set is suspiciously low.",
        isCorrect: true,
      },
      {
        text: "To detect unusual or anomalous text, which tends to have very high perplexity.",
        isCorrect: true,
      },
      {
        text: "To help decide whether new text is sufficiently different from existing data before adding it to a training set.",
        isCorrect: true,
      },
    ],
    explanation:
      "Perplexity is useful for model comparison, contamination detection, anomaly detection, and data deduplication / selection decisions.",
  },
  {
    id: "aie-ch3-q07",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which factors tend to reduce a model’s perplexity on a given dataset (all else equal)?",
    options: [
      {
        text: "Giving the model a longer context window when computing probabilities.",
        isCorrect: true,
      },
      {
        text: "Training the model longer or on more relevant data from the same distribution.",
        isCorrect: true,
      },
      {
        text: "Using a smaller vocabulary of tokens for the same underlying text.",
        isCorrect: true,
      },
      {
        text: "Evaluating on structured text such as markup or source code instead of noisy everyday text.",
        isCorrect: true,
      },
    ],
    explanation:
      "More context, better training, smaller vocabularies, and more structured datasets all tend to make next-token prediction easier, lowering perplexity.",
  },
  {
    id: "aie-ch3-q08",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "For open-ended generation tasks, why is it often impossible to rely solely on simple ground-truth labels?",
    options: [
      {
        text: "There are many acceptable outputs for the same input.",
        isCorrect: true,
      },
      {
        text: "It is infeasible to enumerate all valid responses as references.",
        isCorrect: true,
      },
      {
        text: "Different users may legitimately prefer different styles or trade-offs in the answer.",
        isCorrect: true,
      },
      {
        text: "The space of possible outputs grows combinatorially with length, making exhaustive labeling unrealistic.",
        isCorrect: true,
      },
    ],
    explanation:
      "Open-ended tasks have huge output spaces and multiple valid answers, so exact supervised labels cover only a tiny subset of acceptable responses.",
  },
  {
    id: "aie-ch3-q09",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements are true about functional correctness as an evaluation method?",
    options: [
      {
        text: "It checks whether a system actually achieves the intended real-world effect or output.",
        isCorrect: true,
      },
      {
        text: "It is often implemented via unit tests and execution-based checks for code and programs.",
        isCorrect: true,
      },
      {
        text: "For some tasks, it can be fully automated by comparing behavior against test cases.",
        isCorrect: true,
      },
      {
        text: "It is conceptually the most important metric because it reflects whether the application works.",
        isCorrect: true,
      },
    ],
    explanation:
      "Functional correctness directly measures whether an application does its job; for code or SQL generation it’s often enforced by running tests.",
  },
  {
    id: "aie-ch3-q10",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe pass@k metrics used in code-generation benchmarks?",
    options: [
      {
        text: "A problem is considered solved if at least one of the k sampled programs passes all tests.",
        isCorrect: true,
      },
      {
        text: "Higher k generally increases the chance that a model solves a problem.",
        isCorrect: true,
      },
      {
        text: "pass@k is reported as the fraction of problems solved among all benchmark tasks.",
        isCorrect: true,
      },
      {
        text: "pass@1 is usually lower than pass@k for k > 1 on the same model and benchmark.",
        isCorrect: true,
      },
    ],
    explanation:
      "pass@k summarizes the probability that k sampled generations contain at least one fully correct solution across benchmark problems.",
  },
  {
    id: "aie-ch3-q11",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements are true about exact-match evaluation for generated text?",
    options: [
      {
        text: "It works best when short, unambiguous outputs are expected.",
        isCorrect: true,
      },
      {
        text: "Variations in formatting or phrasing can cause correct answers to be scored as wrong unless normalization is applied.",
        isCorrect: true,
      },
      {
        text: "It often fails for tasks like translation or summarization where many different phrasings are acceptable.",
        isCorrect: true,
      },
      {
        text: "Even when used, it is often supplemented with more flexible similarity metrics.",
        isCorrect: true,
      },
    ],
    explanation:
      "Exact match is simple and precise, but brittle whenever multiple correct wordings exist or formatting details differ.",
  },
  {
    id: "aie-ch3-q12",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements describe lexical similarity metrics such as BLEU or ROUGE?",
    options: [
      {
        text: "They operate on surface tokens like words, subwords, or character n-grams.",
        isCorrect: true,
      },
      {
        text: "They reward overlapping tokens or n-grams between a generated answer and reference text.",
        isCorrect: true,
      },
      {
        text: "They can mis-score good answers if references do not cover certain valid phrasings.",
        isCorrect: true,
      },
      {
        text: "They were widely used in tasks like machine translation and summarization before large language models.",
        isCorrect: true,
      },
    ],
    explanation:
      "Lexical metrics quantify surface overlap with references but can misjudge semantic quality if phrasing differs or references are incomplete.",
  },
  {
    id: "aie-ch3-q13",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe semantic similarity using embeddings?",
    options: [
      {
        text: "Each text is mapped to a vector embedding in some high-dimensional space.",
        isCorrect: true,
      },
      {
        text: "Similarity is computed using functions like cosine similarity on those embeddings.",
        isCorrect: true,
      },
      {
        text: "Semantically similar sentences are intended to lie close together in the embedding space.",
        isCorrect: true,
      },
      {
        text: "Embedding-based similarity is often more robust than pure lexical overlap when wordings differ.",
        isCorrect: true,
      },
    ],
    explanation:
      "Embedding methods represent meaning as vectors; distance or cosine similarity between vectors is used to judge semantic closeness between texts.",
  },
  {
    id: "aie-ch3-q14",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements are advantages of embedding-based similarity over purely lexical metrics?",
    options: [
      {
        text: "They can recognize paraphrases with few shared words.",
        isCorrect: true,
      },
      {
        text: "They support cross-modal comparisons when embeddings are shared across modalities.",
        isCorrect: true,
      },
      {
        text: "They can be reused for retrieval, clustering, and anomaly detection beyond evaluation.",
        isCorrect: true,
      },
      {
        text: "Good embedding models provide a common building block for many downstream tasks.",
        isCorrect: true,
      },
    ],
    explanation:
      "Embeddings offer flexible semantic representations that generalize beyond exact wording and can power multiple application components.",
  },
  {
    id: "aie-ch3-q15",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements are true about joint multimodal embedding spaces such as in CLIP-style models?",
    options: [
      {
        text: "They map different modalities (e.g. images and text) into a shared vector space.",
        isCorrect: true,
      },
      {
        text: "Training typically brings paired items (e.g. an image and its caption) closer together in that space.",
        isCorrect: true,
      },
      {
        text: "They enable cross-modal retrieval, such as searching images using text queries.",
        isCorrect: true,
      },
      {
        text: "They allow similarity comparisons between modalities without converting everything back into raw pixels or tokens.",
        isCorrect: true,
      },
    ],
    explanation:
      "Joint embedding spaces align heterogeneous data types into one space, enabling cross-modal similarity search and downstream multimodal applications.",
  },
  {
    id: "aie-ch3-q16",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the role of embedding quality in downstream evaluation and applications?",
    options: [
      {
        text: "Higher-quality embeddings tend to improve retrieval and RAG performance for the same index.",
        isCorrect: true,
      },
      {
        text: "Weak embeddings can cause semantically similar items to appear far apart, harming evaluation based on semantic similarity.",
        isCorrect: true,
      },
      {
        text: "Embedding models are themselves evaluated on suites of tasks (e.g. classification, clustering, semantic similarity).",
        isCorrect: true,
      },
      {
        text: "Embedding choice can materially affect metrics like BERTScore or other embedding-based similarity scores.",
        isCorrect: true,
      },
    ],
    explanation:
      "Since semantic metrics and many systems depend on embeddings, their quality directly influences retrieval accuracy, similarity evaluation, and benchmark scores.",
  },
  {
    id: "aie-ch3-q17",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements are true about using human-generated reference data for evaluation?",
    options: [
      {
        text: "It can be expensive and slow to create high-quality references.",
        isCorrect: true,
      },
      {
        text: "Human references implicitly define what ‘good’ looks like for a task.",
        isCorrect: true,
      },
      {
        text: "Reference data can itself contain errors or biases.",
        isCorrect: true,
      },
      {
        text: "Coverage limitations mean some correct model answers may be penalized as incorrect.",
        isCorrect: true,
      },
    ],
    explanation:
      "Human-written references are costly, imperfect, and incomplete; evaluation scores reflect both model behavior and the quality of the reference set.",
  },
  {
    id: "aie-ch3-q18",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements describe why reference-free evaluation metrics are attractive for large-scale systems?",
    options: [
      {
        text: "You do not need to collect labeled outputs for every production query.",
        isCorrect: true,
      },
      {
        text: "They can be applied directly to live traffic where no ground truth exists.",
        isCorrect: true,
      },
      {
        text: "They reduce dependence on fixed test sets that may become outdated or contaminated.",
        isCorrect: true,
      },
      {
        text: "They can be combined with spot-checking to trade off cost against coverage.",
        isCorrect: true,
      },
    ],
    explanation:
      "Reference-free metrics scale better to production, avoid constant labeling, and complement small, curated test sets.",
  },
  {
    id: "aie-ch3-q19",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements are true about using AI as a judge (AI-judged evaluation)?",
    options: [
      {
        text: "You can ask a judge model to score answers according to arbitrary criteria described in natural language.",
        isCorrect: true,
      },
      {
        text: "AI judges can operate without human references, evaluating answers directly in context.",
        isCorrect: true,
      },
      {
        text: "Agreement with human raters can be surprisingly high for well-chosen models and prompts.",
        isCorrect: true,
      },
      {
        text: "AI-judged evaluation is increasingly common in both research and production settings.",
        isCorrect: true,
      },
    ],
    explanation:
      "AI judges are flexible, scalable, and empirically correlate well with human judgments when carefully configured, so they are widely adopted despite limitations.",
  },
  {
    id: "aie-ch3-q20",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which design choices matter when configuring an AI judge for a particular evaluation task?",
    options: [
      {
        text: "The underlying model used as the judge.",
        isCorrect: true,
      },
      {
        text: "The judge prompt, including definitions of criteria and scoring scales.",
        isCorrect: true,
      },
      {
        text: "Sampling parameters such as temperature that affect determinism and variance.",
        isCorrect: true,
      },
      {
        text: "Whether examples (few-shot cases) are included to anchor the scoring behavior.",
        isCorrect: true,
      },
    ],
    explanation:
      "An AI judge is effectively a complete system: model choice, prompt, sampling configuration, and examples all influence the resulting scores.",
  },
  {
    id: "aie-ch3-q21",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements correctly describe known biases of AI judges?",
    options: [
      {
        text: "A model often prefers its own generations over those from other models (self-bias).",
        isCorrect: true,
      },
      {
        text: "Judges can show position bias, consistently preferring answers that appear in a certain order.",
        isCorrect: true,
      },
      {
        text: "Many judge models favor longer, more verbose answers even when they contain mistakes.",
        isCorrect: true,
      },
      {
        text: "Bias behavior can vary across judge architectures and may diminish as models improve, but it does not disappear automatically.",
        isCorrect: true,
      },
    ],
    explanation:
      "Self-bias, position bias, and verbosity bias are widely observed; understanding them is key when interpreting AI-judged comparisons.",
  },
  {
    id: "aie-ch3-q22",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements are best practices when using AI judges in a real application?",
    options: [
      {
        text: "Inspect the judge’s prompt and model; avoid opaque ‘black-box’ evaluation endpoints when possible.",
        isCorrect: true,
      },
      {
        text: "Periodically verify judge scores against human spot checks to detect drift or failure modes.",
        isCorrect: true,
      },
      {
        text: "Keep the judge configuration stable when tracking metric trends over time.",
        isCorrect: true,
      },
      {
        text: "Document which evaluation configuration produced which historical metrics.",
        isCorrect: true,
      },
    ],
    explanation:
      "Treat judges as versioned components: make them transparent, stable, and periodically calibrated against human judgments.",
  },
  {
    id: "aie-ch3-q23",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe cost and latency trade-offs when using AI as a judge?",
    options: [
      {
        text: "Evaluating every response with a strong model can multiply inference cost significantly.",
        isCorrect: true,
      },
      {
        text: "Using smaller or cheaper models as judges can reduce cost at the risk of lower alignment with humans.",
        isCorrect: true,
      },
      {
        text: "Guardrail-style online judging can add noticeable latency if placed on the critical user path.",
        isCorrect: true,
      },
      {
        text: "Spot-checking only a fraction of responses can control costs while still providing useful monitoring signals.",
        isCorrect: true,
      },
    ],
    explanation:
      "Each additional judge call adds cost and latency; smaller models and sampling can help, and partial evaluation can still yield actionable signals.",
  },
  {
    id: "aie-ch3-q24",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Why is it important to distinguish exact evaluation methods from subjective ones?",
    options: [
      {
        text: "Exact methods yield deterministic, reproducible scores given the same inputs.",
        isCorrect: true,
      },
      {
        text: "Subjective methods depend on human or AI judgments that can vary over time.",
        isCorrect: true,
      },
      {
        text: "Systematic monitoring often relies on relatively stable exact metrics where possible.",
        isCorrect: true,
      },
      {
        text: "Understanding which metrics are subjective helps avoid over-interpreting small changes in scores.",
        isCorrect: true,
      },
    ],
    explanation:
      "Knowing whether a metric is exact or judgment-based clarifies how to interpret changes, confidence intervals, and reproducibility.",
  },
  {
    id: "aie-ch3-q25",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements are good reasons to invest seriously in evaluation when building AI applications?",
    options: [
      {
        text: "It helps prevent catastrophic failures that damage users and trust.",
        isCorrect: true,
      },
      {
        text: "It supports faster iteration by giving clear feedback on which changes help.",
        isCorrect: true,
      },
      {
        text: "It provides evidence to regulators, partners, or management that the system is under control.",
        isCorrect: true,
      },
      {
        text: "It reveals opportunities for product improvement that are invisible from logs alone.",
        isCorrect: true,
      },
    ],
    explanation:
      "Robust evaluation both reduces risk and accelerates product learning, making it central to responsible AI engineering.",
  },

  // ----------------------------------------------------------------------------
  // Q26–Q50: exactly three correct options (3 correct)
  // ----------------------------------------------------------------------------

  {
    id: "aie-ch3-q26",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements about entropy and predictability of a language are correct?",
    options: [
      {
        text: "Lower entropy implies the next token is easier to guess on average.",
        isCorrect: true,
      },
      {
        text: "A perfectly predictable sequence has zero entropy.",
        isCorrect: true,
      },
      {
        text: "Higher entropy means each token tends to carry more new information.",
        isCorrect: true,
      },
      {
        text: "Entropy must always increase as you add more training data from the same language.",
        isCorrect: false,
      },
    ],
    explanation:
      "Lower entropy corresponds to easier prediction; zero entropy would mean completely predictable text. Entropy need not grow just because more data is observed.",
  },
  {
    id: "aie-ch3-q27",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about bits-per-character (BPC) and bits-per-byte (BPB) are correct?",
    options: [
      {
        text: "They normalize cross entropy to be independent of a specific tokenizer’s token boundaries.",
        isCorrect: true,
      },
      {
        text: "Bits-per-character divides bits per token by average characters per token.",
        isCorrect: true,
      },
      {
        text: "Bits-per-byte accounts for how characters are encoded into bytes (e.g. UTF-8).",
        isCorrect: true,
      },
      {
        text: "Bits-per-byte is always numerically equal to perplexity.",
        isCorrect: false,
      },
    ],
    explanation:
      "BPC and BPB re-express entropy in units of characters or bytes; perplexity is an exponential transform, not the same numeric quantity.",
  },
  {
    id: "aie-ch3-q28",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "A model’s perplexity on dataset A is lower than on dataset B. Which interpretations are reasonable?",
    options: [
      {
        text: "Dataset A is likely more predictable or similar to the training distribution than dataset B.",
        isCorrect: true,
      },
      {
        text: "The model may have memorized parts of dataset A more than dataset B.",
        isCorrect: true,
      },
      {
        text: "For some tasks, lower perplexity on A suggests the model may perform better on A-related downstream tasks.",
        isCorrect: true,
      },
      {
        text: "It proves that A is a higher-quality dataset than B for any purpose.",
        isCorrect: false,
      },
    ],
    explanation:
      "Perplexity reflects predictability and distribution match, not universal ‘quality’; many factors beyond perplexity matter for downstream utility.",
  },
  {
    id: "aie-ch3-q29",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about using perplexity to detect data contamination are correct?",
    options: [
      {
        text: "Very low perplexity on an evaluation set can indicate that the set was included in training.",
        isCorrect: true,
      },
      {
        text: "Comparing perplexity across suspected and clean subsets can surface contamination patterns.",
        isCorrect: true,
      },
      {
        text: "Perplexity alone cannot prove contamination but can provide strong hints to investigate further.",
        isCorrect: true,
      },
      {
        text: "If perplexity is high, it guarantees that the evaluation set was not seen during training.",
        isCorrect: false,
      },
    ],
    explanation:
      "Low perplexity is a useful signal of overlap, but high perplexity does not strictly rule out partial contamination.",
  },
  {
    id: "aie-ch3-q30",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which properties make functional correctness especially appealing for automatic evaluation of code generation?",
    options: [
      {
        text: "You can run the generated code on predefined test cases.",
        isCorrect: true,
      },
      {
        text: "For each test, the outcome is objectively pass or fail.",
        isCorrect: true,
      },
      {
        text: "You do not need human raters for each generated program.",
        isCorrect: true,
      },
      {
        text: "It works equally well for any open-ended creative writing task.",
        isCorrect: false,
      },
    ],
    explanation:
      "Execution-based tests work well for code and other formally specified tasks, but not for creative tasks without precise test oracles.",
  },
  {
    id: "aie-ch3-q31",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about lexical similarity metrics such as BLEU are correct?",
    options: [
      {
        text: "They consider n-gram overlap between candidate and reference texts.",
        isCorrect: true,
      },
      {
        text: "They can penalize valid paraphrases that use different wording.",
        isCorrect: true,
      },
      {
        text: "They may correlate poorly with functional correctness in code-generation benchmarks.",
        isCorrect: true,
      },
      {
        text: "They were originally designed to evaluate how well models minimize perplexity.",
        isCorrect: false,
      },
    ],
    explanation:
      "BLEU and relatives were developed for translation quality estimation based on n-gram overlap, not as direct proxies for perplexity.",
  },
  {
    id: "aie-ch3-q32",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about exact-match evaluation variants are correct?",
    options: [
      {
        text: "Some variants treat outputs that contain the reference answer as correct (substring match).",
        isCorrect: true,
      },
      {
        text: "Substring matching can incorrectly mark wrong answers as correct when they merely mention the right token.",
        isCorrect: true,
      },
      {
        text: "Normalization (e.g. stripping punctuation and case) can reduce spurious mismatches.",
        isCorrect: true,
      },
      {
        text: "Exact match with no normalization avoids all evaluation errors.",
        isCorrect: false,
      },
    ],
    explanation:
      "Relaxed matching and normalization help, but each choice introduces trade-offs; no variant is error-free for all tasks.",
  },
  {
    id: "aie-ch3-q33",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements about embeddings in general are correct?",
    options: [
      {
        text: "They represent complex objects (e.g. text, images) as numeric vectors.",
        isCorrect: true,
      },
      {
        text: "They usually have dimension in the hundreds or thousands rather than one or two.",
        isCorrect: true,
      },
      {
        text: "They are designed so that geometric relationships reflect semantic relationships.",
        isCorrect: true,
      },
      {
        text: "They must always be one-dimensional scalars to be interpretable.",
        isCorrect: false,
      },
    ],
    explanation:
      "Embeddings are high-dimensional vectors whose geometry approximates semantic similarity; scalar representations would be far too limited.",
  },
  {
    id: "aie-ch3-q34",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about cosine similarity for embeddings are correct?",
    options: [
      {
        text: "It measures the angle between two vectors, ignoring their magnitude.",
        isCorrect: true,
      },
      {
        text: "Values range from –1 (opposite direction) to +1 (same direction).",
        isCorrect: true,
      },
      {
        text: "It is often preferred because it is invariant to uniform scaling of vectors.",
        isCorrect: true,
      },
      {
        text: "It is equivalent to simple Euclidean distance in all cases.",
        isCorrect: false,
      },
    ],
    explanation:
      "Cosine similarity focuses on direction; Euclidean distance depends on magnitude as well, so they are not identical.",
  },
  {
    id: "aie-ch3-q35",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about evaluating embedding models themselves are correct?",
    options: [
      {
        text: "They can be benchmarked on tasks like clustering, retrieval, and classification.",
        isCorrect: true,
      },
      {
        text: "Benchmarks such as large text-embedding suites aggregate performance across multiple tasks.",
        isCorrect: true,
      },
      {
        text: "Embedding quality is often judged indirectly by downstream task performance rather than a single theoretical metric.",
        isCorrect: true,
      },
      {
        text: "Once a model has good perplexity, its embeddings are guaranteed to be optimal for all tasks.",
        isCorrect: false,
      },
    ],
    explanation:
      "Good language-model perplexity does not guarantee task-specific embedding quality; separate eval suites remain necessary.",
  },
  {
    id: "aie-ch3-q36",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about using AI judges without reference data are correct?",
    options: [
      {
        text: "You can evaluate answers based only on the question and some rubric in the prompt.",
        isCorrect: true,
      },
      {
        text: "This is useful when acceptable outputs are diverse and not easy to enumerate.",
        isCorrect: true,
      },
      {
        text: "The resulting score depends heavily on how clearly the rubric is described.",
        isCorrect: true,
      },
      {
        text: "It eliminates all subjectivity, because the judge is an algorithm.",
        isCorrect: false,
      },
    ],
    explanation:
      "AI judges remain subjective; their judgments depend on prompt wording and model behavior even without references.",
  },
  {
    id: "aie-ch3-q37",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements about why teams still need humans in evaluation are correct?",
    options: [
      {
        text: "Human review is needed to sanity-check AI-judged metrics and catch new failure modes.",
        isCorrect: true,
      },
      {
        text: "Some subtle harms, such as cultural offense, are hard to capture with existing automatic metrics.",
        isCorrect: true,
      },
      {
        text: "High-stakes domains may legally or ethically require human oversight.",
        isCorrect: true,
      },
      {
        text: "Humans can be fully replaced by AI judges as soon as those judges reach 80% agreement.",
        isCorrect: false,
      },
    ],
    explanation:
      "Even strong AI judges cannot fully replace human judgment, especially for nuanced or high-stakes decisions.",
  },
  {
    id: "aie-ch3-q38",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about building an evaluation pipeline for an AI application are correct?",
    options: [
      {
        text: "You typically combine several metrics, not just one.",
        isCorrect: true,
      },
      {
        text: "You should focus on the parts of the system where failure would be most costly.",
        isCorrect: true,
      },
      {
        text: "Evaluation design may require modifying the system to expose more signals or logs.",
        isCorrect: true,
      },
      {
        text: "Once chosen, metrics should never be revisited or refined.",
        isCorrect: false,
      },
    ],
    explanation:
      "Evaluation is iterative; metrics and logging evolve as you discover new failure modes and improve your system.",
  },
  {
    id: "aie-ch3-q39",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements about consistency of AI judges are correct?",
    options: [
      {
        text: "Sampling randomness can cause the same judge configuration to output different scores on repeated runs.",
        isCorrect: true,
      },
      {
        text: "Adding few-shot examples to the judge prompt can increase consistency across runs.",
        isCorrect: true,
      },
      {
        text: "Higher consistency does not automatically imply higher alignment with human ground truth.",
        isCorrect: true,
      },
      {
        text: "Setting temperature high usually makes scores more deterministic.",
        isCorrect: false,
      },
    ],
    explanation:
      "Lower temperature and better prompting can increase consistency, but being consistently wrong is still possible.",
  },
  {
    id: "aie-ch3-q40",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about standardized evaluation criteria across tools are correct?",
    options: [
      {
        text: "Different tools may use the same label (e.g. ‘faithfulness’) but define it differently in their judge prompts.",
        isCorrect: true,
      },
      {
        text: "Score scales (e.g. 0/1 vs 1–5) can differ even for nominally similar criteria.",
        isCorrect: true,
      },
      {
        text: "Because of this, you cannot assume that ‘faithfulness = 0.8’ means the same thing across two platforms.",
        isCorrect: true,
      },
      {
        text: "The community already has a single standard specification that all providers strictly follow.",
        isCorrect: false,
      },
    ],
    explanation:
      "Criterion names are not standardized; practitioners must read prompts and documentation to interpret scores correctly.",
  },
  {
    id: "aie-ch3-q41",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements about lexical vs semantic similarity are correct?",
    options: [
      {
        text: "Lexical similarity focuses on surface form overlap (e.g. shared words or n-grams).",
        isCorrect: true,
      },
      {
        text: "Semantic similarity focuses on meaning, often via embeddings.",
        isCorrect: true,
      },
      {
        text: "Two texts can be lexically dissimilar but semantically similar.",
        isCorrect: true,
      },
      {
        text: "Two texts that share many words necessarily have the same meaning.",
        isCorrect: false,
      },
    ],
    explanation:
      "Lexical overlap is neither necessary nor sufficient for semantic equivalence; semantic similarity attempts to capture meaning directly.",
  },
  {
    id: "aie-ch3-q42",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about reference-based vs reference-free evaluation approaches are correct?",
    options: [
      {
        text: "Reference-based methods compare outputs to one or more canonical answers.",
        isCorrect: true,
      },
      {
        text: "Reference-free methods rely on criteria such as quality or consistency without explicit ground truth answers.",
        isCorrect: true,
      },
      {
        text: "Reference-based approaches are often more precise when good references are available.",
        isCorrect: true,
      },
      {
        text: "Reference-free methods cannot be combined with reference-based methods.",
        isCorrect: false,
      },
    ],
    explanation:
      "Practitioners mix both approaches; references provide precise targets where available, while reference-free metrics scale to production.",
  },
  {
    id: "aie-ch3-q43",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about designing evaluation suites for a specific application are correct?",
    options: [
      {
        text: "You should weight evaluation more heavily toward realistically frequent and high-impact scenarios.",
        isCorrect: true,
      },
      {
        text: "Synthetic test cases can be useful but should be complemented by real user data where possible.",
        isCorrect: true,
      },
      {
        text: "Coverage of rare but catastrophic failures may deserve targeted tests even if they are unlikely.",
        isCorrect: true,
      },
      {
        text: "A single generic public benchmark is usually sufficient on its own.",
        isCorrect: false,
      },
    ],
    explanation:
      "Application-specific evaluation must reflect actual usage patterns and risk profiles; generic benchmarks are only one ingredient.",
  },
  {
    id: "aie-ch3-q44",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about using AI judges as production guardrails are correct?",
    options: [
      {
        text: "They can screen out obviously unsafe or low-quality responses before they reach users.",
        isCorrect: true,
      },
      {
        text: "They may delay responses, since an extra model call is added on the critical path.",
        isCorrect: true,
      },
      {
        text: "They can be configured to route questionable outputs for human review instead of outright blocking.",
        isCorrect: true,
      },
      {
        text: "They guarantee that no harmful content will ever reach users.",
        isCorrect: false,
      },
    ],
    explanation:
      "Guardrails reduce risk but cannot guarantee absolute safety; residual errors and trade-offs between latency and coverage remain.",
  },
  {
    id: "aie-ch3-q45",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements about ad hoc ‘vibe check’ evaluation are correct?",
    options: [
      {
        text: "It relies on a few hand-picked prompts inspected manually.",
        isCorrect: true,
      },
      {
        text: "It is quick and informal, so it can be useful at the very beginning of prototyping.",
        isCorrect: true,
      },
      {
        text: "It does not scale or provide reliable signals for serious iteration.",
        isCorrect: true,
      },
      {
        text: "It is generally sufficient on its own for high-stakes deployments.",
        isCorrect: false,
      },
    ],
    explanation:
      "Vibe checks are helpful early on but provide neither coverage nor statistical reliability; more systematic evaluation is needed for real systems.",
  },
  {
    id: "aie-ch3-q46",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about edit distance–based fuzzy matching are correct?",
    options: [
      {
        text: "It counts how many insertions, deletions, or substitutions are needed to convert one string into another.",
        isCorrect: true,
      },
      {
        text: "Some variants also treat transposition of adjacent characters as a single edit.",
        isCorrect: true,
      },
      {
        text: "Strings with smaller edit distance are considered more similar.",
        isCorrect: true,
      },
      {
        text: "Edit distance naturally captures deep semantic relationships between sentences.",
        isCorrect: false,
      },
    ],
    explanation:
      "Fuzzy matching is useful for approximate string similarity but still operates on surface form and does not capture high-level meaning.",
  },
  {
    id: "aie-ch3-q47",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about designing scoring scales for AI judges are correct?",
    options: [
      {
        text: "Classification-style outputs (e.g. good/bad) are often easier for models than regressing a fine-grained continuous score.",
        isCorrect: true,
      },
      {
        text: "Discrete numeric scales like 1–5 can work well if the rubric clearly defines each level.",
        isCorrect: true,
      },
      {
        text: "Very wide discrete ranges (e.g. 1–100) tend to be harder for models to use consistently.",
        isCorrect: true,
      },
      {
        text: "Continuous 0–1 scores always yield more reliable judgments than categorical labels.",
        isCorrect: false,
      },
    ],
    explanation:
      "Models generally handle categorical or small discrete scales better than fine-grained continuous scoring without strong anchoring.",
  },
  {
    id: "aie-ch3-q48",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about monitoring evaluation metrics over time are correct?",
    options: [
      {
        text: "Metric drift can be caused either by model changes or by changes in the evaluation configuration.",
        isCorrect: true,
      },
      {
        text: "Tracking versions of both your application and your judges is necessary to interpret trends.",
        isCorrect: true,
      },
      {
        text: "Large sudden jumps in a subjective metric should trigger an investigation rather than be blindly celebrated.",
        isCorrect: true,
      },
      {
        text: "Once you have a dashboard, you can safely ignore how metrics are computed internally.",
        isCorrect: false,
      },
    ],
    explanation:
      "Interpreting metric trends requires understanding both model updates and evolving evaluation setups; dashboards alone are not enough.",
  },
  {
    id: "aie-ch3-q49",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about using AI judges for pairwise model comparisons are correct?",
    options: [
      {
        text: "The judge can be asked to choose which of two answers is better according to a rubric.",
        isCorrect: true,
      },
      {
        text: "Win–loss statistics across many prompts can produce a leaderboard of models.",
        isCorrect: true,
      },
      {
        text: "Self-bias may inflate a model’s apparent performance when it acts as both generator and judge.",
        isCorrect: true,
      },
      {
        text: "Pairwise comparisons are impossible when you lack human-written references.",
        isCorrect: false,
      },
    ],
    explanation:
      "AI judges enable reference-free pairwise comparisons, but bias and rubric design critically shape leaderboard interpretations.",
  },
  {
    id: "aie-ch3-q50",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about combining human and AI-based evaluation are correct?",
    options: [
      {
        text: "Humans can focus on edge cases and high-risk scenarios while AI covers bulk traffic.",
        isCorrect: true,
      },
      {
        text: "AI-judged scores can help prioritize which examples to show to human reviewers.",
        isCorrect: true,
      },
      {
        text: "Disagreements between humans and AI judges can reveal important blind spots.",
        isCorrect: true,
      },
      {
        text: "Once AI judges are in place, human feedback is no longer useful.",
        isCorrect: false,
      },
    ],
    explanation:
      "Hybrid evaluation leverages AI for scale and humans for nuanced judgment, using disagreements as valuable debugging signals.",
  },

  // ----------------------------------------------------------------------------
  // Q51–Q75: exactly two correct options (2 correct)
  // ----------------------------------------------------------------------------

  {
    id: "aie-ch3-q51",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe why benchmarks ‘saturate’ for powerful models?",
    options: [
      {
        text: "Models eventually achieve near-perfect scores, leaving little room to distinguish improvements.",
        isCorrect: true,
      },
      {
        text: "Benchmarks rarely capture newly emerging capabilities that future models might have.",
        isCorrect: true,
      },
      {
        text: "Saturation means the benchmark is mathematically impossible to solve.",
        isCorrect: false,
      },
      {
        text: "Once saturated, the benchmark can still reliably separate models of very different capability.",
        isCorrect: false,
      },
    ],
    explanation:
      "Saturation occurs when many models hit similar top scores, making the benchmark less informative for new progress.",
  },
  {
    id: "aie-ch3-q52",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about using perplexity as a proxy for downstream performance are reasonable?",
    options: [
      {
        text: "For base language models, lower perplexity on relevant text often correlates with better task performance.",
        isCorrect: true,
      },
      {
        text: "After heavy post-training (e.g. instruction tuning), perplexity may no longer correlate cleanly with task quality.",
        isCorrect: true,
      },
      {
        text: "Perplexity directly measures how aligned a model is with user preferences.",
        isCorrect: false,
      },
      {
        text: "Two models with identical perplexity must behave identically on all tasks.",
        isCorrect: false,
      },
    ],
    explanation:
      "Perplexity is useful but imperfect; post-training and other factors can decouple next-token prediction from user-facing quality.",
  },
  {
    id: "aie-ch3-q53",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about the limitations of lexical similarity metrics are correct?",
    options: [
      {
        text: "They may give high scores to outputs that copy the references but miss task requirements.",
        isCorrect: true,
      },
      {
        text: "They may give low scores to outputs that fulfill the task using different phrasing.",
        isCorrect: true,
      },
      {
        text: "They inherently reason about factual correctness and safety.",
        isCorrect: false,
      },
      {
        text: "They are robust to any errors in the reference data.",
        isCorrect: false,
      },
    ],
    explanation:
      "Lexical metrics are blind to factuality and depend heavily on reference coverage and quality.",
  },
  {
    id: "aie-ch3-q54",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about embedding-based semantic similarity are correct?",
    options: [
      {
        text: "Different embedding models may disagree about which sentence pairs are most similar.",
        isCorrect: true,
      },
      {
        text: "Semantic similarity scores can be sensitive to domain mismatch between embedding training data and evaluation texts.",
        isCorrect: true,
      },
      {
        text: "Once trained, embeddings encode semantics without any bias from the training corpus.",
        isCorrect: false,
      },
      {
        text: "Cosine similarity always perfectly matches human judgments of meaning.",
        isCorrect: false,
      },
    ],
    explanation:
      "Embeddings inherit biases and limitations from their training data and architecture; similarity is approximate, not perfect.",
  },
  {
    id: "aie-ch3-q55",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements correctly contrast human vs AI judges?",
    options: [
      {
        text: "Human evaluation is slower and more expensive per sample than AI-based evaluation.",
        isCorrect: true,
      },
      {
        text: "AI judges can process large volumes cheaply but may systematically miss certain failure types.",
        isCorrect: true,
      },
      {
        text: "AI judges never show bias, whereas humans always do.",
        isCorrect: false,
      },
      {
        text: "Human judgments are perfectly consistent across time and annotators.",
        isCorrect: false,
      },
    ],
    explanation:
      "Both humans and AI have biases and inconsistencies; their main differences are in speed, cost, and kinds of errors.",
  },
  {
    id: "aie-ch3-q56",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about using AI judges for toxicity or safety scoring are correct?",
    options: [
      {
        text: "They can flag obviously harmful content according to a safety rubric provided in the prompt.",
        isCorrect: true,
      },
      {
        text: "Their decisions reflect the values encoded in training data and prompts, which may not match all stakeholders.",
        isCorrect: true,
      },
      {
        text: "They give guarantees that content is legally compliant in every jurisdiction.",
        isCorrect: false,
      },
      {
        text: "They eliminate the need for any safety review process.",
        isCorrect: false,
      },
    ],
    explanation:
      "AI-based safety scoring provides scalable heuristics but not legal or ethical guarantees; human governance remains necessary.",
  },
  {
    id: "aie-ch3-q57",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements about AI judge self-bias are correct?",
    options: [
      {
        text: "A judge tends to prefer answers that look stylistically similar to its own outputs.",
        isCorrect: true,
      },
      {
        text: "This can inflate the apparent performance of the model family used as the judge when comparing different systems.",
        isCorrect: true,
      },
      {
        text: "The effect disappears entirely once you add more few-shot examples to the judge prompt.",
        isCorrect: false,
      },
      {
        text: "It is impossible to mitigate by altering prompts or sampling setups.",
        isCorrect: false,
      },
    ],
    explanation:
      "Self-bias is real but can be reduced (not eliminated) with careful experimental design and alternative judge models.",
  },
  {
    id: "aie-ch3-q58",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about pass@k metrics are correct?",
    options: [
      {
        text: "For the same model and benchmark, pass@10 should be at least as high as pass@1.",
        isCorrect: true,
      },
      {
        text: "Increasing k effectively trades extra sampling cost for higher chance of solving each problem.",
        isCorrect: true,
      },
      {
        text: "pass@k directly measures how often the model’s first attempt is correct.",
        isCorrect: false,
      },
      {
        text: "pass@k is unaffected by the number of test cases per problem.",
        isCorrect: false,
      },
    ],
    explanation:
      "pass@k is about any of k samples passing all tests; more tests per problem make it harder to pass.",
  },
  {
    id: "aie-ch3-q59",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements about the role of evaluation in the overall AI engineering pipeline are correct?",
    options: [
      {
        text: "Evaluation guides which model or prompt changes to ship.",
        isCorrect: true,
      },
      {
        text: "Evaluation helps decide whether an application is ready to move from prototype to production.",
        isCorrect: true,
      },
      {
        text: "Evaluation is mostly optional once you have access to a strong foundation model.",
        isCorrect: false,
      },
      {
        text: "Evaluation becomes unnecessary after the first successful deployment.",
        isCorrect: false,
      },
    ],
    explanation:
      "Evaluation is continuous; it steers model choice, iteration, and production monitoring throughout the product lifecycle.",
  },
  {
    id: "aie-ch3-q60",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about using perplexity for data deduplication or dataset curation are correct?",
    options: [
      {
        text: "High perplexity on candidate text can signal that it adds new information not already well covered in training data.",
        isCorrect: true,
      },
      {
        text: "Very low perplexity may indicate that similar content is already present, suggesting diminishing returns from adding it again.",
        isCorrect: true,
      },
      {
        text: "Perplexity alone can precisely estimate the marginal value of every training example.",
        isCorrect: false,
      },
      {
        text: "Perplexity cannot be used at all for any data selection task.",
        isCorrect: false,
      },
    ],
    explanation:
      "Perplexity is a noisy but useful heuristic for novelty and redundancy, not an exact measure of data value.",
  },
  {
    id: "aie-ch3-q61",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about edit-distance-based evaluation for short answers are correct?",
    options: [
      {
        text: "It can be more forgiving than exact string match while still penalizing large deviations.",
        isCorrect: true,
      },
      {
        text: "It is useful when small typos should not cause a full penalty.",
        isCorrect: true,
      },
      {
        text: "It inherently knows whether two answers refer to the same real-world entity.",
        isCorrect: false,
      },
      {
        text: "It can distinguish subtle semantic differences between long paragraphs.",
        isCorrect: false,
      },
    ],
    explanation:
      "Edit distance is helpful for approximate correctness in short strings but remains a surface-level metric.",
  },
  {
    id: "aie-ch3-q62",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about using AI judges for ranking candidate prompts or system variants are correct?",
    options: [
      {
        text: "You can run the same prompts through multiple system variants and let a judge choose which answer is better.",
        isCorrect: true,
      },
      {
        text: "Aggregating many pairwise judgments can approximate a preference ranking over systems.",
        isCorrect: true,
      },
      {
        text: "The resulting ranking is independent of the rubric specified in the judge prompt.",
        isCorrect: false,
      },
      {
        text: "Ranking is impossible without ground-truth references.",
        isCorrect: false,
      },
    ],
    explanation:
      "AI judges support rubric-dependent comparative evaluation even in the absence of labeled references.",
  },
  {
    id: "aie-ch3-q63",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about reference data quality problems are correct?",
    options: [
      {
        text: "Incorrect reference answers can cause good model outputs to be penalized.",
        isCorrect: true,
      },
      {
        text: "Missing valid references can make metrics underestimate a model’s capabilities.",
        isCorrect: true,
      },
      {
        text: "Reference errors are harmless as long as lexical similarity metrics are used.",
        isCorrect: false,
      },
      {
        text: "Reference errors are automatically corrected by AI judges.",
        isCorrect: false,
      },
    ],
    explanation:
      "References are not guaranteed to be perfect; both errors and omissions affect metric reliability.",
  },
  {
    id: "aie-ch3-q64",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements about evaluation for open-ended summarization tasks are correct?",
    options: [
      {
        text: "Multiple different summaries can all be acceptable for the same source text.",
        isCorrect: true,
      },
      {
        text: "Checking purely for lexical overlap with a single reference summary can miss high-quality summaries.",
        isCorrect: true,
      },
      {
        text: "Exact match is usually sufficient because there is only one correct summary.",
        isCorrect: false,
      },
      {
        text: "Evaluation is trivial because summaries are short.",
        isCorrect: false,
      },
    ],
    explanation:
      "Summarization evaluation is difficult precisely because many valid compressions and phrasings exist.",
  },
  {
    id: "aie-ch3-q65",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about using multiple evaluation metrics together are correct?",
    options: [
      {
        text: "You may care about a vector of metrics (e.g. quality, safety, latency) rather than a single scalar score.",
        isCorrect: true,
      },
      {
        text: "Improving one metric can sometimes worsen another, requiring trade-off decisions.",
        isCorrect: true,
      },
      {
        text: "If two metrics disagree, it always means one of them is useless.",
        isCorrect: false,
      },
      {
        text: "Combining many metrics guarantees that you will never miss a failure mode.",
        isCorrect: false,
      },
    ],
    explanation:
      "Real systems juggle multiple objectives; conflicting signals are normal and do not mean a metric is worthless.",
  },
  {
    id: "aie-ch3-q66",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about latency considerations for evaluation are correct?",
    options: [
      {
        text: "Offline batch evaluation can use slower, more expensive judges without affecting user-facing latency.",
        isCorrect: true,
      },
      {
        text: "Online evaluation on the critical path must respect strict latency budgets.",
        isCorrect: true,
      },
      {
        text: "Latency constraints are irrelevant for guardrail-style checks.",
        isCorrect: false,
      },
      {
        text: "Using a fast judge model always yields the same scores as a slower model.",
        isCorrect: false,
      },
    ],
    explanation:
      "Where evaluation runs in the pipeline determines how much latency overhead is acceptable; judge choice directly affects it.",
  },
  {
    id: "aie-ch3-q67",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about annotation guidelines for human evaluators are correct?",
    options: [
      {
        text: "Clear rubrics with examples reduce variance across annotators.",
        isCorrect: true,
      },
      {
        text: "Ambiguous or underspecified criteria lead to noisy and inconsistent scores.",
        isCorrect: true,
      },
      {
        text: "Guidelines are unnecessary when the task seems intuitive.",
        isCorrect: false,
      },
      {
        text: "Annotators will automatically infer the same standards without written guidance.",
        isCorrect: false,
      },
    ],
    explanation:
      "Even ‘obvious’ tasks benefit from clear written criteria and examples to keep human judgments aligned.",
  },
  {
    id: "aie-ch3-q68",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about using sampling (spot-checks) in evaluation are correct?",
    options: [
      {
        text: "Evaluating a random subset of responses can greatly reduce cost.",
        isCorrect: true,
      },
      {
        text: "Larger samples give tighter confidence intervals on estimated metrics.",
        isCorrect: true,
      },
      {
        text: "Sampling makes it impossible to say anything about system quality.",
        isCorrect: false,
      },
      {
        text: "Sampling guarantees that rare failures will be caught immediately.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sampling trades coverage for cost; statistical reasoning can still provide useful estimates, but rare events may be missed.",
  },
  {
    id: "aie-ch3-q69",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements about why black-box models complicate evaluation are correct?",
    options: [
      {
        text: "You cannot inspect training data or architecture to anticipate strengths and weaknesses.",
        isCorrect: true,
      },
      {
        text: "You must rely mostly on observed behavior across tasks and prompts.",
        isCorrect: true,
      },
      {
        text: "You can always compute gradients and internal activations to diagnose errors.",
        isCorrect: false,
      },
      {
        text: "Access to internals usually makes it harder to design evaluation strategies.",
        isCorrect: false,
      },
    ],
    explanation:
      "Lack of transparency pushes evaluation toward black-box testing over many scenarios instead of theory-driven diagnostics.",
  },
  {
    id: "aie-ch3-q70",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about the relationship between evaluation and system design are correct?",
    options: [
      {
        text: "Sometimes you must redesign your system to expose signals that are easier to evaluate.",
        isCorrect: true,
      },
      {
        text: "Improved observability (logs, traces, intermediate states) can make evaluation more targeted and informative.",
        isCorrect: true,
      },
      {
        text: "Evaluation can always be added afterward without changing the system.",
        isCorrect: false,
      },
      {
        text: "A system that hides its internal decisions is always easier to evaluate.",
        isCorrect: false,
      },
    ],
    explanation:
      "Evaluation and system architecture are intertwined; surfaces for logging and testing often need to be designed intentionally.",
  },
  {
    id: "aie-ch3-q71",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about semantic textual similarity tasks are correct?",
    options: [
      {
        text: "They evaluate how well similarity scores align with human judgments of sentence relatedness.",
        isCorrect: true,
      },
      {
        text: "They are one way to benchmark text embedding models.",
        isCorrect: true,
      },
      {
        text: "They require models to output human-written summaries.",
        isCorrect: false,
      },
      {
        text: "They are identical to translation tasks.",
        isCorrect: false,
      },
    ],
    explanation:
      "Semantic textual similarity tasks focus on grading similarity between sentence pairs, not on generating new text.",
  },
  {
    id: "aie-ch3-q72",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about evaluating retrieval-augmented generation (RAG) systems are correct?",
    options: [
      {
        text: "You may need to evaluate both retrieval quality and answer quality.",
        isCorrect: true,
      },
      {
        text: "Faithfulness metrics check whether answers are supported by retrieved documents.",
        isCorrect: true,
      },
      {
        text: "RAG systems can be fully evaluated by perplexity of the underlying base model alone.",
        isCorrect: false,
      },
      {
        text: "Retrieval metrics never require ground-truth relevant documents.",
        isCorrect: false,
      },
    ],
    explanation:
      "RAG introduces additional components; evaluating them requires metrics for retrieval relevance and grounding, not just base-model perplexity.",
  },
  {
    id: "aie-ch3-q73",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about subjective evaluation criteria like ‘helpfulness’ or ‘coherence’ are correct?",
    options: [
      {
        text: "Their meaning must be operationalized in a rubric before humans or AI can score them consistently.",
        isCorrect: true,
      },
      {
        text: "Different organizations may define them differently depending on goals and values.",
        isCorrect: true,
      },
      {
        text: "They have universally accepted mathematical definitions.",
        isCorrect: false,
      },
      {
        text: "Once defined, they never need to be revisited.",
        isCorrect: false,
      },
    ],
    explanation:
      "Subjective criteria require careful, often evolving definitions; there is no single universally correct rubric.",
  },
  {
    id: "aie-ch3-q74",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about using evaluation results to prioritize engineering work are correct?",
    options: [
      {
        text: "You should focus first on failure modes that are both common and harmful.",
        isCorrect: true,
      },
      {
        text: "Benchmark performance that does not affect real users should receive lower priority.",
        isCorrect: true,
      },
      {
        text: "Any metric that improves should automatically receive maximum engineering attention.",
        isCorrect: false,
      },
      {
        text: "Deciding priorities is unrelated to evaluation results.",
        isCorrect: false,
      },
    ],
    explanation:
      "Evaluation highlights where the product is weakest in terms that matter for users; priorities follow impact, not just metric changes.",
  },
  {
    id: "aie-ch3-q75",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about the limitations of public leaderboards are correct?",
    options: [
      {
        text: "They often reflect performance on narrow benchmarks rather than your specific use case.",
        isCorrect: true,
      },
      {
        text: "Over-optimization to leaderboard metrics can harm generalization to real-world tasks.",
        isCorrect: true,
      },
      {
        text: "They remove the need for any in-house evaluation.",
        isCorrect: false,
      },
      {
        text: "They always use perfectly curated, unbiased test sets.",
        isCorrect: false,
      },
    ],
    explanation:
      "Leaderboards are useful signals but cannot replace application-specific evaluation or guarantee robustness.",
  },

  // ----------------------------------------------------------------------------
  // Q76–Q100: exactly one correct option (1 correct)
  // ----------------------------------------------------------------------------

  {
    id: "aie-ch3-q76",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statement best describes perplexity in intuitive terms?",
    options: [
      {
        text: "It is the model’s effective average number of choices for the next token.",
        isCorrect: true,
      },
      {
        text: "It is the fraction of test examples the model classifies correctly.",
        isCorrect: false,
      },
      {
        text: "It is the total number of parameters in the model.",
        isCorrect: false,
      },
      {
        text: "It is the maximum length of sequence the model can process.",
        isCorrect: false,
      },
    ],
    explanation:
      "Perplexity can be read as how many equally likely options the model ‘feels’ it has for each step on average.",
  },
  {
    id: "aie-ch3-q77",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "For a language model evaluated on the same dataset, which single action most reliably *reduces* perplexity?",
    options: [
      {
        text: "Training it longer (with appropriate learning schedule) on relevant data.",
        isCorrect: true,
      },
      {
        text: "Increasing sampling temperature during evaluation.",
        isCorrect: false,
      },
      {
        text: "Randomly permuting the order of tokens in the dataset.",
        isCorrect: false,
      },
      {
        text: "Reducing the context window seen by the model.",
        isCorrect: false,
      },
    ],
    explanation:
      "Better training on relevant data improves predictive accuracy, whereas higher temperature, shuffled text, or less context tend to make prediction harder.",
  },
  {
    id: "aie-ch3-q78",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement best captures why post-training like RLHF can *increase* perplexity?",
    options: [
      {
        text: "It optimizes for human-preferred responses rather than raw next-token likelihood.",
        isCorrect: true,
      },
      {
        text: "It always shrinks the vocabulary, making prediction harder.",
        isCorrect: false,
      },
      {
        text: "It forces the model to output random noise on training data.",
        isCorrect: false,
      },
      {
        text: "It discards all information learned during pre-training.",
        isCorrect: false,
      },
    ],
    explanation:
      "Post-training steers the model toward desired behavior, which may diverge from maximizing likelihood under the original corpus.",
  },
  {
    id: "aie-ch3-q79",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statement describes the main advantage of functional correctness over lexical metrics for code generation?",
    options: [
      {
        text: "It evaluates whether the program actually produces correct outputs on tests.",
        isCorrect: true,
      },
      {
        text: "It only checks whether the generated code uses the same variable names as references.",
        isCorrect: false,
      },
      {
        text: "It primarily scores stylistic similarity to human-written code.",
        isCorrect: false,
      },
      {
        text: "It avoids the need to run any code at all.",
        isCorrect: false,
      },
    ],
    explanation:
      "Execution-based tests focus on behavior, which is what matters for code; stylistic similarity alone is insufficient.",
  },
  {
    id: "aie-ch3-q80",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statement best defines an embedding in this context?",
    options: [
      {
        text: "A vector representation that captures important features of the original object.",
        isCorrect: true,
      },
      {
        text: "A scalar that counts how many tokens a sentence has.",
        isCorrect: false,
      },
      {
        text: "A loss function used to train language models.",
        isCorrect: false,
      },
      {
        text: "A special type of benchmark dataset.",
        isCorrect: false,
      },
    ],
    explanation:
      "Embeddings are vector encodings used to capture semantics or structure for downstream similarity computations.",
  },
  {
    id: "aie-ch3-q81",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement best explains why AI-as-judge is controversial despite strong empirical correlations with humans?",
    options: [
      {
        text: "Because using the same technology to both generate and evaluate raises concerns about hidden biases and circular reasoning.",
        isCorrect: true,
      },
      {
        text: "Because AI models are incapable of following any instructions about evaluation.",
        isCorrect: false,
      },
      {
        text: "Because AI-judged scores cannot be computed at scale.",
        isCorrect: false,
      },
      {
        text: "Because AI judges never agree with human annotators.",
        isCorrect: false,
      },
    ],
    explanation:
      "People worry that AI judges may inherit opaque biases and that systems might optimize to please the judge rather than users.",
  },
  {
    id: "aie-ch3-q82",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which single change is most likely to *improve* the consistency of an AI judge’s scores across repeated runs?",
    options: [
      {
        text: "Lowering the judge model’s temperature toward zero.",
        isCorrect: true,
      },
      {
        text: "Using a very small context window for the judge.",
        isCorrect: false,
      },
      {
        text: "Randomly changing the rubric text before each call.",
        isCorrect: false,
      },
      {
        text: "Sampling outputs from multiple judge models and picking one at random.",
        isCorrect: false,
      },
    ],
    explanation:
      "Lower temperature makes the judge more deterministic, reducing variance in scores for identical inputs.",
  },
  {
    id: "aie-ch3-q83",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "When building a new evaluation for an AI assistant, which kind of metric is the most appropriate *first* one to define?",
    options: [
      {
        text: "A metric that reflects whether the assistant achieves its core user-facing task (functional success).",
        isCorrect: true,
      },
      {
        text: "A metric that measures how many tokens the assistant uses.",
        isCorrect: false,
      },
      {
        text: "A metric that counts how many new features engineers shipped last quarter.",
        isCorrect: false,
      },
      {
        text: "A metric that only considers latency regardless of correctness.",
        isCorrect: false,
      },
    ],
    explanation:
      "Task success is typically the primary goal; other metrics like latency and cost are important but secondary.",
  },
  {
    id: "aie-ch3-q84",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statement best captures a core limitation of exact-match evaluation for open-ended tasks?",
    options: [
      {
        text: "It cannot recognize alternative correct answers that differ from the reference text.",
        isCorrect: true,
      },
      {
        text: "It is computationally impossible to compute for short answers.",
        isCorrect: false,
      },
      {
        text: "It always requires embeddings to compute.",
        isCorrect: false,
      },
      {
        text: "It is only defined for image data.",
        isCorrect: false,
      },
    ],
    explanation:
      "Exact matching is too rigid when many different phrasings or outputs would satisfy the task.",
  },
  {
    id: "aie-ch3-q85",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "When two evaluation metrics disagree about which model is better, which statement is the most sensible reaction?",
    options: [
      {
        text: "Investigate why they differ and relate each metric back to real user needs before deciding.",
        isCorrect: true,
      },
      {
        text: "Immediately discard both metrics as useless.",
        isCorrect: false,
      },
      {
        text: "Assume the metric with the larger numeric scale is correct.",
        isCorrect: false,
      },
      {
        text: "Pick the metric that favors the model you already like.",
        isCorrect: false,
      },
    ],
    explanation:
      "Conflicting signals invite deeper analysis; understanding what each metric captures is crucial for informed decisions.",
  },
  {
    id: "aie-ch3-q86",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which single statement best explains why systematic evaluation beats ad hoc eyeballing as a system grows?",
    options: [
      {
        text: "It provides repeatable, quantitative signals that can track progress and regressions over time.",
        isCorrect: true,
      },
      {
        text: "It ensures that every single user query is manually inspected.",
        isCorrect: false,
      },
      {
        text: "It guarantees that no failures will ever occur in production.",
        isCorrect: false,
      },
      {
        text: "It removes the need for any product judgment or prioritization.",
        isCorrect: false,
      },
    ],
    explanation:
      "Systematic evaluation supports stable iteration and monitoring; it does not completely eliminate risk.",
  },
  {
    id: "aie-ch3-q87",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which single statement best characterizes the role of evaluation in discovering new capabilities of general-purpose models?",
    options: [
      {
        text: "Evaluation must also explore what new tasks the model might perform well at, not just measure known tasks.",
        isCorrect: true,
      },
      {
        text: "Evaluation is only about measuring performance on pre-defined benchmarks.",
        isCorrect: false,
      },
      {
        text: "Evaluation should avoid testing any tasks beyond human ability.",
        isCorrect: false,
      },
      {
        text: "Evaluation ignores tasks that were not in the training distribution.",
        isCorrect: false,
      },
    ],
    explanation:
      "For general models, evaluation includes exploration: discovering surprising capabilities and limits across new task types.",
  },
  {
    id: "aie-ch3-q88",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which single statement best describes why evaluation infrastructure has historically lagged behind modeling infrastructure?",
    options: [
      {
        text: "Research attention and tooling have focused more on new algorithms and models than on standardized evaluation methods.",
        isCorrect: true,
      },
      {
        text: "Evaluation requires fundamentally unsolved hardware breakthroughs.",
        isCorrect: false,
      },
      {
        text: "Evaluation is mathematically impossible for neural networks.",
        isCorrect: false,
      },
      {
        text: "Evaluation is unnecessary because models self-verify their own outputs.",
        isCorrect: false,
      },
    ],
    explanation:
      "Historically, algorithm development outpaced investment in robust evaluation frameworks, leaving a tooling gap.",
  },
  {
    id: "aie-ch3-q89",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statement best summarizes why we sometimes want to evaluate semantic similarity instead of exact equality?",
    options: [
      {
        text: "Because many tasks only require that the meaning matches, not that the wording is identical.",
        isCorrect: true,
      },
      {
        text: "Because humans cannot read exact strings.",
        isCorrect: false,
      },
      {
        text: "Because exact equality cannot be computed by a computer.",
        isCorrect: false,
      },
      {
        text: "Because semantic similarity is always easier to compute than string comparison.",
        isCorrect: false,
      },
    ],
    explanation:
      "For paraphrases and free-form answers, meaning is what matters; surface forms can legitimately vary.",
  },
  {
    id: "aie-ch3-q90",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "You design a new AI judge for ‘faithfulness’ to provided context. Which single step is most important for making its scores interpretable to others?",
    options: [
      {
        text: "Publishing the exact prompt, scoring rubric, and model version you used.",
        isCorrect: true,
      },
      {
        text: "Using an obscure internal codename for the judge.",
        isCorrect: false,
      },
      {
        text: "Rounding all scores to exactly two decimal places.",
        isCorrect: false,
      },
      {
        text: "Ensuring that the judge always returns a score of 0.5.",
        isCorrect: false,
      },
    ],
    explanation:
      "Transparency about configuration lets others understand what the judge is doing and replicate or critique it.",
  },
  {
    id: "aie-ch3-q91",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which single statement best describes the main drawback of using a very strong proprietary model as an AI judge?",
    options: [
      {
        text: "It can make evaluation expensive and may raise privacy or IP concerns with sensitive data.",
        isCorrect: true,
      },
      {
        text: "It cannot follow any instructions about evaluation criteria.",
        isCorrect: false,
      },
      {
        text: "Its judgments are guaranteed to be worse than a weak open-source model.",
        isCorrect: false,
      },
      {
        text: "It is incapable of producing explanations for its scores.",
        isCorrect: false,
      },
    ],
    explanation:
      "Stronger proprietary judges can work well but may be costly and require sending sensitive data to third parties.",
  },
  {
    id: "aie-ch3-q92",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which single statement best captures why we often need *both* automatic and human evaluation?",
    options: [
      {
        text: "Automatic metrics scale cheaply, while humans provide depth and nuance on critical cases.",
        isCorrect: true,
      },
      {
        text: "Humans cannot judge language quality at all.",
        isCorrect: false,
      },
      {
        text: "Automatic metrics are illegal in most countries.",
        isCorrect: false,
      },
      {
        text: "Human evaluation always takes less time than automatic evaluation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Each approach compensates for the other’s weaknesses; together they give a more reliable picture of system behavior.",
  },
  {
    id: "aie-ch3-q93",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which single statement best explains why adversarial testing complements standard benchmark evaluation?",
    options: [
      {
        text: "It deliberately searches for challenging edge cases that benchmarks may not cover.",
        isCorrect: true,
      },
      {
        text: "It only measures average-case performance on random data.",
        isCorrect: false,
      },
      {
        text: "It eliminates the need for any automated metrics.",
        isCorrect: false,
      },
      {
        text: "It focuses exclusively on synthetic, unrealistic prompts.",
        isCorrect: false,
      },
    ],
    explanation:
      "Adversarial probing targets weaknesses that average-case benchmarks miss, especially for safety and robustness.",
  },
  {
    id: "aie-ch3-q94",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which single statement best characterizes the purpose of evaluation dashboards in an AI product team?",
    options: [
      {
        text: "To summarize key metrics so that regressions or improvements are quickly visible.",
        isCorrect: true,
      },
      {
        text: "To replace the need for any detailed analysis or investigation.",
        isCorrect: false,
      },
      {
        text: "To hide evaluation details from engineers.",
        isCorrect: false,
      },
      {
        text: "To guarantee that all failures will be automatically fixed.",
        isCorrect: false,
      },
    ],
    explanation:
      "Dashboards are a monitoring surface; they surface anomalies but do not explain or fix them by themselves.",
  },
  {
    id: "aie-ch3-q95",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statement best explains why consistent evaluation criteria are important when multiple people rate answers?",
    options: [
      {
        text: "Without shared criteria, different raters may score the same answer very differently.",
        isCorrect: true,
      },
      {
        text: "Raters naturally agree on all judgments even without guidance.",
        isCorrect: false,
      },
      {
        text: "Shared criteria are only needed for automatic metrics, not humans.",
        isCorrect: false,
      },
      {
        text: "Inconsistent criteria make it easier to compare models.",
        isCorrect: false,
      },
    ],
    explanation:
      "Common rubrics help keep human scores comparable across annotators and time.",
  },
  {
    id: "aie-ch3-q96",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which single statement best explains why we might *freeze* an evaluation suite for a period of time?",
    options: [
      {
        text: "To ensure that metric changes reflect model changes rather than evolving tests.",
        isCorrect: true,
      },
      {
        text: "To guarantee that the suite will remain useful forever without updates.",
        isCorrect: false,
      },
      {
        text: "To prevent anyone from understanding how evaluation works.",
        isCorrect: false,
      },
      {
        text: "To avoid adding any new test cases even when new failure modes appear.",
        isCorrect: false,
      },
    ],
    explanation:
      "Temporarily freezing the suite provides a stable baseline; it does not mean the suite should never be revised.",
  },
  {
    id: "aie-ch3-q97",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statement best characterizes a key challenge in evaluating models that exceed average human ability on some tasks?",
    options: [
      {
        text: "Few people are qualified to reliably check correctness on very advanced problems.",
        isCorrect: true,
      },
      {
        text: "Models instantly become impossible to run on any hardware.",
        isCorrect: false,
      },
      {
        text: "Benchmarks can no longer store correct answers.",
        isCorrect: false,
      },
      {
        text: "High performance guarantees perfect behavior in all other domains.",
        isCorrect: false,
      },
    ],
    explanation:
      "As model competence rises, human evaluators may struggle to verify complex outputs, especially in technical domains.",
  },
  {
    id: "aie-ch3-q98",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which single statement best explains why evaluation design should focus on likely failure points in a system?",
    options: [
      {
        text: "Because resources are limited, so we get the most value by probing where things are most likely to go wrong or cause harm.",
        isCorrect: true,
      },
      {
        text: "Because random testing always covers all edge cases.",
        isCorrect: false,
      },
      {
        text: "Because we should avoid testing any common user flows.",
        isCorrect: false,
      },
      {
        text: "Because systems never fail in rare or unexpected ways.",
        isCorrect: false,
      },
    ],
    explanation:
      "Targeting high-risk, high-frequency parts of the system makes evaluation more impactful.",
  },
  {
    id: "aie-ch3-q99",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which single statement best describes why it is dangerous to optimize exclusively for a single automatic metric?",
    options: [
      {
        text: "The system may ‘game’ that metric while degrading along important unmeasured dimensions.",
        isCorrect: true,
      },
      {
        text: "Multiple metrics are mathematically impossible to compute.",
        isCorrect: false,
      },
      {
        text: "Optimizing for one metric always optimizes all others simultaneously.",
        isCorrect: false,
      },
      {
        text: "Single-metric optimization guarantees perfect robustness.",
        isCorrect: false,
      },
    ],
    explanation:
      "Goodhart’s law applies: when a measure becomes a target, it can cease to be a good measure of underlying quality.",
  },
  {
    id: "aie-ch3-q100",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Overall, which statement best summarizes the role of evaluation in AI engineering?",
    options: [
      {
        text: "It is the process of systematically measuring and monitoring model behavior so systems can be improved and deployed safely.",
        isCorrect: true,
      },
      {
        text: "It is a one-time checklist step before turning on an API key.",
        isCorrect: false,
      },
      {
        text: "It is mainly a marketing exercise to produce leaderboard numbers.",
        isCorrect: false,
      },
      {
        text: "It is only relevant for academic research, not for real applications.",
        isCorrect: false,
      },
    ],
    explanation:
      "Evaluation is an ongoing engineering discipline that underpins safety, reliability, and iteration for AI-powered products.",
  },
];
