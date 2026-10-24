import type { Question } from "../../quiz";
import { referenceQuestion } from "../../questionReferences";
import { stanfordCME295Lecture2Questions } from "../Stanford CME295 Transformers & LLMs/lecture2_models";
import { stanfordCME295Lecture3LLMsQuestions } from "../Stanford CME295 Transformers & LLMs/lecture3_LLMs";
import { stanfordCME295Lecture4TrainingQuestions } from "../Stanford CME295 Transformers & LLMs/lecture4_training";

export const stanfordCME2952026Lecture2Questions: Question[] = [
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect2-q01",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "An encoder-only BERT-style model is adapted to classify whole documents. Which operations fit this use?",
    options: [
      {
        text: "Provide a causal target decoder as a required part of the classification computation.",
        isCorrect: false,
      },
      {
        text: "Use an aggregate representation such as the [CLS] state for a classification head.",
        isCorrect: true,
      },
      {
        text: "Train the task head and, when appropriate, fine-tune the encoder.",
        isCorrect: true,
      },
      {
        text: "Build contextual token representations with bidirectional attention.",
        isCorrect: true,
      },
    ],
    explanation:
      "BERT, Bidirectional Encoder Representations from Transformers, supplies contextual features rather than a mandatory left-to-right decoder. A document classifier can project the [CLS] representation to task labels and fine-tune it; target-side autoregressive decoding belongs to a different model-family arrangement.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect2-q02",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "A BERT-style encoder predicts a masked word inside a sentence. Which information can it use?",
    options: [
      {
        text: "Visible tokens before the masked position.",
        isCorrect: true,
      },
      {
        text: "Learned representations of the visible context.",
        isCorrect: true,
      },
      {
        text: "The original hidden word supplied directly at the masked position.",
        isCorrect: false,
      },
      {
        text: "Visible tokens after the masked position.",
        isCorrect: true,
      },
    ],
    explanation:
      "Masked language modeling creates a prediction problem by hiding content while leaving context on both sides visible. Bidirectional attention can combine those visible cues, but directly supplying the hidden target would undermine the proxy task.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect2-q03",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Select all true statements comparing masked language modeling and left-to-right next-token prediction.",
    options: [
      {
        text: "A causal next-token model predicts a token from an earlier prefix.",
        isCorrect: true,
      },
      {
        text: "Both can obtain target labels from ordinary text.",
        isCorrect: true,
      },
      {
        text: "Masked language modeling can reconstruct selected hidden tokens using surrounding context.",
        isCorrect: true,
      },
      {
        text: "Masked language modeling must exclude visible tokens to the right of each masked position.",
        isCorrect: false,
      },
    ],
    explanation:
      "Both objectives are self-supervised because the original text supplies the desired token labels. They differ in available context and prediction positions: reconstruction can use right context, while causal prediction excludes the token and its future.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect2-q04",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "In original BERT pretraining, what is the Next Sentence Prediction task asking the classifier to decide?",
    options: [
      {
        text: "Whether two sentences express the same sentiment.",
        isCorrect: false,
      },
      {
        text: "Whether the second segment follows the first in the source text.",
        isCorrect: true,
      },
      {
        text: "Which of the two segments contains the masked word.",
        isCorrect: false,
      },
      {
        text: "Which word is most probable after the final token.",
        isCorrect: false,
      },
    ],
    explanation:
      "Next Sentence Prediction uses sentence-pair inputs and a binary label about their relationship in the corpus. It is distinct from reconstructing masked tokens, generating the next token, or labeling sentiment, even though those tasks can also use encoder representations.",
  },
  // Replaces retired ID cme295-2026-lect2-q05; new reasoning task.
  {
    id: "cme295-2026-lect2-q051",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Compare two translation models for a two-token source and a three-token target. Model E has a bidirectional encoder and causal decoder with cross-attention. Model D is a standard causally masked decoder-only model receiving source then target in one stream. Ignore padding. Select all true visibility comparisons.",
    options: [
      {
        text: "In D, the first target position reads source tokens only through a separate encoder cross-attention sublayer.",
        isCorrect: false,
      },
      {
        text: "In both models, a target position can read preceding target positions without reading later target positions.",
        isCorrect: true,
      },
      {
        text: "In D, the first source position cannot read the second source position through its causal self-attention.",
        isCorrect: true,
      },
      {
        text: "In E, the first source position can read the second source position in its encoder layer.",
        isCorrect: true,
      },
    ],
    explanation:
      "E builds bidirectional source representations, so its first source token can use the second before cross-attention supplies them to the decoder. D's single causal stream preserves left-to-right visibility even inside its source prefix. Both enforce causal target access, but D attends to earlier source tokens through ordinary self-attention rather than through a separate encoder stream.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect2-q06",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "A decoder-only language model generates a continuation of a prompt. Which statements describe the usual autoregressive setup?",
    options: [
      {
        text: "A later token distribution is conditioned on previously generated tokens.",
        isCorrect: true,
      },
      {
        text: "A vocabulary distribution is produced for the next token.",
        isCorrect: true,
      },
      {
        text: "The sequence probability factors into conditional next-token probabilities.",
        isCorrect: true,
      },
      {
        text: "The generated token is appended to the prefix.",
        isCorrect: true,
      },
    ],
    explanation:
      "A decoder-only language model repeatedly extends a causal token sequence. Multiplying its conditional next-token probabilities defines a probability for the completed sequence, and each newly generated token changes the prefix used for subsequent predictions.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect2-q07",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "A training run requires 600 trillion floating-point operations, and its sustained throughput is 20 trillion operations per second. Which statements are correct?",
    options: [
      {
        text: "The idealized arithmetic time is 30 seconds.",
        isCorrect: true,
      },
      {
        text: "The idealized arithmetic time is 12,000 seconds.",
        isCorrect: false,
      },
      {
        text: "Doubling throughput doubles the arithmetic time for the same work.",
        isCorrect: false,
      },
      {
        text: "The work is measured in FLOPs and the throughput in FLOP/s.",
        isCorrect: true,
      },
    ],
    explanation:
      "Time is work divided by rate: 600 trillion operations divided by 20 trillion operations per second gives 30 seconds. FLOPs and FLOP/s therefore represent different quantities, and a higher sustained rate reduces the time for fixed work.",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(
    stanfordCME295Lecture4TrainingQuestions,
    "cme295-lect4-q104",
  ),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(
    stanfordCME295Lecture4TrainingQuestions,
    "cme295-lect4-q105",
  ),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(
    stanfordCME295Lecture4TrainingQuestions,
    "cme295-lect4-q107",
  ),
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect2-q11",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Use the rough compute-optimal heuristics \\(C\\propto ND\\) and \\(D=20N\\). If the compute budget increases fourfold, which changes preserve this balance?",
    options: [
      {
        text: "Increase the product of parameters and tokens fourfold.",
        isCorrect: true,
      },
      {
        text: "Double the parameter count.",
        isCorrect: true,
      },
      {
        text: "Double the training-token count.",
        isCorrect: true,
      },
      {
        text: "Keep the token-to-parameter ratio at 20.",
        isCorrect: true,
      },
    ],
    explanation:
      "Combining the heuristics gives compute proportional to the square of model size, so a fourfold budget supports twice the parameters and twice the tokens. Scaling just parameters or just data by four would satisfy the product budget but break the stated balance; these are approximate training heuristics.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect2-q12",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Where is a sparse Mixture-of-Experts (MoE) layer commonly placed in a transformer, and how is it selected?",
    options: [
      {
        text: "A learned router scores experts from each token representation.",
        isCorrect: true,
      },
      {
        text: "The same expert must process all tokens in a sequence.",
        isCorrect: false,
      },
      {
        text: "Expert feedforward networks replace or augment the dense feedforward sublayer.",
        isCorrect: true,
      },
      {
        text: "The attention-head index directly specifies the expert index.",
        isCorrect: false,
      },
    ],
    explanation:
      "MoE routing typically targets the parameter-heavy feedforward computation and depends on the current token state. Different tokens can choose different experts, and attention heads and experts are separate architectural components rather than matching indexed copies.",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture3LLMsQuestions, "cme295-lect3-q188"),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture3LLMsQuestions, "cme295-lect3-q189"),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture3LLMsQuestions, "cme295-lect3-q190"),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture3LLMsQuestions, "cme295-lect3-q191"),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture3LLMsQuestions, "cme295-lect3-q194"),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture3LLMsQuestions, "cme295-lect3-q195"),
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect2-q19",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "A sparse MoE layer has one shared expert and 16 routed experts, of which the router activates two per token. Which statements follow?",
    options: [
      {
        text: "The stored pool contains 17 experts.",
        isCorrect: true,
      },
      {
        text: "The shared expert contributes regardless of which two routed experts are chosen.",
        isCorrect: true,
      },
      {
        text: "Three expert computations are active for a token in this layer.",
        isCorrect: true,
      },
      {
        text: "The shared expert can learn common computations that routed experts would otherwise duplicate.",
        isCorrect: true,
      },
    ],
    explanation:
      "The two selected routed experts operate alongside the shared expert, yielding three active expert paths from a pool of seventeen. Sharing common computations can free routed experts to specialize; the active count and total stored expert count describe different costs.",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture2Questions, "cme295-lect2-q77"),
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect2-q21",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "For a sinusoidal positional-encoding pair at frequency \\(\\omega_i\\), define \\(p_m^{(i)}=(\\sin(\\omega_i m), \\cos(\\omega_i m))\\). Which expression equals the dot product \\(p_m^{(i)} \\cdot p_n^{(i)}\\)?",
    options: [
      {
        text: "\\(\\cos(\\omega_i(m+n))\\)",
        isCorrect: false,
      },
      {
        text: "\\(\\cos(\\omega_i(m-n))\\)",
        isCorrect: true,
      },
      {
        text: "\\(\\sin(\\omega_i(m-n))\\)",
        isCorrect: false,
      },
      {
        text: "\\(\\omega_i mn\\)",
        isCorrect: false,
      },
    ],
    explanation:
      "The dot product is \\(\\sin(\\omega_i m)\\sin(\\omega_i n)+\\cos(\\omega_i m)\\cos(\\omega_i n)\\), which equals \\(\\cos(\\omega_i(m-n))\\) by the cosine difference identity. The other expressions either use sine instead of the cosine identity, depend on the sum rather than the offset, or replace the trigonometric similarity with an unrelated product.",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture2Questions, "cme295-lect2-q02"),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture2Questions, "cme295-lect2-q17"),
  // Replaces retired ID cme295-2026-lect2-q24; new reasoning task.
  {
    id: "cme295-2026-lect2-q052",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "A retrieval evaluation uses inputs that all fit the context limit. At 4,000 tokens, accuracy is 96% for a fact near the beginning and 95% for a fact in the middle. At 32,000 tokens, the corresponding accuracies are 89% and 62%. Which comparison is supported?",
    options: [
      {
        text: "At 32,000 tokens, middle placement is 27 percent worse in relative accuracy, compared with a one-percent gap at 4,000 tokens.",
        isCorrect: false,
      },
      {
        text: "At 32,000 tokens, middle placement is 27 percentage points worse than beginning placement, compared with a one-point gap at 4,000 tokens.",
        isCorrect: true,
      },
      {
        text: "The 33-point middle-position decline is entirely a placement effect measured at one fixed length.",
        isCorrect: false,
      },
      {
        text: "Increasing length lowers accuracy by seven percentage points at both tested positions.",
        isCorrect: false,
      },
    ],
    explanation:
      "The within-length placement gaps are 89 − 62 = 27 percentage points and 96 − 95 = 1 point. The relative long-context reduction is 27/89, about 30.3%, rather than 27%; length changes reduce beginning accuracy by seven points and middle accuracy by thirty-three. The middle-position decline compares two lengths, so it does not isolate placement at a fixed length. Fitting the advertised context window therefore does not imply uniform retrieval performance across its positions.",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture2Questions, "cme295-lect2-q05"),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture2Questions, "cme295-lect2-q06"),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture2Questions, "cme295-lect2-q07"),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture2Questions, "cme295-lect2-q13"),
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect2-q29",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Rotary Position Embeddings (RoPE) use orthogonal rotations \\(R_m\\) and \\(R_n\\) on a query \\(q\\) and key \\(k\\). Which statements explain the relative-position effect?",
    options: [
      {
        text: "For the same rotation frequencies, \\(R_m^\\top R_n=R_{m+n}\\).",
        isCorrect: false,
      },
      {
        text: "For the same rotation frequencies, \\(R_m^\\top R_n=R_{n-m}\\).",
        isCorrect: true,
      },
      {
        text: "The score can depend on the position difference through the combined rotation.",
        isCorrect: true,
      },
      {
        text: "\\((R_m q)^\\top(R_n k)=q^\\top R_m^\\top R_n k\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The transpose of a rotation reverses its angle, so R_mᵀ R_n leaves the difference n−m rather than the sum m+n. Expanding the rotated dot product gives qᵀR_mᵀR_n k, which introduces relative position into query-key compatibility. The resulting score depends on both this relative rotation and the content vectors.",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture2Questions, "cme295-lect2-q19"),
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect2-q31",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "A bidirectional Longformer-style layer has 8 positions. Position 1 is global and can attend to every position; each other position attends to itself, its immediate neighbors, and position 1. Which attention sets are correct?",
    options: [
      {
        text: "Position 1 attends to {1, 2, 3, 4, 5, 6, 7, 8}.",
        isCorrect: true,
      },
      {
        text: "Position 8 attends to {1, 7, 8}.",
        isCorrect: true,
      },
      {
        text: "Position 6 attends to {1, 5, 6, 7}.",
        isCorrect: true,
      },
      {
        text: "Position 2 attends to {1, 2, 3}.",
        isCorrect: true,
      },
    ],
    explanation:
      "For a nonglobal query, take the radius-one local window and add the global key, removing duplicates and respecting boundaries. The global query instead reads the entire sequence, so its row is dense while the other rows retain sparse local-plus-global structure.",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture2Questions, "cme295-lect2-q18"),
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect2-q33",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "A model has 16 query heads but one shared set of key and value projections. Which statements describe Multi-Query Attention (MQA)?",
    options: [
      {
        text: "MQA is the one-key/value-group endpoint of grouped sharing.",
        isCorrect: true,
      },
      {
        text: "The query projections can remain distinct.",
        isCorrect: true,
      },
      {
        text: "Each query head attends to the shared key/value representations.",
        isCorrect: true,
      },
      {
        text: "Sharing reduces the number of key/value representations that must be stored per token.",
        isCorrect: true,
      },
    ],
    explanation:
      "MQA shares keys and values across query heads while retaining multiple query projections. That changes the representation and storage requirements for K/V tensors, rather than merging every query into a single head; grouped-query attention uses more than one shared group.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect2-q34",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "A decoder uses 24 query heads and 6 key/value groups. Which statements describe Grouped-Query Attention (GQA) when groups are equal sized?",
    options: [
      {
        text: "Each group serves four query heads.",
        isCorrect: true,
      },
      {
        text: "Each query head needs its own distinct key/value head.",
        isCorrect: false,
      },
      {
        text: "The query-head count remains 24.",
        isCorrect: true,
      },
      {
        text: "There are six key projections and six value projections at the head level.",
        isCorrect: true,
      },
    ],
    explanation:
      "Dividing 24 query heads among six groups gives four queries per shared K/V set. GQA retains query diversity while reducing the number of K/V representations; distinct K/V heads for every query would instead describe full multi-head attention.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect2-q35",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Layer normalization over \\(x=(1,3)\\) uses \\(\\mu=\\frac{1}{d}\\sum_j x_j\\) and variance \\(\\frac{1}{d}\\sum_j(x_j-\\mu)^2\\). With \\(\\epsilon=0\\), which intermediate quantities are correct?",
    options: [
      {
        text: "The mean is \\(2\\).",
        isCorrect: true,
      },
      {
        text: "The centered vector is \\((1,-1)\\).",
        isCorrect: false,
      },
      {
        text: "With \\(\\gamma=(2,2)\\) and \\(\\beta=(0,0)\\), the final output is \\((-1,1)\\).",
        isCorrect: false,
      },
      {
        text: "The variance is \\(1\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The average of \\(1\\) and \\(3\\) is \\(2\\), and the squared deviations are \\(1\\) and \\(1\\), so the variance is \\(1\\). The centered vector is \\((-1,1)\\), and multiplying the normalized vector by \\(\\gamma=(2,2)\\) would give \\((-2,2)\\), not \\((-1,1)\\).",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture2Questions, "cme295-lect2-q09"),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture2Questions, "cme295-lect2-q20"),
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect2-q38",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Let \\(F\\) be an attention or feedforward sublayer and \\(N\\) a normalization. Which expressions correctly identify the usual residual-block conventions?",
    options: [
      {
        text: "Pre-norm: \\(x+F(N(x))\\).",
        isCorrect: true,
      },
      {
        text: "Post-norm: \\(N(x+F(x))\\).",
        isCorrect: true,
      },
      {
        text: "Post-norm: \\(x+N(F(x))\\).",
        isCorrect: false,
      },
      {
        text: "Pre-norm: \\(N(x)+F(x)\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Pre-norm normalizes the input to the transformed branch while leaving the residual identity path outside that normalization. Post-norm normalizes the sum after adding the transformed branch; normalizing just one summand gives a different computational graph.",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture3LLMsQuestions, "cme295-lect3-q197"),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture3LLMsQuestions, "cme295-lect3-q199"),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture3LLMsQuestions, "cme295-lect3-q202"),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture3LLMsQuestions, "cme295-lect3-q203"),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture3LLMsQuestions, "cme295-lect3-q204"),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture3LLMsQuestions, "cme295-lect3-q206"),
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect2-q45",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Two deployments use greedy decoding on the same input, but different batching changes floating-point reduction order. Which statements explain why their outputs can still diverge?",
    options: [
      {
        text: "Greedy token selection does not remove nondeterminism in the preceding numerical computation.",
        isCorrect: true,
      },
      {
        text: "Small numerical changes can switch the winning token when logits are close.",
        isCorrect: true,
      },
      {
        text: "Once a different token is chosen, subsequent prefixes differ.",
        isCorrect: true,
      },
      {
        text: "Greedy decoding explicitly samples a random token from the softmax distribution.",
        isCorrect: false,
      },
    ],
    explanation:
      "Greedy decoding chooses the maximum of the logits it receives, but hardware kernels and batching can slightly change those logits. A near tie can therefore resolve differently and alter the rest of the trajectory; random categorical sampling is a separate source of variation.",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture3LLMsQuestions, "cme295-lect3-q209"),
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect2-q47",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "A model has a 10,000-token combined context limit. Its prompt uses 7,500 tokens. Ignoring any additional reserved overhead, which statements are correct?",
    options: [
      {
        text: "Few-shot examples consume part of the prompt budget.",
        isCorrect: true,
      },
      {
        text: "Adding 500 prompt tokens leaves at most 2,000 generated tokens.",
        isCorrect: true,
      },
      {
        text: "The output receives a fresh 10,000-token budget after the 7,500-token prompt has been processed.",
        isCorrect: false,
      },
      {
        text: "At most 2,500 generated tokens fit with that prompt.",
        isCorrect: true,
      },
    ],
    explanation:
      "The stated limit covers the prompt and growing continuation together, leaving 10,000 − 7,500 = 2,500 tokens. Adding 500 tokens of instructions or examples leaves 2,000. Processing the prompt does not reset this shared capacity; few-shot demonstrations occupy the same budget as other prompt tokens.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect2-q48",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "A model follows a new label format after seeing three examples in its prompt, then loses that behavior when the examples are removed. Which explanation fits?",
    options: [
      {
        text: "The trained weights can remain unchanged across both requests.",
        isCorrect: true,
      },
      {
        text: "The first request necessarily fine-tuned the model and the second request rolled back that checkpoint.",
        isCorrect: false,
      },
      {
        text: "The KV cache must have stored the examples as permanent training data shared with later users.",
        isCorrect: false,
      },
      {
        text: "The behavior was induced by in-context learning while the examples were present.",
        isCorrect: true,
      },
    ],
    explanation:
      "In-context learning is temporary conditioning through the current token sequence, so removing the examples can remove the induced behavior without any parameter change. A request-local KV cache supports computation for that request and is not evidence of fine-tuning or cross-user training-data storage.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect2-q49",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "A prompt demonstrates an arithmetic solution with intermediate steps instead of giving the final answer alone. Which statements describe chain-of-thought prompting?",
    options: [
      {
        text: "It supplies an example of producing a reasoning path.",
        isCorrect: true,
      },
      {
        text: "This prompt itself updates the model weights through gradient descent.",
        isCorrect: false,
      },
      {
        text: "Additional generated tokens can increase latency and cost.",
        isCorrect: true,
      },
      {
        text: "Generated intermediate steps consume output tokens.",
        isCorrect: true,
      },
    ],
    explanation:
      "Chain-of-thought prompting elicits intermediate steps through the input context rather than a weight update. Generating those steps uses additional autoregressive computation and output budget, so the approach trades more generation work for a chance of better problem solving.",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(stanfordCME295Lecture3LLMsQuestions, "cme295-lect3-q218"),
];
