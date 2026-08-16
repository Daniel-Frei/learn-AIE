import { Question } from "../../quiz";

type Difficulty = "easy" | "medium" | "hard";
type OptionSeed = readonly [text: string, isCorrect: boolean];
type AssertionReasonChoice = 1 | 2 | 3 | 4 | 5;

const assertionReasonOptionTexts = [
  "Assertion is true, Reason is false.",
  "Assertion is false, Reason is true.",
  "Both are false.",
  "Both are true, and the Reason is the correct explanation of the Assertion.",
  "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
] as const;

function makeQuestion(
  id: string,
  difficulty: Difficulty,
  prompt: string,
  optionSeeds: readonly OptionSeed[],
  explanation: string,
): Question {
  return {
    id,
    chapter: 8,
    difficulty,
    prompt,
    options: optionSeeds.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

function makeAssertionReasonQuestion(
  id: string,
  difficulty: Difficulty,
  assertion: string,
  reason: string,
  correctChoice: AssertionReasonChoice,
  explanation: string,
): Question {
  return {
    id,
    chapter: 8,
    difficulty,
    type: "assertion-reason",
    prompt: "Assertion: " + assertion + "\n\nReason: " + reason,
    options: assertionReasonOptionTexts.map((text, index) => ({
      text,
      isCorrect: index + 1 === correctChoice,
    })),
    explanation,
  };
}

export const stanfordCME296Lecture8TextDiffusionQuestions: Question[] = [
  makeQuestion(
    "cme296-lect8-q01",
    "easy",
    "Why does ordinary autoregressive text generation require a number of decoding iterations that grows with output length?",
    [
      [
        "Each new token is conditioned on the prefix produced so far, so later tokens wait for earlier ones.",
        true,
      ],
      [
        "The model must update its parameters after every emitted token, and that online retraining rather than prefix dependence forces one round per position.",
        false,
      ],
      [
        "Every token is generated independently but the tokenizer serializes the results afterward.",
        false,
      ],
      [
        "The vocabulary size determines one decoding iteration per vocabulary entry.",
        false,
      ],
    ],
    "An autoregressive model factorizes a sequence into next-token conditionals, making token positions sequentially dependent during decoding even though much of each model evaluation is parallel. Cached activations may update, but learned parameters remain fixed rather than being retrained online; token predictions are not independent, and vocabulary size does not set the number of rounds.",
  ),
  makeQuestion(
    "cme296-lect8-q02",
    "easy",
    "Which properties make a dedicated mask token a useful analogue of corruption for discrete text?",
    [
      [
        "It explicitly denotes an unknown token position without inserting a competing lexical meaning.",
        true,
      ],
      [
        "It supports corruption levels defined by masking different fractions of positions.",
        true,
      ],
      [
        "It turns discrete token IDs into continuous Gaussian pixel values.",
        false,
      ],
      [
        "It guarantees that every masked position has only one grammatically valid reconstruction.",
        false,
      ],
    ],
    "A mask token marks missing information in the discrete sequence and lets the corruption level control how many positions are hidden. It does not make tokens continuous, and context can admit several plausible completions; the denoiser therefore predicts distributions over vocabulary items rather than recovering a mathematically unique original token.",
  ),
  makeQuestion(
    "cme296-lect8-q03",
    "medium",
    "Why can replacing tokens with uniformly sampled ordinary vocabulary items be a problematic corruption scheme for a text denoiser?",
    [
      [
        "A replacement token carries semantics and may create a plausible but misleading alternative sentence.",
        true,
      ],
      [
        "The model may not be able to distinguish observed content from content introduced as corruption.",
        true,
      ],
      [
        "The resulting sequence is still discrete, so Gaussian image-noise intuitions do not transfer directly.",
        true,
      ],
      [
        "Uniform replacement prevents the model from assigning probabilities to vocabulary items.",
        false,
      ],
    ],
    "Ordinary tokens are meaningful symbols, so a random replacement can look like genuine evidence and obscure which positions were corrupted; a dedicated mask instead exposes missingness. Discreteness is also a genuine modeling difference from continuous Gaussian corruption, but a softmax model can still assign vocabulary probabilities, so probability prediction is not prevented.",
  ),
  makeQuestion(
    "cme296-lect8-q04",
    "medium",
    "Which steps form a variable-noise masked-text training example?",
    [
      ["Sample a clean token sequence.", true],
      ["Sample a noise level or mask ratio.", true],
      ["Mask positions according to that level.", true],
      [
        "Train the model to predict the hidden token identities from the corrupted sequence and its noise condition.",
        true,
      ],
    ],
    "The training construction samples both data and a corruption level, applies the corresponding random mask, and learns conditional token reconstruction. Varying the level exposes one denoiser to lightly and heavily corrupted sequences, unlike a fixed-ratio masked-language objective that trains at only one characteristic corruption amount.",
  ),
  makeQuestion(
    "cme296-lect8-q05",
    "hard",
    "A six-token training sequence uses noise level \\(t=0.5\\), interpreted as an independent 50% mask probability per position. Which statements are correct?",
    [
      ["The expected number of masked positions is three.", true],
      [
        "Exactly three positions must be masked in every sampled corruption.",
        false,
      ],
      [
        "The probability that no position is masked is \\(0.5^6\\), not zero.",
        true,
      ],
      [
        "The same clean sequence can yield different corrupted inputs on different training draws.",
        true,
      ],
    ],
    "Independent Bernoulli masking gives an expected count of \\(6\\times0.5=3\\), but the realized count can range from zero to six. All six positions remain visible with probability \\(0.5^6\\), and resampling the mask creates multiple corruption patterns for the same clean sentence, which broadens the reconstruction task.",
  ),
  makeQuestion(
    "cme296-lect8-q06",
    "medium",
    "During coarse-to-fine text diffusion, what is the purpose of committing some predictions while remasking others?",
    [
      [
        "Committing preserves predictions currently treated as reliable, while remasking gives uncertain positions another chance to change.",
        true,
      ],
      [
        "The combination lets later iterations revise a globally inconsistent draft rather than freezing every first guess.",
        true,
      ],
      [
        "Committing converts the remaining positions into an autoregressive left-to-right suffix.",
        false,
      ],
      [
        "Remasking restores the exact clean token that occupied a position in a stored training sentence.",
        false,
      ],
    ],
    "Iterative refinement needs both progress and revision: selected tokens provide context for the next pass, while remasked positions remain editable when confidence or consistency is poor. This is not necessarily a left-to-right process, and generation does not retrieve a known clean ancestor from training; it constructs a new sequence through conditional predictions.",
  ),
  makeQuestion(
    "cme296-lect8-q07",
    "hard",
    "A confidence-based remasking policy is compared with uniform random remasking at the same budget. What is the intended advantage of the confidence-based policy?",
    [
      [
        "It allocates revision capacity preferentially to token predictions the model considers least reliable.",
        true,
      ],
      [
        "It proves that high-confidence tokens are globally correct and can never need revision.",
        false,
      ],
      [
        "It makes the number of denoising iterations independent of the chosen schedule.",
        false,
      ],
      [
        "It changes the tokenizer so uncertain tokens receive shorter byte encodings.",
        false,
      ],
    ],
    "Confidence-based remasking uses a limited revision budget where the current model signals the greatest uncertainty, which can be more targeted than choosing positions uniformly. Confidence is not a correctness proof and may itself be miscalibrated, while the iteration schedule and tokenizer remain separate design choices rather than consequences of remasking.",
  ),
  makeQuestion(
    "cme296-lect8-q08",
    "easy",
    "Which statements describe the initialization and output of a masked diffusion language sampler?",
    [
      ["It can initialize a fixed-length canvas with mask tokens.", true],
      ["It repeatedly predicts multiple masked positions in parallel.", true],
      [
        "It can stop the interpreted sequence at an end-of-sequence token.",
        true,
      ],
      [
        "It may need a strategy for output lengths that are not known in advance.",
        true,
      ],
    ],
    "A fully masked canvas is the discrete noisy starting point, and iterative passes can update many positions together before producing an unmasked sequence. Because real outputs have variable length, an end token or blockwise strategy is still needed; choosing a maximum canvas can otherwise waste computation beyond the actual endpoint.",
  ),
  makeAssertionReasonQuestion(
    "cme296-lect8-q09",
    "medium",
    "Diffusion-style text generation can be attractive for fill-in-the-middle code editing.",
    "A bidirectional masked denoiser can condition a missing region on tokens that appear on both sides of it.",
    4,
    "Both statements are true, and the reason explains the fit. A masked denoiser naturally treats the surrounding prefix and suffix as observed context while reconstructing the interior, whereas a strictly left-to-right factorization does not use the future suffix in its ordinary next-token conditional without an adapted representation or objective.",
  ),
  makeQuestion(
    "cme296-lect8-q10",
    "hard",
    "An autoregressive model needs \\(L=800\\) sequential token rounds for a response, while a diffusion language model uses \\(S=80\\) global refinement rounds. Which conclusions follow from this simplified comparison?",
    [
      ["The sequential-depth ratio is \\(800/80=10\\).", true],
      [
        "The diffusion model has fewer sequential rounds, but each round can still have substantial full-sequence compute.",
        true,
      ],
      [
        "The ratio alone proves a tenfold reduction in wall-clock latency on every hardware stack.",
        false,
      ],
      [
        "The diffusion model performs only 80 token predictions in total.",
        false,
      ],
    ],
    "The simplified critical-path count gives a nominal tenfold reduction in sequential rounds, which motivates the speed claim. Real latency also depends on work per round, parallel efficiency, sequence length, and implementation, and each diffusion pass may predict distributions at many positions; the model is not limited to one token prediction per refinement step.",
  ),
  makeAssertionReasonQuestion(
    "cme296-lect8-q11",
    "easy",
    "A masked diffusion language model automatically inherits every post-training method designed for autoregressive next-token policies without modification.",
    "Its generation state and action structure differ because it predicts and revises multiple positions across a noise schedule.",
    2,
    "The assertion is false and the reason is true. Objectives and credit assignment built around a left-to-right next-token trajectory may not map directly to parallel denoising and remasking decisions, so post-training techniques often need adaptation even when the backbone architecture and token vocabulary resemble those of an autoregressive model.",
  ),
  makeQuestion(
    "cme296-lect8-q12",
    "hard",
    "Which tradeoffs are relevant when choosing between pure masked diffusion, pure autoregression, and a block-diffusion hybrid for text?",
    [
      [
        "Pure masked diffusion offers parallel refinement inside a fixed canvas but must handle variable output length.",
        true,
      ],
      [
        "Autoregression handles variable length naturally through next-token generation and an end token but retains sequential dependence.",
        true,
      ],
      [
        "Block diffusion can generate one block in parallel and condition later blocks on earlier completed blocks.",
        true,
      ],
      [
        "Blockwise generation removes both within-block denoising cost and between-block sequential dependence.",
        false,
      ],
    ],
    "The three designs trade different kinds of parallelism and flexibility: global diffusion revises a canvas, autoregression grows a sequence token by token, and block diffusion combines parallel refinement within blocks with sequential continuation across blocks. A hybrid does not erase computation or all dependency; it changes the granularity at which sequential conditioning occurs.",
  ),
];
