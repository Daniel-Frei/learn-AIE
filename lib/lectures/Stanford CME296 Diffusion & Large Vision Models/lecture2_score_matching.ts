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
    chapter: 2,
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
    chapter: 2,
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

export const stanfordCME296Lecture2ScoreMatchingQuestions: Question[] = [
  makeQuestion(
    "cme296-lect2-q01",
    "easy",
    "Which properties characterize a standard Wiener process used to build a continuous diffusion model?",
    [
      ["Its increments over disjoint time intervals are independent.", true],
      [
        "An increment over duration \\(\\Delta t\\) is Gaussian with variance proportional to \\(\\Delta t\\).",
        true,
      ],
      [
        "Every sample path is a differentiable straight line once its starting point is fixed.",
        false,
      ],
      [
        "Its increments are chosen by the learned score network during the forward process.",
        false,
      ],
    ],
    "A Wiener process supplies independent Gaussian increments whose variance grows with elapsed time, making it the continuous analogue of repeatedly adding small Gaussian perturbations. Its paths are stochastic and nowhere classically differentiable, and the forward Wiener noise is defined by the process rather than selected by the learned score model.",
  ),
  makeQuestion(
    "cme296-lect2-q02",
    "easy",
    "In the stochastic differential equation \\(d x=f(x,t)\,dt+g(t)\,dW_t\\), which interpretation is correct?",
    [
      [
        "The drift \\(f\\) gives deterministic motion, while \\(g\\) scales stochastic Wiener increments.",
        true,
      ],
      [
        "The drift \\(f\\) is the sampled Gaussian noise, while \\(g\\) is the probability density.",
        false,
      ],
      [
        "Both terms are deterministic because an infinitesimal time interval removes randomness.",
        false,
      ],
      [
        "The equation specifies only a terminal distribution and no local dynamics.",
        false,
      ],
    ],
    "The drift term describes the systematic local change in state, whereas the diffusion coefficient controls the amplitude of random Wiener motion. Infinitesimal notation does not eliminate randomness, and an SDE specifies local dynamics whose accumulated evolution induces a whole path of probability distributions rather than only an endpoint.",
  ),
  makeQuestion(
    "cme296-lect2-q03",
    "medium",
    "A researcher takes the small-step limit of a discrete Gaussian corruption chain. Which conceptual correspondences are valid?",
    [
      [
        "Many small Gaussian innovations become a Wiener-driven stochastic term.",
        true,
      ],
      ["The systematic shrinkage or motion becomes the SDE drift.", true],
      [
        "A time-varying discrete noise schedule becomes time-dependent continuous coefficients.",
        true,
      ],
      [
        "Taking the limit converts a stochastic corruption process into a deterministic classifier.",
        false,
      ],
    ],
    "The continuous limit retains the two ingredients already visible in small discrete steps: systematic motion and Gaussian randomness. These become drift and diffusion coefficients that can vary with continuous time; the limit changes the mathematical description of corruption, but it neither removes stochasticity nor turns generation into classification.",
  ),
  makeQuestion(
    "cme296-lect2-q04",
    "medium",
    "Which statements correctly contrast variance-preserving and variance-exploding continuous corruption processes?",
    [
      [
        "A variance-preserving construction balances signal attenuation with injected noise so normalized data retain roughly constant total variance.",
        true,
      ],
      [
        "A variance-exploding construction can add noise without an offsetting signal shrinkage, causing variance to grow.",
        true,
      ],
      [
        "The DDPM continuous formulation is associated with a variance-preserving SDE.",
        true,
      ],
      [
        "Noise-conditioned score matching with progressively larger additive noise is associated with a variance-exploding SDE.",
        true,
      ],
    ],
    "Variance-preserving dynamics trade away clean-signal variance while adding noise variance, so normalized inputs remain near a fixed overall scale; this is the continuous view commonly paired with DDPM. Variance-exploding dynamics instead accumulate additive noise and grow the scale, matching the continuous interpretation of noise-conditioned score models with increasing noise levels.",
  ),
  makeQuestion(
    "cme296-lect2-q05",
    "hard",
    "Suppose normalized data have variance one and \\(x_t=\\sqrt{\\bar\\alpha_t}x_0+\\sqrt{1-\\bar\\alpha_t}\\,\\epsilon\\), where \\(x_0\\) and \\(\\epsilon\\) are independent with unit variance. What follows?",
    [
      [
        "The variance is approximately \\(\\bar\\alpha_t+(1-\\bar\\alpha_t)=1\\).",
        true,
      ],
      ["The construction motivates the name variance preserving.", true],
      [
        "The variance must equal \\(\\sqrt{\\bar\\alpha_t}+\\sqrt{1-\\bar\\alpha_t}\\).",
        false,
      ],
      [
        "The result requires the clean signal and noise to have covariance one.",
        false,
      ],
    ],
    "For independent unit-variance components, variances add after squaring their coefficients, giving \\(\\bar\\alpha_t+1-\\bar\\alpha_t=1\\). Adding the unsquared standard-deviation coefficients is incorrect, and nonzero covariance would introduce a cross term rather than being required for the variance-preserving calculation.",
  ),
  makeQuestion(
    "cme296-lect2-q06",
    "medium",
    "Why must a reverse-time diffusion sampler use information about the intermediate density \\(p_t(x)\\)?",
    [
      [
        "Random diffusion alone would continue spreading probability mass instead of concentrating it into data-like regions.",
        true,
      ],
      [
        "The score \\(\\nabla_x\\log p_t(x)\\) supplies a local direction toward higher density at the current noise level.",
        true,
      ],
      [
        "A score-dependent drift correction compensates for the dispersive effect of reverse-time stochastic diffusion.",
        true,
      ],
      [
        "The terminal Gaussian sample uniquely identifies its clean training ancestor without any density information.",
        false,
      ],
    ],
    "The reverse process must undo a distribution-level spreading operation, so it needs the time-dependent score to redirect mass toward regions that are plausible under the current marginal. A terminal noise vector has no uniquely labeled clean ancestor; generation samples from the learned distribution, and the score correction works together with reverse-time noise rather than recovering a stored example.",
  ),
  makeQuestion(
    "cme296-lect2-q07",
    "hard",
    "A reverse SDE contains a drift correction proportional to \\(g(t)^2\\nabla_x\\log p_t(x)\\). Which claim best captures the role of the squared diffusion coefficient?",
    [
      [
        "It scales the density-gradient correction according to how strongly the forward process diffuses at that time.",
        true,
      ],
      [
        "It converts the score into a normalized probability density whose integral is one.",
        false,
      ],
      [
        "It makes the reverse path deterministic by canceling every Wiener increment pointwise.",
        false,
      ],
      [
        "It replaces the need to evaluate a time-dependent score network.",
        false,
      ],
    ],
    "The reverse drift correction is tied to the amount of stochastic spreading introduced by the forward dynamics, so its magnitude contains the diffusion strength squared. This factor neither normalizes the score nor cancels individual random increments, and the correction still requires an estimate of the score at the current state and time.",
  ),
  makeQuestion(
    "cme296-lect2-q08",
    "easy",
    "Which elements belong to score-based generation with a reverse stochastic differential equation?",
    [
      [
        "Sample an initial state from the simple terminal noise distribution.",
        true,
      ],
      ["Evaluate a score model at the current state and time.", true],
      [
        "Numerically advance from the noisy end of time toward the data end.",
        true,
      ],
      [
        "Include appropriately scaled stochastic increments in an Euler-Maruyama update.",
        true,
      ],
    ],
    "Reverse-SDE generation starts from the tractable terminal prior and repeatedly combines a score-informed drift with stochastic increments while integrating toward clean data. The score is time dependent because the marginal density changes along the path, and Euler-Maruyama is the numerical analogue of Euler's method that also accounts for Wiener noise.",
  ),
  makeAssertionReasonQuestion(
    "cme296-lect2-q09",
    "medium",
    "The Wiener process used in the reverse-time SDE is treated as a distinct process from the Wiener process that generated a particular forward trajectory.",
    "Generation samples a valid reverse stochastic process rather than replaying the exact random increments of an observed forward path.",
    4,
    "Both statements are true, and the reason explains the assertion. A generative reverse SDE begins from fresh terminal noise and draws its own stochastic increments while using the learned score; it does not have access to, or need to reconstruct, the increment-by-increment noise history of any particular forward-corrupted training example.",
  ),
  makeQuestion(
    "cme296-lect2-q10",
    "hard",
    "An Euler-Maruyama sampler reduces its time-step magnitude while keeping the same total interval. Which consequences are expected?",
    [
      [
        "It uses more score-network evaluations to traverse the interval.",
        true,
      ],
      [
        "It usually reduces numerical discretization error when the learned field is well behaved.",
        true,
      ],
      [
        "It removes stochasticity because smaller Wiener increments have exactly zero variance.",
        false,
      ],
      [
        "It changes the trained terminal distribution into a different probability family by definition.",
        false,
      ],
    ],
    "A finer discretization takes more updates and therefore more score evaluations, but it generally approximates the continuous reverse dynamics more accurately. Each Wiener increment becomes smaller in scale, with variance proportional to the step duration, yet it is still random; changing solver resolution does not redefine the model's terminal prior.",
  ),
  makeAssertionReasonQuestion(
    "cme296-lect2-q11",
    "easy",
    "The reverse-time correction can be built from the score even when the normalized density \\(p_t(x)\\) is not available in closed form.",
    "Taking \\(\\nabla_x\\log p_t(x)\\) eliminates any normalizing constant that is independent of \\(x\\).",
    4,
    "Both statements are true, and the reason gives the key mathematical explanation. If an unnormalized density is known up to a time-dependent constant, the spatial gradient of its log removes that constant; in learned score models, the network directly approximates this usable local quantity rather than reconstructing a globally normalized density.",
  ),
  makeQuestion(
    "cme296-lect2-q12",
    "hard",
    "A reverse-SDE implementation produces samples with the right broad shape but loses narrow modes. Which changes are conceptually relevant to investigate?",
    [
      [
        "Improve the accuracy of the time-conditioned score in low-density transition regions.",
        true,
      ],
      [
        "Use a finer numerical discretization where the reverse dynamics change rapidly.",
        true,
      ],
      [
        "Check that drift, diffusion, and reverse-time direction conventions are implemented consistently.",
        true,
      ],
      [
        "Replace the stochastic increments with the recorded forward increments of unrelated training samples.",
        false,
      ],
    ],
    "Mode loss can arise when the learned score misdirects mass or when coarse numerical steps fail to resolve sharp reverse dynamics, and sign or time-direction mistakes can corrupt the sampler entirely. Reusing noise histories from unrelated training examples is not part of reverse-SDE generation and would not repair the learned distributional dynamics.",
  ),
];
