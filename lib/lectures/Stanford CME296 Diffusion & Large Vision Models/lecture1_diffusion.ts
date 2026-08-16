import { Question } from "../../quiz";

type Lecture1Difficulty = "easy" | "medium" | "hard";
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
  difficulty: Lecture1Difficulty,
  prompt: string,
  optionSeeds: readonly OptionSeed[],
  explanation: string,
): Question {
  if (optionSeeds.length !== 4) {
    throw new Error(
      "CME296 Lecture 1 question " + id + " must have four options.",
    );
  }

  return {
    id,
    chapter: 1,
    difficulty,
    prompt,
    options: optionSeeds.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

function makeAssertionReasonQuestion(
  id: string,
  difficulty: Lecture1Difficulty,
  assertion: string,
  reason: string,
  correctChoice: AssertionReasonChoice,
  explanation: string,
): Question {
  return {
    id,
    chapter: 1,
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

const lecture1QuestionCandidates: Question[] = [
  makeQuestion(
    "cme296-lect1-q01",
    "easy",
    "A dataset contains many teddy-bear images but no text labels or prompts. Which objective matches unconditioned image generation?",
    [
      [
        "Learn a model distribution whose new samples resemble draws from the dataset distribution.",
        true,
      ],
      [
        "Learn a probability distribution that places high density on teddy-bear-like images without requiring a condition.",
        true,
      ],
      [
        "Learn a prompt-conditioned mapping even though no conditioning variable is part of the setup.",
        false,
      ],
      [
        "Learn a classifier that assigns each image to a predefined teddy-bear category.",
        false,
      ],
    ],
    "Unconditioned generation learns a distribution that assigns substantial probability to data-like images and supports drawing new samples without a prompt or class at generation time. The first two options express those density-modeling and sampling views of the same goal, while prompt conditioning and classification solve different problems with different inputs or outputs.",
  ),
  makeAssertionReasonQuestion(
    "cme296-lect1-q02",
    "easy",
    "Different initial noise samples can produce different images even when every later generation update is deterministic.",
    "For a fixed initial noise sample, a deterministic sequence of updates defines a single endpoint.",
    4,
    "The assertion and reason are both true. Initial Gaussian noise supplies the randomness that selects a trajectory, and a deterministic update rule maps one fixed starting point to one fixed endpoint; changing that starting noise can therefore change the generated image without adding randomness between later steps.",
  ),
  makeQuestion(
    "cme296-lect1-q03",
    "easy",
    "An RGB image has height \\(H\\) and width \\(W\\). Which statements correctly connect the image to the vector-valued diffusion notation?",
    [
      [
        "Flattening the image gives a vector with \\(3HW\\) scalar coordinates.",
        true,
      ],
      [
        "Each coordinate of \\(x_t\\) can represent one channel value at a particular pixel location.",
        true,
      ],
      [
        "Averaging red, green, and blue first gives the same representation with \\(HW\\) coordinates.",
        false,
      ],
      [
        "Flattening necessarily discards spatial location, so the original pixel arrangement cannot be recovered.",
        false,
      ],
    ],
    "An RGB pixel contributes three channel values, so a height-by-width image can be represented as a \\(3HW\\)-dimensional vector. Flattening changes the indexing scheme rather than erasing information: if the ordering convention is known, every channel value can be placed back at its pixel location; channel averaging would instead discard color information.",
  ),
  makeQuestion(
    "cme296-lect1-q04",
    "medium",
    "For an isotropic Gaussian \\(x\\sim\\mathcal{N}(\\mu,\\sigma^2 I)\\), which properties are correct?",
    [
      [
        "Every coordinate has variance \\(\\sigma^2\\), and the off-diagonal covariances are zero.",
        true,
      ],
      [
        "Its covariance ellipsoids reduce to spheres centered at \\(\\mu\\).",
        true,
      ],
      [
        "Its density depends on the squared Euclidean distance \\(\\lVert x-\\mu\\rVert^2\\).",
        true,
      ],
      [
        "Its covariance matrix must be estimated separately from the pixels of each clean training image.",
        false,
      ],
    ],
    "The covariance \\(\\sigma^2 I\\) gives the same variance in every coordinate and no cross-coordinate covariance, producing spherical equal-density contours around the mean. The familiar Gaussian density therefore depends on \\(\\lVert x-\\mu\\rVert^2\\); in the diffusion transition this covariance is chosen by the process design, not re-estimated from each individual image.",
  ),
  makeQuestion(
    "cme296-lect1-q05",
    "easy",
    "A diffusion implementation represents an image by \\(n\\) scalar values and samples \\(\\epsilon\\sim\\mathcal{N}(0,I)\\). Which statements correctly interpret this noise?",
    [
      [
        "The noise vector has the same dimension \\(n\\) as the image vector it perturbs.",
        true,
      ],
      [
        "Each coordinate \\(\\epsilon_i\\) has mean zero and variance one under the standard isotropic Gaussian.",
        true,
      ],
      [
        "The coordinates are sampled independently when \\(\\operatorname{Cov}(\\epsilon)=I\\).",
        true,
      ],
      [
        "Multiplying \\(\\epsilon\\) by a scalar \\(c\\) produces covariance \\(c^2I\\).",
        true,
      ],
    ],
    "Standard Gaussian image noise is a vector in the same ambient space as the image, with independent unit-variance coordinates. Scaling that vector by \\(c\\) scales standard deviations by \\(c\\) and variances by \\(c^2\\), which is why square roots of variance terms appear next to sampled noise in diffusion equations.",
  ),
  makeQuestion(
    "cme296-lect1-q06",
    "easy",
    "Which pairing correctly distinguishes the two processes in a Denoising Diffusion Probabilistic Model (DDPM)?",
    [
      [
        "The forward process \\(q\\) is chosen and adds noise; the reverse process \\(p_\\theta\\) is learned and removes noise.",
        true,
      ],
      [
        "The forward process \\(q\\) creates noisy training inputs; the learned reverse process \\(p_\\theta\\) drives generation.",
        true,
      ],
      [
        "The chosen \\(q\\) has no learned parameters, whereas \\(p_\\theta\\) depends on trainable \\(\\theta\\).",
        true,
      ],
      [
        "The reverse process adds the training noise, while the forward process generates a clean sample at inference.",
        false,
      ],
    ],
    "DDPM deliberately defines a tractable stochastic forward chain that corrupts clean data according to a known noise schedule. Learning is concentrated in the reverse transition \\(p_\\theta(x_{t-1}\\mid x_t)\\), which predicts how to move toward a less noisy state; using those roles in the opposite direction would not train a generative denoiser.",
  ),
  makeQuestion(
    "cme296-lect1-q07",
    "medium",
    "Consider \\(q(x_t\\mid x_{t-1})=\\mathcal{N}(\\sqrt{1-\\beta_t}\\,x_{t-1},\\beta_t I)\\). Which statements follow from this definition?",
    [
      ["The conditional mean is \\(\\sqrt{1-\\beta_t}\\,x_{t-1}\\).", true],
      ["The conditional covariance is \\(\\beta_t I\\).", true],
      [
        "The conditional mean is \\(\\beta_t x_{t-1}\\), because \\(\\beta_t\\) is the retained signal fraction.",
        false,
      ],
      [
        "The conditional covariance is \\(\\sqrt{\\beta_t}I\\), because the coefficient on sampled noise is a variance.",
        false,
      ],
    ],
    "A Gaussian is parameterized by its mean and covariance, so the displayed transition has mean \\(\\sqrt{1-\\beta_t}\\,x_{t-1}\\) and covariance \\(\\beta_t I\\). A sample can be written as that mean plus \\(\\sqrt{\\beta_t}\\epsilon\\), but \\(\\sqrt{\\beta_t}\\) is the noise standard-deviation coefficient rather than the covariance itself.",
  ),
  makeQuestion(
    "cme296-lect1-q08",
    "hard",
    "For one scalar coordinate, let \\(\\beta_t=0.16\\), \\(x_{t-1}=2\\), and sampled standard noise \\(\\epsilon=0.5\\). Which computation gives the forward sample \\(x_t\\)?",
    [
      ["\\(\\sqrt{0.84}(2)+\\sqrt{0.16}(0.5)\\approx 2.033\\)", true],
      ["\\(0.84(2)+0.16(0.5)=1.760\\)", false],
      ["\\(\\sqrt{0.16}(2)+\\sqrt{0.84}(0.5)\\approx 1.258\\)", false],
      ["\\(0.16(2)+0.84(0.5)=0.740\\)", false],
    ],
    "Sampling from the transition uses the square root of the retained-signal coefficient and the square root of the noise variance: \\(x_t=\\sqrt{1-\\beta_t}x_{t-1}+\\sqrt{\\beta_t}\\epsilon\\). Substitution gives \\(\\sqrt{0.84}\\cdot2+0.4\\cdot0.5\\approx2.033\\); the other calculations confuse variances with standard deviations or swap signal and noise weights.",
  ),
  makeQuestion(
    "cme296-lect1-q09",
    "easy",
    "Define \\(\\alpha_t=1-\\beta_t\\) and \\(\\bar{\\alpha}_t=\\prod_{i=1}^{t}\\alpha_i\\). Which statements are correct?",
    [
      [
        "\\(\\alpha_t\\) is the signal-retention factor associated with step \\(t\\).",
        true,
      ],
      [
        "\\(\\bar{\\alpha}_t\\) accumulates retention across the first \\(t\\) forward steps.",
        true,
      ],
      [
        "Increasing a particular \\(\\beta_t\\) decreases its corresponding \\(\\alpha_t\\).",
        true,
      ],
      [
        "\\(\\bar{\\alpha}_t\\) is obtained by adding the \\(\\alpha_i\\) values, so it can exceed one.",
        false,
      ],
    ],
    "The notation separates per-step retention, \\(\\alpha_t=1-\\beta_t\\), from cumulative retention, \\(\\bar{\\alpha}_t=\\prod_{i=1}^{t}\\alpha_i\\). With the usual schedule each factor lies between zero and one, so a larger noise variance reduces that step's retained signal and the cumulative quantity is a product rather than a sum.",
  ),
  makeQuestion(
    "cme296-lect1-q10",
    "hard",
    "At a certain noise level, \\(\\bar{\\alpha}_t=0.64\\). For one coordinate with \\(x_0=1.5\\) and \\(\\epsilon=-0.5\\), which calculations are correct for \\(q(x_t\\mid x_0)\\)?",
    [
      ["The conditional mean is \\(\\sqrt{0.64}(1.5)=1.2\\).", true],
      ["The sampled coordinate is \\(1.2+\\sqrt{0.36}(-0.5)=0.9\\).", true],
      ["The conditional variance is \\(\\sqrt{0.36}=0.6\\).", false],
      ["The sampled coordinate is \\(0.64(1.5)+0.36(-0.5)=0.78\\).", false],
    ],
    "The marginal is \\(q(x_t\\mid x_0)=\\mathcal{N}(\\sqrt{\\bar{\\alpha}_t}x_0,(1-\\bar{\\alpha}_t)I)\\), so its mean is \\(0.8\\cdot1.5=1.2\\), its variance is \\(0.36\\), and its standard deviation is \\(0.6\\). Using the sampled \\(\\epsilon=-0.5\\) gives \\(1.2+0.6(-0.5)=0.9\\); weighting by the variances directly is the nearby but incorrect alternative.",
  ),
  makeAssertionReasonQuestion(
    "cme296-lect1-q11",
    "medium",
    "The closed-form marginal \\(q(x_t\\mid x_0)\\) lets training create \\(x_t\\) without explicitly simulating \\(x_1,\\ldots,x_{t-1}\\).",
    "The cumulative signal coefficient \\(\\bar{\\alpha}_t\\) generally decreases as more factors between zero and one are multiplied.",
    5,
    "Both statements are true, but the reason is not the sufficient explanation for direct sampling. Direct sampling is possible because repeated linear Gaussian transitions collapse analytically into a Gaussian with known mean and covariance; the decreasing product describes how retained signal changes with time but does not by itself establish that closed form.",
  ),
  makeQuestion(
    "cme296-lect1-q12",
    "easy",
    "Which probability factorizations are valid for the forward diffusion variables?",
    [
      [
        "\\(q(x_{1:T}\\mid x_0)=\\prod_{t=1}^{T}q(x_t\\mid x_{t-1})\\) under the Markov forward process.",
        true,
      ],
      [
        "\\(p(x_{1:T})=p(x_1)\\prod_{t=2}^{T}p(x_t\\mid x_{1:t-1})\\) by the general chain rule.",
        true,
      ],
      [
        "\\(q(x_{1:T}\\mid x_0)=\\prod_{t=1}^{T}q(x_{t-1}\\mid x_t)\\) because the forward chain is defined in the reverse direction.",
        false,
      ],
      [
        "\\(p(x_1)=p(x_1,x_2,\\ldots,x_T)\\) because marginalization leaves the joint density unchanged.",
        false,
      ],
    ],
    "The ordinary chain rule conditions each variable on all earlier variables, while the forward diffusion's Markov assumption simplifies each factor to dependence on the immediately preceding state. Reversing those conditionals changes the modeled process, and obtaining a marginal such as \\(p(x_1)\\) requires integrating the other variables out rather than equating the marginal to the joint.",
  ),
  makeQuestion(
    "cme296-lect1-q13",
    "medium",
    "Why is direct maximum-likelihood evaluation of \\(p_\\theta(x_0)\\) difficult in the diffusion latent-variable formulation?",
    [
      [
        "It requires marginalizing the joint model over the unobserved path \\(x_{1:T}\\).",
        true,
      ],
      [
        "The integral spans the high-dimensional noisy states \\(x_1,\\ldots,x_T\\).",
        true,
      ],
      [
        "Computing \\(\\int p_\\theta(x_{0:T})\\,dx_{1:T}\\) exactly would aggregate all possible latent trajectories.",
        true,
      ],
      [
        "Selecting the single most likely trajectory is mathematically identical to marginalizing every trajectory.",
        false,
      ],
    ],
    "The observed-data likelihood is \\(p_\\theta(x_0)=\\int p_\\theta(x_{0:T})\\,dx_{1:T}\\), so all possible latent noise trajectories contribute. Those variables are image-sized and repeated across many time steps, making exact integration infeasible; replacing the integral with one best path changes a sum over probability mass into a maximum and is not equivalent.",
  ),
  makeAssertionReasonQuestion(
    "cme296-lect1-q14",
    "easy",
    "Maximizing \\(\\log p_\\theta(x_0)\\) has the same optimizer as maximizing \\(p_\\theta(x_0)\\) for positive likelihoods.",
    "The logarithm is strictly increasing and also turns products into sums that are easier to compute stably.",
    4,
    "The assertion and reason are true. Strict monotonicity preserves the ordering of positive likelihood values, so the maximizing parameters are unchanged, while the logarithm's product-to-sum conversion and improved numerical scale explain why log-likelihood is the practical objective.",
  ),
  makeQuestion(
    "cme296-lect1-q15",
    "hard",
    "Which statements correctly characterize the diffusion Evidence Lower Bound (ELBO)?",
    [
      [
        "\\(\\mathbb{E}_{q(x_0)}[\\log p_\\theta(x_0)]\\geq\\mathbb{E}_{q(x_{0:T})}\\left[\\log\\frac{p_\\theta(x_{0:T})}{q(x_{1:T}\\mid x_0)}\\right]\\)",
        true,
      ],
      [
        "The right-hand expectation is a lower bound on expected log-likelihood rather than an upper bound.",
        true,
      ],
      [
        "For a fixed \\(x_0\\), the gap to \\(\\log p_\\theta(x_0)\\) is \\(\\mathrm{KL}(q(x_{1:T}\\mid x_0)\\lVert p_\\theta(x_{1:T}\\mid x_0))\\).",
        true,
      ],
      [
        "Jensen's inequality is applied after introducing \\(q(x_{1:T}\\mid x_0)\\) into the marginal-likelihood expression.",
        true,
      ],
    ],
    "The Evidence Lower Bound is the expected log ratio of the model joint density to the known forward-path density, and Jensen's inequality places it below the expected observed-data log-likelihood. For each clean sample, the nonnegative gap equals the KL divergence from the forward path posterior to the model's path posterior, which explains both the inequality direction and when the bound becomes tight.",
  ),
  makeQuestion(
    "cme296-lect1-q16",
    "medium",
    "Which statements describe what the Evidence Lower Bound (ELBO) accomplishes in the DDPM derivation?",
    [
      [
        "It replaces an intractable observed-data log-likelihood with an objective that is bounded below.",
        true,
      ],
      [
        "It introduces the known forward process \\(q(x_{1:T}\\mid x_0)\\) as a convenient variational distribution.",
        true,
      ],
      [
        "Its expanded terms expose KL divergences between forward posteriors and learned reverse transitions.",
        true,
      ],
      [
        "Maximizing it can train \\(p_\\theta\\) without evaluating the full marginal likelihood integral exactly.",
        true,
      ],
    ],
    "The ELBO uses the tractable forward path distribution to form a computable lower bound on log-likelihood. Algebraic expansion produces KL-divergence terms that compare known forward posteriors with learned reverse steps, so optimizing the bound supplies a training signal without explicitly integrating over every latent trajectory.",
  ),
  makeQuestion(
    "cme296-lect1-q17",
    "easy",
    "Which properties of Kullback-Leibler divergence are correct?",
    [
      [
        "\\(\\mathrm{KL}(P\\lVert Q)\\) is nonnegative when the distributions satisfy the usual support conditions.",
        true,
      ],
      ["It is zero when \\(P\\) and \\(Q\\) agree almost everywhere.", true],
      [
        "It is symmetric, so \\(\\mathrm{KL}(P\\lVert Q)=\\mathrm{KL}(Q\\lVert P)\\).",
        false,
      ],
      [
        "It is a metric distance and therefore satisfies the triangle inequality.",
        false,
      ],
    ],
    "KL divergence measures an oriented discrepancy between distributions and is nonnegative, reaching zero when they match almost everywhere. Its direction matters because the expectation is taken under the first distribution, so it is generally asymmetric and does not obey all metric axioms such as the triangle inequality.",
  ),
  makeAssertionReasonQuestion(
    "cme296-lect1-q18",
    "medium",
    "The expanded ELBO can train a reverse transition by comparing \\(q(x_{t-1}\\mid x_t,x_0)\\) with \\(p_\\theta(x_{t-1}\\mid x_t)\\).",
    "Increasing the KL divergence between these distributions makes the lower bound tighter.",
    1,
    "The assertion is true: the expanded bound contains negative KL terms connecting the known forward posterior to the learned reverse model. The reason is false because maximizing the bound pushes those KL divergences downward, not upward; a larger mismatch subtracts more from the bound.",
  ),
  makeQuestion(
    "cme296-lect1-q19",
    "hard",
    "Which statements correctly evaluate or justify the forward posterior \\(q(x_{t-1}\\mid x_t,x_0)\\)?",
    [
      [
        "\\(q(x_{t-1}\\mid x_t,x_0)=\\frac{q(x_t\\mid x_{t-1})q(x_{t-1}\\mid x_0)}{q(x_t\\mid x_0)}\\)",
        true,
      ],
      [
        "The evidence \\(q(x_t\\mid x_0)\\) normalizes the numerator over possible values of \\(x_{t-1}\\).",
        true,
      ],
      [
        "The Markov property gives \\(q(x_t\\mid x_{t-1},x_0)=q(x_t\\mid x_{t-1})\\), supplying the likelihood term.",
        true,
      ],
      [
        "\\(q(x_{t-1}\\mid x_t,x_0)=q(x_t\\mid x_{t-1})q(x_t\\mid x_0)q(x_{t-1}\\mid x_0)\\)",
        false,
      ],
    ],
    "Bayes' rule gives posterior equals likelihood times prior divided by evidence. Here the Markov structure reduces the likelihood to the known step \\(q(x_t\\mid x_{t-1})\\), the prior conditioned on the clean image is \\(q(x_{t-1}\\mid x_0)\\), and integrating their product over \\(x_{t-1}\\) gives the evidence \\(q(x_t\\mid x_0)\\); omitting that normalizer does not produce a posterior density.",
  ),
  makeQuestion(
    "cme296-lect1-q20",
    "hard",
    "Which statements explain why \\(q(x_{t-1}\\mid x_t,x_0)\\) is tractable during DDPM training?",
    [
      [
        "Its Bayes-rule factors are known because the forward process was defined rather than learned.",
        true,
      ],
      [
        "Products and ratios of the relevant Gaussian densities yield a Gaussian posterior with analytic parameters.",
        true,
      ],
      [
        "The clean \\(x_0\\) is available for a sampled training example, even though it will not be available at generation time.",
        true,
      ],
      [
        "Evaluating this posterior requires first solving for the final trained parameters \\(\\theta\\).",
        false,
      ],
    ],
    "The forward schedule supplies analytic Gaussian transitions and marginals, so Bayes' rule produces an exact Gaussian posterior whose mean and variance can be written down. Training starts from an observed clean image, making conditioning on \\(x_0\\) legitimate for the target distribution; the posterior does not require a completed reverse model or known neural parameters.",
  ),
  makeQuestion(
    "cme296-lect1-q21",
    "easy",
    "The reverse transition is modeled as \\(p_\\theta(x_{t-1}\\mid x_t)=\\mathcal{N}(\\mu_\\theta(x_t,t),\\Sigma_\\theta(x_t,t))\\). Which statements correctly interpret this choice?",
    [
      [
        "The model receives the noisy state and its noise level when producing reverse-distribution parameters.",
        true,
      ],
      [
        "A Gaussian reverse family is motivated by the approximately Gaussian reversal of sufficiently small forward steps.",
        true,
      ],
      [
        "Comparing this Gaussian with the Gaussian forward posterior gives an analytic KL expression.",
        true,
      ],
      [
        "The neural parameterization can be rewritten to predict noise instead of outputting the posterior mean directly.",
        true,
      ],
    ],
    "The reverse family is a time-conditioned Gaussian whose parameters are functions of \\(x_t\\) and \\(t\\). Small forward corruptions motivate the Gaussian approximation, and pairing it with the exact Gaussian forward posterior makes the KL term analytic; DDPM then parameterizes the mean through a neural noise prediction, yielding the familiar regression objective.",
  ),
  makeAssertionReasonQuestion(
    "cme296-lect1-q22",
    "hard",
    "The posterior \\(q(x_{t-1}\\mid x_t,x_0)\\) is an exact tractable Gaussian under the chosen forward process.",
    "It becomes exact after the learned reverse distribution \\(p_\\theta(x_{t-1}\\mid x_t)\\) converges to it.",
    1,
    "The assertion is true because the forward chain is linear Gaussian and its Bayes-rule factors are known, so the posterior can be derived exactly during training. The reason is false: tractability is a property of the defined forward process and does not wait for the learned reverse model to converge.",
  ),
  makeQuestion(
    "cme296-lect1-q23",
    "hard",
    "Which statements correctly unpack the simplified DDPM objective \\(\\mathbb{E}\\lVert\\epsilon_\\theta(x_t,t)-\\epsilon\\rVert^2\\)?",
    [
      [
        "A clean training example \\(x_0\\), a time step \\(t\\), and Gaussian noise \\(\\epsilon\\) are sampled.",
        true,
      ],
      [
        "The noisy input is formed as \\(x_t=\\sqrt{\\bar{\\alpha}_t}x_0+\\sqrt{1-\\bar{\\alpha}_t}\\epsilon\\).",
        true,
      ],
      [
        "The network output \\(\\epsilon_\\theta(x_t,t)\\) predicts the noise vector used to construct that \\(x_t\\).",
        true,
      ],
      [
        "The squared error \\(\\lVert\\epsilon_\\theta-\\epsilon\\rVert^2\\) supplies a differentiable regression loss for backpropagation.",
        true,
      ],
    ],
    "The simplified loss turns the variational derivation into supervised noise regression on synthetically corrupted examples. Because the implementation samples \\(x_0\\), \\(t\\), and \\(\\epsilon\\), it knows the target noise exactly, can construct \\(x_t\\) in one formula, and can update \\(\\theta\\) through an ordinary squared-error gradient.",
  ),
  makeQuestion(
    "cme296-lect1-q24",
    "medium",
    "Why is the time step \\(t\\) supplied to the noise-prediction network along with \\(x_t\\)?",
    [
      [
        "The value \\(t\\) identifies the noise level and the signal/noise coefficients the network should account for.",
        true,
      ],
      [
        "Conditioning one shared network on both \\(x_t\\) and \\(t\\) lets it learn behavior across the schedule.",
        true,
      ],
      [
        "It serves as the semantic class label of the clean image, such as teddy bear or giraffe.",
        false,
      ],
      [
        "It makes the noisy image \\(x_t\\) unnecessary, because the schedule alone determines the sampled noise realization.",
        false,
      ],
    ],
    "The same architecture is reused at every diffusion level, so \\(t\\) tells it how strongly signal and noise were mixed and helps calibrate the prediction. Time is not a content label, and the schedule specifies a distribution rather than the realized noise vector; \\(x_t\\) remains essential evidence about the particular sample being denoised.",
  ),
  makeQuestion(
    "cme296-lect1-q25",
    "hard",
    "For one training example, \\(\\epsilon=(0.4,-0.2)\\) and \\(\\epsilon_\\theta(x_t,t)=(0.1,-0.4)\\). What is the unweighted squared-error loss?",
    [
      ["\\((0.1-0.4)^2+(-0.4+0.2)^2=0.13\\)", true],
      ["\\((0.1+0.4)^2+(-0.4-0.2)^2=0.61\\)", false],
      ["\\(\\sqrt{(0.1-0.4)^2+(-0.4+0.2)^2}\\approx0.361\\)", false],
      ["\\(\\lvert0.1-0.4\\rvert+\\lvert-0.4+0.2\\rvert=0.50\\)", false],
    ],
    "The DDPM regression loss uses the squared \\(L_2\\) norm, so the coordinate errors are \\(-0.3\\) and \\(-0.2\\), and their squared sum is \\(0.09+0.04=0.13\\). Taking the square root would give the \\(L_2\\) norm rather than its square, while the absolute-value expression is an \\(L_1\\) loss.",
  ),
  makeQuestion(
    "cme296-lect1-q26",
    "easy",
    "Which operations belong to one DDPM training update?",
    [
      ["Sample a clean image, a Gaussian noise vector, and a time step.", true],
      [
        "Construct the selected noisy state directly with the closed-form marginal.",
        true,
      ],
      ["Backpropagate the error between predicted and sampled noise.", true],
      [
        "Run the complete forward chain from step zero through the sampled time before every gradient update.",
        false,
      ],
    ],
    "A training example is made by sampling \\(x_0\\), \\(t\\), and \\(\\epsilon\\), constructing \\(x_t\\) directly, and regressing the network's prediction toward the known \\(\\epsilon\\). The closed-form marginal is specifically valuable because it removes the need to simulate every preceding corruption step for each minibatch example.",
  ),
  makeQuestion(
    "cme296-lect1-q27",
    "medium",
    "Which consequences follow from sampling \\(t\\) across the DDPM schedule during training?",
    [
      [
        "The same parameters \\(\\theta\\) receive examples from low, intermediate, and high values of \\(t\\).",
        true,
      ],
      [
        "Different minibatch examples can use distinct sampled values \\(t_i\\).",
        true,
      ],
      [
        "The expected loss trains one \\(\\epsilon_\\theta(x_t,t)\\) rather than a separate network for every step.",
        true,
      ],
      [
        "The sampled target remains the exact \\(\\epsilon\\) used to create each example's \\(x_t\\).",
        true,
      ],
    ],
    "Sampling time steps turns the expectation over \\(t\\) into ordinary stochastic training and exposes one shared model to the full range of corruption levels. Each example can have its own \\(t\\), but its target remains known because the implementation also sampled the exact Gaussian noise used in the marginal formula.",
  ),
  makeAssertionReasonQuestion(
    "cme296-lect1-q28",
    "easy",
    "DDPM training must simulate every forward transition from \\(x_0\\) to \\(x_t\\) before it can compute a loss at time \\(t\\).",
    "The closed-form marginal expresses \\(x_t\\) directly as a weighted clean image plus a weighted Gaussian noise sample.",
    2,
    "The assertion is false and the reason is true. Gaussian closure yields \\(x_t=\\sqrt{\\bar{\\alpha}_t}x_0+\\sqrt{1-\\bar{\\alpha}_t}\\epsilon\\), so a training batch can jump directly to its sampled noise level rather than executing all earlier forward transitions.",
  ),
  makeQuestion(
    "cme296-lect1-q29",
    "hard",
    "Which statements correctly unpack the stochastic DDPM reverse update shown for sampling?",
    [
      [
        "\\(x_{t-1}=\\frac{1}{\\sqrt{\\alpha_t}}\\left(x_t-\\frac{1-\\alpha_t}{\\sqrt{1-\\bar{\\alpha}_t}}\\epsilon_\\theta(x_t,t)\\right)+\\sigma_t z\\)",
        true,
      ],
      [
        "Because \\(1-\\alpha_t=\\beta_t\\), the predicted-noise coefficient inside the parentheses is \\(\\beta_t/\\sqrt{1-\\bar{\\alpha}_t}\\).",
        true,
      ],
      [
        "The factor \\(1/\\sqrt{\\alpha_t}\\) rescales the denoised expression before the stochastic term is added.",
        true,
      ],
      [
        "At a nonterminal step, fresh \\(z\\sim\\mathcal{N}(0,I)\\) is multiplied by the reverse standard deviation \\(\\sigma_t\\).",
        true,
      ],
    ],
    "The reverse mean rescales by \\(1/\\sqrt{\\alpha_t}\\) and subtracts the predicted noise with coefficient \\((1-\\alpha_t)/\\sqrt{1-\\bar{\\alpha}_t}=\\beta_t/\\sqrt{1-\\bar{\\alpha}_t}\\). At nonterminal steps, an independent standard Gaussian \\(z\\) is scaled by the reverse standard deviation \\(\\sigma_t\\), so the prediction and fresh sampling noise retain distinct roles.",
  ),
  makeQuestion(
    "cme296-lect1-q30",
    "medium",
    "Which statements correctly describe the DDPM inference recipe?",
    [
      ["Initialize \\(x_T\\) by sampling from \\(\\mathcal{N}(0,I)\\).", true],
      [
        "Apply the learned reverse update repeatedly from high noise levels toward \\(t=0\\).",
        true,
      ],
      [
        "Use \\(\\epsilon_\\theta(x_t,t)\\) to estimate the noise component that should be removed at each step.",
        true,
      ],
      [
        "Use the reverse Gaussian variance to add appropriately scaled stochasticity at nonterminal reverse steps.",
        true,
      ],
    ],
    "Generation begins from an easy-to-sample standard Gaussian and follows the learned reverse chain down the schedule. Each step uses the time-conditioned noise estimate to form a less noisy mean and, in stochastic DDPM sampling, adds a variance-scaled Gaussian draw; after the final transition the state is interpreted as the generated clean image.",
  ),
  makeAssertionReasonQuestion(
    "cme296-lect1-q31",
    "medium",
    "DDPM inference begins from a clean training image and applies the forward corruption process to create a new sample.",
    "The known forward process \\(q\\) can denoise its own terminal noise without a learned reverse model.",
    3,
    "Both statements are false. Unconditioned inference begins from newly sampled Gaussian noise rather than a training image, and the defined forward process moves from clean states toward noise; generating in the other direction requires the learned reverse distribution \\(p_\\theta\\).",
  ),
  makeQuestion(
    "cme296-lect1-q32",
    "easy",
    "Which facts explain the main inference-time limitation of the original DDPM sampler?",
    [
      [
        "The reverse update is evaluated once for each of many time steps.",
        true,
      ],
      [
        "The original setup can use a schedule on the order of \\(T=1000\\) steps.",
        true,
      ],
      [
        "Its sampling cost is \\(O(T)\\) model evaluations rather than a single generator forward pass.",
        true,
      ],
      [
        "Its bottleneck is that the training noise-regression loss cannot be evaluated numerically.",
        false,
      ],
    ],
    "The simplified training loss is straightforward to evaluate, but ancestral generation still traverses the reverse schedule. With roughly a thousand neural evaluations, sampling can be orders of magnitude slower than a Variational Autoencoder (VAE) or Generative Adversarial Network (GAN) that produces an image in one generator pass.",
  ),
  makeQuestion(
    "cme296-lect1-q33",
    "hard",
    "A derivation algebraically unrolls several DDPM reverse steps to connect \\(x_{\\tau_i}\\) with \\(x_{\\tau_{i-1}}\\). Which observations show why this alone does not accelerate sampling?",
    [
      [
        "The expanded expression still contains predictions \\(\\epsilon_\\theta(x_s,s)\\) for intermediate states.",
        true,
      ],
      [
        "Computing each \\(\\epsilon_\\theta(x_s,s)\\) still requires the sequential model evaluations the derivation hoped to skip.",
        true,
      ],
      [
        "An identity linking \\(x_{\\tau_i}\\) and \\(x_{\\tau_{i-1}}\\) does not make intermediate state-dependent terms free.",
        true,
      ],
      [
        "An accelerated rule must be evaluable at selected \\(\\tau_i\\) values rather than at every omitted state.",
        true,
      ],
    ],
    "Algebraic unrolling can place the distant endpoints in one equation while leaving a sum of model outputs evaluated at every intermediate state. Because each later prediction depends on the state produced by the preceding update, those evaluations remain sequential; genuine acceleration needs a new transition rule that operates directly between selected time indices.",
  ),
  makeQuestion(
    "cme296-lect1-q34",
    "medium",
    "What motivates removing inter-step stochasticity when making large jumps between diffusion time indices?",
    [
      [
        "A large skipped interval combined with newly sampled transition noise can noticeably degrade generated quality.",
        true,
      ],
      [
        "A deterministic transition lets a fixed noisy state map directly to the next selected state without an added random perturbation.",
        true,
      ],
      [
        "Increasing the fresh noise at each large jump reliably compensates for the missing model evaluations.",
        false,
      ],
      [
        "Setting every forward variance to one preserves the original DDPM marginals while eliminating quality loss.",
        false,
      ],
    ],
    "Empirically, naively skipping many steps while retaining substantial fresh stochasticity gives poor approximations to the distant lower-noise state. DDIM therefore seeks a compatible process whose generation transition can be deterministic; simply injecting more noise or replacing the carefully designed forward schedule with unit variance would not preserve the intended marginals.",
  ),
  makeAssertionReasonQuestion(
    "cme296-lect1-q35",
    "hard",
    "Denoising Diffusion Implicit Models (DDIM) require a new noise-prediction training loss that is incompatible with a trained DDPM.",
    "DDIM chooses a family of forward processes with the same marginals \\(q_\\sigma(x_t\\mid x_0)=q(x_t\\mid x_0)\\) used by the DDPM loss.",
    2,
    "The assertion is false and the reason is true. The simplified DDPM objective depends on the per-time marginals, so preserving those marginals lets DDIM reuse the same noise-prediction model while changing the coupling between time steps and the generation procedure.",
  ),
  makeQuestion(
    "cme296-lect1-q36",
    "medium",
    "Which properties define the DDIM reformulation used for accelerated sampling?",
    [
      [
        "Its alternative forward family preserves the DDPM marginals \\(q(x_t\\mid x_0)\\).",
        true,
      ],
      [
        "A parameter such as \\(\\sigma_t\\) controls how much stochasticity remains between generation steps.",
        true,
      ],
      [
        "Choosing \\(\\sigma_t=0\\) makes the selected reverse trajectory deterministic once \\(x_T\\) is fixed.",
        true,
      ],
      [
        "It must preserve every original one-step Markov conditional \\(q(x_t\\mid x_{t-1})\\), not just the marginals.",
        false,
      ],
    ],
    "DDIM preserves the marginal corruption distribution that underlies the DDPM training objective but changes how states at different times are jointly coupled. The \\(\\sigma_t\\) family controls inter-step randomness, and the zero setting gives a deterministic path from a fixed \\(x_T\\); retaining every original Markov conditional would remove the freedom needed for this reformulation.",
  ),
  makeQuestion(
    "cme296-lect1-q37",
    "medium",
    "At time \\(t\\), which statements correctly describe the DDIM clean-image estimate?",
    [
      [
        "\\(\\hat{x}_0(t)=\\frac{x_t-\\sqrt{1-\\bar{\\alpha}_t}\\,\\epsilon_\\theta(x_t,t)}{\\sqrt{\\bar{\\alpha}_t}}\\)",
        true,
      ],
      [
        "It rearranges the marginal noising equation after replacing the unknown sampled noise by the model prediction.",
        true,
      ],
      [
        "\\(\\hat{x}_0(t)=\\frac{x_t+\\sqrt{\\bar{\\alpha}_t}\\,\\epsilon_\\theta(x_t,t)}{\\sqrt{1-\\bar{\\alpha}_t}}\\)",
        false,
      ],
      [
        "It is computed from \\(t\\) alone, so the current noisy state does not affect the estimate.",
        false,
      ],
    ],
    "Starting from \\(x_t=\\sqrt{\\bar{\\alpha}_t}x_0+\\sqrt{1-\\bar{\\alpha}_t}\\epsilon\\), solving for \\(x_0\\) gives the displayed estimate once \\(\\epsilon\\) is replaced by \\(\\epsilon_\\theta(x_t,t)\\). Both the current noisy state and the time-dependent coefficients matter; changing the sign or denominator would no longer invert the marginal relation.",
  ),
  makeQuestion(
    "cme296-lect1-q38",
    "medium",
    "For deterministic DDIM sampling with \\(\\sigma_t=0\\), which statements are correct?",
    [
      [
        "The update combines a scaled clean-image estimate with a scaled predicted-noise direction.",
        true,
      ],
      [
        "No fresh Gaussian perturbation is added between the selected generation states.",
        true,
      ],
      [
        "Randomness can still enter through the initial draw \\(x_T\\sim\\mathcal{N}(0,I)\\).",
        true,
      ],
      [
        "Repeating the sampler with the same model, schedule, and \\(x_T\\) yields the same trajectory.",
        true,
      ],
    ],
    "With \\(\\sigma_t=0\\), the DDIM transition contains the predicted clean component and the predicted noise-direction component but no newly sampled inter-step noise. The sampler remains generative because \\(x_T\\) is random; after that draw is fixed, the model and selected time schedule determine a reproducible path and endpoint.",
  ),
  makeQuestion(
    "cme296-lect1-q39",
    "hard",
    "A DDPM uses \\(T=1000\\) model evaluations, while a DDIM schedule keeps \\(S\\) selected steps. Which calculations or conclusions are correct?",
    [
      [
        "Using \\(S=50\\) corresponds to a nominal \\(1000/50=20\\times\\) speedup.",
        true,
      ],
      [
        "Using \\(S=20\\) corresponds to a nominal \\(1000/20=50\\times\\) speedup.",
        true,
      ],
      [
        "Reducing \\(S\\) increases the nominal speedup but also makes the jumps between selected times larger.",
        true,
      ],
      [
        "Using \\(S=100\\) corresponds to a nominal \\(100\\times\\) speedup.",
        false,
      ],
    ],
    "The nominal speedup is \\(T/S\\), so retaining 50 evaluations gives \\(20\\times\\), retaining 20 gives \\(50\\times\\), and retaining 100 gives \\(10\\times\\). Smaller \\(S\\) means fewer neural evaluations but wider schedule gaps, which is why acceleration must be traded against approximation and image quality.",
  ),
  makeQuestion(
    "cme296-lect1-q40",
    "hard",
    "A CIFAR-10 DDIM experiment reports the following relative FID impacts: \\(10\\times\\): \\(+3\\%\\), \\(20\\times\\): \\(+16\\%\\), \\(50\\times\\): \\(+70\\%\\), and \\(100\\times\\): \\(+330\\%\\). If the allowed FID impact is at most \\(20\\%\\), which conclusions are correct?",
    [
      [
        "The \\(10\\times\\) schedule satisfies the budget, although it is not the fastest feasible listed choice.",
        true,
      ],
      [
        "Use the \\(20\\times\\) schedule with a \\(+16\\%\\) FID impact.",
        true,
      ],
      [
        "Use the \\(50\\times\\) schedule with a \\(+70\\%\\) FID impact.",
        false,
      ],
      [
        "Use the \\(100\\times\\) schedule with a \\(+330\\%\\) FID impact.",
        false,
      ],
    ],
    "Both \\(10\\times\\) and \\(20\\times\\) remain within the \\(20\\%\\) quality-impact budget, but \\(20\\times\\) is the faster of those feasible schedules. The \\(50\\times\\) and \\(100\\times\\) settings violate the stated constraint, illustrating that aggressive step skipping can exchange substantial FID degradation for additional speed.",
  ),
];

const diffusionGemmaStudyGuideQuestionIds = new Set([
  "cme296-lect1-q01",
  "cme296-lect1-q02",
  "cme296-lect1-q06",
  "cme296-lect1-q07",
  "cme296-lect1-q08",
  "cme296-lect1-q09",
  "cme296-lect1-q10",
  "cme296-lect1-q23",
  "cme296-lect1-q24",
  "cme296-lect1-q25",
  "cme296-lect1-q26",
  "cme296-lect1-q27",
  "cme296-lect1-q28",
  "cme296-lect1-q29",
  "cme296-lect1-q30",
  "cme296-lect1-q31",
  "cme296-lect1-q32",
  "cme296-lect1-q33",
  "cme296-lect1-q34",
  "cme296-lect1-q35",
  "cme296-lect1-q36",
  "cme296-lect1-q37",
  "cme296-lect1-q38",
  "cme296-lect1-q39",
  "cme296-lect1-q40",
]);

export const stanfordCME296Lecture1DiffusionQuestions: Question[] =
  lecture1QuestionCandidates.filter((question) =>
    diffusionGemmaStudyGuideQuestionIds.has(question.id),
  );
