import type { Question } from "../../quiz";
import { referenceQuestion } from "../../questionReferences";
import { stanfordCME295Lecture5PreferenceTuningQuestions } from "../Stanford CME295 Transformers & LLMs/lecture5_preference_tuning";
import { stanfordCME295Lecture6ReasoningQuestions } from "../Stanford CME295 Transformers & LLMs/lecture6_reasoning";

export const stanfordCME2952026Lecture4Questions: Question[] = [
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(
    stanfordCME295Lecture5PreferenceTuningQuestions,
    "cme295-lect5-q07",
  ),
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q02",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "For one prompt, a two-token response has policy probabilities \\(0.4\\) for its first token and \\(0.25\\) for its second token given the first. Which statements are correct?",
    options: [
      {
        text: "The response probability is 0.1.",
        isCorrect: true,
      },
      {
        text: "The second token has probability 0.25 independently of which first token was chosen.",
        isCorrect: false,
      },
      {
        text: "The response log probability is ln(0.4) + ln(0.25).",
        isCorrect: true,
      },
      {
        text: "The response probability is 0.65.",
        isCorrect: false,
      },
    ],
    explanation:
      "An autoregressive trajectory probability multiplies conditional token probabilities, giving 0.4 times 0.25, or 0.1. Taking logarithms turns that product into a sum; the second probability is conditional on the first token and is not an independence claim.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q03",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements distinguish subjective preference rewards from verifiable task rewards?",
    options: [
      {
        text: "An execution-based coding task can use test outcomes.",
        isCorrect: true,
      },
      {
        text: "A human preference comparison can train a Bradley-Terry-style reward model.",
        isCorrect: true,
      },
      {
        text: "A known-answer math task can use a deterministic answer check.",
        isCorrect: true,
      },
      {
        text: "Both reward types can supply a signal for updating a language-model policy.",
        isCorrect: true,
      },
    ],
    explanation:
      "Preference rewards model judgments about desirable responses, whereas verifiable rewards can check concrete task outcomes. Both can drive policy optimization; which signal is available depends on the task, and test coverage or reward-model quality still determines what the signal actually measures.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q04",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "A toy verifiable reward is\n\n\\[\nR = 0.2F + 0.8C,\n\\]\n\nwhere \\(F=1\\) means the required reasoning/answer format is present and \\(C=1\\) means the final answer is correct. Which statements are correct?",
    options: [
      {
        text: "A formatted wrong answer \\((F=1, C=0)\\) outranks an unformatted correct answer \\((F=0, C=1)\\) because format is the only verifiable part.",
        isCorrect: false,
      },
      {
        text: "A response with \\(F=0, C=0\\) receives the same reward as a formatted wrong answer.",
        isCorrect: false,
      },
      {
        text: "A response with \\(F=0, C=1\\) outranks a response with \\(F=1, C=0\\).",
        isCorrect: true,
      },
      {
        text: "A response with \\(F=1, C=1\\) receives reward 1.0.",
        isCorrect: true,
      },
    ],
    explanation:
      "The arithmetic makes correctness the dominant reward component: an unformatted correct answer scores 0.8, while a formatted wrong answer scores 0.2. This illustrates why formatting rewards are useful but should not be allowed to dominate outcome correctness.",
  },
  // Replaces retired ID cme295-2026-lect4-q05; new reasoning task.
  {
    id: "cme295-2026-lect4-q051",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "A policy is optimized against an automatic judge. On a fixed task set, mean judge reward rises from 8 to 9, while independently verified task success falls from 50% to 40%. Which conclusion is supported by these measurements?",
    options: [
      {
        text: "The judge reward gain implies a ten-percentage-point improvement in independently verified task success.",
        isCorrect: false,
      },
      {
        text: "The policy improved the optimized proxy while regressing on the independent task-success measure.",
        isCorrect: true,
      },
      {
        text: "The policy improved both measures because a one-point reward gain offsets a ten-percentage-point success drop.",
        isCorrect: false,
      },
      {
        text: "The task-success drop means the judge reward must also have fallen on this fixed task set.",
        isCorrect: false,
      },
    ],
    explanation:
      "The two measurements moved in opposite directions: higher judge reward accompanied lower verified success. This is evidence that optimizing the judge can exploit a proxy that does not track the intended task outcome. The scales cannot be exchanged by treating one reward point as a success percentage, and neither metric's movement forces the other to move the same way.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q06",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "A training method defines an objective \\(J(\\theta)\\) to maximize and an equivalent loss \\(L(\\theta)=-J(\\theta)\\) to minimize. Which update conventions are correct?",
    options: [
      {
        text: "Gradient descent on \\(J\\): \\(\\theta\\leftarrow\\theta-\\alpha\\nabla J\\).",
        isCorrect: false,
      },
      {
        text: "Gradient descent: \\(\\theta\\leftarrow\\theta-\\alpha\\nabla L\\).",
        isCorrect: true,
      },
      {
        text: "Gradient ascent on \\(L\\): \\(\\theta\\leftarrow\\theta+\\alpha\\nabla L\\).",
        isCorrect: false,
      },
      {
        text: "Gradient ascent: \\(\\theta\\leftarrow\\theta+\\alpha\\nabla J\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "Negating the maximization objective reverses its gradient, so descent on the negative loss matches ascent on the objective. The name of a symbol in a paper does not determine the update direction; descending on J or ascending on its negative optimizes the opposite behavior.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q07",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Let \\(J(\\theta)=\\sum_y\\pi_\\theta(y)R(y)\\), where the reward has no direct dependence on \\(\\theta\\). Which gradient identity is correct?",
    options: [
      {
        text: "\\(\\nabla J=\\mathbb{E}_{y\\sim\\pi_\\theta}[R(y)/\\nabla\\log\\pi_\\theta(y)]\\).",
        isCorrect: false,
      },
      {
        text: "\\(\\nabla J=\\mathbb{E}_{y\\sim\\pi_\\theta}[R(y)\\nabla\\log\\pi_\\theta(y)]\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\nabla J=\\nabla\\log\\mathbb{E}_{y\\sim\\pi_\\theta}[R(y)]\\).",
        isCorrect: false,
      },
      {
        text: "\\(\\nabla J=\\mathbb{E}_{y\\sim\\pi_\\theta}[\\nabla R(y)]\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Differentiate the policy probabilities in the sum and use the log-derivative identity: the gradient of a probability equals that probability times the gradient of its logarithm. This expresses the gradient as a reward-weighted expectation without requiring the discrete reward function itself to be differentiable.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q08",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "For a positive differentiable policy probability \\(\\pi_\\theta(y)\\), which identities explain the policy-gradient log trick?",
    options: [
      {
        text: "\\(\\nabla\\log\\pi_\\theta(y)=\\pi_\\theta(y)\\nabla\\pi_\\theta(y)\\).",
        isCorrect: false,
      },
      {
        text: "\\(\\sum_y\\nabla\\pi_\\theta(y)=0\\) for a normalized policy.",
        isCorrect: true,
      },
      {
        text: "\\(\\nabla\\log\\pi_\\theta(y)=\\nabla\\pi_\\theta(y)/\\pi_\\theta(y)\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\nabla\\pi_\\theta(y)=\\pi_\\theta(y)\\nabla\\log\\pi_\\theta(y)\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The derivative of a logarithm divides the probability gradient by the probability, and rearranging gives the score-function identity. Summing policy gradients differentiates the constant total mass of one, giving zero; multiplying instead of dividing does not implement the log derivative.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q09",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "A response probability factors as \\(\\pi_\\theta(y\\mid x)=\\prod_{t=1}^T\\pi_\\theta(y_t\\mid x,y_{<t})\\). Which statements follow?",
    options: [
      {
        text: "The gradient of the log response probability is the product of token log-probability gradients.",
        isCorrect: false,
      },
      {
        text: "The response log-probability gradient is the sum of token log-probability gradients.",
        isCorrect: true,
      },
      {
        text: "A completion-level reward can multiply this sum in REINFORCE.",
        isCorrect: true,
      },
      {
        text: "\\(\\log\\pi_\\theta(y\\mid x)=\\sum_t\\log\\pi_\\theta(y_t\\mid x,y_{<t})\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The logarithm converts a product of conditional probabilities into a sum, so differentiation also sums the token contributions. Using one terminal reward with this gradient applies the same outcome signal across the trajectory, rather than discovering a separate verified reward for every token.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q10",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "A one-parameter REINFORCE sample has reward \\(R=3\\), score gradient \\(\\nabla_\\theta\\log\\pi_\\theta(y)=0.2\\), and learning rate \\(\\alpha=0.1\\). Starting at \\(\\theta=1\\), what is the ascent update?",
    options: [
      {
        text: "\\(\\theta_{\\rm new}=1.06\\)",
        isCorrect: true,
      },
      {
        text: "\\(\\theta_{\\rm new}=1.006\\)",
        isCorrect: false,
      },
      {
        text: "\\(\\theta_{\\rm new}=1.6\\)",
        isCorrect: false,
      },
      {
        text: "\\(\\theta_{\\rm new}=0.94\\)",
        isCorrect: false,
      },
    ],
    explanation:
      "The sample gradient is 3 times 0.2, which is 0.6, and the learning-rate-scaled step is 0.06. Adding that step gives 1.06; subtracting it would be descent on the reward objective, while the other results misapply the learning-rate factor.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q11",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Two equally likely responses have rewards 101 and 100, with score gradients \\(+0.5\\) and \\(-0.5\\). Which calculations describe vanilla REINFORCE?",
    options: [
      {
        text: "The two sample gradients are 50.5 and -50.",
        isCorrect: true,
      },
      {
        text: "Their expectation is 0.25.",
        isCorrect: true,
      },
      {
        text: "Both sample gradients point in the same direction because both rewards are positive.",
        isCorrect: false,
      },
      {
        text: "Their expectation is 100.5.",
        isCorrect: false,
      },
    ],
    explanation:
      "Multiplying each score gradient by its reward gives 50.5 and -50, whose equally weighted mean is 0.25. The large opposing sample values create variance even though their mean favors the slightly better response; a positive reward does not determine a parameter-gradient sign by itself.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q12",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Use two equally likely responses: rewards 101 and 100, score gradients \\(+0.5\\) and \\(-0.5\\). Subtract a constant baseline of 100.5. Which statements hold?",
    options: [
      {
        text: "The centered rewards are 0.5 and -0.5.",
        isCorrect: true,
      },
      {
        text: "The expected gradient remains 0.25.",
        isCorrect: true,
      },
      {
        text: "Both centered sample gradients equal 0.25.",
        isCorrect: true,
      },
      {
        text: "The variance of this two-outcome gradient estimator becomes zero.",
        isCorrect: true,
      },
    ],
    explanation:
      "Centering gives 0.5 times 0.5 and -0.5 times -0.5, both equal to 0.25. This toy baseline removes the opposing large fluctuations without changing the mean; an action-independent baseline changes variance, not the expected score-function gradient.",
  },
  // Replaces retired ID cme295-2026-lect4-q13; new reasoning task.
  {
    id: "cme295-2026-lect4-q052",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "For a discrete policy \\(\\pi_\\theta(a\\mid s)\\) with positive action probabilities, let \\(b(s)\\) be independent of the sampled action and detached from differentiation. Which identity explains why subtracting this baseline leaves the expected score-function policy gradient unchanged?",
    options: [
      {
        text: "\\(\\mathbb E_a[b(s)\\nabla_\\theta\\log\\pi_\\theta(a\\mid s)]=b(s)\\mathbb E_a[\\nabla_\\theta\\pi_\\theta(a\\mid s)]=0\\).",
        isCorrect: false,
      },
      {
        text: "\\(\\mathbb E_a[b(s)\\nabla_\\theta\\log\\pi_\\theta(a\\mid s)]=b(s)\\sum_a\\nabla_\\theta\\log\\pi_\\theta(a\\mid s)=0\\).",
        isCorrect: false,
      },
      {
        text: "\\(\\mathbb E_a[b(s)\\nabla_\\theta\\log\\pi_\\theta(a\\mid s)]=b(s)\\nabla_\\theta\\sum_a\\pi_\\theta(a\\mid s)=0\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\mathbb E_a[b(s)\\nabla_\\theta\\log\\pi_\\theta(a\\mid s)]=b(s)\\nabla_\\theta\\pi_\\theta(a_{\\rm sampled}\\mid s)=0\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The expectation weights each score by its action probability, converting π∇logπ into ∇π. Summing those derivatives differentiates total probability one and gives zero. An unweighted sum of log derivatives need not vanish; replacing the log score by a probability derivative inside an already probability-weighted expectation inserts an extra factor of π. The derivative of one sampled action's probability is not generally zero either.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q14",
    chapter: 4,
    difficulty: "medium",
    prompt:
      'A sealed box containing Teddy moves to the kitchen. A response falsely says Teddy climbed out and walked there, but ends with the correct answer "kitchen". Which statements diagnose terminal-outcome supervision?',
    options: [
      {
        text: "One completion score does not identify which individual steps were sound.",
        isCorrect: true,
      },
      {
        text: "A correct final answer establishes that every preceding step was correct.",
        isCorrect: false,
      },
      {
        text: "Repeated outcome learning can improve behavior statistically without validating each rewarded trace.",
        isCorrect: true,
      },
      {
        text: "A final-answer checker can reward the response despite the false intermediate steps.",
        isCorrect: true,
      },
    ],
    explanation:
      "Outcome supervision evaluates the terminal result, so it can credit an internally wrong trajectory that arrives at the right location. It supplies no explicit step-level correctness labels; statistical improvement across many trajectories is different from certifying the reasoning in one rewarded trace.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q15",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "When collecting fresh on-policy language-model rollouts, which operation creates a sequential dependency absent from teacher-forced scoring of a known sequence?",
    options: [
      {
        text: "Sampling a token and inserting it into the prefix before choosing the next token.",
        isCorrect: true,
      },
      {
        text: "Projecting known token embeddings into queries, keys, and values in one layer.",
        isCorrect: false,
      },
      {
        text: "Evaluating a reward model on an already completed response.",
        isCorrect: false,
      },
      {
        text: "Computing a masked next-token loss on a fully supplied target sequence.",
        isCorrect: false,
      },
    ],
    explanation:
      "A rollout must construct the learner trajectory through its own sampled choices, and each choice conditions the next distribution. Known-sequence projections, scoring, and teacher-forced losses can process already available data; they do not require deciding the missing next prefix token.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q16",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements distinguish a reward from an advantage in Proximal Policy Optimization (PPO)?",
    options: [
      {
        text: "A positive raw reward implies positive advantage regardless of the baseline.",
        isCorrect: false,
      },
      {
        text: "Reward evaluates an observed outcome or behavior.",
        isCorrect: true,
      },
      {
        text: "Advantage compares performance with a baseline expectation.",
        isCorrect: true,
      },
      {
        text: "The value function and frozen reward model are the same object by definition.",
        isCorrect: false,
      },
    ],
    explanation:
      "An outcome can have positive reward yet be worse than expected, giving a negative advantage. A learned value function predicts expected future return from a state, while the reward model or verifier supplies an outcome signal; their roles differ even when their units are comparable.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q17",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "A simplified token advantage is the change in predicted final reward after adding the token: \\(A_t=V(s_{t+1})-V(s_t)\\). The predictions move from 0.70 to 0.95. Which statements are correct?",
    options: [
      {
        text: "The simplified advantage is +0.25.",
        isCorrect: true,
      },
      {
        text: "The token increased the predicted final reward.",
        isCorrect: true,
      },
      {
        text: "The difference is an estimate rather than a verified causal label for that token.",
        isCorrect: true,
      },
      {
        text: "The value estimates are conditioned on partial generated prefixes.",
        isCorrect: true,
      },
    ],
    explanation:
      "Subtracting 0.70 from 0.95 gives a positive change of 0.25, so the token appears better than the preceding expectation. This is the stated simplified intuition; practical advantage estimators can include temporal-difference rewards, discounting, and generalized advantage estimation rather than using this difference literally.",
  },
  // Replaces retired ID cme295-2026-lect4-q18; new reasoning task.
  {
    id: "cme295-2026-lect4-q053",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "From a particular prefix, the current policy finishes successfully with probability 0.25 and receives reward 1; otherwise it receives 0. What should an exact value model predicting expected terminal reward return, and what follows? Select all true statements.",
    options: [
      {
        text: "Its value prediction is 0 because failure is the more frequent outcome.",
        isCorrect: false,
      },
      {
        text: "Its value prediction is 0.25.",
        isCorrect: true,
      },
      {
        text: "The value is an expectation over future policy continuations, not the realized outcome of one continuation.",
        isCorrect: true,
      },
      {
        text: "A successful sampled completion can receive reward 1 even though the prefix value is 0.25.",
        isCorrect: true,
      },
    ],
    explanation:
      "The expected reward is 0.25 × 1 + 0.75 × 0 = 0.25. Individual rollouts still receive either zero or one, so a correct expected-value estimate need not equal the sampled reward. Choosing zero because it is the most common outcome predicts the mode rather than the mean used for a value baseline.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q19",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "In a PPO-Clip update, suppose the previous policy assigned probability 0.10 to a sampled token and the current policy assigns 0.13, with \\(\\epsilon=0.2\\). Which statements are correct?",
    options: [
      {
        text: "The clipping comparison is against the previous policy snapshot, not directly against the reward-model score.",
        isCorrect: true,
      },
      {
        text: "The policy ratio for that token is \\(0.13/0.10=1.3\\).",
        isCorrect: true,
      },
      {
        text: "Because 1.3 is greater than 1, the update must be accepted with no clipping.",
        isCorrect: false,
      },
      {
        text: "For a positive advantage, the clipped term would cap the useful ratio at \\(1+\\epsilon=1.2\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The ratio is current probability divided by previous probability, so the example gives 1.3. With \\(\\epsilon=0.2\\), a positive-advantage update receives no extra objective benefit above 1.2, which limits one-step policy movement.",
  },
  // Replaces retired ID cme295-2026-lect4-q20; new reasoning task.
  {
    id: "cme295-2026-lect4-q054",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "An old policy assigns probabilities \\((0.2,0.8)\\) to actions A and B; the new policy assigns \\((0.3,0.7)\\). Their fixed advantages are \\((2,-1)\\). Define \\(\\rho(a)=\\pi_{\\rm new}(a)/\\pi_{\\rm old}(a)\\), with no clipping. Select all true statements.",
    options: [
      {
        text: "\\(\\mathbb E_{\\rm old}[\\rho A]=-0.1=\\mathbb E_{\\rm new}[A]\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\rho(A)=1.5\\) and \\(\\rho(B)=0.875\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\mathbb E_{\\rm old}[A]=-0.1\\), even without importance weights.",
        isCorrect: false,
      },
      {
        text: "\\(\\mathbb E_{\\rm old}[\\rho]=1\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The ratios are 0.3/0.2 = 1.5 and 0.7/0.8 = 0.875. Weighting by the old probabilities gives a ratio mean of 0.3 + 0.7 = 1 and an advantage mean of 0.3 × 2 − 0.7 = −0.1, matching the new policy's expectation. The unweighted old-policy advantage mean is 0.2 × 2 − 0.8 = −0.4, so importance weighting changes the expectation here.",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(
    stanfordCME295Lecture5PreferenceTuningQuestions,
    "cme295-lect5-q47",
  ),
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q22",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "For PPO-Clip, use \\(\\min(\\rho A,\\operatorname{clip}(\\rho,0.8,1.2)A)\\). If \\(\\rho=0.5\\) and \\(A=-2\\), what is the surrogate value?",
    options: [
      {
        text: "\\(-1.6\\)",
        isCorrect: true,
      },
      {
        text: "\\(+1.6\\)",
        isCorrect: false,
      },
      {
        text: "\\(-1.0\\)",
        isCorrect: false,
      },
      {
        text: "\\(-2.4\\)",
        isCorrect: false,
      },
    ],
    explanation:
      "The unclipped term is -1 and the clipped term is 0.8 times -2, or -1.6. Taking the minimum chooses -1.6, producing a flat region once the probability reduction exceeds the beneficial lower bound; this removes extra incentive rather than imposing a hard parameter constraint.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q23",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "A PPO implementation clips its surrogate ratio at 1.2 for positive advantages. Which conclusions are justified?",
    options: [
      {
        text: "A flat clipped term for one sampled action prevents other loss terms from changing that action's probability.",
        isCorrect: false,
      },
      {
        text: "Further increasing that sampled action's ratio above 1.2 gives no extra benefit from its clipped term.",
        isCorrect: true,
      },
      {
        text: "The final policy is mathematically guaranteed to keep every token ratio below 1.2.",
        isCorrect: false,
      },
      {
        text: "Other samples and shared parameters can still move its probability.",
        isCorrect: true,
      },
    ],
    explanation:
      "The minimum surrogate flattens the beneficial side for a sampled action after the threshold. It is an optimization incentive, not a projection of the entire policy into exact ratio bounds; shared weights and other loss terms can still change probabilities.",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(
    stanfordCME295Lecture5PreferenceTuningQuestions,
    "cme295-lect5-q46",
  ),
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q25",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "A trainer keeps both an old-policy snapshot and a reference-policy snapshot. Which statements correctly distinguish their usual roles?",
    options: [
      {
        text: "The old policy identifies the distribution used to collect the current rollout batch.",
        isCorrect: true,
      },
      {
        text: "The old policy and the reference must be updated to the current weights after every gradient step.",
        isCorrect: false,
      },
      {
        text: "The reference can anchor behavior to the model before the RL stage.",
        isCorrect: true,
      },
      {
        text: "Ratio clipping and a reference KL penalty can be used together.",
        isCorrect: true,
      },
    ],
    explanation:
      "The old snapshot supports within-batch update control, while the reference preserves a longer-term behavioral anchor. They may start equal but serve different purposes; replacing both immediately after every update would remove the stated comparisons.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q26",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "An objective to maximize is \\(J=S-\\beta D_{\\rm KL}\\). At one evaluation, \\(S=2.4\\), \\(D_{\\rm KL}=0.2\\), and \\(\\beta=0.5\\). What is \\(J\\)?",
    options: [
      {
        text: "2.5",
        isCorrect: false,
      },
      {
        text: "2.0",
        isCorrect: false,
      },
      {
        text: "1.1",
        isCorrect: false,
      },
      {
        text: "2.3",
        isCorrect: true,
      },
    ],
    explanation:
      "The divergence penalty is 0.5 times 0.2, or 0.1, which is subtracted from the surrogate reward value 2.4. Adding the penalty would instead favor divergence, while subtracting the coefficient alone or multiplying the whole objective changes the stated formula.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q27",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "An objective to maximize is \\(J=S-\\beta D\\), where \\(S\\) is an advantage surrogate, \\(D\\) is a divergence, and \\(\\beta>0\\). Which loss should a minimizing optimizer use?",
    options: [
      {
        text: "\\(L=-S+\\beta D\\).",
        isCorrect: true,
      },
      {
        text: "\\(L=S-\\beta D\\).",
        isCorrect: false,
      },
      {
        text: "\\(L=-S-\\beta D\\).",
        isCorrect: false,
      },
      {
        text: "\\(L=S+\\beta D\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Minimizing the negative of the whole objective gives -S plus beta times D. This favors a larger surrogate and a smaller divergence; changing just one sign would either discourage the desired surrogate or reward deviation from the reference policy.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q28",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "In a common PPO alignment iteration, which model-role statements are correct?",
    options: [
      {
        text: "The already fitted reward model can remain frozen.",
        isCorrect: true,
      },
      {
        text: "The reference model can remain frozen as a behavior anchor.",
        isCorrect: true,
      },
      {
        text: "The value model is trained to predict returns.",
        isCorrect: true,
      },
      {
        text: "The policy is trainable.",
        isCorrect: true,
      },
    ],
    explanation:
      "The policy and critic learn during PPO, whereas a previously fitted reward model and a reference snapshot can supply fixed signals. These distinct roles explain both the algorithm's flexibility and its memory and implementation burden; not every model in the loop is trained simultaneously.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q29",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements explain why PPO can improve over vanilla completion-reward REINFORCE while remaining costly?",
    options: [
      {
        text: "Prefix-conditioned value estimates equal the terminal reward at every state.",
        isCorrect: false,
      },
      {
        text: "The critic stays frozen because reward-model scores already provide exact prefix expectations.",
        isCorrect: false,
      },
      {
        text: "A value baseline can reduce variance.",
        isCorrect: true,
      },
      {
        text: "Clipping or reference penalties control incentives for large changes.",
        isCorrect: true,
      },
    ],
    explanation:
      "A learned baseline can reduce gradient variance and clipping or reference penalties can limit update incentives. The critic predicts conditional expected returns, not the same terminal score at every prefix, so it must be fitted; fitting that extra estimator is a major practical cost of PPO.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q30",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "A GRPO trainer replaces PPO's learned critic with several scored completions for the same prompt. Which statements follow?",
    options: [
      {
        text: "Ratio clipping can still constrain the update incentive.",
        isCorrect: true,
      },
      {
        text: "Sibling completion rewards provide a group baseline.",
        isCorrect: true,
      },
      {
        text: "A learned value model is not required for this baseline.",
        isCorrect: true,
      },
      {
        text: "The policy still needs reward information.",
        isCorrect: true,
      },
    ],
    explanation:
      "GRPO removes the critic by estimating relative quality from a sampled group. It remains reward-driven policy optimization with rollout and update-control costs, so avoiding a value model is not equivalent to avoiding reinforcement learning.",
  },
  // Replaces retired ID cme295-2026-lect4-q31; new reasoning task.
  {
    id: "cme295-2026-lect4-q055",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Two prompts have sampled rewards A: [8, 10] and B: [0, 2]. Use reward minus the same-prompt group mean as the unnormalized advantage. Select all true comparisons with using a single mean over all four rewards.",
    options: [
      {
        text: "Using a global mean gives B's better completion advantage +1.",
        isCorrect: false,
      },
      {
        text: "The better completion in each prompt has same-prompt advantage +1.",
        isCorrect: true,
      },
      {
        text: "The global mean is 5, so B's better completion would have global-centered advantage −3.",
        isCorrect: true,
      },
      {
        text: "A's lower-reward completion has same-prompt advantage +3.",
        isCorrect: false,
      },
    ],
    explanation:
      "The prompt means are 9 and 1, giving local advantages [−1,+1] for both groups. The global mean is (8+10+0+2)/4 = 5, which instead gives [3,5,−5,−3]. A global baseline mixes prompt difficulty with within-prompt quality, whereas same-prompt comparisons recognize the relatively better B completion.",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(
    stanfordCME295Lecture6ReasoningQuestions,
    "cme295-lect6-q45",
  ),
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q33",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "A GRPO group has rewards \\([1,1,1,1]\\). Using a centered reward divided by group standard deviation, which statements diagnose this case?",
    options: [
      {
        text: "Every centered reward is zero.",
        isCorrect: true,
      },
      {
        text: "Direct division requires a numerical guard or a policy for such groups.",
        isCorrect: true,
      },
      {
        text: "The population standard deviation is zero.",
        isCorrect: true,
      },
      {
        text: "The group supplies no relative ranking among these completions.",
        isCorrect: true,
      },
    ],
    explanation:
      "Equal rewards remove both centered differences and within-group variance, so a naive formula produces zero divided by zero. Implementations can stabilize or skip such groups, but should not invent a reward ranking from output length when the verifier gave equal results.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q34",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Compare a GRPO objective with a separate KL penalty and a PPO implementation that incorporates reference pressure into rewards. Which statements are correct?",
    options: [
      {
        text: "Small current-to-old policy ratios bound total deviation from a fixed reference across arbitrarily many update batches.",
        isCorrect: false,
      },
      {
        text: "Both can compare current and old probabilities in a clipped surrogate.",
        isCorrect: true,
      },
      {
        text: "The location of the KL term in the objective or reward construction can differ.",
        isCorrect: true,
      },
      {
        text: "Both can discourage deviation from a reference policy.",
        isCorrect: true,
      },
    ],
    explanation:
      "A local old-policy ratio and a reference anchor address different timescales of policy drift. Implementations can place reference pressure inside reward shaping or outside the advantage term, so formulas should be read for their actual conventions rather than assuming that PPO never uses KL.",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(
    stanfordCME295Lecture6ReasoningQuestions,
    "cme295-lect6-q49",
  ),
  // Replaces retired ID cme295-2026-lect4-q36; new reasoning task.
  {
    id: "cme295-2026-lect4-q056",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Two completions have lengths 20 and 100. Compare per-token averaging weights in three simplified objectives: equal-completion averaging uses \\(1/(G\\,|o_i|)\\); batch-token averaging uses \\(1/\\sum_i|o_i|\\); fixed-length averaging uses \\(1/(G L)\\), with \\(G=2,L=100\\). Select all true calculations.",
    options: [
      {
        text: "Equal-completion averaging gives per-token weights \\(1/40\\) and \\(1/200\\).",
        isCorrect: true,
      },
      {
        text: "Batch-token averaging gives every token weight \\(1/120\\).",
        isCorrect: true,
      },
      {
        text: "With identical per-token losses, equal-completion averaging gives each entire completion the same total weight.",
        isCorrect: true,
      },
      {
        text: "Fixed-length averaging gives every token weight \\(1/200\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The completion-specific denominators are 2 × 20 and 2 × 100, so summing each completion's weights gives one half in both cases. Batch-token averaging uses all 120 sampled tokens, while fixed-length averaging uses 2 × 100 even though only 120 tokens were sampled. These denominators change how length affects each token's contribution and the total weight of a completion.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q37",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Two groups have binary rewards: hard prompt \\([0,0,0,1]\\), easy prompt \\([0,1,1,1]\\). Use population standard deviation in \\(A=(r-\\bar r)/\\sigma\\). Which statement is correct for successful completions?",
    options: [
      {
        text: "The hard-group success has \\(1/\\sqrt{3}\\), while an easy-group success has \\(\\sqrt{3}\\).",
        isCorrect: false,
      },
      {
        text: "Both have advantage 1 because their raw reward is 1.",
        isCorrect: false,
      },
      {
        text: "The hard-group success has advantage \\(\\sqrt{3}\\), while an easy-group success has \\(1/\\sqrt{3}\\).",
        isCorrect: true,
      },
      {
        text: "Both have advantage 0 because their groups include failures.",
        isCorrect: false,
      },
    ],
    explanation:
      "Both groups have standard deviation sqrt(3)/4, but their means are 1/4 and 3/4. Subtracting these means gives 3/4 versus 1/4 for a success, producing advantages sqrt(3) versus 1/sqrt(3); the same raw label therefore has a different normalized scale.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q38",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "A rare token has old probability \\(0.001\\). Compare upper probability-ratio limits 1.2 and 1.5, holding the old probability fixed. Which pair of upper probabilities is correct?",
    options: [
      {
        text: "\\(0.000833\\ldots\\) and \\(0.000666\\ldots\\).",
        isCorrect: false,
      },
      {
        text: "\\(0.012\\) and \\(0.015\\).",
        isCorrect: false,
      },
      {
        text: "\\(0.201\\) and \\(0.501\\).",
        isCorrect: false,
      },
      {
        text: "\\(0.0012\\) and \\(0.0015\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "A ratio limit multiplies the old probability, so 0.001 times 1.2 is 0.0012 and times 1.5 is 0.0015. This illustrates how a modest relative cap permits a tiny absolute change for a rare token, motivating separate consideration of upper and lower clipping bounds.",
  },
  // Replaces retired ID cme295-2026-lect4-q39; new reasoning task.
  {
    id: "cme295-2026-lect4-q057",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "A process-supervision project labels 1,000 traces with ten steps each. Each step label costs $0.02; an alternative labels only one outcome per trace at the same price. Assume independent step-label correctness of 0.95. Select all true calculations.",
    options: [
      {
        text: "The probability that all ten step labels on a trace are correct is 0.95 because each individual label has that accuracy.",
        isCorrect: false,
      },
      {
        text: "The probability that all ten step labels on a trace are correct is \\(0.95^{10}\\approx0.599\\).",
        isCorrect: true,
      },
      {
        text: "Outcome-only labels cost $20.",
        isCorrect: true,
      },
      {
        text: "Process labels cost $200.",
        isCorrect: true,
      },
    ],
    explanation:
      "There are 10,000 step labels at two cents each, costing $200, compared with 1,000 outcome labels costing $20. Under the stated independence assumption, every step label being correct requires ten simultaneous successes, with probability 0.95¹⁰ ≈ 0.599. Per-label accuracy therefore differs from the chance that a whole annotated reasoning path is error-free.",
  },
  // Replaces retired ID cme295-2026-lect4-q40; new reasoning task.
  {
    id: "cme295-2026-lect4-q058",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "A sampled group contains two completions with final-answer reward 1 and two with reward 0. One successful completion contains a false intermediate claim; the other does not. Advantages use only final reward and group statistics. Which statement follows about their training signals before any other penalties?",
    options: [
      {
        text: "The successful completion with the false claim receives a lower group advantage because group normalization checks its intermediate steps.",
        isCorrect: false,
      },
      {
        text: "The two successful completions receive the same group advantage, so that advantage alone does not distinguish the false intermediate claim.",
        isCorrect: true,
      },
      {
        text: "The successful completion with the false claim receives zero advantage because the group contains failures.",
        isCorrect: false,
      },
      {
        text: "The two successful completions receive opposite advantages because each group must contain equally many positive and negative successful samples.",
        isCorrect: false,
      },
    ],
    explanation:
      "Identical final rewards have identical centered and normalized advantages within the same group. The failures make both successful rewards exceed the mean, but do not supply a step-level judgment of their reasoning. A process reward or token-level teacher signal could distinguish the intermediate claim; terminal group normalization alone cannot infer it.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q41",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which arrangement makes distillation on-policy with respect to a student?",
    options: [
      {
        text: "Use human-written target traces and attach a teacher score after training.",
        isCorrect: false,
      },
      {
        text: "Sample a trajectory from the student and evaluate teacher predictions at its prefixes.",
        isCorrect: true,
      },
      {
        text: "Generate a fixed teacher dataset once and train the student on those teacher prefixes.",
        isCorrect: false,
      },
      {
        text: "Let the teacher complete each prompt while the student evaluates no generated prefix.",
        isCorrect: false,
      },
    ],
    explanation:
      "The on-policy property concerns the distribution of states visited by the learner. Student-generated prefixes expose its own mistakes to teacher guidance, whereas fixed teacher or human traces supply a different trajectory distribution even when the same distillation loss is used.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q42",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "A student next-token distribution is \\(S=(0.75,0.25)\\) and its teacher is \\(T=(0.5,0.5)\\). Which expression is the reverse KL \\(D_{\\rm KL}(S\\|T)\\)?",
    options: [
      {
        text: "\\(0.75\\ln(1.5)+0.25\\ln(0.5)\\).",
        isCorrect: true,
      },
      {
        text: "\\(0.75\\ln(2/3)+0.25\\ln(2)\\).",
        isCorrect: false,
      },
      {
        text: "\\(0.5\\ln(2/3)+0.5\\ln(2)\\).",
        isCorrect: false,
      },
      {
        text: "\\(0.5\\ln(1.5)+0.5\\ln(0.5)\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Reverse KL in this convention weights log student-to-teacher ratios by student probabilities. Weighting the inverse ratios by teacher probabilities instead gives forward KL; reversing only the ratio changes the sign without producing the required divergence.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q43",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "For \\(D_{\\rm KL}(S\\|T)=\\sum_i s_i\\ln(s_i/t_i)\\), which support statements are correct?",
    options: [
      {
        text: "A teacher-positive mode omitted by the student always contributes an infinite term.",
        isCorrect: false,
      },
      {
        text: "The summation weights each log ratio by the teacher probability.",
        isCorrect: false,
      },
      {
        text: "A positive s_i with t_i equal to zero makes the divergence infinite.",
        isCorrect: true,
      },
      {
        text: "A term tends to zero as s_i tends to zero with fixed positive t_i.",
        isCorrect: true,
      },
    ],
    explanation:
      "The limit of s log(s/t) is zero as student mass approaches zero for positive teacher mass. The reverse case has an unbounded log ratio with nonzero student weight, explaining the asymmetry: missing teacher support and placing mass outside teacher support incur different penalties.",
  },
  // Replaces retired ID cme295-2026-lect4-q44; new reasoning task.
  {
    id: "cme295-2026-lect4-q059",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "A teacher distribution over three modes is \\(T=(0.49,0.02,0.49)\\). A restricted student family can put all mass on exactly one mode: \\(S_L=(1,0,0)\\), \\(S_M=(0,1,0)\\), or \\(S_R=(0,0,1)\\). With zero-mass terms taken as zero, select all true reverse-KL statements.",
    options: [
      {
        text: "\\(D_{\\rm KL}(S_L\\|T)=\\ln(1/0.49)\\).",
        isCorrect: true,
      },
      {
        text: "\\(D_{\\rm KL}(S_M\\|T)=\\ln(50)\\).",
        isCorrect: true,
      },
      {
        text: "\\(D_{\\rm KL}(S_R\\|T)=D_{\\rm KL}(S_L\\|T)\\).",
        isCorrect: true,
      },
      {
        text: "The lowest reverse-KL value in this restricted family is achieved by either edge mode rather than the middle mode.",
        isCorrect: true,
      },
    ],
    explanation:
      "For a point-mass student, only its occupied mode contributes to the sum S_i ln(S_i/T_i), so the loss is the negative log teacher probability of that mode. The edge values are about 0.713, below the middle value ln(50) ≈ 3.912. This illustrates selection of a high-probability teacher mode under the stated capacity restriction; it does not say an unrestricted student must collapse.",
  },
  // Replaces retired ID cme295-2026-lect4-q45; new reasoning task.
  {
    id: "cme295-2026-lect4-q060",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "At two visited prefixes, a student distribution over the next two tokens is \\(S=(0.5,0.5)\\). Teacher distributions are \\(T_1=(0.99,0.01)\\) and \\(T_2=(0.6,0.4)\\). Use \\(D_{\\rm KL}(S\\|T)=\\sum_i S_i\\ln(S_i/T_i)\\). Select all true statements.",
    options: [
      {
        text: "The first-prefix divergence is approximately \\(1.614\\).",
        isCorrect: true,
      },
      {
        text: "The local teacher penalty is larger at the first prefix despite the identical student distributions.",
        isCorrect: true,
      },
      {
        text: "Identical student probabilities force identical reverse-KL penalties at the two prefixes.",
        isCorrect: false,
      },
      {
        text: "The second-prefix divergence is approximately \\(0.0204\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "At the first prefix the loss is 0.5 ln(0.5/0.99) + 0.5 ln(0.5/0.01) ≈ 1.614; at the second it is 0.5 ln(0.5/0.6) + 0.5 ln(0.5/0.4) ≈ 0.0204. The teacher conditional changes with the prefix, so identical student probabilities can receive very different feedback. The large first penalty reflects substantial student mass on a token the teacher assigns only 0.01.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q46",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements describe the supervision and costs of distribution-based on-policy distillation?",
    options: [
      {
        text: "The student trajectory still needs to be generated.",
        isCorrect: true,
      },
      {
        text: "Teacher inference consumes compute.",
        isCorrect: true,
      },
      {
        text: "A teacher can supply a target distribution at multiple student prefixes.",
        isCorrect: true,
      },
      {
        text: "A separate scalar reward model is not required for distribution matching.",
        isCorrect: true,
      },
    ],
    explanation:
      "Dense distribution guidance can replace a scalar outcome signal for the distillation objective, but it is not free. Student rollouts and teacher evaluations remain costs; avoiding reward-model fitting and value-model training changes which computations are required.",
  },
  // Replaces retired ID cme295-2026-lect4-q47; new reasoning task.
  {
    id: "cme295-2026-lect4-q061",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "For one visited prefix, a teacher assigns probabilities [0.9, 0.1] to [wrong answer, correct answer]. The student can represent any normalized distribution over those two answers and is optimized solely to minimize reverse KL to this fixed teacher. Which student distribution achieves the minimum?",
    options: [
      {
        text: "[0.9, 0.1]",
        isCorrect: true,
      },
      {
        text: "[0.5, 0.5]",
        isCorrect: false,
      },
      {
        text: "[1.0, 0.0]",
        isCorrect: false,
      },
      {
        text: "[0.1, 0.9]",
        isCorrect: false,
      },
    ],
    explanation:
      "KL divergence is nonnegative and is zero when the student matches the teacher distribution. The minimum here is therefore [0.9, 0.1], even though most teacher probability is on the wrong answer. Distillation transfers the teacher's conditional judgments; neither a uniform student nor extra concentration on one answer improves this objective over exact matching, and correctness is not supplied as an independent signal.",
  },
  // Replaces retired ID cme295-2026-lect4-q48; new reasoning task.
  {
    id: "cme295-2026-lect4-q062",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "A self-distillation student has first-token probability \\(s=\\sigma(\\theta)\\); a hint-conditioned teacher supplies a detached target \\(t=0.8\\). For the two-token reverse KL, use \\(\\partial L/\\partial\\theta=s(1-s)\\ln\\!\\frac{s(1-t)}{t(1-s)}\\). At \\(s=0.5\\), select all true statements.",
    options: [
      {
        text: "The first-token probability can move toward the hinted target even when the student does not receive the hint.",
        isCorrect: true,
      },
      {
        text: "A gradient-descent step increases the student's first-token logit.",
        isCorrect: true,
      },
      {
        text: "\\(\\partial L/\\partial\\theta=-\\tfrac14\\ln4\\approx-0.347\\).",
        isCorrect: true,
      },
      {
        text: "The teacher target is held fixed when differentiating this student update.",
        isCorrect: true,
      },
    ],
    explanation:
      "Substituting s = 0.5 and t = 0.8 gives 0.25 ln(0.25) = −0.25 ln 4. Subtracting that negative gradient increases θ and hence the student's first-token probability. Detaching the target makes this a student update toward a fixed conditional teacher distribution; the teacher's extra hint can inform supervision without appearing in the student's input.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect4-q49",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which choices fit multi-teacher on-policy distillation rather than serving-time model selection?",
    options: [
      {
        text: "Select a domain expert to score distributions on student prefixes.",
        isCorrect: true,
      },
      {
        text: "Use different specialists to develop different capabilities in one learner.",
        isCorrect: true,
      },
      {
        text: "Update the student using those teacher distributions.",
        isCorrect: true,
      },
      {
        text: "Return the selected teacher's answer without updating or training a student.",
        isCorrect: false,
      },
    ],
    explanation:
      "Multi-teacher distillation combines supervision from specialists into a trained learner. Routing requests to experts can be part of choosing supervision, but simply returning a teacher answer at inference is model selection rather than transferring that capability through student updates.",
  },
  // Replaces retired ID cme295-2026-lect4-q50; new reasoning task.
  {
    id: "cme295-2026-lect4-q063",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Match four training setups to their required signals. A uses fixed prompt/response targets; B uses student rollouts, terminal rewards, a learned value baseline, and old-policy ratios; C uses sibling student rollouts, group-relative rewards, and no learned critic; D uses teacher distributions at student-visited prefixes with no scalar task reward. Select all correct matches.",
    options: [
      {
        text: "D matches on-policy distillation.",
        isCorrect: true,
      },
      {
        text: "C matches a group relative policy optimization (GRPO) setup.",
        isCorrect: true,
      },
      {
        text: "A can use supervised fine-tuning.",
        isCorrect: true,
      },
      {
        text: "B matches a critic-based proximal policy optimization (PPO) setup.",
        isCorrect: true,
      },
    ],
    explanation:
      "The data and feedback identify the update: fixed response targets support supervised likelihood training, while B's learned value baseline and old-policy ratios describe the stated PPO setup. Same-prompt relative rewards replace a separate critic in C. D instead trains on teacher distributions at student-generated states, so its dense supervisory signal need not be a scalar task reward.",
  },
];
