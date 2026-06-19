import { Question } from "../../quiz";

export const cs224rLecture3PolicyGradientsQuestions: Question[] = [
  // ============================================================
  // Lecture 3 – Policy Gradients
  // 35 questions
  // ============================================================

  // ============================================================
  // Source-backed math replacements for the original opening block
  // ============================================================

  {
    id: "cs224r-lect3-q36",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the policy-gradient objective and its Monte Carlo estimate?",
    options: [
      {
        text: "The optimization target can be written as \\(\\theta^* = \\arg\\max_\\theta \\mathbb{E}_{\\tau \\sim p_\\theta(\\tau)}[\\sum_t r(s_t, a_t)]\\).",
        isCorrect: true,
      },
      {
        text: "\\(J(\\theta)\\) is an expectation over full trajectories induced by the current policy and environment.",
        isCorrect: true,
      },
      {
        text: "A batch estimate uses sampled rollouts, for example \\(\\hat J(\\theta) = \\frac{1}{N}\\sum_i\\sum_t r(s_{i,t}, a_{i,t})\\).",
        isCorrect: true,
      },
      {
        text: "The sampled trajectories must be interpreted under the policy distribution that generated them.",
        isCorrect: true,
      },
    ],
    explanation:
      "The slide objective is an expectation over trajectories, not over isolated state-action pairs. The Monte Carlo approximation replaces that expectation with an average over sampled rollouts, which is why policy-gradient methods are sensitive to the data-collection policy.",
  },

  {
    id: "cs224r-lect3-q37",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements correctly use the log-derivative trick?",
    options: [
      {
        text: "\\(\\nabla_\\theta p_\\theta(x) = p_\\theta(x)\\nabla_\\theta \\log p_\\theta(x)\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\nabla_\\theta \\log p_\\theta(x) = p_\\theta(x)\\nabla_\\theta p_\\theta(x)\\), so no probability ratio is needed.",
        isCorrect: false,
      },
      {
        text: "Applying the identity gives \\(\\nabla_\\theta J(\\theta)=\\mathbb{E}_{\\tau \\sim p_\\theta}[\\nabla_\\theta \\log p_\\theta(\\tau)r(\\tau)]\\).",
        isCorrect: true,
      },
      {
        text: "The trick works by differentiating through each sampled transition \\(s_{t+1}\\sim p(\\cdot\\mid s_t,a_t)\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The log-derivative identity rewrites a probability gradient as a score-function term. That is what makes a Monte Carlo policy-gradient estimator possible without backpropagating through the random sampling process or through unknown environment dynamics.",
  },

  {
    id: "cs224r-lect3-q38",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe why the trajectory-score term only contains policy log-probabilities?",
    options: [
      {
        text: "\\(p_\\theta(\\tau)=p(s_1)\\prod_t\\pi_\\theta(a_t\\mid s_t)p(s_{t+1}\\mid s_t,a_t)\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\log p_\\theta(\\tau)\\) decomposes into an initial-state term, policy terms, and transition-dynamics terms.",
        isCorrect: true,
      },
      {
        text: "\\(p(s_1)\\) and \\(p(s_{t+1}\\mid s_t,a_t)\\) drop out of \\(\\nabla_\\theta \\log p_\\theta(\\tau)\\) because they do not depend on \\(\\theta\\).",
        isCorrect: true,
      },
      {
        text: "The remaining score is \\(\\sum_t \\nabla_\\theta\\log\\pi_\\theta(a_t\\mid s_t)\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The derivation expands the trajectory probability, takes a log, and differentiates with respect to the policy parameters. Initial-state and dynamics terms are still part of the probability model, but their gradients vanish when the environment is not parameterized by the policy.",
  },

  {
    id: "cs224r-lect3-q39",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements correctly describe the vanilla REINFORCE update?",
    options: [
      {
        text: "\\(\\nabla_\\theta J(\\theta) \\approx \\frac{1}{N}\\sum_i(\\sum_t\\nabla_\\theta\\log\\pi_\\theta(a_{i,t}\\mid s_{i,t}))(\\sum_t r(s_{i,t},a_{i,t}))\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\theta \\leftarrow \\theta + \\alpha \\nabla_\\theta J(\\theta)\\) is a gradient-ascent step for maximizing expected return.",
        isCorrect: true,
      },
      {
        text: "\\(\\nabla_\\theta J(\\theta) \\approx \\frac{1}{N}\\sum_i\\sum_t r(s_{i,t},a_{i,t}) - \\sum_t\\nabla_\\theta\\log\\pi_\\theta(a_{i,t}\\mid s_{i,t})\\).",
        isCorrect: false,
      },
      {
        text: "\\(\\theta \\leftarrow \\theta - \\alpha \\nabla_\\theta J(\\theta)\\) is the slide's reward-maximizing policy-gradient update.",
        isCorrect: false,
      },
    ],
    explanation:
      "REINFORCE multiplies the trajectory return by the score of the actions that occurred, then moves parameters in the ascent direction. The reward term is a weight on the log-probability gradient, not a separate subtraction from it.",
  },

  {
    id: "cs224r-lect3-q40",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the causality modification to the policy-gradient estimator?",
    options: [
      {
        text: "The return paired with an action at time \\(t\\) can be replaced by \\(G_t=\\sum_{t'=t}^{T}r(s_{t'},a_{t'})\\).",
        isCorrect: true,
      },
      {
        text: "The score term becomes a sum of per-time contributions, each weighted by rewards from that time onward.",
        isCorrect: true,
      },
      {
        text: "Past rewards are omitted because an action cannot causally influence rewards that happened before it.",
        isCorrect: true,
      },
      {
        text: "The modification reduces variance while preserving the expected policy gradient.",
        isCorrect: true,
      },
    ],
    explanation:
      "The causality trick is not a change to the objective; it is a variance-reduction move in the estimator. It keeps only rewards that could be downstream of the action whose log-probability gradient is being weighted.",
  },

  {
    id: "cs224r-lect3-q41",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly explain why subtracting a constant baseline is unbiased?",
    options: [
      {
        text: "The baseline contribution is \\(\\mathbb{E}_{\\tau}[\\nabla_\\theta\\log p_\\theta(\\tau)b]\\).",
        isCorrect: true,
      },
      {
        text: "For constant \\(b\\), this equals \\(b\\nabla_\\theta\\int p_\\theta(\\tau)d\\tau\\).",
        isCorrect: true,
      },
      {
        text: "Since \\(\\int p_\\theta(\\tau)d\\tau=1\\), the baseline term has expectation zero.",
        isCorrect: true,
      },
      {
        text: "The baseline can change estimator variance without changing the expected gradient.",
        isCorrect: true,
      },
    ],
    explanation:
      "The slide proof factors a constant baseline out of the expectation and turns the remaining score expectation into the gradient of total probability mass. Normalized probability mass is always one, so its gradient is zero.",
  },

  {
    id: "cs224r-lect3-q42",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly diagnose the reward-scale problem in the humanoid example?",
    options: [
      {
        text: "If all sampled rewards are positive, the update can increase probabilities of both falling and stumbling trajectories.",
        isCorrect: true,
      },
      {
        text: "Negative trajectory rewards would push the corresponding sampled actions in the opposite direction.",
        isCorrect: true,
      },
      {
        text: "Large reward magnitudes can make the estimator sensitive to reward scale and batch composition.",
        isCorrect: true,
      },
      {
        text: "Dense reward signals and larger batches help policy gradients learn more reliably.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture's humanoid example shows why raw reward weighting can be noisy: positive but bad rollouts may still be reinforced relative to the policy that produced them. Baselines, reward design, and batch size are practical tools for controlling this variance.",
  },

  {
    id: "cs224r-lect3-q43",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the surrogate objective used to implement policy gradients?",
    options: [
      {
        text: "With a fixed sampled batch, a weighted log-likelihood surrogate can have gradient \\(\\frac{1}{N}\\sum_i\\sum_t\\nabla_\\theta\\log\\pi_\\theta(a_{i,t}\\mid s_{i,t})R_i\\).",
        isCorrect: true,
      },
      {
        text: "The surrogate is exactly equal to \\(J(\\theta)=\\mathbb{E}_{\\tau\\sim p_\\theta}[r(\\tau)]\\) for every \\(\\theta\\) after the batch has been collected.",
        isCorrect: false,
      },
      {
        text: "For a discrete-action policy, the implementation can look like reward-weighted cross entropy on the sampled actions.",
        isCorrect: true,
      },
      {
        text: "The surrogate removes reward weights, so \\(\\nabla_\\theta\\log\\pi_\\theta(a_t\\mid s_t)\\) is averaged uniformly over actions.",
        isCorrect: false,
      },
    ],
    explanation:
      "The surrogate is a computational device: automatic differentiation sees a supervised-looking loss whose gradient matches the policy-gradient estimate for the fixed data. It does not become the true expected return as a function value.",
  },

  {
    id: "cs224r-lect3-q44",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe importance sampling for off-policy policy gradients?",
    options: [
      {
        text: "The identity \\(\\mathbb{E}_{x\\sim p}[f(x)] = \\mathbb{E}_{x\\sim q}[\\frac{p(x)}{q(x)}f(x)]\\) reweights samples from \\(q\\) to estimate an expectation under \\(p\\).",
        isCorrect: true,
      },
      {
        text: "The support condition requires \\(q(x)>0\\) wherever \\(p(x)>0\\), otherwise the ratio can be undefined for events that matter under \\(p\\).",
        isCorrect: true,
      },
      {
        text: "For trajectories from an old policy in the same environment, the policy-ratio correction contains \\(\\prod_t \\frac{\\pi_\\theta(a_t\\mid s_t)}{\\pi_{old}(a_t\\mid s_t)}\\).",
        isCorrect: true,
      },
      {
        text: "Because \\(\\prod_t \\frac{\\pi_\\theta(a_t\\mid s_t)}{\\pi_{old}(a_t\\mid s_t)}\\) multiplies many terms, it is guaranteed to stay near one for long horizons.",
        isCorrect: false,
      },
    ],
    explanation:
      "Importance sampling is the mathematical bridge between old-policy data and a new-policy gradient estimate. The same product that corrects distribution mismatch can explode or vanish over long horizons, motivating per-timestep approximations and policy-change constraints.",
  },

  // ============================================================
  // Q10–Q18: EXACTLY THREE TRUE
  // ============================================================

  {
    id: "cs224r-lect3-q10",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements about policy initialization are correct?",
    options: [
      { text: "Policies can be initialized randomly.", isCorrect: true },
      {
        text: "Policies can be initialized via imitation learning.",
        isCorrect: true,
      },
      { text: "Initialization affects early exploration.", isCorrect: true },
      {
        text: "Initialization determines the optimal final policy.",
        isCorrect: false,
      },
    ],
    explanation:
      'Initialization influences early learning dynamics and exploration but does not uniquely determine the final optimum if learning proceeds correctly. To reason through the choices, select the statements that match the criterion in the prompt: "Policies can be initialized randomly."; "Policies can be initialized via imitation learning."; "Initialization affects early exploration.". Do not select statements that miss that criterion: "Initialization determines the optimal final policy.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect3-q11",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about Monte Carlo estimation of \\(J(\\theta)\\) are correct?",
    options: [
      {
        text: "It approximates expectations using sampled trajectories.",
        isCorrect: true,
      },
      {
        text: "Variance decreases as the number of samples increases.",
        isCorrect: true,
      },
      {
        text: "It requires differentiating through the environment.",
        isCorrect: false,
      },
      {
        text: "It provides an unbiased estimate of expected return.",
        isCorrect: true,
      },
    ],
    explanation:
      'Monte Carlo estimates rely on sampling and averaging. They are unbiased but can suffer from high variance with small sample sizes. To reason through the choices, select the statements that match the criterion in the prompt: "It approximates expectations using sampled trajectories."; "Variance decreases as the number of samples increases."; "It provides an unbiased estimate of expected return.". Do not select statements that miss that criterion: "It requires differentiating through the environment.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect3-q12",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about reward weighting in policy gradients are correct?",
    options: [
      { text: "Higher rewards scale gradients upward.", isCorrect: true },
      {
        text: "Negative rewards can reverse gradient direction.",
        isCorrect: true,
      },
      { text: "Reward scaling affects variance.", isCorrect: true },
      {
        text: "Reward scaling does not affect learning dynamics.",
        isCorrect: false,
      },
    ],
    explanation:
      'Rewards directly weight gradient contributions, affecting both direction and variance. Poorly scaled rewards can destabilize learning. To reason through the choices, select the statements that match the criterion in the prompt: "Higher rewards scale gradients upward."; "Negative rewards can reverse gradient direction."; "Reward scaling affects variance.". Do not select statements that miss that criterion: "Reward scaling does not affect learning dynamics.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect3-q13",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements about causality in policy gradients are correct?",
    options: [
      { text: "Actions cannot influence past rewards.", isCorrect: true },
      {
        text: "Using future rewards reduces gradient variance.",
        isCorrect: true,
      },
      {
        text: "Returns are often defined as \\(G_t = \\sum_{t'=t}^T r_{t'}\\).",
        isCorrect: true,
      },
      {
        text: "Causality introduces bias into the gradient.",
        isCorrect: false,
      },
    ],
    explanation:
      'Restricting credit assignment to future rewards respects causality and reduces variance without introducing bias. To reason through the choices, select the statements that match the criterion in the prompt: "Actions cannot influence past rewards."; "Using future rewards reduces gradient variance."; "Returns are often defined as \\(G_t = \\sum_{t\'=t}^T r_{t\'}\\).". Do not select statements that miss that criterion: "Causality introduces bias into the gradient.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect3-q14",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about baselines in policy gradients are correct?",
    options: [
      {
        text: "Baselines reduce variance of gradient estimates.",
        isCorrect: true,
      },
      { text: "Baselines must not depend on the action.", isCorrect: true },
      {
        text: "Subtracting a constant baseline biases the gradient.",
        isCorrect: false,
      },
      { text: "Average reward is a common baseline choice.", isCorrect: true },
    ],
    explanation:
      'A baseline independent of actions preserves unbiasedness while reducing variance. Average reward is a simple and effective choice. To reason through the choices, select the statements that match the criterion in the prompt: "Baselines reduce variance of gradient estimates."; "Baselines must not depend on the action."; "Average reward is a common baseline choice.". Do not select statements that miss that criterion: "Subtracting a constant baseline biases the gradient.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect3-q15",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements about the unbiasedness of baselines are correct?",
    options: [
      {
        text: "\\(\\mathbb{E}[\\nabla_\\theta \\log p_\\theta(\\tau) b] = 0\\).",
        isCorrect: true,
      },
      {
        text: "This follows from \\(\\int p_\\theta(\\tau) d\\tau = 1\\).",
        isCorrect: true,
      },
      { text: "The gradient of a constant is zero.", isCorrect: true },
      { text: "Baselines eliminate all gradient variance.", isCorrect: false },
    ],
    explanation:
      'Subtracting a constant baseline leaves the expected gradient unchanged. While variance is reduced, it is not eliminated entirely. To reason through the choices, select the statements that match the criterion in the prompt: "\\(\\mathbb{E}[\\nabla_\\theta \\log p_\\theta(\\tau) b] = 0\\)."; "This follows from \\(\\int p_\\theta(\\tau) d\\tau = 1\\)."; "The gradient of a constant is zero.". Do not select statements that miss that criterion: "Baselines eliminate all gradient variance.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect3-q16",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements about reward sparsity are correct?",
    options: [
      { text: "Sparse rewards increase gradient variance.", isCorrect: true },
      { text: "Dense rewards provide more learning signal.", isCorrect: true },
      {
        text: "Policy gradients work best with dense rewards.",
        isCorrect: true,
      },
      {
        text: "Sparse rewards guarantee faster convergence.",
        isCorrect: false,
      },
    ],
    explanation:
      'Sparse rewards provide little feedback, making gradient estimates noisy. Dense rewards improve learning stability. To reason through the choices, select the statements that match the criterion in the prompt: "Sparse rewards increase gradient variance."; "Dense rewards provide more learning signal."; "Policy gradients work best with dense rewards.". Do not select statements that miss that criterion: "Sparse rewards guarantee faster convergence.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect3-q17",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements about the surrogate objective are correct?",
    options: [
      {
        text: "It has the same gradient as the true objective.",
        isCorrect: true,
      },
      {
        text: "It enables efficient automatic differentiation.",
        isCorrect: true,
      },
      {
        text: "It avoids computing many separate backward passes.",
        isCorrect: true,
      },
      { text: "It exactly equals the expected return.", isCorrect: false },
    ],
    explanation:
      "The surrogate objective is designed so its gradient matches the policy gradient. It is not equal to the true objective but is computationally convenient.",
  },

  {
    id: "cs224r-lect3-q18",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about gradient variance are correct?",
    options: [
      { text: "Variance decreases with larger batch sizes.", isCorrect: true },
      { text: "Variance depends on reward scale.", isCorrect: true },
      {
        text: "Variance is zero for deterministic policies.",
        isCorrect: false,
      },
      { text: "Variance affects learning speed.", isCorrect: true },
    ],
    explanation:
      'High variance slows convergence and destabilizes learning. Larger batches and better baselines reduce variance. To reason through the choices, select the statements that match the criterion in the prompt: "Variance decreases with larger batch sizes."; "Variance depends on reward scale."; "Variance affects learning speed.". Do not select statements that miss that criterion: "Variance is zero for deterministic policies.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  // ============================================================
  // Q19–Q27: EXACTLY TWO TRUE
  // ============================================================

  {
    id: "cs224r-lect3-q19",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements about imitation learning versus policy gradients are correct?",
    options: [
      {
        text: "Policy gradients can outperform demonstrators.",
        isCorrect: true,
      },
      { text: "Imitation learning uses reward signals.", isCorrect: false },
      { text: "Policy gradients require online interaction.", isCorrect: true },
      {
        text: "Imitation learning improves through trial and error.",
        isCorrect: false,
      },
    ],
    explanation:
      'Policy gradients improve by interacting with the environment, while imitation learning simply copies demonstrations. To reason through the choices, select the statements that match the criterion in the prompt: "Policy gradients can outperform demonstrators."; "Policy gradients require online interaction.". Do not select statements that miss that criterion: "Imitation learning uses reward signals."; "Imitation learning improves through trial and error.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect3-q20",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about on-policy algorithms are correct?",
    options: [
      { text: "They require fresh data after each update.", isCorrect: true },
      { text: "They can reuse old data indefinitely.", isCorrect: false },
      { text: "REINFORCE is an on-policy algorithm.", isCorrect: true },
      { text: "They eliminate distribution shift entirely.", isCorrect: false },
    ],
    explanation:
      'On-policy methods rely on samples from the current policy. Old data becomes invalid once the policy changes. To reason through the choices, select the statements that match the criterion in the prompt: "They require fresh data after each update."; "REINFORCE is an on-policy algorithm.". Do not select statements that miss that criterion: "They can reuse old data indefinitely."; "They eliminate distribution shift entirely.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect3-q21",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements about importance sampling are correct?",
    options: [
      {
        text: "It reweights samples from a proposal distribution.",
        isCorrect: true,
      },
      {
        text: "It requires \\(q(x) > 0\\) whenever \\(p(x) > 0\\).",
        isCorrect: true,
      },
      { text: "It always reduces variance.", isCorrect: false },
      { text: "It guarantees stable learning.", isCorrect: false },
    ],
    explanation:
      'Importance sampling corrects for distribution mismatch but can increase variance if weights explode or vanish. To reason through the choices, select the statements that match the criterion in the prompt: "It reweights samples from a proposal distribution."; "It requires \\(q(x) > 0\\) whenever \\(p(x) > 0\\).". Do not select statements that miss that criterion: "It always reduces variance."; "It guarantees stable learning.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect3-q22",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about off-policy policy gradients are correct?",
    options: [
      {
        text: "They allow multiple gradient steps per batch.",
        isCorrect: true,
      },
      { text: "They require importance sampling ratios.", isCorrect: true },
      { text: "They are always unbiased in practice.", isCorrect: false },
      { text: "They remove the need for baselines.", isCorrect: false },
    ],
    explanation:
      'Off-policy gradients trade bias and variance to improve sample efficiency. Importance sampling is required to correct for policy mismatch. To reason through the choices, select the statements that match the criterion in the prompt: "They allow multiple gradient steps per batch."; "They require importance sampling ratios.". Do not select statements that miss that criterion: "They are always unbiased in practice."; "They remove the need for baselines.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect3-q23",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about trajectory-level importance sampling are correct?",
    options: [
      {
        text: "It involves a product of per-step probability ratios.",
        isCorrect: true,
      },
      { text: "It can explode or vanish for long horizons.", isCorrect: true },
      { text: "It is numerically stable for large \\(T\\).", isCorrect: false },
      { text: "It avoids distribution mismatch entirely.", isCorrect: false },
    ],
    explanation:
      'Multiplying many probability ratios leads to numerical instability. This motivates per-timestep approximations. To reason through the choices, select the statements that match the criterion in the prompt: "It involves a product of per-step probability ratios."; "It can explode or vanish for long horizons.". Do not select statements that miss that criterion: "It is numerically stable for large \\(T\\)."; "It avoids distribution mismatch entirely.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect3-q24",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about per-timestep importance sampling are correct?",
    options: [
      {
        text: "It approximates trajectory-level importance sampling.",
        isCorrect: true,
      },
      {
        text: "It reduces variance compared to full products.",
        isCorrect: true,
      },
      { text: "It is always theoretically exact.", isCorrect: false },
      { text: "It is commonly used in practice.", isCorrect: true },
    ],
    explanation:
      "Per-timestep importance sampling approximates the trajectory-level correction because the exact product over a long horizon can explode or vanish. The lecture presents this approximation as the practical form used despite the theoretical approximation it introduces.",
  },

  {
    id: "cs224r-lect3-q25",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements about reward scaling are correct?",
    options: [
      { text: "Reward scaling affects gradient magnitude.", isCorrect: true },
      { text: "Reward scaling affects variance.", isCorrect: true },
      { text: "Reward scaling leaves learning unchanged.", isCorrect: false },
      { text: "Reward scaling has no practical impact.", isCorrect: false },
    ],
    explanation:
      'The scale of rewards directly impacts gradient updates and stability. To reason through the choices, select the statements that match the criterion in the prompt: "Reward scaling affects gradient magnitude."; "Reward scaling affects variance.". Do not select statements that miss that criterion: "Reward scaling leaves learning unchanged."; "Reward scaling has no practical impact.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect3-q26",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements about gradient estimation noise are correct?",
    options: [
      { text: "Noise arises from stochastic policies.", isCorrect: true },
      { text: "Noise arises from stochastic environments.", isCorrect: true },
      { text: "Noise disappears with baselines.", isCorrect: false },
      { text: "Noise makes convergence slower.", isCorrect: true },
    ],
    explanation:
      "Multiple sources contribute to noisy policy-gradient estimates, including stochastic policies, stochastic environments, and finite batches. Baselines reduce but do not eliminate the noise, and high-variance estimates slow convergence because updates point in unreliable directions.",
  },

  {
    id: "cs224r-lect3-q27",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about returns \\(G_t\\) are correct?",
    options: [
      {
        text: "\\(G_t\\) sums rewards from time \\(t\\) onward.",
        isCorrect: true,
      },
      { text: "\\(G_t\\) respects causality.", isCorrect: true },
      { text: "\\(G_t\\) includes past rewards.", isCorrect: false },
      { text: "\\(G_t\\) guarantees low variance.", isCorrect: false },
    ],
    explanation:
      'Returns focus credit assignment on future rewards, aligning with causal influence. To reason through the choices, select the statements that match the criterion in the prompt: "\\(G_t\\) sums rewards from time \\(t\\) onward."; "\\(G_t\\) respects causality.". Do not select statements that miss that criterion: "\\(G_t\\) includes past rewards."; "\\(G_t\\) guarantees low variance.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  // ============================================================
  // Q28–Q35: EXACTLY ONE TRUE
  // ============================================================

  {
    id: "cs224r-lect3-q28",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statement about REINFORCE is correct?",
    options: [
      { text: "It is an off-policy algorithm.", isCorrect: false },
      { text: "It reuses data from old policies.", isCorrect: false },
      {
        text: "It estimates gradients from on-policy rollouts.",
        isCorrect: true,
      },
      { text: "It requires a value function.", isCorrect: false },
    ],
    explanation:
      'REINFORCE relies exclusively on samples from the current policy. To reason through the choices, select the statements that match the criterion in the prompt: "It estimates gradients from on-policy rollouts.". Do not select statements that miss that criterion: "It is an off-policy algorithm."; "It reuses data from old policies."; "It requires a value function.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect3-q29",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statement about baselines is correct?",
    options: [
      { text: "They bias the gradient.", isCorrect: false },
      { text: "They change the optimal policy.", isCorrect: false },
      {
        text: "They reduce variance without changing expectation.",
        isCorrect: true,
      },
      { text: "They eliminate stochasticity.", isCorrect: false },
    ],
    explanation:
      'Baselines are designed to reduce variance while keeping the expected gradient unchanged. To reason through the choices, select the statements that match the criterion in the prompt: "They reduce variance without changing expectation.". Do not select statements that miss that criterion: "They bias the gradient."; "They change the optimal policy."; "They eliminate stochasticity.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect3-q30",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statement about importance sampling ratios is correct?",
    options: [
      { text: "They are always close to 1.", isCorrect: false },
      {
        text: "They are unnecessary in off-policy learning.",
        isCorrect: false,
      },
      { text: "They correct for policy mismatch.", isCorrect: true },
      { text: "They eliminate distribution shift.", isCorrect: false },
    ],
    explanation:
      'Importance sampling ratios reweight samples to account for differences between old and new policies. To reason through the choices, select the statements that match the criterion in the prompt: "They correct for policy mismatch.". Do not select statements that miss that criterion: "They are always close to 1."; "They are unnecessary in off-policy learning."; "They eliminate distribution shift.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect3-q31",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statement about policy gradient variance is correct?",
    options: [
      {
        text: "Variance decreases with smaller batch sizes.",
        isCorrect: false,
      },
      { text: "Variance is unaffected by reward design.", isCorrect: false },
      { text: "Variance is a major practical challenge.", isCorrect: true },
      {
        text: "Variance disappears in continuous action spaces.",
        isCorrect: false,
      },
    ],
    explanation:
      'High variance is a central challenge in policy gradient methods and motivates variance-reduction techniques. To reason through the choices, select the statements that match the criterion in the prompt: "Variance is a major practical challenge.". Do not select statements that miss that criterion: "Variance decreases with smaller batch sizes."; "Variance is unaffected by reward design."; "Variance disappears in continuous action spaces.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect3-q32",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statement about off-policy policy gradients is correct?",
    options: [
      {
        text: "They are always more stable than on-policy methods.",
        isCorrect: false,
      },
      { text: "They avoid importance sampling.", isCorrect: false },
      { text: "They allow more gradient updates per batch.", isCorrect: true },
      { text: "They require no assumptions.", isCorrect: false },
    ],
    explanation:
      'Off-policy methods improve sample efficiency but introduce new stability challenges. To reason through the choices, select the statements that match the criterion in the prompt: "They allow more gradient updates per batch.". Do not select statements that miss that criterion: "They are always more stable than on-policy methods."; "They avoid importance sampling."; "They require no assumptions.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect3-q33",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statement best summarizes the intuition of policy gradients?",
    options: [
      { text: "Copy expert demonstrations.", isCorrect: false },
      { text: "Predict rewards directly.", isCorrect: false },
      {
        text: "Do more high-reward actions and less low-reward actions.",
        isCorrect: true,
      },
      { text: "Optimize environment dynamics.", isCorrect: false },
    ],
    explanation:
      'Policy gradients reinforce actions that lead to good outcomes and suppress poor ones. To reason through the choices, select the statements that match the criterion in the prompt: "Do more high-reward actions and less low-reward actions.". Do not select statements that miss that criterion: "Copy expert demonstrations."; "Predict rewards directly."; "Optimize environment dynamics.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect3-q34",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statement about gradient estimation is correct?",
    options: [
      {
        text: "Exact gradients can be computed analytically.",
        isCorrect: false,
      },
      {
        text: "Gradients require differentiating through the environment.",
        isCorrect: false,
      },
      {
        text: "Gradients are estimated using sampled trajectories.",
        isCorrect: true,
      },
      { text: "Gradients are deterministic.", isCorrect: false },
    ],
    explanation:
      'Policy gradients rely on Monte Carlo estimates from sampled rollouts. To reason through the choices, select the statements that match the criterion in the prompt: "Gradients are estimated using sampled trajectories.". Do not select statements that miss that criterion: "Exact gradients can be computed analytically."; "Gradients require differentiating through the environment."; "Gradients are deterministic.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect3-q35",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statement about policy gradient methods is correct?",
    options: [
      { text: "They are low-variance by default.", isCorrect: false },
      { text: "They eliminate the need for exploration.", isCorrect: false },
      {
        text: "They form the foundation of actor-critic methods.",
        isCorrect: true,
      },
      { text: "They are unsuitable for continuous control.", isCorrect: false },
    ],
    explanation:
      'Actor-critic methods build directly on policy gradient ideas while addressing variance and sample efficiency. To reason through the choices, select the statements that match the criterion in the prompt: "They form the foundation of actor-critic methods.". Do not select statements that miss that criterion: "They are low-variance by default."; "They eliminate the need for exploration."; "They are unsuitable for continuous control.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },
];
