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
      "A policy can start from random parameters or from a behavior learned by imitation, and that starting point changes the trajectories collected early in training. Good initialization can improve exploration and sample efficiency because the policy begins in more useful parts of state-action space. It does not by itself determine the optimal final policy; later optimization, data, rewards, and algorithmic stability still matter.",
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
      "Monte Carlo estimation approximates the expected objective by sampling trajectories and averaging their returns. With enough independent samples it gives an unbiased estimate of expected return, but finite-sample variance can be large and decreases as the number of sampled trajectories grows. Policy-gradient methods use the likelihood-ratio trick so they do not require differentiating through the environment dynamics.",
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
      "In the policy-gradient estimator, reward or return weights the score-function term, so larger positive rewards reinforce the sampled actions more strongly. Negative rewards can push probability mass away from the sampled actions, and rescaling rewards changes both gradient magnitude and variance. Reward scaling therefore affects learning dynamics and can make updates unstable or too small if handled poorly.",
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
      "The causality idea is that an action at time \\(t\\) can affect rewards from time \\(t\\) onward, but it cannot affect rewards that already happened. Using \\(G_t = \\sum_{t'=t}^T r_{t'}\\) focuses credit assignment on future rewards and removes irrelevant past-reward noise. This reduces variance without biasing the policy gradient because the omitted past rewards are independent of the current action choice.",
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
      "A baseline subtracts a reference value from returns so the gradient depends on whether an action did better or worse than expected. If the baseline does not depend on the sampled action, it changes variance but not the expected gradient, which is why an average reward or value estimate can be useful. Subtracting a valid constant baseline is not a source of bias; the danger would be using an action-dependent term incorrectly.",
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
      "For a constant or otherwise action-independent baseline \\(b\\), the expected extra term is zero: \\(\\mathbb{E}[\\nabla_\\theta \\log p_\\theta(\\tau)b] = 0\\). This follows because the score-function expectation differentiates the total probability mass \\(\\int p_\\theta(\\tau)d\\tau = 1\\), whose gradient is zero. The baseline can reduce variance, but it does not eliminate all variance because sampled trajectories, rewards, and policy choices can still vary.",
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
      "Sparse rewards give the policy-gradient estimator little information on most rollouts, so many samples may look equally bad until the agent occasionally succeeds. Dense rewards provide more frequent learning signal and can make gradient estimates less noisy and more stable. Sparse rewards therefore tend to increase variance and slow learning rather than guarantee faster convergence.",
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
      "The surrogate objective is built so automatic differentiation produces the same gradient direction as the policy-gradient estimator. It is computationally convenient because it avoids manually running many separate backward passes for each sampled action contribution. The surrogate is not itself the expected return; it is an optimization device whose gradient matches the desired update.",
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
      "Policy-gradient estimates can vary because trajectories, rewards, and sampled actions vary, so high variance makes updates noisy and slows learning. Larger batches average over more samples, and reward scaling or baselines can change the estimator variance. Deterministic policies do not make the practical variance problem disappear, especially when environments, finite data, and gradient estimators are still sources of noise.",
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
      "Policy gradients optimize rewards from the agent's own sampled rollouts, so they can in principle improve beyond the demonstrator when exploration and reward feedback support better behavior. Imitation learning instead fits demonstrations and does not require a reward signal in the basic behavior-cloning setup. Because policy gradients learn from trial-and-error interaction, they typically require online or freshly collected on-policy data.",
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
      "On-policy algorithms estimate updates using data sampled from the current policy distribution. After the policy changes, old rollouts no longer exactly match the distribution assumed by the estimator, so methods such as REINFORCE usually collect fresh data. They reduce one kind of policy mismatch by staying on-policy, but they do not eliminate every distribution-shift issue in learning or deployment.",
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
      "Importance sampling reweights samples drawn from a proposal distribution so they can estimate an expectation under a target distribution. The support condition \\(q(x) > 0\\) whenever \\(p(x) > 0\\) is necessary because a sample distribution cannot correct for events it never samples. The weights can have high variance, so importance sampling does not automatically reduce variance or guarantee stable learning.",
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
      "Off-policy policy gradients try to learn from data collected by a different policy, which can improve sample efficiency by allowing more updates per batch. Importance sampling ratios correct for the mismatch between the behavior policy and the updated policy. In practice these corrections can introduce variance, approximation, or bias issues, and baselines are still useful for variance reduction.",
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
      "Trajectory-level importance sampling multiplies per-step policy probability ratios across the whole rollout. Over long horizons that product can become extremely large or extremely small, which makes estimates numerically unstable and high variance. The product corrects for distribution mismatch; it does not make the mismatch disappear or become automatically stable for large \\(T\\).",
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
      "Reward scale directly affects policy-gradient updates because returns multiply the log-probability gradient terms. Larger or poorly normalized rewards can increase gradient magnitude and estimator variance, while very small rewards can produce weak updates. Reward scaling therefore changes practical learning behavior rather than leaving the algorithm unchanged.",
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
      "The return \\(G_t\\) usually sums rewards from time \\(t\\) onward, matching the fact that an action can only affect current and future rewards. This causal return avoids assigning credit for rewards that happened before the action was taken. It can reduce irrelevant noise, but it does not guarantee low variance because future trajectories and rewards can still vary substantially.",
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
      "REINFORCE estimates the policy gradient from rollouts sampled by the current policy, which makes it an on-policy method. Because its estimator assumes current-policy data, it does not simply reuse old-policy data without correction. It also does not require a value function; adding a critic or baseline is a later variance-reduction extension rather than the core algorithm.",
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
      "A valid baseline subtracts a reference value from returns so updates depend on relative advantage rather than raw return scale. When the baseline is action-independent, it keeps the expected policy gradient unchanged while reducing variance. It does not change the optimal policy or eliminate stochasticity; it only makes the estimator less noisy.",
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
      "Importance sampling ratios compare how likely the sampled actions are under the new policy versus the behavior policy that generated the data. Those ratios correct the estimator for policy mismatch in off-policy learning. They are not guaranteed to stay close to 1, and large deviations are exactly why off-policy policy-gradient updates can become unstable.",
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
      "Policy-gradient variance is a major practical challenge because noisy estimates can point updates in unreliable directions. Larger batches, better baselines, reward normalization, and denser feedback can reduce variance, while smaller batches usually make estimates noisier. Continuous action spaces do not make variance disappear; they often make careful estimator design more important.",
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
      "Off-policy methods can improve sample efficiency by reusing a batch for more than one update or by learning from data generated by older policies. That reuse creates policy-distribution mismatch, so corrections such as importance sampling or constraints are needed. The extra reuse does not make the method automatically more stable, assumption-free, or able to avoid correction terms.",
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
      "The intuition of policy gradients is to increase the probability of sampled actions that led to high return and decrease the probability of actions associated with poor return. This is different from copying expert demonstrations, because the signal comes from reward on the agent's own behavior. It also does not train a reward predictor or optimize the environment dynamics; it updates the policy.",
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
      "Policy-gradient methods estimate gradients from sampled trajectories because the expected-return objective is an expectation over possible rollouts. The likelihood-ratio form avoids differentiating through the environment transition function, which is crucial when the dynamics are unknown or nondifferentiable. The resulting estimate is stochastic rather than a deterministic exact analytic gradient.",
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
      "Policy gradients provide the actor update used by actor-critic methods, while the critic supplies value estimates to reduce variance and improve data use. Plain policy gradients are not low-variance by default, which is one reason actor-critic methods are introduced. They still need exploration and are commonly used for continuous-control problems rather than being unsuitable for them.",
  },
];
