import { Question } from "../../quiz";

export const lecture5_OffPolicyActorCriticQuestions: Question[] = [
  {
    id: "cs224r-lect5-q36",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements correctly set up the off-policy actor-critic problem when one batch is reused for multiple policy-gradient steps?",
    options: [
      {
        text: "Data can be collected by \\(\\pi_\\theta(a\\mid s)\\) while gradients are evaluated for a later policy \\(\\pi_{\\theta'}(a\\mid s)\\).",
        isCorrect: true,
      },
      {
        text: "The likelihood ratio \\(\\rho_{t,i}(\\theta')=\\frac{\\pi_{\\theta'}(a_{t,i}\\mid s_{t,i})}{\\pi_\\theta(a_{t,i}\\mid s_{t,i})}\\) corrects for the policy mismatch in the sampled action.",
        isCorrect: true,
      },
      {
        text: "The advantage estimate \\(\\hat A^{\\pi_\\theta}(s_{t,i},a_{t,i})\\) is still tied to the old data-collection policy \\(\\pi_\\theta\\).",
        isCorrect: true,
      },
      {
        text: "Taking too many steps on one batch can make fixed advantages \\(\\hat A^{\\pi_\\theta}\\) stale relative to \\(\\pi_{\\theta'}\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The key tension is that the data and advantages come from an older policy while the optimizer changes the current policy. Importance ratios let the same batch estimate a gradient for \\(\\theta'\\), but the correction is fragile when the policies drift too far apart.",
  },

  {
    id: "cs224r-lect5-q37",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which expressions correctly describe the importance-weighted policy-gradient estimator for reusing data from \\(\\pi_\\theta\\) while updating \\(\\pi_{\\theta'}\\)?",
    options: [
      {
        text: "\\(\\nabla_{\\theta'}J(\\theta')\\approx \\sum_{t,i}\\frac{\\pi_{\\theta'}(a_{t,i}\\mid s_{t,i})}{\\pi_{\\theta}(a_{t,i}\\mid s_{t,i})}\\nabla_{\\theta'}\\log\\pi_{\\theta'}(a_{t,i}\\mid s_{t,i})\\hat A^{\\pi_\\theta}(s_{t,i},a_{t,i})\\).",
        isCorrect: true,
      },
      {
        text: "The denominator is the behavior-policy probability of the sampled action, so it is treated as fixed when differentiating with respect to \\(\\theta'\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\nabla_{\\theta'}J(\\theta')\\approx \\sum_{t,i}\\frac{\\pi_{\\theta}(a_{t,i}\\mid s_{t,i})}{\\pi_{\\theta'}(a_{t,i}\\mid s_{t,i})}\\nabla_{\\theta'}\\log\\pi_{\\theta}(a_{t,i}\\mid s_{t,i})\\hat A^{\\pi_{\\theta'}}(s_{t,i},a_{t,i})\\).",
        isCorrect: false,
      },
      {
        text: "The ratio can be omitted after the first gradient step because \\(\\hat A^{\\pi_\\theta}\\) already contains the policy mismatch correction.",
        isCorrect: false,
      },
    ],
    explanation:
      "The ratio must put the target policy probability over the behavior policy probability for the same sampled action. The score-function term is differentiated under the updated policy, while the old-policy denominator and old-policy advantage estimates are not themselves recomputed inside that gradient.",
  },

  {
    id: "cs224r-lect5-q38",
    chapter: 5,
    difficulty: "hard",
    prompt: `A transition was collected under \\(\\pi_\\theta\\). For the sampled action, \\(\\pi_\\theta(a\\mid s)=0.10\\), \\(\\pi_{\\theta'}(a\\mid s)=0.25\\), and \\(\\hat A^{\\pi_\\theta}(s,a)=3\\). Ignoring the shared score-vector direction, which scalar multiplier appears in the importance-weighted actor update?`,
    options: [
      {
        text: "\\(7.5\\), because \\(\\frac{0.25}{0.10}\\cdot 3=7.5\\).",
        isCorrect: true,
      },
      {
        text: "\\(1.2\\), because \\(\\frac{0.10}{0.25}\\cdot 3=1.2\\).",
        isCorrect: false,
      },
      {
        text: "\\(3.0\\), because importance weights are used only for negative advantages.",
        isCorrect: false,
      },
      {
        text: "\\(0.75\\), because the target probability multiplies the advantage without division.",
        isCorrect: false,
      },
    ],
    explanation:
      "The off-policy correction uses the target-over-behavior ratio, so the sampled action receives \\(2.5\\) times its old-policy advantage. Reversing the ratio would downweight exactly the action whose probability increased under the new policy.",
  },

  {
    id: "cs224r-lect5-q39",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Why can repeatedly maximizing the same importance-weighted surrogate overfit a single batch of actor-critic data?",
    options: [
      {
        text: "Large positive old-policy advantages can keep pushing \\(\\pi_{\\theta'}(a\\mid s)/\\pi_\\theta(a\\mid s)\\) upward for the same sampled actions.",
        isCorrect: true,
      },
      {
        text: "The advantage labels \\(\\hat A^{\\pi_\\theta}\\) were estimated from a finite batch under \\(\\pi_\\theta\\), so they can become inaccurate after the policy changes substantially.",
        isCorrect: true,
      },
      {
        text: "A neural policy can push \\(\\pi_{\\theta'}(a\\mid s)\\) close to 1 around high-advantage samples and lose useful exploration.",
        isCorrect: true,
      },
      {
        text: "The problem is that automatic differentiation makes \\(\\nabla_{\\theta'}\\pi_\\theta(a\\mid s)\\neq0\\) for the old-policy denominator.",
        isCorrect: false,
      },
    ],
    explanation:
      "The instability comes from optimizing against stale labels and finite-batch quirks, not from a basic autodiff error. Constraining the update keeps the new policy near the data-collection policy so the old advantage estimates remain more meaningful.",
  },

  {
    id: "cs224r-lect5-q40",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly connect the surrogate objective \\(\\tilde J(\\theta')\\approx\\sum_{t,i}\\rho_{t,i}(\\theta')\\hat A^{\\pi_\\theta}_{t,i}\\) to the importance-weighted policy gradient?",
    options: [
      {
        text: "Differentiating \\(\\pi_{\\theta'}(a\\mid s)\\) gives \\(\\pi_{\\theta'}(a\\mid s)\\nabla_{\\theta'}\\log\\pi_{\\theta'}(a\\mid s)\\), which recovers the score-function form.",
        isCorrect: true,
      },
      {
        text: "The surrogate makes it visually clear why \\(\\hat A^{\\pi_\\theta}_{t,i}>0\\) incentivizes increasing \\(\\rho_{t,i}(\\theta')\\).",
        isCorrect: true,
      },
      {
        text: "The surrogate is maximized by differentiating \\(\\hat A^{\\pi_\\theta}_{t,i}\\) through \\(\\phi\\) on every actor step, producing \\(\\nabla_{\\theta'}\\hat A^{\\pi_\\theta}\\).",
        isCorrect: false,
      },
      {
        text: "The surrogate removes the need to know the behavior probability \\(\\pi_\\theta(a_{t,i}\\mid s_{t,i})\\) for each sampled action.",
        isCorrect: false,
      },
    ],
    explanation:
      "The surrogate is a convenient objective whose gradient has the desired importance-weighted score term. It still depends on knowing the behavior probability in the denominator and treats the old batch's advantage estimates as fixed actor-update weights.",
  },

  {
    id: "cs224r-lect5-q41",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe using a Kullback-Leibler (KL) constraint or penalty between \\(\\pi_{\\theta'}\\) and \\(\\pi_\\theta\\) during repeated policy updates?",
    options: [
      {
        text: "It discourages \\(\\pi_{\\theta'}(\\cdot\\mid s)\\) from moving too far from \\(\\pi_\\theta(\\cdot\\mid s)\\), the policy that produced the batch.",
        isCorrect: true,
      },
      {
        text: "Keeping \\(D_{KL}(\\pi_{\\theta'}\\|\\pi_\\theta)\\) small makes old-policy advantages more likely to remain useful for \\(\\pi_{\\theta'}\\).",
        isCorrect: true,
      },
      {
        text: "It can be written as a term such as \\(-\\beta D_{KL}(\\pi_{\\theta'}(\\cdot\\mid s)\\|\\pi_\\theta(\\cdot\\mid s))\\) when maximizing a surrogate.",
        isCorrect: true,
      },
      {
        text: "It exactly fixes replay-buffer state-distribution mismatch by making \\(p_{\\mathcal R}(s)=p_{\\theta'}(s)\\) for all later updates.",
        isCorrect: false,
      },
    ],
    explanation:
      "The KL term is a local step-size control for policy updates on a recent batch. It does not solve the harder replay-buffer problem where the states themselves may come from many older policies.",
  },

  {
    id: "cs224r-lect5-q42",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "For PPO-style clipping with \\(\\rho=\\frac{\\pi_{\\theta'}(a\\mid s)}{\\pi_\\theta(a\\mid s)}\\), which statements are correct?",
    options: [
      {
        text: "The clipped term uses \\(\\mathrm{clip}(\\rho,1-\\epsilon,1+\\epsilon)\\hat A\\) as one candidate objective contribution.",
        isCorrect: true,
      },
      {
        text: "For \\(\\hat A>0\\), clipping above \\(1+\\epsilon\\) stops rewarding further increases in the sampled action's probability.",
        isCorrect: true,
      },
      {
        text: "For \\(\\hat A<0\\), clipping below \\(1-\\epsilon\\) stops rewarding further decreases in the sampled action's probability.",
        isCorrect: true,
      },
      {
        text: "Clipping guarantees \\(D_{KL}(\\pi_{\\theta'}(\\cdot\\mid s)\\|\\pi_\\theta(\\cdot\\mid s))\\le\\epsilon\\) at every state.",
        isCorrect: false,
      },
    ],
    explanation:
      "Clipping controls the incentive created by each sampled ratio, which is weaker than a hard distribution-level KL constraint. The sign of the advantage matters because increasing a bad action and decreasing a good action should not be rewarded.",
  },

  {
    id: "cs224r-lect5-q43",
    chapter: 5,
    difficulty: "hard",
    prompt: `Use the PPO clipped objective contribution
\\[
\\min\\left(\\rho \\hat A,\\; \\mathrm{clip}(\\rho,1-\\epsilon,1+\\epsilon)\\hat A\\right).
\\]
If \\(\\epsilon=0.2\\), \\(\\rho=1.5\\), and \\(\\hat A=4\\), what contribution is used?`,
    options: [
      {
        text: "\\(4.8\\), because \\(\\min(1.5\\cdot4,1.2\\cdot4)=4.8\\).",
        isCorrect: true,
      },
      {
        text: "\\(6.0\\), because \\(1.5\\cdot4\\) is the unclipped contribution.",
        isCorrect: false,
      },
      {
        text: "\\(3.2\\), because the ratio is always clipped to \\(1-\\epsilon=0.8\\).",
        isCorrect: false,
      },
      {
        text: "\\(1.2\\), because clipping returns only the clipped ratio.",
        isCorrect: false,
      },
    ],
    explanation:
      "For a positive advantage, a ratio above \\(1.2\\) would keep rewarding an already-large probability increase. The clipped candidate is \\(4.8\\), and the per-sample minimum selects it over the larger unclipped contribution.",
  },

  {
    id: "cs224r-lect5-q44",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Why does PPO use the per-sample minimum between the unclipped and clipped surrogate terms?",
    options: [
      {
        text: "It prevents clipping from accidentally increasing the objective for a sample when the ratio moved in a harmful direction.",
        isCorrect: true,
      },
      {
        text: "It makes the clipped surrogate a conservative version of the original importance-weighted term for that sample.",
        isCorrect: true,
      },
      {
        text: "It makes the objective exactly unbiased for arbitrarily large changes from \\(\\pi_\\theta\\) to \\(\\pi_{\\theta'}\\).",
        isCorrect: false,
      },
      {
        text: "It eliminates the need to collect new batches after the PPO epochs are finished.",
        isCorrect: false,
      },
    ],
    explanation:
      "The minimum implements a pessimistic choice between the raw and clipped ratio contributions. It improves local stability, but PPO is still a biased practical surrogate and still refreshes data after multiple gradient steps.",
  },

  {
    id: "cs224r-lect5-q45",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe generalized advantage estimation (GAE) as used with PPO?",
    options: [
      {
        text: "It starts from value estimates \\(\\hat V^\\pi\\) that can be fit with Monte Carlo, bootstrapped, or n-step-style targets.",
        isCorrect: true,
      },
      {
        text: "An n-step advantage can be written in the form \\(\\hat A_n(s_t,a_t)=\\sum_{k=0}^{n-1}\\gamma^k r_{t+k}-\\hat V(s_t)+\\gamma^n\\hat V(s_{t+n})\\).",
        isCorrect: true,
      },
      {
        text: "GAE combines several horizon lengths, for example \\(\\hat A_{GAE}(s_t,a_t)=\\sum_{n=1}^{N}w_n\\hat A_n(s_t,a_t)\\).",
        isCorrect: true,
      },
      {
        text: "Choosing weights that emphasize shorter horizons can reduce variance by cutting off the return earlier.",
        isCorrect: true,
      },
    ],
    explanation:
      "GAE is a weighted mixture of advantage estimates with different bootstrap horizons. Shorter horizons rely more on the critic and usually reduce variance, while longer horizons use more observed rewards and can reduce bootstrap bias.",
  },

  {
    id: "cs224r-lect5-q46",
    chapter: 5,
    difficulty: "hard",
    prompt: `For a two-step advantage estimate, use
\\[
\\hat A_2(s_t,a_t)=r_t+\\gamma r_{t+1}-\\hat V(s_t)+\\gamma^2\\hat V(s_{t+2}).
\\]
If \\(r_t=1\\), \\(r_{t+1}=2\\), \\(\\gamma=0.5\\), \\(\\hat V(s_t)=3\\), and \\(\\hat V(s_{t+2})=4\\), what is \\(\\hat A_2\\)?`,
    options: [
      {
        text: "\\(0\\), because \\(1+0.5\\cdot2-3+0.25\\cdot4=0\\).",
        isCorrect: true,
      },
      {
        text: "\\(1\\), because the bootstrap term is \\(0.5\\cdot4\\).",
        isCorrect: false,
      },
      {
        text: "\\(-1\\), because the second reward is omitted in a two-step estimate.",
        isCorrect: false,
      },
      {
        text: "\\(4\\), because the baseline \\(\\hat V(s_t)\\) is added rather than subtracted.",
        isCorrect: false,
      },
    ],
    explanation:
      "The two-step estimate includes two discounted rewards, subtracts the current-state baseline, and then adds the discounted bootstrap value. The exponent on the bootstrap term is \\(\\gamma^2\\), not \\(\\gamma\\).",
  },

  {
    id: "cs224r-lect5-q47",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the bias-variance role of GAE horizon weights such as \\(w_n\\propto\\lambda^{n-1}\\)?",
    options: [
      {
        text: "Smaller \\(\\lambda\\) gives more weight to short-horizon estimates and generally lowers variance.",
        isCorrect: true,
      },
      {
        text: "Larger \\(\\lambda\\) keeps more long-horizon return information and can reduce bootstrap bias.",
        isCorrect: true,
      },
      {
        text: "Setting \\(\\lambda=0\\) makes the estimate independent of the value function.",
        isCorrect: false,
      },
      {
        text: "Setting \\(\\lambda\\) changes the PPO clipping interval from \\([1-\\epsilon,1+\\epsilon]\\) to \\([1-\\lambda,1+\\lambda]\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "GAE's horizon weights control how quickly the advantage estimate cuts off future rewards and bootstraps. The clipping parameter \\(\\epsilon\\) controls policy-ratio incentives, while \\(\\lambda\\) controls the advantage estimator.",
  },

  {
    id: "cs224r-lect5-q48",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which steps belong in the practical PPO loop for actor-critic learning?",
    options: [
      {
        text: "Collect a batch of trajectories from the current policy \\(\\pi_\\theta\\).",
        isCorrect: true,
      },
      {
        text: "Fit a value function \\(\\hat V_\\phi^{\\pi_\\theta}\\) on the collected data.",
        isCorrect: true,
      },
      {
        text: "Estimate advantages, often with GAE, using the fitted value function.",
        isCorrect: true,
      },
      {
        text: "Take several actor gradient steps on the clipped surrogate, then collect a fresh batch.",
        isCorrect: true,
      },
    ],
    explanation:
      "PPO is more off-policy than one-step vanilla policy gradient because it reuses one batch for several actor updates. It is still not a replay-buffer method because the data is refreshed after those epochs rather than sampled indefinitely from all past experience.",
  },

  {
    id: "cs224r-lect5-q49",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which interpretations of PPO implementation hyperparameters are correct?",
    options: [
      {
        text: "A clipping range such as \\(\\epsilon=0.2\\) bounds each sampled ratio contribution to use \\([0.8,1.2]\\) inside the clipped term.",
        isCorrect: true,
      },
      {
        text: "Multiple epochs \\(M>1\\) over a batch increase data reuse but make stale-advantage overfitting more relevant.",
        isCorrect: true,
      },
      {
        text: "A minibatch size such as \\(64\\) means PPO collects only \\(64\\) total environment timesteps before refreshing data.",
        isCorrect: false,
      },
      {
        text: "Increasing \\(M\\) always improves the final policy because clipping makes \\(\\rho\\hat A\\) immune to overfitting risk.",
        isCorrect: false,
      },
    ],
    explanation:
      "The batch can contain thousands of timesteps while the optimizer uses smaller minibatches for many updates. Reusing a batch improves sample efficiency, but PPO's clipping is a stabilizer rather than a guarantee that more updates are always better.",
  },

  {
    id: "cs224r-lect5-q50",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the replay-buffer version of off-policy actor-critic?",
    options: [
      {
        text: "The replay buffer stores previous transitions such as \\((s_i,a_i,r_i,s'_i)\\).",
        isCorrect: true,
      },
      {
        text: "Updates can use minibatches \\(\\{(s_i,a_i,r_i,s'_i)\\}\\sim\\mathcal R\\) rather than only the most recent rollout batch.",
        isCorrect: true,
      },
      {
        text: "The algorithm must adjust value and actor equations because \\(a_i\\) in \\(\\mathcal R\\) can come from older policies.",
        isCorrect: true,
      },
      {
        text: "Once \\(\\mathcal R\\) is used, the actor update is automatically on-policy with respect to \\(p_\\theta(s,a)\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Replay buffers make the algorithm more off-policy by reusing older experience. That reuse is valuable only if the critic and actor updates avoid treating old actions and old future rollouts as if they came from the current policy.",
  },

  {
    id: "cs224r-lect5-q51",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Why is the naive replay-buffer update \\(y_i=r_i+\\gamma\\hat V_\\phi(s'_i)\\) not the right way to learn \\(V^{\\pi_\\theta}\\) for the current policy from all old data?",
    options: [
      {
        text: "The sampled actions and next states in the buffer were produced by past policies, so the fitted value tends to represent a mixture of past behavior rather than \\(\\pi_\\theta\\).",
        isCorrect: true,
      },
      {
        text: "A state-value target \\(\\hat V(s_i)\\) does not condition on \\(a_i\\), even though that old action determined the observed reward and next state.",
        isCorrect: true,
      },
      {
        text: "The update is invalid because \\(s'_i\\) is never observed in replay-buffer tuples \\((s_i,a_i,r_i,s'_i)\\).",
        isCorrect: false,
      },
      {
        text: "The update is invalid because value functions cannot minimize losses such as \\(\\sum_i\\lVert\\hat V(s_i)-y_i\\rVert^2\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The problem is not the loss form or a missing next state; the problem is policy mismatch. A state-value function averages over actions from a policy, and replay-buffer samples do not generally reflect the current policy's action distribution.",
  },

  {
    id: "cs224r-lect5-q52",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly explain why fitting \\(Q^{\\pi_\\theta}(s,a)\\) helps with replay-buffer data?",
    options: [
      {
        text: "\\(Q^{\\pi_\\theta}(s,a)\\) conditions on the first action \\(a\\), so \\(\\hat Q_\\phi(s_i,a_i)\\) can use the buffered action as a critic input.",
        isCorrect: true,
      },
      {
        text: "The Bellman relation can be written \\(Q^{\\pi_\\theta}(s,a)=r(s,a)+\\gamma\\mathbb{E}_{s'\\sim p(\\cdot\\mid s,a),\\bar a'\\sim\\pi_\\theta(\\cdot\\mid s')}[Q^{\\pi_\\theta}(s',\\bar a')]\\).",
        isCorrect: true,
      },
      {
        text: "The next action in \\(\\hat Q(s'_i,a'_i)\\) should use \\(a'_i\\sim\\pi_\\theta(\\cdot\\mid s'_i)\\), giving targets like \\(y_i=r_i+\\gamma\\hat Q(s'_i,a'_i)\\).",
        isCorrect: true,
      },
      {
        text: "The target \\(y_i=r_i+\\gamma\\hat Q(s'_i,a'_i)\\) still relies on coverage because \\(a'_i\\sim\\pi_\\theta(\\cdot\\mid s'_i)\\) may differ from buffered actions.",
        isCorrect: true,
      },
    ],
    explanation:
      "Q-functions let the update accept an off-policy first action while making the future part follow the current policy through a new next-action sample. This does not create information about unseen actions for free, so action coverage and function approximation quality remain central.",
  },

  {
    id: "cs224r-lect5-q53",
    chapter: 5,
    difficulty: "hard",
    prompt: `A replay-buffer transition has \\(r_i=1.5\\) and \\(\\gamma=0.9\\). For the next state, a current-policy sample gives \\(a'_i\\sim\\pi_\\theta(\\cdot\\mid s'_i)\\), and the critic predicts \\(\\hat Q_\\phi(s'_i,a'_i)=4\\). What is the one-sample target \\(y_i\\) for fitting \\(\\hat Q_\\phi(s_i,a_i)\\)?`,
    options: [
      { text: "\\(5.1\\), because \\(y_i=1.5+0.9\\cdot4\\).", isCorrect: true },
      {
        text: "\\(3.6\\), because \\(y_i=0.9\\cdot4\\) omits the immediate reward.",
        isCorrect: false,
      },
      {
        text: "\\(4.0\\), because \\(y_i=\\hat Q(s'_i,a'_i)\\) drops both \\(r_i\\) and \\(\\gamma\\).",
        isCorrect: false,
      },
      {
        text: "\\(1.5\\), because \\(y_i=r_i\\) would ignore the current-policy bootstrap term.",
        isCorrect: false,
      },
    ],
    explanation:
      "The target is the immediate reward plus the discounted current-policy bootstrap estimate. The next action is not taken from the old trajectory; it is newly sampled from the current policy at the buffered next state.",
  },

  {
    id: "cs224r-lect5-q54",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the action-coverage requirement for replay-buffer Q targets?",
    options: [
      {
        text: "The buffer must contain enough nearby state-action experience for \\(\\hat Q(s,a)\\) to generalize to actions the current policy may choose.",
        isCorrect: true,
      },
      {
        text: "If the current policy chooses actions far outside the buffer's support, the target \\(r+\\gamma\\hat Q(s',a')\\) can be an unstable extrapolation.",
        isCorrect: true,
      },
      {
        text: "A stochastic policy and function approximation can help by giving multiple current-policy action samples across a minibatch.",
        isCorrect: true,
      },
      {
        text: "Coverage is irrelevant because the Bellman equation is exact for all neural-network predictions.",
        isCorrect: false,
      },
    ],
    explanation:
      "The Bellman equation specifies the true Q-function, but the learned critic must estimate it from finite data. When the replay buffer lacks relevant actions, the critic may bootstrap from guesses rather than grounded estimates.",
  },

  {
    id: "cs224r-lect5-q55",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the critic loss in the replay-buffer off-policy actor-critic update?",
    options: [
      {
        text: "A typical target is \\(y_i=r_i+\\gamma\\hat Q_\\phi^{\\pi}(s'_i,a'_i)\\), where \\(a'_i\\sim\\pi_\\theta(\\cdot\\mid s'_i)\\).",
        isCorrect: true,
      },
      {
        text: "A typical loss is \\(\\mathcal{L}(\\phi)=\\frac{1}{N}\\sum_i\\lVert\\hat Q_\\phi^{\\pi}(s_i,a_i)-y_i\\rVert^2\\).",
        isCorrect: true,
      },
      {
        text: "The target must use \\(a'_i\\) from the replay buffer's original future trajectory to preserve off-policy correctness.",
        isCorrect: false,
      },
      {
        text: "The critic loss is optimized with respect to \\(\\theta\\) because the actor's log probability appears in \\(\\hat Q_\\phi\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The critic is a regression problem in \\(\\phi\\), with labels built from rewards and current-policy bootstrap actions. Using old future actions would make the target follow the past policy instead of the current actor.",
  },

  {
    id: "cs224r-lect5-q56",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the actor update after switching from \\(\\hat V\\) and \\(\\hat A\\) to \\(\\hat Q\\) in the replay-buffer algorithm?",
    options: [
      {
        text: "It is convenient to use \\(\\hat Q^\\pi(s,a)\\) rather than \\(\\hat A^\\pi(s,a)\\), even though this removes the average-reward baseline.",
        isCorrect: true,
      },
      {
        text: "The action used inside \\(\\nabla_\\theta\\log\\pi_\\theta(a\\mid s)\\hat Q(s,a)\\) should be sampled from the current policy.",
        isCorrect: true,
      },
      {
        text: "The resulting estimator can have higher variance than an advantage estimator, but replay-buffer data reuse helps compensate.",
        isCorrect: true,
      },
      {
        text: "The actor should always use the buffered action \\(a_i\\), because it has an observed reward attached to it.",
        isCorrect: false,
      },
    ],
    explanation:
      "The actor is trying to improve the current policy, so its score term should be evaluated on current-policy actions. Buffered actions are useful critic inputs, but treating them as current actor samples would reintroduce the same action-mismatch problem.",
  },

  {
    id: "cs224r-lect5-q57",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which equations correctly use current-policy action samples in the replay-buffer actor update?",
    options: [
      {
        text: "\\(a_i^\\pi\\sim\\pi_\\theta(\\cdot\\mid s_i)\\), then \\(\\nabla_\\theta J(\\theta)\\approx\\frac{1}{N}\\sum_i\\nabla_\\theta\\log\\pi_\\theta(a_i^\\pi\\mid s_i)\\hat Q^\\pi(s_i,a_i^\\pi)\\).",
        isCorrect: true,
      },
      {
        text: "For the critic target, \\(a'_i\\sim\\pi_\\theta(\\cdot\\mid s'_i)\\), then \\(y_i=r_i+\\gamma\\hat Q^\\pi(s'_i,a'_i)\\).",
        isCorrect: true,
      },
      {
        text: "\\(a_i^\\pi=a_i^{buffer}\\), then \\(\\nabla_\\theta J(\\theta)\\approx\\frac{1}{N}\\sum_i\\nabla_\\theta\\log\\pi_\\theta(a_i^{buffer}\\mid s_i)\\hat Q^\\pi(s_i,a_i^{buffer})\\).",
        isCorrect: false,
      },
      {
        text: "For the critic target, \\(a'_i=a_{i+1}^{buffer}\\), then \\(y_i=r_i+\\gamma\\hat Q^\\pi(s'_i,a_{i+1}^{buffer})\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Both the current-state actor action and the next-state bootstrap action should come from \\(\\pi_\\theta\\). The replay buffer supplies states, actions, rewards, and next states, but not the current policy's action choices.",
  },

  {
    id: "cs224r-lect5-q58",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "After the replay-buffer algorithm fixes the action mismatch with Q-functions and current-policy action samples, what state-distribution issue remains?",
    options: [
      {
        text: "The sampled states \\(s_i\\) still come from the replay buffer rather than exactly from the current policy's state distribution \\(p_\\theta(s)\\).",
        isCorrect: true,
      },
      {
        text: "The resulting policy is optimized on a broader distribution of states than the current policy alone might visit.",
        isCorrect: true,
      },
      {
        text: "This remaining mismatch is accepted in practical off-policy actor-critic rather than fully corrected by the basic algorithm.",
        isCorrect: true,
      },
      {
        text: "The state mismatch disappears because \\(a'_i\\sim\\pi_\\theta(\\cdot\\mid s'_i)\\) in the critic target.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sampling current-policy actions fixes action mismatch, not the distribution of states in the minibatch. The practical algorithm accepts learning over a broader replay distribution, which is useful but not identical to an on-policy state distribution.",
  },

  {
    id: "cs224r-lect5-q59",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statement correctly describes the reparameterization-trick note for Gaussian policies in off-policy actor-critic?",
    options: [
      {
        text: "It can provide an alternative lower-variance way to estimate actor gradients through sampled Gaussian actions.",
        isCorrect: true,
      },
      {
        text: "It replaces the replay buffer by generating synthetic transitions from the policy distribution.",
        isCorrect: false,
      },
      {
        text: "It is the mechanism that clips PPO ratios to \\([1-\\epsilon,1+\\epsilon]\\).",
        isCorrect: false,
      },
      {
        text: "It makes the critic target independent of \\(\\hat Q(s',a')\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The reparameterization trick is about differentiating through stochastic continuous-action samples, such as Gaussian actions. It is separate from replay storage, PPO clipping, and the Bellman target used to fit the critic.",
  },

  {
    id: "cs224r-lect5-q60",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which comparisons between PPO and SAC-style replay-buffer actor-critic are correct?",
    options: [
      {
        text: "PPO is more on-policy because it reuses a recent rollout batch for several updates and then recollects.",
        isCorrect: true,
      },
      {
        text: "SAC-style methods are more off-policy because they sample minibatches from a replay buffer of past experience.",
        isCorrect: true,
      },
      {
        text: "PPO is often more stable and plug-and-play, while SAC-style methods are often more data efficient.",
        isCorrect: true,
      },
      {
        text: "Replay-buffer methods can be harder to tune because Q-function bootstrapping and off-policy data introduce extra instability.",
        isCorrect: true,
      },
    ],
    explanation:
      "PPO and SAC trade off stability and sample efficiency in different ways. PPO constrains local policy updates on fresh data, while SAC-style replay-buffer learning extracts more updates from old data but leans harder on accurate Q-function estimates.",
  },

  {
    id: "cs224r-lect5-q61",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the Markov-property distinction in the PPO-versus-SAC practical guidelines?",
    options: [
      {
        text: "A Monte Carlo value estimate \\(G_t=\\sum_{k\\ge0}\\gamma^k r_{t+k}\\) can reduce reliance on one-step Markov bootstrapping.",
        isCorrect: true,
      },
      {
        text: "Replay-buffer SAC-style targets rely heavily on the Markov structure because they bootstrap with \\(r_i+\\gamma\\hat Q(s'_i,a'_i)\\).",
        isCorrect: true,
      },
      {
        text: "If observations are non-Markov, one-step Q targets can be misleading because \\(s'_i\\) may not contain all information needed for future return.",
        isCorrect: true,
      },
      {
        text: "Using full observed returns can be more robust to hidden state, but \\(G_t\\)-style targets usually cost variance and data efficiency.",
        isCorrect: true,
      },
    ],
    explanation:
      "Bootstrapped Q-learning-style updates assume the next observation is sufficient for predicting future value under the policy. Monte Carlo returns can avoid that particular one-step modeling assumption, but they usually pay with noisier estimates and less efficient learning.",
  },

  {
    id: "cs224r-lect5-q62",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which uses of demonstrations match the PPO/SAC practical comparison?",
    options: [
      {
        text: "For PPO, demonstrations can initialize or pretrain the policy before online reinforcement learning.",
        isCorrect: true,
      },
      {
        text: "For SAC-style replay-buffer learning, demonstrations can be added to the replay buffer as useful prior experience.",
        isCorrect: true,
      },
      {
        text: "For PPO, demonstrations remove the need to estimate advantages during policy updates.",
        isCorrect: false,
      },
      {
        text: "For SAC-style methods, demonstrations make the Bellman target exact even when action coverage is poor.",
        isCorrect: false,
      },
    ],
    explanation:
      "Demonstrations can seed either family of methods, but they do not replace the RL updates. PPO still needs an actor-critic objective, and SAC-style methods still need grounded Q estimates with enough coverage around relevant actions.",
  },

  {
    id: "cs224r-lect5-q63",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which deployment-oriented statements correctly follow from the PPO/SAC comparison?",
    options: [
      {
        text: "SAC-style replay-buffer learning can be sample efficient enough for some real-robot reinforcement-learning settings.",
        isCorrect: true,
      },
      {
        text: "PPO is a common choice when stable learning is more important than raw environment-sample efficiency, such as many simulation or language-model settings.",
        isCorrect: true,
      },
      {
        text: "Imitation learning can be useful for initialization, but RL can improve beyond demonstrations when reward optimization is feasible.",
        isCorrect: true,
      },
      {
        text: "SAC is preferred for all language-model reinforcement learning because replay buffers remove policy-distribution shift.",
        isCorrect: false,
      },
    ],
    explanation:
      "Algorithm choice depends on the cost of data, the stability requirements, and the domain. Replay buffers are powerful for sample efficiency, while PPO-style updates remain attractive when predictable optimization behavior matters.",
  },

  {
    id: "cs224r-lect5-q64",
    chapter: 5,
    difficulty: "hard",
    prompt: `For two PPO samples, use \\(\\epsilon=0.2\\) and
\\[
L_i=\\min(\\rho_i A_i,\\mathrm{clip}(\\rho_i,0.8,1.2)A_i).
\\]

| sample | \\(\\rho_i\\) | \\(A_i\\) |
| --- | ---: | ---: |
| 1 | 1.4 | 2 |
| 2 | 0.6 | -3 |

What is \\(L_1+L_2\\)?`,
    options: [
      {
        text: "\\(0.0\\), because \\(L_1=2.4\\) and \\(L_2=-2.4\\).",
        isCorrect: true,
      },
      {
        text: "\\(1.0\\), because \\(L_1=2.8\\) and \\(L_2=-1.8\\).",
        isCorrect: false,
      },
      {
        text: "\\(-0.6\\), because \\(L_1=2.4\\) and \\(L_2=-3.0\\).",
        isCorrect: false,
      },
      {
        text: "\\(-0.4\\), because \\(L_1=2.4\\) and \\(L_2=-2.8\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "For the positive-advantage sample, the minimum selects the clipped contribution \\(1.2\\cdot2=2.4\\). For the negative-advantage sample, the clipped contribution is \\(0.8\\cdot(-3)=-2.4\\), which is lower than \\(0.6\\cdot(-3)=-1.8\\), so the two contributions sum to zero.",
  },

  {
    id: "cs224r-lect5-q65",
    chapter: 5,
    difficulty: "hard",
    prompt: `With \\(\\epsilon=0.2\\), \\(\\rho=0.6\\), and \\(\\hat A=-5\\), which PPO clipped-objective contribution is used?
\\[
\\min(\\rho\\hat A,\\mathrm{clip}(\\rho,0.8,1.2)\\hat A)
\\]`,
    options: [
      { text: "\\(-4\\), because \\(\\min(-3,-4)=-4\\).", isCorrect: true },
      {
        text: "\\(-3\\), because \\(0.6\\cdot(-5)=-3\\) is the unclipped term.",
        isCorrect: false,
      },
      {
        text: "\\(4\\), because PPO takes the magnitude of negative advantages.",
        isCorrect: false,
      },
      {
        text: "\\(-6\\), because the ratio is clipped to \\(1.2\\) for all negative advantages.",
        isCorrect: false,
      },
    ],
    explanation:
      "For a negative advantage, decreasing the action probability too far can make the unclipped objective less negative and therefore look better under maximization. The per-sample minimum selects the more pessimistic clipped value, \\(-4\\), in that case.",
  },

  {
    id: "cs224r-lect5-q66",
    chapter: 5,
    difficulty: "hard",
    prompt: "Which classifications of actor-critic variants are correct?",
    options: [
      {
        text: "One gradient step on one freshly collected batch is the most on-policy version among the variants discussed.",
        isCorrect: true,
      },
      {
        text: "PPO is partially off-policy because it takes multiple gradient steps on a batch collected by a previous policy.",
        isCorrect: true,
      },
      {
        text: "Replay-buffer actor-critic is more off-policy because it samples updates from many past batches, not only the latest rollout.",
        isCorrect: true,
      },
      {
        text: "A replay buffer makes PPO more on-policy because it stores the exact action probabilities from old policies.",
        isCorrect: false,
      },
    ],
    explanation:
      "The variants differ by how aggressively they reuse data from policies other than the current one. Storing old data does not make an update on-policy; it increases the need for corrections or algorithmic changes.",
  },

  {
    id: "cs224r-lect5-q67",
    chapter: 5,
    difficulty: "hard",
    prompt: `A sampled action had behavior probability \\(\\pi_\\theta(a\\mid s)=0.02\\). During surrogate optimization, \\(\\pi_{\\theta'}(a\\mid s)\\) rises to \\(0.40\\) and \\(\\hat A^{\\pi_\\theta}(s,a)=1\\). Which statement best explains the instability risk?`,
    options: [
      {
        text: "The ratio becomes \\(20\\), so a finite-batch positive advantage can create a very large incentive to overfit that sampled action.",
        isCorrect: true,
      },
      {
        text: "The ratio becomes \\(0.05\\), so the update almost ignores the sampled action despite its positive advantage.",
        isCorrect: false,
      },
      {
        text: "The ratio is irrelevant because \\(\\hat A=1\\) implies the policy is already optimal at that state.",
        isCorrect: false,
      },
      {
        text: "The denominator is differentiated through \\(\\theta'\\), so the ratio necessarily becomes negative.",
        isCorrect: false,
      },
    ],
    explanation:
      "A small behavior probability in the denominator can make the importance ratio very large when the new policy raises that action's probability. PPO clipping or KL penalties are ways to reduce this incentive before it dominates the batch.",
  },

  {
    id: "cs224r-lect5-q68",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly distinguish the replay-buffer \\(\\hat V\\) mistake from the \\(\\hat Q\\) fix?",
    options: [
      {
        text: "The naive \\(\\hat V\\) target uses \\(r_i+\\gamma\\hat V(s'_i)\\), but \\(r_i\\) and \\(s'_i\\) resulted from a buffered action sampled by an old policy.",
        isCorrect: true,
      },
      {
        text: "The \\(\\hat Q\\) fix uses the buffered first action as part of the critic input, so the target is about \\(Q(s_i,a_i)\\) rather than just \\(V(s_i)\\).",
        isCorrect: true,
      },
      {
        text: "The \\(\\hat Q\\) target replaces the buffered future continuation with a current-policy next action \\(a'_i\\sim\\pi_\\theta(\\cdot\\mid s'_i)\\).",
        isCorrect: true,
      },
      {
        text: "The \\(\\hat Q\\) fix still needs enough coverage for the critic to evaluate current-policy actions accurately.",
        isCorrect: true,
      },
    ],
    explanation:
      "The Q-function does not pretend the old action was sampled from the current policy; it conditions on that action. The future part of the target is then made current-policy-specific by sampling the next action from \\(\\pi_\\theta\\), subject to the usual coverage limitations.",
  },

  {
    id: "cs224r-lect5-q69",
    chapter: 5,
    difficulty: "hard",
    prompt: `At a replay-buffer next state \\(s'_i\\), the current policy has

| action | \\(\\pi_\\theta(a\\mid s'_i)\\) | \\(\\hat Q(s'_i,a)\\) |
| --- | ---: | ---: |
| \\(a_1\\) | 0.25 | 2 |
| \\(a_2\\) | 0.75 | 6 |

If using the exact expectation instead of one sampled \\(a'_i\\), what is the bootstrap term \\(\\mathbb{E}_{a'\\sim\\pi_\\theta}[\\hat Q(s'_i,a')]\\)?`,
    options: [
      {
        text: "\\(5.0\\), because \\(0.25\\cdot2+0.75\\cdot6=5.0\\).",
        isCorrect: true,
      },
      {
        text: "\\(4.0\\), because the two Q-values are averaged without policy probabilities.",
        isCorrect: false,
      },
      {
        text: "\\(6.0\\), because the target always uses the greedy action under \\(\\hat Q\\).",
        isCorrect: false,
      },
      {
        text: "\\(2.0\\), because the buffered action must be reused in the next state.",
        isCorrect: false,
      },
    ],
    explanation:
      "The Bellman target for a stochastic current policy takes an expectation over actions from that policy. A one-sample target approximates this expectation, but the exact expectation is the policy-weighted average of the next-action Q-values.",
  },

  {
    id: "cs224r-lect5-q70",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly summarize the SAC-style replay-buffer actor-critic algorithm described by the equations?",
    options: [
      {
        text: "Collect \\((s,a,r,s')\\) by acting with the current policy and add the transition to \\(\\mathcal R\\).",
        isCorrect: true,
      },
      {
        text: "Sample a minibatch from \\(\\mathcal R\\), and fit \\(\\hat Q_\\phi\\) with targets \\(y_i=r_i+\\gamma\\hat Q_\\phi(s'_i,a'_i)\\), where \\(a'_i\\sim\\pi_\\theta(\\cdot\\mid s'_i)\\).",
        isCorrect: true,
      },
      {
        text: "Update the actor with current-policy samples \\(a_i^\\pi\\sim\\pi_\\theta(\\cdot\\mid s_i)\\) and a term like \\(\\nabla_\\theta\\log\\pi_\\theta(a_i^\\pi\\mid s_i)\\hat Q(s_i,a_i^\\pi)\\).",
        isCorrect: true,
      },
      {
        text: "Accept that replay states may come from a broader distribution than the exact current-policy state distribution.",
        isCorrect: true,
      },
    ],
    explanation:
      "The replay-buffer algorithm fixes action mismatch by using Q-functions and current-policy action samples in both critic and actor updates. Its remaining approximation is that the states are sampled from replay, which improves data reuse but is not exactly the current policy's state distribution.",
  },
];
