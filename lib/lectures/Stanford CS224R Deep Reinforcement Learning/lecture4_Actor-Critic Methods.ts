import { Question } from "../../quiz";

export const cs224rLecture4ActorCriticQuestions: Question[] = [
  // ============================================================
  // ALL TRUE (Q1–Q9)
  // ============================================================

  {
    id: "cs224r-lect4-q01",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the value function \\(V^\\pi(s)\\)?",
    options: [
      {
        text: "It represents expected future rewards when following policy \\(\\pi\\) from state \\(s\\).",
        isCorrect: true,
      },
      {
        text: "It depends on both the policy and environment dynamics.",
        isCorrect: true,
      },
      {
        text: "It can be interpreted as a probability of success in some tasks.",
        isCorrect: true,
      },
      {
        text: "It is defined as an expectation over trajectories starting at \\(s\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The value function captures how good a state is under a specific policy. It is the expected cumulative reward when starting from that state and acting according to the policy.",
  },

  {
    id: "cs224r-lect4-q02",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the Q-function \\(Q^\\pi(s,a)\\)?",
    options: [
      {
        text: "It measures expected return after taking action \\(a\\) in state \\(s\\) and then following \\(\\pi\\).",
        isCorrect: true,
      },
      {
        text: "It differs from \\(V^\\pi(s)\\) by conditioning on the first action.",
        isCorrect: true,
      },
      {
        text: "It marginalizes over future stochastic transitions.",
        isCorrect: true,
      },
      {
        text: "It includes immediate reward plus expected future rewards.",
        isCorrect: true,
      },
    ],
    explanation:
      'The Q-function evaluates state–action pairs. It explicitly conditions on the initial action before following the policy thereafter. To reason through the choices, select every statement because each one matches the criterion in the prompt: "It measures expected return after taking action \\(a\\) in state \\(s\\) and then following \\(\\pi\\)."; "It differs from \\(V^\\pi(s)\\) by conditioning on the first action."; "It marginalizes over future stochastic transitions."; "It includes immediate reward plus expected future rewards.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "cs224r-lect4-q03",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements about the advantage function \\(A^\\pi(s,a)\\) are correct?",
    options: [
      { text: "\\(A^\\pi(s,a) = Q^\\pi(s,a) - V^\\pi(s)\\).", isCorrect: true },
      {
        text: "It measures how much better an action is than the policy’s average choice.",
        isCorrect: true,
      },
      {
        text: "Positive advantage implies the action is better than expected.",
        isCorrect: true,
      },
      {
        text: "It centers Q-values around the state value baseline.",
        isCorrect: true,
      },
    ],
    explanation:
      "The advantage function compares a chosen action to the expected action under the policy. It is central to reducing variance in policy gradient updates.",
  },

  {
    id: "cs224r-lect4-q04",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements correctly describe actor–critic methods?",
    options: [
      {
        text: "They use a critic to estimate value functions.",
        isCorrect: true,
      },
      {
        text: "They use advantage estimates to update the actor.",
        isCorrect: true,
      },
      {
        text: "They improve sample efficiency compared to vanilla policy gradients.",
        isCorrect: true,
      },
      {
        text: "They consist of two separate parameterized models.",
        isCorrect: true,
      },
    ],
    explanation:
      'Actor–critic algorithms separate policy learning and value estimation. The critic provides a learned signal that reduces noise in the actor update. To reason through the choices, select every statement because each one matches the criterion in the prompt: "They use a critic to estimate value functions."; "They use advantage estimates to update the actor."; "They improve sample efficiency compared to vanilla policy gradients."; "They consist of two separate parameterized models.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "cs224r-lect4-q05",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about Monte Carlo value estimation are correct?",
    options: [
      {
        text: "Targets are full returns \\(G_t = \\sum_{t'=t}^T r_{t'}\\).",
        isCorrect: true,
      },
      { text: "It is unbiased.", isCorrect: true },
      { text: "It suffers from high variance.", isCorrect: true },
      {
        text: "It does not rely on bootstrapped predictions.",
        isCorrect: true,
      },
    ],
    explanation:
      'Monte Carlo directly regresses on sampled trajectory returns. While unbiased, variance grows with trajectory length. To reason through the choices, select every statement because each one matches the criterion in the prompt: "Targets are full returns \\(G_t = \\sum_{t\'=t}^T r_{t\'}\\)."; "It is unbiased."; "It suffers from high variance."; "It does not rely on bootstrapped predictions.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "cs224r-lect4-q06",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements correctly describe temporal difference learning?",
    options: [
      { text: "Targets use \\(r_t + \\gamma V(s_{t+1})\\).", isCorrect: true },
      {
        text: "It bootstraps from the current value estimate.",
        isCorrect: true,
      },
      { text: "It reduces variance relative to Monte Carlo.", isCorrect: true },
      {
        text: "It introduces bias when value estimates are inaccurate.",
        isCorrect: true,
      },
    ],
    explanation:
      'Temporal difference uses the model’s own prediction as part of the target. This reduces variance but can propagate estimation errors. To reason through the choices, select every statement because each one matches the criterion in the prompt: "Targets use \\(r_t + \\gamma V(s_{t+1})\\)."; "It bootstraps from the current value estimate."; "It reduces variance relative to Monte Carlo."; "It introduces bias when value estimates are inaccurate.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "cs224r-lect4-q07",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements correctly describe n-step returns?",
    options: [
      {
        text: "\\(y_t = \\sum_{k=0}^{n-1} \\gamma^k r_{t+k} + \\gamma^n V(s_{t+n})\\).",
        isCorrect: true,
      },
      { text: "They interpolate between Monte Carlo and TD.", isCorrect: true },
      { text: "Smaller \\(n\\) gives lower variance.", isCorrect: true },
      { text: "Larger \\(n\\) reduces bias.", isCorrect: true },
    ],
    explanation:
      'n-step returns trade off bias and variance. They combine actual rewards for several steps with bootstrapped value predictions. To reason through the choices, select every statement because each one matches the criterion in the prompt: "\\(y_t = \\sum_{k=0}^{n-1} \\gamma^k r_{t+k} + \\gamma^n V(s_{t+n})\\)."; "They interpolate between Monte Carlo and TD."; "Smaller \\(n\\) gives lower variance."; "Larger \\(n\\) reduces bias.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "cs224r-lect4-q08",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about discount factor \\(\\gamma\\) are correct?",
    options: [
      { text: "It weights future rewards exponentially.", isCorrect: true },
      {
        text: "It prevents divergence for infinite horizons.",
        isCorrect: true,
      },
      {
        text: "It is equivalent to adding a termination probability \\(1-\\gamma\\).",
        isCorrect: true,
      },
      {
        text: "Transition probabilities are scaled by \\(\\gamma\\).",
        isCorrect: true,
      },
    ],
    explanation:
      'Discounting reduces the influence of distant rewards. It can be interpreted as a stochastic termination mechanism in the MDP. To reason through the choices, select every statement because each one matches the criterion in the prompt: "It weights future rewards exponentially."; "It prevents divergence for infinite horizons."; "It is equivalent to adding a termination probability \\(1-\\gamma\\)."; "Transition probabilities are scaled by \\(\\gamma\\).". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "cs224r-lect4-q09",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements about advantage-based policy updates are correct?",
    options: [
      {
        text: "Actions with positive advantage are reinforced.",
        isCorrect: true,
      },
      { text: "Advantage serves as a learned baseline.", isCorrect: true },
      { text: "It reduces gradient variance.", isCorrect: true },
      { text: "It still optimizes expected return.", isCorrect: true },
    ],
    explanation:
      'Advantage-centered updates improve stability while keeping the same optimization objective. To reason through the choices, select every statement because each one matches the criterion in the prompt: "Actions with positive advantage are reinforced."; "Advantage serves as a learned baseline."; "It reduces gradient variance."; "It still optimizes expected return.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  // ============================================================
  // THREE TRUE (Q10–Q18)
  // ============================================================

  {
    id: "cs224r-lect4-q10",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements about learning \\(V^\\pi\\) with supervised learning are correct?",
    options: [
      {
        text: "Training data pairs states with observed returns.",
        isCorrect: true,
      },
      {
        text: "The loss is typically \\(L = (V_\\phi(s) - y)^2\\).",
        isCorrect: true,
      },
      { text: "The labels remain fixed during training.", isCorrect: false },
      {
        text: "Multiple trajectories provide amortized supervision.",
        isCorrect: true,
      },
    ],
    explanation:
      'Value learning is framed as regression. Labels change when bootstrapping is used. To reason through the choices, select the statements that match the criterion in the prompt: "Training data pairs states with observed returns."; "The loss is typically \\(L = (V_\\phi(s) - y)^2\\)."; "Multiple trajectories provide amortized supervision.". Do not select statements that miss that criterion: "The labels remain fixed during training.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q11",
    chapter: 4,
    difficulty: "hard",
    prompt: "Why can Monte Carlo learning fail to generalize?",
    options: [
      {
        text: "It only uses rewards from the sampled trajectory.",
        isCorrect: true,
      },
      {
        text: "Similar states across trajectories are not linked.",
        isCorrect: true,
      },
      { text: "It propagates value across similar states.", isCorrect: false },
      { text: "It ignores cross-trajectory information.", isCorrect: true },
    ],
    explanation:
      'Monte Carlo treats each trajectory independently. It does not exploit shared structure between similar states. To reason through the choices, select the statements that match the criterion in the prompt: "It only uses rewards from the sampled trajectory."; "Similar states across trajectories are not linked."; "It ignores cross-trajectory information.". Do not select statements that miss that criterion: "It propagates value across similar states.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q12",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about bootstrapped targets are correct?",
    options: [
      { text: "They depend on current value estimates.", isCorrect: true },
      { text: "They are updated every gradient step.", isCorrect: true },
      { text: "They are unbiased.", isCorrect: false },
      { text: "They reduce variance.", isCorrect: true },
    ],
    explanation:
      'Bootstrapping lowers variance but introduces bias from imperfect predictions. To reason through the choices, select the statements that match the criterion in the prompt: "They depend on current value estimates."; "They are updated every gradient step."; "They reduce variance.". Do not select statements that miss that criterion: "They are unbiased.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q13",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about replay buffers are correct?",
    options: [
      { text: "They store transitions from past policies.", isCorrect: true },
      { text: "They enable off-policy learning.", isCorrect: true },
      { text: "They always improve stability.", isCorrect: false },
      { text: "They increase data efficiency.", isCorrect: true },
    ],
    explanation:
      'Replay buffers allow reuse of experience. However, they may introduce instability due to distribution mismatch. To reason through the choices, select the statements that match the criterion in the prompt: "They store transitions from past policies."; "They enable off-policy learning."; "They increase data efficiency.". Do not select statements that miss that criterion: "They always improve stability.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q14",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about off-policy actor updates are correct?",
    options: [
      { text: "They require importance sampling.", isCorrect: true },
      {
        text: "They can diverge if policies differ too much.",
        isCorrect: true,
      },
      { text: "Advantages remain accurate indefinitely.", isCorrect: false },
      { text: "KL constraints can help stabilize learning.", isCorrect: true },
    ],
    explanation:
      'Off-policy learning reweights samples but suffers if the new policy deviates significantly. To reason through the choices, select the statements that match the criterion in the prompt: "They require importance sampling."; "They can diverge if policies differ too much."; "KL constraints can help stabilize learning.". Do not select statements that miss that criterion: "Advantages remain accurate indefinitely.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q15",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements about actor–critic vs policy gradient are correct?",
    options: [
      { text: "Actor–critic learns what is good vs bad.", isCorrect: true },
      { text: "Policy gradient uses raw sampled returns.", isCorrect: true },
      { text: "Actor–critic removes the need for rewards.", isCorrect: false },
      { text: "Actor–critic improves sample usage.", isCorrect: true },
    ],
    explanation:
      'Actor–critic augments policy gradients with learned value estimates but still depends on rewards. To reason through the choices, select the statements that match the criterion in the prompt: "Actor–critic learns what is good vs bad."; "Policy gradient uses raw sampled returns."; "Actor–critic improves sample usage.". Do not select statements that miss that criterion: "Actor–critic removes the need for rewards.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q16",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about critic bias are correct?",
    options: [
      {
        text: "Errors in the critic propagate into the policy update.",
        isCorrect: true,
      },
      {
        text: "Biased value estimates lead to biased advantages.",
        isCorrect: true,
      },
      { text: "Critic bias affects gradient direction.", isCorrect: true },
      { text: "Critic bias does not affect learning.", isCorrect: false },
    ],
    explanation:
      'Since the policy relies on the critic’s predictions, systematic critic errors distort policy learning. To reason through the choices, select the statements that match the criterion in the prompt: "Errors in the critic propagate into the policy update."; "Biased value estimates lead to biased advantages."; "Critic bias affects gradient direction.". Do not select statements that miss that criterion: "Critic bias does not affect learning.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q17",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about n-step vs TD are correct?",
    options: [
      { text: "n-step uses more real rewards.", isCorrect: true },
      { text: "TD uses only one-step rewards.", isCorrect: true },
      { text: "n-step always has higher bias.", isCorrect: false },
      {
        text: "n-step can reduce variance compared to Monte Carlo.",
        isCorrect: true,
      },
    ],
    explanation:
      'n-step returns balance real rewards and bootstrapping, often yielding better bias-variance tradeoffs. To reason through the choices, select the statements that match the criterion in the prompt: "n-step uses more real rewards."; "TD uses only one-step rewards."; "n-step can reduce variance compared to Monte Carlo.". Do not select statements that miss that criterion: "n-step always has higher bias.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q18",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements about the actor are correct?",
    options: [
      { text: "It represents the policy.", isCorrect: true },
      { text: "It is updated via policy gradients.", isCorrect: true },
      { text: "It estimates values.", isCorrect: false },
      { text: "It depends on advantage estimates.", isCorrect: true },
    ],
    explanation:
      'The actor selects actions while the critic evaluates them. To reason through the choices, select the statements that match the criterion in the prompt: "It represents the policy."; "It is updated via policy gradients."; "It depends on advantage estimates.". Do not select statements that miss that criterion: "It estimates values.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  // ============================================================
  // TWO TRUE (Q19–Q27)
  // ============================================================

  {
    id: "cs224r-lect4-q19",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which are true about Monte Carlo vs TD?",
    options: [
      { text: "Monte Carlo is unbiased.", isCorrect: true },
      { text: "TD has lower variance.", isCorrect: true },
      { text: "TD is unbiased.", isCorrect: false },
      { text: "Monte Carlo has lower variance.", isCorrect: false },
    ],
    explanation:
      'Monte Carlo gives correct expectations but noisy estimates. TD trades bias for variance reduction. To reason through the choices, select the statements that match the criterion in the prompt: "Monte Carlo is unbiased."; "TD has lower variance.". Do not select statements that miss that criterion: "TD is unbiased."; "Monte Carlo has lower variance.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q20",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which are consequences of replay buffers?",
    options: [
      { text: "Distribution mismatch with current policy.", isCorrect: true },
      { text: "Higher sample efficiency.", isCorrect: true },
      { text: "Perfect unbiased gradients.", isCorrect: false },
      { text: "Guaranteed convergence.", isCorrect: false },
    ],
    explanation:
      'Replay buffers help reuse data but introduce off-policy bias. To reason through the choices, select the statements that match the criterion in the prompt: "Distribution mismatch with current policy."; "Higher sample efficiency.". Do not select statements that miss that criterion: "Perfect unbiased gradients."; "Guaranteed convergence.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q21",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements about discounting are correct?",
    options: [
      { text: "It prioritizes near-term rewards.", isCorrect: true },
      { text: "It prevents large value explosions.", isCorrect: true },
      { text: "It increases variance.", isCorrect: false },
      { text: "It removes future rewards.", isCorrect: false },
    ],
    explanation:
      'Discounting controls magnitude and stability of value estimates. To reason through the choices, select the statements that match the criterion in the prompt: "It prioritizes near-term rewards."; "It prevents large value explosions.". Do not select statements that miss that criterion: "It increases variance."; "It removes future rewards.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q22",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about TD targets are correct?",
    options: [
      { text: "They use predictions of \\(V(s_{t+1})\\).", isCorrect: true },
      { text: "They reduce variance.", isCorrect: true },
      { text: "They are unbiased.", isCorrect: false },
      { text: "They ignore rewards.", isCorrect: false },
    ],
    explanation:
      'TD leverages bootstrapped predictions but still includes immediate rewards. To reason through the choices, select the statements that match the criterion in the prompt: "They use predictions of \\(V(s_{t+1})\\)."; "They reduce variance.". Do not select statements that miss that criterion: "They are unbiased."; "They ignore rewards.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q23",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about KL constraints are correct?",
    options: [
      { text: "They limit policy deviation.", isCorrect: true },
      { text: "They stabilize off-policy updates.", isCorrect: true },
      { text: "They eliminate bias.", isCorrect: false },
      { text: "They remove need for importance weights.", isCorrect: false },
    ],
    explanation:
      'KL constraints control step size but do not fix estimation bias. To reason through the choices, select the statements that match the criterion in the prompt: "They limit policy deviation."; "They stabilize off-policy updates.". Do not select statements that miss that criterion: "They eliminate bias."; "They remove need for importance weights.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q24",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which are true about critic training?",
    options: [
      { text: "Uses regression loss.", isCorrect: true },
      { text: "Can take multiple gradient steps per batch.", isCorrect: true },
      { text: "Directly changes the policy.", isCorrect: false },
      { text: "Requires environment resets.", isCorrect: false },
    ],
    explanation:
      'Critic updates are independent supervised learning steps. To reason through the choices, select the statements that match the criterion in the prompt: "Uses regression loss."; "Can take multiple gradient steps per batch.". Do not select statements that miss that criterion: "Directly changes the policy."; "Requires environment resets.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q25",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements about advantages are correct?",
    options: [
      { text: "They guide actor updates.", isCorrect: true },
      { text: "They compare action to baseline.", isCorrect: true },
      { text: "They replace rewards.", isCorrect: false },
      { text: "They are constant.", isCorrect: false },
    ],
    explanation:
      'Advantages determine direction and magnitude of policy updates. To reason through the choices, select the statements that match the criterion in the prompt: "They guide actor updates."; "They compare action to baseline.". Do not select statements that miss that criterion: "They replace rewards."; "They are constant.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q26",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which are true about n-step bias/variance?",
    options: [
      { text: "Smaller n → lower variance.", isCorrect: true },
      { text: "Larger n → lower bias.", isCorrect: true },
      { text: "n-step always unbiased.", isCorrect: false },
      { text: "n-step always low variance.", isCorrect: false },
    ],
    explanation:
      'n controls bias-variance tradeoff. To reason through the choices, select the statements that match the criterion in the prompt: "Smaller n → lower variance."; "Larger n → lower bias.". Do not select statements that miss that criterion: "n-step always unbiased."; "n-step always low variance.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q27",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about actor-critic data use are correct?",
    options: [
      {
        text: "Can learn from partial progress trajectories.",
        isCorrect: true,
      },
      {
        text: "More efficient than sparse reward policy gradients.",
        isCorrect: true,
      },
      {
        text: "Requires full reward success trajectories only.",
        isCorrect: false,
      },
      { text: "Ignores failed rollouts.", isCorrect: false },
    ],
    explanation:
      'Critic extracts signal even from imperfect rollouts. To reason through the choices, select the statements that match the criterion in the prompt: "Can learn from partial progress trajectories."; "More efficient than sparse reward policy gradients.". Do not select statements that miss that criterion: "Requires full reward success trajectories only."; "Ignores failed rollouts.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  // ============================================================
  // ONE TRUE (Q28–Q35)
  // ============================================================

  {
    id: "cs224r-lect4-q28",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statement about the critic is correct?",
    options: [
      { text: "It selects actions.", isCorrect: false },
      { text: "It estimates state values.", isCorrect: true },
      { text: "It replaces the actor.", isCorrect: false },
      { text: "It removes need for rewards.", isCorrect: false },
    ],
    explanation:
      'The critic evaluates how good states are under the current policy. To reason through the choices, select the statements that match the criterion in the prompt: "It estimates state values.". Do not select statements that miss that criterion: "It selects actions."; "It replaces the actor."; "It removes need for rewards.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q29",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which best describes TD learning?",
    options: [
      { text: "Regression on full trajectory returns.", isCorrect: false },
      { text: "Bootstrap using next state value.", isCorrect: true },
      { text: "No supervision.", isCorrect: false },
      { text: "Pure imitation.", isCorrect: false },
    ],
    explanation:
      'TD uses one-step reward plus estimated value. To reason through the choices, select the statements that match the criterion in the prompt: "Bootstrap using next state value.". Do not select statements that miss that criterion: "Regression on full trajectory returns."; "No supervision."; "Pure imitation.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q30",
    chapter: 4,
    difficulty: "hard",
    prompt: "Why can replay-buffer actor updates be incorrect?",
    options: [
      { text: "Actions weren’t sampled from current policy.", isCorrect: true },
      { text: "Rewards are missing.", isCorrect: false },
      { text: "Value function unnecessary.", isCorrect: false },
      { text: "Environment deterministic.", isCorrect: false },
    ],
    explanation:
      'Policy gradients assume actions come from the current policy distribution. To reason through the choices, select the statements that match the criterion in the prompt: "Actions weren’t sampled from current policy.". Do not select statements that miss that criterion: "Rewards are missing."; "Value function unnecessary."; "Environment deterministic.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q31",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which method is unbiased but high variance?",
    options: [
      { text: "Temporal difference.", isCorrect: false },
      { text: "Monte Carlo.", isCorrect: true },
      { text: "Replay buffers.", isCorrect: false },
      { text: "Advantage normalization.", isCorrect: false },
    ],
    explanation:
      'Monte Carlo uses full returns directly. To reason through the choices, select the statements that match the criterion in the prompt: "Monte Carlo.". Do not select statements that miss that criterion: "Temporal difference."; "Replay buffers."; "Advantage normalization.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q32",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statement about discount factor is correct?",
    options: [
      { text: "Gamma greater than one increases stability.", isCorrect: false },
      {
        text: "Gamma less than one reduces long-term reward weight.",
        isCorrect: true,
      },
      { text: "Gamma eliminates variance.", isCorrect: false },
      { text: "Gamma only affects policy network.", isCorrect: false },
    ],
    explanation:
      'Discounting exponentially shrinks distant rewards. To reason through the choices, select the statements that match the criterion in the prompt: "Gamma less than one reduces long-term reward weight.". Do not select statements that miss that criterion: "Gamma greater than one increases stability."; "Gamma eliminates variance."; "Gamma only affects policy network.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q33",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statement about critic bias is correct?",
    options: [
      { text: "Bias has no effect.", isCorrect: false },
      { text: "Bias affects policy updates.", isCorrect: true },
      { text: "Bias only changes rewards.", isCorrect: false },
      { text: "Bias improves exploration.", isCorrect: false },
    ],
    explanation:
      'Actor relies directly on critic estimates. To reason through the choices, select the statements that match the criterion in the prompt: "Bias affects policy updates.". Do not select statements that miss that criterion: "Bias has no effect."; "Bias only changes rewards."; "Bias improves exploration.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q34",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which best describes n-step returns?",
    options: [
      { text: "Only one-step bootstrap.", isCorrect: false },
      { text: "Full trajectory sum.", isCorrect: false },
      { text: "Partial reward sum plus bootstrap.", isCorrect: true },
      { text: "No rewards used.", isCorrect: false },
    ],
    explanation:
      'n-step mixes rewards and predicted values. To reason through the choices, select the statements that match the criterion in the prompt: "Partial reward sum plus bootstrap.". Do not select statements that miss that criterion: "Only one-step bootstrap."; "Full trajectory sum."; "No rewards used.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "cs224r-lect4-q35",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statement about actor–critic efficiency is correct?",
    options: [
      { text: "Less efficient than policy gradients.", isCorrect: false },
      {
        text: "Uses learned value estimates for better gradients.",
        isCorrect: true,
      },
      { text: "Does not require learning.", isCorrect: false },
      { text: "Always unstable.", isCorrect: false },
    ],
    explanation:
      'The critic reduces variance and improves data use. To reason through the choices, select the statements that match the criterion in the prompt: "Uses learned value estimates for better gradients.". Do not select statements that miss that criterion: "Less efficient than policy gradients."; "Does not require learning."; "Always unstable.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },
];
