import { Question } from "../../quiz";

// lib/lectures/Stanford CME295 Transformers & LLMs/lecture5_Off-Policy Actor Critic Methods.ts

export const lecture5_OffPolicyActorCriticQuestions: Question[] = [
  // 1 (Easy) — 1 true
  {
    id: "cs224r-lect5-q01",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "In off-policy actor-critic, what does an importance ratio like \\(w_t = \\frac{\\pi_{\\theta'}(a_t\\mid s_t)}{\\pi_{\\theta}(a_t\\mid s_t)}\\) primarily do?",
    options: [
      {
        text: "It reweights data collected under \\(\\pi_{\\theta}\\) to estimate gradients for \\(\\pi_{\\theta'}\\).",
        isCorrect: true,
      },
      {
        text: "It converts a stochastic policy into a deterministic greedy policy.",
        isCorrect: false,
      },
      {
        text: "It replaces the need to estimate advantages \\(\\hat A\\) entirely.",
        isCorrect: false,
      },
      {
        text: "It makes the learning signal independent of how the data was collected.",
        isCorrect: false,
      },
    ],
    explanation:
      "Importance sampling reweights samples from a behavior policy \\(\\pi_{\\theta}\\) so we can form an estimator for the target policy \\(\\pi_{\\theta'}\\). It does not remove the need for advantage/value estimation, and it does not magically eliminate distribution shift when policies differ a lot.",
  },

  // 2 (Easy) — 2 true
  {
    id: "cs224r-lect5-q02",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements about the advantage function \\(A^{\\pi}(s,a)=Q^{\\pi}(s,a)-V^{\\pi}(s)\\) are correct?",
    options: [
      {
        text: "\\(A^{\\pi}(s,a)\\) measures how much better action \\(a\\) is than the policy’s average action at \\(s\\).",
        isCorrect: true,
      },
      {
        text: "If \\(A^{\\pi}(s,a) > 0\\), increasing \\(\\pi(a\\mid s)\\) tends to improve the objective locally (for that \\(\\pi\\)).",
        isCorrect: true,
      },
      {
        text: "\\(A^{\\pi}(s,a)\\) is defined without reference to any policy \\(\\pi\\).",
        isCorrect: false,
      },
      {
        text: "\\(A^{\\pi}(s,a)\\) must be nonnegative for all \\((s,a)\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Advantage compares an action’s expected return to the policy’s baseline value at that state. Positive advantage means the action is better than the policy’s typical behavior at \\(s\\), so pushing probability mass toward it is beneficial for small updates under that same policy.",
  },

  // 3 (Easy) — 3 true
  {
    id: "cs224r-lect5-q03",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Why can taking many gradient steps on a single batch in off-policy actor-critic become unstable?",
    options: [
      {
        text: "The advantage estimates \\(\\hat A\\) are computed under the old policy and can become outdated as the policy changes.",
        isCorrect: true,
      },
      {
        text: "The surrogate objective can incentivize making the importance ratio \\(\\pi_{\\theta'}(a\\mid s)/\\pi_{\\theta}(a\\mid s)\\) very large for some samples.",
        isCorrect: true,
      },
      {
        text: "Over-optimizing on a finite batch can cause overfitting to that batch’s idiosyncrasies (poor generalization to new rollouts).",
        isCorrect: true,
      },
      {
        text: "Automatic differentiation (e.g., PyTorch) introduces bias that grows with the number of gradient steps.",
        isCorrect: false,
      },
    ],
    explanation:
      "The core issue is mismatch: you keep updating the policy while keeping \\(\\hat A\\) fixed from old data, so the learning signal stops matching the current policy. Because the ratio can be pushed far from 1, the policy can overfit the batch and become unstable.",
  },

  // 4 (Easy) — all true
  {
    id: "cs224r-lect5-q04",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Consider the surrogate objective form \\(\\tilde J(\\theta') = \\sum_{i,t} \\frac{\\pi_{\\theta'}(a_{i,t}\\mid s_{i,t})}{\\pi_{\\theta}(a_{i,t}\\mid s_{i,t})} \\hat A_{\\pi_{\\theta}}(s_{i,t},a_{i,t})\\). Which statements are correct?",
    options: [
      {
        text: "It is constructed so that differentiating w.r.t. \\(\\theta'\\) yields the off-policy policy-gradient estimator with importance weights.",
        isCorrect: true,
      },
      {
        text: "The denominator \\(\\pi_{\\theta}(a\\mid s)\\) is treated as a constant when optimizing over \\(\\theta'\\).",
        isCorrect: true,
      },
      {
        text: "Large positive advantages encourage increasing \\(\\pi_{\\theta'}(a\\mid s)\\) relative to \\(\\pi_{\\theta}(a\\mid s)\\).",
        isCorrect: true,
      },
      {
        text: "If advantages stay accurate for the updated policy, maximizing \\(\\tilde J\\) can improve expected return.",
        isCorrect: true,
      },
    ],
    explanation:
      "This surrogate is a practical way to get the desired gradient via autodiff while reusing behavior-policy data. The denominator is fixed because it came from the data-collection policy, and positive \\(\\hat A\\) pushes the ratio upward for those actions.",
  },

  // 5 (Medium) — 1 true
  {
    id: "cs224r-lect5-q05",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "In PPO-style clipping, what does it mean to clip an importance weight \\(w = \\frac\\pi{\\pi_\\text{old}}\\) to \\([1-\\epsilon,1+\\epsilon]\\)?",
    options: [
      {
        text: "The objective stops giving additional incentive to increase (or decrease) \\(w\\) once it leaves \\([1-\\epsilon,1+\\epsilon]\\).",
        isCorrect: true,
      },
      {
        text: "The policy parameters are projected back to a KL ball after each gradient step.",
        isCorrect: false,
      },
      {
        text: "The advantage estimates \\(\\hat A\\) are rescaled so their magnitudes fall in \\([1-\\epsilon,1+\\epsilon]\\).",
        isCorrect: false,
      },
      {
        text: "The replay buffer is truncated to keep only transitions whose \\(w\\) lies in \\([1-\\epsilon,1+\\epsilon]\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Clipping modifies the surrogate so extreme ratios no longer improve the objective beyond a bound. It is not a projection step, and it does not change the replay buffer or directly rescale \\(\\hat A\\).",
  },

  // 6 (Medium) — 2 true
  {
    id: "cs224r-lect5-q06",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Why does PPO take a per-sample minimum between the unclipped and clipped objectives?",
    options: [
      {
        text: "It ensures the clipped objective is a lower bound on the unclipped surrogate, preventing clipping from artificially improving the objective.",
        isCorrect: true,
      },
      {
        text: "It avoids cases where clipping would reduce the penalty for a badly-updated sample (e.g., huge ratio with negative advantage).",
        isCorrect: true,
      },
      {
        text: "It makes the policy update exactly unbiased even for very large policy changes.",
        isCorrect: false,
      },
      {
        text: "It removes the need to tune \\(\\epsilon\\) because the minimum adapts automatically.",
        isCorrect: false,
      },
    ],
    explanation:
      "Taking the minimum prevents the optimizer from ‘benefiting’ from clipping in pathological cases where clipping would otherwise raise the objective. It improves stability, but it does not guarantee unbiasedness for large deviations and still depends on \\(\\epsilon\\).",
  },

  // 7 (Medium) — 3 true
  {
    id: "cs224r-lect5-q07",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statements about KL-penalized policy updates are correct?",
    options: [
      {
        text: "Adding \\(-\\beta\\,D_{\\mathrm{KL}}(\\pi_{\\theta'}(\\cdot\\mid s)\\|\\pi_{\\theta}(\\cdot\\mid s))\\) discourages large changes to the action distribution.",
        isCorrect: true,
      },
      {
        text: "The coefficient \\(\\beta\\) controls the tradeoff between improving the surrogate and staying close to the old policy.",
        isCorrect: true,
      },
      {
        text: "Keeping the policy close helps because advantage estimates computed under the old policy are more likely to remain valid.",
        isCorrect: true,
      },
      {
        text: "A KL penalty guarantees monotonic improvement in expected return for any \\(\\beta\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "KL penalties are a common way to constrain update size in distribution space. They help with stale advantages by limiting drift, but they do not provide a universal monotonic-improvement guarantee without additional conditions/analysis.",
  },

  // 8 (Medium) — all true
  {
    id: "cs224r-lect5-q08",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Generalized Advantage Estimation (GAE) can be_toggle as combining multiple horizons. Which statements are correct?",
    options: [
      {
        text: "GAE mixes multi-step temporal-difference residuals to trade off bias and variance in \\(\\hat A\\).",
        isCorrect: true,
      },
      {
        text: "Smaller effective horizons typically reduce variance but may increase bias if the value function is imperfect.",
        isCorrect: true,
      },
      {
        text: "Discount \\(\\gamma\\) and the GAE mixing parameter (often \\(\\lambda\\)) appear together in weighting future terms (e.g., \\((\\gamma\\lambda)^{k}\\)).",
        isCorrect: true,
      },
      {
        text: "GAE is used to estimate advantages even if the value function \\(\\hat V\\) is trained with Monte Carlo or bootstrapped targets.",
        isCorrect: true,
      },
    ],
    explanation:
      "GAE constructs \\(\\hat A\\) by exponentially weighting TD residuals, commonly involving \\(\\gamma\\lambda\\). It’s compatible with different value-fitting approaches; the key is that advantage estimation uses a controlled mix of horizons.",
  },

  // 9 (Easy) — 3 true
  {
    id: "cs224r-lect5-q09",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "A replay buffer in off-policy reinforcement learning is best described by which statements?",
    options: [
      {
        text: "It stores past transitions \\((s,a,r,s')\\) from (potentially) many earlier policies.",
        isCorrect: true,
      },
      {
        text: "Training updates sample minibatches from the buffer rather than only using the most recent on-policy batch.",
        isCorrect: true,
      },
      {
        text: "It enables much higher data reuse, often improving sample efficiency.",
        isCorrect: true,
      },
      {
        text: "It guarantees that the state distribution matches the current policy’s on-policy visitation distribution.",
        isCorrect: false,
      },
    ],
    explanation:
      "Replay buffers let you reuse older experience by sampling random minibatches of transitions. The buffer’s state distribution is not necessarily the current on-policy distribution; that mismatch is part of what makes off-policy learning both powerful and tricky.",
  },

  // 10 (Medium) — 1 true
  {
    id: "cs224r-lect5-q10",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "If you train a value function \\(\\hat V\\) using bootstrapped targets \\(y = r + \\gamma\\hat V(s')\\) on data sampled uniformly from a replay buffer, which policy’s value are you most directly approximating?",
    options: [
      {
        text: "A complicated mixture of past behavior policies represented in the buffer, not strictly the current policy.",
        isCorrect: true,
      },
      {
        text: "Exactly the current policy \\(\\pi_{\\theta}\\), because \\(\\hat V\\) is a function only of state.",
        isCorrect: false,
      },
      {
        text: "Exactly the oldest policy in the buffer, because its data dominates early on.",
        isCorrect: false,
      },
      {
        text: "A policy-independent quantity that depends only on environment dynamics.",
        isCorrect: false,
      },
    ],
    explanation:
      "When transitions come from many behavior policies, the induced distribution over next states/actions reflects that history. Bootstrapping \\(\\hat V\\) on the buffer does not cleanly correspond to \\(V^{\\pi_{\\theta}}\\) unless the data distribution matches the current policy’s distribution.",
  },

  // 11 (Hard) — 2 true
  {
    id: "cs224r-lect5-q11",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "To make replay-buffer actor-critic less ‘broken’, the lecture suggests fitting \\(Q^{\\pi}(s,a)\\) instead of \\(V^{\\pi}(s)\\). Which statements are correct?",
    options: [
      {
        text: "Because \\(Q^{\\pi}(s,a)\\) conditions on \\(a\\), it is acceptable that the stored action came from an older policy.",
        isCorrect: true,
      },
      {
        text: "A Bellman-style target can use \\(a'\\sim\\pi_{\\theta}(\\cdot\\mid s')\\) even when \\((s,a,r,s')\\) came from the buffer.",
        isCorrect: true,
      },
      {
        text: "Fitting \\(Q\\) eliminates the need for any coverage of the action space by the replay buffer.",
        isCorrect: false,
      },
      {
        text: "Using \\(Q\\) makes the resulting policy update fully on-policy again.",
        isCorrect: false,
      },
    ],
    explanation:
      "Conditioning on \\(a\\) allows off-policy actions in the dataset to still be valid inputs to the critic. The key trick is to bootstrap with next actions sampled from the current policy, but you still need reasonable coverage; otherwise the critic learns on out-of-distribution actions/states.",
  },

  // 12 (Hard) — 3 true
  {
    id: "cs224r-lect5-q12",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Consider the Bellman-style relation for a fixed policy \\(\\pi_{\\theta}\\): \\n\\(Q^{\\pi_{\\theta}}(s,a) = r(s,a) + \\gamma\\,\\mathbb{E}_{s'\\sim p(\\cdot\\mid s,a),\\,a'\\sim\\pi_{\\theta}(\\cdot\\mid s')}[Q^{\\pi_{\\theta}}(s',a')]\\). Which statements are correct?",
    options: [
      {
        text: "It holds for any \\((s,a)\\) pair, not only those selected by the current policy at \\(s\\).",
        isCorrect: true,
      },
      {
        text: "Using a single sampled \\(s'\\) from the replay buffer is a Monte Carlo approximation of the expectation over next states.",
        isCorrect: true,
      },
      {
        text: "Sampling \\(a'\\) from the current policy at \\(s'\\) targets the current policy’s \\(Q\\) rather than the behavior policy’s future actions.",
        isCorrect: true,
      },
      {
        text: "The equation implies \\(Q^{\\pi_{\\theta}}(s,a)\\) can be estimated accurately even if \\(s\\) is never observed in data.",
        isCorrect: false,
      },
    ],
    explanation:
      "The Bellman equation is policy-defined and applies to any \\((s,a)\\). In practice you approximate expectations using sampled next states from the buffer and next actions from the current policy, but you still need data coverage of relevant states (or generalization) to learn meaningful values.",
  },

  // 13 (Hard) — all true
  {
    id: "cs224r-lect5-q13",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "A practical off-policy critic update described in the lecture resembles supervised regression with target \\(y_i = r_i + \\gamma\\,\\hat Q(s'_i, a'_i)\\) where \\(a'_i\\sim\\pi_{\\theta}(\\cdot\\mid s'_i)\\). Which statements are correct?",
    options: [
      {
        text: "The input to the critic network is \\((s_i,a_i)\\), where \\(a_i\\) may come from an older behavior policy stored in the replay buffer.",
        isCorrect: true,
      },
      {
        text: "The target uses \\(a'_i\\) sampled from the current policy, not necessarily the buffer’s next action, to better match \\(Q^{\\pi_{\\theta}}\\).",
        isCorrect: true,
      },
      {
        text: "This update is a form of bootstrapping (temporal-difference learning) because the target depends on the current critic’s estimate.",
        isCorrect: true,
      },
      {
        text: "Accuracy of \\(y_i\\) depends on having enough coverage of actions/states relevant to the current policy (or strong generalization).",
        isCorrect: true,
      },
    ],
    explanation:
      "This is a TD-style regression objective: fit \\(\\hat Q(s,a)\\) to a bootstrapped label that uses \\(\\hat Q\\) at the next state. The crucial off-policy detail is that next actions come from the current policy, while current actions can be older—yet the update still needs reasonable distribution coverage.",
  },

  // 14 (Medium) — 2 true
  {
    id: "cs224r-lect5-q14",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "In the replay-buffer variant, why might using \\(\\hat Q\\) directly in the actor update be appealing compared to using \\(\\hat A\\)?",
    options: [
      {
        text: "It avoids needing a separate baseline \\(\\hat V\\) (or average reward baseline) to form \\(\\hat A\\).",
        isCorrect: true,
      },
      {
        text: "Higher variance can be tolerable because replay buffers provide many more training samples overall.",
        isCorrect: true,
      },
      {
        text: "It makes the policy gradient unbiased even if \\(\\hat Q\\) is poorly estimated.",
        isCorrect: false,
      },
      {
        text: "It guarantees the policy will remain close to the behavior policies in the buffer.",
        isCorrect: false,
      },
    ],
    explanation:
      "Using \\(\\hat Q\\) can simplify the pipeline by not requiring a baseline to construct advantages. The tradeoff is often higher variance, but the buffer provides lots of data, which can compensate; it does not guarantee unbiasedness or policy closeness by itself.",
  },

  // 15 (Medium) — 1 true
  {
    id: "cs224r-lect5-q15",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "In the replay-buffer actor step described, what action is used inside \\(\\nabla_\\theta \\log \\pi_{\\theta}(a\\mid s)\\,\\hat Q(s,a)\\)?",
    options: [
      {
        text: "An action \\(a\\) sampled from the current policy \\(a\\sim\\pi_{\\theta}(\\cdot\\mid s)\\), not necessarily the buffer’s stored action.",
        isCorrect: true,
      },
      {
        text: "The buffer’s stored action \\(a\\) is always used, because otherwise gradients cannot be computed.",
        isCorrect: false,
      },
      {
        text: "A greedy action \\(a=\\arg\\max_a \\hat Q(s,a)\\) is always used to reduce variance.",
        isCorrect: false,
      },
      {
        text: "An action sampled uniformly at random to encourage exploration.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture’s fix samples the actor’s action from the current policy at the current state, then evaluates it with the critic. This mirrors the same idea used in the critic target: use buffer states, but sample current-policy actions when you need actions that should reflect the current policy.",
  },

  // 16 (Easy) — all true
  {
    id: "cs224r-lect5-q16",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements about on-policy vs. off-policy learning in this lecture’s framing are correct?",
    options: [
      {
        text: "On-policy updates use data collected from the same policy being updated (or very close to it).",
        isCorrect: true,
      },
      {
        text: "Off-policy methods aim to reuse data collected from older or different behavior policies.",
        isCorrect: true,
      },
      {
        text: "Importance sampling is one mechanism to correct for differences between behavior and target policies.",
        isCorrect: true,
      },
      {
        text: "Replay buffers are a common tool to enable large-scale data reuse across many updates.",
        isCorrect: true,
      },
    ],
    explanation:
      "On-policy learning ties data collection tightly to the current policy, limiting reuse. Off-policy learning breaks that link via corrections (importance weights) and/or structural changes (replay buffers), enabling much more sample reuse.",
  },

  // 17 (Easy) — 2 true
  {
    id: "cs224r-lect5-q17",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which of the following are reasonable interpretations of why PPO is called ‘proximal’?",
    options: [
      {
        text: "It aims to keep the new policy distribution relatively close to the old one during updates.",
        isCorrect: true,
      },
      {
        text: "Clipping (or KL penalties) reduces incentives for the policy to move too far in one update.",
        isCorrect: true,
      },
      {
        text: "It guarantees the policy parameters move by a fixed Euclidean distance each step.",
        isCorrect: false,
      },
      {
        text: "It forces the action probabilities to remain unchanged for all states.",
        isCorrect: false,
      },
    ],
    explanation:
      "‘Proximal’ refers to restricting how drastically the policy distribution changes during optimization. PPO does this implicitly via clipping or explicitly via KL penalties, but it is not a fixed Euclidean step rule and does not freeze probabilities.",
  },

  // 18 (Hard) — 1 true
  {
    id: "cs224r-lect5-q18",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Suppose \\(\\hat A(s,a) < 0\\) and the ratio \\(w = \\frac{\\pi_{\\theta'}(a\\mid s)}{\\pi_{\\theta}(a\\mid s)}\\) becomes very large (e.g., \\(w=9\\)). Why can PPO’s ‘min with unclipped’ trick matter here?",
    options: [
      {
        text: "Because clipping \\(w\\) down (e.g., to \\(1+\\epsilon\\)) could make the objective less negative, unintentionally making the objective larger than the unclipped one for this sample.",
        isCorrect: true,
      },
      {
        text: "Because clipping always makes gradients unbiased, but only if advantages are negative.",
        isCorrect: false,
      },
      {
        text: "Because taking the minimum forces \\(w\\) to equal exactly 1 for negative advantages.",
        isCorrect: false,
      },
      {
        text: "Because it replaces \\(\\hat A\\) with \\(-\\hat A\\) to stabilize learning.",
        isCorrect: false,
      },
    ],
    explanation:
      "If \\(\\hat A<0\\), then reducing \\(w\\) via clipping can make \\(w\\hat A\\) less negative, which increases the objective for that sample. The per-sample minimum prevents the optimizer from getting a ‘free boost’ from clipping in such cases.",
  },

  // 19 (Medium) — all true
  {
    id: "cs224r-lect5-q19",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about ‘data efficiency’ vs. ‘stability’ tradeoffs between PPO and SAC (as described here) are correct?",
    options: [
      {
        text: "Replay-buffer off-policy methods (e.g., SAC-like) can be far more data efficient due to extensive reuse of past experience.",
        isCorrect: true,
      },
      {
        text: "These more off-policy methods can be harder to tune and may be less stable than PPO in practice.",
        isCorrect: true,
      },
      {
        text: "PPO is often used when stable learning is prioritized, especially when data is cheap (e.g., simulation).",
        isCorrect: true,
      },
      {
        text: "PPO is commonly cited as a practical choice in reinforcement learning for language models.",
        isCorrect: true,
      },
    ],
    explanation:
      "SAC-style replay-buffer learning reuses old experience heavily, improving sample efficiency but increasing sensitivity to hyperparameters and critic errors. PPO’s clipped (or KL-constrained) updates are often more stable, and PPO is widely used in RLHF-style settings.",
  },

  // 20 (Hard) — 2 true
  {
    id: "cs224r-lect5-q20",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "The replay-buffer actor-critic still has a mismatch: buffer states \\(s\\) are not sampled from the current on-policy state distribution \\(p_{\\pi_{\\theta}}(s)\\). Which statements are correct?",
    options: [
      {
        text: "This mismatch cannot be fully ‘fixed’ by a simple action correction, because it is about which states are visited, not just actions chosen.",
        isCorrect: true,
      },
      {
        text: "Training on a broader state distribution can sometimes be beneficial, yielding a policy that performs well on more states than it currently visits.",
        isCorrect: true,
      },
      {
        text: "This mismatch means the algorithm is equivalent to on-policy actor-critic after enough updates.",
        isCorrect: false,
      },
      {
        text: "The mismatch disappears automatically if you set \\(\\gamma=1\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Off-policy replay buffers change the state distribution used for updates. You can sample current-policy actions for those states, but you cannot undo the fact that those states came from older behaviors; sometimes the broader distribution can be helpful, but it is not ‘on-policy again.’",
  },

  // 21 (Easy) — 1 true
  {
    id: "cs224r-lect5-q21",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "In the lecture’s off-policy actor-critic ‘Version 1’ (multiple gradient steps), what is the key reason importance weights are used?",
    options: [
      {
        text: "To estimate the gradient for the updated policy \\(\\pi_{\\theta'}\\) using data collected under \\(\\pi_{\\theta}\\).",
        isCorrect: true,
      },
      {
        text: "To ensure the value function \\(\\hat V\\) never needs to be retrained.",
        isCorrect: false,
      },
      {
        text: "To turn Monte Carlo value targets into temporal-difference targets.",
        isCorrect: false,
      },
      {
        text: "To remove the need for a stochastic policy during training.",
        isCorrect: false,
      },
    ],
    explanation:
      "Importance weights are the mechanism that lets you reuse the same batch for multiple updates by correcting for the policy change. They do not remove the need to fit \\(\\hat V\\), and they are unrelated to Monte Carlo vs. TD targets.",
  },

  // 22 (Hard) — 3 true
  {
    id: "cs224r-lect5-q22",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements about ‘action coverage’ for the replay-buffer \\(Q\\)-learning-style target \\(y=r+\\gamma\\hat Q(s',a')\\) are correct?",
    options: [
      {
        text: "If the buffer rarely contains transitions near actions the current policy would take, \\(\\hat Q\\) may be inaccurate in the relevant regions.",
        isCorrect: true,
      },
      {
        text: "Neural network generalization can help interpolate between seen actions, but it is not a substitute for any coverage at all.",
        isCorrect: true,
      },
      {
        text: "Keeping the policy from changing too quickly can help maintain overlap between buffer actions and current-policy actions.",
        isCorrect: true,
      },
      {
        text: "If you sample \\(a'\\) from the current policy, action coverage becomes irrelevant.",
        isCorrect: false,
      },
    ],
    explanation:
      "Even though \\(a'\\) is sampled from the current policy, the critic still learns from buffer transitions and must generalize to the policy’s action regions. Without sufficient overlap/coverage, the critic’s bootstrapped targets can become unreliable and destabilize learning.",
  },

  // 23 (Easy) — 2 true
  {
    id: "cs224r-lect5-q23",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements about the clipping range \\(\\epsilon\\) in PPO are correct?",
    options: [
      {
        text: "Smaller \\(\\epsilon\\) usually constrains policy updates more tightly by keeping ratios closer to 1.",
        isCorrect: true,
      },
      {
        text: "Larger \\(\\epsilon\\) generally allows more aggressive policy changes on each batch.",
        isCorrect: true,
      },
      {
        text: "Changing \\(\\epsilon\\) changes the reward function of the environment.",
        isCorrect: false,
      },
      {
        text: "With clipping, the importance ratio is forced to equal 1 exactly for all samples.",
        isCorrect: false,
      },
    ],
    explanation:
      "The clipping range bounds how far the ratio can move before the surrogate stops improving due to that movement. It’s a policy-update constraint, not an environment modification, and ratios can still vary within \\([1-\\epsilon,1+\\epsilon]\\).",
  },

  // 24 (Medium) — 3 true
  {
    id: "cs224r-lect5-q24",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Why can off-policy methods be ‘more data efficient’ than on-policy ones, in the sense used in these lectures?",
    options: [
      {
        text: "They can take many learning updates per unit of collected experience by reusing old transitions.",
        isCorrect: true,
      },
      {
        text: "Replay buffers let training sample from a large pool of past experience rather than discarding it after one update.",
        isCorrect: true,
      },
      {
        text: "They can improve reward faster as a function of environment timesteps, even if wall-clock compute may increase.",
        isCorrect: true,
      },
      {
        text: "They always require fewer gradient steps because each update is perfectly accurate.",
        isCorrect: false,
      },
    ],
    explanation:
      "Data efficiency here refers to performance vs. environment interaction steps: reuse means more learning signal per collected transition. That does not mean fewer gradient steps; often it means more gradient steps per timestep, which can cost more compute.",
  },

  // 25 (Hard) — all true
  {
    id: "cs224r-lect5-q25",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Suppose PPO uses the clipped surrogate \\(\\min(w\\hat A, \\mathrm{clip}(w,1-\\epsilon,1+\\epsilon)\\hat A)\\) per sample. Which statements are correct?",
    options: [
      {
        text: "If \\(\\hat A>0\\) and \\(w>1+\\epsilon\\), the clipped term becomes \\((1+\\epsilon)\\hat A\\), limiting incentive to increase \\(w\\) further.",
        isCorrect: true,
      },
      {
        text: "If \\(\\hat A<0\\) and \\(w<1-\\epsilon\\), clipping can limit incentive to decrease \\(w\\) further in that direction.",
        isCorrect: true,
      },
      {
        text: "Taking the minimum ensures the chosen per-sample objective is never larger than the unclipped objective for that sample.",
        isCorrect: true,
      },
      {
        text: "The mechanism aims to prevent overly large policy updates driven by extreme ratios on a finite batch.",
        isCorrect: true,
      },
    ],
    explanation:
      "Clipping flattens the objective outside a ratio band, preventing runaway incentives. The per-sample minimum maintains a conservative (lower-bound) objective relative to the unclipped surrogate, improving stability when ratios become extreme.",
  },

  // 26 (Medium) — 2 true
  {
    id: "cs224r-lect5-q26",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about the ‘surrogate objective’ vs. the ‘true objective’ are correct?",
    options: [
      {
        text: "The true objective \\(J(\\theta)\\) is the expected (discounted) return under \\(\\pi_{\\theta}\\), which is typically intractable to compute exactly.",
        isCorrect: true,
      },
      {
        text: "A surrogate objective \\(\\tilde J\\) is designed so its gradient matches (or approximates) the policy-gradient estimator used in practice.",
        isCorrect: true,
      },
      {
        text: "The surrogate objective is always exactly equal to the true objective for all \\(\\theta\\).",
        isCorrect: false,
      },
      {
        text: "Using a surrogate objective removes the need for stochastic sampling from the policy.",
        isCorrect: false,
      },
    ],
    explanation:
      "Surrogates are optimization conveniences: they’re built to yield desired gradients under sampling, not to be exactly equal to expected return. They still rely on stochastic sampling and approximate value/advantage estimates.",
  },

  // 27 (Easy) — all true
  {
    id: "cs224r-lect5-q27",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "In the lecture’s summary of PPO, which steps are part of the high-level loop?",
    options: [
      {
        text: "Collect a batch of trajectories (or timesteps) using the current policy.",
        isCorrect: true,
      },
      {
        text: "Fit a value function (critic) using returns or bootstrapped targets on that batch.",
        isCorrect: true,
      },
      {
        text: "Compute advantage estimates (often with GAE) from rewards and the fitted value function.",
        isCorrect: true,
      },
      {
        text: "Update the policy for multiple epochs on the collected batch using the clipped PPO surrogate.",
        isCorrect: true,
      },
    ],
    explanation:
      "PPO alternates between data collection, critic fitting, advantage computation, and multiple policy updates on the same batch. The clipping (and sometimes KL) is what enables multiple stable epochs on one batch.",
  },

  // 28 (Hard) — 1 true
  {
    id: "cs224r-lect5-q28",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "In the replay-buffer ‘Version 2’, why is estimating \\(V^{\\pi}(s)\\) from buffer data conceptually problematic?",
    options: [
      {
        text: "Because the buffer contains trajectories from many policies, the future evolution after \\(s\\) in the data does not reflect following the current policy \\(\\pi\\) thereafter.",
        isCorrect: true,
      },
      {
        text: "Because \\(V^{\\pi}(s)\\) is undefined unless the environment is deterministic.",
        isCorrect: false,
      },
      {
        text: "Because \\(V^{\\pi}(s)\\) cannot be represented by a neural network.",
        isCorrect: false,
      },
      {
        text: "Because value functions require access to the policy’s gradients, which the buffer does not store.",
        isCorrect: false,
      },
    ],
    explanation:
      "The issue is policy mismatch: the data’s continuation from \\(s\\) reflects older policies’ actions, not the current one. That makes a naive \\(V\\)-target built from buffer rollouts correspond to some messy mixture, not \\(V^{\\pi_{\\theta}}\\).",
  },

  // 29 (Medium) — 3 true
  {
    id: "cs224r-lect5-q29",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about how SAC-style updates use replay-buffer transitions are correct (at the level described in this lecture)?",
    options: [
      {
        text: "Transitions \\((s,a,r,s')\\) are sampled from the replay buffer for critic training.",
        isCorrect: true,
      },
      {
        text: "The next action \\(a'\\) used in the critic target is sampled from the current policy \\(\\pi_{\\theta}(\\cdot\\mid s')\\).",
        isCorrect: true,
      },
      {
        text: "The actor update can use \\(a\\sim\\pi_{\\theta}(\\cdot\\mid s)\\) and weight it by \\(\\hat Q(s,a)\\).",
        isCorrect: true,
      },
      {
        text: "The method guarantees that critic targets are exact because replay buffers store ‘future rewards’.",
        isCorrect: false,
      },
    ],
    explanation:
      "Replay buffers provide off-policy transitions, but the critic target bootstraps with current-policy actions at the next state. Nothing is exact: targets depend on function approximation and one-sample estimates of expectations rather than stored true future returns.",
  },

  // 30 (Hard) — 2 true
  {
    id: "cs224r-lect5-q30",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "The lecture mentions the reparameterization trick for Gaussian policies. Which statements are correct?",
    options: [
      {
        text: "It can reduce gradient estimator variance by expressing \\(a=\\mu_{\\theta}(s)+\\sigma_{\\theta}(s)\\odot\\epsilon\\) with \\(\\epsilon\\sim\\mathcal N(0,I)\\).",
        isCorrect: true,
      },
      {
        text: "It enables differentiating through sampled actions by moving randomness into \\(\\epsilon\\).",
        isCorrect: true,
      },
      {
        text: "It is only applicable to discrete action spaces.",
        isCorrect: false,
      },
      {
        text: "It removes the need for any critic/value function in actor-critic.",
        isCorrect: false,
      },
    ],
    explanation:
      "For continuous (e.g., Gaussian) policies, reparameterization rewrites sampling as a deterministic transform of noise, which can lower variance and allow pathwise derivatives. It’s not for discrete spaces by default and does not eliminate the need for a critic in actor-critic methods.",
  },

  // 31 (Easy) — 1 true
  {
    id: "cs224r-lect5-q31",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "In the multiple-gradient-step setting, why does the ratio \\(\\frac{\\pi_{\\theta'}(a\\mid s)}{\\pi_{\\theta}(a\\mid s)}\\) start near 1 and then drift?",
    options: [
      {
        text: "Initially \\(\\theta'\\approx\\theta\\), but after updates \\(\\pi_{\\theta'}\\) changes while \\(\\pi_{\\theta}\\) (data-collection policy) stays fixed.",
        isCorrect: true,
      },
      {
        text: "Because advantages \\(\\hat A\\) are renormalized to have mean 1 each step.",
        isCorrect: false,
      },
      {
        text: "Because rewards are discounted by \\(\\gamma\\), which forces the ratio toward 1.",
        isCorrect: false,
      },
      {
        text: "Because replay buffers automatically resample actions to match \\(\\pi_{\\theta'}\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The denominator comes from the frozen behavior policy that generated the data, while the numerator changes with every policy update. Therefore the ratio begins close to 1 but can become extreme if many updates are taken without collecting new data.",
  },

  // 32 (Hard) — all true
  {
    id: "cs224r-lect5-q32",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Consider why off-policy with replay buffers can enable real-robot learning in short time horizons (as suggested by the examples). Which statements are correct at a conceptual level?",
    options: [
      {
        text: "Reusing each real-world transition many times via replay can drastically reduce required environment interaction.",
        isCorrect: true,
      },
      {
        text: "Critic bootstrapping can propagate value information without requiring full Monte Carlo rollouts for every update.",
        isCorrect: true,
      },
      {
        text: "High sample efficiency matters more on real robots because data collection is expensive and slow compared to simulation.",
        isCorrect: true,
      },
      {
        text: "The approach still requires careful tuning and can be less stable due to function approximation and off-policy issues.",
        isCorrect: true,
      },
    ],
    explanation:
      "Replay buffers and bootstrapped critics let you squeeze more learning out of each physical interaction, which is crucial for robots. The price is sensitivity: off-policy critic errors and hyperparameters can destabilize training, so engineering/tuning matters.",
  },

  // 33 (Medium) — 2 true
  {
    id: "cs224r-lect5-q33",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly relate PPO/SAC to imitation learning as discussed near the end?",
    options: [
      {
        text: "Reinforcement learning can sometimes surpass imitation learning on success rate because it can improve beyond the demonstrations.",
        isCorrect: true,
      },
      {
        text: "Imitation learning can be simpler but may plateau if demonstrations are imperfect or limited.",
        isCorrect: true,
      },
      {
        text: "Imitation learning always yields faster policies than reinforcement learning.",
        isCorrect: false,
      },
      {
        text: "PPO and SAC are forms of supervised imitation learning with different loss functions.",
        isCorrect: false,
      },
    ],
    explanation:
      "Imitation learning copies behavior from demonstrations and can be limited by demonstration quality and coverage. RL optimizes reward directly and can exceed demos, though it may require more tuning; PPO/SAC are RL algorithms, not supervised imitation methods.",
  },

  // 34 (Easy) — 3 true
  {
    id: "cs224r-lect5-q34",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements about why clipping/constraints help with ‘overfitting the advantages’ are correct?",
    options: [
      {
        text: "They reduce incentives to make \\(\\pi_{\\theta'}\\) drastically different from the data-collection policy \\(\\pi_{\\theta}\\).",
        isCorrect: true,
      },
      {
        text: "Smaller policy changes make it more plausible that \\(\\hat A_{\\pi_{\\theta}}\\) remains a useful learning signal for \\(\\pi_{\\theta'}\\).",
        isCorrect: true,
      },
      {
        text: "They can prevent ratios from becoming extreme, which otherwise amplifies noise from finite-batch advantage estimates.",
        isCorrect: true,
      },
      {
        text: "They eliminate the need to collect fresh data ever again.",
        isCorrect: false,
      },
    ],
    explanation:
      "Constraints/clipping keep the update ‘proximal’, so advantage estimates computed from the old policy are less stale. They also bound the influence of any single sample via the ratio, but they do not remove the need to periodically refresh data as the policy evolves.",
  },

  // 35 (Hard) — 1 true
  {
    id: "cs224r-lect5-q35",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Suppose your replay buffer contains transitions from very early random policies and your current policy is much more competent. What is a key risk for the critic target \\(y=r+\\gamma\\hat Q(s',a')\\) with \\(a'\\sim\\pi_{\\theta}(\\cdot\\mid s')\\)?",
    options: [
      {
        text: "The next-state \\(s'\\) distribution may come from parts of the state space rarely visited by the current policy, so \\(\\hat Q(s',a')\\) may be extrapolating and unstable.",
        isCorrect: true,
      },
      {
        text: "The target becomes exact because early random policies explore widely.",
        isCorrect: false,
      },
      {
        text: "The target no longer depends on \\(\\gamma\\) because \\(a'\\) comes from the current policy.",
        isCorrect: false,
      },
      {
        text: "The actor update becomes on-policy as soon as the buffer is large enough.",
        isCorrect: false,
      },
    ],
    explanation:
      "Old-data replay can force the critic to evaluate current-policy actions in next states that are off-distribution for the current policy. That can lead to extrapolation error and unstable bootstrapping, even though replay improves sample efficiency in many settings.",
  },
];
