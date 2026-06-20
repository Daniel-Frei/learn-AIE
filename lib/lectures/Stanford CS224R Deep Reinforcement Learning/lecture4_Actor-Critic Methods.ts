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
        text: "It is defined as \\(V^\\pi(s)\\), an expectation over trajectories starting at \\(s\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The value function \\(V^\\pi(s)\\) evaluates a state under a particular policy by taking an expectation over future trajectories that start at \\(s\\). It depends on the policy, rewards, and environment dynamics because all of those determine the future return. In some tasks that expected return can be interpreted as a probability of success, but the general definition is expected cumulative reward.",
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
      "The Q-function \\(Q^\\pi(s,a)\\) evaluates a state-action pair by fixing the first action and then following policy \\(\\pi\\) afterward. That conditioning is what distinguishes it from \\(V^\\pi(s)\\), which averages over the policy's action choice immediately. The estimate includes immediate reward and expected future rewards, with future stochastic transitions averaged inside the expectation.",
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
        text: "Positive \\(A^\\pi(s,a)\\) implies the action is better than expected.",
        isCorrect: true,
      },
      {
        text: "It centers Q-values around the state value baseline.",
        isCorrect: true,
      },
    ],
    explanation:
      "The advantage function compares the value of taking action \\(a\\) in state \\(s\\) against the policy's usual expected value at that state. The formula \\(A^\\pi(s,a)=Q^\\pi(s,a)-V^\\pi(s)\\) centers action values around the state-value baseline, so positive advantage means the action did better than the policy's average choice. This centered signal is useful in actor updates because it reduces variance while preserving which actions should be reinforced.",
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
      "Actor-critic methods split the problem into an actor that represents the policy and a critic that estimates values or advantages. The critic gives the actor a learned training signal, often an advantage estimate, instead of relying only on raw sampled returns. This can improve sample efficiency and reduce variance compared with vanilla policy gradients, although it introduces a second learned model whose errors can affect the actor.",
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
        text: "It does not rely on bootstrapped predictions such as \\(V(s_{t+1})\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "Monte Carlo value estimation uses the full sampled return, such as \\(G_t = \\sum_{t'=t}^T r_{t'}\\), as the regression target for a state. Because it waits for actual trajectory outcomes instead of bootstrapping from current value predictions, the target is unbiased for the policy's return. The tradeoff is high variance, especially on long horizons where many future random events contribute to the return.",
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
        text: "It introduces bias when estimates such as \\(V(s_{t+1})\\) are inaccurate.",
        isCorrect: true,
      },
    ],
    explanation:
      "Temporal difference learning uses a target such as \\(r_t + \\gamma V(s_{t+1})\\), combining one observed reward with a bootstrapped prediction. Bootstrapping usually lowers variance compared with waiting for the full trajectory return. The cost is bias when the current value estimate is inaccurate, because the learner can propagate its own prediction errors into future targets.",
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
      {
        text: "They interpolate between one-step TD \\(r_t + \\gamma V(s_{t+1})\\) and Monte Carlo \\(G_t\\).",
        isCorrect: true,
      },
      { text: "Smaller \\(n\\) gives lower variance.", isCorrect: true },
      { text: "Larger \\(n\\) reduces bias.", isCorrect: true },
    ],
    explanation:
      "An n-step return uses several real rewards and then bootstraps from a value estimate, as in \\(y_t = \\sum_{k=0}^{n-1} \\gamma^k r_{t+k} + \\gamma^n V(s_{t+n})\\). This interpolates between one-step temporal difference learning and full Monte Carlo returns. Smaller \\(n\\) usually gives lower variance because it relies sooner on a prediction, while larger \\(n\\) reduces bootstrap bias by using more observed rewards.",
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
      "A discount factor weights rewards farther in the future by powers of \\(\\gamma\\), which helps keep infinite-horizon returns finite when \\(\\gamma < 1\\). One useful interpretation is stochastic termination: with probability \\(1-\\gamma\\) the process ends, while continuing transitions receive the remaining probability mass. Under that transformed view, nonterminal transition probabilities are effectively scaled by \\(\\gamma\\), but the conceptual role is still to reduce the influence of distant rewards.",
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
      "Advantage-based updates reinforce actions that did better than the policy's baseline expectation and suppress actions that did worse. The value term acts like a learned baseline, which can reduce gradient variance without changing the expected-return objective. The actor is still optimizing reward-seeking behavior; the advantage only changes the estimator used to assign credit.",
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
      {
        text: "The labels remain fixed during training even when bootstrapped targets \\(y_t = r_t + \\gamma V(s_{t+1})\\) are recomputed.",
        isCorrect: false,
      },
      {
        text: "Multiple trajectories provide amortized supervision.",
        isCorrect: true,
      },
    ],
    explanation:
      "Learning \\(V^\\pi\\) can be framed as supervised regression from states to return targets, often with a squared loss such as \\(L=(V_\\phi(s)-y)^2\\). Multiple trajectories provide many state-target pairs, allowing the value function to amortize information across states rather than memorize one rollout. The labels are not always fixed because bootstrapped targets can change as the current value estimates are updated.",
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
      "A pure Monte Carlo target for a state comes from the rewards observed later in that sampled trajectory. If the method treats those samples independently, it may fail to connect similar states across different trajectories and therefore misses shared structure. Propagating value across similar states requires function approximation or bootstrapping-style generalization, not just using each sampled return in isolation.",
  },

  {
    id: "cs224r-lect4-q12",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about bootstrapped targets are correct?",
    options: [
      { text: "They depend on current value estimates.", isCorrect: true },
      { text: "They are updated every gradient step.", isCorrect: true },
      {
        text: "They are unbiased even when bootstrapped estimates such as \\(V(s_{t+1})\\) are inaccurate.",
        isCorrect: false,
      },
      { text: "They reduce variance.", isCorrect: true },
    ],
    explanation:
      "Bootstrapped targets depend on the current value function because they include a prediction such as \\(V(s_{t+1})\\). As the value network changes, those targets can change from one gradient step to the next. This lowers variance by avoiding the full sampled return, but it is not unbiased when the value predictions used inside the target are wrong.",
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
      "Replay buffers store transitions collected by earlier versions of the policy, so the learner can reuse experience instead of discarding each batch immediately. This enables off-policy learning and can improve data efficiency. The same reuse can create distribution mismatch between buffered actions and the current policy, so replay buffers do not always improve stability.",
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
      "Off-policy actor updates use data from a behavior policy while updating a different current policy, so importance sampling or related corrections are needed. If the policies differ too much, the correction weights and advantage estimates can become unreliable and learning can diverge. Constraints on policy change, such as KL limits, help keep updates in a region where old data and estimated advantages remain useful.",
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
      "Vanilla policy gradients often use raw sampled returns to decide which actions to reinforce. Actor-critic methods add a critic that learns which states or actions are good, giving the actor a lower-variance training signal and often improving sample usage. The critic still learns from rewards or return targets, so actor-critic does not remove the need for reward information.",
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
      "The actor uses the critic's value or advantage estimates to decide which actions to reinforce. If the critic is systematically biased, the advantage estimates can point the policy update in the wrong direction or distort its magnitude. Critic bias therefore does affect learning, even though the critic is not the component that directly selects actions.",
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
      "One-step temporal difference learning uses the immediate reward plus a bootstrap value estimate, while n-step returns include more observed rewards before bootstrapping. Using more real rewards can reduce bootstrap bias, and still bootstrapping after \\(n\\) steps can reduce variance relative to full Monte Carlo. The bias-variance tradeoff depends on \\(n\\), so n-step returns are not always higher bias.",
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
      "The actor is the policy component: it represents how actions are selected from states or observations. It is usually updated with a policy-gradient-style objective that may use advantage estimates supplied by the critic. Estimating values is the critic's role, not the actor's role.",
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
      "Monte Carlo targets use sampled full returns, so they are unbiased for the policy's return but can be very noisy. Temporal difference targets use bootstrapped predictions, which usually lowers variance but can introduce bias when the prediction is inaccurate. That is why TD is not generally unbiased, and Monte Carlo is not the lower-variance choice.",
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
      "Replay buffers improve sample efficiency by letting the learner reuse past transitions for multiple updates. The cost is that buffered data may have been generated by older policies, creating mismatch with the current policy and possible off-policy bias. Reuse does not produce perfectly unbiased gradients or guarantee convergence; it must be paired with algorithms that handle the mismatch.",
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
      "Discounting gives more weight to near-term rewards than distant rewards when \\(\\gamma < 1\\). This can keep long-horizon or infinite-horizon value estimates bounded and numerically stable. It does not remove future rewards entirely, and its purpose is not to increase variance.",
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
      {
        text: "They ignore immediate rewards such as \\(r_t\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "A temporal difference target combines the immediate reward with a bootstrapped prediction such as \\(V(s_{t+1})\\). That prediction reduces variance relative to waiting for the full sampled return. The target does not ignore rewards, and it is not generally unbiased because a wrong value prediction can bias the target.",
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
      "KL constraints limit how far the updated policy can move from the policy that generated or justified the data. That stabilizes off-policy or approximate actor updates by keeping importance ratios and advantage estimates from becoming too stale. A KL constraint does not eliminate estimator bias by itself, and it does not automatically remove the need for importance weighting or other corrections.",
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
      "Critic training is usually a regression problem: fit value predictions to Monte Carlo, temporal difference, or n-step targets. Because this is a supervised-style update on stored data, the critic can often take multiple gradient steps per batch. Updating the critic does not directly change the actor's policy parameters, and the regression update itself does not require resetting the environment.",
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
      "Advantages guide actor updates by measuring whether an action was better or worse than the baseline value for that state. This comparison determines both the sign and strength of the policy-gradient update. Advantages are computed from reward-derived value information, so they do not replace rewards, and they are not constant across states and actions.",
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
      "The choice of \\(n\\) controls how much of the target is observed reward versus bootstrapped value prediction. Smaller \\(n\\) bootstraps earlier, which usually lowers variance but can increase bias from imperfect value estimates. Larger \\(n\\) uses more real rewards and therefore tends to lower bias, but n-step targets are not always unbiased or always low variance.",
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
      "The critic can learn from partial progress because value targets can assign useful estimates to intermediate states, not only to fully successful episodes. That makes actor-critic methods more data-efficient than relying only on sparse full-return policy-gradient signals. Failed rollouts can still teach the critic which states or actions have low value, so they are not ignored and full-success trajectories are not the only usable data.",
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
      "The critic estimates values, such as how good a state is under the current policy or how good a state-action pair is. The actor is the component that selects actions, so the critic does not replace it. The critic also still needs reward-derived targets; it turns rewards into a denser learning signal rather than removing rewards from the problem.",
  },

  {
    id: "cs224r-lect4-q29",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which best describes TD learning?",
    options: [
      {
        text: "Regression on full trajectory returns \\(G_t\\) without bootstrapping.",
        isCorrect: false,
      },
      {
        text: "Bootstrap using the next state value \\(V(s_{t+1})\\).",
        isCorrect: true,
      },
      { text: "No supervision.", isCorrect: false },
      { text: "Pure imitation.", isCorrect: false },
    ],
    explanation:
      "Temporal difference learning bootstraps from the next state's value estimate, typically using a target like \\(r_t + \\gamma V(s_{t+1})\\). That differs from Monte Carlo regression on full trajectory returns. It is still supervised by reward-derived targets, and it is not imitation learning because the target is value prediction rather than copying expert actions.",
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
      "Replay-buffer actor updates can be wrong when the actions in the buffer were sampled from an older policy but the update treats them as if they came from the current policy. That violates the distribution assumption behind an on-policy policy-gradient estimator. The issue is not missing rewards, a deterministic environment, or the absence of a value function; it is policy mismatch in the sampled actions.",
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
      "Monte Carlo estimation uses complete sampled returns, so it is unbiased for the value under the sampled policy but can have high variance. Temporal difference learning lowers variance by bootstrapping, replay buffers are a data-reuse mechanism, and advantage normalization rescales an actor-update signal. Those other techniques do not describe the full-return unbiased-but-noisy estimator.",
  },

  {
    id: "cs224r-lect4-q32",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statement about discount factor is correct?",
    options: [
      {
        text: "Gamma greater than one, \\(\\gamma > 1\\), increases stability for long-horizon value estimates.",
        isCorrect: false,
      },
      {
        text: "Gamma less than one, \\(\\gamma < 1\\), reduces long-term reward weight.",
        isCorrect: true,
      },
      { text: "Gamma eliminates variance.", isCorrect: false },
      { text: "Gamma only affects policy network.", isCorrect: false },
    ],
    explanation:
      "When \\(\\gamma < 1\\), rewards farther in the future are multiplied by smaller powers of \\(\\gamma\\), so their contribution to return is reduced. This helps stabilize long-horizon value estimates, but setting \\(\\gamma > 1\\) would usually make long-term sums less stable rather than more stable. Discounting affects the return and value targets broadly; it does not eliminate variance or apply only to the policy network.",
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
      "Critic bias affects policy updates because the actor uses critic-derived values or advantages to decide which actions to reinforce. If those estimates are systematically wrong, the actor can be pushed toward actions that are not actually better. The bias is in the learned evaluation signal, not in the rewards themselves, and it does not inherently improve exploration.",
  },

  {
    id: "cs224r-lect4-q34",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which best describes n-step returns?",
    options: [
      { text: "Only one-step bootstrap.", isCorrect: false },
      {
        text: "Full trajectory sum \\(G_t\\) without a bootstrap term.",
        isCorrect: false,
      },
      {
        text: "Partial reward sum plus bootstrap value \\(V(s_{t+n})\\).",
        isCorrect: true,
      },
      { text: "No rewards used.", isCorrect: false },
    ],
    explanation:
      "An n-step return uses a partial sum of observed rewards and then bootstraps from a value prediction at the state reached after \\(n\\) steps. It is therefore between one-step temporal difference learning and full Monte Carlo returns. It is not reward-free, and it is not merely a one-step bootstrap unless \\(n=1\\).",
  },

  {
    id: "cs224r-lect4-q35",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statement about actor–critic efficiency is correct?",
    options: [
      {
        text: "Less efficient than policy gradients because a critic can never reduce variance.",
        isCorrect: false,
      },
      {
        text: "Uses learned value estimates for better gradients.",
        isCorrect: true,
      },
      { text: "Does not require learning.", isCorrect: false },
      { text: "Always unstable.", isCorrect: false },
    ],
    explanation:
      "Actor-critic efficiency comes from using learned value estimates to produce better, lower-variance actor gradients than raw-return policy gradients. The critic has to be learned, so the method is not learning-free, and critic errors can create instability if handled poorly. It is not inherently less efficient than vanilla policy gradients, nor is it always unstable.",
  },
];
