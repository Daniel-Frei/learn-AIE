import { Question } from "../../quiz";

export const OtherRL_introductiontoReinforcementLearning: Question[] = [
  {
    id: "other-rl-intro-q01",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe reinforcement learning compared to traditional programming?",
    options: [
      {
        text: "It is useful when explicitly programming every low-level action is infeasible due to complex dynamics.",
        isCorrect: true,
      },
      {
        text: "It replaces hand-coded rules with learning driven by reward feedback.",
        isCorrect: true,
      },
      {
        text: "It is inspired by trial-and-error learning seen in animals.",
        isCorrect: true,
      },
      {
        text: "It does not require complete knowledge of physics before training begins.",
        isCorrect: true,
      },
    ],
    explanation:
      "Reinforcement learning is applied where specifying exact control rules is impractical. Instead of requiring full knowledge of dynamics, it learns behavior through interaction and reward signals.",
  },

  {
    id: "other-rl-intro-q02",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "Which statements about the agent–environment framework are correct?",
    options: [
      {
        text: "The agent represents the part of the system directly controlled by the learner.",
        isCorrect: true,
      },
      {
        text: "The environment includes everything outside the learner’s direct control.",
        isCorrect: true,
      },
      {
        text: "Actions influence the environment, and states are observations returned to the agent.",
        isCorrect: true,
      },
      {
        text: "The boundary between agent and environment is a modeling choice rather than a fixed objective fact.",
        isCorrect: true,
      },
    ],
    explanation:
      "Agent and environment definitions depend on what is considered controllable. The interaction loop consists of actions changing the environment and states being fed back to the agent.",
  },

  {
    id: "other-rl-intro-q03",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "Which of the following are valid examples of state representations?",
    options: [
      { text: "Pixel values from a game screen.", isCorrect: true },
      { text: "Joint angles and velocities of a robot.", isCorrect: true },
      { text: "Battery level and temperature readings.", isCorrect: true },
      {
        text: "A state representation can be engineered or learned rather than provided only as a human symbolic judgment.",
        isCorrect: true,
      },
    ],
    explanation:
      'States are numerical observations describing the environment. Human subjective labels are not intrinsic state variables unless encoded numerically. To reason through the choices, select every statement because each one matches the criterion in the prompt: "Pixel values from a game screen."; "Joint angles and velocities of a robot."; "Battery level and temperature readings."; "A state representation can be engineered or learned rather than provided only as a human symbolic judgment.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "other-rl-intro-q04",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "Which statements about rewards in reinforcement learning are correct?",
    options: [
      { text: "They provide scalar feedback after actions.", isCorrect: true },
      {
        text: "They guide the agent toward desirable behavior.",
        isCorrect: true,
      },
      {
        text: "They are conventionally treated as signals from the environment.",
        isCorrect: true,
      },
      {
        text: "They do not directly specify the optimal action at each step.",
        isCorrect: true,
      },
    ],
    explanation:
      'Rewards do not prescribe exact actions but provide evaluative feedback. The agent must learn how to act to maximize cumulative rewards. To reason through the choices, select every statement because each one matches the criterion in the prompt: "They provide scalar feedback after actions."; "They guide the agent toward desirable behavior."; "They are conventionally treated as signals from the environment."; "They do not directly specify the optimal action at each step.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "other-rl-intro-q05",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which expressions correctly represent discounted return?",
    options: [
      {
        text: "\\(G_t = \\sum_{k=0}^{\\infty} \\gamma^k r_{t+k}\\)",
        isCorrect: true,
      },
      {
        text: "Discount factor \\(\\gamma\\) controls how much future rewards matter.",
        isCorrect: true,
      },
      {
        text: "Setting \\(\\gamma = 1\\) is safe only when episodes terminate.",
        isCorrect: true,
      },
      {
        text: "Discounting does not ensure actions are deterministic.",
        isCorrect: true,
      },
    ],
    explanation:
      'Discounting downweights distant rewards and prevents divergence in infinite horizons. Determinism is unrelated to discounting. To reason through the choices, select every statement because each one matches the criterion in the prompt: "\\(G_t = \\sum_{k=0}^{\\infty} \\gamma^k r_{t+k}\\)"; "Discount factor \\(\\gamma\\) controls how much future rewards matter."; "Setting \\(\\gamma = 1\\) is safe only when episodes terminate."; "Discounting does not ensure actions are deterministic.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "other-rl-intro-q06",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements correctly describe the Markov property?",
    options: [
      {
        text: "The next state depends only on the current state and action.",
        isCorrect: true,
      },
      {
        text: "Missing velocity information can violate the Markov assumption.",
        isCorrect: true,
      },
      {
        text: "Including sufficient variables restores Markov structure.",
        isCorrect: true,
      },
      {
        text: "It does not guarantee fast learning regardless of rewards.",
        isCorrect: true,
      },
    ],
    explanation:
      'Markov states must contain all information needed to predict transitions. It is a structural assumption, not a guarantee of performance. To reason through the choices, select every statement because each one matches the criterion in the prompt: "The next state depends only on the current state and action."; "Missing velocity information can violate the Markov assumption."; "Including sufficient variables restores Markov structure."; "It does not guarantee fast learning regardless of rewards.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "other-rl-intro-q07",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which are formal components of a Markov Decision Process?",
    options: [
      { text: "State space", isCorrect: true },
      { text: "Action space", isCorrect: true },
      { text: "Reward signal", isCorrect: true },
      {
        text: "Transition dynamics relating states and actions",
        isCorrect: true,
      },
    ],
    explanation:
      "The MDP is defined by states, actions, rewards and transition dynamics. The policy is what the agent tries to learn, not part of the environment definition.",
  },

  {
    id: "other-rl-intro-q08",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "Which statements about policy functions \\(\\pi(a|s)\\) are correct?",
    options: [
      { text: "They output probabilities over actions.", isCorrect: true },
      { text: "Stochasticity supports exploration.", isCorrect: true },
      { text: "Softmax is often used for discrete policies.", isCorrect: true },
      {
        text: "They do not directly encode environment dynamics.",
        isCorrect: true,
      },
    ],
    explanation:
      'Policies decide actions, not transitions. The environment dynamics are modeled separately. To reason through the choices, select every statement because each one matches the criterion in the prompt: "They output probabilities over actions."; "Stochasticity supports exploration."; "Softmax is often used for discrete policies."; "They do not directly encode environment dynamics.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "other-rl-intro-q09",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "Which are reasons discounted return is preferred over immediate reward?",
    options: [
      { text: "It captures delayed consequences of actions.", isCorrect: true },
      {
        text: "Greedy immediate rewards can lead to poor long-term outcomes.",
        isCorrect: true,
      },
      {
        text: "It allows optimizing over trajectories instead of single steps.",
        isCorrect: true,
      },
      {
        text: "It does not remove the need for exploration.",
        isCorrect: true,
      },
    ],
    explanation:
      'Return encourages long-term planning. Exploration is still required to discover high-reward strategies. To reason through the choices, select every statement because each one matches the criterion in the prompt: "It captures delayed consequences of actions."; "Greedy immediate rewards can lead to poor long-term outcomes."; "It allows optimizing over trajectories instead of single steps."; "It does not remove the need for exploration.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "other-rl-intro-q10",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements correctly describe world models?",
    options: [
      { text: "They estimate \\(P(s', r | s, a)\\).", isCorrect: true },
      { text: "They encode environment dynamics.", isCorrect: true },
      {
        text: "Having a world model improves sample efficiency.",
        isCorrect: true,
      },
      {
        text: "Model-free RL assumes no access to this transition model.",
        isCorrect: true,
      },
    ],
    explanation:
      'World models predict transitions and rewards. Model-free methods instead learn purely from sampled experience. To reason through the choices, select every statement because each one matches the criterion in the prompt: "They estimate \\(P(s\', r | s, a)\\)."; "They encode environment dynamics."; "Having a world model improves sample efficiency."; "Model-free RL assumes no access to this transition model.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "other-rl-intro-q11",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements about trajectories are correct?",
    options: [
      { text: "They are sequences \\((s_t,a_t,r_t)\\).", isCorrect: true },
      { text: "They are collected through sampling.", isCorrect: true },
      { text: "Returns can be computed from them.", isCorrect: true },
      {
        text: "They are still necessary in model-free RL because returns are estimated from sampled experience.",
        isCorrect: true,
      },
    ],
    explanation:
      'Trajectories are fundamental to learning in model-free RL since they provide all observed experience. To reason through the choices, select every statement because each one matches the criterion in the prompt: "They are sequences \\((s_t,a_t,r_t)\\)."; "They are collected through sampling."; "Returns can be computed from them."; "They are still necessary in model-free RL because returns are estimated from sampled experience.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "other-rl-intro-q12",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which issues characterize Monte Carlo methods?",
    options: [
      {
        text: "They update values only after episode completion.",
        isCorrect: true,
      },
      { text: "They struggle with long episodes.", isCorrect: true },
      {
        text: "They cannot distinguish which intermediate actions were responsible for success.",
        isCorrect: true,
      },
      {
        text: "They are often less sample efficient than temporal difference methods.",
        isCorrect: true,
      },
    ],
    explanation:
      'Monte Carlo waits for full returns, causing slow learning and poor credit assignment. To reason through the choices, select every statement because each one matches the criterion in the prompt: "They update values only after episode completion."; "They struggle with long episodes."; "They cannot distinguish which intermediate actions were responsible for success."; "They are often less sample efficient than temporal difference methods.". No listed statement should be rejected, so the important boundary is that all four claims contribute a valid part of the concept rather than introducing a competing misconception.',
  },

  {
    id: "other-rl-intro-q13",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements about temporal difference learning are correct?",
    options: [
      { text: "It bootstraps from estimated future values.", isCorrect: true },
      { text: "It updates only after every full episode.", isCorrect: false },
      {
        text: "It reduces dependence on episode termination.",
        isCorrect: true,
      },
      {
        text: "It computes full trajectory returns explicitly.",
        isCorrect: false,
      },
    ],
    explanation:
      'Temporal difference uses one-step estimates rather than full rollouts, enabling faster updates. To reason through the choices, select the statements that match the criterion in the prompt: "It bootstraps from estimated future values."; "It reduces dependence on episode termination.". Do not select statements that miss that criterion: "It updates only after every full episode."; "It computes full trajectory returns explicitly.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "other-rl-intro-q14",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "According to the lecture, which of the following are described as off-policy methods?",
    options: [
      { text: "Q-learning", isCorrect: true },
      { text: "SARSA", isCorrect: false },
      { text: "Expected SARSA", isCorrect: false },
      {
        text: "Monte Carlo control (as used in the maze example)",
        isCorrect: false,
      },
    ],
    explanation:
      'The lecture explicitly labels SARSA variants as on-policy and Q-learning as off-policy. To reason through the choices, select the statements that match the criterion in the prompt: "Q-learning". Do not select statements that miss that criterion: "SARSA"; "Expected SARSA"; "Monte Carlo control (as used in the maze example)". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "other-rl-intro-q15",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements about epsilon-greedy exploration are correct?",
    options: [
      {
        text: "With probability \\(\\epsilon\\), a random action is selected.",
        isCorrect: true,
      },
      {
        text: "It always prevents convergence to a suboptimal policy.",
        isCorrect: false,
      },
      {
        text: "It follows a fixed exploration rate and never shifts toward exploitation.",
        isCorrect: false,
      },
      {
        text: "It guarantees finding the optimal policy in finite steps.",
        isCorrect: false,
      },
    ],
    explanation:
      'Epsilon-greedy ensures continued exploration but does not guarantee optimality without sufficient sampling. To reason through the choices, select the statements that match the criterion in the prompt: "With probability \\(\\epsilon\\), a random action is selected.". Do not select statements that miss that criterion: "It always prevents convergence to a suboptimal policy."; "It follows a fixed exploration rate and never shifts toward exploitation."; "It guarantees finding the optimal policy in finite steps.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "other-rl-intro-q16",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "Which statements about sample efficiency from the lecture are correct?",
    options: [
      {
        text: "Temporal difference methods always outperform Monte Carlo in sample efficiency.",
        isCorrect: false,
      },
      {
        text: "In the maze example, Q-learning was always more sample efficient than SARSA in every setting.",
        isCorrect: false,
      },
      {
        text: "Sample efficiency refers to how many interactions are needed to learn good behavior.",
        isCorrect: true,
      },
      {
        text: "Monte Carlo was shown to be more efficient than TD methods.",
        isCorrect: false,
      },
    ],
    explanation:
      'The lecture only makes empirical comparisons (maze) and explains sample efficiency conceptually. To reason through the choices, select the statements that match the criterion in the prompt: "Sample efficiency refers to how many interactions are needed to learn good behavior.". Do not select statements that miss that criterion: "Temporal difference methods always outperform Monte Carlo in sample efficiency."; "In the maze example, Q-learning was always more sample efficient than SARSA in every setting."; "Monte Carlo was shown to be more efficient than TD methods.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "other-rl-intro-q17",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "Which are advantages of neural networks in reinforcement learning?",
    options: [
      { text: "They approximate continuous state spaces.", isCorrect: true },
      {
        text: "They guarantee perfect generalization across all unseen states.",
        isCorrect: false,
      },
      {
        text: "They eliminate the need for choosing architectures or parameterizations.",
        isCorrect: false,
      },
      {
        text: "They naturally handle infinite discrete actions in value-based methods.",
        isCorrect: false,
      },
    ],
    explanation:
      'Networks allow continuous representation but value-based approaches still require discrete actions. To reason through the choices, select the statements that match the criterion in the prompt: "They approximate continuous state spaces.". Do not select statements that miss that criterion: "They guarantee perfect generalization across all unseen states."; "They eliminate the need for choosing architectures or parameterizations."; "They naturally handle infinite discrete actions in value-based methods.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "other-rl-intro-q18",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements about Deep Q Networks are correct?",
    options: [
      {
        text: "They approximate the action-value function with neural networks.",
        isCorrect: true,
      },
      {
        text: "They rely on Q-learning updates without any bootstrapping targets.",
        isCorrect: false,
      },
      {
        text: "They operate naturally over any continuous action space without modification.",
        isCorrect: false,
      },
      { text: "They directly learn stochastic policies.", isCorrect: false },
    ],
    explanation:
      'Deep Q Networks estimate Q-values; the policy is derived greedily rather than modeled probabilistically. To reason through the choices, select the statements that match the criterion in the prompt: "They approximate the action-value function with neural networks.". Do not select statements that miss that criterion: "They rely on Q-learning updates without any bootstrapping targets."; "They operate naturally over any continuous action space without modification."; "They directly learn stochastic policies.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "other-rl-intro-q19",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements about policy gradients are correct?",
    options: [
      {
        text: "They optimize an objective \\(J(\\theta)\\) instead of minimizing supervised loss.",
        isCorrect: true,
      },
      {
        text: "They increase probability of actions without weighting by advantage or return.",
        isCorrect: false,
      },
      {
        text: "They can handle continuous actions via Gaussian policies.",
        isCorrect: true,
      },
      { text: "They rely on tabular value iteration only.", isCorrect: false },
    ],
    explanation:
      'Policy gradient methods perform gradient ascent on expected reward and extend naturally to continuous domains. To reason through the choices, select the statements that match the criterion in the prompt: "They optimize an objective \\(J(\\theta)\\) instead of minimizing supervised loss."; "They can handle continuous actions via Gaussian policies.". Do not select statements that miss that criterion: "They increase probability of actions without weighting by advantage or return."; "They rely on tabular value iteration only.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "other-rl-intro-q20",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements about actor–critic methods are correct?",
    options: [
      { text: "The actor updates the policy.", isCorrect: true },
      { text: "The critic estimates value functions.", isCorrect: true },
      {
        text: "They typically avoid temporal-difference-style value estimates altogether.",
        isCorrect: false,
      },
      { text: "They remove the need for rewards.", isCorrect: false },
    ],
    explanation:
      'Actor–critic combines policy optimization with value estimation; rewards remain essential. To reason through the choices, select the statements that match the criterion in the prompt: "The actor updates the policy."; "The critic estimates value functions.". Do not select statements that miss that criterion: "They typically avoid temporal-difference-style value estimates altogether."; "They remove the need for rewards.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "other-rl-intro-q21",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "Which statements about dopamine and temporal difference error are correct?",
    options: [
      {
        text: "Dopamine spikes correspond to reward prediction errors.",
        isCorrect: true,
      },
      {
        text: "Signals remain fixed on the primary reward and do not shift to predictive cues.",
        isCorrect: false,
      },
      {
        text: "Negative prediction errors occur when expected rewards fail to appear.",
        isCorrect: true,
      },
      {
        text: "Dopamine encodes only immediate reward magnitude.",
        isCorrect: false,
      },
    ],
    explanation:
      'Neuroscience evidence shows dopamine tracks surprise in expected reward, aligning closely with temporal difference error. To reason through the choices, select the statements that match the criterion in the prompt: "Dopamine spikes correspond to reward prediction errors."; "Negative prediction errors occur when expected rewards fail to appear.". Do not select statements that miss that criterion: "Signals remain fixed on the primary reward and do not shift to predictive cues."; "Dopamine encodes only immediate reward magnitude.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "other-rl-intro-q22",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements about imitation learning are correct?",
    options: [
      {
        text: "Behavioral cloning learns policies directly from expert state–action pairs.",
        isCorrect: true,
      },
      {
        text: "Dataset Aggregation improves robustness by adding corrective labels.",
        isCorrect: true,
      },
      {
        text: "It completely removes dependence on expert data once the initial demonstrations are collected.",
        isCorrect: false,
      },
      {
        text: "It guarantees optimal behavior outside demonstrated states.",
        isCorrect: false,
      },
    ],
    explanation:
      'Imitation learning mimics expert trajectories but struggles when encountering unseen states unless augmented. To reason through the choices, select the statements that match the criterion in the prompt: "Behavioral cloning learns policies directly from expert state–action pairs."; "Dataset Aggregation improves robustness by adding corrective labels.". Do not select statements that miss that criterion: "It completely removes dependence on expert data once the initial demonstrations are collected."; "It guarantees optimal behavior outside demonstrated states.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "other-rl-intro-q23",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "Which statements about inverse reinforcement learning are correct?",
    options: [
      {
        text: "It aims to infer the reward function underlying expert behavior.",
        isCorrect: true,
      },
      {
        text: "It can generalize better by learning goals instead of direct actions.",
        isCorrect: true,
      },
      {
        text: "It eliminates the need for expert demonstrations.",
        isCorrect: false,
      },
      {
        text: "Inverse Q-learning is the only method used for inverse reinforcement learning.",
        isCorrect: false,
      },
    ],
    explanation:
      'Inverse reinforcement learning deduces objectives from demonstrations rather than copying actions. To reason through the choices, select the statements that match the criterion in the prompt: "It aims to infer the reward function underlying expert behavior."; "It can generalize better by learning goals instead of direct actions.". Do not select statements that miss that criterion: "It eliminates the need for expert demonstrations."; "Inverse Q-learning is the only method used for inverse reinforcement learning.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "other-rl-intro-q24",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which are limitations of model-free reinforcement learning?",
    options: [
      { text: "Requires large numbers of interactions.", isCorrect: true },
      {
        text: "Learns without understanding environment physics.",
        isCorrect: true,
      },
      {
        text: "Often inefficient compared to model-based approaches.",
        isCorrect: true,
      },
      { text: "Naturally generalizes from few samples.", isCorrect: false },
    ],
    explanation:
      'Model-free RL relies purely on trial-and-error, leading to high sample complexity. To reason through the choices, select the statements that match the criterion in the prompt: "Requires large numbers of interactions."; "Learns without understanding environment physics."; "Often inefficient compared to model-based approaches.". Do not select statements that miss that criterion: "Naturally generalizes from few samples.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "other-rl-intro-q25",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe model-based reinforcement learning?",
    options: [
      { text: "Uses learned or provided transition models.", isCorrect: true },
      { text: "Allows simulated rollouts for planning.", isCorrect: true },
      {
        text: "Can dramatically reduce required real-world samples.",
        isCorrect: true,
      },
      { text: "Cannot be combined with neural networks.", isCorrect: false },
    ],
    explanation:
      'Model-based RL leverages predictive models to imagine futures and plan efficiently. To reason through the choices, select the statements that match the criterion in the prompt: "Uses learned or provided transition models."; "Allows simulated rollouts for planning."; "Can dramatically reduce required real-world samples.". Do not select statements that miss that criterion: "Cannot be combined with neural networks.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "other-rl-intro-q26",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements about RL instability are correct?",
    options: [
      {
        text: "Different random seeds can produce drastically different results.",
        isCorrect: true,
      },
      {
        text: "Policy gradient methods often show high variance across runs.",
        isCorrect: true,
      },
      {
        text: "Training reproducibility is a known challenge.",
        isCorrect: true,
      },
      {
        text: "RL training is fully deterministic given fixed hyperparameters.",
        isCorrect: false,
      },
    ],
    explanation:
      'RL is sensitive to randomness in initialization and sampling, leading to instability. To reason through the choices, select the statements that match the criterion in the prompt: "Different random seeds can produce drastically different results."; "Policy gradient methods often show high variance across runs."; "Training reproducibility is a known challenge.". Do not select statements that miss that criterion: "RL training is fully deterministic given fixed hyperparameters.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "other-rl-intro-q27",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements about reward hacking are correct?",
    options: [
      {
        text: "Poorly designed rewards can lead to unintended behavior.",
        isCorrect: true,
      },
      {
        text: "Agents exploit loopholes rather than achieving true goals.",
        isCorrect: true,
      },
      {
        text: "Expert demonstrations can mitigate this issue.",
        isCorrect: true,
      },
      {
        text: "Reward hacking occurs only in continuous control tasks.",
        isCorrect: false,
      },
    ],
    explanation:
      'Reward misalignment leads to gaming the metric rather than solving the task. To reason through the choices, select the statements that match the criterion in the prompt: "Poorly designed rewards can lead to unintended behavior."; "Agents exploit loopholes rather than achieving true goals."; "Expert demonstrations can mitigate this issue.". Do not select statements that miss that criterion: "Reward hacking occurs only in continuous control tasks.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "other-rl-intro-q28",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "Which statements about Bellman’s principle of optimality are correct?",
    options: [
      {
        text: "Optimal future decisions must remain optimal from the next state onward.",
        isCorrect: true,
      },
      { text: "It underlies dynamic programming in RL.", isCorrect: true },
      {
        text: "It justifies the max operator in Q-learning targets.",
        isCorrect: true,
      },
      { text: "It requires a deterministic environment.", isCorrect: false },
    ],
    explanation:
      'Bellman optimality applies even in stochastic environments and is fundamental to value recursion. To reason through the choices, select the statements that match the criterion in the prompt: "Optimal future decisions must remain optimal from the next state onward."; "It underlies dynamic programming in RL."; "It justifies the max operator in Q-learning targets.". Do not select statements that miss that criterion: "It requires a deterministic environment.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "other-rl-intro-q29",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "Which statements about continuing tasks without terminal states are correct?",
    options: [
      {
        text: "Monte Carlo methods struggle because returns cannot be finalized.",
        isCorrect: true,
      },
      {
        text: "Temporal difference can still learn relative values.",
        isCorrect: true,
      },
      { text: "Discounting prevents divergence.", isCorrect: true },
      { text: "Value estimates become meaningless.", isCorrect: false },
    ],
    explanation:
      'Without terminal anchors values drift but relative ordering still guides decisions. To reason through the choices, select the statements that match the criterion in the prompt: "Monte Carlo methods struggle because returns cannot be finalized."; "Temporal difference can still learn relative values."; "Discounting prevents divergence.". Do not select statements that miss that criterion: "Value estimates become meaningless.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "other-rl-intro-q30",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements about Gaussian policies are correct?",
    options: [
      {
        text: "Actions are sampled from \\(\\mathcal{N}(\\mu, \\sigma^2)\\).",
        isCorrect: true,
      },
      {
        text: "Policy updates shift the mean toward advantageous actions.",
        isCorrect: true,
      },
      {
        text: "Variance can expand or contract based on exploration needs.",
        isCorrect: true,
      },
      { text: "They apply only to discrete action spaces.", isCorrect: false },
    ],
    explanation:
      'Gaussian policies enable continuous control through parameterized distributions. To reason through the choices, select the statements that match the criterion in the prompt: "Actions are sampled from \\(\\mathcal{N}(\\mu, \\sigma^2)\\)."; "Policy updates shift the mean toward advantageous actions."; "Variance can expand or contract based on exploration needs.". Do not select statements that miss that criterion: "They apply only to discrete action spaces.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "other-rl-intro-q31",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements about the policy objective J(?) are correct?",
    options: [
      {
        text: "It measures expected performance over trajectories.",
        isCorrect: true,
      },
      { text: "It is optimized using gradient ascent.", isCorrect: true },
      { text: "Different equivalent definitions exist.", isCorrect: true },
      {
        text: "It provides direct per-state learning targets like Q-values do.",
        isCorrect: false,
      },
    ],
    explanation:
      'J evaluates the policy globally; Q and V provide state/action-level learning structure. To reason through the choices, select the statements that match the criterion in the prompt: "It measures expected performance over trajectories."; "It is optimized using gradient ascent."; "Different equivalent definitions exist.". Do not select statements that miss that criterion: "It provides direct per-state learning targets like Q-values do.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "other-rl-intro-q32",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements about credit assignment are correct?",
    options: [
      {
        text: "It refers to identifying which actions led to outcomes.",
        isCorrect: true,
      },
      { text: "Harder when rewards are delayed.", isCorrect: true },
      { text: "Temporal difference partially alleviates it.", isCorrect: true },
      { text: "It disappears in Monte Carlo methods.", isCorrect: false },
    ],
    explanation:
      'Credit assignment remains challenging; Monte Carlo actually worsens it. To reason through the choices, select the statements that match the criterion in the prompt: "It refers to identifying which actions led to outcomes."; "Harder when rewards are delayed."; "Temporal difference partially alleviates it.". Do not select statements that miss that criterion: "It disappears in Monte Carlo methods.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "other-rl-intro-q33",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements about RL vs human learning are correct?",
    options: [
      { text: "Humans use internal world models.", isCorrect: true },
      {
        text: "Model-free RL resembles blind trial-and-error.",
        isCorrect: true,
      },
      {
        text: "RL can still succeed without explicit understanding.",
        isCorrect: true,
      },
      {
        text: "Humans learn exclusively via scalar rewards.",
        isCorrect: false,
      },
    ],
    explanation:
      'Humans leverage structure and reasoning beyond raw reward feedback. To reason through the choices, select the statements that match the criterion in the prompt: "Humans use internal world models."; "Model-free RL resembles blind trial-and-error."; "RL can still succeed without explicit understanding.". Do not select statements that miss that criterion: "Humans learn exclusively via scalar rewards.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },
];
