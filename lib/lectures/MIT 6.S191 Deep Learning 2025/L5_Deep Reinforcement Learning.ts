import { Question } from "../../quiz";

// lib/lectures/MIT 6.S191 Deep Learning 2025/L5_Deep Reinforcement Learning.ts

export const L5_DeepReinforcementLearning: Question[] = [
  {
    id: "mit6s191-l5-q01",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the reinforcement learning (RL) interaction loop between an agent and an environment?",
    options: [
      {
        text: "The agent selects an action \\(a_t\\), and the environment responds with a new observation/state \\(s_{t+1}\\).",
        isCorrect: true,
      },
      {
        text: "The environment provides a reward signal \\(r_t\\) that indicates how desirable the outcome of the agent’s action was.",
        isCorrect: true,
      },
      {
        text: "An RL agent typically learns from experience collected while interacting, rather than only from a fixed labeled dataset \\((x,y)\\).",
        isCorrect: true,
      },
      {
        text: "A trajectory can be viewed as a sequence of \\((s_t, a_t, r_t)\\) tuples over time.",
        isCorrect: true,
      },
    ],
    explanation:
      "In RL, the agent repeatedly acts and receives feedback from the environment in the form of next state/observation and reward. Learning uses these interaction-generated trajectories rather than relying purely on supervised \\((x,y)\\) pairs.",
  },

  {
    id: "mit6s191-l5-q02",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about rewards, returns, and discounting are correct?",
    options: [
      {
        text: "The (discounted) return from time \\(t\\) is often written \\(R_t = \\sum_{i=t}^{\\infty} \\gamma^{i-t} r_i\\) with \\(0 < \\gamma < 1\\).",
        isCorrect: true,
      },
      {
        text: "Using \\(\\gamma < 1\\) typically makes rewards far in the future contribute less to the return than near-term rewards.",
        isCorrect: true,
      },
      {
        text: "If \\(\\gamma = 0\\), the return \\(R_t\\) depends only on the immediate reward \\(r_t\\).",
        isCorrect: true,
      },
      {
        text: "Discounting is mainly used to ensure the policy is deterministic rather than stochastic.",
        isCorrect: false,
      },
    ],
    explanation:
      "Discounting downweights distant rewards and often stabilizes optimization by making long-horizon sums better behaved. Determinism vs stochasticity is a separate design choice about how actions are selected (e.g., argmax vs sampling).",
  },

  {
    id: "mit6s191-l5-q03",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "A Q-function is defined as \\(Q(s_t, a_t) = \\mathbb{E}[R_t \\mid s_t, a_t]\\). Which statements are correct?",
    options: [
      {
        text: "\\(Q(s_t, a_t)\\) estimates the expected discounted future return starting from \\(s_t\\) after taking action \\(a_t\\).",
        isCorrect: true,
      },
      {
        text: "If you know \\(Q(s,a)\\), a natural greedy policy is \\(\\pi^*(s)=\\arg\\max_a Q(s,a)\\).",
        isCorrect: true,
      },
      {
        text: "\\(Q(s_t, a_t)\\) is the same thing as the immediate reward \\(r_t\\) at time \\(t\\).",
        isCorrect: false,
      },
      {
        text: "The Q-function can only be defined when actions are continuous (not discrete).",
        isCorrect: false,
      },
    ],
    explanation:
      "A Q-function scores state–action pairs by expected return, not by the immediate reward alone. It’s defined for both discrete and continuous action spaces; what changes is how you compute/select actions (e.g., argmax is easy for small discrete sets).",
  },

  {
    id: "mit6s191-l5-q04",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statement best characterizes how a deep Q-network (DQN-style) policy selects actions in a discrete action space?",
    options: [
      {
        text: "It selects an action by computing \\(\\arg\\max_a Q_\\theta(s,a)\\) over the available discrete actions.",
        isCorrect: true,
      },
      {
        text: "It must sample actions from a Gaussian distribution parameterized by a mean and variance.",
        isCorrect: false,
      },
      {
        text: "It outputs a probability distribution only after applying a softmax to ensure outputs sum to one.",
        isCorrect: false,
      },
      {
        text: "It can only choose actions that are labeled by humans during training.",
        isCorrect: false,
      },
    ],
    explanation:
      "In the basic value-learning formulation for discrete actions, you estimate \\(Q_\\theta(s,a)\\) and then act greedily with an argmax. Gaussian parameterizations and softmax-normalized probabilities are typical for policy-based approaches, not the basic greedy Q selection.",
  },

  {
    id: "mit6s191-l5-q05",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly compare supervised learning, unsupervised learning, and reinforcement learning?",
    options: [
      {
        text: "Supervised learning uses labeled pairs \\((x,y)\\) to learn a mapping from inputs to desired outputs.",
        isCorrect: true,
      },
      {
        text: "Unsupervised learning uses data \\(x\\) without labels \\(y\\) to discover structure or patterns.",
        isCorrect: true,
      },
      {
        text: "Reinforcement learning focuses on \\((\\text{state}, \\text{action})\\) interaction to maximize future reward over time.",
        isCorrect: true,
      },
      {
        text: "Reinforcement learning requires a pre-collected dataset and cannot learn from interaction.",
        isCorrect: false,
      },
    ],
    explanation:
      "Supervised learning learns from labeled examples, and unsupervised learning learns structure without labels. RL is different: it learns by acting and observing consequences, aiming to maximize long-term reward rather than predict labels.",
  },

  {
    id: "mit6s191-l5-q06",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Consider a DQN that outputs one Q-value per discrete action for a given state. Which statements are correct?",
    options: [
      {
        text: "A single forward pass can produce \\(Q(s,a)\\) values for all discrete actions if the network is parameterized to output a vector.",
        isCorrect: true,
      },
      {
        text: "Choosing the greedy action means selecting the index of the largest output Q-value.",
        isCorrect: true,
      },
      {
        text: "The Q-values can be interpreted as estimates of expected discounted return for each action from that state.",
        isCorrect: true,
      },
      {
        text: "Because the network outputs multiple values, the outputs must sum to 1 like probabilities.",
        isCorrect: false,
      },
    ],
    explanation:
      "In a vector-output Q-network, each output corresponds to a different action’s value; they are not probabilities and do not need to sum to 1. Greedy action selection simply picks the action with the highest estimated return.",
  },

  {
    id: "mit6s191-l5-q07",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statements about action spaces are correct?",
    options: [
      {
        text: "A discrete action space might be \\{\\text{left},\\text{right},\\text{stay}\\} where only a finite set of actions are allowed.",
        isCorrect: true,
      },
      {
        text: "A continuous action space might represent a steering angle where infinitely many values are possible.",
        isCorrect: true,
      },
      {
        text: "Value-based argmax selection becomes difficult in large or continuous action spaces because \\(\\arg\\max\\) may require searching an infinite set.",
        isCorrect: true,
      },
      {
        text: "Continuous action spaces cannot be handled by any policy-based method.",
        isCorrect: false,
      },
    ],
    explanation:
      "Discrete actions are finite; continuous actions form a continuum. In continuous spaces, policies often represent a distribution (e.g., Gaussian) that you can sample from; argmax over continuous actions is generally harder than over small discrete sets.",
  },

  {
    id: "mit6s191-l5-q08",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statement is a valid reason that a purely deterministic policy (always choosing the same action for the same state) can be limiting?",
    options: [
      {
        text: "It may fail to explore alternative actions, making it harder to discover higher-reward strategies.",
        isCorrect: true,
      },
      {
        text: "It guarantees optimal performance in any stochastic environment.",
        isCorrect: false,
      },
      {
        text: "It eliminates the need for a reward function.",
        isCorrect: false,
      },
      {
        text: "It makes discounting \\(\\gamma\\) unnecessary in all tasks.",
        isCorrect: false,
      },
    ],
    explanation:
      "If a policy is deterministic and never explores, it can get stuck with suboptimal behavior because it never tries alternatives. Stochastic environments and sparse rewards often benefit from exploration and probabilistic action selection.",
  },

  {
    id: "mit6s191-l5-q09",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe policy learning (policy networks) in RL?",
    options: [
      {
        text: "A policy network can output a probability distribution over actions in a discrete action space.",
        isCorrect: true,
      },
      {
        text: "Sampling actions \\(a \\sim \\pi(\\cdot\\mid s)\\) introduces stochasticity, enabling exploration.",
        isCorrect: true,
      },
      {
        text: "For discrete actions, softmax is commonly used so the output probabilities sum to 1.",
        isCorrect: true,
      },
      {
        text: "Policy learning always requires computing \\(\\arg\\max_a Q(s,a)\\) at decision time.",
        isCorrect: false,
      },
    ],
    explanation:
      "Policy networks directly represent how actions are chosen, often as probabilities (discrete) or as a distribution (continuous). They enable sampling-based exploration. Computing \\(\\arg\\max_a Q(s,a)\\) is characteristic of value-based methods, not required for policy learning.",
  },

  {
    id: "mit6s191-l5-q10",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "For a continuous action policy modeled as a Gaussian, which statements are correct?",
    options: [
      {
        text: "A Gaussian policy can be parameterized by a mean \\(\\mu(s)\\) and variance (or standard deviation) \\(\\sigma^2(s)\\).",
        isCorrect: true,
      },
      {
        text: "Sampling from \\(\\mathcal{N}(\\mu(s),\\sigma^2(s))\\) produces a continuous action value.",
        isCorrect: true,
      },
      {
        text: "Predicting \\(\\mu\\) and \\(\\sigma\\) avoids needing to output an infinite number of probabilities for every possible action.",
        isCorrect: true,
      },
      {
        text: "A Gaussian policy can only be used for discrete actions like \\{left,right\\}.",
        isCorrect: false,
      },
    ],
    explanation:
      "A continuous policy often outputs parameters of a distribution, such as \\(\\mu\\) and \\(\\sigma\\), rather than enumerating actions. Sampling then yields a continuous action like a steering angle.",
  },

  {
    id: "mit6s191-l5-q11",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about the role of the state \\(s_t\\) are correct in RL problems like Atari or driving?",
    options: [
      {
        text: "The state (or observation) should contain enough information for the agent to decide what to do next.",
        isCorrect: true,
      },
      {
        text: "In Atari-style settings, the state might be represented by raw pixels (frames) from the screen.",
        isCorrect: true,
      },
      {
        text: "The state is always a single scalar number (e.g., the agent’s x-position).",
        isCorrect: false,
      },
      {
        text: "The state never changes in response to actions.",
        isCorrect: false,
      },
    ],
    explanation:
      "States/observations encode what the agent perceives and typically change as the agent acts. In many deep RL setups, the input can be high-dimensional (e.g., image frames), not just a single number.",
  },

  {
    id: "mit6s191-l5-q12",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the idea of value learning vs policy learning?",
    options: [
      {
        text: "Value learning focuses on estimating \\(Q(s,a)\\) (or related value functions) and then deriving a policy from those values.",
        isCorrect: true,
      },
      {
        text: "Policy learning focuses on learning \\(\\pi(a\\mid s)\\) (or \\(\\pi(s)\\)) directly, often by optimizing expected return.",
        isCorrect: true,
      },
      {
        text: "In value learning with discrete actions, a common action selection is \\(a = \\arg\\max_a Q(s,a)\\).",
        isCorrect: true,
      },
      {
        text: "Policy learning cannot represent stochastic policies.",
        isCorrect: false,
      },
    ],
    explanation:
      "Value learning estimates values and derives decisions from them, while policy learning directly represents how actions are chosen (often stochastically). Stochastic policies are a key advantage of policy learning, not a limitation.",
  },

  {
    id: "mit6s191-l5-q13",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statements about a typical Q-learning target are correct?",
    options: [
      {
        text: "A common one-step target is \\(y_t = r_t + \\gamma \\max_{a'} Q(s_{t+1}, a')\\).",
        isCorrect: true,
      },
      {
        text: "The Q-loss often measures a difference between a predicted value \\(Q(s_t,a_t)\\) and a target value \\(y_t\\), e.g., via squared error.",
        isCorrect: true,
      },
      {
        text: "The target includes both the immediate reward and a discounted estimate of future value.",
        isCorrect: true,
      },
      {
        text: "The Q-loss cannot be optimized with backpropagation because it contains a max operator.",
        isCorrect: false,
      },
    ],
    explanation:
      "Value learning often bootstraps: it uses current reward plus discounted estimate of future value. In deep RL, you still optimize parameters with backpropagation; practical methods handle the target/bootstrapping carefully to improve stability.",
  },

  {
    id: "mit6s191-l5-q14",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statements about exploration are correct?",
    options: [
      {
        text: "Sampling from a stochastic policy can lead to trying actions that are not currently the highest-probability choice.",
        isCorrect: true,
      },
      {
        text: "Exploration can help discover strategies that have higher long-term return than the agent’s current behavior.",
        isCorrect: true,
      },
      {
        text: "Exploration is unnecessary whenever rewards are sparse and delayed.",
        isCorrect: false,
      },
      {
        text: "Exploration only matters in supervised learning, not in reinforcement learning.",
        isCorrect: false,
      },
    ],
    explanation:
      "RL often needs exploration, especially with sparse/delayed rewards, because the agent must discover which behaviors lead to good long-term outcomes. Stochastic policies naturally support exploration via sampling.",
  },

  {
    id: "mit6s191-l5-q15",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statement best explains why a return \\(R_t\\) can prefer a 'smaller reward now' action over a 'bigger reward now' action?",
    options: [
      {
        text: "Because \\(R_t\\) includes future rewards, an action with lower immediate reward can still lead to higher total discounted return.",
        isCorrect: true,
      },
      {
        text: "Because \\(R_t\\) ignores all rewards after time \\(t\\).",
        isCorrect: false,
      },
      {
        text: "Because \\(R_t\\) is defined as \\(R_t = r_t\\) for all tasks.",
        isCorrect: false,
      },
      {
        text: "Because maximizing return forces the environment to become deterministic.",
        isCorrect: false,
      },
    ],
    explanation:
      "Returns aggregate future outcomes (often discounted), so the best long-term decision may sacrifice immediate reward to set up better future rewards. This is exactly what differentiates optimizing return from optimizing immediate reward.",
  },

  {
    id: "mit6s191-l5-q16",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe a basic policy gradient training signal in episodic RL?",
    options: [
      {
        text: "A common loss form is \\(\\mathcal{L} = -\\log \\pi_\\theta(a_t\\mid s_t)\\, R_t\\) (or using an advantage estimate).",
        isCorrect: true,
      },
      {
        text: "If \\(R_t\\) is high, gradient updates tend to increase the probability of the taken action under \\(\\pi_\\theta\\).",
        isCorrect: true,
      },
      {
        text: "If \\(R_t\\) is low (or negative), gradient updates tend to decrease the probability of the taken action under \\(\\pi_\\theta\\).",
        isCorrect: true,
      },
      {
        text: "Policy gradient requires computing \\(\\arg\\max_a Q(s,a)\\) at every step.",
        isCorrect: false,
      },
    ],
    explanation:
      "Policy gradient methods push up the log-probability of actions that led to high return and push down those that led to low return. They optimize the policy directly rather than needing an argmax over Q-values at decision time.",
  },

  {
    id: "mit6s191-l5-q17",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about collecting experience in episodic policy learning are correct?",
    options: [
      {
        text: "An episode can be rolled out by repeatedly sampling actions from \\(\\pi_\\theta(\\cdot\\mid s_t)\\) until termination.",
        isCorrect: true,
      },
      {
        text: "You can store a sequence of \\((s_t,a_t,r_t)\\) and later compute returns for training.",
        isCorrect: true,
      },
      {
        text: "If reward only arrives at the end, credit assignment becomes harder because many earlier actions share the same final outcome.",
        isCorrect: true,
      },
      {
        text: "Because episodes end, discounting \\(\\gamma\\) is mathematically impossible.",
        isCorrect: false,
      },
    ],
    explanation:
      "Trajectory collection is a forward-pass process: interact, record, then train. Sparse end-of-episode rewards make it difficult to know which earlier actions helped or hurt. Discounting is still usable in finite-horizon episodes.",
  },

  {
    id: "mit6s191-l5-q18",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about 'termination' and real-world RL are correct?",
    options: [
      {
        text: "In real driving, 'run until termination' can be unsafe because termination might correspond to a crash.",
        isCorrect: true,
      },
      {
        text: "High-fidelity simulators can allow safe rollouts where termination is acceptable, and learned policies may later transfer to the real world.",
        isCorrect: true,
      },
      {
        text: "A major practical challenge is collecting enough diverse experience without causing real-world harm.",
        isCorrect: true,
      },
      {
        text: "Simulation is unnecessary because real-world trial-and-error is always cheaper and safer.",
        isCorrect: false,
      },
    ],
    explanation:
      "Many RL algorithms assume episodes can terminate, but in the real world termination can be dangerous. Simulation can mitigate safety risks and enable large-scale data collection before deployment.",
  },

  {
    id: "mit6s191-l5-q19",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about deterministic Q-learning behavior are correct?",
    options: [
      {
        text: "If a Q-based policy always chooses \\(\\arg\\max_a Q(s,a)\\), it will return the same action for the same state (given the same Q-function).",
        isCorrect: true,
      },
      {
        text: "Deterministic greedy action selection can reduce exploration because the agent rarely tries alternatives.",
        isCorrect: true,
      },
      {
        text: "A deterministic policy can be brittle in stochastic environments where different actions may be needed for similar-looking states.",
        isCorrect: true,
      },
      {
        text: "Determinism guarantees the learned behavior is optimal even if \\(Q\\) is poorly estimated.",
        isCorrect: false,
      },
    ],
    explanation:
      "Greedy action selection is deterministic given a fixed Q-function, which can limit exploration. Determinism can also be problematic in stochastic settings, but optimality is never guaranteed if the value estimates are wrong.",
  },

  {
    id: "mit6s191-l5-q20",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statement is the most accurate description of the 'action space' \\(\\mathcal{A}\\)?",
    options: [
      {
        text: "It is the set of actions the agent is allowed to take in the environment (which may be discrete or continuous).",
        isCorrect: true,
      },
      {
        text: "It is the set of all possible future trajectories of the agent over its entire lifetime.",
        isCorrect: false,
      },
      {
        text: "It is the set of rewards \\(r_t\\) the environment can output.",
        isCorrect: false,
      },
      {
        text: "It is the set of network weights \\(\\theta\\) used to represent the policy.",
        isCorrect: false,
      },
    ],
    explanation:
      "The action space is about what the agent can do at a single step—like move left/right or choose a steering angle. Trajectories, rewards, and parameters are related but are different objects from the action space itself.",
  },

  {
    id: "mit6s191-l5-q21",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about why policy learning can handle continuous actions are correct?",
    options: [
      {
        text: "A policy can output parameters of a continuous distribution (e.g., \\(\\mu\\) and \\(\\sigma\\)) rather than enumerating all actions.",
        isCorrect: true,
      },
      {
        text: "Sampling from a continuous distribution produces a valid continuous action without requiring a discrete argmax.",
        isCorrect: true,
      },
      {
        text: "Softmax is required for continuous actions because probabilities must sum to 1 across infinitely many actions.",
        isCorrect: false,
      },
      {
        text: "Value-based methods can never be used in continuous action spaces under any circumstances.",
        isCorrect: false,
      },
    ],
    explanation:
      "Policy learning commonly handles continuous actions by representing a distribution (often Gaussian) and sampling from it. Softmax is typical for discrete distributions; continuous distributions are normalized by their density function, not by summing finite outputs. Value methods can be extended to continuous actions too, but doing \\(\\arg\\max\\) over continuous actions is harder and often needs extra machinery.",
  },

  {
    id: "mit6s191-l5-q22",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about the log-likelihood term in policy gradient are correct?",
    options: [
      {
        text: "The term \\(\\log \\pi_\\theta(a_t\\mid s_t)\\) increases when the policy assigns higher probability to the taken action.",
        isCorrect: true,
      },
      {
        text: "Using \\(-\\log \\pi_\\theta(a_t\\mid s_t)\\,R_t\\) ties probability updates to how good the observed return was.",
        isCorrect: true,
      },
      {
        text: "If an action was unlikely under the current policy but led to high return, policy gradient can still increase its probability.",
        isCorrect: true,
      },
      {
        text: "The log-likelihood is used only in Q-learning losses, not in policy learning.",
        isCorrect: false,
      },
    ],
    explanation:
      "Policy gradients are built around differentiating the log probability of actions. Multiplying by return (or advantage) makes the update reinforce actions that worked well and suppress those that didn’t, even if they were initially rare.",
  },

  {
    id: "mit6s191-l5-q23",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about sparse reward settings (e.g., reward only at game end) are correct?",
    options: [
      {
        text: "Sparse rewards make credit assignment harder because many actions occur before any reward signal is observed.",
        isCorrect: true,
      },
      {
        text: "One motivation for learning state values is to provide a denser learning signal than only final win/loss.",
        isCorrect: true,
      },
      {
        text: "Sparse rewards guarantee faster convergence because there is less noise in feedback.",
        isCorrect: false,
      },
      {
        text: "Sparse rewards make exploration more important, since the agent must discover which behaviors lead to reward.",
        isCorrect: true,
      },
    ],
    explanation:
      "When feedback is rare, it’s difficult to know which earlier choices mattered. Value estimation can provide intermediate assessments of states, and exploration is often crucial—sparse rewards do not automatically make learning easier.",
  },

  {
    id: "mit6s191-l5-q24",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about the Go example and scaling RL to complex tasks are correct?",
    options: [
      {
        text: "A 19x19 Go board leads to an enormous space of possible positions and long-horizon consequences.",
        isCorrect: true,
      },
      {
        text: "Self-play allows an agent to generate its own training experience by playing against itself.",
        isCorrect: true,
      },
      {
        text: "A value estimate for a board position helps evaluate 'how good the situation is' without waiting until the end of the game.",
        isCorrect: true,
      },
      {
        text: "Because Go is complex, reinforcement learning cannot be used at all and only supervised learning works.",
        isCorrect: false,
      },
    ],
    explanation:
      "Go’s complexity comes from the huge combinatorial space and long-term planning. Self-play and value estimation help create learning signals beyond end-of-game outcomes, enabling RL to reach very strong play.",
  },

  {
    id: "mit6s191-l5-q25",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly distinguish reward \\(r_t\\) from return \\(R_t\\)?",
    options: [
      {
        text: "\\(r_t\\) is typically the immediate feedback received at time \\(t\\) after taking an action.",
        isCorrect: true,
      },
      {
        text: "\\(R_t\\) aggregates multiple future rewards, often with discounting, e.g., \\(R_t = r_t + \\gamma r_{t+1} + \\gamma^2 r_{t+2} + \\dots\\).",
        isCorrect: true,
      },
      {
        text: "\\(r_t\\) and \\(R_t\\) are always numerically identical in any RL task.",
        isCorrect: false,
      },
      {
        text: "If the horizon is more than one step and \\(\\gamma>0\\), \\(R_t\\) generally depends on rewards after time \\(t\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The reward is a single-step signal, while the return is a multi-step cumulative objective (often discounted). They coincide only in special cases (e.g., \\(\\gamma=0\\) or horizon of one step).",
  },

  {
    id: "mit6s191-l5-q26",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about why a probability distribution output needs constraints are correct (in discrete action policy networks)?",
    options: [
      {
        text: "Each output probability should be non-negative.",
        isCorrect: true,
      },
      {
        text: "The probabilities across discrete actions should sum to 1.",
        isCorrect: true,
      },
      {
        text: "A softmax transformation is a common way to map arbitrary logits to a valid probability distribution.",
        isCorrect: true,
      },
      {
        text: "Those constraints are necessary because Q-values must sum to 1.",
        isCorrect: false,
      },
    ],
    explanation:
      "Probabilities must be non-negative and sum to one, and softmax is a standard way to enforce this from unconstrained network outputs (logits). Q-values are not probabilities and are not constrained to sum to one.",
  },

  {
    id: "mit6s191-l5-q27",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about training with recorded trajectories in policy learning are correct?",
    options: [
      {
        text: "You can increase the likelihood of actions that were part of high-return trajectories by updating \\(\\theta\\) to increase \\(\\log \\pi_\\theta(a_t\\mid s_t)\\).",
        isCorrect: true,
      },
      {
        text: "You can decrease the likelihood of actions that were part of low-return trajectories by updating \\(\\theta\\) to reduce \\(\\pi_\\theta(a_t\\mid s_t)\\).",
        isCorrect: true,
      },
      {
        text: "This approach can work even without human demonstrations, relying only on reward signals from the environment.",
        isCorrect: true,
      },
      {
        text: "Because the environment provides reward, the policy network weights never need gradient-based updates.",
        isCorrect: false,
      },
    ],
    explanation:
      "Policy learning uses gradient-based optimization to adjust network weights based on experience and returns. The environment provides the reward signal, but learning still requires computing gradients and updating parameters.",
  },

  {
    id: "mit6s191-l5-q28",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about why humans can misjudge Q-values in some games (e.g., Atari Breakout) are correct?",
    options: [
      {
        text: "Some actions that look locally reasonable may be worse than actions that set up a future 'high payoff' strategy.",
        isCorrect: true,
      },
      {
        text: "Q-values incorporate long-term return, so a seemingly risky move can have higher expected value if it unlocks future rewards.",
        isCorrect: true,
      },
      {
        text: "Human intuition is always aligned with maximizing discounted return in unfamiliar environments.",
        isCorrect: false,
      },
      {
        text: "Estimating the consequences of actions many steps into the future is cognitively hard, especially with complex dynamics.",
        isCorrect: true,
      },
    ],
    explanation:
      "Because Q-values reflect expected long-term return, policies may learn strategies that look unintuitive but yield more future reward. Long-horizon reasoning is difficult for humans, so our local intuition can be misleading.",
  },

  {
    id: "mit6s191-l5-q29",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about why 'calling the model many times' can be expensive in value learning are correct?",
    options: [
      {
        text: "If a Q-network takes \\((s,a)\\) as input, you may need to evaluate it once per candidate action to compare actions.",
        isCorrect: true,
      },
      {
        text: "If a Q-network outputs a vector of Q-values for all actions given \\(s\\), you can get all action values in a single forward pass.",
        isCorrect: true,
      },
      {
        text: "Reducing the number of forward passes can matter because inference cost can be a bottleneck, especially in real-time systems.",
        isCorrect: true,
      },
      {
        text: "Inference cost is irrelevant because only backpropagation is computationally expensive.",
        isCorrect: false,
      },
    ],
    explanation:
      "When you need values for many actions, repeated forward passes can be costly. A vector-output design can provide all action values at once, which can be important in real-time decision loops.",
  },

  {
    id: "mit6s191-l5-q30",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about the relationship between a policy and a value function are correct?",
    options: [
      {
        text: "A policy \\(\\pi\\) defines how actions are selected given states (deterministically or stochastically).",
        isCorrect: true,
      },
      {
        text: "A value function (like \\(Q\\)) evaluates how good actions or states are in terms of expected return.",
        isCorrect: true,
      },
      {
        text: "You can derive a greedy policy from a Q-function via \\(\\pi(s)=\\arg\\max_a Q(s,a)\\) in discrete action settings.",
        isCorrect: true,
      },
      {
        text: "A policy and a value function are identical objects and always have the same output type.",
        isCorrect: false,
      },
    ],
    explanation:
      "Policies choose actions, while value functions score states/actions by expected return. They are related—values can induce a policy—but they are not the same thing and typically output different kinds of quantities.",
  },

  {
    id: "mit6s191-l5-q31",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about why simulation helps with robotics/autonomy RL are correct?",
    options: [
      {
        text: "Simulation can generate large amounts of experience cheaply compared to real-world data collection.",
        isCorrect: true,
      },
      {
        text: "Simulation makes it safer to encounter failures (like crashes) that would be unacceptable in reality.",
        isCorrect: true,
      },
      {
        text: "If simulation is photorealistic enough, policies may transfer better to real-world deployment.",
        isCorrect: true,
      },
      {
        text: "Simulation removes the need to define rewards because the simulator automatically infers human preferences.",
        isCorrect: false,
      },
    ],
    explanation:
      "Simulators help with scale and safety, and realism can improve transfer. But reward design is still required—simulation does not automatically solve the problem of defining what the agent should optimize.",
  },

  {
    id: "mit6s191-l5-q32",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about using softmax for action selection are correct (discrete actions)?",
    options: [
      {
        text: "Softmax maps logits \\(z_i\\) to probabilities \\(p_i = \\frac{e^{z_i}}{\\sum_j e^{z_j}}\\).",
        isCorrect: true,
      },
      {
        text: "If you sample from the softmax probabilities, you get a stochastic policy that can explore.",
        isCorrect: true,
      },
      {
        text: "Softmax is typically used to turn a vector of unconstrained scores into a valid probability distribution.",
        isCorrect: true,
      },
      {
        text: "Softmax guarantees the chosen action is always the \\(\\arg\\max\\) action.",
        isCorrect: false,
      },
    ],
    explanation:
      "Softmax converts arbitrary scores (logits) into a probability distribution. Sampling from that distribution yields exploration and does not always pick the argmax, unlike greedy selection.",
  },

  {
    id: "mit6s191-l5-q33",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about training pipelines for strong game-playing agents (e.g., Go) are correct?",
    options: [
      {
        text: "An agent can be initialized via supervised learning from expert games to learn plausible moves from board states.",
        isCorrect: true,
      },
      {
        text: "Reinforcement learning via self-play can improve the agent beyond the supervised imitation of humans.",
        isCorrect: true,
      },
      {
        text: "A value estimate can reduce reliance on sparse end-of-game rewards by evaluating intermediate positions.",
        isCorrect: true,
      },
      {
        text: "Self-play is incompatible with neural networks because it requires symbolic search only.",
        isCorrect: false,
      },
    ],
    explanation:
      "A common recipe is imitation (optional) plus self-play RL plus value estimation to address sparse rewards. Neural networks are central to modern self-play systems; search can be used too, but it’s not a contradiction.",
  },

  {
    id: "mit6s191-l5-q34",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statements about stochastic policies are correct?",
    options: [
      {
        text: "A stochastic policy can assign nonzero probability to multiple actions in the same state.",
        isCorrect: true,
      },
      {
        text: "Sampling from a stochastic policy can yield different actions on different visits to the same state.",
        isCorrect: true,
      },
      {
        text: "Stochasticity can help in environments where the dynamics or observations are noisy or unpredictable.",
        isCorrect: true,
      },
      {
        text: "Stochastic policies are only useful when the action space is discrete; they never apply to continuous control.",
        isCorrect: false,
      },
    ],
    explanation:
      "Stochastic policies naturally represent uncertainty and enable exploration; they can be used in both discrete and continuous settings. In continuous control, sampling from distributions like Gaussians is a standard approach.",
  },

  {
    id: "mit6s191-l5-q35",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about why large action spaces are challenging are correct?",
    options: [
      {
        text: "If an agent must consider many actions, computing \\(\\arg\\max_a Q(s,a)\\) can become expensive.",
        isCorrect: true,
      },
      {
        text: "Even when per-step legal actions are manageable, exploring multi-step futures can explode combinatorially.",
        isCorrect: true,
      },
      {
        text: "Large action/state spaces often require function approximation (e.g., neural networks) rather than tabular methods.",
        isCorrect: true,
      },
      {
        text: "Large action spaces make the environment’s reward function unnecessary.",
        isCorrect: false,
      },
    ],
    explanation:
      "Scaling RL is hard because of computation and exploration: many actions and long horizons expand the search/learning problem. Function approximation helps represent values/policies compactly, but reward design remains necessary regardless of action space size.",
  },

  {
    id: "mit6s191-l5-q36",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about policy learning vs value learning tradeoffs are correct?",
    options: [
      {
        text: "Value learning with greedy action selection can be fully deterministic given a fixed \\(Q\\) estimate.",
        isCorrect: true,
      },
      {
        text: "Policy learning naturally supports stochastic policies by modeling \\(\\pi(a\\mid s)\\) and sampling actions.",
        isCorrect: true,
      },
      {
        text: "One motivation for policy learning is handling continuous actions without enumerating all actions.",
        isCorrect: true,
      },
      {
        text: "Policy learning cannot be trained with gradient descent because sampling breaks differentiability completely.",
        isCorrect: false,
      },
    ],
    explanation:
      "Policy learning often optimizes a differentiable objective involving \\(\\log \\pi\\) (with estimators that handle sampling). Value learning is straightforward for small discrete actions but can be limiting in continuous or exploration-heavy settings.",
  },

  {
    id: "mit6s191-l5-q37",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statements about discounted return \\(R_t\\) are correct?",
    options: [
      {
        text: "If \\(0<\\gamma<1\\), the contribution of rewards decays geometrically with how far in the future they occur.",
        isCorrect: true,
      },
      {
        text: "If all rewards are bounded, using \\(\\gamma<1\\) helps keep the infinite-horizon sum \\(\\sum_{k=0}^{\\infty} \\gamma^k r_{t+k}\\) finite.",
        isCorrect: true,
      },
      {
        text: "Discounting is one reason why an agent may prioritize near-term outcomes over equally-sized far-future outcomes.",
        isCorrect: true,
      },
      {
        text: "Discounting implies the agent ignores all future rewards beyond the next step.",
        isCorrect: false,
      },
    ],
    explanation:
      "Discounting reduces the weight of distant rewards but does not necessarily ignore them. It yields geometric decay and often improves mathematical and practical stability in long-horizon settings.",
  },

  {
    id: "mit6s191-l5-q38",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about the difference between 'observation' and 'state' are correct in common RL setups?",
    options: [
      {
        text: "In many simplified treatments, the observation is treated as the state input to the policy/value network.",
        isCorrect: true,
      },
      {
        text: "After taking an action, the agent receives a new observation that reflects how the environment changed.",
        isCorrect: true,
      },
      {
        text: "If the observation is incomplete, the agent may need memory (e.g., recurrence) to act well, even if the underlying state is Markov.",
        isCorrect: true,
      },
      {
        text: "Observations are always identical to the full true environment state in all RL tasks.",
        isCorrect: false,
      },
    ],
    explanation:
      "Intro RL often uses 'state' for whatever the agent observes, but in many real problems observations can be partial. Partial observability can require memory or belief-state methods; observations are not always the full underlying state.",
  },

  {
    id: "mit6s191-l5-q39",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statements about 'self-play from scratch' are correct?",
    options: [
      {
        text: "Self-play can generate training data without human demonstrations by having the agent play against itself.",
        isCorrect: true,
      },
      {
        text: "Learning from scratch typically starts from a weak or random policy and improves via repeated interaction and feedback.",
        isCorrect: true,
      },
      {
        text: "Adding a value estimate can help learning by providing a signal about intermediate positions, not only final outcomes.",
        isCorrect: true,
      },
      {
        text: "Self-play requires that the reward be dense at every time step; otherwise it cannot learn.",
        isCorrect: false,
      },
    ],
    explanation:
      "Self-play is a way to produce experience without expert labels. It can still work with sparse rewards, though learning is harder; value estimates and other techniques help cope with delayed feedback.",
  },

  {
    id: "mit6s191-l5-q40",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about using probabilities vs Q-values for selecting actions are correct?",
    options: [
      {
        text: "Q-values are on an arbitrary scale of expected return, while policy probabilities must satisfy normalization constraints.",
        isCorrect: true,
      },
      {
        text: "Sampling from a policy distribution can select sub-maximal actions sometimes, enabling exploration.",
        isCorrect: true,
      },
      {
        text: "Greedy Q-selection chooses the action with maximum estimated return and is deterministic if the Q-function is fixed.",
        isCorrect: true,
      },
      {
        text: "Policy probabilities are always computed by taking \\(\\arg\\max\\) over Q-values.",
        isCorrect: false,
      },
    ],
    explanation:
      "Policies and values serve different roles: values estimate return, while policies define action selection. Sampling from policies supports exploration; greedy selection from Q-values is deterministic given a fixed estimator.",
  },
];
