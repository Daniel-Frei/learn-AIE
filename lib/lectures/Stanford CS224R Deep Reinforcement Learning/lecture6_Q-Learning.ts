import { Question } from "../../quiz";

export const cs224rLecture6QLearningQuestions: Question[] = [
  {
    id: "cs224r-lect6-q01",
    chapter: 6,
    difficulty: "easy",
    prompt:
      "Which statements correctly distinguish value functions and Q-functions in value-based reinforcement learning?",
    options: [
      {
        text: "\\(V^\\pi(s)\\) is the expected discounted return starting in state \\(s\\) and then following policy \\(\\pi\\).",
        isCorrect: true,
      },
      {
        text: "\\(Q^\\pi(s,a)\\) is the expected discounted return after starting in \\(s\\), taking action \\(a\\), and then following \\(\\pi\\).",
        isCorrect: true,
      },
      {
        text: "A Q-function can induce a greedy policy by choosing \\(\\arg\\max_a Q(s,a)\\), while \\(V(s)\\) alone does not identify the best action without a model of transitions.",
        isCorrect: true,
      },
      {
        text: "With discount \\(0\\le\\gamma<1\\), both \\(V^\\pi\\) and \\(Q^\\pi\\) weight rewards farther in the future by higher powers of \\(\\gamma\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The key extra information in \\(Q^\\pi(s,a)\\) is the action argument. That action dependence is what lets a critic-only method extract a policy by maximizing over actions, without separately learning an actor network.",
  },

  {
    id: "cs224r-lect6-q02",
    chapter: 6,
    difficulty: "medium",
    prompt:
      "Suppose \\(Q^\\pi\\) is accurate and a new deterministic policy is defined by \\(\\pi_{new}(a\\mid s)=1\\) for an action in \\(\\arg\\max_a Q^\\pi(s,a)\\) and \\(0\\) otherwise. Which statements are correct?",
    options: [
      {
        text: "The policy-improvement step is guaranteed not to reduce value relative to \\(\\pi\\) when \\(Q^\\pi\\) is exact.",
        isCorrect: true,
      },
      {
        text: "A single greedy improvement step can be better than \\(\\pi\\) without yet being optimal in every state.",
        isCorrect: true,
      },
      {
        text: "If \\(\\pi\\) is already greedy with respect to its own exact \\(Q^\\pi\\), the improvement step leaves the policy unchanged.",
        isCorrect: true,
      },
      {
        text: "The improvement step requires differentiating \\(\\log \\pi_\\theta(a\\mid s)\\) with respect to actor parameters.",
        isCorrect: false,
      },
    ],
    explanation:
      "Greedy policy improvement uses the values in the critic directly. It can be written as an argmax operation, so it is not a policy-gradient update and does not require an actor score function.",
  },

  {
    id: "cs224r-lect6-q03",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "In a sparse-reward grid problem, an old policy always moves right. An exact \\(Q^\\pi\\) says that moving up from one particular row leads to reward after the old policy resumes, but below that row all old-policy Q-values are still \\(0\\). Which conclusions follow for the greedy policy induced by \\(Q^\\pi\\)?",
    options: [
      {
        text: "The greedy policy can improve over the old policy in states where one action already has a higher \\(Q^\\pi(s,a)\\).",
        isCorrect: true,
      },
      {
        text: "The greedy policy need not be globally optimal after one improvement step because the reward has not yet been backed up to all earlier states.",
        isCorrect: true,
      },
      {
        text: "One greedy step must make every state that can reach the reward take the first optimal action toward it.",
        isCorrect: false,
      },
      {
        text: "Using only \\(V^\\pi(s)\\), with no transition model, gives the same direct action choice as \\(\\arg\\max_a Q^\\pi(s,a)\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "A one-step policy improvement only exploits action values that have already become different under the evaluated policy. Sparse rewards often need repeated evaluation and improvement steps so reward information can propagate backward.",
  },

  {
    id: "cs224r-lect6-q04",
    chapter: 6,
    difficulty: "medium",
    prompt:
      "Why is a Q-function more directly useful than a value function for critic-only control when the transition dynamics are unknown?",
    options: [
      {
        text: "Because \\(Q(s,a)\\) can be maximized over candidate actions at the current state to define a policy.",
        isCorrect: true,
      },
      {
        text: "Because \\(V(s)\\) contains a separate component for every possible action in \\(s\\).",
        isCorrect: false,
      },
      {
        text: "Because \\(Q(s,a)\\) removes the need to observe rewards from the environment.",
        isCorrect: false,
      },
      {
        text: "Because \\(V(s)\\) can only be defined for deterministic policies, while \\(Q(s,a)\\) can be defined for stochastic policies.",
        isCorrect: false,
      },
    ],
    explanation:
      "The state value tells how good a state is under a policy, but not which action caused that goodness. A state-action value already scores the action choices, so a greedy controller can be read off from the critic.",
  },

  {
    id: "cs224r-lect6-q05",
    chapter: 6,
    difficulty: "medium",
    prompt:
      "Which steps correctly describe exact policy iteration when it is written in terms of Q-functions?",
    options: [
      {
        text: "Evaluate the current policy by fitting or solving for \\(Q^\\pi(s,a)\\).",
        isCorrect: true,
      },
      {
        text: "Improve the policy by setting \\(\\pi'(s)\\in\\arg\\max_a Q^\\pi(s,a)\\).",
        isCorrect: true,
      },
      {
        text: "Repeat evaluation and improvement because the greedy policy changes which Q-function should be evaluated next.",
        isCorrect: true,
      },
      {
        text: "Skip policy evaluation and update only the data-collection distribution because \\(Q^\\pi\\) is independent of rewards.",
        isCorrect: false,
      },
    ],
    explanation:
      "Policy iteration alternates policy evaluation and policy improvement. Q-learning can be viewed as making the improvement step appear inside the Bellman backup rather than explicitly learning a new actor.",
  },

  {
    id: "cs224r-lect6-q06",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "Compared with the policy-evaluation target \\(r(s,a)+\\gamma\\mathbb E_{a'\\sim\\pi(\\cdot\\mid s')}[Q^\\pi(s',a')]\\), what changes in the Q-learning target \\(r(s,a)+\\gamma\\max_{a'} Q(s',a')\\)?",
    options: [
      {
        text: "The next action is chosen by a maximization, so policy improvement is built into the backup.",
        isCorrect: true,
      },
      {
        text: "The fixed point is the Bellman optimality equation for \\(Q^*\\), not merely the value equation for one specified \\(\\pi\\).",
        isCorrect: true,
      },
      {
        text: "The update becomes an on-policy policy-gradient estimator that needs \\(\\nabla_\\theta\\log\\pi_\\theta(a\\mid s)\\).",
        isCorrect: false,
      },
      {
        text: "The target averages over the action actually stored in the replay buffer at \\(s'\\), so no max over actions is needed.",
        isCorrect: false,
      },
    ],
    explanation:
      "The max turns a policy-evaluation backup into an optimal-control backup. Instead of asking how good the next state is under a named policy, Q-learning asks how good it would be if the next action were chosen greedily.",
  },

  {
    id: "cs224r-lect6-q07",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "Which equation is the Bellman optimality equation for an action-value function in an MDP with transition distribution \\(p(s'\\mid s,a)\\)?",
    options: [
      {
        text: "\\(Q^*(s,a)=\\mathbb E_{a'\\sim\\pi(\\cdot\\mid s)}[r(s,a')+\\gamma Q^*(s,a')]\\).",
        isCorrect: false,
      },
      {
        text: "\\(Q^*(s,a)=r(s,a)+\\gamma\\mathbb E_{s'\\sim p(\\cdot\\mid s,a)}\\left[\\max_{a'} Q^*(s',a')\\right]\\).",
        isCorrect: true,
      },
      {
        text: "\\(Q^*(s,a)=\\max_{a'} r(s,a')+\\gamma\\mathbb E_{s'\\sim p(\\cdot\\mid s,a')}[Q^*(s',a)]\\).",
        isCorrect: false,
      },
      {
        text: "\\(Q^*(s,a)=V^*(s)+\\gamma\\max_{s'}p(s'\\mid s,a)V^*(s')\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The current action \\(a\\) determines the immediate reward and transition distribution. The optimality part enters at the next state through \\(\\max_{a'}Q^*(s',a')\\).",
  },

  {
    id: "cs224r-lect6-q08",
    chapter: 6,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe fitting \\(Q_\\phi\\) from replay-buffer transitions in Q-learning?",
    options: [
      {
        text: "A sampled transition \\((s_i,a_i,r_i,s_i')\\) supplies the supervised-learning input \\((s_i,a_i)\\).",
        isCorrect: true,
      },
      {
        text: "A one-step target can be written \\(y_i=r_i+\\gamma\\max_{a'}Q(s_i',a')\\) for a nonterminal transition.",
        isCorrect: true,
      },
      {
        text: "The loss is commonly a squared temporal-difference error such as \\((Q_\\phi(s_i,a_i)-y_i)^2\\).",
        isCorrect: true,
      },
      {
        text: "The final controller can be recovered by acting greedily with respect to the learned Q-function.",
        isCorrect: true,
      },
    ],
    explanation:
      "Q-learning turns transitions into regression targets for a critic. The learned critic both predicts the Bellman target and supplies the policy through greedy action selection.",
  },

  {
    id: "cs224r-lect6-q09",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "For a nonterminal transition, let \\(r=2\\), \\(\\gamma=0.9\\), and the target-network values at the next state be \\(Q_{\\phi^-}(s',a_1)=3\\), \\(Q_{\\phi^-}(s',a_2)=5\\), and \\(Q_{\\phi^-}(s',a_3)=4\\). Which statements are correct?",
    options: [
      {
        text: "The one-step Q-learning target is \\(2+0.9\\cdot5=6.5\\).",
        isCorrect: true,
      },
      {
        text: "The greedy next action in the target is \\(a_2\\).",
        isCorrect: true,
      },
      {
        text: "The target is \\(2+0.9\\cdot(3+5+4)/3=5.6\\) because Q-learning averages next actions.",
        isCorrect: false,
      },
      {
        text: "The target is \\(0.9\\cdot5=4.5\\) because the immediate reward is already represented inside \\(Q_{\\phi^-}\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "For a nonterminal one-step target, Q-learning adds the observed immediate reward to the discounted maximum next-state action value. The average over next actions would be a different, policy-evaluation-style backup.",
  },

  {
    id: "cs224r-lect6-q10",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "For a fixed target \\(y_i\\), define \\(L_i(\\phi)=\\frac12(Q_\\phi(s_i,a_i)-y_i)^2\\). Which stochastic-gradient update has the correct sign for minimizing this loss?",
    options: [
      {
        text: "\\(\\phi\\leftarrow\\phi+\\alpha(Q_\\phi(s_i,a_i)-y_i)\\nabla_\\phi Q_\\phi(s_i,a_i)\\).",
        isCorrect: false,
      },
      {
        text: "\\(\\phi\\leftarrow\\phi-\\alpha(y_i-Q_\\phi(s_i,a_i))\\nabla_\\phi y_i\\), with gradients flowing only through the target.",
        isCorrect: false,
      },
      {
        text: "\\(\\phi\\leftarrow\\phi-\\alpha(Q_\\phi(s_i,a_i)-y_i)\\nabla_\\phi Q_\\phi(s_i,a_i)\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\phi\\leftarrow\\phi-\\alpha\\nabla_\\phi\\max_{a'}Q_\\phi(s_i,a')\\), ignoring the sampled action \\(a_i\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "With the target treated as fixed, the derivative of the half-squared error is the prediction error times the gradient of the prediction. Gradient descent subtracts that quantity.",
  },

  {
    id: "cs224r-lect6-q11",
    chapter: 6,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe data coverage and off-policy sampling in Q-learning with a replay buffer?",
    options: [
      {
        text: "The replay buffer must cover the state-action pairs whose Q-values the learner needs to estimate well.",
        isCorrect: true,
      },
      {
        text: "A purely deterministic greedy behavior policy can fail to try alternative actions whose Q-values matter for the max backup.",
        isCorrect: true,
      },
      {
        text: "Using a replay buffer is still called off-policy when the buffer mixes data from current and earlier behavior policies.",
        isCorrect: true,
      },
      {
        text: "The Bellman optimality max removes the need to observe rewards after non-greedy actions.",
        isCorrect: false,
      },
    ],
    explanation:
      "The max backup can ask about actions that were not chosen often. Exploration policies and replay-buffer management matter because the critic cannot accurately learn Q-values for parts of the state-action space it never sees.",
  },

  {
    id: "cs224r-lect6-q12",
    chapter: 6,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe exploration policies commonly paired with Q-learning?",
    options: [
      {
        text: "An \\(\\epsilon\\)-greedy policy mostly takes an action in \\(\\arg\\max_a Q(s,a)\\) and sometimes takes a random action.",
        isCorrect: true,
      },
      {
        text: "It is common to start with larger \\(\\epsilon\\) for exploration and decrease it later as Q-values become more useful.",
        isCorrect: true,
      },
      {
        text: "A Boltzmann policy samples actions with probabilities proportional to a function such as \\(\\exp(Q(s,a))\\).",
        isCorrect: true,
      },
      {
        text: "Both \\(\\epsilon\\)-greedy and Boltzmann exploration are ways to collect broader data than a purely greedy policy would collect.",
        isCorrect: true,
      },
    ],
    explanation:
      "Q-learning usually learns a greedy policy but trains from exploratory behavior. The behavior policy is allowed to differ from the final greedy controller because the critic update is off-policy.",
  },

  {
    id: "cs224r-lect6-q13",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "There are four discrete actions and a unique greedy action. An \\(\\epsilon\\)-greedy behavior policy chooses the greedy action with probability \\(1-\\epsilon\\) and otherwise samples uniformly from all four actions. If \\(\\epsilon=0.2\\), which probabilities are correct?",
    options: [
      {
        text: "The greedy action has probability \\(0.8+0.2/4=0.85\\).",
        isCorrect: true,
      },
      {
        text: "Each non-greedy action has probability \\(0.2/4=0.05\\).",
        isCorrect: true,
      },
      {
        text: "The greedy action has probability \\(0.8\\) because random exploration never selects it.",
        isCorrect: false,
      },
      {
        text: "Each non-greedy action has probability \\(0.2/3\\) under the stated sampling rule.",
        isCorrect: false,
      },
    ],
    explanation:
      "Under the stated convention, the random exploration branch samples from all actions, so the greedy action receives both its exploitation mass and its random-action share.",
  },

  {
    id: "cs224r-lect6-q14",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "For a Boltzmann exploration rule \\(\\pi(a\\mid s)\\propto \\exp(Q(s,a))\\), suppose \\(Q(s,A)=2\\) and \\(Q(s,B)=0\\). Which probability ratio is correct?",
    options: [
      {
        text: "\\(\\pi(A\\mid s)/\\pi(B\\mid s)=2\\), because the ratio equals the Q-value difference.",
        isCorrect: false,
      },
      {
        text: "\\(\\pi(A\\mid s)/\\pi(B\\mid s)=e^{-2}\\), because higher Q-values are downweighted during exploration.",
        isCorrect: false,
      },
      {
        text: "\\(\\pi(A\\mid s)/\\pi(B\\mid s)=1\\), because Boltzmann exploration is uniform over all actions.",
        isCorrect: false,
      },
      {
        text: "\\(\\pi(A\\mid s)/\\pi(B\\mid s)=e^2\\), because the normalizing constant cancels in the ratio.",
        isCorrect: true,
      },
    ],
    explanation:
      "Boltzmann probabilities are exponential in the Q-values. The shared normalizer cancels when two action probabilities are divided, leaving \\(\\exp(Q(A)-Q(B))\\), so the ratio is exponential in the Q gap rather than linear.",
  },

  {
    id: "cs224r-lect6-q15",
    chapter: 6,
    difficulty: "medium",
    prompt:
      "Which steps belong in the full Q-learning-with-replay loop shown by the algorithmic slides?",
    options: [
      {
        text: "Collect transitions \\((s_i,a_i,s_i',r_i)\\) using some behavior policy and add them to the replay buffer \\(\\mathcal R\\).",
        isCorrect: true,
      },
      {
        text: "Sample minibatches of transitions from \\(\\mathcal R\\) for critic updates.",
        isCorrect: true,
      },
      {
        text: "Update \\(\\phi\\) using a TD error containing \\(Q_\\phi(s_i,a_i)-[r_i+\\gamma\\max_{a'}Q(s_i',a')]\\).",
        isCorrect: true,
      },
      {
        text: "Deploy the final policy by keeping the same exploratory \\(\\epsilon\\)-greedy behavior distribution forever.",
        isCorrect: false,
      },
    ],
    explanation:
      "The training behavior policy is used for data collection, but the learned control policy is usually the greedy policy extracted from the final Q-function. Exploration is a training device, not the definition of optimality.",
  },

  {
    id: "cs224r-lect6-q16",
    chapter: 6,
    difficulty: "medium",
    prompt:
      "Which statements correctly interpret the inner update loop \\(K\\) in replay-buffer Q-learning?",
    options: [
      {
        text: "\\(K=1\\) means one critic-update pass per outer collection step, while larger \\(K\\) reuses replay data more heavily.",
        isCorrect: true,
      },
      {
        text: "Larger \\(K\\) can be more sample-efficient because each collected transition can contribute to more gradient updates.",
        isCorrect: true,
      },
      {
        text: "If the bootstrap target uses the same changing \\(Q_\\phi\\), the label can move whenever \\(\\phi\\) changes.",
        isCorrect: true,
      },
      {
        text: "A target network makes the inner loop closer to supervised learning by holding the bootstrap labels fixed for a while.",
        isCorrect: true,
      },
    ],
    explanation:
      "The inner-loop picture highlights the stability problem: the learner is chasing labels computed from its own changing network. Freezing the bootstrap network slows the target motion.",
  },

  {
    id: "cs224r-lect6-q17",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "Let \\(Q_{\\phi^-}\\) denote a frozen target network copied from \\(Q_\\phi\\) periodically. Which statements correctly describe its use?",
    options: [
      {
        text: "A target-network sync can be written \\(\\phi^-\\leftarrow\\phi\\).",
        isCorrect: true,
      },
      {
        text: "A nonterminal target can use \\(y_i=r_i+\gamma\\max_{a'}Q_{\\phi^-}(s_i',a')\\).",
        isCorrect: true,
      },
      {
        text: "Gradients from \\((Q_\\phi(s_i,a_i)-y_i)^2\\) should update \\(\\phi^-\\) through the target on every inner-loop step.",
        isCorrect: false,
      },
      {
        text: "The deployed policy should be greedy with respect to \\(Q_{\\phi^-}\\) only, never with respect to the learned current network \\(Q_\\phi\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The target network is a delayed copy used to compute more stable bootstrap labels. The prediction network is still the one being optimized and is usually the network used for the final greedy controller.",
  },

  {
    id: "cs224r-lect6-q18",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "A DQN implementation copies \\(\\phi^-\\leftarrow\\phi\\) every 100 critic updates. At update 37 after the last copy, which target construction matches the target-network idea?",
    options: [
      {
        text: "Predict with \\(Q_\\phi(s_i,a_i)\\), but compute the bootstrap term with the still-frozen \\(Q_{\\phi^-}(s_i',a')\\).",
        isCorrect: true,
      },
      {
        text: "Predict with \\(Q_{\\phi^-}(s_i,a_i)\\), but compute the bootstrap term with the changing \\(Q_\\phi(s_i',a')\\).",
        isCorrect: false,
      },
      {
        text: "Copy \\(\\phi^-\\leftarrow\\phi\\) before every minibatch so that target and prediction always change together.",
        isCorrect: false,
      },
      {
        text: "Use only the immediate reward \\(r_i\\) until the next copy, because the target network is stale.",
        isCorrect: false,
      },
    ],
    explanation:
      "During the frozen period, the bootstrap labels are computed from the delayed network, while the current network is trained to match those labels. The target is stale by design, but it still supplies the discounted next-state value.",
  },

  {
    id: "cs224r-lect6-q19",
    chapter: 6,
    difficulty: "medium",
    prompt:
      "Which statements correctly connect target networks, replay buffers, and DQN?",
    options: [
      {
        text: "DQN can be viewed as Q-learning with a neural-network critic, replay data, and a frozen target network.",
        isCorrect: true,
      },
      {
        text: "Holding \\(Q_{\\phi^-}\\) fixed makes labels inside the inner loop less nonstationary.",
        isCorrect: true,
      },
      {
        text: "The algorithm is not ordinary fixed-dataset supervised learning forever because the replay buffer and target network continue to evolve over training.",
        isCorrect: true,
      },
      {
        text: "DQN requires a separately parameterized actor network whose gradient is updated by the policy-gradient theorem.",
        isCorrect: false,
      },
    ],
    explanation:
      "DQN stabilizes a critic-only control method rather than adding an actor. Its inner loop resembles supervised regression, but the data and delayed labels still come from an online RL process.",
  },

  {
    id: "cs224r-lect6-q20",
    chapter: 6,
    difficulty: "medium",
    prompt:
      "Which diagnostic interpretations are reasonable when training DQN-style Q-learning?",
    options: [
      {
        text: "Rising predicted Q-values can be consistent with rising returns because Q-values estimate future reward.",
        isCorrect: true,
      },
      {
        text: "The TD loss can increase during training if new, higher-return data changes the scale of the targets.",
        isCorrect: true,
      },
      {
        text: "Predicted Q-values that sit far above realized returns are evidence of overestimation.",
        isCorrect: true,
      },
      {
        text: "Visualizing action-specific Q-values can reveal whether the critic assigns high value to actions that are strategically important in a state.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture distinguishes useful rising Q-values from overestimated Q-values. Loss curves alone can be misleading because the target distribution changes as the agent discovers reward.",
  },

  {
    id: "cs224r-lect6-q21",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "Why can the term \\(\\max_{a'}Q_\\phi(s',a')\\) create overestimation bias when \\(Q_\\phi\\) is noisy?",
    options: [
      {
        text: "The max operator tends to select actions whose estimation noise is positive.",
        isCorrect: true,
      },
      {
        text: "The same noisy Q-function both selects the action and evaluates that selected action.",
        isCorrect: true,
      },
      {
        text: "Overestimation occurs only when \\(\\gamma>1\\), so valid discount factors avoid it.",
        isCorrect: false,
      },
      {
        text: "The bias is caused by replay buffers and disappears completely if every update is on-policy.",
        isCorrect: false,
      },
    ],
    explanation:
      "Maximization over noisy estimates is biased upward because the selected action is partly selected for its noise. Coupling selection and evaluation through the same critic amplifies that effect.",
  },

  {
    id: "cs224r-lect6-q22",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "Which update has the Double Q-learning structure where one Q-function selects the next action and the other evaluates it?",
    options: [
      {
        text: "\\(Q_A(s,a)\\leftarrow r+\\gamma Q_A(s',\\arg\\max_{a'}Q_A(s',a'))\\).",
        isCorrect: false,
      },
      {
        text: "\\(Q_A(s,a)\\leftarrow r+\\gamma Q_B(s',\\arg\\max_{a'}Q_A(s',a'))\\).",
        isCorrect: true,
      },
      {
        text: "\\(Q_A(s,a)\\leftarrow r+\\gamma\\min_{a'}Q_B(s',a')\\).",
        isCorrect: false,
      },
      {
        text: "\\(Q_A(s,a)\\leftarrow r+\\gamma\\mathbb E_{a'\\sim\\pi_B}[Q_A(s',a')]\\), with no greedy selection.",
        isCorrect: false,
      },
    ],
    explanation:
      "Double Q-learning breaks the same-network max by selecting with one estimator and evaluating with another. The target still represents greedy control, not an expectation under a behavior policy.",
  },

  {
    id: "cs224r-lect6-q23",
    chapter: 6,
    difficulty: "medium",
    prompt:
      "Which statements correctly explain the purpose and limits of Double Q-learning or Double DQN?",
    options: [
      {
        text: "Separating action selection from value evaluation can decorrelate estimation errors.",
        isCorrect: true,
      },
      {
        text: "The method usually reduces overestimation but does not mathematically guarantee zero bias with neural networks.",
        isCorrect: true,
      },
      {
        text: "In practice, the current network and the target network can provide the two Q-functions used in the double target.",
        isCorrect: true,
      },
      {
        text: "The method requires two different behavior policies collecting two different replay buffers.",
        isCorrect: false,
      },
    ],
    explanation:
      "The double idea is about the bootstrap target, not about needing two data-collection streams. It tries to prevent the same critic error from deciding and scoring the next action.",
  },

  {
    id: "cs224r-lect6-q24",
    chapter: 6,
    difficulty: "medium",
    prompt:
      "Which statements correctly compare the standard DQN target and the Double DQN target using a target network \\(Q_{\\phi^-}\\)?",
    options: [
      {
        text: "A standard DQN target can be written \\(r+\\gamma Q_{\\phi^-}(s',\\arg\\max_{a'}Q_{\\phi^-}(s',a'))\\).",
        isCorrect: true,
      },
      {
        text: "A Double DQN target can be written \\(r+\\gamma Q_{\\phi^-}(s',\\arg\\max_{a'}Q_\\phi(s',a'))\\).",
        isCorrect: true,
      },
      {
        text: "The target network still evaluates the selected next action in Double DQN, preserving a delayed bootstrap value.",
        isCorrect: true,
      },
      {
        text: "The selection/evaluation split is intended to reduce overestimation from the max over noisy Q-values.",
        isCorrect: true,
      },
    ],
    explanation:
      "Double DQN keeps the stabilizing target network but changes which network chooses the next action. The current network chooses the argmax; the target network evaluates that chosen action.",
  },

  {
    id: "cs224r-lect6-q25",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "At a next state, the current network has \\(Q_\\phi(s',A)=10\\), \\(Q_\\phi(s',B)=9\\), while the target network has \\(Q_{\\phi^-}(s',A)=4\\), \\(Q_{\\phi^-}(s',B)=8\\). With \\(r=1\\) and \\(\\gamma=0.5\\), which statements are correct?",
    options: [
      {
        text: "The standard DQN target using the target network for both selection and evaluation is \\(1+0.5\\cdot8=5\\).",
        isCorrect: true,
      },
      {
        text: "The Double DQN target selects \\(A\\) using \\(Q_\\phi\\) and evaluates it as \\(1+0.5\\cdot4=3\\).",
        isCorrect: true,
      },
      {
        text: "The Double DQN target selects \\(B\\) because \\(Q_{\\phi^-}(s',B)>Q_{\\phi^-}(s',A)\\).",
        isCorrect: false,
      },
      {
        text: "Both targets are \\(1+0.5\\cdot10=6\\) because the largest current-network value always supplies the evaluated value.",
        isCorrect: false,
      },
    ],
    explanation:
      "Standard DQN uses the target network's own max. Double DQN lets the current network choose the action, but it still plugs that action into the target network for the numeric bootstrap value.",
  },

  {
    id: "cs224r-lect6-q26",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "Which formula is the N-step Q-learning target for a trajectory segment beginning at \\((s_{j,t},a_{j,t})\\), using a target network for the final bootstrap?",
    options: [
      {
        text: "\\(y_{j,t}=r_{j,t}+\\gamma r_{j,t+1}+\\cdots+\\gamma^N r_{j,t+N}+\\max_a Q_{\\phi^-}(s_{j,t},a)\\).",
        isCorrect: false,
      },
      {
        text: "\\(y_{j,t}=\\sum_{t'=t}^{t+N-1}r_{j,t'}+\\max_a Q_{\\phi^-}(s_{j,t+N},a)\\), with no discount factors.",
        isCorrect: false,
      },
      {
        text: "\\(y_{j,t}=\\sum_{t'=t}^{t+N-1}\\gamma^{t'-t}r_{j,t'}+\\gamma^N\\max_a Q_{\\phi^-}(s_{j,t+N},a)\\).",
        isCorrect: true,
      },
      {
        text: "\\(y_{j,t}=\\mathbb E_{a'\\sim\\pi(\\cdot\\mid s_{j,t})}[Q_{\\phi^-}(s_{j,t},a')]\\), with no observed rewards.",
        isCorrect: false,
      },
    ],
    explanation:
      "An N-step target uses the observed discounted rewards for the next \\(N\\) steps, then bootstraps once from the state at time \\(t+N\\). The bootstrap is discounted by \\(\\gamma^N\\).",
  },

  {
    id: "cs224r-lect6-q27",
    chapter: 6,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the tradeoff introduced by N-step targets in Q-learning?",
    options: [
      {
        text: "They rely less on a possibly inaccurate one-step bootstrap value early in training.",
        isCorrect: true,
      },
      {
        text: "They can speed learning by propagating several observed rewards in one update.",
        isCorrect: true,
      },
      {
        text: "When \\(N=1\\), the target reduces to the usual one-step Q-learning target.",
        isCorrect: true,
      },
      {
        text: "For any \\(N>1\\), they are exactly off-policy correct no matter which actions the behavior policy took.",
        isCorrect: false,
      },
    ],
    explanation:
      "N-step returns trade a longer sequence of real rewards for a later bootstrap. That often helps early learning, but the intermediate rewards are rewards under the behavior trajectory, which matters in off-policy control.",
  },

  {
    id: "cs224r-lect6-q28",
    chapter: 6,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the off-policy caveat for N-step Q-learning targets and possible responses?",
    options: [
      {
        text: "For \\(N>1\\), rewards after the first action come from the behavior trajectory, not necessarily the greedy policy whose value is being estimated.",
        isCorrect: true,
      },
      {
        text: "A common practical response is to use \\(N>1\\) anyway because performance often improves despite the mismatch.",
        isCorrect: true,
      },
      {
        text: "One mitigation is to shorten or choose \\(N\\) dynamically when the replayed actions stop matching the current greedy policy.",
        isCorrect: true,
      },
      {
        text: "Importance sampling can be used in principle to correct for action-distribution mismatch along the multi-step segment.",
        isCorrect: true,
      },
    ],
    explanation:
      "The N-step target is most cleanly correct when the replayed actions match the policy being evaluated. In off-policy Q-learning, practitioners often accept the approximation or add corrections such as dynamic truncation or importance weights.",
  },

  {
    id: "cs224r-lect6-q29",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "Use a 3-step target with rewards \\(r_t=1\\), \\(r_{t+1}=2\\), \\(r_{t+2}=4\\), discount \\(\\gamma=0.5\\), and final bootstrap \\(\\max_a Q_{\\phi^-}(s_{t+3},a)=8\\). Which statements are correct?",
    options: [
      {
        text: "The discounted reward part is \\(1+0.5\\cdot2+0.5^2\\cdot4=3\\).",
        isCorrect: true,
      },
      {
        text: "The full target is \\(3+0.5^3\\cdot8=4\\).",
        isCorrect: true,
      },
      {
        text: "The full target is \\(1+2+4+8=15\\) because N-step returns do not discount rewards.",
        isCorrect: false,
      },
      {
        text: "The full target is \\(1+0.5\\cdot8=5\\) because all intermediate rewards after \\(r_t\\) are ignored.",
        isCorrect: false,
      },
    ],
    explanation:
      "For \\(N=3\\), the target includes three discounted observed rewards and then a bootstrap term discounted by \\(\\gamma^3\\). The final value is not added at full weight because it starts three time steps after \\((s_t,a_t)\\).",
  },

  {
    id: "cs224r-lect6-q30",
    chapter: 6,
    difficulty: "medium",
    prompt:
      "Why is the off-policy reward-sequence problem for N-step Q-learning not present in the same way when \\(N=1\\)?",
    options: [
      {
        text: "Because \\(N=1\\) removes bootstrapping entirely and becomes a Monte Carlo return.",
        isCorrect: false,
      },
      {
        text: "Because \\(N=1\\) is correct only when \\(\\gamma=0\\).",
        isCorrect: false,
      },
      {
        text: "Because \\(N=1\\) requires the behavior policy and greedy policy to be identical.",
        isCorrect: false,
      },
      {
        text: "Because the only observed reward is the reward for the sampled \\((s_t,a_t)\\); no later behavior-policy actions appear before the bootstrap.",
        isCorrect: true,
      },
    ],
    explanation:
      "The one-step target uses the immediate reward for the transition being updated, then immediately bootstraps with a max at the next state. Multi-step targets additionally include rewards caused by the behavior policy's later actions.",
  },

  {
    id: "cs224r-lect6-q31",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "Which statements correctly characterize convergence and stability claims for Q-learning?",
    options: [
      {
        text: "With tabular representations and adequate coverage, Q-learning has stronger convergence guarantees than arbitrary neural-network Q-learning.",
        isCorrect: true,
      },
      {
        text: "With function approximation, bootstrapping, and replay, the target can be nonstationary enough to make optimization unstable.",
        isCorrect: true,
      },
      {
        text: "Target networks, replay buffers, Double DQN, and N-step returns are practical stabilizers or improvements, not a proof that any finite neural network reaches \\(Q^*\\).",
        isCorrect: true,
      },
      {
        text: "The Bellman optimality equation guarantees that stochastic gradient descent on any neural network will converge globally from any initialization.",
        isCorrect: false,
      },
    ],
    explanation:
      "The exact Bellman equation defines the desired fixed point, but deep Q-learning is an approximate optimization procedure. The practical tricks help, but they do not turn neural-network training into a tabular proof.",
  },

  {
    id: "cs224r-lect6-q32",
    chapter: 6,
    difficulty: "easy",
    prompt:
      "Which high-level algorithm-selection statements are consistent with the online RL methods summarized in the lecture?",
    options: [
      {
        text: "PPO-style methods are often chosen for stability and ease of use when data efficiency is less important.",
        isCorrect: true,
      },
      {
        text: "DQN-style methods are natural when actions are discrete or low-dimensional enough that maximizing over actions is tractable.",
        isCorrect: true,
      },
      {
        text: "SAC-style methods are often attractive when data efficiency matters, but they can require more tuning.",
        isCorrect: true,
      },
      {
        text: "High-dimensional continuous actions make critic-only Q-learning harder because the \\(\\arg\\max_a Q(s,a)\\) step is itself an optimization problem.",
        isCorrect: true,
      },
    ],
    explanation:
      "The Q-learning machinery depends on being able to search over actions. That is easy in small discrete spaces and harder in high-dimensional continuous spaces, which is one reason actor-critic methods remain important.",
  },

  {
    id: "cs224r-lect6-q33",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "For continuous action spaces, which statements correctly describe the \\(\\arg\\max_a Q(s,a)\\) issue in Q-learning?",
    options: [
      {
        text: "The max over actions becomes a continuous optimization problem rather than a finite enumeration.",
        isCorrect: true,
      },
      {
        text: "For low-dimensional actions, sampling candidate actions and taking the one with the highest Q-value can be a plausible approximate strategy.",
        isCorrect: true,
      },
      {
        text: "The max can always be computed exactly by evaluating \\(Q(s,a)\\) once at the action stored in the replay buffer.",
        isCorrect: false,
      },
      {
        text: "Continuous actions remove the need for exploration because every action is automatically covered by a single transition.",
        isCorrect: false,
      },
    ],
    explanation:
      "The action maximization step is easy to write and sometimes hard to solve. Approximate maximization can work in low-dimensional spaces, but one replayed action does not represent the continuum.",
  },

  {
    id: "cs224r-lect6-q34",
    chapter: 6,
    difficulty: "medium",
    prompt:
      "Which statement best distinguishes the behavior policy used during Q-learning from the policy usually deployed after learning?",
    options: [
      {
        text: "Training may use exploratory behavior such as \\(\\epsilon\\)-greedy or Boltzmann sampling, while deployment often uses \\(\\arg\\max_a Q_\\phi(s,a)\\).",
        isCorrect: true,
      },
      {
        text: "Training must use the exact final greedy policy at every step or the update is no longer Q-learning.",
        isCorrect: false,
      },
      {
        text: "Deployment must keep the same exploration probability used at the beginning of training so that the replay distribution is unchanged.",
        isCorrect: false,
      },
      {
        text: "The behavior policy is defined by \\(\\nabla_\\theta\\log\\pi_\\theta(a\\mid s)\\), while the deployed policy is defined by TD-error magnitude.",
        isCorrect: false,
      },
    ],
    explanation:
      "Q-learning is off-policy, so the data-collection policy can be exploratory. The policy implied by the learned Q-function is the greedy policy, even if that was not the exact policy used to gather every transition.",
  },

  {
    id: "cs224r-lect6-q35",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "Which mathematical checks help distinguish a correct Bellman-optimality Q-learning update from nearby but wrong updates?",
    options: [
      {
        text: "For nonterminal transitions, the bootstrap should use values at \\(s'\\), not a max over actions at the original state \\(s\\).",
        isCorrect: true,
      },
      {
        text: "Replacing \\(\\max_{a'}Q(s',a')\\) with \\(\\mathbb E_{a'\\sim\\pi}[Q(s',a')]\\) changes the target from optimal-control backup toward policy evaluation.",
        isCorrect: true,
      },
      {
        text: "When \\(Q_\\phi(s_i,a_i)=y_i\\) for a sampled transition target, that transition's squared TD error is zero.",
        isCorrect: true,
      },
      {
        text: "The immediate reward can be omitted from every nonterminal target because the max over next actions already includes it.",
        isCorrect: false,
      },
    ],
    explanation:
      "The Bellman optimality target has three distinctive pieces: the immediate reward, the next-state bootstrap, and a max over next actions. Moving the max to the wrong state or replacing it with a policy expectation changes the algorithmic meaning.",
  },
];
