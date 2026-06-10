import { Question } from "../../../quiz";

export const CrashCourseProbabilityL4Questions: Question[] = [
  {
    id: "crash-probability-l4-q01",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statement best captures the main probabilistic shift from supervised learning to reinforcement learning?",
    options: [
      {
        text: "Supervised learning usually fits predictions to fixed examples, while reinforcement learning chooses actions whose consequences unfold over time.",
        isCorrect: true,
      },
      {
        text: "Supervised learning is defined by deterministic labels, while reinforcement learning is defined by deterministic rewards.",
        isCorrect: false,
      },
      {
        text: "Supervised learning optimizes expected future reward, while reinforcement learning minimizes one-step label loss.",
        isCorrect: false,
      },
      {
        text: "Supervised learning and reinforcement learning differ mainly by whether logits are converted with softmax.",
        isCorrect: false,
      },
    ],
    explanation:
      "Reinforcement learning adds action and time: the agent chooses actions, receives rewards, and changes the future states it will see. Supervised learning can still use probability, but its usual setup is prediction from a given input rather than sequential decision-making.",
  },
  {
    id: "crash-probability-l4-q02",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "In the interaction loop \\(S_t \\rightarrow A_t \\rightarrow R_{t+1},S_{t+1}\\), which interpretations are correct?",
    options: [
      {
        text: "\\(S_t\\) is the state available before the agent chooses its action.",
        isCorrect: true,
      },
      {
        text: "\\(A_t\\) is the action chosen at time \\(t\\).",
        isCorrect: true,
      },
      {
        text: "\\(R_{t+1}\\) is the previous reward from before state \\(S_t\\) was observed.",
        isCorrect: false,
      },
      {
        text: "\\(S_{t+1}\\) is treated like a fixed supervised label that is independent of \\(A_t\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The time index emphasizes sequence: the agent observes \\(S_t\\), acts with \\(A_t\\), and then receives the next reward and next state. The next state can depend on the chosen action, which is a central difference from a fixed supervised dataset.",
  },
  {
    id: "crash-probability-l4-q03",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "A gridworld robot chooses `up` from state \\(s\\). The environment moves it above with probability \\(0.8\\), left with probability \\(0.1\\), and right with probability \\(0.1\\). Which statements are correct?",
    options: [
      {
        text: "\\(P(\\text{above}\\mid s,\\text{up})=0.8\\).",
        isCorrect: true,
      },
      {
        text: "The probabilities define a conditional distribution over possible next states.",
        isCorrect: true,
      },
      {
        text: "The probabilities sum to one, so the transition model is valid for this state-action pair.",
        isCorrect: true,
      },
      {
        text: "The environment is deterministic because the action `up` is specified.",
        isCorrect: false,
      },
    ],
    explanation:
      "A transition model \\(P(s'\\mid s,a)\\) is a conditional distribution over next states after a state-action pair. Here the same state and action can lead to several next states, so the environment is stochastic even though the action is named exactly.",
  },
  {
    id: "crash-probability-l4-q04",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which examples correctly identify a reinforcement-learning state, action, or reward component?",
    options: [
      {
        text: "For a chess agent, the board position can be part of the state.",
        isCorrect: true,
      },
      {
        text: "For a robot arm, a motor command can be an action.",
        isCorrect: true,
      },
      {
        text: "For a recommender, a click or watch-time signal can contribute to reward.",
        isCorrect: true,
      },
      {
        text: "For a chatbot preference system, the prompt or conversation context can be part of the state.",
        isCorrect: true,
      },
    ],
    explanation:
      "States describe the current situation, actions are choices available to the agent, and rewards are feedback signals. The exact representation depends on the domain, but the same state-action-reward structure appears in games, robotics, recommendation, and language-model alignment settings.",
  },
  {
    id: "crash-probability-l4-q05",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "A recommendation action has possible rewards \\(+5\\) with probability \\(0.2\\), \\(+1\\) with probability \\(0.1\\), and \\(0\\) with probability \\(0.7\\). Which statements are correct?",
    options: [
      {
        text: "The expected immediate reward is \\(5\\cdot0.2+1\\cdot0.1+0\\cdot0.7=1.1\\).",
        isCorrect: true,
      },
      {
        text: "The expected reward calculation weights each reward by its probability.",
        isCorrect: true,
      },
      {
        text: "The expected immediate reward is \\(5+1+0=6\\) because the reward values are summed without probabilities.",
        isCorrect: false,
      },
      {
        text: "The action is deterministic because the rewards are listed in a table.",
        isCorrect: false,
      },
    ],
    explanation:
      "An expectation is a probability-weighted average, so each reward value is multiplied by the probability of that outcome. Listing outcomes in a table does not remove stochasticity; the action can still have uncertain consequences.",
  },
  {
    id: "crash-probability-l4-q06",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements correctly treat \\(P(r,s'\\mid s,a)\\) as a probabilistic environment model?",
    options: [
      {
        text: "It can represent uncertainty in both the reward and the next state after a state-action pair.",
        isCorrect: true,
      },
      {
        text: "Marginalizing over rewards gives a transition distribution over next states, \\(P(s'\\mid s,a)=\\sum_r P(r,s'\\mid s,a)\\), in a discrete setting.",
        isCorrect: true,
      },
      {
        text: "The expected immediate reward can be computed by summing reward values weighted by their conditional probabilities.",
        isCorrect: true,
      },
      {
        text: "It is the same object as a policy because both condition on \\(s,a\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "\\(P(r,s'\\mid s,a)\\) describes the environment response after the agent chooses an action, not the agent's rule for choosing the action. It can be used to derive transition probabilities and expected rewards by summing over the relevant outcomes.",
  },
  {
    id: "crash-probability-l4-q07",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statement best expresses the Markov property for an RL state representation?",
    options: [
      {
        text: "Once the current state and action are known, earlier history adds no extra information about the next-state distribution.",
        isCorrect: true,
      },
      {
        text: "The current reward must equal the cumulative reward already received.",
        isCorrect: false,
      },
      {
        text: "The agent must choose actions uniformly at random in each state.",
        isCorrect: false,
      },
      {
        text: "The next state must be deterministic whenever the current state is known.",
        isCorrect: false,
      },
    ],
    explanation:
      "The Markov property is about sufficiency of the current state for predicting the next state once the action is also known. It does not require deterministic transitions, random policies, or rewards that summarize the past.",
  },
  {
    id: "crash-probability-l4-q08",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which entries are standard pieces of a Markov Decision Process \\((\\mathcal{S},\\mathcal{A},P,R,\\gamma)\\)?",
    options: [
      {
        text: "\\(\\mathcal{S}\\), the set of possible states.",
        isCorrect: true,
      },
      {
        text: "\\(\\mathcal{A}\\), the set of possible actions.",
        isCorrect: true,
      },
      {
        text: "\\(P(s'\\mid s,a)\\), transition probabilities for the environment.",
        isCorrect: true,
      },
      {
        text: "\\(\\gamma\\), a discount factor controlling how future rewards are weighted.",
        isCorrect: true,
      },
    ],
    explanation:
      "An MDP formalizes sequential decision-making by specifying states, actions, transition dynamics, rewards, and discounting. These pieces separate the environment's behavior from the agent's policy for choosing actions.",
  },
  {
    id: "crash-probability-l4-q09",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "A chatbot state contains only the latest user message, but earlier conversation turns affect what response is appropriate. Which statements are correct?",
    options: [
      {
        text: "This state representation is likely not Markov because relevant information from the past is missing.",
        isCorrect: true,
      },
      {
        text: "Adding conversation history or memory can make the state representation closer to Markov.",
        isCorrect: true,
      },
      {
        text: "The Markov property fails whenever rewards are numerical.",
        isCorrect: false,
      },
      {
        text: "A non-Markov observation becomes Markov automatically if the policy is stochastic.",
        isCorrect: false,
      },
    ],
    explanation:
      "Whether the Markov property holds depends on what information is included in the state. Stochastic policies do not fix missing information; the representation may need history, memory, or a belief state to summarize what matters for future outcomes.",
  },
  {
    id: "crash-probability-l4-q10",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "A policy in state \\(s\\) assigns \\(\\pi(\\text{left}\\mid s)=0.6\\), \\(\\pi(\\text{right}\\mid s)=0.3\\), and \\(\\pi(\\text{wait}\\mid s)=0.1\\). Which statements are correct?",
    options: [
      {
        text: "The policy is stochastic because it defines a distribution over actions.",
        isCorrect: true,
      },
      {
        text: "The most likely action is `left`.",
        isCorrect: true,
      },
      {
        text: "The agent can still choose `wait` with probability \\(0.1\\).",
        isCorrect: true,
      },
      {
        text: "The policy is deterministic because one action has the largest probability.",
        isCorrect: false,
      },
    ],
    explanation:
      "A stochastic policy can prefer one action while still assigning nonzero probability to alternatives. Deterministic policies choose a single action with probability one, whereas this policy samples from a non-degenerate distribution.",
  },
  {
    id: "crash-probability-l4-q11",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements correctly connect policy networks to probability notation?",
    options: [
      {
        text: "A policy network can map a state \\(s\\) to action logits.",
        isCorrect: true,
      },
      {
        text: "Softmax can convert action logits into \\(\\pi(a\\mid s)\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\pi(a\\mid s)\\) is a conditional distribution over actions given the state.",
        isCorrect: true,
      },
      {
        text: "The sampled action is a decision drawn from, or chosen using, the policy distribution.",
        isCorrect: true,
      },
    ],
    explanation:
      "A neural policy can use the same logits-to-softmax structure as a classifier, but the outputs are action probabilities. The policy distribution and the eventual chosen action are related but distinct objects.",
  },
  {
    id: "crash-probability-l4-q12",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which analogies between language-model generation and reinforcement-learning policies are useful but imperfect?",
    options: [
      {
        text: "An LLM context is analogous to an RL state, and a sampled token is analogous to a chosen action.",
        isCorrect: true,
      },
      {
        text: "A sequence of generated tokens can be viewed as trajectory-like because choices unfold over time.",
        isCorrect: true,
      },
      {
        text: "An LLM next-token distribution and an RL policy have identical objectives because both are trained by next-token cross-entropy.",
        isCorrect: false,
      },
      {
        text: "The analogy means a language model optimizes expected future reward during ordinary pretraining.",
        isCorrect: false,
      },
    ],
    explanation:
      "The analogy helps connect conditional distributions over tokens and actions, but the training objectives can differ. Next-token pretraining imitates text data, while RL objectives optimize reward signals over choices and their consequences.",
  },
  {
    id: "crash-probability-l4-q13",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe a simplified reinforcement learning from human feedback (RLHF) pipeline?",
    options: [
      {
        text: "A language model generates responses or token sequences from prompts.",
        isCorrect: true,
      },
      {
        text: "Preference data or human comparisons can be used to train a reward model.",
        isCorrect: true,
      },
      {
        text: "The language model can then be optimized toward higher reward-model scores rather than only next-token imitation.",
        isCorrect: true,
      },
      {
        text: "RLHF removes the need for a probabilistic output distribution over language-model responses.",
        isCorrect: false,
      },
    ],
    explanation:
      "RLHF uses reward signals to shape model behavior beyond ordinary next-token prediction. The model still produces probabilistic outputs, but the training signal is influenced by preferences or reward scores rather than only matching the next observed token.",
  },
  {
    id: "crash-probability-l4-q14",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "An agent receives \\(R_{t+1}=1\\), \\(R_{t+2}=2\\), and \\(R_{t+3}=10\\) with discount \\(\\gamma=0.5\\). What is the discounted return over these three rewards?",
    options: [
      {
        text: "\\(1+0.5\\cdot2+0.5^2\\cdot10=4.5\\).",
        isCorrect: true,
      },
      {
        text: "\\(1+2+10=13\\), because the finite horizon makes discounting have no effect.",
        isCorrect: false,
      },
      {
        text: "\\(0.5\\cdot1+0.5\\cdot2+0.5\\cdot10=6.5\\), because each listed reward gets the same discount.",
        isCorrect: false,
      },
      {
        text: "\\(1+0.5^2\\cdot2+0.5^3\\cdot10=2.75\\), because the first future reward is discounted twice.",
        isCorrect: false,
      },
    ],
    explanation:
      "Discounted return weights the immediate next reward by one, the following reward by \\(\\gamma\\), and the next by \\(\\gamma^2\\). With \\(\\gamma=0.5\\), the calculation is \\(1+1+2.5=4.5\\).",
  },
  {
    id: "crash-probability-l4-q15",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements about the discount factor \\(\\gamma\\) are correct when rewards are future rewards in \\(G_t\\)?",
    options: [
      {
        text: "\\(\\gamma=0\\) makes only \\(R_{t+1}\\) matter in the discounted return.",
        isCorrect: true,
      },
      {
        text: "A larger \\(\\gamma\\) makes later rewards matter more relative to immediate rewards.",
        isCorrect: true,
      },
      {
        text: "\\(\\gamma\\) is a probability that the agent chooses a random action.",
        isCorrect: false,
      },
      {
        text: "Lowering \\(\\gamma\\) improves the policy whenever short-term rewards are more reliable.",
        isCorrect: false,
      },
    ],
    explanation:
      "The discount factor controls how strongly future rewards are weighted in return calculations. It is not an exploration probability, and choosing a discount factor involves a modeling tradeoff rather than a universal rule that lower is better.",
  },
  {
    id: "crash-probability-l4-q16",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Action A gives return \\(10\\) with probability \\(0.5\\) and \\(0\\) with probability \\(0.5\\). Action B gives return \\(4\\) with probability \\(1\\). Which statements are correct?",
    options: [
      {
        text: "Action A has expected return \\(5\\).",
        isCorrect: true,
      },
      {
        text: "Action B has expected return \\(4\\).",
        isCorrect: true,
      },
      {
        text: "An expected-return objective prefers A even though A is riskier.",
        isCorrect: true,
      },
      {
        text: "Action B has higher expected return because its outcome is certain.",
        isCorrect: false,
      },
    ],
    explanation:
      "Expected return averages outcomes by probability, so A's expectation is \\(10\\cdot0.5+0\\cdot0.5=5\\), while B's is \\(4\\). This calculation compares expectations, not risk preferences or worst-case guarantees.",
  },
  {
    id: "crash-probability-l4-q17",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements correctly interpret value functions?",
    options: [
      {
        text: "\\(V^\\pi(s)=\\mathbb{E}_\\pi[G_t\\mid S_t=s]\\) estimates expected return from a state under policy \\(\\pi\\).",
        isCorrect: true,
      },
      {
        text: "\\(Q^\\pi(s,a)=\\mathbb{E}_\\pi[G_t\\mid S_t=s,A_t=a]\\) estimates expected return after taking action \\(a\\) in state \\(s\\) and then following \\(\\pi\\).",
        isCorrect: true,
      },
      {
        text: "A value function can include delayed future consequences, not only immediate reward.",
        isCorrect: true,
      },
      {
        text: "A greedy action can be selected by choosing an action with maximal estimated \\(Q(s,a)\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "State values and action values are expectations of future return under a policy or after a state-action choice. They are useful precisely because they summarize long-term consequences that may not be visible in the immediate reward.",
  },
  {
    id: "crash-probability-l4-q18",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "A robot has \\(Q(s,\\text{coin})=1\\), \\(Q(s,\\text{goal path})=8\\), and \\(Q(s,\\text{shortcut})=5\\). The coin action has immediate reward \\(+1\\), while the goal path has immediate reward \\(0\\). Which statements are correct?",
    options: [
      {
        text: "A value-based greedy agent would choose `goal path` because it has the highest \\(Q\\)-value.",
        isCorrect: true,
      },
      {
        text: "The example shows that high long-term value can come from an action with low immediate reward.",
        isCorrect: true,
      },
      {
        text: "The agent should choose `coin` because immediate reward is the definition of \\(Q(s,a)\\).",
        isCorrect: false,
      },
      {
        text: "\\(Q(s,a)\\) compares states but not actions within a fixed state.",
        isCorrect: false,
      },
    ],
    explanation:
      "\\(Q(s,a)\\) estimates expected future return after taking an action, so it can rank actions by long-term consequences. Immediate reward is only one part of return and can be outweighed by later rewards or penalties.",
  },
  {
    id: "crash-probability-l4-q19",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements match the Bellman-style intuition that value is recursive?",
    options: [
      {
        text: "A state's value can be understood as expected immediate reward plus discounted value of possible next states.",
        isCorrect: true,
      },
      {
        text: "For action values, \\(Q(s,a)\\) can depend on the distribution of \\((R_{t+1},S_{t+1})\\) after taking \\(a\\).",
        isCorrect: true,
      },
      {
        text: "The recursion is useful because decisions now affect which states and rewards become possible later.",
        isCorrect: true,
      },
      {
        text: "The Bellman idea says future states should be ignored once immediate reward is known.",
        isCorrect: false,
      },
    ],
    explanation:
      "Value functions are recursive because return itself is immediate reward plus discounted future return. The next-state distribution matters because today's action changes the probability of reaching valuable or costly future states.",
  },
  {
    id: "crash-probability-l4-q20",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly connect value-based, policy-based, and actor-critic methods to function approximation?",
    options: [
      {
        text: "A value-based method may learn \\(Q_\\theta(s,a)\\) and choose high-value actions.",
        isCorrect: true,
      },
      {
        text: "A policy-based method may learn \\(\\pi_\\theta(a\\mid s)\\) directly.",
        isCorrect: true,
      },
      {
        text: "An actor-critic method learns both a policy-like actor and a value-like critic.",
        isCorrect: true,
      },
      {
        text: "Neural networks are useful in deep RL because real state spaces can be too large for simple tables.",
        isCorrect: true,
      },
    ],
    explanation:
      "Modern RL often uses neural networks to approximate policies or value functions when states are high-dimensional or continuous. The three families differ in what they learn directly, but all can rely on function approximation.",
  },
  {
    id: "crash-probability-l4-q21",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly distinguish exploration from exploitation?",
    options: [
      {
        text: "Exploitation chooses an action currently believed to be best.",
        isCorrect: true,
      },
      {
        text: "Exploration tries actions partly to gain information that may improve future decisions.",
        isCorrect: true,
      },
      {
        text: "Exploration means discarding reward information during learning.",
        isCorrect: false,
      },
      {
        text: "Exploitation is optimal behavior even when current value estimates are wrong.",
        isCorrect: false,
      },
    ],
    explanation:
      "Exploitation uses current estimates, while exploration deliberately samples uncertain alternatives to learn. Pure exploitation can lock in early mistakes when value estimates are incomplete or the environment changes.",
  },
  {
    id: "crash-probability-l4-q22",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "In epsilon-greedy action selection with \\(\\epsilon=0.1\\), which statement is correct?",
    options: [
      {
        text: "The agent exploits with probability \\(0.9\\) and explores with probability \\(0.1\\).",
        isCorrect: true,
      },
      {
        text: "The agent chooses the second-best action with probability \\(0.1\\).",
        isCorrect: false,
      },
      {
        text: "\\(\\epsilon\\) is the discount factor for future rewards.",
        isCorrect: false,
      },
      {
        text: "\\(\\epsilon=0.1\\) means each action has probability exactly \\(0.1\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Epsilon-greedy exploration uses \\(\\epsilon\\) as the probability of an exploratory random choice and \\(1-\\epsilon\\) as the probability of exploiting the current best-known action. It is separate from discounting and does not by itself assign every action the same probability.",
  },
  {
    id: "crash-probability-l4-q23",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly connect entropy to exploration in reinforcement learning?",
    options: [
      {
        text: "A higher-entropy policy spreads probability over more actions and can encourage exploration.",
        isCorrect: true,
      },
      {
        text: "A low-entropy policy can be nearly deterministic when most mass is on one action.",
        isCorrect: true,
      },
      {
        text: "Entropy regularization can discourage a policy from becoming too deterministic too early.",
        isCorrect: true,
      },
      {
        text: "High entropy is identical to high expected return in a stochastic environment.",
        isCorrect: false,
      },
    ],
    explanation:
      "Policy entropy measures randomness in action selection, not reward quality itself. Randomness can help learning by maintaining exploration, but useful policies still need to discover actions that lead to high return.",
  },
  {
    id: "crash-probability-l4-q24",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "From state \\(s\\), action \\(a\\) leads to next state \\(g\\) with probability \\(0.7\\) and \\(b\\) with probability \\(0.3\\). Rewards are \\(+2\\) for both outcomes, \\(V(g)=10\\), \\(V(b)=0\\), and \\(\\gamma=0.5\\). Which statements are correct about the one-step lookahead value?",
    options: [
      {
        text: "The expected next-state value is \\(0.7\\cdot10+0.3\\cdot0=7\\).",
        isCorrect: true,
      },
      {
        text: "The discounted expected future value contribution is \\(0.5\\cdot7=3.5\\).",
        isCorrect: true,
      },
      {
        text: "The expected reward contribution is \\(2\\).",
        isCorrect: true,
      },
      {
        text: "The resulting lookahead value is \\(2+3.5=5.5\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The one-step lookahead combines expected immediate reward with discounted expected value of the next state. The transition probabilities weight possible futures before discounting, so this calculation gives \\(2+0.5(7)=5.5\\).",
  },
  {
    id: "crash-probability-l4-q25",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "A policy chooses `up` in state \\(s_0\\) with probability \\(0.6\\). The environment then reaches \\(s_1\\) with probability \\(0.8\\). Which statements about the two-step event are correct, assuming the policy choice and transition factor in the usual conditional way?",
    options: [
      {
        text: "The probability of choosing `up` and then reaching \\(s_1\\) is \\(0.6\\cdot0.8=0.48\\).",
        isCorrect: true,
      },
      {
        text: "The calculation combines the agent's action probability \\(\\pi(a\\mid s)\\) with the environment transition probability \\(P(s'\\mid s,a)\\).",
        isCorrect: true,
      },
      {
        text: "The probability is \\(0.8\\) because the environment transition replaces the policy choice.",
        isCorrect: false,
      },
      {
        text: "The probability must be \\(0.6+0.8=1.4\\) because trajectory probabilities add across time.",
        isCorrect: false,
      },
    ],
    explanation:
      "A trajectory probability combines the probability that the policy selects an action with the probability that the environment produces the next state after that action. These conditional factors multiply for a particular path segment rather than being added.",
  },
  {
    id: "crash-probability-l4-q26",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which notation statement is most accurate for \\(R_{t+1}\\) in the RL loop?",
    options: [
      {
        text: "\\(R_{t+1}\\) is the reward observed after taking action \\(A_t\\) from state \\(S_t\\).",
        isCorrect: true,
      },
      {
        text: "\\(R_{t+1}\\) is the reward predicted before \\(S_t\\) exists.",
        isCorrect: false,
      },
      {
        text: "\\(R_{t+1}\\) is another name for the action chosen at time \\(t+1\\).",
        isCorrect: false,
      },
      {
        text: "\\(R_{t+1}\\) is the same quantity as the return \\(G_t\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The index \\(t+1\\) marks the reward that follows the action at time \\(t\\). Return \\(G_t\\) is usually a cumulative discounted sum of future rewards, so it is not generally the same as one immediate reward.",
  },
  {
    id: "crash-probability-l4-q27",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "For a fixed state-action pair \\((s,a)\\), which rows could be valid transition distributions over three next states?",
    options: [
      {
        text: "\\((0.8,0.1,0.1)\\).",
        isCorrect: true,
      },
      {
        text: "\\((1,0,0)\\).",
        isCorrect: true,
      },
      {
        text: "\\((0.2,0.3,0.5)\\).",
        isCorrect: true,
      },
      {
        text: "\\((0.6,0.6,-0.2)\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "A valid discrete transition distribution must have nonnegative probabilities that sum to one. Deterministic transitions are allowed as a special case, but negative probabilities are invalid even if the entries happen to sum to one.",
  },
  {
    id: "crash-probability-l4-q28",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which state representations are more likely to support a Markov model because they include relevant history or hidden information?",
    options: [
      {
        text: "A robot vacuum state that includes current position, map, battery, and known obstacles.",
        isCorrect: true,
      },
      {
        text: "A medical decision state that includes current measurements, relevant history, medications, and prior diagnoses.",
        isCorrect: true,
      },
      {
        text: "A chatbot state that includes full conversation history and relevant memory.",
        isCorrect: true,
      },
      {
        text: "A chess state that includes enough board information and rule-relevant history to determine legal future moves.",
        isCorrect: true,
      },
    ],
    explanation:
      "The Markov property depends on whether the state summarizes the information needed for future transitions and rewards. Richer state representations are more likely to satisfy that condition than a narrow observation that drops relevant history.",
  },
  {
    id: "crash-probability-l4-q29",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements correctly distinguish deterministic and stochastic environments?",
    options: [
      {
        text: "In a deterministic environment, a given state-action pair has one next state with probability one.",
        isCorrect: true,
      },
      {
        text: "In a stochastic environment, the same state-action pair can lead to different next states with different probabilities.",
        isCorrect: true,
      },
      {
        text: "A stochastic environment is represented with action labels rather than conditional probabilities.",
        isCorrect: false,
      },
      {
        text: "A deterministic environment requires the policy to be deterministic.",
        isCorrect: false,
      },
    ],
    explanation:
      "Determinism and stochasticity describe the environment's transition response after a state-action pair. They are separate from whether the agent's policy is deterministic or stochastic, and both cases can be expressed using conditional probabilities.",
  },
  {
    id: "crash-probability-l4-q30",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Action A gives rewards \\((2,4,8)\\) with certainty over the next three steps. Action B gives reward \\(0\\) now and then \\(12\\) two steps later. With \\(\\gamma=0.5\\), which statements are correct?",
    options: [
      {
        text: "The return for Action A is \\(2+0.5\\cdot4+0.5^2\\cdot8=6\\).",
        isCorrect: true,
      },
      {
        text: "The return for Action B is \\(0+0.5\\cdot0+0.5^2\\cdot12=3\\), if the \\(12\\) reward arrives as \\(R_{t+3}\\).",
        isCorrect: true,
      },
      {
        text: "With these numbers and discounting, Action A has larger discounted return.",
        isCorrect: true,
      },
      {
        text: "Action B must be better because its largest undiscounted reward is bigger.",
        isCorrect: false,
      },
    ],
    explanation:
      "Discounting means timing matters, so a larger delayed reward can be worth less than smaller earlier rewards. The calculation compares discounted sums, not just the largest reward that appears anywhere in the future.",
  },
  {
    id: "crash-probability-l4-q31",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe delayed reward and credit assignment?",
    options: [
      {
        text: "In chess, the main reward may arrive only at the end as win, draw, or loss.",
        isCorrect: true,
      },
      {
        text: "Earlier actions can contribute to later outcomes even when they receive no immediate reward.",
        isCorrect: true,
      },
      {
        text: "Credit assignment asks which earlier actions helped or hurt later return.",
        isCorrect: true,
      },
      {
        text: "Delayed reward makes RL harder than a one-step label-loss problem.",
        isCorrect: true,
      },
    ],
    explanation:
      "RL often requires learning from outcomes that occur many steps after the causal action. Credit assignment is difficult because the agent must infer which choices in a trajectory contributed to the final return.",
  },
  {
    id: "crash-probability-l4-q32",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "A state has \\(Q(s,\\text{up})=2\\), \\(Q(s,\\text{right})=7\\), \\(Q(s,\\text{left})=-1\\), and \\(Q(s,\\text{down})=4\\). Which action does a greedy value-based policy choose?",
    options: [
      {
        text: "`right`, because it has the largest estimated action value.",
        isCorrect: true,
      },
      {
        text: "`left`, because negative values encourage exploration.",
        isCorrect: false,
      },
      {
        text: "`down`, because \\(4\\) is closest to the average value.",
        isCorrect: false,
      },
      {
        text: "`up`, because the first listed action is the default greedy action.",
        isCorrect: false,
      },
    ],
    explanation:
      "A greedy action is selected by maximizing the estimated action value over available actions. Since \\(7\\) is the largest \\(Q\\)-value, the greedy value-based choice is `right`.",
  },
  {
    id: "crash-probability-l4-q33",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "A policy network outputs action logits \\((2,1,0,-1)\\) for up, right, left, and down. Which statements are correct?",
    options: [
      {
        text: "After softmax, `up` has the largest action probability.",
        isCorrect: true,
      },
      {
        text: "`down` can still have nonzero probability under softmax.",
        isCorrect: true,
      },
      {
        text: "The logits themselves are already probabilities because they are ordered.",
        isCorrect: false,
      },
      {
        text: "A stochastic policy samples from actions whose logits are positive after normalization.",
        isCorrect: false,
      },
    ],
    explanation:
      "Softmax preserves the ranking of logits while assigning positive probability to every finite logit. Ordering alone does not make raw scores into probabilities, and negative logits can still correspond to possible sampled actions.",
  },
  {
    id: "crash-probability-l4-q34",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which notation statements correctly distinguish \\(V^\\pi\\), \\(Q^\\pi\\), and the RL objective?",
    options: [
      {
        text: "The superscript \\(\\pi\\) means the value is evaluated under a policy.",
        isCorrect: true,
      },
      {
        text: "\\(Q^\\pi(s,a)\\) conditions on both the current state and the first action.",
        isCorrect: true,
      },
      {
        text: "\\(\\max_\\pi \\mathbb{E}_\\pi[G_t]\\) represents choosing a policy to maximize expected return.",
        isCorrect: true,
      },
      {
        text: "\\(V^\\pi(s)\\) and \\(Q^\\pi(s,a)\\) are immediate rewards, not expectations over future returns.",
        isCorrect: false,
      },
    ],
    explanation:
      "The policy superscript matters because values depend on how the agent behaves in the future. \\(V\\) conditions on the state, \\(Q\\) conditions on a state-action pair, and the objective searches for a policy with high expected return.",
  },
  {
    id: "crash-probability-l4-q35",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which conditional probability structures are central to reinforcement learning?",
    options: [
      {
        text: "\\(P(s'\\mid s,a)\\), the environment transition distribution.",
        isCorrect: true,
      },
      {
        text: "\\(\\pi(a\\mid s)\\), the policy distribution over actions.",
        isCorrect: true,
      },
      {
        text: "\\(P(r,s'\\mid s,a)\\), a possible model of stochastic reward and next state.",
        isCorrect: true,
      },
      {
        text: "\\(\\mathbb{E}_\\pi[G_t]\\), the expected return under a policy.",
        isCorrect: true,
      },
    ],
    explanation:
      "RL uses probability both for the environment's uncertain response and for the agent's action selection. Expected return then aggregates possible futures under a policy, making expectation the central objective-level probability operation.",
  },
  {
    id: "crash-probability-l4-q36",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements correctly compare \\(\\min -\\log P(y\\mid x)\\) with \\(\\max_\\pi \\mathbb{E}_\\pi[G_t]\\)?",
    options: [
      {
        text: "The supervised objective scores probability assigned to observed labels or tokens.",
        isCorrect: true,
      },
      {
        text: "The RL objective scores policies by expected cumulative future reward.",
        isCorrect: true,
      },
      {
        text: "In RL, the agent's actions can change the future data distribution it experiences.",
        isCorrect: true,
      },
      {
        text: "Both objectives can involve probability, but they optimize different quantities.",
        isCorrect: true,
      },
    ],
    explanation:
      "Lecture 3's likelihood objective is about assigning probability to observed outputs in a prediction problem. Reinforcement learning shifts the target to policies and expected future reward, where actions influence future states, rewards, and observations.",
  },
  {
    id: "crash-probability-l4-q37",
    chapter: 4,
    difficulty: "medium",
    type: "assertion-reason",
    prompt:
      "Assertion: A stochastic policy can choose an action that is not currently the most probable action.\n\nReason: A stochastic policy assigns probability zero to every action except the modal action.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: true },
      { text: "Assertion is false, Reason is true.", isCorrect: false },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: false,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "The assertion is true because a stochastic policy can sample from multiple nonzero-probability actions, including actions that are not modal. The reason is false because assigning probability zero to every non-modal action would describe a deterministic modal choice rather than a genuinely stochastic policy.",
  },
  {
    id: "crash-probability-l4-q38",
    chapter: 4,
    difficulty: "medium",
    type: "assertion-reason",
    prompt:
      "Assertion: The Markov property says that the full past history always improves the next-state prediction beyond the current state and action.\n\nReason: A state representation may need to include history or memory for the Markov property to be plausible.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: false },
      { text: "Assertion is false, Reason is true.", isCorrect: true },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: false,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "The assertion is false because the Markov property says the current state and action are sufficient for the next-state distribution, so earlier history adds no extra information once the state is known. The reason is true because the state may need to be designed richly enough to summarize relevant past information.",
  },
  {
    id: "crash-probability-l4-q39",
    chapter: 4,
    difficulty: "hard",
    type: "assertion-reason",
    prompt:
      "Assertion: \\(Q^\\pi(s,a)\\) can prefer an action with zero immediate reward over an action with positive immediate reward.\n\nReason: \\(Q^\\pi(s,a)\\) estimates expected future return after taking \\(a\\) in \\(s\\), not only the immediate reward.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: false },
      { text: "Assertion is false, Reason is true.", isCorrect: false },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: true,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "Both statements are true, and the reason explains the assertion. A zero-reward action can lead to valuable future states, while a positive immediate reward can lead to poor future consequences.",
  },
  {
    id: "crash-probability-l4-q40",
    chapter: 4,
    difficulty: "medium",
    type: "assertion-reason",
    prompt:
      "Assertion: Higher policy entropy can make exploratory action selection more likely.\n\nReason: The discount factor \\(\\gamma\\) changes how future rewards are weighted in return.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: false },
      { text: "Assertion is false, Reason is true.", isCorrect: false },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: false,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: true,
      },
    ],
    explanation:
      "Both statements are true, but the reason does not explain the assertion. Entropy concerns randomness in action selection, while \\(\\gamma\\) belongs to return calculation and controls how much future rewards matter.",
  },
  {
    id: "crash-probability-l4-q41",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "A trajectory segment has \\(\\pi(a_0\\mid s_0)=0.4\\), \\(P(s_1\\mid s_0,a_0)=0.5\\), \\(\\pi(a_1\\mid s_1)=0.25\\), and \\(P(s_2\\mid s_1,a_1)=0.8\\). Which statement gives the probability of this state-action-state-action-state segment under the usual policy/environment factorization?",
    options: [
      {
        text: "\\(0.4\\cdot0.5\\cdot0.25\\cdot0.8=0.04\\).",
        isCorrect: true,
      },
      {
        text: "\\(0.4\\cdot0.25=0.10\\), because the environment factors are not part of a policy trajectory.",
        isCorrect: false,
      },
      {
        text: "\\(0.5\\cdot0.8=0.40\\), because the environment transitions determine the full trajectory probability.",
        isCorrect: false,
      },
      {
        text: "\\(0.4+0.5+0.25+0.8=1.95\\), because sequential probabilities are accumulated additively.",
        isCorrect: false,
      },
    ],
    explanation:
      "For a particular path, the probability factors into the policy probabilities for chosen actions and the environment probabilities for resulting next states. Adding the factors or dropping either the policy or transition terms gives a different quantity, not the probability of the specified segment.",
  },
  {
    id: "crash-probability-l4-q42",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "An epsilon-greedy agent has four actions and one greedy action. With \\(\\epsilon=0.2\\), suppose the exploratory branch samples uniformly from all four actions. Which statements are correct?",
    options: [
      {
        text: "The greedy action is selected with probability \\(0.8+0.2/4=0.85\\).",
        isCorrect: true,
      },
      {
        text: "Each non-greedy action is selected with probability \\(0.2/4=0.05\\).",
        isCorrect: true,
      },
      {
        text: "Each non-greedy action is selected with probability \\(0.2/3\\) under the stated convention.",
        isCorrect: false,
      },
      {
        text: "The action distribution has zero entropy because the greedy action remains most probable.",
        isCorrect: false,
      },
    ],
    explanation:
      "The stated convention samples from all four actions during exploration, so the greedy action receives both the exploitation mass and its share of exploratory mass. The policy is still stochastic because non-greedy actions have positive probability.",
  },
  {
    id: "crash-probability-l4-q43",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "For a one-step lookahead, action \\(a\\) from state \\(s\\) has two outcomes: probability \\(0.7\\) gives reward \\(2\\) and next-state value \\(4\\); probability \\(0.3\\) gives reward \\(-1\\) and next-state value \\(10\\). With \\(\\gamma=0.5\\), which statements are correct?",
    options: [
      {
        text: "The expected immediate reward is \\(0.7\\cdot2+0.3\\cdot(-1)=1.1\\).",
        isCorrect: true,
      },
      {
        text: "The expected next-state value before discounting is \\(0.7\\cdot4+0.3\\cdot10=5.8\\).",
        isCorrect: true,
      },
      {
        text: "The one-step lookahead value is \\(0.7(2+0.5\\cdot4)+0.3(-1+0.5\\cdot10)=4\\).",
        isCorrect: true,
      },
      {
        text: "The discounted next-state contribution is \\(0.5\\cdot4=2\\) because the first outcome has higher probability.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lookahead expectation averages both outcomes after adding immediate reward and discounted next-state value inside each outcome. Using a single outcome's value discards part of the transition distribution and gives the wrong expectation.",
  },
  {
    id: "crash-probability-l4-q44",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements describe valid mathematical components of a finite Markov Decision Process?",
    options: [
      {
        text: "For each fixed \\((s,a)\\), \\(P(\\cdot\\mid s,a)\\) is a probability distribution over next states.",
        isCorrect: true,
      },
      {
        text: "A deterministic transition is represented by assigning probability \\(1\\) to one next state and \\(0\\) to the others.",
        isCorrect: true,
      },
      {
        text: "A reward function may be written as \\(R(s,a)\\) or \\(R(s,a,s')\\), depending on the modeling choice.",
        isCorrect: true,
      },
      {
        text: "A discount factor \\(\\gamma\\) in \\([0,1]\\) specifies how later rewards are weighted in return.",
        isCorrect: true,
      },
    ],
    explanation:
      "These are standard pieces of an MDP formalization. The transition model is probabilistic, deterministic dynamics are a special case, rewards can be represented with different arguments, and discounting controls temporal weighting.",
  },
  {
    id: "crash-probability-l4-q45",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statement is the cleanest mathematical expression of the Markov property for next states?",
    options: [
      {
        text: "\\(P(S_{t+1}\\mid S_t,A_t,S_{t-1},A_{t-1},\\ldots,S_0,A_0)=P(S_{t+1}\\mid S_t,A_t)\\).",
        isCorrect: true,
      },
      {
        text: "\\(P(S_{t+1}\\mid S_t,A_t)=P(S_{t+1})\\), so the next state is unrelated to the current state-action pair.",
        isCorrect: false,
      },
      {
        text: "\\(P(A_t\\mid S_t)=P(S_t\\mid A_t)\\), so actions and states are symmetric events.",
        isCorrect: false,
      },
      {
        text: "\\(P(R_{t+1}\\mid S_t,A_t)=P(A_t\\mid S_t)\\), so rewards and policies are the same distribution.",
        isCorrect: false,
      },
    ],
    explanation:
      "The Markov property states that the current state and action are sufficient for the next-state distribution, relative to the full earlier history. It does not make the next state independent of the current state-action pair and does not identify policy probabilities with reward or transition probabilities.",
  },
  {
    id: "crash-probability-l4-q46",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "An agent receives rewards \\((0,0,10)\\) over the next three reward times. Which discounted-return statements are correct?",
    options: [
      {
        text: "With \\(\\gamma=0.9\\), the three-step discounted return is \\(0+0.9\\cdot0+0.9^2\\cdot10=8.1\\).",
        isCorrect: true,
      },
      {
        text: "With \\(\\gamma=0.5\\), the three-step discounted return is \\(0+0.5\\cdot0+0.5^2\\cdot10=2.5\\).",
        isCorrect: true,
      },
      {
        text: "The return is larger for \\(\\gamma=0.5\\) than for \\(\\gamma=0.9\\) because the reward is delayed.",
        isCorrect: false,
      },
      {
        text: "The discounted return is \\(10\\gamma\\) because the nonzero reward is the third listed reward.",
        isCorrect: false,
      },
    ],
    explanation:
      "The third listed reward is \\(R_{t+3}\\), so it is weighted by \\(\\gamma^2\\). Larger \\(\\gamma\\) preserves more of a delayed positive reward, which is why the value is \\(8.1\\) for \\(0.9\\) and \\(2.5\\) for \\(0.5\\).",
  },
  {
    id: "crash-probability-l4-q47",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "For a stochastic policy \\(\\pi\\), which statements correctly relate \\(V^\\pi(s)\\) and \\(Q^\\pi(s,a)\\) in a finite action set?",
    options: [
      {
        text: "\\(V^\\pi(s)=\\sum_a \\pi(a\\mid s)Q^\\pi(s,a)\\).",
        isCorrect: true,
      },
      {
        text: "If \\(\\pi\\) chooses action \\(a^*\\) with probability \\(1\\) in state \\(s\\), then \\(V^\\pi(s)=Q^\\pi(s,a^*)\\).",
        isCorrect: true,
      },
      {
        text: "\\(V^\\pi(s)\\) averages action values using the policy's action probabilities.",
        isCorrect: true,
      },
      {
        text: "\\(V^\\pi(s)=\\max_a Q^\\pi(s,a)\\) for every stochastic policy \\(\\pi\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The state value under a policy is the policy-weighted expectation of action values. It equals a maximum over actions only for a greedy deterministic choice, not for an arbitrary stochastic policy.",
  },
  {
    id: "crash-probability-l4-q48",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "For a finite-horizon trajectory \\(\\tau=(s_0,a_0,s_1,a_1,\\ldots,s_T)\\), which statements correctly connect trajectory probabilities and expected return?",
    options: [
      {
        text: "A trajectory probability can factor into an initial-state term, policy terms \\(\\pi(a_t\\mid s_t)\\), and transition terms \\(P(s_{t+1}\\mid s_t,a_t)\\).",
        isCorrect: true,
      },
      {
        text: "Expected return can be written as a sum over trajectories, \\(\\sum_\\tau P_\\pi(\\tau)G(\\tau)\\), in a finite trajectory space.",
        isCorrect: true,
      },
      {
        text: "Changing the policy changes the probabilities of trajectories by changing the action-selection factors.",
        isCorrect: true,
      },
      {
        text: "Rewards can be included in the trajectory description or treated as functions of state-action-next-state outcomes.",
        isCorrect: true,
      },
    ],
    explanation:
      "The RL objective is an expectation over possible futures generated jointly by the policy and environment. Writing the objective over trajectories makes clear how action probabilities, transition probabilities, rewards, and returns combine.",
  },
  {
    id: "crash-probability-l4-q49",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Using natural logs, a policy over three actions is \\((0.5,0.25,0.25)\\). Which statement about its entropy is correct?",
    options: [
      {
        text: "Its entropy is \\(-0.5\\log0.5-0.25\\log0.25-0.25\\log0.25\\approx1.04\\), below the uniform three-action entropy \\(\\log3\\).",
        isCorrect: true,
      },
      {
        text: "Its entropy is \\(0\\) because one action has the largest probability.",
        isCorrect: false,
      },
      {
        text: "Its entropy is greater than \\(\\log3\\) because three actions have positive probability.",
        isCorrect: false,
      },
      {
        text: "Its entropy equals \\(0.5+0.25+0.25=1\\), because probabilities sum to one.",
        isCorrect: false,
      },
    ],
    explanation:
      "Entropy sums \\(-p_i\\log p_i\\), not the raw probabilities. The distribution is uncertain but not maximally uncertain, so its entropy is positive and below the entropy of the uniform distribution over three actions.",
  },
  {
    id: "crash-probability-l4-q50",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "A softmax action-selection rule sets \\(\\pi(a\\mid s)\\propto e^{\\beta Q(s,a)}\\), where \\(\\beta\\ge0\\). Which statements are correct?",
    options: [
      {
        text: "If \\(\\beta=0\\), all finite actions receive equal probability.",
        isCorrect: true,
      },
      {
        text: "For two actions \\(a,b\\), the odds ratio satisfies \\(\\pi(a\\mid s)/\\pi(b\\mid s)=e^{\\beta(Q(s,a)-Q(s,b))}\\).",
        isCorrect: true,
      },
      {
        text: "An action with negative \\(Q(s,a)\\) receives probability zero after exponentiation.",
        isCorrect: false,
      },
      {
        text: "Increasing \\(\\beta\\) makes lower-valued actions more probable relative to higher-valued actions.",
        isCorrect: false,
      },
    ],
    explanation:
      "This rule is the same softmax idea applied to action values rather than raw policy logits. The inverse-temperature parameter \\(\\beta\\) controls sharpness, finite exponentials stay positive, and odds depend on value differences.",
  },
  {
    id: "crash-probability-l4-q51",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "In state \\(s\\), a policy chooses action A with probability \\(0.25\\) and action B with probability \\(0.75\\). The expected return after A is \\(6\\), and after B is \\(4\\). Which statements are correct?",
    options: [
      {
        text: "The state value under this policy is \\(0.25\\cdot6+0.75\\cdot4=4.5\\).",
        isCorrect: true,
      },
      {
        text: "This calculation is an instance of \\(V^\\pi(s)=\\sum_a\\pi(a\\mid s)Q^\\pi(s,a)\\).",
        isCorrect: true,
      },
      {
        text: "A deterministic policy choosing A in this state would have value \\(6\\), assuming the same continuation values.",
        isCorrect: true,
      },
      {
        text: "The value under the stochastic policy is \\(6\\) because A has the larger action value.",
        isCorrect: false,
      },
    ],
    explanation:
      "A stochastic policy's value is a probability-weighted average of the action values it induces. The highest action value is not the state value unless the policy puts all its probability on that action.",
  },
  {
    id: "crash-probability-l4-q52",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements correctly extend the discrete RL notation toward continuous state or action spaces?",
    options: [
      {
        text: "Sums over next states may become integrals over next-state densities.",
        isCorrect: true,
      },
      {
        text: "A continuous-action policy can be described by a density over actions given the state.",
        isCorrect: true,
      },
      {
        text: "Expected return is still an expectation over possible trajectories.",
        isCorrect: true,
      },
      {
        text: "Neural networks can approximate policies or value functions when tabular storage is impractical.",
        isCorrect: true,
      },
    ],
    explanation:
      "The core probability ideas survive the move from finite tables to continuous spaces, but sums often become integrals and probabilities may be represented with densities. Function approximation becomes important because enumerating every state or action is no longer practical.",
  },
  {
    id: "crash-probability-l4-q53",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "In a partially observable maze, the agent sees the same camera image in two different locations that require different good actions. Which statement is correct?",
    options: [
      {
        text: "A belief state such as \\(P(S_t\\mid\\text{observation/action history})\\) can summarize uncertainty about the hidden true state.",
        isCorrect: true,
      },
      {
        text: "The latest camera image is a Markov state whenever the policy network is sufficiently large.",
        isCorrect: false,
      },
      {
        text: "Partial observability disappears when rewards are discounted.",
        isCorrect: false,
      },
      {
        text: "The transition model becomes deterministic once two locations share the same observation.",
        isCorrect: false,
      },
    ],
    explanation:
      "When observations do not reveal the full state, the agent may need memory or a belief distribution over hidden states. Bigger function approximators and discounting do not by themselves restore the missing information in the current observation.",
  },
  {
    id: "crash-probability-l4-q54",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "For the Bellman-style action-value target \\(R_{t+1}+\\gamma\\max_{a'}Q(S_{t+1},a')\\), which statements are correct?",
    options: [
      {
        text: "The target combines the immediate reward with a discounted estimate of future action value.",
        isCorrect: true,
      },
      {
        text: "The \\(\\max_{a'}\\) term corresponds to a greedy choice at the next state in this target.",
        isCorrect: true,
      },
      {
        text: "The target is the same as \\(R_{t+1}\\) whenever \\(\\gamma>0\\).",
        isCorrect: false,
      },
      {
        text: "The target averages next-action values using \\(\\pi(a'\\mid S_{t+1})\\) rather than taking a maximum.",
        isCorrect: false,
      },
    ],
    explanation:
      "This target expresses the idea that the value of an action includes reward now plus discounted value later. The maximum makes it a greedy next-state target; a policy-weighted expectation would be a different Bellman expression.",
  },
  {
    id: "crash-probability-l4-q55",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "A simplified RLHF objective for prompts \\(x\\) and responses \\(y\\) can be written \\(\\max_\\theta \\mathbb{E}_{y\\sim\\pi_\\theta(\\cdot\\mid x)}[R(x,y)]\\). Which statements are correct?",
    options: [
      {
        text: "\\(\\pi_\\theta(y\\mid x)\\) is the model's conditional distribution over responses for a prompt.",
        isCorrect: true,
      },
      {
        text: "The expectation averages reward over responses sampled from the model distribution.",
        isCorrect: true,
      },
      {
        text: "Changing \\(\\theta\\) can change which responses receive more probability.",
        isCorrect: true,
      },
      {
        text: "The reward model score is identical to a normalized probability distribution over responses.",
        isCorrect: false,
      },
    ],
    explanation:
      "The objective uses the language model as a policy and evaluates sampled responses with a reward signal. Reward scores guide optimization but are not themselves the response probability distribution.",
  },
  {
    id: "crash-probability-l4-q56",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "For a finite MDP, which expressions are correct pieces of the one-step Bellman expectation equation for a policy \\(\\pi\\)?",
    options: [
      {
        text: "\\(V^\\pi(s)=\\sum_a\\pi(a\\mid s)Q^\\pi(s,a)\\).",
        isCorrect: true,
      },
      {
        text: "\\(Q^\\pi(s,a)=\\sum_{r,s'}P(r,s'\\mid s,a)\\bigl[r+\\gamma V^\\pi(s')\\bigr]\\).",
        isCorrect: true,
      },
      {
        text: "The transition/reward distribution weights the possible next outcomes in the expectation.",
        isCorrect: true,
      },
      {
        text: "The policy distribution weights possible first actions when computing \\(V^\\pi(s)\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The Bellman expectation equations combine policy probabilities, environment probabilities, rewards, discounting, and next-state values. They make explicit how value functions are expectations over both the agent's actions and the environment's next outcomes.",
  },
  {
    id: "crash-probability-l4-q57",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "A state has estimated values \\(Q(\\text{left})=10\\), \\(Q(\\text{right})=6\\), and \\(Q(\\text{wait})=2\\). An epsilon-greedy policy uses \\(\\epsilon=0.1\\), with the random branch uniform over all three actions. Which statement gives the expected selected \\(Q\\)-value for one action choice?",
    options: [
      {
        text: "\\(0.9\\cdot10+0.1\\cdot\\frac{10+6+2}{3}=9.6\\).",
        isCorrect: true,
      },
      {
        text: "\\(0.9\\cdot10+0.1\\cdot\\frac{6+2}{2}=9.4\\), because the random branch excludes the greedy action under the stated convention.",
        isCorrect: false,
      },
      {
        text: "\\(\\frac{10+6+2}{3}=6\\), because epsilon-greedy samples uniformly.",
        isCorrect: false,
      },
      {
        text: "\\(10\\), because epsilon-greedy has the same expected value as pure exploitation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Under the stated convention, exploitation chooses the greedy action with probability \\(0.9\\), while exploration samples uniformly from all actions with probability \\(0.1\\). The expected selected estimate combines those two branches.",
  },
  {
    id: "crash-probability-l4-q58",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "For a reward function depending on next state, \\(R(s,a,s')\\), which statements correctly compute a one-step expected return from \\((s,a)\\)?",
    options: [
      {
        text: "\\(\\sum_{s'}P(s'\\mid s,a)\\bigl[R(s,a,s')+\\gamma V(s')\\bigr]\\) is the standard discrete expectation form.",
        isCorrect: true,
      },
      {
        text: "If rewards are stochastic too, \\(\\sum_{r,s'}P(r,s'\\mid s,a)[r+\gamma V(s')]\\) keeps both sources of uncertainty.",
        isCorrect: true,
      },
      {
        text: "The calculation should multiply the largest reward by the largest next-state probability and ignore the other outcomes.",
        isCorrect: false,
      },
      {
        text: "The calculation should use an unweighted average over next states even when transition probabilities differ.",
        isCorrect: false,
      },
    ],
    explanation:
      "Expected one-step return is a weighted average over possible next outcomes. The weights come from the transition or joint reward-transition distribution, so replacing them with maxima or unweighted averages changes the objective.",
  },
  {
    id: "crash-probability-l4-q59",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which notation statements are correct in the reinforcement-learning formulas from this lecture?",
    options: [
      {
        text: "The subscript \\(t\\) indexes time steps in variables such as \\(S_t\\), \\(A_t\\), and \\(R_{t+1}\\).",
        isCorrect: true,
      },
      {
        text: "The prime in \\(s'\\) denotes a possible next state, not a derivative.",
        isCorrect: true,
      },
      {
        text: "The expectation subscript in \\(\\mathbb{E}_\\pi[G_t]\\) indicates that returns are distributed according to policy \\(\\pi\\) interacting with the environment.",
        isCorrect: true,
      },
      {
        text: "The same symbol \\(s\\) denotes both a reward value and an action value in \\(Q(s,a)\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "RL notation uses time subscripts, primed next-state variables, and policy-indexed expectations to keep the sequential structure clear. In \\(Q(s,a)\\), \\(s\\) is a state and \\(a\\) is an action, while rewards are represented separately.",
  },
  {
    id: "crash-probability-l4-q60",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Assume rewards satisfy \\(|R_{t+k}|\\le R_{\\max}\\). Which statements about discounted infinite-horizon returns are correct?",
    options: [
      {
        text: "If \\(0\\le\\gamma<1\\), then \\(|G_t|\\le R_{\\max}/(1-\gamma)\\).",
        isCorrect: true,
      },
      {
        text: "As \\(\\gamma\\) gets closer to \\(1\\), later rewards have more influence on the return.",
        isCorrect: true,
      },
      {
        text: "With \\(\\gamma=0\\), the discounted return reduces to \\(R_{t+1}\\).",
        isCorrect: true,
      },
      {
        text: "For finite episodes, using \\(\\gamma=1\\) can still give a finite undiscounted return when the episode terminates.",
        isCorrect: true,
      },
    ],
    explanation:
      "For bounded rewards and \\(\\gamma<1\\), the absolute discounted return is bounded by a geometric series. The special cases clarify the role of discounting: \\(\\gamma=0\\) keeps the immediate reward, large \\(\\gamma\\) values emphasize future rewards, and finite terminating episodes can have finite undiscounted sums.",
  },
];
