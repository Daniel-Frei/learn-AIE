import { Question } from "../../quiz";

export const OtherRL_introductiontoReinforcementLearning: Question[] = [
{
id: "other-rl-intro-q01",
chapter: 0,
difficulty: "medium",
prompt: "Which statements correctly describe reinforcement learning compared to traditional programming?",
options: [
{ text: "It is useful when explicitly programming every low-level action is infeasible due to complex dynamics.", isCorrect: true },
{ text: "It replaces hand-coded rules with learning driven by reward feedback.", isCorrect: true },
{ text: "It is inspired by trial-and-error learning seen in animals.", isCorrect: true },
{ text: "It requires complete knowledge of physics before training begins.", isCorrect: false }
],
explanation: "Reinforcement learning is applied where specifying exact control rules is impractical. Instead of requiring full knowledge of dynamics, it learns behavior through interaction and reward signals."
},

{
id: "other-rl-intro-q02",
chapter: 0,
difficulty: "medium",
prompt: "Which statements about the agent–environment framework are correct?",
options: [
{ text: "The agent represents the part of the system directly controlled by the learner.", isCorrect: true },
{ text: "The environment includes everything outside the learner’s direct control.", isCorrect: true },
{ text: "Actions influence the environment, and states are observations returned to the agent.", isCorrect: true },
{ text: "The boundary between agent and environment is fixed and objective.", isCorrect: false }
],
explanation: "Agent and environment definitions depend on what is considered controllable. The interaction loop consists of actions changing the environment and states being fed back to the agent."
},

{
id: "other-rl-intro-q03",
chapter: 0,
difficulty: "medium",
prompt: "Which of the following are valid examples of state representations?",
options: [
{ text: "Pixel values from a game screen.", isCorrect: true },
{ text: "Joint angles and velocities of a robot.", isCorrect: true },
{ text: "Battery level and temperature readings.", isCorrect: true },
{ text: "A symbolic label such as 'good move' chosen by a human.", isCorrect: false }
],
explanation: "States are numerical observations describing the environment. Human subjective labels are not intrinsic state variables unless encoded numerically."
},

{
id: "other-rl-intro-q04",
chapter: 0,
difficulty: "medium",
prompt: "Which statements about rewards in reinforcement learning are correct?",
options: [
{ text: "They provide scalar feedback after actions.", isCorrect: true },
{ text: "They guide the agent toward desirable behavior.", isCorrect: true },
{ text: "They are conventionally treated as signals from the environment.", isCorrect: true },
{ text: "They directly specify the optimal action at each step.", isCorrect: false }
],
explanation: "Rewards do not prescribe exact actions but provide evaluative feedback. The agent must learn how to act to maximize cumulative rewards."
},

{
id: "other-rl-intro-q05",
chapter: 0,
difficulty: "medium",
prompt: "Which expressions correctly represent discounted return?",
options: [
{ text: "\\(G_t = \\sum_{k=0}^{\\infty} \\gamma^k r_{t+k}\\)", isCorrect: true },
{ text: "Discount factor \\(\\gamma\\) controls how much future rewards matter.", isCorrect: true },
{ text: "Setting \\(\\gamma = 1\\) is safe only when episodes terminate.", isCorrect: true },
{ text: "Discounting ensures actions are always deterministic.", isCorrect: false }
],
explanation: "Discounting downweights distant rewards and prevents divergence in infinite horizons. Determinism is unrelated to discounting."
},

{
id: "other-rl-intro-q06",
chapter: 0,
difficulty: "medium",
prompt: "Which statements correctly describe the Markov property?",
options: [
{ text: "The next state depends only on the current state and action.", isCorrect: true },
{ text: "Missing velocity information can violate the Markov assumption.", isCorrect: true },
{ text: "Including sufficient variables restores Markov structure.", isCorrect: true },
{ text: "It guarantees fast learning regardless of rewards.", isCorrect: false }
],
explanation: "Markov states must contain all information needed to predict transitions. It is a structural assumption, not a guarantee of performance."
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
    { text: "Transition dynamics relating states and actions", isCorrect: true }
  ],
  explanation: "The MDP is defined by states, actions, rewards and transition dynamics. The policy is what the agent tries to learn, not part of the environment definition."
},


{
id: "other-rl-intro-q08",
chapter: 0,
difficulty: "medium",
prompt: "Which statements about policy functions \\(\\pi(a|s)\\) are correct?",
options: [
{ text: "They output probabilities over actions.", isCorrect: true },
{ text: "Stochasticity supports exploration.", isCorrect: true },
{ text: "Softmax is often used for discrete policies.", isCorrect: true },
{ text: "They directly encode environment dynamics.", isCorrect: false }
],
explanation: "Policies decide actions, not transitions. The environment dynamics are modeled separately."
},

{
id: "other-rl-intro-q09",
chapter: 0,
difficulty: "medium",
prompt: "Which are reasons discounted return is preferred over immediate reward?",
options: [
{ text: "It captures delayed consequences of actions.", isCorrect: true },
{ text: "Greedy immediate rewards can lead to poor long-term outcomes.", isCorrect: true },
{ text: "It allows optimizing over trajectories instead of single steps.", isCorrect: true },
{ text: "It removes the need for exploration.", isCorrect: false }
],
explanation: "Return encourages long-term planning. Exploration is still required to discover high-reward strategies."
},

{
id: "other-rl-intro-q10",
chapter: 0,
difficulty: "medium",
prompt: "Which statements correctly describe world models?",
options: [
{ text: "They estimate \\(P(s', r | s, a)\\).", isCorrect: true },
{ text: "They encode environment dynamics.", isCorrect: true },
{ text: "Having a world model improves sample efficiency.", isCorrect: true },
{ text: "Model-free RL assumes no access to this transition model.", isCorrect: true }
],
explanation: "World models predict transitions and rewards. Model-free methods instead learn purely from sampled experience."
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
{ text: "They are unnecessary in model-free RL.", isCorrect: false }
],
explanation: "Trajectories are fundamental to learning in model-free RL since they provide all observed experience."
},

{
id: "other-rl-intro-q12",
chapter: 0,
difficulty: "medium",
prompt: "Which issues characterize Monte Carlo methods?",
options: [
{ text: "They update values only after episode completion.", isCorrect: true },
{ text: "They struggle with long episodes.", isCorrect: true },
{ text: "They cannot distinguish which intermediate actions were responsible for success.", isCorrect: true },
{ text: "They are typically more sample efficient than temporal difference methods.", isCorrect: false }
],
explanation: "Monte Carlo waits for full returns, causing slow learning and poor credit assignment."
},

{
id: "other-rl-intro-q13",
chapter: 0,
difficulty: "medium",
prompt: "Which statements about temporal difference learning are correct?",
options: [
{ text: "It bootstraps from estimated future values.", isCorrect: true },
{ text: "It updates after every step.", isCorrect: true },
{ text: "It reduces dependence on episode termination.", isCorrect: true },
{ text: "It computes full trajectory returns explicitly.", isCorrect: false }
],
explanation: "Temporal difference uses one-step estimates rather than full rollouts, enabling faster updates."
},

{
  id: "other-rl-intro-q14",
  chapter: 0,
  difficulty: "medium",
  prompt: "According to the lecture, which of the following are described as off-policy methods?",
  options: [
    { text: "Q-learning", isCorrect: true },
    { text: "SARSA", isCorrect: false },
    { text: "Expected SARSA", isCorrect: false },
    { text: "Monte Carlo control (as used in the maze example)", isCorrect: false }
  ],
  explanation: "The lecture explicitly labels SARSA variants as on-policy and Q-learning as off-policy."
},


{
id: "other-rl-intro-q15",
chapter: 0,
difficulty: "medium",
prompt: "Which statements about epsilon-greedy exploration are correct?",
options: [
{ text: "With probability \\(\\epsilon\\), a random action is selected.", isCorrect: true },
{ text: "It prevents premature convergence to suboptimal policies.", isCorrect: true },
{ text: "It gradually shifts from exploration to exploitation.", isCorrect: true },
{ text: "It guarantees finding the optimal policy in finite steps.", isCorrect: false }
],
explanation: "Epsilon-greedy ensures continued exploration but does not guarantee optimality without sufficient sampling."
},

{
  id: "other-rl-intro-q16",
  chapter: 0,
  difficulty: "medium",
  prompt: "Which statements about sample efficiency from the lecture are correct?",
  options: [
    { text: "Temporal difference methods often outperform Monte Carlo in sample efficiency.", isCorrect: true },
    { text: "In the maze example, Q-learning was more sample efficient than SARSA.", isCorrect: true },
    { text: "Sample efficiency refers to how many interactions are needed to learn good behavior.", isCorrect: true },
    { text: "Monte Carlo was shown to be more efficient than TD methods.", isCorrect: false }
  ],
  explanation: "The lecture only makes empirical comparisons (maze) and explains sample efficiency conceptually."
},


{
id: "other-rl-intro-q17",
chapter: 0,
difficulty: "medium",
prompt: "Which are advantages of neural networks in reinforcement learning?",
options: [
{ text: "They approximate continuous state spaces.", isCorrect: true },
{ text: "They generalize across unseen states.", isCorrect: true },
{ text: "They replace tabular storage with parametric models.", isCorrect: true },
{ text: "They naturally handle infinite discrete actions in value-based methods.", isCorrect: false }
],
explanation: "Networks allow continuous representation but value-based approaches still require discrete actions."
},

{
id: "other-rl-intro-q18",
chapter: 0,
difficulty: "medium",
prompt: "Which statements about Deep Q Networks are correct?",
options: [
{ text: "They approximate the action-value function with neural networks.", isCorrect: true },
{ text: "They rely on Q-learning updates.", isCorrect: true },
{ text: "They operate over discrete actions.", isCorrect: true },
{ text: "They directly learn stochastic policies.", isCorrect: false }
],
explanation: "Deep Q Networks estimate Q-values; the policy is derived greedily rather than modeled probabilistically."
},

{
id: "other-rl-intro-q19",
chapter: 0,
difficulty: "medium",
prompt: "Which statements about policy gradients are correct?",
options: [
{ text: "They optimize an objective \\(J(\\theta)\\) instead of minimizing supervised loss.", isCorrect: true },
{ text: "They increase probability of actions weighted by advantage.", isCorrect: true },
{ text: "They can handle continuous actions via Gaussian policies.", isCorrect: true },
{ text: "They rely on tabular value iteration only.", isCorrect: false }
],
explanation: "Policy gradient methods perform gradient ascent on expected reward and extend naturally to continuous domains."
},

{
id: "other-rl-intro-q20",
chapter: 0,
difficulty: "medium",
prompt: "Which statements about actor–critic methods are correct?",
options: [
{ text: "The actor updates the policy.", isCorrect: true },
{ text: "The critic estimates value functions.", isCorrect: true },
{ text: "They typically use temporal difference for advantage estimation.", isCorrect: true },
{ text: "They remove the need for rewards.", isCorrect: false }
],
explanation: "Actor–critic combines policy optimization with value estimation; rewards remain essential."
},

{
id: "other-rl-intro-q21",
chapter: 0,
difficulty: "medium",
prompt: "Which statements about dopamine and temporal difference error are correct?",
options: [
{ text: "Dopamine spikes correspond to reward prediction errors.", isCorrect: true },
{ text: "Signals shift to earlier predictive cues as learning improves.", isCorrect: true },
{ text: "Negative prediction errors occur when expected rewards fail to appear.", isCorrect: true },
{ text: "Dopamine encodes only immediate reward magnitude.", isCorrect: false }
],
explanation: "Neuroscience evidence shows dopamine tracks surprise in expected reward, aligning closely with temporal difference error."
},

{
id: "other-rl-intro-q22",
chapter: 0,
difficulty: "medium",
prompt: "Which statements about imitation learning are correct?",
options: [
{ text: "Behavioral cloning learns policies directly from expert state–action pairs.", isCorrect: true },
{ text: "Dataset Aggregation improves robustness by adding corrective labels.", isCorrect: true },
{ text: "It avoids manually designing reward functions.", isCorrect: true },
{ text: "It guarantees optimal behavior outside demonstrated states.", isCorrect: false }
],
explanation: "Imitation learning mimics expert trajectories but struggles when encountering unseen states unless augmented."
},

{
id: "other-rl-intro-q23",
chapter: 0,
difficulty: "medium",
prompt: "Which statements about inverse reinforcement learning are correct?",
options: [
{ text: "It aims to infer the reward function underlying expert behavior.", isCorrect: true },
{ text: "It can generalize better by learning goals instead of direct actions.", isCorrect: true },
{ text: "It eliminates the need for expert demonstrations.", isCorrect: false },
{ text: "Inverse Q-learning is one method used.", isCorrect: true }
],
explanation: "Inverse reinforcement learning deduces objectives from demonstrations rather than copying actions."
},

{
id: "other-rl-intro-q24",
chapter: 0,
difficulty: "medium",
prompt: "Which are limitations of model-free reinforcement learning?",
options: [
{ text: "Requires large numbers of interactions.", isCorrect: true },
{ text: "Learns without understanding environment physics.", isCorrect: true },
{ text: "Often inefficient compared to model-based approaches.", isCorrect: true },
{ text: "Naturally generalizes from few samples.", isCorrect: false }
],
explanation: "Model-free RL relies purely on trial-and-error, leading to high sample complexity."
},

{
id: "other-rl-intro-q25",
chapter: 0,
difficulty: "medium",
prompt: "Which statements correctly describe model-based reinforcement learning?",
options: [
{ text: "Uses learned or provided transition models.", isCorrect: true },
{ text: "Allows simulated rollouts for planning.", isCorrect: true },
{ text: "Can dramatically reduce required real-world samples.", isCorrect: true },
{ text: "Cannot be combined with neural networks.", isCorrect: false }
],
explanation: "Model-based RL leverages predictive models to imagine futures and plan efficiently."
},

{
id: "other-rl-intro-q26",
chapter: 0,
difficulty: "medium",
prompt: "Which statements about RL instability are correct?",
options: [
{ text: "Different random seeds can produce drastically different results.", isCorrect: true },
{ text: "Policy gradient methods often show high variance across runs.", isCorrect: true },
{ text: "Training reproducibility is a known challenge.", isCorrect: true },
{ text: "RL training is fully deterministic given fixed hyperparameters.", isCorrect: false }
],
explanation: "RL is sensitive to randomness in initialization and sampling, leading to instability."
},

{
id: "other-rl-intro-q27",
chapter: 0,
difficulty: "medium",
prompt: "Which statements about reward hacking are correct?",
options: [
{ text: "Poorly designed rewards can lead to unintended behavior.", isCorrect: true },
{ text: "Agents exploit loopholes rather than achieving true goals.", isCorrect: true },
{ text: "Expert demonstrations can mitigate this issue.", isCorrect: true },
{ text: "Reward hacking occurs only in continuous control tasks.", isCorrect: false }
],
explanation: "Reward misalignment leads to gaming the metric rather than solving the task."
},

{
id: "other-rl-intro-q28",
chapter: 0,
difficulty: "medium",
prompt: "Which statements about Bellman’s principle of optimality are correct?",
options: [
{ text: "Optimal future decisions must remain optimal from the next state onward.", isCorrect: true },
{ text: "It underlies dynamic programming in RL.", isCorrect: true },
{ text: "It justifies the max operator in Q-learning targets.", isCorrect: true },
{ text: "It requires a deterministic environment.", isCorrect: false }
],
explanation: "Bellman optimality applies even in stochastic environments and is fundamental to value recursion."
},

{
id: "other-rl-intro-q29",
chapter: 0,
difficulty: "medium",
prompt: "Which statements about continuing tasks without terminal states are correct?",
options: [
{ text: "Monte Carlo methods struggle because returns cannot be finalized.", isCorrect: true },
{ text: "Temporal difference can still learn relative values.", isCorrect: true },
{ text: "Discounting prevents divergence.", isCorrect: true },
{ text: "Value estimates become meaningless.", isCorrect: false }
],
explanation: "Without terminal anchors values drift but relative ordering still guides decisions."
},

{
id: "other-rl-intro-q30",
chapter: 0,
difficulty: "medium",
prompt: "Which statements about Gaussian policies are correct?",
options: [
{ text: "Actions are sampled from \\(\\mathcal{N}(\\mu, \\sigma^2)\\).", isCorrect: true },
{ text: "Policy updates shift the mean toward advantageous actions.", isCorrect: true },
{ text: "Variance can expand or contract based on exploration needs.", isCorrect: true },
{ text: "They apply only to discrete action spaces.", isCorrect: false }
],
explanation: "Gaussian policies enable continuous control through parameterized distributions."
},

{
  id: "other-rl-intro-q31",
  chapter: 0,
  difficulty: "medium",
  prompt: "Which statements about the policy objective J(?) are correct?",
  options: [
    { text: "It measures expected performance over trajectories.", isCorrect: true },
    { text: "It is optimized using gradient ascent.", isCorrect: true },
    { text: "Different equivalent definitions exist.", isCorrect: true },
    { text: "It provides direct per-state learning targets like Q-values do.", isCorrect: false }
  ],
  explanation: "J evaluates the policy globally; Q and V provide state/action-level learning structure."
},


{
id: "other-rl-intro-q32",
chapter: 0,
difficulty: "medium",
prompt: "Which statements about credit assignment are correct?",
options: [
{ text: "It refers to identifying which actions led to outcomes.", isCorrect: true },
{ text: "Harder when rewards are delayed.", isCorrect: true },
{ text: "Temporal difference partially alleviates it.", isCorrect: true },
{ text: "It disappears in Monte Carlo methods.", isCorrect: false }
],
explanation: "Credit assignment remains challenging; Monte Carlo actually worsens it."
},

{
id: "other-rl-intro-q33",
chapter: 0,
difficulty: "medium",
prompt: "Which statements about RL vs human learning are correct?",
options: [
{ text: "Humans use internal world models.", isCorrect: true },
{ text: "Model-free RL resembles blind trial-and-error.", isCorrect: true },
{ text: "RL can still succeed without explicit understanding.", isCorrect: true },
{ text: "Humans learn exclusively via scalar rewards.", isCorrect: false }
],
explanation: "Humans leverage structure and reasoning beyond raw reward feedback."
}
];
