import { Question } from "../../../quiz";

type OptionSpec = readonly [text: string, isCorrect: boolean];

function makeQuestion(
  id: string,
  difficulty: Question["difficulty"],
  prompt: string,
  options: readonly [OptionSpec, OptionSpec, OptionSpec, OptionSpec],
  explanation: string,
): Question {
  return {
    id,
    chapter: 4,
    difficulty,
    prompt,
    options: options.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

export const CrashCourseProbabilityL4Questions: Question[] = [
  // States, actions, rewards, transitions, and the Markov property
  makeQuestion(
    "crash-probability-l4-q61",
    "easy",
    "In reinforcement learning, which quantity represents feedback received after an action?",
    [
      ["Reward", true],
      ["State", false],
      ["Policy", false],
      ["Transition probability", false],
    ],
    "Reward is the scalar feedback signal associated with a transition, while state describes the situation and policy describes action selection. Transition probabilities describe uncertainty in what state follows rather than the desirability signal itself.",
  ),
  makeQuestion(
    "crash-probability-l4-q62",
    "easy",
    "A robot in state s chooses action a and may reach several next locations. Which statements are correct?",
    [
      ["\\(P(s'\\mid s,a)\\) describes its next-state distribution.", true],
      [
        "Probabilities over exhaustive next states for fixed s and a sum to one.",
        true,
      ],
      ["A stochastic transition means the robot has no policy.", false],
      [
        "The reward is defined by \\(R(s,a,s')=P(s'\\mid s,a)=\\max_u P(u\\mid s,a)\\).",
        false,
      ],
    ],
    "The transition model conditions on the current state-action pair and normalizes across possible successors. Environment randomness is distinct from the agent's policy, and reward values need not be probabilities or match transition likelihoods.",
  ),
  makeQuestion(
    "crash-probability-l4-q63",
    "medium",
    "From state s, action a reaches \\(s_1\\) with probability 0.7 for reward 2 and \\(s_2\\) with probability 0.3 for reward -1. Which statements are correct?",
    [
      ["The transition probabilities form a valid distribution.", true],
      ["The expected immediate reward is \\(0.7(2)+0.3(-1)=1.1\\).", true],
      [
        "A sampled transition produces one of the two rewards, not the expectation 1.1 itself.",
        true,
      ],
      [
        "The expected immediate reward is \\(2-1=1\\) because probabilities only choose the state.",
        false,
      ],
    ],
    "Expectation weights each transition's reward by its probability, giving 1.1 as an average across repetitions. One interaction realizes a particular transition, while subtracting rewards without their path probabilities misstates the random process.",
  ),
  makeQuestion(
    "crash-probability-l4-q64",
    "hard",
    "A navigation state records only the robot's position, but battery level affects both feasible actions and future motion. Which statements are correct?",
    [
      [
        "Position alone may violate the Markov property because relevant battery information is missing.",
        true,
      ],
      [
        "Adding battery level to the state can make the next-step distribution depend only on the enriched current state and action.",
        true,
      ],
      [
        "The process is Markov merely because decisions occur at discrete time steps.",
        false,
      ],
      [
        "A Markov state must contain the complete raw history rather than a sufficient summary.",
        false,
      ],
    ],
    "The Markov property depends on whether the state contains enough information to predict the future under an action. It need not preserve every past observation, but omitting a causally relevant battery variable can make older history informative beyond position.",
  ),
  makeQuestion(
    "crash-probability-l4-q65",
    "medium",
    "Which statements correctly describe a Markov decision process (MDP)?",
    [
      [
        "States summarize information relevant to future transitions and rewards.",
        true,
      ],
      ["Actions influence the distribution of next states and rewards.", true],
      [
        "A transition kernel assigns probabilities to next states for each state-action pair.",
        true,
      ],
      [
        "A policy and environment dynamics play different roles in generating a trajectory.",
        true,
      ],
    ],
    "An MDP separates what the agent controls—the policy over actions—from how the environment responds through transitions and rewards. A sufficient state makes that response depend on the current state and action rather than the full recorded past.",
  ),
  makeQuestion(
    "crash-probability-l4-q66",
    "hard",
    "For fixed state s and action a, next states \\(u,v,w\\) have probabilities 0.5, 0.3, and 0.2 with immediate rewards 1, 4, and -2. Which statements are correct?",
    [
      [
        "The expected immediate reward is \\(0.5(1)+0.3(4)+0.2(-2)=1.3\\).",
        true,
      ],
      ["The probability of reaching either u or v is 0.8.", true],
      [
        "A model can estimate expected outcomes by summing over the three transition paths.",
        true,
      ],
      [
        "The formula \\(Q(s,a)=\\max_{s'}P(s'\\mid s,a)R(s,a,s')\\) establishes optimality without alternatives or future values.",
        false,
      ],
    ],
    "The transition paths are mutually exclusive, so event probabilities and expected rewards add with their proper weights. An action's immediate expectation is not enough to declare it optimal when other actions and downstream consequences have not been evaluated.",
  ),
  makeQuestion(
    "crash-probability-l4-q67",
    "easy",
    "Which expression states the controlled Markov property?",
    [
      [
        "\\(P(S_{t+1}\\mid S_t,A_t,\\text{history})=P(S_{t+1}\\mid S_t,A_t)\\)",
        true,
      ],
      ["\\(P(S_{t+1})=P(S_t)\\)", false],
      ["\\(P(A_t\\mid S_t)=1\\) for every action", false],
      ["\\(S_{t+1}=S_t\\) on every transition", false],
    ],
    "The Markov condition says the current state-action pair is sufficient for the next-state distribution, so older history adds no predictive information once they are known. It does not require stationary state marginals, deterministic actions, or unchanged states.",
  ),
  makeQuestion(
    "crash-probability-l4-q68",
    "medium",
    "A trajectory contains \\((S_0,A_0,R_1,S_1,A_1,R_2,S_2)\\). Which statements are correct?",
    [
      ["The policy helps determine each \\(A_t\\) from \\(S_t\\).", true],
      [
        "The environment helps determine \\(R_{t+1},S_{t+1}\\) after \\(S_t,A_t\\).",
        true,
      ],
      ["The policy directly chooses the realized next state.", false],
      [
        "The reward observed before action \\(A_t\\) must be labeled \\(R_{t+1}\\).",
        false,
      ],
    ],
    "A trajectory alternates agent action selection and environment response, with the next reward and state following the current action. The agent can influence but does not directly choose stochastic outcomes, and time indices distinguish preceding from following feedback.",
  ),
  makeQuestion(
    "crash-probability-l4-q69",
    "hard",
    "A process appears non-Markov when state is only the current webpage, because whether a user has purchased before affects the next action. Which statements are correct?",
    [
      [
        "Adding purchase history as a compact state feature may restore the Markov property.",
        true,
      ],
      [
        "The failure concerns state representation, not necessarily the underlying world process.",
        true,
      ],
      [
        "A recurrent model can use history when a compact sufficient state is unavailable.",
        true,
      ],
      [
        "Ignoring the missing information makes the Markov assumption exact by definition.",
        false,
      ],
    ],
    "Markovianity is relative to the chosen state: an omitted variable can leave predictive information in the past. Enriching state or using a history-dependent model addresses that missing information, while simply naming the representation a state does not make it sufficient.",
  ),
  makeQuestion(
    "crash-probability-l4-q70",
    "easy",
    "Which examples involve sequential decisions rather than one isolated prediction?",
    [
      ["A robot chooses turns that change its future location.", true],
      ["A recommender's choice changes what feedback it later observes.", true],
      ["A game agent acts repeatedly while scores and positions evolve.", true],
      [
        "A chatbot policy produces responses that affect the continuing conversation state.",
        true,
      ],
    ],
    "Each action changes either the environment, the information available later, or both, so present choices influence future opportunities and rewards. This feedback loop distinguishes sequential decision-making from a fixed one-shot input-to-label task.",
  ),
  makeQuestion(
    "crash-probability-l4-q71",
    "medium",
    "An environment is deterministic given s and a. Which statements are correct?",
    [
      [
        "The transition distribution can place probability one on a single next state.",
        true,
      ],
      [
        "Expected return can still matter if rewards arrive over multiple future steps.",
        true,
      ],
      [
        "A deterministic environment forces the policy to be deterministic.",
        false,
      ],
      [
        "There can be no uncertainty from the initial state or a randomized policy.",
        false,
      ],
    ],
    "Deterministic dynamics are represented by degenerate transition distributions, but the agent may still randomize and initial conditions may vary. Even with fixed transitions, delayed rewards require adding or discounting consequences over time.",
  ),
  makeQuestion(
    "crash-probability-l4-q72",
    "hard",
    "A robot action reaches goal with probability 0.6 for reward 5, remains in place with 0.3 for reward -1, or enters failure with 0.1 for reward -10. Which statements are correct?",
    [
      [
        "The expected immediate reward is \\(0.6(5)+0.3(-1)+0.1(-10)=1.7\\).",
        true,
      ],
      ["The non-goal probability is 0.4.", true],
      [
        "A complete action comparison also needs future values if remaining states continue.",
        true,
      ],
      [
        "The goal path alone gives \\(Q(s,a)=P(\\text{goal}\\mid s,a)R(\\text{goal})=0.6(5)\\).",
        false,
      ],
    ],
    "All mutually exclusive outcomes contribute to expectation, including low-probability failure, giving 1.7. If some successors are nonterminal, their future returns also matter, so selecting only the modal path discards relevant probability and value.",
  ),

  // Policies, exploration, and action sampling
  makeQuestion(
    "crash-probability-l4-q73",
    "easy",
    "What does a stochastic policy \\(\\pi(a\\mid s)\\) provide?",
    [
      ["A probability distribution over actions in state s", true],
      ["A probability distribution over prior states given an action", false],
      ["The environment's reward for every possible trajectory", false],
      ["A guarantee that the highest-value action is selected", false],
    ],
    "A stochastic policy maps the observed state to normalized action probabilities and can be sampled to choose behavior. It is separate from transition and reward models and does not guarantee the maximizing action is taken on each visit.",
  ),
  makeQuestion(
    "crash-probability-l4-q74",
    "easy",
    "A deterministic policy and a stochastic policy are compared. Which statements are correct?",
    [
      ["A deterministic policy chooses one specified action per state.", true],
      [
        "A stochastic policy can assign positive probability to several actions in a state.",
        true,
      ],
      [
        "A deterministic policy requires deterministic environment transitions.",
        false,
      ],
      ["A stochastic policy makes every action equally likely.", false],
    ],
    "Policy randomness describes action selection and is independent of whether the environment itself is stochastic. A stochastic policy can be highly uneven, while a deterministic policy is the special case placing all mass on one action.",
  ),
  makeQuestion(
    "crash-probability-l4-q75",
    "medium",
    "There are four actions and an epsilon-greedy policy with \\(\\epsilon=0.20\\) chooses a uniformly random action during exploration. Which statements are correct?",
    [
      ["The greedy action has probability \\(0.80+0.20/4=0.85\\).", true],
      ["Each non-greedy action has probability \\(0.20/4=0.05\\).", true],
      ["All action probabilities sum to one.", true],
      [
        "The greedy action has probability exactly 0.80 because exploration excludes it.",
        false,
      ],
    ],
    "Under the stated convention, the random exploration draw includes all four actions, so the greedy action receives both exploitation and one exploration share. Excluding it would define a different epsilon-greedy variant and produce different probabilities.",
  ),
  makeQuestion(
    "crash-probability-l4-q76",
    "hard",
    "Action estimates are \\((3,2,1)\\), and softmax action selection uses temperature T. Which statements are correct?",
    [
      [
        "At any positive finite T, the action with estimate 3 has the largest selection probability.",
        true,
      ],
      [
        "Increasing T flattens the action distribution and generally increases exploration.",
        true,
      ],
      [
        "Increasing T changes the estimate ordering and makes action 1 optimal.",
        false,
      ],
      [
        "Softmax selection assigns zero probability to every non-greedy action.",
        false,
      ],
    ],
    "Positive temperature rescales score gaps without changing their ordering, so the best-estimated action remains most probable. A higher temperature gives alternatives more mass, unlike greedy selection, which can give nonleaders zero probability.",
  ),
  makeQuestion(
    "crash-probability-l4-q77",
    "medium",
    "Which statements describe the exploration–exploitation tradeoff?",
    [
      ["Exploitation uses current estimates to seek reward.", true],
      ["Exploration gathers information about uncertain actions.", true],
      [
        "Early exploration can improve later decisions by reducing uncertainty.",
        true,
      ],
      [
        "The best balance can depend on horizon, uncertainty, and how quickly the environment changes.",
        true,
      ],
    ],
    "Exploitation realizes value from current knowledge, while exploration can sacrifice immediate reward to improve that knowledge. How much information is worth acquiring depends on how long it can be used and whether action values remain stable.",
  ),
  makeQuestion(
    "crash-probability-l4-q78",
    "hard",
    "A two-armed bandit has estimated rewards 5 and 4, but the second estimate is based on one sample while the first uses 1,000 samples. Which statements are correct?",
    [
      [
        "Greedy exploitation chooses the first arm from the current means.",
        true,
      ],
      [
        "Exploring the second arm can be valuable because its estimate is much more uncertain.",
        true,
      ],
      [
        "A longer remaining horizon can increase the value of learning about the second arm.",
        true,
      ],
      [
        "The one-sample mean proves the second arm's true value is exactly 4.",
        false,
      ],
    ],
    "Point estimates favor the first arm, but their different evidence amounts imply very different uncertainty. Information about the under-sampled arm can influence many future choices, especially when the remaining horizon is long.",
  ),
  makeQuestion(
    "crash-probability-l4-q79",
    "easy",
    "An entropy bonus is added to a policy objective. What behavior does it usually encourage?",
    [
      ["A less concentrated action distribution", true],
      ["A deterministic environment transition", false],
      ["Removal of all low-reward actions from the action set", false],
      ["A discount factor greater than one", false],
    ],
    "Policy entropy is larger when action mass is spread more broadly, so a positive bonus discourages premature collapse to one action. It changes the optimization preference over policies, not the environment dynamics, available actions, or discount-factor definition.",
  ),
  makeQuestion(
    "crash-probability-l4-q80",
    "medium",
    "A policy assigns action probabilities \\((0.7,0.2,0.1)\\). Which statements are correct?",
    [
      [
        "Sampling can select the third action even though it is least probable.",
        true,
      ],
      [
        "Over many visits to the same state, frequencies should approach the policy probabilities under stable sampling.",
        true,
      ],
      [
        "The first action is guaranteed on the next visit because it is the argmax.",
        false,
      ],
      [
        "The probabilities describe next-state uncertainty rather than action choice.",
        false,
      ],
    ],
    "A stochastic policy turns action probabilities into a random realized action, so positive-mass alternatives remain possible. The frequencies converge in repeated comparable trials, but a single sample is not guaranteed to equal the mode.",
  ),
  makeQuestion(
    "crash-probability-l4-q81",
    "hard",
    "A policy's action distribution changes from \\((0.5,0.5)\\) to \\((0.95,0.05)\\). Which statements are correct?",
    [
      ["Policy entropy decreases.", true],
      ["The first action is selected more often in expectation.", true],
      [
        "Coverage of outcomes reachable mainly through the second action will likely decrease.",
        true,
      ],
      [
        "The first action's true value necessarily increased during this change.",
        false,
      ],
    ],
    "The new policy is more concentrated and therefore explores the second action less, altering the distribution of collected experience. Policy probabilities can change because of optimization or temperature even if the environment's underlying action values did not increase.",
  ),
  makeQuestion(
    "crash-probability-l4-q82",
    "easy",
    "Which mechanisms can deliberately introduce action randomness?",
    [
      ["Epsilon-greedy exploration", true],
      ["Softmax action sampling", true],
      ["A stochastic policy network", true],
      ["An entropy-regularized objective", true],
    ],
    "These mechanisms either define a random action-selection rule directly or encourage the learned distribution to retain spread. Their precise probabilities differ, but each can prevent behavior from collapsing immediately to a single action.",
  ),
  makeQuestion(
    "crash-probability-l4-q83",
    "medium",
    "Which statements distinguish on-policy data from environment dynamics?",
    [
      [
        "Changing the policy changes which state-action pairs are visited.",
        true,
      ],
      [
        "The same transition kernel can generate different trajectory distributions under different policies.",
        true,
      ],
      [
        "Changing the policy mathematically rewrites the environment's fixed transition probabilities.",
        false,
      ],
      [
        "A deterministic policy guarantees every trajectory is deterministic.",
        false,
      ],
    ],
    "Policy and transition probabilities combine to produce trajectory frequencies, so policy changes alter the data distribution even if environment dynamics stay fixed. Stochastic transitions or initial states can still produce varied trajectories under a deterministic policy.",
  ),
  makeQuestion(
    "crash-probability-l4-q84",
    "hard",
    "An agent uses epsilon-greedy behavior while estimating action values. Which statements are correct?",
    [
      [
        "Reducing epsilon over time can shift from information gathering toward exploitation.",
        true,
      ],
      ["Keeping some exploration can help track a changing environment.", true],
      [
        "Exploration changes the distribution of observed rewards and states.",
        true,
      ],
      [
        "An epsilon of zero gives unbiased information about every unchosen action.",
        false,
      ],
    ],
    "An exploration schedule changes both behavior and the evidence available for learning. Pure greed may stop sampling alternatives, leaving their current values uncertain or stale, especially if the environment evolves.",
  ),

  // Return and discounting
  makeQuestion(
    "crash-probability-l4-q85",
    "easy",
    "For rewards 2, 3, and 4 over the next three steps with \\(\\gamma=1\\), what is the finite-horizon return?",
    [
      ["\\(2+3+4=9\\)", true],
      ["\\(2\\times3\\times4=24\\)", false],
      ["\\((2+3+4)/3=3\\)", false],
      ["\\(4\\) because only the final reward counts", false],
    ],
    "With no discounting over this finite horizon, return is the sum of future rewards, giving nine. It is neither a product nor an average, and intermediate rewards remain part of the objective. The finite horizon keeps this undiscounted sum well defined.",
  ),
  makeQuestion(
    "crash-probability-l4-q86",
    "easy",
    "For rewards 5 now and 10 one step later with \\(\\gamma=0.5\\), which return calculations are correct?",
    [
      ["The return is \\(5+0.5(10)=10\\).", true],
      ["The later reward contributes 5 discounted units.", true],
      ["The return is \\(0.5(5)+10=12.5\\).", false],
      [
        "The return is 15 because discounting changes probabilities, not rewards.",
        false,
      ],
    ],
    "The immediate reward has exponent zero and the next reward is multiplied by one factor of gamma. Discounting changes how future reward contributes to the objective; it does not reverse the order or leave the sum unchanged.",
  ),
  makeQuestion(
    "crash-probability-l4-q87",
    "medium",
    "Which statements correctly interpret the discount factor \\(0\\le\\gamma<1\\)?",
    [
      ["Smaller gamma places less weight on distant rewards.", true],
      ["Larger gamma makes long-delayed consequences more influential.", true],
      [
        "Geometric discounting can keep an infinite constant-reward sum finite.",
        true,
      ],
      ["Gamma is the probability that the current action is optimal.", false],
    ],
    "Discounting multiplies rewards k steps away by powers of gamma, controlling horizon preference and convergence of continuing sums. It is an objective parameter, not a confidence or optimal-action probability.",
  ),
  makeQuestion(
    "crash-probability-l4-q88",
    "hard",
    "An action yields reward 1 forever, starting next step. Which statements are correct for \\(0\\le\\gamma<1\\)?",
    [
      ["Its return is \\(1+\\gamma+\\gamma^2+\\cdots=1/(1-\\gamma)\\).", true],
      ["At \\(\\gamma=0.9\\), the return is 10.", true],
      [
        "The return is one for every gamma because each reward equals one.",
        false,
      ],
      ["The series diverges for every gamma below one.", false],
    ],
    "The repeated rewards form a convergent geometric series when the discount factor is below one. At 0.9 its sum is ten, showing how many individually small future rewards can create substantial value. Each additional term is discounted by another factor of 0.9.",
  ),
  makeQuestion(
    "crash-probability-l4-q89",
    "medium",
    "Which statements correctly distinguish reward and return?",
    [
      ["Reward is feedback associated with a particular transition.", true],
      ["Return aggregates current and future rewards from a time step.", true],
      [
        "Return can be random because future states, actions, and rewards are uncertain.",
        true,
      ],
      [
        "Expected return averages that random return under policy and environment probabilities.",
        true,
      ],
    ],
    "A reward is one local signal, whereas return combines a sequence of such signals, often with discounting. Because trajectories are random, value functions usually target the expectation of return rather than a single realized trajectory total.",
  ),
  makeQuestion(
    "crash-probability-l4-q90",
    "hard",
    "Action A gives reward 4 immediately. Action B gives reward 10 two steps later with probability 0.6 and zero otherwise; there are no other rewards and \\(\\gamma=0.8\\). Which statements are correct?",
    [
      ["A has expected return 4.", true],
      ["B has expected return \\(0.6(0.8^2)(10)=3.84\\).", true],
      ["A has slightly larger expected discounted return.", true],
      ["B is preferred because its successful payoff 10 exceeds 4.", false],
    ],
    "B's payoff must be weighted by both its probability and two discount factors, reducing its expected return below four. Comparing only successful payoff ignores failure probability and delay, the two central features of this decision.",
  ),
  makeQuestion(
    "crash-probability-l4-q91",
    "easy",
    "What does \\(G_t=R_{t+1}+\\gamma R_{t+2}+\\gamma^2R_{t+3}+\\cdots\\) represent?",
    [
      ["Discounted return from time t", true],
      ["Probability of the next action", false],
      ["Entropy of the transition model", false],
      ["Immediate reward with future rewards removed", false],
    ],
    "The return is a random sum of rewards following time t, with later terms receiving increasing powers of gamma. It is not a normalized probability, an uncertainty measure, or merely the first reward. Its expectation is what the corresponding value function summarizes.",
  ),
  makeQuestion(
    "crash-probability-l4-q92",
    "medium",
    "Two policies have equal expected undiscounted reward over ten steps, but one earns rewards earlier. Which statements are correct?",
    [
      [
        "With \\(\\gamma<1\\), the earlier-reward policy can have larger expected discounted return.",
        true,
      ],
      [
        "With \\(\\gamma=1\\) and the stated finite horizon, timing alone does not change the sum.",
        true,
      ],
      [
        "Discounting necessarily reverses the ranking regardless of reward timing and size.",
        false,
      ],
      [
        "Gamma changes environment transition probabilities rather than the objective.",
        false,
      ],
    ],
    "Discounting values a reward less when it arrives later, so timing can break a tie in undiscounted totals. At gamma one the finite sum ignores timing, and no universal reversal follows when reward magnitudes and times differ.",
  ),
  makeQuestion(
    "crash-probability-l4-q93",
    "hard",
    "A random return is 0 with probability 0.5, 4 with probability 0.3, and 10 with probability 0.2. Which statements are correct?",
    [
      ["Its expected return is \\(0(0.5)+4(0.3)+10(0.2)=3.2\\).", true],
      ["The most likely return is 0.", true],
      [
        "Expectation can rank the policy even though 3.2 is not a possible realization.",
        true,
      ],
      [
        "Expected return selects the best support value: \\(\\mathbb{E}[G]=\\max\\{0,4,10\\}=10\\).",
        false,
      ],
    ],
    "Expected return is the probability-weighted trajectory outcome, which can differ from both the mode and every possible realization. Maximizing only the best-case payoff would ignore how rarely it occurs and is not the ordinary risk-neutral RL objective.",
  ),
  makeQuestion(
    "crash-probability-l4-q94",
    "easy",
    "Which factors can affect an agent's expected return?",
    [
      ["Its action-selection policy", true],
      ["Environment transition probabilities", true],
      ["Rewards assigned to transitions", true],
      ["The discount factor used by the objective", true],
    ],
    "The policy changes actions, dynamics change successor states, rewards score the paths, and discounting changes how delayed scores are weighted. Expected return averages the combined trajectory consequences of all four ingredients.",
  ),
  makeQuestion(
    "crash-probability-l4-q95",
    "medium",
    "Policy A has expected return 6 and standard deviation 1; Policy B has expected return 6 and standard deviation 8. Which statements are correct?",
    [
      [
        "A risk-neutral expected-return objective is indifferent between them.",
        true,
      ],
      [
        "A risk-sensitive decision maker can prefer A because its outcomes are less variable.",
        true,
      ],
      [
        "Policy B must be better because larger standard deviation increases expectation.",
        false,
      ],
      ["Equal means imply identical return distributions.", false],
    ],
    "The ordinary objective compares means and therefore ties the policies, while variance supplies additional information about outcome risk. Spread does not automatically raise the mean, and equal expectations can arise from very different distributions.",
  ),
  makeQuestion(
    "crash-probability-l4-q96",
    "hard",
    "A trajectory has rewards \\(R_1=2,R_2=-1,R_3=5\\) and \\(\\gamma=0.5\\). Which statements are correct from time 0?",
    [
      ["\\(G_0=2+0.5(-1)+0.5^2(5)=2.75\\).", true],
      ["The final reward contributes 1.25 to \\(G_0\\).", true],
      ["A negative intermediate reward reduces the return by 0.5.", true],
      ["\\(G_0=(2-1+5)/3=2\\) because return averages rewards.", false],
    ],
    "Each reward is weighted by a power determined by its delay, giving \\(2-0.5+1.25=2.75\\). Return is a discounted sum, not an arithmetic mean, so timing and signs both matter. The negative reward receives one discount factor because it arrives one step after the first reward.",
  ),

  // Value functions and Bellman recursion
  makeQuestion(
    "crash-probability-l4-q97",
    "easy",
    "What does \\(V^\\pi(s)\\) represent?",
    [
      [
        "Expected return starting in state s and then following policy \\(\\pi\\)",
        true,
      ],
      ["The immediate reward of the most recent action", false],
      ["The probability that s is visited in the dataset", false],
      ["The number of actions available in s", false],
    ],
    "A state-value function averages the future discounted return induced by a specified policy and the environment from state s. It is not merely one reward, a visitation frequency, or an action count. Changing the policy can therefore change the value of the same environment state.",
  ),
  makeQuestion(
    "crash-probability-l4-q98",
    "easy",
    "Which statements distinguish \\(V^\\pi(s)\\) from \\(Q^\\pi(s,a)\\)?",
    [
      ["V evaluates a state before fixing the next action.", true],
      ["Q evaluates taking action a in s and then following \\(\\pi\\).", true],
      ["V is a transition probability and Q is a reward probability.", false],
      ["Q ignores all rewards after the first transition.", false],
    ],
    "State value averages over the action chosen by the policy, while action value conditions on a particular first action. Both concern expected cumulative return rather than probabilities alone, and both include downstream rewards.",
  ),
  makeQuestion(
    "crash-probability-l4-q99",
    "medium",
    "From state s, a policy chooses actions a and b with probabilities 0.6 and 0.4. Their action values are 5 and 2. Which statements are correct?",
    [
      ["\\(V^\\pi(s)=0.6(5)+0.4(2)=3.8\\).", true],
      [
        "The policy-weighted state value lies between the two action values.",
        true,
      ],
      [
        "Changing the policy probabilities can change \\(V^\\pi(s)\\) even if both Q values stay fixed.",
        true,
      ],
      [
        "\\(V^\\pi(s)=5\\) because a has the largest action probability.",
        false,
      ],
    ],
    "State value under a stochastic policy is the expectation of action value over the policy's action distribution. Selecting only the modal action would describe greedy execution, not the stated randomized policy.",
  ),
  makeQuestion(
    "crash-probability-l4-q100",
    "hard",
    "Action a from s gives reward 1 and then reaches u with probability 0.75 or v with 0.25. Values are \\(V(u)=4,V(v)=0\\), with \\(\\gamma=0.8\\). Which statements are correct?",
    [
      ["The expected next-state value is \\(0.75(4)+0.25(0)=3\\).", true],
      ["\\(Q(s,a)=1+0.8(3)=3.4\\).", true],
      ["\\(Q(s,a)=1+4=5\\) because u is the likely successor.", false],
      [
        "The transition probabilities are applied after discounting by adding them to gamma.",
        false,
      ],
    ],
    "The one-step Bellman backup averages successor values using transition probabilities, discounts that expectation, and adds immediate reward. Using only the modal successor or adding probabilities to gamma discards the structure of the expectation.",
  ),
  makeQuestion(
    "crash-probability-l4-q101",
    "medium",
    "Which statements correctly describe a Bellman expectation equation?",
    [
      [
        "It separates return into immediate reward plus discounted future return.",
        true,
      ],
      ["It averages over actions selected by the policy.", true],
      ["It averages over stochastic environment transitions.", true],
      [
        "It defines a value in terms of successor values, creating a recursive system.",
        true,
      ],
    ],
    "Bellman recursion is first-step analysis applied to decision processes: condition on the first action and transition, then reuse the value definition from the successor state. Policy and environment probabilities supply the weights in that expectation.",
  ),
  makeQuestion(
    "crash-probability-l4-q102",
    "hard",
    "A state s gives reward 2, then remains in s with probability 0.5 or terminates with probability 0.5. Let \\(\\gamma=0.8\\). Which statements are correct?",
    [
      ["Its value satisfies \\(V(s)=2+0.8[0.5V(s)+0.5(0)]\\).", true],
      ["Solving gives \\(V(s)=2/(1-0.4)=10/3\\).", true],
      [
        "The self-loop creates repeated opportunities for reward that are captured recursively.",
        true,
      ],
      [
        "The value is 2 because only the first visit can contribute reward.",
        false,
      ],
    ],
    "Conditioning on the next transition reuses V after a self-loop and zero after termination, yielding a solvable linear equation. Ignoring the loop omits possible future rewards and understates the expected return.",
  ),
  makeQuestion(
    "crash-probability-l4-q103",
    "easy",
    "If a terminal state has no future rewards, what value is conventionally assigned to it?",
    [
      ["0", true],
      ["1", false],
      ["The discount factor", false],
      ["The probability of entering it", false],
    ],
    "With no remaining rewards, the return from the terminal state is the empty sum and has value zero. Entry probability affects predecessor values, but it does not become the terminal state's future return.",
  ),
  makeQuestion(
    "crash-probability-l4-q104",
    "medium",
    "Two actions in state s have \\(Q(s,a)=4\\) and \\(Q(s,b)=6\\). Which statements are correct?",
    [
      ["A greedy policy selects b.", true],
      ["A stochastic policy can still assign positive probability to a.", true],
      ["The values prove b produces reward 6 on every trajectory.", false],
      ["The values are transition probabilities and must sum to one.", false],
    ],
    "Action values are expected returns, so greedy selection chooses the larger mean while exploratory or entropy-regularized policies may still sample the other action. Values are not normalized probabilities and do not guarantee a particular realized return.",
  ),
  makeQuestion(
    "crash-probability-l4-q105",
    "hard",
    "A policy in s chooses a with probability 0.25 and b with 0.75. Each action has deterministic immediate reward 0; a leads to value 8 and b to value 2, with \\(\\gamma=0.5\\). Which statements are correct?",
    [
      ["\\(Q(s,a)=0.5(8)=4\\).", true],
      ["\\(Q(s,b)=0.5(2)=1\\).", true],
      ["\\(V^\\pi(s)=0.25(4)+0.75(1)=1.75\\).", true],
      ["\\(V^\\pi(s)=4\\) because action a has the largest Q value.", false],
    ],
    "Action values discount their successor values, and state value then averages those action values under the actual policy. Replacing the stochastic policy by argmax would evaluate a different policy and give a different V.",
  ),
  makeQuestion(
    "crash-probability-l4-q106",
    "easy",
    "Which uses of value functions are correct?",
    [
      ["Comparing states by expected future return", true],
      ["Comparing actions within a state", true],
      ["Bootstrapping current estimates from successor estimates", true],
      ["Approximating long-term consequences with a neural network", true],
    ],
    "State and action values compress uncertain future trajectories into expected-return summaries that support planning and learning. Bellman recursion permits bootstrapping, and function approximation extends the idea to large state or action spaces.",
  ),
  makeQuestion(
    "crash-probability-l4-q107",
    "medium",
    "A value estimate is high in a state with a small immediate reward. Which statements can explain this?",
    [
      [
        "The state may lead with high probability to large future rewards.",
        true,
      ],
      ["A high discount factor can make delayed rewards influential.", true],
      [
        "Value must equal the current reward, so the estimate is definitionally invalid.",
        false,
      ],
      ["Transition uncertainty cannot affect state value.", false],
    ],
    "Value includes the entire expected discounted future, so good successor states can dominate a modest immediate reward. Transition probabilities determine how likely those successors are, and gamma determines how strongly their rewards matter.",
  ),
  makeQuestion(
    "crash-probability-l4-q108",
    "hard",
    "For a fixed policy, values satisfy simultaneous Bellman equations across several recurrent states. Which statements are correct?",
    [
      ["Each equation can contain values of successor states.", true],
      [
        "Cycles couple the unknown values rather than invalidating expectation.",
        true,
      ],
      [
        "The system can often be solved algebraically or by iterative updates.",
        true,
      ],
      [
        "Every recurrent value must be infinite even when rewards are bounded and \\(\\gamma<1\\).",
        false,
      ],
    ],
    "Recurrence produces a linked system because future paths revisit states, but discounting with bounded rewards can keep the solution finite. Linear solving or repeated Bellman backups exploit the same recursive fixed-point structure.",
  ),

  // First-step expectation and waiting times
  makeQuestion(
    "crash-probability-l4-q109",
    "easy",
    "A fair coin is flipped until the first head. If E is the expected number of flips from the start, which equation is correct?",
    [
      ["\\(E=1+0.5E\\)", true],
      ["\\(E=0.5+E\\)", false],
      ["\\(E=1+E\\)", false],
      ["\\(E=0.5E\\)", false],
    ],
    "Every attempt uses one flip; after a tail, which occurs with probability one half, the process returns to the same state and expects E more flips. A head contributes no additional waiting after that first counted flip, giving \\(E=1+0.5E\\).",
  ),
  makeQuestion(
    "crash-probability-l4-q110",
    "easy",
    "Solving \\(E=1+0.5E\\) for the fair-coin first-head waiting time gives which conclusions?",
    [
      ["\\(E=2\\).", true],
      [
        "The recursive equation replaces an infinite list of possible tail sequences.",
        true,
      ],
      [
        "E must be one because head is the most likely single stopping outcome.",
        false,
      ],
      ["The expected value says every run lasts exactly two flips.", false],
    ],
    "Rearranging gives \\(0.5E=1\\), hence two flips on average, matching the geometric waiting-time formula. The result summarizes repetitions and does not claim that individual runs cannot stop after one flip or continue much longer.",
  ),
  makeQuestion(
    "crash-probability-l4-q111",
    "medium",
    "A trial succeeds independently with probability p on each attempt. Which statements about the waiting time N to first success are correct?",
    [
      ["First-step analysis gives \\(E[N]=1+(1-p)E[N]\\).", true],
      ["Solving gives \\(E[N]=1/p\\).", true],
      ["Lower p increases the expected wait.", true],
      [
        "The expected wait is \\(1/(1-p)\\) because failure repeats the process.",
        false,
      ],
    ],
    "Each attempt costs one step, and only failure returns to the starting state, yielding \\(pE[N]=1\\). The repeat probability appears in the equation, but solving leaves the success probability p in the denominator.",
  ),
  makeQuestion(
    "crash-probability-l4-q112",
    "hard",
    "For fair-coin flips until two consecutive heads, let \\(E_0\\) be the expected remaining flips with no current head and \\(E_1\\) after one current head. Which equations are correct?",
    [
      ["\\(E_0=1+0.5E_0+0.5E_1\\).", true],
      ["\\(E_1=1+0.5E_0\\).", true],
      ["\\(E_1=1+0.5E_1\\) because a tail preserves one current head.", false],
      ["\\(E_0=2\\) because two heads require two expected flips.", false],
    ],
    "From state 0, a tail returns to 0 and a head moves to state 1; from state 1, a head finishes while a tail loses the streak and returns to 0. The state must remember the partial pattern, so treating both steps as independent first-head waits misses overlap and reset behavior.",
  ),
  makeQuestion(
    "crash-probability-l4-q113",
    "medium",
    "Solving \\(E_0=1+0.5E_0+0.5E_1\\) and \\(E_1=1+0.5E_0\\) yields which statements?",
    [
      ["\\(E_0=6\\).", true],
      ["\\(E_1=4\\).", true],
      [
        "Starting with a head already achieved shortens the expected remaining wait.",
        true,
      ],
      [
        "The state equations account for tails resetting a partial HH pattern.",
        true,
      ],
    ],
    "Substitution gives \\(E_0=3+0.75E_0\\), so \\(E_0=6\\), and then \\(E_1=4\\). The difference between states quantifies the useful progress represented by a current head and the cost of losing it after a tail. Individual runs can still finish sooner or later than these expectations.",
  ),
  makeQuestion(
    "crash-probability-l4-q114",
    "hard",
    "A random walk at position i moves right with probability p and left with probability \\(1-p\\). Let \\(E_i\\) be expected steps to an absorbing boundary. Which statements are correct?",
    [
      ["For an interior state, \\(E_i=1+pE_{i+1}+(1-p)E_{i-1}\\).", true],
      ["Absorbing boundary states have expected remaining time zero.", true],
      ["One equation per state can form a simultaneous linear system.", true],
      [
        "Expected time equals the distance to the nearest boundary regardless of p or backward moves.",
        false,
      ],
    ],
    "First-step analysis counts the next move and averages the remaining expectation at each possible successor. Backtracking and drift affect the wait, so distance alone is insufficient except in special deterministic cases.",
  ),
  makeQuestion(
    "crash-probability-l4-q115",
    "easy",
    "What is the first step in building a recursive expected-waiting-time model?",
    [
      [
        "Define states that retain the information needed to predict what happens next.",
        true,
      ],
      ["List every infinite trajectory before writing any equation.", false],
      ["Assume all waiting times equal the number of states.", false],
      ["Replace transition probabilities with rewards.", false],
    ],
    "A useful state summarizes relevant progress, such as whether one head in an HH pattern has already appeared. Once states are sufficient, conditioning on the next outcome produces compact expectation equations without enumerating infinitely many paths.",
  ),
  makeQuestion(
    "crash-probability-l4-q116",
    "medium",
    "A machine succeeds on an attempt with probability 0.25 and otherwise returns to the same ready state. Which statements are correct?",
    [
      ["The expected attempts satisfy \\(E=1+0.75E\\).", true],
      ["The expected number of attempts is 4.", true],
      [
        "The expected attempts are 0.25 because that is the success probability.",
        false,
      ],
      [
        "The expectation is infinite because failure can occur repeatedly.",
        false,
      ],
    ],
    "First-step recursion gives \\(0.25E=1\\), hence four attempts on average. Arbitrarily long failure runs are possible, but their probabilities shrink geometrically enough for the expectation to remain finite.",
  ),
  makeQuestion(
    "crash-probability-l4-q117",
    "hard",
    "A process waits for pattern HT in fair coin flips. Let \\(E_0\\) mean no useful prefix and \\(E_H\\) mean the last flip was H. Which statements are correct?",
    [
      ["\\(E_0=1+0.5E_0+0.5E_H\\).", true],
      [
        "\\(E_H=1+0.5E_H\\) because another H preserves the useful suffix H while T finishes.",
        true,
      ],
      ["Solving gives \\(E_H=2\\) and \\(E_0=4\\).", true],
      ["A second H returns to state 0 because HT has not completed.", false],
    ],
    "After H, another H still leaves a trailing H that can begin HT, while T completes the pattern. Preserving this overlap leads to the two equations and an expected wait of four from the start. Resetting after the second H would discard a useful suffix and overestimate the wait.",
  ),
  makeQuestion(
    "crash-probability-l4-q118",
    "easy",
    "Which problems can use first-step expectation equations?",
    [
      ["Waiting for a success in repeated trials", true],
      ["Waiting for a coin-flip pattern", true],
      ["Expected absorption time in a random walk", true],
      ["Bellman value equations in a Markov process", true],
    ],
    "Each problem conditions on the first transition and expresses the remaining quantity through successor states. The shared method is to define sufficient states, count the first step, average successor expectations, and solve the resulting recursion.",
  ),
  makeQuestion(
    "crash-probability-l4-q119",
    "medium",
    "Which statements correctly compare explicit infinite-series and first-step approaches to a geometric waiting time?",
    [
      [
        "The series approach sums \\(n(1-p)^{n-1}p\\) over stopping times n.",
        true,
      ],
      [
        "The first-step equation \\(E=1+(1-p)E\\) reaches the same mean more directly.",
        true,
      ],
      ["The first-step method assumes failure cannot repeat.", false],
      [
        "The series and recursion describe different random variables because one uses infinity.",
        false,
      ],
    ],
    "Both representations account for arbitrarily long failure runs and yield \\(1/p\\) when the series converges. Recursion compresses the repeated tail structure into a self-reference rather than listing each possible stopping time separately.",
  ),
  makeQuestion(
    "crash-probability-l4-q120",
    "hard",
    "A state s gives reward 1 each step and terminates with probability 0.25 after that reward; otherwise it returns to s. There is no discounting. Which statements are correct?",
    [
      ["Expected remaining steps satisfy \\(E=1+0.75E\\), so \\(E=4\\).", true],
      [
        "Expected total reward also equals 4 because each visited step gives reward 1.",
        true,
      ],
      [
        "The same recursive structure can be viewed as a waiting-time equation or a Bellman equation.",
        true,
      ],
      [
        "Expected reward is 1 because termination is possible after the first step.",
        false,
      ],
    ],
    "The self-loop repeats both the waiting process and the unit reward until termination, so the expected visit count and return coincide at four. This exposes the direct connection between first-step waiting-time reasoning and recursive value equations.",
  ),
];
