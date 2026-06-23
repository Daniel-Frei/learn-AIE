import { Question } from "../../quiz";

export const cs224rLecture4ActorCriticQuestions: Question[] = [
  {
    id: "cs224r-lect4-q36",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements correctly define the value, Q, and advantage functions used by actor-critic methods?",
    options: [
      {
        text: "\\(V^\\pi(s)\\) is the expected future return starting at state \\(s\\) and then following policy \\(\\pi\\).",
        isCorrect: true,
      },
      {
        text: "\\(Q^\\pi(s,a)\\) is the expected future return after starting at \\(s\\), taking action \\(a\\), and then following \\(\\pi\\).",
        isCorrect: true,
      },
      {
        text: "\\(V^\\pi(s)=\\mathbb{E}_{a\\sim\\pi(\\cdot\\mid s)}[Q^\\pi(s,a)]\\), so the state value averages over the policy's action choices.",
        isCorrect: true,
      },
      {
        text: "\\(A^\\pi(s,a)=Q^\\pi(s,a)-V^\\pi(s)\\), so advantage compares an action to the policy's baseline value at the same state.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture defines the critic quantities as expectations over future return under a policy. The key mathematical link is that a state value averages Q-values over the policy's action distribution, and advantage subtracts that state baseline from a specific action value.",
  },

  {
    id: "cs224r-lect4-q37",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which equations correctly relate \\(V^\\pi\\), \\(Q^\\pi\\), and \\(A^\\pi\\)?",
    options: [
      {
        text: "\\(V^\\pi(s)=\\mathbb{E}_{a\\sim\\pi(\\cdot\\mid s)}[Q^\\pi(s,a)]\\).",
        isCorrect: true,
      },
      {
        text: "\\(A^\\pi(s,a)=Q^\\pi(s,a)-V^\\pi(s)\\).",
        isCorrect: true,
      },
      {
        text: "\\(V^\\pi(s)=\\max_a Q^\\pi(s,a)\\) for any stochastic policy.",
        isCorrect: false,
      },
      {
        text: "\\(A^\\pi(s,a)=V^\\pi(s)-Q^\\pi(s,a)\\), so positive advantage means the action is worse than the policy average.",
        isCorrect: false,
      },
    ],
    explanation:
      "The value function is an expectation over actions sampled from the current policy, not a maximum over actions unless the policy has a special greedy form. Advantage is Q minus V, so its sign says whether the chosen action is better or worse than the policy's usual action mixture at that state.",
  },

  {
    id: "cs224r-lect4-q38",
    chapter: 4,
    difficulty: "hard",
    prompt: `A policy at state \\(s\\) chooses actions with these probabilities and Q-values:

| Action | \\(\\pi(a\\mid s)\\) | \\(Q^\\pi(s,a)\\) |
| --- | ---: | ---: |
| \\(a_1\\) | 0.7 | 2 |
| \\(a_2\\) | 0.3 | 5 |

Which option correctly computes \\(V^\\pi(s)\\) and the two advantages?`,
    options: [
      {
        text: "\\(V^\\pi(s)=2.9\\), \\(A^\\pi(s,a_1)=-0.9\\), and \\(A^\\pi(s,a_2)=2.1\\).",
        isCorrect: true,
      },
      {
        text: "\\(V^\\pi(s)=3.5\\), \\(A^\\pi(s,a_1)=-1.5\\), and \\(A^\\pi(s,a_2)=1.5\\).",
        isCorrect: false,
      },
      {
        text: "\\(V^\\pi(s)=5\\), \\(A^\\pi(s,a_1)=-3\\), and \\(A^\\pi(s,a_2)=0\\).",
        isCorrect: false,
      },
      {
        text: "\\(V^\\pi(s)=2.9\\), \\(A^\\pi(s,a_1)=0.9\\), and \\(A^\\pi(s,a_2)=-2.1\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The value is the policy-weighted average \\(0.7\\cdot2+0.3\\cdot5=2.9\\), not the unweighted average or maximum. Advantages are computed as \\(Q-V\\), so \\(a_1\\) is below the policy baseline and \\(a_2\\) is above it.",
  },

  {
    id: "cs224r-lect4-q39",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe replacing reward-to-go with an advantage estimate in the policy-gradient update?",
    options: [
      {
        text: "The actor update can use \\(\\nabla_\\theta J(\\theta)\\approx \\frac{1}{N}\\sum_i\\sum_t\\nabla_\\theta\\log\\pi_\\theta(a_{i,t}\\mid s_{i,t})\\hat A^\\pi(s_{i,t},a_{i,t})\\).",
        isCorrect: true,
      },
      {
        text: "When \\(\\hat A^\\pi(s_{i,t},a_{i,t})>0\\), gradient ascent increases \\(\\log\\pi_\\theta(a_{i,t}\\mid s_{i,t})\\).",
        isCorrect: true,
      },
      {
        text: "When \\(\\hat A^\\pi(s_{i,t},a_{i,t})<0\\), the score term pushes down the likelihood of \\(a_{i,t}\\) at \\(s_{i,t}\\).",
        isCorrect: true,
      },
      {
        text: "Using advantage changes the objective to \\(\\sum_i\\log\\pi_\\theta(a_i\\mid s_i)\\) and removes the expected-return objective.",
        isCorrect: false,
      },
    ],
    explanation:
      "The advantage is a lower-variance weighting signal for the same expected-return objective when it estimates the true advantage. The actor still follows a policy-gradient update; the critic only changes how credit is assigned to sampled actions.",
  },

  {
    id: "cs224r-lect4-q40",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which equations correctly express the one-step estimate used to turn a value estimate into an advantage estimate?",
    options: [
      {
        text: "\\(Q^\\pi(s_t,a_t)\\approx r(s_t,a_t)+V^\\pi(s_{t+1})\\) when \\(s_{t+1}\\) is the sampled next state.",
        isCorrect: true,
      },
      {
        text: "\\(A^\\pi(s_t,a_t)\\approx r(s_t,a_t)+V^\\pi(s_{t+1})-V^\\pi(s_t)\\).",
        isCorrect: true,
      },
      {
        text: "\\(A^\\pi(s_t,a_t)\\approx V^\\pi(s_t)-r(s_t,a_t)-V^\\pi(s_{t+1})\\), so a good action has a negative score.",
        isCorrect: false,
      },
      {
        text: "\\(Q^\\pi(s_t,a_t)\\approx V^\\pi(s_t)-r(s_t,a_t)\\), because the next state is ignored in temporal-difference learning.",
        isCorrect: false,
      },
    ],
    explanation:
      "The slide derives a one-step estimate by taking the observed immediate reward and adding the value of the sampled next state. Subtracting \\(V^\\pi(s_t)\\) converts that Q estimate into an advantage relative to the current state's baseline.",
  },

  {
    id: "cs224r-lect4-q41",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which steps are part of the on-policy actor-critic algorithm described in the lecture?",
    options: [
      {
        text: "Collect trajectories by running the current policy \\(\\pi_\\theta\\).",
        isCorrect: true,
      },
      {
        text: "Fit a critic such as \\(\\hat V^\\pi_\\phi\\) to return targets from the collected batch.",
        isCorrect: true,
      },
      {
        text: "Estimate advantages, for example \\(\\hat A(s_t,a_t)=r(s_t,a_t)+\\gamma\\hat V_\\phi(s_{t+1})-\\hat V_\\phi(s_t)\\).",
        isCorrect: true,
      },
      {
        text: "Update the actor with a step such as \\(\\theta\\leftarrow\\theta+\\alpha\\sum_{t,i}\\nabla_\\theta\\log\\pi_\\theta(a_{t,i}\\mid s_{t,i})\\hat A_{t,i}\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "Actor-critic adds a critic-fitting step between data collection and policy improvement. The actor still updates by policy gradient, but the reward-to-go weight is replaced by a learned estimate of how advantageous each sampled action was.",
  },

  {
    id: "cs224r-lect4-q42",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "For one transition, suppose \\(r_t=0.4\\), \\(\\gamma=0.9\\), \\(\\hat V(s_{t+1})=2.0\\), and \\(\\hat V(s_t)=1.3\\). Which value is the one-step advantage estimate \\(\\hat A_t=r_t+\\gamma\\hat V(s_{t+1})-\\hat V(s_t)\\)?",
    options: [
      { text: "\\(0.9\\)", isCorrect: true },
      { text: "\\(2.2\\)", isCorrect: false },
      { text: "\\(-0.9\\)", isCorrect: false },
      { text: "\\(0.5\\)", isCorrect: false },
    ],
    explanation:
      "The temporal-difference advantage estimate is \\(0.4+0.9\\cdot2.0-1.3=0.9\\). The common mistakes are forgetting the discount, failing to subtract the current value baseline, or reversing the sign of the baseline subtraction.",
  },

  {
    id: "cs224r-lect4-q43",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe Monte Carlo value-function estimation?",
    options: [
      {
        text: "A target can be \\(y_{i,t}=\\sum_{t'=t}^{T}r(s_{i,t'},a_{i,t'})\\).",
        isCorrect: true,
      },
      {
        text: "The critic can be fit with a regression loss such as \\(\\mathcal{L}(\\phi)=\\frac{1}{2}\\sum_i\\lVert\\hat V^\\pi_\\phi(s_i)-y_i\\rVert^2\\).",
        isCorrect: true,
      },
      {
        text: "The target uses the rollout's observed summed rewards rather than a bootstrap value estimate.",
        isCorrect: true,
      },
      {
        text: "The target is \\(y_{i,t}=r(s_{i,t},a_{i,t})+\\hat V_\\phi(s_{i,t+1})\\) and must be recomputed every gradient step.",
        isCorrect: false,
      },
    ],
    explanation:
      "Monte Carlo estimation supervises the value function with full sampled returns from the rollout. That avoids bootstrap bias, but the target can have high variance because it depends on all later stochastic events in the trajectory.",
  },

  {
    id: "cs224r-lect4-q44",
    chapter: 4,
    difficulty: "hard",
    prompt: `A trajectory prefix reaches state \\(p\\), receives reward 0, then reaches state \\(b\\). In the data, state \\(b\\) appears on two continuations with final rewards \\(-1\\) and \\(+1\\), so the current learned estimate is \\(\\hat V(b)=0\\). For state \\(p\\) on the negative trajectory, which statements compare Monte Carlo and bootstrapped targets correctly?`,
    options: [
      {
        text: "The Monte Carlo target for \\(p\\) on that trajectory is \\(-1\\).",
        isCorrect: true,
      },
      {
        text: "The one-step bootstrapped target for \\(p\\) is \\(0+\\hat V(b)=0\\).",
        isCorrect: true,
      },
      {
        text: "The Monte Carlo target for \\(p\\) must average the two continuations from \\(b\\), so it is \\(0\\).",
        isCorrect: false,
      },
      {
        text: "The bootstrapped target for \\(p\\) must ignore \\(\\hat V(b)\\), so it is \\(-1\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Monte Carlo uses the observed future return from the sampled trajectory, so the negative trajectory labels \\(p\\) with \\(-1\\). Bootstrapping can use the learned value of the next state, which has already aggregated information from both continuations through \\(\\hat V(b)=0\\).",
  },

  {
    id: "cs224r-lect4-q45",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe bootstrapping or temporal-difference value learning?",
    options: [
      {
        text: "A bootstrapped target can be \\(y_{i,t}=r(s_{i,t},a_{i,t})+\\hat V^\\pi_\\phi(s_{i,t+1})\\).",
        isCorrect: true,
      },
      {
        text: "The target \\(y_{i,t}\\) can change as \\(\\hat V^\\pi_\\phi(s_{i,t+1})\\) changes during gradient descent.",
        isCorrect: true,
      },
      {
        text: "Bootstrapping can reduce variance compared with full Monte Carlo returns \\(\\sum_{t'=t}^{T}r(s_{i,t'},a_{i,t'})\\).",
        isCorrect: true,
      },
      {
        text: "Bootstrapping can introduce bias when \\(\\hat V^\\pi_\\phi(s_{i,t+1})\\neq V^\\pi(s_{i,t+1})\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "Temporal-difference learning uses the model's own current estimate of the next state's value as part of the label. That can propagate information across related states with less variance, but it also means bad value estimates can contaminate the target.",
  },

  {
    id: "cs224r-lect4-q46",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Suppose an n-step target uses \\(n=3\\), rewards \\((r_t,r_{t+1},r_{t+2})=(1,0,2)\\), \\(\\gamma=0.9\\), and \\(\\hat V(s_{t+3})=5\\). Which value matches \\(y_t=\\sum_{k=0}^{2}\\gamma^k r_{t+k}+\\gamma^3\\hat V(s_{t+3})\\)?",
    options: [
      { text: "\\(6.265\\)", isCorrect: true },
      { text: "\\(8.0\\)", isCorrect: false },
      { text: "\\(5.445\\)", isCorrect: false },
      { text: "\\(3.52\\)", isCorrect: false },
    ],
    explanation:
      "The target is \\(1+0.9\\cdot0+0.9^2\\cdot2+0.9^3\\cdot5=1+0+1.62+3.645=6.265\\). Omitting the bootstrap term, discounting the wrong rewards, or summing undiscounted quantities gives the distractor values.",
  },

  {
    id: "cs224r-lect4-q47",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe n-step returns as the lecture's middle ground between Monte Carlo and one-step bootstrapping?",
    options: [
      {
        text: "\\(y_{i,t}=\\sum_{t'=t}^{t+n-1}r(s_{i,t'},a_{i,t'})+\\hat V^\\pi_\\phi(s_{i,t+n})\\) is the undiscounted version shown before adding \\(\\gamma\\).",
        isCorrect: true,
      },
      {
        text: "\\(n=1\\) recovers a one-step target like \\(r(s_{i,t},a_{i,t})+\\hat V^\\pi_\\phi(s_{i,t+1})\\), which usually has less variance than full Monte Carlo.",
        isCorrect: true,
      },
      {
        text: "Larger \\(n\\) includes more terms from \\(\\sum_{t'=t}^{t+n-1}r(s_{i,t'},a_{i,t'})\\), which can reduce bootstrap bias.",
        isCorrect: true,
      },
      {
        text: "Changing \\(n\\) changes the policy objective to \\(\\max_a Q^\\pi(s,a)\\), so the actor becomes greedier by definition.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture presents n-step returns as a bias-variance interpolation, not a different reward objective. Looking ahead farther includes more actual rewards, while bootstrapping sooner tends to keep the target less noisy.",
  },

  {
    id: "cs224r-lect4-q48",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the discount factor in the actor-critic value target?",
    options: [
      {
        text: "The one-step target becomes \\(y_{i,t}\\approx r(s_{i,t},a_{i,t})+\\gamma\\hat V^\\pi_\\phi(s_{i,t+1})\\).",
        isCorrect: true,
      },
      {
        text: "One interpretation of \\(\\gamma\\in[0,1]\\) is a modified Markov decision process with probability \\(1-\\gamma\\) of transitioning to a zero-reward terminal state.",
        isCorrect: true,
      },
      {
        text: "Discounting multiplies the policy-gradient log-probability term by \\(1-\\gamma\\) and leaves value targets unchanged.",
        isCorrect: false,
      },
      {
        text: "Using \\(\\gamma<1\\) means future rewards are removed entirely from the return.",
        isCorrect: false,
      },
    ],
    explanation:
      "Discounting scales the future value term and therefore reduces the influence of distant rewards. The lecture also gives the equivalent modified-MDP view: most ordinary transitions are scaled by \\(\\gamma\\), with the remaining probability going to a terminal zero-reward state.",
  },

  {
    id: "cs224r-lect4-q49",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "An infinite-horizon state receives reward 1 at every future step. Which option correctly compares undiscounted return with discounted return at \\(\\gamma=0.99\\)?",
    options: [
      {
        text: "The undiscounted sum diverges, while the discounted sum is \\(\\sum_{k=0}^{\\infty}0.99^k=100\\).",
        isCorrect: true,
      },
      {
        text: "The undiscounted sum is 100, while the discounted sum diverges.",
        isCorrect: false,
      },
      {
        text: "Both sums are 100 because the episode is infinite.",
        isCorrect: false,
      },
      {
        text: "Both sums are finite only when \\(\\gamma>1\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture motivates discounting partly because undiscounted infinite-horizon values can become infinitely large. With constant reward 1 and \\(\\gamma=0.99\\), the geometric series is \\(1/(1-0.99)=100\\).",
  },

  {
    id: "cs224r-lect4-q50",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which equations or steps match the full on-policy actor-critic walkthrough?",
    options: [
      {
        text: "Collect a batch \\(\\{(s_{1,i},a_{1,i},\\ldots,s_{T,i},a_{T,i})\\}\\) from \\(\\pi_\\theta\\).",
        isCorrect: true,
      },
      {
        text: "Fit \\(\\hat V^{\\pi_\\theta}_\\phi\\) to summed or n-step return targets in the batch.",
        isCorrect: true,
      },
      {
        text: "Estimate \\(\\hat A^{\\pi_\\theta}(s_{t,i},a_{t,i})=r(s_{t,i},a_{t,i})+\\gamma\\hat V^{\\pi_\\theta}_\\phi(s_{t+1,i})-\\hat V^{\\pi_\\theta}_\\phi(s_{t,i})\\).",
        isCorrect: true,
      },
      {
        text: "Update \\(\\theta\\leftarrow\\theta+\\alpha\\nabla_\\theta J(\\theta)\\) using a policy-gradient estimate weighted by \\(\\hat A\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The actor-critic walkthrough has a critic-fitting phase followed by an advantage-weighted actor update. The parameters \\(\\phi\\) belong to the critic, while \\(\\theta\\) belongs to the actor policy.",
  },

  {
    id: "cs224r-lect4-q51",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly distinguish critic parameters \\(\\phi\\) from actor parameters \\(\\theta\\)?",
    options: [
      {
        text: "\\(\\hat V_\\phi\\) is trained with a critic loss such as \\(\\mathcal{L}(\\phi)=\\frac{1}{2}\\sum_i\\lVert\\hat V_\\phi(s_i)-y_i\\rVert^2\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\pi_\\theta(a\\mid s)\\) is updated by a policy-gradient step such as \\(\\theta\\leftarrow\\theta+\\alpha\\nabla_\\theta J(\\theta)\\).",
        isCorrect: true,
      },
      {
        text: "Updating \\(\\phi\\) directly changes action probabilities in \\(\\pi_\\theta(a\\mid s)\\), so no separate actor step is needed.",
        isCorrect: false,
      },
      {
        text: "The critic loss is minimized with respect to \\(\\theta\\), while the actor update is minimized with respect to \\(\\phi\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The critic is the value estimator and the actor is the policy. They interact because the actor uses critic-derived advantages, but their parameter updates solve different optimization problems.",
  },

  {
    id: "cs224r-lect4-q52",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly explain why actor-critic can use data more efficiently than vanilla policy gradients in the lecture's walking and jacket-folding examples?",
    options: [
      {
        text: "A critic can learn that an intermediate state is progress even when the whole trajectory later receives a poor or sparse final reward.",
        isCorrect: true,
      },
      {
        text: "Bootstrapped values can propagate information backward through nearby states instead of relying only on one full return sample.",
        isCorrect: true,
      },
      {
        text: "Advantage estimates can make the actor reinforce a helpful early action while suppressing a later bad action.",
        isCorrect: true,
      },
      {
        text: "Actor-critic improves efficiency by discarding failed or partial-progress rollouts before training.",
        isCorrect: false,
      },
    ],
    explanation:
      "The point of the critic is not to throw away failures; failures can teach which states or actions have low value. By estimating progress at intermediate states, actor-critic can use partial trajectories that raw return-weighted policy gradients would treat too coarsely.",
  },

  {
    id: "cs224r-lect4-q53",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which expression is the actor-gradient estimate in the on-policy actor-critic walkthrough?",
    options: [
      {
        text: "\\(\\nabla_\\theta J(\\theta)\\approx\\sum_{t,i}\\nabla_\\theta\\log\\pi_\\theta(a_{t,i}\\mid s_{t,i})\\hat A^{\\pi_\\theta}(s_{t,i},a_{t,i})\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\nabla_\\phi J(\\phi)\\approx\\sum_{t,i}\\nabla_\\phi\\log\\pi_\\phi(a_{t,i}\\mid s_{t,i})\\hat V_\\phi(s_{t,i})\\).",
        isCorrect: false,
      },
      {
        text: "\\(\\nabla_\\theta J(\\theta)\\approx\\sum_{t,i}\\nabla_\\theta\\hat V_\\phi(s_{t,i})\\), so the actor follows the critic's value gradient directly.",
        isCorrect: false,
      },
      {
        text: "\\(\\nabla_\\theta J(\\theta)\\approx\\sum_{t,i}\\log\\pi_\\theta(a_{t,i}\\mid s_{t,i})\\nabla_\\theta\\hat A(s_{t,i},a_{t,i})\\), so the score term is not needed.",
        isCorrect: false,
      },
    ],
    explanation:
      "The actor update is still a score-function policy gradient: differentiate the log probability of the sampled action with respect to actor parameters. The advantage supplies the scalar weight; it is not a replacement for the log-probability gradient.",
  },

  {
    id: "cs224r-lect4-q63",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements correctly distinguish the actor and critic in actor-critic methods?",
    options: [
      {
        text: "The actor represents the policy \\(\\pi_\\theta(a\\mid s)\\), whose log probability appears in the actor-gradient term.",
        isCorrect: true,
      },
      {
        text: "The critic estimates quantities such as \\(V^\\pi(s)\\), \\(Q^\\pi(s,a)\\), or \\(A^\\pi(s,a)=Q^\\pi(s,a)-V^\\pi(s)\\).",
        isCorrect: true,
      },
      {
        text: "The actor is trained by minimizing \\(\\frac{1}{2}\\sum_i\\lVert\\hat V_\\phi(s_i)-y_i\\rVert^2\\) with respect to \\(\\phi\\).",
        isCorrect: false,
      },
      {
        text: "The critic samples actions from \\(\\pi_\\theta\\) and directly executes them in the environment.",
        isCorrect: false,
      },
    ],
    explanation:
      "The actor chooses actions through the policy, while the critic evaluates states or actions. The two components interact through advantage or Q estimates, but their roles and parameter updates are distinct.",
  },

  {
    id: "cs224r-lect4-q64",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "At a state \\(s\\), suppose \\(V^\\pi(s)=3\\), \\(Q^\\pi(s,a_1)=2\\), and \\(Q^\\pi(s,a_2)=5\\). Which statements correctly interpret the advantages?",
    options: [
      {
        text: "\\(A^\\pi(s,a_1)=-1\\), so an actor-gradient update should reduce the likelihood of \\(a_1\\).",
        isCorrect: true,
      },
      {
        text: "\\(A^\\pi(s,a_2)=2\\), so an actor-gradient update should increase the likelihood of \\(a_2\\).",
        isCorrect: true,
      },
      {
        text: "The value baseline \\(V^\\pi(s)=3\\) is what makes the sign of the update depend on relative action quality, not just raw return scale.",
        isCorrect: true,
      },
      {
        text: "\\(a_1\\) should be reinforced because its Q-value is positive.",
        isCorrect: false,
      },
    ],
    explanation:
      "Advantage compares each action against the state baseline, so a positive Q-value can still be below average for that state. Actor-critic reinforces actions with positive advantage and suppresses actions with negative advantage.",
  },

  {
    id: "cs224r-lect4-q65",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe labels in bootstrapped critic training?",
    options: [
      {
        text: "The label can depend on the current critic through a term such as \\(\\hat V_\\phi(s_{t+1})\\).",
        isCorrect: true,
      },
      {
        text: "Because \\(\\phi\\) changes, labels like \\(y_t=r_t+\\hat V_\\phi(s_{t+1})\\) may be updated on every gradient step.",
        isCorrect: true,
      },
      {
        text: "Bootstrapped labels such as \\(r_t+\\hat V_\\phi(s_{t+1})\\) can propagate information backward without waiting for \\(\\sum_{t'=t}^{T}r_{t'}\\).",
        isCorrect: true,
      },
      {
        text: "A practical initialization strategy can start with Monte Carlo targets \\(\\sum_{t'=t}^{T}r_{t'}\\) before relying more heavily on bootstrapping.",
        isCorrect: true,
      },
    ],
    explanation:
      "Bootstrapping is powerful because the critic can use its own current estimate as supervision for nearby earlier states. The price is that the target is moving, so implementation choices around initialization and label updates matter.",
  },

  {
    id: "cs224r-lect4-q66",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the critic regression objective used in the lecture?",
    options: [
      {
        text: "A typical critic loss is \\(\\mathcal{L}(\\phi)=\\frac{1}{2}\\sum_i\\lVert\\hat V^\\pi_\\phi(s_i)-y_i\\rVert^2\\).",
        isCorrect: true,
      },
      {
        text: "The critic can take multiple gradient steps on this regression loss without directly changing the policy parameters.",
        isCorrect: true,
      },
      {
        text: "The critic loss is \\(\\sum_i\\log\\pi_\\theta(a_i\\mid s_i)y_i\\), so it is optimized with respect to actor parameters.",
        isCorrect: false,
      },
      {
        text: "The regression target \\(y_i\\) must always be the immediate reward only.",
        isCorrect: false,
      },
    ],
    explanation:
      "The critic is trained as a supervised regression problem from states or state-action pairs to return targets. The target can be Monte Carlo, bootstrapped, or n-step; it is not limited to immediate reward.",
  },

  {
    id: "cs224r-lect4-q67",
    chapter: 4,
    difficulty: "hard",
    prompt: `A stochastic policy has three actions at state \\(s\\):

| Action | \\(\\pi(a\\mid s)\\) | \\(Q^\\pi(s,a)\\) |
| --- | ---: | ---: |
| \\(a_1\\) | 0.2 | 1 |
| \\(a_2\\) | 0.5 | 0 |
| \\(a_3\\) | 0.3 | 4 |

Which option correctly computes \\(V^\\pi(s)\\)?`,
    options: [
      {
        text: "\\(V^\\pi(s)=0.2\\cdot1+0.5\\cdot0+0.3\\cdot4=1.4\\).",
        isCorrect: true,
      },
      {
        text: "\\(V^\\pi(s)=\\frac{1+0+4}{3}=1.67\\), because action probabilities are ignored.",
        isCorrect: false,
      },
      {
        text: "\\(V^\\pi(s)=4\\), because the value uses the best action's Q-value.",
        isCorrect: false,
      },
      {
        text: "\\(V^\\pi(s)=0\\), because the most likely action has Q-value 0.",
        isCorrect: false,
      },
    ],
    explanation:
      "The value function averages Q-values under the policy's action probabilities. It is neither the best action value, the unweighted average, nor just the value of the most likely action.",
  },

  {
    id: "cs224r-lect4-q68",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe policy evaluation in the actor-critic lecture?",
    options: [
      {
        text: "Policy evaluation means estimating how good states or actions are for a fixed policy.",
        isCorrect: true,
      },
      {
        text: "Monte Carlo estimation, bootstrapping, and n-step returns are three value-estimation approaches discussed for policy evaluation.",
        isCorrect: true,
      },
      {
        text: "Policy evaluation can supply the critic signal used by an actor-critic policy update.",
        isCorrect: true,
      },
      {
        text: "Policy evaluation requires knowing the environment transition model exactly.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture uses policy evaluation to mean learning value estimates from sampled data. It does not require an explicit dynamics model; the examples fit values from rollouts and bootstrapped targets.",
  },

  {
    id: "cs224r-lect4-q69",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the modified-MDP interpretation of discounting shown on the slides?",
    options: [
      {
        text: "Ordinary transition probabilities can be viewed as scaled to \\(\\tilde p(s'\\mid s,a)=\\gamma p(s'\\mid s,a)\\).",
        isCorrect: true,
      },
      {
        text: "The remaining probability \\(1-\\gamma\\) can be viewed as going to a zero-reward absorbing terminal state.",
        isCorrect: true,
      },
      {
        text: "The modified-MDP view requires scaling ordinary transitions by \\(1-\\gamma\\) and terminal transitions by \\(\\gamma\\).",
        isCorrect: false,
      },
      {
        text: "The modified-MDP view applies only to the actor network and not to value targets.",
        isCorrect: false,
      },
    ],
    explanation:
      "The slide's interpretation of discounting is a stochastic termination model: continue according to the original dynamics with probability mass \\(\\gamma\\), and terminate with probability mass \\(1-\\gamma\\). This is another way to understand why future rewards are geometrically downweighted.",
  },

  {
    id: "cs224r-lect4-q71",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "A policy always chooses to sit on the beach, and the only final reward is 1 if the person can play the drums in a month. Sitting on the beach or watching TV once gives no chance of success, while practicing once gives success probability \\(p>0\\) before the person follows the old policy. Which interpretation is correct?",
    options: [
      {
        text: "\\(V^\\pi(s)=0\\), the beach and TV actions have zero advantage, and practicing has positive advantage \\(p\\).",
        isCorrect: true,
      },
      {
        text: "\\(V^\\pi(s)=p\\), because value averages over all actions that could be taken at the state.",
        isCorrect: false,
      },
      {
        text: "Practicing has zero advantage because the policy would not normally choose it.",
        isCorrect: false,
      },
      {
        text: "The advantage of sitting on the beach is positive because it matches the current policy exactly.",
        isCorrect: false,
      },
    ],
    explanation:
      "The state value follows the current policy, so a policy that always sits on the beach has expected reward zero in this setup. A Q-value can ask what happens if a different first action is forced, so practicing once can have positive advantage even though the current policy would not choose it.",
  },

  {
    id: "cs224r-lect4-q72",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly explain why actor-critic methods use advantage estimates instead of raw reward-to-go samples in the actor update?",
    options: [
      {
        text: "\\(Q^\\pi(s,a)\\) estimates expected future return after starting in \\(s\\), taking \\(a\\), and then following \\(\\pi\\), rather than using only one sampled future.",
        isCorrect: true,
      },
      {
        text: "Subtracting \\(V^\\pi(s)\\) makes the actor compare an action against the policy's average action quality at the same state.",
        isCorrect: true,
      },
      {
        text: "A better advantage estimate can lower policy-gradient noise while still targeting expected return.",
        isCorrect: true,
      },
      {
        text: "Using an advantage estimate changes the actor's objective from expected return to supervised imitation of the critic.",
        isCorrect: false,
      },
    ],
    explanation:
      "The critic is meant to produce a less noisy signal about whether a sampled action was better or worse than the policy's usual behavior at that state. The actor still optimizes expected return through a policy-gradient update; the critic changes the scalar credit-assignment signal, not the objective into imitation.",
  },

  {
    id: "cs224r-lect4-q73",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the first actor-critic walkthrough's choice to fit a value function \\(\\hat V^\\pi_\\phi\\)?",
    options: [
      {
        text: "A value network can be trained with supervised regression from visited states to return-like targets.",
        isCorrect: true,
      },
      {
        text: "Once \\(\\hat V^\\pi_\\phi\\) is available, an advantage estimate can use \\(r_t+\\gamma\\hat V^\\pi_\\phi(s_{t+1})-\\hat V^\\pi_\\phi(s_t)\\).",
        isCorrect: true,
      },
      {
        text: "The critic can take multiple regression steps on the collected data without those critic steps directly changing actor parameters.",
        isCorrect: true,
      },
      {
        text: "The fitted value function is a practical estimate of policy evaluation, not an assumption that the true \\(V^\\pi\\) is already known.",
        isCorrect: true,
      },
    ],
    explanation:
      "The walkthrough inserts a policy-evaluation stage between data collection and actor improvement. The value network is learned from rollout-derived targets, then reused to form advantage estimates for the actor, while the critic's own regression steps update \\(\\phi\\) rather than \\(\\theta\\).",
  },

  {
    id: "cs224r-lect4-q74",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe Monte Carlo value-function estimation from sampled trajectories?",
    options: [
      {
        text: "Each visited state can become a supervised data point whose target is the observed future return from that trajectory.",
        isCorrect: true,
      },
      {
        text: "The dataset size scales with the number of sampled trajectories times the number of visited time steps.",
        isCorrect: true,
      },
      {
        text: "Monte Carlo targets avoid using the current value estimate as a bootstrap label.",
        isCorrect: true,
      },
      {
        text: "Monte Carlo estimation requires resetting the environment many times from exactly the same state before any value target can be formed.",
        isCorrect: false,
      },
    ],
    explanation:
      "Monte Carlo estimation uses complete observed returns as labels, so it can train from ordinary rollout data without separately restarting from the exact same state. The tradeoff is that one trajectory's future can be a noisy label for the policy's expected future return.",
  },

  {
    id: "cs224r-lect4-q75",
    chapter: 4,
    difficulty: "hard",
    prompt: `A value critic is trained with
\\[
\\mathcal{L}(\\phi)=\\frac{1}{2}\\sum_i\\lVert \\hat V_\\phi(s_i)-y_i\\rVert^2.
\\]
For two states, \\(\\hat V_\\phi(s_1)=1.5\\), \\(y_1=2\\), \\(\\hat V_\\phi(s_2)=-0.5\\), and \\(y_2=-1\\). What is the loss?`,
    options: [
      {
        text: "\\(0.25\\), because \\(\\frac{1}{2}[(1.5-2)^2+(-0.5+1)^2]=0.25\\).",
        isCorrect: true,
      },
      {
        text: "\\(0.5\\), because the squared errors are added without the \\(\\frac{1}{2}\\) factor.",
        isCorrect: false,
      },
      {
        text: "\\(1.0\\), because the absolute errors are added before squaring once.",
        isCorrect: false,
      },
      {
        text: "\\(-0.25\\), because the second target is negative and makes the squared-error term negative.",
        isCorrect: false,
      },
    ],
    explanation:
      "Squared-error regression squares each prediction error, so negative targets do not create negative loss terms. The two errors are \\(-0.5\\) and \\(0.5\\), each squared to \\(0.25\\), and the objective's one-half factor turns their sum into \\(0.25\\).",
  },

  {
    id: "cs224r-lect4-q76",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements correctly compare Monte Carlo, bootstrapped, and n-step value targets?",
    options: [
      {
        text: "Monte Carlo targets use the rollout's observed summed future rewards.",
        isCorrect: true,
      },
      {
        text: "One-step bootstrapped targets use an immediate reward plus the current estimate of the next state's value.",
        isCorrect: true,
      },
      {
        text: "N-step targets use several observed rewards before bootstrapping from a later value estimate.",
        isCorrect: true,
      },
      {
        text: "All three can serve as supervised labels for fitting a critic, with different bias-variance tradeoffs.",
        isCorrect: true,
      },
    ],
    explanation:
      "These targets differ in how much observed future reward they use before relying on the current value estimate. Monte Carlo uses the longest observed future and has high variance, one-step bootstrapping relies heavily on the critic, and n-step returns interpolate between them.",
  },

  {
    id: "cs224r-lect4-q77",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe discounted n-step returns for critic training?",
    options: [
      {
        text: "A discounted n-step target can be written \\(y_t=\\sum_{k=0}^{n-1}\\gamma^k r_{t+k}+\\gamma^n\\hat V(s_{t+n})\\).",
        isCorrect: true,
      },
      {
        text: "Setting \\(n=1\\) gives the one-step temporal-difference target \\(r_t+\\gamma\\hat V(s_{t+1})\\).",
        isCorrect: true,
      },
      {
        text: "Increasing \\(n\\) uses more observed rewards before the bootstrap term, which can reduce bootstrap bias while often increasing variance.",
        isCorrect: true,
      },
      {
        text: "Increasing \\(n\\) changes the actor objective into greedily maximizing \\(\\max_a Q(s,a)\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The n-step choice changes the critic target, not the underlying policy-gradient objective. Looking farther ahead uses more sampled rewards and less immediate bootstrapping, which can be useful when the extra sampled rewards are informative enough to justify their added variance.",
  },

  {
    id: "cs224r-lect4-q78",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements correctly distinguish the parameter roles in a basic actor-critic method?",
    options: [
      {
        text: "\\(\\theta\\) parameterizes the actor policy \\(\\pi_\\theta(a\\mid s)\\), whose log probability is differentiated in the actor update.",
        isCorrect: true,
      },
      {
        text: "\\(\\phi\\) parameterizes the critic \\(\\hat V_\\phi\\), which is trained by regression to return-like targets.",
        isCorrect: true,
      },
      {
        text: "\\(\\theta\\) parameterizes the critic loss, while \\(\\phi\\) parameterizes the action distribution used in the environment.",
        isCorrect: false,
      },
      {
        text: "The same gradient step must update \\(\\theta\\) and \\(\\phi\\) with the identical loss for the method to be actor-critic.",
        isCorrect: false,
      },
    ],
    explanation:
      "Actor-critic methods couple two learned objects but do not collapse them into one update. The critic learns values with its own parameters and loss, while the actor uses critic-derived advantages to update the policy parameters.",
  },

  {
    id: "cs224r-lect4-q79",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which steps belong in the on-policy actor-critic loop before the next batch is collected?",
    options: [
      {
        text: "Run the current policy to collect a batch of trajectories.",
        isCorrect: true,
      },
      {
        text: "Fit a value estimate to Monte Carlo, bootstrapped, or n-step targets from that batch.",
        isCorrect: true,
      },
      {
        text: "Compute advantage estimates for the sampled state-action pairs using the fitted value function.",
        isCorrect: true,
      },
      {
        text: "Apply an advantage-weighted policy-gradient update to the actor parameters.",
        isCorrect: true,
      },
    ],
    explanation:
      "The on-policy walkthrough collects data with the current policy, evaluates that policy with a critic, and then improves the actor using advantages from the same batch. After that actor step, the loop collects a fresh batch from the updated policy.",
  },

  {
    id: "cs224r-lect4-q80",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "An infinite-horizon process gives reward 1 at every step and uses discount \\(\\gamma=0.97\\). Under the stochastic-termination interpretation of discounting, which statement is correct?",
    options: [
      {
        text: "The discounted value is \\(\\frac{1}{1-0.97}\\approx33.33\\), and the equivalent zero-reward termination probability per step is \\(0.03\\).",
        isCorrect: true,
      },
      {
        text: "The discounted value is \\(0.97\\), and the equivalent zero-reward termination probability per step is \\(0.97\\).",
        isCorrect: false,
      },
      {
        text: "The discounted value diverges because any \\(\\gamma<1\\) still leaves infinitely many rewards unscaled.",
        isCorrect: false,
      },
      {
        text: "The discounted value is \\(\\frac{1}{0.97}\\), and the equivalent zero-reward termination probability per step is \\(1.97\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "A constant reward stream discounted by \\(\\gamma\\) forms the geometric series \\(1+\\gamma+\\gamma^2+\\cdots=1/(1-\\gamma)\\). The modified-MDP view interprets the missing probability mass as a \\(1-\\gamma\\) chance of entering a zero-reward absorbing state at each step.",
  },
];
