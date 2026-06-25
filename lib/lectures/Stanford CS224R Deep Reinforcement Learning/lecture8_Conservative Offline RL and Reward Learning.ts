import { Question } from "../../quiz";

export const cs224rLecture8RewardLearningQuestions: Question[] = [
  {
    id: "cs224r-lect8-q01",
    chapter: 8,
    difficulty: "easy",
    prompt:
      "Which statements capture why specifying a reward function is a central problem rather than a minor implementation detail?",
    options: [
      {
        text: "In many real tasks, the intended behavior is easier to recognize after seeing an outcome than to encode as a scalar reward before training.",
        isCorrect: true,
      },
      {
        text: "A proxy reward can select behavior that scores well under the proxy while missing the task designer's actual intent.",
        isCorrect: true,
      },
      {
        text: "Learning from goals, demonstrations, or preferences can be treated as part of task specification, not only as data collection for a fixed reward.",
        isCorrect: true,
      },
      {
        text: "Even when the RL algorithm is unchanged, the reward source determines which behavior the policy is optimized to produce.",
        isCorrect: true,
      },
    ],
    explanation:
      "The main issue is that rewards are the task interface: optimizing the wrong signal can make the learned behavior wrong even if the optimizer works. Goals, demonstrations, and preferences are different ways to supply task information when a hand-written reward is unavailable or unreliable.",
  },

  {
    id: "cs224r-lect8-q02",
    chapter: 8,
    difficulty: "easy",
    prompt:
      "An offline RL critic is fit with a Bellman-style target using a fixed dataset \\(\\mathcal{D}=\\{(s,a,r,s')\\}\\). Which statements identify the distribution-shift problem that motivates conservative value learning?",
    options: [
      {
        text: "The dataset constrains values most directly on actions that were actually logged by the behavior policy.",
        isCorrect: true,
      },
      {
        text: "A learned policy may query \\(Q(s,a)\\) or \\(Q(s',a')\\) at actions that are poorly covered by \\(\\mathcal{D}\\).",
        isCorrect: true,
      },
      {
        text: "Overestimated values for unsupported actions can be amplified by policy improvement or maximization.",
        isCorrect: true,
      },
      {
        text: "With observed rewards, the Bellman target reduces to \\(Q(s,a)=r(s,a)\\), so value extrapolation stops being a concern.",
        isCorrect: false,
      },
    ],
    explanation:
      "Observed rewards do not remove the need to estimate future value, and bootstrapping can still evaluate actions outside the data support. Conservative methods address the fact that high learned values on poorly covered actions are dangerous in a fixed-dataset setting.",
  },

  {
    id: "cs224r-lect8-q03",
    chapter: 8,
    difficulty: "medium",
    prompt:
      "Consider the simplified conservative Q-learning objective\n\\[\\min_Q \\max_\\mu\\; \\mathbb{E}_{\\mathcal{D}}\\left[(Q(s,a)-\\{r(s,a)+\\gamma\\,\\mathbb{E}_{a'\\sim\\pi(\\cdot\\mid s')}Q(s',a')\\})^2\\right] + \\alpha\\,\\mathbb{E}_{s\\sim\\mathcal{D},a\\sim\\mu(\\cdot\\mid s)}[Q(s,a)].\\]\nWhich interpretations of the added conservative term are valid?",
    options: [
      {
        text: "The inner maximization over \\(\\mu\\) searches for actions whose large \\(Q\\)-values would make the penalty large.",
        isCorrect: true,
      },
      {
        text: "The outer minimization pushes down values on actions selected by \\(\\mu\\), especially actions that look too good without data support.",
        isCorrect: true,
      },
      {
        text: "The term is a behavior-cloning loss because it directly maximizes the likelihood of logged actions under \\(\\pi\\).",
        isCorrect: false,
      },
      {
        text: "The term removes bootstrapping by replacing the Bellman target with a one-step supervised reward target.",
        isCorrect: false,
      },
    ],
    explanation:
      "The conservative term is a value penalty, not an imitation likelihood and not a replacement for bootstrapped Bellman fitting. The adversarial \\(\\mu\\) identifies actions with high learned values, and minimizing the objective makes those high values costly.",
  },

  {
    id: "cs224r-lect8-q04",
    chapter: 8,
    difficulty: "medium",
    prompt:
      "For the simplified conservative objective that adds \\(\\alpha\\,\\mathbb{E}_{s\\sim\\mathcal{D},a\\sim\\mu}[Q(s,a)]\\) without a dataset-action correction term, what is the intended large-\\(\\alpha\\) consequence?",
    options: [
      {
        text: "The learned critic is deliberately biased downward, so \\(\\hat Q^\\pi(s,a)\\) can become a lower bound on \\(Q^\\pi(s,a)\\) under the stated conditions.",
        isCorrect: true,
      },
      {
        text: "The learned critic becomes optimistic on unseen actions so that the policy can discover trajectories missing from the dataset.",
        isCorrect: false,
      },
      {
        text: "The learned critic exactly matches \\(Q^\\pi\\) pointwise because the conservative term cancels Bellman approximation error.",
        isCorrect: false,
      },
      {
        text: "The learned critic becomes independent of \\(\\alpha\\), because the maximization over \\(\\mu\\) absorbs the penalty scale.",
        isCorrect: false,
      },
    ],
    explanation:
      "The simple CQL construction is intentionally pessimistic: it pushes down values so unsupported high estimates are less attractive. It is not an exactness guarantee and it is not an exploration bonus for unseen actions.",
  },

  {
    id: "cs224r-lect8-q05",
    chapter: 8,
    difficulty: "hard",
    prompt:
      "A CQL variant adds a dataset-action correction, giving a penalty contrast of the form\n\\[\\alpha\\,\\mathbb{E}_{s\\sim\\mathcal{D},a\\sim\\mu(\\cdot\\mid s)}[Q(s,a)] - \\alpha\\,\\mathbb{E}_{(s,a)\\sim\\mathcal{D}}[Q(s,a)].\\]\nWhich consequences follow from adding the second term?",
    options: [
      {
        text: "The first term still penalizes high values on actions proposed by \\(\\mu\\).",
        isCorrect: true,
      },
      {
        text: "The second term pushes upward on values for actions actually observed in the dataset.",
        isCorrect: true,
      },
      {
        text: "The guarantee becomes an expected conservative bound under the policy action distribution at dataset states, rather than a pointwise lower bound for every action.",
        isCorrect: true,
      },
      {
        text: "The contrast ranks each logged action above each unlogged action at the same state by construction, so \\(Q(s,a_{\\text{logged}})>Q(s,a_{\\text{unlogged}})\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The dataset-action term prevents the method from indiscriminately depressing actions that the data actually supports. The resulting guarantee is about an expectation under the policy at dataset states; it does not sort every logged and unlogged action pair by value.",
  },

  {
    id: "cs224r-lect8-q06",
    chapter: 8,
    difficulty: "hard",
    prompt:
      "In entropy-regularized CQL, suppose the inner objective uses \\(R(\\mu)=\\mathbb{E}_{s\\sim\\mathcal{D}}[H(\\mu(\\cdot\\mid s))]\\). Which statements explain why a log-sum-exp term appears?",
    options: [
      {
        text: "For each state, maximizing \\(\\mathbb{E}_{a\\sim\\mu}[Q(s,a)] + H(\\mu(\\cdot\\mid s))\\) yields \\(\\mu(a\\mid s) \\propto \\exp(Q(s,a))\\).",
        isCorrect: true,
      },
      {
        text: "Substituting the optimal \\(\\mu\\) turns the maximized action-value term into \\(\\log\\sum_a \\exp(Q(s,a))\\) for a discrete action set.",
        isCorrect: true,
      },
      {
        text: "The log-sum-exp appears because the Bellman backup uses the uniform average \\(\\frac{1}{|\\mathcal{A}|}\\sum_a Q(s,a)\\) whenever the policy is unknown.",
        isCorrect: false,
      },
      {
        text: "Entropy regularization removes the dependence on \\(Q\\), so the conservative penalty becomes a statewise constant \\(\\log|\\mathcal{A}|\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The log-sum-exp is the convex conjugate associated with maximizing a linear score plus entropy over a categorical distribution. It is not a uniform-action Bellman backup; it still depends strongly on the relative \\(Q\\)-values at the state.",
  },

  {
    id: "cs224r-lect8-q07",
    chapter: 8,
    difficulty: "hard",
    prompt: `At one dataset state, a discrete-action CQL penalty uses \\(\\log\\sum_a\\exp(Q(s,a)) - \\mathbb{E}_{a\\sim\\mathcal{D}(\\cdot\\mid s)}[Q(s,a)]\\). The critic currently has:

| action | \\(Q(s,a)\\) | logged count |
| --- | ---: | ---: |
| \\(a_1\\) | 0 | 10 |
| \\(a_2\\) | 2 | 0 |

Using \\(e^0=1\\) and \\(e^2\\approx 7.39\\), which value is the approximate contrast at this state?`,
    options: [
      {
        text: "\\(\\log(1+7.39)-0 \\approx 2.13\\)",
        isCorrect: true,
      },
      {
        text: "\\(\\log(1+7.39)-2 \\approx 0.13\\)",
        isCorrect: false,
      },
      {
        text: "\\((0+2)/2-0=1\\)",
        isCorrect: false,
      },
      {
        text: "\\(\\max(0,2)-2=0\\)",
        isCorrect: false,
      },
    ],
    explanation:
      "The dataset expectation uses the logged action distribution, so only \\(a_1\\) contributes to the subtracted term in this table. The log-sum-exp still sees both actions, making the unlogged high-value action increase the conservative penalty.",
  },

  {
    id: "cs224r-lect8-q08",
    chapter: 8,
    difficulty: "medium",
    prompt:
      "Why can a large-scale recommender system or similar production decision system be a natural setting for offline RL and conservative value learning?",
    options: [
      {
        text: "It may already have large logged datasets from earlier recommendation policies.",
        isCorrect: true,
      },
      {
        text: "Online exploration with poor intermediate policies can be costly because bad recommendations affect real users.",
        isCorrect: true,
      },
      {
        text: "The learned policy can be evaluated against logged rewards or downstream metrics before broad deployment.",
        isCorrect: true,
      },
      {
        text: "The fixed-data setting covers the action choices the new policy will consider at each user state, so support mismatch is negligible.",
        isCorrect: false,
      },
    ],
    explanation:
      "Logged interaction data and risk-sensitive deployment make offline RL attractive, but they do not eliminate distribution shift. Conservative value estimation matters precisely because the new policy may prefer actions or recommendations that were rare under the logging policy.",
  },

  {
    id: "cs224r-lect8-q09",
    chapter: 8,
    difficulty: "easy",
    prompt:
      "Which examples illustrate that the reward signal often comes from outside the RL algorithm rather than from the algorithm itself?",
    options: [
      {
        text: "A game environment gives points or win/loss outcomes that can be used as reward.",
        isCorrect: true,
      },
      {
        text: "A robot task uses a learned classifier over goal states to provide a reward-like signal.",
        isCorrect: true,
      },
      {
        text: "An assistant model uses human pairwise preferences over responses to train a reward model.",
        isCorrect: true,
      },
      {
        text: "A value-iteration update automatically defines the human's intended objective without any external task signal.",
        isCorrect: false,
      },
    ],
    explanation:
      "RL algorithms optimize whatever reward or reward surrogate they are given. Game scores, classifiers, and preference models are different sources of that signal; Bellman updates alone do not reveal the desired task.",
  },

  {
    id: "cs224r-lect8-q10",
    chapter: 8,
    difficulty: "medium",
    prompt:
      "A robot is shown successful demonstrations of placing a block in a bin, but the final controller should recover from off-demo starting states. Which limitations of direct imitation motivate learning a reward or goal condition instead?",
    options: [
      {
        text: "Imitation can copy the demonstrated action distribution without learning what outcome makes the behavior successful.",
        isCorrect: true,
      },
      {
        text: "A reward or goal signal can let RL search for actions that achieve the desired outcome from states not covered by the demonstrations.",
        isCorrect: true,
      },
      {
        text: "Successful demonstrations identify the hidden goal well enough for an action imitator to recover from arbitrary off-demo states.",
        isCorrect: false,
      },
      {
        text: "A policy that matches demonstration actions on average will tend to reach the same final states even from shifted states.",
        isCorrect: false,
      },
    ],
    explanation:
      "Demonstrations contain behavior, but they may not identify which parts of the behavior are essential for success in new states. A learned reward or goal classifier can expose the task outcome to an RL optimizer, while pure action matching can fail under distribution shift.",
  },

  {
    id: "cs224r-lect8-q11",
    chapter: 8,
    difficulty: "medium",
    prompt:
      "A binary goal classifier is trained from states labeled by whether they are in a goal set \\(\\mathcal{G}\\). Which design choices match this reward-learning approach?",
    options: [
      {
        text: "Use states as classifier inputs rather than requiring a hand-coded dense reward at every transition.",
        isCorrect: true,
      },
      {
        text: "Assign labels like \\(\\mathbf{1}(s_i\\in\\mathcal{G})\\) to distinguish successful goal states from other states.",
        isCorrect: true,
      },
      {
        text: "Use the classifier score \\(f_\\phi(s)\\) or probability \\(P_\\phi(y=1\\mid s)\\) as a reward-like signal for an RL policy.",
        isCorrect: true,
      },
      {
        text: "Keep successful and unsuccessful examples in \\(\\mathcal{D}_+\\) and \\(\\mathcal{D}_-\\) because the classifier needs both positive and negative evidence.",
        isCorrect: true,
      },
    ],
    explanation:
      "The classifier turns task specification into supervised learning over states, then the policy optimizer uses the classifier score as a reward surrogate. Both positive and negative examples are needed; otherwise the model cannot learn a decision boundary for the goal set.",
  },

  {
    id: "cs224r-lect8-q12",
    chapter: 8,
    difficulty: "medium",
    prompt: `A goal classifier is trained with label \\(y=\\mathbf{1}(s\\in\\mathcal{G})\\). Which row has the correct label assignment?

| state description | in \\(\\mathcal{G}\\)? | proposed \\(y\\) |
| --- | --- | ---: |
| block inside bin | yes | 1 |
| block balanced on rim | no | 1 |
| block on table | no | 0 |`,
    options: [
      {
        text: "The first and third labels are correct, but the rim state is incorrectly labeled positive.",
        isCorrect: true,
      },
      {
        text: "The rim state and bin state should both use \\(y=1\\), because near-goal states receive the same binary label as completed goals.",
        isCorrect: false,
      },
      {
        text: "The table state is the sole correct row because successful states are assigned \\(y=0\\) when training a classifier-based reward.",
        isCorrect: false,
      },
      {
        text: "The bin state should be labeled 0 unless it was reached by the current policy rather than a demonstration.",
        isCorrect: false,
      },
    ],
    explanation:
      "The binary label is about membership in the goal set, not about being visually close to a goal or about which policy generated the state. A near-success state can be useful training data, but if it is outside \\(\\mathcal{G}\\) it should not be labeled as a completed goal.",
  },

  {
    id: "cs224r-lect8-q13",
    chapter: 8,
    difficulty: "medium",
    prompt:
      "A policy trained against a learned goal classifier starts producing states that the classifier scores highly, but human inspection shows those states are not actual successes. Which diagnoses are consistent with classifier exploitation?",
    options: [
      {
        text: "The policy is searching in parts of state space where the classifier has weak negative evidence.",
        isCorrect: true,
      },
      {
        text: "The classifier reward is being optimized, so classifier mistakes can become policy incentives.",
        isCorrect: true,
      },
      {
        text: "Adding visited failure states as new negatives can directly target the exploited region.",
        isCorrect: true,
      },
      {
        text: "The failure rules out classifier rewards as a practical path, even when additional policy data can be aggregated.",
        isCorrect: false,
      },
    ],
    explanation:
      "Classifier exploitation is an adversarial data-coverage problem: the policy finds states that the classifier has not learned to reject. Iteratively adding those visited states to the negative set can repair the reward model in the regions the policy actually visits.",
  },

  {
    id: "cs224r-lect8-q14",
    chapter: 8,
    difficulty: "hard",
    prompt:
      "An iterative goal-classifier method maintains positive examples \\(\\mathcal{D}_+\\) and negative examples \\(\\mathcal{D}_-\\). Which steps belong to the loop that reduces reward-model exploitation?",
    options: [
      {
        text: "Train or update the classifier using positives \\(\\mathcal{D}_+\\) and negatives \\(\\mathcal{D}_-\\), often balancing the batches.",
        isCorrect: true,
      },
      {
        text: "Collect states visited by the current policy after it is trained against the classifier reward.",
        isCorrect: true,
      },
      {
        text: "Add visited policy states to \\(\\mathcal{D}_-\\) so the classifier learns to reject exploited non-goals.",
        isCorrect: true,
      },
      {
        text: "Update the policy \\(\\pi\\) using the current classifier score \\(f_\\phi(s)\\) as the reward signal.",
        isCorrect: true,
      },
    ],
    explanation:
      "The method alternates between classifier fitting, policy optimization, and data aggregation from the policy's own visited states. Adding visited states as negatives makes the reward model harder to exploit because future policies are trained against a classifier with evidence from earlier failure modes.",
  },

  {
    id: "cs224r-lect8-q15",
    chapter: 8,
    difficulty: "hard",
    prompt:
      "Suppose a policy occasionally reaches true goal states during the same process that adds visited states to \\(\\mathcal{D}_-\\). Why can balancing positive and negative batches keep true goals from becoming uselessly low reward?",
    options: [
      {
        text: "If a true goal state appears in both the positive and negative pools, balanced sampling can make the empirical class evidence for that state no worse than tied rather than purely negative.",
        isCorrect: true,
      },
      {
        text: "The classifier can still output at least around \\(0.5\\) for states that are represented as successful under balanced positive evidence.",
        isCorrect: true,
      },
      {
        text: "Balancing works by deleting visited states that were successful before the classifier update.",
        isCorrect: false,
      },
      {
        text: "Balancing makes the classifier ignore \\(\\mathcal{D}_-\\), so adding visited states has no effect on the decision boundary.",
        isCorrect: false,
      },
    ],
    explanation:
      "The point is not to ignore negatives, but to prevent the negative dataset from overwhelming positive evidence simply because every visited state is added. With balanced updates, true goal states can retain non-low classifier scores while exploited non-goal states are learned as negatives.",
  },

  {
    id: "cs224r-lect8-q16",
    chapter: 8,
    difficulty: "hard",
    prompt: `A tabular classifier estimates \\(P(y=1\\mid s)\\) from balanced batches. For a particular state \\(s^*\\), the balanced training stream contains 20 positive occurrences and 20 negative occurrences. Which estimate is most consistent with that evidence before adding function-approximation or smoothing assumptions?`,
    options: [
      {
        text: "\\(P(y=1\\mid s^*)=0.5\\)",
        isCorrect: true,
      },
      {
        text: "\\(P(y=1\\mid s^*)=0\\), because any occurrence in \\(\\mathcal{D}_-\\) overrides positive labels.",
        isCorrect: false,
      },
      {
        text: "\\(P(y=1\\mid s^*)=1\\), because any occurrence in \\(\\mathcal{D}_+\\) overrides negative labels.",
        isCorrect: false,
      },
      {
        text: "\\(P(y=1\\mid s^*)=20\\), because the classifier reward is the count of positive examples.",
        isCorrect: false,
      },
    ],
    explanation:
      "With balanced contradictory labels in a simple empirical estimate, the posterior class frequency is tied at one half. That is why balanced batches can keep genuine successes from being crushed to zero while still allowing the algorithm to add visited states as negative evidence.",
  },

  {
    id: "cs224r-lect8-q17",
    chapter: 8,
    difficulty: "medium",
    prompt:
      "Which practical details help a classifier-based reward method remain usable when the policy is actively trying to optimize the classifier score?",
    options: [
      {
        text: "Use regularization or adversarial-training tricks so the classifier does not overfit narrow artifacts of the current datasets.",
        isCorrect: true,
      },
      {
        text: "Keep adding examples from states visited by the policy so the classifier sees the regions it is being asked to score.",
        isCorrect: true,
      },
      {
        text: "Use final states from demonstrations or known successes as positive examples when those states represent desired outcomes.",
        isCorrect: true,
      },
      {
        text: "Freeze the initial classifier permanently so later policy updates are measured against a fixed reward snapshot.",
        isCorrect: false,
      },
    ],
    explanation:
      "A classifier reward is under pressure from the policy, so data coverage and regularization matter. Freezing a weak initial classifier can preserve exploitable mistakes, while updating with visited states targets the distribution induced by the policy.",
  },

  {
    id: "cs224r-lect8-q18",
    chapter: 8,
    difficulty: "medium",
    prompt:
      "How does the goal-classifier loop relate to the generative adversarial network (GAN) analogy?",
    options: [
      {
        text: "A discriminator-like classifier is trained to distinguish desired examples from generated or visited examples.",
        isCorrect: true,
      },
      {
        text: "A generator-like policy is trained to produce states that the classifier judges as desired.",
        isCorrect: true,
      },
      {
        text: "At a successful equilibrium, generated states can match the desired data distribution well enough that the classifier cannot exploit a simple distinction.",
        isCorrect: true,
      },
      {
        text: "The analogy explains why adversarial reward learning can be unstable and benefit from regularization tricks.",
        isCorrect: true,
      },
    ],
    explanation:
      "The classifier-policy loop mirrors discriminator-generator training: one model judges examples, and the other model optimizes against that judgment. This analogy also explains both the promise and the instability of adversarial reward learning.",
  },

  {
    id: "cs224r-lect8-q19",
    chapter: 8,
    difficulty: "medium",
    prompt:
      "Compared with hand-writing a dense reward, what are the main tradeoffs of learning rewards from goals or demonstrations?",
    options: [
      {
        text: "It gives a practical way to specify tasks through examples of desired behavior or outcomes.",
        isCorrect: true,
      },
      {
        text: "It can require adversarial or iterative training that is less stable than ordinary supervised fitting.",
        isCorrect: true,
      },
      {
        text: "It avoids needing any examples of success because pairwise preferences alone define the goal set.",
        isCorrect: false,
      },
      {
        text: "A learned reward model closes the reward-hacking issue because the signal came from data rather than code.",
        isCorrect: false,
      },
    ],
    explanation:
      "Goal and demonstration based reward learning can make task specification practical, but it still needs examples of what desired behavior or outcomes look like. A learned reward can be exploited or unstable, so it is not a magic fix for reward hacking.",
  },

  {
    id: "cs224r-lect8-q20",
    chapter: 8,
    difficulty: "easy",
    prompt:
      "Why are pairwise human preferences a useful supervision format for learning rewards?",
    options: [
      {
        text: "A person can often say which of two rollouts or responses is better without assigning an absolute scalar reward.",
        isCorrect: true,
      },
      {
        text: "Preference labels can compare full trajectories or partial trajectory segments.",
        isCorrect: true,
      },
      {
        text: "Pairwise data can train a reward model by making preferred items receive higher total reward than less preferred items.",
        isCorrect: true,
      },
      {
        text: "Pairwise data removes the need to optimize a policy after the reward model is trained.",
        isCorrect: false,
      },
    ],
    explanation:
      "Preferences are useful because relative judgments are often easier than calibrated numeric rewards. They still usually feed into a reward-learning and policy-optimization pipeline; preference data alone is not the final policy.",
  },

  {
    id: "cs224r-lect8-q21",
    chapter: 8,
    difficulty: "hard",
    prompt:
      "If a human labels trajectory \\(\\tau_w\\) as better than trajectory \\(\\tau_l\\), which constraints or modeling choices match the preference-reward setup?",
    options: [
      {
        text: "The learned reward should tend to satisfy \\(\\sum_{(s,a)\\in\\tau_w} r_\\theta(s,a) > \\sum_{(s,a)\\in\\tau_l} r_\\theta(s,a)\\).",
        isCorrect: true,
      },
      {
        text: "Writing \\(r_\\theta(\\tau)\\) as the summed reward over a trajectory is a convenient shorthand for preference modeling.",
        isCorrect: true,
      },
      {
        text: "The less-preferred trajectory should receive the higher total reward because the model is trained on mistakes to avoid.",
        isCorrect: false,
      },
      {
        text: "The comparison is valid when \\(\\tau_w\\) and \\(\\tau_l\\) have identical state-action sequences and differ in reward labels.",
        isCorrect: false,
      },
    ],
    explanation:
      "Preference learning converts the label \\(\\tau_w \\succ \\tau_l\\) into pressure for the preferred trajectory to have a larger total learned reward. The two trajectories need not be identical; the whole point is to compare different behaviors or responses.",
  },

  {
    id: "cs224r-lect8-q22",
    chapter: 8,
    difficulty: "hard",
    prompt: `A reward model assigns per-step rewards to two trajectory segments:

| segment | rewards along segment |
| --- | --- |
| \\(\\tau_a\\) | \\(1, 0, 2\\) |
| \\(\\tau_b\\) | \\(0, 1, 1\\) |

Under the shorthand \\(r_\\theta(\\tau)=\\sum_{(s,a)\\in\\tau} r_\\theta(s,a)\\), which comparison is correct?`,
    options: [
      {
        text: "\\(r_\\theta(\\tau_a)=3\\), \\(r_\\theta(\\tau_b)=2\\), so the model assigns a higher total reward to \\(\\tau_a\\).",
        isCorrect: true,
      },
      {
        text: "\\(r_\\theta(\\tau_a)=1\\), \\(r_\\theta(\\tau_b)=0\\), because the segment score uses the first reward in each segment.",
        isCorrect: false,
      },
      {
        text: "\\(r_\\theta(\\tau_a)=2\\), \\(r_\\theta(\\tau_b)=1\\), because the segment score uses the last reward in each segment.",
        isCorrect: false,
      },
      {
        text: "\\(r_\\theta(\\tau_a)=r_\\theta(\\tau_b)\\), because both segments have three time steps.",
        isCorrect: false,
      },
    ],
    explanation:
      "The trajectory shorthand sums the learned rewards across the segment, so \\(1+0+2=3\\) and \\(0+1+1=2\\). Equal segment length does not imply equal learned return, and the comparison is not based only on the first or last step.",
  },

  {
    id: "cs224r-lect8-q23",
    chapter: 8,
    difficulty: "hard",
    prompt:
      "A preference model estimates \\(P(\\tau_a \\succ \\tau_b)=\\sigma(r_\\theta(\\tau_a)-r_\\theta(\\tau_b))\\), where \\(\\sigma(z)=1/(1+e^{-z})\\). Which statements are correct?",
    options: [
      {
        text: "If \\(r_\\theta(\\tau_a)=r_\\theta(\\tau_b)\\), the model assigns probability \\(0.5\\) to \\(\\tau_a\\succ\\tau_b\\).",
        isCorrect: true,
      },
      {
        text: "Increasing \\(r_\\theta(\\tau_a)-r_\\theta(\\tau_b)\\) increases the estimated probability that \\(\\tau_a\\) is preferred.",
        isCorrect: true,
      },
      {
        text: "The log-likelihood for an observed label \\(\\tau_w\\succ\\tau_l\\) is \\(\\log\\sigma(r_\\theta(\\tau_w)-r_\\theta(\\tau_l))\\).",
        isCorrect: true,
      },
      {
        text: "The model depends on the product \\(r_\\theta(\\tau_a)r_\\theta(\\tau_b)\\), so a shared constant changes the preference probability for each comparison.",
        isCorrect: false,
      },
    ],
    explanation:
      "The Bradley-Terry-style probability depends on a reward difference, so equal totals produce a one-half preference probability and larger positive gaps favor \\(\\tau_a\\). A shared additive shift cancels in the difference, while the observed-label likelihood uses the preferred-minus-less-preferred gap.",
  },

  {
    id: "cs224r-lect8-q24",
    chapter: 8,
    difficulty: "hard",
    prompt: `For a labeled preference \\(\\tau_w\\succ\\tau_l\\), a reward model has \\(r_\\theta(\\tau_w)-r_\\theta(\\tau_l)=\\log 4\\). Since \\(e^{\\log 4}=4\\), which quantities are correct?`,
    options: [
      {
        text: "\\(\\sigma(\\log 4)=\\frac{4}{1+4}=0.8\\).",
        isCorrect: true,
      },
      {
        text: "The contribution to the maximized log-likelihood is \\(\\log(0.8)\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\sigma(\\log 4)=\\frac{1}{1+4}=0.2\\), because the preferred trajectory is in the numerator of \\(e^{-z}\\).",
        isCorrect: false,
      },
      {
        text: "The log-likelihood contribution is \\(\\log 4\\), because the sigmoid is skipped after taking reward differences.",
        isCorrect: false,
      },
    ],
    explanation:
      "For \\(z=\\log 4\\), the sigmoid is \\(1/(1+e^{-z})=1/(1+1/4)=4/5\\). The likelihood for the observed preference is the log of that probability, not the raw reward gap.",
  },

  {
    id: "cs224r-lect8-q25",
    chapter: 8,
    difficulty: "medium",
    prompt:
      "A reward-learning batch asks a human to rank \\(k=5\\) trajectories generated for the same prompt or task context. If all pairwise comparisons from that ranked batch are used, how many ordered winner-loser pairs are available?",
    options: [
      {
        text: "\\(\\binom{5}{2}=10\\)",
        isCorrect: true,
      },
      {
        text: "\\(5\\), because the ranking supplies one comparison per trajectory against a reference item.",
        isCorrect: false,
      },
      {
        text: "\\(2^5=32\\), because each trajectory independently wins or loses.",
        isCorrect: false,
      },
      {
        text: "\\(5!=120\\), because the rank order itself supplies a separate label for each permutation.",
        isCorrect: false,
      },
    ],
    explanation:
      "A total ranking over five trajectories determines one preferred item for each unordered pair, so the number of pairwise labels is \\(5\\cdot4/2=10\\). The reward model uses pair comparisons, not only adjacent pairs and not every possible permutation as a separate label.",
  },

  {
    id: "cs224r-lect8-q26",
    chapter: 8,
    difficulty: "hard",
    prompt:
      "Which steps belong to a complete preference-based reward-learning procedure for trajectory data?",
    options: [
      {
        text: "Sample batches \\(\\{\\tau_i\\}_{i=1}^k\\) and obtain human rankings or pairwise preferences over them.",
        isCorrect: true,
      },
      {
        text: "Compute \\(r_\\theta(\\tau_i)\\) for each trajectory in a ranked batch.",
        isCorrect: true,
      },
      {
        text: "For each preferred-less-preferred pair, update \\(\\theta\\) to increase \\(\\log\\sigma(r_\\theta(\\tau_w)-r_\\theta(\\tau_l))\\).",
        isCorrect: true,
      },
      {
        text: "Use the learned reward model \\(r_\\theta\\) as a signal for policy optimization, potentially in an online RL loop.",
        isCorrect: true,
      },
    ],
    explanation:
      "Preference reward learning first converts rankings into pairwise likelihood terms for the reward model. Once trained, the reward model can be used as the reward signal for improving a policy, and the process can be interleaved with data collection.",
  },

  {
    id: "cs224r-lect8-q27",
    chapter: 8,
    difficulty: "hard",
    prompt:
      "Why might preference supervision need to remain in the loop while RL improves the policy?",
    options: [
      {
        text: "As the policy changes, it may produce new trajectories that the current reward model was not trained to compare reliably.",
        isCorrect: true,
      },
      {
        text: "Additional comparisons can correct reward-model mistakes or coverage gaps revealed by optimized policies.",
        isCorrect: true,
      },
      {
        text: "A single initial preference dataset defines the reward reliably for future policy distributions.",
        isCorrect: false,
      },
      {
        text: "Preference labels are needed because policy-gradient updates operate on pairwise comparisons instead of scalar rewards.",
        isCorrect: false,
      },
    ],
    explanation:
      "Optimizing a learned reward model can shift the policy into regions where the reward model is weak, just as classifier rewards can be exploited. New preference labels can refresh the reward model on the evolving distribution; policy-gradient methods can use scalar rewards, but the problem is reward reliability.",
  },

  {
    id: "cs224r-lect8-q28",
    chapter: 8,
    difficulty: "medium",
    prompt:
      "In reward learning for language-model responses, what role does a reward model \\(r(x,y)\\) play?",
    options: [
      {
        text: "It scores how good a response \\(y\\) is for a prompt \\(x\\).",
        isCorrect: true,
      },
      {
        text: "It can be trained from preferences such as one response \\(y\\) being judged better than another response \\(y'\\) for the same prompt.",
        isCorrect: true,
      },
      {
        text: "It supplies a scalar objective that a later RL step can optimize when fine-tuning the language model.",
        isCorrect: true,
      },
      {
        text: "It replaces the language model decoder, so no response generation model is needed after reward training.",
        isCorrect: false,
      },
    ],
    explanation:
      "The reward model is a judge, not the generator itself. It maps a prompt-response pair to a scalar score, and the language model is then fine-tuned to produce responses with higher reward.",
  },

  {
    id: "cs224r-lect8-q29",
    chapter: 8,
    difficulty: "easy",
    prompt:
      "Which stages are part of the standard RLHF-style pipeline for language models?",
    options: [
      {
        text: "Large-scale pretraining by next-token prediction on mixed-quality data.",
        isCorrect: true,
      },
      {
        text: "Supervised fine-tuning on higher-quality prompt-response examples.",
        isCorrect: true,
      },
      {
        text: "Collecting preference data and training a reward model over responses.",
        isCorrect: true,
      },
      {
        text: "Reinforcement learning to fine-tune the model toward high reward under the learned reward model.",
        isCorrect: true,
      },
    ],
    explanation:
      "RLHF for language models is a pipeline: pretraining creates a broad model, supervised fine-tuning improves instruction-following behavior, and preference data trains a reward model. The final RL stage uses that reward model to push generation toward preferred responses.",
  },

  {
    id: "cs224r-lect8-q30",
    chapter: 8,
    difficulty: "medium",
    prompt:
      "What is the key supervision substitution in reinforcement learning with AI feedback (RLAIF)?",
    options: [
      {
        text: "Use another language model to critique or compare responses, such as asking which response is less harmful.",
        isCorrect: true,
      },
      {
        text: "Use next-token likelihood on pretraining data as the reward model in place of preference feedback.",
        isCorrect: false,
      },
      {
        text: "Use the environment transition model as the preference labeler instead of any judge model.",
        isCorrect: false,
      },
      {
        text: "Replace pairwise preferences with demonstrations of target responses for a representative prompt set.",
        isCorrect: false,
      },
    ],
    explanation:
      "RLAIF keeps the broad preference-feedback idea but uses an AI model as the source of critique or comparison. The motivation is that judging or critiquing can be easier than generating an ideal answer, but the pipeline still needs a reward or preference signal.",
  },

  {
    id: "cs224r-lect8-q31",
    chapter: 8,
    difficulty: "medium",
    prompt:
      "Which comparisons between goal/demonstration-based reward learning and human-preference reward learning are accurate?",
    options: [
      {
        text: "Goals and demonstrations provide a practical task-specification framework when examples of desired outcomes or behavior are available.",
        isCorrect: true,
      },
      {
        text: "Goal or demonstration methods may involve adversarial training that is unstable without careful regularization.",
        isCorrect: true,
      },
      {
        text: "Human preferences can be easier to provide pairwise and do not require example goals or demonstrations.",
        isCorrect: true,
      },
      {
        text: "Preference-based methods can require human supervision during the RL loop, often increasing the human-time cost.",
        isCorrect: true,
      },
    ],
    explanation:
      "The two families trade off what kind of supervision is easiest to obtain. Examples of success make goal-based methods attractive but can lead to adversarial training issues, while pairwise preferences scale well conceptually but can require repeated supervision as policies improve.",
  },

  {
    id: "cs224r-lect8-q32",
    chapter: 8,
    difficulty: "hard",
    prompt:
      "A preference model uses \\(P(\\tau_a\\succ\\tau_b)=\\sigma(r_\\theta(\\tau_a)-r_\\theta(\\tau_b))\\). What happens to every predicted preference probability if a constant \\(c\\) is added to all trajectory rewards, so \\(r'_\\theta(\\tau)=r_\\theta(\\tau)+c\\)?",
    options: [
      {
        text: "The probabilities are unchanged because \\((r_\\theta(\\tau_a)+c)-(r_\\theta(\\tau_b)+c)=r_\\theta(\\tau_a)-r_\\theta(\\tau_b)\\).",
        isCorrect: true,
      },
      {
        text: "The probability becomes \\(\\sigma(r_\\theta(\\tau_a)+r_\\theta(\\tau_b)+2c)\\), so larger absolute rewards increase the comparison score.",
        isCorrect: false,
      },
      {
        text: "The probability becomes \\(0.5\\), because \\((r_\\theta(\\tau)+c)\\) erases the reward ordering inside the sigmoid.",
        isCorrect: false,
      },
      {
        text: "The preferred trajectory changes for positive \\(c\\), because \\(P(\\tau_a\\succ\\tau_b)\\) depends on reward sums rather than reward differences.",
        isCorrect: false,
      },
    ],
    explanation:
      "The pairwise model is invariant to a shared additive shift because it only sees differences between trajectory rewards. This is a useful mathematical property of preference modeling and also shows why absolute reward scale and offset are not directly identified by pairwise labels alone.",
  },

  {
    id: "cs224r-lect8-q33",
    chapter: 8,
    difficulty: "medium",
    prompt:
      "When preference labels compare partial rollouts instead of complete episodes, which statements remain true?",
    options: [
      {
        text: "The reward model can sum learned rewards over the compared segment to form \\(r_\\theta(\\tau)\\).",
        isCorrect: true,
      },
      {
        text: "The label still means \\(r_\\theta(\\tau_w)>r_\\theta(\\tau_l)\\) for the preferred segment and less-preferred segment.",
        isCorrect: true,
      },
      {
        text: "Partial-rollout comparisons can reduce the amount of trajectory context a human must inspect for each label.",
        isCorrect: true,
      },
      {
        text: "Partial rollouts are incompatible with \\(\\sigma(r_\\theta(\\tau_w)-r_\\theta(\\tau_l))\\) because the formula expects full-episode returns.",
        isCorrect: false,
      },
    ],
    explanation:
      "The preference formula applies to whatever trajectory segment is being compared, as long as the reward model defines a total score for that segment. Full episodes are not required by the math, although partial segments may omit context that matters for some tasks.",
  },

  {
    id: "cs224r-lect8-q34",
    chapter: 8,
    difficulty: "hard",
    prompt:
      "In an asymmetric self-play setup for agents proposing their own goals, which roles make the setup different from ordinary external-reward training?",
    options: [
      {
        text: "A goal-setter can create a task or endpoint internally, and a goal-reacher then tries to reproduce or reach it.",
        isCorrect: true,
      },
      {
        text: "Self-play episodes can use internal rewards without direct external supervision for each proposed goal.",
        isCorrect: true,
      },
      {
        text: "The setup needs a human preference label for each self-play trajectory pair before either agent can update.",
        isCorrect: false,
      },
      {
        text: "The target-task episode is a logging step rather than a place where the self-play skill can transfer.",
        isCorrect: false,
      },
    ],
    explanation:
      "Asymmetric self-play lets agents generate a curriculum by setting and reaching goals internally. External reward can still matter for target tasks, but the self-play phase is specifically about unsupervised or intrinsically motivated goal proposal.",
  },

  {
    id: "cs224r-lect8-q35",
    chapter: 8,
    difficulty: "hard",
    prompt:
      "A team has no hand-written reward. It has a few images of desired final robot states, a stream of states visited by a current policy, and separate human preferences over short policy rollouts. Which method-source pairings are technically coherent?",
    options: [
      {
        text: "Use desired final-state images as \\(\\mathcal{D}_+\\), visited non-goal states as \\(\\mathcal{D}_-\\), and train a classifier score \\(f_\\phi(s)\\) whose output becomes a reward surrogate.",
        isCorrect: true,
      },
      {
        text: "Use preference labels \\(\\tau_w\\succ\\tau_l\\) to train \\(r_\\theta\\) with \\(\\log\\sigma(r_\\theta(\\tau_w)-r_\\theta(\\tau_l))\\).",
        isCorrect: true,
      },
      {
        text: "Interleave reward-model updates with policy optimization if the policy starts exploiting either the classifier or learned preference reward.",
        isCorrect: true,
      },
      {
        text: "Apply CQL's \\(\\log\\sum_a \\exp Q(s,a)\\) penalty directly to unlabeled goal images, which by itself produces \\(P(\\tau_w\\succ\\tau_l)\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Goal classifiers and preference reward models use different supervision but can both turn non-scalar task information into reward-like signals. CQL is an offline value-learning method for conservative critic estimation; its log-sum-exp penalty is not a replacement for labels defining goals or preferences.",
  },
];
