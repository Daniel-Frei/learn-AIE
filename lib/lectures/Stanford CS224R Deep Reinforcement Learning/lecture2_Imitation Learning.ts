import { Question } from "../../quiz";

export const cs224rLecture2ImitationLearningQuestions: Question[] = [
  // ============================================================
  // Lecture 2 – Imitation Learning
  // Q1–Q35
  // ============================================================

  // ============================================================
  // Introductory imitation-learning questions
  // ============================================================

  {
    id: "cs224r-lect2-q36",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "An autonomous-driving dataset contains trajectories from human drivers, but the human policy itself is unknown. Which statements correctly describe the imitation-learning setup?",
    options: [
      {
        text: "The demonstrations can be treated as samples from an unknown expert policy.",
        isCorrect: true,
      },
      {
        text: "The goal is to learn a policy that performs similarly to the demonstrators by mimicking their actions.",
        isCorrect: true,
      },
      {
        text: "Behavior cloning can be trained without hand-writing a reward function.",
        isCorrect: true,
      },
      {
        text: "Demonstration quality can bound the performance of the learned imitator.",
        isCorrect: true,
      },
    ],
    explanation:
      "Imitation learning uses demonstrated state-action behavior as supervision when the expert's internal policy is unknown. The usual target is expert-level behavior through mimicry without needing a hand-written reward, so the quality and coverage of the demonstrations matter.",
  },

  {
    id: "cs224r-lect2-q37",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "A deterministic behavior-cloning policy is trained with squared error on expert state-action pairs. Which statements correctly describe this version-0 algorithm?",
    options: [
      {
        text: "It treats imitation as supervised prediction of expert actions from states.",
        isCorrect: true,
      },
      {
        text: "After training, the learned policy can be deployed without collecting new online data during training.",
        isCorrect: true,
      },
      {
        text: "The squared-error objective can struggle when the demonstrated action distribution is multimodal.",
        isCorrect: true,
      },
      {
        text: "DAgger-style expert intervention is a separate online correction strategy, not part of basic offline behavior cloning.",
        isCorrect: true,
      },
    ],
    explanation:
      "Basic behavior cloning trains on the fixed demonstration dataset, much like supervised regression or classification. Its simplicity is also a limitation: squared error predicts point actions and does not by itself solve multimodality; online interventions belong to later correction methods such as DAgger.",
  },

  {
    id: "cs224r-lect2-q38",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "A demonstration dataset contains several trajectories from different human drivers. Which statements are appropriate assumptions for imitation learning?",
    options: [
      {
        text: "The data consists of state-action behavior sampled from one or more demonstrators.",
        isCorrect: true,
      },
      {
        text: "Different demonstrators may choose different valid actions in similar states.",
        isCorrect: true,
      },
      {
        text: "Demonstrations may be good enough to imitate even when they are not globally optimal.",
        isCorrect: true,
      },
      {
        text: "The learned policy may visit different states from the expert if its own mistakes compound.",
        isCorrect: true,
      },
    ],
    explanation:
      "Demonstrations provide the state-action examples that define the training distribution, and multiple humans can create real diversity in that data. The lecture does not require global optimality, and it later emphasizes that deployment can shift the state distribution when the learned policy makes mistakes.",
  },

  {
    id: "cs224r-lect2-q39",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "In a driving dataset, some people go straight while others merge left from a similar state. Which statements correctly explain the multimodality problem?",
    options: [
      {
        text: "The action distribution can have separate high-probability modes for different valid strategies.",
        isCorrect: true,
      },
      {
        text: "Averaging the modes can produce a low-probability action such as drifting between lanes.",
        isCorrect: true,
      },
      {
        text: "Adding more drivers can make multiple strategies more visible rather than collapsing them into one mode.",
        isCorrect: true,
      },
      {
        text: "The mean action need not correspond to any demonstrated strategy when modes are separated.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture's driving example shows why a mean action can be a bad action when several valid strategies create separated modes. More human data can make diversity more visible rather than removing it, so the policy distribution must be able to represent that structure.",
  },

  {
    id: "cs224r-lect2-q40",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "A large neural network is still trained with an L2 action loss on a multimodal continuous-action dataset. Which statements correctly describe the limitation?",
    options: [
      {
        text: "The loss still encourages predicting the conditional mean action.",
        isCorrect: true,
      },
      {
        text: "Increasing network size does not by itself make a point-output policy represent several action modes.",
        isCorrect: true,
      },
      {
        text: "A wider single Gaussian may still put probability mass on bad in-between actions.",
        isCorrect: true,
      },
      {
        text: "Changing the output distribution can matter separately from changing network size.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture separates neural-network expressivity from distribution expressivity. A bigger function approximator can predict the mean more accurately, but a point-output or single simple distribution still cannot faithfully represent separated modes in the action distribution.",
  },

  {
    id: "cs224r-lect2-q41",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "A policy network outputs parameters of an action distribution instead of one deterministic action. Which statements correctly describe what distribution expressivity adds?",
    options: [
      {
        text: "The distribution class limits which action distributions the policy can represent.",
        isCorrect: true,
      },
      {
        text: "Expressive distributions can put probability mass on several plausible expert strategies.",
        isCorrect: true,
      },
      {
        text: "A single Gaussian remains limited when continuous actions have separated modes.",
        isCorrect: true,
      },
      {
        text: "Distribution choice remains important even when the neural network mapping into parameters is deep.",
        isCorrect: true,
      },
    ],
    explanation:
      "The network can be very expressive as a function from state to parameters while the chosen output distribution remains restrictive. Expressive policy distributions reduce mean-collapse failures by allowing the policy to represent multiple high-probability actions.",
  },

  {
    id: "cs224r-lect2-q42",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Version-1 imitation learning minimizes \\(-\\mathbb{E}_{(s,a)\\sim\\mathcal{D}}[\\log \\pi_\\theta(a \\mid s)]\\). Which statements correctly interpret this objective?",
    options: [
      {
        text: "It increases the probability that the policy assigns to demonstrated actions in their states.",
        isCorrect: true,
      },
      {
        text: "It trains a conditional action distribution rather than only a deterministic point prediction.",
        isCorrect: true,
      },
      {
        text: "It directly optimizes the expected reward collected by rolling out the current policy.",
        isCorrect: false,
      },
      {
        text: "It avoids the need to condition action probabilities on the current state or observation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Negative log-likelihood fits the policy distribution to expert actions observed in the dataset. It is still imitation learning on demonstrations, not direct reward optimization through online rollouts, and the action probabilities remain conditioned on the state or observation.",
  },

  {
    id: "cs224r-lect2-q43",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A high-dimensional action is modeled autoregressively as conditional factors over action dimensions. Which statements correctly describe this policy distribution?",
    options: [
      {
        text: "The joint action distribution is decomposed into ordered conditional predictions.",
        isCorrect: true,
      },
      {
        text: "At inference time, sampled earlier dimensions can condition later dimensions.",
        isCorrect: true,
      },
      {
        text: "Each action dimension is predicted independently of the dimensions generated before it.",
        isCorrect: false,
      },
      {
        text: "The method only applies when actions are one-dimensional and continuous.",
        isCorrect: false,
      },
    ],
    explanation:
      "Autoregressive modeling handles a joint action by predicting dimensions sequentially, with later predictions conditioned on earlier generated values. That conditioning is the point of the factorization, so treating dimensions as independent or limiting the method to one-dimensional actions misses the mechanism.",
  },

  {
    id: "cs224r-lect2-q44",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A diffusion policy is used as the action distribution for continuous robot actions. Which statements correctly describe why this model class is useful and what it costs?",
    options: [
      {
        text: "It can represent complex continuous distributions through an iterative denoising process.",
        isCorrect: true,
      },
      {
        text: "Its sampling procedure usually requires multiple refinement steps rather than one direct output.",
        isCorrect: true,
      },
      {
        text: "It gets its expressivity by forcing continuous actions into a single categorical bin.",
        isCorrect: false,
      },
      {
        text: "It is equivalent to a single Gaussian policy with a wider variance parameter.",
        isCorrect: false,
      },
    ],
    explanation:
      "Diffusion policies trade inference speed for a much richer continuous action distribution than a single Gaussian. The iterative denoising process is different from discretizing into one bin or merely increasing Gaussian variance, both of which miss the source's expressivity point.",
  },

  // ============================================================
  // Q10–Q18: EXACTLY THREE TRUE
  // ============================================================

  {
    id: "cs224r-lect2-q10",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about demonstrations are correct?",
    options: [
      { text: "They may come from multiple experts.", isCorrect: true },
      {
        text: "They define the training distribution for imitation learning.",
        isCorrect: true,
      },
      { text: "They always correspond to optimal behavior.", isCorrect: false },
      {
        text: "They may include inconsistent action choices.",
        isCorrect: true,
      },
    ],
    explanation:
      "Demonstrations are the behavior data that imitation learning tries to match, so they define the state-action distribution the learner sees during training. They can come from several experts and may contain inconsistent action choices because different humans or systems can solve the same task differently. They should not be treated as automatically optimal; imitation learning can copy the quality, diversity, and flaws present in the demonstration set.",
  },

  {
    id: "cs224r-lect2-q11",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about Gaussian mixture models (GMMs) are correct?",
    options: [
      { text: "They combine multiple Gaussian components.", isCorrect: true },
      {
        text: "They are more expressive than a single Gaussian.",
        isCorrect: true,
      },
      {
        text: "They require choosing the number of components.",
        isCorrect: true,
      },
      {
        text: "They are maximally expressive for continuous actions.",
        isCorrect: false,
      },
    ],
    explanation:
      "A Gaussian mixture model represents an action distribution as several Gaussian components, which lets it capture multiple modes better than a single Gaussian. The modeler still has to choose or learn how many components are available, and that finite mixture limits what shapes can be represented. It is therefore more expressive than one Gaussian but not maximally expressive for arbitrary continuous-action behavior.",
  },

  {
    id: "cs224r-lect2-q12",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about discretization in imitation learning are correct?",
    options: [
      {
        text: "Continuous actions can be binned into discrete categories.",
        isCorrect: true,
      },
      {
        text: "Discretization enables categorical cross-entropy loss.",
        isCorrect: true,
      },
      { text: "Finer discretization increases expressivity.", isCorrect: true },
      {
        text: "Discretization preserves exact distance information between actions.",
        isCorrect: false,
      },
    ],
    explanation:
      "Discretization turns continuous actions into bins so the learner can use categorical modeling and cross-entropy-style objectives. Finer bins can represent more detailed actions, but binning also discards exact metric relationships inside and between bins. That is why discretization can increase categorical expressivity while still failing to preserve exact continuous distance information.",
  },

  {
    id: "cs224r-lect2-q13",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statements about offline imitation learning are correct?",
    options: [
      {
        text: "It trains using a fixed dataset of demonstrations.",
        isCorrect: true,
      },
      {
        text: "It does not require running the learned policy during training.",
        isCorrect: true,
      },
      {
        text: "It avoids safety risks from untrained policies.",
        isCorrect: true,
      },
      {
        text: "It guarantees robustness to compounding errors.",
        isCorrect: false,
      },
    ],
    explanation:
      "Offline imitation learning trains from a fixed demonstration dataset, so it can avoid running an unsafe or untrained policy during training. That makes data collection and safety constraints easier than online correction methods. It does not guarantee robustness at deployment, because the learned policy can still drift into states that were rare or absent in the demonstrations and then suffer compounding errors.",
  },

  {
    id: "cs224r-lect2-q14",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about compounding errors are correct?",
    options: [
      {
        text: "They arise when policy mistakes alter future states.",
        isCorrect: true,
      },
      {
        text: "They cause the policy to visit unseen states.",
        isCorrect: true,
      },
      {
        text: "They create a mismatch between training and deployment distributions.",
        isCorrect: true,
      },
      { text: "They only occur in stochastic environments.", isCorrect: false },
    ],
    explanation:
      "Compounding errors occur because a wrong action can move the system into a different future state, where the policy may be even less prepared to act. This creates a mismatch between the expert-state distribution used for training and the states visited by the learned policy at deployment. Randomness can make the problem harder, but the core failure can happen even in deterministic systems because actions affect later states.",
  },

  {
    id: "cs224r-lect2-q15",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statements about DAgger (Dataset Aggregation) are correct?",
    options: [
      {
        text: "It collects data by running the learned policy.",
        isCorrect: true,
      },
      { text: "It queries the expert at visited states.", isCorrect: true },
      {
        text: "It aggregates corrective data with original demonstrations.",
        isCorrect: true,
      },
      { text: "It is a purely offline algorithm.", isCorrect: false },
    ],
    explanation:
      "Dataset Aggregation (DAgger) addresses compounding errors by running the current learned policy and asking the expert what should have been done in the states the policy actually visits. Those corrective labels are aggregated with earlier demonstrations so the training set better matches deployment states. Because it depends on rollouts and expert queries during learning, it is not a purely offline algorithm.",
  },

  {
    id: "cs224r-lect2-q16",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about categorical policies for discrete actions are correct?",
    options: [
      {
        text: "They output probabilities over discrete actions.",
        isCorrect: true,
      },
      {
        text: "They are maximally expressive for discrete action spaces.",
        isCorrect: true,
      },
      {
        text: "They are trained using classification losses.",
        isCorrect: true,
      },
      {
        text: "They require continuous action representations.",
        isCorrect: false,
      },
    ],
    explanation:
      "For a discrete action space, a categorical policy can assign a probability to each action, so it can represent any distribution over that finite set. That output form naturally fits classification losses such as cross-entropy when learning from expert action labels. It does not require continuous action representations; continuous actions are the case where extra modeling choices such as Gaussians, mixtures, discretization, or diffusion become important.",
  },

  {
    id: "cs224r-lect2-q17",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statements about policy expressivity are correct?",
    options: [
      {
        text: "Distribution expressivity differs from network expressivity.",
        isCorrect: true,
      },
      {
        text: "A large network with L2 loss still predicts a mean.",
        isCorrect: true,
      },
      {
        text: "Expressive distributions enable safer action sampling.",
        isCorrect: true,
      },
      {
        text: "Distribution choice is irrelevant if the network is deep enough.",
        isCorrect: false,
      },
    ],
    explanation:
      "Network expressivity and output-distribution expressivity solve different problems. A large neural network trained with an L2 objective can still collapse multimodal expert behavior toward an average action, which may be unsafe or unrealistic. Choosing a richer policy distribution lets the model sample plausible high-probability actions, so the distribution choice remains important even when the neural network is deep.",
  },

  {
    id: "cs224r-lect2-q18",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about training autoregressive policies are correct?",
    options: [
      {
        text: "Training uses expert actions as conditioning inputs.",
        isCorrect: true,
      },
      {
        text: "Cross-entropy loss is applied per action dimension.",
        isCorrect: true,
      },
      {
        text: "Inference feeds sampled actions back into the model.",
        isCorrect: true,
      },
      {
        text: "Backpropagation requires differentiating through sampled actions.",
        isCorrect: false,
      },
    ],
    explanation:
      "Autoregressive policy training can use teacher forcing, where each action dimension is conditioned on the expert's previous action dimensions and trained with cross-entropy. At inference time, the model samples earlier dimensions and feeds those sampled choices forward to generate later dimensions. Backpropagation does not require differentiating through the sampled discrete actions; the training loss differentiates through the predicted probabilities for the expert actions.",
  },

  // ============================================================
  // Q19–Q27: EXACTLY TWO TRUE
  // ============================================================

  {
    id: "cs224r-lect2-q19",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about imitation learning objectives are correct?",
    options: [
      { text: "They maximize expected reward directly.", isCorrect: false },
      { text: "They maximize likelihood of expert actions.", isCorrect: true },
      { text: "They require a known reward function.", isCorrect: false },
      { text: "They depend on demonstration quality.", isCorrect: true },
    ],
    explanation:
      "Behavior cloning-style imitation learning fits the expert action data, often by maximizing the likelihood of expert actions under the policy. It does not directly maximize expected reward and does not need a known reward function, which is why it can be attractive when reward design is hard. The tradeoff is that performance depends strongly on the quality, coverage, and consistency of the demonstrations.",
  },

  {
    id: "cs224r-lect2-q20",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about discretized autoregressive policies are correct?",
    options: [
      { text: "They convert continuous actions into bins.", isCorrect: true },
      { text: "They preserve exact geometric distances.", isCorrect: false },
      {
        text: "They reduce modeling complexity of joint distributions.",
        isCorrect: true,
      },
      { text: "They eliminate the need for sampling.", isCorrect: false },
    ],
    explanation:
      "A discretized autoregressive policy bins continuous action dimensions and then models the joint action one conditional distribution at a time. That factorization avoids representing every joint action combination independently, which reduces modeling complexity. Binning does not preserve exact geometric distances, and autoregressive generation still requires choosing or sampling action bins at inference time.",
  },

  {
    id: "cs224r-lect2-q21",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statements about covariate shift are correct?",
    options: [
      {
        text: "It arises when policy actions affect future states.",
        isCorrect: true,
      },
      {
        text: "It causes mismatch between expert and policy state distributions.",
        isCorrect: true,
      },
      {
        text: "It is irrelevant when demonstrations are large.",
        isCorrect: false,
      },
      {
        text: "It can be ignored in sequential decision problems.",
        isCorrect: false,
      },
    ],
    explanation:
      "Covariate shift appears in imitation learning when the learned policy's actions push the system into states that differ from the expert demonstrations. A larger dataset can reduce gaps in coverage, but it does not remove the sequential feedback loop that creates new state distributions at deployment. Because actions affect future states, this mismatch is central rather than something sequential decision problems can ignore.",
  },

  {
    id: "cs224r-lect2-q22",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about human-gated DAgger are correct?",
    options: [
      { text: "The expert intervenes when the policy fails.", isCorrect: true },
      {
        text: "Corrections are collected without policy rollouts.",
        isCorrect: false,
      },
      {
        text: "It is more practical than querying every visited state.",
        isCorrect: true,
      },
      {
        text: "It eliminates the need for expert involvement.",
        isCorrect: false,
      },
    ],
    explanation:
      "Human-gated DAgger reduces annotation cost by having the expert intervene or correct the policy when failures matter, rather than labeling every visited state. It is still built around policy rollouts, because the goal is to collect corrections on states the learned policy actually reaches. The method therefore remains an expert-in-the-loop approach and does not eliminate expert involvement.",
  },

  {
    id: "cs224r-lect2-q23",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about diffusion policies are correct?",
    options: [
      { text: "They operate by iterative refinement.", isCorrect: true },
      {
        text: "They generate actions in a single forward pass.",
        isCorrect: false,
      },
      { text: "They require discretization of actions.", isCorrect: false },
      {
        text: "They handle multimodal continuous actions well.",
        isCorrect: true,
      },
    ],
    explanation:
      "Diffusion policies generate actions through iterative denoising or refinement, which is slower than a single forward pass but much more expressive. That expressivity is useful for multimodal continuous-action data, where several distinct expert actions may be plausible. They do not require discretizing the action space; the point is to model rich continuous distributions directly.",
  },

  {
    id: "cs224r-lect2-q24",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statements about expressive policies are correct?",
    options: [
      { text: "They reduce risk of unsafe averaged actions.", isCorrect: true },
      { text: "They guarantee optimal performance.", isCorrect: false },
      { text: "They are unnecessary for multimodal data.", isCorrect: false },
      {
        text: "They enable sampling from high-probability regions.",
        isCorrect: true,
      },
    ],
    explanation:
      "Expressive policies help when demonstration data has multiple valid strategies because the policy can sample from realistic high-probability regions instead of averaging incompatible actions. That can reduce unsafe mean-action behavior in tasks where the average of two good actions is not itself good. Expressivity is not a guarantee of optimal performance, and it is especially relevant rather than unnecessary for multimodal data.",
  },

  {
    id: "cs224r-lect2-q25",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about Gaussian policies are correct?",
    options: [
      { text: "They are unimodal distributions.", isCorrect: true },
      {
        text: "They can represent arbitrary action distributions.",
        isCorrect: false,
      },
      {
        text: "They are limited for multimodal imitation data.",
        isCorrect: true,
      },
      { text: "They eliminate compounding errors.", isCorrect: false },
    ],
    explanation:
      "A single Gaussian policy is simple and often convenient, but it is unimodal, so it tends to represent one central region of action space. Multimodal imitation data may require choosing between several distinct strategies, which a single Gaussian can collapse into an unrealistic average. Gaussian output assumptions also do not solve compounding errors, because those come from sequential distribution shift during deployment.",
  },

  {
    id: "cs224r-lect2-q26",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statements about expert optimality are correct?",
    options: [
      {
        text: "Imitation learning assumes expert behavior is optimal.",
        isCorrect: false,
      },
      { text: "Experts may follow different strategies.", isCorrect: true },
      {
        text: "Imitation learning can outperform experts by default.",
        isCorrect: false,
      },
      {
        text: "Imitation learning aims to match expert performance.",
        isCorrect: true,
      },
    ],
    explanation:
      "Imitation learning treats expert behavior as the target data distribution, but that does not mean every expert action is optimal. Different experts may use different strategies, and the learner is usually trying to match the demonstrated performance rather than discover a better policy by default. Exceeding the expert generally requires additional optimization signal or exploration beyond simply cloning the demonstrations.",
  },

  {
    id: "cs224r-lect2-q27",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about policy deployment are correct?",
    options: [
      { text: "Deployment may expose unseen states.", isCorrect: true },
      {
        text: "Training and deployment distributions are identical.",
        isCorrect: false,
      },
      { text: "Errors can accumulate over time.", isCorrect: true },
      { text: "Deployment does not affect policy behavior.", isCorrect: false },
    ],
    explanation:
      "Deployment is harder than static prediction because the policy's own actions determine future states. A small mistake can expose the policy to states that were uncommon or absent in the training demonstrations, and later mistakes can build on that drift. Training and deployment distributions are therefore not guaranteed to be identical, and deployment behavior matters precisely because the policy is part of the feedback loop.",
  },

  // ============================================================
  // Q28–Q35: EXACTLY ONE TRUE
  // ============================================================

  {
    id: "cs224r-lect2-q28",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statement about imitation learning is correct?",
    options: [
      {
        text: "It always requires online interaction with the environment.",
        isCorrect: false,
      },
      { text: "It inherently solves long-horizon planning.", isCorrect: false },
      { text: "It can suffer from covariate shift.", isCorrect: true },
      {
        text: "It guarantees stable performance without interventions.",
        isCorrect: false,
      },
    ],
    explanation:
      "Imitation learning can suffer from covariate shift because the learned policy may visit states that differ from the expert data once it controls the system. It does not always require online environment interaction, because offline behavior cloning is a valid form of imitation learning. It also does not inherently solve long-horizon planning or guarantee stable performance without interventions such as DAgger-style corrective data.",
  },

  {
    id: "cs224r-lect2-q29",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statement about negative log-likelihood training is correct?",
    options: [
      { text: "It minimizes squared error between actions.", isCorrect: false },
      { text: "It maximizes reward directly.", isCorrect: false },
      {
        text: "It fits a probabilistic model to expert actions.",
        isCorrect: true,
      },
      {
        text: "It removes the need for expressive distributions.",
        isCorrect: false,
      },
    ],
    explanation:
      "Negative log-likelihood training fits a probabilistic policy by making expert actions high probability under the model. That differs from squared-error regression, which targets a point estimate and can average over distinct expert modes. It also does not maximize reward directly or remove the need for expressive distributions; the likelihood objective can only use the distribution family the policy provides.",
  },

  {
    id: "cs224r-lect2-q30",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statement about multimodal demonstrations is correct?",
    options: [
      { text: "They should be averaged for safety.", isCorrect: false },
      { text: "They are rare in human data.", isCorrect: false },
      {
        text: "They require expressive policy distributions.",
        isCorrect: true,
      },
      { text: "They invalidate imitation learning.", isCorrect: false },
    ],
    explanation:
      "Multimodal demonstrations contain several distinct plausible actions for similar states, such as different expert strategies. Averaging those actions can create a behavior that no expert would choose and that may be unsafe in continuous control. The right response is not to discard imitation learning, but to use a policy distribution expressive enough to represent multiple modes.",
  },

  {
    id: "cs224r-lect2-q31",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statement about DAgger is correct?",
    options: [
      { text: "It eliminates expert queries entirely.", isCorrect: false },
      {
        text: "It is less data-efficient than offline cloning.",
        isCorrect: false,
      },
      { text: "It reduces compounding errors.", isCorrect: true },
      { text: "It removes the need for demonstrations.", isCorrect: false },
    ],
    explanation:
      "DAgger reduces compounding errors by collecting expert corrections on states reached by the learned policy, then adding those corrections to the training set. It still needs expert queries and usually starts from demonstrations or earlier supervised data, so it does not remove the expert from the loop. Its benefit is better deployment-state coverage, not a guarantee that it is less data-efficient than offline cloning in every setting.",
  },

  {
    id: "cs224r-lect2-q32",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statement about expressive distributions is correct?",
    options: [
      {
        text: "They are unnecessary for single-expert datasets.",
        isCorrect: false,
      },
      { text: "They only matter for large neural networks.", isCorrect: false },
      {
        text: "They improve robustness under expert diversity.",
        isCorrect: true,
      },
      {
        text: "They replace the need for state representation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Expressive output distributions are useful when expert behavior is diverse because they can represent several plausible action regions instead of forcing one averaged action. Even a single expert can produce multimodal behavior across contexts, so expressivity is not only a multi-expert issue. It also does not replace state representation; the policy still needs the right input information to decide which mode is appropriate.",
  },

  {
    id: "cs224r-lect2-q33",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statement about autoregressive action modeling is correct?",
    options: [
      {
        text: "It predicts all action dimensions simultaneously.",
        isCorrect: false,
      },
      {
        text: "It ignores correlations between action dimensions.",
        isCorrect: false,
      },
      { text: "It factorizes the joint action distribution.", isCorrect: true },
      { text: "It requires a reward function.", isCorrect: false },
    ],
    explanation:
      "Autoregressive action modeling factorizes a joint action distribution into a sequence of conditional distributions. That lets later action dimensions depend on earlier ones, so the model can capture correlations without enumerating the full joint table at once. It is still an imitation-learning model of expert actions and does not require a reward function.",
  },

  {
    id: "cs224r-lect2-q34",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statement about imitation learning limitations is correct?",
    options: [
      {
        text: "It can improve beyond expert performance by default.",
        isCorrect: false,
      },
      {
        text: "It provides a mechanism for self-improvement through exploration.",
        isCorrect: false,
      },
      {
        text: "Its performance is bounded by demonstration quality.",
        isCorrect: true,
      },
      { text: "It eliminates the need for data collection.", isCorrect: false },
    ],
    explanation:
      "Imitation learning is limited by the demonstrations because the learner is trained to reproduce observed behavior rather than to discover better behavior through reward-guided exploration. If the demonstrations are poor, narrow, or inconsistent, the policy inherits those limitations. It still requires data collection, and improving beyond the expert by default is not part of the basic imitation-learning objective.",
  },

  {
    id: "cs224r-lect2-q35",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statement about compounding errors is correct?",
    options: [
      { text: "They are unique to stochastic policies.", isCorrect: false },
      {
        text: "They arise because actions affect future states.",
        isCorrect: true,
      },
      { text: "They disappear with enough training epochs.", isCorrect: false },
      { text: "They only occur in robotics.", isCorrect: false },
    ],
    explanation:
      "Compounding errors arise because each action changes the future states the policy must handle. The issue is not unique to stochastic policies or robotics; any sequential decision system can drift away from its training distribution after early mistakes. More supervised training epochs on the same demonstrations do not automatically fix missing coverage of the states produced by the learned policy.",
  },
];
