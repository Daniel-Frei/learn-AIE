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
      'Demonstrations are often imperfect and diverse. Imitation learning aims to match expert behavior, not necessarily optimal behavior. To reason through the choices, select the statements that match the criterion in the prompt: "They may come from multiple experts."; "They define the training distribution for imitation learning."; "They may include inconsistent action choices.". Do not select statements that miss that criterion: "They always correspond to optimal behavior.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'GMMs extend unimodal Gaussians but remain limited compared to fully flexible generative models like diffusion. To reason through the choices, select the statements that match the criterion in the prompt: "They combine multiple Gaussian components."; "They are more expressive than a single Gaussian."; "They require choosing the number of components.". Do not select statements that miss that criterion: "They are maximally expressive for continuous actions.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'Discretization trades metric precision for expressive categorical modeling. Distance relationships are only approximated through binning. To reason through the choices, select the statements that match the criterion in the prompt: "Continuous actions can be binned into discrete categories."; "Discretization enables categorical cross-entropy loss."; "Finer discretization increases expressivity.". Do not select statements that miss that criterion: "Discretization preserves exact distance information between actions.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'Offline methods are safer and simpler but do not inherently solve distribution shift issues that arise during deployment. To reason through the choices, select the statements that match the criterion in the prompt: "It trains using a fixed dataset of demonstrations."; "It does not require running the learned policy during training."; "It avoids safety risks from untrained policies.". Do not select statements that miss that criterion: "It guarantees robustness to compounding errors.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'Compounding errors stem from sequential decision making, not randomness alone. Even deterministic systems can exhibit them. To reason through the choices, select the statements that match the criterion in the prompt: "They arise when policy mistakes alter future states."; "They cause the policy to visit unseen states."; "They create a mismatch between training and deployment distributions.". Do not select statements that miss that criterion: "They only occur in stochastic environments.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'DAgger is inherently online because it relies on policy rollouts and expert feedback during training. To reason through the choices, select the statements that match the criterion in the prompt: "It collects data by running the learned policy."; "It queries the expert at visited states."; "It aggregates corrective data with original demonstrations.". Do not select statements that miss that criterion: "It is a purely offline algorithm.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'Categorical policies fully represent discrete distributions and naturally align with classification objectives. To reason through the choices, select the statements that match the criterion in the prompt: "They output probabilities over discrete actions."; "They are maximally expressive for discrete action spaces."; "They are trained using classification losses.". Do not select statements that miss that criterion: "They require continuous action representations.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'The output distribution fundamentally limits what behaviors can be represented, independent of network depth. To reason through the choices, select the statements that match the criterion in the prompt: "Distribution expressivity differs from network expressivity."; "A large network with L2 loss still predicts a mean."; "Expressive distributions enable safer action sampling.". Do not select statements that miss that criterion: "Distribution choice is irrelevant if the network is deep enough.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'Training conditions on expert actions (teacher forcing). Gradients flow through probabilities, not discrete samples. To reason through the choices, select the statements that match the criterion in the prompt: "Training uses expert actions as conditioning inputs."; "Cross-entropy loss is applied per action dimension."; "Inference feeds sampled actions back into the model.". Do not select statements that miss that criterion: "Backpropagation requires differentiating through sampled actions.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'Imitation learning optimizes likelihood of demonstrations rather than reward. The quality of the expert data strongly affects performance. To reason through the choices, select the statements that match the criterion in the prompt: "They maximize likelihood of expert actions."; "They depend on demonstration quality.". Do not select statements that miss that criterion: "They maximize expected reward directly."; "They require a known reward function.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'Discretization simplifies modeling at the cost of precision. Autoregression avoids exponential blowup of joint distributions. To reason through the choices, select the statements that match the criterion in the prompt: "They convert continuous actions into bins."; "They reduce modeling complexity of joint distributions.". Do not select statements that miss that criterion: "They preserve exact geometric distances."; "They eliminate the need for sampling.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'Covariate shift is intrinsic to sequential prediction. Even large datasets cannot fully eliminate it. To reason through the choices, select the statements that match the criterion in the prompt: "It arises when policy actions affect future states."; "It causes mismatch between expert and policy state distributions.". Do not select statements that miss that criterion: "It is irrelevant when demonstrations are large."; "It can be ignored in sequential decision problems.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'Human-gated DAgger reduces annotation burden but still requires expert oversight during rollouts. To reason through the choices, select the statements that match the criterion in the prompt: "The expert intervenes when the policy fails."; "It is more practical than querying every visited state.". Do not select statements that miss that criterion: "Corrections are collected without policy rollouts."; "It eliminates the need for expert involvement.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'Diffusion trades off inference speed for expressivity. Its iterative nature enables rich continuous distributions. To reason through the choices, select the statements that match the criterion in the prompt: "They operate by iterative refinement."; "They handle multimodal continuous actions well.". Do not select statements that miss that criterion: "They generate actions in a single forward pass."; "They require discretization of actions.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'Expressive policies improve realism and safety but do not guarantee optimality. To reason through the choices, select the statements that match the criterion in the prompt: "They reduce risk of unsafe averaged actions."; "They enable sampling from high-probability regions.". Do not select statements that miss that criterion: "They guarantee optimal performance."; "They are unnecessary for multimodal data.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'Gaussian policies are simple but restrictive. Multimodal tasks require richer distributions. To reason through the choices, select the statements that match the criterion in the prompt: "They are unimodal distributions."; "They are limited for multimodal imitation data.". Do not select statements that miss that criterion: "They can represent arbitrary action distributions."; "They eliminate compounding errors.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'Imitation learning focuses on matching observed behavior, not exceeding it. To reason through the choices, select the statements that match the criterion in the prompt: "Experts may follow different strategies."; "Imitation learning aims to match expert performance.". Do not select statements that miss that criterion: "Imitation learning assumes expert behavior is optimal."; "Imitation learning can outperform experts by default.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'Sequential execution amplifies small mistakes. This makes deployment significantly harder than supervised prediction. To reason through the choices, select the statements that match the criterion in the prompt: "Deployment may expose unseen states."; "Errors can accumulate over time.". Do not select statements that miss that criterion: "Training and deployment distributions are identical."; "Deployment does not affect policy behavior.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'Covariate shift is a central challenge in imitation learning due to sequential decisions. To reason through the choices, select the statements that match the criterion in the prompt: "It can suffer from covariate shift.". Do not select statements that miss that criterion: "It always requires online interaction with the environment."; "It inherently solves long-horizon planning."; "It guarantees stable performance without interventions.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'Negative log-likelihood fits the full action distribution rather than point estimates. To reason through the choices, select the statements that match the criterion in the prompt: "It fits a probabilistic model to expert actions.". Do not select statements that miss that criterion: "It minimizes squared error between actions."; "It maximizes reward directly."; "It removes the need for expressive distributions.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'Multimodal data motivates probabilistic policies rather than deterministic ones. To reason through the choices, select the statements that match the criterion in the prompt: "They require expressive policy distributions.". Do not select statements that miss that criterion: "They should be averaged for safety."; "They are rare in human data."; "They invalidate imitation learning.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'DAgger explicitly addresses distribution shift by collecting corrective data. To reason through the choices, select the statements that match the criterion in the prompt: "It reduces compounding errors.". Do not select statements that miss that criterion: "It eliminates expert queries entirely."; "It is less data-efficient than offline cloning."; "It removes the need for demonstrations.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'Expressive distributions are especially valuable when experts behave differently. To reason through the choices, select the statements that match the criterion in the prompt: "They improve robustness under expert diversity.". Do not select statements that miss that criterion: "They are unnecessary for single-expert datasets."; "They only matter for large neural networks."; "They replace the need for state representation.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'Autoregressive models capture dependencies by conditioning each dimension on previous ones. To reason through the choices, select the statements that match the criterion in the prompt: "It factorizes the joint action distribution.". Do not select statements that miss that criterion: "It predicts all action dimensions simultaneously."; "It ignores correlations between action dimensions."; "It requires a reward function.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'Imitation learning reproduces what it sees. Poor or limited demonstrations constrain performance. To reason through the choices, select the statements that match the criterion in the prompt: "Its performance is bounded by demonstration quality.". Do not select statements that miss that criterion: "It can improve beyond expert performance by default."; "It provides a mechanism for self-improvement through exploration."; "It eliminates the need for data collection.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
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
      'Sequential decision making causes errors to propagate forward, regardless of domain. To reason through the choices, select the statements that match the criterion in the prompt: "They arise because actions affect future states.". Do not select statements that miss that criterion: "They are unique to stochastic policies."; "They disappear with enough training epochs."; "They only occur in robotics.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },
];
