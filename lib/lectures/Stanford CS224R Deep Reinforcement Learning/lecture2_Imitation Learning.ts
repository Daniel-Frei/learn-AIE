import { Question } from "../../quiz";

export const cs224rLecture2ImitationLearningQuestions: Question[] = [
  // ============================================================
  // Lecture 2 – Imitation Learning
  // Q1–Q35
  // ============================================================

  // ============================================================
  // Q1–Q9: ALL TRUE
  // ============================================================

  {
    id: "cs224r-lect2-q01",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements correctly describe imitation learning?",
    options: [
      {
        text: "It aims to learn a policy that mimics expert behavior.",
        isCorrect: true,
      },
      {
        text: "Training data consists of expert-collected trajectories.",
        isCorrect: true,
      },
      {
        text: "The learned policy maps states or observations to actions.",
        isCorrect: true,
      },
      {
        text: "The goal is to achieve high task performance without explicitly defining a reward.",
        isCorrect: true,
      },
    ],
    explanation:
      "Imitation learning focuses on copying expert behavior directly from demonstrations. Unlike reinforcement learning, it does not require an explicit reward function, relying instead on expert trajectories.",
  },

  {
    id: "cs224r-lect2-q02",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements correctly describe behavior cloning?",
    options: [
      {
        text: "It reduces imitation learning to supervised learning.",
        isCorrect: true,
      },
      {
        text: "The policy is trained to predict expert actions from states.",
        isCorrect: true,
      },
      {
        text: "A common loss is squared error between predicted and expert actions.",
        isCorrect: true,
      },
      {
        text: "The learned policy can be deployed directly after training.",
        isCorrect: true,
      },
    ],
    explanation:
      "Behavior cloning treats imitation as supervised regression or classification. The policy is trained on fixed data and then deployed without further interaction.",
  },

  {
    id: "cs224r-lect2-q03",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements correctly describe expert demonstrations?",
    options: [
      {
        text: "They consist of sequences of states and actions.",
        isCorrect: true,
      },
      {
        text: "They are assumed to come from an unknown expert policy.",
        isCorrect: true,
      },
      {
        text: "They define the data distribution used for training.",
        isCorrect: true,
      },
      {
        text: "They may contain variability due to different human strategies.",
        isCorrect: true,
      },
    ],
    explanation:
      "Demonstrations are trajectories sampled from an expert policy. When multiple experts contribute data, demonstrations often reflect diverse strategies.",
  },

  {
    id: "cs224r-lect2-q04",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe multimodality in imitation learning data?",
    options: [
      {
        text: "Different experts may choose different valid actions in the same state.",
        isCorrect: true,
      },
      {
        text: "Multimodality leads to multiple peaks in the action distribution.",
        isCorrect: true,
      },
      {
        text: "Averaging expert actions can produce low-probability behaviors.",
        isCorrect: true,
      },
      {
        text: "Multimodality is common in human-collected datasets.",
        isCorrect: true,
      },
    ],
    explanation:
      "Human demonstrations are rarely perfectly consistent. Multimodal action distributions arise naturally and must be handled carefully to avoid unsafe averaging.",
  },

  {
    id: "cs224r-lect2-q05",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe why deterministic L2 regression can fail?",
    options: [
      {
        text: "It predicts the mean of the action distribution.",
        isCorrect: true,
      },
      {
        text: "The mean action may have low probability under the data.",
        isCorrect: true,
      },
      {
        text: "It cannot represent multiple distinct expert strategies.",
        isCorrect: true,
      },
      {
        text: "Increasing network size does not fix distributional collapse.",
        isCorrect: true,
      },
    ],
    explanation:
      "L2 regression encourages mean prediction, which is problematic for multimodal data. Even large neural networks cannot overcome the limitation of a unimodal output distribution.",
  },

  {
    id: "cs224r-lect2-q06",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe expressive policy distributions?",
    options: [
      {
        text: "They represent full action distributions rather than single actions.",
        isCorrect: true,
      },
      { text: "They can capture multimodal expert behavior.", isCorrect: true },
      {
        text: "They are typically trained by maximizing log-likelihood of demonstrations.",
        isCorrect: true,
      },
      {
        text: "They reduce the risk of implausible averaged actions.",
        isCorrect: true,
      },
    ],
    explanation:
      "Expressive distributions allow policies to sample high-probability actions rather than collapsing to means. This is critical for realistic and safe behavior.",
  },

  {
    id: "cs224r-lect2-q07",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe learning a policy via maximum likelihood?",
    options: [
      {
        text: "The objective minimizes negative log probability of expert actions.",
        isCorrect: true,
      },
      {
        text: "The policy defines a probability distribution π(a|s).",
        isCorrect: true,
      },
      {
        text: "Training maximizes agreement with demonstrated behavior.",
        isCorrect: true,
      },
      {
        text: "The learned policy induces a distribution over trajectories.",
        isCorrect: true,
      },
    ],
    explanation:
      "Maximum likelihood training fits the policy distribution to the data distribution. The policy affects the induced trajectory distribution through sequential action sampling.",
  },

  {
    id: "cs224r-lect2-q08",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe autoregressive action modeling?",
    options: [
      { text: "Actions are modeled one dimension at a time.", isCorrect: true },
      {
        text: "Each dimension is conditioned on previously generated dimensions.",
        isCorrect: true,
      },
      {
        text: "The full joint distribution factorizes autoregressively.",
        isCorrect: true,
      },
      {
        text: "This approach scales to high-dimensional actions.",
        isCorrect: true,
      },
    ],
    explanation:
      "Autoregressive models decompose a joint distribution into conditional factors. This allows efficient modeling of complex, high-dimensional action spaces.",
  },

  {
    id: "cs224r-lect2-q09",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe diffusion policies for imitation learning?",
    options: [
      {
        text: "They generate actions via iterative denoising.",
        isCorrect: true,
      },
      {
        text: "They can model complex continuous action distributions.",
        isCorrect: true,
      },
      { text: "Sampling requires multiple refinement steps.", isCorrect: true },
      {
        text: "They are more expressive than single Gaussian policies.",
        isCorrect: true,
      },
    ],
    explanation:
      "Diffusion policies gradually transform noise into actions. Their iterative structure allows them to represent highly complex distributions.",
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
      "Demonstrations are often imperfect and diverse. Imitation learning aims to match expert behavior, not necessarily optimal behavior.",
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
      "GMMs extend unimodal Gaussians but remain limited compared to fully flexible generative models like diffusion.",
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
      "Discretization trades metric precision for expressive categorical modeling. Distance relationships are only approximated through binning.",
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
      "Offline methods are safer and simpler but do not inherently solve distribution shift issues that arise during deployment.",
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
      "Compounding errors stem from sequential decision making, not randomness alone. Even deterministic systems can exhibit them.",
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
      "DAgger is inherently online because it relies on policy rollouts and expert feedback during training.",
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
      "Categorical policies fully represent discrete distributions and naturally align with classification objectives.",
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
      "The output distribution fundamentally limits what behaviors can be represented, independent of network depth.",
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
      "Training conditions on expert actions (teacher forcing). Gradients flow through probabilities, not discrete samples.",
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
      "Imitation learning optimizes likelihood of demonstrations rather than reward. The quality of the expert data strongly affects performance.",
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
      "Discretization simplifies modeling at the cost of precision. Autoregression avoids exponential blowup of joint distributions.",
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
      "Covariate shift is intrinsic to sequential prediction. Even large datasets cannot fully eliminate it.",
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
      "Human-gated DAgger reduces annotation burden but still requires expert oversight during rollouts.",
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
      "Diffusion trades off inference speed for expressivity. Its iterative nature enables rich continuous distributions.",
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
      "Expressive policies improve realism and safety but do not guarantee optimality.",
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
      "Gaussian policies are simple but restrictive. Multimodal tasks require richer distributions.",
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
      "Imitation learning focuses on matching observed behavior, not exceeding it.",
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
      "Sequential execution amplifies small mistakes. This makes deployment significantly harder than supervised prediction.",
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
      "Covariate shift is a central challenge in imitation learning due to sequential decisions.",
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
      "Negative log-likelihood fits the full action distribution rather than point estimates.",
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
      "Multimodal data motivates probabilistic policies rather than deterministic ones.",
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
      "DAgger explicitly addresses distribution shift by collecting corrective data.",
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
      "Expressive distributions are especially valuable when experts behave differently.",
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
      "Autoregressive models capture dependencies by conditioning each dimension on previous ones.",
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
      "Imitation learning reproduces what it sees. Poor or limited demonstrations constrain performance.",
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
      "Sequential decision making causes errors to propagate forward, regardless of domain.",
  },
];
