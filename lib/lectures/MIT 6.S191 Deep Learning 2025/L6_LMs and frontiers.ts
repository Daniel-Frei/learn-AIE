import { Question } from "../../quiz";

export const MIT6S191_L6_LMsAndFrontiersQuestions: Question[] = [
  {
    id: "mit6s191-l6-q01",
    chapter: 6,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the Universal Approximation Theorem?",
    options: [
      {
        text: "A feedforward neural network with a single hidden layer can approximate any continuous function to arbitrary precision given sufficient capacity.",
        isCorrect: true,
      },
      {
        text: "The theorem guarantees efficient training via gradient descent.",
        isCorrect: false,
      },
      {
        text: "The theorem does not specify how many hidden units are required.",
        isCorrect: true,
      },
      {
        text: "The theorem does not guarantee good generalization to unseen data.",
        isCorrect: true,
      },
    ],
    explanation:
      "The Universal Approximation Theorem states that a single hidden layer network can approximate any continuous function given enough neurons. However, it does not provide guarantees about training efficiency or generalization. The number of units required may be impractically large.",
  },

  {
    id: "mit6s191-l6-q02",
    chapter: 6,
    difficulty: "medium",
    prompt:
      "In experiments where training labels are randomized, what observations are correct?",
    options: [
      {
        text: "Neural networks can still achieve near 100% training accuracy.",
        isCorrect: true,
      },
      {
        text: "Test accuracy remains high even with completely random labels.",
        isCorrect: false,
      },
      {
        text: "The model effectively memorizes the training set.",
        isCorrect: true,
      },
      {
        text: "This demonstrates the high capacity of deep networks as function approximators.",
        isCorrect: true,
      },
    ],
    explanation:
      "With randomized labels, deep networks can perfectly fit the training data, showing high capacity. However, test accuracy drops because the mapping does not generalize. This illustrates memorization and challenges classical generalization intuitions.",
  },

  {
    id: "mit6s191-l6-q03",
    chapter: 6,
    difficulty: "hard",
    prompt: "Out-of-distribution (OOD) generalization refers to:",
    options: [
      {
        text: "Model behavior on inputs far from the training distribution.",
        isCorrect: true,
      },
      { text: "Performance on the training set.", isCorrect: false },
      {
        text: "Regions of input space where the model has limited training support.",
        isCorrect: true,
      },
      {
        text: "The inability to estimate uncertainty outside the training domain.",
        isCorrect: true,
      },
    ],
    explanation:
      "OOD generalization concerns inputs that differ from the training distribution. Neural networks may behave unpredictably in these regions. Estimating uncertainty in OOD regions is a major open challenge.",
  },

  {
    id: "mit6s191-l6-q04",
    chapter: 6,
    difficulty: "medium",
    prompt: "Adversarial examples are constructed by:",
    options: [
      {
        text: "Perturbing input data to increase prediction error.",
        isCorrect: true,
      },
      {
        text: "Updating network weights to minimize training loss.",
        isCorrect: false,
      },
      { text: "Using gradients with respect to the input.", isCorrect: true },
      {
        text: "Applying small perturbations that are often imperceptible to humans.",
        isCorrect: true,
      },
    ],
    explanation:
      "Adversarial attacks compute gradients with respect to the input and perturb the input to increase loss. These perturbations can be visually imperceptible yet drastically alter predictions.",
  },

  {
    id: "mit6s191-l6-q05",
    chapter: 6,
    difficulty: "easy",
    prompt: "Which are limitations or concerns in deep learning deployment?",
    options: [
      { text: "Data bias embedded in training datasets.", isCorrect: true },
      {
        text: "Uncertainty in predictions for safety-critical applications.",
        isCorrect: true,
      },
      { text: "Adversarial vulnerabilities.", isCorrect: true },
      {
        text: "Guaranteed robustness across all environments.",
        isCorrect: false,
      },
    ],
    explanation:
      "Deep learning systems inherit biases from data, can be uncertain in OOD settings, and are vulnerable to adversarial attacks. Robustness across all environments is not guaranteed.",
  },

  {
    id: "mit6s191-l6-q06",
    chapter: 6,
    difficulty: "medium",
    prompt: "In diffusion models, the forward process:",
    options: [
      { text: "Progressively adds noise to training data.", isCorrect: true },
      {
        text: "Requires learning parameters via gradient descent.",
        isCorrect: false,
      },
      {
        text: "Eventually transforms data into nearly pure noise.",
        isCorrect: true,
      },
      {
        text: "Defines a sequence of noisy intermediate states.",
        isCorrect: true,
      },
    ],
    explanation:
      "The forward diffusion process gradually adds noise to data until it becomes nearly random noise. This process is fixed and does not require learning. It creates training pairs for the reverse denoising model.",
  },

  {
    id: "mit6s191-l6-q07",
    chapter: 6,
    difficulty: "hard",
    prompt: "The reverse diffusion process is trained to:",
    options: [
      {
        text: "Predict the noise added between successive timesteps.",
        isCorrect: true,
      },
      {
        text: "Minimize mean squared error between predicted and true noise.",
        isCorrect: true,
      },
      { text: "Directly classify images.", isCorrect: false },
      { text: "Iteratively remove noise to recover data.", isCorrect: true },
    ],
    explanation:
      "Diffusion models train a neural network to predict noise between timesteps. The loss is typically mean squared error. During sampling, this model iteratively removes noise.",
  },

  {
    id: "mit6s191-l6-q08",
    chapter: 6,
    difficulty: "medium",
    prompt:
      "Compared to single-shot generative models (e.g., GANs), diffusion models:",
    options: [
      { text: "Generate samples iteratively.", isCorrect: true },
      { text: "Often produce high-fidelity outputs.", isCorrect: true },
      { text: "Collapse to a single mode by design.", isCorrect: false },
      {
        text: "Simplify learning by decomposing generation into steps.",
        isCorrect: true,
      },
    ],
    explanation:
      "Diffusion models generate samples step-by-step via denoising. This decomposition simplifies training and leads to high-quality outputs. They are less prone to mode collapse than GANs.",
  },

  {
    id: "mit6s191-l6-q09",
    chapter: 6,
    difficulty: "easy",
    prompt: "Large Language Models (LLMs) are best described as:",
    options: [
      {
        text: "Very large neural networks trained on massive text datasets.",
        isCorrect: true,
      },
      {
        text: "Systems that explicitly store labeled Q&A pairs for all responses.",
        isCorrect: false,
      },
      {
        text: "Models trained using self-supervised learning objectives.",
        isCorrect: true,
      },
      {
        text: "Function approximators over sequences of tokens.",
        isCorrect: true,
      },
    ],
    explanation:
      "LLMs are large neural networks trained on broad text corpora. They use self-supervised objectives such as next-token prediction and act as function approximators over token sequences.",
  },

  {
    id: "mit6s191-l6-q10",
    chapter: 6,
    difficulty: "medium",
    prompt: "Next-token prediction can be framed mathematically as:",
    options: [
      { text: "Estimating \\(p(x_t \\mid x_{1:t-1})\\).", isCorrect: true },
      {
        text: "Minimizing cross-entropy loss between predicted and true tokens.",
        isCorrect: true,
      },
      {
        text: "A multi-class classification problem over the vocabulary.",
        isCorrect: true,
      },
      {
        text: "A regression task predicting continuous pixel intensities.",
        isCorrect: false,
      },
    ],
    explanation:
      "Next-token prediction models the conditional probability of the next token given previous tokens. It is trained using cross-entropy and is equivalent to multi-class classification over the vocabulary.",
  },

  {
    id: "mit6s191-l6-q11",
    chapter: 6,
    difficulty: "hard",
    prompt: "The cross-entropy loss for next-token prediction is:",
    options: [
      {
        text: "\\( \\mathcal{L} = -\\sum_i y_i \\log \\hat{y}_i \\).",
        isCorrect: true,
      },
      {
        text: "Minimized when predicted probabilities match the true one-hot label.",
        isCorrect: true,
      },
      {
        text: "Equivalent to mean squared error for classification tasks.",
        isCorrect: false,
      },
      {
        text: "Encouraging the model to assign high probability to the correct next token.",
        isCorrect: true,
      },
    ],
    explanation:
      "Cross-entropy measures divergence between predicted and true distributions. It is minimized when predicted probabilities align with the one-hot true token. It is not equivalent to mean squared error.",
  },

  {
    id: "mit6s191-l6-q12",
    chapter: 6,
    difficulty: "medium",
    prompt: "Self-supervised learning in LLMs refers to:",
    options: [
      {
        text: "Using the data itself to create training targets.",
        isCorrect: true,
      },
      {
        text: "Requiring human annotation for every training example.",
        isCorrect: false,
      },
      { text: "Predicting masked or next tokens in text.", isCorrect: true },
      {
        text: "Learning representations without explicit labels.",
        isCorrect: true,
      },
    ],
    explanation:
      "Self-supervised learning uses structure within data to generate targets, such as next-token prediction. It avoids manual labeling and allows training on massive unlabeled corpora.",
  },

  {
    id: "mit6s191-l6-q13",
    chapter: 6,
    difficulty: "hard",
    prompt: "Scaling laws in large language models suggest:",
    options: [
      {
        text: "Performance often improves predictably with model size and data scale.",
        isCorrect: true,
      },
      {
        text: "Emergent abilities may appear at certain parameter thresholds.",
        isCorrect: true,
      },
      {
        text: "Increasing parameters always guarantees safe behavior.",
        isCorrect: false,
      },
      {
        text: "Training compute plays a role alongside model size.",
        isCorrect: true,
      },
    ],
    explanation:
      "Scaling laws show systematic improvements with increased parameters, data, and compute. Certain capabilities emerge at scale. However, larger models do not automatically guarantee safety.",
  },

  {
    id: "mit6s191-l6-q14",
    chapter: 6,
    difficulty: "easy",
    prompt: "Hallucinations in LLMs refer to:",
    options: [
      { text: "Fluent outputs that are factually incorrect.", isCorrect: true },
      { text: "Guaranteed model failures on every prompt.", isCorrect: false },
      { text: "Confident but ungrounded responses.", isCorrect: true },
      { text: "A challenge in uncertainty estimation.", isCorrect: true },
    ],
    explanation:
      "Hallucinations are outputs that sound plausible but are not grounded in truth. They highlight uncertainty calibration challenges. They are not guaranteed failures but occur under certain conditions.",
  },

  {
    id: "mit6s191-l6-q15",
    chapter: 6,
    difficulty: "medium",
    prompt: "Diffusion models begin sampling by:",
    options: [
      { text: "Starting from random noise.", isCorrect: true },
      { text: "Applying iterative denoising steps.", isCorrect: true },
      {
        text: "Using a trained model to predict noise residuals.",
        isCorrect: true,
      },
      {
        text: "Feeding labeled supervision signals at inference time.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sampling begins from random noise and iteratively removes noise using the trained denoising network. No supervision is required during inference.",
  },

  {
    id: "mit6s191-l6-q16",
    chapter: 6,
    difficulty: "hard",
    prompt: "Why can diffusion models capture high diversity in outputs?",
    options: [
      {
        text: "They start from maximally variable random noise.",
        isCorrect: true,
      },
      {
        text: "Each noise initialization can lead to a different sample.",
        isCorrect: true,
      },
      {
        text: "The iterative process refines different trajectories.",
        isCorrect: true,
      },
      {
        text: "They deterministically output the same sample each time.",
        isCorrect: false,
      },
    ],
    explanation:
      "Different random noise seeds produce different denoising trajectories. The iterative refinement allows capturing diverse samples. They are not deterministic unless noise seeds are fixed.",
  },

  {
    id: "mit6s191-l6-q17",
    chapter: 6,
    difficulty: "medium",
    prompt: "Uncertainty estimation in deep learning is important because:",
    options: [
      { text: "Models may encounter unseen scenarios.", isCorrect: true },
      {
        text: "Confidence calibration impacts safety-critical decisions.",
        isCorrect: true,
      },
      {
        text: "All neural networks automatically estimate uncertainty perfectly.",
        isCorrect: false,
      },
      {
        text: "It helps detect hallucinations in language models.",
        isCorrect: true,
      },
    ],
    explanation:
      "Uncertainty estimation helps identify unreliable predictions, especially in safety-critical systems. Neural networks do not inherently estimate uncertainty well without special methods.",
  },

  {
    id: "mit6s191-l6-q18",
    chapter: 6,
    difficulty: "easy",
    prompt: "Neural networks can be viewed fundamentally as:",
    options: [
      { text: "Function approximators.", isCorrect: true },
      { text: "Probability distribution estimators.", isCorrect: true },
      { text: "Models mapping data to decisions.", isCorrect: true },
      { text: "Systems guaranteed to generalize perfectly.", isCorrect: false },
    ],
    explanation:
      "Neural networks approximate functions and distributions. They map inputs to outputs but are not guaranteed to generalize perfectly outside the training domain.",
  },

  {
    id: "mit6s191-l6-q19",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "Why does next-token prediction enable broad language capabilities?",
    options: [
      {
        text: "Language structure is implicitly encoded in token sequences.",
        isCorrect: true,
      },
      {
        text: "Maximizing likelihood over large corpora captures grammar and semantics.",
        isCorrect: true,
      },
      {
        text: "Predicting tokens forces learning contextual dependencies.",
        isCorrect: true,
      },
      {
        text: "It explicitly encodes symbolic logic rules into the network.",
        isCorrect: false,
      },
    ],
    explanation:
      "By predicting the next token, models must learn grammar, semantics, and contextual relationships. This objective captures rich structure implicitly without explicitly encoding logic rules.",
  },

  {
    id: "mit6s191-l6-q20",
    chapter: 6,
    difficulty: "medium",
    prompt: "Which tradeoffs characterize modern AI frontiers?",
    options: [
      {
        text: "Scaling improves performance but increases compute cost.",
        isCorrect: true,
      },
      {
        text: "Greater capability may introduce new safety concerns.",
        isCorrect: true,
      },
      { text: "More parameters always eliminate bias.", isCorrect: false },
      {
        text: "Deployment requires careful consideration of uncertainty and ethics.",
        isCorrect: true,
      },
    ],
    explanation:
      "Scaling increases performance but also compute demands and safety risks. Larger models do not automatically remove bias. Responsible deployment requires ethical and uncertainty-aware design.",
  },

  {
    id: "mit6s191-l6-q21",
    chapter: 6,
    difficulty: "medium",
    prompt:
      "In diffusion models, the forward process can be written as:\n\\[\nq(x_t \\mid x_{t-1}) = \\mathcal{N}(\\sqrt{1-\\beta_t}x_{t-1}, \\beta_t I)\n\\]\nWhich statements are correct?",
    options: [
      {
        text: "The process progressively increases variance over time.",
        isCorrect: true,
      },
      {
        text: "As \\(t\\) increases, \\(x_t\\) approaches a Gaussian distribution.",
        isCorrect: true,
      },
      {
        text: "The forward process is fixed and does not require learning.",
        isCorrect: true,
      },
      {
        text: "The forward process learns parameters \\(\\beta_t\\) during training.",
        isCorrect: false,
      },
    ],
    explanation:
      "In diffusion models, the forward process gradually increases noise variance through predefined \\(\\beta_t\\). Over many steps, the data distribution converges toward an isotropic Gaussian. These \\(\\beta_t\\) values are typically fixed, not learned.",
  },

  {
    id: "mit6s191-l6-q22",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "During training, diffusion models often minimize:\n\\[\n\\mathcal{L} = \\mathbb{E}_{x,\\epsilon,t} \\| \\epsilon - \\epsilon_\\theta(x_t, t) \\|^2\n\\]\nWhich interpretations are correct?",
    options: [
      {
        text: "The model learns to predict the added noise \\(\\epsilon\\).",
        isCorrect: true,
      },
      { text: "The objective is a mean squared error loss.", isCorrect: true },
      { text: "The model directly predicts class labels.", isCorrect: false },
      {
        text: "Minimizing this loss helps approximate the reverse diffusion process.",
        isCorrect: true,
      },
    ],
    explanation:
      "Diffusion models are trained to predict the noise added at each timestep. The loss is mean squared error between predicted and true noise. This enables learning the reverse denoising process.",
  },

  {
    id: "mit6s191-l6-q23",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "Adversarial attacks can be formulated as solving:\n\\[\n\\max_{\\delta} \\mathcal{L}(f_\\theta(x+\\delta), y)\n\\]\nsubject to \\(\\|\\delta\\| \\leq \\epsilon\\).\nWhich are correct?",
    options: [
      {
        text: "The objective increases the model’s loss by modifying the input.",
        isCorrect: true,
      },
      {
        text: "The perturbation \\(\\delta\\) is constrained to be small.",
        isCorrect: true,
      },
      {
        text: "This differs from training, where weights are optimized.",
        isCorrect: true,
      },
      {
        text: "The optimization modifies model parameters \\(\\theta\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Adversarial attacks optimize over input perturbations while keeping model weights fixed. The perturbation is constrained in norm. This is the reverse of training, where weights are optimized.",
  },

  {
    id: "mit6s191-l6-q24",
    chapter: 6,
    difficulty: "medium",
    prompt: "Which factors contribute to hallucinations in LLMs?",
    options: [
      {
        text: "Overconfidence in next-token probability estimates.",
        isCorrect: true,
      },
      {
        text: "Lack of grounding in external verified sources.",
        isCorrect: true,
      },
      { text: "Imperfect uncertainty calibration.", isCorrect: true },
      {
        text: "Explicit symbolic fact-checking built into the base objective.",
        isCorrect: false,
      },
    ],
    explanation:
      "LLMs are trained on next-token prediction without explicit grounding. Poor uncertainty calibration can lead to confident but incorrect outputs. The base objective does not enforce fact verification.",
  },

  {
    id: "mit6s191-l6-q25",
    chapter: 6,
    difficulty: "easy",
    prompt: "Which are advantages of diffusion models compared to GANs?",
    options: [
      { text: "More stable training dynamics.", isCorrect: true },
      { text: "Reduced mode collapse.", isCorrect: true },
      { text: "High-quality image synthesis.", isCorrect: true },
      { text: "Single forward-pass generation only.", isCorrect: false },
    ],
    explanation:
      "Diffusion models are typically more stable and less prone to mode collapse than GANs. They generate high-quality images but require iterative sampling rather than a single forward pass.",
  },

  {
    id: "mit6s191-l6-q26",
    chapter: 6,
    difficulty: "medium",
    prompt:
      "Next-token prediction maximizes:\n\\[\n\\prod_{t=1}^{T} p(x_t \\mid x_{1:t-1})\n\\]\nWhich are correct?",
    options: [
      {
        text: "This corresponds to maximum likelihood estimation.",
        isCorrect: true,
      },
      {
        text: "Taking logs yields a sum of log-probabilities.",
        isCorrect: true,
      },
      { text: "The loss becomes cross-entropy over tokens.", isCorrect: true },
      { text: "It directly minimizes mean squared error.", isCorrect: false },
    ],
    explanation:
      "Next-token prediction is equivalent to maximizing likelihood of the sequence. Taking the log turns the product into a sum. Cross-entropy loss is used rather than mean squared error.",
  },

  {
    id: "mit6s191-l6-q27",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "Why does increasing model scale sometimes produce emergent abilities?",
    options: [
      {
        text: "Nonlinear scaling effects allow new behaviors at parameter thresholds.",
        isCorrect: true,
      },
      {
        text: "Larger models can represent more complex functions.",
        isCorrect: true,
      },
      {
        text: "Training on broader data exposes more task structure.",
        isCorrect: true,
      },
      {
        text: "Emergent abilities are mathematically guaranteed by theory.",
        isCorrect: false,
      },
    ],
    explanation:
      "Emergent abilities appear as nonlinear improvements when scaling models and data. Larger models capture richer structure. However, theory does not strictly guarantee emergence.",
  },

  {
    id: "mit6s191-l6-q28",
    chapter: 6,
    difficulty: "medium",
    prompt:
      "In uncertainty estimation, predictive confidence may not reflect true accuracy because:",
    options: [
      {
        text: "Softmax outputs are not calibrated probabilities.",
        isCorrect: true,
      },
      {
        text: "OOD inputs can produce high-confidence predictions.",
        isCorrect: true,
      },
      {
        text: "Neural networks inherently model epistemic uncertainty perfectly.",
        isCorrect: false,
      },
      {
        text: "Training data distribution shapes model beliefs.",
        isCorrect: true,
      },
    ],
    explanation:
      "Softmax outputs can be overconfident. Models can assign high confidence to OOD inputs. Neural networks do not automatically provide calibrated uncertainty.",
  },

  {
    id: "mit6s191-l6-q29",
    chapter: 6,
    difficulty: "easy",
    prompt: "Tokenization in LLMs:",
    options: [
      {
        text: "Splits raw text into discrete units called tokens.",
        isCorrect: true,
      },
      { text: "Allows numerical encoding of text.", isCorrect: true },
      { text: "Defines the vocabulary space.", isCorrect: true },
      { text: "Guarantees semantic understanding.", isCorrect: false },
    ],
    explanation:
      "Tokenization converts text into discrete tokens for numerical processing. It defines the vocabulary but does not guarantee semantic comprehension.",
  },

  {
    id: "mit6s191-l6-q30",
    chapter: 6,
    difficulty: "hard",
    prompt: "Which statements about diffusion sampling are correct?",
    options: [
      {
        text: "Sampling iteratively applies the learned denoiser.",
        isCorrect: true,
      },
      {
        text: "Each step approximates \\(p(x_{t-1} \\mid x_t)\\).",
        isCorrect: true,
      },
      {
        text: "Starting from pure noise increases diversity.",
        isCorrect: true,
      },
      {
        text: "The process requires ground-truth images at inference.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sampling iteratively applies the learned reverse transition. Each step approximates the conditional distribution. It starts from noise and does not require labeled data during inference.",
  },

  {
    id: "mit6s191-l6-q31",
    chapter: 6,
    difficulty: "medium",
    prompt: "Why does memorization not imply generalization?",
    options: [
      { text: "A model can perfectly fit random labels.", isCorrect: true },
      {
        text: "Generalization depends on performance on unseen data.",
        isCorrect: true,
      },
      {
        text: "High training accuracy alone proves robustness.",
        isCorrect: false,
      },
      { text: "Capacity allows fitting arbitrary mappings.", isCorrect: true },
    ],
    explanation:
      "Neural networks can memorize random mappings, showing high training accuracy. True generalization requires performance on new data. High capacity alone does not guarantee robustness.",
  },

  {
    id: "mit6s191-l6-q32",
    chapter: 6,
    difficulty: "hard",
    prompt: "Which limitations remain for modern LLMs?",
    options: [
      {
        text: "Difficulty with long-term planning and reasoning.",
        isCorrect: true,
      },
      {
        text: "Challenges in grounding outputs in verified knowledge.",
        isCorrect: true,
      },
      { text: "Perfect calibration of uncertainty.", isCorrect: false },
      { text: "Risk of bias from training data.", isCorrect: true },
    ],
    explanation:
      "LLMs still struggle with long-term planning and robust reasoning. They can hallucinate and inherit biases. Uncertainty calibration remains imperfect.",
  },

  {
    id: "mit6s191-l6-q33",
    chapter: 6,
    difficulty: "easy",
    prompt: "Deep learning systems depend critically on:",
    options: [
      { text: "Data quality.", isCorrect: true },
      { text: "Appropriate task formulation.", isCorrect: true },
      { text: "Garbage-in-garbage-out effects.", isCorrect: true },
      { text: "Magic automatic correctness.", isCorrect: false },
    ],
    explanation:
      "Deep learning performance depends heavily on data quality and task design. Poor data leads to poor outcomes. These systems are not magical guarantees of correctness.",
  },

  {
    id: "mit6s191-l6-q34",
    chapter: 6,
    difficulty: "medium",
    prompt:
      "Which are consequences of OOD failures in safety-critical systems?",
    options: [
      {
        text: "Autonomous systems may misinterpret unseen scenarios.",
        isCorrect: true,
      },
      { text: "Prediction confidence may be misleading.", isCorrect: true },
      { text: "OOD guarantees safe fallback behavior.", isCorrect: false },
      { text: "Uncertainty estimation becomes crucial.", isCorrect: true },
    ],
    explanation:
      "OOD failures can lead to catastrophic errors in autonomous systems. Confidence estimates may be unreliable. Reliable uncertainty estimation is critical.",
  },

  {
    id: "mit6s191-l6-q35",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "Which conceptual parallels exist between diffusion models and maximum likelihood?",
    options: [
      { text: "Both attempt to model the data distribution.", isCorrect: true },
      {
        text: "Diffusion training approximates likelihood objectives.",
        isCorrect: true,
      },
      {
        text: "Iterative denoising estimates probability transitions.",
        isCorrect: true,
      },
      {
        text: "Diffusion explicitly computes exact likelihood in closed form.",
        isCorrect: false,
      },
    ],
    explanation:
      "Diffusion models approximate likelihood-based generative modeling. They learn transition distributions but do not typically compute exact likelihood in closed form.",
  },

  {
    id: "mit6s191-l6-q36",
    chapter: 6,
    difficulty: "medium",
    prompt: "LLM post-training often includes:",
    options: [
      { text: "Alignment tuning.", isCorrect: true },
      { text: "Safety guardrails.", isCorrect: true },
      { text: "Reinforcement learning from human feedback.", isCorrect: true },
      { text: "Removal of all uncertainty.", isCorrect: false },
    ],
    explanation:
      "Post-training includes alignment methods and reinforcement learning from human feedback. Safety guardrails are implemented, but uncertainty remains.",
  },

  {
    id: "mit6s191-l6-q37",
    chapter: 6,
    difficulty: "easy",
    prompt: "Diffusion models generate diversity because:",
    options: [
      { text: "Random initial noise seeds differ.", isCorrect: true },
      { text: "Denoising is iterative and stochastic.", isCorrect: true },
      { text: "Multiple sampling paths are possible.", isCorrect: true },
      { text: "They always output identical images.", isCorrect: false },
    ],
    explanation:
      "Different noise seeds and stochastic transitions yield different outputs. This supports high diversity. Outputs are not identical unless seeds are fixed.",
  },

  {
    id: "mit6s191-l6-q38",
    chapter: 6,
    difficulty: "hard",
    prompt: "If \\(f_\\theta(x)\\) is overconfident OOD, which are risks?",
    options: [
      { text: "Misleading high-probability predictions.", isCorrect: true },
      { text: "Failure to trigger safety mechanisms.", isCorrect: true },
      { text: "Robust extrapolation guarantees.", isCorrect: false },
      {
        text: "Poor decision-making under distribution shift.",
        isCorrect: true,
      },
    ],
    explanation:
      "Overconfidence on OOD inputs leads to dangerous mispredictions. Safety mechanisms may fail if uncertainty is misestimated. Robust extrapolation is not guaranteed.",
  },

  {
    id: "mit6s191-l6-q39",
    chapter: 6,
    difficulty: "medium",
    prompt: "Scaling laws imply diminishing returns because:",
    options: [
      {
        text: "Performance improvements may follow power-law trends.",
        isCorrect: true,
      },
      { text: "Compute cost grows significantly with scale.", isCorrect: true },
      {
        text: "Emergent gains are not linear with parameter growth.",
        isCorrect: true,
      },
      { text: "Scaling always eliminates hallucinations.", isCorrect: false },
    ],
    explanation:
      "Scaling often follows power-law improvements. Gains are nonlinear and costly. Scaling alone does not remove hallucinations.",
  },

  {
    id: "mit6s191-l6-q40",
    chapter: 6,
    difficulty: "hard",
    prompt:
      "Why is next-token prediction sufficient to learn rich representations?",
    options: [
      {
        text: "It forces modeling of contextual dependencies.",
        isCorrect: true,
      },
      { text: "It captures long-range statistical patterns.", isCorrect: true },
      { text: "It indirectly encodes syntax and semantics.", isCorrect: true },
      {
        text: "It explicitly trains symbolic reasoning modules.",
        isCorrect: false,
      },
    ],
    explanation:
      "Next-token prediction requires capturing contextual structure and statistical regularities of language. Syntax and semantics emerge implicitly. Symbolic reasoning is not explicitly encoded.",
  },
];
