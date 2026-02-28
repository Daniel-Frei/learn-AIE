import { Question } from "../../quiz";

export const L1_IntroductionToNeuralNetworksAndDeepLearning: Question[] = [
  {
    id: "mit6s191-l1-q01",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the relationship between Artificial Intelligence, Machine Learning, and Deep Learning?",
    options: [
      {
        text: "Deep Learning is a subset of Machine Learning.",
        isCorrect: true,
      },
      {
        text: "Machine Learning is a subset of Artificial Intelligence.",
        isCorrect: true,
      },
      {
        text: "Deep Learning typically uses neural networks to learn patterns from data.",
        isCorrect: true,
      },
      {
        text: "Machine Learning focuses on learning patterns from data rather than being explicitly programmed with rules.",
        isCorrect: true,
      },
    ],
    explanation:
      "Artificial Intelligence is the broad goal of building systems that exhibit intelligent behavior. Machine Learning is a subset that learns patterns from data rather than being rule-based. Deep Learning is a further subset that uses neural networks—often deep, multi-layered ones—to extract hierarchical representations from data.",
  },

  {
    id: "mit6s191-l1-q02",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which of the following are true about a single perceptron (neuron)?",
    options: [
      {
        text: "It computes a weighted sum of inputs, adds a bias, and applies a nonlinear activation.",
        isCorrect: true,
      },
      {
        text: "The bias term allows shifting the activation function left or right.",
        isCorrect: true,
      },
      {
        text: "Without an activation function, the perceptron represents a linear model.",
        isCorrect: true,
      },
      {
        text: "The operation can be written as \\( y = g(\\mathbf{w}^T \\mathbf{x} + b) \\).",
        isCorrect: true,
      },
    ],
    explanation:
      "A perceptron performs a dot product \\(\\mathbf{w}^T \\mathbf{x}\\), adds a bias \\(b\\), and applies a nonlinear activation \\(g\\). The bias shifts the activation boundary. Without the nonlinearity, the entire transformation remains linear, limiting expressiveness.",
  },

  {
    id: "mit6s191-l1-q03",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about activation functions are correct?",
    options: [
      {
        text: "Activation functions introduce nonlinearity into the model.",
        isCorrect: true,
      },
      {
        text: "The sigmoid function maps real numbers to values between 0 and 1.",
        isCorrect: true,
      },
      {
        text: "ReLU (Rectified Linear Unit) is piecewise linear with a nonlinearity at 0.",
        isCorrect: true,
      },
      {
        text: "Stacking linear layers without nonlinearities increases model expressiveness arbitrarily.",
        isCorrect: false,
      },
    ],
    explanation:
      "Activation functions allow neural networks to model nonlinear relationships. Sigmoid squashes values to (0,1), while ReLU is linear for positive values and zero otherwise. Without nonlinearities, stacking linear layers collapses into a single linear transformation.",
  },

  {
    id: "mit6s191-l1-q04",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Why are nonlinear activation functions essential in deep neural networks?",
    options: [
      {
        text: "They allow networks to approximate complex nonlinear functions.",
        isCorrect: true,
      },
      {
        text: "They prevent the model from collapsing into a single linear transformation.",
        isCorrect: true,
      },
      {
        text: "They make it possible to separate data that is not linearly separable.",
        isCorrect: true,
      },
      {
        text: "They guarantee that gradient descent will always converge to the global minimum.",
        isCorrect: false,
      },
    ],
    explanation:
      "Nonlinearities are what give deep networks their expressive power. Without them, multiple layers reduce to a single linear mapping. However, nonlinear activations do not guarantee convergence to a global minimum—optimization remains difficult.",
  },

  {
    id: "mit6s191-l1-q05",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "In gradient descent, the weight update rule is \\( w \\leftarrow w - \\eta \\frac{\\partial J}{\\partial w} \\). Which statements are correct?",
    options: [
      {
        text: "\\( \\frac{\\partial J}{\\partial w} \\) indicates the direction of steepest increase of the loss.",
        isCorrect: true,
      },
      {
        text: "The negative sign ensures movement toward decreasing loss.",
        isCorrect: true,
      },
      {
        text: "The learning rate \\(\\eta\\) controls the step size taken in parameter space.",
        isCorrect: true,
      },
      {
        text: "A sufficiently large learning rate always speeds up convergence without risk.",
        isCorrect: false,
      },
    ],
    explanation:
      "The gradient points in the direction of steepest ascent. Subtracting it moves downhill. The learning rate scales the step size. If too large, updates may overshoot and diverge rather than converge.",
  },

  {
    id: "mit6s191-l1-q06",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statement best defines intelligence as described in the lecture?",
    options: [
      {
        text: "The ability to process information in order to inform future decisions.",
        isCorrect: true,
      },
      {
        text: "The ability to memorize large amounts of training data.",
        isCorrect: false,
      },
      {
        text: "The ability to perform matrix multiplication efficiently.",
        isCorrect: false,
      },
      {
        text: "The ability to execute code without human intervention.",
        isCorrect: false,
      },
    ],
    explanation:
      "Intelligence was defined as the ability to process information to inform future decisions. Memorization or computation speed alone does not constitute intelligence in this framing.",
  },

  {
    id: "mit6s191-l1-q07",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which of the following are true about backpropagation?",
    options: [
      {
        text: "It relies on repeated application of the chain rule.",
        isCorrect: true,
      },
      {
        text: "It computes gradients of the loss with respect to each weight.",
        isCorrect: true,
      },
      {
        text: "It propagates gradients from output layers backward toward input layers.",
        isCorrect: true,
      },
      {
        text: "It guarantees that the model will not overfit.",
        isCorrect: false,
      },
    ],
    explanation:
      "Backpropagation uses the chain rule to compute gradients efficiently across many layers. It enables weight updates for optimization. However, it does not address overfitting—that requires regularization or data considerations.",
  },

  {
    id: "mit6s191-l1-q08",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose a network predicts a continuous output and uses Mean Squared Error (MSE) loss \\( J = \\frac{1}{N} \\sum_{i=1}^N (y_i - \\hat{y}_i)^2 \\). Which statements are correct?",
    options: [
      {
        text: "MSE penalizes larger errors more strongly due to squaring.",
        isCorrect: true,
      },
      {
        text: "MSE is appropriate for regression tasks.",
        isCorrect: true,
      },
      {
        text: "MSE directly models probabilities for binary classification.",
        isCorrect: false,
      },
      {
        text: "Minimizing MSE is equivalent to maximizing likelihood under Gaussian noise assumptions.",
        isCorrect: true,
      },
    ],
    explanation:
      "MSE squares errors, amplifying larger deviations. It is commonly used in regression. Under Gaussian noise assumptions, minimizing MSE corresponds to maximum likelihood estimation. It is not ideal for modeling probabilities in binary classification.",
  },

  {
    id: "mit6s191-l1-q09",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about stochastic gradient descent (SGD) are true?",
    options: [
      {
        text: "It computes gradients using a single randomly selected data point.",
        isCorrect: true,
      },
      {
        text: "It is generally noisier than full-batch gradient descent.",
        isCorrect: true,
      },
      {
        text: "It can lead to faster iterations compared to full-batch methods.",
        isCorrect: true,
      },
      {
        text: "It always converges faster in terms of wall-clock time.",
        isCorrect: false,
      },
    ],
    explanation:
      "SGD uses a single sample per update, making gradients noisy but cheap to compute. This often allows faster iterations. However, convergence behavior depends on problem and hyperparameters.",
  },

  {
    id: "mit6s191-l1-q10",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statement about mini-batch gradient descent is correct?",
    options: [
      {
        text: "It computes gradients over small subsets of the dataset.",
        isCorrect: true,
      },
      {
        text: "It requires using the entire dataset at every update step.",
        isCorrect: false,
      },
      {
        text: "It eliminates all gradient noise.",
        isCorrect: false,
      },
      {
        text: "It prevents overfitting automatically.",
        isCorrect: false,
      },
    ],
    explanation:
      "Mini-batch gradient descent uses small subsets (e.g., 32 or 128 samples). It balances stability and efficiency. It does not eliminate noise nor automatically prevent overfitting.",
  },

  {
    id: "mit6s191-l1-q11",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which of the following are true about the empirical loss \\( J(\\mathbf{w}) = \\frac{1}{N} \\sum_{i=1}^N L_i(\\mathbf{w}) \\)?",
    options: [
      {
        text: "It averages loss across all training samples.",
        isCorrect: true,
      },
      {
        text: "Minimizing it is equivalent to minimizing each individual sample loss independently.",
        isCorrect: false,
      },
      {
        text: "It serves as a proxy for performance on unseen data.",
        isCorrect: true,
      },
      {
        text: "It depends on the model parameters \\(\\mathbf{w}\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "Empirical loss averages over the dataset. Minimizing it does not imply each sample loss is minimized independently—there are tradeoffs. It is used as a proxy for generalization but does not guarantee it.",
  },

  {
    id: "mit6s191-l1-q12",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe overfitting?",
    options: [
      {
        text: "It occurs when a model memorizes training data and performs poorly on new data.",
        isCorrect: true,
      },
      {
        text: "It is more likely when model capacity is high relative to dataset size.",
        isCorrect: true,
      },
      {
        text: "It typically results in low training loss and high test loss.",
        isCorrect: true,
      },
      {
        text: "It means the model cannot fit the training data at all.",
        isCorrect: false,
      },
    ],
    explanation:
      "Overfitting happens when the model captures noise or specific details of training data. This leads to low training loss but poor generalization. It is common when models are too expressive for the data size.",
  },

  {
    id: "mit6s191-l1-q13",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements about dropout are correct?",
    options: [
      {
        text: "During training, dropout randomly sets some activations to zero.",
        isCorrect: true,
      },
      {
        text: "Dropout reduces model capacity during training.",
        isCorrect: true,
      },
      {
        text: "Dropout forces the model to rely on multiple pathways.",
        isCorrect: true,
      },
      {
        text: "Dropout guarantees perfect generalization.",
        isCorrect: false,
      },
    ],
    explanation:
      "Dropout introduces stochasticity by zeroing activations during training. This discourages reliance on single neurons. While it improves robustness, it does not guarantee perfect generalization.",
  },

  {
    id: "mit6s191-l1-q14",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about learning rate are correct?",
    options: [
      {
        text: "A very small learning rate can cause slow convergence.",
        isCorrect: true,
      },
      {
        text: "A very large learning rate can cause divergence.",
        isCorrect: true,
      },
      {
        text: "Adaptive optimizers adjust learning rates based on gradient information.",
        isCorrect: true,
      },
      {
        text: "The optimal learning rate is always the same across all tasks.",
        isCorrect: false,
      },
    ],
    explanation:
      "Learning rate determines update size. Too small slows learning; too large destabilizes it. Adaptive methods like Adam adjust learning rates dynamically. There is no universal best learning rate.",
  },

  {
    id: "mit6s191-l1-q15",
    chapter: 1,
    difficulty: "hard",
    prompt: "Why has deep learning become dominant in recent years?",
    options: [
      {
        text: "Increased availability of large datasets.",
        isCorrect: true,
      },
      {
        text: "Advances in GPU hardware enabling parallel computation.",
        isCorrect: true,
      },
      {
        text: "Improved software frameworks like TensorFlow and PyTorch.",
        isCorrect: true,
      },
      {
        text: "Because neural networks were invented recently.",
        isCorrect: false,
      },
    ],
    explanation:
      "Neural networks date back decades. Their recent dominance is driven by big data, powerful GPUs, and mature open-source frameworks. These factors made deep models practical at scale.",
  },

  {
    id: "mit6s191-l1-q16",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statement about dense (fully connected) layers is correct?",
    options: [
      {
        text: "Every input unit connects to every output unit.",
        isCorrect: true,
      },
      {
        text: "Each output depends only on one input feature.",
        isCorrect: false,
      },
      {
        text: "Dense layers cannot include bias terms.",
        isCorrect: false,
      },
      {
        text: "Dense layers must always use sigmoid activation.",
        isCorrect: false,
      },
    ],
    explanation:
      "In a dense layer, each output is connected to all inputs. Bias terms are typically included. Activation functions are flexible and task-dependent.",
  },

  {
    id: "mit6s191-l1-q17",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about hierarchical feature learning are correct?",
    options: [
      {
        text: "Lower layers often learn simple features like edges.",
        isCorrect: true,
      },
      {
        text: "Higher layers combine simpler features into complex structures.",
        isCorrect: true,
      },
      {
        text: "Hierarchical learning reduces the need for hand-engineered features.",
        isCorrect: true,
      },
      {
        text: "Feature hierarchies prevent any classification errors.",
        isCorrect: false,
      },
    ],
    explanation:
      "Deep networks learn layered representations, from edges to parts to objects. This reduces reliance on manual feature engineering. However, errors can still occur.",
  },

  {
    id: "mit6s191-l1-q18",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements about loss landscapes are correct?",
    options: [
      {
        text: "They are typically high-dimensional in deep networks.",
        isCorrect: true,
      },
      {
        text: "They can contain multiple local minima and saddle points.",
        isCorrect: true,
      },
      {
        text: "Gradient descent follows local gradient information only.",
        isCorrect: true,
      },
      {
        text: "All loss landscapes are convex in deep learning.",
        isCorrect: false,
      },
    ],
    explanation:
      "Deep models have extremely high-dimensional parameter spaces. Their loss surfaces are complex and non-convex. Gradient descent only uses local slope information, not global structure.",
  },

  {
    id: "mit6s191-l1-q19",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about early stopping are correct?",
    options: [
      {
        text: "It monitors validation performance during training.",
        isCorrect: true,
      },
      {
        text: "It selects the model checkpoint with best validation performance.",
        isCorrect: true,
      },
      {
        text: "It prevents the training loss from decreasing.",
        isCorrect: false,
      },
      {
        text: "It can reduce overfitting.",
        isCorrect: true,
      },
    ],
    explanation:
      "Early stopping uses a validation set to detect when performance stops improving. It does not prevent training loss from decreasing but halts training when generalization begins to degrade.",
  },

  {
    id: "mit6s191-l1-q20",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statement about deep neural networks is correct?",
    options: [
      {
        text: "They are constructed by stacking linear transformations and nonlinear activations.",
        isCorrect: true,
      },
      {
        text: "They eliminate the need for any training data.",
        isCorrect: false,
      },
      {
        text: "They always outperform simpler models.",
        isCorrect: false,
      },
      {
        text: "They guarantee perfect predictions with enough layers.",
        isCorrect: false,
      },
    ],
    explanation:
      "Deep networks are built by composing linear layers and nonlinearities. They require training data and careful optimization. While powerful, they do not guarantee perfect performance.",
  },

  {
    id: "mit6s191-l1-q21",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Consider a perceptron with two inputs and sigmoid activation: \\( y = \\sigma(w_1 x_1 + w_2 x_2 + b) \\). Which statements are correct?",
    options: [
      {
        text: "The decision boundary before applying the sigmoid is linear in \\(x_1, x_2\\).",
        isCorrect: true,
      },
      {
        text: "The sigmoid transforms the linear output into a value between 0 and 1.",
        isCorrect: true,
      },
      {
        text: "The decision boundary corresponds to \\( w_1 x_1 + w_2 x_2 + b = 0 \\).",
        isCorrect: true,
      },
      {
        text: "The model can represent arbitrary nonlinear boundaries with a single perceptron.",
        isCorrect: false,
      },
    ],
    explanation:
      "Before the sigmoid, the model computes a linear function of the inputs. The decision boundary occurs when the argument equals zero. The sigmoid only squashes outputs; it does not make the boundary nonlinear. A single perceptron cannot represent arbitrary nonlinear boundaries.",
  },

  {
    id: "mit6s191-l1-q22",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe deep neural networks?",
    options: [
      {
        text: "They increase expressive capacity by stacking multiple layers.",
        isCorrect: true,
      },
      {
        text: "Each additional nonlinear layer increases representational power.",
        isCorrect: true,
      },
      {
        text: "Depth enables hierarchical feature composition.",
        isCorrect: true,
      },
      {
        text: "Adding depth guarantees improved performance on all datasets.",
        isCorrect: false,
      },
    ],
    explanation:
      "Depth increases representational capacity and enables hierarchical features. However, more depth does not automatically guarantee better performance. Optimization and generalization challenges remain.",
  },

  {
    id: "mit6s191-l1-q23",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose \\( J(w) \\) is differentiable. Which statements about gradient descent are correct?",
    options: [
      {
        text: "Gradient descent moves in the direction of steepest local decrease.",
        isCorrect: true,
      },
      {
        text: "It relies only on first-order derivative information.",
        isCorrect: true,
      },
      {
        text: "It can converge to local minima in non-convex problems.",
        isCorrect: true,
      },
      {
        text: "It always finds the global minimum for deep networks.",
        isCorrect: false,
      },
    ],
    explanation:
      "Gradient descent uses first-order derivatives and follows the negative gradient. In non-convex landscapes like deep networks, it can converge to local minima or saddle points. There is no guarantee of reaching the global optimum.",
  },

  {
    id: "mit6s191-l1-q24",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statement about bias terms in neural networks is correct?",
    options: [
      {
        text: "Bias terms allow shifting the activation threshold.",
        isCorrect: true,
      },
      {
        text: "Bias terms multiply input features.",
        isCorrect: false,
      },
      {
        text: "Bias terms are only used in output layers.",
        isCorrect: false,
      },
      {
        text: "Bias terms prevent overfitting.",
        isCorrect: false,
      },
    ],
    explanation:
      "Bias terms shift the activation boundary and increase flexibility. They are additive, not multiplicative. They are used across layers and do not directly prevent overfitting.",
  },

  {
    id: "mit6s191-l1-q25",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about backpropagation in deep networks are correct?",
    options: [
      {
        text: "Gradients are computed layer by layer from output to input.",
        isCorrect: true,
      },
      {
        text: "The chain rule allows decomposition of derivatives across layers.",
        isCorrect: true,
      },
      {
        text: "Backpropagation is computationally efficient compared to naive differentiation.",
        isCorrect: true,
      },
      {
        text: "Backpropagation requires second-order derivatives.",
        isCorrect: false,
      },
    ],
    explanation:
      "Backpropagation propagates gradients backward using the chain rule. It efficiently reuses intermediate results, avoiding exponential computation. It requires first-order derivatives, not second-order.",
  },

  {
    id: "mit6s191-l1-q26",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about learning rate scheduling are correct?",
    options: [
      {
        text: "Adaptive optimizers adjust effective learning rates using gradient history.",
        isCorrect: true,
      },
      {
        text: "Adam is an example of an adaptive optimizer.",
        isCorrect: true,
      },
      {
        text: "Learning rates may change during training.",
        isCorrect: true,
      },
      {
        text: "Stochastic gradient descent automatically adapts learning rates without modification.",
        isCorrect: false,
      },
    ],
    explanation:
      "Adaptive methods like Adam use gradient statistics to adjust updates. Learning rates can vary across training. Plain SGD does not automatically adapt unless extended.",
  },

  {
    id: "mit6s191-l1-q27",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "For binary classification with softmax cross-entropy loss, which statements are correct?",
    options: [
      {
        text: "It compares predicted probabilities with true class labels.",
        isCorrect: true,
      },
      {
        text: "It penalizes confident wrong predictions strongly.",
        isCorrect: true,
      },
      {
        text: "It is appropriate for modeling categorical outputs.",
        isCorrect: true,
      },
      {
        text: "It requires outputs to be negative values.",
        isCorrect: false,
      },
    ],
    explanation:
      "Cross-entropy compares predicted probability distributions to true labels. Confident wrong predictions incur large penalties. It models categorical outcomes, and probabilities are non-negative.",
  },

  {
    id: "mit6s191-l1-q28",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statement about feature learning in deep networks is correct?",
    options: [
      {
        text: "Lower layers often learn edges and simple patterns.",
        isCorrect: true,
      },
      {
        text: "Deep networks require fully handcrafted features.",
        isCorrect: false,
      },
      {
        text: "Feature learning prevents the need for data.",
        isCorrect: false,
      },
      {
        text: "All features are manually specified.",
        isCorrect: false,
      },
    ],
    explanation:
      "Deep networks automatically learn hierarchical features from data. Early layers detect simple patterns like edges. Feature learning reduces manual engineering but still requires data.",
  },

  {
    id: "mit6s191-l1-q29",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about mini-batch size are correct?",
    options: [
      {
        text: "Larger batch sizes reduce gradient noise.",
        isCorrect: true,
      },
      {
        text: "Mini-batching enables GPU parallelization.",
        isCorrect: true,
      },
      {
        text: "Very small batches behave similarly to stochastic gradient descent.",
        isCorrect: true,
      },
      {
        text: "Batch size does not influence optimization behavior.",
        isCorrect: false,
      },
    ],
    explanation:
      "Larger batches provide more stable gradient estimates. GPUs process batches in parallel efficiently. Very small batches resemble SGD. Batch size significantly influences optimization dynamics.",
  },

  {
    id: "mit6s191-l1-q30",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about loss landscapes in deep learning are correct?",
    options: [
      {
        text: "They are typically non-convex.",
        isCorrect: true,
      },
      {
        text: "They may contain saddle points.",
        isCorrect: true,
      },
      {
        text: "Visualization is difficult due to high dimensionality.",
        isCorrect: true,
      },
      {
        text: "All local minima perform poorly compared to global minima.",
        isCorrect: false,
      },
    ],
    explanation:
      "Deep learning loss surfaces are high-dimensional and non-convex. They contain saddle points and many local minima. Interestingly, many local minima generalize similarly, so not all are poor solutions.",
  },

  {
    id: "mit6s191-l1-q31",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about overfitting and generalization are correct?",
    options: [
      {
        text: "Generalization refers to performance on unseen data.",
        isCorrect: true,
      },
      {
        text: "Overfitting often results from high model capacity and limited data.",
        isCorrect: true,
      },
      {
        text: "Validation sets help estimate generalization performance.",
        isCorrect: true,
      },
      {
        text: "Training accuracy is the ultimate goal of model development.",
        isCorrect: false,
      },
    ],
    explanation:
      "Generalization measures performance on new data. Overfitting arises from excessive capacity relative to data. Validation sets estimate generalization. Training accuracy alone is not the goal.",
  },

  {
    id: "mit6s191-l1-q32",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statement about stacking linear layers without nonlinearities is correct?",
    options: [
      {
        text: "They collapse into a single linear transformation.",
        isCorrect: true,
      },
      {
        text: "They increase expressive power exponentially.",
        isCorrect: false,
      },
      {
        text: "They allow modeling arbitrary nonlinear functions.",
        isCorrect: false,
      },
      {
        text: "They eliminate the need for bias terms.",
        isCorrect: false,
      },
    ],
    explanation:
      "Composing linear transformations results in another linear transformation. Without nonlinearities, depth does not increase expressive capacity beyond linear models.",
  },

  {
    id: "mit6s191-l1-q33",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "If \\( f(x) = g(h(x)) \\), which statements about \\( \\frac{df}{dx} \\) are correct?",
    options: [
      {
        text: "It can be computed using the chain rule: \\( \\frac{df}{dx} = g'(h(x)) \\cdot h'(x) \\).",
        isCorrect: true,
      },
      {
        text: "This principle underlies backpropagation.",
        isCorrect: true,
      },
      {
        text: "Gradients are propagated through intermediate variables.",
        isCorrect: true,
      },
      {
        text: "The chain rule only applies to linear functions.",
        isCorrect: false,
      },
    ],
    explanation:
      "The chain rule decomposes derivatives across composed functions. Backpropagation repeatedly applies this rule layer by layer. It applies broadly to differentiable functions, not just linear ones.",
  },

  {
    id: "mit6s191-l1-q34",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about dropout during inference (testing) are correct?",
    options: [
      {
        text: "Dropout is typically disabled during inference.",
        isCorrect: true,
      },
      {
        text: "All neurons are used during inference.",
        isCorrect: true,
      },
      {
        text: "Dropout reduces stochasticity during testing.",
        isCorrect: true,
      },
      {
        text: "Inference requires randomly dropping neurons again.",
        isCorrect: false,
      },
    ],
    explanation:
      "During inference, dropout is disabled so the full network is used. This ensures stable predictions. Dropout is only active during training.",
  },

  {
    id: "mit6s191-l1-q35",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements about optimization tradeoffs are correct?",
    options: [
      {
        text: "Smaller batches increase gradient noise.",
        isCorrect: true,
      },
      {
        text: "Noisy gradients can help escape shallow local minima.",
        isCorrect: true,
      },
      {
        text: "Larger batches typically provide more stable updates.",
        isCorrect: true,
      },
      {
        text: "Noise always prevents convergence.",
        isCorrect: false,
      },
    ],
    explanation:
      "Small batches introduce noise, which can sometimes help escape poor local regions. Larger batches stabilize updates. Noise does not inherently prevent convergence.",
  },

  {
    id: "mit6s191-l1-q36",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statement about neural network outputs is correct?",
    options: [
      {
        text: "Output layer design depends on the task type.",
        isCorrect: true,
      },
      {
        text: "All tasks use sigmoid outputs.",
        isCorrect: false,
      },
      {
        text: "All tasks use mean squared error loss.",
        isCorrect: false,
      },
      {
        text: "Output layers cannot contain bias terms.",
        isCorrect: false,
      },
    ],
    explanation:
      "Output layers vary depending on regression vs classification tasks. Different losses and activations are used. Bias terms are typically included.",
  },

  {
    id: "mit6s191-l1-q37",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about model capacity are correct?",
    options: [
      {
        text: "Increasing depth increases model capacity.",
        isCorrect: true,
      },
      {
        text: "Increasing width increases model capacity.",
        isCorrect: true,
      },
      {
        text: "High capacity increases risk of overfitting.",
        isCorrect: true,
      },
      {
        text: "Capacity guarantees good generalization.",
        isCorrect: false,
      },
    ],
    explanation:
      "Depth and width both increase expressive capacity. However, excessive capacity relative to data increases overfitting risk. Capacity alone does not guarantee generalization.",
  },

  {
    id: "mit6s191-l1-q38",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements about empirical risk minimization are correct?",
    options: [
      {
        text: "It optimizes average loss over training data.",
        isCorrect: true,
      },
      {
        text: "It approximates minimizing expected loss over the true data distribution.",
        isCorrect: true,
      },
      {
        text: "It assumes training and test data are drawn from similar distributions.",
        isCorrect: true,
      },
      {
        text: "It guarantees optimal performance on any distribution shift.",
        isCorrect: false,
      },
    ],
    explanation:
      "Empirical risk minimization minimizes average training loss. It approximates minimizing expected loss if data is representative. It assumes train and test distributions match. It does not handle distribution shifts automatically.",
  },

  {
    id: "mit6s191-l1-q39",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about GPUs in deep learning are correct?",
    options: [
      {
        text: "They accelerate matrix multiplications through parallel computation.",
        isCorrect: true,
      },
      {
        text: "They enable efficient mini-batch training.",
        isCorrect: true,
      },
      {
        text: "They were a key factor in the rise of modern deep learning.",
        isCorrect: true,
      },
      {
        text: "They eliminate the need for optimization algorithms.",
        isCorrect: false,
      },
    ],
    explanation:
      "GPUs perform parallel matrix operations efficiently. This enables scalable mini-batch training. Their availability significantly boosted deep learning progress. Optimization algorithms remain necessary.",
  },

  {
    id: "mit6s191-l1-q40",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about deep learning training dynamics are correct?",
    options: [
      {
        text: "Training involves iteratively updating weights to reduce loss.",
        isCorrect: true,
      },
      {
        text: "Gradients depend on both model parameters and data.",
        isCorrect: true,
      },
      {
        text: "Optimization and generalization are distinct challenges.",
        isCorrect: true,
      },
      {
        text: "Once training loss decreases, generalization is guaranteed.",
        isCorrect: false,
      },
    ],
    explanation:
      "Training updates parameters iteratively using gradients. Gradients depend on current weights and input data. Even if training loss decreases, generalization is not guaranteed because overfitting may occur.",
  },
];
