import { Question } from "../../quiz";

// lib/lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L2_Training Deep NNs.ts

export const L2_TrainingDeepNNs: Question[] = [
  {
    id: "mit15773-l2-q01",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "When designing a neural network for a supervised learning task, which statements are typically true about what is fixed by the problem versus chosen by the practitioner?",
    options: [
      {
        text: "The input and output are usually dictated by the problem setting (what features you have and what target you must predict).",
        isCorrect: true,
      },
      {
        text: "The number of hidden layers is a design choice the practitioner can vary.",
        isCorrect: true,
      },
      {
        text: "The number of neurons (units) per hidden layer is a design choice the practitioner can vary.",
        isCorrect: true,
      },
      {
        text: "The activation functions used in hidden and output layers can be chosen, though the output activation is constrained by what the output should represent.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture emphasizes a clear split: your dataset and prediction target constrain inputs and outputs, but you choose the architecture in the middle. Hidden-layer depth/width and hidden activations are flexible, while the output layer/activation must match the output type (e.g., probability vs numeric).",
  },

  {
    id: "mit15773-l2-q02",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "In the heart disease example, why can 13 original input variables become 29 input nodes to the neural network?",
    options: [
      {
        text: "Categorical variables can be one-hot encoded, which expands one variable into multiple binary indicator columns.",
        isCorrect: true,
      },
      {
        text: "One-hot encoding can increase dimensionality when a categorical feature has multiple possible levels.",
        isCorrect: true,
      },
      {
        text: "After one-hot encoding, the model sees a longer input vector whose length equals the total number of resulting numeric columns.",
        isCorrect: true,
      },
      {
        text: "One-hot encoding reduces every categorical variable into a single continuous feature, which increases input size.",
        isCorrect: false,
      },
    ],
    explanation:
      "One-hot encoding turns a categorical variable with multiple levels into multiple 0/1 columns, increasing the number of input features. That’s why the lecture’s heart-disease dataset goes from 13 original variables to 29 numeric inputs after encoding.",
  },

  {
    id: "mit15773-l2-q03",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "A dense network has 29 inputs, 16 hidden units, and 1 sigmoid output, with a bias for every hidden unit and the output unit. Which statements about parameter counting are correct?",
    options: [
      {
        text: "Input→hidden contributes \\(29\\times 16\\) weights.",
        isCorrect: true,
      },
      {
        text: "Hidden layer contributes 16 bias parameters (one per hidden unit).",
        isCorrect: true,
      },
      {
        text: "Hidden→output contributes \\(16\\times 1\\) weights and 1 output bias.",
        isCorrect: true,
      },
      {
        text: "The total number of parameters is \\(29\\times 16 + 16 + 16\\times 1 + 1 = 497\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "In a fully connected (dense) layer, weights scale as (in units) × (out units), and biases add one per neuron. Summing input→hidden weights, hidden biases, hidden→output weights, and the output bias yields the lecture’s total of 497 parameters.",
  },

  {
    id: "mit15773-l2-q04",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe how the lecture defines a model in Keras for the heart disease network?",
    options: [
      {
        text: "An input layer can be created with something like `keras.Input(shape=29)` to indicate a 29-dimensional input vector.",
        isCorrect: true,
      },
      {
        text: 'A hidden dense layer with ReLU can be defined as something like `keras.layers.Dense(16, activation="relu")` and then applied to the prior layer’s output.',
        isCorrect: true,
      },
      {
        text: 'A binary classification output can be represented with a single unit using a sigmoid activation, e.g. `Dense(1, activation="sigmoid")`.',
        isCorrect: true,
      },
      {
        text: "A Keras model object can be created by specifying its input tensor and output tensor, e.g. `keras.Model(input, output)`.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture shows a left-to-right wiring: define input → define hidden Dense(16, ReLU) → define output Dense(1, sigmoid). Finally, you wrap the computational graph into a `keras.Model` so Keras can train/evaluate/predict using that model object.",
  },

  {
    id: "mit15773-l2-q05",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe what “training” means for linear regression, logistic regression, and deep neural networks?",
    options: [
      {
        text: "Training means finding parameter values (weights/biases or coefficients/intercepts) that make predictions close to targets on the training data.",
        isCorrect: true,
      },
      {
        text: "In linear and logistic regression, training is often done by optimization routines hidden inside tools like `lm` or `glm` (or equivalents).",
        isCorrect: true,
      },
      {
        text: "Training a deep neural network follows the same high-level idea as regression: choose parameters to minimize prediction discrepancy, just with many more parameters.",
        isCorrect: true,
      },
      {
        text: "Training means changing the dataset values \\(x\\) and \\(y\\) until the model fits, rather than changing the model parameters.",
        isCorrect: false,
      },
    ],
    explanation:
      "Across regression and deep nets, training is parameter estimation: adjust weights/biases to reduce error according to a chosen loss function. The data \\(x\\) and labels \\(y\\) are fixed observations; you optimize the model parameters that determine the predictions.",
  },

  {
    id: "mit15773-l2-q06",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about loss functions are correct?",
    options: [
      {
        text: "A loss function quantifies how “bad” a model’s predictions are compared to the actual targets.",
        isCorrect: true,
      },
      {
        text: "If predictions are close to actual values, the loss is typically small.",
        isCorrect: true,
      },
      {
        text: "A perfect model (on the data being evaluated) would have loss exactly 0 under many common loss definitions.",
        isCorrect: true,
      },
      {
        text: "A loss function is only defined for classification and cannot be used for regression.",
        isCorrect: false,
      },
    ],
    explanation:
      "A loss function is a numeric score of mismatch between prediction and target: smaller is better, and many losses reach 0 when predictions match targets exactly. Loss functions exist for both regression (e.g., Mean Squared Error) and classification (e.g., Binary Cross-Entropy).",
  },

  {
    id: "mit15773-l2-q07",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe Mean Squared Error (MSE) loss and when it is commonly used?",
    options: [
      {
        text: "MSE is commonly used when the model output is a general numerical value (e.g., predicting a continuous quantity).",
        isCorrect: true,
      },
      {
        text: "A typical MSE form averages squared differences \\(\\frac{1}{n}\\sum_{i=1}^n (y^{(i)}-\\hat{y}^{(i)})^2\\).",
        isCorrect: true,
      },
      {
        text: "MSE uses squared errors, which penalizes larger deviations more strongly than smaller deviations.",
        isCorrect: true,
      },
      {
        text: "MSE is the standard loss for a probability output in \\((0,1)\\) against a binary label because it is always the best choice for classification.",
        isCorrect: false,
      },
    ],
    explanation:
      "MSE is a go-to loss for predicting numeric targets and is easy to interpret as average squared prediction error. For classification with probabilistic outputs, other losses (like cross-entropy) are often a better match because they penalize confident wrong predictions more sharply and align with probability modeling.",
  },

  {
    id: "mit15773-l2-q08",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "For binary classification with target \\(y\\in\\{0,1\\}\\) and predicted probability \\(p\\in(0,1)\\), which statements correctly reflect the intuition behind using a log-based loss?",
    options: [
      {
        text: "If \\(y=1\\), predicting a very small \\(p\\) should incur a large loss, because the model is confidently wrong about a positive case.",
        isCorrect: true,
      },
      {
        text: "If \\(y=0\\), predicting a very large \\(p\\) should incur a large loss, because the model is confidently wrong about a negative case.",
        isCorrect: true,
      },
      {
        text: "Using \\(-\\log(p)\\) for \\(y=1\\) makes the loss grow rapidly as \\(p\\) approaches 0.",
        isCorrect: true,
      },
      {
        text: "Using a log-based loss guarantees the model will never overfit, regardless of architecture size.",
        isCorrect: false,
      },
    ],
    explanation:
      "The log creates a steep penalty for confident mistakes: when \\(p\\) is tiny for a true positive, \\(-\\log(p)\\) becomes large, and similarly \\(-\\log(1-p)\\) becomes large when \\(p\\) is near 1 for a true negative. This choice doesn’t prevent overfitting by itself; overfitting depends on model capacity, data, and training setup.",
  },

  {
    id: "mit15773-l2-q09",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which expression correctly represents the per-example Binary Cross-Entropy (BCE) loss for a binary label \\(y\\in\\{0,1\\}\\) and predicted probability \\(p\\)?",
    options: [
      {
        text: "\\(\\ell(y,p) = -\\,y\\log(p) - (1-y)\\log(1-p)\\)",
        isCorrect: true,
      },
      {
        text: "\\(\\ell(y,p) = (y-p)^2\\)",
        isCorrect: false,
      },
      {
        text: "\\(\\ell(y,p) = -\\log(y-p)\\)",
        isCorrect: false,
      },
      {
        text: "\\(\\ell(y,p) = y\\log(1-p) + (1-y)\\log(p)\\)",
        isCorrect: false,
      },
    ],
    explanation:
      "Binary cross-entropy combines the two log penalties into one smooth expression: the \\(y\\log(p)\\) term activates when \\(y=1\\), and the \\((1-y)\\log(1-p)\\) term activates when \\(y=0\\). This avoids an explicit IF–THEN case split while keeping the desired penalty shape.",
  },

  {
    id: "mit15773-l2-q10",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about minimizing functions with derivatives match the lecture’s gradient-descent intuition in one dimension?",
    options: [
      {
        text: "If \\(\\frac{dg(w)}{dw} > 0\\) at a point, increasing \\(w\\) slightly will increase \\(g(w)\\), so to reduce \\(g\\) you should decrease \\(w\\) slightly.",
        isCorrect: true,
      },
      {
        text: "If \\(\\frac{dg(w)}{dw} < 0\\) at a point, increasing \\(w\\) slightly will decrease \\(g(w)\\), so to reduce \\(g\\) you should increase \\(w\\) slightly.",
        isCorrect: true,
      },
      {
        text: "If \\(\\frac{dg(w)}{dw}\\approx 0\\), small changes in \\(w\\) won’t change \\(g(w)\\) much, so you may stop the descent (or be near a stationary point).",
        isCorrect: true,
      },
      {
        text: "The derivative tells you the value of \\(w\\) that minimizes \\(g\\) directly without needing any iteration.",
        isCorrect: false,
      },
    ],
    explanation:
      "The derivative gives local slope information: it tells you how \\(g\\) changes for a small move in \\(w\\). Gradient descent uses this local slope repeatedly (iteratively) to step downhill; it doesn’t magically output the minimizer in one shot for complex functions.",
  },

  {
    id: "mit15773-l2-q11",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly interpret the learning rate \\(\\alpha\\) in gradient descent updates like \\(w \\leftarrow w - \\alpha\\,\\frac{dg(w)}{dw}\\)?",
    options: [
      {
        text: "The learning rate scales how large a step you take along the (negative) derivative/gradient direction.",
        isCorrect: true,
      },
      {
        text: "If \\(\\alpha\\) is too large, you can overshoot and behave erratically because the derivative is only locally informative.",
        isCorrect: true,
      },
      {
        text: "If \\(\\alpha\\) is too small, progress can be very slow even if the gradient points in a useful direction.",
        isCorrect: true,
      },
      {
        text: "The learning rate is the number of hidden units in the network.",
        isCorrect: false,
      },
    ],
    explanation:
      "The learning rate controls step size. Large \\(\\alpha\\) can jump past good regions or cause divergence, while tiny \\(\\alpha\\) may converge very slowly. It’s a hyperparameter you tune; it’s not an architectural parameter like layer width.",
  },

  {
    id: "mit15773-l2-q12",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which historical fact about gradient descent is presented in the lecture?",
    options: [
      {
        text: "Gradient descent was introduced in the 19th century (1847) by Augustin-Louis Cauchy.",
        isCorrect: true,
      },
      {
        text: "Gradient descent was first invented in 2012 to train AlexNet.",
        isCorrect: false,
      },
      {
        text: "Gradient descent requires GPUs to exist; it could not be defined before modern hardware.",
        isCorrect: false,
      },
      {
        text: "Gradient descent was discovered by Isaac Newton in 1687 as part of universal gravitation.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture notes that gradient descent dates back to Cauchy in 1847, which is striking given how central it still is in training modern deep learning systems. Modern compute makes it practical at scale, but the core idea predates GPUs by a long time.",
  },

  {
    id: "mit15773-l2-q13",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "For a multivariable function \\(g(w_1,w_2)\\), which statements correctly describe the gradient and its interpretation?",
    options: [
      {
        text: "The gradient is the vector of partial derivatives, e.g. \\(\\nabla g = \\big[\\frac{\\partial g}{\\partial w_1},\\frac{\\partial g}{\\partial w_2}\\big]\\).",
        isCorrect: true,
      },
      {
        text: "The component \\(\\frac{\\partial g}{\\partial w_1}\\) measures how \\(g\\) changes for a small increase in \\(w_1\\) while holding \\(w_2\\) fixed.",
        isCorrect: true,
      },
      {
        text: "Gradient descent in multiple dimensions can be written compactly as \\(\\mathbf{w} \\leftarrow \\mathbf{w} - \\alpha\\nabla g(\\mathbf{w})\\).",
        isCorrect: true,
      },
      {
        text: "The gradient is a single number even when there are billions of parameters; it does not scale with the number of variables.",
        isCorrect: false,
      },
    ],
    explanation:
      "In multiple dimensions, the gradient has one component per parameter, so with many parameters it becomes a very long vector. Each component is a partial derivative holding other coordinates fixed, and the update subtracts a scaled gradient vector to step downhill.",
  },

  {
    id: "mit15773-l2-q14",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Gradient descent can stop at undesirable stationary points. Which statements are correct about what can happen and why practitioners often don’t panic?",
    options: [
      {
        text: "Gradient descent can converge to a local minimum that is not the global minimum.",
        isCorrect: true,
      },
      {
        text: "Gradient descent can also get near saddle points where the gradient is small even though the point is not a true minimum in every direction.",
        isCorrect: true,
      },
      {
        text: "In large neural networks, finding a “good enough” solution can still yield excellent predictive performance even if it is not a global minimum.",
        isCorrect: true,
      },
      {
        text: "If you do not reach the global minimum of training loss, the model is guaranteed to be useless.",
        isCorrect: false,
      },
    ],
    explanation:
      "Nonconvex optimization landscapes can contain local minima and saddle points, so the endpoint depends on initialization, data, and the optimizer path. In practice, neural networks often work extremely well with solutions that are not global minima, and chasing the absolute minimum can even increase overfitting risk.",
  },

  {
    id: "mit15773-l2-q15",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "When minimizing the Binary Cross-Entropy loss over a dataset, which quantities are the optimization variables?",
    options: [
      {
        text: "The neural network parameters (weights and biases) inside \\(\\mathrm{model}(x)\\).",
        isCorrect: true,
      },
      {
        text: "The input features \\(x\\) are optimization variables we update to reduce the loss.",
        isCorrect: false,
      },
      {
        text: "The labels \\(y\\) are optimization variables we adjust so the loss becomes small.",
        isCorrect: false,
      },
      {
        text: "The number of data points \\(n\\) is optimized to minimize the loss; we change \\(n\\) during training.",
        isCorrect: false,
      },
    ],
    explanation:
      "The dataset \\((x,y)\\) is treated as fixed. The parameters that change during training are the model’s weights and biases, which determine \\(\\mathrm{model}(x)\\) and therefore the loss value.",
  },

  {
    id: "mit15773-l2-q16",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe what 'backprop' does and why it was crucial for deep learning?",
    options: [
      {
        text: "Backprop is an efficient method for computing the gradient of the loss with respect to all parameters in a layered neural network.",
        isCorrect: true,
      },
      {
        text: "Backprop leverages the layer-by-layer structure by organizing computation as a computational graph and reusing intermediate results to avoid redundant work.",
        isCorrect: true,
      },
      {
        text: "Backprop reduces gradient computation largely to sequences of matrix multiplications and simple operations that can be accelerated on Graphics Processing Units (GPUs).",
        isCorrect: true,
      },
      {
        text: "Backprop replaces gradient descent; once you have backprop, you no longer need an optimizer step like \\(w\\leftarrow w-\\alpha\\nabla\\ell\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Backprop is about computing gradients efficiently; gradient descent (or another optimizer) is about using those gradients to update parameters. Backprop + GPU acceleration made it feasible to train large networks because the required operations (notably matrix multiplications) can be performed very quickly in parallel.",
  },

  {
    id: "mit15773-l2-q17",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe Stochastic Gradient Descent (SGD) and minibatches as presented in the lecture?",
    options: [
      {
        text: "Instead of computing gradients using all \\(n\\) data points each step, SGD uses only a small random subset (a minibatch) to approximate the gradient.",
        isCorrect: true,
      },
      {
        text: "Using minibatches can make each update much cheaper, enabling training on very large datasets that might not fit in memory all at once.",
        isCorrect: true,
      },
      {
        text: "Because minibatch gradients are noisy approximations, SGD can sometimes help escape local minima or avoid getting stuck.",
        isCorrect: true,
      },
      {
        text: "In the strict theoretical definition, SGD can mean using a single example per update, but in practice “SGD” is often used to describe minibatch training.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture describes using small random minibatches (e.g., 32 or 64 examples) per update, which is computationally efficient and introduces helpful noise. Strictly speaking, SGD can mean batch size 1, but practitioners often say “SGD” even when they use minibatches.",
  },

  {
    id: "mit15773-l2-q18",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statement best characterizes the relationship between gradient descent and SGD for large datasets?",
    options: [
      {
        text: "SGD replaces full-dataset gradient computation with minibatch-based gradient approximations to reduce cost per update.",
        isCorrect: true,
      },
      {
        text: "SGD requires computing exact gradients on the entire dataset at every iteration.",
        isCorrect: false,
      },
      {
        text: "SGD is only used for convex loss functions and cannot train neural networks.",
        isCorrect: false,
      },
      {
        text: "SGD is a loss function that measures prediction discrepancy.",
        isCorrect: false,
      },
    ],
    explanation:
      "Full gradient descent uses all \\(n\\) points to compute a gradient step, which can be expensive for large datasets. SGD makes training feasible by using minibatches to approximate that gradient, dramatically reducing computation per update.",
  },

  {
    id: "mit15773-l2-q19",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "In the lecture, why can increasing model capacity (e.g., more neurons than 16) sometimes reduce performance on new data in the heart disease example?",
    options: [
      {
        text: "With more parameters, a model can fit idiosyncrasies/noise in the training set more easily, which can hurt generalization to unseen data.",
        isCorrect: true,
      },
      {
        text: "A larger model always has lower test error because it is more expressive.",
        isCorrect: false,
      },
      {
        text: "Overfitting refers to when a model is too simple to capture the training patterns.",
        isCorrect: false,
      },
      {
        text: "If a model overfits, training loss must be exactly zero for every dataset.",
        isCorrect: false,
      },
    ],
    explanation:
      "More capacity increases the risk of overfitting: the model may learn patterns that don’t generalize beyond the training data. The lecture mentions this as a reason performance can worsen when the network is made larger than needed, even if training performance improves.",
  },

  {
    id: "mit15773-l2-q20",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "The lecture mentions that SGD has many 'flavors' and names one default used in the course. Which statements are correct?",
    options: [
      {
        text: "There are multiple variants ('siblings') of SGD that modify how gradients are scaled or accumulated across steps.",
        isCorrect: true,
      },
      {
        text: "Adam is a widely used optimizer that can be viewed as a particular flavor of minibatch SGD in practice.",
        isCorrect: true,
      },
      {
        text: "Choosing an optimizer is separate from choosing the loss function; you still need both to train a network.",
        isCorrect: true,
      },
      {
        text: "Adam eliminates the need for backprop because it computes gradients without any derivatives.",
        isCorrect: false,
      },
    ],
    explanation:
      "Optimizers like SGD variants and Adam define how parameter updates use gradients over time, while the loss function defines what you are trying to minimize. Adam still relies on gradients computed via backprop; it changes the update rule, not the need for derivatives.",
  },
  // Questions 21–40 (append to the existing array)

  {
    id: "mit15773-l2-q21",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements best reflect the lecture’s rule-of-thumb for choosing model complexity (number of layers/units) during network design?",
    options: [
      {
        text: "A practical approach is to start with the simplest network that could plausibly work and only increase complexity if needed.",
        isCorrect: true,
      },
      {
        text: "If a simple network solves the problem well enough, it is reasonable to stop rather than making it more complex.",
        isCorrect: true,
      },
      {
        text: "Adding layers/units indefinitely is always beneficial because more parameters always improve generalization.",
        isCorrect: false,
      },
      {
        text: "Trial-and-error over a small set of architectures (e.g., 4, 8, 16 units) is a common practical way to pick hidden-layer width.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture emphasizes a “start simple” mindset: use the least complex model that meets the goal, then scale up only if necessary. In practice, choosing widths like 4/8/16 and evaluating performance is common, while making networks arbitrarily large can increase overfitting risk.",
  },

  {
    id: "mit15773-l2-q22",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe why the lecture uses a sigmoid output for the heart disease task?",
    options: [
      {
        text: "The task is binary classification (heart disease yes/no), so a single output can represent \\(P(y=1\\mid x)\\).",
        isCorrect: true,
      },
      {
        text: "A sigmoid maps a real-valued logit to a number in \\((0,1)\\), which is consistent with interpreting the output as a probability.",
        isCorrect: true,
      },
      {
        text: "Using a sigmoid implies the loss must be Mean Squared Error because only MSE works with sigmoid outputs.",
        isCorrect: false,
      },
      {
        text: "A sigmoid output is appropriate when you need multiple class probabilities that sum to 1 across 10 classes.",
        isCorrect: false,
      },
    ],
    explanation:
      "For binary outcomes, a single sigmoid output naturally produces a probability-like value between 0 and 1. Multi-class problems typically use multiple outputs with a softmax, while the loss choice (e.g., cross-entropy vs MSE) depends on modeling goals and is not forced to be MSE.",
  },

  {
    id: "mit15773-l2-q23",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "For Binary Cross-Entropy loss \\(\\ell(y,p)=-y\\log(p)-(1-y)\\log(1-p)\\), which statements are correct about its behavior?",
    options: [
      {
        text: "If \\(y=1\\), the loss decreases as \\(p\\) increases toward 1, and grows large as \\(p\\) approaches 0.",
        isCorrect: true,
      },
      {
        text: "If \\(y=0\\), the loss decreases as \\(p\\) decreases toward 0, and grows large as \\(p\\) approaches 1.",
        isCorrect: true,
      },
      {
        text: "BCE strongly penalizes confident wrong predictions (e.g., \\(p\\approx 0\\) when \\(y=1\\)).",
        isCorrect: true,
      },
      {
        text: "BCE is minimized by setting \\(p=0.5\\) for every example, regardless of labels.",
        isCorrect: false,
      },
    ],
    explanation:
      "Binary cross-entropy is designed so correct confident predictions have small loss, while confident mistakes incur large loss due to the log terms. Predicting 0.5 everywhere avoids extreme penalties but cannot minimize loss when labels vary; the loss encourages matching probabilities to the data.",
  },

  {
    id: "mit15773-l2-q24",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "The lecture emphasizes using one combined expression for BCE instead of an IF–THEN definition. Which statements explain why that is useful?",
    options: [
      {
        text: "A single formula \\(-y\\log(p)-(1-y)\\log(1-p)\\) avoids case splits (\\(y=1\\) vs \\(y=0\\)) while still selecting the correct term because \\(y\\in\\{0,1\\}\\).",
        isCorrect: true,
      },
      {
        text: "Avoiding explicit IF–THEN cases makes it easier to take derivatives and optimize with gradient-based methods.",
        isCorrect: true,
      },
      {
        text: "The combined BCE form guarantees convexity for deep networks, ensuring gradient descent always finds the global minimum.",
        isCorrect: false,
      },
      {
        text: "The combined form is mathematically equivalent to choosing \\(-\\log(p)\\) when \\(y=1\\) and \\(-\\log(1-p)\\) when \\(y=0\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "Because \\(y\\) is either 0 or 1, the BCE formula automatically turns on the correct log penalty without an explicit branch. This makes the loss differentiable in a clean way and convenient for gradient-based optimization, even though deep network training remains nonconvex overall.",
  },

  {
    id: "mit15773-l2-q25",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly capture the lecture’s view of 'loss' versus 'accuracy' when evaluating a model?",
    options: [
      {
        text: "Loss is a numeric measure of how wrong the model’s predictions are according to a chosen discrepancy function (e.g., BCE or MSE).",
        isCorrect: true,
      },
      {
        text: "Lower loss generally indicates better fit under that loss definition, but it does not automatically guarantee best real-world usefulness or interpretability.",
        isCorrect: true,
      },
      {
        text: "Accuracy is a task-specific metric (e.g., fraction of correct classifications) and is not necessarily the same objective as the loss being optimized.",
        isCorrect: true,
      },
      {
        text: "If a model’s loss decreases on training data, the test loss must also decrease by the same amount.",
        isCorrect: false,
      },
    ],
    explanation:
      "The optimizer typically minimizes a loss, while practitioners often also track metrics like accuracy. Improvements in training loss do not guarantee improvements on unseen data because overfitting can occur, and real-world adoption may also depend on interpretability and other constraints.",
  },

  {
    id: "mit15773-l2-q26",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe local minima and saddle points in high-dimensional neural network loss landscapes (as discussed in the lecture)?",
    options: [
      {
        text: "In nonconvex landscapes, gradient descent can end up in a local minimum where the gradient is approximately zero.",
        isCorrect: true,
      },
      {
        text: "A saddle point can have small gradient even though it is not a true minimum in every direction.",
        isCorrect: true,
      },
      {
        text: "In very high dimensions, it may be unlikely to land at a point where the function slopes upward in every direction, so saddle-like behavior can be common.",
        isCorrect: true,
      },
      {
        text: "High dimensionality guarantees gradient descent always finds the global minimum because there are no local minima.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture highlights that nonconvex optimization can involve many local minima and saddle points. In high dimensions, the geometry can make strict local minima less central than saddle-like regions, and in practice “good enough” solutions often work very well without guaranteeing global optimality.",
  },

  {
    id: "mit15773-l2-q27",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "A student asks whether reaching the global minimum of training loss would eliminate issues like hallucinations. Which statements align with the lecture’s reasoning about training vs generalization?",
    options: [
      {
        text: "Minimizing training loss as much as possible can increase overfitting, which may worsen performance on unseen (test) data.",
        isCorrect: true,
      },
      {
        text: "The training loss is computed on the training distribution; future/test behavior can differ if the model over-specializes to training examples.",
        isCorrect: true,
      },
      {
        text: "A global minimum of training loss implies the model is perfect on every future input drawn from any distribution.",
        isCorrect: false,
      },
      {
        text: "The best solution for training loss may not be the best solution for test loss because they are effectively different objectives (different data).",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture stresses the separation between training and test performance. Even if you optimize training loss extremely well, the result can fail to generalize due to overfitting, and the loss landscape for unseen data can differ from the training data’s landscape.",
  },

  {
    id: "mit15773-l2-q28",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements are correct about what gradient descent actually needs at each iteration?",
    options: [
      {
        text: "A current parameter vector \\(\\mathbf{w}\\).",
        isCorrect: true,
      },
      {
        text: "A gradient estimate \\(\\nabla\\ell(\\mathbf{w})\\) (exact or approximate).",
        isCorrect: true,
      },
      {
        text: "A learning rate \\(\\alpha\\) to scale the step size.",
        isCorrect: true,
      },
      {
        text: "An exact closed-form solution for \\(\\arg\\min_{\\mathbf{w}} \\ell(\\mathbf{w})\\) computed symbolically.",
        isCorrect: false,
      },
    ],
    explanation:
      "Gradient descent is an iterative method: it repeatedly updates parameters using a gradient and a step size. It does not require solving the minimization problem in closed form, which is often infeasible for neural networks.",
  },

  {
    id: "mit15773-l2-q29",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly connect backprop, the chain rule, and computational graphs?",
    options: [
      {
        text: "Backprop is essentially an efficient organization of the chain rule applied to a layered computation (the network), propagating derivatives from output back to earlier layers.",
        isCorrect: true,
      },
      {
        text: "A computational graph helps reuse intermediate quantities so the same derivative components aren’t recomputed redundantly.",
        isCorrect: true,
      },
      {
        text: "Backprop computes gradients for all parameters in one pass backward through the graph (per forward pass), rather than differentiating each parameter independently from scratch.",
        isCorrect: true,
      },
      {
        text: "Backprop only works for single-layer networks; it cannot compute gradients for deep networks with many layers.",
        isCorrect: false,
      },
    ],
    explanation:
      "Backprop takes advantage of the network’s compositional structure: the chain rule naturally implies a backward flow of gradients. The computational graph stores intermediate results from the forward pass so that gradients can be computed efficiently and reused across many parameters and layers.",
  },

  {
    id: "mit15773-l2-q30",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "The lecture emphasizes GPUs as a major factor in deep learning’s practicality. Which statements are correct?",
    options: [
      {
        text: "Many operations in forward passes and backprop reduce to matrix multiplications and related linear algebra.",
        isCorrect: true,
      },
      {
        text: "GPUs are specialized for parallel computation, which makes large matrix operations much faster than on many CPUs for these workloads.",
        isCorrect: true,
      },
      {
        text: "GPUs were originally developed for graphics/video game rendering, and their strengths map well to neural network computation.",
        isCorrect: true,
      },
      {
        text: "GPUs eliminate the need for optimization; training becomes a single forward pass with no iterative updates.",
        isCorrect: false,
      },
    ],
    explanation:
      "Deep learning relies heavily on linear algebra, especially matrix multiplications, and GPUs excel at parallelizing such computations. GPUs speed up training dramatically but do not remove the need for iterative optimization and gradient-based updates.",
  },

  {
    id: "mit15773-l2-q31",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe why Keras uses `shape=` when defining inputs (e.g., `keras.Input(shape=29)`)?",
    options: [
      {
        text: "Keras may accept inputs that are not simple vectors, such as matrices or higher-dimensional tensors, so it needs a general 'shape' description.",
        isCorrect: true,
      },
      {
        text: "Using `shape=` allows the same API pattern to work for images (height × width × channels), sequences, and other structured tensors.",
        isCorrect: true,
      },
      {
        text: "In this lecture’s tabular example, the input is a vector so the shape is simply 29.",
        isCorrect: true,
      },
      {
        text: "Keras uses `shape=` because it cannot represent vectors at all; it only supports 3D images.",
        isCorrect: false,
      },
    ],
    explanation:
      "The `shape` argument generalizes across many input types: vectors, matrices, and higher-dimensional tensors. In the heart-disease tabular case, the input is just a 29-dimensional feature vector, so the shape is a single number.",
  },

  {
    id: "mit15773-l2-q32",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly distinguish the roles of the loss function and the optimizer in neural network training?",
    options: [
      {
        text: "The loss function defines what “discrepancy” means between predictions and targets (what you want to minimize).",
        isCorrect: true,
      },
      {
        text: "The optimizer specifies how parameters are updated using gradients (the update rule and related state, such as momentum-like terms).",
        isCorrect: true,
      },
      {
        text: "Backprop is the mechanism for computing gradients needed by the optimizer; it is not itself the update rule.",
        isCorrect: true,
      },
      {
        text: "Choosing Adam versus SGD changes the definition of the loss function being minimized.",
        isCorrect: false,
      },
    ],
    explanation:
      "Loss and optimizer are separate: loss defines the objective, optimizer defines the procedure for moving parameters to reduce that objective. Backprop computes gradients of the loss with respect to parameters; optimizers like Adam decide how to use those gradients to update weights over time.",
  },

  {
    id: "mit15773-l2-q33",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe stochastic gradient descent (SGD) as an approximation and why that can be beneficial?",
    options: [
      {
        text: "A minibatch gradient is an estimate of the full-dataset gradient because it averages loss over only a subset of examples.",
        isCorrect: true,
      },
      {
        text: "Because minibatches differ from step to step, SGD introduces noise into updates compared to full-batch gradient descent.",
        isCorrect: true,
      },
      {
        text: "This noise can sometimes help training by preventing the optimizer from getting stuck in certain local regions of the loss landscape.",
        isCorrect: true,
      },
      {
        text: "SGD always finds the exact same parameter values as full-batch gradient descent in the same number of steps because the gradients are identical.",
        isCorrect: false,
      },
    ],
    explanation:
      "SGD uses approximate gradients, which are cheaper to compute and introduce stochasticity. That randomness can be a feature, not a bug: it can improve exploration of the loss landscape and sometimes helps avoid poor stationary points compared to perfectly deterministic full-batch updates.",
  },

  {
    id: "mit15773-l2-q34",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "A student asks whether you should fully optimize one minibatch before moving to the next. Which statement matches the lecture’s description of minibatch training?",
    options: [
      {
        text: "Typically, you do one gradient update per minibatch (compute loss/gradient on the minibatch, update weights once, then move to the next minibatch).",
        isCorrect: true,
      },
      {
        text: "You must run gradient descent to convergence on a single minibatch before you are allowed to sample a new minibatch.",
        isCorrect: false,
      },
      {
        text: "Minibatches are only used to compute accuracy metrics; gradients are still computed on the full dataset each step.",
        isCorrect: false,
      },
      {
        text: "Minibatch training means you keep the weights separate for each minibatch and average them at the end of training.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture describes minibatch SGD as a streaming process: each minibatch yields a gradient estimate and you take one update step, then continue with the next minibatch using the updated weights. You do not maintain separate per-minibatch weight sets or 'converge' on a single minibatch.",
  },

  {
    id: "mit15773-l2-q35",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe why weight initialization is nontrivial in deep networks (as mentioned in the lecture)?",
    options: [
      {
        text: "Weights are typically initialized randomly, but the distribution depends on layer sizes to keep activations and gradients well-behaved.",
        isCorrect: true,
      },
      {
        text: "Bad initialization can contribute to vanishing gradients or exploding gradients during backprop, especially in deep networks.",
        isCorrect: true,
      },
      {
        text: "Popular initialization schemes (e.g., He initialization or Xavier/Glorot initialization) aim to stabilize signal propagation through depth.",
        isCorrect: true,
      },
      {
        text: "Initialization is irrelevant because SGD will always correct any initialization in exactly one update step.",
        isCorrect: false,
      },
    ],
    explanation:
      "Deep networks can be sensitive to initialization: if activations or gradients shrink/grow too fast across layers, training can stall or become unstable. Initialization schemes choose weight scales based on fan-in/fan-out so signals and gradients remain in a workable range early in training.",
  },

  {
    id: "mit15773-l2-q36",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "The lecture discusses a tradeoff in choosing between simpler models (like logistic regression) and more complex models (like neural networks). Which statements are correct?",
    options: [
      {
        text: "Explainability/interpretability can matter for adoption; simpler models can be easier to communicate to non-technical stakeholders.",
        isCorrect: true,
      },
      {
        text: "When predictive accuracy is the dominant priority, practitioners may prefer more complex models even if they are less interpretable.",
        isCorrect: true,
      },
      {
        text: "There are research efforts (e.g., mechanistic interpretability) that aim to provide insight into complex black-box models, so the story is not “simple vs complex” forever.",
        isCorrect: true,
      },
      {
        text: "Neural networks are always more interpretable than logistic regression because they have more layers.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture frames model choice around competing priorities: interpretability can drive simpler choices, while accuracy can justify black-box methods. Interpretability research exists, but more layers typically make models harder, not easier, to explain in simple terms.",
  },

  {
    id: "mit15773-l2-q37",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements are correct about dense layers in the lecture’s structured-data setup?",
    options: [
      {
        text: "A dense layer means each unit in a layer is connected to all units in the previous layer by default (fully connected).",
        isCorrect: true,
      },
      {
        text: "In structured/tabular data problems, dense layers are a common default baseline architecture.",
        isCorrect: true,
      },
      {
        text: "The number of weights between two dense layers scales roughly with the product of the layer widths.",
        isCorrect: true,
      },
      {
        text: "Dense connectivity usually reduces the number of parameters compared to sparse connectivity, which is why it prevents overfitting.",
        isCorrect: false,
      },
    ],
    explanation:
      "Dense layers connect all-to-all between consecutive layers, which makes parameter counts grow quickly with layer width. That expressiveness can help modeling, but it also increases overfitting risk if the model becomes too large relative to the available data.",
  },

  {
    id: "mit15773-l2-q38",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the overall training loop shown in the lecture (forward pass → loss → optimizer → update)?",
    options: [
      {
        text: "A forward pass uses current weights to produce predictions from inputs.",
        isCorrect: true,
      },
      {
        text: "A loss function compares predictions to true labels and produces a scalar loss value (often averaged over a minibatch).",
        isCorrect: true,
      },
      {
        text: "An optimizer uses gradients (computed via backprop) to update weights, typically with an update like \\(\\mathbf{w}\\leftarrow\\mathbf{w}-\\alpha\\nabla\\ell\\).",
        isCorrect: true,
      },
      {
        text: "Once you compute the loss for a minibatch, you never need to do another forward pass; training finishes immediately.",
        isCorrect: false,
      },
    ],
    explanation:
      "Training repeats: predict with current weights, compute loss, compute gradients, update weights, and then do it again on new minibatches. It’s inherently iterative because each update changes the model, and repeated exposure to data is needed to reduce loss across the dataset.",
  },

  {
    id: "mit15773-l2-q39",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A student asks whether it’s possible to penalize false negatives more than false positives in classification. Which statements are correct in the lecture’s framing?",
    options: [
      {
        text: "You can modify the loss function to weight errors differently (e.g., higher penalty for certain types of mistakes), depending on application needs.",
        isCorrect: true,
      },
      {
        text: "Binary cross-entropy as written is symmetric in the sense that it does not inherently encode different costs for false positives vs false negatives.",
        isCorrect: true,
      },
      {
        text: "Changing the loss changes what the optimizer is encouraged to do during training, which can shift the tradeoff between sensitivity and specificity.",
        isCorrect: true,
      },
      {
        text: "It is impossible to alter the loss to prefer one error type; classification costs are fixed by the sigmoid function.",
        isCorrect: false,
      },
    ],
    explanation:
      "The loss function is where you encode what you care about, including asymmetric penalties for different error types. If false negatives are more costly, you can increase their weight in the loss so the optimizer prioritizes avoiding them, though it may affect other metrics like false positives.",
  },

  {
    id: "mit15773-l2-q40",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly reflect how learning rate choice relates to 'confidence' in the local gradient direction (as explained in the lecture)?",
    options: [
      {
        text: "A larger learning rate can make sense when the local gradient direction seems reliable for a longer distance, allowing bigger steps.",
        isCorrect: true,
      },
      {
        text: "A smaller learning rate can be safer when you suspect the loss surface may curve quickly, so large steps might overshoot or require backtracking.",
        isCorrect: true,
      },
      {
        text: "Learning rate is determined primarily by the number of training examples; more data always forces a larger learning rate.",
        isCorrect: false,
      },
      {
        text: "Learning rate is a hyperparameter that often involves trial-and-error and can be part of the practical 'recipe' for training success.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture’s intuition is that the gradient is local information: if you trust it beyond a tiny neighborhood, you can step more aggressively; if not, you step cautiously. In practice, learning rates are tuned and can meaningfully affect training stability and speed.",
  },
];
