import { Question } from "../../quiz";

export const MIT15773L3DeepLearningForComputerVisionQuestions: Question[] = [
  {
    id: "mit15773-l3-q01",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe what an epoch represents during neural network training?",
    options: [
      {
        text: "An epoch is one full pass through the entire training dataset.",
        isCorrect: true,
      },
      {
        text: "In stochastic gradient descent, an epoch consists of multiple parameter updates.",
        isCorrect: true,
      },
      {
        text: "In full gradient descent, there is typically one weight update per epoch.",
        isCorrect: true,
      },
      {
        text: "An epoch refers to the number of neurons in the hidden layer.",
        isCorrect: false,
      },
    ],
    explanation:
      "An epoch means processing every training example once. In full gradient descent this leads to a single weight update after computing gradients on the full dataset. In stochastic gradient descent (SGD), the dataset is processed in batches, producing multiple updates per epoch. The number of neurons in a hidden layer is unrelated to the concept of an epoch.",
  },

  {
    id: "mit15773-l3-q02",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which of the following statements about stochastic gradient descent (SGD) are correct?",
    options: [
      {
        text: "SGD updates model parameters multiple times within a single epoch.",
        isCorrect: true,
      },
      {
        text: "SGD computes gradients using only a subset of the training data at each step.",
        isCorrect: true,
      },
      {
        text: "SGD always produces the exact same gradient as full gradient descent.",
        isCorrect: false,
      },
      {
        text: "SGD processes training data in batches or minibatches.",
        isCorrect: true,
      },
    ],
    explanation:
      "SGD estimates gradients using subsets of the data called batches. Each batch produces a gradient estimate and triggers a weight update. Because batches contain only part of the dataset, the gradient is only an approximation of the true gradient computed in full gradient descent.",
  },

  {
    id: "mit15773-l3-q03",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Assume a training set has 194 samples and the batch size is 32. Which statements are correct about the number of batches in one epoch?",
    options: [
      {
        text: "The number of batches equals \\(\\lceil 194 / 32 \\rceil\\).",
        isCorrect: true,
      },
      { text: "There will be exactly 6 batches.", isCorrect: false },
      {
        text: "The final batch may contain fewer samples than the others.",
        isCorrect: true,
      },
      {
        text: "All batches must contain exactly the same number of samples.",
        isCorrect: false,
      },
    ],
    explanation:
      "The number of batches is computed as \\(\\lceil \\text{training size} / \\text{batch size} \\rceil\\). With 194 samples and batch size 32, this yields 7 batches. The first six batches contain 32 samples and the last batch contains the remaining samples.",
  },

  {
    id: "mit15773-l3-q04",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe tensors in deep learning frameworks such as TensorFlow?",
    options: [
      {
        text: "A scalar value can be considered a tensor of rank 0.",
        isCorrect: true,
      },
      { text: "A vector is typically a rank-1 tensor.", isCorrect: true },
      { text: "A matrix is typically a rank-2 tensor.", isCorrect: true },
      {
        text: "A tensor can represent data structures with more than two dimensions.",
        isCorrect: true,
      },
    ],
    explanation:
      "Tensors generalize scalars, vectors, and matrices to arbitrary numbers of dimensions. A scalar is rank-0, a vector rank-1, and a matrix rank-2. Higher-rank tensors represent multidimensional arrays such as images or videos.",
  },

  {
    id: "mit15773-l3-q05",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements about grayscale image representation are correct?",
    options: [
      {
        text: "Each pixel is represented by a numerical intensity value.",
        isCorrect: true,
      },
      {
        text: "Pixel values typically lie between 0 and 255.",
        isCorrect: true,
      },
      { text: "Higher values correspond to brighter pixels.", isCorrect: true },
      {
        text: "Each pixel is represented by three color channels.",
        isCorrect: false,
      },
    ],
    explanation:
      "In grayscale images, each pixel has a single intensity value. These values typically range from 0 (black) to 255 (white). Color images use multiple channels such as RGB, but grayscale images use only one.",
  },

  {
    id: "mit15773-l3-q06",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which of the following statements about flattening an image before feeding it into a dense neural network are correct?",
    options: [
      { text: "Flattening converts a matrix into a vector.", isCorrect: true },
      {
        text: "Flattening changes the numerical values of the pixels.",
        isCorrect: false,
      },
      {
        text: "A \\(28 \\times 28\\) grayscale image becomes a vector of length 784.",
        isCorrect: true,
      },
      {
        text: "Flattening is required when using dense layers that expect vector inputs.",
        isCorrect: true,
      },
    ],
    explanation:
      "Flattening reorganizes image data into a one-dimensional vector without altering pixel values. For example, a \\(28 \\times 28\\) image becomes a vector with 784 elements. This allows fully connected layers to process image data.",
  },

  {
    id: "mit15773-l3-q07",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements correctly describe the softmax function?",
    options: [
      {
        text: "Softmax converts arbitrary numbers into probabilities.",
        isCorrect: true,
      },
      { text: "The outputs of softmax sum to 1.", isCorrect: true },
      {
        text: "Softmax requires each input value to already lie between 0 and 1.",
        isCorrect: false,
      },
      {
        text: "Softmax often appears in the output layer for multi-class classification.",
        isCorrect: true,
      },
    ],
    explanation:
      "Softmax transforms arbitrary real numbers into a probability distribution. It exponentiates each input and divides by the sum of all exponentials. This guarantees that outputs are positive and sum to 1, making it suitable for multi-class classification.",
  },

  {
    id: "mit15773-l3-q08",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about overfitting are correct?",
    options: [
      {
        text: "Overfitting occurs when a model learns patterns specific to the training data rather than generalizable patterns.",
        isCorrect: true,
      },
      {
        text: "Increasing model complexity can increase the risk of overfitting.",
        isCorrect: true,
      },
      {
        text: "Overfitting typically results in low training error but higher validation error.",
        isCorrect: true,
      },
      {
        text: "Overfitting means the model cannot capture patterns in the training data at all.",
        isCorrect: false,
      },
    ],
    explanation:
      "Overfitting occurs when a model fits noise or idiosyncrasies of the training data. This often happens with complex models containing many parameters. The model may achieve low training error but perform worse on validation or test data.",
  },

  {
    id: "mit15773-l3-q09",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe early stopping as a regularization technique?",
    options: [
      {
        text: "Early stopping monitors validation performance during training.",
        isCorrect: true,
      },
      {
        text: "Training is halted once validation performance stops improving.",
        isCorrect: true,
      },
      {
        text: "Early stopping requires calculating gradients on the validation set.",
        isCorrect: false,
      },
      { text: "Early stopping can help prevent overfitting.", isCorrect: true },
    ],
    explanation:
      "Early stopping uses validation metrics to determine when training should stop. The validation set is never used to compute gradients—it only evaluates the model. If validation performance stops improving, training is stopped to avoid overfitting.",
  },

  {
    id: "mit15773-l3-q10",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe TensorFlow’s role in deep learning workflows?",
    options: [
      {
        text: "TensorFlow can automatically compute gradients using automatic differentiation.",
        isCorrect: true,
      },
      {
        text: "TensorFlow provides implementations of many optimization algorithms.",
        isCorrect: true,
      },
      {
        text: "TensorFlow automatically distributes computations across hardware such as GPUs.",
        isCorrect: true,
      },
      {
        text: "TensorFlow requires users to manually compute derivatives for backpropagation.",
        isCorrect: false,
      },
    ],
    explanation:
      "TensorFlow performs automatic differentiation, meaning it can compute gradients automatically for complex computational graphs. It also includes many optimizers and supports distributed and GPU-accelerated computation.",
  },

  {
    id: "mit15773-l3-q11",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Consider the softmax function defined as \\( p_i = \\frac{e^{a_i}}{\\sum_j e^{a_j}} \\). Which statements are correct?",
    options: [
      { text: "Each \\(p_i\\) is guaranteed to be positive.", isCorrect: true },
      { text: "The probabilities sum to 1.", isCorrect: true },
      {
        text: "The output distribution is invariant to adding the same constant to all \\(a_i\\).",
        isCorrect: true,
      },
      {
        text: "The softmax output is independent of the relative magnitudes of the \\(a_i\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Softmax exponentiates each input and divides by the total sum. This ensures outputs are positive and sum to one. Adding a constant to all logits cancels out in numerator and denominator, but relative magnitudes still strongly influence probabilities.",
  },

  {
    id: "mit15773-l3-q12",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about loss functions for classification tasks are correct?",
    options: [
      {
        text: "Binary cross-entropy is commonly used for binary classification problems.",
        isCorrect: true,
      },
      {
        text: "Categorical cross-entropy is used when predicting probabilities across multiple classes.",
        isCorrect: true,
      },
      {
        text: "Sparse categorical cross-entropy is used when class labels are encoded as integers.",
        isCorrect: true,
      },
      {
        text: "Mean squared error is always preferred for classification problems.",
        isCorrect: false,
      },
    ],
    explanation:
      "Binary cross-entropy is used when predicting a single probability. For multi-class classification, categorical cross-entropy or sparse categorical cross-entropy is typically used depending on the label encoding. Mean squared error is rarely ideal for classification.",
  },

  {
    id: "mit15773-l3-q13",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Consider a neural network with 29 input features and a dense hidden layer with 16 neurons followed by a single output neuron. Which statements are correct about the parameter count?",
    options: [
      {
        text: "The hidden layer has \\(29 \\times 16\\) weight parameters.",
        isCorrect: true,
      },
      { text: "The hidden layer has 16 bias parameters.", isCorrect: true },
      {
        text: "The output layer has \\(16 \\times 1\\) weights and 1 bias.",
        isCorrect: true,
      },
      { text: "The total number of parameters equals 497.", isCorrect: true },
    ],
    explanation:
      "The hidden layer contains 29 inputs × 16 neurons = 464 weights plus 16 biases. The output layer contains 16 weights plus one bias. Summing these gives 464 + 16 + 16 + 1 = 497 parameters.",
  },

  {
    id: "mit15773-l3-q14",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements about color image representation are correct?",
    options: [
      {
        text: "Each pixel is typically represented by three values corresponding to RGB channels.",
        isCorrect: true,
      },
      { text: "Each channel typically ranges from 0 to 255.", isCorrect: true },
      {
        text: "A color image can be represented as a tensor of rank 3.",
        isCorrect: true,
      },
      {
        text: "A grayscale image uses the same number of channels as a color image.",
        isCorrect: false,
      },
    ],
    explanation:
      "Color images typically use three channels—red, green, and blue—each with intensity values between 0 and 255. This creates a rank-3 tensor: height × width × channels. Grayscale images use only one channel.",
  },

  {
    id: "mit15773-l3-q15",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which of the following tasks are common problems in computer vision?",
    options: [
      { text: "Image classification.", isCorrect: true },
      { text: "Object detection.", isCorrect: true },
      { text: "Semantic segmentation.", isCorrect: true },
      { text: "Sorting images alphabetically.", isCorrect: false },
    ],
    explanation:
      "Computer vision tasks include classification, object detection, and segmentation. These tasks involve identifying objects or structures within images. Alphabetical sorting is unrelated to visual recognition tasks.",
  },

  {
    id: "mit15773-l3-q16",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe how validation sets are used during training?",
    options: [
      {
        text: "Validation data is used to monitor model performance during training.",
        isCorrect: true,
      },
      {
        text: "Validation data is used to update model weights.",
        isCorrect: false,
      },
      {
        text: "Validation metrics can help detect overfitting.",
        isCorrect: true,
      },
      {
        text: "Validation sets are typically created by splitting the training data.",
        isCorrect: true,
      },
    ],
    explanation:
      "The validation set helps evaluate how well the model generalizes during training. It is not used to compute gradients or update weights. Instead, it provides signals about overfitting and model performance.",
  },

  {
    id: "mit15773-l3-q17",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe accuracy as a metric for classification?",
    options: [
      {
        text: "Accuracy measures the proportion of correct predictions.",
        isCorrect: true,
      },
      {
        text: "Accuracy can change dramatically even if predicted probabilities change only slightly.",
        isCorrect: true,
      },
      {
        text: "Accuracy is a continuous differentiable function used for gradient descent.",
        isCorrect: false,
      },
      {
        text: "Accuracy may not fully capture model performance when classes are imbalanced.",
        isCorrect: true,
      },
    ],
    explanation:
      "Accuracy measures the fraction of correct predictions. Small probability changes around classification thresholds can flip predictions and significantly change accuracy. Because accuracy is not differentiable, it is not typically used directly as a training loss.",
  },

  {
    id: "mit15773-l3-q18",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements about batch size selection are correct?",
    options: [
      {
        text: "Common default batch sizes include 32 and 64.",
        isCorrect: true,
      },
      {
        text: "Batch size determines how many samples are used to compute each gradient update.",
        isCorrect: true,
      },
      {
        text: "Batch size must equal the size of the entire training dataset.",
        isCorrect: false,
      },
      {
        text: "Batch size can influence computational efficiency on GPUs.",
        isCorrect: true,
      },
    ],
    explanation:
      "Batch size determines how many training samples are used for each gradient update. Common defaults such as 32 or 64 often align well with GPU parallelization. Batch size does not need to equal the full dataset size.",
  },

  {
    id: "mit15773-l3-q19",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe preprocessing steps commonly applied before training neural networks?",
    options: [
      {
        text: "Categorical variables are often converted to numeric form using one-hot encoding.",
        isCorrect: true,
      },
      {
        text: "Numeric features are often standardized by subtracting the mean and dividing by the standard deviation.",
        isCorrect: true,
      },
      {
        text: "Test data should influence the normalization parameters used for training.",
        isCorrect: false,
      },
      {
        text: "Neural networks require purely numeric inputs.",
        isCorrect: true,
      },
    ],
    explanation:
      "Neural networks require numeric inputs, so categorical variables must be encoded numerically. Numeric features are often standardized to stabilize training. Importantly, normalization parameters should be computed using only training data.",
  },

  {
    id: "mit15773-l3-q20",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the relationship between logits and probabilities in multi-class classification?",
    options: [
      {
        text: "Logits are the raw outputs of the neural network before softmax.",
        isCorrect: true,
      },
      { text: "Softmax converts logits into probabilities.", isCorrect: true },
      {
        text: "The largest logit corresponds to the most likely class after softmax.",
        isCorrect: true,
      },
      {
        text: "Logits must already sum to 1 before applying softmax.",
        isCorrect: false,
      },
    ],
    explanation:
      "Logits are raw scores produced by the final linear layer of a network. The softmax function converts these scores into probabilities that sum to one. The class with the highest logit typically produces the highest probability after softmax.",
  },

  {
    id: "mit15773-l3-q21",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statement best describes the relationship between training, validation, and test datasets in supervised learning?",
    options: [
      {
        text: "The training set is used to update model parameters during learning.",
        isCorrect: true,
      },
      {
        text: "The validation set is used to compute gradients during training.",
        isCorrect: false,
      },
      {
        text: "The test set is repeatedly used during training to tune hyperparameters.",
        isCorrect: false,
      },
      {
        text: "The test set should influence normalization parameters during preprocessing.",
        isCorrect: false,
      },
    ],
    explanation:
      "The training set is used to compute gradients and update parameters. The validation set is used only for monitoring performance or hyperparameter tuning, while the test set is held aside and used only once to estimate real-world performance.",
  },

  {
    id: "mit15773-l3-q22",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which of the following statements about baseline models in classification are correct?",
    options: [
      {
        text: "A baseline model helps determine whether a complex model provides meaningful improvement.",
        isCorrect: true,
      },
      {
        text: "Predicting the majority class for every example can serve as a simple baseline.",
        isCorrect: true,
      },
      {
        text: "Baseline models must always be neural networks.",
        isCorrect: false,
      },
      {
        text: "A baseline accuracy can often be estimated by looking at class distribution.",
        isCorrect: true,
      },
    ],
    explanation:
      "A baseline model provides a simple reference point for evaluating more complex models. For example, predicting the majority class may produce a surprisingly strong baseline in imbalanced datasets. Baselines are not restricted to neural networks.",
  },

  {
    id: "mit15773-l3-q23",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Consider the gradient descent update rule \\( w \\leftarrow w - \\alpha \\nabla L(w) \\). Which statements are correct?",
    options: [
      {
        text: "The learning rate \\(\\alpha\\) controls the step size of each update.",
        isCorrect: true,
      },
      {
        text: "The gradient \\(\\nabla L(w)\\) indicates the direction of steepest increase in loss.",
        isCorrect: true,
      },
      {
        text: "Subtracting the gradient moves the parameters toward lower loss.",
        isCorrect: true,
      },
      {
        text: "Gradient descent guarantees finding the global minimum for all neural networks.",
        isCorrect: false,
      },
    ],
    explanation:
      "The gradient indicates the direction of greatest increase in the loss function, so subtracting it moves parameters downhill. The learning rate determines how large each update step is. However, neural networks have non-convex loss surfaces, so global optimality is not guaranteed.",
  },

  {
    id: "mit15773-l3-q24",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statement correctly describes the concept of 'rank' in tensors?",
    options: [
      {
        text: "The rank of a tensor equals the number of dimensions or axes.",
        isCorrect: true,
      },
      {
        text: "The rank of a tensor equals the number of elements stored inside it.",
        isCorrect: false,
      },
      {
        text: "Rank refers only to matrices and not other tensors.",
        isCorrect: false,
      },
      { text: "Rank is unrelated to dimensionality.", isCorrect: false },
    ],
    explanation:
      "The rank of a tensor refers to how many axes it has. For example, a scalar has rank 0, a vector rank 1, a matrix rank 2, and higher-dimensional tensors extend this concept.",
  },

  {
    id: "mit15773-l3-q25",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about GPU acceleration in deep learning are correct?",
    options: [
      {
        text: "GPUs are effective because they can perform many operations in parallel.",
        isCorrect: true,
      },
      {
        text: "Matrix operations used in neural networks map well to GPU architectures.",
        isCorrect: true,
      },
      {
        text: "Deep learning frameworks like TensorFlow automatically use GPUs when available.",
        isCorrect: true,
      },
      {
        text: "GPUs eliminate the need for optimization algorithms such as Adam.",
        isCorrect: false,
      },
    ],
    explanation:
      "GPUs excel at parallel numerical computation, especially large matrix operations common in deep learning. Frameworks like TensorFlow automatically leverage GPU hardware when available. However, optimization algorithms are still necessary.",
  },

  {
    id: "mit15773-l3-q26",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements about binary cross-entropy loss are correct?",
    options: [
      {
        text: "Binary cross-entropy compares predicted probabilities to true binary labels.",
        isCorrect: true,
      },
      {
        text: "Binary cross-entropy is commonly used with sigmoid output layers.",
        isCorrect: true,
      },
      {
        text: "Binary cross-entropy measures squared error between predictions and labels.",
        isCorrect: false,
      },
      {
        text: "Binary cross-entropy penalizes confident incorrect predictions heavily.",
        isCorrect: true,
      },
    ],
    explanation:
      "Binary cross-entropy measures how well predicted probabilities match true labels. It is commonly paired with sigmoid outputs. Because of the logarithmic terms, confident incorrect predictions produce large penalties.",
  },

  {
    id: "mit15773-l3-q27",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about dropout as a regularization technique are correct?",
    options: [
      {
        text: "Dropout randomly disables a fraction of neurons during training.",
        isCorrect: true,
      },
      {
        text: "Dropout encourages networks to learn more robust features.",
        isCorrect: true,
      },
      {
        text: "Dropout is typically applied only during inference.",
        isCorrect: false,
      },
      {
        text: "Dropout reduces overfitting by preventing reliance on specific neurons.",
        isCorrect: true,
      },
    ],
    explanation:
      "Dropout randomly sets some neuron outputs to zero during training, forcing the network to distribute learning across multiple pathways. This helps reduce overfitting. Dropout is typically disabled during inference.",
  },

  {
    id: "mit15773-l3-q28",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which of the following statements about logits are correct?",
    options: [
      {
        text: "Logits are the outputs of the final linear layer before applying softmax.",
        isCorrect: true,
      },
      { text: "Logits can take any real value.", isCorrect: true },
      {
        text: "Applying softmax converts logits into probabilities.",
        isCorrect: true,
      },
      { text: "Logits must lie between 0 and 1.", isCorrect: false },
    ],
    explanation:
      "Logits are unconstrained real-valued outputs produced by the network before normalization. Softmax converts them into probabilities that sum to one. Logits themselves are not restricted to the range [0,1].",
  },

  {
    id: "mit15773-l3-q29",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about model.compile() in Keras are correct?",
    options: [
      {
        text: "The compile step specifies the optimizer used for training.",
        isCorrect: true,
      },
      {
        text: "The compile step specifies the loss function.",
        isCorrect: true,
      },
      {
        text: "Metrics specified in compile() directly influence gradient updates.",
        isCorrect: false,
      },
      {
        text: "The compile step prepares the computational graph for efficient execution.",
        isCorrect: true,
      },
    ],
    explanation:
      "In Keras, model.compile() defines the optimizer, loss function, and optional metrics. Metrics are used only for reporting and evaluation, not for gradient computation. The compile step prepares the model for efficient execution.",
  },

  {
    id: "mit15773-l3-q30",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which of the following best describes image classification?",
    options: [
      {
        text: "Predicting a category label for an entire image.",
        isCorrect: true,
      },
      {
        text: "Identifying the coordinates of objects inside the image.",
        isCorrect: false,
      },
      { text: "Labeling each pixel with a category.", isCorrect: false },
      { text: "Grouping pixels into object instances.", isCorrect: false },
    ],
    explanation:
      "Image classification assigns a single label to the entire image, such as 'dog' or 'cat'. Tasks like localization, segmentation, or detection involve identifying positions or classifying pixels instead.",
  },

  {
    id: "mit15773-l3-q31",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about object detection are correct?",
    options: [
      {
        text: "Object detection identifies multiple objects in an image.",
        isCorrect: true,
      },
      {
        text: "Object detection typically outputs bounding box coordinates.",
        isCorrect: true,
      },
      {
        text: "Object detection produces a single class label for the whole image.",
        isCorrect: false,
      },
      {
        text: "Object detection is widely used in applications such as self-driving cars.",
        isCorrect: true,
      },
    ],
    explanation:
      "Object detection identifies multiple objects in an image and predicts bounding boxes for each instance. It is widely used in autonomous vehicles, surveillance systems, and robotics.",
  },

  {
    id: "mit15773-l3-q32",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Suppose the output layer of a neural network uses softmax across 10 classes. Which statements are correct?",
    options: [
      { text: "The output vector contains 10 probabilities.", isCorrect: true },
      { text: "The probabilities sum to 1.", isCorrect: true },
      {
        text: "The predicted class is usually the index of the largest probability.",
        isCorrect: true,
      },
      {
        text: "Softmax guarantees the correct class always receives the highest probability.",
        isCorrect: false,
      },
    ],
    explanation:
      "Softmax outputs a probability distribution across classes. The predicted class is typically the class with the highest probability. However, the model may still make mistakes if the highest probability corresponds to an incorrect class.",
  },

  {
    id: "mit15773-l3-q33",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements about preprocessing numeric inputs for neural networks are correct?",
    options: [
      {
        text: "Standardization often improves training stability.",
        isCorrect: true,
      },
      {
        text: "Standardization typically subtracts the mean and divides by the standard deviation.",
        isCorrect: true,
      },
      {
        text: "Neural networks require inputs to be exactly integers.",
        isCorrect: false,
      },
      {
        text: "Large feature scales can slow or destabilize learning.",
        isCorrect: true,
      },
    ],
    explanation:
      "Standardizing inputs ensures features have similar scales, improving numerical stability and convergence. Neural networks accept real-valued inputs, not only integers.",
  },

  {
    id: "mit15773-l3-q34",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements correctly describe instance segmentation?",
    options: [
      {
        text: "Instance segmentation classifies every pixel in an image.",
        isCorrect: true,
      },
      {
        text: "Instance segmentation distinguishes between different objects of the same class.",
        isCorrect: true,
      },
      {
        text: "Instance segmentation produces only one bounding box per image.",
        isCorrect: false,
      },
      {
        text: "Instance segmentation combines segmentation and object identification.",
        isCorrect: true,
      },
    ],
    explanation:
      "Instance segmentation extends semantic segmentation by distinguishing separate objects belonging to the same class. For example, multiple sheep in an image would each be labeled separately.",
  },

  {
    id: "mit15773-l3-q35",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about randomness in neural network training are correct?",
    options: [
      { text: "Weights are often initialized randomly.", isCorrect: true },
      {
        text: "Training data may be shuffled before forming batches.",
        isCorrect: true,
      },
      {
        text: "Randomness can lead to slightly different training results.",
        isCorrect: true,
      },
      { text: "Random seeds help improve reproducibility.", isCorrect: true },
    ],
    explanation:
      "Neural network training often involves randomness in weight initialization, data shuffling, and regularization techniques. Setting random seeds helps reproduce results more reliably.",
  },

  {
    id: "mit15773-l3-q36",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements about accuracy and loss are correct?",
    options: [
      {
        text: "Accuracy measures discrete prediction correctness.",
        isCorrect: true,
      },
      {
        text: "Loss functions typically provide smoother signals for optimization.",
        isCorrect: true,
      },
      {
        text: "A model's loss can increase even while accuracy improves.",
        isCorrect: true,
      },
      {
        text: "Accuracy is always the best loss function for gradient descent.",
        isCorrect: false,
      },
    ],
    explanation:
      "Accuracy is a discrete metric, whereas loss functions are continuous and differentiable, making them suitable for gradient-based optimization. Small probability shifts can improve accuracy while increasing loss.",
  },

  {
    id: "mit15773-l3-q37",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about the Adam optimizer are correct?",
    options: [
      {
        text: "Adam is an extension of stochastic gradient descent.",
        isCorrect: true,
      },
      {
        text: "Adam adapts learning rates based on past gradients.",
        isCorrect: true,
      },
      {
        text: "Adam is commonly used as a default optimizer.",
        isCorrect: true,
      },
      { text: "Adam removes the need to compute gradients.", isCorrect: false },
    ],
    explanation:
      "Adam is a variant of SGD that uses adaptive learning rates and momentum-like mechanisms. It is widely used as a default optimizer but still relies on gradient computation.",
  },

  {
    id: "mit15773-l3-q38",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Consider training a model with 242 training examples, batch size 32, and one epoch. Which statements are correct?",
    options: [
      {
        text: "The number of batches equals \\(\\lceil 242/32 \\rceil\\).",
        isCorrect: true,
      },
      { text: "There will be 8 batches in one epoch.", isCorrect: true },
      {
        text: "The final batch may contain fewer than 32 examples.",
        isCorrect: true,
      },
      {
        text: "The model parameters are updated after each batch.",
        isCorrect: true,
      },
    ],
    explanation:
      "The number of batches is computed using ceiling division. For 242 samples with batch size 32, there are 8 batches. The final batch may contain fewer samples, and parameters are updated after each batch during SGD.",
  },

  {
    id: "mit15773-l3-q39",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statement correctly describes semantic segmentation?",
    options: [
      {
        text: "Every pixel in an image is assigned a category label.",
        isCorrect: true,
      },
      {
        text: "Semantic segmentation predicts a single label for the entire image.",
        isCorrect: false,
      },
      {
        text: "Semantic segmentation requires bounding boxes around objects.",
        isCorrect: false,
      },
      {
        text: "Semantic segmentation ignores spatial structure of images.",
        isCorrect: false,
      },
    ],
    explanation:
      "Semantic segmentation assigns a class label to each pixel, allowing models to identify regions such as road, grass, or animals within an image.",
  },

  {
    id: "mit15773-l3-q40",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Suppose a neural network predicts logits \\(z_1, z_2, z_3\\) for three classes. After applying softmax, probabilities are computed as \\(p_i = \\frac{e^{z_i}}{\\sum_j e^{z_j}}\\). Which statements are correct?",
    options: [
      { text: "All probabilities will lie between 0 and 1.", isCorrect: true },
      { text: "The probabilities must sum to 1.", isCorrect: true },
      { text: "If \\(z_2 > z_1\\), then \\(p_2 > p_1\\).", isCorrect: true },
      {
        text: "Softmax probabilities are independent of the logits.",
        isCorrect: false,
      },
    ],
    explanation:
      "Softmax converts logits into normalized probabilities between 0 and 1. The relative ordering of logits determines the ordering of probabilities. However, the probabilities depend entirely on the logits.",
  },
];
