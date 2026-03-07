import { Question } from "../../quiz";

export const MIT6S191_L3_CNNsQuestions: Question[] = [
  {
    id: "mit6s191-l3-q01",
    chapter: 3,
    difficulty: "easy",
    prompt: "How are digital images typically represented inside a computer?",
    options: [
      {
        text: "A grayscale image can be represented as a 2D matrix of intensity values.",
        isCorrect: true,
      },
      {
        text: "An RGB image can be represented as a 3D tensor of shape \\(H \\times W \\times 3\\).",
        isCorrect: true,
      },
      {
        text: "Pixel values are often integers in a bounded range such as \\([0,255]\\).",
        isCorrect: true,
      },
      {
        text: "An image is fundamentally a numerical array rather than a symbolic object.",
        isCorrect: true,
      },
    ],
    explanation:
      "Images are stored as numerical arrays. A grayscale image is a 2D matrix, while a color image includes a third channel dimension for red, green, and blue. Pixel values are typically bounded, for example between 0 and 255. This numerical structure enables direct use in neural networks.",
  },

  {
    id: "mit6s191-l3-q02",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe regression and classification tasks?",
    options: [
      { text: "Regression outputs continuous values.", isCorrect: true },
      {
        text: "Classification outputs discrete class labels.",
        isCorrect: true,
      },
      {
        text: "Classification models can output probabilities over classes.",
        isCorrect: true,
      },
      { text: "Regression requires outputs to sum to one.", isCorrect: false },
    ],
    explanation:
      "Regression predicts continuous quantities such as steering angles. Classification predicts discrete labels and often uses a softmax to produce probabilities that sum to one. Regression outputs do not need to sum to one because they are not probability distributions.",
  },

  {
    id: "mit6s191-l3-q03",
    chapter: 3,
    difficulty: "medium",
    prompt: "Why is manual feature extraction challenging in computer vision?",
    options: [
      {
        text: "Objects can vary due to viewpoint and scale changes.",
        isCorrect: true,
      },
      {
        text: "Illumination and occlusion can significantly alter pixel values.",
        isCorrect: true,
      },
      {
        text: "Intra-class variation makes defining fixed rules difficult.",
        isCorrect: true,
      },
      {
        text: "Pixel values are always identical for objects of the same class.",
        isCorrect: false,
      },
    ],
    explanation:
      "Objects may appear differently depending on viewpoint, scale, lighting, and occlusion. Even within the same class, there can be large variability. This makes it extremely difficult to hand-engineer robust rules based purely on fixed pixel patterns.",
  },

  {
    id: "mit6s191-l3-q04",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Why is flattening an image into a vector before a fully connected network problematic?",
    options: [
      {
        text: "Flattening destroys spatial relationships between neighboring pixels.",
        isCorrect: true,
      },
      {
        text: "Fully connected layers introduce a very large number of parameters.",
        isCorrect: true,
      },
      {
        text: "Flattening removes inherent 2D structure of images.",
        isCorrect: true,
      },
      {
        text: "Flattening prevents the use of nonlinear activations.",
        isCorrect: false,
      },
    ],
    explanation:
      "Flattening converts a structured 2D grid into a 1D vector, discarding spatial locality. Fully connected layers then connect every pixel to every neuron, leading to many parameters. Nonlinear activations are still possible, so that is not the core issue.",
  },

  {
    id: "mit6s191-l3-q05",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Given a \\(5 \\times 5\\) input and a \\(3 \\times 3\\) filter with stride 1 and no padding, which statements are correct?",
    options: [
      {
        text: "The output spatial dimension is \\((5-3+1) \\times (5-3+1) = 3 \\times 3\\).",
        isCorrect: true,
      },
      {
        text: "Each output neuron depends on 9 input pixels.",
        isCorrect: true,
      },
      {
        text: "The filter contains 9 learnable weights (excluding bias).",
        isCorrect: true,
      },
      {
        text: "The output will have the same spatial size as the input.",
        isCorrect: false,
      },
    ],
    explanation:
      "With stride 1 and no padding, output size is \\(5-3+1=3\\). Each output location computes a dot product over a \\(3 \\times 3\\) region. The filter has 9 weights. Without padding, spatial resolution shrinks.",
  },

  {
    id: "mit6s191-l3-q06",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which best describe the convolution operation?",
    options: [
      {
        text: "It performs element-wise multiplication between a filter and a local image patch.",
        isCorrect: true,
      },
      {
        text: "It sums the multiplied values to produce a scalar output per location.",
        isCorrect: true,
      },
      {
        text: "The same filter is reused across spatial positions.",
        isCorrect: true,
      },
      {
        text: "Each output neuron connects to all pixels in the image.",
        isCorrect: false,
      },
    ],
    explanation:
      "Convolution slides a small filter across the image, performing element-wise multiplication and summation. The same filter weights are shared across spatial positions. Unlike fully connected layers, each neuron only sees a local patch.",
  },

  {
    id: "mit6s191-l3-q07",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "What is the role of nonlinear activation functions such as the Rectified Linear Unit (ReLU)?",
    options: [
      { text: "They introduce nonlinearity into the model.", isCorrect: true },
      {
        text: "They increase expressive power beyond linear filters.",
        isCorrect: true,
      },
      { text: "ReLU maps negative values to zero.", isCorrect: true },
      { text: "They ensure the outputs sum to one.", isCorrect: false },
    ],
    explanation:
      "Nonlinearities allow neural networks to model complex functions beyond linear transformations. ReLU specifically sets negative activations to zero while preserving positive values. Summing to one is a property of softmax, not ReLU.",
  },

  {
    id: "mit6s191-l3-q08",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Why does stacking convolutional layers create hierarchical feature representations?",
    options: [
      {
        text: "Early layers detect low-level features such as edges.",
        isCorrect: true,
      },
      {
        text: "Deeper layers combine simpler features into more complex ones.",
        isCorrect: true,
      },
      {
        text: "Hierarchical composition increases representational capacity.",
        isCorrect: true,
      },
      {
        text: "All layers detect identical features at different scales.",
        isCorrect: false,
      },
    ],
    explanation:
      "Early layers often detect edges and simple patterns. Deeper layers combine those into higher-level structures such as facial parts. This hierarchical composition increases expressive power. Layers typically learn different representations.",
  },

  {
    id: "mit6s191-l3-q09",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements describe max pooling?",
    options: [
      { text: "It reduces spatial resolution.", isCorrect: true },
      {
        text: "It selects the maximum value in each local region.",
        isCorrect: true,
      },
      {
        text: "During backpropagation, gradients pass only through the maximum element.",
        isCorrect: true,
      },
      { text: "It introduces additional learnable weights.", isCorrect: false },
    ],
    explanation:
      "Max pooling downsamples feature maps by selecting the maximum value in a region. This reduces spatial resolution and increases effective receptive field. Only the maximum element receives gradient, and no learnable weights are added.",
  },

  {
    id: "mit6s191-l3-q10",
    chapter: 3,
    difficulty: "hard",
    prompt: "Why is weight sharing in convolution important?",
    options: [
      {
        text: "It drastically reduces the number of parameters compared to fully connected layers.",
        isCorrect: true,
      },
      { text: "It enables translation robustness.", isCorrect: true },
      {
        text: "It enforces that the same feature detector operates across positions.",
        isCorrect: true,
      },
      {
        text: "It ensures rotation invariance automatically.",
        isCorrect: false,
      },
    ],
    explanation:
      "Weight sharing means one filter scans across the entire image, reducing parameter count. It allows the same feature detector to respond anywhere in the image, improving translation robustness. However, it does not automatically provide rotation invariance.",
  },

  {
    id: "mit6s191-l3-q11",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which tasks can use convolutional neural networks as feature extractors?",
    options: [
      { text: "Image classification.", isCorrect: true },
      { text: "Object detection.", isCorrect: true },
      { text: "Semantic segmentation.", isCorrect: true },
      {
        text: "Regression tasks such as steering angle prediction.",
        isCorrect: true,
      },
    ],
    explanation:
      "The convolutional backbone extracts spatial features. These features can then feed into classification, detection, segmentation, or regression heads. The same feature extractor can be reused across different tasks.",
  },

  {
    id: "mit6s191-l3-q12",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe object detection compared to classification?",
    options: [
      {
        text: "It predicts both class labels and bounding box coordinates.",
        isCorrect: true,
      },
      {
        text: "The number of detected objects can vary per image.",
        isCorrect: true,
      },
      {
        text: "It requires localization in addition to classification.",
        isCorrect: true,
      },
      {
        text: "It outputs exactly one class label per image.",
        isCorrect: false,
      },
    ],
    explanation:
      "Object detection goes beyond classification by predicting bounding boxes and class labels for multiple objects. The number of outputs can vary depending on how many objects appear. Classification alone outputs a single label per image.",
  },

  {
    id: "mit6s191-l3-q13",
    chapter: 3,
    difficulty: "hard",
    prompt: "Semantic segmentation differs from object detection because:",
    options: [
      {
        text: "It predicts a class for every pixel in the image.",
        isCorrect: true,
      },
      { text: "It outputs a 2D map with class assignments.", isCorrect: true },
      {
        text: "It can be implemented using upsampling layers.",
        isCorrect: true,
      },
      { text: "It removes convolutional layers entirely.", isCorrect: false },
    ],
    explanation:
      "Semantic segmentation assigns a label to each pixel, producing a full-resolution map. Upsampling layers are often used to restore spatial resolution. Convolutions remain central to the architecture.",
  },

  {
    id: "mit6s191-l3-q14",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Why can convolutional neural networks generalize to unseen roads in autonomous driving?",
    options: [
      {
        text: "They learn visual patterns correlated with control signals.",
        isCorrect: true,
      },
      {
        text: "They optimize end-to-end from perception to action.",
        isCorrect: true,
      },
      {
        text: "They do not require explicit hand-engineered maps.",
        isCorrect: true,
      },
      {
        text: "They memorize every possible road configuration.",
        isCorrect: false,
      },
    ],
    explanation:
      "By training on diverse driving data, CNNs learn features associated with steering behavior. End-to-end optimization connects perception directly to control. They generalize based on learned patterns rather than memorizing all roads.",
  },

  {
    id: "mit6s191-l3-q15",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Suppose a convolutional layer learns filters \\(W\\). During training, which statements are correct?",
    options: [
      {
        text: "The filters are randomly initialized and updated via backpropagation.",
        isCorrect: true,
      },
      {
        text: "The optimization objective guides filters toward improving task performance.",
        isCorrect: true,
      },
      {
        text: "Different filters within a layer typically learn different features.",
        isCorrect: true,
      },
      { text: "Filters remain fixed throughout training.", isCorrect: false },
    ],
    explanation:
      "Filters start from random initialization and are updated by gradient-based optimization. The loss function encourages filters that improve accuracy. Different filters specialize in detecting different patterns.",
  },

  {
    id: "mit6s191-l3-q16",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statement about feature learning in deep learning is correct?",
    options: [
      {
        text: "Features are learned directly from data rather than manually specified.",
        isCorrect: true,
      },
      {
        text: "Domain experts must explicitly define every edge detector.",
        isCorrect: false,
      },
      {
        text: "Feature extraction is unnecessary in image models.",
        isCorrect: false,
      },
      {
        text: "Deep learning eliminates the need for labeled data.",
        isCorrect: false,
      },
    ],
    explanation:
      "Deep learning learns hierarchical features directly from data. Unlike traditional pipelines, we do not manually design every detector. However, labeled data is typically still required for supervised tasks.",
  },

  {
    id: "mit6s191-l3-q17",
    chapter: 3,
    difficulty: "medium",
    prompt: "Why is pooling often used between convolutional layers?",
    options: [
      { text: "To reduce computational cost.", isCorrect: true },
      { text: "To increase effective receptive field.", isCorrect: true },
      { text: "To provide some translation robustness.", isCorrect: true },
      { text: "To increase spatial resolution.", isCorrect: false },
    ],
    explanation:
      "Pooling reduces spatial size, lowering computation in later layers. It increases receptive field size and introduces some robustness to small translations. It does not increase spatial resolution.",
  },

  {
    id: "mit6s191-l3-q18",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "In a convolutional neural network used for classification, why is flattening typically applied only after feature extraction?",
    options: [
      {
        text: "Spatial information has already been encoded into learned features.",
        isCorrect: true,
      },
      {
        text: "Flattening early would discard important spatial structure.",
        isCorrect: true,
      },
      {
        text: "Classification outputs are typically one-dimensional.",
        isCorrect: true,
      },
      { text: "Flattening increases spatial resolution.", isCorrect: false },
    ],
    explanation:
      "Early flattening destroys spatial structure prematurely. After multiple convolutional layers, spatial patterns are encoded into high-level feature maps. At that stage, flattening is appropriate for producing class predictions.",
  },

  {
    id: "mit6s191-l3-q19",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Why is softmax commonly used at the final layer for classification?",
    options: [
      { text: "It converts logits into probabilities.", isCorrect: true },
      { text: "Its outputs sum to one.", isCorrect: true },
      { text: "It enables multi-class prediction.", isCorrect: true },
      { text: "It introduces spatial locality.", isCorrect: false },
    ],
    explanation:
      "Softmax transforms raw scores into a probability distribution. This ensures outputs sum to one and supports multi-class classification. It does not operate on spatial structure.",
  },

  {
    id: "mit6s191-l3-q20",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement about convolutional neural networks and invariance is correct?",
    options: [
      {
        text: "Convolution provides robustness to translation through weight sharing.",
        isCorrect: true,
      },
      {
        text: "Convolution alone guarantees invariance to rotation.",
        isCorrect: false,
      },
      {
        text: "Convolution ensures identical responses for all scales.",
        isCorrect: false,
      },
      {
        text: "Convolution removes the need for nonlinearities.",
        isCorrect: false,
      },
    ],
    explanation:
      "Weight sharing makes CNNs more robust to translations. However, rotation and scale invariance are not automatically guaranteed and often require data augmentation or architectural modifications. Nonlinearities remain essential.",
  },

  {
    id: "mit6s191-l3-q21",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Consider multi-channel convolution with an RGB input \\(H \\times W \\times 3\\) and a filter of size \\(3 \\times 3\\). Which statements are correct?",
    options: [
      { text: "The filter must also have depth 3.", isCorrect: true },
      {
        text: "The filter contains \\(3 \\times 3 \\times 3 = 27\\) weights (excluding bias).",
        isCorrect: true,
      },
      {
        text: "Element-wise multiplication occurs across spatial and channel dimensions.",
        isCorrect: true,
      },
      {
        text: "The filter is applied independently to each channel and summed later externally.",
        isCorrect: false,
      },
    ],
    explanation:
      "For RGB input, each filter spans all three channels. The dot product is computed across spatial dimensions and channel depth simultaneously. The summation across channels happens inside the convolution operation, not externally.",
  },

  {
    id: "mit6s191-l3-q22",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "If two \\(3 \\times 3\\) convolutional layers (stride 1, no pooling) are stacked, what is the effective receptive field size of a neuron in the second layer?",
    options: [
      { text: "It is \\(5 \\times 5\\).", isCorrect: true },
      {
        text: "It grows beyond the original \\(3 \\times 3\\) size.",
        isCorrect: true,
      },
      {
        text: "Stacking small filters increases receptive field with fewer parameters than a single large filter.",
        isCorrect: true,
      },
      {
        text: "The receptive field remains \\(3 \\times 3\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Two stacked \\(3 \\times 3\\) convolutions produce an effective receptive field of \\(5 \\times 5\\). This happens because each neuron in the second layer depends on a \\(3 \\times 3\\) region of the first layer, which itself depends on \\(3 \\times 3\\) regions of the input. Stacking small filters increases receptive field efficiently.",
  },

  {
    id: "mit6s191-l3-q23",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Why are learned convolutional filters preferred over hand-engineered filters?",
    options: [
      { text: "They adapt automatically to the dataset.", isCorrect: true },
      { text: "They can capture task-specific features.", isCorrect: true },
      { text: "They remove the need for domain knowledge.", isCorrect: false },
      {
        text: "They always outperform any hand-designed filter.",
        isCorrect: false,
      },
    ],
    explanation:
      "Learned filters adapt to data and task objectives. They capture patterns that may not be obvious to humans. However, domain knowledge is still valuable and learned filters do not guarantee universal superiority.",
  },

  {
    id: "mit6s191-l3-q24",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "In object detection using a Region-based Convolutional Neural Network (R-CNN), which statements are correct?",
    options: [
      {
        text: "The region proposal network learns candidate bounding boxes.",
        isCorrect: true,
      },
      {
        text: "Feature extraction is shared between region proposal and classification.",
        isCorrect: true,
      },
      {
        text: "The network predicts both bounding box coordinates and class labels.",
        isCorrect: true,
      },
      {
        text: "Each candidate region requires an independent full image forward pass.",
        isCorrect: false,
      },
    ],
    explanation:
      "Modern R-CNN variants share convolutional features between region proposal and classification stages. This enables efficient computation in a single forward pass. The model outputs both bounding boxes and class predictions.",
  },

  {
    id: "mit6s191-l3-q25",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Suppose a convolutional layer has 64 filters of size \\(3 \\times 3\\) applied to an input of depth 32. How many weights does this layer have (excluding bias)?",
    options: [
      {
        text: "Each filter has \\(3 \\times 3 \\times 32 = 288\\) weights.",
        isCorrect: true,
      },
      { text: "Total weights equal \\(64 \\times 288\\).", isCorrect: true },
      {
        text: "Weight count scales linearly with the number of filters.",
        isCorrect: true,
      },
      { text: "Weight count is independent of input depth.", isCorrect: false },
    ],
    explanation:
      "Each filter spans all 32 input channels, giving 288 weights per filter. With 64 filters, total weights are \\(64 \\times 288\\). Parameter count depends directly on input depth.",
  },

  {
    id: "mit6s191-l3-q26",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements describe semantic segmentation?",
    options: [
      { text: "It assigns a label to every pixel.", isCorrect: true },
      { text: "It outputs a 2D classification map.", isCorrect: true },
      {
        text: "It can be implemented with upsampling layers.",
        isCorrect: true,
      },
      { text: "It predicts exactly one label per image.", isCorrect: false },
    ],
    explanation:
      "Semantic segmentation produces dense predictions, assigning each pixel a class label. It maintains spatial structure and often uses upsampling layers. It differs from single-label classification.",
  },

  {
    id: "mit6s191-l3-q27",
    chapter: 3,
    difficulty: "medium",
    prompt: "Why is downsampling useful in convolutional neural networks?",
    options: [
      { text: "It increases effective receptive field.", isCorrect: true },
      { text: "It reduces computational cost.", isCorrect: true },
      { text: "It helps capture multi-scale features.", isCorrect: true },
      { text: "It increases spatial resolution.", isCorrect: false },
    ],
    explanation:
      "Downsampling reduces spatial size, making computation more efficient. It increases effective receptive field size and enables multi-scale feature learning. It does not increase resolution.",
  },

  {
    id: "mit6s191-l3-q28",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "For a convolution defined as \\(y(i,j) = \\sum_{u,v} W(u,v)x(i+u,j+v)\\), which statements are correct?",
    options: [
      { text: "It represents a local weighted sum.", isCorrect: true },
      {
        text: "The same weights \\(W\\) are used for all spatial positions.",
        isCorrect: true,
      },
      { text: "It preserves spatial locality.", isCorrect: true },
      {
        text: "It connects every input pixel to every output neuron.",
        isCorrect: false,
      },
    ],
    explanation:
      "This equation defines convolution as a local weighted sum over a neighborhood. The weights are shared across positions. Unlike fully connected layers, it does not connect all pixels globally.",
  },

  {
    id: "mit6s191-l3-q29",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "In autonomous driving with CNNs, predicting steering angle is best described as:",
    options: [
      { text: "A regression problem.", isCorrect: true },
      { text: "An end-to-end learning setup.", isCorrect: true },
      {
        text: "A mapping from images to continuous control outputs.",
        isCorrect: true,
      },
      { text: "A purely unsupervised task.", isCorrect: false },
    ],
    explanation:
      "Steering angle prediction is regression because outputs are continuous. CNNs map perception directly to control signals. Training typically uses supervised driving data.",
  },

  {
    id: "mit6s191-l3-q30",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Why does stacking convolutional layers with nonlinearities increase expressivity?",
    options: [
      {
        text: "Nonlinearities allow modeling non-linear decision boundaries.",
        isCorrect: true,
      },
      {
        text: "Compositions of linear layers without nonlinearities collapse to a single linear transformation.",
        isCorrect: true,
      },
      {
        text: "Depth enables hierarchical feature composition.",
        isCorrect: true,
      },
      { text: "Expressivity is unaffected by depth.", isCorrect: false },
    ],
    explanation:
      "Without nonlinearities, multiple linear layers reduce to a single linear transformation. Nonlinear activations allow complex decision boundaries. Depth enables hierarchical representation learning.",
  },

  {
    id: "mit6s191-l3-q31",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements are correct about feature learning?",
    options: [
      {
        text: "Low-level features often resemble edges or simple textures.",
        isCorrect: true,
      },
      {
        text: "Mid-level features may correspond to object parts.",
        isCorrect: true,
      },
      {
        text: "High-level features may represent full object structures.",
        isCorrect: true,
      },
      { text: "All layers learn identical representations.", isCorrect: false },
    ],
    explanation:
      "CNNs learn hierarchical representations. Early layers detect simple patterns, while deeper layers combine them into complex structures. Representations differ across depth.",
  },

  {
    id: "mit6s191-l3-q32",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Why is convolution considered more parameter-efficient than fully connected layers for images?",
    options: [
      { text: "Weights are shared across spatial locations.", isCorrect: true },
      {
        text: "Local connectivity reduces the number of connections.",
        isCorrect: true,
      },
      {
        text: "Parameter count does not scale with image size in the same way.",
        isCorrect: true,
      },
      { text: "Convolution removes the need for biases.", isCorrect: false },
    ],
    explanation:
      "Weight sharing and local connectivity drastically reduce parameters. Parameter count does not explode with image resolution. Bias terms are still typically included.",
  },

  {
    id: "mit6s191-l3-q33",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Suppose an image classification network outputs logits \\(z_i\\). After applying softmax \\(\\sigma(z_i)=\\frac{e^{z_i}}{\\sum_j e^{z_j}}\\), which are true?",
    options: [
      { text: "Outputs are non-negative.", isCorrect: true },
      { text: "Outputs sum to one.", isCorrect: true },
      {
        text: "Softmax is invariant to adding the same constant to all logits.",
        isCorrect: true,
      },
      { text: "Softmax is a linear function.", isCorrect: false },
    ],
    explanation:
      "Softmax produces non-negative outputs summing to one. Adding a constant to all logits cancels in numerator and denominator. It is nonlinear due to exponentiation.",
  },

  {
    id: "mit6s191-l3-q34",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements describe why invariance is important in vision?",
    options: [
      {
        text: "Objects may appear under different lighting conditions.",
        isCorrect: true,
      },
      { text: "Viewpoint changes alter pixel arrangements.", isCorrect: true },
      {
        text: "Robust models must tolerate deformations and occlusion.",
        isCorrect: true,
      },
      {
        text: "Objects of the same class always have identical pixel patterns.",
        isCorrect: false,
      },
    ],
    explanation:
      "Real-world images vary due to viewpoint, illumination, deformation, and occlusion. A robust model must generalize across such variation. Pixel patterns are rarely identical across instances.",
  },

  {
    id: "mit6s191-l3-q35",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Why does using many small filters (e.g., multiple \\(3 \\times 3\\)) instead of one large \\(7 \\times 7\\)) often improve efficiency?",
    options: [
      {
        text: "Multiple small filters can approximate large receptive fields.",
        isCorrect: true,
      },
      {
        text: "They typically require fewer parameters than one large filter.",
        isCorrect: true,
      },
      {
        text: "They allow insertion of nonlinearities between layers.",
        isCorrect: true,
      },
      {
        text: "They increase parameter count relative to large filters.",
        isCorrect: false,
      },
    ],
    explanation:
      "Stacking small filters expands receptive field while using fewer parameters than a single large filter. It also enables additional nonlinear transformations. This increases efficiency and expressivity.",
  },

  {
    id: "mit6s191-l3-q36",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which of the following are true about convolutional neural networks?",
    options: [
      { text: "They exploit spatial structure in images.", isCorrect: true },
      { text: "They use local receptive fields.", isCorrect: true },
      { text: "They rely on shared weights.", isCorrect: true },
      {
        text: "They require flattening before any convolution.",
        isCorrect: false,
      },
    ],
    explanation:
      "CNNs preserve spatial structure using local receptive fields and shared weights. Flattening typically occurs only after convolutional feature extraction.",
  },

  {
    id: "mit6s191-l3-q37",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about end-to-end training in CNNs are correct?",
    options: [
      {
        text: "All parameters are optimized jointly using backpropagation.",
        isCorrect: true,
      },
      {
        text: "Feature extraction and classification can be trained together.",
        isCorrect: true,
      },
      {
        text: "Loss gradients propagate through convolutional layers.",
        isCorrect: true,
      },
      { text: "Filters remain fixed during training.", isCorrect: false },
    ],
    explanation:
      "In end-to-end training, gradients propagate through the entire network. Feature extraction and prediction layers are optimized jointly. Filters are updated during training.",
  },

  {
    id: "mit6s191-l3-q38",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "In segmentation networks using upsampling, which statements are correct?",
    options: [
      {
        text: "Upsampling increases spatial resolution of feature maps.",
        isCorrect: true,
      },
      { text: "It enables pixel-wise predictions.", isCorrect: true },
      {
        text: "It can be implemented via transposed convolution.",
        isCorrect: true,
      },
      { text: "It eliminates the need for convolution.", isCorrect: false },
    ],
    explanation:
      "Upsampling restores spatial resolution after downsampling. This enables dense predictions. Transposed convolutions are one common implementation. Convolution remains essential.",
  },

  {
    id: "mit6s191-l3-q39",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Why is classification insufficient for understanding 'what is where' in vision?",
    options: [
      { text: "It does not provide spatial location.", isCorrect: true },
      { text: "It predicts only global labels.", isCorrect: true },
      {
        text: "It cannot detect multiple instances independently.",
        isCorrect: true,
      },
      { text: "It always fails in real-world images.", isCorrect: false },
    ],
    explanation:
      "Classification assigns a global label to an image but does not localize objects. Detection and segmentation extend classification to spatial understanding. Classification does not inherently fail, but it lacks localization.",
  },

  {
    id: "mit6s191-l3-q40",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about convolutional neural networks and generalization are correct?",
    options: [
      {
        text: "Generalization depends on diversity of training data.",
        isCorrect: true,
      },
      {
        text: "CNNs can struggle with rotations not seen during training.",
        isCorrect: true,
      },
      { text: "Data augmentation can improve robustness.", isCorrect: true },
      {
        text: "Convolution guarantees invariance to all transformations.",
        isCorrect: false,
      },
    ],
    explanation:
      "CNN performance depends strongly on training data diversity. Without exposure to rotations or scale changes, performance may degrade. Data augmentation improves robustness. Convolution does not guarantee invariance to all transformations.",
  },
];
