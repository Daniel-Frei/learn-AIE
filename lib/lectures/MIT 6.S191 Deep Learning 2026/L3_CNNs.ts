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
      {
        text: "Regression does not require outputs to sum to one.",
        isCorrect: true,
      },
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
        text: "Pixel values are not always identical for objects of the same class.",
        isCorrect: true,
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
        text: "Flattening does not prevent the use of nonlinear activations.",
        isCorrect: true,
      },
    ],
    explanation:
      "Flattening converts a structured 2D grid into a 1D vector, discarding spatial locality. Fully connected layers then connect every pixel to every neuron, leading to many parameters. Nonlinear activations are still possible, so that is not the core issue.",
  },

  {
    id: "mit6s191-l3-q05",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Why is using a small local filter on an image more appropriate than fully connecting every pixel to a hidden neuron?",
    options: [
      {
        text: "A local filter preserves nearby spatial relationships within a patch.",
        isCorrect: true,
      },
      {
        text: "It greatly reduces the number of parameters compared with a fully connected first layer on the whole image.",
        isCorrect: true,
      },
      {
        text: "It encourages the model to learn local visual patterns rather than ignoring image structure.",
        isCorrect: true,
      },
      {
        text: "It is not the case that it guarantees that the model will automatically recognize every object category without training.",
        isCorrect: true,
      },
    ],
    explanation:
      "A central point of the lecture is that images have spatial structure, so local filters are a much better inductive bias than flattening and fully connecting everything. Local filters both preserve useful neighborhood information and make the model far more parameter-efficient, but they still need to be trained from data.",
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
        text: "Each output neuron does not connect to all pixels in the image.",
        isCorrect: true,
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
      {
        text: "They do not ensure that the outputs sum to one.",
        isCorrect: true,
      },
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
        text: "Early layers directly detect whole objects rather than low-level features.",
        isCorrect: false,
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
      { text: "It increases spatial resolution.", isCorrect: false },
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
      { text: "It removes translation robustness.", isCorrect: false },
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
        text: "It predicts only one class label for the whole image and no boxes.",
        isCorrect: false,
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
        text: "It predicts a single class label for the whole image.",
        isCorrect: false,
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
        text: "They rely on explicit hand-engineered maps for every scene.",
        isCorrect: false,
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
      {
        text: "It is not the case that filters remain fixed throughout training.",
        isCorrect: true,
      },
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
      { text: "To increase computational cost.", isCorrect: false },
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
        text: "Spatial information has not been encoded into learned features at all.",
        isCorrect: false,
      },
      {
        text: "Flattening early preserves all important spatial structure.",
        isCorrect: false,
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
      { text: "It converts probabilities into logits.", isCorrect: false },
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
      "Which statements correctly describe how a convolutional neural network processes an RGB image?",
    options: [
      {
        text: "An RGB image can be treated as a three-channel input rather than as three unrelated images.",
        isCorrect: true,
      },
      {
        text: "Convolutional feature extraction can still be applied even when the input has multiple color channels.",
        isCorrect: true,
      },
      {
        text: "The lecture's main point is that the network should preserve spatial structure instead of flattening the image immediately.",
        isCorrect: true,
      },
      {
        text: "It is not the case that using RGB inputs means convolution is no longer applicable because the data is not two-dimensional.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture explains that color images are naturally represented as spatial arrays with an added channel dimension. That does not break convolution; it simply means the input has richer structure, which the model can still exploit while preserving spatial relationships.",
  },

  {
    id: "mit6s191-l3-q22",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Why does going deeper in a convolutional neural network help the model capture larger-scale patterns?",
    options: [
      {
        text: "Later layers operate on features extracted by earlier layers rather than only on raw pixels.",
        isCorrect: true,
      },
      {
        text: "Pooling reduces spatial resolution, which increases the relative scale of what later filters can represent.",
        isCorrect: true,
      },
      {
        text: "Deeper layers can combine low-level patterns into more abstract structures.",
        isCorrect: true,
      },
      {
        text: "It is not the case that depth is useful only because it forces every layer to learn identical edge detectors at different image sizes.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture emphasizes scale and hierarchy rather than exact receptive-field arithmetic. As the network gets deeper and pooling reduces resolution, later layers can represent larger and more abstract patterns by composing earlier features.",
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
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the transition from image classification to object detection?",
    options: [
      {
        text: "Object detection requires localization in addition to classification.",
        isCorrect: true,
      },
      {
        text: "A detector may need to return multiple outputs because an image can contain multiple objects.",
        isCorrect: true,
      },
      {
        text: "A useful strategy is to extract convolutional features first and then use them to propose likely regions.",
        isCorrect: true,
      },
      {
        text: "It is not the case that object detection is identical to image classification because both always produce one fixed class label for the whole image.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture explains that classification answers 'what is in the image,' while detection adds 'where is it.' It also describes the idea of using shared convolutional features first and then proposing candidate regions, rather than classifying the whole image with one single label.",
  },

  {
    id: "mit6s191-l3-q25",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about learned convolutional filters are correct?",
    options: [
      {
        text: "They can begin from initialization and then be improved through training.",
        isCorrect: true,
      },
      {
        text: "Different filters in the same layer can learn to respond to different kinds of visual patterns.",
        isCorrect: true,
      },
      {
        text: "Historically, people sometimes hand-designed filters such as edge detectors before learning-based methods became dominant.",
        isCorrect: true,
      },
      {
        text: "Once a filter is chosen at initialization, it cannot change during training.",
        isCorrect: false,
      },
    ],
    explanation:
      "A key theme of the lecture is the shift from hand-engineered features to learned feature extractors. CNN filters are learned parameters, so they can adapt during optimization, and different filters typically specialize in different visual patterns.",
  },

  {
    id: "mit6s191-l3-q26",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements describe semantic segmentation?",
    options: [
      {
        text: "It predicts exactly one label for the whole image.",
        isCorrect: false,
      },
      {
        text: "It outputs a single scalar rather than a 2D map.",
        isCorrect: false,
      },
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
      { text: "It decreases effective receptive field.", isCorrect: false },
      { text: "It always increases computational cost.", isCorrect: false },
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
      {
        text: "It is not the case that it is a regression problem.",
        isCorrect: false,
      },
      {
        text: "It is not the case that it is an end-to-end learning setup.",
        isCorrect: false,
      },
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
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the classification head of a convolutional neural network for image classification?",
    options: [
      {
        text: "After feature extraction, the network can flatten learned features and feed them into fully connected layers.",
        isCorrect: true,
      },
      {
        text: "The final output is often a one-dimensional vector over classes.",
        isCorrect: true,
      },
      {
        text: "Softmax is useful because it converts class scores into a probability distribution over categories.",
        isCorrect: true,
      },
      {
        text: "The classification head should preserve full two-dimensional image structure all the way to the final label vector.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture presents CNNs as having a feature extractor followed by a task-specific head. For image classification, that later part usually compresses learned spatial features into a class prediction vector, often using softmax at the end.",
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
    difficulty: "medium",
    prompt:
      "Which statements correctly describe semantic segmentation in convolutional neural networks?",
    options: [
      {
        text: "It aims to assign a class label to each pixel rather than only one label to the whole image.",
        isCorrect: true,
      },
      {
        text: "The network output is image-like because it must preserve or recover spatial structure.",
        isCorrect: true,
      },
      {
        text: "Downsampling alone is not enough if the goal is dense pixel-wise prediction at the output.",
        isCorrect: true,
      },
      {
        text: "Semantic segmentation removes the need for convolutional feature extraction.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture contrasts segmentation with global classification by emphasizing pixel-wise predictions. Because the output is itself spatial, the network must recover or maintain image structure rather than reducing everything to a single class vector.",
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
    difficulty: "medium",
    prompt:
      "Which statements about generalization in deep computer vision are most consistent with the lecture?",
    options: [
      {
        text: "A vision model should learn features that remain useful despite variations such as viewpoint, illumination, scale, and occlusion.",
        isCorrect: true,
      },
      {
        text: "One reason manual feature engineering is difficult is that real visual categories have large intra-class variation.",
        isCorrect: true,
      },
      {
        text: "Learning features directly from data can be more robust than relying only on hand-written visual rules.",
        isCorrect: true,
      },
      {
        text: "Once convolution is used, robustness to all visual variation is guaranteed automatically.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture repeatedly emphasizes that real-world vision is hard because images vary in many ways even within the same class. CNNs help by learning useful features from data, but no architecture alone guarantees complete robustness to every variation.",
  },
];
