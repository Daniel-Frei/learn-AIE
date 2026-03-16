import { Question } from "../../quiz";

export const MIT15773L4ComputerVisionTransferLearningQuestions: Question[] = [
  {
    id: "mit15773-l4-q01",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which of the following limitations occur when images are flattened into vectors and fed directly into dense neural networks?",
    options: [
      {
        text: "Flattening removes spatial adjacency information between pixels.",
        isCorrect: true,
      },
      {
        text: "Flattening guarantees translation invariance.",
        isCorrect: false,
      },
      {
        text: "Flattening automatically reduces the number of model parameters.",
        isCorrect: false,
      },
      {
        text: "Flattening ensures that vertical lines can be detected regardless of position.",
        isCorrect: false,
      },
    ],
    explanation:
      "When images are flattened into vectors, the model loses spatial relationships between nearby pixels. Dense layers treat every pixel independently, which ignores the structure of images. Convolutional neural networks address this limitation by preserving spatial structure and exploiting local patterns.",
  },

  {
    id: "mit15773-l4-q02",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe convolutional filters in convolutional neural networks (CNNs)?",
    options: [
      {
        text: "A convolutional filter is typically a small matrix of weights.",
        isCorrect: true,
      },
      {
        text: "Filters can detect visual patterns such as edges or curves.",
        isCorrect: true,
      },
      {
        text: "Each convolutional layer usually contains multiple filters.",
        isCorrect: true,
      },
      {
        text: "Filters are usually applied repeatedly across the entire image.",
        isCorrect: true,
      },
    ],
    explanation:
      "A convolutional filter is a small matrix of weights that slides across an image and performs a dot product with the local pixel values. Different filters specialize in detecting patterns such as edges, corners, or textures. Convolutional layers contain multiple filters that scan the entire image.",
  },

  {
    id: "mit15773-l4-q03",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "During the convolution operation, which steps are typically performed?",
    options: [
      { text: "Overlay the filter on a region of the image.", isCorrect: true },
      { text: "Multiply corresponding values and sum them.", isCorrect: true },
      { text: "Apply an activation function such as ReLU.", isCorrect: true },
      {
        text: "Repeat this operation across many spatial locations as the filter slides over the image.",
        isCorrect: true,
      },
    ],
    explanation:
      "The convolution operation involves placing the filter over a small patch of the image, multiplying corresponding values, and summing the results. The resulting value is often passed through an activation function such as ReLU. Softmax is typically used only in the final classification layer.",
  },

  {
    id: "mit15773-l4-q04",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Suppose a grayscale image has dimensions \\(6 \\times 6\\) and a \\(3 \\times 3\\) convolutional filter is applied with stride 1 and no padding. What is the spatial dimension of the output feature map?",
    options: [
      { text: "\\(4 \\times 4\\)", isCorrect: true },
      { text: "\\(6 \\times 6\\)", isCorrect: false },
      { text: "\\(3 \\times 3\\)", isCorrect: false },
      { text: "\\(5 \\times 5\\)", isCorrect: false },
    ],
    explanation:
      "The output dimension of a convolution without padding is given by \\(n - f + 1\\), where \\(n\\) is the input size and \\(f\\) is the filter size. Here, \\(6 - 3 + 1 = 4\\), so the output feature map is \\(4 \\times 4\\).",
  },

  {
    id: "mit15773-l4-q05",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe convolution with color images?",
    options: [
      {
        text: "Color images typically have three channels: red, green, and blue.",
        isCorrect: true,
      },
      {
        text: "The depth of a convolutional filter must match the depth of the input.",
        isCorrect: true,
      },
      {
        text: "A \\(3 \\times 3\\) filter for an RGB image usually contains \\(3 \\times 3 \\times 3\\) weights.",
        isCorrect: true,
      },
      {
        text: "Information from different color channels is combined within each convolutional filter.",
        isCorrect: true,
      },
    ],
    explanation:
      "Color images are typically represented as tensors with three channels (RGB). Convolutional filters must have the same depth as the input so that they can combine information across channels. The convolution operation multiplies values across all channels simultaneously.",
  },

  {
    id: "mit15773-l4-q06",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements about pooling layers are correct?",
    options: [
      {
        text: "Pooling layers reduce the spatial dimensions of feature maps.",
        isCorrect: true,
      },
      {
        text: "Pooling layers contain trainable parameters that must be learned.",
        isCorrect: false,
      },
      {
        text: "Pooling helps summarize information in a region.",
        isCorrect: true,
      },
      {
        text: "Pooling increases the number of filters in the network.",
        isCorrect: false,
      },
    ],
    explanation:
      "Pooling layers reduce the height and width of feature maps while preserving important information. They do not contain trainable weights. Instead, they apply simple operations such as taking the maximum or average value within a small window.",
  },

  {
    id: "mit15773-l4-q07",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements correctly describe max pooling?",
    options: [
      {
        text: "Max pooling selects the largest value within a local region.",
        isCorrect: true,
      },
      {
        text: "Max pooling can be interpreted as detecting whether a feature exists in a region.",
        isCorrect: true,
      },
      {
        text: "Max pooling usually reduces spatial resolution.",
        isCorrect: true,
      },
      {
        text: "Max pooling is similar to an OR-like operation across nearby activations.",
        isCorrect: true,
      },
    ],
    explanation:
      "Max pooling selects the maximum value within a region of the feature map. This acts as a form of feature detection because if any strong activation appears in the region, the pooled value will remain high. It reduces spatial dimensions rather than increasing them.",
  },

  {
    id: "mit15773-l4-q08",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements about convolutional neural network architecture are correct?",
    options: [
      {
        text: "CNNs typically contain multiple convolutional blocks.",
        isCorrect: true,
      },
      {
        text: "A convolutional block often consists of convolution layers followed by pooling.",
        isCorrect: true,
      },
      {
        text: "Later layers in CNNs usually have smaller spatial dimensions.",
        isCorrect: true,
      },
      {
        text: "CNNs often end with dense layers for classification.",
        isCorrect: true,
      },
    ],
    explanation:
      "Typical CNN architectures consist of convolutional blocks that extract features from images. These blocks often include convolution layers followed by pooling layers. As the network gets deeper, spatial resolution decreases while the number of filters increases.",
  },

  {
    id: "mit15773-l4-q09",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Why do convolutional neural networks require far fewer parameters than fully connected networks for image processing?",
    options: [
      {
        text: "CNNs reuse the same filter weights across the entire image.",
        isCorrect: true,
      },
      {
        text: "CNNs exploit local connectivity between nearby pixels.",
        isCorrect: true,
      },
      {
        text: "CNNs preserve and exploit spatial structure rather than ignoring it.",
        isCorrect: true,
      },
      {
        text: "CNNs reduce parameter counts through weight sharing.",
        isCorrect: true,
      },
    ],
    explanation:
      "Convolutional networks drastically reduce parameters by sharing weights across spatial locations. Instead of learning separate weights for every pixel connection, the same filter is applied across the image. This enables efficient feature detection and improves generalization.",
  },

  {
    id: "mit15773-l4-q10",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements about translation invariance in CNNs are correct?",
    options: [
      {
        text: "Translation invariance means a feature can be recognized regardless of its position in the image.",
        isCorrect: true,
      },
      {
        text: "Convolutional filters can detect the same pattern in different locations.",
        isCorrect: true,
      },
      {
        text: "Translation invariance does not require flattening the image first.",
        isCorrect: true,
      },
      {
        text: "Weight sharing across spatial positions supports translation invariance.",
        isCorrect: true,
      },
    ],
    explanation:
      "Translation invariance means that patterns such as edges or shapes can be recognized anywhere in the image. Convolution achieves this because the same filter is applied across all spatial locations. This allows the network to detect features regardless of their position.",
  },

  {
    id: "mit15773-l4-q11",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements about hierarchical feature learning in CNNs are correct?",
    options: [
      {
        text: "Early layers often detect simple features such as edges.",
        isCorrect: true,
      },
      {
        text: "Intermediate layers always detect exactly the same low-level edges as the first layer.",
        isCorrect: false,
      },
      { text: "Deeper layers can represent complex objects.", isCorrect: true },
      {
        text: "All layers detect exactly the same type of features.",
        isCorrect: false,
      },
    ],
    explanation:
      "CNNs learn hierarchical representations of images. Early layers detect basic features like edges, while deeper layers combine these into shapes and object parts. The deepest layers often represent complex semantic concepts such as faces or objects.",
  },

  {
    id: "mit15773-l4-q12",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Suppose a convolutional layer has 32 filters. Which statements are correct about the resulting output tensor?",
    options: [
      {
        text: "The output tensor will have depth equal to the number of filters.",
        isCorrect: true,
      },
      { text: "Each filter produces its own feature map.", isCorrect: true },
      {
        text: "The depth of the output tensor will always equal the input image width rather than 32.",
        isCorrect: false,
      },
      {
        text: "Each filter produces multiple output tensors independently.",
        isCorrect: false,
      },
    ],
    explanation:
      "In convolutional layers, each filter generates one feature map. If a layer contains 32 filters, the output tensor will have depth 32. These feature maps are stacked together to form the final output tensor.",
  },

  {
    id: "mit15773-l4-q13",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements about data augmentation are correct?",
    options: [
      {
        text: "Data augmentation creates additional training examples from existing data.",
        isCorrect: true,
      },
      {
        text: "Common augmentation techniques include rotation and zoom.",
        isCorrect: true,
      },
      {
        text: "Augmented images must preserve the original label meaning.",
        isCorrect: true,
      },
      {
        text: "Data augmentation can improve generalization without guaranteeing perfection.",
        isCorrect: true,
      },
    ],
    explanation:
      "Data augmentation increases the effective size of the training dataset by applying transformations such as rotation or zoom. These transformations must preserve the semantic meaning of the image so that the label remains valid.",
  },

  {
    id: "mit15773-l4-q14",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about transfer learning are correct?",
    options: [
      {
        text: "Transfer learning reuses knowledge from a model trained on another dataset.",
        isCorrect: true,
      },
      {
        text: "Pretrained models can provide useful feature representations.",
        isCorrect: true,
      },
      {
        text: "Transfer learning often works well with small datasets.",
        isCorrect: true,
      },
      {
        text: "Transfer learning often starts from pretrained weights instead of training from scratch.",
        isCorrect: true,
      },
    ],
    explanation:
      "Transfer learning allows models trained on large datasets to be adapted for new tasks. The pretrained model provides useful feature representations that can be reused. This approach is especially useful when training data is limited.",
  },

  {
    id: "mit15773-l4-q15",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements correctly describe the ImageNet dataset?",
    options: [
      {
        text: "ImageNet contains millions of labeled images.",
        isCorrect: true,
      },
      {
        text: "ImageNet includes thousands of object categories.",
        isCorrect: true,
      },
      {
        text: "Many pretrained CNN models were trained on ImageNet.",
        isCorrect: true,
      },
      {
        text: "ImageNet contains large numbers of color images, not only grayscale images.",
        isCorrect: true,
      },
    ],
    explanation:
      "ImageNet is a large-scale dataset with millions of labeled images across roughly 1000 object categories. It played a major role in advancing deep learning for computer vision. Many pretrained models such as AlexNet and ResNet were trained on this dataset.",
  },

  {
    id: "mit15773-l4-q16",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about the ResNet architecture are correct?",
    options: [
      {
        text: "ResNet is a deep convolutional neural network architecture.",
        isCorrect: true,
      },
      {
        text: "ResNet models are never pretrained on ImageNet.",
        isCorrect: false,
      },
      {
        text: "ResNet can be used as a feature extractor for transfer learning.",
        isCorrect: true,
      },
      {
        text: "ResNet can only classify handbags and shoes.",
        isCorrect: false,
      },
    ],
    explanation:
      "ResNet is a deep convolutional network architecture widely used in computer vision. It is commonly pretrained on ImageNet and reused for other tasks through transfer learning. It can be applied to many classification problems, not just specific categories.",
  },

  {
    id: "mit15773-l4-q17",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which of the following statements about convolutional neural networks are correct?",
    options: [
      {
        text: "CNNs are particularly ineffective for image data.",
        isCorrect: false,
      },
      { text: "CNNs ignore spatial structure in images.", isCorrect: false },
      {
        text: "CNNs typically combine convolutional layers and pooling layers.",
        isCorrect: true,
      },
      { text: "CNNs always require billions of parameters.", isCorrect: false },
    ],
    explanation:
      "Convolutional neural networks are designed to process spatial data such as images. They use convolution and pooling layers to extract hierarchical features efficiently. Due to weight sharing, CNNs often require far fewer parameters than fully connected networks.",
  },

  {
    id: "mit15773-l4-q18",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements about feature extraction using pretrained networks are correct?",
    options: [
      {
        text: "Images can be passed through a pretrained network up to the final layer.",
        isCorrect: true,
      },
      {
        text: "The resulting tensor can serve as a learned feature representation.",
        isCorrect: true,
      },
      {
        text: "A small classifier can be trained on top of the extracted features.",
        isCorrect: true,
      },
      {
        text: "The pretrained network must always be retrained from scratch.",
        isCorrect: false,
      },
    ],
    explanation:
      "In feature extraction, images are passed through a pretrained network and the outputs of an intermediate layer are used as feature representations. These features can then be used as inputs to a smaller classifier. This avoids retraining the entire network.",
  },

  {
    id: "mit15773-l4-q19",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about CNN feature maps are correct?",
    options: [
      {
        text: "Feature maps are random tensors unrelated to convolutional filters.",
        isCorrect: false,
      },
      {
        text: "All filters produce the same feature map.",
        isCorrect: false,
      },
      {
        text: "Feature maps always have the same spatial size as the input.",
        isCorrect: false,
      },
      {
        text: "Pooling layers often reduce the spatial size of feature maps.",
        isCorrect: true,
      },
    ],
    explanation:
      "A feature map represents how strongly a filter responds to patterns across an image. Each filter produces its own feature map. Pooling layers and convolution operations often reduce spatial resolution while preserving important features.",
  },

  {
    id: "mit15773-l4-q20",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements about training convolutional neural networks are correct?",
    options: [
      {
        text: "Filter weights cannot be learned using backpropagation.",
        isCorrect: false,
      },
      {
        text: "Filters are fixed, non-trainable parameters in the network.",
        isCorrect: false,
      },
      {
        text: "Gradient-based optimization algorithms such as stochastic gradient descent can update filter weights.",
        isCorrect: true,
      },
      {
        text: "Filters must always be manually designed by humans.",
        isCorrect: false,
      },
    ],
    explanation:
      "In modern CNNs, convolutional filter weights are learned automatically from data. They are treated as trainable parameters and optimized using backpropagation and gradient-based optimization methods such as stochastic gradient descent. Earlier computer vision methods relied on hand-designed filters.",
  },

  {
    id: "mit15773-l4-q21",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which of the following statements about convolutional filters are correct?",
    options: [
      {
        text: "The values inside a convolutional filter are learned during training.",
        isCorrect: true,
      },
      {
        text: "Filters are manually designed in modern CNN training pipelines.",
        isCorrect: false,
      },
      {
        text: "Each filter scans the entire image by sliding across spatial positions.",
        isCorrect: true,
      },
      {
        text: "Filters are only applied once to the center of the image.",
        isCorrect: false,
      },
    ],
    explanation:
      "In modern convolutional neural networks, the values of convolutional filters are learned automatically through backpropagation. A filter slides across the image, performing the convolution operation at each spatial location to detect patterns such as edges or textures.",
  },

  {
    id: "mit15773-l4-q22",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the relationship between convolutional filters and feature maps?",
    options: [
      {
        text: "Each convolutional filter produces one feature map.",
        isCorrect: true,
      },
      {
        text: "Feature maps highlight where specific patterns appear in the image.",
        isCorrect: true,
      },
      {
        text: "Feature maps are always the same size as the original input image.",
        isCorrect: false,
      },
      {
        text: "The depth of the output tensor equals the number of filters.",
        isCorrect: true,
      },
    ],
    explanation:
      "Each filter generates one feature map that highlights locations where that filter strongly activates. Multiple filters therefore produce multiple feature maps that are stacked into a tensor whose depth equals the number of filters.",
  },

  {
    id: "mit15773-l4-q23",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Suppose an RGB image has dimensions \\(224 \\times 224 \\times 3\\). A convolutional layer uses 32 filters of size \\(3 \\times 3\\). Ignoring biases, how many weight parameters are learned in this layer?",
    options: [
      { text: "\\(3 \\times 3 \\times 3 \\times 32 = 864\\)", isCorrect: true },
      { text: "\\(224 \\times 224 \\times 3 \\times 32\\)", isCorrect: false },
      {
        text: "The number of parameters does not depend on the image size.",
        isCorrect: true,
      },
      {
        text: "The parameter count equals \\(3 \\times 3 \\times 32\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The number of parameters depends on filter size and input depth. Each filter contains \\(3 \\times 3 \\times 3 = 27\\) weights because it spans the RGB channels. With 32 filters, the total parameter count is \\(27 \\times 32 = 864\\). The parameter count does not depend on the spatial size of the image.",
  },

  {
    id: "mit15773-l4-q24",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which of the following statements about spatial adjacency in images are correct?",
    options: [
      {
        text: "Pixels close to each other in an image are often correlated.",
        isCorrect: true,
      },
      {
        text: "CNNs exploit spatial adjacency through local receptive fields.",
        isCorrect: true,
      },
      {
        text: "Dense neural networks automatically preserve spatial relationships.",
        isCorrect: false,
      },
      {
        text: "Spatial adjacency is irrelevant for computer vision tasks.",
        isCorrect: false,
      },
    ],
    explanation:
      "In images, nearby pixels often represent related parts of objects. CNNs exploit this by using filters that operate on small local regions. Dense networks do not preserve spatial structure because they treat inputs as flat vectors.",
  },

  {
    id: "mit15773-l4-q25",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements about the receptive field of neurons in CNNs are correct?",
    options: [
      {
        text: "The receptive field refers only to the output region a neuron produces.",
        isCorrect: false,
      },
      {
        text: "Receptive fields shrink in deeper layers.",
        isCorrect: false,
      },
      {
        text: "Pooling layers contribute to increasing the effective receptive field.",
        isCorrect: true,
      },
      {
        text: "The receptive field always stays constant across layers.",
        isCorrect: false,
      },
    ],
    explanation:
      "The receptive field describes which part of the input affects a particular neuron. As layers are stacked, each neuron aggregates information from larger regions of the input image. Pooling layers and convolution layers both increase the effective receptive field.",
  },

  {
    id: "mit15773-l4-q26",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about average pooling are correct?",
    options: [
      {
        text: "Average pooling computes the mean of values in a region.",
        isCorrect: true,
      },
      {
        text: "Average pooling introduces trainable parameters.",
        isCorrect: false,
      },
      {
        text: "Average pooling can reduce spatial resolution.",
        isCorrect: true,
      },
      {
        text: "Average pooling always performs better than max pooling.",
        isCorrect: false,
      },
    ],
    explanation:
      "Average pooling replaces a region with the average of its values. Like max pooling, it reduces spatial dimensions but contains no learnable parameters. Whether average or max pooling works better depends on the problem.",
  },

  {
    id: "mit15773-l4-q27",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements about the parameter efficiency of CNNs compared to fully connected layers are correct?",
    options: [
      {
        text: "CNNs avoid weight sharing, which is why they need fewer parameters.",
        isCorrect: false,
      },
      { text: "CNNs ignore local connectivity patterns.", isCorrect: false },
      {
        text: "Fully connected layers typically require far more parameters for image inputs.",
        isCorrect: true,
      },
      {
        text: "CNNs require more parameters than dense layers when processing images.",
        isCorrect: false,
      },
    ],
    explanation:
      "CNNs are parameter efficient because they reuse the same filter across spatial positions. Dense layers must connect every input pixel to every neuron, which leads to extremely large parameter counts for images.",
  },

  {
    id: "mit15773-l4-q28",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements about convolutional blocks are correct?",
    options: [
      {
        text: "A convolutional block usually contains only pooling layers with no convolutions.",
        isCorrect: false,
      },
      {
        text: "Convolutional blocks are not used to extract hierarchical image features.",
        isCorrect: false,
      },
      {
        text: "Pooling layers typically reduce the spatial size of feature maps.",
        isCorrect: true,
      },
      {
        text: "Convolutional blocks are used only for text data.",
        isCorrect: false,
      },
    ],
    explanation:
      "A convolutional block usually consists of one or more convolution layers followed by pooling. These blocks progressively extract higher-level features while reducing spatial dimensions.",
  },

  {
    id: "mit15773-l4-q29",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements about flattening in CNN architectures are correct?",
    options: [
      { text: "Flattening converts a tensor into a vector.", isCorrect: true },
      {
        text: "Flattening is often used before dense layers.",
        isCorrect: true,
      },
      {
        text: "Flattening preserves the spatial structure of feature maps.",
        isCorrect: false,
      },
      {
        text: "Flattening always increases the dimensionality of the data.",
        isCorrect: false,
      },
    ],
    explanation:
      "Flattening reshapes multidimensional tensors into one-dimensional vectors. This step is typically used before feeding the representation into dense layers for classification.",
  },

  {
    id: "mit15773-l4-q30",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about fine-tuning pretrained models are correct?",
    options: [
      {
        text: "Fine-tuning keeps pretrained weights frozen and never updates them.",
        isCorrect: false,
      },
      {
        text: "Fine-tuning typically starts from random initialization rather than pretrained weights.",
        isCorrect: false,
      },
      {
        text: "Fine-tuning can improve performance on new tasks.",
        isCorrect: true,
      },
      {
        text: "Fine-tuning requires discarding the pretrained weights entirely.",
        isCorrect: false,
      },
    ],
    explanation:
      "In fine-tuning, a pretrained model is used as a starting point and its weights are updated during training on a new dataset. This allows the model to adapt to the new task while leveraging previously learned features.",
  },

  {
    id: "mit15773-l4-q31",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements about normalization of image inputs are correct?",
    options: [
      {
        text: "Pixel values are often divided by 255 to scale them into the range \\([0,1]\\).",
        isCorrect: true,
      },
      {
        text: "Normalization can help stabilize neural network training.",
        isCorrect: true,
      },
      {
        text: "Normalization changes the semantic meaning of images.",
        isCorrect: false,
      },
      {
        text: "Normalization eliminates the need for activation functions.",
        isCorrect: false,
      },
    ],
    explanation:
      "Image normalization rescales pixel values to smaller ranges such as \\([0,1]\\). This helps stabilize training and improves numerical behavior. It does not change the semantic content of the image.",
  },

  {
    id: "mit15773-l4-q32",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about data augmentation are correct?",
    options: [
      {
        text: "Data augmentation increases the diversity of training examples.",
        isCorrect: true,
      },
      { text: "Augmentation can help reduce overfitting.", isCorrect: true },
      {
        text: "Augmentation always produces perfectly realistic images.",
        isCorrect: false,
      },
      {
        text: "Augmentation techniques include flips, rotations, and zooming.",
        isCorrect: true,
      },
    ],
    explanation:
      "Data augmentation increases dataset diversity by creating modified versions of training images. Techniques include rotations, flips, and zooming. These transformations help improve generalization but do not guarantee realism.",
  },

  {
    id: "mit15773-l4-q33",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements about transfer learning with pretrained CNNs are correct?",
    options: [
      {
        text: "Earlier layers often learn highly task-specific output labels.",
        isCorrect: false,
      },
      {
        text: "Later layers tend to learn only generic edge detectors.",
        isCorrect: false,
      },
      {
        text: "Pretrained networks can be reused for new classification tasks.",
        isCorrect: true,
      },
      {
        text: "Transfer learning only works if the datasets are identical.",
        isCorrect: false,
      },
    ],
    explanation:
      "Early layers in CNNs learn general patterns such as edges and textures that are useful across many tasks. Later layers learn more task-specific representations. This makes pretrained networks valuable for transfer learning.",
  },

  {
    id: "mit15773-l4-q34",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements about convolutional neural networks are correct?",
    options: [
      {
        text: "CNNs are designed to process grid-like data such as images.",
        isCorrect: true,
      },
      { text: "CNNs rely heavily on convolutional filters.", isCorrect: true },
      {
        text: "CNNs exploit spatial relationships between pixels.",
        isCorrect: true,
      },
      {
        text: "CNNs cannot be used for image classification.",
        isCorrect: false,
      },
    ],
    explanation:
      "CNNs are specialized neural networks designed for spatial data such as images. They rely on convolutional filters to extract patterns and hierarchical features.",
  },

  {
    id: "mit15773-l4-q35",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements about model training with small image datasets are correct?",
    options: [
      {
        text: "Small datasets increase the risk of overfitting.",
        isCorrect: true,
      },
      {
        text: "Transfer learning can help when labeled data is limited.",
        isCorrect: true,
      },
      {
        text: "Data augmentation can effectively increase dataset diversity.",
        isCorrect: true,
      },
      { text: "CNNs cannot be trained with small datasets.", isCorrect: false },
    ],
    explanation:
      "Small datasets often lead to overfitting because the model memorizes the training data. Transfer learning and data augmentation help mitigate this problem by leveraging existing knowledge and expanding training examples.",
  },

  {
    id: "mit15773-l4-q36",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about pretrained feature extraction are correct?",
    options: [
      {
        text: "Images can be passed through a pretrained network up to an intermediate layer.",
        isCorrect: true,
      },
      {
        text: "The resulting tensor acts as a feature representation of the image.",
        isCorrect: true,
      },
      {
        text: "A new classifier can be trained on top of these features.",
        isCorrect: true,
      },
      {
        text: "The pretrained network must always be retrained fully.",
        isCorrect: false,
      },
    ],
    explanation:
      "In feature extraction, images are processed through a pretrained network and the resulting intermediate activations are used as features. These features can then be fed into a smaller classifier.",
  },

  {
    id: "mit15773-l4-q37",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about CNN depth are correct?",
    options: [
      {
        text: "Depth increases when more filters are added to layers.",
        isCorrect: true,
      },
      {
        text: "Deeper networks can capture more complex features.",
        isCorrect: true,
      },
      {
        text: "Depth refers only to image height and width.",
        isCorrect: false,
      },
      {
        text: "Increasing depth allows more feature combinations to be represented.",
        isCorrect: true,
      },
    ],
    explanation:
      "The depth of a CNN refers to the number of feature channels produced by filters. Increasing depth allows networks to represent more complex combinations of features.",
  },

  {
    id: "mit15773-l4-q38",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements about convolution operations are correct?",
    options: [
      { text: "The filter slides across the input image.", isCorrect: true },
      {
        text: "Each filter computes a weighted sum of pixel values.",
        isCorrect: true,
      },
      {
        text: "The same filter weights are reused across spatial locations.",
        isCorrect: true,
      },
      { text: "Convolution operations cannot detect edges.", isCorrect: false },
    ],
    explanation:
      "A convolution filter slides across the image and computes weighted sums of pixel values. Because the same weights are reused across locations, CNNs efficiently detect patterns such as edges.",
  },

  {
    id: "mit15773-l4-q39",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements about model evaluation are correct?",
    options: [
      {
        text: "Validation accuracy measures performance on unseen validation data.",
        isCorrect: true,
      },
      {
        text: "Training accuracy measures performance on training data.",
        isCorrect: true,
      },
      {
        text: "Test accuracy should be evaluated only after training is complete.",
        isCorrect: true,
      },
      {
        text: "Validation accuracy should always be higher than training accuracy.",
        isCorrect: false,
      },
    ],
    explanation:
      "Training accuracy measures how well the model fits the training data, while validation accuracy evaluates generalization during training. Test accuracy is usually measured only after training is finalized.",
  },

  {
    id: "mit15773-l4-q40",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about CNN feature hierarchies are correct?",
    options: [
      {
        text: "Early CNN layers detect simple patterns such as edges.",
        isCorrect: true,
      },
      {
        text: "Middle layers combine edges into shapes or textures.",
        isCorrect: true,
      },
      {
        text: "Deeper layers detect higher-level concepts such as object parts.",
        isCorrect: true,
      },
      { text: "All CNN layers learn identical features.", isCorrect: false },
    ],
    explanation:
      "CNNs learn hierarchical representations. Early layers detect simple patterns like edges, while deeper layers combine them into more complex shapes and objects.",
  },
];
