import { Question } from "../../quiz";

// lib/lectures/MIT 15.773 Hands-On Deep Learning Spring 2024/L1_ Introduction to Neural Networks and Deep Learning.ts

export const L1_IntroductionToNeuralNetworksAndDeepLearning: Question[] = [
  {
    id: "mit15773-l1-q01",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly distinguish Artificial Intelligence (AI) and Machine Learning (ML) as used in modern practice?",
    options: [
      {
        text: "Machine Learning is a common practical approach used to achieve Artificial Intelligence goals by learning from data.",
        isCorrect: true,
      },
      {
        text: "Artificial Intelligence is an umbrella term that includes Machine Learning as one of its major sub-approaches.",
        isCorrect: true,
      },
      {
        text: "In many real-world conversations, people say “Artificial Intelligence” when they really mean “Machine Learning.”",
        isCorrect: true,
      },
      {
        text: "Machine Learning always relies on hand-written IF–THEN rules provided by domain experts.",
        isCorrect: false,
      },
    ],
    explanation:
      "AI is the broad goal of making computers perform tasks that typically require human intelligence. ML is a widely used approach within AI that learns mappings from data rather than relying purely on hand-coded rules, and people often casually say “AI” when they mean ML in practice.",
  },

  {
    id: "mit15773-l1-q02",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements describe why the traditional IF–THEN rule-based approach to AI struggles in the real world?",
    options: [
      {
        text: "Real-world environments contain many edge cases and novel situations that were not anticipated when rules were written.",
        isCorrect: true,
      },
      {
        text: "Rule-based systems often generalize poorly because rules are brittle outside the scenarios they were designed for.",
        isCorrect: true,
      },
      {
        text: "Polanyi’s paradox (“we know more than we can tell”) makes it hard for experts to fully articulate how they perform many tasks.",
        isCorrect: true,
      },
      {
        text: "Rule-based systems fail mainly because computers cannot execute many rules quickly enough.",
        isCorrect: false,
      },
    ],
    explanation:
      "The key issues are brittleness and poor generalization: the world presents unbounded variety, and hand-written rules rarely cover all meaningful cases. Polanyi’s paradox adds a deeper limitation: people can often do tasks easily but cannot reliably explain the decision procedure in a way that can be turned into complete rules.",
  },

  {
    id: "mit15773-l1-q03",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "A core distinction in the lecture is between structured and unstructured input data. Which statements are correct?",
    options: [
      {
        text: "Structured data is typically easy to put into rows and columns (e.g., spreadsheets) where each feature is a number or a simple categorical variable that can be numericalized.",
        isCorrect: true,
      },
      {
        text: "Unstructured data (e.g., images, audio, text) often requires learning or engineering a representation before simple models can use it effectively.",
        isCorrect: true,
      },
      {
        text: "A raw image can be viewed as multiple grids of pixel intensity values (e.g., red/green/blue channels), which do not directly encode high-level meaning like “dog” or “sky.”",
        isCorrect: true,
      },
      {
        text: "Unstructured data is unusable for Machine Learning because it cannot be represented numerically in a computer.",
        isCorrect: false,
      },
    ],
    explanation:
      "Unstructured data is absolutely numeric in computers (pixels, waveforms, tokens), but the challenge is that the raw numeric form does not directly line up with the semantic concepts we care about. Historically, practitioners often engineered intermediate representations (“features”) to bridge this gap before applying standard models.",
  },

  {
    id: "mit15773-l1-q04",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe 'representation' (also called feature extraction/feature engineering in many contexts) for unstructured data?",
    options: [
      {
        text: "A representation is a transformed form of the raw input that makes downstream prediction easier (e.g., from pixels to beak length / wingspan / primary color for bird classification).",
        isCorrect: true,
      },
      {
        text: "Historically, creating good representations for unstructured data often required significant manual effort by researchers or domain experts.",
        isCorrect: true,
      },
      {
        text: "Once a strong representation is available, relatively simple models (e.g., linear regression or logistic regression) can sometimes perform surprisingly well.",
        isCorrect: true,
      },
      {
        text: "Representations are only needed for images and never for text or audio.",
        isCorrect: false,
      },
    ],
    explanation:
      "A representation is any transformation of raw data into a form that better exposes the signal for the task. Historically this was a major human bottleneck, and deep learning’s key advantage is learning such representations automatically from raw inputs across many modalities, including images, audio, and text.",
  },

  {
    id: "mit15773-l1-q05",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly characterize Deep Learning (DL) relative to Machine Learning (ML) and the 'human bottleneck'?",
    options: [
      {
        text: "Deep Learning is a subfield within Machine Learning that is especially effective at learning representations directly from raw, unstructured data.",
        isCorrect: true,
      },
      {
        text: "Deep Learning can reduce the need for manual feature engineering by learning intermediate representations automatically.",
        isCorrect: true,
      },
      {
        text: "Deep Learning’s success is often described as demolishing a 'human bottleneck' where researchers previously spent years hand-crafting representations.",
        isCorrect: true,
      },
      {
        text: "Deep Learning replaces Machine Learning; they are disjoint fields with no overlap.",
        isCorrect: false,
      },
    ],
    explanation:
      "Deep learning sits inside machine learning, not outside it. The central shift emphasized in the lecture is that deep networks can learn powerful intermediate representations automatically, reducing reliance on hand-crafted features and enabling effective learning on raw unstructured inputs.",
  },

  {
    id: "mit15773-l1-q06",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "The lecture describes three forces that jointly enabled the modern Deep Learning breakthrough. Which options match those forces?",
    options: [
      { text: "New algorithmic ideas.", isCorrect: true },
      {
        text: "Access to large amounts of data due to broad digitization.",
        isCorrect: true,
      },
      {
        text: "Increased compute power via parallel hardware such as Graphics Processing Units (GPUs).",
        isCorrect: true,
      },
      {
        text: "Replacing numerical computation with hand-written symbolic IF–THEN rules.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture highlights a convergence: better algorithms, more data, and significantly more compute (especially parallel compute on GPUs). Together these revitalized older ideas like neural networks and made deep learning practical at large scale.",
  },

  {
    id: "mit15773-l1-q07",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about Generative Artificial Intelligence (Generative AI) are correct in the lecture’s framing?",
    options: [
      {
        text: "Generative AI refers to systems that can generate unstructured outputs such as text, images, or audio.",
        isCorrect: true,
      },
      {
        text: "Generative AI is typically built using deep learning methods, so it can be viewed as sitting within Deep Learning.",
        isCorrect: true,
      },
      {
        text: "Many deep learning applications (e.g., classification/detection behind sensors) are not generative AI.",
        isCorrect: true,
      },
      {
        text: "Generative AI is synonymous with traditional rule-based AI from the 1950s.",
        isCorrect: false,
      },
    ],
    explanation:
      "In the lecture’s hierarchy, Generative AI is a type of deep learning focused on producing unstructured outputs. Deep learning also includes many non-generative tasks like classification, detection, and prediction from sensor data.",
  },

  {
    id: "mit15773-l1-q08",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly explain why attaching Deep Learning to sensors is a powerful product idea pattern?",
    options: [
      {
        text: "Sensors (cameras, microphones, etc.) naturally produce unstructured data, and deep learning can learn useful representations from that raw data.",
        isCorrect: true,
      },
      {
        text: "Once perception is automated (e.g., detection/classification), a system can enable downstream actions like counting, monitoring, or decision support.",
        isCorrect: true,
      },
      {
        text: "The key idea generalizes across many sensors: camera → vision models, microphone → audio models, and so on.",
        isCorrect: true,
      },
      {
        text: "This pattern only works when the sensor data is already in a clean spreadsheet format.",
        isCorrect: false,
      },
    ],
    explanation:
      "A sensor is essentially a stream of unstructured inputs. Deep learning lets you add perception capabilities—detecting, recognizing, classifying—directly behind the sensor, enabling new products and services that turn raw sensory data into structured decisions or insights.",
  },

  {
    id: "mit15773-l1-q09",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly define logistic regression in the binary classification setting?",
    options: [
      {
        text: "Logistic regression often maps a linear score \\(z\\) to a probability using the sigmoid \\(\\sigma(z)=\\frac{1}{1+e^{-z}}\\).",
        isCorrect: true,
      },
      {
        text: "In logistic regression, the model output can be interpreted as a value in \\((0,1)\\), commonly treated as \\(P(y=1\\mid x)\\).",
        isCorrect: true,
      },
      {
        text: "A typical linear score has the form \\(z=b+\\sum_i w_i x_i\\), where \\(w_i\\) are coefficients and \\(b\\) is an intercept.",
        isCorrect: true,
      },
      {
        text: "Logistic regression outputs any real number in \\(( -\\infty, \\infty )\\) and does not constrain outputs to probabilities.",
        isCorrect: false,
      },
    ],
    explanation:
      "Binary logistic regression uses a linear score \\(z\\) and applies the sigmoid to turn it into a probability-like output. This output is bounded between 0 and 1, which is why the sigmoid is a natural choice for binary classification probabilities.",
  },

  {
    id: "mit15773-l1-q10",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "In neural network terminology, which statements about weights and biases are correct?",
    options: [
      {
        text: "The multipliers applied to inputs (called coefficients in regression) are typically called weights in neural networks.",
        isCorrect: true,
      },
      {
        text: "The intercept term in regression is typically called a bias in neural networks.",
        isCorrect: true,
      },
      {
        text: "A model’s learned parameters can be described as its weights and biases (sometimes casually just 'weights').",
        isCorrect: true,
      },
      {
        text: "A bias must always be zero in a neural network layer to ensure stability.",
        isCorrect: false,
      },
    ],
    explanation:
      "Neural-network language mostly renames familiar regression pieces: coefficients → weights and intercept → bias. Bias terms are generally learnable and nonzero; they allow shifting activations and often improve expressiveness and training behavior.",
  },

  {
    id: "mit15773-l1-q11",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A key reason neural networks need nonlinear activation functions is that stacking only linear transformations is not useful. Which statements correctly justify that claim?",
    options: [
      {
        text: "A composition of linear functions is still a linear function, so multiple linear layers without nonlinearities collapse to a single linear layer.",
        isCorrect: true,
      },
      {
        text: "Without nonlinearities, adding more layers does not increase the class of functions the network can represent beyond linear mappings.",
        isCorrect: true,
      },
      {
        text: "Nonlinear activations allow the network to represent more complex input–output relationships than a single linear/logistic model.",
        isCorrect: true,
      },
      {
        text: "Nonlinear activations are used mainly to ensure the model’s parameters sum to 1 like probabilities.",
        isCorrect: false,
      },
    ],
    explanation:
      "If you stack only linear maps, the result is mathematically equivalent to one linear map, so depth provides no added representational power. Nonlinear activations break this equivalence, enabling the network to model complex, non-linear relationships and learn richer intermediate representations.",
  },

  {
    id: "mit15773-l1-q12",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the Rectified Linear Unit (ReLU) activation function?",
    options: [
      {
        text: "ReLU is commonly defined as \\(\\mathrm{ReLU}(a)=\\max(0,a)\\).",
        isCorrect: true,
      },
      {
        text: "For negative inputs, ReLU outputs 0; for positive inputs, ReLU outputs the input unchanged.",
        isCorrect: true,
      },
      {
        text: "ReLU maps a single scalar input to a single scalar output (scalar in → scalar out).",
        isCorrect: true,
      },
      {
        text: "ReLU always outputs values strictly between 0 and 1.",
        isCorrect: false,
      },
    ],
    explanation:
      "ReLU is a simple piecewise function: it zeroes out negative values and passes positive values through unchanged. Unlike the sigmoid, its output is not constrained to \\((0,1)\\), which is one reason it is often preferred in hidden layers.",
  },

  {
    id: "mit15773-l1-q13",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Consider a feedforward network with input dimension \\(d\\), one hidden layer of \\(h\\) neurons, and one output neuron. All layers are fully connected and each neuron has a bias. Which formula gives the total number of parameters (weights + biases)?",
    options: [
      {
        text: "\\(d\\cdot h + h + h\\cdot 1 + 1\\)",
        isCorrect: true,
      },
      {
        text: "\\(d + h + 1\\)",
        isCorrect: false,
      },
      {
        text: "\\(d\\cdot h + h\\cdot 1\\) (biases are never counted as parameters)",
        isCorrect: false,
      },
      {
        text: "\\(d\\cdot h\\cdot 1\\) (multiply all layer sizes together)",
        isCorrect: false,
      },
    ],
    explanation:
      "From input to hidden you have \\(d\\cdot h\\) weights and \\(h\\) biases (one per hidden neuron). From hidden to output you have \\(h\\cdot 1\\) weights and \\(1\\) output bias. Adding them yields \\(d\\cdot h + h + h + 1\\).",
  },

  {
    id: "mit15773-l1-q14",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "In the lecture’s terminology, which statements correctly describe layers and connectivity in a standard feedforward neural network?",
    options: [
      {
        text: "A vertical stack of neurons is called a layer.",
        isCorrect: true,
      },
      {
        text: "The input 'layer' is often called a layer even though it does not apply a transformation by itself.",
        isCorrect: true,
      },
      {
        text: "If every neuron in one layer connects to every neuron in the next layer, the layer-to-layer connection is called fully connected (dense).",
        isCorrect: true,
      },
      {
        text: "A hidden layer must always use the sigmoid activation function to be valid.",
        isCorrect: false,
      },
    ],
    explanation:
      "A layer is a collection of neurons operating in parallel, and dense connectivity means all-to-all connections between consecutive layers. Hidden layers can use many activations (ReLU is a common default), while the output activation is chosen to match the output type.",
  },

  {
    id: "mit15773-l1-q15",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the role of the output activation function in a neural network?",
    options: [
      {
        text: "The output activation should match the nature of the prediction target (e.g., sigmoid for a single probability in \\((0,1)\\)).",
        isCorrect: true,
      },
      {
        text: "If the output must represent a probability distribution over multiple classes that sums to 1, a common choice is the softmax function.",
        isCorrect: true,
      },
      {
        text: "Choosing the output activation incorrectly can make the network’s outputs incompatible with the interpretation you want (e.g., probabilities).",
        isCorrect: true,
      },
      {
        text: "The output activation is arbitrary and has no practical effect as long as the hidden layers use ReLU.",
        isCorrect: false,
      },
    ],
    explanation:
      "Output activations encode constraints and interpretations. For binary probabilities, sigmoid is natural; for multi-class probabilities that must sum to 1, softmax is commonly used. If you pick an incompatible output activation, the numerical outputs won’t match the semantics you need.",
  },

  {
    id: "mit15773-l1-q16",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about the sigmoid activation function are correct?",
    options: [
      {
        text: "The sigmoid can be written as \\(\\sigma(a)=\\frac{1}{1+e^{-a}}\\).",
        isCorrect: true,
      },
      {
        text: "The sigmoid maps any real input \\(a\\in(-\\infty,\\infty)\\) to an output in \\((0,1)\\).",
        isCorrect: true,
      },
      {
        text: "Very negative inputs produce outputs close to 0, and very positive inputs produce outputs close to 1.",
        isCorrect: true,
      },
      {
        text: "The sigmoid outputs 0 for all negative inputs and outputs the input unchanged for positive inputs.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sigmoid smoothly squashes real numbers into \\((0,1)\\), which is why it is often used for binary probability outputs. The behavior is gradual: negative values trend toward 0 and positive values trend toward 1, unlike ReLU’s hard cutoff at 0.",
  },

  {
    id: "mit15773-l1-q17",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A neuron in the lecture is described as a linear function followed by an activation function. Which statements are correct about a typical neuron in a dense feedforward network?",
    options: [
      {
        text: "A neuron can receive many inputs (one from each connected upstream unit) and produces a single scalar output.",
        isCorrect: true,
      },
      {
        text: "A common form is \\(a = \\phi(b + \\sum_i w_i x_i)\\), where \\(\\phi\\) is an activation function, \\(w_i\\) are weights, and \\(b\\) is a bias.",
        isCorrect: true,
      },
      {
        text: "Even if a neuron outputs a single number, that number can be forwarded (copied) to multiple neurons in the next layer.",
        isCorrect: true,
      },
      {
        text: "A neuron must output a full probability distribution over classes; otherwise it is not a valid neuron.",
        isCorrect: false,
      },
    ],
    explanation:
      "A neuron is a scalar-to-scalar computation after aggregating many inputs: compute a weighted sum plus bias, then apply a nonlinearity. In dense layers, that scalar output is used as an input to many downstream neurons via multiple outgoing connections.",
  },

  {
    id: "mit15773-l1-q18",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "The lecture emphasizes that viewing logistic regression as a network makes an important point about neural networks. Which statements are correct?",
    options: [
      {
        text: "Logistic regression can be viewed as a neural network with no hidden layers: a linear function followed by a sigmoid output.",
        isCorrect: true,
      },
      {
        text: "Adding hidden layers can be seen as inserting additional transformations between the raw input and the final logistic regression-style output.",
        isCorrect: true,
      },
      {
        text: "The main freedom in designing a neural network is choosing what happens “in the middle” (hidden layers, number of units, activations).",
        isCorrect: true,
      },
      {
        text: "Adding hidden layers changes the training labels \\(y\\) into new labels automatically.",
        isCorrect: false,
      },
    ],
    explanation:
      "Logistic regression is a helpful baseline: it already looks like a tiny network. Neural networks generalize this by adding hidden layers that learn intermediate representations, while the target labels remain the same—the network is just learning a more powerful function from \\(x\\) to \\(y\\).",
  },

  {
    id: "mit15773-l1-q19",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "In the worked example network (2 inputs → 3 ReLU hidden units → 1 sigmoid output), the lecture counts 13 parameters. Which statements correctly explain that count?",
    options: [
      {
        text: "There are \\(2\\times 3=6\\) weights from the 2 inputs to the 3 hidden units in a fully connected layer.",
        isCorrect: true,
      },
      {
        text: "There are 3 hidden-unit biases (one per hidden neuron).",
        isCorrect: true,
      },
      {
        text: "There are \\(3\\times 1=3\\) weights from the 3 hidden units to the single output unit, plus 1 output bias.",
        isCorrect: true,
      },
      {
        text: "There are 13 parameters only if the hidden activation is sigmoid; with ReLU the count changes.",
        isCorrect: false,
      },
    ],
    explanation:
      "Parameter counts depend on connectivity and whether biases are included—not on which activation function you use. Fully connected input→hidden contributes 6 weights and 3 biases; hidden→output contributes 3 weights and 1 bias, totaling 13.",
  },

  {
    id: "mit15773-l1-q20",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements best define key high-level terminology used for neural networks in the lecture?",
    options: [
      {
        text: "A feedforward (vanilla) neural network is one where information flows from input to output without cycles (left to right).",
        isCorrect: true,
      },
      {
        text: "The architecture of a neural network refers to how neurons are arranged into layers, which activations are used, and how layers are connected.",
        isCorrect: true,
      },
      {
        text: "Hidden layers are the intermediate layers between input and output that perform learned transformations of the data.",
        isCorrect: true,
      },
      {
        text: "A recurrent neural network is the standard default for modern large language models and has largely replaced transformers.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture’s baseline is the feedforward network: data moves forward through layers to produce an output. Architecture means the blueprint (layers, units, activations, and connections). Modern large language models commonly use transformer architectures rather than recurrent neural networks.",
  },
  // Questions 21–40 (append to the existing array)

  {
    id: "mit15773-l1-q21",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly capture the idea behind Polanyi’s paradox (“we know more than we can tell”) and why it matters for rule-based AI?",
    options: [
      {
        text: "Humans can often perform tasks (e.g., dog vs cat recognition) quickly, but cannot reliably articulate the full decision procedure as explicit rules.",
        isCorrect: true,
      },
      {
        text: "Even when people offer explanations for their decisions, those explanations can be incomplete or not reflect the true internal process.",
        isCorrect: true,
      },
      {
        text: "If the expert cannot fully specify the rules, building a comprehensive IF–THEN system becomes difficult or impossible for many perceptual tasks.",
        isCorrect: true,
      },
      {
        text: "Polanyi’s paradox implies that collecting more data is unnecessary because rules are always sufficient.",
        isCorrect: false,
      },
    ],
    explanation:
      "Polanyi’s paradox highlights that expertise often cannot be fully translated into explicit IF–THEN rules, even if the expert performs the task well. That limitation makes rule-based AI brittle and incomplete for many real-world problems, especially perception-like tasks.",
  },

  {
    id: "mit15773-l1-q22",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements are correct about why raw pixel values are a challenging input representation for standard (non-deep) Machine Learning?",
    options: [
      {
        text: "A pixel intensity value (e.g., 251 in the blue channel) directly encodes a semantic concept such as “sky” or “water,” so it is immediately meaningful to a simple model.",
        isCorrect: false,
      },
      {
        text: "The same numeric pixel value can correspond to very different underlying causes (sky, paint, reflections), so the raw number is weakly tied to the semantic object.",
        isCorrect: true,
      },
      {
        text: "Raw images are typically stored as multiple grids of numbers (e.g., red/green/blue), which are easy to compute with but not automatically aligned to high-level concepts.",
        isCorrect: true,
      },
      {
        text: "Pixels are not numerical, so they cannot be used in any statistical model without converting them to letters first.",
        isCorrect: false,
      },
    ],
    explanation:
      "Pixels are numeric, but their meaning is not directly semantic: a large blue value does not uniquely imply a specific real-world object. This is why historically people engineered features (representations) and why deep learning’s representation learning is so impactful.",
  },

  {
    id: "mit15773-l1-q23",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Consider a dense layer with input \\(x\\in\\mathbb{R}^d\\), weights \\(W\\in\\mathbb{R}^{h\\times d}\\), bias \\(b\\in\\mathbb{R}^h\\), and ReLU activation. Which option correctly expresses the layer output \\(a\\in\\mathbb{R}^h\\)?",
    options: [
      {
        text: "\\(a = \\max(0, Wx + b)\\) (elementwise max).",
        isCorrect: true,
      },
      { text: "\\(a = W + x + b\\).", isCorrect: false },
      {
        text: "\\(a = \\sigma(Wx + b)\\) where \\(\\sigma\\) is the sigmoid.",
        isCorrect: false,
      },
      { text: "\\(a = \\max(0, x)W + b\\).", isCorrect: false },
    ],
    explanation:
      "A dense layer first computes the affine transform \\(z = Wx + b\\), producing an \\(h\\)-dimensional vector. ReLU is then applied elementwise: \\(a_i = \\max(0, z_i)\\), which keeps positive components and zeros out negative ones.",
  },

  {
    id: "mit15773-l1-q24",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "A model’s 'weights' are sometimes said to be 'leaked' online. Which statements are correct in the lecture’s framing?",
    options: [
      {
        text: "If you know a model’s learned parameters (weights/biases) and its architecture, you can often reconstruct the model’s behavior.",
        isCorrect: true,
      },
      {
        text: "In neural network terminology, regression coefficients are typically called weights and intercepts are called biases.",
        isCorrect: true,
      },
      {
        text: "Leaking weights has no impact because the model’s predictions come only from the training dataset, not from the parameters.",
        isCorrect: false,
      },
      {
        text: "Weights are only used in the output layer; hidden layers do not have weights.",
        isCorrect: false,
      },
    ],
    explanation:
      "A trained model’s behavior is determined by its architecture plus the learned parameters (weights and biases). If those parameters are exposed, someone can usually run the same network forward to reproduce the model outputs.",
  },

  {
    id: "mit15773-l1-q25",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the lecture’s point about modern systems learning mappings from 'anything to anything'?",
    options: [
      {
        text: "Inputs \\(X\\) can be multimodal (e.g., text + image), not restricted to a single data type.",
        isCorrect: true,
      },
      {
        text: "Outputs \\(Y\\) can be unstructured (e.g., images, audio, text), not just a small set of numbers.",
        isCorrect: true,
      },
      {
        text: "Modern models increasingly support arbitrary sequences of modalities (e.g., text, image, text, image) within one interaction.",
        isCorrect: true,
      },
      {
        text: "Such mappings are feasible in practice when enough data (and the right training setup) is available.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture emphasizes that both the input and output sides have expanded: models can consume and produce unstructured data. Multimodal systems generalize this further by mixing modalities across turns, enabling richer interactions than text-only pipelines.",
  },

  {
    id: "mit15773-l1-q26",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "In the lecture’s definition of a node (neuron) in a feedforward network, which statement is correct?",
    options: [
      {
        text: "A node outputs a single scalar value after applying an activation function to a weighted sum plus bias.",
        isCorrect: true,
      },
      {
        text: "A node must output an entire probability distribution over classes, otherwise it is not a node.",
        isCorrect: false,
      },
      {
        text: "A node outputs a full vector with the same dimensionality as the entire input layer.",
        isCorrect: false,
      },
      {
        text: "A node has no parameters; only layers have parameters.",
        isCorrect: false,
      },
    ],
    explanation:
      "A typical neuron aggregates many incoming signals via a weighted sum plus a bias, then applies a scalar activation to produce one scalar output. That scalar can be copied to multiple downstream neurons through outgoing connections.",
  },

  {
    id: "mit15773-l1-q27",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A fully connected (dense) layer maps an input vector of size \\(d\\) to an output vector of size \\(h\\). Which statements about dimensions are correct?",
    options: [
      {
        text: "A valid weight matrix shape is \\(W\\in\\mathbb{R}^{h\\times d}\\) so that \\(Wx\\) is in \\(\\mathbb{R}^h\\) when \\(x\\in\\mathbb{R}^d\\).",
        isCorrect: true,
      },
      {
        text: "A valid bias shape is \\(b\\in\\mathbb{R}^h\\) so it can be added to \\(Wx\\).",
        isCorrect: true,
      },
      {
        text: "A valid weight matrix shape is \\(W\\in\\mathbb{R}^{d\\times h}\\) so that \\(Wx\\) is in \\(\\mathbb{R}^h\\) when \\(x\\in\\mathbb{R}^d\\).",
        isCorrect: false,
      },
      {
        text: "The bias must have shape \\(b\\in\\mathbb{R}^d\\) to match the input size, otherwise addition is undefined.",
        isCorrect: false,
      },
    ],
    explanation:
      "To map \\(\\mathbb{R}^d\\rightarrow\\mathbb{R}^h\\), the affine transform is typically \\(z = Wx + b\\) with \\(W\\in\\mathbb{R}^{h\\times d}\\) and \\(b\\in\\mathbb{R}^h\\). This ensures dimensions align so the output is an \\(h\\)-dimensional vector.",
  },

  {
    id: "mit15773-l1-q28",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe what 'fully connected (dense)' means between two consecutive layers?",
    options: [
      {
        text: "Each neuron’s output in the earlier layer is connected (fed) to every neuron in the next layer.",
        isCorrect: true,
      },
      {
        text: "A single scalar output from a neuron can be forwarded to multiple neurons in the next layer through multiple connections.",
        isCorrect: true,
      },
      {
        text: "Dense connectivity is a common default for basic feedforward networks, especially for tabular (spreadsheet-like) inputs.",
        isCorrect: true,
      },
      {
        text: "Dense layers never use biases; they only use weights.",
        isCorrect: false,
      },
    ],
    explanation:
      "Dense connectivity means all-to-all connections between layers, which is why parameter counts grow as the product of layer sizes. In practice, dense layers typically include both weights and biases; biases are part of the standard affine transform before applying the activation.",
  },

  {
    id: "mit15773-l1-q29",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly reflect the lecture’s discussion of explainability/interpretability challenges in neural networks?",
    options: [
      {
        text: "For a fixed input, the forward computations are precisely defined, but attributing the prediction to a single original input feature can be difficult because information gets entangled across layers.",
        isCorrect: true,
      },
      {
        text: "Neural network predictions are unexplainable because we do not know which mathematical operations are executed during inference.",
        isCorrect: false,
      },
      {
        text: "Because activations are nonlinear, it is impossible to compute outputs exactly; we only estimate them statistically at inference time.",
        isCorrect: false,
      },
      {
        text: "Explainability is irrelevant for neural networks because they always outperform humans and therefore require no scrutiny.",
        isCorrect: false,
      },
    ],
    explanation:
      "The challenge is not that the computations are unknown—forward passes are deterministic given weights and inputs. The challenge is interpretability: once features are transformed and mixed through many layers, it’s hard to cleanly assign credit to one original input feature in a human-friendly way.",
  },

  {
    id: "mit15773-l1-q30",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about training labels, 'ground truth', and label noise are correct (as discussed in the lecture’s mammogram example)?",
    options: [
      {
        text: "In supervised learning, the training signal typically comes from pairs of inputs and labels (e.g., image + radiologist-provided label), which are treated as 'ground truth.'",
        isCorrect: true,
      },
      {
        text: "Labels can sometimes be wrong (noisy), and neural networks may still learn useful patterns despite some label noise.",
        isCorrect: true,
      },
      {
        text: "If even one training label is wrong, neural networks cannot learn anything useful and training must be abandoned.",
        isCorrect: false,
      },
      {
        text: "The lecture claims the solution to label noise is to avoid human labels and instead hard-code IF–THEN rules for every case.",
        isCorrect: false,
      },
    ],
    explanation:
      "Supervised learning relies on labeled examples that serve as the practical 'ground truth' for training, even though humans can be imperfect. The lecture notes that models can still perform well under some label noise, but high-quality, comprehensive data remains important for robust performance.",
  },

  {
    id: "mit15773-l1-q31",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "You build a dense network with 2 inputs, one hidden layer of 3 neurons, and 1 output neuron; every neuron has a bias. Which statements about parameter counting are correct?",
    options: [
      {
        text: "Input→hidden has \\(2\\times 3 = 6\\) weights.",
        isCorrect: true,
      },
      {
        text: "Hidden layer contributes 3 biases (one per hidden neuron).",
        isCorrect: true,
      },
      {
        text: "Hidden→output has \\(3\\times 1 = 3\\) weights.",
        isCorrect: true,
      },
      {
        text: "Output layer contributes 1 bias, giving a total of \\(6+3+3+1=13\\) parameters.",
        isCorrect: true,
      },
    ],
    explanation:
      "In dense layers, weights count as (number of incoming units) × (number of outgoing units). Biases add one parameter per neuron that performs an affine transform; including the output bias is a common place people forget, which is why this hand-check is useful.",
  },

  {
    id: "mit15773-l1-q32",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about activation functions (as used in the lecture) are correct?",
    options: [
      {
        text: "An activation function in this context takes a single scalar input and produces a single scalar output.",
        isCorrect: true,
      },
      {
        text: "The sigmoid activation maps any real number to a value strictly between 0 and 1.",
        isCorrect: true,
      },
      {
        text: "Activation functions are only applied in the input layer, not in hidden or output layers.",
        isCorrect: false,
      },
      {
        text: "ReLU is defined as \\(\\mathrm{ReLU}(a)=\\min(0,a)\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "In the lecture’s framing, the neuron outputs a scalar which is then passed through a scalar activation function. Sigmoid squashes values into \\((0,1)\\), while ReLU uses \\(\\max(0,a)\\), not \\(\\min\\).",
  },

  {
    id: "mit15773-l1-q33",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "When designing a neural network for a given task, which choices are typically under the practitioner’s control (as described in the lecture)?",
    options: [
      { text: "The number of hidden layers.", isCorrect: true },
      { text: "The number of neurons in each hidden layer.", isCorrect: true },
      {
        text: "The activation functions used in hidden layers (e.g., choosing ReLU as a default).",
        isCorrect: true,
      },
      {
        text: "Ensuring the output layer/activation matches the output type you need (e.g., probabilities vs coordinates).",
        isCorrect: true,
      },
    ],
    explanation:
      "The inputs and outputs are usually dictated by the problem, but the architecture in the middle is a design space you choose. The lecture emphasizes that the output activation is constrained by what you want the output to mean (e.g., a probability or a multi-class distribution).",
  },

  {
    id: "mit15773-l1-q34",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about architecture choices and 'searching' for architectures are correct?",
    options: [
      {
        text: "Neural Architecture Search (NAS) refers to methods that try many candidate architectures (automatically) to find one that works well for a task.",
        isCorrect: true,
      },
      {
        text: "Dropout is an example of a training-time technique that can randomly disable some hidden units to reduce overfitting, without changing the declared architecture.",
        isCorrect: true,
      },
      {
        text: "NAS means the network changes its number of layers on every single forward pass during training (e.g., 2 layers on one example, 7 layers on the next) as the standard approach.",
        isCorrect: false,
      },
      {
        text: "The lecture claims there is one universally optimal architecture that works best for all tasks and datasets.",
        isCorrect: false,
      },
    ],
    explanation:
      "NAS is about automating the exploration of architectural designs, often by evaluating many candidates under some optimization strategy. Dropout is different: it’s a regularization method applied during training that stochastically drops units, but you still keep a fixed underlying architecture definition.",
  },

  {
    id: "mit15773-l1-q35",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly relate logistic regression to neural networks in the lecture’s view?",
    options: [
      {
        text: "A binary logistic regression model can be seen as a network that computes \\(z=b+\\sum_i w_i x_i\\) and then applies \\(\\sigma(z)=\\frac{1}{1+e^{-z}}\\).",
        isCorrect: true,
      },
      {
        text: "Under this lens, logistic regression is a neural network with zero hidden layers.",
        isCorrect: true,
      },
      {
        text: "Logistic regression is only possible if the hidden layers use ReLU; otherwise it cannot represent a probability.",
        isCorrect: false,
      },
      {
        text: "Adding hidden layers changes the definition of the sigmoid function itself.",
        isCorrect: false,
      },
    ],
    explanation:
      "Logistic regression is structurally a linear transform followed by a sigmoid, which fits neatly into the 'network of operations' view. Neural networks generalize it by inserting learned nonlinear transformations (hidden layers) before an output layer that may still be logistic-regression-like for binary classification.",
  },

  {
    id: "mit15773-l1-q36",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements are correct about why Deep Learning dramatically expanded what ML could do with unstructured data?",
    options: [
      {
        text: "It can learn intermediate representations automatically from raw inputs instead of relying on manual feature engineering.",
        isCorrect: true,
      },
      {
        text: "It made it practical to attach perception capabilities (detect/recognize/classify) behind sensors that produce unstructured signals.",
        isCorrect: true,
      },
      {
        text: "It benefited from large datasets and parallel compute, making large-scale training feasible for complex models.",
        isCorrect: true,
      },
      {
        text: "It primarily works by asking experts to write more complete IF–THEN rules than before.",
        isCorrect: false,
      },
    ],
    explanation:
      "The key shift is representation learning: deep networks can transform raw signals into useful features internally. This enables practical perception systems behind sensors and scales well with data and compute, unlike rule-writing approaches that struggle with generalization and coverage.",
  },

  {
    id: "mit15773-l1-q37",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statement best captures the lecture’s intuition about where the 'action' happens for the sigmoid activation function?",
    options: [
      {
        text: "Sigmoid outputs change most rapidly for inputs near 0 (the middle region), while very negative or very positive inputs produce outputs close to 0 or 1 with little change.",
        isCorrect: true,
      },
      {
        text: "Sigmoid outputs always increase linearly with the input at a constant slope.",
        isCorrect: false,
      },
      {
        text: "Sigmoid outputs are exactly 0 for all negative inputs and exactly 1 for all positive inputs.",
        isCorrect: false,
      },
      {
        text: "Sigmoid produces outputs larger than 1 for sufficiently large positive inputs.",
        isCorrect: false,
      },
    ],
    explanation:
      "The sigmoid saturates: far in the negative region it is near 0, and far in the positive region it is near 1. The most sensitive region is around the center where small changes in input lead to the largest changes in output.",
  },

  {
    id: "mit15773-l1-q38",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly compare feedforward networks, recurrent neural networks (RNNs), and transformers as described in the lecture?",
    options: [
      {
        text: "A feedforward network is one where information flows from input to output without cycles (no recurrence).",
        isCorrect: true,
      },
      {
        text: "Recurrent neural networks introduce cyclic connections to process sequences, but the lecture notes transformers have become the dominant approach for many sequence tasks.",
        isCorrect: true,
      },
      {
        text: "Transformers are mentioned as largely replaced by recurrent networks because recurrence is inherently more expressive for language.",
        isCorrect: false,
      },
      {
        text: "Feedforward networks cannot be trained with gradient-based optimization methods, which is why RNNs were invented.",
        isCorrect: false,
      },
    ],
    explanation:
      "Feedforward networks have acyclic computation graphs, while RNNs include recurrence for sequential data. The lecture’s framing is that transformers have proven highly capable and are now the norm in many settings where RNNs were once standard.",
  },

  {
    id: "mit15773-l1-q39",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the 'deep learning pipeline' idea emphasized in the lecture (raw input → learned representations → simple predictor)?",
    options: [
      {
        text: "Intermediate layers can learn representations that make the final prediction step easier, often allowing a simple linear/logistic head at the end.",
        isCorrect: true,
      },
      {
        text: "Learning multiple successive transformations gives the network more capacity to discover useful structure in the data than applying only one transformation.",
        isCorrect: true,
      },
      {
        text: "The input and output are fixed by the problem; the network’s flexibility largely comes from what it learns in the hidden layers.",
        isCorrect: true,
      },
      {
        text: "Deep learning works by keeping representations fixed and only learning the last layer, which is why it avoids overfitting entirely.",
        isCorrect: false,
      },
    ],
    explanation:
      "Deep learning is powerful because hidden layers can learn representations automatically, and stacking transformations increases the model’s capacity to fit complex relationships. However, deep learning does not keep representations fixed by default—representations are learned—and overfitting is still a real concern, which is why regularization techniques matter.",
  },

  {
    id: "mit15773-l1-q40",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly reflect the lecture’s 'sensor + deep learning' framing for practical applications?",
    options: [
      {
        text: "A camera combined with a deep learning model can enable tasks like face recognition (e.g., Face ID-style unlocking).",
        isCorrect: true,
      },
      {
        text: "Deep learning can support medical imaging detection systems by learning patterns from labeled images (e.g., mammograms).",
        isCorrect: true,
      },
      {
        text: "In manufacturing, camera-based deep learning systems can be used for visual inspection tasks like scratch or dent detection.",
        isCorrect: true,
      },
      {
        text: "Because sensors produce unstructured data, deep learning is a natural fit for enabling automated recognition and classification behind sensors.",
        isCorrect: true,
      },
    ],
    explanation:
      "The core idea is that sensors are sources of unstructured signals, and deep learning can convert those signals into structured decisions such as detections and classifications. This pattern shows up across consumer devices, healthcare imaging, and industrial inspection workflows.",
  },
];
