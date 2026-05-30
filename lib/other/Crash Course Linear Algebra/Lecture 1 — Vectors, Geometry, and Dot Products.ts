import { Question } from "../../quiz";

export const CrashCourseLinearAlgebraLecture1Questions: Question[] = [
  {
    id: "la-crash-l1-q01",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe a vector \\(x \\in \\mathbb{R}^n\\)?",
    options: [
      {
        text: "A vector can represent a list of numerical features such as age, height, and income.",
        isCorrect: true,
      },
      {
        text: "A vector can be interpreted as a point in an n-dimensional geometric space.",
        isCorrect: true,
      },
      {
        text: "A vector can represent a direction starting at the origin.",
        isCorrect: true,
      },
      {
        text: "In machine learning, vectors are commonly used to represent inputs such as embeddings or states.",
        isCorrect: true,
      },
    ],
    explanation:
      "A vector is fundamentally a list of numbers that can be interpreted geometrically or as features. In machine learning, vectors frequently represent data points, embeddings, or states in reinforcement learning. Thinking of vectors as points and directions in space helps build geometric intuition for operations like dot products and norms.",
  },

  {
    id: "la-crash-l1-q02",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements about row and column vectors are correct?",
    options: [
      {
        text: "A column vector \\(x\\) can be written as \\(x = [x_1, x_2, ..., x_n]^T\\).",
        isCorrect: true,
      },
      {
        text: "A row vector is typically written as \\([x_1 \\; x_2 \\; ... \\; x_n]\\).",
        isCorrect: true,
      },
      {
        text: "Taking the transpose of a column vector produces a row vector.",
        isCorrect: true,
      },
      {
        text: "Row vectors are mathematically valid objects in linear algebra.",
        isCorrect: true,
      },
    ],
    explanation:
      "Column vectors are commonly used in machine learning notation and can be transposed to become row vectors. The transpose operation flips the orientation of the vector. Row vectors are perfectly valid in linear algebra and are often used in matrix multiplication expressions such as \\(x^T y\\).",
  },

  {
    id: "la-crash-l1-q03",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Consider the vector \\(x = [3,4]^T\\). Which statements are correct?",
    options: [
      {
        text: "The Euclidean (L2) norm of the vector equals 5.",
        isCorrect: true,
      },
      {
        text: "The vector lies at the point (3,4) in a 2-dimensional coordinate system.",
        isCorrect: true,
      },
      {
        text: "The vector has magnitude \\(\\sqrt{3^2 + 4^2}\\).",
        isCorrect: true,
      },
      {
        text: "The vector does not have to represent a probability distribution.",
        isCorrect: true,
      },
    ],
    explanation:
      "The L2 norm is computed as \\(\\sqrt{3^2 + 4^2} = 5\\), which comes directly from the Pythagorean theorem. Vectors can represent many different kinds of quantities, but they do not automatically represent probability distributions unless their elements satisfy probability constraints such as summing to 1 and being non-negative.",
  },

  {
    id: "la-crash-l1-q04",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which of the following statements correctly describe the dot product \\(a \\cdot b\\)?",
    options: [
      { text: "It can be computed as \\(a^T b\\).", isCorrect: true },
      { text: "It is equal to \\(\\sum_i a_i b_i\\).", isCorrect: true },
      {
        text: "It measures the geometric alignment between two vectors.",
        isCorrect: true,
      },
      {
        text: "It is not identical to multiplying two matrices element-wise.",
        isCorrect: true,
      },
    ],
    explanation:
      "The dot product is defined as the sum of element-wise products of vector components and can be written compactly as \\(a^T b\\). Geometrically, it measures how aligned two vectors are. It is different from element-wise multiplication, which multiplies corresponding components but does not sum them.",
  },

  {
    id: "la-crash-l1-q05",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Compute the dot product of \\(a=[1,2]^T\\) and \\(b=[3,4]^T\\). Which statements are correct?",
    options: [
      { text: "The dot product equals 11.", isCorrect: true },
      { text: "The dot product equals \\(1*3 + 2*4\\).", isCorrect: true },
      {
        text: "The dot product equals \\(\\|a\\| \\|b\\| \\cos \\theta\\).",
        isCorrect: true,
      },
      {
        text: "The dot product does not have to be positive.",
        isCorrect: true,
      },
    ],
    explanation:
      "The algebraic computation gives \\(1*3 + 2*4 = 11\\). The geometric interpretation expresses the dot product as \\(\\|a\\|\\|b\\|\\cos\\theta\\). However, the dot product can be negative if the vectors point in opposite directions.",
  },

  {
    id: "la-crash-l1-q06",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about cosine similarity are correct?",
    options: [
      {
        text: "Cosine similarity is defined as \\(\\frac{a \\cdot b}{\\|a\\|\\|b\\|}\\).",
        isCorrect: true,
      },
      {
        text: "Cosine similarity measures the angle between vectors rather than their magnitudes.",
        isCorrect: true,
      },
      {
        text: "Two identical vectors have cosine similarity equal to 1.",
        isCorrect: true,
      },
      {
        text: "Cosine similarity does not require the vectors to have exactly the same length.",
        isCorrect: true,
      },
    ],
    explanation:
      "Cosine similarity compares vectors based on their orientation rather than their magnitude. It equals the cosine of the angle between the vectors, which ranges from −1 to 1. Vectors do not need to have equal length to compute cosine similarity because the normalization divides by their magnitudes.",
  },

  {
    id: "la-crash-l1-q07",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements are true about orthogonal vectors?",
    options: [
      { text: "Their dot product equals zero.", isCorrect: true },
      { text: "They are perpendicular in geometric space.", isCorrect: true },
      { text: "Their cosine similarity equals zero.", isCorrect: true },
      { text: "They do not need to have equal magnitudes.", isCorrect: true },
    ],
    explanation:
      "Orthogonal vectors form a right angle and therefore have cosine of the angle equal to zero. Because the dot product equals \\(\\|a\\|\\|b\\|\\cos\\theta\\), it becomes zero when the cosine is zero. Their magnitudes can differ arbitrarily.",
  },

  {
    id: "la-crash-l1-q08",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Consider the expression \\(y = W x\\). Which statements are correct?",
    options: [
      { text: "The result is a vector.", isCorrect: true },
      {
        text: "Each element of \\(y\\) is the dot product between a row of \\(W\\) and \\(x\\).",
        isCorrect: true,
      },
      {
        text: "The operation can represent a neural network layer.",
        isCorrect: true,
      },
      {
        text: "The operation does not require \\(W\\) to be a square matrix.",
        isCorrect: true,
      },
    ],
    explanation:
      "In matrix–vector multiplication, each output component corresponds to a dot product between a row of the matrix and the input vector. This operation is exactly what a linear layer in a neural network computes. The matrix does not need to be square; its shape determines the output dimension.",
  },

  {
    id: "la-crash-l1-q09",
    chapter: 1,
    difficulty: "medium",
    prompt: "Suppose \\(x \\in \\mathbb{R}^5\\). Which statements are correct?",
    options: [
      { text: "The vector has five components.", isCorrect: true },
      {
        text: "The vector can be interpreted as a point in five-dimensional space.",
        isCorrect: true,
      },
      {
        text: "The vector could represent five features of a data point.",
        isCorrect: true,
      },
      {
        text: "The vector does not have to represent a physical geometric object.",
        isCorrect: true,
      },
    ],
    explanation:
      "A vector in \\(\\mathbb{R}^5\\) contains five numerical components and can be interpreted as a point in five-dimensional space. In machine learning, vectors often represent feature sets rather than physical geometric objects. The geometric interpretation is useful for intuition but does not require a physical meaning.",
  },

  {
    id: "la-crash-l1-q10",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe Euclidean distance between two vectors \\(x\\) and \\(y\\)?",
    options: [
      {
        text: "The distance is given by the dot product \\(x^T y\\).",
        isCorrect: false,
      },
      {
        text: "It measures the straight-line distance between two points in space.",
        isCorrect: true,
      },
      { text: "It is derived from the Pythagorean theorem.", isCorrect: true },
      {
        text: "It is identical to the dot product of the vectors.",
        isCorrect: false,
      },
    ],
    explanation:
      "Euclidean distance measures how far apart two points are in space and is computed as the norm of their difference. It directly follows from the Pythagorean theorem. The dot product measures alignment rather than distance.",
  },

  {
    id: "la-crash-l1-q11",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Let \\(a=[1,2,3]^T\\) and \\(b=[2,1,0]^T\\). Which statements are correct?",
    options: [
      { text: "The dot product equals 0.", isCorrect: false },
      {
        text: "The dot product equals \\(1*2 + 2*1 + 3*0\\).",
        isCorrect: true,
      },
      {
        text: "The dot product equals \\(\\|a\\|\\|b\\|\\cos\\theta\\).",
        isCorrect: true,
      },
      { text: "The vectors are orthogonal.", isCorrect: false },
    ],
    explanation:
      "The dot product calculation gives \\(1*2 + 2*1 + 3*0 = 4\\). The geometric relationship still holds: the dot product equals \\(\\|a\\|\\|b\\|\\cos\\theta\\). Because the dot product is not zero, the vectors are not orthogonal.",
  },

  {
    id: "la-crash-l1-q12",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements describe how dot products are used in neural networks?",
    options: [
      {
        text: "Each neuron computes an unweighted sum of its inputs.",
        isCorrect: false,
      },
      {
        text: "A neuron output often involves \\(w^T x + b\\).",
        isCorrect: true,
      },
      {
        text: "Weights determine which input directions the neuron responds strongly to.",
        isCorrect: true,
      },
      {
        text: "Neural networks rely exclusively on element-wise multiplication instead of dot products.",
        isCorrect: false,
      },
    ],
    explanation:
      "A neuron computes a dot product between weights and inputs and then typically adds a bias term. The weights define which input patterns activate the neuron strongly. Neural networks depend heavily on dot products rather than purely element-wise operations.",
  },

  {
    id: "la-crash-l1-q13",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe cosine similarity in embedding spaces?",
    options: [
      {
        text: "Cosine similarity focuses on magnitude only and ignores direction.",
        isCorrect: false,
      },
      {
        text: "Vectors representing similar concepts often have high cosine similarity.",
        isCorrect: true,
      },
      {
        text: "Cosine similarity is commonly used in semantic search and retrieval.",
        isCorrect: true,
      },
      {
        text: "Cosine similarity requires vectors to be orthogonal.",
        isCorrect: false,
      },
    ],
    explanation:
      "In embedding spaces, cosine similarity measures the angle between vectors and ignores scale differences. Similar concepts often appear in similar directions in embedding space. This property is why cosine similarity is widely used for retrieval tasks and semantic search.",
  },

  {
    id: "la-crash-l1-q14",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements are correct about the L2 norm \\(\\|x\\|_2\\)?",
    options: [
      { text: "It equals \\(\\sum_i x_i\\).", isCorrect: false },
      {
        text: "It represents the length of the vector in Euclidean space.",
        isCorrect: true,
      },
      {
        text: "It increases when the vector components grow in magnitude.",
        isCorrect: true,
      },
      {
        text: "It is defined only for two-dimensional vectors.",
        isCorrect: false,
      },
    ],
    explanation:
      "The L2 norm generalizes the concept of length to any dimension. It is computed as the square root of the sum of squared components. This norm exists for vectors of any dimension, not just two-dimensional vectors.",
  },

  {
    id: "la-crash-l1-q15",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about high-dimensional vectors in machine learning are correct?",
    options: [
      {
        text: "Word embeddings always contain exactly three dimensions.",
        isCorrect: false,
      },
      {
        text: "High-dimensional vectors can still be analyzed using the same geometric concepts as 2-D vectors.",
        isCorrect: true,
      },
      {
        text: "Dot products and norms extend naturally to high dimensions.",
        isCorrect: true,
      },
      {
        text: "Vectors with more than three dimensions cannot be used in machine learning models.",
        isCorrect: false,
      },
    ],
    explanation:
      "Linear algebra operations such as dot products and norms generalize naturally to high-dimensional spaces. Many machine learning models use very high dimensional vectors, including word embeddings and hidden representations in neural networks. The geometry remains conceptually the same even though visualization becomes difficult.",
  },

  {
    id: "la-crash-l1-q16",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about matrix–vector multiplication \\(y = Wx\\) are correct?",
    options: [
      {
        text: "Each column of \\(W\\) defines the output dimension regardless of input shape.",
        isCorrect: false,
      },
      {
        text: "The output dimension equals the number of rows in \\(W\\).",
        isCorrect: true,
      },
      {
        text: "The operation computes several dot products simultaneously.",
        isCorrect: true,
      },
      {
        text: "The input vector must contain exactly the same number of elements as rows in \\(W\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "In matrix–vector multiplication, each row of the matrix produces one output component by computing a dot product with the input vector. Therefore the number of rows determines the output dimension. The input dimension must match the number of columns, not the number of rows.",
  },

  {
    id: "la-crash-l1-q17",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about the geometric meaning of the dot product are correct?",
    options: [
      {
        text: "It equals the Euclidean distance \\(\\|a-b\\|\\).",
        isCorrect: false,
      },
      {
        text: "It measures how strongly one vector points in the direction of another.",
        isCorrect: true,
      },
      {
        text: "It can be interpreted as the projection of one vector onto another.",
        isCorrect: true,
      },
      {
        text: "It equals the Euclidean distance between two vectors.",
        isCorrect: false,
      },
    ],
    explanation:
      "The dot product captures the alignment between vectors and can be interpreted as the projection of one vector onto another. This geometric perspective explains why dot products are useful for measuring similarity. Euclidean distance measures separation rather than alignment.",
  },

  {
    id: "la-crash-l1-q18",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose \\(\\|a\\|=2\\), \\(\\|b\\|=3\\), and the angle between them is \\(60^\\circ\\). Which statements are correct?",
    options: [
      { text: "The dot product equals 3.", isCorrect: true },
      {
        text: "The formula \\(a\\cdot b = \\|a\\|\\|b\\|\\cos\\theta\\) cannot be used here.",
        isCorrect: false,
      },
      {
        text: "Because \\(\\cos 60^\\circ = 1\\), the dot product equals \\(2*3\\).",
        isCorrect: false,
      },
      { text: "The vectors are orthogonal.", isCorrect: false },
    ],
    explanation:
      "The geometric dot product formula gives \\(2 * 3 * 0.5 = 3\\). Orthogonal vectors would require the angle to be 90 degrees and the cosine to equal zero. Therefore these vectors are not orthogonal.",
  },

  {
    id: "la-crash-l1-q19",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about vectors in machine learning applications are correct?",
    options: [
      {
        text: "Images can be represented as vectors of pixel values.",
        isCorrect: true,
      },
      {
        text: "Reinforcement learning states cannot be encoded as vectors.",
        isCorrect: false,
      },
      {
        text: "Word embeddings are scalar values rather than vectors.",
        isCorrect: false,
      },
      {
        text: "Vectors cannot represent categorical information.",
        isCorrect: false,
      },
    ],
    explanation:
      "Vectors are a universal representation format in machine learning. Images, states, and embeddings are all encoded as vectors. Even categorical variables can be represented using vector encodings such as one-hot vectors or embeddings.",
  },

  {
    id: "la-crash-l1-q20",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe why dot products appear frequently in machine learning models?",
    options: [
      {
        text: "Dot products measure similarity between vectors.",
        isCorrect: true,
      },
      {
        text: "Neural network layers never compute dot products in parallel.",
        isCorrect: false,
      },
      {
        text: "Attention mechanisms do not rely on dot products between queries and keys.",
        isCorrect: false,
      },
      {
        text: "Dot products can only be used in two-dimensional spaces.",
        isCorrect: false,
      },
    ],
    explanation:
      "Dot products are fundamental because they measure alignment between vectors, which corresponds to similarity in many machine learning contexts. Neural networks compute weighted sums (dot products) between inputs and weights. Attention mechanisms in transformers also use dot products between query and key vectors to determine relevance.",
  },

  {
    id: "la-crash-l1-q21",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Consider the vectors \\(a=[2,0]^T\\) and \\(b=[0,3]^T\\). Which statements are correct?",
    options: [
      { text: "The dot product \\(a \\cdot b = 0\\).", isCorrect: true },
      { text: "The vectors are parallel.", isCorrect: false },
      {
        text: "The angle between the vectors is \\(0^\\circ\\).",
        isCorrect: false,
      },
      {
        text: "The cosine similarity between them equals 1.",
        isCorrect: false,
      },
    ],
    explanation:
      "The dot product is computed as \\(2*0 + 0*3 = 0\\). When the dot product is zero, vectors are orthogonal, meaning they form a right angle (90°). Cosine similarity would also equal 0 in this case, not 1.",
  },

  {
    id: "la-crash-l1-q22",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the relationship between dot product and vector length?",
    options: [
      {
        text: "If \\(a \\cdot b > 0\\), the angle between vectors is less than 90 degrees.",
        isCorrect: true,
      },
      {
        text: "If \\(a \\cdot b < 0\\), the vectors point in roughly the same direction.",
        isCorrect: false,
      },
      {
        text: "If \\(a \\cdot b = 0\\), the vectors are identical.",
        isCorrect: false,
      },
      {
        text: "The dot product ignores the magnitudes of the vectors.",
        isCorrect: false,
      },
    ],
    explanation:
      "The dot product includes both magnitude and directional information through the formula \\(a \\cdot b = \\|a\\|\\|b\\|\\cos\\theta\\). A positive dot product indicates an acute angle, while a negative value indicates vectors pointing in opposite directions. A zero value corresponds to orthogonal vectors.",
  },

  {
    id: "la-crash-l1-q23",
    chapter: 1,
    difficulty: "easy",
    prompt: "Let \\(x=[1,1]^T\\). Which statements are correct?",
    options: [
      { text: "The L2 norm equals \\(\\sqrt{2}\\).", isCorrect: true },
      { text: "The squared norm equals \\(\\sqrt{2}\\).", isCorrect: false },
      { text: "The vector length is less than 1.", isCorrect: false },
      { text: "The norm equals the sum of the elements.", isCorrect: false },
    ],
    explanation:
      'The L2 norm is \\(\\sqrt{1^2 + 1^2} = \\sqrt{2}\\). The squared norm equals 2. Norms measure geometric length and are not simply the sum of components. To reason through the choices, select the statements that match the criterion in the prompt: "The L2 norm equals \\(\\sqrt{2}\\).". Do not select statements that miss that criterion: "The squared norm equals \\(\\sqrt{2}\\)."; "The vector length is less than 1."; "The norm equals the sum of the elements.". This contrast makes the conceptual boundary explicit instead of relying on familiar-sounding wording.',
  },

  {
    id: "la-crash-l1-q24",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Suppose \\(a=[1,2]^T\\) and \\(b=[2,4]^T\\). Which statements are correct?",
    options: [
      { text: "The vectors point in the same direction.", isCorrect: true },
      { text: "The cosine similarity equals 0.", isCorrect: false },
      { text: "The vectors are linearly independent.", isCorrect: false },
      { text: "The vectors are orthogonal.", isCorrect: false },
    ],
    explanation:
      "The vector \\(b\\) is a scalar multiple of \\(a\\), so they point in the same direction and have cosine similarity equal to 1. This also means they are linearly dependent. Orthogonal vectors would require their dot product to equal zero.",
  },

  {
    id: "la-crash-l1-q25",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Consider a neural network neuron computing \\(y = w^T x + b\\). Which statements are correct?",
    options: [
      { text: "The term \\(w^T x\\) is a dot product.", isCorrect: true },
      {
        text: "The weights \\(w\\) do not determine which directions in input space activate the neuron.",
        isCorrect: false,
      },
      {
        text: "The bias \\(b\\) is determined entirely by the input vector.",
        isCorrect: false,
      },
      {
        text: "The neuron output depends only on the length of the input vector.",
        isCorrect: false,
      },
    ],
    explanation:
      "A neuron computes the dot product between its weight vector and input vector and then adds a bias term. The weight vector determines which directions in the input space produce stronger responses. The output depends on alignment with weights rather than only the vector magnitude.",
  },

  {
    id: "la-crash-l1-q26",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about Euclidean distance \\(\\|x-y\\|\\) are correct?",
    options: [
      {
        text: "It measures the straight-line distance between two vectors.",
        isCorrect: true,
      },
      { text: "It equals \\(\\sqrt{\\sum_i (x_i-y_i)^2}\\).", isCorrect: true },
      {
        text: "It depends on both direction and magnitude differences.",
        isCorrect: true,
      },
      { text: "It is identical to cosine similarity.", isCorrect: false },
    ],
    explanation:
      "Euclidean distance is derived from the Pythagorean theorem and measures the straight-line separation between two points in space. Unlike cosine similarity, which measures orientation, Euclidean distance incorporates magnitude differences as well.",
  },

  {
    id: "la-crash-l1-q27",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose \\(x \\in \\mathbb{R}^4\\) and \\(W \\in \\mathbb{R}^{3 \\times 4}\\). Which statements are correct about \\(y = Wx\\)?",
    options: [
      { text: "The output vector \\(y\\) has dimension 3.", isCorrect: true },
      {
        text: "Each component of \\(y\\) is a dot product between a row of \\(W\\) and \\(x\\).",
        isCorrect: true,
      },
      {
        text: "The matrix computes three linear detectors of the input vector.",
        isCorrect: true,
      },
      {
        text: "The multiplication is invalid because the matrix is not square.",
        isCorrect: false,
      },
    ],
    explanation:
      "The output dimension of a matrix–vector multiplication equals the number of rows in the matrix. Each row defines a linear detector that computes a dot product with the input vector. The matrix does not need to be square for the multiplication to be valid.",
  },

  {
    id: "la-crash-l1-q28",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements correctly describe the L1 norm \\(\\|x\\|_1\\)?",
    options: [
      {
        text: "It equals the sum of absolute values of vector components.",
        isCorrect: true,
      },
      { text: "It is defined as \\(\\sum_i |x_i|\\).", isCorrect: true },
      {
        text: "It measures distance using a different geometry than L2.",
        isCorrect: true,
      },
      { text: "It is always identical to the L2 norm.", isCorrect: false },
    ],
    explanation:
      "The L1 norm sums the absolute values of vector components. It represents a different geometric distance than L2 and is often associated with sparsity in optimization. The L1 and L2 norms generally produce different values.",
  },

  {
    id: "la-crash-l1-q29",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Suppose \\(a=[1,0,0]^T\\) and \\(b=[0,1,0]^T\\). Which statements are correct?",
    options: [
      { text: "The dot product equals 0.", isCorrect: true },
      { text: "The vectors are orthogonal.", isCorrect: true },
      { text: "Cosine similarity equals 0.", isCorrect: true },
      { text: "The vectors must have equal length.", isCorrect: false },
    ],
    explanation:
      "The vectors share no overlapping dimensions with nonzero values, so their dot product is zero. This means they are orthogonal and have cosine similarity equal to zero. Orthogonality does not require equal magnitude.",
  },

  {
    id: "la-crash-l1-q30",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe cosine similarity values?",
    options: [
      { text: "Cosine similarity ranges from -1 to 1.", isCorrect: true },
      {
        text: "Vectors pointing in identical directions have cosine similarity 1.",
        isCorrect: true,
      },
      {
        text: "Vectors pointing in opposite directions have cosine similarity -1.",
        isCorrect: true,
      },
      {
        text: "Cosine similarity is undefined for vectors with zero magnitude.",
        isCorrect: true,
      },
    ],
    explanation:
      "Cosine similarity measures the cosine of the angle between vectors and ranges from −1 to 1. A zero vector has no direction and therefore cannot be normalized, which makes cosine similarity undefined for such vectors.",
  },

  {
    id: "la-crash-l1-q31",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements explain why cosine similarity is often used in embedding spaces?",
    options: [
      {
        text: "It compares vectors based on direction rather than magnitude.",
        isCorrect: true,
      },
      {
        text: "It reduces sensitivity to differences in vector scale.",
        isCorrect: true,
      },
      {
        text: "It often captures semantic similarity in embedding spaces.",
        isCorrect: true,
      },
      {
        text: "It requires vectors to have identical magnitudes.",
        isCorrect: false,
      },
    ],
    explanation:
      "Cosine similarity focuses on orientation and ignores scale, which makes it useful when magnitude differences are not meaningful. In embedding spaces, semantically related concepts often appear in similar directions, which cosine similarity captures well.",
  },

  {
    id: "la-crash-l1-q32",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements are true about vectors used in machine learning models?",
    options: [
      {
        text: "Feature vectors represent attributes of data points.",
        isCorrect: true,
      },
      {
        text: "Embeddings represent semantic information as vectors.",
        isCorrect: true,
      },
      {
        text: "States in reinforcement learning can be represented as vectors.",
        isCorrect: true,
      },
      { text: "Vectors cannot represent text data.", isCorrect: false },
    ],
    explanation:
      "In machine learning, vectors represent structured information such as features, embeddings, or state representations. Even text is converted into vector representations through token embeddings. Vectors are a universal representation format across many ML models.",
  },

  {
    id: "la-crash-l1-q33",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose \\(a=[2,1]^T\\) and \\(b=[1,3]^T\\). Which statements are correct?",
    options: [
      { text: "The dot product equals 5.", isCorrect: true },
      { text: "The dot product equals \\(2*1 + 1*3\\).", isCorrect: true },
      { text: "The vectors are not orthogonal.", isCorrect: true },
      { text: "The cosine similarity must equal 1.", isCorrect: false },
    ],
    explanation:
      "The dot product calculation gives \\(2*1 + 1*3 = 5\\). Since the dot product is nonzero, the vectors are not orthogonal. Cosine similarity would equal 1 only if the vectors pointed in exactly the same direction.",
  },

  {
    id: "la-crash-l1-q34",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements describe geometric interpretations of vectors?",
    options: [
      { text: "A vector can represent a point in space.", isCorrect: true },
      {
        text: "A vector can represent a direction and magnitude.",
        isCorrect: true,
      },
      {
        text: "Vectors can represent displacements between points.",
        isCorrect: true,
      },
      {
        text: "Vectors only exist in two-dimensional space.",
        isCorrect: false,
      },
    ],
    explanation:
      "Vectors can represent points, directions, or displacements in any dimensional space. The geometric interpretation is fundamental in understanding operations such as norms and dot products. These concepts generalize beyond two dimensions.",
  },

  {
    id: "la-crash-l1-q35",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe matrix–vector multiplication in neural networks?",
    options: [
      { text: "It produces a new vector of outputs.", isCorrect: true },
      {
        text: "Each output corresponds to a weighted combination of input features.",
        isCorrect: true,
      },
      {
        text: "It is used in linear layers of neural networks.",
        isCorrect: true,
      },
      {
        text: "It only works when the matrix has the same number of rows and columns.",
        isCorrect: false,
      },
    ],
    explanation:
      "Matrix–vector multiplication combines input features through weighted sums to produce outputs. This is exactly how linear layers operate in neural networks. The matrix does not need to be square; its shape determines the output dimension.",
  },

  {
    id: "la-crash-l1-q36",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the role of vectors in transformer attention mechanisms?",
    options: [
      {
        text: "Queries, keys, and values are represented as vectors.",
        isCorrect: true,
      },
      {
        text: "Attention scores are computed using dot products between queries and keys.",
        isCorrect: true,
      },
      {
        text: "The weighted combination of value vectors produces attention outputs.",
        isCorrect: true,
      },
      {
        text: "Attention mechanisms require vectors to be orthogonal.",
        isCorrect: false,
      },
    ],
    explanation:
      "In transformers, tokens are mapped to vectors called queries, keys, and values. Attention scores are computed via dot products between queries and keys. The resulting weights are applied to value vectors to compute the final attention output.",
  },

  {
    id: "la-crash-l1-q37",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about norms and distances are correct?",
    options: [
      { text: "Norms measure vector magnitude.", isCorrect: true },
      {
        text: "Distances measure separation between vectors.",
        isCorrect: true,
      },
      {
        text: "The L2 norm is derived from Euclidean geometry.",
        isCorrect: true,
      },
      {
        text: "Norms and dot products are completely unrelated concepts.",
        isCorrect: false,
      },
    ],
    explanation:
      "Norms measure vector magnitude, while distances measure separation between vectors. The L2 norm arises from Euclidean geometry and is closely connected to the dot product through relationships such as \\(\\|x\\|^2 = x^T x\\).",
  },

  {
    id: "la-crash-l1-q38",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements about the expression \\(x^T x\\) are correct?",
    options: [
      { text: "It equals the squared L2 norm of the vector.", isCorrect: true },
      { text: "It is always nonnegative.", isCorrect: true },
      { text: "It equals \\(\\sum_i x_i^2\\).", isCorrect: true },
      {
        text: "It represents Euclidean distance between two different vectors.",
        isCorrect: false,
      },
    ],
    explanation:
      "The expression \\(x^T x\\) computes the dot product of a vector with itself. This equals the squared L2 norm and is always nonnegative. Euclidean distance requires subtracting two vectors before computing the norm.",
  },

  {
    id: "la-crash-l1-q39",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about high-dimensional geometry in machine learning are correct?",
    options: [
      {
        text: "Operations like dot products extend naturally to high dimensions.",
        isCorrect: true,
      },
      {
        text: "Many machine learning models operate in spaces with hundreds or thousands of dimensions.",
        isCorrect: true,
      },
      {
        text: "Geometric concepts like angles and norms still apply in high dimensions.",
        isCorrect: true,
      },
      {
        text: "Linear algebra only applies to two- or three-dimensional vectors.",
        isCorrect: false,
      },
    ],
    explanation:
      "Linear algebra operations such as dot products, norms, and projections generalize naturally to high-dimensional spaces. Modern machine learning models often operate in spaces with hundreds or thousands of dimensions, such as embedding spaces.",
  },

  {
    id: "la-crash-l1-q40",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements explain why vectors and dot products are central to machine learning models?",
    options: [
      { text: "Data is typically represented as vectors.", isCorrect: true },
      {
        text: "Neural network computations rely heavily on dot products.",
        isCorrect: true,
      },
      {
        text: "Similarity between representations can be measured using vector operations.",
        isCorrect: true,
      },
      {
        text: "Vector representations are limited to classical statistics models only.",
        isCorrect: false,
      },
    ],
    explanation:
      "Vectors are the fundamental representation of data in most machine learning systems. Neural networks perform computations through dot products and matrix multiplications. These operations enable models to measure similarity, combine features, and transform representations across layers.",
  },

  {
    id: "la-crash-l1-q41",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "A displacement vector is \\(x=[6,8]^T\\). Which statement correctly describes its L2 norm?",
    options: [
      {
        text: "The L2 norm is 14 because the components add to 14.",
        isCorrect: false,
      },
      {
        text: "The squared L2 norm is 10.",
        isCorrect: false,
      },
      { text: "The L2 norm is 10.", isCorrect: true },
      { text: "The vector is already a unit vector.", isCorrect: false },
    ],
    explanation:
      "The L2 norm is computed with the Pythagorean theorem: \\(\\sqrt{6^2+8^2}=\\sqrt{100}=10\\). Adding components gives 14, which is not the Euclidean length, and the squared norm is 100 rather than 10. A unit vector would have length 1, so this vector is not a unit vector.",
  },

  {
    id: "la-crash-l1-q42",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Two points are represented by \\(x=[2,5]^T\\) and \\(y=[-1,1]^T\\). Which statement gives their Euclidean distance?",
    options: [
      { text: "The distance is 4.", isCorrect: false },
      { text: "The distance is 5.", isCorrect: true },
      { text: "The distance is \\(\\sqrt{41}\\).", isCorrect: false },
      { text: "The distance is 9.", isCorrect: false },
    ],
    explanation:
      "The difference vector is \\(x-y=[3,4]^T\\), so the Euclidean distance is \\(\\sqrt{3^2+4^2}=5\\). The values 4 and 9 come from using only part of the displacement or adding components instead of using squared differences. The value \\(\\sqrt{41}\\) would come from a different pair of component differences.",
  },

  {
    id: "la-crash-l1-q43",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Let \\(a=[3,0]^T\\) and \\(b=[0,4]^T\\). Which statement correctly describes their cosine similarity?",
    options: [
      { text: "The cosine similarity is 1.", isCorrect: false },
      { text: "The cosine similarity is -1.", isCorrect: false },
      {
        text: "The cosine similarity is undefined because both vectors are nonzero.",
        isCorrect: false,
      },
      { text: "The cosine similarity is 0.", isCorrect: true },
    ],
    explanation:
      "The dot product is \\(3\\cdot0+0\\cdot4=0\\), while both vectors have nonzero length. Therefore \\(\\frac{a\\cdot b}{\\|a\\|\\|b\\|}=0\\), which means the vectors are orthogonal. Cosine similarity would be 1 for the same direction and -1 for opposite directions.",
  },

  {
    id: "la-crash-l1-q44",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "A simple neuron computes \\(y=w^T x+b\\), where \\(w=[2,-1]^T\\), \\(x=[3,4]^T\\), and \\(b=5\\). Which value of \\(y\\) is correct?",
    options: [
      { text: "\\(y=6\\)", isCorrect: false },
      { text: "\\(y=15\\)", isCorrect: false },
      { text: "\\(y=7\\)", isCorrect: true },
      { text: "\\(y=-3\\)", isCorrect: false },
    ],
    explanation:
      "The dot product is \\(w^Tx=2\\cdot3+(-1)\\cdot4=2\\). Adding the bias gives \\(y=2+5=7\\). The other values come from ignoring the negative weight, omitting the bias, or combining the terms incorrectly.",
  },

  {
    id: "la-crash-l1-q45",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe normalizing \\(v=[3,4]^T\\) to unit length?",
    options: [
      {
        text: "The normalized vector is \\([\\frac{3}{5},\\frac{4}{5}]^T\\).",
        isCorrect: true,
      },
      {
        text: "The normalized vector points in a perpendicular direction to \\(v\\).",
        isCorrect: false,
      },
      {
        text: "The normalized vector has L2 norm 1.",
        isCorrect: true,
      },
      {
        text: "The normalized vector has the same length as \\(v\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The vector \\(v\\) has length 5, so dividing each component by 5 gives \\([\\frac{3}{5},\\frac{4}{5}]^T\\). Normalization preserves direction but changes length to 1. It does not rotate the vector or leave its original length unchanged.",
  },

  {
    id: "la-crash-l1-q46",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Let \\(a=[2,1]^T\\) and \\(b=[-3,-1]^T\\). Which statements are correct?",
    options: [
      { text: "The dot product \\(a\\cdot b\\) equals -7.", isCorrect: true },
      {
        text: "The angle between the vectors is obtuse.",
        isCorrect: true,
      },
      {
        text: "The cosine similarity between the vectors is positive.",
        isCorrect: false,
      },
      { text: "The vectors are orthogonal.", isCorrect: false },
    ],
    explanation:
      "The dot product is \\(2\\cdot(-3)+1\\cdot(-1)=-7\\). A negative dot product means the angle is greater than 90 degrees, so the cosine similarity is negative rather than positive. Orthogonal vectors would have dot product zero, not -7.",
  },

  {
    id: "la-crash-l1-q47",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Let \\(a=[2,0]^T\\) and \\(b=[3,4]^T\\). Which statements are correct?",
    options: [
      { text: "\\(a\\cdot b=6\\).", isCorrect: true },
      {
        text: "All of \\(b\\) lies in the direction of \\(a\\).",
        isCorrect: false,
      },
      {
        text: "The component of \\(b\\) along the positive x-axis is 3.",
        isCorrect: true,
      },
      { text: "\\(a\\) and \\(b\\) are orthogonal.", isCorrect: false },
    ],
    explanation:
      "The dot product is \\(2\\cdot3+0\\cdot4=6\\). Since \\(a\\) points along the positive x-axis, the x-component of \\(b\\) is the part aligned with that direction, which is 3. The y-component of \\(b\\) is nonzero, so not all of \\(b\\) lies along \\(a\\), and the dot product is not zero.",
  },

  {
    id: "la-crash-l1-q48",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose \\(W=\\begin{bmatrix}1&2\\\\-1&3\\\\0&4\\end{bmatrix}\\) and \\(x=[2,1]^T\\). Which statements are correct about \\(y=Wx\\)?",
    options: [
      { text: "The output vector has 3 components.", isCorrect: true },
      {
        text: "The multiplication is invalid because \\(W\\) is not square.",
        isCorrect: false,
      },
      { text: "\\(y=[4,1,4]^T\\).", isCorrect: true },
      {
        text: "Each output component is the dot product of \\(x\\) with a column of \\(W\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The matrix has shape \\(3\\times2\\), so multiplying by a 2-component vector is valid and produces a 3-component output. The row dot products are \\(1\\cdot2+2\\cdot1=4\\), \\((-1)\\cdot2+3\\cdot1=1\\), and \\(0\\cdot2+4\\cdot1=4\\). Matrix-vector multiplication uses rows of the matrix for output components, not columns.",
  },

  {
    id: "la-crash-l1-q49",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "A small scoring model uses a feature vector \\(x=[\\text{size},\\text{bedrooms},\\text{age}]^T\\) and weights \\(w=[0.4,0.2,-0.1]^T\\). Which statements are correct?",
    options: [
      {
        text: "The feature vector can be interpreted as a point in \\(\\mathbb{R}^3\\).",
        isCorrect: true,
      },
      {
        text: "\\(w^Tx\\) is a weighted sum of the three input features.",
        isCorrect: true,
      },
      {
        text: "The negative age weight lowers the score as age increases, holding the other features fixed.",
        isCorrect: true,
      },
      {
        text: "Before adding any bias, the score depends on alignment between \\(w\\) and \\(x\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The feature vector has three numerical components, so it can be viewed geometrically in \\(\\mathbb{R}^3\\). The expression \\(w^Tx\\) computes a weighted sum, and the negative weight on age subtracts from the score when age increases. Geometrically, the dot product is larger when the input vector aligns more strongly with the weight vector.",
  },

  {
    id: "la-crash-l1-q50",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Let \\(c>0\\), and let \\(x\\) and \\(z\\) be nonzero vectors. Which statements about replacing \\(x\\) with \\(cx\\) are correct?",
    options: [
      {
        text: "The direction of \\(x\\) is unchanged.",
        isCorrect: true,
      },
      {
        text: "The L2 norm is multiplied by \\(c\\).",
        isCorrect: true,
      },
      {
        text: "The dot product with \\(z\\) is multiplied by \\(c\\).",
        isCorrect: true,
      },
      {
        text: "The cosine similarity with \\(z\\) stays the same.",
        isCorrect: true,
      },
    ],
    explanation:
      "Multiplying a nonzero vector by a positive scalar changes its length but preserves its direction. The norm becomes \\(\\|cx\\|=c\\|x\\|\\), and the dot product becomes \\((cx)\\cdot z=c(x\\cdot z)\\). In cosine similarity, that same positive factor appears in both the numerator and denominator, so it cancels out.",
  },
];

export const CrashCourseLinearAlgebraL1Questions =
  CrashCourseLinearAlgebraLecture1Questions;
