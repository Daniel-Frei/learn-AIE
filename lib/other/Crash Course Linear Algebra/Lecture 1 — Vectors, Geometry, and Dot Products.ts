import { Question } from "../../quiz";

export const CrashCourseLinearAlgebraL1Questions: Question[] = [
  {
    id: "crash-linalg-l1-q01",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which of the following are valid interpretations of a vector \\(\\mathbf{x} \\in \\mathbb{R}^n\\) in machine learning?",
    options: [
      {
        text: "A vector can represent a point in an \\(n\\)-dimensional space.",
        isCorrect: true,
      },
      {
        text: "A vector can represent a direction starting from the origin.",
        isCorrect: true,
      },
      {
        text: "A vector can represent a collection of numerical features.",
        isCorrect: true,
      },
      {
        text: "A vector can represent a word embedding in a language model.",
        isCorrect: true,
      },
    ],
    explanation:
      "In machine learning, vectors are geometric objects and feature containers. A word embedding, for example, is simply a vector in a high-dimensional space. All four interpretations are valid and commonly used.",
  },
  {
    id: "crash-linalg-l1-q02",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Let \\(\\mathbf{x} = \\begin{bmatrix}3 \\\\ 4\\end{bmatrix}\\). Which statements about its L2 norm \\(\\|\\mathbf{x}\\|_2\\) are correct?",
    options: [
      { text: "\\(\\|\\mathbf{x}\\|_2 = 5\\).", isCorrect: true },
      {
        text: "The norm corresponds to the Euclidean length of the vector.",
        isCorrect: true,
      },
      {
        text: "The norm can be computed using the Pythagorean theorem.",
        isCorrect: true,
      },
      { text: "The norm equals \\(3 + 4 = 7\\).", isCorrect: false },
    ],
    explanation:
      "The L2 norm is computed as \\(\\sqrt{3^2 + 4^2} = 5\\). It corresponds to Euclidean length and directly follows from the Pythagorean theorem. The sum of components is not the L2 norm.",
  },
  {
    id: "crash-linalg-l1-q03",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Given \\(\\mathbf{a}, \\mathbf{b} \\in \\mathbb{R}^n\\), which statements about the dot product \\(\\mathbf{a}^T \\mathbf{b}\\) are correct?",
    options: [
      { text: "It can be written as \\(\\sum_i a_i b_i\\).", isCorrect: true },
      {
        text: "It equals \\(\\|\\mathbf{a}\\| \\|\\mathbf{b}\\| \\cos \\theta\\).",
        isCorrect: true,
      },
      { text: "It measures alignment between two vectors.", isCorrect: true },
      {
        text: "It always equals the Euclidean distance between vectors.",
        isCorrect: false,
      },
    ],
    explanation:
      "The dot product has both algebraic and geometric definitions. It measures alignment through the cosine of the angle between vectors. It does not measure Euclidean distance; that requires subtracting vectors first.",
  },
  {
    id: "crash-linalg-l1-q04",
    chapter: 1,
    difficulty: "easy",
    prompt: "If two vectors are orthogonal, which statements are correct?",
    options: [
      { text: "Their dot product equals zero.", isCorrect: true },
      { text: "The angle between them is 90 degrees.", isCorrect: true },
      { text: "They have no linear overlap in direction.", isCorrect: true },
      { text: "They must have the same magnitude.", isCorrect: false },
    ],
    explanation:
      "Orthogonal vectors have zero dot product and are at 90 degrees to each other. This means they share no directional alignment. Their magnitudes can be completely different.",
  },
  {
    id: "crash-linalg-l1-q05",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "In a neural network layer \\( y = \\mathbf{w}^T \\mathbf{x} + b \\), which statements are correct?",
    options: [
      {
        text: "The term \\(\\mathbf{w}^T \\mathbf{x}\\) is a dot product.",
        isCorrect: true,
      },
      {
        text: "The neuron output measures alignment between input and weights.",
        isCorrect: true,
      },
      {
        text: "Each component of \\(\\mathbf{x}\\) contributes linearly before activation.",
        isCorrect: true,
      },
      {
        text: "The dot product computes Euclidean distance between \\(\\mathbf{w}\\) and \\(\\mathbf{x}\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "A neuron computes a weighted sum, which is a dot product. This measures how aligned the input features are with learned weights. It does not compute Euclidean distance.",
  },
  {
    id: "crash-linalg-l1-q06",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about cosine similarity \\( \\frac{\\mathbf{a}^T \\mathbf{b}}{\\|\\mathbf{a}\\|\\|\\mathbf{b}\\|} \\) are correct?",
    options: [
      {
        text: "It depends only on the angle between vectors.",
        isCorrect: true,
      },
      { text: "It is invariant to scaling of vectors.", isCorrect: true },
      {
        text: "It equals 1 when vectors point in the same direction.",
        isCorrect: true,
      },
      {
        text: "It increases if we multiply only one vector by a scalar.",
        isCorrect: false,
      },
    ],
    explanation:
      "Cosine similarity normalizes by magnitude, so scaling either vector does not change the result. It purely reflects directional similarity. If vectors align perfectly, cosine similarity equals 1.",
  },
  {
    id: "crash-linalg-l1-q07",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Let \\( W \\in \\mathbb{R}^{m \\times n} \\) and \\( \\mathbf{x} \\in \\mathbb{R}^n \\). Which statements about \\( W\\mathbf{x} \\) are correct?",
    options: [
      { text: "The result is in \\(\\mathbb{R}^m\\).", isCorrect: true },
      {
        text: "Each component of the output is a dot product between a row of \\(W\\) and \\(\\mathbf{x}\\).",
        isCorrect: true,
      },
      {
        text: "Matrix–vector multiplication performs multiple dot products in parallel.",
        isCorrect: true,
      },
      {
        text: "The result is always orthogonal to \\(\\mathbf{x}\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Matrix–vector multiplication produces an m-dimensional vector. Each output component is the dot product of one row of W with x. There is no guarantee of orthogonality.",
  },
  {
    id: "crash-linalg-l1-q08",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about Euclidean distance \\( \\|\\mathbf{x} - \\mathbf{y}\\|_2 \\) are correct?",
    options: [
      {
        text: "It measures geometric distance between two points.",
        isCorrect: true,
      },
      {
        text: "It equals zero if and only if \\(\\mathbf{x} = \\mathbf{y}\\).",
        isCorrect: true,
      },
      {
        text: "It can be derived from the Pythagorean theorem.",
        isCorrect: true,
      },
      { text: "It equals \\(\\mathbf{x}^T \\mathbf{y}\\).", isCorrect: false },
    ],
    explanation:
      "Euclidean distance is defined as the norm of the difference vector. It is zero only when vectors are identical. It is not computed via dot product alone.",
  },
  {
    id: "crash-linalg-l1-q09",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which of the following are true about high-dimensional vectors in machine learning?",
    options: [
      {
        text: "They follow the same geometric rules as 2D vectors.",
        isCorrect: true,
      },
      {
        text: "Dot products are still defined as \\(\\sum_i a_i b_i\\).",
        isCorrect: true,
      },
      {
        text: "They can represent embeddings of words or images.",
        isCorrect: true,
      },
      {
        text: "Angles between vectors become undefined in higher dimensions.",
        isCorrect: false,
      },
    ],
    explanation:
      "High-dimensional spaces obey the same linear algebra rules as low dimensions. Dot products and angles are well-defined. Embeddings are simply high-dimensional vectors.",
  },
  {
    id: "crash-linalg-l1-q10",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose \\( \\mathbf{a}^T \\mathbf{b} < 0 \\). Which statements are correct?",
    options: [
      {
        text: "The angle between \\(\\mathbf{a}\\) and \\(\\mathbf{b}\\) is greater than 90 degrees.",
        isCorrect: true,
      },
      { text: "The cosine of the angle is negative.", isCorrect: true },
      {
        text: "The vectors point in generally opposite directions.",
        isCorrect: true,
      },
      { text: "The vectors must have equal magnitudes.", isCorrect: false },
    ],
    explanation:
      "A negative dot product implies \\(\\cos \\theta < 0\\), meaning the angle exceeds 90 degrees. This indicates opposite directional alignment. Magnitudes are unrelated to the sign.",
  },

  {
    id: "crash-linalg-l1-q21",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Let \\(\\mathbf{x} \\in \\mathbb{R}^n\\). Which of the following operations produce a scalar?",
    options: [
      { text: "\\(\\mathbf{x}^T \\mathbf{x}\\)", isCorrect: true },
      {
        text: "The L2 norm squared \\(\\|\\mathbf{x}\\|_2^2\\)",
        isCorrect: true,
      },
      {
        text: "The dot product \\(\\mathbf{x}^T \\mathbf{y}\\)",
        isCorrect: true,
      },
      {
        text: "The cosine similarity between \\(\\mathbf{x}\\) and \\(\\mathbf{y}\\)",
        isCorrect: true,
      },
    ],
    explanation:
      "All four expressions result in scalars. A dot product is a scalar, and \\(\\mathbf{x}^T \\mathbf{x}\\) equals the squared L2 norm. Cosine similarity is also a scalar between -1 and 1.",
  },
  {
    id: "crash-linalg-l1-q22",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about \\(\\mathbf{x}^T \\mathbf{x}\\) are correct?",
    options: [
      { text: "It equals \\(\\|\\mathbf{x}\\|_2^2\\).", isCorrect: true },
      { text: "It is always non-negative.", isCorrect: true },
      {
        text: "It equals zero if and only if \\(\\mathbf{x} = 0\\).",
        isCorrect: true,
      },
      {
        text: "It measures Euclidean distance between two distinct vectors.",
        isCorrect: false,
      },
    ],
    explanation:
      "\\(\\mathbf{x}^T \\mathbf{x}\\) sums squared components, which equals the squared L2 norm. It is zero only for the zero vector. It does not measure distance between two different vectors.",
  },
  {
    id: "crash-linalg-l1-q23",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose \\(\\mathbf{a}\\) and \\(\\mathbf{b}\\) are non-zero and \\(\\mathbf{a}^T \\mathbf{b} = \\|\\mathbf{a}\\| \\|\\mathbf{b}\\|\\). Which statements are correct?",
    options: [
      { text: "The angle between them is 0 degrees.", isCorrect: true },
      { text: "They point in the same direction.", isCorrect: true },
      { text: "Cosine similarity equals 1.", isCorrect: true },
      { text: "They must be orthogonal.", isCorrect: false },
    ],
    explanation:
      "If the dot product equals the product of norms, then \\(\\cos \\theta = 1\\), meaning the vectors are perfectly aligned. Orthogonality would imply a dot product of zero.",
  },
  {
    id: "crash-linalg-l1-q24",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which of the following are true about a weight matrix \\(W\\) in a neural network layer?",
    options: [
      {
        text: "Each row of \\(W\\) defines a weight vector for one neuron.",
        isCorrect: true,
      },
      {
        text: "Multiplying \\(W\\mathbf{x}\\) computes multiple dot products.",
        isCorrect: true,
      },
      {
        text: "The output dimension equals the number of rows in \\(W\\).",
        isCorrect: true,
      },
      {
        text: "The matrix multiplication computes Euclidean distances.",
        isCorrect: false,
      },
    ],
    explanation:
      "Each row corresponds to a neuron’s weights. Matrix–vector multiplication computes parallel dot products, producing one output per row. It does not compute distances.",
  },
  {
    id: "crash-linalg-l1-q25",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about scaling a vector \\(c\\mathbf{x}\\) (where \\(c \\neq 0\\)) are correct?",
    options: [
      {
        text: "Its magnitude becomes \\(|c| \\|\\mathbf{x}\\|\\).",
        isCorrect: true,
      },
      {
        text: "Its direction remains the same if \\(c > 0\\).",
        isCorrect: true,
      },
      { text: "Its direction flips if \\(c < 0\\).", isCorrect: true },
      {
        text: "Cosine similarity with \\(\\mathbf{x}\\) remains unchanged if both vectors are scaled.",
        isCorrect: true,
      },
    ],
    explanation:
      "Scaling multiplies the norm by \\(|c|\\). Positive scaling preserves direction, negative scaling reverses it. Cosine similarity is invariant to scaling both vectors.",
  },
  {
    id: "crash-linalg-l1-q26",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about the expression \\(\\|\\mathbf{x} - \\mathbf{y}\\|_2^2\\) are correct?",
    options: [
      {
        text: "It equals \\((\\mathbf{x} - \\mathbf{y})^T (\\mathbf{x} - \\mathbf{y})\\).",
        isCorrect: true,
      },
      { text: "It is always non-negative.", isCorrect: true },
      {
        text: "It equals zero only when \\(\\mathbf{x} = \\mathbf{y}\\).",
        isCorrect: true,
      },
      { text: "It equals \\(\\mathbf{x}^T \\mathbf{y}\\).", isCorrect: false },
    ],
    explanation:
      "Squared Euclidean distance can be written as a dot product of the difference vector with itself. It is non-negative and only zero when the vectors are identical. It is not simply \\(\\mathbf{x}^T \\mathbf{y}\\).",
  },
  {
    id: "crash-linalg-l1-q27",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "If \\(\\mathbf{a}^T \\mathbf{b} = 0\\) and both are non-zero, which statements are correct?",
    options: [
      { text: "They are orthogonal.", isCorrect: true },
      { text: "Their cosine similarity is 0.", isCorrect: true },
      { text: "They share no directional alignment.", isCorrect: true },
      { text: "They must have equal magnitudes.", isCorrect: false },
    ],
    explanation:
      "A zero dot product for non-zero vectors implies orthogonality. This corresponds to cosine similarity equal to zero. Their magnitudes are independent.",
  },
  {
    id: "crash-linalg-l1-q28",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about vectors in \\(\\mathbb{R}^{768}\\) (like word embeddings) are correct?",
    options: [
      {
        text: "They follow the same algebraic rules as vectors in \\(\\mathbb{R}^2\\).",
        isCorrect: true,
      },
      {
        text: "Dot products are computed component-wise and summed.",
        isCorrect: true,
      },
      {
        text: "They can be compared using cosine similarity.",
        isCorrect: true,
      },
      { text: "They cannot be visualized directly in 2D.", isCorrect: true },
    ],
    explanation:
      "High-dimensional vectors follow identical linear algebra rules. Dot products and cosine similarity work the same way. Direct visualization is not possible, though projections can be used.",
  },
  {
    id: "crash-linalg-l1-q29",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which of the following are true about linear transformations represented by matrices?",
    options: [
      { text: "They preserve vector addition.", isCorrect: true },
      { text: "They preserve scalar multiplication.", isCorrect: true },
      { text: "They can rotate or scale vectors.", isCorrect: true },
      {
        text: "They can introduce nonlinear curvature in space.",
        isCorrect: false,
      },
    ],
    explanation:
      "Linear transformations satisfy additivity and homogeneity. They can rotate, scale, or shear space. Nonlinear curvature requires nonlinear functions.",
  },
  {
    id: "crash-linalg-l1-q30",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose \\(W\\mathbf{x} = 0\\) for a non-zero vector \\(\\mathbf{x}\\). Which statements are correct?",
    options: [
      {
        text: "\\(\\mathbf{x}\\) lies in the null space of \\(W\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\mathbf{x}\\) is orthogonal to every row of \\(W\\).",
        isCorrect: true,
      },
      {
        text: "Each row of \\(W\\) has zero dot product with \\(\\mathbf{x}\\).",
        isCorrect: true,
      },
      { text: "\\(W\\) must be the zero matrix.", isCorrect: false },
    ],
    explanation:
      "If \\(W\\mathbf{x} = 0\\), then each row’s dot product with \\(\\mathbf{x}\\) is zero. That means \\(\\mathbf{x}\\) lies in the null space. The matrix itself does not need to be zero.",
  },
  {
    id: "crash-linalg-l1-q31",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about L1 norm \\(\\|\\mathbf{x}\\|_1 = \\sum_i |x_i|\\) are correct?",
    options: [
      { text: "It sums absolute values of components.", isCorrect: true },
      { text: "It can encourage sparsity in optimization.", isCorrect: true },
      { text: "It equals Euclidean length.", isCorrect: false },
      {
        text: "It is always greater than or equal to the L2 norm.",
        isCorrect: false,
      },
    ],
    explanation:
      "The L1 norm sums absolute values. It encourages sparsity in regularization. It is not equal to Euclidean length, and it is not always greater than or equal to L2 norm.",
  },
  {
    id: "crash-linalg-l1-q32",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about cosine similarity are correct?",
    options: [
      { text: "It ranges between -1 and 1.", isCorrect: true },
      { text: "It equals 0 for orthogonal vectors.", isCorrect: true },
      { text: "It depends on vector magnitudes.", isCorrect: false },
      {
        text: "It increases if both vectors are scaled equally.",
        isCorrect: false,
      },
    ],
    explanation:
      "Cosine similarity measures directional similarity only. It ranges from -1 to 1 and equals zero for orthogonal vectors. It is invariant to scaling.",
  },
  {
    id: "crash-linalg-l1-q33",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements are true about the expression \\(\\mathbf{w}^T \\mathbf{x} + b\\) in neural networks?",
    options: [
      {
        text: "It defines a linear decision boundary before applying activation.",
        isCorrect: true,
      },
      { text: "It is affine due to the bias term.", isCorrect: true },
      {
        text: "Without nonlinearity, stacking such layers collapses into one linear transform.",
        isCorrect: true,
      },
      { text: "It computes cosine similarity directly.", isCorrect: false },
    ],
    explanation:
      "The expression defines an affine transformation. Without nonlinearities, multiple such layers reduce to a single linear mapping. It does not directly compute cosine similarity.",
  },
  {
    id: "crash-linalg-l1-q34",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements about the angle between vectors are correct?",
    options: [
      {
        text: "It can be derived from the dot product formula.",
        isCorrect: true,
      },
      {
        text: "It is computed via \\(\\cos^{-1}(\\frac{\\mathbf{a}^T \\mathbf{b}}{\\|\\mathbf{a}\\|\\|\\mathbf{b}\\|})\\).",
        isCorrect: true,
      },
      { text: "It reflects directional similarity.", isCorrect: true },
      { text: "It depends only on magnitudes.", isCorrect: false },
    ],
    explanation:
      "The angle between vectors is derived from the geometric definition of the dot product. It reflects alignment and does not depend solely on magnitudes.",
  },
  {
    id: "crash-linalg-l1-q35",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which of the following are true regarding \\(\\mathbf{x}^T \\mathbf{y}\\)?",
    options: [
      { text: "It equals \\(\\mathbf{y}^T \\mathbf{x}\\).", isCorrect: true },
      {
        text: "It is bilinear in \\(\\mathbf{x}\\) and \\(\\mathbf{y}\\).",
        isCorrect: true,
      },
      { text: "It is symmetric for real vectors.", isCorrect: true },
      { text: "It always produces a vector.", isCorrect: false },
    ],
    explanation:
      "The dot product is symmetric for real vectors and bilinear. It always produces a scalar, not a vector.",
  },
  {
    id: "crash-linalg-l1-q36",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose \\(\\mathbf{a}^T \\mathbf{b} = \\|\\mathbf{a}\\| \\|\\mathbf{b}\\| \\cos \\theta\\). Which statements are correct?",
    options: [
      {
        text: "If \\(\\theta = 90^\\circ\\), the dot product is zero.",
        isCorrect: true,
      },
      {
        text: "If \\(\\theta = 180^\\circ\\), the dot product is negative.",
        isCorrect: true,
      },
      {
        text: "If \\(\\theta = 0^\\circ\\), the dot product is maximal for fixed norms.",
        isCorrect: true,
      },
      { text: "The dot product depends only on the angle.", isCorrect: false },
    ],
    explanation:
      "The dot product depends on both magnitude and angle. At 90 degrees it is zero, at 180 degrees it is negative, and at 0 degrees it is maximal for fixed magnitudes.",
  },
  {
    id: "crash-linalg-l1-q37",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about feature vectors in machine learning are correct?",
    options: [
      {
        text: "Each component represents a measurable attribute.",
        isCorrect: true,
      },
      { text: "They can be inputs to neural networks.", isCorrect: true },
      {
        text: "They can represent states in reinforcement learning.",
        isCorrect: true,
      },
      { text: "They cannot be transformed by matrices.", isCorrect: false },
    ],
    explanation:
      "Feature vectors represent measurable quantities and are fed into models. They are transformed via matrix multiplication in neural networks.",
  },
  {
    id: "crash-linalg-l1-q38",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about orthogonality are correct?",
    options: [
      { text: "Orthogonal vectors have zero dot product.", isCorrect: true },
      { text: "They can still have non-zero magnitude.", isCorrect: true },
      {
        text: "They correspond to 90-degree separation in Euclidean space.",
        isCorrect: true,
      },
      { text: "They must be unit vectors.", isCorrect: false },
    ],
    explanation:
      "Orthogonality only requires zero dot product. Vectors can have arbitrary magnitudes. Unit length is not required.",
  },
  {
    id: "crash-linalg-l1-q39",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "If \\(W\\mathbf{x}\\) performs multiple dot products, which statements are correct?",
    options: [
      {
        text: "Each row of \\(W\\) acts as a feature detector.",
        isCorrect: true,
      },
      {
        text: "The operation is linear in \\(\\mathbf{x}\\).",
        isCorrect: true,
      },
      {
        text: "It preserves addition and scalar multiplication.",
        isCorrect: true,
      },
      {
        text: "It introduces nonlinear activation effects by itself.",
        isCorrect: false,
      },
    ],
    explanation:
      "Matrix multiplication is linear and preserves vector addition and scaling. Nonlinearity must be introduced separately through activation functions.",
  },
  {
    id: "crash-linalg-l1-q40",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about geometric interpretation of neural networks are correct?",
    options: [
      {
        text: "Layers perform linear transformations of vector spaces.",
        isCorrect: true,
      },
      {
        text: "Dot products measure alignment between learned weights and inputs.",
        isCorrect: true,
      },
      {
        text: "Cosine similarity can be used for embedding comparison.",
        isCorrect: true,
      },
      {
        text: "Neural networks operate outside vector space mathematics.",
        isCorrect: false,
      },
    ],
    explanation:
      "Neural networks are built from linear algebra operations. Dot products and cosine similarity are fundamental geometric tools. They operate entirely within vector space mathematics.",
  },

  {
    id: "crash-linalg-l1-q11",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about column vectors in neural network notation are correct?",
    options: [
      {
        text: "They are typically written as \\(n \\times 1\\) matrices.",
        isCorrect: true,
      },
      {
        text: "They allow \\(\\mathbf{w}^T \\mathbf{x}\\) to compute a dot product.",
        isCorrect: true,
      },
      {
        text: "They represent points in \\(\\mathbb{R}^n\\).",
        isCorrect: true,
      },
      { text: "They must always have unit length.", isCorrect: false },
    ],
    explanation:
      "Column vectors are typically represented as \\(n \\times 1\\) matrices in deep learning. This allows \\(\\mathbf{w}^T \\mathbf{x}\\) to compute a scalar dot product. Vectors do not need to have unit length unless explicitly normalized.",
  },
  {
    id: "crash-linalg-l1-q12",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements are true about the geometric interpretation of the dot product?",
    options: [
      {
        text: "It measures how much one vector projects onto another.",
        isCorrect: true,
      },
      { text: "It depends on both magnitude and angle.", isCorrect: true },
      { text: "It becomes zero when vectors are orthogonal.", isCorrect: true },
      {
        text: "It is unaffected by changing the magnitude of only one vector.",
        isCorrect: false,
      },
    ],
    explanation:
      "The dot product captures projection and depends on both magnitudes and the cosine of the angle. If vectors are orthogonal, the cosine is zero. Changing magnitude changes the dot product unless cosine similarity is used.",
  },
  {
    id: "crash-linalg-l1-q13",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Let \\(\\mathbf{x} \\in \\mathbb{R}^n\\) and define \\(f(\\mathbf{x}) = \\mathbf{w}^T \\mathbf{x}\\). Which statements are correct?",
    options: [
      { text: "The function is linear in \\(\\mathbf{x}\\).", isCorrect: true },
      {
        text: "Scaling \\(\\mathbf{x}\\) by \\(c\\) scales the output by \\(c\\).",
        isCorrect: true,
      },
      {
        text: "It satisfies \\(f(\\mathbf{x}_1 + \\mathbf{x}_2) = f(\\mathbf{x}_1) + f(\\mathbf{x}_2)\\).",
        isCorrect: true,
      },
      {
        text: "It introduces nonlinear curvature in the input space.",
        isCorrect: false,
      },
    ],
    explanation:
      "A dot product defines a linear function. It satisfies both homogeneity and additivity. It does not introduce nonlinearity; curvature requires nonlinear activation functions.",
  },
  {
    id: "crash-linalg-l1-q14",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which of the following are true about the vector difference \\(\\mathbf{x} - \\mathbf{y}\\)?",
    options: [
      {
        text: "It represents the displacement from \\(\\mathbf{y}\\) to \\(\\mathbf{x}\\).",
        isCorrect: true,
      },
      {
        text: "Its L2 norm equals Euclidean distance between the two vectors.",
        isCorrect: true,
      },
      {
        text: "It is zero when the two vectors are identical.",
        isCorrect: true,
      },
      {
        text: "It always has larger magnitude than either \\(\\mathbf{x}\\) or \\(\\mathbf{y}\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The difference vector encodes displacement. Its L2 norm gives Euclidean distance. Its magnitude depends on relative positions and is not guaranteed to be larger than individual vectors.",
  },
  {
    id: "crash-linalg-l1-q15",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about vector magnitude \\(\\|\\mathbf{x}\\|_2\\) are correct?",
    options: [
      { text: "It equals zero only for the zero vector.", isCorrect: true },
      { text: "It is always non-negative.", isCorrect: true },
      { text: "It measures distance from the origin.", isCorrect: true },
      {
        text: "It depends on the angle with another vector.",
        isCorrect: false,
      },
    ],
    explanation:
      "The L2 norm is the Euclidean length of a vector. It is non-negative and equals zero only at the origin. It depends solely on the vector’s components, not its angle with others.",
  },
  {
    id: "crash-linalg-l1-q16",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose \\(\\mathbf{a}^T \\mathbf{b} = 5\\), \\(\\|\\mathbf{a}\\| = 1\\), and \\(\\|\\mathbf{b}\\| = 10\\). Which statements are correct?",
    options: [
      { text: "The cosine of the angle equals \\(0.5\\).", isCorrect: true },
      { text: "The angle between the vectors is acute.", isCorrect: true },
      { text: "The angle is 90 degrees.", isCorrect: false },
      { text: "The vectors are orthogonal.", isCorrect: false },
    ],
    explanation:
      "Using \\(\\mathbf{a}^T \\mathbf{b} = \\|\\mathbf{a}\\|\\|\\mathbf{b}\\| \\cos\\theta\\), we get \\(5 = 1 \\cdot 10 \\cdot \\cos\\theta\\), so \\(\\cos\\theta = 0.5\\). That corresponds to an acute angle. Orthogonality would require cosine zero.",
  },
  {
    id: "crash-linalg-l1-q17",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about row vectors \\(\\mathbf{x}^T\\) are correct?",
    options: [
      {
        text: "They are typically \\(1 \\times n\\) matrices.",
        isCorrect: true,
      },
      {
        text: "They allow multiplication with column vectors to produce scalars.",
        isCorrect: true,
      },
      { text: "They represent transposes of column vectors.", isCorrect: true },
      { text: "They cannot represent geometric directions.", isCorrect: false },
    ],
    explanation:
      "A row vector is the transpose of a column vector. Multiplying a row vector with a column vector produces a scalar dot product. Geometrically, they represent the same object, just differently oriented.",
  },
  {
    id: "crash-linalg-l1-q18",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements are true about attention scores computed as \\(QK^T\\)?",
    options: [
      {
        text: "They are dot products between query and key vectors.",
        isCorrect: true,
      },
      {
        text: "They measure alignment between representations.",
        isCorrect: true,
      },
      {
        text: "Large positive values indicate stronger similarity.",
        isCorrect: true,
      },
      { text: "They compute Euclidean distance directly.", isCorrect: false },
    ],
    explanation:
      "Attention scores are dot products between query and key vectors. They measure alignment, not distance. Larger dot products imply stronger similarity before softmax scaling.",
  },
  {
    id: "crash-linalg-l1-q19",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements about unit vectors are correct?",
    options: [
      { text: "They have L2 norm equal to 1.", isCorrect: true },
      {
        text: "They preserve direction but remove magnitude information.",
        isCorrect: true,
      },
      {
        text: "They are useful for computing cosine similarity.",
        isCorrect: true,
      },
      {
        text: "All vectors are automatically unit vectors in neural networks.",
        isCorrect: false,
      },
    ],
    explanation:
      "Unit vectors have length 1. Normalizing vectors preserves direction while discarding magnitude. Neural networks do not automatically normalize vectors unless explicitly designed to.",
  },
  {
    id: "crash-linalg-l1-q20",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about matrix–vector multiplication \\(W\\mathbf{x}\\) are correct?",
    options: [
      {
        text: "It defines a linear transformation of \\(\\mathbf{x}\\).",
        isCorrect: true,
      },
      {
        text: "The output dimension equals the number of rows of \\(W\\).",
        isCorrect: true,
      },
      { text: "Each output component equals a dot product.", isCorrect: true },
      {
        text: "It introduces nonlinear activation effects automatically.",
        isCorrect: false,
      },
    ],
    explanation:
      "Matrix–vector multiplication is linear. Each output is a dot product between a row of W and the vector x. Nonlinearity must be introduced separately via activation functions.",
  },
];
