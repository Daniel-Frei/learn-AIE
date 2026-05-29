import { Question } from "../../quiz";

export const CrashCourseLinearAlgebraLecture2Questions: Question[] = [
  {
    id: "la-crash-l2-q01",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements correctly describe a matrix?",
    options: [
      {
        text: "A matrix is a rectangular grid of numbers.",
        isCorrect: true,
      },
      {
        text: "A matrix can be interpreted as an operator that transforms vectors.",
        isCorrect: true,
      },
      {
        text: "A matrix can represent how an entire space is stretched, rotated, sheared, or projected.",
        isCorrect: true,
      },
      {
        text: "A matrix can be used as the learned weight object in a neural network layer.",
        isCorrect: true,
      },
    ],
    explanation:
      "A matrix is a rectangular arrangement of numbers, but in linear algebra it is more useful to view it as a transformation. Matrices can move vectors through geometric operations such as scaling, rotation, shearing, and projection. Neural network weight matrices use this same idea to transform input features into output features.",
  },

  {
    id: "la-crash-l2-q02",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe matrix-vector multiplication \\(Ax\\)?",
    options: [
      {
        text: "Each row of \\(A\\) computes a dot product with the input vector \\(x\\).",
        isCorrect: true,
      },
      {
        text: "The result has one entry for each row of \\(A\\).",
        isCorrect: true,
      },
      {
        text: "It can be viewed as a feature transformation.",
        isCorrect: true,
      },
      {
        text: "It is the core linear operation inside many neural network layers.",
        isCorrect: true,
      },
    ],
    explanation:
      "Matrix-vector multiplication combines the input vector with each row of the matrix through dot products. That produces one output value per row, so the number of rows controls the output dimension. This is why a weight matrix can act like a learned detector system or feature transformer in a neural layer.",
  },

  {
    id: "la-crash-l2-q03",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "For \\(A=\\begin{bmatrix}1 & 2 \\\\ 3 & 4\\end{bmatrix}\\) and \\(x=\\begin{bmatrix}5 \\\\ 6\\end{bmatrix}\\), which value equals \\(Ax\\)?",
    options: [
      {
        text: "\\(\\begin{bmatrix}17 \\\\ 39\\end{bmatrix}\\)",
        isCorrect: true,
      },
      {
        text: "\\(\\begin{bmatrix}23 \\\\ 34\\end{bmatrix}\\)",
        isCorrect: false,
      },
      {
        text: "\\(\\begin{bmatrix}5 \\\\ 6\\end{bmatrix}\\)",
        isCorrect: false,
      },
      {
        text: "\\(\\begin{bmatrix}39 \\\\ 17\\end{bmatrix}\\)",
        isCorrect: false,
      },
    ],
    explanation:
      "The first output entry is the first row dotted with the vector: \\(1\\cdot5+2\\cdot6=17\\). The second output entry is \\(3\\cdot5+4\\cdot6=39\\). The other values either use the wrong arithmetic, leave the vector unchanged, or swap the output order.",
  },

  {
    id: "la-crash-l2-q04",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "If \\(A \\in \\mathbb{R}^{m \\times n}\\) and \\(x \\in \\mathbb{R}^{n}\\), which statements are correct?",
    options: [
      {
        text: "\\(Ax\\) is defined because the input dimension \\(n\\) matches.",
        isCorrect: true,
      },
      {
        text: "The output \\(Ax\\) lies in \\(\\mathbb{R}^{m}\\).",
        isCorrect: true,
      },
      {
        text: "The number of rows of \\(A\\) determines the output dimension.",
        isCorrect: true,
      },
      {
        text: "The output dimension must be \\(n\\) because \\(x\\) has \\(n\\) entries.",
        isCorrect: false,
      },
    ],
    explanation:
      "For \\(A \\in \\mathbb{R}^{m \\times n}\\), the matrix accepts an \\(n\\)-dimensional vector and returns an \\(m\\)-dimensional vector. The matching inner dimension makes the multiplication valid. The incorrect statement confuses the input dimension with the output dimension.",
  },

  {
    id: "la-crash-l2-q05",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "A neural layer maps a 768-dimensional embedding to a 2048-dimensional hidden representation. Which statements are correct about a weight matrix \\(W \\in \\mathbb{R}^{2048 \\times 768}\\)?",
    options: [
      {
        text: "The matrix expects 768 input features.",
        isCorrect: true,
      },
      {
        text: "The matrix produces 2048 output features for one input vector.",
        isCorrect: true,
      },
      {
        text: "The layer can be written as \\(Wx\\) for \\(x \\in \\mathbb{R}^{768}\\).",
        isCorrect: true,
      },
      {
        text: "The matrix expects 2048 input features because 2048 appears first.",
        isCorrect: false,
      },
    ],
    explanation:
      "With the convention \\(W \\in \\mathbb{R}^{output \\times input}\\), the 768 columns match the input embedding dimension. The 2048 rows produce the hidden dimension. The incorrect statement reverses the input and output roles.",
  },

  {
    id: "la-crash-l2-q06",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "For a batch matrix \\(X \\in \\mathbb{R}^{32 \\times 768}\\), which statements are correct?",
    options: [
      {
        text: "The batch contains 32 examples.",
        isCorrect: true,
      },
      {
        text: "Each example has 768 features.",
        isCorrect: true,
      },
      {
        text: "The batch contains 768 examples.",
        isCorrect: false,
      },
      {
        text: "Each example has 32 features.",
        isCorrect: false,
      },
    ],
    explanation:
      "The common batch-by-features convention places the number of examples in the first dimension and the feature count in the second dimension. Thus \\(32\\) is the batch size and \\(768\\) is the feature dimension. The incorrect statements swap these meanings.",
  },

  {
    id: "la-crash-l2-q07",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "If \\(W \\in \\mathbb{R}^{128 \\times 512}\\) and \\(x \\in \\mathbb{R}^{512}\\), what is the dimension of \\(Wx\\)?",
    options: [
      { text: "128", isCorrect: true },
      { text: "512", isCorrect: false },
      { text: "640", isCorrect: false },
      { text: "The product is undefined.", isCorrect: false },
    ],
    explanation:
      "The 512 columns of \\(W\\) match the 512 entries of \\(x\\), so the product is defined. The output has one entry for each row of \\(W\\), giving dimension 128. The other answers either confuse input and output dimensions or ignore the valid shape match.",
  },

  {
    id: "la-crash-l2-q08",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which operations can be represented geometrically by matrices?",
    options: [
      { text: "Scaling space along coordinate directions.", isCorrect: true },
      { text: "Rotating vectors to a new orientation.", isCorrect: true },
      { text: "Shearing space so coordinates mix together.", isCorrect: true },
      {
        text: "Projecting a higher-dimensional object into fewer dimensions.",
        isCorrect: true,
      },
    ],
    explanation:
      "Matrices can represent many geometric transformations, including scaling, rotation, shearing, and projection. These operations show that matrices are more than tables of values. They describe how vectors, directions, and spaces are changed.",
  },

  {
    id: "la-crash-l2-q09",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "For the scaling matrix \\(A=\\begin{bmatrix}2 & 0 \\\\ 0 & 3\\end{bmatrix}\\), which statements are correct?",
    options: [
      {
        text: "The x-direction is stretched by a factor of 2.",
        isCorrect: true,
      },
      {
        text: "The y-direction is stretched by a factor of 3.",
        isCorrect: true,
      },
      {
        text: "The matrix rotates every vector by 90 degrees.",
        isCorrect: false,
      },
      {
        text: "The matrix maps every vector to the origin.",
        isCorrect: false,
      },
    ],
    explanation:
      "A diagonal matrix with entries 2 and 3 scales the coordinate axes independently. It doubles the x-coordinate and triples the y-coordinate. It is not a rotation matrix, and it does not collapse all vectors to zero.",
  },

  {
    id: "la-crash-l2-q10",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "What is the effect of \\(A=\\begin{bmatrix}2 & 0 \\\\ 0 & 3\\end{bmatrix}\\) on \\(x=\\begin{bmatrix}1 \\\\ 1\\end{bmatrix}\\)?",
    options: [
      {
        text: "It maps \\(x\\) to \\(\\begin{bmatrix}2 \\\\ 3\\end{bmatrix}\\).",
        isCorrect: true,
      },
      {
        text: "It maps \\(x\\) to \\(\\begin{bmatrix}3 \\\\ 2\\end{bmatrix}\\).",
        isCorrect: false,
      },
      {
        text: "It preserves the length of \\(x\\).",
        isCorrect: false,
      },
      {
        text: "It maps \\(x\\) to \\(\\begin{bmatrix}1 \\\\ 1\\end{bmatrix}\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The matrix multiplies the first coordinate by 2 and the second coordinate by 3, so \\([1,1]^T\\) becomes \\([2,3]^T\\). The coordinates are not swapped, and the vector is not left unchanged. The length changes because the coordinates have been scaled.",
  },

  {
    id: "la-crash-l2-q11",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe a two-dimensional rotation matrix \\(R=\\begin{bmatrix}\\cos\\theta & -\\sin\\theta \\\\ \\sin\\theta & \\cos\\theta\\end{bmatrix}\\)?",
    options: [
      {
        text: "It changes the orientation of vectors.",
        isCorrect: true,
      },
      {
        text: "It preserves distances between points.",
        isCorrect: true,
      },
      {
        text: "It preserves vector lengths.",
        isCorrect: true,
      },
      {
        text: "It can rotate an embedding space without changing its internal distance structure.",
        isCorrect: true,
      },
    ],
    explanation:
      "A rotation changes orientation while preserving lengths and distances. This means the geometric relationships inside a space can remain intact even when the coordinate representation changes. That intuition helps explain why embedding spaces can be transformed without destroying semantic structure when the transformation preserves distances.",
  },

  {
    id: "la-crash-l2-q12",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "For the shear matrix \\(A=\\begin{bmatrix}1 & 1 \\\\ 0 & 1\\end{bmatrix}\\), which statements are correct?",
    options: [
      {
        text: "The transformed x-coordinate depends on both the original x-coordinate and the original y-coordinate.",
        isCorrect: true,
      },
      {
        text: "The transformation slants space rather than merely rescaling each axis independently.",
        isCorrect: true,
      },
      {
        text: "The transformation can be understood as mixing features.",
        isCorrect: true,
      },
      {
        text: "The transformation deletes the y-coordinate for every input vector.",
        isCorrect: false,
      },
    ],
    explanation:
      "This shear maps \\([x,y]^T\\) to \\([x+y,y]^T\\), so the first output coordinate mixes both input coordinates. That slants the coordinate grid and gives a geometric picture of feature mixing. The y-coordinate is preserved, so it is not deleted.",
  },

  {
    id: "la-crash-l2-q13",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about projection are correct?",
    options: [
      {
        text: "Projection can reduce dimension, such as mapping a 3D object onto a 2D plane.",
        isCorrect: true,
      },
      {
        text: "Projection can remove information by collapsing different inputs to the same output.",
        isCorrect: true,
      },
      {
        text: "Projection must preserve all distances exactly.",
        isCorrect: false,
      },
      {
        text: "Projection always increases the number of independent directions.",
        isCorrect: false,
      },
    ],
    explanation:
      "A projection can map from a higher-dimensional space into a lower-dimensional one, which often loses information. Different inputs can land on the same projected point, so the original cannot always be recovered. Projection generally does not preserve every distance and does not increase independent directions.",
  },

  {
    id: "la-crash-l2-q14",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "What is the key consequence of stacking only linear layers such as \\(y=A(Bx)\\)?",
    options: [
      {
        text: "The stacked operation is equivalent to one linear transformation \\((AB)x\\).",
        isCorrect: true,
      },
      {
        text: "The stacked operation becomes nonlinear solely because two matrices are used.",
        isCorrect: false,
      },
      {
        text: "The stacked operation cannot be represented by a matrix.",
        isCorrect: false,
      },
      {
        text: "The stacked operation automatically becomes more expressive than any single linear layer.",
        isCorrect: false,
      },
    ],
    explanation:
      "The composition of linear transformations is still a linear transformation, so \\(A(Bx)\\) can be written as \\((AB)x\\). This means depth alone does not add nonlinear expressive power when every layer is linear. The incorrect statements confuse stacking with nonlinearity.",
  },

  {
    id: "la-crash-l2-q15",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which activation functions can provide the nonlinearity needed between linear layers?",
    options: [
      { text: "Rectified Linear Unit (ReLU).", isCorrect: true },
      { text: "Gaussian Error Linear Unit (GELU).", isCorrect: true },
      { text: "Sigmoid.", isCorrect: true },
      { text: "Hyperbolic tangent (tanh).", isCorrect: true },
    ],
    explanation:
      "ReLU, GELU, sigmoid, and tanh are all nonlinear activation functions used in neural networks. Placing nonlinearities between linear transformations prevents a deep stack from collapsing into a single linear transformation. This is a central reason deep neural networks can model complex relationships.",
  },

  {
    id: "la-crash-l2-q16",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "In the feed-forward pattern \\(x \\to W_1x \\to \\text{GELU} \\to W_2x\\), which statements are correct?",
    options: [
      {
        text: "\\(W_1\\) and \\(W_2\\) are linear transformations.",
        isCorrect: true,
      },
      {
        text: "GELU is what prevents the whole block from being just one linear transformation.",
        isCorrect: true,
      },
      {
        text: "The block can increase expressive power compared with a purely linear stack.",
        isCorrect: true,
      },
      {
        text: "Removing GELU would make the block more nonlinear.",
        isCorrect: false,
      },
    ],
    explanation:
      "The matrices \\(W_1\\) and \\(W_2\\) perform linear transformations, while GELU introduces nonlinearity between them. That nonlinearity is what lets the block represent functions that a single linear map cannot. Removing GELU would make the block less nonlinear, not more.",
  },

  {
    id: "la-crash-l2-q17",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements correctly describe matrix rank?",
    options: [
      {
        text: "Rank measures how many independent directions a matrix can represent.",
        isCorrect: true,
      },
      {
        text: "Low rank can indicate that some directions collapse together.",
        isCorrect: true,
      },
      {
        text: "Rank is the same thing as the largest entry in the matrix.",
        isCorrect: false,
      },
      {
        text: "A low-rank matrix must have only small numerical values.",
        isCorrect: false,
      },
    ],
    explanation:
      "Rank is about independent directions, not the magnitudes of individual entries. A matrix is low rank when it represents fewer independent output directions than the surrounding space might allow. Large entries can appear in a low-rank matrix, so value size and rank are different concepts.",
  },

  {
    id: "la-crash-l2-q18",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement best describes a full-rank transformation in intuitive terms?",
    options: [
      {
        text: "It preserves as many independent directions as the dimensions allow.",
        isCorrect: true,
      },
      {
        text: "It necessarily maps every input to zero.",
        isCorrect: false,
      },
      {
        text: "It always reduces a 3D space to a 2D plane.",
        isCorrect: false,
      },
      {
        text: "It means every matrix entry is nonzero.",
        isCorrect: false,
      },
    ],
    explanation:
      "Full rank means the matrix keeps the maximum possible number of independent directions for its shape. It is not about whether every entry is nonzero. Mapping everything to zero or collapsing 3D space to a plane would reduce independent directions, which is the opposite intuition.",
  },

  {
    id: "la-crash-l2-q19",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements explain why low-rank structure matters in machine learning?",
    options: [
      {
        text: "Low-rank approximations can reduce parameter count.",
        isCorrect: true,
      },
      {
        text: "Low-rank approximations can reduce memory use.",
        isCorrect: true,
      },
      {
        text: "Low-rank approximations can reduce computation.",
        isCorrect: true,
      },
      {
        text: "Low-rank structure can model redundancy in large neural network weights.",
        isCorrect: true,
      },
    ],
    explanation:
      "Large models often contain redundant structure, so useful changes or approximations may live in fewer independent directions than a full matrix suggests. Low-rank methods exploit this by using fewer parameters and often less memory or computation. This is especially useful when the full weight matrices are very large.",
  },

  {
    id: "la-crash-l2-q20",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe Low-Rank Adaptation (LoRA) intuition?",
    options: [
      {
        text: "LoRA can update a large weight matrix using a low-rank change.",
        isCorrect: true,
      },
      {
        text: "The expression \\(W + AB\\) represents keeping a base matrix and adding a learned low-rank update.",
        isCorrect: true,
      },
      {
        text: "LoRA is efficient when only a small number of learned directions are needed.",
        isCorrect: true,
      },
      {
        text: "LoRA requires replacing the base model with a larger full-rank matrix update.",
        isCorrect: false,
      },
    ],
    explanation:
      "LoRA keeps the original weight matrix and learns a low-rank update, often written as \\(W+AB\\). The factors \\(A\\) and \\(B\\) capture a small number of update directions, which can make fine-tuning cheaper. The incorrect statement reverses the point by claiming LoRA requires a larger full-rank update.",
  },

  {
    id: "la-crash-l2-q21",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly distinguish low rank from small numerical values?",
    options: [
      {
        text: "Rank concerns independent directions represented by a matrix.",
        isCorrect: true,
      },
      {
        text: "A matrix can have large entries and still be low rank.",
        isCorrect: true,
      },
      {
        text: "A matrix is low rank exactly when all of its entries are close to zero.",
        isCorrect: false,
      },
      {
        text: "Multiplying every entry by a large constant necessarily makes the matrix full rank.",
        isCorrect: false,
      },
    ],
    explanation:
      "Rank depends on linear independence, not on the absolute size of entries. Scaling a matrix changes magnitudes but does not create new independent directions by itself. A low-rank matrix can contain large values if its rows or columns remain dependent.",
  },

  {
    id: "la-crash-l2-q22",
    chapter: 2,
    difficulty: "medium",
    prompt: "What does the transpose operation do to a matrix?",
    options: [
      {
        text: "It flips rows and columns.",
        isCorrect: true,
      },
      {
        text: "It squares every entry.",
        isCorrect: false,
      },
      {
        text: "It deletes all off-diagonal entries.",
        isCorrect: false,
      },
      {
        text: "It sorts each row from smallest to largest.",
        isCorrect: false,
      },
    ],
    explanation:
      "The transpose turns rows into columns and columns into rows. It changes the orientation and shape of a non-square matrix. It does not square entries, zero out off-diagonal values, or sort anything.",
  },

  {
    id: "la-crash-l2-q23",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "If \\(A \\in \\mathbb{R}^{3 \\times 5}\\), which statements about \\(A^T\\) are correct?",
    options: [
      {
        text: "\\(A^T \\in \\mathbb{R}^{5 \\times 3}\\).",
        isCorrect: true,
      },
      {
        text: "Rows of \\(A\\) become columns of \\(A^T\\).",
        isCorrect: true,
      },
      {
        text: "Columns of \\(A\\) become rows of \\(A^T\\).",
        isCorrect: true,
      },
      {
        text: "The transpose changes multiplication compatibility with other matrices.",
        isCorrect: true,
      },
    ],
    explanation:
      "A transpose swaps the shape, so a \\(3 \\times 5\\) matrix becomes a \\(5 \\times 3\\) matrix. Rows and columns exchange roles. Because shapes change, the set of valid matrix multiplications involving the transposed matrix can also change.",
  },

  {
    id: "la-crash-l2-q24",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about matrix multiplication as composition are correct?",
    options: [
      {
        text: "\\(ABx\\) can be interpreted as first applying \\(B\\) to \\(x\\), then applying \\(A\\).",
        isCorrect: true,
      },
      {
        text: "The product \\(AB\\) represents a composed transformation when the dimensions align.",
        isCorrect: true,
      },
      {
        text: "Changing the order of transformations can change the final result.",
        isCorrect: true,
      },
      {
        text: "\\(AB\\) and \\(BA\\) are guaranteed to be equal whenever both products are defined.",
        isCorrect: false,
      },
    ],
    explanation:
      "In \\(ABx\\), the vector is transformed by \\(B\\) first and then by \\(A\\), so matrix multiplication encodes composition. Order matters because different sequences of geometric transformations can produce different outcomes. Matrix multiplication is not generally commutative, so \\(AB\\) and \\(BA\\) are not guaranteed to match.",
  },

  {
    id: "la-crash-l2-q25",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Why does attention commonly use \\(QK^T\\) rather than \\(QK\\) when \\(Q\\) and \\(K\\) store query and key vectors as rows?",
    options: [
      {
        text: "The transpose aligns dimensions so each query can be dotted with each key.",
        isCorrect: true,
      },
      {
        text: "The result can contain pairwise query-key similarity scores.",
        isCorrect: true,
      },
      {
        text: "The transpose is used because dot products require every key vector to be discarded.",
        isCorrect: false,
      },
      {
        text: "The transpose makes attention independent of vector alignment.",
        isCorrect: false,
      },
    ],
    explanation:
      "When query and key vectors are stored as rows, transposing \\(K\\) makes its key vectors available as columns for dot products. The product \\(QK^T\\) can then hold all query-key similarity scores. The transpose does not discard keys or remove the importance of vector alignment; it enables the alignment calculation.",
  },

  {
    id: "la-crash-l2-q26",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which condition defines a symmetric matrix?",
    options: [
      { text: "\\(A = A^T\\).", isCorrect: true },
      { text: "\\(A = -A^T\\).", isCorrect: false },
      { text: "\\(A\\) has more rows than columns.", isCorrect: false },
      { text: "Every entry of \\(A\\) is positive.", isCorrect: false },
    ],
    explanation:
      "A matrix is symmetric when it equals its transpose, which means entries mirror across the main diagonal. The condition \\(A=-A^T\\) describes skew-symmetry, not symmetry. Rectangular shape or positive entries alone do not define symmetry.",
  },

  {
    id: "la-crash-l2-q27",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the covariance-style matrix \\(\\Sigma = X^T X\\)?",
    options: [
      {
        text: "It is symmetric.",
        isCorrect: true,
      },
      {
        text: "It captures feature relationships through dot products between feature columns.",
        isCorrect: true,
      },
      {
        text: "It is important for Principal Component Analysis (PCA) intuition.",
        isCorrect: true,
      },
      {
        text: "It appears naturally in statistics and representation learning.",
        isCorrect: true,
      },
    ],
    explanation:
      "The matrix \\(X^T X\\) is symmetric because its transpose is also \\(X^T X\\). Its entries are dot products between feature columns, which capture how features vary together. This is why covariance-style matrices are central to statistics, PCA, and representation learning.",
  },

  {
    id: "la-crash-l2-q28",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements give correct high-level intuition for why transposes appear in backpropagation?",
    options: [
      {
        text: "Backward propagation moves gradient information through a layer in the reverse direction.",
        isCorrect: true,
      },
      {
        text: "For a linear map, the transpose is the matrix that naturally sends output-side sensitivities back toward input-side sensitivities.",
        isCorrect: true,
      },
      {
        text: "A transpose changes multiplication compatibility so the gradient shapes can align.",
        isCorrect: true,
      },
      {
        text: "Transposes appear because gradients must ignore the original forward transformation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Backpropagation sends sensitivity information backward through transformations that were applied forward. For a linear map, the transpose is the natural object that moves output-side gradients back to the input side while respecting shapes. The original forward transformation is not ignored; its structure determines how gradients flow.",
  },

  {
    id: "la-crash-l2-q29",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly explain why matrix multiplication order matters?",
    options: [
      {
        text: "Rotating then stretching can produce a different result from stretching then rotating.",
        isCorrect: true,
      },
      {
        text: "Sequential neural network layers depend on the order in which transformations are applied.",
        isCorrect: true,
      },
      {
        text: "The equality \\(AB = BA\\) holds for every pair of square matrices.",
        isCorrect: false,
      },
      {
        text: "The product \\(AB\\) applies \\(A\\) to the input vector before \\(B\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Matrix multiplication represents ordered composition, so changing the order can change the geometry. A stretch followed by a rotation is generally not the same as a rotation followed by a stretch, and neural networks also rely on ordered layers. The product \\(ABx\\) applies \\(B\\) before \\(A\\), and \\(AB=BA\\) is not generally true.",
  },

  {
    id: "la-crash-l2-q30",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A layer computes \\(y = Wx + b\\). Which statement is most accurate?",
    options: [
      {
        text: "\\(W\\) performs a learned linear transformation, and \\(b\\) shifts the result.",
        isCorrect: true,
      },
      {
        text: "\\(b\\) determines the input dimension, while \\(W\\) only stores labels.",
        isCorrect: false,
      },
      {
        text: "The expression has no geometric interpretation.",
        isCorrect: false,
      },
      {
        text: "The expression is nonlinear even when no activation function is used.",
        isCorrect: false,
      },
    ],
    explanation:
      "The matrix \\(W\\) transforms the input features, and the bias vector \\(b\\) shifts the transformed result. This gives a clear geometric interpretation as an affine transformation. Without an activation function, adding a bias does not create the kind of nonlinear composition that deep networks rely on for expressive power.",
  },

  {
    id: "la-crash-l2-q31",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements connect matrix columns and basis-vector geometry correctly?",
    options: [
      {
        text: "A matrix is determined by where it sends the standard basis vectors.",
        isCorrect: true,
      },
      {
        text: "The columns of a matrix can be interpreted as transformed basis directions.",
        isCorrect: true,
      },
      {
        text: "Thinking about transformed basis vectors helps visualize how the whole space moves.",
        isCorrect: true,
      },
      {
        text: "A vector input can be transformed by combining the transformed basis directions using the vector's coordinates.",
        isCorrect: true,
      },
    ],
    explanation:
      "A linear transformation is fully determined by what it does to the standard basis vectors. The columns of the matrix are exactly those transformed basis directions under the usual column-vector convention. Any input vector is a coordinate-weighted combination of basis vectors, so its image is the same combination of the transformed basis directions.",
  },

  {
    id: "la-crash-l2-q32",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly connect rank to expressivity in learned transformations?",
    options: [
      {
        text: "A low-rank matrix can create an information bottleneck.",
        isCorrect: true,
      },
      {
        text: "A full-rank matrix can preserve more independent directions than a low-rank matrix of the same shape.",
        isCorrect: true,
      },
      {
        text: "Rank helps describe how many independent directions a weight matrix can use.",
        isCorrect: true,
      },
      {
        text: "Low rank means the transformation is nonlinear.",
        isCorrect: false,
      },
    ],
    explanation:
      "Rank describes independent linear directions available to the transformation. Low rank can force information through fewer directions, creating a bottleneck, while full rank can preserve more independent directions when the shape permits it. Low rank is still a property of a linear transformation, not a sign of nonlinearity.",
  },

  {
    id: "la-crash-l2-q33",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A map sends vectors from \\(\\mathbb{R}^{512}\\) to \\(\\mathbb{R}^{64}\\). Which statements are correct?",
    options: [
      {
        text: "The output has fewer coordinates than the input.",
        isCorrect: true,
      },
      {
        text: "The map can act as a bottleneck that compresses information.",
        isCorrect: true,
      },
      {
        text: "The map must be invertible for every possible input.",
        isCorrect: false,
      },
      {
        text: "The map increases the maximum possible number of independent output directions beyond 512.",
        isCorrect: false,
      },
    ],
    explanation:
      "Mapping from 512 dimensions to 64 dimensions reduces the number of output coordinates. Such a transformation can act as a compression bottleneck because many input distinctions may not survive. It cannot be invertible for all inputs, and it cannot create more independent output directions than the smaller output space allows.",
  },

  {
    id: "la-crash-l2-q34",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statement identifies a shape error for \\(A \\in \\mathbb{R}^{4 \\times 3}\\), \\(B \\in \\mathbb{R}^{5 \\times 4}\\), and \\(x \\in \\mathbb{R}^{3}\\)?",
    options: [
      {
        text: "\\(ABx\\) is not defined because \\(AB\\) is not defined.",
        isCorrect: true,
      },
      {
        text: "\\(Ax\\) is not defined because 4 and 3 are different.",
        isCorrect: false,
      },
      {
        text: "\\(BAx\\) is not defined because \\(Ax\\) has dimension 4.",
        isCorrect: false,
      },
      {
        text: "\\(Bx\\) is defined because \\(B\\) has 5 rows.",
        isCorrect: false,
      },
    ],
    explanation:
      "The product \\(AB\\) would require the 3 columns of \\(A\\) to match the 5 rows of \\(B\\), so \\(AB\\) is not defined. The product \\(Ax\\) is defined and has dimension 4, which then makes \\(B(Ax)\\) defined because \\(B\\) expects 4 inputs. The product \\(Bx\\) is not defined because \\(B\\) expects a 4-dimensional vector, not a 3-dimensional one.",
  },

  {
    id: "la-crash-l2-q35",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "For a low-rank factorization \\(AB\\) with \\(A \\in \\mathbb{R}^{m \\times r}\\) and \\(B \\in \\mathbb{R}^{r \\times n}\\), which statements are correct?",
    options: [
      {
        text: "The product \\(AB\\) has shape \\(m \\times n\\).",
        isCorrect: true,
      },
      {
        text: "The number of stored parameters in the factors is \\(r(m+n)\\).",
        isCorrect: true,
      },
      {
        text: "The factorization can be parameter-efficient when \\(r\\) is much smaller than \\(m\\) and \\(n\\).",
        isCorrect: true,
      },
      {
        text: "The rank of \\(AB\\) is at most \\(r\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The inner \\(r\\) dimensions match, so \\(AB\\) has shape \\(m \\times n\\). The factors store \\(mr + rn = r(m+n)\\) parameters, which can be far fewer than \\(mn\\) when \\(r\\) is small. The product cannot have rank greater than the bottleneck dimension \\(r\\).",
  },

  {
    id: "la-crash-l2-q36",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "For a LoRA-style update \\(W + AB\\) where \\(W \\in \\mathbb{R}^{m \\times n}\\), which statements are correct?",
    options: [
      {
        text: "\\(A\\) can have shape \\(m \\times r\\) and \\(B\\) can have shape \\(r \\times n\\).",
        isCorrect: true,
      },
      {
        text: "\\(AB\\) has the same shape as \\(W\\), so the addition is shape-compatible.",
        isCorrect: true,
      },
      {
        text: "A small \\(r\\) means the learned update uses a limited number of independent directions.",
        isCorrect: true,
      },
      {
        text: "The addition is shape-compatible only when \\(A\\) and \\(B\\) are both square matrices of shape \\(m \\times m\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "If \\(A\\) is \\(m \\times r\\) and \\(B\\) is \\(r \\times n\\), their product is \\(m \\times n\\), matching \\(W\\). A small rank parameter \\(r\\) constrains the update to relatively few independent directions. The factors do not need to be square to make the update shape-compatible.",
  },

  {
    id: "la-crash-l2-q37",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statements correctly connect symmetry and dot products?",
    options: [
      {
        text: "The matrix \\(X^T X\\) is symmetric because feature-column dot products satisfy \\(x_i \\cdot x_j = x_j \\cdot x_i\\).",
        isCorrect: true,
      },
      {
        text: "Symmetry means mirrored entries across the main diagonal are equal.",
        isCorrect: true,
      },
      {
        text: "A symmetric matrix must be rectangular with unequal side lengths.",
        isCorrect: false,
      },
      {
        text: "Changing the order of the two vectors in a dot product changes the scalar value.",
        isCorrect: false,
      },
    ],
    explanation:
      "Dot products are symmetric for real vectors, so swapping the two vectors does not change the scalar result. That is why a Gram or covariance-style matrix such as \\(X^T X\\) mirrors across the diagonal. A symmetric matrix must be square, not rectangular with unequal side lengths.",
  },

  {
    id: "la-crash-l2-q38",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statement best explains why a 100-layer network with no activation functions behaves like a single linear layer?",
    options: [
      {
        text: "The product of all weight matrices can be combined into one matrix representing the same overall linear map.",
        isCorrect: true,
      },
      {
        text: "Every additional linear layer creates a new nonlinear bend in the input space.",
        isCorrect: false,
      },
      {
        text: "The network stops using matrix multiplication after the first layer.",
        isCorrect: false,
      },
      {
        text: "A deep linear stack automatically computes attention scores.",
        isCorrect: false,
      },
    ],
    explanation:
      "Composing linear maps produces another linear map, so all the weight matrices can be multiplied into one equivalent matrix when shapes align. Additional linear layers may change the matrix product, but they do not introduce nonlinear bends. Attention scores and nonlinear expressivity require additional structure beyond a plain stack of linear maps.",
  },

  {
    id: "la-crash-l2-q39",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements are correct about rotating an embedding space with a distance-preserving rotation?",
    options: [
      {
        text: "The coordinates of individual embeddings may change.",
        isCorrect: true,
      },
      {
        text: "Pairwise distances can remain unchanged.",
        isCorrect: true,
      },
      {
        text: "The semantic neighborhood structure can be preserved when relationships depend on distances or angles.",
        isCorrect: true,
      },
      {
        text: "Every rotation necessarily destroys all similarity relationships.",
        isCorrect: false,
      },
    ],
    explanation:
      "A rotation can change the coordinate values used to describe each vector while preserving geometric relationships such as distances. If semantic neighborhoods are encoded through those geometric relationships, the structure can remain intact after a distance-preserving rotation. The incorrect statement is too strong because rotations do not automatically destroy similarity.",
  },

  {
    id: "la-crash-l2-q40",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statements correctly connect matrices to modern AI systems?",
    options: [
      {
        text: "Transformers use matrix multiplication for projections, attention scores, and feed-forward layers.",
        isCorrect: true,
      },
      {
        text: "Embeddings, LoRA updates, and optimization all rely on linear-algebraic structure.",
        isCorrect: true,
      },
      {
        text: "Matrix shapes are irrelevant once a model has many layers.",
        isCorrect: false,
      },
      {
        text: "Rank is unrelated to compression or parameter-efficient fine-tuning.",
        isCorrect: false,
      },
    ],
    explanation:
      "Modern AI systems use matrices for learned projections, attention computations, and feature transformations. Embeddings live in vector spaces, LoRA uses low-rank matrix updates, and optimization moves through parameter spaces shaped by linear algebra. Shapes and rank remain important because they control compatibility, capacity, compression, and efficiency.",
  },
];

export const CrashCourseLinearAlgebraL2Questions =
  CrashCourseLinearAlgebraLecture2Questions;
