import { Question } from "../../quiz";

export const CrashCourseLinearAlgebraLecture4Questions: Question[] = [
  {
    id: "la-crash-l4-q01",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements correctly describe eigenvectors and eigenvalues?",
    options: [
      {
        text: "An eigenvector keeps its direction under a matrix transformation.",
        isCorrect: true,
      },
      {
        text: "An eigenvalue describes the scaling applied along an eigenvector direction.",
        isCorrect: true,
      },
      {
        text: "The relationship can be written \\(Av=\\lambda v\\).",
        isCorrect: true,
      },
      {
        text: "Eigenvectors reveal special directions of a transformation.",
        isCorrect: true,
      },
    ],
    explanation:
      "An eigenvector is a direction that a matrix does not rotate away from; the matrix only scales it. The eigenvalue tells how strong that scaling is, which is why eigenvectors are useful for understanding stable directions in transformations.",
  },
  {
    id: "la-crash-l4-q02",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements about \\(Av=\\lambda v\\) are correct?",
    options: [
      { text: "\\(A\\) is the matrix transformation.", isCorrect: true },
      {
        text: "\\(v\\) is an eigenvector when the equation holds for nonzero \\(v\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\lambda\\) is the eigenvalue for that eigenvector.",
        isCorrect: true,
      },
      {
        text: "The equation means \\(v\\) must become perpendicular to itself after the transformation.",
        isCorrect: false,
      },
    ],
    explanation:
      "The equation says that applying \\(A\\) to \\(v\\) gives a scaled version of the same direction. It does not say that the vector becomes perpendicular; that would mean the direction has changed rather than being preserved up to scale.",
  },
  {
    id: "la-crash-l4-q03",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements correctly interpret eigenvalue size and sign?",
    options: [
      {
        text: "An eigenvalue greater than 1 expands that eigenvector direction.",
        isCorrect: true,
      },
      {
        text: "An eigenvalue between 0 and 1 shrinks that eigenvector direction.",
        isCorrect: true,
      },
      {
        text: "A negative eigenvalue keeps orientation while changing length within the same line.",
        isCorrect: false,
      },
      {
        text: "Eigenvalues in this setting are fixed at 1 by definition.",
        isCorrect: false,
      },
    ],
    explanation:
      "The eigenvalue is the scaling factor along the eigenvector direction. Values above 1 expand, values between 0 and 1 shrink, and negative values can flip orientation, so saying a negative eigenvalue can never flip direction is incorrect.",
  },
  {
    id: "la-crash-l4-q04",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statement best describes what makes eigenvectors geometrically special?",
    options: [
      {
        text: "They are directions preserved by the transformation up to scaling.",
        isCorrect: true,
      },
      {
        text: "They are directions that must disappear after transformation.",
        isCorrect: false,
      },
      {
        text: "They are random directions chosen independently of the matrix rather than tied to its geometry.",
        isCorrect: false,
      },
      {
        text: "They are vectors outside the matrix's domain.",
        isCorrect: false,
      },
    ],
    explanation:
      "Eigenvectors are special because the transformation does not rotate them into a new direction. The matrix may stretch, shrink, or flip them, but the line of direction remains the same.",
  },
  {
    id: "la-crash-l4-q05",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe why eigenvectors matter in AI and data analysis?",
    options: [
      {
        text: "They can reveal dominant directions in data or transformations.",
        isCorrect: true,
      },
      {
        text: "They help explain principal components in Principal Component Analysis (PCA).",
        isCorrect: true,
      },
      {
        text: "They can expose stable modes of variation.",
        isCorrect: true,
      },
      {
        text: "They support geometric reasoning about representation spaces.",
        isCorrect: true,
      },
    ],
    explanation:
      "Eigenvectors reveal directions that have special behavior under a transformation, and data-analysis methods use this to find important structure. PCA is a key example because it uses eigenvectors of a covariance matrix to identify major directions of variation.",
  },
  {
    id: "la-crash-l4-q06",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements correctly describe power-iteration intuition?",
    options: [
      {
        text: "Repeatedly applying a matrix can amplify the direction associated with a large eigenvalue.",
        isCorrect: true,
      },
      {
        text: "A vector can increasingly align with a dominant eigenvector after repeated transformations.",
        isCorrect: true,
      },
      {
        text: "Dominant structure can emerge from repeated matrix application.",
        isCorrect: true,
      },
      {
        text: "Power iteration treats the starting direction as equally amplified at each step.",
        isCorrect: false,
      },
    ],
    explanation:
      "If one eigenvalue dominates, repeated applications of the matrix tend to emphasize that direction relative to weaker ones. This is why power-iteration intuition is useful for thinking about dominant latent directions.",
  },
  {
    id: "la-crash-l4-q07",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements correctly describe covariance?",
    options: [
      {
        text: "Positive covariance means two variables tend to increase together.",
        isCorrect: true,
      },
      {
        text: "Negative covariance means one variable tends to increase while the other decreases.",
        isCorrect: true,
      },
      {
        text: "Near-zero covariance rules out a curved relationship between variables.",
        isCorrect: false,
      },
      {
        text: "Covariance measures the row count rather than joint variation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Covariance describes how variables vary together linearly, including whether they move in the same or opposite directions. Near-zero covariance does not rule out every possible nonlinear relationship, and covariance is not simply about counting examples.",
  },
  {
    id: "la-crash-l4-q08",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statement best describes a covariance matrix?",
    options: [
      {
        text: "It summarizes pairwise variation and co-variation among features.",
        isCorrect: true,
      },
      {
        text: "It stores class labels instead of pairwise feature variation inside the covariance table.",
        isCorrect: false,
      },
      {
        text: "It is unrelated to feature correlations.",
        isCorrect: false,
      },
      {
        text: "It is restricted to one-dimensional data.",
        isCorrect: false,
      },
    ],
    explanation:
      "A covariance matrix records how each feature varies with itself and with other features. This makes it useful for identifying the dominant directions in a multidimensional data cloud.",
  },
  {
    id: "la-crash-l4-q09",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe symmetric covariance matrices?",
    options: [
      { text: "A covariance matrix is symmetric.", isCorrect: true },
      {
        text: "Symmetry means \\(\\Sigma=\\Sigma^T\\).",
        isCorrect: true,
      },
      {
        text: "Symmetric matrices have useful eigenvector structure.",
        isCorrect: true,
      },
      {
        text: "This structure helps make PCA possible.",
        isCorrect: true,
      },
    ],
    explanation:
      "Covariance between feature A and feature B is the same relationship as covariance between feature B and feature A, so the matrix is symmetric. This symmetry gives covariance matrices especially useful eigenvector structure for PCA.",
  },
  {
    id: "la-crash-l4-q10",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe Principal Component Analysis (PCA)?",
    options: [
      {
        text: "PCA finds directions of high variance in data.",
        isCorrect: true,
      },
      {
        text: "Its directions are called principal components.",
        isCorrect: true,
      },
      {
        text: "It can support dimensionality reduction.",
        isCorrect: true,
      },
      {
        text: "It is designed to preserve directions with the least variance first.",
        isCorrect: false,
      },
    ],
    explanation:
      "PCA looks for directions where the data varies the most, because those directions often preserve important structure. It does not start by prioritizing the least-variance directions, which are more likely to contain weak signal or noise.",
  },
  {
    id: "la-crash-l4-q11",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe dimensionality reduction with PCA?",
    options: [
      {
        text: "Data can be projected onto a smaller number of principal directions.",
        isCorrect: true,
      },
      {
        text: "If most variation lies in a few directions, compression can preserve much of the structure.",
        isCorrect: true,
      },
      {
        text: "Dimensionality reduction preserves original coordinates one by one.",
        isCorrect: false,
      },
      {
        text: "PCA requires adding random dimensions to compress data.",
        isCorrect: false,
      },
    ],
    explanation:
      "PCA compression works by keeping important directions and discarding weaker ones. It does not preserve every original coordinate exactly, because the point is to represent the data in a lower-dimensional subspace.",
  },
  {
    id: "la-crash-l4-q12",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statement best connects PCA to eigenvectors?",
    options: [
      {
        text: "Principal components are eigenvectors of the covariance matrix.",
        isCorrect: true,
      },
      {
        text: "PCA ignores covariance entirely.",
        isCorrect: false,
      },
      {
        text: "Principal components are chosen by alphabetical feature names in the input table.",
        isCorrect: false,
      },
      {
        text: "Eigenvectors prevent dimensionality reduction.",
        isCorrect: false,
      },
    ],
    explanation:
      "PCA uses the eigenvectors of the covariance matrix to identify stable directions of variation. This is why eigenvectors, covariance, and dimensionality reduction are tightly connected.",
  },
  {
    id: "la-crash-l4-q13",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe embeddings and lower-dimensional structure?",
    options: [
      {
        text: "Embeddings can contain redundant or correlated dimensions.",
        isCorrect: true,
      },
      {
        text: "Semantic structure can cluster geometrically in embedding spaces.",
        isCorrect: true,
      },
      {
        text: "Important information can lie near a lower-dimensional manifold.",
        isCorrect: true,
      },
      {
        text: "Compression can be useful when representations contain structured redundancy.",
        isCorrect: true,
      },
    ],
    explanation:
      "Embedding spaces often organize meaning geometrically rather than using fully independent coordinates. When structure is redundant or concentrated in fewer directions, compression can preserve much of the useful information.",
  },
  {
    id: "la-crash-l4-q14",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe Singular Value Decomposition (SVD)?",
    options: [
      {
        text: "It can be written \\(A=U\\Sigma V^T\\).",
        isCorrect: true,
      },
      {
        text: "It decomposes a matrix transformation into rotations/reflections and scaling.",
        isCorrect: true,
      },
      {
        text: "It applies to rectangular matrices as well as square matrices.",
        isCorrect: true,
      },
      {
        text: "It works best when matrix entries are zero across the dataset.",
        isCorrect: false,
      },
    ],
    explanation:
      "SVD factors a matrix into components that can be understood geometrically as coordinate alignment, directional scaling, and output-space rotation or reflection. It is broader than eigendecomposition because it applies to rectangular matrices too.",
  },
  {
    id: "la-crash-l4-q15",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly interpret the factors in \\(A=U\\Sigma V^T\\)?",
    options: [
      {
        text: "\\(V^T\\) can be viewed as aligning input coordinates with important directions.",
        isCorrect: true,
      },
      {
        text: "\\(\\Sigma\\) scales directions by singular values.",
        isCorrect: true,
      },
      {
        text: "\\(U\\) discards output-space geometry from the decomposition.",
        isCorrect: false,
      },
      {
        text: "\\(\\Sigma\\) stores labels rather than numerical scales.",
        isCorrect: false,
      },
    ],
    explanation:
      "The geometric view of SVD is rotate or align, scale, then rotate or map again. The singular values in \\(\\Sigma\\) are numerical strengths for the directions, and \\(U\\) contributes output-space structure rather than deleting it.",
  },
  {
    id: "la-crash-l4-q16",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statement best describes singular values?",
    options: [
      {
        text: "They measure the strength or importance of directions in the SVD transformation.",
        isCorrect: true,
      },
      {
        text: "They are usually measurement noise rather than useful signal.",
        isCorrect: false,
      },
      {
        text: "They should be equal for SVD to work well.",
        isCorrect: false,
      },
      {
        text: "They are the same thing as row names in a data table.",
        isCorrect: false,
      },
    ],
    explanation:
      "Singular values quantify how much the transformation scales particular directions. Large singular values often correspond to stronger signal or more important structure, while small ones can represent weaker directions or redundancy.",
  },
  {
    id: "la-crash-l4-q17",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements correctly describe low-rank approximation?",
    options: [
      {
        text: "It keeps only the largest singular values and associated directions.",
        isCorrect: true,
      },
      {
        text: "It can compress a matrix while preserving major structure.",
        isCorrect: true,
      },
      {
        text: "It uses the fact that real-world data often contains redundancy.",
        isCorrect: true,
      },
      {
        text: "It can be written conceptually as \\(A\\approx A_k\\) for a lower-rank matrix \\(A_k\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "Low-rank approximation keeps the strongest directions and drops weaker ones. This works well when data or transformations contain redundant structure, allowing a smaller representation to preserve much of the important information.",
  },
  {
    id: "la-crash-l4-q18",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements correctly connect low rank to modern AI?",
    options: [
      {
        text: "Attention matrices can contain redundant structure.",
        isCorrect: true,
      },
      {
        text: "Fine-tuning updates can sometimes be approximated with low-rank changes.",
        isCorrect: true,
      },
      {
        text: "Embedding matrices can sometimes be compressed with SVD-like ideas.",
        isCorrect: true,
      },
      {
        text: "Low-rank structure means the model has little useful information despite its learned weights.",
        isCorrect: false,
      },
    ],
    explanation:
      "Low-rank structure often means information is concentrated in fewer directions, not that information is absent. This is why low-rank ideas appear in attention analysis, embedding compression, and parameter-efficient fine-tuning.",
  },
  {
    id: "la-crash-l4-q19",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements correctly describe Low-Rank Adaptation (LoRA)?",
    options: [
      {
        text: "LoRA assumes useful fine-tuning updates can be represented with low-rank matrices.",
        isCorrect: true,
      },
      {
        text: "LoRA can reduce the number of trainable parameters during adaptation.",
        isCorrect: true,
      },
      {
        text: "LoRA requires directly updating the dense model-weight matrix.",
        isCorrect: false,
      },
      {
        text: "LoRA is unrelated to matrix rank.",
        isCorrect: false,
      },
    ],
    explanation:
      "LoRA represents updates through lower-rank factors rather than directly training every full weight matrix entry. Its efficiency depends on the assumption that useful adaptation can be captured by a smaller set of directional changes.",
  },
  {
    id: "la-crash-l4-q20",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statement best explains why high-dimensional embeddings can often be compressed?",
    options: [
      {
        text: "Their useful semantic information may be concentrated in fewer structured directions.",
        isCorrect: true,
      },
      {
        text: "Embedding coordinates carry unrelated information with equal importance.",
        isCorrect: false,
      },
      {
        text: "Compression works by deleting labels rather than numerical structure.",
        isCorrect: false,
      },
      {
        text: "Embeddings are categorical tags rather than numerical vectors.",
        isCorrect: false,
      },
    ],
    explanation:
      "Embeddings are numerical vectors, and their dimensions can contain correlated or redundant information. If semantic structure is concentrated in fewer directions, methods such as PCA or SVD can preserve much of that structure with fewer dimensions.",
  },
  {
    id: "la-crash-l4-q21",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements correctly distinguish PCA and SVD?",
    options: [
      {
        text: "PCA focuses on directions of variance in data.",
        isCorrect: true,
      },
      {
        text: "SVD decomposes a matrix into directional scaling components.",
        isCorrect: true,
      },
      {
        text: "PCA can be computed through eigenvectors of a covariance matrix.",
        isCorrect: true,
      },
      {
        text: "SVD is useful for understanding low-rank approximation.",
        isCorrect: true,
      },
    ],
    explanation:
      "PCA and SVD are closely related but emphasize different views: PCA is about variance directions in data, while SVD is a general matrix factorization. Both support geometric reasoning about dominant directions and compression.",
  },
  {
    id: "la-crash-l4-q22",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe why covariance is useful for PCA?",
    options: [
      {
        text: "It captures how data spreads across feature directions.",
        isCorrect: true,
      },
      {
        text: "Its eigenvectors can identify principal directions.",
        isCorrect: true,
      },
      {
        text: "Its large-variance directions can preserve important structure.",
        isCorrect: true,
      },
      {
        text: "It intentionally removes feature relationships before analysis.",
        isCorrect: false,
      },
    ],
    explanation:
      "The covariance matrix summarizes relationships among features and how the data cloud spreads. PCA uses that structure to identify the directions where variation is largest.",
  },
  {
    id: "la-crash-l4-q23",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe semantic directions in embeddings?",
    options: [
      {
        text: "Certain directions can correspond to interpretable changes in meaning.",
        isCorrect: true,
      },
      {
        text: "Clusters can reflect semantic similarity.",
        isCorrect: true,
      },
      {
        text: "Each coordinate maps to a simple human-readable concept.",
        isCorrect: false,
      },
      {
        text: "Embedding geometry is unrelated to similarity search.",
        isCorrect: false,
      },
    ],
    explanation:
      "Embeddings can organize related concepts near each other and sometimes contain interpretable directions. That does not mean each raw coordinate has a clean human label, and similarity search relies directly on the geometry.",
  },
  {
    id: "la-crash-l4-q24",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statement best explains why PCA is not simply feature selection?",
    options: [
      {
        text: "PCA finds new directions that can combine multiple original features.",
        isCorrect: true,
      },
      {
        text: "PCA chooses existing columns by name.",
        isCorrect: false,
      },
      {
        text: "PCA changes labels without reducing dimensionality.",
        isCorrect: false,
      },
      {
        text: "PCA ignores variance.",
        isCorrect: false,
      },
    ],
    explanation:
      "PCA creates principal directions that are usually linear combinations of the original features. This differs from simply choosing a subset of existing columns.",
  },
  {
    id: "la-crash-l4-q25",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements correctly describe signal and redundancy in SVD?",
    options: [
      {
        text: "Large singular values often indicate strong directions in the matrix.",
        isCorrect: true,
      },
      {
        text: "Small singular values can correspond to weak signal, redundancy, or noise.",
        isCorrect: true,
      },
      {
        text: "Dropping small singular values can sometimes preserve the main structure.",
        isCorrect: true,
      },
      {
        text: "Not every small singular value must be kept to preserve the main pattern.",
        isCorrect: true,
      },
    ],
    explanation:
      "Singular values order directions by strength, so large values usually carry more of the transformation's energy. In low-rank approximation, smaller singular values are often discarded because they contribute less to the major structure.",
  },
  {
    id: "la-crash-l4-q26",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements correctly describe representation learning?",
    options: [
      {
        text: "Models can learn latent spaces that organize useful structure.",
        isCorrect: true,
      },
      {
        text: "Learned representations can compress relevant information.",
        isCorrect: true,
      },
      {
        text: "Geometry can help explain why representations support downstream tasks.",
        isCorrect: true,
      },
      {
        text: "Representation learning separates latent dimensions into independent factors by default.",
        isCorrect: false,
      },
    ],
    explanation:
      "Representation learning builds internal vector spaces where useful information is arranged for later computation. The dimensions need not be independent, and correlations or low-dimensional structure are often part of why compression is possible.",
  },
  {
    id: "la-crash-l4-q27",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe attention matrices and low rank?",
    options: [
      {
        text: "Attention patterns can contain redundancy.",
        isCorrect: true,
      },
      {
        text: "Information is spread uniformly rather than concentrating in major attention directions.",
        isCorrect: false,
      },
      {
        text: "Approximate low-rank structure can motivate compression or efficient computation.",
        isCorrect: true,
      },
      {
        text: "An attention matrix must be full rank to contain any useful pattern.",
        isCorrect: false,
      },
    ],
    explanation:
      "Attention matrices can have repeated or correlated structure, so their important behavior may be captured by fewer directions than their full size suggests. Information can concentrate in major directions, and full rank is not required for a matrix to encode useful relationships.",
  },
  {
    id: "la-crash-l4-q28",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statement best explains why the covariance matrix of an elongated data cloud has a meaningful principal direction?",
    options: [
      {
        text: "The data varies most along the long axis, and covariance captures that spread.",
        isCorrect: true,
      },
      {
        text: "The covariance matrix ignores the cloud's shape.",
        isCorrect: false,
      },
      {
        text: "The principal direction must be whichever axis has the shortest spread.",
        isCorrect: false,
      },
      {
        text: "PCA chooses a direction unrelated to the data.",
        isCorrect: false,
      },
    ],
    explanation:
      "An elongated cloud has one direction where points are spread out more strongly than others. Covariance captures that spread, and PCA identifies it as a principal direction.",
  },
  {
    id: "la-crash-l4-q29",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements correctly explain why low-rank approximation can be high quality rather than low quality?",
    options: [
      {
        text: "Many real datasets contain correlated structure.",
        isCorrect: true,
      },
      {
        text: "Important variation can concentrate in a small number of directions.",
        isCorrect: true,
      },
      {
        text: "Redundant dimensions can be removed with limited loss of useful information.",
        isCorrect: true,
      },
      {
        text: "Compression is effective when weak directions contribute little to the main pattern.",
        isCorrect: true,
      },
    ],
    explanation:
      "Low-rank does not automatically mean low quality; it means the representation uses fewer independent directions. When structure is redundant or concentrated, the largest directions can preserve the main signal well.",
  },
  {
    id: "la-crash-l4-q30",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements correctly connect SVD to model compression?",
    options: [
      {
        text: "A large matrix can be approximated by lower-rank factors.",
        isCorrect: true,
      },
      {
        text: "Keeping top singular values can preserve dominant behavior.",
        isCorrect: true,
      },
      {
        text: "Lower-rank factors can require fewer parameters or less computation.",
        isCorrect: true,
      },
      {
        text: "SVD compression increases the stored matrix dimensions.",
        isCorrect: false,
      },
    ],
    explanation:
      "SVD can express a matrix through ranked directional components, and keeping the strongest ones can approximate the original matrix. This can reduce storage or computation when the lower-rank approximation is accurate enough.",
  },
  {
    id: "la-crash-l4-q31",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe how PCA can help inspect embedding spaces?",
    options: [
      {
        text: "It can reveal major directions of variation.",
        isCorrect: true,
      },
      {
        text: "It can project high-dimensional embeddings into fewer dimensions for visualization or compression.",
        isCorrect: true,
      },
      {
        text: "A cluster in the plot is direct causal evidence for a biological or semantic category.",
        isCorrect: false,
      },
      {
        text: "It requires ignoring vector geometry during interpretation.",
        isCorrect: false,
      },
    ],
    explanation:
      "PCA is useful for summarizing and visualizing high-dimensional embeddings through their largest variance directions. It can reveal patterns, but interpretation still requires care because geometric clusters do not automatically prove a causal explanation.",
  },
  {
    id: "la-crash-l4-q32",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statement best explains why SVD is more general than an eigenvector-only view for AI matrices?",
    options: [
      {
        text: "SVD works for rectangular matrices, which are common in neural-network weight and embedding matrices.",
        isCorrect: true,
      },
      {
        text: "SVD is designed for diagonal covariance matrices rather than general matrices.",
        isCorrect: false,
      },
      {
        text: "SVD is disconnected from learned model weights.",
        isCorrect: false,
      },
      {
        text: "SVD removes geometric interpretation from the matrix.",
        isCorrect: false,
      },
    ],
    explanation:
      "Many neural-network matrices map between spaces of different sizes, so they are rectangular. SVD can still decompose these transformations geometrically, which makes it broadly useful for model analysis and compression.",
  },
  {
    id: "la-crash-l4-q33",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements correctly describe the geometric meaning of SVD?",
    options: [
      {
        text: "It can align input space with important directions.",
        isCorrect: true,
      },
      {
        text: "It can scale those directions by different strengths.",
        isCorrect: true,
      },
      {
        text: "It can rotate or map the scaled result into output space.",
        isCorrect: true,
      },
      {
        text: "It shows that any matrix transformation can be understood through structured direction changes.",
        isCorrect: true,
      },
    ],
    explanation:
      "The SVD view breaks a matrix into direction alignment, directional scaling, and output-space mapping. This makes abstract matrix behavior easier to understand geometrically.",
  },
  {
    id: "la-crash-l4-q34",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements correctly describe common misconceptions?",
    options: [
      {
        text: "Eigenvectors are not just abstract symbols; they are stable transformation directions.",
        isCorrect: true,
      },
      {
        text: "PCA finds directions of variation, not necessarily original named features.",
        isCorrect: true,
      },
      {
        text: "SVD reveals general geometric structure, not only compression.",
        isCorrect: true,
      },
      {
        text: "Low-rank structure indicates low-quality information rather than useful compression in model weights.",
        isCorrect: false,
      },
    ],
    explanation:
      "Eigenvectors, PCA, and SVD all have concrete geometric interpretations. Low rank can indicate efficient structure rather than poor quality, especially when the data contains redundancy.",
  },
  {
    id: "la-crash-l4-q35",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements correctly connect eigenvectors, SVD, and low-rank ideas to transformer internals?",
    options: [
      {
        text: "Embedding spaces can contain dominant semantic directions.",
        isCorrect: true,
      },
      {
        text: "Attention matrices are too irregular to show structured redundancy.",
        isCorrect: false,
      },
      {
        text: "Low-rank methods can support efficient adaptation or compression.",
        isCorrect: true,
      },
      {
        text: "Transformer behavior is separate from linear-algebra analysis.",
        isCorrect: false,
      },
    ],
    explanation:
      "Transformers rely on vector spaces, matrices, and learned representations, so linear-algebra tools are natural for analyzing them. Attention matrices can show structured redundancy, and dominant directions, low-rank structure, and embedding geometry all help explain how these systems organize information.",
  },
  {
    id: "la-crash-l4-q36",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statement best captures the central idea behind PCA, SVD, and embeddings?",
    options: [
      {
        text: "High-dimensional representations often contain structured directions that carry more information than others.",
        isCorrect: true,
      },
      {
        text: "Directions in high-dimensional data tend to carry equal importance.",
        isCorrect: false,
      },
      {
        text: "Numerical data resists compression because coordinates are already numbers.",
        isCorrect: false,
      },
      {
        text: "Representation learning avoids geometry.",
        isCorrect: false,
      },
    ],
    explanation:
      "PCA and SVD both exploit unequal importance across directions, and embeddings often show similar structure. High-dimensional data can be organized geometrically, with some directions carrying much more useful information than others.",
  },
  {
    id: "la-crash-l4-q37",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe how covariance relates to representation learning?",
    options: [
      {
        text: "Covariance can reveal correlated dimensions in learned representations.",
        isCorrect: true,
      },
      {
        text: "Covariance can help identify dominant directions of variation.",
        isCorrect: true,
      },
      {
        text: "Representation spaces may contain redundant or compressed structure.",
        isCorrect: true,
      },
      {
        text: "Studying covariance can support diagnostics of embedding geometry.",
        isCorrect: true,
      },
    ],
    explanation:
      "Learned representations are still data points in vector spaces, so covariance can summarize how their coordinates vary together. This can expose dominant directions, redundancy, and structure that matter for compression or interpretation.",
  },
  {
    id: "la-crash-l4-q38",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe when a low-rank update is plausible?",
    options: [
      {
        text: "The desired change is concentrated in a small number of directions.",
        isCorrect: true,
      },
      {
        text: "The full matrix update contains substantial redundancy.",
        isCorrect: true,
      },
      {
        text: "The adaptation task needs targeted directional changes rather than arbitrary independent changes to every entry.",
        isCorrect: true,
      },
      {
        text: "Fine-tuning tasks are expected to be rank one in this picture.",
        isCorrect: false,
      },
    ],
    explanation:
      "Low-rank updates are plausible when the useful change is structured and concentrated rather than fully independent in every parameter direction. That does not mean every task is rank one or that low-rank adaptation is perfect for every situation.",
  },
  {
    id: "la-crash-l4-q39",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements correctly explain why SVD appears in recommendation systems, retrieval, and embeddings?",
    options: [
      {
        text: "These systems often involve large matrices with hidden lower-dimensional structure.",
        isCorrect: true,
      },
      {
        text: "Dominant directions can capture major patterns of similarity or preference.",
        isCorrect: true,
      },
      {
        text: "Low-rank factors make storage and computation larger in typical recommender systems.",
        isCorrect: false,
      },
      {
        text: "SVD is useful mainly for matrices with no repeated patterns.",
        isCorrect: false,
      },
    ],
    explanation:
      "Recommendation, retrieval, and embedding systems often contain structured relationships that can be approximated with fewer latent factors. Low-rank factors can make storage and computation more efficient, so saying they always make systems larger is incorrect.",
  },
  {
    id: "la-crash-l4-q40",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statement best summarizes the practical importance of eigenvectors, PCA, and SVD for modern AI?",
    options: [
      {
        text: "They explain how high-dimensional systems can contain dominant directions, compressed structure, and meaningful representation geometry.",
        isCorrect: true,
      },
      {
        text: "They show that embeddings are unrelated to vector spaces.",
        isCorrect: false,
      },
      {
        text: "They suggest model adaptation should use dense full-rank updates.",
        isCorrect: false,
      },
      {
        text: "They eliminate the need to understand matrices.",
        isCorrect: false,
      },
    ],
    explanation:
      "Eigenvectors, PCA, and SVD provide tools for understanding important directions, compression, and latent structure in high-dimensional systems. These ideas directly support reasoning about embeddings, attention, LoRA, and representation learning.",
  },
  {
    id: "la-crash-l4-q41",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "For \\(A=\\begin{bmatrix}3 & 0 \\\\ 0 & 1\\end{bmatrix}\\), which statements correctly identify eigenvector behavior?",
    options: [
      {
        text: "\\(\\begin{bmatrix}1 \\\\ 0\\end{bmatrix}\\) is an eigenvector with eigenvalue \\(3\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\begin{bmatrix}0 \\\\ 1\\end{bmatrix}\\) is an eigenvector with eigenvalue \\(1\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\begin{bmatrix}2 \\\\ 0\\end{bmatrix}\\) is also an eigenvector with eigenvalue \\(3\\).",
        isCorrect: true,
      },
      {
        text: "Any nonzero vector in \\(\\mathbb{R}^2\\) is treated as an eigenvector of \\(A\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The coordinate axes are stable directions for this diagonal transformation, so vectors on the \\(x\\)-axis scale by \\(3\\) and vectors on the \\(y\\)-axis scale by \\(1\\). Any nonzero scalar multiple of an eigenvector is still an eigenvector with the same eigenvalue. A mixed vector such as \\((1,1)\\) changes direction because its two coordinates are scaled differently.",
  },
  {
    id: "la-crash-l4-q42",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "For \\(A=\\begin{bmatrix}2 & 0 \\\\ 0 & -1\\end{bmatrix}\\) and \\(v=\\begin{bmatrix}0 \\\\ 4\\end{bmatrix}\\), which statement is correct?",
    options: [
      {
        text: "\\(v\\) is an eigenvector with eigenvalue \\(-1\\).",
        isCorrect: true,
      },
      {
        text: "\\(v\\) is an eigenvector with eigenvalue \\(2\\).",
        isCorrect: false,
      },
      {
        text: "\\(v\\) is not an eigenvector because its second coordinate is nonzero.",
        isCorrect: false,
      },
      {
        text: "\\(Av\\) is treated as undefined because \\(A\\) has a negative entry.",
        isCorrect: false,
      },
    ],
    explanation:
      "Multiplying gives \\(Av=\\begin{bmatrix}0 \\\\ -4\\end{bmatrix}=-1v\\). The vector keeps its line of direction but flips because the eigenvalue is negative. A negative entry or eigenvalue does not prevent matrix multiplication.",
  },
  {
    id: "la-crash-l4-q43",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements correctly describe covariance intuition?",
    options: [
      {
        text: "Positive covariance means two variables tend to increase together.",
        isCorrect: true,
      },
      {
        text: "Negative covariance means one variable tends to decrease when the other increases.",
        isCorrect: true,
      },
      {
        text: "A covariance matrix is diagonal when feature interactions are being studied by default.",
        isCorrect: false,
      },
      {
        text: "Covariance ignores how data spreads through space.",
        isCorrect: false,
      },
    ],
    explanation:
      "Covariance measures how variables vary together around their means. Positive and negative covariance describe different kinds of joint movement, while off-diagonal covariance entries capture feature relationships. This is why covariance is useful for identifying directions of spread in data.",
  },
  {
    id: "la-crash-l4-q44",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "For a centered data matrix \\(X\\), which statements about the covariance-style matrix \\(X^TX\\) are correct?",
    options: [
      {
        text: "\\(X^TX\\) is symmetric.",
        isCorrect: true,
      },
      {
        text: "The entries of \\(X^TX\\) are feature-feature dot products.",
        isCorrect: true,
      },
      {
        text: "The diagonal entries of \\(X^TX\\) are sums of squared feature values.",
        isCorrect: true,
      },
      {
        text: "Multiplying by a positive scalar normalization factor changes eigenvalues but not eigenvector directions.",
        isCorrect: true,
      },
    ],
    explanation:
      "The matrix \\(X^TX\\) compares columns of \\(X\\), so it captures feature relationships. It is symmetric because transposing the product gives the same product, and its diagonal contains squared norms of feature columns. A positive scalar normalization rescales eigenvalues but leaves the eigendirections unchanged.",
  },
  {
    id: "la-crash-l4-q45",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe projecting two-dimensional data onto its first principal component?",
    options: [
      {
        text: "The first principal component is the direction of maximum variance.",
        isCorrect: true,
      },
      {
        text: "The projection discards variation orthogonal to that component.",
        isCorrect: true,
      },
      {
        text: "For centered data, the first principal component is an eigenvector of the covariance matrix.",
        isCorrect: true,
      },
      {
        text: "The first principal component must be one of the original coordinate axes.",
        isCorrect: false,
      },
    ],
    explanation:
      "Principal Component Analysis (PCA) chooses directions that capture as much variance as possible. Projecting onto the first component keeps the strongest one-dimensional direction and discards the orthogonal residual information. The component can be a rotated linear combination of the original features, not necessarily a coordinate axis.",
  },
  {
    id: "la-crash-l4-q46",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statement correctly describes Singular Value Decomposition (SVD)?",
    options: [
      {
        text: "SVD can decompose rectangular as well as square matrices.",
        isCorrect: true,
      },
      {
        text: "SVD is restricted to square symmetric matrices.",
        isCorrect: false,
      },
      {
        text: "Singular values are allowed to be negative because eigenvalues can be negative.",
        isCorrect: false,
      },
      {
        text: "SVD is unrelated to rotations, scaling, or low-rank approximation.",
        isCorrect: false,
      },
    ],
    explanation:
      "SVD applies broadly to matrices, including rectangular matrices that do not have ordinary eigenvectors in the same way square matrices do. Its singular values are nonnegative and describe scaling strengths. The decomposition is central to geometric interpretation and low-rank approximation.",
  },
  {
    id: "la-crash-l4-q47",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements correctly interpret singular values?",
    options: [
      {
        text: "Large singular values correspond to strong transformation directions.",
        isCorrect: true,
      },
      {
        text: "Zero singular values indicate missing independent directions and reduce rank.",
        isCorrect: true,
      },
      {
        text: "Singular values are negative when a direction flips.",
        isCorrect: false,
      },
      {
        text: "Singular values are simply the original rows of the data matrix.",
        isCorrect: false,
      },
    ],
    explanation:
      "Singular values measure how strongly a matrix scales special orthogonal directions. Zero singular values indicate that some directions are collapsed, which lowers rank. Direction flips are represented in singular vectors or eigenvalues in other decompositions, while singular values themselves remain nonnegative.",
  },
  {
    id: "la-crash-l4-q48",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "For the covariance matrix \\(\\Sigma=\\begin{bmatrix}4 & 0 \\\\ 0 & 1\\end{bmatrix}\\), which PCA statements are correct?",
    options: [
      {
        text: "The first principal component is the \\(x\\)-axis direction.",
        isCorrect: true,
      },
      {
        text: "The first component explains \\(\\frac{4}{5}\\) of the total variance.",
        isCorrect: true,
      },
      {
        text: "Projecting onto the first component keeps the higher-variance direction.",
        isCorrect: true,
      },
      {
        text: "\\(\\Sigma\\) is symmetric, so its principal directions are well behaved.",
        isCorrect: true,
      },
    ],
    explanation:
      "The diagonal entries show variance \\(4\\) along the \\(x\\)-axis and variance \\(1\\) along the \\(y\\)-axis. Total variance is \\(5\\), so the first component explains \\(4/5\\) of it. This simple example shows how PCA chooses high-variance directions.",
  },
  {
    id: "la-crash-l4-q49",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "A matrix has singular values \\(10,3,0,0\\). Which statements are correct?",
    options: [
      {
        text: "The rank of the matrix is \\(2\\).",
        isCorrect: true,
      },
      {
        text: "A rank-\\(2\\) SVD approximation can represent the matrix exactly.",
        isCorrect: true,
      },
      {
        text: "A rank-\\(1\\) approximation keeps the two displayed nonzero singular directions.",
        isCorrect: false,
      },
      {
        text: "The matrix must have rank \\(4\\) because four singular values are listed.",
        isCorrect: false,
      },
    ],
    explanation:
      "Rank equals the number of nonzero singular values, so this matrix has rank \\(2\\). Keeping the two nonzero singular directions reconstructs the matrix exactly, while keeping only one loses the direction with singular value \\(3\\). Listing zeros does not make those directions independent.",
  },
  {
    id: "la-crash-l4-q50",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "For \\(A=\\begin{bmatrix}5 & 0 \\\\ 0 & 2\\end{bmatrix}\\), power iteration starts from a vector with a nonzero first coordinate. Which statement best describes the long-run direction?",
    options: [
      {
        text: "The iterates tend to align with the first coordinate direction because eigenvalue \\(5\\) dominates eigenvalue \\(2\\).",
        isCorrect: true,
      },
      {
        text: "The iterates tend to align with the second coordinate direction because \\(2\\) is smaller.",
        isCorrect: false,
      },
      {
        text: "The iterates have no preferred direction because diagonal matrices lack eigenvectors.",
        isCorrect: false,
      },
      {
        text: "The iterates rotate by 90 degrees at each step.",
        isCorrect: false,
      },
    ],
    explanation:
      "Repeated multiplication scales the first coordinate by powers of \\(5\\) and the second coordinate by powers of \\(2\\). If the first coordinate is initially nonzero, the \\(5^k\\) term eventually dominates. This is the basic intuition behind power iteration.",
  },
  {
    id: "la-crash-l4-q51",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe best low-rank approximation with SVD?",
    options: [
      {
        text: "Keeping the largest singular values preserves the strongest transformation directions.",
        isCorrect: true,
      },
      {
        text: "The approximation error is controlled by the singular values that are discarded.",
        isCorrect: true,
      },
      {
        text: "Compression works best when singular values decay quickly.",
        isCorrect: true,
      },
      {
        text: "Discarding the largest singular values is the standard way to preserve signal.",
        isCorrect: false,
      },
    ],
    explanation:
      "SVD orders directions by their scaling strength, so keeping the largest singular values keeps the highest-energy structure. When the remaining singular values are small, the discarded information has relatively low impact. Throwing away the largest singular values would remove the strongest signal first.",
  },
  {
    id: "la-crash-l4-q52",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly connect low-rank adaptation to SVD and representation learning?",
    options: [
      {
        text: "A product \\(AB\\) with inner dimension \\(r\\) has rank at most \\(r\\).",
        isCorrect: true,
      },
      {
        text: "Low-rank adaptation assumes useful parameter changes may lie in a small set of directions.",
        isCorrect: true,
      },
      {
        text: "SVD can help analyze which directions in a matrix carry the strongest signal.",
        isCorrect: true,
      },
      {
        text: "Representation learning often benefits when high-dimensional data has lower-dimensional structure.",
        isCorrect: true,
      },
    ],
    explanation:
      "A low-rank factorization restricts an update to a limited directional subspace. SVD provides a mathematical language for identifying strong and weak directions in matrices. Representation learning is effective partly because real data often has structure that can be captured with fewer directions than the ambient dimension.",
  },
  {
    id: "la-crash-l4-q53",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which statements correctly describe PCA on an embedding matrix?",
    options: [
      {
        text: "Principal components are directions that can be linear combinations of many embedding coordinates.",
        isCorrect: true,
      },
      {
        text: "Centering the embeddings changes the covariance calculation and can affect PCA directions.",
        isCorrect: true,
      },
      {
        text: "A two-dimensional PCA plot preserves pairwise distances from the original space.",
        isCorrect: false,
      },
      {
        text: "PCA requires class labels because it is a supervised classifier.",
        isCorrect: false,
      },
    ],
    explanation:
      "PCA is an unsupervised geometric method based on covariance, so class labels are not required. Its directions are usually linear combinations of many original coordinates, which is why PCA is not simple feature selection. A low-dimensional plot can be informative while still losing some distances and variance.",
  },
  {
    id: "la-crash-l4-q54",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statement correctly distinguishes symmetric matrices from general square matrices?",
    options: [
      {
        text: "Real symmetric matrices have real eigenvalues and can be described using orthogonal eigenvector directions.",
        isCorrect: true,
      },
      {
        text: "A general square matrix has an orthogonal real eigenbasis by default.",
        isCorrect: false,
      },
      {
        text: "Nonsymmetric matrices lack eigenvectors in the real setting.",
        isCorrect: false,
      },
      {
        text: "Symmetry matters for storage efficiency rather than geometry.",
        isCorrect: false,
      },
    ],
    explanation:
      "Symmetry gives covariance matrices and related operators especially clean geometry: real eigenvalues and orthogonal eigendirections. General square matrices can behave less neatly and may lack a full real orthogonal eigenbasis. This is one reason covariance matrices are so useful for PCA.",
  },
  {
    id: "la-crash-l4-q55",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "For a matrix \\(A\\), which statements correctly connect SVD to eigenvectors?",
    options: [
      {
        text: "Right singular vectors of \\(A\\) are eigenvectors of \\(A^TA\\).",
        isCorrect: true,
      },
      {
        text: "Squared singular values of \\(A\\) are eigenvalues of \\(A^TA\\).",
        isCorrect: true,
      },
      {
        text: "Left singular vectors describe important output-space directions.",
        isCorrect: true,
      },
      {
        text: "Singular values are ordinary eigenvalues of \\(A\\) for rectangular matrices.",
        isCorrect: false,
      },
    ],
    explanation:
      "SVD is closely related to eigenanalysis of the symmetric matrix \\(A^TA\\). The right singular vectors come from input-space directions, and the corresponding singular values are square roots of eigenvalues of \\(A^TA\\). Rectangular matrices do not have ordinary eigenvalues in the same way square matrices do.",
  },
  {
    id: "la-crash-l4-q56",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "For a rank-\\(k\\) truncated SVD \\(A_k=U_k\\Sigma_kV_k^T\\) of an \\(m\\times n\\) matrix, which statements are correct?",
    options: [
      {
        text: "\\(A_k\\) has rank at most \\(k\\).",
        isCorrect: true,
      },
      {
        text: "Storing \\(U_k\\), the \\(k\\) singular values, and \\(V_k\\) can require about \\(k(m+n+1)\\) numbers.",
        isCorrect: true,
      },
      {
        text: "The approximation keeps the strongest \\(k\\) singular directions.",
        isCorrect: true,
      },
      {
        text: "The approximation error comes from the singular directions that were discarded.",
        isCorrect: true,
      },
    ],
    explanation:
      "A truncated SVD keeps only \\(k\\) singular directions, so its rank cannot exceed \\(k\\). It can save storage when \\(k(m+n+1)\\) is much smaller than \\(mn\\). The quality depends on how much signal remains in the discarded singular values.",
  },
  {
    id: "la-crash-l4-q57",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements correctly explain why PCA is not the same as feature selection?",
    options: [
      {
        text: "A principal component can combine many original features into one direction.",
        isCorrect: true,
      },
      {
        text: "PCA may rotate the coordinate system before projecting.",
        isCorrect: true,
      },
      {
        text: "PCA simply chooses a subset of original columns and discards the rest rather than rotating directions.",
        isCorrect: false,
      },
      {
        text: "PCA ignores covariance between features.",
        isCorrect: false,
      },
    ],
    explanation:
      "Feature selection keeps original variables, while PCA constructs new axes as linear combinations of the original variables. Those axes are chosen from covariance structure and can be rotated relative to the original coordinates. This is why PCA can preserve structure that no single original feature captures by itself.",
  },
  {
    id: "la-crash-l4-q58",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statement best explains why \\(X^TX\\) has no negative eigenvalues?",
    options: [
      {
        text: "For every vector \\(z\\), \\(z^TX^TXz=\\|Xz\\|^2\\ge 0\\).",
        isCorrect: true,
      },
      {
        text: "A matrix product has positive entries when it is written as \\(X^TX\\).",
        isCorrect: false,
      },
      {
        text: "A transpose removes zero directions from a matrix product.",
        isCorrect: false,
      },
      {
        text: "Real square matrices have nonnegative eigenvalues by default.",
        isCorrect: false,
      },
    ],
    explanation:
      "The quadratic form of \\(X^TX\\) is a squared norm, so it cannot be negative. This property is called positive semidefiniteness and is central to covariance geometry. General real square matrices can have negative eigenvalues, so the argument depends on the special \\(X^TX\\) structure.",
  },
  {
    id: "la-crash-l4-q59",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe what it means for an attention matrix to be approximately low rank?",
    options: [
      {
        text: "Some rows or columns may be well approximated by combinations of a smaller number of patterns.",
        isCorrect: true,
      },
      {
        text: "Its singular values may decay so that only a few directions carry most of the signal.",
        isCorrect: true,
      },
      {
        text: "A lower-rank approximation may reduce computation or storage while preserving the main structure.",
        isCorrect: true,
      },
      {
        text: "Approximate low rank means attention scores are mostly zero.",
        isCorrect: false,
      },
    ],
    explanation:
      "Approximate low rank means the matrix has redundant structure that can be summarized by fewer dominant directions. SVD expresses this through singular values and singular vectors. It does not mean the matrix is empty or that all entries vanish.",
  },
  {
    id: "la-crash-l4-q60",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements correctly connect eigenvectors, SVD, and representation learning?",
    options: [
      {
        text: "Learned transformations can make useful semantic or decision directions easier to separate.",
        isCorrect: true,
      },
      {
        text: "Covariance and PCA can help inspect dominant directions in learned embeddings.",
        isCorrect: true,
      },
      {
        text: "SVD can reveal redundancy and compression opportunities in learned matrices.",
        isCorrect: true,
      },
      {
        text: "Eigenvector and SVD analysis can support model understanding without replacing gradient-based training.",
        isCorrect: true,
      },
    ],
    explanation:
      "Representation learning creates vector spaces where useful information is organized geometrically. Eigenvectors, covariance, PCA, and SVD provide tools for studying dominant directions, redundancy, and low-dimensional structure in those spaces. These analyses complement training rather than replacing the optimization process.",
  },
];

export const CrashCourseLinearAlgebraL4Questions =
  CrashCourseLinearAlgebraLecture4Questions;
