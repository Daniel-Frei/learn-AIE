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
        text: "The equation means \\(v\\) must become perpendicular to itself.",
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
        text: "A negative eigenvalue can never flip direction.",
        isCorrect: false,
      },
      {
        text: "Every eigenvalue must be exactly 1.",
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
        text: "They are random directions chosen independently of the matrix.",
        isCorrect: false,
      },
      {
        text: "They are vectors that cannot be multiplied by a matrix.",
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
        text: "Power iteration proves that every direction is amplified equally.",
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
        text: "Near-zero covariance proves there is no nonlinear relationship of any kind.",
        isCorrect: false,
      },
      {
        text: "Covariance measures only the number of rows in a dataset.",
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
        text: "It stores only class labels.",
        isCorrect: false,
      },
      {
        text: "It is unrelated to feature correlations.",
        isCorrect: false,
      },
      {
        text: "It can only be formed from one-dimensional data.",
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
        text: "Dimensionality reduction always preserves every original coordinate exactly.",
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
        text: "Principal components are chosen only by alphabetical feature names.",
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
        text: "It only works when every matrix entry is zero.",
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
        text: "\\(U\\) removes all output-space geometry from the decomposition.",
        isCorrect: false,
      },
      {
        text: "\\(\\Sigma\\) stores only unrelated labels, not numerical scales.",
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
        text: "They are always meaningless noise.",
        isCorrect: false,
      },
      {
        text: "They must all be equal for SVD to work.",
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
        text: "Low-rank structure means the model cannot contain useful information.",
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
        text: "LoRA requires updating every full model weight directly.",
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
        text: "Every embedding coordinate always carries unrelated information of equal importance.",
        isCorrect: false,
      },
      {
        text: "Compression works only by deleting labels, not numerical structure.",
        isCorrect: false,
      },
      {
        text: "Embeddings are not numerical vectors, so compression is impossible.",
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
        text: "It intentionally removes all relationships between features before analysis.",
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
        text: "Every coordinate must have a simple human-readable meaning.",
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
        text: "PCA only chooses existing columns by name.",
        isCorrect: false,
      },
      {
        text: "PCA cannot reduce dimensionality.",
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
        text: "Representation learning guarantees every latent dimension is independent.",
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
        text: "Information can never concentrate in major attention directions.",
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
        text: "SVD compression requires increasing every matrix dimension.",
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
        text: "It proves every cluster has a causal biological or semantic explanation.",
        isCorrect: false,
      },
      {
        text: "It requires ignoring all vector geometry.",
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
        text: "SVD works only for diagonal covariance matrices.",
        isCorrect: false,
      },
      {
        text: "SVD cannot be applied to learned model weights.",
        isCorrect: false,
      },
      {
        text: "SVD removes all geometric interpretation.",
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
        text: "Low-rank structure always means low-quality information.",
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
        text: "Attention matrices cannot show structured redundancy.",
        isCorrect: false,
      },
      {
        text: "Low-rank methods can support efficient adaptation or compression.",
        isCorrect: true,
      },
      {
        text: "Transformers cannot be analyzed using linear algebra.",
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
        text: "All directions in high-dimensional data are always equally important.",
        isCorrect: false,
      },
      {
        text: "Compression is impossible when data is numerical.",
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
        text: "Every possible fine-tuning task must be exactly rank one.",
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
        text: "Low-rank factors always make storage and computation larger.",
        isCorrect: false,
      },
      {
        text: "SVD is useful only for matrices with no repeated patterns.",
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
        text: "They prove every model must use full-rank updates for adaptation.",
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
];

export const CrashCourseLinearAlgebraL4Questions =
  CrashCourseLinearAlgebraLecture4Questions;
