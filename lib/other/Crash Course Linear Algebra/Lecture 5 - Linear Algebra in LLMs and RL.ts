import { Question } from "../../quiz";

export const CrashCourseLinearAlgebraLecture5Questions: Question[] = [
  {
    id: "la-crash-l5-q01",
    chapter: 5,
    difficulty: "easy",
    prompt: "Which statements correctly describe why attention is useful?",
    options: [
      {
        text: "It lets tokens dynamically reference other tokens.",
        isCorrect: true,
      },
      {
        text: "It helps model long-range relationships in a sequence.",
        isCorrect: true,
      },
      {
        text: "It lets token representations become context-dependent.",
        isCorrect: true,
      },
      {
        text: "It uses learned geometry rather than a fixed one-vector sentence summary.",
        isCorrect: true,
      },
    ],
    explanation:
      "Attention allows each token to form a weighted view of other tokens, so representations can depend on context. This avoids forcing the whole sequence through one fixed bottleneck vector and helps with long-range relationships.",
  },
  {
    id: "la-crash-l5-q02",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "If \\(X\\in\\mathbb{R}^{n\\times d}\\) is an input embedding matrix, which statements are correct?",
    options: [
      { text: "\\(n\\) can represent the number of tokens.", isCorrect: true },
      {
        text: "\\(d\\) can represent the embedding dimension.",
        isCorrect: true,
      },
      {
        text: "Each row can represent one token embedding.",
        isCorrect: true,
      },
      {
        text: "\\(X\\) must contain exactly one token regardless of sequence length.",
        isCorrect: false,
      },
    ],
    explanation:
      "A common convention stores one token embedding per row, giving \\(n\\) rows and \\(d\\) features. The matrix can represent many tokens, so it is not limited to a single-token sequence.",
  },
  {
    id: "la-crash-l5-q03",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the transformer projections \\(Q=XW_Q\\), \\(K=XW_K\\), and \\(V=XW_V\\)?",
    options: [
      {
        text: "They are learned linear transformations of the input embeddings.",
        isCorrect: true,
      },
      {
        text: "They produce query, key, and value representations.",
        isCorrect: true,
      },
      {
        text: "They prevent the model from comparing and routing information in learned spaces.",
        isCorrect: false,
      },
      {
        text: "They replace learned projections with fixed routing weights.",
        isCorrect: false,
      },
    ],
    explanation:
      "The query, key, and value matrices come from multiplying embeddings by learned projection matrices. These projections are trainable and create the spaces in which attention compares relevance and passes information, so they do not prevent learned routing.",
  },
  {
    id: "la-crash-l5-q04",
    chapter: 5,
    difficulty: "easy",
    prompt: "Which statement best describes a query vector in attention?",
    options: [
      {
        text: "It represents what information a token is looking for.",
        isCorrect: true,
      },
      {
        text: "It stores the final predicted class label rather than contextual states.",
        isCorrect: false,
      },
      {
        text: "It is unrelated to dot products.",
        isCorrect: false,
      },
      {
        text: "It replaces keys and values with zero vectors before scoring.",
        isCorrect: false,
      },
    ],
    explanation:
      "A query is the representation used to ask which keys are relevant. It is compared with keys using dot products, so it is central to the similarity computation.",
  },
  {
    id: "la-crash-l5-q05",
    chapter: 5,
    difficulty: "easy",
    prompt: "Which statements correctly describe keys and values in attention?",
    options: [
      {
        text: "A key represents what information a token offers for matching.",
        isCorrect: true,
      },
      {
        text: "A value represents information that can be passed forward.",
        isCorrect: true,
      },
      {
        text: "Keys are compared with queries to produce attention scores.",
        isCorrect: true,
      },
      {
        text: "Values are combined using attention weights.",
        isCorrect: true,
      },
    ],
    explanation:
      "Keys provide the vectors that queries compare against, while values carry the information to aggregate after relevance is computed. Attention scores determine how strongly each value contributes to the output.",
  },
  {
    id: "la-crash-l5-q06",
    chapter: 5,
    difficulty: "easy",
    prompt: "Which statements correctly describe dot-product attention scores?",
    options: [
      {
        text: "They are computed with \\(QK^T\\).",
        isCorrect: true,
      },
      {
        text: "They measure alignment between query and key vectors.",
        isCorrect: true,
      },
      {
        text: "Large dot products can indicate strong relevance.",
        isCorrect: true,
      },
      {
        text: "They are based on alphabetical token order rather than learned projections.",
        isCorrect: false,
      },
    ],
    explanation:
      "Dot products compare the geometric alignment between queries and keys. The resulting score matrix is learned from the representations, not from token names or alphabetical ordering.",
  },
  {
    id: "la-crash-l5-q07",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "If \\(Q\\in\\mathbb{R}^{n\\times d}\\) and \\(K\\in\\mathbb{R}^{n\\times d}\\), which statements about \\(QK^T\\) are correct?",
    options: [
      {
        text: "\\(K^T\\in\\mathbb{R}^{d\\times n}\\).",
        isCorrect: true,
      },
      {
        text: "\\(QK^T\\in\\mathbb{R}^{n\\times n}\\).",
        isCorrect: true,
      },
      {
        text: "The result uses one row per embedding dimension rather than per token.",
        isCorrect: false,
      },
      {
        text: "The transpose is a cosmetic notation choice because \\(QK\\) already has shape \\(n\\times n\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Transposing \\(K\\) changes its shape to \\(d\\times n\\), so the inner dimensions align and the output is \\(n\\times n\\). This gives one score row per query token, not one row per embedding dimension.",
  },
  {
    id: "la-crash-l5-q08",
    chapter: 5,
    difficulty: "easy",
    prompt: "Which statement best describes softmax in attention?",
    options: [
      {
        text: "It converts raw scores into a weighted distribution over tokens.",
        isCorrect: true,
      },
      {
        text: "It discards value vectors before aggregation.",
        isCorrect: false,
      },
      {
        text: "It gives token weights negative values after normalization.",
        isCorrect: false,
      },
      {
        text: "It prevents attention from being differentiable.",
        isCorrect: false,
      },
    ],
    explanation:
      "Softmax turns raw attention scores into nonnegative weights that sum to one. Those weights form a soft focus mechanism over tokens rather than deleting values or breaking differentiability.",
  },
  {
    id: "la-crash-l5-q09",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the attention output \\(\\text{softmax}(QK^T)V\\)?",
    options: [
      {
        text: "It uses attention weights to combine value vectors.",
        isCorrect: true,
      },
      {
        text: "It produces contextualized token representations.",
        isCorrect: true,
      },
      {
        text: "It aggregates information from relevant tokens.",
        isCorrect: true,
      },
      {
        text: "It connects dot-product similarity to information routing.",
        isCorrect: true,
      },
    ],
    explanation:
      "After scores are converted into weights, the model takes weighted sums of value vectors. The result is contextualized because each token representation can include information from other relevant tokens.",
  },
  {
    id: "la-crash-l5-q10",
    chapter: 5,
    difficulty: "easy",
    prompt: "Which statements correctly describe multi-head attention?",
    options: [
      {
        text: "Different heads can learn different projection matrices.",
        isCorrect: true,
      },
      {
        text: "Different heads can attend to different relationships.",
        isCorrect: true,
      },
      {
        text: "Heads can capture complementary patterns such as syntax or entity relationships.",
        isCorrect: true,
      },
      {
        text: "The heads share identical learned projections.",
        isCorrect: false,
      },
    ],
    explanation:
      "Multi-head attention gives the model several learned ways to project and compare tokens. The heads are not forced to be identical, so they can specialize in different geometric or semantic relationships.",
  },
  {
    id: "la-crash-l5-q11",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe neural networks as stacks of transformations?",
    options: [
      {
        text: "A layer can apply a learned matrix transformation.",
        isCorrect: true,
      },
      {
        text: "Layers can reshape representation geometry over depth.",
        isCorrect: true,
      },
      {
        text: "Nonlinearities contribute little to expressivity in stacked layers.",
        isCorrect: false,
      },
      {
        text: "Neural networks rely on lookup tables instead of matrix multiplication.",
        isCorrect: false,
      },
    ],
    explanation:
      "Neural networks repeatedly transform representations using matrix operations, and nonlinearities are what prevent deep linear stacks from collapsing into one linear map. These transformations can rotate, stretch, compress, and separate data in ways that support prediction.",
  },
  {
    id: "la-crash-l5-q12",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statement best explains why nonlinearities are needed between linear layers?",
    options: [
      {
        text: "Without nonlinearities, stacked linear layers collapse into a single linear transformation.",
        isCorrect: true,
      },
      {
        text: "Nonlinearities are needed because matrix products fail on vector inputs.",
        isCorrect: false,
      },
      {
        text: "Nonlinearities remove the learned parameter matrices.",
        isCorrect: false,
      },
      {
        text: "A stack of linear layers without activations is automatically more expressive than any nonlinear network.",
        isCorrect: false,
      },
    ],
    explanation:
      "The composition \\(W_2(W_1x)\\) can be rewritten as \\((W_2W_1)x\\), which is still linear. Nonlinear activations prevent this collapse and allow deeper networks to form richer decision boundaries.",
  },
  {
    id: "la-crash-l5-q13",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe representation learning across layers?",
    options: [
      {
        text: "Early layers can represent simpler features.",
        isCorrect: true,
      },
      {
        text: "Later layers can represent more abstract concepts.",
        isCorrect: true,
      },
      {
        text: "An LLM can build representations related to syntax, semantics, or reasoning patterns.",
        isCorrect: true,
      },
      {
        text: "Layered models can gradually reshape the representation space.",
        isCorrect: true,
      },
    ],
    explanation:
      "Deep networks can transform raw inputs into increasingly useful internal representations. In language models, that can mean moving from token-level information toward syntax, semantics, and task-relevant abstractions.",
  },
  {
    id: "la-crash-l5-q14",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe reinforcement-learning states and actions?",
    options: [
      {
        text: "A state can be represented as a vector.",
        isCorrect: true,
      },
      {
        text: "An action is something the agent can choose.",
        isCorrect: true,
      },
      {
        text: "Rewards provide feedback about outcomes.",
        isCorrect: true,
      },
      {
        text: "Reinforcement learning has no concept of future reward.",
        isCorrect: false,
      },
    ],
    explanation:
      "Reinforcement learning involves an agent choosing actions in states and receiving rewards. The objective usually involves future reward, so it is not limited to immediate feedback.",
  },
  {
    id: "la-crash-l5-q15",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe a value function \\(Q(s,a)\\)?",
    options: [
      {
        text: "It estimates expected future reward for taking action \\(a\\) in state \\(s\\).",
        isCorrect: true,
      },
      {
        text: "It can be approximated by a neural network \\(Q_\\theta(s,a)\\).",
        isCorrect: true,
      },
      {
        text: "Its inputs are separate symbols rather than vectorized state or action information.",
        isCorrect: false,
      },
      {
        text: "It is represented as a handwritten table in modern deep RL systems.",
        isCorrect: false,
      },
    ],
    explanation:
      "\\(Q(s,a)\\) is a prediction about future return, and modern deep RL often represents it with a neural network. State and action information can be vectorized, and neural approximation is common when the space is too large for a simple table.",
  },
  {
    id: "la-crash-l5-q16",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statement best describes the loss \\(L=(Q_{target}-Q_\\theta)^2\\) in value-function learning?",
    options: [
      {
        text: "It penalizes disagreement between a target value and the model's current estimate.",
        isCorrect: true,
      },
      {
        text: "It prevents gradients from updating the value network.",
        isCorrect: false,
      },
      {
        text: "It is unrelated to prediction error.",
        isCorrect: false,
      },
      {
        text: "It is used when there are no parameters \\(\\theta\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The squared error is small when the model's value estimate is close to the target and large when they differ. Gradients of this loss can update the parameters \\(\\theta\\) to improve future value estimates.",
  },
  {
    id: "la-crash-l5-q17",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statements correctly describe loss landscapes?",
    options: [
      {
        text: "A loss function \\(L(W)\\) defines a surface over parameter space.",
        isCorrect: true,
      },
      {
        text: "Neural-network parameter spaces can be extremely high-dimensional.",
        isCorrect: true,
      },
      {
        text: "Gradient descent moves iteratively through this landscape.",
        isCorrect: true,
      },
      {
        text: "A loss landscape can have far more dimensions than a two-dimensional drawing can show.",
        isCorrect: true,
      },
    ],
    explanation:
      "The loss depends on many parameters, so its landscape can have far more than two dimensions. The first three statements are correct, but the last one is false and should not be selected.",
  },
  {
    id: "la-crash-l5-q18",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe local minima and saddle points?",
    options: [
      {
        text: "A local minimum has nearby directions that increase loss.",
        isCorrect: true,
      },
      {
        text: "A saddle point can be uphill in some directions and downhill in others.",
        isCorrect: true,
      },
      {
        text: "High-dimensional systems can contain many saddle points.",
        isCorrect: true,
      },
      {
        text: "A critical point in a neural network is treated as a bad local minimum.",
        isCorrect: false,
      },
    ],
    explanation:
      "A saddle point is not simply a minimum; it can have escape directions where loss decreases. In high-dimensional neural-network landscapes, saddle points are common and not every critical point is a bad local minimum.",
  },
  {
    id: "la-crash-l5-q19",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe why high-dimensional optimization can behave differently?",
    options: [
      {
        text: "There can be many possible directions to move.",
        isCorrect: true,
      },
      {
        text: "Some critical points may have escape directions.",
        isCorrect: true,
      },
      {
        text: "Bad local minima are the main optimization concern in this setting.",
        isCorrect: false,
      },
      {
        text: "High dimensions make gradient directions unusable.",
        isCorrect: false,
      },
    ],
    explanation:
      "High-dimensional spaces have many directions, and saddle points can allow movement downhill in at least some of them. Bad local minima are not the only concern, so the claim that they are the only concern is incorrect.",
  },
  {
    id: "la-crash-l5-q20",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statement best summarizes why linear algebra dominates modern AI?",
    options: [
      {
        text: "Data, models, learning, similarity, and compression are all naturally expressed with vectors, matrices, gradients, dot products, and low-rank structure.",
        isCorrect: true,
      },
      {
        text: "Modern AI avoids numerical representations.",
        isCorrect: false,
      },
      {
        text: "Transformers do not use matrix multiplication.",
        isCorrect: false,
      },
      {
        text: "Embeddings are unrelated to geometry.",
        isCorrect: false,
      },
    ],
    explanation:
      "Modern AI systems represent data as vectors, transform it with matrices, learn with gradients, compare it with dot products, and compress it with low-rank structure. These are all linear-algebraic ideas.",
  },
  {
    id: "la-crash-l5-q21",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statements correctly connect dot products to AI systems?",
    options: [
      {
        text: "Attention uses dot products between queries and keys.",
        isCorrect: true,
      },
      {
        text: "Retrieval systems can use cosine similarity, which is based on vector alignment.",
        isCorrect: true,
      },
      {
        text: "Embedding similarity can be interpreted geometrically.",
        isCorrect: true,
      },
      {
        text: "Dot products help quantify alignment or relevance.",
        isCorrect: true,
      },
    ],
    explanation:
      "Dot products are a core way to measure alignment between vectors. Attention, retrieval, and embedding comparison all rely on this geometric idea in different forms.",
  },
  {
    id: "la-crash-l5-q22",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statements correctly describe contextualized embeddings?",
    options: [
      {
        text: "They can depend on surrounding tokens.",
        isCorrect: true,
      },
      {
        text: "They are produced by aggregating relevant information through attention.",
        isCorrect: true,
      },
      {
        text: "They can represent the same token differently in different contexts.",
        isCorrect: true,
      },
      {
        text: "They are fixed forever before seeing a sentence.",
        isCorrect: false,
      },
    ],
    explanation:
      "Attention lets a token representation incorporate information from its context. That is why the same token can receive different contextualized embeddings depending on the sequence around it.",
  },
  {
    id: "la-crash-l5-q23",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe transformers as geometric routing systems?",
    options: [
      {
        text: "Queries and keys route information by learned similarity.",
        isCorrect: true,
      },
      {
        text: "Values carry information that is combined using learned weights.",
        isCorrect: true,
      },
      {
        text: "The routing is nondifferentiable and trained outside gradient methods.",
        isCorrect: false,
      },
      {
        text: "The routing is a hand-written symbolic lookup table.",
        isCorrect: false,
      },
    ],
    explanation:
      "Transformers route information through differentiable vector operations rather than fixed symbolic rules. The projections and resulting attention patterns are trainable, so the routing is not a nondifferentiable fixed mechanism.",
  },
  {
    id: "la-crash-l5-q24",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statement best explains why \\(W_2(W_1x)\\) is not enough by itself to create deep nonlinear expressivity?",
    options: [
      {
        text: "Because it can be combined into a single linear map \\((W_2W_1)x\\).",
        isCorrect: true,
      },
      {
        text: "Because matrix multiplication is undefined in neural-network layers.",
        isCorrect: false,
      },
      {
        text: "Because \\(W_1\\) is constrained to equal \\(W_2\\).",
        isCorrect: false,
      },
      {
        text: "Because matrix multiplication removes information in this architecture.",
        isCorrect: false,
      },
    ],
    explanation:
      "A stack of linear transformations without nonlinear activations is still one linear transformation. Nonlinearities are what let depth build more expressive transformations than a single matrix map.",
  },
  {
    id: "la-crash-l5-q25",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the RL connection to linear algebra?",
    options: [
      {
        text: "States can be encoded as vectors.",
        isCorrect: true,
      },
      {
        text: "Value networks use matrix multiplications and nonlinear transformations.",
        isCorrect: true,
      },
      {
        text: "Gradients can update value-function parameters.",
        isCorrect: true,
      },
      {
        text: "Learning value functions can reshape representation space.",
        isCorrect: true,
      },
    ],
    explanation:
      "Deep RL represents states, actions, and value estimates with numerical objects that neural networks can process. Matrix transformations and gradients are central to how these representations are learned and updated.",
  },
  {
    id: "la-crash-l5-q26",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statements correctly describe optimization in transformers?",
    options: [
      {
        text: "Training adjusts many projection and feed-forward parameters.",
        isCorrect: true,
      },
      {
        text: "Backpropagation sends gradient signals through attention and other layers.",
        isCorrect: true,
      },
      {
        text: "Large language model training is an enormous high-dimensional optimization problem.",
        isCorrect: true,
      },
      {
        text: "Transformer training does not use loss functions.",
        isCorrect: false,
      },
    ],
    explanation:
      "Transformer training defines losses and uses backpropagation to update many matrix parameters. The resulting optimization problem can have billions of dimensions in large models.",
  },
  {
    id: "la-crash-l5-q27",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe low-rank structure in the final synthesis?",
    options: [
      {
        text: "Real-world data can contain redundancy.",
        isCorrect: true,
      },
      {
        text: "Compression can exploit correlated directions.",
        isCorrect: true,
      },
      {
        text: "Low-rank methods are unrelated to efficient adaptation.",
        isCorrect: false,
      },
      {
        text: "Low-rank structure makes vector representations invalid.",
        isCorrect: false,
      },
    ],
    explanation:
      "Redundancy and correlated directions are why compression can work in high-dimensional systems. Low-rank methods can help explain efficient adaptation, so saying they are unrelated to efficient adaptation is incorrect.",
  },
  {
    id: "la-crash-l5-q28",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statement best explains why attention uses \\(K^T\\) in \\(QK^T\\)?",
    options: [
      {
        text: "It aligns dimensions so every query can be dotted with every key, producing an \\(n\\times n\\) score matrix.",
        isCorrect: true,
      },
      {
        text: "It is used to make notation look symmetric rather than to align dimensions.",
        isCorrect: false,
      },
      {
        text: "It prevents query-key comparisons.",
        isCorrect: false,
      },
      {
        text: "It changes the number of tokens to zero.",
        isCorrect: false,
      },
    ],
    explanation:
      "With \\(Q\\) and \\(K\\) shaped as \\(n\\times d\\), transposing \\(K\\) makes the multiplication valid and yields all token-pair scores. The transpose has a concrete shape purpose, not just a cosmetic one.",
  },
  {
    id: "la-crash-l5-q29",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the end-to-end attention computation?",
    options: [
      {
        text: "Input embeddings are projected into query, key, and value spaces.",
        isCorrect: true,
      },
      {
        text: "Query-key dot products produce relevance scores.",
        isCorrect: true,
      },
      {
        text: "Softmax turns scores into weights.",
        isCorrect: true,
      },
      {
        text: "Weighted sums of values produce contextual outputs.",
        isCorrect: true,
      },
    ],
    explanation:
      "Attention starts with learned projections, compares queries and keys, normalizes the scores, and aggregates values. This sequence links linear transformations, dot products, softmax, and weighted sums into one differentiable operation.",
  },
  {
    id: "la-crash-l5-q30",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly distinguish attention from symbolic reasoning?",
    options: [
      {
        text: "Attention is differentiable similarity-based routing.",
        isCorrect: true,
      },
      {
        text: "Its scores arise from vector geometry.",
        isCorrect: true,
      },
      {
        text: "It can support complex behavior without being a hand-coded symbolic rule system.",
        isCorrect: true,
      },
      {
        text: "Attention directly stores facts as database rows.",
        isCorrect: false,
      },
    ],
    explanation:
      "Attention routes information using learned vector comparisons and differentiable weighting. It can contribute to reasoning-like behavior, but it is not the same as a manually written symbolic database lookup.",
  },
  {
    id: "la-crash-l5-q31",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly connect neural-network geometry to decision making?",
    options: [
      {
        text: "Layers can separate representations that were previously mixed together.",
        isCorrect: true,
      },
      {
        text: "Nonlinear transformations can create more flexible decision boundaries.",
        isCorrect: true,
      },
      {
        text: "Each layer preserves distances as it transforms representations.",
        isCorrect: false,
      },
      {
        text: "Representation geometry has little connection to prediction.",
        isCorrect: false,
      },
    ],
    explanation:
      "Neural networks can reshape spaces so that useful distinctions become easier for later layers to use. They do not need to preserve every distance exactly, and geometry is central to how representations support prediction.",
  },
  {
    id: "la-crash-l5-q32",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statement best explains why deep RL value learning is a geometry problem as well as a reward problem?",
    options: [
      {
        text: "The network must shape state-action representations so useful reward predictions become easier.",
        isCorrect: true,
      },
      {
        text: "Rewards eliminate the need for state representations.",
        isCorrect: false,
      },
      {
        text: "Value functions use scalar tables rather than vector features.",
        isCorrect: false,
      },
      {
        text: "Gradient descent is separate from RL network adjustment.",
        isCorrect: false,
      },
    ],
    explanation:
      "A value network must transform state and action inputs into representations that support accurate future-reward estimates. Rewards define the learning signal, but the network still uses vector geometry and gradient updates.",
  },
  {
    id: "la-crash-l5-q33",
    chapter: 5,
    difficulty: "hard",
    prompt: "Which statements correctly describe saddle-point intuition?",
    options: [
      {
        text: "A saddle point can have zero or small gradient while not being a useful minimum.",
        isCorrect: true,
      },
      {
        text: "Some directions near a saddle can reduce loss.",
        isCorrect: true,
      },
      {
        text: "High-dimensional landscapes can contain many saddle-like regions.",
        isCorrect: true,
      },
      {
        text: "Saddle points help explain why optimization is not only about local minima.",
        isCorrect: true,
      },
    ],
    explanation:
      "A saddle point is a critical region that can look flat or stationary while still having downhill directions. This matters in high-dimensional optimization because training may need to escape such regions rather than only avoid local minima.",
  },
  {
    id: "la-crash-l5-q34",
    chapter: 5,
    difficulty: "hard",
    prompt: "Which statements correctly summarize the course-level synthesis?",
    options: [
      {
        text: "Data becomes vectors.",
        isCorrect: true,
      },
      {
        text: "Models use matrices and nonlinear transformations.",
        isCorrect: true,
      },
      {
        text: "Learning uses gradients.",
        isCorrect: true,
      },
      {
        text: "Similarity in modern AI avoids vector alignment.",
        isCorrect: false,
      },
    ],
    explanation:
      "The first three statements are central to the synthesis: data becomes vectors, models use transformations, and learning uses gradients. Similarity often does use vector alignment, so the final absolute claim is incorrect.",
  },
  {
    id: "la-crash-l5-q35",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly explain why LLM knowledge is distributed?",
    options: [
      {
        text: "Information can be encoded across many parameters and representation directions.",
        isCorrect: true,
      },
      {
        text: "Embeddings and hidden states organize relationships geometrically.",
        isCorrect: true,
      },
      {
        text: "Model behavior comes from a simple row-per-fact table.",
        isCorrect: false,
      },
      {
        text: "LLMs store facts in a single human-readable lookup table.",
        isCorrect: false,
      },
    ],
    explanation:
      "LLM knowledge is distributed across weights, activations, and learned geometric representations. This differs from a simple row-per-fact table, even though models can sometimes retrieve fact-like information.",
  },
  {
    id: "la-crash-l5-q36",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statement best describes the relationship between attention and matrix multiplication?",
    options: [
      {
        text: "Attention is built from matrix multiplications that project embeddings, compute scores, and aggregate values.",
        isCorrect: true,
      },
      {
        text: "Attention compares token strings directly rather than vector projections.",
        isCorrect: false,
      },
      {
        text: "Attention is treated as a single-example operation.",
        isCorrect: false,
      },
      {
        text: "Attention works in the single-token case rather than sequences.",
        isCorrect: false,
      },
    ],
    explanation:
      "Attention relies on matrix multiplication for projections, query-key scoring, and value aggregation. These operations are naturally batched and are one reason transformers fit modern linear-algebra hardware well.",
  },
  {
    id: "la-crash-l5-q37",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly connect all five linear-algebra themes?",
    options: [
      {
        text: "Vectors represent data, states, and embeddings.",
        isCorrect: true,
      },
      {
        text: "Matrices transform representations.",
        isCorrect: true,
      },
      {
        text: "Gradients update transformations during learning.",
        isCorrect: true,
      },
      {
        text: "Low-rank and eigenstructure help explain representation compression.",
        isCorrect: true,
      },
    ],
    explanation:
      "The course builds a chain from vectors to matrices, gradients, representation structure, and modern AI systems. These ideas combine in transformers, deep networks, and reinforcement-learning value approximators.",
  },
  {
    id: "la-crash-l5-q38",
    chapter: 5,
    difficulty: "hard",
    prompt: "Which statements correctly describe common misconceptions?",
    options: [
      {
        text: "Attention is not simply symbolic reasoning; it is differentiable vector-based routing.",
        isCorrect: true,
      },
      {
        text: "Deep learning is heavily geometric and optimization-based.",
        isCorrect: true,
      },
      {
        text: "High-dimensional spaces still use core ideas such as dot products, projections, and gradients.",
        isCorrect: true,
      },
      {
        text: "LLMs store facts exactly like a normalized relational database.",
        isCorrect: false,
      },
    ],
    explanation:
      "Attention and deep learning rely on numerical geometry, not just symbolic rules. High-dimensional models still use familiar linear-algebra ideas, and LLMs do not store knowledge as clean database rows.",
  },
  {
    id: "la-crash-l5-q39",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly explain why transformer training is possible at scale?",
    options: [
      {
        text: "Core operations are matrix multiplications that hardware accelerates well.",
        isCorrect: true,
      },
      {
        text: "Backpropagation provides gradients through the composed operations.",
        isCorrect: true,
      },
      {
        text: "Vectorized computations prevent tokens or examples from being processed efficiently.",
        isCorrect: false,
      },
      {
        text: "Training works because transformers have no parameters.",
        isCorrect: false,
      },
    ],
    explanation:
      "Transformers are parameterized models with many matrix operations, and modern hardware is very good at those operations. Backpropagation and vectorization make it possible to train these large composed systems efficiently.",
  },
  {
    id: "la-crash-l5-q40",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statement best summarizes modern AI from the linear-algebra perspective?",
    options: [
      {
        text: "Modern AI is high-dimensional geometric optimization over vector spaces.",
        isCorrect: true,
      },
      {
        text: "Modern AI avoids vectors, matrices, and gradients.",
        isCorrect: false,
      },
      {
        text: "Attention, embeddings, RL value functions, and optimization share no mathematical structure.",
        isCorrect: false,
      },
      {
        text: "Linear algebra stops being useful once models become large.",
        isCorrect: false,
      },
    ],
    explanation:
      "Vectors, matrices, dot products, gradients, low-rank structure, and representation geometry show up across LLMs, deep learning, and reinforcement learning. Model scale increases the importance of these ideas rather than making them irrelevant.",
  },
  {
    id: "la-crash-l5-q41",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "If \\(X\\in\\mathbb{R}^{n\\times d}\\) and \\(W_Q\\in\\mathbb{R}^{d\\times d_k}\\), which statements about \\(Q=XW_Q\\) are correct?",
    options: [
      {
        text: "\\(Q\\in\\mathbb{R}^{n\\times d_k}\\).",
        isCorrect: true,
      },
      {
        text: "Each row of \\(Q\\) is a learned query representation for one token.",
        isCorrect: true,
      },
      {
        text: "\\(W_Q\\) is a learned linear transformation of the embedding space.",
        isCorrect: true,
      },
      {
        text: "The multiplication is valid because the inner \\(d\\) dimensions align.",
        isCorrect: true,
      },
    ],
    explanation:
      "The input rows are token embeddings, and multiplying on the right applies the same learned projection to each token. The product shape is \\(n\\times d_k\\) because the shared \\(d\\) dimension contracts. This is a direct use of matrices as learned transformations.",
  },
  {
    id: "la-crash-l5-q42",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "If \\(Q,K\\in\\mathbb{R}^{n\\times d}\\), which statements about the attention score matrix \\(QK^T\\) are correct?",
    options: [
      {
        text: "\\(QK^T\\in\\mathbb{R}^{n\\times n}\\).",
        isCorrect: true,
      },
      {
        text: "Entry \\((i,j)\\) compares token \\(i\\)'s query with token \\(j\\)'s key using a dot product.",
        isCorrect: true,
      },
      {
        text: "\\(QK^T\\in\\mathbb{R}^{d\\times d}\\) because attention compares embedding coordinates rather than token pairs.",
        isCorrect: false,
      },
      {
        text: "The values matrix \\(V\\) is needed before any query-key score can be computed.",
        isCorrect: false,
      },
    ],
    explanation:
      "The transpose changes \\(K\\) to shape \\(d\\times n\\), so each query can be compared with every key. The result is a token-by-token matrix, not an embedding-coordinate matrix. Values are used after the attention weights are computed.",
  },
  {
    id: "la-crash-l5-q43",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "A row of attention scores is \\([2,1,0]\\). Which statement best describes the softmax of this row?",
    options: [
      {
        text: "The largest score receives the largest weight, but the weights still form a soft distribution that sums to \\(1\\).",
        isCorrect: true,
      },
      {
        text: "The largest score becomes weight \\(1\\), while lower scores become weight \\(0\\).",
        isCorrect: false,
      },
      {
        text: "Softmax makes the smallest score receive the largest weight.",
        isCorrect: false,
      },
      {
        text: "Softmax changes the number of tokens in the sequence.",
        isCorrect: false,
      },
    ],
    explanation:
      "Softmax converts raw scores into nonnegative weights that sum to one. Higher scores receive higher weights, but ordinary softmax is not hard selection unless the score differences become extreme. The sequence length is unchanged.",
  },
  {
    id: "la-crash-l5-q44",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "If \\(A=\\operatorname{softmax}(QK^T)\\in\\mathbb{R}^{n\\times n}\\) and \\(V\\in\\mathbb{R}^{n\\times d_v}\\), which statements about \\(AV\\) are correct?",
    options: [
      {
        text: "\\(AV\\in\\mathbb{R}^{n\\times d_v}\\).",
        isCorrect: true,
      },
      {
        text: "Each output row is a weighted sum of value vectors.",
        isCorrect: true,
      },
      {
        text: "When softmax is applied row-wise, each row of \\(A\\) can be interpreted as attention weights over tokens.",
        isCorrect: true,
      },
      {
        text: "\\(AV\\) collapses the whole sequence into one vector regardless of \\(n\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The attention matrix has one row of weights per query token, and multiplying by \\(V\\) mixes value vectors according to those weights. The output keeps one row per token, now contextualized by information from other tokens. Attention does not automatically collapse the sequence to a single vector.",
  },
  {
    id: "la-crash-l5-q45",
    chapter: 5,
    difficulty: "easy",
    prompt: "Which statements correctly describe multi-head attention?",
    options: [
      {
        text: "Different heads use different learned projection matrices.",
        isCorrect: true,
      },
      {
        text: "Different heads can focus on different geometric or semantic relationships.",
        isCorrect: true,
      },
      {
        text: "The head outputs are combined to form a richer representation.",
        isCorrect: true,
      },
      {
        text: "Multiple heads let the same token sequence be viewed through several learned subspaces.",
        isCorrect: true,
      },
    ],
    explanation:
      "Multi-head attention repeats the query-key-value idea with multiple learned projections. Each head can form its own similarity geometry and route different kinds of information. Combining heads gives the model a richer contextual representation than a single attention pattern.",
  },
  {
    id: "la-crash-l5-q46",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements correctly explain why nonlinearities matter in a neural network stack?",
    options: [
      {
        text: "Without nonlinearities, \\(W_2(W_1x)\\) is equivalent to \\((W_2W_1)x\\).",
        isCorrect: true,
      },
      {
        text: "Nonlinearities allow learned transformations to create decision boundaries that a single linear map cannot represent.",
        isCorrect: true,
      },
      {
        text: "Depth alone makes a purely linear network nonlinear.",
        isCorrect: false,
      },
      {
        text: "Nonlinearities remove the need for matrix multiplication in deep networks.",
        isCorrect: false,
      },
    ],
    explanation:
      "A stack of linear transformations still represents one linear transformation if no nonlinear operation intervenes. Nonlinearities such as ReLU or GELU let the network bend and reshape representation space. They complement matrix multiplication rather than replacing it.",
  },
  {
    id: "la-crash-l5-q47",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statement best describes a reinforcement-learning value function \\(Q(s,a)\\)?",
    options: [
      {
        text: "It estimates expected future reward for taking action \\(a\\) in state \\(s\\).",
        isCorrect: true,
      },
      {
        text: "It is the same object as the transformer query matrix \\(Q\\).",
        isCorrect: false,
      },
      {
        text: "It stores the immediate reward and ignores future outcomes.",
        isCorrect: false,
      },
      {
        text: "It is unrelated to function approximation in modern deep RL.",
        isCorrect: false,
      },
    ],
    explanation:
      "The notation \\(Q(s,a)\\) in reinforcement learning refers to an action-value function, not to a transformer query matrix. It estimates future return after choosing an action in a state. Deep RL often approximates this function with neural networks built from linear algebra operations.",
  },
  {
    id: "la-crash-l5-q48",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Let \\(q=\\begin{bmatrix}1 \\\\ 2\\end{bmatrix}\\), \\(k_1=\\begin{bmatrix}2 \\\\ 0\\end{bmatrix}\\), and \\(k_2=\\begin{bmatrix}0 \\\\ 3\\end{bmatrix}\\). Which statements correctly describe dot-product attention scores?",
    options: [
      {
        text: "The score with \\(k_1\\) is \\(2\\).",
        isCorrect: true,
      },
      {
        text: "The score with \\(k_2\\) is \\(6\\).",
        isCorrect: true,
      },
      {
        text: "Before softmax, \\(k_2\\) has the higher dot-product score.",
        isCorrect: true,
      },
      {
        text: "The scores are Euclidean distances, so smaller values mean stronger attention.",
        isCorrect: false,
      },
    ],
    explanation:
      "The dot products are \\(1\\cdot2+2\\cdot0=2\\) and \\(1\\cdot0+2\\cdot3=6\\). A larger dot product indicates stronger alignment before softmax. Dot-product attention is not based on Euclidean distance in this formulation.",
  },
  {
    id: "la-crash-l5-q49",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "A sequence has \\(n=5\\) tokens, attention head dimension \\(d=8\\), and value dimension \\(d_v=6\\). If \\(Q,K\\in\\mathbb{R}^{5\\times8}\\) and \\(V\\in\\mathbb{R}^{5\\times6}\\), which statements are correct?",
    options: [
      {
        text: "\\(QK^T\\) has shape \\(5\\times5\\).",
        isCorrect: true,
      },
      {
        text: "The row-wise softmax attention matrix has shape \\(5\\times5\\).",
        isCorrect: true,
      },
      {
        text: "The final attention output has shape \\(5\\times6\\).",
        isCorrect: true,
      },
      {
        text: "The projection matrices that created \\(Q\\), \\(K\\), and \\(V\\) are learned transformations.",
        isCorrect: true,
      },
    ],
    explanation:
      "The score matrix compares every token to every token, giving a \\(5\\times5\\) matrix. Multiplying those weights by \\(V\\) keeps one output row per token and uses the value dimension \\(6\\). These shapes are a concrete example of transformer attention as matrix multiplication.",
  },
  {
    id: "la-crash-l5-q50",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "For the value-function loss \\(L=(Q_{target}-Q_\\theta)^2\\), suppose \\(Q_{target}=10\\) and \\(Q_\\theta=7\\). Which statements are correct?",
    options: [
      {
        text: "The loss value is \\(9\\).",
        isCorrect: true,
      },
      {
        text: "Gradient descent on \\(Q_\\theta\\) would locally push the prediction upward.",
        isCorrect: true,
      },
      {
        text: "The derivative with respect to \\(Q_\\theta\\) is \\(+6\\).",
        isCorrect: false,
      },
      {
        text: "The loss is zero because the target and prediction are both positive.",
        isCorrect: false,
      },
    ],
    explanation:
      "The prediction error is \\(10-7=3\\), so the squared loss is \\(9\\). The derivative with respect to \\(Q_\\theta\\) is \\(2(Q_\\theta-Q_{target})=-6\\), so subtracting the gradient increases the prediction. Positivity alone does not make the loss zero.",
  },
  {
    id: "la-crash-l5-q51",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "For the loss surface \\(f(x,y)=x^2-y^2\\), which statement correctly describes the point \\((0,0)\\)?",
    options: [
      {
        text: "It is a saddle point because the gradient is zero but the surface curves upward in one direction and downward in another.",
        isCorrect: true,
      },
      {
        text: "It is a strict local minimum because nearby directions increase the loss.",
        isCorrect: false,
      },
      {
        text: "It is a strict local maximum because nearby directions decrease the loss.",
        isCorrect: false,
      },
      {
        text: "It is not a critical point because the function has two variables.",
        isCorrect: false,
      },
    ],
    explanation:
      "At \\((0,0)\\), both partial derivatives are zero. Along the \\(x\\)-axis the function is positive, while along the \\(y\\)-axis it is negative, so the point is neither a local minimum nor a local maximum. This is the basic geometry of a saddle point.",
  },
  {
    id: "la-crash-l5-q52",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe high-dimensional optimization landscapes?",
    options: [
      {
        text: "There can be many directions in which the loss changes differently.",
        isCorrect: true,
      },
      {
        text: "Saddle points can be common and can slow optimization.",
        isCorrect: true,
      },
      {
        text: "Gradients provide local directions for improvement, not a full map of the entire landscape.",
        isCorrect: true,
      },
      {
        text: "A critical point in a high-dimensional neural network is treated as a bad local minimum.",
        isCorrect: false,
      },
    ],
    explanation:
      "High-dimensional parameter spaces have many directions, so curvature and slope can vary across them. Saddle points can have zero gradient while still offering downhill escape directions. Gradients are local signals, not complete global descriptions of the loss surface.",
  },
  {
    id: "la-crash-l5-q53",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly support the view that modern AI is high-dimensional geometric optimization?",
    options: [
      {
        text: "Inputs such as text, images, audio, states, and actions can be represented as vectors.",
        isCorrect: true,
      },
      {
        text: "Model layers often apply learned matrix transformations to those vectors.",
        isCorrect: true,
      },
      {
        text: "Learning adjusts parameters using gradients of loss functions.",
        isCorrect: true,
      },
      {
        text: "Similarity and routing often depend on dot products or related geometric comparisons.",
        isCorrect: true,
      },
    ],
    explanation:
      "Modern AI systems convert many data types into vector representations and then transform those representations with learned matrices. Training uses gradients to improve loss functions over very high-dimensional parameter spaces. Attention, retrieval, and embeddings also rely heavily on geometric similarity.",
  },
  {
    id: "la-crash-l5-q54",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Suppose every row of \\(Q\\) is identical while \\(K\\) is fixed. Which statements about \\(QK^T\\) and attention are correct before any masking is applied?",
    options: [
      {
        text: "All rows of \\(QK^T\\) are identical.",
        isCorrect: true,
      },
      {
        text: "Row-wise softmax gives identical attention-weight rows.",
        isCorrect: true,
      },
      {
        text: "The values matrix \\(V\\) determines the query-key scores.",
        isCorrect: false,
      },
      {
        text: "Identical scores force attention to choose exactly one token with probability \\(1\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "If every query row is the same, then each row takes the same set of dot products with the key rows. Applying the same row-wise softmax to identical score rows gives identical attention distributions. The values are mixed after scores are computed, and softmax remains a soft distribution unless scores become extreme.",
  },
  {
    id: "la-crash-l5-q55",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "If the number of tokens \\(n\\) doubles while the attention head dimension \\(d\\) stays fixed, which statement best describes the raw score matrix \\(QK^T\\)?",
    options: [
      {
        text: "Its number of entries grows by about a factor of \\(4\\) because the matrix is \\(n\\times n\\).",
        isCorrect: true,
      },
      {
        text: "Its number of entries grows by about a factor of \\(2\\) because attention is modeled as linear in sequence length.",
        isCorrect: false,
      },
      {
        text: "Its shape remains \\(d\\times d\\) because \\(d\\) is fixed.",
        isCorrect: false,
      },
      {
        text: "Its size is independent of the number of tokens.",
        isCorrect: false,
      },
    ],
    explanation:
      "The score matrix compares every token query with every token key, so it has \\(n^2\\) entries. Doubling \\(n\\) changes \\(n^2\\) to \\((2n)^2=4n^2\\). This shape fact is one reason attention cost is sensitive to sequence length.",
  },
  {
    id: "la-crash-l5-q56",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe a transformer block as a sequence of geometric operations?",
    options: [
      {
        text: "Linear projections create query, key, and value spaces from token embeddings.",
        isCorrect: true,
      },
      {
        text: "Dot products route information by comparing directions in learned spaces.",
        isCorrect: true,
      },
      {
        text: "Feed-forward nonlinearities help reshape representations beyond a single linear map.",
        isCorrect: true,
      },
      {
        text: "The block performs symbolic rule lookup instead of differentiable vector operations.",
        isCorrect: false,
      },
    ],
    explanation:
      "Transformer computations are built from learned linear projections, dot products, softmax weighting, and nonlinear feed-forward transformations. These operations reshape and route vectors in representation space. They are differentiable geometric operations rather than explicit symbolic database lookup.",
  },
  {
    id: "la-crash-l5-q57",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly connect reinforcement-learning states to vector representations?",
    options: [
      {
        text: "Robot sensor readings can be represented as state vectors.",
        isCorrect: true,
      },
      {
        text: "A game board can be encoded as a vector or structured tensor before being processed by a value network.",
        isCorrect: true,
      },
      {
        text: "A language context can be represented by embeddings when an RL-style objective is used with language models.",
        isCorrect: true,
      },
      {
        text: "A value network can transform state representations into estimates for actions.",
        isCorrect: true,
      },
    ],
    explanation:
      "Deep RL relies on numerical representations of states so neural networks can process them. Those representations may come from sensors, game encodings, or language embeddings depending on the environment. Value or policy networks then use matrix transformations and nonlinearities to produce decision-relevant outputs.",
  },
  {
    id: "la-crash-l5-q58",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe saddle-point intuition in optimization?",
    options: [
      {
        text: "A saddle point can have uphill directions and downhill directions nearby.",
        isCorrect: true,
      },
      {
        text: "In high dimensions, there may be many possible directions for escaping a saddle-like region.",
        isCorrect: true,
      },
      {
        text: "A zero-gradient point is a local minimum.",
        isCorrect: false,
      },
      {
        text: "Large models eliminate the need to choose a learning rate.",
        isCorrect: false,
      },
    ],
    explanation:
      "A saddle point is not simply a minimum; it can curve upward in some directions and downward in others. High-dimensional spaces often contain many directions, which changes optimization behavior compared with low-dimensional pictures. Learning rate still matters because gradient steps must have a usable size.",
  },
  {
    id: "la-crash-l5-q59",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statement most accurately compares transformer attention and deep RL value approximation?",
    options: [
      {
        text: "Both use vector representations and learned matrix transformations, but attention routes token information while value approximation estimates future reward.",
        isCorrect: true,
      },
      {
        text: "They are the same algorithm because both use the symbol \\(Q\\).",
        isCorrect: false,
      },
      {
        text: "Attention uses string matching, while value approximation uses dot products.",
        isCorrect: false,
      },
      {
        text: "Deep RL does not use gradients when value functions are represented by neural networks.",
        isCorrect: false,
      },
    ],
    explanation:
      "The letter \\(Q\\) is overloaded: transformer queries and reinforcement-learning action values are different objects. Both systems still depend on vector spaces, matrix transformations, and gradient-based learning in modern neural versions. Their objectives and roles are different even though the underlying linear algebra tools overlap.",
  },
  {
    id: "la-crash-l5-q60",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly synthesize the linear-algebra themes behind LLMs, deep learning, and RL?",
    options: [
      {
        text: "Low-rank structure helps explain why some large matrices or updates can be compressed.",
        isCorrect: true,
      },
      {
        text: "Gradients adjust learned matrices and representations to reduce task losses.",
        isCorrect: true,
      },
      {
        text: "Covariance, PCA, and SVD help analyze structure in learned representation spaces.",
        isCorrect: true,
      },
      {
        text: "Dot products are mainly a text-model tool rather than a retrieval, attention, or RL feature.",
        isCorrect: false,
      },
    ],
    explanation:
      "The same mathematical tools recur across AI systems: vectors represent data, matrices transform it, gradients train the transformations, and decompositions reveal structure. Low-rank and covariance ideas help explain compression and representation organization. Dot products appear broadly in similarity, routing, and feature interactions, not only in text models.",
  },
];

export const CrashCourseLinearAlgebraL5Questions =
  CrashCourseLinearAlgebraLecture5Questions;
