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
        text: "They remove all trainable parameters from attention.",
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
        text: "It stores only the final predicted class label.",
        isCorrect: false,
      },
      {
        text: "It is unrelated to dot products.",
        isCorrect: false,
      },
      {
        text: "It replaces all keys and values with zeros.",
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
        text: "They are based only on alphabetical order of tokens.",
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
        text: "The result contains one row for every embedding dimension rather than every token.",
        isCorrect: false,
      },
      {
        text: "The transpose is unnecessary because \\(QK\\) already has shape \\(n\\times n\\).",
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
        text: "It deletes all value vectors before aggregation.",
        isCorrect: false,
      },
      {
        text: "It makes every token weight negative.",
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
        text: "All heads are forced to have identical learned projections.",
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
        text: "Nonlinearities are irrelevant to the expressivity of stacked layers.",
        isCorrect: false,
      },
      {
        text: "Neural networks never use matrix multiplication.",
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
        text: "Nonlinearities are needed because matrices cannot multiply vectors.",
        isCorrect: false,
      },
      {
        text: "Nonlinearities remove all learned parameters.",
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
        text: "Its inputs cannot be vectorized state or action information.",
        isCorrect: false,
      },
      {
        text: "It must be a handwritten table in all modern deep RL systems.",
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
        text: "It can only be used when there are no parameters \\(\\theta\\).",
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
        text: "Every critical point in a neural network is necessarily a bad local minimum.",
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
        text: "Bad local minima are the only optimization concern.",
        isCorrect: false,
      },
      {
        text: "High dimensions make gradients mathematically impossible.",
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
        text: "The routing is nondifferentiable and cannot be trained.",
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
        text: "Because matrix multiplication is undefined for all neural networks.",
        isCorrect: false,
      },
      {
        text: "Because \\(W_1\\) must always equal \\(W_2\\).",
        isCorrect: false,
      },
      {
        text: "Because multiplying by matrices removes all information in every case.",
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
        text: "Low-rank structure means all vectors are invalid.",
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
        text: "It is used only to make the notation look symmetric.",
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
        text: "Every layer must preserve all distances exactly.",
        isCorrect: false,
      },
      {
        text: "Representation geometry is irrelevant to prediction.",
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
        text: "Value functions cannot use vectors.",
        isCorrect: false,
      },
      {
        text: "Gradient descent cannot adjust RL networks.",
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
        text: "Similarity in modern AI never uses vector alignment.",
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
        text: "Model behavior comes only from a simple row-per-fact table.",
        isCorrect: false,
      },
      {
        text: "LLMs store all facts in a single human-readable lookup table.",
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
        text: "Attention avoids linear algebra by using only strings.",
        isCorrect: false,
      },
      {
        text: "Attention cannot be batched.",
        isCorrect: false,
      },
      {
        text: "Attention works only when there is exactly one token.",
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
];

export const CrashCourseLinearAlgebraL5Questions =
  CrashCourseLinearAlgebraLecture5Questions;
