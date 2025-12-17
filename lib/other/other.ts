import { Question } from "../quiz";

export const mixedQuestions: Question[] = [
  // ============================================================
  //  Q1–Q8: ALL TRUE (4 correct)
  // ============================================================

  {
    id: "infra-q01",
    chapter: 0,
    difficulty: "easy",
    prompt: "Which statements correctly describe what a Floating-Point Operation (FLOP) represents in machine learning workloads?",
    options: [
      { text: "A Floating-Point Operation represents a basic arithmetic computation such as addition or multiplication on floating-point numbers.", isCorrect: true },
      { text: "Large neural networks require extremely large numbers of Floating-Point Operations during training.", isCorrect: true },
      { text: "Matrix multiplications in neural networks are composed of many Floating-Point Operations.", isCorrect: true },
      { text: "Floating-Point Operations are a useful abstraction for estimating computational cost.", isCorrect: true },
    ],
    explanation: "FLOPs quantify raw mathematical work. Neural networks rely heavily on matrix operations, which expand into huge numbers of floating-point computations."
  },

  {
    id: "infra-q02",
    chapter: 0,
    difficulty: "easy",
    prompt: "Which statements describe why Graphics Processing Units (GPUs) are well suited for machine learning?",
    options: [
      { text: "They can perform many arithmetic operations in parallel.", isCorrect: true },
      { text: "They are optimized for dense linear algebra workloads such as matrix multiplication.", isCorrect: true },
      { text: "They provide high memory bandwidth compared to traditional Central Processing Units.", isCorrect: true },
      { text: "They can accelerate both training and inference workloads.", isCorrect: true },
    ],
    explanation: "GPUs were designed for massively parallel numerical computation, which closely matches the computational patterns of neural networks."
  },

  {
    id: "infra-q03",
    chapter: 0,
    difficulty: "easy",
    prompt: "Which statements describe the role of cloud providers in the AI infrastructure ecosystem?",
    options: [
      { text: "They rent access to large-scale compute resources such as GPUs and specialized accelerators.", isCorrect: true },
      { text: "They manage provisioning, scaling, and reliability of compute infrastructure.", isCorrect: true },
      { text: "They abstract hardware complexity behind application programming interfaces.", isCorrect: true },
      { text: "They enable organizations to run large models without owning physical hardware.", isCorrect: true },
    ],
    explanation: "Cloud providers turn capital-intensive hardware into on-demand services and handle operational complexity for users."
  },

  {
    id: "infra-q04",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements correctly describe memory bandwidth in AI accelerators?",
    options: [
      { text: "Memory bandwidth measures how fast data can be moved between memory and compute units.", isCorrect: true },
      { text: "Insufficient memory bandwidth can cause compute units to remain idle.", isCorrect: true },
      { text: "Attention-heavy models are often limited by memory bandwidth rather than raw compute.", isCorrect: true },
      { text: "High-bandwidth memory technologies are critical for modern accelerators.", isCorrect: true },
    ],
    explanation: "Many machine learning workloads are memory-bound, meaning data movement limits performance more than arithmetic capability."
  },

  {
    id: "infra-q05",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements describe why interconnect technology matters in large-scale model training?",
    options: [
      { text: "Large models are often split across multiple accelerators.", isCorrect: true },
      { text: "Slow interconnects increase synchronization and communication overhead.", isCorrect: true },
      { text: "High-speed interconnects allow accelerators to behave like a single logical device.", isCorrect: true },
      { text: "Interconnect performance directly affects overall training efficiency.", isCorrect: true },
    ],
    explanation: "Distributed training relies on frequent data exchange; weak interconnects waste expensive compute by forcing devices to wait."
  },

  {
    id: "infra-q06",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements describe why NVIDIA’s ecosystem has become dominant in AI infrastructure?",
    options: [
      { text: "Its Compute Unified Device Architecture (CUDA) platform is widely supported by machine learning frameworks.", isCorrect: true },
      { text: "It offers mature libraries, compilers, and debugging tools.", isCorrect: true },
      { text: "Most production machine learning workloads are optimized for its software stack.", isCorrect: true },
      { text: "Its hardware and software are tightly co-designed.", isCorrect: true },
    ],
    explanation: "NVIDIA’s advantage is driven as much by software and ecosystem lock-in as by raw hardware performance."
  },

  {
    id: "infra-q07",
    chapter: 0,
    difficulty: "hard",
    prompt: "Which statements correctly characterize the relationship between Floating-Point Operations per second (FLOP/s) and real-world performance?",
    options: [
      { text: "Peak FLOP/s represents a theoretical maximum rather than guaranteed performance.", isCorrect: true },
      { text: "Actual utilization depends on software, memory bandwidth, and workload structure.", isCorrect: true },
      { text: "Two systems with similar peak FLOP/s can have very different effective throughput.", isCorrect: true },
      { text: "Idle FLOPs represent wasted computational investment.", isCorrect: true },
    ],
    explanation: "Headline FLOP/s numbers are only meaningful when paired with utilization, memory efficiency, and software quality."
  },

  {
    id: "infra-q08",
    chapter: 0,
    difficulty: "hard",
    prompt: "Which statements describe why large language model training is extremely resource intensive?",
    options: [
      { text: "Training involves repeated forward and backward passes through large networks.", isCorrect: true },
      { text: "The number of Floating-Point Operations scales with both model size and token count.", isCorrect: true },
      { text: "Training requires storing intermediate activations for gradient computation.", isCorrect: true },
      { text: "Training often requires distributed computation across many accelerators.", isCorrect: true },
    ],
    explanation: "Training cost grows rapidly due to scaling laws involving parameters, tokens, and optimization requirements."
  },

  // ============================================================
  //  Q9–Q16: THREE TRUE
  // ============================================================

  {
    id: "infra-q09",
    chapter: 0,
    difficulty: "easy",
    prompt: "Which statements describe Tensor Processing Units (TPUs)?",
    options: [
      { text: "They are specialized accelerators designed primarily for machine learning workloads.", isCorrect: true },
      { text: "They are most commonly used within Google’s internal infrastructure.", isCorrect: true },
      { text: "They can be accessed externally through Google Cloud services.", isCorrect: true },
      { text: "They are designed to run arbitrary non-numerical workloads efficiently.", isCorrect: false },
    ],
    explanation: "TPUs are optimized for dense numerical computation and are not general-purpose processors."
  },

  {
    id: "infra-q10",
    chapter: 0,
    difficulty: "easy",
    prompt: "Which statements describe the role of inference in deployed machine learning systems?",
    options: [
      { text: "Inference refers to running a trained model to generate outputs.", isCorrect: true },
      { text: "Inference workloads often prioritize low latency and cost efficiency.", isCorrect: true },
      { text: "Inference typically requires fewer Floating-Point Operations than training.", isCorrect: true },
      { text: "Inference always requires distributed multi-node computation.", isCorrect: false },
    ],
    explanation: "Inference is often latency- and cost-sensitive and may run on a single accelerator."
  },

  {
    id: "infra-q11",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements describe the NVIDIA Blackwell architecture?",
    options: [
      { text: "It was announced after the Hopper architecture.", isCorrect: true },
      { text: "It targets both large-scale training and efficient inference.", isCorrect: true },
      { text: "It introduces improvements in memory and interconnect capabilities.", isCorrect: true },
      { text: "It has completely replaced all previous GPU architectures worldwide.", isCorrect: false },
    ],
    explanation: "Blackwell is new and still rolling out; Hopper remains widely deployed."
  },

  {
    id: "infra-q12",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements describe why lower numerical precision is attractive in AI workloads?",
    options: [
      { text: "Lower precision reduces memory usage per number.", isCorrect: true },
      { text: "Lower precision allows more operations per second on the same hardware.", isCorrect: true },
      { text: "Lower precision can reduce energy consumption.", isCorrect: true },
      { text: "Lower precision guarantees identical numerical accuracy in all models.", isCorrect: false },
    ],
    explanation: "Reduced precision improves efficiency but may introduce numerical trade-offs."
  },

  {
    id: "infra-q13",
    chapter: 0,
    difficulty: "hard",
    prompt: "Which statements describe the economic risks of underutilized AI infrastructure?",
    options: [
      { text: "Idle accelerators represent wasted capital expenditure.", isCorrect: true },
      { text: "Low utilization increases cost per inference or training step.", isCorrect: true },
      { text: "Inefficient workloads can negate theoretical hardware advantages.", isCorrect: true },
      { text: "Underutilization has no effect once hardware is purchased.", isCorrect: false },
    ],
    explanation: "The economics of AI infrastructure strongly depend on keeping expensive hardware busy."
  },

  {
    id: "infra-q14",
    chapter: 0,
    difficulty: "hard",
    prompt: "Which statements describe why agent-based systems stress infrastructure differently than training jobs?",
    options: [
      { text: "Agent systems often involve many short, bursty model calls.", isCorrect: true },
      { text: "They frequently involve input/output operations and tool calls.", isCorrect: true },
      { text: "They are often latency-bound rather than throughput-bound.", isCorrect: true },
      { text: "They consistently saturate GPUs with dense matrix multiplications.", isCorrect: false },
    ],
    explanation: "Agent workloads rarely resemble the dense, continuous computation of training loops."
  },

  {
    id: "infra-q15",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements describe the role of systems software such as compilers and runtimes?",
    options: [
      { text: "They translate high-level model descriptions into efficient hardware execution.", isCorrect: true },
      { text: "They manage memory allocation and kernel scheduling.", isCorrect: true },
      { text: "They influence how much of peak FLOP/s can actually be achieved.", isCorrect: true },
      { text: "They are irrelevant once hardware performance is high enough.", isCorrect: false },
    ],
    explanation: "Software quality is a major determinant of real-world accelerator performance."
  },

  {
    id: "infra-q16",
    chapter: 0,
    difficulty: "hard",
    prompt: "Which statements describe why hyperscalers invest in custom accelerators?",
    options: [
      { text: "They aim to reduce dependence on third-party hardware vendors.", isCorrect: true },
      { text: "They can optimize chips for their own dominant workloads.", isCorrect: true },
      { text: "They seek better performance per dollar or per watt.", isCorrect: true },
      { text: "They eliminate the need for general-purpose GPUs entirely.", isCorrect: false },
    ],
    explanation: "Custom accelerators complement rather than fully replace GPUs in most ecosystems."
  },

  // ============================================================
  //  Q17–Q24: TWO TRUE
  // ============================================================

  {
    id: "infra-q17",
    chapter: 0,
    difficulty: "easy",
    prompt: "Which statements correctly describe Floating-Point Operations per second (FLOP/s)?",
    options: [
      { text: "It measures how many floating-point operations a system can perform per second.", isCorrect: true },
      { text: "It directly guarantees application-level latency.", isCorrect: false },
      { text: "It is often reported in trillions or quadrillions per second.", isCorrect: true },
      { text: "It fully captures memory and communication bottlenecks.", isCorrect: false },
    ],
    explanation: "FLOP/s measures compute throughput, not overall system efficiency."
  },

  {
    id: "infra-q18",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements describe differences between training and inference workloads?",
    options: [
      { text: "Training typically requires gradient computation and parameter updates.", isCorrect: true },
      { text: "Inference generally requires storing all intermediate activations for backpropagation.", isCorrect: false },
      { text: "Inference often prioritizes latency and cost over peak throughput.", isCorrect: true },
      { text: "Training always runs on a single accelerator.", isCorrect: false },
    ],
    explanation: "Training and inference differ substantially in computation patterns and optimization goals."
  },

  {
    id: "infra-q19",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements describe why memory can be a bottleneck in large language models?",
    options: [
      { text: "Model parameters must be repeatedly read from memory.", isCorrect: true },
      { text: "Attention mechanisms require accessing large key-value caches.", isCorrect: true },
      { text: "Memory access speed always scales with compute speed.", isCorrect: false },
      { text: "Memory usage is irrelevant once a model fits on a device.", isCorrect: false },
    ],
    explanation: "Data movement, not arithmetic, often limits throughput in large models."
  },

  {
    id: "infra-q20",
    chapter: 0,
    difficulty: "hard",
    prompt: "Which statements describe the role of networking in multi-node AI systems?",
    options: [
      { text: "It enables synchronization across distributed accelerators.", isCorrect: true },
      { text: "Slow networks can negate gains from faster accelerators.", isCorrect: true },
      { text: "Networking is only relevant for inference workloads.", isCorrect: false },
      { text: "Networking overhead disappears at large batch sizes.", isCorrect: false },
    ],
    explanation: "Distributed systems are constrained by communication as much as computation."
  },

  {
    id: "infra-q21",
    chapter: 0,
    difficulty: "easy",
    prompt: "Which statements describe the role of specialized inference engines?",
    options: [
      { text: "They optimize execution for serving trained models efficiently.", isCorrect: true },
      { text: "They can apply quantization and batching techniques.", isCorrect: true },
      { text: "They replace the need for model training entirely.", isCorrect: false },
      { text: "They eliminate memory constraints automatically.", isCorrect: false },
    ],
    explanation: "Inference engines focus on efficiency and scalability, not learning."
  },

  {
    id: "infra-q22",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements describe why peak hardware specifications can be misleading?",
    options: [
      { text: "They assume ideal workloads and perfect utilization.", isCorrect: true },
      { text: "They ignore software inefficiencies.", isCorrect: true },
      { text: "They directly measure end-to-end application performance.", isCorrect: false },
      { text: "They fully account for communication overhead.", isCorrect: false },
    ],
    explanation: "Real workloads rarely match the assumptions behind peak metrics."
  },

  {
    id: "infra-q23",
    chapter: 0,
    difficulty: "hard",
    prompt: "Which statements describe why agentic systems challenge traditional GPU utilization?",
    options: [
      { text: "They involve frequent control-flow branching.", isCorrect: true },
      { text: "They often wait on external tools or databases.", isCorrect: true },
      { text: "They continuously stream dense tensors at full throughput.", isCorrect: false },
      { text: "They guarantee predictable execution patterns.", isCorrect: false },
    ],
    explanation: "Irregular workloads reduce sustained utilization of accelerators."
  },

  {
    id: "infra-q24",
    chapter: 0,
    difficulty: "hard",
    prompt: "Which statements describe the relationship between FLOPs and cost?",
    options: [
      { text: "Higher FLOP requirements generally increase compute cost.", isCorrect: true },
      { text: "Reducing FLOPs can lower infrastructure expenses.", isCorrect: true },
      { text: "FLOPs are independent of hardware pricing.", isCorrect: false },
      { text: "Cost is unaffected by utilization efficiency.", isCorrect: false },
    ],
    explanation: "Cost scales with both computational demand and how efficiently hardware is used."
  },

  // ============================================================
  //  Q25–Q30: ONE TRUE
  // ============================================================

  {
    id: "infra-q25",
    chapter: 0,
    difficulty: "easy",
    prompt: "Which statement best describes the primary purpose of interconnect technologies such as NVLink?",
    options: [
      { text: "To connect accelerators so they can efficiently share data.", isCorrect: true },
      { text: "To replace on-chip memory entirely.", isCorrect: false },
      { text: "To eliminate the need for software synchronization.", isCorrect: false },
      { text: "To increase numerical precision automatically.", isCorrect: false },
    ],
    explanation: "Interconnects enable fast communication between devices, not automatic computation improvements."
  },

  {
    id: "infra-q26",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statement correctly characterizes why Google primarily uses TPUs internally?",
    options: [
      { text: "They are tightly integrated with Google’s software stack and workloads.", isCorrect: true },
      { text: "They are universally faster than all GPUs for every workload.", isCorrect: false },
      { text: "They eliminate the need for distributed training.", isCorrect: false },
      { text: "They require no specialized compilers or frameworks.", isCorrect: false },
    ],
    explanation: "TPUs are optimized for Google’s internal ecosystem rather than general-purpose use."
  },

  {
    id: "infra-q27",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statement best explains why training adoption of new hardware precedes inference adoption?",
    options: [
      { text: "Training benefits most from higher throughput and can tolerate higher latency.", isCorrect: true },
      { text: "Inference never benefits from hardware improvements.", isCorrect: false },
      { text: "Inference workloads do not use accelerators.", isCorrect: false },
      { text: "Training hardware does not require software support.", isCorrect: false },
    ],
    explanation: "Training workloads can exploit raw throughput earlier than latency-sensitive inference."
  },

  {
    id: "infra-q28",
    chapter: 0,
    difficulty: "hard",
    prompt: "Which statement best captures why FLOPs alone do not define system performance?",
    options: [
      { text: "Memory, interconnect, and software efficiency heavily influence real throughput.", isCorrect: true },
      { text: "FLOPs measure energy consumption directly.", isCorrect: false },
      { text: "FLOPs determine correctness of model outputs.", isCorrect: false },
      { text: "FLOPs automatically include input/output costs.", isCorrect: false },
    ],
    explanation: "Compute capacity is only one component of end-to-end performance."
  },

  {
    id: "infra-q29",
    chapter: 0,
    difficulty: "hard",
    prompt: "Which statement best explains why idle FLOPs are considered economically damaging?",
    options: [
      { text: "They represent paid-for compute capacity that produces no value.", isCorrect: true },
      { text: "They permanently damage hardware.", isCorrect: false },
      { text: "They increase numerical error rates.", isCorrect: false },
      { text: "They improve long-term utilization.", isCorrect: false },
    ],
    explanation: "Idle accelerators still incur capital and operational costs without delivering output."
  },

  {
    id: "infra-q30",
    chapter: 0,
    difficulty: "easy",
    prompt: "Which statement best summarizes the role of AI infrastructure within the broader AI industry?",
    options: [
      { text: "It provides the computational foundation on which models and applications are built.", isCorrect: true },
      { text: "It replaces the need for machine learning models.", isCorrect: false },
      { text: "It determines application logic and user interfaces.", isCorrect: false },
      { text: "It guarantees superior model quality regardless of design.", isCorrect: false },
    ],
    explanation: "Infrastructure enables AI systems but does not replace model or application design."
  },

  // ============================================================
  //  Q1–Q5: 4 correct answers (ALL TRUE)
  // ============================================================

  {
    id: "other-q01",
    chapter: 0,
    difficulty: "easy",
    prompt: "Which statements correctly describe entropy in the context of language models?",
    options: [
      { text: "Entropy is a single numeric summary of how uncertain a probability distribution is.", isCorrect: true },
      { text: "Entropy is computed from the full distribution over possible next tokens, not just from one probability.", isCorrect: true },
      { text: "Higher entropy means the model spreads probability mass over many tokens and is less certain.", isCorrect: true },
      { text: "Lower entropy means the model concentrates probability mass on fewer tokens and is more certain.", isCorrect: true },
    ],
    explanation:
      "Entropy is defined on the entire probability distribution and measures expected uncertainty: high when probabilities are spread out, low when they are concentrated on a few outcomes."
  },

  {
    id: "other-q02",
    chapter: 0,
    difficulty: "easy",
    prompt: "Which statements about bits as a unit of information are accurate?",
    options: [
      { text: "A bit is the information needed to distinguish between two equally likely alternatives.", isCorrect: true },
      { text: "Bits measure how many binary (yes/no) questions you need on average to identify an outcome.", isCorrect: true },
      { text: "When entropy is measured in bits, 1 bit corresponds to the uncertainty of a fair coin flip.", isCorrect: true },
      { text: "Bits quantify uncertainty or information, not the physical storage size of a particular computer.", isCorrect: true },
    ],
    explanation:
      "A bit is defined in information theory as the uncertainty in a fair binary choice. Entropy in bits tells you how many such binary decisions are needed on average to resolve uncertainty."
  },

  {
    id: "other-q03",
    chapter: 0,
    difficulty: "easy",
    prompt: "Which statements correctly distinguish a probability distribution from entropy?",
    options: [
      { text: "A probability distribution can be represented as a vector of probabilities over all tokens.", isCorrect: true },
      { text: "Entropy is a single scalar value computed from the entire probability distribution.", isCorrect: true },
      { text: "Two different distributions can have the same entropy even if their probabilities differ token by token.", isCorrect: true },
      { text: "Knowing the entropy alone is not enough to reconstruct the full probability distribution.", isCorrect: true },
    ],
    explanation:
      "The distribution is a full vector of probabilities, while entropy compresses that uncertainty into one scalar. Different distributions can share the same entropy, so entropy alone cannot reconstruct the distribution."
  },

  {
    id: "other-q04",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements correctly describe entropy as expected uncertainty?",
    options: [
      { text: "Entropy averages the surprisal of each possible outcome, weighted by its probability.", isCorrect: true },
      { text: "Surprisal of an outcome is typically defined as minus the logarithm of its probability.", isCorrect: true },
      { text: "Very rare events can have high surprisal but contribute little to entropy if their probability is tiny.", isCorrect: true },
      { text: "Entropy reflects the average unpredictability of outcomes drawn from a distribution.", isCorrect: true },
    ],
    explanation:
      "Entropy H(p) = −∑ p(x) log p(x) is the expectation of surprisal −log p(x). Rare events have high surprisal but are weighted by small probabilities, so entropy captures average unpredictability."
  },

  {
    id: "other-q05",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements about sampling temperature and its interaction with entropy are correct?",
    options: [
      { text: "Temperature rescales logits before the softmax, changing the shape of the predicted distribution.", isCorrect: true },
      { text: "Higher temperature makes the distribution flatter and generally increases entropy.", isCorrect: true },
      { text: "Lower temperature makes the distribution more peaked and generally decreases entropy.", isCorrect: true },
      { text: "Temperature can be seen as a way to artificially adjust the model’s effective uncertainty during sampling.", isCorrect: true },
    ],
    explanation:
      "Dividing logits by temperature T before softmax flattens the distribution for T>1 and sharpens it for T<1, which respectively increase or decrease entropy during sampling."
  },

  // ============================================================
  //  Q6–Q10: 3 correct answers, 1 incorrect (3 TRUE)
  // ============================================================

  {
    id: "other-q06",
    chapter: 0,
    difficulty: "easy",
    prompt: "Which statements about the relationship between bits and Booleans are accurate?",
    options: [
      { text: "A Boolean variable can be represented using one bit in a computer system.", isCorrect: true },
      { text: "Bits in information theory quantify uncertainty, not just stored values in memory.", isCorrect: true },
      { text: "A bit is conceptually the information content of resolving one fair binary choice.", isCorrect: true },
      { text: "Bits and Booleans are the same concept, just with different names in different fields.", isCorrect: false },
    ],
    explanation:
      "Booleans use bits for storage, but bits in information theory represent the information gained when resolving uncertainty in a binary choice. They are related but not literally the same concept."
  },

  {
    id: "other-q07",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements correctly describe cross-entropy in the context of training language models?",
    options: [
      { text: "Cross-entropy compares a true distribution and a model’s predicted distribution.", isCorrect: true },
      { text: "When training with one-hot targets, cross-entropy reduces to the negative log probability of the correct token.", isCorrect: true },
      { text: "Minimizing cross-entropy during training encourages the model to assign higher probability to the correct tokens.", isCorrect: true },
      { text: "Cross-entropy loss is computed only from the entropy of the model’s predictions without reference to any targets.", isCorrect: false },
    ],
    explanation:
      "Cross-entropy H(p, q) = −∑ p(x) log q(x) uses the true distribution p and predicted distribution q. With one-hot labels, it is just −log q(correct), and minimizing it pushes probability mass onto correct tokens."
  },

  {
    id: "other-q08",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements about perplexity are correct?",
    options: [
      { text: "Perplexity is the exponential of the average negative log probability assigned to the correct tokens.", isCorrect: true },
      { text: "Perplexity can be interpreted as the effective number of equally likely choices the model is confused between.", isCorrect: true },
      { text: "Lower perplexity generally indicates that the model fits the evaluation data better under the language modeling objective.", isCorrect: true },
      { text: "Perplexity directly measures whether a model follows human instructions well in dialogue settings.", isCorrect: false },
    ],
    explanation:
      "Perplexity is exp of the average negative log likelihood and can be seen as an effective branching factor. It measures language modeling fit, not directly the quality of instruction-following behaviour."
  },

  {
    id: "other-q09",
    chapter: 0,
    difficulty: "hard",
    prompt: "Which statements correctly relate entropy and perplexity for next-token prediction?",
    options: [
      { text: "If entropy is measured in bits, then perplexity is 2 raised to the power of that entropy.", isCorrect: true },
      { text: "If the model’s distribution is uniform over N tokens, then perplexity equals N.", isCorrect: true },
      { text: "A model with higher entropy necessarily has lower perplexity on the same data.", isCorrect: false },
      { text: "Perplexity and entropy are unrelated; they are defined on different probability spaces.", isCorrect: false },
    ],
    explanation:
      "Perplexity is defined as base^entropy (2^H if H is in bits). For a uniform distribution over N outcomes, H = log2 N, so perplexity is N. Higher entropy implies higher, not lower, perplexity."
  },

  {
    id: "other-q10",
    chapter: 0,
    difficulty: "hard",
    prompt: "Which statements explain why perplexity is less meaningful as a primary metric for instruction-tuned language models?",
    options: [
      { text: "Instruction tuning shifts the objective toward matching human preferences rather than pure next-token likelihood.", isCorrect: true },
      { text: "Alignment methods like Reinforcement Learning from Human Feedback modify token probabilities in ways that are not optimized for low perplexity.", isCorrect: true },
      { text: "High-quality instruction-following behaviour does not always correspond to lower perplexity on generic text corpora.", isCorrect: true },
      { text: "Instruction-tuned models no longer produce probability distributions over tokens, so perplexity cannot be computed.", isCorrect: false },
    ],
    explanation:
      "Instruction tuning and preference-based alignment distort the distribution away from maximum likelihood objectives. Models still output probabilities, but perplexity no longer fully reflects their practical quality as assistants."
  },

  // ============================================================
  //  Q11–Q15: 2 correct answers, 2 incorrect (2 TRUE)
  // ============================================================

  {
    id: "other-q11",
    chapter: 0,
    difficulty: "easy",
    prompt: "Consider entropy for a fair six-sided die. Which statements are correct?",
    options: [
      { text: "The outcomes (1 through 6) are discrete, but the entropy value is a real number.", isCorrect: true },
      { text: "The entropy in bits for a fair six-sided die is log2(6), which is approximately 2.585 bits.", isCorrect: true },
      { text: "Entropy must always be an integer whenever the number of outcomes is an integer.", isCorrect: false },
      { text: "Entropy cannot be defined for discrete random variables like dice.", isCorrect: false },
    ],
    explanation:
      "Discrete outcomes can still have non-integer entropy; for a fair die it is log2(6)≈2.585 bits. Entropy is well-defined and widely used for discrete variables."
  },

  {
    id: "other-q12",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements correctly compare entropy, cross-entropy, and the model’s probability vector?",
    options: [
      { text: "The probability vector lists the probabilities for each possible token in the vocabulary.", isCorrect: true },
      { text: "Entropy is a single scalar derived from the probability vector that summarizes expected uncertainty.", isCorrect: true },
      { text: "Cross-entropy is computed without reference to any target distribution and only uses the model’s entropy.", isCorrect: false },
      { text: "Entropy and cross-entropy are identical whenever the model assigns non-zero probability to all tokens.", isCorrect: false },
    ],
    explanation:
      "The probability vector defines the full distribution; entropy compresses it into one number. Cross-entropy needs both a target distribution and a model distribution; only when they match does cross-entropy equal entropy."
  },

  {
    id: "other-q13",
    chapter: 0,
    difficulty: "medium",
    prompt: "In large language models, which statements about the fixed set of outcomes and entropy are correct?",
    options: [
      { text: "The model predicts the next token from a fixed vocabulary of discrete tokens.", isCorrect: true },
      { text: "Entropy measures uncertainty over this fixed set of possible tokens at a given position.", isCorrect: true },
      { text: "Entropy requires a continuously changing set of possible outcomes and cannot be defined on a fixed vocabulary.", isCorrect: false },
      { text: "Having a fixed vocabulary implies that entropy is always zero for next-token prediction.", isCorrect: false },
    ],
    explanation:
      "The vocabulary is fixed and discrete, and entropy is defined over the distribution on that set. A fixed set of outcomes does not imply zero entropy; uncertainty depends on how probabilities are distributed."
  },

  {
    id: "other-q14",
    chapter: 0,
    difficulty: "hard",
    prompt: "Which statements correctly describe surprisal and its relationship to entropy?",
    options: [
      { text: "Surprisal of an outcome is defined as minus the logarithm of its probability under a given distribution.", isCorrect: true },
      { text: "Entropy is the expected value of the surprisal across all possible outcomes.", isCorrect: true },
      { text: "Changing the logarithm base changes the ranking of entropies for different distributions.", isCorrect: false },
      { text: "Using a different log base changes the numerical value of entropy but not the underlying ordering of uncertainty levels.", isCorrect: false },
    ],
    explanation:
      "Surprisal is −log p(x), and entropy is E[−log p(x)]. Changing the log base rescales entropy values but preserves the ordering of distributions by uncertainty. Thus the third statement is incorrect and the fourth, as written, is also incorrect because it claims that changing base changes the ordering."
  },

  {
    id: "other-q15",
    chapter: 0,
    difficulty: "hard",
    prompt: "Which statements about alignment methods and their effect on entropy and perplexity are correct?",
    options: [
      { text: "Reinforcement Learning from Human Feedback can modify token probabilities to favor helpful or safe responses, even if this increases perplexity on generic text.", isCorrect: true },
      { text: "Instruction tuning datasets can shift the distribution toward instruction-like inputs and assistant-style outputs, changing entropy patterns compared to pure web text.", isCorrect: true },
      { text: "Because of alignment methods, it is impossible to compute cross-entropy loss for instruction-tuned models.", isCorrect: false },
      { text: "Alignment guarantees that lower perplexity will always correspond to better human-rated responses.", isCorrect: false },
    ],
    explanation:
      "Alignment methods reshape the distribution according to human preferences, sometimes at the cost of higher perplexity on generic corpora. They do not prevent computing cross-entropy, nor do they guarantee a monotonic link between perplexity and human ratings."
  },

  // ============================================================
  //  Q16–Q20: 1 correct answer, 3 incorrect (1 TRUE)
  // ============================================================

  {
    id: "other-q16",
    chapter: 0,
    difficulty: "easy",
    prompt: "Which single statement best characterizes bits in information theory?",
    options: [
      { text: "Bits are defined as the number of distinct tokens in the model’s vocabulary.", isCorrect: false },
      { text: "Bits are a measure of the physical size of a neural network’s weights in computer memory.", isCorrect: false },
      { text: "Bits quantify the information gained when resolving uncertainty in binary choices.", isCorrect: true },
      { text: "Bits are only relevant for discrete random variables and do not apply to probability distributions.", isCorrect: false },
    ],
    explanation:
      "In information theory, bits measure the information associated with resolving uncertainty in binary decisions. They are independent of vocabulary size or storage format and apply to probability distributions in general."
  },

  {
    id: "other-q17",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which single statement best explains why entropy is called expected uncertainty?",
    options: [
      { text: "Because entropy is defined as the maximum possible uncertainty over all probability distributions on a given set.", isCorrect: false },
      { text: "Because entropy equals the probability of the most likely outcome under the distribution.", isCorrect: false },
      { text: "Because entropy averages the uncertainty (surprisal) of each outcome, weighted by how likely that outcome is.", isCorrect: true },
      { text: "Because entropy measures only the uncertainty of the least likely event in the distribution.", isCorrect: false },
    ],
    explanation:
      "Entropy is the expectation of surprisal, −log p(x), under the distribution p. This averaging over outcomes justifies the interpretation as expected uncertainty."
  },

  {
    id: "other-q18",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which single statement best describes the effect of increasing sampling temperature on model outputs?",
    options: [
      { text: "It sharpens the distribution, making the most likely token even more dominant and reducing entropy.", isCorrect: false },
      { text: "It leaves the distribution over tokens unchanged, but only affects how logits are stored in memory.", isCorrect: false },
      { text: "It flattens the distribution, increasing entropy and typically producing more diverse and less predictable outputs.", isCorrect: true },
      { text: "It deterministically forces the model to always sample the least likely token.", isCorrect: false },
    ],
    explanation:
      "Higher temperature divides logits by a larger T, flattening the softmax distribution. This raises entropy and yields more varied, less predictable samples."
  },

  {
    id: "other-q19",
    chapter: 0,
    difficulty: "hard",
    prompt: "Which single statement best describes the relationship between perplexity and the training objective of a pure language model?",
    options: [
      { text: "Perplexity is directly derived from the cross-entropy loss and decreases as the model assigns higher probability to observed tokens.", isCorrect: true },
      { text: "Perplexity is unrelated to cross-entropy and instead measures only the size of the training dataset.", isCorrect: false },
      { text: "Perplexity increases whenever the model’s vocabulary size increases, regardless of the learned probabilities.", isCorrect: false },
      { text: "Perplexity is defined only for models trained without gradients and cannot be used during gradient-based optimization.", isCorrect: false },
    ],
    explanation:
      "For standard language models, cross-entropy loss and perplexity are monotonic transforms of each other: lowering cross-entropy corresponds to lowering perplexity by better matching the training data distribution."
  },

  {
    id: "other-q20",
    chapter: 0,
    difficulty: "hard",
    prompt: "Which single statement best captures why evaluation of instruction-tuned language models often uses metrics other than perplexity?",
    options: [
      { text: "Perplexity cannot be computed for any model that was fine-tuned on instruction-style data.", isCorrect: false },
      { text: "Instruction-tuned models output continuous vectors instead of token probabilities, so perplexity is undefined.", isCorrect: false },
      { text: "Human-oriented behaviours like helpfulness, safety, and following instructions are not fully captured by how well the model predicts next tokens on generic text.", isCorrect: true },
      { text: "Perplexity always increases after instruction tuning, so it is no longer meaningful for any purpose.", isCorrect: false },
    ],
    explanation:
      "Instruction-tuned models are optimized for human preferences, not pure likelihood. As a result, metrics like human ratings, task success, or specialized benchmarks are needed to complement or replace perplexity as primary evaluation signals."
  },
];
