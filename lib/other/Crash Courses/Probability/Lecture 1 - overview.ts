import { Question } from "../../../quiz";

export const CrashCourseProbabilityL1Questions: Question[] = [
  {
    id: "crash-probability-l1-q01",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Why is probability a natural language for many AI systems, including classifiers, Large Language Models (LLMs), reinforcement learning agents, and diffusion models?",
    options: [
      {
        text: "Inputs and labels can be incomplete, noisy, or ambiguous.",
        isCorrect: true,
      },
      {
        text: "Several outputs can be plausible before a final output is selected or sampled.",
        isCorrect: true,
      },
      {
        text: "Models can represent uncertainty by spreading probability across possible outcomes.",
        isCorrect: true,
      },
      {
        text: "Learning can adjust probability assignments based on observed data.",
        isCorrect: true,
      },
    ],
    explanation:
      "Probability is useful because AI systems often work before the true outcome is known and must compare several possible outcomes. Each option names a central reason probability appears in AI: uncertainty in inputs, multiple plausible outputs, model confidence, and learning from data.",
  },
  {
    id: "crash-probability-l1-q02",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Two image classifiers both choose cat. Model A assigns cat 0.99 and dog 0.01; Model B assigns cat 0.51 and dog 0.49. Which interpretations are correct?",
    options: [
      {
        text: "The displayed label alone hides that Model B is much less confident.",
        isCorrect: true,
      },
      {
        text: "Both models have the same probability distribution because they choose the same class.",
        isCorrect: false,
      },
      {
        text: "A maximum-probability decision rule can produce the same label from very different distributions.",
        isCorrect: true,
      },
      {
        text: "Model B is more confident because it keeps probability mass on more than one animal class.",
        isCorrect: false,
      },
    ],
    explanation:
      "Both models choose cat if the rule is to pick the largest probability, but their distributions communicate different confidence. Model B is nearly split between cat and dog, so treating the final label as the whole answer loses important uncertainty information.",
  },
  {
    id: "crash-probability-l1-q03",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "A next-token model gives probabilities: the 0.40, a 0.25, an 0.10, dog 0.05, other 0.20. Which statements correctly use the distribution?",
    options: [
      {
        text: "The listed sample space is the five possible next-token outcomes represented in the table.",
        isCorrect: true,
      },
      {
        text: "The probability of the event next token is an article is 0.75.",
        isCorrect: true,
      },
      {
        text: "The probabilities sum to 1.00, so the listed outcomes are exhaustive for this modeled step.",
        isCorrect: true,
      },
      {
        text: "The tokens the and a are not mutually exclusive because both are articles.",
        isCorrect: false,
      },
    ],
    explanation:
      "The sample space here is the set of listed possible next-token outcomes, and the article event groups the, a, and an for a total probability of 0.75. At one token position only one token is chosen, so the and a are mutually exclusive outcomes even though they belong to the same event category.",
  },
  {
    id: "crash-probability-l1-q04",
    chapter: 1,
    difficulty: "medium",
    type: "assertion-reason",
    prompt:
      "Assertion: A model that outputs \\(P(\\text{cat}) = 0.72\\) is making a probabilistic claim about a possible class rather than simply saying the image is cat.\n\nReason: The value 0.72 is the model's assigned probability for that class under its learned representation.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: false },
      { text: "Assertion is false, Reason is true.", isCorrect: false },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: true,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "The assertion is true because the model represents a distribution over classes, and the final displayed label may be a later decision. The reason is also true and explains the assertion: 0.72 is the model's assigned probability, not a statement that the object is literally 72 percent cat.",
  },
  {
    id: "crash-probability-l1-q05",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "For one die roll, let \\(\\Omega=\\{1,2,3,4,5,6\\}\\), \\(A=\\{2,4,6\\}\\), and \\(B=\\{1,3,5\\}\\). Which statements use outcome, event, and sample-space notation correctly?",
    options: [
      {
        text: "\\(4\\) is an outcome, while \\(\\{4\\}\\) is the singleton event that the outcome is 4.",
        isCorrect: true,
      },
      {
        text: "\\(A\\) is an event because it contains several possible outcomes from \\(\\Omega\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\Omega\\) is the event that records which outcome actually happened after the die is rolled.",
        isCorrect: false,
      },
      {
        text: "\\(A \\cup B = A\\), because even outcomes already cover all possible die results.",
        isCorrect: false,
      },
    ],
    explanation:
      "An outcome is a single possible result, an event is a set of possible results, and the sample space is the set of all possible results. Here \\(A\\cup B=\\Omega\\), not \\(A\\), and \\(\\Omega\\) exists before the roll rather than being the realized outcome itself.",
  },
  {
    id: "crash-probability-l1-q06",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "A discrete random variable has probabilities \\(P(X=1)=0.05\\), \\(P(X=2)=0.15\\), \\(P(X=3)=0.20\\), \\(P(X=4)=0.25\\), \\(P(X=5)=0.10\\), and \\(P(X=6)=0.25\\). What is \\(P(X\\text{ is even})\\)?",
    options: [
      { text: "\\(0.35\\)", isCorrect: false },
      { text: "\\(0.55\\)", isCorrect: false },
      { text: "\\(0.65\\)", isCorrect: true },
      { text: "\\(0.75\\)", isCorrect: false },
    ],
    explanation:
      "The even event is \\(\\{2,4,6\\}\\), so its probability is \\(0.15+0.25+0.25=0.65\\). This calculation uses probability mass from the listed distribution, not the fair-die shortcut of counting outcomes equally.",
  },
  {
    id: "crash-probability-l1-q07",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe mutually exclusive and exhaustive outcomes in AI classification?",
    options: [
      {
        text: "In a single-label cat/dog/fox classifier, the class outcomes are modeled as mutually exclusive.",
        isCorrect: true,
      },
      {
        text: "In a multi-label image task, person and bicycle can both be true for the same image.",
        isCorrect: true,
      },
      {
        text: "An exhaustive label set covers every modeled possibility for the task.",
        isCorrect: true,
      },
      {
        text: "Multi-label probabilities must sum to 1 because each label has its own yes/no question.",
        isCorrect: false,
      },
    ],
    explanation:
      "Single-label classification usually treats classes as mutually exclusive, while multi-label classification asks separate label questions that can be true together. Exhaustiveness means the modeled possibilities cover the whole space, but separate multi-label probabilities do not have to share one unit of probability mass.",
  },
  {
    id: "crash-probability-l1-q08",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "For \\(X =\\) next token, which statements correctly connect random variables, distributions, and outcomes?",
    options: [
      {
        text: "Before sampling, \\(X\\) is an uncertain quantity.",
        isCorrect: true,
      },
      {
        text: "A distribution gives probabilities for the possible values of \\(X\\).",
        isCorrect: true,
      },
      {
        text: "After sampling, one realized value such as \\(X = \\text{cat}\\) is the outcome.",
        isCorrect: true,
      },
      {
        text: "The random variable, the distribution, and the realized outcome describe different parts of the same uncertainty setup.",
        isCorrect: true,
      },
    ],
    explanation:
      "A random variable is the named uncertain thing, the distribution describes how likely its possible values are, and an outcome is what actually happens. All four statements preserve that distinction instead of collapsing uncertainty, probability assignments, and realization into one idea.",
  },
  {
    id: "crash-probability-l1-q09",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly interpret \\(P(X_t \\mid X_1, X_2, \\ldots, X_{t-1})\\) for an LLM next-token model?",
    options: [
      {
        text: "It is a distribution over the token at position \\(t\\) conditioned on previous tokens.",
        isCorrect: true,
      },
      {
        text: "It says the next token is independent of the previous tokens.",
        isCorrect: false,
      },
      {
        text: "It supports repeated next-token prediction during text generation.",
        isCorrect: true,
      },
      {
        text: "It is an expected numeric reward rather than a token probability expression.",
        isCorrect: false,
      },
    ],
    explanation:
      "The expression represents a conditional next-token distribution: the model assigns probabilities to possible \\(X_t\\) values given earlier tokens. It is not an independence statement and not an expected reward, though it can be used repeatedly as generation proceeds token by token.",
  },
  {
    id: "crash-probability-l1-q10",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "An RL agent is in state \\(S_t\\), chooses action \\(A_t\\), receives reward \\(R_{t+1}\\), and moves to \\(S_{t+1}\\). Which statements correctly explain why probability enters this setup?",
    options: [
      {
        text: "The next state can be uncertain even after the current state and action are known.",
        isCorrect: true,
      },
      {
        text: "The reward can be noisy or delayed rather than guaranteed by the immediate action alone.",
        isCorrect: true,
      },
      {
        text: "The best action can depend on expected future reward rather than only immediate reward.",
        isCorrect: true,
      },
      {
        text: "Probability is unnecessary once the agent lists the available actions.",
        isCorrect: false,
      },
    ],
    explanation:
      "Reinforcement learning uses probability because transitions, rewards, and future returns can be uncertain. Listing actions defines choices, but it does not determine which next state or long-term reward path will occur.",
  },
  {
    id: "crash-probability-l1-q11",
    chapter: 1,
    difficulty: "medium",
    type: "assertion-reason",
    prompt:
      "Assertion: In a multi-label image model, \\(P(\\text{person})\\), \\(P(\\text{bicycle})\\), and \\(P(\\text{helmet})\\) do not have to sum to 1.\n\nReason: Several of those labels can be true for the same image at the same time.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: false },
      { text: "Assertion is false, Reason is true.", isCorrect: false },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: true,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "The assertion is true because multi-label classification usually treats each label as its own possible yes/no target. The reason is true and explains the assertion: if person and bicycle can both be true, their probabilities are not competing for a single categorical probability mass.",
  },
  {
    id: "crash-probability-l1-q12",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which examples are best modeled as discrete random variables rather than continuous random variables in an introductory AI probability setting?",
    options: [
      {
        text: "\\(Y\\in\\{\\text{cat},\\text{dog},\\text{fox}\\}\\), where \\(Y\\) is a single image label.",
        isCorrect: true,
      },
      {
        text: "\\(X_t\\), where the possible values are the finite tokens in a vocabulary.",
        isCorrect: true,
      },
      {
        text: "\\(\\epsilon\\), where a noise coordinate can take any real value in an interval.",
        isCorrect: false,
      },
      {
        text: "A model weight represented as a real-valued scalar during training.",
        isCorrect: false,
      },
    ],
    explanation:
      "Discrete variables take countable values, so labels and tokens fit the introductory discrete-distribution setup. Real-valued noise coordinates and model weights are continuous quantities, even if a computer stores them with finite precision.",
  },
  {
    id: "crash-probability-l1-q13",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "For \\(X\\sim\\mathrm{Bernoulli}(p)\\), which statements are mathematically correct?",
    options: [
      {
        text: "The support can be written as \\(X\\in\\{0,1\\}\\).",
        isCorrect: true,
      },
      {
        text: "If \\(P(X=1)=p\\), then \\(P(X=0)=1-p\\).",
        isCorrect: true,
      },
      {
        text: "The expected value is \\(\\mathbb{E}[X]=p\\).",
        isCorrect: true,
      },
      {
        text: "When \\(p=0.8\\), the probability mass on \\(X=0\\) is \\(0.2\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "A Bernoulli random variable is a two-outcome discrete variable usually encoded as 0 and 1. The mass on 1 is \\(p\\), the mass on 0 is \\(1-p\\), and the probability-weighted average of those numeric outcomes is \\(p\\).",
  },
  {
    id: "crash-probability-l1-q14",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which tasks naturally use a categorical distribution rather than a Bernoulli distribution?",
    options: [
      {
        text: "Choosing one diagnosis from flu, COVID, allergy, and other.",
        isCorrect: true,
      },
      {
        text: "Predicting the next token from a 50,000-token vocabulary.",
        isCorrect: true,
      },
      {
        text: "Choosing one action from left, right, and wait.",
        isCorrect: true,
      },
      {
        text: "Modeling a yes/no click outcome with one success probability.",
        isCorrect: false,
      },
    ],
    explanation:
      "A categorical distribution describes one outcome chosen from more than two categories, such as diagnoses, vocabulary tokens, or action choices. A yes/no click variable is binary, so it is better described by a Bernoulli distribution in this introductory setting.",
  },
  {
    id: "crash-probability-l1-q15",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "For a discrete probability mass function \\(p_i=P(X=x_i)\\), which statements correctly describe probability mass?",
    options: [
      {
        text: "Each mass must satisfy \\(0\\leq p_i\\leq 1\\).",
        isCorrect: true,
      },
      {
        text: "For a complete mutually exclusive outcome set, \\(\\sum_i p_i=1\\).",
        isCorrect: true,
      },
      {
        text: "An event probability is found by adding the masses of outcomes inside the event.",
        isCorrect: true,
      },
      {
        text: "A categorical next-token model spreads one unit of mass across vocabulary tokens at that step.",
        isCorrect: true,
      },
    ],
    explanation:
      "A probability mass function assigns nonnegative probability to each discrete possible value and sums to one over a complete outcome set. Event probabilities are sums of the relevant masses, which is why token events such as punctuation or article can be computed by grouping tokens.",
  },
  {
    id: "crash-probability-l1-q16",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "A classifier outputs cat 0.51 and dog 0.49, and a downstream rule displays only the largest-probability label. Which statements are correct?",
    options: [
      {
        text: "The model distribution is nearly split between two classes.",
        isCorrect: true,
      },
      {
        text: "The displayed label cat means the model assigned zero probability to dog.",
        isCorrect: false,
      },
      {
        text: "The decision rule hides information about uncertainty.",
        isCorrect: true,
      },
      {
        text: "The model is well-calibrated because cat has the largest probability.",
        isCorrect: false,
      },
    ],
    explanation:
      "The decision rule chooses cat, but the underlying distribution says dog remains almost as plausible under the model. Calibration cannot be inferred from one largest-probability label; it requires checking whether predicted probabilities match empirical frequencies over many cases.",
  },
  {
    id: "crash-probability-l1-q17",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly distinguish calibration from accuracy?",
    options: [
      {
        text: "A model is well-calibrated at 70 percent confidence if about 70 percent of those predictions are correct over many cases.",
        isCorrect: true,
      },
      {
        text: "A model can be accurate but overconfident.",
        isCorrect: true,
      },
      {
        text: "Calibration concerns the meaning of probability values, not just whether the top label is right.",
        isCorrect: true,
      },
      {
        text: "If a model's top-1 accuracy is high, its probabilities must be calibrated.",
        isCorrect: false,
      },
    ],
    explanation:
      "Accuracy counts how often the chosen answer is correct, while calibration asks whether confidence values correspond to real frequencies. A high-accuracy model can still make 99 percent confidence predictions that are correct much less often than 99 percent of the time.",
  },
  {
    id: "crash-probability-l1-q18",
    chapter: 1,
    difficulty: "hard",
    type: "assertion-reason",
    prompt:
      "Assertion: In a single-label categorical distribution over cat, dog, and fox, the probabilities should sum to 1.\n\nReason: The classes are modeled as mutually exclusive and exhaustive possible outcomes for that prediction.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: false },
      { text: "Assertion is false, Reason is true.", isCorrect: false },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: true,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "The assertion is true for a complete single-label categorical distribution because exactly one modeled class is selected. The reason is also true and explains why the probabilities share one total unit of mass: the categories are competing outcomes that together cover the model's possibilities.",
  },
  {
    id: "crash-probability-l1-q19",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "A reward random variable has \\(P(R=0)=0.2\\), \\(P(R=5)=0.5\\), and \\(P(R=20)=0.3\\). What is \\(\\mathbb{E}[R]\\)?",
    options: [
      { text: "\\(5.5\\)", isCorrect: false },
      { text: "\\(8.5\\)", isCorrect: true },
      { text: "\\(10.0\\)", isCorrect: false },
      { text: "\\(25.0\\)", isCorrect: false },
    ],
    explanation:
      "The expected reward is \\(0\\cdot0.2+5\\cdot0.5+20\\cdot0.3=8.5\\). The value 8.5 is a probability-weighted average and does not need to be one of the possible reward outcomes.",
  },
  {
    id: "crash-probability-l1-q20",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Action A gives reward 10 with probability 0.5 and reward 0 with probability 0.5. Action B gives reward 4 with probability 1. Which statements correctly compare them?",
    options: [
      {
        text: "Action A has expected reward 5.",
        isCorrect: true,
      },
      {
        text: "Action B has expected reward 4.",
        isCorrect: true,
      },
      {
        text: "Action B has the higher expected reward because it is guaranteed.",
        isCorrect: false,
      },
      {
        text: "Action A is less risky because it has a higher possible reward.",
        isCorrect: false,
      },
    ],
    explanation:
      "Action A has expectation \\(10 \\cdot 0.5 + 0 \\cdot 0.5 = 5\\), while Action B has expectation \\(4\\). A guarantee affects risk, not the expected value calculation, and a higher possible reward can come with more uncertainty.",
  },
  {
    id: "crash-probability-l1-q21",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe expected loss in machine learning?",
    options: [
      {
        text: "A loss measures how bad a model prediction is for a task.",
        isCorrect: true,
      },
      {
        text: "Training often adjusts parameters to reduce average prediction error.",
        isCorrect: true,
      },
      {
        text: "A dataset average \\(\\frac{1}{n}\\sum_i L(f(x_i), y_i)\\) is an empirical approximation to expected loss.",
        isCorrect: true,
      },
      {
        text: "Expected loss is minimized by choosing the most likely class once and never updating model parameters.",
        isCorrect: false,
      },
    ],
    explanation:
      "Expected loss is the average badness of predictions under the relevant data distribution, and training usually estimates and reduces it using data. Picking a class once is a decision, not a training procedure for minimizing average prediction error across examples.",
  },
  {
    id: "crash-probability-l1-q22",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly handle expectation for categorical outputs such as words?",
    options: [
      {
        text: "An ordinary arithmetic average of words such as \\(0.4 \\cdot \\text{cat} + 0.3 \\cdot \\text{dog}\\) is not meaningful as text.",
        isCorrect: true,
      },
      {
        text: "For categorical outputs, a model can still use the full distribution or choose or sample from it.",
        isCorrect: true,
      },
      {
        text: "If categorical items are represented as vectors, expectations over those vectors can be mathematically meaningful.",
        isCorrect: true,
      },
      {
        text: "Expectation is most straightforward when the random variable has numeric values.",
        isCorrect: true,
      },
    ],
    explanation:
      "Expectation is a weighted average, so it directly makes sense for numeric values but not for raw words as strings. Categorical models still use distributions, and neural networks can represent categories with vectors where averaging can become a meaningful mathematical operation.",
  },
  {
    id: "crash-probability-l1-q23",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "A regression-style model represents possible house prices as \\(P(400k)=0.2\\), \\(P(500k)=0.5\\), and \\(P(600k)=0.3\\). What is the expected price?",
    options: [
      { text: "480k", isCorrect: false },
      { text: "500k", isCorrect: false },
      { text: "510k", isCorrect: true },
      { text: "540k", isCorrect: false },
    ],
    explanation:
      "The expected price is \\(400k \\cdot 0.2 + 500k \\cdot 0.5 + 600k \\cdot 0.3 = 510k\\). The mode is 500k because it has the largest single probability, but expectation uses the full distribution rather than only the most likely value.",
  },
  {
    id: "crash-probability-l1-q24",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Action A gives reward 5 with probability 1. Action B gives reward 10 with probability 0.5 and reward 0 with probability 0.5. Which statements are correct?",
    options: [
      {
        text: "Both actions have expected reward 5.",
        isCorrect: true,
      },
      {
        text: "Action B has higher spread around its expected value.",
        isCorrect: true,
      },
      {
        text: "Expectation alone captures every important difference between the two actions.",
        isCorrect: false,
      },
      {
        text: "Action A is higher variance because it never changes reward.",
        isCorrect: false,
      },
    ],
    explanation:
      "Both actions average to 5, but Action A is certain while Action B alternates between high and zero reward. Variance captures this spread, so expectation alone misses a major risk difference.",
  },
  {
    id: "crash-probability-l1-q25",
    chapter: 1,
    difficulty: "hard",
    type: "assertion-reason",
    prompt:
      "Assertion: The expected value of a fair die roll is not itself a possible die-roll outcome.\n\nReason: Expectation is a probability-weighted long-run average rather than a claim that the average value must occur on an individual trial.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: false },
      { text: "Assertion is false, Reason is true.", isCorrect: false },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: true,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "The assertion is true because a fair die has expected value 3.5, and 3.5 is not a face of the die. The reason is also true and explains the assertion: expectation summarizes averages over repetitions rather than requiring the average to appear in a single trial.",
  },
  {
    id: "crash-probability-l1-q26",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which AI interpretations correctly use variance or spread rather than only expectation?",
    options: [
      {
        text: "Two RL actions with the same expected reward can differ in real-world risk.",
        isCorrect: true,
      },
      {
        text: "A classifier with cat 0.36, dog 0.34, and fox 0.30 is less certain than one with cat 0.98, dog 0.01, and fox 0.01.",
        isCorrect: true,
      },
      {
        text: "More noise in a diffusion process means more uncertainty about the original clean data.",
        isCorrect: true,
      },
      {
        text: "If expected rewards match, a risk-sensitive application has no further probabilistic distinction to evaluate.",
        isCorrect: false,
      },
    ],
    explanation:
      "Spread and uncertainty matter in addition to averages, especially for actions or predictions used in high-stakes systems. Matching expectations does not erase differences in risk, confidence, or noise level, so the final option incorrectly treats expectation as complete.",
  },
  {
    id: "crash-probability-l1-q27",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly distinguish variance from entropy in the probability concepts used for AI?",
    options: [
      {
        text: "For a numerical random variable, \\(\\mathrm{Var}(X)=\\mathbb{E}[X^2]-(\\mathbb{E}[X])^2\\) is equivalent to \\(\\mathbb{E}[(X-\\mathbb{E}[X])^2]\\).",
        isCorrect: true,
      },
      {
        text: "For categorical token distributions, entropy is usually a more meaningful uncertainty measure than treating token ids as ordinary numerical values.",
        isCorrect: true,
      },
      {
        text: "A uniform categorical distribution has zero entropy because no outcome has higher probability than another.",
        isCorrect: false,
      },
      {
        text: "The variance of arbitrary token ids is invariant to how the vocabulary is numbered.",
        isCorrect: false,
      },
    ],
    explanation:
      "Variance is a spread measure for meaningful numerical values, and \\(\\mathbb{E}[X^2]-(\\mathbb{E}[X])^2\\) is the common algebraic form. Entropy is high, not zero, for a uniform categorical distribution, and token-id variance depends on arbitrary numbering rather than semantic uncertainty.",
  },
  {
    id: "crash-probability-l1-q28",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly connect diffusion models to probability?",
    options: [
      {
        text: "They can start from random noise.",
        isCorrect: true,
      },
      {
        text: "They learn structure by learning to reverse a noise process.",
        isCorrect: true,
      },
      {
        text: "Different images can be plausible for the same prompt.",
        isCorrect: true,
      },
      {
        text: "Denoising steps operate under uncertainty about the clean data.",
        isCorrect: true,
      },
    ],
    explanation:
      "Diffusion models are probabilistic because noise, denoising, and sampling are built into the generation process. The model does not simply retrieve one fixed image; it learns how likely data structure can emerge as uncertainty is gradually reduced.",
  },
  {
    id: "crash-probability-l1-q29",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A medical AI system reports flu 0.45, COVID 0.35, allergy 0.15, and other 0.05. Which conclusions are justified from this distribution?",
    options: [
      {
        text: "Flu is the most likely category under the model.",
        isCorrect: true,
      },
      {
        text: "The probability of flu or COVID is 0.80 if those categories are mutually exclusive.",
        isCorrect: true,
      },
      {
        text: "The model is not extremely confident because COVID remains a substantial competing possibility.",
        isCorrect: true,
      },
      {
        text: "The final diagnosis is objectively flu because its probability is above every other single category.",
        isCorrect: false,
      },
    ],
    explanation:
      "The largest probability is flu, and adding flu plus COVID gives 0.80 when the categories are mutually exclusive. But the distribution is not a guarantee of truth, and the 0.35 COVID probability means the model's uncertainty remains clinically relevant.",
  },
  {
    id: "crash-probability-l1-q30",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which formulas or statements are valid in the introductory probability setup?",
    options: [
      {
        text: "\\(0 \\leq P(A) \\leq 1\\) for an event \\(A\\).",
        isCorrect: true,
      },
      {
        text: "For a complete mutually exclusive set of outcomes, \\(\\sum_i P(x_i)=1\\).",
        isCorrect: true,
      },
      {
        text: "For a discrete numeric random variable, \\(\\mathbb{E}[X]=\\sum_i x_iP(X=x_i)\\).",
        isCorrect: true,
      },
      {
        text: "Variance can be written as \\(\\mathrm{Var}(X)=\\mathbb{E}[(X-\\mathbb{E}[X])^2]\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "These are the core mathematical summaries behind the introductory concepts: event probabilities are bounded, complete mutually exclusive distributions sum to one, expectation is a weighted average, and variance measures squared spread around the expectation. The formulas apply in the settings named by each option.",
  },
  {
    id: "crash-probability-l1-q31",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "An RL agent can take Action A for reward 100 with probability 0.1 and reward 0 with probability 0.9, or Action B for reward 20 with probability 1. Which statements are correct?",
    options: [
      {
        text: "Action A has expected reward 10.",
        isCorrect: true,
      },
      {
        text: "Action B has higher expected reward.",
        isCorrect: true,
      },
      {
        text: "Action A is riskier because most of its probability mass is on zero reward despite a high upside.",
        isCorrect: true,
      },
      {
        text: "Action A should be preferred by expected reward because its maximum possible reward is 100.",
        isCorrect: false,
      },
    ],
    explanation:
      "Action A has expectation \\(100 \\cdot 0.1 + 0 \\cdot 0.9 = 10\\), while Action B has expectation 20. The maximum possible reward is not the expected reward, and Action A's distribution also carries more risk.",
  },
  {
    id: "crash-probability-l1-q32",
    chapter: 1,
    difficulty: "medium",
    type: "assertion-reason",
    prompt:
      "Assertion: A Bernoulli random variable has exactly two possible outcomes.\n\nReason: A categorical distribution describes a random variable with more than two possible categories.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: false },
      { text: "Assertion is false, Reason is true.", isCorrect: false },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: false,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: true,
      },
    ],
    explanation:
      "The assertion is true because Bernoulli distributions model binary outcomes such as success/failure or click/no click. The reason is also true as a description of categorical distributions, but it does not explain why Bernoulli variables have two outcomes; it describes a neighboring distribution family.",
  },
  {
    id: "crash-probability-l1-q33",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly identify random variables in common AI examples?",
    options: [
      {
        text: "\\(Y\\) can represent the class label in classification.",
        isCorrect: true,
      },
      {
        text: "\\(X_t\\) can represent the token at position \\(t\\) in a language model.",
        isCorrect: true,
      },
      {
        text: "\\(R_t\\) can represent reward at time \\(t\\) in reinforcement learning.",
        isCorrect: true,
      },
      {
        text: "\\(\\epsilon\\) can represent random noise in a diffusion model.",
        isCorrect: true,
      },
    ],
    explanation:
      "Each symbol names an uncertain quantity whose value may not be known in advance. The point is not the exact letter choice but the modeling idea: labels, tokens, rewards, and noise can be represented as random variables.",
  },
  {
    id: "crash-probability-l1-q34",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which vector can be a valid categorical probability distribution over four mutually exclusive outcomes?",
    options: [
      {
        text: "\\((0.10, 0.20, 0.30, 0.40)\\)",
        isCorrect: true,
      },
      {
        text: "\\((0.10, 0.20, 0.30, 0.50)\\)",
        isCorrect: false,
      },
      {
        text: "\\((-0.10, 0.30, 0.40, 0.40)\\)",
        isCorrect: false,
      },
      {
        text: "\\((0.25, 0.25, 0.25, 0.20)\\)",
        isCorrect: false,
      },
    ],
    explanation:
      "A categorical distribution must have nonnegative entries that sum to one. The valid vector sums to 1.00 with no negative components, while the other vectors either sum above or below one or include a negative probability.",
  },
  {
    id: "crash-probability-l1-q35",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly distinguish model probability from objective reality?",
    options: [
      {
        text: "\\(P(\\text{cat})=0.72\\) means the model assigns 0.72 probability to cat under its representation.",
        isCorrect: true,
      },
      {
        text: "The assigned probability can be useful even when the final interface shows only cat.",
        isCorrect: true,
      },
      {
        text: "A calibrated model's 70 percent predictions should be correct about 70 percent of the time over comparable cases.",
        isCorrect: true,
      },
      {
        text: "A 0.72 class probability means the image is physically 72 percent cat.",
        isCorrect: false,
      },
    ],
    explanation:
      "The probability is a model-assigned value that can inform confidence, calibration, and downstream decisions. It is not a literal physical fraction of cat in the image, and calibration must be checked statistically over many similar predictions.",
  },
  {
    id: "crash-probability-l1-q36",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which mistakes confuse a distribution with a decision?",
    options: [
      {
        text: "Treating a top label as if it contained no information about runner-up probabilities.",
        isCorrect: true,
      },
      {
        text: "Replacing a 0.51 versus 0.49 distribution with a displayed cat label and then claiming the model was certain.",
        isCorrect: true,
      },
      {
        text: "Using the full probability table to compare confidence across two classifiers.",
        isCorrect: false,
      },
      {
        text: "Separating the model's distribution from the downstream rule that turns it into a label.",
        isCorrect: false,
      },
    ],
    explanation:
      "The mistakes are the ones that collapse a probability distribution into a final label and then invent certainty that the distribution did not express. Using the full table and separating distribution from decision are correct ways to preserve the uncertainty information.",
  },
  {
    id: "crash-probability-l1-q37",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which notation-to-AI translations are valid for the probability concepts in this material?",
    options: [
      {
        text: "\\(\\Omega\\) can represent the set of possible labels, tokens, actions, or states for a modeled choice.",
        isCorrect: true,
      },
      {
        text: "\\(P(X_t=w_i\\mid X_1,\\ldots,X_{t-1})\\) can represent next-token probability given previous tokens.",
        isCorrect: true,
      },
      {
        text: "\\(\\mathbb{E}[\\text{loss}]\\) can represent the average prediction error objective used during training.",
        isCorrect: true,
      },
      {
        text: "\\(\\mathrm{Var}(R)\\) can represent spread or risk in a numerical reward distribution.",
        isCorrect: true,
      },
    ],
    explanation:
      "Each expression maps a probability symbol to a concrete AI use case without changing the mathematical role of the symbol. The sample space lists possible values, conditional next-token probabilities define a categorical distribution, expectation averages loss, and variance measures numerical reward spread.",
  },
  {
    id: "crash-probability-l1-q38",
    chapter: 1,
    difficulty: "hard",
    type: "assertion-reason",
    prompt:
      "Assertion: A model can be overconfident even when it is often correct.\n\nReason: Calibration asks whether stated probabilities match observed frequencies, while accuracy asks how often the chosen answers are right.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: false },
      { text: "Assertion is false, Reason is true.", isCorrect: false },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: true,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "The assertion is true because high accuracy does not guarantee that confidence scores are numerically trustworthy. The reason is true and explains the assertion: calibration evaluates probability meaning over many cases, so a frequently correct model can still assign probabilities that are too extreme.",
  },
  {
    id: "crash-probability-l1-q39",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly prepare the move from \\(P(y)\\) to \\(P(y \\mid x)\\) in prediction?",
    options: [
      {
        text: "\\(P(y)\\) describes probability for an outcome without explicitly conditioning on a specific input.",
        isCorrect: true,
      },
      {
        text: "\\(P(y \\mid x)\\) asks how likely an output is given input information.",
        isCorrect: true,
      },
      {
        text: "Classifiers and LLMs commonly use input or context to shape their output distributions.",
        isCorrect: true,
      },
      {
        text: "Conditional probability is central to prediction because inputs change which outputs are plausible.",
        isCorrect: true,
      },
    ],
    explanation:
      "Prediction usually depends on information: an image changes class probabilities, and previous tokens change next-token probabilities. Moving from \\(P(y)\\) to \\(P(y \\mid x)\\) formalizes that the distribution is conditioned on input rather than floating free of context.",
  },
  {
    id: "crash-probability-l1-q40",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A system must choose between showing the top class immediately or preserving the full distribution for a later high-stakes decision. Which considerations support preserving the distribution?",
    options: [
      {
        text: "The gap between the top two probabilities can affect whether a human review or fallback is appropriate.",
        isCorrect: true,
      },
      {
        text: "Calibration checks require probability values, not just final labels.",
        isCorrect: true,
      },
      {
        text: "Risk-sensitive decisions may treat a 0.51 top class differently from a 0.99 top class.",
        isCorrect: true,
      },
      {
        text: "The top class label alone contains every uncertainty signal needed for downstream use.",
        isCorrect: false,
      },
    ],
    explanation:
      "A full distribution can drive confidence thresholds, escalation, calibration analysis, and risk-sensitive policies. A top label may be enough for a simple display, but it discards uncertainty that matters when decisions have cost, safety, or reliability consequences.",
  },
  {
    id: "crash-probability-l1-q41",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Let \\(\\Omega=\\{a,b,c,d\\}\\) with \\(P(a)=0.10\\), \\(P(b)=0.20\\), \\(P(c)=0.25\\), and \\(P(d)=0.45\\). What is \\(P(\\{b,d\\})\\)?",
    options: [
      { text: "\\(0.20\\)", isCorrect: false },
      { text: "\\(0.45\\)", isCorrect: false },
      { text: "\\(0.65\\)", isCorrect: true },
      { text: "\\(0.90\\)", isCorrect: false },
    ],
    explanation:
      "The event \\(\\{b,d\\}\\) contains two mutually exclusive outcomes, so its probability is \\(P(b)+P(d)=0.20+0.45=0.65\\). The answer is not the larger individual mass, and it is not found by adding unrelated outcomes outside the event.",
  },
  {
    id: "crash-probability-l1-q42",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A single-label classifier has \\(P(\\text{cat})=0.50\\), \\(P(\\text{dog})=0.30\\), and \\(P(\\text{fox})=0.20\\). Let \\(A=\\{\\text{cat},\\text{dog}\\}\\) and \\(B=\\{\\text{dog},\\text{fox}\\}\\). Which statements are correct?",
    options: [
      { text: "\\(P(A)=0.80\\).", isCorrect: true },
      { text: "\\(P(A\\cap B)=0.30\\).", isCorrect: true },
      { text: "\\(P(A\\cup B)=0.70\\).", isCorrect: false },
      {
        text: "\\(P(A)+P(B)=P(A\\cup B)\\) because \\(A\\) and \\(B\\) overlap on dog.",
        isCorrect: false,
      },
    ],
    explanation:
      "\\(A\\) contains cat and dog, so its probability is 0.80, and \\(A\\cap B\\) contains only dog, so its probability is 0.30. The union contains all three classes and has probability 1.00; adding \\(P(A)+P(B)\\) double-counts dog because the events overlap.",
  },
  {
    id: "crash-probability-l1-q43",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A reward variable has \\(P(R=-2)=0.10\\), \\(P(R=0)=0.40\\), and \\(P(R=5)=0.50\\). Which statements are correct?",
    options: [
      { text: "\\(\\mathbb{E}[R]=2.3\\).", isCorrect: true },
      { text: "\\(P(R>0)=0.50\\).", isCorrect: true },
      {
        text: "The expected reward equals the maximum possible reward because the maximum reward has the largest probability mass.",
        isCorrect: false,
      },
      {
        text: "The expectation is 5 because 5 has the largest probability mass.",
        isCorrect: false,
      },
    ],
    explanation:
      "The expected reward is \\((-2)\\cdot0.10+0\\cdot0.40+5\\cdot0.50=2.3\\), and the only positive reward outcome has probability 0.50. The most likely or highest reward is not automatically the expected reward because expectation weighs every outcome by its probability, so the maximum outcome 5 does not become the average.",
  },
  {
    id: "crash-probability-l1-q44",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A model has per-example losses \\(0.2,0.6,0.4,0.8\\) on four equally weighted training examples. Which statements correctly interpret the empirical expected-loss calculation?",
    options: [
      {
        text: "The empirical average loss is \\(\\frac{0.2+0.6+0.4+0.8}{4}=0.5\\).",
        isCorrect: true,
      },
      {
        text: "The denominator is 4 because there are four equally weighted examples.",
        isCorrect: true,
      },
      {
        text: "This average is an empirical approximation to an expected loss over data.",
        isCorrect: true,
      },
      {
        text: "The average loss is not the same as the single worst loss in the batch.",
        isCorrect: true,
      },
    ],
    explanation:
      "The empirical loss average is a finite-data estimate of expected loss, so all four losses contribute equally in this setup. It summarizes average prediction error; it does not become the maximum loss just because one example is worse than the others.",
  },
  {
    id: "crash-probability-l1-q45",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A reward variable has \\(P(R=10)=0.5\\) and \\(P(R=0)=0.5\\). What is \\(\\mathrm{Var}(R)\\)?",
    options: [
      { text: "\\(5\\)", isCorrect: false },
      { text: "\\(10\\)", isCorrect: false },
      { text: "\\(25\\)", isCorrect: true },
      { text: "\\(50\\)", isCorrect: false },
    ],
    explanation:
      "The expectation is \\(10\\cdot0.5+0\\cdot0.5=5\\). The variance is \\(0.5(10-5)^2+0.5(0-5)^2=25\\), which measures spread around the mean rather than the mean itself; choosing 5 confuses variance with expected reward.",
  },
  {
    id: "crash-probability-l1-q46",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which vectors can be valid probability distributions over three mutually exclusive categorical outcomes?",
    options: [
      { text: "\\((0.20,0.30,0.50)\\)", isCorrect: true },
      { text: "\\((0.10,0.10,0.10)\\)", isCorrect: false },
      { text: "\\((-0.10,0.60,0.50)\\)", isCorrect: false },
      { text: "\\((0,0.25,0.75)\\)", isCorrect: true },
    ],
    explanation:
      "A valid categorical probability vector must have no negative entries and must sum to one. The valid vectors satisfy both conditions, while the second sums to only 0.30 and the third includes a negative mass even though its entries sum to one.",
  },
  {
    id: "crash-probability-l1-q47",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "For an LLM step modeled as \\(P(X_t=w_i\\mid X_1,\\ldots,X_{t-1})\\), which statements are correct?",
    options: [
      {
        text: "The probabilities are indexed by candidate token values \\(w_i\\).",
        isCorrect: true,
      },
      {
        text: "For a fixed context, the probabilities across the vocabulary should sum to one.",
        isCorrect: true,
      },
      {
        text: "Changing the previous tokens can change the whole next-token distribution.",
        isCorrect: true,
      },
      {
        text: "After sampling, every token with nonzero probability occurs at position \\(t\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The expression describes a categorical distribution over candidate next-token values, conditioned on the previous tokens. Sampling or choosing from that distribution produces one realized token at position \\(t\\), not all nonzero-probability tokens simultaneously.",
  },
  {
    id: "crash-probability-l1-q48",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which notation statements are correct for a discrete random variable \\(X\\) with possible values \\(x_i\\)?",
    options: [
      {
        text: "\\(P(X=x_i)\\) is the probability mass assigned to value \\(x_i\\).",
        isCorrect: true,
      },
      {
        text: "For a complete discrete distribution, \\(\\sum_i P(X=x_i)=1\\).",
        isCorrect: true,
      },
      {
        text: "For an event \\(A\\), \\(P(A)\\) can be computed by summing masses for outcomes in \\(A\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\mathbb{E}[X]=\\sum_i x_iP(X=x_i)\\) when the values \\(x_i\\) are numerical.",
        isCorrect: true,
      },
    ],
    explanation:
      "These are the core discrete-probability formulas behind the course examples. They connect probability mass, total mass, event probability, and expectation without confusing the random variable with a realized outcome.",
  },
  {
    id: "crash-probability-l1-q49",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A classifier makes 200 predictions at reported confidence 0.70, and 130 of those predictions are correct. Relative to perfect calibration at that confidence level, what does this show?",
    options: [
      {
        text: "It is overconfident by about 5 percentage points at the 0.70 confidence level.",
        isCorrect: true,
      },
      {
        text: "It is underconfident by about 5 percentage points at the 0.70 confidence level.",
        isCorrect: false,
      },
      {
        text: "It is exactly calibrated because more than half of the predictions are correct.",
        isCorrect: false,
      },
      {
        text: "Calibration cannot be discussed because the model produced probabilities rather than labels.",
        isCorrect: false,
      },
    ],
    explanation:
      "Perfect calibration at 0.70 would mean about \\(200\\cdot0.70=140\\) correct predictions, but the observed count is 130, or 65 percent. The model's stated confidence is therefore higher than its observed frequency in this group.",
  },
  {
    id: "crash-probability-l1-q50",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A training batch has equally weighted losses \\(0.1,0.4,0.9,0.6\\). Which statements correctly interpret the average-loss objective?",
    options: [
      {
        text: "The empirical average loss is \\(0.5\\).",
        isCorrect: true,
      },
      {
        text: "Reducing the third loss from \\(0.9\\) to \\(0.3\\) would reduce the average by \\(0.15\\).",
        isCorrect: true,
      },
      {
        text: "The expected-loss estimate is \\(0.9\\) because the largest loss dominates the objective.",
        isCorrect: false,
      },
      {
        text: "The losses must sum to one because they are treated as probability masses.",
        isCorrect: false,
      },
    ],
    explanation:
      "The average is \\((0.1+0.4+0.9+0.6)/4=0.5\\). Loss values are not probability masses, so they do not need to sum to one, and changing one loss by \\(0.6\\) changes the four-example average by \\(0.6/4=0.15\\).",
  },
  {
    id: "crash-probability-l1-q51",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A single-label diagnosis model reports flu 0.45, COVID 0.35, allergy 0.15, and other 0.05. Which statements are correct?",
    options: [
      {
        text: "\\(P(\\text{flu or COVID})=0.80\\).",
        isCorrect: true,
      },
      {
        text: "\\(P(\\text{not other})=0.95\\).",
        isCorrect: true,
      },
      {
        text: "\\(P(\\text{allergy or other})=0.20\\).",
        isCorrect: true,
      },
      {
        text: "Flu and COVID can both be the realized label in this single-label categorical setup.",
        isCorrect: false,
      },
    ],
    explanation:
      "Because the categories are single-label and mutually exclusive, event probabilities are sums of the included category masses. Flu or COVID sums to 0.80, not other is 0.95, and allergy or other is 0.20; only one label is realized.",
  },
  {
    id: "crash-probability-l1-q52",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly connect diffusion-model notation to probability and uncertainty?",
    options: [
      {
        text: "\\(X_0\\) can denote clean data, such as the original image.",
        isCorrect: true,
      },
      {
        text: "\\(X_t\\) can denote a noisier version of the data after adding noise for \\(t\\) steps.",
        isCorrect: true,
      },
      {
        text: "\\(\\epsilon\\) can represent random noise injected into the process.",
        isCorrect: true,
      },
      {
        text: "The reverse process tries to remove uncertainty introduced by the noise process.",
        isCorrect: true,
      },
    ],
    explanation:
      "Diffusion models use random variables for clean data, noisy data, and noise, so the notation is naturally probabilistic. The learned reverse process does not erase the need for probability; it models how plausible clean data can be recovered from noisy states.",
  },
  {
    id: "crash-probability-l1-q53",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A model uses one softmax distribution over four classes, and the class probabilities sum to one. Which modeling assumption best fits that setup?",
    options: [
      {
        text: "Exactly one of the four modeled labels is treated as the realized class.",
        isCorrect: true,
      },
      {
        text: "Each label is an independent Bernoulli variable that can be true together with the others.",
        isCorrect: false,
      },
      {
        text: "The output is a continuous random variable with a probability density over real values.",
        isCorrect: false,
      },
      {
        text: "The probabilities represent four separate expected losses rather than class probabilities.",
        isCorrect: false,
      },
    ],
    explanation:
      "A softmax-style categorical distribution over classes treats the labels as competing outcomes, so exactly one modeled class is realized. Multi-label Bernoulli outputs, continuous regression, and expected-loss values use different mathematical objects.",
  },
  {
    id: "crash-probability-l1-q54",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "An agent compares Action A, which gives immediate reward 2 and no future reward, with Action B, which gives immediate reward 0 and then future reward 10 with probability 0.4 or 0 with probability 0.6. Which statements are correct?",
    options: [
      {
        text: "Action B has expected total reward 4.",
        isCorrect: true,
      },
      {
        text: "Action A has the larger immediate reward.",
        isCorrect: true,
      },
      {
        text: "Action B guarantees total reward 4.",
        isCorrect: false,
      },
      {
        text: "Action A has expected total reward 4 because it is less risky.",
        isCorrect: false,
      },
    ],
    explanation:
      "Action B's expected total reward is \\(0+10\\cdot0.4+0\\cdot0.6=4\\), while Action A's total reward is 2. The expected value is not guaranteed, and lower risk does not change Action A's arithmetic expectation.",
  },
  {
    id: "crash-probability-l1-q55",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Action C gives reward 4 with probability 1. Action D gives reward 0 with probability 0.5 and reward 8 with probability 0.5. Which statements are correct?",
    options: [
      {
        text: "Both actions have expected reward 4.",
        isCorrect: true,
      },
      {
        text: "Action C has variance 0.",
        isCorrect: true,
      },
      {
        text: "Action D has variance 16.",
        isCorrect: true,
      },
      {
        text: "Action D has lower spread because its rewards average to the same value as Action C.",
        isCorrect: false,
      },
    ],
    explanation:
      "Action C is constant, so its variance is 0, while Action D has the same mean but spreads outcomes equally around 4. For Action D, \\(0.5(0-4)^2+0.5(8-4)^2=16\\), showing why equal expectations can hide different risk.",
  },
  {
    id: "crash-probability-l1-q56",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "An LLM has a vocabulary of 50,000 tokens and assigns probabilities \\(p_i=P(X_t=w_i\\mid\\text{context})\\). Which statements are correct?",
    options: [
      {
        text: "For the modeled vocabulary, \\(\\sum_i p_i=1\\) at that generation step.",
        isCorrect: true,
      },
      {
        text: "Argmax decoding and sampling are different ways to turn the distribution into a realized token.",
        isCorrect: true,
      },
      {
        text: "If the top token has probability 0.82, then 0.18 probability mass remains on other tokens.",
        isCorrect: true,
      },
      {
        text: "Changing the context can shift probability mass across many token values.",
        isCorrect: true,
      },
    ],
    explanation:
      "The next-token distribution is categorical over the vocabulary, so its masses sum to one for a fixed context. A decoding rule or sampling step produces a realized token, but the full distribution still contains confidence and uncertainty information.",
  },
  {
    id: "crash-probability-l1-q57",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which interpretation of \\(P(y\\mid x)\\) best matches the prediction setting emphasized in AI?",
    options: [
      {
        text: "The probability of output \\(y\\) given input information \\(x\\).",
        isCorrect: true,
      },
      {
        text: "The probability of input \\(x\\) after the output \\(y\\) has already been observed.",
        isCorrect: false,
      },
      {
        text: "The joint probability that \\(x\\) and \\(y\\) occur together without conditioning.",
        isCorrect: false,
      },
      {
        text: "The expected numerical difference between \\(y\\) and \\(x\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "\\(P(y\\mid x)\\) is conditional probability: how likely an output is when the input information is known. This is the mathematical shape behind image classification, next-token prediction, and many other prediction problems.",
  },
  {
    id: "crash-probability-l1-q58",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly handle the discrete/continuous boundary in AI probability examples?",
    options: [
      {
        text: "A raw token identity is discrete because it is one value from a vocabulary.",
        isCorrect: true,
      },
      {
        text: "An embedding coordinate can be treated as continuous because it is a real-valued component.",
        isCorrect: true,
      },
      {
        text: "A categorical distribution over tokens is a probability density over real numbers.",
        isCorrect: false,
      },
      {
        text: "Diffusion noise is discrete because it is stored in an image-shaped tensor.",
        isCorrect: false,
      },
    ],
    explanation:
      "Token identities are categorical values, while embedding coordinates and many noise values are real-valued quantities. Tensor shape does not make a variable discrete, and categorical token probabilities are probability masses rather than continuous densities.",
  },
  {
    id: "crash-probability-l1-q59",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A random variable has \\(P(X=-1)=0.2\\), \\(P(X=0)=0.5\\), and \\(P(X=2)=0.3\\). Which statements are correct?",
    options: [
      { text: "\\(\\mathbb{E}[X]=0.4\\).", isCorrect: true },
      { text: "\\(P(X\\geq 0)=0.8\\).", isCorrect: true },
      { text: "\\(\\mathrm{Var}(X)=1.24\\).", isCorrect: true },
      {
        text: "\\(\\mathbb{E}[X]=0\\) because \\(0\\) is the most likely outcome.",
        isCorrect: false,
      },
    ],
    explanation:
      "The expectation is \\((-1)\\cdot0.2+0\\cdot0.5+2\\cdot0.3=0.4\\), and \\(P(X\\geq0)=0.5+0.3=0.8\\). Since \\(\\mathbb{E}[X^2]=1.4\\), the variance is \\(1.4-0.4^2=1.24\\), not the most likely outcome.",
  },
  {
    id: "crash-probability-l1-q60",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which synthesis statements correctly connect the mathematical objects in probability to AI behavior?",
    options: [
      {
        text: "A distribution can encode uncertainty that a final displayed label discards.",
        isCorrect: true,
      },
      {
        text: "Expected reward can favor an action with lower immediate reward when future outcomes have enough probability mass on high rewards.",
        isCorrect: true,
      },
      {
        text: "Calibration checks whether numerical confidence values behave like empirical frequencies over comparable cases.",
        isCorrect: true,
      },
      {
        text: "Variance or spread can distinguish two numerical reward distributions with the same expectation.",
        isCorrect: true,
      },
    ],
    explanation:
      "The concepts fit together: probability distributions represent uncertainty, decisions summarize or sample from them, expectation averages numerical outcomes, and variance captures spread. AI systems often need all of these distinctions because a single output can hide uncertainty, risk, and confidence quality.",
  },
];
