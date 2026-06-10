import { Question } from "../../../quiz";

export const CrashCourseProbabilityL3Questions: Question[] = [
  {
    id: "crash-probability-l3-q01",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "A neural network emits logits \\(z=[2.1,1.3,-0.5]\\) for cat, dog, and car. Which interpretation is most accurate before applying softmax?",
    options: [
      {
        text: "The logits are raw, unnormalized preference scores whose relative sizes rank the classes.",
        isCorrect: true,
      },
      {
        text: "The logits are already valid probabilities because there is one number for each class.",
        isCorrect: false,
      },
      {
        text: "The logits are log probabilities, so exponentiating each one directly gives probabilities that sum to one.",
        isCorrect: false,
      },
      {
        text: "The logits are the final sampled outputs, so no probability distribution remains after they are produced.",
        isCorrect: false,
      },
    ],
    explanation:
      "Logits are unconstrained scores: they may be negative, larger than one, and need not sum to anything meaningful. Softmax is needed because probabilities must be nonnegative and sum to one, while a final decision or sample is a separate step after probabilities are available.",
  },
  {
    id: "crash-probability-l3-q02",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "For logits \\(z_i\\), which steps are part of the softmax construction \\(P(y_i)=\\frac{e^{z_i}}{\\sum_j e^{z_j}}\\)?",
    options: [
      {
        text: "Exponentiate each logit so the unnormalized scores become positive.",
        isCorrect: true,
      },
      {
        text: "Divide each exponentiated score by the sum of all exponentiated scores.",
        isCorrect: true,
      },
      {
        text: "Subtract the largest probability from each class so the outputs can be negative.",
        isCorrect: false,
      },
      {
        text: "Set every class with a negative logit to probability zero before normalizing.",
        isCorrect: false,
      },
    ],
    explanation:
      "Softmax uses exponentials to turn all scores into positive quantities and then normalizes by their total. Negative logits can still receive positive probability, and the normalization is over exponentiated logits, not over already formed probabilities.",
  },
  {
    id: "crash-probability-l3-q03",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Suppose logits for cat, dog, and car are \\((2,1,0)\\), with \\(e^2\\approx7.39\\), \\(e^1\\approx2.72\\), and \\(e^0=1\\). Which statements about the softmax probabilities are correct?",
    options: [
      {
        text: "The probability for cat is approximately \\(7.39/(7.39+2.72+1)\\approx0.67\\).",
        isCorrect: true,
      },
      {
        text: "The probability for dog is approximately \\(2.72/(7.39+2.72+1)\\approx0.24\\).",
        isCorrect: true,
      },
      {
        text: "The probability for car is approximately \\(1/(7.39+2.72+1)\\approx0.09\\).",
        isCorrect: true,
      },
      {
        text: "The probability for car is exactly zero because its logit is zero.",
        isCorrect: false,
      },
    ],
    explanation:
      "Softmax exponentiates zero to one, so a zero logit still contributes positive mass. The three normalized values are approximately 0.67, 0.24, and 0.09, and together they form a valid distribution.",
  },
  {
    id: "crash-probability-l3-q04",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "If the same constant \\(c\\) is added to every logit before softmax, which statements are correct?",
    options: [
      {
        text: "The final softmax probabilities are unchanged.",
        isCorrect: true,
      },
      {
        text: "Each numerator and the denominator are multiplied by the same factor \\(e^c\\).",
        isCorrect: true,
      },
      {
        text: "The ranking of classes by probability is unchanged.",
        isCorrect: true,
      },
      {
        text: "The entropy necessarily increases because all logits are numerically larger.",
        isCorrect: false,
      },
    ],
    explanation:
      "Adding a common constant gives \\(e^{z_i+c}=e^c e^{z_i}\\), and the shared factor cancels in the softmax ratio. Because the probability distribution is identical, ranking, decisions based on argmax, and entropy are unchanged.",
  },
  {
    id: "crash-probability-l3-q05",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "For two classes \\(A\\) and \\(B\\) inside a softmax, suppose \\(z_A-z_B=1\\). Which statements follow from the role of exponentials?",
    options: [
      {
        text: "The unnormalized exponentiated score for \\(A\\) is \\(e\\) times the score for \\(B\\).",
        isCorrect: true,
      },
      {
        text: "The ratio \\(P(A)/P(B)\\) under softmax is \\(e^{z_A-z_B}=e\\).",
        isCorrect: true,
      },
      {
        text: "The probability of \\(A\\) must be \\(e/(1+e)\\) no matter how many other classes are present.",
        isCorrect: false,
      },
      {
        text: "The class with the larger logit receives probability exactly one after softmax.",
        isCorrect: false,
      },
    ],
    explanation:
      "Softmax makes odds depend on logit differences: \\(P(A)/P(B)=e^{z_A}/e^{z_B}=e^{z_A-z_B}\\). The absolute probability of \\(A\\) also depends on all other classes in the denominator, and a larger logit does not make the probability equal to one unless the others vanish in a limiting sense.",
  },
  {
    id: "crash-probability-l3-q06",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which distinctions correctly separate model scores, probability distributions, and decisions?",
    options: [
      {
        text: "A logit is a raw model score before probability normalization.",
        isCorrect: true,
      },
      {
        text: "A softmax probability distribution expresses normalized uncertainty across possible outputs.",
        isCorrect: true,
      },
      {
        text: "A decision rule or sampler chooses an output after the distribution has been produced.",
        isCorrect: true,
      },
      {
        text: "Once a highest-probability output is chosen, the remaining probabilities never contain useful information.",
        isCorrect: false,
      },
    ],
    explanation:
      "The pipeline separates raw scores, normalized probabilities, and the final chosen or sampled output. The probability distribution remains useful because it encodes confidence and alternatives even when a decision rule selects only one class or token.",
  },
  {
    id: "crash-probability-l3-q07",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "The correct class for one example is dog. Model A assigns \\(P(\\text{dog}\\mid x)=0.70\\), while Model B assigns \\(P(\\text{dog}\\mid x)=0.30\\). Which statement is correct?",
    options: [
      {
        text: "Model A has higher likelihood for this observed example because it assigns more probability to the true label.",
        isCorrect: true,
      },
      {
        text: "Model B has higher likelihood because it spreads more probability away from dog.",
        isCorrect: false,
      },
      {
        text: "The likelihoods are equal because both models include dog among the possible labels.",
        isCorrect: false,
      },
      {
        text: "Likelihood cannot be evaluated for one example because it only applies to entire datasets.",
        isCorrect: false,
      },
    ],
    explanation:
      "For one labeled example, the likelihood contribution is the model probability assigned to the observed label. Model A assigns 0.70 to dog rather than 0.30, so it makes the observed outcome more probable under the model.",
  },
  {
    id: "crash-probability-l3-q08",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "A model assigns probabilities \\(0.80\\), \\(0.60\\), and \\(0.50\\) to the correct labels of three independent training examples. Which statements about the dataset likelihood are correct?",
    options: [
      {
        text: "The likelihood contribution for the three examples is \\(0.80\\cdot0.60\\cdot0.50=0.24\\).",
        isCorrect: true,
      },
      {
        text: "The log-likelihood is \\(\\log(0.80)+\\log(0.60)+\\log(0.50)\\).",
        isCorrect: true,
      },
      {
        text: "The likelihood is the sum \\(0.80+0.60+0.50\\) because probabilities should be added across examples.",
        isCorrect: false,
      },
      {
        text: "The likelihood is the average \\((0.80+0.60+0.50)/3\\) because each example has equal weight.",
        isCorrect: false,
      },
    ],
    explanation:
      "The likelihood of multiple observed labels is formed by multiplying the model probabilities of those observed labels. Taking logs converts that product into a sum, while a plain sum or average of probabilities is not the likelihood.",
  },
  {
    id: "crash-probability-l3-q09",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe likelihood as a function of model parameters \\(\\theta\\)?",
    options: [
      {
        text: "The observed dataset is treated as fixed while training searches over parameter values.",
        isCorrect: true,
      },
      {
        text: "The objective can be written as \\(\\theta^*=\\arg\\max_\\theta L(\\theta)\\).",
        isCorrect: true,
      },
      {
        text: "Changing \\(\\theta\\) changes the probabilities \\(P_\\theta(y_i\\mid x_i)\\) assigned to the observed labels.",
        isCorrect: true,
      },
      {
        text: "Likelihood is automatically the posterior probability \\(P(\\theta\\mid\\text{data})\\) even when no prior distribution over parameters is specified.",
        isCorrect: false,
      },
    ],
    explanation:
      "Likelihood asks which parameter values make the fixed observed data probable under \\(P_\\theta\\). It is not the same as a posterior distribution over parameters unless a prior and Bayesian update are added.",
  },
  {
    id: "crash-probability-l3-q10",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "For the token sequence The cat sleeps, which terms belong in a left-to-right next-token likelihood factorization?",
    options: [
      {
        text: "\\(P(\\text{The})P(\\text{cat}\\mid\\text{The})P(\\text{sleeps}\\mid\\text{The cat})\\).",
        isCorrect: true,
      },
      {
        text: "A product of probabilities assigned to the actual observed tokens at their positions.",
        isCorrect: true,
      },
      {
        text: "A product over every vocabulary token at every position, regardless of which token appeared.",
        isCorrect: false,
      },
      {
        text: "\\(P(\\text{The}\\mid\\text{cat sleeps})P(\\text{cat}\\mid\\text{sleeps})P(\\text{sleeps})\\) as the standard next-token training direction.",
        isCorrect: false,
      },
    ],
    explanation:
      "Next-token likelihood multiplies probabilities assigned to the tokens that actually appear, conditioned on the previous context. It does not multiply over all vocabulary entries, and the usual autoregressive direction conditions on earlier tokens rather than future ones.",
  },
  {
    id: "crash-probability-l3-q11",
    chapter: 3,
    difficulty: "medium",
    prompt: "Why are logs used when optimizing likelihood over many examples?",
    options: [
      {
        text: "They turn products of probabilities into sums of log probabilities.",
        isCorrect: true,
      },
      {
        text: "They reduce numerical problems caused by multiplying many small probabilities.",
        isCorrect: true,
      },
      {
        text: "They change which parameter setting maximizes the objective because log is not monotonic.",
        isCorrect: false,
      },
      {
        text: "They make probabilities larger than one so gradient methods can optimize them.",
        isCorrect: false,
      },
    ],
    explanation:
      "Logs are useful because \\(\\log(ab)=\\log a+\\log b\\), turning a long product into a sum that is easier to compute and optimize. The logarithm is monotonic, so maximizing likelihood and maximizing log-likelihood choose the same parameters.",
  },
  {
    id: "crash-probability-l3-q12",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Using natural logs, the probabilities assigned to three correct labels are \\(0.80\\), \\(0.60\\), and \\(0.50\\). What is the approximate sum log-likelihood?",
    options: [
      {
        text: "\\(-1.427\\), because \\(\\log(0.80)+\\log(0.60)+\\log(0.50)\\approx-0.223-0.511-0.693\\).",
        isCorrect: true,
      },
      {
        text: "\\(0.24\\), because the log-likelihood is the product of the three probabilities.",
        isCorrect: false,
      },
      {
        text: "\\(1.427\\), because log-likelihood values for probabilities below one are positive.",
        isCorrect: false,
      },
      {
        text: "\\(-0.476\\), because the sum log-likelihood must be divided by three before it is a log-likelihood.",
        isCorrect: false,
      },
    ],
    explanation:
      "The sum log-likelihood is the sum of the log probabilities, which is approximately \\(-1.427\\). Dividing by three would give an average log-likelihood, and the product 0.24 is the original likelihood rather than its logarithm.",
  },
  {
    id: "crash-probability-l3-q13",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which comparisons correctly relate probability, log-likelihood, and negative log-likelihood for a correct answer?",
    options: [
      {
        text: "\\(\\log(0.8)>\\log(0.2)\\), so assigning 0.8 to the correct answer gives higher log-likelihood than assigning 0.2.",
        isCorrect: true,
      },
      {
        text: "\\(-\\log(0.8)<-\\log(0.2)\\), so assigning 0.8 gives lower negative log-likelihood than assigning 0.2.",
        isCorrect: true,
      },
      {
        text: "\\(\\log(0.8)\\) is positive because 0.8 is close to one.",
        isCorrect: false,
      },
      {
        text: "Lower probability for the correct answer gives lower negative log-likelihood because the model is less confident.",
        isCorrect: false,
      },
    ],
    explanation:
      "For probabilities between zero and one, log values are negative, but larger probabilities have logs closer to zero. Negative log-likelihood reverses the sign, so low correct-answer probability creates a larger loss.",
  },
  {
    id: "crash-probability-l3-q14",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "If the correct class is cat and a model assigns \\(P(\\text{cat}\\mid x)=0.20\\), what is the negative log-likelihood using natural logs?",
    options: [
      {
        text: "\\(-\\log(0.20)\\approx1.609\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\log(0.20)\\approx-1.609\\), because loss values should be negative.",
        isCorrect: false,
      },
      {
        text: "\\(1-0.20=0.80\\), because negative log-likelihood is the complement of probability.",
        isCorrect: false,
      },
      {
        text: "\\(0.20\\), because the loss equals the probability of the correct class.",
        isCorrect: false,
      },
    ],
    explanation:
      "For one example, negative log-likelihood is \\(-\\log P_\\theta(y_{\\text{true}}\\mid x)\\). It is not a probability complement; the logarithm makes the penalty grow rapidly as the correct-class probability becomes small.",
  },
  {
    id: "crash-probability-l3-q15",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "For a dataset \\(\\mathcal{D}=\\{(x_i,y_i)\\}_{i=1}^n\\), which statements about total and average negative log-likelihood are correct?",
    options: [
      {
        text: "The total NLL can be written as \\(-\\sum_{i=1}^n \\log P_\\theta(y_i\\mid x_i)\\).",
        isCorrect: true,
      },
      {
        text: "The average NLL \\(-\\frac{1}{n}\\sum_i\\log P_\\theta(y_i\\mid x_i)\\) has the same minimizer as the total NLL when \\(n\\) is fixed.",
        isCorrect: true,
      },
      {
        text: "The average NLL is interpretable as an average per-example penalty.",
        isCorrect: true,
      },
      {
        text: "Averaging NLL means training only maximizes the single largest correct-label probability in the dataset.",
        isCorrect: false,
      },
    ],
    explanation:
      "The total NLL sums penalties across examples, and the average NLL divides that sum by a fixed constant. Averaging changes the scale and aids comparison across dataset sizes, but it does not turn the objective into a maximum over only one example.",
  },
  {
    id: "crash-probability-l3-q16",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "For a one-hot target distribution \\(p\\) and model distribution \\(q\\), which statements explain why classification cross-entropy becomes negative log-likelihood?",
    options: [
      {
        text: "Cross-entropy is \\(H(p,q)=-\\sum_i p_i\\log q_i\\).",
        isCorrect: true,
      },
      {
        text: "When the correct class is cat, the one-hot target has \\(p_{\\text{cat}}=1\\) and \\(p_i=0\\) for other classes.",
        isCorrect: true,
      },
      {
        text: "Every incorrect class contributes a nonzero direct term to the loss even when its target probability is zero.",
        isCorrect: false,
      },
      {
        text: "Cross-entropy compares the target labels to raw logits without using the model probability distribution.",
        isCorrect: false,
      },
    ],
    explanation:
      "With one-hot labels, all target mass is on the correct class, so the cross-entropy sum leaves only \\(-\\log q_{\\text{correct}}\\). Incorrect probabilities matter indirectly because probabilities must sum to one, not because they have nonzero one-hot target weights.",
  },
  {
    id: "crash-probability-l3-q17",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "A soft target distribution is \\(p=(0.70,0.20,0.10)\\) for cat, dog, and fox. Which statements about cross-entropy with a model distribution \\(q\\) are correct?",
    options: [
      {
        text: "The loss uses all target weights: \\(-0.70\\log q_{\\text{cat}}-0.20\\log q_{\\text{dog}}-0.10\\log q_{\\text{fox}}\\).",
        isCorrect: true,
      },
      {
        text: "Soft labels can represent ambiguity, label smoothing, uncertain human labels, or knowledge distillation.",
        isCorrect: true,
      },
      {
        text: "One-hot cross-entropy is a special case where all target mass is placed on one class.",
        isCorrect: true,
      },
      {
        text: "Classes with nonzero target probability are ignored as long as the largest model probability is on cat.",
        isCorrect: false,
      },
    ],
    explanation:
      "Soft labels require the full cross-entropy sum because more than one class can have nonzero target probability. A model cannot satisfy such a target merely by putting the argmax on cat; it is being trained to match the distributional target more closely.",
  },
  {
    id: "crash-probability-l3-q18",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "The correct class is cat. Model A predicts \\((P(\\text{cat}),P(\\text{dog}))=(0.51,0.49)\\), and Model B predicts \\((0.99,0.01)\\). Which statements are correct?",
    options: [
      {
        text: "Both models are accurate under an argmax decision rule.",
        isCorrect: true,
      },
      {
        text: "Model B has lower cross-entropy loss because it assigns much more probability to the correct class.",
        isCorrect: true,
      },
      {
        text: "Both models have the same cross-entropy loss because both choose cat.",
        isCorrect: false,
      },
      {
        text: "Accuracy captures the difference in confidence between 0.51 and 0.99.",
        isCorrect: false,
      },
    ],
    explanation:
      "Accuracy only checks whether the selected class is correct, so both models count as correct under argmax. Cross-entropy uses the probability assigned to the correct class, so 0.99 gives a much smaller loss than 0.51.",
  },
  {
    id: "crash-probability-l3-q19",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe how cross-entropy or negative log-likelihood handles confident mistakes?",
    options: [
      {
        text: "Assigning \\(q_{\\text{correct}}=0.01\\) gives loss \\(-\\log(0.01)\\approx4.605\\).",
        isCorrect: true,
      },
      {
        text: "Assigning \\(q_{\\text{correct}}=0.90\\) gives loss \\(-\\log(0.90)\\approx0.105\\).",
        isCorrect: true,
      },
      {
        text: "The penalty grows rapidly as the probability assigned to the true class approaches zero.",
        isCorrect: true,
      },
      {
        text: "A confidently wrong model has low loss because high confidence is always rewarded.",
        isCorrect: false,
      },
    ],
    explanation:
      "The loss rewards probability assigned to the true label, not confidence by itself. A model that is confident in the wrong class leaves little probability for the correct class, which makes \\(-\\log q_{\\text{correct}}\\) large.",
  },
  {
    id: "crash-probability-l3-q20",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly connect next-token prediction in large language models to cross-entropy?",
    options: [
      {
        text: "At each position, predicting the next token is a classification problem over the vocabulary.",
        isCorrect: true,
      },
      {
        text: "The target is often represented as a one-hot distribution on the actual next token.",
        isCorrect: true,
      },
      {
        text: "The per-position loss is \\(-\\log P_\\theta(x_t\\mid x_1,\\ldots,x_{t-1})\\) for the observed token \\(x_t\\).",
        isCorrect: true,
      },
      {
        text: "Training commonly averages or sums the token-level losses over many positions.",
        isCorrect: true,
      },
    ],
    explanation:
      "Language-model training treats each observed next token as the label for a large vocabulary classification problem. Cross-entropy penalizes low probability on the token that actually appeared, and sequence training aggregates those token-level penalties.",
  },
  {
    id: "crash-probability-l3-q21",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe entropy for a discrete probability distribution?",
    options: [
      {
        text: "Entropy can be written as \\(H(p)=-\\sum_i p_i\\log p_i\\).",
        isCorrect: true,
      },
      {
        text: "Entropy is high when probability mass is spread across many plausible outcomes.",
        isCorrect: true,
      },
      {
        text: "Entropy is low when most probability mass is concentrated on one or a few outcomes.",
        isCorrect: true,
      },
      {
        text: "Entropy requires numeric outcome values and distances between them, so it cannot be used for tokens.",
        isCorrect: false,
      },
    ],
    explanation:
      "Entropy measures uncertainty in the probabilities themselves, so it works naturally for categorical outcomes such as labels or tokens. Variance needs numeric values, but entropy only needs a probability distribution over possibilities.",
  },
  {
    id: "crash-probability-l3-q22",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which distribution over yes, no, and maybe has higher entropy?",
    options: [
      {
        text: "\\((0.40,0.35,0.25)\\), because probability mass is more spread out across the three outcomes.",
        isCorrect: true,
      },
      {
        text: "\\((0.95,0.03,0.02)\\), because one outcome has the largest single probability.",
        isCorrect: false,
      },
      {
        text: "\\((0.95,0.03,0.02)\\), because lower uncertainty means higher entropy.",
        isCorrect: false,
      },
      {
        text: "Both distributions have the same entropy because they have the same number of outcomes.",
        isCorrect: false,
      },
    ],
    explanation:
      "Entropy is higher when the distribution is less concentrated and more outcomes remain plausible. The distribution \\((0.95,0.03,0.02)\\) is low entropy because it is already highly concentrated on one outcome.",
  },
  {
    id: "crash-probability-l3-q23",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly compare entropy and variance in the context of AI outputs?",
    options: [
      {
        text: "Variance is most natural for numerical random variables.",
        isCorrect: true,
      },
      {
        text: "Entropy is natural for categorical distributions such as tokens or class labels.",
        isCorrect: true,
      },
      {
        text: "It is awkward to ask for the variance of labels like cat, dog, and car without assigning arbitrary numbers to them.",
        isCorrect: true,
      },
      {
        text: "Entropy can measure uncertainty from the probabilities assigned to categorical outcomes.",
        isCorrect: true,
      },
    ],
    explanation:
      "Variance measures spread around a numerical mean, so it fits numerical outcomes best. Entropy instead measures uncertainty in a probability distribution, which makes it well suited to classes, tokens, and actions.",
  },
  {
    id: "crash-probability-l3-q24",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe temperature as a sampling-time modification of a probability distribution?",
    options: [
      {
        text: "Lower temperature makes high-probability tokens more dominant and tends to reduce randomness.",
        isCorrect: true,
      },
      {
        text: "Higher temperature spreads probability more evenly and tends to increase diversity.",
        isCorrect: true,
      },
      {
        text: "Changing temperature affects how outputs are sampled from the model distribution rather than changing what the model has learned.",
        isCorrect: true,
      },
      {
        text: "Increasing temperature retrains the model so the training-set cross-entropy is automatically lower.",
        isCorrect: false,
      },
    ],
    explanation:
      "Temperature changes the sharpness of the distribution used for sampling at generation time. It can make outputs more deterministic or more diverse, but it is not a training update and does not by itself improve the learned likelihood objective.",
  },
  {
    id: "crash-probability-l3-q25",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "In a reinforcement learning policy \\(\\pi(a\\mid s)\\), which statements about entropy are correct?",
    options: [
      {
        text: "A high-entropy policy assigns meaningful probability to several actions and can support exploration.",
        isCorrect: true,
      },
      {
        text: "A low-entropy policy chooses one action almost always when most probability mass is concentrated there.",
        isCorrect: true,
      },
      {
        text: "A high-entropy policy always has higher expected return than a low-entropy policy.",
        isCorrect: false,
      },
      {
        text: "Policy entropy is the same quantity as the expected future return.",
        isCorrect: false,
      },
    ],
    explanation:
      "Entropy describes how spread out the action probabilities are, not how much reward the policy will obtain. Exploration can be useful, but a policy still needs to choose actions that lead to reward in the environment.",
  },
  {
    id: "crash-probability-l3-q26",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Using natural logs, compare distributions \\(A=(0.5,0.5)\\) and \\(B=(0.9,0.1)\\). Which statement is correct?",
    options: [
      {
        text: "\\(H(A)=\\log 2\\) and \\(H(A)>H(B)\\), because \\(A\\) is more evenly spread.",
        isCorrect: true,
      },
      {
        text: "\\(H(B)=\\log 2\\), because both distributions have two outcomes.",
        isCorrect: false,
      },
      {
        text: "\\(H(A)=0\\), because the two probabilities in \\(A\\) are equal.",
        isCorrect: false,
      },
      {
        text: "\\(H(B)>H(A)\\), because \\(B\\) has a larger maximum probability.",
        isCorrect: false,
      },
    ],
    explanation:
      "For two outcomes, entropy is maximized by the uniform distribution \\((0.5,0.5)\\), where the uncertainty is one natural-log unit of \\(\\log 2\\). Concentrating mass at \\((0.9,0.1)\\) lowers uncertainty even though the number of possible outcomes is unchanged.",
  },
  {
    id: "crash-probability-l3-q27",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the supervised neural-network training pipeline for probabilistic prediction?",
    options: [
      {
        text: "An input is processed by a neural network to produce logits.",
        isCorrect: true,
      },
      {
        text: "Softmax can convert logits into a probability distribution over classes or tokens.",
        isCorrect: true,
      },
      {
        text: "A loss such as negative log-likelihood or cross-entropy compares the predicted distribution to the observed target.",
        isCorrect: true,
      },
      {
        text: "Gradient updates adjust parameters so the model can assign higher probability to observed correct outputs.",
        isCorrect: true,
      },
    ],
    explanation:
      "The core pipeline is input to network, logits, softmax probabilities, loss, and parameter update. This connects the notation \\(P(y\\mid x)\\) to the practical mechanics of neural-network training.",
  },
  {
    id: "crash-probability-l3-q28",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "In the cross-entropy expression \\(H(p,q)=-\\sum_i p_i\\log q_i\\), which interpretations of \\(p\\) and \\(q\\) are correct?",
    options: [
      {
        text: "\\(p\\) is the target distribution, such as a one-hot label or a soft label.",
        isCorrect: true,
      },
      {
        text: "\\(q\\) is the model distribution produced after normalization.",
        isCorrect: true,
      },
      {
        text: "For one-hot cat labels, \\(p_{\\text{cat}}=1\\) and the other target probabilities are zero.",
        isCorrect: true,
      },
      {
        text: "For soft labels, multiple entries of \\(p\\) can be nonzero and should be included in the loss.",
        isCorrect: true,
      },
    ],
    explanation:
      "Cross-entropy compares a target distribution with the model's predicted distribution. The one-hot case collapses to negative log-likelihood for the correct class, while soft labels keep multiple weighted terms.",
  },
  {
    id: "crash-probability-l3-q29",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "After a training example produces high loss because the correct class has low probability, which effects are consistent with a gradient-based update on softmax-cross-entropy?",
    options: [
      {
        text: "The model parameters are adjusted rather than changing the observed label in the dataset.",
        isCorrect: true,
      },
      {
        text: "If the update increases \\(q_{\\text{correct}}\\) while other conditions are comparable, the negative log-likelihood for that example decreases.",
        isCorrect: true,
      },
      {
        text: "Changing logits can change normalized probabilities because softmax couples the classes through a shared denominator.",
        isCorrect: true,
      },
      {
        text: "Repeating this across examples is a practical way to increase probability assigned to observed outputs.",
        isCorrect: true,
      },
    ],
    explanation:
      "Training treats the observed data as fixed and updates parameters so the model distribution better fits that data. Because softmax probabilities are coupled, changing logits can redistribute probability mass and reduce the loss when the correct label receives more probability.",
  },
  {
    id: "crash-probability-l3-q30",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Mathematically, what happens to likelihood-based objectives if a model assigns probability zero to one observed correct label?",
    options: [
      {
        text: "The product likelihood for the dataset becomes zero if that example is included.",
        isCorrect: true,
      },
      {
        text: "The log-likelihood includes \\(\\log 0\\), which is not finite and behaves like \\(-\\infty\\).",
        isCorrect: true,
      },
      {
        text: "The negative log-likelihood has an infinite penalty in the idealized mathematical objective.",
        isCorrect: true,
      },
      {
        text: "This illustrates why near-zero probability on the observed answer is punished so strongly.",
        isCorrect: true,
      },
    ],
    explanation:
      "A single zero probability makes a product likelihood zero and sends the log-likelihood to negative infinity in the mathematical limit. Softmax with finite logits usually gives positive probabilities, but the near-zero case still creates a very large NLL penalty.",
  },
  {
    id: "crash-probability-l3-q31",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which properties help softmax convert arbitrary logits into a valid categorical distribution?",
    options: [
      {
        text: "Exponentials make each unnormalized score positive.",
        isCorrect: true,
      },
      {
        text: "Dividing by the sum of exponentials makes the probabilities sum to one.",
        isCorrect: true,
      },
      {
        text: "Higher logits still receive higher probabilities than lower logits.",
        isCorrect: true,
      },
      {
        text: "Negative logits can still produce positive probabilities after exponentiation.",
        isCorrect: true,
      },
    ],
    explanation:
      "Softmax handles arbitrary real-valued logits by exponentiating and normalizing them. This produces positive probabilities that sum to one while preserving the ordering induced by the original logits.",
  },
  {
    id: "crash-probability-l3-q32",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "For one-hot classification cross-entropy, which statements correctly describe how incorrect classes affect the loss?",
    options: [
      {
        text: "The direct loss term is \\(-\\log q_{\\text{correct}}\\).",
        isCorrect: true,
      },
      {
        text: "Incorrect classes matter indirectly because increasing \\(q_{\\text{correct}}\\) requires probability mass to move away from other classes.",
        isCorrect: true,
      },
      {
        text: "Increasing the correct-class probability from 0.70 to 0.80 lowers the loss even if the incorrect classes share the remaining mass differently.",
        isCorrect: true,
      },
      {
        text: "The largest incorrect-class probability contributes a separate positive term even when its one-hot target weight is zero.",
        isCorrect: false,
      },
    ],
    explanation:
      "With a one-hot target, the sum directly keeps only the correct class term. Incorrect probabilities still matter through the normalization constraint, because probability assigned to them is probability not assigned to the correct class.",
  },
  {
    id: "crash-probability-l3-q33",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "For an LLM token sequence of length \\(T\\), which statements match the average next-token negative log-likelihood objective?",
    options: [
      {
        text: "A common form is \\(-\\frac{1}{T}\\sum_{t=1}^T\\log P_\\theta(x_t\\mid x_1,\\ldots,x_{t-1})\\).",
        isCorrect: true,
      },
      {
        text: "Each term penalizes low probability assigned to the observed token \\(x_t\\).",
        isCorrect: true,
      },
      {
        text: "Averaging by \\(T\\) gives a per-token loss scale that is easier to compare across sequence lengths.",
        isCorrect: true,
      },
      {
        text: "The conditioning context grows from previous tokens in an autoregressive model.",
        isCorrect: true,
      },
    ],
    explanation:
      "The LLM objective sums or averages the log-probability penalties for observed next tokens. The average form scales the loss per token while preserving the goal of assigning high probability to the actual sequence.",
  },
  {
    id: "crash-probability-l3-q34",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "For a fixed dataset of \\(n\\) examples, which optimization objectives choose the same parameter values when probabilities are positive?",
    options: [
      {
        text: "Maximizing \\(\\prod_i P_\\theta(y_i\\mid x_i)\\).",
        isCorrect: true,
      },
      {
        text: "Maximizing \\(\\sum_i\\log P_\\theta(y_i\\mid x_i)\\).",
        isCorrect: true,
      },
      {
        text: "Minimizing \\(-\\sum_i\\log P_\\theta(y_i\\mid x_i)\\).",
        isCorrect: true,
      },
      {
        text: "Minimizing \\(-\\frac{1}{n}\\sum_i\\log P_\\theta(y_i\\mid x_i)\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "Log is monotonic, so maximizing likelihood and maximizing log-likelihood are equivalent for positive probabilities. Negating converts maximization into minimization, and dividing by fixed \\(n\\) only rescales the objective.",
  },
  {
    id: "crash-probability-l3-q35",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which AI examples correctly instantiate the idea of producing a probability distribution from logits?",
    options: [
      {
        text: "An image classifier can map class logits to \\(P(\\text{class}\\mid\\text{image})\\).",
        isCorrect: true,
      },
      {
        text: "An LLM can map vocabulary-token logits to \\(P(\\text{next token}\\mid\\text{context})\\).",
        isCorrect: true,
      },
      {
        text: "A policy network can map action logits to \\(\\pi(a\\mid s)\\).",
        isCorrect: true,
      },
      {
        text: "A classifier, language model, or policy can still separate probability estimation from the later decision or sampling step.",
        isCorrect: true,
      },
    ],
    explanation:
      "Classifiers, LLMs, and policy networks all commonly produce logits that are normalized into probabilities. The resulting distribution can then support classification, token generation, or action sampling without collapsing probability estimation and decision-making into the same concept.",
  },
  {
    id: "crash-probability-l3-q36",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about softmax probabilities and final choices are correct?",
    options: [
      {
        text: "Softmax preserves ranking, so the largest logit receives the largest probability.",
        isCorrect: true,
      },
      {
        text: "An argmax decision rule chooses the class or token with the largest softmax probability.",
        isCorrect: true,
      },
      {
        text: "A stochastic sampler must always output the highest-probability token.",
        isCorrect: false,
      },
      {
        text: "A class with a lower logit can never be sampled because softmax gives it zero probability.",
        isCorrect: false,
      },
    ],
    explanation:
      "Softmax ranking and argmax decisions line up, but sampling can still choose lower-probability outputs when they have positive probability. This is why the distribution contains more information than a single chosen output.",
  },
  {
    id: "crash-probability-l3-q37",
    chapter: 3,
    difficulty: "medium",
    type: "assertion-reason",
    prompt:
      "Assertion: Softmax outputs a valid probability distribution over classes.\n\nReason: The softmax denominator is the sum of the original logits, so the raw scores themselves are forced to sum to one.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: true },
      { text: "Assertion is false, Reason is true.", isCorrect: false },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: false,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "The assertion is true because softmax produces positive probabilities that sum to one. The reason is false because the denominator is the sum of exponentiated logits, not the sum of the original raw logits.",
  },
  {
    id: "crash-probability-l3-q38",
    chapter: 3,
    difficulty: "medium",
    type: "assertion-reason",
    prompt:
      "Assertion: In maximum likelihood training, the observed dataset is the quantity adjusted by the optimizer.\n\nReason: The model parameters \\(\\theta\\) are changed so observed labels or tokens receive higher probability.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: false },
      { text: "Assertion is false, Reason is true.", isCorrect: true },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: false,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "The assertion is false because likelihood treats the observed data as fixed during training. The reason is true: optimization changes parameters so the model assigns higher probability to what actually occurred in the dataset.",
  },
  {
    id: "crash-probability-l3-q39",
    chapter: 3,
    difficulty: "hard",
    type: "assertion-reason",
    prompt:
      "Assertion: With one-hot labels, classification cross-entropy equals the negative log-likelihood of the correct class.\n\nReason: The target distribution puts all probability mass on the correct class, so \\(-\\sum_i p_i\\log q_i\\) reduces to \\(-\\log q_{\\text{correct}}\\).",
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
      "Both statements are true, and the reason explains the assertion by showing how the cross-entropy sum collapses under a one-hot target. The incorrect classes have zero target weight in the direct sum, leaving the negative log probability of the observed correct class.",
  },
  {
    id: "crash-probability-l3-q40",
    chapter: 3,
    difficulty: "medium",
    type: "assertion-reason",
    prompt:
      "Assertion: A low-entropy next-token distribution is more concentrated than a high-entropy next-token distribution.\n\nReason: Softmax maps logits to positive probabilities that sum to one.",
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
      "Both statements are true: low entropy means probability mass is more concentrated, and softmax does produce valid probabilities from logits. The reason does not explain the entropy comparison, because normalization alone does not say whether the resulting distribution is concentrated or spread out.",
  },
  {
    id: "crash-probability-l3-q41",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "For logits \\(z=(4,1,-2)\\) and softmax probabilities \\(q_i=\\frac{e^{z_i}}{\\sum_j e^{z_j}}\\), which statement is correct?",
    options: [
      {
        text: "The odds ratio \\(q_1/q_2\\) is \\(e^{4-1}=e^3\\).",
        isCorrect: true,
      },
      {
        text: "The probability difference \\(q_1-q_2\\) is exactly \\(4-1=3\\).",
        isCorrect: false,
      },
      {
        text: "The third class receives probability zero because its logit is negative.",
        isCorrect: false,
      },
      {
        text: "Subtracting 4 from all logits makes the first class probability exactly one.",
        isCorrect: false,
      },
    ],
    explanation:
      "Softmax odds between two classes depend only on the difference between their logits, so \\(q_1/q_2=e^{z_1-z_2}\\). Probability differences are not logit differences, negative logits still exponentiate to positive values, and a common shift does not change any softmax probability.",
  },
  {
    id: "crash-probability-l3-q42",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Let \\(q_i(T)=\\frac{e^{z_i/T}}{\\sum_j e^{z_j/T}}\\) for temperature \\(T>0\\). Which statements are correct?",
    options: [
      {
        text: "For two logits with \\(z_a>z_b\\), lowering \\(T\\) below 1 increases the odds ratio \\(q_a(T)/q_b(T)\\).",
        isCorrect: true,
      },
      {
        text: "As \\(T\\to\\infty\\), the distribution approaches uniform over the classes with finite logits.",
        isCorrect: true,
      },
      {
        text: "A positive temperature can reverse the ranking of two unequal logits.",
        isCorrect: false,
      },
      {
        text: "Changing \\(T\\) changes the learned parameters \\(\\theta\\) in the same way as a gradient update.",
        isCorrect: false,
      },
    ],
    explanation:
      "Temperature rescales logit differences: \\(q_a/q_b=e^{(z_a-z_b)/T}\\), so smaller positive \\(T\\) sharpens odds and very large \\(T\\) washes finite differences toward uniformity. Positive rescaling preserves rankings, and sampling temperature is not itself a training update to the model parameters.",
  },
  {
    id: "crash-probability-l3-q43",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "For one example with one-hot target class \\(c\\), softmax probabilities \\(q_j\\), and loss \\(\\ell=-\\log q_c\\), which gradient statements are correct?",
    options: [
      {
        text: "For the correct class, \\(\\frac{\\partial \\ell}{\\partial z_c}=q_c-1\\).",
        isCorrect: true,
      },
      {
        text: "For an incorrect class \\(j\\neq c\\), \\(\\frac{\\partial \\ell}{\\partial z_j}=q_j\\).",
        isCorrect: true,
      },
      {
        text: "The gradient components over all logits sum to zero.",
        isCorrect: true,
      },
      {
        text: "Increasing the correct-class logit while holding other logits fixed increases the loss.",
        isCorrect: false,
      },
    ],
    explanation:
      "For softmax plus one-hot cross-entropy, the logit gradient has the compact form \\(q-p\\). The correct-class component is negative when \\(q_c<1\\), so gradient descent tends to raise the correct logit relative to the others, while incorrect-class components push their logits down.",
  },
  {
    id: "crash-probability-l3-q44",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "For a two-class softmax with logits \\(z_1,z_2\\), which statements are correct?",
    options: [
      {
        text: "\\(q_1=\\frac{e^{z_1}}{e^{z_1}+e^{z_2}}=\\frac{1}{1+e^{-(z_1-z_2)}}\\).",
        isCorrect: true,
      },
      {
        text: "The log odds satisfy \\(\\log(q_1/q_2)=z_1-z_2\\).",
        isCorrect: true,
      },
      {
        text: "Adding the same constant to \\(z_1\\) and \\(z_2\\) leaves both probabilities unchanged.",
        isCorrect: true,
      },
      {
        text: "For a positive example of class 1, the loss can be written as \\(-\\log q_1\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The two-class softmax reduces to the logistic sigmoid applied to the logit difference. This makes the log-odds interpretation explicit, shows common-shift invariance, and connects binary classification loss to the same negative log-likelihood principle.",
  },
  {
    id: "crash-probability-l3-q45",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "A model assigns correct-label probabilities \\(0.9\\) and \\(0.1\\) to two examples. Which statement correctly describes the average negative log-likelihood?",
    options: [
      {
        text: "It is \\(-\\frac{1}{2}(\\log0.9+\\log0.1)=-\\log\\sqrt{0.09}\\approx1.204\\).",
        isCorrect: true,
      },
      {
        text: "It is \\(-\\log((0.9+0.1)/2)\\approx0.693\\) because likelihood uses the arithmetic mean probability.",
        isCorrect: false,
      },
      {
        text: "It is \\(0.9\\cdot0.1=0.09\\) because average NLL is the product likelihood.",
        isCorrect: false,
      },
      {
        text: "It changes if the two probabilities are assigned to the examples in the opposite order.",
        isCorrect: false,
      },
    ],
    explanation:
      "Average NLL is the negative mean of log probabilities, which equals the negative log of the geometric mean of the correct-label probabilities. It is not the arithmetic-mean loss, not the raw likelihood product, and it is symmetric in the examples.",
  },
  {
    id: "crash-probability-l3-q46",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "For an autoregressive token sequence \\(x_1,\\ldots,x_T\\), which statements correctly interpret \\(\\prod_{t=1}^T P_\\theta(x_t\\mid x_1,\\ldots,x_{t-1})\\)?",
    options: [
      {
        text: "It is a chain-rule factorization of the probability assigned to the observed sequence under the model.",
        isCorrect: true,
      },
      {
        text: "It multiplies probabilities of the observed tokens, each conditioned on the previous context.",
        isCorrect: true,
      },
      {
        text: "It assumes the tokens are independent because all factors are multiplied.",
        isCorrect: false,
      },
      {
        text: "It requires summing over all vocabulary tokens at every position before any observed-token probability is used.",
        isCorrect: false,
      },
    ],
    explanation:
      "The product is the standard left-to-right decomposition of a sequence probability into conditional next-token probabilities. Multiplication does not mean token independence here, because each factor can condition on the previous tokens.",
  },
  {
    id: "crash-probability-l3-q47",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "For classes ordered as \\((\\text{cat},\\text{dog},\\text{fox})\\), a soft target is \\(p=(0.7,0.2,0.1)\\). Model A predicts \\(q_A=(0.7,0.2,0.1)\\), while Model B predicts \\(q_B=(0.9,0.05,0.05)\\). Which statements are correct?",
    options: [
      {
        text: "Model A minimizes cross-entropy among model distributions that can match \\(p\\) exactly.",
        isCorrect: true,
      },
      {
        text: "Model B can have the correct argmax class while still being worse under soft-label cross-entropy.",
        isCorrect: true,
      },
      {
        text: "The loss for soft labels includes terms for dog and fox because their target probabilities are nonzero.",
        isCorrect: true,
      },
      {
        text: "Soft-label cross-entropy ignores all classes except the largest target-probability class.",
        isCorrect: false,
      },
    ],
    explanation:
      "Soft-label cross-entropy trains the whole predicted distribution, not only the most likely class. A model can put the argmax on the same class but still assign too little probability to other target-supported outcomes, increasing the distributional mismatch.",
  },
  {
    id: "crash-probability-l3-q48",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about \\(-\\log q_{\\text{correct}}\\) are correct for \\(0<q_{\\text{correct}}\\leq1\\)?",
    options: [
      {
        text: "The loss is nonnegative.",
        isCorrect: true,
      },
      {
        text: "The loss is zero exactly when the correct answer receives probability one.",
        isCorrect: true,
      },
      {
        text: "The loss increases without bound as the correct-answer probability approaches zero.",
        isCorrect: true,
      },
      {
        text: "For independent examples, total NLL adds the per-example losses.",
        isCorrect: true,
      },
    ],
    explanation:
      "Natural logarithms of probabilities at most one are nonpositive, so negating them gives a nonnegative loss. The log form also explains both the additive structure over examples and the severe penalty for assigning near-zero probability to the observed answer.",
  },
  {
    id: "crash-probability-l3-q49",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "For fixed target distribution \\(p\\), which statement correctly uses the relationship \\(H(p,q)=H(p)+D_{\\mathrm{KL}}(p\\parallel q)\\)?",
    options: [
      {
        text: "Minimizing cross-entropy over \\(q\\) is equivalent to minimizing \\(D_{\\mathrm{KL}}(p\\parallel q)\\) because \\(H(p)\\) does not depend on \\(q\\).",
        isCorrect: true,
      },
      {
        text: "Minimizing cross-entropy over \\(q\\) is equivalent to changing \\(p\\) until \\(H(p)\\) is as small as possible.",
        isCorrect: false,
      },
      {
        text: "The equality implies \\(D_{\\mathrm{KL}}(p\\parallel q)=D_{\\mathrm{KL}}(q\\parallel p)\\) for all distributions.",
        isCorrect: false,
      },
      {
        text: "For soft labels, cross-entropy is minimized by assigning probability one to the single most likely target class.",
        isCorrect: false,
      },
    ],
    explanation:
      "When \\(p\\) is fixed, \\(H(p)\\) is a constant, so optimizing cross-entropy with respect to \\(q\\) is optimizing the mismatch term. KL divergence is not symmetric, and soft targets generally require matching probability mass beyond only the modal class.",
  },
  {
    id: "crash-probability-l3-q50",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "For a categorical distribution over \\(k\\) possible outcomes using natural logs, which entropy statements are correct?",
    options: [
      {
        text: "The maximum entropy is achieved by the uniform distribution \\(p_i=1/k\\).",
        isCorrect: true,
      },
      {
        text: "At the uniform distribution, \\(H(p)=\\log k\\).",
        isCorrect: true,
      },
      {
        text: "A distribution concentrated entirely on one outcome has entropy \\(\\log k\\).",
        isCorrect: false,
      },
      {
        text: "Entropy must increase whenever the most likely class probability increases.",
        isCorrect: false,
      },
    ],
    explanation:
      "Uniform mass is the most uncertain categorical distribution, giving \\(-k(1/k)\log(1/k)=\\log k\\). Concentrating mass on one outcome lowers entropy, and increasing the largest probability usually makes the distribution less spread out rather than more uncertain.",
  },
  {
    id: "crash-probability-l3-q51",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "The correct class is \\(c\\). Model A assigns \\(q_c=0.45\\) but gives a different class probability \\(0.46\\); Model B assigns \\(q_c=0.40\\) and makes \\(c\\) the argmax. Which statements are correct?",
    options: [
      {
        text: "Model A is inaccurate under argmax but has lower cross-entropy loss for this example than Model B.",
        isCorrect: true,
      },
      {
        text: "For a one-hot target, the direct cross-entropy comparison depends on \\(-\\log q_c\\).",
        isCorrect: true,
      },
      {
        text: "Accuracy and cross-entropy can rank these two models differently on a single example.",
        isCorrect: true,
      },
      {
        text: "Model B must have lower cross-entropy because it chooses the correct class.",
        isCorrect: false,
      },
    ],
    explanation:
      "Cross-entropy is sensitive to the probability assigned to the true class, while accuracy only checks the final argmax decision. Since \\(-\\log(0.45)<-\\log(0.40)\\), Model A has lower loss even though its top predicted class is wrong.",
  },
  {
    id: "crash-probability-l3-q52",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "For one-hot class \\(c\\), logits \\(z_j\\), and loss \\(\\ell=-\\log \\frac{e^{z_c}}{\\sum_j e^{z_j}}\\), which statements are correct?",
    options: [
      {
        text: "The loss can be written as \\(\\ell=-z_c+\\log\\sum_j e^{z_j}\\).",
        isCorrect: true,
      },
      {
        text: "Adding the same constant to every logit leaves \\(\\ell\\) unchanged.",
        isCorrect: true,
      },
      {
        text: "Increasing \\(z_c\\) while holding the other logits fixed lowers \\(\\ell\\).",
        isCorrect: true,
      },
      {
        text: "Increasing an incorrect-class logit while holding \\(z_c\\) fixed raises \\(\\ell\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The log-sum-exp form follows by expanding \\(-\\log q_c\\). It shows both shift invariance and the competing effects of correct versus incorrect logits: raising the correct logit helps, while raising a rival logit increases the normalization term.",
  },
  {
    id: "crash-probability-l3-q53",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "For two logits \\((2,0)\\), compare sampling temperatures \\(T=1\\) and \\(T=2\\). Which statement is correct?",
    options: [
      {
        text: "At \\(T=2\\), the log-odds shrink from \\(2\\) to \\(1\\), making the distribution less concentrated and higher entropy.",
        isCorrect: true,
      },
      {
        text: "The entropy is identical because the higher-logit class stays the higher-probability class.",
        isCorrect: false,
      },
      {
        text: "The top-class probability at \\(T=2\\) is computed as \\(2/(2+0)\\).",
        isCorrect: false,
      },
      {
        text: "Changing from \\(T=1\\) to \\(T=2\\) performs a maximum-likelihood update on the model parameters.",
        isCorrect: false,
      },
    ],
    explanation:
      "Temperature divides logits before softmax, so the two-class log-odds are divided by \\(T\\). The rank is preserved for positive temperature, but the distribution becomes flatter at \\(T=2\\), and no model weights are retrained by changing sampling temperature.",
  },
  {
    id: "crash-probability-l3-q54",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "For \\(H(p,q)=-\\sum_i p_i\\log q_i\\), which conditions or consequences are correct?",
    options: [
      {
        text: "\\(q\\) must be a valid probability distribution, with nonnegative entries summing to one.",
        isCorrect: true,
      },
      {
        text: "If \\(p_i>0\\) and \\(q_i=0\\), the idealized cross-entropy has an infinite penalty.",
        isCorrect: true,
      },
      {
        text: "\\(q_i\\) can be negative when the corresponding logit is negative.",
        isCorrect: false,
      },
      {
        text: "\\(p\\) must be one-hot; otherwise cross-entropy is undefined.",
        isCorrect: false,
      },
    ],
    explanation:
      "Cross-entropy is defined between distributions, so model probabilities must be nonnegative and normalized. Soft labels are valid targets, but assigning zero model probability where the target has positive mass makes the logarithmic penalty diverge.",
  },
  {
    id: "crash-probability-l3-q55",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly connect maximum likelihood to empirical risk minimization for a fixed dataset?",
    options: [
      {
        text: "Average NLL is the empirical mean of \\(-\\log P_\\theta(y_i\\mid x_i)\\) over training examples.",
        isCorrect: true,
      },
      {
        text: "For independent examples, minimizing total NLL is equivalent to maximizing the product likelihood.",
        isCorrect: true,
      },
      {
        text: "Duplicating every example the same number of times scales total NLL but leaves the average NLL value and minimizer unchanged.",
        isCorrect: true,
      },
      {
        text: "Maximum likelihood training lowers the probabilities assigned to observed labels so that the model stays uncertain.",
        isCorrect: false,
      },
    ],
    explanation:
      "Average NLL is an empirical expectation of a per-example loss over the observed dataset. Total and average forms have the same minimizer for a fixed-size dataset, while duplicating all examples uniformly changes total scale but not the average objective.",
  },
  {
    id: "crash-probability-l3-q56",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly connect softmax probability models to action policies and imitation-style losses?",
    options: [
      {
        text: "A policy network can use softmax logits to define \\(\\pi_\\theta(a\\mid s)\\).",
        isCorrect: true,
      },
      {
        text: "If an observed action \\(a\\) is treated as the target, a supervised imitation loss can use \\(-\\log \\pi_\\theta(a\\mid s)\\).",
        isCorrect: true,
      },
      {
        text: "A non-argmax action can still be sampled when it has positive policy probability.",
        isCorrect: true,
      },
      {
        text: "Policy entropy can be used as a separate term to encourage exploration in reinforcement learning.",
        isCorrect: true,
      },
    ],
    explanation:
      "Softmax over action logits gives a categorical policy, and observed-action imitation fits naturally into the same negative-log-probability framework. Reinforcement learning then adds decision-making over time, where entropy and exploration can matter beyond plain supervised prediction.",
  },
  {
    id: "crash-probability-l3-q57",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "For a soft binary target \\(p=(0.5,0.5)\\) and model prediction \\(q=(0.8,0.2)\\), which value is the cross-entropy using natural logs?",
    options: [
      {
        text: "\\(-0.5\\log0.8-0.5\\log0.2\\approx0.916\\).",
        isCorrect: true,
      },
      {
        text: "\\(-\\log0.8\\approx0.223\\), because the first class has the larger model probability.",
        isCorrect: false,
      },
      {
        text: "\\(-\\log0.5\\approx0.693\\), because the target distribution is uniform.",
        isCorrect: false,
      },
      {
        text: "\\(0.8\\cdot0.5+0.2\\cdot0.5=0.5\\), because cross-entropy is an arithmetic dot product.",
        isCorrect: false,
      },
    ],
    explanation:
      "Soft-label cross-entropy weights the log model probabilities by the target probabilities, so both entries contribute. The one-hot shortcut \\(-\\log q_{\\text{correct}}\\) does not apply when the target has nonzero mass on both classes.",
  },
  {
    id: "crash-probability-l3-q58",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "If a model's average NLL over observed tokens is \\(a\\), which statements correctly interpret \\(e^{-a}\\)?",
    options: [
      {
        text: "\\(e^{-a}\\) is the geometric mean probability assigned to the observed tokens.",
        isCorrect: true,
      },
      {
        text: "Reducing average NLL from \\(0.7\\) to \\(0.5\\) multiplies this geometric mean by \\(e^{0.2}\\).",
        isCorrect: true,
      },
      {
        text: "\\(e^{-a}\\) is the arithmetic mean of the observed-token probabilities.",
        isCorrect: false,
      },
      {
        text: "A larger average NLL means a larger geometric mean probability assigned to the observed tokens.",
        isCorrect: false,
      },
    ],
    explanation:
      "Average NLL is \\(-\\frac{1}{T}\\sum_t\\log p_t\\), so exponentiating its negative gives \\((\\prod_t p_t)^{1/T}\\). This is a geometric mean, and lowering average NLL increases that geometric mean multiplicatively.",
  },
  {
    id: "crash-probability-l3-q59",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which notation statements are correct across the likelihood and cross-entropy formulas?",
    options: [
      {
        text: "In \\(P_\\theta(y_i\\mid x_i)\\), \\(\\theta\\) denotes model parameters that training changes.",
        isCorrect: true,
      },
      {
        text: "In \\(\\sum_{i=1}^n \\log P_\\theta(y_i\\mid x_i)\\), the index \\(i\\) ranges over dataset examples.",
        isCorrect: true,
      },
      {
        text: "In \\(H(p,q)=-\\sum_i p_i\\log q_i\\) for one example, the index \\(i\\) usually ranges over classes or outcomes.",
        isCorrect: true,
      },
      {
        text: "The symbol \\(i\\) must always refer to the same kind of object in every probability formula.",
        isCorrect: false,
      },
    ],
    explanation:
      "Mathematical notation reuses indices locally, so the meaning of \\(i\\) depends on the formula: examples in a dataset sum, classes in a cross-entropy sum. The parameter subscript \\(\\theta\\) marks that the probability model changes as training updates the network.",
  },
  {
    id: "crash-probability-l3-q60",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "For a minibatch of examples with logits \\(z^{(m)}\\), correct classes \\(c_m\\), and per-example losses \\(\\ell_m=-z^{(m)}_{c_m}+\\log\\sum_j e^{z^{(m)}_j}\\), which statements are correct?",
    options: [
      {
        text: "The minibatch total NLL is \\(\\sum_m \\ell_m\\).",
        isCorrect: true,
      },
      {
        text: "The minibatch average NLL is \\(\\frac{1}{M}\\sum_{m=1}^M \\ell_m\\) for \\(M\\) examples.",
        isCorrect: true,
      },
      {
        text: "The same parameter vector \\(\\theta\\) can affect many \\(z^{(m)}\\), so a gradient step aggregates information across examples.",
        isCorrect: true,
      },
      {
        text: "For a fixed example, decreasing the margin of the correct logit relative to competing logits tends to increase the loss.",
        isCorrect: true,
      },
    ],
    explanation:
      "Minibatch training adds or averages the same per-example log-sum-exp losses used for individual examples. Because the logits are produced by shared parameters, the gradient combines evidence from multiple examples about how to shift probability toward observed correct outputs.",
  },
];
