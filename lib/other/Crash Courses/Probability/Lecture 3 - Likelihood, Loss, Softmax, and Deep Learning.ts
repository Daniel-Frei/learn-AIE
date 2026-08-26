import { Question } from "../../../quiz";

type OptionSpec = readonly [text: string, isCorrect: boolean];

function makeQuestion(
  id: string,
  difficulty: Question["difficulty"],
  prompt: string,
  options: readonly [OptionSpec, OptionSpec, OptionSpec, OptionSpec],
  explanation: string,
): Question {
  return {
    id,
    chapter: 3,
    difficulty,
    prompt,
    options: options.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

export const CrashCourseProbabilityL3Questions: Question[] = [
  // Logits and softmax
  makeQuestion(
    "crash-probability-l3-q61",
    "easy",
    "A classifier produces logits \\((2.0,1.0,-1.0)\\). Which statement correctly describes these numbers before softmax?",
    [
      [
        "They are unnormalized scores whose ordering influences class probabilities.",
        true,
      ],
      ["They are probabilities because each lies below 2.", false],
      ["They must sum to one before the prediction is valid.", false],
      ["They are observed class frequencies from the training set.", false],
    ],
    "Logits are real-valued model scores, so they can be negative and need not sum to one. Softmax exponentiates and normalizes them into probabilities; their relative differences, not a probability interpretation before that step, drive the result.",
  ),
  makeQuestion(
    "crash-probability-l3-q62",
    "easy",
    "Which properties does softmax give to a finite vector of logits?",
    [
      ["Every output probability is positive.", true],
      ["The output probabilities sum to one.", true],
      ["Every output probability is equal when logits differ.", false],
      ["The largest logit is converted to probability exactly one.", false],
    ],
    "Exponentials are positive and division by their sum normalizes the outputs to one. Unequal logits generally yield unequal probabilities, and a finite largest logit still competes with positive mass from the other classes.",
  ),
  makeQuestion(
    "crash-probability-l3-q63",
    "medium",
    "Two logits are \\(z_A=\\ln 3\\) and \\(z_B=0\\). Softmax is applied only to these classes. Which statements are correct?",
    [
      ["The unnormalized weights are 3 and 1.", true],
      ["\\(P(A)=3/4\\).", true],
      ["\\(P(B)=1/4\\).", true],
      ["The logit difference implies \\(P(A)-P(B)=\\ln 3\\).", false],
    ],
    "Exponentiating produces weights \\(e^{\\ln3}=3\\) and \\(e^0=1\\), which normalize to three quarters and one quarter. Logit differences control probability ratios, not probability differences in the same numerical units.",
  ),
  makeQuestion(
    "crash-probability-l3-q64",
    "hard",
    "A model has logits \\((1001,1000,999)\\). Which softmax procedures are mathematically equivalent and numerically sensible?",
    [
      ["Subtract 1001 and apply softmax to \\((0,-1,-2)\\).", true],
      ["Use \\(e^{z_i-\\max_j z_j}/\\sum_k e^{z_k-\\max_j z_j}\\).", true],
      ["Divide each logit by their sum and call the result softmax.", false],
      [
        "Clip the largest logit to zero while leaving the others unchanged.",
        false,
      ],
    ],
    "Adding or subtracting the same constant from every logit leaves softmax unchanged because the common exponential factor cancels. Max subtraction prevents huge exponentials, whereas dividing raw scores or changing only one logit changes the modeled distribution.",
  ),
  makeQuestion(
    "crash-probability-l3-q65",
    "medium",
    "Which statements about softmax probability ratios are correct?",
    [
      ["\\(p_i/p_j=e^{z_i-z_j}\\).", true],
      ["Increasing only \\(z_i\\) increases \\(p_i/p_j\\).", true],
      [
        "Adding the same constant to all logits leaves every ratio unchanged.",
        true,
      ],
      [
        "A logit gap of zero gives equal probability to the two compared classes.",
        true,
      ],
    ],
    "The normalizing denominator cancels in a ratio, leaving the exponential of the logit difference. This explains softmax shift invariance, monotonic response to a class's score, and equal odds for equal logits.",
  ),
  makeQuestion(
    "crash-probability-l3-q66",
    "hard",
    "Class A has logit 2 and class B has logit 0. Which statements are correct about their softmax probabilities within any larger class set?",
    [
      ["The odds ratio \\(p_A/p_B=e^2\\).", true],
      [
        "Adding another class changes each absolute probability but not \\(p_A/p_B\\).",
        true,
      ],
      ["A has higher probability than B.", true],
      [
        "The probability of A is exactly \\(e^2\\), independent of normalization.",
        false,
      ],
    ],
    "Softmax preserves pairwise odds as exponentiated logit gaps, so A is \\(e^2\\) times as probable as B. Other classes enlarge the denominator and change absolute probabilities, but the shared denominator cancels from the A-to-B ratio.",
  ),
  makeQuestion(
    "crash-probability-l3-q67",
    "easy",
    "What probabilities result when softmax is applied to three equal logits?",
    [
      ["\\((1/3,1/3,1/3)\\)", true],
      ["\\((1,1,1)\\)", false],
      ["\\((0,0,0)\\)", false],
      ["The result depends on the common logit value.", false],
    ],
    "Equal logits have equal exponential weights, and normalization divides each by three identical weights. The common value cancels, so logits \\((0,0,0)\\) and \\((50,50,50)\\) produce the same uniform distribution.",
  ),
  makeQuestion(
    "crash-probability-l3-q68",
    "medium",
    "A three-class model changes logits from \\((2,1,0)\\) to \\((7,6,5)\\). Which comparisons are correct?",
    [
      ["The softmax distribution is unchanged.", true],
      ["Every pairwise logit difference is unchanged.", true],
      ["Each class probability increases by 5.", false],
      ["The new logits must overflow because one exceeds 1.", false],
    ],
    "The second vector adds the same constant five to every score, so all exponential weights gain a common factor that cancels in normalization. Probabilities are not obtained by adding constants to logits, and logits are not restricted to the probability interval.",
  ),
  makeQuestion(
    "crash-probability-l3-q69",
    "hard",
    "A binary classifier uses logits \\((a,b)\\). Which statements correctly relate its softmax output to the logit gap?",
    [
      ["\\(P(A)=1/(1+e^{-(a-b)})\\).", true],
      [
        "Only the difference \\(a-b\\), not their common offset, determines \\(P(A)\\).",
        true,
      ],
      ["If \\(a-b\\) grows, \\(P(A)\\) approaches one.", true],
      ["If \\(a=b\\), \\(P(A)=1\\) because A is listed first.", false],
    ],
    "Factoring \\(e^a\\) from the two-class softmax yields the logistic function of the score difference. Equal scores give equal half probabilities, while a growing positive gap makes A increasingly dominant without using class order.",
  ),
  makeQuestion(
    "crash-probability-l3-q70",
    "easy",
    "Which model outputs commonly use softmax to represent a categorical distribution?",
    [
      ["A single-label image classifier over classes", true],
      ["An LLM distribution over the next vocabulary token", true],
      ["A stochastic policy over mutually exclusive actions", true],
      ["An attention row that allocates weight across keys", true],
    ],
    "Each example needs positive normalized weights over a finite set of alternatives, which softmax supplies. Their interpretations differ—class, token, action, or attention allocation—but the normalization mechanism is shared.",
  ),
  makeQuestion(
    "crash-probability-l3-q71",
    "medium",
    "A model outputs softmax probabilities \\((0.50,0.30,0.20)\\). Which score changes preserve the predicted argmax class?",
    [
      ["Adding the same constant to every original logit", true],
      ["Increasing only the current largest logit", true],
      ["Swapping the largest and smallest logits", false],
      ["Replacing the distribution by its complement class by class", false],
    ],
    "A common shift preserves the whole distribution, and increasing the leading score preserves and strengthens its lead. Swapping scores changes the ordering, while componentwise complements do not form the same categorical decision rule and need not even normalize.",
  ),
  makeQuestion(
    "crash-probability-l3-q72",
    "hard",
    "Two models produce logits \\((4,2,0)\\) and \\((2,1,0)\\) for the same classes. Which statements are correct?",
    [
      ["Both models have the same argmax class.", true],
      ["The first model has larger pairwise logit gaps.", true],
      [
        "The first model's softmax distribution is more concentrated on the top class.",
        true,
      ],
      [
        "The distributions are identical because the logits have the same ordering.",
        false,
      ],
    ],
    "Multiplying all gaps by two preserves rank but sharpens the exponential ratios, so the first model assigns more mass to the leader. Argmax alone cannot reveal this difference in concentration or confidence.",
  ),

  // Likelihood and log-likelihood
  makeQuestion(
    "crash-probability-l3-q73",
    "easy",
    "A model assigns probability 0.8 to an observed class label. What is the likelihood contribution of that observation?",
    [
      ["0.8", true],
      ["0.2", false],
      ["1.0 because the label was observed", false],
      ["The probability assigned to the model parameters", false],
    ],
    "For fixed observed data, likelihood evaluates how much probability the model assigns to what occurred, so the contribution is 0.8. Observation does not retroactively make the model probability one, and likelihood is a function of parameters rather than a posterior over them.",
  ),
  makeQuestion(
    "crash-probability-l3-q74",
    "easy",
    "Two independent labeled examples receive correct-class probabilities 0.5 and 0.8. Which dataset likelihood calculations are correct?",
    [
      ["The likelihood is \\(0.5\\times0.8=0.4\\).", true],
      ["The log-likelihood is \\(\\log0.5+\\log0.8\\).", true],
      ["The likelihood is \\(0.5+0.8=1.3\\).", false],
      ["The log-likelihood is \\(\\log(0.5+0.8)\\).", false],
    ],
    "Under the stated independence factorization, joint likelihood multiplies per-example probabilities, and a logarithm converts that product to a sum. Adding probabilities describes a union-like operation, not simultaneous observation of both training examples.",
  ),
  makeQuestion(
    "crash-probability-l3-q75",
    "medium",
    "A Bernoulli model with parameter p observes outcomes 1, 1, 0. Which statements are correct?",
    [
      ["The likelihood is \\(L(p)=p^2(1-p)\\).", true],
      ["The log-likelihood is \\(2\\log p+\\log(1-p)\\).", true],
      ["The maximum-likelihood estimate is \\(p=2/3\\).", true],
      [
        "The likelihood is a probability distribution over p that must integrate to one.",
        false,
      ],
    ],
    "Each observed success contributes p and the failure contributes \\(1-p\\), producing the stated product and log sum. Maximizing it matches p to the empirical success rate, but likelihood as a function of p is not automatically a normalized parameter distribution.",
  ),
  makeQuestion(
    "crash-probability-l3-q76",
    "hard",
    "Model A assigns observed-token probabilities \\((0.8,0.5,0.25)\\); Model B assigns \\((0.6,0.6,0.4)\\). Which comparisons are correct?",
    [
      ["Model A likelihood is \\(0.8(0.5)(0.25)=0.10\\).", true],
      [
        "Model B likelihood is \\(0.6(0.6)(0.4)=0.144\\), so B fits this sequence better by likelihood.",
        true,
      ],
      [
        "A fits better because its largest individual probability 0.8 exceeds B's 0.6.",
        false,
      ],
      [
        "The two likelihoods should be compared by adding their per-token probabilities.",
        false,
      ],
    ],
    "Sequence likelihood rewards assigning probability to every observed step, so one strong token cannot compensate automatically for weaker factors elsewhere. Multiplication gives B the larger joint likelihood even though A has the largest single probability.",
  ),
  makeQuestion(
    "crash-probability-l3-q77",
    "medium",
    "Why is log-likelihood usually optimized instead of a long raw probability product?",
    [
      ["The logarithm turns products into sums.", true],
      [
        "Log is strictly increasing, so it preserves which positive likelihood is largest.",
        true,
      ],
      [
        "Summed log terms are easier to differentiate and aggregate across examples.",
        true,
      ],
      [
        "Working in log space reduces numerical underflow from multiplying many small probabilities.",
        true,
      ],
    ],
    "The logarithm changes representation without changing the maximizing parameters because it is monotonic. Its additive form supports stable computation, minibatch aggregation, and gradients while avoiding tiny products that can round to zero.",
  ),
  makeQuestion(
    "crash-probability-l3-q78",
    "hard",
    "A sequence has token probabilities 0.9, 0.1, and 0.9 under a model. Which statements are correct?",
    [
      ["Its likelihood is \\(0.081\\).", true],
      ["Its log-likelihood is \\(2\\log0.9+\\log0.1\\).", true],
      [
        "The low-probability middle token contributes the largest negative log penalty.",
        true,
      ],
      [
        "Replacing 0.1 by 0.2 would lower the likelihood because the sequence becomes less surprising.",
        false,
      ],
    ],
    "The chain factors multiply, and the logarithms add; the 0.1 token is the bottleneck and has the most negative log probability. Raising its assigned probability increases both likelihood and log-likelihood, making the observed sequence less surprising to the model.",
  ),
  makeQuestion(
    "crash-probability-l3-q79",
    "easy",
    "Which model is preferred by maximum likelihood on fixed observed data?",
    [
      [
        "The model assigning the larger probability to the complete observed dataset",
        true,
      ],
      [
        "The model with the largest number of classes regardless of its probabilities",
        false,
      ],
      ["The model whose probabilities are closest to zero", false],
      [
        "The model that assigns probability one to an unobserved alternative",
        false,
      ],
    ],
    "Maximum likelihood compares parameter settings by the probability they give the data that actually occurred. Model size and probability on unobserved alternatives matter only through how they affect that observed-data probability and any separately chosen regularization.",
  ),
  makeQuestion(
    "crash-probability-l3-q80",
    "medium",
    "A parameter value has likelihood 0.02 and another has likelihood 0.01 for the same data. Which statements are correct?",
    [
      ["The likelihood ratio is \\(0.02/0.01=2\\).", true],
      ["Its log-likelihood exceeds the second by \\(\\log2\\).", true],
      [
        "The first parameter has posterior probability \\(2/3\\) without any prior or normalization.",
        false,
      ],
      [
        "A likelihood ratio of 2 proves the first model will generalize better.",
        false,
      ],
    ],
    "The likelihood ratio is two, and taking logs turns that ratio into a difference of \\(\\log2\\). Likelihood alone is neither a normalized posterior over parameters nor a guarantee about unseen-data performance.",
  ),
  makeQuestion(
    "crash-probability-l3-q81",
    "hard",
    "A dataset duplicates every observation once and otherwise keeps the same model. Which statements describe the effect on likelihood?",
    [
      [
        "The new likelihood is the square of the original likelihood under the same factorization.",
        true,
      ],
      ["The new log-likelihood is twice the original log-likelihood.", true],
      ["The average log-likelihood per observation is unchanged.", true],
      [
        "The raw likelihood is unchanged because no new unique outcome was added.",
        false,
      ],
    ],
    "Duplicating all factors repeats the same product, squaring likelihood and doubling its logarithm. Dividing by the doubled observation count restores the same average, which is why average loss permits comparisons across dataset sizes.",
  ),
  makeQuestion(
    "crash-probability-l3-q82",
    "easy",
    "Which statements correctly distinguish probability and likelihood?",
    [
      [
        "Probability views \\(P_\\theta(x)\\) across data outcomes x for fixed \\(\\theta\\).",
        true,
      ],
      [
        "Likelihood views \\(L(\\theta;x)=P_\\theta(x)\\) across parameter values for fixed x.",
        true,
      ],
      [
        "The same expression \\(P_\\theta(x)\\) can be viewed in either direction depending on what varies.",
        true,
      ],
      [
        "A likelihood function need not sum or integrate to one over the parameter.",
        true,
      ],
    ],
    "The numerical model expression can support two viewpoints: a distribution over data for fixed parameters or a score over parameters for fixed data. Normalization is required in the data direction, not automatically in the likelihood-as-function-of-parameters direction.",
  ),
  makeQuestion(
    "crash-probability-l3-q83",
    "medium",
    "Two independent batches have likelihoods \\(L_1\\) and \\(L_2\\) under the same model. Which statements are correct for their combined data?",
    [
      ["The combined likelihood is \\(L_1L_2\\).", true],
      ["The combined log-likelihood is \\(\\log L_1+\\log L_2\\).", true],
      ["The combined likelihood is \\(L_1+L_2\\).", false],
      [
        "The lower-likelihood batch can be discarded without changing the objective.",
        false,
      ],
    ],
    "Independence across the two batches makes their joint data probability a product, which becomes an additive log objective. Both batches contribute evidence; dropping one changes the data and the optimization target.",
  ),
  makeQuestion(
    "crash-probability-l3-q84",
    "hard",
    "A next-token model gives a correct five-token continuation conditional probabilities \\((0.5,0.4,0.8,0.25,0.5)\\). Which statements are correct?",
    [
      [
        "The continuation likelihood is \\(0.5(0.4)(0.8)(0.25)(0.5)=0.02\\).",
        true,
      ],
      [
        "Its total log-likelihood is the sum of the five log probabilities.",
        true,
      ],
      [
        "Improving the 0.25 factor to 0.50 doubles the sequence likelihood if other factors stay fixed.",
        true,
      ],
      [
        "The sequence likelihood is the arithmetic mean \\((0.5+0.4+0.8+0.25+0.5)/5=0.49\\).",
        false,
      ],
    ],
    "Autoregressive sequence likelihood multiplies the conditional probability at every observed position, giving 0.02. A single factor's ratio scales the whole product, while averaging probabilities does not represent the probability of all tokens occurring in sequence.",
  ),

  // Negative log-likelihood and cross-entropy
  makeQuestion(
    "crash-probability-l3-q85",
    "easy",
    "A model assigns probability 0.5 to the correct class. What is its natural-log negative log-likelihood for this example?",
    [
      ["\\(-\\log0.5=\\log2\\approx0.693\\)", true],
      ["\\(1-0.5=0.5\\)", false],
      ["\\(\\log0.5\\approx-0.693\\)", false],
      ["\\(-\\log1=0\\)", false],
    ],
    "Negative log-likelihood negates the log probability assigned to the observed class, yielding about 0.693. Raw error, unnegated log probability, and the loss for probability one are different quantities.",
  ),
  makeQuestion(
    "crash-probability-l3-q86",
    "easy",
    "Which changes lower the negative log-likelihood of a fixed correct class?",
    [
      ["Increasing its predicted probability from 0.4 to 0.7", true],
      ["Increasing its logit relative to competing logits", true],
      ["Moving its probability from 0.4 to 0.1", false],
      [
        "Assigning more probability to an incorrect class while holding the correct probability fixed and normalization unchanged",
        false,
      ],
    ],
    "The function \\(-\\log p\\) decreases as the correct-class probability rises, and a larger relative logit tends to produce that rise. Lowering correct probability increases loss, while categorical normalization prevents freely adding mass to competitors without changing something else.",
  ),
  makeQuestion(
    "crash-probability-l3-q87",
    "medium",
    "For one-hot target \\(y=(0,1,0)\\) and prediction \\(p=(0.2,0.5,0.3)\\), which statements are correct?",
    [
      ["Cross-entropy is \\(-\\sum_i y_i\\log p_i=-\\log0.5\\).", true],
      [
        "Only the target class contributes directly because the other target entries are zero.",
        true,
      ],
      [
        "This cross-entropy equals the example's negative log-likelihood.",
        true,
      ],
      ["Cross-entropy is \\(-\\log0.2-\\log0.5-\\log0.3\\).", false],
    ],
    "A one-hot target selects the log probability of the observed class, making categorical cross-entropy identical to negative log-likelihood. Summing every negative log would pretend all three mutually exclusive classes were simultaneously observed.",
  ),
  makeQuestion(
    "crash-probability-l3-q88",
    "hard",
    "A soft target is \\(q=(0.7,0.3)\\). Which statements correctly compare predictions \\(p_A=(0.7,0.3)\\) and \\(p_B=(0.9,0.1)\\)?",
    [
      ["Cross-entropy for A is \\(-0.7\\log0.7-0.3\\log0.3\\).", true],
      [
        "A has lower cross-entropy than B because it matches the full target distribution.",
        true,
      ],
      [
        "B has \\(H(q,p_B)=0\\) because its argmax matches the target's largest class.",
        false,
      ],
      [
        "Only \\(-0.7\\log p_1\\) matters because soft targets behave like one-hot labels.",
        false,
      ],
    ],
    "With soft targets, every positive target component weights a log predicted probability, so matching only the top class is insufficient. Prediction A reproduces q, while B places too little mass on the target's second component and incurs extra cross-entropy.",
  ),
  makeQuestion(
    "crash-probability-l3-q89",
    "medium",
    "Which statements about negative log-likelihood (NLL) are correct?",
    [
      [
        "Minimizing NLL is equivalent to maximizing likelihood on the same data.",
        true,
      ],
      ["A correct-class probability near one gives loss near zero.", true],
      [
        "A correct-class probability approaching zero gives an increasingly large loss.",
        true,
      ],
      [
        "Dataset NLL adds the per-observation negative log probabilities under the usual factorization.",
        true,
      ],
    ],
    "Negating log-likelihood turns a maximization into the minimization convention used by optimizers. The logarithm strongly penalizes assigning tiny probability to an observed outcome, and independent data factors become additive per-example losses.",
  ),
  makeQuestion(
    "crash-probability-l3-q90",
    "hard",
    "Two examples have correct-class probabilities 0.8 and 0.2. Which statements correctly describe their mean NLL?",
    [
      ["It is \\(-[\\log0.8+\\log0.2]/2\\).", true],
      ["It equals \\(-\\log\\sqrt{0.8(0.2)}\\).", true],
      ["The 0.2 example contributes the larger loss.", true],
      ["It equals \\(-\\log[(0.8+0.2)/2]=-\\log0.5\\).", false],
    ],
    "Averaging log losses is equivalent to taking the negative log of the geometric mean probability, not the arithmetic mean. The smaller probability produces a much larger penalty, reflecting the model's greater surprise at that observed label.",
  ),
  makeQuestion(
    "crash-probability-l3-q91",
    "easy",
    "A model assigns probability 1 to the observed correct class. What is the categorical negative log-likelihood?",
    [
      ["0", true],
      ["1", false],
      ["\\(-1\\)", false],
      ["Undefined because logarithms cannot use probabilities", false],
    ],
    "Because \\(\\log1=0\\), a perfectly predicted observed class incurs zero negative log-likelihood. Logarithms are defined for positive probabilities; the problematic boundary is probability zero, whose negative log diverges.",
  ),
  makeQuestion(
    "crash-probability-l3-q92",
    "medium",
    "A batch has per-example NLL values 0.2, 0.8, 0.4, and 0.6. Which statements are correct?",
    [
      ["The mean batch loss is \\((0.2+0.8+0.4+0.6)/4=0.5\\).", true],
      [
        "The summed NLL is 2.0 and corresponds to the negative log of the batch likelihood under factorization.",
        true,
      ],
      [
        "The mean loss is \\(\\bar L=2.0\\) because likelihood multiplies.",
        false,
      ],
      ["The lowest individual loss 0.2 is the correct batch objective.", false],
    ],
    "Summed NLL aggregates the joint log objective, while dividing by batch size gives the common mean used for scale-stable optimization. Selecting a minimum ignores the other observations, and multiplication occurs before taking logs rather than among NLL values.",
  ),
  makeQuestion(
    "crash-probability-l3-q93",
    "hard",
    "A language model's average token NLL is \\(\\log4\\). Which statements are correct?",
    [
      ["Its perplexity is \\(e^{\\log4}=4\\).", true],
      [
        "The geometric mean probability assigned to observed tokens is \\(e^{-\\log4}=1/4\\).",
        true,
      ],
      ["Lowering average NLL lowers perplexity.", true],
      [
        "Perplexity 4 means exactly four vocabulary tokens have nonzero probability at every step.",
        false,
      ],
    ],
    "Perplexity exponentiates average negative log-likelihood, so it is inversely related to the geometric mean observed-token probability. Its effective-choice interpretation is not a claim that the support literally contains four tokens at every context.",
  ),
  makeQuestion(
    "crash-probability-l3-q94",
    "easy",
    "Which statements connect cross-entropy training to classification and language modeling?",
    [
      [
        "The target label or next token identifies which observed outcome receives credit.",
        true,
      ],
      [
        "The model is penalized for assigning little probability to that observed outcome.",
        true,
      ],
      [
        "Loss is averaged or summed across training examples or token positions.",
        true,
      ],
      [
        "Optimization changes model parameters so future predicted distributions better fit data.",
        true,
      ],
    ],
    "Both tasks train normalized output distributions against observed outcomes, typically using one-hot targets and per-position negative log probabilities. Aggregating those losses produces an objective whose gradients adjust the logits and the network that generated them.",
  ),
  makeQuestion(
    "crash-probability-l3-q95",
    "medium",
    "Model A gives the correct class probability 0.6 and Model B gives 0.3 on the same example. Which comparisons are correct?",
    [
      ["A has lower NLL because \\(-\\log0.6<-\\log0.3\\).", true],
      [
        "A has likelihood ratio \\(0.6/0.3=2\\) relative to B on this example.",
        true,
      ],
      ["B has lower loss because \\(-\\log0.3<-\\log0.6\\).", false],
      ["Their losses are equal if their argmax labels match.", false],
    ],
    "NLL evaluates probability assigned to the observed outcome, so A both doubles the likelihood contribution and receives the smaller loss. Matching argmax decisions can hide substantial probability and loss differences.",
  ),
  makeQuestion(
    "crash-probability-l3-q96",
    "hard",
    "A target distribution is \\(q=(0.5,0.5,0)\\). Which statements about cross-entropy \\(H(q,p)\\) are correct?",
    [
      ["It is \\(-0.5\\log p_1-0.5\\log p_2\\).", true],
      [
        "Putting zero probability on either of the first two classes makes the cross-entropy diverge.",
        true,
      ],
      [
        "Among normalized p, matching \\(p=(0.5,0.5,0)\\) minimizes the cross-entropy.",
        true,
      ],
      [
        "Class 3 contributes \\(-\\log p_3\\) despite its target weight being zero.",
        false,
      ],
    ],
    "Cross-entropy weights each predicted log probability by the target mass, so the zero-target class has no direct term. Both positive-target classes must receive probability, and the optimum reproduces the target distribution rather than collapsing onto only one of them.",
  ),

  // Entropy, temperature, and calibration
  makeQuestion(
    "crash-probability-l3-q97",
    "easy",
    "Which of these three-class distributions has the highest entropy?",
    [
      ["\\((1/3,1/3,1/3)\\)", true],
      ["\\((0.8,0.1,0.1)\\)", false],
      ["\\((1,0,0)\\)", false],
      ["\\((0.98,0.01,0.01)\\)", false],
    ],
    "Entropy is largest when probability mass is spread uniformly across the available outcomes. Concentrated distributions are more predictable and have lower entropy, with a point mass such as \\((1,0,0)\\) attaining zero.",
  ),
  makeQuestion(
    "crash-probability-l3-q98",
    "easy",
    "Which comparisons between \\((0.5,0.5)\\) and \\((0.9,0.1)\\) are correct?",
    [
      ["The first distribution has higher entropy.", true],
      ["The second distribution is more concentrated on one outcome.", true],
      ["The second has higher entropy because 0.9 is a larger number.", false],
      [
        "Both have zero entropy because they each contain two probabilities.",
        false,
      ],
    ],
    "Entropy measures uncertainty across the whole distribution, not the magnitude of its largest entry in isolation. The balanced distribution is less predictable, while the 0.9/0.1 distribution concentrates mass and therefore has lower entropy.",
  ),
  makeQuestion(
    "crash-probability-l3-q99",
    "medium",
    "For a fair binary distribution, which entropy statements are correct when natural logarithms are used?",
    [
      ["\\(H=-2(0.5\\log0.5)=\\log2\\).", true],
      ["The entropy is larger than that of \\((1,0)\\).", true],
      ["Swapping the two probabilities leaves entropy unchanged.", true],
      [
        "Entropy is \\(0.5\\) because only the largest probability is counted.",
        false,
      ],
    ],
    "Entropy sums \\(-p_i\\log p_i\\) over every outcome, giving \\(\\log2\\) for a fair binary choice and zero for a deterministic one. The formula is symmetric in outcome labels, so reordering probabilities cannot change uncertainty.",
  ),
  makeQuestion(
    "crash-probability-l3-q100",
    "hard",
    "A model's logits are divided by temperature T before softmax. Which statements correctly compare \\(T=0.5\\) with \\(T=2\\)?",
    [
      [
        "\\(T=0.5\\) enlarges logit gaps and produces a sharper distribution.",
        true,
      ],
      ["\\(T=2\\) shrinks logit gaps and generally raises entropy.", true],
      [
        "Higher temperature changes which logit is largest whenever logits are distinct.",
        false,
      ],
      [
        "Temperature retrains the model parameters before each token is sampled.",
        false,
      ],
    ],
    "Dividing by a small positive T magnifies score differences, while a large T flattens them without changing their ordering. Temperature is a decoding transformation of a fixed logit vector, not a new training pass or a source of new knowledge.",
  ),
  makeQuestion(
    "crash-probability-l3-q101",
    "medium",
    "Which statements correctly distinguish entropy, cross-entropy, and negative log-likelihood?",
    [
      ["Entropy summarizes uncertainty within one distribution.", true],
      [
        "Cross-entropy scores predicted probabilities using a target distribution.",
        true,
      ],
      [
        "With one-hot targets, per-example cross-entropy equals negative log-likelihood.",
        true,
      ],
      [
        "All three use logarithms and probability weights but answer different questions.",
        true,
      ],
    ],
    "Entropy depends only on a distribution's own mass, whereas cross-entropy compares target weights with model probabilities. A one-hot target selects the observed class, recovering NLL, but that special equality does not erase the conceptual distinction among the quantities.",
  ),
  makeQuestion(
    "crash-probability-l3-q102",
    "hard",
    "A binary classifier changes predictions from \\((0.6,0.4)\\) to \\((0.9,0.1)\\) while the first class is correct. Which statements are correct?",
    [
      [
        "The correct-class NLL decreases from \\(-\\log0.6\\) to \\(-\\log0.9\\).",
        true,
      ],
      [
        "The entropy satisfies \\(H(0.9,0.1)<H(0.6,0.4)\\) because mass becomes more concentrated.",
        true,
      ],
      [
        "If the first class had been wrong, the sharper prediction would incur a larger NLL.",
        true,
      ],
      ["Lower entropy guarantees better calibration across a dataset.", false],
    ],
    "Sharpening toward the correct outcome improves this example's likelihood and lowers entropy, but sharpening toward a wrong outcome is punished strongly. Calibration is an empirical frequency property across predictions, so confidence alone cannot guarantee it.",
  ),
  makeQuestion(
    "crash-probability-l3-q103",
    "easy",
    "A model makes many predictions at confidence 0.70, and about 70% are correct. What property does this illustrate at that confidence level?",
    [
      ["Calibration", true],
      ["Maximum entropy", false],
      ["Zero cross-entropy", false],
      ["Class independence", false],
    ],
    "Calibration aligns stated probabilities with observed frequencies in comparable forecast groups. It does not require a uniform distribution, zero training loss, or independence among class events. The comparison must use proportions from repeated forecasts rather than one outcome.",
  ),
  makeQuestion(
    "crash-probability-l3-q104",
    "medium",
    "In a confidence bin, a model reports 0.80 on 500 predictions and gets 350 correct. Which statements are correct?",
    [
      ["Observed accuracy is \\(350/500=0.70\\).", true],
      ["The model is overconfident by 0.10 in this bin.", true],
      [
        "The model is underconfident by \\(0.80-0.70=0.10\\) because 350 is greater than 80.",
        false,
      ],
      [
        "This bin alone determines the model's entropy on every prediction.",
        false,
      ],
    ],
    "The relevant comparison is between two proportions, reported confidence 0.80 and observed frequency 0.70, yielding overconfidence. The raw count is not comparable with a percentage, and calibration-bin data do not reconstruct each prediction's entropy.",
  ),
  makeQuestion(
    "crash-probability-l3-q105",
    "hard",
    "Temperature scaling divides every logit by one learned positive scalar on validation data. Which statements are correct?",
    [
      [
        "It preserves the class ranking and therefore preserves argmax predictions.",
        true,
      ],
      [
        "It can adjust confidence concentration without changing model features.",
        true,
      ],
      [
        "A temperature above one can soften an overconfident distribution.",
        true,
      ],
      [
        "It guarantees calibrated probabilities under every future distribution shift.",
        false,
      ],
    ],
    "A shared positive scale leaves logit order intact but changes score gaps and probability sharpness, which can improve validation calibration. That fitted relationship may fail after data shift, so scaling is not a universal guarantee.",
  ),
  makeQuestion(
    "crash-probability-l3-q106",
    "easy",
    "Which statements about entropy are correct?",
    [
      ["A deterministic categorical distribution has entropy zero.", true],
      [
        "A uniform distribution over a fixed finite support has maximum entropy.",
        true,
      ],
      ["Entropy depends on the whole probability distribution.", true],
      [
        "Changing decoding temperature can change entropy without changing model weights.",
        true,
      ],
    ],
    "Entropy measures distributional uncertainty, ranging from zero for a point mass to its maximum at uniformity on fixed support. Because temperature reshapes probabilities produced from the same logits, it can change entropy at inference time without retraining.",
  ),
  makeQuestion(
    "crash-probability-l3-q107",
    "medium",
    "Two predictions have equal argmax class: A is \\((0.51,0.49)\\), B is \\((0.99,0.01)\\). Which statements are correct?",
    [
      ["A has higher entropy than B.", true],
      ["B assigns lower NLL if the first class is observed.", true],
      [
        "They communicate identical uncertainty because the chosen class matches.",
        false,
      ],
      [
        "B must be better calibrated on a dataset because it is more confident.",
        false,
      ],
    ],
    "A is nearly balanced and therefore more uncertain, while B strongly favors the observed first class and gives it smaller example loss. Whether that confidence matches long-run frequencies requires calibration evidence beyond these two distributions.",
  ),
  makeQuestion(
    "crash-probability-l3-q108",
    "hard",
    "A four-class distribution changes from uniform \\((0.25,0.25,0.25,0.25)\\) toward one class while remaining normalized. Which statements are correct?",
    [
      [
        "Its entropy decreases as the distribution becomes more concentrated.",
        true,
      ],
      [
        "The largest probability increases while at least some other mass decreases.",
        true,
      ],
      [
        "If the favored class is the target, its one-hot cross-entropy decreases.",
        true,
      ],
      [
        "Normalization prevents any change in uncertainty because the total stays one.",
        false,
      ],
    ],
    "Normalization fixes total mass but not how that mass is distributed, so concentration lowers entropy. When concentration moves toward the observed target it also increases target probability and lowers NLL, though concentrating on a wrong class would do the opposite for loss.",
  ),

  // Optimization and deep-learning synthesis
  makeQuestion(
    "crash-probability-l3-q109",
    "easy",
    "For softmax cross-entropy with one-hot target y and probabilities p, which expression is the gradient with respect to logits?",
    [
      ["\\(p-y\\)", true],
      ["\\(p+y\\)", false],
      ["\\(y/p\\) for every class", false],
      ["The scalar entropy H(p) copied to every logit", false],
    ],
    "The softmax and cross-entropy derivatives combine to the simple classwise signal \\(p_i-y_i\\). It lowers the correct-class loss by pushing its logit upward and pushes competing logits according to the probability mass they currently receive.",
  ),
  makeQuestion(
    "crash-probability-l3-q110",
    "easy",
    "A three-class prediction is \\(p=(0.2,0.5,0.3)\\) and the target is the second class. Which gradient components are correct?",
    [
      ["The correct-class component is \\(0.5-1=-0.5\\).", true],
      ["The first-class component is \\(0.2-0=0.2\\).", true],
      [
        "Every component is negative because the total loss is positive.",
        false,
      ],
      ["The components sum to one because p sums to one.", false],
    ],
    "The gradient \\(p-y\\) is negative for the underweighted correct class and positive for incorrect classes with probability mass. Its components sum to zero because both p and y sum to one, reflecting softmax's invariance to a common logit shift.",
  ),
  makeQuestion(
    "crash-probability-l3-q111",
    "medium",
    "How does gradient descent respond to the gradient \\(p-y\\) for a one-hot target? Which statements are correct?",
    [
      [
        "It tends to increase the correct-class logit because that gradient component is negative.",
        true,
      ],
      [
        "It tends to decrease incorrect-class logits with positive probability mass.",
        true,
      ],
      [
        "Larger incorrect probabilities receive larger positive gradient components.",
        true,
      ],
      [
        "It increases every logit equally, which leaves softmax unchanged.",
        false,
      ],
    ],
    "Gradient descent subtracts the gradient, so a negative correct-class component raises that logit and positive competing components lower theirs. The adjustment is probability-sensitive, focusing more pressure on incorrect classes the model currently favors.",
  ),
  makeQuestion(
    "crash-probability-l3-q112",
    "hard",
    "A correct class currently has probability 0.01 in a 100-class model. Which training-signal statements are correct?",
    [
      [
        "Its NLL is \\(-\\log0.01\\approx4.605\\), indicating a large error.",
        true,
      ],
      [
        "Its logit gradient component is \\(0.01-1=-0.99\\), strongly pushing it upward under gradient descent.",
        true,
      ],
      [
        "Its low probability yields almost zero loss, so the example contributes little learning signal.",
        false,
      ],
      [
        "The gradient requires sampling a class rather than using the predicted probability vector.",
        false,
      ],
    ],
    "Assigning only one percent to an observed class is strongly penalized by the logarithm and produces a near-minus-one correct-logit gradient. Standard cross-entropy computes this signal directly from probabilities and the target without sampling the output class.",
  ),
  makeQuestion(
    "crash-probability-l3-q113",
    "medium",
    "Which statements correctly describe empirical risk minimization with cross-entropy?",
    [
      [
        "Training averages or sums losses over observed examples or token positions.",
        true,
      ],
      ["Minibatches provide noisy estimates of the full-data gradient.", true],
      [
        "Parameter updates aim to increase probability assigned to observed targets in their contexts.",
        true,
      ],
      [
        "Calibration and generalization still require evaluation beyond the training loss.",
        true,
      ],
    ],
    "Cross-entropy training uses sample averages to approximate expected loss, and stochastic minibatches make the update direction noisy but useful. Lower training loss improves fit to observed data; by itself it does not prove future calibration or generalization, so those require separate evaluation.",
  ),
  makeQuestion(
    "crash-probability-l3-q114",
    "hard",
    "A dataset contains 90% class A and 10% class B. Which statements about unweighted cross-entropy training are correct?",
    [
      [
        "Class A examples contribute more terms simply because they are more numerous.",
        true,
      ],
      [
        "A model predicting the class prior for every input can achieve 90% accuracy but still ignore useful features.",
        true,
      ],
      [
        "Class weighting can change the effective contribution of minority examples.",
        true,
      ],
      [
        "Cross-entropy automatically gives each class equal aggregate weight regardless of frequency.",
        false,
      ],
    ],
    "An ordinary example average reflects the empirical class frequencies, so majority examples dominate the aggregate count. Weighting can alter the target tradeoff, and accuracy alone can hide a model that merely repeats the base rate rather than learning conditional structure.",
  ),
  makeQuestion(
    "crash-probability-l3-q115",
    "easy",
    "A neural network assigns more probability to the observed correct label after an update. What happens to that example's NLL?",
    [
      ["It decreases.", true],
      ["It increases because confidence is penalized.", false],
      ["It stays fixed because labels are one-hot.", false],
      ["It becomes the entropy of the dataset.", false],
    ],
    "Negative log-likelihood is a decreasing function of the observed-label probability, so raising that probability lowers the example loss. One-hot targets select the relevant probability; they do not freeze loss or turn it into a dataset-level entropy.",
  ),
  makeQuestion(
    "crash-probability-l3-q116",
    "medium",
    "Which statements correctly separate training, decoding, and evaluation?",
    [
      ["Cross-entropy trains parameters using target outcomes.", true],
      [
        "Temperature can reshape a fixed model's output distribution during decoding.",
        true,
      ],
      [
        "Greedy decoding changes the likelihood objective used during completed training.",
        false,
      ],
      [
        "Low validation NLL automatically selects the best action for every downstream cost structure.",
        false,
      ],
    ],
    "Training adjusts model weights to fit observed data, while decoding chooses how to turn a resulting distribution into outputs. Evaluation loss describes probabilistic fit but does not retroactively change the objective or encode every application's decision costs.",
  ),
  makeQuestion(
    "crash-probability-l3-q117",
    "hard",
    "Two models have identical accuracy, but Model A assigns 0.51 to every correct prediction while Model B assigns 0.90. Which statements are correct on those examples?",
    [
      [
        "Model B has lower NLL because it assigns more probability to observed labels.",
        true,
      ],
      [
        "Accuracy cannot distinguish their confidence on the correct examples.",
        true,
      ],
      [
        "Calibration still requires comparing each confidence level with empirical correctness frequencies.",
        true,
      ],
      [
        "Model B must be globally better because confidence 0.90 is always desirable.",
        false,
      ],
    ],
    "Proper probabilistic loss distinguishes predictions that accuracy treats identically, rewarding more probability on realized outcomes. However, confidence can be harmful on mistakes or miscalibrated groups, so overall model quality requires evaluating all examples and frequency alignment.",
  ),
  makeQuestion(
    "crash-probability-l3-q118",
    "easy",
    "Which steps form the standard probabilistic classification training pipeline?",
    [
      ["The network produces logits.", true],
      ["Softmax converts logits to class probabilities.", true],
      ["Cross-entropy compares those probabilities with the target.", true],
      [
        "Backpropagation carries the loss gradient into model parameters.",
        true,
      ],
    ],
    "These steps connect raw neural scores to a normalized predictive distribution and then to an optimization signal. Backpropagation applies the resulting derivatives through the network so future logits can assign more appropriate probabilities.",
  ),
  makeQuestion(
    "crash-probability-l3-q119",
    "medium",
    "A model improves mean NLL from 1.2 to 0.9 on held-out data. Which statements are justified?",
    [
      [
        "Its geometric-mean observed-outcome probability rises from \\(e^{-1.2}\\) to \\(e^{-0.9}\\).",
        true,
      ],
      ["Its held-out perplexity \\(e^{\\text{NLL}}\\) decreases.", true],
      ["Every individual held-out prediction improved.", false],
      ["The model is necessarily calibrated in every confidence bin.", false],
    ],
    "Lower average NLL means the product-equivalent or geometric-mean probability of observed outcomes increased, and exponentiation therefore lowers perplexity. An aggregate improvement can include worse individual cases and does not by itself establish calibration.",
  ),
  makeQuestion(
    "crash-probability-l3-q120",
    "hard",
    "A three-class model predicts \\(p=(0.7,0.2,0.1)\\) and the observed class is the second. Which statements are correct?",
    [
      ["The example NLL is \\(-\\log0.2\\).", true],
      ["The logit gradient is \\((0.7,-0.8,0.1)\\).", true],
      [
        "Gradient descent tends to lower the first and third logits and raise the second.",
        true,
      ],
      [
        "The loss uses \\(-\\log0.7\\) because the first class is the model's argmax.",
        false,
      ],
    ],
    "Training scores the probability of the observed target, not the model's chosen argmax, so the relevant mass is 0.2. Subtracting the one-hot target gives the displayed gradient, which corrects the model's misplaced confidence toward class two.",
  ),
];
