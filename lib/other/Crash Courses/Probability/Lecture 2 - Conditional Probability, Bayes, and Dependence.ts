import { Question } from "../../../quiz";

export const CrashCourseProbabilityL2Questions: Question[] = [
  {
    id: "crash-probability-l2-q01",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statement best interprets \\(P(A \\mid B)\\) in ordinary probability language?",
    options: [
      {
        text: "The probability of \\(A\\) after restricting attention to cases where \\(B\\) is true.",
        isCorrect: true,
      },
      {
        text: "The probability that \\(A\\) and \\(B\\) are unrelated variables.",
        isCorrect: false,
      },
      {
        text: "The probability of \\(B\\) after \\(A\\) has been observed.",
        isCorrect: false,
      },
      {
        text: "A quantity that can be computed as \\(P(A,B)/P(B)\\) when \\(P(B)>0\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "\\(P(A\\mid B)\\) means the probability of \\(A\\) given that \\(B\\) is true, and it can be computed from the joint probability as \\(P(A,B)/P(B)\\) when \\(P(B)>0\\). It is different from \\(P(B\\mid A)\\), different from independence, and different from the unconditional probability \\(P(A)\\).",
  },
  {
    id: "crash-probability-l2-q02",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which examples correctly translate an AI prediction problem into conditional probability notation?",
    options: [
      {
        text: "\\(P(\\text{spam}\\mid\\text{email text})\\) for spam classification.",
        isCorrect: true,
      },
      {
        text: "\\(P(X_t\\mid X_1,\\ldots,X_{t-1})\\) for next-token prediction.",
        isCorrect: true,
      },
      {
        text: "\\(P(S_{t+1}\\mid S_t,A_t)\\) for a reinforcement learning transition.",
        isCorrect: true,
      },
      {
        text: "\\(P(\\epsilon\\mid x_t,t)\\) for predicting noise from a noisy diffusion state.",
        isCorrect: true,
      },
    ],
    explanation:
      "All four examples have the same structure: a target is predicted after conditioning on input information. Conditional probability is central because AI systems usually estimate how likely outputs are given inputs, contexts, states, or noisy observations.",
  },
  {
    id: "crash-probability-l2-q03",
    chapter: 2,
    difficulty: "easy",
    prompt:
      'Before reading an email, a model has \\(P(\\text{spam})=0.20\\). After seeing the phrase "you have won a prize," it estimates \\(P(\\text{spam}\\mid\\text{phrase})=0.85\\). Which statements are correct?',
    options: [
      {
        text: "\\(P(\\text{spam})=0.20\\) is unconditional in this comparison.",
        isCorrect: true,
      },
      {
        text: "\\(P(\\text{spam}\\mid\\text{phrase})=0.85\\) is conditional on observed text.",
        isCorrect: true,
      },
      {
        text: "The phrase changes the model's probability estimate for spam.",
        isCorrect: true,
      },
      {
        text: "The two probabilities must be equal because they concern the same spam event.",
        isCorrect: false,
      },
    ],
    explanation:
      "The unconditional probability describes spam before using the phrase, while the conditional probability uses the phrase as information. Conditional probability is useful because evidence can change which outcomes are plausible.",
  },
  {
    id: "crash-probability-l2-q04",
    chapter: 2,
    difficulty: "medium",
    type: "assertion-reason",
    prompt:
      "Assertion: \\(P(y\\mid x)\\) is a natural mathematical form for prediction in supervised learning.\n\nReason: The expression asks for a probability distribution over an output after an input is known.",
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
      "The assertion is true because supervised prediction uses input information to estimate likely outputs. The reason is also true and explains the assertion: \\(P(y\\mid x)\\) explicitly conditions the output distribution on the input.",
  },
  {
    id: "crash-probability-l2-q05",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly distinguish joint, marginal, and conditional probabilities?",
    options: [
      {
        text: "\\(P(A,B)\\) is the probability that \\(A\\) and \\(B\\) both happen.",
        isCorrect: true,
      },
      {
        text: "\\(P(A)\\) can be a marginal probability that ignores the value of another variable.",
        isCorrect: true,
      },
      {
        text: "\\(P(A\\mid B)\\) asks about \\(A\\) after \\(B\\) is known.",
        isCorrect: true,
      },
      {
        text: "A joint probability is the same object as an expected numerical value.",
        isCorrect: false,
      },
    ],
    explanation:
      "Joint probability describes variables occurring together, marginal probability focuses on one variable, and conditional probability uses information about another event or variable. Expected value is a weighted average of numerical outcomes, so it is a different concept.",
  },
  {
    id: "crash-probability-l2-q06",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "In a 100-patient table, 30 patients have both fever and infection, and 40 patients have fever. What is \\(P(\\text{infection}\\mid\\text{fever})\\)?",
    options: [
      { text: "\\(0.30\\)", isCorrect: false },
      { text: "\\(0.40\\)", isCorrect: false },
      { text: "\\(0.75\\)", isCorrect: true },
      { text: "\\(1.33\\)", isCorrect: false },
    ],
    explanation:
      "Conditional probability restricts the denominator to the fever cases. The calculation is \\(P(\\text{infection},\\text{fever})/P(\\text{fever})=0.30/0.40=0.75\\), not 30 divided by all 100 patients.",
  },
  {
    id: "crash-probability-l2-q07",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Using the same 100-patient table with 30 fever-and-infection cases, 10 fever-and-no-infection cases, 5 no-fever-and-infection cases, and 55 no-fever-and-no-infection cases, which statements are correct?",
    options: [
      {
        text: "\\(P(\\text{fever})=0.40\\).",
        isCorrect: true,
      },
      {
        text: "\\(P(\\text{infection})=0.35\\).",
        isCorrect: true,
      },
      {
        text: "\\(P(\\text{fever},\\text{infection})=0.35\\).",
        isCorrect: false,
      },
      {
        text: "\\(P(\\text{infection}\\mid\\text{fever})=P(\\text{infection})\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The fever marginal is \\((30+10)/100=0.40\\), and the infection marginal is \\((30+5)/100=0.35\\). The joint fever-and-infection probability is \\(30/100=0.30\\), not 0.35, and \\(P(\\text{infection}\\mid\\text{fever})=0.75\\), so fever and infection are not independent in this table.",
  },
  {
    id: "crash-probability-l2-q08",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which formulas correctly connect joint and conditional probability when \\(P(B)>0\\)?",
    options: [
      {
        text: "\\(P(A\\mid B)=\\frac{P(A,B)}{P(B)}\\).",
        isCorrect: true,
      },
      {
        text: "\\(P(A,B)=P(A\\mid B)P(B)\\).",
        isCorrect: true,
      },
      {
        text: "\\(P(A,B)=P(B\\mid A)P(B)\\) when \\(P(A)>0\\).",
        isCorrect: false,
      },
      {
        text: "\\(P(A\\mid B)=P(B\\mid A)\\) for all events with positive probability.",
        isCorrect: false,
      },
    ],
    explanation:
      "Conditional probability can be rearranged to express a joint probability as \\(P(A\\mid B)P(B)\\) or as \\(P(B\\mid A)P(A)\\). Multiplying \\(P(B\\mid A)\\) by \\(P(B)\\) mixes the wrong marginal with that conditional, and the reversed conditionals are not generally equal.",
  },
  {
    id: "crash-probability-l2-q09",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "In a click table, \\(P(\\text{click},\\text{likes sports})=0.20\\) and \\(P(\\text{click},\\text{does not like sports})=0.05\\). What is \\(P(\\text{click})\\)?",
    options: [
      { text: "\\(0.05\\)", isCorrect: false },
      { text: "\\(0.20\\)", isCorrect: false },
      { text: "\\(0.25\\)", isCorrect: true },
      { text: "\\(0.75\\)", isCorrect: false },
    ],
    explanation:
      "Marginalization adds all mutually exclusive ways the click event can occur. A user either likes sports or does not, so \\(P(\\text{click})=0.20+0.05=0.25\\).",
  },
  {
    id: "crash-probability-l2-q10",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe marginalization in probability models?",
    options: [
      {
        text: "It can compute \\(P(A)\\) by summing joint probabilities over values of another variable.",
        isCorrect: true,
      },
      {
        text: "It is useful when hidden causes or latent variables can produce the same visible outcome.",
        isCorrect: true,
      },
      {
        text: "It replaces the conditional transition model in reinforcement learning with the single most likely next state.",
        isCorrect: false,
      },
      {
        text: "It requires choosing the single most likely hidden variable and ignoring the rest.",
        isCorrect: false,
      },
    ],
    explanation:
      "Marginalization sums over alternative possibilities rather than choosing only one. It can be used with hidden causes, latent topics, or possible future states, but it does not replace a transition model with only the most likely next state.",
  },
  {
    id: "crash-probability-l2-q11",
    chapter: 2,
    difficulty: "medium",
    type: "assertion-reason",
    prompt:
      "Assertion: \\(P(A)=\\sum_B P(A,B)\\) is a marginalization formula.\n\nReason: The formula adds the joint probabilities for the mutually exclusive ways that \\(A\\) can occur with different values of \\(B\\).",
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
      "The assertion is true because marginalization recovers a probability for one variable by summing over another variable. The reason explains why the sum works: each value of \\(B\\) gives a mutually exclusive way for \\(A\\) to appear.",
  },
  {
    id: "crash-probability-l2-q12",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements correctly describe independence and dependence?",
    options: [
      {
        text: "If \\(A\\) and \\(B\\) are independent, knowing \\(B\\) does not change the probability of \\(A\\).",
        isCorrect: true,
      },
      {
        text: "One independence condition is \\(P(A\\mid B)=P(A)\\), when \\(P(B)>0\\).",
        isCorrect: true,
      },
      {
        text: "An equivalent independence condition is \\(P(A,B)=P(A)+P(B)\\).",
        isCorrect: false,
      },
      {
        text: "Dependence means two variables must always have the same value.",
        isCorrect: false,
      },
    ],
    explanation:
      "Independence means information about one event does not change the probability of the other. The joint-product condition is \\(P(A,B)=P(A)P(B)\\), not a sum, and dependence means probabilities change when information is known rather than variables literally taking identical values.",
  },
  {
    id: "crash-probability-l2-q13",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "A table gives \\(P(\\text{infection})=0.35\\) and \\(P(\\text{infection}\\mid\\text{fever})=0.75\\). Which conclusion is justified?",
    options: [
      {
        text: "Infection and fever are dependent in this table.",
        isCorrect: true,
      },
      {
        text: "Fever provides information about infection probability.",
        isCorrect: true,
      },
      {
        text: "Infection is more likely among fever cases than in the overall population.",
        isCorrect: true,
      },
      {
        text: "The events are independent because both probabilities are between 0 and 1.",
        isCorrect: false,
      },
    ],
    explanation:
      "The conditional probability differs from the marginal probability, so knowing fever changes the probability of infection. Being a valid probability between 0 and 1 does not imply independence.",
  },
  {
    id: "crash-probability-l2-q14",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which AI examples rely on dependence between variables rather than independence?",
    options: [
      {
        text: "Next-token prediction uses dependence between previous tokens and the next token.",
        isCorrect: true,
      },
      {
        text: "Image models use dependence among pixels and objects.",
        isCorrect: true,
      },
      {
        text: "Medical prediction uses dependence between symptoms and diagnoses.",
        isCorrect: true,
      },
      {
        text: "RL control uses dependence between actions and possible future states.",
        isCorrect: true,
      },
    ],
    explanation:
      "Machine learning is useful because inputs often contain information about outputs. If the relevant variables were independent, then the input, context, symptoms, or action would not change the target distribution in a useful way.",
  },
  {
    id: "crash-probability-l2-q15",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "If \\(X\\) and \\(Y\\) were independent in a supervised learning task, so that \\(P(Y\\mid X)=P(Y)\\), what would follow?",
    options: [
      {
        text: "Knowing \\(X\\) would not improve the probability distribution for \\(Y\\).",
        isCorrect: true,
      },
      {
        text: "The input would be useless for predicting the output in the probabilistic sense.",
        isCorrect: true,
      },
      {
        text: "A learned model could still improve prediction from \\(X\\) by exploiting hidden dependence between \\(X\\) and \\(Y\\) under this condition.",
        isCorrect: false,
      },
      {
        text: "The model would automatically become perfectly accurate because independence removes uncertainty.",
        isCorrect: false,
      },
    ],
    explanation:
      "If \\(P(Y\\mid X)=P(Y)\\), the input does not change the output distribution. That condition rules out exploitable dependence between \\(X\\) and \\(Y\\) in this probabilistic setup; it does not remove uncertainty or guarantee accuracy.",
  },
  {
    id: "crash-probability-l2-q16",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly distinguish ordinary independence from conditional independence?",
    options: [
      {
        text: "Ordinary independence can be written as \\(A\\perp B\\).",
        isCorrect: true,
      },
      {
        text: "Conditional independence can be written as \\(A\\perp B\\mid C\\).",
        isCorrect: true,
      },
      {
        text: "Two variables can be statistically related overall but become closer to independent after conditioning on a third variable.",
        isCorrect: true,
      },
      {
        text: "Conditional independence means \\(A\\) and \\(B\\) have no relationship under any possible information.",
        isCorrect: false,
      },
    ],
    explanation:
      "Conditional independence says the relationship is evaluated after conditioning on another variable. It does not require the variables to be unrelated in every unconditional view; the ice-cream and drowning example illustrates how season or temperature can explain an observed association.",
  },
  {
    id: "crash-probability-l2-q17",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "In Bayes' theorem \\(P(H\\mid D)=\\frac{P(D\\mid H)P(H)}{P(D)}\\), which term is the prior?",
    options: [
      { text: "\\(P(H)\\)", isCorrect: true },
      { text: "\\(P(D\\mid H)\\)", isCorrect: false },
      { text: "\\(P(H\\mid D)\\)", isCorrect: false },
      { text: "\\(P(D)\\)", isCorrect: false },
    ],
    explanation:
      "The prior is \\(P(H)\\), the probability assigned to the hypothesis before using the new data \\(D\\). The likelihood is \\(P(D\\mid H)\\), the posterior is \\(P(H\\mid D)\\), and \\(P(D)\\) is the evidence or normalizer.",
  },
  {
    id: "crash-probability-l2-q18",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly identify the parts of Bayes' theorem \\(P(H\\mid D)=\\frac{P(D\\mid H)P(H)}{P(D)}\\)?",
    options: [
      { text: "\\(P(H\\mid D)\\) is the posterior.", isCorrect: true },
      { text: "\\(P(D\\mid H)\\) is the likelihood.", isCorrect: true },
      { text: "\\(P(H)\\) is the prior.", isCorrect: true },
      {
        text: "\\(P(D)\\) is a normalizer that includes all ways the data can occur.",
        isCorrect: true,
      },
    ],
    explanation:
      "Bayes' theorem updates a prior into a posterior by weighting hypotheses according to how likely they make the data. The evidence term normalizes the result so the posterior probabilities behave like probabilities.",
  },
  {
    id: "crash-probability-l2-q19",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A disease has \\(P(D)=0.01\\), test sensitivity \\(P(+\\mid D)=0.95\\), and false-positive rate \\(P(+\\mid \\neg D)=0.05\\). What is \\(P(D\\mid +)\\) approximately?",
    options: [
      { text: "\\(0.059\\)", isCorrect: false },
      { text: "\\(0.161\\)", isCorrect: true },
      { text: "\\(0.500\\)", isCorrect: false },
      { text: "\\(0.950\\)", isCorrect: false },
    ],
    explanation:
      "The evidence probability is \\(P(+)=0.95\\cdot0.01+0.05\\cdot0.99=0.059\\). The posterior is \\((0.95\\cdot0.01)/0.059\\approx0.161\\), which is much lower than sensitivity because the disease is rare and false positives exist.",
  },
  {
    id: "crash-probability-l2-q20",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "For the same disease-test setup with \\(P(D)=0.01\\), \\(P(+\\mid D)=0.95\\), and \\(P(+\\mid \\neg D)=0.05\\), which statements are correct?",
    options: [
      {
        text: "The true-positive contribution to \\(P(+)\\) is \\(0.95\\cdot0.01=0.0095\\).",
        isCorrect: true,
      },
      {
        text: "The false-positive contribution to \\(P(+)\\) is \\(0.05\\cdot0.99=0.0495\\).",
        isCorrect: true,
      },
      {
        text: "The false-positive contribution is smaller than the true-positive contribution in this example.",
        isCorrect: false,
      },
      {
        text: "The posterior equals 0.95 because sensitivity and posterior probability are the same conditional.",
        isCorrect: false,
      },
    ],
    explanation:
      "The base rate makes no-disease cases far more common, so even a modest false-positive rate contributes many positives. Here \\(0.0495\\) is larger than \\(0.0095\\), and the posterior is not the sensitivity because \\(P(+\\mid D)\\) and \\(P(D\\mid +)\\) ask opposite conditional questions.",
  },
  {
    id: "crash-probability-l2-q21",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly explain why base rates matter in Bayes' theorem?",
    options: [
      {
        text: "A rare hypothesis can have a low posterior even when the evidence is fairly likely under that hypothesis.",
        isCorrect: true,
      },
      {
        text: "The denominator \\(P(D)\\) includes evidence produced by alternative hypotheses.",
        isCorrect: true,
      },
      {
        text: "False positives cannot dominate the evidence pool when the test has high sensitivity.",
        isCorrect: false,
      },
      {
        text: "Bayes' theorem ignores the prior once likelihoods are known.",
        isCorrect: false,
      },
    ],
    explanation:
      "Bayesian updating combines prior probability with likelihood, so the base rate remains important. A rare disease can still have many false positives relative to true positives because the no-disease population is much larger, even when sensitivity is high.",
  },
  {
    id: "crash-probability-l2-q22",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Suppose \\(P(D)=0.02\\), \\(P(+\\mid D)=0.90\\), and \\(P(+\\mid \\neg D)=0.10\\). Which statements are correct?",
    options: [
      {
        text: "\\(P(+)=0.90\\cdot0.02+0.10\\cdot0.98=0.116\\).",
        isCorrect: true,
      },
      {
        text: "\\(P(D\\mid +)=0.018/0.116\\approx0.155\\).",
        isCorrect: true,
      },
      {
        text: "The posterior is much lower than 0.90 because \\(P(+\\mid D)\\) must be subtracted from \\(P(+\\mid \\neg D)\\).",
        isCorrect: false,
      },
      {
        text: "\\(P(D\\mid +)=P(+\\mid D)\\) because both conditionals mention disease and a positive test.",
        isCorrect: false,
      },
    ],
    explanation:
      "The evidence term adds true positives and false positives, giving \\(0.116\\). Dividing the true-positive contribution \\(0.018\\) by that evidence gives about 15.5%; no subtraction of sensitivity from the false-positive rate is involved.",
  },
  {
    id: "crash-probability-l2-q23",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which pairs of conditional probabilities are generally different and should not be casually reversed?",
    options: [
      {
        text: "\\(P(\\text{positive test}\\mid\\text{disease})\\) and \\(P(\\text{disease}\\mid\\text{positive test})\\).",
        isCorrect: true,
      },
      {
        text: "\\(P(\\text{word bank}\\mid\\text{finance context})\\) and \\(P(\\text{finance context}\\mid\\text{word bank})\\).",
        isCorrect: true,
      },
      {
        text: "\\(P(\\text{class}\\mid\\text{image})\\) and \\(P(\\text{image}\\mid\\text{class})\\).",
        isCorrect: true,
      },
      {
        text: "\\(P(A\\mid B)\\) and \\(P(B\\mid A)\\), unless extra information shows they are equal.",
        isCorrect: true,
      },
    ],
    explanation:
      "Conditionals have direction: the event after the bar is what is known. These pairs may be related by Bayes' theorem, but they are not interchangeable without priors and normalizing evidence.",
  },
  {
    id: "crash-probability-l2-q24",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe Bayes' theorem as reweighting?",
    options: [
      {
        text: "It starts with prior possibilities.",
        isCorrect: true,
      },
      {
        text: "It weights hypotheses by how likely they make the observed data.",
        isCorrect: true,
      },
      {
        text: "It normalizes by the prior probability of the favored hypothesis rather than by the probability of the observed data.",
        isCorrect: false,
      },
      {
        text: "It removes the need to consider hypotheses that also could have produced the data.",
        isCorrect: false,
      },
    ],
    explanation:
      "Bayes' theorem multiplies prior probability by likelihood and then normalizes across possible explanations for the data. The denominator is the evidence probability \\(P(D)\\), not merely the prior probability of whichever hypothesis looks favored.",
  },
  {
    id: "crash-probability-l2-q25",
    chapter: 2,
    difficulty: "medium",
    type: "assertion-reason",
    prompt:
      "Assertion: \\(P(D\\mid H)\\) and \\(P(H\\mid D)\\) are generally not the same quantity.\n\nReason: The evidence term \\(P(D)\\) normalizes posterior probabilities in Bayes' theorem.",
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
      "The assertion is true because reversing the conditioning direction changes the question being asked. The reason is also true because Bayes' theorem uses \\(P(D)\\) as a normalizer, but that fact does not by itself explain why the two reversed conditionals are different.",
  },
  {
    id: "crash-probability-l2-q26",
    chapter: 2,
    difficulty: "hard",
    prompt:
      'A text classifier uses Naive Bayes for spam detection. Which statements correctly describe the "naive" part?',
    options: [
      {
        text: "It commonly assumes words are conditionally independent given the class.",
        isCorrect: true,
      },
      {
        text: "The assumption can be mathematically false while still practically useful.",
        isCorrect: true,
      },
      {
        text: "The model still uses Bayes-style reasoning to estimate \\(P(\\text{class}\\mid\\text{words})\\).",
        isCorrect: true,
      },
      {
        text: "The assumption means words are independent before conditioning on the class in every text corpus.",
        isCorrect: false,
      },
    ],
    explanation:
      "Naive Bayes often assumes conditional independence of features given the class, not universal unconditional independence. The assumption is simplifying and imperfect, but it can make classification tractable and surprisingly effective.",
  },
  {
    id: "crash-probability-l2-q27",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe Bayesian machine learning at the high level introduced here?",
    options: [
      {
        text: "It can represent uncertainty over plausible models or parameters given data.",
        isCorrect: true,
      },
      {
        text: "It is useful in settings such as scientific modeling, small data, and medical decision support.",
        isCorrect: true,
      },
      {
        text: "It asks which models remain plausible after observing data.",
        isCorrect: true,
      },
      {
        text: "It always replaces neural-network training with exact closed-form posterior calculations.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture introduces Bayesian machine learning as a way to reason about uncertainty over models and predictions, including uncertainty over plausible models or parameters. It does not claim that every Bayesian method has simple exact formulas or replaces all neural-network training procedures.",
  },
  {
    id: "crash-probability-l2-q28",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements accurately connect LLM behavior to Bayesian language without overclaiming the implementation?",
    options: [
      {
        text: "LLMs are usually not implemented as explicit classical Bayesian inference systems.",
        isCorrect: true,
      },
      {
        text: "Prompt context can change which continuations are plausible.",
        isCorrect: true,
      },
      {
        text: "It is reasonable to describe some prompt effects as evidence shifting a conditional distribution.",
        isCorrect: true,
      },
      {
        text: 'A context about a laptop can shift the likely interpretation of "Apple" toward the company sense.',
        isCorrect: true,
      },
    ],
    explanation:
      "The key nuance is to use Bayesian language as an interpretation of evidence and plausibility without claiming the model literally runs a hand-coded Bayes formula. Context changes next-token and meaning distributions in a way that fits conditional-probability language.",
  },
  {
    id: "crash-probability-l2-q29",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe supervised learning as estimating \\(P(y\\mid x)\\)?",
    options: [
      {
        text: "Training data can be represented as pairs \\((x_i,y_i)\\).",
        isCorrect: true,
      },
      {
        text: "A classifier can estimate \\(P(Y=k\\mid X=x)\\) for class \\(k\\).",
        isCorrect: true,
      },
      {
        text: "A probabilistic regression model can estimate a distribution over possible numeric outputs.",
        isCorrect: true,
      },
      {
        text: "A non-probabilistic regression model may output an expected value such as \\(\\mathbb{E}[Y\\mid X=x]\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "Supervised learning uses input-output examples to learn how outputs depend on inputs. Classification often estimates class probabilities, while regression may estimate either a distribution or a conditional expected value.",
  },
  {
    id: "crash-probability-l2-q30",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which factorization matches repeated conditional next-token prediction for a token sequence \\(X_1,\\ldots,X_T\\)?",
    options: [
      {
        text: "\\(P(X_1,\\ldots,X_T)=\\prod_{t=1}^{T}P(X_t\\mid X_1,\\ldots,X_{t-1})\\), using an empty context for \\(t=1\\).",
        isCorrect: true,
      },
      {
        text: "\\(P(X_1,\\ldots,X_T)=\\prod_{t=1}^{T}P(X_t)\\), regardless of context.",
        isCorrect: false,
      },
      {
        text: "\\(P(X_1,\\ldots,X_T)=P(X_T\\mid X_1)\\), because only the first and last tokens matter.",
        isCorrect: false,
      },
      {
        text: "\\(P(X_1,\\ldots,X_T)=\\sum_{t=1}^{T}P(X_t\\mid X_1,\\ldots,X_{t-1})\\), because sequence probabilities add across positions.",
        isCorrect: false,
      },
    ],
    explanation:
      "Autoregressive language modeling decomposes a sequence probability into a product of next-token conditional probabilities. Multiplying reflects joint probability factorization; adding the conditional probabilities would not produce a valid sequence probability.",
  },
  {
    id: "crash-probability-l2-q31",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly compare an RL transition model and a policy?",
    options: [
      {
        text: "\\(P(S_{t+1}\\mid S_t,A_t)\\) describes environment dynamics.",
        isCorrect: true,
      },
      {
        text: "\\(\\pi(A_t\\mid S_t)\\) describes a conditional action distribution for the agent.",
        isCorrect: true,
      },
      {
        text: "Both expressions are conditional probability structures.",
        isCorrect: true,
      },
      {
        text: "The transition model and policy condition on exactly the same variables and predict exactly the same target.",
        isCorrect: false,
      },
    ],
    explanation:
      "The environment transition predicts next state from state and action, while the policy predicts an action distribution from the current state. Both use conditional probability, but they describe different parts of the reinforcement learning setup.",
  },
  {
    id: "crash-probability-l2-q32",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which conditional probability expressions fit diffusion-style denoising at the level introduced here?",
    options: [
      {
        text: "\\(P(x_{t-1}\\mid x_t)\\), for a less noisy state given a noisy state.",
        isCorrect: true,
      },
      {
        text: "\\(P(\\epsilon\\mid x_t,t)\\), for noise prediction given the noisy state and time step.",
        isCorrect: true,
      },
      {
        text: "\\(P(\\text{clean structure}\\mid\\text{noisy observation})\\), as an informal denoising description.",
        isCorrect: true,
      },
      {
        text: "\\(P(x_t)\\) alone, with no conditioning on the noisy input during denoising.",
        isCorrect: false,
      },
    ],
    explanation:
      "Diffusion denoising is conditional because the model receives a noisy input and predicts something about the less noisy image or the noise itself. An unconditional probability of a noisy state alone misses the input-output structure emphasized in the lecture.",
  },
  {
    id: "crash-probability-l2-q33",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A robot action has transition probabilities \\(P(s_1\\mid s,a)=0.70\\), \\(P(s_2\\mid s,a)=0.20\\), and \\(P(s_3\\mid s,a)=0.10\\), with rewards 5, 0, and -10 in those next states. Which statements are correct?",
    options: [
      {
        text: "The transition probabilities form a conditional distribution over next states.",
        isCorrect: true,
      },
      {
        text: "The expected immediate reward from these next states is \\(0.70\\cdot5+0.20\\cdot0+0.10\\cdot(-10)=2.5\\).",
        isCorrect: true,
      },
      {
        text: "The action has a nonzero chance of a negative next-state reward.",
        isCorrect: true,
      },
      {
        text: "The most likely next state alone is enough to compute the expected reward exactly.",
        isCorrect: false,
      },
    ],
    explanation:
      "The conditional transition probabilities sum to one over possible next states for the given state-action pair. Expected reward uses every possible next state and its probability, not only the most likely next state.",
  },
  {
    id: "crash-probability-l2-q34",
    chapter: 2,
    difficulty: "hard",
    prompt:
      'In the spam table, 45 of 100 emails are spam and contain "prize," 15 contain "prize" but are not spam, 5 are spam without "prize," and 35 are not spam without "prize." Which statements are correct?',
    options: [
      {
        text: "\\(P(\\text{spam})=0.50\\).",
        isCorrect: true,
      },
      {
        text: "\\(P(\\text{contains prize})=0.60\\).",
        isCorrect: true,
      },
      {
        text: "\\(P(\\text{spam},\\text{contains prize})=0.45\\).",
        isCorrect: true,
      },
      {
        text: "\\(P(\\text{spam}\\mid\\text{contains prize})=0.75\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The marginal spam count is 50, the marginal prize count is 60, and the joint spam-and-prize count is 45 out of 100. The conditional probability among prize-containing emails is \\(45/60=0.75\\).",
  },
  {
    id: "crash-probability-l2-q35",
    chapter: 2,
    difficulty: "hard",
    prompt:
      'Using the same spam table, are spam and "contains prize" independent?',
    options: [
      {
        text: "No, because \\(P(\\text{spam}\\mid\\text{contains prize})=0.75\\) differs from \\(P(\\text{spam})=0.50\\).",
        isCorrect: true,
      },
      {
        text: "No, because \\(P(\\text{spam},\\text{contains prize})=0.45\\) differs from \\(P(\\text{spam})P(\\text{contains prize})=0.30\\).",
        isCorrect: true,
      },
      {
        text: "The phrase contains predictive information about spam status in this table.",
        isCorrect: true,
      },
      {
        text: "Yes, because both spam and contains prize have marginal probabilities at least 0.50.",
        isCorrect: false,
      },
    ],
    explanation:
      "The conditional-probability and product-of-marginals checks both reject independence. Large marginal probabilities do not imply independence; what matters is whether knowing one event changes the probability of the other.",
  },
  {
    id: "crash-probability-l2-q36",
    chapter: 2,
    difficulty: "medium",
    type: "assertion-reason",
    prompt:
      "Assertion: If \\(X\\) and \\(Y\\) are independent, then \\(X\\) is useful for predicting \\(Y\\) in the probabilistic sense.\n\nReason: Independence means \\(P(Y\\mid X)=P(Y)\\).",
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
      "The assertion is false because if \\(P(Y\\mid X)=P(Y)\\), then observing \\(X\\) does not change the distribution of \\(Y\\). The reason is true: that equality is exactly the probabilistic statement of independence in this prediction setting.",
  },
  {
    id: "crash-probability-l2-q37",
    chapter: 2,
    difficulty: "hard",
    prompt:
      'Which statements correctly connect conditional probability to the statement "prediction = conditional probability"?',
    options: [
      {
        text: "For classification, prediction can mean estimating \\(P(\\text{class}\\mid\\text{input})\\).",
        isCorrect: true,
      },
      {
        text: "For language modeling, prediction can mean estimating \\(P(\\text{next token}\\mid\\text{text prefix})\\).",
        isCorrect: true,
      },
      {
        text: "For RL dynamics, prediction can mean estimating \\(P(\\text{next state}\\mid\\text{state, action})\\).",
        isCorrect: true,
      },
      {
        text: "For denoising, prediction can mean estimating noise or cleaner structure conditioned on a noisy input.",
        isCorrect: true,
      },
    ],
    explanation:
      "The same conditional form appears across AI domains: an output distribution is estimated after input information is known. The targets differ across domains, but the mathematical structure remains conditional probability.",
  },
  {
    id: "crash-probability-l2-q38",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly identify the evidence term \\(P(D)\\) in Bayes' theorem?",
    options: [
      {
        text: "It can be computed by summing \\(P(D\\mid H_i)P(H_i)\\) over a complete set of hypotheses.",
        isCorrect: true,
      },
      {
        text: "In a binary disease example, it includes both true positives and false positives.",
        isCorrect: true,
      },
      {
        text: "It acts as a normalizer for posterior probabilities.",
        isCorrect: true,
      },
      {
        text: "It is generally not the same as the likelihood \\(P(D\\mid H)\\) for one hypothesis.",
        isCorrect: true,
      },
    ],
    explanation:
      "The evidence term describes how likely the observed data is overall, considering all ways the data can occur. It normalizes the posterior and is generally not equal to one hypothesis's likelihood because alternative hypotheses can also produce the data.",
  },
  {
    id: "crash-probability-l2-q39",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A hidden-topic model has \\(P(\\text{word}=w,Z=\\text{sports})=0.06\\), \\(P(w,Z=\\text{finance})=0.03\\), and \\(P(w,Z=\\text{health})=0.01\\). What is \\(P(\\text{word}=w)\\) after marginalizing over topic \\(Z\\)?",
    options: [
      { text: "\\(0.01\\)", isCorrect: false },
      { text: "\\(0.03\\)", isCorrect: false },
      { text: "\\(0.06\\)", isCorrect: false },
      { text: "\\(0.10\\)", isCorrect: true },
    ],
    explanation:
      "The word can occur through any of the hidden topic values, so marginalization sums those joint probabilities. The calculation is \\(0.06+0.03+0.01=0.10\\), not the largest single topic contribution.",
  },
  {
    id: "crash-probability-l2-q40",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly synthesize Lecture 2's core probability ideas for AI?",
    options: [
      {
        text: "Conditional probability explains how input information changes an output distribution.",
        isCorrect: true,
      },
      {
        text: "Joint probability describes variables or events appearing together.",
        isCorrect: true,
      },
      {
        text: "Marginalization sums over hidden or alternative possibilities.",
        isCorrect: true,
      },
      {
        text: "Dependence is what lets inputs contain useful information about outputs.",
        isCorrect: true,
      },
    ],
    explanation:
      "These ideas work together: prediction uses conditional probability, joint probabilities describe co-occurrence, marginalization handles hidden alternatives, and dependence makes learning possible. If inputs and outputs were independent, conditioning on the input would not improve prediction.",
  },
  {
    id: "crash-probability-l2-q41",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A table has \\(P(A,B)=0.18\\), \\(P(\\neg A,B)=0.12\\), \\(P(A,\\neg B)=0.22\\), and \\(P(\\neg A,\\neg B)=0.48\\). What is \\(P(A\\mid B)\\)?",
    options: [
      { text: "\\(0.18\\)", isCorrect: false },
      { text: "\\(0.40\\)", isCorrect: false },
      { text: "\\(0.60\\)", isCorrect: true },
      { text: "\\(0.82\\)", isCorrect: false },
    ],
    explanation:
      "The conditioning event has probability \\(P(B)=0.18+0.12=0.30\\). Therefore \\(P(A\\mid B)=P(A,B)/P(B)=0.18/0.30=0.60\\), not the joint probability by itself.",
  },
  {
    id: "crash-probability-l2-q42",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly use the table with \\(P(A,B)=0.18\\), \\(P(\\neg A,B)=0.12\\), \\(P(A,\\neg B)=0.22\\), and \\(P(\\neg A,\\neg B)=0.48\\)?",
    options: [
      { text: "\\(P(A)=0.40\\).", isCorrect: true },
      { text: "\\(P(B)=0.30\\).", isCorrect: true },
      {
        text: "\\(P(A\\mid B)=P(A)\\), so \\(A\\) and \\(B\\) are independent.",
        isCorrect: false,
      },
      { text: "\\(P(A,B)=P(A)P(B)\\).", isCorrect: false },
    ],
    explanation:
      "The marginal probabilities are \\(P(A)=0.18+0.22=0.40\\) and \\(P(B)=0.18+0.12=0.30\\). Independence would require \\(P(A,B)=0.40\\cdot0.30=0.12\\), but the observed joint probability is 0.18.",
  },
  {
    id: "crash-probability-l2-q43",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Suppose \\(P(S\\mid D_1)=0.80\\), \\(P(S\\mid D_2)=0.30\\), \\(P(D_1)=0.25\\), and \\(P(D_2)=0.75\\), where \\(D_1\\) and \\(D_2\\) are exhaustive disease states. Which statements are correct?",
    options: [
      {
        text: "\\(P(S)=0.80\\cdot0.25+0.30\\cdot0.75=0.425\\).",
        isCorrect: true,
      },
      { text: "\\(P(D_1,S)=0.20\\).", isCorrect: true },
      { text: "\\(P(D_1\\mid S)\\approx0.471\\).", isCorrect: true },
      {
        text: "\\(P(D_1\\mid S)=0.80\\), because sensitivity and posterior are the same conditional.",
        isCorrect: false,
      },
    ],
    explanation:
      "The symptom probability is found by marginalizing over the exhaustive disease states. The posterior is \\(P(D_1\\mid S)=P(S\\mid D_1)P(D_1)/P(S)=0.20/0.425\\approx0.471\\), which is not the same as \\(P(S\\mid D_1)\\).",
  },
  {
    id: "crash-probability-l2-q44",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A binary classifier gives \\(P(Y=1\\mid X=x)=0.70\\) and \\(P(Y=0\\mid X=x)=0.30\\). Which statements are correct?",
    options: [
      {
        text: "This is a conditional distribution over \\(Y\\) for the fixed input \\(x\\).",
        isCorrect: true,
      },
      {
        text: "The two probabilities sum to one for this binary output space.",
        isCorrect: true,
      },
      {
        text: "A maximum-probability decision rule would choose \\(Y=1\\).",
        isCorrect: true,
      },
      {
        text: "The distribution still represents uncertainty about \\(Y\\) for this input.",
        isCorrect: true,
      },
    ],
    explanation:
      "For a fixed input, the classifier gives a conditional probability distribution over the possible labels. Choosing the largest probability is a decision rule applied to that distribution, not a removal of uncertainty.",
  },
  {
    id: "crash-probability-l2-q45",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Events \\(A\\) and \\(B\\) satisfy \\(P(A)=0.40\\), \\(P(B)=0.50\\), and \\(P(A,B)=0.20\\). Which conclusion is correct?",
    options: [
      { text: "\\(A\\) and \\(B\\) are independent.", isCorrect: true },
      {
        text: "\\(A\\) and \\(B\\) are dependent because \\(P(A,B)\\) is smaller than \\(P(B)\\).",
        isCorrect: false,
      },
      { text: "\\(P(A\\mid B)=0.20\\).", isCorrect: false },
      { text: "\\(P(B\\mid A)=0.50\\) proves dependence.", isCorrect: false },
    ],
    explanation:
      "Independence holds because \\(P(A)P(B)=0.40\\cdot0.50=0.20=P(A,B)\\). Also \\(P(A\\mid B)=0.20/0.50=0.40=P(A)\\), so knowing \\(B\\) does not change the probability of \\(A\\).",
  },
  {
    id: "crash-probability-l2-q46",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Ice cream purchases and drowning incidents are associated overall, but both vary strongly with temperature. Which statements correctly express the conditional-independence lesson?",
    options: [
      {
        text: "The overall association can be partly explained by a third variable.",
        isCorrect: true,
      },
      {
        text: "A notation such as \\(A\\perp B\\mid C\\) means independence after conditioning on \\(C\\).",
        isCorrect: true,
      },
      {
        text: "Conditional independence means the variables were necessarily independent before conditioning.",
        isCorrect: false,
      },
      {
        text: "Conditioning on temperature forces both variables to have equal probabilities.",
        isCorrect: false,
      },
    ],
    explanation:
      "Conditional independence is about what remains after a third variable is known. It does not say the variables were unassociated overall, and it does not require their probabilities to become equal.",
  },
  {
    id: "crash-probability-l2-q47",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A test has \\(P(D)=0.05\\), \\(P(+\\mid D)=0.90\\), and \\(P(+\\mid \\neg D)=0.20\\). Which statements are correct?",
    options: [
      {
        text: "\\(P(+)=0.90\\cdot0.05+0.20\\cdot0.95=0.235\\).",
        isCorrect: true,
      },
      { text: "\\(P(D\\mid +)=0.045/0.235\\approx0.191\\).", isCorrect: true },
      {
        text: "The false-positive contribution is \\(0.190\\).",
        isCorrect: true,
      },
      {
        text: "\\(P(D\\mid +)=0.90\\), because the test is 90% sensitive.",
        isCorrect: false,
      },
    ],
    explanation:
      "Bayes' theorem uses both the prior and the likelihood, and the evidence term includes true positives and false positives. The false-positive contribution is large because \\(\\neg D\\) has probability 0.95, so the posterior is about 19.1%, not 90%.",
  },
  {
    id: "crash-probability-l2-q48",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the chain-rule factorization behind autoregressive language modeling?",
    options: [
      {
        text: "\\(P(X_1,X_2,X_3)=P(X_1)P(X_2\\mid X_1)P(X_3\\mid X_1,X_2)\\).",
        isCorrect: true,
      },
      {
        text: "The factorization uses conditional probabilities rather than assuming tokens are independent.",
        isCorrect: true,
      },
      {
        text: "A transformer language model can approximate the next-token conditionals in this product.",
        isCorrect: true,
      },
      {
        text: "Multiplying conditional probabilities is how the joint sequence probability is assembled.",
        isCorrect: true,
      },
    ],
    explanation:
      "The chain rule decomposes a joint sequence probability into a product of conditional next-token probabilities. Autoregressive LLMs use this structure by repeatedly estimating the next token from the previous context.",
  },
  {
    id: "crash-probability-l2-q49",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Two hypotheses have unnormalized Bayes weights \\(w_1=0.12\\) and \\(w_2=0.03\\) after multiplying prior by likelihood. What is the posterior probability of hypothesis 1?",
    options: [
      { text: "\\(0.12\\)", isCorrect: false },
      { text: "\\(0.15\\)", isCorrect: false },
      { text: "\\(0.80\\)", isCorrect: true },
      { text: "\\(4.00\\)", isCorrect: false },
    ],
    explanation:
      "Posterior probabilities are normalized weights, not the raw prior-times-likelihood scores. The normalizer is \\(0.12+0.03=0.15\\), so \\(P(H_1\\mid D)=0.12/0.15=0.80\\); choosing 0.12 would skip the normalization step.",
  },
  {
    id: "crash-probability-l2-q50",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "An RL policy has \\(\\pi(a_1\\mid s)=0.70\\) and \\(\\pi(a_2\\mid s)=0.30\\). Action \\(a_1\\) has expected reward 4 and action \\(a_2\\) has expected reward 10. Which statements are correct?",
    options: [
      {
        text: "The policy is a conditional distribution over actions given state \\(s\\).",
        isCorrect: true,
      },
      {
        text: "The policy-weighted expected immediate reward is \\(0.70\\cdot4+0.30\\cdot10=5.8\\).",
        isCorrect: true,
      },
      {
        text: "The policy guarantees reward 5.8 on every trial.",
        isCorrect: false,
      },
      {
        text: "\\(\\pi(a_1\\mid s)=0.70\\) describes the environment transition probability.",
        isCorrect: false,
      },
    ],
    explanation:
      "A policy gives action probabilities conditioned on the state, while an environment transition model predicts next states. The expected reward averages over the stochastic action choice, so 5.8 is an average rather than a guaranteed realized reward.",
  },
  {
    id: "crash-probability-l2-q51",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A Naive Bayes spam model compares unnormalized scores \\(P(\\text{spam})P(w_1\\mid\\text{spam})P(w_2\\mid\\text{spam})\\) and \\(P(\\text{not spam})P(w_1\\mid\\text{not spam})P(w_2\\mid\\text{not spam})\\). Which statements are correct?",
    options: [
      {
        text: "The product structure reflects a conditional-independence assumption for words given the class.",
        isCorrect: true,
      },
      {
        text: "The scores must be normalized before being interpreted as posterior probabilities.",
        isCorrect: true,
      },
      {
        text: "The class prior can change the final comparison even with the same word likelihoods.",
        isCorrect: true,
      },
      {
        text: "The model assumes \\(P(\\text{spam}\\mid w_1,w_2)=P(w_1,w_2\\mid\\text{spam})\\) without normalization.",
        isCorrect: false,
      },
    ],
    explanation:
      "Naive Bayes multiplies a class prior by feature likelihoods under a simplifying conditional-independence assumption. The resulting class scores need normalization to become posterior probabilities, and priors can materially affect the result.",
  },
  {
    id: "crash-probability-l2-q52",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A latent variable \\(Z\\) can take values \\(z_1,z_2,z_3\\). Which statements correctly describe marginalizing it out of \\(P(Y,Z\\mid X=x)\\)?",
    options: [
      { text: "\\(P(Y\\mid X=x)=\\sum_z P(Y,z\\mid X=x)\\).", isCorrect: true },
      {
        text: "Each term in the sum keeps the same observed input \\(X=x\\).",
        isCorrect: true,
      },
      {
        text: "The operation accounts for multiple hidden explanations for \\(Y\\).",
        isCorrect: true,
      },
      {
        text: "The operation is useful when \\(Z\\) is not directly observed.",
        isCorrect: true,
      },
    ],
    explanation:
      "Marginalizing a latent variable sums over its possible hidden values while keeping the observed conditioning information fixed. This lets the model account for several hidden explanations rather than committing to only one.",
  },
  {
    id: "crash-probability-l2-q53",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "In a corpus, \\(P(\\text{word bank}\\mid\\text{finance context})=0.08\\) and \\(P(\\text{finance context}\\mid\\text{word bank})=0.60\\). Which statement is correct?",
    options: [
      {
        text: "The two numbers can differ because they condition on different known information.",
        isCorrect: true,
      },
      {
        text: "The two numbers must be equal because they mention the same word and context.",
        isCorrect: false,
      },
      {
        text: "\\(0.08\\) is the posterior probability of finance context after observing the word bank.",
        isCorrect: false,
      },
      {
        text: "\\(0.60\\) is the probability of seeing the word bank inside finance contexts.",
        isCorrect: false,
      },
    ],
    explanation:
      "The event after the conditioning bar is what is assumed known. Seeing finance context and asking about the word is not the same question as seeing the word bank and asking about the context.",
  },
  {
    id: "crash-probability-l2-q54",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which rows can be valid conditional distributions over next states for a fixed state-action pair?",
    options: [
      { text: "\\((0.70,0.20,0.10)\\).", isCorrect: true },
      { text: "\\((0.00,0.40,0.60)\\).", isCorrect: true },
      { text: "\\((0.60,0.60,-0.20)\\).", isCorrect: false },
      { text: "\\((0.50,0.30,0.30)\\).", isCorrect: false },
    ],
    explanation:
      "For a fixed conditioning state-action pair, probabilities over next states must be nonnegative and sum to one. The invalid rows either include a negative probability or sum to more than one.",
  },
  {
    id: "crash-probability-l2-q55",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A diagnostic system has prior odds \\(P(H_1):P(H_2)=1:3\\). Evidence is four times as likely under \\(H_1\\) as under \\(H_2\\). Which statements are correct?",
    options: [
      { text: "The likelihood ratio favors \\(H_1\\).", isCorrect: true },
      { text: "The posterior odds are \\(4\\cdot1:3=4:3\\).", isCorrect: true },
      {
        text: "The posterior probability of \\(H_1\\) is \\(4/7\\).",
        isCorrect: true,
      },
      {
        text: "The posterior probability of \\(H_1\\) is \\(4/5\\) because the likelihood ratio is 4.",
        isCorrect: false,
      },
    ],
    explanation:
      "Bayesian updating can be expressed in odds form: posterior odds equal prior odds times the likelihood ratio. Starting from \\(1:3\\) and multiplying by 4 gives \\(4:3\\), so the normalized posterior for \\(H_1\\) is \\(4/(4+3)=4/7\\).",
  },
  {
    id: "crash-probability-l2-q56",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which equations are valid consequences of the conditional-probability definition when denominators are positive?",
    options: [
      { text: "\\(P(A,B)=P(A\\mid B)P(B)\\).", isCorrect: true },
      { text: "\\(P(A,B,C)=P(A\\mid B,C)P(B\\mid C)P(C)\\).", isCorrect: true },
      {
        text: "\\(P(A\\mid B)=\\frac{P(B\\mid A)P(A)}{P(B)}\\).",
        isCorrect: true,
      },
      {
        text: "\\(P(A)=\\sum_b P(A,b)\\) when the values \\(b\\) partition the other variable.",
        isCorrect: true,
      },
    ],
    explanation:
      "These equations are the basic algebra behind conditioning, Bayes' theorem, chain-rule factorization, and marginalization. They are valid only when the relevant conditioning denominators are positive and the summed values form a complete partition.",
  },
  {
    id: "crash-probability-l2-q57",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A feature \\(X\\) has \\(P(Y=1)=0.50\\), \\(P(Y=1\\mid X=0)=0.50\\), and \\(P(Y=1\\mid X=1)=0.50\\). Which conclusion best follows?",
    options: [
      {
        text: "This feature does not change the distribution of \\(Y\\) in the shown cases.",
        isCorrect: true,
      },
      {
        text: "\\(X\\) is guaranteed to be causally unrelated to \\(Y\\) in every possible setting.",
        isCorrect: false,
      },
      {
        text: "\\(X=1\\) makes \\(Y=1\\) more likely than \\(X=0\\).",
        isCorrect: false,
      },
      {
        text: "The feature proves the classifier will be perfectly calibrated.",
        isCorrect: false,
      },
    ],
    explanation:
      "The conditional probabilities shown are equal to the marginal probability, so this feature does not help distinguish \\(Y\\) in the displayed distribution. That is a statistical-prediction statement, not a universal causal proof or a calibration guarantee.",
  },
  {
    id: "crash-probability-l2-q58",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly interpret \\(P(\\epsilon\\mid x_t,t)\\) in a diffusion-style denoising model?",
    options: [
      {
        text: "The noise prediction is conditioned on the noisy state \\(x_t\\).",
        isCorrect: true,
      },
      {
        text: "The time step \\(t\\) can change the relevant denoising distribution.",
        isCorrect: true,
      },
      {
        text: "The expression says the noise is predicted without seeing the noisy input.",
        isCorrect: false,
      },
      {
        text: "The expression is a joint probability of clean image and text prompt with no conditioning.",
        isCorrect: false,
      },
    ],
    explanation:
      "The vertical bar indicates that the noisy state and time step are information used for the prediction. Diffusion denoising is therefore a conditional problem, not an unconditional joint statement.",
  },
  {
    id: "crash-probability-l2-q59",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Events \\(A\\) and \\(B\\) satisfy \\(P(A)=0.60\\), \\(P(B)=0.50\\), and \\(P(A,B)=0.40\\). Which statements are correct?",
    options: [
      { text: "\\(P(A\\cup B)=0.70\\).", isCorrect: true },
      { text: "\\(P(A\\mid B)=0.80\\).", isCorrect: true },
      { text: "\\(P(B\\mid A)=\\frac{2}{3}\\).", isCorrect: true },
      {
        text: "The events are independent because \\(P(A\\cup B)<1\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The union is \\(0.60+0.50-0.40=0.70\\). The conditionals are \\(P(A\\mid B)=0.40/0.50=0.80\\) and \\(P(B\\mid A)=0.40/0.60=2/3\\), while independence would require \\(0.40=0.60\\cdot0.50\\), which is false.",
  },
  {
    id: "crash-probability-l2-q60",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which applied-math habits are essential when using conditional probability in AI systems?",
    options: [
      {
        text: "Keep track of which event is after the conditioning bar.",
        isCorrect: true,
      },
      {
        text: "Check whether probabilities are joint, marginal, or conditional before combining them.",
        isCorrect: true,
      },
      {
        text: "Normalize Bayesian weights before treating them as posterior probabilities.",
        isCorrect: true,
      },
      {
        text: "Use marginalization when hidden alternatives can produce the observed event.",
        isCorrect: true,
      },
    ],
    explanation:
      "Most mistakes in this material come from mixing probability types, reversing conditionals, or forgetting hidden alternatives. Careful notation and normalization are what make the same probability tools reliable in classification, language modeling, reinforcement learning, and denoising.",
  },
];
