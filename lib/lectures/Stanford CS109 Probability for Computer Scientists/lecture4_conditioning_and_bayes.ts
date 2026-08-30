import { Question } from "../../quiz";

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
    chapter: 4,
    difficulty,
    prompt,
    options: options.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

export const stanfordCS109Lecture4ConditioningAndBayesQuestions: Question[] = [
  makeQuestion(
    "cs109-lect4-q01",
    "easy",
    "Which statements correctly distinguish intersection, union, and conditional probability for events \\(E\\) and \\(F\\)?",
    [
      ["\\(P(E\\cap F)\\) is the probability that both events occur.", true],
      [
        "\\(P(E\\cup F)\\) is the probability that at least one event occurs.",
        true,
      ],
      [
        "\\(P(E\\mid F)\\) evaluates the target event after taking the conditioning event as observed.",
        true,
      ],
      [
        "The order in \\(P(E\\mid F)\\) identifies the target event on the left and the observed event on the right.",
        true,
      ],
    ],
    "Intersection asks for two events jointly, union asks for either or both, and conditioning restricts attention to the world in which the event after the bar has occurred. The order around the conditioning bar therefore carries semantic information that ordinary intersection does not.",
  ),
  makeQuestion(
    "cs109-lect4-q02",
    "easy",
    "What changes when calculating \\(P(E\\mid F)\\) after learning that \\(F\\) occurred?",
    [
      ["The effective sample space becomes the outcomes in \\(F\\).", true],
      ["The favorable outcomes become those in \\(E\\cap F\\).", true],
      [
        "The probability can differ from the unconditional value \\(P(E)\\).",
        true,
      ],
      [
        "Every outcome outside \\(F\\) remains in the denominator with its original weight.",
        false,
      ],
    ],
    "Conditioning discards outcomes inconsistent with the observation and then renormalizes probability within the remaining world. The target event is correspondingly restricted to its overlap with \\(F\\), which is why new information can increase, decrease, or leave unchanged the probability of \\(E\\).",
  ),
  makeQuestion(
    "cs109-lect4-q03",
    "easy",
    "For events with \\(P(F)>0\\), which formulas are correct?",
    [
      ["\\(P(E\\mid F)=P(E\\cap F)/P(F)\\).", true],
      ["\\(P(E\\cap F)=P(F)P(E\\mid F)\\).", true],
      ["\\(P(E\\mid F)=P(E)/P(F)\\) for all events.", false],
      ["\\(P(E\\cap F)=P(E)+P(F)\\) for all events.", false],
    ],
    "The definition of conditional probability divides the joint mass inside both events by the mass of the observed event. Rearranging gives the two-event chain rule; omitting the intersection from the numerator or replacing joint occurrence with addition changes the question being computed.",
  ),
  makeQuestion(
    "cs109-lect4-q04",
    "easy",
    "Why is the elementary definition \\(P(E\\mid F)=P(E\\cap F)/P(F)\\) undefined when \\(P(F)=0\\)?",
    [
      [
        "It would require division by zero after conditioning on a zero-probability event.",
        true,
      ],
      ["It would force the target-event probability to equal zero.", false],
      [
        "It would make every event mutually exclusive with the observed event.",
        false,
      ],
      [
        "It would imply that the full sample space has probability zero.",
        false,
      ],
    ],
    "The introductory discrete definition normalizes by the probability of the observed event, so a zero denominator does not produce a defined ratio. This limitation concerns the conditioning operation; it does not force unrelated event probabilities or the total probability of the sample space to change.",
  ),
  makeQuestion(
    "cs109-lect4-q05",
    "easy",
    "Two independent fair dice produce values \\(D_1,D_2\\). Let \\(E\\) be the event \\(D_1+D_2=4\\). Which statements are correct?",
    [
      ["\\(P(E)=3/36=1/12\\).", true],
      ["\\(P(E\\mid D_1=1)=1/6\\).", true],
      ["\\(P(E\\mid D_1=2)=1/6\\).", true],
      ["\\(P(E\\mid D_1=6)=0\\).", true],
    ],
    "Without an observation, the favorable ordered pairs are \\((1,3),(2,2),(3,1)\\). Fixing the first die leaves six equally likely second-die values; exactly one works when the first die is 1, 2, or 3, while no second-die value can offset a first value of 6 to reach 4.",
  ),
  makeQuestion(
    "cs109-lect4-q06",
    "easy",
    "Within the conditioned world where \\(F\\) occurred and \\(P(F)>0\\), which statements are correct?",
    [
      ["\\(P(E\\mid F)+P(E^c\\mid F)=1\\).", true],
      ["\\(P(S\\mid F)=1\\).", true],
      [
        "The ordinary probability axioms still apply after conditioning on \\(F\\).",
        true,
      ],
      ["\\(P(E^c\\mid F)=1-P(F)\\).", false],
    ],
    "Conditioning creates a normalized probability distribution over the outcomes in \\(F\\), so the usual axioms and complement rule continue to hold there. The complement of \\(E\\) within that world is calculated from \\(P(E\\mid F)\\), not from the unconditional probability of the conditioning event.",
  ),
  makeQuestion(
    "cs109-lect4-q07",
    "easy",
    "A device fails with probability 0.08, and an alarm sounds with probability 0.75 given a failure. Which quantities can be calculated directly from these facts?",
    [
      ["\\(P(\\text{failure and alarm})=0.08\\times0.75=0.06\\).", true],
      ["\\(P(\\text{no failure})=0.92\\).", true],
      ["\\(P(\\text{failure}\\mid\\text{alarm})=0.75\\).", false],
      [
        "\\(P(\\text{alarm})=0.75\\) without knowing alarm behavior when there is no failure.",
        false,
      ],
    ],
    "The chain rule multiplies the failure prior by the alarm likelihood to obtain their joint probability, and the complement gives the no-failure probability. Reversing a conditional is not valid, and the total alarm rate also depends on how often false alarms occur without a failure.",
  ),
  makeQuestion(
    "cs109-lect4-q08",
    "easy",
    "Which statement best explains why \\(P(E\\mid F)\\) and \\(P(F\\mid E)\\) are generally different?",
    [
      [
        "They normalize the same joint mass by different observed-event probabilities.",
        true,
      ],
      ["Intersection changes order when the conditional is reversed.", false],
      [
        "Only the expression with the more likely event on the right is defined.",
        false,
      ],
      ["The two expressions use different definitions of probability.", false],
    ],
    "Both conditionals contain the same numerator \\(P(E\\cap F)\\), because intersection is symmetric, but their denominators are \\(P(F)\\) and \\(P(E)\\). They therefore describe different restricted worlds and agree only in special cases, not as a general rule.",
  ),
  makeQuestion(
    "cs109-lect4-q09",
    "easy",
    "Which statements correctly apply the generalized chain rule to events \\(E_1,E_2,E_3\\)?",
    [
      [
        "The three-event joint probability factors into the first event's probability, the second event given the first, and the third event given both earlier events.",
        true,
      ],
      [
        "The last factor is evaluated after both earlier events are taken as observed.",
        true,
      ],
      [
        "The rule can extend to more than three events by adding successive conditional factors.",
        true,
      ],
      [
        "The order may be changed if every factor is changed consistently to match the new order.",
        true,
      ],
    ],
    "The chain rule decomposes a joint event into one starting probability followed by conditional probabilities in an explicit sequence. Any event ordering can be used, but changing the order requires rewriting the conditioning sets so that multiplying the factors still reconstructs the same joint probability.",
  ),
  makeQuestion(
    "cs109-lect4-q10",
    "easy",
    "A streaming service has 50 million users; 10 million watched film \\(E\\), 8 million watched film \\(F\\), and 4 million watched both. Which statements are correct?",
    [
      ["\\(P(E)\\) is estimated as \\(10/50=0.20\\).", true],
      ["\\(P(E\\mid F)\\) is estimated as \\(4/8=0.50\\).", true],
      [
        "The total user count \\(50\\) million cancels when computing the conditional ratio from audience counts.",
        true,
      ],
      [
        "\\(P(E\\mid F)\\) must equal \\(P(E)\\) because both are based on the same service.",
        false,
      ],
    ],
    "The unconditional estimate uses all users as its reference population, while the conditional estimate uses only viewers of \\(F\\). Expressing the conditional definition through counts cancels the total population, leaving viewers of both films divided by viewers of the observed film.",
  ),
  makeQuestion(
    "cs109-lect4-q11",
    "easy",
    "Which formulas correctly decompose \\(P(E)\\) according to whether event \\(F\\) occurs?",
    [
      ["\\(P(E)=P(E\\cap F)+P(E\\cap F^c)\\).", true],
      ["\\(P(E)=P(E\\mid F)P(F)+P(E\\mid F^c)P(F^c)\\).", true],
      ["\\(P(E)=P(E\\mid F)+P(E\\mid F^c)\\).", false],
      ["\\(P(E)=P(F)P(F^c)\\).", false],
    ],
    "The cases \\(F\\) and \\(F^c\\) partition the sample space, so the portions of \\(E\\) inside those cases are disjoint and add. Applying the chain rule to each joint term supplies the weights \\(P(F)\\) and \\(P(F^c)\\), which cannot be omitted.",
  ),
  makeQuestion(
    "cs109-lect4-q12",
    "easy",
    "What property must background events \\(B_1,\\ldots,B_m\\) have to support the partition form \\(P(E)=\\sum_i P(E\\mid B_i)P(B_i)\\)?",
    [
      [
        "They must be mutually exclusive and together cover the sample space.",
        true,
      ],
      ["They must all have the same probability.", false],
      ["They must each be subsets of the target event.", false],
      ["They must be independent of the target event.", false],
    ],
    "A partition assigns every outcome to exactly one background case, allowing the disjoint pieces \\(E\\cap B_i\\) to be added. Equal probabilities and independence are unnecessary, while requiring every background event to lie inside \\(E\\) would fail to describe the rest of the sample space.",
  ),
  makeQuestion(
    "cs109-lect4-q13",
    "medium",
    "A baby has probability 0.5 of having pooped overnight. The probability of crying is 0.5 given poop and 0.125 given no poop. Which statements are correct?",
    [
      ["\\(P(\\text{poop and cry})=0.5\\times0.5=0.25\\).", true],
      ["\\(P(\\text{no poop and cry})=0.5\\times0.125=0.0625\\).", true],
      ["\\(P(\\text{cry})=0.25+0.0625=0.3125\\).", true],
      [
        "The two joint crying cases, with masses \\(0.25\\) and \\(0.0625\\), are mutually exclusive and exhaustive for crying.",
        true,
      ],
    ],
    "The chain rule computes each joint branch, and poop versus no poop forms a partition of all possibilities. Adding the two disjoint ways to cry gives the total crying probability, demonstrating why both conditional rates are needed for an unconditional answer.",
  ),
  makeQuestion(
    "cs109-lect4-q14",
    "medium",
    "Ten percent of bacteria carry a mutation. A bacterium survives with probability 0.20 if mutated and 0.01 otherwise. Which statements are correct?",
    [
      ["\\(P(\\text{survive})=0.20(0.10)+0.01(0.90)=0.029\\).", true],
      [
        "Mutated survivors contribute \\(0.20(0.10)=0.020\\) probability mass.",
        true,
      ],
      [
        "Nonmutated survivors contribute \\(0.01(0.90)=0.009\\) probability mass.",
        true,
      ],
      [
        "The survival probability is \\(0.20+0.01=0.21\\) because the two conditional rates add.",
        false,
      ],
    ],
    "Mutation status partitions the population, so each survival rate must be weighted by the prevalence of its group. The two joint contributions are 0.020 and 0.009, which add to 0.029; adding the conditional rates alone ignores how frequently each background state occurs.",
  ),
  makeQuestion(
    "cs109-lect4-q15",
    "medium",
    "Events \\(B_1,B_2,B_3\\) partition the sample space with probabilities 0.2, 0.5, and 0.3. If \\(P(E\\mid B_1)=0.1\\), \\(P(E\\mid B_2)=0.4\\), and \\(P(E\\mid B_3)=0.8\\), which calculations are correct?",
    [
      ["\\(P(E)=0.1(0.2)+0.4(0.5)+0.8(0.3)=0.46\\).", true],
      ["The \\(B_3\\) branch contributes 0.24 to \\(P(E)\\).", true],
      ["\\(P(E)=0.1+0.4+0.8=1.3\\).", false],
      [
        "The largest conditional probability alone determines \\(P(E)\\).",
        false,
      ],
    ],
    "Total probability forms a weighted average of the branch-specific event probabilities, using the probability of each branch as its weight. The third branch contributes \\(0.8\\times0.3=0.24\\), and all three contributions sum to 0.46 rather than the unweighted sum of conditional rates.",
  ),
  makeQuestion(
    "cs109-lect4-q16",
    "medium",
    "For events with positive probabilities, which expression is Bayes' theorem for reversing the conditioning direction?",
    [
      ["\\(P(F\\mid E)=\\dfrac{P(E\\mid F)P(F)}{P(E)}\\)", true],
      ["\\(P(F\\mid E)=P(E\\mid F)\\)", false],
      ["\\(P(F\\mid E)=\\dfrac{P(F)}{P(E\\mid F)}\\)", false],
      ["\\(P(F\\mid E)=P(E)P(F)\\)", false],
    ],
    "Both conditional directions share the joint probability \\(P(E\\cap F)\\). Writing that joint mass as \\(P(E\\mid F)P(F)\\) and dividing by the new evidence probability \\(P(E)\\) yields Bayes' theorem; simply swapping labels discards the required normalization.",
  ),
  makeQuestion(
    "cs109-lect4-q17",
    "medium",
    "In the Bayes expression \\(P(F\\mid E)=P(E\\mid F)P(F)/P(E)\\), which terminology matches the roles of the terms?",
    [
      [
        "\\(P(F)\\) is the prior belief about \\(F\\) before observing \\(E\\).",
        true,
      ],
      [
        "\\(P(E\\mid F)\\) is the likelihood of the evidence under \\(F\\).",
        true,
      ],
      [
        "\\(P(F\\mid E)\\) is the posterior belief after observing \\(E\\).",
        true,
      ],
      [
        "\\(P(E)\\) normalizes the posterior so probability mass is scaled correctly.",
        true,
      ],
    ],
    "Bayes' theorem updates a prior using how compatible the evidence is with the hypothesized state. The evidence probability in the denominator aggregates all ways the evidence could arise, ensuring that the resulting posterior is a properly normalized conditional probability.",
  ),
  makeQuestion(
    "cs109-lect4-q18",
    "medium",
    "Suppose 60% of email is spam, 20% of spam contains the word 'Dear,' and 1% of non-spam contains 'Dear.' Which statements are correct for an email containing 'Dear'?",
    [
      [
        "The spam-and-'Dear' joint probability is \\(0.60\\times0.20=0.12\\).",
        true,
      ],
      [
        "The non-spam-and-'Dear' joint probability is \\(0.40\\times0.01=0.004\\).",
        true,
      ],
      ["The posterior spam probability is approximately \\(0.968\\).", true],
      [
        "The posterior is \\(0.20\\) because that is the given conditional probability.",
        false,
      ],
    ],
    "The given 0.20 is the likelihood of seeing the word under spam, not the reversed probability that the message is spam after seeing the word. Bayes combines both possible sources of the evidence and normalizes the spam contribution by their total, producing a posterior near 0.968.",
  ),
  makeQuestion(
    "cs109-lect4-q19",
    "medium",
    "A disease has prevalence 0.5%. A test is positive for 98% of diseased people and 1% of people without the disease. Which expressions correctly calculate the probability of disease after a positive test?",
    [
      ["\\(\\dfrac{0.98(0.005)}{0.98(0.005)+0.01(0.995)}\\)", true],
      ["Approximately \\(0.33\\).", true],
      [
        "\\(P(D\\mid +)=0.98\\), because sensitivity is the posterior probability.",
        false,
      ],
      [
        "\\(P(D\\mid +)=0.005\\), because a test result cannot update prevalence.",
        false,
      ],
    ],
    "A positive result can arise from a true positive or from a false positive, and Bayes compares those two joint masses. The low prevalence makes the false-positive branch large enough that the posterior is about one-third, despite the test's high sensitivity.",
  ),
  makeQuestion(
    "cs109-lect4-q20",
    "medium",
    "Using the same disease model—0.5% prevalence, 98% sensitivity, and 1% false-positive rate—which value is closest to the probability of disease after a negative test?",
    [
      [
        "\\(\\dfrac{0.02(0.005)}{0.02(0.005)+0.99(0.995)}\\approx0.00010\\)",
        true,
      ],
      ["\\(\\dfrac{0.02}{0.02+0.99}\\approx0.0198\\)", false],
      ["\\(\\dfrac{0.005}{0.005+0.995}=0.005\\)", false],
      [
        "\\(\\dfrac{0.99(0.995)}{0.02(0.005)+0.99(0.995)}\\approx0.9999\\)",
        false,
      ],
    ],
    "The negative evidence occurs for 2% of diseased people and 99% of disease-free people. Weighting those likelihoods by prevalence and normalizing makes disease after a negative result roughly one in ten thousand, substantially below the original 0.5% prior.",
  ),
  makeQuestion(
    "cs109-lect4-q21",
    "medium",
    "In a population of 1,000 under the disease model, about 5 people have the disease. Which statements correctly explain the positive-test posterior using expected counts?",
    [
      ["About \\(0.98(5)=4.9\\) diseased people test positive.", true],
      ["About \\(0.01(995)=9.95\\) disease-free people test positive.", true],
      [
        "Among positive tests, the diseased fraction is about \\(4.9/(4.9+9.95)\\).",
        true,
      ],
      [
        "False positives can outnumber true positives even with a \\(1\\%\\) false-positive rate.",
        true,
      ],
    ],
    "Expected-count reasoning is the same Bayes calculation with a concrete population scale. Because the disease-free group is so much larger, one percent of it yields more positive results than 98 percent of the small diseased group, leaving roughly one-third of positive tests as true cases.",
  ),
  makeQuestion(
    "cs109-lect4-q22",
    "medium",
    "Which statements correctly interpret the two directions in a diagnostic test?",
    [
      [
        "Sensitivity measures the positive-test rate among people with disease.",
        true,
      ],
      [
        "Positive predictive value measures disease probability among people who tested positive.",
        true,
      ],
      [
        "Bayes' theorem connects the two directions using prevalence and the overall positive rate.",
        true,
      ],
      [
        "High sensitivity alone guarantees a high disease probability after a positive test.",
        false,
      ],
    ],
    "Sensitivity conditions on the true disease state, while the posterior conditions on the observed test result. Reversing that direction requires the disease prior and competing positive-test likelihoods; with low prevalence, even a sensitive test can have a moderate positive predictive value.",
  ),
  makeQuestion(
    "cs109-lect4-q23",
    "medium",
    "Which facts are sufficient to compute \\(P(F\\mid E)\\) by the expanded two-case form of Bayes' theorem?",
    [
      [
        "The prior for the state, the evidence likelihood under that state, and the evidence likelihood under its complement.",
        true,
      ],
      ["The complement prior, obtained as one minus the state prior.", true],
      [
        "Only the forward evidence likelihood, because the conditioning direction can be reversed directly.",
        false,
      ],
      [
        "Only the evidence probability and the state prior, because joint behavior is unnecessary.",
        false,
      ],
    ],
    "The numerator needs the joint contribution from \\(F\\), while the denominator needs every way the evidence can occur. The stated prior and two likelihoods supply both branches, with the complementary prior derived from \\(P(F)\\); a single likelihood cannot account for competing explanations of the evidence.",
  ),
  makeQuestion(
    "cs109-lect4-q24",
    "medium",
    "A classifier flags 8% of inputs. Among truly harmful inputs it flags 90%, and harmful inputs have prevalence 2%. What additional quantity is needed to determine \\(P(\\text{harmful}\\mid\\text{flag})\\) directly from Bayes' theorem?",
    [
      ["No additional quantity; \\(0.90(0.02)/0.08\\) is sufficient.", true],
      [
        "The value \\(0.10(0.02)/0.08\\), using the no-flag rate among harmful inputs.",
        false,
      ],
      [
        "The ratio \\(0.90(0.08)/0.02\\), using the flag rate as though it were the prior.",
        false,
      ],
      [
        "The ratio \\(0.08/[0.90(0.02)]\\), which reverses the Bayes numerator and denominator.",
        false,
      ],
    ],
    "Bayes' compact form needs the likelihood \\(P(\\text{flag}\\mid\\text{harmful})=0.90\\), the prior 0.02, and the evidence probability \\(P(\\text{flag})=0.08\\), all of which are given. The posterior is therefore \\(0.90\\times0.02/0.08=0.225\\).",
  ),
  makeQuestion(
    "cs109-lect4-q25",
    "hard",
    "Locations \\(L_1,L_2,L_3\\) have priors 0.2, 0.5, and 0.3. A satellite observation \\(O\\) has likelihoods 0.1, 0.4, and 0.2 at those locations. Which statements correctly compute \\(P(L_2\\mid O)\\)?",
    [
      [
        "The unnormalized weight for \\(L_2\\) is \\(0.5\\times0.4=0.20\\).",
        true,
      ],
      [
        "The evidence probability is \\(0.2(0.1)+0.5(0.4)+0.3(0.2)=0.28\\).",
        true,
      ],
      ["The posterior is \\(0.20/0.28=5/7\\).", true],
      [
        "All three posterior probabilities follow by normalizing their weights so \\(\\sum_i P(L_i\\mid O)=1\\).",
        true,
      ],
    ],
    "For mutually exclusive candidate locations, total probability sums the likelihood-weighted priors to obtain the observation probability. Bayes then divides the weight for the chosen location by that total, and applying the same normalization to every location produces posteriors that sum to 1.",
  ),
  makeQuestion(
    "cs109-lect4-q26",
    "hard",
    "Suppose \\(P(A)=0.5\\), \\(P(B\\mid A)=0.4\\), and \\(P(C\\mid A,B)=0.3\\). Which statements correctly use the chain rule?",
    [
      ["\\(P(A\\cap B)=0.5\\times0.4=0.20\\).", true],
      ["\\(P(A\\cap B\\cap C)=0.5\\times0.4\\times0.3=0.06\\).", true],
      [
        "The factor 0.3 is conditioned on both earlier events, not on \\(B\\) alone.",
        true,
      ],
      ["\\(P(C)=0.3\\) follows from the given conditional probability.", false],
    ],
    "The generalized chain rule multiplies probabilities along a sequence in which each new factor is conditioned on everything already included. The value 0.3 describes \\(C\\) only inside the joint world \\(A\\cap B\\), so it does not determine the unconditional probability of \\(C\\).",
  ),
  makeQuestion(
    "cs109-lect4-q27",
    "hard",
    "An event \\(E\\) has \\(P(E)=0.4\\), and an observation \\(F\\) has \\(P(F)=0.25\\) and \\(P(E\\cap F)=0.15\\). Which statements are correct?",
    [
      ["\\(P(E\\mid F)=0.15/0.25=0.60\\).", true],
      ["\\(P(F\\mid E)=0.15/0.40=0.375\\).", true],
      [
        "Observing \\(F\\) raises the probability of \\(E\\) from 0.4 to 0.6.",
        true,
      ],
      [
        "The conditionals use denominators \\(P(F)=0.25\\) and \\(P(E)=0.40\\), so their normalization events differ.",
        true,
      ],
    ],
    "Both conditionals start from the same intersection mass but divide by the probability of the event being taken as known. Comparing the conditional 0.60 with the prior 0.40 shows how the observation changes belief, while the reverse conditional answers a different population question.",
  ),
  makeQuestion(
    "cs109-lect4-q28",
    "hard",
    "Two independent fair dice are rolled. Given that their sum is 7, what is the probability that the first die is at most 2?",
    [
      [
        "\\(2/6=1/3\\), because two of the six conditioned outcomes have first die at most 2.",
        true,
      ],
      [
        "\\(2/36=1/18\\), because the original sample space must remain in the denominator.",
        false,
      ],
      [
        "\\(1/2\\), because the first die is either at most 2 or greater than 2.",
        false,
      ],
      [
        "\\(1/6\\), because only one value of the first die is possible after conditioning.",
        false,
      ],
    ],
    "Conditioning on sum 7 restricts the sample space to six equally likely ordered pairs. Two of those pairs have first value 1 or 2, so the conditional probability is \\(2/6\\); retaining all 36 original pairs would fail to condition on the observed sum.",
  ),
  makeQuestion(
    "cs109-lect4-q29",
    "hard",
    "Before a four-option question, 75% of learners know the concept. A knowledgeable learner answers correctly with probability 0.9, while a learner who does not know guesses correctly with probability 0.25. Which statements are correct after observing a correct answer?",
    [
      ["The know-and-correct mass is \\(0.75\\times0.9=0.675\\).", true],
      ["The not-know-and-correct mass is \\(0.25\\times0.25=0.0625\\).", true],
      [
        "The posterior knowledge probability is \\(0.675/(0.675+0.0625)\\approx0.915\\).",
        true,
      ],
      [
        "The posterior is exactly \\(0.9\\) because that is the knowledgeable accuracy.",
        false,
      ],
    ],
    "A correct response has two possible sources: genuine knowledge followed by success, or lack of knowledge followed by a lucky guess. Bayes normalizes the first joint mass by their sum, giving about 91.5%; the 0.9 likelihood is not itself the probability of knowledge after seeing the answer.",
  ),
  makeQuestion(
    "cs109-lect4-q30",
    "hard",
    "A search area has four possible cells with priors \\((0.1,0.2,0.3,0.4)\\). An observation has likelihoods \\((0.6,0.3,0.2,0.1)\\) in those cells. Which statements correctly describe the posterior?",
    [
      ["The unnormalized weights are \\((0.06,0.06,0.06,0.04)\\).", true],
      ["The evidence probability is \\(0.22\\).", true],
      [
        "The first three cells have equal posterior probability \\(0.06/0.18\\) because the fourth cell can be omitted from normalization.",
        false,
      ],
      [
        "The fourth cell remains most probable because its prior was \\(0.4\\).",
        false,
      ],
    ],
    "Bayesian updating balances prior plausibility with how well each location predicts the observation. The first three prior-times-likelihood products tie, while the fourth falls to a smaller weight despite its larger prior; normalizing all four weights by 0.22 gives the posterior distribution.",
  ),
  makeQuestion(
    "cs109-lect4-q31",
    "hard",
    "Which statements are required for the multi-case total-probability identity \\(P(E)=\\sum_i P(E\\mid B_i)P(B_i)\\)?",
    [
      ["The background events must cover every possible outcome.", true],
      ["No outcome may belong to two different background events.", true],
      [
        "Each branch's conditional factor by itself represents the branch's joint mass.",
        false,
      ],
      [
        "Every branch-specific conditional probability must have the same value.",
        false,
      ],
    ],
    "Coverage and mutual exclusivity make the background events a partition, so the joint pieces of \\(E\\) neither omit nor double-count probability mass. A full term is the product \\(P(E\\mid B_i)P(B_i)=P(E\\cap B_i)\\); the conditional factor alone is not the joint mass and may vary by branch.",
  ),
  makeQuestion(
    "cs109-lect4-q32",
    "hard",
    "A population has \\(P(F)=0.3\\), \\(P(E)=0.5\\), and \\(P(E\\mid F)=0.8\\). What is \\(P(E\\mid F^c)\\)?",
    [
      ["\\((0.5-0.8(0.3))/0.7\\approx0.3714\\)", true],
      ["\\(1-0.8=0.2\\)", false],
      ["\\(0.5/0.7\\approx0.7143\\)", false],
      ["\\(0.8(0.3)=0.24\\)", false],
    ],
    "Total probability gives \\(0.5=0.8(0.3)+P(E\\mid F^c)(0.7)\\). Subtracting the known joint contribution and dividing by the complement prior yields about 0.3714; taking a complement of 0.8 would instead calculate \\(P(E^c\\mid F)\\).",
  ),
  makeQuestion(
    "cs109-lect4-q33",
    "hard",
    "A test has fixed sensitivity 0.95 and false-positive rate 0.05. Which statements correctly predict how \\(P(D\\mid +)\\) changes as disease prevalence \\(P(D)\\) increases?",
    [
      ["The positive-test posterior increases.", true],
      [
        "The false-positive joint mass grows relative to the true-positive joint mass.",
        false,
      ],
      [
        "At very low prevalence, true positives must dominate positive results.",
        false,
      ],
      [
        "Sensitivity alone remains insufficient to determine the posterior without a prevalence assumption.",
        true,
      ],
    ],
    "Bayes compares \\(0.95P(D)\\) with \\(0.05(1-P(D))\\). Raising prevalence makes the true-positive contribution grow relative to the false-positive contribution, so the posterior rises; at very low prevalence the false-positive branch can dominate, and sensitivity alone cannot determine the balance.",
  ),
  makeQuestion(
    "cs109-lect4-q34",
    "hard",
    "For events with \\(P(F)>0\\), which statements correctly distinguish complements under conditioning?",
    [
      ["\\(P(E^c\\mid F)=1-P(E\\mid F)\\).", true],
      [
        "\\(P(E\\mid F^c)\\) concerns a different conditioned world from \\(P(E^c\\mid F)\\).",
        true,
      ],
      [
        "Knowing \\(P(E\\mid F)\\) alone does not determine \\(P(E\\mid F^c)\\).",
        true,
      ],
      ["\\(P(E\\mid F^c)=1-P(E\\mid F)\\) for all events.", false],
    ],
    "Within the fixed world \\(F\\), the events \\(E\\) and \\(E^c\\) are complements and their conditional probabilities sum to 1. Switching the event after the bar changes the reference population entirely, so the corresponding rate needs separate information rather than a simple complement.",
  ),
  makeQuestion(
    "cs109-lect4-q35",
    "hard",
    "Which identities correctly connect conditional probability, the chain rule, total probability, and Bayes' theorem?",
    [
      ["\\(P(E\\mid F)=P(E\\cap F)/P(F)\\) when \\(P(F)>0\\).", true],
      ["\\(P(E\\cap F)=P(F)P(E\\mid F)=P(E)P(F\\mid E)\\).", true],
      ["\\(P(E)=P(E\\mid F)P(F)+P(E\\mid F^c)P(F^c)\\).", true],
      [
        "\\(P(F\\mid E)=P(E\\mid F)P(F)/P(E)\\) when the denominators are positive.",
        true,
      ],
    ],
    "These identities form one connected toolkit: conditioning normalizes a joint event, the chain rule reconstructs that joint event, total probability expands an evidence probability across a partition, and Bayes equates the two chain-rule factorizations to reverse a conditional.",
  ),
];
