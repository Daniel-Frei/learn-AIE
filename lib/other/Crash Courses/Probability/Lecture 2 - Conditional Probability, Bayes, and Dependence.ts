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
    chapter: 2,
    difficulty,
    prompt,
    options: options.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

export const CrashCourseProbabilityL2Questions: Question[] = [
  // Conditional and joint probability
  makeQuestion(
    "crash-probability-l2-q61",
    "easy",
    "What does \\(P(A\\mid B)\\) measure when \\(P(B)>0\\)?",
    [
      ["The fraction of probability inside B that also belongs to A", true],
      ["The probability that A causes B", false],
      ["The probability of A and B added together", false],
      ["The probability of B after A has been ruled out", false],
    ],
    "Conditioning restricts attention to the outcomes in B and asks what share of that remaining probability also lies in A. It is an informational update, not by itself a causal claim, an addition rule, or a complement operation.",
  ),
  makeQuestion(
    "crash-probability-l2-q62",
    "easy",
    "A card is drawn uniformly from a standard deck. Given that it is a face card, which calculations are correct?",
    [
      ["The conditioned sample space contains 12 face cards.", true],
      ["\\(P(\\text{king}\\mid\\text{face card})=4/12=1/3\\).", true],
      [
        "The denominator remains 52 because conditioning does not change the relevant universe.",
        false,
      ],
      ["The answer is \\(P(\\text{face card}\\mid\\text{king})\\).", false],
    ],
    "Once face card is known, the relevant equally likely outcomes are the twelve jacks, queens, and kings, four of which are kings. Keeping 52 or reversing the conditional answers a different probability question.",
  ),
  makeQuestion(
    "crash-probability-l2-q63",
    "medium",
    "Among 200 emails, 50 are spam, 80 contain the word “free,” and 40 are both spam and contain “free.” Which statements are correct?",
    [
      ["\\(P(\\text{spam}\\cap\\text{free})=40/200=0.20\\).", true],
      ["\\(P(\\text{spam}\\mid\\text{free})=40/80=0.50\\).", true],
      ["\\(P(\\text{free}\\mid\\text{spam})=40/50=0.80\\).", true],
      [
        "The two conditional probabilities must be equal because they use the same intersection.",
        false,
      ],
    ],
    "Both conditionals have the same joint numerator but different conditioned populations, producing 0.50 and 0.80. The joint probability uses all 200 emails as its denominator, so it is distinct from either conditional probability.",
  ),
  makeQuestion(
    "crash-probability-l2-q64",
    "hard",
    "Suppose \\(P(A)=0.30\\), \\(P(B)=0.50\\), and \\(P(A\\cap B)=0.18\\). Which conditional probabilities are correct?",
    [
      ["\\(P(A\\mid B)=0.18/0.50=0.36\\).", true],
      ["\\(P(B\\mid A)=0.18/0.30=0.60\\).", true],
      [
        "\\(P(A\\mid B)=P(B\\mid A)\\) because both concern the same events.",
        false,
      ],
      ["\\(P(A\\mid B)=0.18/0.30\\) because A is written first.", false],
    ],
    "The event after the conditioning bar supplies the denominator, so the two directions divide the same intersection by different marginal probabilities. Reversing the denominator is the classic inverse-probability error.",
  ),
  makeQuestion(
    "crash-probability-l2-q65",
    "medium",
    "Which statements correctly connect conditional and joint probability when the required denominators are positive?",
    [
      ["\\(P(A\\mid B)=P(A\\cap B)/P(B)\\).", true],
      ["\\(P(A\\cap B)=P(B)P(A\\mid B)\\).", true],
      ["\\(P(A\\cap B)=P(A)P(B\\mid A)\\).", true],
      [
        "The two multiplication forms are equal because they describe the same intersection.",
        true,
      ],
    ],
    "Conditional probability and the multiplication rule are algebraic rearrangements of the same relationship. Either event can be taken first, provided the second factor is conditioned on the event used in the first factor.",
  ),
  makeQuestion(
    "crash-probability-l2-q66",
    "hard",
    "A user opens an app with probability 0.60, and among users who open it, 25% purchase. Which conclusions are correct?",
    [
      ["\\(P(\\text{open}\\cap\\text{purchase})=0.60(0.25)=0.15\\).", true],
      [
        "The multiplication uses \\(P(\\text{purchase}\\mid\\text{open})\\), not the unconditional purchase rate.",
        true,
      ],
      [
        "Among 1,000 comparable users, about 150 are expected to both open and purchase.",
        true,
      ],
      [
        "The joint rate is 0.85 because open and purchase are consecutive steps.",
        false,
      ],
    ],
    "A path probability multiplies the probability of reaching the open group by the conditional purchase rate within that group. Adding the two rates ignores that purchase is measured inside a restricted population and cannot produce the joint event.",
  ),
  makeQuestion(
    "crash-probability-l2-q67",
    "easy",
    "In a dataset, 30 of 120 images contain a bicycle, and 18 of those 30 also contain a helmet. What is \\(P(\\text{helmet}\\mid\\text{bicycle})\\)?",
    [
      ["\\(18/30=0.60\\)", true],
      ["\\(18/120=0.15\\)", false],
      ["\\(30/120=0.25\\)", false],
      ["\\(30/18\\approx1.67\\)", false],
    ],
    "The bicycle condition restricts the denominator to the 30 bicycle images, of which 18 also have helmets. Dividing by all 120 gives a joint probability, while the other ratios answer marginal or inverted questions.",
  ),
  makeQuestion(
    "crash-probability-l2-q68",
    "medium",
    "For three sequential tokens, which factorizations of their joint probability are valid?",
    [
      ["\\(P(x_1,x_2,x_3)=P(x_1)P(x_2\\mid x_1)P(x_3\\mid x_1,x_2)\\).", true],
      [
        "The chain rule remains valid without assuming token independence.",
        true,
      ],
      [
        "\\(P(x_1,x_2,x_3)=P(x_1)P(x_2)P(x_3)\\) for every language model.",
        false,
      ],
      [
        "Conditioning on previous tokens can be dropped because all tokens share one vocabulary.",
        false,
      ],
    ],
    "The chain rule decomposes a joint sequence probability into conditional next-step factors without an independence assumption. Replacing those factors by marginals would assert that context conveys no information, which contradicts the point of language modeling.",
  ),
  makeQuestion(
    "crash-probability-l2-q69",
    "hard",
    "A joint table gives \\(P(X=0,Y=0)=0.15\\), \\(P(0,1)=0.25\\), \\(P(1,0)=0.10\\), and \\(P(1,1)=0.50\\). Which statements are correct?",
    [
      ["\\(P(X=1)=0.60\\).", true],
      ["\\(P(Y=1)=0.75\\).", true],
      ["\\(P(X=1\\mid Y=1)=0.50/0.75=2/3\\).", true],
      [
        "\\(P(X=1\\mid Y=1)=0.50\\) because the joint cell is already conditional.",
        false,
      ],
    ],
    "Row or column sums give the marginals, and conditioning renormalizes the relevant column to total one. A joint cell is measured against the full population, so it must be divided by \\(P(Y=1)\\) to become a conditional probability.",
  ),
  makeQuestion(
    "crash-probability-l2-q70",
    "easy",
    "Which interpretations of joint and conditional probability are correct?",
    [
      ["\\(P(A\\cap B)\\) measures both events occurring.", true],
      [
        "\\(P(A\\mid B)\\) measures A inside the restricted B population.",
        true,
      ],
      ["A joint table can be summed to obtain marginal probabilities.", true],
      [
        "A conditional distribution over all A outcomes sums to one for a fixed B.",
        true,
      ],
    ],
    "Joint probability locates overlap in the full population, while conditional probability rescales a selected slice. Summing a joint table removes an unwanted variable, and renormalization makes each well-defined conditional distribution total one.",
  ),
  makeQuestion(
    "crash-probability-l2-q71",
    "medium",
    "Which statements about conditioning on a zero-probability event are correct in the elementary formula \\(P(A\\mid B)=P(A\\cap B)/P(B)\\)?",
    [
      ["The ratio is undefined when \\(P(B)=0\\).", true],
      [
        "One cannot repair the ratio by declaring its denominator to be one.",
        true,
      ],
      [
        "The conditional must equal zero because the numerator is also zero.",
        false,
      ],
      [
        "The conditional must equal \\(P(A)\\) because an impossible event carries no information.",
        false,
      ],
    ],
    "The elementary definition requires division by a positive conditioned-event probability, and zero divided by zero has no determined value. More advanced continuous conditioning uses additional machinery, but neither zero nor the marginal follows from this undefined ratio.",
  ),
  makeQuestion(
    "crash-probability-l2-q72",
    "hard",
    "An LLM assigns \\(P(\\text{bank}\\mid\\text{river context})=0.30\\) and \\(P(\\text{bank}\\mid\\text{finance context})=0.70\\). Which statements are correct?",
    [
      [
        "The same token outcome can have different probabilities under different conditions.",
        true,
      ],
      ["The context changes the conditional distribution being queried.", true],
      [
        "These two values alone do not determine the unconditional probability of bank without context frequencies.",
        true,
      ],
      [
        "The token bank is independent of context because its spelling is unchanged.",
        false,
      ],
    ],
    "Conditional probabilities depend on which contextual population is considered, even when the token string is identical. An unconditional probability would average across context types using their frequencies, and the differing conditionals are direct evidence of dependence.",
  ),

  // Marginalization and the law of total probability
  makeQuestion(
    "crash-probability-l2-q73",
    "easy",
    "A joint distribution has \\(P(X=1,Y=0)=0.20\\) and \\(P(X=1,Y=1)=0.35\\). What is \\(P(X=1)\\)?",
    [
      ["\\(0.20+0.35=0.55\\)", true],
      ["\\(0.35-0.20=0.15\\)", false],
      ["\\(0.20\\times0.35=0.07\\)", false],
      ["\\(0.35/0.20=1.75\\)", false],
    ],
    "The two Y values describe mutually exclusive ways for X to equal one, so marginalization adds their joint masses. Subtraction, multiplication, and division represent different relationships and do not remove Y from this joint distribution.",
  ),
  makeQuestion(
    "crash-probability-l2-q74",
    "easy",
    "A factory uses machine A for 70% of items and machine B for 30%. Their defect rates are 2% and 5%. Which path probabilities are correct?",
    [
      ["\\(P(A\\cap\\text{defect})=0.70(0.02)=0.014\\).", true],
      ["\\(P(B\\cap\\text{defect})=0.30(0.05)=0.015\\).", true],
      ["The total defect rate is \\(0.02+0.05=0.07\\).", false],
      [
        "Machine usage rates are irrelevant because the defect rates are conditional.",
        false,
      ],
    ],
    "Each joint path multiplies the machine's population share by its within-machine defect rate. Adding raw conditional rates would weight the small and large production streams equally, so the usage probabilities are essential.",
  ),
  makeQuestion(
    "crash-probability-l2-q75",
    "medium",
    "Continuing with machine A producing 70% at 2% defects and B producing 30% at 5% defects, which conclusions are correct?",
    [
      ["\\(P(\\text{defect})=0.014+0.015=0.029\\).", true],
      [
        "The calculation adds mutually exclusive A-defect and B-defect paths.",
        true,
      ],
      ["The expected defective count is \\(1000(0.029)=29\\).", true],
      [
        "The total defect probability is \\((0.02+0.05)/2=0.035\\), the unweighted average of the machine rates.",
        false,
      ],
    ],
    "The machine events form a partition, so the law of total probability adds their weighted defect paths to obtain 2.9%. An unweighted average would apply only if the machines produced equal shares, which they do not.",
  ),
  makeQuestion(
    "crash-probability-l2-q76",
    "hard",
    "A request is routed to region E with probability 0.50, W with 0.30, or C with 0.20. Timeout rates are 0.01, 0.04, and 0.03 respectively. Which calculations are correct?",
    [
      ["The E timeout path has probability \\(0.50(0.01)=0.005\\).", true],
      ["The total timeout probability is \\(0.005+0.012+0.006=0.023\\).", true],
      ["The total timeout probability is \\(0.01+0.04+0.03=0.08\\).", false],
      ["The region paths overlap because every path ends in timeout.", false],
    ],
    "Routing regions are mutually exclusive causes of a request's path, so each conditional timeout rate must be weighted and the joint paths can then be added. Sharing the same final event does not make the preceding region paths overlap.",
  ),
  makeQuestion(
    "crash-probability-l2-q77",
    "medium",
    "For \\(B_1,\\ldots,B_n\\) to support \\(P(A)=\\sum_iP(A\\mid B_i)P(B_i)\\), which statements are required or useful?",
    [
      ["The \\(B_i\\) events cover the relevant sample space.", true],
      ["Distinct \\(B_i\\) events do not overlap.", true],
      ["Each term is the joint probability \\(P(A\\cap B_i)\\).", true],
      ["Adding the terms counts every way A occurs exactly once.", true],
    ],
    "An exhaustive, mutually exclusive partition breaks A into disjoint paths, and each weighted conditional equals the joint mass on one path. Because every outcome belongs to one partition cell, summing includes all of A without double-counting.",
  ),
  makeQuestion(
    "crash-probability-l2-q78",
    "hard",
    "A latent topic \\(Z\\) is sports with probability 0.40 and finance with 0.60. A model emits “score” with probabilities 0.50 and 0.10 under those topics. Which statements are correct?",
    [
      ["The sports-and-score path is \\(0.40(0.50)=0.20\\).", true],
      ["The finance-and-score path is \\(0.60(0.10)=0.06\\).", true],
      ["The marginal \\(P(\\text{score})=0.26\\).", true],
      [
        "The marginal is 0.60 because the larger topic prior determines the output.",
        false,
      ],
    ],
    "Marginalizing the hidden topic adds the weighted paths \\(0.40(0.50)+0.60(0.10)=0.26\\). The most common topic does not by itself determine the observed-token probability because likelihood under each topic also matters.",
  ),
  makeQuestion(
    "crash-probability-l2-q79",
    "easy",
    "A user is on mobile with probability 0.65 and desktop with probability 0.35. Conversion rates are 0.08 and 0.12. What is the overall conversion probability?",
    [
      ["\\(0.65(0.08)+0.35(0.12)=0.094\\)", true],
      ["\\(0.08+0.12=0.20\\)", false],
      ["\\((0.08+0.12)/2=0.10\\)", false],
      ["\\(0.65(0.35)=0.2275\\)", false],
    ],
    "The device types partition users, so their conditional conversion rates receive weights equal to device prevalence. The result is 9.4%; raw addition, equal averaging, or multiplying device shares does not describe conversion paths.",
  ),
  makeQuestion(
    "crash-probability-l2-q80",
    "medium",
    "A bag is chosen uniformly from two bags. Bag 1 has 3 red and 1 blue marble; Bag 2 has 1 red and 3 blue. Which statements are correct?",
    [
      ["\\(P(\\text{red})=0.5(3/4)+0.5(1/4)=0.5\\).", true],
      [
        "The red paths through the two bags are mutually exclusive and can be added.",
        true,
      ],
      ["\\(P(\\text{red})=3/4\\) because Bag 1 has more red marbles.", false],
      [
        "The bag-selection probability can be omitted because a bag is chosen before the marble.",
        false,
      ],
    ],
    "The experiment has two stages, so each red path includes both the bag choice and the conditional marble draw. The equally likely bags make the opposing compositions balance, and omitting the first-stage probability doubles the total mass.",
  ),
  makeQuestion(
    "crash-probability-l2-q81",
    "hard",
    "A symptom occurs in 80% of infected patients and 10% of uninfected patients; prevalence is 5%. Which statements correctly compute the marginal symptom rate?",
    [
      ["The infected-symptom path is \\(0.05(0.80)=0.04\\).", true],
      ["The uninfected-symptom path is \\(0.95(0.10)=0.095\\).", true],
      ["\\(P(\\text{symptom})=0.135\\).", true],
      [
        "The symptom rate is 0.80 because sensitivity is the probability of infection after the symptom.",
        false,
      ],
    ],
    "The symptom can arise through infected and uninfected paths, which are disjoint and sum to 13.5%. Sensitivity is \\(P(\\text{symptom}\\mid\\text{infected})\\), not the marginal symptom rate or the reversed diagnostic probability.",
  ),
  makeQuestion(
    "crash-probability-l2-q82",
    "easy",
    "Which statements correctly describe marginalization?",
    [
      [
        "It removes an unwanted discrete variable through \\(P(X)=\\sum_zP(X,z)\\).",
        true,
      ],
      ["It can turn a joint distribution \\(P(X,Z)\\) into \\(P(X)\\).", true],
      [
        "It accounts for multiple hidden paths that can produce the same observation.",
        true,
      ],
      [
        "For a continuous latent, the analogous operation is \\(p(x)=\\int p(x,z)\\,dz\\).",
        true,
      ],
    ],
    "Marginalization aggregates all mutually compatible values of the variable being removed while retaining the observed quantity of interest. Sums handle discrete alternatives and integrals handle continuous ones, but the path-aggregation idea is the same.",
  ),
  makeQuestion(
    "crash-probability-l2-q83",
    "medium",
    "A calculation applies the law of total probability using events “young” and “employed.” Which concerns are valid?",
    [
      [
        "The events overlap, so their paths can double-count people who are both young and employed.",
        true,
      ],
      [
        "The events may fail to cover people who are neither young nor employed.",
        true,
      ],
      [
        "The calculation is valid merely because both event probabilities are known.",
        false,
      ],
      [
        "Conditional rates can be added without weights whenever event names differ.",
        false,
      ],
    ],
    "A total-probability decomposition needs a mutually exclusive and exhaustive partition, which these two attributes do not provide. Knowing marginal probabilities does not fix overlap or uncovered cases, and weighted joint paths—not raw conditionals—must be added.",
  ),
  makeQuestion(
    "crash-probability-l2-q84",
    "hard",
    "A three-class latent cause \\(Z\\) has priors \\((0.2,0.5,0.3)\\), and \\(P(D\\mid Z)=(0.9,0.4,0.1)\\). Which statements are correct?",
    [
      [
        "The three D path masses are \\(0.2(0.9)=0.18\\), \\(0.5(0.4)=0.20\\), and \\(0.3(0.1)=0.03\\).",
        true,
      ],
      ["\\(P(D)=0.41\\).", true],
      [
        "The middle cause contributes the most D mass despite not having the largest conditional likelihood.",
        true,
      ],
      [
        "The first cause must be the most common posterior cause because 0.9 is the largest likelihood.",
        false,
      ],
    ],
    "Posterior evidence depends on prior times likelihood, so the path masses—not likelihoods alone—determine contributions to D. The middle cause contributes 0.20, slightly above the first cause's 0.18, and all paths sum to the evidence probability 0.41.",
  ),

  // Independence, dependence, and causal caution
  makeQuestion(
    "crash-probability-l2-q85",
    "easy",
    "Which equation expresses independence of events A and B?",
    [
      ["\\(P(A\\cap B)=P(A)P(B)\\)", true],
      ["\\(P(A\\cup B)=P(A)+P(B)\\)", false],
      ["\\(P(A\\mid B)=P(B\\mid A)\\)", false],
      ["\\(P(A)+P(B)=1\\)", false],
    ],
    "Independence means the joint probability factorizes into the product of marginals. Direct union addition characterizes disjointness, equality of reversed conditionals is not sufficient, and probabilities summing to one suggests a complement relationship instead.",
  ),
  makeQuestion(
    "crash-probability-l2-q86",
    "easy",
    "When \\(P(A)>0\\) and \\(P(B)>0\\), which checks are each equivalent to independence?",
    [
      ["\\(P(A\\mid B)=P(A)\\).", true],
      ["\\(P(B\\mid A)=P(B)\\).", true],
      ["\\(P(A\\mid B)=1-P(A)\\).", false],
      ["\\(P(A\\cap B)=0\\).", false],
    ],
    "If learning either event leaves the probability of the other unchanged, the events are independent and the joint factorizes. A zero intersection instead describes disjoint events, which are usually dependent when both have positive probability.",
  ),
  makeQuestion(
    "crash-probability-l2-q87",
    "medium",
    "Suppose \\(P(A)=0.4\\), \\(P(B)=0.5\\), and \\(P(A\\cap B)=0.2\\). Which statements are correct?",
    [
      ["\\(P(A)P(B)=0.2\\).", true],
      ["A and B satisfy the product test for independence.", true],
      ["\\(P(A\\mid B)=0.4=P(A)\\).", true],
      [
        "A and B are disjoint because their joint probability equals their product.",
        false,
      ],
    ],
    "The joint equals the product of marginals, and conditioning on B leaves A at 0.4, so both equivalent tests establish independence. Their positive intersection means they can co-occur and therefore are not disjoint.",
  ),
  makeQuestion(
    "crash-probability-l2-q88",
    "hard",
    "Two cards are drawn from a 52-card deck without replacement. Which statements are correct about the events that the first and second cards are aces?",
    [
      ["\\(P(\\text{second ace}\\mid\\text{first ace})=3/51\\).", true],
      [
        "The two ace events are dependent because the first draw changes the second draw's probability.",
        true,
      ],
      [
        "The events are independent because both marginal ace probabilities equal \\(4/52\\).",
        false,
      ],
      ["Their joint probability is \\((4/52)^2\\).", false],
    ],
    "Without replacement, observing an ace leaves three aces among 51 cards and lowers the second conditional probability. Equal marginals at the two positions do not imply independence; the joint must use \\((4/52)(3/51)\\).",
  ),
  makeQuestion(
    "crash-probability-l2-q89",
    "medium",
    "Which statements correctly describe dependence in data and AI?",
    [
      [
        "Word order creates statistical dependence among tokens in a sentence.",
        true,
      ],
      [
        "Nearby pixels can be dependent because they often belong to the same object.",
        true,
      ],
      [
        "Dependence can make one variable informative for predicting another.",
        true,
      ],
      [
        "Observed dependence alone does not identify the direction or existence of a causal effect.",
        true,
      ],
    ],
    "Machine learning exploits statistical structure such as related words, pixels, symptoms, or actions and next states. Predictive dependence is not a causal proof because reverse direction, shared causes, or selection mechanisms can create the same association.",
  ),
  makeQuestion(
    "crash-probability-l2-q90",
    "hard",
    "Weather \\(W\\) affects both umbrella use \\(U\\) and wet pavement \\(P\\). Suppose U and P are dependent marginally but independent after conditioning on W. Which statements are correct?",
    [
      [
        "Weather is a common factor that can induce marginal association between U and P.",
        true,
      ],
      [
        "Conditional independence can be written \\(P(U,P\\mid W)=P(U\\mid W)P(P\\mid W)\\).",
        true,
      ],
      [
        "Marginal dependence does not prove umbrellas cause wet pavement.",
        true,
      ],
      [
        "Conditional factorization implies \\(P(U,P)=P(U)P(P)\\) after weather is marginalized.",
        false,
      ],
    ],
    "Mixing weather conditions associates umbrella use with wet pavement even when they separate within each weather stratum. Conditioning can remove the shared-factor path, but that within-stratum factorization does not force the mixed marginal distribution to factorize.",
  ),
  makeQuestion(
    "crash-probability-l2-q91",
    "easy",
    "A fair coin is flipped twice. What is the relationship between the first-flip-head and second-flip-head events?",
    [
      ["They are independent.", true],
      ["They are mutually exclusive.", false],
      ["They are complements.", false],
      ["They are conditionally impossible.", false],
    ],
    "The first result does not change the fair probability of the second, so the joint head-head probability is \\(1/4=(1/2)(1/2)\\). Both heads can occur together, so the events are neither disjoint nor complements.",
  ),
  makeQuestion(
    "crash-probability-l2-q92",
    "medium",
    "Two cards are drawn with replacement and the deck is reshuffled after the first draw. Which statements are correct about drawing an ace at each position?",
    [
      [
        "The second ace probability remains \\(4/52\\) after any first draw.",
        true,
      ],
      [
        "The ace events factorize because replacement restores the deck composition.",
        true,
      ],
      [
        "The second ace probability becomes \\(3/51\\) after a first ace.",
        false,
      ],
      [
        "Replacement makes the events disjoint because the first card is returned.",
        false,
      ],
    ],
    "Replacement plus reshuffling restores the original probability distribution before the second draw, giving independence. The \\(3/51\\) calculation belongs to no replacement, and independence permits both events rather than making them disjoint.",
  ),
  makeQuestion(
    "crash-probability-l2-q93",
    "hard",
    "A dataset shows ice-cream sales and sunburn cases increasing together. Which conclusions are statistically responsible?",
    [
      ["The variables are associated in the observed data.", true],
      ["Temperature or sunny weather is a plausible common cause.", true],
      ["Conditioning on weather could change or remove the association.", true],
      [
        "The association establishes that buying ice cream causes sunburn.",
        false,
      ],
    ],
    "Dependence supports prediction but does not determine a causal arrow, and a shared seasonal factor can increase both variables. Stratifying by weather tests whether the marginal association was induced by mixing conditions rather than by one measured variable causing the other.",
  ),
  makeQuestion(
    "crash-probability-l2-q94",
    "easy",
    "For independent events A and B with positive probabilities, which statements are correct?",
    [
      ["\\(P(A\\mid B)=P(A)\\).", true],
      ["\\(P(B\\mid A)=P(B)\\).", true],
      ["\\(P(A\\cap B)=P(A)P(B)\\).", true],
      [
        "Observing either event gives no probability information about the other.",
        true,
      ],
    ],
    "These equations are equivalent ways to state that conditioning on one event does not change the other event's probability. Their equivalence requires the displayed conditionals to have positive denominators, which is guaranteed here.",
  ),
  makeQuestion(
    "crash-probability-l2-q95",
    "medium",
    "Which distinctions between statistical and causal claims are correct?",
    [
      [
        "A predictive association can be useful even when its cause is unknown.",
        true,
      ],
      [
        "Intervening on a variable asks a different question from merely conditioning on its observed value.",
        true,
      ],
      [
        "Independence proves that no hidden causal relationship exists under any measurement scheme.",
        false,
      ],
      [
        "Dependence identifies which variable should be changed to control the other.",
        false,
      ],
    ],
    "Prediction can exploit observed dependence without resolving why it exists, whereas causal decisions concern effects of interventions. Independence and dependence are properties of a specified distribution and do not alone settle hidden causes, direction, or intervention outcomes.",
  ),
  makeQuestion(
    "crash-probability-l2-q96",
    "hard",
    "In each of two age groups, treatment and recovery are independent, but the treated population is mostly younger and the untreated population mostly older; younger patients recover more often. Which statements are correct?",
    [
      [
        "Treatment and recovery can appear dependent after age groups are pooled.",
        true,
      ],
      [
        "Age can act as a common factor associated with treatment allocation and recovery.",
        true,
      ],
      [
        "Checking conditional recovery rates within age groups can reveal the stratified relationship.",
        true,
      ],
      [
        "Within-group independence guarantees pooled independence regardless of group proportions.",
        false,
      ],
    ],
    "Pooling groups with different baseline recovery and treatment shares can induce a marginal association even when each stratum factorizes. This is why conditioning structure and population composition matter when interpreting dependence or evaluating a possible treatment effect.",
  ),

  // Bayes' theorem and base rates
  makeQuestion(
    "crash-probability-l2-q97",
    "easy",
    "In Bayes' theorem \\(P(H\\mid D)=P(D\\mid H)P(H)/P(D)\\), which term is the prior probability of the hypothesis?",
    [
      ["\\(P(H)\\)", true],
      ["\\(P(D\\mid H)\\)", false],
      ["\\(P(H\\mid D)\\)", false],
      ["\\(P(D)\\)", false],
    ],
    "The prior \\(P(H)\\) represents belief in H before observing D. The likelihood describes data under H, the posterior is the updated belief, and the evidence normalizes across all hypotheses that could produce D.",
  ),
  makeQuestion(
    "crash-probability-l2-q98",
    "easy",
    "For a medical test, which term pairings are correct?",
    [
      ["Sensitivity: \\(P(+\\mid\\text{disease})\\).", true],
      ["Specificity: \\(P(-\\mid\\text{no disease})\\).", true],
      ["Positive predictive value: \\(P(+\\mid\\text{disease})\\).", false],
      ["False-positive rate: \\(P(+\\mid\\text{disease})\\).", false],
    ],
    "Sensitivity and specificity condition on true disease status and describe test behavior in those groups. Positive predictive value reverses the condition to \\(P(\\text{disease}\\mid+)\\), while false-positive rate is \\(P(+\\mid\\text{no disease})\\).",
  ),
  makeQuestion(
    "crash-probability-l2-q99",
    "medium",
    "In 1,000 people, disease prevalence is 1%. A test has 90% sensitivity and a 5% false-positive rate. Which natural-frequency statements are correct?",
    [
      ["About 10 people have the disease.", true],
      ["About 9 diseased people test positive.", true],
      ["About 49.5 of the 990 disease-free people test positive.", true],
      ["About 900 people test positive because sensitivity is 90%.", false],
    ],
    "Sensitivity applies only to the ten diseased people, while the false-positive rate applies to the much larger disease-free group. Applying 90% to all 1,000 confuses a conditional test characteristic with the population's positive-test rate.",
  ),
  makeQuestion(
    "crash-probability-l2-q100",
    "hard",
    "Using prevalence 1%, sensitivity 90%, and false-positive rate 5%, which posterior calculations are correct after a positive result?",
    [
      ["\\(P(+)=0.01(0.90)+0.99(0.05)=0.0585\\).", true],
      ["\\(P(\\text{disease}\\mid+)=0.009/0.0585\\approx0.154\\).", true],
      ["The posterior is 0.90 because sensitivity reverses directly.", false],
      ["The posterior is 0.01 because evidence cannot alter a prior.", false],
    ],
    "The evidence probability includes true-positive and false-positive paths, so Bayes gives a posterior near 15.4%. The low base rate means false positives outnumber true positives despite good sensitivity, which is exactly why reversing sensitivity is wrong.",
  ),
  makeQuestion(
    "crash-probability-l2-q101",
    "medium",
    "Which statements correctly explain the denominator \\(P(D)\\) in Bayes' theorem?",
    [
      ["It is the marginal probability of observing the data.", true],
      [
        "It can be computed by summing prior-times-likelihood paths over an exhaustive hypothesis partition.",
        true,
      ],
      [
        "It makes posterior probabilities across the hypotheses sum to one.",
        true,
      ],
      [
        "It includes ways the data can occur under alternatives to the focal hypothesis.",
        true,
      ],
    ],
    "The evidence term totals every disjoint route to D and therefore sets the scale for comparing hypotheses. Dividing each weighted path by this common total normalizes the posterior distribution while accounting for alternative explanations of the observation.",
  ),
  makeQuestion(
    "crash-probability-l2-q102",
    "hard",
    "Prior odds for a fault are 1:19, and an alarm is four times as likely under a fault as under no fault. Which statements are correct?",
    [
      ["The likelihood ratio is 4.", true],
      ["Posterior odds are \\(4:19\\).", true],
      ["The posterior fault probability is \\(4/(4+19)=4/23\\).", true],
      [
        "The posterior probability is \\(4/5\\) because the likelihood ratio is four.",
        false,
      ],
    ],
    "Bayes in odds form multiplies prior odds by the likelihood ratio, turning 1:19 into 4:19. Converting odds to probability requires dividing by the sum of both sides, so a likelihood ratio is not itself a posterior probability.",
  ),
  makeQuestion(
    "crash-probability-l2-q103",
    "easy",
    "Why can a rare condition still have a modest posterior probability after an accurate positive test?",
    [
      [
        "The large condition-free population can generate many false positives.",
        true,
      ],
      [
        "Sensitivity is defined as the posterior probability after a positive test.",
        false,
      ],
      ["Bayes' theorem ignores prevalence when the test is accurate.", false],
      ["A positive result makes the prior exactly zero.", false],
    ],
    "Even a small false-positive rate applied to a very large disease-free group can produce more positives than the true-positive path. Sensitivity conditions in the opposite direction, and Bayes combines rather than discards the prevalence prior.",
  ),
  makeQuestion(
    "crash-probability-l2-q104",
    "medium",
    "A test has sensitivity 0.92 and specificity 0.96. Which statements are correct?",
    [
      ["Its false-negative rate is \\(1-0.92=0.08\\).", true],
      ["Its false-positive rate is \\(1-0.96=0.04\\).", true],
      ["Its positive predictive value is 0.92 for every population.", false],
      ["Its disease prevalence is \\(1-0.96=0.04\\).", false],
    ],
    "False-negative and false-positive rates complement sensitivity and specificity within their respective true-status groups. Predictive value additionally depends on prevalence, while specificity says nothing about how common disease is.",
  ),
  makeQuestion(
    "crash-probability-l2-q105",
    "hard",
    "Disease prevalence is 20%. A test has sensitivity 0.80 and specificity 0.90. Which statements correctly analyze a negative result?",
    [
      ["\\(P(-\\cap\\text{disease})=0.20(0.20)=0.04\\).", true],
      ["\\(P(-\\cap\\text{no disease})=0.80(0.90)=0.72\\).", true],
      ["\\(P(\\text{disease}\\mid-)=0.04/(0.04+0.72)\\approx0.0526\\).", true],
      [
        "The posterior disease probability is the false-negative rate 0.20.",
        false,
      ],
    ],
    "A negative result can occur through diseased false negatives or disease-free true negatives, so both weighted paths form the denominator. The false-negative rate conditions on disease and must not be reversed into the posterior after observing a negative.",
  ),
  makeQuestion(
    "crash-probability-l2-q106",
    "easy",
    "Which Bayes terms have the stated interpretation?",
    [
      ["Prior: belief before the current observation", true],
      ["Likelihood: probability of the observation under a hypothesis", true],
      [
        "Evidence: total probability of the observation across hypotheses",
        true,
      ],
      ["Posterior: updated hypothesis probability after the observation", true],
    ],
    "Bayes combines prior plausibility and data compatibility, then normalizes by the total probability of seeing the evidence. The result is a posterior distribution that reflects both the base rates and how strongly each hypothesis predicts the observation.",
  ),
  makeQuestion(
    "crash-probability-l2-q107",
    "medium",
    "The same test is used in Population A with prevalence 1% and Population B with prevalence 20%. Sensitivity and specificity are unchanged. Which statements are correct?",
    [
      [
        "A positive result generally has higher positive predictive value in Population B.",
        true,
      ],
      [
        "The likelihood terms can stay fixed while the posterior changes with the prior.",
        true,
      ],
      [
        "Both populations must have the same posterior because the test hardware is identical.",
        false,
      ],
      [
        "Prevalence affects sensitivity by definition, so sensitivity must rise to 20%.",
        false,
      ],
    ],
    "Higher prevalence increases the prior odds of disease, so the same likelihood ratio produces higher posterior odds after a positive result. Sensitivity and specificity are conditional test characteristics here; they do not equal prevalence or force population-invariant predictive values.",
  ),
  makeQuestion(
    "crash-probability-l2-q108",
    "hard",
    "A screening model gives a posterior disease probability of 0.08. Missing disease costs 50 units, a false alarm costs 2, and correct decisions cost 0. Which statements are correct?",
    [
      ["Expected cost of no alarm is \\(0.08(50)=4\\).", true],
      ["Expected cost of alarm is \\(0.92(2)=1.84\\).", true],
      [
        "The alarm minimizes expected cost even though disease is not the most probable state.",
        true,
      ],
      [
        "Bayesian updating alone fixes a universal 0.50 decision threshold.",
        false,
      ],
    ],
    "Bayes supplies a posterior probability, while a decision rule combines it with asymmetric consequences. Here the expensive miss makes an alarm preferable at 8%; a 0.50 threshold assumes a different and much more symmetric cost structure.",
  ),

  // Conditional distributions in AI
  makeQuestion(
    "crash-probability-l2-q109",
    "easy",
    "Which expression best represents probabilistic supervised prediction of an output y from input x?",
    [
      ["\\(P(y\\mid x)\\)", true],
      ["\\(P(x\\mid y)\\) only", false],
      ["\\(P(x)+P(y)\\)", false],
      ["\\(P(y)/P(x)\\) without a joint model", false],
    ],
    "Supervised prediction asks for a distribution over possible outputs after the input is known, which is \\(P(y\\mid x)\\). The reversed likelihood can be useful in a generative model, but it needs a prior and Bayes' theorem to answer the predictive direction.",
  ),
  makeQuestion(
    "crash-probability-l2-q110",
    "easy",
    "Which statements correctly describe next-token prediction?",
    [
      [
        "The model estimates a distribution conditioned on the preceding context.",
        true,
      ],
      [
        "Changing the context can change the probability of the same candidate token.",
        true,
      ],
      [
        "All tokens are independent because the output uses a categorical distribution.",
        false,
      ],
      [
        "The context is a future outcome that is marginalized away before prediction.",
        false,
      ],
    ],
    "An LLM uses the observed prefix as conditioning information, allowing syntactic and semantic dependence to reshape the next-token distribution. Categorical output describes mutually exclusive next choices, not independence from the context that produced their probabilities.",
  ),
  makeQuestion(
    "crash-probability-l2-q111",
    "medium",
    "A generative classifier models \\(P(x\\mid y)P(y)\\), while a discriminative classifier models \\(P(y\\mid x)\\) directly. Which statements are correct?",
    [
      [
        "The generative model can use Bayes' theorem to obtain \\(P(y\\mid x)\\).",
        true,
      ],
      [
        "The class prior \\(P(y)\\) can affect the generative classifier's posterior.",
        true,
      ],
      [
        "Both approaches ultimately can produce a conditional distribution over labels.",
        true,
      ],
      [
        "The likelihood \\(P(x\\mid y)\\) is numerically identical to \\(P(y\\mid x)\\) for every dataset.",
        false,
      ],
    ],
    "Generative classification combines a class prior with how each class produces features, then normalizes over classes; discriminative modeling targets the posterior directly. The reversed conditionals describe different populations and are not interchangeable.",
  ),
  makeQuestion(
    "crash-probability-l2-q112",
    "hard",
    "A model has hidden state \\(Z\\in\\{0,1\\}\\), with \\(P(Z=1)=0.3\\), \\(P(X=x\\mid Z=1)=0.8\\), and \\(P(X=x\\mid Z=0)=0.2\\). Which calculations are correct?",
    [
      ["\\(P(X=x)=0.3(0.8)+0.7(0.2)=0.38\\).", true],
      ["\\(P(Z=1\\mid X=x)=0.24/0.38\\approx0.632\\).", true],
      [
        "\\(P(X=x)=0.8\\) because the more explanatory state has the larger likelihood.",
        false,
      ],
      [
        "\\(P(Z=1\\mid X=x)=0.8\\) because posterior and likelihood reverse directly.",
        false,
      ],
    ],
    "The observation probability marginalizes both hidden-state paths, and Bayes divides the state-1 joint path by that evidence. Neither the largest likelihood nor the likelihood alone accounts for the prior probability of the hidden state.",
  ),
  makeQuestion(
    "crash-probability-l2-q113",
    "medium",
    "Which statements correctly interpret a classifier output \\(P(Y\\mid X=x)\\)?",
    [
      [
        "For fixed x, probabilities across mutually exclusive exhaustive Y classes sum to one.",
        true,
      ],
      [
        "The distribution can express ambiguity between plausible classes.",
        true,
      ],
      [
        "A decision rule may use the posterior together with costs rather than simply choosing the largest class.",
        true,
      ],
      [
        "Calibration asks whether repeated conditional probabilities align with observed class frequencies.",
        true,
      ],
    ],
    "A conditional classifier provides a normalized distribution for the observed input, preserving uncertainty that a label alone hides. That distribution can feed either an argmax or a cost-sensitive decision, and calibration evaluates whether its numerical probabilities behave like frequencies.",
  ),
  makeQuestion(
    "crash-probability-l2-q114",
    "hard",
    "A prediction model observes feature X but a relevant discrete feature Z is missing. Which statements correctly describe \\(P(Y\\mid X)\\)?",
    [
      [
        "It can be obtained by summing \\(P(Y\\mid X,Z)P(Z\\mid X)\\) over Z.",
        true,
      ],
      [
        "The weights are \\(P(Z=z\\mid X)\\), the conditional probabilities of missing-feature values given observed X.",
        true,
      ],
      [
        "The result averages predictions across possible Z values rather than selecting one without evidence.",
        true,
      ],
      [
        "The missing feature can be discarded without any averaging because conditioning removes uncertainty.",
        false,
      ],
    ],
    "The law of total probability marginalizes the unobserved feature using its distribution after the observed input is known. Picking one hidden value would understate uncertainty and generally produce a different predictive distribution.",
  ),
  makeQuestion(
    "crash-probability-l2-q115",
    "easy",
    "In a stochastic environment, what does \\(P(S_{t+1}=s'\\mid S_t=s,A_t=a)\\) describe?",
    [
      [
        "The distribution of the next state after a given current state and action",
        true,
      ],
      ["The unconditional frequency of action a", false],
      ["The probability that state s caused every prior reward", false],
      ["The marginal distribution of all states with time removed", false],
    ],
    "The expression conditions on the current state-action pair and assigns probability to possible next states, forming a transition model. It does not specify how often the action is selected or make a causal statement about the entire past.",
  ),
  makeQuestion(
    "crash-probability-l2-q116",
    "medium",
    "Why is dependence useful for prediction? Which statements are correct?",
    [
      [
        "If Y depends on X, observing X can change the conditional distribution of Y.",
        true,
      ],
      [
        "A model can exploit stable dependence to reduce predictive uncertainty.",
        true,
      ],
      ["Perfect independence makes X maximally informative about Y.", false],
      [
        "Dependence guarantees the learned relationship is causal and stable after every distribution shift.",
        false,
      ],
    ],
    "Predictive features matter because their values change what outputs are probable; under independence, X leaves the Y distribution unchanged. Association can still be noncausal or unstable across environments, so useful prediction does not guarantee intervention validity or robustness.",
  ),
  makeQuestion(
    "crash-probability-l2-q117",
    "hard",
    "A risk model is evaluated separately for two groups. In each group, among cases assigned probability 0.20, about 20% experience the event. Which statements are correct?",
    [
      ["The model is calibrated at 0.20 within each evaluated group.", true],
      [
        "This is a conditional frequency statement, not proof that every individual has identical causal risk.",
        true,
      ],
      [
        "Pooling the groups could hide group-specific miscalibration if only the aggregate were checked.",
        true,
      ],
      [
        "Calibration at one probability value proves perfect classification accuracy.",
        false,
      ],
    ],
    "Group-conditional calibration compares forecast bins with event frequencies within each group, while individual causal risk is a stronger concept. Aggregate checks can average away opposite subgroup errors, and calibration does not say the most likely label is always correct.",
  ),
  makeQuestion(
    "crash-probability-l2-q118",
    "easy",
    "Which AI quantities are naturally conditional distributions?",
    [
      ["A class label given image features", true],
      ["A next token given the preceding text", true],
      ["A next state given the current state and action", true],
      ["A denoised state given a noisy state and a text condition", true],
    ],
    "Classification, language modeling, reinforcement learning transitions, and conditional diffusion all predict uncertain outputs from known information. The objects differ, but each uses conditioning to restrict and reshape the relevant probability distribution.",
  ),
  makeQuestion(
    "crash-probability-l2-q119",
    "medium",
    "A classifier knows \\(P(Y=1)=0.30\\) and \\(P(X=1\\mid Y=1)=0.80\\). Which statements are correct?",
    [
      ["\\(P(X=1,Y=1)=0.30(0.80)=0.24\\).", true],
      [
        "Computing \\(P(Y=1\\mid X=1)\\) also requires the total probability \\(P(X=1)\\).",
        true,
      ],
      [
        "\\(P(Y=1\\mid X=1)=0.80\\) because X and Y appear in both expressions.",
        false,
      ],
      ["The joint probability is \\(0.30+0.80=1.10\\).", false],
    ],
    "The prior-times-likelihood product gives the joint path for class 1 and feature 1. Reversing to a posterior requires normalizing against all feature-1 paths, so the likelihood is insufficient without the evidence probability.",
  ),
  makeQuestion(
    "crash-probability-l2-q120",
    "hard",
    "A mixture model has priors \\(P(Z=A)=0.25\\), \\(P(Z=B)=0.75\\) and likelihoods \\(P(x\\mid A)=0.60\\), \\(P(x\\mid B)=0.20\\). Which statements are correct?",
    [
      ["The A-and-x path is \\(0.25(0.60)=0.15\\).", true],
      ["The B-and-x path is \\(0.75(0.20)=0.15\\).", true],
      [
        "The evidence probability is 0.30 and the posterior over A and B is \\((0.5,0.5)\\).",
        true,
      ],
      [
        "A must have larger posterior probability because its likelihood is three times larger.",
        false,
      ],
    ],
    "Posterior comparison uses prior times likelihood: the smaller A prior exactly offsets its larger likelihood, giving equal joint path masses. Normalizing their sum produces equal posterior probabilities, illustrating why likelihood alone does not determine belief after data.",
  ),
];
