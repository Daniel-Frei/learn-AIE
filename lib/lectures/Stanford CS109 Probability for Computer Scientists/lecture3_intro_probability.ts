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
    chapter: 3,
    difficulty,
    prompt,
    options: options.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

export const stanfordCS109Lecture3IntroProbabilityQuestions: Question[] = [
  makeQuestion(
    "cs109-lect3-q01",
    "easy",
    "A fair six-sided die is rolled once and \\(E\\) denotes the result being at most 3. Which statements correctly describe this experiment?",
    [
      ["The sample space is \\(S=\\{1,2,3,4,5,6\\}\\).", true],
      ["A single number such as \\(4\\) is an outcome.", true],
      ["The event is \\(E=\\{1,2,3\\}\\).", true],
      ["The event \\(E\\) is a subset of the sample space.", true],
    ],
    "The sample space contains every complete result the experiment can produce, while an outcome is one member of that set. An event groups the outcomes with the desired property, so the results at most 3 form the subset \\(\\{1,2,3\\}\\).",
  ),
  makeQuestion(
    "cs109-lect3-q02",
    "easy",
    "Which descriptions give valid sample spaces for the stated experiments?",
    [
      [
        "For the number of emails received in a day, the nonnegative integers form a possible sample space.",
        true,
      ],
      [
        "For hours of video watched in a day, the real interval from 0 through 24 forms a possible sample space.",
        true,
      ],
      [
        "For two labeled coin flips, the four sequences HH, HT, TH, and TT form a possible sample space.",
        true,
      ],
      [
        "For one die roll, the event 'the result is even' must itself be the complete sample space.",
        false,
      ],
    ],
    "Sample spaces may be finite, countably infinite, or continuous as long as they enumerate all possible outcomes at the chosen level of description. An even-result set is an event inside the die's full sample space, not the complete set of possible die results.",
  ),
  makeQuestion(
    "cs109-lect3-q03",
    "easy",
    "Two labeled coins are flipped. Which sets are events in the sample space \\(S=\\{HH,HT,TH,TT\\}\\)?",
    [
      ["At least one head: \\(\\{HH,HT,TH\\}\\).", true],
      ["Exactly one head: \\(\\{HT,TH\\}\\).", true],
      [
        "The number \\(6\\), because any object can be treated as an event.",
        false,
      ],
      [
        "The set containing \\(HH,HT,TH,TT,HHH\\), because an event may extend the sample space.",
        false,
      ],
    ],
    "An event must be a subset of the selected sample space, and both head-related sets meet that requirement. The number 6 is not one of these coin-flip outcomes, while \\(HHH\\) describes three flips and therefore cannot be added to an event for this two-flip experiment.",
  ),
  makeQuestion(
    "cs109-lect3-q04",
    "easy",
    "Which number could be the probability of an event under the probability axioms?",
    [
      ["0.37", true],
      ["-0.04", false],
      ["1.08", false],
      ["37", false],
    ],
    "Every event probability lies in the closed interval from 0 to 1, inclusive, so 0.37 is admissible. Negative values and values greater than 1 cannot represent probability mass under the axioms, regardless of whether they arise from a calculation or an estimate.",
  ),
  makeQuestion(
    "cs109-lect3-q05",
    "easy",
    "A program repeatedly performs the same randomized experiment and records whether event \\(E\\) occurs. Which statements correctly connect the simulation to \\(P(E)\\)?",
    [
      [
        "After \\(n\\) trials, the empirical frequency is the number of hits divided by \\(n\\).",
        true,
      ],
      [
        "The empirical frequency can differ from \\(P(E)\\) after finitely many trials.",
        true,
      ],
      [
        "With more trials, the empirical frequency is expected to settle near \\(P(E)\\).",
        true,
      ],
      [
        "A simulation using \\(n<\\infty\\) trials can still check an analytic calculation.",
        true,
      ],
    ],
    "The frequentist interpretation connects probability to the limiting fraction of repeated trials in which the event occurs. A finite simulation produces an approximation with random error, but increasing the trial count generally makes that approximation more stable and useful as a check.",
  ),
  makeQuestion(
    "cs109-lect3-q06",
    "easy",
    "A self-driving car assigns probability 0.9 to a motorcycle being beside it, although the motorcycle is physically either present or absent. Which interpretations are appropriate?",
    [
      [
        "The probability can represent uncertainty in the car's information about the world.",
        true,
      ],
      [
        "The value can guide action even though the underlying state is binary.",
        true,
      ],
      [
        "Across many comparable cases assigned probability 0.9, roughly 90% should contain a motorcycle if the probabilities are well calibrated.",
        true,
      ],
      [
        "The value means the motorcycle is physically 90% present in this single case.",
        false,
      ],
    ],
    "Probability can express limited knowledge, not merely randomness in an object's physical state. A calibrated 0.9 belief has a repeated-case interpretation and can support decisions, but it does not divide one motorcycle into a 90%-present physical object.",
  ),
  makeQuestion(
    "cs109-lect3-q07",
    "easy",
    "Which statements are probability axioms or immediate requirements in the introductory setup?",
    [
      ["For every event \\(E\\), \\(0\\le P(E)\\le1\\).", true],
      ["The complete sample space has probability \\(P(S)=1\\).", true],
      ["Every nonempty event must have probability exactly \\(1/2\\).", false],
      [
        "For any events \\(E,F\\), \\(P(E\\cup F)=P(E)+P(F)\\) without conditions.",
        false,
      ],
    ],
    "Probabilities must be nonnegative and cannot exceed the total mass of 1, and the full sample space receives all of that mass. Additivity without an overlap correction applies to mutually exclusive events, while nonempty events can have many different probabilities.",
  ),
  makeQuestion(
    "cs109-lect3-q08",
    "easy",
    "An event occurs with probability 0.73. What is the probability that it does not occur?",
    [
      ["\\(1-0.73=0.27\\)", true],
      ["\\(0.73\\)", false],
      ["\\(1/0.73\\)", false],
      ["\\(0.73^2\\)", false],
    ],
    "An event and its complement are mutually exclusive and together cover the entire sample space. Their probabilities therefore add to 1, so the complement probability is \\(1-P(E)=0.27\\), not a reciprocal or a repeated-event calculation.",
  ),
  makeQuestion(
    "cs109-lect3-q09",
    "easy",
    "Suppose a finite sample space has equally likely outcomes. Which statements correctly justify \\(P(E)=|E|/|S|\\)?",
    [
      ["Each individual outcome has probability \\(1/|S|\\).", true],
      [
        "The event probability \\(P(E)\\) is the sum of the masses of its outcomes.",
        true,
      ],
      [
        "The formula requires the chosen outcomes in \\(S\\) to be equally likely.",
        true,
      ],
      [
        "Counting \\(|E|\\) and \\(|S|\\) is sufficient once equal likelihood is established.",
        true,
      ],
    ],
    "Equal likelihood assigns the same mass to every outcome, and the total mass of the sample space forces that common value to be \\(1/|S|\\). Adding the mass over the \\(|E|\\) outcomes in the event produces the ratio, but the argument fails if the outcomes have unequal weights.",
  ),
  makeQuestion(
    "cs109-lect3-q10",
    "easy",
    "Two fair labeled coins are flipped. Which statements about the equally likely sample space \\(\\{HH,HT,TH,TT\\}\\) are correct?",
    [
      ["Each ordered outcome has probability \\(1/4\\).", true],
      ["The event of exactly one head has probability \\(2/4=1/2\\).", true],
      ["The event of at least one head has probability \\(3/4\\).", true],
      [
        "The outcome \\(HT\\) is the same as \\(TH\\) because both contain one head.",
        false,
      ],
    ],
    "Labeling the first and second flips makes HT and TH distinct complete outcomes, and fairness gives all four outcomes equal probability. Exactly one head has two favorable outcomes, while at least one head also includes HH and therefore has three favorable outcomes.",
  ),
  makeQuestion(
    "cs109-lect3-q11",
    "easy",
    "Two independent fair distinguishable six-sided dice are rolled. Which calculations are correct?",
    [
      ["There are \\(6\\times6=36\\) equally likely ordered outcomes.", true],
      ["The probability that the sum is 7 is \\(6/36=1/6\\).", true],
      ["The probability that the sum is 2 is \\(2/36\\).", false],
      [
        "Each sum \\(s\\in\\{2,3,\\ldots,12\\}\\) has probability \\(1/11\\).",
        false,
      ],
    ],
    "The ordered pair of die values gives \\(6\\times6=36\\) equally likely outcomes. Six pairs sum to 7, but only \\((1,1)\\) sums to 2; grouping the pairs by their sum produces categories with unequal numbers of underlying outcomes.",
  ),
  makeQuestion(
    "cs109-lect3-q12",
    "easy",
    "A lottery is described only by the two outcomes 'win' and 'lose.' Why is \\(P(\\text{win})=1/2\\) not justified by that description?",
    [
      ["The two named outcomes need not be equally likely.", true],
      [
        "A probability can be computed only when a sample space has more than two outcomes.",
        false,
      ],
      ["The event 'win' is not a subset of the sample space.", false],
      ["Probabilities cannot be assigned to verbal outcomes.", false],
    ],
    "The ratio-of-counts formula depends on equal likelihood, not merely on there being two labels. Winning can correspond to far less probability mass than losing, so counting one favorable category out of two categories silently makes an unsupported symmetry assumption.",
  ),
  makeQuestion(
    "cs109-lect3-q13",
    "medium",
    "Why is the ordered-pair sample space \\(S=\\{(i,j):1\\le i,j\\le6\\}\\) effective for two fair dice?",
    [
      ["It distinguishes which die produced each value.", true],
      [
        "It contains exactly one outcome for each physical pair of die results.",
        true,
      ],
      [
        "Every outcome has probability one thirty-sixth when the dice are fair and independent.",
        true,
      ],
      [
        "Events such as a given sum can be represented as subsets of these pairs.",
        true,
      ],
    ],
    "The ordered representation preserves the identities of the two dice, so equally likely elementary results are neither merged nor duplicated. Once that sample space is established, any property of the roll, including its sum, becomes a subset whose probability can be found by counting pairs.",
  ),
  makeQuestion(
    "cs109-lect3-q14",
    "medium",
    "Two fair dice are represented by unordered value multisets such as \\(\\{1,1\\}\\) and \\(\\{1,2\\}\\). Which statements correctly diagnose this sample space?",
    [
      [
        "It is a valid way to describe the observable values if die identity is ignored.",
        true,
      ],
      ["Its outcomes are not equally likely.", true],
      [
        "The unordered result containing 1 and 2 is twice as likely as the double 1 result.",
        true,
      ],
      [
        "The ratio of favorable multisets to all multisets can always be used directly.",
        false,
      ],
    ],
    "An unordered multiset is a legitimate descriptive sample space, but merging \\((1,2)\\) with \\((2,1)\\) gives that multiset twice the probability of a double such as \\((1,1)\\). Direct counting therefore treats unequal-probability outcomes as equal and gives incorrect probabilities unless weights are included.",
  ),
  makeQuestion(
    "cs109-lect3-q15",
    "medium",
    "When designing a sample space for a counting-based probability calculation, which practices are useful?",
    [
      [
        "Make physically distinguishable items explicit when doing so exposes symmetry.",
        true,
      ],
      [
        "Check equal likelihood before using a favorable-over-total counting ratio.",
        true,
      ],
      [
        "Prefer the sample space with the fewest labels even if its outcomes have unequal probability.",
        false,
      ],
      [
        "Assume any unordered representation is equally likely because order was removed.",
        false,
      ],
    ],
    "The goal is not merely to find a small representation but to find elementary outcomes with defensible equal probability. Distinguishing objects often prevents unequal cases from being merged, while unordered outcomes may still work when they remain equally likely, as with uniformly chosen subsets of distinct objects.",
  ),
  makeQuestion(
    "cs109-lect3-q16",
    "medium",
    "A bag contains 4 distinct cow toys and 3 distinct pig toys. Three toys are selected uniformly without replacement, with order ignored. What is the probability of selecting exactly 1 cow and 2 pigs?",
    [
      ["\\(\\dfrac{\\binom41\\binom32}{\\binom73}=\\dfrac{12}{35}\\)", true],
      ["\\(\\dfrac{1}{2}\\), because the toy types are cow or pig.", false],
      [
        "\\(\\dfrac{3}{4}\\), because two of the three selected toys are pigs.",
        false,
      ],
      ["\\(\\dfrac{\\binom31\\binom42}{\\binom73}\\)", false],
    ],
    "The denominator counts all three-toy subsets of seven distinct toys. A favorable subset chooses one of the four cows and two of the three pigs, giving \\(4\\times3=12\\) subsets; the reversed binomial factors count two cows and one pig instead.",
  ),
  makeQuestion(
    "cs109-lect3-q17",
    "medium",
    "The cow-and-pig selection is instead modeled as three ordered draws of distinct toys. Which statements correctly recover the same probability of exactly 1 cow and 2 pigs?",
    [
      [
        "The ordered sample space has \\(7\\times6\\times5=210\\) outcomes.",
        true,
      ],
      ["The cow can occupy any one of the \\(3\\) draw positions.", true],
      [
        "The favorable ordered count is \\(3\\times4\\times3\\times2=72\\).",
        true,
      ],
      ["The ratio \\(72/210\\) simplifies to \\(12/35\\).", true],
    ],
    "Ordering multiplies every unordered three-toy subset by \\(3!\\), so both the numerator and denominator grow by the same factor. Counting the cow position, cow identity, and ordered pig identities gives 72 favorable sequences, and the probability agrees with the unordered model.",
  ),
  makeQuestion(
    "cs109-lect3-q18",
    "medium",
    "A five-card poker hand is an unordered subset of a standard 52-card deck. A straight has one of 10 allowed consecutive rank sequences, with each card's suit unrestricted. Which statements are correct?",
    [
      ["The sample space has \\(\\binom{52}{5}\\) equally likely hands.", true],
      [
        "There are \\(10\\cdot4^5\\) straights when straight flushes are included.",
        true,
      ],
      [
        "A straight flush count of \\(10\\cdot4\\) must be subtracted if straight flushes are excluded.",
        true,
      ],
      [
        "There are \\(52^5\\) equally likely hands because five card values are chosen independently.",
        false,
      ],
    ],
    "A hand is selected without replacement and without order, which gives the binomial denominator rather than \\(52^5\\). Choosing the rank sequence and then one of four suits for each rank gives all straights, while requiring one common suit leaves four choices per rank sequence.",
  ),
  makeQuestion(
    "cs109-lect3-q19",
    "medium",
    "Among \\(n\\) distinct chips exactly one is defective, and \\(k\\) chips are chosen uniformly for testing. Which expressions correctly give the probability that the defective chip is selected?",
    [
      ["\\(\\dfrac{\\binom{n-1}{k-1}}{\\binom nk}\\)", true],
      ["\\(k/n\\)", true],
      ["\\(\\binom nk/n\\)", false],
      ["\\((n-k)/n\\)", false],
    ],
    "A favorable test set must contain the one defective chip and choose its other \\(k-1\\) members from the \\(n-1\\) good chips. Simplifying that ratio yields \\(k/n\\), which also follows by symmetry because each chip has the same chance of occupying one of the \\(k\\) selected positions.",
  ),
  makeQuestion(
    "cs109-lect3-q20",
    "medium",
    "A point is uniformly distributed over an \\(800\\times800\\) square screen. What is the probability that it lands inside a fully contained circular target of radius 200?",
    [
      ["\\(\\dfrac{\\pi(200)^2}{800^2}=\\dfrac{\\pi}{16}\\)", true],
      ["\\(200/800=1/4\\)", false],
      ["\\(\\pi(200)/800\\)", false],
      ["\\(\\pi(400)^2/800^2=\\pi/4\\)", false],
    ],
    "Uniform location makes probability proportional to area, so the event area is the circle's \\(\\pi r^2\\) and the sample-space area is the square's side squared. Ratios of radii or perimeters do not measure the fraction of two-dimensional outcomes inside the target.",
  ),
  makeQuestion(
    "cs109-lect3-q21",
    "medium",
    "Events \\(E_1,E_2,E_3\\) are pairwise mutually exclusive. Which statements are correct?",
    [
      ["No outcome belongs to more than one of the three events.", true],
      [
        "The probability of their union equals the sum of their three probabilities.",
        true,
      ],
      ["The events may have different probabilities.", true],
      ["Their union need not be the entire sample space.", true],
    ],
    "Mutual exclusivity removes overlap, so probability mass can be added without double-counting and without requiring equal event sizes. The union covers the full sample space only when the events also form an exhaustive partition, which is an additional condition.",
  ),
  makeQuestion(
    "cs109-lect3-q22",
    "medium",
    "Which facts justify the complement identity \\(P(E^c)=1-P(E)\\)?",
    [
      ["\\(E\\) and \\(E^c\\) are mutually exclusive.", true],
      ["\\(E\\cup E^c=S\\).", true],
      ["\\(P(S)=1\\).", true],
      ["\\(E\\) and \\(E^c\\) must have equal probability.", false],
    ],
    "An event and its complement partition the sample space: they do not overlap, and every outcome belongs to one of them. Additivity and \\(P(S)=1\\) therefore give \\(P(E)+P(E^c)=1\\); no symmetry requires the two terms to be equal.",
  ),
  makeQuestion(
    "cs109-lect3-q23",
    "medium",
    "A campus has \\(N\\) people, you know \\(f\\) of them, and a room contains a uniformly selected subset of \\(r\\) people. Which statements correctly express the chance that you know at least one person in the room?",
    [
      [
        "The probability of knowing nobody is \\(\\binom{N-f}{r}/\\binom Nr\\).",
        true,
      ],
      ["The desired probability is \\(1-\\binom{N-f}{r}/\\binom Nr\\).", true],
      [
        "The desired probability is simply \\(f/N\\), regardless of \\(r\\).",
        false,
      ],
      [
        "The event must be counted by adding the cases for exactly 1 through exactly \\(r\\) friends.",
        false,
      ],
    ],
    "A room with no acquaintances must draw all \\(r\\) people from the \\(N-f\\) strangers, while the denominator counts all possible rooms. Taking the complement captures every room with one or more acquaintances at once and avoids a long sum of disjoint exact-count cases.",
  ),
  makeQuestion(
    "cs109-lect3-q24",
    "medium",
    "An analytic calculation gives \\(P(E)=1/6\\), while a simulation of 10,000 trials reports 0.1649. What is the best interpretation?",
    [
      [
        "The finite simulation is reasonably close to the analytic probability and need not match it exactly.",
        true,
      ],
      [
        "The analytic answer is disproved because 0.1649 is not exactly repeating 6.",
        false,
      ],
      [
        "The simulation establishes that the true probability is exactly 0.1649.",
        false,
      ],
      [
        "The difference shows that the simulated experiment had no randomness.",
        false,
      ],
    ],
    "Finite trial counts produce sampling variation, so empirical frequencies fluctuate around the underlying probability. The analytic value remains the exact model-based result, while 0.1649 is one noisy estimate that should tend to stabilize nearer \\(1/6\\) as the number of trials grows.",
  ),
  makeQuestion(
    "cs109-lect3-q25",
    "hard",
    "Two analysts model the same random experiment with different sample spaces. Which statements correctly describe when their probability answers can still agree?",
    [
      [
        "They may use different outcome descriptions if each assigns the correct probability mass.",
        true,
      ],
      [
        "A coarser sample space can work when merged outcomes are weighted by their underlying probabilities.",
        true,
      ],
      [
        "Two equally likely sample spaces can differ in size if favorable and total counts change by the same multiplicative refinement factor.",
        true,
      ],
      [
        "Agreement depends on representing the same physical event, not on using identical outcome labels.",
        true,
      ],
    ],
    "A sample space is a modeling choice, and refinements or coarsenings can preserve probabilities when mass is transferred correctly. The danger arises only when a ratio-of-counts calculation assigns equal weight to coarse outcomes that actually represent different numbers or masses of elementary results.",
  ),
  makeQuestion(
    "cs109-lect3-q26",
    "hard",
    "A fair die is rolled independently twice, but only the sum is recorded. Which statements are correct about using sums \\(2,3,\\ldots,12\\) as the sample space?",
    [
      [
        "The sum-based sample space is valid as a description of the recorded result.",
        true,
      ],
      [
        "The sums are not equally likely because they have different numbers of ordered-pair realizations.",
        true,
      ],
      [
        "A weighted sum-space calculation can reproduce probabilities from the ordered-pair model.",
        true,
      ],
      [
        "The probability of sum 7 is one eleventh because there are eleven possible sums.",
        false,
      ],
    ],
    "Recording only the sum legitimately coarsens the experiment, but the resulting outcomes inherit unequal masses: sum 7 has six ordered realizations while sum 2 has one. Weighting each sum by its number of realizations recovers the correct distribution; treating the eleven sums uniformly does not.",
  ),
  makeQuestion(
    "cs109-lect3-q27",
    "hard",
    "Under the poker convention with 10 allowed rank sequences, which expressions correctly distinguish a straight from a straight flush in an unordered five-card hand?",
    [
      [
        "\\(P(\\text{straight including flushes})=10\\cdot4^5/\\binom{52}{5}\\).",
        true,
      ],
      [
        "\\(P(\\text{straight but not flush})=10(4^5-4)/\\binom{52}{5}\\).",
        true,
      ],
      ["\\(P(\\text{straight flush})=10\\cdot4^5/\\binom{52}{5}\\).", false],
      [
        "Subtracting \\(10\\cdot4\\) removes every straight with mixed suits.",
        false,
      ],
    ],
    "For each rank sequence, unrestricted suits yield \\(4^5\\) hands, but a straight flush has only four common-suit choices. Subtracting those \\(10\\cdot4\\) straight flushes leaves the mixed-suit straights; it does not remove the mixed-suit cases themselves.",
  ),
  makeQuestion(
    "cs109-lect3-q28",
    "hard",
    "A student computes the probability that two independent fair dice sum to 7 as \\(3/21\\) by counting unordered value pairs. Which diagnosis is correct?",
    [
      [
        "The calculation incorrectly treats unequal-probability unordered pairs as equally likely.",
        true,
      ],
      [
        "The event should contain no unordered pairs because order always matters.",
        false,
      ],
      ["The denominator must be 11 because only the sum is relevant.", false],
      ["The numerator should be 6 while the denominator remains 21.", false],
    ],
    "The three favorable unordered pairs \\(\\{1,6\\},\\{2,5\\},\\{3,4\\}\\) each have two ordered realizations, while the six doubles in the 21-pair space have only one. The sample space is valid, but a raw count ratio ignores these unequal weights; the 36 ordered pairs give \\(6/36=1/6\\).",
  ),
  makeQuestion(
    "cs109-lect3-q29",
    "hard",
    "Exactly one of \\(n\\) chips is defective, and a uniformly random \\(k\\)-chip test set is chosen. Which statements are correct?",
    [
      ["The probability of detecting the defect is \\(k/n\\).", true],
      ["The probability of missing it is \\((n-k)/n\\).", true],
      ["Testing all \\(n\\) chips gives detection probability 1.", true],
      [
        "Doubling \\(k\\) doubles the detection probability whenever the new value remains at most \\(n\\).",
        true,
      ],
    ],
    "Uniform subset selection treats every chip symmetrically, so the unique defective chip is included in exactly a \\(k/n\\) fraction of test sets. Its complement is the miss probability, and the formula has the expected boundary at \\(k=n\\); within the feasible range it is linear in \\(k\\).",
  ),
  makeQuestion(
    "cs109-lect3-q30",
    "hard",
    "A bag contains 4 distinct cows and 3 distinct pigs, and three toys are drawn without replacement. Which alternative counting arguments correctly yield the probability of exactly 1 cow and 2 pigs?",
    [
      ["Use unordered subsets: \\(\\binom41\\binom32/\\binom73\\).", true],
      ["Use ordered draws: \\(3(4)(3)(2)/(7\\cdot6\\cdot5)\\).", true],
      [
        "Both ratios equal \\(12/35\\) because ordering refines every three-toy subset by \\(3!\\).",
        true,
      ],
      [
        "Use toy types alone: \\(1/2\\), from one favorable type sequence out of two type outcomes.",
        false,
      ],
    ],
    "The first two models retain distinct toy identities and produce equally likely elementary outcomes, so they agree after the common ordering factor cancels. A type-only model merges many selections into cow/pig patterns with different multiplicities and cannot use an unweighted count of type labels.",
  ),
  makeQuestion(
    "cs109-lect3-q31",
    "hard",
    "A dart is equally likely to land anywhere on a rectangular board. Which statements correctly describe geometric probability for a target region entirely inside the board?",
    [
      ["The target probability is target area divided by board area.", true],
      [
        "The result depends on uniformity over location, not merely on naming two outcomes 'hit' and 'miss.'",
        true,
      ],
      [
        "Replacing area by perimeter generally gives a different and unjustified quantity.",
        false,
      ],
      [
        "Every target shape with the same perimeter must have the same hit probability.",
        false,
      ],
    ],
    "Uniform spatial outcomes distribute probability mass in proportion to area, so a region's shape matters only through its area under this model. A two-label hit/miss sample space is not equally likely in general, and perimeter neither measures two-dimensional mass nor determines enclosed area uniquely.",
  ),
  makeQuestion(
    "cs109-lect3-q32",
    "hard",
    "Which condition is essential before replacing an event probability by the counting ratio \\(|E|/|S|\\)?",
    [
      [
        "Every elementary outcome in the finite sample space must be equally likely.",
        true,
      ],
      ["The event must contain exactly half the sample space.", false],
      ["The outcomes must be written in increasing numerical order.", false],
      ["The experiment must use physical dice or cards.", false],
    ],
    "The ratio works because equal likelihood gives every elementary outcome the common mass \\(1/|S|\\). Neither the event's size, the notation used, nor the physical type of randomizer establishes that symmetry; it must follow from the experiment and the chosen representation.",
  ),
  makeQuestion(
    "cs109-lect3-q33",
    "hard",
    "A room is a uniformly selected \\(r\\)-person subset of a population of \\(N\\), and you know \\(f\\) people. Which statements correctly compare direct and complement counting for the event of seeing at least one acquaintance?",
    [
      [
        "Complement counting uses the single term \\(1-\\binom{N-f}{r}/\\binom Nr\\).",
        true,
      ],
      [
        "Direct counting can sum the disjoint cases with exactly \\(j\\) acquaintances for feasible \\(j\\ge1\\).",
        true,
      ],
      [
        "Complement counting changes the result to \\(1-r/N\\) by using a different experiment.",
        false,
      ],
      [
        "The direct sum must differ from \\(1-\\binom{N-f}{r}/\\binom Nr\\) because it uses several cases.",
        false,
      ],
    ],
    "Both methods describe the same uniformly selected room and therefore must agree when counted correctly. The complement is compact because 'no acquaintances' has one simple binomial count, whereas the direct method partitions the desired event by the exact number of acquaintances present.",
  ),
  makeQuestion(
    "cs109-lect3-q34",
    "hard",
    "A probability problem will be solved by counting equally likely outcomes. Which steps form a sound workflow?",
    [
      ["Define what one complete outcome records before counting.", true],
      ["Argue why the chosen elementary outcomes are equally likely.", true],
      [
        "Construct the event as a subset using the same ordered or unordered convention as the sample space.",
        true,
      ],
      [
        "Choose the smallest denominator first and infer the experiment from it afterward.",
        false,
      ],
    ],
    "A correct ratio requires the numerator and denominator to live in the same representation and to count elementary outcomes with equal mass. Starting from a convenient number without defining the experiment invites mismatched order conventions, merged outcomes, and unsupported equal-likelihood assumptions.",
  ),
  makeQuestion(
    "cs109-lect3-q35",
    "hard",
    "Which conclusions follow from the probability axioms for any event \\(E\\) and mutually exclusive events \\(A_1,\\ldots,A_m\\)?",
    [
      ["\\(P(E^c)=1-P(E)\\).", true],
      ["\\(P(\\varnothing)=0\\).", true],
      ["\\(P(A_1\\cup\\cdots\\cup A_m)=\\sum_{i=1}^m P(A_i)\\).", true],
      ["If \\(E\\subseteq S\\), then \\(0\\le P(E)\\le1\\).", true],
    ],
    "The complement identity follows because \\(E\\) and \\(E^c\\) partition a sample space of mass 1, and the empty event then has zero mass as the complement of \\(S\\). Finite additivity handles mutually exclusive unions, while the bound on event probabilities is the first axiom.",
  ),
];
