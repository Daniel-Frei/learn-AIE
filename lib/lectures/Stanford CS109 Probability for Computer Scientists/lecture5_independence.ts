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
    chapter: 5,
    difficulty,
    prompt,
    options: options.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

export const stanfordCS109Lecture5IndependenceQuestions: Question[] = [
  makeQuestion(
    "cs109-lect5-q01",
    "easy",
    "Events \\(A\\) and \\(B\\) have positive probability. Which statements are equivalent ways to express that they are independent?",
    [
      ["\\(P(A\\mid B)=P(A)\\).", true],
      ["\\(P(B\\mid A)=P(B)\\).", true],
      ["\\(P(A\\cap B)=P(A)P(B)\\).", true],
      [
        "Learning that one event occurred does not change the probability assigned to the other.",
        true,
      ],
    ],
    "Independence says that observing one event leaves the probability of the other unchanged. Substituting that condition into the chain rule produces the product identity, and Bayes' theorem shows that the unchanged-information relationship is reciprocal when the conditioning probabilities are positive.",
  ),
  makeQuestion(
    "cs109-lect5-q02",
    "easy",
    "For arbitrary events \\(E\\) and \\(F\\), which statements correctly calculate or interpret \\(P(E\\cup F)\\)?",
    [
      ["\\(P(E\\cup F)=P(E)+P(F)-P(E\\cap F)\\).", true],
      [
        "The intersection is subtracted because adding the two event probabilities counts its mass twice.",
        true,
      ],
      [
        "If the events are mutually exclusive, the intersection term is zero.",
        true,
      ],
      [
        "If the events are independent, the intersection term must be omitted rather than evaluated.",
        false,
      ],
    ],
    "Two-set inclusion-exclusion adds the probability mass in each event and removes one copy of their overlap. Mutual exclusivity makes that overlap zero, whereas independence generally makes it \\(P(E)P(F)\\), which is still part of the calculation rather than a reason to omit the term.",
  ),
  makeQuestion(
    "cs109-lect5-q03",
    "easy",
    "Suppose \\(P(A)>0\\) and \\(P(B)>0\\). Which statements correctly contrast mutual exclusivity with independence?",
    [
      [
        "Mutually exclusive events cannot occur together, so \\(P(A\\cap B)=0\\).",
        true,
      ],
      [
        "Independent events with positive probabilities satisfy \\(P(A\\cap B)=P(A)P(B)>0\\).",
        true,
      ],
      [
        "Mutual exclusivity implies that observing \\(B\\) leaves \\(P(A)\\) unchanged.",
        false,
      ],
      [
        "Independence implies that the two event sets have no outcomes in common.",
        false,
      ],
    ],
    "Positive-probability mutually exclusive events are dependent: observing one rules out the other and changes its probability to zero. Independent positive-probability events must have positive overlap equal to the product of their marginal probabilities, even though neither event informs the other.",
  ),
  makeQuestion(
    "cs109-lect5-q04",
    "easy",
    "A sensor succeeds on a trial with probability 0.8, independently of a second sensor that succeeds with probability 0.7. What is the probability that both succeed?",
    [
      ["0.56", true],
      ["0.94", false],
      ["0.75", false],
      ["0.10", false],
    ],
    "For independent success events, the probability of their intersection is the product \\(0.8\\times0.7=0.56\\). The value 0.94 would come from an incorrect complement calculation, 0.75 merely averages the rates, and 0.10 is their difference rather than a joint probability.",
  ),
  makeQuestion(
    "cs109-lect5-q05",
    "easy",
    "Which expression is the inclusion-exclusion formula for three events \\(E,F,G\\)?",
    [
      [
        "\\(P(E)+P(F)+P(G)-P(E\\cap F)-P(E\\cap G)-P(F\\cap G)+P(E\\cap F\\cap G)\\)",
        true,
      ],
      ["\\(P(E)+P(F)+P(G)-P(E\\cap F\\cap G)\\)", false],
      ["\\(P(E)+P(F)+P(G)+P(E\\cap F)+P(E\\cap G)+P(F\\cap G)\\)", false],
      ["\\(P(E)+P(F)+P(G)-P(E\\cap F)-P(E\\cap G)-P(F\\cap G)\\)", false],
    ],
    "Adding the three single-event probabilities counts pairwise overlaps twice and the triple overlap three times. Subtracting every pair removes those extra copies but removes the triple overlap once too many, so the three-way intersection must be added back exactly once.",
  ),
  makeQuestion(
    "cs109-lect5-q06",
    "easy",
    "If \\(A\\) and \\(B\\) are independent, which additional pairs must also be independent?",
    [
      ["\\(A\\) and \\(B^c\\)", true],
      ["\\(A^c\\) and \\(B^c\\)", true],
      ["\\(A\\) and \\(A^c\\)", false],
      ["\\(A\\cap B\\) and \\(A\\cup B\\)", false],
    ],
    "Independence is preserved when either or both events are replaced by their complements, so the first two pairs inherit the property. An event and its own complement are mutually exclusive, and the intersection is contained in the union; neither relationship is generally independent when the relevant probabilities are nonzero.",
  ),
  makeQuestion(
    "cs109-lect5-q07",
    "easy",
    "A sequence of coin flips is modeled as mutually independent trials. Which consequences follow for a particular outcome containing three heads and two tails?",
    [
      [
        "Its probability is the product of the five corresponding head or tail probabilities.",
        true,
      ],
      [
        "For a head probability \\(p\\), every particular ordering with three heads has probability \\(p^3(1-p)^2\\).",
        true,
      ],
      [
        "Replacing a head event by its tail complement preserves the independence needed for multiplication.",
        true,
      ],
      [
        "The five single-flip events must be mutually exclusive because their probabilities are multiplied.",
        false,
      ],
    ],
    "Mutual independence lets the joint probability of a specified flip sequence factor into its per-flip probabilities, and complement events remain independent. The events are not mutually exclusive: a head on one flip and a tail on another can occur in the same sequence, so multiplication comes from independence rather than exclusivity.",
  ),
  makeQuestion(
    "cs109-lect5-q08",
    "easy",
    "For events \\(E_1,E_2,E_3\\), which statements correctly distinguish pairwise independence from mutual independence?",
    [
      [
        "Pairwise independence checks the product identity for each of the three pairs.",
        true,
      ],
      [
        "Mutual independence also checks the product identity for the intersection of all three events.",
        true,
      ],
      ["Mutual independence implies pairwise independence.", true],
      ["Pairwise independence alone need not imply mutual independence.", true],
    ],
    "Mutual independence is a collection of product identities for every subset containing at least two events. With three events this includes all three pairwise checks and the three-way check, so it is stronger than pairwise independence and cannot be established from the pairwise identities alone.",
  ),
  makeQuestion(
    "cs109-lect5-q09",
    "easy",
    "Two fair dice are rolled independently. Let \\(E=\\{D_1=1\\}\\) and \\(F=\\{D_2=1\\}\\). Which calculations establish that \\(E\\) and \\(F\\) are independent?",
    [
      ["\\(P(E)=P(F)=1/6\\).", true],
      ["\\(P(E\\cap F)=1/36=(1/6)(1/6)\\).", true],
      ["\\(P(E\\cup F)=1/6\\).", false],
      ["\\(P(E\\mid F)=1/36\\).", false],
    ],
    "The marginal probabilities are each \\(1/6\\), and exactly one of the 36 ordered outcomes makes both dice equal 1. Thus the joint probability equals the product of the marginals; the union is \\(11/36\\), and conditioning on the second die leaves the first-die probability at \\(1/6\\), not \\(1/36\\).",
  ),
  makeQuestion(
    "cs109-lect5-q10",
    "easy",
    "In a uniform finite sample space, what geometric relationships characterize independent events \\(A\\) and \\(B\\)?",
    [
      [
        "The fraction of the whole sample space occupied by \\(A\\) equals the fraction of \\(B\\) occupied by \\(A\\cap B\\).",
        true,
      ],
      [
        "Restricting attention to \\(B\\) leaves the relative frequency of \\(A\\) unchanged.",
        true,
      ],
      [
        "The overlap has relative size \\(P(A)P(B)\\) in the full sample space.",
        true,
      ],
      [
        "The event regions must be disjoint whenever neither event has probability zero.",
        false,
      ],
    ],
    "In the equally likely setting, probability is proportional to area or outcome count. Independence therefore means that \\(A\\) occupies the same fraction inside \\(B\\) as it does globally, making the overlap fraction the product of the two marginal fractions rather than forcing the regions apart.",
  ),
  makeQuestion(
    "cs109-lect5-q11",
    "easy",
    "Which statements correctly describe the generalized chain rule for \\(A\\cap B\\cap C\\)?",
    [
      ["One valid factorization is \\(P(A)P(B\\mid A)P(C\\mid A,B)\\).", true],
      [
        "The event introduced by each later factor is conditioned on all events already introduced.",
        true,
      ],
      [
        "The factors may be written in a different event order if the conditioning sets change consistently.",
        true,
      ],
      [
        "Under mutual independence, the factorization reduces to \\(P(A)P(B)P(C)\\).",
        true,
      ],
    ],
    "The chain rule builds a joint event one component at a time, conditioning each new event on the components already included. Any consistent ordering reaches the same intersection, and mutual independence removes the effect of every accumulated conditioning set so only marginal probabilities remain.",
  ),
  makeQuestion(
    "cs109-lect5-q12",
    "easy",
    "A biased coin has head probability \\(p\\). What is the probability of the particular six-flip sequence HHTHTT?",
    [
      ["\\(p^3(1-p)^3\\)", true],
      ["\\(\\binom{6}{3}p^3(1-p)^3\\)", false],
      ["\\(p^2(1-p)^4\\)", false],
      ["\\(p^3+(1-p)^3\\)", false],
    ],
    "The specified sequence contains three independent head events and three independent tail events, so their probabilities multiply to \\(p^3(1-p)^3\\). The binomial coefficient belongs to the broader event of three heads in any order, while the other expressions use the wrong counts or add mutually compatible requirements.",
  ),
  makeQuestion(
    "cs109-lect5-q13",
    "medium",
    "Independent network routes succeed with probabilities \\(p_1,\\ldots,p_n\\). The network works when at least one route succeeds. Which steps correctly derive its reliability?",
    [
      ["Take the complement of the event that every route fails.", true],
      ["Route \\(i\\) fails with probability \\(1-p_i\\).", true],
      [
        "The probability that every route fails is \\(\\prod_{i=1}^n(1-p_i)\\).",
        true,
      ],
      ["The reliability is \\(1-\\prod_{i=1}^n(1-p_i)\\).", true],
    ],
    "The at-least-one event is awkward as a union because successful routes need not be mutually exclusive. Its complement is an intersection of independent failure events, whose probability factors into the product of failure rates; subtracting that product from one gives the desired reliability.",
  ),
  makeQuestion(
    "cs109-lect5-q14",
    "medium",
    "Three independent routes succeed with probabilities 0.5, 0.7, and 0.9. Which statements about the network are correct if any successful route provides connectivity?",
    [
      [
        "The probability that all routes fail is \\(0.5\\times0.3\\times0.1=0.015\\).",
        true,
      ],
      ["The probability of connectivity is \\(1-0.015=0.985\\).", true],
      [
        "Directly adding 0.5, 0.7, and 0.9 would double-count outcomes with multiple successful routes.",
        true,
      ],
      [
        "The probability of connectivity is \\(0.5\\times0.7\\times0.9=0.315\\).",
        false,
      ],
    ],
    "Connectivity is the complement of simultaneous failure, and independence makes that failure probability the product of 0.5, 0.3, and 0.1. Multiplying the success rates instead asks for all three routes to work, while adding them treats overlapping success events as though they were mutually exclusive.",
  ),
  makeQuestion(
    "cs109-lect5-q15",
    "medium",
    "Each child independently has probability 0.25 of inheriting a particular trait. Which statements correctly compute the probability that all three children have the trait?",
    [
      [
        "The event is an intersection of three independent child-level events.",
        true,
      ],
      ["Its probability is \\(0.25^3=1/64\\).", true],
      [
        "Its probability is \\(3(0.25)\\) because the child-level events are mutually exclusive.",
        false,
      ],
      [
        "The generalized chain rule requires an unknown dependence correction even after independence is stated.",
        false,
      ],
    ],
    "The three trait events can occur together and are modeled as independent once the parental setup is fixed, so their joint probability is the product \\(0.25^3\\). Adding would describe a mutually exclusive union, and no extra conditional factor is needed after independence removes the influence of earlier child outcomes.",
  ),
  makeQuestion(
    "cs109-lect5-q16",
    "medium",
    "A coin is flipped independently \\(n\\) times with head probability \\(p\\). Which expression gives the probability of exactly \\(k\\) heads?",
    [
      ["\\(\\binom{n}{k}p^k(1-p)^{n-k}\\)", true],
      ["\\(p^k(1-p)^{n-k}\\)", false],
      ["\\(\\binom{n}{k}p^{n-k}(1-p)^k\\)", false],
      ["\\(\\binom{n}{k}[p+(1-p)]^n\\)", false],
    ],
    "Every particular sequence with \\(k\\) heads has probability \\(p^k(1-p)^{n-k}\\), and there are \\(\\binom{n}{k}\\) mutually exclusive placements of those heads. Omitting the coefficient counts only one ordering, while swapping the exponents describes exactly \\(k\\) tails instead.",
  ),
  makeQuestion(
    "cs109-lect5-q17",
    "medium",
    "Ten independent flips have head probability 0.6. Which expression gives the probability that the first four flips are heads and the final six are tails?",
    [
      ["\\(0.6^4 0.4^6\\)", true],
      ["\\(\\binom{10}{4}0.6^4 0.4^6\\)", false],
      ["\\(0.6^6 0.4^4\\)", false],
      ["\\(1-0.6^4 0.4^6\\)", false],
    ],
    "The prompt fixes one ordering, so independence contributes four head factors and six tail factors without a combinatorial multiplier. Multiplying by \\(\\binom{10}{4}\\) would broaden the event to all possible placements of four heads, and the other choices reverse the counts or take an unrelated complement.",
  ),
  makeQuestion(
    "cs109-lect5-q18",
    "medium",
    "Why does the exactly-\\(k\\)-heads calculation combine both independence and mutual exclusivity?",
    [
      [
        "Independence gives the product probability for each particular head-tail sequence.",
        true,
      ],
      [
        "Distinct complete sequences are mutually exclusive, so their probabilities may be added.",
        true,
      ],
      ["Independence makes distinct sequences disjoint.", false],
      [
        "Mutual exclusivity makes the flips within one sequence independent.",
        false,
      ],
    ],
    "Two different structural facts do separate jobs: independence handles the intersection of flip outcomes inside one sequence, while mutual exclusivity handles the union across distinct sequences. Neither property implies the other, so exchanging their roles would not justify the binomial probability formula.",
  ),
  makeQuestion(
    "cs109-lect5-q19",
    "medium",
    "For events \\(E_1,\\ldots,E_n\\), which statements correctly use De Morgan's law to reason about at least one event occurring?",
    [
      ["The complement of 'at least one occurs' is 'none occur.'", true],
      [
        "\\((E_1\\cup\\cdots\\cup E_n)^c=E_1^c\\cap\\cdots\\cap E_n^c\\).",
        true,
      ],
      [
        "If the events are mutually independent, then \\(P(\\bigcup_i E_i)=1-\\prod_i[1-P(E_i)]\\).",
        true,
      ],
      [
        "The at-least-one probability is \\(\\prod_i P(E_i)\\) whenever the events are independent.",
        false,
      ],
    ],
    "De Morgan's law converts a union into an intersection of complements, and mutual independence then makes that intersection easy to multiply. The direct product of the event probabilities describes every event occurring, not at least one, so it answers a different joint-event question.",
  ),
  makeQuestion(
    "cs109-lect5-q20",
    "medium",
    "Independent components fail with probabilities \\(q_1,\\ldots,q_n\\). A parallel system works whenever at least one component works. Which statements are correct?",
    [
      ["The system fails exactly when all components fail.", true],
      ["System failure has probability \\(\\prod_i q_i\\).", true],
      ["System success has probability \\(1-\\prod_i q_i\\).", true],
      [
        "The formula allows the components to have different failure probabilities.",
        true,
      ],
    ],
    "A parallel system's failure event is the intersection of all component failures, so independence yields the product of their possibly unequal failure probabilities. Taking the complement then gives system success; no identical-rate assumption is required because each factor retains its own component's rate.",
  ),
  makeQuestion(
    "cs109-lect5-q21",
    "medium",
    "A coin with head probability 0.6 is flipped ten times. Which statements correctly describe the event of exactly four heads?",
    [
      ["Its probability is \\(\\binom{10}{4}0.6^4 0.4^6\\).", true],
      [
        "There are \\(\\binom{10}{4}=210\\) mutually exclusive sequences in the event.",
        true,
      ],
      [
        "Its probability is \\(0.6^4 0.4^6\\) because order is fixed by the phrase 'exactly four.'",
        false,
      ],
      [
        "Its probability is \\(\\binom{10}{6}0.6^6 0.4^4\\) because choosing tails reverses the head and tail probabilities.",
        false,
      ],
    ],
    "Exactly four heads permits every placement of four heads among ten positions, producing 210 disjoint sequences. Each such sequence has four factors of 0.6 and six factors of 0.4; choosing the six tail positions gives the same coefficient but does not exchange which probability belongs to heads or tails.",
  ),
  makeQuestion(
    "cs109-lect5-q22",
    "medium",
    "A complete outcome of four coin flips is HHTT. Which statements are correct about HHTT, HTHT, and the event of exactly two heads?",
    [
      ["HHTT and HTHT are mutually exclusive complete outcomes.", true],
      ["Both outcomes belong to the event of exactly two heads.", true],
      [
        "For independent identically biased flips, both outcomes have probability \\(p^2(1-p)^2\\).",
        true,
      ],
      [
        "The two complete outcomes are independent events because their probabilities match.",
        false,
      ],
    ],
    "A single run cannot equal two distinct full sequences, so those sequence events are disjoint even though they contribute to the same head-count event. Equal probabilities do not establish independence: their intersection is zero while the product of their positive probabilities is not zero.",
  ),
  makeQuestion(
    "cs109-lect5-q23",
    "medium",
    "Each of \\(m\\) strings is hashed independently, and a string lands in bucket \\(i\\) with probability \\(p_i\\). Which statements correctly describe the event that bucket \\(i\\) receives at least one string?",
    [
      [
        "A particular string misses bucket \\(i\\) with probability \\(1-p_i\\).",
        true,
      ],
      [
        "All \\(m\\) strings miss bucket \\(i\\) with probability \\((1-p_i)^m\\).",
        true,
      ],
      ["Bucket \\(i\\) is nonempty with probability \\(1-(1-p_i)^m\\).", true],
      [
        "The derivation uses independence across string hashes and a complement for the at-least-one event.",
        true,
      ],
    ],
    "The nonempty-bucket event is easiest through its complement: every string misses that bucket. Independent hash trials make the miss probabilities multiply to \\((1-p_i)^m\\), and subtracting from one gives the probability that at least one string lands there.",
  ),
  makeQuestion(
    "cs109-lect5-q24",
    "medium",
    "Events \\(E,F,G\\) have probabilities 0.4, 0.5, and 0.3; pairwise intersections 0.2, 0.1, and 0.15; and triple intersection 0.05. What is \\(P(E\\cup F\\cup G)\\)?",
    [
      ["0.80", true],
      ["0.75", false],
      ["0.85", false],
      ["1.20", false],
    ],
    "Three-set inclusion-exclusion gives \\(0.4+0.5+0.3-0.2-0.1-0.15+0.05=0.80\\). Omitting the triple correction gives 0.75, while 0.85 adds too much back; adding only the three marginals gives 1.20 and counts overlapping probability mass multiple times.",
  ),
  makeQuestion(
    "cs109-lect5-q25",
    "hard",
    "Assume \\(A\\) and \\(B\\) are independent. Which steps form a valid proof that \\(A\\) and \\(B^c\\) are independent?",
    [
      [
        "Partition \\(A\\) as the disjoint union of \\(A\\cap B\\) and \\(A\\cap B^c\\).",
        true,
      ],
      ["Write \\(P(A\\cap B^c)=P(A)-P(A\\cap B)\\).", true],
      [
        "Substitute \\(P(A\\cap B)=P(A)P(B)\\) and factor \\(P(A)[1-P(B)]\\).",
        true,
      ],
      [
        "Recognize \\(1-P(B)=P(B^c)\\), obtaining \\(P(A\\cap B^c)=P(A)P(B^c)\\).",
        true,
      ],
    ],
    "The partition isolates the part of \\(A\\) outside \\(B\\), and independence supplies a product for the part inside \\(B\\). Algebra then turns the remainder into \\(P(A)P(B^c)\\), which is exactly the product criterion required for independence of \\(A\\) and \\(B^c\\).",
  ),
  makeQuestion(
    "cs109-lect5-q26",
    "hard",
    "Which conditions must hold for three events \\(A,B,C\\) to be mutually independent?",
    [
      [
        "\\(P(A\\cap B)=P(A)P(B)\\), and likewise for the other two pairs.",
        true,
      ],
      ["\\(P(A\\cap B\\cap C)=P(A)P(B)P(C)\\).", true],
      [
        "Every subset containing at least two of the events satisfies the corresponding product identity.",
        true,
      ],
      [
        "It is sufficient that the three marginal probabilities are equal.",
        false,
      ],
    ],
    "Mutual independence requires the product rule for all pairs and for the three-way intersection, equivalently for every relevant subset. Matching marginal probabilities says nothing about how events overlap, so equal rates cannot replace the joint-probability checks.",
  ),
  makeQuestion(
    "cs109-lect5-q27",
    "hard",
    "Two fair dice define \\(E=\\{D_1=1\\}\\), \\(F=\\{D_2=6\\}\\), and \\(G=\\{D_1+D_2=7\\}\\). Which statements explain why these events are pairwise independent but not mutually independent?",
    [
      [
        "Each event has probability \\(1/6\\), and every pair intersects in the single outcome \\((1,6)\\), so each pair has intersection probability \\(1/36\\).",
        true,
      ],
      [
        "All three also intersect in \\((1,6)\\), giving probability \\(1/36\\) rather than \\(1/216\\).",
        true,
      ],
      [
        "The events are mutually independent because every pair passes the product test.",
        false,
      ],
      [
        "The events fail pairwise independence because \\(G\\) is defined using both dice.",
        false,
      ],
    ],
    "The carefully chosen sum-seven event meets the product criterion with each die event separately, so structural involvement of both dice does not by itself create pairwise dependence. The three-way event is still just one ordered outcome, and its probability is too large to equal the product of all three marginals.",
  ),
  makeQuestion(
    "cs109-lect5-q28",
    "hard",
    "For \\(X\\sim\\text{Binomial}(10,0.6)\\), which expression equals \\(P(X=5)/P(X=4)\\)?",
    [
      ["\\(\\frac{6}{5}\\cdot\\frac{0.6}{0.4}=1.8\\)", true],
      ["\\(\\frac{5}{6}\\cdot\\frac{0.6}{0.4}=1.25\\)", false],
      ["\\(\\frac{6}{5}\\cdot\\frac{0.4}{0.6}=0.8\\)", false],
      ["\\(\\frac{\\binom{10}{5}}{\\binom{10}{4}}=1.2\\)", false],
    ],
    "Dividing the two binomial probabilities cancels most factors: the coefficient ratio is \\(\\binom{10}{5}/\\binom{10}{4}=6/5\\), one additional head contributes 0.6, and one fewer tail removes a factor 0.4. Combining both changes gives 1.8, so five heads are more likely than four.",
  ),
  makeQuestion(
    "cs109-lect5-q29",
    "hard",
    "Three independent jobs succeed with probabilities 0.8, 0.7, and 0.6. Which statements correctly compute the probability that at least two succeed?",
    [
      [
        "The exactly-two contribution is \\(0.8(0.7)(0.4)+0.8(0.3)(0.6)+0.2(0.7)(0.6)=0.452\\).",
        true,
      ],
      [
        "Adding the all-three contribution \\(0.8(0.7)(0.6)=0.336\\) gives 0.788.",
        true,
      ],
      [
        "The answer is \\(1-(0.2)(0.3)(0.4)=0.976\\), the probability of at least one success.",
        false,
      ],
      [
        "The answer is \\(\\binom{3}{2}(0.8)^2(0.2)\\), as though all jobs had the same success rate.",
        false,
      ],
    ],
    "At least two successes is the disjoint union of the three ways exactly two jobs succeed and the way all three succeed. Unequal job rates require keeping the appropriate success and failure factor for each named job; the simple binomial shortcut is unavailable, and the complement of all failures answers a broader event.",
  ),
  makeQuestion(
    "cs109-lect5-q30",
    "hard",
    "A parallel network has independent route success probabilities \\(p_1,\\ldots,p_n\\). Which changes must weakly increase the reliability \\(1-\\prod_i(1-p_i)\\)?",
    [
      ["Increasing any one \\(p_i\\) while holding the others fixed.", true],
      [
        "Adding another independent route with success probability greater than zero.",
        true,
      ],
      ["Replacing a route by one with a smaller failure probability.", true],
      ["Removing a route whose success probability is positive.", false],
    ],
    "Each listed improvement reduces the product of all failure probabilities, so its complement cannot decrease. Removing a useful route deletes a factor below one from the failure product, making simultaneous failure more likely and reliability lower unless that route had zero chance of success.",
  ),
  makeQuestion(
    "cs109-lect5-q31",
    "hard",
    "Four strings are hashed independently and uniformly into three labeled buckets. Which statements correctly calculate the probability that every bucket is nonempty?",
    [
      [
        "There are \\(3^4=81\\) equally likely bucket-assignment outcomes.",
        true,
      ],
      [
        "Inclusion-exclusion counts \\(3^4-\\binom{3}{1}2^4+\\binom{3}{2}1^4=36\\) onto assignments.",
        true,
      ],
      ["The desired probability is \\(36/81=4/9\\).", true],
      [
        "The events that individual buckets are empty overlap, so their probabilities cannot simply be subtracted once without correction.",
        true,
      ],
    ],
    "Every string independently chooses one of three buckets, giving 81 assignments. Excluding assignments that miss a named bucket initially removes overlaps more than once, so inclusion-exclusion adds back assignments confined to a single bucket; 36 assignments use all buckets, yielding probability \\(4/9\\).",
  ),
  makeQuestion(
    "cs109-lect5-q32",
    "hard",
    "Each of \\(m\\) independent strings goes to bucket 1 with probability \\(p\\) and bucket 2 with probability \\(1-p\\). Which statements correctly describe the probability that both buckets are nonempty?",
    [
      ["It is \\(1-p^m-(1-p)^m\\).", true],
      [
        "The two excluded cases are 'all strings in bucket 1' and 'all strings in bucket 2.'",
        true,
      ],
      [
        "It is \\([1-p^m][1-(1-p)^m]\\) because the nonempty-bucket events are independent.",
        false,
      ],
      [
        "It is \\(1-[p+(1-p)]^m\\) because every string must miss both buckets.",
        false,
      ],
    ],
    "With only two buckets, failure to occupy both means every string chose bucket 1 or every string chose bucket 2. Those cases are disjoint for positive \\(m\\), so their probabilities subtract directly from one; the two nonempty-bucket events are dependent because filling one bucket affects what remains for the other.",
  ),
  makeQuestion(
    "cs109-lect5-q33",
    "hard",
    "Events \\(A\\) and \\(B\\) satisfy \\(P(A)=0.4\\), \\(P(B)=0.5\\), and \\(P(A\\cap B)=0.2\\). Which conclusions are correct?",
    [
      ["The events are independent because \\(0.2=(0.4)(0.5)\\).", true],
      [
        "\\(P(A\\mid B)=0.4\\), so observing \\(B\\) leaves the probability of \\(A\\) unchanged.",
        true,
      ],
      ["\\(P(A\\cup B)=0.4+0.5-0.2=0.7\\).", true],
      [
        "The events are mutually exclusive because their intersection is smaller than either marginal.",
        false,
      ],
    ],
    "The joint probability exactly matches the product of the marginals, and the corresponding conditional probability equals the original marginal, establishing independence. Their positive intersection rules out mutual exclusivity, while inclusion-exclusion still uses that overlap to obtain the union probability.",
  ),
  makeQuestion(
    "cs109-lect5-q34",
    "hard",
    "Independent trials have possibly different success probabilities \\(p_1,\\ldots,p_n\\), and \\(X\\) counts their successes. Which statements are correct?",
    [
      ["\\(P(X=0)=\\prod_{i=1}^n(1-p_i)\\).", true],
      ["\\(P(X=n)=\\prod_{i=1}^n p_i\\).", true],
      ["\\(P(X=1)=\\sum_{i=1}^n p_i\\prod_{j\\ne i}(1-p_j)\\).", true],
      [
        "The usual binomial formula applies as written when all \\(p_i\\) share a common value \\(p\\).",
        true,
      ],
    ],
    "Independence supplies a product for any specified success-failure pattern, including the all-failure and all-success extremes. Exactly one success is a disjoint union over which trial succeeds, so its terms must retain the individual rates; only identical rates make those terms equal and collapse the distribution to the ordinary binomial formula.",
  ),
  makeQuestion(
    "cs109-lect5-q35",
    "hard",
    "A system has five independent components with success probabilities \\(p_1,\\ldots,p_5\\), and \\(X\\) is the number that succeed. Which expression gives \\(P(X\\ge1)\\)?",
    [
      ["\\(1-\\prod_{i=1}^5(1-p_i)\\)", true],
      ["\\(\\prod_{i=1}^5 p_i\\)", false],
      ["\\(\\sum_{i=1}^5 p_i\\prod_{j\\ne i}(1-p_j)\\)", false],
      ["\\(\\prod_{i=1}^5(1-p_i)\\)", false],
    ],
    "The complement of at least one success is the event that all five components fail, whose probability factors as \\(\\prod_i(1-p_i)\\). The other expressions respectively calculate all five succeeding, exactly one succeeding, and all five failing, so they are nearby but distinct events of the same system.",
  ),
];
