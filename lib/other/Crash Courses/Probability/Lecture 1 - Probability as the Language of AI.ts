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
    chapter: 1,
    difficulty,
    prompt,
    options: options.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

export const CrashCourseProbabilityL1Questions: Question[] = [
  // Outcomes, events, axioms, and event algebra
  makeQuestion(
    "crash-probability-l1-q61",
    "easy",
    "A fair die is rolled once. Which object is an event rather than a single outcome or the whole experiment?",
    [
      ["The set \\(\\{2,4,6\\}\\)", true],
      ["The result \\(4\\)", false],
      ["The act of rolling the die", false],
      ["The probability value \\(1/2\\)", false],
    ],
    "An event is a set of outcomes, so the even-result event is \\(\\{2,4,6\\}\\). The value 4 is one outcome, the roll is the random experiment, and \\(1/2\\) is a probability assigned to an event rather than an event itself.",
  ),
  makeQuestion(
    "crash-probability-l1-q62",
    "easy",
    "A single-label classifier is modeled with sample space \\(\\Omega=\\{\\text{cat},\\text{dog},\\text{other}\\}\\). Which claims follow from that modeling choice?",
    [
      [
        "Exactly one element of \\(\\Omega\\) is the modeled outcome for an image.",
        true,
      ],
      ["The three class probabilities must sum to 1.", true],
      [
        "Equal likelihood would additionally require \\(P(\\text{cat})=P(\\text{dog})=1/3\\).",
        false,
      ],
      [
        "The label other has probability zero because it is not a named animal.",
        false,
      ],
    ],
    "A single-label sample space treats its listed classes as mutually exclusive and exhaustive, so one class occurs and their probabilities sum to one. Exhaustiveness does not make outcomes equally likely, and an other class can carry substantial probability mass.",
  ),
  makeQuestion(
    "crash-probability-l1-q63",
    "medium",
    "An LLM can emit one of \\(\\{\\text{a},\\text{an},\\text{the},\\text{dog}\\}\\) with probabilities \\(0.20,0.10,0.45,0.25\\). Let \\(A\\) be “an article” and \\(B\\) be “a token containing the letter a.” Which statements are correct?",
    [
      ["\\(A=\\{\\text{a},\\text{an},\\text{the}\\}\\).", true],
      ["\\(B=\\{\\text{a},\\text{an}\\}\\).", true],
      ["\\(P(A)=0.75\\).", true],
      ["\\(A\\) and \\(B\\) are disjoint.", false],
    ],
    "Events group token outcomes by a property: the article event has mass \\(0.20+0.10+0.45=0.75\\), while the letter-a event contains a and an. Their intersection is \\(B\\), so calling the events disjoint ignores their shared outcomes.",
  ),
  makeQuestion(
    "crash-probability-l1-q64",
    "hard",
    "A bag contains four physically different tokens labeled A, A, B, and C. One token is drawn uniformly. Which probability calculations are valid?",
    [
      [
        "\\(P(\\text{label A})=2/4\\) because the four physical tokens are equally likely.",
        true,
      ],
      [
        "\\(P(\\text{label B or C})=2/4\\) because that event contains two physical outcomes.",
        true,
      ],
      [
        "\\(P(\\text{label A})=1/3\\) because there are three distinct labels.",
        false,
      ],
      [
        "Every event probability equals its number of labels divided by three.",
        false,
      ],
    ],
    "Favorable-over-total counting applies to the four equally likely physical tokens, not automatically to the three label names. Two tokens produce A and two produce B or C, whereas treating labels as equiprobable changes the experiment and gives the wrong denominator.",
  ),
  makeQuestion(
    "crash-probability-l1-q65",
    "medium",
    "Which statements are direct consequences or valid restatements of the probability axioms?",
    [
      ["Every event has nonnegative probability.", true],
      ["The complete sample space has probability 1.", true],
      [
        "Probabilities add across a finite collection of pairwise disjoint events.",
        true,
      ],
      ["The empty event has probability 0.", true],
    ],
    "Nonnegativity, normalization, and additivity over disjoint events are the foundational axioms. The empty-event rule follows because adding the empty event to the sample space changes nothing, forcing its probability to be zero under additivity.",
  ),
  makeQuestion(
    "crash-probability-l1-q66",
    "hard",
    "For events \\(A\\) and \\(B\\), \\(P(A)=0.55\\), \\(P(B)=0.40\\), and \\(P(A\\cap B)=0.20\\). Which results are correct?",
    [
      ["\\(P(A\\cup B)=0.75\\).", true],
      ["\\(P(A^c)=0.45\\).", true],
      ["The events are not disjoint.", true],
      [
        "\\(P(A\\cup B)=0.95\\) because union probabilities always add directly.",
        false,
      ],
    ],
    "Inclusion-exclusion subtracts the overlap once: \\(0.55+0.40-0.20=0.75\\), and the complement of A has mass \\(1-0.55=0.45\\). A positive intersection proves the events overlap, so direct addition would double-count that shared mass.",
  ),
  makeQuestion(
    "crash-probability-l1-q67",
    "easy",
    "In the sample space \\(\\Omega=\\{h,t\\}\\) for one coin flip, what is the singleton event that a head occurs?",
    [
      ["\\(\\{h\\}\\)", true],
      ["\\(h\\)", false],
      ["\\(\\Omega\\)", false],
      ["\\(P(h)\\)", false],
    ],
    "A singleton event is a set containing exactly one outcome, so it is written \\(\\{h\\}\\). The symbol h denotes the outcome itself, \\(\\Omega\\) contains both outcomes, and probability notation names a numerical measure rather than the event.",
  ),
  makeQuestion(
    "crash-probability-l1-q68",
    "medium",
    "A triage model assigns exactly one case to urgent, routine, or reject. Which pairs of properties describe these three modeled events?",
    [
      ["They are mutually exclusive because a case receives one label.", true],
      [
        "They are exhaustive if every case receives one of the three labels.",
        true,
      ],
      ["They are independent because their probabilities sum to one.", false],
      ["They are equiprobable because their events do not overlap.", false],
    ],
    "The labels are disjoint under a single-label rule, and they are exhaustive when no modeled case falls outside them. Neither disjointness nor normalization implies independence or equal probability; those are different claims about joint behavior and probability mass.",
  ),
  makeQuestion(
    "crash-probability-l1-q69",
    "hard",
    "A dataset event \\(A\\) means “contains a person” and \\(B\\) means “contains a bicycle.” Their probabilities are \\(0.60\\), \\(0.35\\), and \\(P(A\\cup B)=0.75\\). Which conclusions follow?",
    [
      ["\\(P(A\\cap B)=0.20\\).", true],
      ["\\(P(A^c\\cap B^c)=0.25\\).", true],
      ["The events can occur together for one image.", true],
      [
        "The events are disjoint because the union probability is below one.",
        false,
      ],
    ],
    "Rearranging inclusion-exclusion gives the overlap \\(0.60+0.35-0.75=0.20\\), and the complement of the union has probability \\(1-0.75=0.25\\). A positive overlap means person and bicycle can co-occur, so the union being below one says nothing about disjointness.",
  ),
  makeQuestion(
    "crash-probability-l1-q70",
    "easy",
    "Which event identities hold for any events \\(A\\) and \\(B\\)?",
    [
      ["\\((A^c)^c=A\\).", true],
      ["\\(A\\cup A^c=\\Omega\\).", true],
      ["\\(A\\cap A^c=\\varnothing\\).", true],
      ["\\((A\\cup B)^c=A^c\\cap B^c\\).", true],
    ],
    "A complement reverses membership, so complementing twice restores A, while A together with its complement covers the universe and their intersection is empty. De Morgan's law also says that being outside a union means being outside both events.",
  ),
  makeQuestion(
    "crash-probability-l1-q71",
    "medium",
    "A safety event \\(F\\) means “the first check fails” and \\(S\\) means “the second check fails.” Which descriptions match the notation?",
    [
      ["No check fails is \\((F\\cup S)^c=F^c\\cap S^c\\).", true],
      ["At least one check fails is \\(F\\cup S\\).", true],
      ["Exactly one check fails is \\(F\\cap S\\).", false],
      ["Both checks fail is \\(F\\cup S\\).", false],
    ],
    "At least one failure is a union, and its complement is the intersection of both successes by De Morgan's law. The intersection \\(F\\cap S\\) means both fail; exactly one failure would need the two non-overlapping cases \\((F\\cap S^c)\\cup(F^c\\cap S)\\).",
  ),
  makeQuestion(
    "crash-probability-l1-q72",
    "hard",
    "A four-class model reports probabilities spam 0.42, promotion 0.28, social 0.18, and primary 0.12. Let \\(M\\) be “marketing” = {spam, promotion} and \\(N\\) be “not spam.” Which statements are correct?",
    [
      ["\\(P(M)=0.70\\).", true],
      ["\\(P(N)=0.58\\).", true],
      ["\\(P(M\\cap N)=0.28\\).", true],
      ["\\(M\\) and \\(N\\) are complements.", false],
    ],
    "Marketing combines spam and promotion, while not-spam combines the other three classes. Their overlap is promotion with mass 0.28, so these events are not complements even though each can be formed from the same categorical sample space.",
  ),

  // Counting large sample spaces
  makeQuestion(
    "crash-probability-l1-q73",
    "easy",
    "A four-digit PIN allows digits 0–9, permits repetition, and treats different orders as different PINs. How many PINs are possible?",
    [
      ["\\(10^4=10{,}000\\)", true],
      ["\\(4^{10}\\)", false],
      ["\\(10!/(10-4)!\\)", false],
      ["\\({10\\choose4}\\)", false],
    ],
    "Each of four ordered positions has ten choices, so the multiplication principle gives \\(10\\times10\\times10\\times10=10^4\\). A permutation would forbid repeated digits, while a combination would ignore the positions that distinguish PINs.",
  ),
  makeQuestion(
    "crash-probability-l1-q74",
    "easy",
    "Which questions should be answered before selecting a counting formula for a sequence or group?",
    [
      ["Does the definition of an outcome make order matter?", true],
      ["Can an item be selected more than once?", true],
      ["Is the final probability larger than one?", false],
      ["Does the random variable have a Gaussian density?", false],
    ],
    "Order and repetition determine whether powers, permutations, or combinations describe the outcomes. Probability normalization and distribution family are important elsewhere, but they do not resolve what counts as a distinct arrangement in the sample space.",
  ),
  makeQuestion(
    "crash-probability-l1-q75",
    "medium",
    "Match each sampling task with its count. Which pairings are correct?",
    [
      [
        "Length-\\(k\\) strings from \\(n\\) symbols with repetition: \\(n^k\\).",
        true,
      ],
      [
        "Assigning \\(k\\) distinct roles from \\(n\\) people: \\(n!/(n-k)!\\).",
        true,
      ],
      [
        "Choosing an unordered \\(k\\)-person committee: \\({n\\choose k}\\).",
        true,
      ],
      [
        "Choosing an ordered sequence without replacement: \\({n\\choose k}\\).",
        false,
      ],
    ],
    "Powers count ordered choices with replacement, permutations count ordered choices without replacement, and combinations remove the ordering among selected items. The final pairing loses information about order and therefore undercounts ordered sequences.",
  ),
  makeQuestion(
    "crash-probability-l1-q76",
    "hard",
    "A security code has three distinct letters chosen from 26, followed by two distinct digits chosen from 10. Repetition is forbidden within each part. Which expressions give the number of codes?",
    [
      ["\\((26\\cdot25\\cdot24)(10\\cdot9)\\)", true],
      ["\\(P(26,3)P(10,2)\\)", true],
      ["\\({26\\choose3}{10\\choose2}\\)", false],
      ["\\(26^3 10^2\\)", false],
    ],
    "Positions make order matter, and the no-repetition rule reduces the choices after each selection, giving two permutation factors. Combinations would forget position order, while powers would allow the same letter or digit to reappear.",
  ),
  makeQuestion(
    "crash-probability-l1-q77",
    "medium",
    "A team of eight people must fill president, treasurer, and secretary roles. Which statements correctly define and count the outcomes?",
    [
      [
        "One outcome records both the selected people and which role each holds.",
        true,
      ],
      ["There are \\(8\\cdot7\\cdot6=336\\) assignments.", true],
      ["Swapping president and treasurer creates a different outcome.", true],
      ["The count is \\(P(8,3)=8!/(8-3)!\\).", true],
    ],
    "The roles are distinct, so the outcome definition includes assignment as well as membership. The multiplication principle and permutation formula both give 336, and a role swap changes the ordered assignment even when the same three people participate.",
  ),
  makeQuestion(
    "crash-probability-l1-q78",
    "hard",
    "From ten labeled examples, a data curator chooses three for an unordered audit set. Which statements are correct?",
    [
      ["There are \\({10\\choose3}=120\\) possible audit sets.", true],
      [
        "The ordered count \\(10\\cdot9\\cdot8\\) counts each audit set \\(3!\\) times.",
        true,
      ],
      [
        "Dividing the ordered count by \\(3!\\) removes the internal order of the selected examples.",
        true,
      ],
      [
        "There are \\(10^3\\) sets because a selected example can appear repeatedly.",
        false,
      ],
    ],
    "An audit set contains three distinct examples and has no positions, so combinations are appropriate. The permutation count distinguishes all six orders of the same three examples, while \\(10^3\\) additionally allows repeats that are not part of a set.",
  ),
  makeQuestion(
    "crash-probability-l1-q79",
    "easy",
    "How many unordered groups of three can be selected from eight distinct objects?",
    [
      ["\\({8\\choose3}=56\\)", true],
      ["\\(8^3=512\\)", false],
      ["\\(8\\cdot7\\cdot6=336\\)", false],
      ["\\(3^8=6{,}561\\)", false],
    ],
    "Because selection order does not distinguish a group, the count is \\({8\\choose3}=8!/(3!5!)=56\\). The value 336 counts ordered triples, and the power expressions describe different repeated-choice experiments.",
  ),
  makeQuestion(
    "crash-probability-l1-q80",
    "medium",
    "A model samples two tokens from a vocabulary of size 5. Which counts correspond to the stated sampling rule?",
    [
      ["With replacement and order recorded: \\(5^2=25\\).", true],
      ["Without replacement and order recorded: \\(5\\cdot4=20\\).", true],
      ["With replacement and order recorded: \\({5\\choose2}=10\\).", false],
      ["Without replacement and order recorded: \\(5^2=25\\).", false],
    ],
    "Two sequential token positions make order relevant. Replacement keeps five choices at the second position, while no replacement leaves four; the combination count instead describes an unordered pair and cannot represent token sequences.",
  ),
  makeQuestion(
    "crash-probability-l1-q81",
    "hard",
    "Five cards are dealt from a standard 52-card deck without replacement, and every five-card hand is equally likely. Which statements support computing the probability of exactly four aces?",
    [
      ["The denominator is \\({52\\choose5}\\).", true],
      ["The favorable count is \\({4\\choose4}{48\\choose1}\\).", true],
      [
        "The probability is the favorable count divided by the denominator because hands are equally likely.",
        true,
      ],
      [
        "The denominator is \\(52^5\\) because each draw has 52 choices.",
        false,
      ],
    ],
    "A hand is unordered and has no repeated physical card, so combinations count both the full sample space and favorable hands. The \\(52^5\\) expression describes ordered draws with replacement and therefore does not match the stated experiment.",
  ),
  makeQuestion(
    "crash-probability-l1-q82",
    "easy",
    "Which factorial identities are correct for positive integers in their valid ranges?",
    [
      ["\\(n!=n(n-1)!\\).", true],
      ["\\(0!=1\\).", true],
      ["\\(P(n,n)=n!\\).", true],
      ["\\({n\\choose k}={n\\choose n-k}\\).", true],
    ],
    "The recursive definition of factorial includes the convention \\(0!=1\\), and arranging all n objects gives \\(n!\\) permutations. Choosing k included objects is in one-to-one correspondence with choosing the \\(n-k\\) excluded objects, which explains the combination symmetry.",
  ),
  makeQuestion(
    "crash-probability-l1-q83",
    "medium",
    "A fair six-sided die is rolled twice. Which calculations correctly use equally likely outcome counting?",
    [
      ["There are \\(6^2=36\\) ordered outcome pairs.", true],
      ["The probability of a sum of 7 is \\(6/36=1/6\\).", true],
      [
        "There are 11 equally likely possible sums, so every sum has probability \\(1/11\\).",
        false,
      ],
      [
        "The unordered pair {1,6} is as likely as {3,4} only because each names one pair.",
        false,
      ],
    ],
    "The 36 ordered die pairs are equally likely, and six of them sum to seven. Sums and unordered descriptions combine different numbers of elementary outcomes, so counting those labels as though they were equally likely produces incorrect probabilities.",
  ),
  makeQuestion(
    "crash-probability-l1-q84",
    "hard",
    "A generated sequence has length 5 and uses a vocabulary of 20 tokens. Which statements correctly distinguish possible sequence counts?",
    [
      ["Allowing token reuse gives \\(20^5\\) ordered sequences.", true],
      ["Forbidding reuse gives \\(P(20,5)\\) ordered sequences.", true],
      [
        "If exactly two positions are fixed in advance, the remaining unrestricted positions give \\(20^3\\) completions.",
        true,
      ],
      [
        "Choosing five token types with \\({20\\choose5}\\) counts every length-5 sequence when order matters.",
        false,
      ],
    ],
    "Sequence positions make order part of the outcome, while replacement determines whether a token may recur. Fixing two positions leaves three independent choices, but a combination records only an unordered set of distinct token types and omits most sequences.",
  ),

  // Random variables, probability mass, and density
  makeQuestion(
    "crash-probability-l1-q85",
    "easy",
    "Which description correctly defines a random variable?",
    [
      ["A function that maps each outcome in a sample space to a value", true],
      ["A probability selected randomly after an outcome occurs", false],
      ["A set that must contain every outcome in the sample space", false],
      ["A number that is uncertain because it has no distribution", false],
    ],
    "A random variable assigns a value to each underlying outcome; the uncertainty comes from not yet knowing which outcome will occur. It need not itself be a probability, it need not contain outcomes as a set, and its behavior is described through a distribution.",
  ),
  makeQuestion(
    "crash-probability-l1-q86",
    "easy",
    "Which quantities are naturally modeled as discrete random variables?",
    [
      ["The next token selected from a finite vocabulary", true],
      ["The number of failed requests in an hour", true],
      ["The exact time required for a server response", false],
      ["The exact electrical voltage measured by a sensor", false],
    ],
    "Token identity and a count take values from countable sets, making them discrete. Time and voltage are usually modeled on continuous ranges, even though a digital instrument may later round their recorded values.",
  ),
  makeQuestion(
    "crash-probability-l1-q87",
    "medium",
    "A proposed probability mass function has \\(p(0)=0.15\\), \\(p(1)=0.50\\), \\(p(2)=0.35\\), and zero elsewhere. Which statements are correct?",
    [
      ["The probabilities are nonnegative.", true],
      ["The listed masses sum to one.", true],
      ["\\(P(X\\ge1)=0.85\\).", true],
      [
        "\\(P(X=1)=0\\) because a single exact value has zero probability.",
        false,
      ],
    ],
    "The proposed masses satisfy the PMF requirements and give \\(P(X\\ge1)=0.50+0.35=0.85\\). Exact values can have positive mass for a discrete variable; the zero-probability-at-a-point rule belongs to continuous density models.",
  ),
  makeQuestion(
    "crash-probability-l1-q88",
    "hard",
    "Two fair coins are flipped. Define \\(X\\) as the number of heads. Which statements correctly derive the distribution of \\(X\\)?",
    [
      ["\\(P(X=1)=2/4\\) because HT and TH map to the same value.", true],
      ["\\(P(X=0)=P(X=2)=1/4\\).", true],
      [
        "The values 0, 1, and 2 are equally likely because there are three values.",
        false,
      ],
      [
        "X is the sample space rather than a function on coin-flip outcomes.",
        false,
      ],
    ],
    "The four equally likely outcomes HH, HT, TH, and TT are mapped by X to head counts, and two outcomes map to one. Random-variable values need not be equiprobable, and X is the mapping rather than the underlying coin-flip sample space.",
  ),
  makeQuestion(
    "crash-probability-l1-q89",
    "medium",
    "Which statements correctly contrast a probability mass function (PMF) and a probability density function (PDF)?",
    [
      ["A PMF assigns probability directly to discrete values.", true],
      [
        "A PDF gives interval probabilities through area under the density.",
        true,
      ],
      [
        "A continuous density can exceed 1 at some points if its total area is 1.",
        true,
      ],
      [
        "For a continuous variable, changing the density at one isolated point does not change interval probabilities.",
        true,
      ],
    ],
    "Mass is attached to individual discrete values, whereas continuous probability is accumulated as area. A density is not itself a point probability, so its height may exceed one on a narrow region and isolated point changes contribute zero area.",
  ),
  makeQuestion(
    "crash-probability-l1-q90",
    "hard",
    "A continuous variable has density \\(f(x)=1/4\\) for \\(0\\le x\\le4\\) and zero elsewhere. Which results are correct?",
    [
      ["\\(P(1\\le X\\le3)=1/2\\).", true],
      ["\\(P(X=2)=0\\).", true],
      ["The total area under \\(f\\) is 1.", true],
      [
        "\\(P(1\\le X\\le3)=1/4\\) because the density height is \\(1/4\\).",
        false,
      ],
    ],
    "The interval from 1 to 3 has width two, so its rectangular area is \\(2\\times1/4=1/2\\); the full support has area \\(4\\times1/4=1\\). A single point has zero width and therefore zero probability in this continuous model.",
  ),
  makeQuestion(
    "crash-probability-l1-q91",
    "easy",
    "For a genuinely continuous response-time variable \\(T\\), what is \\(P(T=2.000000\\text{ seconds})\\)?",
    [
      ["0", true],
      ["The density evaluated at 2", false],
      ["1 because the time can be measured", false],
      ["The width of the support", false],
    ],
    "An exact point has zero width, so a continuous random variable assigns it probability zero even when the density there is positive. Density is probability per unit on a local scale; only integrating it across an interval produces a probability.",
  ),
  makeQuestion(
    "crash-probability-l1-q92",
    "medium",
    "A loading time is uniform on the interval from 2 to 10 seconds. Which calculations are correct?",
    [
      ["The density on the interval is \\(1/8\\).", true],
      ["\\(P(4\\le T\\le6)=2/8=0.25\\).", true],
      ["\\(P(T=5)=1/8\\).", false],
      [
        "The interval from 2 to 4 has more probability than the equal-width interval from 7 to 9.",
        false,
      ],
    ],
    "A uniform density spreads unit area evenly across width eight, and any subinterval probability is its width divided by eight. The value \\(1/8\\) is a density height, not point mass, and equal-width intervals inside the support have equal probability.",
  ),
  makeQuestion(
    "crash-probability-l1-q93",
    "hard",
    "A categorical variable \\(Y\\) has class probabilities \\((0.55,0.25,0.20)\\), while a continuous score \\(S\\) has density \\(f\\). Which statements are correct?",
    [
      [
        "Sampling Y returns one class according to the three probability masses.",
        true,
      ],
      [
        "Computing \\(P(a\\le S\\le b)\\) requires area under \\(f\\) from a to b.",
        true,
      ],
      [
        "The three masses for Y must sum to one, and the total area under f must also equal one.",
        true,
      ],
      [
        "Both \\(P(Y=1)\\) and \\(P(S=1)\\) must be zero because they name exact values.",
        false,
      ],
    ],
    "A categorical distribution assigns direct mass to class values, while a continuous model assigns probability through integrated density. Both distributions normalize to one, but only the continuous exact point is forced to have zero probability.",
  ),
  makeQuestion(
    "crash-probability-l1-q94",
    "easy",
    "Which distribution-to-task pairings are appropriate at an introductory level?",
    [
      ["Bernoulli: whether a transaction is fraudulent", true],
      ["Categorical: which one of several tokens comes next", true],
      ["PMF: probabilities for a discrete reward count", true],
      ["PDF: a model for a continuously measured temperature", true],
    ],
    "Bernoulli models one binary trial, categorical models one choice among several classes, and a PMF describes countable values. A PDF is suitable for idealized continuous measurements where interval area, rather than point mass, determines probability.",
  ),
  makeQuestion(
    "crash-probability-l1-q95",
    "medium",
    "Let \\(X\\) be a fair die result and define \\(Y=2X+1\\). Which statements are correct?",
    [
      ["Y can take values \\(3,5,7,9,11,13\\).", true],
      ["\\(P(Y=9)=P(X=4)=1/6\\).", true],
      [
        "Y has the same numerical values as X because both depend on one roll.",
        false,
      ],
      [
        "Y is not a random variable because its value is computed deterministically from X.",
        false,
      ],
    ],
    "A deterministic function of a random outcome is still a random variable, and transforming each die value produces the six listed odd values. Probability transfers through the mapping, so the event \\(Y=9\\) is exactly the event \\(X=4\\).",
  ),
  makeQuestion(
    "crash-probability-l1-q96",
    "hard",
    "A density equals 2 on \\([0,0.25]\\), 1 on \\((0.25,0.75]\\), and 0 elsewhere. Which statements are correct?",
    [
      [
        "The total area is \\(2(0.25)+1(0.50)=1\\), so this is a valid density.",
        true,
      ],
      ["\\(P(X\\le0.25)=0.50\\).", true],
      ["\\(P(0.25<X\\le0.75)=0.50\\).", true],
      [
        "The first interval is impossible because a density value of 2 exceeds 1.",
        false,
      ],
    ],
    "Density height may exceed one provided the integrated area remains one; here the two rectangles each contribute probability 0.50. Confusing density with probability would wrongly reject a narrow high-density region even though its area is valid.",
  ),

  // Expectation, variance, and risk
  makeQuestion(
    "crash-probability-l1-q97",
    "easy",
    "What is the expected value of a fair six-sided die roll?",
    [
      ["\\(3.5\\)", true],
      ["\\(3\\)", false],
      ["\\(6\\)", false],
      ["\\(1/6\\)", false],
    ],
    "The expectation is \\((1+2+3+4+5+6)/6=3.5\\), the probability-weighted average of the possible values. It need not itself be a possible one-roll result, so rounding to 3 or selecting a probability confuses different quantities.",
  ),
  makeQuestion(
    "crash-probability-l1-q98",
    "easy",
    "An action returns 0 with probability 0.7 and 10 with probability 0.3. Which statements are correct?",
    [
      ["Its expected reward is \\(0(0.7)+10(0.3)=3\\).", true],
      [
        "Over many independent repetitions, the average reward should tend toward 3.",
        true,
      ],
      [
        "The expected value \\(3\\) is the reward produced by a single execution.",
        false,
      ],
      [
        "The modal reward is \\(10\\) because it lies above \\(\\mathbb{E}[R]\\).",
        false,
      ],
    ],
    "Expectation weights each reward by its probability, yielding 3 as a long-run average rather than an available single-trial payoff. Reward 0 is actually more likely than reward 10, showing that expectation is not the same as the mode.",
  ),
  makeQuestion(
    "crash-probability-l1-q99",
    "medium",
    "Random losses \\(X\\) and \\(Y\\) may be dependent. Which statements about \\(X+Y\\) are correct when both expectations exist?",
    [
      ["\\(\\mathbb{E}[X+Y]=\\mathbb{E}[X]+\\mathbb{E}[Y]\\).", true],
      ["Independence is not required for this addition rule.", true],
      ["The rule extends to a finite sum of dependent losses.", true],
      [
        "Dependence forces \\(\\mathbb{E}[X+Y]\\) to include a covariance term.",
        false,
      ],
    ],
    "Linearity of expectation applies to sums regardless of dependence, so expected component losses add directly. Covariance matters for the variance of a sum, not for its expectation, and inserting it here mixes two different summary measures.",
  ),
  makeQuestion(
    "crash-probability-l1-q100",
    "hard",
    "Let \\(X\\) be 1 for a head and 0 for a tail, and let \\(Y=X\\) for the same fair coin flip. Which statements are correct?",
    [
      ["\\(\\mathbb{E}[XY]=1/2\\) because \\(XY=X\\).", true],
      [
        "\\(\\mathbb{E}[X]\\mathbb{E}[Y]=1/4\\), so the product rule fails here.",
        true,
      ],
      [
        "\\(\\mathbb{E}[XY]=1/4\\) because expectations always multiply.",
        false,
      ],
      [
        "X and Y are independent because they have the same marginal distribution.",
        false,
      ],
    ],
    "The variables are perfectly dependent copies, so \\(XY=X\\) and its expectation is one half. Multiplying separate expectations is justified under independence, not merely because two variables share the same Bernoulli marginal distribution.",
  ),
  makeQuestion(
    "crash-probability-l1-q101",
    "medium",
    "A game costs 4 credits and pays 10 with probability 0.4 or 0 with probability 0.6. Which statements are correct?",
    [
      ["The expected payout is 4 credits.", true],
      ["The expected net gain is 0 credits.", true],
      ["The game is fair in the expected-value sense.", true],
      [
        "A risk-averse player may still reject the game because most plays lose the entry cost.",
        true,
      ],
    ],
    "The expected payout equals the entry cost, so the expected net gain is zero and the game is actuarially fair. Expected value alone does not determine preference: because most plays lose four credits, a risk-averse player can reasonably reject the game despite its zero mean.",
  ),
  makeQuestion(
    "crash-probability-l1-q102",
    "hard",
    "A reward is 0 or 10 with equal probability. Which variance calculations and interpretations are correct?",
    [
      ["The mean is 5.", true],
      ["The variance is \\(0.5(0-5)^2+0.5(10-5)^2=25\\).", true],
      ["The standard deviation is 5 reward units.", true],
      [
        "The variance is \\(\\sqrt{25}=5\\) because it uses the same units as reward.",
        false,
      ],
    ],
    "Both outcomes lie five units from the mean, so the average squared deviation is 25 and its square root is 5. Variance uses squared reward units, whereas standard deviation returns to the original units; confusing the two changes both magnitude and interpretation.",
  ),
  makeQuestion(
    "crash-probability-l1-q103",
    "easy",
    "A batch contains equally weighted losses 0.2, 0.6, 0.4, and 0.8. What empirical expected loss is minimized when the batch mean is used?",
    [
      ["\\(0.50\\)", true],
      ["\\(2.00\\)", false],
      ["\\(0.20\\)", false],
      ["\\(0.80\\)", false],
    ],
    "The equally weighted empirical expectation is the arithmetic mean, \\((0.2+0.6+0.4+0.8)/4=0.50\\). The sum 2.00 omits normalization, while the minimum and maximum report individual examples rather than average batch loss.",
  ),
  makeQuestion(
    "crash-probability-l1-q104",
    "medium",
    "Action A pays 5 for certain. Action B pays 0 or 10 with equal probability. Which comparisons are correct?",
    [
      ["Both actions have expected reward 5.", true],
      ["Action B has greater variance than Action A.", true],
      ["Equal expectation implies identical reward distributions.", false],
      [
        "A risk-neutral and a risk-averse decision maker must rank the actions identically.",
        false,
      ],
    ],
    "The actions have the same mean, but A has zero variance and B has outcomes spread around that mean. A risk-neutral criterion using only expectation is indifferent, whereas risk preference can distinguish the certain and variable rewards.",
  ),
  makeQuestion(
    "crash-probability-l1-q105",
    "hard",
    "Let \\(X\\) count heads in two fair coin flips and \\(Y\\) count tails in those same flips. Which statements are correct?",
    [
      ["\\(X+Y=2\\) for every outcome.", true],
      ["\\(\\mathbb{E}[X]+\\mathbb{E}[Y]=2\\).", true],
      ["X and Y are dependent even though their expectations add.", true],
      [
        "\\(\\operatorname{Var}(X+Y)=\\operatorname{Var}(X)+\\operatorname{Var}(Y)\\) because variance always adds.",
        false,
      ],
    ],
    "Heads and tails must sum to two, making the variables perfectly negatively dependent, while linearity still gives an expected sum of two. Their covariance cancels the separate variances, so variance additivity without a covariance term is not valid here.",
  ),
  makeQuestion(
    "crash-probability-l1-q106",
    "easy",
    "Which quantities are naturally expressed as expectations in AI systems?",
    [
      ["Average training loss over sampled data", true],
      ["Expected future return of an action", true],
      ["Mean prediction under a probabilistic regression model", true],
      ["Average utility of a decision under uncertain outcomes", true],
    ],
    "Each quantity averages a value—loss, return, prediction, or utility—under a relevant probability distribution. The distributions differ across applications, but the same weighted-average operation connects learning, prediction, and decision-making.",
  ),
  makeQuestion(
    "crash-probability-l1-q107",
    "medium",
    "A latency variable is measured in milliseconds. Which unit statements are correct?",
    [
      ["Its expectation is measured in milliseconds.", true],
      ["Its variance is measured in squared milliseconds.", true],
      ["Its standard deviation is measured in squared milliseconds.", false],
      ["Its probability mass is measured in milliseconds.", false],
    ],
    "Expectation and standard deviation use the variable's original units, while variance averages squared deviations and therefore has squared units. Probabilities are dimensionless, so attaching milliseconds to probability mass confuses value units with probability.",
  ),
  makeQuestion(
    "crash-probability-l1-q108",
    "hard",
    "A total loss is \\(L=0.7L_{\\text{task}}+0.2L_{\\text{safety}}+0.1L_{\\text{style}}\\). Which expectation statements are correct?",
    [
      [
        "\\(\\mathbb{E}[L]=0.7\\mathbb{E}[L_{\\text{task}}]+0.2\\mathbb{E}[L_{\\text{safety}}]+0.1\\mathbb{E}[L_{\\text{style}}]\\).",
        true,
      ],
      [
        "The identity holds even when the three losses are dependent on the same examples.",
        true,
      ],
      [
        "Changing a coefficient changes that component's contribution to expected total loss.",
        true,
      ],
      [
        "Dependence requires adding covariance terms to \\(\\mathbb{E}[L]\\), even though the coefficients already weight each loss.",
        false,
      ],
    ],
    "Linearity passes constants through expectation and adds the component expectations without any independence or distribution-family assumption. Dependence affects joint variability, but it does not alter this weighted expected-loss identity.",
  ),

  // AI probability interpretation and uncertainty
  makeQuestion(
    "crash-probability-l1-q109",
    "easy",
    "Two classifiers both display “cat,” but one assigns cat probability 0.95 and the other 0.52. What is the most accurate comparison?",
    [
      [
        "The same decision label can hide very different predictive distributions.",
        true,
      ],
      [
        "The models have identical uncertainty because the displayed label matches.",
        false,
      ],
      ["The 0.52 model must be more accurate on future images.", false],
      ["A probability below 1 is not a valid classifier output.", false],
    ],
    "An argmax decision retains only the largest-probability class and discards how probability was distributed among alternatives. The probabilities are valid without equaling one individually, and one example does not establish either model's future accuracy.",
  ),
  makeQuestion(
    "crash-probability-l1-q110",
    "easy",
    "Which distinctions between single-label and multi-label prediction are correct?",
    [
      [
        "A single-label softmax commonly represents one categorical distribution across classes.",
        true,
      ],
      [
        "A multi-label model can use separate Bernoulli probabilities because several labels may co-occur.",
        true,
      ],
      ["Every multi-label probability must sum with the others to one.", false],
      [
        "Single-label classes are independent events merely because they have separate probabilities.",
        false,
      ],
    ],
    "A categorical model allocates one unit of mass among mutually exclusive classes, while separate Bernoulli outputs ask distinct yes/no questions that may all be true. Neither setup makes class events statistically independent by default.",
  ),
  makeQuestion(
    "crash-probability-l1-q111",
    "medium",
    "A calibrated weather model issues a 0.70 rain probability on many comparable days. Which observations are consistent with calibration?",
    [
      ["Rain occurs on roughly 70% of those days.", true],
      ["The model can be wrong on individual 0.70 days.", true],
      [
        "Calibration is assessed across groups of comparable probability forecasts.",
        true,
      ],
      [
        "The model must predict rain with probability 1 on every day that eventually rains.",
        false,
      ],
    ],
    "Calibration compares stated probabilities with empirical frequencies over repeated comparable forecasts, not with certainty after the outcome is known. A well-calibrated 0.70 forecast is expected to fail about 30% of the time on that subset.",
  ),
  makeQuestion(
    "crash-probability-l1-q112",
    "hard",
    "Among 400 predictions made at reported confidence 0.80, 300 are correct. Which conclusions are justified for this confidence bin?",
    [
      ["The observed accuracy is \\(300/400=0.75\\).", true],
      [
        "The model is overconfident by 0.05 in this bin because 0.80 exceeds 0.75.",
        true,
      ],
      [
        "The model is underconfident by \\(0.80-0.75=0.05\\) because 300 correct predictions is a large count.",
        false,
      ],
      [
        "This single bin proves the model is perfectly calibrated at every confidence level.",
        false,
      ],
    ],
    "Calibration compares the 0.80 forecast with the observed 0.75 frequency, indicating five percentage points of overconfidence in this bin. The raw count alone has no direction, and performance in one bin cannot establish global calibration.",
  ),
  makeQuestion(
    "crash-probability-l1-q113",
    "medium",
    "An LLM produces a categorical next-token distribution before decoding. Which statements correctly describe that object?",
    [
      [
        "Every vocabulary token is a possible value of the next-token random variable.",
        true,
      ],
      ["The token probabilities are nonnegative and sum to one.", true],
      [
        "A decoder can sample from or take the argmax of the same distribution.",
        true,
      ],
      [
        "Keeping the full distribution preserves uncertainty information that an argmax label discards.",
        true,
      ],
    ],
    "The model distribution and the decoder are distinct stages: the distribution normalizes mass over token outcomes, and a decoding rule converts it to a token. A decoder may transform or filter probabilities before selection, but it does not rewrite the original model output that existed beforehand.",
  ),
  makeQuestion(
    "crash-probability-l1-q114",
    "hard",
    "Which statements correctly distinguish uncertainty summaries for predictions?",
    [
      ["Variance summarizes squared spread for numerical outcomes.", true],
      [
        "Entropy summarizes concentration or dispersion of probability mass across possible outcomes.",
        true,
      ],
      [
        "Calibration compares stated probabilities with observed frequencies.",
        true,
      ],
      [
        "A class label alone contains the same uncertainty information as the full probability distribution.",
        false,
      ],
    ],
    "Variance, entropy, and calibration answer different questions about numerical spread, distributional concentration, and long-run probability reliability. Collapsing a distribution to one label discards the alternative masses needed for those uncertainty judgments.",
  ),
  makeQuestion(
    "crash-probability-l1-q115",
    "easy",
    "In a diffusion model, which quantity is most naturally treated as a continuous random variable?",
    [
      ["A Gaussian noise value added to a pixel or latent coordinate", true],
      ["The name of the image file", false],
      ["The fixed number of model layers", false],
      ["The set of all prompts used for evaluation", false],
    ],
    "Gaussian noise coordinates are modeled on continuous numerical ranges and described by densities. File names and prompt sets are categorical objects, while a fixed architecture count is a constant rather than an uncertain quantity in the stated experiment.",
  ),
  makeQuestion(
    "crash-probability-l1-q116",
    "medium",
    "A medical model assigns a patient disease probability 0.30. Which interpretations are warranted without additional evidence?",
    [
      [
        "The model represents disease as possible rather than impossible.",
        true,
      ],
      [
        "A treatment decision can combine this probability with costs, benefits, and thresholds.",
        true,
      ],
      ["Exactly 30% of this individual patient's body is diseased.", false],
      [
        "The number is automatically calibrated because it lies between zero and one.",
        false,
      ],
    ],
    "The value is a model probability for an uncertain event and can inform a decision rule together with utilities or consequences. It is not a physical fraction of a person, and mathematical validity alone does not establish empirical calibration.",
  ),
  makeQuestion(
    "crash-probability-l1-q117",
    "hard",
    "A classifier estimates \\(P(\\text{disease})=0.20\\). Treating costs in the same units, a missed disease costs 100, a false alarm costs 5, and correct decisions cost 0. Which statements are correct?",
    [
      ["Predicting no disease has expected cost \\(0.20(100)=20\\).", true],
      ["Predicting disease has expected cost \\(0.80(5)=4\\).", true],
      [
        "Under these costs, predicting disease minimizes expected cost despite probability 0.20 being below 0.50.",
        true,
      ],
      [
        "Argmax classification must minimize expected cost for every cost structure.",
        false,
      ],
    ],
    "A decision depends on outcome probabilities and their consequences: the low-probability missed disease is expensive enough to dominate. The 0.50 argmax threshold is suitable only for particular symmetric costs, not for this asymmetric medical decision.",
  ),
  makeQuestion(
    "crash-probability-l1-q118",
    "easy",
    "Which examples show probability serving a central role in modern AI?",
    [
      ["A classifier distributes mass across possible labels.", true],
      ["An LLM assigns probabilities to next tokens.", true],
      ["An RL agent compares expected future rewards.", true],
      [
        "A diffusion model starts from random noise and samples a denoising path.",
        true,
      ],
    ],
    "Prediction, language generation, sequential decision-making, and diffusion all manipulate uncertainty using distributions or expectations. Their outputs and training objectives differ, but probability provides a common language for alternatives, random transitions, and average consequences.",
  ),
  makeQuestion(
    "crash-probability-l1-q119",
    "medium",
    "A token distribution assigns masses \\((0.4,0.3,0.2,0.1)\\). Which statements are correct?",
    [
      [
        "The event containing the first and fourth tokens has probability 0.5.",
        true,
      ],
      [
        "Sampling converts the distribution into one realized token outcome.",
        true,
      ],
      [
        "The expected token is obtained by averaging token names as ordinary numbers.",
        false,
      ],
      [
        "The second token is impossible because it is not the largest-probability token.",
        false,
      ],
    ],
    "Event probability adds the masses of included token outcomes, and sampling realizes one value according to those masses. Arbitrary token names do not have a meaningful numerical mean, and non-maximum tokens remain possible whenever they have positive probability.",
  ),
  makeQuestion(
    "crash-probability-l1-q120",
    "hard",
    "A model predicts numerical loss \\(L\\) with \\(P(L=0)=0.5\\), \\(P(L=2)=0.3\\), and \\(P(L=5)=0.2\\). Which statements are correct?",
    [
      ["The PMF is valid because its nonnegative masses sum to one.", true],
      ["\\(P(L\\ge2)=0.5\\).", true],
      ["\\(\\mathbb{E}[L]=0(0.5)+2(0.3)+5(0.2)=1.6\\).", true],
      [
        "The expected loss 1.6 identifies the most probable realized loss.",
        false,
      ],
    ],
    "The distribution satisfies the probability axioms, its high-loss event combines masses 0.3 and 0.2, and the weighted average is 1.6. The mode is 0 with probability 0.5, so expectation and most likely outcome are distinct summaries.",
  ),
];
