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
    chapter: 2,
    difficulty,
    prompt,
    options: options.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

export const stanfordCS109Lecture2CombinatoricsQuestions: Question[] = [
  makeQuestion(
    "cs109-lect2-q01",
    "easy",
    "Which statements correctly match common combinatorial tasks to the feature that determines their count?",
    [
      [
        "A permutation counts arrangements in which the order of selected objects matters.",
        true,
      ],
      [
        "A combination counts selections of distinct objects in which the selected order does not matter.",
        true,
      ],
      [
        "Assigning distinct objects to labeled buckets treats each object as making a bucket choice.",
        true,
      ],
      [
        "Assigning indistinguishable objects to labeled buckets can be represented by objects separated with dividers.",
        true,
      ],
    ],
    "The main modeling decision is whether objects are distinguishable, whether order matters, and whether the task selects, arranges, or assigns. Those decisions lead respectively to factorial, binomial-coefficient, power, or divider-method counts.",
  ),
  makeQuestion(
    "cs109-lect2-q02",
    "easy",
    "The five letters in CHRIS are distinct. Which statements correctly count their full orderings?",
    [
      ["The count is \\(5!=120\\).", true],
      [
        "A position-by-position construction has choice counts 5, 4, 3, 2, and 1.",
        true,
      ],
      [
        "Every generated ordering uses each of the five letters exactly once.",
        true,
      ],
      [
        "The count is \\(5^5\\) because every position can independently reuse all five letters.",
        false,
      ],
    ],
    "A permutation is an ordered arrangement without reuse, so the number of available letters decreases at each position. The product \\(5\\times4\\times3\\times2\\times1\\) is 120; \\(5^5\\) would describe length-five strings that allow repeated letters.",
  ),
  makeQuestion(
    "cs109-lect2-q03",
    "medium",
    "A phone shows smudges on six distinct digits. Compare two six-digit passcode models: Model S uses each smudged digit exactly once; Model T uses six different digits chosen from all ten decimal digits. Which statements are correct?",
    [
      ["Model S has \\(6!=720\\) possible passcodes.", true],
      [
        "Model T has \\(10\\times9\\times8\\times7\\times6\\times5=10!/4!=151{,}200\\) possible passcodes.",
        true,
      ],
      [
        "Model T has \\(10^6\\) possibilities because it forbids repeated digits.",
        false,
      ],
      [
        "Model S has \\(\\binom{6}{6}=1\\) possibility because it uses all six digits.",
        false,
      ],
    ],
    "Model S orders a fixed set of six digits, while Model T selects and orders six different digits from ten. A combination alone discards order, and \\(10^6\\) allows repetitions, so neither matches the stated passcode constraints.",
  ),
  makeQuestion(
    "cs109-lect2-q04",
    "hard",
    "A six-digit passcode uses all five digits visible as smudges, so exactly one digit appears twice and the other four appear once. How many passcodes satisfy the evidence?",
    [
      ["\\(5\\times\\frac{6!}{2!}=1{,}800\\)", true],
      ["\\(5^6=15{,}625\\)", false],
      ["\\(\\frac{6!}{5!}=6\\)", false],
      ["\\(\\binom{6}{5}\\times5!=720\\)", false],
    ],
    "First choose which of the five smudged digits is repeated. The resulting six-symbol multiset has one pair of identical symbols, so it has \\(6!/2!\\) visible orderings; multiplying by the five repeated-digit choices gives 1,800.",
  ),
  makeQuestion(
    "cs109-lect2-q05",
    "medium",
    "A collection of \\(n\\) objects contains indistinguishable groups of sizes \\(n_1,n_2,\\ldots,n_r\\), where the group sizes sum to \\(n\\). Which statements correctly explain its visible orderings?",
    [
      ["Temporarily labeling every object gives \\(n!\\) orderings.", true],
      [
        "Permuting labels within group \\(i\\) creates \\(n_i!\\) labeled versions of the same visible ordering.",
        true,
      ],
      [
        "Dividing by \\(n_1!n_2!\\cdots n_r!\\) removes all within-group multiplicative overcounting.",
        true,
      ],
      ["The final count is \\(\\frac{n!}{n_1!n_2!\\cdots n_r!}\\).", true],
    ],
    "Labeling converts the task into an ordinary distinct-object permutation count, but those labels create duplicates. Every visible arrangement has the same number \\(n_1!\\cdots n_r!\\) of labeled realizations, so division removes that uniform overcount.",
  ),
  makeQuestion(
    "cs109-lect2-q06",
    "medium",
    "A 5-bit string contains exactly three 0s and two 1s. Which statements are correct?",
    [
      ["The number of strings is \\(5!/(3!2!)=10\\).", true],
      [
        "The same count is \\(\\binom{5}{2}\\), obtained by choosing the positions of the 1s.",
        true,
      ],
      [
        "Labeling all five bits first creates \\(5!\\) orderings and overcounts by \\(3!2!\\).",
        true,
      ],
      [
        "The number is \\(2^5=32\\) because every bit position is unrestricted.",
        false,
      ],
    ],
    "The fixed multiplicities make this a semi-distinct permutation problem. It is equivalently a position-selection problem: choosing the two 1 positions determines all three 0 positions, so both derivations give 10 rather than the unrestricted 32.",
  ),
  makeQuestion(
    "cs109-lect2-q07",
    "easy",
    "The two B letters in BOBA are indistinguishable. Which statements correctly count the distinct letter orderings?",
    [
      ["The count is \\(4!/2!=12\\).", true],
      [
        "The factor \\(2!\\) removes the two labeled swaps of the B letters within each visible arrangement.",
        true,
      ],
      [
        "The count is \\(4!=24\\) because the two B occurrences occupy different positions.",
        false,
      ],
      [
        "The count is \\(\\binom{4}{2}/2!=3\\) because O and A need no positions.",
        false,
      ],
    ],
    "Choosing the two B positions gives \\(\\binom{4}{2}=6\\), and the remaining O and A can be assigned to their positions in two orders, producing 12. The factorial derivation reaches the same result by dividing the 24 fully labeled arrangements by the two invisible B swaps.",
  ),
  makeQuestion(
    "cs109-lect2-q08",
    "hard",
    "How many distinct orderings does the word MISSISSIPPI have?",
    [
      ["\\(\\frac{11!}{4!4!2!}=34{,}650\\)", true],
      ["\\(\\frac{11!}{4!2!}=831{,}600\\)", false],
      ["\\(11!=39{,}916{,}800\\)", false],
      ["\\(\\binom{11}{4}=330\\)", false],
    ],
    "MISSISSIPPI has four Is, four Ss, two Ps, and one M. Starting from \\(11!\\) labeled orderings, divide by the internal permutations of each repeated-letter group; omitting either group leaves duplicates, while choosing only the I positions does not finish the arrangement.",
  ),
  makeQuestion(
    "cs109-lect2-q09",
    "medium",
    "Which statements correctly distinguish the two overcounting corrections used in basic combinatorics?",
    [
      [
        "In a union count, outcomes in an intersection are duplicated additively and one copy is subtracted.",
        true,
      ],
      [
        "In a semi-distinct permutation count, each visible arrangement has a fixed number of labeled copies, so the total is divided by that factor.",
        true,
      ],
      [
        "For three 0s and two 1s, the uniform labeled-copy factor is \\(3!2!\\).",
        true,
      ],
      [
        "Dividing \\(5!\\) by \\(3!2!\\) correctly counts strings with three 0s and two 1s.",
        true,
      ],
    ],
    "Union overlap and repeated-object labels create different kinds of duplication. Repeated-object permutations create a uniform multiplicative overcount, so \\(5!/(3!2!)=10\\); subtracting the denominator instead would have no valid counting interpretation.",
  ),
  makeQuestion(
    "cs109-lect2-q10",
    "easy",
    "A code uses six different digits chosen from ten, and order matters. Which statements correctly describe the count?",
    [
      ["The position counts are \\(10,9,8,7,6,5\\).", true],
      ["The product is \\(10!/4!\\).", true],
      ["The numerical count is 151,200.", true],
      [
        "The count is \\(\\binom{10}{6}\\) because a code ignores the order of its digits.",
        false,
      ],
    ],
    "The code is an ordered selection without replacement, also called a partial permutation. Choosing a six-digit set would give only \\(\\binom{10}{6}\\); each such set has \\(6!\\) orders, and their product equals \\(10!/4!=151{,}200\\).",
  ),
  makeQuestion(
    "cs109-lect2-q11",
    "hard",
    "A 10-bit string must contain exactly four 1s. Which statements are correct?",
    [
      [
        "Choosing the four positions of the 1s gives \\(\\binom{10}{4}=210\\) strings.",
        true,
      ],
      [
        "Treating the symbols as four indistinguishable 1s and six indistinguishable 0s gives \\(10!/(4!6!)\\).",
        true,
      ],
      [
        "The count is \\(10!/4!\\) because only the 1s create duplicate labels.",
        false,
      ],
      [
        "The count is \\(2^{10}\\) because every position still contains a bit.",
        false,
      ],
    ],
    "Fixing the number of 1s turns the unrestricted bit-string problem into a position selection or, equivalently, a two-group semi-distinct permutation. Both the 1 labels and 0 labels are invisible, so both factorials appear in the denominator.",
  ),
  makeQuestion(
    "cs109-lect2-q12",
    "easy",
    "Eight finalists are available. Which expression counts choosing a three-person committee when committee order does not matter?",
    [
      ["\\(\\binom{8}{3}=56\\)", true],
      ["\\(8\\times7\\times6=336\\)", false],
      ["\\(8^3=512\\)", false],
      ["\\(8!=40{,}320\\)", false],
    ],
    "A committee is an unordered subset of three distinct people, so each group should be counted once. The ordered product \\(8\\times7\\times6\\) counts every committee in \\(3!\\) different member orders; dividing by \\(3!\\) gives 56.",
  ),
  makeQuestion(
    "cs109-lect2-q13",
    "easy",
    "For \\(0\\le k\\le n\\), which statements about choosing \\(k\\) objects from \\(n\\) distinct objects are correct?",
    [
      [
        "The number of unordered selections is \\(\\binom{n}{k}=\\frac{n!}{k!(n-k)!}\\).",
        true,
      ],
      [
        "The factor \\(k!\\) removes internal orderings of the selected group.",
        true,
      ],
      [
        "The factor \\((n-k)!\\) removes internal orderings of the unselected group from the initial \\(n!\\) lineups.",
        true,
      ],
      [
        "\\(\\binom{n}{k}=\\binom{n}{n-k}\\) because selecting a group also determines its complement.",
        true,
      ],
    ],
    "Starting from every ordering of all \\(n\\) objects overcounts both the selected prefix and unselected suffix. Dividing by their internal permutations gives the binomial coefficient, and the same partition can be described by choosing either side of it.",
  ),
  makeQuestion(
    "cs109-lect2-q14",
    "medium",
    "At a gathering of 20 distinct people, exactly 5 receive cake. Which statements correctly count the possible recipient groups?",
    [
      ["The count is \\(\\binom{20}{5}=15{,}504\\).", true],
      ["The factorial form is \\(20!/(5!15!)\\).", true],
      ["Choosing the 15 people without cake gives the same count.", true],
      [
        "The count is \\(20!/5!\\) because only the cake recipients can be reordered.",
        false,
      ],
    ],
    "The outcome records which five people receive cake, not their order and not the order of the other fifteen. The initial \\(20!\\) lineups therefore overcount by both \\(5!\\) and \\(15!\\), producing the combination count 15,504.",
  ),
  makeQuestion(
    "cs109-lect2-q15",
    "easy",
    "How many ways can 3 books be chosen from 6 distinct books when the order in which the chosen books are handed over does not matter?",
    [
      ["\\(\\binom{6}{3}=20\\)", true],
      [
        "\\(\\binom{6}{3}=\\binom{6}{6-3}\\), so choosing the kept books or the omitted books gives the same count.",
        true,
      ],
      ["\\(6\\times5\\times4=120\\)", false],
      ["\\(6^3=216\\)", false],
    ],
    "The selected books form an unordered subset. The ordered product counts each three-book set in \\(3!\\) handover orders, and allowing independent reuse would count sequences rather than selections; dividing 120 by 6 gives 20.",
  ),
  makeQuestion(
    "cs109-lect2-q16",
    "medium",
    "How many distinct 5-card hands can be drawn from a standard 52-card deck when card order within a hand is irrelevant?",
    [
      ["\\(\\binom{52}{5}=2{,}598{,}960\\)", true],
      ["\\(52\\times51\\times50\\times49\\times48\\)", false],
      ["\\(52^5\\)", false],
      ["\\(52!/5!\\)", false],
    ],
    "A hand is an unordered selection of five distinct cards, so the appropriate count is the binomial coefficient. The descending product counts every hand in \\(5!\\) draw orders, while the power permits repeated cards and \\(52!/5!\\) leaves the unselected-card order overcount unresolved.",
  ),
  makeQuestion(
    "cs109-lect2-q17",
    "easy",
    "A programmer uses enumeration to verify combinatorial formulas on small inputs. Which statements are sound?",
    [
      [
        "Python's itertools.permutations generates ordered tuples, so repeated input values can produce duplicate-looking tuples.",
        true,
      ],
      [
        "Converting repeated-value permutations to a set can reveal the number of unique visible arrangements.",
        true,
      ],
      [
        "Python's itertools.combinations(cards, 5) enumerates unordered five-card selections without enumerating their draw orders.",
        true,
      ],
      [
        "A formula can count a large outcome set without materializing every outcome in memory.",
        true,
      ],
    ],
    "Enumeration and counting answer related but different computational questions. Python's iterator tools are useful for checking small cases, while factorial and combination formulas obtain the cardinality directly and avoid constructing millions of outcomes.",
  ),
  makeQuestion(
    "cs109-lect2-q18",
    "medium",
    "A group of \\(n\\) people is used for several tasks. Which statements correctly match the task to a count?",
    [
      ["Choosing a \\(k\\)-person committee gives \\(\\binom{n}{k}\\).", true],
      [
        "Choosing a president and a different vice president gives \\(n(n-1)\\).",
        true,
      ],
      [
        "Choosing a \\(k\\)-person committee and then its chair gives \\(\\binom{n}{k}k\\).",
        true,
      ],
      [
        "Choosing a committee and assigning its members to \\(k\\) distinct roles still gives \\(\\binom{n}{k}\\).",
        false,
      ],
    ],
    "Committees ignore internal order, but named offices or roles make assignments distinguishable. After selecting a committee, choosing one chair adds a factor of \\(k\\), while assigning every member to a distinct role would add a factor of \\(k!\\).",
  ),
  makeQuestion(
    "cs109-lect2-q19",
    "easy",
    "There are \\(n\\) distinct strings and \\(r\\) labeled hash buckets, with no capacity restrictions. Which statements are correct?",
    [
      ["Each string independently has \\(r\\) bucket choices.", true],
      ["The number of complete assignments is \\(r^n\\).", true],
      [
        "The number is \\(\\binom{n+r-1}{r-1}\\) because the strings are indistinguishable.",
        false,
      ],
      [
        "The number is \\(n!\\) because assigning buckets is the same as sorting the strings.",
        false,
      ],
    ],
    "Distinct strings create \\(n\\) separate assignment steps, each with \\(r\\) labeled destinations, so the product rule yields \\(r^n\\). The divider formula applies when objects are indistinguishable and only bucket occupancies matter.",
  ),
  makeQuestion(
    "cs109-lect2-q20",
    "easy",
    "Four distinct strings are hashed independently into three labeled buckets. How many assignments are possible if buckets may be empty?",
    [
      ["\\(3^4=81\\)", true],
      ["\\(4^3=64\\)", false],
      ["\\(\\binom{6}{2}=15\\)", false],
      ["\\(4!=24\\)", false],
    ],
    "Each of the four distinct strings chooses one of three destinations, and no choice changes the options for another string. The product rule gives \\(3\\times3\\times3\\times3=81\\); the divider count would collapse string identities.",
  ),
  makeQuestion(
    "cs109-lect2-q21",
    "medium",
    "There are \\(n\\) indistinguishable requests to distribute among \\(r\\) labeled servers, and empty servers are allowed. Which statements correctly describe the divider method?",
    [
      [
        "An assignment is represented by an ordering of \\(n\\) identical request symbols and \\(r-1\\) identical dividers.",
        true,
      ],
      ["The total number of symbols in that ordering is \\(n+r-1\\).", true],
      [
        "Dividing by \\(n!\\) and \\((r-1)!\\) removes permutations of identical requests and identical dividers.",
        true,
      ],
      [
        "The number of assignments is \\(\\binom{n+r-1}{r-1}=\\binom{n+r-1}{n}\\).",
        true,
      ],
    ],
    "The divider positions encode where one labeled server's count ends and the next begins. Because requests and divider marks are indistinguishable within their own groups, the semi-distinct permutation formula reduces to the two equivalent binomial coefficients shown.",
  ),
  makeQuestion(
    "cs109-lect2-q22",
    "medium",
    "Five indistinguishable orders are assigned to four labeled fulfillment centers, with centers allowed to receive zero orders. Which statements are correct?",
    [
      [
        "The divider representation uses five object symbols and three dividers.",
        true,
      ],
      ["The number of assignments is \\(\\binom{8}{3}=56\\).", true],
      ["The same count is \\(8!/(5!3!)\\).", true],
      [
        "The number is \\(4^5=1{,}024\\) because the five orders retain separate identities.",
        false,
      ],
    ],
    "Only the four occupancy counts matter because the orders are indistinguishable. Stars and bars uses \\(r-1=3\\) dividers among eight total symbols, giving 56; \\(4^5\\) would distinguish each individual order.",
  ),
  makeQuestion(
    "cs109-lect2-q23",
    "medium",
    "In a stars-and-bars encoding for labeled buckets with no minimum occupancy, which statements are correct?",
    [
      [
        "\\(r\\) buckets require \\(r-1\\) dividers because the spaces before, between, and after them form \\(r\\) regions.",
        true,
      ],
      ["Adjacent dividers encode an empty bucket between them.", true],
      [
        "The dividers must be distinctly labeled, or the bucket order is lost.",
        false,
      ],
      [
        "A divider at either end is forbidden because the first and last buckets must be nonempty.",
        false,
      ],
    ],
    "The left-to-right regions already identify the labeled buckets, so swapping imaginary divider labels does not change an assignment. Allowing dividers to touch or appear at an end is exactly what permits interior or endpoint buckets to have zero objects.",
  ),
  makeQuestion(
    "cs109-lect2-q24",
    "hard",
    "How many nonnegative integer solutions satisfy \\(x_1+x_2+x_3=7\\)?",
    [
      ["\\(\\binom{7+3-1}{3-1}=\\binom{9}{2}=36\\)", true],
      ["\\(3^7=2{,}187\\)", false],
      ["\\(\\binom{7}{3}=35\\)", false],
      ["\\(7!/3!=840\\)", false],
    ],
    "View the seven units as indistinguishable objects and the three variables as labeled buckets. Two dividers split the seven object symbols into three possibly empty groups, so the count is \\(\\binom{9}{2}=36\\); \\(3^7\\) would distinguish the units.",
  ),
  makeQuestion(
    "cs109-lect2-q25",
    "hard",
    "For positive integer solutions to \\(x_1+x_2+x_3=7\\), which statements are correct?",
    [
      [
        "Setting \\(y_i=x_i-1\\) converts the problem to \\(y_1+y_2+y_3=4\\) with \\(y_i\\ge0\\).",
        true,
      ],
      ["The transformed problem has \\(\\binom{6}{2}=15\\) solutions.", true],
      [
        "Giving one unit to each variable first enforces positivity before applying the divider method to the remaining four units.",
        true,
      ],
      [
        "There are fewer positive solutions than the 36 nonnegative solutions because boundary cases containing zeros are excluded.",
        true,
      ],
    ],
    "A lower bound of one is handled by reserving one unit for each variable, leaving four indistinguishable units to distribute freely. Stars and bars then gives \\(\\binom{4+3-1}{3-1}=15\\), a strict subset of the nonnegative solution set.",
  ),
  makeQuestion(
    "cs109-lect2-q26",
    "medium",
    "Which statements correctly compare placing objects into labeled buckets?",
    [
      [
        "For \\(n\\) distinct objects with unrestricted capacities, the count is \\(r^n\\).",
        true,
      ],
      [
        "For \\(n\\) indistinguishable objects with unrestricted capacities, the count is \\(\\binom{n+r-1}{r-1}\\).",
        true,
      ],
      [
        "Bucket labels matter in both formulas because occupancy vectors such as \\((2,0,1)\\) and \\((0,2,1)\\) are different.",
        true,
      ],
      [
        "The two formulas agree for every \\(n\\) and \\(r\\) because both describe bucket occupancy.",
        false,
      ],
    ],
    "Distinct-object assignments remember which object went where, whereas indistinguishable-object assignments remember only the occupancy vector. Both use labeled destinations, but collapsing object identity makes many \\(r^n\\) assignments correspond to one stars-and-bars outcome.",
  ),
  makeQuestion(
    "cs109-lect2-q27",
    "hard",
    "Ten ordered experimental outcomes contain exactly 7 results of type A, 2 of type B, and 1 of type C. Which statements are correct?",
    [
      ["The number of type sequences is \\(10!/(7!2!1!)=360\\).", true],
      [
        "Equivalently, choose 7 A positions and then 2 of the remaining 3 positions for B: \\(\\binom{10}{7}\\binom{3}{2}=360\\).",
        true,
      ],
      [
        "The number is \\(3^{10}\\) because every position may independently be A, B, or C.",
        false,
      ],
      [
        "The number is \\(10!\\) because time makes all repeated type labels distinct.",
        false,
      ],
    ],
    "Time positions are distinct, but outcomes of the same type are visually interchangeable within a type sequence. The multinomial and sequential-position-selection derivations both enforce the fixed 7-2-1 counts and yield 360.",
  ),
  makeQuestion(
    "cs109-lect2-q28",
    "easy",
    "For each task use \\(n=5\\) distinct objects, choose-size \\(k=2\\), and \\(r=3\\) labeled buckets where relevant. Which row gives, in order, the counts for (i) arranging all objects, (ii) choosing a subset, (iii) assigning distinct objects to buckets, and (iv) assigning five indistinguishable objects to buckets?",
    [
      ["\\(120,\\ 10,\\ 243,\\ 21\\)", true],
      ["\\(120,\\ 20,\\ 243,\\ 21\\)", false],
      ["\\(25,\\ 10,\\ 125,\\ 10\\)", false],
      ["\\(120,\\ 10,\\ 21,\\ 243\\)", false],
    ],
    "The four counts are \\(5!=120\\), \\(\\binom{5}{2}=10\\), \\(3^5=243\\), and \\(\\binom{5+3-1}{3-1}=\\binom{7}{2}=21\\). The competing rows reflect common confusions: doubling a combination, reversing bases and exponents, or swapping the distinct- and indistinguishable-bucket models.",
  ),
  makeQuestion(
    "cs109-lect2-q29",
    "hard",
    "Six distinct books are placed into three labeled boxes with exactly two books per box. Which statements correctly count the assignments?",
    [
      [
        "A sequential count is \\(\\binom{6}{2}\\binom{4}{2}\\binom{2}{2}\\).",
        true,
      ],
      ["The product evaluates to \\(15\\times6\\times1=90\\).", true],
      ["The same count is \\(6!/(2!2!2!)\\).", true],
      [
        "The unrestricted count \\(3^6\\) is too large because it includes box occupancies other than 2-2-2.",
        true,
      ],
    ],
    "The boxes are labeled, so choosing the two books for each box in box order counts every valid assignment once. The multinomial form removes only the internal order of each box, while the ordinary power includes assignments that violate the fixed capacities.",
  ),
  makeQuestion(
    "cs109-lect2-q30",
    "medium",
    "Why does \\(\\binom{n}{k}\\) have the same factorial form as arranging \\(k\\) identical selected markers and \\(n-k\\) identical unselected markers?",
    [
      [
        "Each length-\\(n\\) marker string identifies exactly which \\(k\\) positions are selected.",
        true,
      ],
      ["The semi-distinct permutation count is \\(n!/(k!(n-k)!)\\).", true],
      [
        "Choosing the selected positions uniquely determines the unselected positions.",
        true,
      ],
      [
        "The equivalence requires the original \\(n\\) objects to be indistinguishable.",
        false,
      ],
    ],
    "The marker symbols are indistinguishable, but their positions refer to \\(n\\) distinct original objects. A marker string with \\(k\\) selected symbols is therefore another encoding of a \\(k\\)-element subset, which explains the shared factorial formula.",
  ),
  makeQuestion(
    "cs109-lect2-q31",
    "hard",
    "A seven-digit passcode uses each of five known distinct digits at least once, and exactly two of those digits appear twice. Which statements are correct?",
    [
      [
        "There are \\(\\binom{5}{2}\\) choices for the two repeated digits.",
        true,
      ],
      [
        "For each repeated pair, the multiset has \\(7!/(2!2!)\\) orderings, giving \\(\\binom{5}{2}7!/(2!2!)=12{,}600\\) passcodes.",
        true,
      ],
      [
        "The count is \\(5^7\\) because the condition allows any frequency pattern using the five digits.",
        false,
      ],
      [
        "The count is \\(\\binom{7}{5}5!\\) because choosing one position per digit accounts for both repetitions.",
        false,
      ],
    ],
    "The frequency pattern is fixed at 2-2-1-1-1, so first identify which two digits receive multiplicity two. The semi-distinct permutation formula then orders that multiset; unrestricted sequences and a one-position-per-digit construction do not enforce the stated frequencies.",
  ),
  makeQuestion(
    "cs109-lect2-q32",
    "hard",
    "Eight indistinguishable requests are distributed among three labeled servers, and every server must receive at least one request. How many occupancy assignments are possible?",
    [
      ["\\(\\binom{8-1}{3-1}=\\binom{7}{2}=21\\)", true],
      ["\\(\\binom{10}{2}=45\\)", false],
      ["\\(3^8=6{,}561\\)", false],
      ["\\(8!/3!=6{,}720\\)", false],
    ],
    "Reserve one request for each server to enforce positivity, leaving five indistinguishable requests to distribute with zeros allowed. Stars and bars then gives \\(\\binom{5+3-1}{2}=\\binom{7}{2}=21\\); the unrestricted nonnegative count would include empty servers.",
  ),
  makeQuestion(
    "cs109-lect2-q33",
    "hard",
    "Which statements correctly describe how bucket constraints change the counting model?",
    [
      [
        "The unrestricted distinct-object count \\(r^n\\) allows empty buckets and arbitrary occupancies.",
        true,
      ],
      [
        "The unrestricted divider method allows empty buckets through adjacent or endpoint dividers.",
        true,
      ],
      [
        "A requirement that every bucket be nonempty changes the count and must be enforced, for indistinguishable objects, by reserving one object per bucket.",
        true,
      ],
      [
        "Fixed bucket capacities such as 2-2-2 require a constrained count rather than the unrestricted power or stars-and-bars formula.",
        true,
      ],
    ],
    "The canonical formulas assume no occupancy limits beyond nonnegativity. Nonempty or fixed-capacity requirements remove outcomes from those unrestricted spaces, so the constraint must be built into the construction through shifts, subset choices, or multinomial counts.",
  ),
  makeQuestion(
    "cs109-lect2-q34",
    "easy",
    "Which sanity checks for the basic combinatorial formulas are correct?",
    [
      [
        "Placing any number of objects into one labeled bucket gives one assignment.",
        true,
      ],
      [
        "Choosing zero objects from \\(n\\) distinct objects gives \\(\\binom{n}{0}=1\\).",
        true,
      ],
      [
        "Distributing zero indistinguishable objects among \\(r\\) labeled buckets gives the single all-zero occupancy vector.",
        true,
      ],
      [
        "Ordering \\(n\\) indistinguishable copies gives \\(n!\\) visible arrangements.",
        false,
      ],
    ],
    "Boundary cases are useful checks: one destination, an empty selection, and an empty distribution each have a single outcome. Fully indistinguishable copies also have only one visible ordering, so applying \\(n!\\) without removing label overcounting fails that sanity check.",
  ),
  makeQuestion(
    "cs109-lect2-q35",
    "hard",
    "Eight ordered observations contain exactly 4 results of type A, 3 of type B, and 1 of type C. Which statements are correct?",
    [
      ["The number of type sequences is \\(8!/(4!3!1!)=280\\).", true],
      [
        "The same count is \\(\\binom{8}{4}\\binom{4}{3}=70\\times4=280\\).",
        true,
      ],
      [
        "The number is \\(8!\\) because the observation times distinguish repeated type labels.",
        false,
      ],
      [
        "The number is \\(3^8\\) because it enforces exactly four A results, three B results, and one C result.",
        false,
      ],
    ],
    "The positions are distinct but labels within each result type are not, so the multinomial denominator removes within-type permutations. Sequentially choosing A positions and then B positions encodes the same frequency constraint and confirms the value 280.",
  ),
];
