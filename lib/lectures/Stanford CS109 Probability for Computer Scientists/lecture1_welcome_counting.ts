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
    chapter: 1,
    difficulty,
    prompt,
    options: options.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

export const stanfordCS109Lecture1WelcomeCountingQuestions: Question[] = [
  makeQuestion(
    "cs109-lect1-q01",
    "easy",
    "A probability experiment is performed once. Which statements correctly distinguish the experiment, its outcomes, and an event?",
    [
      [
        "The experiment is the process that produces an observation, such as rolling a die.",
        true,
      ],
      [
        "An outcome is one complete possible result of the experiment, such as rolling a 4.",
        true,
      ],
      [
        "An event is a set of outcomes sharing a property, such as the even results \\(\\{2,4,6\\}\\).",
        true,
      ],
      [
        "Counting an event asks how many possible outcomes satisfy the event's defining property.",
        true,
      ],
    ],
    "An experiment generates one outcome from its sample space, while an event collects one or more outcomes of interest. Counting connects these ideas by determining the cardinality of the event, not by changing the underlying experiment.",
  ),
  makeQuestion(
    "cs109-lect1-q02",
    "easy",
    "A fair six-sided die is rolled once and \\(E\\) is the event that the result is even. Which statements are correct?",
    [
      ["The experiment has six possible outcomes.", true],
      ["The event is \\(E=\\{2,4,6\\}\\).", true],
      ["The event contains three outcomes.", true],
      [
        "The event contains six outcomes because the die remains six-sided.",
        false,
      ],
    ],
    "The sample space is \\(\\{1,2,3,4,5,6\\}\\), but the event filters that space to the outcomes satisfying the evenness condition. The physical die still has six faces; that does not make every face a member of the event.",
  ),
  makeQuestion(
    "cs109-lect1-q03",
    "hard",
    "A two-step construction has \\(m\\) first-step choices. After any first choice, there are exactly \\(n\\) legal second-step choices, although the identities of those choices can differ. Which conclusions follow?",
    [
      [
        "The product rule gives \\(mn\\) complete outcomes because every first-step branch has the same number of continuations.",
        true,
      ],
      [
        "The first choice may change which second choices remain without invalidating the product rule.",
        true,
      ],
      [
        "The product rule requires the second-step choices to be the same named objects on every branch.",
        false,
      ],
      [
        "The total must be \\(m+n\\) because the construction contains two steps.",
        false,
      ],
    ],
    "The product rule needs a fixed continuation count on every branch, not identical continuation labels. Multiplication counts one complete outcome for each pairing of a first choice with one of its legal second-step continuations; addition would instead combine alternative, nonsequential cases.",
  ),
  makeQuestion(
    "cs109-lect1-q04",
    "easy",
    "An avatar generator independently offers 4 shirts, 3 pairs of trousers, and 2 hats. Which total counts the avatars with one item from each category?",
    [
      ["\\(4\\times3\\times2=24\\)", true],
      ["\\(4+3+2=9\\)", false],
      ["\\(4\\times(3+2)=20\\)", false],
      ["\\(4^3=64\\)", false],
    ],
    "Constructing an avatar is a three-step process: choose a shirt, then trousers, then a hat. Each branch has 3 choices at the second step and 2 at the third, so the product rule gives 24; the other expressions model different structures.",
  ),
  makeQuestion(
    "cs109-lect1-q05",
    "medium",
    "A digital color uses 8 bits each for red, green, and blue, and an image has \\(n\\) pixels whose colors can be chosen freely. Which statements correctly count the possibilities?",
    [
      ["One color channel has \\(2^8=256\\) settings.", true],
      ["One pixel has \\(256^3\\) possible colors.", true],
      ["The image has \\((256^3)^n\\) possible color assignments.", true],
      ["The same image count can be written as \\(2^{24n}\\).", true],
    ],
    "Each channel is an eight-step binary construction, and the three channel settings combine to form one pixel color. Repeating that fixed color choice across \\(n\\) labeled pixels gives \\((2^{24})^n=2^{24n}\\), so all four expressions describe compatible levels of the same product-rule construction.",
  ),
  makeQuestion(
    "cs109-lect1-q06",
    "medium",
    "A message contains 12 freely chosen bits. A second format fixes the first three bits to 101 and leaves the other nine bits free. Which statements are correct?",
    [
      ["The unrestricted format has \\(2^{12}\\) messages.", true],
      ["The fixed-prefix format has \\(2^9\\) messages.", true],
      ["Fixing the prefix reduces the count by a factor of \\(2^3\\).", true],
      [
        "The fixed-prefix format has \\(3\\times2^9\\) messages because the prefix contains three positions.",
        false,
      ],
    ],
    "Every free bit contributes a factor of two, while a fixed bit contributes a factor of one. Removing three binary choices changes \\(2^{12}\\) to \\(2^9\\), an eightfold reduction; the number of fixed positions is not an additional choice factor.",
  ),
  makeQuestion(
    "cs109-lect1-q07",
    "medium",
    "Four distinct letters are arranged without reuse. Which facts justify the count \\(4\\times3\\times2\\times1\\)?",
    [
      ["After the first letter is chosen, exactly three letters remain.", true],
      [
        "The first choice changes which letters remain but not how many remain.",
        true,
      ],
      [
        "Every position has four choices because the letters are distinct.",
        false,
      ],
      [
        "The count uses the sum rule because the four positions are alternatives.",
        false,
      ],
    ],
    "The positions are sequential construction steps, and the number of legal choices falls from four to one as letters are used. The product rule still applies because each branch at a given depth has the same remaining-choice count, even though the identities of the remaining letters depend on earlier choices.",
  ),
  makeQuestion(
    "cs109-lect1-q08",
    "hard",
    "A condition affects 0.8% of a population. A test is positive for 90% of affected people and for 7% of unaffected people. Approximately what fraction of people with a positive result actually have the condition?",
    [
      ["About 9.4%.", true],
      ["About 90.0%.", false],
      ["About 7.2%.", false],
      ["About 0.8%.", false],
    ],
    "Among 10,000 people, about 80 are affected and 72 of those test positive. About 9,920 are unaffected and roughly 694 of those test positive, so only about \\(72/(72+694)\\approx9.4\\%\\) of positive results are true positives; the sensitivity alone is not the desired reversed conditional.",
  ),
  makeQuestion(
    "cs109-lect1-q09",
    "easy",
    "Each situation partitions the allowed outcomes into two disjoint cases. Which totals correctly apply the sum rule?",
    [
      [
        "A menu with 5 pasta dishes and 4 curry dishes offers \\(5+4=9\\) main dishes when no dish belongs to both categories.",
        true,
      ],
      [
        "A die result in \\(\\{1,2,3\\}\\) or \\(\\{4,5,6\\}\\) has \\(3+3=6\\) possible outcomes.",
        true,
      ],
      [
        "Choosing either one of 7 novels or one of 2 biographies gives \\(7+2=9\\) book choices when the shelves do not overlap.",
        true,
      ],
      [
        "A toy collection with 2 balls and 3 plush animals has \\(2+3=5\\) toys when every toy belongs to exactly one category.",
        true,
      ],
    ],
    "The sum rule combines alternative cases when each final outcome appears in exactly one case. In every example the two sets are disjoint, so adding their sizes counts every allowed outcome once and no overlap correction is needed.",
  ),
  makeQuestion(
    "cs109-lect1-q10",
    "medium",
    "Sets \\(A\\) and \\(B\\) describe two ways an outcome can qualify. Which statements correctly relate disjointness to counting \\(A\\cup B\\)?",
    [
      ["If \\(A\\cap B=\\varnothing\\), then \\(|A\\cup B|=|A|+|B|\\).", true],
      [
        "If \\(A\\cap B\\neq\\varnothing\\), adding \\(|A|+|B|\\) counts each overlap outcome twice.",
        true,
      ],
      ["Subtracting \\(|A\\cap B|\\) once corrects that double count.", true],
      [
        "Disjoint sets must have equal cardinalities before their sizes can be added.",
        false,
      ],
    ],
    "Disjointness concerns membership, not whether the sets have the same size. With no shared outcomes, simple addition is exact; with overlap, the shared outcomes appear once in each set count, so inclusion-exclusion removes one duplicate copy.",
  ),
  makeQuestion(
    "cs109-lect1-q11",
    "medium",
    "Suppose \\(|A|=18\\), \\(|B|=12\\), and \\(|A\\cap B|=5\\). Which statements are correct?",
    [
      ["\\(|A\\cup B|=18+12-5=25\\).", true],
      ["The naive sum 30 overcounts the five shared outcomes once each.", true],
      ["\\(|A\\cup B|=35\\) because the overlap must be added.", false],
      ["The sets are mutually exclusive because both sizes are finite.", false],
    ],
    "The union should contain each distinct outcome once. Adding the set sizes produces two copies of every shared member, and subtracting the intersection removes the extra copy, yielding 25; finiteness has no bearing on mutual exclusivity.",
  ),
  makeQuestion(
    "cs109-lect1-q12",
    "hard",
    "How many 6-bit strings start with 01 or end with 10?",
    [
      ["\\(2^4+2^4-2^2=28\\)", true],
      ["\\(2^4+2^4=32\\)", false],
      ["\\(2^6-2^2=60\\)", false],
      ["\\(2^2+2^2-1=7\\)", false],
    ],
    "Fixing either the first two or last two bits leaves four free bits, giving 16 strings in each set. Strings that both start with 01 and end with 10 have only two free middle bits, so four strings were counted twice and inclusion-exclusion gives \\(16+16-4=28\\).",
  ),
  makeQuestion(
    "cs109-lect1-q13",
    "easy",
    "An 8-bit string is valid if it starts with 00 or starts with 11. Which statements correctly derive the number of valid strings?",
    [
      ["Each prefix case leaves six freely chosen bits.", true],
      ["Each prefix case therefore contains \\(2^6\\) strings.", true],
      ["The two prefix cases are disjoint.", true],
      ["The total is \\(2^6+2^6=128\\).", true],
    ],
    "A string cannot begin with both 00 and 11, so the cases share no outcomes. The product rule counts the free positions within each case, and the sum rule combines the two disjoint cases, giving 128 valid strings.",
  ),
  makeQuestion(
    "cs109-lect1-q14",
    "hard",
    "A 5-bit string is accepted if its first bit is 0 or its last bit is 0. Which statements are correct?",
    [
      ["There are \\(2^4=16\\) strings in each qualifying set.", true],
      [
        "There are \\(2^3=8\\) strings whose first and last bits are both 0.",
        true,
      ],
      ["The union contains \\(16+16-8=24\\) strings.", true],
      [
        "The union contains 32 strings because the word 'or' always means simple addition.",
        false,
      ],
    ],
    "Fixing one endpoint leaves four free positions, while fixing both endpoints leaves three. The eight strings satisfying both descriptions are duplicated in the sum of 16 and 16, so inclusion-exclusion gives 24 rather than 32.",
  ),
  makeQuestion(
    "cs109-lect1-q15",
    "medium",
    "Which statements correctly count 6-bit strings containing at least one 1?",
    [
      [
        "There are \\(2^6-1=63\\) such strings because only 000000 is excluded.",
        true,
      ],
      [
        "Counting the complement is shorter than separately adding the cases with one, two, three, four, five, or six 1s.",
        true,
      ],
      [
        "There are \\(2^5=32\\) such strings because one position must be fixed to 1.",
        false,
      ],
      [
        "There are 64 such strings because 'at least one' imposes no restriction.",
        false,
      ],
    ],
    "The unrestricted sample space has 64 strings, and exactly one of them has no 1s. Fixing an unspecified position to 1 would double-count strings containing multiple 1s, so subtracting the single complement outcome is the clean count.",
  ),
  makeQuestion(
    "cs109-lect1-q16",
    "easy",
    "A club has 14 members who know Python, 9 who know Java, and 4 who know both. How many members know at least one of the two languages?",
    [
      ["\\(14+9-4=19\\)", true],
      ["\\(14+9=23\\)", false],
      ["\\(14-9+4=9\\)", false],
      ["\\(14\\times9-4=122\\)", false],
    ],
    "Knowing at least one language is the union of the two member sets. The four bilingual members are included in both individual counts, so subtracting the intersection once leaves 19 distinct members.",
  ),
  makeQuestion(
    "cs109-lect1-q17",
    "easy",
    "Which calculations are valid uses of the product rule?",
    [
      [
        "Two distinguishable dice have \\(6\\times6=36\\) ordered outcomes.",
        true,
      ],
      ["An 8-bit channel has \\(2^8=256\\) settings.", true],
      [
        "A 4-digit code with no repeated digit has \\(10\\times9\\times8\\times7\\) possibilities.",
        true,
      ],
      [
        "Four distinct letters have \\(4\\times3\\times2\\times1\\) full orderings.",
        true,
      ],
    ],
    "Each calculation describes a sequence of choices with a fixed number of legal continuations at each step. The counts can remain constant, as with dice, or decline predictably, as with no-reuse codes and permutations; both structures satisfy the product rule.",
  ),
  makeQuestion(
    "cs109-lect1-q18",
    "hard",
    "A counting problem is decomposed into branches. Which statements correctly diagnose when a single fixed-factor product is or is not justified?",
    [
      [
        "Choosing a chair and then a different secretary from 5 people gives \\(5\\times4\\) assignments.",
        true,
      ],
      [
        "If one first-step branch has 3 continuations and another has 5, the branch totals should be counted separately and added.",
        true,
      ],
      [
        "A 4-digit code allowing repetition has \\(10^4\\) outcomes because every position retains 10 choices.",
        true,
      ],
      [
        "Whenever earlier choices change the names of later options, multiplication is invalid even if every branch has the same number of options.",
        false,
      ],
    ],
    "The relevant condition is equality of continuation counts at each stage, not equality of labels. Unequal branch sizes call for conditional case counts followed by addition, while role assignments and repeated-digit codes have well-defined products.",
  ),
  makeQuestion(
    "cs109-lect1-q19",
    "medium",
    "Which statements correctly distinguish sequential choices from alternative cases?",
    [
      [
        "Use multiplication when one complete outcome is built by making one choice at each of several steps.",
        true,
      ],
      [
        "Use addition for alternative cases after ensuring that shared outcomes are not counted twice.",
        true,
      ],
      [
        "Use addition for a shirt-and-trousers outfit because the garment categories are different.",
        false,
      ],
      [
        "Use multiplication for choosing either a novel or a biography because both are books.",
        false,
      ],
    ],
    "An outfit requires both component choices, so its choices form a sequence and multiply. Choosing one book from one category or the other forms alternative cases and adds, with inclusion-exclusion needed if the categories overlap.",
  ),
  makeQuestion(
    "cs109-lect1-q20",
    "easy",
    "How many 4-bit strings begin with 1 and end with 0?",
    [
      ["\\(2^2=4\\)", true],
      ["\\(2^4=16\\)", false],
      ["\\(2^3=8\\)", false],
      ["\\(1+2+2+1=6\\)", false],
    ],
    "The first and last bits are fixed and therefore contribute one choice each. Only the two middle positions remain free, so the product is \\(1\\times2\\times2\\times1=4\\).",
  ),
  makeQuestion(
    "cs109-lect1-q21",
    "medium",
    "A product-rule argument says the first choice must not affect the number of choices at the next step. Which clarifications are correct?",
    [
      [
        "The first choice is allowed to change which particular choices remain.",
        true,
      ],
      [
        "The continuation count must be the same for every branch being represented by one factor.",
        true,
      ],
      [
        "Without-replacement arrangements can satisfy the rule because each depth has a fixed number of unused objects.",
        true,
      ],
      [
        "If continuation counts differ, the problem can still be solved by counting the branch cases separately and adding them.",
        true,
      ],
    ],
    "The product rule is a statement about the shape of the branching tree. Equal branch widths support one common factor, while unequal widths require finer case decomposition; neither situation says that later options must have identical names.",
  ),
  makeQuestion(
    "cs109-lect1-q22",
    "medium",
    "A student writes a short program to enumerate every outcome of a small counting problem. Which uses of that enumeration are sound?",
    [
      [
        "It can check a derived count by comparing the number of generated unique outcomes.",
        true,
      ],
      [
        "It can reveal duplicates that indicate an overcounting argument needs correction.",
        true,
      ],
      [
        "It can make the structure of outcomes concrete before a formula is derived.",
        true,
      ],
      [
        "It proves that the same formula works for every larger instance without further reasoning.",
        false,
      ],
    ],
    "Enumeration is a valuable diagnostic for small instances: it exposes the sample space and makes accidental duplicates visible. A finite program run does not by itself establish a general formula, so the structural counting argument remains necessary.",
  ),
  makeQuestion(
    "cs109-lect1-q23",
    "hard",
    "A 7-bit identifier is valid if it starts with 1 or contains exactly the fixed suffix 001. Which statements correctly set up inclusion-exclusion?",
    [
      [
        "The start-with-1 set has \\(2^6\\) identifiers, and the suffix-001 set has \\(2^4\\).",
        true,
      ],
      [
        "Their intersection has \\(2^3\\) identifiers, so the union has \\(64+16-8=72\\).",
        true,
      ],
      [
        "The intersection is empty because a prefix and suffix describe different positions.",
        false,
      ],
      [
        "The union has \\(2^{6+4}\\) identifiers because the two conditions form sequential steps.",
        false,
      ],
    ],
    "A string can satisfy both a prefix and a suffix condition, so the sets overlap. Fixing the first bit and the three-bit suffix leaves three free positions in the intersection, and subtracting those eight duplicated strings gives 72.",
  ),
  makeQuestion(
    "cs109-lect1-q24",
    "hard",
    "One image format has 12 pixels and another has 300 pixels; every pixel independently has \\(c\\) possible colors. Which expression gives the factor by which the larger format has more possible images?",
    [
      ["\\(c^{300}/c^{12}=c^{288}\\)", true],
      ["\\(c^{300-12}=288c\\)", false],
      ["\\(300c-12c=288c\\)", false],
      ["\\(c^{300}/12\\)", false],
    ],
    "The product rule gives \\(c^n\\) images for \\(n\\) labeled pixels. Dividing the two counts subtracts exponents with the same base, so the multiplicative gap is \\(c^{288}\\), illustrating why modest increases in pixel count create enormous spaces.",
  ),
  makeQuestion(
    "cs109-lect1-q25",
    "easy",
    "Among 80 devices, 40 can run system A, 27 can run system B, and 12 can run both. Which statements are correct?",
    [
      ["Exactly \\(40+27-12=55\\) devices can run at least one system.", true],
      ["Exactly \\(80-55=25\\) devices can run neither system.", true],
      [
        "Exactly \\(40-12=28\\) devices can run system A but not system B.",
        true,
      ],
      [
        "Exactly \\((40-12)+(27-12)=43\\) devices can run exactly one of the systems.",
        true,
      ],
    ],
    "The 12 dual-system devices are counted in both the A and B totals, so inclusion-exclusion gives a union of 55. Removing the overlap from each set gives 28 A-only and 15 B-only devices, and these disjoint groups total 43; the complement of the union contains 25 devices.",
  ),
  makeQuestion(
    "cs109-lect1-q26",
    "hard",
    "A deployment configuration chooses 1 of 4 regions, 1 of 3 model versions, and then either 1 of 2 CPU builds or 1 of 5 GPU builds. CPU and GPU builds are disjoint. Which statements are correct?",
    [
      ["There are \\(2+5=7\\) hardware-build choices.", true],
      ["There are \\(4\\times3\\times7=84\\) complete configurations.", true],
      [
        "The calculation uses the sum rule inside the hardware choice and the product rule across configuration steps.",
        true,
      ],
      [
        "There are \\(4+3+2+5=14\\) configurations because every listed choice is an alternative.",
        false,
      ],
    ],
    "The CPU/GPU split is an alternative between disjoint cases, so those counts add. Region, model, and hardware are all required components of one configuration, so their choice counts multiply, producing 84.",
  ),
  makeQuestion(
    "cs109-lect1-q27",
    "medium",
    "A construction begins with one of two route types. Route A has 3 possible completions and Route B has 5 possible completions. Which statements are correct?",
    [
      [
        "Counting the route branches separately gives \\(3+5=8\\) complete outcomes.",
        true,
      ],
      [
        "A single expression \\(2\\times n\\) is unjustified unless both route types have the same continuation count \\(n\\).",
        true,
      ],
      [
        "The result is \\(3\\times5=15\\) because every Route A completion can be paired with every Route B completion.",
        false,
      ],
      [
        "The result is \\(2+3+5=10\\) because choosing the route is an additional final outcome.",
        false,
      ],
    ],
    "A complete outcome follows Route A or Route B, not both, so the branch totals are alternatives and add. The route label is already represented by membership in its branch; it is not a separate object to add to the completed outcomes.",
  ),
  makeQuestion(
    "cs109-lect1-q28",
    "easy",
    "A café offers either one of 5 desserts or one of 4 drinks, and every selected item can be ordered in 3 sizes. Desserts and drinks are separate catalog items. How many item-size orders are possible?",
    [
      ["\\((5+4)\\times3=27\\)", true],
      ["\\(5+4+3=12\\)", false],
      ["\\(5\\times4\\times3=60\\)", false],
      ["\\(5\\times3+4=19\\)", false],
    ],
    "The item itself is chosen from two disjoint alternative categories, giving nine item choices. Size is then a required second step with three choices for every item, so the combined count is \\(9\\times3=27\\).",
  ),
  makeQuestion(
    "cs109-lect1-q29",
    "medium",
    "The letters in BOBA are rearranged, and the two B tiles are visually indistinguishable. Which statements are correct?",
    [
      [
        "Treating the B tiles as temporarily distinct gives \\(4!\\) labeled arrangements.",
        true,
      ],
      [
        "Swapping the two labeled B tiles does not create a new visible word.",
        true,
      ],
      [
        "Each visible arrangement corresponds to \\(2!\\) labeled arrangements.",
        true,
      ],
      ["The number of visible arrangements is \\(4!/2!=12\\).", true],
    ],
    "The distinct-tile construction is easy to count but duplicates every visible arrangement once for each ordering of the two B labels. Dividing the 24 labeled arrangements by \\(2!\\) removes that multiplicative overcount and leaves 12.",
  ),
  makeQuestion(
    "cs109-lect1-q30",
    "easy",
    "The letters A, B, C, and D are all distinct. Which statements correctly count or explain their full orderings?",
    [
      ["There are \\(4!=24\\) orderings.", true],
      ["The position-choice counts are 4, then 3, then 2, then 1.", true],
      [
        "Earlier choices change the remaining letters but leave a fixed number of choices at each depth.",
        true,
      ],
      [
        "There are \\(4^4\\) orderings because every position can reuse every letter.",
        false,
      ],
    ],
    "A full ordering uses each letter exactly once, so reuse is not allowed. The branching count therefore decreases with each position, and multiplying \\(4\\times3\\times2\\times1\\) gives 24.",
  ),
  makeQuestion(
    "cs109-lect1-q31",
    "hard",
    "Which statements correctly distinguish the set union \\(A\\cup B\\) from the Cartesian product \\(A\\times B\\)?",
    [
      [
        "\\(A\\cup B\\) contains outcomes belonging to at least one set and is counted with addition plus any overlap correction.",
        true,
      ],
      [
        "\\(A\\times B\\) contains ordered pairs with one choice from each set and has size \\(|A||B|\\) for finite sets.",
        true,
      ],
      [
        "\\(A\\cup B\\) and \\(A\\times B\\) are interchangeable whenever \\(|A|=|B|\\).",
        false,
      ],
      [
        "\\(|A\\times B|=|A|+|B|-|A\\cap B|\\) because pairs can overlap.",
        false,
      ],
    ],
    "Union models an 'A or B' qualification, while Cartesian product models an 'A choice and a B choice' construction. Equal set sizes do not erase this structural difference: members of a union are original outcomes, whereas members of a product are ordered pairs.",
  ),
  makeQuestion(
    "cs109-lect1-q32",
    "hard",
    "A first step has two choices, X and Y. Choosing X leaves 3 possible completions, while choosing Y leaves 5. Which expression correctly counts the complete outcomes?",
    [
      ["\\(3+5=8\\)", true],
      ["\\(2\\times5=10\\)", false],
      ["\\(2\\times3=6\\)", false],
      ["\\(2\\times(3+5)=16\\)", false],
    ],
    "The second-step branch width is not constant, so neither 3 nor 5 can serve as a common product factor for both first choices. Count the three X outcomes and five Y outcomes as disjoint branch cases, then add them to get eight.",
  ),
  makeQuestion(
    "cs109-lect1-q33",
    "medium",
    "Two distinguishable six-sided dice are rolled. Which statements about the ordered sample space are correct?",
    [
      ["There are \\(6\\times6=36\\) ordered outcomes.", true],
      ["\\((2,5)\\) is one complete outcome.", true],
      ["Exactly six outcomes have a sum of 7.", true],
      ["Exactly 18 outcomes have an even first die.", true],
    ],
    "Distinguishable dice make \\((a,b)\\) and \\((b,a)\\) different unless the values match. The sum-seven event contains \\((1,6)\\) through \\((6,1)\\), while each of the three even first-die values can be paired with any of six second-die values.",
  ),
  makeQuestion(
    "cs109-lect1-q34",
    "easy",
    "A queue contains \\(n\\) distinct tasks, where \\(n\\ge3\\). Only three designated tasks are permitted in the first position; after the first task is chosen, the remaining tasks may appear in any order. Which statements are correct?",
    [
      ["The number of valid queues is \\(3(n-1)!\\).", true],
      [
        "The count can be seen as three disjoint first-task cases, each with \\((n-1)!\\) completions.",
        true,
      ],
      [
        "If the first-position restriction were removed, the count would become \\(n(n-1)!=n!\\).",
        true,
      ],
      [
        "The count is \\(\\binom{n}{3}(n-1)!\\) because any three tasks must first be selected for the queue.",
        false,
      ],
    ],
    "The restriction applies only to the identity of the first task, giving three choices, and each choice leaves all \\(n-1\\) remaining tasks to be permuted. Selecting a three-task subset would impose a different condition and greatly overcount the allowed first-task decision.",
  ),
  makeQuestion(
    "cs109-lect1-q35",
    "hard",
    "A 7-bit string is valid if it starts with 10 or ends with 01. Which statements are correct?",
    [
      [
        "The two qualifying sets each contain \\(2^5=32\\) strings, their intersection contains \\(2^3=8\\), and the union contains 56.",
        true,
      ],
      ["The three free intersection bits are positions 3, 4, and 5.", true],
      ["The union contains 64 strings because \\(32+32=64\\).", false],
      [
        "The intersection contains \\(2^5\\) strings because each individual condition leaves five free bits.",
        false,
      ],
    ],
    "Each single condition fixes two bits, but satisfying both fixes four distinct endpoint bits and leaves only the three middle positions free. Inclusion-exclusion therefore gives \\(32+32-8=56\\), not the uncorrected sum of 64.",
  ),
];
