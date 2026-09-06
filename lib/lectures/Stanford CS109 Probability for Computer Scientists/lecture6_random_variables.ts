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
    chapter: 6,
    difficulty,
    prompt,
    options: options.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

export const stanfordCS109Lecture6RandomVariablesQuestions: Question[] = [
  makeQuestion(
    "cs109-lect6-q01",
    "easy",
    "A probability problem is analyzed entirely within an event \\(G\\) with \\(P(G)>0\\). Which familiar probability rules remain valid when every term is conditioned on \\(G\\)?",
    [
      ["Complement: \\(P(E^c\\mid G)=1-P(E\\mid G)\\).", true],
      ["Chain rule: \\(P(E\\cap F\\mid G)=P(E\\mid F,G)P(F\\mid G)\\).", true],
      [
        "Total probability across a partition inside the conditioned world.",
        true,
      ],
      [
        "Bayes' theorem with \\(G\\) included consistently in every probability.",
        true,
      ],
    ],
    "Conditioning restricts the sample space to the world where \\(G\\) occurred and renormalizes probabilities there. Within that world the probability axioms and the formulas derived from them still hold, provided the common condition is carried consistently and all required denominators are positive.",
  ),
  makeQuestion(
    "cs109-lect6-q02",
    "easy",
    "Which statements correctly express that events \\(E\\) and \\(F\\) are conditionally independent given \\(G\\)?",
    [
      ["\\(P(E\\cap F\\mid G)=P(E\\mid G)P(F\\mid G)\\).", true],
      [
        "When the relevant probabilities are positive, \\(P(E\\mid F,G)=P(E\\mid G)\\).",
        true,
      ],
      [
        "After restricting to \\(G\\), learning \\(F\\) supplies no further information about \\(E\\).",
        true,
      ],
      ["\\(P(E\\cap F\\cap G)=P(E)P(F)P(G)\\) must hold.", false],
    ],
    "Conditional independence applies the ordinary independence criterion inside the probability space defined by \\(G\\). It does not assert mutual independence of \\(E,F,G\\) under the original distribution, so the unconditioned three-way product identity is neither required nor generally true.",
  ),
  makeQuestion(
    "cs109-lect6-q03",
    "easy",
    "Which claims correctly compare marginal independence with conditional independence?",
    [
      [
        "Events can be dependent marginally yet independent after conditioning on another event.",
        true,
      ],
      [
        "Events can be independent marginally yet dependent after conditioning on another event.",
        true,
      ],
      [
        "Marginal independence always implies conditional independence under every possible condition.",
        false,
      ],
      [
        "Conditional independence given one event always implies marginal independence.",
        false,
      ],
    ],
    "Conditioning changes the reference population and can reveal, remove, or introduce statistical relationships. Therefore neither marginal independence nor independence under one condition logically guarantees the other; each relationship must be checked in the probability space where it is being used.",
  ),
  makeQuestion(
    "cs109-lect6-q04",
    "easy",
    "Which description best defines a discrete random variable?",
    [
      [
        "A numerical variable whose realized value is uncertain and whose possible values have associated probabilities.",
        true,
      ],
      ["An event that is true whenever any outcome occurs.", false],
      ["A fixed but unknown constant that has no probability model.", false],
      [
        "A probability value that must itself lie strictly between zero and one.",
        false,
      ],
    ],
    "A random variable maps experimental outcomes to values, and uncertainty about the outcome induces a probability distribution over those values. It is not itself a Boolean event or a single probability; events arise by making claims such as \\(X=2\\) or \\(X<5\\) about the variable.",
  ),
  makeQuestion(
    "cs109-lect6-q05",
    "easy",
    "Let \\(Y\\) be the number of heads in five coin flips. Which expression is an event rather than a random variable?",
    [
      ["\\(Y=2\\)", true],
      ["\\(Y\\)", false],
      ["The number of heads produced by the experiment", false],
      ["The mapping from each flip sequence to its head count", false],
    ],
    "The variable \\(Y\\) assigns a head count to every five-flip outcome, while the equality \\(Y=2\\) selects the outcomes satisfying a true-or-false statement. A random variable, its verbal definition, and its outcome-to-value mapping describe the same object rather than a Boolean subset of the sample space.",
  ),
  makeQuestion(
    "cs109-lect6-q06",
    "easy",
    "For a random variable \\(Y\\), which probability expressions are meaningful events to which probability can be assigned?",
    [
      ["\\(P(Y=3)\\)", true],
      ["\\(P(Y<2)\\)", true],
      ["\\(P(Y)\\) without any condition on the value", false],
      ["\\(P(3)\\) when 3 is merely a possible numerical value", false],
    ],
    "Probability is assigned to events, so an equality or inequality involving \\(Y\\) creates a set of experimental outcomes whose probability is defined. The random variable alone and a bare number do not state true-or-false conditions and therefore do not identify events to measure.",
  ),
  makeQuestion(
    "cs109-lect6-q07",
    "easy",
    "Let \\(Y\\) count heads in three coin flips. Which statements about its possible values are correct?",
    [
      ["Its support is \\(\\{0,1,2,3\\}\\).", true],
      ["\\(P(Y=4)=0\\).", true],
      ["The event \\(Y=1\\) contains three distinct flip sequences.", true],
      [
        "The variable can take the value HTT because random variables store full outcomes.",
        false,
      ],
    ],
    "A head-count variable records a number rather than the underlying sequence, so its values range from zero through three. The value 1 groups HTT, THT, and TTH into one event, while four heads are impossible in three flips and a sequence such as HTT is an outcome mapped to a value, not a value of \\(Y\\).",
  ),
  makeQuestion(
    "cs109-lect6-q08",
    "easy",
    "Which statements correctly describe a probability mass function (PMF) for a discrete random variable \\(X\\)?",
    [
      ["It maps each possible value \\(x\\) to \\(P(X=x)\\).", true],
      ["It may be represented by a formula, table, graph, or program.", true],
      ["Its values are nonnegative and sum to one over the support.", true],
      [
        "It packages the full distribution over the discrete values of \\(X\\).",
        true,
      ],
    ],
    "A PMF is a function whose input is a possible discrete value and whose output is the probability of the corresponding equality event. Different representations can encode the same function, but every valid PMF must assign nonnegative mass and account for exactly one unit of total probability.",
  ),
  makeQuestion(
    "cs109-lect6-q09",
    "easy",
    "Why must \\(\\sum_k P(Y=k)=1\\) when the sum ranges over every possible value of a discrete random variable \\(Y\\)?",
    [
      [
        "The events \\(Y=k\\) are mutually exclusive for distinct values of \\(k\\).",
        true,
      ],
      [
        "Their union is the entire sample space because \\(Y\\) must take one supported value.",
        true,
      ],
      [
        "The events are independent, so their probabilities multiply to one.",
        false,
      ],
      ["Every individual value must have the same probability.", false],
    ],
    "A realized random variable cannot equal two different values at once, so the equality events form disjoint pieces. Together those pieces cover every possible outcome, and the probability axiom for the full sample space makes their sum one without requiring equal probabilities or independence.",
  ),
  makeQuestion(
    "cs109-lect6-q10",
    "easy",
    "Let \\(X\\) be the result of one fair six-sided die roll. Which statements about its PMF are correct?",
    [
      ["\\(P(X=x)=1/6\\) for \\(x\\in\\{1,2,3,4,5,6\\}\\).", true],
      [
        "\\(P(X=x)=0\\) for integer values outside \\(1\\) through \\(6\\).",
        true,
      ],
      ["The six supported masses sum to one.", true],
      [
        "The PMF assigns probability \\(1/6\\) to every real number between 1 and 6.",
        false,
      ],
    ],
    "The die-result variable has exactly six supported values, each corresponding to one equally likely face. Values outside that discrete support receive zero mass; assigning mass \\(1/6\\) to infinitely many real numbers would neither describe a discrete die nor produce a normalized PMF.",
  ),
  makeQuestion(
    "cs109-lect6-q11",
    "easy",
    "Let \\(S\\) be the sum of two independent fair dice. Which statements correctly describe its PMF?",
    [
      ["Its support is \\(\\{2,3,\\ldots,12\\}\\).", true],
      ["\\(P(S=7)=6/36\\).", true],
      [
        "The masses increase from sum 2 through sum 7 and then decrease symmetrically.",
        true,
      ],
      [
        "The masses sum to one because the eleven sum events partition the 36 ordered die outcomes.",
        true,
      ],
    ],
    "Each possible sum collects the ordered die pairs that produce it, with six pairs producing the central sum of 7 and fewer pairs toward the extremes. The sum events are disjoint and exhaustive, so their triangular PMF covers all 36 equally likely ordered outcomes and has total mass one.",
  ),
  makeQuestion(
    "cs109-lect6-q12",
    "easy",
    "What is the expected value of the result \\(X\\) of one fair six-sided die roll?",
    [
      ["\\(E[X]=3.5\\)", true],
      [
        "\\(E[X]=3\\), because 3 is the largest value below the midpoint.",
        false,
      ],
      ["\\(E[X]=1/6\\), because every outcome has that probability.", false],
      [
        "The expectation is undefined because 3.5 is not a possible die result.",
        false,
      ],
    ],
    "Expectation is the probability-weighted sum \\(\\sum_{x=1}^6 x/6=21/6=3.5\\). It summarizes the distribution's center and need not itself be a supported outcome, so the impossibility of rolling 3.5 does not make the expectation invalid.",
  ),
  makeQuestion(
    "cs109-lect6-q13",
    "medium",
    "Two independent frisbees each land heads with probability \\(p\\). Call the result even when both orientations match. Which statements are correct?",
    [
      ["The two matching cases are HH and TT.", true],
      ["The cases are mutually exclusive, so their probabilities add.", true],
      ["The even probability is \\(p^2+(1-p)^2\\).", true],
      ["For \\(p=0.6\\), the even probability is \\(0.36+0.16=0.52\\).", true],
    ],
    "Independence multiplies the two head probabilities for HH and the two tail probabilities for TT. Since a pair of flips cannot be both HH and TT, the cases are disjoint and their masses add; the result exceeds one half when the two orientations are not equally likely.",
  ),
  makeQuestion(
    "cs109-lect6-q14",
    "medium",
    "Two players have independent frisbees with the same unknown head probability \\(p\\). They repeat whenever the orientations match and otherwise the player showing heads wins. Which statements explain why the procedure is fair?",
    [
      [
        "In a decisive round, HT and TH each have probability \\(p(1-p)\\).",
        true,
      ],
      [
        "Conditioned on a decisive round, each ordering therefore has probability \\(1/2\\).",
        true,
      ],
      [
        "Repeating matching rounds treats both players symmetrically and does not favor either decisive ordering.",
        true,
      ],
      [
        "The procedure is fair because the original frisbees must have \\(p=1/2\\).",
        false,
      ],
    ],
    "The two decisive orientations have equal probability for every common value of \\(p\\), even when the individual frisbees are biased. Conditioning on eventually reaching one of those orientations normalizes equal masses to one half each, while symmetric repeats merely postpone that decisive round.",
  ),
  makeQuestion(
    "cs109-lect6-q15",
    "medium",
    "A gene \\(G_5\\) affects a trait \\(T\\) only by influencing another gene \\(G_2\\). Which statements fit the proposed conditional-independence pattern?",
    [
      [
        "Before observing \\(G_2\\), \\(G_5\\) and \\(T\\) may be dependent because \\(G_5\\) changes the chance that \\(G_2\\) is expressed.",
        true,
      ],
      [
        "After conditioning on the state of \\(G_2\\), learning \\(G_5\\) may provide no further information about \\(T\\).",
        true,
      ],
      [
        "Conditional independence proves from observational data alone that the proposed causal direction is correct.",
        false,
      ],
      [
        "The pattern requires \\(G_5\\) and \\(T\\) to be marginally independent as well.",
        false,
      ],
    ],
    "A mediated pathway can create marginal dependence because the upstream gene shifts the mediator, which shifts the trait. Once the mediator is fixed, the upstream gene may add no predictive information, but that statistical pattern alone supports only a causal hypothesis and does not prove direction or eliminate alternatives.",
  ),
  makeQuestion(
    "cs109-lect6-q16",
    "medium",
    "Within the event \\(G\\), suppose \\(P(E\\mid G)=0.4\\), \\(P(F\\mid G)=0.5\\), and \\(P(E\\cap F\\mid G)=0.2\\). What follows?",
    [
      [
        "\\(E\\) and \\(F\\) are conditionally independent given \\(G\\).",
        true,
      ],
      ["\\(E\\) and \\(F\\) must be mutually exclusive given \\(G\\).", false],
      [
        "\\(E\\) and \\(F\\) must be marginally independent without conditioning.",
        false,
      ],
      ["\\(P(E\\mid F,G)=0.2\\).", false],
    ],
    "The conditioned joint probability equals the product \\(0.4\\times0.5=0.2\\), exactly satisfying conditional independence. Their conditioned overlap is positive rather than exclusive, nothing here determines the unconditioned relationship, and \\(P(E\\mid F,G)=0.2/0.5=0.4\\).",
  ),
  makeQuestion(
    "cs109-lect6-q17",
    "medium",
    "A user chooses 30 distinct titles uniformly from 13,000. What is the probability that the chosen set contains four particular titles?",
    [
      ["\\(\\dfrac{\\binom{12996}{26}}{\\binom{13000}{30}}\\)", true],
      [
        "\\(\\dfrac{\\binom{4}{4}\\binom{12996}{30}}{\\binom{13000}{30}}\\)",
        false,
      ],
      ["\\(\\dfrac{\\binom{13000}{4}}{\\binom{13000}{30}}\\)", false],
      ["\\((30/13000)^4\\)", false],
    ],
    "Every 30-title set is an equally likely outcome. A favorable set must include the four specified titles and choose its remaining 26 titles from the other 12,996; the alternatives choose the wrong number of remaining titles, compare incompatible counts, or treat dependent without-replacement inclusions as independent draws.",
  ),
  makeQuestion(
    "cs109-lect6-q18",
    "medium",
    "A recommendation model assumes movie events \\(E_1,\\ldots,E_4\\) are conditionally independent given a latent preference event \\(K\\). Which simplifications are justified?",
    [
      ["\\(P(E_4\\mid E_1,E_2,E_3,K)=P(E_4\\mid K)\\).", true],
      [
        "Given \\(K\\), a joint likelihood for the movie events can factor into a product of conditionally evaluated terms.",
        true,
      ],
      ["The model may ignore uncertainty about whether \\(K\\) holds.", false],
      [
        "The movie events must also be independent before conditioning on \\(K\\).",
        false,
      ],
    ],
    "Conditional independence says that once the shared preference state is known, the other movie observations add no information about \\(E_4\\), and their conditioned joint behavior can factor. The latent state may still need to be inferred, and marginal dependence can remain after averaging over its possible values.",
  ),
  makeQuestion(
    "cs109-lect6-q19",
    "medium",
    "Let \\(Y\\) count heads in five independent flips of a coin with head probability \\(p\\). Which statements about its PMF are correct?",
    [
      [
        "\\(P(Y=k)=\\binom{5}{k}p^k(1-p)^{5-k}\\) for \\(k=0,\\ldots,5\\).",
        true,
      ],
      ["\\(P(Y=2)=\\binom{5}{2}p^2(1-p)^3\\).", true],
      [
        "The six PMF values sum to one by the binomial expansion of \\(p+(1-p)\\).",
        true,
      ],
      [
        "\\(P(Y=3)=p^3(1-p)^2\\) because a PMF describes only one ordering.",
        false,
      ],
    ],
    "The head-count event combines all \\(\\binom{5}{k}\\) disjoint placements of \\(k\\) heads, each with the same product probability. Summing the resulting binomial terms gives \\([p+(1-p)]^5=1\\); omitting the coefficient counts only one specified sequence rather than the random variable's full value event.",
  ),
  makeQuestion(
    "cs109-lect6-q20",
    "medium",
    "A proposed PMF assigns masses 0.2, 0.5, and 0.3 to values \\(-1,0,2\\), respectively. Which statements are correct?",
    [
      ["The proposal is normalized because \\(0.2+0.5+0.3=1\\).", true],
      ["Its support is \\(\\{-1,0,2\\}\\).", true],
      ["\\(P(X>0)=0.3\\).", true],
      ["\\(E[X]=(-1)(0.2)+0(0.5)+2(0.3)=0.4\\).", true],
    ],
    "The three nonnegative masses sum to one, so they define a valid distribution on the listed support. Only the value 2 satisfies \\(X>0\\), and weighting each support value by its mass gives the expectation 0.4, which need not itself appear in the support.",
  ),
  makeQuestion(
    "cs109-lect6-q21",
    "medium",
    "A discrete random variable has \\(P(X=1)=0.3\\), \\(P(X=2)=0.2\\), and \\(P(X=4)=0.5\\). Which statements correctly compute its expectation?",
    [
      ["\\(E[X]=1(0.3)+2(0.2)+4(0.5)=2.7\\).", true],
      [
        "The value 4 contributes more to the expectation than 1 because both its value and its probability are larger.",
        true,
      ],
      [
        "\\(E[X]=(1+2+4)/3=7/3\\) because expectation averages the distinct support values equally.",
        false,
      ],
      [
        "\\(E[X]=0.5\\) because the most likely value determines the mean.",
        false,
      ],
    ],
    "Expectation weights support values by their probabilities rather than treating distinct values equally or keeping only the modal mass. Here the weighted contributions are 0.3, 0.4, and 2.0, so the value 4 dominates the total and the mean is 2.7.",
  ),
  makeQuestion(
    "cs109-lect6-q22",
    "medium",
    "A school has classes of sizes 5, 10, and 150, with each student enrolled in exactly one class. Which statements correctly compare two sampling procedures?",
    [
      [
        "Choosing one class uniformly gives expected class size \\((5+10+150)/3=55\\).",
        true,
      ],
      [
        "Choosing one student uniformly gives expected class size \\(5(5/165)+10(10/165)+150(150/165)\\approx137.18\\).",
        true,
      ],
      [
        "The student-based procedure size-biases the distribution because larger classes contain more possible sampled students.",
        true,
      ],
      [
        "Both procedures must have the same expectation because their random variables share the support \\(\\{5,10,150\\}\\).",
        false,
      ],
    ],
    "The possible class sizes match, but the probabilities attached to them do not. Uniform class sampling assigns one third to each size, whereas uniform student sampling assigns probability in proportion to class enrollment, heavily weighting the 150-person class and producing a much larger expectation.",
  ),
  makeQuestion(
    "cs109-lect6-q23",
    "medium",
    "Which statements correctly describe properties of expectation for discrete random variables when the relevant sums exist?",
    [
      ["\\(E[aX+b]=aE[X]+b\\) for constants \\(a,b\\).", true],
      [
        "\\(E[X+Y]=E[X]+E[Y]\\) even when \\(X\\) and \\(Y\\) are dependent.",
        true,
      ],
      ["\\(E[g(X)]=\\sum_x g(x)P(X=x)\\).", true],
      [
        "An expected value can lie between supported values without itself being a possible outcome.",
        true,
      ],
    ],
    "Expectation is linear without an independence assumption, and a function of a discrete random variable can be averaged directly over the original variable's PMF. Because the result is a weighted center rather than a predicted single outcome, it may fall between values in the support.",
  ),
  makeQuestion(
    "cs109-lect6-q24",
    "medium",
    "Random variables \\(X\\) and \\(Y\\) may be dependent, but \\(E[X]=3\\) and \\(E[Y]=-1\\). What is \\(E[2X+Y+5]\\)?",
    [
      ["10", true],
      ["7", false],
      ["12", false],
      [
        "It cannot be determined without the joint PMF of \\(X\\) and \\(Y\\).",
        false,
      ],
    ],
    "Linearity gives \\(E[2X+Y+5]=2E[X]+E[Y]+5=6-1+5=10\\). Dependence affects many joint quantities but not the expectation of a sum, so no joint PMF or independence assumption is needed for this calculation.",
  ),
  makeQuestion(
    "cs109-lect6-q25",
    "hard",
    "A discrete random variable \\(X\\) has PMF \\(p_X(x)\\), and \\(g\\) is a real-valued function. Which statements correctly use \\(E[g(X)]=\\sum_x g(x)p_X(x)\\)?",
    [
      [
        "The sum ranges over supported values of \\(X\\), not separately over outcomes with identical \\(X\\)-values.",
        true,
      ],
      ["The formula can compute \\(E[X^2]\\) by taking \\(g(x)=x^2\\).", true],
      ["It avoids first deriving a separate PMF for \\(g(X)\\).", true],
      ["For \\(g(x)=ax+b\\), it yields \\(aE[X]+b\\).", true],
    ],
    "The law of the unconscious statistician averages the transformed value using the original PMF, so it can compute moments and nonlinear rewards directly. Applying it to an affine function separates the weighted sum into \\(a\\sum_x xp_X(x)+b\\sum_x p_X(x)\\), and normalization turns the second sum into \\(b\\).",
  ),
  makeQuestion(
    "cs109-lect6-q26",
    "hard",
    "A random variable \\(X\\) takes values 0, 1, and 2 with probabilities 0.2, 0.5, and 0.3. Let \\(Y=2X-1\\). Which statements are correct?",
    [
      ["The support of \\(Y\\) is \\(\\{-1,1,3\\}\\).", true],
      ["\\(P(Y=1)=0.5\\).", true],
      ["\\(E[Y]=2E[X]-1=1.2\\).", true],
      ["\\(P(Y=3)=P(X=3)=0\\).", false],
    ],
    "The transformation maps the three supported \\(X\\)-values to -1, 1, and 3 without merging them, so their probabilities transfer to those images. Since \\(E[X]=1.1\\), linearity gives 1.2; the event \\(Y=3\\) corresponds to \\(X=2\\), not \\(X=3\\), and has probability 0.3.",
  ),
  makeQuestion(
    "cs109-lect6-q27",
    "hard",
    "Let \\(D_1\\) and \\(D_2\\) each have the fair-die PMF, and define \\(S=D_1+D_2\\). Which statements correctly compute \\(E[S]\\)?",
    [
      ["Linearity gives \\(E[S]=E[D_1]+E[D_2]=3.5+3.5=7\\).", true],
      [
        "The expectation calculation does not require independence, although independence affects the joint distribution and the PMF of \\(S\\).",
        true,
      ],
      [
        "\\(E[S]=3.5^2=12.25\\) because the die expectations must multiply.",
        false,
      ],
      [
        "The triangular PMF of \\(S\\) implies that the expectation must equal the most likely mass \\(1/6\\).",
        false,
      ],
    ],
    "Expectation of a sum is the sum of expectations for dependent or independent variables alike, so the two die means add to 7. Multiplication pertains to products or independent event probabilities, and the height of the PMF at its mode is a probability rather than a possible value or expectation of the sum.",
  ),
  makeQuestion(
    "cs109-lect6-q28",
    "hard",
    "An object occupies exactly one of locations \\(L_1,\\ldots,L_m\\). After observation \\(O\\), which expression correctly gives the posterior probability of location \\(L_j\\)?",
    [
      [
        "\\(P(L_j\\mid O)=\\dfrac{P(O\\mid L_j)P(L_j)}{\\sum_{i=1}^m P(O\\mid L_i)P(L_i)}\\)",
        true,
      ],
      ["\\(P(L_j\\mid O)=\\dfrac{P(L_j\\mid O)P(O)}{P(L_j)}\\)", false],
      ["\\(P(L_j\\mid O)=\\dfrac{P(O\\mid L_j)}{\\sum_i P(L_i)}\\)", false],
      ["\\(P(L_j\\mid O)=P(O\\mid L_j)P(L_j)\\)", false],
    ],
    "Bayes' theorem gives each location an unnormalized weight equal to its observation likelihood times its prior. The law of total probability sums those weights over the mutually exclusive and exhaustive locations to obtain \\(P(O)\\), which normalizes the selected weight into a posterior distribution.",
  ),
  makeQuestion(
    "cs109-lect6-q29",
    "hard",
    "A latent preference \\(K\\) influences whether a user watches either of two movies \\(E\\) and \\(F\\). Which statements can hold in a model where \\(E\\) and \\(F\\) are conditionally independent given \\(K\\)?",
    [
      [
        "The movies can be marginally dependent because observing one changes beliefs about the unobserved preference \\(K\\).",
        true,
      ],
      [
        "Within each fixed state of \\(K\\), their joint watch probability can factor into the product of conditioned probabilities.",
        true,
      ],
      [
        "The movies must be marginally independent after averaging over the possible states of \\(K\\).",
        false,
      ],
      [
        "Observing \\(E\\) must change \\(P(F\\mid K)\\) even when the value of \\(K\\) is already known.",
        false,
      ],
    ],
    "A shared latent cause can correlate the movie events in the overall population, because one watched movie provides evidence about the preference state that also predicts the other. Once that state is fixed, the conditional-independence assumption removes the extra information supplied by the other movie and permits factorization.",
  ),
  makeQuestion(
    "cs109-lect6-q30",
    "hard",
    "For a discrete \\(X\\), suppose \\(P(X=0,G)=0.1\\), \\(P(X=1,G)=0.3\\), \\(P(X=2,G)=0.1\\), and \\(P(G)=0.5\\). Which statements about the conditional PMF given \\(G\\) are correct?",
    [
      ["\\(P(X=1\\mid G)=0.3/0.5=0.6\\).", true],
      [
        "The conditional masses for \\(X=0,1,2\\) are 0.2, 0.6, and 0.2 and sum to one.",
        true,
      ],
      ["\\(E[X\\mid G]=0(0.2)+1(0.6)+2(0.2)=1\\).", true],
      [
        "\\(P(X=2\\mid G)=0.1\\) because conditioning does not renormalize joint mass.",
        false,
      ],
    ],
    "Conditioning divides every joint mass inside \\(G\\) by the total mass \\(P(G)=0.5\\), producing a normalized PMF on the restricted world. That renormalization doubles each listed joint mass and gives conditional expectation 1; retaining 0.1 would confuse joint and conditional probability.",
  ),
  makeQuestion(
    "cs109-lect6-q31",
    "hard",
    "Two discrete random variables have the same expected value but different PMFs. Which statements correctly describe what can still differ between them?",
    [
      ["Their probabilities of exceeding a threshold can differ.", true],
      ["Their most likely values can differ.", true],
      [
        "Their tail risks can differ even though their weighted centers match.",
        true,
      ],
      [
        "The expectation alone is therefore a lossy summary of the distribution.",
        true,
      ],
    ],
    "A single weighted average does not determine how probability mass is arranged around that center. Distinct PMFs can share a mean while assigning different mass to typical values, extremes, and decision-relevant events, so comparing expectations alone can conceal substantial differences in behavior and risk.",
  ),
  makeQuestion(
    "cs109-lect6-q32",
    "hard",
    "In a St. Petersburg-style game, \\(N\\) is the number of heads before the first tail and the payout is \\(X=2^N\\) dollars. The coin is fair. Which statements are correct for the uncapped game?",
    [
      ["\\(P(N=i)=2^{-(i+1)}\\) for \\(i=0,1,2,\\ldots\\).", true],
      [
        "Each term in \\(E[X]=\\sum_{i=0}^{\\infty}2^i2^{-(i+1)}\\) equals \\(1/2\\), so the expectation diverges.",
        true,
      ],
      [
        "The expectation is finite because large payouts have probabilities approaching zero.",
        false,
      ],
      [
        "The most likely payout is infinite because the expected payout is infinite.",
        false,
      ],
    ],
    "Reaching payout \\(2^i\\) requires \\(i\\) heads followed by a tail, giving probability \\(2^{-(i+1)}\\). The shrinking probability exactly offsets the growing payout in every expectation term, leaving infinitely many contributions of one half; this does not make an infinite payout likely or even possible on a finite play.",
  ),
  makeQuestion(
    "cs109-lect6-q33",
    "hard",
    "Modify the St. Petersburg game so payouts corresponding to \\(N>16\\) are not honored, while \\(X=2^N\\) is paid for \\(0\\le N\\le16\\). Which statements are correct?",
    [
      [
        "Each honored level contributes \\(2^i2^{-(i+1)}=1/2\\) to the expectation.",
        true,
      ],
      [
        "There are 17 honored levels, so the expected honored payout is \\(17/2=8.5\\) dollars.",
        true,
      ],
      ["The cap turns the divergent expectation into a finite sum.", true],
      [
        "The expected payout becomes \\(2^{16}\\) because that is the largest honored amount.",
        false,
      ],
    ],
    "The uncapped paradox comes from infinitely many equal one-half contributions, not from any single likely huge payment. Limiting honored outcomes to indices 0 through 16 leaves 17 such contributions and expectation 8.5; a maximum payout is merely a support bound and is not the probability-weighted average.",
  ),
  makeQuestion(
    "cs109-lect6-q34",
    "hard",
    "Why can expected payout alone be inadequate for deciding whether to play a high-stakes game?",
    [
      [
        "Expectation compresses an entire PMF into one number and hides how mass is distributed.",
        true,
      ],
      [
        "A rare, enormous advertised payout may be impossible to collect if the payer has limited resources.",
        true,
      ],
      [
        "A player's finite budget and sensitivity to losses can matter even when expected winnings are positive.",
        true,
      ],
      [
        "Changing real payout limits changes the random variable and can radically change its expectation.",
        true,
      ],
    ],
    "Expected value is mathematically useful, but a decision also depends on whether the stated payoff distribution is real and on how outcomes interact with finite resources and risk preferences. A solvency cap changes the actual PMF, while a learner who sees only the mean cannot recover the frequency or severity of losses and extreme gains.",
  ),
  makeQuestion(
    "cs109-lect6-q35",
    "hard",
    "For \\(k=0,1,\\ldots,n\\), a candidate PMF has the form \\(P(X=k)=c(k+1)\\). Which pair correctly normalizes the PMF and gives its expectation?",
    [
      ["\\(c=\\dfrac{2}{(n+1)(n+2)}\\) and \\(E[X]=\\dfrac{2n}{3}\\)", true],
      ["\\(c=\\dfrac{1}{n+1}\\) and \\(E[X]=\\dfrac{n}{2}\\)", false],
      ["\\(c=\\dfrac{2}{n(n+1)}\\) and \\(E[X]=\\dfrac{2n+1}{3}\\)", false],
      ["\\(c=\\dfrac{2}{(n+1)(n+2)}\\) and \\(E[X]=\\dfrac{n}{2}\\)", false],
    ],
    "Normalization requires \\(c\\sum_{k=0}^n(k+1)=c(n+1)(n+2)/2=1\\), which gives the stated value of \\(c\\). Then \\(E[X]=c\\sum_{k=0}^n k(k+1)\\); using the standard sums for \\(k\\) and \\(k^2\\) simplifies the result to \\(2n/3\\), while the alternatives correspond to uniform or off-by-one normalizations.",
  ),
];
