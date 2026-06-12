import { Question } from "../../../quiz";

type PrereqDifficulty = "easy" | "medium" | "hard";
type OptionSeed = readonly [text: string, isCorrect: boolean];

function makeQuestion(
  number: number,
  difficulty: PrereqDifficulty,
  prompt: string,
  optionSeeds: readonly OptionSeed[],
  explanation: string,
): Question {
  if (optionSeeds.length !== 4) {
    throw new Error(
      `Probability Lecture 0 question ${number} needs 4 options.`,
    );
  }

  return {
    id: `crash-probability-l0-q${String(number).padStart(2, "0")}`,
    chapter: 0,
    difficulty,
    prompt,
    options: optionSeeds.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

export const CrashCourseProbabilityL0Questions: Question[] = [
  makeQuestion(
    1,
    "easy",
    "A recipe mixes syrup and water in the ratio \\(3:7\\). Which statements are correct?",
    [
      ["Syrup is \\(\\frac{3}{10}\\) of the mixture.", true],
      ["Water is \\(70\\%\\) of the mixture.", true],
      ["The odds of syrup to water are \\(7:3\\).", false],
      ["A 500 ml mixture contains 150 ml of syrup.", true],
    ],
    "A ratio of \\(3:7\\) has \\(3+7=10\\) total parts, so syrup is three tenths and water is seven tenths of the mixture. The false statement reverses the order of the odds: syrup to water is \\(3:7\\), not \\(7:3\\).",
  ),
  makeQuestion(
    2,
    "easy",
    "The odds in favor of a result are \\(4:1\\). Which conversions are correct?",
    [
      ["The favorable share is \\(\\frac{4}{5}\\).", true],
      ["The unfavorable share is \\(25\\%\\).", false],
      ["The odds against the result are \\(1:4\\).", true],
      ["The favorable share is \\(\\frac{4}{1}=4\\).", false],
    ],
    "Odds compare favorable to unfavorable parts, while a share compares favorable parts to total parts. With odds \\(4:1\\), there are five total parts, so the favorable share is four fifths and the unfavorable share is one fifth, which is \\(20\\%\\), not \\(25\\%\\).",
  ),
  makeQuestion(
    3,
    "easy",
    "A normalized share is computed by \\(s=\\frac{a}{a+b}\\), where \\(a=6\\) and \\(b=9\\). Which statements are correct?",
    [
      ["\\(s=\\frac{6}{15}=0.4\\).", true],
      [
        "The complementary share for \\(b\\) is \\(\\frac{9}{15}=0.4\\).",
        false,
      ],
      ["The two shares sum to 1.", true],
      [
        "\\(s=\\frac{6}{9}\\) because only \\(b\\) belongs in the denominator.",
        false,
      ],
    ],
    "A normalized share divides one part by the total of all included parts. The denominator is \\(a+b=15\\), so the share for \\(a\\) is \\(0.4\\), the share for \\(b\\) is \\(0.6\\), and the two shares add to 1.",
  ),
  makeQuestion(
    4,
    "easy",
    "Rearrange \\(y=3x-5\\) to solve for \\(x\\). Which statements are correct?",
    [
      ["\\(x=\\frac{y+5}{3}\\).", true],
      ["If \\(y=10\\), then \\(x=\\frac{5}{3}\\).", false],
      ["\\(x=\\frac{y-5}{3}\\).", false],
      ["Substituting \\(x=5\\) into \\(3x-5\\) gives 15.", false],
    ],
    "To solve for \\(x\\), add 5 to both sides and then divide by 3. If \\(y=10\\), the correct value is \\((10+5)/3=5\\), and substituting \\(x=5\\) gives \\(3\\cdot5-5=10\\), not 15.",
  ),
  makeQuestion(
    5,
    "easy",
    "A temperature conversion is \\(F=\\frac{9}{5}C+32\\). Which rearrangements or substitutions are correct?",
    [
      ["\\(C=\\frac{5}{9}(F-32)\\).", true],
      ["If \\(C=20\\), then \\(F=68\\).", true],
      ["If \\(F=50\\), then \\(C=12\\).", false],
      ["\\(C=\\frac{9}{5}(F-32)\\).", false],
    ],
    "Solving for \\(C\\) requires subtracting 32 and then multiplying by the reciprocal \\(5/9\\). For \\(F=50\\), the correct calculation is \\(\\frac{5}{9}(50-32)=10\\), not 12, and the false rearrangement uses \\(9/5\\) again instead of undoing the original multiplication.",
  ),
  makeQuestion(
    6,
    "easy",
    "A function family is \\(f_a(x)=a x+2\\). Which statements are correct?",
    [
      ["The subscript \\(a\\) names a parameter of the function family.", true],
      ["If \\(a=3\\), then \\(f_a(4)=14\\).", true],
      [
        "Changing \\(a\\) can change the output for the same input \\(x\\).",
        true,
      ],
      [
        "If both \\(a\\) and the constant 2 are doubled, the output for every \\(x\\) is doubled.",
        true,
      ],
    ],
    "A subscript can label which member of a function family is being used; it is not automatically multiplication. With \\(a=3\\) and \\(x=4\\), the output is \\(3\\cdot4+2=14\\), and doubling both terms in the formula changes \\(ax+2\\) to \\(2ax+4=2(ax+2)\\).",
  ),
  makeQuestion(
    7,
    "easy",
    "For the sequence \\(x_1=4\\), \\(x_2=7\\), \\(x_3=10\\), and \\(x_4=13\\), which statements are correct?",
    [
      ["\\(x_3=10\\).", true],
      ["If \\(t=3\\), then \\(x_{t-1}=x_2=7\\).", true],
      ["The sequence increases by 4 each step.", false],
      ["\\(x_t\\) means \\(x\\) multiplied by \\(t\\).", false],
    ],
    "Subscripts identify positions in a sequence. They do not mean multiplication, so \\(x_3\\) is the third listed value and \\(x_{t-1}\\) points one position before \\(x_t\\); the listed sequence increases by 3 each step, not 4.",
  ),
  makeQuestion(
    8,
    "easy",
    "Numbers \\((2,5,1,5)\\) are assigned to labels \\((a,b,c,d)\\). Which max/min statements are correct?",
    [
      ["The maximum value is 5.", true],
      ["Both \\(b\\) and \\(d\\) attain the maximum.", true],
      ["The minimum value is 1.", true],
      [
        "\\(\\arg\\max\\) must return \\(a\\) because \\(a\\) is listed first.",
        false,
      ],
    ],
    "The maximum is the largest value, and the argmax is the label or set of labels where that largest value occurs. Here there is a tie for the maximum, while \\(c\\) is where the minimum value occurs.",
  ),
  makeQuestion(
    9,
    "easy",
    "Positive weights \\((4,1,5)\\) are normalized by their total. Which statements are correct?",
    [
      ["The total weight is 10.", true],
      ["The normalized weights are \\((0.4,0.1,0.5)\\).", true],
      ["The normalized weights sum to 1.", true],
      ["The middle normalized weight is \\(\\frac{1}{10}\\).", true],
    ],
    "Normalization divides every weight by the same total, so each entry becomes a share of all the weight being considered. The total is 10, so the normalized weights are \\((4/10,1/10,5/10)\\), and those entries sum to 1.",
  ),
  makeQuestion(
    10,
    "easy",
    "Which power and root calculations are correct?",
    [
      ["\\(3^2=9\\).", true],
      ["\\(\\sqrt{0.25}=0.5\\).", true],
      ["\\(16^{1/2}=8\\).", false],
      ["\\((-2)^2=-4\\).", false],
    ],
    "A square root asks for the nonnegative number whose square gives the original number, and an exponent of \\(1/2\\) represents a square root. Therefore \\(16^{1/2}=4\\), not 8, and squaring a negative number gives a positive result.",
  ),
  makeQuestion(
    11,
    "medium",
    "A retained set has original weights \\(0.40\\), \\(0.15\\), and \\(0.05\\). The weights are renormalized after discarding everything else. Which statements are correct?",
    [
      ["The retained total is \\(0.60\\).", true],
      [
        "The first retained renormalized weight is \\(\\frac{0.40}{0.60}=\\frac{2}{3}\\).",
        true,
      ],
      ["The third retained renormalized weight is \\(\\frac{1}{10}\\).", false],
      [
        "The retained weights already sum to 1, so no renormalization changes them.",
        false,
      ],
    ],
    "Renormalizing after a restriction means dividing each retained weight by the retained total. The retained total is \\(0.60\\), so the three new weights are \\(2/3\\), \\(1/4\\), and \\(1/12\\), not \\(1/10\\) for the third retained weight.",
  ),
  makeQuestion(
    12,
    "medium",
    "Find the normalizing constant \\(k\\) for weights \\((2k,3k,5k)\\) that must sum to 1. Which statements are correct?",
    [
      ["\\(2k+3k+5k=10k\\).", true],
      ["\\(k=0.1\\).", true],
      ["The normalized weights are \\((0.2,0.3,0.5)\\).", true],
      ["\\(k=10\\) because the coefficients sum to 10.", false],
    ],
    "The requirement that the entries sum to 1 gives \\(10k=1\\). Therefore \\(k=0.1\\), and multiplying the coefficients by \\(0.1\\) gives the normalized entries.",
  ),
  makeQuestion(
    13,
    "medium",
    "Let \\(f_{m,b}(x)=mx+b\\). Which statements are correct for \\(m=-2\\), \\(b=5\\), and \\(x=3\\)?",
    [
      ["\\(f_{m,b}(3)=-1\\).", true],
      ["The parameter \\(m\\) controls the coefficient of \\(x\\).", true],
      [
        "The parameter \\(b\\) is multiplied by \\(x\\) before \\(m\\) is added.",
        false,
      ],
      [
        "\\(f_{m,b}(3)=11\\) because the negative sign on \\(m\\) is ignored.",
        false,
      ],
    ],
    "The calculation is \\((-2)\\cdot3+5=-6+5=-1\\). The two parameters play different algebraic roles: \\(m\\) multiplies the input, while \\(b\\) is added afterward rather than multiplied by \\(x\\).",
  ),
  makeQuestion(
    14,
    "medium",
    "Let \\(g_c(x)=c x^2\\). Which statements are correct?",
    [
      ["If \\(c=2\\) and \\(x=-3\\), then \\(g_c(x)=18\\).", true],
      ["If \\(c=-1\\) and \\(x=4\\), then \\(g_c(x)=-16\\).", true],
      [
        "Changing \\(c\\) leaves all outputs unchanged as long as \\(x\\) is fixed.",
        false,
      ],
      ["If \\(x=-3\\), then \\(x^2=-9\\).", false],
    ],
    "The square is applied to \\(x\\) before multiplication by \\(c\\), so \\((-3)^2=9\\). A parameter such as \\(c\\) chooses a member of the function family, so changing it changes outputs for fixed nonzero inputs.",
  ),
  makeQuestion(
    15,
    "medium",
    "A sequence is defined by \\(x_1=2\\) and \\(x_{t+1}=3x_t-1\\). Which statements are correct?",
    [
      ["\\(x_2=5\\).", true],
      ["\\(x_3=14\\).", true],
      [
        "The recurrence uses the next term to compute the previous term.",
        false,
      ],
      ["\\(x_3=3x_1-1=5\\).", false],
    ],
    "The recurrence must be applied step by step: \\(x_2=3\\cdot2-1=5\\), then \\(x_3=3\\cdot5-1=14\\). The rule uses the previous term to compute the next term, and the shortcut expression applies the rule to the wrong value.",
  ),
  makeQuestion(
    16,
    "medium",
    "For \\(a=(3,-1,4,2)\\), which summation statements are correct?",
    [
      ["\\(\\sum_{i=1}^4 a_i=8\\).", true],
      ["\\(\\sum_{i=1}^4 a_i^2=30\\).", true],
      ["\\(\\sum_{i\\in\\{2,4\\}}a_i=3\\).", false],
      ["\\(\\sum_{i=1}^4 a_i\\) means multiplying all four entries.", false],
    ],
    "Summation notation represents repeated addition over the specified indices. The full sum is \\(3-1+4+2=8\\), the squared sum is \\(9+1+16+4=30\\), and the subset sum over indices 2 and 4 is \\(-1+2=1\\), not 3.",
  ),
  makeQuestion(
    17,
    "medium",
    "For \\(b=(2,\\frac{1}{2},5)\\), which product-notation statements are correct?",
    [
      ["\\(\\prod_{i=1}^3 b_i=5\\).", true],
      ["\\(\\prod_{i=1}^2 b_i=1\\).", true],
      ["\\(\\prod_{i=1}^3 b_i\\) means adding the three entries.", false],
      [
        "The product becomes \\(2+\\frac{1}{2}+5\\) if the factors are listed in increasing index order.",
        false,
      ],
    ],
    "Product notation represents repeated multiplication, not repeated addition. Here \\(2\\cdot\\frac{1}{2}\\cdot5=5\\), and listing factors in index order does not turn the operation into addition.",
  ),
  makeQuestion(
    18,
    "medium",
    "Values \\((10,20,50)\\) have weights \\((0.2,0.3,0.5)\\). Which weighted-average statements are correct?",
    [
      ["The weighted average is 33.", true],
      ["The value 50 contributes 25 to the weighted sum.", true],
      ["The unweighted mean is 33.", false],
      ["The weighted average is \\(10+20+50=80\\).", false],
    ],
    "A weighted average multiplies each value by its weight and then adds the products. The calculation is \\(0.2\\cdot10+0.3\\cdot20+0.5\\cdot50=33\\), while the unweighted mean is \\(80/3\\), not 33.",
  ),
  makeQuestion(
    19,
    "medium",
    "A class has 12 students scoring 70 and 8 students scoring 90. Which average statements are correct?",
    [
      ["The weighted average score is 78.", true],
      ["The total number of students is 20.", true],
      ["The score 70 has weight \\(0.6\\).", true],
      [
        "The average is 80 because \\((70+90)/2=80\\) ignores group sizes correctly.",
        false,
      ],
    ],
    "Group sizes become weights in the average. The correct calculation is \\((12\\cdot70+8\\cdot90)/20=1560/20=78\\), while averaging the two displayed scores equally ignores that there are more students with score 70.",
  ),
  makeQuestion(
    20,
    "medium",
    "Which exponent rules are correct for positive \\(a\\) and \\(b\\)?",
    [
      ["\\(a^2a^3=a^5\\).", true],
      ["\\(\\frac{a^5}{a^2}=a^3\\).", true],
      ["\\((ab)^2=ab^2\\).", false],
      ["\\((a+b)^2=a^2+b^2\\).", false],
    ],
    "Exponent rules apply cleanly to products and powers, but each factor inside a product must be squared. Thus \\((ab)^2=a^2b^2\\), and \\((a+b)^2\\) also has a middle term \\(2ab\\), so the two false statements each drop a required part.",
  ),
  makeQuestion(
    21,
    "medium",
    "Which statements about the natural exponential \\(e^x\\) are correct?",
    [
      ["\\(e^0=1\\).", true],
      ["\\(e^{u+v}=e^u+e^v\\).", false],
      ["\\(\\frac{e^5}{e^2}=e^3\\).", true],
      ["\\(e^{-2}=-e^2\\).", false],
    ],
    "The exponential turns addition in the exponent into multiplication, so \\(e^{u+v}=e^ue^v\\), not \\(e^u+e^v\\). It also turns subtraction in the exponent into division, and a negative exponent gives a reciprocal.",
  ),
  makeQuestion(
    22,
    "medium",
    "A quantity starts at 80 and is multiplied by \\(0.75\\) each step. Which statements are correct?",
    [
      ["After one step the quantity is 60.", true],
      ["After two steps the quantity is 45.", true],
      ["After \\(n\\) steps the quantity is \\(80(0.75n)\\).", false],
      ["After two steps the quantity is \\(80-2(0.75)\\).", false],
    ],
    "Repeated multiplicative change is represented with powers. Multiplying by \\(0.75\\) twice gives \\(80(0.75)^2=45\\), and after \\(n\\) steps the factor is \\((0.75)^n\\), not \\(0.75n\\).",
  ),
  makeQuestion(
    23,
    "medium",
    "Which logarithm statements are correct for positive inputs?",
    [
      ["\\(\\ln(1)=0\\).", true],
      ["\\(\\ln(ab)=\\ln a+\\ln b\\).", true],
      ["\\(\\ln(a^3)=\\ln a+3\\).", false],
      ["\\(\\ln(a+b)=\\ln a+\\ln b\\).", false],
    ],
    "Logarithms turn products into sums and powers into multipliers. Therefore \\(\\ln(a^3)=3\\ln a\\), and logarithms do not turn ordinary addition inside the logarithm into addition outside the logarithm.",
  ),
  makeQuestion(
    24,
    "medium",
    "Solve logarithmic and exponential equations. Which statements are correct?",
    [
      ["If \\(\\log_2 x=5\\), then \\(x=32\\).", true],
      ["If \\(\\ln x=0\\), then \\(x=1\\).", true],
      ["If \\(e^x=7\\), then \\(x=e^7\\).", false],
      ["If \\(\\log_{10}x=3\\), then \\(x=30\\).", false],
    ],
    "A logarithm asks what exponent is needed to produce a number from a given base. Thus \\(e^x=7\\) gives \\(x=\\ln7\\), and \\(\\log_{10}x=3\\) means \\(x=10^3=1000\\), not \\(10\\cdot3\\).",
  ),
  makeQuestion(
    25,
    "medium",
    "A rectangular data block has shape \\(12\\times8\\times5\\). Which dimension calculations are correct?",
    [
      ["The block contains \\(12\\cdot8\\cdot5=480\\) entries.", true],
      ["Fixing the first coordinate leaves an \\(8\\times5\\) slice.", true],
      [
        "There are 5 slices of shape \\(12\\times8\\) along the first dimension.",
        false,
      ],
      ["The block contains \\(12+8+5=25\\) entries.", false],
    ],
    "A rectangular multidimensional block is counted by multiplying its dimensions. Fixing the first coordinate leaves an \\(8\\times5\\) slice, while slicing along the third dimension would give 5 slices of shape \\(12\\times8\\).",
  ),
  makeQuestion(
    26,
    "medium",
    "A batch contains 40 tables, each with shape \\(6\\times10\\). Which statements are correct?",
    [
      ["The batch contains \\(40\\cdot6\\cdot10=2400\\) entries.", true],
      ["Each table contains 60 entries.", true],
      [
        "Selecting one row from each table gives \\(6\\cdot10=60\\) entries total.",
        false,
      ],
      ["The batch contains \\(40+6+10=56\\) entries.", false],
    ],
    "The total count multiplies the number of tables by the entries per table. Selecting one row from each table keeps the batch dimension and the column dimension, so it gives \\(40\\cdot10=400\\) entries, not 60 total.",
  ),
  makeQuestion(
    27,
    "medium",
    "For Gaussian notation \\(X\\sim\\mathcal{N}(10,9)\\), which statements are correct?",
    [
      ["The mean is 10.", true],
      ["The variance is 9.", true],
      ["The standard deviation is 3.", true],
      ["The second parameter is the variance in this notation.", true],
    ],
    "In the notation \\(\\mathcal{N}(\\mu,\\sigma^2)\\), the second parameter is the variance. The standard deviation is the square root of the variance, so \\(\\sqrt{9}=3\\).",
  ),
  makeQuestion(
    28,
    "medium",
    "A normal quantity has mean 50 and standard deviation 4. Which standard-score statements are correct?",
    [
      ["A value of 58 is 2 standard deviations above the mean.", true],
      ["A value of 46 has standard score \\(-1\\).", true],
      ["A value equal to the mean has standard score 1.", false],
      ["A value of 54 has standard score 4.", false],
    ],
    "A standard score is computed as \\((x-\\mu)/\\sigma\\). For example, \\((58-50)/4=2\\), \\((46-50)/4=-1\\), and a value equal to the mean has standard score 0, not 1.",
  ),
  makeQuestion(
    29,
    "medium",
    "A measurement has mean \\(\\mu=20\\) and variance \\(\\sigma^2=16\\). Which statements are correct?",
    [
      ["The standard deviation is \\(\\sigma=4\\).", true],
      [
        "The interval one standard deviation from the mean is \\([18,22]\\).",
        false,
      ],
      ["The variance is the square of the standard deviation.", true],
      [
        "The interval one standard deviation from the mean is \\([4,36]\\).",
        false,
      ],
    ],
    "The standard deviation is the square root of the variance. One standard deviation from the mean means subtracting and adding 4 to 20, which gives \\([16,24]\\), not \\([18,22]\\).",
  ),
  makeQuestion(
    30,
    "medium",
    "A device succeeds on one attempt with share \\(0.98\\). Attempts are independent. Which complement calculations for 10 attempts are correct?",
    [
      ["The share for one failure is \\(0.02\\).", true],
      ["The share for no failures in 10 attempts is \\(0.02^{10}\\).", false],
      [
        "The share for at least one failure in 10 attempts is \\(1-0.98^{10}\\).",
        true,
      ],
      [
        "The share for at least one failure in 10 attempts is \\(0.02^{10}\\).",
        false,
      ],
    ],
    "The complement of at least one failure is no failures. Since attempts are independent, the no-failure share multiplies across attempts, giving \\(0.98^{10}\\), and the complement is \\(1-0.98^{10}\\); \\(0.02^{10}\\) would mean every attempt fails.",
  ),
  makeQuestion(
    31,
    "hard",
    "A delayed-value calculation uses \\(G=5+0.8\\cdot10+0.8^2\\cdot20\\). Which statements are correct?",
    [
      ["\\(G=25.8\\).", true],
      ["The third term uses \\(0.8^2=0.64\\).", true],
      ["The later values both receive weight \\(0.8\\).", false],
      [
        "\\(G=5+0.8(10+20)=29\\) because both later values get the same weight.",
        false,
      ],
    ],
    "The weights are \\(1\\), \\(0.8\\), and \\(0.8^2\\), so the later terms do not get the same multiplier. The calculation is \\(5+8+12.8=25.8\\), and the final term uses \\(0.64\\), not another \\(0.8\\).",
  ),
  makeQuestion(
    32,
    "hard",
    "For \\(0\\le r<1\\), the infinite geometric sum is \\(1+r+r^2+\\cdots\\). Which statements are correct?",
    [
      ["If \\(r=0.5\\), the sum is 2.", true],
      ["The general sum is \\(\\frac{1}{1-r}\\).", true],
      ["If \\(r=0.9\\), the sum is 10.", true],
      [
        "The general sum is \\(\\frac{r}{1-r}\\) because the first term is 1.",
        false,
      ],
    ],
    "The series starts with the term 1, so its total is \\(1/(1-r)\\). The expression \\(r/(1-r)\\) would be the sum \\(r+r^2+r^3+\\cdots\\), which starts one term later.",
  ),
  makeQuestion(
    33,
    "hard",
    "A finite geometric sum is \\(S_n=3+3r+3r^2+\\cdots+3r^{n-1}\\), with \\(r\\ne1\\). Which formulas are correct?",
    [
      ["\\(S_n=3\\frac{1-r^n}{1-r}\\).", true],
      ["If \\(r=0.5\\) and \\(n=3\\), then \\(S_n=5.25\\).", true],
      ["The final term shown by the pattern is \\(3r^n\\).", false],
      ["\\(S_n=3nr\\) for every \\(r\\ne1\\).", false],
    ],
    "A finite geometric sum multiplies the first term by \\((1-r^n)/(1-r)\\). For \\(r=0.5\\) and \\(n=3\\), the direct sum is \\(3+1.5+0.75=5.25\\), and the final term is \\(3r^{n-1}\\), not \\(3r^n\\).",
  ),
  makeQuestion(
    34,
    "hard",
    "The odds for three categories are \\(2:3:5\\), and the total count is 240. Which statements are correct?",
    [
      ["The category shares are \\(0.2\\), \\(0.3\\), and \\(0.5\\).", true],
      ["The three category counts are 48, 72, and 120.", true],
      ["The first category count is \\(\\frac{2}{10}\\cdot240\\).", true],
      ["The second category count is \\(\\frac{3}{5}\\cdot240\\).", false],
    ],
    "The ratio has \\(2+3+5=10\\) total parts, so each part corresponds to 24 counts when the total is 240. The second category uses \\(3/10\\) of the total, not \\(3/5\\), because the denominator must include all three ratio parts.",
  ),
  makeQuestion(
    35,
    "hard",
    "A quantity is defined by \\(q_i=\\frac{e^{z_i}}{e^{z_1}+e^{z_2}+e^{z_3}}\\). Let \\((z_1,z_2,z_3)=(0,\\ln 2,\\ln 3)\\). Which statements are correct?",
    [
      ["\\((e^{z_1},e^{z_2},e^{z_3})=(1,2,3)\\).", true],
      ["The denominator is \\(e^6\\).", false],
      ["\\((q_1,q_2,q_3)=(\\frac{1}{6},\\frac{1}{3},\\frac{1}{2})\\).", true],
      ["\\(q_3=3\\) because \\(e^{z_3}=3\\).", false],
    ],
    "The exponential and logarithm undo each other for positive values, so the unnormalized values are \\(1\\), \\(2\\), and \\(3\\). Normalization then divides each by their total 6; the denominator is the sum \\(1+2+3\\), not \\(e^6\\).",
  ),
  makeQuestion(
    36,
    "hard",
    "For \\(0<x<1\\), which negative-log statements are correct?",
    [
      ["\\(-\\ln(0.25)=\\ln 4\\).", true],
      ["\\(-\\ln(x)=\\ln(1/x)\\).", true],
      ["\\(-\\ln(1)=0\\).", true],
      ["\\(-\\ln(0.25)=-\\ln 4\\).", false],
    ],
    "The identity \\(-\\ln x=\\ln(x^{-1})=\\ln(1/x)\\) follows from the power rule for logarithms. Since \\(1/0.25=4\\), \\(-\\ln(0.25)=\\ln4\\), while \\(-\\ln4\\) would be a negative number rather than the positive value of \\(-\\ln(0.25)\\).",
  ),
  makeQuestion(
    37,
    "hard",
    "A variable takes values 2 and 8 with weights 0.75 and 0.25. Which variance calculations are correct?",
    [
      ["The weighted mean is 3.5.", true],
      [
        "The weighted variance is \\(0.75(2-3.5)^2+0.25(8-3.5)^2=6.75\\).",
        true,
      ],
      ["The standard deviation is \\(6.75\\).", false],
      [
        "The weighted mean is 5 because \\((2+8)/2=5\\) uses the weights.",
        false,
      ],
    ],
    "The weighted mean is \\(0.75\\cdot2+0.25\\cdot8=3.5\\). Variance then averages squared deviations using the same weights, and the standard deviation is the square root of that variance rather than the variance itself.",
  ),
  makeQuestion(
    38,
    "hard",
    "Each of 12 independent components remains working with share \\(0.97\\). Which statements are correct?",
    [
      ["The share with all 12 working is \\(0.97^{12}\\).", true],
      [
        "The share with at least one failed component is \\(1-0.97^{12}\\).",
        true,
      ],
      [
        "The expression \\(12\\cdot0.03\\) is not the exact share for at least one failure.",
        true,
      ],
      ["The share with all 12 working is \\(1-0.03^{12}\\).", false],
    ],
    "For independent components, all-working means multiplying the working share 12 times. At least one failure is the complement of all working, so it is \\(1-0.97^{12}\\), not the complement of all failing.",
  ),
  makeQuestion(
    39,
    "hard",
    "For \\(h(x)=\\sqrt{1-x}\\), which statements are correct?",
    [
      ["The real-valued domain satisfies \\(x\\le1\\).", true],
      ["\\(h(0.75)=0.5\\).", true],
      ["Solving \\(h(x)=0.2\\) gives \\(x=0.96\\).", true],
      ["\\(h(1.25)=0.5\\) because \\(1-1.25=0.25\\).", false],
    ],
    "For real square roots, the expression inside the root must be nonnegative, so \\(1-x\\ge0\\). Solving \\(\\sqrt{1-x}=0.2\\) gives \\(1-x=0.04\\), hence \\(x=0.96\\).",
  ),
  makeQuestion(
    40,
    "hard",
    "A score is computed as \\(S=\\sum_{t=1}^4 \\gamma^{t-1}r_t\\), with \\(\\gamma=0.5\\) and \\((r_1,r_2,r_3,r_4)=(4,0,8,16)\\). Which statements are correct?",
    [
      ["\\(S=4+0+2+2=8\\).", true],
      ["The fourth value is weighted by \\(0.5^3=0.125\\).", true],
      ["The weights are \\((1,0.5,0.25,0.0625)\\).", false],
      [
        "The fourth value is weighted by \\(0.5^4\\) because it is the fourth term.",
        false,
      ],
    ],
    "The exponent is \\(t-1\\), so the first term has exponent 0 and weight 1. The fourth term has exponent 3, giving weight \\(0.125\\), not \\(0.0625\\), and the total is \\(4+0+2+2=8\\).",
  ),
  makeQuestion(
    41,
    "easy",
    "Which statements correctly use Euler's number \\(e\\) and the natural logarithm \\(\\ln\\)?",
    [
      ["\\(\\ln(e)=1\\).", true],
      ["\\(e^0=1\\).", true],
      ["\\(\\ln(1)=0\\).", true],
      ["\\(e^{\\ln 5}=5\\).", true],
    ],
    "The natural logarithm is the inverse operation for exponentiation with base \\(e\\). These identities are core facts: \\(e^0=1\\), \\(\\ln(1)=0\\), \\(\\ln(e)=1\\), and applying \\(e^x\\) to \\(\\ln 5\\) returns 5.",
  ),
  makeQuestion(
    42,
    "easy",
    "Which logarithm values are correct?",
    [
      ["\\(\\log_{10}(1000)=3\\).", true],
      ["\\(\\log_2(32)=5\\).", true],
      ["\\(\\log_5(1)=0\\).", true],
      ["\\(\\log_3(9)=6\\).", false],
    ],
    "A logarithm gives the exponent needed on the base to produce the input. Since \\(10^3=1000\\), \\(2^5=32\\), and \\(5^0=1\\), the first three statements are correct; \\(3^2=9\\), so \\(\\log_3(9)=2\\), not 6.",
  ),
  makeQuestion(
    43,
    "easy",
    "Values \\((4,-2,4,7)\\) are assigned to labels \\((a,b,c,d)\\). Which extrema statements are correct?",
    [
      ["The maximum value is 7.", true],
      ["The argmax is \\(d\\).", true],
      ["The minimum value is \\(-2\\).", true],
      ["The argmin is \\(a\\) because \\(a\\) is listed first.", false],
    ],
    "The maximum and minimum are values, while argmax and argmin identify where those values occur. The largest value is 7 at label \\(d\\), and the smallest value is \\(-2\\) at label \\(b\\), not at the first label.",
  ),
  makeQuestion(
    44,
    "easy",
    "Which power and root statements are correct over the real numbers?",
    [
      ["\\(27^{1/3}=3\\).", true],
      ["\\(9^{-1}=\\frac{1}{9}\\).", true],
      ["\\(\\sqrt{49}=7\\).", true],
      ["\\(\\sqrt{-4}=2\\).", false],
    ],
    "A fractional exponent of \\(1/3\\) represents a cube root, and a negative exponent represents a reciprocal. The principal square root of 49 is 7, but \\(\\sqrt{-4}\\) is not a real number.",
  ),
  makeQuestion(
    45,
    "easy",
    "Which distribution and Gaussian-notation statements are correct?",
    [
      [
        "\\((0.2,0.5,0.3)\\) is a valid distribution over three outcomes.",
        true,
      ],
      ["The entries \\((0.2,0.5,0.3)\\) sum to 1.", true],
      [
        "In \\(X\\sim\\mathcal{N}(4,9)\\), the mean is 4 and the standard deviation is 3.",
        true,
      ],
      ["In \\(X\\sim\\mathcal{N}(4,9)\\), the standard deviation is 9.", false],
    ],
    "A finite distribution must have nonnegative entries that sum to 1. In the common notation \\(\\mathcal{N}(\\mu,\\sigma^2)\\), the second parameter is the variance, so \\(\\mathcal{N}(4,9)\\) has standard deviation \\(\\sqrt9=3\\).",
  ),
  makeQuestion(
    46,
    "medium",
    "Which logarithm identities are correct for positive inputs?",
    [
      ["\\(\\ln(12)-\\ln(3)=\\ln(4)\\).", true],
      ["\\(2\\ln(5)=\\ln(25)\\).", true],
      ["\\(\\ln(a/b)=\\ln a-\\ln b\\).", true],
      ["\\(\\ln(7-2)=\\ln 7-\\ln 2\\).", false],
    ],
    "Logarithms turn quotients into differences and powers into multipliers. They do not split subtraction inside the logarithm, so \\(\\ln(7-2)\\) is \\(\\ln5\\), not \\(\\ln7-\\ln2\\).",
  ),
  makeQuestion(
    47,
    "medium",
    "Which exponential equations are solved correctly?",
    [
      ["If \\(e^{2x}=9\\), then \\(x=\\ln 3\\).", true],
      ["If \\(2^x=\\frac{1}{8}\\), then \\(x=-3\\).", true],
      ["If \\(10^{x-1}=100\\), then \\(x=3\\).", true],
      ["If \\(e^{x+1}=e^4\\), then \\(x=5\\).", false],
    ],
    "Solving exponential equations means matching exponents or applying logarithms. Since \\(e^{2x}=9=e^{\\ln9}\\), \\(2x=\\ln9=2\\ln3\\), and \\(x=\\ln3\\); for the last equation, \\(x+1=4\\), so \\(x=3\\), not 5.",
  ),
  makeQuestion(
    48,
    "medium",
    "Which change-of-base and base-conversion statements are correct?",
    [
      [
        "\\(\\log_b a=\\frac{\\ln a}{\\ln b}\\) for positive \\(a\\) and valid base \\(b\\ne1\\).",
        true,
      ],
      ["\\(\\frac{\\log_2 8}{\\log_2 4}=\\frac{3}{2}\\).", true],
      ["\\(\\log_9 3=\\frac{1}{2}\\).", true],
      [
        "\\(\\log_a b=\\log_b a\\) for all valid positive bases and inputs.",
        false,
      ],
    ],
    "The change-of-base formula lets any logarithm be written as a ratio of natural logarithms or logarithms in another base. The reciprocal-looking expression is not symmetric: generally \\(\\log_a b=1/\\log_b a\\), not \\(\\log_b a\\).",
  ),
  makeQuestion(
    49,
    "medium",
    "Scores are \\(z=(1,3,2)\\). Which statements about transformations and extrema are correct?",
    [
      ["The argmax of \\(z\\) is the second position.", true],
      ["The argmax of \\((e^1,e^3,e^2)\\) is also the second position.", true],
      ["Adding 5 to every score leaves the argmax unchanged.", true],
      [
        "Multiplying every score by \\(-1\\) leaves both the argmax and argmin unchanged.",
        false,
      ],
    ],
    "Strictly increasing transformations such as exponentiation and adding the same constant preserve the order of scores. Multiplying by a negative number reverses the order, so it swaps maxima and minima rather than preserving both.",
  ),
  makeQuestion(
    50,
    "medium",
    "A table gives \\(f(-1)=3\\), \\(f(0)=5\\), \\(f(1)=5\\), and \\(f(2)=1\\). Which statements are correct?",
    [
      ["The maximum value of \\(f\\) on the table is 5.", true],
      ["The argmax set is \\(\\{0,1\\}\\).", true],
      ["The argmin is 2.", true],
      ["The argmax set is \\(\\{5\\}\\).", false],
    ],
    "The maximum value is the output 5, but argmax asks for the input locations that attain it. Here both inputs 0 and 1 attain the maximum, while input 2 gives the minimum value.",
  ),
  makeQuestion(
    51,
    "medium",
    "Which root-equation statements are correct over the real numbers?",
    [
      ["If \\(\\sqrt{x+5}=4\\), then \\(x=11\\).", true],
      ["If \\(x^{1/3}=-2\\), then \\(x=-8\\).", true],
      ["For all real \\(x\\), \\(\\sqrt{x^2}=x\\).", false],
      ["For \\(x\\ge0\\), \\((x^{1/2})^2=x\\).", true],
    ],
    "Solving \\(\\sqrt{x+5}=4\\) gives \\(x+5=16\\), so \\(x=11\\), and cubing both sides of \\(x^{1/3}=-2\\) gives \\(x=-8\\). The expression \\(\\sqrt{x^2}\\) equals \\(|x|\\), so it is not equal to \\(x\\) for negative \\(x\\).",
  ),
  makeQuestion(
    52,
    "medium",
    "Weights are defined as \\((e^0,e^{\\ln2},e^{\\ln5})\\) and then normalized by their total. Which statements are correct?",
    [
      ["The unnormalized weights are \\((1,2,5)\\).", true],
      ["The total unnormalized weight is 8.", true],
      ["The normalized third weight is \\(\\frac{5}{8}\\).", true],
      ["The normalized second weight is 2.", false],
    ],
    "The exponential and natural logarithm undo each other, so \\(e^{\\ln2}=2\\) and \\(e^{\\ln5}=5\\). Normalization divides by the total 8, so the second normalized weight is \\(2/8=1/4\\), not 2.",
  ),
  makeQuestion(
    53,
    "medium",
    "Let \\(X\\sim\\mathcal{N}(100,25)\\), using the \\(\\mathcal{N}(\\mu,\\sigma^2)\\) convention. Which standardization statements are correct?",
    [
      ["The standard deviation is 5.", true],
      ["A value of 110 has standard score 2.", true],
      ["A value of 95 has standard score \\(-1\\).", true],
      ["A value of 105 has standard score 5.", false],
    ],
    "The variance is 25, so the standard deviation is \\(\\sqrt{25}=5\\). Standard scores use \\((x-\\mu)/\\sigma\\), giving \\((110-100)/5=2\\), \\((95-100)/5=-1\\), and \\((105-100)/5=1\\).",
  ),
  makeQuestion(
    54,
    "medium",
    "A distribution over values \\((0,2,4)\\) has weights \\((0.25,0.5,0.25)\\). Which statements are correct?",
    [
      ["The weights sum to 1.", true],
      ["The weighted mean is 2.", true],
      ["The weighted variance is 2.", true],
      ["The standard deviation is \\(\\sqrt2\\).", true],
    ],
    "The weights form a valid distribution because they are nonnegative and sum to 1. The mean is \\(0.25\\cdot0+0.5\\cdot2+0.25\\cdot4=2\\), the second moment is 6, and the variance is \\(6-2^2=2\\), so the standard deviation is \\(\\sqrt2\\).",
  ),
  makeQuestion(
    55,
    "medium",
    "Which vectors can be valid finite distributions?",
    [
      ["\\((0.1,0.2,0.7)\\).", true],
      ["\\((\\frac{1}{3},\\frac{1}{3},\\frac{1}{3})\\).", true],
      ["\\((0.5,0.6,-0.1)\\).", false],
      ["\\((2,3,5)\\).", false],
    ],
    "A finite distribution must have entries that are all nonnegative and sum exactly to 1. The first two vectors satisfy both requirements; the third contains a negative entry, and the fourth contains positive weights that still need normalization.",
  ),
  makeQuestion(
    56,
    "hard",
    "A positive quantity satisfies \\(\\ln y=2x+\\ln3\\). Which statements are correct?",
    [
      ["\\(y=3e^{2x}\\).", true],
      ["When \\(x=0\\), \\(y=3\\).", true],
      ["Increasing \\(x\\) by 1 multiplies \\(y\\) by \\(e^2\\).", true],
      ["\\(y=2x+3\\).", false],
    ],
    "Exponentiating both sides gives \\(y=e^{2x+\ln3}=e^{2x}e^{\ln3}=3e^{2x}\\). This is an exponential relationship, not a linear one, and adding 1 to \\(x\\) multiplies the result by \\(e^2\\).",
  ),
  makeQuestion(
    57,
    "hard",
    "A decreasing quantity is \\(A(t)=100e^{-kt}\\), and \\(A(5)=50\\). Which statements are correct?",
    [
      ["\\(e^{-5k}=0.5\\).", true],
      ["\\(k=\\frac{\\ln2}{5}\\).", true],
      ["\\(A(10)=25\\).", true],
      ["\\(k=5\\ln2\\).", false],
    ],
    "The condition \\(A(5)=50\\) gives \\(100e^{-5k}=50\\), so \\(e^{-5k}=1/2\\). Taking logs gives \\(-5k=-\\ln2\\), hence \\(k=\\ln2/5\\), and after two five-unit half-steps the quantity is 25.",
  ),
  makeQuestion(
    58,
    "hard",
    "For \\(C>0\\) and \\(s>0\\), define \\(g(x)=C e^{-\\frac{(x-m)^2}{2s^2}}\\). Which statements are correct?",
    [
      ["The maximum occurs at \\(x=m\\).", true],
      ["The function is symmetric around \\(m\\).", true],
      ["Larger \\(|x-m|\\) makes the exponent more negative.", true],
      ["The minimum over all real \\(x\\) occurs at \\(x=m\\).", false],
    ],
    "The squared term \\((x-m)^2\\) is smallest at \\(x=m\\), making the exponent 0 and the value \\(C\\). Moving equally far left or right from \\(m\\) gives the same squared distance, while moving farther away lowers the exponential value toward 0.",
  ),
  makeQuestion(
    59,
    "hard",
    "Let \\(P=0.8\\cdot0.5\\cdot0.25\\). Which product-and-log statements are correct?",
    [
      ["\\(P=0.1\\).", true],
      ["\\(\\ln P=\\ln(0.8)+\\ln(0.5)+\\ln(0.25)\\).", true],
      ["\\(-\\ln P=-\\ln(0.8)-\\ln(0.5)-\\ln(0.25)\\).", true],
      [
        "The average negative log over the three factors is \\(-\\frac{1}{3}\\ln(0.1)\\).",
        true,
      ],
    ],
    "Products become sums after taking logarithms, which is why the log of \\(P\\) splits into three log terms. Since \\(P=0.8\\cdot0.5\\cdot0.25=0.1\\), the average negative log is the total negative log divided by 3.",
  ),
  makeQuestion(
    60,
    "hard",
    "A distribution over centers \\((-1,0,2)\\) has weights \\((0.2,0.3,0.5)\\). Which statements are correct?",
    [
      ["The weighted mean is 0.8.", true],
      ["\\(\\mathbb{E}[X^2]=2.2\\).", true],
      ["The variance is \\(2.2-0.8^2=1.56\\).", true],
      [
        "The mean is \\(\\frac{-1+0+2}{3}=\\frac{1}{3}\\) because all centers are equally weighted.",
        false,
      ],
    ],
    "The weighted mean is \\(0.2(-1)+0.3(0)+0.5(2)=0.8\\). The second moment is \\(0.2(1)+0.3(0)+0.5(4)=2.2\\), so the variance is \\(\\mathbb{E}[X^2]-(\\mathbb{E}[X])^2=1.56\\).",
  ),
];
