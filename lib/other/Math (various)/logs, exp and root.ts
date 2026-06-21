import { Question } from "../../quiz";

type MathDifficulty = "easy" | "medium" | "hard";
type OptionSeed = readonly [text: string, isCorrect: boolean];

function makeQuestion(
  id: string,
  difficulty: MathDifficulty,
  prompt: string,
  optionSeeds: readonly OptionSeed[],
  explanation: string,
): Question {
  if (optionSeeds.length !== 4) {
    throw new Error(`Math logs/exp/roots question ${id} needs 4 options.`);
  }

  return {
    id,
    chapter: 1,
    difficulty,
    prompt,
    options: optionSeeds.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

export const MathLogsExpRootsQuestions: Question[] = [
  makeQuestion(
    "math-logs-exp-roots-q01",
    "easy",
    "Which inverse relationships between logarithms and exponentials are correct?",
    [
      ["If \\(y=2^x\\), then \\(x=\\log_2 y\\) for \\(y>0\\).", true],
      ["\\(\\log_{10}(1000)=3\\).", true],
      ["\\(\\ln(e^4)=4\\).", true],
      ["\\(e^{\\ln 7}=7\\).", true],
    ],
    "A logarithm asks which exponent is needed to produce a positive number from a valid base. The listed statements all use that inverse relationship correctly: powers of 10, powers of \\(e\\), and a general base-2 exponential are being undone by the matching logarithm.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q02",
    "easy",
    "For positive \\(x\\) and \\(y\\), and base \\(b>0\\), \\(b\\ne1\\), which logarithm rules are correct?",
    [
      ["\\(\\log_b(xy)=\\log_b x+\\log_b y\\).", true],
      ["\\(\\log_b(x/y)=\\log_b x-\\log_b y\\).", true],
      ["\\(\\log_b(x+y)=\\log_b x+\\log_b y\\).", false],
      ["\\(\\log_b(x^2)=\\log_b x+2\\).", false],
    ],
    "Logarithms turn multiplication into addition and division into subtraction, so the product and quotient rules apply to \\(xy\\) and \\(x/y\\). They do not distribute over ordinary addition, and a power moves down as a multiplier: \\(\\log_b(x^2)=2\\log_b x\\), not \\(\\log_b x+2\\).",
  ),
  makeQuestion(
    "math-logs-exp-roots-q03",
    "easy",
    "Which root or fractional-exponent calculations are correct?",
    [
      ["\\(\\sqrt{49}=7\\).", true],
      ["\\(27^{1/3}=3\\).", true],
      [
        "\\(\\sqrt{a+b}=\\sqrt a+\\sqrt b\\) for all nonnegative \\(a,b\\).",
        false,
      ],
      ["\\((1/2)^{-2}=-4\\).", false],
    ],
    "The principal square root of 49 is 7, and the cube root of 27 is 3 because \\(3^3=27\\). Square roots do not distribute over sums, and a negative exponent means take a reciprocal before applying the power, so \\((1/2)^{-2}=2^2=4\\), not \\(-4\\).",
  ),
  makeQuestion(
    "math-logs-exp-roots-q04",
    "easy",
    "Which expression has a real logarithm value under the usual real-number rules?",
    [
      ["\\(\\log_2(-8)\\).", false],
      ["\\(\\ln(0)\\).", false],
      ["\\(\\log_{10}(0.2)\\).", true],
      ["\\(\\log_1(5)\\).", false],
    ],
    "A real logarithm needs a positive input and a positive base other than 1. The input 0.2 is positive and base 10 is valid, while negative inputs, zero inputs, and base 1 do not produce a real logarithm in the usual definition.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q05",
    "easy",
    "A quantity starts at \\(A\\) and grows continuously at rate \\(r\\) for \\(t\\) years. Which statements are correct?",
    [
      ["The model is \\(A e^{rt}\\).", true],
      [
        "The number \\(e\\) is about 2.718 and is the base of natural logarithms.",
        true,
      ],
      ["If \\(r=0\\), the model stays at \\(A\\).", true],
      ["Continuous growth is modeled by \\(A e^{r+t}\\).", false],
    ],
    "Continuous exponential growth multiplies the starting amount by \\(e^{rt}\\), so the rate and time appear as a product in the exponent. Setting \\(r=0\\) gives \\(e^0=1\\), while adding \\(r+t\\) in the exponent would give the wrong units and the wrong behavior at time zero.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q06",
    "easy",
    "The pH scale is \\(\\text{pH}=-\\log_{10}[H^+]\\), where \\([H^+]\\) is hydrogen ion concentration in moles per liter. Which statements are correct?",
    [
      ["If \\([H^+]=10^{-3}\\), then \\(\\text{pH}=3\\).", true],
      ["Lowering pH by 1 means \\([H^+]\\) is multiplied by 10.", true],
      ["Changing from pH 5 to pH 3 doubles \\([H^+]\\).", false],
      ["The pH formula uses the square root of \\([H^+]\\).", false],
    ],
    "The minus sign makes \\([H^+]=10^{-3}\\) correspond to pH 3. Because the scale is base-10 logarithmic, a one-unit pH drop means a tenfold concentration increase, so a drop from 5 to 3 means a hundredfold increase rather than a doubling.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q07",
    "easy",
    "Which root-based scaling statements are correct?",
    [
      ["A square with area 144 has side length \\(\\sqrt{144}=12\\).", true],
      ["A cube with volume 27 has side length \\(\\sqrt[3]{27}=3\\).", true],
      [
        "Doubling a square's area multiplies its side length by \\(\\sqrt2\\).",
        true,
      ],
      [
        "A square with area 144 has side length 72 because area is divided by 2.",
        false,
      ],
    ],
    "Square side lengths come from square roots of area, and cube side lengths come from cube roots of volume. Doubling area does not double side length; it multiplies side length by \\(\\sqrt2\\), because the side length is squared to produce area.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q08",
    "easy",
    "Solve \\(e^{0.4t}=5\\) for \\(t\\). Which expression is correct?",
    [
      ["\\(t=\\frac{\\ln 5}{0.4}\\).", true],
      ["\\(t=0.4\\ln 5\\).", false],
      ["\\(t=\\frac{\\log_{10}5}{0.4}\\) without changing the base.", false],
      ["\\(t=\\frac{e^5}{0.4}\\).", false],
    ],
    "Taking natural logs of both sides gives \\(0.4t=\\ln 5\\), so dividing by 0.4 isolates \\(t\\). A common logarithm could be used only with the matching base-change relationship, and exponentiating 5 does not undo the exponent on the left.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q09",
    "easy",
    "A machine keeps \\(85\\%\\) of its value each year. Which statements are correct?",
    [
      [
        "After \\(t\\) years, a suitable model is \\(V(t)=V_0(0.85)^t\\).",
        true,
      ],
      ["After 2 years, the retained fraction is \\(0.85^2=0.7225\\).", true],
      [
        "After 2 years, the retained fraction is exactly \\(1-2(0.15)=0.70\\).",
        false,
      ],
      [
        "The continuous form with the same yearly factor is \\(V_0 e^{0.85t}\\).",
        false,
      ],
    ],
    "A fixed percentage loss compounds, so each year multiplies the previous value by 0.85 rather than subtracting the original 15 percentage points again. The equivalent continuous exponent would use \\(\\ln(0.85)t\\), not \\(0.85t\\), because \\(e^{\\ln(0.85)}=0.85\\).",
  ),
  makeQuestion(
    "math-logs-exp-roots-q10",
    "easy",
    "Which basic logarithm identities are correct for a valid base \\(b\\)?",
    [
      ["\\(\\log_{10}(1)=0\\).", true],
      ["\\(\\ln(1)=0\\).", true],
      ["\\(\\log_b b=1\\).", true],
      ["\\(\\log_b(1/x)=-\\log_b x\\) for \\(x>0\\).", true],
    ],
    "Every valid base raised to the zero power gives 1, which explains both \\(\\log_{10}(1)=0\\) and \\(\\ln(1)=0\\). A base raised to the first power gives itself, and reciprocals correspond to changing the sign of the exponent.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q11",
    "medium",
    "A $2000 deposit earns \\(6\\%\\) interest per year for 3 years. Which statements are correct?",
    [
      ["Annual compounding gives \\(2000(1.06)^3\\), about $2382.", true],
      ["Continuous compounding gives \\(2000e^{0.18}\\), about $2394.", true],
      [
        "Annual compounding is exactly \\(2000(1+3\\cdot0.06)\\), or $2360.",
        false,
      ],
      ["Continuous compounding must be smaller because \\(e<3\\).", false],
    ],
    "Annual compounding multiplies by 1.06 once per year, while continuous compounding uses \\(e^{rt}\\) with \\(rt=0.18\\). The simple-interest expression \\(2000(1+0.18)\\) does not compound prior interest, and the comparison is about the exponential factors, not the isolated fact that \\(e\\) is less than 3.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q12",
    "medium",
    "A medicine dose starts at 80 mg and has a half-life of 6 hours. How much remains after 18 hours?",
    [
      ["10 mg.", true],
      ["20 mg.", false],
      ["40 mg.", false],
      ["\\(80e^{-18/6}\\) mg, about 4 mg.", false],
    ],
    "Eighteen hours is three half-lives, so the dose is multiplied by \\((1/2)^3=1/8\\). That leaves \\(80/8=10\\) mg; the expression \\(80e^{-18/6}\\) uses 3 as the decay constant instead of converting the half-life through \\(\\ln 2\\).",
  ),
  makeQuestion(
    "math-logs-exp-roots-q13",
    "medium",
    "A population grows continuously at \\(8\\%\\) per year. Which doubling-time statements are correct?",
    [
      ["The doubling time solves \\(e^{0.08t}=2\\).", true],
      ["\\(t=\\frac{\\ln 2}{0.08}\\), about 8.66 years.", true],
      ["The rate must be written as 0.08 rather than 8 in the exponent.", true],
      [
        "\\(t=\\frac{2}{0.08}=25\\) years because the target multiplier is 2.",
        false,
      ],
    ],
    "A continuous percentage rate belongs in \\(e^{rt}\\) as a decimal, so doubling means solving \\(e^{0.08t}=2\\). Taking natural logs gives \\(t=\\ln 2/0.08\\); using 2 directly in the numerator ignores the logarithm needed to undo the exponential.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q14",
    "medium",
    "Using \\(\\text{pH}=-\\log_{10}[H^+]\\), a solution changes from \\([H^+]=10^{-7}\\) to \\([H^+]=10^{-5}\\). Which pH statements are correct?",
    [
      ["The pH changes from 7 to 5.", true],
      ["The hydrogen ion concentration becomes 100 times larger.", true],
      ["The pH decreases by 2 units.", true],
      [
        "The pH increases because \\(10^{-5}\\) has the larger exponent.",
        false,
      ],
    ],
    "The concentration \\(10^{-5}\\) is larger than \\(10^{-7}\\), but the pH formula has a negative base-10 logarithm. That makes the pH drop from 7 to 5, and the two-exponent difference corresponds to a factor of \\(10^2=100\\) in concentration.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q15",
    "medium",
    "Sound intensity level changes by \\(10\\log_{10}(I_2/I_1)\\) decibels, where \\(I_1\\) is the starting intensity and \\(I_2\\) is the final intensity. Which statements are correct?",
    [
      ["A 10 dB increase means the intensity is multiplied by 10.", true],
      ["A 20 dB increase means the intensity is multiplied by 100.", true],
      [
        "A 5 dB increase means the intensity is multiplied by exactly 5.",
        false,
      ],
      ["A 0 dB change means the final intensity is 0.", false],
    ],
    "Decibels use a base-10 logarithm with a factor of 10, so 10 dB corresponds to \\(\\log_{10}(I_2/I_1)=1\\) and therefore a tenfold ratio. A 0 dB change means the ratio is 1, and a 5 dB change means \\(10^{0.5}\\), not a factor of 5.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q16",
    "medium",
    "On a base-10 earthquake magnitude scale, each 1.0 increase means 10 times the wave amplitude. Which statements are correct?",
    [
      ["A magnitude difference of 1.0 gives an amplitude ratio of 10.", true],
      ["A magnitude difference of 2.0 gives an amplitude ratio of 100.", true],
      [
        "A magnitude difference of 1.5 gives a ratio \\(10^{1.5}\\), about 31.6.",
        true,
      ],
      ["A magnitude difference of 0.5 gives a ratio of exactly 5.", false],
    ],
    "A magnitude difference \\(d\\) corresponds to an amplitude ratio \\(10^d\\). Therefore a 2-unit change squares the tenfold factor, a 1.5-unit change is between 10 and 100, and a half-unit change is \\(\\sqrt{10}\\), not exactly 5.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q17",
    "medium",
    "The root mean square (RMS) of readings 3 and 4 is \\(\\sqrt{(3^2+4^2)/2}\\). Which statements are correct?",
    [
      ["The RMS is \\(\\sqrt{12.5}\\), about 3.54.", true],
      ["The calculation squares the readings before averaging them.", true],
      ["The final square root returns the answer to the original units.", true],
      [
        "For these positive unequal readings, the RMS is slightly larger than the arithmetic mean.",
        true,
      ],
    ],
    "RMS first averages squared values, then takes a square root, so the result is \\(\\sqrt{12.5}\\). Squaring emphasizes larger readings, which is why the RMS is above the arithmetic mean of 3.5 for these unequal positive readings, while the final square root restores the original measurement units.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q18",
    "medium",
    "A value is multiplied by 1.2 in the first year and by 0.8 in the second year. What is the average yearly multiplicative factor?",
    [
      ["\\(\\sqrt{1.2\\cdot0.8}\\), about 0.98.", true],
      [
        "\\((1.2+0.8)/2=1.0\\), because ordinary averaging preserves compounding.",
        false,
      ],
      [
        "\\(1.2\\cdot0.8=0.96\\), because the two-year factor is the yearly average.",
        false,
      ],
      ["\\(\\sqrt{1.2+0.8}\\), about 1.41.", false],
    ],
    "For multiplicative changes, the average per-period factor is the geometric mean, so two periods use a square root of the product. The product 0.96 is the combined two-year factor, while the arithmetic mean does not preserve the same compounded endpoint.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q19",
    "medium",
    "Solve \\(\\log_2(x-1)+\\log_2(x+1)=3\\). Which statements are correct?",
    [
      ["The domain requires \\(x>1\\).", true],
      ["Combining logs gives \\(\\log_2(x^2-1)=3\\).", true],
      ["\\(x=-3\\) is also a solution because \\((-3)^2-1=8\\).", false],
      ["The product rule gives \\(\\log_2(x^2)=3\\).", false],
    ],
    "The two logarithm inputs must both be positive, so \\(x-1>0\\) and \\(x+1>0\\), which together require \\(x>1\\). The product rule gives \\((x-1)(x+1)=x^2-1\\), so \\(x^2-1=8\\) leads to candidates \\(\\pm3\\), but only \\(x=3\\) is in the domain.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q20",
    "medium",
    "A measurement follows \\(y=C e^{kt}\\) with \\(C>0\\). Which statements about linearizing the data are correct?",
    [
      ["Taking natural logs gives \\(\\ln y=\\ln C+kt\\).", true],
      [
        "A plot of \\(\\ln y\\) against \\(t\\) should be a line with slope \\(k\\).",
        true,
      ],
      ["The intercept of that line is \\(\\ln C\\).", true],
      [
        "A plot of \\(y\\) against \\(t\\) must be a straight line for any nonzero \\(k\\).",
        false,
      ],
    ],
    "Natural logs undo the exponential and turn the multiplicative constant into an additive intercept. The transformed relationship is linear in \\(t\\), while the original \\(y\\) values curve upward for positive \\(k\\) or downward for negative \\(k\\).",
  ),
  makeQuestion(
    "math-logs-exp-roots-q21",
    "medium",
    "A decay model is \\(y(t)=50e^{-0.2t}\\). Which statements are correct?",
    [
      ["The initial value is 50.", true],
      ["The decay constant is 0.2 per unit time.", true],
      ["At \\(t=5\\), \\(y=50/e\\), about 18.4.", true],
      ["The half-life is \\(\\ln 2/0.2\\), about 3.47.", true],
    ],
    "At time zero, the exponential factor is \\(e^0=1\\), so the starting value is 50. The negative exponent causes decay; substituting \\(t=5\\) gives \\(e^{-1}\\), and setting the exponential factor equal to \\(1/2\\) gives the half-life formula.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q22",
    "medium",
    "Which statements correctly convert or estimate \\(\\log_2(1000)\\)?",
    [
      ["\\(\\log_2(1000)=\\frac{\\ln 1000}{\\ln 2}\\).", true],
      ["The value is about 10 because \\(2^{10}=1024\\).", true],
      ["\\(\\log_2(1000)=\\ln 1000\\cdot\\ln 2\\).", false],
      ["The value is exactly 3 because \\(\\log_{10}(1000)=3\\).", false],
    ],
    "The change-of-base formula divides one logarithm by the logarithm of the target base. Since 1000 is close to 1024, and 1024 is \\(2^{10}\\), the base-2 logarithm is close to 10 rather than equal to the base-10 logarithm.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q23",
    "medium",
    "A cube has volume \\(500\\text{ cm}^3\\). Which expression gives its side length in centimeters?",
    [
      ["\\(\\sqrt[3]{500}\\), about 7.94.", true],
      ["\\(\\sqrt{500}\\), about 22.36.", false],
      ["\\(500/3\\), about 166.67.", false],
      ["10, because \\(10^3=100\\).", false],
    ],
    "A cube's volume is side length cubed, so recovering side length requires a cube root. A square root would be appropriate for square area, division by 3 does not undo cubing, and a side length of 10 would give volume 1000 rather than 500.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q24",
    "medium",
    "For \\(x>0\\), which fractional-exponent statements are correct?",
    [
      ["\\(x^{3/2}=(\\sqrt{x})^3\\).", true],
      ["\\(x^{-1/2}=1/\\sqrt{x}\\).", true],
      ["If \\(x=9\\), then \\(x^{3/2}=27\\).", true],
      ["\\(x^{1/2}+x^{1/2}=x\\) for all positive \\(x\\).", false],
    ],
    "The denominator of a fractional exponent gives a root, and a negative exponent means reciprocal. For \\(x=9\\), \\(9^{3/2}=(\\sqrt9)^3=3^3=27\\), while adding two square roots gives \\(2\\sqrt{x}\\), which equals \\(x\\) only for special values.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q25",
    "hard",
    "An exponential model has \\(P(0)=100\\) and \\(P(4)=200\\), with \\(P(t)=100e^{kt}\\). Which statements are correct?",
    [
      ["The growth constant is \\(k=\\frac{\\ln 2}{4}\\).", true],
      ["The model predicts \\(P(8)=400\\).", true],
      ["The growth constant is \\(k=2/4=0.5\\).", false],
      [
        "The model predicts \\(P(8)=300\\) because the first 4 years added 100.",
        false,
      ],
    ],
    "The value doubles over 4 time units, so \\(e^{4k}=2\\) and \\(k=\\ln 2/4\\). Exponential growth repeats multipliers over equal time intervals, so another 4 time units doubles 200 to 400 rather than adding the same absolute amount.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q26",
    "hard",
    "A cooling object follows \\(T(t)=20+60e^{-0.1t}\\). When does it reach \\(40\\)?",
    [
      ["\\(t=10\\ln 3\\), about 11.0.", true],
      ["\\(t=\\ln 3/10\\), about 0.11.", false],
      ["\\(t=10\\ln(1/3)\\), which is negative.", false],
      ["\\(t=20/0.1=200\\), because the temperature must fall by 20.", false],
    ],
    "Set \\(40=20+60e^{-0.1t}\\), so \\(e^{-0.1t}=1/3\\). Taking natural logs gives \\(-0.1t=\\ln(1/3)=-\\ln3\\), hence \\(t=10\\ln3\\); subtracting temperatures alone does not account for exponential decay.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q27",
    "hard",
    "A $1000 investment earns \\(12\\%\\) for 2 years. Which comparisons are correct?",
    [
      ["Annual compounding gives \\(1000(1.12)^2\\), or $1254.40.", true],
      ["Continuous compounding gives \\(1000e^{0.24}\\), about $1271.", true],
      ["The continuous result is larger by about $17.", true],
      [
        "Both methods give exactly the same result because the quoted rate is 12%.",
        false,
      ],
    ],
    "Annual compounding applies the yearly factor twice, while continuous compounding spreads the same nominal rate continuously through \\(e^{rt}\\). The two methods use different compounding assumptions, so the continuous result is slightly larger for the same positive nominal rate and time.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q28",
    "hard",
    "Which statements about \\(\\left(1+\\frac{1}{n}\\right)^n\\) are correct for positive integer \\(n\\)?",
    [
      ["As \\(n\\) grows, the expression approaches \\(e\\).", true],
      ["For \\(n=100\\), the value is about 2.70.", true],
      ["For finite positive \\(n\\), the value is below \\(e\\).", true],
      [
        "Using \\(n=1000\\) gives a closer approximation to \\(e\\) than using \\(n=10\\).",
        true,
      ],
    ],
    "The expression is a classic limit definition of Euler's number. Increasing \\(n\\) makes the compound-growth approximation finer, so values such as \\(n=100\\) are near 2.70 and larger \\(n\\) values move closer to \\(e\\) from below.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q29",
    "hard",
    "A power-law model is \\(y=3x^{1.5}\\) for \\(x>0\\). Which statements are correct?",
    [
      [
        "A 1% increase in \\(x\\) corresponds to about a 1.5% increase in \\(y\\) for small changes.",
        true,
      ],
      [
        "Doubling \\(x\\) multiplies \\(y\\) by \\(2^{1.5}\\), about 2.83.",
        true,
      ],
      ["Taking logs gives \\(\\ln y=\\ln3+1.5\\ln x\\).", true],
      ["In a log-log plot, the slope is 1.5.", true],
    ],
    "The exponent in a power law becomes the slope after taking logs, which also gives the small-percent-change elasticity. Doubling \\(x\\) is a multiplicative change, so the output multiplier is \\(2^{1.5}\\), not an additive change of 1.5 units.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q30",
    "hard",
    "A log-log plot of data is well described by \\(\\ln y=\\ln C+0.75\\ln x\\). Which model matches that line?",
    [
      ["\\(y=Cx^{0.75}\\).", true],
      ["\\(y=Ce^{0.75x}\\).", false],
      ["\\(y=0.75Cx\\).", false],
      ["\\(y=C\\ln(0.75x)\\).", false],
    ],
    "Exponentiating both sides gives \\(y=e^{\\ln C}e^{0.75\\ln x}=Cx^{0.75}\\). The slope on a log-log plot is the power-law exponent, not the rate constant of an exponential-in-\\(x\\) model or a simple linear multiplier.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q31",
    "hard",
    "A standard error scales like \\(\\sigma/\\sqrt n\\). Which statements are correct?",
    [
      ["To cut the standard error in half, multiply \\(n\\) by 4.", true],
      ["Increasing \\(n\\) from 100 to 400 halves the standard error.", true],
      [
        "The square root creates diminishing returns as sample size grows.",
        true,
      ],
      ["Doubling \\(n\\) halves the standard error exactly.", false],
    ],
    "Because \\(n\\) appears under a square root in the denominator, reducing the error by a factor of 2 requires increasing \\(\\sqrt n\\) by a factor of 2, which means increasing \\(n\\) by a factor of 4. Doubling \\(n\\) only divides the standard error by \\(\\sqrt2\\).",
  ),
  makeQuestion(
    "math-logs-exp-roots-q32",
    "hard",
    "For \\(f(x)=\\sqrt{x-3}+\\ln(10-x)\\), which domain statements are correct?",
    [
      ["The domain is \\([3,10)\\).", true],
      ["\\(x=3\\) is allowed.", true],
      ["\\(x=10\\) is allowed because the square-root term is defined.", false],
      ["Values below 3 are allowed as long as \\(10-x\\) is positive.", false],
    ],
    "The square root requires \\(x-3\\ge0\\), so \\(x\\ge3\\), while the logarithm requires \\(10-x>0\\), so \\(x<10\\). Both conditions must hold at the same time, which includes 3 but excludes 10 and every value below 3.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q33",
    "hard",
    "Solve \\(3e^{2t}=12\\). Which expression gives \\(t\\)?",
    [
      ["\\(t=\\frac{\\ln4}{2}=\\ln2\\).", true],
      ["\\(t=\\ln4\\).", false],
      ["\\(t=2\\ln4\\).", false],
      ["\\(t=\\frac{\\log_{10}4}{2}\\) without any base conversion.", false],
    ],
    "First divide by 3 to get \\(e^{2t}=4\\), then take natural logs to get \\(2t=\\ln4\\). Dividing by 2 gives \\(t=\\ln4/2\\), which equals \\(\\ln2\\); using a different log base requires a conversion factor.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q34",
    "hard",
    "Which statements about \\(\\ln(e^3+e^5)\\) are correct?",
    [
      ["Factoring out \\(e^5\\) gives \\(5+\\ln(1+e^{-2})\\).", true],
      ["The value is larger than 5 but smaller than 6.", true],
      [
        "It is not equal to 8 because logarithms do not distribute over addition.",
        true,
      ],
      [
        "Using the largest exponent as an anchor preserves the value and improves numerical stability.",
        true,
      ],
    ],
    "The sum can be written as \\(e^5(1+e^{-2})\\), so the logarithm becomes \\(5+\\ln(1+e^{-2})\\). Since \\(1+e^{-2}\\) is between 1 and 2, the extra logarithm is between 0 and \\(\\ln2\\), and the rewrite avoids unnecessarily huge exponentials.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q35",
    "hard",
    "A radioactive isotope has half-life 5730 years. Which decay statements are correct?",
    [
      [
        "The remaining fraction after \\(t\\) years is \\((1/2)^{t/5730}\\).",
        true,
      ],
      ["The same model can be written \\(e^{-t\\ln2/5730}\\).", true],
      ["After two half-lives, the remaining fraction is \\(1/4\\).", true],
      [
        "If the remaining fraction is \\(f\\), then \\(t=-5730\\ln f/\\ln2\\).",
        true,
      ],
    ],
    "Each half-life multiplies the remaining amount by \\(1/2\\), so the exponent counts how many half-lives have elapsed. Rewriting the base with \\(e\\) uses \\((1/2)^a=e^{-a\\ln2}\\), and solving for time requires taking logs and reversing the negative sign.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q36",
    "medium",
    "The upper half of a circle of radius 5 centered at the origin can be written \\(y=\\sqrt{25-x^2}\\). Which statements are correct?",
    [
      ["The expression is defined for \\(-5\\le x\\le5\\).", true],
      ["At \\(x=3\\), the upper-half value is \\(y=4\\).", true],
      ["The square root selects the nonnegative \\(y\\) values.", true],
      ["The lower half would use \\(y=-\\sqrt{25-x^2}\\).", true],
    ],
    "The quantity under the square root must be nonnegative, giving \\(x^2\\le25\\) and therefore \\(-5\\le x\\le5\\). At \\(x=3\\), the radicand is 16, and the principal square root gives the upper semicircle; the negative square root gives the matching lower semicircle.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q37",
    "easy",
    "Which statements about principal square roots and cube roots are correct?",
    [
      ["\\(\\sqrt9=3\\) under the principal square-root convention.", true],
      [
        "The equation \\(x^2=9\\) has solutions \\(x=3\\) and \\(x=-3\\).",
        true,
      ],
      ["\\(\\sqrt[3]{-8}=-2\\).", true],
      ["\\(\\sqrt9=\\pm3\\) as a value of the square-root symbol.", false],
    ],
    "The square-root symbol denotes the principal nonnegative root, even though the equation \\(x^2=9\\) has two solutions. Cube roots preserve sign for real numbers, so the real cube root of \\(-8\\) is \\(-2\\).",
  ),
  makeQuestion(
    "math-logs-exp-roots-q38",
    "medium",
    "Solve \\(\\sqrt{x+5}=x-1\\). Which answer is correct?",
    [
      ["\\(x=4\\).", true],
      ["\\(x=-1\\).", false],
      ["Both \\(x=4\\) and \\(x=-1\\).", false],
      ["There is no real solution.", false],
    ],
    "The right side must be nonnegative, so any solution must satisfy \\(x\\ge1\\). Squaring gives \\(x+5=(x-1)^2\\), whose algebraic candidates are 4 and -1, but only 4 satisfies the original equation and the domain condition.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q39",
    "hard",
    "A runtime estimate is \\(c\\log_2 n\\). If the formula is rewritten using \\(\\ln n\\), which coefficient preserves the same numerical predictions?",
    [
      ["\\(c/\\ln2\\), because \\(\\log_2 n=\\ln n/\\ln2\\).", true],
      ["\\(c\\ln2\\), because natural logs are larger for \\(n>1\\).", false],
      [
        "\\(c\\), because changing log base never changes numerical values.",
        false,
      ],
      [
        "\\(2c\\), because base 2 logarithms always differ from natural logs by a factor of 2.",
        false,
      ],
    ],
    "Changing the base of a logarithm multiplies values by a constant factor, and the exact factor from natural log to base 2 is \\(1/\\ln2\\). The asymptotic class may stay logarithmic, but the numerical coefficient must change if the estimate is meant to preserve actual predicted counts.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q40",
    "easy",
    "A repeated-scaling model is \\(R(t)=ab^t\\) with \\(a>0\\) and \\(b>0\\). Which statements are correct?",
    [
      ["\\(R(0)=a\\).", true],
      ["\\(R(t+1)/R(t)=b\\).", true],
      ["If \\(b=1\\), the model stays constant.", true],
      ["If \\(0<b<1\\), the model decays as \\(t\\) increases.", true],
    ],
    "The starting value is found by using \\(b^0=1\\), and each one-step increase in \\(t\\) multiplies the output by another factor of \\(b\\). A factor of 1 leaves the value unchanged, while a positive factor below 1 repeatedly shrinks the value.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q41",
    "easy",
    "A dashboard plots \\(\\ln(\\text{cases})\\) on the vertical axis against time in days. For a stretch of days, the points lie close to a straight upward line. Which interpretations are correct?",
    [
      [
        "The original case counts are growing by roughly a constant multiplicative factor each day.",
        true,
      ],
      [
        "Equal vertical gaps on the log plot correspond to equal ratios in the original counts.",
        true,
      ],
      [
        "The raw case counts must be increasing by the same absolute number each day.",
        false,
      ],
      [
        "Using a logarithmic vertical axis makes any data set look like a straight line.",
        false,
      ],
    ],
    "A straight line on a log-versus-time plot is the visual signature of exponential growth, because equal time steps add equal amounts to the log and therefore multiply the original quantity by equal factors. Linear additive growth would not stay straight after taking logs.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q42",
    "easy",
    "On a simplified earthquake scale, each increase of 1 in magnitude corresponds to about 32 times as much released energy. About how much more energy does a magnitude 6 event release than a magnitude 4 event?",
    [
      ["About \\(6-4\\), or 2 times as much.", false],
      ["About \\(32\\), or 32 times as much.", false],
      ["About \\(2\\cdot32\\), or 64 times as much.", false],
      ["About \\(32^2\\), or roughly 1000 times as much.", true],
    ],
    "A two-step increase on this logarithmic scale multiplies by 32 twice. That gives \\(32^2=1024\\), so magnitudes that look only two units apart can represent about a thousandfold difference underneath.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q43",
    "easy",
    "Which statements correctly distinguish roots and logarithms as inverse operations for exponentiation?",
    [
      [
        "\\(\\sqrt[3]{1000}=10\\) solves for the unknown base in \\(x^3=1000\\).",
        true,
      ],
      [
        "\\(\\log_{10}(1000)=3\\) solves for the unknown exponent in \\(10^x=1000\\).",
        true,
      ],
      [
        "Roots and logs both ask for the unknown exponent in the same equation.",
        false,
      ],
      [
        "Exponentiation has different inverse questions depending on whether the base or exponent is unknown.",
        true,
      ],
    ],
    "A root answers a base question such as \\(x^3=1000\\). A logarithm answers an exponent question such as \\(10^x=1000\\). Both undo exponentiation, but they undo different missing pieces.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q44",
    "easy",
    "Which statements about logarithm base conventions are correct?",
    [
      [
        "In many engineering or calculator contexts, \\(\\log x\\) means \\(\\log_{10}x\\).",
        true,
      ],
      ["In calculus, \\(\\ln x\\) means \\(\\log_e x\\).", true],
      [
        "In computer science, base-2 logarithms often appear when counting doublings or bits.",
        true,
      ],
      [
        "Changing the base of a logarithm rescales its values by a constant factor.",
        true,
      ],
    ],
    "Different fields often choose the base that makes their calculations natural: 10 for decimal magnitude, \\(e\\) for calculus, and 2 for binary growth. The change-of-base formula explains why these choices differ by constant scaling factors.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q45",
    "medium",
    "Using base-10 logarithms, which equalities are correct?",
    [
      ["\\(\\log_{10}(10^4\\cdot10^3)=7\\).", true],
      ["\\(\\log_{10}(100^3)=6\\).", true],
      ["\\(\\log_{10}(10^4+10^3)=7\\).", false],
      ["\\(\\log_{10}((10^4)^3)=7\\).", false],
    ],
    "Products inside a logarithm become sums, so \\(10^4\\cdot10^3=10^7\\). Powers pull down as multipliers, so \\(100^3=(10^2)^3=10^6\\). Addition inside a log has no comparable general simplification.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q46",
    "medium",
    "An analyst wants to simplify \\(\\log(a+b)\\) for positive \\(a\\) and \\(b\\). Which statement is generally valid?",
    [
      ["\\(\\log(a+b)=\\log a+\\log b\\).", false],
      ["\\(\\log(a+b)=(\\log a)(\\log b)\\).", false],
      ["\\(\\log(a+b)=1/(\\log a+\\log b)\\).", false],
      ["None of these product-style rewrites is generally valid.", true],
    ],
    "Logarithms have clean rules for products, quotients, and powers, but not for sums inside the input. Trying a simple example such as \\(a=10\\) and \\(b=100\\) quickly shows that the proposed formulas cannot hold in general.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q47",
    "hard",
    "Which statements correctly use the reciprocal relationship between swapped logarithm bases?",
    [
      ["\\(\\log_{10}(1000)=3\\).", true],
      ["\\(\\log_{1000}(10)=1/3\\).", true],
      ["\\(\\log_{10}(1000)\\,\\log_{1000}(10)=1\\).", true],
      ["\\(\\log_{10}(1000)=\\log_{1000}(10)\\).", false],
    ],
    "Since \\(10^3=1000\\), going the other way asks for the exponent \\(x\\) in \\(1000^x=10\\), which is \\(1/3\\). In general, \\(\\log_a b\\) and \\(\\log_b a\\) are reciprocals when both are defined.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q48",
    "hard",
    "Use the approximation \\(2^{10}\\approx1000=10^3\\). Which approximations follow?",
    [
      ["\\(\\log_{10}2\\approx0.3\\).", true],
      ["\\(\\log_2 10\\approx10/3\\).", true],
      ["\\(\\log_2 1000\\approx3\\).", false],
      ["\\(\\log_{10}2\\approx10/3\\).", false],
    ],
    "The relationship \\(2^{10}\\approx10^3\\) says that ten doublings give about three decimal orders of magnitude. Thus \\(\\log_{10}2\\approx3/10\\), while the reciprocal relationship gives \\(\\log_2 10\\approx10/3\\).",
  ),
  makeQuestion(
    "math-logs-exp-roots-q49",
    "hard",
    "What is the value of \\(\\displaystyle \\sum_{b=2}^{100}\\frac{1}{\\log_b(100!)}\\)?",
    [
      ["1.", true],
      ["100.", false],
      ["0.", false],
      ["\\(\\log(100!)\\).", false],
    ],
    "By change of base, \\(1/\\log_b(100!)=\\log b/\\log(100!)\\), using any common log base. Adding from \\(b=2\\) to 100 gives \\((\\log2+\\log3+\\cdots+\\log100)/\\log(100!)\\), and the numerator is \\(\\log(2\\cdot3\\cdots100)=\\log(100!)\\).",
  ),
  makeQuestion(
    "math-logs-exp-roots-q50",
    "medium",
    "An account earns 100% interest over a year, split evenly into \\(n\\) compounding periods. Which statements are correct?",
    [
      [
        "With one compounding period, principal \\(P\\) becomes \\(2P\\).",
        true,
      ],
      [
        "With two periods at 50% each, the account becomes \\(P(1+1/2)^2=2.25P\\).",
        true,
      ],
      [
        "With \\(n\\) equal periods, the account becomes \\(P(1+1/n)^n\\).",
        true,
      ],
      ["As \\(n\\) grows, the multiplier approaches \\(e\\).", true],
    ],
    "More frequent compounding replaces one large multiplication by many smaller multiplications. The limiting multiplier for 100% continuously compounded growth is \\(\\lim_{n\\to\\infty}(1+1/n)^n=e\\).",
  ),
  makeQuestion(
    "math-logs-exp-roots-q51",
    "medium",
    "A population model is \\(P(t)=2^t\\), where \\(t\\) is measured in days. Which statements about the instantaneous growth rate are correct?",
    [
      [
        "The instantaneous rate is proportional to the current population.",
        true,
      ],
      ["The proportionality constant is \\(\\ln2\\) per day.", true],
      [
        "The derivative at day 5 is exactly the average increase from day 5 to day 6.",
        false,
      ],
      [
        "The derivative equals \\(P(t)\\) because any exponential is its own derivative.",
        false,
      ],
    ],
    "For \\(2^t\\), the derivative is \\((\\ln2)2^t\\), so the rate is a constant fraction of the current amount. A one-day average increase is a finite-interval change, while the derivative is the limiting instantaneous rate.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q52",
    "medium",
    "Which derivative is correct?",
    [
      ["\\(\\dfrac{d}{dt}2^t=(\\ln2)2^t\\).", true],
      ["\\(\\dfrac{d}{dt}2^t=2^t\\).", false],
      ["\\(\\dfrac{d}{dt}2^t=t2^{t-1}\\).", false],
      ["\\(\\dfrac{d}{dt}2^t=2t\\).", false],
    ],
    "The base-\\(e\\) exponential is the one whose derivative equals itself. For another positive base, rewrite \\(2^t=e^{t\\ln2}\\), then differentiate to get the extra factor \\(\\ln2\\).",
  ),
  makeQuestion(
    "math-logs-exp-roots-q53",
    "hard",
    "For \\(F(t)=8^t\\), which statements are correct?",
    [
      ["\\(F'(t)=(\\ln8)8^t\\).", true],
      ["\\(\\ln8=3\\ln2\\).", true],
      ["\\(8^t=e^{t\\ln8}\\).", true],
      ["The rate constant is 8 because the base is 8.", false],
    ],
    "Any positive-base exponential can be written with base \\(e\\): \\(8^t=e^{t\\ln8}\\). Since \\(8=2^3\\), the log rule gives \\(\\ln8=3\\ln2\\), and that logarithm is the proportionality constant in the derivative.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q54",
    "easy",
    "Which statements describe why the base \\(e\\) is special for exponentials?",
    [
      ["\\(\\dfrac{d}{dt}e^t=e^t\\).", true],
      [
        "At each point on \\(e^t\\), the tangent slope equals the graph height.",
        true,
      ],
      [
        "The base is chosen so the proportionality constant between value and derivative is 1.",
        true,
      ],
      [
        "Writing models as \\(e^{kt}\\) makes the growth or decay rate constant \\(k\\) visible.",
        true,
      ],
    ],
    "The number \\(e\\) is the exponential base that removes any extra constant from the derivative of \\(e^t\\). In the model \\(e^{kt}\\), the chain rule then leaves the meaningful rate constant \\(k\\).",
  ),
  makeQuestion(
    "math-logs-exp-roots-q55",
    "medium",
    "For \\(G(t)=e^{3t}\\), which statements are correct?",
    [
      ["\\(G'(t)=3e^{3t}\\).", true],
      ["The instantaneous rate is 3 times the current value.", true],
      [
        "\\(G'(t)=e^{3t}\\) because every expression containing \\(e\\) is its own derivative.",
        false,
      ],
      [
        "The factor 3 disappears when differentiating because it is inside the exponent.",
        false,
      ],
    ],
    "The chain rule matters: differentiating the exponent \\(3t\\) contributes a factor of 3. Thus \\(e^{3t}\\) is still proportional to its derivative, but the proportionality constant is 3.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q56",
    "medium",
    "An investment follows \\(A(t)=A_0e^{0.06t}\\), with \\(t\\) in years. Which expression gives the doubling time?",
    [
      ["\\(\\ln2/0.06\\).", true],
      ["\\(2/0.06\\).", false],
      ["\\(0.06/\\ln2\\).", false],
      ["\\(e^{0.06}/2\\).", false],
    ],
    "Set \\(2A_0=A_0e^{0.06t}\\), cancel \\(A_0\\), and take natural logs: \\(\\ln2=0.06t\\). Dividing by 0.06 gives the time; using 2 directly would confuse the target multiplier with the exponent that produces it.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q57",
    "hard",
    "A quantity satisfies the differential equation \\(dQ/dt=kQ\\). Which statements are correct?",
    [
      ["Its solutions have the form \\(Q(t)=Q(0)e^{kt}\\).", true],
      [
        "Positive \\(k\\) gives growth, while negative \\(k\\) gives decay.",
        true,
      ],
      ["The units of \\(k\\) are reciprocal time units.", true],
      [
        "If \\(k=0.05\\), the amount increases by exactly 5 units each time unit.",
        false,
      ],
    ],
    "The equation says the rate is a constant fraction of the current amount, not a constant absolute increase. The solution is exponential, and the rate constant must have units that cancel the time units in the exponent.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q58",
    "medium",
    "An object cools toward room temperature, and the temperature difference \\(D(t)\\) from the room satisfies \\(dD/dt=-kD\\) for \\(k>0\\). Which statements are correct?",
    [
      ["\\(D(t)=D(0)e^{-kt}\\).", true],
      [
        "A larger temperature difference gives a proportionally larger cooling rate in magnitude.",
        true,
      ],
      ["The object's absolute temperature must approach 0.", false],
      [
        "The difference drops by the same number of degrees during every minute.",
        false,
      ],
    ],
    "The model applies to the difference from the surrounding temperature. A negative proportional rate creates exponential decay toward zero difference, not a fixed-degree-per-minute linear drop and not necessarily cooling toward absolute zero.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q59",
    "easy",
    "Which statements about the natural logarithm are correct for positive inputs?",
    [
      [
        "\\(\\ln x\\) is the exponent to which \\(e\\) must be raised to get \\(x\\).",
        true,
      ],
      ["\\(e^{\\ln x}=x\\).", true],
      ["\\(\\ln(e^r)=r\\).", true],
      [
        "The graph of \\(\\ln x\\) is the inverse reflection of the graph of \\(e^x\\).",
        true,
      ],
    ],
    "The natural log is the inverse function of the base-\\(e\\) exponential. That is why applying one after the other returns the original value, as long as the logarithm's positive-input domain is respected.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q60",
    "medium",
    "Which calculus statement about \\(\\ln x\\) is correct?",
    [
      ["\\(\\dfrac{d}{dx}\\ln x=1/x\\) for \\(x>0\\).", true],
      ["\\(\\dfrac{d}{dx}e^x=\\ln x\\).", false],
      ["\\(\\int 1/x\\,dx=x^2/2+C\\).", false],
      [
        "The derivative of \\(\\ln x\\) is constant because logarithms grow by equal steps.",
        false,
      ],
    ],
    "The natural log grows slowly, but not with a constant slope. Its derivative is \\(1/x\\), which also means \\(\\ln x\\) is an antiderivative of \\(1/x\\) on the positive real numbers.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q61",
    "hard",
    "A substance has half-life \\(h\\). Which statements about its remaining fraction after time \\(t\\) are correct?",
    [
      ["The remaining fraction is \\((1/2)^{t/h}\\).", true],
      ["The same fraction can be written \\(e^{-(\\ln2)t/h}\\).", true],
      [
        "If the remaining fraction is \\(f\\), then \\(t=-h\\ln f/\\ln2\\).",
        true,
      ],
      ["The continuous decay constant is \\(h/\\ln2\\).", false],
    ],
    "The exponent \\(t/h\\) counts elapsed half-lives. Rewriting with base \\(e\\) gives a decay constant \\(\\ln2/h\\), and solving \\(f=e^{-(\\ln2)t/h}\\) for time produces the negative log expression.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q62",
    "easy",
    "For ordinary real logarithms, which domain and base statements are correct?",
    [
      ["The input to the logarithm must be positive.", true],
      ["The base must be positive and not equal to 1.", true],
      ["Base 0 is valid because \\(0^x=0\\) for many \\(x\\).", false],
      ["Base 1 is valid because \\(1^x=1\\) is easy to compute.", false],
    ],
    "A real logarithm asks for the exponent that maps a positive base to a positive input. Bases 0 and 1 do not produce a useful one-to-one exponential function, so they cannot support an ordinary real log function.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q63",
    "hard",
    "A runtime estimate is \\(c\\log_2 n\\), but a report must use \\(\\log_{10}n\\). Which coefficient preserves the same numerical values?",
    [
      ["\\(c/\\log_{10}2\\), about \\((10/3)c\\).", true],
      ["\\(c\\log_{10}2\\), about \\(0.3c\\).", false],
      ["\\(c\\), because logarithm bases only change notation.", false],
      ["\\(c/10\\), because \\(2^{10}\\approx1000\\).", false],
    ],
    "Change of base gives \\(\\log_2 n=\\log_{10}n/\\log_{10}2\\). Since \\(\\log_{10}2\\approx0.3\\), the coefficient must be multiplied by about \\(10/3\\), not by 0.3.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q64",
    "medium",
    "Which statements explain why logarithmic scales are useful for measurements with huge ranges?",
    [
      [
        "Additive steps on the scale can represent multiplicative changes in the measured quantity.",
        true,
      ],
      [
        "They compress values that span many orders of magnitude into a manageable numerical range.",
        true,
      ],
      [
        "Scales such as decibels and earthquake magnitude use this idea in different forms.",
        true,
      ],
      [
        "A step that looks small on the scale can represent a large underlying ratio.",
        true,
      ],
    ],
    "When the underlying quantity varies by factors rather than small additions, a log scale turns those factors into manageable additive increments. This is why log scales appear in contexts such as sound intensity, earthquake energy, and fast-growing public data.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q65",
    "hard",
    "A semi-log plot of data is described by \\(\\ln y=\\ln50+0.2t\\). Which statements are correct?",
    [
      ["The original model is \\(y=50e^{0.2t}\\).", true],
      ["The instantaneous rate satisfies \\(dy/dt=0.2y\\).", true],
      [
        "Each one-unit increase in \\(t\\) multiplies \\(y\\) by \\(e^{0.2}\\).",
        true,
      ],
      ["The model doubles exactly every 5 time units.", false],
    ],
    "Exponentiating both sides gives \\(y=50e^{0.2t}\\). The slope on the \\(\\ln y\\) plot is the continuous growth rate, so doubling time is \\(\\ln2/0.2\\), not simply \\(1/0.2=5\\).",
  ),
  makeQuestion(
    "math-logs-exp-roots-q66",
    "easy",
    "Suppose a simplified decibel scale says that every 10-decibel increase multiplies sound intensity by 10. Which statements are correct?",
    [
      ["Going from 70 dB to 80 dB multiplies intensity by 10.", true],
      ["Going from 60 dB to 80 dB multiplies intensity by 100.", true],
      ["Going from 60 dB to 80 dB multiplies intensity by 20.", false],
      ["Every 1-decibel increase multiplies intensity by 10.", false],
    ],
    "On this scale, the multiplicative factor is attached to a 10-decibel step. Two such steps multiply by 10 twice, so a 20-decibel increase corresponds to a factor of 100.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q67",
    "easy",
    "What is \\(\\log_3(81)\\)?",
    [
      ["3.", false],
      ["4.", true],
      ["27.", false],
      ["81.", false],
    ],
    "The logarithm asks for the exponent in \\(3^x=81\\). Because \\(81=3\\cdot3\\cdot3\\cdot3=3^4\\), the missing exponent is 4; the answer is not the base, the input, or an additive count.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q68",
    "medium",
    "For a positive base \\(a\\), which statements about \\(a^x\\) are correct?",
    [
      ["\\(a^x=e^{x\\ln a}\\).", true],
      ["\\(\\dfrac{d}{dx}a^x=(\\ln a)a^x\\).", true],
      [
        "If \\(0<a<1\\), then \\(\\ln a<0\\), so the model decays as \\(x\\) increases.",
        true,
      ],
      [
        "Every exponential \\(a^x\\) has derivative exactly equal to itself.",
        false,
      ],
    ],
    "Rewriting with base \\(e\\) reveals the rate constant \\(\\ln a\\). Only when \\(a=e\\) is that constant exactly 1; bases below 1 have negative logarithms and describe decay.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q69",
    "hard",
    "Which statements connect \\(e\\) to limiting compounding expressions?",
    [
      ["\\(\\lim_{n\\to\\infty}(1+1/n)^n=e\\).", true],
      ["\\(\\lim_{k\\to0}(1+k)^{1/k}=e\\).", true],
      [
        "Splitting 100% growth into more and more equal compounding periods approaches a finite multiplier, not infinity.",
        true,
      ],
      [
        "Those limits explain why \\(e\\) is the natural base for continuous compounding.",
        true,
      ],
    ],
    "The expressions \\((1+1/n)^n\\) and \\((1+k)^{1/k}\\) describe the same limiting idea from different parameterizations. As compounding becomes continuous, the multiplier approaches \\(e\\), giving a finite and reusable growth constant.",
  ),
  makeQuestion(
    "math-logs-exp-roots-q70",
    "hard",
    "A measurement scale is defined by \\(S=\\log_{32}(E/C)\\), where \\(E\\) is energy and \\(C\\) is a fixed reference energy. Which statements are correct?",
    [
      ["Increasing \\(S\\) by 1 multiplies \\(E\\) by 32.", true],
      ["The same relationship can be written \\(E=C\\,32^S\\).", true],
      [
        "The scale can be computed with natural logs as \\(S=\\ln(E/C)/\\ln32\\).",
        true,
      ],
      ["\\(S=0\\) corresponds to \\(E=C\\).", true],
    ],
    "The logarithmic form says how many factors of 32 separate the measured energy from the reference. Exponentiating recovers \\(E=C32^S\\), and change of base lets the same scale be computed with any convenient logarithm.",
  ),
];
