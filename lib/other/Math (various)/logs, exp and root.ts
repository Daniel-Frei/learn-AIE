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
    "A \\(\\$2000\\) deposit earns \\(6\\%\\) interest per year for 3 years. Which statements are correct?",
    [
      [
        "Annual compounding gives \\(2000(1.06)^3\\), about \\(\\$2382\\).",
        true,
      ],
      [
        "Continuous compounding gives \\(2000e^{0.18}\\), about \\(\\$2394\\).",
        true,
      ],
      [
        "Annual compounding is exactly \\(2000(1+3\\cdot0.06)=\\$2360\\).",
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
    "A solution changes from \\([H^+]=10^{-7}\\) to \\([H^+]=10^{-5}\\). Which pH statements are correct?",
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
    "Sound intensity level changes by \\(10\\log_{10}(I_2/I_1)\\) decibels. Which statements are correct?",
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
    "A \\(\\$1000\\) investment earns \\(12\\%\\) for 2 years. Which comparisons are correct?",
    [
      ["Annual compounding gives \\(1000(1.12)^2=\\$1254.40\\).", true],
      [
        "Continuous compounding gives \\(1000e^{0.24}\\), about \\(\\$1271\\).",
        true,
      ],
      ["The continuous result is larger by about \\(\\$17\\).", true],
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
];
