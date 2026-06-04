import { Question } from "../../../quiz";

export const ClinicalTrialsLecture3Questions: Question[] = [
  {
    id: "clinical-trials-l3-q01",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "A trial reports mortality of 10% in the control group and 5% in the treatment group. Which interpretations of the treatment effect are correct?",
    options: [
      {
        text: "The absolute risk reduction is 5 percentage points.",
        isCorrect: true,
      },
      {
        text: "The relative risk reduction is 50%.",
        isCorrect: true,
      },
      {
        text: "The risk ratio is 0.5.",
        isCorrect: true,
      },
      {
        text: "The number needed to treat is 20 for this outcome and time frame.",
        isCorrect: true,
      },
    ],
    explanation:
      "The same result can be described several ways. The absolute risk reduction is 10% minus 5%, the relative risk reduction is 5 divided by 10, the risk ratio is treatment risk divided by control risk, and the Number Needed to Treat (NNT) is 1 divided by 0.05.",
  },
  {
    id: "clinical-trials-l3-q02",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Study A reduces an adverse event from 20% to 10%, while Study B reduces it from 2% to 1%. Which statements correctly compare the results?",
    options: [
      {
        text: "Both studies have a 50% relative risk reduction.",
        isCorrect: true,
      },
      {
        text: "Study A prevents 10 events per 100 treated patients, while Study B prevents 1 per 100.",
        isCorrect: true,
      },
      {
        text: "Both studies have the same number needed to treat.",
        isCorrect: false,
      },
      {
        text: "Study B has the larger absolute risk reduction because its baseline risk is lower.",
        isCorrect: false,
      },
    ],
    explanation:
      "Both studies cut the baseline risk in half, so the relative reduction is identical. Their absolute impact is different: Study A has a 10 percentage point absolute risk reduction and an NNT of 10, while Study B has a 1 percentage point reduction and an NNT of 100.",
  },
  {
    id: "clinical-trials-l3-q03",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "A treatment has a number needed to treat of 500 for preventing a mild symptom over one year. Which interpretation is most appropriate?",
    options: [
      {
        text: "It means 500 patients are treated for one additional person to benefit, so outcome severity, harms, cost, and alternatives matter.",
        isCorrect: true,
      },
      {
        text: "It ranks the treatment above any treatment with a number needed to treat of 20.",
        isCorrect: false,
      },
      {
        text: "It describes the proportional risk reduction between groups, so it is read like relative risk reduction rather than absolute benefit.",
        isCorrect: false,
      },
      {
        text: "It can be interpreted without knowing the outcome being prevented or the follow-up period.",
        isCorrect: false,
      },
    ],
    explanation:
      "NNT translates an absolute effect into a patient-level scale, but it is not a universal ranking. An NNT of 500 for a mild symptom may be unattractive if treatment is costly or harmful, while the same NNT for a very serious outcome could be viewed differently.",
  },
  {
    id: "clinical-trials-l3-q04",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "A blood thinner prevents one stroke for every 50 treated patients and causes one major bleed for every 100 treated patients. Which considerations belong in a benefit-risk interpretation?",
    options: [
      {
        text: "The severity and long-term consequences of strokes versus major bleeds.",
        isCorrect: true,
      },
      {
        text: "The patient's baseline stroke and bleeding risks.",
        isCorrect: true,
      },
      {
        text: "Patient values and available alternative treatments.",
        isCorrect: true,
      },
      {
        text: "The number needed to treat by itself is sufficient to recommend treatment.",
        isCorrect: false,
      },
    ],
    explanation:
      "Benefit and harm should be interpreted together because clinical decisions are not based on efficacy alone. The NNT and Number Needed to Harm (NNH) become meaningful when paired with outcome severity, patient risk, patient preferences, and alternatives.",
  },
  {
    id: "clinical-trials-l3-q05",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "A study reports a risk ratio of 1.2 for recovery, where recovery is the desired outcome. Which statements are correct?",
    options: [
      {
        text: "The treatment group had a higher probability of recovery than the comparison group.",
        isCorrect: true,
      },
      {
        text: "The direction of a risk ratio depends on whether the outcome is beneficial or harmful.",
        isCorrect: true,
      },
      {
        text: "A risk ratio above 1 has the same meaning for recovery as it has for death.",
        isCorrect: false,
      },
      {
        text: "A risk ratio of 1.2 means the absolute recovery rate increased by 20 percentage points.",
        isCorrect: false,
      },
    ],
    explanation:
      "A risk ratio compares probabilities, but the interpretation depends on what outcome is counted. For a beneficial outcome such as recovery, a value above 1 favors treatment; it does not directly state the absolute percentage-point difference.",
  },
  {
    id: "clinical-trials-l3-q06",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "An event occurs in 40% of control patients and 20% of treatment patients. Which statement best distinguishes the risk ratio from the odds ratio?",
    options: [
      {
        text: "The risk ratio is 0.5, while the odds ratio is 0.375, so reading the odds ratio as a risk ratio overstates the effect.",
        isCorrect: true,
      },
      {
        text: "The risk ratio and odds ratio are both 0.5 because both compare treated and control outcomes using the same numerator and denominator.",
        isCorrect: false,
      },
      {
        text: "The odds ratio is closer to 1 than the risk ratio because odds and risks converge as events become common.",
        isCorrect: false,
      },
      {
        text: "The risk ratio is 0.375 because risk is calculated as events divided by non-events.",
        isCorrect: false,
      },
    ],
    explanation:
      "Risk is event probability, so the risk ratio is 20% divided by 40%, or 0.5. Odds are events divided by non-events, so the odds ratio is (0.20/0.80) divided by (0.40/0.60), which is 0.375; confusing the two can make common-outcome effects sound larger.",
  },
  {
    id: "clinical-trials-l3-q07",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly match common treatment-effect measures to the questions they answer?",
    options: [
      {
        text: "Absolute risk reduction asks how many fewer events occur.",
        isCorrect: true,
      },
      {
        text: "Relative risk reduction asks how large the proportional reduction is.",
        isCorrect: true,
      },
      {
        text: "Number needed to treat asks how many patients must be treated for one additional event to be prevented or achieved.",
        isCorrect: true,
      },
      {
        text: "Odds ratio asks how many fewer events occur per 100 treated patients.",
        isCorrect: false,
      },
    ],
    explanation:
      "Different measures answer different interpretive questions. Absolute risk reduction gives the event difference, relative risk reduction gives the proportional change, NNT translates the absolute change into patients treated per additional outcome, and odds ratios compare odds rather than event counts per 100 patients.",
  },
  {
    id: "clinical-trials-l3-q08",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which pieces of information can a confidence interval add to a clinical trial result?",
    options: [
      {
        text: "The plausible range of effect sizes supported by the sample and model.",
        isCorrect: true,
      },
      {
        text: "The precision of the estimate.",
        isCorrect: true,
      },
      {
        text: "Whether the data are compatible with no effect.",
        isCorrect: true,
      },
      {
        text: "Whether clinically meaningful benefit or harm remains compatible with the data.",
        isCorrect: true,
      },
    ],
    explanation:
      "A confidence interval gives more information than a point estimate because it shows uncertainty around the estimate. It helps the reader judge precision, whether no effect is within the interval, and whether the range includes effects large enough to matter clinically.",
  },
  {
    id: "clinical-trials-l3-q09",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "A trial reports a risk ratio of 0.80 with a 95% confidence interval from 0.40 to 1.60. Which interpretations are justified?",
    options: [
      {
        text: "The point estimate suggests benefit, but the interval is highly uncertain.",
        isCorrect: true,
      },
      {
        text: "The data remain compatible with substantial benefit, no difference, or substantial harm.",
        isCorrect: true,
      },
      {
        text: "The estimate precisely establishes a 20% relative risk reduction.",
        isCorrect: false,
      },
      {
        text: "The interval excludes the no-effect value for a risk ratio.",
        isCorrect: false,
      },
    ],
    explanation:
      "The point estimate of 0.80 points toward lower risk with treatment, but the interval is wide. Because the interval crosses 1.0, the data are compatible with benefit, no difference, and harm, so the result should be described as uncertain rather than precise.",
  },
  {
    id: "clinical-trials-l3-q10",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Study A reports a risk ratio of 0.80 with a 95% confidence interval from 0.78 to 0.82. Study B reports a risk ratio of 0.50 with a 95% confidence interval from 0.10 to 2.50. Which comparison is best?",
    options: [
      {
        text: "Study A is less dramatic but much more precise, while Study B has a dramatic point estimate with high uncertainty.",
        isCorrect: true,
      },
      {
        text: "Study B is more reliable because a point estimate of 0.50 is farther from 1 than 0.80 and therefore carries more information.",
        isCorrect: false,
      },
      {
        text: "Study A is inconclusive because narrow confidence intervals indicate small clinical effects.",
        isCorrect: false,
      },
      {
        text: "Study B rules out harm because its point estimate favors treatment.",
        isCorrect: false,
      },
    ],
    explanation:
      "A dramatic point estimate is not the same as precise evidence. Study A gives a tight estimate near a 20% relative reduction, while Study B could reflect a very large benefit, no effect, or harm because its interval is extremely wide.",
  },
  {
    id: "clinical-trials-l3-q11",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly connect confidence-interval width with clinical-trial information?",
    options: [
      {
        text: "A narrow interval usually reflects more information, such as a larger sample size or more events.",
        isCorrect: true,
      },
      {
        text: "A wide interval often reflects limited information, few events, high variability, or poor data quality.",
        isCorrect: true,
      },
      {
        text: "A narrow interval can estimate a clinically trivial effect very precisely.",
        isCorrect: true,
      },
      {
        text: "A wide interval identifies bias as the source of uncertainty rather than random variation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Interval width is mainly about precision, not importance. More data and more events tend to narrow the interval, while limited information widens it; a precise estimate can still describe a clinically tiny effect, and a wide interval does not by itself diagnose the cause of uncertainty.",
  },
  {
    id: "clinical-trials-l3-q12",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "When using a confidence interval to interpret a treatment effect, which practical questions are useful?",
    options: [
      {
        text: "Does the interval include the no-effect value?",
        isCorrect: true,
      },
      {
        text: "Does the interval include clinically meaningful benefit?",
        isCorrect: true,
      },
      {
        text: "Does the interval include clinically meaningful harm?",
        isCorrect: true,
      },
      {
        text: "Is the whole interval within a range that would support the decision being considered?",
        isCorrect: true,
      },
    ],
    explanation:
      "Confidence intervals are useful because they connect statistical uncertainty to clinical decisions. Asking whether the interval includes no effect, meaningful benefit, meaningful harm, and decision-supporting values is more informative than looking at the point estimate alone.",
  },
  {
    id: "clinical-trials-l3-q13",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "A hypothesis test gives p = 0.03 for a treatment comparison. Which statements correctly interpret that result?",
    options: [
      {
        text: "If the null hypothesis were true, data this extreme or more extreme would occur about 3% of the time under the test assumptions.",
        isCorrect: true,
      },
      {
        text: "The p-value is calculated within a model that starts by assuming the null hypothesis.",
        isCorrect: true,
      },
      {
        text: "There is a 97% probability that the treatment works.",
        isCorrect: false,
      },
      {
        text: "There is a 3% probability that random chance produced the observed treatment difference after accounting for the trial design and endpoint.",
        isCorrect: false,
      },
    ],
    explanation:
      "A p-value describes how surprising the observed data would be if the null hypothesis and model assumptions were true. It does not directly give the probability that a treatment works, the probability that the null is true, or the probability that the result happened by chance.",
  },
  {
    id: "clinical-trials-l3-q14",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Two similar trials report p = 0.049 and p = 0.051. Which interpretation best reflects sound evidence reasoning?",
    options: [
      {
        text: "The results should be interpreted with effect sizes, confidence intervals, endpoints, prior evidence, and trial quality rather than as a sharp split.",
        isCorrect: true,
      },
      {
        text: "The first result demonstrates a real effect and the second demonstrates absence of an effect.",
        isCorrect: false,
      },
      {
        text: "The second result is stronger because it is closer to the conventional 0.05 threshold from above.",
        isCorrect: false,
      },
      {
        text: "The first result gives the probability that the treatment works after the analysis, while the second gives the probability that it does not.",
        isCorrect: false,
      },
    ],
    explanation:
      "The 0.05 threshold is a convention, not a natural boundary where evidence changes abruptly. Results near the threshold should be interpreted by looking at the total evidence, including effect size, precision, endpoint quality, plausibility, and study conduct.",
  },
  {
    id: "clinical-trials-l3-q15",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe how sample size affects p-values and clinical interpretation?",
    options: [
      {
        text: "A very large trial can produce a small p-value for a tiny effect.",
        isCorrect: true,
      },
      {
        text: "A small trial can miss conventional statistical significance despite a potentially important effect.",
        isCorrect: true,
      },
      {
        text: "Effect size and confidence intervals are needed to judge meaning beyond the p-value.",
        isCorrect: true,
      },
      {
        text: "The p-value directly measures whether the effect is large enough for patients to notice.",
        isCorrect: false,
      },
    ],
    explanation:
      "P-values are affected by both effect size and information size. Large studies can detect very small differences, and small studies can leave important effects uncertain, so clinical interpretation must include magnitude, precision, and patient relevance.",
  },
  {
    id: "clinical-trials-l3-q16",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "A clinical team is deciding whether a statistically significant trial result should change practice. Which questions belong in the assessment?",
    options: [
      {
        text: "How large is the treatment effect in absolute and relative terms?",
        isCorrect: true,
      },
      {
        text: "How precise is the estimate and what does the confidence interval include?",
        isCorrect: true,
      },
      {
        text: "Is the endpoint meaningful to patients and clinicians?",
        isCorrect: true,
      },
      {
        text: "What harms, burdens, costs, alternatives, and patient preferences affect the benefit-risk balance?",
        isCorrect: true,
      },
    ],
    explanation:
      "Clinical significance asks whether an effect matters in the real world, not merely whether the null hypothesis was rejected. A practice decision should combine effect size, precision, endpoint relevance, safety, burden, alternatives, and patient values.",
  },
  {
    id: "clinical-trials-l3-q17",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "A trial of 100,000 patients finds that a drug lowers systolic blood pressure by 0.5 mmHg with p < 0.001. Which interpretations are appropriate?",
    options: [
      {
        text: "The result may show strong statistical evidence for a nonzero blood-pressure effect.",
        isCorrect: true,
      },
      {
        text: "The effect could still be clinically trivial if it does not improve symptoms, decisions, or patient outcomes.",
        isCorrect: true,
      },
      {
        text: "The small p-value establishes that the blood-pressure change is clinically important.",
        isCorrect: false,
      },
      {
        text: "The large sample size removes the need to examine the effect size.",
        isCorrect: false,
      },
    ],
    explanation:
      "A huge study can detect very small nonzero changes. Statistical evidence that the effect is real does not establish that a 0.5 mmHg reduction changes outcomes, symptoms, treatment decisions, or practice.",
  },
  {
    id: "clinical-trials-l3-q18",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "A rare-cancer trial enrolls 40 patients, estimates a six-month median survival improvement, and reports p = 0.08. Which interpretation is most appropriate?",
    options: [
      {
        text: "The result is not definitive, but it may indicate a clinically meaningful signal that deserves more evidence.",
        isCorrect: true,
      },
      {
        text: "The p-value demonstrates that the treatment has no survival effect once conventional thresholds are applied to the trial result.",
        isCorrect: false,
      },
      {
        text: "The six-month estimate is definitive because the disease is rare.",
        isCorrect: false,
      },
      {
        text: "The p-value means there is a 92% probability that the treatment works.",
        isCorrect: false,
      },
    ],
    explanation:
      "A non-significant p-value does not establish absence of effect, especially in a small and uncertain study. In a rare severe disease, a six-month estimated survival difference may be clinically meaningful, but the evidence remains insufficient without more precision and supporting data.",
  },
  {
    id: "clinical-trials-l3-q19",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the Minimal Clinically Important Difference (MCID)?",
    options: [
      {
        text: "It is the smallest outcome change that patients would perceive as beneficial or that would justify a management change.",
        isCorrect: true,
      },
      {
        text: "It helps judge whether a statistically detectable effect is large enough to matter.",
        isCorrect: true,
      },
      {
        text: "It depends on the condition, outcome instrument, baseline severity, and patient situation.",
        isCorrect: true,
      },
      {
        text: "It is a fixed numerical threshold that transfers unchanged across diseases and outcome scales.",
        isCorrect: false,
      },
    ],
    explanation:
      "MCID links measurement to patient-relevant meaning. It is especially useful when a study detects a small change on a symptom or quality-of-life scale, but the threshold depends on the disease, instrument, baseline status, and patient context.",
  },
  {
    id: "clinical-trials-l3-q20",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which factors can change whether a treatment effect is clinically meaningful?",
    options: [
      {
        text: "The severity of the disease and outcome.",
        isCorrect: true,
      },
      {
        text: "Treatment harms, toxicity, and patient burden.",
        isCorrect: true,
      },
      {
        text: "Available alternatives and baseline risk.",
        isCorrect: true,
      },
      {
        text: "Patient values, quality of life, and cost or feasibility.",
        isCorrect: true,
      },
    ],
    explanation:
      "Clinical significance is contextual because the same numerical effect can matter differently across diseases, outcomes, and patients. Benefit-risk interpretation should include disease severity, endpoint importance, safety, burden, alternatives, values, and practical feasibility.",
  },
  {
    id: "clinical-trials-l3-q21",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly explain why survival analysis is used for time-to-event outcomes?",
    options: [
      {
        text: "It uses information about when events occur, not just whether they occur.",
        isCorrect: true,
      },
      {
        text: "It can use censored observations, where a participant is known to be event-free up to a recorded time.",
        isCorrect: true,
      },
      {
        text: "It treats a death after one month and a death after five years as equivalent observations.",
        isCorrect: false,
      },
      {
        text: "It applies to death outcomes but not to relapse, hospitalization, recovery, or device failure.",
        isCorrect: false,
      },
    ],
    explanation:
      "Time-to-event methods preserve timing information and use partial follow-up rather than reducing every participant to a yes-or-no endpoint. Censoring means the event time is not fully observed, but the known event-free period still contributes information.",
  },
  {
    id: "clinical-trials-l3-q22",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which description best matches a Kaplan-Meier curve?",
    options: [
      {
        text: "A stepwise estimate of the proportion surviving or remaining event-free over time, starting at 100% and dropping as events occur.",
        isCorrect: true,
      },
      {
        text: "A bar chart that compares the final number of events in each arm and places censored observations into the non-event group.",
        isCorrect: false,
      },
      {
        text: "A scatter plot that shows individual patient risk scores against treatment assignment.",
        isCorrect: false,
      },
      {
        text: "A table that calculates number needed to treat at a single follow-up time.",
        isCorrect: false,
      },
    ],
    explanation:
      "A Kaplan-Meier curve displays the estimated survival or event-free probability over time. The curve starts at 1 or 100% and steps downward when events occur, while censored observations may be marked separately.",
  },
  {
    id: "clinical-trials-l3-q23",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Two Kaplan-Meier curves are being compared in an oncology trial. Which interpretation practices are appropriate?",
    options: [
      {
        text: "A treatment curve above the control curve generally suggests better event-free survival, but magnitude and uncertainty still matter.",
        isCorrect: true,
      },
      {
        text: "The number-at-risk table helps judge how stable late parts of the curves are.",
        isCorrect: true,
      },
      {
        text: "Crossing or delayed-separation curves should prompt closer examination of the time pattern of treatment effect.",
        isCorrect: true,
      },
      {
        text: "Late curve separation is interpreted the same way whether hundreds of patients or a few patients remain under observation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Kaplan-Meier curves require more than a quick visual comparison. The number at risk, censoring, uncertainty, follow-up duration, and time pattern of separation all affect interpretation, especially late in a study when few patients may remain.",
  },
  {
    id: "clinical-trials-l3-q24",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe median survival in time-to-event analysis?",
    options: [
      {
        text: "It is the time by which 50% of participants have experienced the event.",
        isCorrect: true,
      },
      {
        text: "If median overall survival is 15 months with treatment and 12 months with control, the median improvement is 3 months.",
        isCorrect: true,
      },
      {
        text: "It gives an intuitive absolute summary of a survival result.",
        isCorrect: true,
      },
      {
        text: "It can miss differences in the full survival curve, such as long-term tails or subgroup benefit.",
        isCorrect: true,
      },
    ],
    explanation:
      "Median survival is easy to understand because it gives a time point at which half the group has had the event. It is incomplete because two survival curves can have the same median while differing in early events, late tails, or benefit concentrated in a subgroup.",
  },
  {
    id: "clinical-trials-l3-q25",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "A trial reports a hazard ratio of 0.75 for death. Which statements correctly interpret the measure?",
    options: [
      {
        text: "The hazard is the event rate among participants who have not yet had the event at a given time.",
        isCorrect: true,
      },
      {
        text: "A hazard ratio of 0.75 estimates a 25% lower event rate over time in the treatment group, assuming the model is appropriate.",
        isCorrect: true,
      },
      {
        text: "A hazard ratio of 0.75 means 25% fewer participants died by the end of follow-up.",
        isCorrect: false,
      },
      {
        text: "The hazard ratio is a direct calculation of the cumulative risk ratio at a chosen time point.",
        isCorrect: false,
      },
    ],
    explanation:
      "A hazard ratio compares event rates over time among those still at risk. It is not the same as a cumulative risk ratio or a statement that a certain percentage fewer patients experienced the event by study end.",
  },
  {
    id: "clinical-trials-l3-q26",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "At 12 months, death occurred in 15% of treatment patients and 20% of control patients, while the trial also reports a hazard ratio from a survival model. Which statement best distinguishes the two quantities?",
    options: [
      {
        text: "The 12-month risk ratio compares cumulative risk at a specified time, while the hazard ratio compares event rates across follow-up.",
        isCorrect: true,
      },
      {
        text: "The hazard ratio is calculated by dividing 15% by 20%, while the risk ratio is estimated from all event times and censoring patterns.",
        isCorrect: false,
      },
      {
        text: "The risk ratio and hazard ratio answer the same question whenever death is the endpoint.",
        isCorrect: false,
      },
      {
        text: "The 12-month risk ratio summarizes censoring patterns more directly than a Kaplan-Meier curve.",
        isCorrect: false,
      },
    ],
    explanation:
      "A risk ratio is tied to cumulative event probability over a defined time window, such as death by 12 months. A hazard ratio summarizes relative event rates over follow-up, often using a proportional hazards model, so it should not be read as the same quantity.",
  },
  {
    id: "clinical-trials-l3-q27",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which situations can make a single hazard ratio hide important survival-pattern details?",
    options: [
      {
        text: "The survival curves overlap early and separate late.",
        isCorrect: true,
      },
      {
        text: "The curves cross during follow-up.",
        isCorrect: true,
      },
      {
        text: "A small subgroup has long-term benefit while the median changes little.",
        isCorrect: true,
      },
      {
        text: "The reported p-value is below the conventional threshold.",
        isCorrect: false,
      },
    ],
    explanation:
      "A single hazard ratio can hide time patterns such as delayed benefit, crossing curves, or long-term tails. The p-value does not reveal those shapes, so the Kaplan-Meier curves and clinical context are needed to understand the result.",
  },
  {
    id: "clinical-trials-l3-q28",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly distinguish overall survival from progression-free survival in oncology trials?",
    options: [
      {
        text: "Overall survival measures time until death from any cause.",
        isCorrect: true,
      },
      {
        text: "Progression-free survival measures time until disease progression or death.",
        isCorrect: true,
      },
      {
        text: "Progression-free survival may be observed earlier than overall survival.",
        isCorrect: true,
      },
      {
        text: "A progression-free survival benefit should still be interpreted with overall survival, quality of life, symptoms, and toxicity.",
        isCorrect: true,
      },
    ],
    explanation:
      "Overall survival is a hard and patient-relevant endpoint because it measures death from any cause. Progression-free survival can be useful and available earlier, but delaying progression does not necessarily mean longer life or better quality of life, especially if toxicity is substantial.",
  },
  {
    id: "clinical-trials-l3-q29",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "A cancer trial reports median overall survival of 14 months with treatment versus 11 months with control, hazard ratio 0.78, 95% confidence interval 0.65 to 0.94, and p = 0.01. Which interpretations are appropriate?",
    options: [
      {
        text: "The result provides statistical evidence of survival benefit, with a 3-month median survival gain.",
        isCorrect: true,
      },
      {
        text: "The hazard ratio suggests a lower event rate over time, and the confidence interval excludes 1.",
        isCorrect: true,
      },
      {
        text: "The 3-month median gain is sufficient by itself to declare the therapy practice-changing.",
        isCorrect: false,
      },
      {
        text: "The p-value means there is a 99% probability that the treatment improves survival.",
        isCorrect: false,
      },
    ],
    explanation:
      "The numbers support evidence of benefit: the median survival difference is 3 months, the hazard ratio favors treatment, the confidence interval excludes 1, and the p-value is small under the null model. A full interpretation still needs toxicity, quality of life, population relevance, follow-up, subsequent therapies, and alternatives.",
  },
  {
    id: "clinical-trials-l3-q30",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement best distinguishes a systematic review from a narrative review?",
    options: [
      {
        text: "A systematic review uses predefined search strategies, inclusion criteria, outcomes, and methods to reduce reviewer selection bias.",
        isCorrect: true,
      },
      {
        text: "A systematic review is defined by using statistical pooling of treatment effects, while a narrative review is defined by avoiding confidence intervals.",
        isCorrect: false,
      },
      {
        text: "A narrative review includes all relevant studies, while a systematic review selects examples that support one interpretation.",
        isCorrect: false,
      },
      {
        text: "A systematic review is a single-trial analysis that compares two treatment arms using a survival model.",
        isCorrect: false,
      },
    ],
    explanation:
      "The key feature of a systematic review is a structured, transparent process for finding and selecting evidence. Meta-analysis may be included, but systematic review and statistical pooling are not the same thing.",
  },
  {
    id: "clinical-trials-l3-q31",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements correctly describe meta-analysis?",
    options: [
      {
        text: "It statistically combines results from multiple studies to estimate an overall effect.",
        isCorrect: true,
      },
      {
        text: "It can increase precision when included studies are sufficiently comparable and informative.",
        isCorrect: true,
      },
      {
        text: "It can be misleading when included studies are biased, heterogeneous, poorly designed, or incomplete.",
        isCorrect: true,
      },
      {
        text: "It converts weak or selectively published studies into a strong evidence base through pooling.",
        isCorrect: false,
      },
    ],
    explanation:
      "Meta-analysis is useful because combining studies can improve precision and summarize a broader evidence base. It is not magic: biased, incomparable, or selectively published studies can produce a misleading pooled estimate.",
  },
  {
    id: "clinical-trials-l3-q32",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which components are commonly found in a forest plot for a meta-analysis?",
    options: [
      {
        text: "Individual study point estimates.",
        isCorrect: true,
      },
      {
        text: "Horizontal confidence intervals for individual studies.",
        isCorrect: true,
      },
      {
        text: "A vertical line representing no effect.",
        isCorrect: true,
      },
      {
        text: "A pooled estimate, often shown as a diamond.",
        isCorrect: true,
      },
    ],
    explanation:
      "A forest plot shows how each study estimates an effect and how uncertain each estimate is. It usually includes a no-effect line and a pooled estimate so readers can judge direction, precision, consistency, and overall effect.",
  },
  {
    id: "clinical-trials-l3-q33",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "In a forest plot, how should the no-effect line and confidence intervals be interpreted?",
    options: [
      {
        text: "For risk ratios, odds ratios, and hazard ratios, the no-effect value is 1.",
        isCorrect: true,
      },
      {
        text: "For mean differences, the no-effect value is 0.",
        isCorrect: true,
      },
      {
        text: "A confidence interval crossing the no-effect line establishes that the treatments are equivalent.",
        isCorrect: false,
      },
      {
        text: "A point estimate farther from the no-effect line is necessarily more precise.",
        isCorrect: false,
      },
    ],
    explanation:
      "Ratio measures use 1 as the no-effect value because equal risks, odds, or hazards give a ratio of 1; difference measures use 0 because equal means give no difference. Crossing the no-effect line means the interval includes no effect, not that equivalence has been demonstrated.",
  },
  {
    id: "clinical-trials-l3-q34",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "A meta-analysis combines trials with different populations, doses, endpoints, follow-up times, and study quality. Which interpretation of heterogeneity is most appropriate?",
    options: [
      {
        text: "The studies may differ enough that the meaning of a single pooled estimate should be questioned.",
        isCorrect: true,
      },
      {
        text: "The pooled estimate should be preferred because heterogeneity mainly improves generalizability.",
        isCorrect: false,
      },
      {
        text: "Differences in endpoints and follow-up are separate from heterogeneity and do not affect pooling decisions.",
        isCorrect: false,
      },
      {
        text: "Heterogeneity indicates publication bias rather than differences among study results.",
        isCorrect: false,
      },
    ],
    explanation:
      "Heterogeneity means study results differ more than expected from chance alone or differ for clinically important reasons. Before focusing on the pooled estimate, readers should ask whether the studies are similar enough in population, intervention, comparator, endpoint, timing, and quality to combine meaningfully.",
  },
  {
    id: "clinical-trials-l3-q35",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe publication bias and funnel plots?",
    options: [
      {
        text: "Positive or favorable studies are more likely to appear in the published literature than negative or inconclusive studies.",
        isCorrect: true,
      },
      {
        text: "A meta-analysis based mainly on visible published studies can overstate benefit if missing studies are systematically different.",
        isCorrect: true,
      },
      {
        text: "Funnel-plot asymmetry can suggest missing small studies, but it can also arise for other reasons.",
        isCorrect: true,
      },
      {
        text: "Trial registration has removed publication bias from modern evidence synthesis.",
        isCorrect: false,
      },
    ],
    explanation:
      "Publication bias distorts the evidence base when favorable studies are easier to see than negative or inconclusive studies. Funnel plots can provide clues about missing small studies, but asymmetry is not definitive and trial registration has reduced rather than eliminated the problem.",
  },
  {
    id: "clinical-trials-l3-q36",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "When synthesizing evidence across multiple clinical studies, which questions are important?",
    options: [
      {
        text: "How good are the included studies?",
        isCorrect: true,
      },
      {
        text: "How consistent are the results across studies?",
        isCorrect: true,
      },
      {
        text: "How large and precise is the effect?",
        isCorrect: true,
      },
      {
        text: "How complete is the evidence base, including unpublished or negative studies?",
        isCorrect: true,
      },
    ],
    explanation:
      "Evidence synthesis is not a vote count. Strong interpretation requires study quality, consistency, effect size, precision, and completeness of the evidence base, including concern for missing negative or inconclusive data.",
  },
  {
    id: "clinical-trials-l3-q37",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "A randomized trial of 2,000 patients with chronic kidney disease reports that Drug X reduced a composite endpoint of hospitalization or biomarker worsening versus placebo, with hazard ratio 0.82, 95% confidence interval 0.70 to 0.96, and p = 0.01. Which follow-up questions are most important for interpretation?",
    options: [
      {
        text: "What were the absolute event rates, and which component drove the composite endpoint?",
        isCorrect: true,
      },
      {
        text: "Was the biomarker validated, and what were the safety and quality-of-life findings?",
        isCorrect: true,
      },
      {
        text: "Was the result statistically significant enough to replace questions about endpoint meaning?",
        isCorrect: false,
      },
      {
        text: "Did the hazard ratio of 0.82 establish a large absolute benefit regardless of baseline risk?",
        isCorrect: false,
      },
    ],
    explanation:
      "The abstract suggests a statistically positive result, but clinical interpretation needs more detail. A composite endpoint can be driven by a less meaningful component, and a hazard ratio does not reveal the absolute benefit, safety profile, biomarker validity, or patient relevance.",
  },
  {
    id: "clinical-trials-l3-q38",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "A Phase II oncology trial reports median progression-free survival of 5.5 months with treatment versus 4.0 months with control, hazard ratio 0.72, p = 0.04, immature overall survival data, and grade 3 or higher adverse events in 45% versus 20% of patients. Which interpretation is most appropriate?",
    options: [
      {
        text: "The trial shows a progression-free survival signal, but clinical meaning depends on toxicity, quality of life, overall survival, disease setting, and alternatives.",
        isCorrect: true,
      },
      {
        text: "The progression-free survival result is sufficient for a practice-changing conclusion because p = 0.04 despite immature overall survival and higher toxicity.",
        isCorrect: false,
      },
      {
        text: "Progression-free survival improvement and overall survival improvement have identical patient meaning in this setting.",
        isCorrect: false,
      },
      {
        text: "The higher adverse-event rate should be separated from interpretation of the efficacy result.",
        isCorrect: false,
      },
    ],
    explanation:
      "The result may justify further development, but it is not automatically practice-changing. A 1.5-month progression-free survival improvement must be weighed against toxicity and interpreted with symptoms, quality of life, overall survival maturity, disease context, and available options.",
  },
  {
    id: "clinical-trials-l3-q39",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "A small rare-disease trial misses conventional statistical significance with p = 0.09, but the estimated effect exceeds a pre-specified clinically meaningful threshold and the confidence interval is wide. Which interpretations are appropriate?",
    options: [
      {
        text: "The result is inconclusive rather than definitive evidence of no effect.",
        isCorrect: true,
      },
      {
        text: "The estimate may justify more study if supported by disease severity, unmet need, natural history, biomarkers, and patient outcomes.",
        isCorrect: true,
      },
      {
        text: "The wide interval means the true effect could be smaller, larger, or less favorable than the point estimate suggests.",
        isCorrect: true,
      },
      {
        text: "A p-value above 0.05 is enough to discard the totality of evidence in rare diseases.",
        isCorrect: false,
      },
    ],
    explanation:
      "A small rare-disease study can be underpowered and uncertain while still suggesting a clinically meaningful signal. The confidence interval matters because it shows the range of compatible effects, and the result should be interpreted alongside total evidence rather than dismissed by threshold alone.",
  },
  {
    id: "clinical-trials-l3-q40",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "A clinical research professional is reading a trial result before advising a sponsor or clinical team. Which lenses should guide interpretation?",
    options: [
      {
        text: "Effect size: how large the benefit or harm is and which metric is being used.",
        isCorrect: true,
      },
      {
        text: "Precision: how uncertain the estimate is and what the confidence interval contains.",
        isCorrect: true,
      },
      {
        text: "Clinical relevance: whether the endpoint, magnitude, safety, and patient burden matter.",
        isCorrect: true,
      },
      {
        text: "Total evidence: whether the result fits with prior trials, systematic reviews, biology, and missing-data concerns.",
        isCorrect: true,
      },
    ],
    explanation:
      "Clinical trial interpretation should not collapse a result into positive or negative. A strong interpretation combines effect size, uncertainty, clinical relevance, statistical evidence, safety, endpoint quality, prior evidence, and context such as disease severity and feasibility.",
  },
];
