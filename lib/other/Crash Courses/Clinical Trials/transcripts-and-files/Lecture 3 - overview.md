# Lecture 3 — Statistics and Evidence Interpretation

**Duration:** 60 minutes

**Theme:** _Statistical significance is not the same as clinical significance._

---

# Learning Objectives

By the end of this lecture, students should be able to:

1. Interpret the most common treatment effect measures reported in clinical trials.
2. Understand what confidence intervals communicate about uncertainty.
3. Correctly interpret p-values and hypothesis tests.
4. Distinguish statistical significance from clinical significance.
5. Read and interpret Kaplan-Meier survival curves.
6. Understand hazard ratios conceptually.
7. Interpret forest plots and understand the purpose of meta-analysis.
8. Recognize common statistical mistakes frequently encountered in medicine, pharma, and CRO environments.

---

# Lecture Overview

| Section | Topic                                | Time   |
| ------- | ------------------------------------ | ------ |
| 1       | Measuring Treatment Effects          | 12 min |
| 2       | Confidence Intervals and Uncertainty | 10 min |
| 3       | Hypothesis Testing and P-values      | 8 min  |
| 4       | Clinical vs Statistical Significance | 10 min |
| 5       | Survival Analysis                    | 12 min |
| 6       | Meta-Analysis and Evidence Synthesis | 8 min  |

---

# Part 1 — Measuring Treatment Effects

## Learning Goal

Students should understand the most common ways clinical trial results are expressed and why the choice of metric can dramatically affect interpretation.

---

# Opening Question

Suppose a new drug reduces mortality from:

10% → 5%

How impressive is that?

Most people answer:

"Very impressive."

But there are multiple ways to describe exactly the same result.

This is a major source of confusion in medicine.

---

# Absolute Risk

Treatment group:

5 deaths per 100 patients

Control group:

10 deaths per 100 patients

Absolute Risk Reduction (ARR):

5%

---

## Interpretation

For every 100 patients treated:

5 deaths are prevented.

This is often the most intuitive measure.

---

# Relative Risk Reduction

Same data:

10% → 5%

Relative Risk Reduction (RRR):

50%

---

## Interpretation

The treatment reduced risk by half.

This sounds much more impressive.

---

## Important Lesson

The exact same study result can be reported as:

- 5% absolute reduction
- 50% relative reduction

Both are mathematically correct.

The framing influences perception.

---

# Number Needed to Treat (NNT)

One of the most important concepts in clinical medicine.

Formula:

NNT = 1 / ARR

For ARR = 5%:

NNT = 20

---

## Interpretation

20 patients must be treated to prevent one additional adverse event.

This often provides the most clinically useful perspective.

---

## Discussion

Compare:

Treatment A:

NNT = 10

Treatment B:

NNT = 500

Both may be statistically significant.

Only one may be clinically attractive.

---

# Risk Ratios

Risk Ratio (RR):

Treatment Risk / Control Risk

Example:

5% / 10% = 0.5

Interpretation:

Patients receiving treatment experience half the risk.

---

# Odds Ratios

Frequently encountered in:

- observational studies
- case-control studies
- logistic regression

Students already know the mathematics.

Focus on interpretation:

Odds ratios approximate risk ratios for rare events.

For common outcomes they can exaggerate perceived effects.

---

# Key Takeaway

Different measures answer different questions:

| Measure | Answers                                  |
| ------- | ---------------------------------------- |
| ARR     | How many events are prevented?           |
| RRR     | How large is the proportional reduction? |
| NNT     | How many patients must be treated?       |
| RR      | How much risk changes proportionally?    |
| OR      | How odds change between groups?          |

---

# Part 2 — Confidence Intervals and Uncertainty

## Learning Goal

Students should understand that every estimate contains uncertainty.

---

# The Problem with Point Estimates

Suppose a study reports:

"Mortality reduced by 20%."

Question:

How certain are we?

The estimate alone does not tell us.

---

# Why Confidence Intervals Exist

Clinical trials observe only a sample.

Not the entire population.

Different samples would produce different results.

Therefore:

Every estimate is uncertain.

---

# Example

Risk Ratio = 0.80

95% CI:

0.40 – 1.60

Interpretation:

Data are consistent with:

- substantial benefit
- no effect
- substantial harm

The estimate is highly uncertain.

---

# Narrow Confidence Interval

Example:

0.78 – 0.82

Suggests:

- high precision
- large sample size
- less uncertainty

---

# Wide Confidence Interval

Example:

0.30 – 1.90

Suggests:

- limited information
- small sample size
- large uncertainty

---

# Important Interpretation Principle

Confidence intervals are often more informative than p-values.

They communicate:

- effect size
- uncertainty
- precision

simultaneously.

---

# Example Discussion

Which result inspires more confidence?

Study A:

RR = 0.80
CI: 0.78–0.82

Study B:

RR = 0.50
CI: 0.10–2.50

Students should recognize that the more dramatic estimate may actually provide less reliable evidence.

---

# Part 3 — Hypothesis Testing and P-values

## Learning Goal

Students should understand what p-values do—and do not—mean.

---

# Brief Review

Null hypothesis:

No treatment effect.

Alternative hypothesis:

Treatment effect exists.

---

# What Is a P-value?

Conceptually:

> If there were truly no effect, how surprising would these data be?

Small p-values imply:

Observed results would be unlikely under the null hypothesis.

---

# Common Misinterpretations

A p-value is NOT:

- probability the treatment works
- probability the null hypothesis is true
- probability the results occurred by chance

These misunderstandings are extremely common.

---

# Example

p = 0.03

Incorrect interpretation:

"There is a 97% chance the drug works."

Correct interpretation:

"If no treatment effect existed, data this extreme would occur roughly 3% of the time."

---

# Statistical Significance

Convention:

p < 0.05

Historically adopted threshold.

Important:

Nature does not care about 0.05.

The distinction is ultimately arbitrary.

---

# Industry Perspective

Increasingly:

- effect size
- confidence intervals
- clinical relevance

matter more than simply "crossing 0.05."

---

# Part 4 — Clinical vs Statistical Significance

## Learning Goal

Students should understand one of the most important distinctions in evidence interpretation.

---

# Statistical Significance

Answers:

> Is there evidence that an effect exists?

---

# Clinical Significance

Answers:

> Does the effect actually matter?

These are different questions.

---

# Example 1

Massive trial:

100,000 patients

Blood pressure reduced by:

0.5 mmHg

p < 0.001

Highly statistically significant.

Clinically:

Probably irrelevant.

---

# Example 2

Rare cancer study:

40 patients

Median survival improvement:

6 months

p = 0.08

Not statistically significant.

Clinically:

Potentially very important.

---

# Effect Size Matters

When evaluating trials:

Ask:

1. Is the effect real?
2. How large is it?
3. Does it matter to patients?

---

# Minimal Clinically Important Difference (MCID)

Introduce concept:

Smallest improvement patients would consider meaningful.

Common in:

- pain research
- quality-of-life research
- rehabilitation

---

# Key Lesson

Never ask only:

> "Is it significant?"

Ask:

> "How much benefit is there?"

---

# Part 5 — Survival Analysis

## Learning Goal

Students should understand the most common statistical framework used in oncology, cardiology, and many pharmaceutical trials.

---

# Why Survival Analysis Exists

Many clinical outcomes involve time.

Examples:

- time until death
- time until relapse
- time until hospitalization

Traditional methods ignore timing information.

Survival analysis incorporates it.

---

# Time-to-Event Data

Focus:

Not simply whether an event occurred.

But:

When it occurred.

---

# Censoring

Critical concept.

Some patients:

- leave study
- are lost to follow-up
- remain event-free when study ends

We have partial information.

Survival analysis accommodates this.

---

# Kaplan-Meier Curves

Industry-standard visualization.

Students should learn how to read:

- x-axis = time
- y-axis = proportion surviving/event-free

---

# Interpretation

Steeper decline:

More events occurring.

Flatter curve:

Fewer events occurring.

---

# Comparing Curves

Treatment curve above control curve:

Generally suggests benefit.

But must evaluate:

- magnitude
- uncertainty
- statistical testing

---

# Hazard Ratios

Among the most commonly reported statistics in modern trials.

---

## Conceptual Interpretation

Hazard Ratio = 0.75

Interpretation:

At any point in time:

Treatment group experiences roughly 25% lower event rate.

---

## Important Caveat

Hazard ratios are not risk ratios.

Many industry professionals confuse these.

Students should recognize the distinction.

---

# Example Discussion

Which sounds better?

- HR = 0.80
- Median survival improvement = 2 weeks

Students should recognize that hazard ratios alone can be misleading without clinical context.

---

# Part 6 — Meta-Analysis and Evidence Synthesis

## Learning Goal

Students should understand how multiple studies are combined into a broader evidence base.

---

# Why Individual Trials Are Limited

Any single study may suffer from:

- random error
- small sample size
- unusual patient populations

Medicine rarely relies on one study alone.

---

# Systematic Reviews

Structured process for identifying:

- all relevant studies
- predefined inclusion criteria
- transparent methodology

Goal:

Minimize reviewer bias.

---

# Meta-Analysis

Statistical combination of study results.

Purpose:

Increase precision.

Improve overall estimates.

---

# Forest Plots

Students should learn the anatomy of a forest plot.

Key components:

- individual study estimates
- confidence intervals
- pooled estimate

---

# Heterogeneity

Not all studies agree.

Sources:

- different populations
- different interventions
- different methodologies

Question:

Should studies even be combined?

---

# Publication Bias

Negative studies often remain unpublished.

This can distort the evidence base.

---

# Funnel Plot Intuition

If only positive studies are visible:

The literature becomes systematically biased.

Meta-analysis cannot fully fix missing evidence.

---

# Final Takeaway

Evidence synthesis is not simply:

"Count studies."

Instead ask:

- How good are the studies?
- How consistent are the results?
- How large is the effect?
- How certain are we?

---

# End-of-Lecture Synthesis

The central lesson of this lecture is that clinical research is fundamentally about **estimating effects under uncertainty**.

A trial result should never be reduced to:

> "The study was positive."

Instead, every result should be interpreted through four lenses:

1. **Effect Size** — How large is the benefit?
2. **Precision** — How certain are we?
3. **Clinical Relevance** — Does the benefit matter?
4. **Total Evidence** — How does this fit with the broader literature?

Students who internalize these principles will be able to read trial abstracts, interpret pharmaceutical claims, understand CRO reports, and participate meaningfully in discussions with clinicians, biostatisticians, medical affairs teams, and clinical development professionals.
