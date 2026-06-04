# Lecture 3 Transcript — Statistics and Evidence Interpretation

Welcome back. In the first lecture, we asked why clinical trials exist. The central answer was that medicine is full of uncertainty, bias, natural fluctuation, placebo effects, confounding, and random variation. Improvement after treatment does not necessarily mean improvement because of treatment.

In the second lecture, we asked how clinical trials are designed. We discussed PICO: population, intervention, comparator, outcome, and time. We looked at randomization, blinding, endpoint selection, phases of clinical development, and the tradeoff between internal and external validity.

Today we turn to a new question:

Once a clinical trial has been run, how do we interpret the results?

This is where statistics enters the picture. But I want to be clear about the purpose of this lecture. This is not a mathematical statistics lecture. I am assuming that you already have decent knowledge of mathematics and statistics. So I will not spend much time deriving formulas or proving properties of estimators. Instead, we will focus on interpretation.

In clinical research, the statistical mistake is often not that people cannot calculate a p-value. The deeper problem is that they misunderstand what the result means. They confuse relative and absolute effects. They confuse statistical significance with clinical importance. They treat confidence intervals as decorative additions rather than central information. They interpret hazard ratios as if they were simple risk ratios. They read a positive trial abstract and miss that the endpoint was weak, the effect was tiny, or the confidence interval was wide.

So the theme of today’s lecture is:

**Statistical significance is not the same as clinical significance.**

That sentence is one of the most important lessons in evidence interpretation.

By the end of today’s lecture, you should be able to interpret common treatment effect measures, understand what confidence intervals communicate, correctly interpret p-values, distinguish statistical significance from clinical significance, read Kaplan-Meier curves, understand hazard ratios conceptually, interpret forest plots, and recognize common statistical misunderstandings in medicine, pharma, and CRO environments.

Let us begin with treatment effects.

Suppose a new drug reduces mortality from 10% to 5%.

How impressive is that?

At first glance, very impressive. The death rate is cut in half. But now we need to slow down and ask: how exactly should we describe this effect?

This single result can be described in several mathematically correct ways.

In the control group, 10 out of 100 patients die.

In the treatment group, 5 out of 100 patients die.

The absolute risk reduction is 5 percentage points.

That means that for every 100 patients treated, 5 deaths are prevented compared with control.

This is often the most intuitive way to describe the effect.

Now let us describe the same result differently.

The risk went from 10% to 5%. Five is half of ten. So the relative risk reduction is 50%.

The treatment reduced mortality by half.

That sounds much more dramatic than “5 percentage points.”

Both statements are true. The treatment reduced absolute risk by 5 percentage points and relative risk by 50%. The difference is framing.

This is one of the most common sources of misunderstanding in medicine and public communication. Relative risk reductions often sound more impressive than absolute risk reductions.

If a newspaper headline says, “New drug cuts risk by 50%,” that might mean risk went from 10% to 5%, which is substantial. But it could also mean risk went from 0.2% to 0.1%, which is a much smaller absolute difference. The relative reduction is still 50%, but the absolute benefit is tiny.

So whenever you hear a relative risk reduction, ask: relative to what baseline risk?

Baseline risk matters enormously.

Let us take two examples.

In Study A, a drug reduces risk from 20% to 10%.

In Study B, the same relative effect reduces risk from 2% to 1%.

In both studies, the relative risk reduction is 50%. But the absolute risk reduction in Study A is 10 percentage points, while in Study B it is 1 percentage point.

In Study A, 10 events are prevented per 100 patients treated. In Study B, 1 event is prevented per 100 patients treated.

Same relative effect. Very different clinical implications.

This leads us to the number needed to treat, or NNT.

The NNT tells us how many patients need to be treated to prevent one additional bad outcome, or to achieve one additional beneficial outcome, compared with control.

The formula is simple:

NNT equals 1 divided by the absolute risk reduction.

If the absolute risk reduction is 5%, or 0.05, then the NNT is 1 divided by 0.05, which equals 20.

So if mortality goes from 10% to 5%, the NNT is 20. We need to treat 20 patients to prevent one additional death.

NNT is powerful because it translates trial results into something clinically tangible.

An NNT of 20 for preventing death may be very attractive. An NNT of 500 for preventing a mild symptom may be less attractive. But NNT must always be interpreted in context.

What outcome are we preventing? Death? Stroke? Mild nausea? A lab abnormality?

Over what time period? An NNT of 20 over five years is different from an NNT of 20 over one month.

What are the harms? If the treatment is safe, cheap, and easy, a higher NNT may be acceptable. If it is expensive, toxic, invasive, or burdensome, we may require a lower NNT.

This is why NNT should not be treated as a universal ranking. It is a helpful interpretive tool, not a complete decision rule.

There is also a related concept: number needed to harm, or NNH.

If a drug causes serious bleeding in 1 additional patient per 100 treated, the number needed to harm is 100. That means for every 100 patients treated, one additional serious bleeding event occurs because of the drug.

Clinical decisions often require comparing NNT and NNH.

Suppose a blood thinner prevents one stroke for every 50 patients treated but causes one major bleed for every 100 patients treated. Is that acceptable? It depends on the severity of strokes, severity of bleeds, patient risk, patient values, and available alternatives.

So when interpreting clinical evidence, benefit and harm should be considered together.

Now let us discuss risk ratios.

A risk ratio compares the risk in the treatment group to the risk in the control group.

If the treatment risk is 5% and the control risk is 10%, the risk ratio is 0.5.

A risk ratio of 0.5 means the treatment group has half the risk of the control group.

A risk ratio of 1 means no difference.

A risk ratio below 1 means lower risk in the treatment group, assuming the outcome is bad.

A risk ratio above 1 means higher risk in the treatment group.

For beneficial outcomes, interpretation reverses slightly. If the outcome is recovery, a risk ratio above 1 may be good. So always ask: what is the outcome?

Risk ratios are intuitive for many clinical outcomes. But you will also encounter odds ratios, especially in observational studies, case-control studies, and logistic regression.

Odds are not the same as risk.

Risk is probability: events divided by total people.

Odds are events divided by non-events.

If 10 out of 100 people have an event, the risk is 10%. The odds are 10 to 90, or about 0.111.

For rare events, odds and risk are similar. For common events, they diverge.

This matters because odds ratios can exaggerate perceived effects when outcomes are common.

For example, if an event occurs in 40% of the control group and 20% of the treatment group, the risk ratio is 0.5. The risk is cut in half. But the odds ratio is 0.375, which can sound like a larger effect if interpreted incorrectly.

Many non-statisticians interpret odds ratios as risk ratios. This is a common mistake.

If you are speaking with people in clinical research, and someone says “odds ratio,” listen carefully. Ask whether the outcome is rare. Ask whether the effect is being interpreted as a risk difference, risk ratio, or odds ratio. In many contexts, especially logistic regression output, odds ratios are reported because of the model used, not because they are the most intuitive effect measure.

Now let us summarize the first major lesson.

Different treatment effect measures answer different questions.

Absolute risk reduction asks: how many fewer events occur?

Relative risk reduction asks: how large is the proportional reduction?

Number needed to treat asks: how many patients must be treated to prevent one event?

Risk ratio asks: how much lower or higher is risk in one group compared with another?

Odds ratio asks: how do the odds differ between groups?

None of these is automatically the “correct” measure in every context. But in clinical interpretation, absolute effects are often essential because they connect statistical results to patient impact.

Now we move to confidence intervals.

Suppose a study reports: mortality was reduced by 20%.

That sounds informative, but something is missing.

How certain are we?

Was the estimate based on 50 patients or 50,000 patients? Was the result precise or very uncertain? Could the true effect be much smaller? Could it even be harmful?

A point estimate alone does not answer these questions.

That is why confidence intervals are so important.

A clinical trial observes a sample, not the entire population. If we repeated the trial with a different sample, we would get a different estimate. Random variation means that every estimate is uncertain.

A confidence interval gives a range of values compatible with the observed data, under the statistical model.

Let us use a simple example.

A trial reports a risk ratio of 0.80 with a 95% confidence interval from 0.40 to 1.60.

The point estimate suggests a 20% relative risk reduction. But the confidence interval is very wide. It includes 0.40, which would be a large benefit. It includes 1.0, which means no difference. It includes 1.60, which would mean substantial harm.

So the correct interpretation is not simply, “The drug reduced risk by 20%.”

A better interpretation is:

“The point estimate suggests benefit, but the data are compatible with substantial benefit, no effect, or harm. The result is highly uncertain.”

That is very different.

Now consider another study.

Risk ratio: 0.80.

95% confidence interval: 0.78 to 0.82.

Here the point estimate is the same: 0.80. But the interval is narrow. The result is precise. We have much stronger evidence that the effect is close to a 20% relative reduction.

So confidence intervals tell us about precision.

A narrow interval suggests high precision, often because of large sample size, many events, or low variability.

A wide interval suggests limited information, often because of small sample size, few events, high variability, or poor data quality.

Confidence intervals also help us judge clinical relevance.

Suppose a trial reports a mean improvement of 2 points on a symptom scale, with a 95% confidence interval from 0.1 to 3.9. If the minimal clinically important difference is 3 points, then the interval includes both clinically trivial and clinically meaningful effects. We should be cautious.

Or suppose the mean improvement is 0.5 points with a confidence interval from 0.4 to 0.6. This is statistically precise, but if 0.5 points is clinically meaningless, the treatment may not matter despite precision.

So confidence intervals combine effect size and uncertainty.

This is why many statisticians and clinical methodologists emphasize confidence intervals over p-values. P-values reduce evidence to a threshold. Confidence intervals show the range of plausible effects.

Let us compare two studies.

Study A reports a risk ratio of 0.80 with a confidence interval from 0.78 to 0.82.

Study B reports a risk ratio of 0.50 with a confidence interval from 0.10 to 2.50.

Which result inspires more confidence?

At first, Study B looks more dramatic. A risk ratio of 0.50 suggests the risk is cut in half. But the interval is enormous. The true effect could be a 90% reduction, no effect, or a large increase in risk. Study A is less dramatic but far more precise.

This is a common clinical interpretation issue. A dramatic point estimate from a small study can be less reliable than a modest estimate from a large study.

Now, let us address a common misunderstanding about confidence intervals.

A 95% confidence interval is often described informally as “the range where the true value probably lies.” That is intuitive, but technically not quite right under frequentist interpretation. The strict frequentist interpretation is that if we repeated the same procedure many times, 95% of the constructed intervals would contain the true parameter.

However, in practical clinical communication, people often use confidence intervals as a way to express uncertainty about the plausible range of effects. That is acceptable as long as we understand the basic idea: the interval reflects uncertainty, not a guarantee.

For industry conversations, the most important thing is not the philosophical interpretation. It is to ask:

Is the estimate precise?

Does the interval include no effect?

Does the interval include clinically meaningful benefit?

Does the interval include clinically meaningful harm?

Does the whole interval lie within a range that would support a decision?

These questions are practical and extremely useful.

Now we turn to hypothesis testing and p-values.

Many people treat p-values as the central output of a trial. The trial is “positive” if p is less than 0.05 and “negative” if p is greater than 0.05.

This is too simplistic.

Let us begin with the basic idea.

In a conventional hypothesis test, we define a null hypothesis. Usually, the null hypothesis says there is no treatment effect.

Then we ask: if the null hypothesis were true, how surprising would the observed data be?

That surprise is summarized by the p-value.

A small p-value means that the observed data would be unlikely under the null hypothesis.

For example, p = 0.03 means that if there were truly no effect, data this extreme, or more extreme, would occur about 3% of the time under the assumptions of the test.

That is what it means.

Now let us say what it does not mean.

A p-value is not the probability that the treatment works.

A p-value is not the probability that the null hypothesis is true.

A p-value is not the probability that the result occurred by chance.

A p-value of 0.03 does not mean there is a 97% chance the drug works.

That mistake is extremely common.

The p-value is calculated under the assumption that the null hypothesis is true. It does not directly tell us the probability that the null hypothesis is true.

Now, why do people use p < 0.05?

The 0.05 threshold is a convention. It became historically common, but nature does not care about 0.05.

A result with p = 0.049 is not fundamentally different from p = 0.051. Treating one as “real” and the other as “nothing” is bad reasoning.

This matters in clinical trials.

Suppose one study reports p = 0.048 and another reports p = 0.06. The first is conventionally statistically significant; the second is not. But interpretation should depend on effect size, confidence interval, endpoint, prior evidence, biological plausibility, trial quality, and clinical relevance.

P-values are especially problematic when they become a binary gate.

In regulatory settings, thresholds matter because decisions require rules. But scientifically, we should avoid thinking that evidence changes abruptly at 0.05.

There is another issue: p-values are influenced by sample size.

A very large trial can produce a tiny p-value for a trivial effect.

A small trial can fail to reach statistical significance despite a potentially important effect.

So p-values alone do not answer the clinical question.

Let us use an example.

A trial enrolls 100,000 patients and finds that a drug lowers systolic blood pressure by 0.5 mmHg, with p < 0.001.

Statistically, this is strong evidence that the effect is not zero. But clinically, a 0.5 mmHg reduction may be meaningless for most purposes.

Now take the opposite example.

A small trial in a rare cancer enrolls 40 patients. Median survival appears to improve by six months, but p = 0.08.

This is not conventionally statistically significant. But in a rare, severe disease with few options, a six-month survival improvement may be clinically very important and worth further study.

So statistical significance asks: is there evidence against the null?

Clinical significance asks: does the effect matter?

They are different questions.

This brings us to Part 4: clinical versus statistical significance.

Statistical significance is about evidence that an effect exists, relative to a statistical model and threshold.

Clinical significance is about whether the effect is meaningful for patients, clinicians, payers, regulators, or health systems.

A tiny effect can be statistically significant.

A meaningful effect can be statistically non-significant if the study is too small or uncertain.

So when evaluating a trial, never ask only: was it significant?

Ask:

How large was the effect?

How precise was the estimate?

Was the endpoint meaningful?

What were the harms?

What was the baseline risk?

What alternatives exist?

What do patients care about?

Let us expand the blood pressure example.

Suppose Drug A reduces systolic blood pressure by 0.5 mmHg in a huge trial. The p-value is less than 0.001. This means the study provides strong statistical evidence that the drug changes blood pressure. But if the effect is so small that it does not reduce strokes, heart attacks, symptoms, or treatment decisions, the effect may be clinically irrelevant.

Now suppose Drug B reduces blood pressure by 10 mmHg but the trial includes only 25 patients, and the confidence interval is wide. The p-value is 0.07. This result is uncertain, but the possible effect is clinically meaningful. The correct conclusion is not “Drug B failed and does nothing.” The correct conclusion may be: “This small study suggests a potentially meaningful effect, but evidence is insufficient; larger trials are needed.”

In industry, the distinction matters because development decisions often occur under uncertainty. A Phase II study may not provide definitive proof, but it may show enough signal to justify Phase III. Conversely, a statistically significant Phase II result may not justify further investment if the effect is too small or the endpoint is weak.

Now let us introduce the minimal clinically important difference, or MCID.

The MCID is the smallest change in an outcome that patients would perceive as beneficial or that would justify a change in management.

This concept is common in pain research, quality-of-life research, rehabilitation, psychiatry, and chronic disease.

For example, on a 0-to-10 pain scale, a reduction of 0.2 points may be statistically detectable in a huge study but not meaningful to patients. A reduction of 2 points may be meaningful. The exact MCID depends on the condition, instrument, baseline severity, and patient context.

MCID helps interpret whether an effect matters.

But MCID is not always simple. Different patients may value changes differently. A small improvement in a severe disease may matter. A modest improvement with no side effects may matter. A modest improvement with severe toxicity may not.

Clinical significance is therefore contextual.

Let us also discuss benefit-risk balance.

A treatment effect cannot be interpreted without harms.

Suppose a drug reduces migraines by one day per month but causes severe nausea in 20% of patients. Is that worthwhile? Some patients may say yes; others may say no.

Suppose a cancer therapy extends survival by three months but causes serious toxicity. Is that clinically meaningful? It depends on disease severity, alternatives, quality of life, patient preferences, and cost.

This is why regulators and clinicians do not look only at efficacy. They evaluate benefit-risk.

Now we move to survival analysis.

Survival analysis is one of the most important areas of statistics for clinical trials, especially in oncology, cardiology, infectious disease, transplant medicine, and any field where time-to-event outcomes matter.

The word “survival” can be slightly misleading. Survival analysis is not only about death. It is about time until an event.

The event could be death, relapse, disease progression, hospitalization, stroke, heart attack, infection, device failure, treatment discontinuation, or recovery.

Why do we need special methods for time-to-event data?

Because we care not only whether an event happened, but when it happened.

Suppose two patients both die during a study. One dies after one month. The other dies after five years. Treating both simply as “death: yes” loses important information.

Also, not every patient will have an event during the study. Some are still alive or event-free when the study ends. Some are lost to follow-up. Some withdraw. Some enter the study later and therefore have shorter observation time.

This creates censoring.

Censoring means we have partial information. We know the patient did not experience the event up to a certain time, but we do not know what happened afterward.

For example, if a patient is followed for 18 months and remains alive at the end of the study, we know they survived at least 18 months. We do not know their eventual survival time. That observation is censored.

Survival analysis allows us to use this partial information instead of discarding it.

The most common visualization is the Kaplan-Meier curve.

A Kaplan-Meier curve shows the estimated probability of remaining event-free over time.

On the x-axis, we have time.

On the y-axis, we have the proportion surviving or event-free.

At time zero, the curve starts at 1, or 100%, because everyone is event-free at the beginning.

As events occur, the curve steps downward.

A steep decline means many events are occurring quickly.

A flatter curve means fewer events are occurring.

Censored observations are often marked with small tick marks, depending on the graph.

When comparing two treatment groups, we often plot two Kaplan-Meier curves: one for treatment, one for control.

If the treatment curve stays above the control curve, that generally suggests better event-free survival in the treatment group. But interpretation requires more than visual inspection.

We need to consider magnitude, confidence intervals, number at risk, follow-up duration, censoring patterns, and statistical tests.

The “number at risk” table below a Kaplan-Meier curve is very important. It tells us how many patients remain under observation at different time points. Late in the study, very few patients may remain at risk, making the tail of the curve unstable. People sometimes overinterpret differences at late time points when only a small number of patients are still being followed.

Now let us discuss median survival.

Median survival is the time by which 50% of patients have experienced the event.

If median overall survival is 12 months in the control group and 15 months in the treatment group, the median survival improvement is 3 months.

This is intuitive, but it does not capture the full survival curve. Two treatments could have the same median survival but different long-term survival. One might help a small subgroup survive much longer without changing the median much. Another might shift the entire curve modestly.

So median survival is useful but incomplete.

Now we come to hazard ratios.

Hazard ratios are among the most commonly reported statistics in clinical trials, especially oncology.

Conceptually, the hazard is the instantaneous event rate among patients who have not yet experienced the event. More simply, it is the rate at which events are occurring at a given time among those still at risk.

A hazard ratio compares hazards between groups.

If the hazard ratio is 0.75, people often say the treatment reduces the risk of the event by 25%. That is a rough shorthand, but we need to be careful.

A hazard ratio of 0.75 means that, at any given time, the event rate in the treatment group is estimated to be 25% lower than in the control group, assuming the model is appropriate.

It is not the same as saying that 25% fewer patients will have the event by the end of the study.

Hazard ratios are not risk ratios.

This is a common misunderstanding.

A risk ratio compares cumulative risk over a specified time period. For example, the risk of death by 12 months.

A hazard ratio compares event rates over time, often under a proportional hazards assumption.

The proportional hazards assumption means that the hazard ratio is roughly constant over time. But in real trials, hazards may not be proportional. The treatment effect may be stronger early, weaker later, delayed, or present only in a subgroup.

This is especially relevant in immuno-oncology, where survival curves may separate late, cross early, or show long-term tails.

So when interpreting hazard ratios, look at the Kaplan-Meier curves too.

A hazard ratio alone can hide important patterns.

For example, two trials might both report HR = 0.80. In one trial, the curves separate early and remain parallel. In another, they overlap for a year and then separate. In another, they cross. The clinical interpretation may differ substantially.

Also, a hazard ratio may sound impressive, but the absolute benefit may be small.

Suppose HR = 0.80, but median survival improves by only two weeks. Is that clinically meaningful? Maybe not, depending on toxicity and context.

Or suppose HR = 0.90 in a very common, severe condition with minimal side effects. A modest relative effect could still have large public health importance.

Again, context matters.

Let us practice reading a survival result.

A cancer trial reports:

Median overall survival: 14 months in treatment group versus 11 months in control group.

Hazard ratio: 0.78.

95% confidence interval: 0.65 to 0.94.

p = 0.01.

How should we interpret this?

The treatment appears to improve survival. The median survival gain is 3 months. The hazard ratio suggests a 22% lower event rate over time. The confidence interval excludes 1, suggesting statistical evidence of benefit. The p-value supports that the result is unlikely under the null. But we still need to ask: what were the toxicities? What was quality of life? What was the patient population? What alternative treatments exist? Were subsequent therapies balanced? Did the curves separate early or late? Was the endpoint overall survival or progression-free survival? Was the effect consistent across subgroups?

That is proper clinical interpretation.

Now let us briefly distinguish overall survival and progression-free survival, because these are common in oncology.

Overall survival is time until death from any cause. It is a hard, clinically meaningful endpoint.

Progression-free survival is time until disease progression or death. It can be measured earlier and is often used in oncology trials. But progression-free survival does not always translate into overall survival or quality-of-life benefit.

A drug may delay tumor progression on scans but not extend life. It may also cause toxicity. So progression-free survival can be meaningful, but interpretation depends on context.

Now we move to meta-analysis and evidence synthesis.

Individual trials are limited. Any single study may be affected by random error, design choices, unusual populations, operational issues, or chance findings.

Medicine rarely relies on one study alone. Instead, evidence accumulates.

A systematic review is a structured process for identifying, selecting, appraising, and synthesizing all relevant studies on a question.

The word “systematic” matters. A narrative review may cite studies selectively. A systematic review pre-specifies search strategies, inclusion criteria, outcomes, and methods to reduce reviewer bias.

A meta-analysis is the statistical combination of results from multiple studies.

The purpose is often to increase precision and estimate an overall effect.

If five small trials all point in the same direction, a meta-analysis may provide more confidence than any one trial alone.

But meta-analysis is not magic.

If the included studies are biased, heterogeneous, or poorly designed, the pooled estimate may be misleading.

There is a saying: garbage in, garbage out.

Now let us discuss forest plots.

A forest plot is the standard visualization for meta-analysis.

Each study is shown as a point estimate with a horizontal confidence interval. The point may be a square, often sized according to study weight. The horizontal line shows uncertainty.

There is usually a vertical line representing no effect. For risk ratios, odds ratios, or hazard ratios, no effect is 1. For mean differences, no effect is 0.

At the bottom, there is often a diamond representing the pooled estimate. The center of the diamond is the combined effect estimate, and the width shows the confidence interval.

When reading a forest plot, ask:

Do most studies point in the same direction?

Are confidence intervals wide or narrow?

Do studies cross the line of no effect?

How large is the pooled effect?

Is there heterogeneity?

Are results driven by one large study?

Are small studies showing larger effects than large studies?

Heterogeneity means that study results differ from each other beyond what we might expect from chance alone.

Heterogeneity can arise from differences in populations, interventions, comparators, endpoints, follow-up time, study quality, dose, setting, or analysis methods.

For example, a meta-analysis of “exercise interventions for depression” may include very different interventions: supervised aerobic exercise, home-based walking, resistance training, group classes, different durations, different patient populations, and different depression severity. Combining them may produce a pooled estimate, but what does it mean?

So before asking “what is the pooled result?” ask whether pooling makes sense.

There are statistical measures of heterogeneity, such as I-squared, but conceptually the question is: are these studies similar enough that a combined estimate is meaningful?

Now let us discuss publication bias.

Publication bias occurs when studies with positive or favorable results are more likely to be published than studies with negative or inconclusive results.

This can distort the evidence base.

Imagine ten trials are conducted. Two are positive by chance, eight are negative. If only the two positive trials are published, the literature appears positive. A meta-analysis of published studies may then conclude that the treatment works, even though the total evidence would not support that conclusion.

Trial registration, reporting requirements, and regulatory data access help reduce this problem, but it has not disappeared.

A funnel plot is one tool used to assess possible publication bias. In a simple intuition, if many small negative studies are missing, the plot may look asymmetric. But funnel plots are not definitive. Asymmetry can have other causes.

The key lesson is that evidence synthesis is not simply counting studies.

Do not say, “There are five studies showing benefit,” without asking how large they are, how good they are, whether negative studies are missing, whether endpoints are meaningful, and whether results are consistent.

Now let us connect all of today’s ideas.

Clinical research is about estimating effects under uncertainty.

A trial result should not be reduced to “positive” or “negative.”

Instead, every result should be interpreted through several lenses.

First: effect size.

How large is the benefit? Is it an absolute reduction, relative reduction, odds ratio, hazard ratio, mean difference, or something else? Does the size matter?

Second: precision.

How certain are we? Are confidence intervals narrow or wide? Does the interval include no effect? Does it include clinically important benefit or harm?

Third: clinical relevance.

Does the endpoint matter to patients? Is the effect large enough to change practice? What are the harms? What are the alternatives?

Fourth: statistical evidence.

What does the p-value indicate? Was the analysis pre-specified? Are there multiple comparisons? Was the study powered appropriately?

Fifth: totality of evidence.

Does this result fit with previous trials, observational evidence, biological plausibility, and meta-analyses? Is the finding replicated?

Sixth: context.

Is the disease severe? Are there existing treatments? What do patients value? What are regulators likely to require? What is feasible in practice?

Let us now examine a few common mistakes.

Mistake one: reporting only relative risk reduction.

A sponsor says, “Our drug reduced risk by 40%.”

Always ask: what was the absolute risk reduction?

If risk went from 10% to 6%, that is a 4 percentage point absolute reduction. If risk went from 0.10% to 0.06%, the absolute reduction is 0.04 percentage points. The relative reduction is the same; the clinical meaning is very different.

Mistake two: treating p > 0.05 as proof of no effect.

A non-significant result does not prove no effect. It may mean the study was underpowered or uncertain. Look at the confidence interval. If the interval includes clinically meaningful benefit and harm, the study is inconclusive, not negative in a strong sense.

Mistake three: treating p < 0.05 as proof of importance.

A statistically significant result may be tiny, biased, based on a weak endpoint, or clinically irrelevant. Again, look at effect size and endpoint.

Mistake four: confusing odds ratios with risk ratios.

This can exaggerate effects when outcomes are common. Be especially careful with observational studies and logistic regression.

Mistake five: interpreting hazard ratios as cumulative risk reductions.

A hazard ratio is not the same as saying “25% fewer patients died.” It refers to event rates over time under model assumptions. Always examine absolute survival, median survival, and Kaplan-Meier curves.

Mistake six: ignoring confidence intervals.

Point estimates are not enough. A result with a dramatic point estimate and a huge confidence interval is unstable.

Mistake seven: overinterpreting subgroup analyses.

Trials often report effects in subgroups: men versus women, older versus younger, biomarker-positive versus biomarker-negative, different regions, different severity levels.

Subgroup analyses can be useful, but they are risky. Many are underpowered. Many are exploratory. If enough subgroups are tested, some will appear positive by chance. Unless subgroup effects are pre-specified, biologically plausible, statistically supported, and replicated, they should be interpreted cautiously.

Mistake eight: ignoring missing data.

If many patients drop out, especially if dropout differs by group, results can be biased. For example, if patients experiencing side effects leave the treatment group and are not counted properly, efficacy may look better than it is.

Mistake nine: focusing only on the primary endpoint and ignoring safety.

A trial may meet its primary endpoint but reveal serious harms. The benefit-risk balance matters.

Mistake ten: assuming meta-analysis settles the question.

A meta-analysis can be strong, but only if the underlying evidence is strong, comparable, and complete.

Now let us simulate how a clinical research professional might interpret a trial abstract.

Imagine the abstract says:

“In a randomized, double-blind trial of 2,000 patients with moderate chronic kidney disease, Drug X significantly reduced the composite endpoint of hospitalization or biomarker worsening compared with placebo. The hazard ratio was 0.82, 95% CI 0.70 to 0.96, p = 0.01.”

At first glance, positive.

But now we ask:

What was the absolute event rate? A hazard ratio of 0.82 may correspond to a large or small absolute benefit depending on baseline risk.

What components drove the composite endpoint? Was it hospitalization, which matters clinically, or biomarker worsening, which may be less directly meaningful?

Was the biomarker validated?

What was the follow-up duration?

Were there safety issues?

Was quality of life measured?

Did the effect apply across disease severity?

Was the comparator standard of care?

Was the population representative?

This is the difference between reading a result and interpreting evidence.

Now let us do another example.

A Phase II oncology trial reports:

“Median progression-free survival improved from 4.0 to 5.5 months. HR = 0.72, p = 0.04. Overall survival data immature. Grade 3 or higher adverse events occurred in 45% of treatment patients and 20% of control patients.”

How should we think?

The trial suggests a delay in progression. The hazard ratio and p-value suggest statistical evidence. But the median improvement is 1.5 months. Overall survival is not yet known. Toxicity is substantially higher. Whether this is clinically meaningful depends on disease context, alternatives, symptoms, quality of life, and whether progression-free survival is accepted as meaningful in this setting.

So again: positive does not automatically mean practice-changing.

Now a third example.

A small rare disease trial reports:

“Twenty-four patients were randomized. The primary endpoint did not reach statistical significance: p = 0.09. The estimated treatment effect exceeded the pre-specified clinically meaningful threshold, but the 95% confidence interval was wide.”

How should we interpret this?

Not definitive. The trial is underpowered or uncertain. But it may provide evidence of a potentially meaningful effect. In rare diseases, regulators and clinicians may consider the totality of evidence, including natural history, biomarkers, patient outcomes, and unmet need. The result should not be dismissed simply because p = 0.09.

This is the level of nuance needed in real clinical development.

Now, before concluding, let us discuss why this matters for CROs and industry conversations.

In pharma and CRO settings, trial interpretation affects major decisions.

Should the sponsor proceed from Phase II to Phase III?

Was the endpoint strong enough for regulatory approval?

Is the effect size large enough for clinicians to adopt the product?

Is the safety profile acceptable?

Did the trial fail because the drug does not work, or because the design was flawed?

Was the study underpowered?

Did recruitment issues affect the population?

Did missing data or protocol deviations undermine interpretation?

Can the result support a label claim?

Will payers consider the benefit meaningful?

These questions are not purely statistical. They combine statistics, medicine, operations, regulation, and business. But statistical interpretation is at the center.

A CRO project manager may not calculate hazard ratios personally, but they need to understand why events matter, why database quality matters, why censoring matters, why endpoint definitions matter, and why missing data can threaten interpretation.

A clinical scientist needs to understand whether a result is medically meaningful.

A medical writer must accurately communicate effect sizes, uncertainty, and limitations.

A regulatory strategist must understand what evidence regulators will find persuasive.

A data manager must understand why clean endpoint data are essential.

So statistics is not just a technical layer at the end. It is connected to the entire trial.

Let us now synthesize today’s lecture in one central framework.

When you see a clinical trial result, ask four core questions.

First: how big is the effect?

Look at absolute differences, relative measures, NNT, mean differences, hazard ratios, or odds ratios. Understand what metric is being used.

Second: how uncertain is the effect?

Look at confidence intervals, sample size, number of events, missing data, and whether the study was powered.

Third: does the effect matter?

Look at clinical endpoints, patient relevance, MCID, safety, burden, alternatives, and disease severity.

Fourth: does this fit with the total evidence?

Look at previous trials, systematic reviews, meta-analyses, consistency, biological plausibility, and publication bias.

If you do that, you will avoid many of the most common mistakes.

Let me now return to the theme:

**Statistical significance is not the same as clinical significance.**

Statistical significance can tell us that the data are unlikely under a null model. It does not tell us that the effect is large, important, patient-relevant, safe, generalizable, or worth paying for.

Clinical significance asks whether the result matters in the real world.

The best evidence combines both: a meaningful effect, measured on a meaningful endpoint, estimated precisely, with acceptable safety, in a relevant population, supported by the broader evidence base.

That is what clinical research is aiming for.

In the next lecture, we will shift from interpreting trial results to running trials in practice. We will discuss protocol development, site selection, recruitment, data collection, electronic data capture, monitoring, Good Clinical Practice, CRO operations, and common reasons trials fail.

Today was about understanding evidence. Next time will be about the operational machinery that produces that evidence.
