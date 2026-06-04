# Lecture 2 — Designing Clinical Trials

**Duration:** 60 minutes

**Theme:** _Trial design determines whether evidence is trustworthy._

---

# Learning Objectives

By the end of this lecture, students should be able to:

1. Describe the core components of a clinical trial.
2. Explain how randomization helps establish causality.
3. Understand the purpose and limitations of blinding.
4. Distinguish between different types of endpoints.
5. Explain the logic of Phase I-IV clinical development.
6. Evaluate tradeoffs between internal and external validity.
7. Critically assess whether a trial design is likely to produce reliable evidence.

---

# Lecture Overview

| Section | Topic                          | Time   |
| ------- | ------------------------------ | ------ |
| 1       | Anatomy of a Clinical Trial    | 10 min |
| 2       | Randomization                  | 12 min |
| 3       | Blinding                       | 8 min  |
| 4       | Endpoints and Outcome Measures | 12 min |
| 5       | Clinical Development Phases    | 10 min |
| 6       | Internal vs External Validity  | 8 min  |

---

# Part 1 — Anatomy of a Clinical Trial

## Learning Goal

Students should understand the fundamental structure shared by nearly all clinical trials.

---

## Starting Question

Imagine a company claims:

> "Our new treatment reduces chronic back pain."

Before evaluating the claim, what questions should we ask?

Students typically ask:

- Which patients?
- Compared to what?
- How was pain measured?
- For how long?

These questions form the basis of trial design.

---

## The PICO Framework

Most clinical questions can be structured using PICO:

### P — Population

Who is being studied?

Examples:

- adults with hypertension
- children with asthma
- patients with metastatic breast cancer

Important considerations:

- age
- disease severity
- comorbidities
- prior treatments

---

### I — Intervention

What is being tested?

Examples:

- drug
- medical device
- surgical procedure
- behavioral intervention
- AI software

---

### C — Comparator

Compared against what?

Examples:

- placebo
- standard of care
- competing drug
- no treatment

A treatment is rarely judged in isolation.

It is judged relative to alternatives.

---

### O — Outcome

What is being measured?

Examples:

- survival
- symptom reduction
- blood pressure
- quality of life

---

### T — Time

Over what period?

Examples:

- 30 days
- 6 months
- 5 years

Many interventions show short-term benefits but fail long-term.

---

## Example Trial

Population:

Adults with Type 2 Diabetes

Intervention:

New diabetes drug

Comparator:

Standard therapy

Outcome:

HbA1c reduction

Time:

12 months

Students should be able to identify all PICO components.

---

# Part 2 — Randomization

## Learning Goal

Students should understand why randomization is the foundation of modern clinical trials.

---

## The Core Problem

Suppose physicians choose who receives a new treatment.

What might happen?

Doctors may preferentially treat:

- younger patients
- healthier patients
- more severe patients

Groups become systematically different.

This creates confounding.

---

## What Randomization Does

Randomization assigns treatments by chance.

Example:

200 patients

Randomly allocate:

- 100 treatment
- 100 control

The goal:

Create groups that differ only by treatment assignment.

---

## Important Clarification

Randomization does not guarantee identical groups.

Instead:

It makes systematic differences unlikely.

---

## Why Randomization Is Powerful

It balances:

### Known confounders

Examples:

- age
- sex
- disease severity

---

### Unknown confounders

Examples:

- genetics
- lifestyle factors
- unmeasured risk factors

This is why RCTs are considered the strongest design for causal inference.

---

## Types of Randomization

Students do not need implementation details.

They should understand why different methods exist.

---

### Simple Randomization

Like repeated coin flips.

Advantages:

- simple

Disadvantages:

- imbalance can occur in small studies

---

### Block Randomization

Ensures groups remain balanced throughout enrollment.

Useful for smaller studies.

---

### Stratified Randomization

Balances important characteristics.

Example:

Randomize separately within:

- male/female
- disease stage

Helps avoid imbalance in critical variables.

---

## Discussion Question

Would you rather compare:

- 100 randomly assigned patients

or

- 100 patients chosen by physicians

Why?

This discussion reinforces the logic of randomization.

---

# Part 3 — Blinding

## Learning Goal

Students should understand how expectations influence outcomes.

---

## Why Blinding Exists

Humans are biased.

Patients are biased.

Researchers are biased.

Clinicians are biased.

Even when acting honestly.

---

## Example

Pain study.

Patients know they received the new drug.

They may:

- report lower pain
- expect improvement

Even without biological benefit.

---

## Open-Label Trial

Everyone knows treatment assignments.

Advantages:

- simple
- cheaper

Disadvantages:

- more vulnerable to bias

---

## Single-Blind Trial

Patients do not know assignment.

Investigators may know.

Reduces placebo-related bias.

---

## Double-Blind Trial

Neither patients nor investigators know assignments.

Considered ideal when feasible.

---

## Examples of Bias Without Blinding

### Patient Bias

Patients report better outcomes.

---

### Investigator Bias

Clinicians evaluate outcomes differently.

---

### Treatment Behavior Bias

Healthcare providers treat groups differently.

---

## Important Limitation

Some interventions cannot realistically be blinded.

Examples:

- surgery
- rehabilitation programs
- lifestyle interventions

Researchers must then use other methods to reduce bias.

---

# Part 4 — Endpoints and Outcome Measures

## Learning Goal

Students should understand what clinical trials actually measure.

---

## What Is an Endpoint?

An endpoint is the outcome used to evaluate treatment success.

Everything in a trial ultimately depends on endpoint selection.

---

## Clinical Endpoints

Directly relevant to patients.

Examples:

- death
- stroke
- heart attack
- hospitalization

These are generally preferred.

---

## Example

Cardiovascular trial.

Strong endpoint:

Reduction in heart attacks.

Weak endpoint:

Reduction in cholesterol alone.

---

## Patient-Reported Outcomes (PROs)

Information reported directly by patients.

Examples:

- pain scores
- fatigue
- quality of life

Increasingly important in modern research.

---

## Biomarkers

Objective biological measurements.

Examples:

- blood pressure
- HbA1c
- tumor size
- inflammatory markers

Advantages:

- measurable
- often available sooner

---

## Surrogate Endpoints

A biomarker used as a substitute for a clinical outcome.

Examples:

| Surrogate       | Intended Clinical Outcome |
| --------------- | ------------------------- |
| Blood pressure  | Stroke                    |
| LDL cholesterol | Heart attack              |
| Tumor shrinkage | Survival                  |

---

## Why Surrogates Can Be Dangerous

The central lesson:

Improving a surrogate does not guarantee improving patient outcomes.

---

## Famous Example

Certain anti-arrhythmic drugs:

- improved ECG measurements
- reduced abnormal heart rhythms

Yet later trials showed:

- increased mortality

The surrogate improved.

Patients did worse.

---

## Endpoint Hierarchy

Generally:

Clinical outcomes > validated surrogates > exploratory biomarkers

Students should understand why.

---

# Part 5 — Clinical Development Phases

## Learning Goal

Students should understand how evidence accumulates during development.

---

## Why Multiple Phases Exist

Drug development is fundamentally a process of risk reduction.

At the beginning:

- high uncertainty
- limited data

Over time:

- uncertainty decreases
- investment increases

---

## Phase I

### Primary Goal

Safety

---

Typical participants:

- healthy volunteers

Questions:

- Is the drug tolerated?
- What dose is safe?
- What are side effects?

Typical size:

20–100 participants

---

## Phase II

### Primary Goal

Early efficacy

Dose selection

---

Questions:

- Does it appear to work?
- Which dose should be used?

Typical size:

100–300 participants

---

## Phase III

### Primary Goal

Confirmatory evidence

---

Questions:

- Does the treatment outperform current alternatives?
- Is the benefit clinically meaningful?

Typical size:

Hundreds to thousands of participants.

Largest and most expensive phase.

---

## Regulatory Submission

Results submitted to:

- FDA
- EMA
- Swissmedic

Approval decisions occur here.

---

## Phase IV

Conducted after approval.

Focus:

- long-term safety
- rare adverse events
- real-world effectiveness

---

## Key Concept

Every phase answers a different question.

| Phase | Main Question                   |
| ----- | ------------------------------- |
| I     | Is it safe?                     |
| II    | Does it appear to work?         |
| III   | Does it truly work?             |
| IV    | What happens in the real world? |

---

# Part 6 — Internal vs External Validity

## Learning Goal

Students should understand one of the most important tradeoffs in clinical research.

---

## Internal Validity

Question:

> Is the observed result actually caused by the intervention?

Threats include:

- confounding
- bias
- poor randomization
- protocol deviations

RCTs are designed to maximize internal validity.

---

## External Validity

Question:

> Will the result generalize to real-world patients?

---

## Example

Trial population:

- age 40–60
- few comorbidities
- highly adherent patients

Real-world population:

- age 75+
- multiple chronic conditions
- inconsistent adherence

The trial result may not fully generalize.

---

## Typical Tradeoff

To improve internal validity:

Researchers often use strict inclusion criteria.

This reduces heterogeneity.

But:

It may reduce external validity.

---

## Explanatory vs Pragmatic Trials

### Explanatory Trials

Ask:

> Can the treatment work under ideal conditions?

Focus on internal validity.

---

### Pragmatic Trials

Ask:

> Does the treatment work in routine practice?

Focus on external validity.

---

## Final Discussion

Ask students:

Would you prefer a trial that is:

- extremely rigorous
- but includes only ideal patients

or

- less controlled
- but resembles everyday clinical practice

There is no perfect answer.

Modern clinical research constantly balances these competing goals.

---

# End-of-Lecture Synthesis

By the end of this lecture, students should understand that clinical trial design is fundamentally an exercise in answering a deceptively simple question:

> Did the intervention cause the observed outcome?

To answer that question reliably, researchers must carefully define:

- who is studied (population)
- what is tested (intervention)
- what it is compared against (comparator)
- what outcomes matter (endpoints)
- how bias is controlled (randomization and blinding)
- how evidence accumulates (Phase I-IV)
- whether results are both trustworthy and generalizable (internal vs external validity)

This lecture provides the conceptual toolkit needed to critically evaluate almost any clinical trial discussed in pharma, biotech, academia, healthcare, or CRO settings.
