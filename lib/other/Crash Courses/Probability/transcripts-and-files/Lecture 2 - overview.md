# Lecture 2 — Conditional Probability, Bayes, and Dependence

**Theme:** AI is mostly about learning relationships between variables: **what becomes more likely given something else?**

**Duration:** 60 minutes
**Level:** Rusty high-school probability, after Lecture 1
**Style:** Mathematical but applied
**Core AI connection:** Most AI prediction problems can be understood as estimating conditional probability distributions.

---

# Lecture 2 Overview

## Central Message

In Lecture 1, students learned that AI systems often represent uncertainty using probability distributions.

Lecture 2 adds the most important upgrade:

[
P(y)
]

becomes:

[
P(y \mid x)
]

Instead of asking:

> How likely is this outcome?

we now ask:

> How likely is this outcome **given some information**?

This is the core of prediction.

Examples:

[
P(\text{disease} \mid \text{symptoms})
]

[
P(\text{spam} \mid \text{email text})
]

[
P(\text{next token} \mid \text{previous tokens})
]

[
P(\text{next state} \mid \text{current state, action})
]

[
P(\text{clean image} \mid \text{noisy image})
]

The whole lecture should make students feel that conditional probability is not an abstract formula. It is the basic structure of AI.

---

# Lecture Structure

| Part |                                       Topic |   Time |
| ---- | ------------------------------------------: | -----: |
| 1    | From probability to conditional probability | 10 min |
| 2    |                           Joint probability |  8 min |
| 3    |    Marginal probability and marginalization |  8 min |
| 4    |                 Independence and dependence | 10 min |
| 5    |                              Bayes’ theorem | 14 min |
| 6    |       Prediction as conditional probability |  8 min |
| 7    |             Summary and bridge to Lecture 3 |  2 min |

---

# Learning Goals

By the end of the lecture, students should understand:

- what conditional probability means,
- why (P(y \mid x)) is the mathematical form of prediction,
- what joint probability means,
- what marginal probability means,
- how marginalization works,
- what independence means,
- why dependence is what makes learning possible,
- how Bayes’ theorem updates beliefs using evidence,
- why LLMs, classifiers, RL systems, and diffusion models all rely on conditional structure.

---

# Part 1 — From Probability to Conditional Probability

**Time:** 10 minutes

## 1.1 Recap from Lecture 1

Start with the basic idea from Lecture 1:

A probability distribution assigns probabilities to possible outcomes.

Example:

| Diagnosis | Probability |
| --------- | ----------: |
| flu       |        0.40 |
| allergy   |        0.30 |
| COVID     |        0.20 |
| other     |        0.10 |

This is an unconditional distribution:

[
P(Y)
]

It tells us how likely each diagnosis is **before** seeing any particular patient information.

But AI almost never works only with unconditional probabilities.

Usually, we have information.

For example:

- symptoms,
- text,
- image pixels,
- previous tokens,
- patient history,
- current state,
- noisy input.

So we need conditional probability.

---

## 1.2 Conditional Probability

Introduce:

[
P(A \mid B)
]

Read this as:

> The probability of (A), given (B).

Or:

> How likely is (A), once we know (B)?

Examples:

[
P(\text{disease} \mid \text{symptoms})
]

[
P(\text{spam} \mid \text{words in email})
]

[
P(\text{cat} \mid \text{image})
]

[
P(\text{next token} \mid \text{previous tokens})
]

[
P(\text{successful action} \mid \text{state})
]

The vertical bar (\mid) means **given**.

---

## 1.3 Intuition: Information Changes Probabilities

Use a simple example.

Before seeing symptoms:

| Disease  | Probability |
| -------- | ----------: |
| flu      |        0.10 |
| allergy  |        0.20 |
| migraine |        0.05 |
| other    |        0.65 |

After learning the patient has fever and body aches:

| Disease  | Probability given symptoms |
| -------- | -------------------------: |
| flu      |                       0.55 |
| allergy  |                       0.05 |
| migraine |                       0.03 |
| other    |                       0.37 |

So:

[
P(\text{flu}) = 0.10
]

but:

[
P(\text{flu} \mid \text{fever and body aches}) = 0.55
]

The event “fever and body aches” changes the probability.

Core teaching sentence:

> Conditional probability tells us how probabilities change when we receive information.

---

## 1.4 AI Examples

### Image Classification

[
P(Y = \text{cat} \mid X = \text{image})
]

The model estimates the probability of a label given the image.

Without the image, “cat” may not be especially likely.

Given the image, “cat” may become very likely.

---

### LLMs

Given:

> The capital of France is

The model estimates:

[
P(X_t = \text{Paris} \mid X_1, X_2, ..., X_{t-1})
]

In words:

> The probability that the next token is “Paris,” given the previous tokens.

This is the central probabilistic structure behind next-token prediction.

---

### Reinforcement Learning

In RL, the next state depends on the current state and action:

[
P(S_{t+1} \mid S_t, A_t)
]

In words:

> Given where I am and what I do, what state might I end up in?

This conditional structure is essential because actions affect future probabilities.

---

### Diffusion Models

A denoising model learns something like:

[
P(\text{cleaner image} \mid \text{noisy image})
]

or it predicts the noise that was added, given the noisy input.

Core idea:

> The model learns how likely clean structures are, given noisy observations.

---

## 1.5 Mini-Exercise

A model estimates the probability that an email is spam.

Before reading the email:

[
P(\text{spam}) = 0.20
]

After seeing the phrase “you have won a prize”:

[
P(\text{spam} \mid \text{phrase}) = 0.85
]

Ask:

1. Which probability is unconditional?
2. Which probability is conditional?
3. What information changed the probability?
4. Why is this useful for AI?

Expected answers:

1. (P(\text{spam}))
2. (P(\text{spam} \mid \text{phrase}))
3. The phrase “you have won a prize”
4. Because prediction improves when the model uses input information.

---

# Part 2 — Joint Probability

**Time:** 8 minutes

## 2.1 What Is Joint Probability?

Introduce:

[
P(A, B)
]

or equivalently:

[
P(A \cap B)
]

Meaning:

> The probability that (A) and (B) both happen.

Examples:

[
P(\text{fever}, \text{infection})
]

[
P(\text{word is “bank”}, \text{context is financial})
]

[
P(\text{state}=s, \text{action}=a)
]

[
P(\text{image contains dog}, \text{image contains leash})
]

Joint probability is about variables appearing together.

---

## 2.2 Simple Table Example

Use a small medical-style example.

Suppose we have 100 patients.

|          | Infection | No infection | Total |
| -------- | --------: | -----------: | ----: |
| Fever    |        30 |           10 |    40 |
| No fever |         5 |           55 |    60 |
| Total    |        35 |           65 |   100 |

From this table:

[
P(\text{fever}) = \frac{40}{100} = 0.40
]

[
P(\text{infection}) = \frac{35}{100} = 0.35
]

[
P(\text{fever}, \text{infection}) = \frac{30}{100} = 0.30
]

The joint probability is the probability that both are true.

---

## 2.3 Joint Probability Connects to Conditional Probability

From the same table:

[
P(\text{infection} \mid \text{fever})
=====================================

\frac{P(\text{infection}, \text{fever})}{P(\text{fever})}
]

Using numbers:

[
P(\text{infection} \mid \text{fever})
=====================================

# \frac{0.30}{0.40}

0.75
]

So among patients with fever, 75% have infection.

This gives the definition:

[
P(A \mid B) = \frac{P(A,B)}{P(B)}
]

provided:

[
P(B) > 0
]

Core intuition:

> Conditional probability means restricting attention to the cases where (B) is true, then asking how often (A) is also true.

---

## 2.4 AI Examples of Joint Probability

### Text

[
P(\text{token = “bank”}, \text{context = finance})
]

This is high in financial documents.

[
P(\text{token = “bank”}, \text{context = river})
]

This is high in geography or nature contexts.

The same word has different meaning depending on what appears with it.

---

### Vision

[
P(\text{dog}, \text{leash})
]

[
P(\text{road}, \text{car})
]

[
P(\text{table}, \text{chair})
]

Images contain statistical relationships between objects.

Computer vision models learn these relationships.

---

### RL

[
P(S_t=s, A_t=a)
]

This means:

> How often is the agent in state (s) and taking action (a)?

This matters because RL algorithms learn from state-action-reward patterns.

---

## 2.5 Mini-Exercise

Using the table:

|          | Infection | No infection | Total |
| -------- | --------: | -----------: | ----: |
| Fever    |        30 |           10 |    40 |
| No fever |         5 |           55 |    60 |
| Total    |        35 |           65 |   100 |

Ask:

1. What is (P(\text{fever}, \text{infection}))?
2. What is (P(\text{infection}))?
3. What is (P(\text{infection} \mid \text{fever}))?
4. Is infection more likely when fever is present?

Expected answers:

1. (0.30)
2. (0.35)
3. (0.30 / 0.40 = 0.75)
4. Yes, because (0.75 > 0.35)

---

# Part 3 — Marginal Probability and Marginalization

**Time:** 8 minutes

## 3.1 What Is Marginal Probability?

A marginal probability is the probability of one variable without focusing on the others.

In the table above:

[
P(\text{infection}) = 0.35
]

This is a marginal probability.

It ignores whether the patient has fever or not.

---

## 3.2 Why “Marginal”?

In old probability tables, the totals were written in the margins of the table.

Example:

|          | Infection | No infection | Total |
| -------- | --------: | -----------: | ----: |
| Fever    |        30 |           10 |    40 |
| No fever |         5 |           55 |    60 |
| Total    |        35 |           65 |   100 |

The totals at the margins give marginal counts.

So:

[
P(\text{infection}) = \frac{35}{100}
]

[
P(\text{fever}) = \frac{40}{100}
]

---

## 3.3 Marginalization

Marginalization means summing over the possible values of another variable.

Formula:

[
P(A) = \sum_B P(A,B)
]

Example:

[
P(\text{infection})
===================

P(\text{infection}, \text{fever})

- P(\text{infection}, \text{no fever})
  ]

Using the table:

[
P(\text{infection}) = 0.30 + 0.05 = 0.35
]

Interpretation:

> To get the probability of infection, add up all the ways infection can occur.

It can occur with fever or without fever.

---

## 3.4 Why Marginalization Matters in AI

Marginalization appears whenever there are hidden or alternative possibilities.

### Hidden Causes

A symptom may be caused by several diseases.

[
P(\text{symptom}) =
\sum_{\text{disease}} P(\text{symptom}, \text{disease})
]

To understand the symptom overall, we sum over possible diseases.

---

### Latent Variables

Suppose (Z) is a hidden topic of a document.

[
P(\text{word}) = \sum_Z P(\text{word}, Z)
]

A word’s probability depends on possible hidden topics.

This idea appears in latent variable models.

---

### Future States in RL

An action may lead to many possible next states.

[
P(\text{reward} \mid s,a)
]

may require considering possible next states:

[
\sum_{s'} P(s' \mid s,a) \cdot \text{reward}(s')
]

This is the beginning of expected return reasoning.

---

### Diffusion Models

A generated image can be influenced by many hidden noise paths or latent causes.

Even if students do not need the formal math yet, they should understand the intuition:

> Sometimes the visible outcome depends on hidden variables, and probability lets us sum over them.

---

## 3.5 Mini-Exercise

Given:

|                           | Click | No click | Total |
| ------------------------- | ----: | -------: | ----: |
| User likes sports         |    20 |       30 |    50 |
| User does not like sports |     5 |       45 |    50 |
| Total                     |    25 |       75 |   100 |

Ask:

1. What is (P(\text{click}))?
2. What is (P(\text{click}, \text{likes sports}))?
3. What is (P(\text{click}, \text{does not like sports}))?
4. Show marginalization for (P(\text{click})).

Expected answers:

1. (25/100 = 0.25)
2. (20/100 = 0.20)
3. (5/100 = 0.05)
4. (P(\text{click}) = 0.20 + 0.05 = 0.25)

---

# Part 4 — Independence and Dependence

**Time:** 10 minutes

## 4.1 What Is Independence?

Two events (A) and (B) are independent if knowing (B) does not change the probability of (A).

Formula:

[
P(A \mid B) = P(A)
]

Equivalent formula:

[
P(A,B) = P(A)P(B)
]

Meaning:

> (A) and (B) are statistically unrelated.

Example:

If a fair coin is tossed twice:

[
P(\text{second toss heads} \mid \text{first toss heads})
========================================================

# P(\text{second toss heads})

0.5
]

The first toss does not tell us anything about the second toss.

---

## 4.2 Dependence

Events are dependent if knowing one changes the probability of the other.

[
P(A \mid B) \neq P(A)
]

Example:

[
P(\text{infection}) = 0.35
]

but:

[
P(\text{infection} \mid \text{fever}) = 0.75
]

So fever and infection are dependent.

The presence of fever changes the probability of infection.

---

## 4.3 Why Dependence Matters in AI

This is one of the most important conceptual points in the whole course:

> If variables were independent, learning would often be useless.

Why?

Because if:

[
P(Y \mid X) = P(Y)
]

then knowing (X) gives no information about (Y).

In supervised learning:

- (X) = input,
- (Y) = output.

If input and output were independent, prediction would be impossible.

The model learns because (X) and (Y) are dependent.

---

## 4.4 Examples of Dependence in AI

### Language

Words in a sentence are strongly dependent.

After:

> The capital of France is

the token “Paris” becomes much more likely.

So:

[
P(\text{Paris} \mid \text{“The capital of France is”})
]

is much higher than:

[
P(\text{Paris})
]

Language modeling works because tokens depend on previous tokens.

---

### Images

Pixels are not independent.

If one pixel is blue, neighboring pixels are more likely to also be blue.

If an image contains a wheel, it is more likely to contain a car or bicycle.

Vision models work because images contain dependence structures.

---

### Medicine

Symptoms are not independent of diseases.

If a patient has fever, cough, and fatigue, some diagnoses become more likely.

Medical prediction works because clinical observations are related to underlying conditions.

---

### Reinforcement Learning

Actions affect future states.

[
P(S_{t+1} \mid S_t, A_t)
]

depends on (A_t).

If actions and future states were independent, the agent could not control anything.

---

### Diffusion Models

Noisy images still contain statistical traces of the clean image.

The denoising model works because:

[
P(\text{clean structure} \mid \text{noisy image})
]

is not random. There is dependence between noisy inputs and clean data.

---

## 4.5 Conditional Independence

Briefly introduce conditional independence because it is useful later, but do not overdo it.

Sometimes two variables are not independent overall, but become independent once we know a third variable.

Example:

Ice cream sales and drowning accidents may be correlated.

But both are influenced by temperature or season.

Given the weather, the direct relationship may weaken.

In notation:

[
A \perp B \mid C
]

Meaning:

> (A) and (B) are independent given (C).

This matters for:

- graphical models,
- causal reasoning,
- confounding,
- medical diagnosis,
- representation learning.

For this crash course, the main takeaway is enough:

> Dependence can be real, indirect, or explained by another variable.

---

## 4.6 Mini-Exercise

Ask:

Which pairs are likely dependent?

1. Next token and previous tokens.
2. First fair coin toss and second fair coin toss.
3. Fever and infection.
4. A robot’s action and its next state.
5. Shoe size and next token in a sentence.

Expected answers:

1. Dependent.
2. Independent.
3. Dependent.
4. Dependent.
5. Probably independent, unless there is some unusual context.

---

# Part 5 — Bayes’ Theorem

**Time:** 14 minutes

## 5.1 Why Bayes’ Theorem Matters

Bayes’ theorem is the mathematical rule for updating beliefs when we observe evidence.

It answers questions like:

[
P(\text{hypothesis} \mid \text{data})
]

Examples:

[
P(\text{disease} \mid \text{positive test})
]

[
P(\text{spam} \mid \text{email words})
]

[
P(\text{class} \mid \text{image})
]

[
P(\text{hidden state} \mid \text{observation})
]

Core sentence:

> Bayes’ theorem turns evidence into updated belief.

---

## 5.2 The Formula

Introduce:

[
P(H \mid D) = \frac{P(D \mid H)P(H)}{P(D)}
]

Where:

| Term          | Meaning               |
| ------------- | --------------------- |
| (H)           | hypothesis            |
| (D)           | data or evidence      |
| (P(H))        | prior probability     |
| (P(D \mid H)) | likelihood            |
| (P(H \mid D)) | posterior probability |
| (P(D))        | evidence / normalizer |

Explain in words:

[
\text{posterior}
================

\frac{\text{likelihood} \times \text{prior}}{\text{evidence}}
]

---

## 5.3 The Terms Intuitively

### Prior

[
P(H)
]

What we believed before seeing the new evidence.

Example:

[
P(\text{disease}) = 0.01
]

The disease is rare.

---

### Likelihood

[
P(D \mid H)
]

How likely the data is if the hypothesis is true.

Example:

[
P(\text{positive test} \mid \text{disease}) = 0.95
]

If the person has the disease, the test is likely positive.

Important warning:

[
P(D \mid H)
]

is not the same as:

[
P(H \mid D)
]

That is one of the most common mistakes in probability.

---

### Posterior

[
P(H \mid D)
]

What we believe after seeing the evidence.

Example:

[
P(\text{disease} \mid \text{positive test})
]

This is usually what we actually want.

---

### Evidence / Normalizer

[
P(D)
]

How likely the data is overall.

It includes all possible ways the data could occur.

For a medical test:

[
P(\text{positive test})
]

includes:

- true positives,
- false positives.

This is what makes the probabilities sum correctly.

---

## 5.4 Medical Test Example

Use a small numerical example.

Suppose:

[
P(\text{disease}) = 0.01
]

[
P(\text{positive} \mid \text{disease}) = 0.95
]

[
P(\text{positive} \mid \text{no disease}) = 0.05
]

Question:

> If someone tests positive, what is (P(\text{disease} \mid \text{positive}))?

First compute:

[
P(\text{positive})
==================

P(\text{positive} \mid \text{disease})P(\text{disease})

- P(\text{positive} \mid \text{no disease})P(\text{no disease})
  ]

[
P(\text{positive})
==================

0.95 \cdot 0.01

- 0.05 \cdot 0.99
  ]

[
P(\text{positive})
==================

0.0095 + 0.0495 = 0.059
]

Now:

[
P(\text{disease} \mid \text{positive})
======================================

\frac{0.95 \cdot 0.01}{0.059}
]

[
P(\text{disease} \mid \text{positive})
\approx 0.161
]

So the probability is about 16.1%.

This surprises many students.

Why is it not 95%?

Because the disease is rare, and false positives are relatively common compared with the base rate.

Important concept:

> Base rates matter.

---

## 5.5 Bayes’ Theorem as Reweighting

A useful non-technical explanation:

Bayes’ theorem does this:

1. Start with prior possibilities.
2. Ask how well each possibility explains the data.
3. Increase the probability of hypotheses that make the data likely.
4. Normalize so the probabilities sum to 1.

This is close to how many people intuitively think about evidence.

Example:

If we hear barking, the hypothesis “there is a dog nearby” becomes more likely.

Why?

Because barking is more likely if there is a dog than if there is not.

---

## 5.6 Bayes in AI

### Classification

A classifier often estimates:

[
P(Y \mid X)
]

That is a posterior-like quantity:

> probability of class given input.

Examples:

[
P(\text{cat} \mid \text{image})
]

[
P(\text{spam} \mid \text{email})
]

[
P(\text{disease} \mid \text{patient data})
]

---

### Naive Bayes

Briefly mention as historical/simple ML model.

Naive Bayes uses Bayes’ theorem plus a strong independence assumption.

For text classification:

[
P(\text{spam} \mid \text{words})
]

The “naive” assumption is that words are conditionally independent given the class.

This assumption is not fully true, but it can work surprisingly well.

Teaching point:

> Sometimes AI uses simplifying probability assumptions that are mathematically false but practically useful.

---

### Bayesian Neural Networks and Uncertainty

Briefly mention, but do not go deep.

Bayesian ML tries to represent uncertainty over model parameters or predictions.

Instead of only finding one best model, it asks:

> Which models are plausible given the data?

This is useful for:

- uncertainty estimation,
- small data,
- scientific modeling,
- medical decision support.

---

### LLMs and Bayes

Do not claim LLMs literally perform Bayesian inference in a strict classical sense.

Say:

> LLMs are not usually implemented as explicit Bayesian models, but many tasks they perform can be described using Bayesian language: given evidence in the prompt, update which continuations are plausible.

Example:

Before context:

[
P(\text{“apple”})
]

After context:

> She opened her laptop and clicked on the Apple logo.

[
P(\text{Apple as company} \mid \text{context})
]

The context changes the likely interpretation.

---

## 5.7 Common Mistake: Reversing the Conditional

Make this explicit.

These are different:

[
P(\text{positive test} \mid \text{disease})
]

and:

[
P(\text{disease} \mid \text{positive test})
]

The first asks:

> If the person has the disease, how likely is a positive test?

The second asks:

> If the test is positive, how likely is the disease?

These are often very different.

AI analogy:

[
P(\text{word “bank”} \mid \text{finance context})
]

is not the same as:

[
P(\text{finance context} \mid \text{word “bank”})
]

The first asks:

> In finance contexts, how likely is the word “bank”?

The second asks:

> Given the word “bank,” how likely is the context financial?

---

## 5.8 Mini-Exercise

Suppose:

[
P(\text{disease}) = 0.02
]

[
P(\text{positive} \mid \text{disease}) = 0.90
]

[
P(\text{positive} \mid \text{no disease}) = 0.10
]

Ask students to compute:

[
P(\text{disease} \mid \text{positive})
]

Solution:

[
P(\text{positive})
==================

0.90 \cdot 0.02 + 0.10 \cdot 0.98
]

[
= 0.018 + 0.098 = 0.116
]

[
P(\text{disease} \mid \text{positive})
======================================

# \frac{0.90 \cdot 0.02}{0.116}

\frac{0.018}{0.116}
\approx 0.155
]

So:

[
P(\text{disease} \mid \text{positive}) \approx 15.5%
]

Teaching point:

> Even a fairly accurate test can have a surprisingly low posterior probability when the condition is rare.

---

# Part 6 — Prediction as Conditional Probability

**Time:** 8 minutes

## 6.1 The Core AI Formula

Write this prominently:

[
P(y \mid x)
]

This is one of the most important formulas in machine learning.

Meaning:

> Given input (x), what is the probability distribution over output (y)?

Examples:

| Input (x)        | Output (y)        | Conditional Distribution                        |
| ---------------- | ----------------- | ----------------------------------------------- |
| image            | class             | (P(\text{class} \mid \text{image}))             |
| text prefix      | next token        | (P(\text{next token} \mid \text{text prefix}))  |
| symptoms         | disease           | (P(\text{disease} \mid \text{symptoms}))        |
| user profile     | click             | (P(\text{click} \mid \text{user, item}))        |
| state and action | next state        | (P(s' \mid s,a))                                |
| noisy image      | clean image/noise | (P(\text{clean/noise} \mid \text{noisy image})) |

---

## 6.2 Supervised Learning

In supervised learning, we have examples:

[
(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)
]

The goal is to learn a function/model that estimates:

[
P(y \mid x)
]

For classification:

[
P(Y = k \mid X=x)
]

For regression, the model may estimate:

[
P(Y \mid X=x)
]

or simply an expected value:

[
\mathbb{E}[Y \mid X=x]
]

Important distinction:

- classification usually predicts a distribution over classes,
- regression often predicts a number,
- probabilistic regression predicts a distribution over possible numbers.

---

## 6.3 LLMs as Conditional Probability Models

An LLM models:

[
P(X_t \mid X_1, X_2, ..., X_{t-1})
]

Text generation is repeated conditional prediction:

[
P(X_1)
]

[
P(X_2 \mid X_1)
]

[
P(X_3 \mid X_1, X_2)
]

[
P(X_4 \mid X_1, X_2, X_3)
]

and so on.

A full sentence probability can be decomposed as:

[
P(X_1, X_2, ..., X_T)
=====================

\prod*{t=1}^{T}
P(X_t \mid X_1, ..., X*{t-1})
]

Do not derive this deeply yet; just use it as a preview.

Teaching point:

> A transformer language model generates text by repeatedly estimating the probability of the next token conditioned on the previous tokens.

---

## 6.4 RL as Conditional Probability Plus Expected Reward

In RL, the environment is described by conditional probabilities:

[
P(S_{t+1} \mid S_t, A_t)
]

The policy may also be conditional:

[
\pi(A_t \mid S_t)
]

Meaning:

> Given the current state, what probability does the agent assign to each action?

So RL contains two important conditional structures:

1. the environment:

[
P(\text{next state} \mid \text{state, action})
]

2. the policy:

[
P(\text{action} \mid \text{state})
]

This links Lecture 2 to Lecture 4.

---

## 6.5 Diffusion as Conditional Denoising

Diffusion models learn conditional denoising.

At each step, the model receives a noisy input and predicts something about the less noisy version or the noise itself.

Conceptually:

[
P(x_{t-1} \mid x_t)
]

or:

[
P(\epsilon \mid x_t, t)
]

Meaning:

> Given the current noisy image, what noise should be removed?

This links Lecture 2 to Lecture 5.

---

## 6.6 Final Mini-Exercise

Ask students to translate each AI task into a conditional probability.

1. Predict whether an email is spam from its text.
2. Predict the next token from previous tokens.
3. Predict whether a patient has disease from symptoms.
4. Predict the next state from current state and action.
5. Predict the noise in a noisy image.

Expected answers:

1. (P(\text{spam} \mid \text{text}))
2. (P(\text{next token} \mid \text{previous tokens}))
3. (P(\text{disease} \mid \text{symptoms}))
4. (P(S\_{t+1} \mid S_t, A_t))
5. (P(\epsilon \mid x_t, t)), or informally (P(\text{noise} \mid \text{noisy image}))

---

# Part 7 — Lecture Summary

**Time:** 2 minutes

## Core Ideas Students Should Remember

1. **Conditional probability is the core of prediction.**

[
P(A \mid B)
]

means:

> How likely is (A), given (B)?

---

2. **Most AI models learn relationships between inputs and outputs.**

The central form is:

[
P(y \mid x)
]

Given input (x), predict a distribution over output (y).

---

3. **Joint probability describes things happening together.**

[
P(A,B)
]

Examples:

- fever and infection,
- token and context,
- state and action.

---

4. **Marginalization sums over possibilities.**

[
P(A) = \sum_B P(A,B)
]

This matters when there are hidden causes, latent variables, or multiple future states.

---

5. **Independence means information does not help.**

[
P(A \mid B) = P(A)
]

If (X) and (Y) are independent, (X) is useless for predicting (Y).

---

6. **Dependence is what makes learning possible.**

Language, images, medicine, RL, and diffusion all work because variables are statistically related.

---

7. **Bayes’ theorem formalizes belief updating.**

[
P(H \mid D) = \frac{P(D \mid H)P(H)}{P(D)}
]

It explains how evidence changes belief.

---

# Board Summary

A compact final board could look like this:

[
P(A \mid B) = \frac{P(A,B)}{P(B)}
]

[
P(A,B) = P(A \mid B)P(B)
]

[
P(A) = \sum_B P(A,B)
]

[
A \perp B \iff P(A \mid B) = P(A)
]

[
P(H \mid D) = \frac{P(D \mid H)P(H)}{P(D)}
]

[
P(y \mid x) = \text{prediction}
]

And the AI translation:

| Probability Concept | AI Meaning                                      |
| ------------------- | ----------------------------------------------- |
| (P(A \mid B))       | probability after seeing information            |
| (P(y \mid x))       | prediction from input                           |
| (P(A,B))            | variables appearing together                    |
| Marginalization     | summing over hidden possibilities               |
| Independence        | one variable gives no information about another |
| Dependence          | input contains useful predictive information    |
| Bayes’ theorem      | updating belief from evidence                   |

---

# Suggested Running Examples

Use the same few examples repeatedly so students do not get lost.

## Main Running Example 1: Medical Diagnosis

Use for:

- conditional probability,
- joint probability,
- Bayes’ theorem,
- base rates,
- reversing conditionals.

Example expressions:

[
P(\text{disease})
]

[
P(\text{symptoms} \mid \text{disease})
]

[
P(\text{disease} \mid \text{symptoms})
]

---

## Main Running Example 2: LLM Next Token

Use for:

- conditional probability,
- dependence,
- prediction,
- sequence modeling.

Example expression:

[
P(\text{next token} \mid \text{previous tokens})
]

Useful prompt:

> The capital of France is ...

Possible distribution:

| Token | Probability |
| ----- | ----------: |
| Paris |        0.82 |
| Lyon  |        0.04 |
| the   |        0.03 |
| other |        0.11 |

---

## Main Running Example 3: RL State Transition

Use for:

- conditional probability,
- dependence,
- action consequences,
- future states.

Example expression:

[
P(S_{t+1} \mid S_t, A_t)
]

Example:

A robot takes action “move forward.”

| Next state    | Probability |
| ------------- | ----------: |
| moved forward |        0.80 |
| slipped left  |        0.10 |
| slipped right |        0.10 |

---

# Suggested Exercises

## Exercise 1 — Conditional Probability from Counts

Given:

|                          | Spam | Not spam | Total |
| ------------------------ | ---: | -------: | ----: |
| Contains “prize”         |   45 |       15 |    60 |
| Does not contain “prize” |    5 |       35 |    40 |
| Total                    |   50 |       50 |   100 |

Questions:

1. What is (P(\text{spam}))?
2. What is (P(\text{contains prize}))?
3. What is (P(\text{spam}, \text{contains prize}))?
4. What is (P(\text{spam} \mid \text{contains prize}))?
5. Are spam and “contains prize” independent?

Expected answers:

1. (50/100 = 0.50)
2. (60/100 = 0.60)
3. (45/100 = 0.45)
4. (0.45/0.60 = 0.75)
5. No, because (P(\text{spam} \mid \text{contains prize}) = 0.75), but (P(\text{spam}) = 0.50)

---

## Exercise 2 — Translate to Conditional Probability

Translate each sentence into probability notation.

1. Probability of disease given symptoms.
2. Probability of next token given previous tokens.
3. Probability of next state given current state and action.
4. Probability of class label given image.
5. Probability of click given user and item.

Expected answers:

1. (P(\text{disease} \mid \text{symptoms}))
2. (P(X*t \mid X_1, ..., X*{t-1}))
3. (P(S\_{t+1} \mid S_t, A_t))
4. (P(Y \mid X)), or (P(\text{class} \mid \text{image}))
5. (P(\text{click} \mid \text{user}, \text{item}))

---

## Exercise 3 — Bayes’ Theorem

Suppose:

[
P(\text{disease}) = 0.05
]

[
P(\text{positive} \mid \text{disease}) = 0.90
]

[
P(\text{positive} \mid \text{no disease}) = 0.20
]

Question:

Compute:

[
P(\text{disease} \mid \text{positive})
]

Solution:

[
P(\text{positive})
==================

0.90 \cdot 0.05 + 0.20 \cdot 0.95
]

[
= 0.045 + 0.19 = 0.235
]

[
P(\text{disease} \mid \text{positive})
======================================

# \frac{0.90 \cdot 0.05}{0.235}

\frac{0.045}{0.235}
\approx 0.191
]

So:

[
P(\text{disease} \mid \text{positive}) \approx 19.1%
]

Teaching point:

> The posterior can be much lower than the test sensitivity if the disease is rare and false positives exist.

---

# What to Emphasize Most

The most important ideas in Lecture 2 are:

1. (P(y \mid x)) is the core structure of prediction.
2. Conditional probabilities are not symmetric.
3. Joint probabilities describe variables together.
4. Marginalization means summing over hidden or alternative possibilities.
5. Independence means information does not help.
6. Dependence is what makes machine learning possible.
7. Bayes’ theorem updates beliefs using evidence.

---

# What Not to Overdo

Avoid spending too much time on:

- formal probability axioms,
- measure-theoretic definitions,
- complex Bayesian networks,
- causal inference,
- continuous conditional densities,
- advanced graphical models,
- conjugate priors,
- Bayesian neural network details,
- detailed transformer math.

Those topics can come later.

Lecture 2 should mainly make students fluent in this idea:

[
\text{prediction} = \text{conditional probability}
]

---

# Recommended Ending

End with this transition:

> In this lecture, we moved from simple probability to conditional probability. That shift is the heart of AI prediction: instead of asking how likely an outcome is in general, we ask how likely it is given an input. In the next lecture, we will see how neural networks turn raw scores into probabilities and how training can be understood as making the observed data more likely.

Final board line:

[
P(y \mid x)
\quad \longrightarrow \quad
\text{learned by a neural network}
\quad \longrightarrow \quad
\text{trained with likelihood / loss}
]

Then say:

> Lecture 3 explains how this becomes softmax, likelihood, negative log-likelihood, and cross-entropy.
