# Lecture 1 — Probability as the Language of AI

**Theme:** AI systems do not usually output “answers”; they output or manipulate **probability distributions**.

**Duration:** 60 minutes
**Level:** Rusty high-school probability
**Style:** Mathematical but applied
**Core AI connection:** Prediction, uncertainty, learning, and generation all require probability.

---

# Lecture 1 Overview

## Central Message

Modern AI systems are best understood as systems that:

1. receive uncertain or incomplete information,
2. represent possible outcomes,
3. assign probabilities to those outcomes,
4. learn from data by adjusting those probabilities,
5. make predictions or decisions based on those probabilities.

A classifier does not simply “know this is a cat.”
An LLM does not simply “know the next word.”
An RL agent does not simply “know the best action.”
A diffusion model does not simply “draw an image.”

They all work with uncertainty.

---

# Lecture Structure

| Part |                              Topic |   Time |
| ---- | ---------------------------------: | -----: |
| 1    |      Why probability matters in AI |  8 min |
| 2    |  Outcomes, events, and probability | 10 min |
| 3    |                   Random variables | 10 min |
| 4    | Discrete probability distributions | 12 min |
| 5    |                        Expectation | 12 min |
| 6    |           Variance and uncertainty |  6 min |
| 7    |    Summary and bridge to Lecture 2 |  2 min |

---

# Learning Goals

By the end of the lecture, students should understand:

- what probability means in AI,
- what a probability distribution is,
- what a random variable is,
- why AI predictions are often distributions rather than single answers,
- how expectation represents an average under uncertainty,
- why variance measures uncertainty or spread,
- why probability is foundational for deep learning, LLMs, RL, and diffusion models.

---

# Part 1 — Why Probability Matters in AI

**Time:** 8 minutes

## 1.1 Motivation

Start with the following claim:

> AI systems are not usually deterministic answer machines. They are uncertainty machines.

This is especially true in modern AI.

A model often receives input and produces something like:

[
P(\text{output} \mid \text{input})
]

That means:

> Given this input, how likely are the possible outputs?

This basic idea appears everywhere:

| AI Area                | Probability Question                                  |
| ---------------------- | ----------------------------------------------------- |
| Image classification   | What class is most likely?                            |
| LLMs                   | What token is most likely next?                       |
| RL                     | Which action has the highest expected future reward?  |
| Diffusion models       | How can random noise be transformed into likely data? |
| Medical AI             | Given symptoms or data, how likely is each diagnosis? |
| Recommendation systems | How likely is the user to click, buy, or like this?   |

## 1.2 Opening Examples

### Example 1: Image Classifier

An image classifier sees an image and outputs:

| Class | Probability |
| ----- | ----------: |
| cat   |        0.72 |
| dog   |        0.21 |
| fox   |        0.07 |

Ask students:

> Did the model say “cat,” or did it say “cat is most likely”?

The important answer:

> The model produced a probability distribution over possible classes.

The final system may display only “cat,” but internally the probabilistic structure matters.

Why?

Because there is a big difference between:

| Class | Probability |
| ----- | ----------: |
| cat   |        0.99 |
| dog   |        0.01 |

and:

| Class | Probability |
| ----- | ----------: |
| cat   |        0.51 |
| dog   |        0.49 |

Both choose “cat” if we pick the maximum probability, but the second model is much less confident.

---

### Example 2: LLM Next-Token Prediction

Given the sentence:

> The patient was diagnosed with ...

An LLM might assign probabilities like:

| Next token     | Probability |
| -------------- | ----------: |
| cancer         |        0.25 |
| diabetes       |        0.18 |
| pneumonia      |        0.12 |
| flu            |        0.08 |
| something else |        0.37 |

The model does not “know” the next word in a deterministic way. It assigns probabilities to possible next tokens.

Core statement:

> LLMs generate text by repeatedly predicting a probability distribution over the next token.

This is the first major connection between probability and transformers.

---

### Example 3: Reinforcement Learning

An RL agent is in a state and can choose between actions.

| Action   | Immediate reward | Possible future value |
| -------- | ---------------: | --------------------: |
| go left  |                2 |                   low |
| go right |                0 |                  high |
| wait     |                1 |                medium |

The best action may not be the one with the highest immediate reward.

The agent cares about:

[
\text{expected future reward}
]

So probability is needed because:

- actions may have uncertain outcomes,
- rewards may be noisy,
- future states are uncertain,
- the best action depends on long-term expectation.

---

### Example 4: Diffusion Models

A diffusion model starts with random noise and gradually transforms it into an image.

The process is probabilistic because:

- the starting point is random noise,
- each denoising step involves uncertainty,
- many different images could match the same prompt,
- generation involves sampling from a learned distribution.

Core intuition:

> Diffusion models learn the structure of data by learning how to reverse a noise process.

---

## 1.3 Key Takeaway

Probability is central to AI because AI systems usually deal with:

- incomplete information,
- uncertain labels,
- noisy data,
- multiple plausible outputs,
- uncertain future rewards,
- stochastic generation.

Write this on the board:

[
\text{AI} = \text{learning useful structure under uncertainty}
]

---

# Part 2 — Outcomes, Events, and Probability

**Time:** 10 minutes

## 2.1 Sample Space

A **sample space** is the set of all possible outcomes.

Notation:

[
\Omega
]

Example: rolling a die.

[
\Omega = {1,2,3,4,5,6}
]

Example: binary classifier.

[
\Omega = {\text{positive}, \text{negative}}
]

Example: image classifier.

[
\Omega = {\text{cat}, \text{dog}, \text{fox}, \text{car}, ...}
]

Example: LLM next token.

[
\Omega = {\text{all tokens in the vocabulary}}
]

If the vocabulary has 50,000 tokens, then the sample space has 50,000 possible next-token outcomes.

---

## 2.2 Outcome

An **outcome** is one specific result from the sample space.

Examples:

| Setting    | Outcome                      |
| ---------- | ---------------------------- |
| Die roll   | (4)                          |
| Classifier | “cat”                        |
| LLM        | next token is “therefore”    |
| RL         | next state is (s')           |
| Diffusion  | one possible generated image |

Important:

> The outcome is what actually happens.
> The probability distribution describes what could happen before we know the outcome.

---

## 2.3 Event

An **event** is a set of outcomes.

Example: die roll is even.

[
A = {2,4,6}
]

Example: classifier output is an animal.

[
A = {\text{cat}, \text{dog}, \text{fox}, \text{horse}, ...}
]

Example: LLM next token is a punctuation mark.

[
A = {., ,, ;, :, !, ?}
]

So:

- an outcome is one possibility,
- an event is a group of possibilities.

---

## 2.4 Probability of an Event

A probability assigns a number between 0 and 1 to an event.

[
0 \leq P(A) \leq 1
]

Interpretation:

| Probability | Meaning                      |
| ----------: | ---------------------------- |
|           0 | impossible                   |
|         0.1 | unlikely                     |
|         0.5 | equally possible / uncertain |
|         0.9 | likely                       |
|           1 | certain                      |

Example:

[
P(\text{cat}) = 0.72
]

means:

> According to the model, “cat” has probability 0.72.

It does **not** necessarily mean the image is objectively 72% cat. It means the model assigns 72% probability to that class under its learned representation.

This distinction matters.

---

## 2.5 Probabilities Must Sum to 1

For a complete set of mutually exclusive possible outcomes:

[
\sum_i P(x_i) = 1
]

Example:

| Class | Probability |
| ----- | ----------: |
| cat   |        0.72 |
| dog   |        0.21 |
| fox   |        0.07 |

Total:

[
0.72 + 0.21 + 0.07 = 1.00
]

For an LLM:

[
\sum_{\text{token } t} P(t \mid \text{context}) = 1
]

This means:

> At every generation step, the model distributes one full unit of probability mass across all possible next tokens.

---

## 2.6 Mutually Exclusive Events

Two events are **mutually exclusive** if they cannot both happen at the same time.

Example:

For one die roll:

- rolling a 2,
- rolling a 5.

These cannot both happen at once.

For one classifier prediction:

- true class is cat,
- true class is dog.

Usually, these are mutually exclusive if the dataset assumes only one label.

But be careful: in multi-label classification, the image could be both “person” and “bicycle.”

Important AI distinction:

| Task Type                   | Classes                |
| --------------------------- | ---------------------- |
| Single-label classification | mutually exclusive     |
| Multi-label classification  | not mutually exclusive |

Example:

Single-label:

[
P(\text{cat}) + P(\text{dog}) + P(\text{fox}) = 1
]

Multi-label:

[
P(\text{person}) = 0.91,\quad P(\text{bicycle}) = 0.86,\quad P(\text{helmet}) = 0.44
]

These probabilities do not need to sum to 1, because several labels can be true at the same time.

This is a useful early warning that probability in AI depends on the modeling setup.

---

## 2.7 Exhaustive Events

A set of events is **exhaustive** if it covers all possibilities.

Example:

For a binary classifier:

[
{\text{positive}, \text{negative}}
]

is exhaustive if there are no other possible labels.

For an LLM:

The vocabulary is exhaustive at a given token step:

[
\Omega = {\text{all possible next tokens}}
]

The model must place all probability somewhere.

---

## 2.8 Mini-Exercise

Give students this distribution:

| Next token | Probability |
| ---------- | ----------: |
| the        |        0.40 |
| a          |        0.25 |
| an         |        0.10 |
| dog        |        0.05 |
| other      |        0.20 |

Ask:

1. What is the sample space here?
2. What is the probability of the event “next token is an article”?
3. Do the probabilities sum to 1?
4. Is “the” mutually exclusive with “a” at this token position?

Expected answers:

1. ({\text{the}, \text{a}, \text{an}, \text{dog}, \text{other}})
2. (0.40 + 0.25 + 0.10 = 0.75)
3. Yes.
4. Yes, only one next token is chosen at this step.

---

# Part 3 — Random Variables

**Time:** 10 minutes

## 3.1 What Is a Random Variable?

A **random variable** is a variable whose value is uncertain.

It maps possible outcomes to values.

For this course, keep it intuitive:

> A random variable is a named uncertain quantity.

Examples:

| Symbol     | Meaning         |
| ---------- | --------------- |
| (X)        | next token      |
| (Y)        | class label     |
| (R)        | reward          |
| (S)        | state           |
| (A)        | action          |
| (Z)        | latent variable |
| (\epsilon) | noise           |

---

## 3.2 Random Variables in AI

### Classification

[
Y = \text{class label}
]

Possible values:

[
Y \in {\text{cat}, \text{dog}, \text{fox}}
]

The model estimates:

[
P(Y = \text{cat})
]

or more generally:

[
P(Y \mid X)
]

where (X) is the input image.

---

### LLMs

[
X_t = \text{token at position } t
]

An LLM estimates:

[
P(X_t \mid X_1, X_2, ..., X_{t-1})
]

In words:

> What is the probability of the next token, given the previous tokens?

This is one of the most important probability expressions in modern AI.

---

### Reinforcement Learning

[
S_t = \text{state at time } t
]

[
A_t = \text{action at time } t
]

[
R_t = \text{reward at time } t
]

An RL problem is built from random variables changing over time:

[
S_t \rightarrow A_t \rightarrow R_{t+1}, S_{t+1}
]

The next state may be uncertain:

[
P(S_{t+1} \mid S_t, A_t)
]

This will be studied more deeply in Lecture 4.

---

### Diffusion Models

[
X_0 = \text{clean image}
]

[
X_t = \text{noisier version of the image}
]

[
\epsilon = \text{random noise}
]

Diffusion models work with random variables representing clean data, noisy data, and noise.

The forward process gradually adds random noise.

The reverse process tries to remove it.

---

## 3.3 Outcome vs Random Variable vs Distribution

This distinction is crucial.

Use this example:

[
X = \text{next token}
]

Before sampling, (X) is uncertain.

The model gives a distribution:

| Value of (X) | Probability |
| ------------ | ----------: |
| “cat”        |        0.50 |
| “dog”        |        0.30 |
| “fox”        |        0.20 |

After sampling, one outcome happens:

[
X = \text{“cat”}
]

So:

| Concept         | Meaning                            |
| --------------- | ---------------------------------- |
| Random variable | uncertain quantity                 |
| Distribution    | probabilities over possible values |
| Outcome         | actual realized value              |

Core teaching sentence:

> A random variable is the uncertain thing; a distribution tells us how likely its possible values are; an outcome is what actually occurs.

---

## 3.4 Discrete vs Continuous Random Variables

Briefly introduce the distinction.

### Discrete random variables

Take countable values.

Examples:

- class label,
- token,
- action choice,
- die roll.

### Continuous random variables

Take values on a continuum.

Examples:

- temperature,
- model weight,
- image pixel intensity if treated continuously,
- Gaussian noise,
- latent embedding coordinate.

For Lecture 1, focus mostly on discrete variables.

Why?

Because classification and LLM token prediction are easier starting points.

Diffusion models need continuous probability later, but not yet in detail.

---

## 3.5 Mini-Exercise

Ask students:

For each of the following, identify the random variable and possible values.

1. An email spam classifier.
2. A language model predicting the next token.
3. An RL agent choosing between left, right, and wait.
4. A diffusion model adding noise to an image.

Expected answers:

1. (Y =) email label, values: spam/not spam.
2. (X_t =) next token, values: all vocabulary tokens.
3. (A_t =) action, values: left/right/wait.
4. (\epsilon =) noise, values: continuous noise values.

---

# Part 4 — Discrete Probability Distributions

**Time:** 12 minutes

## 4.1 What Is a Probability Distribution?

A probability distribution assigns probabilities to possible values of a random variable.

For a discrete random variable (X):

[
P(X=x)
]

means:

> The probability that random variable (X) takes value (x).

Example:

[
P(X = \text{cat}) = 0.72
]

A full distribution gives probabilities for all possible values.

---

## 4.2 Bernoulli Distribution

A Bernoulli distribution is the simplest distribution.

It describes a random variable with two possible outcomes.

Examples:

- heads/tails,
- yes/no,
- spam/not spam,
- click/no click,
- success/failure.

Let:

[
X \in {0,1}
]

where:

- (X=1): success,
- (X=0): failure.

If:

[
P(X=1)=p
]

then:

[
P(X=0)=1-p
]

Example:

A spam classifier says:

[
P(Y=\text{spam}) = 0.8
]

Then:

[
P(Y=\text{not spam}) = 0.2
]

assuming these are the only two possibilities.

AI examples:

| Task              | Bernoulli variable    |
| ----------------- | --------------------- |
| Click prediction  | click or no click     |
| Medical test      | disease or no disease |
| Fraud detection   | fraud or not fraud    |
| Binary classifier | class 1 or class 0    |

---

## 4.3 Categorical Distribution

A categorical distribution describes a random variable with more than two possible categories.

Example:

[
Y \in {\text{cat}, \text{dog}, \text{fox}}
]

Distribution:

| Value | Probability |
| ----- | ----------: |
| cat   |        0.72 |
| dog   |        0.21 |
| fox   |        0.07 |

This is central in AI because many models choose between multiple possible outputs.

Examples:

- image class,
- next token,
- action choice,
- diagnosis category,
- topic label.

---

## 4.4 Token Distribution in LLMs

For LLMs, the next token is modeled as a categorical random variable.

Let:

[
X_t = \text{next token}
]

Then:

[
P(X_t = w_i \mid \text{previous tokens})
]

where (w_i) is one token from the vocabulary.

Example:

Prompt:

> The capital of France is

Possible next-token distribution:

| Token  | Probability |
| ------ | ----------: |
| Paris  |        0.82 |
| Lyon   |        0.04 |
| the    |        0.03 |
| France |        0.02 |
| other  |        0.09 |

Important:

The model may output “Paris,” but internally it created a whole distribution.

Core phrase:

> An LLM is repeatedly solving a categorical prediction problem over a huge vocabulary.

---

## 4.5 Probability Mass

For discrete distributions, probabilities are often called **probability mass**.

Intuition:

> The model has one unit of probability mass and must spread it across possible outcomes.

Example:

A confident distribution:

| Token  | Probability |
| ------ | ----------: |
| Paris  |        0.95 |
| Lyon   |        0.02 |
| London |        0.01 |
| other  |        0.02 |

An uncertain distribution:

| Token  | Probability |
| ------ | ----------: |
| Paris  |        0.35 |
| Lyon   |        0.25 |
| London |        0.20 |
| other  |        0.20 |

Both sum to 1, but they express very different uncertainty.

---

## 4.6 Prediction: Distribution vs Decision

This is an important distinction.

The model outputs a distribution.

A downstream rule turns that distribution into a decision.

Example:

Distribution:

| Class | Probability |
| ----- | ----------: |
| cat   |        0.51 |
| dog   |        0.49 |

Decision rule:

[
\text{choose class with highest probability}
]

Output:

[
\text{cat}
]

But the distribution tells us the model was barely confident.

This distinction matters in:

- medical AI,
- autonomous driving,
- RL,
- high-stakes decisions,
- LLM reliability.

A model’s final answer can hide uncertainty.

---

## 4.7 Calibration

Introduce very briefly.

A model is **well-calibrated** if its probabilities mean what they say.

For example:

Among cases where the model says 70% confidence, the correct answer should happen about 70% of the time.

This is not the same as accuracy.

A model can be accurate but overconfident.

Example:

A model predicts many answers with 99% confidence, but it is correct only 80% of the time.

That model is overconfident.

This matters because AI systems often sound confident even when uncertain.

---

## 4.8 Mini-Exercise

Give this classifier output:

| Diagnosis | Probability |
| --------- | ----------: |
| flu       |        0.45 |
| COVID     |        0.35 |
| allergy   |        0.15 |
| other     |        0.05 |

Ask:

1. Is this a Bernoulli or categorical distribution?
2. What is the most likely diagnosis?
3. Is the model very confident?
4. What is the probability that the diagnosis is either flu or COVID?

Expected answers:

1. Categorical.
2. Flu.
3. Not extremely confident; COVID is also plausible.
4. (0.45 + 0.35 = 0.80)

---

# Part 5 — Expectation

**Time:** 12 minutes

## 5.1 What Is Expectation?

Expectation is the probability-weighted average value of a random variable.

For a discrete random variable:

[
\mathbb{E}[X] = \sum_i x_i P(X=x_i)
]

In words:

> Multiply each possible value by its probability, then add them up.

Expectation is not necessarily the value that will happen.

It is the average value we would expect over many repetitions.

---

## 5.2 Simple Example: Die Roll

For a fair six-sided die:

[
P(X=1)=P(X=2)=...=P(X=6)=\frac{1}{6}
]

Expected value:

[
\mathbb{E}[X]
=============

1\cdot \frac{1}{6}

- 2\cdot \frac{1}{6}
- 3\cdot \frac{1}{6}
- 4\cdot \frac{1}{6}
- 5\cdot \frac{1}{6}
- 6\cdot \frac{1}{6}
  ]

[
\mathbb{E}[X] = 3.5
]

Important:

You can never roll 3.5.

Expectation is not necessarily a possible outcome.

It is a long-run average.

---

## 5.3 Expected Reward

Now connect to reinforcement learning.

Suppose an agent can choose action A.

Action A has uncertain reward:

| Reward | Probability |
| -----: | ----------: |
|     10 |         0.5 |
|      0 |         0.5 |

Expected reward:

[
\mathbb{E}[R] = 10 \cdot 0.5 + 0 \cdot 0.5 = 5
]

Action B has guaranteed reward:

| Reward | Probability |
| -----: | ----------: |
|      4 |         1.0 |

Expected reward:

[
\mathbb{E}[R] = 4
]

So Action A has higher expected reward, even though it is risky.

This introduces the core RL idea:

> Agents often choose actions based on expected return, not guaranteed return.

---

## 5.4 Expected Loss

In machine learning, we often care about expected loss.

A loss measures how bad a model’s prediction is.

For example:

- wrong classification,
- large prediction error,
- bad next-token prediction,
- bad action choice.

Training tries to minimize average loss over data:

[
\mathbb{E}[\text{loss}]
]

In practice, we usually estimate this average from a dataset.

If we have training examples:

[
(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)
]

then we minimize average training loss:

[
\frac{1}{n}\sum_{i=1}^{n} L(f(x_i), y_i)
]

This is an empirical approximation of expected loss.

Important bridge to deep learning:

> Neural network training usually means adjusting parameters to reduce expected prediction error.

---

## 5.5 Expected Next Token? A Useful Warning

For numeric variables, expectation is straightforward.

For categories like words, expectation is less direct.

You cannot average:

[
0.4 \cdot \text{“cat”} + 0.3 \cdot \text{“dog”}
]

That does not make sense as ordinary arithmetic.

For categorical variables, we usually use the distribution itself or choose/sample from it.

However, inside neural networks, tokens are represented as vectors, and expectations over vectors can make mathematical sense.

For this lecture, keep the main point simple:

- expectation is natural for numerical quantities,
- distributions are natural for categorical outputs,
- later, embeddings allow categorical things to be represented numerically.

---

## 5.6 Expectation in AI: Three Major Uses

### 1. Expected loss

Used in supervised learning and deep learning.

[
\min \mathbb{E}[\text{loss}]
]

Meaning:

> Make the model wrong by as little as possible on average.

---

### 2. Expected reward

Used in reinforcement learning.

[
\max \mathbb{E}[\text{return}]
]

Meaning:

> Choose actions that lead to high reward on average over time.

---

### 3. Expected prediction

Used in regression and uncertainty estimation.

Example:

A model predicts house price:

| Price | Probability |
| ----: | ----------: |
|  400k |         0.2 |
|  500k |         0.5 |
|  600k |         0.3 |

Expected price:

[
400k \cdot 0.2 + 500k \cdot 0.5 + 600k \cdot 0.3 = 510k
]

---

## 5.7 Mini-Exercise

An RL agent considers two actions.

Action A:

| Reward | Probability |
| -----: | ----------: |
|    100 |         0.1 |
|      0 |         0.9 |

Action B:

| Reward | Probability |
| -----: | ----------: |
|     20 |         1.0 |

Ask:

1. What is the expected reward of Action A?
2. What is the expected reward of Action B?
3. Which action has higher expected reward?
4. Which action is riskier?

Expected answers:

1. (100 \cdot 0.1 + 0 \cdot 0.9 = 10)
2. (20)
3. Action B.
4. Action A.

Teaching point:

> Higher possible reward is not the same as higher expected reward.

---

# Part 6 — Variance and Uncertainty

**Time:** 6 minutes

## 6.1 Why Expectation Is Not Enough

Expectation tells us the average.

But two random variables can have the same expectation and very different uncertainty.

Example:

Action A:

| Reward | Probability |
| -----: | ----------: |
|      5 |         1.0 |

Expected reward:

[
5
]

Action B:

| Reward | Probability |
| -----: | ----------: |
|     10 |         0.5 |
|      0 |         0.5 |

Expected reward:

[
5
]

Both have expected reward 5.

But Action B is more uncertain.

That difference is captured by variance.

---

## 6.2 Informal Definition of Variance

Variance measures how spread out outcomes are around the expectation.

Low variance:

- outcomes are close to the average,
- predictions are stable,
- uncertainty is low.

High variance:

- outcomes are spread out,
- predictions are unstable,
- uncertainty is high.

Formal definition, optional:

[
\mathrm{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]
]

Do not spend too long deriving this.

The intuition matters more for this lecture.

---

## 6.3 AI Examples of Variance

### Classification

Low uncertainty:

| Class | Probability |
| ----- | ----------: |
| cat   |        0.98 |
| dog   |        0.01 |
| fox   |        0.01 |

High uncertainty:

| Class | Probability |
| ----- | ----------: |
| cat   |        0.36 |
| dog   |        0.34 |
| fox   |        0.30 |

The second distribution is more uncertain.

Strictly speaking, this is often discussed using entropy rather than variance, but variance is the first intuitive step.

---

### Reinforcement Learning

Two actions can have the same expected reward but different risk.

One action may reliably produce medium rewards.

Another may sometimes produce very high rewards and sometimes fail completely.

This matters in real-world RL:

- medicine,
- robotics,
- finance,
- autonomous driving.

---

### Diffusion and Noise

Diffusion models explicitly add noise.

More noise means more uncertainty about the original data.

The model learns to reverse that uncertainty step by step.

---

## 6.4 Variance vs Entropy

Very briefly preview:

- variance measures spread for numerical variables,
- entropy measures uncertainty in a probability distribution.

Entropy will appear later, especially for:

- LLM token distributions,
- classification confidence,
- exploration in RL.

Do not fully teach entropy yet; just introduce it as a future concept.

---

# Part 7 — Lecture Summary

**Time:** 2 minutes

## Core Ideas Students Should Remember

1. **Probability is the language of uncertainty.**

AI systems use probability because data, predictions, actions, and generated outputs are uncertain.

2. **A prediction is often a distribution, not just an answer.**

A classifier may output:

[
P(\text{cat}) = 0.72
]

not simply:

[
\text{cat}
]

3. **A random variable is an uncertain quantity.**

Examples:

[
X = \text{next token}
]

[
Y = \text{class label}
]

[
R = \text{reward}
]

[
S = \text{next state}
]

4. **A distribution describes how likely each possible value is.**

For LLMs, the next-token distribution is a categorical distribution over the vocabulary.

5. **Expectation is a probability-weighted average.**

It is central to:

- expected loss,
- expected reward,
- expected prediction.

6. **Variance describes spread or uncertainty.**

Expectation alone is not enough because two choices can have the same average but different risk.

---

# Board Summary

A compact final board could look like this:

[
\Omega = \text{set of possible outcomes}
]

[
P(A) = \text{probability of event } A
]

[
X = \text{random variable}
]

[
P(X=x) = \text{probability that } X \text{ takes value } x
]

[
\sum_i P(X=x_i)=1
]

[
\mathbb{E}[X] = \sum_i x_i P(X=x_i)
]

[
\mathrm{Var}(X) = \mathbb{E}[(X-\mathbb{E}[X])^2]
]

And the AI translation:

| Probability Concept | AI Meaning                               |
| ------------------- | ---------------------------------------- |
| Sample space        | possible labels, tokens, actions, states |
| Event               | group of possible outcomes               |
| Random variable     | uncertain token, class, reward, state    |
| Distribution        | model’s uncertainty over possibilities   |
| Expectation         | average loss, reward, or prediction      |
| Variance            | uncertainty, instability, risk           |
| Sampling            | choosing an output from a distribution   |

---

# Suggested In-Lecture Examples

Use only a few recurring examples instead of many disconnected ones.

## Main Running Example 1: LLM Next Token

Prompt:

> The capital of France is

Distribution:

| Token  | Probability |
| ------ | ----------: |
| Paris  |        0.82 |
| Lyon   |        0.04 |
| the    |        0.03 |
| France |        0.02 |
| other  |        0.09 |

Use this example to teach:

- sample space,
- outcome,
- probability distribution,
- categorical distribution,
- uncertainty,
- prediction vs decision.

---

## Main Running Example 2: Image Classifier

Distribution:

| Class | Probability |
| ----- | ----------: |
| cat   |        0.72 |
| dog   |        0.21 |
| fox   |        0.07 |

Use this example to teach:

- model confidence,
- mutually exclusive outcomes,
- probabilities summing to 1,
- prediction vs probability.

---

## Main Running Example 3: RL Reward

Action A:

| Reward | Probability |
| -----: | ----------: |
|     10 |         0.5 |
|      0 |         0.5 |

Action B:

| Reward | Probability |
| -----: | ----------: |
|      4 |         1.0 |

Use this example to teach:

- expectation,
- risk,
- variance,
- expected reward,
- why RL needs probability.

---

# Suggested Exercises

## Exercise 1 — Token Probabilities

An LLM produces this distribution:

| Token | Probability |
| ----- | ----------: |
| yes   |        0.50 |
| no    |        0.30 |
| maybe |        0.15 |
| other |        0.05 |

Questions:

1. What is the sample space?
2. What is (P(\text{yes}))?
3. What is (P(\text{yes or no}))?
4. Do the probabilities sum to 1?
5. Is the model certain?

Expected answers:

1. yes, no, maybe, other.
2. 0.50.
3. 0.80.
4. Yes.
5. No.

---

## Exercise 2 — Classification Confidence

Two classifiers predict the same class.

Model A:

| Class | Probability |
| ----- | ----------: |
| cat   |        0.95 |
| dog   |        0.03 |
| fox   |        0.02 |

Model B:

| Class | Probability |
| ----- | ----------: |
| cat   |        0.45 |
| dog   |        0.40 |
| fox   |        0.15 |

Questions:

1. Which class does each model choose?
2. Which model is more confident?
3. Why is the final label alone not enough information?

Expected answers:

1. Both choose cat.
2. Model A.
3. Because Model B is nearly uncertain between cat and dog.

---

## Exercise 3 — Expected Reward

An agent can choose between two actions.

Action A:

| Reward | Probability |
| -----: | ----------: |
|     30 |         0.2 |
|      0 |         0.8 |

Action B:

| Reward | Probability |
| -----: | ----------: |
|      5 |         1.0 |

Questions:

1. Compute expected reward of Action A.
2. Compute expected reward of Action B.
3. Which action has higher expected reward?
4. Which action is riskier?

Expected answers:

1. (30 \cdot 0.2 + 0 \cdot 0.8 = 6)
2. (5)
3. Action A.
4. Action A.

---

# What Not to Overdo in Lecture 1

Avoid spending too much time on:

- combinatorics,
- permutations,
- binomial coefficients,
- formal probability axioms,
- measure theory,
- continuous densities,
- Gaussian formulas,
- Bayes’ theorem.

Bayes should come in Lecture 2.

Lecture 1 should be about building intuition around:

[
\text{uncertainty} \rightarrow \text{random variable} \rightarrow \text{distribution} \rightarrow \text{expectation}
]

---

# Recommended Ending

End with this transition:

> Today we learned how AI systems represent uncertainty using random variables and probability distributions. In the next lecture, we move from simple probabilities to conditional probabilities: instead of asking “how likely is this outcome?”, we ask “how likely is this outcome given some information?” That shift — from (P(y)) to (P(y \mid x)) — is the mathematical heart of prediction in AI.

Final board line:

[
P(y \mid x)
]

Then say:

> This expression will be the center of Lecture 2.
