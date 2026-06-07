# Lecture 3 — Likelihood, Loss, Softmax, and Deep Learning

**Theme:** Training neural networks is largely about making observed data more probable.

**Duration:** 60 minutes
**Level:** After Lectures 1–2
**Style:** Mathematical but applied
**Core AI connection:** Neural networks learn by adjusting their parameters so that correct labels, tokens, or actions receive higher probability.

---

# Lecture 3 Overview

## Central Message

In Lecture 1, students learned that AI models often output probability distributions.

In Lecture 2, they learned that prediction usually means estimating:

[
P(y \mid x)
]

Lecture 3 explains how neural networks actually do this in practice.

The pipeline is:

[
\text{input}
\rightarrow
\text{neural network}
\rightarrow
\text{logits}
\rightarrow
\text{softmax probabilities}
\rightarrow
\text{loss}
\rightarrow
\text{gradient update}
]

The most important idea:

> Training a neural network often means increasing the probability assigned to the correct observed answer.

For classifiers:

[
P(\text{correct class} \mid \text{image})
]

For LLMs:

[
P(\text{correct next token} \mid \text{previous tokens})
]

For policies in RL:

[
P(\text{action} \mid \text{state})
]

---

# Lecture Structure

| Part |                              Topic |   Time |
| ---- | ---------------------------------: | -----: |
| 1    | From model scores to probabilities | 12 min |
| 2    |                         Likelihood |  8 min |
| 3    |                     Log-likelihood |  8 min |
| 4    |            Negative log-likelihood |  8 min |
| 5    |                 Cross-entropy loss | 12 min |
| 6    |            Entropy and uncertainty |  8 min |
| 7    |    Summary and bridge to Lecture 4 |  4 min |

---

# Learning Goals

By the end of the lecture, students should understand:

- what logits are,
- why logits are not probabilities,
- how softmax converts logits into probabilities,
- what likelihood means,
- why training often maximizes likelihood,
- why logs are used,
- why deep learning minimizes negative log-likelihood,
- how cross-entropy relates to negative log-likelihood,
- why LLMs are trained with next-token cross-entropy,
- what entropy says about uncertainty.

---

# Part 1 — From Model Scores to Probabilities

**Time:** 12 minutes

## 1.1 Recap: Prediction as (P(y \mid x))

Start with the central formula from Lecture 2:

[
P(y \mid x)
]

In supervised learning:

- (x) is the input,
- (y) is the output,
- the model tries to estimate a probability distribution over (y).

Examples:

[
P(\text{class} \mid \text{image})
]

[
P(\text{next token} \mid \text{previous tokens})
]

[
P(\text{disease} \mid \text{symptoms})
]

But a neural network does not naturally output probabilities first.

It usually outputs raw scores.

These raw scores are called **logits**.

---

## 1.2 What Are Logits?

A neural network often produces one raw number per possible class or token.

Example:

| Token | Logit |
| ----- | ----: |
| cat   |   2.1 |
| dog   |   1.3 |
| car   |  -0.5 |

These numbers are not probabilities.

Why not?

Because probabilities must satisfy:

[
0 \leq P(x) \leq 1
]

and:

[
\sum_i P(x_i)=1
]

But logits:

- can be negative,
- can be larger than 1,
- do not need to sum to anything meaningful.

So logits are better understood as **unnormalized preference scores**.

The model is saying:

> “cat” currently scores higher than “dog,” and “dog” scores higher than “car.”

But it has not yet produced a valid probability distribution.

---

## 1.3 From Logits to Probabilities

To turn logits into probabilities, we use **softmax**.

[
P(y_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
]

Where:

- (z_i) is the logit for class/token (i),
- (e^{z_i}) makes the score positive,
- the denominator makes all probabilities sum to 1.

Softmax does three things:

1. turns raw scores into positive numbers,
2. preserves ranking: higher logit means higher probability,
3. normalizes the values so they sum to 1.

---

## 1.4 Numerical Example

Suppose the model outputs:

| Token | Logit |
| ----- | ----: |
| cat   |   2.0 |
| dog   |   1.0 |
| car   |   0.0 |

First exponentiate:

| Token | Logit | (e^{z}) |
| ----- | ----: | ------: |
| cat   |   2.0 |    7.39 |
| dog   |   1.0 |    2.72 |
| car   |   0.0 |    1.00 |

Sum:

[
7.39 + 2.72 + 1.00 = 11.11
]

Now divide each exponentiated score by the sum:

| Token |               Probability |
| ----- | ------------------------: |
| cat   | (7.39/11.11 \approx 0.67) |
| dog   | (2.72/11.11 \approx 0.24) |
| car   | (1.00/11.11 \approx 0.09) |

So:

[
P(\text{cat}) \approx 0.67
]

[
P(\text{dog}) \approx 0.24
]

[
P(\text{car}) \approx 0.09
]

The model now has a valid probability distribution.

---

## 1.5 Why Exponentials?

Do not spend too much time on this, but give intuition.

Exponentials are useful because:

- they turn all values positive,
- larger logits become much larger positive scores,
- differences between logits matter,
- the output can be normalized into probabilities.

Example:

A difference of 1 in logits corresponds to multiplying odds by about (e \approx 2.7).

So softmax is sensitive to relative differences.

If one logit is much larger than the others, softmax gives it much higher probability.

---

## 1.6 Softmax in AI

Softmax appears in many places.

### Classifiers

An image classifier outputs logits for classes:

| Class | Logit |
| ----- | ----: |
| cat   |   4.2 |
| dog   |   1.1 |
| car   |  -0.8 |

Softmax converts this into:

[
P(\text{class} \mid \text{image})
]

---

### LLMs

An LLM outputs one logit per vocabulary token.

If the vocabulary has 50,000 tokens, the model outputs 50,000 logits.

Softmax turns those into:

[
P(\text{next token} \mid \text{context})
]

This is the distribution used for next-token prediction and generation.

---

### Reinforcement Learning Policies

A policy network may output logits for possible actions:

| Action | Logit |
| ------ | ----: |
| left   |   1.5 |
| right  |   0.2 |
| wait   |  -0.7 |

Softmax turns these into:

[
\pi(a \mid s)
]

That is:

> probability of action (a), given state (s).

---

## 1.7 Important Distinction: Score, Probability, Decision

Students should clearly separate three things:

| Stage           | Meaning                |
| --------------- | ---------------------- |
| Logit           | raw model score        |
| Probability     | normalized uncertainty |
| Decision/sample | chosen output          |

Example:

| Token | Logit | Probability |
| ----- | ----: | ----------: |
| cat   |   2.0 |        0.67 |
| dog   |   1.0 |        0.24 |
| car   |   0.0 |        0.09 |

A decision rule may then choose the most likely token:

[
\text{choose cat}
]

But the probability distribution contains more information than the final choice.

---

## 1.8 Mini-Exercise

Given logits:

| Class | Logit |
| ----- | ----: |
| A     |     3 |
| B     |     1 |
| C     |     0 |

Ask:

1. Are these probabilities?
2. Which class will have the highest softmax probability?
3. Will the probabilities sum to 1 after softmax?
4. Why is softmax needed?

Expected answers:

1. No.
2. A.
3. Yes.
4. To convert arbitrary scores into a valid probability distribution.

No need to calculate exact values unless students are comfortable.

---

# Part 2 — Likelihood

**Time:** 8 minutes

## 2.1 What Is Likelihood?

Likelihood asks:

> How probable is the observed data under the model?

Suppose the correct class is “cat.”

Model A predicts:

| Class | Probability |
| ----- | ----------: |
| cat   |        0.80 |
| dog   |        0.15 |
| car   |        0.05 |

Model B predicts:

| Class | Probability |
| ----- | ----------: |
| cat   |        0.10 |
| dog   |        0.70 |
| car   |        0.20 |

Model A assigns higher probability to the observed correct label.

So Model A has higher likelihood for this example.

---

## 2.2 Likelihood for One Example

For one training example:

[
(x, y)
]

the model predicts:

[
P_\theta(y \mid x)
]

where (\theta) represents the model parameters.

The likelihood of the observed label is:

[
P_\theta(y_{\text{true}} \mid x)
]

Example:

Input:

> image of a cat

Correct label:

[
y = \text{cat}
]

Model assigns:

[
P_\theta(\text{cat} \mid \text{image}) = 0.80
]

Then the likelihood for this example is 0.80.

---

## 2.3 Likelihood for Many Examples

For a dataset:

[
(x_1,y_1), (x_2,y_2), ..., (x_n,y_n)
]

the likelihood is the probability the model assigns to all observed correct outputs:

[
L(\theta)
=========

\prod*{i=1}^{n}
P*\theta(y_i \mid x_i)
]

In words:

> Multiply the probabilities that the model assigns to the correct labels across all training examples.

Example:

| Example | Correct Label | Model Probability for Correct Label |
| ------- | ------------- | ----------------------------------: |
| 1       | cat           |                                0.80 |
| 2       | dog           |                                0.60 |
| 3       | car           |                                0.50 |

Likelihood:

[
0.80 \cdot 0.60 \cdot 0.50 = 0.24
]

The better the model assigns high probability to correct labels, the higher the likelihood.

---

## 2.4 Likelihood in LLMs

For LLMs, each training example is often a sequence of tokens.

Example:

> The cat sleeps

The model is trained to predict each next token.

Simplified:

[
P(\text{The}) \cdot
P(\text{cat} \mid \text{The}) \cdot
P(\text{sleeps} \mid \text{The cat})
]

The likelihood of the text is the product of probabilities assigned to the actual observed tokens.

Core idea:

> An LLM is trained to assign high probability to the real next tokens in its training text.

---

## 2.5 Likelihood Is a Function of Parameters

This is subtle but important.

Probability treats the data as random given fixed parameters:

[
P_\theta(y \mid x)
]

Likelihood treats the observed data as fixed and asks:

> Which parameters (\theta) make this observed data most probable?

So during training, the data is fixed.

The model parameters change.

Training asks:

[
\theta^* = \arg\max_\theta L(\theta)
]

Meaning:

> Find parameters that maximize the likelihood of the observed data.

Do not overdo this technically, but make the intuition clear:

> We adjust the neural network so that it gives higher probability to what actually happened in the dataset.

---

## 2.6 Mini-Exercise

Correct class: dog.

Model A:

| Class | Probability |
| ----- | ----------: |
| cat   |        0.20 |
| dog   |        0.70 |
| car   |        0.10 |

Model B:

| Class | Probability |
| ----- | ----------: |
| cat   |        0.40 |
| dog   |        0.30 |
| car   |        0.30 |

Ask:

1. Which model has higher likelihood for this example?
2. Why?
3. What would training try to do?

Expected answers:

1. Model A.
2. It assigns 0.70 to the observed correct class, while Model B assigns 0.30.
3. Increase probability of the correct class.

---

# Part 3 — Log-Likelihood

**Time:** 8 minutes

## 3.1 Why Logs?

For many data points, likelihood multiplies many probabilities.

[
L(\theta)
=========

P*\theta(y_1 \mid x_1)
P*\theta(y*2 \mid x_2)
\cdots
P*\theta(y_n \mid x_n)
]

But probabilities are between 0 and 1.

Multiplying many small numbers becomes extremely tiny.

Example:

[
0.8^{1000}
]

is a very small number.

Computers do not handle extremely tiny numbers well.

Also, sums are easier to optimize than products.

So we use logs.

---

## 3.2 Log Turns Products into Sums

The key identity:

[
\log(ab) = \log(a) + \log(b)
]

So instead of maximizing likelihood:

[
\prod_{i=1}^{n} P_\theta(y_i \mid x_i)
]

we maximize log-likelihood:

[
\sum_{i=1}^{n}
\log P_\theta(y_i \mid x_i)
]

This is easier numerically and mathematically.

---

## 3.3 Numerical Example

Suppose the probabilities assigned to correct labels are:

| Example | Correct Probability |
| ------- | ------------------: |
| 1       |                0.80 |
| 2       |                0.60 |
| 3       |                0.50 |

Likelihood:

[
0.80 \cdot 0.60 \cdot 0.50 = 0.24
]

Log-likelihood:

[
\log(0.80) + \log(0.60) + \log(0.50)
]

Using natural logs:

[
\log(0.80) \approx -0.223
]

[
\log(0.60) \approx -0.511
]

[
\log(0.50) \approx -0.693
]

So:

[
\log L \approx -1.427
]

The value is negative because logs of probabilities less than 1 are negative.

---

## 3.4 Higher Probability Means Less Negative Log

Important intuition:

| Probability of correct answer | (\log(p)) |
| ----------------------------: | --------: |
|                          1.00 |         0 |
|                          0.90 |    -0.105 |
|                          0.50 |    -0.693 |
|                          0.10 |    -2.303 |
|                          0.01 |    -4.605 |

The higher the probability of the correct answer, the closer (\log(p)) is to 0.

The lower the probability, the more negative it becomes.

So maximizing log-likelihood rewards the model for assigning high probability to the correct answer.

---

## 3.5 LLM Connection

For an LLM, training can be thought of as maximizing:

[
\sum_t \log P_\theta(x_t \mid x_1, ..., x_{t-1})
]

In words:

> Add up the log probabilities of all correct next tokens.

The model gets better when it assigns higher log probability to the actual next tokens in the training data.

---

## 3.6 Mini-Exercise

Ask:

Which is better log-likelihood for the correct answer?

Model A:

[
\log(0.8)
]

Model B:

[
\log(0.2)
]

Expected answer:

Model A, because 0.8 is larger than 0.2, and (\log(0.8)) is less negative than (\log(0.2)).

Teaching point:

> Larger probability of the correct answer means larger log-likelihood.

---

# Part 4 — Negative Log-Likelihood

**Time:** 8 minutes

## 4.1 Why Negative?

Machine learning training is usually framed as **minimization**.

We minimize loss.

But likelihood is something we want to maximize.

So we convert:

[
\max \log L(\theta)
]

into:

[
\min -\log L(\theta)
]

This gives us **negative log-likelihood**, often abbreviated as NLL.

For one example:

[
\text{NLL} = -\log P_\theta(y_{\text{true}} \mid x)
]

---

## 4.2 Intuition

If the model assigns high probability to the correct answer, the loss is low.

If the model assigns low probability to the correct answer, the loss is high.

| Probability of correct answer | Negative log-likelihood |
| ----------------------------: | ----------------------: |
|                          1.00 |                       0 |
|                          0.90 |                   0.105 |
|                          0.50 |                   0.693 |
|                          0.10 |                   2.303 |
|                          0.01 |                   4.605 |

So:

[
-\log(0.9)
]

is small.

[
-\log(0.01)
]

is large.

This creates a strong penalty for confidently assigning low probability to the correct answer.

---

## 4.3 Why This Loss Makes Sense

Negative log-likelihood has desirable behavior:

1. It rewards high probability on the correct answer.
2. It heavily punishes very low probability on the correct answer.
3. It turns probability training into a standard loss minimization problem.
4. It works naturally with softmax outputs.
5. It scales well to large datasets.

Core sentence:

> Negative log-likelihood is the loss version of maximum likelihood.

---

## 4.4 Example

Correct class:

[
\text{cat}
]

Model A:

[
P(\text{cat}) = 0.8
]

Model B:

[
P(\text{cat}) = 0.2
]

Losses:

[
-\log(0.8) \approx 0.223
]

[
-\log(0.2) \approx 1.609
]

Model A has lower loss.

Training pushes the model toward Model A-like behavior.

---

## 4.5 NLL for a Dataset

For a dataset:

[
\mathcal{D} = {(x_i,y_i)}_{i=1}^n
]

negative log-likelihood is:

[
-\sum_{i=1}^n \log P_\theta(y_i \mid x_i)
]

Often, we use the average:

[
-\frac{1}{n}\sum_{i=1}^n \log P_\theta(y_i \mid x_i)
]

This is the average penalty for not assigning enough probability to the correct answer.

---

## 4.6 LLM Version

For an LLM, the loss is usually averaged over many tokens:

[
-\frac{1}{T}\sum_{t=1}^{T}
\log P_\theta(x_t \mid x_1, ..., x_{t-1})
]

In words:

> Penalize the model when it assigns low probability to the actual next token.

This is why LLM training is often called **next-token prediction**.

---

## 4.7 Mini-Exercise

Correct token: “Paris.”

Model A:

[
P(\text{Paris}) = 0.75
]

Model B:

[
P(\text{Paris}) = 0.05
]

Ask:

1. Which model has lower negative log-likelihood?
2. Which model would training prefer?
3. Why is the penalty much larger for 0.05?

Expected answers:

1. Model A.
2. Model A.
3. Because (-\log(p)) grows rapidly when (p) becomes very small.

---

# Part 5 — Cross-Entropy Loss

**Time:** 12 minutes

## 5.1 From NLL to Cross-Entropy

Cross-entropy is one of the central loss functions in deep learning.

For classification, cross-entropy often becomes the same thing as negative log-likelihood.

The target distribution is usually **one-hot**.

Example: correct class is cat.

| Class | Target probability |
| ----- | -----------------: |
| cat   |                  1 |
| dog   |                  0 |
| car   |                  0 |

The model predicts:

| Class | Model probability |
| ----- | ----------------: |
| cat   |              0.70 |
| dog   |              0.20 |
| car   |              0.10 |

Cross-entropy compares the target distribution to the model distribution.

---

## 5.2 Cross-Entropy Formula

For target distribution (p) and model distribution (q):

[
H(p,q) = -\sum_i p_i \log q_i
]

Where:

- (p_i) is the target probability,
- (q_i) is the model probability.

For one-hot targets, only the correct class has (p_i = 1).

So if the correct class is cat:

[
H(p,q)
======

-\log q\_{\text{cat}}
]

That is exactly negative log-likelihood.

Core point:

> With one-hot labels, cross-entropy is just negative log probability of the correct class.

---

## 5.3 Numerical Example

Correct class: cat.

Target:

| Class | Target (p) |
| ----- | ---------: |
| cat   |          1 |
| dog   |          0 |
| car   |          0 |

Model:

| Class | Model (q) |
| ----- | --------: |
| cat   |      0.70 |
| dog   |      0.20 |
| car   |      0.10 |

Cross-entropy:

[
H(p,q)
======

-\left[
1\cdot \log(0.70)

- 0\cdot \log(0.20)
- 0\cdot \log(0.10)
  \right]
  ]

[
H(p,q) = -\log(0.70)
]

[
H(p,q) \approx 0.357
]

Only the probability assigned to the correct class matters directly in the one-hot case.

But indirectly, increasing the correct class probability means decreasing some other probabilities because they must sum to 1.

---

## 5.4 Why Cross-Entropy Punishes Confident Mistakes

Compare three models for correct class cat.

| Model | (P(\text{cat})) | Loss (-\log P(\text{cat})) |
| ----- | --------------: | -------------------------: |
| A     |            0.90 |                      0.105 |
| B     |            0.50 |                      0.693 |
| C     |            0.01 |                      4.605 |

Model C is very confidently wrong.

Cross-entropy strongly punishes this.

This is useful because a model that assigns near-zero probability to the true answer is making a serious probabilistic error.

---

## 5.5 Cross-Entropy in LLMs

In an LLM, each token prediction is a large classification problem.

The classes are vocabulary tokens.

If the vocabulary has 50,000 tokens, the model outputs a probability distribution over 50,000 possible next tokens.

At each training position:

- target is the actual next token,
- model predicts a probability distribution,
- cross-entropy penalizes low probability assigned to the actual next token.

Example:

Context:

> The capital of France is

Correct next token:

> Paris

Model distribution:

| Token        | Probability |
| ------------ | ----------: |
| Paris        |        0.82 |
| Lyon         |        0.04 |
| London       |        0.03 |
| other tokens |        0.11 |

Loss:

[
-\log P(\text{Paris} \mid \text{context})
]

[
-\log(0.82) \approx 0.198
]

If the model assigned only 0.01 to Paris, the loss would be much larger:

[
-\log(0.01) \approx 4.605
]

---

## 5.6 Cross-Entropy vs Accuracy

This is a useful practical distinction.

Accuracy only asks:

> Did the model choose the correct class?

Cross-entropy asks:

> How much probability did the model assign to the correct class?

Example:

Correct class: cat.

Model A:

| Class | Probability |
| ----- | ----------: |
| cat   |        0.51 |
| dog   |        0.49 |

Model B:

| Class | Probability |
| ----- | ----------: |
| cat   |        0.99 |
| dog   |        0.01 |

Both are accurate if we choose the highest probability class.

But Model B has much lower cross-entropy loss.

Cross-entropy contains information about confidence.

---

## 5.7 Soft Labels

Mention briefly.

Sometimes the target is not one-hot.

Example:

| Class | Target probability |
| ----- | -----------------: |
| cat   |               0.70 |
| dog   |               0.20 |
| fox   |               0.10 |

This can happen in:

- label smoothing,
- uncertain human labels,
- knowledge distillation,
- ambiguous classification.

Then cross-entropy uses the full target distribution:

[
H(p,q) = -\sum_i p_i \log q_i
]

This is useful because reality is often not perfectly one-hot.

---

## 5.8 Cross-Entropy, KL Divergence, and Matching Distributions

For this crash course, keep this as a preview.

Cross-entropy is closely related to making the model distribution (q) similar to the target distribution (p).

A related quantity is KL divergence:

[
D_{\mathrm{KL}}(p \parallel q)
]

Intuition:

> KL divergence measures the extra cost of using the model distribution instead of the true distribution.

Do not teach KL in detail here unless students are ready.

Just give the conceptual bridge:

- cross-entropy trains the model distribution,
- KL divergence measures mismatch between distributions,
- these ideas appear later in RL, variational inference, and diffusion.

---

## 5.9 Mini-Exercise

Correct class: dog.

Model predictions:

| Class | Model A | Model B |
| ----- | ------: | ------: |
| cat   |    0.10 |    0.40 |
| dog   |    0.80 |    0.30 |
| car   |    0.10 |    0.30 |

Ask:

1. Which model has lower cross-entropy?
2. What is the loss for Model A?
3. What is the loss for Model B?
4. Which model would training prefer?

Solution:

Model A:

[
-\log(0.80) \approx 0.223
]

Model B:

[
-\log(0.30) \approx 1.204
]

Training prefers Model A.

---

# Part 6 — Entropy and Uncertainty

**Time:** 8 minutes

## 6.1 What Is Entropy?

Entropy measures uncertainty in a probability distribution.

For a discrete distribution:

[
H(p) = -\sum_i p_i \log p_i
]

But start with intuition:

> Entropy is high when probability is spread out across many possibilities.
> Entropy is low when probability is concentrated on a few possibilities.

---

## 6.2 Low Entropy

Example:

| Token | Probability |
| ----- | ----------: |
| cat   |        0.95 |
| dog   |        0.03 |
| car   |        0.02 |

This distribution has low entropy.

Why?

Because the model is very confident.

Most of the probability mass is on “cat.”

---

## 6.3 High Entropy

Example:

| Token | Probability |
| ----- | ----------: |
| cat   |        0.34 |
| dog   |        0.33 |
| car   |        0.33 |

This distribution has high entropy.

Why?

Because the probability is spread out.

The model is uncertain.

---

## 6.4 Entropy vs Variance

Lecture 1 introduced variance as spread for numerical random variables.

Entropy is more general for categorical distributions.

For LLMs, labels, and tokens, entropy is often more natural than variance.

| Concept  | Best for                              |
| -------- | ------------------------------------- |
| Variance | numerical outcomes                    |
| Entropy  | categorical probability distributions |

Example:

For token prediction, it is awkward to ask for the variance of “cat/dog/car.”

But it is natural to ask:

> How uncertain is the token distribution?

That is entropy.

---

## 6.5 Entropy in LLMs

In LLMs:

- low entropy means the next token is obvious,
- high entropy means many continuations are plausible.

Example:

Low entropy context:

> The capital of France is ...

Likely next token:

> Paris

High entropy context:

> I think that ...

Many continuations are possible.

This explains why some prompts lead to stable answers and others lead to diverse outputs.

---

## 6.6 Temperature Preview

Temperature modifies the sharpness of the distribution during sampling.

Low temperature:

- makes high-probability tokens even more dominant,
- reduces randomness,
- produces more deterministic output.

High temperature:

- spreads probability more evenly,
- increases randomness,
- produces more diverse output.

Important:

> Temperature changes how we sample from the model. It does not change what the model has learned.

Temperature belongs more fully in Lecture 5, but this is a good preview.

---

## 6.7 Entropy in RL

Entropy also appears in reinforcement learning.

A policy with low entropy chooses one action almost always.

A policy with high entropy explores many actions.

Example:

Low-entropy policy:

| Action | Probability |
| ------ | ----------: |
| left   |        0.98 |
| right  |        0.01 |
| wait   |        0.01 |

High-entropy policy:

| Action | Probability |
| ------ | ----------: |
| left   |        0.34 |
| right  |        0.33 |
| wait   |        0.33 |

In RL, high entropy can help exploration.

This prepares students for Lecture 4.

---

## 6.8 Mini-Exercise

Which distribution has higher entropy?

Distribution A:

| Token | Probability |
| ----- | ----------: |
| yes   |        0.95 |
| no    |        0.03 |
| maybe |        0.02 |

Distribution B:

| Token | Probability |
| ----- | ----------: |
| yes   |        0.40 |
| no    |        0.35 |
| maybe |        0.25 |

Expected answer:

Distribution B has higher entropy because its probability mass is more spread out.

---

# Part 7 — Lecture Summary

**Time:** 4 minutes

## Core Ideas Students Should Remember

1. **Neural networks often output logits.**

Logits are raw scores.

They are not probabilities.

---

2. **Softmax turns logits into probabilities.**

[
P(y_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
]

Softmax ensures:

[
P(y_i) > 0
]

and:

[
\sum_i P(y_i)=1
]

---

3. **Likelihood measures how probable the observed data is under the model.**

For one example:

[
P_\theta(y_{\text{true}} \mid x)
]

For many examples:

[
\prod_i P_\theta(y_i \mid x_i)
]

---

4. **Training often maximizes likelihood.**

The model should assign high probability to what actually occurred in the data.

---

5. **Log-likelihood turns products into sums.**

[
\log \prod*i P*\theta(y_i \mid x_i)
===================================

\sum*i \log P*\theta(y_i \mid x_i)
]

This is easier numerically and mathematically.

---

6. **Negative log-likelihood turns maximization into minimization.**

[
-\log P_\theta(y_{\text{true}} \mid x)
]

Low probability for the correct answer creates high loss.

---

7. **Cross-entropy is the standard classification and next-token loss.**

With one-hot labels:

[
\text{cross-entropy}
====================

-\log P\_\theta(\text{correct class} \mid x)
]

For LLMs:

[
-\log P_\theta(\text{correct next token} \mid \text{context})
]

---

8. **Entropy measures uncertainty.**

Low entropy:

> The model is confident.

High entropy:

> The model is uncertain.

---

# Board Summary

A compact final board could look like this:

[
\text{input } x
\rightarrow
\text{neural network}
\rightarrow
\text{logits } z
\rightarrow
\text{softmax}
\rightarrow
P(y \mid x)
]

[
P(y_i \mid x) = \frac{e^{z_i}}{\sum_j e^{z_j}}
]

[
L(\theta) = \prod_i P_\theta(y_i \mid x_i)
]

[
\log L(\theta) = \sum_i \log P_\theta(y_i \mid x_i)
]

[
\text{NLL} = -\sum_i \log P_\theta(y_i \mid x_i)
]

[
H(p,q) = -\sum_i p_i \log q_i
]

[
H(p) = -\sum_i p_i \log p_i
]

And the AI translation:

| Probability Concept     | Deep Learning Meaning                    |
| ----------------------- | ---------------------------------------- |
| Logit                   | raw neural-network score                 |
| Softmax                 | converts scores to probabilities         |
| Likelihood              | probability assigned to observed data    |
| Log-likelihood          | easier version of likelihood to optimize |
| Negative log-likelihood | loss to minimize                         |
| Cross-entropy           | standard classification / LLM loss       |
| Entropy                 | uncertainty of the model’s distribution  |

---

# Suggested Running Examples

Use a few recurring examples.

## Running Example 1: Image Classifier

Input:

> image of a cat

Logits:

| Class | Logit |
| ----- | ----: |
| cat   |   2.0 |
| dog   |   1.0 |
| car   |   0.0 |

Softmax probabilities:

| Class | Probability |
| ----- | ----------: |
| cat   |        0.67 |
| dog   |        0.24 |
| car   |        0.09 |

Correct class:

[
\text{cat}
]

Loss:

[
-\log(0.67)
]

Use this for:

- logits,
- softmax,
- likelihood,
- NLL,
- cross-entropy.

---

## Running Example 2: LLM Next Token

Context:

> The capital of France is

Correct token:

> Paris

Model distribution:

| Token  | Probability |
| ------ | ----------: |
| Paris  |        0.82 |
| Lyon   |        0.04 |
| London |        0.03 |
| other  |        0.11 |

Loss:

[
-\log P(\text{Paris} \mid \text{The capital of France is})
]

Use this for:

- next-token prediction,
- likelihood,
- cross-entropy,
- entropy,
- temperature preview.

---

## Running Example 3: RL Policy Network

State:

> robot is at a junction

Policy logits:

| Action | Logit |
| ------ | ----: |
| left   |   1.5 |
| right  |   0.2 |
| wait   |  -0.7 |

Softmax gives:

[
\pi(a \mid s)
]

Use this for:

- policy as distribution,
- action sampling,
- entropy and exploration,
- bridge to Lecture 4.

---

# Suggested Exercises

## Exercise 1 — Logits vs Probabilities

A model outputs:

| Class | Logit |
| ----- | ----: |
| cat   |   4.0 |
| dog   |   2.0 |
| car   |  -1.0 |

Questions:

1. Are these probabilities?
2. Which class has the highest predicted probability after softmax?
3. Why do we need softmax?
4. Can logits be negative?

Expected answers:

1. No.
2. Cat.
3. To convert arbitrary scores into positive values that sum to 1.
4. Yes.

---

## Exercise 2 — Negative Log-Likelihood

Correct class: dog.

Model A:

[
P(\text{dog}) = 0.90
]

Model B:

[
P(\text{dog}) = 0.20
]

Questions:

1. Which model has higher likelihood?
2. Which model has lower NLL?
3. Which model would training prefer?
4. Why?

Expected answers:

1. Model A.
2. Model A.
3. Model A.
4. It assigns higher probability to the correct class.

---

## Exercise 3 — Cross-Entropy

Correct class: car.

Model distribution:

| Class | Probability |
| ----- | ----------: |
| cat   |        0.20 |
| dog   |        0.30 |
| car   |        0.50 |

Questions:

1. What is the target distribution?
2. What is the cross-entropy loss?
3. What would happen to the loss if (P(\text{car})) increased?

Expected answers:

1. ([0,0,1]) for cat, dog, car.
2. (-\log(0.50) \approx 0.693).
3. The loss would decrease.

---

## Exercise 4 — Entropy

Which distribution is more uncertain?

Distribution A:

| Token  | Probability |
| ------ | ----------: |
| Paris  |        0.90 |
| Lyon   |        0.05 |
| London |        0.05 |

Distribution B:

| Token  | Probability |
| ------ | ----------: |
| Paris  |        0.40 |
| Lyon   |        0.30 |
| London |        0.30 |

Expected answer:

Distribution B is more uncertain and has higher entropy.

---

# What to Emphasize Most

The most important concepts in Lecture 3 are:

1. **Logits are not probabilities.**
2. **Softmax converts scores into a distribution.**
3. **Likelihood means probability of observed data under the model.**
4. **Training increases probability of correct outputs.**
5. **Negative log-likelihood is the loss version of maximum likelihood.**
6. **Cross-entropy is the practical loss used for classification and LLM next-token prediction.**
7. **Entropy measures uncertainty in the model’s predicted distribution.**

The central educational move is:

[
P(y \mid x)
]

is not just notation.

A neural network actually produces this through:

[
\text{logits} \rightarrow \text{softmax} \rightarrow \text{probabilities}
]

and learns it through:

[
\text{cross-entropy loss}
]

---

# What Not to Overdo

Avoid spending too much time on:

- deriving softmax gradients,
- full information theory,
- KL divergence proofs,
- perplexity details,
- numerical stability tricks such as log-sum-exp,
- backpropagation mechanics,
- optimizer details,
- calibration methods,
- label smoothing details,
- transformer architecture.

Those are important, but Lecture 3 should stay focused on the probabilistic meaning of training.

A small mention is enough:

> In practice, libraries compute softmax and cross-entropy together using numerically stable methods.

But do not turn this into a programming lecture.

---

# Recommended Ending

End by connecting Lecture 3 to Lecture 4:

> Today we saw how supervised deep learning trains models to assign high probability to correct outputs. But not all AI problems are just prediction problems. In reinforcement learning, the model’s outputs are actions, those actions affect future states, and the goal is not just to predict correctly but to maximize expected future reward.

Final board line:

[
\text{Supervised learning: } \min -\log P(y \mid x)
]

[
\text{Reinforcement learning: } \max \mathbb{E}[\text{return}]
]

Then say:

> Lecture 4 moves from probabilistic prediction to probabilistic decision-making over time.
