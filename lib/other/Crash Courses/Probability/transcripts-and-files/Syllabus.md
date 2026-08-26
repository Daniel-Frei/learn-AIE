# Crash Course: Probability for AI

---

# Overall Structure

| Lecture | Topic                                                                | AI Connection                                       |
| ------- | -------------------------------------------------------------------- | --------------------------------------------------- |
| 1       | Probability foundations, distributions, expectation, and uncertainty | AI models as probability machines                   |
| 2       | Conditional, joint, total, and Bayesian probability                  | Inference, prediction, classification, transformers |
| 3       | Likelihood, loss functions, and entropy                              | Deep learning, softmax, cross-entropy               |
| 4       | Sequential probability, recursive expectation, and decisions         | Reinforcement learning, Markov processes            |
| 5       | Sampling, latent variables, noise, and diffusion                     | LLM decoding, VAEs, diffusion models                |

---

# Lecture 1 — Probability as the Language of AI

**Theme:** AI systems do not usually output “answers”; they output or manipulate **probability distributions**.

## Learning Goals

Students should understand:

- what probability means in AI
- how outcomes differ from events, which are sets of outcomes
- the three probability axioms and the rules they imply
- how complements, intersections, unions, and disjoint events combine
- how to count large sample spaces using the multiplication principle,
  factorials, permutations, and combinations
- what a distribution is
- what random variables are
- how discrete probability mass differs from continuous probability density
- what expectation means
- why expectations add even when random variables are dependent
- why probability is central to prediction, uncertainty, and learning

## Part 1 — Why Probability Matters in AI

Start with intuitive examples:

- Image classifier:
  “cat: 0.72, dog: 0.21, fox: 0.07”
- LLM next-token prediction:
  “the model assigns probabilities to possible next words”
- RL agent:
  “which action has the highest expected future reward?”
- Diffusion model:
  “start with noise and gradually sample a plausible image”

Core message:

> Probability is the mathematical language for uncertainty, prediction, learning from data, and making decisions under incomplete information.

## Part 2 — Events, Outcomes, and Probability

Explain:

- sample space
- event
- singleton event
- probability of an event
- probability between 0 and 1
- mutually exclusive events
- exhaustive events

Make the set structure explicit:

- complement, (A^c): “not (A)”
- intersection, (A \cap B): “(A) and (B)”
- union, (A \cup B): “(A) or (B), including both”
- disjoint events: events with no overlap

Introduce the three axioms:

1. (P(A) \ge 0)
2. (P(\Omega) = 1)
3. probabilities of disjoint events add

Then derive the working rules:

[
P(A^c) = 1 - P(A)
]

[
P(A \cup B) = P(A) + P(B) - P(A \cap B)
]

Use Venn diagrams and outcome lists to make double-counting visible. When using
“favorable outcomes divided by total outcomes,” state the required assumption:
the elementary outcomes must be equally likely.

Example:

A language model has a vocabulary of 50,000 tokens. At one step, it assigns a probability to every possible next token. The total probability over all tokens must sum to 1.

## Part 3 — Counting Large Sample Spaces

Students should be able to count outcomes without listing them all.

Use a two-question decision process:

1. Is repetition allowed?
2. Does order matter?

Cover:

- the multiplication principle for sequences of choices
- (n^k) when repetition is allowed and order matters
- factorial notation, (n!)
- permutations, (P(n,k) = n!/(n-k)!), when order matters without replacement
- combinations, ({n \choose k} = n!/[k!(n-k)]), when order does not matter

Examples should include PINs/passwords, ordered officer roles, committees, and
card hands. Emphasize that a counting formula is valid only after the outcome
definition, repetition rule, and role of order are clear.

## Part 4 — Random Variables

Introduce a random variable as a numerical or symbolic quantity whose value is uncertain.

Examples:

- (X =) next token
- (Y =) class label
- (R =) reward
- (S =) next state
- (Z =) latent representation
- (\epsilon =) noise in a diffusion model

Important distinction:

- the **outcome** is uncertain
- the **random variable** is a function that maps each outcome to a value
- the **distribution** describes how likely each outcome is

Distinguish:

- discrete random variables, described by a probability mass function (PMF)
- continuous random variables, described by a probability density function (PDF)

For a continuous variable, a single exact value has probability zero; interval
probabilities are areas under the density:

[
P(a \le X \le b) = \int_a^b f_X(x)\,dx
]

Use a uniform bread-length or arrival-time example so the area calculation is
simple geometry rather than a calculus exercise.

## Part 5 — Discrete Probability Distributions

Focus on discrete distributions first because they are central to classification and LLMs.

Examples:

- Bernoulli distribution: yes/no
- categorical distribution: one of many classes
- token distribution: one of many possible next tokens

For LLMs, the next token is a categorical random variable.

The model does not merely “choose a word.” It produces a distribution over possible words.

## Part 6 — Expectation

Expectation is the weighted average outcome under uncertainty.

Use simple examples:

- expected value of a die roll
- expected reward of an action
- expected loss of a model

AI connection:

> Training a model usually means minimizing expected error or expected loss over data.

In reinforcement learning:

> The agent chooses actions not just for immediate reward, but for expected future reward.

Add three important extensions:

- a zero expected-value game is fair in the long run
- expected value alone does not encode risk tolerance or how painful a rare loss
  is
- linearity of expectation does not require independence

[
\mathbb{E}[X+Y] = \mathbb{E}[X] + \mathbb{E}[Y]
]

Contrast that with multiplication: (\mathbb{E}[XY] =
\mathbb{E}[X]\mathbb{E}[Y]) is guaranteed under independence, not in general.

## Part 7 — Variance and Uncertainty

Introduce variance informally:

- expectation tells us the average
- variance tells us how spread out or uncertain the outcome is

AI examples:

- uncertain classifier prediction
- unstable reward in RL
- noisy observations
- generated outputs with randomness

## Part 8 — Lecture Summary

Students should leave with these core ideas:

1. AI systems often represent uncertainty using probability distributions.
2. A model prediction is often a probability distribution, not a single answer.
3. Expectation is the mathematical idea behind average loss, average reward, and average prediction.
4. Variance describes uncertainty or spread.
5. Probability is the foundation for learning from data.
6. Event algebra and counting make complex sample spaces manageable.
7. Discrete probability assigns mass to values; continuous probability assigns
   density whose area gives interval probability.
8. Expectations add whether or not the variables are independent.

---

# Lecture 2 — Conditional Probability, Bayes, and Dependence

**Theme:** AI is mostly about learning relationships between variables: what becomes more likely given something else?

## Learning Goals

Students should understand:

- conditional probability
- independence
- Bayes’ theorem
- joint and marginal probability
- why prediction means estimating conditional distributions
- the multiplication rule for dependent and independent events
- the law of total probability
- why base rates matter when interpreting test accuracy

## Part 1 — Conditional Probability

Introduce:

[
P(A \mid B)
]

Meaning:

> The probability of (A), given that (B) is known.

Examples:

- probability of disease given symptoms
- probability of spam given words in an email
- probability of next token given previous tokens
- probability of action success given current state

LLM connection:

[
P(\text{next token} \mid \text{previous tokens})
]

This is one of the most important probability expressions in modern AI.

## Part 2 — Joint Probability

Explain:

[
P(A, B)
]

Meaning:

> The probability that (A) and (B) both happen.

Example:

- probability that a patient has fever and infection
- probability that a token is “bank” and the previous sentence is about money
- probability of being in state (s) and taking action (a)

## Part 3 — Marginal Probability

Explain marginalization:

[
P(A) = \sum_B P(A, B)
]

Meaning:

> To get the probability of one thing, sum over the possible ways it can happen.

AI examples:

- summing over hidden causes
- marginalizing over latent variables
- considering all possible future states

## Part 4 — The Law of Total Probability

If (B_1,\ldots,B_n) form a mutually exclusive and exhaustive partition, then:

[
P(A) = \sum_i P(A \mid B_i)P(B_i)
]

Teach it as a path decomposition:

1. list every mutually exclusive way the outcome can occur
2. multiply along each path
3. add the path probabilities

Use the two-bag marble example, then connect it to disease base rates,
machine-specific defect rates, latent causes, and next-state uncertainty.

## Part 5 — Independence

Explain:

[
P(A \mid B) = P(A)
]

Meaning:

> Learning (B) gives no information about (A).

Give students three equivalent checks when the required probabilities are
defined:

[
P(A \cap B)=P(A)P(B)
]

[
P(A \mid B)=P(A)
]

[
P(B \mid A)=P(B)
]

For sequential events, show the general multiplication rule first:

[
P(A \cap B)=P(A)P(B \mid A)
]

Only simplify it to (P(A)P(B)) when independence is justified. Use sampling
with and without replacement as the central contrast.

Then explain why independence is rare in AI.

Examples:

- words in a sentence are not independent
- pixels in an image are not independent
- actions in a game affect future states
- symptoms are often statistically related

Important message:

> Much of machine learning is about discovering and exploiting dependence.

Also distinguish statistical from causal claims. Dependence can be produced by
a shared factor, and conditional independence may appear after conditioning on
that factor. Independence alone does not prove that no causal path or hidden
cause exists.

## Part 6 — Bayes’ Theorem

Introduce Bayes’ theorem:

[
P(H \mid D) = \frac{P(D \mid H)P(H)}{P(D)}
]

Explain the terms:

- (P(H)): prior belief
- (P(D \mid H)): likelihood of seeing the data if the hypothesis is true
- (P(H \mid D)): updated belief after seeing data
- (P(D)): normalization term

Compute (P(D)) with the law of total probability, rather than treating the
denominator as an unexplained quantity.

Use a medical-test example with natural frequencies (for example, 1,000 people)
to separate:

- sensitivity, (P(+ \mid \text{disease}))
- specificity, (P(- \mid \text{no disease}))
- false-positive rate
- prevalence/base rate
- positive predictive value, (P(\text{disease} \mid +))

This directly addresses the reversed-conditional/base-rate error: high test
accuracy does not imply an equally high posterior probability after a positive
result.

AI examples:

- diagnosis given symptoms
- classification given features
- updating beliefs after observations
- Bayesian approaches to uncertainty

## Part 7 — Prediction as Conditional Probability

Central AI framing:

[
P(y \mid x)
]

Most supervised learning can be framed as learning:

> Given input (x), what is the probability distribution over outputs (y)?

Examples:

- image → class
- text prefix → next token
- patient data → risk
- state → action value
- noisy image → clean image estimate

## Part 8 — Lecture Summary

Students should leave with these ideas:

1. Conditional probability is the core of prediction.
2. AI models often learn (P(y \mid x)).
3. LLMs learn something like (P(\text{next token} \mid \text{context})).
4. Bayes’ theorem formalizes belief updating.
5. Dependence between variables is what makes learning possible.
6. The law of total probability adds the probabilities of every possible path
   to an outcome.
7. Multiplication uses conditional probability unless independence has been
   established.
8. Base rates are essential when reversing a conditional with Bayes’ theorem.

---

# Lecture 3 — Likelihood, Loss, Softmax, and Deep Learning

**Theme:** Training neural networks is largely about making observed data more probable.

## Learning Goals

Students should understand:

- likelihood
- log-likelihood
- negative log-likelihood
- cross-entropy loss
- softmax
- why deep learning training is probabilistic

## Part 1 — From Model Scores to Probabilities

Neural networks often output raw numbers called **logits**.

Example:

| Token | Logit |
| ----- | ----: |
| cat   |   2.1 |
| dog   |   1.3 |
| car   |  -0.5 |

These are not probabilities yet.

Softmax converts logits into probabilities.

[
P(y_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
]

Explain:

- larger logits become larger probabilities
- all probabilities are positive
- probabilities sum to 1

AI connection:

- classifier output layer
- LLM next-token prediction
- policy networks in reinforcement learning

## Part 2 — Likelihood

Likelihood asks:

> How probable is the observed data under the model?

If the correct next token is “cat,” and the model assigns:

[
P(\text{cat}) = 0.7
]

that is better than assigning:

[
P(\text{cat}) = 0.05
]

The model is better when it assigns high probability to the things that actually happened in the training data.

## Part 3 — Log-Likelihood

Because multiplying many probabilities becomes numerically difficult, we usually use logs.

Instead of maximizing:

[
P(x_1)P(x_2)P(x_3)\cdots P(x_n)
]

we maximize:

[
\log P(x_1) + \log P(x_2) + \log P(x_3) + \cdots + \log P(x_n)
]

Logs turn products into sums.

This makes optimization easier.

## Part 4 — Negative Log-Likelihood

Deep learning usually minimizes loss, so instead of maximizing log-likelihood, we minimize negative log-likelihood:

[
-\log P(\text{correct answer})
]

Example:

- if the model gives high probability to the correct answer, loss is low
- if the model gives low probability to the correct answer, loss is high

This is one of the key ideas behind training classifiers and LLMs.

## Part 5 — Cross-Entropy Loss

Cross-entropy measures how far the model’s predicted distribution is from the target distribution.

For classification, the target distribution is often one-hot:

| Class | Target probability |
| ----- | -----------------: |
| cat   |                  1 |
| dog   |                  0 |
| car   |                  0 |

The model might predict:

| Class | Model probability |
| ----- | ----------------: |
| cat   |               0.7 |
| dog   |               0.2 |
| car   |               0.1 |

Cross-entropy punishes the model if it puts too little probability on the correct class.

LLM connection:

> Next-token prediction is usually trained with cross-entropy loss.

## Part 6 — Entropy and Uncertainty

Introduce entropy intuitively:

- low entropy: model is confident
- high entropy: model is uncertain

Example:

Low entropy:

| Token | Probability |
| ----- | ----------: |
| cat   |        0.95 |
| dog   |        0.03 |
| car   |        0.02 |

High entropy:

| Token | Probability |
| ----- | ----------: |
| cat   |        0.34 |
| dog   |        0.33 |
| car   |        0.33 |

LLM connection:

- high entropy means many tokens are plausible
- low entropy means the next token is obvious
- decoding temperature changes the randomness of sampling

## Part 7 — Lecture Summary

Students should leave with these ideas:

1. Neural networks often output logits.
2. Softmax turns logits into probabilities.
3. Likelihood measures how probable the observed data is under the model.
4. Training often means maximizing likelihood or minimizing negative log-likelihood.
5. Cross-entropy is central to classifiers and LLMs.
6. Entropy measures uncertainty in a distribution.

---

# Lecture 4 — Probability Over Time: Reinforcement Learning

**Theme:** Reinforcement learning is probability plus decisions over time.

## Learning Goals

Students should understand:

- sequential decision-making
- states, actions, rewards
- Markov property
- transition probabilities
- policy
- expected return
- value functions
- first-step equations and recursive expectation
- expected waiting times for repeated random trials
- exploration vs exploitation

## Part 1 — From Prediction to Action

Supervised learning:

[
x \rightarrow y
]

Reinforcement learning:

[
\text{state} \rightarrow \text{action} \rightarrow \text{reward} \rightarrow \text{next state}
]

The agent does not merely predict. It acts, receives feedback, and changes future situations.

## Part 2 — States, Actions, and Rewards

Define:

- state (s): current situation
- action (a): what the agent does
- reward (r): feedback signal
- next state (s'): situation after the action

Examples:

- robot navigation
- game playing
- recommendation systems
- chatbot policy optimization
- RLHF for LLMs

## Part 3 — Transition Probabilities

In RL, the environment is often stochastic.

[
P(s' \mid s, a)
]

Meaning:

> Given current state (s) and action (a), what is the probability of ending up in next state (s')?

Example:

A robot tries to move forward, but due to uncertainty it may:

- move forward with probability 0.8
- slip left with probability 0.1
- slip right with probability 0.1

## Part 4 — The Markov Property

The Markov assumption says:

> The future depends on the current state, not the full past history.

Formally:

[
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)
]

This simplifies decision-making.

Connection to AI:

- many RL algorithms assume Markov decision processes
- transformers, by contrast, often use long context because the current token representation depends on previous tokens

## Part 5 — Policies

A policy tells the agent how to act.

Deterministic policy:

[
a = \pi(s)
]

Stochastic policy:

[
\pi(a \mid s)
]

Meaning:

> Given state (s), the policy gives a probability distribution over actions.

AI connection:

- an RL agent samples actions from a policy
- an LLM samples tokens from a probability distribution
- RLHF adjusts a language model’s behavior using reward signals

## Part 6 — Expected Return

The agent wants to maximize expected cumulative reward.

[
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
]

Here (\gamma) is the discount factor.

Explain:

- immediate rewards matter
- future rewards also matter
- (\gamma) controls how much the future matters

High (\gamma): patient agent.
Low (\gamma): short-sighted agent.

## Part 7 — Value Functions

A value function estimates expected future reward.

State value:

[
V(s) = \mathbb{E}[G_t \mid S_t = s]
]

Action value:

[
Q(s,a) = \mathbb{E}[G_t \mid S_t = s, A_t = a]
]

Interpretation:

- (V(s)): how good is this state?
- (Q(s,a)): how good is this action in this state?

Deep RL connection:

> A neural network can approximate (Q(s,a)) or a policy (\pi(a \mid s)).

## Part 8 — Recursive Expectation and First-Step Analysis

Show how an expected value can be defined in terms of itself by conditioning on
the first random step.

For the number of fair-coin flips until the first head:

[
E = 1 + \frac{1}{2}(0) + \frac{1}{2}E
]

so (E=2). This replaces an infinite enumeration of tail sequences with one
state equation.

Then use two states—“no current head” and “one head so far”—to show why the
expected wait for two consecutive heads is six flips. Connect the technique to:

- geometric waiting times and convergent infinite probability series
- random walks and queueing models
- Markov states
- the recursive structure of value/Bellman equations

Core method:

1. define enough states to preserve the relevant information
2. condition on the next outcome
3. write one expectation equation per state
4. solve the simultaneous equations

## Part 9 — Exploration vs Exploitation

The agent faces a tradeoff:

- exploitation: choose the action currently believed to be best
- exploration: try uncertain actions to learn more

Probability enters because policies may deliberately randomize actions.

Examples:

- epsilon-greedy
- softmax action selection
- entropy regularization

## Part 10 — Lecture Summary

Students should leave with these ideas:

1. RL is about sequential decisions under uncertainty.
2. Transition probabilities describe how actions change the world.
3. A policy can be a probability distribution over actions.
4. The goal is to maximize expected future reward.
5. Value functions estimate expected return.
6. Exploration requires controlled randomness.
7. First-step analysis turns repeated or infinite random processes into
   solvable recursive expectation equations.

---

# Lecture 5 — Sampling, Latent Variables, and Diffusion Models

**Theme:** Modern generative AI is probability plus sampling.

## Learning Goals

Students should understand:

- sampling from distributions
- why generation is probabilistic
- temperature and randomness
- latent variables
- Gaussian noise
- diffusion models as learned denoising processes

## Part 1 — What Is Sampling?

A probability distribution gives possible outcomes and their probabilities.

Sampling means:

> Randomly drawing one outcome according to the distribution.

Example:

| Token | Probability |
| ----- | ----------: |
| cat   |         0.5 |
| dog   |         0.3 |
| car   |         0.2 |

If we sample many times, “cat” should appear roughly half the time.

LLM connection:

> Text generation is often sampling from the model’s next-token distribution.

## Part 2 — Greedy Decoding vs Sampling

Greedy decoding:

- always choose the most likely token
- more predictable
- can be repetitive or dull

Sampling:

- choose tokens probabilistically
- more diverse
- can be more creative
- can also become less reliable

This explains why LLMs can produce different answers to the same prompt.

## Part 3 — Temperature

Temperature changes how sharp or flat a probability distribution is.

Low temperature:

- makes high-probability tokens even more dominant
- more deterministic
- safer but less diverse

High temperature:

- gives lower-probability tokens more chance
- more creative
- more random and error-prone

Important AI intuition:

> Temperature does not make the model smarter. It changes how randomly we sample from what the model already believes.

## Part 4 — Latent Variables

A latent variable is a hidden variable that helps explain observed data.

Examples:

- topic behind a document
- style behind an image
- intent behind a user query
- hidden state of an environment
- compressed representation in a neural network

In generative modeling:

[
z \rightarrow x
]

A model may sample a hidden latent variable (z), then generate observed data (x).

This is central to models such as VAEs and many representation-learning methods.

## Part 5 — Gaussian Distributions and Noise

Introduce the normal distribution:

[
X \sim \mathcal{N}(\mu, \sigma^2)
]

Explain:

- (\mu): mean
- (\sigma^2): variance
- Gaussian noise is mathematically convenient
- many models inject or assume Gaussian noise

AI examples:

- noisy observations
- latent variables
- diffusion models
- optimization noise
- uncertainty modeling

## Part 6 — Diffusion Models: The Core Idea

Diffusion models use two processes:

### Forward process

Gradually add noise to data.

[
x_0 \rightarrow x_1 \rightarrow x_2 \rightarrow \cdots \rightarrow x_T
]

Eventually the data becomes almost pure noise.

### Reverse process

Train a neural network to reverse this process.

[
x_T \rightarrow x_{T-1} \rightarrow x_{T-2} \rightarrow \cdots \rightarrow x_0
]

The model learns:

> Given a noisy input, predict how to remove some noise.

## Part 7 — Why Diffusion Is Probabilistic

Diffusion models involve:

- random noise
- stochastic sampling
- learned probability distributions
- step-by-step generation

When generating an image, the model starts from noise and repeatedly denoises it toward something structured.

That is why the same prompt can produce different images.

## Part 8 — Connecting Diffusion to LLMs and RL

LLMs:

- generate discrete tokens
- sample from next-token distributions
- use cross-entropy training

Diffusion models:

- generate continuous data such as images
- sample through denoising
- often use Gaussian noise

RL:

- samples actions
- estimates expected return
- learns from stochastic environments

Shared probabilistic ideas:

- distributions
- conditioning
- expectation
- likelihood
- sampling
- entropy
- uncertainty

## Part 9 — Final Course Summary

By the end, students should understand that probability appears in AI in four major ways:

### 1. Prediction

Models estimate probabilities over outputs.

Example:

[
P(y \mid x)
]

### 2. Learning

Models are trained to assign high probability to observed data.

Example:

[
-\log P(\text{correct output})
]

### 3. Decision-making

Agents choose actions to maximize expected reward.

Example:

[
\mathbb{E}[G_t]
]

### 4. Generation

Models sample from learned distributions.

Examples:

- LLMs sample tokens
- diffusion models sample images
- policies sample actions

---

# What to Emphasize Most

The most important concepts for this audience are:

1. **Distributions**
2. **Conditional probability**
3. **Expectation**
4. **Likelihood**
5. **Cross-entropy**
6. **Sampling**
7. **Entropy**
8. **Markov processes**
9. **Expected return**
10. **Gaussian noise**

These are the probabilistic ideas that show up again and again in AI.

---

# Supplied-Lecture Coverage Map

This map records where every substantive concept from the supplied foundational
probability lecture belongs in the course. It is a curriculum coverage contract,
not a requirement to copy the source's examples or teaching order verbatim.

| Supplied lecture concept                                                     | Course placement |
| ---------------------------------------------------------------------------- | ---------------- |
| Sample spaces; outcomes versus events; singleton events                      | Lecture 1        |
| Non-negativity, normalization, and disjoint-additivity axioms                | Lecture 1        |
| Complements, intersections, unions, Venn diagrams, and the addition rule     | Lecture 1        |
| Multiplication principle, repetition, factorials, permutations, combinations | Lecture 1        |
| Random variables as functions from outcomes to values                        | Lecture 1        |
| Expected value, fair games, risk limitation, and linearity of expectation    | Lecture 1        |
| Discrete versus continuous variables; PMFs, PDFs, interval area, uniform law | Lecture 1        |
| Conditional probability as a restricted sample space                         | Lecture 2        |
| Joint tables, marginals, and marginalization                                 | Lecture 2        |
| Independence tests, dependence, multiplication rule, and common factors      | Lecture 2        |
| Law of total probability as mutually exclusive path decomposition            | Lecture 2        |
| Bayes’ theorem, priors, likelihood, evidence, posterior, and base rates      | Lecture 2        |
| Recursive first-step expectation, waiting times, and state equations         | Lecture 4        |

The lecture's examples—marbles, cards, coin flips, tests, games, continuous
measurements, and waiting-time problems—should appear as worked practice where
they clarify these concepts. AI examples then transfer the same reasoning to
classification, language modeling, reinforcement learning, latent variables,
and diffusion.

---
