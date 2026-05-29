# Lecture 5 — Linear Algebra in LLMs and RL

**Crash Course in Linear Algebra for LLMs, DL & RL**
**Duration:** 60 minutes
**Theme:** _Putting everything together._

---

# Lecture Objectives

By the end of this lecture, students should be able to:

- Understand the attention mechanism mathematically and geometrically
- Understand why transformers rely heavily on dot products and matrix multiplication
- Understand neural networks as stacks of learned transformations
- Understand how RL value functions are approximated using neural networks
- Understand optimization landscapes conceptually
- Understand why linear algebra dominates modern AI

---

# Big Picture Narrative

Lecture 1:

> Data becomes vectors.

Lecture 2:

> Neural networks are matrix transformations.

Lecture 3:

> Learning happens through gradients.

Lecture 4:

> Representations contain lower-dimensional structure.

Lecture 5:

> Modern AI systems combine all these ideas simultaneously.

This lecture ties together:

- vectors
- matrices
- gradients
- embeddings
- attention
- optimization
- representation learning

into one coherent mental model.

---

# Structure Overview (Time Plan)

| Section | Topic                            | Time   |
| ------- | -------------------------------- | ------ |
| 1       | Attention mechanism              | 20 min |
| 2       | Neural networks as matrix stacks | 10 min |
| 3       | RL value functions               | 10 min |
| 4       | Optimization intuition           | 10 min |
| 5       | Why linear algebra dominates AI  | 8 min  |
| 6       | Wrap-up                          | 2 min  |

---

# 1. Attention Mechanism (20 min)

This is the conceptual centerpiece.

---

# 1.1 Motivation — Why Attention Exists

Older models:

- compressed entire sentence into one vector
- struggled with long-range relationships

Attention solves this by allowing:

- tokens to dynamically reference other tokens

Core idea:

> Tokens decide which other tokens are relevant.

---

# 1.2 Input Embeddings

Suppose:

[
X \in \mathbb{R}^{n \times d}
]

Where:

- (n) = number of tokens
- (d) = embedding dimension

Each row:

- token embedding vector

---

# 1.3 Linear Projections

Transformer creates:

[
Q = XW_Q
]

[
K = XW_K
]

[
V = XW_V
]

Interpretation:

---

## Query (Q)

“What information am I looking for?”

---

## Key (K)

“What information do I contain?”

---

## Value (V)

“What information should be passed forward?”

---

# Key Insight

These are learned linear transformations.

Lecture 2 connection:

- matrices transform representation space

---

# 1.4 Dot Product Attention

Attention scores:

[
QK^T
]

This computes:

- dot products between queries and keys

---

# Geometric Interpretation

Dot product measures:

- alignment
- similarity
- relevance

Large dot product:

- strong match

Small/negative:

- weak match

---

# Why Transpose?

Suppose:

- (Q): shape (n \times d)
- (K): shape (n \times d)

Need:

- all pairwise similarities

So:

[
K^T \in \mathbb{R}^{d \times n}
]

Then:

[
QK^T \in \mathbb{R}^{n \times n}
]

Meaning:

- every token compared to every token

---

# 1.5 Softmax

Raw scores transformed via:

[
\text{softmax}(QK^T)
]

Softmax:

- converts scores into probabilities
- emphasizes strongest matches

---

# Geometric Interpretation

Attention becomes:

- weighted focus mechanism

Not hard selection:

- soft distribution over tokens

---

# 1.6 Weighted Sum of Values

Final attention output:

[
\text{softmax}(QK^T)V
]

Interpretation:

Tokens aggregate information from:

- relevant value vectors

This creates:

- contextualized embeddings

---

# 1.7 Multi-Head Attention Intuition

Different heads learn:

- different geometric relationships
- different semantic structures

Examples:

- syntax
- long-term dependencies
- entity tracking

Each head:

- different projection matrices
- different representation geometry

---

# LLM Connection

Transformers are fundamentally:

- matrix multiplications
- dot products
- learned geometric routing systems

---

# Mini Exercise

True or false:

Attention scores are based on Euclidean distance between queries and keys.

Answer:
False.

They use dot products.

---

# 2. Neural Networks as Matrix Stacks (10 min)

---

# 2.1 Basic Neural Network Structure

[
x \to W_1x \to \sigma \to W_2x \to \sigma \to ...
]

Interpretation:

- repeated learned transformations

---

# 2.2 Why Nonlinearities Matter

Lecture 2 recall:

Without nonlinearities:

[
W_2(W_1x)
=========

(W_2W_1)x
]

Still linear.

Therefore:

- depth alone insufficient

Nonlinearities create:

- expressive decision boundaries
- hierarchical representations

---

# 2.3 Representation Learning

Early layers:

- simple features

Later layers:

- abstract concepts

Example in LLMs:

- characters
- words
- syntax
- semantics
- reasoning patterns

---

# 2.4 Deep Learning as Geometry

Each layer:

- rotates
- stretches
- compresses
- separates representations

Neural networks:

- gradually reshape representation space

---

# Mini Exercise

True or false:

Stacking many linear layers without activations increases expressivity significantly.

Answer:
False.

---

# 3. RL Value Functions (10 min)

---

# 3.1 RL Core Idea

Agent interacts with environment:

- states
- actions
- rewards

Goal:

- maximize future reward

---

# 3.2 State as Vector

State:

[
s \in \mathbb{R}^n
]

Examples:

- robot sensor readings
- game board representation
- language context

---

# 3.3 Value Functions

Value estimate:

[
Q(s,a)
]

Meaning:

> Expected future reward for action (a) in state (s).

---

# 3.4 Neural Approximation

Modern RL uses neural networks:

[
Q_\theta(s,a)
]

Parameters:

- weights (\theta)

---

# 3.5 Connection to Linear Algebra

Neural network computes:

- repeated matrix multiplications
- dot products
- nonlinear transformations

---

# 3.6 Gradient-Based Learning

RL updates parameters using:

- gradient descent

Loss example:

[
L = (Q_{target}-Q_\theta)^2
]

Gradients adjust weights to:

- improve future predictions

---

# RL Intuition

Learning value functions:

- reshapes representation space
- aligns states with useful predictions

---

# Mini Exercise

True or false:

Modern deep RL often uses neural networks to approximate value functions.

Answer:
True.

---

# 4. Optimization Intuition (10 min)

---

# 4.1 Loss Landscape

Loss function:

[
L(W)
]

defines surface over parameter space.

Neural networks:

- extremely high-dimensional landscapes

---

# 4.2 Local Minima

Point where:

- nearby directions increase loss

Historically considered major issue.

Modern insight:

- less problematic in high dimensions

---

# 4.3 Saddle Points

Critical modern intuition.

Saddle point:

- uphill in some directions
- downhill in others

High-dimensional systems contain MANY saddle points.

---

# 4.4 Why High Dimensions Behave Differently

In high dimensions:

- true bad minima rarer
- many escape directions exist

Optimization often easier than expected.

---

# 4.5 Gradient Descent Intuition

Training:

- iterative movement through parameter space

Gradients guide:

- local improvement steps

---

# 4.6 Optimization in Transformers

Training large LLMs:

- enormous optimization problem
- billions of dimensions

Yet:

- gradient methods still work remarkably well

---

# Mini Exercise

True or false:

All critical points in neural network optimization are bad local minima.

Answer:
False.

Many are saddle points.

---

# 5. Bonus — Why Linear Algebra Dominates AI (8 min)

This section ties the entire course together.

---

# 5.1 Data = Vectors

Everything becomes vectors:

- text
- images
- audio
- states
- actions

---

# 5.2 Models = Matrices

Neural networks:

- matrix transformations

Transformers:

- giant structured matrix systems

---

# 5.3 Learning = Gradients

Optimization:

- derivatives
- gradients
- backpropagation

---

# 5.4 Representation = Geometry

Embeddings organize:

- semantics
- concepts
- relationships

through geometry.

---

# 5.5 Compression = Low Rank

Real-world data contains:

- redundancy
- structure
- correlated directions

Low-rank methods exploit this.

---

# 5.6 Similarity = Dot Products

Attention:

- dot products

Retrieval:

- cosine similarity

Embeddings:

- geometric alignment

---

# Final Core Insight

Modern AI is fundamentally:

> High-dimensional geometric optimization over vector spaces.

---

# 6. Wrap-Up (2 min)

Students should now understand:

✔ What (Wx) means
✔ Why transformers use (QK^T)
✔ Why gradients enable learning
✔ Why embeddings are geometric
✔ Why low-rank structure matters
✔ Why modern AI is dominated by linear algebra

They are now equipped to:

- read ML papers
- understand transformer internals
- reason about embeddings and optimization
- follow modern AI discussions without getting lost in notation

---

# Suggested Exercises

1. Determine attention score matrix shapes
2. Explain why (QK^T) uses transpose
3. Explain why nonlinearities are required
4. Describe value function approximation
5. Explain saddle points intuitively
6. Explain why embeddings are geometric

---

# Common Misconceptions

---

### “Attention is symbolic reasoning.”

No.

Attention is differentiable similarity routing using vector geometry.

---

### “LLMs store facts like databases.”

Not exactly.

Knowledge emerges through distributed geometric representations.

---

### “Deep learning is mostly statistics.”

Modern deep learning is heavily geometric and optimization-based.

---

### “High-dimensional spaces are completely different.”

The same core geometric principles still apply:

- dot products
- projections
- norms
- transformations
- gradients

just at much larger scale.
