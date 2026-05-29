# Lecture 2 — Matrices as Transformations

**Crash Course in Linear Algebra for LLMs, DL & RL**
**Duration:** 60 minutes
**Theme:** _Neural networks are stacked linear transformations._

---

# Lecture Objectives

By the end of this lecture, students should be able to:

- Understand matrices as transformations of space
- Interpret matrix multiplication geometrically
- Understand why order matters in matrix multiplication
- Understand shapes/dimensions in deep learning
- Understand rank intuitively
- Understand why low-rank structure matters in modern AI
- Understand transpose operations and symmetry
- Connect all concepts to transformers, LoRA, embeddings, and optimization

---

# Big Picture Narrative

Lecture 1 taught:

> Vectors are points/directions and dot products measure alignment.

Lecture 2 teaches:

> Matrices transform entire vector spaces.

This is the core abstraction of neural networks.

A neural network layer:

[
y = Wx + b
]

is fundamentally:

> A learned geometric transformation.

---

# Structure Overview (Time Plan)

| Section | Topic                                 | Time   |
| ------- | ------------------------------------- | ------ |
| 1       | Matrix multiplication intuition       | 15 min |
| 2       | Matrices as geometric transformations | 20 min |
| 3       | Rank and expressivity                 | 10 min |
| 4       | Transpose and symmetry                | 13 min |
| 5       | Wrap-up                               | 2 min  |

---

# 1. Matrix Multiplication (15 min)

---

# 1.1 Quick Refresher — What is a Matrix?

A matrix is a rectangular grid of numbers.

Example:

[
A =
\begin{bmatrix}
1 & 2 \
3 & 4
\end{bmatrix}
]

Interpretation:

- Not just data
- A transformation operator

---

# 1.2 Matrix–Vector Multiplication Revisited

Given:

[
A =
\begin{bmatrix}
1 & 2 \
3 & 4
\end{bmatrix},
\quad
x =
\begin{bmatrix}
5 \
6
\end{bmatrix}
]

Then:

[
Ax =
\begin{bmatrix}
1\cdot5 + 2\cdot6 \
3\cdot5 + 4\cdot6
\end{bmatrix}
=============

\begin{bmatrix}
17 \
39
\end{bmatrix}
]

Re-emphasize:

> Each row computes a dot product with the input vector.

This is:

- A neural layer
- A feature transformation
- A learned detector system

---

# 1.3 Matrix Multiplication as Composition

Now introduce:

[
AB
]

Key idea:

[
ABx = A(Bx)
]

Meaning:

1. First transform with (B)
2. Then transform result with (A)

This is composition of transformations.

---

# 1.4 Why Order Matters

Important:

[
AB \neq BA
]

in general.

Use geometric intuition:

- Rotate then stretch
- Stretch then rotate

Different results.

This matters in:

- Deep neural networks
- Attention blocks
- Sequential transformations

---

# 1.5 Dimensions and Shapes in DL

Critical practical section.

If:

[
A \in \mathbb{R}^{m \times n}
]

then:

- (m) = output dimension
- (n) = input dimension

And:

[
x \in \mathbb{R}^{n}
]

Then:

[
Ax \in \mathbb{R}^{m}
]

---

# Deep Learning Example

Suppose:

- 768-dimensional embedding
- Hidden layer size 2048

Then:

[
W \in \mathbb{R}^{2048 \times 768}
]

Meaning:

- Input: 768 features
- Output: 2048 features

---

# Batch Dimension

In practice:

[
X \in \mathbb{R}^{batch \times features}
]

Example:

[
X \in \mathbb{R}^{32 \times 768}
]

Meaning:

- 32 examples
- each with 768 features

Important practical intuition.

---

# Mini Exercise

If:

[
W \in \mathbb{R}^{128 \times 512}
]

what is the output dimension of:

[
Wx
]

for:

[
x \in \mathbb{R}^{512}
]

Answer:

[
128
]

---

# 2. Matrices as Geometric Transformations (20 min)

This is the conceptual core.

---

# 2.1 Matrix = Space Transformation

A matrix transforms:

- points
- directions
- coordinate systems

Not just numbers.

---

# 2.2 Scaling

Example:

[
A =
\begin{bmatrix}
2 & 0 \
0 & 3
\end{bmatrix}
]

Effects:

- x-direction stretched by 2
- y-direction stretched by 3

Connect to:

- Feature amplification
- Learned weighting

---

# 2.3 Rotation

Example:

[
R =
\begin{bmatrix}
\cos\theta & -\sin\theta \
\sin\theta & \cos\theta
\end{bmatrix}
]

Interpretation:

- Changes orientation
- Preserves distances

Important intuition:

- Embedding spaces can rotate without changing semantic structure

---

# 2.4 Shearing

Example:

[
A =
\begin{bmatrix}
1 & 1 \
0 & 1
\end{bmatrix}
]

Effect:

- Slants space

Explain:

- Coordinates influence each other
- Features mix together

This is important in neural networks:

- Learned features are mixtures of prior features

---

# 2.5 Projection

Projection removes information.

Example:
Project 3D object onto 2D plane.

Important in ML:

- Dimensionality reduction
- Bottlenecks
- Compression

---

# 2.6 Why Neural Networks Need Nonlinearities

Very important section.

Suppose:

[
y = A(Bx)
]

Then:

[
y = (AB)x
]

Still linear.

Meaning:

> Stacking linear layers without nonlinearities collapses into one linear transformation.

This is a major insight.

Therefore:

Neural networks require:

- ReLU
- GELU
- sigmoid
- tanh

Otherwise depth gives no extra expressive power.

---

# Deep Learning Connection

Transformer block:

[
x \to W_1 x \to \text{GELU} \to W_2 x
]

Nonlinearity creates expressive power.

---

# Mini Exercise

True or false:

A 100-layer neural network without activation functions is equivalent to a single linear layer.

Answer:
True.

---

# 3. Rank and Expressivity (10 min)

---

# 3.1 Intuitive Meaning of Rank

Rank measures:

> How many independent directions a matrix can represent.

---

# Full Rank

A full-rank matrix:

- preserves information
- spans full space

---

# Low Rank

Low-rank matrix:

- compresses information
- loses dimensions
- creates bottlenecks

---

# Geometric Interpretation

Imagine mapping:

- 3D → 2D plane
- many directions collapse together

This reduces rank.

---

# 3.2 Why Rank Matters in AI

Low-rank structure appears everywhere.

---

# Compression

Large neural networks often contain redundant structure.

Low-rank approximations:

- reduce parameters
- reduce memory
- reduce compute

---

# 3.3 LoRA Intuition

Very important modern example.

Instead of updating huge matrix:

[
W
]

LoRA learns:

[
W + AB
]

where:

- (A) and (B) are low-rank

Meaning:

- small number of learned directions
- efficient fine-tuning

---

# Key Insight

Large models often only need:

- small directional updates
- not full matrix changes

---

# 4. Transpose and Symmetry (13 min)

---

# 4.1 Transpose Operation

Transpose flips rows and columns.

[
A =
\begin{bmatrix}
1 & 2 \
3 & 4
\end{bmatrix}
]

[
A^T =
\begin{bmatrix}
1 & 3 \
2 & 4
\end{bmatrix}
]

---

# 4.2 Why Transpose Matters

Transpose changes:

- orientation
- multiplication compatibility

---

# Attention Example

[
QK^T
]

Why transpose?

Suppose:

- (Q): queries
- (K): keys

Each query vector needs dot products with ALL keys.

Transpose aligns dimensions properly.

---

# 4.3 Symmetric Matrices

Definition:

[
A = A^T
]

Symmetric matrices appear constantly.

---

# Covariance Matrix

[
\Sigma = X^T X
]

Properties:

- symmetric
- captures feature relationships

Important for:

- PCA
- representation learning
- statistics

---

# 4.4 Why Gradients Use Transpose

High-level intuition only.

During backprop:

- gradients flow backward through layers

Transpose appears because:

- backward propagation reverses transformations

Do NOT derive rigorously yet.

Just intuition.

---

# Mini Exercise

If:

[
A \in \mathbb{R}^{3 \times 5}
]

what shape is:

[
A^T
]

Answer:

[
5 \times 3
]

---

# 5. Wrap-Up (2 min)

Key idea:

> Neural networks are learned geometric transformation systems.

Everything in modern AI:

- transformers
- embeddings
- LoRA
- attention
- RL function approximators

depends on:

- matrix multiplication
- transformations
- rank
- transpose operations

---

# Suggested Exercises

1. Compute matrix-vector products
2. Determine output shapes
3. Compute transposes
4. Identify whether matrices are symmetric
5. Determine whether simple matrices are full-rank or low-rank
6. Explain why stacked linear layers collapse

---

# Common Misconceptions

---

### “Matrices are just tables of numbers.”

No.

Matrices transform spaces.

---

### “Deep networks become powerful just because they are deep.”

Not without nonlinearities.

---

### “Matrix multiplication is commutative.”

Usually false.

[
AB \neq BA
]

---

### “Low-rank means small values.”

No.

Rank concerns independent directions, not magnitude.

---

# After This Lecture Students Should:

✔ Understand matrices as transformations
✔ Understand composition of transformations
✔ Understand shapes in neural networks
✔ Understand why order matters
✔ Understand why nonlinearities are necessary
✔ Understand low-rank intuition and LoRA
✔ Understand transpose and symmetry
✔ Understand why attention uses (QK^T)
