# Course Philosophy

We focus only on concepts that:

- Directly appear in **neural networks**
- Are essential for **backprop**
- Explain **attention**
- Are needed for **PCA / embeddings**
- Show up in **RL value functions**
- Help understand **optimization**

No abstract vector spaces unless necessary.

---

# Lecture 1 — Vectors, Geometry, and Dot Products

**Theme: Geometry is everywhere in ML**

### Goals

By the end students should:

- Understand vectors as geometry
- Understand dot product deeply
- Understand cosine similarity
- Understand norms
- See why embeddings are geometry

---

## 1. Vectors as Points and Directions (15 min)

- 2D, 3D intuition
- n-dimensional vectors
- Column vs row vectors (important for NN notation)

Example:
[
x = \begin{bmatrix}2 \ -1\end{bmatrix}
]

Interpretation:

- Point
- Direction
- Feature vector

Connect to:

- Word embeddings
- State representations in RL

---

## 2. Dot Product (20 min)

[
a \cdot b = a^T b = \sum_i a_i b_i
]

Three interpretations:

1. Algebraic formula
2. Projection: ( |a||b|\cos\theta )
3. Similarity measure

Explain:

- Why attention uses (QK^T)
- Why cosine similarity is used in embedding retrieval
- Why larger dot product = “more aligned”

Practical:

- Why normalization matters

---

## 3. Norms (15 min)

L2 norm:

[
|x|_2 = \sqrt{\sum x_i^2}
]

Explain:

- Euclidean distance
- Why loss functions use L2
- Why gradient descent moves “downhill”

Mention:

- L1 vs L2 regularization (intuitive)

---

## 4. Matrix–Vector Multiplication (10 min)

Interpretation:

[
Wx
]

NOT just multiplication.

Explain as:

- Linear transformation
- Weighted sum of features
- Neural network layer

Example:
[
y = Wx + b
]

Connect to:

- Single neuron
- Linear model

---

# Lecture 2 — Matrices as Transformations

**Theme: Neural networks are stacked linear transformations**

---

## 1. Matrix Multiplication (15 min)

[
AB
]

Interpretation:

- Composition of transformations
- Why order matters

Important:

- Dimensions must align
- What shape means in DL

Example:

- Batch of inputs: ( X \in \mathbb{R}^{batch \times features} )

---

## 2. Linear Transformations (20 min)

Geometric intuition:

Matrix does:

- Rotation
- Scaling
- Shearing
- Projection

Explain:
Why neural networks without nonlinearities collapse into a single linear transform.

---

## 3. Rank and Expressivity (10 min)

Explain:

- What rank means (intuitively)
- Why low-rank ≈ information bottleneck
- Why LoRA works (low-rank updates)

---

## 4. Transpose and Symmetry (15 min)

[
A^T
]

Explain:

- Why (QK^T)
- Why covariance matrix is symmetric
- Why gradients use transpose

---

# Lecture 3 — Derivatives and Gradients

**Theme: Learning = optimization**

This is the most important lecture.

---

## 1. Scalar Derivatives (10 min)

Quick refresher:
[
\frac{d}{dx} x^2 = 2x
]

Interpret as slope.

---

## 2. Partial Derivatives (10 min)

For multivariable functions:

[
f(x,y)
]

What does derivative w.r.t. x mean?

---

## 3. Gradient Vector (15 min)

[
\nabla f(x)
]

Interpretation:

- Direction of steepest ascent
- Why we move opposite direction

Connect to:

- Gradient descent
- Loss minimization

---

## 4. Chain Rule (15 min)

Explain carefully:

[
\frac{d}{dx} f(g(x))
]

Then extend to vector case.

Explain:
Why backprop is repeated chain rule.

---

## 5. Matrix Derivatives (10 min)

High-level only.

Understand:

[
\frac{\partial}{\partial W}(Wx)
]

Main takeaway:

- Gradients have same shape as parameters.
- Backprop is efficient reuse of intermediate gradients.

---

# Lecture 4 — Eigenvectors, SVD, and Representation Learning

**Theme: Why embeddings and PCA work**

---

## 1. Eigenvectors (15 min)

[
Av = \lambda v
]

Interpretation:

- Direction preserved under transformation
- Scaling factor

Intuition:

- Stable directions
- Why power iteration works

---

## 2. Covariance Matrix (10 min)

Explain:

- Data spread
- Why covariance captures structure

---

## 3. PCA (15 min)

Explain:

- Find directions of maximum variance
- Dimensionality reduction
- Why embeddings can compress information

---

## 4. Singular Value Decomposition (20 min)

[
A = U \Sigma V^T
]

Interpret as:

- Rotation
- Scaling
- Rotation

Explain why:

- Low-rank approximation
- Why transformers can be compressed
- Why attention matrices are often low-rank

No heavy proofs.

Just geometric meaning.

---

# Lecture 5 — Linear Algebra in LLMs and RL

**Theme: Put everything together**

---

## 1. Attention Mechanism (20 min)

Given:

[
Q = XW_Q
]
[
K = XW_K
]
[
V = XW_V
]

Explain step-by-step:

1. Linear projections
2. Dot products (QK^T)
3. Softmax
4. Weighted sum of V

Tie back to:

- Dot product geometry
- Matrix multiplication
- Transpose

---

## 2. Neural Networks as Matrix Stacks (10 min)

[
x \to W_1 x \to \sigma \to W_2 \to ...
]

Explain:

- Why nonlinearity is necessary
- Why depth increases expressivity

---

## 3. RL Value Functions (10 min)

Explain:

- State vector
- Q(s,a) approximated by neural net
- Why gradient descent updates value estimates

Connect:

- Gradient from loss
- Dot product between weights and features

---

## 4. Optimization Intuition (10 min)

Explain:

- Loss landscape
- Local minima
- Saddle points
- Why high dimensions behave differently

---

## 5. Bonus: Why Linear Algebra Dominates AI (10 min)

Because:

- Data = vectors
- Models = matrices
- Learning = gradients
- Representation = eigenvectors
- Compression = low rank
- Similarity = dot product

---

# What Students Should Be Able To Do After 5 Hours

They should:

✓ Understand what (Wx) means in neural networks
✓ Understand why attention uses (QK^T)
✓ Understand gradient descent mathematically
✓ Understand cosine similarity in embeddings
✓ Understand PCA intuition
✓ Understand why low-rank ≈ compression
✓ Understand how value functions are approximated

They will NOT:

- Prove SVD
- Derive backprop rigorously
- Master tensor calculus

But they will no longer feel lost reading ML papers.

---

# If You Want To Make This Even Stronger

For each lecture include:

- 2–3 geometric visualizations
- 3 short exercises
- 1 ML application example
- 1 “common misconception” section
