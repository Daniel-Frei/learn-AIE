# Lecture 1 — Vectors, Geometry, and Dot Products

**Crash Course in Linear Algebra for LLMs, DL & RL**
**Duration:** 60 minutes
**Theme:** _Modern AI is geometry in high dimensions._

---

# Lecture Objectives

By the end of this lecture, students should be able to:

- Interpret vectors as geometric objects and feature representations
- Understand the dot product algebraically and geometrically
- Understand cosine similarity and why it matters in embeddings
- Compute and interpret L2 norms
- Interpret matrix–vector multiplication as a linear transformation
- Connect all of the above to attention, embeddings, and value functions

---

# Structure Overview (Time Plan)

| Section | Topic                        | Time   |
| ------- | ---------------------------- | ------ |
| 1       | What is a vector in ML?      | 10 min |
| 2       | Geometry of vectors          | 10 min |
| 3       | Dot product (deep intuition) | 20 min |
| 4       | Norms and distances          | 10 min |
| 5       | Matrix–vector multiplication | 8 min  |
| 6       | Wrap-up & ML connections     | 2 min  |

---

# 1. What is a Vector in Machine Learning? (10 min)

### 1.1 From High School to High Dimensions

Start simple:

[
x = \begin{bmatrix}2 \ -1\end{bmatrix}
]

Interpretations:

- A point in 2D
- A direction from the origin
- A list of features

Now generalize:

[
x \in \mathbb{R}^n
]

In ML:

- Image → vector of pixels
- Word → embedding vector
- State in RL → feature vector
- Patient record → feature vector

Emphasize:

> In AI, almost everything becomes a vector.

---

### 1.2 Row vs Column Vectors

Important for later neural network notation.

Column vector:

[
x =
\begin{bmatrix}
x_1 \
x_2 \
\vdots \
x_n
\end{bmatrix}
]

Row vector:

[
x^T = [x_1 ; x_2 ; ... ; x_n]
]

Explain:

- Convention in DL: inputs are usually column vectors
- ( x^T y ) gives dot product

---

### Mini Exercise (2 min)

If a word embedding has 768 dimensions, what does that mean geometrically?

Expected understanding:
→ It’s a point in 768-dimensional space.

---

# 2. Geometry of Vectors (10 min)

### 2.1 Length (Magnitude)

Define:

[
|x| = \sqrt{x_1^2 + x_2^2 + ... + x_n^2}
]

Example:

[
x = \begin{bmatrix}3 \ 4\end{bmatrix}
]

[
|x| = 5
]

Connect to:

- Pythagoras
- Euclidean distance

---

### 2.2 Direction

Two vectors can:

- Point in same direction
- Be orthogonal
- Be opposite

Introduce concept of angle between vectors.

This leads directly to dot product.

---

# 3. Dot Product — The Most Important Operation in AI (20 min)

This is the core of the lecture.

---

## 3.1 Algebraic Definition

[
a \cdot b = a^T b = \sum_{i=1}^n a_i b_i
]

Example:

[
a = \begin{bmatrix}1 \ 2\end{bmatrix}, \quad
b = \begin{bmatrix}3 \ 4\end{bmatrix}
]

[
a \cdot b = 1\cdot3 + 2\cdot4 = 11
]

---

## 3.2 Geometric Interpretation

[
a \cdot b = |a| |b| \cos \theta
]

This is critical.

Interpretation:

- Measures alignment
- Large positive → similar direction
- Zero → orthogonal
- Negative → opposite direction

This is what attention uses.

---

## 3.3 Projection Interpretation

The dot product tells you:

> How much of vector b lies in direction of vector a.

This is why:

- Neurons compute weighted sums
- Attention scores similarity
- Value functions evaluate states

---

## 3.4 Why Dot Product Appears Everywhere

### In Neural Networks

Single neuron:

[
y = w^T x + b
]

This is a dot product.

Interpretation:

- The neuron “measures alignment” between weights and input.

---

### In Attention

[
\text{Attention Score} = Q K^T
]

Why?

Because:

- Query asks “what am I looking for?”
- Key represents “what I contain”
- Dot product measures match

---

### In Embeddings

Cosine similarity:

[
\frac{a \cdot b}{|a| |b|}
]

Used in:

- Retrieval
- RAG systems
- Semantic search

---

### Mini Exercise

If two vectors are orthogonal, what is their dot product?

Answer:
0

What does that mean in embeddings?
→ Completely unrelated features.

---

# 4. Norms and Distance (10 min)

---

## 4.1 L2 Norm

[
|x|_2 = \sqrt{\sum x_i^2}
]

Interpret:

- Distance from origin
- Vector magnitude

---

## 4.2 Distance Between Two Vectors

[
|x - y|
]

This is Euclidean distance.

Used in:

- k-NN
- Clustering
- Contrastive learning

---

## 4.3 L1 vs L2 (Intuition Only)

L1:

[
\sum |x_i|
]

Explain:

- L2 penalizes large deviations more strongly
- L1 promotes sparsity

Mention:
Why L1 leads to sparse weights.

---

# 5. Matrix–Vector Multiplication (8 min)

Now connect to neural networks.

Given:

[
W \in \mathbb{R}^{m \times n}, \quad x \in \mathbb{R}^n
]

[
y = Wx
]

Interpretation:

- Each row of W is a weight vector.
- Each output is a dot product.

[
y_i = w_i^T x
]

This means:

> A matrix multiplication is just many dot products in parallel.

This is a neural layer.

---

## Geometric Interpretation

Matrix transforms space:

- Rotates
- Scales
- Shears

Neural network layer:

[
y = Wx + b
]

is a linear transformation + shift.

---

# 6. Wrap-Up — Connecting to AI (2 min)

Everything from this lecture appears in:

- Transformers (dot products in attention)
- Neural networks (weighted sums)
- Embeddings (cosine similarity)
- RL (value approximators compute dot products)

Key takeaway:

> AI models are geometric systems operating in high-dimensional vector spaces.

---

# Suggested Exercises (Optional Homework)

1. Compute cosine similarity between two vectors.
2. Show that orthogonal vectors have zero dot product.
3. Given weight vector and input vector, compute neuron output.
4. Normalize a vector and compute similarity again.

---

# Common Misconceptions to Address

- “Vectors are just lists of numbers.”
  → They represent geometry.

- “Dot product is just multiplication.”
  → It measures alignment.

- “High dimensions are abstract.”
  → Same geometry, just more axes.

---

# After This Lecture Students Should:

✔ Understand vector geometry
✔ Interpret dot products geometrically
✔ Understand cosine similarity
✔ Understand why neurons compute dot products
✔ Understand what a linear layer does
