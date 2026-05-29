# Lecture 4 — Eigenvectors, SVD, and Representation Learning

**Crash Course in Linear Algebra for LLMs, DL & RL**
**Duration:** 60 minutes
**Theme:** _Why embeddings, compression, and representation learning work._

---

# Lecture Objectives

By the end of this lecture, students should be able to:

- Understand eigenvectors and eigenvalues geometrically
- Understand covariance intuitively
- Understand why PCA finds “important directions”
- Understand dimensionality reduction conceptually
- Understand SVD geometrically
- Understand why low-rank structure matters in modern AI
- Understand why embeddings can compress information
- Connect these ideas to transformers, LoRA, attention, and representation learning

---

# Big Picture Narrative

Lecture 1:

> Vectors represent information geometrically.

Lecture 2:

> Matrices transform spaces.

Lecture 3:

> Learning adjusts transformations using gradients.

Lecture 4:

> Some directions in space matter more than others.

This is the foundation of:

- embeddings
- PCA
- compression
- latent representations
- low-rank adaptation
- semantic structure in AI systems

---

# Structure Overview (Time Plan)

| Section | Topic                              | Time   |
| ------- | ---------------------------------- | ------ |
| 1       | Eigenvectors and eigenvalues       | 15 min |
| 2       | Covariance matrices                | 10 min |
| 3       | PCA and dimensionality reduction   | 15 min |
| 4       | Singular Value Decomposition (SVD) | 18 min |
| 5       | Wrap-up                            | 2 min  |

---

# 1. Eigenvectors and Eigenvalues (15 min)

---

# 1.1 Motivation — What Happens When a Matrix Transforms Space?

Lecture 2 showed:

- matrices rotate
- stretch
- shear
- compress space

Question:

> Are there special directions that behave nicely under transformation?

Answer:
Yes.

These are eigenvectors.

---

# 1.2 Definition

[
Av = \lambda v
]

Where:

- (A) = matrix transformation
- (v) = eigenvector
- (\lambda) = eigenvalue

Meaning:

> After transformation, the vector still points in the same direction.

Only its magnitude changes.

---

# 1.3 Geometric Interpretation

Most vectors:

- rotate
- change direction

Eigenvectors:

- keep direction
- only scale

---

# Example Intuition

Imagine:

- stretching rubber sheet

Some directions:

- stretch strongly
- stretch weakly
- stay unchanged

These stable directions are eigenvectors.

---

# 1.4 Meaning of Eigenvalues

[
\lambda
]

tells us:

- how much scaling occurs

---

## Cases

### (\lambda > 1)

Direction expands.

---

### (0 < \lambda < 1)

Direction shrinks.

---

### (\lambda = 1)

No scaling.

---

### (\lambda < 0)

Direction flips.

---

# 1.5 Why This Matters in AI

Eigenvectors reveal:

- dominant directions
- stable structure
- important modes of variation

This appears in:

- PCA
- embeddings
- diffusion models
- optimization dynamics
- covariance analysis

---

# 1.6 Power Iteration Intuition

Suppose repeatedly apply matrix:

[
x, Ax, A^2x, A^3x, ...
]

Eventually:

- largest eigenvalue dominates
- vector aligns with principal eigenvector

This explains:

- why dominant semantic directions emerge
- why repeated transformations amplify major structure

---

# Deep Learning Connection

Transformer hidden states often contain:

- dominant latent directions
- principal semantic components

---

# Mini Exercise

True or false:

An eigenvector changes direction under transformation.

Answer:
False.

---

# 2. Covariance Matrix (10 min)

---

# 2.1 Motivation — Understanding Data Structure

Suppose dataset contains:

- height
- weight
- income
- age

Question:

> Which features vary together?

Covariance answers this.

---

# 2.2 Covariance Intuition

Positive covariance:

- variables increase together

Negative covariance:

- one increases while other decreases

Near zero:

- weak relationship

---

# 2.3 Covariance Matrix

For data matrix:

[
X
]

Covariance matrix roughly:

[
\Sigma = X^T X
]

(ignoring normalization for simplicity)

---

# Important Property

Covariance matrix is symmetric:

[
\Sigma = \Sigma^T
]

Important because:

- symmetric matrices have nice eigenvector structure
- enables PCA

---

# 2.4 Geometric Interpretation

Covariance matrix captures:

- how data spreads through space
- dominant directions of variation

Imagine:

- elongated cloud of points

Covariance identifies:

- long axis
- major directions

---

# ML Connection

Embedding spaces:

- contain correlated features
- covariance reveals semantic structure

---

# RL Connection

State representations:

- often compressed using covariance-based methods

---

# Mini Exercise

If two features always increase together, their covariance is likely:

- positive
- negative
- zero

Answer:
Positive.

---

# 3. PCA — Principal Component Analysis (15 min)

This is the conceptual center.

---

# 3.1 Core Idea

PCA asks:

> Which directions preserve the most information?

---

# 3.2 Principal Components

PCA finds:

- directions of maximum variance

These become:

- principal components

---

# Geometric Picture

Suppose data forms elongated cloud.

PCA finds:

1. longest direction
2. second-longest orthogonal direction
3. etc.

These directions:

- compress data efficiently
- preserve important structure

---

# 3.3 Dimensionality Reduction

Suppose:

- data originally 1000-dimensional

But most variation lies in:

- 20 directions

Then:

- project data onto those directions
- preserve most information

---

# 3.4 Why This Matters for Embeddings

Embeddings often contain:

- redundant dimensions
- correlated structure

PCA:

- compresses representation
- preserves semantic structure

---

# 3.5 PCA and Eigenvectors

Important connection:

PCA computes eigenvectors of covariance matrix.

Meaning:

- principal components are stable directions of variation

---

# 3.6 Representation Learning Intuition

Modern AI learns:

- latent spaces
- compressed representations

Embeddings work because:

- important information often lies in lower-dimensional structure

---

# LLM Connection

Word embeddings:

- semantic relationships cluster in low-dimensional manifolds

Examples:

- gender directions
- tense directions
- semantic clusters

---

# Mini Exercise

True or false:

PCA tries to preserve directions with the least variance.

Answer:
False.

It preserves directions with MOST variance.

---

# 4. Singular Value Decomposition (SVD) (18 min)

This is the mathematically deepest part.

Keep geometric.

---

# 4.1 Definition

[
A = U \Sigma V^T
]

Do NOT focus on proof.

Focus on meaning.

---

# 4.2 Geometric Interpretation

SVD decomposes transformation into:

---

## Step 1 — Rotate Space

[
V^T
]

Aligns coordinate system with important directions.

---

## Step 2 — Scale Directions

[
\Sigma
]

Scales axes differently.

Singular values determine:

- importance
- strength
- energy

---

## Step 3 — Rotate Again

[
U
]

Maps transformed coordinates into final space.

---

# Core Interpretation

SVD says:

> Any matrix transformation can be decomposed into rotations and scaling.

This is profound.

---

# 4.3 Singular Values

Large singular values:

- important directions
- strong signal

Small singular values:

- weak information
- redundancy/noise

---

# 4.4 Low-Rank Approximation

Critical AI concept.

Keep only largest singular values.

Then:

[
A \approx A_k
]

where:

- (A_k) is lower rank

Meaning:

- compress matrix
- preserve most information

---

# Why This Works

Real-world data often contains:

- redundancy
- correlated structure
- low-dimensional manifolds

---

# 4.5 Transformer Connection

Attention matrices often:

- approximately low-rank

Meaning:

- attention patterns contain redundancy
- information concentrates in major directions

---

# 4.6 LoRA Connection

LoRA assumes:

- fine-tuning updates are low-rank

Instead of updating full matrix:

- learn only important directional adjustments

Huge efficiency gain.

---

# 4.7 Embedding Compression

Large embedding matrices:

- often compressible via SVD

Used for:

- model compression
- distillation
- inference optimization

---

# 4.8 Why SVD Appears Everywhere

SVD is fundamental because:

- identifies dominant structure
- separates signal from redundancy
- provides optimal low-rank approximation

This appears throughout:

- recommendation systems
- embeddings
- diffusion models
- transformers
- retrieval systems

---

# Mini Exercise

True or false:

SVD only works for square matrices.

Answer:
False.

SVD works for rectangular matrices too.

---

# 5. Wrap-Up (2 min)

Key idea:

> High-dimensional data usually contains lower-dimensional structure.

This enables:

- embeddings
- compression
- latent spaces
- semantic organization
- efficient fine-tuning

Modern AI depends heavily on:

- eigenvectors
- covariance structure
- PCA
- SVD
- low-rank representations

---

# Suggested Exercises

1. Identify eigenvector intuition geometrically
2. Compute covariance direction intuitively
3. Explain PCA compression
4. Determine effects of large vs small singular values
5. Explain why low-rank approximation compresses information
6. Explain why transformers may contain low-rank structure

---

# Common Misconceptions

---

### “Eigenvectors are just abstract math.”

No.

They represent stable directions in transformations.

---

### “PCA finds important features.”

Not directly.

PCA finds important directions of variation.

---

### “Low-rank means low quality.”

No.

Many systems contain highly redundant structure.

---

### “SVD is only for compression.”

SVD reveals geometric structure generally.

Compression is one application.

---

# After This Lecture Students Should:

✔ Understand eigenvectors geometrically
✔ Understand eigenvalues as scaling factors
✔ Understand covariance intuitively
✔ Understand PCA conceptually
✔ Understand dimensionality reduction
✔ Understand latent representations
✔ Understand SVD geometrically
✔ Understand low-rank approximation
✔ Understand why LoRA works
✔ Understand why embeddings and attention matrices can compress information
