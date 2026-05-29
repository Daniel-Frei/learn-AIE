# Lecture 3 — Derivatives and Gradients

**Crash Course in Linear Algebra for LLMs, DL & RL**
**Duration:** 60 minutes
**Theme:** _Learning = optimization._

---

# Lecture Objectives

By the end of this lecture, students should be able to:

- Understand derivatives as rates of change
- Interpret gradients geometrically
- Understand why gradients point uphill
- Understand gradient descent intuitively
- Understand the chain rule conceptually
- Understand why backpropagation works
- Understand the role of gradients in neural networks and RL
- Understand high-level matrix derivatives

---

# Big Picture Narrative

Lecture 1:

> Vectors represent data and geometry.

Lecture 2:

> Matrices transform vector spaces.

Lecture 3:

> Learning means changing parameters to reduce error.

This lecture explains:

> How neural networks learn.

---

# Structure Overview (Time Plan)

| Section | Topic                               | Time   |
| ------- | ----------------------------------- | ------ |
| 1       | Scalar derivatives                  | 10 min |
| 2       | Partial derivatives                 | 10 min |
| 3       | Gradient vectors & gradient descent | 15 min |
| 4       | Chain rule & backpropagation        | 15 min |
| 5       | Matrix derivatives (high-level)     | 8 min  |
| 6       | Wrap-up                             | 2 min  |

---

# 1. Scalar Derivatives (10 min)

---

# 1.1 Intuition — What is a Derivative?

Core idea:

> A derivative measures how fast something changes.

Example:

[
f(x)=x^2
]

If:

- (x=2)
- slightly increase (x)

How quickly does (f(x)) increase?

Derivative answers this.

---

# 1.2 Slope Interpretation

Derivative = slope of tangent line.

Example:

[
\frac{d}{dx}x^2 = 2x
]

At:

- (x=1): slope = 2
- (x=5): slope = 10

Meaning:

- function becomes steeper as x increases

---

# 1.3 Why This Matters in ML

Neural networks optimize:

[
\text{Loss}(W)
]

Derivative tells us:

- which direction increases loss
- which direction decreases loss

Without derivatives:

- no gradient descent
- no modern deep learning

---

# 1.4 Intuitive Physical Analogy

Imagine:

- standing on a hill
- slope tells you where uphill is

Derivative:

- local steepness information

Gradient descent:

- walk downhill

---

# Mini Exercise

If:

[
f(x)=3x^2
]

what is:

[
f'(x)
]

Answer:

[
6x
]

---

# 2. Partial Derivatives (10 min)

---

# 2.1 Functions of Multiple Variables

Neural networks depend on MANY variables.

Example:

[
f(x,y)=x^2+y^2
]

Question:

How does output change if:

- only x changes?
- only y changes?

---

# 2.2 Partial Derivative Definition

[
\frac{\partial f}{\partial x}
]

means:

> change x slightly while holding everything else fixed.

---

# Example

[
f(x,y)=x^2+y^2
]

Then:

[
\frac{\partial f}{\partial x}=2x
]

[
\frac{\partial f}{\partial y}=2y
]

---

# 2.3 Why Partial Derivatives Matter

Neural networks contain:

- millions/billions of parameters

Need:

- derivative w.r.t. EACH parameter

Partial derivatives allow this.

---

# RL Connection

Value functions:

[
Q_\theta(s,a)
]

depend on many parameters:

[
\theta
]

Training requires:

- partial derivative for every parameter

---

# 2.4 Visual Intuition

Surface analogy:

- landscape in 3D
- slope in x-direction
- slope in y-direction

Partial derivatives measure:

- directional slopes along axes

---

# Mini Exercise

Given:

[
f(x,y)=3x+y^2
]

What is:

[
\frac{\partial f}{\partial x}
]

Answer:

[
3
]

---

# 3. Gradient Vector and Gradient Descent (15 min)

This is the conceptual core.

---

# 3.1 Gradient Vector

Combine all partial derivatives:

[
\nabla f(x,y)
=============

\begin{bmatrix}
\frac{\partial f}{\partial x}\
\frac{\partial f}{\partial y}
\end{bmatrix}
]

Example:

[
f(x,y)=x^2+y^2
]

[
\nabla f(x,y)=
\begin{bmatrix}
2x\
2y
\end{bmatrix}
]

---

# 3.2 Geometric Interpretation

Critical insight:

> Gradient points in direction of steepest increase.

Therefore:

[
-\nabla f
]

points downhill.

---

# 3.3 Gradient Descent

Optimization update rule:

[
x_{new}=x-\eta\nabla f(x)
]

Where:

- (\eta) = learning rate

Interpretation:

- take small step downhill

---

# 3.4 Why Learning Rate Matters

Too small:

- slow learning

Too large:

- overshoot
- instability

Explain:

- optimization dynamics
- training instability

---

# 3.5 Neural Network Interpretation

Parameters:

[
W
]

Loss:

[
L(W)
]

Gradient:

[
\nabla_W L
]

tells us:

- how to change weights
- to reduce error

---

# 3.6 RL Interpretation

In RL:

- policy parameters updated via gradients
- value functions updated via gradients

Examples:

- policy gradient methods
- actor-critic methods

---

# Mini Exercise

True or false:

The gradient points in the direction of fastest decrease.

Answer:
False.

It points toward fastest increase.

---

# 4. Chain Rule and Backpropagation (15 min)

This is the most important practical concept.

---

# 4.1 Basic Chain Rule

Suppose:

[
y=f(g(x))
]

Then:

[
\frac{dy}{dx}
=============

\frac{dy}{dg}
\cdot
\frac{dg}{dx}
]

Interpretation:

- change propagates through intermediate computations

---

# Example

[
g(x)=x^2
]

[
f(g)=3g
]

Then:

[
y=3x^2
]

Derivative:

[
\frac{dy}{dx}
=============

# 3\cdot2x

6x
]

---

# 4.2 Why Chain Rule Matters in Neural Networks

Neural network:

[
x \to W_1x \to \text{ReLU} \to W_2 \to \text{Loss}
]

Loss depends indirectly on early layers.

Chain rule propagates:

- influence backward through network

---

# 4.3 Backpropagation

Key idea:

> Backpropagation is repeated application of the chain rule.

Neural networks:

- compute forward
- compute error
- propagate gradients backward

---

# 4.4 Computational Graph Intuition

Represent computation as graph.

Example:

[
x \to z=x^2 \to y=3z
]

Backprop:

- compute local derivatives
- multiply along paths

---

# 4.5 Why Backprop is Efficient

Naively:

- recompute everything repeatedly

Backprop:

- reuses intermediate gradients

Critical efficiency breakthrough.

---

# 4.6 Deep Learning Connection

Transformer training:

- billions of parameters
- gradients computed via backprop

Without chain rule:

- modern AI impossible

---

# Mini Exercise

Suppose:

[
y=(2x+1)^2
]

What intermediate function could help apply chain rule?

Answer:

[
g(x)=2x+1
]

then:

[
y=g(x)^2
]

---

# 5. Matrix Derivatives (High-Level Only) (8 min)

No tensor calculus yet.

Focus on intuition.

---

# 5.1 Derivatives of Vector-Valued Systems

Neural networks use matrices:

[
y=Wx
]

Need derivatives w.r.t:

- weights
- inputs
- activations

---

# 5.2 Key Intuition

Derivative shape matches parameter shape.

If:

[
W \in \mathbb{R}^{m\times n}
]

then:

[
\frac{\partial L}{\partial W}
\in
\mathbb{R}^{m\times n}
]

Important practical intuition.

---

# 5.3 Why Matrix Gradients Matter

Gradient matrix tells:

- how every weight should change

Each parameter:

- has its own slope
- its own contribution to loss

---

# 5.4 Outer Product Intuition

Very high-level intuition:

For linear layer:

[
y=Wx
]

weight gradients roughly depend on:

- input activations
- output error signals

This creates outer-product-like structure.

---

# 5.5 Why GPUs Love Deep Learning

Most operations become:

- matrix multiplications
- vectorized gradient computations

This is why:

- linear algebra hardware dominates AI

---

# 6. Wrap-Up (2 min)

Key idea:

> Learning is optimization using gradients.

Modern AI depends on:

- derivatives
- gradients
- chain rule
- backpropagation

Everything from:

- transformers
- CNNs
- RL agents
- diffusion models

learns through gradient-based optimization.

---

# Suggested Exercises

1. Compute simple scalar derivatives
2. Compute partial derivatives
3. Compute gradients of simple functions
4. Apply one gradient descent update
5. Use chain rule on nested functions
6. Identify gradient shapes

---

# Common Misconceptions

---

### “Gradient points downhill.”

False.

Gradient points uphill.

Negative gradient points downhill.

---

### “Backpropagation is magic.”

No.

It is repeated chain rule.

---

### “Derivatives only apply to simple functions.”

Neural networks are compositions of simple differentiable operations.

---

### “Learning rate only changes speed.”

Too large learning rates can:

- destabilize training
- cause divergence

---

# After This Lecture Students Should:

✔ Understand derivatives as local change
✔ Understand partial derivatives
✔ Understand gradients geometrically
✔ Understand gradient descent
✔ Understand learning rates
✔ Understand chain rule intuition
✔ Understand backpropagation conceptually
✔ Understand why gradients match parameter shapes
✔ Understand how neural networks learn mathematically
