import { Question } from "../../quiz";

export const CrashCourseLinearAlgebraLecture3Questions: Question[] = [
  {
    id: "la-crash-l3-q01",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements correctly describe a derivative?",
    options: [
      { text: "A derivative measures local rate of change.", isCorrect: true },
      {
        text: "A derivative can be interpreted as the slope of a tangent line.",
        isCorrect: true,
      },
      {
        text: "A derivative can tell whether a small change in an input increases or decreases an output.",
        isCorrect: true,
      },
      {
        text: "Derivatives are central to gradient-based learning.",
        isCorrect: true,
      },
    ],
    explanation:
      "A derivative describes how a function changes near a particular input, which is why it can be understood as a tangent slope. In machine learning, derivatives tell how a loss changes when parameters move slightly, making them essential for optimization.",
  },
  {
    id: "la-crash-l3-q02",
    chapter: 3,
    difficulty: "easy",
    prompt: "For \\(f(x)=x^2\\), which statements are correct?",
    options: [
      { text: "The derivative is \\(f'(x)=2x\\).", isCorrect: true },
      { text: "At \\(x=1\\), the slope is 2.", isCorrect: true },
      { text: "At \\(x=5\\), the slope is 10.", isCorrect: true },
      {
        text: "The slope is the same at every value of \\(x\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The derivative of \\(x^2\\) is \\(2x\\), so the local slope depends on the input value. The incorrect statement treats the function as if it had constant slope, which would be true for a linear function but not for a quadratic.",
  },
  {
    id: "la-crash-l3-q03",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly connect derivatives to loss minimization?",
    options: [
      {
        text: "A derivative can indicate which direction increases loss locally.",
        isCorrect: true,
      },
      {
        text: "Moving opposite the derivative can reduce loss for a small enough step.",
        isCorrect: true,
      },
      {
        text: "A derivative guarantees the global minimum is known immediately.",
        isCorrect: false,
      },
      {
        text: "Loss minimization in neural networks avoids derivatives entirely.",
        isCorrect: false,
      },
    ],
    explanation:
      "Derivatives provide local information about how loss changes, so stepping against that direction can lower loss when the step size is reasonable. They do not reveal the entire loss surface at once, and modern neural-network training relies heavily on derivative information.",
  },
  {
    id: "la-crash-l3-q04",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statement best describes the derivative of \\(f(x)=3x^2\\)?",
    options: [
      { text: "\\(f'(x)=6x\\).", isCorrect: true },
      { text: "\\(f'(x)=3x\\).", isCorrect: false },
      { text: "\\(f'(x)=x^3\\).", isCorrect: false },
      { text: "\\(f'(x)=6\\) for every \\(x\\).", isCorrect: false },
    ],
    explanation:
      "The constant multiplier remains, and the derivative of \\(x^2\\) is \\(2x\\), giving \\(6x\\). The other choices either drop a factor, change the power incorrectly, or confuse a variable slope with a constant one.",
  },
  {
    id: "la-crash-l3-q05",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements correctly describe partial derivatives?",
    options: [
      {
        text: "A partial derivative measures change with respect to one variable.",
        isCorrect: true,
      },
      {
        text: "Other variables are held fixed while taking a partial derivative.",
        isCorrect: true,
      },
      {
        text: "Partial derivatives are needed when a function has many inputs.",
        isCorrect: true,
      },
      {
        text: "Training a model with many parameters requires derivative information for each parameter.",
        isCorrect: true,
      },
    ],
    explanation:
      "A partial derivative isolates the local effect of one variable while treating the others as fixed. This is exactly the setup in neural networks, where the loss depends on many parameters and each parameter needs its own local slope.",
  },
  {
    id: "la-crash-l3-q06",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "For \\(f(x,y)=x^2+y^2\\), which partial derivative statements are correct?",
    options: [
      { text: "\\(\\frac{\\partial f}{\\partial x}=2x\\).", isCorrect: true },
      { text: "\\(\\frac{\\partial f}{\\partial y}=2y\\).", isCorrect: true },
      {
        text: "The gradient is \\([2x,2y]^T\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\frac{\\partial f}{\\partial x}=2y\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "When differentiating with respect to \\(x\\), the \\(y^2\\) term is treated as constant, so the result is \\(2x\\). When differentiating with respect to \\(y\\), the result is \\(2y\\), and the gradient combines those partial derivatives.",
  },
  {
    id: "la-crash-l3-q07",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the gradient vector \\(\\nabla f\\)?",
    options: [
      {
        text: "It combines partial derivatives into one vector.",
        isCorrect: true,
      },
      {
        text: "It points in the direction of steepest local increase.",
        isCorrect: true,
      },
      {
        text: "It always points in the direction of fastest decrease.",
        isCorrect: false,
      },
      {
        text: "It can only be defined for functions with exactly one input variable.",
        isCorrect: false,
      },
    ],
    explanation:
      "The gradient packages all partial derivatives together and gives the direction of steepest ascent. The negative gradient points downhill, so saying the gradient itself always points toward fastest decrease reverses the sign.",
  },
  {
    id: "la-crash-l3-q08",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statement correctly identifies the downhill direction for a differentiable loss \\(L\\)?",
    options: [
      { text: "\\(-\\nabla L\\).", isCorrect: true },
      { text: "\\(\\nabla L\\).", isCorrect: false },
      { text: "\\(L^2\\).", isCorrect: false },
      {
        text: "The largest coordinate axis regardless of the loss.",
        isCorrect: false,
      },
    ],
    explanation:
      "The gradient points toward steepest local increase of the loss, so the negative gradient is the local downhill direction. Squaring the loss or choosing an axis without considering the gradient does not identify the descent direction.",
  },
  {
    id: "la-crash-l3-q09",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements correctly describe gradient descent?",
    options: [
      {
        text: "It updates parameters by taking steps opposite the gradient.",
        isCorrect: true,
      },
      {
        text: "Its basic form can be written \\(x_{new}=x-\\eta\\nabla f(x)\\).",
        isCorrect: true,
      },
      {
        text: "The learning rate controls the step size.",
        isCorrect: true,
      },
      {
        text: "It uses local slope information to seek lower loss.",
        isCorrect: true,
      },
    ],
    explanation:
      "Gradient descent moves against the gradient because that is the local direction of decreasing loss. The learning rate \\(\\eta\\) scales the step, so it controls how aggressively the update moves through parameter space.",
  },
  {
    id: "la-crash-l3-q10",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements about learning rate are correct?",
    options: [
      {
        text: "A very small learning rate can make learning slow.",
        isCorrect: true,
      },
      {
        text: "A very large learning rate can overshoot useful updates.",
        isCorrect: true,
      },
      {
        text: "A too-large learning rate can contribute to unstable training.",
        isCorrect: true,
      },
      {
        text: "The learning rate has no effect on optimization dynamics.",
        isCorrect: false,
      },
    ],
    explanation:
      "The learning rate determines how large each gradient step is. Too small a value can crawl slowly, while too large a value can overshoot or destabilize the update process.",
  },
  {
    id: "la-crash-l3-q11",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements correctly connect gradients to neural networks?",
    options: [
      {
        text: "A neural-network loss can be viewed as a function of weights.",
        isCorrect: true,
      },
      {
        text: "The gradient \\(\\nabla_W L\\) indicates how the loss changes with weights.",
        isCorrect: true,
      },
      {
        text: "Weights are updated without using information from the loss.",
        isCorrect: false,
      },
      {
        text: "Gradient-based training changes weights without using the error signal.",
        isCorrect: false,
      },
    ],
    explanation:
      "A neural network is trained by treating the loss as a function of parameters and computing gradients with respect to those parameters. Weight updates use the loss-derived gradient, so claims that training ignores the error signal are incorrect.",
  },
  {
    id: "la-crash-l3-q12",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statement correctly describes a reinforcement-learning value function with parameters \\(\\theta\\)?",
    options: [
      {
        text: "Training \\(Q_\\theta(s,a)\\) can require gradients with respect to \\(\\theta\\).",
        isCorrect: true,
      },
      {
        text: "The parameters \\(\\theta\\) cannot affect the value estimate.",
        isCorrect: false,
      },
      {
        text: "Gradient methods are irrelevant to actor-critic systems.",
        isCorrect: false,
      },
      {
        text: "Partial derivatives only apply to supervised learning.",
        isCorrect: false,
      },
    ],
    explanation:
      "A parameterized value function depends on its parameters, so training can use gradients with respect to those parameters. Actor-critic and value-based deep reinforcement learning methods frequently use gradient-based updates.",
  },
  {
    id: "la-crash-l3-q13",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements correctly describe the chain rule?",
    options: [
      {
        text: "It computes derivatives of composed functions.",
        isCorrect: true,
      },
      {
        text: "For \\(y=f(g(x))\\), it multiplies the derivative of the outer function by the derivative of the inner function.",
        isCorrect: true,
      },
      {
        text: "It explains how changes propagate through intermediate computations.",
        isCorrect: true,
      },
      {
        text: "It is the core mathematical idea behind backpropagation.",
        isCorrect: true,
      },
    ],
    explanation:
      "The chain rule handles functions built from nested computations by multiplying local rates of change along the dependency path. Backpropagation uses this principle repeatedly through the computational graph of a neural network.",
  },
  {
    id: "la-crash-l3-q14",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "For \\(g(x)=x^2\\), \\(f(g)=3g\\), and \\(y=f(g(x))\\), which statements are correct?",
    options: [
      { text: "The composed function is \\(y=3x^2\\).", isCorrect: true },
      { text: "The derivative is \\(\\frac{dy}{dx}=6x\\).", isCorrect: true },
      {
        text: "The chain rule multiplies \\(\\frac{dy}{dg}\\) by \\(\\frac{dg}{dx}\\).",
        isCorrect: true,
      },
      {
        text: "The derivative is \\(3x^2\\) because the function and derivative are identical.",
        isCorrect: false,
      },
    ],
    explanation:
      "Substituting \\(g(x)=x^2\\) into \\(f(g)=3g\\) gives \\(y=3x^2\\), whose derivative is \\(6x\\). The incorrect statement confuses the original function with its derivative.",
  },
  {
    id: "la-crash-l3-q15",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements correctly describe backpropagation?",
    options: [
      {
        text: "It applies the chain rule backward through a computational graph.",
        isCorrect: true,
      },
      {
        text: "It reuses intermediate gradient information efficiently.",
        isCorrect: true,
      },
      {
        text: "It avoids computing any local derivatives.",
        isCorrect: false,
      },
      {
        text: "It only works for models with one parameter.",
        isCorrect: false,
      },
    ],
    explanation:
      "Backpropagation computes local derivatives and combines them through the chain rule from the loss back toward earlier computations. Its efficiency comes from reusing intermediate results instead of recomputing every derivative from scratch.",
  },
  {
    id: "la-crash-l3-q16",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statement best describes why early layers in a neural network receive gradient information?",
    options: [
      {
        text: "The chain rule propagates the loss influence through later operations back to earlier parameters.",
        isCorrect: true,
      },
      {
        text: "Early layers receive gradients because their weights are always copied from later layers.",
        isCorrect: false,
      },
      {
        text: "Early layers cannot affect the loss once later layers are added.",
        isCorrect: false,
      },
      {
        text: "Only the final layer can be trained with gradient descent.",
        isCorrect: false,
      },
    ],
    explanation:
      "Even early parameters influence the final loss through a chain of later computations. The chain rule lets the training algorithm compute how much those earlier parameters contributed and update them accordingly.",
  },
  {
    id: "la-crash-l3-q17",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe a computational graph such as \\(x \\to z=x^2 \\to y=3z\\)?",
    options: [
      {
        text: "Nodes can represent intermediate values such as \\(z\\).",
        isCorrect: true,
      },
      {
        text: "Edges can represent dependencies between computations.",
        isCorrect: true,
      },
      {
        text: "Backpropagation can multiply local derivatives along dependency paths.",
        isCorrect: true,
      },
      {
        text: "The graph helps organize how the loss depends on earlier values.",
        isCorrect: true,
      },
    ],
    explanation:
      "A computational graph breaks a complicated expression into simpler intermediate operations. Backpropagation follows these dependencies backward, multiplying local derivatives so the effect of earlier values on later loss can be computed.",
  },
  {
    id: "la-crash-l3-q18",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "For \\(y=(2x+1)^2\\), which statements correctly support a chain-rule calculation?",
    options: [
      {
        text: "An intermediate function can be \\(g(x)=2x+1\\).",
        isCorrect: true,
      },
      { text: "The outer function can be \\(y=g^2\\).", isCorrect: true },
      { text: "The derivative is \\(4(2x+1)\\).", isCorrect: true },
      { text: "The derivative is \\(2x+1\\).", isCorrect: false },
    ],
    explanation:
      "Writing \\(g(x)=2x+1\\) and \\(y=g^2\\) separates the inner and outer functions. The derivative is \\(2g\\cdot2=4(2x+1)\\), so the shorter expression misses necessary factors.",
  },
  {
    id: "la-crash-l3-q19",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements correctly describe matrix gradients?",
    options: [
      {
        text: "A gradient with respect to a matrix parameter usually has the same shape as that matrix.",
        isCorrect: true,
      },
      {
        text: "Each entry of a weight-gradient matrix describes a slope for one parameter.",
        isCorrect: true,
      },
      {
        text: "A scalar loss cannot have derivatives with respect to a matrix.",
        isCorrect: false,
      },
      {
        text: "All entries in a matrix gradient must be identical.",
        isCorrect: false,
      },
    ],
    explanation:
      "If \\(W\\) is a matrix parameter, \\(\\partial L/\\partial W\\) records how the scalar loss changes with each entry of \\(W\\). Those entries need not be equal because different weights can influence the loss differently.",
  },
  {
    id: "la-crash-l3-q20",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "If \\(W\\in\\mathbb{R}^{m\\times n}\\), which statement best describes the shape of \\(\\frac{\\partial L}{\\partial W}\\)?",
    options: [
      {
        text: "It has shape \\(m\\times n\\), matching \\(W\\).",
        isCorrect: true,
      },
      {
        text: "It must be a scalar because the loss is a scalar.",
        isCorrect: false,
      },
      {
        text: "It must have shape \\(n\\times m\\) for every matrix.",
        isCorrect: false,
      },
      {
        text: "It has no shape because gradients are not numerical objects.",
        isCorrect: false,
      },
    ],
    explanation:
      "The gradient with respect to a matrix parameter stores one derivative per matrix entry, so it matches the parameter shape. The loss can be scalar while its gradient with respect to many parameters is a vector or matrix of slopes.",
  },
  {
    id: "la-crash-l3-q21",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the high-level weight-gradient intuition for a linear layer \\(y=Wx\\)?",
    options: [
      {
        text: "Weight gradients depend on input activations.",
        isCorrect: true,
      },
      {
        text: "Weight gradients depend on output-side error signals.",
        isCorrect: true,
      },
      {
        text: "The pattern can resemble an outer product between error and input information.",
        isCorrect: true,
      },
      {
        text: "The same structure helps explain why batches of examples can be handled with matrix operations.",
        isCorrect: true,
      },
    ],
    explanation:
      "For a linear layer, the update signal for a weight depends on what input entered that connection and what error signal came back through the output. This creates an outer-product-like structure, and batching many examples turns the same idea into efficient matrix computations.",
  },
  {
    id: "la-crash-l3-q22",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly explain why GPUs are useful for deep learning?",
    options: [
      {
        text: "Neural-network forward passes contain many matrix multiplications.",
        isCorrect: true,
      },
      {
        text: "Gradient computations can often be vectorized.",
        isCorrect: true,
      },
      {
        text: "Backpropagation can reuse structured linear-algebra operations.",
        isCorrect: true,
      },
      {
        text: "GPUs remove the need to compute gradients.",
        isCorrect: false,
      },
    ],
    explanation:
      "Deep learning workloads contain large numbers of similar arithmetic operations, especially matrix multiplications and vectorized gradient calculations. GPUs accelerate those operations, but they do not eliminate the need for gradients or backpropagation.",
  },
  {
    id: "la-crash-l3-q23",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly distinguish gradients from function values?",
    options: [
      {
        text: "A function value gives the output at a point.",
        isCorrect: true,
      },
      {
        text: "A gradient gives local change information around a point.",
        isCorrect: true,
      },
      {
        text: "A gradient is always the same thing as the loss value.",
        isCorrect: false,
      },
      {
        text: "Knowing only the function value always identifies the best descent direction.",
        isCorrect: false,
      },
    ],
    explanation:
      "The loss value tells how good or bad the current parameters are, while the gradient tells how the loss changes nearby. A single scalar value does not by itself reveal which direction to move in a high-dimensional parameter space.",
  },
  {
    id: "la-crash-l3-q24",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "For \\(f(x,y)=3x+y^2\\), which statement correctly gives \\(\\frac{\\partial f}{\\partial x}\\)?",
    options: [
      { text: "\\(3\\).", isCorrect: true },
      { text: "\\(2y\\).", isCorrect: false },
      { text: "\\(3+2y\\).", isCorrect: false },
      { text: "\\(y^2\\).", isCorrect: false },
    ],
    explanation:
      "When differentiating with respect to \\(x\\), the term \\(3x\\) contributes 3 and the \\(y^2\\) term is treated as constant. The other choices either differentiate with respect to \\(y\\), mix both partial derivatives, or leave part of the original function unchanged.",
  },
  {
    id: "la-crash-l3-q25",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe local optimization information?",
    options: [
      {
        text: "A derivative describes behavior near the current point.",
        isCorrect: true,
      },
      {
        text: "Gradient descent uses repeated local steps rather than one global jump.",
        isCorrect: true,
      },
      {
        text: "A local slope can be useful even when the whole surface is complicated.",
        isCorrect: true,
      },
      {
        text: "A small enough downhill step can reduce the loss even without solving the entire optimization problem.",
        isCorrect: true,
      },
    ],
    explanation:
      "Gradient methods use local information to make incremental progress through a complex landscape. They do not require solving the full global problem in one step, which is one reason they are practical for large neural networks.",
  },
  {
    id: "la-crash-l3-q26",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe differentiable operations in neural networks?",
    options: [
      {
        text: "A network can be seen as a composition of many simpler operations.",
        isCorrect: true,
      },
      {
        text: "Backpropagation works when the needed local derivatives can be computed or handled by the framework.",
        isCorrect: true,
      },
      {
        text: "The chain rule can connect derivatives across layers.",
        isCorrect: true,
      },
      {
        text: "A network cannot be trained if it contains any matrix multiplication.",
        isCorrect: false,
      },
    ],
    explanation:
      "Neural networks are built from operations such as matrix multiplication, nonlinearities, and losses, and the chain rule connects their local derivatives. Matrix multiplication is not an obstacle to training; it is one of the main operations that training differentiates through.",
  },
  {
    id: "la-crash-l3-q27",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe policy-gradient intuition in reinforcement learning?",
    options: [
      {
        text: "Policy parameters can be updated using gradients.",
        isCorrect: true,
      },
      {
        text: "A gradient can indicate how changing policy parameters affects an objective.",
        isCorrect: true,
      },
      {
        text: "Policy-gradient methods never optimize expected reward.",
        isCorrect: false,
      },
      {
        text: "Reinforcement learning removes the need for differentiable models in every possible method.",
        isCorrect: false,
      },
    ],
    explanation:
      "Policy-gradient methods use gradient estimates to change policy parameters in a direction that improves an objective such as expected reward. Not every reinforcement-learning method is identical, but gradients are central to many modern deep RL algorithms.",
  },
  {
    id: "la-crash-l3-q28",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement best explains why gradient descent uses \\(-\\nabla L\\) rather than \\(\\nabla L\\) when minimizing loss?",
    options: [
      {
        text: "\\(\\nabla L\\) points toward steepest local increase, so its negative points toward steepest local decrease.",
        isCorrect: true,
      },
      {
        text: "\\(\\nabla L\\) and \\(-\\nabla L\\) always point in the same direction.",
        isCorrect: false,
      },
      {
        text: "The negative sign is only a notation convention with no effect on updates.",
        isCorrect: false,
      },
      {
        text: "The gradient is unrelated to the loss surface.",
        isCorrect: false,
      },
    ],
    explanation:
      "For minimization, the goal is to move toward lower loss, not higher loss. Since the gradient points uphill, the negative gradient is the natural local descent direction.",
  },
  {
    id: "la-crash-l3-q29",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe what happens when backpropagation is applied through many layers?",
    options: [
      {
        text: "Gradients are computed for parameters in multiple layers.",
        isCorrect: true,
      },
      {
        text: "Local derivatives are combined through the chain rule.",
        isCorrect: true,
      },
      {
        text: "Intermediate activations can be reused when computing gradients.",
        isCorrect: true,
      },
      {
        text: "The final loss can influence early parameters indirectly.",
        isCorrect: true,
      },
    ],
    explanation:
      "Backpropagation tracks how the final loss depends on intermediate activations and parameters across the network. The chain rule carries that dependence backward, allowing early weights to receive update signals even though they are far from the loss output.",
  },
  {
    id: "la-crash-l3-q30",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe why a too-large learning rate can cause divergence?",
    options: [
      {
        text: "The update can jump past a lower-loss region.",
        isCorrect: true,
      },
      {
        text: "The loss can increase instead of decrease after an update.",
        isCorrect: true,
      },
      {
        text: "Large steps can create oscillation around a valley.",
        isCorrect: true,
      },
      {
        text: "Large learning rates guarantee faster convergence for every problem.",
        isCorrect: false,
      },
    ],
    explanation:
      "A gradient gives local information, so a step that is too large may move outside the region where that information was useful. This can overshoot good regions, increase loss, or create unstable oscillation rather than reliable convergence.",
  },
  {
    id: "la-crash-l3-q31",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly connect gradient shape to implementation sanity checks?",
    options: [
      {
        text: "A weight matrix and its gradient should have compatible shapes for an update.",
        isCorrect: true,
      },
      {
        text: "Shape mismatches can reveal an incorrect derivative or matrix orientation.",
        isCorrect: true,
      },
      {
        text: "A gradient for a matrix parameter should always be a single number.",
        isCorrect: false,
      },
      {
        text: "Shape reasoning is irrelevant once automatic differentiation is used.",
        isCorrect: false,
      },
    ],
    explanation:
      "To update a parameter, the gradient must align with the parameter entries being changed. Automatic differentiation computes gradients, but shape reasoning remains useful for detecting mistaken transposes, broadcasts, or incompatible parameter layouts.",
  },
  {
    id: "la-crash-l3-q32",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement best explains why backpropagation is more efficient than recomputing every parameter derivative independently?",
    options: [
      {
        text: "It reuses shared intermediate computations and gradient signals across the graph.",
        isCorrect: true,
      },
      {
        text: "It ignores most parameters during training.",
        isCorrect: false,
      },
      {
        text: "It replaces the chain rule with random search.",
        isCorrect: false,
      },
      {
        text: "It only works because neural networks have no intermediate values.",
        isCorrect: false,
      },
    ],
    explanation:
      "Many parameter derivatives share intermediate computations, especially in layered models. Backpropagation exploits that shared structure, which makes gradient computation practical for networks with many parameters.",
  },
  {
    id: "la-crash-l3-q33",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe gradients in high-dimensional parameter spaces?",
    options: [
      {
        text: "A gradient has one component per parameter direction.",
        isCorrect: true,
      },
      {
        text: "It gives local first-order information about the loss surface.",
        isCorrect: true,
      },
      {
        text: "The negative gradient can be used as a descent direction.",
        isCorrect: true,
      },
      {
        text: "It can be computed for large models using automatic differentiation and backpropagation.",
        isCorrect: true,
      },
    ],
    explanation:
      "A high-dimensional gradient is a structured collection of local slopes, one for each parameter direction. Even for very large models, automatic differentiation and backpropagation make these local slope calculations feasible enough for training.",
  },
  {
    id: "la-crash-l3-q34",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the relationship between scalar, vector, and matrix derivatives?",
    options: [
      {
        text: "A scalar function of many variables can have a vector gradient.",
        isCorrect: true,
      },
      {
        text: "A scalar loss can have a matrix-shaped gradient with respect to a matrix parameter.",
        isCorrect: true,
      },
      {
        text: "Vector-valued systems can require Jacobian-like derivative structures.",
        isCorrect: true,
      },
      {
        text: "Derivatives are only meaningful when both input and output are scalars.",
        isCorrect: false,
      },
    ],
    explanation:
      "Derivatives generalize beyond single-input, single-output functions by organizing local sensitivities into vectors, matrices, or Jacobian-like arrays. Neural networks rely on these generalized derivative structures because parameters and activations are often vectors or matrices.",
  },
  {
    id: "la-crash-l3-q35",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe why gradients can train embeddings or hidden representations?",
    options: [
      {
        text: "Embedding vectors can be treated as parameters or outputs of parameterized transformations.",
        isCorrect: true,
      },
      {
        text: "A loss can send gradient information back to the weights that produced an embedding.",
        isCorrect: true,
      },
      {
        text: "Embeddings cannot be changed by optimization because they are geometric objects.",
        isCorrect: false,
      },
      {
        text: "Gradient descent only updates scalar parameters, not vectors.",
        isCorrect: false,
      },
    ],
    explanation:
      "Embeddings are numerical vectors, and the parameters that create or store them can receive gradients from a loss. Their geometric interpretation does not prevent optimization; it is exactly what optimization reshapes during representation learning.",
  },
  {
    id: "la-crash-l3-q36",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement best explains the phrase 'learning is optimization using gradients'?",
    options: [
      {
        text: "Model parameters are adjusted by using derivative information to reduce a training objective.",
        isCorrect: true,
      },
      {
        text: "A model learns by storing every training example as a separate rule.",
        isCorrect: false,
      },
      {
        text: "Gradients remove the need for a loss function.",
        isCorrect: false,
      },
      {
        text: "Optimization means choosing parameters without feedback from errors.",
        isCorrect: false,
      },
    ],
    explanation:
      "Gradient-based learning defines an objective such as loss and then uses derivatives to decide how parameters should move. This is different from memorizing rules or changing parameters without an error signal.",
  },
  {
    id: "la-crash-l3-q37",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe how the chain rule appears in transformer training?",
    options: [
      {
        text: "Loss gradients pass through attention computations.",
        isCorrect: true,
      },
      {
        text: "Gradients pass through feed-forward layers and projection matrices.",
        isCorrect: true,
      },
      {
        text: "Earlier embedding or projection parameters can receive updates through later computations.",
        isCorrect: true,
      },
      {
        text: "Large models still depend on repeated local derivative composition.",
        isCorrect: true,
      },
    ],
    explanation:
      "Transformers are composed of many differentiable operations, including projections, attention, feed-forward layers, and losses. The chain rule connects them so the final training objective can update parameters throughout the model.",
  },
  {
    id: "la-crash-l3-q38",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly identify misconceptions about gradients and backpropagation?",
    options: [
      {
        text: "Saying that the gradient points downhill is a sign error for minimization.",
        isCorrect: true,
      },
      {
        text: "Backpropagation is repeated chain rule rather than magic.",
        isCorrect: true,
      },
      {
        text: "Derivatives can apply to complex composed systems, not only simple classroom functions.",
        isCorrect: true,
      },
      {
        text: "Learning rate only changes speed and cannot affect stability.",
        isCorrect: false,
      },
    ],
    explanation:
      "The gradient points uphill, while the negative gradient points downhill. Backpropagation is a systematic application of the chain rule, and learning rate affects both speed and stability because steps can be too small or too large.",
  },
  {
    id: "la-crash-l3-q39",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the role of gradients in both supervised learning and reinforcement learning?",
    options: [
      {
        text: "Supervised losses can be minimized with gradient descent.",
        isCorrect: true,
      },
      {
        text: "Deep RL value or policy networks cannot use gradient-based updates.",
        isCorrect: false,
      },
      {
        text: "Reinforcement learning objectives can involve delayed reward, which changes the learning setup but not the usefulness of gradients.",
        isCorrect: true,
      },
      {
        text: "Gradients are useful only when labels are immediate and deterministic.",
        isCorrect: false,
      },
    ],
    explanation:
      "Supervised learning often has direct labels, while reinforcement learning may optimize objectives involving delayed reward. Gradients remain useful in both settings, including many deep RL value and policy networks.",
  },
  {
    id: "la-crash-l3-q40",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement best captures the practical importance of derivatives and gradients for modern AI?",
    options: [
      {
        text: "Derivatives, gradients, the chain rule, and backpropagation explain how large models adjust parameters to reduce error.",
        isCorrect: true,
      },
      {
        text: "Modern AI mainly learns by avoiding optimization.",
        isCorrect: false,
      },
      {
        text: "Backpropagation is unrelated to transformer or diffusion-model training.",
        isCorrect: false,
      },
      {
        text: "Matrix gradients are optional because neural-network parameters are not numerical.",
        isCorrect: false,
      },
    ],
    explanation:
      "Modern AI systems learn by computing how losses depend on many numerical parameters and then updating those parameters. Derivatives, gradients, the chain rule, and backpropagation are the core tools that make this process work at scale.",
  },
];

export const CrashCourseLinearAlgebraL3Questions =
  CrashCourseLinearAlgebraLecture3Questions;
