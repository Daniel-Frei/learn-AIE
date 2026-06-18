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
        text: "The slope is constant across the graph.",
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
        text: "A derivative by itself identifies the global minimum immediately.",
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
      { text: "\\(f'(x)=6\\) across the whole domain.", isCorrect: false },
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
        text: "It points in the direction of fastest decrease.",
        isCorrect: false,
      },
      {
        text: "It is limited to functions with exactly one input variable.",
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
        text: "The parameters \\(\\theta\\) are disconnected from the value estimate.",
        isCorrect: false,
      },
      {
        text: "Actor-critic systems avoid gradient methods in their parameter updates.",
        isCorrect: false,
      },
      {
        text: "Partial derivatives are a tool reserved for supervised-learning losses.",
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
        text: "It is limited to models with one parameter.",
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
        text: "Early layers receive gradients because their weights are copied from later layers during the backward pass.",
        isCorrect: false,
      },
      {
        text: "Early layers stop influencing the loss once later layers are added.",
        isCorrect: false,
      },
      {
        text: "The final layer is the part trained by gradient descent.",
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
        text: "A scalar loss is disconnected from matrix-valued parameters.",
        isCorrect: false,
      },
      {
        text: "Matrix-gradient entries share a single identical value.",
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
        text: "It uses shape \\(n\\times m\\) by convention for this matrix.",
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
        text: "A gradient is the loss value written as a vector.",
        isCorrect: false,
      },
      {
        text: "Knowing the function value identifies the best descent direction.",
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
        text: "Matrix multiplication blocks gradient-based training in a network.",
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
        text: "Policy-gradient methods optimize a supervised-label objective instead of expected reward.",
        isCorrect: false,
      },
      {
        text: "Reinforcement learning removes differentiability requirements from actor-critic updates.",
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
        text: "\\(\\nabla L\\) and \\(-\\nabla L\\) point in the same direction during descent.",
        isCorrect: false,
      },
      {
        text: "The negative sign is a notation convention with no effect on update direction.",
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
      "When backpropagation carries a final loss backward through many layers, which statements correctly describe the gradient computation?",
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
      "Backpropagation reuses intermediate activations and local derivatives to compute how the final loss changes with parameters across multiple layers. The chain rule carries this dependence backward, so early weights can receive update signals indirectly through later computations.",
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
        text: "Large learning rates make convergence faster across problem settings.",
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
        text: "A gradient for a matrix parameter should be summarized as a single number.",
        isCorrect: false,
      },
      {
        text: "Automatic differentiation replaces shape reasoning during gradient flow.",
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
        text: "It works because neural networks have no intermediate values.",
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
        text: "Derivatives are meaningful mainly for scalar-to-scalar functions.",
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
        text: "Embeddings are fixed geometric objects rather than trainable parameters.",
        isCorrect: false,
      },
      {
        text: "Gradient descent updates scalar parameters but not vector parameters.",
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
        text: "A model learns by storing each training example as a separate rule.",
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
        text: "Learning rate changes step speed without affecting stability.",
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
        text: "Deep RL value or policy networks use update rules separate from gradients.",
        isCorrect: false,
      },
      {
        text: "Reinforcement learning objectives can involve delayed reward, which changes the learning setup but not the usefulness of gradients.",
        isCorrect: true,
      },
      {
        text: "Gradients are useful mainly when labels are immediate and deterministic.",
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
  {
    id: "la-crash-l3-q41",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "For \\(f(x)=5x^3\\), which statement correctly gives the derivative and its value at \\(x=2\\)?",
    options: [
      {
        text: "\\(f'(x)=15x^2\\), so \\(f'(2)=60\\).",
        isCorrect: true,
      },
      {
        text: "\\(f'(x)=5x^2\\), so \\(f'(2)=20\\).",
        isCorrect: false,
      },
      {
        text: "\\(f'(x)=15x^3\\), so \\(f'(2)=120\\).",
        isCorrect: false,
      },
      {
        text: "\\(f'(x)=x^4\\), so \\(f'(2)=16\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The power rule gives \\(\\frac{d}{dx}x^3=3x^2\\), so multiplying by \\(5\\) gives \\(15x^2\\). Evaluating at \\(x=2\\) gives \\(15\\cdot4=60\\). The other options either miss the power-rule coefficient or keep the wrong power.",
  },
  {
    id: "la-crash-l3-q42",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "For \\(f(x,y)=xy+x^2\\), which partial-derivative statements are correct?",
    options: [
      {
        text: "\\(\\frac{\\partial f}{\\partial x}=y+2x\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\frac{\\partial f}{\\partial y}=x\\).",
        isCorrect: true,
      },
      {
        text: "At \\((x,y)=(2,3)\\), \\(\\frac{\\partial f}{\\partial x}=7\\).",
        isCorrect: true,
      },
      {
        text: "When computing \\(\\frac{\\partial f}{\\partial x}\\), \\(y\\) must also be changed by the same amount.",
        isCorrect: false,
      },
    ],
    explanation:
      "A partial derivative changes one variable while holding the others fixed. Differentiating \\(xy+x^2\\) with respect to \\(x\\) gives \\(y+2x\\), and differentiating with respect to \\(y\\) gives \\(x\\). At \\((2,3)\\), the first expression equals \\(3+4=7\\).",
  },
  {
    id: "la-crash-l3-q43",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "For \\(f(x,y)=x^2+4y^2\\) at \\((1,-2)\\), which statements are correct?",
    options: [
      {
        text: "\\(\\nabla f(1,-2)=\\begin{bmatrix}2 \\\\ -16\\end{bmatrix}\\).",
        isCorrect: true,
      },
      {
        text: "The negative-gradient direction is \\(\\begin{bmatrix}-2 \\\\ 16\\end{bmatrix}\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\nabla f(1,-2)=\\begin{bmatrix}1 \\\\ -2\\end{bmatrix}\\) because the gradient equals the input point.",
        isCorrect: false,
      },
      {
        text: "The gradient must point downhill for a loss function.",
        isCorrect: false,
      },
    ],
    explanation:
      "The partial derivatives are \\(2x\\) and \\(8y\\), giving \\((2,-16)\\) at \\((1,-2)\\). The gradient points in the direction of steepest local increase, so the negative gradient points downhill. The gradient is not generally the same as the input point.",
  },
  {
    id: "la-crash-l3-q44",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "For \\(f(x)=(x-3)^2\\), starting at \\(x=1\\) with learning rate \\(\\eta=0.1\\), which statements about one gradient-descent step are correct?",
    options: [
      {
        text: "The gradient at \\(x=1\\) is \\(-4\\).",
        isCorrect: true,
      },
      {
        text: "The update rule is \\(x_{new}=x-\\eta f'(x)\\).",
        isCorrect: true,
      },
      {
        text: "The new value is \\(x_{new}=1.4\\).",
        isCorrect: true,
      },
      {
        text: "This step moves \\(x\\) toward the minimizer \\(x=3\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The derivative is \\(2(x-3)\\), so at \\(x=1\\) it equals \\(-4\\). Gradient descent subtracts the gradient, giving \\(1-0.1(-4)=1.4\\). Since the minimizer is at \\(3\\), the update moves in the correct direction.",
  },
  {
    id: "la-crash-l3-q45",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statement best explains why a learning rate can make training unstable?",
    options: [
      {
        text: "A step size that is too large can overshoot useful downhill regions and increase the loss.",
        isCorrect: true,
      },
      {
        text: "A larger learning rate gives faster convergence across the run.",
        isCorrect: false,
      },
      {
        text: "Learning rate affects the display of a model rather than its parameter changes.",
        isCorrect: false,
      },
      {
        text: "A nonzero learning rate prevents gradients from being computed.",
        isCorrect: false,
      },
    ],
    explanation:
      "The learning rate controls how far parameters move in the negative-gradient direction. If the step is too large, a locally downhill direction can still lead past the useful region and make the loss worse. This is why learning rate affects stability as well as speed.",
  },
  {
    id: "la-crash-l3-q46",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "For \\(y=(3x-2)^4\\), which statements correctly support a chain-rule derivative?",
    options: [
      {
        text: "A useful intermediate variable is \\(u=3x-2\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\frac{dy}{du}=4u^3\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\frac{du}{dx}=3\\).",
        isCorrect: true,
      },
      {
        text: "The derivative is \\(4(3x-2)^3\\) because the inner derivative can be ignored.",
        isCorrect: false,
      },
    ],
    explanation:
      "The chain rule multiplies the outer derivative by the inner derivative. With \\(u=3x-2\\), the outer derivative is \\(4u^3\\) and the inner derivative is \\(3\\), so \\(\\frac{dy}{dx}=12(3x-2)^3\\). Ignoring the inner derivative misses how the inside changes with \\(x\\).",
  },
  {
    id: "la-crash-l3-q47",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "If \\(W\\in\\mathbb{R}^{3\\times5}\\), which statements about the matrix gradient \\(\\frac{\\partial L}{\\partial W}\\) are correct?",
    options: [
      {
        text: "\\(\\frac{\\partial L}{\\partial W}\\) has shape \\(3\\times5\\).",
        isCorrect: true,
      },
      {
        text: "Each entry of \\(\\frac{\\partial L}{\\partial W}\\) describes the local sensitivity of the loss to one weight.",
        isCorrect: true,
      },
      {
        text: "\\(\\frac{\\partial L}{\\partial W}\\) has shape \\(5\\times3\\) because gradients use the transpose of the parameter shape.",
        isCorrect: false,
      },
      {
        text: "A matrix parameter has a single derivative value, so the gradient is a scalar.",
        isCorrect: false,
      },
    ],
    explanation:
      "The gradient with respect to a parameter array has the same shape as that parameter array. Each weight gets its own local slope telling how a small change would affect the loss. Transposes may appear in formulas, but they do not change the final gradient shape for \\(W\\).",
  },
  {
    id: "la-crash-l3-q48",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "For a linear layer \\(y=Wx\\) with upstream gradient \\(g=\\frac{\\partial L}{\\partial y}\\), which statements are correct?",
    options: [
      {
        text: "\\(\\frac{\\partial L}{\\partial W}=gx^T\\).",
        isCorrect: true,
      },
      {
        text: "The weight gradient has one entry for each weight in \\(W\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\frac{\\partial L}{\\partial x}=W^Tg\\).",
        isCorrect: true,
      },
      {
        text: "The entry \\((i,j)\\) of the weight gradient is proportional to \\(g_i x_j\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "For a linear layer, the weight gradient is an outer product between the output error signal and the input activation. The input gradient uses \\(W^T\\) to send the output sensitivity back to the input coordinates. These formulas also provide useful shape checks during implementation.",
  },
  {
    id: "la-crash-l3-q49",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "For \\(f(x,y)=x^2+xy+y^2\\), which statement correctly gives \\(\\nabla f(1,2)\\)?",
    options: [
      {
        text: "\\(\\nabla f(1,2)=\\begin{bmatrix}4 \\\\ 5\\end{bmatrix}\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\nabla f(1,2)=\\begin{bmatrix}2 \\\\ 4\\end{bmatrix}\\).",
        isCorrect: false,
      },
      {
        text: "\\(\\nabla f(1,2)=\\begin{bmatrix}5 \\\\ 4\\end{bmatrix}\\).",
        isCorrect: false,
      },
      {
        text: "\\(\\nabla f(1,2)=7\\) because the function has scalar output.",
        isCorrect: false,
      },
    ],
    explanation:
      "The partial derivatives are \\(\\frac{\\partial f}{\\partial x}=2x+y\\) and \\(\\frac{\\partial f}{\\partial y}=x+2y\\). At \\((1,2)\\), these become \\(4\\) and \\(5\\). A scalar-valued function of two variables has a vector gradient with one component per input variable.",
  },
  {
    id: "la-crash-l3-q50",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the directional derivative of a differentiable function \\(f\\) in a unit direction \\(u\\)?",
    options: [
      {
        text: "It is given by \\(\\nabla f\\cdot u\\).",
        isCorrect: true,
      },
      {
        text: "It is largest when \\(u\\) points in the gradient direction.",
        isCorrect: true,
      },
      {
        text: "It is zero when \\(u\\) is orthogonal to \\(\\nabla f\\).",
        isCorrect: true,
      },
      {
        text: "It ignores the gradient and uses the value \\(f(x)\\) as the step direction.",
        isCorrect: false,
      },
    ],
    explanation:
      "The directional derivative measures the first-order change in a chosen direction and equals the dot product with the gradient. The dot product is largest in the gradient direction and zero for directions perpendicular to the gradient. Function value alone does not determine local slope.",
  },
  {
    id: "la-crash-l3-q51",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "For a computation \\(z=Wx\\), \\(a=\\operatorname{ReLU}(z)\\), followed by a loss \\(L\\), which statements about backpropagation are correct?",
    options: [
      {
        text: "The ReLU derivative can mask gradient components where \\(z\\le 0\\).",
        isCorrect: true,
      },
      {
        text: "Gradients reaching \\(W\\) depend on both the upstream error signal and the input \\(x\\).",
        isCorrect: true,
      },
      {
        text: "The chain rule stops at ReLU because ReLU is not a matrix.",
        isCorrect: false,
      },
      {
        text: "The weight gradient has no relationship to the forward activations.",
        isCorrect: false,
      },
    ],
    explanation:
      "Backpropagation applies the chain rule through both matrix operations and elementwise nonlinearities. ReLU passes gradient through positive pre-activations and blocks it for nonpositive pre-activations in the usual convention. The gradient for \\(W\\) still depends on the input activation because each weight multiplies an input coordinate.",
  },
  {
    id: "la-crash-l3-q52",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "For a value-learning loss \\(L=(t-Q_\\theta(s,a))^2\\), treating target \\(t\\) as fixed, which statements are correct?",
    options: [
      {
        text: "If \\(q=Q_\\theta(s,a)\\), then \\(\\frac{\\partial L}{\\partial q}=2(q-t)\\).",
        isCorrect: true,
      },
      {
        text: "The gradient with respect to \\(\\theta\\) depends on \\(\\nabla_\\theta Q_\\theta(s,a)\\).",
        isCorrect: true,
      },
      {
        text: "The loss gradient is zero when \\(Q_\\theta(s,a)=t\\).",
        isCorrect: true,
      },
      {
        text: "Gradient descent changes parameters in a direction that locally reduces this squared error when the step size is appropriate.",
        isCorrect: true,
      },
    ],
    explanation:
      "Writing \\(q=Q_\\theta(s,a)\\) makes the scalar derivative unambiguous: \\(\\frac{\\partial}{\\partial q}(t-q)^2=2(q-t)\\). To update parameters, the chain rule multiplies this scalar error derivative by how \\(Q_\\theta(s,a)\\) changes with \\(\\theta\\). When prediction equals target, the squared-error gradient vanishes.",
  },
  {
    id: "la-crash-l3-q53",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "For \\(f(x,y)=(x-2)^2+(y+1)^2\\), starting at \\((0,0)\\) with \\(\\eta=0.25\\), which statement gives one gradient-descent update?",
    options: [
      {
        text: "The update moves to \\((1,-0.5)\\).",
        isCorrect: true,
      },
      {
        text: "The update moves to \\((-1,0.5)\\).",
        isCorrect: false,
      },
      {
        text: "The update stays at \\((0,0)\\) because the loss is already minimized.",
        isCorrect: false,
      },
      {
        text: "The update moves to \\((2,-1)\\) in one step for the usual step sizes.",
        isCorrect: false,
      },
    ],
    explanation:
      "The gradient is \\((2(x-2),2(y+1))\\), so at \\((0,0)\\) it is \\((-4,2)\\). Gradient descent gives \\((0,0)-0.25(-4,2)=(1,-0.5)\\). This moves toward the minimizer \\((2,-1)\\) but does not necessarily reach it in one step.",
  },
  {
    id: "la-crash-l3-q54",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Let \\(h=Wx\\) and \\(L=\\frac{1}{2}\\|h-y\\|^2\\). Which gradient statements are correct?",
    options: [
      {
        text: "\\(\\frac{\\partial L}{\\partial h}=h-y\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\frac{\\partial L}{\\partial W}=(h-y)x^T\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\frac{\\partial L}{\\partial x}=W^T(h-y)\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\frac{\\partial L}{\\partial W}\\) must be a scalar because \\(L\\) is scalar.",
        isCorrect: false,
      },
    ],
    explanation:
      "The squared-error derivative with respect to the prediction vector is the residual \\(h-y\\). The chain rule through \\(h=Wx\\) gives an outer product for the weight gradient and a transpose multiplication for the input gradient. A scalar loss can have a matrix-shaped gradient when the parameter is a matrix.",
  },
  {
    id: "la-crash-l3-q55",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe what a zero gradient does and does not imply?",
    options: [
      {
        text: "A zero gradient means the first-order local slope is zero.",
        isCorrect: true,
      },
      {
        text: "A saddle point can have zero gradient without being a local minimum.",
        isCorrect: true,
      },
      {
        text: "A zero gradient is enough to identify the best possible parameters.",
        isCorrect: false,
      },
      {
        text: "A zero gradient means the loss function must be constant everywhere.",
        isCorrect: false,
      },
    ],
    explanation:
      "The gradient captures first-order local change, so a zero gradient indicates no linear slope at that point. It does not classify the point as a global optimum, because the point could be a saddle, a local maximum, or a flat non-optimal region. Additional curvature or global information is needed.",
  },
  {
    id: "la-crash-l3-q56",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "For a matrix product \\(C=AB\\) and scalar loss \\(L\\), let \\(G=\\frac{\\partial L}{\\partial C}\\). Which backpropagation formulas and shape facts are correct?",
    options: [
      {
        text: "\\(\\frac{\\partial L}{\\partial A}=GB^T\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\frac{\\partial L}{\\partial B}=A^TG\\).",
        isCorrect: true,
      },
      {
        text: "The transpose on \\(B\\) helps the gradient with respect to \\(A\\) match the shape of \\(A\\).",
        isCorrect: true,
      },
      {
        text: "The transpose on \\(A\\) helps the gradient with respect to \\(B\\) match the shape of \\(B\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "Matrix-product gradients are a compact example of the chain rule with shape constraints. The upstream gradient \\(G\\) has the same shape as \\(C\\), and multiplying by the appropriate transpose sends that sensitivity back to each factor. These transpose patterns are one reason shape reasoning is essential in backpropagation.",
  },
  {
    id: "la-crash-l3-q57",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "For \\(f(x)=\\frac{1}{2}x^THx\\) with \\(H=\\begin{bmatrix}1 & 0 \\\\ 0 & 10\\end{bmatrix}\\), which statement correctly describes fixed-step gradient descent?",
    options: [
      {
        text: "A learning rate below \\(0.2\\) is required for convergence in the steepest eigendirection of this quadratic.",
        isCorrect: true,
      },
      {
        text: "Any positive learning rate converges because the gradient points uphill.",
        isCorrect: false,
      },
      {
        text: "The largest curvature direction has little connection to stability.",
        isCorrect: false,
      },
      {
        text: "A learning rate of \\(1\\) is safe for this quadratic.",
        isCorrect: false,
      },
    ],
    explanation:
      "For this diagonal quadratic, the steep direction has curvature \\(10\\), and fixed-step gradient descent is stable along that direction only when \\(0<\\eta<\\frac{2}{10}=0.2\\). This illustrates why high-curvature directions constrain learning rates. The gradient direction alone does not make every step size safe.",
  },
  {
    id: "la-crash-l3-q58",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe vanishing or exploding gradients through many composed layers?",
    options: [
      {
        text: "Backpropagation multiplies local derivative or Jacobian factors along computational paths.",
        isCorrect: true,
      },
      {
        text: "Repeated factors with singular values mostly below \\(1\\) can shrink gradients.",
        isCorrect: true,
      },
      {
        text: "Repeated factors with singular values mostly above \\(1\\) can amplify gradients.",
        isCorrect: true,
      },
      {
        text: "Each layer receives the same gradient because the layers belong to one network.",
        isCorrect: false,
      },
    ],
    explanation:
      "The chain rule propagates gradients through products of local sensitivities. Products of many shrinking factors can make gradients very small, while products of many amplifying factors can make them very large. Different layers can therefore receive very different gradient magnitudes.",
  },
  {
    id: "la-crash-l3-q59",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe mini-batch gradients in neural-network training?",
    options: [
      {
        text: "A mini-batch gradient averages or sums per-example gradient contributions for the examples in that batch.",
        isCorrect: true,
      },
      {
        text: "Batch matrix operations let many gradient contributions be computed in parallel.",
        isCorrect: true,
      },
      {
        text: "A mini-batch gradient matches the full-dataset gradient on a typical step.",
        isCorrect: false,
      },
      {
        text: "Mini-batching removes the need for the chain rule.",
        isCorrect: false,
      },
    ],
    explanation:
      "Mini-batch training estimates the full gradient using a subset of examples, usually by averaging their contributions. Matrix operations make these computations efficient on hardware such as GPUs. The estimate can differ from the full-dataset gradient, but it still relies on the same chain-rule machinery.",
  },
  {
    id: "la-crash-l3-q60",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly qualify the idea that learning is optimization using gradients?",
    options: [
      {
        text: "Gradients give local information about how parameters affect loss.",
        isCorrect: true,
      },
      {
        text: "Backpropagation efficiently computes many parameter gradients by reusing intermediate derivatives.",
        isCorrect: true,
      },
      {
        text: "Gradient descent does not guarantee a global optimum for every deep network.",
        isCorrect: true,
      },
      {
        text: "Matrix gradients and chain rule reasoning are central when parameters are matrices.",
        isCorrect: true,
      },
    ],
    explanation:
      "Gradients are local tools: they tell how to change parameters to improve the loss near the current point. Backpropagation makes this practical by sharing computations across many parameters, including matrices. The method is powerful, but local gradient information alone does not guarantee global optimality in every deep-learning landscape.",
  },
];

export const CrashCourseLinearAlgebraL3Questions =
  CrashCourseLinearAlgebraLecture3Questions;
