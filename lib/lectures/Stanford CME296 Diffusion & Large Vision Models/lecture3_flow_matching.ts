import { Question } from "../../quiz";

type Difficulty = "easy" | "medium" | "hard";
type OptionSeed = readonly [text: string, isCorrect: boolean];
type AssertionReasonChoice = 1 | 2 | 3 | 4 | 5;

const assertionReasonOptionTexts = [
  "Assertion is true, Reason is false.",
  "Assertion is false, Reason is true.",
  "Both are false.",
  "Both are true, and the Reason is the correct explanation of the Assertion.",
  "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
] as const;

function makeQuestion(
  id: string,
  difficulty: Difficulty,
  prompt: string,
  optionSeeds: readonly OptionSeed[],
  explanation: string,
): Question {
  return {
    id,
    chapter: 3,
    difficulty,
    prompt,
    options: optionSeeds.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

function makeAssertionReasonQuestion(
  id: string,
  difficulty: Difficulty,
  assertion: string,
  reason: string,
  correctChoice: AssertionReasonChoice,
  explanation: string,
): Question {
  return {
    id,
    chapter: 3,
    difficulty,
    type: "assertion-reason",
    prompt: "Assertion: " + assertion + "\n\nReason: " + reason,
    options: assertionReasonOptionTexts.map((text, index) => ({
      text,
      isCorrect: index + 1 === correctChoice,
    })),
    explanation,
  };
}

export const stanfordCME296Lecture3FlowMatchingQuestions: Question[] = [
  makeQuestion(
    "cme296-lect3-q01",
    "easy",
    "In flow matching, what does a trajectory \\(x_t\\) describe?",
    [
      [
        "The path of one observation through state space as time changes.",
        true,
      ],
      [
        "The probability density of the entire population at one fixed time.",
        false,
      ],
      ["The scalar training loss averaged over all data points.", false],
      [
        "The gradient of log density evaluated without a time coordinate.",
        false,
      ],
    ],
    "A trajectory is the microscopic path followed by one sample, with \\(x_t\\) naming its state at time \\(t\\). The population density is the probability path, the loss is an optimization objective, and a score is a different vector quantity; confusing these levels obscures how individual motion transports a distribution.",
  ),
  makeQuestion(
    "cme296-lect3-q02",
    "easy",
    "Which descriptions belong to a flow \\(\\psi_t\\) rather than to a single trajectory?",
    [
      [
        "It maps many possible initial points to their states at time \\(t\\).",
        true,
      ],
      [
        "It is the collection of trajectories induced by shared dynamics.",
        true,
      ],
      [
        "It is one sampled endpoint \\(x_1\\) drawn from the target dataset.",
        false,
      ],
      [
        "It is necessarily the gradient of a scalar probability density.",
        false,
      ],
    ],
    "A flow is a family of maps acting on all initial conditions, so each starting point obtains a trajectory under the same time-dependent dynamics. One endpoint is merely a sample, and a general velocity field need not be a score or even be expressible as the gradient of a scalar density.",
  ),
  makeQuestion(
    "cme296-lect3-q03",
    "easy",
    "Which statements correctly interpret the probability path \\(p_t(x)\\)?",
    [
      [
        "It describes the distribution of transported observations at time \\(t\\).",
        true,
      ],
      [
        "Its endpoints are chosen to match the initial noise and target data distributions.",
        true,
      ],
      [
        "It gives a population-level view rather than the route of one particle.",
        true,
      ],
      [
        "It assigns one deterministic destination to every initial sample by itself.",
        false,
      ],
    ],
    "The probability path tracks how the full density changes between prescribed endpoint distributions and is therefore a macroscopic description. A particular coupling or flow is additionally needed to say which initial sample reaches which endpoint; many different trajectories can induce the same sequence of marginal densities.",
  ),
  makeQuestion(
    "cme296-lect3-q04",
    "medium",
    "Which properties correctly characterize a time-dependent vector field \\(u_t(x)\\)?",
    [
      [
        "It assigns a direction of motion at state \\(x\\) and time \\(t\\).",
        true,
      ],
      [
        "Its magnitude controls instantaneous speed along an ordinary differential equation trajectory.",
        true,
      ],
      [
        "A trajectory satisfying \\(d x_t/dt=u_t(x_t)\\) follows the field locally.",
        true,
      ],
      [
        "When compatible with the continuity equation, it transports the corresponding probability path.",
        true,
      ],
    ],
    "The vector field is the local velocity rule: direction and magnitude determine how each sample changes under the ODE. At the distribution level, compatibility with the continuity equation ensures that the induced flow moves probability mass according to the desired path rather than creating or destroying density arbitrarily.",
  ),
  makeAssertionReasonQuestion(
    "cme296-lect3-q05",
    "medium",
    "A flow-matching implementation can be mathematically correct while using \\(t=0\\) for noise and \\(t=1\\) for clean data.",
    "The labels assigned to the time endpoints are conventional as long as the probability path, targets, and integration direction use the same convention.",
    4,
    "Both statements are true, and the reason explains why. Some diffusion presentations place clean data at the low-time endpoint and noise at the high-time endpoint, while this flow convention reverses them; formulas should be translated consistently rather than compared by the symbol \\(t\\) alone.",
  ),
  makeQuestion(
    "cme296-lect3-q06",
    "medium",
    "Which statement best distinguishes velocity from score at a point on a probability path?",
    [
      [
        "Velocity specifies temporal transport, whereas the score is the spatial gradient of log density at that time.",
        true,
      ],
      [
        "Velocity and score are identical by definition for every generative path.",
        false,
      ],
      [
        "Velocity is a scalar density, whereas the score is the probability normalization constant.",
        false,
      ],
      [
        "Velocity is defined only at the endpoints, whereas the score is defined only between them.",
        false,
      ],
    ],
    "Velocity answers where a particle should move as time advances, while the score points in the spatial direction of increasing log density for the current marginal. Special constructions can relate them, but they are not synonymous, scalar normalizers, or quantities restricted to complementary parts of the interval.",
  ),
  makeQuestion(
    "cme296-lect3-q07",
    "hard",
    "Two smooth vector fields induce the same endpoint distributions but trace different intermediate paths. Which conclusions are valid?",
    [
      [
        "Endpoint matching alone does not uniquely identify the intermediate trajectories.",
        true,
      ],
      [
        "Both fields can be valid transports if each satisfies its own compatible continuity equation and boundary conditions.",
        true,
      ],
      [
        "Their velocities must agree at every intermediate state because their endpoints agree.",
        false,
      ],
      [
        "Their numerical integration costs must be identical because both end at the data distribution.",
        false,
      ],
    ],
    "Transport between two distributions is not unique: different couplings and vector fields can bend, stretch, or route mass differently while preserving the same endpoints. Validity depends on the induced density evolution, and different curvature or stiffness can yield different solver costs even when terminal distributions coincide.",
  ),
  makeQuestion(
    "cme296-lect3-q08",
    "hard",
    "Which relationships connect microscopic trajectories and macroscopic density evolution in flow matching?",
    [
      [
        "The ODE \\(d x_t/dt=u_t(x_t)\\) describes motion of individual samples.",
        true,
      ],
      [
        "Pushing an initial random variable through the flow induces the marginal \\(p_t\\).",
        true,
      ],
      [
        "The continuity equation expresses conservation of probability mass under the vector field.",
        true,
      ],
      [
        "The particle ODE can create extra total probability mass when trajectories converge.",
        false,
      ],
    ],
    "The first three statements connect the particle, mapping, and density views of the same dynamics: an ODE creates trajectories, the flow pushes distributions forward, and the continuity equation governs the resulting density. Trajectories may concentrate locally, but a valid transport redistributes probability rather than creating additional total mass when paths converge.",
  ),
  makeQuestion(
    "cme296-lect3-q09",
    "easy",
    "For a target example \\(x_1\\), consider \\(p_t(x\mid x_1)=\\mathcal N(t x_1,(1-t)^2I)\\). Which endpoint statements are correct?",
    [
      [
        "At \\(t=0\\), the conditional distribution is standard Gaussian noise.",
        true,
      ],
      [
        "As \\(t\\) approaches one, the variance contracts toward zero around \\(x_1\\).",
        true,
      ],
      [
        "At \\(t=1\\), the limiting distribution is a point mass at \\(x_1\\).",
        true,
      ],
      [
        "For \\(0<t<1\\), its covariance remains isotropic with scale \\((1-t)^2\\).",
        true,
      ],
    ],
    "The conditional Gaussian begins with mean zero and identity covariance, then moves its mean toward the selected data point while retaining isotropic covariance that shrinks as \\((1-t)^2\\). Its limiting endpoint is therefore a Dirac mass at \\(x_1\\), consistent with the noise-at-zero and data-at-one convention.",
  ),
  makeAssertionReasonQuestion(
    "cme296-lect3-q10",
    "medium",
    "The conditional velocity \\(u_t(x\mid x_1)=(x_1-x)/(1-t)\\) must diverge along every sample from the linear conditional path as \\(t\\to1\\).",
    "Along \\(x_t=t x_1+(1-t)x_0\\), substituting gives \\(u_t(x_t\mid x_1)=x_1-x_0\\).",
    2,
    "The assertion is false and the reason is true. Although the field formula contains \\(1-t\\) in the denominator, the numerator along a valid conditional trajectory is \\((1-t)(x_1-x_0)\\), so the apparent singularity cancels and the path has finite constant velocity.",
  ),
  makeQuestion(
    "cme296-lect3-q11",
    "medium",
    "If \\(x_0=-2\\), \\(x_1=6\\), and \\(t=0.25\\), what is the scalar state on the linear conditional path \\(x_t=(1-t)x_0+t x_1\\)?",
    [
      ["\\(x_t=0\\)", true],
      ["\\(x_t=2\\)", false],
      ["\\(x_t=4\\)", false],
      ["\\(x_t=-1\\)", false],
    ],
    "The interpolation weights the initial noise by \\(0.75\\) and the target by \\(0.25\\), giving \\(0.75(-2)+0.25(6)=-1.5+1.5=0\\). The other values arise from using equal weights, reversing the weights, or applying the time fraction to only one endpoint.",
  ),
  makeQuestion(
    "cme296-lect3-q12",
    "hard",
    "For the path \\(x_t=(1-t)x_0+t x_1\\), which calculations correctly recover its conditional velocity?",
    [
      [
        "Differentiating with respect to time gives \\(d x_t/dt=x_1-x_0\\).",
        true,
      ],
      [
        "Substituting \\(x_t\\) into \\((x_1-x_t)/(1-t)\\) also gives \\(x_1-x_0\\).",
        true,
      ],
      [
        "Differentiating gives \\(t(x_1-x_0)\\), so speed vanishes at the noise endpoint.",
        false,
      ],
      [
        "The velocity equals \\(x_t\\), because a trajectory is its own time derivative.",
        false,
      ],
    ],
    "A linear interpolation has a constant derivative equal to the endpoint displacement, and the state-dependent conditional field reduces to that same value along the trajectory. Multiplying by time or equating position with velocity confuses the path formula with its derivative and would not reach the target in one unit of time.",
  ),
  makeQuestion(
    "cme296-lect3-q13",
    "easy",
    "How is the marginal probability path formed from target-conditioned paths?",
    [
      [
        "Average \\(p_t(x\mid x_1)\\) over target examples \\(x_1\\) drawn from the data distribution.",
        true,
      ],
      [
        "The result is a mixture whose components correspond to possible target data points.",
        true,
      ],
      [
        "At the final endpoint, the mixture recovers the target data distribution.",
        true,
      ],
      [
        "Choose the single nearest target and discard all other conditional paths before defining the density.",
        false,
      ],
    ],
    "Marginalization integrates the conditional density against the data distribution, producing an aggregate probability path that begins at noise and ends at data. It retains contributions from all possible targets weighted by their probability; selecting only the nearest target would define a different hard assignment rather than the required marginal mixture.",
  ),
  makeQuestion(
    "cme296-lect3-q14",
    "hard",
    "At an intermediate state \\(x\\), how does the marginal vector field combine conditional vector fields associated with possible destinations \\(x_1\\)?",
    [
      [
        "It weights each conditional velocity by the posterior plausibility of that destination given the current state.",
        true,
      ],
      [
        "It forms a conditional expectation of velocity over possible \\(x_1\\).",
        true,
      ],
      [
        "Destinations whose conditional paths assign more density near \\(x\\) receive greater influence, all else equal.",
        true,
      ],
      [
        "The posterior weights normalize across destinations rather than being arbitrary unscaled scores.",
        true,
      ],
    ],
    "The marginal field is the posterior mean \\(u_t(x)=\\mathbb E[u_t(x\mid x_1)\mid x_t=x]\\), so each candidate destination contributes according to how compatible it is with the observed intermediate state. Normalized Bayesian weights make the average a valid conditional expectation and connect the tractable conditional fields to population transport.",
  ),
  makeAssertionReasonQuestion(
    "cme296-lect3-q15",
    "hard",
    "The posterior-weighted marginal vector field generates the marginal probability path obtained by mixing the conditional paths.",
    "Averaging conditional velocities with the same posterior weights that decompose the marginal probability flux preserves the continuity equation after marginalization.",
    4,
    "Both statements are true, and the reason describes the distribution-level mechanism. Conditional paths each satisfy their transport equation; integrating their probability fluxes over targets and dividing by the marginal density yields the posterior-mean velocity whose flux matches the derivative of the mixed marginal path.",
  ),
  makeQuestion(
    "cme296-lect3-q16",
    "medium",
    "Why is directly supervising a neural network with the marginal vector field generally impractical?",
    [
      [
        "Evaluating it exactly requires integrating posterior-weighted contributions over the unknown data distribution at each state.",
        true,
      ],
      [
        "A vector field cannot be represented by a neural network because its output has more than one component.",
        false,
      ],
      [
        "The marginal field is undefined at every intermediate time by construction.",
        false,
      ],
      [
        "Observed target samples contain exact marginal velocities as metadata.",
        false,
      ],
    ],
    "The desired marginal velocity exists, but computing its posterior average at an arbitrary state would require access to the full data distribution and its mixture density. Neural networks can represent vector outputs, and individual examples do not arrive with the intractable population average attached, which motivates conditional flow matching.",
  ),
  makeQuestion(
    "cme296-lect3-q17",
    "hard",
    "Which statements explain why conditional flow matching gives a tractable surrogate for marginal flow matching?",
    [
      [
        "A target example and a noise sample let training sample from a conditional path without evaluating the full marginal density.",
        true,
      ],
      [
        "The conditional target velocity is analytic for the chosen Gaussian interpolation, and the two objectives have the same parameter-dependent optimum.",
        true,
      ],
      [
        "The surrogate works because every conditional velocity equals the marginal velocity pointwise.",
        false,
      ],
      [
        "The surrogate removes the expectation over time, data, and noisy states from training.",
        false,
      ],
    ],
    "Conditional flow matching replaces an inaccessible posterior mean with sampled analytic conditional velocities, and expanding the squared losses shows that their parameter-dependent terms agree up to a constant. Individual conditional velocities need not equal the marginal field at a point, and stochastic expectations over time, targets, and path states remain essential.",
  ),
  makeQuestion(
    "cme296-lect3-q18",
    "easy",
    "Which random quantities are sampled to construct a standard conditional flow-matching training example for a linear path?",
    [
      ["A time \\(t\\) in the interpolation interval.", true],
      ["An initial noise sample \\(x_0\\).", true],
      ["A target data sample \\(x_1\\).", true],
      [
        "A complete numerical ODE trajectory produced by the current network.",
        false,
      ],
    ],
    "The analytic interpolation constructs \\(x_t\\) and its target velocity directly from a sampled time, noise endpoint, and data endpoint. Training therefore does not need to solve the model ODE inside every update; numerical integration is chiefly needed later when the learned field generates new samples.",
  ),
  makeQuestion(
    "cme296-lect3-q19",
    "medium",
    "Which statements correctly describe the linear conditional flow-matching loss?",
    [
      [
        "The network receives an interpolated state \\(x_t\\) and time \\(t\\).",
        true,
      ],
      [
        "Its regression target can be the endpoint displacement \\(x_1-x_0\\).",
        true,
      ],
      [
        "The loss can be a squared norm between predicted and conditional velocity.",
        true,
      ],
      [
        "Gradients update a shared time-conditioned vector field across the path.",
        true,
      ],
    ],
    "For the linear Gaussian path, sampling endpoints makes both the intermediate state and constant conditional velocity available analytically. Squared-error regression across random times trains one time-conditioned network to approximate the marginal transport after averaging, without requiring a separately learned vector field for every time slice.",
  ),
  makeAssertionReasonQuestion(
    "cme296-lect3-q20",
    "medium",
    "Conditional flow matching can train on randomly paired noise and data samples without declaring that each pair is the uniquely correct generative correspondence.",
    "For the linear interpolation, each sampled pair has the time-independent conditional velocity target \\(x_1-x_0\\).",
    5,
    "Both statements are true, but constant velocity for an individual linear pair does not by itself explain why the pairing need not be a unique correspondence. That conclusion comes from averaging tractable targets over many random pairs to learn a distributional marginal field rather than memorizing a noise-to-example lookup table.",
  ),
  makeQuestion(
    "cme296-lect3-q21",
    "hard",
    "For one-dimensional endpoints \\(x_0=3\\) and \\(x_1=-1\\), what conditional velocity target is used along the linear path?",
    [
      ["\\(x_1-x_0=-4\\)", true],
      ["\\(x_0-x_1=4\\)", false],
      ["\\((x_0+x_1)/2=1\\)", false],
      ["\\(t(x_1-x_0)=-4t\\)", false],
    ],
    "Differentiating \\(x_t=(1-t)x_0+t x_1\\) gives the constant target \\(x_1-x_0=-1-3=-4\\). Reversing the subtraction points back toward noise, averaging endpoints gives a location rather than a velocity, and multiplying by time would incorrectly vary the speed of a linear trajectory.",
  ),
  makeQuestion(
    "cme296-lect3-q22",
    "easy",
    "Which operations belong to basic flow-matching inference?",
    [
      ["Draw \\(x_0\\) from the initial noise distribution.", true],
      [
        "Integrate the learned velocity field from the noise-time endpoint toward the data-time endpoint.",
        true,
      ],
      [
        "Select a clean training example and reveal it gradually to the sampler.",
        false,
      ],
      [
        "Add a fresh mandatory Gaussian perturbation after every Euler step.",
        false,
      ],
    ],
    "Generation starts from new noise and solves an ODE driven by the learned field toward the target endpoint. It does not require a hidden training image, and ordinary flow matching is deterministic once the initial noise and solver are fixed; mandatory per-step stochastic noise belongs to an SDE-style sampler rather than Euler integration of this ODE.",
  ),
  makeQuestion(
    "cme296-lect3-q23",
    "medium",
    "For Euler integration of \\(d x/dt=v_\\theta(x,t)\\) with step \\(h\\), which statements are correct?",
    [
      ["The update is \\(x_{t+h}\\approx x_t+h v_\\theta(x_t,t)\\).", true],
      ["The velocity is reevaluated at the new state on the next step.", true],
      [
        "Halving \\(h\\) over the same interval roughly doubles the number of function evaluations.",
        true,
      ],
      [
        "Euler integration evaluates the exact future velocity \\(v_\\theta(x_{t+h},t+h)\\) before constructing \\(x_{t+h}\\).",
        false,
      ],
    ],
    "Explicit Euler uses the current state and time to extrapolate one step, then evaluates the field again after reaching the approximation. A smaller step usually requires more evaluations and can reduce discretization error; using the unknown future velocity in the same update would describe an implicit method, not basic explicit Euler.",
  ),
  makeQuestion(
    "cme296-lect3-q24",
    "hard",
    "Which factors can make a flow-matching sampler inaccurate even if its training loss is low?",
    [
      [
        "A coarse Euler grid can poorly approximate a curved or rapidly changing learned trajectory.",
        true,
      ],
      [
        "Training error in regions actually visited by generated trajectories can accumulate during integration.",
        true,
      ],
      [
        "A mismatch between training and inference time conventions can reverse the intended transport.",
        true,
      ],
      [
        "The chosen initial distribution may fail to match the endpoint assumed during training.",
        true,
      ],
    ],
    "Regression quality alone does not guarantee numerical or deployment correctness. Solver discretization, compounding field error, reversed endpoint conventions, and a mismatched initial prior each change the path followed at inference, so all four must be checked when samples fail despite an apparently small average loss.",
  ),
  makeAssertionReasonQuestion(
    "cme296-lect3-q25",
    "easy",
    "A standard flow-matching sampler is deterministic after its initial noise sample and numerical schedule are fixed.",
    "It generates by solving an ordinary differential equation rather than requiring a fresh Wiener increment at every step.",
    4,
    "Both statements are true, and the reason explains the determinism. Randomness enters through the initial draw, but the learned velocity and an explicit fixed solver determine subsequent states; reverse-SDE samplers instead include new stochastic increments unless converted to a deterministic counterpart.",
  ),
  makeQuestion(
    "cme296-lect3-q26",
    "easy",
    "Which single learned quantity is most directly associated with each paradigm?",
    [
      [
        "Flow matching learns a velocity field that transports samples through time.",
        true,
      ],
      [
        "Discrete DDPM learns only the data density normalization constant.",
        false,
      ],
      [
        "Score-based diffusion learns a fixed clean image for every noise seed.",
        false,
      ],
      [
        "All three paradigms necessarily learn the same vector with the same units and interpretation.",
        false,
      ],
    ],
    "Flow matching directly regresses temporal velocity, while discrete DDPM is commonly parameterized through noise prediction and score-based diffusion learns a spatial log-density gradient. These quantities can be transformed or related under particular paths, but their definitions and roles are not automatically identical.",
  ),
  makeQuestion(
    "cme296-lect3-q27",
    "medium",
    "Which generation-process pairings are correct?",
    [
      [
        "Score-based diffusion can use a reverse-time SDE with a learned score.",
        true,
      ],
      ["Flow matching uses an ODE driven by a learned velocity field.", true],
      [
        "Discrete DDPM generation is defined as a forward corruption chain from clean data.",
        false,
      ],
      [
        "Flow-matching inference must use the exact training endpoint \\(x_1\\) as an input.",
        false,
      ],
    ],
    "Score models supply the correction needed to simulate reverse stochastic dynamics, while flow models directly provide an ODE velocity. DDPM generation runs its learned reverse chain rather than forward corruption, and flow inference begins from new noise without knowing the future target example used to form any training pair.",
  ),
  makeQuestion(
    "cme296-lect3-q28",
    "hard",
    "Which deterministic counterparts or interpretations are correctly paired?",
    [
      [
        "DDIM supplies a deterministic sampling path compatible with a DDPM-trained noise predictor.",
        true,
      ],
      [
        "A probability-flow ODE provides a deterministic counterpart to score-based SDE dynamics with matching marginals under ideal conditions.",
        true,
      ],
      [
        "Ordinary flow matching already defines deterministic ODE dynamics once the starting noise is fixed.",
        true,
      ],
      [
        "Deterministic means every generated sample is identical even when the initial noise changes.",
        false,
      ],
    ],
    "DDIM and the probability-flow ODE remove inter-step stochasticity from two diffusion formulations, while flow matching starts from an ODE description. Determinism is conditional on the initial state: different noise draws can still map to different samples, so eliminating later random increments does not collapse the model to one universal output.",
  ),
  makeQuestion(
    "cme296-lect3-q29",
    "hard",
    "What does a stochastic-interpolant viewpoint contribute when comparing flows and diffusions?",
    [
      [
        "It specifies a time-indexed random bridge between endpoint distributions.",
        true,
      ],
      [
        "It can expose both velocity-like transport and score-like density information associated with the same path.",
        true,
      ],
      [
        "It provides a common language for deterministic and stochastic generative dynamics.",
        true,
      ],
      [
        "It emphasizes that several couplings can connect the same endpoint distributions.",
        true,
      ],
    ],
    "A stochastic interpolant starts from a chosen random construction connecting endpoints and lets one derive complementary descriptions of its evolving law. That perspective clarifies how ODE velocities, density scores, and SDE formulations can be related without claiming that their learned targets are definitionally the same or that endpoints determine a unique coupling.",
  ),
  makeQuestion(
    "cme296-lect3-q30",
    "easy",
    "Which statements are true for the straight conditional interpolation \\(x_t=(1-t)x_0+t x_1\\)?",
    [
      ["At \\(t=0\\), the state equals \\(x_0\\).", true],
      ["At \\(t=1\\), the state equals \\(x_1\\).", true],
      ["At \\(t=1/2\\), the state is the midpoint \\((x_0+x_1)/2\\).", true],
      ["Its velocity is constant and equals \\(x_1-x_0\\).", true],
    ],
    "The interpolation is the affine line segment between its endpoints, so evaluating it at zero, one, and one half gives the initial point, target point, and midpoint. Differentiation removes the time-dependent weights and yields the constant endpoint displacement, which is the conditional training target used by the straight path.",
  ),
  makeQuestion(
    "cme296-lect3-q31",
    "easy",
    "At an ambiguous intermediate state, two possible targets have posterior probabilities 0.75 and 0.25 and conditional velocities 2 and -2. What is the marginal velocity?",
    [
      ["\\(0.75(2)+0.25(-2)=1\\)", true],
      ["\\(2+(-2)=0\\)", false],
      ["\\(0.75-0.25=0.5\\)", false],
      ["\\(2/0.75-2/0.25\\approx-5.33\\)", false],
    ],
    "The marginal field is the posterior-weighted mean of conditional velocities, giving \\(1.5-0.5=1\\). An unweighted sum ignores destination plausibility, subtracting probabilities omits velocities, and dividing velocities by their weights is not the conditional-expectation rule.",
  ),
  makeQuestion(
    "cme296-lect3-q32",
    "medium",
    "What is required for conditional and marginal flow-matching squared losses to train the same optimal vector field?",
    [
      [
        "The conditional target's expectation given \\(x_t=x\\) must equal the marginal velocity at that state and time.",
        true,
      ],
      [
        "Any difference between the expanded objectives that depends only on the sampled target variance, not on model parameters, does not change the optimizer.",
        true,
      ],
      [
        "Every sampled conditional target must equal the marginal velocity exactly before averaging.",
        false,
      ],
      [
        "The neural prediction must be independent of \\(x_t\\) so posterior averaging is unnecessary.",
        false,
      ],
    ],
    "Squared regression learns a conditional mean: if averaging the analytic conditional velocity at a given state and time produces the desired marginal field, then model-dependent cross terms match. Residual conditional variance can add a parameter-independent constant, while pointwise equality and an input-independent predictor are neither required nor generally true.",
  ),
  makeQuestion(
    "cme296-lect3-q33",
    "hard",
    "A paper defines \\(t=0\\) as clean data and \\(t=1\\) as noise, while a flow implementation uses the opposite convention. Which translations are necessary before comparing formulas?",
    [
      ["Map time by a reversal such as \\(s=1-t\\).", true],
      [
        "Reverse the integration direction and account for the resulting sign change in temporal velocity.",
        true,
      ],
      [
        "Match which endpoint distribution each coefficient represents before comparing schedules.",
        true,
      ],
      [
        "Assume identically named \\(t\\) values denote the same noise level without checking endpoints.",
        false,
      ],
    ],
    "Changing from \\(t\\) to \\(1-t\\) swaps endpoints and changes the derivative sign, so both schedule interpretation and vector direction must be translated. Comparing formulas symbol by symbol without this mapping can make equivalent dynamics appear contradictory or, worse, make a sampler integrate from data toward noise.",
  ),
];
