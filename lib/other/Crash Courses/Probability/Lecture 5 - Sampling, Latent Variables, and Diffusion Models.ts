import { Question } from "../../../quiz";

type OptionSpec = readonly [text: string, isCorrect: boolean];

function makeQuestion(
  id: string,
  difficulty: Question["difficulty"],
  prompt: string,
  options: readonly [OptionSpec, OptionSpec, OptionSpec, OptionSpec],
  explanation: string,
): Question {
  return {
    id,
    chapter: 5,
    difficulty,
    prompt,
    options: options.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

export const CrashCourseProbabilityL5Questions: Question[] = [
  // Sampling, greedy decoding, and repeated draws
  makeQuestion(
    "crash-probability-l5-q61",
    "easy",
    "A token distribution is cat 0.5, dog 0.3, and fox 0.2. What does sampling from it mean?",
    [
      ["Randomly drawing one token with the listed probabilities", true],
      [
        "Treating the modal token cat as the definition of a categorical draw",
        false,
      ],
      ["Averaging the three token names into one output", false],
      ["Retraining the model until one probability becomes one", false],
    ],
    "Sampling turns a probability distribution into one random realized outcome while preserving positive chances for all three tokens. Always choosing cat is greedy decoding, not sampling, and neither averaging labels nor retraining is part of the draw.",
  ),
  makeQuestion(
    "crash-probability-l5-q62",
    "easy",
    "The same cat/dog/fox distribution is sampled 1,000 independent times. Which expectations are correct?",
    [
      ["Cat should appear about 500 times.", true],
      ["Dog should appear about 300 times.", true],
      [
        "Constraining each block of ten samples to contain exactly five cats.",
        false,
      ],
      [
        "Assigning fox zero sampling mass because it is not the modal token.",
        false,
      ],
    ],
    "Expected counts are sample size times probability, so cat and dog have expected counts 500 and 300. Random frequencies fluctuate in finite runs, and every positive-probability outcome—including fox—can be sampled.",
  ),
  makeQuestion(
    "crash-probability-l5-q63",
    "medium",
    "A model distribution is \\((0.45,0.35,0.20)\\). Which statements correctly compare greedy decoding and sampling?",
    [
      ["Greedy decoding selects the first outcome.", true],
      ["Sampling can select any of the three outcomes.", true],
      [
        "Repeated sampling should approach the full 45/35/20 frequency split.",
        true,
      ],
      [
        "Greedy decoding is a random draw whose frequency vector approaches \\((0.45,0.35,0.20)\\).",
        false,
      ],
    ],
    "Greedy decoding deterministically takes the argmax for this step, whereas sampling realizes the categorical distribution. Over repeated comparable draws, sampling frequencies approach the probabilities; greedy outputs remain concentrated on the first outcome.",
  ),
  makeQuestion(
    "crash-probability-l5-q64",
    "hard",
    "Three independent samples are drawn from outcomes A and B with probabilities 0.7 and 0.3. Which calculations are correct for exactly two A outcomes?",
    [
      ["There are \\({3\\choose2}=3\\) positions for the single B.", true],
      ["The probability is \\(3(0.7)^2(0.3)=0.441\\).", true],
      [
        "The probability is \\((0.7)^2(0.3)=0.147\\) because one arrangement represents the full count event.",
        false,
      ],
      [
        "The probability is \\(2/3\\) because two of the three outcomes are A.",
        false,
      ],
    ],
    "Each particular AAB ordering has probability \\(0.7^2 0.3\\), and three distinct orderings satisfy the count event. Omitting the combination factor counts only one sequence, while the observed fraction two thirds is not the model probability.",
  ),
  makeQuestion(
    "crash-probability-l5-q65",
    "medium",
    "Which statements correctly describe multinomial sampling from a categorical distribution?",
    [
      [
        "Each draw returns one category according to its probability mass.",
        true,
      ],
      ["Counts across categories sum to the number of draws.", true],
      ["Expected count for category i is \\(\\mathbb{E}[N_i]=np_i\\).", true],
      [
        "The joint count probability includes the arrangement factor \\(n!/\\prod_i N_i!\\).",
        true,
      ],
    ],
    "Repeated categorical draws create random category counts whose means follow directly from linearity of expectation. A given count vector can arise through many ordered sequences, and the multinomial coefficient counts those arrangements.",
  ),
  makeQuestion(
    "crash-probability-l5-q66",
    "hard",
    "A vocabulary has probabilities \\((0.5,0.3,0.2)\\). In four independent draws, what statements are correct for counts \\((2,1,1)\\)?",
    [
      [
        "The number of ordered sequences with these counts is \\(4!/(2!1!1!)=12\\).",
        true,
      ],
      ["Each such sequence has probability \\(0.5^2(0.3)(0.2)\\).", true],
      ["The count-vector probability is \\(12(0.5^2)(0.3)(0.2)=0.18\\).", true],
      [
        "The count-vector probability is \\(P(N_1=2,N_2=1,N_3=1)=0.5\\) because the first category appears most often.",
        false,
      ],
    ],
    "The multinomial coefficient counts every ordering with two first-category draws and one of each other category. Multiplying that count by one sequence probability gives 0.18; the modal count does not itself determine the event probability.",
  ),
  makeQuestion(
    "crash-probability-l5-q67",
    "easy",
    "Why can two generations from the same prompt differ when token sampling is enabled?",
    [
      [
        "Different random draws can realize different positive-probability tokens.",
        true,
      ],
      ["The prompt changes automatically after the first generation.", false],
      ["Sampling forces model weights to update between outputs.", false],
      [
        "Token probabilities become uniform when the sampler is activated.",
        false,
      ],
    ],
    "A fixed model and prompt can define the same conditional distributions while the random sampler realizes different branches. No weight update or uniformization is needed; later distributions can then diverge further because earlier sampled tokens become part of context.",
  ),
  makeQuestion(
    "crash-probability-l5-q68",
    "medium",
    "A generated two-token sequence has \\(P(x_1=A)=0.6\\) and \\(P(x_2=B\\mid x_1=A)=0.4\\). Which statements are correct?",
    [
      ["The sequence probability \\(P(A,B)=0.6(0.4)=0.24\\).", true],
      [
        "Sampling A first changes the relevant next-token distribution to the one conditioned on A.",
        true,
      ],
      [
        "The sequence probability becomes \\(P(A,B)=1\\) because both tokens were eventually observed.",
        false,
      ],
      ["The sequence probability is \\(0.6+0.4=1.0\\).", false],
    ],
    "Autoregressive generation multiplies the probability of each realized token under its current context. Observation does not turn model probabilities into certainty, and adding conditional factors does not give a joint sequence probability.",
  ),
  makeQuestion(
    "crash-probability-l5-q69",
    "hard",
    "A decoder samples without replacement from three items with initial weights 0.5, 0.3, and 0.2. The first item is drawn first. Which statements are correct for the second draw?",
    [
      [
        "The remaining weights must be renormalized over the second and third items.",
        true,
      ],
      [
        "Their second-draw probabilities become \\(0.3/0.5=0.6\\) and \\(0.2/0.5=0.4\\).",
        true,
      ],
      ["The removed item has second-draw probability zero.", true],
      [
        "The remaining probabilities stay \\((0.3,0.2)\\) even though their total is \\(0.5\\).",
        false,
      ],
    ],
    "Removing an item changes the allowed sample space, so the surviving mass must be divided by its total 0.5. Renormalization preserves the survivors' relative odds while making the new conditional distribution sum to one.",
  ),
  makeQuestion(
    "crash-probability-l5-q70",
    "easy",
    "Which statements about pseudo-random sampling in a model are correct?",
    [
      [
        "A fixed seed can make a sampling run reproducible in a fixed implementation.",
        true,
      ],
      [
        "Changing the seed can produce a different valid sequence from the same probabilities.",
        true,
      ],
      [
        "Reproducibility of draws does not make the modeled distribution deterministic.",
        true,
      ],
      [
        "A sampled low-probability output is possible without being the model's preferred output.",
        true,
      ],
    ],
    "A seed controls the pseudo-random number stream used to realize a stochastic rule, allowing experiments to be repeated. It does not remove probability mass or change which outcomes the distribution regards as more likely.",
  ),
  makeQuestion(
    "crash-probability-l5-q71",
    "medium",
    "A model samples a token with probability 0.02 at each of 100 independent comparable trials. Which statements are correct?",
    [
      ["The expected count is \\(100(0.02)=2\\).", true],
      [
        "The probability of seeing it at least once is \\(1-(0.98)^{100}\\).",
        true,
      ],
      ["It must occur exactly twice because the expected count is two.", false],
      [
        "Its 0.02 mass makes \\(P(N\\ge1)=0\\) when the sample contains 100 trials.",
        false,
      ],
    ],
    "Linearity gives expected count two, while the complement of no occurrences gives the at-least-once probability. Expectation is an average over repeated experiments and does not fix the realized count in one batch.",
  ),
  makeQuestion(
    "crash-probability-l5-q72",
    "hard",
    "A three-token autoregressive sequence has conditional probabilities 0.8, 0.5, and 0.25 along one realized path. Which statements are correct?",
    [
      ["The path probability is \\(0.8(0.5)(0.25)=0.10\\).", true],
      ["The path log-probability is \\(\\log0.8+\\log0.5+\\log0.25\\).", true],
      [
        "Changing the first sampled token can change the later conditional distributions and path probability.",
        true,
      ],
      [
        "The path probability is \\((0.8+0.5+0.25)/3\\), the arithmetic mean of its factors.",
        false,
      ],
    ],
    "The chain rule multiplies each token probability under the context created by earlier realized tokens, and logs convert that product to a sum. Averaging factors ignores the requirement that all three conditional events occur along the sequence.",
  ),

  // Temperature, top-k, top-p, and entropy
  makeQuestion(
    "crash-probability-l5-q73",
    "easy",
    "What does lowering a positive softmax temperature usually do to a nonuniform token distribution?",
    [
      ["Makes it more concentrated on high-logit tokens", true],
      [
        "Pushes each \\(p_i\\) toward \\(1/V\\) more strongly than raising the temperature",
        false,
      ],
      ["Changes the model weights before sampling", false],
      ["Adds new tokens to the vocabulary", false],
    ],
    "Dividing logits by a smaller temperature magnifies their gaps, making high-scoring tokens more dominant. Temperature reshapes a fixed output distribution at decoding time; it neither retrains parameters nor changes the vocabulary.",
  ),
  makeQuestion(
    "crash-probability-l5-q74",
    "easy",
    "Which comparisons are correct when temperature rises from 0.5 to 2 for fixed unequal logits?",
    [
      ["The distribution becomes flatter.", true],
      ["Lower-ranked tokens generally receive more probability mass.", true],
      [
        "Positive scaling gives \\(\\arg\\max_i z_i/T\\ne\\arg\\max_i z_i\\).",
        false,
      ],
      [
        "The model acquires new factual knowledge from the higher temperature.",
        false,
      ],
    ],
    "A larger positive temperature shrinks score gaps without changing their order, spreading probability toward alternatives. This alters sampling diversity and error risk, not the information encoded in the model weights.",
  ),
  makeQuestion(
    "crash-probability-l5-q75",
    "medium",
    "Two tokens have logits 2 and 0. Which statements correctly compare their odds under temperature T?",
    [
      ["Their odds ratio is \\(e^{2/T}\\).", true],
      ["At \\(T=1\\), the ratio is \\(e^2\\).", true],
      [
        "At \\(T=2\\), the ratio falls to e, so the distribution is flatter.",
        true,
      ],
      [
        "At \\(T=0.5\\), the ratio is \\(e\\), so the distribution is flatter than at T=2.",
        false,
      ],
    ],
    "Temperature divides the logit gap before exponentiation, making the odds \\(e^{2/T}\\). Small T enlarges the effective gap—at 0.5 the ratio is \\(e^4\\)—while large T reduces concentration. The positive scaling preserves which token has the higher logit.",
  ),
  makeQuestion(
    "crash-probability-l5-q76",
    "hard",
    "A distribution over four tokens is \\((0.50,0.25,0.15,0.10)\\). Top-k decoding uses \\(k=2\\). Which statements are correct?",
    [
      ["The retained mass is \\(0.75\\).", true],
      ["The renormalized distribution is \\((2/3,1/3,0,0)\\).", true],
      [
        "The retained sampling vector remains \\((0.50,0.25)\\) with total mass \\(0.75\\).",
        false,
      ],
      [
        "Top-k retains the smallest two probabilities to increase surprise.",
        false,
      ],
    ],
    "Top-k removes all but the two highest-mass candidates and divides their probabilities by the surviving total. This preserves their 2:1 relative odds while forming a valid distribution over the restricted set.",
  ),
  makeQuestion(
    "crash-probability-l5-q77",
    "medium",
    "Which statements correctly describe top-p (nucleus) sampling?",
    [
      ["Tokens are ordered from higher to lower probability.", true],
      [
        "The smallest prefix whose cumulative mass reaches the threshold is retained.",
        true,
      ],
      ["Retained probabilities are renormalized before sampling.", true],
      [
        "The candidate-set size can vary with how concentrated the original distribution is.",
        true,
      ],
    ],
    "Top-p uses a probability-mass threshold rather than a fixed number of tokens, so sharp distributions may keep few candidates and flat ones more. Renormalization then turns the retained prefix into the actual sampling distribution.",
  ),
  makeQuestion(
    "crash-probability-l5-q78",
    "hard",
    "Sorted token probabilities are \\((0.40,0.30,0.15,0.10,0.05)\\), and top-p uses \\(p=0.80\\). Which statements are correct?",
    [
      [
        "The first three tokens are retained because cumulative masses are \\(0.40,0.70,0.85\\).",
        true,
      ],
      [
        "Their renormalized probabilities are \\((0.40/0.85,0.30/0.85,0.15/0.85)\\).",
        true,
      ],
      [
        "The fourth and fifth tokens receive zero probability after filtering.",
        true,
      ],
      [
        "The first two tokens are retained because \\(0.40+0.30=0.70\\) is closest to \\(p=0.80\\).",
        false,
      ],
    ],
    "Nucleus sampling keeps adding tokens until the cumulative mass first reaches or exceeds the threshold, so stopping at 0.70 would fall short. The retained 0.85 mass is divided out to form a normalized three-token distribution.",
  ),
  makeQuestion(
    "crash-probability-l5-q79",
    "easy",
    "Which decoding method is deterministic for a fixed distribution when ties are resolved deterministically?",
    [
      ["Greedy argmax decoding", true],
      ["Categorical sampling", false],
      ["Top-p sampling after renormalization", false],
      ["Temperature sampling at any positive temperature", false],
    ],
    "Greedy decoding chooses the largest-probability token rather than drawing from the distribution, so a fixed tie rule makes it deterministic. Filtering or temperature can restrict or reshape probabilities, but the final sampling step remains random.",
  ),
  makeQuestion(
    "crash-probability-l5-q80",
    "medium",
    "A decoder filters probabilities from \\((0.6,0.25,0.10,0.05)\\) to the first three tokens. Which statements are correct?",
    [
      ["The retained mass is \\(0.6+0.25+0.10=0.95\\).", true],
      ["The third token's new probability is \\(0.10/0.95\\).", true],
      ["Renormalization changes the retained tokens' pairwise odds.", false],
      ["The removed token retains filtered probability \\(P=0.05\\).", false],
    ],
    "Filtering conditions the draw on the retained set, so each surviving probability is divided by their total 0.95 and removed mass becomes zero. A common divisor preserves pairwise odds among survivors even while increasing their absolute probabilities.",
  ),
  makeQuestion(
    "crash-probability-l5-q81",
    "hard",
    "A distribution changes from \\((0.7,0.2,0.1)\\) to \\((0.5,0.3,0.2)\\). Which statements are correct?",
    [
      ["The second distribution has higher entropy.", true],
      [
        "Sampling from the second gives lower-ranked outcomes more total chance.",
        true,
      ],
      ["The argmax outcome remains the first category.", true],
      [
        "The second distribution must have been produced by retraining rather than temperature or filtering.",
        false,
      ],
    ],
    "Spreading mass away from the leader increases uncertainty and diversity while preserving the same most likely category. Many decoding transformations can reshape a distribution, so the probability change alone does not identify a weight update.",
  ),
  makeQuestion(
    "crash-probability-l5-q82",
    "easy",
    "Which statements correctly describe decoding controls?",
    [
      ["Temperature changes relative sharpness through scaled logits.", true],
      ["Top-k limits the candidate set to a fixed count.", true],
      ["Top-p limits the candidate set by cumulative probability mass.", true],
      ["Renormalization is needed after probability mass is removed.", true],
    ],
    "The three controls act at different stages but all shape the distribution used for the final draw. Filtering removes mass and therefore requires renormalization, while temperature changes the mass allocation before any optional filtering.",
  ),
  makeQuestion(
    "crash-probability-l5-q83",
    "medium",
    "Which statements compare top-k and top-p correctly?",
    [
      [
        "Top-k keeps the same maximum number of candidates across contexts.",
        true,
      ],
      ["Top-p can adapt candidate count to distribution concentration.", true],
      ["Top-p keeps exactly \\(p|V|\\) tokens in each context.", false],
      ["Top-k chooses tokens uniformly after retaining them.", false],
    ],
    "Top-k uses a rank cutoff, while top-p uses a cumulative-mass cutoff and can therefore vary in set size. Both normally retain and renormalize the model's relative probabilities rather than sampling retained candidates uniformly.",
  ),
  makeQuestion(
    "crash-probability-l5-q84",
    "hard",
    "A two-token distribution has logit gap \\(d>0\\). Which statements are correct as temperature varies?",
    [
      [
        "As \\(T\\to0^+\\), probability concentrates on the higher logit.",
        true,
      ],
      [
        "As \\(T\\to\\infty\\), the two probabilities approach one half each.",
        true,
      ],
      [
        "The entropy approaches zero at the low-temperature limit and \\(\\log2\\) at the high-temperature limit.",
        true,
      ],
      [
        "At high positive T, \\(\\arg\\max_i z_i/T\\) becomes the lower-logit token.",
        false,
      ],
    ],
    "Temperature changes the scale but not the sign of a positive logit gap, so ordering remains fixed. Extreme scaling connects nearly greedy concentration with a nearly uniform high-entropy distribution.",
  ),

  // Latent variables, marginalization, and posterior inference
  makeQuestion(
    "crash-probability-l5-q85",
    "easy",
    "Which quantity is a plausible latent variable for a collection of observed documents?",
    [
      ["An unobserved topic that influences word choices", true],
      ["The exact visible words after they have been read", false],
      ["The number of documents fixed by the dataset", false],
      ["The filename extension stored with each document", false],
    ],
    "A latent variable is unobserved but helps explain patterns in observed data, such as a topic shaping vocabulary. Visible words and recorded metadata are observed variables, while a fixed dataset size is a constant rather than a hidden per-document cause.",
  ),
  makeQuestion(
    "crash-probability-l5-q86",
    "easy",
    "A generative latent-variable model first samples z and then x. Which probability components are involved?",
    [
      ["A prior \\(P(z)\\) over latent values", true],
      ["A conditional generator \\(P(x\\mid z)\\)", true],
      [
        "A posterior \\(P(z\\mid x)\\) that is sufficient for generation without \\(P(x\\mid z)\\)",
        false,
      ],
      ["A deterministic requirement mapping each z to a single x", false],
    ],
    "The generative story specifies how likely latent causes are and how each cause produces observations. Posterior inference reverses that story after x is seen, and the conditional generator can remain stochastic with many possible x values for one z.",
  ),
  makeQuestion(
    "crash-probability-l5-q87",
    "medium",
    "A latent variable z has values 0 and 1 with priors 0.6 and 0.4; \\(P(x\\mid0)=0.2\\) and \\(P(x\\mid1)=0.7\\). Which statements are correct?",
    [
      ["The z=0 path to x is \\(0.6(0.2)=0.12\\).", true],
      ["The z=1 path to x is \\(0.4(0.7)=0.28\\).", true],
      ["The marginal \\(P(x)=0.40\\).", true],
      [
        "The marginal is 0.70 because that is the largest conditional likelihood.",
        false,
      ],
    ],
    "Marginal probability adds the prior-weighted latent paths: \\(0.6(0.2)+0.4(0.7)=0.40\\). Selecting only the largest likelihood ignores both the alternative path and the latent priors. Both hidden states remain possible explanations of the observed x.",
  ),
  makeQuestion(
    "crash-probability-l5-q88",
    "hard",
    "Using priors \\(P(z=0)=0.6,P(z=1)=0.4\\) and likelihoods \\(P(x\\mid0)=0.2,P(x\\mid1)=0.7\\), which posterior calculations are correct?",
    [
      ["\\(P(z=1,x)=0.28\\).", true],
      ["\\(P(z=1\\mid x)=0.28/0.40=0.70\\).", true],
      ["\\(P(z=1\\mid x)=0.40\\) because the prior is unchanged by x.", false],
      ["\\(P(z=1\\mid x)=0.7/0.4=1.75\\).", false],
    ],
    "Bayes normalizes the z=1 path by the total evidence, raising the latent probability from prior 0.4 to posterior 0.7. Dividing likelihood by prior reverses the required relationship and can even produce an invalid probability above one.",
  ),
  makeQuestion(
    "crash-probability-l5-q89",
    "medium",
    "Which statements correctly describe latent-variable marginalization?",
    [
      ["For discrete z, \\(P(x)=\\sum_zP(x\\mid z)P(z)\\).", true],
      ["For continuous z, \\(p(x)=\\int p(x\\mid z)p(z)\\,dz\\).", true],
      [
        "Marginalization accounts for multiple hidden explanations of the same observation.",
        true,
      ],
      [
        "Posterior inference uses the same joint paths but normalizes them after x is observed.",
        true,
      ],
    ],
    "The generative joint factors into prior and conditional terms, and summing or integrating removes the hidden variable. Bayes reuses those path weights in the opposite direction to compare which latent explanations are plausible after observing x.",
  ),
  makeQuestion(
    "crash-probability-l5-q90",
    "hard",
    "A model has three equally likely latent styles, with likelihoods of an observed image x equal to 0.6, 0.3, and 0.1. Which statements are correct?",
    [
      ["The marginal likelihood is \\((0.6+0.3+0.1)/3=1/3\\).", true],
      [
        "Posterior style probabilities are \\((0.6,0.3,0.1)\\) after normalization.",
        true,
      ],
      ["The first style has six times the posterior odds of the third.", true],
      [
        "The first style has posterior probability one because it has the largest likelihood.",
        false,
      ],
    ],
    "Equal priors make posterior weights proportional to the likelihoods, which already sum to one in their displayed ratio. The best explanation is more probable but not certain because the other latent paths also assign positive probability to x.",
  ),
  makeQuestion(
    "crash-probability-l5-q91",
    "easy",
    "In a variational autoencoder (VAE), what is the encoder used to approximate?",
    [
      ["A latent posterior distribution given an observation", true],
      ["The fixed list of training filenames", false],
      ["A greedy next-token decoder", false],
      ["The environment transition kernel in reinforcement learning", false],
    ],
    "A VAE encoder maps an observed x to parameters of an approximate distribution over latent z values, supporting inference and sampling. The decoder then maps latent samples toward observations; the unrelated alternatives belong to other data or model structures.",
  ),
  makeQuestion(
    "crash-probability-l5-q92",
    "medium",
    "Which statements correctly distinguish VAE training-time inference from generative sampling?",
    [
      ["The encoder approximates \\(q(z\\mid x)\\) for observed x.", true],
      [
        "Generation can sample z from a prior and then sample or decode x from \\(P(x\\mid z)\\).",
        true,
      ],
      [
        "Generation starts by copying the latent code of an existing observation.",
        false,
      ],
      [
        "The latent prior is a categorical next-token distribution by definition.",
        false,
      ],
    ],
    "Inference asks which hidden codes explain a given observation, while generation starts with a sampled latent and uses the decoder to create a possible observation. A latent prior may be continuous, often Gaussian, and generation need not begin from an existing example.",
  ),
  makeQuestion(
    "crash-probability-l5-q93",
    "hard",
    "A continuous latent z has density \\(p(z)\\) and decoder likelihood \\(p(x\\mid z)\\). Which statements are correct?",
    [
      ["The marginal density is \\(p(x)=\\int p(x\\mid z)p(z)\\,dz\\).", true],
      [
        "The posterior is proportional to \\(p(x\\mid z)p(z)\\) as a function of z.",
        true,
      ],
      [
        "A point estimate of z can miss uncertainty across several plausible latent explanations.",
        true,
      ],
      [
        "The marginal is obtained by maximizing \\(p(x\\mid z)\\) and discarding the remaining latent mass.",
        false,
      ],
    ],
    "Continuous marginalization integrates contributions from the entire latent space, while posterior inference renormalizes those contributions after observing x. A maximum-latent approximation answers a different question and can underrepresent ambiguity.",
  ),
  makeQuestion(
    "crash-probability-l5-q94",
    "easy",
    "Which are plausible roles for latent variables?",
    [
      ["Representing document topic", true],
      ["Representing image pose or lighting", true],
      ["Representing a user's hidden preference", true],
      ["Representing an unobserved environment state", true],
    ],
    "Each variable names hidden structure that can influence observed words, pixels, clicks, or transitions. Latent variables are modeling constructs rather than necessarily single human-interpretable truths, but these are all useful candidate interpretations.",
  ),
  makeQuestion(
    "crash-probability-l5-q95",
    "medium",
    "An observation has equal posterior probability under two very different latent codes. Which statements are correct?",
    [
      [
        "The observation leaves latent ambiguity rather than identifying one code with certainty.",
        true,
      ],
      [
        "Sampling from the posterior can represent both explanations across repeated draws.",
        true,
      ],
      [
        "The two codes have identical decoder distributions because their posteriors match.",
        false,
      ],
      [
        "Marginalization should delete one code to avoid double-counting uncertainty.",
        false,
      ],
    ],
    "A posterior distribution represents uncertainty over hidden explanations compatible with the same evidence. Equal posterior mass does not make the latent states behaviorally identical, and marginalization should retain and weight both paths rather than choose arbitrarily.",
  ),
  makeQuestion(
    "crash-probability-l5-q96",
    "hard",
    "A latent mixture has path masses for observation x equal to 0.05, 0.15, and 0.30. Which statements are correct?",
    [
      ["The marginal \\(P(x)=0.50\\).", true],
      [
        "The posterior probabilities of the three latent states are \\((0.10,0.30,0.60)\\).",
        true,
      ],
      ["The third state is most plausible after x but not certain.", true],
      [
        "The posterior masses remain 0.05, 0.15, and 0.30 because joint paths already sum to one.",
        false,
      ],
    ],
    "The path masses sum to the evidence 0.50 and must be divided by that total to become a posterior distribution. The third state receives most posterior mass, while the other positive paths preserve uncertainty.",
  ),

  // Gaussian noise and the forward diffusion process
  makeQuestion(
    "crash-probability-l5-q97",
    "easy",
    "In \\(X\\sim\\mathcal{N}(\\mu,\\sigma^2)\\), what does \\(\\sigma^2\\) denote?",
    [
      ["Variance", true],
      ["Mean", false],
      ["Standard deviation", false],
      ["Probability that X equals zero", false],
    ],
    "The second Gaussian parameter is variance, while its positive square root \\(\\sigma\\) is standard deviation and \\(\\mu\\) is the mean. A continuous normal variable assigns zero probability to any one exact point.",
  ),
  makeQuestion(
    "crash-probability-l5-q98",
    "easy",
    "Which effects follow from increasing the variance of zero-mean Gaussian noise?",
    [
      ["Samples spread farther from zero on average.", true],
      [
        "The standard deviation increases as the square root of variance.",
        true,
      ],
      ["Increasing variance makes a sampled noise value positive.", false],
      ["The Gaussian mean necessarily shifts away from zero.", false],
    ],
    "Variance controls spread, and standard deviation is its square root, while the distribution remains centered at the same mean. A zero-mean Gaussian remains symmetric with positive and negative samples even as its variance grows.",
  ),
  makeQuestion(
    "crash-probability-l5-q99",
    "medium",
    "Let \\(X=\\mu+\\sigma Z\\) with \\(Z\\sim\\mathcal{N}(0,1)\\) and \\(\\sigma>0\\). Which statements are correct?",
    [
      ["\\(\\mathbb{E}[X]=\\mu\\).", true],
      ["\\(\\operatorname{Var}(X)=\\sigma^2\\).", true],
      [
        "Sampling X can be implemented by sampling standard noise then scaling and shifting it.",
        true,
      ],
      [
        "\\(\\operatorname{Var}(X)=\\sigma\\) because scaling affects variance linearly.",
        false,
      ],
    ],
    "Shifting changes the mean and scaling by sigma multiplies variance by \\(\\sigma^2\\), producing the stated Gaussian. This reparameterized view is useful because one standard-noise sampler can generate many Gaussian distributions.",
  ),
  makeQuestion(
    "crash-probability-l5-q100",
    "hard",
    "A forward noising step is \\(x_t=\\sqrt{\\bar\\alpha_t}x_0+\\sqrt{1-\\bar\\alpha_t}\\epsilon\\), with \\(\\epsilon\\sim\\mathcal{N}(0,I)\\). Which statements are correct?",
    [
      [
        "Conditioned on \\(x_0\\), the mean of \\(x_t\\) is \\(\\sqrt{\\bar\\alpha_t}x_0\\).",
        true,
      ],
      [
        "Conditioned on \\(x_0\\), the noise covariance is \\((1-\\bar\\alpha_t)I\\).",
        true,
      ],
      ["The formula is deterministic because \\(x_0\\) is known.", false],
      [
        "The noise coefficient should be \\(1-\\bar\\alpha_t\\) rather than its square root to obtain that variance.",
        false,
      ],
    ],
    "The standard Gaussian has covariance I, and multiplying it by the square root coefficient produces variance \\(1-\\bar\\alpha_t\\). Random epsilon keeps the noised sample stochastic even when the clean input and schedule are fixed.",
  ),
  makeQuestion(
    "crash-probability-l5-q101",
    "medium",
    "Which statements correctly describe the forward diffusion process?",
    [
      ["It gradually corrupts real data with noise.", true],
      ["Its schedule controls how much signal remains at each time.", true],
      [
        "For sufficiently late time, the distribution is designed to approach simple Gaussian noise.",
        true,
      ],
      [
        "It supplies paired clean and noisy examples for learning a reverse prediction.",
        true,
      ],
    ],
    "Forward diffusion is a fixed corruption mechanism rather than the generative direction, and its known noise construction creates supervised denoising targets. The schedule trades clean-signal strength against accumulated noise until the terminal state is easy to sample.",
  ),
  makeQuestion(
    "crash-probability-l5-q102",
    "hard",
    "In the noising formula, \\(\\bar\\alpha_t=0.64\\), \\(x_0=2\\), and a realized scalar noise value is \\(\\epsilon=-1\\). Which statements are correct?",
    [
      ["The signal coefficient is \\(\\sqrt{0.64}=0.8\\).", true],
      ["The noise coefficient is \\(\\sqrt{0.36}=0.6\\).", true],
      ["The realized \\(x_t=0.8(2)+0.6(-1)=1.0\\).", true],
      [
        "The realized \\(x_t\\) equals the conditional mean 1.6 because Gaussian noise has mean zero.",
        false,
      ],
    ],
    "The conditional mean averages over possible epsilon values and equals 1.6, but one realized sample includes its actual noise contribution. Substituting the given coefficients and epsilon yields 1.0, illustrating the distinction between mean and draw.",
  ),
  makeQuestion(
    "crash-probability-l5-q103",
    "easy",
    "What is the role of the forward process in standard diffusion training?",
    [
      ["Create controlled noisy versions of real data", true],
      ["Generate final samples by removing noise", false],
      ["Choose the highest-probability text token", false],
      ["Infer a reward-maximizing action policy", false],
    ],
    "The forward process is the known corruption path used to create inputs and targets for denoising training. Final generation follows a learned reverse direction, while token decoding and reinforcement-learning policies solve different probabilistic tasks.",
  ),
  makeQuestion(
    "crash-probability-l5-q104",
    "medium",
    "Which statements correctly interpret \\(x_0,x_t,x_T\\) in diffusion notation?",
    [
      ["\\(x_0\\) denotes clean data at the start of forward noising.", true],
      [
        "\\(x_T\\) is designed to be close to the simple terminal noise distribution.",
        true,
      ],
      [
        "\\(x_t\\) must be either perfectly clean or pure noise with no mixture.",
        false,
      ],
      ["Time t records model-training epochs rather than noise level.", false],
    ],
    "Intermediate states combine attenuated signal and noise according to the schedule, connecting clean data to an approximately Gaussian terminal state. The diffusion time index labels corruption level, not the outer optimization epoch count.",
  ),
  makeQuestion(
    "crash-probability-l5-q105",
    "hard",
    "The forward process is Markov with \\(q(x_t\\mid x_{t-1})\\). Which statements are correct?",
    [
      [
        "Its joint path factorizes as \\(q(x_{1:T}\\mid x_0)=\\prod_{t=1}^Tq(x_t\\mid x_{t-1})\\).",
        true,
      ],
      [
        "Given \\(x_{t-1}\\), the next noisy state does not require the full earlier path.",
        true,
      ],
      [
        "A closed-form \\(q(x_t\\mid x_0)\\) can permit direct sampling of a training noise level.",
        true,
      ],
      [
        "Markov factorization implies \\(q(x_t\\mid x_0)=q(x_t)\\), so \\(x_t\\) is marginally independent of \\(x_0\\).",
        false,
      ],
    ],
    "The Markov property makes the next corruption depend on the current noisy state, giving a product of local transitions. Marginal dependence on the clean origin remains, and Gaussian composition can provide a direct conditional from \\(x_0\\) to an arbitrary time.",
  ),
  makeQuestion(
    "crash-probability-l5-q106",
    "easy",
    "Which properties make Gaussian noise convenient in generative modeling?",
    [
      ["It is easy to sample.", true],
      ["Linear transformations have tractable means and covariances.", true],
      ["Independent Gaussian increments combine into Gaussian noise.", true],
      ["A standard Gaussian provides a simple terminal distribution.", true],
    ],
    "Gaussian distributions support efficient sampling and algebraically tractable transformations, making a controlled noising schedule practical. Their closure properties also help summarize many incremental corruption steps with a simple direct formula.",
  ),
  makeQuestion(
    "crash-probability-l5-q107",
    "medium",
    "A clean scalar has value \\(x_0=0\\), and \\(x_t=\\sqrt{1-\\bar\\alpha_t}\\epsilon\\). Which statements are correct?",
    [
      ["The conditional mean of \\(x_t\\) is zero.", true],
      ["Its conditional variance is \\(1-\\bar\\alpha_t\\).", true],
      [
        "A zero conditional mean makes the realized \\(x_t\\) equal zero.",
        false,
      ],
      ["Its variance is \\(\\sqrt{1-\\bar\\alpha_t}\\).", false],
    ],
    "Zero clean signal leaves a scaled standard-Gaussian random variable, whose mean is zero and variance is the square of its scale. A distribution's mean describes an average and does not force individual noise samples to equal it.",
  ),
  makeQuestion(
    "crash-probability-l5-q108",
    "hard",
    "Two noising times use \\(\\bar\\alpha_a=0.9\\) and \\(\\bar\\alpha_b=0.1\\) for the same clean data. Which statements are correct?",
    [
      ["Time a retains a larger clean-signal coefficient.", true],
      ["Time b uses a larger noise variance.", true],
      [
        "Predicting the original data is generally more uncertain at time b.",
        true,
      ],
      [
        "Both times have identical conditional distributions because their coefficients still sum in quadrature to one.",
        false,
      ],
    ],
    "Although signal and noise variances are balanced to a common total scale, their allocations differ strongly. The later, low-alpha state hides more clean structure under noise and therefore poses a harder denoising inference problem.",
  ),

  // Reverse diffusion, guidance, and generative synthesis
  makeQuestion(
    "crash-probability-l5-q109",
    "easy",
    "What is learned in the reverse direction of a diffusion model?",
    [
      [
        "How to move from a noisy state toward a less noisy state or an equivalent denoising target",
        true,
      ],
      [
        "How to add the fixed forward noise schedule to clean training data",
        false,
      ],
      ["How to reduce generation to one greedy class argmax", false],
      ["How to assign a fixed predetermined image to each prompt", false],
    ],
    "The forward corruption is known, while a neural network learns a reverse conditional or related target such as noise or clean data prediction. The learned process remains generative and can map different initial noise samples to different valid outputs.",
  ),
  makeQuestion(
    "crash-probability-l5-q110",
    "easy",
    "Why can the same text prompt produce different diffusion images?",
    [
      ["Generation can begin from different random noise samples.", true],
      ["The prompt leaves many visual details unspecified.", true],
      ["The model retrains on the prompt before producing an image.", false],
      [
        "Conditional generation removes its random initial state by definition.",
        false,
      ],
    ],
    "A prompt constrains but rarely uniquely determines an image, and random initial states select different trajectories through that conditional distribution. Conditioning guides generation without collapsing all probability mass onto one fixed output.",
  ),
  makeQuestion(
    "crash-probability-l5-q111",
    "medium",
    "A diffusion model predicts the noise \\(\\epsilon\\) used to create \\(x_t\\). Which statements are correct?",
    [
      [
        "Training can compare predicted noise with the known sampled noise.",
        true,
      ],
      [
        "The network receives the noisy state and time index because denoising depends on noise level.",
        true,
      ],
      ["A text condition can be supplied for conditional generation.", true],
      [
        "The network can ignore x_t because target noise is shared across the training batch.",
        false,
      ],
    ],
    "Forward noising records the sampled target, turning reverse learning into a supervised prediction problem at varied times. The target differs across examples and draws, so the noisy input, time, and optional condition carry essential information.",
  ),
  makeQuestion(
    "crash-probability-l5-q112",
    "hard",
    "A learned reverse Markov model uses transitions \\(p_\\theta(x_{t-1}\\mid x_t,c)\\). Which statements are correct?",
    [
      [
        "Its conditional path factorizes as \\(p(x_T)\\prod_{t=1}^Tp_\\theta(x_{t-1}\\mid x_t,c)\\).",
        true,
      ],
      [
        "Generation begins by sampling \\(x_T\\) from a simple noise distribution.",
        true,
      ],
      [
        "Generation follows \\(q(x_{1:T}\\mid x_0)=\\prod_{t=1}^Tq(x_t\\mid x_{t-1})\\) after sampling \\(x_0\\) from the dataset.",
        false,
      ],
      [
        "The condition c is marginalized out even when prompt-guided output is requested.",
        false,
      ],
    ],
    "Reverse generation starts from terminal noise and follows learned local conditionals toward data, optionally conditioned on information such as text. Sampling clean training data and running forward would be corruption rather than novel generation.",
  ),
  makeQuestion(
    "crash-probability-l5-q113",
    "medium",
    "Which statements correctly connect diffusion to conditional probability?",
    [
      ["A denoising step can depend on the current noisy state.", true],
      ["The time index conditions the model on the current noise level.", true],
      ["A text prompt can condition which clean outputs are plausible.", true],
      [
        "Repeated reverse steps compose local conditional predictions into a generated sample.",
        true,
      ],
    ],
    "Diffusion does not remove noise using one unconditional rule; it predicts an update from the noisy state, time, and optional guidance. Chaining those conditionals gradually turns a simple random initial state into a structured conditional output.",
  ),
  makeQuestion(
    "crash-probability-l5-q114",
    "hard",
    "Classifier-free guidance combines an unconditional prediction \\(u\\) and conditional prediction \\(c\\) as \\(g=u+w(c-u)\\). Which statements are correct?",
    [
      ["At \\(w=0\\), \\(g=u\\).", true],
      ["At \\(w=1\\), \\(g=c\\).", true],
      [
        "For \\(w>1\\), the prediction extrapolates beyond c in the conditional direction.",
        true,
      ],
      [
        "Increasing w obeys \\(d(\\text{naturalness})/dw>0\\) and \\(d(\\text{prompt fit})/dw>0\\) as a monotonic rule.",
        false,
      ],
    ],
    "The formula interpolates from unconditional to conditional prediction and extrapolates when guidance exceeds one. Strong guidance can increase condition adherence but may reduce diversity or introduce distortions, so its effect is a controllable tradeoff rather than a guarantee.",
  ),
  makeQuestion(
    "crash-probability-l5-q115",
    "easy",
    "Which description best contrasts LLM and diffusion generation?",
    [
      [
        "LLMs usually sample discrete next tokens, while diffusion iteratively denoises continuous representations.",
        true,
      ],
      [
        "LLMs use unconditional token probabilities, while diffusion uses unconditional denoising probabilities.",
        false,
      ],
      [
        "Diffusion generates in one step, while LLMs retrain after appending a token.",
        false,
      ],
      ["Both methods require greedy argmax at each generation step.", false],
    ],
    "Both systems repeatedly use learned conditional distributions, but the state and update differ: an LLM appends a discrete token, while diffusion refines a noisy continuous sample. Neither process requires retraining or greedy selection during ordinary inference.",
  ),
  makeQuestion(
    "crash-probability-l5-q116",
    "medium",
    "Which statements correctly compare LLM, RL-policy, and diffusion sampling?",
    [
      ["An LLM samples a token conditioned on context.", true],
      [
        "An RL transition kernel samples an action from \\(P(A_t\\mid S_t)\\).",
        false,
      ],
      [
        "A diffusion update uses a noisy state and optional prompt to parameterize a cleaner next state.",
        true,
      ],
      [
        "Each system requires its output to be a discrete vocabulary item.",
        false,
      ],
    ],
    "An LLM draws a discrete token, while a diffusion model uses a noisy continuous state to parameterize the next denoising update. In reinforcement learning the policy samples the action; the transition kernel instead samples the environment's next state after that action.",
  ),
  makeQuestion(
    "crash-probability-l5-q117",
    "hard",
    "An autoregressive model assigns a sequence conditional probabilities \\((0.9,0.8,0.2,0.5)\\). Which statements are correct?",
    [
      ["The sequence likelihood is \\(0.9(0.8)(0.2)(0.5)=0.072\\).", true],
      [
        "Its log-likelihood is \\(\\log0.9+\\log0.8+\\log0.2+\\log0.5\\).",
        true,
      ],
      ["The 0.2 token contributes the largest negative log penalty.", true],
      [
        "A high \\(p(x_1)\\) cancels \\(\\prod_{t=2}^Tp(x_t\\mid x_{<t})\\) in sequence probability.",
        false,
      ],
    ],
    "Every observed step contributes multiplicatively, so one low conditional probability can substantially reduce the complete path likelihood. Log space makes that effect additive and shows the 0.2 factor as the largest surprise.",
  ),
  makeQuestion(
    "crash-probability-l5-q118",
    "easy",
    "Which statements summarize probabilistic generation?",
    [
      ["A learned distribution represents multiple possible outputs.", true],
      ["Sampling turns that distribution into a concrete realization.", true],
      [
        "Conditioning restricts generation toward information such as context or a prompt.",
        true,
      ],
      [
        "Repeated random generation can produce diversity while following the same model.",
        true,
      ],
    ],
    "Generative models preserve uncertainty over valid outputs until a sampling or decoding process realizes one. Context narrows the distribution, while residual probability and random initial choices allow repeated runs to differ.",
  ),
  makeQuestion(
    "crash-probability-l5-q119",
    "medium",
    "A diffusion sampler is made deterministic given its initial noise, while initial noise is still sampled randomly. Which statements are correct?",
    [
      [
        "Different initial noise samples can still produce different outputs.",
        true,
      ],
      [
        "Reusing the same initial noise and condition can make the trajectory reproducible.",
        true,
      ],
      [
        "The overall generator has no randomness because each transition is deterministic.",
        false,
      ],
      [
        "A deterministic path implies a single valid image for a prompt.",
        false,
      ],
    ],
    "Randomness can enter at the initial condition even when later updates are deterministic, so the complete generator remains a pushforward of a noise distribution. Fixing that noise can reproduce one path without collapsing the model's broader output space.",
  ),
  makeQuestion(
    "crash-probability-l5-q120",
    "hard",
    "A conditional generator has latent prior \\(P(z)\\), decoder \\(P(x\\mid z,c)\\), and condition c. Which statements are correct?",
    [
      [
        "Its conditional output distribution marginalizes latents: \\(P(x\\mid c)=\\sum_zP(x\\mid z,c)P(z\\mid c)\\) in the general discrete case.",
        true,
      ],
      ["Sampling z can select one hidden route to a concrete output.", true],
      [
        "Posterior inference \\(P(z\\mid x,c)\\) asks which latent routes explain an observed conditioned output.",
        true,
      ],
      [
        "Conditioning gives \\(P(x\\mid z,c)=1\\) for the same x at each latent value z.",
        false,
      ],
    ],
    "Conditional generation can preserve many hidden explanations and outcomes, combining their paths through marginalization and realizing one through sampling. Observing x reverses the question toward posterior latent inference, while the condition guides rather than uniquely determines every output.",
  ),
];
