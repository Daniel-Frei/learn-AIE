import { Question } from "../../../quiz";

export const CrashCourseProbabilityL5Questions: Question[] = [
  {
    id: "crash-probability-l5-q01",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statement best captures the central probabilistic idea behind modern generative AI?",
    options: [
      {
        text: "A generative model learns a probability structure and then samples concrete outputs from it.",
        isCorrect: true,
      },
      {
        text: "A generative model stores one canonical answer for each possible prompt and retrieves it deterministically.",
        isCorrect: false,
      },
      {
        text: "A generative model avoids uncertainty by converting every distribution into a fixed label before generation begins.",
        isCorrect: false,
      },
      {
        text: "A generative model is probabilistic only during training and becomes nonprobabilistic when producing text or images.",
        isCorrect: false,
      },
    ],
    explanation:
      "The core idea is that generation turns learned probability distributions into concrete outputs by sampling or by a related decoding rule. The incorrect options confuse generation with lookup, deterministic classification, or probability that disappears at inference time.",
  },
  {
    id: "crash-probability-l5-q02",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Let \\(X\\) be one sample from \\(P(A)=0.70\\), \\(P(B)=0.20\\), and \\(P(C)=0.10\\). If \\(N_C\\) counts how many times \\(C\\) appears in 1,000 independent samples, which statements are correct?",
    options: [
      {
        text: "Greedy choice selects \\(A\\), but that rule is not the same random variable as \\(X\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\mathbb{E}[N_C]=100\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\operatorname{Var}(N_C)=1000\\cdot0.10\\cdot0.90=90\\).",
        isCorrect: true,
      },
      {
        text: "The probability of seeing at least one \\(B\\) in 1,000 independent samples is \\(1-0.8^{1000}\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "A repeated sampling count for one outcome follows a binomial distribution with parameters \\(n=1000\\) and \\(p=0.10\\), giving mean \\(np\\) and variance \\(np(1-p)\\). Greedy decoding is a deterministic maximum rule, while sampling is random and supports event calculations such as the complement probability for seeing at least one \\(B\\).",
  },
  {
    id: "crash-probability-l5-q03",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "A language model assigns probabilities 0.50 to `mat`, 0.25 to `sofa`, 0.15 to `floor`, 0.07 to `chair`, and 0.03 to `car` after the words `The animal sat on the`. Which statements correctly distinguish sampling from choosing the maximum?",
    options: [
      {
        text: "Choosing the maximum always selects `mat` for this one step.",
        isCorrect: true,
      },
      {
        text: "Sampling can produce `sofa` or `floor`, even though their probabilities are lower than `mat`.",
        isCorrect: true,
      },
      {
        text: "Sampling means choosing the most likely token and then adding random punctuation afterward.",
        isCorrect: false,
      },
      {
        text: "Choosing the maximum should produce `car` about 3 percent of the time because `car` has probability 0.03.",
        isCorrect: false,
      },
    ],
    explanation:
      "Choosing the maximum is deterministic for a fixed distribution: it returns the highest-probability token. Sampling draws according to the full distribution, so lower-probability tokens can appear with their assigned probabilities.",
  },
  {
    id: "crash-probability-l5-q04",
    chapter: 5,
    difficulty: "easy",
    type: "assertion-reason",
    prompt:
      "Assertion: Sampling turns a probability distribution into an actual generated outcome.\n\nReason: A sample is a random draw whose long-run frequencies are governed by the distribution's probabilities.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: false },
      { text: "Assertion is false, Reason is true.", isCorrect: false },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: true,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "The assertion is true because a distribution only describes possible outcomes until a decoding or sampling rule produces a concrete output. The reason is true and explains the assertion: sampling is the random draw process that links probabilities to realized outcomes over time.",
  },
  {
    id: "crash-probability-l5-q05",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "At a next-token step, logits \\(z_i\\) are converted to probabilities by softmax. Which operation is greedy decoding?",
    options: [
      {
        text: "Choose \\(\\arg\\max_i z_i\\), equivalently \\(\\arg\\max_i P(y_i\\mid\\text{context})\\) after softmax.",
        isCorrect: true,
      },
      {
        text: "Draw \\(y_i\\) from the categorical distribution \\(P(y_i\\mid\\text{context})\\).",
        isCorrect: false,
      },
      {
        text: "Lower the temperature until the entropy is small, then sample from the remaining distribution.",
        isCorrect: false,
      },
      {
        text: "Keep the top-k tokens, renormalize their probabilities, and sample from that restricted set.",
        isCorrect: false,
      },
    ],
    explanation:
      "Softmax is monotone in each logit, so the largest logit also has the largest softmax probability. Greedy decoding selects that maximum directly; categorical sampling, temperature sampling, and top-k sampling are different decoding strategies.",
  },
  {
    id: "crash-probability-l5-q06",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements describe why probabilistic sampling is often used for open-ended generation?",
    options: [
      {
        text: "It can produce multiple plausible continuations from the same prompt.",
        isCorrect: true,
      },
      {
        text: "It can make creative or diverse outputs more likely than a purely greedy rule.",
        isCorrect: true,
      },
      {
        text: "It can also choose low-probability tokens that hurt coherence if randomness is too high.",
        isCorrect: true,
      },
      {
        text: "It guarantees that every sampled output is more factual than the greedy output.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sampling is useful when many outputs could be reasonable, such as story openings, brainstormed ideas, or image variations. Its tradeoff is that diversity comes with less predictability and a greater chance of poor low-probability choices, so it does not guarantee factuality.",
  },
  {
    id: "crash-probability-l5-q07",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "A model assigns probabilities 0.60 to `Paris`, 0.15 to `Lyon`, 0.10 to `London`, 0.08 to `the`, and 0.07 to `banana`. Which statements correctly apply top-k and top-p sampling?",
    options: [
      {
        text: "With top-k sampling and \\(k=3\\), the candidate set is `Paris`, `Lyon`, and `London` before renormalization.",
        isCorrect: true,
      },
      {
        text: "With top-p sampling and \\(p=0.90\\), the smallest prefix exceeding 0.90 includes `Paris`, `Lyon`, `London`, and `the`.",
        isCorrect: true,
      },
      {
        text: "With top-k sampling and \\(k=3\\), `banana` remains eligible because it has nonzero probability.",
        isCorrect: false,
      },
      {
        text: "With top-p sampling and \\(p=0.90\\), only `Paris` is eligible because it is the single most likely token.",
        isCorrect: false,
      },
    ],
    explanation:
      "Top-k keeps the k highest-probability tokens, so k=3 keeps the first three tokens listed by probability. Nucleus or top-p sampling keeps the smallest high-probability set whose cumulative mass exceeds p, so 0.60+0.15+0.10=0.85 is not enough, while adding 0.08 gives 0.93.",
  },
  {
    id: "crash-probability-l5-q08",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Why is greedy decoding not automatically the best strategy for every generation task?",
    options: [
      {
        text: "Because locally selecting the most likely next token can still produce a dull, repetitive, or lower-quality full sequence.",
        isCorrect: true,
      },
      {
        text: "Because greedy decoding changes the model's training loss after every generated token.",
        isCorrect: false,
      },
      {
        text: "Because greedy decoding samples too often from extremely low-probability tokens.",
        isCorrect: false,
      },
      {
        text: "Because greedy decoding is the same as using the highest possible temperature.",
        isCorrect: false,
      },
    ],
    explanation:
      "Generation is sequential, so the best local token at one step need not lead to the best whole output. The other options mix up decoding with training updates, random sampling, and temperature scaling.",
  },
  {
    id: "crash-probability-l5-q09",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "For two token logits with \\(z_a-z_b=2\\), temperature-modified softmax gives odds \\(P(a)/P(b)=e^{(z_a-z_b)/T}\\). Which statements are correct?",
    options: [
      {
        text: "At \\(T=1\\), the odds ratio is \\(e^2\\).",
        isCorrect: true,
      },
      {
        text: "At \\(T=2\\), the odds ratio is \\(e\\), so the distribution is less sharp than at \\(T=1\\).",
        isCorrect: true,
      },
      {
        text: "For fixed unequal logits, lowering \\(T\\) below 1 increases the odds advantage of the larger logit.",
        isCorrect: true,
      },
      {
        text: "At \\(T=2\\), the lower-logit token becomes more likely than the higher-logit token.",
        isCorrect: false,
      },
    ],
    explanation:
      "Temperature rescales logit differences before softmax, so it changes odds without changing which logit is larger for positive \\(T\\). Higher temperature compresses odds toward 1, while lower temperature expands odds away from 1.",
  },
  {
    id: "crash-probability-l5-q10",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "For temperature-modified softmax, \\(P(y_i)=\\frac{e^{z_i/T}}{\\sum_j e^{z_j/T}}\\). Which statements are correct?",
    options: [
      {
        text: "When \\(T<1\\), differences between logits have a stronger effect on the probabilities.",
        isCorrect: true,
      },
      {
        text: "When \\(T>1\\), probabilities tend to move closer together than they were at lower temperature.",
        isCorrect: true,
      },
      {
        text: "Changing \\(T\\) changes which logits the neural network computed for the prompt.",
        isCorrect: false,
      },
      {
        text: "Setting \\(T\\) changes the target labels used during model training.",
        isCorrect: false,
      },
    ],
    explanation:
      "Dividing logits by a smaller positive temperature amplifies their relative differences, while a larger temperature compresses those differences. Temperature is a decoding-time control over probabilities derived from logits, not a change to the model's computed logits or training labels.",
  },
  {
    id: "crash-probability-l5-q11",
    chapter: 5,
    difficulty: "medium",
    type: "assertion-reason",
    prompt:
      "Assertion: Raising temperature does not make a model more knowledgeable.\n\nReason: Temperature changes how randomly we sample from the model's existing probability distribution.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: false },
      { text: "Assertion is false, Reason is true.", isCorrect: false },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: true,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "The assertion is true: temperature can make outputs more or less varied, but it does not add facts, reasoning ability, or verification. The reason is also true and explains the assertion because temperature operates on the sampling distribution produced by the model.",
  },
  {
    id: "crash-probability-l5-q12",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements about choosing decoding temperature match sound probabilistic practice?",
    options: [
      {
        text: "Lower temperature is often appropriate for factual or constrained tasks that need stable outputs.",
        isCorrect: true,
      },
      {
        text: "Medium to higher temperature can be useful when several creative alternatives are desired.",
        isCorrect: true,
      },
      {
        text: "Lower temperature can make output more stable without automatically making it more truthful.",
        isCorrect: true,
      },
      {
        text: "Truthfulness can still depend on model knowledge, retrieval, reasoning, prompting, and verification.",
        isCorrect: true,
      },
    ],
    explanation:
      "Temperature is a practical diversity and stability control, not a truthfulness guarantee. Lower values often fit constrained tasks, higher values can support creative variation, and high-stakes reliability still needs evidence and verification outside the temperature setting.",
  },
  {
    id: "crash-probability-l5-q13",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly connect temperature, entropy, and output diversity?",
    options: [
      {
        text: "A lower-temperature distribution usually has lower entropy because probability mass is concentrated on fewer outcomes.",
        isCorrect: true,
      },
      {
        text: "A higher-temperature distribution usually has higher entropy because lower-probability outcomes receive more probability mass.",
        isCorrect: true,
      },
      {
        text: "Changing temperature can alter output diversity even when the underlying prompt and model weights are unchanged.",
        isCorrect: true,
      },
      {
        text: "Higher entropy guarantees that the sampled answer is semantically better than a lower-entropy answer.",
        isCorrect: false,
      },
    ],
    explanation:
      "Entropy measures uncertainty or spread in a distribution, so sharper distributions tend to have lower entropy and flatter distributions tend to have higher entropy. Higher entropy can support diversity, but it does not guarantee correctness, usefulness, or semantic quality.",
  },
  {
    id: "crash-probability-l5-q14",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements correctly define a latent variable in a generative setting?",
    options: [
      {
        text: "A latent variable is hidden or not directly observed.",
        isCorrect: true,
      },
      {
        text: "A latent variable can help explain or generate observed data.",
        isCorrect: true,
      },
      {
        text: "A latent variable must be a visible label printed in the dataset.",
        isCorrect: false,
      },
      {
        text: "A latent variable is the same thing as the final sampled output in every generative model.",
        isCorrect: false,
      },
    ],
    explanation:
      "A latent variable represents hidden structure, such as topic, style, intent, disease state, or user preference. It is useful because observed data often has underlying causes or factors that are not directly labeled.",
  },
  {
    id: "crash-probability-l5-q15",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "A document contains the words `stock`, `inflation`, `market`, and `central bank`, but no topic label is printed. Which interpretations are appropriate?",
    options: [
      {
        text: "A topic such as economics can be treated as a possible latent variable explaining the observed words.",
        isCorrect: true,
      },
      {
        text: "The observed words are data \\(X\\), while a hidden topic can be represented as \\(Z\\).",
        isCorrect: true,
      },
      {
        text: "The topic cannot be latent because humans can guess it from the words.",
        isCorrect: false,
      },
      {
        text: "The topic must be the next token selected by greedy decoding.",
        isCorrect: false,
      },
    ],
    explanation:
      "A variable can be latent even if it is inferable from evidence; latent means it is not directly observed as a recorded variable. In this example, the words are observed data and the topic is a hidden explanatory factor.",
  },
  {
    id: "crash-probability-l5-q16",
    chapter: 5,
    difficulty: "medium",
    type: "assertion-reason",
    prompt:
      "Assertion: In a simple latent-variable generative model, \\(Z\\) can be sampled before generating \\(X\\).\n\nReason: The model can represent generation as \\(z \\sim P(z)\\), then \\(x \\sim P(x \\mid z)\\).",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: false },
      { text: "Assertion is false, Reason is true.", isCorrect: false },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: true,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "The assertion is true because latent-variable generation often starts by drawing a hidden factor. The reason is true and explains it: first sample the latent variable from a prior, then sample observed data conditioned on that latent value.",
  },
  {
    id: "crash-probability-l5-q17",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the probabilistic structure \\(z \\sim P(z)\\), \\(x \\sim P(x \\mid z)\\)?",
    options: [
      {
        text: "\\(P(z)\\) describes how hidden variables are sampled before observed data is generated.",
        isCorrect: true,
      },
      {
        text: "\\(P(x \\mid z)\\) describes a distribution over visible data conditioned on a latent value.",
        isCorrect: true,
      },
      {
        text: "Different sampled values of \\(z\\) can produce different styles, topics, identities, or other hidden structures in \\(x\\).",
        isCorrect: true,
      },
      {
        text: "The structure can model hidden causes that make observed data less like random pixels or random words.",
        isCorrect: true,
      },
    ],
    explanation:
      "The notation says that a hidden variable is sampled and then observed data is sampled given that hidden value. All four statements preserve the key distinction between hidden structure and visible data in a generative model.",
  },
  {
    id: "crash-probability-l5-q18",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly distinguish latent variables, latent spaces, and embeddings?",
    options: [
      {
        text: "A probabilistic latent variable can be an explicitly sampled hidden variable in a generative process.",
        isCorrect: true,
      },
      {
        text: "An embedding or latent space can be a learned hidden representation even when it is not used as an explicitly sampled probabilistic variable.",
        isCorrect: true,
      },
      {
        text: "Every embedding vector must be sampled from \\(P(z)\\) before a model can use it.",
        isCorrect: false,
      },
      {
        text: "Latent spaces and latent variables are unrelated because one appears in neural networks and the other appears only in classical statistics.",
        isCorrect: false,
      },
    ],
    explanation:
      "Latent variables and embeddings share the idea of hidden internal structure, but explicit probabilistic sampling and learned vector representation are not identical operations. The incorrect options overstate the connection in one direction or deny a real conceptual relationship.",
  },
  {
    id: "crash-probability-l5-q19",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements fit a high-level Variational Autoencoder (VAE) generative model?",
    options: [
      {
        text: "A data point \\(x\\) can be encoded into a latent representation \\(z\\).",
        isCorrect: true,
      },
      {
        text: "A latent representation can be sampled or regularized before decoding.",
        isCorrect: true,
      },
      {
        text: "A decoder can map \\(z\\) back toward visible data such as \\(\\hat{x}\\) or a generated sample.",
        isCorrect: true,
      },
      {
        text: "Using a Variational Autoencoder (VAE) for generation requires greedy decoding over next-token logits.",
        isCorrect: false,
      },
    ],
    explanation:
      "The VAE-style intuition is encode data into a latent representation, sample or regularize that representation, and decode back toward visible data. Greedy next-token decoding belongs to language-model generation and is not the defining operation of a VAE.",
  },
  {
    id: "crash-probability-l5-q20",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "In the simplified generative use of a Variational Autoencoder (VAE), which sequence is most appropriate?",
    options: [
      {
        text: "Sample \\(z\\) from a prior such as \\(P(z)\\), then decode from \\(z\\) to data using something like \\(P(x \\mid z)\\).",
        isCorrect: true,
      },
      {
        text: "Choose the most frequent training example, copy its pixels, and call the copied image a latent variable.",
        isCorrect: false,
      },
      {
        text: "Start from text-token logits, apply top-p sampling, and treat the resulting token as a Gaussian prior.",
        isCorrect: false,
      },
      {
        text: "Train only a reward function \\(R(s,a,s')\\), then use discounted return as the decoder.",
        isCorrect: false,
      },
    ],
    explanation:
      "The simplified generative story is to sample a latent value and decode it into visible data. The incorrect options confuse VAE-style generation with memorization, language-model decoding, or reinforcement-learning reward notation.",
  },
  {
    id: "crash-probability-l5-q21",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "For a Gaussian or normal distribution written \\(X \\sim \\mathcal{N}(\\mu,\\sigma^2)\\), which statements are correct?",
    options: [
      {
        text: "\\(\\mu\\) is the mean.",
        isCorrect: true,
      },
      {
        text: "\\(\\sigma^2\\) is the variance.",
        isCorrect: true,
      },
      {
        text: "\\(\\sigma\\) is the standard deviation.",
        isCorrect: true,
      },
      {
        text: "Larger variance means samples are more spread out.",
        isCorrect: true,
      },
    ],
    explanation:
      "The notation \\(\\mathcal{N}(\\mu,\\sigma^2)\\) uses mean and variance as its two parameters. Standard deviation is the square root of variance, and increasing variance spreads probability mass farther from the mean.",
  },
  {
    id: "crash-probability-l5-q22",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Why is Gaussian noise especially common in generative AI methods such as diffusion models?",
    options: [
      {
        text: "It is mathematically convenient, easy to sample, and useful for representing uncertainty in high-dimensional data.",
        isCorrect: true,
      },
      {
        text: "It removes the need for a learned reverse process because Gaussian samples already contain clean images.",
        isCorrect: false,
      },
      {
        text: "It guarantees that every noised image has the same visible object identity as the original.",
        isCorrect: false,
      },
      {
        text: "It is used only because language models generate discrete tokens.",
        isCorrect: false,
      },
    ],
    explanation:
      "Gaussian noise is useful because it is easy to generate, analytically manageable, and a good controlled source of randomness. Diffusion still needs a learned reverse process because random noise by itself is not a clean image or a complete generator.",
  },
  {
    id: "crash-probability-l5-q23",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "An RGB image has resolution \\(512 \\times 512\\). Which statements correctly connect images and Gaussian noise?",
    options: [
      {
        text: "The image can be viewed as a high-dimensional vector with \\(512 \\times 512 \\times 3\\) pixel-channel values.",
        isCorrect: true,
      },
      {
        text: "Adding larger-variance Gaussian noise generally makes the original image less recognizable.",
        isCorrect: true,
      },
      {
        text: "Adding Gaussian noise changes only the image caption, not the pixel values.",
        isCorrect: false,
      },
      {
        text: "If Gaussian noise is added to every pixel channel, the result must remain as clear as the original image.",
        isCorrect: false,
      },
    ],
    explanation:
      "Images can be represented as high-dimensional arrays or vectors, with separate channels for red, green, and blue. Noise perturbs those values, and larger noise variance makes visible structure harder to recover without a denoising model.",
  },
  {
    id: "crash-probability-l5-q24",
    chapter: 5,
    difficulty: "hard",
    type: "assertion-reason",
    prompt:
      "Assertion: In generative AI, noise can be useful rather than merely an error.\n\nReason: Gaussian noise is mathematically convenient and easy to sample.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: false },
      { text: "Assertion is false, Reason is true.", isCorrect: false },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: false,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: true,
      },
    ],
    explanation:
      "The assertion is true because noise can serve as uncertainty, regularization, exploration, latent randomness, or the starting material for diffusion generation. The reason is also true, but it is not the whole explanation for usefulness; convenience and easy sampling help, while the generative role comes from learning to transform noise into structured data.",
  },
  {
    id: "crash-probability-l5-q25",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements correctly identify the forward and reverse processes in a diffusion model?",
    options: [
      {
        text: "The forward process gradually adds noise to real data, moving from \\(x_0\\) toward \\(x_T\\).",
        isCorrect: true,
      },
      {
        text: "The reverse process starts from noise and gradually denoises toward a clean generated sample.",
        isCorrect: true,
      },
      {
        text: "The forward process is the learned image-generation process used at inference time.",
        isCorrect: false,
      },
      {
        text: "The reverse process means sorting tokens by probability from highest to lowest.",
        isCorrect: false,
      },
    ],
    explanation:
      "Diffusion uses a forward noising process to corrupt real data and a reverse denoising process to generate data. The forward process is usually fixed corruption, while the reverse process is the learned generative direction.",
  },
  {
    id: "crash-probability-l5-q26",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements are correct about a forward noising process \\(q(x_t\\mid x_{t-1})\\) in a diffusion model?",
    options: [
      {
        text: "\\(x_0\\) represents clean real data such as an image.",
        isCorrect: true,
      },
      {
        text: "Each later step is noisier than the previous step when the noise schedule gradually increases corruption.",
        isCorrect: true,
      },
      {
        text: "By \\(x_T\\), the sample is close to pure Gaussian noise.",
        isCorrect: true,
      },
      {
        text: "The forward process is generation because it turns random noise directly into a clean image.",
        isCorrect: false,
      },
    ],
    explanation:
      "The forward process is a Markov noising chain that starts with clean data and gradually corrupts it. It creates noisy training inputs for a learned reverse process, but the forward direction itself destroys structure rather than generating new clean data.",
  },
  {
    id: "crash-probability-l5-q27",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "During diffusion training, what can the neural network be trained to predict from a noisy sample \\(x_t\\) and time step \\(t\\)?",
    options: [
      {
        text: "The added noise \\(\\epsilon\\).",
        isCorrect: true,
      },
      {
        text: "The original clean image \\(x_0\\).",
        isCorrect: true,
      },
      {
        text: "A previous less-noisy image such as \\(x_{t-1}\\), depending on the implementation.",
        isCorrect: true,
      },
      {
        text: "The next token in a discrete sentence, regardless of the image input.",
        isCorrect: false,
      },
    ],
    explanation:
      "Diffusion implementations can parameterize the prediction target in several related ways, including noise, clean data, or the previous step. The common point is that the network learns how to move from noisy data toward cleaner data, not how to perform ordinary next-token prediction on unrelated text.",
  },
  {
    id: "crash-probability-l5-q28",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which expression best represents the conditional generation target for a text-to-image diffusion model guided by a prompt \\(c\\)?",
    options: [
      {
        text: "\\(P(\\text{image} \\mid c)\\), with denoising steps such as \\(P(x_{t-1} \\mid x_t,t,c)\\).",
        isCorrect: true,
      },
      {
        text: "\\(P(c \\mid \\text{image})\\), with no distribution over generated images.",
        isCorrect: false,
      },
      {
        text: "\\(P(\\text{image})=1\\) for the single image specified exactly by the prompt.",
        isCorrect: false,
      },
      {
        text: "\\(\\pi(a \\mid s)\\), because every prompt-conditioned image model is an RL policy.",
        isCorrect: false,
      },
    ],
    explanation:
      "Text-to-image diffusion is conditional generation: the prompt guides a distribution over possible images. The reversed conditional, a deterministic one-image claim, and a reinforcement-learning policy notation all miss the prompt-guided denoising structure.",
  },
  {
    id: "crash-probability-l5-q29",
    chapter: 5,
    difficulty: "medium",
    type: "assertion-reason",
    prompt:
      "Assertion: Diffusion models use many denoising steps partly because turning pure static into a detailed image in one step is hard.\n\nReason: Each denoising step can learn a smaller conditional move from a noisy sample toward a cleaner sample.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: false },
      { text: "Assertion is false, Reason is true.", isCorrect: false },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: true,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "The assertion is true because direct one-step generation from random noise to a detailed sample is a difficult transformation. The reason explains the design: many smaller conditional denoising steps let the model gradually form rough structure, objects, details, and a final image.",
  },
  {
    id: "crash-probability-l5-q30",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Why can the same text prompt produce different diffusion images across runs?",
    options: [
      {
        text: "Generation can start from different random noise samples.",
        isCorrect: true,
      },
      {
        text: "The prompt often leaves many valid details unspecified, such as viewpoint, lighting, season, or style.",
        isCorrect: true,
      },
      {
        text: "The model must delete the prompt after every denoising step and sample a new prompt.",
        isCorrect: false,
      },
      {
        text: "The model is forced to choose the same highest-probability image every time.",
        isCorrect: false,
      },
    ],
    explanation:
      "A prompt such as a red house in the mountains describes a broad set of possible images, not a single fully specified picture. Different noise starts and stochastic denoising choices can select different plausible members of that set.",
  },
  {
    id: "crash-probability-l5-q31",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe uncertainty during diffusion generation?",
    options: [
      {
        text: "At \\(x_T\\), many final images are possible because the sample is close to random noise.",
        isCorrect: true,
      },
      {
        text: "As denoising proceeds, the sample becomes increasingly committed to a particular structured output.",
        isCorrect: true,
      },
      {
        text: "Uncertainty increases until the final clean image has no relation to the prompt.",
        isCorrect: false,
      },
      {
        text: "The reverse process is deterministic lookup from a fixed table of all possible images.",
        isCorrect: false,
      },
    ],
    explanation:
      "Diffusion can be understood as turning random uncertainty into structured data through repeated denoising. The prompt and learned model guide the process, so uncertainty does not simply increase and the output is not a table lookup.",
  },
  {
    id: "crash-probability-l5-q32",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly compare language-model generation and diffusion generation?",
    options: [
      {
        text: "A language model often generates one discrete token at a time.",
        isCorrect: true,
      },
      {
        text: "A diffusion model often generates by updating a continuous noisy representation through denoising steps.",
        isCorrect: true,
      },
      {
        text: "Both can be conditional generative systems, using context or a prompt to guide outputs.",
        isCorrect: true,
      },
      {
        text: "Both use exactly the same data type, objective, and decoding algorithm.",
        isCorrect: false,
      },
    ],
    explanation:
      "Language models and diffusion models both rely on probability and conditioning, but their generation mechanisms differ. Language models work over token sequences, while diffusion models denoise continuous representations such as images, audio, or video latents.",
  },
  {
    id: "crash-probability-l5-q33",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "How does reinforcement learning (RL) fit a repeated conditional-sampling pattern?",
    options: [
      {
        text: "A policy \\(\\pi(a \\mid s)\\) can define a distribution over actions given a state.",
        isCorrect: true,
      },
      {
        text: "An agent can sample or choose an action, receive reward, and transition to another state.",
        isCorrect: true,
      },
      {
        text: "Exploration can involve deliberate randomness in actions or policies.",
        isCorrect: true,
      },
      {
        text: "RL is unrelated to probability because rewards are always known before actions are chosen.",
        isCorrect: false,
      },
    ],
    explanation:
      "Reinforcement learning uses probability in policies, transitions, exploration, and expected return. It fits the repeated conditional pattern because agents repeatedly choose actions from state-conditioned rules and then observe uncertain consequences.",
  },
  {
    id: "crash-probability-l5-q34",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which formulas or descriptions correctly match the shared repeated-conditional-step pattern across AI systems?",
    options: [
      {
        text: "Language model generation uses terms like \\(P(x_t \\mid x_1,\\ldots,x_{t-1})\\) for the next token.",
        isCorrect: true,
      },
      {
        text: "Diffusion generation uses steps like \\(P(x_{t-1} \\mid x_t,\\text{condition})\\) or a related noise-prediction target.",
        isCorrect: true,
      },
      {
        text: "Reinforcement learning policies can use \\(\\pi(a \\mid s)\\) to choose actions from a state.",
        isCorrect: true,
      },
      {
        text: "All three can be viewed as repeatedly making conditional probabilistic moves rather than only producing one isolated label.",
        isCorrect: true,
      },
    ],
    explanation:
      "The common pattern is repeated conditional probability: next token from context, denoising step from noisy state and condition, or action from state. The details differ, but each system uses probability to move from a current context or state to a next concrete step.",
  },
  {
    id: "crash-probability-l5-q35",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which notation-role matches correctly connect probability to major AI workflows?",
    options: [
      {
        text: "Prediction estimates probabilities over outputs, such as \\(P(y \\mid x)\\).",
        isCorrect: true,
      },
      {
        text: "Learning often makes observed training data more probable under the model, as in negative log-likelihood or cross-entropy.",
        isCorrect: true,
      },
      {
        text: "Decision-making can involve choosing actions to maximize expected future reward.",
        isCorrect: true,
      },
      {
        text: "Generation turns learned distributions into concrete outputs through sampling or related decoding.",
        isCorrect: true,
      },
    ],
    explanation:
      "These four roles connect probability to prediction, learning, decision-making, and generation. The notation emphasizes that modern AI systems use distributions not only to score outputs, but also to train models, select actions, and produce samples.",
  },
  {
    id: "crash-probability-l5-q36",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which notation-to-idea matches are correct in the final course summary?",
    options: [
      {
        text: "\\(P(y \\mid x)\\) represents prediction from information or input.",
        isCorrect: true,
      },
      {
        text: "\\(-\\log P(\\text{correct output})\\) represents a learning loss that penalizes low probability on observed correct outputs.",
        isCorrect: true,
      },
      {
        text: "\\(x \\sim P_\\theta(x)\\) represents sampling from a learned generative distribution.",
        isCorrect: true,
      },
      {
        text: "\\(\\mathbb{E}[G_t]\\) represents selecting the rarest token in language-model decoding.",
        isCorrect: false,
      },
    ],
    explanation:
      "The prediction, learning, and generation formulas match the roles emphasized in the final synthesis. \\(\\mathbb{E}[G_t]\\) belongs to reinforcement-learning decision-making and expected return, not rare-token selection.",
  },
  {
    id: "crash-probability-l5-q37",
    chapter: 5,
    difficulty: "hard",
    type: "assertion-reason",
    prompt:
      "Assertion: Greedy decoding is always more correct than sampling because it chooses the most likely token at every step.\n\nReason: A locally most likely token at one step does not always lead to the best full sequence.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: false },
      { text: "Assertion is false, Reason is true.", isCorrect: true },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: false,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "The assertion is false because greedy decoding can be useful but is not automatically more correct for every task or full sequence. The reason is true: generation is sequential, so local next-token choices can create poor global outputs.",
  },
  {
    id: "crash-probability-l5-q38",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly capture the optional classifier-free guidance intuition for text-to-image diffusion?",
    options: [
      {
        text: "Guidance changes how strongly the prompt condition influences denoising.",
        isCorrect: true,
      },
      {
        text: "Very high guidance can improve prompt adherence while sometimes making images less natural or more distorted.",
        isCorrect: true,
      },
      {
        text: "Guidance removes the need for random initial noise because the prompt becomes the image.",
        isCorrect: false,
      },
      {
        text: "Guidance is the same as lowering token-sampling temperature in a language model.",
        isCorrect: false,
      },
    ],
    explanation:
      "Classifier-free guidance is presented as an intuition about changing the strength of the conditioning signal during denoising. It is not a replacement for noise, and it is not identical to temperature control in next-token sampling.",
  },
  {
    id: "crash-probability-l5-q39",
    chapter: 5,
    difficulty: "hard",
    type: "assertion-reason",
    prompt:
      "Assertion: A score-based intuition for diffusion is that the model learns a direction from noisy data back toward likely data.\n\nReason: This intuition says the model can replace all denoising steps with a single exact reconstruction of the final image.",
    options: [
      { text: "Assertion is true, Reason is false.", isCorrect: true },
      { text: "Assertion is false, Reason is true.", isCorrect: false },
      { text: "Both are false.", isCorrect: false },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: false,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "The assertion is true as a high-level score-based intuition: each step asks which direction makes a noisy sample look more like real data. The reason is false because this intuition does not imply that diffusion becomes a single exact reconstruction step.",
  },
  {
    id: "crash-probability-l5-q40",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which explanation best separates sampling controls from the learned probability model in generative AI?",
    options: [
      {
        text: "Controls such as temperature, top-k, top-p, guidance, or random seed influence how a learned distribution is used, but they do not by themselves create new model knowledge.",
        isCorrect: true,
      },
      {
        text: "Sampling controls retrain the neural network after each prompt, so decoding and learning are the same operation.",
        isCorrect: false,
      },
      {
        text: "Sampling controls remove uncertainty from generative AI, so every prompt has exactly one correct output.",
        isCorrect: false,
      },
      {
        text: "Sampling controls matter only for diffusion models and have no connection to language models or RL policies.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sampling controls shape how outputs are drawn from, restricted by, or guided through the model's existing probability structure. They can change diversity, adherence, or repeatability, but the underlying learned knowledge and the need for validation remain separate issues.",
  },
  {
    id: "crash-probability-l5-q41",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "In 200 independent samples from a next-token distribution with \\(P(\\text{mat})=0.50\\), \\(P(\\text{sofa})=0.25\\), \\(P(\\text{floor})=0.15\\), \\(P(\\text{chair})=0.07\\), and \\(P(\\text{car})=0.03\\), which statements are correct?",
    options: [
      {
        text: "The expected count of `sofa` is \\(200\\cdot0.25=50\\).",
        isCorrect: true,
      },
      {
        text: "The variance of the `car` count is \\(200\\cdot0.03\\cdot0.97\\).",
        isCorrect: true,
      },
      {
        text: "The covariance between the `sofa` count and the `car` count is \\(-200\\cdot0.25\\cdot0.03\\) under the multinomial model.",
        isCorrect: true,
      },
      {
        text: "The probability that the first two independent samples are both `mat` is \\(0.50^2=0.25\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "Repeated independent token sampling can be modeled with binomial or multinomial count variables. Means, variances, covariances, and multi-sample event probabilities come from the same categorical distribution rather than from the single greedy choice.",
  },
  {
    id: "crash-probability-l5-q42",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Tokens are sorted by probability as \\(a:0.36\\), \\(b:0.24\\), \\(c:0.18\\), \\(d:0.12\\), and \\(e:0.10\\). With nucleus sampling threshold \\(p=0.75\\), which statements are correct?",
    options: [
      {
        text: "The nucleus contains \\(a,b,c\\) because \\(0.36+0.24=0.60\\) is too small and adding \\(c\\) gives \\(0.78\\).",
        isCorrect: true,
      },
      {
        text: "After renormalization, \\(P'(c)=0.18/0.78\\).",
        isCorrect: true,
      },
      {
        text: "Token \\(d\\) is excluded even though it has nonzero original probability.",
        isCorrect: true,
      },
      {
        text: "After truncation, the probabilities assigned to \\(a,b,c\\) still sum to \\(0.78\\) rather than 1.",
        isCorrect: false,
      },
    ],
    explanation:
      "Top-p sampling first chooses the smallest high-probability prefix whose cumulative mass crosses the threshold, then renormalizes within that prefix. Excluded tokens have zero probability under the truncated sampler, and included probabilities must sum to one after renormalization.",
  },
  {
    id: "crash-probability-l5-q43",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "For logits \\((3,1,0)\\), temperature-modified softmax uses \\(P_i(T)=\\frac{e^{z_i/T}}{\\sum_j e^{z_j/T}}\\) with \\(T>0\\). Which statements are correct?",
    options: [
      {
        text: "At \\(T=1\\), the odds of the first token over the second token are \\(e^{3-1}=e^2\\).",
        isCorrect: true,
      },
      {
        text: "At \\(T=2\\), the odds of the first token over the second token are \\(e^{(3-1)/2}=e\\).",
        isCorrect: true,
      },
      {
        text: "For every positive \\(T\\), the first token remains the highest-probability token.",
        isCorrect: true,
      },
      {
        text: "As \\(T\\rightarrow\\infty\\), the distribution approaches uniform over the three tokens.",
        isCorrect: true,
      },
    ],
    explanation:
      "Temperature rescales logit gaps before exponentiation, so pairwise odds become \\(e^{(z_i-z_j)/T}\\). Positive temperature preserves the ordering of unequal logits, while very large temperature washes out the differences and approaches a uniform distribution.",
  },
  {
    id: "crash-probability-l5-q44",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Use base-2 entropy \\(H(P)=-\\sum_i p_i\\log_2 p_i\\). Which statements are correct for \\(P=(1/2,1/4,1/4)\\) and \\(Q=(1/3,1/3,1/3)\\)?",
    options: [
      {
        text: "\\(H(P)=1.5\\) bits.",
        isCorrect: true,
      },
      {
        text: "\\(H(Q)=\\log_2 3\\), which is about 1.585 bits.",
        isCorrect: true,
      },
      {
        text: "\\(Q\\) has lower entropy than \\(P\\) because all outcomes in \\(Q\\) are equally likely.",
        isCorrect: false,
      },
      {
        text: "Entropy is the sampled token itself, not the expected surprisal of the distribution.",
        isCorrect: false,
      },
    ],
    explanation:
      "For \\(P\\), the entropy is \\(0.5\\cdot1+0.25\\cdot2+0.25\\cdot2=1.5\\) bits. The uniform three-outcome distribution has maximum entropy among three-outcome distributions, so it is slightly higher than \\(P\\), and entropy is a property of the distribution rather than the realized token.",
  },
  {
    id: "crash-probability-l5-q45",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "At the first token, a model assigns \\(P(A)=0.6\\) and \\(P(B)=0.4\\). If \\(A\\) is chosen, the most likely second token has probability 0.2; if \\(B\\) is chosen, the most likely second token has probability 0.9. Which statements are correct?",
    options: [
      {
        text: "Greedy decoding at the first step chooses \\(A\\).",
        isCorrect: true,
      },
      {
        text: "The best two-token sequence beginning with \\(A\\) has probability \\(0.6\\cdot0.2=0.12\\).",
        isCorrect: true,
      },
      {
        text: "The best two-token sequence beginning with \\(B\\) has probability \\(0.4\\cdot0.9=0.36\\).",
        isCorrect: true,
      },
      {
        text: "Because \\(A\\) has the larger first-step probability, every two-token sequence beginning with \\(A\\) is more likely than every two-token sequence beginning with \\(B\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The calculation shows why a locally largest first token need not produce the highest-probability full sequence. Greedy decoding optimizes one step at a time, while sequence probability multiplies conditional probabilities across steps.",
  },
  {
    id: "crash-probability-l5-q46",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "A sorted token distribution is \\(a:0.40\\), \\(b:0.22\\), \\(c:0.18\\), \\(d:0.12\\), and \\(e:0.08\\). Which statements are correct for top-k with \\(k=3\\) and top-p with \\(p=0.70\\)?",
    options: [
      {
        text: "Top-k with \\(k=3\\) keeps \\(a,b,c\\), whose original probability mass is \\(0.80\\).",
        isCorrect: true,
      },
      {
        text: "Under top-k with \\(k=3\\), the renormalized probability of \\(b\\) is \\(0.22/0.80\\).",
        isCorrect: true,
      },
      {
        text: "Top-p with \\(p=0.70\\) also keeps \\(a,b,c\\), because \\(0.40+0.22=0.62\\) is below the threshold and adding \\(c\\) reaches \\(0.80\\).",
        isCorrect: true,
      },
      {
        text: "Token \\(d\\) is excluded by both of these restricted samplers.",
        isCorrect: true,
      },
    ],
    explanation:
      "Top-k fixes the number of retained tokens, while top-p fixes a cumulative-probability threshold. In this example they happen to retain the same set, and both require renormalization over the retained mass.",
  },
  {
    id: "crash-probability-l5-q47",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "A decoding rule restricts a distribution \\(p_i\\) to a retained set \\(S\\) with total mass \\(m=\\sum_{i\\in S}p_i\\). Which statements are correct?",
    options: [
      {
        text: "The restricted categorical distribution should use \\(p'_i=p_i/m\\) for \\(i\\in S\\).",
        isCorrect: true,
      },
      {
        text: "Expected values under the restricted distribution can differ from expected values under the original distribution.",
        isCorrect: true,
      },
      {
        text: "The unnormalized values \\(p_i\\) can be used directly as probabilities after truncation even though they sum to \\(m<1\\).",
        isCorrect: false,
      },
      {
        text: "Tokens outside \\(S\\) keep their original probabilities during the restricted sample.",
        isCorrect: false,
      },
    ],
    explanation:
      "Truncation changes the support of the categorical distribution, so the remaining probabilities must be divided by their total retained mass. This can change probabilities and expectations because excluded outcomes receive zero probability under the restricted sampler.",
  },
  {
    id: "crash-probability-l5-q48",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "A latent variable model has \\(P(Z=0)=0.3\\), \\(P(Z=1)=0.7\\), \\(P(X=x\\mid Z=0)=0.8\\), and \\(P(X=x\\mid Z=1)=0.2\\). What is \\(P(X=x)\\)?",
    options: [
      {
        text: "\\(0.3\\cdot0.8+0.7\\cdot0.2=0.38\\).",
        isCorrect: true,
      },
      {
        text: "\\(0.8+0.2=1.0\\), because the conditional probabilities are added without latent weights.",
        isCorrect: false,
      },
      {
        text: "\\(0.3+0.7=1.0\\), because the prior over \\(Z\\) already sums to one.",
        isCorrect: false,
      },
      {
        text: "\\(0.3\\cdot0.2+0.7\\cdot0.8=0.62\\), because the conditional probabilities should be swapped across latent states.",
        isCorrect: false,
      },
    ],
    explanation:
      "The marginal probability of the observed event sums over the latent alternatives: \\(P(x)=\\sum_z P(z)P(x\\mid z)\\). The latent prior weights matter, and each conditional probability must stay paired with the latent state it conditions on.",
  },
  {
    id: "crash-probability-l5-q49",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Using \\(P(Z=0)=0.3\\), \\(P(Z=1)=0.7\\), \\(P(X=x\\mid Z=0)=0.8\\), and \\(P(X=x\\mid Z=1)=0.2\\), which statements about the posterior after observing \\(X=x\\) are correct?",
    options: [
      {
        text: "\\(P(Z=0,X=x)=0.3\\cdot0.8=0.24\\).",
        isCorrect: true,
      },
      {
        text: "\\(P(Z=0\\mid X=x)=0.24/0.38\\).",
        isCorrect: true,
      },
      {
        text: "\\(P(Z=0\\mid X=x)=0.3\\), because observing \\(x\\) cannot update a latent variable.",
        isCorrect: false,
      },
      {
        text: "\\(P(Z=0\\mid X=x)=0.8/0.2\\), because posterior probabilities compare only likelihoods.",
        isCorrect: false,
      },
    ],
    explanation:
      "Bayes' rule updates the latent-state probability using both the prior and the likelihood. The posterior numerator is the joint probability for \\(Z=0\\) and \\(x\\), and the denominator is the marginal probability of \\(x\\) from summing over latent states.",
  },
  {
    id: "crash-probability-l5-q50",
    chapter: 5,
    difficulty: "medium",
    prompt: "For \\(X\\sim\\mathcal{N}(5,4)\\), which statements are correct?",
    options: [
      {
        text: "The standard deviation is \\(2\\).",
        isCorrect: true,
      },
      {
        text: "\\(Z=(X-5)/2\\) has a standard normal distribution.",
        isCorrect: true,
      },
      {
        text: "\\(P(|X-5|\\le4)=P(|Z|\\le2)\\).",
        isCorrect: true,
      },
      {
        text: "The parameter \\(4\\) is the variance, not the standard deviation.",
        isCorrect: true,
      },
    ],
    explanation:
      "The notation \\(\\mathcal{N}(\\mu,\\sigma^2)\\) uses variance as the second parameter, so \\(\\sigma=2\\). Standardizing subtracts the mean and divides by the standard deviation, which converts interval statements about \\(X\\) into equivalent statements about a standard normal variable.",
  },
  {
    id: "crash-probability-l5-q51",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Suppose a noising step is written \\(x_t=\\sqrt{\\alpha}\\,x_0+\\sqrt{1-\\alpha}\\,\\epsilon\\), where \\(0\\le\\alpha\\le1\\), \\(x_0\\) is fixed, and \\(\\epsilon\\sim\\mathcal{N}(0,I)\\). Which statements are correct?",
    options: [
      {
        text: "\\(\\mathbb{E}[x_t\\mid x_0]=\\sqrt{\\alpha}\\,x_0\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\operatorname{Cov}(x_t\\mid x_0)=(1-\\alpha)I\\).",
        isCorrect: true,
      },
      {
        text: "Smaller \\(\\alpha\\) gives the noise term more relative influence.",
        isCorrect: true,
      },
      {
        text: "For every data distribution of \\(x_0\\), the marginal distribution of \\(x_t\\) is exactly \\(\\mathcal{N}(0,I)\\) for all \\(\\alpha\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Conditioning on fixed \\(x_0\\), the only random term is the Gaussian noise, so expectation and covariance follow from affine transformation rules. The marginal distribution over \\(x_t\\) also depends on the data distribution unless the process has fully washed out the data contribution.",
  },
  {
    id: "crash-probability-l5-q52",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the Markov structure of a forward diffusion noising chain?",
    options: [
      {
        text: "A common factorization is \\(q(x_{1:T}\\mid x_0)=\\prod_{t=1}^T q(x_t\\mid x_{t-1})\\).",
        isCorrect: true,
      },
      {
        text: "Given \\(x_{t-1}\\), the next noised state \\(x_t\\) does not need the full earlier history \\(x_{0:t-2}\\) in a Markov forward chain.",
        isCorrect: true,
      },
      {
        text: "Closed-form expressions for \\(q(x_t\\mid x_0)\\) are useful because training often samples a time step directly.",
        isCorrect: true,
      },
      {
        text: "A noise schedule can control how much corruption is added at different time steps.",
        isCorrect: true,
      },
    ],
    explanation:
      "The forward process is usually designed as a Markov chain, which gives a product factorization over one-step noising transitions. Closed-form noising from \\(x_0\\) and a controlled noise schedule make training practical because the model can see noisy data at many levels of corruption.",
  },
  {
    id: "crash-probability-l5-q53",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly interpret a learned reverse diffusion transition \\(p_\\theta(x_{t-1}\\mid x_t,c)\\), where \\(c\\) is a condition such as a text prompt?",
    options: [
      {
        text: "It is a model-parameterized conditional distribution for moving from a noisy state toward a less-noisy state.",
        isCorrect: true,
      },
      {
        text: "Sampling from it can produce different denoising paths even with the same condition.",
        isCorrect: true,
      },
      {
        text: "It is the same object as the fixed forward noising transition \\(q(x_t\\mid x_{t-1})\\).",
        isCorrect: false,
      },
      {
        text: "Removing \\(c\\) from the conditioning set cannot change the distribution because prompts are not probabilistic information.",
        isCorrect: false,
      },
    ],
    explanation:
      "The reverse transition is learned and conditionally generates a less-noisy state from the current noisy state and optional guidance information. It is not the fixed corruption process, and conditioning information can change the distribution of plausible denoising moves.",
  },
  {
    id: "crash-probability-l5-q54",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "A diffusion model forms \\(x_t\\) by adding known noise \\(\\epsilon\\) to \\(x_0\\), then trains \\(\\epsilon_\\theta(x_t,t,c)\\) with squared error. Which statements are correct?",
    options: [
      {
        text: "The loss can be written in simplified form as \\(\\mathbb{E}[\\|\\epsilon-\\epsilon_\\theta(x_t,t,c)\\|^2]\\).",
        isCorrect: true,
      },
      {
        text: "Including \\(t\\) matters because the amount and character of noise depend on the time step.",
        isCorrect: true,
      },
      {
        text: "Under squared error, the best predictor is a conditional mean of the target noise given the inputs.",
        isCorrect: true,
      },
      {
        text: "Training maximizes this squared-error loss so the model predicts noise as poorly as possible.",
        isCorrect: false,
      },
    ],
    explanation:
      "A common diffusion training target is noise prediction, where the model receives the noisy sample, time step, and condition and predicts the added noise. Squared-error minimization encourages the prediction to approximate the conditional expectation of that noise, not to maximize error.",
  },
  {
    id: "crash-probability-l5-q55",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "A classifier-free guidance style update can be written schematically as \\(\\hat{\\epsilon}=\\epsilon_{\\text{uncond}}+w(\\epsilon_{\\text{cond}}-\\epsilon_{\\text{uncond}})\\). Which statements are correct?",
    options: [
      {
        text: "When \\(w=0\\), \\(\\hat{\\epsilon}=\\epsilon_{\\text{uncond}}\\).",
        isCorrect: true,
      },
      {
        text: "When \\(w=1\\), \\(\\hat{\\epsilon}=\\epsilon_{\\text{cond}}\\).",
        isCorrect: true,
      },
      {
        text: "When \\(w>1\\), the update extrapolates beyond the conditional prediction in the direction away from the unconditional prediction.",
        isCorrect: true,
      },
      {
        text: "Increasing guidance can improve prompt adherence while sometimes harming naturalness or introducing distortion.",
        isCorrect: true,
      },
    ],
    explanation:
      "The guidance expression interpolates or extrapolates between unconditional and conditional predictions. It changes how strongly the condition influences denoising, which can improve adherence but also creates a tradeoff with sample quality when pushed too far.",
  },
  {
    id: "crash-probability-l5-q56",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "A Variational Autoencoder (VAE) uses a reparameterization \\(z=\\mu(x)+\\sigma(x)\\odot\\epsilon\\), where \\(\\epsilon\\sim\\mathcal{N}(0,I)\\). Which statements are correct?",
    options: [
      {
        text: "Conditioned on \\(x\\), this represents a Gaussian latent variable with mean \\(\\mu(x)\\) and diagonal standard deviations \\(\\sigma(x)\\).",
        isCorrect: true,
      },
      {
        text: "The randomness is isolated in \\(\\epsilon\\), which helps gradients flow through \\(\\mu(x)\\) and \\(\\sigma(x)\\).",
        isCorrect: true,
      },
      {
        text: "If \\(\\sigma(x)=0\\), then \\(z=\\mu(x)\\) becomes deterministic for that input.",
        isCorrect: true,
      },
      {
        text: "\\(\\epsilon\\) must be sampled from the decoder output distribution \\(P(x\\mid z)\\), not from a standard normal.",
        isCorrect: false,
      },
    ],
    explanation:
      "The reparameterization trick rewrites latent sampling as a deterministic transformation of parameters and independent standard Gaussian noise. This keeps the generative role of noise while making optimization through the latent parameters more tractable.",
  },
  {
    id: "crash-probability-l5-q57",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly reason about a latent prior \\(z\\sim P(z)\\), latent space, and decoder distribution \\(P(x\\mid z)\\)?",
    options: [
      {
        text: "The prior \\(P(z)\\) defines where latent samples are drawn before decoding.",
        isCorrect: true,
      },
      {
        text: "Interpolating between two latent vectors explores a path in latent space, but it is not the same operation as drawing an independent sample from the prior.",
        isCorrect: true,
      },
      {
        text: "Nearby latent points may decode to similar outputs in a smooth learned model, but the notation \\(z\\sim P(z)\\) alone does not guarantee semantic smoothness.",
        isCorrect: true,
      },
      {
        text: "A decoder distribution \\(P(x\\mid z)\\) can remain stochastic even after a particular \\(z\\) is fixed.",
        isCorrect: true,
      },
    ],
    explanation:
      "Latent modeling separates the prior over hidden variables from the decoder distribution over visible data. Smooth latent spaces are a learned modeling property, not a consequence of notation alone, and a conditional decoder can still represent uncertainty after the latent value is chosen.",
  },
  {
    id: "crash-probability-l5-q58",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "A prompt condition \\(c\\) leaves image style ambiguous. Let \\(P(Z=\\text{modern}\\mid c)=0.6\\), \\(P(Z=\\text{rustic}\\mid c)=0.4\\), \\(P(R\\mid Z=\\text{modern},c)=0.2\\), and \\(P(R\\mid Z=\\text{rustic},c)=0.7\\), where \\(R\\) is the event that the generated image has a red roof. Which statements are correct?",
    options: [
      {
        text: "\\(P(R\\mid c)=0.6\\cdot0.2+0.4\\cdot0.7=0.40\\).",
        isCorrect: true,
      },
      {
        text: "\\(P(Z=\\text{rustic}\\mid R,c)=0.28/0.40=0.70\\).",
        isCorrect: true,
      },
      {
        text: "The prompt condition does not uniquely determine the image because latent or unspecified factors can remain uncertain.",
        isCorrect: true,
      },
      {
        text: "\\(P(R\\mid c)=0.2+0.7=0.9\\), because conditional probabilities across latent styles should be added without weighting.",
        isCorrect: false,
      },
    ],
    explanation:
      "This is the law of total probability and Bayes' rule applied inside a conditional generative setting. The prompt narrows the distribution, but hidden style variables or unspecified details can still affect the probability of visible image properties.",
  },
  {
    id: "crash-probability-l5-q59",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly compare probability factorizations for language models, diffusion models, and reinforcement learning (RL)?",
    options: [
      {
        text: "An autoregressive language model can factor a sequence as \\(P(x_{1:T})=\\prod_{t=1}^T P(x_t\\mid x_{1:t-1})\\).",
        isCorrect: true,
      },
      {
        text: "A reverse diffusion sampler can be written schematically as starting from \\(p(x_T)\\) and applying transitions such as \\(\\prod_t p_\\theta(x_{t-1}\\mid x_t,c)\\).",
        isCorrect: true,
      },
      {
        text: "An RL trajectory probability can involve policy terms such as \\(\\pi(a_t\\mid s_t)\\) and environment transition terms such as \\(P(s_{t+1}\\mid s_t,a_t)\\).",
        isCorrect: true,
      },
      {
        text: "All three settings use conditional distributions to move from current context or state toward a next sampled object.",
        isCorrect: true,
      },
    ],
    explanation:
      "The mathematical objects differ, but each setting uses a product or repetition of conditional distributions. This shared structure is why next-token generation, denoising, and sequential action selection can all be discussed as probabilistic step-by-step processes.",
  },
  {
    id: "crash-probability-l5-q60",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "A generated three-token sequence has conditional probabilities \\(0.5\\), \\(0.2\\), and \\(0.4\\) for the tokens that were actually selected. Which statements are correct?",
    options: [
      {
        text: "The sequence probability under the autoregressive model is \\(0.5\\cdot0.2\\cdot0.4=0.04\\).",
        isCorrect: true,
      },
      {
        text: "The negative log-likelihood is \\(-\\log(0.04)\\), using the same log base consistently.",
        isCorrect: true,
      },
      {
        text: "The average per-token negative log-likelihood is \\(-\\frac{1}{3}\\log(0.04)\\).",
        isCorrect: true,
      },
      {
        text: "Increasing the model probability assigned to the selected tokens would lower the negative log-likelihood for this fixed sequence.",
        isCorrect: true,
      },
    ],
    explanation:
      "Autoregressive sequence probability multiplies the conditional probabilities of the realized tokens. Negative log-likelihood turns that product into a sum of token-level losses, so assigning more probability to the observed sequence lowers the loss.",
  },
];
