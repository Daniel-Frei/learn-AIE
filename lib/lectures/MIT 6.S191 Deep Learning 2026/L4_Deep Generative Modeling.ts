import { Question } from "../../quiz";

export const MIT6S191_L4_DeepGenerativeModelingQuestions: Question[] = [
  {
    id: "mit6s191-l4-q01",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements correctly describe generative modeling?",
    options: [
      {
        text: "Generative models aim to learn the underlying data distribution \\(p_{model}(x)\\).",
        isCorrect: true,
      },
      {
        text: "Once trained, generative models can sample new data instances from the learned distribution.",
        isCorrect: true,
      },
      {
        text: "Generative modeling can be framed as density estimation.",
        isCorrect: true,
      },
      {
        text: "Learning \\(p_{model}(x)\\) allows comparison to \\(p_{data}(x)\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "Generative modeling focuses on learning a probability distribution over data. Once this distribution is learned, we can sample from it to generate new data instances. This process is often framed as density estimation, where the goal is to approximate the true data distribution.",
  },

  {
    id: "mit6s191-l4-q02",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements correctly distinguish supervised from unsupervised learning?",
    options: [
      {
        text: "Supervised learning uses labeled pairs \\((x, y)\\).",
        isCorrect: true,
      },
      {
        text: "Unsupervised learning attempts to uncover hidden structure in data \\(x\\).",
        isCorrect: true,
      },
      {
        text: "Generative modeling is typically framed as unsupervised learning.",
        isCorrect: true,
      },
      {
        text: "Unsupervised learning does not require ground-truth labels.",
        isCorrect: true,
      },
    ],
    explanation:
      "Supervised learning relies on labeled data pairs. Unsupervised learning operates on unlabeled data and seeks hidden patterns or structure. Generative modeling typically falls under unsupervised learning because it models \\(p(x)\\) without labels.",
  },

  {
    id: "mit6s191-l4-q03",
    chapter: 4,
    difficulty: "medium",
    prompt: "What is a latent variable in the context of generative modeling?",
    options: [
      {
        text: "A hidden explanatory factor underlying observed data.",
        isCorrect: true,
      },
      {
        text: "A latent variable is not directly observed in the dataset.",
        isCorrect: true,
      },
      {
        text: "A variable that governs the distribution of observed samples.",
        isCorrect: true,
      },
      {
        text: "A latent variable is not simply a supervised label provided during training.",
        isCorrect: true,
      },
    ],
    explanation:
      "Latent variables are hidden factors that explain observed data. They are not directly observed and are not the same as labels. In latent variable models, the observed data is assumed to be generated from these hidden variables.",
  },

  {
    id: "mit6s191-l4-q04",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "In an autoencoder, what is the purpose of the bottleneck latent space \\(z\\)?",
    options: [
      {
        text: "To compress the data into a lower-dimensional representation.",
        isCorrect: true,
      },
      {
        text: "To force the model to reconstruct the input from limited information.",
        isCorrect: true,
      },
      {
        text: "Autoencoders do not guarantee perfect reconstruction.",
        isCorrect: true,
      },
      {
        text: "Autoencoders do not eliminate the need for a decoder network.",
        isCorrect: true,
      },
    ],
    explanation:
      "The bottleneck enforces compression by restricting dimensionality. This forces the encoder to retain essential information. However, it does not guarantee perfect reconstruction and still requires a decoder.",
  },

  {
    id: "mit6s191-l4-q05",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which statements about reconstruction loss \\(\\mathcal{L}(x, \\hat{x}) = \\|x - \\hat{x}\\|^2\\) are correct?",
    options: [
      {
        text: "It ignores the relationship between the input and its reconstruction.",
        isCorrect: false,
      },
      { text: "It can be used without labeled data.", isCorrect: true },
      {
        text: "It encourages the latent space to encode useful information.",
        isCorrect: true,
      },
      {
        text: "It directly supervises the latent variables with labels.",
        isCorrect: false,
      },
    ],
    explanation:
      "Reconstruction loss measures how close the reconstruction \\(\\hat{x}\\) is to the input \\(x\\). It does not require labels, making autoencoders unsupervised. The loss encourages latent variables to encode meaningful information.",
  },

  {
    id: "mit6s191-l4-q06",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Why does a smaller latent dimension typically reduce reconstruction quality?",
    options: [
      {
        text: "It increases compression and limits representational capacity.",
        isCorrect: true,
      },
      {
        text: "It introduces a stronger information bottleneck.",
        isCorrect: true,
      },
      { text: "It always improves reconstruction accuracy.", isCorrect: false },
      {
        text: "It never prevents the model from encoding fine details.",
        isCorrect: false,
      },
    ],
    explanation:
      "A smaller latent space restricts how much information can pass through. This compression may discard fine details and degrade reconstruction quality. Therefore, there is a tradeoff between compression and fidelity.",
  },

  {
    id: "mit6s191-l4-q07",
    chapter: 4,
    difficulty: "medium",
    prompt: "In a Variational Autoencoder (VAE), the encoder learns:",
    options: [
      { text: "A deterministic latent vector only.", isCorrect: false },
      {
        text: "Parameters \\(\\mu\\) and \\(\\sigma\\) of a latent distribution.",
        isCorrect: true,
      },
      {
        text: "A probability distribution \\(q_\\phi(z|x)\\).",
        isCorrect: true,
      },
      { text: "A classifier for real vs fake samples.", isCorrect: false },
    ],
    explanation:
      "VAEs learn a probabilistic latent representation. The encoder outputs parameters (mean and standard deviation) defining a distribution \\(q_\\phi(z|x)\\). This distinguishes VAEs from deterministic autoencoders.",
  },

  {
    id: "mit6s191-l4-q08",
    chapter: 4,
    difficulty: "hard",
    prompt: "The VAE loss function consists of which components?",
    options: [
      { text: "A reconstruction term.", isCorrect: true },
      {
        text: "A regularization term based on KL divergence.",
        isCorrect: true,
      },
      { text: "A classification loss.", isCorrect: false },
      {
        text: "A term encouraging closeness to a prior distribution.",
        isCorrect: true,
      },
    ],
    explanation:
      "The VAE objective includes reconstruction loss and a regularization term using KL divergence. The regularizer encourages the approximate posterior to match a chosen prior. There is no classification component in the standard VAE loss.",
  },

  {
    id: "mit6s191-l4-q09",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements about KL divergence \\(D_{KL}(q(z|x) \\| p(z))\\) are correct?",
    options: [
      {
        text: "It measures similarity between two probability distributions.",
        isCorrect: true,
      },
      { text: "It is always symmetric in its arguments.", isCorrect: false },
      {
        text: "It does not regularize latent distributions toward a prior.",
        isCorrect: false,
      },
      {
        text: "It ensures continuity and completeness in the latent space.",
        isCorrect: true,
      },
    ],
    explanation:
      "KL divergence measures how one distribution differs from another. It is not symmetric. In VAEs, it encourages the approximate posterior to align with a prior, supporting smoothness and coverage of the latent space.",
  },

  {
    id: "mit6s191-l4-q10",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Why is a normal prior \\(p(z) = \\mathcal{N}(0, I)\\) commonly used in VAEs?",
    options: [
      {
        text: "It encourages smooth coverage of the latent space.",
        isCorrect: true,
      },
      { text: "It simplifies KL divergence computation.", isCorrect: true },
      {
        text: "It guarantees disentangled latent variables.",
        isCorrect: false,
      },
      {
        text: "It pushes latent representations arbitrarily far from zero.",
        isCorrect: false,
      },
    ],
    explanation:
      "A standard normal prior simplifies mathematical computation of KL divergence. It encourages smooth and centered latent representations. However, it does not automatically guarantee disentanglement.",
  },

  {
    id: "mit6s191-l4-q11",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Why is direct sampling \\(z \\sim \\mathcal{N}(\\mu, \\sigma^2)\\) problematic for backpropagation?",
    options: [
      {
        text: "Sampling is fully differentiable and introduces no stochasticity.",
        isCorrect: false,
      },
      {
        text: "Gradients cannot pass through random nodes directly.",
        isCorrect: true,
      },
      {
        text: "Backpropagation requires deterministic operations.",
        isCorrect: true,
      },
      { text: "Sampling increases reconstruction loss.", isCorrect: false },
    ],
    explanation:
      "Direct stochastic sampling blocks gradient flow. Backpropagation requires differentiable operations. Therefore, special techniques are required to enable training of VAEs.",
  },

  {
    id: "mit6s191-l4-q12",
    chapter: 4,
    difficulty: "hard",
    prompt: "The reparameterization trick expresses latent variables as:",
    options: [
      {
        text: "\\(z = \\mu \\times \\sigma\\).",
        isCorrect: false,
      },
      {
        text: "A deterministic transformation of \\(\\mu, \\sigma, \\epsilon\\).",
        isCorrect: true,
      },
      {
        text: "A method allowing gradients to pass through \\(\\mu\\) and \\(\\sigma\\).",
        isCorrect: true,
      },
      { text: "\\(z = \\mu \\times \\sigma\\).", isCorrect: false },
    ],
    explanation:
      "The reparameterization trick isolates randomness in \\(\\epsilon\\), enabling gradients to flow through \\(\\mu\\) and \\(\\sigma\\). This makes end-to-end training feasible.",
  },

  {
    id: "mit6s191-l4-q13",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which properties are desirable for a VAE latent space?",
    options: [
      {
        text: "Continuity: nearby points decode to similar samples.",
        isCorrect: true,
      },
      {
        text: "Completeness: sampling anywhere yields meaningful outputs.",
        isCorrect: true,
      },
      { text: "Large empty gaps in latent space.", isCorrect: false },
      { text: "Discontinuity between similar data points.", isCorrect: false },
    ],
    explanation:
      "Continuity ensures smooth interpolation between data points. Completeness ensures meaningful decoding across latent space. Gaps and discontinuities reduce generative quality.",
  },

  {
    id: "mit6s191-l4-q14",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "In a Generative Adversarial Network (GAN), which networks are trained together?",
    options: [
      { text: "A generator network.", isCorrect: true },
      { text: "A discriminator network.", isCorrect: true },
      { text: "A classifier for labels.", isCorrect: false },
      { text: "An encoder network for reconstruction.", isCorrect: false },
    ],
    explanation:
      "GANs consist of two networks: a generator and a discriminator. The generator produces synthetic samples while the discriminator distinguishes real from fake.",
  },

  {
    id: "mit6s191-l4-q15",
    chapter: 4,
    difficulty: "medium",
    prompt: "What is the generator's objective in a GAN?",
    options: [
      {
        text: "To minimize the probability that generated samples are classified as fake.",
        isCorrect: true,
      },
      { text: "To maximize reconstruction accuracy.", isCorrect: false },
      {
        text: "To transform noise into realistic data samples.",
        isCorrect: true,
      },
      { text: "To maximize discriminator accuracy.", isCorrect: false },
    ],
    explanation:
      "The generator attempts to fool the discriminator by generating realistic samples. It does not perform reconstruction like a VAE. Instead, it transforms noise into samples resembling real data.",
  },

  {
    id: "mit6s191-l4-q16",
    chapter: 4,
    difficulty: "hard",
    prompt: "The GAN objective can be written as:",
    options: [
      {
        text: "GAN training is not based on a minimax objective.",
        isCorrect: false,
      },
      {
        text: "A minimax game between generator and discriminator.",
        isCorrect: true,
      },
      {
        text: "A purely supervised classification objective.",
        isCorrect: false,
      },
      {
        text: "Jointly optimized through competing gradients.",
        isCorrect: true,
      },
    ],
    explanation:
      "GANs are formulated as a minimax optimization problem. The discriminator maximizes correct classification of real and fake samples, while the generator minimizes the discriminator's ability to detect fakes.",
  },

  {
    id: "mit6s191-l4-q17",
    chapter: 4,
    difficulty: "medium",
    prompt: "Why can GAN training be unstable?",
    options: [
      {
        text: "The generator and discriminator objectives compete.",
        isCorrect: true,
      },
      {
        text: "The loss landscape cannot lead to oscillations.",
        isCorrect: false,
      },
      {
        text: "The discriminator can never overpower the generator.",
        isCorrect: false,
      },
      { text: "GANs have no differentiable components.", isCorrect: false },
    ],
    explanation:
      "GAN training is a minimax game, which can lead to instability and oscillations. If the discriminator becomes too strong, gradients for the generator may vanish. However, GANs are differentiable in principle.",
  },

  {
    id: "mit6s191-l4-q18",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements describe sampling in generative models?",
    options: [
      {
        text: "Sampling from \\(p_{model}(x)\\) does not produce new data instances.",
        isCorrect: false,
      },
      {
        text: "In GANs, sampling never begins from random noise.",
        isCorrect: false,
      },
      {
        text: "In VAEs, sampling occurs in latent space before decoding.",
        isCorrect: true,
      },
      { text: "Sampling requires labeled data.", isCorrect: false },
    ],
    explanation:
      "Generative models allow sampling from learned distributions. GANs sample from noise and transform it, while VAEs sample from latent space distributions. Labels are not required for sampling.",
  },

  {
    id: "mit6s191-l4-q19",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "In a trained GAN, interpolating between two noise vectors typically:",
    options: [
      {
        text: "Produces smooth transitions in generated outputs.",
        isCorrect: true,
      },
      {
        text: "Indicates the generator has not learned a continuous mapping.",
        isCorrect: false,
      },
      { text: "Always results in identical outputs.", isCorrect: false },
      {
        text: "Suggests there is no structure in the learned data manifold.",
        isCorrect: false,
      },
    ],
    explanation:
      "Smooth interpolation suggests that the generator has learned a structured and continuous mapping from noise space to data space. This indicates that the learned data manifold is coherent.",
  },

  {
    id: "mit6s191-l4-q20",
    chapter: 4,
    difficulty: "medium",
    prompt: "Compared to VAEs, GANs primarily focus on:",
    options: [
      { text: "High-fidelity sample generation.", isCorrect: true },
      { text: "Explicit density estimation via likelihood.", isCorrect: false },
      { text: "Adversarial training dynamics.", isCorrect: true },
      {
        text: "Learning interpretable latent variables by default.",
        isCorrect: false,
      },
    ],
    explanation:
      "GANs prioritize generating realistic samples through adversarial training. Unlike VAEs, they do not explicitly model likelihood or emphasize interpretable latent variables by default.",
  },

  {
    id: "mit6s191-l4-q21",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "In a Variational Autoencoder (VAE), the decoder models which probability distribution?",
    options: [
      {
        text: "\\(p_\\theta(x|z)\\), the likelihood of data given latent variables.",
        isCorrect: true,
      },
      {
        text: "\\(q_\\phi(z|x)\\), the decoder distribution in a VAE.",
        isCorrect: false,
      },
      {
        text: "A mapping from latent space back to the data space is not part of the decoder.",
        isCorrect: false,
      },
      {
        text: "A conditional generative distribution parameterized by \\(\\theta\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "In a VAE, the decoder models \\(p_\\theta(x|z)\\), the probability of the data given latent variables. The encoder models \\(q_\\phi(z|x)\\). The decoder acts as a conditional generative model mapping latent codes to data.",
  },

  {
    id: "mit6s191-l4-q22",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which properties of KL divergence \\(D_{KL}(q(z|x) \\| p(z))\\) are correct?",
    options: [
      { text: "It is always non-negative.", isCorrect: true },
      {
        text: "It can be negative for sufficiently similar distributions.",
        isCorrect: false,
      },
      {
        text: "It is symmetric: \\(D_{KL}(p\\|q) = D_{KL}(q\\|p)\\).",
        isCorrect: false,
      },
      {
        text: "It cannot be interpreted as information loss when approximating one distribution with another.",
        isCorrect: false,
      },
    ],
    explanation:
      "KL divergence is always non-negative and equals zero only when distributions match. It is not symmetric. It can be interpreted as the information lost when using one distribution to approximate another.",
  },

  {
    id: "mit6s191-l4-q23",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Why does regularizing the latent space improve sampling quality in VAEs?",
    options: [
      {
        text: "It encourages continuity so nearby latent points decode to similar outputs.",
        isCorrect: true,
      },
      {
        text: "It discourages coverage of the latent space.",
        isCorrect: false,
      },
      { text: "It guarantees perfect disentanglement.", isCorrect: false },
      {
        text: "It creates large empty gaps in latent space.",
        isCorrect: false,
      },
    ],
    explanation:
      "Regularization aligns latent encodings with a smooth prior distribution. This encourages continuity and coverage, making sampling meaningful across the space. It does not automatically guarantee disentangled representations.",
  },

  {
    id: "mit6s191-l4-q24",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Consider the GAN minimax objective:\n\\[\n\\min_G \\max_D \\; \\mathbb{E}_{x \\sim p_{data}}[\\log D(x)] + \\mathbb{E}_{z \\sim p(z)}[\\log(1 - D(G(z)))]\n\\]\nWhich statements are correct?",
    options: [
      {
        text: "The discriminator maximizes correct classification of real and fake samples.",
        isCorrect: true,
      },
      {
        text: "The generator maximizes the discriminator's ability to detect fake samples.",
        isCorrect: false,
      },
      { text: "The objective is not a two-player game.", isCorrect: false },
      {
        text: "Both networks minimize the same objective simultaneously.",
        isCorrect: false,
      },
    ],
    explanation:
      "GAN training is a minimax game between generator and discriminator. The discriminator maximizes classification accuracy, while the generator minimizes its detectability. They optimize opposing objectives, not identical ones.",
  },

  {
    id: "mit6s191-l4-q25",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which are valid applications of generative models?",
    options: [
      {
        text: "Outlier detection in autonomous driving scenarios.",
        isCorrect: true,
      },
      {
        text: "Debiasing datasets by uncovering underrepresented features is impossible with generative models.",
        isCorrect: false,
      },
      {
        text: "Generating synthetic images, text, or audio is not a valid use of generative models.",
        isCorrect: false,
      },
      { text: "Replacing all supervised learning tasks.", isCorrect: false },
    ],
    explanation:
      "Generative models can detect rare events, uncover biases, and synthesize new samples. However, they do not replace supervised learning entirely; the two paradigms serve different purposes.",
  },

  {
    id: "mit6s191-l4-q26",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about the reparameterization trick are correct?",
    options: [
      {
        text: "It separates deterministic parameters from stochastic noise.",
        isCorrect: true,
      },
      {
        text: "It allows gradients to flow through \\(\\mu\\) and \\(\\sigma\\).",
        isCorrect: true,
      },
      {
        text: "It does not remove all randomness from VAEs.",
        isCorrect: true,
      },
      {
        text: "It rewrites \\(z \\sim \\mathcal{N}(\\mu, \\sigma^2)\\) as \\(z = \\mu + \\sigma \\epsilon\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The reparameterization trick isolates randomness in \\(\\epsilon\\), allowing gradients to pass through \\(\\mu\\) and \\(\\sigma\\). It does not remove randomness but restructures it for differentiability.",
  },

  {
    id: "mit6s191-l4-q27",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "From a distribution transformation perspective, GANs can be viewed as:",
    options: [
      {
        text: "Learning a mapping from noise distribution \\(p(z)\\) to data distribution \\(p_{data}(x)\\).",
        isCorrect: true,
      },
      {
        text: "Transforming one data manifold into another (e.g., CycleGAN).",
        isCorrect: true,
      },
      {
        text: "GANs do not perform explicit maximum likelihood estimation of \\(p(x)\\).",
        isCorrect: true,
      },
      { text: "Learning a continuous function approximator.", isCorrect: true },
    ],
    explanation:
      "GANs learn a mapping from a simple noise distribution to a complex data distribution. Extensions like CycleGAN map between two data manifolds. GANs do not explicitly maximize likelihood.",
  },

  {
    id: "mit6s191-l4-q28",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "In a VAE, the full objective (Evidence Lower Bound) can be written as:\n\\[\n\\mathcal{L} = \\mathbb{E}_{q_\\phi(z|x)}[\\log p_\\theta(x|z)] - D_{KL}(q_\\phi(z|x) \\| p(z))\n\\]\nWhich interpretations are correct?",
    options: [
      {
        text: "The first term encourages accurate reconstruction.",
        isCorrect: true,
      },
      {
        text: "The second term regularizes the approximate posterior toward the prior.",
        isCorrect: true,
      },
      {
        text: "Maximizing this objective approximates maximizing data likelihood.",
        isCorrect: true,
      },
      {
        text: "The objective contains no probabilistic interpretation.",
        isCorrect: false,
      },
    ],
    explanation:
      "The ELBO consists of a reconstruction term and a KL regularization term. Maximizing it corresponds to approximating maximum likelihood training under a latent variable model. It has a clear probabilistic interpretation.",
  },

  {
    id: "mit6s191-l4-q29",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements describe latent space interpolation?",
    options: [
      {
        text: "Interpolating between latent codes can produce smooth semantic transitions.",
        isCorrect: true,
      },
      {
        text: "It indicates structure in the learned manifold.",
        isCorrect: true,
      },
      {
        text: "It proves that the model memorized training data.",
        isCorrect: false,
      },
      {
        text: "It reflects continuity of the latent representation.",
        isCorrect: true,
      },
    ],
    explanation:
      "Smooth interpolation suggests the model has learned a continuous manifold. It demonstrates structure rather than simple memorization. Continuity in latent space leads to smooth transitions.",
  },

  {
    id: "mit6s191-l4-q30",
    chapter: 4,
    difficulty: "medium",
    prompt: "CycleGANs differ from vanilla GANs because they:",
    options: [
      { text: "Learn mappings between two data domains.", isCorrect: true },
      { text: "Use a cyclic consistency loss.", isCorrect: true },
      { text: "Require paired training data.", isCorrect: false },
      {
        text: "Employ two generators and two discriminators.",
        isCorrect: true,
      },
    ],
    explanation:
      "CycleGANs perform domain translation between two manifolds. They use cyclic consistency to ensure reversibility. They typically work with unpaired data and employ two generators and discriminators.",
  },

  {
    id: "mit6s191-l4-q31",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "If the discriminator in a GAN becomes too strong early in training:",
    options: [
      { text: "Generator gradients may vanish.", isCorrect: true },
      { text: "Training may become unstable.", isCorrect: true },
      {
        text: "The generator receives little learning signal.",
        isCorrect: true,
      },
      {
        text: "The model converges immediately to optimal equilibrium.",
        isCorrect: false,
      },
    ],
    explanation:
      "If the discriminator perfectly separates real and fake, generator gradients may vanish. This leads to instability and stalled learning. Convergence to equilibrium is not guaranteed.",
  },

  {
    id: "mit6s191-l4-q32",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which differences between VAEs and GANs are correct?",
    options: [
      {
        text: "VAEs explicitly model likelihood via reconstruction.",
        isCorrect: true,
      },
      { text: "GANs rely on adversarial training.", isCorrect: true },
      {
        text: "VAEs provide an interpretable latent distribution by design.",
        isCorrect: true,
      },
      {
        text: "GANs optimize a KL divergence term explicitly.",
        isCorrect: false,
      },
    ],
    explanation:
      "VAEs maximize a probabilistic objective including KL divergence. GANs use adversarial training instead of explicit likelihood optimization. VAEs tend to provide structured latent spaces.",
  },

  {
    id: "mit6s191-l4-q33",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which statements describe density estimation?",
    options: [
      { text: "It involves modeling \\(p_{model}(x)\\).", isCorrect: true },
      {
        text: "It allows evaluation of likelihood of new samples.",
        isCorrect: true,
      },
      { text: "It is central to many generative approaches.", isCorrect: true },
      { text: "It requires labeled data.", isCorrect: false },
    ],
    explanation:
      "Density estimation models the probability distribution over data. It enables likelihood evaluation and sampling. It does not require labels.",
  },

  {
    id: "mit6s191-l4-q34",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which conditions ideally hold at Nash equilibrium of a GAN?",
    options: [
      {
        text: "The generator distribution matches the data distribution.",
        isCorrect: true,
      },
      {
        text: "The discriminator cannot distinguish real from fake better than chance.",
        isCorrect: true,
      },
      {
        text: "The discriminator outputs approximately 0.5 for all samples.",
        isCorrect: true,
      },
      { text: "The generator collapses to a single sample.", isCorrect: false },
    ],
    explanation:
      "At equilibrium, the generator matches the data distribution. The discriminator cannot distinguish real from fake and outputs roughly 0.5. Mode collapse is a failure case, not equilibrium.",
  },

  {
    id: "mit6s191-l4-q35",
    chapter: 4,
    difficulty: "medium",
    prompt: "Why can generative models help with debiasing datasets?",
    options: [
      {
        text: "They can uncover latent features correlated with bias.",
        isCorrect: true,
      },
      {
        text: "They can reveal over- or under-represented attributes.",
        isCorrect: true,
      },
      {
        text: "They automatically remove all societal biases.",
        isCorrect: false,
      },
      {
        text: "They can guide rebalancing of data distributions.",
        isCorrect: true,
      },
    ],
    explanation:
      "By learning latent structure, generative models can expose imbalances in representation. This insight can guide dataset rebalancing. However, they do not automatically eliminate bias.",
  },

  {
    id: "mit6s191-l4-q36",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which statements about sampling from a trained VAE are correct?",
    options: [
      {
        text: "We can sample \\(z \\sim p(z)\\) and decode to generate new samples.",
        isCorrect: true,
      },
      {
        text: "Sampling from regions far outside the prior may produce unrealistic outputs.",
        isCorrect: true,
      },
      {
        text: "Decoding random latent vectors produces identical outputs.",
        isCorrect: false,
      },
      {
        text: "The prior regularization supports meaningful generation.",
        isCorrect: true,
      },
    ],
    explanation:
      "After training, sampling from the prior and decoding generates new data. Regions outside the learned support may produce unrealistic outputs. Regularization helps ensure meaningful coverage.",
  },

  {
    id: "mit6s191-l4-q37",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which characteristics are shared by VAEs and GANs?",
    options: [
      { text: "Both are generative models.", isCorrect: true },
      { text: "Both can generate new synthetic samples.", isCorrect: true },
      { text: "Both rely on neural networks.", isCorrect: true },
      { text: "Both explicitly compute exact likelihoods.", isCorrect: false },
    ],
    explanation:
      "VAEs and GANs are both neural generative models capable of producing new samples. However, GANs do not explicitly compute likelihoods.",
  },

  {
    id: "mit6s191-l4-q38",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements describe outlier detection using generative models?",
    options: [
      {
        text: "Outliers lie in low-probability regions of \\(p_{model}(x)\\).",
        isCorrect: true,
      },
      {
        text: "Generative models can estimate whether a sample is rare.",
        isCorrect: true,
      },
      {
        text: "Outlier detection is unrelated to density estimation.",
        isCorrect: false,
      },
      {
        text: "Outlier detection can improve safety in autonomous systems.",
        isCorrect: true,
      },
    ],
    explanation:
      "Outliers correspond to low-density regions under the learned distribution. Generative models help identify rare cases, improving robustness in safety-critical applications.",
  },

  {
    id: "mit6s191-l4-q39",
    chapter: 4,
    difficulty: "hard",
    prompt: "Why can GAN interpolation produce smooth semantic transitions?",
    options: [
      {
        text: "The generator learns a continuous mapping from noise to data space.",
        isCorrect: true,
      },
      { text: "The learned manifold structure is smooth.", isCorrect: true },
      {
        text: "Interpolation in noise space maps to interpolation in data space.",
        isCorrect: true,
      },
      {
        text: "Noise vectors correspond to discrete lookup indices.",
        isCorrect: false,
      },
    ],
    explanation:
      "GAN generators implement continuous functions from noise space to data space. Interpolating in noise space often yields smooth semantic changes in generated outputs, reflecting learned manifold structure.",
  },

  {
    id: "mit6s191-l4-q40",
    chapter: 4,
    difficulty: "medium",
    prompt: "Which tradeoffs commonly arise when comparing VAEs and GANs?",
    options: [
      {
        text: "VAEs often produce blurrier samples but stable training.",
        isCorrect: true,
      },
      {
        text: "GANs often produce sharper samples but can be unstable.",
        isCorrect: true,
      },
      {
        text: "VAEs completely avoid probabilistic modeling.",
        isCorrect: false,
      },
      {
        text: "GANs explicitly regularize latent variables with KL divergence.",
        isCorrect: false,
      },
    ],
    explanation:
      "VAEs optimize a probabilistic objective and tend to produce smoother, sometimes blurrier outputs. GANs can generate sharper images but are harder to train. GANs do not use KL regularization.",
  },
];
