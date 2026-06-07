# Lecture 5 — Sampling, Latent Variables, and Diffusion Models

**Theme:** Modern generative AI is probability plus sampling.

**Duration:** 60 minutes
**Level:** After Lectures 1–4
**Style:** Mathematical but applied
**Core AI connection:** Generative AI systems do not merely predict labels; they sample outputs from learned probability distributions.

---

# Lecture 5 Overview

## Central Message

In the previous lectures, students learned that AI uses probability for:

[
P(y \mid x)
]

prediction,

[
-\log P(\text{correct output})
]

training,

and

[
\mathbb{E}[G_t]
]

decision-making.

Lecture 5 completes the picture by showing how probability becomes **generation**.

Modern generative AI often works like this:

[
\text{learn a distribution}
\rightarrow
\text{sample from it}
\rightarrow
\text{produce text, image, action, or data}
]

The major examples are:

| Model Type                  | What It Samples               |
| --------------------------- | ----------------------------- |
| LLM                         | next tokens                   |
| Diffusion model             | image through denoising steps |
| VAE / latent variable model | hidden latent variables       |
| RL policy                   | actions                       |
| Generative classifier       | labels or outputs             |

The key idea:

> A generative model does not store one answer. It learns a probability structure from which many possible outputs can be sampled.

---

# Lecture Structure

| Part |                               Topic |   Time |
| ---- | ----------------------------------: | -----: |
| 1    |                   What is sampling? |  8 min |
| 2    |         Greedy decoding vs sampling |  8 min |
| 3    |          Temperature and randomness |  8 min |
| 4    |                    Latent variables |  8 min |
| 5    |    Gaussian distributions and noise |  8 min |
| 6    |         Diffusion models: core idea | 10 min |
| 7    |      Why diffusion is probabilistic |  4 min |
| 8    | Connecting diffusion to LLMs and RL |  4 min |
| 9    |                Final course summary |  2 min |

---

# Learning Goals

By the end of the lecture, students should understand:

- what sampling from a probability distribution means,
- why generative AI can produce different outputs from the same input,
- the difference between greedy decoding and probabilistic sampling,
- what temperature does to a probability distribution,
- what latent variables are,
- why Gaussian noise is central in many generative models,
- how diffusion models use forward noising and reverse denoising,
- why diffusion models are probabilistic,
- how LLMs, diffusion models, and RL policies share common probabilistic ideas.

---

# Running Examples for the Lecture

Use three examples throughout.

## Running Example 1: LLM Next-Token Sampling

Prompt:

> The animal sat on the ...

Possible next-token distribution:

| Token | Probability |
| ----- | ----------: |
| mat   |        0.50 |
| sofa  |        0.25 |
| floor |        0.15 |
| chair |        0.07 |
| car   |        0.03 |

Use this for:

- sampling,
- greedy decoding,
- temperature,
- entropy,
- randomness.

---

## Running Example 2: Latent Style Variable

Suppose we generate an image of a house.

Hidden variables might include:

- style: modern, rustic, gothic, alpine,
- lighting: morning, sunset, night,
- season: summer, winter,
- viewpoint: front, aerial, interior.

Use this for latent variables.

---

## Running Example 3: Diffusion Image Generation

Generation starts from random noise:

[
x_T
]

Then a neural network gradually denoises:

[
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_0
]

Use this for Gaussian noise, denoising, and probabilistic generation.

---

# Part 1 — What Is Sampling?

**Time:** 8 minutes

## 1.1 Recap: A Distribution Gives Possibilities

From Lecture 1, a probability distribution assigns probabilities to possible outcomes.

Example:

| Token | Probability |
| ----- | ----------: |
| cat   |        0.50 |
| dog   |        0.30 |
| car   |        0.20 |

This does not yet choose an output.

It only says how likely each output is.

---

## 1.2 Sampling Definition

Sampling means:

> Randomly drawing one outcome according to the probability distribution.

So if we sample once from:

| Token | Probability |
| ----- | ----------: |
| cat   |        0.50 |
| dog   |        0.30 |
| car   |        0.20 |

we might get:

[
\text{cat}
]

or:

[
\text{dog}
]

or:

[
\text{car}
]

But over many samples, the frequencies should roughly match the probabilities.

If we sample 1,000 times, we expect approximately:

| Token | Expected frequency |
| ----- | -----------------: |
| cat   |                500 |
| dog   |                300 |
| car   |                200 |

Not exactly, but approximately.

---

## 1.3 Sampling Is Not Choosing the Maximum

This distinction is crucial.

Given:

| Token | Probability |
| ----- | ----------: |
| cat   |        0.50 |
| dog   |        0.30 |
| car   |        0.20 |

Choosing the maximum always gives:

[
\text{cat}
]

Sampling sometimes gives:

[
\text{dog}
]

or:

[
\text{car}
]

This is why sampling creates diversity.

---

## 1.4 LLM Connection

An LLM repeatedly predicts:

[
P(\text{next token} \mid \text{previous tokens})
]

Then the system must decide what to do with that distribution.

It can:

1. choose the most likely token,
2. sample from the distribution,
3. modify the distribution first, then sample.

Text generation is often:

[
\text{predict distribution}
\rightarrow
\text{sample token}
\rightarrow
\text{append token}
\rightarrow
\text{repeat}
]

Example:

Prompt:

> The cat sat on the

Model distribution:

| Token | Probability |
| ----- | ----------: |
| mat   |        0.50 |
| sofa  |        0.25 |
| floor |        0.15 |
| table |        0.07 |
| moon  |        0.03 |

If the system samples, different generations are possible:

- The cat sat on the mat.
- The cat sat on the sofa.
- The cat sat on the floor.
- The cat sat on the table.

Some are more likely than others.

---

## 1.5 Sampling in Other AI Systems

Sampling is not only for LLMs.

| AI Area           | What Is Sampled?                 |
| ----------------- | -------------------------------- |
| LLMs              | tokens                           |
| Diffusion models  | noise / denoising paths / images |
| RL                | actions                          |
| Bayesian models   | hypotheses or parameters         |
| VAEs              | latent variables                 |
| Simulation models | possible futures                 |

Core sentence:

> Sampling turns a probability distribution into an actual generated outcome.

---

## 1.6 Mini-Exercise

Given:

| Output | Probability |
| ------ | ----------: |
| A      |        0.70 |
| B      |        0.20 |
| C      |        0.10 |

Ask:

1. If we choose greedily, what output do we get?
2. If we sample, can we get B?
3. If we sample 1,000 times, roughly how often should C appear?
4. Why might sampling be useful in generative AI?

Expected answers:

1. A.
2. Yes.
3. About 100 times.
4. It creates diversity and allows multiple plausible outputs.

---

# Part 2 — Greedy Decoding vs Sampling

**Time:** 8 minutes

## 2.1 The Decoding Problem

An LLM outputs a probability distribution.

But users see a concrete text.

So the system needs a **decoding strategy**.

A decoding strategy answers:

> How do we turn the model’s probability distribution into actual tokens?

Two basic strategies:

1. greedy decoding,
2. sampling.

---

## 2.2 Greedy Decoding

Greedy decoding always chooses the most likely token.

Example:

| Token | Probability |
| ----- | ----------: |
| mat   |        0.50 |
| sofa  |        0.25 |
| floor |        0.15 |
| chair |        0.07 |
| car   |        0.03 |

Greedy output:

[
\text{mat}
]

At every step, choose:

[
\arg\max_t P(t \mid \text{context})
]

Advantages:

- predictable,
- stable,
- simple,
- useful when one answer is clearly best.

Disadvantages:

- can be repetitive,
- can be dull,
- can get stuck in local patterns,
- may ignore plausible alternatives.

---

## 2.3 Sampling

Sampling chooses tokens according to their probabilities.

Same distribution:

| Token | Probability |
| ----- | ----------: |
| mat   |        0.50 |
| sofa  |        0.25 |
| floor |        0.15 |
| chair |        0.07 |
| car   |        0.03 |

Sampling might choose:

- mat,
- sofa,
- floor,
- chair,
- rarely car.

Advantages:

- more diverse,
- more creative,
- better for open-ended generation,
- can produce multiple alternatives.

Disadvantages:

- less predictable,
- can choose bad low-probability tokens,
- may hallucinate more,
- can become incoherent if randomness is too high.

---

## 2.4 Example: Same Prompt, Different Outputs

Prompt:

> Write a first sentence for a detective story.

Greedy-style output:

> The detective arrived at the crime scene just after midnight.

Sampling-style outputs:

> Rain slid down the windows as Inspector Vale opened the letter.

> The body was gone before anyone admitted it had been there.

> At 3:17 a.m., the city cameras all turned black.

Teaching point:

> Open-ended tasks often benefit from sampling because there are many good possible answers.

---

## 2.5 Greedy Is Not Always “More Correct”

Greedy decoding can be useful for factual or constrained tasks.

But it is not automatically more intelligent.

Why?

Because generation is sequential.

Choosing the locally most likely token at each step does not always produce the best full sequence.

This is similar to RL:

> A locally best action may not lead to the best long-term outcome.

That analogy can help connect Lecture 4 to Lecture 5.

---

## 2.6 Other Decoding Methods

Briefly mention, but do not go deep.

### Top-k sampling

Only sample from the top (k) most likely tokens.

Example:

If (k=3), sample only from the three most likely tokens.

### Top-p / nucleus sampling

Sample from the smallest set of tokens whose total probability exceeds (p).

Example:

If (p=0.9), sample from the most likely tokens that together cover 90% of probability mass.

These methods reduce the chance of sampling extremely unlikely tokens.

Teaching point:

> Practical LLM generation usually modifies or restricts sampling to balance diversity and reliability.

---

## 2.7 Mini-Exercise

Given:

| Token  | Probability |
| ------ | ----------: |
| Paris  |        0.60 |
| Lyon   |        0.15 |
| London |        0.10 |
| the    |        0.08 |
| banana |        0.07 |

Questions:

1. What does greedy decoding choose?
2. Can sampling choose “Lyon”?
3. Why might unrestricted sampling choose bad tokens?
4. Why might top-k or top-p help?

Expected answers:

1. Paris.
2. Yes.
3. Low-probability but inappropriate tokens may be selected.
4. They restrict sampling to more plausible tokens.

---

# Part 3 — Temperature and Randomness

**Time:** 8 minutes

## 3.1 What Temperature Does

Temperature changes how sharp or flat the probability distribution is before sampling.

Informally:

- low temperature makes the model more deterministic,
- high temperature makes the model more random.

The model has logits (z_i).

Temperature modifies softmax:

[
P(y_i) =
\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
]

Where:

[
T = \text{temperature}
]

---

## 3.2 Low Temperature

If:

[
T < 1
]

the distribution becomes sharper.

High-probability tokens become even more dominant.

Example:

Original:

| Token | Probability |
| ----- | ----------: |
| mat   |        0.50 |
| sofa  |        0.25 |
| floor |        0.15 |
| chair |        0.10 |

Low temperature might become:

| Token | Probability |
| ----- | ----------: |
| mat   |        0.80 |
| sofa  |        0.12 |
| floor |        0.06 |
| chair |        0.02 |

Effect:

- less randomness,
- more predictable,
- safer,
- less creative.

---

## 3.3 High Temperature

If:

[
T > 1
]

the distribution becomes flatter.

Lower-probability tokens get more chance.

Original:

| Token | Probability |
| ----- | ----------: |
| mat   |        0.50 |
| sofa  |        0.25 |
| floor |        0.15 |
| chair |        0.10 |

High temperature might become:

| Token | Probability |
| ----- | ----------: |
| mat   |        0.35 |
| sofa  |        0.27 |
| floor |        0.22 |
| chair |        0.16 |

Effect:

- more randomness,
- more diversity,
- more creativity,
- more risk of incoherence or errors.

---

## 3.4 Temperature Does Not Make the Model Smarter

This is an important practical point.

Temperature does not add knowledge.

It does not improve reasoning.

It does not fix wrong beliefs.

It only changes how strongly the model follows its existing probability distribution.

Core sentence:

> Temperature changes how randomly we sample from what the model already believes.

If the model assigns high probability to a wrong answer, temperature does not necessarily solve that.

If the model assigns some probability to a creative continuation, temperature may make it more likely to appear.

---

## 3.5 Temperature and Entropy

From Lecture 3:

Entropy measures uncertainty in a distribution.

Temperature affects entropy.

Low temperature:

[
\text{lower entropy}
]

High temperature:

[
\text{higher entropy}
]

So temperature is one way to control output diversity.

---

## 3.6 Temperature in Different Use Cases

| Use Case                       | Better Temperature                                    |
| ------------------------------ | ----------------------------------------------------- |
| Factual QA                     | lower                                                 |
| Code generation                | lower to medium                                       |
| Brainstorming                  | medium to high                                        |
| Poetry/story ideas             | higher                                                |
| Legal/medical advice           | lower, but reliability requires more than temperature |
| Multiple creative alternatives | higher                                                |

Caveat:

> Lower temperature can make output more stable, but not automatically more truthful.

Truthfulness depends on model knowledge, retrieval, reasoning, prompting, and verification.

---

## 3.7 Mini-Exercise

A model gives:

| Token | Probability |
| ----- | ----------: |
| yes   |        0.55 |
| no    |        0.35 |
| maybe |        0.10 |

Ask:

1. What happens if temperature is lowered?
2. What happens if temperature is raised?
3. Does higher temperature make the model more knowledgeable?
4. Which setting is better for creative brainstorming?

Expected answers:

1. “yes” becomes more dominant.
2. The distribution becomes flatter; “no” and “maybe” become more likely.
3. No.
4. Usually higher or medium-high temperature.

---

# Part 4 — Latent Variables

**Time:** 8 minutes

## 4.1 What Is a Latent Variable?

A latent variable is a hidden variable that helps explain observed data.

“Latent” means hidden or not directly observed.

Notation:

[
Z
]

Observed data:

[
X
]

A simple generative structure:

[
Z \rightarrow X
]

Meaning:

> Hidden cause (Z) helps generate observed data (X).

---

## 4.2 Examples of Latent Variables

| Observed Data    | Possible Latent Variable         |
| ---------------- | -------------------------------- |
| document         | topic                            |
| sentence         | speaker intent                   |
| image            | style, object identity, lighting |
| patient symptoms | disease state                    |
| user behavior    | preference                       |
| generated face   | age, pose, expression            |
| RL observation   | hidden environment state         |

Example:

A document contains the words:

> stock, inflation, market, central bank

A possible latent variable is:

[
Z = \text{economics topic}
]

The topic is not directly printed as a label, but it helps explain the words.

---

## 4.3 Latent Variables in Generative Models

A generative model may work like this:

1. sample a hidden variable (z),
2. generate data (x) from that hidden variable.

[
z \sim P(z)
]

[
x \sim P(x \mid z)
]

In words:

> First sample a hidden cause, then sample the observed data conditioned on that hidden cause.

Example:

1. Sample style: “watercolor.”
2. Sample scene: “mountain village.”
3. Generate image.

---

## 4.4 Why Latent Variables Are Useful

Latent variables are useful because real data has hidden structure.

Images are not random pixels.

Texts are not random words.

User behavior is not random clicks.

Latent variables can represent:

- topic,
- style,
- intent,
- object identity,
- semantic meaning,
- compressed representation,
- hidden state.

Core sentence:

> Latent variables are a way to model hidden structure behind observed data.

---

## 4.5 Latent Space

Many AI systems learn a latent space.

A latent space is a compressed mathematical space where meaningful hidden features are represented.

Example:

In an image model, nearby points in latent space may correspond to visually similar images.

In a text model, embeddings represent semantic relationships.

Important distinction:

- latent variables in probabilistic models are explicitly sampled hidden variables,
- embeddings in neural networks are learned hidden representations.

They are related ideas, but not always the same thing.

For this lecture, the shared intuition is enough:

> AI often represents observed data through hidden internal variables.

---

## 4.6 VAE-Style Intuition

Briefly introduce Variational Autoencoders only as an example.

A VAE roughly works like this:

1. encode data (x) into a latent representation (z),
2. sample or regularize (z),
3. decode (z) back into data.

[
x \rightarrow z \rightarrow \hat{x}
]

Generative use:

[
z \sim P(z)
]

[
x \sim P(x \mid z)
]

Do not go into ELBO derivations.

Teaching point:

> Some generative models create data by sampling hidden latent variables and decoding them into visible outputs.

---

## 4.7 Mini-Exercise

For each observed data type, suggest a latent variable.

1. A movie review.
2. A face image.
3. A patient with symptoms.
4. A user clicking on products.
5. A generated landscape image.

Expected answers:

1. Sentiment, topic, writing style.
2. Identity, age, pose, lighting, expression.
3. Disease state, severity, risk factors.
4. Preferences, intent, budget.
5. Style, season, geography, lighting, composition.

---

# Part 5 — Gaussian Distributions and Noise

**Time:** 8 minutes

## 5.1 The Normal Distribution

Introduce the Gaussian / normal distribution:

[
X \sim \mathcal{N}(\mu, \sigma^2)
]

Where:

- (\mu) is the mean,
- (\sigma^2) is the variance,
- (\sigma) is the standard deviation.

Intuition:

- values near the mean are more likely,
- values far from the mean are less likely,
- larger variance means more spread.

---

## 5.2 Why Gaussian Noise Matters

Gaussian noise is central in many AI methods because it is:

- mathematically convenient,
- easy to sample,
- stable under many transformations,
- common in statistical modeling,
- useful for representing uncertainty.

Examples:

[
\epsilon \sim \mathcal{N}(0, I)
]

This means:

> sample random noise with mean 0 and standard Gaussian spread.

---

## 5.3 Noise in AI

Noise appears in many places:

| AI Context             | Role of Noise                                             |
| ---------------------- | --------------------------------------------------------- |
| Data                   | measurements are noisy                                    |
| Optimization           | stochastic gradient descent uses noisy gradient estimates |
| Regularization         | noise can prevent overfitting                             |
| Latent variable models | latent variables often use Gaussian priors                |
| Diffusion models       | data is gradually corrupted with Gaussian noise           |
| RL                     | exploration can involve random actions or noisy policies  |

Noise is not merely an error.

In generative AI, noise can be the starting material for creation.

---

## 5.4 Gaussian Noise in Images

An image can be represented as a high-dimensional vector.

For example, a (512 \times 512) RGB image has:

[
512 \times 512 \times 3
]

pixel values.

Gaussian noise means adding random values to many pixels.

A little noise:

> image still recognizable.

A lot of noise:

> image becomes almost pure static.

Diffusion models use this idea systematically.

---

## 5.5 Noise as Uncertainty

Noise can represent uncertainty.

If we know exactly what something is, uncertainty is low.

If many values are plausible, uncertainty is high.

In diffusion:

- early noising steps: low uncertainty,
- later noising steps: high uncertainty,
- final noisy state: almost no visible structure remains.

---

## 5.6 Mini-Exercise

Ask:

What happens when we increase the variance of Gaussian noise added to an image?

Expected answer:

The noise becomes stronger; the image becomes less recognizable; uncertainty about the original image increases.

Then ask:

Why might this be useful for a generative model?

Expected answer:

Because the model can learn how to reverse the noising process, which teaches it the structure of natural images.

---

# Part 6 — Diffusion Models: The Core Idea

**Time:** 10 minutes

## 6.1 The Big Picture

Diffusion models generate data by learning to reverse a noise process.

There are two processes:

1. forward process: add noise,
2. reverse process: remove noise.

The forward process is usually fixed.

The reverse process is learned.

---

## 6.2 Forward Process

Start with real data:

[
x_0
]

For example:

[
x_0 = \text{real image}
]

Gradually add noise:

[
x_0 \rightarrow x_1 \rightarrow x_2 \rightarrow \cdots \rightarrow x_T
]

At each step, the image becomes noisier.

Eventually:

[
x_T \approx \text{pure Gaussian noise}
]

The forward process is not generation yet.

It is corruption.

---

## 6.3 Intuitive Forward Example

Imagine a photo.

Step 0:

> clear image of a dog.

Step 1:

> tiny amount of static.

Step 10:

> image is noisy but recognizable.

Step 100:

> dog is barely visible.

Step 1000:

> almost pure noise.

The model learns from this artificial corruption process.

---

## 6.4 Reverse Process

The reverse process goes backward:

[
x_T \rightarrow x_{T-1} \rightarrow x_{T-2} \rightarrow \cdots \rightarrow x_0
]

Start from random noise.

At each step, the neural network predicts how to remove a little noise.

Eventually, the output becomes a clean generated sample.

Core sentence:

> A diffusion model generates by repeatedly denoising noise into data.

---

## 6.5 What Does the Network Learn?

At training time, we know:

- the original clean image (x_0),
- the noisy image (x_t),
- the noise that was added.

The neural network is trained to predict one of these:

- the noise (\epsilon),
- the clean image (x_0),
- the previous less-noisy image (x\_{t-1}),
- or a related quantity depending on the implementation.

Simplified version:

[
\text{network}(x_t, t) \approx \epsilon
]

Meaning:

> Given noisy image (x_t) and time step (t), predict the noise.

Then remove predicted noise step by step.

---

## 6.6 Conditioning on Prompts

Text-to-image diffusion models are conditional generative models.

They generate:

[
P(\text{image} \mid \text{text prompt})
]

Example:

Prompt:

> a red fox in a snowy forest

The model denoises in a way guided by the prompt.

So generation is not just:

[
P(x)
]

but:

[
P(x \mid c)
]

where (c) is the condition, such as text.

This connects back to Lecture 2.

---

## 6.7 Diffusion as Many Small Conditional Predictions

At each step, the model learns something like:

[
P(x_{t-1} \mid x_t, c)
]

or predicts noise:

[
P(\epsilon \mid x_t, t, c)
]

In words:

> Given the noisy image, time step, and condition, predict how to move toward a cleaner image.

This makes diffusion another example of conditional probability.

---

## 6.8 Why Many Steps?

Denoising all at once is hard.

Denoising a little at a time is easier.

Analogy:

It is hard to turn random static directly into a detailed image in one step.

It is easier to gradually refine:

[
\text{noise}
\rightarrow
\text{rough shapes}
\rightarrow
\text{objects}
\rightarrow
\text{details}
\rightarrow
\text{final image}
]

This stepwise structure is one reason diffusion models can generate high-quality images.

---

## 6.9 Mini-Exercise

Ask students to fill in the blanks:

1. The forward process gradually adds **\_\_**.
2. The reverse process gradually removes **\_\_**.
3. The model is trained on noisy versions of **\_\_**.
4. Text-to-image diffusion models estimate something like (P(\text{image} \mid \_\_\_\_)).

Expected answers:

1. noise.
2. noise.
3. real data/images.
4. text prompt.

---

# Part 7 — Why Diffusion Is Probabilistic

**Time:** 4 minutes

## 7.1 Sources of Probability in Diffusion

Diffusion models are probabilistic because they involve:

- random initial noise,
- random noising process during training,
- learned conditional distributions,
- stochastic or partially stochastic denoising steps,
- multiple possible outputs for the same prompt.

Same prompt:

> a small cabin near a lake at sunset

can produce many different images.

Why?

Because there are many valid images matching the prompt, and generation begins from random noise.

---

## 7.2 One Prompt, Many Possible Images

The prompt does not uniquely determine the image.

For example:

> a red house in the mountains

Many things are unspecified:

- exact house shape,
- mountain type,
- lighting,
- camera angle,
- weather,
- style,
- season,
- colors,
- surrounding objects.

The model samples one plausible version from a huge space of possible images.

---

## 7.3 Diffusion and Uncertainty

At the beginning:

[
x_T
]

is almost pure noise.

Many final images are possible.

As denoising proceeds, the sample becomes more committed to a particular image.

So uncertainty gradually collapses into structure.

Teaching sentence:

> Diffusion generation is the process of turning random uncertainty into structured data.

---

## 7.4 Mini-Exercise

Ask:

Why can the same prompt produce different images?

Expected answer:

Because generation starts from different random noise and the prompt allows many possible valid images.

---

# Part 8 — Connecting Diffusion to LLMs and RL

**Time:** 4 minutes

## 8.1 LLMs vs Diffusion Models

| Feature      | LLMs                                | Diffusion Models                              |
| ------------ | ----------------------------------- | --------------------------------------------- |
| Data type    | discrete tokens                     | continuous images/audio/video representations |
| Generation   | one token at a time                 | one denoising step at a time                  |
| Distribution | next-token distribution             | denoising / image distribution                |
| Training     | cross-entropy next-token prediction | denoising/noise prediction objective          |
| Randomness   | token sampling                      | initial noise and denoising path              |
| Conditioning | previous tokens / prompt            | text prompt / image / other condition         |

Both are generative.

Both rely on probability.

But they generate differently.

---

## 8.2 LLMs and Sampling

LLM generation:

[
P(x_t \mid x_1, ..., x_{t-1})
]

At each step:

1. predict next-token distribution,
2. sample or choose token,
3. append token,
4. repeat.

---

## 8.3 Diffusion and Sampling

Diffusion generation:

[
x_T \sim \mathcal{N}(0,I)
]

Then:

[
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_0
]

At each step:

1. take noisy sample,
2. predict noise / denoising direction,
3. update sample,
4. repeat.

---

## 8.4 RL and Sampling

RL policy:

[
\pi(a \mid s)
]

At each step:

1. observe state,
2. sample or choose action,
3. receive reward,
4. transition to next state,
5. repeat.

---

## 8.5 Shared Pattern

All three involve repeated conditional steps:

| System    | Repeated Step                                            |
| --------- | -------------------------------------------------------- |
| LLM       | (P(\text{next token} \mid \text{context}))               |
| RL        | (\pi(\text{action} \mid \text{state})), (P(s' \mid s,a)) |
| Diffusion | (P(x\_{t-1} \mid x_t, \text{condition}))                 |

Core capstone sentence:

> Modern AI often works by repeatedly sampling from conditional distributions.

---

# Part 9 — Final Course Summary

**Time:** 2 minutes

## The Four Roles of Probability in AI

By the end of the course, students should see probability in four major roles.

---

## 1. Prediction

Models estimate probabilities over outputs.

[
P(y \mid x)
]

Examples:

- image classification,
- disease prediction,
- next-token prediction,
- noisy image denoising.

Core idea:

> Prediction means estimating what is likely given information.

---

## 2. Learning

Models are trained to assign high probability to observed data.

[
-\log P(\text{correct output})
]

Examples:

- cross-entropy for classifiers,
- next-token loss for LLMs,
- likelihood-based generative modeling.

Core idea:

> Learning often means making the training data more probable under the model.

---

## 3. Decision-Making

Agents choose actions to maximize expected future reward.

[
\mathbb{E}[G_t]
]

Examples:

- robot control,
- game playing,
- recommendation systems,
- RLHF-style optimization.

Core idea:

> Decision-making means choosing actions under uncertainty.

---

## 4. Generation

Models sample from learned distributions.

Examples:

- LLMs sample tokens,
- diffusion models sample images,
- VAEs sample latent variables,
- RL policies sample actions.

Core idea:

> Generation means turning probability distributions into concrete outputs.

---

# Board Summary

A compact final board could look like this:

[
\text{Sampling: } x \sim P(x)
]

[
P(x_t \mid x_1,...,x_{t-1})
\quad
\text{LLM next token}
]

[
P(y_i) =
\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
\quad
\text{temperature-modified softmax}
]

[
z \sim P(z), \quad x \sim P(x \mid z)
\quad
\text{latent variable generation}
]

[
X \sim \mathcal{N}(\mu,\sigma^2)
\quad
\text{Gaussian noise}
]

[
x_0 \rightarrow x_1 \rightarrow \cdots \rightarrow x_T
\quad
\text{forward noising}
]

[
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_0
\quad
\text{reverse denoising}
]

[
P(y \mid x)
\quad
\text{prediction}
]

[
-\log P(\text{correct output})
\quad
\text{learning}
]

[
\mathbb{E}[G_t]
\quad
\text{decision-making}
]

And the AI translation:

| Probability Concept     | AI Meaning                              |
| ----------------------- | --------------------------------------- |
| Sampling                | turning a distribution into an output   |
| Temperature             | controlling randomness during sampling  |
| Latent variable         | hidden structure behind observed data   |
| Gaussian noise          | mathematically convenient randomness    |
| Diffusion               | learned reverse noising process         |
| Entropy                 | uncertainty/diversity of a distribution |
| Conditional probability | generation guided by context or prompt  |
| Expected return         | RL objective                            |
| Likelihood              | training objective                      |

---

# Suggested Exercises

## Exercise 1 — Sampling vs Greedy

Given:

| Token  | Probability |
| ------ | ----------: |
| blue   |        0.45 |
| green  |        0.30 |
| red    |        0.20 |
| yellow |        0.05 |

Questions:

1. What does greedy decoding choose?
2. Can sampling choose “red”?
3. If sampled 1,000 times, roughly how often should “green” appear?
4. Why might sampling be better than greedy decoding for creative writing?

Expected answers:

1. Blue.
2. Yes.
3. Around 300 times.
4. It allows diverse plausible outputs.

---

## Exercise 2 — Temperature

Suppose a distribution is:

| Token   | Probability |
| ------- | ----------: |
| safe    |        0.70 |
| risky   |        0.20 |
| strange |        0.10 |

Questions:

1. What happens at low temperature?
2. What happens at high temperature?
3. Does temperature change the model’s knowledge?
4. Which setting is better for factual QA?

Expected answers:

1. “safe” becomes even more likely.
2. “risky” and “strange” become more likely.
3. No.
4. Usually lower temperature.

---

## Exercise 3 — Latent Variables

For each observed output, name two possible latent variables.

1. A generated portrait.
2. A restaurant review.
3. A song recommendation.
4. A medical symptom pattern.
5. A generated city image.

Possible answers:

1. age, pose, expression, lighting.
2. sentiment, cuisine type, reviewer preference.
3. user taste, mood, genre preference.
4. disease state, severity.
5. architectural style, time of day, weather.

---

## Exercise 4 — Gaussian Noise

Questions:

1. What does (\mu) mean in (\mathcal{N}(\mu,\sigma^2))?
2. What does (\sigma^2) mean?
3. What happens when variance increases?
4. Why is Gaussian noise useful in diffusion models?

Expected answers:

1. Mean.
2. Variance/spread.
3. Samples become more spread out/noisy.
4. It provides a controlled noising process the model can learn to reverse.

---

## Exercise 5 — Diffusion Process

Fill in the blanks:

1. The forward process turns data into **\_\_**.
2. The reverse process turns noise into **\_\_**.
3. The model learns to predict or remove **\_\_**.
4. Text-to-image diffusion models generate images conditioned on **\_\_**.

Expected answers:

1. noise.
2. data/images.
3. noise.
4. text prompts.

---

# What to Emphasize Most

The most important ideas in Lecture 5 are:

1. **Sampling turns probabilities into outputs.**
2. **Greedy decoding and sampling are different.**
3. **Temperature controls randomness, not intelligence.**
4. **Latent variables represent hidden structure.**
5. **Gaussian noise is central to modern generative modeling.**
6. **Diffusion models learn to reverse a noising process.**
7. **The same prompt can produce different outputs because generation is probabilistic.**
8. **LLMs, RL policies, and diffusion models all rely on repeated conditional sampling.**

The central conceptual move is:

[
\text{probability distribution}
\rightarrow
\text{sampling}
\rightarrow
\text{generated output}
]

---

# What Not to Overdo

Avoid spending too much time on:

- full diffusion ELBO derivations,
- score matching derivations,
- stochastic differential equations,
- DDPM vs DDIM details,
- classifier-free guidance formulas,
- VAE ELBO derivation,
- advanced latent diffusion architecture,
- transformer decoding implementation details,
- beam search details,
- exact sampling algorithms.

Those are important later, but not for a five-hour crash course.

The priority is that students understand:

> Generative AI works by learning probability distributions and sampling from them, often through repeated conditional steps.

---

# Optional Advanced Add-On If Time Allows

## Classifier-Free Guidance Intuition

For text-to-image models, the prompt guides generation.

Very roughly, classifier-free guidance increases how strongly the model follows the prompt.

Low guidance:

- more natural variation,
- less strict prompt adherence.

High guidance:

- stronger prompt adherence,
- but sometimes less natural or more distorted.

Do not present formulas unless students are strong.

Core intuition:

> Guidance changes how strongly the condition influences denoising.

---

## Score-Based Intuition

Another way to describe diffusion:

The model learns the direction that points from noisy data back toward likely data.

In simple terms:

> At each step, the model asks: “Which direction makes this noisy sample look more like real data?”

This is a useful intuition, but avoid formal score matching unless this becomes a longer course.

---

# Final Course Wrap-Up

End the entire five-hour crash course with a synthesis.

## One-Line Summary of Each Lecture

### Lecture 1

Probability describes uncertainty.

[
P(x)
]

### Lecture 2

Conditional probability describes prediction from information.

[
P(y \mid x)
]

### Lecture 3

Likelihood and cross-entropy explain how neural networks learn probabilistic predictions.

[
-\log P(y_{\text{true}} \mid x)
]

### Lecture 4

Reinforcement learning uses probability for decisions over time.

[
\max*\pi \mathbb{E}*\pi[G_t]
]

### Lecture 5

Generative AI samples from learned distributions.

[
x \sim P_\theta(x)
]

---

# Final Concept Map

The whole course can be summarized as:

[
\text{uncertainty}
\rightarrow
\text{distributions}
\rightarrow
\text{conditional probability}
\rightarrow
\text{likelihood}
\rightarrow
\text{learning}
\rightarrow
\text{sampling}
\rightarrow
\text{generation and decision-making}
]

Or, in AI terms:

[
\text{data}
\rightarrow
\text{model learns probabilities}
\rightarrow
\text{model predicts, decides, or generates}
]

---

# Final Takeaway for Students

Students should leave with this mental model:

> Modern AI is not just linear algebra and optimization. It is also probability. Neural networks produce distributions, training makes observed data more probable, RL maximizes expected future reward, LLMs sample tokens, and diffusion models turn random noise into structured outputs through learned denoising.

Final board line:

[
\boxed{
\text{Modern AI = learned probability distributions + optimization + sampling}
}
]
