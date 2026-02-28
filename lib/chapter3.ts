// lib/chapter3.ts
import { Question } from "./quiz";

export const chapter3Questions: Question[] = [
  // --------------------------------------------------------------------------------
  // Q1–Q25: all four options are correct (pattern: 4 correct)
  // --------------------------------------------------------------------------------

  {
    id: "ch3-q01",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements about large language models (LLMs) are correct?",
    options: [
      {
        text: "Most modern LLMs are very large transformer models trained with next-token prediction.",
        isCorrect: true,
      },
      {
        text: "LLMs typically have at least tens of billions of parameters.",
        isCorrect: true,
      },
      {
        text: "They are trained on diverse text corpora such as web pages, books, and code.",
        isCorrect: true,
      },
      {
        text: "They can be adapted to many NLP tasks without changing the core architecture.",
        isCorrect: true,
      },
    ],
    explanation:
      "LLMs are large transformer-based models (often >10B parameters) trained with autoregressive language modeling on very diverse corpora, and reused for many downstream tasks.",
  },
  {
    id: "ch3-q02",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements describe why many practical NLP tasks can be cast as language modeling (next-token prediction)?",
    options: [
      {
        text: "We can phrase classification as asking the model to predict a label token after a natural-language prompt.",
        isCorrect: true,
      },
      {
        text: "Question answering can be expressed as predicting an answer string given the question and any context.",
        isCorrect: true,
      },
      {
        text: "Summarization can be seen as predicting a summary continuation conditioned on the source document.",
        isCorrect: true,
      },
      {
        text: "Framing tasks as next-token prediction lets us reuse the same pretraining objective for many applications.",
        isCorrect: true,
      },
    ],
    explanation:
      "By embedding the task description and inputs into a text prompt, almost any NLP task can be reframed as predicting appropriate next tokens, so one LM objective can support many tasks.",
  },
  {
    id: "ch3-q03",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about scaling laws for LLMs (relating loss to model size, data size, and compute) are correct?",
    options: [
      {
        text: "Empirically, the cross-entropy loss decreases in a predictable way as we increase model size, dataset size, or compute.",
        isCorrect: true,
      },
      {
        text: "The scaling law can be expressed with power-law relationships between loss and each of N, D, and C when the other two are held fixed.",
        isCorrect: true,
      },
      {
        text: "Irreducible loss represents the part of the loss that remains even with arbitrarily large models and data, due to data entropy.",
        isCorrect: true,
      },
      {
        text: "Scaling laws help us estimate how much we should increase model or dataset size to reach a target loss before actually training.",
        isCorrect: true,
      },
    ],
    explanation:
      "OpenAI-style scaling laws show smooth power-law curves for loss versus model size, data size, and compute, separating reducible and irreducible loss and guiding design choices.",
  },
  {
    id: "ch3-q04",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the idea of emergent abilities in LLMs?",
    options: [
      {
        text: "Some capabilities appear only once the model exceeds a certain scale, called a critical scale.",
        isCorrect: true,
      },
      {
        text: "Performance on some tasks can jump from near-random to near–state of the art over a relatively small size range.",
        isCorrect: true,
      },
      {
        text: "Examples of such abilities include multi-step reasoning and following complex instructions.",
        isCorrect: true,
      },
      {
        text: "There is ongoing debate about whether these abilities are truly discontinuous or just look abrupt due to how we measure performance.",
        isCorrect: true,
      },
    ],
    explanation:
      "Emergent abilities refer to behaviors that seem to appear suddenly when crossing a size threshold (for example better reasoning), though later work questions how sharp these transitions really are.",
  },
  {
    id: "ch3-q05",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements about context length in LLMs are correct?",
    options: [
      {
        text: "Context length is the maximum number of tokens the model can attend to in a single forward pass.",
        isCorrect: true,
      },
      {
        text: "Longer context windows allow the model to condition on more information, which helps tasks such as summarization and long-document QA.",
        isCorrect: true,
      },
      {
        text: "The computational and memory cost of vanilla self-attention grows roughly quadratically with context length.",
        isCorrect: true,
      },
      {
        text: "Extending context length is therefore useful but also expensive, motivating efficient attention variants.",
        isCorrect: true,
      },
    ],
    explanation:
      "Context length determines how many tokens are visible at once; more context helps but quadratic attention cost makes very long contexts expensive.",
  },
  {
    id: "ch3-q06",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe mixture-of-experts (MoE) layers?",
    options: [
      {
        text: "An MoE layer contains several expert networks (often FFNs) and a router that decides which experts to apply to each token.",
        isCorrect: true,
      },
      {
        text: "Only a subset of experts is activated per token, so computation is sparse even if the total parameter count is large.",
        isCorrect: true,
      },
      {
        text: "Experts can specialize on different regions of the input space, such as different languages or styles.",
        isCorrect: true,
      },
      {
        text: "Modern very large models like GPT-4 and Gemini are believed to use MoE-style components internally.",
        isCorrect: true,
      },
    ],
    explanation:
      "MoE layers route each token through only a few learned experts, making it possible to increase parameter count without proportional compute per token.",
  },
  {
    id: "ch3-q07",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements are advantages often cited for mixture-of-experts architectures?",
    options: [
      {
        text: "They can achieve similar quality to dense models with less training compute by activating only a few experts per token.",
        isCorrect: true,
      },
      {
        text: "Different experts can specialize, potentially improving performance on heterogeneous data distributions.",
        isCorrect: true,
      },
      {
        text: "MoEs are highly scalable because we can add more experts without increasing the number of active parameters per token.",
        isCorrect: true,
      },
      {
        text: "Their ensemble nature can resemble a 'wisdom of the crowd' effect across experts.",
        isCorrect: true,
      },
    ],
    explanation:
      "Because MoEs activate only some experts, they save compute, allow specialization, scale to many parameters, and often behave like ensembles.",
  },
  {
    id: "ch3-q08",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements describe common challenges or downsides of mixture-of-experts models?",
    options: [
      {
        text: "They require keeping all experts in memory even though only a few are used per token.",
        isCorrect: true,
      },
      {
        text: "Training can be unstable if the router collapses to using only a few experts much of the time.",
        isCorrect: true,
      },
      {
        text: "Fine-tuning MoE models can be more complex than fine-tuning dense models.",
        isCorrect: true,
      },
      {
        text: "Interpretability becomes harder because we must reason about both routing and many expert subnetworks.",
        isCorrect: true,
      },
    ],
    explanation:
      "MoEs increase memory usage, can suffer from expert imbalance, complicate fine-tuning, and add interpretability challenges.",
  },
  {
    id: "ch3-q09",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements about Low-Rank Adaptation (LoRA) for fine-tuning LLMs are correct?",
    options: [
      {
        text: "LoRA keeps the original pretrained weight matrix W frozen.",
        isCorrect: true,
      },
      {
        text: "It learns a low-rank update ΔW that is decomposed as the product of two smaller matrices A and B.",
        isCorrect: true,
      },
      {
        text: "The effective weight during fine-tuning is W′ = W + BA (or W + AB, depending on convention).",
        isCorrect: true,
      },
      {
        text: "Only the new small matrices are trained, so the number of trainable parameters is much smaller than in full fine-tuning.",
        isCorrect: true,
      },
    ],
    explanation:
      "LoRA freezes W and learns a low-rank correction factorized into two smaller matrices; this greatly reduces trainable parameters while preserving the base model.",
  },
  {
    id: "ch3-q10",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements correctly describe adapter-based fine-tuning?",
    options: [
      {
        text: "Small trainable adapter modules are inserted inside transformer blocks.",
        isCorrect: true,
      },
      {
        text: "Adapters typically implement a bottleneck MLP that projects down to a smaller dimension and back up.",
        isCorrect: true,
      },
      {
        text: "During fine-tuning, only the adapter parameters are trained while the original transformer weights remain frozen.",
        isCorrect: true,
      },
      {
        text: "Adapters allow one base model to support many tasks by loading different adapter parameter sets.",
        isCorrect: true,
      },
    ],
    explanation:
      "Adapters are lightweight task-specific modules that plug into transformer layers, enabling efficient fine-tuning across many tasks with a shared backbone.",
  },
  {
    id: "ch3-q11",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about instruction tuning (IT) of LLMs are correct?",
    options: [
      {
        text: "Instruction tuning trains an LLM on pairs of instructions and desired outputs.",
        isCorrect: true,
      },
      {
        text: "Datasets often include a wide variety of tasks such as QA, summarization, classification, and translation.",
        isCorrect: true,
      },
      {
        text: "Instruction-tuned models tend to follow natural-language instructions better, even for tasks not seen during IT.",
        isCorrect: true,
      },
      {
        text: "Instruction tuning is often combined with other alignment methods such as RLHF.",
        isCorrect: true,
      },
    ],
    explanation:
      "IT exposes the model to diverse instruction–response pairs, improving instruction-following and transfer to unseen tasks, and is frequently paired with RLHF.",
  },
  {
    id: "ch3-q12",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe Reinforcement Learning from Human Feedback (RLHF)?",
    options: [
      {
        text: "It typically starts with supervised fine-tuning on human-written responses to prompts.",
        isCorrect: true,
      },
      {
        text: "A separate reward model is trained to predict human preference rankings over multiple candidate completions.",
        isCorrect: true,
      },
      {
        text: "The main policy model is then optimized (for example with PPO) to maximize the reward model’s scores while staying close to the original model.",
        isCorrect: true,
      },
      {
        text: "A KL-divergence penalty is usually used so the policy does not drift too far from the supervised model’s behavior.",
        isCorrect: true,
      },
    ],
    explanation:
      "RLHF uses SFT to get a helpful base model, a reward model trained on preference data, and RL (often PPO with a KL penalty) to steer the policy toward human-preferred outputs.",
  },
  {
    id: "ch3-q13",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements describe Direct Preference Optimization (DPO) as an alternative to RLHF?",
    options: [
      {
        text: "DPO avoids training a separate reward model and avoids online reinforcement learning.",
        isCorrect: true,
      },
      {
        text: "Training data consists of tuples like ⟨prompt, worse completion, better completion⟩.",
        isCorrect: true,
      },
      {
        text: "The loss is designed to increase the likelihood of preferred completions relative to dispreferred ones.",
        isCorrect: true,
      },
      {
        text: "DPO can be implemented using standard backpropagation on the language model itself.",
        isCorrect: true,
      },
    ],
    explanation:
      "DPO uses paired preference data and a contrastive-style loss directly on the LM, bypassing a separate reward model and RL.",
  },
  {
    id: "ch3-q14",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements are typical motivations for using smaller language models (SLMs) instead of huge LLMs?",
    options: [
      {
        text: "SLMs require less memory and compute, so they are cheaper to deploy and run.",
        isCorrect: true,
      },
      {
        text: "For many business cases, a small, domain-specific model is sufficient.",
        isCorrect: true,
      },
      {
        text: "SLMs can sometimes run on a single commercial GPU or even CPU.",
        isCorrect: true,
      },
      {
        text: "Fine-tuning and experimentation with SLMs are usually faster than with hundred-billion parameter models.",
        isCorrect: true,
      },
    ],
    explanation:
      "Small models can be sufficient, cheaper, and easier to run and iterate on, especially in focused domains.",
  },
  {
    id: "ch3-q15",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe quantization for neural networks and LLMs?",
    options: [
      {
        text: "Quantization maps weights or activations from higher-precision types (like float32) to lower-precision types (like int8).",
        isCorrect: true,
      },
      {
        text: "Affine quantization uses a scale factor and a zero-point to map real-valued ranges into discrete integer ranges.",
        isCorrect: true,
      },
      {
        text: "Properly designed quantization schemes aim to reduce memory and compute cost while keeping accuracy as high as possible.",
        isCorrect: true,
      },
      {
        text: "Some quantization methods require clipping values so that quantized numbers remain within the representable integer range.",
        isCorrect: true,
      },
    ],
    explanation:
      "Quantization replaces high-precision numbers with low-precision approximations using scale and zero-point parameters, reducing resource usage while trying to preserve model quality.",
  },
  {
    id: "ch3-q16",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe pruning as applied to large language models?",
    options: [
      {
        text: "Pruning removes parameters, connections, or even entire layers that contribute little to the model’s performance.",
        isCorrect: true,
      },
      {
        text: "Unstructured pruning sets individual small-magnitude weights to zero, creating sparsity.",
        isCorrect: true,
      },
      {
        text: "Structured pruning can remove neurons, attention heads, or blocks in a way that keeps the architecture hardware-friendly.",
        isCorrect: true,
      },
      {
        text: "Aggressive pruning can cause model collapse if too much important structure is removed.",
        isCorrect: true,
      },
    ],
    explanation:
      "Both unstructured and structured pruning try to eliminate redundant parameters, but over-pruning can severely degrade performance or destabilize the model.",
  },
  {
    id: "ch3-q17",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements describe the basic idea of a Vision Transformer (ViT)?",
    options: [
      {
        text: "The image is split into fixed-size patches that play a role similar to tokens in text.",
        isCorrect: true,
      },
      {
        text: "Each patch is flattened and linearly projected into an embedding vector.",
        isCorrect: true,
      },
      {
        text: "Positional encodings are added so the model knows where each patch came from in the image.",
        isCorrect: true,
      },
      {
        text: "The sequence of patch embeddings is processed by a transformer encoder stack.",
        isCorrect: true,
      },
    ],
    explanation:
      "ViTs tokenize images into patches, embed them, add positional information, and process them with transformer encoders much like word tokens.",
  },
  {
    id: "ch3-q18",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements correctly describe CLIP-style multimodal models?",
    options: [
      {
        text: "CLIP trains a text encoder and an image encoder jointly so that corresponding image–text pairs have similar embeddings.",
        isCorrect: true,
      },
      {
        text: "Training uses contrastive learning: matching pairs are pulled together, mismatched pairs are pushed apart in embedding space.",
        isCorrect: true,
      },
      {
        text: "After training, CLIP embeddings can be used for zero-shot image classification by comparing image embeddings with label prompt embeddings.",
        isCorrect: true,
      },
      {
        text: "The similarity between images and texts is often measured with cosine similarity.",
        isCorrect: true,
      },
    ],
    explanation:
      "CLIP uses contrastive learning on (image, caption) pairs to align image and text embeddings, enabling cross-modal retrieval and zero-shot classification.",
  },
  {
    id: "ch3-q19",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about vision–language models (VLMs) such as BLIP-2 are correct?",
    options: [
      {
        text: "They connect an image encoder (such as a ViT) with a text-generating LLM.",
        isCorrect: true,
      },
      {
        text: "A bridge module (such as Q-Former in BLIP-2) maps visual features into a form the LLM can condition on.",
        isCorrect: true,
      },
      {
        text: "Once trained, such models can generate captions and answer questions about images.",
        isCorrect: true,
      },
      {
        text: "Only the bridge module may need to be trained from scratch; image encoder and LLM can often remain frozen.",
        isCorrect: true,
      },
    ],
    explanation:
      "VLMs tie visual encoders to LLMs via a trainable connector; this lets a largely frozen LLM reason about visual content.",
  },
  {
    id: "ch3-q20",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly summarize the high-level components of a text-to-image diffusion model like Stable Diffusion?",
    options: [
      {
        text: "A text encoder (often CLIP-based) turns the prompt into an embedding that conditions image generation.",
        isCorrect: true,
      },
      {
        text: "A U-Net–style network operates in a latent space, iteratively denoising a noisy latent into a structured latent.",
        isCorrect: true,
      },
      {
        text: "An image decoder (for example, a variational autoencoder decoder) maps the latent representation into pixel space.",
        isCorrect: true,
      },
      {
        text: "During training, the model learns to predict and remove noise that was artificially added to images.",
        isCorrect: true,
      },
    ],
    explanation:
      "Text-to-image diffusion models combine a text encoder, a latent-space denoising U-Net, and a decoder, using a noise-prediction objective to learn how to generate images from noise conditioned on text.",
  },
  {
    id: "ch3-q21",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements describe hallucinations in large language models?",
    options: [
      {
        text: "Hallucinations are outputs that are fluent but factually wrong or not supported by the evidence.",
        isCorrect: true,
      },
      {
        text: "Factual hallucinations contradict real-world facts that could in principle be verified.",
        isCorrect: true,
      },
      {
        text: "Faithfulness hallucinations occur when the answer is inconsistent with the user’s instructions or supplied context.",
        isCorrect: true,
      },
      {
        text: "Hallucinations are especially problematic in high-stakes domains such as medicine and law.",
        isCorrect: true,
      },
    ],
    explanation:
      "LLMs can produce confident but incorrect statements; these are hallucinations, which may be factual or faithfulness-related and are dangerous in safety-critical uses.",
  },
  {
    id: "ch3-q22",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe common types of harm and bias associated with LLMs?",
    options: [
      {
        text: "Representational harm occurs when the model reinforces stereotypes or offensive portrayals of certain groups.",
        isCorrect: true,
      },
      {
        text: "Allocational harm occurs when biased models lead to unfair distribution of opportunities or resources (for example, in hiring or loans).",
        isCorrect: true,
      },
      {
        text: "Bias can arise from imbalanced or toxic pretraining data as well as from biased annotations.",
        isCorrect: true,
      },
      {
        text: "Measuring and mitigating bias often requires dedicated evaluation datasets and metrics beyond simple accuracy.",
        isCorrect: true,
      },
    ],
    explanation:
      "LLMs can embed harmful stereotypes (representational harm) and cause downstream unfair treatment (allocational harm); mitigating this requires careful data curation and specialized evaluations.",
  },
  {
    id: "ch3-q23",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe in-context learning (ICL) in LLMs?",
    options: [
      {
        text: "ICL is the ability of an LLM to perform a new task at inference time using examples provided in the prompt.",
        isCorrect: true,
      },
      {
        text: "The model’s parameters are not updated; it relies on patterns learned during pretraining.",
        isCorrect: true,
      },
      {
        text: "Few-shot prompting shows input–output examples and asks the model to complete a new test example in the same format.",
        isCorrect: true,
      },
      {
        text: "ICL emerged as a surprising capability: LLMs can often match fine-tuned models on some benchmarks using only prompt examples.",
        isCorrect: true,
      },
    ],
    explanation:
      "ICL lets LLMs learn from prompt examples without weight updates, using few-shot prompts to infer the input–output mapping on the fly.",
  },
  {
    id: "ch3-q24",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly contrast zero-shot, few-shot, and chain-of-thought prompting?",
    options: [
      {
        text: "Zero-shot prompting gives only an instruction or question, with no input–output examples.",
        isCorrect: true,
      },
      {
        text: "Few-shot prompting provides several example pairs before asking for a new answer.",
        isCorrect: true,
      },
      {
        text: "Chain-of-thought prompting includes intermediate reasoning steps along with the final answer in the examples.",
        isCorrect: true,
      },
      {
        text: "Zero-shot chain-of-thought prompting can sometimes be triggered by simple phrases such as “Let’s think step by step.”",
        isCorrect: true,
      },
    ],
    explanation:
      "Zero-shot uses only instructions, few-shot adds labeled examples, and chain-of-thought additionally exposes reasoning steps, which can be requested explicitly even in zero-shot mode.",
  },
  {
    id: "ch3-q25",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe advanced prompting and programmatic prompting ideas such as self-consistency, Tree of Thoughts, and DSPy?",
    options: [
      {
        text: "Self-consistency generates multiple reasoning paths and chooses the majority or best-scoring answer.",
        isCorrect: true,
      },
      {
        text: "Tree of Thoughts explores multiple partial reasoning paths using search algorithms over intermediate states.",
        isCorrect: true,
      },
      {
        text: "DSPy treats prompts and prompting strategies as declarative programs that can be automatically optimized.",
        isCorrect: true,
      },
      {
        text: "These approaches typically trade extra compute for improved reliability or reasoning quality.",
        isCorrect: true,
      },
    ],
    explanation:
      "Methods like self-consistency, ToT, and DSPy structure or optimize prompting, often sampling multiple paths or automatically tuning prompts to improve reasoning at the cost of more computation.",
  },

  // --------------------------------------------------------------------------------
  // Q26–Q50: exactly three options are correct (pattern: 3 correct)
  // --------------------------------------------------------------------------------

  {
    id: "ch3-q26",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements about the definition of an LLM in this chapter are correct?",
    options: [
      {
        text: "An LLM is roughly a transformer model with more than about 10 billion parameters.",
        isCorrect: true,
      },
      {
        text: "Current mainstream LLMs are almost all trained with an autoregressive next-token prediction objective.",
        isCorrect: true,
      },
      {
        text: "LLMs are always recurrent neural networks without self-attention.",
        isCorrect: false,
      },
      {
        text: "LLMs are typically trained on a huge variety of text sources and sometimes code.",
        isCorrect: true,
      },
    ],
    explanation:
      "LLMs are large transformer models (not RNNs) trained on massive diverse corpora with autoregressive language modeling.",
  },
  {
    id: "ch3-q27",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Consider the OpenAI-style scaling law L(N) = (N_c / N)^{α_N} when data size and compute are fixed. Which statements are correct?",
    options: [
      {
        text: "N denotes the number of model parameters and N_c is a reference scale where the law is calibrated.",
        isCorrect: true,
      },
      {
        text: "α_N is a positive exponent that controls how quickly loss decreases as model size increases.",
        isCorrect: true,
      },
      {
        text: "If N grows larger than N_c, the term (N_c / N)^{α_N} becomes smaller, corresponding to lower loss.",
        isCorrect: true,
      },
      {
        text: "The formula implies that loss increases when we make the model larger than N_c.",
        isCorrect: false,
      },
    ],
    explanation:
      "N is parameter count, N_c a reference, and α_N > 0; for N > N_c the ratio N_c/N < 1 so the power term decreases, representing lower loss.",
  },
  {
    id: "ch3-q28",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about the Chinchilla viewpoint on scaling are correct?",
    options: [
      {
        text: "Chinchilla argues that performance depends more strongly on token count than some earlier scaling analyses suggested.",
        isCorrect: true,
      },
      {
        text: "It suggests many existing LLMs are undertrained relative to their size because they see too few tokens.",
        isCorrect: true,
      },
      {
        text: "It recommends, for a fixed compute budget, training smaller models for more tokens rather than ever-larger models.",
        isCorrect: true,
      },
      {
        text: "It concludes that data quality is irrelevant as long as token count is large enough.",
        isCorrect: false,
      },
    ],
    explanation:
      "Chinchilla highlights the importance of token count and compute-optimal trade-offs; it does not claim data quality is irrelevant.",
  },
  {
    id: "ch3-q29",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe why long context windows are useful?",
    options: [
      {
        text: "They allow the model to see larger portions of long documents at once when summarizing.",
        isCorrect: true,
      },
      {
        text: "They help multi-turn dialogue agents remember more of the conversation history.",
        isCorrect: true,
      },
      {
        text: "They make it impossible for the model to overfit on the training data.",
        isCorrect: false,
      },
      {
        text: "They help question answering systems reason over many scattered evidence passages in a single prompt.",
        isCorrect: true,
      },
    ],
    explanation:
      "Longer context improves access to relevant information across tasks like summarization, QA, and conversation, but does not magically prevent overfitting.",
  },
  {
    id: "ch3-q30",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about the router in a mixture-of-experts layer are correct?",
    options: [
      {
        text: "It learns to map each token (or hidden vector) to one or more experts based on its content.",
        isCorrect: true,
      },
      {
        text: "Its parameters are trained jointly with the rest of the model during pretraining.",
        isCorrect: true,
      },
      {
        text: "With appropriate regularization, it can encourage balanced use of experts rather than always picking the same few.",
        isCorrect: true,
      },
      {
        text: "After training, the router is typically discarded and experts are all run on every token.",
        isCorrect: false,
      },
    ],
    explanation:
      "The router is a learnable module that decides which experts to activate and must be kept at inference time; load balancing is a key concern.",
  },
  {
    id: "ch3-q31",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements are advantages of parameter-efficient fine-tuning (PEFT) methods such as LoRA and adapters?",
    options: [
      {
        text: "They drastically reduce the number of trainable parameters compared with full-model fine-tuning.",
        isCorrect: true,
      },
      {
        text: "They allow multiple domain-specific adaptations to be stored as small extra parameter sets.",
        isCorrect: true,
      },
      {
        text: "They completely eliminate the need for GPUs during training.",
        isCorrect: false,
      },
      {
        text: "They avoid overwriting the base model weights, helping to preserve original capabilities.",
        isCorrect: true,
      },
    ],
    explanation:
      "PEFT methods save compute and storage, preserve base weights, and support multiple light-weight adaptations, but GPUs are usually still helpful.",
  },
  {
    id: "ch3-q32",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about instruction tuning datasets are correct?",
    options: [
      {
        text: "They contain natural-language instructions paired with target outputs.",
        isCorrect: true,
      },
      {
        text: "Instructions are often drawn from many task types such as QA, summarization, and translation.",
        isCorrect: true,
      },
      {
        text: "High-quality instruction datasets may include chain-of-thought style rationales in the outputs.",
        isCorrect: true,
      },
      {
        text: "Instruction tuning datasets cannot contain any machine-generated outputs.",
        isCorrect: false,
      },
    ],
    explanation:
      "Instruction tuning uses diverse tasks; some outputs may be human-written or LLM-generated. Including rationales can improve reasoning abilities.",
  },
  {
    id: "ch3-q33",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about the reward model used in RLHF are correct?",
    options: [
      {
        text: "It is trained on human preference rankings over multiple candidate responses per prompt.",
        isCorrect: true,
      },
      {
        text: "It outputs a scalar score that estimates how aligned a response is with human preferences.",
        isCorrect: true,
      },
      {
        text: "During RL, the policy model generates responses that are scored by the reward model.",
        isCorrect: true,
      },
      {
        text: "Once trained, the reward model directly replaces the language model for generation.",
        isCorrect: false,
      },
    ],
    explanation:
      "The reward model scores candidate outputs but does not itself generate text; it guides RL updates to the main language model.",
  },
  {
    id: "ch3-q34",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly distinguish small language models (SLMs) from very large LLMs?",
    options: [
      {
        text: "SLMs may have only millions or a few billions of parameters, instead of tens or hundreds of billions.",
        isCorrect: true,
      },
      {
        text: "SLMs are often specialized to narrower domains or tasks.",
        isCorrect: true,
      },
      {
        text: "SLMs always outperform LLMs on open-ended reasoning tasks.",
        isCorrect: false,
      },
      {
        text: "SLMs can sometimes be deployed on edge devices such as mobile phones.",
        isCorrect: true,
      },
    ],
    explanation:
      "Small models trade generality and raw capability for efficiency and deployability; they do not universally beat large models.",
  },
  {
    id: "ch3-q35",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about the trade-offs of quantizing an LLM from 16-bit to 8-bit weights are correct?",
    options: [
      {
        text: "Model size in memory roughly halves when moving from 16-bit to 8-bit weights.",
        isCorrect: true,
      },
      {
        text: "Matrix multiplications can be faster on hardware with efficient low-precision support.",
        isCorrect: true,
      },
      {
        text: "There is usually some degradation in model accuracy, although good schemes try to make it small.",
        isCorrect: true,
      },
      {
        text: "Quantization guarantees zero loss in accuracy compared to the original model.",
        isCorrect: false,
      },
    ],
    explanation:
      "Quantization reduces memory and can speed up compute but often introduces small errors that may slightly degrade quality.",
  },
  {
    id: "ch3-q36",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly compare unstructured and structured pruning?",
    options: [
      {
        text: "Unstructured pruning can set arbitrary individual weights to zero, leading to irregular sparsity patterns.",
        isCorrect: true,
      },
      {
        text: "Structured pruning removes groups such as neurons, channels, or entire layers, keeping a regular structure.",
        isCorrect: true,
      },
      {
        text: "Structured pruning is often easier to accelerate on standard hardware than unstructured pruning.",
        isCorrect: true,
      },
      {
        text: "Unstructured pruning never harms model performance, regardless of how many weights are removed.",
        isCorrect: false,
      },
    ],
    explanation:
      "Unstructured pruning zeros individual weights; structured pruning removes larger units and is more hardware-friendly, but both can hurt performance if overused.",
  },
  {
    id: "ch3-q37",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the basic tokenization step in a Vision Transformer?",
    options: [
      {
        text: "The image is divided into fixed-size patches like 16×16 pixels.",
        isCorrect: true,
      },
      {
        text: "Each patch is flattened into a 1D vector that includes all color channels.",
        isCorrect: true,
      },
      {
        text: "These patch vectors are linearly projected into a common embedding dimension.",
        isCorrect: true,
      },
      {
        text: "Each individual pixel is always treated as a separate token.",
        isCorrect: false,
      },
    ],
    explanation:
      "ViTs use patches, not individual pixels, as tokens; each patch is flattened and projected to an embedding.",
  },
  {
    id: "ch3-q38",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about CLIP’s training objective are correct?",
    options: [
      {
        text: "In each batch, the model sees multiple images and multiple captions.",
        isCorrect: true,
      },
      {
        text: "The objective encourages the embedding of each image to be closest to the embedding of its true caption.",
        isCorrect: true,
      },
      {
        text: "The objective simultaneously discourages similarity between an image and non-matching captions in the batch.",
        isCorrect: true,
      },
      {
        text: "The model is trained to reconstruct pixel values of the input images.",
        isCorrect: false,
      },
    ],
    explanation:
      "CLIP uses a contrastive objective on image–caption pairs rather than pixel reconstruction.",
  },
  {
    id: "ch3-q39",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe why LLM-based systems may pose privacy risks?",
    options: [
      {
        text: "Models can memorize rare sequences from training data, including personal information.",
        isCorrect: true,
      },
      {
        text: "Adversarial prompting can sometimes extract sensitive training examples from the model.",
        isCorrect: true,
      },
      {
        text: "Training on databases containing personal data without proper safeguards can violate privacy regulations.",
        isCorrect: true,
      },
      {
        text: "Once trained, LLMs are guaranteed to forget all individual training records.",
        isCorrect: false,
      },
    ],
    explanation:
      "LLMs may memorize sensitive data and leak it; forgetting is not automatic, which motivates machine unlearning and regulatory controls.",
  },
  {
    id: "ch3-q40",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about LLM-generated misinformation are correct?",
    options: [
      {
        text: "LLMs can generate high-volume, plausible-sounding text that could be used for disinformation campaigns.",
        isCorrect: true,
      },
      {
        text: "They can be misused to craft targeted phishing emails or rage-bait posts.",
        isCorrect: true,
      },
      {
        text: "Research taxonomies distinguish between types, domains, sources, intents, and errors of misinformation.",
        isCorrect: true,
      },
      {
        text: "LLMs always reliably distinguish between true and false content, making misinformation impossible.",
        isCorrect: false,
      },
    ],
    explanation:
      "Because LLMs generate convincing text without deep understanding, they can amplify misinformation; taxonomies help study this risk.",
  },
  {
    id: "ch3-q41",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements correctly describe zero-shot prompting?",
    options: [
      {
        text: "The prompt contains an instruction or question but no input–output examples.",
        isCorrect: true,
      },
      {
        text: "The model relies on capabilities learned during pretraining and any instruction tuning.",
        isCorrect: true,
      },
      {
        text: "Zero-shot performance can be improved by carefully specifying the task and desired format in the prompt.",
        isCorrect: true,
      },
      {
        text: "Zero-shot prompting requires updating model parameters for each new task.",
        isCorrect: false,
      },
    ],
    explanation:
      "Zero-shot prompts use only instructions; the model’s weights stay fixed and good task descriptions can significantly improve results.",
  },
  {
    id: "ch3-q42",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about few-shot prompting are correct?",
    options: [
      {
        text: "The prompt includes several example input–output pairs before the query to be answered.",
        isCorrect: true,
      },
      {
        text: "Examples help the model infer both the task and the desired output style.",
        isCorrect: true,
      },
      {
        text: "In practice, there is often a trade-off between including more examples and leaving room for long contexts or documents.",
        isCorrect: true,
      },
      {
        text: "Few-shot prompting only works if examples are drawn from the training data.",
        isCorrect: false,
      },
    ],
    explanation:
      "Few-shot examples guide the model by demonstration; they need not come from pretraining data, but context length limits how many can be used.",
  },
  {
    id: "ch3-q43",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe chain-of-thought (CoT) prompting?",
    options: [
      {
        text: "It provides intermediate reasoning steps, not just final answers, in example solutions.",
        isCorrect: true,
      },
      {
        text: "It can help LLMs perform better on multi-step reasoning tasks such as math word problems.",
        isCorrect: true,
      },
      {
        text: "Zero-shot CoT prompting can be triggered by adding phrases like “Let’s think step by step.”",
        isCorrect: true,
      },
      {
        text: "CoT guarantees mathematically correct reasoning on all problems.",
        isCorrect: false,
      },
    ],
    explanation:
      "CoT exposes reasoning steps and encourages the model to reason explicitly, but it is not a guarantee of correctness.",
  },
  {
    id: "ch3-q44",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe self-consistency in combination with chain-of-thought prompting?",
    options: [
      {
        text: "The model is sampled multiple times to produce diverse reasoning paths.",
        isCorrect: true,
      },
      {
        text: "Final answers are aggregated, for example by majority vote, to decide the output.",
        isCorrect: true,
      },
      {
        text: "The method aims to reduce the chance that a single flawed reasoning path dominates the result.",
        isCorrect: true,
      },
      {
        text: "Self-consistency reduces compute requirements compared with using a single CoT sample.",
        isCorrect: false,
      },
    ],
    explanation:
      "Self-consistency ensembles multiple CoT samples, improving robustness at the cost of extra generations.",
  },
  {
    id: "ch3-q45",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about Tree of Thoughts (ToT)–style prompting are correct?",
    options: [
      {
        text: "ToT treats intermediate reasoning states as nodes in a search tree.",
        isCorrect: true,
      },
      {
        text: "The model can generate multiple candidate continuations from each state.",
        isCorrect: true,
      },
      {
        text: "Search strategies such as breadth-first or depth-first search can be used to explore reasoning paths.",
        isCorrect: true,
      },
      {
        text: "ToT is guaranteed to find the globally optimal reasoning path with zero extra cost compared with standard prompting.",
        isCorrect: false,
      },
    ],
    explanation:
      "ToT structures reasoning as tree search with multiple candidates and explicit exploration, but it is more computationally expensive and not guaranteed to be optimal.",
  },
  {
    id: "ch3-q46",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the main idea of DSPy (Declarative Self-improving Language Programs)?",
    options: [
      {
        text: "DSPy treats prompts and prompting strategies as declarative programs composed of modules.",
        isCorrect: true,
      },
      {
        text: "It introduces signatures that specify the input–output behavior for model calls.",
        isCorrect: true,
      },
      {
        text: "Optimizers automatically tune prompts or module parameters to improve performance on a validation set.",
        isCorrect: true,
      },
      {
        text: "DSPy requires manual hand-tuning of every prompt without any automation.",
        isCorrect: false,
      },
    ],
    explanation:
      "DSPy turns prompt engineering into an optimization problem over declarative programs, with automated tuning instead of only manual tweaking.",
  },
  {
    id: "ch3-q47",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about evaluating bias in LLM outputs are correct?",
    options: [
      {
        text: "Libraries such as Hugging Face Evaluate provide metrics for toxicity and sentiment.",
        isCorrect: true,
      },
      {
        text: "Researchers may compare completions for prompts that vary only in a sensitive attribute, such as gender or profession.",
        isCorrect: true,
      },
      {
        text: "Heatmaps or distributions over these metrics can reveal systematic differences between groups.",
        isCorrect: true,
      },
      {
        text: "Bias can be fully understood by looking only at overall accuracy on a single benchmark.",
        isCorrect: false,
      },
    ],
    explanation:
      "Bias evaluation often uses group-wise metrics like toxicity or sentiment and comparisons across controlled prompt variations, beyond simple accuracy.",
  },
  {
    id: "ch3-q48",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the relation between emergent abilities and evaluation design?",
    options: [
      {
        text: "Some apparent emergent jumps may be due to coarse evaluation scales that hide gradual improvements.",
        isCorrect: true,
      },
      {
        text: "For certain tasks, performance appears flat and near-random until crossing a size threshold, after which it rises quickly.",
        isCorrect: true,
      },
      {
        text: "Different tasks can have different critical scales where abilities seem to emerge.",
        isCorrect: true,
      },
      {
        text: "All researchers agree that emergent abilities necessarily imply new algorithms emerging inside the model.",
        isCorrect: false,
      },
    ],
    explanation:
      "Emergent curves depend strongly on metric design; some tasks show sharp-looking transitions, but their interpretation is debated and does not universally imply qualitatively new algorithms.",
  },
  {
    id: "ch3-q49",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about catastrophic forgetting and model collapse when using synthetic data are correct?",
    options: [
      {
        text: "Training repeatedly on synthetic outputs from other models can degrade performance on real data.",
        isCorrect: true,
      },
      {
        text: "In extreme cases, the model may forget skills learned from human-written data.",
        isCorrect: true,
      },
      {
        text: "These phenomena motivate care when mixing human and synthetic data during training.",
        isCorrect: true,
      },
      {
        text: "Using only synthetic data is guaranteed to continually improve the model without risk.",
        isCorrect: false,
      },
    ],
    explanation:
      "Relying too heavily on synthetic data can lead to model collapse and forgetting, so data mixtures must be chosen carefully.",
  },
  {
    id: "ch3-q50",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about combining pruning and quantization for LLM compression are correct?",
    options: [
      {
        text: "Pruning removes weights or structures, while quantization lowers numerical precision.",
        isCorrect: true,
      },
      {
        text: "Together they can substantially reduce memory footprint and inference cost.",
        isCorrect: true,
      },
      {
        text: "Designing good compression pipelines requires monitoring quality to avoid severe degradation or collapse.",
        isCorrect: true,
      },
      {
        text: "Because the methods are independent, combining them always preserves original accuracy exactly.",
        isCorrect: false,
      },
    ],
    explanation:
      "Pruning and quantization address different axes of compression; combining them is powerful but must be done carefully to avoid harming quality.",
  },

  // --------------------------------------------------------------------------------
  // Q51–Q75: exactly two options are correct (pattern: 2 correct)
  // --------------------------------------------------------------------------------

  {
    id: "ch3-q51",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements about the training objective of an autoregressive LLM are correct?",
    options: [
      {
        text: "The model is trained to maximize the probability of each next token given previous tokens.",
        isCorrect: true,
      },
      {
        text: "The loss is typically the cross-entropy between predicted token distributions and true next tokens.",
        isCorrect: true,
      },
      {
        text: "The model is trained to predict both past and future tokens simultaneously.",
        isCorrect: false,
      },
      {
        text: "The loss does not depend on the observed training data at all.",
        isCorrect: false,
      },
    ],
    explanation:
      "Autoregressive LMs minimize cross-entropy for next-token prediction conditioned on past context only.",
  },
  {
    id: "ch3-q52",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the constants N_c, D_c, and C_c in scaling laws?",
    options: [
      {
        text: "They represent reference values for model size, dataset size, and compute used to express the power-law relationships.",
        isCorrect: true,
      },
      {
        text: "They depend on architecture choices and training setup.",
        isCorrect: true,
      },
      {
        text: "They are always equal to 1 by definition.",
        isCorrect: false,
      },
      {
        text: "They are independent of the specific model family and dataset.",
        isCorrect: false,
      },
    ],
    explanation:
      "The c-subscripted constants are fitted reference scales that depend on model family and training configuration; they are not universal or fixed at 1.",
  },
  {
    id: "ch3-q53",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about MoE memory and compute are correct?",
    options: [
      {
        text: "All experts must reside in memory even though only a subset are used per token.",
        isCorrect: true,
      },
      {
        text: "For a given token, only the selected experts contribute to forward and backward computation.",
        isCorrect: true,
      },
      {
        text: "MoE reduces both parameter count and compute compared with a dense model of the same width.",
        isCorrect: false,
      },
      {
        text: "MoE eliminates the need for routing decisions at inference time.",
        isCorrect: false,
      },
    ],
    explanation:
      "MoE saves compute by activating only a few experts but still stores all of them; parameter count is usually larger than a comparable dense model.",
  },
  {
    id: "ch3-q54",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly compare LoRA with classical full-model fine-tuning?",
    options: [
      {
        text: "LoRA keeps the base weights frozen and trains low-rank updates, whereas full fine-tuning updates all weights.",
        isCorrect: true,
      },
      {
        text: "LoRA typically requires far fewer trainable parameters than full fine-tuning.",
        isCorrect: true,
      },
      {
        text: "Full fine-tuning guarantees strictly better results than LoRA on every task.",
        isCorrect: false,
      },
      {
        text: "LoRA cannot be combined with other PEFT methods.",
        isCorrect: false,
      },
    ],
    explanation:
      "LoRA is more parameter-efficient than full fine-tuning and can perform competitively; results depend on the task, and it can be combined with other PEFT ideas.",
  },
  {
    id: "ch3-q55",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about adapters and LoRA are correct?",
    options: [
      {
        text: "Both approaches add a relatively small number of new trainable parameters compared with the base model.",
        isCorrect: true,
      },
      {
        text: "Both can support multiple task-specific parameter sets while sharing the same frozen backbone.",
        isCorrect: true,
      },
      {
        text: "Both require deleting the original pretrained weights before fine-tuning.",
        isCorrect: false,
      },
      {
        text: "Both necessarily increase inference latency by a factor of 10× or more.",
        isCorrect: false,
      },
    ],
    explanation:
      "Adapters and LoRA are PEFT methods that leave base weights intact and add small task-specific modules; they do not require discarding the base model.",
  },
  {
    id: "ch3-q56",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements correctly relate RLHF and DPO?",
    options: [
      {
        text: "Both use human preference data to steer model behavior toward preferred responses.",
        isCorrect: true,
      },
      {
        text: "DPO replaces the RL stage of RLHF with a supervised-style loss directly on the LM probabilities.",
        isCorrect: true,
      },
      {
        text: "Both require training a separate reward model that scores responses.",
        isCorrect: false,
      },
      {
        text: "RLHF never uses supervised fine-tuning as an initial step.",
        isCorrect: false,
      },
    ],
    explanation:
      "RLHF uses SFT + reward model + RL; DPO uses preference pairs with a direct loss. Both rely on preference data but DPO avoids a separate reward model.",
  },
  {
    id: "ch3-q57",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements about why companies care about copyright in LLM training data are correct?",
    options: [
      {
        text: "Training on copyrighted text without permission may infringe on authors’ rights depending on jurisdiction.",
        isCorrect: true,
      },
      {
        text: "Models can sometimes reproduce long passages similar to their training data, raising legal concerns.",
        isCorrect: true,
      },
      {
        text: "Copyright law explicitly exempts all AI training uses in every country.",
        isCorrect: false,
      },
      {
        text: "Licensing agreements with content owners are impossible.",
        isCorrect: false,
      },
    ],
    explanation:
      "Copyright questions arise because models may memorize and reproduce protected text; regulations are evolving and some companies pursue licensing deals.",
  },
  {
    id: "ch3-q58",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe how CLIP embeddings can be used after training?",
    options: [
      {
        text: "Given an image and several textual labels, we can classify the image by choosing the label whose embedding is most similar to the image embedding.",
        isCorrect: true,
      },
      {
        text: "We can retrieve images relevant to a text query by ranking images by similarity to the query embedding.",
        isCorrect: true,
      },
      {
        text: "We can reconstruct the original image pixel-by-pixel from its CLIP embedding.",
        isCorrect: false,
      },
      {
        text: "We can obtain exact captions automatically without any additional model.",
        isCorrect: false,
      },
    ],
    explanation:
      "CLIP is powerful for similarity-based retrieval and zero-shot classification, but it does not reconstruct images or generate fluent captions by itself.",
  },
  {
    id: "ch3-q59",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements about VLMs like BLIP-2 are correct?",
    options: [
      {
        text: "They can answer natural-language questions about images by combining visual and textual understanding.",
        isCorrect: true,
      },
      {
        text: "They typically reuse strong pretrained components (a ViT and an LLM) plus a newly trained connector module.",
        isCorrect: true,
      },
      {
        text: "They must be trained entirely from scratch with randomly initialized vision and language backbones.",
        isCorrect: false,
      },
      {
        text: "They cannot generate captions for images.",
        isCorrect: false,
      },
    ],
    explanation:
      "BLIP-2 demonstrates that we can reuse powerful pretrained vision and language models and connect them with a trainable bridge to enable captioning and visual QA.",
  },
  {
    id: "ch3-q60",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about latent representations in diffusion models like Stable Diffusion are correct?",
    options: [
      {
        text: "The U-Net operates on a lower-dimensional latent space rather than full-resolution pixel space.",
        isCorrect: true,
      },
      {
        text: "The decoder maps the final latent back to an image in pixel space.",
        isCorrect: true,
      },
      {
        text: "The latent representation is always a single scalar per image.",
        isCorrect: false,
      },
      {
        text: "Operating in latent space makes training dramatically more expensive than working directly on pixels.",
        isCorrect: false,
      },
    ],
    explanation:
      "Stable Diffusion runs the denoising process in a latent space to reduce cost; the latent is still a tensor, not a scalar, and this design speeds training.",
  },
  {
    id: "ch3-q61",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly distinguish factuality and faithfulness hallucinations?",
    options: [
      {
        text: "Factual hallucinations are statements that do not match real-world facts.",
        isCorrect: true,
      },
      {
        text: "Faithfulness hallucinations contradict the provided instructions or context even if they might be factually true elsewhere.",
        isCorrect: true,
      },
      {
        text: "Both types necessarily involve offensive language.",
        isCorrect: false,
      },
      {
        text: "Hallucinations only occur when the model refuses to answer.",
        isCorrect: false,
      },
    ],
    explanation:
      "Factuality concerns correctness with respect to the world; faithfulness concerns adherence to instructions and context. Neither requires toxicity.",
  },
  {
    id: "ch3-q62",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements about representational and allocational harm are correct?",
    options: [
      {
        text: "Representational harm involves harmful or stereotyped portrayals of groups in model outputs.",
        isCorrect: true,
      },
      {
        text: "Allocational harm arises when biased models influence decisions that allocate opportunities or resources.",
        isCorrect: true,
      },
      {
        text: "Allocational harm is irrelevant when using models for hiring or credit scoring.",
        isCorrect: false,
      },
      {
        text: "Representational harm is always harmless because it affects only language, not people.",
        isCorrect: false,
      },
    ],
    explanation:
      "Harmful representations can shape attitudes and decisions, while biased decision support systems can unfairly allocate resources.",
  },
  {
    id: "ch3-q63",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements about zero-shot and few-shot capabilities of LLMs are correct?",
    options: [
      {
        text: "Instruction-tuned LLMs often perform surprisingly well in zero-shot settings.",
        isCorrect: true,
      },
      {
        text: "Few-shot prompts can further improve performance by demonstrating the desired task.",
        isCorrect: true,
      },
      {
        text: "Zero-shot capabilities require retraining the model from scratch for each new task.",
        isCorrect: false,
      },
      {
        text: "Few-shot learning always requires thousands of labeled examples.",
        isCorrect: false,
      },
    ],
    explanation:
      "Zero-shot and few-shot learning exploit pretraining and instruction tuning; a handful of examples in the prompt can be enough.",
  },
  {
    id: "ch3-q64",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the role of format in in-context learning?",
    options: [
      {
        text: "Consistent input–output formatting in the prompt helps the model infer the mapping between them.",
        isCorrect: true,
      },
      {
        text: "Even when labels are simple, specifying them clearly in the prompt can improve performance.",
        isCorrect: true,
      },
      {
        text: "The model completely ignores formatting cues in the prompt.",
        isCorrect: false,
      },
      {
        text: "Randomizing formatting for each example usually improves ICL performance.",
        isCorrect: false,
      },
    ],
    explanation:
      "Regular, clear formatting gives the model a strong signal about how inputs and outputs relate, which improves ICL.",
  },
  {
    id: "ch3-q65",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about the computational cost of advanced prompting methods (CoT, self-consistency, ToT) are correct?",
    options: [
      {
        text: "They typically require more forward passes because they generate multiple reasoning paths or candidate solutions.",
        isCorrect: true,
      },
      {
        text: "For a fixed model, their wall-clock inference time per query is usually higher than simple greedy decoding.",
        isCorrect: true,
      },
      {
        text: "They always reduce cost compared with using a smaller base model.",
        isCorrect: false,
      },
      {
        text: "They make decoding completely independent of model size.",
        isCorrect: false,
      },
    ],
    explanation:
      "Reasoning-enhancing prompting schemes often sample multiple paths or explore trees, increasing compute relative to simple single-path decoding.",
  },
  {
    id: "ch3-q66",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about the advantages of using a small proxy model when studying scaling laws are correct?",
    options: [
      {
        text: "We can test architectural ideas cheaply before scaling them to hundred-billion parameter models.",
        isCorrect: true,
      },
      {
        text: "Observed power-law trends on small models can guide expectations about larger models’ performance.",
        isCorrect: true,
      },
      {
        text: "Proxy models always perfectly predict the behavior of arbitrarily larger models.",
        isCorrect: false,
      },
      {
        text: "Results on small models are irrelevant when deciding how to scale up.",
        isCorrect: false,
      },
    ],
    explanation:
      "Small models provide experimental guidance and reveal trends but are approximations; extrapolation to giant models is useful but imperfect.",
  },
  {
    id: "ch3-q67",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe why under-trained LLMs may be considered 'data-limited' rather than 'capacity-limited'?",
    options: [
      {
        text: "They have more parameters than needed for the amount of data they see, so they could benefit from additional tokens.",
        isCorrect: true,
      },
      {
        text: "Increasing token count for a fixed-size model can continue to reduce loss according to scaling laws.",
        isCorrect: true,
      },
      {
        text: "Increasing model size without increasing data always guarantees better performance.",
        isCorrect: false,
      },
      {
        text: "Data-limited models are defined as those trained on infinite data.",
        isCorrect: false,
      },
    ],
    explanation:
      "If a model is under-trained relative to size, more data helps; simply scaling parameters without more data can be suboptimal.",
  },
  {
    id: "ch3-q68",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about SparseGPT-style pruning results are correct at a high level?",
    options: [
      {
        text: "Careful pruning methods can sometimes remove a large fraction of weights (for example over 50%) while retaining good performance.",
        isCorrect: true,
      },
      {
        text: "Naive aggressive pruning often leads to model collapse or severe degradation.",
        isCorrect: true,
      },
      {
        text: "SparseGPT guarantees no loss of performance regardless of pruning level.",
        isCorrect: false,
      },
      {
        text: "Pruning is unnecessary for large models because they never contain redundant parameters.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sophisticated pruning such as SparseGPT can compress large models significantly, but success depends on method choice and pruning ratio.",
  },
  {
    id: "ch3-q69",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the role of KL divergence in RLHF?",
    options: [
      {
        text: "A KL penalty term keeps the updated policy close to the original supervised model.",
        isCorrect: true,
      },
      {
        text: "Without such a penalty, the model might drift into strange behaviors that are poorly supported by the base distribution.",
        isCorrect: true,
      },
      {
        text: "The KL term measures similarity between the reward model and the training dataset.",
        isCorrect: false,
      },
      {
        text: "The KL term is unnecessary if we never care about preserving base model behavior.",
        isCorrect: false,
      },
    ],
    explanation:
      "The KL penalty is computed between the new policy distribution and the reference model distribution so RLHF does not move too far from the base model’s behavior.",
  },
  {
    id: "ch3-q70",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about machine unlearning for LLMs are correct at a conceptual level?",
    options: [
      {
        text: "It aims to remove or reduce the influence of specific training examples or groups from a trained model.",
        isCorrect: true,
      },
      {
        text: "Potential regulations may require models to forget personal data when users request it.",
        isCorrect: true,
      },
      {
        text: "Unlearning is trivial because we can simply delete text from the training corpus after training.",
        isCorrect: false,
      },
      {
        text: "Once a model is trained, it is mathematically impossible to alter what it has learned.",
        isCorrect: false,
      },
    ],
    explanation:
      "Machine unlearning tries to retroactively limit the influence of certain data; deleting raw text is insufficient and exact unlearning is challenging but not impossible in principle.",
  },
  {
    id: "ch3-q71",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the few-shot learning paper 'Language Models are Few-Shot Learners'?",
    options: [
      {
        text: "It showed that sufficiently large autoregressive LMs can perform many tasks given only a few prompt examples.",
        isCorrect: true,
      },
      {
        text: "It popularized the terminology of zero-shot, one-shot, and few-shot prompting.",
        isCorrect: true,
      },
      {
        text: "The models in that work were trained specifically as supervised multi-task classifiers.",
        isCorrect: false,
      },
      {
        text: "It concluded that prompt design has no effect on performance.",
        isCorrect: false,
      },
    ],
    explanation:
      "The GPT-3 paper demonstrated strong ICL with pure language modeling pretraining and highlighted the importance of prompt formatting.",
  },
  {
    id: "ch3-q72",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe zero-shot chain-of-thought prompting ('LLMs are Zero-Shot Reasoners')?",
    options: [
      {
        text: "Adding phrases like “Let’s think step by step” can cause the model to produce explicit reasoning before answering.",
        isCorrect: true,
      },
      {
        text: "This technique can improve performance on some reasoning benchmarks without providing worked examples.",
        isCorrect: true,
      },
      {
        text: "It requires re-training the model with a special loss function.",
        isCorrect: false,
      },
      {
        text: "It removes the need for careful evaluation of reasoning quality.",
        isCorrect: false,
      },
    ],
    explanation:
      "Zero-shot CoT uses prompt cues to elicit reasoning behavior; it does not involve retraining but still needs careful evaluation.",
  },
  {
    id: "ch3-q73",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly characterize DSPy’s use of metrics and optimizers?",
    options: [
      {
        text: "Users specify evaluation metrics or objectives for the overall pipeline (for example, accuracy or F1).",
        isCorrect: true,
      },
      {
        text: "DSPy optimizers adjust prompts or module parameters to maximize these metrics on a development set.",
        isCorrect: true,
      },
      {
        text: "DSPy forbids any automated modification of prompts.",
        isCorrect: false,
      },
      {
        text: "Metrics are irrelevant because DSPy guarantees optimal prompts analytically.",
        isCorrect: false,
      },
    ],
    explanation:
      "DSPy treats prompt design as an optimization problem driven by explicit evaluation metrics and automated optimizers.",
  },
  {
    id: "ch3-q74",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements about the interaction of alignment techniques and base capabilities are correct?",
    options: [
      {
        text: "Alignment methods such as RLHF or DPO rely on the underlying pretrained model already having strong general capabilities.",
        isCorrect: true,
      },
      {
        text: "Poor base models cannot be turned into strong general reasoners solely through alignment on small preference datasets.",
        isCorrect: true,
      },
      {
        text: "Alignment always increases raw language modeling accuracy.",
        isCorrect: false,
      },
      {
        text: "Alignment replaces the need for large-scale pretraining.",
        isCorrect: false,
      },
    ],
    explanation:
      "Alignment primarily steers existing capabilities; it cannot substitute for large-scale pretraining and may trade off raw LM loss for safety or helpfulness.",
  },
  {
    id: "ch3-q75",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the relationship between emergent abilities and task difficulty?",
    options: [
      {
        text: "Harder reasoning tasks often appear to have higher critical scales than simpler tasks.",
        isCorrect: true,
      },
      {
        text: "Some narrow abilities, like simple arithmetic, may emerge at smaller model sizes than complex multi-step reasoning.",
        isCorrect: true,
      },
      {
        text: "All tasks share the same critical scale regardless of their complexity.",
        isCorrect: false,
      },
      {
        text: "Task difficulty is unrelated to when abilities appear in scaling experiments.",
        isCorrect: false,
      },
    ],
    explanation:
      "Different tasks show emergent behavior at different scales; harder tasks often require larger models to reach non-trivial performance.",
  },

  // --------------------------------------------------------------------------------
  // Q76–Q100: exactly one option is correct (pattern: 1 correct)
  // --------------------------------------------------------------------------------

  {
    id: "ch3-q76",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statement best describes what 'logits' are in the context of LLM outputs?",
    options: [
      {
        text: "Unnormalized real-valued scores for each vocabulary token before applying softmax.",
        isCorrect: true,
      },
      {
        text: "The final probabilities after softmax.",
        isCorrect: false,
      },
      {
        text: "Binary indicators of whether a token is correct or not.",
        isCorrect: false,
      },
      {
        text: "The hidden states of the transformer layers.",
        isCorrect: false,
      },
    ],
    explanation:
      "Logits are the raw scores produced by the output layer; softmax turns them into probabilities.",
  },
  {
    id: "ch3-q77",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statement correctly describes the main purpose of instruction tuning?",
    options: [
      {
        text: "To make the model better at following natural-language instructions across many tasks.",
        isCorrect: true,
      },
      {
        text: "To replace pretraining so that no large-scale language modeling is needed.",
        isCorrect: false,
      },
      {
        text: "To reduce model size by pruning unused parameters.",
        isCorrect: false,
      },
      {
        text: "To convert an encoder-only model into a decoder-only model.",
        isCorrect: false,
      },
    ],
    explanation:
      "Instruction tuning continues training on instruction–response pairs so the model better follows instructions, complementing but not replacing pretraining.",
  },
  {
    id: "ch3-q78",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statement correctly describes why MoE models are called 'sparse'?",
    options: [
      {
        text: "For each token, only a small subset of experts is activated, so computation is sparse over experts.",
        isCorrect: true,
      },
      {
        text: "Most of the token positions in the input sequence are discarded.",
        isCorrect: false,
      },
      {
        text: "The model has no parameters in its attention layers.",
        isCorrect: false,
      },
      {
        text: "The model always uses sparsity-inducing L1 regularization on every weight.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sparsity refers to activating only a few experts per token, not discarding tokens or removing attention parameters.",
  },
  {
    id: "ch3-q79",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statement correctly explains why long context windows are computationally expensive with standard self-attention?",
    options: [
      {
        text: "The attention matrix has size T×T for sequence length T, so both time and memory scale quadratically in T.",
        isCorrect: true,
      },
      {
        text: "The model must retrain from scratch for every new context length.",
        isCorrect: false,
      },
      {
        text: "The embeddings cannot be reused across different context lengths.",
        isCorrect: false,
      },
      {
        text: "The softmax function is undefined for long sequences.",
        isCorrect: false,
      },
    ],
    explanation:
      "Self-attention computes all pairwise token interactions, giving an O(T²) cost in time and memory.",
  },
  {
    id: "ch3-q80",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statement correctly describes a key difference between RLHF and simple supervised fine-tuning?",
    options: [
      {
        text: "RLHF uses a learned reward model and RL algorithm (such as PPO) to optimize policy behavior based on human preferences.",
        isCorrect: true,
      },
      {
        text: "RLHF only minimizes token-level cross-entropy on ground-truth text.",
        isCorrect: false,
      },
      {
        text: "Supervised fine-tuning never uses human-written examples.",
        isCorrect: false,
      },
      {
        text: "RLHF does not require any human feedback.",
        isCorrect: false,
      },
    ],
    explanation:
      "RLHF optimizes via a reward model and RL, going beyond cross-entropy on fixed outputs.",
  },
  {
    id: "ch3-q81",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statement best captures the risk of using only synthetic data when training new generations of LLMs?",
    options: [
      {
        text: "The model may progressively lose diversity and accuracy, a phenomenon sometimes called model collapse.",
        isCorrect: true,
      },
      {
        text: "The model will automatically debias itself relative to human-written text.",
        isCorrect: false,
      },
      {
        text: "Synthetic data always contains more information than human data.",
        isCorrect: false,
      },
      {
        text: "Using synthetic data guarantees perfect factual accuracy.",
        isCorrect: false,
      },
    ],
    explanation:
      "Feeding models mostly their own outputs can amplify errors and reduce diversity, leading to collapse.",
  },
  {
    id: "ch3-q82",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statement correctly describes an advantage of quantization-aware training over naive post-training quantization?",
    options: [
      {
        text: "The model is trained while simulating low-precision effects, allowing it to adapt weights to be more robust to quantization.",
        isCorrect: true,
      },
      {
        text: "It removes the need to store any parameters at inference time.",
        isCorrect: false,
      },
      {
        text: "It guarantees that quantized weights are always exactly equal to the original float weights.",
        isCorrect: false,
      },
      {
        text: "It forces the model to use only binary weights of ±1.",
        isCorrect: false,
      },
    ],
    explanation:
      "Quantization-aware training incorporates quantization effects into training, typically yielding better accuracy than naive post-training quantization.",
  },
  {
    id: "ch3-q83",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement correctly characterizes why structured pruning can be more hardware-friendly than unstructured pruning?",
    options: [
      {
        text: "Structured pruning removes whole channels, heads, or blocks, so the resulting tensors are still dense and efficiently handled by standard kernels.",
        isCorrect: true,
      },
      {
        text: "Structured pruning always results in random sparsity patterns that are hard to exploit.",
        isCorrect: false,
      },
      {
        text: "Unstructured pruning guarantees speedups on all hardware without any special support.",
        isCorrect: false,
      },
      {
        text: "Structured pruning requires a custom sparse-matrix library for every deployment.",
        isCorrect: false,
      },
    ],
    explanation:
      "By dropping entire structures like channels or layers, structured pruning keeps dense tensor shapes, which existing BLAS kernels can accelerate.",
  },
  {
    id: "ch3-q84",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement correctly explains why alignment alone cannot fix all hallucinations?",
    options: [
      {
        text: "The underlying model may simply lack the knowledge or tools needed to verify facts, so no preference training can guarantee truthfulness.",
        isCorrect: true,
      },
      {
        text: "Alignment always gives the model access to a perfect world-knowledge database.",
        isCorrect: false,
      },
      {
        text: "Alignment changes the architecture from a transformer to a recurrent network.",
        isCorrect: false,
      },
      {
        text: "Alignment ensures the model never outputs uncertain or approximate answers.",
        isCorrect: false,
      },
    ],
    explanation:
      "Alignment can reduce harmful or unhelpful outputs but cannot give models capabilities (like reliable verification) they fundamentally do not have.",
  },
  {
    id: "ch3-q85",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement best describes a key motivation for DSPy-style programmatic prompting?",
    options: [
      {
        text: "To replace ad-hoc manual prompt tinkering with systematic, optimizable prompt and module configurations.",
        isCorrect: true,
      },
      {
        text: "To prevent any use of evaluation metrics when designing prompts.",
        isCorrect: false,
      },
      {
        text: "To ensure that all prompts are fixed and unchangeable.",
        isCorrect: false,
      },
      {
        text: "To train new LLMs from scratch without prompts.",
        isCorrect: false,
      },
    ],
    explanation:
      "DSPy views prompt design as a declarative program plus optimization, reducing reliance on manual trial-and-error.",
  },
  {
    id: "ch3-q86",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statement correctly describes why chain-of-thought prompting can help arithmetic word problems?",
    options: [
      {
        text: "It encourages the model to decompose the problem into intermediate steps instead of jumping directly to an answer.",
        isCorrect: true,
      },
      {
        text: "It forces the model to use exact symbolic algebra internally.",
        isCorrect: false,
      },
      {
        text: "It removes the need to check whether the final numeric answer is reasonable.",
        isCorrect: false,
      },
      {
        text: "It changes the loss function used during pretraining.",
        isCorrect: false,
      },
    ],
    explanation:
      "By writing out intermediate steps, the model is nudged toward more structured reasoning, which often improves accuracy on arithmetic questions.",
  },
  {
    id: "ch3-q87",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statement correctly describes a common way to use CLIP inside text-to-image models?",
    options: [
      {
        text: "Use the CLIP text encoder to turn the user’s prompt into an embedding that conditions the diffusion process.",
        isCorrect: true,
      },
      {
        text: "Use CLIP to generate the final image directly without any diffusion or decoder.",
        isCorrect: false,
      },
      {
        text: "Use CLIP to quantize the model weights.",
        isCorrect: false,
      },
      {
        text: "Use CLIP to prune attention heads.",
        isCorrect: false,
      },
    ],
    explanation:
      "Stable Diffusion-like models often reuse CLIP’s text encoder to provide a rich semantic embedding of the prompt for the denoising U-Net.",
  },
  {
    id: "ch3-q88",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement correctly explains why emergent abilities complicate capability forecasting?",
    options: [
      {
        text: "Because performance on some tasks may remain low for a wide range of sizes and then improve rapidly, making extrapolation from smaller models unreliable.",
        isCorrect: true,
      },
      {
        text: "Because emergent abilities ensure linear improvement with model size on all tasks.",
        isCorrect: false,
      },
      {
        text: "Because emergent abilities only occur in computer vision models.",
        isCorrect: false,
      },
      {
        text: "Because emergent abilities guarantee that future models will be weaker.",
        isCorrect: false,
      },
    ],
    explanation:
      "If some abilities appear only beyond certain scales, forecasts based solely on small models may underestimate future capabilities.",
  },
  {
    id: "ch3-q89",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement correctly describes a reason to care about irreducible loss in scaling laws?",
    options: [
      {
        text: "It represents a lower bound due to data entropy, so further scaling cannot reduce loss below this floor.",
        isCorrect: true,
      },
      {
        text: "It can always be eliminated by adding more parameters.",
        isCorrect: false,
      },
      {
        text: "It measures the training instability of the optimizer.",
        isCorrect: false,
      },
      {
        text: "It directly equals the number of parameters in the model.",
        isCorrect: false,
      },
    ],
    explanation:
      "Irreducible loss is the portion we cannot beat given the data distribution; knowing it helps set realistic expectations for scaling.",
  },
  {
    id: "ch3-q90",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement correctly explains why long contexts may not always guarantee better performance?",
    options: [
      {
        text: "If the model cannot reliably focus on the truly relevant parts, extra context can introduce distraction or noise.",
        isCorrect: true,
      },
      {
        text: "Self-attention becomes linear in sequence length, making training trivial.",
        isCorrect: false,
      },
      {
        text: "Long contexts force the model to memorize the entire training set.",
        isCorrect: false,
      },
      {
        text: "Models always use all contextual information optimally.",
        isCorrect: false,
      },
    ],
    explanation:
      "Longer context only helps if the model can identify and use the relevant pieces; otherwise it may add clutter.",
  },
  {
    id: "ch3-q91",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statement correctly describes a benefit of using small specialized models in production?",
    options: [
      {
        text: "They can be cheaper and faster while still performing well on a narrow, well-defined task.",
        isCorrect: true,
      },
      {
        text: "They automatically outperform general LLMs on every possible task.",
        isCorrect: false,
      },
      {
        text: "They remove the need for any domain-specific evaluation.",
        isCorrect: false,
      },
      {
        text: "They cannot be fine-tuned.",
        isCorrect: false,
      },
    ],
    explanation:
      "Focused small models can be efficient and strong for specific use-cases, though they are not universal solutions.",
  },
  {
    id: "ch3-q92",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statement correctly defines 'preference data' in the context of RLHF and DPO?",
    options: [
      {
        text: "Human judgments that rank or compare multiple candidate responses for the same prompt.",
        isCorrect: true,
      },
      {
        text: "Random tokens sampled from the model with no human involvement.",
        isCorrect: false,
      },
      {
        text: "Only binary labels indicating whether an answer is grammatically correct.",
        isCorrect: false,
      },
      {
        text: "Pairs of prompts without any outputs.",
        isCorrect: false,
      },
    ],
    explanation:
      "Preference data asks humans to compare different responses and indicate which they prefer, giving richer signals than simple correctness labels.",
  },
  {
    id: "ch3-q93",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statement correctly describes a benefit of using evaluation libraries such as Hugging Face Evaluate when auditing models?",
    options: [
      {
        text: "They provide ready-made metrics (for example toxicity, sentiment) that can be applied systematically across prompts.",
        isCorrect: true,
      },
      {
        text: "They guarantee that a model has no bias.",
        isCorrect: false,
      },
      {
        text: "They replace the need for any human review of outputs.",
        isCorrect: false,
      },
      {
        text: "They prevent models from being trained on biased data.",
        isCorrect: false,
      },
    ],
    explanation:
      "Tools like Evaluate help quantify properties such as toxicity but do not themselves fix or guarantee absence of bias.",
  },
  {
    id: "ch3-q94",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement correctly explains a limitation of prompt-only approaches compared with proper fine-tuning?",
    options: [
      {
        text: "Prompting cannot change the model’s underlying parameters, so it may be unable to learn new capabilities that require weight updates.",
        isCorrect: true,
      },
      {
        text: "Prompting always yields strictly better performance than fine-tuning.",
        isCorrect: false,
      },
      {
        text: "Prompting directly edits the training dataset.",
        isCorrect: false,
      },
      {
        text: "Prompting is incompatible with in-context learning.",
        isCorrect: false,
      },
    ],
    explanation:
      "Prompting can steer behavior but cannot imbue fundamentally new skills the base model does not already support.",
  },
  {
    id: "ch3-q95",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement correctly characterizes the main risk of allocational harm from biased LLM-based decision support?",
    options: [
      {
        text: "Certain groups may systematically receive fewer opportunities or worse outcomes because recommendations reflect biased patterns.",
        isCorrect: true,
      },
      {
        text: "Only the user interface colors are affected.",
        isCorrect: false,
      },
      {
        text: "It only matters for toy examples and not for real-world decisions.",
        isCorrect: false,
      },
      {
        text: "It occurs only when models are perfectly fair.",
        isCorrect: false,
      },
    ],
    explanation:
      "Allocational harm concerns how decisions impact people’s access to opportunities and resources, which can be heavily affected by biased recommendations.",
  },
  {
    id: "ch3-q96",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement correctly captures why measuring hallucinations is challenging?",
    options: [
      {
        text: "It often requires external knowledge or tools to verify whether model statements are actually true or supported by evidence.",
        isCorrect: true,
      },
      {
        text: "Hallucinations can be detected simply by checking whether the output is fluent.",
        isCorrect: false,
      },
      {
        text: "Any disagreement between users and the model is automatically a hallucination.",
        isCorrect: false,
      },
      {
        text: "Hallucinations can be measured only on arithmetic problems.",
        isCorrect: false,
      },
    ],
    explanation:
      "Determining whether an answer is hallucinated needs ground-truth references or tools, which can be expensive or domain-specific.",
  },
  {
    id: "ch3-q97",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement correctly explains a key reason why long-context models are interesting for retrieval-augmented generation (RAG)?",
    options: [
      {
        text: "They can condition on many retrieved passages at once, reducing the need for strict pre-filtering.",
        isCorrect: true,
      },
      {
        text: "They eliminate the need for any retrieval.",
        isCorrect: false,
      },
      {
        text: "They replace embeddings with raw documents.",
        isCorrect: false,
      },
      {
        text: "They make it impossible to include user instructions.",
        isCorrect: false,
      },
    ],
    explanation:
      "Long contexts let RAG systems stuff more retrieved evidence into a single prompt, which can improve answer quality.",
  },
  {
    id: "ch3-q98",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement correctly describes one reason why emergent abilities may be sensitive to the choice of evaluation metric?",
    options: [
      {
        text: "Coarse metrics (for example accuracy thresholds) can hide gradual improvements and then suddenly flip from 'fail' to 'pass' near a certain performance level.",
        isCorrect: true,
      },
      {
        text: "Metrics never affect how we perceive performance trends.",
        isCorrect: false,
      },
      {
        text: "Continuous metrics always produce step-like jumps.",
        isCorrect: false,
      },
      {
        text: "Metrics are irrelevant for scaling-law analyses.",
        isCorrect: false,
      },
    ],
    explanation:
      "Discontinuous-looking curves can arise from thresholded or coarse metrics even when underlying performance changes smoothly.",
  },
  {
    id: "ch3-q99",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement correctly describes a high-level purpose of using optimizers in DSPy?",
    options: [
      {
        text: "To search over prompt templates and module configurations that maximize a chosen evaluation metric.",
        isCorrect: true,
      },
      {
        text: "To remove the need for any evaluation data.",
        isCorrect: false,
      },
      {
        text: "To guarantee human-level understanding of every prompt.",
        isCorrect: false,
      },
      {
        text: "To force the model to ignore instructions.",
        isCorrect: false,
      },
    ],
    explanation:
      "DSPy optimizers treat prompt and program choices as parameters to be tuned against a metric on validation data.",
  },
  {
    id: "ch3-q100",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement correctly characterizes the relationship between model size, data quality, and performance?",
    options: [
      {
        text: "Scaling parameters helps only if we also provide enough high-quality data and compute; poor-quality or insufficient data can limit returns from larger models.",
        isCorrect: true,
      },
      {
        text: "Model size alone fully determines performance regardless of training data.",
        isCorrect: false,
      },
      {
        text: "Any data, even highly noisy or synthetic-only data, always has the same effect on performance.",
        isCorrect: false,
      },
      {
        text: "Data quality is irrelevant once the model exceeds 100 billion parameters.",
        isCorrect: false,
      },
    ],
    explanation:
      "Scaling laws and empirical results show that parameter count, token count, and data quality all matter; simply making models bigger without adequate good data is suboptimal.",
  },
];
