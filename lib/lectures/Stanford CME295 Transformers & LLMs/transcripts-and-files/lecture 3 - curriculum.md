# Lecture 3 Curriculum: Large Language Models, MoE, Response Generation, Prompting, and Inference

Source materials:

- Transcript: `lecture 3 - transcript.md`
- Slides: `lecture 3 - slides.pdf`

## Course Role

This lecture introduces the course's practical definition of an LLM and explains how decoder-only models generate responses, scale capacity with mixture-of-experts layers, respond to prompting strategies, and serve tokens efficiently.

## Learning Objectives

By the end, students should be able to:

- Define an LLM as a large decoder-only transformer-based language model that assigns probabilities to token sequences and generates text autoregressively.
- Explain "large" in terms of parameters, training tokens, and compute.
- Explain dense versus sparse mixture-of-experts, token-level routing, active parameters, routing collapse, auxiliary load-balancing losses, and noisy gating.
- Compare greedy decoding, beam search, sampling, top-k, top-p, temperature scaling, and guided decoding.
- Explain context length/window size and why prompt structure affects behavior.
- Distinguish zero-shot, few-shot, chain-of-thought, and self-consistency prompting.
- Explain key inference optimizations: KV caching, MHA/GQA/MQA, PagedAttention, latent attention, speculative decoding, and multi-token prediction.

## Prerequisite Assumptions

Students should understand decoder-only transformer structure, masked self-attention, probability distributions over vocabulary tokens, softmax, and basic model family taxonomy.

## Curriculum Sequence

### 1. Define the Modern LLM

Start by contrasting BERT-style encoder-only models with decoder-only text-to-text models. Define a language model as assigning probabilities to token sequences, then define LLM scale by parameters, training data, and compute. Emphasize that the course treats GPT, LLaMA, Gemma, DeepSeek, Mistral, and Qwen-like models as the core examples.

Active learning:

- Ask whether BERT, T5, and GPT are LLMs under the lecture's working definition and why.
- Give model descriptions and ask students to identify whether they are encoder-only, encoder-decoder, or decoder-only.

Assessment targets:

- Students can justify why decoder-only autoregressive generation is central to the LLM definition used in this course.

### 2. Scale Capacity With Mixture-of-Experts

Teach MoE using the transcript's expert-room metaphor: not every question needs every expert. Explain dense MoE versus sparse MoE, the gate/router, top-k expert selection, and why modern transformer MoE layers usually replace or augment the FFN rather than attention. Explain active versus total parameters.

Active learning:

- Route a batch of example tokens to experts and identify which experts are active.
- Ask where to insert experts in a decoder block and why the FFN is the likely target.
- Diagnose a routing-collapse trace where one expert receives most tokens.

Assessment targets:

- Students can explain why MoE can increase total model capacity without proportionally increasing per-token inference FLOPs.
- Students can explain auxiliary load-balancing loss at the conceptual level.

### 3. Generate the Next Token

Trace autoregressive next-token prediction from `[BOS]` to a growing prefix. Explain that the model outputs probabilities over the vocabulary at each step and that decoding policy determines how those probabilities become tokens.

Active learning:

- Given a toy probability distribution, generate one token using greedy, top-k, top-p, and temperature-scaled sampling.
- Ask students to predict how low and high temperature change diversity.

Assessment targets:

- Students can compare deterministic and stochastic decoding.
- Students can explain why beam search can still produce unnatural or non-optimal outputs.
- Students can explain guided decoding as constraining the valid next-token set for a required output format.

### 4. Prompt for Behavior

Teach prompt structure, context length, in-context learning, zero-shot/few-shot prompting, chain-of-thought prompting, and self-consistency. Emphasize the transcript's point that self-consistency samples multiple independent reasoning paths in parallel, extracts final answers, and uses majority vote.

Active learning:

- Turn a zero-shot prompt into a few-shot prompt.
- Ask students when chain-of-thought is worth the extra tokens.
- Run a conceptual self-consistency vote over several candidate answers.

Assessment targets:

- Students can explain why examples in the prompt can change behavior without changing weights.
- Students can explain the latency/cost tradeoff of chain-of-thought and self-consistency.

### 5. Avoid Redundant Work During Inference

Teach KV caching: previous keys and values can be stored because the current token attends to prior tokens, but prior queries are not needed for the new token. Then connect GQA/MQA to reducing the number of stored K/V heads.

Active learning:

- Ask which tensors must be recomputed and which can be cached when generating token `t+1`.
- Compare KV cache size under MHA, GQA, and MQA.

Assessment targets:

- Students can explain why KV caching is an inference-time technique rather than a standard training-time technique.
- Students can connect grouped query attention to memory savings.

### 6. Manage Memory and Compress Attention State

Introduce PagedAttention as memory management for KV cache blocks instead of reserving a full maximum context window per request. Then introduce latent attention as storing compressed token representations and decompressing for keys/values.

Active learning:

- Show two requests with short outputs and ask how naive full-window reservation wastes memory.
- Ask students to compare PagedAttention and latent attention: one manages allocation, the other changes what is stored.

Assessment targets:

- Students can explain internal memory fragmentation in serving.
- Students can distinguish exact memory management from representation compression.

### 7. Speed Token Generation

Teach speculative decoding with a small draft model and a large target model, including accept/reject logic that preserves the target distribution. Then teach multi-token prediction as embedding the draft-like heads in the same model and changing the training objective.

Active learning:

- Trace a speculative decoding example where some draft tokens are accepted and one is rejected.
- Ask why a single target-model pass over draft tokens can be faster than serial big-model generation.

Assessment targets:

- Students can explain draft model, target model, acceptance, rejection, and fallback.
- Students can explain how multi-token prediction differs from ordinary next-token prediction.

## Misconceptions to Address

- "Large" is not only parameter count; data and compute also matter.
- MoE total parameters and active parameters are different quantities.
- Sampling randomness is not inherently bad; it enables diverse outputs and preference-data generation.
- Temperature does not choose top tokens directly; it reshapes the probability distribution.
- Context length is not the same as "model understanding" of every token; retrieval and attention quality still matter.
- KV caching saves previous K/V projections, not all previous computation.

## Assessment Blueprint

Use applied traces:

- Define an LLM and reject near-miss definitions.
- Diagnose MoE routing collapse and propose a mitigation.
- Decode from a toy distribution under greedy, top-k, top-p, and temperature.
- Write a guided decoding constraint for JSON-like output.
- Trace KV cache reuse during generation.
- Compare PagedAttention, latent attention, speculative decoding, and multi-token prediction by what resource each targets.

## Follow-Up Practice

- Read the Switch Transformer, PagedAttention, and speculative decoding summaries.
- Implement a paper-and-pencil decoding simulator with temperature/top-k/top-p.
- Build a one-page "LLM serving bottlenecks" table: compute, memory, latency, quality, exactness, and approximation.
