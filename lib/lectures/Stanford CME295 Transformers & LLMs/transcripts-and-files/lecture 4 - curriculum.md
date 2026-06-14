# Lecture 4 Curriculum: LLM Training, Scaling, Fine-Tuning, and PEFT

Source materials:

- Transcript: `lecture 4 - transcript.md`
- Slides: `lecture 4 - slides.pdf`

## Course Role

This lecture explains how an LLM moves from initialized weights to a pretrained model, then to an instruction-following assistant, and finally to a model that can be adapted efficiently. It is the training and systems bridge between model architecture and preference tuning.

## Learning Objectives

By the end, students should be able to:

- Explain the modern LLM training paradigm: pretrain broadly, then tune for desired tasks and behavior.
- Define FLOPs versus FLOP/s and use them to reason about training cost.
- Explain scaling laws, sample efficiency, and the Chinchilla-style token/parameter tradeoff.
- Identify the main training memory objects: parameters, activations, gradients, and optimizer state.
- Compare data parallelism, ZeRO stages, and model-parallel families such as tensor, pipeline, sequence, and context parallelism.
- Explain FlashAttention as exact attention that reduces slow-memory reads/writes through tiling and recomputation.
- Explain mixed precision training and why low precision can improve speed and memory while retaining high-precision weights.
- Explain supervised fine-tuning, instruction tuning, benchmark evaluation, and Chatbot Arena-style human preference evaluation.
- Explain LoRA, where it is applied, why it trains fewer parameters, and how adapter swapping supports multiple tasks.
- Explain QLoRA, NF4 quantization, double quantization, and the memory-quality tradeoff.

## Prerequisite Assumptions

Students should know decoder-only LLMs, next-token prediction, attention, FFN layers, gradients/backpropagation, optimizer state at a basic level, and GPU memory as a constrained resource.

## Curriculum Sequence

### 1. Establish the LLM Training Lifecycle

Contrast traditional task-specific ML with LLM training. In traditional ML, each task gets a model. In the LLM paradigm, a model first learns broad language/code patterns through pretraining, then gets tuned for specific tasks or assistant behavior.

Active learning:

- Ask students to place pretraining, SFT/instruction tuning, preference tuning, and PEFT on a lifecycle diagram.
- Ask what capabilities a pretrained autocomplete model has before instruction tuning.

Assessment targets:

- Students can explain why pretraining alone does not produce a helpful assistant.
- Students can distinguish "learn language/code patterns" from "follow instructions."

### 2. Teach Pretraining and Scaling

Cover next-token pretraining on large data mixtures: web-scraped text, Wikipedia-like knowledge sources, code, and other corpora. Introduce FLOPs and FLOP/s, then scaling laws and the Chinchilla insight that models can be undertrained if parameter count grows faster than token count.

Active learning:

- Given a parameter count, ask students to compute a Chinchilla-style rough token target using a 20x token-to-parameter rule of thumb.
- Ask what happens when compute budget is fixed and model size/data size are imbalanced.

Assessment targets:

- Students can define FLOPs and FLOP/s without mixing them.
- Students can explain why "bigger model" is not automatically compute-optimal.
- Students can explain knowledge cutoff as a consequence of fixed pretraining data.

### 3. Identify Training Bottlenecks

Trace the training loop: initialization, forward pass, loss, activations, backward pass, gradients, optimizer state, and parameter update. Explain why memory is a bottleneck even on large GPUs.

Active learning:

- Label which training objects must be stored for forward, backward, and update.
- Ask students which objects grow with model size, batch size, and context length.

Assessment targets:

- Students can explain why activations matter for backpropagation.
- Students can explain why Adam optimizer state multiplies memory pressure.

### 4. Parallelize Training

Teach data parallelism and ZeRO as reducing redundant state across devices. Then teach model parallelism: tensor, pipeline, sequence, and context parallelism as ways to split computation or sequence/context across devices.

Active learning:

- Give four GPUs and ask what each stores under ordinary data parallelism, ZeRO-1, ZeRO-2, and ZeRO-3.
- Ask when splitting the batch is insufficient and model parallelism becomes necessary.

Assessment targets:

- Students can compare ZeRO stages by which objects are sharded.
- Students can distinguish data parallelism from model parallelism.

### 5. Optimize Attention and Precision

Teach FlashAttention through the GPU memory hierarchy: HBM is large and slow, SRAM is small and fast. Explain tiling, avoiding materializing the full attention matrix, and recomputation in backward pass. Then teach FP32/FP16/BF16-style precision and mixed precision training.

Active learning:

- Ask students why doing more FLOPs through recomputation can still reduce runtime.
- Compare storing a full attention matrix versus computing blocks in SRAM.
- Identify which quantities stay high precision and which can be lower precision in mixed precision training.

Assessment targets:

- Students understand FlashAttention as exact, not an approximation.
- Students can explain why memory movement can dominate runtime.
- Students can explain the purpose of keeping master weights in higher precision.

### 6. Supervised Fine-Tuning and Instruction Tuning

Teach SFT as training on input/output pairs using the same next-token objective, and instruction tuning as SFT on instruction-following data to "graduate" a pretrained model into an assistant. Cover human-written and synthetic data, task mixtures, and evaluation challenges.

Active learning:

- Turn a raw instruction-answer pair into an autoregressive next-token training sequence.
- Ask students why high-quality SFT data matters more than sheer quantity.

Assessment targets:

- Students can explain the difference between pretraining examples and instruction-tuning examples.
- Students can name benchmark dimensions such as knowledge, reasoning, math, code, and real-life arena-style preference.

### 7. Parameter-Efficient Fine-Tuning With LoRA

Teach LoRA as freezing the pretrained weight matrix `W0` and learning a low-rank update `BA`. Emphasize that only a fraction of parameters are trained, adapters can be swapped for tasks, and modern guidance differs from the original placement experiments.

Active learning:

- Compute the trainable parameter count for a full matrix versus rank-r LoRA matrices.
- Ask students why LoRA may need different learning-rate and batch-size choices than full fine-tuning.

Assessment targets:

- Students can explain low-rank adaptation without treating it as compression of the base model.
- Students can explain adapter swapping as task-specific deltas over a shared base.

### 8. QLoRA and Quantized Fine-Tuning

Teach QLoRA as quantizing frozen base weights to relieve VRAM bottlenecks while training LoRA matrices in higher precision. Explain NF4 as quantization designed for normally distributed weights, and double quantization as quantizing quantization constants for extra savings.

Active learning:

- Compare memory layouts for full fine-tuning, LoRA, and QLoRA.
- Ask why computations may still happen in higher precision even when frozen weights are stored quantized.

Assessment targets:

- Students can explain why QLoRA enables fine-tuning large models on smaller GPUs.
- Students can explain the memory-quality tradeoff and the idea of double quantization.

## Misconceptions to Address

- Pretraining is not the same as instruction tuning.
- FLOPs and FLOP/s are different: amount of work versus rate of work.
- FlashAttention is not sparse attention; it computes exact attention more efficiently.
- Lower precision is not universally safe; mixed precision is a managed compromise.
- LoRA does not update the full base matrix; it learns a low-rank delta while freezing the base.
- QLoRA is not just LoRA with smaller adapters; it quantizes frozen base weights.

## Assessment Blueprint

Use calculation, diagnosis, and comparison:

- Compute token/parameter ratios and identify undertraining.
- Identify memory bottlenecks from a training setup.
- Compare ZeRO stages and model-parallel strategies.
- Explain why FlashAttention can be faster despite recomputation.
- Convert SFT data into next-token prediction format.
- Compute LoRA parameter savings for a given matrix shape and rank.
- Explain NF4 and double quantization conceptually.

## Follow-Up Practice

- Read FlashAttention's introduction with a GPU memory hierarchy diagram.
- Work a LoRA parameter-count example for Q, K, V, O, and FFN matrices.
- Create a lifecycle table: pretraining, SFT/instruction tuning, preference tuning, LoRA/QLoRA adaptation, evaluation.
