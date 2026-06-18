# Stanford CME295 Transformers & LLMs Course Topics

This inventory lists the topics covered across the Stanford CME295 Transformers & LLMs course, organized by lecture.
It was synthesized from the matching transcript and slide sources in this folder, with the existing per-lecture curriculum files used as cross-checks.

## Source Coverage

- Lecture 1: `lecture 1 - transformers.md`, `lecture 1 - transformers.pdf`, `lecture 1 - curriculum.md`
- Lecture 2: `lecture 2 - transcript.md`, `lecture 2 - slides.pdf`, `lecture 2 - curriculum.md`
- Lecture 3: `lecture 3 - transcript.md`, `lecture 3 - slides.pdf`, `lecture 3 - curriculum.md`
- Lecture 4: `lecture 4 - transcript.md`, `lecture 4 - slides.pdf`, `lecture 4 - curriculum.md`
- Lecture 5: `lecture 5 - transcript.md`, `lecture 5 - slides.pdf`, `lecture 5 - curriculum.md`
- Lecture 6: `lecture 6 - transcript.md`, `lecture 6 - slides.pdf`, `lecture 6 - curriculum.md`
- Lecture 7: `lecture 7 - transcript.md`, `lecture 7 - slides.pdf`, `lecture 7 - curriculum.md`
- Lecture 8: `lecture 8 - transcript.md`, `lecture 8 - slides.pdf`, `lecture 8 - curriculum.md`
- Lecture 9: `lecture 9 - transcript.md`, `lecture 9 - slides.pdf`, `lecture 9 - curriculum.md`

## Course-Level Arc

- Foundations: natural language processing tasks, tokenization, token representations, sequence models, self-attention, and transformer architecture.
- Transformer model families: encoder-decoder, encoder-only, and decoder-only variants; BERT/T5/GPT-style use cases.
- Modern LLMs: decoder-only language modeling, mixture-of-experts, decoding, prompting, and inference optimization.
- Training and tuning: pretraining, scaling laws, distributed training, FlashAttention, mixed precision, SFT, LoRA, QLoRA, preference tuning, RLHF, PPO, Best-of-N, and DPO.
- Reasoning and systems: reasoning models, test-time scaling, GRPO, RAG, tool calling, agents, MCP, A2A, and agent safety.
- Evaluation and frontiers: human ratings, rule-based metrics, LLM-as-a-Judge, factuality, benchmark interpretation, multimodal transformers, diffusion-style language models, data quality, hardware, cost, safety, interpretability, and future learning practice.

## Lecture 1: Transformers

Topics covered:

- Course orientation: course goals, expected ML and linear algebra background, class resources, the study guide, the VIP cheat sheet, source citations on slides, and the abbreviation-heavy nature of the field.
- NLP task taxonomy: text classification, token/entity-level prediction, and text-to-text generation.
- Classification examples and metrics: sentiment extraction, intent detection, language detection, topic modeling, accuracy, precision, recall, F1, and class imbalance.
- Token/entity-level examples and metrics: named entity recognition, part-of-speech tagging, dependency parsing, constituency parsing, token-level aggregation, and entity-type-level evaluation.
- Generation examples and metrics: machine translation, question answering, summarization, code generation, poem/creative generation, BLEU, ROUGE, METEOR, perplexity, and the need for reference text in many older metrics.
- NLP history and motivation: RNNs, LSTMs, the role of internet-scale data and compute, Word2vec, the 2017 transformer paper, and the scaling path toward LLMs.
- Tokenization: arbitrary token units, word-level tokenization, subword tokenization, BPE and WordPiece examples, character-level tokenization, out-of-vocabulary behavior, root sharing, sequence length, and compute tradeoffs.
- Token representations: one-hot vectors, learned embeddings, embedding lookup tables, semantic similarity, and proxy-task learning.
- Word2vec: learning embeddings from neighboring-word prediction, center/context word examples, hidden embedding layers, and limits of context-independent representations.
- Sequence models before transformers: recurrent processing, hidden state, LSTM motivation, long-range dependency limits, and limited parallelism.
- Self-attention: direct token-to-token links, queries, keys, values, query-key dot products, attention weights, softmax, scaled dot-product attention, and value-weighted sums.
- Transformer architecture: token embeddings, positional information, encoder self-attention, decoder masked self-attention, encoder-decoder cross-attention, feed-forward networks, stacked encoder/decoder blocks, and softmax output over the vocabulary.
- Translation generation loop: beginning-of-sequence and end-of-sequence tokens, decoder autoregression, masking future target tokens, using encoder keys/values with decoder queries, and iterative next-token generation.
- Multi-head attention: multiple learned query/key/value projections, parallel heads, concatenation, output projection, and representational capacity.
- Label smoothing: replacing hard one-hot labels with softened targets, epsilon mass over non-target vocabulary items, reduced overconfidence, and translation-metric improvements.

## Lecture 2: Transformer-Based Models and Tricks

Topics covered:

- Attention-head recap: attention maps, interpreting what different heads attend to, projection matrices per head, and caution about treating attention maps as complete causal explanations.
- Position information: why self-attention loses order information, learned absolute position embeddings, hardcoded sinusoidal embeddings, and the dot-product relationship between positions.
- Relative position modeling: why attention cares about relative distance, relative position bias, T5-style bucketed bias, ALiBi linear attention bias, and the intuition that farther tokens should decay in attention relevance.
- Rotary position embeddings: RoPE rotations applied to query/key vectors, pairwise 2D rotations within the embedding dimension, relative-distance capture, and modern use as a default position strategy.
- Layer normalization: LayerNorm, post-norm transformer blocks, pre-norm transformer blocks, pre-norm with RMSNorm, and stability/architecture tradeoffs.
- Attention approximations for long context: sparse attention, Longformer, sliding window attention, local attention, global tokens such as `[CLS]`, and the cost/coverage tradeoff relative to full attention.
- Attention-head sharing: multi-head attention, multi-query attention, grouped-query attention, key/value head sharing, and implications for KV cache size and inference cost.
- Transformer model families: encoder-decoder models for text-to-text tasks, encoder-only models for representation and classification tasks, and decoder-only models for autoregressive generation.
- Encoder-decoder examples: original transformer translation setup, T5, mT5, ByT5, and text-to-text framing.
- Encoder-only examples: BERT, DistilBERT, RoBERTa, ALBERT, encoded embeddings, and classification-style downstream tasks.
- Decoder-only examples: GPT-style models and why decoder-only architecture becomes central for later LLMs.
- BERT motivation: bidirectional encoder representations, comparison to ELMo/contextual embeddings, and why BERT keeps only the encoder stack.
- BERT token/input construction: WordPiece tokenization, `[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`, cased versus uncased variants, token embeddings, position embeddings, and segment embeddings.
- BERT pretraining tasks: masked language modeling, the 80/10/10 mask/random/unchanged pattern, next sentence prediction, and segment A/B classification context.
- BERT fine-tuning: using the `[CLS]` output for sentence classification, using token outputs for span/question-answering tasks, and optionally freezing or tuning pretrained weights.
- BERT limitations and variants: 512-token context limits, latency/parameter cost, MLM/NSP training complexity, distillation to smaller student models, KL-divergence matching, DistilBERT, RoBERTa removal of NSP/segment encodings, dynamic masking, and larger/more diverse pretraining data.

## Lecture 3: Large Language Models, MoE, Response Generation, Prompting, and Inference

Topics covered:

- LLM definition: language models as probability models over token sequences, next-token prediction, model size, training-token scale, compute scale, and why modern LLMs are usually decoder-only transformer models.
- Decoder-only characteristics: removal of the encoder and cross-attention, masked self-attention for autoregressive generation, and examples including GPT, LLaMA, Gemma, DeepSeek, Mistral, and Qwen.
- Mixture-of-experts motivation: avoiding activation of every parameter for every input, active versus total parameters, and scaling model capacity without proportional inference cost.
- MoE designs: dense MoE, sparse MoE, top-k expert selection, experts as feed-forward sublayers inside transformer blocks, gate/router networks, token-level routing, and hardware-placement motivation.
- MoE training issues: routing collapse, load imbalance, auxiliary load-balancing losses, noisy or diversified routing, expert specialization, and expert-usage visualization.
- Autoregressive response generation: beginning-of-sequence prompts, predicting one token at a time, probability distributions over the vocabulary, and feeding generated tokens back into the decoder.
- Decoding strategies: greedy decoding, beam search, sampling, top-k sampling, top-p/nucleus sampling, and the tradeoff between likelihood, diversity, naturalness, and latency.
- Temperature: rescaling logits/probabilities, low-temperature deterministic behavior, high-temperature diversity, and how temperature changes distribution sharpness.
- Guided/constrained decoding: generating structured formats, only allowing valid next tokens, JSON-like constraints, schema enforcement, and format reliability.
- Context and prompting: context length/window size, prompt structure, examples in the prompt, zero-shot prompting, few-shot prompting, and the cost of using more prompt tokens.
- Reasoning-oriented prompting: chain-of-thought prompting, asking for intermediate reasoning, self-consistency, multiple sampled reasoning paths, answer extraction, and majority voting.
- Inference efficiency framing: exact methods versus approximate methods, attention-level optimizations versus output-token-level optimizations, memory-bound inference, and avoiding redundant computation.
- KV caching: storing prior keys and values during autoregressive decoding, why prior queries are not reused, and why teacher-forced training does not need KV caching in the same way.
- KV cache size reduction: reusing MQA/GQA ideas for shared key/value heads, memory savings, and the link between attention-head sharing and serving cost.
- PagedAttention: KV cache memory fragmentation, reserved memory, internal/external fragmentation, fixed-size cache blocks, vLLM-style paging, and higher request throughput.
- Latent/multi-latent attention: compressing key/value representations through a lower-dimensional latent space, sharing compressed state across heads and between keys/values, and reducing KV cache storage.
- Speculative decoding: draft and target models, generating candidate tokens with a smaller model, verifying them in parallel with the larger model, acceptance/rejection sampling, and preserving the target distribution.
- Multi-token prediction: training or attaching multiple output heads to predict several future tokens, using embedded draft heads, and accelerating generation with a modified objective.

## Lecture 4: LLM Training, Scaling, Fine-Tuning, and PEFT

Topics covered:

- Training paradigm shift: traditional task-specific training, transfer learning, pretraining a broadly useful model, and tuning it for downstream tasks.
- Pretraining: next-token prediction over massive text/code corpora, Common Crawl, Wikipedia, social media, GitHub, Stack Overflow, multilingual/code data, and training-token scale from hundreds of billions to trillions.
- Pretraining risks and constraints: training cost, time, environmental cost, knowledge cutoff, difficult knowledge editing, regression risk, memorization, and plagiarism risk.
- Compute notation: FLOPs as floating-point operations, FLOP/s as hardware throughput, and compute as a function of token count, parameter count, and architecture.
- Scaling laws: bigger data, bigger models, more compute, sample efficiency, compute-optimal model/data tradeoffs, Chinchilla-style undertraining analysis, and the rule of thumb relating parameters to training tokens.
- Training loop: initialization, forward pass, loss computation, backward pass, gradients, optimizer updates, and activation storage.
- Training memory objects: parameters, activations, gradients, optimizer states, and why training a large model stresses accelerator memory.
- Hardware framing: GPUs, TPUs, matrix multiplication, HBM, SRAM, memory bandwidth, and why memory movement can dominate runtime.
- Data parallelism: splitting batches across devices, redundant model copies, synchronization, and the motivation for ZeRO.
- ZeRO optimization: ZeRO-1 optimizer-state sharding, ZeRO-2 gradient sharding, ZeRO-3 parameter sharding, and memory savings.
- Model parallelism families: tensor parallelism, pipeline parallelism, sequence parallelism, context parallelism, and splitting large-model computations across devices.
- FlashAttention: exact attention with IO-aware tiling, loading blocks into SRAM, avoiding full attention-matrix materialization, recomputation in the backward pass, and reducing HBM reads/writes.
- Numerical precision: floating-point representation, FP32, FP16, BF16, mixed precision training, high-precision weights where needed, speed/memory tradeoffs, and precision-related stability.
- Supervised fine-tuning: adapting a pretrained model to task-specific or instruction-following data, prompt/response pairs, and the distinction between pretraining and tuning.
- Instruction tuning: SFT on instruction-following datasets, zero-shot behavior, benchmarks, and human-preference-oriented evaluation such as Chatbot Arena.
- Parameter-efficient fine-tuning: freezing base weights, learning small task-specific deltas, LoRA, rank selection, low-rank matrices, swapping adapters for tasks, and where LoRA can be applied.
- LoRA practical details: attention versus feed-forward insertion points, higher learning-rate behavior, weaker large-batch behavior, and comparison to prefix tuning/adapters.
- QLoRA: quantized base-model weights plus trainable LoRA matrices, 4-bit NF4 quantization, normally distributed weight assumptions, single versus double quantization, quantization constants, BF16 LoRA computation, and VRAM savings.

## Lecture 5: Preference Tuning, RLHF, PPO, Best-of-N, and DPO

Topics covered:

- Preference tuning motivation: SFT can follow tasks while still producing undesirable tone, unsafe behavior, brittle helpfulness, or responses that do not match human preferences.
- Preference data: pointwise, pairwise, and listwise ratings; prompts with multiple candidate responses; human labels; proxy labels; and pairwise data collection recipes.
- Why comparisons matter: preference comparison can be easier than writing an ideal answer from scratch, and preference data can express negative signals.
- RL mapping for LLMs: policy as the LLM, state as input so far, action as next token, environment as token sequence dynamics, reward as human preference or reward-model score, and probability of next token as the policy distribution.
- RLHF overview: train a reward model from preference data, then optimize the policy using the reward model.
- Reward modeling: Bradley-Terry pairwise preference formulation, probability that one response is preferred over another, reward-model scoring, pairwise loss, and RewardBench-style reward-model evaluation.
- Policy optimization with PPO: using reward scores to update the policy, advantage rather than raw reward, value-model baselines, old/reference/base models, and preventing excessive policy drift.
- Reward hacking and stability: why reward optimization can exploit reward-model weaknesses, why KL/reference constraints matter, and why exploration/diverse completions matter in RL.
- PPO variants: PPO-Clip, ratio clipping between new and old policies, PPO-KL penalty, KL divergence to a base/reference model, and the practical role of beta-like tradeoff coefficients.
- RL training mechanics: on-policy training, off-policy contrast with SFT, model-generated rollouts, and why PPO requires several model copies such as policy, value, reward, and reference/base models.
- PPO limitations: reward-model training cost, many hyperparameters, stability issues, heavy memory requirements, and operational complexity.
- Best-of-N reranking: generating multiple completions, scoring each with a reward model, returning the highest-scoring completion, avoiding RL training, and shifting cost/latency to inference.
- Best-of-N caveats: response quality depends on candidate diversity and reward-model quality; parallel generation still increases maximum-latency risk and serving cost.
- DPO motivation: direct supervised preference optimization without training a separate reward model or running an RL loop.
- DPO formulation: current policy, reference policy, preference pairs, beta as a KL tradeoff coefficient, sigmoid/logistic preference loss, and the insight that the language model can act as an implicit reward model.
- DPO derivation path: starting from the PPO-style reward/KL objective, solving for the optimal policy, rearranging reward in terms of policy, and plugging that into a Bradley-Terry preference model.
- RLHF versus DPO: PPO can perform better when tuned well, DPO is simpler and more stable to run, DPO can suffer distribution-shift issues, and SFT on preference-related data can affect results.
- Preference-tuned behavior examples: preserving factual content while changing tone, gentleness, safety, and helpfulness.

## Lecture 6: LLM Reasoning, Test-Time Scaling, GRPO, and Distillation

Topics covered:

- Vanilla LLM strengths and weaknesses: imitation, idea generation, code generation/debugging, limited reasoning, static knowledge, inability to act, and evaluation difficulty.
- Reasoning definition: solving a problem through intermediate steps rather than immediate answer emission.
- Chain-of-thought: prompting or training models to reason before answering, output as reasoning plus answer, reasoning-chain scale, hidden complete chains, and user-visible thought summaries.
- Reasoning-model trend: reasoning model releases, thinking modes, and why reasoning traces are usually hidden or summarized.
- Reasoning benchmarks: coding tasks, bug fixing, math/olympiad tasks, HumanEval, CodeForces, SWE-bench, AIME-style tasks, and ground-truth/verifiable answer setups.
- Reasoning metrics: Pass@k, Pass@1, Cons@k, majority-vote consensus, latency tradeoffs, and when each metric fits.
- Test-time scaling: spending more inference tokens, attempts, or thinking budget to improve answer quality.
- Verifiable rewards: formatting rewards for explicit reasoning delimiters, answer-correctness rewards from tests or exact answers, and why reasoning data is difficult to write by hand.
- Thinking controls: dynamic token budgets, context awareness, budget forcing, continuous/latent thoughts, and matching reasoning effort to prompt difficulty.
- GRPO: Group Relative Policy Optimization, group completions for the same prompt, rewards normalized relative to the group, no separate value model, and comparison to PPO.
- GRPO objective behavior: token-level policy ratios, clipping, KL to a reference, and summing over completions and output tokens.
- Increasing-output-length phenomenon: RL training can improve reasoning while making outputs longer, then continue increasing length after performance plateaus.
- Length-bias diagnosis: normalization by output length changes token contribution, causing different incentives for short versus long bad outputs.
- Length-bias mitigations: DAPO-style equalized token-level contribution, Dr. GRPO removal of the length factor, modified standard-deviation treatment for hard problems, and asymmetric epsilon bounds for low-probability tokens.
- DeepSeek R1-Zero: pretrained base model, reasoning RL without prior SFT, verifiable rewards, formatting rewards, emerging reasoning behavior, and issues such as language mixing and syntax/readability problems.
- DeepSeek R1 pipeline: cold-start SFT on cleaned reasoning traces, RL with verifiable and language-consistency rewards, larger SFT with reasoning and non-reasoning data, rejection sampling, final RL for helpfulness/harmlessness, and competitive reasoning benchmark results.
- Reasoning distillation: using a large reasoning teacher to generate reasoning traces, training smaller student models with SFT on those traces, and the distinction from classic probability-distribution distillation.

## Lecture 7: RAG, Tool Calling, and Agents

Topics covered:

- Motivation for system-augmented LLMs: knowledge cutoffs, changing world knowledge, difficult weight editing, limited context length, distraction from irrelevant context, token pricing, and inability to take actions.
- RAG overview: retrieval-augmented generation as retrieve, augment, generate; fetching relevant external information; and giving the answer inside the prompt context.
- Knowledge-base construction: document collection, chunking, chunk size, chunk overlap, embedding size, document structure awareness, and embedding chunks for retrieval.
- Retrieval architecture: candidate retrieval to maximize recall, reranking to improve final ordering, and the search/recommendation-style two-stage design.
- Candidate retrieval methods: semantic embedding search, cosine similarity, pre-trained embedding models, approximate nearest neighbor search, BM25 keyword search, and hybrid semantic/keyword retrieval.
- Retrieval extensions: mitigating query/document embedding mismatch, contextual retrieval for chunks, prompt caching, and reranking with more expensive models.
- Reranking: cross-encoders, query/chunk joint scoring, ranking smaller candidate sets, and improving top-ranked relevance.
- Retrieval evaluation: relevance judgments, rank position of relevant chunks, Reciprocal Rank at k, Normalized Discounted Cumulative Gain at k, and measuring retrieval quality separately from generation quality.
- Tool calling motivation: accessing structured databases, computation, APIs, and actions that cannot be solved by text generation alone.
- Tool-calling flow: describe available tools, predict tool name and arguments, execute backend function, observe tool output, and synthesize a natural-language response.
- Tool design concerns: tool descriptions, argument schemas, function-call formats, structured data access, and model-mediated API execution.
- Tool selection: reducing latency by selecting a subset of relevant tools, avoiding unnecessary tool overload, and automatic tool-selection methods.
- MCP: Model Context Protocol, standardization of tool/resource access, MCP hosts, clients, and servers.
- Agents: systems that pursue goals or complete tasks on a user's behalf, often using RAG and tool calling across multiple steps.
- ReAct: reason plus act loops, observe-plan-act structure, intermediate observations, repeated tool calls, and examples such as weather/thermostat or coding agents.
- Multi-agent systems: motivation for agent-to-agent communication, Agent2Agent/A2A standardization, AgentCard, AgentSkill, and AgentExecutor concepts.
- Agent safety and reliability: data exfiltration, unsafe tool access, inference safeguards, agent-safety benchmarks, transparency, observability, debuggability, and user trust.

## Lecture 8: LLM Evaluation

Topics covered:

- Evaluation scope: output quality, task performance, alignment, factuality, coherence, instruction following, system performance, latency, pricing, uptime, and reliability.
- Why LLM evaluation is hard: free-form text, code, math reasoning, multiple valid outputs, subjective criteria, and the lack of universal metrics.
- Human ratings: humans as the closest practical ground truth, rubrics, usefulness criteria, cost, slowness, subjectivity, and the need for rater alignment.
- Inter-rater agreement: agreement rate limitations, chance agreement baselines, Cohen's kappa, Fleiss' kappa, Krippendorff's alpha, and agreement sessions.
- Rule-based reference metrics: fixed reference answers, comparing model outputs to references, and iterating models without rerating every output.
- METEOR: precision/recall-style unigram matching, ordering penalty, contiguous chunks, hyperparameters, and use in translation.
- BLEU: n-gram precision, brevity penalty, translation evaluation, and limits of reference-based matching.
- ROUGE: recall-oriented summary evaluation, ROUGE-N/ROUGE-L-style variants, and summarization use.
- Limits of rule-based metrics: weak stylistic variation handling, imperfect correlation with human judgment, and the need for human-written references.
- LLM-as-a-Judge: using another LLM to grade responses, prompt/response/criteria inputs, rationale plus score outputs, and criteria-specific judging.
- Structured judge outputs: enforced output formats, response schemas, rationale-before-score prompting, binary scales, crisp guidelines, few-shot examples, and reproducibility concerns.
- Judge variants: pointwise grading, pairwise comparison, pass/fail scoring, preference-style comparison, and judge calibration.
- Judge biases: position bias, verbosity bias, self-enhancement bias, mitigation through swapping/averaging positions, explicit length rules, external judges, and careful prompts.
- Factuality evaluation: claim decomposition, checking long-form factuality, scoring individual claims, and aggregating factual-support judgments.
- Agentic workflow evaluation: decomposing failures into tool prediction errors, tool-call execution errors, and response-generation errors.
- Tool prediction failures: not using a tool, hallucinating a tool, choosing the wrong tool, and inferring wrong arguments.
- Benchmarks by capability: knowledge, math reasoning, common-sense reasoning, coding, safety, agents, and system-like benchmarks.
- Named benchmark examples: MMLU for knowledge, AIME for math reasoning, SWE-bench for software engineering, HarmBench for safety, and tau-bench for tool-agent-user interaction.
- Agent reliability metrics: pass^k as the probability that all k attempts succeed, contrasted with Pass@k as at least one success.
- Benchmark interpretation: model profiles, use-case fit, Pareto frontiers across quality/cost/latency, data contamination, hash/blocklist/new-test mitigations, Goodhart's law, Chatbot Arena, and not over-indexing on leaderboards.

## Lecture 9: Course Synthesis, Beyond Transformer-Based LLMs, and Future Directions

Topics covered:

- Course recap: tokenization, embeddings, Word2vec, RNNs/LSTMs, self-attention, transformers, position encodings, model families, LLMs, MoE, decoding, training, tuning, reasoning, agents, and evaluation.
- Transformers beyond text: using attention for other modalities and adapting token-like representations to images.
- Vision Transformer: image patches as tokens, patch projection, `[CLS]` token, positional embeddings, transformer encoder processing, and image classification.
- Vision-language models: visual tokens, visual instruction tuning, decoder-only LLMs conditioned on image tokens, and cross-attention approaches for vision-language answering.
- Multimodal distinctions: image understanding, image generation, and vision-language question answering.
- Autoregressive bottleneck: left-to-right inference, serial token generation, training/inference differences, and latency constraints.
- Diffusion for images: noising/denoising intuition, diffusion-style generation, and why image diffusion motivates alternatives to left-to-right language generation.
- Masked diffusion for text: masked diffusion models, diffusion LLMs, iterative token unmasking, fewer passes, and the discrete analog of diffusion rather than Gaussian noise over words.
- Cross-modal cross-pollination: transformers replacing convolutions in image systems, diffusion ideas moving into text, visual tokens as compact representations, OCR-like visual-token compression, and 2D RoPE.
- Foundational transformer research: optimizers such as AdamW and Muon/MuonClip, normalization choices, attention variants, activation functions, MoE decisions, layer count, attention heads, FFN dimensions, and unsettled design choices.
- Data as a frontier: human-generated versus synthetic data, LLM-generated data on the web, curation, high-quality corpora, mid-training, and model collapse from low-diversity synthetic data.
- Cost and deployment frontiers: quality/cost Pareto curves, small language models, serving economics, latency, energy, and choosing models by use case rather than headline benchmark score.
- Hardware specialization: GPU matrix multiplication, attention-specific memory movement, FlashAttention as evidence of memory bottlenecks, analog or attention-native hardware ideas, latency savings, and energy savings.
- Current and near-future use cases: coding assistants, text-to-query/text-to-code workflows, visualization generation, general assistants, web browsing agents, desktop/mobile agents, customer service, and learning support.
- Open problems: fixed weights, lack of continuous learning, hallucinations as next-token behavior rather than fact verification, personalization, interpretability, safety, agent reliability, prompt injection, and data exfiltration.
- Staying current: arXiv, NeurIPS and other venues, code repositories, Hugging Face trending papers, X/Twitter research discussions, YouTube explainers, company blogs, the course study guide, and building a personal paper/code/practice loop.
