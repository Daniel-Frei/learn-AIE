# Lecture 9 Curriculum: Course Synthesis, Beyond Transformer-Based LLMs, and Future Directions

Source materials:

- Transcript: `lecture 9 - transcript.md`
- Slides: `lecture 9 - slides.pdf`

## Course Role

This final lecture consolidates the whole course, shows how transformer ideas migrate beyond text, introduces diffusion-style language modeling, and frames ongoing research directions, product uses, limitations, and learning habits after the class.

## Learning Objectives

By the end, students should be able to:

- Reconstruct the course arc from tokenization and self-attention through LLM training, preference tuning, reasoning, agents, and evaluation.
- Explain how transformers can process images by turning image patches into token-like vectors and using encoder-style architectures such as ViT.
- Compare VLM design patterns: recycling decoder-only LLMs with visual tokens versus using cross-attention to condition on visual representations.
- Explain the autoregressive modeling bottleneck: inference-time generation is serial even if training can be parallelized.
- Explain the intuition behind diffusion for images and masked diffusion for text.
- Describe masked diffusion language models as iteratively unmasking tokens rather than generating strictly left-to-right.
- Identify cross-pollination between modalities: diffusion ideas for text, transformers for images, visual token representations, and 2D RoPE.
- Explain why foundational transformer research remains active in optimizers, normalization, MHA/MQA/GQA design, activation functions, MoE choices, layer counts, and data quality.
- Explain model collapse risk from training on low-diversity LLM-generated data and the motivation for curated data and mid-training.
- Reason about performance-cost Pareto frontiers, small language models, hardware specialization, and energy/latency constraints.
- Identify ongoing challenges: fixed weights, hallucinations as next-token behavior rather than fact mapping, personalization, interpretability, and safety.
- Plan how to stay current through papers, code, venues, blogs, videos, and the course study guide.

## Prerequisite Assumptions

Students should have completed the prior lectures or have equivalent understanding of transformer architecture, model families, LLM training, preference tuning, reasoning models, agents, and evaluation.

## Curriculum Sequence

### 1. Rewind the Quarter as a Concept Map

Start with the recap: lecture 1 tokenization, embeddings, RNN limitations, self-attention, and encoder-decoder transformers; lecture 2 transformer variants and BERT/T5/GPT families; lecture 3 decoder-only LLMs, MoE, decoding, prompting, and inference; lecture 4 pretraining/SFT/LoRA/QLoRA; lecture 5 preference tuning; lecture 6 reasoning; lecture 7 RAG/tools/agents; lecture 8 evaluation.

Active learning:

- Have students build a dependency graph connecting tokenization -> attention -> decoder-only LLMs -> training -> post-training -> reasoning/tools/evaluation.
- Ask which earlier concept is reused most often in later lectures.

Assessment targets:

- Students can explain how a single user-facing LLM answer depends on architecture, training, decoding, post-training, tools, and evaluation.

### 2. Generalize Transformers Beyond Text

Teach why self-attention's weaker inductive bias allows use beyond language. Explain Vision Transformer: divide an image into patches, project patches into vectors, add position information, process with transformer encoder, and use a `[CLS]`-like representation for classification.

Active learning:

- Convert a small image grid into patch tokens and identify where position information is needed.
- Compare text tokens and image patches: what is analogous and what is different?

Assessment targets:

- Students can explain why ViT resembles BERT-like encoder use more than decoder-only generation.
- Students can explain why images need 2D positional structure.

### 3. Introduce Vision-Language Models

Teach two VLM patterns from the slides: feed visual tokens/features into a decoder-only architecture, or use cross-attention where a decoder conditions on visual encoder outputs. Emphasize that multimodal systems reuse core transformer ideas rather than being entirely separate.

Active learning:

- Given an image-question-answer task, draw a high-level architecture using visual tokens and a language decoder.
- Ask where cross-attention would be useful.

Assessment targets:

- Students can distinguish image understanding, image generation, and vision-language answering.
- Students can explain why visual instruction tuning is needed for useful multimodal behavior.

### 4. Contrast Autoregressive and Diffusion-Style Generation

Explain the bottleneck of autoregressive modeling: tokens are generated sequentially at inference time. Then review diffusion intuition from images: sample noise, learn a denoising/reverse process, and iteratively transform noise toward data. Translate that to text: masking tokens is the text analog of adding noise, and unmasking is the reverse process.

Active learning:

- Compare a left-to-right generation trace with a masked-diffusion trace that fills multiple tokens over fewer passes.
- Ask which tasks might benefit from non-left-to-right generation.

Assessment targets:

- Students can explain why autoregressive inference is hard to parallelize.
- Students can explain masked diffusion models (MDM/DLLM) without assuming continuous image noise.

### 5. Study Cross-Pollination Between Modalities

Teach the trend that modalities trade ideas. Examples: diffusion training for text output, transformers inside image generation, vision-token compression/representation ideas, DeepSeek-OCR-style visual tokens representing text, and adapting RoPE to 2D image or multimodal grids.

Active learning:

- Match each borrowed idea to its source modality and target modality.
- Ask what must change when RoPE moves from 1D text positions to 2D image positions.

Assessment targets:

- Students can explain why architecture ideas are not locked to one data type.
- Students can identify input representation as a research frontier, not just model architecture.

### 6. Return to Foundational Research

Teach that basic design choices remain unsettled: AdamW versus Muon/MuonClip-like optimizers, post-norm/pre-norm/RMSNorm choices, MHA/MQA/GQA patterns, activation functions, MoE decisions, layer count and head count, and whether the transformer is the best long-term architecture.

Active learning:

- Pick one design axis and ask what tradeoff it affects: stability, memory, speed, quality, scaling, or interpretability.
- Ask why modern papers still differ on these choices.

Assessment targets:

- Students can name several "basic" components that remain active research areas.
- Students avoid treating the current transformer stack as final.

### 7. Treat Data as a Core Research Problem

Explain how the early internet had more human-generated data, while current web data increasingly includes LLM-generated text. Teach model collapse as a risk when generated data is less diverse and shifts the training distribution. Introduce data curation and mid-training as responses.

Active learning:

- Ask students to identify signs that a dataset might contain synthetic or low-diversity content.
- Design a data pipeline step that improves quality before training.

Assessment targets:

- Students can explain why more data is not always better.
- Students can explain why mid-training on higher-quality corpora can matter.

### 8. Optimize for Cost, Latency, and Hardware

Teach the shift from pure benchmark maximization toward quality/cost Pareto frontiers. Cover small language models, inference economics, latency, energy, and hardware specialization. Use analog in-memory attention as an example of encoding transformer operations more directly in hardware.

Active learning:

- Choose a model from a hypothetical quality/cost frontier for a high-volume product versus a low-volume expert workflow.
- Ask why memory movement, not just matrix multiplication, motivates new hardware.

Assessment targets:

- Students can interpret a Pareto frontier and avoid choosing models by quality alone.
- Students can explain why hardware design may need to specialize for attention and KV movement.

### 9. Connect to Real Uses and Open Problems

Discuss current and expected uses: coding, text-to-query, visualization, general assistants, creative drafting, learning support, democratized agents, browser/OS-level assistants, and customer service. Then identify open problems: continuous learning, hallucinations, personalization, interpretability, safety, reliability, and security.

Active learning:

- Pick a real application and identify which open problems block trustworthy deployment.
- Ask whether RAG/tools solve fixed weights or only work around them.

Assessment targets:

- Students can explain hallucination as a consequence of next-token prediction not being direct fact verification.
- Students can distinguish product potential from unresolved safety/reliability constraints.

### 10. Build a Post-Course Learning Practice

Close by teaching how to stay current: arXiv computation and language, NeurIPS/ICML/ICLR/ACL/EMNLP, authors' GitHub repos, Hugging Face trending papers, social research communities, paper-explainer videos, company blogs, the VIP cheatsheet, and the Super Study Guide.

Active learning:

- Have students choose one source channel for papers, one for code, and one for practitioner updates.
- Ask students to write a monthly learning loop: find paper, read abstract/figures, inspect code, reproduce one tiny example, update notes.

Assessment targets:

- Students can sustain learning after the course without relying only on viral model announcements.

## Misconceptions to Address

- Transformers are not only for text; they are a general attention-based architecture pattern.
- Diffusion for text does not mean adding Gaussian image-like noise to words; masking/unmasking is the discrete analog.
- Faster generation is not automatically better if quality, controllability, or evaluation degrade.
- Current LLM architecture choices are not settled facts.
- Synthetic data is not automatically bad, but distribution shift and diversity loss must be managed.
- RAG and tools work around fixed weights but do not create true continuous learning by themselves.

## Assessment Blueprint

Use synthesis and transfer:

- Draw a full course concept map.
- Compare text transformers, ViT, and VLM architectures.
- Compare autoregressive and masked-diffusion language generation.
- Identify cross-modal borrowed ideas and what changes in the transfer.
- Interpret quality/cost Pareto tradeoffs.
- Diagnose whether a future-looking claim is an architecture, data, training, evaluation, hardware, or product issue.
- Build a personal research-following plan.

## Follow-Up Practice

- Read a Vision Transformer explainer and map every component to the transformer vocabulary from lectures 1-2.
- Read one masked diffusion language model abstract and write a paragraph comparing it to next-token prediction.
- Track one current frontier topic for a month: optimizers, data curation, diffusion LLMs, VLMs, agent safety, or hardware for attention.
