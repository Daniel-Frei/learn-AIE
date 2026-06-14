# Lecture 2 Curriculum: Transformer-Based Models and Tricks

Source materials:

- Transcript: `lecture 2 - transcript.md`
- Slides: `lecture 2 - slides.pdf`

## Course Role

This lecture turns the original transformer into the family of transformer-based models students will encounter in practice. It explains how position information, normalization, attention variants, and pretraining objectives evolved, then grounds the encoder-only branch through BERT.

## Learning Objectives

By the end, students should be able to:

- Explain why self-attention loses order information unless position information is injected.
- Compare learned absolute positions, sinusoidal positions, relative position bias, ALiBi, and RoPE at the level of what each changes.
- Explain why modern designs often inject position information directly into attention rather than only adding embeddings at the input.
- Distinguish post-norm, pre-norm, and RMS-style normalization choices at a practical architecture level.
- Compare full attention with sparse/local/long-context approximations and understand what is being saved.
- Explain MHA, MQA, and GQA as different query/key/value sharing patterns.
- Classify transformer models as encoder-decoder, encoder-only, or decoder-only and map them to T5, BERT, and GPT-like models.
- Trace BERT's tokenization, input representation, masked language modeling, next sentence prediction, fine-tuning, distillation, and RoBERTa-style changes.

## Prerequisite Assumptions

Students should understand the lecture 1 transformer diagram, Q/K/V self-attention, positional encoding as an input addition, softmax, classification heads, and basic fine-tuning.

## Curriculum Sequence

### 1. Revisit Attention Heads Through Attention Maps

Start with the lecture's recap of attention maps: each head has its own projections, and different heads can emphasize different relationships such as anaphora. Use this as the bridge from the original architecture to modern variants.

Active learning:

- Show a sentence with a pronoun and ask students which source tokens should receive attention.
- Ask why attention heads can differ even without a hard constraint forcing them to.

Assessment targets:

- Students can explain multi-head attention as parallel projection spaces, not repeated identical calculations.
- Students can interpret an attention map cautiously without overclaiming causal explanation.

### 2. Teach Position Information as an Attention Problem

Contrast learned absolute position embeddings with sinusoidal hardcoded embeddings. Use the transcript's trigonometric explanation: the dot product of sinusoidal position vectors depends on relative distance, so nearby positions can be represented as more similar. Then transition to relative bias, ALiBi, and RoPE as ways to affect attention scores directly.

Active learning:

- Ask students why learned positions cannot naturally extrapolate past the maximum training length.
- Have students identify which part of `softmax(QK^T / sqrt(d_k))V` relative position bias changes.
- Compare "add a vector to the token" with "rotate Q and K" and ask what computation each affects.

Assessment targets:

- Students can state the limitation of learned absolute positions.
- Students can explain why RoPE operates on Q and K.
- Students can distinguish relative distance from absolute index.

### 3. Survey Architecture and Efficiency Variants

Introduce normalization placement and type as stability/design choices. Then cover sparse attention and attention-head sharing as ways to scale longer contexts and reduce inference memory. Use MHA -> GQA -> MQA as a continuum of key/value sharing across query heads.

Active learning:

- Give a model with 32 query heads and ask how many K/V caches are needed under MHA, GQA, and MQA.
- Ask students what a local attention pattern gives up compared with full attention.

Assessment targets:

- Students understand that long-context efficiency can be exact or approximate depending on the technique.
- Students can connect GQA/MQA to KV cache size and inference cost.

### 4. Organize Transformer Model Families

Teach the three major families:

- Encoder-decoder models for text-to-text tasks such as T5.
- Encoder-only models for representations and classification-style tasks such as BERT.
- Decoder-only models for autoregressive generation such as GPT-like LLMs.

Active learning:

- Give task cards: sentiment classification, translation, chat completion, document retrieval embedding, and question answering. Students choose a model family and justify the choice.

Assessment targets:

- Students can say why encoder-only BERT is strong for representation tasks but not a natural free-form text generator.
- Students can say why decoder-only models are central to later LLM lectures.

### 5. Deep Dive BERT

Use BERT as the concrete encoder-only example. Cover WordPiece tokenization, `[CLS]`, `[SEP]`, padding, token embeddings, position embeddings, segment embeddings, stacked bidirectional encoders, masked language modeling, and next sentence prediction.

Active learning:

- Convert "This teddy bear is so cute" into a BERT-style input with `[CLS]`, WordPiece-like tokens, `[SEP]`, and padding.
- Mark which embedding components are added for each token.
- Ask which output embedding is used for sentence classification and which outputs are used for token/span prediction.

Assessment targets:

- Students can explain why the `[CLS]` vector can summarize a sequence after bidirectional self-attention.
- Students can explain the 80/10/10 masking idea for MLM.
- Students understand segment embeddings as learned sentence-A/sentence-B markers.

### 6. Discuss BERT Limitations and Variants

Close with limitations and variants: finite context length, latency and model size, the question of whether NSP is needed, DistilBERT-style teacher-student distillation, and RoBERTa-style removal of NSP with dynamic masking and more data.

Active learning:

- Ask students when they would prefer BERT, DistilBERT, or a decoder-only model.
- Compare hard-label cross-entropy with matching a teacher distribution through KL divergence.

Assessment targets:

- Students can explain what knowledge distillation transfers.
- Students can explain why RoBERTa's result casts doubt on NSP as a necessary objective.

## Misconceptions to Address

- "Position encoding" is not one technique; different designs affect different parts of attention.
- Sinusoidal positions are not learned, but they still encode a useful geometry.
- BERT is not an LLM under the course's later working definition because it is not a text-to-text generator.
- The `[CLS]` token is not magical before training; self-attention and objectives make its final embedding useful.
- Distillation is not just smaller architecture; it trains the student against teacher behavior.

## Assessment Blueprint

Prioritize comparison and mechanism questions:

- Compare learned positions, sinusoidal positions, ALiBI, and RoPE.
- Identify where relative bias enters the attention computation.
- Map task types to encoder-only, decoder-only, or encoder-decoder models.
- Trace a BERT input through WordPiece, embeddings, encoder output, and classification/span heads.
- Explain one benefit and one cost of GQA/MQA, sparse attention, distillation, and removing NSP.

## Follow-Up Practice

- Read the BERT paper and annotate the MLM and NSP objectives.
- Read the RoPE or ALiBi summaries and write a one-paragraph comparison to sinusoidal input embeddings.
- Build a table of model families with example models, training objective, natural tasks, and major limitations.
