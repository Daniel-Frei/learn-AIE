# Lecture 1 Curriculum: Transformers

Source materials:

- Transcript: `lecture 1 - transformers.md`
- Slides: `lecture 1 - transformers.pdf`

## Course Role

This lecture gives students the vocabulary and mechanism-level foundation for the rest of CME295. The curriculum should move from "how do models even process text?" to "how does a transformer translate or generate one token at a time?" The course logistics in the source can be treated as context; the durable learning curriculum begins with the NLP overview.

## Learning Objectives

By the end, students should be able to:

- Classify NLP tasks as text classification, token/entity-level prediction, or text-to-text generation, and choose suitable evaluation metrics.
- Explain tokenization as an arbitrary but consequential design choice, including word-level, subword, and character-level tradeoffs.
- Distinguish one-hot vectors from learned embeddings and explain why proxy tasks such as Word2vec create useful token representations.
- Explain why RNNs and LSTMs helped with sequences but struggle with long-range dependencies and parallelism.
- Interpret queries, keys, values, attention weights, and the scaled dot-product attention formula.
- Trace an encoder-decoder transformer from tokenization through embeddings, positional information, encoder self-attention, decoder masked self-attention, cross-attention, and softmax output.
- Explain why multi-head attention and label smoothing are useful training and architecture tricks.

## Prerequisite Assumptions

Students should know basic supervised learning, classification metrics, vectors and matrices, matrix multiplication, softmax, and the idea of gradient-based training. Do not assume they already know NLP abbreviations or transformer notation.

## Curriculum Sequence

### 1. Orient the Course Around NLP Tasks

Teach the three task families using the lecture examples: sentiment extraction and intent/language/topic classification; named entity recognition, part-of-speech tagging, and parsing; machine translation, question answering, summarization, and text generation. Connect each family to output shape and evaluation.

Active learning:

- Give three short product prompts and ask students to classify the task family, output type, and likely metric.
- Compare accuracy with precision, recall, and F1 on an imbalanced sentiment dataset.

Assessment targets:

- Students can say why accuracy is misleading under class imbalance.
- Students can say why BLEU/ROUGE need references and why perplexity measures model surprise rather than user usefulness.

### 2. Turn Text Into Model Inputs

Teach tokenization as cutting text into tokens, then compare arbitrary, word-level, subword, and character-level tokenization. Emphasize the "bear"/"bears" and "run"/"runs" style tradeoff from the transcript: subwords share roots but increase sequence length.

Active learning:

- Tokenize "A cute teddy bear is reading." three ways and compare vocabulary size, sequence length, out-of-vocabulary risk, and semantic sharing.
- Ask students to predict which tokenization choices increase attention cost.

Assessment targets:

- Students can explain why subword tokenization is a compromise between word-level interpretability and character-level coverage.
- Students connect longer token sequences to higher transformer compute.

### 3. Build Token Representations

Move from one-hot encodings to learned dense embeddings. Use Word2vec as the bridge: a neural network trains on a proxy task such as predicting a center/context/next word and learns an embedding layer as a useful intermediate representation.

Active learning:

- Have students compare two one-hot vectors and two learned embedding vectors and explain which one can express similarity.
- Trace one Word2vec-style training example and identify input token, hidden embedding, output distribution, and target.

Assessment targets:

- Students understand that embeddings are learned parameters, not hand-coded meanings.
- Students can explain why context-independent embeddings are useful but limited.

### 4. Motivate Attention From Sequence Limits

Teach RNNs as sequential processors that maintain a hidden state, then explain the long-range dependency problem and why attention originally helped translation systems attend to relevant source tokens directly.

Active learning:

- Give a long sentence with a pronoun and ask which earlier token must remain available.
- Compare the information path length in an RNN versus a direct attention link.

Assessment targets:

- Students can explain why recurrence gives order but makes long-distance signal propagation hard.
- Students can state the motivation for attention before seeing the transformer formula.

### 5. Explain Self-Attention Mechanically

Introduce query, key, and value as learned projections of token representations. Teach the formula `softmax(QK^T / sqrt(d_k))V` as three steps: score query-key compatibility, normalize scores into attention weights, then take a weighted average of values. Explain the `sqrt(d_k)` scaling as controlling dot-product magnitude as dimensionality grows.

Active learning:

- Use a tiny three-token matrix and ask students to identify rows of Q, columns of K, and attention-weight rows.
- Ask students to decide which tokens should receive high attention for a pronoun or noun phrase.

Assessment targets:

- Students can distinguish the role of Q/K from V.
- Students can explain why attention output is a weighted mixture of value vectors.
- Students do not treat attention heads as separate explicit grammar rules; they are different learned projections.

### 6. Assemble the Transformer

Trace the original encoder-decoder architecture for translation. The encoder contextualizes the source tokens. The decoder predicts the target sequence using masked self-attention over already generated target tokens and cross-attention over encoder outputs. Positional encodings add order information because direct attention links alone do not encode sequence order.

Active learning:

- Draw the source sentence, target prefix, encoder outputs, decoder masked self-attention, and cross-attention arrows.
- Ask which component supplies queries, keys, and values in encoder self-attention, decoder self-attention, and cross-attention.

Assessment targets:

- Students can say why decoder self-attention is masked.
- Students can explain that cross-attention uses decoder-side queries and encoder-side keys/values.
- Students can explain how generation stops at an end-of-sequence token.

### 7. Add Practical Transformer Tricks

Teach multi-head attention as parallel learned projections whose outputs are concatenated and projected back. Teach label smoothing as changing hard one-hot targets into slightly softer target distributions to reduce overconfidence in next-token prediction.

Active learning:

- Ask students why multiple heads can learn different associations even without explicit constraints.
- Convert a one-hot label into a smoothed label with epsilon spread over the non-target vocabulary.

Assessment targets:

- Students can connect multi-head attention to representational capacity.
- Students understand label smoothing as a target modification, not a softmax replacement.

## Misconceptions to Address

- Tokenization is not "just preprocessing"; it changes sequence length, vocabulary behavior, and downstream compute.
- A learned embedding is not a dictionary definition; it is a vector learned because it helps the proxy task.
- Attention is not automatically an explanation of the model's full behavior.
- Encoder-decoder transformers and decoder-only LLMs are related but not identical; decoder-only models are introduced later.
- Label smoothing does not mean the ground truth is unknown; it deliberately trains against less brittle targets.

## Assessment Blueprint

Use a mix of short conceptual checks and mechanism tracing:

- Classify an NLP task and choose a metric.
- Compare two tokenization strategies for a sentence containing inflections or rare words.
- Identify Q, K, V, attention weights, and values in a small attention computation.
- Trace which prior tokens the decoder can see at a specific generation step.
- Explain why positional information must be injected.
- Explain when BLEU/ROUGE/perplexity would or would not be adequate.

## Follow-Up Practice

- Read "Attention Is All You Need" with the lecture diagram open and annotate each block in the architecture.
- Work through a hand-sized self-attention example with 2-3 tokens.
- Build a one-page glossary of abbreviations introduced in the lecture: NLP, NER, BLEU, ROUGE, RNN, LSTM, Q/K/V, BOS, EOS, MHA.
