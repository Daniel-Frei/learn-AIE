# CME295 2026 question coverage

The 2026 edition is a separate selectable series with four sources and 50 questions per source. Its lecture boundaries follow the supplied 2026 slides and transcripts rather than the legacy lecture numbers. Existing CME295 banks and database identities remain intact.

The paired slides and transcripts are stored under [the 2026 source folder](<../lib/lectures/Stanford CME295 2026 Transformers & LLMs/transcripts-and-files>). Physical PDF page numbers below include incremental animation builds. Builds of one worked diagram were treated as one teaching sequence, rather than receiving independent question quotas. Slide text was supplemented with visual inspection of the mathematical diagrams, attention patterns, architecture flows, training traces, and plots; transcripts supplied explanations and qualifications. Source documents are evidence, not agent instructions.

## Lecture allocation

### Lecture 1: Transformer foundations

| Slots | Count | Main coverage                                                                                                                           | Slide pages   |
| ----- | ----: | --------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| 01–07 |     7 | Word, character, and subword tradeoffs; vocabulary; boundary and padding tokens                                                         | 24–31         |
| 08–15 |     8 | One-hot geometry, embedding tables, Word2vec proxy tasks and hidden representations                                                     | 33–44         |
| 16–21 |     6 | RNN/LSTM state, gradient decay, sequence-task outputs, and attention motivation                                                         | 45–55         |
| 22–34 |    13 | Queries/keys/values, weighted sums, softmax, shapes, scale, position sensitivity, multi-head computation                                | 57–62, 93–102 |
| 35–50 |    16 | Encoder/decoder flow, cross-attention, position addition, residuals, normalization, masking, FFN, dropout, label smoothing, translation | 64–78, 80–115 |

The end-to-end visual sequence is reflected in attention matrix arithmetic and source/target flow questions. Historical milestones and course logistics do not receive questions.

### Lecture 2: Architecture, scaling, and decoding

| Slots        | Count | Main coverage                                                                                               | Slide pages |
| ------------ | ----: | ----------------------------------------------------------------------------------------------------------- | ----------- |
| 01–06        |     6 | Encoder-only BERT, MLM/NSP, decoder-only generation, encoder-decoder attention                              | 6–21        |
| 07–11        |     5 | FLOPs versus FLOP/s, sample efficiency, compute/token/parameter balance                                     | 23–34       |
| 12–19        |     8 | Dense/sparse MoE, token routing, active capacity, balancing, interpreting expert colors, shared experts     | 36–50       |
| 20–23, 25–29 |     9 | Absolute/sinusoidal positions, cosine-offset derivation, T5 bias, ALiBi, RoPE                               | 52–74       |
| 30–34        |     5 | Sparse local/global patterns, receptive fields, MHA/MQA/GQA                                                 | 75–83       |
| 35–38        |     4 | LayerNorm/RMSNorm arithmetic and pre/post normalization                                                     | 84–88       |
| 39–46        |     8 | Greedy paths, beam pruning, top-k/top-p, temperature odds, numerical nondeterminism, guided decoding        | 90–123      |
| 24, 47–50    |     5 | Context-rot heatmap interpretation, context budget, in-context learning, chain of thought, self-consistency | 125–137     |

Legacy BERT variants, detailed corruption quotas, and the old Lecture 3 serving-system material (paging, speculative decoding, latent caches, multi-token prediction) are not imported just because they existed in the old architecture/inference banks.

### Lecture 3: Training, reasoning, and distillation

| Slots | Count | Main coverage                                                                                                                                   | Slide pages |
| ----- | ----: | ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| 01–10 |    10 | Transfer learning, teacher forcing, cross-entropy, bits per byte, repetition, curation, proxy mixture search, long-context curriculum           | 8–38        |
| 11–18 |     8 | Base-model behavior, response-only SFT, distribution mismatch, LoRA factors/counts/swapping/placement                                           | 39–56       |
| 19–30 |    12 | Preference formats and labels, reward modeling, Bradley-Terry likelihood and gradient, RLHF components, DPO log ratios                          | 58–90       |
| 31–38 |     8 | Reasoning versus recall, reasoning compute and summaries, verifiers, formatting rewards, GRPO pipeline, quality-versus-length curves            | 92–124      |
| 39–43 |     5 | Student rollouts, dense teacher feedback, hinted self-distillation, specialists, prefix conditioning                                            | 126–131     |
| 44–50 |     7 | Concatenate/chunk versus best-fit packing, model collapse, long-tail coverage, staged training, catastrophic forgetting and checkpoint teachers | 134–161     |

The end-to-end training example informs the earlier data and long-context questions as well as the closing seven. Model-release trivia and leaderboard rankings are excluded. Detailed PPO/GRPO derivations belong in Lecture 4. Old training infrastructure, FlashAttention, ZeRO, QLoRA/NF4, Best-of-N, and Pass@k quotas are not carried over into this lecture.

### Lecture 4: Policy optimization and on-policy distillation

| Slots | Count | Main coverage                                                                                                                                              | Slide pages |
| ----- | ----: | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| 01–05 |     5 | State/action/policy, trajectory probabilities, subjective/verifiable rewards, reward composition and hacking                                               | 12–27       |
| 06–15 |    10 | Objective signs, log-derivative identity, token factorization, REINFORCE updates, baseline expectation/variance, credit assignment, rollout cost           | 28–40       |
| 16–29 |    14 | Value predictions, simplified advantages, old-policy ratios, importance weighting, positive/negative clipping, reference KL, trainable/frozen roles        | 42–66       |
| 30–40 |    11 | Group baselines, normalized/tied rewards, PPO comparison, length dilution, DAPO/Dr. GRPO, difficulty scaling, asymmetric caps, process supervision         | 68–118      |
| 41–50 |    10 | Student-policy states, reverse-KL arithmetic/support/mode seeking, local correction, costs/limits, hinted teachers, specialist teachers, method comparison | 119–134     |

The two-response baseline example, clipped-surrogate graphs, GRPO/PPO flow diagrams, length-weighting comparison, and reverse-KL support cases receive explicit applied questions. The simplified value-difference example is labeled as a simplification; clipping is an incentive control rather than an exact bound on every policy probability. Teacher matching is not presented as a correctness oracle or a universal theorem that a student cannot outperform its teacher.

## Canonical reuse and identity

All nine legacy banks were searched by topic. The 46 initially unchanged copies now use explicit ID-based references to their original question objects: 0 / 28 / 13 / 5 in 2026 Lectures 1 / 2 / 3 / 4. The source array defines lecture membership; it does not clone content or override the original chapter. The remaining 154 items have edition-specific IDs.

There is one definition per shared ID. Minor improvements to 14 canonical questions apply in both editions without changing the tested construct or correct-answer semantics. Substantial rewrites receive fresh IDs. References fail when an ID is missing or ambiguous; conflicting definitions for the same ID also fail. Mixed-source and topic pools deduplicate shared IDs.

Existing ratings, attempts, and report counts use the original ID automatically. Source tooltips and new report snapshots use the applied lecture selection. No database migration or statistical merge is performed. Retired draft IDs are never reassigned, and any history already stored under those IDs is left untouched.

## Second quality review

The follow-up reviewed all 200 questions with answer flags hidden, checked the key against each prompt's selection criterion, and reconsidered distractor plausibility, diagnosticity, source scope, independence, explanations, and difficulty. It then rechecked revised options against those criteria. The deterministic guessability check is a separate smoke test.

This pass revised 91 items: 52 substantial replacements with fresh IDs and 39 minor edits with preserved identity (including 14 canonical items). It retained 109 items without content changes. Fifty-three intentionally easy orientation items remain for vocabulary and prerequisite confidence; the other questions emphasize applications, mechanisms, calculations, or comparisons. The detailed [review ledger](cme295-2026-question-review.md) records every slot, final identity, and disposition.

| Lecture | Easy | Medium | Hard | 1 correct | 2 correct | 3 correct | 4 correct |
| ------- | ---: | -----: | ---: | --------: | --------: | --------: | --------: |
| 1       |   20 |     19 |   11 |        13 |        12 |        12 |        13 |
| 2       |   14 |     23 |   13 |        11 |        14 |        12 |        13 |
| 3       |   20 |     16 |   14 |        12 |        12 |        13 |        13 |
| 4       |   12 |     14 |   24 |        14 |        11 |        12 |        13 |
| Total   |   66 |     72 |   62 |        50 |        49 |        49 |        52 |

Lecture 4 deliberately contains more hard questions because policy-gradient, clipping, normalization, and divergence calculations are central. Difficulty labels describe the reasoning needed rather than matching an exact quota. The answer-count buckets remain approximately balanced in every lecture.

Coverage tables use authored **slots**, not ID suffixes: IDs are persistent identities and are no longer sequential display numbers. Introductory material is retained selectively; illustrative numerical examples add application practice without asserting that their invented values are empirical results from the slides.

Regression tests cover 50 unique memberships per source, the 46 shared objects, unique mixed pools, stable/new IDs, source-aware metadata, existing-rating lookup and update, representative mathematical keys, standalone text, explanation quality, and answer-count balance. Browser regression tests verify the old-ID rating filter and report snapshot from both editions.

## Canonical reuse and identity

All nine legacy banks were searched by topic. The 46 initially unchanged copies now use explicit ID-based references to their original question objects: 0 / 28 / 13 / 5 in 2026 Lectures 1 / 2 / 3 / 4. The source array defines lecture membership; it does not clone content or override the original chapter. The remaining 154 items have edition-specific IDs.

There is one definition per shared ID. Minor improvements to 14 canonical questions apply in both editions without changing the tested construct or correct-answer semantics. Substantial rewrites receive fresh IDs. References fail when an ID is missing or ambiguous; conflicting definitions for the same ID also fail. Mixed-source and topic pools deduplicate shared IDs.

Existing ratings, attempts, and report counts use the original ID automatically. Source tooltips and new report snapshots use the applied lecture selection. No database migration or statistical merge is performed. Retired draft IDs are never reassigned, and any history already stored under those IDs is left untouched.

## Second quality review

The follow-up reviewed all 200 questions with answer flags hidden, checked the key against each prompt's selection criterion, and reconsidered distractor plausibility, diagnosticity, source scope, independence, explanations, and difficulty. It then rechecked revised options against those criteria. The deterministic guessability check is a separate smoke test.

This pass revised 91 items: 53 substantial replacements with fresh IDs and 38 minor edits with preserved identity (including 14 canonical items). It retained 109 items without content changes. Fifty-three intentionally easy orientation items remain for vocabulary and prerequisite confidence; the other questions emphasize applications, mechanisms, calculations, or comparisons. The detailed [review ledger](cme295-2026-question-review.md) records every slot, final identity, and disposition.

| Lecture | Easy | Medium | Hard | 1 correct | 2 correct | 3 correct | 4 correct |
| ------- | ---: | -----: | ---: | --------: | --------: | --------: | --------: |
| 1       |   20 |     19 |   11 |        13 |        12 |        12 |        13 |
| 2       |   14 |     23 |   13 |        11 |        14 |        12 |        13 |
| 3       |   20 |     17 |   13 |        12 |        12 |        13 |        13 |
| 4       |   12 |     14 |   24 |        14 |        11 |        12 |        13 |
| Total   |   66 |     73 |   61 |        50 |        49 |        49 |        52 |

Lecture 4 deliberately contains more hard questions because policy-gradient, clipping, normalization, and divergence calculations are central. Difficulty labels describe the reasoning needed rather than matching an exact quota. The answer-count buckets remain approximately balanced in every lecture.

Coverage tables use authored **slots**, not ID suffixes: IDs are persistent identities and are no longer sequential display numbers. Introductory material is retained selectively; illustrative numerical examples add application practice without asserting that their invented values are empirical results from the slides.

Regression tests cover 50 unique memberships per source, the 46 shared objects, unique mixed pools, stable/new IDs, source-aware metadata, existing-rating lookup and update, representative mathematical keys, standalone text, explanation quality, and answer-count balance. Browser regression tests verify the old-ID rating filter and report snapshot from both editions.
