# Lecture 6 Curriculum: LLM Reasoning, Test-Time Scaling, GRPO, and Distillation

Source materials:

- Transcript: `lecture 6 - transcript.md`
- Slides: `lecture 6 - slides.pdf`

## Course Role

This lecture builds on preference tuning and RLHF to explain reasoning models: systems trained or prompted to produce intermediate reasoning before answering, often optimized with verifiable rewards and RL algorithms such as GRPO. It prepares students for agentic systems and evaluation by showing how reasoning quality is trained, measured, controlled, and distilled.

## Learning Objectives

By the end, students should be able to:

- Explain practical strengths and weaknesses of vanilla LLMs: imitation, coding, static knowledge, limited reasoning, inability to act, and evaluation difficulty.
- Define reasoning operationally as solving a problem through intermediate steps rather than immediate answer emission.
- Explain chain-of-thought prompting, hidden/full reasoning traces, and why providers may expose summaries rather than complete thoughts.
- Compare reasoning benchmarks for coding and math and interpret Pass@k, Pass@1, and Cons@k.
- Explain test-time scaling as spending more inference tokens/attempts/thinking to improve answer quality.
- Explain verifiable rewards for reasoning: formatting rewards and answer-correctness rewards.
- Explain inference-time thinking controls such as dynamic budgets, context awareness, budget forcing, and latent/continuous thoughts.
- Compare PPO and GRPO, especially value-model use versus group-relative advantage estimation.
- Explain the increasing-output-length phenomenon and mitigation ideas such as equalizing token-level contribution.
- Trace the DeepSeek R1-Zero and R1 training recipes at a high level.
- Explain reasoning distillation as training smaller models on generated reasoning traces rather than only matching next-token distributions.

## Prerequisite Assumptions

Students should know preference tuning, reward models, PPO/KL constraints, autoregressive generation, chain-of-thought prompting, self-consistency, and benchmark basics.

## Curriculum Sequence

### 1. Frame Reasoning as a Gap in Vanilla LLMs

Start with the lecture's contrast: vanilla LLMs are strong at imitation, idea generation, and coding assistance but weak where static knowledge, multi-step reasoning, or acting are required. Define reasoning tentatively as the ability to solve a problem, then distinguish factual lookup from reasoning.

Active learning:

- Sort prompts into "lookup", "reasoning", "tool needed", and "ambiguous" categories.
- Ask why "What is the course code?" and "How old will the bear be next year?" stress different capabilities.

Assessment targets:

- Students can identify when reasoning is the bottleneck versus knowledge retrieval or tool use.

### 2. Teach Chain-of-Thought and Reasoning Outputs

Review chain-of-thought as thinking in steps before producing an answer. Explain the new reasoning-model paradigm: output = reasoning + answer, though complete chains may be hidden and summarized in product interfaces.

Active learning:

- Compare a direct answer with a step-by-step answer for a small arithmetic or coding prompt.
- Ask what extra debugging signal a reasoning trace provides.

Assessment targets:

- Students can explain why reasoning tokens can improve accuracy but increase latency/cost.
- Students can explain why providers may hide full chains while exposing thought summaries.

### 3. Evaluate Reasoning Models

Teach benchmarks for coding and math: HumanEval, CodeForces, SWE-bench-like tasks, GSM8K, AIME, and similar evaluation sets. Explain Pass@k as the probability at least one of k attempts succeeds, Pass@1 for single-generation use cases, and Cons@k/self-consistency as majority-vote style evaluation.

Active learning:

- Given five generated solutions with pass/fail labels, compute or reason qualitatively about Pass@1 and Pass@k.
- Given multiple final answers, perform a Cons@k majority vote.

Assessment targets:

- Students can explain why Pass@k is useful when verification is easy and multiple attempts are acceptable.
- Students can distinguish "solved once" from "reliably solved."

### 4. Train for Test-Time Scaling With Verifiable Rewards

Explain the DeepSeek-R1-style idea: if reasoning traces are expensive to write manually and human-written reasoning may limit the model, use RL with verifiable rewards. Rewards can check that required reasoning format is present and that the final answer passes tests or matches a verifiable solution.

Active learning:

- Design a reward for a coding task where tests provide correctness.
- Design a reward for a math task where the final numeric answer can be verified.
- Ask what reward is easy to verify and what remains hard to verify.

Assessment targets:

- Students can explain why verifiable rewards reduce dependence on hand-written chain-of-thought data.
- Students can state the limits of formatting rewards.

### 5. Control Thinking at Inference Time

Teach that not every prompt deserves the same reasoning budget. Cover dynamic budgets, context awareness, budget forcing, and continuous/latent thoughts as methods to control or represent thinking.

Active learning:

- Assign reasoning budgets to easy, medium, and hard prompts.
- Ask when forcing more thinking can waste tokens or harm UX.

Assessment targets:

- Students can explain test-time scaling as a tradeoff, not a universal improvement.

### 6. Compare PPO and GRPO

Teach GRPO as Group Relative Policy Optimization: sample a group of responses, compute rewards, and estimate advantage relative to the group's average instead of training a separate value model. Compare with PPO's old-policy ratio, clipping, KL/reference logic, and value-function baseline.

Active learning:

- Given rewards for a group of sampled responses, compute which responses have positive and negative group-relative advantage qualitatively.
- Ask which extra model PPO needs that GRPO can avoid.

Assessment targets:

- Students can explain GRPO's main simplification over PPO.
- Students can identify similarities: policy ratios, clipping-style update control, and KL/reference constraints.

### 7. Diagnose Length Incentives and Other RL Pathologies

Teach the increasing-output-length phenomenon: when reward and token-level credit assignment are misaligned, RL may incentivize longer outputs. Introduce mitigation families such as Dr. GRPO and DAPO that adjust token-level contribution and encourage diversity or handle difficulty bias.

Active learning:

- Ask why "longer reasoning" can become a bad incentive even if reasoning helps some tasks.
- Propose a reward or normalization change that avoids rewarding verbosity.

Assessment targets:

- Students can explain why optimizing reasoning length is different from optimizing reasoning quality.

### 8. Stitch Together DeepSeek R1-Zero and R1

Teach the high-level recipes. R1-Zero starts from a base model and applies GRPO with reasoning/verifiable rewards, producing strong reasoning but rough formatting/readability. R1 adds small-scale SFT on reasoning data, GRPO, broader SFT/rejection-sampled data, and helpfulness/harmlessness preference tuning to improve usability.

Active learning:

- Compare R1-Zero and R1 on data stages, advantages, and challenges.
- Ask why a model might reason well but produce messy outputs without SFT.

Assessment targets:

- Students can describe why pure RL can discover reasoning but may need SFT and preference tuning for readability and assistant behavior.

### 9. Distill Reasoning

Close with distillation. Contrast earlier distribution-matching distillation with reasoning distillation: a large reasoning model generates traces and answers that a smaller model learns through SFT-style training.

Active learning:

- Ask what is transferred in ordinary distillation versus reasoning-trace distillation.
- Decide when distillation is a good use of compute compared with running the large model directly.

Assessment targets:

- Students can explain why smaller models can become competitive on reasoning tasks when trained on high-quality traces.

## Misconceptions to Address

- Chain-of-thought is not guaranteed truth; it is a generated trace that can be useful, misleading, hidden, or summarized.
- More tokens are not automatically better reasoning.
- Pass@k can look strong even when a single response is unreliable.
- Verifiable rewards only cover tasks where correctness can be checked.
- GRPO is not "RL without constraints"; it still needs careful update control and reward design.
- R1-Zero and R1 differ in usability/data stages, not just benchmark scores.

## Assessment Blueprint

Use reasoning traces, benchmark interpretation, and RL comparison:

- Classify prompts by capability bottleneck.
- Compare direct answer, chain-of-thought, and self-consistency approaches.
- Interpret Pass@k and Cons@k from sample outputs.
- Design verifiable rewards for math/coding tasks.
- Compare PPO and GRPO model requirements and advantage estimates.
- Diagnose length reward pathologies.
- Explain a reasoning distillation pipeline.

## Follow-Up Practice

- Read the DeepSeek R1 paper overview and draw the R1-Zero versus R1 pipelines.
- Build a table of reasoning evaluation metrics and when each is appropriate.
- Write a short policy for when an assistant should use a reasoning model versus a faster non-reasoning model.
