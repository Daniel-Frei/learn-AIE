# Lecture 5 Curriculum: Preference Tuning, RLHF, PPO, Best-of-N, and DPO

Source materials:

- Transcript: `lecture 5 - transcript.md`
- Slides: `lecture 5 - slides.pdf`

## Course Role

This lecture explains how post-training injects preference and negative-signal information after pretraining and supervised fine-tuning. It is the alignment bridge from "model follows instructions" to "model prefers responses humans or evaluators rate as helpful, safe, and appropriate."

## Learning Objectives

By the end, students should be able to:

- Explain why SFT alone may produce task-following but undesirable tone, safety, or preference behavior.
- Compare pointwise, pairwise, and listwise preference data.
- Describe how pairwise preference data is generated from prompts, candidate responses, and human or proxy labels.
- Map LLM preference tuning to reinforcement learning terms: policy, state, action, environment, reward.
- Explain reward modeling with the Bradley-Terry formulation and pairwise training.
- Explain RLHF as reward model training followed by policy optimization.
- Explain PPO at a high level: maximize advantage/reward while preventing large policy drift from an old or reference model.
- Distinguish PPO-Clip and PPO-KL penalty motivations.
- Explain reward hacking and why a KL/reference constraint matters.
- Compare RLHF/PPO, Best-of-N reranking, and DPO by models required, training complexity, inference cost, and stability.

## Prerequisite Assumptions

Students should know the pretraining/SFT lifecycle, next-token prediction, probability distributions, log probabilities, KL divergence at an intuition level, and the basic RL vocabulary introduced in the lecture.

## Curriculum Sequence

### 1. Motivate Preference Tuning

Start with the SFT model that responds in an undesirable way despite following the broad instruction. Explain that preference tuning adds negative and comparative signal: "prefer this response over that response." Use tone, helpfulness, safety, and style as dimensions.

Active learning:

- Give two responses to the same prompt and ask students what SFT target would be hard to write but preference comparison would be easy to label.
- Ask when "model misbehavior" should trigger preference tuning versus a review of SFT data quality.

Assessment targets:

- Students can explain why comparing two responses is often easier than writing the ideal response from scratch.
- Students can explain that preference tuning is not a substitute for poor SFT data distribution.

### 2. Build Preference Data

Teach observations as `(prompt, response)` pairs, then compare pointwise scores, pairwise preferences, and listwise rankings. Explain the recipe: collect prompts, generate candidate responses with the SFT model or other sources, then label via human raters or proxy evaluators.

Active learning:

- Convert a listwise ranking of four responses into pairwise comparisons.
- Ask which label type is easiest for humans and which gives the richest signal.

Assessment targets:

- Students can explain why pairwise data is common.
- Students can identify the risks of proxy labels such as LLM judges or rule-based metrics.

### 3. Map LLMs to RL

Introduce the RL formulation carefully. The LLM is the policy. The input-so-far is the state. The next token is the action. The token sequence environment evolves as actions are appended. A reward can be assigned to a completed response through human preference or a reward model.

Active learning:

- Given a generated sentence, label states, actions, and final reward.
- Ask why the reward is usually delayed until the response is complete.

Assessment targets:

- Students can map RL vocabulary to autoregressive generation without over-literalizing the analogy.
- Students understand why RL techniques become relevant to preference optimization.

### 4. Train a Reward Model

Teach the Bradley-Terry formulation: the probability that response `i` is preferred over response `j` depends on the difference between their reward scores. The reward model scores prompt-response pairs and is trained so preferred responses receive higher scores than rejected responses.

Active learning:

- Given two reward scores, compute which response has higher preference probability qualitatively.
- Ask why the reward model is trained pairwise but can score one response at inference time.

Assessment targets:

- Students can explain reward model input/output.
- Students can explain why the pairwise loss pushes preferred scores above rejected scores.
- Students understand that "human feedback" in RLHF refers to the labels used to train the reward model.

### 5. Optimize the Policy With PPO-Style RLHF

Teach the second RLHF stage: sample a response from the current policy, score it with the reward model, then update the policy to improve reward while staying close to a base/reference model. Introduce advantages as reward relative to a baseline/value estimate.

Active learning:

- Ask what can go wrong if the policy maximizes the reward model without a reference constraint.
- Compare a raw reward to an advantage estimate in a simple example.

Assessment targets:

- Students can explain why PPO uses constraints/clipping/KL-like terms to prevent destructive updates.
- Students can distinguish old policy, current policy, and reference/base model.
- Students can explain why value models and reward models are different.

### 6. Diagnose Reward Hacking

Use the transcript's explanation that an imperfect reward model can be exploited if optimization pressure is too strong. Teach reward hacking as optimizing the proxy instead of the true human goal.

Active learning:

- Create a toy reward function for "informative lecture" and ask how a model could exploit it.
- Ask students to propose mitigations: better reward model, KL constraint, human review, adversarial evaluation, or balanced objectives.

Assessment targets:

- Students can define reward hacking and connect it to Goodhart-like proxy failure.
- Students can explain why diversity in completions matters during RL.

### 7. Best-of-N as Inference-Time Reranking

Teach Best-of-N as skipping policy RL: generate N candidates from the SFT model, score each with the reward model, and return the highest-scoring response. Explain the tradeoff: simpler training but higher inference cost and dependence on the reward model at serving time.

Active learning:

- Rank four candidate responses by reward score and choose the returned output.
- Ask when Best-of-N is impractical due to traffic/cost.

Assessment targets:

- Students can compare training-time and inference-time costs.
- Students can explain why Best-of-N can improve quality without changing model weights.

### 8. Direct Preference Optimization

Teach DPO as a supervised preference-optimization method derived by rewriting the RL objective and Bradley-Terry preference probability so no separate reward model is needed. Explain that it operates directly on chosen/rejected preference pairs and a reference model.

Active learning:

- Compare the pipeline diagrams for RLHF/PPO and DPO.
- Ask students to list what models are required for each approach.

Assessment targets:

- Students can explain DPO's appeal: simpler, supervised-style optimization over preference data.
- Students can explain that DPO is not "no alignment"; it still uses preference data and a reference.

## Misconceptions to Address

- Preference tuning is not only about safety; it covers helpfulness, tone, usefulness, harmlessness, and other human preference axes.
- RLHF does not mean humans directly reward every RL rollout; humans label data used to train the reward model.
- A reward model is not guaranteed to represent true human intent.
- PPO does not simply maximize reward; it controls update size and reference drift.
- Best-of-N is not free; it moves cost to inference.
- DPO removes the explicit reward model stage but still depends on preference pairs and reference-model probabilities.

## Assessment Blueprint

Use method comparison and failure analysis:

- Construct preference pairs from candidate responses.
- Explain the Bradley-Terry reward-model loss qualitatively.
- Map an LLM generation episode to RL terms.
- Diagnose reward hacking in a proxy reward.
- Compare PPO, Best-of-N, and DPO across data needs, extra models, training complexity, serving cost, and failure modes.
- Explain why KL/reference constraints are important.

## Follow-Up Practice

- Read the InstructGPT/RLHF and DPO paper introductions.
- Make a pipeline diagram for SFT -> reward modeling -> PPO, then a separate DPO diagram.
- Write three preference-labeling guidelines that would reduce inconsistent human ratings.
