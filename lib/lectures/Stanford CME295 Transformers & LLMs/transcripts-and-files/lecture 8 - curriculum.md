# Lecture 8 Curriculum: LLM Evaluation

Source materials:

- Transcript: `lecture 8 - transcript.md`
- Slides: `lecture 8 - slides.pdf`

## Course Role

This lecture teaches students how to measure LLM output quality and system behavior. It connects earlier topics - generation, tools, agents, benchmarks, and preference data - to the practical question: if we cannot measure performance, we do not know what to improve.

## Learning Objectives

By the end, students should be able to:

- Define evaluation scope: output quality, system performance, task performance, alignment, latency, cost, uptime, and reliability.
- Explain why free-form text evaluation is difficult and why human ratings are closest to ground truth but slow, expensive, and subjective.
- Interpret inter-rater agreement metrics such as Cohen's kappa, Fleiss' kappa, and Krippendorff's alpha at a conceptual level.
- Explain rule-based reference metrics such as METEOR, BLEU, and ROUGE, and their limitations.
- Design an LLM-as-a-Judge evaluation prompt with criteria, rationale, and structured output.
- Compare pointwise and pairwise LLM-as-a-Judge setups.
- Identify and mitigate position bias, verbosity bias, self-enhancement bias, and reproducibility issues.
- Evaluate factuality through claim decomposition and nuanced scoring.
- Diagnose agentic workflow failures: tool prediction errors, tool call errors, and response generation errors.
- Compare common benchmark categories: knowledge, math reasoning, common-sense reasoning, coding, safety, and agents.
- Explain Pass^k as an agent reliability metric and distinguish it from Pass@k.
- Explain model profiles, Pareto frontiers, data contamination, and Goodhart's law.

## Prerequisite Assumptions

Students should know generation and decoding, preference labels, RAG, tool calling, agents, and benchmark examples from earlier lectures.

## Curriculum Sequence

### 1. Define What Is Being Evaluated

Start with scope. Evaluation can mean output quality or system performance. Output quality can include instruction following, coherence, factuality, usefulness, relevance, tone, safety, and alignment. System performance can include latency, cost, uptime, and failure rates.

Active learning:

- Given a product use case, ask students to list output metrics and system metrics separately.
- Ask which metric would be misleading if used alone.

Assessment targets:

- Students can explain why "evaluate the LLM" is underspecified.
- Students can separate model quality from serving-system performance.

### 2. Use Human Ratings Carefully

Teach human ratings as the closest available gold signal for open-ended outputs. Then explain subjectivity, rubric quality, rater alignment, slow throughput, and cost. Introduce agreement metrics as asking: how much better is observed agreement than chance agreement?

Active learning:

- Have students rate two outputs with a vague rubric, then with a crisp rubric, and compare disagreement.
- Interpret a positive, zero, or negative kappa qualitatively.

Assessment targets:

- Students can explain why rater calibration matters.
- Students can explain agreement metrics without memorizing formulas.

### 3. Understand Rule-Based Metrics

Teach reference-based metrics: METEOR, BLEU, and ROUGE. Explain n-gram overlap, precision/recall emphasis, brevity penalty, synonym/stem handling, and why these metrics struggle with stylistic variation and semantic equivalence.

Active learning:

- Compare three paraphrases with the same meaning and ask which rule-based metric may under-score them.
- Ask when BLEU/ROUGE are still useful.

Assessment targets:

- Students can explain why reference-based metrics require labels.
- Students can explain the difference between precision-focused and recall-focused text metrics.

### 4. Design LLM-as-a-Judge Evaluations

Teach LLM-as-a-Judge as using an LLM to rate responses against criteria. Include structured outputs and rationale-first prompting. Explain benefits: no reference required for some tasks, faster than human evaluation, and interpretable rationales.

Active learning:

- Write a judge prompt for "relevance to the user request" with binary output.
- Define a structured response schema with `rationale` and `score`.
- Compare pointwise scoring and pairwise comparison.

Assessment targets:

- Students can design criteria that are specific enough to evaluate.
- Students can explain why low temperature and structured output improve reproducibility.

### 5. Mitigate Judge Biases

Teach common biases: position bias in pairwise comparisons, verbosity bias favoring longer answers, self-enhancement bias when a judge favors model-like or self-generated responses, and inconsistent scoring. Cover mitigations: position swapping/averaging, explicit guidelines, length penalties or relevance criteria, few-shot calibration, rationale before score, human calibration, and low temperature.

Active learning:

- Swap response order in a pairwise judge prompt and ask how to aggregate.
- Revise a rubric to avoid rewarding verbosity.

Assessment targets:

- Students can name and mitigate major LLM-as-a-Judge biases.
- Students can explain why judge scores should be calibrated against human ratings.

### 6. Evaluate Factuality

Teach factuality as a nuanced dimension. Use claim decomposition: break a response into atomic claims, check each claim against evidence or ground truth, and aggregate. Explain that factuality is not a simple all-or-nothing score when outputs mix true, false, and misleading claims.

Active learning:

- Decompose a short answer about teddy bear history into claims and classify each as supported, contradicted, or unverifiable.
- Ask whether a fluent answer can receive low factuality.

Assessment targets:

- Students can distinguish coherence from factuality.
- Students can explain why factuality needs evidence or a trusted reference.

### 7. Evaluate Tool and Agent Workflows

Use the lecture's agentic failure taxonomy:

- Tool prediction errors: no tool use when needed, hallucinated tool, wrong tool, wrong argument.
- Tool call errors: wrong response, backend error, no response.
- Response generation errors: final answer fails to convey or ground in tool output.

Active learning:

- Diagnose failures in a toy "find a bear near me" workflow.
- Match each failure to likely causes and remedies: better router, clearer API, stronger model, helper tool, fixed backend, trimmed/structured tool output, or improved synthesis model.

Assessment targets:

- Students can locate where an agent failed rather than saying "the model failed."
- Students can propose targeted fixes instead of generic model upgrades.

### 8. Interpret Benchmarks

Teach benchmark categories and examples: MMLU for broad knowledge, AIME/GSM8K-style math reasoning, PIQA for physical common sense, SWE-bench for coding, HarmBench for safety, and tau-bench for tool-agent-user interaction. Introduce Pass^k for consistency/reliability: all k attempts must succeed, unlike Pass@k where at least one succeeds.

Active learning:

- Match a product need to benchmark categories.
- Compare Pass@k and Pass^k on a set of successes/failures.

Assessment targets:

- Students can explain what each benchmark category projects and what it misses.
- Students can explain why agents require reliability metrics, not just one-off success.

### 9. Avoid Benchmark Overreach

Teach model profiles and Pareto frontiers: models differ across axes such as quality, cost, latency, safety, and context length. Cover data contamination risks and mitigations such as hashes, blocklists, and newer test versions. Close with Goodhart's law and the role of organic evaluation like Chatbot Arena and direct testing on one's own workload.

Active learning:

- Choose between two models with different cost/quality/latency profiles.
- Identify contamination risk in a benchmark and propose a mitigation.

Assessment targets:

- Students can explain why no benchmark leaderboard fully answers "which model is best for me?"
- Students can explain Goodhart's law in the context of LLM benchmarks.

## Misconceptions to Address

- Human ratings are not automatically objective; they need rubrics and rater alignment.
- BLEU/ROUGE are not general semantic quality metrics.
- LLM-as-a-Judge is not a perfect replacement for humans; it inherits biases and needs calibration.
- A high benchmark score does not guarantee product fit.
- Agent evaluation must inspect intermediate tool and response stages.
- Pass@k and Pass^k measure different reliability stories.

## Assessment Blueprint

Use evaluation design and interpretation tasks:

- Define evaluation scope for a target application.
- Draft a human-rating rubric and an LLM-as-a-Judge rubric.
- Diagnose judge bias and propose mitigation.
- Decompose factuality into claims.
- Classify agent workflow failures by stage.
- Match benchmarks to capabilities.
- Interpret Pareto tradeoffs and contamination risk.

## Follow-Up Practice

- Create an evaluation plan for a RAG or tool-using assistant with human checks, automated judges, and system metrics.
- Build a small table comparing BLEU, ROUGE, METEOR, LLM-as-a-Judge, human ratings, and benchmark suites.
- Read tau-bench or SWE-bench summaries and identify what makes them more system-like than ordinary MCQ benchmarks.
