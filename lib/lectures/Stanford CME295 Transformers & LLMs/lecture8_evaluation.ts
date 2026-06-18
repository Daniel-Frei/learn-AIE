import { Question } from "../../quiz";

type Lecture8Difficulty = "easy" | "medium" | "hard";
type OptionSeed = readonly [text: string, isCorrect: boolean];
type AssertionReasonChoice = 1 | 2 | 3 | 4 | 5;

const assertionReasonOptionTexts = [
  "Assertion is true, Reason is false.",
  "Assertion is false, Reason is true.",
  "Both are false.",
  "Both are true, and the Reason is the correct explanation of the Assertion.",
  "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
] as const;

function makeQuestion(
  id: string,
  difficulty: Lecture8Difficulty,
  prompt: string,
  optionSeeds: readonly OptionSeed[],
  explanation: string,
): Question {
  if (optionSeeds.length !== 4) {
    throw new Error(`CME295 Lecture 8 question ${id} needs 4 options.`);
  }

  return {
    id,
    chapter: 8,
    difficulty,
    prompt,
    options: optionSeeds.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

function makeAssertionReasonQuestion(
  id: string,
  difficulty: Lecture8Difficulty,
  prompt: string,
  correctChoice: AssertionReasonChoice,
  explanation: string,
): Question {
  return {
    id,
    chapter: 8,
    difficulty,
    type: "assertion-reason",
    prompt,
    options: assertionReasonOptionTexts.map((text, index) => ({
      text,
      isCorrect: index + 1 === correctChoice,
    })),
    explanation,
  };
}

export const stanfordCME295Lecture8EvaluationQuestions: Question[] = [
  makeQuestion(
    "cme295-lect8-q01",
    "easy",
    "A team says it wants to evaluate its language model. Which measurement targets belong to output quality rather than serving-system performance?",
    [
      ["Whether the response follows the user's instruction.", true],
      ["Whether the response is coherent and relevant.", true],
      ["Whether the response is factually supported.", true],
      ["Whether the hosted endpoint has low latency and high uptime.", false],
    ],
    "Output quality is about the content the user receives: instruction following, coherence, relevance, factuality, usefulness, tone, safety, and alignment. Latency, cost, pricing, uptime, and failure rates are system-performance metrics, and they can matter a lot without directly measuring whether a particular answer is good.",
  ),
  makeQuestion(
    "cme295-lect8-q02",
    "easy",
    "Why is free-form language-model output harder to evaluate than a fixed multiple-choice prediction?",
    [
      [
        "The model can answer in many semantically equivalent phrasings that do not share surface wording.",
        true,
      ],
      [
        "A response can mix correct, incorrect, irrelevant, and stylistically acceptable content in one passage.",
        true,
      ],
      [
        "The right judgment often depends on a criterion such as usefulness, factuality, or tone.",
        true,
      ],
      [
        "Free-form answers are always impossible to compare against human judgments.",
        false,
      ],
    ],
    "Open-ended text is difficult because there is not usually one exact answer string to match. Human judgments and automated judges can still be useful, but they need explicit criteria and careful handling of partial correctness, paraphrase, and subjective dimensions.",
  ),
  makeQuestion(
    "cme295-lect8-q03",
    "hard",
    `Two independent raters assign a binary "good" label with these marginal probabilities:

| Rater | \\(P(\\text{good})\\) | \\(P(\\text{not good})\\) |
| --- | ---: | ---: |
| A | 0.8 | 0.2 |
| B | 0.7 | 0.3 |

What is their expected agreement rate by chance?`,
    [
      [
        "\\(0.8 \\times 0.7 + 0.2 \\times 0.3 = 0.62\\), because chance agreement includes both matching positive and matching negative labels.",
        true,
      ],
      [
        "\\(0.8 \\times 0.3 + 0.2 \\times 0.7 = 0.38\\), because chance agreement counts the cases where the raters disagree.",
        false,
      ],
      [
        "\\((0.8 + 0.7) / 2 = 0.75\\), because chance agreement is the average positive-label rate.",
        false,
      ],
      [
        "\\(1 - |0.8 - 0.7| = 0.90\\), because similar marginal rates force the raters to agree most of the time.",
        false,
      ],
    ],
    "The chance baseline is the probability that both independently choose the positive label plus the probability that both independently choose the negative label. The disagreement probability is useful context, but it is not the expected agreement that agreement-adjusted metrics compare against.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect8-q04",
    "medium",
    "Assertion: Cohen's kappa can be negative even when two raters sometimes choose the same labels.\n\nReason: Kappa compares observed agreement against the agreement expected by chance from the raters' label frequencies.",
    4,
    "The assertion is true because observed agreement can fall below the chance baseline implied by the raters' marginal label use. The reason is also true and explains the assertion: kappa is not just raw agreement, but agreement adjusted relative to an expected-by-chance baseline.",
  ),
  makeQuestion(
    "cme295-lect8-q05",
    "easy",
    "A product team wants to use human ratings as the closest available signal for open-ended answer quality. Which practices make those ratings more useful?",
    [
      [
        "Use a clear rubric that tells raters what dimension to evaluate.",
        true,
      ],
      [
        "Track inter-rater agreement as a health metric for rating consistency.",
        true,
      ],
      [
        "Run alignment or calibration sessions when raters disagree too much.",
        true,
      ],
      [
        "Budget for human ratings as a slower and more expensive signal than most automated metrics.",
        true,
      ],
    ],
    "Human ratings are valuable because humans can judge open-ended outputs with nuance, but they are not automatically objective or cheap. Rubrics, agreement tracking, calibration, and realistic throughput planning make the signal more reliable and usable.",
  ),
  makeQuestion(
    "cme295-lect8-q06",
    "easy",
    "Which statements correctly describe reference-based rule metrics such as METEOR, BLEU, and ROUGE?",
    [
      [
        "They compare a model output with one or more reference outputs written or selected ahead of time.",
        true,
      ],
      [
        "They can be reused across model iterations once the reference set is fixed.",
        true,
      ],
      [
        "They eliminate the need to define a task-specific notion of quality.",
        false,
      ],
      [
        "They directly inspect a model's hidden reasoning rather than its output text.",
        false,
      ],
    ],
    "Rule-based text metrics make repeated evaluation cheaper by comparing model outputs to fixed references. They still depend on task design and references, and they measure surface output overlap rather than hidden model reasoning.",
  ),
  makeQuestion(
    "cme295-lect8-q07",
    "medium",
    "Which distinctions among common rule-based text metrics are correct?",
    [
      [
        "METEOR combines precision and recall-style matching with an ordering penalty.",
        true,
      ],
      [
        "BLEU is more precision-focused and uses a brevity penalty to discourage overly short outputs.",
        true,
      ],
      [
        "ROUGE is commonly associated with summarization and recall-oriented overlap variants.",
        true,
      ],
      [
        "All three metrics reliably give full credit to paraphrases that preserve meaning but share few words.",
        false,
      ],
    ],
    "The metrics differ in emphasis, but they all depend heavily on overlap with a reference. That makes them useful for some constrained comparisons and weak for semantic equivalence, stylistic variation, and nuanced quality judgments.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect8-q08",
    "medium",
    "Assertion: A reference-overlap metric can underrate an answer that preserves the intended meaning with very different wording.\n\nReason: Reference-overlap metrics generally compare generated text against fixed reference wording rather than fully solving semantic equivalence.",
    4,
    "The assertion is true because a strong paraphrase can share few n-grams with a reference. The reason is also true and explains the assertion: overlap metrics are anchored to reference text, so semantic sameness without surface overlap is a known failure mode.",
  ),
  makeQuestion(
    "cme295-lect8-q09",
    "easy",
    "In an LLM-as-a-Judge evaluation, which pieces normally belong in the judge call or judge output?",
    [
      ["The original user prompt or task context.", true],
      ["The model response being evaluated.", true],
      ["The evaluation criterion or rubric.", true],
      [
        "A mandatory weight update to the model that produced the response.",
        false,
      ],
    ],
    "An LLM-as-a-Judge setup usually gives the judge the prompt, the response, and the criteria, then asks for a score and often a rationale. It is an evaluation call, not a training step, so it does not require updating the generator model's weights.",
  ),
  makeQuestion(
    "cme295-lect8-q10",
    "medium",
    "Which implementation choices make LLM-as-a-Judge outputs easier to use in an automated evaluation pipeline?",
    [
      [
        "Define an explicit output schema, such as fields for `rationale` and `score`.",
        true,
      ],
      [
        "Ask for the rationale before the final score so the judge evaluates before committing to the label.",
        true,
      ],
      ["Use a low temperature when reproducibility matters.", true],
      [
        "Prefer a simple binary pass/fail score when a granular scale would add noise without useful signal.",
        true,
      ],
    ],
    "Structured outputs make parsing and aggregation practical, while a rationale-first instruction can improve the judgment process. Low temperature, crisp guidelines, and simple scales reduce avoidable variance when the evaluation needs to be repeatable.",
  ),
  makeQuestion(
    "cme295-lect8-q11",
    "easy",
    "Which statements correctly distinguish pointwise and pairwise LLM-as-a-Judge setups?",
    [
      ["Pointwise judging rates a single response against a criterion.", true],
      [
        "Pairwise judging asks the judge to compare two responses and choose or rank between them.",
        true,
      ],
      [
        "Pointwise judging requires two model responses to be present in every prompt.",
        false,
      ],
      [
        "Pairwise judging removes every ordering and preference bias by construction.",
        false,
      ],
    ],
    "Pointwise and pairwise judging answer different evaluation questions: absolute quality versus relative preference. Pairwise comparisons are useful, but they introduce ordering effects and still require bias mitigation.",
  ),
  makeQuestion(
    "cme295-lect8-q12",
    "easy",
    "A pairwise judge chooses Response A when asked `A or B?` and chooses Response B when asked `B or A?`, even though the response contents are unchanged. Which conclusions are appropriate?",
    [
      [
        "The judge may have position bias because the preferred output changes with presentation order.",
        true,
      ],
      [
        "Swapping order and averaging or aggregating judgments is a practical mitigation to test.",
        true,
      ],
      [
        "The result proves that both responses are equally good under the target rubric.",
        false,
      ],
      [
        "The result means pairwise evaluation should never be used for language-model outputs.",
        false,
      ],
    ],
    "A preference flip after order swapping is evidence that the judge may be sensitive to position rather than only content. That does not prove a tie or invalidate all pairwise judging, but it does mean the evaluation design needs order controls.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect8-q13",
    "medium",
    "Assertion: A verbose response can be preferred by a judge even when the extra detail does not improve correctness or usefulness.\n\nReason: Verbosity bias is a tendency to reward longer or more detailed answers because of length itself rather than task-relevant quality.",
    4,
    "The assertion is true because a judge can confuse length with helpfulness. The reason is also true and explains the assertion: verbosity bias is exactly the failure mode where length becomes a misleading signal for quality.",
  ),
  makeQuestion(
    "cme295-lect8-q14",
    "hard",
    "A team evaluates answers from Model X by using Model X as the judge. Which risks and mitigations are source-grounded?",
    [
      [
        "The judge can show self-enhancement bias by favoring outputs similar to its own generations.",
        true,
      ],
      [
        "Using a different, often stronger, judge model can reduce the specific risk of judging one's own outputs.",
        true,
      ],
      [
        "Calibration against human ratings remains useful because a different judge model can still have systematic preferences.",
        true,
      ],
      [
        "Using the same model for generation and judging guarantees a perfectly objective estimate of human preference.",
        false,
      ],
    ],
    "Self-enhancement bias is the concern that a model prefers responses like the ones it would produce. A different and capable judge plus human calibration does not make evaluation perfect, but it reduces a clear source of circularity.",
  ),
  makeQuestion(
    "cme295-lect8-q15",
    "easy",
    "Which LLM-as-a-Judge best practices are appropriate when the evaluation task has some subjectivity?",
    [
      ["Write crisp guidelines for what counts as success or failure.", true],
      ["Use examples or calibration when the rubric is easy to misread.", true],
      [
        "Mitigate known biases such as position, verbosity, and self-enhancement.",
        true,
      ],
      [
        "Compare judge outputs with human ratings when the budget allows it.",
        true,
      ],
    ],
    "A judge prompt is only as useful as the evaluation protocol around it. Clear criteria, calibration, bias checks, and human comparison make automated judging more trustworthy without pretending it is a perfect replacement for people.",
  ),
  makeQuestion(
    "cme295-lect8-q16",
    "medium",
    "Which statements correctly describe claim-decomposition approaches to factuality evaluation?",
    [
      [
        "A long answer can be split into atomic factual claims before each claim is checked.",
        true,
      ],
      [
        "Claim scores can be aggregated, sometimes with importance weights, instead of treating factuality as all-or-nothing.",
        true,
      ],
      [
        "A fluent and coherent answer should automatically receive a high factuality score.",
        false,
      ],
      [
        "A factuality check can avoid evidence or trusted references as long as the output sounds confident.",
        false,
      ],
    ],
    "Factuality is different from fluency: a polished answer can contain false or unsupported claims. Decomposing the answer into claims lets the evaluator check evidence item by item and aggregate a more nuanced score.",
  ),
  makeQuestion(
    "cme295-lect8-q17",
    "hard",
    `A factuality evaluator decomposes one answer into four claims and assigns weights:

| Claim | Weight | Status |
| --- | ---: | --- |
| A | 0.30 | supported |
| B | 0.25 | contradicted |
| C | 0.20 | supported |
| D | 0.25 | unverifiable |

If supported claims receive credit and contradicted or unverifiable claims receive no credit, what factuality score follows from this setup?`,
    [
      ["0.50, because the supported claim weights are 0.30 and 0.20.", true],
      ["0.75, because only the contradicted claim should lose credit.", false],
      [
        "0.25, because the score should average the two unsupported statuses.",
        false,
      ],
      ["1.00, because the answer has at least one supported claim.", false],
    ],
    "The weighted score is the sum of the weights for the supported claims under the stated scoring rule. Contradicted and unverifiable claims do not receive credit here, so the score is \\(0.30 + 0.20 = 0.50\\).",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect8-q18",
    "medium",
    "Assertion: A fluent answer can receive a low factuality score.\n\nReason: Factuality evaluation asks whether the answer's claims are supported by evidence or ground truth, not merely whether the prose is coherent.",
    4,
    "The assertion is true because coherence and factuality are separate evaluation dimensions. The reason is true and explains the assertion: an answer can read well while containing unsupported, contradicted, or misleading claims.",
  ),
  makeQuestion(
    "cme295-lect8-q19",
    "easy",
    "A tool-using assistant processes a request by selecting a function and arguments, executing the backend call, and synthesizing a final response from the returned data. Which failure categories match those stages?",
    [
      [
        "Tool prediction error when the model chooses no tool, the wrong tool, a nonexistent tool, or wrong arguments.",
        true,
      ],
      [
        "Tool call error when the backend returns an incorrect value, an error, or no usable response.",
        true,
      ],
      [
        "Response generation error when the final answer fails to convey or ground in the tool result.",
        true,
      ],
      [
        "Benchmark contamination error when the model selects a tool before seeing any user request.",
        false,
      ],
    ],
    "Agent evaluation should localize the failure stage instead of collapsing every issue into a generic model failure. The taxonomy separates selecting the tool, executing the call, and using the result in the final answer.",
  ),
  makeQuestion(
    "cme295-lect8-q20",
    "easy",
    "A user asks for a current account balance, an account-balance API is available, and the assistant answers from memory without calling it. Which diagnosis and remedies fit this failure?",
    [
      [
        "This is a tool prediction error because the assistant did not use a tool when the task required one.",
        true,
      ],
      [
        "Possible remedies include improving the tool router, the model's tool-use prompt, or supervised fine-tuning for API use.",
        true,
      ],
      [
        "This is primarily a tool call error because the backend returned an invalid JSON object.",
        false,
      ],
      [
        "The correct fix is to trim the backend response even though no backend call was made.",
        false,
      ],
    ],
    "The failure occurs before backend execution: the model or router failed to select the needed tool. Backend-output fixes address a different stage and cannot help if the tool was never called.",
  ),
  makeQuestion(
    "cme295-lect8-q21",
    "medium",
    "An assistant predicts the right function name but passes coordinates `(0, 0)` because the user's location was not available in context. Which evaluation conclusions are appropriate?",
    [
      [
        "The failure can be categorized as a tool prediction error involving wrong arguments.",
        true,
      ],
      [
        "A helper tool or clearer context path for obtaining the missing argument can be a targeted remedy.",
        true,
      ],
      [
        "The backend implementation must be wrong because any argument value chosen by the model should succeed.",
        false,
      ],
      [
        "The final answer stage is the only stage worth evaluating because the function name was correct.",
        false,
      ],
    ],
    "Wrong arguments are part of the tool prediction layer even if the chosen function name is valid. The remedy should address argument availability and tool-use modeling before assuming the backend or final synthesis is the main problem.",
  ),
  makeQuestion(
    "cme295-lect8-q22",
    "medium",
    "A backend tool executes successfully but returns a huge unstructured dump. The final response ignores the relevant field and states the opposite of the tool output. Which fixes are plausible for this response-generation failure?",
    [
      ["Use a stronger synthesis model with better grounding.", true],
      [
        "Trim or structure the tool output so the relevant fields are easier to use.",
        true,
      ],
      [
        "Make the tool output format descriptive, such as named fields instead of uninterpretable raw text.",
        true,
      ],
      [
        "Remove the tool result from the context so the model can answer from pretraining alone.",
        false,
      ],
    ],
    "The failure is in using the returned data, not necessarily in selecting or executing the tool. Better grounding, less noisy context, and meaningful output structure target the stage where the model synthesizes the final answer.",
  ),
  makeQuestion(
    "cme295-lect8-q23",
    "medium",
    "Which statements correctly summarize the common issue clusters in tool and agent evaluation?",
    [
      [
        "Some failures are modeling issues, such as weak reasoning, weak grounding, or poor tool-route modeling.",
        true,
      ],
      [
        "Some failures are tool issues, such as an implementation bug or output that is hard to interpret.",
        true,
      ],
      [
        "Too much irrelevant material in the context window can make grounding and synthesis worse.",
        true,
      ],
      [
        "Every tool-agent failure is best fixed by replacing the entire agent with a larger language model.",
        false,
      ],
    ],
    "The source emphasizes methodical error categorization because agent failures can come from several layers. A bigger model may help some failures, but API design, routing, backend behavior, context relevance, and output format often require targeted fixes.",
  ),
  makeQuestion(
    "cme295-lect8-q24",
    "easy",
    "Which benchmark-category mappings are correct?",
    [
      [
        "MMLU probes broad knowledge through constrained multiple-choice tasks.",
        true,
      ],
      [
        "AIME probes math reasoning with answers constrained to a fixed numeric format.",
        true,
      ],
      [
        "SWE-bench probes coding by checking whether generated patches pass tests.",
        true,
      ],
      [
        "HarmBench probes safety behavior such as harmful or policy-violating outputs.",
        true,
      ],
    ],
    "These benchmarks project model behavior onto different axes: knowledge, math reasoning, coding, and safety. They are not interchangeable, so a model can be strong on one axis and weaker on another.",
  ),
  makeQuestion(
    "cme295-lect8-q25",
    "easy",
    "Which claims about MMLU-style knowledge benchmarks are correct?",
    [
      [
        "They use constrained choices, making scoring easier than judging free-form prose.",
        true,
      ],
      [
        "They emphasize broad factual and domain knowledge, which partly reflects pretraining retention.",
        true,
      ],
      [
        "They primarily test whether a tool-using agent can complete multi-turn airline or retail tasks.",
        false,
      ],
      [
        "They remove the possibility of data contamination because all tasks are multiple choice.",
        false,
      ],
    ],
    "MMLU-like tasks standardize broad knowledge evaluation through answer choices that can be checked directly. The format does not make them agent benchmarks, and it does not by itself prevent the benchmark content from appearing in training data.",
  ),
  makeQuestion(
    "cme295-lect8-q26",
    "medium",
    "Which statements correctly compare AIME and PIQA as reasoning benchmarks?",
    [
      [
        "AIME-style tasks test multi-step mathematical reasoning with a constrained final numeric answer.",
        true,
      ],
      [
        "PIQA-style tasks test physical common-sense reasoning with constrained answer choices.",
        true,
      ],
      [
        "AIME is mainly a broad professional knowledge benchmark like MMLU.",
        false,
      ],
      [
        "PIQA evaluates whether a generated code patch passes repository tests.",
        false,
      ],
    ],
    "Both benchmarks are reasoning-oriented, but they probe different kinds of reasoning. AIME is math-heavy and numerically constrained, while PIQA is grounded in everyday physical situations rather than code patches or broad domain facts.",
  ),
  makeQuestion(
    "cme295-lect8-q27",
    "medium",
    "Why is SWE-bench a coding benchmark rather than just another free-form text metric?",
    [
      ["It asks a model to produce a patch for a real software issue.", true],
      [
        "The generated patch can be evaluated by running tests before and after applying it.",
        true,
      ],
      [
        "It scores answers by counting n-gram overlap with a reference explanation.",
        false,
      ],
      [
        "It requires human raters to score every generated patch during every model iteration.",
        false,
      ],
    ],
    "SWE-bench is grounded in executable repository behavior, not in text overlap. Passing tests provides a hard-coded signal about whether the proposed code change solved the issue under the benchmark's setup.",
  ),
  makeQuestion(
    "cme295-lect8-q28",
    "medium",
    "Which statements about safety benchmarks such as HarmBench are correct?",
    [
      [
        "Safety benchmark results should be interpreted against the provider's policy goals and the benchmark's content.",
        true,
      ],
      [
        "Some safety evaluations use a classifier because harmful open-ended outputs cannot always be scored by exact string matching.",
        true,
      ],
      [
        "A model can count as failing if it attempts harmful behavior even when the resulting harmful instruction is low quality.",
        true,
      ],
      [
        "Safety benchmarks are often less directly comparable than constrained knowledge benchmarks because safety policies can vary.",
        true,
      ],
    ],
    "Safety evaluation is partly policy-dependent and often more open-ended than multiple-choice knowledge evaluation. The source distinguishes model quality from safety intent: attempting the harmful behavior can be enough to count as an attack success even if the output is not skillful.",
  ),
  makeQuestion(
    "cme295-lect8-q29",
    "medium",
    "Which statements correctly describe tau-bench-style agent evaluation?",
    [
      [
        "It evaluates tool-agent-user interaction in domains with APIs, database state, and policies.",
        true,
      ],
      [
        "The user side can be simulated by another language model because later turns depend on earlier agent actions.",
        true,
      ],
      [
        "Success can be tied to task reward and resulting database or action state.",
        true,
      ],
      [
        "It is mainly a single-turn MMLU variant with four fixed answer choices.",
        false,
      ],
    ],
    "Agent benchmarks need to test the interaction among tools, policies, users, and stateful outcomes. Tau-bench-style evaluation is system-like rather than a static multiple-choice knowledge test.",
  ),
  makeQuestion(
    "cme295-lect8-q30",
    "hard",
    `An agent benchmark runs \\(n=8\\) independent attempts and observes \\(c=6\\) successes. If \\(Pass^k\\) is estimated as the probability that all \\(k=3\\) sampled attempts are successful, using \\(\\binom{c}{k} / \\binom{n}{k}\\), which value is correct?`,
    [
      ["\\(\\binom{6}{3} / \\binom{8}{3} = 20 / 56 \\approx 0.36\\).", true],
      [
        "\\(1 - \\binom{2}{3} / \\binom{8}{3} = 1.00\\), because at least one sampled attempt succeeds.",
        false,
      ],
      [
        "\\(6 / 8 = 0.75\\), because reliability for three attempts is the same as one-attempt success rate.",
        false,
      ],
      [
        "\\((6 / 8)^0 = 1.00\\), because successful attempts cancel out unsuccessful attempts.",
        false,
      ],
    ],
    "For this stated estimator, all three sampled attempts must come from the six successes, giving \\(\\binom{6}{3}\\) successful triples out of \\(\\binom{8}{3}\\) possible triples. The at-least-one-success formula is the Pass@k story, not the all-successes Pass^k story.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect8-q31",
    "hard",
    "Assertion: Pass^k is a stricter reliability signal than Pass@k for an agent that may be run repeatedly on the same task.\n\nReason: Pass^k asks whether all k attempts succeed, while Pass@k asks whether at least one of k attempts succeeds.",
    4,
    "The assertion is true because repeated real-world automation needs consistency, not merely one lucky success. The reason is true and explains the assertion: requiring every attempt to succeed is stricter than requiring any one attempt to succeed.",
  ),
  makeQuestion(
    "cme295-lect8-q32",
    "easy",
    "Which statements correctly capture the role of benchmark profiles?",
    [
      [
        "Benchmarks project performance onto particular axes such as knowledge, reasoning, coding, safety, or tool use.",
        true,
      ],
      [
        "Different models can be strong on different axes, so no single score fully describes a model.",
        true,
      ],
      [
        "A product team should compare benchmark results with its own workload rather than relying only on a leaderboard.",
        true,
      ],
      [
        "A high score on one benchmark guarantees the model is best for every use case.",
        false,
      ],
    ],
    "A benchmark is a projection, not a complete model identity. Product fit depends on the capabilities, costs, latency, safety constraints, and task mix that matter for the actual workload.",
  ),
  makeQuestion(
    "cme295-lect8-q33",
    "hard",
    `A team compares four models by quality score and cost per 1,000 requests:

| Model | Quality | Cost |
| --- | ---: | ---: |
| A | 88 | 20 |
| B | 86 | 10 |
| C | 84 | 10 |
| D | 80 | 6 |

Assume higher quality is better and lower cost is better. Which statement is correct about the quality-cost Pareto frontier?`,
    [
      [
        "Model C is not on the frontier because Model B has higher quality at the same cost.",
        true,
      ],
      [
        "Model A is not on the frontier because it is the most expensive model.",
        false,
      ],
      [
        "Model D is not on the frontier because every model with higher quality is also cheaper.",
        false,
      ],
      [
        "All four models are on the frontier because each one has a different quality score.",
        false,
      ],
    ],
    "A model is dominated when another model is at least as good on every objective and better on one. Model B dominates Model C by matching its cost and improving quality; A, B, and D each represent a different quality-cost tradeoff in this table.",
  ),
  makeQuestion(
    "cme295-lect8-q34",
    "hard",
    "Which precautions address benchmark data contamination risk?",
    [
      [
        "Use identifiers or hashes to detect benchmark examples in training data.",
        true,
      ],
      [
        "Block access to websites or resources that may contain benchmark answers during tool-use evaluations.",
        true,
      ],
      [
        "Evaluate on newer test versions when possible, especially for public exams.",
        true,
      ],
      [
        "Treat a high benchmark score as less decisive if the model may have seen the benchmark content during training.",
        true,
      ],
    ],
    "Data contamination weakens the assumption that a benchmark measures generalization rather than memorization. Hashes, blocklists, newer test versions, and cautious interpretation all target that risk from different angles.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect8-q35",
    "medium",
    "Assertion: Optimizing a model or product only for a benchmark leaderboard can make the benchmark less useful as a measure of real usefulness.\n\nReason: Goodhart's law says that when a measure becomes a target, it can stop being a good measure.",
    4,
    "The assertion is true because leaderboard optimization can overfit to the measured task while missing the product's real needs. The reason is true and explains the assertion: Goodhart's law is the lecture's framing for why benchmark scores need organic and workload-specific evaluation.",
  ),
  makeQuestion(
    "cme295-lect8-q36",
    "hard",
    "A team is evaluating a RAG answerer that retrieves policy snippets and writes final answers for employees. Which evaluation signals should be separated rather than merged into one vague score?",
    [
      [
        "Retrieval relevance: whether the retrieved snippets contain the needed policy evidence.",
        true,
      ],
      [
        "Answer factuality: whether claims in the answer are supported by the retrieved or trusted evidence.",
        true,
      ],
      [
        "User-facing quality: whether the answer follows instructions and is useful for the request.",
        true,
      ],
      [
        "Serving performance: latency, cost, and availability for the deployed system.",
        true,
      ],
    ],
    "A vague single evaluation target hides where the system succeeds or fails. Separating retrieval, factuality, user-facing answer quality, and system performance lets the team improve the right component instead of guessing.",
  ),
  makeQuestion(
    "cme295-lect8-q37",
    "hard",
    "A judge prompt says only: `Rate this answer from 1 to 10.` Which revisions make it closer to the recommended LLM-as-a-Judge design?",
    [
      [
        "Specify the criterion, such as relevance to the user request or factual support against provided evidence.",
        true,
      ],
      ["Ask for a short rationale before the score.", true],
      [
        "Return a structured object with parseable fields such as `rationale` and `score`.",
        true,
      ],
      [
        "Consider a binary pass/fail scale if the task does not truly need ten levels.",
        true,
      ],
    ],
    "The bare prompt gives the judge too little information and produces a noisy, hard-to-parse signal. Criteria, rationale, structure, and an appropriate scale make the evaluation more reproducible and more useful for downstream analysis.",
  ),
  makeQuestion(
    "cme295-lect8-q38",
    "hard",
    "A model is strong on broad knowledge and math benchmarks but weak on a company's tool-heavy support workflow. Which responses are consistent with the benchmark guidance?",
    [
      [
        "Add workflow-specific evaluations that include the relevant tools, policies, and state changes.",
        true,
      ],
      [
        "Use benchmark scores as part of the model profile rather than as the final product-fit decision.",
        true,
      ],
      [
        "Compare quality with cost, latency, safety, and context limits when choosing a deployment model.",
        true,
      ],
      [
        "Treat the weak tool-workflow result as evidence that knowledge and math benchmarks do not fully cover agent reliability.",
        true,
      ],
    ],
    "The lecture treats benchmark scores as projections of performance across specific axes, not as universal guarantees. A weak workflow-specific result is actionable evidence that the team needs agent and product evaluations in addition to broad knowledge or math scores.",
  ),
  makeQuestion(
    "cme295-lect8-q39",
    "hard",
    "An airline-support agent calls the right booking API and receives a successful database update, but its final message tells the user that nothing changed. Which diagnoses best fit?",
    [
      [
        "This is a response generation error because the final answer fails to convey the tool result.",
        true,
      ],
      [
        "Potential fixes include improving grounding in the synthesis model or making the tool output easier to interpret.",
        true,
      ],
      [
        "This is mainly a tool prediction error because the agent failed to choose the booking API.",
        false,
      ],
      [
        "This proves the benchmark should score only the final natural-language message and ignore database state.",
        false,
      ],
    ],
    "The tool selection and backend update succeeded, so the visible failure is in synthesizing the final response from the result. Agent benchmarks often need both state/action checks and final-answer checks because either layer can fail.",
  ),
  makeQuestion(
    "cme295-lect8-q40",
    "hard",
    "Which choices belong in a mature evaluation plan for an LLM product rather than in a single benchmark-only report?",
    [
      [
        "Human ratings with agreement tracking for subjective or high-value cases.",
        true,
      ],
      [
        "Automated judges with explicit criteria, structured outputs, and bias mitigation.",
        true,
      ],
      [
        "Reference or hard-coded metrics where the task format supports them.",
        true,
      ],
      [
        "Direct testing on the product's real workload, including cost, latency, reliability, and safety constraints.",
        true,
      ],
    ],
    "The lecture frames evaluation as a portfolio of signals rather than one universal metric. Human ratings, rule-based or hard-coded checks, LLM judges, system metrics, and workload-specific tests answer different questions and need to be interpreted together.",
  ),
  makeQuestion(
    "cme295-lect8-q41",
    "hard",
    `Two raters evaluate 100 model outputs with binary pass/fail labels:

|  | Rater B pass | Rater B fail |
| --- | ---: | ---: |
| Rater A pass | 48 | 12 |
| Rater A fail | 18 | 22 |

Using Cohen's kappa \\(\\kappa = \\frac{p_o - p_e}{1 - p_e}\\), which computation is correct?`,
    [
      [
        "\\(p_o = 0.70\\), \\(p_e = 0.60 \\times 0.66 + 0.40 \\times 0.34 = 0.532\\), so \\(\\kappa \\approx 0.36\\).",
        true,
      ],
      [
        "\\(p_o = 0.70\\), \\(p_e = 0.30\\), so \\(\\kappa = (0.70 - 0.30) / (1 - 0.30) \\approx 0.57\\) because all disagreements are treated as chance agreement.",
        false,
      ],
      [
        "\\(p_o = 0.48\\), \\(p_e = 0.532\\), so \\(\\kappa = (0.48 - 0.532) / (1 - 0.532) \\approx -0.11\\) because only pass-pass counts as observed agreement.",
        false,
      ],
      [
        "\\(p_o = 0.88\\), \\(p_e = 0.468\\), so \\(\\kappa = (0.88 - 0.468) / (1 - 0.468) \\approx 0.77\\) because off-diagonal cells are counted as agreement.",
        false,
      ],
    ],
    "Observed agreement counts both diagonal cells, so \\((48 + 22) / 100 = 0.70\\). The expected chance agreement uses the raters' marginal pass and fail rates, giving \\(0.532\\), and kappa is therefore about \\(0.36\\), not the raw agreement.",
  ),
  makeQuestion(
    "cme295-lect8-q42",
    "hard",
    `Two rubric versions produce the same raw agreement but different chance baselines:

| Rubric | Observed agreement \\(p_o\\) | Expected chance agreement \\(p_e\\) |
| --- | ---: | ---: |
| A | 0.82 | 0.70 |
| B | 0.82 | 0.45 |

Which conclusions follow from the chance-adjusted view?`,
    [
      [
        "Raw agreement alone hides the fact that Rubric A has a much higher expected agreement by chance.",
        true,
      ],
      [
        "Rubric B has the stronger kappa because \\((0.82 - 0.45) / (1 - 0.45)\\) exceeds \\((0.82 - 0.70) / (1 - 0.70)\\).",
        true,
      ],
      [
        "Rubric A is necessarily better because \\((0.82 - 0.70) / (1 - 0.70)\\) should be treated as larger than Rubric B's adjusted score.",
        false,
      ],
      [
        "A high kappa such as \\((0.82 - 0.45) / (1 - 0.45)\\) removes the need to inspect whether the rubric measures the intended evaluation criterion.",
        false,
      ],
    ],
    "Chance adjustment changes the interpretation of the same raw agreement. Rubric B has more agreement above chance, but even a stronger kappa is still a rater-consistency signal, not proof that the rubric captures the right product quality dimension.",
  ),
  makeQuestion(
    "cme295-lect8-q43",
    "hard",
    `A reference contains 10 unigrams and a generated answer contains 8 unigrams. Five generated unigrams match the reference. Ignoring ordering penalties, which metric interpretations are correct?`,
    [
      [
        "The unigram precision is \\(5/8 = 0.625\\), because five of the generated unigrams match the reference.",
        true,
      ],
      [
        "The unigram recall is \\(5/10 = 0.5\\), because five of the reference unigrams are recovered.",
        true,
      ],
      [
        "A metric that combines precision and recall would penalize both extra generated words and missing reference words.",
        true,
      ],
      [
        "A semantically perfect paraphrase with \\(0/8 = 0\\) shared generated unigrams would receive a perfect score under pure unigram overlap.",
        false,
      ],
    ],
    "Precision and recall answer different questions about overlap: how much generated text matched, and how much reference text was recovered. Pure overlap still misses semantic equivalence when a good paraphrase uses different wording.",
  ),
  makeQuestion(
    "cme295-lect8-q44",
    "hard",
    "A team wants pairwise LLM-as-a-Judge results that can support preference-data generation. Which design choices reduce avoidable bias or parsing noise?",
    [
      [
        "Evaluate both response orders and aggregate the swapped-order results instead of trusting one presentation order.",
        true,
      ],
      [
        "Use explicit criteria that tell the judge whether to prioritize correctness, relevance, concision, safety, or another dimension.",
        true,
      ],
      [
        "Return structured fields such as `rationale`, `winner`, and `confidence` so downstream code does not scrape prose.",
        true,
      ],
      [
        "Calibrate a sample of judge decisions against human preference labels before trusting the labels at scale.",
        true,
      ],
    ],
    "Pairwise judging can be useful for synthetic preference labels, but it is vulnerable to position, verbosity, and rubric ambiguity. Swapping order, specifying criteria, using structured output, and calibrating against humans make the labels more defensible.",
  ),
  makeQuestion(
    "cme295-lect8-q45",
    "hard",
    `A factuality evaluator decomposes one answer into weighted claims and uses this scoring rule: supported claims get 1 credit, contradicted claims get 0 credit, and unverifiable claims get 0.5 credit.

| Claim | Weight | Status |
| --- | ---: | --- |
| A | 0.35 | supported |
| B | 0.25 | contradicted |
| C | 0.20 | unverifiable |
| D | 0.20 | supported |

Which score follows from this rule?`,
    [
      [
        "\\(0.35 + 0.5 \\times 0.20 + 0.20 = 0.65\\), because supported claims receive full credit and the unverifiable claim receives half credit.",
        true,
      ],
      [
        "\\(0.35 + 0.25 + 0.20 + 0.20 = 1.00\\), because decomposition always preserves the full answer score.",
        false,
      ],
      [
        "\\(0.35 + 0.20 = 0.55\\), because unverifiable claims should receive full contradiction credit.",
        false,
      ],
      [
        "\\((1 + 0 + 0.5 + 1) / 4 = 0.625\\), because claim weights should be ignored after decomposition.",
        false,
      ],
    ],
    "The weighted score is computed by multiplying each claim's weight by the credit assigned to its status. This is why claim decomposition can represent partial factuality more precisely than an all-or-nothing label.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect8-q46",
    "hard",
    "Assertion: Low temperature can make repeated LLM-as-a-Judge calls more reproducible.\n\nReason: Structured output schemas make rationale and score fields easier to parse.",
    5,
    "The assertion is true because lower sampling randomness tends to reduce run-to-run variation in judge outputs. The reason is also true, but it explains parsing reliability rather than the causal mechanism for lower stochastic variation, so it is related but not the correct explanation.",
  ),
  makeQuestion(
    "cme295-lect8-q47",
    "hard",
    `A pairwise judge is tested on the same response pairs with swapped order:

| Prompt order | Judge selects first response | Judge selects second response |
| --- | ---: | ---: |
| A then B | 62% | 38% |
| B then A | 58% | 42% |

Which interpretations are justified?`,
    [
      [
        "The judge shows evidence of position bias because the first response is favored in both prompt orders.",
        true,
      ],
      [
        "A fairer aggregate should account for the swapped-order result rather than treating the 62% A-then-B result as the whole preference signal.",
        true,
      ],
      [
        "Response A is clearly better because it wins when shown first, regardless of what happens after swapping order.",
        false,
      ],
      [
        "The experiment proves pairwise evaluation is unusable for every task, even with order controls.",
        false,
      ],
    ],
    "The preference follows presentation position more than stable content, which is the signature risk behind position bias. Swapping and aggregating do not make judging perfect, but they reveal a problem that a single order would hide.",
  ),
  makeQuestion(
    "cme295-lect8-q48",
    "hard",
    `An agent workflow has three conditional success probabilities:

| Stage | Conditional success probability |
| --- | ---: |
| Tool prediction | 0.90 |
| Tool call after correct prediction | 0.80 |
| Response synthesis after successful call | 0.75 |

Assuming the stages must all succeed, which statements are correct?`,
    [
      [
        "The end-to-end success probability is \\(0.90 \\times 0.80 \\times 0.75 = 0.54\\).",
        true,
      ],
      [
        "Raising synthesis success from 0.75 to 0.90 would raise end-to-end success to \\(0.90 \\times 0.80 \\times 0.90 = 0.648\\).",
        true,
      ],
      [
        "Only judging the final answer would not reveal whether the main loss came from tool prediction, backend execution, or synthesis.",
        true,
      ],
      [
        "Improving tool prediction from 0.90 to 0.95 adds \\(0.05 \\times 0.80 \\times 0.75 = 0.03\\), which is larger than improving tool-call success by \\(0.90 \\times 0.05 \\times 0.75 = 0.03375\\).",
        false,
      ],
    ],
    "The chain succeeds only when each stage succeeds, so the probabilities multiply under this simplified setup. The marginal gain from improving a stage depends on the other stage probabilities; here improving tool-call success by 0.05 gives \\(0.90 \\times 0.05 \\times 0.75\\), which is larger than \\(0.05 \\times 0.80 \\times 0.75\\).",
  ),
  makeQuestion(
    "cme295-lect8-q49",
    "hard",
    "A RAG assistant is being evaluated for policy-question answering. Which metrics or checks target different parts of the system rather than duplicating the same signal?",
    [
      [
        "Retrieval recall@k or relevance judgments for whether the needed policy evidence appears in the retrieved context.",
        true,
      ],
      [
        "Claim-level factuality or faithfulness checks for whether answer claims are supported by the retrieved or trusted evidence.",
        true,
      ],
      [
        "Human or judge ratings for whether the final response is useful, concise, and responsive to the user request.",
        true,
      ],
      [
        "Latency and cost measurements for whether the deployed system can serve the workflow within operational constraints.",
        true,
      ],
    ],
    "A RAG system can fail by missing evidence, misusing evidence, writing a poor answer, or being too slow or expensive. Separating these measurements makes the evaluation actionable because each metric points to a different intervention.",
  ),
  makeQuestion(
    "cme295-lect8-q50",
    "hard",
    `A task attempt succeeds independently with probability \\(p = 0.8\\). For \\(k=3\\) attempts, which comparison between Pass@k and Pass^k is correct?`,
    [
      [
        "\\(Pass@3 = 1 - (1 - 0.8)^3 = 0.992\\), while \\(Pass^3 = 0.8^3 = 0.512\\).",
        true,
      ],
      [
        "\\(Pass@3 = 0.8^3 = 0.512\\), while \\(Pass^3 = 1 - (1 - 0.8)^3 = 0.992\\).",
        false,
      ],
      [
        "\\(Pass@3 = Pass^3 = 0.8\\), because both metrics reduce to the one-attempt success rate.",
        false,
      ],
      [
        "\\(Pass@3 = 3 \\times 0.8 = 2.4\\), while \\(Pass^3 = 1 / 3\\), because Pass@k counts expected successes.",
        false,
      ],
    ],
    "Pass@k asks whether at least one attempt succeeds, so it uses the complement of all attempts failing. Pass^k asks whether every attempt succeeds, which is the stricter reliability story for repeated agent use.",
  ),
  makeQuestion(
    "cme295-lect8-q51",
    "hard",
    `An agent benchmark run over 100 tasks records these outcomes:

| Outcome | Count |
| --- | ---: |
| Final message sounds successful | 90 |
| Database state is correct | 70 |
| Both final message and database state are correct | 65 |

Which evaluation conclusions are justified?`,
    [
      [
        "Scoring only the final message can overestimate task success when the required state change is missing.",
        true,
      ],
      [
        "A metric requiring both final-message correctness and database-state correctness is exactly 65% in this run.",
        true,
      ],
      [
        "The database state should be ignored because a fluent final message is the user-facing artifact.",
        false,
      ],
      [
        "Language-model-simulated users prevent hard-coded reward checks from being used at the end of a task.",
        false,
      ],
    ],
    "Stateful agent tasks need outcome checks in addition to natural-language quality. Here the database is correct in 70 tasks, but only 65 tasks have both a successful-sounding final message and the correct database state, so the stricter combined metric is 65%, not merely the looser upper bound of 70%.",
  ),
  makeQuestion(
    "cme295-lect8-q52",
    "hard",
    `A team compares models on quality, cost, and latency:

| Model | Quality | Cost | Latency |
| --- | ---: | ---: | ---: |
| A | 90 | 20 | 800 |
| B | 88 | 12 | 500 |
| C | 86 | 12 | 600 |
| D | 82 | 6 | 350 |
| E | 88 | 14 | 450 |

Higher quality is better; lower cost and latency are better. Which Pareto-frontier statements are correct?`,
    [
      [
        "Model C is dominated by Model B because B has higher quality, the same cost, and lower latency.",
        true,
      ],
      [
        "Model B does not dominate Model E because E has lower latency even though it costs more.",
        true,
      ],
      [
        "Model A can be on the frontier despite high cost and latency because no listed model has quality at least 90 with lower cost and latency.",
        true,
      ],
      [
        "Model D is dominated by every higher-quality model because quality is the only dimension used in a Pareto comparison.",
        false,
      ],
    ],
    "Pareto dominance requires being at least as good on every tracked objective and better on at least one. Model C is clearly dominated by B, while models with different tradeoffs can remain on the frontier even when they are not best on every dimension.",
  ),
  makeQuestion(
    "cme295-lect8-q53",
    "hard",
    "A lab worries that a public benchmark may have leaked into pretraining data or into tool-accessible web pages. Which precautions target that contamination risk?",
    [
      [
        "Hash or otherwise identify benchmark examples so overlaps with training or evaluation-accessible corpora can be detected.",
        true,
      ],
      [
        "Block tool access to known pages that contain benchmark answers during tool-use evaluations.",
        true,
      ],
      [
        "Prefer newer test versions when available, especially for public exams whose old solutions are widely available.",
        true,
      ],
      [
        "Report contamination checks alongside scores so readers know whether high performance might reflect memorization.",
        true,
      ],
    ],
    "Contamination is a threat to benchmark interpretation because the score may reflect exposure rather than general capability. Hash checks, blocklists, newer tests, and transparent reporting all reduce or disclose that risk.",
  ),
  makeQuestion(
    "cme295-lect8-q54",
    "hard",
    `A safety classifier is used to score open-ended harmful-behavior attempts. In one evaluation set, there are 200 harmful attempts and 800 safe outputs. The classifier has true positive rate 0.90 and false positive rate 0.05. Which quantities follow?`,
    [
      ["The expected true positives are \\(200 \\times 0.90 = 180\\).", true],
      ["The expected false positives are \\(800 \\times 0.05 = 40\\).", true],
      ["The expected number of flagged outputs is \\(180 + 40 = 220\\).", true],
      [
        "The expected precision among flagged outputs is \\(180 / 220 \\approx 0.82\\).",
        true,
      ],
    ],
    "Classifier-based safety evaluation adds another model whose errors affect the final metric. Even with a high true positive rate, false positives matter because they change how many flagged outputs are actually harmful attempts.",
  ),
  makeQuestion(
    "cme295-lect8-q55",
    "hard",
    `A team must evaluate 10,000 model answers. Human review costs \\($0.50\\) per answer. An automated judge costs \\($0.002\\) per answer, and the team plans to human-review a 500-answer calibration sample. Which statements are sound?`,
    [
      [
        "Judging all 10,000 answers automatically costs \\($20\\), while human-reviewing all 10,000 costs \\($5,000\\).",
        true,
      ],
      [
        "A 500-answer human calibration sample costs \\($250\\), which can be used to compare judge outputs against human ratings.",
        true,
      ],
      [
        "The lower automated cost ratio \\(5000 / 20 = 250\\) proves the automated judge is more accurate than human raters.",
        false,
      ],
      [
        "Because the automated judge costs \\(10000 \\times 0.002 = 20\\), the evaluation no longer needs criteria, structure, or bias checks.",
        false,
      ],
    ],
    "Cost and speed motivate automated judging, but they do not establish accuracy. A calibration sample is a practical compromise: it keeps human cost manageable while checking whether the judge approximates the desired human signal.",
  ),
  makeQuestion(
    "cme295-lect8-q56",
    "hard",
    `Two raters agree on 95% of examples, but because almost every example belongs to the same label, the expected chance agreement is 93%. Which conclusions are appropriate?`,
    [
      [
        "The kappa-like chance-adjusted signal is modest because \\((0.95 - 0.93) / (1 - 0.93) \\approx 0.29\\).",
        true,
      ],
      [
        "Class imbalance can make raw agreement look high even when agreement above chance is limited.",
        true,
      ],
      [
        "Inspecting the confusion matrix and rubric is still important because raw agreement alone is not enough context.",
        true,
      ],
      [
        "The 95% raw agreement proves the raters are strongly aligned on the difficult minority cases even though \\((0.95 - 0.93) / (1 - 0.93) \\approx 0.29\\).",
        false,
      ],
    ],
    "When one label dominates, raters can agree often by following the base rate. Chance-adjusted metrics expose that issue, but the next step is still diagnostic inspection of where raters disagree and whether the rubric handles minority cases well.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect8-q57",
    "hard",
    "Assertion: A model with a high MMLU score is guaranteed to be reliable in multi-step tool workflows.\n\nReason: MMLU uses constrained answer choices across many knowledge tasks.",
    2,
    "The assertion is false because broad constrained knowledge accuracy does not guarantee correct tool selection, backend use, policy compliance, state changes, or repeated reliability. The reason is true: MMLU-style evaluation is built from many constrained knowledge tasks, which is exactly why it should not be treated as a full agent-workflow guarantee.",
  ),
  makeQuestion(
    "cme295-lect8-q58",
    "hard",
    "A team wants to compare two answer generators with an LLM judge and then decide whether the difference is meaningful. Which additions make the evaluation more statistically and operationally credible?",
    [
      [
        "Randomize or swap answer order so position does not systematically favor one generator.",
        true,
      ],
      [
        "Report the number of evaluated examples and uncertainty, such as a confidence interval or bootstrap interval.",
        true,
      ],
      [
        "Stratify or inspect results by task type so one easy slice does not hide failures on harder tasks.",
        true,
      ],
      [
        "Check a human-labeled sample so the automated judge's preferences are not accepted without calibration.",
        true,
      ],
    ],
    "Judge quality and sampling uncertainty both matter when model comparisons become decisions. Order controls, uncertainty estimates, slice analysis, and human calibration all reduce the chance that a leaderboard-style number hides a fragile result.",
  ),
  makeQuestion(
    "cme295-lect8-q59",
    "hard",
    `A BLEU-style brevity penalty is \\(BP = e^{1-r/c}\\) when candidate length \\(c\\) is shorter than reference length \\(r\\), and \\(BP = 1\\) otherwise. If \\(c=8\\) and \\(r=10\\), which statements are correct?`,
    [
      ["The penalty is \\(e^{1-10/8} = e^{-0.25} \\approx 0.78\\).", true],
      [
        "The penalty discourages a candidate from gaining precision by being too short.",
        true,
      ],
      [
        "The penalty becomes \\(e^{1-8/10} = e^{0.2} \\approx 1.22\\) because the candidate is shorter than the reference.",
        false,
      ],
      [
        "The penalty directly measures factuality as \\(1 - \\text{unsupported claim weight}\\) rather than measuring length.",
        false,
      ],
    ],
    "The brevity penalty is a length correction inside an overlap metric, not a factuality check. It addresses one way to game precision-focused overlap scores: producing a short candidate that matches a few reference words but omits important content.",
  ),
  makeQuestion(
    "cme295-lect8-q60",
    "hard",
    `A product team has these deployment constraints: latency must be at most 700 ms and safety score must be at least 93. It compares three models:

| Model | Quality | Cost | Latency | Safety |
| --- | ---: | ---: | ---: | ---: |
| A | 92 | 30 | 1200 | 98 |
| B | 89 | 10 | 500 | 94 |
| C | 91 | 15 | 900 | 88 |

Which model-selection conclusions are consistent with benchmark-overreach guidance?`,
    [
      [
        "Model B is the only listed model that satisfies both the latency and safety constraints.",
        true,
      ],
      [
        "Model C's high quality score does not make it feasible because it fails both the latency and safety constraints.",
        true,
      ],
      [
        "A quality-only leaderboard would choose Model A, but the product constraints point to a different deployment choice.",
        true,
      ],
      [
        "The Pareto-frontier idea makes safety constraints irrelevant once quality and cost are known.",
        false,
      ],
    ],
    "The model with the highest quality score is not automatically the right product choice. Real selection depends on the whole profile, including latency, cost, safety, and workload constraints that a single benchmark axis may not represent.",
  ),
];
