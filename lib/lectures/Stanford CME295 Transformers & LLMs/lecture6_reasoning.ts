import { Question } from "../../quiz";

type Lecture6Difficulty = "easy" | "medium" | "hard";
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
  difficulty: Lecture6Difficulty,
  prompt: string,
  optionSeeds: readonly OptionSeed[],
  explanation: string,
): Question {
  if (optionSeeds.length !== 4) {
    throw new Error(`CME295 Lecture 6 question ${id} needs 4 options.`);
  }

  return {
    id,
    chapter: 6,
    difficulty,
    prompt,
    options: optionSeeds.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

function makeAssertionReasonQuestion(
  id: string,
  difficulty: Lecture6Difficulty,
  prompt: string,
  correctChoice: AssertionReasonChoice,
  explanation: string,
): Question {
  return {
    id,
    chapter: 6,
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

export const stanfordCME295Lecture6ReasoningQuestions: Question[] = [
  makeQuestion(
    "cme295-lect6-q01",
    "easy",
    "A product team is deciding whether a standard next-token assistant is enough for several tasks. Which statements correctly separate ordinary strengths from the reasoning-focused weaknesses that motivate reasoning models?",
    [
      [
        "Generating plausible prose, summaries, or code snippets fits the imitation strengths of a standard language model.",
        true,
      ],
      [
        "Solving a hard math or coding problem may require multi-step decomposition rather than immediate answer emission.",
        true,
      ],
      [
        "A pretraining knowledge cutoff can block questions whose answer depends on events after the training data was collected.",
        true,
      ],
      [
        "Static pretraining gives the model live access to current external tools, databases, and ordering systems.",
        false,
      ],
    ],
    "The motivation is not that standard language models are useless; they are strong imitators and code/prose generators. The limits are different: multi-step reasoning, stale knowledge, lack of action, and free-form evaluation require additional mechanisms or external systems.",
  ),
  makeQuestion(
    "cme295-lect6-q02",
    "easy",
    "Which prompts most directly test reasoning rather than only stored factual knowledge or tool use?",
    [
      [
        "Given a bear born in 2020 and a stated current year of 2025, compute the bear's age.",
        true,
      ],
      [
        "Given three failed unit-test traces and a small function, infer the bug and propose a fixed implementation.",
        true,
      ],
      [
        "Name the course code for a class when the code is already a memorized fact.",
        false,
      ],
      [
        "Place an order in a live external system without any tool interface.",
        false,
      ],
    ],
    "Reasoning tasks require the model to combine provided facts through steps, such as arithmetic or debugging. A memorized lookup is mainly a knowledge task, while live ordering is an action/tool-use problem rather than a pure reasoning problem.",
  ),
  makeQuestion(
    "cme295-lect6-q03",
    "easy",
    "Why can chain-of-thought style generation improve a model's answer on hard problems?",
    [
      [
        "It lets the model decompose a hard prompt into smaller steps that resemble patterns it has learned.",
        true,
      ],
      [
        "It spends more generation steps, and each generated token involves another forward pass through the model.",
        true,
      ],
      [
        "It gives the model new post-training facts that were not present in the prompt, tools, or parameters.",
        false,
      ],
      [
        "It guarantees that every intermediate statement in the trace is faithful and correct.",
        false,
      ],
    ],
    "The useful intuition is decomposition plus extra test-time compute, not guaranteed truth. Reasoning traces can help a model search through a problem, but the trace is still generated text and must be judged by correctness and reliability.",
  ),
  makeQuestion(
    "cme295-lect6-q04",
    "easy",
    "A chat product shows a brief thought summary while billing for additional output tokens used by a reasoning model. Which statements correctly interpret this interface?",
    [
      [
        "The model may produce internal reasoning tokens before the final answer even if the raw chain is not shown to the user.",
        true,
      ],
      [
        "A displayed summary can be different from the complete generated reasoning trace.",
        true,
      ],
      [
        "Reasoning tokens affect latency and cost because they are part of the generated output budget.",
        true,
      ],
      [
        "Hiding the full trace proves that the model did not generate any intermediate reasoning.",
        false,
      ],
    ],
    "Reasoning-model interfaces often expose a short summary or a thinking indicator rather than the full chain. The hidden or summarized tokens can still consume compute and cost, so users and providers care about getting enough reasoning without unlimited output length.",
  ),
  makeQuestion(
    "cme295-lect6-q05",
    "easy",
    "Which benchmark setups fit reasoning-model evaluation with verifiable outcomes?",
    [
      [
        "A coding task where generated solutions are run against test cases.",
        true,
      ],
      [
        "A math task where the final answer can be compared with a known ground truth.",
        true,
      ],
      [
        "A bug-fixing task where the patch is checked by compilation and regression tests.",
        true,
      ],
      [
        "A creative writing task scored only by whether the output sounds long and confident.",
        false,
      ],
    ],
    "Coding and math benchmarks are attractive because they can often verify outcomes without a learned reward model. Length and confidence are not correctness checks, and using them alone would reward style rather than reliable reasoning.",
  ),
  makeQuestion(
    "cme295-lect6-q06",
    "medium",
    "A benchmark reports Pass@k, Pass@1, and Cons@k for the same model. Which interpretations are correct?",
    [
      [
        "Pass@k estimates whether at least one of k sampled attempts succeeds.",
        true,
      ],
      [
        "Pass@1 is the single-generation special case and matters when only one answer will be used.",
        true,
      ],
      [
        "Cons@k compares the majority-vote answer among k generations with the ground truth.",
        true,
      ],
      [
        "A high Pass@k score means every individual generation is reliable enough for single-shot use.",
        false,
      ],
    ],
    "Pass@k can be high because one attempt out of several succeeds, especially when verification is cheap. Pass@1 asks about one response, while Cons@k is closer to self-consistency by aggregating the most common answer.",
  ),
  makeQuestion(
    "cme295-lect6-q07",
    "hard",
    `A model generates \\(n=10\\) candidate solutions for a coding problem, and \\(c=3\\) pass all tests. Using the standard without-replacement estimate,

\\[
\\mathrm{Pass@}k = 1 - \\frac{\\binom{n-c}{k}}{\\binom{n}{k}},
\\]

what is \\(\\mathrm{Pass@}2\\)?`,
    [
      [
        "\\(1 - \\frac{\\binom{7}{2}}{\\binom{10}{2}} = \\frac{24}{45}\\), about 0.533.",
        true,
      ],
      [
        "\\(\\frac{3}{10} = 0.3\\), because Pass@2 is the same as the observed pass fraction.",
        false,
      ],
      [
        "\\(1 - \\frac{\\binom{3}{2}}{\\binom{10}{2}} = \\frac{42}{45}\\), because the numerator should count successful pairs.",
        false,
      ],
      [
        "\\(\\frac{6}{10} = 0.6\\), because two attempts double the single-attempt success rate exactly.",
        false,
      ],
    ],
    "The complement event is that both sampled attempts come from the seven failed candidates, so the estimate is \\(1 - \\binom{7}{2}/\\binom{10}{2}\\). Doubling the observed pass fraction ignores overlap and the without-replacement calculation, while using successful pairs computes the wrong complement.",
  ),
  makeQuestion(
    "cme295-lect6-q08",
    "easy",
    "When the Pass@k estimator is specialized to k = 1, what should it reduce to?",
    [
      [
        "The fraction of generated candidates that were successful, \\(c/n\\).",
        true,
      ],
      [
        "The probability that a single sampled attempt succeeds, equivalently \\(1 - (n-c)/n\\).",
        true,
      ],
      [
        "The probability that all \\(n\\) generated attempts succeed, which would be closer to \\((c/n)^n\\) under an independence approximation.",
        false,
      ],
      [
        "The majority-vote answer across \\(k\\) attempts, closer to a Cons@\\(k\\) aggregation.",
        false,
      ],
    ],
    "With one attempt, the complement formula reduces to the ordinary observed success fraction. Majority vote belongs to Cons@k or self-consistency, and requiring every generated attempt to succeed is a much stricter event.",
  ),
  makeQuestion(
    "cme295-lect6-q09",
    "medium",
    "Why do benchmark papers usually report the sampling temperature used for Pass@k-style evaluation?",
    [
      [
        "Very low temperature can make repeated samples nearly identical, limiting the benefit of larger k.",
        true,
      ],
      [
        "Very high temperature can add diversity while also lowering the quality of individual generations.",
        true,
      ],
      [
        "Temperature changes the benchmark ground truth labels, so the correct answer may become different.",
        false,
      ],
      [
        "Temperature removes the need to verify generated coding or math solutions.",
        false,
      ],
    ],
    "Pass@k depends on both quality and diversity. Temperature is a sampling control, not a change to the task or labels, so it should be documented because it can affect how often multiple attempts discover a correct solution.",
  ),
  makeQuestion(
    "cme295-lect6-q10",
    "easy",
    "A model generates five final answers to the same math problem: 17, 17, 16, 17, and 15. The ground truth is 17. Which metric behavior is being used if the evaluator takes 17 because it appears most often, then compares it with the ground truth?",
    [
      ["Cons@k or self-consistency-style majority voting.", true],
      [
        "PPO clipping, because the largest answer is clipped to the old policy.",
        false,
      ],
      [
        "Formatting reward, because the answer appears in a consistent tag.",
        false,
      ],
      [
        "Knowledge-cutoff evaluation, because the answer was memorized before training ended.",
        false,
      ],
    ],
    "The key operation is aggregating several sampled answers by their consensus before checking correctness. PPO clipping and formatting rewards are training mechanisms, while knowledge cutoff is about stale facts rather than majority-vote evaluation.",
  ),
  makeQuestion(
    "cme295-lect6-q11",
    "easy",
    "Why is reinforcement learning attractive for training reasoning behavior when no high-quality reasoning traces are already available?",
    [
      [
        "Hand-writing long reasoning chains at scale is expensive and difficult.",
        true,
      ],
      [
        "Human-written reasoning may constrain the model to human-style traces rather than whatever trace helps it solve problems.",
        true,
      ],
      [
        "Coding and math tasks can provide natural outcome rewards through tests or answer checks.",
        true,
      ],
      ["Reinforcement learning removes the need for any reward signal.", false],
    ],
    "The appeal is that verifiable tasks supply outcome rewards even when supervised reasoning traces are scarce. Reinforcement learning still needs rewards and careful constraints; it is not training without feedback.",
  ),
  makeQuestion(
    "cme295-lect6-q12",
    "easy",
    "Which reward components match the simple verifiable-reward setup used to teach a model to reason before answering?",
    [
      [
        "A formatting reward that checks for required reasoning and answer delimiters.",
        true,
      ],
      [
        "An answer-correctness reward that checks tests, compilation, or a known math answer.",
        true,
      ],
      [
        "A reward that treats longer reasoning as automatically better than shorter reasoning.",
        false,
      ],
      [
        "A reward that gives full credit for a nicely formatted trace even when the final answer is wrong.",
        false,
      ],
    ],
    "Formatting rewards encourage the model to produce the expected structure, while correctness rewards check whether the solution works. Formatting alone is too weak because a well-tagged but wrong answer is not successful reasoning.",
  ),
  makeQuestion(
    "cme295-lect6-q13",
    "medium",
    "A team wants to use verifiable rewards for every assistant task. Which limitations should they keep in mind?",
    [
      [
        "Many open-ended tasks do not have a cheap deterministic correctness check.",
        true,
      ],
      [
        "A formatting reward can be satisfied without the reasoning being useful or true.",
        true,
      ],
      [
        "A final-answer check may miss whether the intermediate reasoning was faithful or robust.",
        true,
      ],
      [
        "Verifiable rewards are useless for coding and math because tests and ground-truth answers cannot be used.",
        false,
      ],
    ],
    "Verifiable rewards are powerful in domains with checkable outcomes, but they do not solve every evaluation problem. They can miss process quality and are harder to apply when correctness is subjective, open-ended, or dependent on missing external information.",
  ),
  makeQuestion(
    "cme295-lect6-q14",
    "medium",
    "A reasoning service must decide how much thinking to allocate per prompt. Which controls or constraints are relevant?",
    [
      [
        "Dynamic budgets can allocate more thinking to harder prompts and less to simple ones.",
        true,
      ],
      [
        "Context awareness matters because reasoning tokens consume finite context window space.",
        true,
      ],
      [
        "Budget forcing can push a model to continue thinking or stop and answer.",
        true,
      ],
      [
        "Continuous or latent thoughts explore reasoning representations that are not ordinary visible language tokens.",
        true,
      ],
    ],
    "All four are real control ideas from the reasoning-model setting. The shared theme is that test-time scaling is a resource allocation problem: more thinking can help, but context, latency, cost, and representation choices all matter.",
  ),
  makeQuestion(
    "cme295-lect6-q15",
    "easy",
    "Which interventions are examples of budget forcing rather than training-data curation?",
    [
      [
        'Injecting a token such as "wait" to make the model continue its reasoning path.',
        true,
      ],
      [
        'Injecting a phrase like "time is up, now answer" to force the model to stop reasoning.',
        true,
      ],
      [
        "Collecting human-rewritten reasoning traces for cold-start supervised fine-tuning.",
        false,
      ],
      [
        "Filtering generated responses with a judge to create a later supervised fine-tuning set.",
        false,
      ],
    ],
    "Budget forcing is an inference-time control over how long the model continues or when it stops. Human rewriting and rejection sampling are data-pipeline choices used during training or fine-tuning, not direct token-level controls during one inference.",
  ),
  makeQuestion(
    "cme295-lect6-q16",
    "medium",
    "Which statements correctly describe Group Relative Policy Optimization (GRPO) at a high level?",
    [
      ["It samples a group of completions for the same prompt.", true],
      [
        "It estimates each completion's advantage relative to rewards from the group.",
        true,
      ],
      [
        "It avoids training a separate value model as part of the policy update.",
        true,
      ],
      [
        "It updates the policy without any clipping, KL, or other drift control.",
        false,
      ],
    ],
    "GRPO keeps the policy-optimization framing but changes how the advantage is estimated. It uses group-relative rewards instead of a jointly trained value model, while still relying on update-control ideas such as ratios, clipping, and reference-model pressure.",
  ),
  makeQuestion(
    "cme295-lect6-q17",
    "hard",
    `A GRPO batch samples four completions for the same prompt and obtains these rewards:

| Completion | Reward |
| --- | ---: |
| A | 1.0 |
| B | 0.0 |
| C | 0.5 |
| D | 0.5 |

Ignoring standard-deviation scaling, which completion has the largest positive group-relative advantage?`,
    [
      [
        "Completion A, because its reward is above the group average of 0.5.",
        true,
      ],
      [
        "Completion B, because it gives the shortest response and therefore should be upweighted.",
        false,
      ],
      [
        "Completions C and D, because matching the group average creates the largest positive advantage.",
        false,
      ],
      [
        "All completions, because GRPO treats every sampled response as better than a value model.",
        false,
      ],
    ],
    "The average reward is \\((1.0 + 0.0 + 0.5 + 0.5)/4 = 0.5\\). A is above that baseline, B is below it, and C and D are at the baseline, so only A has the largest positive relative advantage in this simplified calculation.",
  ),
  makeQuestion(
    "cme295-lect6-q18",
    "medium",
    "Which comparisons between GRPO and PPO are correct in reasoning-model training?",
    [
      [
        "PPO commonly trains or uses a value function to estimate advantages, while GRPO replaces that with group-relative reward comparison.",
        true,
      ],
      [
        "Both methods can use a current-policy to old-policy probability ratio to control the update.",
        true,
      ],
      [
        "GRPO requires training a larger value model than PPO, which is its main cost.",
        false,
      ],
      [
        "PPO and GRPO are unrelated to reinforcement learning because they are pure supervised fine-tuning losses.",
        false,
      ],
    ],
    "The key simplification of GRPO is avoiding the separate value-function training used in PPO-style advantage estimation. Both still live in the policy-optimization family and can use ratio/clipping mechanics to keep updates from moving too far.",
  ),
  makeQuestion(
    "cme295-lect6-q19",
    "hard",
    "In many language-model reinforcement-learning implementations, how does the KL/reference-model term fit into PPO-style or GRPO-style training?",
    [
      [
        "It discourages the tuned policy from drifting too far from a reference or base model.",
        true,
      ],
      [
        "It can be incorporated as a per-token penalty or appear explicitly in the objective, depending on the formulation.",
        true,
      ],
      [
        "It complements clipping because clipping controls update ratios while KL controls distance from the reference distribution.",
        true,
      ],
      [
        "It replaces the task reward, so answer correctness is no longer needed.",
        false,
      ],
    ],
    "The KL term is a guardrail, not the whole objective. Correctness or preference reward says what behavior is desirable, while KL/reference pressure and clipping help prevent destructive policy drift during optimization.",
  ),
  makeQuestion(
    "cme295-lect6-q20",
    "hard",
    "A GRPO trainer normalizes each completion's token contributions by that completion's own length. Which consequence follows when an output has negative advantage?",
    [
      [
        "Tokens in a short bad output can receive a larger downweighting per token than tokens in a long bad output.",
        true,
      ],
      [
        "The objective can create an incentive where long bad outputs are penalized less per token than short bad outputs.",
        true,
      ],
      [
        "The normalization proves that long outputs are always more correct than short outputs.",
        false,
      ],
      [
        "The normalization makes output length irrelevant to the objective.",
        false,
      ],
    ],
    "Dividing by each output's length changes the per-token contribution based on where the token appears. For negative-advantage completions, a long output can spread the penalty over more tokens, which helps explain why length can keep growing even after performance saturates.",
  ),
  makeQuestion(
    "cme295-lect6-q21",
    "medium",
    "Which statements correctly diagnose the increasing-output-length phenomenon in reasoning RL?",
    [
      [
        "Average response length can rise through RL training as reasoning traces become longer.",
        true,
      ],
      [
        "Early length increases can correlate with better benchmark performance.",
        true,
      ],
      [
        "After performance stabilizes, continued length growth can waste cost and latency.",
        true,
      ],
      [
        "Continued length growth also creates provider-side efficiency pressure because more output tokens must be generated.",
        true,
      ],
    ],
    "Longer traces can be useful up to a point because they give the model more steps to solve the problem. The pathology appears when optimization keeps increasing length after accuracy has stopped improving, turning reasoning budget into unnecessary user cost and provider-side generation load.",
  ),
  makeQuestion(
    "cme295-lect6-q22",
    "medium",
    "What is the basic mitigation idea behind DAPO-style or Dr. GRPO-style changes to the length term?",
    [
      [
        "Avoid giving tokens different contribution scales merely because they sit in longer or shorter completions.",
        true,
      ],
      [
        "Equalize or remove the output-length normalization factor that can create a verbosity incentive.",
        true,
      ],
      [
        "Keep correctness rewards while reducing the incentive for incorrect long answers to be relatively cheap.",
        true,
      ],
      [
        "Delete all reasoning tokens so the model can only answer directly.",
        false,
      ],
    ],
    "The fix is not to ban reasoning. It is to change the objective so token-level credit and penalty do not make verbosity attractive for the wrong reason, especially for incorrect completions.",
  ),
  makeQuestion(
    "cme295-lect6-q23",
    "hard",
    "Which later GRPO-related adjustments target issues other than the basic value-model simplification?",
    [
      [
        "Changing how the standard deviation in group-relative advantages affects very easy or very hard prompts.",
        true,
      ],
      [
        "Using asymmetric clipping bounds so very low-probability tokens have a fairer chance to increase without letting high-probability tokens collapse.",
        true,
      ],
      [
        "Replacing all rewards with a fixed command to produce a longer answer.",
        false,
      ],
      [
        "Removing sampling diversity so every group completion is identical.",
        false,
      ],
    ],
    "The technical adjustments refine optimization behavior after the basic GRPO idea is in place. Difficulty bias, clipping asymmetry, and diversity are nearby concerns; making every sample identical or rewarding length alone would undermine the group-relative setup.",
  ),
  makeQuestion(
    "cme295-lect6-q24",
    "hard",
    "A group contains rewards [0, 0, 0, 1] for a hard prompt and [0, 1, 1, 1] for an easy prompt. Why can the standard deviation term in a group-relative advantage formula require care?",
    [
      [
        "It can change the scale of advantages depending on how rewards are distributed inside the group, creating difficulty-related bias.",
        true,
      ],
      [
        "It makes all completions with reward 1 have exactly the same update in every prompt regardless of the rest of the group.",
        false,
      ],
      ["It removes the need to know which completions were correct.", false],
      [
        "It turns the algorithm into supervised fine-tuning on human-written chain-of-thought data.",
        false,
      ],
    ],
    "Group-relative normalization does more than subtract an average; the scale term can interact with whether most sampled completions fail or succeed. That is why later work studies difficulty bias instead of assuming the same reward label has the same optimization meaning in every group.",
  ),
  makeQuestion(
    "cme295-lect6-q25",
    "easy",
    "Which statements correctly describe the R1-Zero style recipe?",
    [
      [
        "It starts from a pretrained base model rather than a fully aligned assistant.",
        true,
      ],
      ["It applies GRPO on reasoning data using verifiable rewards.", true],
      [
        "It can improve reasoning benchmarks without an initial supervised fine-tuning stage.",
        true,
      ],
      [
        "It may produce rough chains with formatting, readability, syntax, or language-mixing issues.",
        true,
      ],
    ],
    "R1-Zero is useful as a proof of concept: reinforcement learning with verifiable rewards can elicit reasoning from a base model. The same minimal recipe also explains its rough edges, because it lacks the supervised anchoring that improves formatting and readability.",
  ),
  makeQuestion(
    "cme295-lect6-q26",
    "medium",
    "Which properties distinguish the full R1-style training pipeline from the R1-Zero proof of concept?",
    [
      [
        "The full R1-style pipeline adds a small cold-start supervised fine-tuning stage using cleaned reasoning traces.",
        true,
      ],
      [
        "The full R1-style pipeline runs a reasoning-focused GRPO stage after that cold start.",
        true,
      ],
      [
        "The full R1-style pipeline includes a larger supervised fine-tuning mixture with reasoning and non-reasoning data.",
        true,
      ],
      [
        "The full R1-style pipeline adds a final RL stage that also accounts for helpfulness and harmlessness on general data.",
        true,
      ],
    ],
    "The full pipeline is not just base model plus one RL stage. It uses supervised cleanup, reasoning RL, larger mixed SFT, and final RL so the model is both strong on reasoning tasks and more usable as an assistant.",
  ),
  makeQuestion(
    "cme295-lect6-q27",
    "medium",
    "Which data sources or filters fit the full R1 training recipe?",
    [
      [
        "Human-rewritten long reasoning traces from an earlier reasoning model can serve as cold-start SFT data.",
        true,
      ],
      [
        "General non-reasoning instruction data can be mixed in so the model remains useful outside math and code.",
        true,
      ],
      [
        "Rejection sampling can keep high-quality generated reasoning responses and discard weaker ones.",
        true,
      ],
      [
        "The supervised stages and RL stages serve different roles rather than being mutually exclusive.",
        true,
      ],
    ],
    "The R1-style pipeline combines reinforcement learning with supervised data rather than treating them as mutually exclusive. Rewritten traces, general SFT data, and rejection-sampled reasoning outputs each target a different usability or capability gap.",
  ),
  makeQuestion(
    "cme295-lect6-q28",
    "hard",
    "In a final RL stage that mixes reasoning and general assistant data, which reward design choices are appropriate?",
    [
      [
        "Use formatting and answer-correctness rewards for math, coding, and logic-style reasoning tasks.",
        true,
      ],
      [
        "Use helpfulness rewards for user-visible answers on general assistant tasks.",
        true,
      ],
      [
        "Apply harmlessness pressure broadly enough that unsafe content is not hidden inside the reasoning trace.",
        true,
      ],
      [
        "Avoid treating hidden reasoning length by itself as sufficient evidence of quality.",
        true,
      ],
    ],
    "Reasoning rewards and assistant-behavior rewards serve different parts of the training mixture. A model that reasons well but is unsafe, unreadable, or unhelpful is not the final target, so final RL has to cover both capability and user-facing behavior.",
  ),
  makeQuestion(
    "cme295-lect6-q29",
    "hard",
    "Which statements correctly compare R1-Zero and full R1?",
    [
      [
        "R1-Zero demonstrates that RL with verifiable rewards can discover reasoning behavior from a base model.",
        true,
      ],
      [
        "Full R1 adds data stages to improve readability, language consistency, and general assistant usefulness.",
        true,
      ],
      [
        "Full R1's added stages mean the model is not just optimized for benchmark correctness and tags.",
        true,
      ],
      [
        "Their difference is a training-recipe and usability difference, not merely a benchmark-score label.",
        true,
      ],
    ],
    "The important contrast is recipe and usability, not just scores. R1-Zero shows what pure reasoning RL can do, while full R1 adds cold-start SFT, mixed SFT, and broader RL to address the roughness that pure RL leaves behind.",
  ),
  makeQuestion(
    "cme295-lect6-q30",
    "medium",
    "How does reasoning distillation differ from the earlier distribution-matching distillation idea?",
    [
      [
        "Earlier distillation can train a student to match the teacher's next-token probability distribution.",
        true,
      ],
      [
        "Reasoning distillation can generate full teacher responses, including reasoning traces, then use SFT-style training on those sequences.",
        true,
      ],
      [
        "Reasoning distillation requires the student to be larger than the teacher.",
        false,
      ],
      [
        "Reasoning distillation means training only on final answers while discarding all reasoning traces.",
        false,
      ],
    ],
    "The distillation label is used for two related but different transfer mechanisms. In the reasoning setting, the practical move is often to let a strong teacher generate trace-plus-answer sequences and train a smaller student to imitate those outputs.",
  ),
  makeQuestion(
    "cme295-lect6-q31",
    "hard",
    "Why can distilling a large reasoning model into smaller students be a good use of compute?",
    [
      [
        "The expensive teacher can generate high-quality reasoning examples offline.",
        true,
      ],
      [
        "A smaller student can be trained with supervised fine-tuning on those examples.",
        true,
      ],
      [
        "For smaller models, imitation from a strong reasoning teacher can outperform trying to rediscover the behavior with RL from scratch.",
        true,
      ],
      [
        "Successful distillation removes the need to run the full teacher for every future query.",
        true,
      ],
    ],
    "Distillation shifts some cost from repeated inference to an offline data-generation and training pipeline. It is not free, but it can make reasoning behavior more available when running the giant teacher model directly is too expensive or too slow.",
  ),
  makeQuestion(
    "cme295-lect6-q32",
    "hard",
    "A benchmark table shows non-reasoning models clustered below reasoning models on math and coding tasks, while a new open model is competitive with closed reasoning models. Which conclusions are justified?",
    [
      [
        "Reasoning-specific training can produce a distinct performance cluster on reasoning-heavy benchmarks.",
        true,
      ],
      [
        "Open recipes with verifiable rewards and GRPO-style training can narrow the gap to closed reasoning systems.",
        true,
      ],
      [
        "The result still needs to be interpreted with the benchmark's sampling and verification setup in mind.",
        true,
      ],
      [
        "A strong benchmark table proves the model will act correctly in live external environments without tools.",
        false,
      ],
    ],
    "Math and coding benchmarks measure important reasoning capabilities, especially when verification is clear. They do not prove action-taking ability, live knowledge, or safe deployment in tool-using environments, so the conclusion should stay tied to the evaluated setup.",
  ),
  makeQuestion(
    "cme295-lect6-q33",
    "medium",
    "Which statements correctly connect reasoning models to the earlier preference-tuning/RLHF setup?",
    [
      ["The language model can be viewed as a policy over next tokens.", true],
      [
        "Choosing the next token is analogous to taking an action from that policy.",
        true,
      ],
      [
        "A generated completion can receive a reward signal used for policy optimization.",
        true,
      ],
      [
        "KL or clipping pressure can keep the tuned model from drifting too far from an old or reference model.",
        true,
      ],
    ],
    "The reinforcement-learning analogy remains useful in reasoning training: the model samples token actions under a policy and receives feedback on the resulting completion. The update is still constrained because reasoning rewards should improve the model without erasing useful behavior from the earlier policy.",
  ),
  makeQuestion(
    "cme295-lect6-q34",
    "hard",
    "Which implementation details can matter when estimating or comparing reasoning benchmark scores?",
    [
      ["The number of generated samples used before computing Pass@k.", true],
      [
        "The sampling temperature used to create diverse candidate solutions.",
        true,
      ],
      [
        "Whether the evaluator reports Pass@1, Pass@k, Cons@k, exact match, or accuracy.",
        true,
      ],
      [
        "Whether correctness is checked by tests, ground-truth final answers, or another verifier.",
        true,
      ],
    ],
    "Reasoning benchmark numbers are not a single universal property of a model. Sampling count, temperature, aggregation metric, and verifier all affect what the score means and whether it reflects single-shot reliability, multi-shot search, or consensus behavior.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect6-q35",
    "easy",
    "Assertion: A formatting reward can check whether required reasoning delimiters appear.\n\nReason: A formatting reward guarantees that the final answer is mathematically or programmatically correct.",
    1,
    "The assertion is true because a simple parser can check for tags such as reasoning and answer delimiters. The reason is false because tag presence says little about whether the solution passes tests or matches the ground truth.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect6-q36",
    "medium",
    "Assertion: Pass@k is appropriate when verification is cheap and multiple attempts are acceptable.\n\nReason: Pass@1 is the special case of Pass@k with \\(k=1\\).",
    5,
    "Both statements are true, but the reason does not explain the use case. Pass@k is appropriate in multi-attempt settings because at least one successful candidate can be found and verified, while the Pass@1 identity is only a mathematical special case.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect6-q37",
    "easy",
    "Assertion: A reasoning model's output can include both intermediate reasoning tokens and a final answer.\n\nReason: Generating intermediate reasoning tokens gives the model additional test-time forward passes before it commits to an answer.",
    5,
    "Both statements are true, but the reason does not explain why the output format can contain both reasoning tokens and a final answer. The reason explains why intermediate reasoning can be useful as extra test-time computation; the assertion is about the structure of the generated output, while the reason is about the compute intuition behind producing those intermediate tokens.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect6-q38",
    "hard",
    "Assertion: GRPO can avoid training a separate value model for advantage estimation.\n\nReason: GRPO estimates a completion's advantage by comparing its reward with rewards from other sampled completions for the same prompt.",
    4,
    "Both statements are true, and the group-relative reward comparison is the reason GRPO can skip the value-model route used in PPO-style advantage estimation. The method still needs sampled completions, rewards, and update constraints, but not a jointly trained value function.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect6-q39",
    "medium",
    "Assertion: R1-Zero uses a cold-start SFT stage with human-cleaned reasoning traces before its GRPO reasoning stage.\n\nReason: The full R1 pipeline adds cold-start SFT to improve readability, formatting, and language consistency before continuing with reasoning RL.",
    2,
    "The assertion is false because the R1-Zero proof of concept starts from the base model and applies reasoning RL without that cold-start SFT stage. The reason is true for the full R1 recipe, where supervised cleanup helps address the rough traces produced by pure RL.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect6-q40",
    "hard",
    "Assertion: Reasoning distillation requires the student model to rediscover reasoning only through GRPO with verifiable rewards.\n\nReason: In reasoning distillation, the teacher's generated reasoning traces are discarded so the student only sees final answers.",
    3,
    "Both statements are false. Reasoning distillation can use a strong teacher to generate full trace-plus-answer responses, then train a smaller student with SFT-style imitation rather than requiring the student to rediscover the behavior from RL alone.",
  ),
  makeQuestion(
    "cme295-lect6-q41",
    "hard",
    `A model generates \\(n=12\\) candidate solutions for a programming task, and \\(c=4\\) pass all tests. Using the standard without-replacement estimator,

\\[
\\mathrm{Pass@}k = 1 - \\frac{\\binom{n-c}{k}}{\\binom{n}{k}},
\\]

what is \\(\\mathrm{Pass@}3\\)?`,
    [
      [
        "\\(1 - \\frac{\\binom{8}{3}}{\\binom{12}{3}} = \\frac{41}{55}\\), about 0.745.",
        true,
      ],
      [
        "\\(1 - \\frac{\\binom{4}{3}}{\\binom{12}{3}} = \\frac{54}{55}\\), because the numerator should count successful triples.",
        false,
      ],
      [
        "\\(\\frac{4}{12} = \\frac{1}{3}\\), because Pass@3 is just the observed pass fraction.",
        false,
      ],
      [
        "\\(1 - (\\frac{8}{12})^3 = \\frac{19}{27}\\), because attempts are sampled independently with replacement.",
        false,
      ],
    ],
    "The estimator uses the complement: all three selected attempts fail. There are eight failed samples among twelve total, so the correct calculation is \\(1 - \\binom{8}{3}/\\binom{12}{3}\\); using successful triples or an independent-with-replacement approximation changes the event being estimated.",
  ),
  makeQuestion(
    "cme295-lect6-q42",
    "hard",
    "Assume, as a simplified model, that each sampled solution succeeds independently with probability \\(p=0.25\\). Which statements correctly reason about \\(\\mathrm{Pass@}4\\) under this independent-sampling assumption?",
    [
      [
        "\\(\\mathrm{Pass@}4 = 1 - (1 - 0.25)^4 = 1 - 0.75^4\\), about 0.684.",
        true,
      ],
      [
        "\\(\\mathrm{Pass@}1\\) remains 0.25, so the multi-attempt score is not the same as single-shot reliability.",
        true,
      ],
      [
        "\\(4 \\cdot 0.25 \\cdot 0.75^3\\) is \\(\\mathrm{Pass@}4\\) because exactly one success is required.",
        false,
      ],
      [
        "Doubling \\(k\\) from 4 to 8 must exactly double Pass@k because each attempt has the same success probability.",
        false,
      ],
    ],
    "For independent attempts, Pass@k is one minus the probability that every sampled attempt fails. The exactly-one-success formula omits cases with two or more successes, and the curve cannot double forever because it saturates toward 1.",
  ),
  makeQuestion(
    "cme295-lect6-q43",
    "hard",
    "Which statements correctly connect sampling temperature, diversity, and Pass@k?",
    [
      [
        "If temperature is so low that all samples are nearly identical, the effective benefit of increasing \\(k\\) can be close to the benefit of one attempt.",
        true,
      ],
      [
        "A moderate temperature can improve \\(\\mathrm{Pass@}k\\) when it adds useful diversity without destroying per-sample quality.",
        true,
      ],
      [
        "A very high temperature can reduce \\(p_{\\text{correct}}\\), which can lower Pass@k despite extra diversity.",
        true,
      ],
      [
        "Because Pass@k needs only one success, the per-sample success probability \\(p\\) can be ignored in \\(1-(1-p)^k\\).",
        false,
      ],
    ],
    "Pass@k is driven by both sample quality and useful diversity. Temperature is reported because it changes that tradeoff: identical high-quality samples do not explore much, while very noisy diverse samples may stop solving the problem.",
  ),
  makeQuestion(
    "cme295-lect6-q44",
    "hard",
    `A model samples nine final answers for a math problem. A verifier can check final answers, and the ground-truth answer is B.

| Final answer | Number of samples | Verifier result |
| --- | ---: | --- |
| A | 5 | fail |
| B | 3 | pass |
| C | 1 | fail |

Which statements are correct?`,
    [
      [
        "Cons@9 fails because majority vote returns A, which does not match the ground truth.",
        true,
      ],
      [
        "Pass@9 succeeds because at least one of the nine sampled attempts passes the verifier.",
        true,
      ],
      [
        "The example separates multi-attempt search success from consensus reliability.",
        true,
      ],
      [
        "A system allowed to run the verifier on each candidate could choose a passing B answer even though majority vote fails.",
        true,
      ],
    ],
    "The majority answer and the existence of at least one correct answer are different events. This is why Pass@k can look strong in settings with cheap verification even when a consensus-style answer would be wrong.",
  ),
  makeQuestion(
    "cme295-lect6-q45",
    "hard",
    `A GRPO-style update samples four completions for one prompt and obtains rewards \\([1, 1, 0, 0]\\). Using population standard deviation, the mean is \\(0.5\\) and the standard deviation is \\(0.5\\). If advantage is approximated by

\\[
A_i = \\frac{r_i - \\bar r}{\\sigma_r},
\\]

which statements are correct?`,
    [
      ["Each reward-1 completion has advantage \\((1 - 0.5)/0.5 = 1\\).", true],
      [
        "Each reward-0 completion has advantage \\((0 - 0.5)/0.5 = -1\\).",
        true,
      ],
      [
        "All four completions have \\(A_i > 0\\) because every completion came from the current policy.",
        false,
      ],
      [
        "The group-relative normalization makes \\(\\bar r\\) impossible to compute.",
        false,
      ],
    ],
    "GRPO's advantage estimate is relative to sibling completions for the same prompt. In this simple group, the passing completions are above the group mean and the failing completions are below it, giving symmetric positive and negative advantages.",
  ),
  makeQuestion(
    "cme295-lect6-q46",
    "hard",
    "A GRPO group has rewards \\([1,1,1,1]\\) for four completions of the same prompt. Which statements correctly diagnose this edge case for a group-relative advantage formula?",
    [
      [
        "Subtracting the group mean gives zero reward advantage for every completion.",
        true,
      ],
      [
        "A naive division by the within-group standard deviation needs special handling because the standard deviation is zero.",
        true,
      ],
      [
        "The group does not distinguish which passing completion is better unless another signal is added.",
        true,
      ],
      [
        "The group-relative calculation should strongly upweight the longest completion solely because it has more tokens.",
        false,
      ],
    ],
    "When every sampled completion receives the same reward, the group-relative signal has no ranking information by itself. Implementations need to handle zero or tiny variance carefully rather than inventing a preference for length.",
  ),
  makeQuestion(
    "cme295-lect6-q47",
    "hard",
    `A PPO-style clipped term uses \\(\\epsilon=0.2\\), positive advantage \\(A=3\\), and

\\[
\\min(rA, \\mathrm{clip}(r, 1-\\epsilon, 1+\\epsilon)A).
\\]

Which calculations are correct?`,
    [
      [
        "For \\(r=1.3\\), the term is \\(\\min(3.9, 1.2 \\cdot 3)=3.6\\).",
        true,
      ],
      [
        "For \\(r=0.7\\), the term is \\(\\min(2.1, 0.8 \\cdot 3)=2.1\\).",
        true,
      ],
      [
        "For \\(r=1.3\\), clipping still allows the full unclipped value \\(3.9\\) to be used.",
        false,
      ],
      [
        "For \\(r=0.7\\), clipping raises the objective term because the clipped value is larger.",
        false,
      ],
    ],
    "For positive advantage, the clipped objective prevents an overly large probability-ratio increase from receiving extra reward. If the ratio is too low, the minimum keeps the lower unclipped value, so clipping does not rescue an update that moved in the wrong direction.",
  ),
  makeQuestion(
    "cme295-lect6-q48",
    "hard",
    `A reference model and tuned policy assign probabilities to three possible next tokens:

| Token | Reference probability | Tuned probability |
| --- | ---: | ---: |
| A | 0.50 | 0.80 |
| B | 0.25 | 0.10 |
| C | 0.25 | 0.10 |

Which statements correctly interpret a KL/reference penalty in this situation?`,
    [
      [
        "The tuned policy has shifted probability mass toward token A relative to the reference model.",
        true,
      ],
      [
        "A KL penalty can be positive here because the tuned distribution differs from the reference distribution.",
        true,
      ],
      [
        "Reference pressure would penalize this kind of distributional drift if the training objective weights the KL term.",
        true,
      ],
      [
        "The KL term by itself does not say whether the final answer to the task is correct.",
        true,
      ],
    ],
    "KL/reference pressure controls how far the tuned policy moves from the reference distribution. It is a stability guardrail, not a correctness verifier, so it must be combined with reward signals that say whether the solution actually works.",
  ),
  makeQuestion(
    "cme295-lect6-q49",
    "hard",
    "Two incorrect completions have the same negative advantage, but one has length 20 tokens and the other has length 100 tokens. Under a per-output \\(1/|o_i|\\) normalization, which statements correctly reason about token-level contribution?",
    [
      [
        "A token in the 20-token completion receives five times the per-token weight of a token in the 100-token completion.",
        true,
      ],
      [
        "This can make a long bad completion cheaper per token than a short bad completion.",
        true,
      ],
      [
        "The total penalty across the long completion is necessarily larger because it has more tokens.",
        false,
      ],
      [
        "Removing the length factor guarantees that the shortest output will always be preferred.",
        false,
      ],
    ],
    "The pathology is about how credit or penalty is distributed across tokens, not a proof that shorter outputs are always better. Length normalization can dilute the per-token penalty for a long incorrect trace, which creates an incentive that is not the same as rewarding reasoning quality.",
  ),
  makeQuestion(
    "cme295-lect6-q50",
    "hard",
    "Which statements correctly describe the purpose of DAPO-style or Dr. GRPO-style changes to the GRPO length normalization issue?",
    [
      [
        "They target an optimization incentive that can make longer incorrect outputs less penalized per token.",
        true,
      ],
      [
        "Equalizing token-level contributions can reduce dependence on the individual completion's length.",
        true,
      ],
      [
        "These changes still need correctness rewards and update-control mechanisms such as clipping or reference pressure.",
        true,
      ],
      ["They make Pass@k and Cons@k mathematically identical metrics.", false],
    ],
    "The length-normalization fixes operate inside the training objective. They do not redefine benchmark metrics, and they do not remove the need for a reward signal or guardrails against destructive policy movement.",
  ),
  makeQuestion(
    "cme295-lect6-q51",
    "hard",
    "Suppose a clipped policy update constrains a new token probability by a ratio bound around the old probability. Which statements explain why asymmetric clipping bounds can matter?",
    [
      [
        "If an old token probability is \\(0.001\\), an upper ratio of 1.2 only raises it to \\(0.0012\\), a tiny absolute increase.",
        true,
      ],
      [
        "If an old token probability is \\(0.5\\), a loose downward ratio can create a large absolute decrease.",
        true,
      ],
      [
        "Separate upper \\(u\\) and lower \\(\\ell\\) clipping choices can let rare useful tokens grow while still limiting collapse of already likely tokens.",
        true,
      ],
      [
        "Changing clipping bounds changes which final answers receive verifier label \\(y=1\\).",
        false,
      ],
    ],
    "Ratio bounds have different absolute effects depending on the old probability. The motivation for asymmetric bounds is optimization control, not changing the task labels or verifier outcomes.",
  ),
  makeQuestion(
    "cme295-lect6-q52",
    "hard",
    "A later supervised fine-tuning stage mixes about 200k general instruction pairs with about 600k reasoning-focused pairs. Which quantitative and conceptual interpretations are correct?",
    [
      ["The reasoning-to-general data ratio is about 3:1.", true],
      [
        "The reasoning-focused share is about 75% of the 800k-pair mixture.",
        true,
      ],
      ["The general instruction share is about 25% of the mixture.", true],
      [
        "Keeping general data in the mix helps preserve broad assistant behavior instead of training only math, code, and logic.",
        true,
      ],
    ],
    "The arithmetic is 600k reasoning pairs and 200k general pairs out of 800k total, giving a 3:1 ratio and a 75% reasoning share. The conceptual reason for the mixture is that a final assistant must still handle general requests, not only benchmark-style reasoning prompts.",
  ),
  makeQuestion(
    "cme295-lect6-q53",
    "hard",
    "A rejection-sampling pipeline takes 100k reasoning prompts, generates 8 candidate responses per prompt, and keeps 1 response per prompt after rule checks and judge filtering. Which statements are correct?",
    [
      [
        "The pipeline generates \\(100{,}000 \\times 8 = 800{,}000\\) candidate responses before filtering.",
        true,
      ],
      [
        "The retained supervised fine-tuning set has \\(100{,}000 \\times 1 = 100{,}000\\) responses.",
        true,
      ],
      ["The response-level retention rate is \\(1/8 = 12.5\\%\\).", true],
      [
        "The filtering step replaces accepting \\(8/8\\) sampled traces with keeping \\(1/8\\) after quality checks.",
        true,
      ],
    ],
    "Rejection sampling spends generation compute to create a cleaner supervised dataset. The retained set is smaller than the generated candidate pool, and the value comes from filtering for correctness, format, or judge-assessed quality.",
  ),
  makeQuestion(
    "cme295-lect6-q54",
    "hard",
    "A large reasoning teacher costs 100 compute units per query. A distilled student costs 5 units per query, but creating the distilled training set costs 1,000,000 units up front. Ignoring quality differences, which break-even statements are correct?",
    [
      [
        "Each student-served query saves \\(100 - 5 = 95\\) units compared with the teacher.",
        true,
      ],
      [
        "The up-front cost is recovered after \\(\\lceil 1{,}000{,}000 / 95 \\rceil = 10{,}527\\) student-served queries.",
        true,
      ],
      [
        "For \\(N\\) far above \\(10{,}527\\) queries, the offline distillation cost is amortized over many cheaper inferences.",
        true,
      ],
      [
        "This calculation is only a cost model; it leaves quality differences such as \\(Q_{\\text{student}} < Q_{\\text{teacher}}\\) outside the arithmetic.",
        true,
      ],
    ],
    "Distillation can be attractive because it converts repeated expensive teacher inference into a one-time data-generation and training cost plus cheaper student inference. The arithmetic does not replace quality evaluation, but it explains why offline distillation can be a good use of compute at scale.",
  ),
  makeQuestion(
    "cme295-lect6-q55",
    "hard",
    `A toy verifiable reward is

\\[
R = 0.2F + 0.8C,
\\]

where \\(F=1\\) means the required reasoning/answer format is present and \\(C=1\\) means the final answer is correct. Which statements are correct?`,
    [
      ["A response with \\(F=1, C=1\\) receives reward 1.0.", true],
      [
        "A response with \\(F=0, C=1\\) outranks a response with \\(F=1, C=0\\).",
        true,
      ],
      [
        "A formatted wrong answer \\((F=1, C=0)\\) outranks an unformatted correct answer \\((F=0, C=1)\\) because format is the only verifiable part.",
        false,
      ],
      [
        "A response with \\(F=0, C=0\\) receives the same reward as a formatted wrong answer.",
        false,
      ],
    ],
    "The arithmetic makes correctness the dominant reward component: an unformatted correct answer scores 0.8, while a formatted wrong answer scores 0.2. This illustrates why formatting rewards are useful but should not be allowed to dominate outcome correctness.",
  ),
  makeQuestion(
    "cme295-lect6-q56",
    "hard",
    "A reasoning model has an 8192-token context window. A prompt uses 1800 tokens, the service reserves 600 tokens for the final answer, and it keeps a 200-token safety margin. Which budget statements are correct?",
    [
      [
        "The maximum reasoning budget under these assumptions is \\(8192 - 1800 - 600 - 200 = 5592\\) tokens.",
        true,
      ],
      [
        "Increasing the reasoning budget to \\(5593\\) tokens would consume space reserved for the final answer or safety margin.",
        true,
      ],
      [
        "Dynamic budgeting should usually assign \\(b_e < b_h\\) when latency and context are scarce.",
        true,
      ],
      [
        "Context awareness can be ignored because reasoning tokens occupy \\(0\\) positions in the \\(8192\\)-token context window.",
        false,
      ],
    ],
    "Reasoning tokens are part of the inference budget, so they compete with prompt length, answer length, and safety margins. This is why test-time scaling needs controls rather than simply telling every prompt to think as long as possible.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect6-q57",
    "hard",
    "Assertion: In the without-replacement Pass@k estimator, if \\(k > n-c\\), the estimated Pass@k is 1.\n\nReason: This condition means every individual generated attempt is correct.",
    1,
    "The assertion is true because there are fewer than k failed samples, so it is impossible to choose k all-failed samples and the complement term is zero. The reason is false because there may still be failed attempts; there are just not enough failed attempts to fill all k selected slots.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect6-q58",
    "hard",
    "Assertion: Cons@k must succeed whenever Pass@k succeeds.\n\nReason: Pass@k only requires at least one successful sample, while Cons@k uses the most common answer among the samples.",
    2,
    "The assertion is false because one correct sample can coexist with a wrong majority answer. The reason is true and states the distinction: Pass@k is an existence-of-success metric, while Cons@k is an aggregation-by-consensus metric.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect6-q59",
    "hard",
    "Assertion: GRPO's group-relative advantage removes the need for rewards.\n\nReason: The group baseline in GRPO is computed from old-policy logits rather than from rewards assigned to sampled completions.",
    3,
    "Both statements are false. GRPO still needs rewards for sampled completions, and its group-relative baseline is built from the rewards of sibling completions for the same prompt, not from old-policy logits alone.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect6-q60",
    "hard",
    "Assertion: Reasoning distillation can train a smaller model with teacher-generated trace-plus-answer sequences using supervised fine-tuning.\n\nReason: Distillation shifts some compute cost offline and can be more efficient for smaller models than making them rediscover reasoning behavior from scratch.",
    5,
    "Both statements are true, but the reason is a motivation rather than the mechanism that defines the assertion. The mechanism is sequence imitation on teacher-generated reasoning traces, while the efficiency argument explains why this training route can be attractive.",
  ),
];
