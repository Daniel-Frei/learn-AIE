import { Question } from "../../quiz";

type Lecture5Difficulty = "easy" | "medium" | "hard";
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
  difficulty: Lecture5Difficulty,
  prompt: string,
  optionSeeds: readonly OptionSeed[],
  explanation: string,
): Question {
  if (optionSeeds.length !== 4) {
    throw new Error(`CME295 Lecture 5 question ${id} needs 4 options.`);
  }

  return {
    id,
    chapter: 5,
    difficulty,
    prompt,
    options: optionSeeds.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

function makeAssertionReasonQuestion(
  id: string,
  difficulty: Lecture5Difficulty,
  prompt: string,
  correctChoice: AssertionReasonChoice,
  explanation: string,
): Question {
  return {
    id,
    chapter: 5,
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

export const stanfordCME295Lecture5PreferenceTuningQuestions: Question[] = [
  makeQuestion(
    "cme295-lect5-q01",
    "easy",
    "A team has a pretrained decoder-only model and wants to turn it into a helpful assistant. Which lifecycle statements correctly distinguish pretraining, supervised fine-tuning, and preference tuning?",
    [
      [
        "Pretraining teaches broad next-token prediction from large language and code corpora.",
        true,
      ],
      [
        "Supervised fine-tuning (SFT) teaches task or assistant behavior from high-quality target responses.",
        true,
      ],
      [
        "Preference tuning shifts the SFT model toward outputs that humans or a chosen metric prefer.",
        true,
      ],
      [
        "Preference tuning is the first stage because the model must learn basic language only after it has learned human preferences.",
        false,
      ],
    ],
    "The staged view is pretraining for broad language and code structure, supervised fine-tuning for the desired task behavior, and preference tuning for preference-aligned behavior. Reversing that order would leave the preference stage without a capable base model to align.",
  ),
  makeQuestion(
    "cme295-lect5-q02",
    "easy",
    "Why can preference tuning be useful even after a model has gone through supervised fine-tuning?",
    [
      [
        "It can inject negative signal about responses the model should avoid.",
        true,
      ],
      [
        "It can be easier for annotators to compare two responses than to write an ideal response from scratch.",
        true,
      ],
      [
        "It can target qualities such as tone, helpfulness, or safety without adding many fragile SFT target examples.",
        true,
      ],
      [
        "It guarantees that flawed SFT data never needs to be inspected or repaired.",
        false,
      ],
    ],
    "Supervised fine-tuning mainly increases the likelihood of desired target responses, while preference tuning can explicitly compare preferred and rejected behavior. A badly behaving SFT model can still indicate that the SFT data itself needs review, so preference tuning is not a substitute for data quality.",
  ),
  makeQuestion(
    "cme295-lect5-q03",
    "easy",
    "An annotator sees two candidate responses to the same prompt and chooses which response is better. Which preference-data format is being collected?",
    [
      ["Pairwise preference data.", true],
      ["Pointwise scalar scoring of a single observation.", false],
      ["Listwise ranking of several observations at once.", false],
      ["A next-token supervised fine-tuning target.", false],
    ],
    "Pairwise data compares two prompt-response observations and records which one is preferred. Pointwise scoring assigns a score to one observation, listwise ranking orders a larger set, and ordinary supervised fine-tuning supplies a target response rather than a preference comparison.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect5-q04",
    "easy",
    "Assertion: Pairwise preference labels are often easier for human raters than pointwise numerical scores.\n\nReason: Pointwise scoring asks raters to place each response on an absolute scale whose calibration can be hard to define consistently.",
    4,
    "The assertion is true because comparing two responses is usually more natural than assigning a stable standalone number. The reason is also true and explains the assertion: an unclear absolute scale makes pointwise labels noisy, while pairwise labels can avoid some of that calibration burden.",
  ),
  makeQuestion(
    "cme295-lect5-q05",
    "easy",
    "Which steps can be part of constructing pairwise preference data for an assistant model?",
    [
      [
        "Use prompts from logs or another reference distribution that represents intended use.",
        true,
      ],
      [
        "Sample two completions from the same SFT model with a positive temperature to get variation.",
        true,
      ],
      [
        "Rewrite a bad logged response into a better response and treat the two as a preference pair.",
        true,
      ],
      [
        "Label the pair with either a binary better/worse decision or a more nuanced preference scale.",
        true,
      ],
    ],
    "Preference data needs a prompt, candidate responses, and a preference signal. The candidates can come from sampling, synthetic generation, or rewrites, and the label can be binary or more graded depending on the task design.",
  ),
  makeQuestion(
    "cme295-lect5-q06",
    "medium",
    "Which statements correctly separate human-feedback preference labels from proxy preference labels?",
    [
      [
        "Human ratings are the feedback source that makes the method reinforcement learning from human feedback (RLHF).",
        true,
      ],
      [
        "Labels from an LLM-as-a-judge or a rule-based metric are proxy labels rather than direct human feedback.",
        true,
      ],
      [
        "A BLEU or ROUGE proxy automatically measures friendliness and safety better than human raters can.",
        false,
      ],
      [
        "Using proxy labels by itself makes a training run on-policy reinforcement learning.",
        false,
      ],
    ],
    "The human-feedback part of RLHF refers to the labels used to train the reward model. Automated judges and metrics can be useful proxies, but they do not by themselves make the feedback human or turn a static preference dataset into on-policy reinforcement learning.",
  ),
  makeQuestion(
    "cme295-lect5-q07",
    "easy",
    "In the reinforcement-learning view of language-model generation, which mappings are correct?",
    [
      ["The LLM is the agent being trained or evaluated.", true],
      ["The state is the input context generated or supplied so far.", true],
      ["The action is choosing the next token from the vocabulary.", true],
      [
        "The policy is the model's probability distribution over next tokens.",
        true,
      ],
    ],
    "The reinforcement-learning framing maps naturally onto autoregressive generation: the model sees a context, chooses a next token, and does so according to its policy distribution. Preference-tuning methods then add a reward signal for the generated completion or behavior.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect5-q08",
    "medium",
    "Assertion: RLHF assigns a separate human preference label to every generated token in a completion.\n\nReason: The reward model scores only the prompt and the first generated token.",
    3,
    "The assertion is false because preference labels and reward-model scores are much sparser than token-level supervised labels; they are usually tied to a full response or prompt-response pair. The reason is also false because the reward model evaluates the prompt together with the completion, not only the first generated token.",
  ),
  makeQuestion(
    "cme295-lect5-q09",
    "easy",
    "Which operation belongs to the reward-modeling stage of RLHF rather than the later policy-optimization stage?",
    [
      [
        "Train a model to score prompt-response pairs using chosen-versus-rejected examples.",
        true,
      ],
      [
        "Update the assistant policy by sampling completions from the current policy.",
        false,
      ],
      [
        "Penalize the tuned policy for moving too far from a frozen reference policy.",
        false,
      ],
      [
        "Estimate token-level advantages for a Proximal Policy Optimization update.",
        false,
      ],
    ],
    "The first RLHF stage learns a reward model that can score prompt-response pairs. The later reinforcement-learning stage uses that reward model to update the assistant policy, often with constraints or advantage estimates that do not belong to reward-model fitting itself.",
  ),
  makeQuestion(
    "cme295-lect5-q10",
    "medium",
    "A reward model is trained on pairs \\((x, y_w, y_l)\\), where \\(y_w\\) is preferred over \\(y_l\\). Which statements describe what the Bradley-Terry training setup is trying to learn?",
    [
      [
        "The reward for \\((x, y_w)\\) should be higher than the reward for \\((x, y_l)\\).",
        true,
      ],
      [
        "The trained model can score one prompt-response pair at a time even though its loss used pairs.",
        true,
      ],
      [
        "The labels must provide exact absolute utility values for each response.",
        false,
      ],
      [
        "The losing response must come from a different model architecture than the winning response.",
        false,
      ],
    ],
    "Bradley-Terry training turns pairwise preferences into pressure on score differences. The resulting reward model is pointwise at inference time: it takes one prompt-response pair and emits a scalar score, even though the loss compared winners and losers during training.",
  ),
  makeQuestion(
    "cme295-lect5-q11",
    "hard",
    "For two responses with reward scores \\(r_i\\) and \\(r_j\\), a Bradley-Terry model can write \\(P(y_i \\succ y_j)=\\sigma(r_i-r_j)\\). Which statements follow from this formulation?",
    [
      [
        "Increasing \\(r_i-r_j\\) increases the modeled probability that \\(y_i\\) is preferred.",
        true,
      ],
      [
        "A loss term such as \\(-\\log \\sigma(r_w-r_l)\\) encourages the winning response score to exceed the losing response score.",
        true,
      ],
      [
        "If \\(r_i=r_j\\), the modeled preference probability for \\(y_i\\) over \\(y_j\\) is exactly 1.",
        false,
      ],
      [
        "Adding the same constant to both rewards leaves the modeled preference probability unchanged.",
        true,
      ],
    ],
    "The sigmoid depends on the reward difference, so larger winner-minus-loser gaps push the preferred probability toward one. Equal scores give \\(\\sigma(0)=0.5\\), and shifting both scores by the same constant cancels in the difference.",
  ),
  makeQuestion(
    "cme295-lect5-q12",
    "medium",
    "A team wants preference labels for an assistant but finds that raters disagree. Which statements correctly address reward dimensions and label quality?",
    [
      [
        "The team should specify the dimension being rated, such as helpfulness, safety, or tone.",
        true,
      ],
      [
        "Different preference dimensions may justify different reward models or different labeling guidelines.",
        true,
      ],
      [
        "Clear rater guidelines can reduce noise even when the underlying task has subjective elements.",
        true,
      ],
      [
        "A single scalar score automatically resolves every conflict among usefulness, safety, and style.",
        false,
      ],
    ],
    "Preference data is only as meaningful as the rating task it defines. Helpfulness, safety, and tone can conflict, so guidelines and reward dimensions need to be explicit rather than hidden inside an ambiguous scalar label.",
  ),
  makeQuestion(
    "cme295-lect5-q13",
    "medium",
    "A decoder-only language model is adapted into a reward model for prompt-response scoring. Which design change directly turns it from next-token prediction toward scalar reward prediction?",
    [
      [
        "Attach a classification or regression-style head that produces a score for the prompt-response input.",
        true,
      ],
      [
        "Remove the prompt so the model only scores the response text in isolation.",
        false,
      ],
      [
        "Force the model to emit the word `good` or `bad` as its next token and use no separate scoring head.",
        false,
      ],
      [
        "Replace the pairwise preference objective with ordinary masked-language modeling.",
        false,
      ],
    ],
    "A reward model needs a scalar score for a prompt-response pair, so decoder-only models are commonly adapted with a scoring head rather than used only as next-token predictors. Encoder-only models can also be adapted through a pooled representation, and benchmarks such as RewardBench evaluate reward-model quality.",
  ),
  makeQuestion(
    "cme295-lect5-q14",
    "easy",
    "In the reinforcement-learning stage of RLHF, which statements describe the training setup?",
    [
      ["The policy is commonly initialized from the SFT model.", true],
      [
        "The reward model is frozen and supplies scores for generated completions.",
        true,
      ],
      [
        "A frozen reference or base model can be used to discourage excessive drift.",
        true,
      ],
      [
        "The reward model and policy are trained from scratch together on every update.",
        false,
      ],
    ],
    "The policy starts from the supervised fine-tuned model and is updated to receive better reward-model scores. The reward model has already been trained, and a reference model helps keep the tuned policy near the useful behavior learned before preference tuning.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect5-q15",
    "medium",
    "Assertion: A policy optimized only for higher reward-model scores can produce behavior that does not satisfy the real human objective.\n\nReason: A reward model is an imperfect proxy for the desired behavior and can be exploited by the policy.",
    4,
    "The assertion is true because optimizing a proxy too aggressively can improve the proxy score while damaging the actual objective. The reason is true and explains the assertion: reward hacking occurs when the policy finds ways to score well under an imperfect reward model without producing the intended behavior.",
  ),
  makeQuestion(
    "cme295-lect5-q16",
    "medium",
    "Why do RLHF-style policy updates usually try not to move too far from the SFT reference model?",
    [
      [
        "The SFT model already contains useful language, task, and instruction-following behavior.",
        true,
      ],
      [
        "Large moves can exploit errors in the reward model instead of improving the real objective.",
        true,
      ],
      ["Constraining policy drift can reduce training instability.", true],
      [
        "A reference-model penalty can express this preference for staying close to the base policy.",
        true,
      ],
    ],
    "The base model is not a blank slate; it carries valuable knowledge and behavior. Preference tuning is meant to adjust that behavior, not erase it or chase an imperfect reward signal so aggressively that the policy becomes unstable or reward-hacked.",
  ),
  makeQuestion(
    "cme295-lect5-q17",
    "easy",
    "Which statements correctly characterize KL divergence when it is used to compare policy distributions?",
    [
      [
        "It measures how far one probability distribution is from another in an information-theoretic sense.",
        true,
      ],
      ["It is nonnegative.", true],
      ["It equals zero when the two distributions are the same.", true],
      [
        "It is always symmetric, so \\(D_{KL}(P\\|Q)=D_{KL}(Q\\|P)\\) for any two distributions.",
        false,
      ],
    ],
    "KL divergence is a nonnegative divergence measure that becomes zero when the two distributions match. It is not a metric distance because it is not generally symmetric, which matters when choosing which policy is treated as the reference.",
  ),
  makeQuestion(
    "cme295-lect5-q18",
    "medium",
    "Which statements describe the advantage and value-function pieces used in PPO-style RLHF?",
    [
      [
        "An advantage can be understood as reward relative to a baseline expectation.",
        true,
      ],
      [
        "The value function estimates, at a partial generation state, what reward is expected if generation continues under the policy.",
        true,
      ],
      [
        "The value function is the same frozen model as the reward model.",
        false,
      ],
      [
        "Using an advantage eliminates the need to score completions with any reward signal.",
        false,
      ],
    ],
    "Advantages reduce variance by asking whether an outcome is better or worse than expected, not merely whether its raw reward is high. The value function is trained with the policy to estimate expected return from partial generations, while the reward model remains the source of completion-level reward information.",
  ),
  makeQuestion(
    "cme295-lect5-q19",
    "hard",
    "In PPO-Clip, the ratio \\(r_\\theta\\) compares the current policy probability with the previous policy probability for an action. Which update intuitions are correct?",
    [
      [
        "With a positive advantage, the objective encourages increasing the probability of that action but caps the benefit after the ratio becomes too large.",
        true,
      ],
      [
        "With a negative advantage, the objective encourages lowering the probability of that action but prevents an excessively large one-step change.",
        true,
      ],
      [
        "The ratio \\(r_\\theta\\) is the scalar score emitted by the reward model.",
        false,
      ],
      [
        "The `old` policy in this ratio is always the original SFT reference model.",
        false,
      ],
    ],
    "PPO-Clip controls step size by clipping how much the current policy can differ from the previous RL iteration for the sampled action. The ratio is about policy probabilities, not reward-model scores, and `old` refers to the previous policy snapshot rather than necessarily the fixed reference model.",
  ),
  makeQuestion(
    "cme295-lect5-q20",
    "hard",
    "Which statements describe PPO variants that use a KL penalty?",
    [
      [
        "A KL term can penalize divergence between the trained policy and a comparison policy.",
        true,
      ],
      [
        "A coefficient such as \\(\\beta\\) controls how strongly the divergence penalty trades off against reward or advantage.",
        true,
      ],
      [
        "Adding a KL term means the reward model no longer needs to supply any preference signal.",
        false,
      ],
      [
        "The KL penalty is the same quantity as the reward score assigned to a completion.",
        false,
      ],
    ],
    "The KL penalty is a proximity control, not a replacement for the reward signal. In modern LLM use, the comparison policy is often the frozen reference or base model, while clipping may still be used to limit changes between nearby training iterations.",
  ),
  makeQuestion(
    "cme295-lect5-q21",
    "medium",
    "Which practical challenges are associated with PPO-style RLHF?",
    [
      [
        "It may need several model components, such as policy, value function, reward model, and reference model.",
        true,
      ],
      [
        "It introduces multiple hyperparameters, including KL penalties, clipping ranges, and advantage-estimation choices.",
        true,
      ],
      [
        "It can be unstable, and average reward is an imperfect training-monitoring signal.",
        true,
      ],
      [
        "It needs enough diversity in sampled completions to learn which behaviors to reinforce or discourage.",
        true,
      ],
    ],
    "PPO-style RLHF is powerful but operationally heavy. The algorithm combines multiple models, sensitive hyperparameters, noisy reward feedback, and on-policy sampling, so getting useful behavior is not just a matter of adding one simple loss term.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect5-q22",
    "medium",
    "Assertion: On-policy RLHF can be run without model exploration because policy updates only need static preferred completions.\n\nReason: On-policy training updates the model using completions sampled from the current policy.",
    2,
    "The assertion is false because on-policy training depends on what the current policy actually samples, so exploration and completion diversity matter. The reason is true: the defining contrast with ordinary supervised fine-tuning is that the policy generates the data used for its own updates.",
  ),
  makeQuestion(
    "cme295-lect5-q23",
    "easy",
    "How does Best-of-N use a trained reward model at inference time?",
    [
      [
        "Generate several candidate completions for the same prompt and return the one with the highest reward-model score.",
        true,
      ],
      [
        "Keep the SFT model weights fixed while the reward model ranks the candidates.",
        true,
      ],
      [
        "Backpropagate through the selected completion before returning it to the user.",
        false,
      ],
      [
        "Avoid using a reward model by choosing a completion uniformly at random.",
        false,
      ],
    ],
    "Best-of-N skips policy training and uses the reward model as a reranker. The base generator can stay fixed, but inference becomes more expensive because several completions must be generated and scored before one is returned.",
  ),
  makeQuestion(
    "cme295-lect5-q24",
    "medium",
    "Which statements identify limitations of Best-of-N reranking?",
    [
      [
        "Generating and scoring multiple completions can raise inference cost.",
        true,
      ],
      [
        "Even if completions are generated in parallel, response latency is tied to the slowest sampled completion.",
        true,
      ],
      [
        "If every sampled completion is bad, reranking only selects the least bad candidate from that set.",
        true,
      ],
      [
        "A monotonic rescaling of reward scores preserves the ranking, but it does not remove the generation cost.",
        true,
      ],
    ],
    "Best-of-N can improve selected outputs without RL training, but it moves work to inference. It still depends on candidate diversity and reward-model quality, and it cannot choose a good answer that was never generated.",
  ),
  makeQuestion(
    "cme295-lect5-q25",
    "easy",
    "Which statements describe Direct Preference Optimization (DPO) at a high level?",
    [
      [
        "It optimizes a supervised preference loss directly on chosen-versus-rejected examples.",
        true,
      ],
      [
        "It avoids training a separate reward model for the preference-tuning step.",
        true,
      ],
      [
        "It uses a frozen reference policy in the loss while updating the trainable policy.",
        true,
      ],
      [
        "It requires online rollouts and a learned value function in the same way PPO does.",
        false,
      ],
    ],
    "DPO recasts preference tuning as a supervised objective over preference pairs. It still compares the optimized policy to a reference policy, but it avoids the separate reward-model and value-function machinery used in PPO-style RLHF.",
  ),
  makeQuestion(
    "cme295-lect5-q26",
    "hard",
    "Which derivation steps explain why the DPO objective is connected to the Bradley-Terry preference model?",
    [
      [
        "Start from a KL-constrained reward-maximization objective for a policy.",
        true,
      ],
      [
        "Identify an implicit reward term that can be written using the optimized policy and reference policy.",
        true,
      ],
      [
        "Discard the reference policy because pairwise preference data alone fully fixes the reward scale.",
        false,
      ],
      [
        "Learn \\(\\beta\\) as the output of the reward model for each prompt-response pair.",
        false,
      ],
    ],
    "DPO comes from manipulating the KL-constrained policy objective until a reward-like term can be written in terms of policy probabilities. That implicit reward is then plugged into a Bradley-Terry-style pairwise likelihood, with \\(\\beta\\) acting as a tunable strength parameter rather than a per-example reward output.",
  ),
  makeQuestion(
    "cme295-lect5-q27",
    "medium",
    "Which statements support choosing Direct Preference Optimization for a quick preference-tuning implementation over PPO-style RLHF?",
    [
      [
        "It uses a supervised preference objective rather than a multi-stage RL loop.",
        true,
      ],
      [
        "It does not require separate reward-model and value-model components in the policy update.",
        true,
      ],
      [
        "It is guaranteed to outperform PPO on every task and implementation.",
        false,
      ],
      ["It removes the need for a frozen reference policy in the loss.", false],
    ],
    "DPO is attractive because it is simpler to implement and run than PPO-style RLHF. That simplicity is not an across-the-board performance guarantee, and the DPO loss still uses a reference policy to define how the tuned model should move relative to its base behavior.",
  ),
  makeQuestion(
    "cme295-lect5-q28",
    "easy",
    "An SFT assistant answers, `No, it might get damaged. Try hand washing instead.` A preference-tuned version answers, `It's better not to. Your teddy could get hurt! A gentle hand wash is safer.` What is the main behavior being adjusted?",
    [
      [
        "The assistant keeps the factual recommendation while shifting tone toward a gentler preferred style.",
        true,
      ],
      [
        "The assistant learns a new physical fact that hand washing is possible only during preference tuning.",
        false,
      ],
      [
        "The assistant changes from a text generator into a retrieval-only search system.",
        false,
      ],
      [
        "The assistant is encouraged to ignore the user's emotional attachment to the object.",
        false,
      ],
    ],
    "Preference tuning is often about aligning style, safety, and helpfulness around knowledge the model already has. The hand-washing answer preserves the core factual advice but changes the phrasing to better match the desired assistant behavior.",
  ),
  makeQuestion(
    "cme295-lect5-q29",
    "hard",
    "Which statements correctly distinguish preference tuning from simply adding more SFT examples?",
    [
      [
        "A preference pair can say which response should be preferred and which response should be discouraged for the same prompt.",
        true,
      ],
      [
        "Ordinary SFT mainly increases likelihood of target completions unless rejected behavior is rewritten into explicit targets.",
        true,
      ],
      [
        "Preference tuning should be used to hide known defects in an SFT dataset instead of fixing the dataset.",
        false,
      ],
      [
        "Pairwise preference labels require annotators to write an ideal completion token by token.",
        false,
      ],
    ],
    "Preference data carries a contrast between a better and worse response, which supplies information ordinary target-only SFT does not directly express. But if the SFT data distribution is flawed, repairing that data can still be the right intervention instead of relying on a later preference stage.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect5-q30",
    "medium",
    "Assertion: Best-of-N can improve the selected response without changing the generator's weights.\n\nReason: Best-of-N updates the SFT policy by backpropagating the reward score through the selected completion before returning it.",
    1,
    "The assertion is true because Best-of-N can leave the generator fixed and use the reward model only for scoring and selection. The reason is false because backpropagation through the generator is not part of the Best-of-N inference procedure; that method reranks candidates rather than tuning weights.",
  ),
  makeQuestion(
    "cme295-lect5-q31",
    "hard",
    "How should reward-model score scale be interpreted across RLHF and Best-of-N use cases?",
    [
      [
        "For Best-of-N, a monotonic transformation preserves which completion ranks highest.",
        true,
      ],
      [
        "For RL optimization, reward scaling can matter because the score enters an objective or advantage calculation.",
        true,
      ],
      [
        "A raw reward-model score is a learned scalar proxy, not a universally calibrated measure of all human values.",
        true,
      ],
      [
        "Reward scale is irrelevant in PPO because the reward never affects the policy update.",
        false,
      ],
    ],
    "Best-of-N only needs the relative ranking among candidates, so order-preserving transformations keep the same winner. In PPO-style optimization, the numeric scale interacts with rewards, advantages, and penalties, so normalization and scaling choices can affect training behavior.",
  ),
  makeQuestion(
    "cme295-lect5-q32",
    "hard",
    "In a PPO-Clip update, suppose the previous policy assigned probability 0.10 to a sampled token and the current policy assigns 0.13, with \\(\\epsilon=0.2\\). Which statements are correct?",
    [
      ["The policy ratio for that token is \\(0.13/0.10=1.3\\).", true],
      [
        "For a positive advantage, the clipped term would cap the useful ratio at \\(1+\\epsilon=1.2\\).",
        true,
      ],
      [
        "The clipping comparison is against the previous policy snapshot, not directly against the reward-model score.",
        true,
      ],
      [
        "Because 1.3 is greater than 1, the update must be accepted with no clipping.",
        false,
      ],
    ],
    "The ratio is current probability divided by previous probability, so the example gives 1.3. With \\(\\epsilon=0.2\\), a positive-advantage update receives no extra objective benefit above 1.2, which limits one-step policy movement.",
  ),
  makeQuestion(
    "cme295-lect5-q33",
    "hard",
    "Which issues are especially relevant when DPO is trained on preference data that was not generated by the current model?",
    [
      [
        "The preference pairs may come from a distribution that differs from the completions the current model would produce.",
        true,
      ],
      [
        "Additional SFT or data-generation choices may be used to reduce mismatch with the intended policy distribution.",
        true,
      ],
      [
        "A separate reward model must be trained before DPO can compute its loss.",
        false,
      ],
      [
        "DPO removes all sensitivity to how preference data was collected.",
        false,
      ],
    ],
    "DPO's supervised convenience can come with distribution-shift concerns when the preference pairs do not match the model's own likely completions. It avoids a separate reward model, but it does not make the preference-data source irrelevant.",
  ),
  makeQuestion(
    "cme295-lect5-q34",
    "easy",
    "Which statements correctly distinguish RLHF from reinforcement learning from AI feedback (RLAIF)?",
    [
      ["RLHF relies on human-generated preference labels.", true],
      [
        "RLAIF can use AI-generated judgments or other nonhuman feedback as the preference source.",
        true,
      ],
      [
        "The `HF` in RLHF refers to the hidden-state format of the transformer.",
        false,
      ],
      [
        "RLAIF means the policy is always trained without any reward or preference signal.",
        false,
      ],
    ],
    "The distinction is about where the feedback labels come from. The surrounding optimization machinery can be similar, but human ratings make the method human-feedback based, while AI judgments or other proxies move it into nonhuman feedback territory.",
  ),
  makeQuestion(
    "cme295-lect5-q35",
    "hard",
    "A reward model assigns score 2.0 to a preferred response and 0.5 to a rejected response for the same prompt. Under \\(P(y_w \\succ y_l)=\\sigma(r_w-r_l)\\), which interpretation is correct?",
    [
      [
        "The modeled probability is above 0.5 because the preferred response has the higher score.",
        true,
      ],
      [
        "The modeled probability is exactly 0.5 because Bradley-Terry ignores score differences.",
        false,
      ],
      [
        "The modeled probability is below 0.5 because lower reward scores are treated as better.",
        false,
      ],
      [
        "The modeled probability is exactly 1 because the preference label names a winner.",
        false,
      ],
    ],
    "The score gap is \\(2.0-0.5=1.5\\), and \\(\\sigma(1.5)\\) is greater than 0.5 but still less than 1. Bradley-Terry models preferences probabilistically, so even a labeled winner does not make the model assign certainty unless the score gap tends to infinity.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect5-q36",
    "hard",
    "Assertion: DPO usually needs fewer distinct model components in memory than PPO-style RLHF.\n\nReason: DPO can still use a frozen reference model to keep the optimized policy tied to the base model.",
    5,
    "The assertion is true because DPO does not need separate reward-model and value-function components for the policy update. The reason is also true, but it does not explain the reduced component count; the reduction comes from avoiding reward and value models, while the frozen reference model is one component DPO still keeps.",
  ),
  makeQuestion(
    "cme295-lect5-q37",
    "hard",
    "A team is choosing between PPO-style RLHF and DPO for a production alignment pass. Which tradeoff statements are correct?",
    [
      [
        "PPO-style RLHF can be more complex because reward modeling, value estimation, policy updates, and reference constraints interact.",
        true,
      ],
      [
        "DPO can be easier to implement because it turns preference tuning into a supervised loss over preference pairs.",
        true,
      ],
      [
        "There is no universal result that one method is best for every task, model, and implementation.",
        true,
      ],
      [
        "The choice depends on compute budget, tuning expertise, desired performance, and tolerance for training complexity.",
        true,
      ],
    ],
    "PPO-style RLHF and DPO trade implementation complexity, compute, and possible performance differently. DPO is often the simpler route, while PPO-style methods may be worth the operational cost when a team can tune the RL loop and needs the last bit of performance.",
  ),
  makeQuestion(
    "cme295-lect5-q38",
    "medium",
    "A lecturer wants to maximize informativeness but uses clap volume as the only reward. The lecturer discovers that telling jokes raises clap volume more than teaching clearly. Which concept does this illustrate?",
    [
      [
        "Reward hacking caused by optimizing an imperfect proxy for the real objective.",
        true,
      ],
      [
        "Generalized advantage estimation caused by subtracting a token-level baseline.",
        false,
      ],
      [
        "Pairwise annotation noise caused by asking raters to rank several lectures at once.",
        false,
      ],
      ["KL divergence being negative when two distributions are equal.", false],
    ],
    "The measured reward captures something related to audience reaction, but not the true objective of informativeness. Optimizing the proxy too hard can produce high reward and poor real-world behavior, which is the core reward-hacking failure mode.",
  ),
  makeQuestion(
    "cme295-lect5-q39",
    "hard",
    "During on-policy preference optimization, which interventions directly improve exploration over possible completions?",
    [
      [
        "Sample candidate completions with nonzero temperature instead of always using the same deterministic output.",
        true,
      ],
      [
        "Generate multiple candidate completions for a prompt so the reward signal can compare different behaviors.",
        true,
      ],
      [
        "Use greedy decoding with one completion per prompt because identical completions make preference learning clearer.",
        false,
      ],
      [
        "Randomize preference labels after generation so the policy does not overfit the reward model.",
        false,
      ],
    ],
    "Exploration requires the model to produce a useful variety of completions. Deterministic or near-identical outputs provide little information about alternatives, while random labels would destroy the preference signal rather than improve exploration.",
  ),
  makeQuestion(
    "cme295-lect5-q40",
    "hard",
    "Which end-to-end statements correctly place preference tuning inside the broader LLM training pipeline?",
    [
      [
        "Pretraining supplies broad next-token knowledge before task-specific behavior is shaped.",
        true,
      ],
      [
        "Supervised fine-tuning can turn the base model into an assistant or another specific behavior mode.",
        true,
      ],
      [
        "Preference tuning can further align outputs with preferred tone, safety, helpfulness, or other target dimensions.",
        true,
      ],
      [
        "Parameter-efficient methods such as LoRA can be compatible with preference tuning because they change which parameters are updated, not the high-level preference objective itself.",
        true,
      ],
    ],
    "The pipeline builds capability first and then shapes behavior. Preference tuning is a post-training stage that can use different optimization methods, and parameter-efficient adapters can be combined with those objectives when a team wants to update only a small set of trainable parameters.",
  ),
];
