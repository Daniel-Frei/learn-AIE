import { Question } from "../../quiz";

type Difficulty = "easy" | "medium" | "hard";
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
  difficulty: Difficulty,
  prompt: string,
  optionSeeds: readonly OptionSeed[],
  explanation: string,
): Question {
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
  difficulty: Difficulty,
  assertion: string,
  reason: string,
  correctChoice: AssertionReasonChoice,
  explanation: string,
): Question {
  return {
    id,
    chapter: 6,
    difficulty,
    type: "assertion-reason",
    prompt: "Assertion: " + assertion + "\n\nReason: " + reason,
    options: assertionReasonOptionTexts.map((text, index) => ({
      text,
      isCorrect: index + 1 === correctChoice,
    })),
    explanation,
  };
}

export const stanfordCME296Lecture6ModelTrainingQuestions: Question[] = [
  makeQuestion(
    "cme296-lect6-q01",
    "easy",
    "A pretrained image generator knows little about a newly important visual domain. Which post-training action most directly targets that knowledge gap?",
    [
      ["Continue training on a curated dataset from the new domain.", true],
      [
        "Include representative new-domain concepts in the continued-training mixture.",
        true,
      ],
      ["Reduce the sampler to one step before teaching the new domain.", false],
      ["Replace preference labels with random pairwise rankings.", false],
    ],
    "Continued training exposes the model to curated, representative domain content and therefore targets what it knows how to represent and generate. Sampler compression concerns speed, and random preference signals do not provide the missing visual knowledge; those interventions address different stages of the model lifecycle.",
  ),
  makeQuestion(
    "cme296-lect6-q02",
    "easy",
    "Which goals are most naturally associated with supervised fine-tuning rather than continued pretraining?",
    [
      [
        "Improve instruction following for prompts the model already broadly understands.",
        true,
      ],
      [
        "Shape output behavior toward preferred lighting or aesthetic conventions.",
        true,
      ],
      [
        "Acquire broad factual coverage of a previously unseen domain as the sole objective.",
        false,
      ],
      [
        "Cut the number of sampling function evaluations without changing behavior data.",
        false,
      ],
    ],
    "Supervised fine-tuning primarily shapes how an existing model behaves, including prompt adherence and visual style, while continued training is framed around adding or strengthening knowledge. Reducing function evaluations is a distillation objective, so it should not be confused with either behavioral supervision or domain-content acquisition.",
  ),
  makeQuestion(
    "cme296-lect6-q03",
    "medium",
    "A preference dataset contains several images generated for each prompt. Which steps can turn those judgments into reward-feedback learning?",
    [
      [
        "Collect pairwise or listwise rankings among images for the same prompt.",
        true,
      ],
      [
        "Fit a reward model that scores prompt-image compatibility or preference.",
        true,
      ],
      [
        "Backpropagate through a differentiable generation-and-reward path to increase predicted reward.",
        true,
      ],
      [
        "Treat the lowest-ranked image as the clean target for every prompt.",
        false,
      ],
    ],
    "Preference rankings can supervise a reward model, for example through pairwise Bradley-Terry comparisons, and a differentiable reward pathway can then tune the generator toward higher scores. The lowest-ranked image represents a rejected behavior rather than a universal reconstruction target, so training directly toward it would reverse the preference signal.",
  ),
  makeQuestion(
    "cme296-lect6-q04",
    "medium",
    "Which statements correctly distinguish a reward model from the image-generation policy it helps tune?",
    [
      [
        "The reward model maps a prompt-image pair to a scalar assessment.",
        true,
      ],
      [
        "The generation policy maps a prompt and sampling state toward an image.",
        true,
      ],
      [
        "Human rankings can supervise the reward model even when they do not provide pixel-level target images.",
        true,
      ],
      [
        "Optimizing the policy against the reward model can expose imperfections in what the reward model measures.",
        true,
      ],
    ],
    "The reward model is an evaluator learned from comparisons, while the policy is the system that produces candidates and is updated using that evaluator. Ranking data need not identify a unique target image, and policy optimization can exploit blind spots in the learned score, which is why reward quality and regularization matter.",
  ),
  makeAssertionReasonQuestion(
    "cme296-lect6-q05",
    "hard",
    "Maximizing a learned image reward can produce reward hacking even when the measured reward rises during training.",
    "A learned reward is an imperfect proxy that the generator may exploit outside the comparisons on which the reward model was trained.",
    4,
    "Both statements are true, and the reason supplies the mechanism. Optimization searches for outputs that score well under the proxy, including pathological regions where that score diverges from actual human preference; a rising proxy score therefore does not guarantee that true visual quality or alignment improves.",
  ),
  makeQuestion(
    "cme296-lect6-q06",
    "medium",
    "What quantity is central to the group-relative update in Flow-Group Reward Policy Optimization (Flow-GRPO)?",
    [
      [
        "An image's reward measured relative to the rewards of other samples generated for the same prompt.",
        true,
      ],
      [
        "The absolute pixel distance between every generated image and a single ground-truth image.",
        false,
      ],
      [
        "The number of model parameters relative to the reward-model parameter count.",
        false,
      ],
      [
        "The token-level likelihood ratio of a separate autoregressive captioner.",
        false,
      ],
    ],
    "Flow-GRPO generates a group of candidates and constructs an advantage from how each reward compares with its peers for the prompt. It does not require one pixel-perfect target, and model size or an unrelated captioner's likelihood is not the group-relative learning signal used to favor better samples.",
  ),
  makeQuestion(
    "cme296-lect6-q07",
    "hard",
    "Why can a Kullback-Leibler (KL) penalty to an older or reference policy help during reward-based tuning?",
    [
      [
        "It discourages the updated generator from moving too far into regions where the reward proxy may be unreliable.",
        true,
      ],
      [
        "It trades some reward improvement for behavioral stability relative to a known policy.",
        true,
      ],
      [
        "It proves that the learned reward exactly matches every human preference.",
        false,
      ],
      [
        "It makes all candidates in a reward group receive the same advantage.",
        false,
      ],
    ],
    "The KL term constrains policy drift, limiting how aggressively optimization can exploit a possibly misspecified reward and retaining useful behavior from the reference model. It cannot certify the reward model or erase within-group reward differences; instead, it creates an explicit optimization tradeoff between proxy reward and deviation.",
  ),
  makeQuestion(
    "cme296-lect6-q08",
    "hard",
    "Which properties distinguish diffusion Direct Preference Optimization (DPO) from first fitting a separate reward model and then running reinforcement learning?",
    [
      [
        "It can learn directly from preferred-versus-rejected image pairs.",
        true,
      ],
      [
        "It compares behavior against a reference model to control the preference update.",
        true,
      ],
      [
        "It avoids making a standalone scalar reward model the required intermediate artifact.",
        true,
      ],
      [
        "It removes the need for preference data because diffusion noise supplies the ranking labels.",
        false,
      ],
    ],
    "Diffusion-DPO adapts direct preference learning so pairwise comparisons update the generator relative to a reference without requiring a separately deployed reward model and policy-optimization loop. The preference labels are still essential supervision; sampled diffusion noise creates training states, not evidence that one completed image is preferred to another.",
  ),
  makeAssertionReasonQuestion(
    "cme296-lect6-q09",
    "medium",
    "Traditional knowledge distillation usually uses a smaller student, whereas diffusion sampling distillation may keep teacher and student architectures the same size.",
    "For diffusion generation, a principal target can be fewer sampling steps rather than fewer network parameters per evaluation.",
    4,
    "Both statements are true, and the reason explains the contrast. In ordinary compression the student often saves compute by shrinking the network, while diffusion distillation can instead preserve model capacity but train one student evaluation to imitate the effect of several teacher evaluations, reducing the number of function evaluations.",
  ),
  makeQuestion(
    "cme296-lect6-q10",
    "easy",
    "Which practical motivations favor distilling a many-step diffusion sampler?",
    [
      ["Interactive applications need lower response latency.", true],
      [
        "High-volume generation makes per-sample compute economically important.",
        true,
      ],
      ["Animation workloads may require many related frames or samples.", true],
      ["End users may have limited compute, time, and money.", true],
    ],
    "All four pressures make repeated denoiser evaluations costly in deployment, even when the original model has strong quality. Distillation deliberately spends training effort to move along the quality-speed frontier so the resulting sampler can serve interactive, high-throughput, or resource-constrained use cases more effectively.",
  ),
  makeQuestion(
    "cme296-lect6-q11",
    "easy",
    "What is the basic target used in one round of progressive diffusion distillation?",
    [
      [
        "Train one student transition to reproduce the endpoint reached by multiple smaller teacher transitions.",
        true,
      ],
      [
        "Train the student to add arbitrary noise until its output differs from the teacher.",
        false,
      ],
      [
        "Train a reward model to rank the teacher's intermediate latent states.",
        false,
      ],
      [
        "Delete half of the student's parameters while keeping the original sampler unchanged.",
        false,
      ],
    ],
    "A progressive round converts a short sequence of teacher steps, commonly two, into a single supervised student transition toward the same endpoint. This compresses temporal integration rather than deliberately diverging from the teacher, learning preferences, or pruning parameters while leaving the sampling trajectory intact.",
  ),
  makeQuestion(
    "cme296-lect6-q12",
    "medium",
    "A sampler starts with 1024 denoising steps and each progressive-distillation round halves the step count. Which calculations are correct?",
    [
      [
        "After five rounds, the target sampler uses \\(1024/2^5=32\\) steps.",
        true,
      ],
      [
        "Reaching one step requires \\(\\log_2 1024=10\\) halving rounds.",
        true,
      ],
      [
        "Five rounds produce a 5-step sampler because each round removes one step.",
        false,
      ],
      [
        "Ten rounds require the original teacher to execute only ten steps in total across all training data.",
        false,
      ],
    ],
    "Each round halves rather than subtracts from the current count, so the schedule follows \\(T/2^k\\): five rounds give 32 steps and ten give one. The logarithm counts sequential distillation stages, not the total teacher computation used to construct targets across minibatches, which can remain substantial.",
  ),
  makeAssertionReasonQuestion(
    "cme296-lect6-q13",
    "hard",
    "Directly asking a student to map terminal noise to a clean image in one initial distillation round can be much harder than progressive step compression.",
    "Progressive distillation presents a sequence of smaller imitation problems by repeatedly replacing two teacher transitions with one student transition.",
    4,
    "Both statements are true, and the reason identifies why the progressive curriculum is tractable. A one-shot target asks a single update to capture an entire curved, many-step trajectory immediately, whereas successive two-to-one compression keeps each new supervised mapping closer to behavior already represented by the current teacher.",
  ),
  makeQuestion(
    "cme296-lect6-q14",
    "hard",
    "Why can progressive distillation outperform simply skipping to the same number of deterministic DDIM evaluations?",
    [
      [
        "The student is trained specifically to approximate endpoints produced by the teacher's finer transitions.",
        true,
      ],
      [
        "Each distillation round adapts the learned transition to the coarser sampling grid.",
        true,
      ],
      [
        "The method transfers teacher behavior instead of applying an untrained large numerical jump.",
        true,
      ],
      [
        "The student exactly solves every possible reverse trajectory, so quality cannot decrease at one step.",
        false,
      ],
    ],
    "A distilled student learns how to make the larger transition that deployment will actually request, while naive step skipping asks the original model and solver to operate on a coarser grid without such adaptation. This can improve quality at equal numbers of function evaluations, but approximation and optimization errors remain, so exact one-step equivalence is not guaranteed.",
  ),
  makeQuestion(
    "cme296-lect6-q15",
    "medium",
    "Which limitations remain for the progressive-distillation procedure?",
    [
      [
        "It accepts the curvature of the teacher's path rather than simplifying that path first.",
        true,
      ],
      [
        "It is tied to discrete target transitions and a chosen sampling schedule.",
        true,
      ],
      [
        "Repeated halving requires about \\(\\log_2 T\\) sequential distillation rounds to compress \\(T\\) steps to one.",
        true,
      ],
      [
        "Its final quality still depends on student capacity, optimization, and accumulated approximation error.",
        true,
      ],
    ],
    "Progressive distillation compresses a given trajectory through fixed discrete transitions, so path curvature and schedule choices constrain the imitation problems it must solve. The logarithmic number of stages is still sequential, and student capacity, optimization, and accumulated approximation error determine how much quality survives aggressive compression.",
  ),
  makeAssertionReasonQuestion(
    "cme296-lect6-q16",
    "easy",
    "Using more sampling function evaluations commonly improves quality for a fixed distilled model family.",
    "Distillation eliminates the quality-speed tradeoff by making every supported step count reproduce the teacher exactly.",
    1,
    "The assertion is generally true because additional evaluations resolve the denoising trajectory more finely and give the model more opportunities to correct the sample. The reason is false: distillation improves the frontier and offers selectable operating points, but fewer evaluations can still sacrifice fidelity and no exact equality is guaranteed across all budgets.",
  ),
];
